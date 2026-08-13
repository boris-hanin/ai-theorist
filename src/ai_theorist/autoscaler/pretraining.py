from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .batch_scaling import BatchRunRecord, OptimizerHyperparameters
from .critical_batch import (
    ContinuationObservation,
    CriticalBatchEstimate,
    StepsToTargetObservation,
    combine_critical_batch_estimates,
    estimate_direct_checkpoint_critical_batch,
    estimate_gradient_noise_critical_batch,
    estimate_loss_optimal_batch,
    estimate_steps_to_target_critical_batch,
)
from .study import atomic_write_json
from .tokenization import (
    PINNED_TOKENIZER_REGISTRY,
    builtin_byte_tokenizer_manifest,
    load_token_stream_manifest,
    token_stream_identity,
)


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return result


def _strict_keys(payload: Mapping[str, Any], allowed: Sequence[str], context: str) -> None:
    extras = sorted(set(payload) - set(allowed))
    if extras:
        raise ValueError(f"Unknown {context} field(s): {', '.join(extras)}")


@dataclass(frozen=True)
class StandardTransformerSpec:
    vocab_size: int = 260
    context_length: int = 128
    width: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_multiplier: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StandardTransformerSpec":
        _strict_keys(
            payload,
            (
                "vocab_size",
                "context_length",
                "width",
                "depth",
                "num_heads",
                "mlp_multiplier",
                "dropout",
                "tie_embeddings",
            ),
            "standard Transformer",
        )
        result = cls(**dict(payload))
        _positive_int(result.vocab_size, "vocab_size", 8)
        _positive_int(result.context_length, "context_length", 2)
        _positive_int(result.width, "width", 4)
        _positive_int(result.depth, "depth")
        _positive_int(result.num_heads, "num_heads")
        _positive_int(result.mlp_multiplier, "mlp_multiplier")
        if result.width % result.num_heads:
            raise ValueError("width must be divisible by num_heads")
        dropout = _finite(result.dropout, "dropout", 0.0)
        if dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        if not isinstance(result.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be boolean")
        return result


@dataclass(frozen=True)
class TokenizedTextSpec:
    train_path: str = ""
    validation_path: str = ""
    tokenizer: str = "byte_v1"
    text_field: str = "text"
    maximum_bytes: int = 536_870_912
    token_stream_manifest_path: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenizedTextSpec":
        _strict_keys(
            payload,
            (
                "train_path",
                "validation_path",
                "tokenizer",
                "text_field",
                "maximum_bytes",
                "token_stream_manifest_path",
            ),
            "tokenized text dataset",
        )
        result = cls(**dict(payload))
        has_manifest = bool(
            result.token_stream_manifest_path
            and result.token_stream_manifest_path.strip()
        )
        has_paths = bool(result.train_path and result.validation_path)
        if has_manifest == has_paths:
            raise ValueError(
                "provide either token_stream_manifest_path or both train_path and "
                "validation_path"
            )
        if has_manifest:
            if result.tokenizer not in PINNED_TOKENIZER_REGISTRY:
                raise ValueError(
                    "token_stream_manifest_path requires an allow-listed pinned tokenizer"
                )
        elif result.tokenizer not in {"byte_v1", "uint16_bin_v1", "uint32_bin_v1"}:
            raise ValueError(
                "raw datasets require byte_v1, uint16_bin_v1, or uint32_bin_v1"
            )
        if not result.text_field:
            raise ValueError("text_field must be non-empty")
        _positive_int(result.maximum_bytes, "maximum_bytes", 1024)
        return result


@dataclass(frozen=True)
class PretrainingRuntimeSpec:
    precision: str = "fp32"
    attention_backend: str = "auto"
    distributed: str = "none"
    num_processes: int = 1
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = False
    checkpoint_interval_steps: int = 0
    checkpoint_interval_seconds: float = 0.0
    resume: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PretrainingRuntimeSpec":
        _strict_keys(
            payload,
            (
                "precision",
                "attention_backend",
                "distributed",
                "num_processes",
                "gradient_accumulation_steps",
                "activation_checkpointing",
                "checkpoint_interval_steps",
                "checkpoint_interval_seconds",
                "resume",
            ),
            "pretraining runtime",
        )
        result = cls(**dict(payload))
        if result.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if result.attention_backend not in {"auto", "math", "flash"}:
            raise ValueError("attention_backend must be auto, math, or flash")
        if result.distributed not in {"none", "ddp", "fsdp"}:
            raise ValueError("distributed must be none, ddp, or fsdp")
        _positive_int(result.num_processes, "num_processes")
        if result.distributed == "none" and result.num_processes != 1:
            raise ValueError("num_processes must be 1 when distributed is none")
        if result.distributed in {"ddp", "fsdp"} and result.num_processes < 2:
            raise ValueError("distributed training requires at least two processes")
        if result.attention_backend == "flash" and result.precision != "bf16":
            raise ValueError("the explicit FlashAttention path requires bf16")
        _positive_int(
            result.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        if (
            isinstance(result.checkpoint_interval_steps, bool)
            or not isinstance(result.checkpoint_interval_steps, int)
            or result.checkpoint_interval_steps < 0
        ):
            raise ValueError("checkpoint_interval_steps must be a non-negative integer")
        if (
            isinstance(result.checkpoint_interval_seconds, bool)
            or not isinstance(result.checkpoint_interval_seconds, (int, float))
            or not math.isfinite(float(result.checkpoint_interval_seconds))
            or result.checkpoint_interval_seconds < 0
        ):
            raise ValueError(
                "checkpoint_interval_seconds must be finite and non-negative"
            )
        if not isinstance(result.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        if not isinstance(result.resume, bool):
            raise ValueError("resume must be boolean")
        return result


def runtime_checkpoint_due(
    runtime: PretrainingRuntimeSpec,
    *,
    step: int,
    total_steps: int,
    last_checkpoint_at: float,
    now: Optional[float] = None,
) -> bool:
    """Return whether an enabled step- or wall-clock checkpoint is due."""

    enabled = bool(
        runtime.checkpoint_interval_steps
        or runtime.checkpoint_interval_seconds
    )
    if not enabled:
        return False
    if step == total_steps:
        return True
    if (
        runtime.checkpoint_interval_steps
        and step % runtime.checkpoint_interval_steps == 0
    ):
        return True
    observed_at = time.monotonic() if now is None else now
    return bool(
        runtime.checkpoint_interval_seconds
        and observed_at - last_checkpoint_at
        >= float(runtime.checkpoint_interval_seconds)
    )


def synchronized_runtime_checkpoint_due(
    runtime: PretrainingRuntimeSpec,
    context: "DistributedContext",
    *,
    step: int,
    total_steps: int,
    last_checkpoint_at: float,
    now: Optional[float] = None,
) -> bool:
    """Make rank zero's wall-clock checkpoint decision authoritative.

    Step-based decisions are naturally identical, but independent wall clocks
    can cross the checkpoint boundary on different updates. Every rank must
    enter checkpoint collectives together, so rank zero broadcasts one decision
    on every update.
    """

    due = runtime_checkpoint_due(
        runtime,
        step=step,
        total_steps=total_steps,
        last_checkpoint_at=last_checkpoint_at,
        now=now,
    )
    if context.world_size == 1:
        return due
    marker = torch.tensor(
        1 if due and context.is_primary else 0,
        dtype=torch.int32,
        device=context.device,
    )
    torch.distributed.broadcast(marker, src=0)
    return bool(marker.item())


class ByteTokenizer:
    """Deterministic UTF-8 tokenizer with four explicit special tokens."""

    vocab_size = 260
    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258
    separator_token_id = 259

    def encode(self, text: str, *, add_document_tokens: bool = True) -> List[int]:
        tokens = list(text.encode("utf-8"))
        if add_document_tokens:
            return [self.bos_token_id, *tokens, self.eos_token_id]
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        values = bytes(token for token in tokens if 0 <= token < 256)
        return values.decode("utf-8", errors="replace")


class _ShardedTokenStream:
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        context_length: int,
        maximum_bytes: int,
    ) -> None:
        manifest = load_token_stream_manifest(manifest_path, verify_files=True)
        metadata = manifest["splits"][split]
        total_bytes = sum(int(row["bytes"]) for row in metadata["shards"])
        if total_bytes > maximum_bytes:
            raise ValueError(
                f"{split} token shards exceed dataset.maximum_bytes {maximum_bytes}"
            )
        self.shards = tuple(
            np.memmap(
                manifest_path.parent / row["path"],
                mode="c",
                dtype="<u4",
            )
            for row in metadata["shards"]
        )
        self.lengths = tuple(int(array.size) for array in self.shards)
        self.valid_starts = tuple(max(0, length - context_length) for length in self.lengths)
        self.cumulative_valid_starts = np.cumsum(self.valid_starts, dtype=np.int64)
        self.total_valid_starts = int(self.cumulative_valid_starts[-1])
        self._numel = sum(self.lengths)
        self.maximum_token = max(
            int(row["maximum_token_id"]) for row in metadata["shards"]
        )
        if self.total_valid_starts <= 0:
            raise ValueError(f"{split} token shards contain no complete context window")

    def numel(self) -> int:
        return self._numel

    def sample_windows(
        self,
        batch_size: int,
        context_length: int,
        generator: torch.Generator,
    ) -> Tensor:
        global_starts = torch.randint(
            0,
            self.total_valid_starts,
            (batch_size,),
            generator=generator,
        ).numpy()
        shard_indices = np.searchsorted(
            self.cumulative_valid_starts, global_starts, side="right"
        )
        preceding = np.where(
            shard_indices > 0,
            self.cumulative_valid_starts[np.maximum(shard_indices - 1, 0)],
            0,
        )
        local_starts = global_starts - preceding
        offsets = np.arange(context_length + 1, dtype=np.int64)
        windows = np.empty((batch_size, context_length + 1), dtype=np.int64)
        # Grouping by shard turns one Python/memmap slice per example into one
        # vectorized read per represented shard while preserving the sampled
        # row order and exact RNG contract.
        for shard_index in np.unique(shard_indices):
            rows = np.flatnonzero(shard_indices == shard_index)
            indices = local_starts[rows, None] + offsets[None, :]
            windows[rows] = self.shards[int(shard_index)][indices]
        return torch.from_numpy(windows)


def _read_documents(path: Path, text_field: str, maximum_bytes: int) -> List[str]:
    if not path.is_file():
        raise ValueError(f"text dataset does not exist: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"text dataset size must be in [1, {maximum_bytes}] bytes: {path}")
    if path.suffix.lower() == ".jsonl":
        documents = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                    raise ValueError(
                        f"{path}:{line_number} must contain string field {text_field!r}"
                    )
                documents.append(row[text_field])
        if not documents:
            raise ValueError(f"JSONL dataset contains no documents: {path}")
        return documents
    return [path.read_text(encoding="utf-8")]


class TokenizedTextCorpus:
    def __init__(
        self, spec: TokenizedTextSpec, context_length: int, vocab_size: int = 260
    ) -> None:
        spec = TokenizedTextSpec.from_dict(asdict(spec))
        self.spec = spec
        self.context_length = context_length
        self.tokenizer = ByteTokenizer()
        self.tokenizer_is_pinned = False
        self.tokenizer_fingerprint: Optional[str] = None
        self.tokenizer_manifest: Optional[Dict[str, Any]] = None
        if spec.token_stream_manifest_path:
            manifest_path = Path(spec.token_stream_manifest_path)
            manifest = load_token_stream_manifest(manifest_path, verify_files=True)
            if manifest["tokenizer_id"] != spec.tokenizer:
                raise ValueError(
                    "dataset tokenizer does not match the token stream manifest"
                )
            if int(manifest["vocab_size"]) != vocab_size:
                raise ValueError(
                    f"token stream requires vocab_size {manifest['vocab_size']}"
                )
            tokenizer_manifest_path = (
                manifest_path.parent / manifest["tokenizer_manifest_path"]
            ).resolve()
            with tokenizer_manifest_path.open("r", encoding="utf-8") as handle:
                self.tokenizer_manifest = json.load(handle)
            self.tokenizer_is_pinned = True
            self.tokenizer_fingerprint = str(manifest["tokenizer_fingerprint"])
            self.train_tokens = _ShardedTokenStream(
                manifest_path, "train", context_length, spec.maximum_bytes
            )
            self.validation_tokens = _ShardedTokenStream(
                manifest_path, "validation", context_length, spec.maximum_bytes
            )
            self._fingerprint = str(manifest["content_fingerprint"])
            self._identity_fingerprint = str(manifest["fingerprint"])
        elif spec.tokenizer in {"uint16_bin_v1", "uint32_bin_v1"}:
            dtype = np.uint16 if spec.tokenizer == "uint16_bin_v1" else np.dtype("<u4")
            self.train_tokens = self._map_binary(Path(spec.train_path), dtype)
            self.validation_tokens = self._map_binary(Path(spec.validation_path), dtype)
            self._fingerprint = sha256(
                (
                    self._hash_file(Path(spec.train_path))
                    + self._hash_file(Path(spec.validation_path))
                ).encode("ascii")
            ).hexdigest()
            self._identity_fingerprint = sha256(
                json.dumps(
                    {
                        "format": spec.tokenizer,
                        "content_fingerprint": self._fingerprint,
                        "vocab_size": vocab_size,
                        "tokenizer_pinned": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        else:
            train_documents = _read_documents(
                Path(spec.train_path), spec.text_field, spec.maximum_bytes
            )
            validation_documents = _read_documents(
                Path(spec.validation_path), spec.text_field, spec.maximum_bytes
            )
            self.train_tokens = self._tokenize(train_documents)
            self.validation_tokens = self._tokenize(validation_documents)
            digest = sha256()
            digest.update(self.train_tokens.numpy().tobytes())
            digest.update(self.validation_tokens.numpy().tobytes())
            self._fingerprint = digest.hexdigest()
            self.tokenizer_manifest = builtin_byte_tokenizer_manifest()
            self.tokenizer_fingerprint = str(self.tokenizer_manifest["fingerprint"])
            self.tokenizer_is_pinned = True
            self._identity_fingerprint = sha256(
                json.dumps(
                    {
                        "format": "byte_v1",
                        "content_fingerprint": self._fingerprint,
                        "tokenizer_fingerprint": self.tokenizer_fingerprint,
                        "packing": "bos_document_eos_separator_v1",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        for split, tokens in (
            ("training", self.train_tokens),
            ("validation", self.validation_tokens),
        ):
            if tokens.numel() < context_length + 2:
                raise ValueError(f"{split} token stream is shorter than one context window")
            # NumPy supports uint16 reductions directly while PyTorch's CPU
            # uint16 reduction coverage varies by release. This remains a
            # zero-copy scan for memory-mapped token streams.
            maximum_token = (
                tokens.maximum_token
                if isinstance(tokens, _ShardedTokenStream)
                else int(tokens.numpy().max())
            )
            if maximum_token >= vocab_size:
                raise ValueError(
                    f"{split} token id {maximum_token} is outside vocab_size {vocab_size}"
                )

    def _tokenize(self, documents: Sequence[str]) -> Tensor:
        tokens: List[int] = []
        for index, document in enumerate(documents):
            if index:
                tokens.append(self.tokenizer.separator_token_id)
            tokens.extend(self.tokenizer.encode(document))
        return torch.tensor(tokens, dtype=torch.long)

    def _map_binary(self, path: Path, dtype: Any) -> Tensor:
        if not path.is_file():
            raise ValueError(f"binary token stream does not exist: {path}")
        size = path.stat().st_size
        item_size = np.dtype(dtype).itemsize
        if size <= 0 or size > self.spec.maximum_bytes or size % item_size:
            raise ValueError(
                "binary token streams must have a positive aligned byte size "
                "within maximum_bytes"
            )
        array = np.memmap(path, mode="c", dtype=dtype)
        return torch.from_numpy(array)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def identity_fingerprint(self) -> str:
        return self._identity_fingerprint

    def sample_batch(
        self,
        split: str,
        batch_size: int,
        generator: torch.Generator,
        device: str,
    ) -> Tuple[Tensor, Tensor]:
        tokens = self.train_tokens if split == "train" else self.validation_tokens
        if isinstance(tokens, _ShardedTokenStream):
            windows = tokens.sample_windows(batch_size, self.context_length, generator)
            windows = windows.to(device, non_blocking=device.startswith("cuda"))
            return windows[:, :-1], windows[:, 1:]
        maximum_start = tokens.numel() - self.context_length - 1
        starts = torch.randint(0, maximum_start + 1, (batch_size,), generator=generator)
        offsets = torch.arange(self.context_length + 1)
        indices = starts[:, None] + offsets[None, :]
        if tokens.dtype in {torch.uint16, torch.uint32}:
            # PyTorch does not implement advanced CPU indexing for uint16 on
            # every supported release. Index the memmap through NumPy and copy
            # only the sampled windows into the model's int64 token dtype.
            sampled = tokens.numpy()[indices.numpy()].astype(np.int64, copy=False)
            windows = torch.from_numpy(sampled)
        else:
            windows = tokens[indices].long()
        windows = windows.to(device, non_blocking=device.startswith("cuda"))
        return windows[:, :-1], windows[:, 1:]


class StandardCausalSelfAttention(nn.Module):
    def __init__(self, spec: StandardTransformerSpec, attention_backend: str) -> None:
        super().__init__()
        self.width = spec.width
        self.num_heads = spec.num_heads
        self.head_dimension = spec.width // spec.num_heads
        self.attention_backend = attention_backend
        self.query_key_value = nn.Linear(spec.width, 3 * spec.width, bias=True)
        self.output = nn.Linear(spec.width, spec.width, bias=True)
        self.dropout = spec.dropout

    def _kernel_context(self):
        if self.attention_backend == "auto":
            return nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = (
            SDPBackend.FLASH_ATTENTION
            if self.attention_backend == "flash"
            else SDPBackend.MATH
        )
        return sdpa_kernel([backend])

    def forward(self, hidden: Tensor) -> Tensor:
        batch, time_steps, _ = hidden.shape
        qkv = self.query_key_value(hidden)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch, time_steps, self.num_heads, self.head_dimension
            ).transpose(1, 2)

        with self._kernel_context():
            attended = F.scaled_dot_product_attention(
                split_heads(query),
                split_heads(key),
                split_heads(value),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, time_steps, self.width)
        return self.output(attended)


class StandardTransformerBlock(nn.Module):
    def __init__(self, spec: StandardTransformerSpec, attention_backend: str) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(spec.width)
        self.attention = StandardCausalSelfAttention(spec, attention_backend)
        self.mlp_norm = nn.LayerNorm(spec.width)
        hidden_width = spec.mlp_multiplier * spec.width
        self.mlp = nn.Sequential(
            nn.Linear(spec.width, hidden_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_width, spec.width),
            nn.Dropout(spec.dropout),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class StandardTransformer(nn.Module):
    """Small GPT-style pre-norm decoder using PyTorch SDPA kernels."""

    def __init__(
        self,
        spec: StandardTransformerSpec,
        *,
        attention_backend: str = "auto",
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.activation_checkpointing = activation_checkpointing
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.position_embedding = nn.Embedding(spec.context_length, spec.width)
        self.dropout = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList(
            StandardTransformerBlock(spec, attention_backend) for _ in range(spec.depth)
        )
        self.final_norm = nn.LayerNorm(spec.width)
        self.language_model_head = nn.Linear(spec.width, spec.vocab_size, bias=False)
        self.apply(self._initialize)
        if spec.tie_embeddings:
            self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.spec.context_length:
            raise ValueError("tokens must have shape [batch, time <= context_length]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = activation_checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.language_model_head(self.final_norm(hidden))


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: str
    initialized_here: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def prepare_distributed(runtime: PretrainingRuntimeSpec, device: str) -> DistributedContext:
    if runtime.distributed == "none":
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return DistributedContext(0, 1, 0, device)
    if not torch.cuda.is_available():
        raise RuntimeError("distributed pretraining requires CUDA")
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if min(rank, world_size, local_rank) < 0:
        raise RuntimeError("distributed pretraining must be launched with torchrun")
    if world_size != runtime.num_processes:
        raise RuntimeError(
            f"torchrun world size {world_size} does not match configured {runtime.num_processes}"
        )
    torch.cuda.set_device(local_rank)
    initialized_here = False
    if not torch.distributed.is_initialized():
        # Rank zero runs the direct-checkpoint and gradient-noise assays while
        # its peers wait. A long timeout keeps legitimate large-model assays
        # from being mistaken for a dead worker group.
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(hours=12)
        )
        initialized_here = True
    return DistributedContext(
        rank, world_size, local_rank, f"cuda:{local_rank}", initialized_here
    )


def close_distributed(context: DistributedContext) -> None:
    if context.initialized_here and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def preflight_runtime(
    model_spec: StandardTransformerSpec,
    runtime: PretrainingRuntimeSpec,
    device: str,
) -> Dict[str, Any]:
    if runtime.attention_backend == "flash":
        if not device.startswith("cuda"):
            raise ValueError("FlashAttention requires a CUDA device")
        if model_spec.width // model_spec.num_heads % 8:
            raise ValueError("FlashAttention requires a head dimension divisible by 8")
    if runtime.precision == "bf16" and device.startswith("cuda"):
        cuda_device = torch.device(device)
        cuda_index = (
            cuda_device.index
            if cuda_device.index is not None
            else torch.cuda.current_device()
        )
        major, _ = torch.cuda.get_device_capability(cuda_index)
        if major < 8:
            raise ValueError("CUDA bf16 requires compute capability 8.0 or newer")
    return {
        "precision": runtime.precision,
        "attention_backend": runtime.attention_backend,
        "distributed": runtime.distributed,
        "num_processes": runtime.num_processes,
        "device": device,
        "uses_torch_sdpa": True,
        "flash_attention_requested": runtime.attention_backend == "flash",
        "gradient_accumulation_steps": runtime.gradient_accumulation_steps,
        "activation_checkpointing": runtime.activation_checkpointing,
        "checkpoint_interval_steps": runtime.checkpoint_interval_steps,
        "checkpoint_interval_seconds": runtime.checkpoint_interval_seconds,
        "mid_trial_resume": runtime.resume
        and bool(
            runtime.checkpoint_interval_steps
            or runtime.checkpoint_interval_seconds
        ),
    }


def wrap_distributed_model(
    model: nn.Module,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
    block_types: Sequence[type[nn.Module]],
) -> nn.Module:
    if runtime.distributed == "none":
        return model
    if runtime.distributed == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
        )
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    mixed_precision = (
        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        if runtime.precision == "bf16"
        else None
    )
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=ModuleWrapPolicy(set(block_types)),
        device_id=torch.device(context.device),
        mixed_precision=mixed_precision,
        use_orig_params=True,
        sync_module_states=True,
    )


def _wrap_fsdp(
    model: StandardTransformer,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
) -> nn.Module:
    return wrap_distributed_model(
        model, runtime, context, (StandardTransformerBlock,)
    )


def _autocast(runtime: PretrainingRuntimeSpec, device: str):
    if runtime.precision != "bf16":
        return nullcontext()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def _optimizer(model: nn.Module, spec: OptimizerHyperparameters) -> torch.optim.Optimizer:
    if spec.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=spec.learning_rate,
            momentum=spec.momentum,
            weight_decay=spec.weight_decay,
        )
    optimizer_class = torch.optim.AdamW if spec.name == "adamw" else torch.optim.Adam
    return optimizer_class(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
    )


def _distributed_mean(value: Tensor, context: DistributedContext) -> Tensor:
    if context.world_size == 1:
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result / context.world_size


def _evaluate(
    model: nn.Module,
    corpus: TokenizedTextCorpus,
    model_spec: StandardTransformerSpec,
    runtime: PretrainingRuntimeSpec,
    context: DistributedContext,
    validation_examples: int,
    seed: int,
) -> float:
    local_examples = max(1, math.ceil(validation_examples / context.world_size))
    generator = torch.Generator(device="cpu").manual_seed(
        9_000_017 + seed + 10_000 * context.rank
    )
    inputs, targets = corpus.sample_batch(
        "validation", local_examples, generator, context.device
    )
    model.eval()
    with torch.no_grad(), _autocast(runtime, context.device):
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, model_spec.vocab_size), targets.reshape(-1)
        )
    mean_loss = _distributed_mean(loss, context)
    model.train()
    return float(mean_loss.cpu())


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def runtime_checkpoint_path(base_path: Path, context: DistributedContext) -> Path:
    if context.world_size == 1:
        return base_path.with_suffix(".pt")
    return base_path.with_name(
        f"{base_path.name}.rank-{context.rank:05d}-of-{context.world_size:05d}.pt"
    )


def save_runtime_checkpoint(
    *,
    base_path: Path,
    model: nn.Module,
    plain_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context: DistributedContext,
    runtime: PretrainingRuntimeSpec,
    identity_fingerprint: str,
    step: int,
    generator: torch.Generator,
    extra: Mapping[str, Any],
) -> None:
    """Atomically persist a same-topology mid-trial restart point.

    Non-distributed runs store one ordinary state dictionary. FSDP runs store
    one sharded model/optimizer file per local rank, avoiding rank-zero model
    consolidation and its memory spike. Resumption therefore deliberately
    requires the same world size.
    """
    if runtime.distributed == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            model_state = model.state_dict()
            optimizer_state = FullyShardedDataParallel.optim_state_dict(
                model, optimizer
            )
    else:
        model_state = {
            name: value.detach().cpu()
            for name, value in plain_model.state_dict().items()
        }
        optimizer_state = optimizer.state_dict()
    _atomic_torch_save(
        {
            "schema_version": 1,
            "identity_fingerprint": identity_fingerprint,
            "world_size": context.world_size,
            "rank": context.rank,
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "generator_state": generator.get_state().cpu(),
            "extra": dict(extra),
        },
        runtime_checkpoint_path(base_path, context),
    )
    if context.world_size > 1:
        torch.distributed.barrier()


def load_runtime_checkpoint(
    *,
    base_path: Path,
    model: nn.Module,
    plain_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context: DistributedContext,
    runtime: PretrainingRuntimeSpec,
    identity_fingerprint: str,
    generator: torch.Generator,
) -> Optional[Dict[str, Any]]:
    path = runtime_checkpoint_path(base_path, context)
    local_exists = path.is_file()
    if context.world_size > 1:
        marker = torch.tensor(
            1 if local_exists else 0,
            dtype=torch.int32,
            device=context.device,
        )
        torch.distributed.all_reduce(marker, op=torch.distributed.ReduceOp.MIN)
        local_exists = bool(marker.item())
    if not runtime.resume or not local_exists:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("identity_fingerprint") != identity_fingerprint
        or checkpoint.get("world_size") != context.world_size
        or checkpoint.get("rank") != context.rank
    ):
        raise ValueError("runtime checkpoint identity or topology mismatch")
    if context.world_size > 1:
        local_step = torch.tensor(
            int(checkpoint["step"]), dtype=torch.int64, device=context.device
        )
        minimum_step = local_step.clone()
        maximum_step = local_step.clone()
        torch.distributed.all_reduce(
            minimum_step, op=torch.distributed.ReduceOp.MIN
        )
        torch.distributed.all_reduce(
            maximum_step, op=torch.distributed.ReduceOp.MAX
        )
        if int(minimum_step.item()) != int(maximum_step.item()):
            raise ValueError("distributed runtime checkpoint steps are inconsistent")
    if runtime.distributed == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            ShardedOptimStateDictConfig,
            ShardedStateDictConfig,
            StateDictType,
        )

        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = FullyShardedDataParallel.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer_state_dict"]
            )
        optimizer.load_state_dict(optimizer_state)
    else:
        plain_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    generator.set_state(checkpoint["generator_state"])
    return {
        "step": int(checkpoint["step"]),
        "extra": dict(checkpoint.get("extra", {})),
    }


def clear_runtime_checkpoint(
    base_path: Path, context: DistributedContext
) -> None:
    runtime_checkpoint_path(base_path, context).unlink(missing_ok=True)
    if context.world_size > 1:
        torch.distributed.barrier()


def run_standard_pretraining_trial(
    *,
    model_spec: StandardTransformerSpec,
    corpus: TokenizedTextCorpus,
    runtime: PretrainingRuntimeSpec,
    distributed_context: DistributedContext,
    optimizer_spec: OptimizerHyperparameters,
    total_tokens: int,
    batch_examples: int,
    seed: int,
    target_validation_loss: Optional[float] = None,
    validation_interval: int = 1,
    validation_examples: int = 32,
    warmup_steps: int = 0,
    minimum_learning_rate_ratio: float = 0.1,
    cache_directory: Optional[Path] = None,
    cache_suffix: str = "",
    initial_state: Optional[Mapping[str, Tensor]] = None,
    initial_optimizer_state: Optional[Mapping[str, Any]] = None,
    return_state: bool = False,
) -> Tuple[BatchRunRecord, Dict[str, Any]]:
    context = distributed_context
    runtime_diagnostics = preflight_runtime(model_spec, runtime, context.device)
    batch_examples = _positive_int(batch_examples, "batch_examples")
    total_tokens = _positive_int(total_tokens, "total_tokens")
    validation_interval = _positive_int(validation_interval, "validation_interval")
    validation_examples = _positive_int(validation_examples, "validation_examples")
    batch_tokens = batch_examples * model_spec.context_length
    if total_tokens % batch_tokens:
        raise ValueError("total_tokens must be divisible by global batch tokens")
    data_parallel_microbatches = (
        context.world_size * runtime.gradient_accumulation_steps
    )
    if batch_examples % data_parallel_microbatches:
        raise ValueError(
            "global batch examples must be divisible by world size times "
            "gradient_accumulation_steps"
        )
    if (initial_state is None) != (initial_optimizer_state is None):
        raise ValueError("model and optimizer checkpoint states must be supplied together")
    if initial_state is not None and runtime.distributed != "none":
        raise ValueError("direct state injection is supported only without FSDP")
    steps = total_tokens // batch_tokens
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be non-negative and smaller than total steps")
    if not 0.0 <= minimum_learning_rate_ratio <= 1.0:
        raise ValueError("minimum_learning_rate_ratio must be in [0, 1]")

    identity = {
        "model": asdict(model_spec),
        "dataset_fingerprint": corpus.identity_fingerprint,
        "runtime": asdict(runtime),
        "optimizer": optimizer_spec.to_dict(),
        "total_tokens": total_tokens,
        "batch_examples": batch_examples,
        "seed": seed,
        "target_validation_loss": target_validation_loss,
        "validation_interval": validation_interval,
        "validation_examples": validation_examples,
        "warmup_steps": warmup_steps,
        "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
        "checkpoint_state_schema_version": 2,
        "suffix": cache_suffix,
    }
    identity_fingerprint = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    digest = identity_fingerprint[:12]
    run_id = (
        f"text-{model_spec.width}x{model_spec.depth}-{optimizer_spec.name}"
        f"-b{batch_tokens}-t{total_tokens}-s{seed}-{digest}{cache_suffix}"
    )
    record_path = cache_directory / f"{run_id}.json" if cache_directory else None
    state_path = cache_directory / f"{run_id}.pt" if cache_directory else None
    cache_hit = False
    if context.is_primary:
        cache_hit = bool(
            record_path is not None
            and record_path.exists()
            and (not return_state or (state_path is not None and state_path.exists()))
        )
    if context.world_size > 1:
        marker = [cache_hit]
        torch.distributed.broadcast_object_list(marker, src=0)
        cache_hit = bool(marker[0])
    if cache_hit:
        assert record_path is not None
        with record_path.open("r", encoding="utf-8") as handle:
            record = BatchRunRecord.from_dict(json.load(handle))
        state = {}
        optimizer_state = {}
        if return_state:
            if runtime.distributed != "none":
                raise ValueError("cached full states are not returned from FSDP trials")
            assert state_path is not None
            checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
            state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer_state_dict"]
        return record, {
            "state_dict": state,
            "optimizer_state_dict": optimizer_state,
            "initial_validation_loss": record.validation_checkpoints[0]["validation_loss"],
            "cache_hit": True,
        }

    torch.manual_seed(seed)
    if context.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(context.device)
    plain_model = StandardTransformer(
        model_spec,
        attention_backend=runtime.attention_backend,
        activation_checkpointing=runtime.activation_checkpointing,
    ).to(context.device)
    if initial_state is not None:
        plain_model.load_state_dict(initial_state)
    parameter_count = sum(parameter.numel() for parameter in plain_model.parameters())
    model = _wrap_fsdp(plain_model, runtime, context)
    optimizer = _optimizer(model, optimizer_spec)
    if initial_optimizer_state is not None:
        optimizer.load_state_dict(dict(initial_optimizer_state))
    local_batch_examples = batch_examples // data_parallel_microbatches
    generator = torch.Generator(device="cpu").manual_seed(
        100_003 + seed + 1_000_003 * context.rank
    )
    checkpoints: List[Dict[str, float]] = []
    crossing_step = None
    resume_path = (
        cache_directory / f"{run_id}.resume"
        if cache_directory is not None
        else None
    )
    resumed = None
    if resume_path is not None:
        resumed = load_runtime_checkpoint(
            base_path=resume_path,
            model=model,
            plain_model=plain_model,
            optimizer=optimizer,
            context=context,
            runtime=runtime,
            identity_fingerprint=identity_fingerprint,
            generator=generator,
        )
    start_step = int(resumed["step"]) if resumed is not None else 0
    elapsed_before_resume = 0.0
    if resumed is not None:
        resume_extra = resumed["extra"]
        checkpoints = [dict(row) for row in resume_extra["validation_checkpoints"]]
        crossing_step = resume_extra.get("crossing_step")
        elapsed_before_resume = float(resume_extra.get("elapsed_seconds", 0.0))
        if not checkpoints or int(checkpoints[0]["step"]) != 0:
            raise ValueError("runtime checkpoint is missing its initial validation")
        initial_loss = float(checkpoints[0]["validation_loss"])
    else:
        initial_loss = _evaluate(
            model,
            corpus,
            model_spec,
            runtime,
            context,
            validation_examples,
            seed,
        )
        checkpoints.append(
            {"step": 0.0, "tokens": 0.0, "validation_loss": initial_loss}
        )
    started = time.monotonic()
    last_checkpoint_at = started
    peak_learning_rate = optimizer_spec.learning_rate
    for step in range(start_step + 1, steps + 1):
        if step <= warmup_steps and warmup_steps:
            multiplier = step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            multiplier = minimum_learning_rate_ratio + (
                1.0 - minimum_learning_rate_ratio
            ) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = peak_learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        for accumulation_index in range(runtime.gradient_accumulation_steps):
            inputs, targets = corpus.sample_batch(
                "train", local_batch_examples, generator, context.device
            )
            synchronization = (
                model.no_sync()  # type: ignore[attr-defined]
                if context.world_size > 1
                and accumulation_index + 1 < runtime.gradient_accumulation_steps
                else nullcontext()
            )
            with synchronization, _autocast(runtime, context.device):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, model_spec.vocab_size),
                    targets.reshape(-1),
                ) / runtime.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError("standard Transformer trial diverged")
            loss.backward()
        if runtime.distributed == "fsdp":
            model.clip_grad_norm_(1.0)  # type: ignore[attr-defined]
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % validation_interval == 0 or step == steps:
            validation_loss = _evaluate(
                model,
                corpus,
                model_spec,
                runtime,
                context,
                validation_examples,
                seed,
            )
            checkpoints.append(
                {
                    "step": float(step),
                    "tokens": float(step * batch_tokens),
                    "validation_loss": validation_loss,
                }
            )
            if (
                crossing_step is None
                and target_validation_loss is not None
                and validation_loss <= target_validation_loss
            ):
                crossing_step = step
        checkpoint_now = time.monotonic()
        if resume_path is not None and synchronized_runtime_checkpoint_due(
            runtime,
            context,
            step=step,
            total_steps=steps,
            last_checkpoint_at=last_checkpoint_at,
            now=checkpoint_now,
        ):
            save_runtime_checkpoint(
                base_path=resume_path,
                model=model,
                plain_model=plain_model,
                optimizer=optimizer,
                context=context,
                runtime=runtime,
                identity_fingerprint=identity_fingerprint,
                step=step,
                generator=generator,
                extra={
                    "validation_checkpoints": checkpoints,
                    "crossing_step": crossing_step,
                    "elapsed_seconds": elapsed_before_resume
                    + time.monotonic()
                    - started,
                },
            )
            last_checkpoint_at = checkpoint_now
    duration = elapsed_before_resume + time.monotonic() - started
    final_loss = checkpoints[-1]["validation_loss"]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(context.device))
        if context.device.startswith("cuda")
        else 0
    )
    record = BatchRunRecord(
        run_id=run_id,
        model_family="standard_pre_norm_transformer_tokenized_text",
        optimizer=optimizer_spec,
        seed=seed,
        parameter_count=parameter_count,
        width=model_spec.width,
        depth=model_spec.depth,
        total_tokens=total_tokens,
        batch_tokens=batch_tokens,
        microbatch_tokens=local_batch_examples * model_spec.context_length,
        accumulation_steps=runtime.gradient_accumulation_steps,
        data_parallel_replicas=context.world_size,
        optimizer_steps=steps,
        nonpadding_tokens_seen=total_tokens,
        learning_rate_schedule="linear_warmup_then_cosine",
        final_validation_loss=float(final_loss),
        estimated_flops=float(6 * parameter_count * total_tokens),
        wall_time_seconds=duration,
        target_loss_crossings=(
            {f"{target_validation_loss:g}": crossing_step}
            if target_validation_loss is not None
            else {}
        ),
        validation_checkpoints=tuple(checkpoints),
        metadata={
            **runtime_diagnostics,
            "tokenizer": corpus.spec.tokenizer,
            "dataset_fingerprint": corpus.identity_fingerprint,
            "corpus_content_fingerprint": corpus.fingerprint,
            "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
            "tokenizer_is_pinned": corpus.tokenizer_is_pinned,
            "peak_memory_bytes": peak_memory,
            "global_batch_examples": batch_examples,
            "local_microbatch_examples": local_batch_examples,
            "activation_checkpointing": runtime.activation_checkpointing,
            "resumed_from_step": start_step,
        },
    )
    state: Mapping[str, Tensor] = {}
    optimizer_state: Mapping[str, Any] = {}
    if return_state:
        if runtime.distributed != "none":
            raise ValueError("return_state is supported only for non-FSDP trials")
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in plain_model.state_dict().items()
        }
        optimizer_state = optimizer.state_dict()
    if context.is_primary and record_path is not None:
        atomic_write_json(record_path, record.to_dict())
        if return_state and state_path is not None:
            _atomic_torch_save(
                {
                    "state_dict": state,
                    "optimizer_state_dict": optimizer_state,
                },
                state_path,
            )
    if context.world_size > 1:
        torch.distributed.barrier()
    if resume_path is not None:
        clear_runtime_checkpoint(resume_path, context)
    return record, {
        "state_dict": state,
        "optimizer_state_dict": optimizer_state,
        "initial_validation_loss": initial_loss,
        "cache_hit": False,
    }


def _optimizer_from_payload(
    payload: Mapping[str, Any], learning_rate: float
) -> OptimizerHyperparameters:
    return OptimizerHyperparameters(
        name=str(payload["name"]),
        learning_rate=learning_rate,
        momentum=float(payload.get("momentum", 0.0)),
        beta1=float(payload.get("beta1", 0.9)),
        beta2=float(payload.get("beta2", 0.999)),
        epsilon=float(payload.get("epsilon", 1e-8)),
        weight_decay=float(payload.get("weight_decay", 0.0)),
    )


def _float_grid(value: Any, name: str) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    result = tuple(float(item) for item in value)
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"{name} must contain positive finite numbers")
    return result


def compile_standard_pretraining_plan(config: Mapping[str, Any]) -> Dict[str, Any]:
    model = StandardTransformerSpec.from_dict(config["model"])
    dataset = TokenizedTextSpec.from_dict(config["dataset"])
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    if dataset.tokenizer == "byte_v1" and model.vocab_size != ByteTokenizer.vocab_size:
        raise ValueError(
            f"byte_v1 requires vocab_size {ByteTokenizer.vocab_size}"
        )
    stream_identity = (
        token_stream_identity(Path(dataset.token_stream_manifest_path))
        if dataset.token_stream_manifest_path
        else None
    )
    if stream_identity is not None and stream_identity["vocab_size"] != model.vocab_size:
        raise ValueError(
            f"token stream requires vocab_size {stream_identity['vocab_size']}"
        )
    if stream_identity is not None and stream_identity["tokenizer_id"] != dataset.tokenizer:
        raise ValueError("dataset tokenizer does not match the token stream manifest")
    scales = []
    for index, payload in enumerate(config["scales"]):
        if not isinstance(payload, Mapping):
            raise ValueError(f"scales[{index}] must be an object")
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError(f"scales[{index}].name is required")
        width = _positive_int(payload.get("width"), f"scales[{index}].width", 4)
        depth = _positive_int(payload.get("depth"), f"scales[{index}].depth")
        heads = _positive_int(payload.get("num_heads"), f"scales[{index}].num_heads")
        if width % heads:
            raise ValueError(f"scales[{index}].width must be divisible by num_heads")
        scales.append({"name": name, "width": width, "depth": depth, "num_heads": heads})
    if not scales:
        raise ValueError("at least one scale is required")
    batches = tuple(
        _positive_int(int(value), "batch_examples") for value in config["batch_examples"]
    )
    if len(set(batches)) < 4:
        raise ValueError("critical-batch census requires at least four batch sizes")
    total_tokens = _positive_int(config["total_tokens"], "total_tokens")
    for batch in batches:
        if total_tokens % (batch * model.context_length):
            raise ValueError("total_tokens must divide every global batch token count")
        if batch % runtime.num_processes:
            raise ValueError("every global batch must be divisible by num_processes")
    optimizers = config["optimizers"]
    if not isinstance(optimizers, Sequence) or not optimizers:
        raise ValueError("optimizers must be a non-empty list")
    trial_count = 0
    for optimizer in optimizers:
        _float_grid(optimizer["learning_rates"], "learning_rates")
        _optimizer_from_payload(optimizer, float(optimizer["learning_rates"][0]))
        trial_count += len(scales) * len(batches) * len(optimizer["learning_rates"]) * len(
            config.get("seeds", [11, 29])
        )
    return {
        "schema_version": 1,
        "campaign": "standard_text_pretraining_batch_census",
        "model": asdict(model),
        "dataset": asdict(dataset),
        "dataset_identity": stream_identity,
        "runtime": asdict(runtime),
        "scales": scales,
        "batch_examples": list(batches),
        "total_tokens": total_tokens,
        "planned_grid_trials": trial_count,
        "capabilities": {
            "real_tokenized_text": True,
            "bf16": True,
            "torch_sdpa": True,
            "explicit_flash_attention": True,
            "single_node_ddp": True,
            "single_node_fsdp": True,
            "gradient_accumulation": True,
            "activation_checkpointing": True,
            "mid_trial_resume": True,
        },
    }


def _measure_gradient_noise(
    model_spec: StandardTransformerSpec,
    corpus: TokenizedTextCorpus,
    runtime: PretrainingRuntimeSpec,
    optimizer_seed: int,
    microbatch_examples: int,
    sample_count: int,
    device: str,
) -> CriticalBatchEstimate:
    if sample_count < 8:
        raise ValueError("gradient_noise_samples must be at least 8")
    torch.manual_seed(optimizer_seed)
    model = StandardTransformer(
        model_spec, attention_backend=runtime.attention_backend
    ).to(device)
    generator = torch.Generator(device="cpu").manual_seed(optimizer_seed + 71_003)
    samples = []
    for _ in range(sample_count):
        inputs, targets = corpus.sample_batch(
            "train", microbatch_examples, generator, device
        )
        model.zero_grad(set_to_none=True)
        with _autocast(runtime, device):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, model_spec.vocab_size), targets.reshape(-1)
            )
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        samples.append(
            torch.cat(
                [gradient.detach().float().reshape(-1).cpu() for gradient in gradients]
            ).numpy()
        )
    return estimate_gradient_noise_critical_batch(
        np.stack(samples),
        microbatch_tokens=microbatch_examples * model_spec.context_length,
    )


def _emit(
    callback: ProgressCallback,
    phase: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                "message": message,
            }
        )


def run_standard_pretraining_batch_census(
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    plan = compile_standard_pretraining_plan(config)
    base_model = StandardTransformerSpec.from_dict(config["model"])
    dataset_spec = TokenizedTextSpec.from_dict(config["dataset"])
    runtime = PretrainingRuntimeSpec.from_dict(config.get("runtime", {}))
    context = prepare_distributed(runtime, device)
    try:
        corpus = TokenizedTextCorpus(
            dataset_spec, base_model.context_length, base_model.vocab_size
        )
        scales = tuple(plan["scales"])
        batches = tuple(plan["batch_examples"])
        total_tokens = int(plan["total_tokens"])
        target_loss = float(config["target_validation_loss"])
        validation_interval = _positive_int(
            config.get("validation_interval", 4), "validation_interval"
        )
        validation_examples = _positive_int(
            config.get("validation_examples", 32), "validation_examples"
        )
        warmup_steps = _positive_int(config.get("warmup_steps", 0), "warmup_steps", 0)
        minimum_lr_ratio = _finite(
            config.get("minimum_learning_rate_ratio", 0.1),
            "minimum_learning_rate_ratio",
            0.0,
        )
        seeds = tuple(int(value) for value in config.get("seeds", [11, 29]))
        cache_directory = (
            Path(config["cache_directory"]) if config.get("cache_directory") else None
        )
        planned = int(plan["planned_grid_trials"])
        completed = 0
        records: List[BatchRunRecord] = []
        analyses = []
        _emit(progress if context.is_primary else None, "preflight", 0, planned, "Validated text, model, and runtime")
        for scale in scales:
            model_spec = replace(
                base_model,
                width=int(scale["width"]),
                depth=int(scale["depth"]),
                num_heads=int(scale["num_heads"]),
            )
            for optimizer_payload in config["optimizers"]:
                learning_rates = _float_grid(
                    optimizer_payload["learning_rates"], "learning_rates"
                )
                current_records = []
                for batch_examples in batches:
                    for learning_rate in learning_rates:
                        optimizer_spec = _optimizer_from_payload(
                            optimizer_payload, learning_rate
                        )
                        for trial_seed in seeds:
                            _emit(
                                progress if context.is_primary else None,
                                "training-grid",
                                completed,
                                planned,
                                f"{scale['name']} · {optimizer_spec.name} · batch {batch_examples}",
                            )
                            record, _ = run_standard_pretraining_trial(
                                model_spec=model_spec,
                                corpus=corpus,
                                runtime=runtime,
                                distributed_context=context,
                                optimizer_spec=optimizer_spec,
                                total_tokens=total_tokens,
                                batch_examples=batch_examples,
                                seed=trial_seed,
                                target_validation_loss=target_loss,
                                validation_interval=validation_interval,
                                validation_examples=validation_examples,
                                warmup_steps=warmup_steps,
                                minimum_learning_rate_ratio=minimum_lr_ratio,
                                cache_directory=cache_directory,
                            )
                            records.append(record)
                            current_records.append(record)
                            completed += 1
                if context.world_size > 1:
                    torch.distributed.barrier()
                if not context.is_primary:
                    # Rank zero performs definition-preserving single-device
                    # continuation/noise assays while every FSDP peer waits.
                    torch.distributed.barrier()
                    continue
                steps_rows = []
                losses_by_batch: Dict[int, List[float]] = {}
                crossing_key = f"{target_loss:g}"
                for batch_examples in batches:
                    batch_tokens = batch_examples * model_spec.context_length
                    for trial_seed in seeds:
                        choices = [
                            record
                            for record in current_records
                            if record.batch_tokens == batch_tokens and record.seed == trial_seed
                        ]
                        reached = [
                            record
                            for record in choices
                            if record.target_loss_crossings[crossing_key] is not None
                        ]
                        if reached:
                            best = min(
                                reached,
                                key=lambda record: int(
                                    record.target_loss_crossings[crossing_key]
                                ),
                            )
                            steps_rows.append(
                                StepsToTargetObservation(
                                    batch_tokens,
                                    int(best.target_loss_crossings[crossing_key]),
                                    trial_seed,
                                )
                            )
                        losses_by_batch.setdefault(batch_tokens, []).append(
                            min(record.final_validation_loss for record in choices)
                        )
                steps_estimate = estimate_steps_to_target_critical_batch(steps_rows)
                smallest_tokens = batches[0] * model_spec.context_length
                smallest_records = [
                    record for record in current_records if record.batch_tokens == smallest_tokens
                ]
                best_source = min(
                    smallest_records, key=lambda record: record.final_validation_loss
                )

                # Direct and gradient-noise assays are deliberately plain single-device
                # forks. This keeps their definitions independent of data-parallel
                # gradient averaging while the main grid may run under FSDP.
                assay_runtime = replace(runtime, distributed="none", num_processes=1)
                assay_device = context.device if context.is_primary else device
                checkpoint_tokens = int(config.get("checkpoint_tokens", total_tokens // 4))
                checkpoint_tokens -= checkpoint_tokens % smallest_tokens
                checkpoint_tokens = max(smallest_tokens, checkpoint_tokens)
                continuation_tokens = int(config.get("continuation_tokens", total_tokens // 4))
                checkpoint_record, checkpoint_extra = run_standard_pretraining_trial(
                    model_spec=model_spec,
                    corpus=corpus,
                    runtime=assay_runtime,
                    distributed_context=DistributedContext(0, 1, 0, assay_device),
                    optimizer_spec=best_source.optimizer,
                    total_tokens=checkpoint_tokens,
                    batch_examples=batches[0],
                    seed=80_000 + seeds[0],
                    validation_interval=max(1, checkpoint_tokens // smallest_tokens),
                    validation_examples=validation_examples,
                    warmup_steps=0,
                    minimum_learning_rate_ratio=minimum_lr_ratio,
                    cache_directory=cache_directory,
                    cache_suffix="-cbs-checkpoint",
                    return_state=True,
                )
                continuation_rows = []
                for batch_examples in batches:
                    batch_tokens = batch_examples * model_spec.context_length
                    matched_tokens = continuation_tokens - continuation_tokens % batch_tokens
                    if matched_tokens <= 0:
                        continue
                    for trial_seed in seeds:
                        continuation_record, extra = run_standard_pretraining_trial(
                            model_spec=model_spec,
                            corpus=corpus,
                            runtime=assay_runtime,
                            distributed_context=DistributedContext(0, 1, 0, assay_device),
                            optimizer_spec=best_source.optimizer,
                            total_tokens=matched_tokens,
                            batch_examples=batch_examples,
                            seed=90_000 + trial_seed,
                            validation_interval=max(1, matched_tokens // batch_tokens),
                            validation_examples=validation_examples,
                            warmup_steps=0,
                            minimum_learning_rate_ratio=minimum_lr_ratio,
                            cache_directory=cache_directory,
                            cache_suffix=(
                                "-cbs-cont-"
                                + sha256(checkpoint_record.run_id.encode("utf-8")).hexdigest()[:12]
                            ),
                            initial_state=checkpoint_extra["state_dict"],
                            initial_optimizer_state=checkpoint_extra[
                                "optimizer_state_dict"
                            ],
                        )
                        continuation_rows.append(
                            ContinuationObservation(
                                batch_tokens,
                                extra["initial_validation_loss"],
                                continuation_record.final_validation_loss,
                                matched_tokens,
                                trial_seed,
                            )
                        )
                direct_estimate = estimate_direct_checkpoint_critical_batch(
                    continuation_rows
                )
                noise_estimate = _measure_gradient_noise(
                    model_spec,
                    corpus,
                    assay_runtime,
                    seeds[0],
                    batches[0],
                    int(config.get("gradient_noise_samples", 8)),
                    assay_device,
                )
                consensus = combine_critical_batch_estimates(
                    [steps_estimate, direct_estimate, noise_estimate]
                )
                analyses.append(
                    {
                        "scale": dict(scale),
                        "parameter_count": current_records[0].parameter_count,
                        "optimizer": optimizer_payload["name"],
                        "steps_to_target": steps_estimate.to_dict(),
                        "direct_checkpoint": direct_estimate.to_dict(),
                        "gradient_noise": noise_estimate.to_dict(),
                        "consensus": consensus.to_dict(),
                        "loss_optimal_batch": estimate_loss_optimal_batch(
                            losses_by_batch
                        ),
                    }
                )
                if context.world_size > 1:
                    torch.distributed.barrier()
        if context.world_size > 1:
            torch.distributed.barrier()
        if not context.is_primary:
            return {"schema_version": 1, "rank": context.rank, "status": "worker_complete"}
        _emit(progress, "complete", planned, planned, "Pretraining batch census complete")
        return {
            "schema_version": 1,
            "campaign": "standard_text_pretraining_batch_census",
            "status": "completed",
            "plan": plan,
            "runtime": preflight_runtime(base_model, runtime, context.device),
            "dataset": {
                "tokenizer": dataset_spec.tokenizer,
                "fingerprint": corpus.fingerprint,
                "identity_fingerprint": corpus.identity_fingerprint,
                "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
                "tokenizer_is_pinned": corpus.tokenizer_is_pinned,
                "training_tokens": int(corpus.train_tokens.numel()),
                "validation_tokens": int(corpus.validation_tokens.numel()),
            },
            "records": [record.to_dict() for record in records],
            "scale_optimizer_analyses": analyses,
        }
    finally:
        close_distributed(context)
