from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .pretraining import TokenizedTextCorpus, TokenizedTextSpec


@dataclass(frozen=True)
class FrozenLanguageModelData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    metadata: Dict[str, object]

    @property
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_train, self.y_train, self.x_validation, self.y_validation


def load_frozen_text_windows(
    *,
    train_path: Optional[Path],
    validation_path: Optional[Path],
    tokenizer: str,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    seed: int,
    device: torch.device,
    maximum_bytes: int = 536_870_912,
    token_stream_manifest_path: Optional[Path] = None,
) -> FrozenLanguageModelData:
    corpus = TokenizedTextCorpus(
        TokenizedTextSpec(
            train_path=str(train_path) if train_path is not None else "",
            validation_path=str(validation_path) if validation_path is not None else "",
            tokenizer=tokenizer,
            maximum_bytes=maximum_bytes,
            token_stream_manifest_path=(
                str(token_stream_manifest_path)
                if token_stream_manifest_path is not None
                else None
            ),
        ),
        context_length=context_length,
        vocab_size=vocab_size,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_train, y_train = corpus.sample_batch("train", n_train, generator, "cpu")
    x_validation, y_validation = corpus.sample_batch(
        "validation", n_validation, generator, "cpu"
    )
    return FrozenLanguageModelData(
        x_train.to(device),
        y_train.to(device),
        x_validation.to(device),
        y_validation.to(device),
        {
            "kind": "frozen_real_text_windows",
            "train_path": str(train_path.resolve()) if train_path is not None else None,
            "validation_path": (
                str(validation_path.resolve()) if validation_path is not None else None
            ),
            "token_stream_manifest_path": (
                str(token_stream_manifest_path.resolve())
                if token_stream_manifest_path is not None
                else None
            ),
            "tokenizer": tokenizer,
            "corpus_fingerprint": corpus.fingerprint,
            "dataset_identity_fingerprint": corpus.identity_fingerprint,
            "tokenizer_fingerprint": corpus.tokenizer_fingerprint,
            "tokenizer_is_pinned": corpus.tokenizer_is_pinned,
            "corpus_training_tokens": int(corpus.train_tokens.numel()),
            "corpus_validation_tokens": int(corpus.validation_tokens.numel()),
            "sampled_training_windows": n_train,
            "sampled_validation_windows": n_validation,
            "context_length": context_length,
            "sampling_seed": seed,
            "sampling_policy": "paired torch randint windows, frozen before all trials",
        },
    )

def resolve_transfer_data(
    *,
    train_path: Optional[Path],
    validation_path: Optional[Path],
    tokenizer: str,
    vocab_size: int,
    context_length: int,
    n_train: int,
    n_validation: int,
    dataset_seed: int,
    device: torch.device,
    synthetic_builder,
    maximum_bytes: int = 536_870_912,
    token_stream_manifest_path: Optional[Path] = None,
) -> FrozenLanguageModelData:
    if (train_path is None) != (validation_path is None):
        raise ValueError("train_path and validation_path must be supplied together")
    has_paths = train_path is not None and validation_path is not None
    has_manifest = token_stream_manifest_path is not None
    if has_manifest and has_paths:
        raise ValueError(
            "supply either a token stream manifest or train/validation paths"
        )
    if has_paths or token_stream_manifest_path is not None:
        return load_frozen_text_windows(
            train_path=train_path,
            validation_path=validation_path,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            context_length=context_length,
            n_train=n_train,
            n_validation=n_validation,
            seed=dataset_seed,
            device=device,
            maximum_bytes=maximum_bytes,
            token_stream_manifest_path=token_stream_manifest_path,
        )
    tensors = synthetic_builder(
        vocab_size=vocab_size,
        context_length=context_length,
        n_train=n_train,
        n_validation=n_validation,
        seed=dataset_seed,
        device=device,
    )
    return FrozenLanguageModelData(
        *tensors,
        {
            "kind": "synthetic_markov",
            "vocab_size": vocab_size,
            "context_length": context_length,
            "n_train": n_train,
            "n_validation": n_validation,
            "sampling_seed": dataset_seed,
        },
    )
