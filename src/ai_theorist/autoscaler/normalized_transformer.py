from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .lr_contract import LearningRateTheory, audit_optimizer_groups, theory_group
from .schema import ArchitectureTemplate, DatasetSpec, ScaleLevel


NUGPT_ADAM_THEORY = LearningRateTheory(
    contract_id="shigida-hanin-gromov-nugpt-adam-v2",
    architecture="normalized transformer (nuGPT)",
    optimizer="adam",
    source_title="Learning Rate Transfer in Normalized Transformers",
    source_url="https://arxiv.org/abs/2604.27077",
    source_version="arXiv:2604.27077v2, Table 1",
    base_coordinate="eta_global * (iteration_count/base_iteration_count)^(-1/3)",
    applicability=(
        "nuGPT width/depth/token-horizon transfer with unit-sphere projection, "
        "fixed head dimension, Adam betas (0.9, 0.95), zero weight decay, and "
        "depth correction in residual-rescaler initialization"
    ),
)


def unit_norm(x: Tensor, dim: int = -1, epsilon: float = 1e-12) -> Tensor:
    """Normalize in float32, then restore the input dtype."""
    dtype = x.dtype
    values = x.float()
    normalized = values / values.norm(p=2, dim=dim, keepdim=True).clamp_min(epsilon)
    return normalized.to(dtype=dtype)


def apply_rotary_embeddings(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply standard RoPE to tensors shaped ``[batch, heads, time, head_dim]``."""
    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("Rotary embeddings require an even head dimension")
    positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    frequencies = torch.exp(
        torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
        * (-math.log(10_000.0) / head_dim)
    )
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]

    def rotate(x: Tensor) -> Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


class NormalizedTransformerBlock(nn.Module):
    """One nGPT attention/SwiGLU block operating on the unit hypersphere."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_multiplier: int,
        *,
        depth_multiplier: float,
        parameterization: str,
        attention_backend: str = "math",
        capture_attention_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.mlp_width = mlp_multiplier * width
        self.depth_multiplier = depth_multiplier
        if attention_backend not in {"auto", "math", "flash"}:
            raise ValueError("attention_backend must be auto, math, or flash")
        self.attention_backend = attention_backend
        self.capture_attention_diagnostics = capture_attention_diagnostics
        if parameterization not in {"nugpt", "baseline_ngpt"}:
            raise ValueError("parameterization must be nugpt or baseline_ngpt")
        self.parameterization = parameterization
        self.alpha_initial_value = (
            0.05 / depth_multiplier if parameterization == "nugpt" else 0.05
        )
        self.rescaler_parameter_scale = (
            0.03 if parameterization == "nugpt" else 1.0 / math.sqrt(width)
        )
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.mlp_input = nn.Linear(width, 2 * self.mlp_width, bias=False)
        self.mlp_output = nn.Linear(self.mlp_width, width, bias=False)

        self.attention_alpha = nn.Parameter(
            torch.full((width,), self.rescaler_parameter_scale)
        )
        self.mlp_alpha = nn.Parameter(torch.full((width,), self.rescaler_parameter_scale))
        self.qk_scale = nn.Parameter(torch.full((width,), self.rescaler_parameter_scale))
        self.mlp_scale = nn.Parameter(torch.ones(2 * self.mlp_width))
        self.register_buffer("_last_attention_entropy", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("_last_hidden_norm_error", torch.tensor(float("nan")), persistent=False)

    def _effective_alpha(self, parameter: Tensor) -> Tensor:
        return parameter.abs() * (
            self.alpha_initial_value / self.rescaler_parameter_scale
        )

    def _attention(self, hidden: Tensor) -> Tensor:
        batch, time, _ = hidden.shape
        q = self.query(hidden).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_embeddings(q, k)
        coordinate_scale = (self.qk_scale / self.rescaler_parameter_scale).view(
            1, self.num_heads, 1, self.head_dim
        )
        q = coordinate_scale * unit_norm(q)
        k = coordinate_scale * unit_norm(k)
        kernel_context = nullcontext()
        if self.attention_backend != "auto":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backend = (
                SDPBackend.FLASH_ATTENTION
                if self.attention_backend == "flash"
                else SDPBackend.MATH
            )
            kernel_context = sdpa_kernel([backend])
        # nGPT deliberately uses sqrt(d_head) as the scale on normalized Q/K.
        with kernel_context:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                scale=math.sqrt(self.head_dim),
            )
        if self.capture_attention_diagnostics:
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) * math.sqrt(
                    self.head_dim
                )
                causal_mask = torch.ones(
                    time, time, device=hidden.device, dtype=torch.bool
                ).triu(1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
                probabilities = scores.softmax(dim=-1)
                entropy = -(
                    probabilities.float()
                    * probabilities.float().clamp_min(1e-12).log()
                )
                self._last_attention_entropy.copy_(entropy.sum(dim=-1).mean())
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.width)
        return self.attention_output(attended)

    def _sphere_update(self, hidden: Tensor, branch: Tensor, alpha: Tensor) -> Tensor:
        hidden_unit = unit_norm(hidden)
        branch_unit = unit_norm(branch)
        mixed = hidden_unit + alpha.view(1, 1, -1) * (branch_unit - hidden_unit)
        return unit_norm(mixed)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self._sphere_update(
            hidden,
            self._attention(hidden),
            self._effective_alpha(self.attention_alpha),
        )
        u, v = self.mlp_input(hidden).chunk(2, dim=-1)
        u_scale, v_scale = self.mlp_scale.chunk(2)
        u = u * u_scale.view(1, 1, -1)
        v = v * v_scale.view(1, 1, -1) * math.sqrt(self.width)
        mlp_branch = self.mlp_output(u * F.silu(v))
        hidden = self._sphere_update(
            hidden,
            mlp_branch,
            self._effective_alpha(self.mlp_alpha),
        )
        with torch.no_grad():
            self._last_hidden_norm_error.copy_(
                (hidden.float().norm(dim=-1) - 1.0).abs().max()
            )
        return hidden


class NormalizedTransformer(nn.Module):
    """Decoder-only 2024 nGPT with explicit post-step sphere projection."""

    def __init__(
        self,
        architecture: ArchitectureTemplate,
        scale: ScaleLevel,
        *,
        parameterization: str = "nugpt",
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
        capture_attention_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if parameterization not in {"nugpt", "baseline_ngpt"}:
            raise ValueError("parameterization must be nugpt or baseline_ngpt")
        self.parameterization = parameterization
        self.activation_checkpointing = activation_checkpointing
        self.width = scale.width
        self.depth = scale.repeats
        self.reference_width = architecture.reference_width
        self.reference_depth = architecture.reference_depth
        self.width_multiplier = self.width / self.reference_width
        self.depth_multiplier = self.depth / self.reference_depth
        self.context_length = architecture.context_length
        self.vocab_size = architecture.vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, self.width)
        self.blocks = nn.ModuleList(
            NormalizedTransformerBlock(
                self.width,
                self.width // architecture.head_dimension,
                architecture.mlp_multiplier,
                depth_multiplier=self.depth_multiplier,
                parameterization=parameterization,
                attention_backend=attention_backend,
                capture_attention_diagnostics=capture_attention_diagnostics,
            )
            for _ in range(scale.repeats)
        )
        # nGPT intentionally does not tie input and output embeddings.
        self.language_model_head = nn.Linear(self.width, self.vocab_size, bias=False)
        self.rescaler_parameter_scale = (
            0.03 if parameterization == "nugpt" else 1.0 / math.sqrt(self.width)
        )
        self.logit_initial_value = (
            math.sqrt(self.width_multiplier) if parameterization == "nugpt" else 1.0
        )
        self.logit_scale = nn.Parameter(
            torch.full((self.vocab_size,), self.rescaler_parameter_scale)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        standard_deviation = 1.0 / math.sqrt(self.width)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
        with torch.no_grad():
            for block in self.blocks:
                block.attention_alpha.fill_(block.rescaler_parameter_scale)
                block.mlp_alpha.fill_(block.rescaler_parameter_scale)
                block.qk_scale.fill_(block.rescaler_parameter_scale)
                block.mlp_scale.fill_(1.0)
            self.logit_scale.fill_(self.rescaler_parameter_scale)
        self.project_normalized_weights()

    def _normalized_weight_axes(self) -> Iterable[Tuple[str, Tensor, int]]:
        yield "token_embedding", self.token_embedding.weight, 1
        yield "language_model_head", self.language_model_head.weight, 1
        for index, block in enumerate(self.blocks):
            yield f"blocks.{index}.query", block.query.weight, 1
            yield f"blocks.{index}.key", block.key.weight, 1
            yield f"blocks.{index}.value", block.value.weight, 1
            yield f"blocks.{index}.attention_output", block.attention_output.weight, 0
            yield f"blocks.{index}.mlp_input", block.mlp_input.weight, 1
            yield f"blocks.{index}.mlp_output", block.mlp_output.weight, 0

    @torch.no_grad()
    def project_normalized_weights(self) -> None:
        for _, weight, dimension in self._normalized_weight_axes():
            weight.copy_(unit_norm(weight, dim=dimension))

    def optimizer_parameter_groups(
        self,
        global_learning_rate: float,
        *,
        data_multiplier: float = 1.0,
        output_learning_rate_multiplier: float = 0.5,
        adam_epsilon: float = 1e-16,
    ) -> List[Dict[str, object]]:
        """Table-1 nuGPT rates under the mid-alignment recommendation.

        The output multiplier is a scale-independent base-model hyperparameter;
        the paper predicts the same width exponent as hidden matrices.  Keeping
        it explicit prevents a tuned constant from being mistaken for a scaling
        exponent.
        """
        for value, name in (
            (global_learning_rate, "global_learning_rate"),
            (data_multiplier, "data_multiplier"),
            (output_learning_rate_multiplier, "output_learning_rate_multiplier"),
            (adam_epsilon, "adam_epsilon"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        base_learning_rate = global_learning_rate * data_multiplier ** (-1.0 / 3.0)
        input_rate = base_learning_rate * self.width_multiplier ** -0.5
        hidden_rate = base_learning_rate * self.width_multiplier ** -0.75
        output_rate = output_learning_rate_multiplier * hidden_rate
        hidden_weights: List[Tensor] = []
        rescalers: List[Tensor] = [self.logit_scale]
        for block in self.blocks:
            hidden_weights.extend(
                [
                    block.query.weight,
                    block.key.weight,
                    block.value.weight,
                    block.attention_output.weight,
                    block.mlp_input.weight,
                    block.mlp_output.weight,
                ]
            )
            rescalers.extend(
                [
                    block.attention_alpha,
                    block.mlp_alpha,
                    block.qk_scale,
                    block.mlp_scale,
                ]
            )
        factors = {
            "width_ratio": self.width_multiplier,
            "depth_ratio": self.depth_multiplier,
            "iteration_ratio": data_multiplier,
            "output_lr_multiplier": output_learning_rate_multiplier,
        }
        return [
            theory_group(
                name="nugpt_input",
                params=[self.token_embedding.weight],
                lr=input_rate,
                lr_formula="eta_global * m_data^(-1/3) * m_width^(-1/2)",
                theory=NUGPT_ADAM_THEORY,
                scale_factors=factors,
                eps=adam_epsilon,
                eps_formula="epsilon_optimizer; model-scale invariant before declared batch/duration transforms",
            ),
            theory_group(
                name="nugpt_hidden",
                params=hidden_weights,
                lr=hidden_rate,
                lr_formula="eta_global * m_data^(-1/3) * m_width^(-3/4); no depth LR factor",
                theory=NUGPT_ADAM_THEORY,
                scale_factors=factors,
                eps=adam_epsilon,
                eps_formula="epsilon_optimizer; model-scale invariant before declared batch/duration transforms",
            ),
            theory_group(
                name="nugpt_output",
                params=[self.language_model_head.weight],
                lr=output_rate,
                lr_formula="c_output * eta_global * m_data^(-1/3) * m_width^(-3/4)",
                theory=NUGPT_ADAM_THEORY,
                scale_factors=factors,
                eps=adam_epsilon,
                eps_formula="epsilon_optimizer; model-scale invariant before declared batch/duration transforms",
            ),
            theory_group(
                name="nugpt_rescalers",
                params=rescalers,
                lr=base_learning_rate,
                lr_formula="eta_global * m_data^(-1/3) on raw rescaler coordinates",
                theory=NUGPT_ADAM_THEORY,
                scale_factors=factors,
                eps=adam_epsilon,
                eps_formula="epsilon_optimizer; model-scale invariant before declared batch/duration transforms",
            ),
        ]

    def optimizer_contract_audit(
        self,
        global_learning_rate: float,
        *,
        data_multiplier: float = 1.0,
        output_learning_rate_multiplier: float = 0.5,
        adam_epsilon: float = 1e-16,
    ) -> Dict[str, object]:
        groups = self.optimizer_parameter_groups(
            global_learning_rate,
            data_multiplier=data_multiplier,
            output_learning_rate_multiplier=output_learning_rate_multiplier,
            adam_epsilon=adam_epsilon,
        )
        return audit_optimizer_groups(self, groups, NUGPT_ADAM_THEORY)

    @torch.no_grad()
    def sphere_diagnostics(self) -> Dict[str, float]:
        errors: List[Tensor] = []
        for _, weight, dimension in self._normalized_weight_axes():
            errors.append((weight.float().norm(dim=dimension) - 1.0).abs().max())
        hidden_errors = [block._last_hidden_norm_error for block in self.blocks]
        entropies = [block._last_attention_entropy for block in self.blocks]
        finite_hidden = [value for value in hidden_errors if torch.isfinite(value)]
        finite_entropies = [value for value in entropies if torch.isfinite(value)]
        return {
            "maximum_matrix_norm_error": float(torch.stack(errors).max().cpu()),
            "maximum_hidden_norm_error": (
                float(torch.stack(finite_hidden).max().cpu()) if finite_hidden else float("nan")
            ),
            "mean_attention_entropy": (
                float(torch.stack(finite_entropies).mean().cpu())
                if finite_entropies
                else float("nan")
            ),
            "mean_attention_alpha": float(
                torch.stack(
                    [block._effective_alpha(block.attention_alpha).mean() for block in self.blocks]
                ).mean().cpu()
            ),
            "mean_mlp_alpha": float(
                torch.stack(
                    [block._effective_alpha(block.mlp_alpha).mean() for block in self.blocks]
                ).mean().cpu()
            ),
            "mean_logit_scale": float(
                (
                    self.logit_scale
                    * (self.logit_initial_value / self.rescaler_parameter_scale)
                ).mean().cpu()
            ),
        }

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        if tokens.shape[1] > self.context_length:
            raise ValueError("token sequence exceeds configured context length")
        hidden = unit_norm(self.token_embedding(tokens))
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden = activation_checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        logits = self.language_model_head(hidden)
        effective_logit_scale = self.logit_scale * (
            self.logit_initial_value / self.rescaler_parameter_scale
        )
        return logits * effective_logit_scale.view(1, 1, -1)


def make_synthetic_markov_dataset(
    architecture: ArchitectureTemplate,
    dataset: DatasetSpec,
    *,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a configurable Markov language with fixed held-out sequences."""
    vocab_size = architecture.vocab_size
    length = architecture.context_length + 1
    transition_generator = torch.Generator(device="cpu").manual_seed(dataset.seed + 10_003)
    transition = torch.stack(
        [
            torch.randperm(vocab_size, generator=transition_generator)
            for _ in range(dataset.markov_states)
        ],
        dim=0,
    )
    noise_probability = min(1.0, float(dataset.noise_std))

    def generate(count: int, seed: int) -> Tensor:
        initial_generator = torch.Generator(device="cpu").manual_seed(seed)
        replace_generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
        random_token_generator = torch.Generator(device="cpu").manual_seed(seed + 2017)
        tokens = torch.empty(count, length, dtype=torch.long)
        tokens[:, : dataset.markov_order] = torch.randint(
            vocab_size,
            (count, dataset.markov_order),
            generator=initial_generator,
        )
        replacement_draws = torch.rand(count, length, generator=replace_generator)
        random_tokens = torch.randint(
            vocab_size,
            (count, length),
            generator=random_token_generator,
        )
        for position in range(dataset.markov_order, length):
            state = torch.zeros(count, dtype=torch.long)
            for lag in range(dataset.markov_order, 0, -1):
                state = (
                    state * 131 + tokens[:, position - lag]
                ).remainder(dataset.markov_states)
            previous = tokens[:, position - 1]
            next_token = transition[state, previous]
            if noise_probability:
                replace = replacement_draws[:, position] < noise_probability
                next_token = torch.where(replace, random_tokens[:, position], next_token)
            tokens[:, position] = next_token
        return tokens

    training = generate(dataset.n_train, dataset.seed)
    validation = generate(dataset.n_validation, dataset.seed + 1)
    return tuple(
        value.to(device)
        for value in (
            training[:, :-1],
            training[:, 1:],
            validation[:, :-1],
            validation[:, 1:],
        )
    )  # type: ignore[return-value]
