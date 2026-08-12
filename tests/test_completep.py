import math

import pytest
import torch
from torch.nn import functional as F

from ai_theorist.autoscaler.completep import (
    CompletePReference,
    CompletePShape,
    CompletePTransformer,
)


def make_model(
    shape: CompletePShape = CompletePShape(2, 16, 4, 4),
    **kwargs,
) -> CompletePTransformer:
    torch.manual_seed(17)
    return CompletePTransformer(
        shape,
        vocab_size=32,
        context_length=8,
        reference=CompletePReference(2, 16),
        **kwargs,
    )


def test_completep_forward_is_causal_and_readout_is_untied() -> None:
    model = make_model().eval()
    tokens = torch.arange(8)[None, :]
    changed = tokens.clone()
    changed[0, -1] = 19
    with torch.no_grad():
        logits = model(tokens)
        changed_logits = model(changed)
    assert logits.shape == (1, 8, 32)
    torch.testing.assert_close(logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(logits[:, -1], changed_logits[:, -1])
    assert model.token_embedding.weight is not model.unembedding.weight


def test_completep_width_depth_forward_and_initialization_rules() -> None:
    shape = CompletePShape(8, 64, 8, 4)
    model = make_model(shape)
    assert model.width_ratio == 4.0
    assert model.depth_ratio == 4.0
    assert model.blocks[0].residual_scale == pytest.approx(0.25)
    expected_hidden_std = 0.02 / math.sqrt(4.0)
    assert float(model.blocks[0].attention.qkv.weight.detach().std()) == pytest.approx(
        expected_hidden_std, rel=0.04
    )
    assert float(model.token_embedding.weight.detach().std()) == pytest.approx(
        0.02, rel=0.04
    )
    assert model.diagnostics()["unembedding_forward_scale"] == pytest.approx(0.25)


def test_completep_table1_adamw_groups_are_complete_and_exact() -> None:
    model = make_model(CompletePShape(8, 32, 8, 4))
    groups = {
        str(group["name"]): group
        for group in model.optimizer_parameter_groups(
            0.01, epsilon0=1e-16, weight_decay0=0.1
        )
    }
    assert set(groups) == {
        "completep_embeddings",
        "completep_hidden_norms",
        "completep_hidden_weights",
        "completep_hidden_biases",
        "completep_final_norm",
        "completep_unembedding",
    }
    assert groups["completep_embeddings"]["lr"] == pytest.approx(0.01)
    assert groups["completep_hidden_weights"]["lr"] == pytest.approx(0.005)
    assert groups["completep_hidden_norms"]["lr"] == pytest.approx(0.01)
    assert groups["completep_hidden_biases"]["lr"] == pytest.approx(0.01)
    assert groups["completep_unembedding"]["lr"] == pytest.approx(0.01)
    assert groups["completep_embeddings"]["eps"] == pytest.approx(0.5e-16)
    assert groups["completep_hidden_weights"]["eps"] == pytest.approx(0.125e-16)
    assert groups["completep_final_norm"]["eps"] == pytest.approx(0.5e-16)
    assert groups["completep_embeddings"]["weight_decay"] == pytest.approx(0.1)
    assert groups["completep_hidden_weights"]["weight_decay"] == pytest.approx(0.2)
    assert groups["completep_hidden_norms"]["weight_decay"] == 0.0
    assigned = [id(parameter) for group in groups.values() for parameter in group["params"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {id(parameter) for parameter in model.parameters()}
    audit = model.optimizer_contract_audit(
        0.01, epsilon0=1e-16, weight_decay0=0.1
    )
    assert audit["complete"] is True
    assert audit["disjoint"] is True
    assert len(audit["groups"]) == 6


def test_completep_one_adamw_step_is_finite() -> None:
    model = make_model().train()
    groups = model.optimizer_parameter_groups(
        0.001, epsilon0=1e-16, weight_decay0=0.01
    )
    optimizer = torch.optim.AdamW(groups, lr=0.001, betas=(0.9, 0.95), eps=1e-16)
    tokens = torch.randint(0, 32, (4, 8), generator=torch.Generator().manual_seed(3))
    targets = tokens.roll(-1, dims=1)
    initial = F.cross_entropy(model(tokens).reshape(-1, 32), targets.reshape(-1))
    initial.backward()
    optimizer.step()
    final = F.cross_entropy(model(tokens).reshape(-1, 32), targets.reshape(-1))
    assert torch.isfinite(initial)
    assert torch.isfinite(final)


def test_completep_accelerated_sdpa_matches_math() -> None:
    reference = make_model(
        attention_backend="math", capture_attention_diagnostics=False
    ).train()
    accelerated = make_model(
        attention_backend="auto", capture_attention_diagnostics=False
    ).train()
    accelerated.load_state_dict(reference.state_dict())
    tokens = torch.randint(0, 32, (2, 8), generator=torch.Generator().manual_seed(29))
    reference_logits = reference(tokens)
    accelerated_logits = accelerated(tokens)
    torch.testing.assert_close(accelerated_logits, reference_logits, rtol=1e-5, atol=1e-6)


def test_completep_attention_uses_paper_width_normalization() -> None:
    model = make_model(
        attention_backend="math", capture_attention_diagnostics=False
    ).eval()
    attention = model.blocks[0].attention
    hidden = torch.randn(
        2, 5, attention.width, generator=torch.Generator().manual_seed(41)
    )

    with torch.no_grad():
        actual = attention(hidden)
        q, k, v = attention.qkv(hidden).chunk(3, dim=-1)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(
                hidden.shape[0],
                hidden.shape[1],
                attention.num_heads,
                attention.head_dimension,
            ).transpose(1, 2)

        q, k, v = (split_heads(value) for value in (q, k, v))
        logits = torch.matmul(q, k.transpose(-2, -1)) / attention.width
        causal_mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
        probabilities = logits.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)
        attended = torch.matmul(probabilities, v)
        attended = attended.transpose(1, 2).contiguous().view(2, 5, attention.width)
        expected = attention.output(attended)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
