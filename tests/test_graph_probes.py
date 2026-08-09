import math

import torch

import gt


def test_attention_formula_g_is_exact_on_recorded_values():
    data = gt.dataset(B=3, N=8, n0=4, radius=0.6, seed=3)
    net = gt.GraphTransformer(D=32, L=1, H=4, n0=4, alpha_A=0.5, seed=2)
    with torch.no_grad():
        _, record = net.forward(data, record=True)
        weights = record["S"][0]
        values = record["value"][0].permute(0, 2, 1, 3)
        values = (values / values.pow(2).sum(-1, keepdim=True).sqrt()
                  * math.sqrt(net.Dh))
        rho = torch.einsum("bhuq,bhvq->bhuv", values, values) / net.Dh
        predicted = torch.einsum("bhuv,bhuw,bhvw->bhu", weights, weights, rho).mean()
        measured = torch.einsum("bhuv,bhvq->bhuq", weights, values).pow(2).mean()
    assert torch.allclose(predicted, measured, rtol=1e-12, atol=1e-12)


def test_attention_stats_uses_aligned_recorded_tensors():
    data = gt.dataset(B=3, N=8, n0=4, radius=0.6, seed=4)
    net = gt.GraphTransformer(D=32, L=2, H=4, n0=4, seed=5)
    stats = gt.attention_stats(net, data)
    assert len(stats) == 2
    assert all(0.0 < row["gamma_A"] <= 1.0 for row in stats)
