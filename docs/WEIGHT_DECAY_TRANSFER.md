# AdamW weight-decay transfer

The 100M CompleteP and Jiang-Chizat campaigns tune weight decay in the
`tau_EMA` coordinate used by Dey et al. and Wang-Aitchison:

```text
tau_EMA = 1 / (eta_base * lambda_base * n_steps)
lambda_base = 1 / (tau_EMA * eta_base * n_steps)
```

The reference search is the preregistered Cartesian product of the base
learning-rate grid, the `tau_EMA` grid, and all three seeds. A target ladder is
refused unless the mean-loss optimum is interior in both the learning-rate and
`tau_EMA` coordinates. Both coordinates are then frozen.

For every target scale, `lambda_base` is recomputed from the frozen `eta_base`
and `tau_EMA` and that scale's optimizer-step count. This follows Appendix G.1
of [Dey et al.](https://arxiv.org/abs/2505.01618). As the paper notes, this
timescale does not integrate the learning-rate schedule; therefore each ladder
must preserve one schedule family across every scale.

## Parameter groups

CompleteP applies Table 1 directly:

- embeddings and unembedding: `lambda_base`;
- hidden matrices: `lambda_base * (N/N0)`;
- norms and biases: zero decay.

Jiang-Chizat preserves every Jiang learning-rate and epsilon group, while
applying the CompleteP hidden-matrix decay rule to its rectangular matrices:

- tied token/position embeddings: `lambda_base`;
- attention QKV, attention output, and FFN up matrices:
  `lambda_base * (D/D0)`;
- FFN down matrices: `lambda_base * (M/M0)`;
- norms and biases: zero decay.

The distinction between `D` and `M` is required because the CompleteP decay
factor follows the hidden matrix's input width. Every trial records the base
decay, transferred group decays, formulas, chosen `tau_EMA`, step count, theory
contract, and complete/disjoint parameter assignment audit.

## Production campaigns

- `configs/autoscaler/jiang_mistral_100m_adamw_tau_ema.json`
- `configs/autoscaler/completep_mistral_100m_adamw_tau_ema.json`
- `scripts/run_adamw_100m_ladder_suite.sh`

The suite runs Jiang-Chizat first. Only after its exact AdamW 100M result is
complete does it freeze that result (including its SHA-256 digest) as the
matched baseline for CompleteP. Both campaigns use the same immutable Mistral
token stream, 20 tokens per parameter, three seeds, bf16, FlashAttention, and
eight independent A100 workers. No wrong-learning-rate controls are included.
