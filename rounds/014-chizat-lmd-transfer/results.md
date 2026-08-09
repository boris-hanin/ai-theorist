# Round 014 — joint Chizat `L`, `M`, and `D` transfer

## Status

Retrospective validation round.  The initial CPU screens preceded this record,
so this is not labelled preregistered.  The two A100 confirmation horizons and
the constant-`LM/D` comparison used fixed rules and common seeds.

## Primary parameterization

```text
lr_U = L M eta / D
lr_W = L M D eta
eta = 0.003
```

Five seeds were split across two A100 workers, with seed 0 duplicated.  Every
reported campaign verified 20 or 30 exact duplicate trials after merging.

## Verdict table

| path | horizon | fractional progress across five shapes | log-progress slope | verdict |
|---|---:|---|---:|---|
| pure `L` | 80 | 0.0240–0.0246 | -0.00869 | pass |
| pure `M` | 80 | 0.0236–0.0241 | +0.00682 | pass |
| pure `D`, coupled | 80 | 0.0216–0.0279 | +0.00560 | pass |
| joint `L,M,D`, `LM/D` 16→256 | 80 | 0.0262–0.0303 | -0.0503 | pass |
| same joint path | 320 | 0.102–0.115 | -0.0280 | pass |
| joint `L,M,D`, `LM/D=8` | 80 | 0.0243–0.0274 | **+0.00473** | **pass; preferred** |

The constant-`LM/D` path is the cleanest observed joint path, matching the
theory-motivated invariant.

## Controls and channel isolation

- Removing `L` rejects the pure-`L` and joint claims.
- Removing `M` rejects the pure-`M` and joint claims.
- The incoherent square-root-`D` surrogate has joint slope `-0.493` and is
  rejected; the coherent law has slope `-0.0503` on the same shapes.
- A single fixed-`D` rate and a global `LMD` rate reject at the discriminating
  80-step joint horizon.  They can saturate at 320 steps, so that horizon is
  evidence for persistence of the primary law, not for control separation.
- `W`-only pure-`D` training passes with slope `+0.0470`.
- `U`-only training fails the coupled-law gate because a frozen random `W`
  cannot form the coherent alignment assumed by the `U` rate.  This failure is
  retained rather than hidden.

## Product consequence

MoE scale ladders expose `L`, expert width `M`, and embedding width `D`
independently.  Their default joint path keeps `LM/D` constant.  Optimizer
group rates remain explicit; the UI never presents one ambiguous global raw
learning rate for a scale-dependent MoE component.
