# Ticket 050 / Slice C — v1 recovery-shaping FAILED (deficit tax); fixed → v2

**Date**: 2026-06-17
**Run (killed at 159k/200k)**: `2026-06-16_21-06-09_mappo_rnn_torch_t050c_shaping`
**Config (v1)**: `reacq_shaping.enabled=True, shaping_scale=2.0, gamma=0.99, gate=current-deficit`,
on top of the Slice-B info reward, warm-start t048 wide. = `t050b_team_reward` + shaping.

## Result — regressed tracking (NO-GO), killed early
| metric (late) | t050c v1 | t050b_team | C2 |
|---|---|---|---|
| `Coop/reacq_success_rate` | **0.264** (↓ from 0.38) | 0.367 | 0.385 |
| `Detection/pair_valid_rate` | **0.405** (↓ from 0.54) | 0.566 | — |
| `Coop/team_track_maintenance` | 0.402 | 0.564 | 0.53 |
| `Coop/track_loss_event_rate` | 2.98 | 2.53 | 2.05 |
| `Episode_Reward/..._reacq_shaping` | **−0.05** (net negative) | n/a | — |

Every cooperative metric below both t050b_team and C2; `pair_valid` monotonically declining over
training. Training numerically healthy (no NaN, σ≈1.24, value-loss 0.001) → a **reward-design** failure.

## Root cause — the shaping rewarded the opposite of recovery
1. **Deficit tax.** With γ=0.99, `F = (γ−1)·Φ = −0.01·Φ` *every step* you hold alignment during a
   deficit — a steady penalty for staying aimed at the target (the desired behavior). The agent
   reduces it by *lowering Φ* (pointing away) during deficits → sabotages re-acquisition.
2. **Recovery never credited.** On the re-acquisition step the tracker flips `DEFICIT→HOLD`, so
   `active = region & region_prev` gated out the big positive `ΔΦ`. Net shaping = negative drip only.

(Magnitude was small (−0.05), so single-seed variance can't be fully excluded — but the monotonic,
across-the-board decline vs t050b_team's stability + the concrete perverse gradient make it a clear
no-go and a real bug.)

## Fix (v2) — committed
- **γ = 1.0** (`F = Φ′ − Φ`): holding alignment → F=0 (no tax); only *improvement* rewarded.
- **Gate on `region_prev`** (in deficit at the START of the step): credits entry=0, mid-deficit
  re-pointing, AND the recovery step.
- **`shaping_scale` 2.0 → 10.0** default (recovery now telescopes to ~scale·ΔΦ; recalibrate via the
  Phase-2 sweep ~10/20/40).
- Regression tests added (no-holding-tax at γ=1; γ<1 tax contrast). Module 11/11.

v2 re-run: `t050c_shaping` with `reacq_shaping.shaping_scale=10.0` (γ=1 default), same warm-start.
Watch `Coop/reacq_success_rate` and `pair_valid_rate` early — kill again if they decline like v1.
