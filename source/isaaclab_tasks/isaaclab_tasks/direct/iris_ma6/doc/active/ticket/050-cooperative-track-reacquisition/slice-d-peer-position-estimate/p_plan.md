# Stage P — Implementation Plan (Slice D): peer target-position estimate

Default-off until A/B confirms. **ADD point alongside bearing** (deploy review) ⇒ obs_dim grows.
**CLEAN START** (engineer decision 2026-06-18): train from scratch — NO warm-start, NO weight surgery.
Range source = **A (bbox-depth / oracle range)**. Reward axis OFF.

## Phase 0 — decide + verify (no training) [range source DECIDED = A]
- ✓ Range source = A (bbox-raycaster depth); B/C/D future (D = EKF bearing-only, deploy-faithful).
- Verify A is **non-GT-leaking**: audit `bbox_raycaster_v2._depths` provenance — it derives from the
  detected bbox geometry, but confirm the obs path never substitutes `_target_pos_w`. (Sim depth is
  true range = a SENSOR signal; the *deploy* realizability of range is a tracked sim2real item.)
- Confirm per-(env,agent) depth is available at obs-build time (delayed + GT paths).
- Scope the obs-dim delta (append `*_target_pos_est_w(3)` + age(1) per agent at the END) and the
  cascade to state_space (finalize_observation_and_state_spaces already sizes it).

## Phase 1 — build + test (no training)
- 1a. `target_estimate/` helper (or method): `p_i_est = cam_pos_i + r_i·d_i`, valid-masked; pure,
  unit-tested (0/1/N defined; invalid→zeros+flag=0; point recovers bearing vs peer_pos; range error →
  bounded ego-aim error at small baseline; **assert it never references the GT target** — the GT-leak
  guard that this whole arc earned).
- 1b. Obs wiring: **ADD** `*_target_pos_est_w(3)` (+ validity from bbox_empty [+ age]) APPENDED at the
  end of the obs in BOTH paths (delayed + GT); KEEP `*_ray_w`. Behind cfg flag
  `peer_target_estimate.enabled` (off ⇒ exact baseline, original obs).
- 1c. CLEAN START — no warm-start surgery (obs layout free; net trained from scratch on the new obs).
- 1d. Ablation flag masks the new estimate slots (`peer_target_estimate_ablate`); bearing ablation
  (`peer_bearing_ablate`) stays separate for clean attribution.
- 1e. Experiments: `t050d_posest` (estimate ON, reward axis OFF, scenario+metrics) + `t050d_ablation`;
  evaluate.py `--peer_target_estimate[_ablate]` flags added.
- DONE 2026-06-18: built + env smoke 8/8 (obs 54->57; sizing consistent; point==GT(0.0000m) when peer
  sees, 0 when blind/ablated; flag-off bit-exact dim). Helper `_peer_target_point` (oracle range A).

## Phase 2 — go/no-go (1 seed, CLEAN START ~400k ≈ 37h)
- `t050d_posest` trained FROM SCRATCH (--timesteps 400000, full curriculum, no --checkpoint) vs
  `t050b_baseline_no_info` (C2). **+ ablation** (`--peer_target_estimate_ablate` on the trained policy,
  must match obs dim with `--peer_target_estimate`). Auto-eval queue (reuse /tmp/t050c_eval_queue.sh).
- **Gate:** reacq_success beats C2 0.385 AND the ablation is clearly worse (channel now used).
- Watch `reacq_success_rate` + `pair_valid_rate` early (~20–40k) — kill if it drifts like the v1/v2 runs.

## Phase 3 — rigor (if GO)
- Multi-seed ≥3; the clean single-policy channel-mask ablation figure; deploy-range realizability check
  on a PX4 bag (option B/C if A was GT-range-dependent); optional privileged critic (fix already in).
- If NO-GO: escalate to the learned cooperative observer (reserve).

## Gate (acceptance) — see ticket.md
reacq_success beats C2 0.385; ablation worse; base tracking intact; (then seed-confirmed).
