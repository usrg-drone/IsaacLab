# Stage P — Implementation Plan (Slice D): peer target-position estimate

Default-off until A/B confirms. Dim-preserving swap ⇒ no warm-start surgery. Reward axis OFF.

## Phase 0 — decide + verify (no training)
- **Resolve the range-source Q** (i_design table A/B/C) with the engineer — gates everything.
- Verify the chosen range is **non-GT** (audit: bbox_raycaster `_depths` provenance; confirm it is not
  read from `_target_pos_w`). Confirm depth is exposed per-(env, agent) at reward/obs build time.
- Confirm dim-preserving swap leaves obs_dim/state_space unchanged (so t048 loads directly).

## Phase 1 — build + test (no training)
- 1a. `target_estimate/` helper (or a method): `p_i_est = cam_pos_i + r_i·d_i`, valid-masked; pure,
  unit-tested (0/1/N detections defined; invalid→zeros+flag=0; point recovers bearing vs peer_pos;
  range error → bounded ego-aim error at small baseline; NO GT input — assert the function never
  references the GT target).
- 1b. Obs wiring: replace `ego_ray_w`/`other_ray_w` with `*_target_pos_est_w` in BOTH obs paths
  (delayed + GT); add validity (reuse bbox_empty) [+ optional age]. Bit-exact guarded behind a cfg
  flag `peer_target_estimate.enabled` (off ⇒ original bearing, exact baseline).
- 1c. Extend the ablation flag to mask the estimate slots (`peer_bearing_ablate` → covers the new
  channel; keep obs dim fixed).
- 1d. Experiments: `t050d_posest` (estimate ON, reward axis OFF, scenario+metrics) and reuse
  `t050b_baseline_no_info` as the C2 control. Update evaluate.py ablation flag wiring.
- Test checkpoint: env smoke (16 envs) — estimate present, finite, points near GT when detected
  (sanity, using GT only to CHECK, never to feed); flag-off bit-exact; warm-start loads (no surgery).

## Phase 2 — go/no-go (1 seed, ~15h)
- `t050d_posest` (warm-start t048 wide) vs `t050b_baseline_no_info` (C2). **+ ablation read** (mask the
  point on the trained policy). Auto-eval-on-completion queue (reuse /tmp/t050c_eval_queue.sh pattern).
- **Gate:** reacq_success beats C2 0.385 AND the ablation is clearly worse (channel now used).
- Watch `reacq_success_rate` + `pair_valid_rate` early (~20–40k) — kill if it drifts like the v1/v2 runs.

## Phase 3 — rigor (if GO)
- Multi-seed ≥3; the clean single-policy channel-mask ablation figure; deploy-range realizability check
  on a PX4 bag (option B/C if A was GT-range-dependent); optional privileged critic (fix already in).
- If NO-GO: escalate to the learned cooperative observer (reserve).

## Gate (acceptance) — see ticket.md
reacq_success beats C2 0.385; ablation worse; base tracking intact; (then seed-confirmed).
