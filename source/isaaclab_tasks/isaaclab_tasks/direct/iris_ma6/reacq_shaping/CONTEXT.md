# reacq_shaping — CONTEXT

## Purpose
Potential-based shaping (PBRS) reward that gives a lost agent a **dense, per-agent gradient to
re-point at the target during a single-agent deficit** — the sparse/delayed-credit gap Slice B left
(ticket 050, Slice C). Reward-side only; privileged GT target, never enters obs.

## Inputs
- `boresight_w` [N, A, 3]: camera optical axis in world frame = `quat_rotate(camera_orientation_w, +z)`.
  Use the **boresight**, not the bbox-center ray (`camera_ray_directions_w`) — the latter is undefined
  exactly when the agent has lost the target.
- `cam_pos_w` [N, A, 3]: agent/camera world position.
- `target_pos_w` [N, 3]: GT target world position (privileged, reward-only).
- `deficit_mask` [N, A] bool: agent in a peer-assisted DEFICIT (from `ReacquisitionTracker.in_deficit()`).
- `t` float: current sim time (idempotency guard).

## Outputs
- `compute_shaping(...) -> F[N, A]`: per-agent shaping reward `scale * (gamma*Phi' - Phi)`, gated to
  the deficit regime. Fed into the env reward dict as the `reacq_shaping` key (per-step PBRS term —
  added WITHOUT the `step_dt` factor the rate-based terms carry).

## Dependencies
- `torch`, `isaaclab.utils.configclass`. No Isaac Sim handles. Pure tensor math + small internal state.
- Consumes the DEFICIT mask from `cooperation_metrics.ReacquisitionTracker` (gate requires
  `cooperation_metrics.enable=True`).

## Calling Contract (§4.4)
- `compute_shaping(...)`: **WRITE** — advances `Phi_prev` / `region_prev`. Call **once per sim step**
  in `_get_rewards`. Idempotent within a step via a `_last_update_time` guard (returns cached F on a
  repeat call at the same `t`). Entry step / first post-reset step yield F=0 (no spurious spike).
- `reset_idx(env_ids)`: **WRITE** — clears per-(env,agent) state for the given envs. Call in `_reset_idx`.

## Invariants
- `Phi in [0, 1]`; `F` bounded by `scale * [-1, gamma]` per step.
- PBRS leaves the task optimum unchanged (gating relaxes this at regime boundaries — accepted; the
  goal is a realizable dense recovery signal, not optimality purity).
- `enabled=False` -> not instantiated by the env -> bit-exact baseline.

## Key files
- `reacq_shaping.py` — `ReacqShaper` (state machine + potential).
- `reacq_shaping_cfg.py` — `ReacqShapingCfg`.
- `tests/run_tests.py` — standalone unit tests.
