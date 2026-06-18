# Stage I — Design (Slice D): usable peer target-position estimate in obs

## Problem
The Slice-C ablation proved the peer **bearing** channel is unused: masking `other_ray_w` did not hurt
re-acquisition (0.289→0.311). A bearing is a *line*; to re-aim, the lost agent must resolve it to a
*point* (range along the peer ray) and transform to its own frame — an implicit triangulation the RNN
does not learn from raw (peer_pos, peer_ray, ego_pos). Give it the point directly.

## Proposed approach — swap bearing → resolved point (dimension-preserving)
For each agent i with a valid detection, compute a single-agent target POINT:

    p_i_est = cam_pos_i + r_i · d_i        # d_i = bbox-center bearing (world); r_i = SENSOR range
                                           # invalid (no detection) -> zeros + valid flag = 0

Place `p_i_est` (world, 3D) in the obs **in the slot currently holding the bearing** `*_ray_w`:
- peer block: `other_ray_w(3)` -> `other_target_pos_est_w(3)`  (the load-bearing one)
- ego block:  `ego_ray_w(3)`   -> `ego_target_pos_est_w(3)`    (symmetry; ego's own estimate)

**Dimension-preserving** ⇒ obs_dim / state_space / preprocessor unchanged ⇒ **t048/t050 warm-start
loads directly, no surgery**. The swapped slots carry new semantics (point, not ray); the policy
re-learns them — fine, since the ablation showed the ray slots were already ignored. The lost agent
recovers via `normalize(other_target_pos_est_w − ego_pos)` — a trivial transform vs implicit
triangulation. (peer_pos stays in obs, so the point also encodes the bearing: d = (p_est−peer_pos).)

## THE central design decision (Stage Q) — the range source r_i  [REALIZABILITY]
The whole arc's lesson: a reward/obs signal must be **realizable from non-GT information**. `p_i_est`
MUST NOT be built from `_target_pos_w` (GT) — that would leak the answer into obs and make the
ablation meaningless. r_i options, to be chosen with the engineer (deploy-faithfulness vs simplicity):

| option | r_i source | realizable at deploy? | notes |
|---|---|---|---|
| **A (rec.)** | per-agent **bbox-raycaster depth** (`_depths`) | depth/RGBD or known-size monocular | sim depth is true range — a SENSOR-class signal; deploy needs a depth estimate. Cleanest test of the RL question; harden range for deploy as a tracked sim2real item. |
| B | **carry-forward last team triangulation** range/point through the gap | yes (derived from obs) | "belief through the measurement gap" (Dec-POMDP); stale (target moved) but peer ray refines it. More state to manage. |
| C | monocular **size→range** (target apparent size) | yes if target size known | noisiest; closest to a real monocular detector. |

Recommendation: **A** for the first go/no-go (isolates "does a usable point enable cooperation?"),
with the deploy range-realizability flagged for the bag validation. If A works but is GT-range-
dependent, re-test under B/C before claiming sim2real.

Add per estimate: a **validity flag** (reuse `bbox_empty`) and optionally **age** so the policy can
discount a stale peer point. Keep z handling consistent with the env's wxyz/world conventions.

## Reward / scenario
Reward = **C2 baseline** (`information_reward.enabled=False`, `reacq_shaping.enabled=False`) — both
reward-axis interventions failed; isolate the obs channel. Scenario + cooperation_metrics ON (reused).

## Ablation (the paper claim)
`peer_bearing_ablate`-style flag (rename/extend to mask the new `*_target_pos_est_w` slots). Obs dim
fixed ⇒ single-policy on/off eval, and a trained-without arm for the stronger claim. Expectation now:
masking the point **degrades** re-acquisition (the channel is finally usable).

## Fallback — learned cooperative observer (reserve)
If the env-computed point is insufficient (range too noisy, stale estimate unusable), escalate to a
recurrent belief / differentiable filter over own+peer bearings (GT-aux-supervised, privileged-critic
style — actor sees only the belief output). The parent's architecture story; build only on need.

## Open risks
1. **GT leak (critical):** r_i must be a sensor signal, NOT `_target_pos_w`. Audit the data path.
2. **Deploy range realizability:** sim depth ≠ deploy range; validate on the bag (option B/C if needed).
3. **Stale estimate during a deficit:** the peer point lags the moving target; small-baseline geometry
   makes ego re-aim tolerant to range error — quantify; `age` lets the policy discount.
4. **Swapped-slot relearning:** warm-start helps the rest of the net; the 3 (×2) swapped input columns
   relearn. Acceptable (they were ignored), but watch early training.
5. Vertical (z) weakly constrained by a single bearing + range; the point's z is range-sensitive.
