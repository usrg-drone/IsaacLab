# Stage I — Design (Slice D): usable peer target-position estimate in obs

## Problem
The Slice-C ablation proved the peer **bearing** channel is unused: masking `other_ray_w` did not hurt
re-acquisition (0.289→0.311). A bearing is a *line*; to re-aim, the lost agent must resolve it to a
*point* (range along the peer ray) and transform to its own frame — an implicit triangulation the RNN
does not learn from raw (peer_pos, peer_ray, ego_pos). Give it the point directly.

## Proposed approach — ADD the resolved point alongside the bearing (deployment review, 2026-06-18)
For each agent i with a valid detection, compute a single-agent target POINT:

    p_i_est = cam_pos_i + r_i · d_i        # d_i = bbox-center bearing (world); r_i = SENSOR range
                                           # invalid (no detection) -> zeros + valid flag = 0

**ADD `*_target_pos_est_w(3)` (+ validity/age, + range-uncertainty when available) and KEEP the
existing `*_ray_w` bearing** — do NOT replace it. Rationale (deploy-driven, supersedes the earlier
dim-preserving "swap"):
- The **bearing is the reliable deploy signal** (direct detector ray, low-assumption, always
  available); the **point is the uncertain one** (only as good as r_i, which is the weak part at
  deploy). Replacing would discard the robust signal and keep only the range-contaminated one — if
  deploy range degrades, the policy loses the clean bearing fallback.
- A robust policy wants **both**: bearing for *direction* (trust always) + point/uncertainty for
  *where* (trust per confidence). Pre-composing the point does the fusion the RNN failed at, while the
  bearing stays as the dependable direction.
- Aligns with the EKF future (option D): a bearing-only estimator emits position+covariance; the policy
  should see the raw bearing AND the estimate.
- Enables **separable ablations**: mask the point (range info) vs mask all peer target info.

Layout: **append** the new per-agent estimate channels at the END of the obs vector (old indices
preserved). The lost agent recovers via `normalize(other_target_pos_est_w − ego_pos)`.

**Cost — warm-start obs surgery returns** (obs_dim grows): load t048 weights, copy old first-layer
input columns, zero-init the new ones (new dims contribute 0 at init ⇒ behavior ≡ t048), expand the
skrl obs preprocessor (mean 0 / var 1 for new dims). One-time, testable utility; cascade obs_dim ->
state_space (the finalize_* method already handles sizing).

## THE central design decision (Stage Q) — the range source r_i  [REALIZABILITY]
The whole arc's lesson: a reward/obs signal must be **realizable from non-GT information**. `p_i_est`
MUST NOT be built from `_target_pos_w` (GT) — that would leak the answer into obs and make the
ablation meaningless. r_i options, to be chosen with the engineer (deploy-faithfulness vs simplicity):

| option | r_i source | realizable at deploy? | status |
|---|---|---|---|
| **A** | per-agent **bbox-raycaster depth** (`_depths`) | depth/RGBD or known-size monocular | **CHOSEN (2026-06-18)** — hypothesis test. Sim depth is true range (SENSOR-class). Cleanest test of "does a usable point enable cooperation?"; deploy range realizability flagged for the bag. |
| B | **carry-forward last team triangulation** through the gap | yes (obs-derived) | FUTURE (post-A). Belief-through-the-gap (Dec-POMDP); stale but peer ray refines. |
| C | monocular **size→range** | yes if target size known | FUTURE (post-A). Noisiest; closest to a raw monocular detector. |
| **D** | **EKF bearing-only state estimate** (recursive: bearings + motion model → position + covariance) | **yes — no depth sensor needed** | FUTURE (post-A), the **deploy-faithful** successor. Emits point + uncertainty natively; pairs with keeping the raw bearing. The principled deploy version of A. |

Decision: **A now** (validate the cooperation hypothesis with a clean range), then **D** as the
deploy-faithful estimator (B/C as alternatives). If A is GO but GT-range-dependent, re-validate under
D/B/C before any sim2real claim.

Add per estimate: a **validity flag** (reuse `bbox_empty`) and optionally **age** so the policy can
discount a stale peer point. Keep z handling consistent with the env's wxyz/world conventions.

## Reward / scenario
Reward = **C2 baseline** (`information_reward.enabled=False`, `reacq_shaping.enabled=False`) — both
reward-axis interventions failed; isolate the obs channel. Scenario + cooperation_metrics ON (reused).

## Ablation (the paper claim)
`peer_bearing_ablate`-style flag that masks (zeros) the new `*_target_pos_est_w` slots — and,
separately, the bearing slots — for clean attribution (range/point vs all-peer-target-info). Masking =
zeroing ⇒ single-policy on/off eval works (obs dim unchanged at eval); add a trained-without arm for
the stronger claim. Expectation now: masking the point **degrades** re-acquisition (the channel is
finally usable) — the opposite of Slice C.

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
