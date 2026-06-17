# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the gated recovery-shaping reward (ticket 050, Slice C)."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ReacqShapingCfg:
    """Configuration for :class:`ReacqShaper`.

    A potential-based shaping reward (Ng et al. 1999) on the agent's *pointing alignment* to the
    target, **gated to the single-agent-deficit regime** so it is realizable from observation (the
    peer's measured bearing ``other_ray_w`` is in the obs exactly when a peer holds the target). It
    densifies the otherwise sparse/delayed credit for re-pointing during a deficit — the gap Slice B
    left. Reward-side only; uses the privileged GT target (never enters obs).
    """

    enabled: bool = False
    """Whether the recovery shaping is active. Default False = bit-exact baseline."""

    shaping_scale: float = 10.0
    """Scale on the per-step potential difference F = scale * (gamma*Phi' - Phi). With gamma=1 a
    successful recovery telescopes to ~scale*(Phi_recovered - Phi_entry) ~ scale; calibrate so that is
    a meaningful (non-dominant) fraction of the bbox/triangulation gained on re-acquisition
    (Phase-2 sweep, e.g. 10/20/40). NOTE: per-step term, NOT a rate*dt term — added without step_dt.
    (v1 used 2.0 with gamma=0.99 + gate-on-current-deficit; that net-NEGATIVE 'deficit tax' regressed
    tracking — see doc/experiments 2026-06-17.)"""

    gamma: float = 1.0
    """Discount in the potential difference F = gamma*Phi' - Phi. **1.0** (pure potential difference):
    holding alignment during a deficit gives F=0 (NO 'deficit tax'); only *improvement* in pointing is
    rewarded and *degradation* penalized. gamma<1 (e.g. 0.99) re-introduces a -(1-gamma)*Phi per-step
    tax on staying aimed during a deficit — the v1 failure mode; do not use unless ablating."""

    gate_to_deficit: bool = True
    """If True (default), F is nonzero only on steps where the agent was in a peer-assisted DEFICIT at
    the START of the step (ReacquisitionTracker state last step) — the realizable regime. This credits
    the entry step with 0, mid-deficit re-pointing, AND the recovery step (DEFICIT->HOLD). If False, F
    applies in every state (pure PBRS); requires nothing from the tracker (an ablation arm)."""

    potential_kernel: str = "cosine"
    """Pointing potential kernel. "cosine": Phi = 0.5*(1 + d . d*), d = camera boresight (world),
    d* = GT bearing to target. Only "cosine" is implemented; field reserved for future kernels."""
