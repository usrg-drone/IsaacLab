# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gated potential-based recovery-shaping reward (ticket 050, Slice C).

Densifies the per-agent credit for re-pointing at the target during a single-agent deficit — the
sparse/delayed-credit gap Slice B left. See reacq_shaping.py for the design.
"""

from .reacq_shaping import ReacqShaper
from .reacq_shaping_cfg import ReacqShapingCfg

__all__ = ["ReacqShaper", "ReacqShapingCfg"]
