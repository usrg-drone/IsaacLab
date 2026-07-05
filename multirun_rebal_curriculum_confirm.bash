#!/bin/bash
# (b) From-scratch confirmation of the reward-tuning winner via a bbox-rebalance CURRICULUM (not a
# constant reward). Bootstrap tracking with bbox_center=90 early, then ramp 90->30 over
# reward_rebalance_{start,end}_step (80k-200k), holding bbox_size=30 and triangulation=20. End state =
# rebal_strong (30/30/20), which failed from scratch as a CONSTANT reward (0.418) because bbox=30 from
# step 0 starves the tracking bootstrap. Baseline env otherwise (ticket-050 features OFF). 400k each,
# sequential. First (seed 42) result gates whether the curriculum recovers the warm-start ~0.602.
#
# Usage:  bash multirun_rebal_curriculum_confirm.bash --dry_run
#         tmux new -d -s rcconfirm "bash multirun_rebal_curriculum_confirm.bash"

set -u
cd /home/usrg/IsaacPX4/IsaacLab
TRAIN="./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"
TASK="Isaac-Iris-MA6-Direct-Test-v0"
TIMESTEPS=400000
SEEDS=(42 123 7)

DRY=false; [[ "${1:-}" == "--dry_run" ]] && DRY=true
i=0; N=${#SEEDS[@]}
echo "=== bbox-rebalance-curriculum confirm: bbox_center 90->30 ramp, bbox_size=30, tri=20; ${TIMESTEPS} steps; seeds ${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do
  i=$((i+1))
  CMD="$TRAIN --task $TASK --headless --seed $s --experiment_name rcconfirm_rebalcurric_seed${s} \
    --timesteps $TIMESTEPS \
    env.enable_bbox_rebalance_curriculum=True \
    env.bbox_size_reward_scale_team=30.0 \
    env.triangulation_reward_scale=20.0"
  echo ""; echo "[$i/$N] rcconfirm seed=$s (from scratch, bbox-rebalance curriculum)"
  if $DRY; then echo "  CMD: $CMD"; else
    eval "$CMD 2>&1 | tee /tmp/rcconfirm_seed${s}.log"
    echo "[$i/$N] done seed $s ($(date))"; sleep 5
  fi
done
echo "=== ALL DONE — eval logs/skrl/iris_ma6/*rcconfirm_* vs constant-30/30/20 from-scratch (0.418) & warm-start (0.602) ==="
