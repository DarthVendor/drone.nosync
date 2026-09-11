#!/usr/bin/env bash
# Start a composer training run.  This is the configuration that was VERIFIED
# to learn on Sept 10 2026 -- see "What working looks like" below, and do not
# change the pinned settings without re-checking against it.
#
# What was wrong before, and what fixes it
# ----------------------------------------
# `stop_quantile` was 0.9.  `rollout.py` reads that as "end the batch once this
# fraction of episodes has ARRIVED", and scores every flight still in the air as
# a failure -- so the reported arrival rate was capped near 0.90 no matter how
# good the policy got.  The curriculum gate wanted 0.95 from that bounded
# number, so difficulty could never leave 0% buildings and the composer spent
# its whole life on an empty map the low level already solves at reach 1.000.
# `finish_frac` is 1.0 for those same truncated rows, so the imitation filter
# was also told ~8% of the batch had failed while it was still flying.
#
# Measured: 0.908 +- 0.013 over thirteen iterations at q=0.9, against 0.988 on
# the identical config with the early exit off.  cotrain_v11.py pins q=1.0,
# where the exit is exact (everyone arrived or died) and still skips the dead
# tail.  tests/test_stop_quantile_exit.py pins it so it cannot come back.
#
# Usage
# -----
#   ./train.sh                          # control arm, 400 iterations
#   ARM=k4s ITERS=800 ./train.sh        # 4 samples per task, signed weights
#   WORK=/some/dir ./train.sh           # somewhere other than the default
#   GENOME=/path/to/genome.json ./train.sh
#
# Arms (the 2x2; see the block comment in cotrain_v11.py)
#   k1u  1 sample per task, unsigned  -- the control
#   k4u  4 samples per task, unsigned -- per-task baseline, imitation only
#   k1s  1 sample per task, signed    -- no baseline, failures push down
#   k4s  4 samples per task, signed   -- both
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$REPO/../.venv/bin/python}"

ARM="${ARM:-k1u}"
ITERS="${ITERS:-400}"
WORK="${WORK:-$HOME/drone_runs/$(date +%Y%m%d)}"
GENOME="${GENOME:-$HERE/frozen_low_level.json}"

case "$ARM" in
  k1u) REPEAT=1; SIGNED=0 ;;
  k4u) REPEAT=4; SIGNED=0 ;;
  k1s) REPEAT=1; SIGNED=1 ;;
  k4s) REPEAT=4; SIGNED=1 ;;
  *) echo "unknown arm '$ARM' (want k1u, k4u, k1s or k4s)" >&2; exit 2 ;;
esac

[ -x "$PY" ] || { echo "no interpreter at $PY -- set PY=..." >&2; exit 2; }
[ -f "$GENOME" ] || { echo "no frozen low level at $GENOME -- set GENOME=..." >&2; exit 2; }

mkdir -p "$WORK"
# cotrain_v11.py reads its own scripts out of the work directory and warm-starts
# the FROZEN low level from cotrain_v10_genome.json there (G0).  The low level
# is frozen for the whole run: SIGMA=0 and selection is skipped, so this file is
# read and never written.
cp -f "$HERE/cotrain_v11.py" "$HERE/cotrain_v9.py" "$WORK/"
cp -f "$GENOME" "$WORK/cotrain_v10_genome.json"

LOG="$WORK/v11_${ARM}.out"
PIDF="$WORK/v11_${ARM}.pid"

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "arm $ARM is already running (pid $(cat "$PIDF")); stop it first" >&2
  exit 1
fi

# A fresh net for a fresh arm.  Leave these in place to RESUME instead: the
# trainer picks up the composer weights, the genome and the curriculum state.
if [ "${FRESH:-1}" = "1" ]; then
  rm -f "$WORK/v11_${ARM}_composer.pt" "$WORK/v11_${ARM}_composer_best.pt" \
        "$WORK/v11_${ARM}_state.json" "$WORK/v11_${ARM}_genome.json"
fi

# `caffeinate -i` because the Mac sleeps mid-training and a 100+ minute gap
# between iterations reads as a hang when it is just a nap.
cd "$WORK"
LES_REPEAT="$REPEAT" LES_SIGNED="$SIGNED" LES_TAU="${TAU:-1.0}" \
  nohup caffeinate -i "$PY" "$WORK/cotrain_v11.py" "$WORK" "$ITERS" \
        0.05 100 3.0 40.0 2e-4 > "$LOG" 2>&1 &
echo $! > "$PIDF"

cat <<EOF
started arm $ARM (repeat $REPEAT, signed $SIGNED) for $ITERS iterations
  pid   $(cat "$PIDF")
  work  $WORK
  log   $LOG

  tail -f $LOG

What working looks like (verified Sept 10 2026, arm k1u, empty map):

  iter 1  arrive 0.865  t_arr 0.624  cost 189.0
  iter 4  arrive 0.946  t_arr 0.571  cost 148.0
  iter 5  arrive 0.957  t_arr 0.539  cost 131.2   <- crosses the 0.95 gate
  iter 6  arrive 0.958  t_arr 0.521  cost 135.7

  * arrive, t_arr and cost all improve TOGETHER.  If cost falls while arrive
    stays flat near 0.90, the early-exit truncation is back -- check
    stop_quantile.
  * "weighted ess N" must be well BELOW the kept-flight count (~0.4 of it).
    ess equal to the count means the arrival-time weights collapsed to uniform
    and nothing is being selected.
  * "buildings" must leave 0% within ~10 iterations.  Stuck at 0% is the
    signature of a curriculum bar the measurement cannot reach.
EOF
