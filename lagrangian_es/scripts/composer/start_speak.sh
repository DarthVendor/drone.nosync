#!/usr/bin/env bash
# The SIMPLE composer run: a 2-armed contextual bandit.
#
# WHY THIS IS THE WHOLE MODEL.  Measured Sept 11 2026 on 256 paired tasks with
# identical seeds, a WAYPOINT forced at full reach on every decision:
#
#     muted (no tokens at all)      arrive 0.539
#     forced WAYPOINT   +90 deg     arrive 0.281
#     forced WAYPOINT  +180 deg     arrive 0.285
#     forced WAYPOINT    0 deg      arrive 0.285
#
# Emitting a waypoint costs a QUARTER of the arrival rate, and three bearings
# spanning 180 degrees do identical damage to three decimal places.  So the
# placement carries no signal and the decision to speak carries all of it: the
# action space is binary, and the learner is one log-probability times one
# paired advantage.  No critic, no time model, no argument density.
#
# WHAT THIS REPLACES.  LES_LOSS=time fit T_hat(s,g) and descended it, but `g`
# is built from the WAYPOINT ARGUMENTS -- the type logits appear in neither of
# its terms, so it had no gradient path to the speak/EOS decision at all.  Over
# 23 iterations `nll` fell 0.113 -> 0.047 while `speak` sat at 0.509-0.515 and
# `arrive` never left the sampling floor (observed sd 0.0155 vs a binomial
# 0.0133 at 1152 episodes; trend +0.0004 +- 0.0005, t = 0.79).  It converged by
# iteration 11 and then reproduced itself for 12 more.
#
# WHAT SUCCESS LOOKS LIKE.  `speak` must MOVE.  If emitting is uniformly bad it
# should fall toward the 0.1 exploration floor and `arrive` climb toward the
# muted 0.539; if it is contextually useful the bandit keeps it where it pays.
# Either way this is the first objective in the project that can reach the
# decision, so a flat `speak` now means a real bug, not a weak signal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LES_LOSS="${LES_LOSS:-speak}"
export LES_INJECT="${LES_INJECT:-1}"   # ONE decision a flight: the whole paired
                                       # outcome attributes to that single token
export LES_SPEAK="${LES_SPEAK:-0.1}"   # exploration floor.  A forced WAYPOINT is
                                       # off-policy for REINFORCE, but it reports
                                       # its real advantage back, so the bias is
                                       # in magnitude only -- the sign is right.
export LES_VART="${LES_VART:-0}"       # the variational sampler picks WHERE; where
                                       # does not matter, so it is pure cost now
export LES_LEG="${LES_LEG:-20}"        # 20 m legs: where the low level fails
export LES_DIFF="${LES_DIFF:-0.25}"
export LES_EPS="${LES_EPS:-1152}"
export LES_EPOCHS="${LES_EPOCHS:-3}"
export LES_MB="${LES_MB:-1024}"
export LES_KCHAIN="${LES_KCHAIN:-64}"
# NOTE: a paired loss flies the batch TWICE (injected + muted control), so the
# rollout costs about double what LES_LOSS=time did.  That control IS the value
# baseline -- it is why this learner needs no critic.

WORK="${WORK:-$HOME/drone_runs/speak_$(date +%H%M)}"
ARM="${ARM:-k1u}" ITERS="${ITERS:-400}" WORK="$WORK" exec "$HERE/train.sh"
