#!/usr/bin/env bash
# Start the MSE (time-regression) composer run.  This is the configuration
# arrived at on Sept 11 2026 after the session below; the reasoning matters
# because most of these settings exist to close a specific measured failure.
#
# THE LOSS.  LES_LOSS=time regresses T_hat(state, placement) against the time
# the flight actually needed, then descends the model:
#     L_model  = (T_hat(s, g_emitted) - T)^2
#     L_policy = T_hat(s, g(theta))
# Chosen over the paired counterfactual because, MEASURED with matched
# controls, a waypoint barely moves a flight -- arrive 0.685 against its own
# control's 0.678, one rescue in 256 -- so 99.6% of flights tied at advantage
# zero and were discarded.  Regression keeps every flight (a placement that
# changed nothing still observed a real time) and yields dT_hat/dg, a DIRECTION
# in placement space rather than one scalar per token.
#   Watch `match`: it is the SIGNED model error, and a negative drift means the
#   policy has found placements where T_hat is optimistically wrong and is
#   exploiting the model rather than flying better.  It has happened here
#   before (predictions to -0.4 while the fit error rose 0.98 -> 1.21), which
#   is why the head is sigmoid-bounded.
#
# THE RUNG.  20 m legs / 25% buildings, pinned.  Building density is NOT a
# difficulty knob for this frozen low level -- it flies 8 m legs at ~0.99 at
# 10%, 25% AND 50% -- so a composer trained there has nothing to contribute and
# correctly learns to stay silent.  Leg length is the knob: the low level alone
# drops to 0.531 at 20 m.  Pinned so the curriculum cannot promote itself back
# onto a rung where it measures nothing.
#
# THE VOCABULARY is EOS + WAYPOINT only (N_TOKENS=2).  TURN, PRIORITY and LOOK
# are each an exact no-op at a zero argument, and a no-op is an absorbing
# state: it produces a flight identical to its control, earns advantage zero,
# and is DROPPED from the batch rather than penalised -- so it can never be
# pushed back down.  The policy found all three in turn (LOOK 78%, then
# heading 100%).
#
# LES_SPEAK=0.1 forces a WAYPOINT on a tenth of decisions.  Without it the
# policy goes fully silent, and silence IS the control, so every advantage is
# zero and the loss is nan -- a run died exactly that way when this was
# omitted from the command line.
#
# Usage
#   ./start_mse.sh                      # the pinned configuration
#   LES_EPOCHS=4 LES_INJECT=2 ./start_mse.sh    # faster iterations, less data
#   LES_EPS=576 ./start_mse.sh          # smaller batch
#   WORK=/some/dir ./start_mse.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LES_LOSS="${LES_LOSS:-time}"      # MSE time regression
export LES_SPEAK="${LES_SPEAK:-0.1}"     # forced WAYPOINT rate; 0 lets it go silent
export LES_INJECT="${LES_INJECT:-4}"     # decisions per flight
export LES_LEG="${LES_LEG:-20}"          # 20 m legs: where the low level fails
export LES_DIFF="${LES_DIFF:-0.25}"      # 25% buildings
# EPOCHS IS NOT THE UNIT -- steps are: epochs * ceil(n_samples / LES_MB).  With
# MB 1024 and ~3300 samples that is 4 steps an epoch, so 3 epochs ~ 12 steps.
# Leaving this at 10 alongside the smaller minibatch took the update 130 s ->
# 605 s (40 steps).  Raise LES_MB with it if you want more epochs.
export LES_EPOCHS="${LES_EPOCHS:-3}"
export LES_EPS="${LES_EPS:-1152}"        # episodes per iteration
export LES_VART="${LES_VART:-2.0}"       # variational waypoint temperature; 0 = off
# PERFORMANCE.  Measured cost to push 4096 samples through one fwd+bwd:
#   mb 4096  81.3 s  |  mb 2048  18.9 s  |  mb 1024  10.2 s  |  mb 512  11.4 s
# Superlinear above ~1024, so the big minibatch cost 8x and bought nothing --
# and a minibatch larger than the batch meant ONE optimizer step per epoch,
# when steps were the scarce resource.  k_chain is quadratic in the first
# chain block's causal self-attention (the last block already computes only
# the final position): 128 -> 10.2 s, 64 -> 6.1 s, 32 -> 5.1 s at mb 1024.
export LES_MB="${LES_MB:-1024}"          # update minibatch
export LES_KCHAIN="${LES_KCHAIN:-64}"    # chain memory in events (~21 s of a 36 s flight)

WORK="${WORK:-$HOME/drone_runs/mse_$(date +%H%M)}"
ARM="${ARM:-k1u}" ITERS="${ITERS:-400}" WORK="$WORK" exec "$HERE/train.sh"
