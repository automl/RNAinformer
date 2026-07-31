#!/bin/bash
# Adapter: reorganize the human-readable predictions/ browse layout into the runs/ layout
# that RNAinformer's run_evaluations.sh + eval.py expect. Symlinks only (no copies).
# Usage:  bash adapt_to_eval_layout.sh <predictions_dir> <output_runs_dir>
#   then: cp -r <datasets_dir> data ; bash run_evaluations.sh   (from the repo)
set -e
PRED="${1:-predictions}"; RUNS="${2:-runs}"
rm -rf "$RUNS"; mkdir -p "$RUNS"

# ---- RNAinformer models ----
for m in "$PRED"/rnainformer/*/; do
  model=$(basename "$m")
  gc=0; case "$model" in *_gc) gc=1;; esac
  for v in "$m"version_*/; do
    [ -d "$v" ] || continue
    ver=$(basename "$v")
    # riboswitch uses predictions_20/, others predictions/
    case "$model" in ribo*) sub="predictions_20";; *) sub="predictions";; esac
    if [ "$gc" = 1 ]; then dst="$RUNS/$model/$ver/$sub/gc"; else dst="$RUNS/$model/$ver/$sub"; fi
    mkdir -p "$dst"
    for f in "$v"*.plk.gz; do [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$dst/$(basename "$f")"; done
  done
done

# ---- baselines: learna_suite + samfeo ----
for b in learna meta_learna meta_learna_adapt liblearna liblearna_gc; do
  s="$PRED/baselines/$b"; [ -d "$s" ] || continue
  mkdir -p "$RUNS/learna_suite/$b"
  for f in "$s"/*.plk.gz; do [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$RUNS/learna_suite/$b/$(basename "$f")"; done
done
for b in samfeo antarna antarna2; do
  s="$PRED/baselines/$b"; [ -d "$s" ] || continue
  mkdir -p "$RUNS/$b"
  for f in "$s"/*.plk.gz "$s"/*.plk; do [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$RUNS/$b/$(basename "$f")"; done
done

echo "adapter done -> $RUNS/  (RNAinformer models + learna_suite + samfeo/antarna)"
echo "next: put datasets as ./data , then run:  bash run_evaluations.sh"
