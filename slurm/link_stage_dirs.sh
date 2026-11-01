#!/bin/bash
# Makes <cluster_root>/<stage> a link to this repository for every stage the Slurm scripts use, so that $B/<stage>/src, $B/<stage>/analysis and the other folders resolve to one tree. Run: slurm/link_stage_dirs.sh
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
root="$(cd "$repo" && python3 -c 'import sys; sys.path.insert(0, "."); from slurm import cxr_settings; print(cxr_settings.get("cluster_root"))')"
mkdir -p "$root"
stages="$(cd "$here" && for d in */; do [ "$d" = "lib/" ] || printf '%s ' "${d%/}"; done) replication_512_score replication_512_ctl224"
for stage in $stages; do
  [ -e "$root/$stage" ] || ln -s "$repo" "$root/$stage"
done
echo "linked the stage folders under $root to $repo"
