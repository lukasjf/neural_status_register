#!/bin/bash

for scale in 0.001 0.01 0.1 1.0 10 100 1000 10000 100000; do
for zeta in 0.001 0.01 0.1 1.0 10 100 1000; do
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=gt --model=nsr --scale=$scale --zeta=$zeta --prefix="scaling/"
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=eq --model=nsr --scale=$scale --zeta=$zeta --prefix="scaling/"
done
done
