#!/bin/bash

for red in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
for task in f feq; do
echo $task $red
sbatch --cpus-per-task=4 run.sh functions/functions.py --task=$task --model=nsr --redundancy=$red --alumodel=nalu --epochs=500000 --prefix="redundancy/"
done
done
