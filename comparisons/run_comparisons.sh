#!/bin/bash

sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=gt --model=2nn --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=geq --model=2nn --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=eq --model=2nn --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=ne --model=2nn --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=leq --model=2nn --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=lt --model=2nn --prefix=""

sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=gt --model=npu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=geq --model=npu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=eq --model=npu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=ne --model=npu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=leq --model=npu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=lt --model=npu --prefix=""

sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=gt --model=nalu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=geq --model=nalu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=eq --model=nalu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=ne --model=nalu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=leq --model=nalu --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=lt --model=nalu --prefix=""

sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=gt --model=nau --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=geq --model=nau --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=eq --model=nau --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=ne --model=nau --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=leq --model=nau --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=lt --model=nau --prefix=""

sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=gt --model=nsr --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=geq --model=nsr --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=eq --model=nsr --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=ne --model=nsr --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=leq --model=nsr --prefix=""
sbatch --cpus-per-task=2 run.sh comparisons/comparisons.py --task=lt --model=nsr --prefix=""
