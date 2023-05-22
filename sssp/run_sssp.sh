#!/bin/bash
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 0
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 1
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 2
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 3
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 4
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 5
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 6
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 7
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 8
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model nsr --epochs 1000 --seed 9
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 0
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 1
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 2
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 3
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 4
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 5
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 6
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 7
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 8
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model neg --epochs 1000 --seed 9
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 0
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 1
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 2
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 3
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 4
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 5
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 6
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 7
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 8
sbatch --cpus-per-task=8 run.sh sssp/sssp_tf.py --model itergnn --epochs 1000 --seed 9



sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 0
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 1
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 2
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 3
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 4
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 5
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 6
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 7
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 8
sbatch --cpus-per-task=8 run.sh sssp/sssp.py --model nsr --epochs 1000 --seed 9
