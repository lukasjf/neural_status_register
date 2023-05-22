#!/bin/bash

sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=lstm --task=rmin
sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=lstm --task=rcount

sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=2nn --task=rmin
sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=2nn --task=rcount

sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=nalu --task=rmin
sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=nalu --task=rcount

sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=nau --task=rmin
sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=nau --task=rcount

sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=nsr --task=rmin
sbatch --cpus-per-task=2 run.sh recurrent/recurrent.py --model=nsr --task=rcount
