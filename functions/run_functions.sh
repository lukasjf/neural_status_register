#!/bin/bash

#SBATCH --mail-type=NONE                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%j.err                  # where to store error messages
#SBATCH --cpus-per-task=4
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=2nn --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=2nn --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=2nn --alumodel=nau --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=2nn --alumodel=nau --state --epochs=500000

sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=nalu --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=nalu --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=nalu --alumodel=nau --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=nalu --alumodel=nau --state --epochs=500000

sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=nau --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=nau --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=nau --alumodel=nau --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=nau --alumodel=nau --state --epochs=500000

sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=nsr --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=nsr --alumodel=nalu --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=f --model=nsr --alumodel=nau --state --epochs=500000
sbatch --cpus-per-task=8 run.sh functions/functions.py --task=feq --model=nsr --alumodel=nau --state --epochs=500000
