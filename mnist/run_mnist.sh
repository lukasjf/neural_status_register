#!/bin/bash

sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=0
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=1
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=2
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=3
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=4
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=5
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=6
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=7
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=8
sbatch --mem=8G --nodelist=tikgpu06 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=2nn --seed=9

sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=0
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=1
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=2
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=3
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=4
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=5
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=6
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=7
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=8
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nalu --seed=9

sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=0
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=1
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=2
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=3
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=4
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=5
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=6
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=7
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=8
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nau --seed=9


sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=0
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=1
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=2
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=3
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=4
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=5
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=6
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=7
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=8
sbatch --mem=8G --nodelist=tikgpu09 --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=gt --epochs=13 --batch-size=50 --model=nsr --seed=9


sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=0
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=1
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=2
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=3
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=4
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=5
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=6
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=7
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=8
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=2nn --seed=9

sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=0
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=1
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=2
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=3
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=4
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=5
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=6
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=7
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=8
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nalu --seed=9

sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=0
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=1
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=2
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=3
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=4
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=5
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=6
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=7
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=8
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nau --seed=9

sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=0
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=1
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=2
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=3
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=4
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=5
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=6
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=7
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=8
sbatch --mem=16G --cpus-per-task=8 run.sh mnist/mnist_comparison.py --comparison=eq --epochs=13 --batch-size=100 --model=nsr --seed=9

