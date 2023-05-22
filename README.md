This files constains code for the paper "Neural Status Registers"

All files needed are available in this folder

1. You can install the dependencies with conda from the file nsr.yml
    conda env create -f nsr.yaml


2. The experiments are structure in different folders. Every folder contains shell files to run all
related experiments. If you use SLURM, these files can launch parallel jobs for you. If you
 want to run direct Python code, you can replace the part up to ``run.sh'' with just python.

 For example you can run ``` python comparisons/comparisons.py --task=gt --model=nsr --prefix=""```
 to train the NSR on learning the > comparison

Experiments write in a results/ subfolder. In each directory we also include
parse_* files to parse the result files into Latex code for tables and/or plots.

We give a quick overview for every directory:
* Comparisons: Contains code for the initial table for just learning comparisons and the
  ablation study to trade of the resolution limit delta versus the scaling factor lambda.
* Functions: Contains the two experiments for the piecewise-defined functions (normal experiment
  and the ablation study on redundancy levels)
* Mnist: Contains the code for the digit image comparison task
* Recurrent: Contains the code for the recurrent experiments min and count
* Sssp: Contains the code for running the shortest path experiments

The top folder contains the line plotsignzero which creates Figures 2, 3, and 7.


## Citation

Please consider citing the paper if you find it or its code useful. For example with the following Bibentry:
```latex
@inproceedings{faber2023neural,
  title={Neural Status Registers},
  author={Faber, Lukas and Wattenhofer, Roger},
  booktile={International Conference on Machine Learning (ICML), Honolulu, USA},
  year={2023}
```
