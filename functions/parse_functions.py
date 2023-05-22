
import argparse
from collections import defaultdict
import numpy as np
from ast import literal_eval as evallist
import os

def tolist(liststring):
    lst = evallist(liststring)
    return np.array(lst)

firstk = 7
seeds = 10

fcts = ["f", "feq"]
models = ["2nn", "nalu", "nau", "nsr"]
alumodels = ["nalu", "nau"]

results = {m: {am: {f: [] for f in fcts} for am in alumodels} for m in models}
for m in models:
    for am in alumodels:
        for f in fcts:
            for _ in range(firstk):
                results[m][am][f].append([])
seen = {m: {am: {f:[False] * seeds for f in fcts} for am in alumodels} for m in models}

path = "functions/results/"
for file in os.listdir(path):
    if not os.path.isfile(path + file):
        continue
    for line in open(path+file, "r").readlines():
        print(line)
        task, seed, redundancy, model, alumodel, scores = line.split(";")
        if not seen[model][alumodel][task][int(seed)]:
            for i, score in enumerate(tolist(scores)[:firstk]):
                results[model][alumodel][task][i].append(score)
            seen[model][alumodel][task][int(seed)] = True
        else:
            print("duplicate {} {} {} {}".format(model, alumodel, task, seed))

for m in seen:
    for am in seen[m]:
        for f in seen[m][am]:
            for i,seed in enumerate(seen[m][am][f]):
                if not seed:
                    print("sbatch --cpus-per-task=2 run_comparisons.sh comparisons.py --outfile fcts "
                          " --task {} --model {} --seed {}".format(f, m, i))

#1/0

scores = {task:{m: {am: [] for am in alumodels} for m in models} for task in fcts}

for task in fcts:
    for model in models:
        for alumodel in alumodels:
            for i in range(firstk):
                mean = round(np.mean(results[model][alumodel][task][i]), 2)
                std = round(np.std(results[model][alumodel][task][i]), 2)
                string = "\\makebox{{{:.2f} \\rpm {:.2f}}}".format(float(mean), float(std))
                scores[task][model][alumodel].append(string)

for task in fcts:
    print("-----------------------------")
    for model in models:
        for alu in alumodels:
            print("    & {{{}}} & {}\\\\"
                  .format(model+"-"+alu, " & ".join(scores[task][model][alu])))
1/0

for task in fcts:
    print("    \multirow{{2}}{{*}}{{{}}} & {} & {}\\\\"
      .format(task, "NALU-nalu", " & ".join(map(str, np.round(results["nalu"]["nalu"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nalu-nau", " & ".join(map(str, np.round(results["nalu"]["nau"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nau-nalu", " & ".join(map(str, np.round(results["nau"]["nalu"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nau-nau", " & ".join(map(str, np.round(results["nau"]["nau"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nsr-nalu", " & ".join(map(str, np.round(results["nsr"]["nalu"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nsr-nau", " & ".join(map(str, np.round(results["nsr"]["nau"][task] / seeds, 2)))))
    print("    \midrule")


for task in fcts:
    print("    \multirow{{2}}{{*}}{{{}}} & {} & {}\\\\"
      .format(task, "NALU-nalu", " & ".join(map(str, np.round(results["nalu"]["nalu"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nalu-nau", " & ".join(map(str, np.round(results["nalu"]["nau"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nau-nalu", " & ".join(map(str, np.round(results["nau"]["nalu"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nau-nau", " & ".join(map(str, np.round(results["nau"]["nau"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nsr-nalu", " & ".join(map(str, np.round(results["nsr"]["nalu"][task] / seeds, 2)))))
    print("    & {} & {}\\\\"
          .format("nsr-nau", " & ".join(map(str, np.round(results["nsr"]["nau"][task] / seeds, 2)))))
    print("    \midrule")
