
import argparse
import numpy as np
from ast import literal_eval as evallist
import os

tasks = ["gt", "eq", "lt", "ne", "geq", "leq"]

models = ["nsr", "2nn", "nalu", "nau", "npu"]

def tolist(liststring):
    lst = evallist(liststring)
    return np.array(lst)

firstk = 7
seeds = 10

results = {m:{op:[] for op in tasks} for m in models}
for model in models:
    for task in tasks:
        for _ in range(firstk):
            results[model][task].append([])

seen = {m:{op:[False] * seeds for op in tasks} for m in models}

path = "comparisons/results/"
for file in os.listdir(path):
    if not os.path.isfile(path + file):
        continue
    for line in open(path + file, "r").readlines():
        task, seed, redundancy, scale, zeta, model, scores = line.split(";")
        if not seen[model][task][int(seed)]:
            for i, score in enumerate(tolist(scores)[:firstk]):
                results[model][task][i].append(score)
            seen[model][task][int(seed)] = True
        else:
            print("duplicate {} {} {}".format(model, task, seed))



for m in seen:
    for op in seen[m]:
        for i,seed in enumerate(seen[m][op]):
            if not seed:
                print("sbatch --cpus-per-task=2 run_comparisons.sh comparisons.py --task {} "
                "--model {} --seed {}".format(op, m, i))

#1/0

optotex = {
    "gt":"$>$",
    "lt":"$<$",
    "eq":"$=$",
    "ne":"$\\ne$",
    "geq":"$\\geq$",
    "leq":"$\\leq$",
}

print(results["2nn"]["gt"])

scores = {task:{m:[] for m in models} for task in tasks}

for task in tasks:
    for model in models:
        for i in range(firstk):
            mean = round(np.mean(results[model][task][i]), 2)
            std = round(np.std(results[model][task][i]), 2)
            string = "\\makebox{{{:.2f} \\rpm {:.2f}}}".format(float(mean), float(std))
            scores[task][model].append(string)

print(scores)

print("    {{Comparison}} & {{Model}} & {}\\\\\midrule"
      .format(" & ".join(["{$10^{" + str(i) + "}$}" for i in range(1, firstk+1)])))
print("    \multirow{{5}}{{*}}{{$>$}} & MLP & {}\\\\"
      .format(" & ".join(scores["gt"]["2nn"])))
for model in ["npu", "nalu", "nau", "nsr"]:
    print("    & {} & {}\\\\"
          .format(model, " & ".join(scores["gt"][model])))
print("    \midrule")
print("    \multirow{{5}}{{*}}{{$=$}} & MLP & {}\\\\"
      .format(" & ".join(scores["eq"]["2nn"])))
for model in ["npu", "nalu", "nau", "nsr"]:
    print("    & {} & {}\\\\"
          .format(model, " & ".join(scores["eq"][model])))
print("    \\bottomrule")

print("\n\n\n\n\n\n######################\n\n\n\n\n\n")

print("    {{Comparison}} & {{Model}} & {}\\\\\midrule"
      .format(" & ".join(["{$10^{" + str(i) + "}$}" for i in range(1, firstk+1)])))
for op in optotex:
    print("    \midrule")
    print("    \multirow{{5}}{{*}}{{{}}} & MLP & {}\\\\"
            .format(optotex[op], " & ".join(scores[op]["2nn"])))
    for model in ["npu", "nalu", "nau", "nsr"]:
        print("    & {} & {}\\\\"
              .format(model, " & ".join(scores[op][model])))
