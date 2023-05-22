import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
import argparse
from collections import defaultdict
import numpy as np
from ast import literal_eval as evallist
import os

def tolist(liststring):
    lst = evallist(liststring)
    return np.array(lst)

ops = ["f", "feq"]
maxbits = 15
parallel_bits = np.arange(maxbits)+1
seeds = 10

results = {o:{p:0 for p in parallel_bits} for o in ops}
seen = {o:{p:[False] * seeds for p in parallel_bits} for o in ops}

path = "functions/results/redundancy/"
for file in os.listdir(path):
    if not os.path.isfile(path + file):
        continue
    for line in open(path+file, "r").readlines():
        task, seed, redundancy, model, alumodel, scores = line.split(";")
        if int(redundancy) > maxbits:
            continue
        results[task][int(redundancy)] += tolist(scores)[0]
        if not seen[task][int(redundancy)][int(seed)]:
            seen[task][int(redundancy)][int(seed)] = True
        else:
            print("duplicate {} {} {}".format(model, task, seed))

for o in ops:
    for p in parallel_bits:
        for s in range(seeds):
            if not seen[o][p][s]:
                print("sbatch --cpus-per-task=2 run_comparisons.sh comparisons.py --model nsr --outfile lottery "
                      "--task {} --redundancy {} --seed {}".format(o, p, s))

#1/0

optotex = {
    "f":"$f$",
    "feq": "$g$",
}

print(results)

styles = ["solid", "solid", "dashed", "dashed"]
plt.figure(figsize=(5,2))
x = np.arange(maxbits) + 1
for o, op in enumerate(ops):
    y = np.array([results[op][i]/seeds for i in range(1,maxbits+1)])
    y = [y[i] for i in range(maxbits)]
    plt.plot(x, y, label=optotex[op], ls=styles[o])
plt.legend()
plt.xticks(x)
plt.yticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.xlabel("$Redundancy$")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("plots/ablation_redundancy.pdf")
plt.show()   
