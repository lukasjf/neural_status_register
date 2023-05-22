
import argparse
from collections import defaultdict
import os
import numpy as np
from ast import literal_eval as evallist
import seaborn
import matplotlib.pyplot as plt
#plt.rcParams["font.size"] = 13

def tolist(liststring):
    lst = evallist(liststring)
    return np.array(lst)

num_nodes = 7
num_weights = 7
seeds = 10

models = ["nsr", "itergnn", "neg"]

nodes = [10, 20, 40, 80, 160, 320, 640]
weights = [10, 20, 40, 80, 160, 320, 640]


results = {m: np.zeros((len(nodes), len(weights))) for m in models}
seen = {m: np.zeros(seeds) for m in models}

path = "sssp/results/"
for file in os.listdir(path):
    if not os.path.isfile(path + file):
        continue
    print(file)
    for line in open(path + file, "r").readlines():
        model, node, weight, seed, error = line.split(",")
        node, weight, seed = int(node), int(weight), int(seed)
        print(line, node)
        if seen[model][seed] < num_nodes * num_weights:
            results[model][nodes.index(node), weights.index(weight)] += float(error)
            seen[model][seed] += 1
        else:
            print("duplicate")

print(seen)

for m in models:
    for k in range(seeds):
        if seen[m][k] < num_nodes * num_weights:
            print("sbatch --cpus-per-task=8 run_comparisons.sh trainsssp.py --model {} --epochs 1000 --seed {}".format(m, k))


fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
fig.set_size_inches(6, 2.1)
(a0, a1, a2) = axs.flat
cbar_ax = fig.add_axes([.91, .3, .03, .4])

colormap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
s = seaborn.heatmap(results["neg"]/seeds, ax=a0, cbar_ax=cbar_ax, cmap=colormap,
                    vmin=0, vmax=5, xticklabels=weights, yticklabels=nodes, rasterized=True,
                    square=True)
s.set(xlabel="Weight Scale", ylabel="Graph Scale")
a0.set_title("NEG")
s = seaborn.heatmap(results["itergnn"]/seeds, ax=a1, cbar_ax=cbar_ax, cmap=colormap,
                    vmin=0, vmax=5, xticklabels=weights, yticklabels=nodes, rasterized=True,
                    square=True)
s.set(xlabel="Weight Scale")
a1.set_title("IterGNN")
s = seaborn.heatmap(results["nsr"]/seeds, ax=a2, cbar_ax=cbar_ax, cmap=colormap,
                    vmin=0, vmax=5, xticklabels=weights, yticklabels=nodes, rasterized=True,
                    square=True)
s.set(xlabel="Weight Scale")
a2.set_title("NSR-GNN")
plt.savefig("plots/sssp.pdf")
plt.show()

