
import argparse
from collections import defaultdict
import numpy as np
from ast import literal_eval as evallist
import seaborn
import matplotlib.pyplot as plt
import os

def tolist(liststring):
    lst = evallist(liststring)
    return np.array(lst)


plt.rcParams["font.size"] = 13
seeds = 10
#firstk = 12
comps = ["gt", "eq"]#["gt", "leq", "lt", "geq", "eq", "ne"]

lambdas = [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
lambdas.reverse()
deltas = ["0.001", "0.01", "0.1", "1.0", "10.0", "100.0", "1000.0", "10000.0", "100000.0"]
#deltas.reverse()

results = {comp: {delta:{lamb: 0 for lamb in lambdas} for delta in deltas} for comp in comps}

seen = {comp: {delta: {lamb: [False] * seeds for lamb in lambdas} for delta in deltas} for comp in comps}

path = "comparisons/results/scaling/"
for file in os.listdir(path):
    if not os.path.isfile(path + file):
        continue
    for line in open(path + file, "r").readlines():
        task, seed, redundancy, delta, lamb, model, scores = line.split(";")
        if model != "nsr":
            continue
        results[task][delta][float(lamb)] += tolist(scores)[0]
        if not seen[task][delta][float(lamb)][int(seed)]:
            seen[task][delta][float(lamb)][int(seed)] = True
        else:
            print("duplicate {} {} {} {}".format(task, delta, lamb, seed))


for op in seen:
    for delta in seen[op]:
        for lamb in seen[op][delta]:
            for i,seed in enumerate(seen[op][delta][lamb]):
                if not seed:
                    print("sbatch --cpus-per-task=2 run_comparisons.sh comparisons.py --outfile scaling "
                          "--task {} --scale {} --zeta {} --seed {}".format(op, delta, lamb, i))

deltatotex = {
    "1.0":"$10^0$",
    "0.1":"$10^{-1}$",
    "0.01":"$10^{-2}$",
    "0.001":"$10^{-3}$",
    "10.0":"$10^{1}$",
    "100.0":"$10^{2}$",
    "1000.0":"$10^{3}$",
    "10000.0":"$10^{4}$",
    "100000.0": "$10^{5}$",
}

gt = np.zeros((len(lambdas), len(deltas)))
eq = np.zeros((len(lambdas), len(deltas)))

for i, lamb in enumerate(lambdas):
    for j, delta in enumerate(deltas):
        print(i, lamb, j, delta, gt.shape)
        gt[i,j] = results["gt"][delta][lamb]/seeds#/firstk
        eq[i,j] = results["eq"][delta][lamb]/seeds#/firstk

colormap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(6, 2.1)
(a0, a1) = axs.flat
cbar_ax = fig.add_axes([.91, .3, .03, .4])

#cbar_ax = fig.add_axes([.91, .26, .03, .47])
ax = seaborn.heatmap(gt, cmap=colormap, ax=a0, vmin=0.0, vmax=0.5, square=True, rasterized=True,
                     cbar_ax=cbar_ax,
                     xticklabels=[deltatotex[delta] for delta in deltas],
                     yticklabels=[deltatotex[str(l)] for l in lambdas])
ax.set_title("Learning $>$")
ax.set_ylabel("$\\lambda$")
#ax.set_xticks(np.arange(7) + 0.5)
ax.set_xlabel("$\\delta$")
#plt.savefig("plots/scalinggt.pdf")
#plt.show()
#plt.clf()

ax = seaborn.heatmap(eq, cmap=colormap, ax=a1, vmin=0.0, vmax=0.5, square=True, rasterized=True,
                     cbar_ax=cbar_ax,
                     xticklabels=[deltatotex[delta] for delta in deltas],
                     yticklabels=[deltatotex[str(l)] for l in lambdas])
ax.set_title("Learning $=$")
#ax.set_xticks(np.arange(7) + 0.5)
ax.set_xlabel("$\\delta$")
plt.savefig("plots/scaling.pdf")
plt.show()
plt.clf()
