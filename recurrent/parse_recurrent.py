import argparse
from collections import defaultdict
import numpy as np
from ast import literal_eval as evallist
import seaborn
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 10
import os

def tolist(liststring):
    lst = evallist(liststring)
    return np.array(lst)


firstk = 7
lengths = 46
seeds = 10

fcts = ["rmin", "rcount"]
models = ["lstm", "2nn", "nsr", "nalu", "nau"]

results = {m: {f: np.zeros((firstk, lengths)) for f in fcts} for m in models}
seen = {m: {f: [False] * seeds for f in fcts} for m in models}


path = "recurrent/results/"
for file in os.listdir(path):
    if not os.path.isfile(path + file):
        continue
    for line in open(path + file, "r").readlines():
        task, seed, redundancy, model, scores = line.split(";")
        for i,score in enumerate(tolist(scores)):
            if i // lengths >= firstk:
                continue
            results[model][task][i//lengths,i%lengths] += float(score)
        if not seen[model][task][int(seed)]:
            seen[model][task][int(seed)] = True
        else:
            print("duplicate {} {} {}".format(model, task, seed))


for m in seen:
    for f in seen[m]:
        for i, seed in enumerate(seen[m][f]):
            if not seed:
                print("sbatch --cpus-per-task=2 run_comparisons.sh comparisons.py --outfile recurrent --epochs 50000"
                      " --task {} --model {} --seed {}".format(f, m, i))

#1 / 0





data = {0:results["lstm"]["rmin"], 1:results["2nn"]["rmin"], 2:results["nalu"]["rmin"],
        3:results["nau"]["rmin"], 4:results["nsr"]["rmin"],
        5:results["lstm"]["rcount"], 6:results["2nn"]["rcount"], 7:results["nalu"]["rcount"],
        8:results["nau"]["rcount"], 9:results["nsr"]["rcount"]}
yticks = ["$2^{" + str(i + 3) + "}$" if i%2==0 else "" for i in range(firstk)]
#xticks = np.arange(0,45, 5) + 5

fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
fig.set_size_inches(10.5, 4.2)
(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) = axs.flat
cbar_ax = fig.add_axes([.91, .3, .03, .4])

colormap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
s = seaborn.heatmap(results["lstm"]["rmin"]/seeds, ax=a0, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=1, yticklabels=yticks, rasterized=True)
s.set(ylabel="Minimum\nInput Scale")
a0.set_title("LSTM")
seaborn.heatmap(results["2nn"]["rmin"]/seeds, ax=a1, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=1, yticklabels=yticks, rasterized=True)
a1.set_title("MLP")
seaborn.heatmap(results["nalu"]["rmin"]/seeds, ax=a2, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
a2.set_title("NALU")
seaborn.heatmap(results["nau"]["rmin"]/seeds, ax=a3, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
a3.set_title("NAU")
seaborn.heatmap(results["nsr"]["rmin"]/seeds, ax=a4, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
a4.set_title("NSR")
s = seaborn.heatmap(results["lstm"]["rcount"]/seeds, ax=a5, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
s.set(xlabel="Sequence Length", ylabel="Count\nInput Scale")
s = seaborn.heatmap(results["2nn"]["rcount"]/seeds, ax=a6, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
s.set(xlabel="Sequence Length")
s = seaborn.heatmap(results["nalu"]["rcount"]/seeds, ax=a7, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
s.set(xlabel="Sequence Length")
s = seaborn.heatmap(results["nau"]["rcount"]/seeds, ax=a8, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
s.set(xlabel="Sequence Length")
s = seaborn.heatmap(results["nsr"]["rcount"]/seeds, ax=a9, cbar_ax=cbar_ax, cmap=colormap,
                vmin=0, vmax=2, yticklabels=yticks, rasterized=True)
s.set_xticks(np.arange(5, 46, 10))
s.set_xticklabels(np.arange(10, 51, 10))
s.set(xlabel="Sequence Length")
#s.figure.tight_layout()
fig.tight_layout(rect=[0, 0, .9, 1])
fig.savefig("plots/recurrent.pdf")
plt.show()


#for i, ax in enumerate(axs.flat):
#    print(i)
#    print(data[i]/seeds)
#    continue
#    seaborn.heatmap(data[i]/seeds, ax=ax, cbar_ax=cbar_ax, cmap="rainbow_r", vmin=0, vmax=1,
#                    xticklabels=xticks, yticklabels=yticks)
#    ax.set_title(titles[i])
#    ax.set_xlabel("Sequence Length" if i//3 == 1 else "")
    #ax.set_xticks(np.arange(10) + 0.5, [(i + 1) * 5 for i in range(10)])
#    ax.set_ylabel("Scale for Inputs" if i%3 == 0 else "")
#plt.savefig("plots/recurrent.pdf")
#plt.show()
# plt.yticks(np.arange(15) + 0.5, ["$3^{" + str(i + 1) + "}$" for i in range(15)])
# plt.ylabel("Extrapolation Scale")
# plt.savefig("plots/recurrent_{}_{}.pdf".format(m, f))

    #plt.show()
    #plt.clf()
