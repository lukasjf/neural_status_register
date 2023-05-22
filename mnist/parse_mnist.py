import argparse
from collections import defaultdict
import numpy as np
from ast import literal_eval as tolist
import matplotlib.pyplot as plt
import os

plt.rcParams["font.size"] = 14


parser = argparse.ArgumentParser(description="")
parser.add_argument("--mnistpath", default="old_results.csv/mnist.csv", action="store_true")
parser.add_argument("--basepath", default="old_results.csv/basemnist.csv", action="store_true")
args = parser.parse_args()

ep = 13
seeds = 10

"""basicmnist = {i:[] for i in range(1,ep+1)}
gt2nn = {i:[] for i in range(1,ep+1)}
gtnsr = {i:[] for i in range(1,ep+1)}
gtnalu = {i:[] for i in range(1,ep+1)}
gtnau = {i:[] for i in range(1,ep+1)}
eq2nn = {i:[] for i in range(1,ep+1)}
eqnsr = {i:[] for i in range(1,ep+1)}
eqnau = {i:[] for i in range(1,ep+1)}
eqnalu = {i:[] for i in range(1,ep+1)}"""

basicmnist = {i:[] for i in range(10)}
gt2nn = {i:[] for i in range(10)}
gtnsr = {i:[] for i in range(10)}
gtnalu = {i:[] for i in range(10)}
gtnau = {i:[] for i in range(10)}
eq2nn = {i:[] for i in range(10)}
eqnsr = {i:[] for i in range(10)}
eqnau = {i:[] for i in range(10)}
eqnalu = {i:[] for i in range(10)}

path = "mnist/results/"
for file in os.listdir(path):
    if not os.path.isfile(path+file):
        continue
    if file == "number.csv":
        data = basicmnist
    elif "nau_gt" in file:
        data = gtnau
    elif "nsr_gt" in file:
        data = gtnsr
    elif "nalu_gt" in file:
        data = gtnalu
    elif "2nn_gt" in file:
        data = gt2nn
    elif "nau_eq" in file:
        data = eqnau
    elif "2nn_eq" in file:
        data = eq2nn
    elif "nalu_eq" in file:
        data = eqnalu
    elif "nsr_eq" in file:
        data = eqnsr
    else:
        print(file)
        continue
    for line in open(path+file, "r").readlines():
        output, epoch, seed, model, mnisterror, accuracy  = line.split(",")
        data[int(seed)].append(float(accuracy))
        #if int(epoch) > ep:
        #    continue
        #data[int(epoch)].append(float(accuracy))

x = range(10)
print(np.mean([max(gt2nn[i]) for i in x]), np.std([max(gt2nn[i]) for i in x]))
print(np.mean([max(gtnalu[i]) for i in x]), np.std([max(gtnalu[i]) for i in x]))
print(np.mean([max(gtnau[i]) for i in x]), np.std([max(gtnau[i]) for i in x]))
print(np.mean([max(gtnsr[i]) for i in x]), np.std([max(gtnsr[i]) for i in x]))
print("__________")
print(np.mean([max(eq2nn[i]) for i in x]), np.std([max(eq2nn[i]) for i in x]))
print(np.mean([max(eqnalu[i]) for i in x]), np.std([max(eqnalu[i]) for i in x]))
print(np.mean([max(eqnau[i]) for i in x]), np.std([max(eqnau[i]) for i in x]))
print(np.mean([max(eqnsr[i]) for i in x]), np.std([max(eqnsr[i]) for i in x]))


x = np.arange(ep) + 1
plt.plot(x, [np.mean(basicmnist[i])*100 for i in x], label="Mnist")
plt.plot(x, [np.mean(gt2nn[i]) for i in x], label="GT-2nn")
plt.plot(x, [np.mean(gtnau[i]) for i in x], label="GT-NAU")
plt.plot(x, [np.mean(gtnalu[i]) for i in x], label="GT-NALU")
plt.plot(x, [np.mean(gtnsr[i]) for i in x], label="GT-NSR")
plt.legend()
plt.savefig("plots/mnistgt.pdf")
plt.show()

x = np.arange(ep) + 1
#plt.plot(x, [np.mean(basicmnist[i])*100 for i in x], label="Mnist")
plt.plot(x, [np.mean(eq2nn[i]) for i in x], label="EQ-2nn")
plt.plot(x, [np.mean(eqnau[i]) for i in x], label="EQ-NAU")
plt.plot(x, [np.mean(eqnalu[i]) for i in x], label="EQ-NALU")
plt.plot(x, [np.mean(eqnsr[i]) for i in x], label="EQ-NSR")
plt.legend()
plt.savefig("plots/mnisteq.pdf")
plt.show()


