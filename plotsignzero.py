import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import numpy as np
import torch
import seaborn
import random

plt.rcParams["font.size"] = 14


x = np.arange(9) - 4
cx = np.linspace(-4,4,401)
print(np.shape(cx))

lamb = 3

plt.figure(figsize=(8.5,4.5))
signy = np.array([1 if i > 0 else -1 if i < 0 else 0 for i in x])
plt.scatter(x, signy, marker="o")#, label="$B_+$")
signhaty = np.tanh(cx)
signhatyscaled = np.tanh(lamb * (cx))
signhatyscaled2 = np.tanh(1/lamb * (cx))
plt.plot(cx, signhaty, label="$\hat{S}$", linestyle="--")

plt.axhline(linewidth=0.5, color="black")
plt.axvline(linewidth=0.5, color="black")
plt.xlabel("$x$")
plt.ylabel("$sign(x)$")
plt.legend()


zeroy = np.array([1 if i == 0 else -1 for i in x])
plt.scatter(x,zeroy, marker="x")#, label="$B_0$")
zerohaty = (1 - np.tanh(cx)**2) * 2 - 1
zerohatyscaled = (1 - np.tanh(lamb*cx)**2) * 2 - 1
zerohatyscaled2 = (1 - np.tanh(1/lamb*cx)**2) * 2 - 1
plt.plot(cx, zerohaty, label="$\hat{Z}$")

plt.axhline(linewidth=0.5, color="black")
plt.axvline(linewidth=0.5, color="black")
plt.xlabel("$x$")
plt.ylabel("$\hat{S}$ and $\hat{Z}$")
plt.legend(loc="center left")
plt.savefig("plots/signzero.pdf")
plt.show()
plt.clf()

print(np.tanh(0.01))
print(np.tanh(0.001))
print(np.tanh(0.0001))
plt.figure(figsize=(8.5,4.5))
#plt.plot(cx, signhaty, label="$\widehat{sign}_{\lambda=1}$")
plt.plot(cx, signhatyscaled, color="#1f77b4", linestyle="dashed",
         label = "$\hat{S}_{\lambda="+str(lamb)+"}}$")
plt.plot(cx, signhatyscaled2, color="#1f77b4", linestyle="dotted",
         label = "$\hat{S}_{\lambda=\\frac{1}{"+str(lamb)+"}}}$")
plt.axhline(linewidth=0.5, color="black")
plt.axvline(linewidth=0.5, color="black")
plt.xlabel("$x$")
plt.legend()
#plt.plot(cx, zerohaty, label="$\widehat{zero}_{\lambda=1}$")
plt.plot(cx, zerohatyscaled, color="#ff7f0e", linestyle="dashed",
         label="$\hat{Z}_{\lambda="+str(lamb)+"}}$")
plt.plot(cx, zerohatyscaled2, color="#ff7f0e", linestyle="dotted",
         label="$\hat{Z}_{\lambda=\\frac{1}{"+str(lamb)+"}}}$")
plt.axhline(linewidth=0.5, color="black")
plt.axvline(linewidth=0.5, color="black")
plt.xlabel("$x$")
plt.ylabel("$\hat{S}$ and $\hat{Z}$")
plt.legend(loc="center left")
plt.savefig("plots/signzeroscaled.pdf")
plt.show()
plt.clf()
plt.show()
plt.clf()

##########################################################################################

cx = np.linspace(-2,2,201)
colormap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)


grad = np.zeros((len(cx), len(cx)))
for i,x1 in enumerate(cx):
    for j,x2 in enumerate(cx):
        grad[i,j] = 0.25 * (1 - np.tanh((x1-x2))**2)
seaborn.heatmap(grad, vmin=-0.25, vmax=0.25, cmap=colormap, rasterized=True)
plt.xticks(np.arange(9) * 25, cx[0:201:25])
plt.xlabel("$m$")
plt.yticks(np.arange(9) * 25, cx[0:201:25])
plt.ylabel("$s$")
plt.tight_layout()
plt.savefig("plots/gradients_zeros.pdf")
plt.show()

cx = np.linspace(-2,2,201)
lower = np.linspace(0, 175, 201)
upper = np.linspace(25, 200, 201)
grad = np.zeros((len(cx), len(cx)))
for i,x1 in enumerate(cx):
    for j,x2 in enumerate(cx):
        grad[i,j] = 0.25 * (1 - 2*np.tanh((x1-x2))**2)
seaborn.heatmap(grad, vmin=-0.25, vmax=0.25, cmap=colormap, rasterized=True)
plt.plot(lower, lower + 25, c="y")
plt.plot(upper, upper - 25, c="y")
plt.xticks(np.arange(9) * 25, cx[0:201:25])
plt.xlabel("$m$")
plt.yticks(np.arange(9) * 25, cx[0:201:25])
plt.ylabel("$s$")
plt.tight_layout()
plt.savefig("plots/gradients_unscaled.pdf")
plt.show()


cx = np.linspace(-2, 2, 201)
for i,x1 in enumerate(cx):
    for j,x2 in enumerate(cx):
        zeroprime = -4 * np.sinh(x1 - x2) / np.cosh(x1 - x2)**3
        grad[i,j] = 0.25 * (zeroprime)
seaborn.heatmap(grad, vmin=-0.25, vmax=0.25, cmap=colormap, rasterized=True)
plt.xticks(np.arange(9) * 25, cx[0:201:25])
plt.xlabel("$m$")
plt.yticks(np.arange(9) * 25, cx[0:201:25])
plt.ylabel("$s$")
plt.tight_layout()
plt.savefig("plots/gradients_scale.pdf")
plt.show()



inputs = []
y = []
for i in range(-10, 10):
    for j in range(-10, 10):
        inputs.append(torch.tensor([i, j]))
        y.append(1 if i == j else 0)
inputs = torch.stack([row for row in inputs]).double()
y = torch.tensor(y).double()


class SimpleNSR(torch.nn.Module):
    def __init__(self, wsign, wzero):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1,1))
        self.Woperand1 = torch.nn.Parameter(torch.ones(2, 1))
        self.Woperand2 = torch.nn.Parameter(torch.ones(2, 1))
        self.Wzero = torch.nn.Parameter(torch.ones(1,1) * wzero)
        self.Wsign = torch.nn.Parameter(torch.ones(1,1) * wsign)

        torch.nn.init.xavier_normal_(self.Woperand1)
        torch.nn.init.xavier_normal_(self.Woperand2)
        print(self.Woperand1)
        print(self.Woperand2)
        #torch.nn.init.normal_(self.Wzero, 0, 0.1)
        #torch.nn.init.normal_(self.Wsign, 0, 0.1)

    def forward(self, storage):
        o1sel = torch.softmax(self.Woperand1, dim=0)
        o2sel = torch.softmax(self.Woperand2, dim=0)
        o1, o2 = storage @ o1sel, storage @ o2sel
        x = o1 - o2
        sign = torch.tanh(x)
        zero = -1 + 2 * (1 - torch.tanh(x).pow(2))
        sign, zero = sign @ self.Wsign, zero @ self.Wzero
        return torch.sigmoid(self.bias + sign + zero)


space = np.linspace(0, 0.25, 11)
matrix = np.zeros((len(space), len(space)))
for i, sign in enumerate(space):
    for j, zero in enumerate(space):
        count = 0
        for _ in range(10):
            model = SimpleNSR(sign, zero).double()
            opt = torch.optim.Adam(model.parameters())
            for epoch in range(1, 12501):
                opt.zero_grad()
                pred = model(inputs)
                pred = pred[:, 0]
                yerror = torch.mean(torch.abs(pred - y))
                if yerror < 0.04:
                    print("reached error {}, success".format(yerror.item()))
                    count += 1
                    break
                yerror.backward()
                opt.step()
                if epoch % 1000 == 0:
                    print("training in epoch ", epoch, " error ", yerror.item())
        matrix[i,j] = count
labels = [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]
colormap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
seaborn.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap=colormap, rasterized=True)
plt.xlabel("$W_O$")
plt.ylabel("$W_+$")
plt.savefig("plots/success_eq.pdf")
plt.show()
