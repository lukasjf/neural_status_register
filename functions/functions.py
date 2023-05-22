import torch
import random
import pickle
import numpy as np
import argparse
import os
import random

import sys
sys.path.append('.')

from models import *


class AbsTask:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self, scale=1.0):
        for i in range(-10, 10):
            for j in range(-10, 10):
                self.inputs.append(torch.tensor([i, j]))
                self.y.append(abs(i - j))
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, x1, scale):
        xs = []
        ys = []
        for x2 in range(-5, 6):
            xs.append(torch.tensor([x1, x2]))
            ys.append(abs(x1 - x2))
        return torch.stack(xs).double(), torch.tensor(ys).double()


class FTask:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self, scale=1.0):
        for i in range(10):
            for x0 in range(-10, 10):
                for x1 in range(-10, 10):
                    x2 = random.randint(-100, 100)
                    x3 = random.randint(-100, 100)
                    x4 = random.randint(-100, 100)
                    x = torch.tensor([x0, x1, x2, x3, x4])
                    label = x[4] + 4 if x[0] > x[1] else x[3] - x[2]
                    self.inputs.append(x)
                    self.y.append(label)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, x0, scale):
        xs = []
        ys = []
        for i in range(-5, 6):
            x1 = x0 + i
            x2 = random.randint(-100, 100)
            x3 = random.randint(-100, 100)
            x4 = random.randint(-100, 100)
            x = torch.tensor([x0, x1, x2, x3, x4])
            xs.append(x)
            ys.append(x[4] + 4 if x[0] > x[1] else x[3] - x[2])
        return torch.stack(xs).double(), torch.tensor(ys).double()


class FEQTask:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self, scale=1.0):
        for i in range(100):
            for x0 in range(-10, 10):
                #for x1 in range(-10, 10):
                x1 = random.randint(-10, 9)
                while x0 == x1:
                    x1 = random.randint(-10, 9)
                x2 = random.randint(-100, 100)
                x3 = random.randint(-100, 100)
                x4 = random.randint(-100, 100)
                x = torch.tensor([x0, x1, x2, x3, x4])
                label = x[3] - x[2]
                self.inputs.append(x)
                self.y.append(label)

                x1 = x0
                x2 = random.randint(-100, 100)
                x3 = random.randint(-100, 100)
                x4 = random.randint(-100, 100)
                x = torch.tensor([x0, x1, x2, x3, x4])
                label = x[4] + 4
                self.inputs.append(x)
                self.y.append(label)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, x0, scale):
        xs = []
        ys = []
        for i in range(-5, 6):
            x1 = x0 + i
            x2 = random.randint(-100, 100)
            x3 = random.randint(-100, 100)
            x4 = random.randint(-100, 100)
            x = torch.tensor([x0, x1, x2, x3, x4])
            xs.append(x)
            ys.append(x[4] + 4 if x[0] == x[1] else x[3] - x[2])
            x = torch.tensor([x1, x1, x2, x3, x4])
            xs.append(x)
            ys.append(x[4] + 4 if x[0] == x[1] else x[3] - x[2])
        return torch.stack(xs).double(), torch.tensor(ys).double()


def train_and_test(X, y, seed, args, device):
    input_size = 2 if args.task=="abs" else 5
    model = AluWithComparison(args.model, args.alumodel, storage=input_size, redundancy=args.redundancy).to(device)
    opt = torch.optim.Adam(model.parameters())
    lr_step = torch.optim.lr_scheduler.StepLR(opt, step_size=11111, gamma=1.0)
    for i in range(1, args.epochs + 1):
        errors = []
        opt.zero_grad()
        comp, output = model(X)
        pred = output[:, 0]
        yerror = torch.mean(torch.abs(pred - y))
        #regerror = torch.mean(torch.minimum(comp[:,0:1], 1-comp[:,0:1]))
        errors.append(yerror)
        if i > 40000:
            (yerror + model.regloss()).backward()
        #elif args.state:
        #    (yerror + regerror).backward()
        else:
            yerror.backward()
        opt.step()
        lr_step.step()
        if i % 1000 == 0:
            print("training in epoch ", i, " error ", yerror, "states ", 0)#regerror)
        if args.dbg:
            print(i, " ".join([str(e.item()) for e in errors]))

    scores = []
    with torch.no_grad():
        model.start_testing()
        for i in range(3, 15):
            base = 2 ** i
            x1 = random.randint(base, 2 * base - 1)
            testX, testy = task.create_test_at(x1, args.scale)
            comp, pred = model(testX)
            acc = torch.abs(testy - pred[:, 0]).mean()
            print(base, acc.item())
            scores.append(acc.item())

        with open("functions/results/test{}_{}_{}_{}_{}_{}.csv".format(args.prefix, args.model, args.alumodel,
                                                                         args.task, seed, args.redundancy),
                  "w") as f:
            output = "{};{};{};{};{};{}\n".format(args.task, seed, args.redundancy, args.model, args.alumodel,
                                                  scores)
            print(output)
            f.write(output)
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="nsr")
    parser.add_argument("--alumodel", type=str, default="nalu")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--task", type=str, default="gt")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--redundancy", type=int, default=10)
    parser.add_argument("--trainrange", type=int, default=[-9, 10], nargs="+")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--dbg", action="store_true")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument('--state', action='store_true')
    parser.add_argument('--no-state', dest='feature', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.task == "f":
        task = FTask()
    elif args.task == "feq":
        task = FEQTask()
    else:
        task = None

    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)
    for seed in range(10):
        print("####################seed", seed, "#########")
        random.seed(seed)
        torch.manual_seed(seed)
        X, y = task.create_training_set(args.scale)
        X, y = X.to(device), y.to(device)
        train_and_test(X, y, seed, args, device)





