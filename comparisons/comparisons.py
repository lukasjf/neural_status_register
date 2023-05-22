import torch
import random
import pickle
import numpy as np
import argparse
import os
import random

import sys
sys.path.append('.')

from models import NeuralStatusRegister, NALU, NAU, TwoNN, NPU


class CompTask:
    def __init__(self, op):
        self.inputs = []
        self.y = []
        self.op = op

    def create_training_set(self, scale=1.0):
        for i in range(1, 10):
            for j in range(1, 10):
                x1, x2 = i * scale, j * scale
                self.inputs.append(torch.tensor([x1, x2]))
                self.y.append(self.op(x1 * 10000, x2))
        print(self.y)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, a, scale):
        xs = []
        ys = []
        for i in range(-5, 6):
            x1 = a
            x2 = x1 + i * scale
            xs.append(torch.tensor([x1, x2]))
            ys.append(self.op(x1 * 10000, x2))
            if i != 0:
                xs.append(torch.tensor([x2, x1]))
                ys.append(self.op(x2 * 10000, x1))
        print(self.y)
        return torch.stack(xs).double(), torch.tensor(ys).double()


class EqualityTask(CompTask):
    def __init__(self, op):
        super().__init__(op)

    def create_training_set(self, scale=1.0):
        for i in range(1, 10):
            j = random.randint(1, 10)
            while i == j:
                j = random.randint(1, 10)
            x1, x2 = i * scale, j * scale
            self.inputs.append(torch.tensor([x1, x2]))
            self.y.append(self.op(x1* 10000, x2))

            j = i
            x1, x2 = i * scale, j * scale
            self.inputs.append(torch.tensor([x1, x2]))
            self.y.append(self.op(x1* 10000, x2))
        print(self.y)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, a, scale):
        xs = []
        ys = []
        for i in range(-5, 6):
            if i == 0:
                continue
            x1 = a
            x2 = x1 + i * scale
            xs.append(torch.tensor([x1, x2]))
            ys.append(self.op(x1* 10000, x2))
            xs.append(torch.tensor([x2, x2]))
            ys.append(self.op(x2* 10000, x2))
        print(self.y)
        return torch.stack(xs).double(), torch.tensor(ys).double()


def train_and_test(X, y, seed, args, device):
    input_size = 2
    if args.model == "nsr":
        model = NeuralStatusRegister(storage=input_size, zeta=args.zeta, redundancy=args.redundancy).to(device)
    elif args.model == "nalu":
        model = NALU(storage=input_size, redundancy=args.redundancy).to(device)
    elif args.model == "nau":
        model = NAU(storage=input_size, redundancy=args.redundancy).to(device)
    elif args.model == "npu":
        model = NPU(storage=input_size, redundancy=args.redundancy).to(device)
    elif args.model == "2nn":
        model = TwoNN(storage=input_size, redundancy=args.redundancy).to(device)
    else:
        1/0
    opt = torch.optim.Adam(model.parameters())
    lr_step = torch.optim.lr_scheduler.StepLR(opt, step_size=11111, gamma=1)
    for i in range(1, args.epochs + 1):
        errors = []
        opt.zero_grad()
        output = model(X)
        pred = output[:, 0]
        yerror = torch.mean(torch.abs(pred - y))
        errors.append(yerror)
        if i > 40000:
            (yerror + model.regloss()).backward()
        else:
            yerror.backward()
        opt.step()
        #lr_step.step()
        if i % 1000 == 0:
            print("training in epoch ", i, " error ", yerror.item())
        if args.dbg:
            print(i, " ".join([str(e.item()) for e in errors]))

    scores = []
    with torch.no_grad():
        model.start_testing()
        for i in range(1, 19):
            base = 10 ** i
            x1 = random.randint(base, 10 * base - 1)
            testX, testy = task.create_test_at(x1, args.scale)
            pred = model(testX)[:, 0]
            acc = (torch.abs(testy - pred)).mean()
            print(base, acc.item())
            scores.append(acc.item())

        with open("comparisons/results/test2{}_{}_{}_{}_{}_{}_{}.csv".format(args.prefix, args.model, args.task, seed,
                                                                        args.redundancy, args.scale, args.zeta),
                  "w") as f:
            output = "{};{};{};{};{};{};{}\n".format(args.task, seed, args.redundancy,
                                                      args.scale, args.zeta, args.model, scores)
            print(output)
            f.write(output)
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="nsr")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--task", type=str, default="gt")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--redundancy", type=int, default=10)
    parser.add_argument("--trainrange", type=int, default=[-9, 10], nargs="+")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--dbg", action="store_true")
    parser.add_argument("--prefix", type=str, default="")

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.task == "gt":
        task = CompTask(op=lambda a, b: 1 if a > b else 0)
    elif args.task == "lt":
        task = CompTask(op=lambda a, b: 1 if a < b else 0)
    elif args.task == "leq":
        task = CompTask(op=lambda a, b: 1 if a <= b else 0)
    elif args.task == "geq":
        task = CompTask(op=lambda a, b: 1 if a > + b else 0)
    elif args.task == "eq":
        task = EqualityTask(op=lambda a, b: 1 if a == b else 0)
    elif args.task == "ne":
        task = EqualityTask(op=lambda a, b: 1 if a != b else 0)

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





