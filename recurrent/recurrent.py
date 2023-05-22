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

"""
class RecurrentMinTask:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self):
        for _ in range(400):
            x0 = random.randint(-10, 10)
            x1 = random.randint(-10, 10)
            x2 = random.randint(-10, 10)
            x3 = random.randint(-10, 10)
            x4 = random.randint(-10, 10)
            x = torch.tensor([x0, x1, x2, x3, x4])
            label = min(x)
            self.inputs.append(x)
            self.y.append(label)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, x0, length):
        xs = []
        ys = []
        lower = x0
        upper = x0 * 3 - 1
        pivot = random.randint(lower, upper)
        for _ in range(50):
            vector = [pivot] + [random.randint(pivot - 5, pivot + 5) for _ in range(length)]
            x = torch.tensor(vector)
            xs.append(x)
            ys.append(min(x))
        return torch.stack(xs).double(), torch.tensor(ys).double()


class RecurrentCountTask:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self):
        for _ in range(400):
            x0 = random.randint(-10, 10)
            x1 = random.randint(-10, 10)
            x2 = random.randint(-10, 10)
            x3 = random.randint(-10, 10)
            x4 = random.randint(-10, 10)
            vector = [x0, x1, x2, x3, x4]
            x = torch.tensor(vector)
            label = vector[1:].count(vector[0])
            self.inputs.append(x)
            self.y.append(label)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at(self, x0, length):
        xs = []
        ys = []
        lower = x0
        upper = x0 * 3 - 1
        pivot = random.randint(lower, upper)
        for _ in range(50):
            vector = [pivot] + [random.randint(pivot - 5, pivot + 5) for _ in range(length)]
            x = torch.tensor(vector)
            xs.append(x)
            ys.append(vector[1:].count(vector[0]))
        return torch.stack(xs).double(), torch.tensor(ys).double()
"""

class RecurrentMinTask2:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self):
        for _ in range(400):
            x = []
            for _ in range(5):
                x.append(random.randint(1, 10))
            x = torch.tensor(x)
            label = min(x)
            self.inputs.append(x)
            self.y.append(label)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at2(self, x0, length):
        xs = []
        ys = []
        lower = 1
        upper = 10 * x0
        for _ in range(50):
            x = []
            for _ in range(length):
                x.append(random.randint(lower, upper))
            x = torch.tensor(x)
            xs.append(x)
            ys.append(min(x))
        return torch.stack(xs).double(), torch.tensor(ys).double()


class RecurrentCountTask2:
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.y = []

    def create_training_set(self):
        for _ in range(400):
            x = []
            for _ in range(6):
                x.append(random.randint(1, 10))
            vector = x
            x = torch.tensor(vector)
            label = vector[1:].count(vector[0])
            self.inputs.append(x)
            self.y.append(label)
        return torch.stack([i for i in self.inputs]).double(), torch.tensor(self.y).double()

    def create_test_at2(self, x0, length):
        xs = []
        ys = []
        lower = 1
        upper = 10 * x0
        pivot = random.randint(lower, upper)
        for _ in range(50):
            vector = [pivot] + [random.randint(pivot - 5, pivot + 5) for _ in range(length)]
            x = torch.tensor(vector)
            xs.append(x)
            ys.append(vector[1:].count(vector[0]))
        return torch.stack(xs).double(), torch.tensor(ys).double()


def train_and_test(X, y, seed, args, device):
    input_size = 2
    if args.task == "min":
        if args.model == "lstm":
            model = MinLSTM()
        else:
            model = MinArchitecture(args.model, storage=input_size, redundancy=args.redundancy)
    else:
        if args.model == "lstm":
            model = CountLSTM()
        else:
            model = CountArchitecture(args.model, storage=input_size, redundancy=args.redundancy)
    opt = torch.optim.Adam(model.parameters())
    lr_step = torch.optim.lr_scheduler.StepLR(opt, step_size=11111, gamma=1.0)
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
        lr_step.step()
        if i % 1000 == 0:
            print("training in epoch ", i, " error ", yerror)
        if args.dbg:
            print(i, " ".join([str(e.item()) for e in errors]))

    scores = []

    with torch.no_grad():
        model.start_testing()
        for number in range(1, 15):
            for length in range(5, 51, 1):
                x1 = int(2 ** number)
                X, y = task.create_test_at2(x1, length)
                pred = model(X)[:, 0]
                acc = torch.mean(torch.abs(y - pred))
                scores.append(acc.item())

        with open("recurrent/results/{}_{}_{}_{}.csv".format(args.model, args.task, seed, args.redundancy), "w") as f:
            output = "{};{};{};{};{}\n".format(args.task, seed, args.redundancy, args.model, scores)
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
    parser.add_argument("--dbg", action="store_true")

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.task == "min":
        task = RecurrentMinTask2()
    else:
        task = RecurrentCountTask2()
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)
    for seed in range(10):
        print("####################seed", seed, "#########")
        random.seed(seed)
        torch.manual_seed(seed)
        X, y = task.create_training_set()
        X, y = X.to(device), y.to(device)
        train_and_test(X, y, seed, args, device)





