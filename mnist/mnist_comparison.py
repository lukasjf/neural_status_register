from __future__ import print_function
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import random

import sys
sys.path.append('.')
from models import *

#from https://github.com/pytorch/examples/blob/234bcff4a2d8480f156799e6b9baae06f7ddc96a/mnist/main.py
class Net(nn.Module):
    def __init__(self, digits, sr):
        super(Net, self).__init__()
        self.digits = digits
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sr = sr

    def forward(self, x):
        batchsize = len(x)
        digits = self.cnn(x)
        digits = digits.view(batchsize//2,2)
        pred = self.sr(digits)
        return pred

    def cnn(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        #return torch.softmax(x, dim=1)
        num = torch.softmax(x, dim=1) @ self.digits.T
        return num

def reshape_gt(X, y):
    data = []
    target = []
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            data.append(X[i])
            data.append(X[j])
            target.append(torch.tensor(1 if y[i] > y[j] else 0))
    data, target = torch.stack(data).double().to(device),\
                   torch.stack(target).double().to(device)
    return data, target

def reshape_eq(X, y):
    index = {}
    for i in range(10):
        index[i] = []
    data, target = [], []
    for i in range(len(y)):
        index[y[i].item()].append(i)
    for digit in index:
        for i1 in index[digit]:
            for i2 in index[digit]:
                if i1 >= i2:
                    continue
                data.append(X[i1])
                data.append(X[i2])
                target.append(torch.tensor(1))
                otherdigit = random.randint(0, 9)
                while digit == otherdigit or not index[otherdigit]:
                    otherdigit = random.randint(0, 9)
                data.append(X[i1])
                data.append(X[random.choice(index[otherdigit])])
                target.append(torch.tensor(0))
    data, target = torch.stack(data).double().to(device),\
                   torch.stack(target).double().to(device)
    return data, target

def train(args, model, device, optimizer, epoch, bidx, X, y):
    model.train()
    if args.comparison == "gt":
        data, target = reshape_gt(X, y)
    else:
        data, target = reshape_eq(X, y)
    optimizer.zero_grad()
    pred = model(data)
    errors = []
    for i, p in enumerate(pred):
        error = torch.abs(target[i] - p[0])
        errors.append(error)
    loss = torch.mean(torch.stack(errors))
    loss.backward()
    optimizer.step()
    if bidx % args.log_interval == 0:
        print("Train Epoch {}, Batch {}: Loss {}".format(epoch, bidx, loss.item()))


def test(args, model, device, X, y, result, epoch):
    model.eval()
    if args.comparison == "gt":
        data, target = reshape_gt(X, y)
    else:
        data, target = reshape_eq(X, y)
    pred = model(data)
    errors = []
    correct = []
    for i, p in enumerate(pred):
        error = torch.abs(target[i] - p[0])
        errors.append(error)
        correct.append(1 if error < 0.5 else 0)
    return errors, correct



if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--trainsize', type=int, default=1000)
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--comparison', type=str, default="")
    parser.add_argument("--model", type=str, default="nsr")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    result = open("mnist/results/{}_{}_{}.csv".format(args.model, args.comparison, args.seed), "w")
    digits = torch.tensor([[0., 1, 2, 3, 4, 5, 6, 7, 8, 9]]).double().to(device)
    if args.model == "nsr":
        comp = NeuralStatusRegister(storage=2, redundancy=10, zeta=10)
    elif args.model == "nau":
        comp = NAU(storage=2, redundancy=10)
    elif args.model == "nalu":
        comp = NALU(storage=2, redundancy=10)
    else:
        comp = TwoNN(storage=2, redundancy=10)
    model = Net(digits, comp).double().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        for i, (xsample, ysample) in enumerate(train_loader):
            image = xsample[0][0]
            image = torch.flatten((image > 0).int())
            print(image, ysample[0])
            train(args, model, device, optimizer, epoch, i//args.batch_size + 1, xsample, ysample)


        losses = []
        corrects = []

        with torch.no_grad():
            for i, (xsample, ysample) in enumerate(test_loader):
                loss, correct = test(args, model, device, xsample, ysample, result, epoch)
                losses.extend(loss)
                corrects.extend(correct)
            loss = np.array(losses).mean()
            acc = 100. * np.array(corrects).mean()
            print("Loss: {:.4f}, with Accuracy: {:.2f}%".format(loss, acc))
            result.write("mnist/old_results/{},{},{},{},{},{}\n".format(args.comparison,
                                                                epoch,
                                                                args.seed,
                                                                args.model,
                                                                loss, acc))
        #scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
