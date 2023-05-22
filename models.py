import torch
import math
import dgl
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NeuralStatusRegister(torch.nn.Module):
    def __init__(self, storage, redundancy, zeta):
        super().__init__()

        self.testing = False

        self.storage_size = storage
        self.zeta = zeta
        self.redundancy = 1#redundancy

        self.Woperand1 = torch.nn.Parameter(torch.Tensor(self.storage_size, self.redundancy))
        self.Woperand2 = torch.nn.Parameter(torch.Tensor(self.storage_size, self.redundancy))


        self.bias = torch.nn.Parameter(torch.zeros(1, 1))
        self.Wzero = torch.nn.Parameter(torch.randn(self.redundancy, 1))
        self.Wsign = torch.nn.Parameter(torch.randn(self.redundancy, 1))

        self.register_buffer("zero", torch.tensor(0.))
        self.register_buffer("one", torch.tensor(1.))
        self.register_buffer("mone", torch.tensor(-1.))

        self.states = None

        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.Woperand1)
        torch.nn.init.xavier_normal_(self.Woperand2)

        torch.nn.init.xavier_normal_(self.Wzero)
        torch.nn.init.xavier_normal_(self.Wsign)

    def start_testing(self):
        self.testing = True

    def forward(self, storage):
        o1sel = torch.softmax(self.Woperand1, dim=0)
        o2sel = torch.softmax(self.Woperand2, dim=0)
        o1, o2 = storage @ o1sel, storage @ o2sel
        if self.testing:
            print(self.Woperand1)
            print(self.Woperand2)
            #for i in range(storage.size()[0]):
            #    stor = storage[i]
            #    stor2 = torch.tensor([stor[1], stor[0]])
                #print(stor, stor @ o1sel, stor @ o2sel)
                #print(stor2, stor2 @ o1sel, stor2 @ o2sel)
            print(self.bias, self.Wsign, self.Wzero)
        x = o1 - o2
        sign = torch.tanh(self.zeta * x)
        zero = -1 + 2 * (1 - torch.tanh(self.zeta * x).pow(2))
        sign, zero = sign @ self.Wsign, zero @ self.Wzero

        return self.nextstate(sign, zero)

    def nextstate(self, sign, zero):
        zstate = self.bias + sign + zero
        state = torch.sigmoid(zstate)
        return torch.cat([state, 1 - state], dim=1)

    def regloss(self):
        return 0


class TwoNN(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.l1 = torch.nn.Linear(storage, 2*redundancy)
        self.l2 = torch.nn.Linear(2*redundancy, 1)

    def forward(self, storage):
        x = torch.sigmoid(self.l1(storage))
        x = torch.sigmoid(self.l2(x))
        return torch.cat([x, 1-x], dim=1)

    def regloss(self):
        return 0

    def start_testing(self):
        pass


class NALU(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.storage_size = storage

        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Parameter(torch.Tensor(self.storage_size, redundancy))
        self.tanh = torch.nn.Parameter(torch.Tensor(self.storage_size, redundancy))

        self.combine = torch.nn.Linear(redundancy, 1)

        self.testing = False
        self.init()

    def regloss(self):
        return 0

    def start_testing(self):
        self.testing = True

    def init(self):
        torch.nn.init.xavier_normal_(self.sigmoid)
        torch.nn.init.xavier_normal_(self.tanh)

    def forward(self, storage):
        control = torch.sigmoid(self.sigmoid) + torch.tanh(self.tanh)
        plusminus = storage @ control
        value = plusminus

        out = torch.sigmoid(self.bias + self.combine(value))
        return torch.cat([out, 1-out], dim=1)


class NAU(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.storage_size = storage

        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.W = torch.nn.Parameter(torch.Tensor(self.storage_size, 2*redundancy))
        self.W2 = torch.nn.Parameter(torch.Tensor(2*redundancy, 1))

        self.testing = False
        self.init()

    def regloss(self):
        w = torch.abs(self.W)
        w2 = torch.abs(self.W)
        return torch.mean(torch.min(w, 1 - w)) + torch.mean(torch.min(w2, 1 - w2))

    def start_testing(self):
        self.testing = True

    def init(self):
        torch.nn.init.xavier_normal_(self.W)
        torch.nn.init.xavier_normal_(self.W2)

    def forward(self, storage):
        w = torch.clamp(self.W, -1, 1)
        w2 = torch.clamp(self.W2, -1, 1)
        out = torch.sigmoid(self.bias + (storage @ w) @ w2)
        return torch.cat([out, 1-out], dim=1)


class NPU(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.storage_size = storage

        self.bias = torch.nn.Parameter(torch.zeros(1,1))
        self.gate = torch.nn.Parameter(torch.ones(1, storage) * 0.5)
        self.W = torch.nn.Parameter(torch.Tensor(storage, 2*redundancy))
        self.combine = torch.nn.Linear(2*redundancy, 1)
        self.init()

    def forward(self, storage):
        g = torch.clamp(self.gate, 0, 1)
        r = torch.abs(storage) + 0.001
        r = g * r + (1-g)
        k = torch.maximum(torch.sign(r), torch.zeros_like(r)) * math.pi
        out = torch.sigmoid(self.bias +
                            self.combine(torch.exp(torch.log(r) @ self.W) * torch.cos(k @ self.W)))
        return torch.cat([out, 1-out], dim=1)

    def init(self):
        torch.nn.init.xavier_normal_(self.W)

    def start_testing(self):
        pass

    def regloss(self):
        return 0


class MinArchitecture2(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.storage = storage
        self.redundancy = redundancy
        self.bias = torch.nn.Parameter(torch.ones(1, 1)*5)
        self.Wzero = torch.nn.Parameter(torch.zeros(1, 1))
        self.Wsign = torch.nn.Parameter(torch.zeros(1, 1))

    def forward(self, X):
        h = X[:,0:1] #to take first column but keep 2D
        for i in range(1, X.size()[1]):
            x = h - X[:,i:i+1]
            sign = torch.tanh(x)
            zero = -1 + 2 * (1 - torch.tanh(x).pow(2))
            sign, zero = sign @ self.Wsign, zero @ self.Wzero
            state = torch.sigmoid(self.bias + sign + zero)
            h = h * (1-state) + X[:, i:i+1] * state
        return h


class MinArchitecture(torch.nn.Module):
    def __init__(self, model, storage, redundancy):
        super().__init__()
        if model == "nsr":
            self.model = NeuralStatusRegister(storage=storage, redundancy=redundancy, zeta=1.0)
        elif model == "2nn":
            self.model = TwoNN(storage=storage, redundancy=redundancy)
        elif model == "nalu":
            self.model = NALU(storage=storage, redundancy=redundancy)
        elif model == "nau":
            self.model = NAU(storage=storage, redundancy = redundancy)
        else:
            self.model = None

    def regloss(self):
        return self.model.regloss()

    def start_testing(self):
        self.model.start_testing()

    def forward(self, X):
        h = X[:,0:1] #to take first column but keep 2D
        for i in range(1, X.size()[1]):
            current = torch.cat([h, X[:,i:i+1]], dim=1)
            state = self.model(current)
            h = torch.einsum("bo, bo -> b", current, state).unsqueeze(1)
        return h


class CountArchitecture(torch.nn.Module):
    def __init__(self, model, storage, redundancy):
        super().__init__()
        self.one = torch.arange(2).double()
        if model == "nsr":
            self.model = NeuralStatusRegister(storage=storage, redundancy=redundancy, zeta=1.0)
        elif model == "2nn":
            self.model = TwoNN(storage=storage, redundancy=redundancy)
        elif model == "nalu":
            self.model = NALU(storage=storage, redundancy=redundancy)
        elif model == "nau":
            self.model = NAU(storage=storage, redundancy=redundancy)
        else:
            self.model = None

    def regloss(self):
        return self.model.regloss()

    def start_testing(self):
        self.model.start_testing()

    def forward(self, X):
        val = X[:,0:1] #to take first column but keep 2D
        count = torch.zeros(X.size()[0])
        for i in range(1, X.size()[1]):
            current = torch.cat([val, X[:, i:i+1]], dim=1)
            state = self.model(current)
            count += state @ self.one
        return count.unsqueeze(1)


class MinLSTM(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.states = 10
        self.update_input = torch.nn.Linear(1, self.states)
        self.update_state = torch.nn.Linear(self.states, self.states)
        self.forget_input = torch.nn.Linear(1, self.states)
        self.forget_state = torch.nn.Linear(self.states, self.states)
        self.newstate_input = torch.nn.Linear(1, self.states)
        self.newstate_state = torch.nn.Linear(self.states, self.states)
        self.readout = torch.nn.Linear(self.states, 1)

    def forward(self, X):
        samples = X.size(0)
        length = X.size(1)
        h = torch.zeros((samples, self.states))
        h[:, 0] = X[:, 0]
        for i in range(1, length):
            x = X[:,i:i+1]
            input = torch.sigmoid(self.update_input(x) + self.update_state(h))
            forget = torch.sigmoid(self.forget_input(x) + self.forget_state(h))
            newh = self.newstate_input(x) + self.newstate_state(h)
            h = input * newh + forget * h
        return self.readout(h)

    def regloss(self):
        return 0

    def start_testing(self):
        pass


class CountLSTM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.states = 10
        self.update_input = torch.nn.Linear(2, self.states)
        self.update_state = torch.nn.Linear(self.states, self.states)
        self.forget_input = torch.nn.Linear(2, self.states)
        self.forget_state = torch.nn.Linear(self.states, self.states)
        self.newstate_input = torch.nn.Linear(2, self.states)
        self.newstate_state = torch.nn.Linear(self.states, self.states)
        self.newc = torch.nn.Linear(2, 1)
        self.newc = torch.nn.Linear(self.states, 1)
        self.readout = torch.nn.Linear(self.states, 1)

    def forward(self, X):
        samples = X.size(0)
        length = X.size(1)
        h = torch.zeros((samples, self.states))
        c = torch.zeros(samples, 1)
        search = X[:, 0:1]
        for i in range(1, length):
            x = torch.cat([search, X[:,i:i+1]], dim=1)
            input = torch.sigmoid(self.update_input(x) + self.update_state(h))
            forget = torch.sigmoid(self.forget_input(x) + self.forget_state(h))
            newc = torch.sigmoid(self.newstate_input(x) + self.newstate_state(h))
            c = forget * c + input * newc
            h = torch.sigmoid(self.newstate_input(x) + self.newstate_state(h))
        return c + self.readout(h)

    def regloss(self):
        return 0

    def start_testing(self):
        pass


class NALU_ALU(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.storage_size = storage
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.sigmoid = torch.nn.Parameter(torch.Tensor(self.storage_size, 1))
        self.tanh = torch.nn.Parameter(torch.Tensor(self.storage_size, 1))
        self.testing = False
        self.init()

    def regloss(self):
        return 0

    def start_testing(self):
        self.testing = True

    def init(self):
        torch.nn.init.xavier_normal_(self.sigmoid)
        torch.nn.init.xavier_normal_(self.tanh)

    def forward(self, storage):
        control = torch.sigmoid(self.sigmoid) + torch.tanh(self.tanh)
        return self.bias + storage @ control


class NAU_ALU(torch.nn.Module):
    def __init__(self, storage, redundancy):
        super().__init__()
        self.storage_size = storage
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.W = torch.nn.Parameter(torch.Tensor(self.storage_size, 1))
        self.testing = False
        self.init()

    def regloss(self):
        w = torch.abs(self.W)
        return torch.mean(torch.min(w, 1 - w))

    def init(self):
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, storage):
        w = torch.clamp(self.W, -1, 1)
        return self.bias + storage @ w



class AluWithComparison(torch.nn.Module):
    def __init__(self, model, alu_model, storage, redundancy):
        super().__init__()
        alu = NALU_ALU if alu_model == "nalu" else NAU_ALU
        self.alu_true = alu(storage=storage, redundancy=redundancy)
        self.alu_false = alu(storage=storage, redundancy=redundancy)
        if model == "nsr":
            self.model = NeuralStatusRegister(storage=storage, redundancy=redundancy, zeta=1.0)
        elif model == "2nn":
            self.model = TwoNN(storage=storage, redundancy=redundancy)
        elif model == "nalu":
            self.model = NALU(storage=storage, redundancy=redundancy)
        elif model == "nau":
            self.model = NAU(storage=storage, redundancy=redundancy)
        else:
            self.model = None

    def regloss(self):
        return self.alu_false.regloss() + self.alu_true.regloss() + self.model.regloss()

    def start_testing(self):
        self.model.start_testing()

    def forward(self, X):
        comparison = self.model(X)
        c_true = comparison[:,0:1]
        return comparison, self.alu_true(X) * c_true + self.alu_false(X) * (1 - c_true)