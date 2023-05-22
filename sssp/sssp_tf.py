import torch
import random
import dgl
import networkx as nx
import argparse
import heapq

import sys
sys.path.append('.')
from models import *


# https://stackoverflow.com/questions/2041517/random-simple-connected-graph-generation-with-given-sparseness
def randomgraph(n, **args):
    g = dgl.DGLGraph()
    g.add_nodes(n)
    tree = set()
    nodes = list(range(n))
    current = random.choice(nodes)
    tree.add(current)
    while(len(tree) < n):
        nxt = random.choice(nodes)
        if not nxt in tree:
            tree.add(nxt)
            g.add_edges(current, nxt)
            g.add_edges(nxt, current)
        current = nxt
    for i in range(n - 1):
        for j in range(i+1, n):
            if g.has_edges_between(i, j):
                continue
            else:
                if random.random() < args.get("p", 0.5/n):
                    g.add_edges(i, j)
                    g.add_edges(j, i)
    return g


class SSSPTask:
    def __init__(self, g, start, n, wmax):
        self.attr = {}
        self.g = g
        self.N = [int(i) for i in g.nodes()]
        self.y_dict = {}
        self.An_dict = {}
        self.Ae_dict = {}

        self.g = g
        num_edges = len(g.edges()[0])//2
        w = torch.randint(1, wmax, (num_edges,)).float()
        w = torch.stack([w,w]).transpose(0, 1).contiguous().view(-1)  # mirror for symmetric edges
        g.edata["w"] = w.unsqueeze(1)
        g.ndata["orig_d"] = torch.ones(len(g.nodes()), 1).float() * n * wmax
        self.start = start
        self.n = n
        self.w = wmax

        for n in g.nodes():
            g.add_edges(n, n)
            g.edata["w"][g.edge_ids(n, n)] = torch.zeros(1, 1).float()

        seen = []
        done = set()
        self.maxhop = 0
        seen.append((0, 0, start))
        while seen:
            dist, hops, curr = heapq.heappop(seen)
            self.maxhop = self.maxhop if self.maxhop > hops else hops
            if curr in done:
                continue
            done.add(curr)
            self.y_dict[curr] = torch.tensor([1.0 * dist])
            for n in g.successors(curr):
                if int(n) in done:
                    continue
                distance = g.edges[(n, curr)].data["w"]
                heapq.heappush(seen, (dist + distance, hops+1, int(n)))

        g.ndata["y"] = torch.cat([self.y_dict[n] for n in self.N])

    @property
    def y(self):
        return torch.stack([self.y_dict[n] for n in self.N], dim=0)


parser = argparse.ArgumentParser(description="")
parser.add_argument("--usenew", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", type=str, default="nsr")
parser.add_argument("--trainsize", type=int, default=10)
parser.add_argument("--testscale", type=float, default=3)
parser.add_argument("--weightscale", type=float, default=3)
parser.add_argument("--trainnumber", type=int, default=10)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--weight", type=int, default=10)

args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True




messageweights_nsr = torch.nn.Linear(2, 1)
sr = MinArchitecture(model="nsr", storage=2, redundancy=10)

messageweights_neg = torch.nn.Linear(2, 10)
aggregate_weight = torch.nn.Linear(10, 1)

edge_mlp = torch.nn.Linear(2, 10, bias=False)#MLPIterGNN(2, 1, 2, bias=False)
score_mlp = torch.nn.Linear(2, 10, bias=False)#MLPIterGNN(2, 1, 2, bias=False)
aggregate_weight_iter = torch.nn.Linear(10, 1, bias=False)#MLPIterGNN(2, 1, 2, bias=False)


def homo_softmax(x, dim=0):
    '''
    Homogenious softmax based on the paper. Normalizes softmax inputs
    '''
    assert (not torch.sum(torch.isnan(x)))
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    assert (not torch.sum(torch.isnan(x_max)))
    x_min = -torch.max(-x, dim=dim, keepdim=True)[0]
    assert (not torch.sum(torch.isnan(x_min)))
    x_diff = (x_max - x_min)
    assert (not torch.sum(torch.isnan(x_diff)))
    zero_mask = (x_diff == 0).type(torch.float)
    x_diff = torch.ones_like(x_diff) * zero_mask + x_diff * (1. - zero_mask)
    x = x / x_diff
    assert (not torch.sum(torch.isnan(x)))
    return F.softmax(x, dim=dim)

def message_plus(edges):
    return {"messages": edges.src["d"] + edges.data["w"]}
def message_nsr(edges):
    return {"messages": messageweights_nsr(torch.cat([edges.src["d"], edges.data["w"]], dim=1))}
def message_neg(edges):
    return {"messages": messageweights_neg(torch.cat([edges.src["d"], edges.data["w"]], dim=1))}
def message_itergnn(edges):
    inputs = torch.cat([edges.src["d"], edges.data["w"]], dim=1)
    e_k = edge_mlp(inputs)
    score_k = score_mlp(inputs)
    return {"m": e_k, "score": score_k}

def aggregate_min(nodes):
    msgs = nodes.mailbox["messages"]
    return {"d": torch.min(msgs, dim=1)[0]}
def aggregate_nsr(nodes):
    msgs = nodes.mailbox["messages"]
    return {"d": torch.cat([sr(m.T) for m in msgs], dim=0)}
def aggregate_neg(nodes):
    msgs = nodes.mailbox["messages"]
    newdata = torch.min(msgs, dim=1)[0]
    return {"d": aggregate_weight(torch.cat([newdata], dim=1))}
def aggregate_itergnn(nodes):
    scores = homo_softmax(nodes.mailbox['score'], dim=1)
    newdata = torch.max(nodes.mailbox['m'], dim=1)[0]
    return {'d': aggregate_weight_iter(newdata)}
    #return {'d': torch.min(nodes.data['d'], aggregate_weight(
    #    torch.sum(nodes.mailbox['m'] * scores, dim=1)))}

parameters = list(messageweights_nsr.parameters()) + list(sr.parameters()) +\
             list(messageweights_neg.parameters()) + list(aggregate_weight.parameters())+\
             list(edge_mlp.parameters()) + list(score_mlp.parameters())


if args.model == "min":
    message = message_plus
    aggregate = aggregate_min
elif args.model == "nsr":
    message = message_nsr
    aggregate = aggregate_nsr
elif args.model == "neg":
    message = message_neg
    aggregate = aggregate_neg
elif args.model == "itergnn":
    message = message_itergnn
    aggregate = aggregate_itergnn

n = args.trainsize
w = args.weight
train = []
for i in range(1):
    g = randomgraph(n)
    task = SSSPTask(g, random.randint(0, n-1), n, w)
    src = g.edges()[0]
    dest = g.edges()[1]
    print(src)
    print(dest)
    print("[{}]".format(",".join(["({},{},{})".format(src[i], dest[i],
                                  int(g.edges[(int(src[i]), int(dest[i]))].data["w"].item()))
                                  for i in range(0, len(src), 2)])))
    print(task.start)
    print(g.ndata["y"])
    train.append(task)

opt = torch.optim.Adam(parameters)

def compute(sssp):
    sssp.g.ndata["d"] = sssp.g.ndata["orig_d"].clone()
    sssp.g.ndata["d"][sssp.start] = torch.zeros(1, )
    sssp.g.ndata["h"] = torch.zeros(sssp.g.number_of_nodes(), 1)
    sssp.g.edata["f"] = sssp.g.edata["w"]
    sssp.g.ndata["x"] = sssp.g.ndata["d"]
    sssp.g.ndata["y"] = sssp.g.ndata["y"]
    for _ in range(sssp.maxhop):
        sssp.g.update_all(message, aggregate)
    return sssp.g.ndata["d"]

def train_model(problems, model):
    opt = torch.optim.Adam(parameters)
    for i in range(1, args.epochs + 1):
        distances = []
        for sssp in problems:
            sssp.g.ndata["d"] = sssp.g.ndata["orig_d"].clone()
            sssp.g.ndata["d"][sssp.start] = torch.zeros(1, )
            sssp.g.ndata["h"] = torch.zeros(sssp.g.number_of_nodes(), 1)
            sssp.g.edata["f"] = sssp.g.edata["w"]
            sssp.g.ndata["x"] = sssp.g.ndata["d"]
            sssp.g.ndata["y"] = sssp.g.ndata["y"]
            g = dgl.from_networkx(sssp.g.to_networkx(node_attrs=["d", "h"], edge_attrs="w"),
                                  node_attrs=["d", "h"], edge_attrs=["w"])
            if model in ["nsr", "neg", "itergnn"]:
                for _ in range (sssp.maxhop):
                    sssp.g.update_all(message, aggregate)
                    g.update_all(message_plus, aggregate_min)
                    error = torch.mean(torch.abs(sssp.g.ndata["d"] - g.ndata["d"]))
                    opt.zero_grad()
                    error.backward()
                    opt.step()
                    sssp.g.ndata["d"] = g.ndata["d"]
                distance = sssp.g.ndata["d"]
            else:
                distance = 0
                1/0
            distances.append(compute(sssp))
        pred = torch.cat(distances, dim=0).squeeze(1)
        y = torch.cat([sssp.g.ndata["y"] for sssp in problems], dim=0)
        error = torch.mean(torch.abs(pred - y))
        print("Epoch", i, "error", error)
        print(torch.round(pred))
        print(y)
    return distances

train_model(train, args.model)


with open("sssp/results/sssptf_{}_{}.csv".format(args.model, args.seed), "w") as f:
    with torch.no_grad():
        sr.start_testing()
        for sizescale in range(0, 7):
            for weightscale in range(0, 7):
                test = []
                new_n = n * 2 ** sizescale
                new_w = w * 2 ** weightscale
                for i in range(10):
                    g = randomgraph(n)
                    test.append(SSSPTask(g, random.randint(0, n - 1), n, new_w))
                pred = [compute(testsssp) for testsssp in test]
                pred = torch.cat(pred, dim=0).squeeze(1)
                y = torch.cat([sssp.g.ndata["y"] for sssp in test], dim=0)
                error = torch.mean(torch.abs(pred - y))
                mean_y = torch.mean(y)
                ss_total = torch.sum((y - mean_y).pow(2))
                ss_error = torch.sum((y - pred).pow(2))
                fvu = ss_error / ss_total
                print("FVU", fvu)
                writestring = "{},{},{},{},{}\n"\
                    .format(args.model, new_n, new_w, args.seed, error, fvu)
                print(writestring[:-1])
                f.write(writestring)