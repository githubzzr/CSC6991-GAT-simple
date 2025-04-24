import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
import torch.nn.functional as F
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

from graphsage.GATLayer import GATModel

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


class SupervisedGAT(nn.Module):
    def __init__(self, num_classes, enc, dropout=0.5):
        super(SupervisedGAT, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.embed_dim = enc.embed_dim

        # 用 nn.Linear 替代手动 weight.mm
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        init.xavier_uniform_(self.classifier.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes).t()  # [B, F]
        embeds = F.dropout(embeds, p=self.dropout, training=self.training)
        embeds = F.relu(embeds)
        scores = self.classifier(embeds)  # [B, num_classes]
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)  # [B, C]
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora(GAT=False):
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()



    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    if GAT:


        model = GATModel(nfeat=1433, nhid=128, nclass=7, dropout=0.6, alpha=0.2,use_multihead=False)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)

        model.train()


        times = []
        for epoch in range(100):
            batch_nodes = train[:256]
            random.shuffle(train)

            # ------------------ Sample batch neighbors and build subgraphs ------------------
            batch_and_neighbors = set(batch_nodes)
            for node in batch_nodes:
                batch_and_neighbors.update(adj_lists[node])  # include 1-hop neighbors
            batch_and_neighbors = list(batch_and_neighbors)

            # old id → new id mapping
            id_map = {old_id: idx for idx, old_id in enumerate(batch_and_neighbors)}

            # Construct the subgraph feature matrix
            batch_x = features(torch.LongTensor(batch_and_neighbors))

            # Constructing the subgraph adjacency matrix
            sub_adj = torch.zeros(len(batch_and_neighbors), len(batch_and_neighbors))
            for i, old_i in enumerate(batch_and_neighbors):
                sub_adj[i, i] = 1   # Self Loop
                for old_j in adj_lists[old_i]:
                    if old_j in id_map:
                        j = id_map[old_j]
                        sub_adj[i, j] = 1


            batch_label_indices = torch.LongTensor([id_map[nid] for nid in batch_nodes])
            batch_labels = Variable(torch.LongTensor(labels[np.array(batch_nodes)]).squeeze())

            # ------------------ Start training ------------------
            start_time = time.time()
            optimizer.zero_grad()

            out = model(batch_x, sub_adj)
            loss = F.cross_entropy(out[batch_label_indices], batch_labels)

            loss.backward()
            optimizer.step()
            end_time = time.time()

            times.append(end_time - start_time)
            print("Epoch:", epoch, "Loss:", loss.item())


        model.eval()

        val_preds = []
        batch_size = 256

        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                batch_nodes = val[i:i + batch_size]

                batch_and_neighbors = set(batch_nodes)
                for node in batch_nodes:
                    batch_and_neighbors.update(adj_lists[node])
                batch_and_neighbors = list(batch_and_neighbors)

                id_map = {old_id: idx for idx, old_id in enumerate(batch_and_neighbors)}

                batch_x = features(torch.LongTensor(batch_and_neighbors))

                sub_adj = torch.zeros(len(batch_and_neighbors), len(batch_and_neighbors))
                for idx, old_i in enumerate(batch_and_neighbors):
                    sub_adj[idx, idx] = 1  # Self Loop
                    for old_j in adj_lists[old_i]:
                        if old_j in id_map:
                            sub_adj[idx, id_map[old_j]] = 1
                #      # No need to add self-Loop (already added within GATLayer)

                out = model(batch_x, sub_adj)

                batch_idx_map = [id_map[nid] for nid in batch_nodes]
                preds = out[batch_idx_map].data.numpy().argmax(axis=1)

                val_preds.append(preds)

        preds = np.concatenate(val_preds)
        print("Validation F1:", f1_score(labels[val], preds, average="micro"))
        print("Average batch time:", np.mean(times))

    else:
        print("Running GraphSAGE model on Cora...")
        #  Using the original GraphSAGE two-tier structure
        agg1 = MeanAggregator(features, cuda=False)
        enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False)
        enc1.num_samples = 5
        enc2.num_samples = 5
        graphsage = SupervisedGraphSage(num_classes=7, enc=enc2)

    #    graphsage.cuda()


        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
        times = []
        for batch in range(100):
            batch_nodes = train[:256]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes,
                                  Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            print(batch, loss.data.item())

        val_output = graphsage.forward(val)
        print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        print("Average batch time:", np.mean(times))


def load_pubmed():
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed(GAT=False):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()

    features = nn.Embedding(num_nodes, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    if GAT:
        print("Running Sparse GAT model on Pubmed...")

        model = GATModel(nfeat=500, nhid=128, nclass=3, dropout=0.6, alpha=0.2, use_sparse=True,use_multihead=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

        model.train()

        times = []
        for epoch in range(200):
            batch_nodes = train[:1024]
            random.shuffle(train)

            batch_and_neighbors = set(batch_nodes)
            for node in batch_nodes:
                batch_and_neighbors.update(adj_lists[node])
            batch_and_neighbors = list(batch_and_neighbors)

            id_map = {old_id: idx for idx, old_id in enumerate(batch_and_neighbors)}
            batch_x = features(torch.LongTensor(batch_and_neighbors))

            # Build sparse sub_adj
            row_indices = []
            col_indices = []
            for i, old_i in enumerate(batch_and_neighbors):
                for old_j in adj_lists[old_i]:
                    if old_j in id_map:
                        j = id_map[old_j]
                        row_indices.append(i)
                        col_indices.append(j)

            edge_index = torch.LongTensor([row_indices, col_indices])
            edge_values = torch.ones(edge_index.shape[1])
            # Add self-loops
            self_loop_index = torch.arange(len(batch_and_neighbors))
            self_loop_edges = torch.stack([self_loop_index, self_loop_index], dim=0)
            sub_adj = torch.sparse_coo_tensor(
                torch.cat([edge_index, self_loop_edges], dim=1),
                torch.cat([edge_values, torch.ones(len(batch_and_neighbors))]),
                (len(batch_and_neighbors), len(batch_and_neighbors))
            ).coalesce()

            batch_label_indices = torch.LongTensor([id_map[nid] for nid in batch_nodes])
            batch_labels = Variable(torch.LongTensor(labels[np.array(batch_nodes)]).squeeze())

            start_time = time.time()
            optimizer.zero_grad()

            out = model(batch_x, sub_adj)
            loss = F.cross_entropy(out[batch_label_indices], batch_labels)

            loss.backward()
            optimizer.step()
            end_time = time.time()

            times.append(end_time - start_time)
            print("Epoch:", epoch, "Loss:", loss.item())
            scheduler.step()

        # Validation
        model.eval()

        val_preds = []
        batch_size = 1024

        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                batch_nodes = val[i:i + batch_size]

                batch_and_neighbors = set(batch_nodes)
                for node in batch_nodes:
                    batch_and_neighbors.update(adj_lists[node])
                batch_and_neighbors = list(batch_and_neighbors)

                id_map = {old_id: idx for idx, old_id in enumerate(batch_and_neighbors)}
                batch_x = features(torch.LongTensor(batch_and_neighbors))

                # Build sparse sub_adj
                row_indices = []
                col_indices = []
                for idx, old_i in enumerate(batch_and_neighbors):
                    for old_j in adj_lists[old_i]:
                        if old_j in id_map:
                            row_indices.append(idx)
                            col_indices.append(id_map[old_j])

                edge_index = torch.LongTensor([row_indices, col_indices])
                edge_values = torch.ones(edge_index.shape[1])


                # Add self-loops
                self_loop_index = torch.arange(len(batch_and_neighbors))
                self_loop_edges = torch.stack([self_loop_index, self_loop_index], dim=0)
                sub_adj = torch.sparse_coo_tensor(
                    torch.cat([edge_index, self_loop_edges], dim=1),
                    torch.cat([edge_values, torch.ones(len(batch_and_neighbors))]),
                    (len(batch_and_neighbors), len(batch_and_neighbors))
                ).coalesce()

                out = model(batch_x, sub_adj)

                batch_idx_map = [id_map[nid] for nid in batch_nodes]
                preds = out[batch_idx_map].data.numpy().argmax(axis=1)

                val_preds.append(preds)

        preds = np.concatenate(val_preds)
        print("Validation F1:", f1_score(labels[val], preds, average="micro"))
        print("Average batch time:", np.mean(times))




    else:
        print("Running GraphSAGE model on Pubmed...")

        agg1 = MeanAggregator(features, cuda=False)
        enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False)
        enc1.num_samples = 10
        enc2.num_samples = 25

        graphsage = SupervisedGraphSage(3, enc2)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

        times = []
        for batch in range(200):
            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes,
                                  Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            print(batch, loss.data.item())

        val_output = graphsage.forward(val)
        print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        print("Average batch time:", np.mean(times))


def main():
    parser = argparse.ArgumentParser(description="GraphSAGE model runner")


    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'pubmed'], help='Dataset to use (default: cora)')


    parser.add_argument('--gat', action='store_true', help='Enable GAT model (default: False)')

    args = parser.parse_args()

    if args.dataset == 'cora':
        run_cora(GAT=args.gat)
    elif args.dataset == 'pubmed':
        run_pubmed(GAT=args.gat)

if __name__ == "__main__":
    main()