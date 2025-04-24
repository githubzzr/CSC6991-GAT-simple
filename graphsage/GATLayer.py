import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear transformation: input features -> output features
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        # h: node feature matrix, shape [N, in_features]
        # adj: Adjacency matrix, shape [N, N]


        Wh = torch.mm(h, self.W)

        N = Wh.size(0)

        # Splice each pair of node features
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]

        # Use mask on neighboring nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax
        attention = F.softmax(attention, dim=1)

        attention = self.dropout(attention)


        h_prime = torch.matmul(attention, Wh)

        return h_prime

class SparseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(SparseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        # h: [N, in_features]
        # adj: sparse adjacency matrix [N, N]

        Wh = torch.mm(h, self.W)  # [N, out_features]

        # Get the edges indices (i, j) from the sparse adjacency matrix
        edge_index = adj.coalesce().indices()  # [2, E], where E = number of edges
        row, col = edge_index[0], edge_index[1]

        # Compute attention scores only for existing edges
        Wh_i = Wh[row]  # [E, out_features]
        Wh_j = Wh[col]  # [E, out_features]

        a_input = torch.cat([Wh_i, Wh_j], dim=1)  # [E, 2 * out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))  # [E]

        # Compute softmax for each node's outgoing edges
        # First, for numerical stability, shift e by max for each node
        e_rowmax = torch.zeros_like(Wh[:, 0]).index_add_(0, row, e)
        e_exp = torch.exp(e - e_rowmax[row])
        e_exp_sum = torch.zeros_like(Wh[:, 0]).index_add_(0, row, e_exp)
        attention = e_exp / (e_exp_sum[row] + 1e-16)  # [E]

        attention = self.dropout(attention)

        # Message passing: weighted sum of neighbor features
        h_prime = torch.zeros_like(Wh)
        h_prime.index_add_(0, row, attention.unsqueeze(1) * Wh_j)

        return h_prime






class GATModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, use_multihead=False, use_sparse=False, nheads=8):
        super(GATModel, self).__init__()
        self.use_multihead = use_multihead
        self.use_sparse = use_sparse

        if self.use_multihead and self.use_sparse:
            self.gat1 = SparseMultiHeadGATLayer(nfeat, nhid, num_heads=nheads, dropout=dropout, alpha=alpha, concat=True)
            self.gat2 = SparseMultiHeadGATLayer(nhid * nheads, nclass, num_heads=1, dropout=dropout, alpha=alpha, concat=False)
        elif self.use_multihead:
            self.gat1 = MultiHeadGATLayer(nfeat, nhid, num_heads=nheads, dropout=dropout, alpha=alpha, concat=True)
            self.gat2 = MultiHeadGATLayer(nhid * nheads, nclass, num_heads=1, dropout=dropout, alpha=alpha, concat=False)
        elif self.use_sparse:
            self.gat1 = SparseGATLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            self.gat2 = SparseGATLayer(nhid, nclass, dropout=dropout, alpha=alpha)
        else:
            self.gat1 = GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            self.gat2 = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # if not self.use_sparse:
        #     adj = adj + torch.eye(adj.size(0), device=adj.device) # self-loop for dense adj
        # else:
        #     pass

        # N = adj.size(0)
        # self_loop = torch.eye(N, device=adj.device).to_sparse()
        # adj = (adj + self_loop).coalesce() # self-loop for sparse adj

        x = self.gat1(x, adj)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.gat2(x, adj)
        return x

    def loss(self, nodes, labels, adj):
        output = self.forward(nodes, adj)
        return F.cross_entropy(output, labels)


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat   # True = concat multiple outputs; False = averaging

        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(size=(in_features, out_features)))
            for _ in range(num_heads)
        ])
        self.a = nn.ParameterList([
            nn.Parameter(torch.empty(size=(2 * out_features, 1)))
            for _ in range(num_heads)
        ])

        for w in self.W:
            nn.init.xavier_uniform_(w.data, gain=1.414)
        for a in self.a:
            nn.init.xavier_uniform_(a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        N = h.size(0)
        outputs = []

        for head in range(self.num_heads):
            Wh = torch.mm(h, self.W[head])

            a_input = torch.cat([
                Wh.repeat(1, N).view(N * N, -1),
                Wh.repeat(N, 1)
            ], dim=1).view(N, N, 2 * self.out_features)

            e = self.leakyrelu(torch.matmul(a_input, self.a[head]).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = self.dropout(attention)

            h_prime = torch.matmul(attention, Wh)
            outputs.append(h_prime)

        if self.concat:
            return torch.cat(outputs, dim=1)
        else:
            return torch.mean(torch.stack(outputs), dim=0)

class SparseMultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6, alpha=0.2, concat=True):
        super(SparseMultiHeadGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        self.W = nn.ParameterList([nn.Parameter(torch.empty(size=(in_features, out_features))) for _ in range(num_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.empty(size=(2 * out_features, 1))) for _ in range(num_heads)])

        for w in self.W:
            nn.init.xavier_uniform_(w.data, gain=1.414)
        for a in self.a:
            nn.init.xavier_uniform_(a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        edge_index = adj.coalesce().indices()  # [2, E]
        row, col = edge_index[0], edge_index[1]
        outputs = []

        for head in range(self.num_heads):
            Wh = torch.mm(h, self.W[head])  # [N, out_features]
            Wh_i = Wh[row]  # [E, out_features]
            Wh_j = Wh[col]  # [E, out_features]

            a_input = torch.cat([Wh_i, Wh_j], dim=1)  # [E, 2*out_features]
            e = self.leakyrelu(torch.matmul(a_input, self.a[head]).squeeze(1))  # [E]

            e_rowmax = torch.zeros_like(Wh[:, 0]).index_add_(0, row, e)
            e_exp = torch.exp(e - e_rowmax[row])
            e_exp_sum = torch.zeros_like(Wh[:, 0]).index_add_(0, row, e_exp)
            attention = e_exp / (e_exp_sum[row] + 1e-16)
            attention = self.dropout(attention)

            h_prime = torch.zeros_like(Wh)
            h_prime.index_add_(0, row, attention.unsqueeze(1) * Wh_j)  # [N, out_features]
            outputs.append(h_prime)

        if self.concat:
            return torch.cat(outputs, dim=1)  # [N, out_features * heads]
        else:
            return torch.mean(torch.stack(outputs, dim=0), dim=0)  # [N, out_features]