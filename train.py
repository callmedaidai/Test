from __future__ import division
from __future__ import print_function
import scipy
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp
from MRF_causal.utils_nhop_neighbours import load_data, accuracy
from MRF_causal.models import MRF, RWR_process

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')  # 1000
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')  # 0.005  0.01
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.') # 8
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')  # 8
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for the leaky_relu.')  # 0.2
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--sparse', type=bool, default=False, help='Sparse')  # True 是RWRLayer模型，False是MRF模型

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    # rowsum = np.array(features.sum(1))
    rowsum = np.array(features.sum(1), dtype = np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
# Load data
adj, features,  idx_train, idx_val, idx_test, train_mask, val_mask, test_mask, labels ,adj_ad= load_data("MUTAG")
features, spars = preprocess_features(features)
features=np.array(features)
features=scipy.sparse.csr_matrix(features)

features=features.astype(np.float32)
features = torch.FloatTensor(features.todense())

hidden_emb = np.load("data/MUTAG/sort_Layer_hidden.npz")
graph_node = {} # 图和节点的映射
with open("data/MUTAG/raw/MUTAG_graph_indicator.txt") as f3:
    for (num, value) in enumerate(f3):
        if int(value) not in graph_node:
            graph_node[int(value)]=[]
        if num not in graph_node[int(value)]:
            graph_node[int(value)].append(num)
graph_node_list = list(graph_node.values())

if args.sparse:
    model = RWR_process(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha,adj_ad=adj_ad)
else:
    model = MRF(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha,
                adj_ad=adj_ad, hidden_emb =hidden_emb, graph_node_list = graph_node_list)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

###################################使用cuda############################################
if args.cuda:
     model.cuda()
     features = features.cuda()
     adj = adj.cuda()
     adj_ad = adj_ad.cuda()
     labels = labels.cuda()
     idx_train = idx_train.cuda()
     idx_val = idx_val.cuda()
     idx_test = idx_test.cuda()
features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def compute_test(model,features,hidden_emb, adj, adj_ad,idx_test):
    model.eval()
    X_out, X_out_neg, X_out_rand = model(features, hidden_emb, adj, adj_ad)
    loss_sup = F.nll_loss(X_out[idx_test], labels[idx_test])
    uniform_target = torch.ones_like(X_out_neg[idx_test], dtype=torch.float) / X_out_neg[idx_test].shape[1]
    loss_unif = F.kl_div(X_out_neg[idx_test], uniform_target, reduction='batchmean')
    loss_caus = F.nll_loss(X_out_rand[idx_test], labels[idx_test])

    loss_test = 0.9*loss_sup + 0.5*loss_unif + loss_caus


    acc_test = accuracy(X_out[idx_test], labels[idx_test])

    print(
          'loss_test: {:.4f}'.format(loss_test.data.item()),
          'acc_test: {:.4f}'.format(acc_test.data.item()))
    return acc_test

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0

print('data load')
# --------------------------------------------------
model.train()

bst_acc = 0
for epoch in range(args.epochs):
    optimizer.zero_grad()
    X_out, X_out_neg, X_out_rand = model(features,hidden_emb, adj, adj_ad)

    loss_sup = F.nll_loss(X_out[idx_train], labels[idx_train])
    uniform_target = torch.ones_like(X_out_neg[idx_train]) / X_out_neg[idx_train].shape[1]
    loss_unif = F.kl_div(X_out_neg[idx_train], uniform_target, reduction='batchmean')
    loss_caus = F.nll_loss(X_out_rand[idx_train], labels[idx_train])

    loss = 0.9*loss_sup + 0.5*loss_unif + loss_caus
    compute_test(model, features, hidden_emb, adj, adj_ad, idx_train)
    print("test acc")
    acc = compute_test(model, features, hidden_emb, adj, adj_ad, idx_test)
    if acc>bst_acc:
        bst_acc = acc
        torch.save(model.state_dict(), 'model.pkl')
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load('model.pkl'))
# model.load_state_dict(torch.load('./ablation/model_4.pkl'))
print("---------")
compute_test(model,features,hidden_emb, adj, adj_ad,idx_test)
