import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import pickle

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
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


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the structural fingerprints for citeseer"""
    for i in range(len(ri_index)):
        for j in range(len(ri_index)):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            g[i][j] = 0
            if len(intersection) == 0:
                g[i][j] = 0.0001
                break
            else:
                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num

    return g

def load_data(dataset_str):
      # """Load data."""
    if dataset_str == 'MUTAG':
        # graph 的idx
        orin_val = np.load("data/MUTAG/val_graph_id.npz")
        orin_train = np.load("data/MUTAG/train_graph_id.npz")
        test_idx_reorder = orin_val["val_graph_id"]
        test_idx_range = test_idx_reorder
        train_idx_reorder = orin_train["train_graph_id"]

        MUTAG = np.load('data/MUTAG/MUTAG_degree_feature.npz')  #  MUTAG_features
        # MUTAG_feature = MUTAG["features"]
        MUTAG_feature = MUTAG["MUTAG_features"]
        features = sp.lil_matrix(MUTAG_feature)
        features[test_idx_reorder, :] = features[test_idx_range, :]

        MUTAG_graph = {}
        with open("data/MUTAG/raw/MUTAG_A.txt") as f1:
            for line1 in f1.readlines():
                node1 = int(line1.split(',')[0])
                node2 = int(line1.split(',')[1])
                if node1 not in MUTAG_graph:
                    MUTAG_graph[node1] = []
                if node2 not in MUTAG_graph[node1]:
                    MUTAG_graph[node1].append(node2)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(MUTAG_graph))
        adj = adj.astype(np.float32)
        adj_ad1 = adj
        adj_sum_ad = np.sum(adj_ad1, axis=0)
        adj_sum_ad = np.asarray(adj_sum_ad)
        adj_sum_ad = adj_sum_ad.tolist()
        adj_sum_ad = adj_sum_ad[0]
        adj_ad_cov = adj
        Mc = adj_ad_cov.tocoo()
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        adj_delta = adj

        def one_hot(labels, Label_class):
            one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
            return one_hot_label

        MUTAG_graph_label = []
        with open("data/MUTAG/raw/MUTAG_graph_labels.txt") as f_graph:
            for graphline in f_graph.readlines():
                MUTAG_graph_label.append(int(graphline))
        # graph_Label_class = max(MUTAG_graph_label)+1
        graph_Label_class = 2
        labels = one_hot(MUTAG_graph_label, graph_Label_class)

        idx_test = test_idx_reorder
        idx_train = train_idx_reorder
        idx_val = test_idx_reorder

        train_mask = 1
        val_mask = 1
        test_mask = 1

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        # print(adj_delta)
        G = nx.DiGraph()
        inf = pickle.load(open('data/MUTAG/adj_MUTAG.pkl', 'rb'))
        for i in range(len(inf)):
          for j in range(len(inf[i])):
              G.add_edge(i, inf[i][j], weight=1)

        a = open("data/MUTAG/dijskra_MUTAG.pkl", 'wb')
        pickle.dump(adj_delta, a)
        #######


        fw = open('ri_index_c_0.5_MUTAG_highorder_1_x_abs.pkl', 'rb')
        ri_index = pickle.load(fw)
        fw.close()

        fw = open('ri_all_c_0.5_MUTAG_highorder_1_x_abs.pkl', 'rb')
        ri_all = pickle.load(fw)
        fw.close()
        # Evaluate structural interaction between the structural fingerprints of node i and j
        adj_delta = structural_interaction(ri_index, ri_all, adj_delta)
        # print("adj_delta", adj_delta)

        # labels = torch.LongTensor(np.where(labels)[1])
        labels = torch.tensor(np.argmax(labels, axis=1))
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        return adj, features,idx_train, idx_val, idx_test, train_mask, val_mask, test_mask, labels, adj_delta

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

