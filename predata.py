import networkx as nx
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import pickle
import torch

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

# -----------------------------------------------------
# MUTAG 数据集处理得到adj
MUTAG_graph = {}
with open("data/MUTAG/raw/MUTAG_A.txt") as f1:
	for line1 in f1.readlines():
		node1 = int(line1.split(',')[0])
		node2 = int(line1.split(',')[1])
		if node1 not in MUTAG_graph:
			MUTAG_graph[node1] = []
		if node2 not in MUTAG_graph[node1]:
			MUTAG_graph[node1].append(node2)
print("MUTAG_graph", MUTAG_graph)
# -----------------------------------------------------
# one-hot 使用示例
def one_hot(labels,Label_class):
	one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
	return one_hot_label
# y = [2,5,6,7,8]
# Label_class = 20
# print ("label",one_hot(y,Label_class))
# MUTAG 数据集处理得到节点的label
MUTAG_label = []
with open("data/MUTAG/raw/MUTAG_node_labels.txt") as f2:
	for line2 in f2.readlines():
		MUTAG_label.append(int(line2))
Label_class = max(MUTAG_label)+1
labels = one_hot(MUTAG_label,Label_class)
# MUTAG 数据集处理得到图的label

MUTAG_graph_label = []
with open("data/MUTAG/raw/MUTAG_graph_labels.txt") as f_graph:
	for graphline in f_graph.readlines():
		MUTAG_graph_label.append(int(graphline))
graph_Label_class = max(MUTAG_graph_label)+1
labels = one_hot(MUTAG_graph_label,Label_class)



# -----------------------------------------------------
# MUTAG 数据集处理得到feature
N = len(MUTAG_graph)
adj = np.zeros((N,N))
with open("data/MUTAG/raw/MUTAG_A.txt") as f1:
	for line1 in f1.readlines():
		node1 = int(line1.split(',')[0])
		node2 = int(line1.split(',')[1])
		adj[node1-1][node2-1] = 1
print(adj)
degree = (adj.sum(axis=1)).astype(np.int32)
print(min(degree), max(degree))
# def one_hot(labels,Label_class):
# 	one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
# 	return one_hot_label
Label_class = max(degree)
MUTAG_features = one_hot(degree, Label_class)
np.savez("/MRF_model/data/MUTAG/MUTAG_degree_feature.npz", MUTAG_features=MUTAG_features)
# 存三份
MUTAG_features_float = MUTAG_features.astype(np.float32)
np.savez("/MRF_model/data/MUTAG/test_features.npz", h0=MUTAG_features_float, h1=MUTAG_features_float, h2=MUTAG_features_float)




exit()
MUTAG = np.load('data/MUTAG/MUTAG_degree_feature.npz')
MUTAG_feature = MUTAG["MUTAG_features"]
features = sp.lil_matrix(MUTAG_feature) # 变为稀疏矩阵
# A=np.array([[1,0,2,0],[0,0,0,0],[3,0,0,0],[1,0,0,4]])
# AS=sp.lil_matrix(A)
# np.savez("/MRF_model/data/MUTAG/test_features.npz", h0=MUTAG_feature, h1=MUTAG_feature, h2=MUTAG_feature)
# -----------------------------------------------------
# 处理soft mask过来的数据
index_all = np.load('data/MUTAG/index.npz')
# print(index_all.files)
train_idx, val_idx = index_all['trian_idx'],index_all['val_idx'] # graph的划分
graph_node = {} # 图和节点的映射
with open("data/MUTAG/raw/MUTAG_graph_indicator.txt") as f3:
	for (num, value) in enumerate(f3):
		if int(value) not in graph_node:
			graph_node[int(value)]=[]
		if num not in graph_node[int(value)]:
			graph_node[int(value)].append(num)

node_map = []
for i in train_idx:
	node_map.append(graph_node[i+1])
for j in val_idx:
	node_map.append(graph_node[j+1])
node_map = sum(node_map,[])
# print(node_map)

hidden = np.load('data/MUTAG/Layer_hidden.npz')
h0, h1, h2 = hidden['h0'],hidden['h1'],hidden['h2']  # train+val--> node  在后面做了排序
# ----------------------------------------
# # 排序demo例子
# import operator
# list_=[]
# dict_data={6:[9,10],10:[5,55],3:[11,89],8:[2,66],7:[6,77]}
# test_data_4=sorted(dict_data.items(),key=operator.itemgetter(0))
# for i in range(len(test_data_4)):
# 	t = test_data_4[i][1]
# 	list_.append(t)
# print(list_)
# print(np.array(list_))
# np.savez("/MRF_model/data/MUTAG/sort_hidden.npz", list_)
# print(type(test_data_4),test_data_4)
# print(t)
# ----------------------------------------
# # 根据idx排序后的hidden结果
# list_h0 = []
# node_h0 = {}
# for i in range(len(node_map)):
# 	node_h0[node_map[i]] = h0[i]
# hidden0 = sorted(node_h0.items(),key=operator.itemgetter(0))
# for j in range(len(hidden0)):
# 	t0 = hidden0[j][1]
# 	list_h0.append(t0)
# np.savez("/MRF_model/data/MUTAG/sort_hidden0.npz", list_h0)
#
# list_h1 = []
# node_h1 = {}
# for i in range(len(node_map)):
# 	node_h1[node_map[i]] = h1[i]
# hidden1 = sorted(node_h1.items(),key=operator.itemgetter(0))
# for j in range(len(hidden1)):
# 	t1 = hidden1[j][1]
# 	list_h1.append(t1)
# np.savez("/MRF_model/data/MUTAG/sort_hidden1.npz", list_h1)
#
# list_h2 = []
# node_h2 = {}
# for i in range(len(node_map)):
# 	node_h2[node_map[i]] = h2[i]
# hidden2 = sorted(node_h2.items(),key=operator.itemgetter(0))
# for j in range(len(hidden2)):
# 	t2 = hidden0[j][1]
# 	list_h2.append(t2)
# np.savez("/MRF_model/data/MUTAG/sort_hidden2.npz", list_h2)
sh1 = np.load('data/MUTAG/sort_hidden0.npz')
sh2 = np.load('data/MUTAG/sort_hidden1.npz')
sh3 = np.load('data/MUTAG/sort_hidden2.npz')
np.savez("/MRF_model/data/MUTAG/sort_Layer_hidden.npz", h0=sh1['arr_0'], h1=sh2['arr_0'], h2=sh3['arr_0'])
# ----------------------------------------
# 生成 dijskra_MUTAG.pkl 文件
MUTAG_adj = []
for key,value in MUTAG_graph.items():
    # print(key,value)
    MUTAG_adj.append(value)
print("MUTAG_adj",MUTAG_adj)
# fw = open('/MRF_model/data/MUTAG/adj_MUTAG.pkl','wb')
# pickle.dump(MUTAG_adj,fw)

# 后面处理
adj = nx.adjacency_matrix(nx.from_dict_of_lists(MUTAG_graph))
adj=adj.astype(np.float32)
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
# caculate n-hop neighbors
G = nx.DiGraph()
inf= pickle.load(open('data/MUTAG/adj_MUTAG.pkl', 'rb'))
for i in range(len(inf)):
    for j in range(len(inf[i])):
      G.add_edge(i, inf[i][j], weight=1)
print(G.nodes)
from tqdm import tqdm
for i in tqdm(range(3371)):
      for j in range(3371):
          try:
              rs = nx.astar_path_length \
                      (
                      G,
                      i,
                      j,
                  )
          except nx.NetworkXNoPath:
             rs = 0
          if rs == 0:
              length = 0
          else:
              # print(rs)
              # length = len(rs)
              length = rs
          adj_delta[i][j] = length
a = open("/MRF_model/data/MUTAG/dijskra_MUTAG.pkl", 'wb')
pickle.dump(adj_delta, a)

exit()
# -------------------------------------------------------------------------
# citeseer 数据集示例，方便处理其他数据集
dataset_str = "citeseer"
names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
for i in range(len(names)):
    with open("data/citeseer/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)
test_idx_reorder = parse_index_file("data/citeseer/ind.{}.test.index".format(dataset_str))
test_idx_range = np.sort(test_idx_reorder)
print(graph)

if dataset_str == 'citeseer':
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

features = sp.vstack((allx, tx)).tolil()
features[test_idx_reorder, :] = features[test_idx_range, :]
adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
