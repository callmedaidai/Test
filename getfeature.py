import numpy as np

# def get_dataset(name):
#     path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', name)
#     dataset = TUDataset(path, name)
#     dataset.data.edge_attr = None
#
#     if dataset.data.x is None:
#         max_degree = 0
#         degs = []
#         for data in dataset:
#             degs += [degree(data.edge_index[0], dtype=torch.long)]
#             max_degree = max(max_degree, degs[-1].max().item())
#
#         if max_degree < 1000:
#             dataset.transform = T.OneHotDegree(max_degree)
#         else:
#             deg = torch.cat(degs, dim=0).to(torch.float)
#             mean, std = deg.mean().item(), deg.std().item()
#             dataset.transform = NormalizedDegree(mean, std)
#     return dataset
# class NormalizedDegree(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std


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
def one_hot(labels,Label_class):
	one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
	return one_hot_label
Label_class = max(degree)
MUTAG_features = one_hot(degree, Label_class)
np.savez("/MRF_model/data/MUTAG/MUTAG_degree_feature.npz", MUTAG_features=MUTAG_features)
# y = [2,5,6,7,8]
# Label_class = 20
# print ("label",one_hot(y,Label_class))
