import numpy as np
import random

# allgraph = list(range(1,189,1))
allgraph = list(range(188))
random.shuffle(allgraph)
train_graph_id = allgraph[:150]
val_graph_id = allgraph[150:]
np.savez("/MRF_causal/data/MUTAG/train_graph_id.npz", train_graph_id=train_graph_id)
np.savez("/MRF_causal/data/MUTAG/val_graph_id.npz", val_graph_id=val_graph_id)
graph_node = {} # 图和节点的映射
with open("data/MUTAG/raw/MUTAG_graph_indicator.txt") as f3:
    for (num, value) in enumerate(f3):
        if int(value) not in graph_node:
            graph_node[int(value)]=[]
        if num not in graph_node[int(value)]:
            graph_node[int(value)].append(num)
exit()
node_map_train = []
node_map_val = []
node_map = []
for i in train_graph_id:
    node_map.append(graph_node[i])
    node_map_train.append(graph_node[i])
for j in val_graph_id:
    node_map.append(graph_node[j])
    node_map_val.append(graph_node[j])
node_map_train = sum(node_map_train,[])
node_map_val = sum(node_map_val,[])
node_map = sum(node_map,[])
np.savez("/MRF_causal/data/MUTAG/node_map_train.npz", node_map_train=node_map_train)
np.savez("/MRF_causal/data/MUTAG/node_map_val.npz", node_map_val=node_map_val)

