import torch
import torch.nn as nn
import torch.nn.functional as F
from MRF_causal.layers import StructuralFingerprintLayer
from MRF_causal.rwr_process import RWRLayer
from torch_geometric.nn import global_mean_pool, global_add_pool
import numpy as np
import random
from torch.nn import BatchNorm1d
# class MRF(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad):
#         """version of MRF."""
#         super(MRF, self).__init__()
#         self.dropout = dropout
#         self.attentions = [StructuralFingerprintLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad,concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.out_att =StructuralFingerprintLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,adj_ad=adj_ad, concat=False)

    # def forward(self, x, adj, adj_ad):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = F.elu(self.out_att(x, adj))
    #     return F.log_softmax(x, dim=1)
class MRF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad, hidden_emb, graph_node_list):
        """version of MRF."""
        super(MRF, self).__init__()
        self.graph_node_list = graph_node_list
        self.dropout = dropout
        self.mlp4x = nn.Linear(nfeat,nhid)
        self.global_pool = global_add_pool
        self.s = nn.Parameter(torch.FloatTensor(nhid, 1))
        self.s_neg = nn.Parameter(torch.FloatTensor(nhid, 1))
        # self.attentions = [StructuralFingerprintLayer(128, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, concat=True) for _ in range(nheads)]
        self.attentions = [StructuralFingerprintLayer(8, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, concat=True) for _ in range(nheads)]
        # self.attentions = [StructuralFingerprintLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = StructuralFingerprintLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad, concat=False)


    def forward(self, features,hidden_emb, adj, adj_ad):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        hs = []
        # hidden_emb = [h1,h2,h3]

        Z = self.mlp4x(features)
        hidden_emb0 = torch.matmul(adj, Z)
        A2 = torch.matmul(adj, adj)
        A3 = torch.matmul(A2, adj)
        hidden_emb1 = torch.matmul(A2, Z)
        hidden_emb2 = torch.matmul(A3, Z)


        hidden_emb_list = []
        hidden_emb_list.append(hidden_emb0)
        hidden_emb_list.append(hidden_emb1)
        hidden_emb_list.append(hidden_emb2)

        graph_mask = np.zeros((188, 3371, 8))
        indictor = np.ones((3371, 8))
        for i in range(len(self.graph_node_list)):
            tmp = self.sample_mask(self.graph_node_list[i], 3371)
            graph_mask[i][tmp, :] = indictor[tmp, :]
        graph_mask = torch.tensor(graph_mask).cuda() # (188, 3371, 8) 其中 dim=8

        for x,att in zip(hidden_emb_list, self.attentions):
            hs.append(att(torch.tensor(x).cuda(), adj))

        H1 = hs[0][0].cuda()
        H2 = hs[1][0].cuda()
        H3 = hs[2][0].cuda()

        H = torch.stack([Z, H1, H2, H3], axis=1)  # n*c*(k+1)  k = 3层数  c类别数量 n节点个数

        S = torch.sigmoid(torch.matmul(H, self.s.cuda()))
        S_ = torch.reshape(S, [S.shape[0], 1, 4])  # n*1*(k+1)

        # X_out = self.global_pool(torch.squeeze(torch.matmul(S_, H)), S_.shape[0])
        # X_out = torch.log_softmax(X_out, dim=1)  # 1*dim
        # new
        H = torch.squeeze(torch.matmul(S_, H))   # 3371*8 dim
        H = torch.unsqueeze(H, 0) # (1,3371,8)
        H = torch.mul(graph_mask, H) # (188, 3371, 8) * (1,3371,8) = (188,3371,8) 【对应位置相乘】
        X_out = torch.sum(H, dim=1) # （188, 8）dim = 8

        X_out = torch.log_softmax(X_out, dim=1)

        H1_neg = hs[0][1].cuda()
        H2_neg = hs[1][1].cuda()
        H3_neg = hs[2][1].cuda()

        H_neg = torch.stack([Z, H1_neg, H2_neg, H3_neg], axis=1)

        S_neg = torch.sigmoid(torch.matmul(H_neg, self.s_neg.cuda()))
        S_neg_ = torch.reshape(S_neg, [S_neg.shape[0], 1, 4])  # n*1*(k+1)
        # X_out_neg = self.global_pool(torch.squeeze(torch.matmul(S_neg_, H_neg)), S_neg_.shape[0])
        # X_out_neg = torch.log_softmax(X_out_neg, dim=1)  # 1*dim
        # new
        H_neg = torch.squeeze(torch.matmul(S_neg_, H_neg))  # 3371*8 dim
        H_neg = torch.unsqueeze(H_neg, 0)  # (1,3371,8)
        H_neg = torch.mul(graph_mask, H_neg)  # (188, 3371, 8) * (1,3371,8) = (188,3371,8) 【对应位置相乘】
        X_out_neg = torch.sum(H_neg, dim=1)  # （188, 8）dim = 8

        X_out_neg = torch.log_softmax(X_out_neg, dim=1)

        # 打乱的h H + H_neg 随机
        l = [i for i in range(188)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        H_rand = H_neg[random_idx] + H
        X_out_rand = torch.sum(H_rand, dim=1)

        X_out_rand = torch.log_softmax(X_out_rand, dim=1)

        return X_out, X_out_neg, X_out_rand

    def ablation(self, features,hidden_emb, adj, adj_ad,part):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        hs = []
        # hidden_emb = [h1,h2,h3]

        Z = self.mlp4x(features)
        hidden_emb0 = torch.matmul(adj, Z)
        A2 = torch.matmul(adj, adj)
        A3 = torch.matmul(A2, adj)
        hidden_emb1 = torch.matmul(A2, Z)
        hidden_emb2 = torch.matmul(A3, Z)


        hidden_emb_list = []
        hidden_emb_list.append(hidden_emb0)
        hidden_emb_list.append(hidden_emb1)
        hidden_emb_list.append(hidden_emb2)

        graph_mask = np.zeros((188, 3371, 8))
        indictor = np.ones((3371, 8))
        for i in range(len(self.graph_node_list)):
            tmp = self.sample_mask(self.graph_node_list[i], 3371)
            graph_mask[i][tmp, :] = indictor[tmp, :]
        graph_mask = torch.tensor(graph_mask).cuda() # (188, 3371, 8) 其中 dim=8

        for x,att in zip(hidden_emb_list, self.attentions):
            hs.append(att(torch.tensor(x).cuda(), adj))

        H1 = hs[0][0].cuda()
        H2 = hs[1][0].cuda()
        H3 = hs[2][0].cuda()

        if part ==0:
            H = torch.stack([torch.zeros_like(Z), H1, H2, H3], axis=1)  # n*c*(k+1)  k = 3层数  c类别数量 n节点个数
        if part ==1:
            H = torch.stack([Z, torch.zeros_like(H1), H2, H3], axis=1)
        if part ==2:
            H = torch.stack([Z, H1, torch.zeros_like(H2), H3], axis=1)
        if part ==3:
            H = torch.stack([Z, H1, H2, torch.zeros_like(H3)], axis=1)

        S = torch.sigmoid(torch.matmul(H, self.s.cuda()))
        S_ = torch.reshape(S, [S.shape[0], 1, 4])  # n*1*(k+1)

        # X_out = self.global_pool(torch.squeeze(torch.matmul(S_, H)), S_.shape[0])
        # X_out = torch.log_softmax(X_out, dim=1)  # 1*dim
        # new
        H = torch.squeeze(torch.matmul(S_, H))   # 3371*8 dim
        H = torch.unsqueeze(H, 0) # (1,3371,8)
        H = torch.mul(graph_mask, H) # (188, 3371, 8) * (1,3371,8) = (188,3371,8) 【对应位置相乘】
        X_out = torch.sum(H, dim=1) # （188, 8）dim = 8

        X_out = torch.log_softmax(X_out, dim=1)

        H1_neg = hs[0][1].cuda()
        H2_neg = hs[1][1].cuda()
        H3_neg = hs[2][1].cuda()


        if part ==0:
            H_neg = torch.stack([torch.zeros_like(Z), H1_neg, H2_neg, H3_neg], axis=1)  # n*c*(k+1)  k = 3层数  c类别数量 n节点个数
        if part ==1:
            H_neg = torch.stack([Z, torch.zeros_like(H1_neg), H2_neg, H3_neg], axis=1)
        if part ==2:
            H_neg = torch.stack([Z, H1_neg, torch.zeros_like(H2_neg), H3_neg], axis=1)
        if part ==3:
            H_neg = torch.stack([Z, H1_neg, H2_neg, torch.zeros_like(H3_neg)], axis=1)
        # H_neg = torch.stack([Z, H1_neg, H2_neg, H3_neg], axis=1)

        S_neg = torch.sigmoid(torch.matmul(H_neg, self.s_neg.cuda()))
        S_neg_ = torch.reshape(S_neg, [S_neg.shape[0], 1, 4])  # n*1*(k+1)
        # X_out_neg = self.global_pool(torch.squeeze(torch.matmul(S_neg_, H_neg)), S_neg_.shape[0])
        # X_out_neg = torch.log_softmax(X_out_neg, dim=1)  # 1*dim
        # new
        H_neg = torch.squeeze(torch.matmul(S_neg_, H_neg))  # 3371*8 dim
        H_neg = torch.unsqueeze(H_neg, 0)  # (1,3371,8)
        H_neg = torch.mul(graph_mask, H_neg)  # (188, 3371, 8) * (1,3371,8) = (188,3371,8) 【对应位置相乘】
        X_out_neg = torch.sum(H_neg, dim=1)  # （188, 8）dim = 8

        X_out_neg = torch.log_softmax(X_out_neg, dim=1)

        # 打乱的h H + H_neg 随机
        l = [i for i in range(188)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        H_rand = H_neg[random_idx] + H
        X_out_rand = torch.sum(H_rand, dim=1)

        X_out_rand = torch.log_softmax(X_out_rand, dim=1)

        return X_out, X_out_neg, X_out_rand

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

class RWR_process(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj_ad):
        """version of RWR_process."""
        super(RWR_process, self).__init__()
        self.dropout = dropout

        self.attentions = [RWRLayer(nfeat, nhid, dropout=dropout, alpha=alpha, adj_ad=adj_ad, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = RWRLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, adj_ad=adj_ad,concat=False)

    def forward(self, x, adj, adj_ad):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

