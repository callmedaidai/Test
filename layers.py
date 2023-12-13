import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
# np.set_printoptions(threshold=np.nan)
from torch.nn import BatchNorm1d

class StructuralFingerprintLayer(nn.Module):
    """
    adaptive structural fingerprint layer
    """

    def __init__(self, in_features, out_features, dropout, alpha,adj_ad, concat=True,):
        super(StructuralFingerprintLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_ad=adj_ad
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)


    def forward(self, input, adj):
        # print(input.shape, self.W.shape)
        # exit()
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N* N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        s = self.adj_ad
        e = e.cuda()
        s = s.cuda()

        # combine sij and eij
        e = abs(self.W_ei)*e+abs(self.W_si)*s

        zero_vec = -9e15*torch.ones_like(e)
        k_vec=-9e15*torch.ones_like(e)
        adj=adj.cuda()
        # np.set_printoptions(threshold=np.nan)
        attention = torch.where(adj>0 , e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        self.attention_result = attention

        I = torch.ones_like(attention).float().cuda()
        I = torch.mul(I,adj)
        h_prime = torch.matmul(attention, h)
        attention_neg = I - attention  # attention_neg 需要进行归一化
        self.attention_neg = attention_neg

        h_neg = torch.matmul(attention_neg, h)

        if self.concat:
            return F.elu(h_prime), F.elu(h_neg)
        else:
            return h_prime, h_neg

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


