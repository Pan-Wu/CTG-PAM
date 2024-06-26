'''
Description: Model used for graph feature generating and classification.
'''
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm

class LightGCN():
    def __init__(self, K):
        self.K = K
        self.alpha = 1 /(K+1)
    def compute(self, A, D, X):
        D_sum = D.sum(axis=1)
        if D_sum[0] == 0:
            return self.alpha * X
        else:
            ww = np.argwhere(D_sum!=0)
            A = A[ww[:,0]].T[ww[:,0]]
            D = D[ww[:,0]].T[ww[:,0]]
            X = X[ww[:,0]]
        E_i = X
        D2 = sqrtm(D)
        D1 = np.linalg.inv(D2)
        A_tau = np.dot(D1, A)
        A_tau = np.dot(A_tau, D1)
        result = np.zeros(X.shape)
        result += self.alpha * E_i
        for i in range(self.K):
            E_i = np.dot(A_tau, E_i)
            result += self.alpha * E_i
        return result
class CellMLP(nn.Module):
    def __init__(self, n=50, input_dim=24):
        super(CellMLP, self).__init__()
        self.act = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(input_dim, n),
            self.act,
            nn.Linear(n, n),
            self.act,
            nn.Linear(n, n),
            self.act,
            nn.Linear(n, 2),
        )
    def forward(self, x):
        x_final = self.network(x)
        return x_final
