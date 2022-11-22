import math
import numpy as np
import pandas as pd

data = pd. read_csv("fakeData归一化.csv")
x1, x2, x3, x4,y = data['x1'], data['x2'], data['x3'],data['x4'],data['y']
sample_num = len(x1)
X = np.matrix([x1, x2, x3, x4]).T
Y = np.matrix(y).T

betaBar = np.matrix([1.0,0.9,0.8,1.3]).T

Gradient_L_betabar = np.matrix([20,25,14,60]).T
DLj_betaBar = np.matrix([1.2,2.5,1.4,2.5]).T

secondthDl_betaBar = np.matrix([[20,25,14,60],[20,25,14,60],[20,25,14,60],[20,25,14,60]])
secondthDlj_betaBar = np.matrix([[1.2,2.5,1.4,2.5],[1.2,2.5,1.4,2.5],[1.2,2.5,1.4,2.5],[1.2,2.5,1.4,2.5]])


def gradient_Lj(X, Y, nj, beta):
    """
    value transfer among sites
    :param x: shape(1,4)
    :param y: (1)
    :param nj: samples of the site
    :param beta: (4,1)
    :return: (1,4).T
    """
    out = np.matrix([0., 0., 0., 0.])
    for row_num in range(len(X[:, 0])):
        x = X[row_num, :]
        y = Y[row_num]
        out += np.matrix(y - (1 / (1 + math.exp(-x @ beta)))) @ x
    return out.T / nj

def Lj_beta(X, Y, beta):
    """
    :param X:(n,4)
    :param Y:(n,1)
    :param beta:(4,1)
    :return: likehood value(1)
    """
    out = 0
    for row_num in range(len(X[:, 0])):
        x, y = X[row_num, :], Y[row_num]
        # print("\nx",x,"\n beta",beta)
        # print("1+math.exp(x @ beta) :",1+math.exp(x @ beta))
        out += (x @ beta * y) - math.log(1 + math.exp(x @ beta), math.e)
    return out

def L2_tilde_beta(beta):
    return Lj_beta(X, Y, beta) + (Gradient_L_betabar - DLj_betaBar).T @ beta + 0.5 * (beta - betaBar).T @ (secondthDl_betaBar - secondthDlj_betaBar) @ (beta - betaBar)


def Gradient_L2_tilde_beta(beta):
    return gradient_Lj(X, Y, sample_num,beta) + (Gradient_L_betabar - DLj_betaBar) + 0.5 * (((secondthDl_betaBar - secondthDlj_betaBar) @ beta) + ((secondthDl_betaBar - secondthDlj_betaBar).T @ beta))


def argmaxL2_tilde():
    iteration = 500
    lr = .000001
    beta = np.random.rand(4, 1)
    print("\n[local site estimation] start iteration to solve the best beta of likehood function...\n")
    for i in range(iteration):
        betagradient = Gradient_L2_tilde_beta(beta)
        beta = beta + betagradient * lr
        if i % 50 == 0:
            print("iteration:{0}".format(i), "\nbeta:\n", beta)
            print("likehood_value:", math.exp(L2_tilde_beta(beta)))
    return beta

res = argmaxL2_tilde()
print(res)