# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File       :  setbgoldbeta.py.py
@Time       :  2021/10/12
@Author     :  Yuanting Ma
@Version    :  1.0
@Site       :  https://github.com/YuantingMaSC
@Contact    :  yuantingma@189.cn
"""

"""
金标准迭代
"""
import math
import numpy as np
import pandas as pd
max_min_scaler = lambda x : (x-np.mean(x))/np.std(x)

data = pd. read_csv("fakedata_generated.csv")
data = data.loc[data.index<10000]
data['x1']=data[['x1']].apply(max_min_scaler)
data['x2']=data[['x2']].apply(max_min_scaler)
data['x0'] = np.random.randint(0,1,len(data['x1']))+1


x0, x1, x2, x3, x4,y = data['x0'],data['x1'], data['x2'], data['x3'],data['x4'],data['y']
sample_num = len(x1)

X = np.matrix([x0,x1, x2, x3, x4]).T
Y = np.matrix(y).T

iteration = 90001
lr = 1
threshhold = 1e-9

def sigmoid(Z):
    # 解决溢出问题
    # 把大于0和小于0的元素分别处理
    # 原来的sigmoid函数是 1/(1+np.exp(-Z))
    # 当Z是比较小的负数时会出现上溢，此时可以通过计算exp(Z) / (1+exp(Z)) 来解决

    mask = (Z > 0)
    positive_out = np.zeros_like(Z, dtype='float64')
    negative_out = np.zeros_like(Z, dtype='float64')

    # 大于0的情况
    positive_out = 1 / (1 + np.exp(-Z, positive_out, where=mask))
    # 清除对小于等于0元素的影响
    positive_out[~mask] = 0

    # 小于等于0的情况
    expZ = np.exp(Z, negative_out, where=~mask)
    negative_out = expZ / (1 + expZ)
    # 清除对大于0元素的影响
    negative_out[mask] = 0

    return positive_out + negative_out

def Lj_beta( X, Y, beta):
    """
    :param X:(n,5)
    :param Y:(n,1)
    :param beta:(5,1)
    :return: likehood value(1)
    """
    out = (np.multiply(X @ beta, Y) - np.log(1 + np.exp(X @ beta))).sum()
    return out


def gradient_Lj( X, Y, nj, beta):
    """
    value transfer among sites
    :param x: shape(1,5)
    :param y: (1)
    :param nj: samples of the site
    :param beta: (5,1)
    :return: (1,5).T
    """
    x_beta = X @ beta
    pij = sigmoid(x_beta)
    grad = ((Y - pij).T @ X) / nj
    return grad.T


def argmaxLj( X, Y, nj):
    """
    Gradient Descent method is used !
    :param X: (n,5)
    :param Y: (n,1)
    :param iteration: iteration num
    :return: beta(4,1)
    """
    # initiate beta (4,1)
    beta = np.random.randn(5, 1)
    for i in range(iteration):
        betagradient = gradient_Lj(X, Y, nj, beta)
        beta = beta + lr * betagradient
        if i % 50 == 0:
            print("iteration:{0}\n beta:\n{1}".format(i, beta))
            print("likehood_value:", np.exp(Lj_beta(X, Y, beta)))
        if sum(abs(lr * betagradient)) / 5 * 500 < threshhold:
            return beta
    return beta

gold_beta = argmaxLj(X, Y, Y.shape[0])
print("res!\n",gold_beta)
