# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File       :  algorithm.py
@Time       :  2021/10/12
@Author     :  Yuanting Ma
@Version    :  1.0
@Site       :  https://github.com/YuantingMaSC
@Contact    :  yuantingma@189.cn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import math

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
iteration = 100001
lr = 1
threshhold = 1e-7


def max_min_scaler(x):
    return (x - np.mean(x)) / np.std(x)


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


class Site():
    def __init__(self, x0, x1, x2, x3, x4, y, site='site'):
        """
        :param x1: age
        :param x2: weight
        :param x3: binary variables
        :param x4: binary variables
        :param y: binary variables
        """
        # self.localParameter = np.random.rand(4, 1)  ##the final parameter beta that should be estimated in local site
        # x1 = max_min_scaler(x1)
        # x2 = max_min_scaler(x2)
        self.X = np.matrix([x0, x1, x2, x3, x4]).T
        self.Y = np.matrix(y).T
        self.sample_num = x0.shape[0]
        if site == 'site':  # local 节点计算，其他节点不计算，节约时间
            self.betaBar = np.matrix([0, 0, 0, 0, 0]).T
            self.DLj_betaBar = np.matrix(np.zeros(shape=(5, 5)))
            self.secondthDlj_betaBar = np.matrix(np.zeros(shape=(5, 5)))

        else:
            if site == 'local':
                self.betaBar = self.argmaxLj(self.X, self.Y, )  # 每个站点都会生成自己本地的beta,非local不计算，local要调用一次本函数
                self.DLj_betaBar = self.gradient_Lj(self.X, self.Y, self.betaBar)
                self.secondthDlj_betaBar = self.secondD_Lj(self.X, self.Y, self.betaBar)
            else:
                print("site category error !")
        self.localN = 0
        self.transfered_local_betabar = np.matrix([0, 0, 0, 0, 0]).T
        self.L_beta = 0.
        # build temporary variable to store the info from other sites (nj * gradient_Lj)
        self.storage = np.matrix([0, 0, 0, 0, 0]).T
        self.Gradient_L_betabar = np.matrix([0., 0., 0., 0., 0.]).T
        ## odal2 need transfer 2th gradient lj of site
        self.storage2 = np.matrix(np.zeros(shape=(5, 5)))
        self.secondthDl_betaBar = np.matrix(np.zeros(shape=(5, 5)))

    def Lj_beta(self, X, Y, beta):
        """
        :param X:(n,5)
        :param Y:(n,1)
        :param beta:(5,1)
        :return: likehood value(1)
        """
        out = (np.multiply(X @ beta, Y) - np.log(1 + np.exp(X @ beta))).sum()
        return out

    def gradient_Lj(self, X, Y, beta):
        """
        value transfer among sites
        :param x: shape(n,5)
        :param y: (n,1)
        :param beta: (5,1)
        :return: (1,5).T
        """
        x_beta = X @ beta
        pij = sigmoid(x_beta)
        grad = (X.T @ (Y - pij)) / X.shape[0]
        return grad

    def argmaxLj(self, X, Y):
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
            betagradient = self.gradient_Lj(X, Y, beta)
            beta = beta + lr * betagradient
            # if i % 500 == 0:
            #     print("iteration:{0}\n beta:\n{1}".format(i, beta))
            #     print("likehood_value:", np.exp(self.Lj_beta(X, Y, beta)))
            if sum(abs(lr * betagradient)) / 5 * 500 < threshhold:
                return beta
        return beta

    def Transfer(self, other, N):
        """
        local site summerize the DL_betabar to estimate the beta
        the communication among sites should be watched
        :param other: other site
        :return: the accumulative storage of info, the the final value
        """
        other.transfered_local_betabar = self.betaBar  # 把local的beta传到其他site
        other.L_beta = other.Lj_beta(other.X, other.Y, other.transfered_local_betabar)
        other.DLj_betaBar = other.gradient_Lj(other.X, other.Y, other.transfered_local_betabar)
        other.storage = other.sample_num / N * other.DLj_betaBar

        self.storage += other.storage
        self.localN += other.sample_num
        return self.storage

    def cul_gradient_L_betabar(self, N):
        """
        :param N: the total number of sites should be input else
        :return:
        """
        self.Gradient_L_betabar = self.storage / N
        return self.Gradient_L_betabar

    def L_tilde_beta(self, localParameter, ):
        return self.Lj_beta(self.X, self.Y, localParameter) + (
                    self.Gradient_L_betabar.T - self.DLj_betaBar.T) @ localParameter

    def Gradient_L_tilde_beta(self, X, Y, localParameter):
        """
        gradient of likehood in local site
        :param x: shape(1,5)
        :param y: (1)
        :param nj: samples of the site
        :param beta: (5,1)
        :return: (1,5)
        """
        return self.gradient_Lj(X, Y, localParameter) + self.Gradient_L_betabar - self.DLj_betaBar

    def argmax_L_tilde_beta(self, initial_beta=np.matrix([-8, -5, 1.5, 1, 0.5]).T):
        beta = initial_beta  # 初始值
        # print("\n[local site estimation ODAL1] start iteration to solve the best beta of likehood function...\n")
        for i in range(iteration):
            betagradient = self.Gradient_L_tilde_beta(self.X, self.Y, beta)
            beta = beta + betagradient * lr
            # if i % 500 == 0 :
            # print("iteration:{0}".format(i),"\nbeta:\n",beta)
            # print("likehood_value:",math.exp(self.L_tilde_beta(beta)))
            if sum(abs(lr * betagradient)) / 5 * 500 < threshhold:
                return beta
        return beta

    """odal2 part"""

    def Transfer2(self, other, N):  # 传递信息
        """
        local site summerize the DL_betabar to estimate the beta
        the communication among sites should be watched
        :param other: other site
        :return: the accumulative storage of info, the the final value
        """
        other.transfered_local_betabar = self.betaBar
        other.DLj_betaBar = other.gradient_Lj(other.X, other.Y, other.transfered_local_betabar)
        other.storage = other.sample_num / N * other.DLj_betaBar
        other.secondthDlj_betaBar = other.secondD_Lj(other.X, other.Y, other.transfered_local_betabar)
        other.storage2 = other.sample_num / N * other.secondthDlj_betaBar

        if (self.storage == np.matrix([0, 0, 0, 0, 0]).T).all():
            self.storage = self.sample_num / N * self.DLj_betaBar
            self.storage2 = self.sample_num / N * self.secondthDlj_betaBar

        self.storage += other.storage
        self.localN += other.sample_num
        self.storage2 += other.storage2
        return self.storage, self.storage2

    def resetstorage(self):
        self.localN = 0
        self.transfered_local_betabar = np.matrix([0, 0, 0, 0, 0]).T
        self.L_beta = 0.

        # build temporary variable to store the info from other sites (nj * gradient_Lj)
        self.storage = self.sample_num * self.DLj_betaBar
        self.Gradient_L_betabar = np.matrix([0., 0., 0., 0., 0.])
        ## odal2 need transfer 2th gradient lj of site
        self.storage2 = self.sample_num * self.secondthDlj_betaBar
        self.secondthDl_betaBar = np.matrix(
            [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]])

    def secondD_Lj(self, X, Y, beta):
        """
        :param beta:(5,1)
        :return: (5,5)matrix
        """
        pij_betaBar = sigmoid(X @ beta)
        out = (X.T @ np.diagflat(np.multiply(-pij_betaBar, (1 - pij_betaBar))) @ X) / X.shape[0]
        # out = 0
        # for row_num in range(len(X[:, 0])):
        #     x,y = X[row_num, :], Y[row_num]
        #     pij_betaBar = 1/(1+math.exp(-x @ beta))
        #     out += pij_betaBar*(1-pij_betaBar) * x.T @ x
        return out

    def cul_secondD_L_betaBar(self, N=1):
        self.Gradient_L_betabar = self.storage / N
        self.secondthDl_betaBar = self.storage2 / N
        return self.Gradient_L_betabar, self.secondthDl_betaBar

    def L2_tilde_beta(self, beta):
        temp = (beta - self.betaBar).T @ (self.secondthDl_betaBar - self.secondthDlj_betaBar) @ (beta - self.betaBar)
        return self.L_tilde_beta(beta) + 0.5 * temp

    def Gradient_L2_tilde_beta(self, beta):
        """
        oadl2 Gradient_L2_tilde
        :param beta: (5,1)
        :return: (1)
        """
        minus = self.secondthDl_betaBar - self.secondthDlj_betaBar
        plus = 0.5 * (minus + minus.T) @ (beta - self.betaBar)
        return self.Gradient_L_tilde_beta(self.X, self.Y, beta) + plus

    def argmaxL2_tilde(self, initial_beta=np.matrix([-8, -5, 1.5, 1, 0.5]).T):
        """
        odal2 max surrogate likehood function
        :return:(4,1) beta
        """
        beta = initial_beta  # 初始值
        # print("[local site estimation ODAL2] start iteration to solve the best beta of likehood function...\n")
        for i in range(iteration):
            betagradient = self.Gradient_L2_tilde_beta(beta)
            beta = beta + betagradient * lr
            if (sum(abs(lr * betagradient)) * 100) < threshhold:
                return beta
            # if i % 500 == 0:
            #     print("iteration:{0}".format(i), "\nbeta:\n", beta)
            #     print("likehood_value:", math.exp(self.L2_tilde_beta(beta)))
        return beta


class simulation():
    def __init__(self, data, set='A', K=None, n=None):
        """
        set A: K is needed,
        set B: n is needed,
        :param data: initiate sites according to data file [x1,x2,x3,x4,Y,site]
        """
        print("\ninitiating local site....")
        self.sites_list = []
        local_data = pd.DataFrame()
        if set == 'A':  # k个站点，local 1000个样本，其余k-1个为10**r*1000个样本
            local_data = data.sample(n=1000, frac=None, replace=False)
            local_data['site'] = 0
            self.local_site = Site(local_data['x0'], local_data['x1'], local_data['x2'], local_data['x3'],
                                   local_data['x4'], local_data['y'], site='local')
            othersites_data = data.drop(local_data.index)
            for site_num in range(1, K):
                r = 2 * np.random.random() - 1  # r->(-1,1)
                sample_num = int((10 ** r) * 1000)
                datai = othersites_data.sample(n=sample_num, replace=False)
                othersites_data = othersites_data.drop(datai.index)
                exec(
                    "self.site_{0} = Site(datai['x0'],datai['x1'],datai['x2'],datai['x3'],datai['x4'],datai['y'])".format(
                        site_num))
                exec("self.sites_list.append(self.site_{0})".format(site_num))
            self.data_used_seta = data.drop(othersites_data.index)

        else:
            if set == 'B':  # local site n 个, 其余9个site random split
                local_data = data.sample(n=n, replace=False)  # 本地只抽取n个,模拟时应该控制
                othersites_data = data.drop(local_data.index)
                othersites_data['site'] = np.random.randint(1, 10,
                                                            size=data.shape[0] - local_data.shape[0])  # 1到k-1，其他站点

                # """initiate the local site and other sites based on data formed above"""
                self.local_site = Site(local_data['x0'], local_data['x1'], local_data['x2'], local_data['x3'],
                                       local_data['x4'], local_data['y'], site='local')
                # print("initiating other sites....")
                for site_num in range(1, 10):
                    # print("initiating site{0}".format(site_num))
                    datai = othersites_data.loc[othersites_data.site == site_num]
                    exec(
                        "self.site_{0} = Site(datai['x0'],datai['x1'],datai['x2'],datai['x3'],datai['x4'],datai['y'])".format(
                            site_num))
                    exec("self.sites_list.append(self.site_{0})".format(site_num))
            else:
                print("set category error !")

    def trans1(self, N):
        for site in self.sites_list:
            self.local_site.Transfer(site, N)

    def trans2(self, N=1):
        for site in self.sites_list:
            self.local_site.Transfer2(site, N)

    def getsitesnum(self):
        num = [self.local_site.sample_num]
        for site in self.sites_list:
            num.append(site.sample_num)
        return num


def avg_relative_bias(beta, gold_beta):
    relative_error = np.average((gold_beta - beta) / gold_beta, axis=0)
    return relative_error


def setA_simulate(data, k_range, sim_num):
    # simulation on set a
    file_time_mark = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_path = file_time_mark + "A"
    os.mkdir(file_path)
    Aavg_relative_bias_set_A_odal1 = []
    Aavg_relative_bias_set_A_odal2 = []
    Aavg_relative_bias_set_A_local = []

    for iter in range(sim_num):  # 4次模拟取平均值
        site_distribute_A = []
        avg_relative_bias_set_A_odal1 = []
        avg_relative_bias_set_A_odal2 = []
        avg_relative_bias_set_A_local = []
        for k in k_range:
            """odal1"""
            sm1 = simulation(data, set='A', K=k, n=1000)
            site_distribute_A.append(sm1.getsitesnum())
            print("sites are distributed as \n:", sm1.getsitesnum())
            data_used = sm1.data_used_seta
            gold_beta = sm1.local_site.argmaxLj(np.matrix(data_used[['x0', 'x1', 'x2', 'x3', 'x4']]),
                                                np.matrix(data_used['y']).T)
            print('gold beta\n', gold_beta)
            avg_relative_bias_set_A_local.append(avg_relative_bias(sm1.local_site.betaBar, gold_beta).A[0][0])
            print("LOCAL ESTIMATE:\n", sm1.local_site.betaBar)
            sm1.trans2(N=sm1.data_used_seta.shape[0])
            sm1.local_site.cul_secondD_L_betaBar()
            beta_estimate_odal1 = sm1.local_site.argmax_L_tilde_beta(initial_beta=sm1.local_site.betaBar)
            print("ODAL1 ESTIMATE:\n", beta_estimate_odal1)
            avg_relative_bias_set_A_odal1.append(avg_relative_bias(beta_estimate_odal1, gold_beta).A[0][0])
            """odal2"""
            beta_estimate_odal2 = sm1.local_site.argmaxL2_tilde(initial_beta=sm1.local_site.betaBar)
            print("ODAL2 ESTIMATE:\n", beta_estimate_odal2)
            print('dataused shape', sm1.data_used_seta.shape)
            avg_relative_bias_set_A_odal2.append(avg_relative_bias(beta_estimate_odal2, gold_beta).A[0][0])
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # pd.DataFrame(site_distribute_A).to_csv(
        #     "{2}A\\simulation{0}_A_sitedistribute_{1}.csv".format(iter, nowTime, file_time_mark))  # 每一次的模拟站点样本分布单独储存
        Aavg_relative_bias_set_A_odal1.append(avg_relative_bias_set_A_odal1)
        Aavg_relative_bias_set_A_odal2.append(avg_relative_bias_set_A_odal2)
        Aavg_relative_bias_set_A_local.append(avg_relative_bias_set_A_local)
    Avg_relative_bias_set_A_odal1 = pd.DataFrame(Aavg_relative_bias_set_A_odal1).mean()
    Avg_relative_bias_set_A_odal2 = pd.DataFrame(Aavg_relative_bias_set_A_odal2).mean()
    Avg_relative_bias_set_A_local = pd.DataFrame(Aavg_relative_bias_set_A_local).mean()
    # 保存数次模拟的相对偏差结果
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 现在

    # odal1 结果储存
    setaResult_odal1_relativebias_ = pd.DataFrame(Aavg_relative_bias_set_A_odal1)
    setaResult_odal1_relativebias_.to_csv("{1}A\\setbResult_odal1_relativebias_{0}.csv".format(nowTime, file_time_mark))

    setaResult_odal2_relativebias_ = pd.DataFrame(Aavg_relative_bias_set_A_odal2)
    setaResult_odal2_relativebias_.to_csv("{1}A\\setbResult_odal2_relativebias_{0}.csv".format(nowTime, file_time_mark))

    setaResult_local_relativebias_ = pd.DataFrame(Aavg_relative_bias_set_A_local)
    setaResult_local_relativebias_.to_csv("{1}A\\setbResult_local_relativebias_{0}.csv".format(nowTime, file_time_mark))

    table_res = pd.DataFrame([])
    table_res['odal1_min'] = setaResult_odal1_relativebias_.min()
    table_res['odal1_max'] = setaResult_odal1_relativebias_.max()
    table_res['odal1_avg'] = setaResult_odal1_relativebias_.mean()
    table_res['odal1_stdv'] = setaResult_odal1_relativebias_.std()

    table_res['odal2_min'] = setaResult_odal2_relativebias_.min()
    table_res['odal2_max'] = setaResult_odal2_relativebias_.max()
    table_res['odal2_avg'] = setaResult_odal2_relativebias_.mean()
    table_res['odal2_stdv'] = setaResult_odal2_relativebias_.std()

    table_res['local_min'] = setaResult_local_relativebias_.min()
    table_res['local_max'] = setaResult_local_relativebias_.max()
    table_res['local_avg'] = setaResult_local_relativebias_.mean()
    table_res['local_stdv'] = setaResult_local_relativebias_.std()

    table_res.to_csv("{1}A\\res_table{0}.csv".format(nowTime, file_time_mark))

    plt.plot(k_range, Avg_relative_bias_set_A_odal1, label='odal1')
    plt.plot(k_range, Avg_relative_bias_set_A_odal2, label='odal2')
    plt.plot(k_range, Avg_relative_bias_set_A_local, label='local')
    plt.title("Relative Bias on Set A")
    plt.ylabel("relative bias")
    plt.xlabel("K")
    plt.legend()
    plt.savefig("{1}A\\setaResult_meanRB_{0}.png".format(nowTime, file_time_mark))
    plt.clf()
    return 1


def setB_simulate(data, gold_beta, n_range, sim_num):
    # simulation on set b
    file_time_mark = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_path = file_time_mark + "B"
    os.mkdir(file_path)

    Aavg_relative_bias_set_B_odal1 = []
    Aavg_relative_bias_set_B_odal2 = []
    Aavg_relative_bias_set_B_local = []

    for iter in range(sim_num):  # sim_num次模拟取平均值
        site_distribute_B = []
        avg_relative_bias_set_B_odal1 = []
        avg_relative_bias_set_B_odal2 = []
        avg_relative_bias_set_B_local = []
        for n in n_range:
            sm2 = simulation(data, set='B', K=None, n=n)
            site_distribute_B.append(sm2.getsitesnum())
            print("sites are distributed as \n:", sm2.getsitesnum())
            print("LOCAL ESTIMATE:\n", sm2.local_site.betaBar)
            avg_relative_bias_set_B_local.append(avg_relative_bias(sm2.local_site.betaBar, gold_beta).A[0][0])
            """odal1"""
            sm2.trans2()
            sm2.local_site.cul_secondD_L_betaBar(N=10000)
            beta_estimate_odal1 = sm2.local_site.argmax_L_tilde_beta(initial_beta=sm2.local_site.betaBar)
            print("ODAL ESTIMATE:\n", beta_estimate_odal1)
            avg_relative_bias_set_B_odal1.append(avg_relative_bias(beta_estimate_odal1, gold_beta).A[0][0])
            """odal2"""
            beta_estimate_odal2 = sm2.local_site.argmaxL2_tilde(initial_beta=sm2.local_site.betaBar)
            print("ODAL2 ESTIMATE:\n", beta_estimate_odal2)
            avg_relative_bias_set_B_odal2.append(avg_relative_bias(beta_estimate_odal2, gold_beta).A[0][0])

        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # pd.DataFrame(site_distribute_B).to_csv(
        #     "{2}B\\simulation{0}_B_sitedistribute_{1}.csv".format(iter, nowTime, file_time_mark))
        Aavg_relative_bias_set_B_odal1.append(avg_relative_bias_set_B_odal1)
        Aavg_relative_bias_set_B_odal2.append(avg_relative_bias_set_B_odal2)
        Aavg_relative_bias_set_B_local.append(avg_relative_bias_set_B_local)
    Avg_relative_bias_set_B_odal1 = pd.DataFrame(Aavg_relative_bias_set_B_odal1).mean()
    Avg_relative_bias_set_B_odal2 = pd.DataFrame(Aavg_relative_bias_set_B_odal2).mean()
    Avg_relative_bias_set_B_local = pd.DataFrame(Aavg_relative_bias_set_B_local).mean()
    # 保存数次模拟的相对偏差结果
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # 时间戳
    # odal1 结果储存
    setbResult_odal1_relativebias_ = pd.DataFrame(Aavg_relative_bias_set_B_odal1)
    setbResult_odal1_relativebias_.to_csv("{1}B\\setbResult_odal1_relativebias_{0}.csv".format(nowTime, file_time_mark))

    setbResult_odal2_relativebias_ = pd.DataFrame(Aavg_relative_bias_set_B_odal2)
    setbResult_odal2_relativebias_.to_csv("{1}B\\setbResult_odal2_relativebias_{0}.csv".format(nowTime, file_time_mark))

    setbResult_local_relativebias_ = pd.DataFrame(Aavg_relative_bias_set_B_local)
    setbResult_local_relativebias_.to_csv("{1}B\\setbResult_local_relativebias_{0}.csv".format(nowTime, file_time_mark))

    table_res = pd.DataFrame([])
    table_res['odal1_min'] = setbResult_odal1_relativebias_.min()
    table_res['odal1_max'] = setbResult_odal1_relativebias_.max()
    table_res['odal1_avg'] = setbResult_odal1_relativebias_.mean()
    table_res['odal1_stdv'] = setbResult_odal1_relativebias_.std()

    table_res['odal2_min'] = setbResult_odal2_relativebias_.min()
    table_res['odal2_max'] = setbResult_odal2_relativebias_.max()
    table_res['odal2_avg'] = setbResult_odal2_relativebias_.mean()
    table_res['odal2_stdv'] = setbResult_odal2_relativebias_.std()

    table_res['local_min'] = setbResult_local_relativebias_.min()
    table_res['local_max'] = setbResult_local_relativebias_.max()
    table_res['local_avg'] = setbResult_local_relativebias_.mean()
    table_res['local_stdv'] = setbResult_local_relativebias_.std()

    table_res.to_csv("{1}B\\res_table{0}.csv".format(nowTime, file_time_mark))

    n_range = np.array(n_range) / 10000
    plt.plot(n_range, Avg_relative_bias_set_B_odal1, label='odal1')
    plt.plot(n_range, Avg_relative_bias_set_B_odal2, label='odal2')
    plt.plot(n_range, Avg_relative_bias_set_B_local, label='local')
    plt.title("Relative Bias on Set B")
    plt.xlabel("P")
    plt.ylabel("relative bias")
    plt.legend()
    plt.savefig("{1}B\\setbResult_meanRB_{0}.png".format(nowTime, file_time_mark))
    plt.clf()
    return 1


def main():
    """
    distribution of samples should be set in the csv file
    :return:
    """

    data = pd.read_csv("fakedata_generated.csv")

    data['x0'] = np.random.randint(0, 1, len(data['x1'])) + 1
    data['site'] = 0
    data = data[['x0', 'x1', 'x2', 'x3', 'x4', 'y', 'site']]
    # print(data)
    """simulation A/B 可以分别注释只运行一部分，结果图保存不直接显示"""
    sim_num = 500  # 模拟sim_num次取平均

    # # simulation A
    k_range = range(2, 53, 10)  # 站点数量模拟（start，end，interval）
    data_seta = data
    x1 = max_min_scaler(data_seta['x1'])
    x2 = max_min_scaler(data_seta['x2'])
    data_seta.drop('x1',axis = 1)
    data_seta.drop('x2',axis = 1)
    data_seta.loc[:,'x1'] = x1
    data_seta.loc[:,'x2'] = x2
    res1 = setA_simulate(data,k_range,sim_num) #由于每次模拟产生的总体并不一样,实际的金标准仍需要在内部计算
    print("setA:\n", res1)

    # simulation B
    # n_range = range(900, 9100, 820)  # 本地站点的样本数量（start，end，interval）
    # gold_beta_10000 = np.matrix([-7.9329, -4.9901, 1.5696, 1.0167, 0.4302]).T
    # data_setb = data.loc[data.index < 10000]
    # x1 = max_min_scaler(data_setb['x1'])
    # x2 = max_min_scaler(data_setb['x2'])
    # data_setb.drop('x1', axis=1)
    # data_setb.drop('x2', axis=1)
    # data_setb.loc[:, 'x1'] = x1
    # data_setb.loc[:, 'x2'] = x2
    # print('datasetb', data_setb)
    # setB_simulate(data_setb, gold_beta_10000, n_range, sim_num)


# if __name__ == "mian":
main()
