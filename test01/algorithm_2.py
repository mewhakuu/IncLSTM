# -*- coding: utf-8 -*-
"""
@Copyright (C) 2022 mewhaku . All Rights Reserved 
@Time ： 2022/8/11 9:40
@Author ： mewhaku
@File ：algorithm_2.py
@IDE ：PyCharm
"""
import numpy as np
from sklearn import tree
from manager import main
# 算法二
# H 测试样本分类结果
# train_old 原训练样本 np数组
# train_new 辅助训练样本
# label_old 原训练样本标签
# label_new 辅助训练样本标签
# 使用老数据集训练的弱学习器weak_learner
# S_Test  测试样本
# N 迭代次数

def algorithm_2(train_old, train_new, label_old, label_new, S_test, weak_learner, N,predict):

    #合并数据集train_old U train_new
    train_data = np.concatenate((train_new, train_old), axis=0)
    train_label = np.concatenate((label_new, label_old), axis=0)

    # 赋值训练集的行数
    row_new = train_new.shape[0]
    row_old = train_old.shape[0]
    row_T = S_test.shape[0]

    test_data = np.concatenate((train_data, S_test), axis=0)

    # 初始化权重

    #获得beta值
    beta = main(train_old, predict)
    beta_max = np.max(beta)

    # beta 和 max_beta由veryfastKMM得到
    weight_1 = beta
    weight_2 = beta_max
    weights = np.concatenate((weight_1, weight_2), axis=0)

    # 设置rho值 #
    rho = 1 / (1 + np.sqrt(2 * np.log(row_new / N)))

    # 存储每次迭代的标签和bata值
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_new + row_old + row_T, N])

    predict = np.zeros([row_T])

    print('params initial finished.')
    train_data = np.asarray(train_data, order='C')
    train_label = np.asarray(train_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        P = calculate_P(weights)

        result_label[:, i] = train_regressor(train_data, train_label,
                                            test_data, P)
        print('result,', result_label[:, i], row_new, row_old, i, result_label.shape)

        error_rate = calculate_error_rate(label_old, result_label[row_new:row_new + row_old, i],
                                          weights[row_new:row_new + row_old, :])
        print('Error rate:', error_rate)

        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break  # 防止过拟合
            # error_rate = 0.001

        # 计算bete_t
        bata_T[0, i] = error_rate / (1 - error_rate)

        # 权重更新
        # 调整源域样本权重
        for j in range(row_old):
            weights[row_new + j] = weights[row_new + j] * np.power(bata_T[0, i],
                                                                   (-np.abs(
                                                                       result_label[row_new + j, i] - label_old[j])))

        # 调整辅域样本权重
        for j in range(row_new):
            weights[j] = weights[j] * np.power(rho, np.abs(result_label[j, i] - label_new[j]))


    # print bata_T
    for i in range(row_T):
        # 跳过训练数据的标签
        left = np.sum(
            result_label[row_new + row_old + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]
    print(predict)
    return predict


# 计算p_t
def calculate_P(weights):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


# 训练回归器
def train_regressor(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeRegressor(criterion="gini", max_features="log2", splitter="random")
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)


# 输入用旧数据集训练的弱学习器
def calculate_error_rate(row_new, y_test, predicted, weight,sample_size):
    e_i = np.zeros(sample_size)
    E_k = np.zeros(sample_size)
    for j in range(row_new):
        E_k[j] = abs(y_test[j, 0] - predicted[j, 0])

    E_k_max = np.max(E_k)

    for j in range(row_new):
        e_i[j] = 1 - np.exp(-abs(y_test[j, 0] - predicted[j, 0]) / E_k_max)
    regression_error_rate = np.sum(weight * e_i)
    return regression_error_rate
