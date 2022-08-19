# -*- coding: utf-8 -*-
"""
@Copyright (C) 2022 mewhaku . All Rights Reserved
@Time ： 2022/8/8 11:19
@Author ： mewhaku
@File ：vfkmm.py
@IDE ：PyCharm
"""
import numpy as np
from sklearn import tree


# 算法二
# H 测试样本分类结果
# train_old 原训练样本 np数组
# train_new 辅助训练样本
# label_old 原训练样本标签
# label_new 辅助训练样本标签
# 使用老数据集训练的弱学习器weak_learner
# S_Test  测试样本
# N 迭代次数
def algorithm_2(train_old, train_new, label_old, label_new, S_test, weak_learner, N):
    train_data = np.concatenate((train_new, train_old), axis=0)
    train_label = np.concatenate((label_new, label_old), axis=0)

    # 赋值训练集的行数
    row_new = train_new.shape[0]
    row_old = train_old.shape[0]
    row_T = S_test.shape[0]

    test_data = np.concatenate((train_data, S_test), axis=0)

    # 初始化权重
    #缺少veryfastKMM
    #合并数据集train_old U train_new
    beta = train_new.shape[0]
    max_beta = train_old.shape[0]
    # beta 和 max_beta由veryfastKMM得到
    weight_1 = beta
    weight_2 = np.max(max_beta)
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
        P = calculate_P(weights, train_label)

        result_label[:, i] = train_classify(train_data, train_label,
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

        bata_T[0, i] = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_old):
            weights[row_new + j] = weights[row_new + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_new + j, i] - label_old[j])))

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

    return predict


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeRegressor(criterion="gini", max_features="log2", splitter="random")
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    print(weight[:, 0] / total)
    print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))