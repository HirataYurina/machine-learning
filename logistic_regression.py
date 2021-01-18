# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:max_entropy.py
# software: PyCharm

import numpy as np
import time

"""
    logistic regression是一种最大似然估计参数学习方法
    train_data: (60000, 784)
    test_data: (10000, 784)
    accuracy is 0.992200
    time span: 110.94647264480591
"""


def load_data(file_path):
    result = np.load(file_path)
    results = {}
    for key, value in result.items():
        results[key] = value
    # 归一化
    results['x_train'] = results['x_train'] / 255.0
    results['x_test'] = results['x_test'] / 255.0
    # 建立一个二分类支持向量机，将数字0设置为正类，其他数字设置为负类
    y_train = []
    y_test = []
    for key, value in results.items():
        if key == 'y_train':
            for label in value:
                if label == 0:
                    y_train.append(1)
                else:
                    y_train.append(0)
        if key == 'y_test':
            for label in value:
                if label == 0:
                    y_test.append(1)
                else:
                    y_test.append(0)
    # 更新dict
    results['y_train'] = np.array(y_train)
    results['y_test'] = np.array(y_test)
    return results


def logistic_regression(train_data: list, train_labels: list, steps: int):
    # 6.1.3模型参数估计
    # logistic回归，本质是将模型输出转化为概率（类似于sigmoid激活函数），而
    # 概率模型则默认为logistic模型（模拟概率模型）
    # 问题则转化为：测试数据(x, y) + 模型（参数w需要学习）-> 学习参数w
    # 可以发现，logistic regression和神经网络优化十分相似
    # 因此，使用梯度上升法迭代学习参数w，也是情理之中

    for i in range(len(train_data)):
        # 这里，需要将wx+b合并为[x, 1]*[w, b]
        train_data[i].append(1.0)

    lr = 0.001
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # 初始化w
    w = np.zeros((train_data.shape[1],))

    # 开始学习
    for i in range(steps):
        for j in range(len(train_data)):
            # 梯度上升
            wx = np.dot(w, train_data[j])
            w += lr * (train_data[j] * train_labels[j] -
                       (np.exp(wx) * train_data[j]) / (1 + np.exp(wx)))

    return w


def predict(test_data: list, w):
    test_data.append(1)
    test_data = np.array(test_data)

    # 利用logistic概率模型计算该手写数字是否为0的概率
    wx = np.dot(test_data, w)
    p = np.exp(wx) / (1 + np.exp(wx))
    if p > 0.5:
        return 1
    else:
        return 0


def cal_accuracy(test_data, test_labels, w):
    count = 0
    num = len(test_data)

    for i in range(num):
        is_zero = predict(test_data[i], w)
        ground_truth = test_labels[i]
        if ground_truth == is_zero:
            count += 1

    accuracy = count / num
    return accuracy


if __name__ == '__main__':
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    results_ = load_data(r'C:\Users\Administrator\.keras\datasets\mnist.npz')
    # print(results)
    x_train_ = results_['x_train'].reshape(60000, -1)
    y_train_ = results_['y_train']
    x_test_ = results_['x_test'].reshape(10000, -1)
    y_test_ = results_['y_test']
    x_train_ = [list(x) for x in x_train_]
    x_test_ = [list(x) for x in x_test_]

    print('start train logistic regression')
    w = logistic_regression(x_train_, y_train_, steps=100)

    # 开始测试
    accuracy = cal_accuracy(x_test_, y_test_, w)
    print('accuracy is %f' % accuracy)

    # 打印时间
    print('time span:', time.time() - start)
