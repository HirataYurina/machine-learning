# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:max_entropy.py
# software: PyCharm

import numpy as np
import time
from collections import defaultdict

"""
    最大熵模型由于其训练过程中嵌套迭代过多，
    直接导致训练时间较长，在accuracy和speed之间trade off，
    最大熵模型学习模型不推荐工程应用。
"""


def load_data(file_path):
    result = np.load(file_path)
    results = {}
    for key, value in result.items():
        results[key] = value
    # # 归一化
    # results['x_train'] = results['x_train'] / 255.0  # (60000, 28, 28)
    # results['x_test'] = results['x_test'] / 255.0  # (10000, 28, 28)
    # TODO: 对图像进行二值化，这样每个节点分支只有0和1两种情况
    results['x_train'] = np.where(results['x_train'] > 128, 1, 0)
    results['x_test'] = np.where(results['x_test'] > 128, 1, 0)
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


# ##################################
# 其实M这个值可以理解为学习率，从模型学习
# 角度去考虑，是无关紧要的
M = 784
# ##################################


def cal_fxy(train_data, train_lables):
    # 特征函数fxy是一个二值函数，表示输入x和输出y之间的某一事实
    # 式子6.34中的M应该为28*28=784，因为每个特征维度的x均可以取0和1
    # y也可能为任意手写数字
    # 那么，我们是否应该严格训练数据集去统计特诊函数fxy？即也许存在fi(x, y)=0
    # 个人觉得，直接使用784较好，因为mnist数据集并不能包含所有的手写数字
    # 存在fi(x, y)=0，并不表示fi(x, y)不存在（只不过没统计到而已）
    # 所以，直接使用M=784更加符合逻辑

    n = 0

    # [default_dict, default_dict, ...]
    fixy = [defaultdict(int) for _ in range(len(train_data[0]))]
    num_data = len(train_data)
    num_feature = len(train_data[0])

    for i in range(num_data):
        data = train_data[i]
        label = train_lables[i]
        for j in range(num_feature):
            # {(x, y): count}
            # 字典key值为一个元祖
            fixy[j][(data[j], label)] += 1
    for d in fixy:
        n += len(d)

    # 这里的n是一共有多少个特征函数
    return fixy, n


def cal_ep_(feature_function_num, data_num, feature_num, fixy, xy2id):
    # ep_的个数应该和fixy的个数相等
    ep_ = [0] * feature_function_num

    for i in range(feature_num):
        for key, val in fixy[i].items():
            ep_[xy2id[i][key]] = val / data_num
    return ep_


def cal_pwy_x(fixy,
              X,
              y,
              key2index,
              w):
    # 其中，X形状为(784,)
    # w形状为(feature_fuc_num,)

    sun = 0
    mother = 0

    # 式6.28与式6.29
    # 注式6.29有个对于标签y的积分
    for i in range(len(X)):
        feature_label = (X[i], y)
        if feature_label in fixy[i]:
            sun += w[key2index[i][feature_label]]
        # 本任务是一个二分类任务，因此另外一个标签为1-y
        feature_label_other = (X[i], 1-y)
        if feature_label_other in fixy[i]:
            mother += w[key2index[i][feature_label_other]]
    sun_exp = np.exp(sun)
    mother_exp = np.exp(sun + mother)
    pwyx = sun_exp / mother_exp

    return pwyx


def iis_train(train_data,
              train_labels,
              key2index,
              steps,
              fixy,
              n):
    """改进的尺度迭代算法
    算法6.1
    输入：特征函数f1, f2, f3...
         经验分布p_xy
         条件概率模型pw_y|x

    Args:
        train_data: (60000, 784) 已经二值化
        train_labels: (60000,) 把原始问题转化为一个二分类问题，0 VS 其他数字

    """
    w = [0] * n

    for step in range(steps):
        # 先迭代一下train_data，计算出Ep
        ep = [0] * n
        num = len(train_data)
        for i in range(num):
            # 这里在计算ep期望的的时候，我们需要注意一个点，由于pw(y|x)是模拟模型，所以模型预测的结果我们是不知道的
            # 因此，我们需要计算pw(0|x)和pw(1|x)（因为我们不知道概率模型预测的标签是哪个）
            for j in range(len(train_data[0])):
                pwyx = [0] * 2
                if (train_data[i][j], 0) in fixy[j]:
                    pwyx[0] = cal_pwy_x(fixy, train_data[i], 0, key2index, w)
                    ep[key2index[j][(train_data[i][j], 0)]] += 1 / num * pwyx[0]
                if (train_data[i][j], 1) in fixy[j]:
                    pwyx[1] = cal_pwy_x(fixy, train_data[i], 1, key2index, w)
                    ep[key2index[j][(train_data[i][j], 1)]] += 1 / num * pwyx[1]

        # 计算ep
        ep_ = cal_ep_(n, num, len(train_data[0]), fixy, key2index)

        sigmas = [0] * n
        for k in range(n):
            sigmas[k] = 1 / M * np.log(ep_[k] / ep[k])
        # 更新w
        # 算法6.1更新w值
        w = [w[i] + sigmas[i] for i in range(n)]

    return w


def get_search_dic(fixy):
    xy2id = [defaultdict(int) for i in range(len(fixy))]
    id = 0
    for i, dic in enumerate(fixy):
        for key in dic:
            xy2id[i][key] = id
            id += 1

    return xy2id


def predict(data,
            w,
            fixy,
            xy2id):
    pw1_x = cal_pwy_x(fixy, data, 1, xy2id, w)
    pw0_x = cal_pwy_x(fixy, data, 0, xy2id, w)
    # 比较大小，相似于softmax
    if pw1_x > pw0_x:
        return 1
    else:
        return 0


def cal_accuracy(test_data, test_labels, w, fixy, xy2id):
    right = 0
    num_data = len(test_data)

    for i, d in enumerate(test_data):
        prediction = predict(d, w, fixy, xy2id)
        if prediction == test_labels[i]:
            right += 1

    return right / num_data


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
    # 计算(x, y)数值对在整个训练集中出现的频率
    fixy, n = cal_fxy(train_data=x_train_[:100], train_lables=y_train_[:100])

    # 得到xy2id矩阵
    xy2id = get_search_dic(fixy)

    # 开始训练
    w = iis_train(train_data=x_train_[:100],
                  train_labels=y_train_[:100],
                  key2index=xy2id,
                  steps=100,
                  fixy=fixy,
                  n=n)

    # 模型评估
    accuracy = cal_accuracy(x_test_[:10], y_test_[:10], w, fixy, xy2id)
    print('accuracy is %f' % accuracy)

    # 打印时间
    print('time span:', time.time() - start)