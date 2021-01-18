# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:max_entropy.py
# software: PyCharm

import numpy as np
import time

"""
    
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


def cal_fxy():
    pass


def cal_pxy():
    pass


def cal_py_x():
    pass


def iis_train():
    pass


def cal_accuracy():
    pass


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