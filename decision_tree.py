# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:decision_tree.py
# software: PyCharm


import numpy as np


def load_data(file_path):
    result = np.load(file_path)
    results = {}
    for key, value in result.items():
        results[key] = value
    # # 归一化
    # results['x_train'] = results['x_train'] / 255.0  # (60000, 28, 28)
    # results['x_test'] = results['x_test'] / 255.0  # (10000, 28, 28)
    # TODO: 对图像进行二值化，这样每个节点分支只有0和1两种情况
    # 建立一个二分类支持向量机，将数字0设置为正类，其他数字设置为负类
    y_train = []
    y_test = []
    for key, value in results.items():
        if key == 'y_train':
            for label in value:
                if label == 0:
                    y_train.append(1)
                else:
                    y_train.append(-1)
        if key == 'y_test':
            for label in value:
                if label == 0:
                    y_test.append(1)
                else:
                    y_test.append(-1)
    # 更新dict
    results['y_train'] = np.array(y_train)
    results['y_test'] = np.array(y_test)
    return results


class DecisionTree:

    def __init__(self):
        self.thres = 0.1

    def hd(self, train_labels):
        # ################################
        # 计算数据集D的经验熵
        # 式5-7
        # ################################
        train_labels_uni = set([x for x in train_labels])
        hd = 0
        for i in train_labels_uni:
            prob = train_labels[train_labels == i].shape[0] / train_labels.shape[0]
            hd += -prob * np.log2(prob)

        return hd

    def gda(self, train_feature, train_labels, hd):
        # #############################################
        # 计算信息增益|计算特征A对数据集D的经验条件熵
        # g(d, a) = h(d) - h(d|a)
        # 式5-8
        # 式5-9
        # #############################################
        feature_uniq = set([x for x in train_feature])
        hda = 0
        for i in feature_uniq:
            prob = train_feature[train_feature == i].shape[0] / train_feature.shape[0]
            train_labels_i = train_labels[train_feature == i]
            hda += -prob * self.hd(train_labels_i)

        return hd - hda

    def find_best_feature(self, train_feature, train_labels):
        """find best feature by information gain

        Args:
            train_feature: (10000, 784)
            train_labels: (10000,)

        Returns:
            index of best feature

        """
        hd = self.hd(train_labels)
        num_feature = train_feature.shape[1]
        gains = []
        for i in num_feature:
            train_feature_single = train_feature[:, i]
            gain = self.gda(train_feature_single, train_labels, hd)
            gains.append(gain)

        best_id = np.argmax(gains)
        best_gain = np.max(gains)

        return best_id, best_gain

    def train(self):
        # ################################################
        # 开始训练
        # (1)计算数据集D的经验熵
        # (2)计算数据集D在特征A下的条件经验熵
        # (3)计算信息增益
        # (4)选择信息增益最大的特征值Ag
        # (5)如果Ag的信息增益小于阈值，那么T为单节点树，并将
        #    实例数最大的类Ck作为该节点的类标记，返回T。
        # (6)否则，依照Ag=ai对D进行分割，
        #    将Di中实例数最大的类作为该子节点的类标记
        # (7)对第i个子节点，以Di为训练集，A-Ag为特征集，
        #    递归调用第1-6步骤
        # ################################################
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    res = load_data(r'C:\Users\Administrator\.keras\mnist.npz')
    print('载入数据成功！')
    x_test = res['x_test'].reshape(-1, 28 * 28)  # (10000, 784)
    y_test = res['y_test']  # (10000,)
    # 使用决策树对手写数字进行分类
    # 这里，我们总共有784个特征
