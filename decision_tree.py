# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:my_dt.py
# software: PyCharm


import numpy as np

'''
    使用算法ID3
    不适用剪枝
    训练数据集60000
    测试数据集10000
    决策树准确率：85.89%
    决策树速度是优于SVM
'''


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
    return results


class DecisionTree:

    def __init__(self, thres=0.1):
        self.thres = thres
        # self.feature_ids = [x for x in range(784)]

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
            hda += prob * self.hd(train_labels_i)

        return hd - hda

    def find_best_feature(self, train_feature, train_labels, ids):
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
        for i in range(num_feature):
            train_feature_single = train_feature[:, i]
            gain = self.gda(train_feature_single, train_labels, hd)
            gains.append(gain)

        best_id = ids[np.argmax(gains)]
        best_gain = np.max(gains)

        return best_id, best_gain, np.argmax(gains)

    def find_max_class(self, train_labels):
        # if not train_labels:
        #     return None
        labels_uniq = set([x for x in train_labels])
        counts = [0 for _ in labels_uniq]
        for i, j in enumerate(labels_uniq):
            for k in train_labels:
                if j == k:
                    counts[i] = counts[i] + 1
        max_labels_id = np.argmax(counts)
        return list(labels_uniq)[max_labels_id]

    def get_sub_data(self, train_data, train_labels, remove_id, classi, ids):
        select_id = train_data[:, remove_id] == classi
        train_data = np.concatenate([train_data[:, :remove_id], train_data[:, remove_id + 1:]], axis=-1)
        train_data = train_data[select_id]
        train_labels = train_labels[select_id]

        return train_data, train_labels, ids

    def train(self, train_data, train_labels, feature_ids):
        # ################################################
        # ID3算法：
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
        # 如果特征集为空集，则直接返回实例数最大的类别
        if len(train_data) == 0:
            return self.find_max_class(train_labels)
        train_labels_uniq = set([x for x in train_labels])
        # 如果只有唯一标签值，则直接返回该标签值
        if len(train_labels_uniq) == 1:
            return train_labels[0]

        # 开始训练
        best_id, best_gain, remove_id = self.find_best_feature(train_data, train_labels, feature_ids)
        if best_gain < self.thres:
            return self.find_max_class(train_labels)

        # tree的节点应该包含特征值和该分支类别两个信息，直接使用元祖作为字典的key
        ag = (best_id, self.find_max_class(train_labels))
        tree = {ag: {}}

        feature_ids = feature_ids[:remove_id] + feature_ids[remove_id + 1:]

        tree[ag][0] = self.train(*self.get_sub_data(train_data, train_labels, remove_id, 0, feature_ids))
        tree[ag][1] = self.train(*self.get_sub_data(train_data, train_labels, remove_id, 1, feature_ids))

        return tree

    def predict(self, tree, predict_data):
        # 该预测方法不适合batch prediction
        # 使用一个死循环，直到Tree的尽头，返回一个有效的分类
        while True:
            if type(tree).__name__ == 'dict':
                ((feature_id, classi), value),  = tree.items()
                tree = value[predict_data[feature_id]]
                if type(tree).__name__ == 'int':
                    return tree
            else:
                return tree

    def accuracy(self, tree, predict_data, predict_labels):
        # 计算准确率
        num = len(predict_data)
        correct_num = 0
        for i in range(num):
            prediction = self.predict(tree, predict_data[i])
            if prediction == predict_labels[i]:
                correct_num += 1
        accuracy = correct_num / num
        return accuracy


if __name__ == '__main__':
    res = load_data(r'C:\Users\Administrator\.keras\mnist.npz')
    print('载入数据成功！')
    x_test = res['x_test'].reshape(-1, 28 * 28)  # (10000, 784)
    y_test = res['y_test']  # (10000,)
    x_train = res['x_train'].reshape(-1, 28 * 28)
    y_train = res['y_train']
    # 使用决策树对手写数字进行分类
    # 这里，我们总共有784个特征
    decision_tree = DecisionTree()
    tree = decision_tree.train(train_data=x_train,
                               train_labels=y_train,
                               feature_ids=[x for x in range(784)])

    print('决策树已经训练完毕！')
    print('开始使用决策树预测...')
    accuracy = decision_tree.accuracy(tree, x_test, y_test)
    print('决策树准确率为{}'.format(accuracy))
