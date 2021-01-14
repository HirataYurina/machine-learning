# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:bayes_inference.py
# software: PyCharm


import numpy as np

'''
    贝叶斯估计+拉普拉斯平滑
    贝叶斯估计模型的准确率和训练数据集的大小有直接关系
    训练集：10000
    测试集：100
    准确率：73%
'''


def load_data(file_path):
    """load dataset

    Args:
        file_path: npz file

    Returns:

    """
    result = np.load(file_path)
    results = {}
    for key, value in result.items():
        results[key] = value

    results['x_train'] = np.where(results['x_train'] > 128, 1, 0)
    results['x_test'] = np.where(results['x_test'] > 128, 1, 0)

    return results


class BayesInfer:

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.labels = list(set(train_labels))
        self.si = len(train_data[0])
        self.lambd = 1

    def cal_pxy(self, x, yi, i):
        index_bool = self.train_labels == yi
        train_data = self.train_data[index_bool][:, i]
        train_data_x = train_data[train_data == x]

        return (len(train_data_x) + self.lambd) / (len(train_data) + self.si * self.lambd)

    def cal_py(self, yi):
        index_bool = self.train_labels == yi
        train_yi = self.train_labels[index_bool]

        return len(train_yi) / len(self.train_labels)

    def get_max_prob(self, inputs):
        probs = []
        for label in self.labels:
            temp_prob = self.cal_py(label)
            for i, j in enumerate(inputs):
                temp_prob *= self.cal_pxy(j, label, i)
            probs.append(temp_prob)

        return np.argmax(probs)

    def predict(self, inputs):
        prediction_id = self.get_max_prob(inputs)
        prediction = self.labels[prediction_id]
        return prediction

    def accuracy(self, test_data, test_labels):
        num = len(test_data)
        correct = 0
        for i, data in enumerate(test_data):
            prediction = self.predict(data)
            if prediction == test_labels[i]:
                correct += 1
        accuracy = correct / num

        return accuracy


if __name__ == '__main__':
    res = load_data(r'C:\Users\Administrator\.keras\datasets\mnist.npz')
    print('载入数据成功！')
    x_test = res['x_test'].reshape(-1, 28 * 28)  # (10000, 784)
    y_test = res['y_test']  # (10000,)
    x_train = res['x_train'].reshape(-1, 28 * 28)
    y_train = res['y_train']

    bayes_infer = BayesInfer(train_data=x_train[:10000], train_labels=y_train[:10000])
    acc = bayes_infer.accuracy(x_test[:100], y_test[:100])

    print(acc)
