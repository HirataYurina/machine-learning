# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:gboost.py
# software: PyCharm


import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


'''
数据集：Mnist
训练集数量：60000(实际使用：56000)
使用60000个训练数据集内存会爆
测试集数量：10000（实际使用：14000)
------------------------------
运行结果：
    正确率：99.135% 准确率和SVM不相上下
'''


def get_dataset(file_path):
    # get train and test dataset
    data = np.load(file_path)
    results = {}
    for key, value in data.items():
        results[key] = value
    results['x_train'] = results['x_train'] / 255.0
    results['x_test'] = results['x_test'] / 255.0
    y_test = results['y_test']
    y_train = results['y_train']
    # for i in results['y_train']:
    #     if i == 0:
    #         y_train.append(0)
    #     else:
    #         y_train.append(1)
    # for j in results['y_test']:
    #     if j == 0:
    #         y_test.append(0)
    #     else:
    #         y_test.append(1)
    x_train = results['x_train']
    x_test = results['x_test']

    x_total = np.concatenate([x_train, x_test], axis=0)
    y_total = np.concatenate([y_train, y_test], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, shuffle=False)
    x_train = np.reshape(x_train, newshape=(-1, 28 * 28))
    x_test = np.reshape(x_test, newshape=(-1, 28 * 28))

    return x_train, x_test, y_train, y_test


# set XGBoost's parameters
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 回归任务设置为：'objective': 'reg:gamma',
    'num_class': 10,  # 回归任务没有这个参数
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}


if __name__ == '__main__':
    file_path_ = r'C:\Users\Administrator\.keras\datasets\mnist.npz'
    x_train, x_test, y_train, y_test = get_dataset(file_path_)

    dtrain = xgb.DMatrix(x_train, y_train)
    # steps = 100
    # model = xgb.train(params.items(), dtrain, num_boost_round=100)

    # load model
    model = pickle.load(open('techi.pickle.dat', 'rb'))

    data_test = xgb.DMatrix(x_test)
    prediction = model.predict(data_test)
    # save model to file
    # pickle.dump(model, open('techi.pickle.dat', 'wb'))

    num_total = np.shape(prediction)[0]
    correct = 0
    for i in range(num_total):
        if prediction[i] == y_test[i]:
            correct += 1

    accuracy = correct / num_total

    print(accuracy)
