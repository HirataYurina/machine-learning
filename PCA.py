# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:PCA.py
# software: PyCharm

from sklearn.decomposition import PCA
import numpy as np


# 主成分分析
# 利用PCA对数据进行线性变换，使得变换后的数据在维度之间是线性无关的
# 从而，提取出原始数据中的重要特征
X = np.array([[-1, -1],
              [-2, -1],
              [-3, -2],
              [1, 1],
              [2, 1],
              [3, 2]])
pca = PCA()
pca.fit(X)
print(pca.components_)  # [[-0.83849224 -0.54491354], [ 0.54491354 -0.83849224]]
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)  # 每个主成分的l2范数

linear_trans = np.array([-0.83849224, -0.54491354]).T
first_component = np.matmul(X, linear_trans)
l2_norm = np.linalg.norm(first_component)
print(l2_norm)
