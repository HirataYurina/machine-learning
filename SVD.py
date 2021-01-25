# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:svd.py
# software: PyCharm


from scipy.linalg import svd
import numpy as np


# SVD:奇异值分解
# 1.利用奇异值分解简化数据，即对数据进行压缩
# 2.去除数据中的噪声
# 3.去噪，从而提升算法的效果
# M = U @ S @ V
M = np.random.randint(0, 10, (10, 9))
# 对M进行奇异值分解
U, S, V = svd(M)
print(U.shape)
print(S.shape)
print(V.shape)
print(U)
print(S)
print(V)
