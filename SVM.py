# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:SVM.py
# software: PyCharm

import time
import numpy as np
import math
import random


'''
数据集：Mnist
训练集数量：60000(实际使用：1000)
使用60000个训练数据集内存会爆
测试集数量：10000（实际使用：100)
------------------------------
运行结果：
    正确率：99%
    运行时长：48.44022345542908s
'''


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


class SVM:
    """
    Support Vector Machine
    """

    def __init__(self, x_train, y_train, sigma=10, C=200, toler=0.001):
        """SVM
        SVM的代码实现主要是实现SMO（序列最小化）算法
        关于参数的选择：
        这几个超参数需要根据任务进行动态调整，
        例如这里的sigma，这个超参数高度依赖样本特征值范围，特征值范围较大时应该也增大sigma，
        如果不动态增大sigma，会导致高斯核计算的核函数变为0。

        Args:
            x_train: 训练数据集，用于计算Ei(式7.105)
            y_train: 训练数据集的标签，用于计算Ei(式7.105)
            sigma:   高斯核中的参数
            C:       软间隔最大化中的惩罚系数
            toler:   松弛变量
        """
        self.trainDataMat = np.mat(x_train)
        self.trainLabelMat = np.mat(y_train).T  # 训练标签集，为了方便后续运算提前做了转置，变为列向量

        self.m, self.n = np.shape(self.trainDataMat)  # m：训练集数量 n：样本特征维度（28*28=784）
        self.sigma = sigma  # 高斯核分母中的σ（高度依赖样本特征值范围）
        self.C = C  # 惩罚参数
        self.toler = toler  # 松弛变量

        self.k = self.calculate_kernel()  # 核函数（初始化时提前计算）
        # b的初始化值为0
        self.b = 0  # SVM中的偏置b
        # alpha的初始化值也为0
        self.alpha = [0] * self.trainDataMat.shape[0]  # SMO算法就是不断迭代alpha的过程
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]  # SMO运算过程中的Ei
        self.supportVecIndex = []

    def calculate_kernel(self):
        """
        式7.90
        高斯核函数
        Returns:
            k: (m, m)的列表

        """
        
        # 初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        # k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        # 大循环遍历Xi，Xi为式7.90中的x
        for i in range(self.m):

            if i % 100 == 0:
                print('construct the kernel:', i, self.m)

            # 得到式7.90中的X
            X = self.trainDataMat[i, :]  # (784,)
            # 小循环遍历Xj，Xj为式7.90中的Z
            # 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
            # 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
            # 所以小循环直接从i开始
            for j in range(i, self.m):
                # 获得Z
                Z = self.trainDataMat[j, :]
                # 先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                # 将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        # 返回高斯核矩阵
        return k

    def is_satisfy_kkt(self, i):
        """变量alpha是否满足KKT条件
        因为KKT条件是最优解的充分必要条件，所以SMO的算法核心思想就是：
        将所有变量alpha都更新至满足KKT条件

        Args:
            i: 第i个变量

        Returns:
            True or False

        """
        gxi = self.calc_gxi(i)
        yi = self.trainLabelMat[i]

        # 判断依据参照“7.4.2 变量的选择方法”中“1.第1个变量的选择”
        # 第一个变量选择违背KKT条件的alpha
        # 依据7.111
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            # 直接return True
            return True
        # 依据7.113
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        # 依据7.112
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def calc_gxi(self, i):
        """
        gxi用于计算Ei，Ei的具体作用可以查看式7.105与式7.106
        Args:
            i: 计算第i个Ei

        Returns:
            gxi

        """
        # 初始化g(xi)
        gxi = 0

        # 从数学角度看，如果α为0，αi*yi*K(xi, xj)本身也必为0，所以直接选择非零的alpha进行计算
        # index获得非零α的下标，并做成列表形式方便后续遍历
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        # 遍历每一个非零α，i为非零α的下标
        for j in index:
            # 计算g(xi)
            # 式7.104
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]
        # 求和结束后再单独加上偏置b
        gxi += self.b

        # 返回
        return gxi

    def calculate_ei(self, i):
        # 计算g(xi)
        gxi = self.calc_gxi(i)
        # 式7.105
        return gxi - self.trainLabelMat[i]

    def get_alpha_j(self, E1, i):
        """第二个变量选择
        第二个变量选择策略尤为重要
        按照《统计学习方法》中，第二个变量选择的策略是：
        假设在外层循环中已经找到第一个变量alpha1（alpha1违背KKT条件），
        在内循环中，应该找取能够使得alpha_new变化最大的alpha2。
        alpha_new是依赖于math.fabs(E1-E2)的，所以简单的做法是，选择alpha2使得math.fabs(E1-E2)最大。

        Args:
            E1: E2的选择是根据E1进行选择的
            i:  E1的下标

        Returns:
            E2
            maxIndex

        """

        # 初始化E2
        E2 = 0
        # 初始化|E1-E2|为-1
        maxE1_E2 = -1
        # 初始化第二个变量的下标
        maxIndex = -1

        # 这一步是一个优化性的算法
        # 实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
        # 然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
        # 作者最初是全部按照书中缩写，但是本函数在需要3秒左右，所以进行了一些优化措施
        # ============================================================================
        # 在Ei的初始化中，由于所有α为0，所以一开始是设置Ei初始值为-yi。这里修改为与α
        # 一致，初始状态所有Ei为0，在运行过程中再逐步更新
        # 因此在挑选第二个变量时，只考虑更新过Ei的变量，但是存在问题
        # 1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
        #   当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
        # 2.怎么保证能和书中的方法保持一样的有效性呢？
        #   在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
        # 在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行
        # 的前半程进行了时间的加速，在程序后期其实与未优化的情况无异
        # ============================================================================

        # 获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        # 对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            # 计算E2
            E2_tmp = self.calculate_ei(j)
            # 如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                # 更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                # 更新最大值E2
                E2 = E2_tmp
                # 更新最大值E2的索引j
                maxIndex = j
        # 如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            # 获得E2
            E2 = self.calculate_ei(maxIndex)

        # 返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def train(self, iter=100):
        # iterStep：迭代次数，超过设置次数还未收敛则强制停止
        # parameterChanged：单次迭代中有参数改变则增加1
        iterStep = 0
        parameterChanged = 1

        # 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        # parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明达到了收敛状态，可以停止了
        while (iterStep < iter) and (parameterChanged > 0):
            # 打印当前迭代轮数
            print('iter:%d:%d' % (iterStep, iter))
            # 迭代步数加1
            iterStep += 1
            # 新的一轮将参数改变标志位重新置0
            parameterChanged = 0

            # 大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                # 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if not self.is_satisfy_kkt(i):
                    # 如果下标为i的α不满足KKT条件，则进行优化

                    # 第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步
                    # 选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.calculate_ei(i)

                    # 选择第2个变量
                    E2, j = self.get_alpha_j(E1, i)

                    # 参考“7.4.1两个变量二次规划的求解方法” P126 下半部分
                    # 获得两个变量的标签
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    # 复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:
                        continue

                    # 计算α的新值
                    # 依据“7.4.1两个变量二次规划的求解方法”式7.106更新α2值
                    # 先获得几个k值，用来计算事7.106中的分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    # 依据式7.106更新α2，该α2还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    # ================================
                    # 剪切α2
                    # ================================
                    if alphaNew_2 < L:
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    # 更新α1，依据式7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)

                    # 依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calculate_ei(i)
                    self.E[j] = self.calculate_ei(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))

        # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            # 如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calculate_kernel_single(self, x1, x2):
        # 按照“7.3.3 常用核函数”式7.90计算高斯核
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        # 返回结果
        return np.exp(result)

    def predict(self, x):
        """
        决策函数
        依据定义7.8（非线性支持向量机）中的式7.94
        Args:
            x: 待预测的样本

        Returns:
            np.sign(决策函数)

        """

        result = 0
        for i in self.supportVecIndex:
            # =================================================
            # 遍历所有支持向量，计算求和式
            # 如果是非支持向量，求和子式必为0，没有必要进行计算
            # 这也是为什么在SVM最后只有支持向量起作用
            # =================================================
            # 先单独将核函数计算出来
            tmp = self.calculate_kernel_single(self.trainDataMat[i, :], np.mat(x))
            # 对每一项子式进行求和，最终计算得到求和项的值
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        # 求和项计算结束后加上偏置b
        result += self.b
        # 使用sign函数返回预测结果
        return np.sign(result)

    def test(self, test_data_list, test_label_list):
        # 错误计数值
        errorCnt = 0
        # 遍历测试集所有样本
        for i in range(len(test_data_list)):
            # 打印目前进度
            print('test:%d:%d' % (i, len(test_data_list)))
            # 获取预测结果
            result = self.predict(test_data_list[i])
            # 如果预测与标签不一致，错误计数值加一
            if result != test_label_list[i]:
                errorCnt += 1
        # 返回正确率
        return 1 - errorCnt / len(test_data_list)


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
    # print(x_train_)
    # print(y_train_)

    # 初始化SVM类
    print('start init SVM')
    svm = SVM(x_train_[:1000], y_train_[:1000], 10, 200, 0.001)
    # svm = SVM(x_train_, y_train_, 10, 200, 0.001)

    # 开始训练
    print('start to train')
    svm.train()

    # 开始测试
    print('start to test')
    accuracy = svm.test(x_test_[:100], y_test_[:100])
    # accuracy = svm.test(x_test_, y_test_)
    print('the accuracy is:%d' % (accuracy * 100), '%')

    # 打印时间
    print('time span:', time.time() - start)
