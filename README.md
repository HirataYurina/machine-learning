# Machine Learning

> 利用python实现机器学习算法：
>
> 支持向量机
>
> 决策树
>
> 朴素贝叶斯
>
> 集成学习：AdaBoost, Bagging, Random Forest
>
> 期望最大化算法（EM）
>
> k-nn
>
> k-means
>
> 主成分分析
>
> 奇异值分解
>
> 马尔科夫蒙特卡洛算法

## Support Vector Machine

支持向量机的推演思想主要是：

* 间隔最大化
* 拉格朗日乘子法
* 拉格朗日对偶函数

支持向量机代码实现的主要思想：

* 基于SMO的支持向量机实现

**最基本的支持向量机只支持二分类任务，如果需要实现多分类任务，可以使用以下两种方法：**

以手写数字识别为例：任务中一共有10个类别，需要利用支持向量机对手写数字进行分类。

* 一对多：将1设置为正类，其余数字为负类；将2设置为正类，其余数字设置为负类；...以此类推；一共需要10个分类器，最后将预测数据依次输入10个分类器中，结果为正类则预测为该数字。如果多个分类器判定结果为正类，则使用到分类超平面的距离大小来选择预测结果。
* 一对一：一共训练10*9/2个分类器，依次为1VS2，1VS3，1VS4，...，2VS3，...。但是，一对一投票机制有个缺点，就是随着分类目标数目增多，分类器数目显著上升。
* 有一些改进的支持向量机算法，专门针对多分类。

***

## AdaBoost

集成学习通过构建并组合多个学习器来完成学习任务。

集成学习分为：

* 同质集成学习：学习器都是同一类的，里面的学习器称之为基学习器。
* 异质集成学习：学习器不是同一类的，里面的学习器称之为个体学习器。

集成学习将多个弱学习器集成为一个强学习器。

* 弱学习器：二分类问题上，精度略高于50%的学习器。

**集成学习主要思想：**

1. 在每一轮迭代中，改变数据分布，使得预测错误的样本在下一轮学习中受到更多的关注，可以使用重赋权法和重采样法。
2. 给每个学习器赋予一个权重，表现好的学习器（资深专家）权重越高（在投票过程中，对结果的影响越大）。
3. **总结为：两个指标，样本关注度和学习器影响力。**

## MCMC

马尔科夫蒙特卡洛算法：

从一组数据分布中进行采样，利用采样得到的随机样本分析数据分布的特征。

这是，一种采样算法。

### 蒙特卡洛算法应用：

根据随机采样获得的随机数，去分析数据样本分布的特征。

1. 求π的值：在圆和正方形之间进行随机取样，利用圆内的点与圆外的点计算π的值。
2. 计算积分：与圆类似，利用随机取样，在区域内部产生大量的随机模拟点，计算有多少点落在积分区域内。
3. 计算积分：将待积分函数变换为f(x)*g(x)，这样就将积分变换为在g(x)分布下，求f(x)的期望值。即可利用g(x)概率密度进行随机采样，从而获得f(x)的期望值。
4. **蒙特卡洛在强化学习中的应用。**

***
> **蒙特卡罗算法的核心思想是：产生已知概率分布的随机变量。**
>
> **对于无法进行随机采样的复杂样本概率密度（维度过高，难以采样），可以使用马尔科夫算法进行模拟采样。**
>
> **马尔科夫蒙特卡洛算法采样效率高，容易实现。**

**经典的MCMC算法：**

**1. MH算法**

第一步，MH算法模拟一个转移核p(x', x)，然后随机初始化一组x0，利用p(x', x0)对x'进行采样。

第二步，计算接受概率，判断是否接受x‘。

PS. MH算法也有单分量MH算法，即对于难以采样的多维转移核，可以对其*满概率*分布进行单变量采样。Gibbs sampling就是单分量MH算法的特殊情景。

**2. Gibbs sampling**

吉布斯采样时单变量MH算法的特殊情景。

MH算法的采样样本会停留，而Gibbs sampling的采样样本不会被拒绝（不会停留）。

## ToDO

* [x] 支持向量机 优点：最优秀的机器学习算法，识别精度高，具有可解释性，计算量不大 缺点：对核函数和超参数敏感
* [x] 隐马尔科夫链
* [x] AdaBoost 优点：最常用的集成学习算法，实现简单，可以降低模型偏差 缺点：对离群点敏感                    
* [x] MCMC算法（MH，Gibbs sampling）
* [x] Bagging
* [x] Random Forest 优点：实现简单，降低模型的偏差，减小过拟合风险
* [x] Decision Tree 优点：可解释性强、计算量小、实现简单 缺点：容易过拟合，识别精度不高
* [x] Bayes Inference 优点：数据集小的时候依然有效，容易实现 缺点：精度较低
* [x] Logistic Regression 优点：模型容易理解，精度还不错，计算复杂度不高
* [x] Exception Maximum
* [x] Maximum Entropy 优点：模型挺复杂 缺点：嵌套循环过多，计算量很大，不适合工程应用
* [ ] k近邻 优点：对噪声不敏感，容易实现 缺点：计算复杂度高，空间复杂度高
* [x] 奇异值分解
* [x] 主成分分析
* [ ] 潜在狄利克雷分布
