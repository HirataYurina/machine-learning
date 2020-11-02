# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:MCMC_gibbs_sampling.py
# software: PyCharm

import numpy as np

"""
    1.利用蒙特卡洛法进行建模
    某面包店生产糕点类食品，一般当天生产的产品必须当天售出，
    否则就会出现不能保质、或变质、造成一定的经济损失，如果当天需求量大而生产量不足，
    则也会影响面包店的销售收入，该面包的单位成本为3.5元，单位产品售价为10元。
    工厂为了避免面包滞销存货过多而造成的经济损失，提出了如何制定合理的生产与库存数量的方案问题，
    能够使得面包店能有尽可能多的收益，经初步考虑拟从以下两种生产与库存方案中选出一个较好的方案：
    方案(1)：按前一天的销售量作为当天的生产库存量。
    方案(2)：按前两天的平均销售量作为当天的生产库存量。
    ===================================================================================
    2.马尔科夫蒙特卡洛算法
    3.Gibbs sampling
"""


def profit(days, p1, p2, p3, mu, sigma):
    """利用蒙特卡洛方法，随机模拟市场对该产品需求量
    假设每天的需求量符合正态分布:N(1000, 25^2)

    Args:
        days:  模拟的天数
        p1:    第一种方案，第一天的销售量
        p2:    第二种方案，第一天销售量
        p3:    第二种方案，第二天销售量
        mu:    概率密度中的mu
        sigma: 概率密度中的sigma

    Returns:
        sum_p1
        sum_p2

    """
    sum_p1 = 0
    sum_p2 = 0
    day = 1
    sale_first_project = p1
    sale_second_project_day1 = p2
    sale_second_project_day2 = p3

    while day <= days:

        sale_need = np.random.normal(mu, sigma)
        sale_average = (sale_second_project_day1 + sale_second_project_day2) / 2.0

        # 第一种方案
        if sale_first_project <= sale_need:
            sale_first_pro_new = sale_first_project
        else:
            sale_first_pro_new = sale_need

        # 第二种方案
        if sale_average <= sale_need:
            sale_second_pro_new = sale_average
        else:
            sale_second_pro_new = sale_need

        # 计算当天的销售利润
        profit1 = 10 * sale_first_project - 3.5 * sale_first_pro_new
        profit2 = 10 * sale_average - 3.5 * sale_second_pro_new
        sum_p1 += profit1
        sum_p2 += profit2

        # 更新当天最新的销量值
        sale_first_project = sale_first_pro_new
        sale_second_project_day1 = sale_second_project_day2
        sale_second_project_day2 = sale_second_pro_new
        day += 1

    return sum_p1, sum_p2


if __name__ == '__main__':
    sum_pro1, sum_pro2 = profit(30, 998, 995, 1001, 1000, 25)
    print(sum_pro1, sum_pro2)
