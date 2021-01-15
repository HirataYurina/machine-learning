# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:HMM.py
# software: PyCharm

import numpy as np


'''
    维比特算法：
    维比特算法基于动态规划的思想，假设有一个最佳路径[i1, i2, i3, i4, i5]
    那么，i4->i5必然是最佳路径，不然就会存在另外一条最佳路径，这是与假设相矛盾的
    所以，当求得最佳最优路径概率P极其终节点i，之后可以通过回溯求得整体最优路径
'''


def find_best_road(state_matrix, observation_matrix, initial_state, observation):
    """find the best end road

    Args:
        state_matrix:       (m, m)
                             m is number of state
        observation_matrix: (m, j)
                             j is number of observation
        initial_state:      (m, 1)
        observation:        [0, 1, 2, ...] 0 is the first of observations
                            shape is (k,)

    Returns:

    """
    num_sequence = len(observation)
    num_road = len(state_matrix)

    # 初始化转态1的概率
    initial_state = np.transpose(initial_state)
    first_observation = observation[0]
    init_prob = np.multiply(initial_state, observation_matrix[:, first_observation])  # (m, 1)
    max_id = np.argmax(init_prob)

    prob_sequence = [init_prob]

    # 第一层循环：循环每个观测值
    # 10.4.2维比特算法
    for i in range(1, num_sequence):
        temp_obs = observation[i]
        temp_prob = []
        for j in range(num_road):

            # 第二层循环：循环每个转态值
            # 式10.45
            prob_j = np.max(init_prob * state_matrix[:, j] * observation_matrix[j][temp_obs])
            temp_prob.append(prob_j)
        init_prob = np.array(temp_prob).T
        max_id = np.argmax(init_prob)
        prob_sequence.append(init_prob)

    # 返回最后一个最优节点和概率序列
    return max_id, prob_sequence


def find_pre_node(pre_road, state_matrix, current_road):
    # 从后往前递推，求得前一个节点路径
    # 式10.46
    probs = pre_road * state_matrix[:, current_road]
    return np.argmax(probs)


def get_best_roads(prob_sequence, end_road, state_matrix):
    best_roads = [end_road]
    for i in range(len(prob_sequence) - 1, 0, -1):
        end_road = find_pre_node(prob_sequence[i], state_matrix, end_road)
        best_roads.append(end_road)

    return best_roads.reverse()


if __name__ == '__main__':
    # 例10.3
    # 状态转移矩阵
    A = np.array([[0.5, 0.3, 0.2],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    # 观测概率矩阵
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    # 初始状态
    pi = np.array([0.2, 0.4, 0.4]).T
    # 观测结果(红，白，红)
    observe_res = [0, 1, 0]

    end_road, pro_seq = find_best_road(A, B, pi, observe_res)
    best_roads = get_best_roads(pro_seq, end_road, state_matrix=A)
    print(best_roads)
