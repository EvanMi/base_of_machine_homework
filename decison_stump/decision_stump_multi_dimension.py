# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np
from decison_stump import util


def sign_zero_as_neg(x):
    """
    这里修改了np自带的sign函数，当传入的值为0的时候，不再返回0，而是-1；
    也就是说在边界上的点按反例处理
    :param x:
    :return:
    """
    result = np.sign(x)
    result[result == 0] = -1
    return result


def err_in_counter(x_arr, y_arr, s, theta):
    """
    计算E_in
    :param x_arr:
    [[x1, x2, x3, ... ,xn]
     [x1, x2, x3, ... ,xn]
     [x1, x2, x3, ... ,xn]
            ...
     [x1, x2, x3, ... ,xn]]
    :param y_arr:
    [[y1, y2, y3, ... ,yn]
     [y1, y2, y3, ... ,yn]
     [y1, y2, y3, ... ,yn]
            ...
     [y1, y2, y3, ... ,yn]]
    :param s:{-1,1}
    :param theta:
    [[theta1, theta1, theta1, ... ,theta1]
     [theta2, theta2, theta2, ..., theta2]
     [theta3, theta3, theta3, ..., theta3]
                ...
     [thetak, thetak, thetak, ..., thetak]]
    :return:[err_theta1, err_theta2, ..., err_thetak] 中最小的以及下标
    """
    result = s * sign_zero_as_neg(x_arr - theta)
    err_tile = np.where(result == y_arr, 0, 1).sum(1)
    return err_tile.min(), err_tile.argmin()


def err_out_counter(x_arr, y_arr, s, theta, dimension):
    temp = s * sign_zero_as_neg(x_arr.T[dimension] - theta)
    e_out = np.where(temp == y_arr, 0, 1).sum() / np.size(x_arr, 0)
    return e_out


def decision_stump_1d(x_arr, y_arr):
    theta = x_arr
    theta_tile = np.tile(theta, (len(x_arr), 1)).T
    x_tile = np.tile(x_arr, (len(theta), 1))
    y_tile = np.tile(y_arr, (len(theta), 1))
    err_pos, index_pos = err_in_counter(x_tile, y_tile, 1, theta_tile)
    err_neg, index_neg = err_in_counter(x_tile, y_tile, -1, theta_tile)
    if err_pos < err_neg:
        return err_pos / len(y_arr), index_pos, 1
    else:
        return err_neg / len(y_arr), index_neg, -1


def decision_stump_multi_d(x, y):
    x = x.T
    dimension, e_in, theta, s = 0, float('inf'), 0, 0
    for i in range(np.size(x, 0)):
        e_in_temp, index, s_temp = decision_stump_1d(x[i], y)
        if e_in_temp < e_in:
            dimension, e_in, theta, s = i, e_in_temp, x[i][index], s_temp
        # 错误率相等的时候随机选择
        if e_in_temp == e_in:
            pick_rate = np.random.uniform(0, 1)
            if pick_rate > 0.5:
                dimension, e_in, theta, s = i, e_in_temp, x[i][index], s_temp
    return dimension, e_in, theta, s


if __name__ == '__main__':
    x_train, y_train = util.load_data('data/train.txt')
    x_test, y_test = util.load_data('data/test.txt')
    determined_dimension, e_in_result, theta_result, s_result = decision_stump_multi_d(x_train, y_train)
    print("E_IN:", e_in_result)
    print("E_OUT:", err_out_counter(x_test, y_test, s_result, theta_result, determined_dimension))
