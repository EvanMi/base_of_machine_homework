# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np


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


def data_generator(size):
    """
    生成[-1, 1)之间的随机数， 然后加入20%的噪声，即20%的概率观测值取了相反数
    :param size:
    :return:
    """
    x_arr = np.random.uniform(-1, 1, size)
    y_arr = sign_zero_as_neg(x_arr)
    y_arr = np.where(np.random.uniform(0, 1, size) < 0.2, -y_arr, y_arr)
    return x_arr, y_arr


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


def err_out_calculator(s, theta):
    return 0.5 + 0.3 * s * (abs(theta) - 1)


def decision_stump_1d(x_arr, y_arr):
    theta = x_arr
    theta_tile = np.tile(theta, (len(x_arr), 1)).T
    x_tile = np.tile(x_arr, (len(theta), 1))
    y_tile = np.tile(y_arr, (len(theta), 1))
    err_pos, index_pos = err_in_counter(x_tile, y_tile, 1, theta_tile)
    err_neg, index_neg = err_in_counter(x_tile, y_tile, -1, theta_tile)
    if err_pos < err_neg:
        return err_pos / len(y_arr), err_out_calculator(1, theta[index_pos])
    else:
        return err_neg / len(y_arr), err_out_calculator(-1, theta[index_neg])


if __name__ == '__main__':
    avg_err_in = 0
    avg_err_out = 0
    for i in range(5000):
        x, y = data_generator(20)
        e_in, e_out = decision_stump_1d(x, y)
        avg_err_in = avg_err_in + (1.0 / (i + 1)) * (e_in - avg_err_in)
        avg_err_out = avg_err_out + (1.0 / (i + 1)) * (e_out - avg_err_out)
    print("e_in:", avg_err_in)
    print("e_out:", avg_err_out)
