# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np


def data_generator(size):
    x_arr = np.concatenate((np.array([np.random.uniform(-1, 1, size)]).T, np.array([np.random.uniform(-1, 1, size)]).T),
                           axis=1)
    y_arr = target_function(x_arr)
    tem = np.ones((size, 1))
    x_arr = np.concatenate((tem, x_arr), axis=1)
    y_arr = np.where(np.random.uniform(0, 1, size) < 0.1, -y_arr, y_arr)
    return x_arr, y_arr


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


def target_function(x):
    x_tem = (x * x).sum(axis=1) - 0.6
    result = sign_zero_as_neg(x_tem)
    return result
