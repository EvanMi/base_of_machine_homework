# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np
from linear_regression import common


def e_in_counter(x_arr, y_arr):
    w_lin = np.dot(np.dot(np.linalg.pinv(np.dot(x_arr.T, x_arr)), x_arr.T), y_arr.T)
    y_in = common.sign_zero_as_neg(np.dot(x_arr, w_lin))
    errs = np.where(y_in == y_arr, 0, 1)
    return errs.sum()/errs.size, w_lin


def transfrom(x_arr):
    ones_tem = x_arr[:, 0]
    x1_tem = x_arr[:, 1]
    x2_tem = x_arr[:, 2]
    print(ones_tem)
    print(x1_tem)
    print(x2_tem)


if __name__ == '__main__':
    avg = 0
    w_avg = 0
    avg_transform = 0
    w_transform = 0
    for i in range(1000):
        xo, yo = common.data_generator(1000)
        e_in, w_in = e_in_counter(xo, yo)
        avg = avg + (1.0 / (i + 1)) * (e_in - avg)
        w_avg = w_avg + (1.0 / (i + 1)) * (w_in - w_avg)
    print("avg:", avg, "w_avg:", w_avg)
