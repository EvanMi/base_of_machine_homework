# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np
from linear_regression import common


def e_in_counter(x_arr, y_arr):
    w_lin = np.dot(np.dot(np.linalg.pinv(np.dot(x_arr.T, x_arr)), x_arr.T), y_arr.T)
    y_in = common.sign_zero_as_neg(np.dot(x_arr, w_lin))
    errs = np.where(y_in == y_arr, 0, 1)
    return errs.sum()/errs.size, w_lin


def e_out_counter(x_arr, y_arr, w_lin):
    y_in = common.sign_zero_as_neg(np.dot(x_arr, w_lin))
    errs = np.where(y_in == y_arr, 0, 1)
    return errs.sum() / errs.size


def transform(x_arr):
    ones_tem = x_arr[:, 0]
    x1_tem = x_arr[:, 1]
    x2_tem = x_arr[:, 2]
    return np.concatenate((np.array([ones_tem]).T, np.array([x1_tem]).T, np.array([x2_tem]).T,
                           np.array([x1_tem * x2_tem]).T, np.array([x1_tem ** 2]).T, np.array([x2_tem ** 2]).T), axis=1)


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

        x_trans = transform(xo)
        e_tran, w_trans = e_in_counter(x_trans, yo)
        avg_transform = avg_transform + (1.0 / (i + 1)) * (e_tran - avg_transform)
        w_transform = w_transform + (1.0 / (i + 1)) * (w_trans - w_transform)

    print("avg:", avg, "w_avg:", w_avg)
    print("avg_trans:", avg_transform, "w_trans", w_transform)

    xo, yo = common.data_generator(1000)
    x_trans = transform(xo)
    e_out = e_out_counter(x_trans, yo, w_transform)
    print("e_out:", e_out)

