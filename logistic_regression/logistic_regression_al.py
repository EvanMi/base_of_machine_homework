# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np


def load_data(file_name):
    x = []
    y = []
    with open(file_name, 'r+') as f:
        for line in f:
            line = line.rstrip("\n").strip(' ')
            temp = line.split(" ")
            temp.insert(0, '1')
            x_temp = [float(val) for val in temp[:-1]]
            y_tem = [int(val) for val in temp[-1:]][0]
            x.append(x_temp)
            y.append(y_tem)

    nx = np.array(x)
    ny = np.array(y)
    return nx, ny


def gradient_decent_logistic_regression(x, y, eta, w, times):
    local_w = w
    for i in range(times):
        tem_w = np.dot((1.0 / (1 + np.exp(-((-y) * np.dot(x, local_w))))), np.array([-y]).T * x) / np.size(y)
        local_w = local_w - eta * tem_w
    return local_w


def stochastic_gradient_decent_logistic_regression(x, y, eta, w, times):
    local_w = w
    index = 0
    for i in range(times):
        x_tem = x[index, :]
        y_tem = y[index]
        tem_w = (1.0 / (1 + np.exp(-((-y_tem) * np.dot(local_w, x_tem))))) * (-y_tem) * x_tem
        local_w = local_w - eta * tem_w
        index = (index + 1) % np.size(y)
    return local_w


def e_out_counter(x, y, w):
    local_tem = 1.0 / (1 + np.exp(np.dot(x, w)))
    vec_result = np.where(local_tem > 0.5, 1, -1)
    result = np.where(vec_result == y, 1, 0)
    return sum(result)/np.size(result)


if __name__ == '__main__':
    x_train, y_train = load_data('data/train.dat')
    x_val, y_val = load_data('data/test.dat')
    w_one = gradient_decent_logistic_regression(x_train, y_train, 0.001, np.zeros(np.size(x_train, 1)), 2000)
    e_out_one = e_out_counter(x_val, y_val, w_one)
    print("e_out_one:", e_out_one)

    w_two = gradient_decent_logistic_regression(x_train, y_train, 0.01, np.zeros(np.size(x_train, 1)), 2000)
    e_out_two = e_out_counter(x_val, y_val, w_two)
    print("e_out_two:", e_out_two)

    w_s = stochastic_gradient_decent_logistic_regression(x_train, y_train, 0.001, np.zeros(np.size(x_train, 1)), 2000)
    e_out_s = e_out_counter(x_val, y_val, w_s)
    print("e_out_s:", e_out_s)

