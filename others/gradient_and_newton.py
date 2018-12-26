# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np

"""
作业三中使用梯度下降和牛顿法进行迭代
"""


def update(u, v, eta):
    u_tem = u - eta * (np.exp(u) + v * np.exp(u*v) + 2 * u - 2 * v - 3)
    v_tem = v - eta * (2 * np.exp(2*v) + u * np.exp(u * v) - 2 * u + 4 * v -2)
    return u_tem, v_tem


def iter_update(u, v, times):
    uo = u
    vo = v
    for i in range(times):
        uo, vo = update(uo, vo, 0.01)
    return np.exp(uo) + np.exp(2 * vo) + np.exp(uo * vo) + uo ** 2 - 2 * uo * vo + 2 * vo ** 2 - 3 * uo - 2 * vo


def update_newton(u, v):
    gradient_tem = np.array([np.exp(u) + v * np.exp(u*v) + 2 * u - 2 * v - 3,
                             2 * np.exp(2*v) + u * np.exp(u * v) - 2 * u + 4 * v - 2])
    laplace_tem = np.array([[np.exp(u) + (v ** 2) * np.exp(u * v) + 2, u * v * np.exp(u * v) + np.exp(u * v) - 2],
                            [u * v * np.exp(u * v) + np.exp(u * v) - 2, 4 * np.exp(2 * v) + (u ** 2) * np.exp(u * v) + 4
                             ]])
    result = np.array([u, v]) - np.dot(np.linalg.pinv(laplace_tem), np.transpose(gradient_tem))
    return result


def iter_update_newton(u, v, times):
    uo = u
    vo = v
    for i in range(times):
        uo, vo = update_newton(uo, vo)
    return np.exp(uo) + np.exp(2 * vo) + np.exp(uo * vo) + uo ** 2 - 2 * uo * vo + 2 * vo ** 2 - 3 * uo - 2 * vo


print(iter_update(0, 0, 5))
print(iter_update_newton(0, 0, 5))
