# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np
from pla_and_pocket_pla import util


def error_counter(x, y, w):
    result = np.where(x.dot(w[0].T) > 0, 1, -1)
    compare_result = np.where(result == y.T[0], 0, 1)
    return (1.0 * np.sum(compare_result)) / np.size(y, 0)


def pocket_pla(nx, ny, rate=1, max_iter=50):
    """
    :param nx:属性矩阵，格式是[[...],[...],[...]]
    :param ny: 值，格式是 [[.],[.],[.]]
    :param rate:  学习率，默认是1
    :param max_iter: 最大迭代次数，默认50
    :return: w_pocket和w
    """
    total_update_nums = 0
    total_train_example_nums = np.size(nx, 0)
    w_pocket = np.zeros((1, 5))  # w_pocket 就是一个口袋里的桃子，观察着w的变化，一旦比自己好，立马把w放进口袋里
    w = np.zeros((1, 5))  # 初始化参数为0

    while True:
        rand_index = np.random.randint(0, total_train_example_nums)
        this_x = nx[rand_index]
        result = util.sign(np.dot(this_x, w.T)[0])
        this_y = ny[rand_index, 0]
        if int(result) != int(this_y):
            w = w + rate * (this_x * this_y)
            total_update_nums += 1
            if error_counter(nx, ny, w) < error_counter(nx, ny, w_pocket):
                w_pocket = w
        if total_update_nums == max_iter:
            break
    return w_pocket, w


if __name__ == '__main__':
    x_train, y_train = util.load_data("data/train.txt")
    x_test, y_test = util.load_data("data/test.txt")
    avg_pocket = 0
    avg = 0
    for index in range(2000):
        w_out_pocket, w_out = pocket_pla(x_train, y_train, max_iter=100)
        error_out_pocket = error_counter(x_test, y_test, w_out_pocket)
        error_out = error_counter(x_test, y_test, w_out)
        avg_pocket = avg_pocket + (1.0 / (index + 1)) * (error_out_pocket - avg_pocket)
        avg = avg + (1.0 / (index + 1)) * (error_out - avg)
    print(avg_pocket)
    print(avg)
