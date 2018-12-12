# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np
from pla_and_pocket_pla import util


def pla(nx, ny, rate=1):
    """
    :param nx: 属性矩阵，格式是[[...],[...],[...]]
    :param ny: 值，格式是 [[.],[.],[.]]
    :param rate: 学习率，默认是1
    :return: 迭代次数
    """
    total_update_nums = 0
    total_train_example_nums = np.size(nx, 0)
    continue_right_nums = 0  # 连续不犯错的次数，当continue_right_nums==total_train_example_nums的时候，程序结束
    w = np.zeros((1, 5))  # 初始化参数为0
    loop_index = 0
    while True:
        this_x = nx[loop_index]
        result = util.sign(np.dot(this_x, w.T)[0])
        this_y = ny[loop_index, 0]
        if result == this_y:
            continue_right_nums += 1
        else:
            continue_right_nums = 0
            w = w + rate * (this_x * this_y)
            total_update_nums += 1
        loop_index = (loop_index + 1) % total_train_example_nums

        if continue_right_nums == total_train_example_nums:
            break
    return total_update_nums


if __name__ == '__main__':
    """
    这里展示的是随机打乱样本，以0.5的学习率运行1000次的结果
    """
    out_nx, out_ny = util.load_data("data/data.txt")
    avg = 0
    for i in range(1000):
        shuffle_index = np.arange(0, np.size(out_nx, 0))
        np.random.shuffle(shuffle_index)
        shuffled_x = out_nx[shuffle_index]
        shuffled_y = out_ny[shuffle_index]
        result_out = pla(shuffled_x, shuffled_y, 0.5)
        print("第%d次的更新次数为：%d" % ((i + 1), result_out))
        avg = avg + (1.0 / (i + 1)) * (result_out - avg)
    print("平均迭代次数为：%d" % avg)
