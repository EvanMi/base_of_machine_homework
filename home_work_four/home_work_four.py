# -*- coding:utf-8 -*-
# Author: Evan Mi
import numpy as np


def load_data(file_name):
    x = []
    y = []
    with open(file_name, 'r+') as f:
        for line in f:
            line = line.rstrip("\n")
            temp = line.split(" ")
            temp.insert(0, '1')
            x_temp = [float(val) for val in temp[:-1]]
            y_tem = [int(val) for val in temp[-1:]][0]
            x.append(x_temp)
            y.append(y_tem)

    nx = np.array(x)
    ny = np.array(y)
    return nx, ny


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


def get_w_reg(x, y, lambdas):
    w_reg = np.dot(np.linalg.pinv(np.dot(np.transpose(x), x) + lambdas * np.eye(np.size(x, axis=1))), np.dot(np.transpose(x), y))
    return w_reg.flatten()


def e_counter(x, y, w):
    local_result = sign_zero_as_neg(np.dot(x, w))
    e = np.where(local_result == y, 0, 1)
    return e.sum()/np.size(e)


def exe_13():
    print('#13:')
    train_x, train_y = load_data("data/train.txt")
    test_x, test_y = load_data("data/test.txt")
    w_reg_one = get_w_reg(train_x, train_y, 10)
    e_in = e_counter(train_x, train_y, w_reg_one)
    e_out = e_counter(test_x, test_y, w_reg_one)
    print("E_IN:", e_in)
    print("E_OUT:", e_out)


def exe_14_15():
    print('#14,#15')
    train_x, train_y = load_data("data/train.txt")
    test_x, test_y = load_data("data/test.txt")
    for i in range(-10, 3):
        lambda_tem = 10 ** i
        w_reg_tem = get_w_reg(train_x, train_y, lambda_tem)
        e_in_tem = e_counter(train_x, train_y, w_reg_tem)
        e_out_tem = e_counter(test_x, test_y, w_reg_tem)
        print("log_10(%d)" % i, e_in_tem, e_out_tem)


def exe_16_17():
    print('#16,17')
    x_tem, y_tem = load_data("data/train.txt")
    test_x, test_y = load_data("data/test.txt")
    train_x = x_tem[:120, :]
    val_x = x_tem[120:, :]
    train_y = y_tem[:120]
    val_y = y_tem[120:]
    for i in range(-10, 3):
        lambda_tem = 10 ** i
        w_reg_tem = get_w_reg(train_x, train_y, lambda_tem)
        e_in_tem = e_counter(train_x, train_y, w_reg_tem)
        e_val_tem = e_counter(val_x, val_y, w_reg_tem)
        e_out_tem = e_counter(test_x, test_y, w_reg_tem)
        print("log_10(%d)" % i, e_in_tem, e_val_tem, e_out_tem)


def exe_18():
    print('#18:')
    train_x, train_y = load_data("data/train.txt")
    test_x, test_y = load_data("data/test.txt")
    # lambda = log_10(0)
    w_reg_one = get_w_reg(train_x, train_y, 1)
    e_in = e_counter(train_x, train_y, w_reg_one)
    e_out = e_counter(test_x, test_y, w_reg_one)
    print("E_IN:", e_in)
    print("E_OUT:", e_out)


def exe_19():
    print('#19')
    train_x, train_y = load_data("data/train.txt")
    for i in range(-10, 3):
        lambda_tem = 10 ** i
        e_cross = []
        for j in range(0, 200, 40):
            x_val = train_x[j:j+40, :]
            y_val = train_y[j:j+40]
            x_remain_left = train_x[0:j, :]
            x_remain_right = train_x[j+40:, :]
            y_remain_left = train_y[0:j]
            y_remain_right = train_y[j + 40:]
            if np.size(x_remain_left, axis=0) == 0:
                x_train = x_remain_right
                y_train = y_remain_right
            elif np.size(x_remain_right, axis=0) == 0:
                x_train = x_remain_left
                y_train = y_remain_left
            else:
                x_train = np.concatenate((train_x[0:j, :], train_x[j + 40:, :]), axis=0)
                y_train = np.concatenate((train_y[0:j], train_y[j + 40:]), axis=0)

            w_reg_tem = get_w_reg(x_train, y_train, lambda_tem)
            e_cross.append(e_counter(x_val, y_val, w_reg_tem))
        print("lambda:", "log_10(%d)" % i, "E_CV", np.array(e_cross).mean())


def exe_20():
    print('#20:')
    train_x, train_y = load_data("data/train.txt")
    test_x, test_y = load_data("data/test.txt")
    # lambda = log_10(-8)
    w_reg_one = get_w_reg(train_x, train_y, 10 ** -8)
    e_in = e_counter(train_x, train_y, w_reg_one)
    e_out = e_counter(test_x, test_y, w_reg_one)
    print("E_IN:", e_in)
    print("E_OUT:", e_out)


if __name__ == '__main__':
    exe_13()
    exe_14_15()
    exe_16_17()
    exe_18()
    exe_19()
    exe_20()

