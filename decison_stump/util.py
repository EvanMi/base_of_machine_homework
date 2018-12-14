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
