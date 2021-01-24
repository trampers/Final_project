import numpy as np
from scipy.linalg import hadamard
import math


def filter_Hadamard(Group_3D, sigma, Hard_threshold, Is_weight_usesd):  # Group_3D shape=(n*n, Sx_r)
    '''
        对第三维进行哈达马变换
    '''
    Sx_r = Group_3D.shape[-1]
    coef_norm = math.sqrt(Sx_r)
    coef = 1.0 / Sx_r

    Group_3D_h = hadamard_transform(Group_3D)

    threshold = Hard_threshold * sigma * coef_norm
    #用于统计权重
    threshold_num = np.where(np.abs(Group_3D_h) > threshold, 1, 0)
    weight = np.sum(threshold_num)
    #滤去低于Hard_threshold 的值
    Group_3D_h = np.where(np.abs(Group_3D_h) > threshold, Group_3D_h, 0.)
    Group_3D = hadamard_transform(Group_3D_h) * coef


    if Is_weight_usesd < 1 :
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return Group_3D, weight


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n).astype(np.float64)
    v_h = vec @ h_mat
    return v_h
