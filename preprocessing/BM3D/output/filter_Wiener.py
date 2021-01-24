import numpy as np
from scipy.linalg import hadamard


def filter_Wiener(Group_3D_img, Group_3D_est, sigma, Is_weight_usesd):
    '''
        先对第三维进行哈达玛变换
        进行维纳滤波
        w = img_est**2 /(img_est**2 + sigma**2) 
    '''
    assert Group_3D_img.shape == Group_3D_est.shape
    Sx_r = Group_3D_img.shape[-1]
    coef = 1.0 / Sx_r
    #更好地利用稀疏性
    Group_3D_img_h = hadamard_transform(Group_3D_img) 
    Group_3D_est_h = hadamard_transform(Group_3D_est)

    value = np.power(Group_3D_est_h, 2) * coef
    value /= (value + sigma * sigma)
    Group_3D_est_h = Group_3D_img_h * value * coef
    weight = np.sum(value)

    Group_3D_est = hadamard_transform(Group_3D_est_h)

    if Is_weight_usesd < 1:
        weight = 1. / (sigma * sigma * weight) if weight > 0. else 1.

    return Group_3D_est, weight


def hadamard_transform(vec):
    n = vec.shape[-1]
    h_mat = hadamard(n).astype(np.float64)
    v_h = vec @ h_mat
    return v_h
