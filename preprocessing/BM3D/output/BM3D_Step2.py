import numpy as np
import cv2.cv2 as cv2
from Haar import Haar,IHaar
from DCT_block_2 import DCT_block_filter,DCT,IDCT
from tools import weighting, ind_step, get_kaiserWindow, GetPatch, GetGroup
from Match import Match
from filter_Hadamard import filter_Hadamard
from filter_Wiener import filter_Wiener

def BM3D_Step2(img_raw_estimate_paded,img_raw_paded,img_raw_U = 0,img_raw_V = 0 , color_mode = 0 , 
                kEst = 8,nEst = 16,NEst = 8 , pEst = 3,threshold_match = 2500,sigma = 20,
                mode = 'DCT',control_p = 3,Is_kaiser = 1, Is_weight_usesd = 1):
    '''
        进行第二部分处理
        img_raw_estimate_paded:经过第一部分处理的估计图像
        img_raw_paded:原图像
        color_mode :图像颜色模式，1 为彩色
    '''

    height, width = img_raw_paded.shape[0], img_raw_paded.shape[1]

    #row_ind,column_ind 为返回的图像块左上点的坐标，按照一定步长选定待匹配块
    row_ind = ind_step(height - kEst + 1, nEst, pEst)
    column_ind = ind_step(width - kEst + 1, nEst, pEst)

    #对图像进行匹配块搜索，K 为图像块的大小，N 为图像块可匹配块数的上限，n为待搜索区域的大小，threshold_match 为阈值
    #返回值为patch_pair (height - kEst + 1) * (height - kEst + 1) * （match_patch_i,match_patch_j）:每个图像块对应的匹配的块
    # patch_pair_count  (height - kEst + 1) * (height - kEst + 1) * numbers :每个图像块对应的匹配的块的个数
    patch_pair_points, patch_pair_count = Match(
        img_raw_estimate_paded, kHW=kEst, NHW=NEst, nHW=nEst, threshold_match=threshold_match)

    Group_num = int(np.sum(patch_pair_count))
    #用于存放所有匹配块的序列
    Group_3D_sequence = np.zeros((Group_num, kEst, kEst))
    #用于存放融合时的权重，三通道则共用
    weight_sequence = np.zeros((height, width))

    #对于三通道图像，构造相应序列
    if(color_mode>0):
        Group_3D_sequence_U = np.zeros((Group_num, kEst, kEst))
        Group_3D_sequence_V = np.zeros((Group_num, kEst, kEst))

    #返回目标区域块()
    Patches = GetPatch(img_raw_paded, k=kEst)
    Patches_estimate = GetPatch(img_raw_estimate_paded, k=kEst)

    #对于三通道图像，相应返回同一位置的目标区域块
    if(color_mode > 0):
        Patches_U = GetPatch(img_raw_U, k=kEst)
        Patches_U = Patches_U.reshape(
            (height - kEst + 1, height - kEst + 1, kEst, kEst))
        Patches_V = GetPatch(img_raw_V, k=kEst)
        Patches_V = Patches_V.reshape(
            (height - kEst + 1, height - kEst + 1, kEst, kEst))
    
    #同时对去噪估计图像和噪声图像做二维线性变换
    Patches_freq = DCT(Patches)
    Patches_freq = Patches_freq.reshape(
        (height - kEst + 1, height - kEst + 1, kEst, kEst))
        
    Patches_freq_estimate = DCT(Patches_estimate)
    Patches_freq_estimate = Patches_freq_estimate.reshape(
        (height - kEst + 1, height - kEst + 1, kEst, kEst))
        
    pointer = 0
    for row in row_ind:
        for col in column_ind:
            Sx_r = patch_pair_count[row, col]
            Group_3D = GetGroup(
                M=Patches_freq, points=patch_pair_points[row, col], num=Sx_r, k=kEst) 
            Group_3D_estimate = GetGroup(
                M=Patches_freq_estimate, points=patch_pair_points[row, col], num=Sx_r, k=kEst)
            if(color_mode > 0):
                Group_3D_raw_U = GetGroup(
                    M=Patches_U, points=patch_pair_points[row, col], num=Sx_r, k=kEst)
                Group_3D_raw_V = GetGroup(
                    M=Patches_V, points=patch_pair_points[row, col], num=Sx_r, k=kEst)

            #对匹配块进行三维滤波：哈达玛变换

            Group_3D, weight = filter_Wiener(
                Group_3D, Group_3D_estimate, sigma,Is_weight_usesd)
            Group_3D = Group_3D.transpose((2, 0, 1))#变为Sx_r*k*k
            if(color_mode > 0):
                Group_3D_raw_U = Group_3D_raw_U.transpose((2, 0, 1))
                Group_3D_raw_V = Group_3D_raw_V.transpose((2, 0, 1))
            #if Is_weight_usesd == 1 ： 归一化权重分配
            if Is_weight_usesd > 0:
                weight = weighting(Group_3D,Is_weight_usesd)
                               
            weight_sequence[row, col] = weight
            Group_3D_sequence[pointer:pointer + Sx_r] = Group_3D

            if(color_mode > 0):
                Group_3D_sequence_U[pointer:pointer + Sx_r] = Group_3D_raw_U
                Group_3D_sequence_V[pointer:pointer + Sx_r] = Group_3D_raw_V
            pointer += Sx_r

    Group_3D_sequence = IDCT(Group_3D_sequence)

    #融合图像并输出
    cumulator = np.zeros_like(img_raw_paded, dtype=np.float64)
    coefficient = np.zeros((img_raw_paded.shape[0] - 2 * nEst, img_raw_paded.shape[1] - 2 * nEst), dtype=np.float64)
    coefficient = np.pad(coefficient, nEst, 'constant', constant_values=1.)
    pointer = 0

    if(color_mode > 0):
        cumulator_U = cumulator.copy()
        cumulator_V = cumulator.copy()


    for i_r in row_ind:
        for j_r in column_ind:
            Sx_r = patch_pair_count[i_r, j_r]
            Point = patch_pair_points[i_r, j_r]
            Group_3D = Group_3D_sequence[pointer:pointer + Sx_r]

            if(color_mode > 0):
                Group_3D_raw_U = Group_3D_sequence_U[pointer:pointer + Sx_r]
                Group_3D_raw_V = Group_3D_sequence_V[pointer:pointer + Sx_r]

            pointer += Sx_r
            weight = weight_sequence[i_r, j_r]
            for n in range(Sx_r):
                ni, nj = Point[n]
                patch = Group_3D[n]
                if(color_mode > 0):
                    patch_U = Group_3D_raw_U[n]
                    patch_V = Group_3D_raw_V[n]
                #进行凯撒窗滤波
                if (Is_kaiser > 0):
                    kaiserWindow = get_kaiserWindow(kHW=kEst)
                else:
                    kaiserWindow = 1
                if(color_mode > 0):
                    cumulator[ni:ni + kEst, nj:nj + kEst] += patch * kaiserWindow * weight
                    coefficient[ni:ni + kEst, nj:nj + kEst] += kaiserWindow * weight
                    cumulator_U[ni:ni + kEst, nj:nj + kEst] += patch_U * kaiserWindow * weight
                    cumulator_V[ni:ni + kEst, nj:nj + kEst] += patch_V * kaiserWindow * weight                                
                else:
                    cumulator[ni:ni + kEst, nj:nj + kEst] += patch * kaiserWindow * weight
                    coefficient[ni:ni + kEst, nj:nj + kEst] += kaiserWindow * weight
    if(color_mode > 0):
        img_raw_estimate = np.zeros((height, width,3),dtype = float)
        img_raw_estimate[:,:,0] = cumulator / coefficient
        img_raw_estimate[:,:,1] = cumulator_U / coefficient
        img_raw_estimate[:,:,2] = cumulator_V / coefficient
        
    else:
        img_raw_estimate = cumulator / coefficient

    return img_raw_estimate
