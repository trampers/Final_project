import numpy as np
import cv2.cv2 as cv2
from Haar import Haar,IHaar
from DCT_block_2 import DCT_block_filter,DCT,IDCT
from tools import weighting, ind_step, get_kaiserWindow, GetPatch, GetGroup
from Match import Match
from filter_Hadamard import filter_Hadamard


def BM3D_Step1(img_raw_paded,kHard = 8,NHard = 16,nHard = 16,pHard = 3,threshold_match = 2500,sigma = 20,mode = 'DCT',control_p = 3,Hard_threshold = 2.7,Is_kaiser = 1,Is_weight_usesd = 1):
    '''
        对图像进行第一部分的处理
        mode:对应第一阶段不同滤波方式
            DCT：DCT分块滤波
            Haar：哈尔小波
        K 为图像块的大小，N 为图像块可匹配块数的上限，n为待搜索区域的大小，p为选择待搜索区域的步长，threshold_match 为匹配的阈值
        control_p:哈尔变换的层数，越多越关注细节

        Hard_threshold:三维滤波硬阈值
        weight_mode : 相似度函数
        Is_kaiser: 图像融合时是否要进行滤波

    '''
    height, width = img_raw_paded.shape[0], img_raw_paded.shape[1]

    #row_ind,column_ind 为返回的图像块左上点的坐标，按照一定步长选定待匹配块
    row_ind = ind_step(height - kHard + 1, nHard, pHard)
    column_ind = ind_step(width - kHard + 1, nHard, pHard)

    #对图像进行匹配块搜索，K 为图像块的大小，N 为图像块可匹配块数的上限，n为待搜索区域的大小，threshold_match 为阈值
    #返回值为patch_pair (height - kHard + 1) * (height - kHard + 1) * （match_patch_i,match_patch_j）:每个图像块对应的匹配的块
    # patch_pair_count  (height - kHard + 1) * (height - kHard + 1) * numbers :每个图像块对应的匹配的块的个数
    patch_pair_points, patch_pair_count = Match(img_raw_paded, kHW=kHard, NHW=NHard, nHW=nHard, threshold_match=threshold_match)


    Group_num = int(np.sum(patch_pair_count))
    #用于存放所有匹配块的序列
    Group_3D_sequence = np.zeros((Group_num, kHard, kHard))
    #用于存放融合时的权重
    weight_sequence = np.zeros((height, width))
    
    #返回目标区域块()
    Patches = GetPatch(img_raw_paded, k=kHard)
    Patches_freq = DCT(Patches)
    Patches_freq = Patches_freq.reshape(
        (height - kHard + 1, height - kHard + 1, kHard, kHard))

    pointer = 0
    for row in row_ind:
        for col in column_ind:
            Sx_r = patch_pair_count[row, col]
            Group_3D = GetGroup(
                M=Patches_freq, points=patch_pair_points[row, col], num=Sx_r, k=kHard)

            #对匹配块进行三维滤波：哈达玛变换
            Group_3D, weight = filter_Hadamard(
                Group_3D, sigma, Hard_threshold, Is_weight_usesd)
            Group_3D = Group_3D.transpose((2, 0, 1))#变为Sx_r*k*k

            #if Is_weight_usesd == 1 ： 归一化权重分配
            if Is_weight_usesd > 0:
                weight = weighting(Group_3D,Is_weight_usesd)
                               
            weight_sequence[row, col] = weight
            Group_3D_sequence[pointer:pointer + Sx_r] = Group_3D
            pointer += Sx_r
    
    Group_3D_sequence = IDCT(Group_3D_sequence)

    #融合图像并输出
    cumulator = np.zeros_like(img_raw_paded, dtype=np.float64)
    coefficient = np.zeros((img_raw_paded.shape[0] - 2 * nHard, img_raw_paded.shape[1] - 2 * nHard), dtype=np.float64)
    coefficient = np.pad(coefficient, nHard, 'constant', constant_values=1.)
    pointer = 0

    #np.save('thresh_self.npy',patch_pair_count)

    for row in row_ind:
        for col in column_ind:
            Sx_r = patch_pair_count[row, col]
            Point = patch_pair_points[row, col]
            Group_3D = Group_3D_sequence[pointer:pointer + Sx_r]
            pointer += Sx_r
            weight = weight_sequence[row, col]
            for n in range(Sx_r):
                ni, nj = Point[n]
                patch = Group_3D[n]
                #进行kaiser窗滤波
                if (Is_kaiser > 0):
                    kaiserWindow = get_kaiserWindow(kHW = kHard)
                else:
                    kaiserWindow = 1
                cumulator[ni:ni + kHard, nj:nj + kHard] += patch * kaiserWindow * weight
                coefficient[ni:ni + kHard, nj:nj + kHard] += kaiserWindow * weight

    img_raw_estimate = cumulator / coefficient
    return img_raw_estimate
