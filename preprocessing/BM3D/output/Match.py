import numpy as np


def Match(img, kHW, NHW, nHW, threshold_match):
    '''
        对每一块目标区域寻找对应的匹配块
        kHW:寻找块的大小
        nHW:len of search area
        返回值：
        ri_rj_N__ni_nj：相似块的左上角坐标，以矩阵形式
        threshold：协同Hard阈值
        threshold_count:匹配的个数
    '''
    img = img.astype(np.float64)
    height, width = img.shape
    Ns = 2 * nHW + 1
    threshold = threshold_match * kHW * kHW
    sum_table = np.ones((Ns, Ns, height, width)) * 2 * threshold  # di, dj, ph, pw
    #为了加速需要使用矩阵运算
    add_mat = get_add_patch_matrix(width, nHW, kHW)
    diff_margin = np.pad(np.ones((height - 2 * nHW, width - 2 * nHW)), nHW, 'constant', constant_values=0.)
    sum_margin = (1 - diff_margin) * 2 * threshold

    #sum_table 记录相似度
    for di in range(-nHW, nHW + 1):
        for dj in range(-nHW, nHW + 1):
            t_img = translation_2d_mat(img, right=-dj, down=-di)
            diff_table_2 = (img - t_img) * (img - t_img) * diff_margin

            sum_diff_2 = add_mat @ diff_table_2 @ add_mat.T
            sum_table[di + nHW, dj + nHW] = np.maximum(sum_diff_2, sum_margin)  # sum_table (2n+1, 2n+1, height, width)

    sum_table = sum_table.reshape((Ns * Ns, height * width))  # di_dj, ph_pw
    sum_table_T = sum_table.transpose((1, 0))  # 对sum_table进行转置 
    argsort = np.argpartition(sum_table_T, range(NHW))[:, :NHW]#得到前NHW个小的数，并依次返回他们在原数组的索引
    argsort[:, 0] = (Ns * Ns - 1) // 2
    argsort_di = argsort // Ns - nHW
    argsort_dj = argsort % Ns - nHW
    near_pi = argsort_di.reshape((height, width, -1)) + np.arange(height)[:, np.newaxis, np.newaxis]
    near_pj = argsort_dj.reshape((height, width, -1)) + np.arange(width)[np.newaxis, :, np.newaxis]
    ri_rj_N__ni_nj = np.concatenate((near_pi[:, :, :, np.newaxis], near_pj[:, :, :, np.newaxis]), axis=-1)

    sum_filter = np.where(sum_table_T < threshold, 1, 0)
    threshold_count = np.sum(sum_filter, axis=1)
    threshold_count = closest_power_of_2(threshold_count, max_=NHW)
    threshold_count = threshold_count.reshape((height, width))

    return ri_rj_N__ni_nj, threshold_count


def get_add_patch_matrix(n, nHW, kHW):
    """
    :param n: len of mat
    :param nHW: len of search area
    :param kHW: len of patch
    :return: manipulate mat
    """
    mat = np.eye(n - 2 * nHW)
    mat = np.pad(mat, nHW, 'constant')
    res_mat = mat.copy()
    for k in range(1, kHW):
        res_mat += translation_2d_mat(mat, right=k, down=0)
    return res_mat


def translation_2d_mat(mat, right, down):
    mat = np.roll(mat, right, axis=1)
    mat = np.roll(mat, down, axis=0)
    return mat


def closest_power_of_2(M, max_):
    '''
        找出M各元素在小于max_的2的幂中最接近的一个
    '''
    M = np.where(max_ < M, max_, M) #等价于三元运算符
    while max_ > 1:
        M = np.where((max_ // 2 < M) * (M < max_), max_ // 2, M)
        max_ //= 2
    return M

