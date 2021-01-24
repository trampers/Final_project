import numpy as np 
import math

def Pad_sym(img,N):
    '''
        对图像边缘以对称映射方式填充
    ''' 
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad

def Add_Guassian_Noise(img, sigma):
    '''
        为图像添加高斯噪声
    '''
    img = img + (sigma * np.random.randn(*img.shape)).astype(np.int)
    #限制数据为0~255
    img = np.clip(img, 0., 255., out=None)
    img = img.astype(np.uint8)
    return img


def evaluate(img1, img2):
    '''
        通过返回img2相对于img1的PSNR值评估图像对比质量
    '''
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "Exactly Same"
    return 10 * math.log10(1. / mse)

def get_transport_mat(im_s, k):
    '''
        返回同时在两个轴上循环位移k的矩阵
    '''
    temp = np.zeros((im_s, (im_s - k + 1) * k), dtype=np.int)
    for i in range(k):
        temp[i, i] = 1
    Trans = temp.copy()
    for i in range(1, im_s - k + 1):
        dT = np.roll(temp, i, axis=0)
        dT = np.roll(dT, i * k, axis=1)
        Trans += dT
    return Trans

def get_allpatches(img_noisy,k):
    '''
        依次返回(h-k+1)*(w-k+1)*(k*k)
        k*k是以(x,y)为左上角坐标的图像块
    '''
    height,width = img_noisy.shape
    allPatches = np.zeros((height - k + 1 , width - k + 1 , k , k))
    for i in range(height - k + 1):
        for j in range(width - k + 1):
            allPatches[i,j] = img_noisy[i:i+k,j:j+k].copy()
    return allPatches


def weighting(group_3D,Is_weight_usesd):
    '''
        归一化权重
    '''
    if(Is_weight_usesd > 0):
        N = group_3D.size

        mean = np.sum(group_3D)
        std = np.sum(group_3D * group_3D)

        res = (std - mean * mean / N) / (N - 1)
        weight = 1.0 / np.sqrt(res) if res > 0. else 0.
    return weight

def ind_step(max_size, N, step):
    '''
        用于按步长返回目标区域点
    '''
    ind = range(N, max_size - N, step)
    if ind[-1] < max_size - N - 1:
        ind = np.append(ind, np.array([max_size - N - 1]), axis=0)
    return ind

def get_kaiserWindow(kHW):
    '''
        返回kaiser窗
    '''
    k = np.kaiser(kHW, 2)
    k_2d = k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


def GetPatch(img, k):
    '''
        返回目标区域块集合，便于之后的处理
    '''

    '''
    h,w = img.shape
    Patches = np.zeros((k*(h-k+1),k*(w-k+1)),dtype = np.float32)
    for row in range(h):
        for col in range(w):
            Patches[row*k:(row+1)*k,col*k:(col+1)*k] = img[row:row+k,col:col+k]

    return Patches
    弃用，时间复杂度太太高，转用矩阵形式
    '''

    h = img.shape[0]

    Trans = np.zeros((h, (h - k + 1) * k), dtype=np.int)
    temp = Trans.copy()
    for i in range(k):
        temp[i, i] = 1
    for i in range(1, h - k + 1):
        Trans_tmp = np.roll(temp, i, axis=0)
        Trans_tmp = np.roll(Trans_tmp, i * k, axis=1)
        Trans += Trans_tmp
    Patches = Trans.T @ img @ Trans
    Patches = Patches.reshape((h - k + 1, k, h - k + 1, k))
    Patches = Patches.transpose((0, 2, 1, 3))
    Patches = Patches.reshape((-1, k, k))

    return Patches

def GetGroup(M,points,num,k):
    '''
        返回对应点的匹配块组 k*k*num
    '''
    Group_3D = np.zeros((num, k, k))
    for i in range(num):
        row,col = points[i]
        Group_3D[i, :, :] = M[row, col]
    Group_3D = Group_3D.transpose((1, 2, 0))
    return Group_3D
