import numpy as np 

def GetPatch(img,k):
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
        temp[i,i] = 1
    for i in range(1, h - k + 1):
        Trans_tmp = np.roll(temp, i, axis=0)
        Trans_tmp = np.roll(Trans_tmp, i * k, axis=1)
        Trans += Trans_tmp
    Patches = Trans.T @ img @ Trans
    Patches = Patches.reshape((h - k + 1, k, h - k + 1, k))
    Patches = Patches.transpose((0, 2, 1, 3))
    Patches = Patches.reshape((-1, k, k))
    
    return Patches
