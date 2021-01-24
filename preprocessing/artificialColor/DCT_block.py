import numpy as np 
import cv2.cv2 as cv2
from math import sqrt,cos,pi

def DCT(img):
    '''
        进行DCT变换
    '''
    h,w = img.shape
    Eh = np.zeros((h,h),dtype = float)
    Ew = np.zeros((w,w),dtype = float)     #Assistance Caculating Matrix
    Gh = np.ones((h,h),dtype = float)/sqrt(h)
    Gw = np.ones((w,w),dtype = float)/sqrt(w)  #G Caculating Matrix
    Freq_proc = np.zeros((h,w),dtype = np.float32)   #Frequency domain result

    for row in range(1,h):  
        for col  in range(0,w):
            Eh[row,col] = (row) * (col+1/2)
            Gh[row,col] = sqrt(2)*cos(pi*Eh[row,col]/h)

    for row in range(0,h):  
        for col  in range(1,w):
            Ew[row,col] = (row+1/2) * (col)
            Gw[row,col] = sqrt(2) * cos(pi*Ew[row,col]/w)

    Freq_proc = Gh @ img @ Gw
    return Freq_proc

def IDCT(Freq):
    '''
        进行IDCT变换
    '''
    h,w = Freq.shape
    Eh = np.zeros((h,h),dtype = float)
    Ew = np.zeros((w,w),dtype = float)     #Assistance Caculating Matrix
    Gh = np.ones((h,h),dtype = float)/sqrt(h)
    Gw = np.ones((w,w),dtype = float)/sqrt(w)  #G Caculating Matrix
    img_result = np.zeros((h,w),dtype = np.float32)   #Frequency domain result

    for row in range(1,h):  
        for col  in range(0,w):
            Eh[row,col] = (row) * (col+1/2)
            Gh[row,col] = sqrt(2)*cos(pi*Eh[row,col]/h)
    Dh = np.linalg.inv(Gh)


    for row in range(0,h):  
        for col  in range(1,w):
            Ew[row,col] = (row+1/2) * (col)
            Gw[row,col] = sqrt(2) * cos(pi*Ew[row,col]/w)
    Dw = np.linalg.inv(Gw)

    img_result = Dh @ Freq @ Dw
    return img_result

def DCT_block_filter(img,N):
    '''
        对图像进行8x8分块DCT滤波
        N：保留前八条对角线中的N条信息
    '''
    h,w = img.shape
    #向上对8的倍数取整，以保证准确
    if np.mod(h,8) < 1:
        tmp_h = h
    else:
        tmp_h = (h//8 + 1)*8
    if np.mod(w,8) < 1:
        tmp_w = w
    else:
        tmp_w = (w//8 + 1)*8
    img_t = cv2.resize(img,(tmp_w,tmp_h))#切记opencv有时候定义是反的

    Eh = np.zeros((8,8),dtype = float)
    Ew = np.zeros((8,8),dtype = float)     #Assistance Caculating Matrix
    Gh = np.ones((8,8),dtype = float)/sqrt(tmp_h)
    Gw = np.ones((8,8),dtype = float)/sqrt(tmp_w)  #G Caculating Matrix
    Freq_proc = np.zeros((h,w),dtype = np.float32)   #Frequency domain result


    for row in range(1,8):  
        for col  in range(0,8):
            Eh[row,col] = (row) * (col+1/2)
            Gh[row,col] = sqrt(2)*cos(pi*Eh[row,col]/8)

    Dh = np.linalg.inv(Gh)

    for row in range(0,8):  
        for col  in range(1,8):
            Ew[row,col] = (row+1/2) * (col)
            Gw[row,col] = sqrt(2) * cos(pi*Ew[row,col]/8)

    Dw = np.linalg.inv(Gw)
        
    mask = np.zeros((8,8),float)
    #mask用于滤去高次噪声
    for row in range(0,N):
        for col in range(0,N-row-1):
            mask[row,col] = 1

    img_result = np.zeros((tmp_h,tmp_w),dtype = np.float32)
    tmp = np.zeros((8,8),dtype = np.float32)

    for i in range(tmp_h // 8):
        for j in range(tmp_w // 8):
            img_proc = img_t[i*8:(i+1)*8,j*8:(j+1)*8]
            #print(img_proc)
            Freq_proc = Gh @ img_proc @ Gw
            Freq_proc = Freq_proc * mask
            tmp = Dh @ Freq_proc @ Dw
            img_result[i*8:(i+1)*8,j*8:(j+1)*8] = tmp
    
    img_result = cv2.resize(img_result,(w,h))
    return img_result
    


