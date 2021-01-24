import cv2.cv2 as cv2
import numpy as np 
import math
from DCT_block import DCT,IDCT

def artificialColor(img,mode):
    '''
        将灰度图像转为伪彩色
        img:输入图像
        mode：1：时域变换 2：频域变换
    '''
    if(mode > 0):
        img_R = get_img_R(img)
        img_G = get_img_G(img)
        img_B = get_img_B(img)
        img_result = np.array([img_R, img_G, img_B])
        img_result = img_result.transpose((1, 2, 0))
    else:
        img_freq = DCT(img)

        h, w = img_freq.shape

        threshold_h_m = math.sqrt(2)*6*h/16 # boundary between middle frequency and higher frequency
        
        threshold_m_l = math.sqrt(2)*4*h/16# boundary between lower frequency and middle frequency

        threshold_gap = math.sqrt(2)*3*h/16
        


        freq_high = np.zeros_like(img_freq, dtype=float)
        freq_middle = np.zeros_like(img_freq, dtype=float)
        freq_low = np.zeros_like(img_freq, dtype=float)

        for i in range(h):
            for j in range(w):
                distance = math.sqrt((i)**2 + (j)**2)
                if distance < threshold_m_l + threshold_gap:
                    freq_low[i, j] = img_freq[i, j]
                    freq_middle[i, j] = 0
                    freq_high[i, j] = 0
                if distance > threshold_m_l - threshold_gap:
                    freq_middle[i, j] = img_freq[i, j]
                if distance > threshold_h_m - threshold_gap:
                    freq_high[i, j] = img_freq[i, j]
                    if distance > threshold_h_m + threshold_gap:
                        freq_middle[i, j] = 0

        img_G = IDCT(freq_low)
        img_R = IDCT(freq_middle)
        img_B = IDCT(freq_high)
        img_result = np.array([img_R, img_G, img_B])
        img_result = img_result.transpose((1, 2, 0))
    return img_result


def get_img_R(img):
    img_R = np.zeros_like(img,dtype = np.uint8)
    img_R[img < 128] = 0
    img_R[img > 127] = 4*img[img > 127] - 511
    img_R[img > 191] = 255
    return img_R   


def get_img_G(img):
    img_G = np.zeros_like(img, dtype=np.uint8)
    img_G[img < 192] = 255
    img_G[img < 64] = 4*img[img < 64] 
    img_G[img > 191] = 1023 - img[img > 191] *4
    return img_G


def get_img_B(img):
    img_B = np.zeros_like(img, dtype=np.uint8)
    img_B[img > 127] = 0
    img_B[img < 128] = 511 - 4*img[img < 128]
    img_B[img < 64] = 255
    return img_B


mode = 1
IMG_PATH = 'Alley.jpg'

img = cv2.imread(IMG_PATH,0)
img_result = artificialColor(img,mode)

if(mode > 0):
    RESULT_PATH = IMG_PATH[:-4]+'_time.jpg'
    cv2.imwrite(RESULT_PATH,img_result)
else:
    RESULT_PATH = IMG_PATH[:-4]+'_freq.jpg'
    cv2.imwrite(RESULT_PATH,img_result)

