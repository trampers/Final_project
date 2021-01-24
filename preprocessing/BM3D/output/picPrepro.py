import cv2.cv2 as cv2
import numpy as np 

def picPrePro(IMG_PATH):
    img_original = imread(IMG_PATH)
    d = img_original.shpe[-1]
    color_mode = 1 #用于储存颜色模式的变量，1 为彩色，0 为灰色
    if(d > 1):
        #为三通道彩色
        color_mode = 1
        #转为YUV模式处理，利用Y模式的