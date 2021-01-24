from tools import Pad_sym, Add_Guassian_Noise,evaluate
from BM3D_Step1 import BM3D_Step1
from BM3D_Step2 import BM3D_Step2
import numpy as np 
import cv2.cv2 as cv2

def BM3D(img_raw,color_mode,
                kHard = 8,NHard = 16,nHard = 16,pHard = 3,threshold_match_H = 2500,sigma_H = 20,control_p_H = 3,Hard_threshold = 2.7,Is_weight_usesd_H = 1,
                kEst = 8,nEst = 16,NEst = 8 , pEst = 3,threshold_match_W = 2500,sigma_W = 20,control_p_W = 3,Is_weight_usesd_W = 1,
                mode_H = 'DCT', mode_W = 'DCT',Is_kaiser = 1):
    '''
        对图像进行BM3D滤波
        color_mode: 1 为彩色模式，0 为灰度模式
    '''
    if(color_mode < 1):
        #灰度图
        img_raw = cv2.cvtColor(img_raw,cv2.COLOR_RGB2GRAY)
        img_raw_U = 0
        img_raw_V = 0
    else:
        #彩色图像转为YUV处理
        img_raw_YUV = cv2.cvtColor(img_raw, cv2.COLOR_RGB2YUV)
        img_raw = img_raw_YUV[:,:,0]
        img_raw_U = img_raw_YUV[:, :, 1]
        img_raw_V = img_raw_YUV[:, :, 2]
        img_raw_U = Pad_sym(img_raw_U, n_H)
        img_raw_V = Pad_sym(img_raw_V, n_H)
        

    img_raw_paded = Pad_sym(img_raw,n_H)
    img_raw_estimate = BM3D_Step1(img_raw_paded,
                                kHard = kHard,NHard = NHard,nHard = nHard,pHard = pHard,threshold_match = threshold_match_H,sigma = sigma_H,
                                mode = mode_H,control_p = control_p_H,Hard_threshold = Hard_threshold ,Is_kaiser = 1, Is_weight_usesd = Is_weight_usesd_H)
    img_raw_estimate = img_raw_estimate[n_H:-n_H,n_H:-n_H]  #去除增加的边缘

    img_raw_estimate_paded = Pad_sym(img_raw_estimate,n_H)
    
    img_result = BM3D_Step2(img_raw_estimate_paded, img_raw_paded, img_raw_U, img_raw_V,color_mode,
                                kEst = kEst,nEst = nEst,NEst = NEst , pEst = pEst,threshold_match = threshold_match_W,sigma = sigma_W,
                                mode = mode_W,control_p = control_p_W , Is_kaiser = Is_kaiser, Is_weight_usesd = Is_weight_usesd_W)
    if(color_mode > 0):
        img_result = img_result[n_W:-n_W, n_W:-n_W,:]  # 去除增加的边缘
    else:
        img_result = img_result[n_W:-n_W,n_W:-n_W]  #去除增加的边缘



    return img_raw_estimate,img_result


#超参数

#===========STEP1==========

#搜索区域的大小
n_H = 16
#图像块的大小
k_H = 8
#可匹配图像块数量的上届
N_H = 16
#选取图像块的步长
p_H = 3
#寻找匹配块的阈值
threshold_match_H = 2500
#硬滤波的阈值
Hard_threshold = 2.7 
#用于控制哈尔小波变换的层数
control_p_H = 3
#用于选择二维线性变换的方式： 'DCT','Haar'
mode_H = 'DCT'

#===========STEP2==========

#搜索区域的大小
n_W = 16
#图像块的大小
k_W = 8
#可匹配图像块数量的上届
N_W = 16
#选取图像块的步长
p_W = 3
#寻找匹配块的阈值
threshold_match_W = 400
#用于控制哈尔小波变换的层数
control_p_W = 3
#用于选择二维线性变换的方式： 'DCT','Haar'
mode_W = 'DCT'

#===========全局参数==========

#用于设置图像噪声
sigma = 2
#设定图像融合时是否要采用Kaiser窗滤波
Is_kaiser = 1

'''
#设定图像融合时的权重分配模式,1为使用SD，0为不使用
Is_weight_usesd = 1
#图片路径
IMG_PATH = 'Alley.jpg'
color_mode = 0

img_raw = cv2.imread(IMG_PATH,1)

img_noisy = Add_Guassian_Noise(img_raw,sigma)


img_H,img_W = BM3D(img_noisy,color_mode,
                        kHard = k_H , NHard = N_H,nHard = n_H , pHard = p_H,threshold_match_H = threshold_match_H,sigma_H = sigma,
                        control_p_H = control_p_H , Hard_threshold = Hard_threshold , Is_weight_usesd_H = Is_weight_usesd,
                        kEst = k_W,nEst = n_W,NEst = N_W , pEst = p_W,threshold_match_W = threshold_match_W,sigma_W = sigma,
                        control_p_W = control_p_W , Is_weight_usesd_W = Is_weight_usesd ,
                        mode_H = mode_H , mode_W = mode_W , Is_kaiser = Is_kaiser)

if color_mode > 0:
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2YUV)

    img_W = (np.clip(img_W, 0, 255)).astype(np.uint8)

    Psnr_W_Y = evaluate(img_raw[:,:,0], img_W[:,:,0])
    
    img_W = cv2.cvtColor(img_W, cv2.COLOR_YUV2RGB)

    if Is_weight_usesd > 0:
        img_W_name = '.\\output\\usesd\\' + IMG_PATH[:-4] + '_' + \
            str(sigma) + '_W_Y' + '%.4f' % Psnr_W_Y + '_useSD.jpg'
        cv2.imwrite(img_W_name, img_W)
    else:
        img_W_name = '.\\output\\unusesd\\' + IMG_PATH[:-4] + '_' + \
            str(sigma) + '_W_Y' + '%.4f' % Psnr_W_Y +  '.jpg'
        cv2.imwrite(img_W_name, img_W)        
else:
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)


    img_H = (np.clip(img_H, 0, 255)).astype(np.uint8)
    img_W = (np.clip(img_W, 0, 255)).astype(np.uint8)

    Psnr_H = evaluate(img_raw, img_H)
    Psnr_W = evaluate(img_raw, img_W)
    if Is_weight_usesd > 0:
        img_H_name = '.\\output\\usesd\\' + \
            IMG_PATH[:-4] + '_' + str(sigma) + '_H' + \
            '%.4f' % Psnr_H + '_useSD.jpg'
        cv2.imwrite(img_H_name, img_H)
        img_W_name = '.\\output\\usesd\\' + IMG_PATH[:-4] + '_' + str(sigma) + '_W' + '%.4f' % Psnr_W + '_useSD.jpg'
        cv2.imwrite(img_W_name, img_W)
    else:
        img_H_name = '.\\output\\unusesd\\' + IMG_PATH[:-4] + '_' + str(sigma) + '_H' + '%.4f' % Psnr_H + '.jpg'
        cv2.imwrite(img_H_name, img_H)
        img_W_name = '.\\output\\unusesd\\' + IMG_PATH[:-4] + '_' + str(sigma) + '_W' + '%.4f' % Psnr_W + '.jpg'

'''
'''

for sigma in {2,5,10,20,50,80}:
    for Is_weight_usesd in [0,1]:
        IMG_PATH = 'Baboon.jpg'
        color_mode = 1

        img_raw = cv2.imread(IMG_PATH, 1)
        img_noisy = Add_Guassian_Noise(img_raw,sigma)

        img_H,img_W = BM3D(img_noisy,color_mode,
                            kHard = k_H , NHard = N_H,nHard = n_H , pHard = p_H,threshold_match_H = threshold_match_H,sigma_H = sigma,
                            control_p_H = control_p_H , Hard_threshold = Hard_threshold , Is_weight_usesd_H = Is_weight_usesd,
                            kEst = k_W,nEst = n_W,NEst = N_W , pEst = p_W,threshold_match_W = threshold_match_W,sigma_W = sigma,
                            control_p_W = control_p_W , Is_weight_usesd_W = Is_weight_usesd ,
                            mode_H = mode_H , mode_W = mode_W , Is_kaiser = Is_kaiser)
        
        if color_mode > 0:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2YUV)

            img_W = (np.clip(img_W, 0, 255)).astype(np.uint8)

            Psnr_W_Y = evaluate(img_raw[:, :, 0], img_W[:, :, 0])

            img_W = cv2.cvtColor(img_W, cv2.COLOR_YUV2RGB)

            if Is_weight_usesd > 0:
                img_W_name = '.\\output\\usesd\\' + IMG_PATH[:-4] + '_' + \
                    str(sigma) + '_W_Y' + '%.4f' % Psnr_W_Y + '_useSD.jpg'
                cv2.imwrite(img_W_name, img_W)
            else:
                img_W_name = '.\\output\\unusesd\\' + IMG_PATH[:-4] + '_' + \
                    str(sigma) + '_W_Y' + '%.4f' % Psnr_W_Y + '.jpg'
                cv2.imwrite(img_W_name, img_W)
        else:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

            img_H = (np.clip(img_H, 0, 255)).astype(np.uint8)
            img_W = (np.clip(img_W, 0, 255)).astype(np.uint8)

            Psnr_H = evaluate(img_raw, img_H)
            Psnr_W = evaluate(img_raw, img_W)
            if Is_weight_usesd > 0:
                img_H_name = '.\\output\\usesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_H' + \
                    '%.4f' % Psnr_H + '_useSD.jpg'
                cv2.imwrite(img_H_name, img_H)
                img_W_name = '.\\output\\usesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_W' + \
                    '%.4f' % Psnr_W + '_useSD.jpg'
                cv2.imwrite(img_W_name, img_W)
            else:
                img_H_name = '.\\output\\unusesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_H' + '%.4f' % Psnr_H + '.jpg'
                cv2.imwrite(img_H_name, img_H)
                img_W_name = '.\\output\\unusesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_W' + '%.4f' % Psnr_W + '.jpg'
'''
sigma = 5
Is_weight_usesd = 1
for Hard_threshold in [3.2]:
    for Is_kaiser in [0,1]:
        IMG_PATH = 'Alley.jpg'
        color_mode = 0

        img_raw = cv2.imread(IMG_PATH, 1)
        img_noisy = Add_Guassian_Noise(img_raw,sigma)

        img_H,img_W = BM3D(img_noisy,color_mode,
                            kHard = k_H , NHard = N_H,nHard = n_H , pHard = p_H,threshold_match_H = threshold_match_H,sigma_H = sigma,
                            control_p_H = control_p_H , Hard_threshold = Hard_threshold , Is_weight_usesd_H = Is_weight_usesd,
                            kEst = k_W,nEst = n_W,NEst = N_W , pEst = p_W,threshold_match_W = threshold_match_W,sigma_W = sigma,
                            control_p_W = control_p_W , Is_weight_usesd_W = Is_weight_usesd ,
                            mode_H = mode_H , mode_W = mode_W , Is_kaiser = Is_kaiser)
        
        if color_mode > 0:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2YUV)

            img_W = (np.clip(img_W, 0, 255)).astype(np.uint8)

            Psnr_W_Y = evaluate(img_raw[:, :, 0], img_W[:, :, 0])

            img_W = cv2.cvtColor(img_W, cv2.COLOR_YUV2RGB)

            if Is_weight_usesd > 0:
                img_W_name = '.\\output\\usesd\\' + IMG_PATH[:-4] + '_' + \
                    str(sigma) + '_W_Y' + '%.4f' % Psnr_W_Y + '_useSD.jpg'
                cv2.imwrite(img_W_name, img_W)
            else:
                img_W_name = '.\\output\\unusesd\\' + IMG_PATH[:-4] + '_' + \
                    str(sigma) + '_W_Y' + '%.4f' % Psnr_W_Y + '.jpg'
                cv2.imwrite(img_W_name, img_W)
        else:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

            img_H = (np.clip(img_H, 0, 255)).astype(np.uint8)
            img_W = (np.clip(img_W, 0, 255)).astype(np.uint8)
0
            Psnr_H = evaluate(img_raw, img_H)
            Psnr_W = evaluate(img_raw, img_W)
            if Is_weight_usesd > 0:
                img_H_name = '.\\output\\usesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_H' + \
                    '%.4f' % Psnr_H + '_useSD.jpg'
                #cv2.imwrite(img_H_name, img_H)
                img_W_name = '.\\output\\usesd\\' + \
                    IMG_PATH[:-4] + '_' + str(Is_kaiser) + '_' + str(Hard_threshold) + '_' + str(sigma) + '_W' + \
                    '%.4f' % Psnr_W + '_useSD.jpg'
                cv2.imwrite(img_W_name, img_W)
            else:
                img_H_name = '.\\output\\unusesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_H' + '%.4f' % Psnr_H + '.jpg'
                cv2.imwrite(img_H_name, img_H)
                img_W_name = '.\\output\\unusesd\\' + \
                    IMG_PATH[:-4] + '_' + str(sigma) + '_W' + '%.4f' % Psnr_W + '.jpg'

