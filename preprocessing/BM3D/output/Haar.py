import cv2.cv2 as cv2
import numpy as np 

def Haar_forward(img):
    '''
        对图像进行哈尔小波变换
    '''
    h,w = img.shape
    tmp = np.ones((h,w),dtype = np.float32)
    wavelet = np.ones((h,w),dtype = np.float32)
    img_copy = img.copy()

    height = h
    width = w
    #先进行列变换
    for i in range(width):
        for j in range(height// 2):
            tmp[i,j] = (img_copy[i,2*j]+img_copy[i,2*j+1]) / 2
            tmp[i,j + height // 2 ] = (img_copy[i,2*j] - img_copy[i,2*j+1]) / 2
    #再进行行变换
    for i in range(width//2):
        for j in range(height):
            wavelet[i,j] = (tmp[2*i,j]+tmp[2*i +1,j]) / 2
            wavelet[i + width // 2 ,j ] = (tmp[2*i,j] - tmp[2*i + 1,j]) / 2
    return wavelet

def Haar_backward(img):
    '''
        对图像进行哈尔小波逆变换
    '''
    h,w = img.shape
    tmp = np.ones((h,w),dtype = np.float32)
    wave_recover = np.ones((h,w),dtype = np.float32)
    img_copy = img.copy()

    height = h
    width = w

    #先进行列逆变换
    for i in range(width):
        for j in range(height// 2):
            tmp[i,2*j] = img_copy[i,j] + img_copy[i,j + height // 2 ]
            tmp[i,2*j + 1] = img_copy[i,j] - img_copy[i,j + height // 2 ]
    #再进行行逆变换
    for i in range(width//2):
        for j in range(height):
            wave_recover[2*i,j] = tmp[i,j] + tmp[i + width //2 ,j]
            wave_recover[2*i + 1,j] = tmp[i,j] - tmp[i + width //2 ,j]
    return wave_recover

def Haar(img,depth = 3):
    '''
        进行哈尔小波变换
        depth：哈尔小波分解的层数，相当于进行变换的次数
    '''
    img = np.array(img)

    h,w = img.shape
    h_raw = h
    w_raw = w
    #向上对8的倍数取整，以保证准确
    if np.mod(h_raw,8) < 1:
        tmp_h = h_raw
    else:
        tmp_h = (h_raw//8 + 1)*8
    if np.mod(w_raw,8) < 1:
        tmp_w = w_raw
    else:
        tmp_w = (w_raw//8 + 1)*8
    img_t = cv2.resize(img,(tmp_w,tmp_h))#切记opencv有时候定义是反的
    
    #用于记录每次变换输入矩阵的大小
    h = np.zeros(depth).astype(int)
    w = np.zeros(depth).astype(int)

    #正变换
    #哈尔变换会出现小数，为避免自动取整，需要先转为整型
    img_t = img_t.astype(float)
    #用于存储最终变换结果
    Haar = img_t.copy()
    Haar_in = img_t.copy()
    
    for i in range(depth):
        h[i],w[i] = Haar_in.shape
        Haar_out = Haar_forward(Haar_in)
        Haar[0:h[i],0:w[i]] = Haar_out
        #提取下一层变换的输入，为这一层输出的左上角,在最后一次循环时不用迭代
        if(i < depth-1):
            Haar_in = Haar_out[0:h[i]//2,0:w[i]//2]
    return Haar,h,w

def IHaar(Haar,h,w,depth):
    '''
        进行哈尔小波反变换
        h,w：之前哈尔变换时每次变换的尺度
    '''
    IHaar = Haar.copy().astype(float)
    IHaar_in =Haar.copy().astype(float)
    
    for i in range(depth):
        #相当于是反推
        IHaar_in = IHaar[0:h[-(i+1)],0:w[-(i+1)]].copy()
        IHaar_out = Haar_backward(IHaar_in)
        IHaar[0:h[-(i+1)],0:w[-(i+1)]] = IHaar_out
        #提取下一层逆变换的输入，为这一层输出的向右向下延拓,在最后一次循环时不用迭代
        if(i<depth-1):
            IHaar_in = IHaar[0:h[-(i+2)],0:w[-(i+2)]]

    img_t = cv2.resize(IHaar,(w[0],h[0]))

    return  img_t

def Haar_filter(img,depth,threshold):
    '''
        哈尔小波变换滤波的顶层函数
        depth：哈尔小波分解的层数，相当于进行变换的次数
        threshold: 所有变换结果中<threshold 的值都会被置为0
    '''
    h_raw,w_raw = img.shape
    #向上对8的倍数取整，以保证准确
    if np.mod(h_raw,8) < 1:
        tmp_h = h_raw
    else:
        tmp_h = (h_raw//8 + 1)*8
    if np.mod(w_raw,8) < 1:
        tmp_w = w_raw
    else:
        tmp_w = (w_raw//8 + 1)*8
    img_t = cv2.resize(img,(tmp_w,tmp_h))#切记opencv有时候定义是反的
    
    #用于记录每次变换输入矩阵的大小
    h = np.zeros(depth).astype(int)
    w = np.zeros(depth).astype(int)

    #正变换
    #哈尔变换会出现小数，为避免自动取整，需要先转为整型
    img_t = img_t.astype(float)
    #用于存储最终变换结果
    Haar = img_t.copy()
    Haar_in = img_t.copy()
    
    for i in range(depth):
        h[i],w[i] = Haar_in.shape
        Haar_out = Haar_forward(Haar_in)
        Haar[0:h[i],0:w[i]] = Haar_out
        #提取下一层变换的输入，为这一层输出的左上角,在最后一次循环时不用迭代
        if(i < depth-1):
            Haar_in = Haar_out[0:h[i]//2,0:w[i]//2]


    #根据设定阈值进行滤波
    Haar[abs(Haar) < threshold] = 0       

    #逆变换
    #用于存储最终逆变换结果
    IHaar = Haar.copy().astype(float)
    IHaar_in =Haar.copy().astype(float)
    
    for i in range(depth):
        #相当于是反推
        IHaar_in = IHaar[0:h[-(i+1)],0:w[-(i+1)]].copy()
        IHaar_out = Haar_backward(IHaar_in)
        IHaar[0:h[-(i+1)],0:w[-(i+1)]] = IHaar_out
        #提取下一层逆变换的输入，为这一层输出的向右向下延拓,在最后一次循环时不用迭代
        if(i<depth-1):
            IHaar_in = IHaar[0:h[-(i+2)],0:w[-(i+2)]]

    img_t = cv2.resize(IHaar,(w_raw,h_raw))

    return img_t
