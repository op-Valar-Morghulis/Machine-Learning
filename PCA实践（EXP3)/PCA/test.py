# 加载数据集
# 利用 PCA 算法对数据集内所有人进行降维和特征提取
# 然后将得到的主成分特征向量还原成图像进行观察
# 这里可以尝试采用不同的降维维度 K 进行操作，分别观 察不同 K 下的特征图像

# 拓展实验
# 尝试对刚降维的特征图像进行 PCA 逆变换，观察变换前后的图像差异

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import TruncatedSVD
from numpy import linalg as la
FACE_PATH = "/Users/chenjiarui/Desktop/orl_faces"
PERSON_NUM = 1
PERSON_FACE_NUM = 10
K = 10    # Number of principle components 主成分个数

raw_img = []
data_set = []
data_set_label = []

# 1, 对所有样本进行均值化, xi=xi−1m∑mi=1xi
# 2. 计算样本的协方差矩阵, XXT
# 3. 对协方差矩阵进行SVD分解
# 4. 取前ll个最大特征值的特征向量得到输出(目标矩阵)D

def read_data():
    for i in range(1, PERSON_NUM + 1):
        person_path = FACE_PATH + '/s' + str(i)
        for j in range(1, PERSON_FACE_NUM + 1):
            img = cv2.imread(person_path + '/' + str(j) + '.pgm')
            # 只取了每个人脸的第一张图像
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img_gray.shape
            img_col = img_gray.reshape(height * width)
            # 数据和标签存入data_set 和data_set_label
            data_set.append(img_col)
            data_set_label.append(i)
    return height, width

# Import data
# 只存储了最后的那组数据
# 但是所有的图片都是一样规格的
height, width = read_data()
print(height,width)
X = np.array(data_set)
Y = np.array(data_set_label)
n_sample, n_feature = X.shape

# 求解数据均值
def meanX(dataX):
    # axis=0表示依照列来求均值
    return np.mean(dataX,axis=0)

def pca(Matrix,k):
    # Matrix代表400*10304的X矩阵
    # 返回一个1*10304的矩阵
    average = meanX(Matrix)
    # print('avelen', len(average))
    m, n = np.shape(Matrix)
    # m=400 n =10304
    # print(m,n);
    # 纵向堆叠1*10304的均值矩阵
    avgs = np.tile(average, (m, 1))
    # avgs 400*10304
    data_adjust = Matrix - avgs
    # 上式对原400*10304的数据矩阵进行归一化处理，减去均值
    # 计算协方差矩阵
    # covX = np.cov(data_adjust.transpose())
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        U, sigma, VT = la.svd(data_adjust)
        U = U[:,0:k]
        UUT = np.matmul(U,U.T)

        temp = np.matmul(UUT, data_adjust)
        temp2 = temp + avgs
        return temp2


while 1:
    k = int(input('k:'))
    reconMat = pca(data_set, k)
    # reconMat 400 * 10304
    # k2 = int(input('pno:'))  # 0 --  399
    reconMat1 = np.array(reconMat)
    for i in range(10):
        reconMatTP = np.array(reconMat1[i])
        # print(reconMatTP)
        pic = reconMatTP.reshape((height,width))
        raw_img.append(pic)
        cv2.namedWindow("pic")
        cv2.imshow("pic",pic)
    raw_img = np.hstack(raw_img)
    cv2.namedWindow("Image")
    cv2.imshow('Image', raw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







