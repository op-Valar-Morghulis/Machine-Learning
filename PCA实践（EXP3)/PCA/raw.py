import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
path="/Users/chenjiarui/Desktop/orl_faces"
FACE_PATH = path  # \\ can be ambiguous
PERSON_NUM = 40
PERSON_FACE_NUM = 10
K = 10  # Number of principle components

raw_img = []
data_set = []
data_set_label = []

def read_data():
    for i in range(1, PERSON_NUM + 1):
        person_path = FACE_PATH + '/s' + str(i)
        for j in range(1, PERSON_FACE_NUM + 1):
            img = cv2.imread(person_path + '/' + str(j) + '.pgm')
            if j == 1:
                raw_img.append(img)
                # raw_img 拼接每个数据组的第一张图片
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 遍历所有图片 将其转变为2值灰度图
            height, width = img_gray.shape
            img_col = img_gray.reshape(height * width)
            data_set.append(img_col)
            data_set_label.append(i)
    return height, width

height, width = read_data()

# print('dataset',data_set)
# print('datasetlabel',data_set_label)

X = np.array(data_set)
Y = np.array(data_set_label)
n_sample, n_feature = X.shape
# print(n_sample,n_feature)


# Print some samples

raw_img = np.hstack(raw_img)
cv2.namedWindow("Image")
cv2.imshow('Image', raw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------
# data_set 400*10304 number
# data_set_label 400*1  include 1 1 1 1 1 1 ...-...  40 40 40 40
# ------------------------

#print average

# average_face = np.mean(X, axis=0)
# print('av_face',len(average_face))
# fig = plt.figure()
# plt.imshow(average_face.reshape((height, width)), cmap=plt.cm.gray)
# plt.title("Average Face", size=12)
# plt.xticks(())
# plt.yticks(())
# plt.show()
# print('averge face loaded!')
#
#
# def meanX(dataX):
#     return np.mean(dataX,axis=0)#axis=0表示依照列来求均值。假设输入list,则axis=1
#
#
#
# data_set1=np.array(data_set)
# data_set=data_set1.tolist()
#
#
# def pca(XMat, k):
#     average = meanX(XMat)
#     print('avelen',len(average))
#
#
#     m, n = np.shape(XMat)
#     print(m,n)
#     avgs = np.tile(average, (m, 1))
#     print('avgslen',len(avgs))
#     data_adjust = XMat - avgs
#     covX = np.cov(data_adjust.T)   #计算协方差矩阵
#     print('covlen',len(covX))
#     if k > n:
#         print ("k must lower than feature number")
#         return
#     else:
#         svd1 = TruncatedSVD(n_components=k)
#         svd1.fit(covX)
#         Udp=svd1.components_
#         tempm1=np.matmul(Udp.T,Udp)
#         tempm2=np.matmul(tempm1,data_adjust.T)
#         tempm3=tempm2+avgs.T
#         tempm4=tempm3.T
#         return tempm4
#
#
# while 1:
#     k=int(input('k:'))
#     reconMat = pca(data_set,k)
#     #reconMat 400 * 10304
#     k2=int(input('pno:'))  #0 --  399
#     reconMat1=np.array(reconMat)
#     reconMatTP=np.array(reconMat1[k2])
#     fig = plt.figure()
#     plt.imshow(reconMatTP.reshape((height, width)), cmap=plt.cm.gray)
#     plt.title("AfterPca", size=12)
#     plt.xticks(())
#     plt.yticks(())
#     plt.show()