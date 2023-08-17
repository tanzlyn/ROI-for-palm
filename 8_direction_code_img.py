import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,  ImageOps
from PIL import Image
"""
算法参考2011xx学报
1.采用opencv库的Log算子进行边缘检测
2.将边缘检测线进行闭合构成回路（转换其他方法）
3.采用8-邻域编码确定特征点

不同数据集的二值化阈值设置：
polyu：65
tongji：17
CASIA：

注释的代码都是学习测试的过程中的一些无用的代码，不对最终结果产生影响。
"""


# ---------------------------采用opencv库的二值化和findContours进行边缘检测-----------------------------------
# # image = cv2.imread("F:/1_01.jpg")  # 65
# # image = cv2.imread("F:/1_00004_10.bmp")  # 200
# image = cv2.imread("F:/Tongji/contactlesspalmvein/Palmvein/session1/00015.tiff")
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # 二值化
# # """
# # 计算灰度直方图
# hist = cv2.calcHist([gray], [], None, [256], [0, 256])
# # 或者使用 np.histogram() 函数
# # hist, bins = np.histogram(gray.ravel(), 256, [, 256])
#
# # 绘制灰度直方图
# plt.plot(hist)
# plt.xlim([0, 100])
# plt.show()
# # """
# ret,binary = cv2.threshold(gray,65,255,cv2.THRESH_BINARY)  # 二值化阈值处理
# print("threshold value: %s"%ret)
# # 打印阈值，前面先进行了灰度处理0-255，我们使用该阈值进行处理，
# # 低于该阈值的图像部分全为黑，高于该阈值则为白色
# contours,  hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  # 找到边缘轮廓线
# cv2.drawContours(image,contours,-1,(0,0,255),2,8)   # 画出边缘轮廓线
#
#
# cv2.imshow('gray',binary)
# cv2.imshow('contours',image)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#  -----------------------------------------------------------------------------------------------------


#  它根据图像的信噪比来求检测边缘的最优滤波器。该算法首先对图像做高斯滤波，然后再求其拉普拉斯（ Laplacian ）二阶导数，
#  根据二阶导数的过零点来检测图像的边界，即通过检测滤波结果的零交叉（ Zero crossings ）来获得图像或物体的边缘
# 读取图像
image = cv2.imread("../datas/1_05.jpg")
# image = cv2.imread("F:/palm1.png")
# image = cv2.imread("F:/1_00004_10.bmp")
# image = cv2.imread("E:\Images\\1656402918474.png")
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)  # 图像灰度化

ret, binary = cv2.threshold(gray_img,70,250,cv2.THRESH_BINARY)  # 二值化阈值处理

# 以下图像四周像素为0用于后面的闭合手掌轮廓线
# 获取图片宽高及通道数
img=binary
height, width = img.shape
print(height)
print(width)
# print(channels)
# 将上下左右四周的像素值变成黑色像素值
img[0, :] = 0  # 上
img[1, :] = 0  # 上
img[height-1, :] = 0  # 下
img[height-2, :] = 0  # 下
img[:, 0] = 0  # 左
img[:, 1] = 0  # 左
img[:, width-1] = 0  # 右
img[:, width-2] = 0  # 右

# cv2.imshow('bihe', img)

# # 先通过高斯滤波降噪
gaussian = cv2.GaussianBlur(binary, (3, 3), 0)
# 再通过拉普拉斯算子做边缘检测
dst = cv2.Laplacian(gaussian, -1,cv2.BORDER_DEFAULT, ksize=5)
LOG = cv2.convertScaleAbs(dst)


# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

img_row = LOG.shape[0]
img_col = LOG.shape[1]

# 利用遍历操作对二维灰度图作取反操作
for i in range(img_row):
    for j in range(img_col):
        temp1 = LOG[i, j]
        LOG[i, j] = 255-temp1

cv2.imshow('image_gray_reverse', LOG)



# ----------------------------------将手掌轮廓线闭合(转换为四周为0）------------------------------
# import math
#
# # 判断两点之间的距离是否小于阈值
# def is_close(p1, p2, threshold):
#     dx = p1[0] - p2[0]
#     dy = p1[1] - p2[1]
#     dist = math.sqrt(dx*dx + dy*dy)
#     return dist < threshold
#
# # 将手掌轮廓线闭合
# def close_contour(points, threshold=5):
#     n = len(points)
#     # 判断轮廓线是否已经闭合
#     if is_close(points[0], points[-1], threshold):
#         return points
#     # 找到轮廓线上离起点最近的点
#     min_dist = float('inf')
#     min_idx = 0
#     for i in range(n):
#         dist = math.sqrt(points[i][0]*points[i][0] + points[i][1]*points[i][1])
#         if dist < min_dist:
#             min_dist = dist
#             min_idx = i
#     # 将起点和终点连接起来
#     closed_points = points[:min_idx+1] + [points[min_idx]] + points[min_idx+1:] + [points[0]]
#     return closed_points
# def bihe_counter(image):
#     height, width = image.shape
#
#     # 遍历图像像素
#
#     for x in range(width):
#         # 如果上面边缘当前像素为黑色
#         if image[0, x] == 0:
#             arr=[]
#             arr.append(x)
#             print("黑色像素的点：",arr)
#             for i in range(len(arr)):
#
#                 if arr[i] - arr[i - 1] > 10:
#                     # for j in range()
#                     #  image[0, i] == 0
#                     print("需要连接的点：",arr[i])
#                     for j in range(arr[i - 1], arr[i]):
#                         print(j + 1)
#
#             # neighbor_values = []
#     print("黑色像素的点：", arr)
# bihe_counter(LOG)
# print(bihe_counter(LOG))
# ------------------------------------------------------------------------------------------


# 获取手掌轮廓线的编码
def eight_neighborhood_encoding(image):
    # 定义8-邻域编码的符号表
    neighbors = [(-1, 0), (-1, 1),(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    symbols = ['0', '1', '2', '3', '4', '5', '6', '7']
    encoding = ''
    # encoding = []  #存入数组
    height, width = image.shape

    # 遍历图像像素
    for y in range(height):
        for x in range(width):
            # 如果当前像素为黑色
            if image[y, x] == 0:
                neighbor_values = []
                # 获取当前像素周围8个像素的灰度值
                for n in neighbors:
                    neighbor_x = x + n[1]
                    neighbor_y = y + n[0]
                    if neighbor_x >= 0 and neighbor_x < width and neighbor_y >= 0 and neighbor_y < height:
                        neighbor_values.append(image[neighbor_y, neighbor_x])
                # 将周围像素的灰度值转换为8-邻域编码的符号
                neighbor_symbols = [symbols[i] for i, v in enumerate(neighbor_values) if v == 0]
                # 将8-邻域编码符号拼接起来
                encoding += ''.join(neighbor_symbols)
                # encoding.append(neighbor_symbols) # 存入数组
            # else:
                # 如果当前像素为白色，用'.'表示
                # encoding += '.'

    return encoding

code=eight_neighborhood_encoding(LOG)
codes=[int(num) for num in code]
# int(num) for num in code:
    # codes.append(num)
print(codes)
print(len(code))

# 通过8-方向链码的编码确定手掌指谷特征点


















# 显示图形
titles = ['原始图像', '灰度图像','高斯降噪','LOG 算子']
images = [image, binary,gaussian, LOG]

for i in range(4):
    plt.subplot(1, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

