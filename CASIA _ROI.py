#!/user/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: UTF-8 -*-
# 对2012年论文中的方法(PROIE.py)进行更改以用于非接触式数据集CASIA
# 2023.05.13-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


class CASIA():

    def __init__(self):
        #####
        pass

    # PRIVATE METHODS
    """
    二值化图像、阈值区分手掌和背景
    """
    def _crop_img(self):
        self.crop_img = self.in_img_g[50:650, 60:500] # 画图输出时也使用截取
        # cv2.imshow("crop_img", self.crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _threshold(self):
        #####
        # 高斯滤波
        self.blur_img = cv2.GaussianBlur(self.crop_img, (7, 7), 0)
        # 二值化阈值处理
        o, self.thresh_img = cv2.threshold(
            self.blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.Binary_img_for_show = cv2.normalize(self.thresh_img, None, 0, 255, 32)
        # cv2.imshow("Gaussian_img", self.blur_img)
        cv2.imshow("Binary_img", self.Binary_img_for_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    """
      在二值图像中轮廓线并画出轮廓线
    """
    def _contours(self):
        #####
        # 寻找找轮廓线
        self.contours, o = cv2.findContours(  # uint8图像
            self.thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print("轮廓线：",  self.contours)
        # self.contours = self.contours[0]  # 取第一个轴上的所有元素
        # print("---", self.contours)

        # 遍历所有轮廓线，找到面积最大的轮廓线
        self.max_area = 0
        self.max_contour = None
        for contour in self.contours:
            area = cv2.contourArea(contour)
            if area > self.max_area:
                self.max_area = area
                self.max_contour = contour


        self.contour_img = self.in_img_c[50:650, 60:500].copy()
        # print("原图复制：", self.contour_img)
        # 画出最大轮廓线轮廓线，红色轮廓线
        self.contour_img = cv2.drawContours(
            self.contour_img, [self.max_contour], 0, (255, 0, 0), 1)
        self.selected = self.contour_img.copy()
        # print("轮廓线图：", self.contour_img)
        cv2.imshow("contour_img", self.contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def _landmarks(self):  # 手指谷点标记
        #####
        # 计算质心
        M = cv2.moments(self.thresh_img)  # 图像的矩
        x_c = M['m10'] // M['m00']
        y_c = M['m01'] // M['m00']
        self.center_point = {"x": x_c, "y": y_c}   # 质心坐标（x_c,y_c)
        print("质心坐标：", self.center_point)
        self.max_contour = self.max_contour.reshape(-1, 2)  # 把轮廓坐标数组变成两维
        # print("轮廓坐标:", self.contours)
        left_id = np.argmin(self.max_contour.sum(-1))  # 最小值下标。sum(1)和sum(-1)求数组每一行的和
        # print("left_id:", left_id)
        self.max_contour = np.concatenate(  # 对array进行拼接
            [self.max_contour[left_id:, :], self.max_contour[:left_id, :]])
        # print("left_id拼接：", self.contours)
        dist_c = np.sqrt(np.square(  # 欧式距离
            self.max_contour-[self.center_point["x"], self.center_point["y"]]).sum(-1))
        f = np.fft.rfft(dist_c)  # 快速离散傅里叶变换、rfft表示去除那些共轭对称的值，减小存储
        cutoff = 15
        f_new = np.concatenate([f[:cutoff], 0*f[cutoff:]])
        dist_c_1 = np.fft.irfft(f_new)  # 逆变换
        derivative = np.diff(dist_c_1)  # 差分
        sign_change = np.diff(np.sign(derivative))/2
        self.landmarks = {"x": [], "y": []}
        for landmark in self.max_contour[np.where(sign_change > 0)[0]]:
            self.landmarks["x"].append(landmark[0])
            self.landmarks["y"].append(landmark[1])

        print("关键点坐标：",self.landmarks)
        point_1 = (self.landmarks["x"][0], self.landmarks["y"][0])
        point_2 = (self.landmarks["x"][1], self.landmarks["y"][1])
        point_3 = (self.landmarks["x"][2], self.landmarks["y"][2])
        point_4 = (self.landmarks["x"][3], self.landmarks["y"][3])
        point_5 = (self.landmarks["x"][4], self.landmarks["y"][4])
        # point_6 = (self.landmarks["x"][5], self.landmarks["y"][5])
        # point = (self.landmarks["x"][6], self.landmarks["y"][6])
        radius = 1
        color = (0, 0, 255)
        thickness = 2
        self.landmarks_img1 = cv2.circle(self.contour_img, point_1, radius,color, thickness)
        self.landmarks_img2 = cv2.circle(self.landmarks_img1, point_2, radius,color, thickness)
        self.landmarks_img3 = cv2.circle(self.landmarks_img2, point_3, radius,color, thickness)
        self.landmarks_img4 = cv2.circle(self.landmarks_img3, point_4, radius,color, thickness)
        self.landmarks_img5 = cv2.circle(self.landmarks_img4, point_5, radius,color, thickness)
        # self.landmarks_img6 = cv2.circle(self.landmarks_img5, point_6, radius, color, thickness)
        # self.landmarks_img = cv2.circle(self.landmarks_img6, point, radius,color, thickness)
        # cv2.imshow("landmarks_img", self.landmarks_img6)
        cv2.waitKey()
        cv2.destroyAllWindows()





    def _landmarks_select(self):  # 标记选择
        #####
        y_rank = np.array(np.argsort(self.landmarks["y"]))  # 返回关键点纵y坐标数组值从小到大的索引值
        # 输出纵坐标y数组值对应的[:3]三个关键点坐标
        self.landmarks_selected = {"x": np.array(self.landmarks["x"])[
            y_rank][:3], "y": np.array(self.landmarks["y"])[y_rank][:3]}

        print("关键点y坐标-索引值-排序：", y_rank)
        print("选择标记：", self.landmarks_selected)

        x_rank = np.array(np.argsort(self.landmarks_selected["x"])) # 返回【被选择的】[:3]关键点横x坐标数组值从小到大的索引值
        # 输出横坐标x数组值对应的索引值0和2两个关键点坐标[0, 2]
        self.landmarks_selected = {
            "x": self.landmarks_selected["x"][x_rank][[0, 2]], "y": self.landmarks_selected["y"][x_rank][[0, 2]]}



        print("关键点x坐标-索引值-排序：",x_rank)
        print("选择标记：",self.landmarks_selected)

        select_point_1 = (self.landmarks_selected["x"][0], self.landmarks_selected["y"][0])
        select_point_2 = (self.landmarks_selected["x"][1], self.landmarks_selected["y"][1])
        # print(self.landmarks_selected["x"][1])
        radius = 1
        color = (0, 0, 255)
        thickness = 2
        self.landmarks_img1 = cv2.circle(self.selected, select_point_1, radius, color, thickness)
        self.landmarks_img2 = cv2.circle(self.landmarks_img1, select_point_2, radius, color, thickness)
        # cv2.imshow("landmarks_select_img", self.landmarks_img2)



    def _alignement(self):
        #####
        h, w = self.crop_img.shape  # 图像的长、宽
        # arctan2的值域是[ − π , π ]  、arctan的值域是[ − π/ 2 , π /2 ]
        # 计算反正切，返回值是弧度，*180/np.pi得到角度值。
        theta = np.arctan2((self.landmarks_selected["y"][1] - self.landmarks_selected["y"][0]), (
            self.landmarks_selected["x"][1] - self.landmarks_selected["x"][0]))*180/np.pi
        R = cv2.getRotationMatrix2D(  # 三个参数：旋转中心点、旋转角度、缩放因子
            (int(self.landmarks_selected["x"][1]), int(self.landmarks_selected["y"][1])), theta, 1)
        self.align_img = cv2.warpAffine(self.crop_img, R, (w, h))  # 三个参数：原图、仿射变换矩阵、输出图像尺寸。

        point_1 = [self.landmarks_selected["x"]
                   [0], self.landmarks_selected["y"][0]]
        point_2 = [self.landmarks_selected["x"]
                   [1], self.landmarks_selected["y"][1]]

        point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(int)
        point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(int)

        self.landmarks_selected_align = {
            "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}

        align_point_1 = (self.landmarks_selected_align["x"][0], self.landmarks_selected_align["y"][0])
        align_point_2 = (self.landmarks_selected_align["x"][1], self.landmarks_selected_align["y"][1])

        # print("对齐后关键点：",align_point_1)
        # print("对齐后关键点：", align_point_2)
        radius = 1
        color = (255, 0, 0)
        thickness = 2
        # self.align_img1 = cv2.circle(self.align_img, align_point_1, radius, color, thickness)
        # self.align_img2 = cv2.circle(self.align_img1, align_point_2, radius, color, thickness)
        # cv2.imshow("align_img",self.align_img2)


    def _roi_extract(self):
        #####
        point_1 = np.array([self.landmarks_selected_align["x"]
                            [0], self.landmarks_selected_align["y"][0]])
        point_2 = np.array([self.landmarks_selected_align["x"]
                            [1], self.landmarks_selected_align["y"][1]])
        # 截取ROI区域的4个坐标点
        # self.ux = point_1[0]
        # self.uy = point_1[1] + (point_2-point_1)[0]//3
        # self.lx = point_2[0]
        # self.ly = point_2[1] + 4*(point_2-point_1)[0]//3
        Origin_X = (point_1[0] + point_2[0]) / 2.0
        Origin_Y = (point_1[1] + point_2[1]) / 2.0
        Uleft = (int(Origin_X - 128 / 2),int(Origin_Y + 50))

        # print(point_1[0],point_1[1],point_2[0],point_2[1])
        print("坐标原点：：",Origin_X,Origin_Y)
        print("ROI的左上点：",Uleft)
        # print("ROI坐标点y：", self.uy, self.ly)
        print("---------------------------------------------")
        # 画出ROI区域
        self.roi_zone_img = cv2.cvtColor(self.align_img, cv2.COLOR_GRAY2BGR)
        # self.roi_img = self.roi_zone_img[Uleft[1]:Uleft[1] + 128, Uleft[0]:Uleft[0] + 128]

        # cv2.rectangle(self.roi_zone_img, (point_1[0], point_1[1]+30),
        #               (point_2[0], point_2[1]+128+30+30), (0, 255, 0), 2)
        cv2.rectangle(self.roi_zone_img, (Uleft[0], Uleft[1]),
                      (Uleft[0]+128 , Uleft[1] + 128 ), (0, 255, 0), 2)
        #
        self.roi_img = self.align_img[Uleft[1]:Uleft[1] + 128, Uleft[0]:Uleft[0] + 128]


        # cv2.imshow("rectangle",self.roi_zone_img)
        # cv2.imshow("ROI",self.roi_img)

        return self.roi_img

    # PUBLIC METHODS

    def extract_roi(self,path_in_img, rotate=True):  # 图像具体信息 和 图像路径
        #####
        # self.in_img_c = img  # 整个数据集需要加这行代码
        self.in_img = cv2.imread(path_in_img)
        self.in_img_g = self.in_img
        # self.in_img_c = path_in_img
        if(rotate):
            self.in_img_c = cv2.rotate(self.in_img_g, cv2.ROTATE_90_CLOCKWISE)


        if len(self.in_img_c.shape) == 3:
            self.in_img_g = cv2.cvtColor(self.in_img_c, cv2.COLOR_BGR2GRAY)
        else:
            self.in_img_g = self.in_img_c

        self._crop_img()
        self._threshold()
        self._contours()
        self._landmarks()
        self._landmarks_select()
        self._alignement()
        self._roi_extract()
        # print("ROI最后输出：", self.roi_img)
        return self.roi_img

    def save(self, path_out_img):
        #####
        cv2.imwrite(path_out_img, self.roi_img)



    def show_result(self):
        #####
        # 'x'表示叉号，'o'表示圆圈，'+'表示十字形，'s'表示正方形
        plt.figure()

        plt.subplot(341)
        plt.imshow(self.in_img, cmap="gray")
        plt.title("original")

        plt.subplot(342)
        plt.imshow(self.in_img_c, cmap="gray")
        plt.title("rotate")


        plt.subplot(343)
        plt.imshow(self.thresh_img, cmap="gray")
        plt.title("threshold")

        plt.subplot(344)
        plt.imshow(self.contour_img, cmap="gray")
        plt.plot(self.center_point["x"], self.center_point["y"], 'bx')  # 蓝色标记。
        plt.title("contours")

        plt.subplot(345)
        plt.imshow(self.contour_img, cmap="gray")  # self.contour_img
        for idx in range(len(self.landmarks["x"])):
            plt.plot(self.landmarks["x"][idx], self.landmarks["y"][idx], 'rx',markersize=4)  # 红色标记。
        plt.title("landmarks")

        plt.subplot(346)
        plt.imshow(self.contour_img, cmap="gray")
        plt.plot(self.landmarks_selected["x"][0],
                 self.landmarks_selected["y"][0], 'rx')
        plt.plot(self.landmarks_selected["x"][1],
                 self.landmarks_selected["y"][1], 'rx')
        plt.title("selected")

        plt.subplot(347)
        plt.imshow(self.align_img, cmap="gray")
        plt.plot(self.landmarks_selected_align["x"][0],
                 self.landmarks_selected_align["y"][0], 'rx')
        plt.plot(self.landmarks_selected_align["x"][1],
                 self.landmarks_selected_align["y"][1], 'rx')
        plt.title("alignement")

        plt.subplot(348)
        plt.imshow(self.roi_zone_img, cmap="gray")
        plt.title("roi zone")

        plt.subplot(349)
        plt.imshow(self.roi_img, cmap="gray")
        plt.title("extraction")

        plt.show()






# 测试数据集的单张图像是否正确
if __name__ == '__main__':
    ####
    # path_in_img = "resources/2.png"  # CASIA
    path_in_img = "E:\datasets\CASIA_palmvein/014_l_850_05.jpg"  # 测试数据集是否正确
    # path_in_img = "resources/palmprint.jpg"
    # path_in_img = "E:/datasets/Tongji/Palmvein/session1/00054.tiff"
    # path_in_img = "F:\palm.png"
    # path_in_img = "F:\My Snapshot.JPG"
    # path_in_img = "resources/tongji.tiff"

    proie = CASIA()

    proie.extract_roi(path_in_img, rotate=True)
    proie.show_result()
    proie.save("resources/CASIA_roi.jpg")
#



# # 对整个数据集进行批量操作ROI
# def mkdir(path):
#     folder = os.path.exists(path)
#     if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
#         print("---  OK  ---")
#     else:
#         print("---  There is this folder!  ---")
#
# if __name__ == '__main__':
#     path = 'E:/datasets/CASIA_palmvein/'  # 原数据集文件夹
#
#     casia = CASIA()
#
#     # proie.extract_roi(path_in_img, rotate=True)
#     # proie.show_result()
#     # proie.save("resources/palmvein_roi.jpg")
#     # save(self, path_out_img)
#     number = 0
#     filelist = os.listdir(path)  # 获取指定的文件夹包含的文件或文件夹的名字的列表
#     print(filelist)
#     total_num = len(filelist)  # 获取文件夹内所有文件个数
#     print(total_num)
#
#     c = 0  # 想看总共 重命名了多少张图片
#
#     for files in filelist:
#         figsPath = path   #  + files  # + '/visual'  # 原来的视觉路径信息
#         figures = os.listdir(figsPath)  # 原先视觉文件夹中的所有图片
#         print(figures)
#
#         # total_figure = len(figures)
#         # print(total_figure)
#
#         for fig in figures:
#             # 想要去新建文件夹: 关于数据增强是翻转的，让其添加 rotated
#             # 对原数据的文件夹切割，主要就是为了便于获取标签和文件夹名字
#             # yearMonthDate, Hour, minute, label = files.split("-")
#             # yearNewFileName = yearMonthDate + "rotated90"  # 翻转90
#             # 新的保存数据增强的视觉 文件夹名称
#             # filesNew = yearNewFileName + "-" + Hour + "-" + minute + "-" + label
#             # 新的保存数据增强的视觉 文件夹具体位置：'jiaodai_2_test/20211201rotated90-11-26-2/visual'
#             # figsPathNew = path + filesNew + '/visual'
#             # newDate = files + "roi"
#             savePath = 'E:\datasets\CASIA_ROI/'  # 存放ROI图像的文件夹
#             figsPathNew = savePath    # + files
#             mkdir(figsPathNew)
#             print(fig)
#             fig_path = figsPath + fig  # 单个图片的完整表示，包括路径，如：'20211124-10-23-1/visual/0.jpg'
#             print("hhhh",fig_path)
#             img = cv2.imread(fig_path)  # img表示一张图片的具体信息。
#
#
#
#             # polyu.extract_roi(img,fig_path)
#             # PolyU._roi(fig_path)
#             # print("主函数调用ROI：", polyu.extract_roi(img,fig_path))
#             # rotated_90 = rotate(img, 90)  # rotated_90表示一张图片进行数据增强后的信息
#             # cv2.imwrite(os.path.join(figsPathNew , fig[0:-4] + '.jpg'),  rotated_90) # 保存图片到指定文件夹中
#             # cv2.imwrite(os.path.join(figsPathNew, fig[0:-4] + '.tiff'), proie.extract_roi(img, rotate=True))
#             cv2.imwrite(os.path.join(figsPathNew, fig), casia.extract_roi(img,fig_path))
#             # newpath=os.path.join(figsPathNew, fig[0:-4] + '.jpg')
#             # proie.save("figsPathNew")
#             # proie.save(self, path_out_img)
#
#             print("我是不是快成功了呀？\n")
#             number = number + 1
#             if number == 2400:
#                 break
#
#     print("ye ！！！")