#!/user/bin/env python3
# -*- coding: utf-8 -*-
# revised by tan 2023.5.12
# 用于PolyU-Palmvein掌脉数据集批量处理和保存
import math
import cv2
import numpy as np
import os
import os.path
import copy

A=10 # 为了避免干扰和准确定位，图像上下各去掉10个像素，可全局设置
PI = 3.14159265
debug = 1

class PolyU():

    def __init__(self):
        #####
        pass

    def _crop_img(self):
        # 2.1 crop image
        self.crop_img = self.in_img[0+A:288-A, 0:100]

    def _low_pass_gaussion(self):
        # 2.2 low-pass Gaussion Filter
        # before_Gaussion = np.zeros((284, 160), dtype=int)
        # roi = self.crop_img[7:self.crop_img.shape[0] - 7, 7:self.crop_img.shape[1] - 7]
        # before_Gaussion[7:self.crop_img.shape[0] - 7, 7:self.crop_img.shape[1] - 7] = roi.copy()
        self.Gaussion_img = cv2.GaussianBlur(np.uint8(self.crop_img), (23, 23), 6, 6) #

    def _threasholding(self):
        # 2.3 thresholding
        ret, self.Binary_img = cv2.threshold(self.Gaussion_img, 65, 1, 0) # 阈值为
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 28))  # 28。
        # 进行闭运算操作
        self.Binary_img = cv2.morphologyEx(self.Binary_img, cv2.MORPH_CLOSE, kernel)
        # 将白色像素变成黑色像素
        self.Binary_img[100:150, 0:50] = 0
        # if debug:
            # Binary_img_for_show = cv2.normalize(self.Binary_img, None, 0, 255, 32)
            # cv2.imshow("Binary_img", Binary_img_for_show)

    def _find_Expoint(self):

        # 2.4 Find Reference Points
        # (a)Find External Reference Point
        self.Out_top = (0, 0)
        self.Out_bottom = (0, 0)
        for row in range(self.Binary_img.shape[0]):
            is_get = 0
            for col in range(self.Binary_img.shape[1]):
                if self.Binary_img[row][col] == 1:
                    self.Out_top = (col, row)
                    outopcol = col
                    outoprow = row
                    is_get = 1
                    break
            if is_get:
                break
        for row in range(self.Binary_img.shape[0] - 1, -1, -1):
            is_get = 0
            for col in range(self.Binary_img.shape[1]):
                if self.Binary_img[row][col] == 1:
                    self.Out_bottom = (col, row)
                    outbottomcol = col
                    outbottomrow = row
                    is_get = 1
                    break
            if is_get:
                break
        if debug:
            print("Out_top(x,y):{}".format(self.Out_top))
            print("Out_bottom(x,y):{}\n".format(self.Out_bottom))

    def _find_Inpoint(self):

        # (b)Find Internal Reference Point  寻找内部参考点
        self.In_top = (0, 0)
        self.In_bottom = (0, 0)
        gap_x = 0
        for col in range(self.Binary_img.shape[1]):
            gap_width = 0
            for row in range(self.Binary_img.shape[0]):
                if self.Binary_img[row][col] == 0:
                    gap_width += 1
            if gap_width < 200:
                gap_x = col
                break
        self.In_top = (gap_x, 0)
        self.In_bottom = (gap_x, 0)
        center_y = self.Binary_img.shape[0] // 2
        for row in range(center_y, -1, -1):
            if self.Binary_img[row][gap_x] == 1:
                self.In_top = (gap_x, row)
                break
        for row in range(center_y, self.Binary_img.shape[0]):
            if self.Binary_img[row][gap_x] == 1:
                self.In_bottom = (gap_x, row)
                break
        if debug:
            print('In_top(x,y):{}'.format(self.In_top))
            print('In_bottom(x,y):{}\n'.format(self.In_bottom))

    def _find_countours(self):

        # 2.5.1 Find Countours 寻找轮廓线
        self.Out_top_j = self.Out_bottom_j = self.In_top_j = self.In_bottom_j = 0
        reference_point_num = 0
        self.contours, hierarchy = cv2.findContours(self.Binary_img, 0, 1)
        self.Contours = np.zeros(self.Binary_img.shape, int)
        for j in range(len(self.contours[0])):
            if self.contours[0][j][0][0] == self.Out_top[0] and self.contours[0][j][0][1] == self.Out_top[1]:
                self.Out_top_j = j
                reference_point_num += 1
                # print("1:",reference_point_num)
            if self.contours[0][j][0][0] == self.Out_bottom[0] and self.contours[0][j][0][1] == self.Out_bottom[1]:
                self.Out_bottom_j = j
                reference_point_num += 1
                # print("2:", reference_point_num)
            if self.contours[0][j][0][0] == self.In_top[0] and self.contours[0][j][0][1] == self.In_top[1]:
                self.In_top_j = j
                reference_point_num += 1
                # print("3:", reference_point_num)
            if self.contours[0][j][0][0] == self.In_bottom[0] and self.contours[0][j][0][1] == self.In_bottom[1]:
                self.In_bottom_j = j
                reference_point_num += 1
                # print("4:", reference_point_num)
        if reference_point_num != 4:
            print('not four:',reference_point_num)
            # exit(0)
        for j in range(self.Out_top_j, self.In_top_j + 1):
            P = (self.contours[0][j][0][0], self.contours[0][j][0][1])
            self.Contours[P[1]][P[0]] = 255
        for j in range(self.In_bottom_j, self.Out_bottom_j + 1):
            P = (self.contours[0][j][0][0], self.contours[0][j][0][1])
            self.Contours[P[1]][P[0]] = 255

    def _key_point(self):
        # 2.5.2 Key Point Positioning  关键点定位
        self.Top_x = self.Bottom_x = 0.0
        Top_y_vector = []
        Bottom_y_vector = []
        for j in range(self.Out_top_j, self.In_top_j + 1):
            if self.contours[0][j][0][0] > self.Top_x:
                self.Top_x = self.contours[0][j][0][0]
        for j in range(self.In_bottom_j, self.Out_bottom_j + 1):
            if self.contours[0][j][0][0] > self.Bottom_x:
                self.Bottom_x = self.contours[0][j][0][0]
        for j in range(self.Out_top_j, self.In_top_j + 1):
            if self.contours[0][j][0][0] == self.Top_x:
                Top_y_vector.append(self.contours[0][j][0][1])
        for j in range(self.In_bottom_j, self.Out_bottom_j + 1):
            if self.contours[0][j][0][0] == self.Bottom_x:
                Bottom_y_vector.append(self.contours[0][j][0][1])

        top_sum = sum(Top_y_vector)
        bottom_sum = sum(Bottom_y_vector)
        self.Top_y = top_sum / float(len(Top_y_vector))
        self.Bottom_y = bottom_sum / float(len(Bottom_y_vector))

        print('Top:({},{})'.format(self.Top_x, self.Top_y))
        print('Bottom:({},{})\n'.format(self.Bottom_x, self.Bottom_y))

    def _roi(self,path_img):

        src = cv2.imread(path_img, 0)
        if src is None:
            print('图片没读到')
            exit(0)
        # if debug:
            # cv2.imshow("src", src)

        # 2.6 Build a Coordinate System on the Oridinal Image
        Top = (self.Top_x, self.Top_y)
        Bottom = (self.Bottom_x, self.Bottom_y)
        Origin_X = (Top[0] + Bottom[0]) / 2.0
        Origin_Y = (Top[1] + Bottom[1]) / 2.0
        Origin = (Origin_X, Origin_Y)
        Slope_y_axis = (self.Top_y - self.Bottom_y) / (self.Top_x - self.Bottom_x)
        Slope_x_axis = -1 / Slope_y_axis
        angle = -1 * math.atan(1 / Slope_y_axis) * (180 / PI)
        angle1 = angle if angle<-10 else -10
        if angle < -5:
            angle = -5
        if angle > 5:
            angle = 5
        rotated_sz = (src.shape[1], src.shape[0])
        center = (Origin_X, Origin_Y)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        Rotated_img = cv2.warpAffine(src, rot_mat, rotated_sz, 1, 0)
        self.roi_img = Rotated_img.copy()
        Uleft = (int(Origin_X + 50), int(Origin_Y - 128 / 2))
        self.roi_img = self.roi_img[Uleft[1]:Uleft[1] + 128, Uleft[0]:Uleft[0] + 128]
        # if debug:
        #     cv2.imshow("dst", dst)
        #
        # if debug:
        #     cv2.waitKey(0)

        return self.roi_img

    def extract_roi(self,img, path_img):  # 图像具体信息 和 图像路径

        self.in_img_c = img
        if len(self.in_img_c.shape) == 3:
            self.in_img = cv2.cvtColor(self.in_img_c, cv2.COLOR_BGR2GRAY)
        else:
            self.in_img = self.in_img_c

        self._crop_img()
        self._low_pass_gaussion()
        self._threasholding()
        self._find_Expoint()
        self._find_Inpoint()
        self._find_countours()
        self._key_point()
        self._roi(path_img)
        # print("ROI最后输出：", self.roi_img)
        return self.roi_img

    def save(self, path_out_img):
        #####
        cv2.imwrite(path_out_img, self.roi_img)


# 对整个数据集进行批量操作ROI
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

if __name__ == '__main__':
    path = 'E:/datasets/PolyU/'  # 原数据集文件夹

    polyu = PolyU()

    # proie.extract_roi(path_in_img, rotate=True)
    # proie.show_result()
    # proie.save("resources/palmvein_roi.jpg")
    # save(self, path_out_img)

    filelist = os.listdir(path)  # 获取指定的文件夹包含的文件或文件夹的名字的列表
    print(filelist)
    total_num = len(filelist)  # 获取文件夹内所有文件个数
    print(total_num)

    c = 0  # 想看总共 重命名了多少张图片

    for files in filelist:
        figsPath = path + files  # + '/visual'  # 原来的视觉路径信息
        figures = os.listdir(figsPath)  # 原先视觉文件夹中的所有图片
        print(figures)

        total_figure = len(figures)
        print(total_figure)

        for fig in figures:
            # 想要去新建文件夹: 关于数据增强是翻转的，让其添加 rotated
            # 对原数据的文件夹切割，主要就是为了便于获取标签和文件夹名字
            # yearMonthDate, Hour, minute, label = files.split("-")
            # yearNewFileName = yearMonthDate + "rotated90"  # 翻转90
            # 新的保存数据增强的视觉 文件夹名称
            # filesNew = yearNewFileName + "-" + Hour + "-" + minute + "-" + label
            # 新的保存数据增强的视觉 文件夹具体位置：'jiaodai_2_test/20211201rotated90-11-26-2/visual'
            # figsPathNew = path + filesNew + '/visual'
            # newDate = files + "roi"
            savePath = 'E:/datasets/PolyU_ROI/'  # 存放ROI图像的文件夹
            figsPathNew = savePath + files
            mkdir(figsPathNew)

            fig_path = figsPath + '/' + fig  # 单个图片的完整表示，包括路径，如：'20211124-10-23-1/visual/0.jpg'
            print("hhhh",fig_path)
            img = cv2.imread(fig_path)  # img表示一张图片的具体信息。



            # polyu.extract_roi(img,fig_path)
            # PolyU._roi(fig_path)
            # print("主函数调用ROI：", polyu.extract_roi(img,fig_path))
            # rotated_90 = rotate(img, 90)  # rotated_90表示一张图片进行数据增强后的信息
            # cv2.imwrite(os.path.join(figsPathNew , fig[0:-4] + '.jpg'),  rotated_90) # 保存图片到指定文件夹中
            # cv2.imwrite(os.path.join(figsPathNew, fig[0:-4] + '.tiff'), proie.extract_roi(img, rotate=True))
            cv2.imwrite(os.path.join(figsPathNew, fig), polyu.extract_roi(img,fig_path))
            # newpath=os.path.join(figsPathNew, fig[0:-4] + '.jpg')
            # proie.save("figsPathNew")
            # proie.save(self, path_out_img)

            print("我是不是快成功了呀？\n")
    print("ye ！！！")