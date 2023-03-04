import time

import cv2

####################################################
#    top, left, bottom, right
#    72 430 1070 1217
# 具体含义如下：
# top：边界框的顶部位置坐标，通常是从图像的顶部开始计算，单位为像素。
# left：边界框的左侧位置坐标，通常是从图像的左侧开始计算，单位为像素。
# bottom：边界框的底部位置坐标，通常是从图像的顶部开始计算，单位为像素。
# right：边界框的右侧位置坐标，通常是从图像的左侧开始计算，单位为像素。
# 左上角坐标：(left, top) = (430, 72)
# 右下角坐标：(right, bottom) = (1217, 1070)
# 左下角坐标：(left, bottom) = (430, 1070)
# 右上角坐标：(right, top) = (1217, 72)
###################################################
import numpy as np


def ORB_Feature(img1, n):
    # 初始化ORB
    orb = cv2.ORB_create(n)

    # 寻找关键点
    kp1 = orb.detect(img1)
    # print(kp1)

###########
    # 我在这里加入了手动特征点
    # points2f = cv2.KeyPoint_convert(kp1)
    # # print(points2f)
    # temp = [[1.00000, 2.00000]]
    # temp = np.array(temp)
    # points2f = np.append(points2f, temp, axis=0)
    # kp1 = cv2.KeyPoint_convert(points2f)

    ##################################################
    # 作弊  增加kp ：kp=cv2.KeyPoint(x=1,y=2,size=3)
    #              print(kp)
    ###################################################

    #              print(kp)
    # kp2=kp1+kp

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    return outimg1

    # 显示关键点
    # import numpy as np
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv.imshow("Key Points", outimg3)
    # cv.waitKey(0)

    # 初始化 BFMatcher
    # bf = cv.BFMatcher(cv.NORM_HAMMING)
    #
    # # 对描述子进行匹配
    # matches = bf.match(des1, des2)
    #
    # # 计算最大距离和最小距离
    # min_distance = matches[0].distance
    # max_distance = matches[0].distance
    # for x in matches:
    #     if x.distance < min_distance:
    #         min_distance = x.distance
    #     if x.distance > max_distance:
    #         max_distance = x.distance
    #
    # # 筛选匹配点
    # '''
    #     当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
    #     但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    # '''
    # good_match = []
    # for x in matches:
    #     if x.distance <= max(2 * min_distance, 30):
    #         good_match.append(x)
    #
    # # 绘制匹配结果
    # draw_match(img1, img2, kp1, kp2, good_match)
    # print(good_match)
bigimg=cv2.imread('img/bigr.png')
start_orb=time.time()
outimg=ORB_Feature(bigimg, 5000)
end_orb=time.time()
print('big-orb-cost', end_orb-start_orb)

bigimg=cv2.imread('img/bigr.png')
start_orb=time.time()
outimg=ORB_Feature(bigimg, 5000)
end_orb=time.time()
print('big-orb-cost', end_orb-start_orb)

smallimg=cv2.imread('img/small.png')
start_orb=time.time()
outimg=ORB_Feature(smallimg, 5000)
end_orb=time.time()
print('small-orb-cost', end_orb-start_orb)
cv2.imshow('pic', outimg)
cv2.waitKey(0)