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
import time
from enum import Enum
import cv2
import numpy as np
from cv2.xfeatures2d import matchGMS
from matplotlib import pyplot as plt
import pandas as pd
###########################################################计算三维坐标

K_l = np.array([[878.35, 0, 572.27],
                [0, 878.14, 383.91],
                [0, 0, 1]])
K_r = np.array([[880.71, 0, 577.32],
                [0, 897.92, 382.93],
                [0, 0, 1]])
R = np.array([[1, 0.0009, -0.0054],
              [-0.0009, 1, -0.0031],
              [0.0054, 0.0031, 1.0000]])
T = np.array([[-119.7164],
              [-0.0681],
              [0.6998]])
leftIntrinsic = np.array([[878.35, 0, 572.27],
                          [0, 878.14, 383.91],
                          [0, 0, 1]])
leftRotation = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
leftTranslation = np.array([[0],
                            [0],
                            [0]])
rightIntrinsic = np.array([[880.71, 0, 577.32],
                           [0, 897.92, 382.93],
                           [0, 0, 1]])
rightRotation = np.array([[1, 0.0009, -0.0054],
                          [-0.0009, 1, -0.0031],
                          [0.0054, 0.0031, 1.0000]])
rightTranslation = np.array([[-119.7164],
                             [-0.0681],
                             [0.6998]])
# 函数参数为左右相片同名点的像素坐标，获取方式后面介绍
# lx，ly为左相机某点像素坐标，rx，ry为右相机对应点像素坐标
def uvToXYZ(lx, ly, rx, ry):
    mLeft = np.hstack([leftRotation, leftTranslation])
    mLeftM = np.dot(leftIntrinsic, mLeft)
    mRight = np.hstack([rightRotation, rightTranslation])
    mRightM = np.dot(rightIntrinsic, mRight)
    A = np.zeros(shape=(4, 3))
    for i in range(0, 3):
        A[0][i] = lx * mLeftM[2, i] - mLeftM[0][i]
    for i in range(0, 3):
        A[1][i] = ly * mLeftM[2][i] - mLeftM[1][i]
    for i in range(0, 3):
        A[2][i] = rx * mRightM[2][i] - mRightM[0][i]
    for i in range(0, 3):
        A[3][i] = ry * mRightM[2][i] - mRightM[1][i]
    B = np.zeros(shape=(4, 1))
    for i in range(0, 2):
        B[i][0] = mLeftM[i][3] - lx * mLeftM[2][3]
    for i in range(2, 4):
        B[i][0] = mRightM[i - 2][3] - rx * mRightM[2][3]
    XYZ = np.zeros(shape=(3, 1))
    # 根据大佬的方法，采用最小二乘法求其空间坐标
    cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)
    # print(XYZ)
    XYZ = XYZ.reshape(1, 3)
    xyz = XYZ[0]
    return xyz
def calculate_3d_point(u_l, v_l, u_r, v_r):
    # Compute disparity
    K_l = np.array([[878.35, 0, 572.27],
                    [0, 878.14, 383.91],
                    [0, 0, 1]])
    K_r = np.array([[880.71, 0, 577.32],
                    [0, 897.92, 382.93],
                    [0, 0, 1]])
    R = np.array([[1, 0.0009, -0.0054],
                  [-0.0009, 1, -0.0031],
                  [0.0054, 0.0031, 1.0000]])
    T = np.array([[-119.7164],
                  [-0.0681],
                  [0.6998]])

    d = u_l - u_r
    # d = u_r - u_l

    # Compute camera coordinates
    p_l = np.array([u_l, v_l, 1]).reshape(3, 1)
    p_r = np.array([u_r, v_r, 1]).reshape(3, 1)
    p_l_cam = np.linalg.inv(K_l).dot(p_l) * d
    p_r_cam = np.linalg.inv(K_r).dot(p_r) * d

    # Compute world coordinates
    p_l_world = R.dot(p_l_cam) + T
    p_r_world = R.dot(p_r_cam) + T
    X = (p_l_world[0, 0] + p_r_world[0, 0]) / 2.0
    Y = (p_l_world[1, 0] + p_r_world[1, 0]) / 2.0
    Z = (p_l_world[2, 0] + p_r_world[2, 0]) / 2.0

    return [X, Y, Z]

def slam_get_3d(good_match, K_l, K_r, R, T):
    # 提取配准点
    points1 = []
    points2 = []
    for i in good_match:
        points1.append(list(kp1[i.queryIdx].pt));
        points2.append(list(kp2[i.trainIdx].pt));
    points1 = np.array(points1)
    points2 = np.array(points2)

    projMatr1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 第一个相机参数
    projMatr2 = np.concatenate((R, T), axis=1)  # 第二个相机参数
    projMatr1 = np.matmul(K_l, projMatr1)  # 相机内参 相机外参
    projMatr2 = np.matmul(K_r, projMatr2)  #
    points4D = cv2.triangulatePoints(projMatr1, projMatr2, points1.T, points2.T)
    points4D /= points4D[3]  # 归一化
    points4D = points4D.T[:, 0:3]  # 取坐标点
    print(points4D[0:5])
    return points4D
#######################################################################################

def ORB_Feature(img1, img2, n):
    # 初始化ORB
    orb = cv2.ORB_create(n)

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)
    # cv2.imshow('kp',outimg1)
    # cv2.waitKey(0)

    ########## 显示关键点,把两张图放在一起
    ##########
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv.imshow("Key Points", outimg3)
    # cv.waitKey(0)


def distance_select(matches):
    ################################################################################
    #####这里是BF匹配

    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)

    return good_match
######输入坐标点矩阵->矩阵
def use_meanAndstd_2select(X):
    # 计算z轴坐标值的均值和标准差，并计算阈值
    z_mean = np.mean(X[:, 2])
    z_std = np.std(X[:, 2])
    z_threshold = z_mean + 3 * z_std

    # 根据阈值删除偏离较大的数据点
    X = X[X[:, 2] <= 4000]
    X_clean = X[abs(X[:, 2] - z_mean) <= z_threshold, :]
    X_clean = X[X[:, 2] > 0]
    # 打印删除误差后的三维点集
    print(X_clean)
    return X_clean

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (135, 206, 235))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (135, 206, 235))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1] * .5 + left[1] - src1.shape[0] * .5) * 256. / (
                        src1.shape[0] * .5 + src1.shape[1] * .5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

#############################################
########把选好的点计算成三维坐标#################
def get_3dpoints(matches):
    my3D_points = []
    for i in range(len(matches)):
        tempmatch = matches[i]
        keypoint_left = kp1[tempmatch.queryIdx]
        keypoint_right = kp2[tempmatch.queryIdx]
        # print(keypoint_right)
        # keypoint_right2=(keypoint_right)
        tupler = ()
        tupler = tupler + (keypoint_right,)
        rpoints2f = cv2.KeyPoint_convert(tupler)
        # print(points2f)
        tuplerl = ()
        tuplerl = tuplerl + (keypoint_left,)
        lpoints2f = cv2.KeyPoint_convert(tuplerl)
        xr = rpoints2f[0][0]
        yr = rpoints2f[0][1]
        xl = lpoints2f[0][0]
        yl = lpoints2f[0][1]
        # print(xr,yr)
        out = calculate_3d_point(xl, yl, xr, yr)
        # print(out)
        # xyz = np.expand_dims(np.array(out), axis=0)
        # my3D_points=np.append(my3D_points,xyz,axis=0)
        my3D_points.append(list(out))
    print(my3D_points)
    return my3D_points

###########################################
##############把特征点画在一张图上#############
def draw_point_in_one(img,matches):
    tupler_match_kp = ()
    for i in range(len(matches)):
        tempmatch=matches[i]
        keypoint = kp1[tempmatch.queryIdx]
        tupler_match_kp = tupler_match_kp + (keypoint,)
    outimg1 = cv2.drawKeypoints(img, keypoints=tupler_match_kp, outImage=None)
    cv2.imshow("Match Result", outimg1)
    # cv.imwrite('./images/only_points.png',outimg1)
    cv2.waitKey(0)


img1 = cv2.imread("img/V2308l.png")
img2 = cv2.imread("img/V2308R.png")

orb_start = time.time()
orb = cv2.ORB_create(2000)
# 设施fast角点阈值，值越小越灵敏
orb.setFastThreshold(0)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
orb_end = time.time()

# 初始化 BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# 对描述子进行匹配
matches = bf.match(des1, des2)
start = time.time()
matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches, withScale=False, withRotation=False,
                       thresholdFactor=6)
end = time.time()

#     draw_types
#     ONLY_LINES = 1
#     LINES_AND_POINTS = 2
#     COLOR_CODED_POINTS_X = 3
#     COLOR_CODED_POINTS_Y = 4
#     COLOR_CODED_POINTS_XpY = 5
output = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType. COLOR_CODED_POINTS_X)
cv2.imshow("show", output)
cv2.waitKey(0)

#得到3d坐标
# my3dpoints = get_3dpoints(matches_gms)
slam_point = slam_get_3d(matches_gms,K_l, K_r, R, T)
# 可视化三维点
my3D_points=np.array(slam_point)
my3D_points = use_meanAndstd_2select(my3D_points)
my3D_points = use_meanAndstd_2select(my3D_points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(my3D_points[:, 0], my3D_points[:, 2], my3D_points[:, 1])
# ax.scatter(my3D_points[:, 0], my3D_points[:, 1], my3D_points[:, 2])
ax.scatter(0, 0, 0, marker='o', color='r', s=100)
# ax.set_xlim(ax.get_xlim()[::-1])
ax.set_zlim(ax.get_zlim()[::-1])
# ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel('X')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
plt.show()

# draw_point_in_one(img1,matches_gms)

# 转换为 pandas 的 DataFrame 格式
df = pd.DataFrame(my3D_points)

# 将 DataFrame 写入 Excel 文件
writer = pd.ExcelWriter('kp.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()

