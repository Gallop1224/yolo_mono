# 这个实验是为了从匹配到的点对中获得左右坐标
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
    cv.solve(A, B, XYZ, cv.DECOMP_SVD)
    # print(XYZ)
    XYZ = XYZ.reshape(1, 3)
    xyz = XYZ[0]

    return xyz


img1 = cv.imread('./images/V894L.png')
img2 = cv.imread('./images/V894R.png')
# 初始化ORB
orb = cv.ORB_create(500)

# 寻找关键点
kp1 = orb.detect(img1)
kp2 = orb.detect(img2)

# 计算描述符
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

# 画出关键点
outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)

# 显示关键点


# outimg3 = np.hstack([outimg1, outimg2])
# cv.imshow("Key Points", outimg3)
# cv.waitKey(0)

# 初始化 BFMatcher
bf = cv.BFMatcher(cv.NORM_HAMMING)

# 对描述子进行匹配
matches = bf.match(des1, des2)

# 选择好点
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
print('选完后点的个数', len(good_match))
#############################################################
n = len(good_match)
my3D_points = []
for i in range(n):
    tempmatch = good_match[i]
    keypoint_left = kp1[tempmatch.queryIdx]
    keypoint_right = kp2[tempmatch.queryIdx]
    # print(keypoint_right)
    keypoint_right2 = (keypoint_right)
    tupler = ()
    tupler = tupler + (keypoint_right,)
    rpoints2f = cv.KeyPoint_convert(tupler)
    # print(points2f)
    tuplerl = ()
    tuplerl = tuplerl + (keypoint_left,)
    lpoints2f = cv.KeyPoint_convert(tuplerl)
    xr, yr = rpoints2f[0][0], rpoints2f[0][1]
    xl, yl = lpoints2f[0][0], lpoints2f[0][1]
    # print(xr,yr)
    # out = uvToXYZ(xl, yl, xr, yr)
    out = calculate_3d_point(xl, yl, xr, yr)
    # print(out)
    # xyz = np.expand_dims(np.array(out), axis=0)
    # my3D_points=np.append(my3D_points,xyz,axis=0)
    my3D_points.append(list(out))

print(my3D_points)

# 可视化三维点
my3D_points = np.array(my3D_points)
data = np.random.rand(10, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(my3D_points[:, 0], my3D_points[:, 2], my3D_points[:, 1])
ax.set_xlim(ax.get_xlim()[::-1])
# ax.set_zlim(ax.get_zlim()[::-1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 创建一个元组，把每一个good_match里的kp找出来，放进元组
tupler_goodmatch_kp = ()
for i in range(n):
    tempmatch = good_match[i]
    keypoint_left = kp2[tempmatch.queryIdx]

    tupler_goodmatch_kp = tupler_goodmatch_kp + (keypoint_left,)
outimg1 = cv.drawKeypoints(img1, keypoints=tupler_goodmatch_kp, outImage=None)
cv.imshow("Match Result", outimg1)
cv.imwrite('./images/only_points.png', outimg1)
cv.waitKey(0)
