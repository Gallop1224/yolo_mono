import cv2
import numpy as np


# 左/右相机内参数、旋转、平移矩阵
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
    print(XYZ)
    return XYZ

