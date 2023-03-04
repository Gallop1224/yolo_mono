import cv2 as cv  ## opencv-python==3.4.8.29
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
im1 = cv.imread('images/V894l.png')
im2 = cv.imread('images/V894R.png')

# ORB特征提取
orb = cv.ORB_create()
kp1 = orb.detect(im1)
kp2 = orb.detect(im2)
kp1, des1 = orb.compute(im1, kp1)  # 求特征
kp2, des2 = orb.compute(im2, kp2)  # 求特征
bf = cv.BFMatcher(cv.NORM_HAMMING)  # 初始化Matcher
matches = bf.match(des1, des2)  # 配准
# 进行初步筛选
min_distance = 10000
max_distance = 0
for x in matches:
    if x.distance < min_distance: min_distance = x.distance
    if x.distance > max_distance: max_distance = x.distance
print('最小距离：%f' % min_distance)
print('最大距离：%f' % max_distance)
good_match = []
for x in matches:
    if x.distance <= max(2 * min_distance, 30):
        good_match.append(x)
print('匹配数：%d' % len(good_match))
outimage = cv.drawMatches(im1, kp1, im2, kp2, good_match, outImg=None)
plt.imshow(outimage[:, :, ::-1])
plt.show()

K_l = np.array([[878.35, 0, 572.27],
                [0, 878.14, 383.91],
                [0, 0, 1]])
K_r = np.array([[880.71, 0, 577.32],
                [0, 897.92, 382.93],
                [0, 0, 1]])

R = np.array([[1, 0.0009, -0.0054],
                  [-0.0009, 1, -0.0031],
                  [0.0054, 0.0031, 1.0000]])
t = np.array([[-119.7164],
                  [-0.0681],
                  [0.6998]])
# 提取配准点
points1 = []
points2 = []
for i in good_match:
    points1.append(list(kp1[i.queryIdx].pt));
    points2.append(list(kp2[i.trainIdx].pt));
points1 = np.array(points1)
points2 = np.array(points2)


projMatr1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])    # 第一个相机参数
projMatr2 = np.concatenate((R, t), axis=1)               # 第二个相机参数
projMatr1 = np.matmul(K_l, projMatr1) # 相机内参 相机外参
projMatr2 = np.matmul(K_r, projMatr2) #
points4D = cv.triangulatePoints(projMatr1, projMatr2, points1.T, points2.T)
points4D /= points4D[3]       # 归一化
points4D = points4D.T[:,0:3]  # 取坐标点
print(points4D[0:5])


my3D_points=np.array(points4D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(my3D_points[:, 0], my3D_points[:, 2], my3D_points[:, 1])
ax.scatter(0, 0, 0, marker='o', color='r', s=100)
# ax.set_xlim(ax.get_xlim()[::-1])
# ax.set_zlim(ax.get_zlim()[::-1])
# ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
