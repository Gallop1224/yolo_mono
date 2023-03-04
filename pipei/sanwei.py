import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取左右相机的图像
img_left = cv2.imread('./images/V2267L.png',cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('./images/V2267R.png',cv2.IMREAD_GRAYSCALE)

# 创建SIFT特征检测器
sift = cv2.xfeatures2d.SIFT_create()

# 在左右两幅图像中提取特征点和描述符
keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

# 创建FLANN匹配器并进行特征点匹配
matcher = cv2.FlannBasedMatcher_create()
matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

# 过滤匹配点对，只保留最佳匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
#print(good_matches)

# 获取匹配点对的二维坐标
points_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用StereoBM算法计算视差图
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img_left, img_right)

# 进行三角测量，获取匹配点对的三维坐标
points_3d = cv2.triangulatePoints(np.eye(3, 4), np.eye(3, 4), points_left, points_right)
points_3d /= points_3d[3]

# 提取视差值大于0的匹配点对的三维坐标
points_3d_good = []
for i in range(len(good_matches)):
    u_left = int(points_left[i, 0, 0])
    v_left = int(points_left[i, 0, 1])
    if disparity[v_left, u_left] > 0:
        point_3d = np.array([points_3d[0, i], points_3d[1, i], points_3d[2, i]])
        points_3d_good.append(point_3d)

# 可视化三维坐标
points_3d_good = np.array(points_3d_good)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d_good[:, 0], points_3d_good[:, 1], points_3d_good[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
