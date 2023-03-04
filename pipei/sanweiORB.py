import numpy as np
import cv2
import matplotlib.pyplot as plt
# 读取左右两幅图像
img_left = cv2.imread('./images/V2267L.png')
img_right = cv2.imread('./images/V2267R.png')

# 创建ORB对象并设置参数
orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)

# 提取左右两幅图像的特征点和描述符
kp1, des1 = orb.detectAndCompute(img_left, None)
kp2, des2 = orb.detectAndCompute(img_right, None)

# 使用FLANN匹配器进行特征点匹配
#matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.match(des1, des2)

# 筛选出最佳匹配点对
matches = sorted(matches, key=lambda x: x.distance)
best_matches = matches[:100]
print(best_matches)
# 使用StereoBM算法计算左右两幅图像的视差图
sbm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
disparity = sbm.compute(gray_left, gray_right)

# 三角测量计算匹配点对的三维坐标
focal_length = 0.8 # 焦距
baseline = 0.12 # 基线长度
Q = np.float32([[1, 0, 0, -img_left.shape[1]/2],
                [0, 1, 0, -img_left.shape[0]/2],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]])
points_3d = cv2.reprojectImageTo3D(disparity, Q)
points_3d = points_3d[:, :, :3]

# 根据视差值筛选出视差大于0的匹配点对
points = []
for match in best_matches:
    u_left = int(kp1[match.queryIdx].pt[0])
    v_left = int(kp1[match.queryIdx].pt[1])
    u_right = int(kp2[match.trainIdx].pt[0])
    v_right = int(kp2[match.trainIdx].pt[1])
    if disparity[v_left, u_left] > 0:
        point_3d = np.array([points_3d[v_left, u_left, 0], points_3d[v_left, u_left, 1], points_3d[v_left, u_left, 2]])
        points.append(point_3d)

# 可视化三维坐标
points = np.array(points)
print(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
plt.show()
