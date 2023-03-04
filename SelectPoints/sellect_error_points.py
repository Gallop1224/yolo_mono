import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor

# 生成一组三维点，其中含有明显误差
np.random.seed(0)
X = np.random.randn(100, 3)
X[0, :] = [100, 100, 100]
X[1, :] = [-100, -100, -100]

def use_ransac_2select(X):
    # 使用RANSAC算法拟合一个平面模型
    model = RANSACRegressor()
    model.fit(X[:, :2], X[:, 2])
    inliers_mask = model.inlier_mask_
    outliers_mask = np.logical_not(inliers_mask)

    # 仅保留模型拟合的内点
    X_clean = X[inliers_mask, :]

    # 打印删除误差后的三维点集
    print(X_clean)
    return X_clean

def use_meanAndstd_2select(X):
    # 计算z轴坐标值的均值和标准差，并计算阈值
    z_mean = np.mean(X[:, 2])
    z_std = np.std(X[:, 2])
    z_threshold = z_mean + 3 * z_std

    # 根据阈值删除偏离较大的数据点
    X_clean = X[abs(X[:, 2] - z_mean) <= z_threshold, :]

    # 打印删除误差后的三维点集
    print(X_clean)
    return X_clean

X_clean = use_meanAndstd_2select(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_clean[:, 0], X_clean[:, 1], X_clean[:, 2], c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()