import numpy as np

# 坐标值
coords = np.array([[-600.00, 100.00, 900.00],
                   [-600.00, 100.00, 1500.00],
                   [-600.00, 100.00, 2100.00],
                   [-600.00, 100.00, 2700.00],
                   [0.00, 100.00, 900.00],
                   [0.00, 100.00, 1500.00],
                   [0.00, 100.00, 2100.00],
                   [0.00, 100.00, 2700.00],
                   [600.00, 100.00, 900.00],
                   [600.00, 100.00, 1500.00],
                   [600.00, 100.00, 2100.00],
                   [600.00, 100.00, 2700.00]])

# 生成三个随机误差矩阵
error1 = np.random.uniform(low=-0.1, high=0.1, size=coords.shape[0])
error2 = np.random.uniform(low=-0.1, high=0.1, size=coords.shape[0])
error3 = np.random.uniform(low=-0.1, high=0.1, size=coords.shape[0])

# 将三个误差矩阵合并为一个矩阵
errors = np.stack((error1, error2, error3), axis=-1)

# 加上随机误差
coords_with_error = coords * (1 + errors)

print(coords_with_error)
