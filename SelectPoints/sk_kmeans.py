import time

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# 读取 Excel 文件中的数据
kp = pd.read_excel('kp.xls').to_numpy()
# 打印数据框的内容
print(kp)


# 使用 k-means 聚类
starttime=time.time()
kmeans = KMeans(n_clusters=3, random_state=0).fit(kp)
endtime=time.time()
print('聚类时间', starttime-endtime)
# 将每个点的标签转换为颜色
colors = ['r', 'g', 'b']
labels = kmeans.labels_
c = [colors[labels[i]] for i in range(len(kp))]

# 将三维点绘制在三维坐标系中
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(kp[:,0], kp[:,2], kp[:,1], c=c)
ax.set_xlim(ax.get_xlim()[::-1])
# ax.set_zlim(ax.get_zlim()[::-1])
ax.set_ylim(ax.get_ylim()[::-1])
# 显示图像
plt.show()
