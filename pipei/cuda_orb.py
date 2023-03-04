import cv2

# 读取图像
image1 = cv2.imread('./images/V894L.png')
image2 = cv2.imread('./images/V894R.png')

# 定义ORB参数
nfeatures = 500
scaleFactor = 1.2
nlevels = 8
edgeThreshold = 31
patchSize = 31

# 创建ORB和描述符匹配器对象
orb = cv2.cuda_ORB.create(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels,
                          edgeThreshold=edgeThreshold, patchSize=patchSize)
matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)

# 将图像上传到CUDA设备
d_img1 = cv2.cuda_GpuMat(image1)
d_img2= cv2.cuda_GpuMat(image2)

# 在CUDA设备上计算ORB特征和描述符
kp1, des1 = orb.detectAndComputeAsync(d_img1, None)
kp2, des2 = orb.detectAndComputeAsync(d_img2, None)

# 在CUDA设备上进行描述符匹配
matches = matcher.matchAsync(des1, des2)

# 将结果下载到主机内存
kp1 = kp1.download()
des1 = des1.download()
kp2 = kp2.download()
des2 = des2.download()
matches = matches.download()

# 在主机上绘制ORB特征点
img_with_keypoints = cv2.drawKeypoints(d_img1, kp1, None)

# 在主机上绘制匹配结果
img_matches = cv2.drawMatches(d_img1, kp1, d_img2, kp2, matches, None)
cv2.imshow("Match Result", img_matches)
cv2.waitKey(0)
