import time

import cv2
import numpy as np

# Load the images
imgL = cv2.imread('img/V894l.png')
imgR = cv2.imread('img/V894R.png')
img1 = cv2.cvtColor(imgL, cv2. COLOR_BGR2GRAY)
img2 = cv2.cvtColor(imgR, cv2. COLOR_BGR2GRAY)
# Initialize the ORB detector

time1 = time.time()
for i in range (100):
    orb = cv2.ORB_create(500)


        # Find keypoints and descriptors in the images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

        # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the descriptors
    matches = bf.match(des1, des2)

time2 = time.time()

# 设置RANSAC算法参数
ransacReprojThreshold = 3.0
confidence = 0.99
# Use RANSAC to estimate the homography matrix
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold, confidence)

# Apply the homography to image 1
h, w = img1.shape
im1_aligned = cv2.warpPerspective(img1, M, (w, h))

# Show the images
# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 2", img2)
# cv2.imshow("Aligned Image 1", im1_aligned)
# cv2.waitKey(0)

# Calculate the number of inliers
num_inliers = np.count_nonzero(mask)

# Calculate the percentage of inliers
inlier_ratio = num_inliers / len(matches) * 100

# Print the results
print("Number of matches:", len(matches))
print("Number of inliers:", num_inliers)
print("Inlier ratio: %.2f%%" % inlier_ratio)
print('time:', time2-time1)

matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
img3 = cv2.drawMatches(imgL, kp1, imgR, kp2, matches, None, **draw_params)

# 显示匹配结果
# cv2.imshow('Matches', img3)
# cv2.imwrite('out/orb_ransac_book.png', img3)
# cv2.waitKey()
# cv2.destroyAllWindows()