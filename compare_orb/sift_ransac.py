import time

import cv2
import numpy as np

# Load the images
imgL = cv2.imread('img/V894l.png')
imgR = cv2.imread('img/V894R.png')
img1 = cv2.cvtColor(imgL, cv2. COLOR_BGR2GRAY)
img2 = cv2.cvtColor(imgR, cv2. COLOR_BGR2GRAY)

time1 = time.time()
for i in range (100):
    # Initialize the SIFT detector
    sift = cv2.SIFT_create(500)

    # Find keypoints and descriptors in the images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize the BFMatcher
    bf = cv2.BFMatcher()

    # Match the descriptors
    matches = bf.knnMatch(des1, des2, k=2)

time2 = time.time()
# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.70 * n.distance:
        good_matches.append(m)

# Use RANSAC to estimate the homography matrix
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply the homography to image 1
h, w = img1.shape
im1_aligned = cv2.warpPerspective(img1, M, (w, h))

# Show the images
# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 2", img2)
# cv2.imshow("Aligned Image 1", im1_aligned)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 计算匹配准确率
total_matches = len(matches)
correct_matches = np.sum(mask)

accuracy = correct_matches / total_matches

print('Total matches:', total_matches)
print('Correct matches:', correct_matches)
print('Accuracy:', accuracy)
print('time:', time2-time1)
# Draw matches and show result
matchesMask = mask.ravel().tolist()
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
result = cv2.drawMatches(imgL, kp1, imgR, kp2, good_matches, None,  **draw_params)

# Save result image
# cv2.imwrite('out/sift_ransac_matches_book.png', result)
