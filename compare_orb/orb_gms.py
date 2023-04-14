import time

import cv2
import numpy as np
from cv2.xfeatures2d import matchGMS


def calculateRepeatability(kp1, kp2, matches, thr=3):
    kp1 = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
    kp2 = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])
    idx1 = [m.queryIdx for m in matches]
    idx2 = [m.trainIdx for m in matches]
    p1 = kp1[idx1, :]
    p2 = kp2[idx2, :]
    if len(p1) == 0 or len(p2) == 0:
        return 0
    d = np.sqrt(np.sum((p1 - p2) ** 2, axis=1))
    return np.sum(d <= thr) / float(len(matches))

# Load the images
img1 = cv2.imread('img/V894l.png')
img2 = cv2.imread('img/V894R.png')


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

        # Use GMS to refine the matches
    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches, withScale=False, withRotation=False,
                               thresholdFactor=6)


time2 = time.time()
# 计算匹配准确率
total_matches = len(matches)

matches_dict = {m.queryIdx: (m.trainIdx, m.distance) for m in matches}
correct_matches = 0
for match in matches_gms:
    ref_idx = match.queryIdx
    if ref_idx in matches_dict:
        train_idx, distance = matches_dict[ref_idx]
        if match.trainIdx == train_idx and match.distance > 0.8 * distance:
            correct_matches += 1

accuracy = correct_matches / total_matches



print('Total matches:', total_matches)
print('Correct matches:', correct_matches)
print('Accuracy:', accuracy)
print('time:', time2-time1)

matchColor=(0, 255, 0)

outimage = cv2.drawMatches(img1, kp1, img2, kp2, matches_gms, outImg=None, flags=2, matchColor=(0, 255, 0))
# cv2.imshow("Match Result", outimage)
# # cv2.imwrite('out/orb_gms_book.png',outimage)
# cv2.waitKey(0)