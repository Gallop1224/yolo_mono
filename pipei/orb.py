import cv2 as cv


def ORB_Feature(img1, img2):
    # 初始化ORB
    orb = cv.ORB_create(100)

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # 画出关键点
    outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)
    cv.imshow('kp',outimg1)
    cv.waitKey(0)
    #显示关键点
    import numpy as np
    outimg3 = np.hstack([outimg1, outimg2])
    # cv.imshow("Key Points", outimg3)
    # cv.waitKey(0)

    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)

    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)
    print(good_match)
    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)
    #print(good_match)
    # draw_match(img1, img2, kp1, kp2, matches)

def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    #cv.imwrite('./images/orb_detec_no_select.png',outimage)
    cv.waitKey(0)










image1 = cv.imread('./images/iphone1.jpg')
image2 = cv.imread('./images/iphone2.jpg')

ORB_Feature(image1, image2)
