import cv2 as cv
import numpy as np

def ORB_Feature(img1):
    # 初始化ORB
    orb = cv.ORB_create(1000,2)

    # 寻找关键点
    kp1 = orb.detect(img1)
    print(kp1)
    points2f = cv.KeyPoint_convert(kp1)
    print(points2f)
    
    temp=[[1.00000,2.00000]]                  ###########我在这里加入了手动特征点
    temp=np.array(temp)
    points2f=np.append(points2f,temp,axis=0)
    print(points2f)
    kp1 = cv.KeyPoint_convert(points2f)

    ##################################################
    #作弊  增加kp ：kp=cv2.KeyPoint(x=1,y=2,size=3)
    #              print(kp)
    ###################################################

    #              print(kp)
   # kp2=kp1+kp


    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)



    # 画出关键点
    outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    return outimg1

    # 显示关键点
    # import numpy as np
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv.imshow("Key Points", outimg3)
    # cv.waitKey(0)

    # 初始化 BFMatcher
    # bf = cv.BFMatcher(cv.NORM_HAMMING)
    #
    # # 对描述子进行匹配
    # matches = bf.match(des1, des2)
    #
    # # 计算最大距离和最小距离
    # min_distance = matches[0].distance
    # max_distance = matches[0].distance
    # for x in matches:
    #     if x.distance < min_distance:
    #         min_distance = x.distance
    #     if x.distance > max_distance:
    #         max_distance = x.distance
    #
    # # 筛选匹配点
    # '''
    #     当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
    #     但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    # '''
    # good_match = []
    # for x in matches:
    #     if x.distance <= max(2 * min_distance, 30):
    #         good_match.append(x)
    #
    # # 绘制匹配结果
    # draw_match(img1, img2, kp1, kp2, good_match)
    #print(good_match)

# def draw_match(img1, img2, kp1, kp2, match):
#     outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
#     cv.imshow("Match Result", outimage)
#     cv.imwrite('./images/out.png',outimage)
#     cv.waitKey(0)










image1 = cv.imread('./images/flow3.png')
#image2 = cv.imread('./images/V894R.png')
outL = ORB_Feature(image1)
#outR=ORB_Feature(image2)
cv.imshow("Match ResultL", outL)
#cv.imshow("Match ResultR", outR)
#cv.imwrite('./images/outzidane.png',out)
cv.waitKey(0)