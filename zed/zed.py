import numpy as np
import os
import pyzed.sl as sl
import cv2
# 2. 捕获图像
def image_capture():
    zed = sl.Camera()
    # 设置相机的分辨率1080和采集帧率30fps
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # fps可选：15、30、60、100

    err = zed.open(init_params)  # 根据自定义参数打开相机
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    runtime_parameters = sl.RuntimeParameters()  # 设置相机获取参数
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
    i = 0
    # 创建sl.Mat对象来存储图像（容器），Mat类可以处理1到4个通道的多种矩阵格式（定义储存图象的类型）
    image = sl.Mat()  # 图像
    imageR = sl.Mat()
    disparity = sl.Mat()  # 视差值
    dep = sl.Mat()  # 深度图
    depth = sl.Mat()  # 深度值
    point_cloud = sl.Mat()  # 点云数据
    # 获取分辨率
    resolution = zed.get_camera_information().camera_resolution
    w, h = resolution.width , resolution.height
    x,y = int(w/2),int(h/2)  # 中心点

    while True:
        # 获取最新的图像，修正它们，并基于提供的RuntimeParameters(深度，点云，跟踪等)计算测量值。
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  # 相机成功获取图象
            # 获取图像
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # 获取图像被捕获时的时间点
            zed.retrieve_image(image, sl.VIEW.LEFT)  # image：容器，sl.VIEW.LEFT：内容
            imgL = image.get_data()  # 转换成图像数组，便于后续的显示或者储存
            zed.retrieve_image(imageR, sl.VIEW.RIGHT)
            imgR = imageR.get_data()
            # 获取视差值
            zed.retrieve_measure(disparity,sl.MEASURE.DISPARITY,sl.MEM.CPU)
            dis_map = disparity.get_data()
            # 获取深度
            zed.retrieve_measure(depth,sl.MEASURE.DEPTH,sl.MEM.CPU)  # 深度值
            zed.retrieve_image(dep,sl.VIEW.DEPTH)  # 深度图
            depth_map = depth.get_data()
            dep_map = dep.get_data()
            # 获取点云
            zed.retrieve_measure(point_cloud,sl.MEASURE.XYZBGRA,sl.MEM.CPU)
            point_map = point_cloud.get_data()
            print('时间点',timestamp.get_seconds(),'中心点视差值',dis_map[x,y],'中心点深度值',depth_map[x,y],'中心点云数据',point_map[x,y])
            # 利用cv2.imshow显示视图，并对想要的视图进行保存
            view = np.concatenate((cv2.resize(imgL,(640,360)),cv2.resize(dep_map,(640,360))),axis=1)
            cv2.imshow("View", view)
            key = cv2.waitKey(1)
            if key & 0xFF == 27:  # esc退出
                break
            if key & 0xFF == ord('s'):  # 图像保存
                savePath = os.path.join("./img_flower", "V{:0>3d}.png".format(i))  # 注意根目录是否存在"./images"文件夹
                cv2.imwrite(savePath, view)
                savePathL = os.path.join("./img_flower", "V{:0>3d}l.png".format(i))
                cv2.imwrite(savePathL, imgL)
                savePathR = os.path.join("./img_flower", "V{:0>3d}R.png".format(i))
                cv2.imwrite(savePathR, imgR)
            i = i + 1
    zed.close()

def hello_zed():
    # 创建相机对象
    zed = sl.Camera()  # Camera是非常重要的一个类

    # 创建初始化参数对象并配置初始化参数
    init_params = sl.InitParameters()
    init_params.sdk_verbose = False  # 相机有很多可以初始化的参数，用到一个认识一个

    # 打开相机（终端打开，但是看不到相机的画面，需要用到cv2.imshow显示相机画面，后面再介绍）
    err = zed.open(init_params)  # 指定参数打开相机
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    # 获得相机的信息，笔者列举了一部分，并不是全部信息，读者可以自行探究
    zed_info = zed.get_camera_information()
    print('相机序列号：%s' % zed_info.serial_number)
    print('相机型号：%s' % zed_info.camera_model)
    print('相机分辨率: width:%s, height:%s' % (zed_info.camera_resolution.width, zed_info.camera_resolution.height))
    print('相机FPS：%s' % zed_info.camera_fps)
    print('相机外部参数：')
    print('相机旋转矩阵R：%s' % zed_info.calibration_parameters.R)
    print('相机变换矩阵T：%s' % zed_info.calibration_parameters.T)
    print('相机基距：%s' % zed_info.calibration_parameters.get_camera_baseline())
    print('初始化参数：')
    zed_init = zed.get_init_parameters()
    print('相机分辨率：%s' % (zed_init.camera_resolution))
    print('深度最小：%s' % (zed_init.depth_minimum_distance))
    print('深度最大：%s' % (zed_init.depth_maximum_distance))
    # 关闭相机
    zed.close()

if __name__ == "__main__":
    #hello_zed()
    image_capture()
