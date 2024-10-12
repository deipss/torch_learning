import cv2
import numpy as np


def windows_show():
    cv2.namedWindow("new", cv2.WINDOW_AUTOSIZE)
    cv2.imshow('new', 0)
    cv2.waitKey(0)


def capture_face():
    # 加载 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用XVID编码格式
    vw = cv2.VideoWriter('../data/out2.mp4', fourcc, 25, (1280, 720))

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # 读取帧
        ret, frame = cap.read()

        # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 在检测到的人脸周围绘制矩形
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示结果帧
        cv2.imshow('Face Tracking', frame)

        vw.write(frame)
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    vw.release()
    cv2.destroyAllWindows()


def print_vedio():
    # 打开摄像头。参数0表示第一个摄像头设备，如果有多个摄像头，可以使用1, 2等
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    while True:
        # 逐帧捕获
        ret, frame = cap.read()

        # 如果读取帧失败，则退出循环qq
        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 显示帧
        cv2.imshow('摄像头', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()


def mouse_capture():
    # 鼠标回调函数
    def draw_rectangle(event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # 创建一个黑色的图像窗口
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.setMouseCallback('test', draw_rectangle)
    while True:
        cv2.imshow('test', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def track_bar():
    # 加载图像

    # 定义调整亮度和对比度的函数
    def adjust_brightness_contrast(image, brightness=0, contrast=0):

        adjusted = image.copy()
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            adjusted = cv2.addWeighted(adjusted, alpha_b, adjusted, 0, gamma_b)
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            adjusted = cv2.addWeighted(adjusted, alpha_c, adjusted, 0, gamma_c)
        return adjusted

    # 创建窗口
    cv2.namedWindow('Image Adjustment')

    image = cv2.imread('../data/img_align_celeba/000001.jpg')

    # 创建轨迹条
    cv2.createTrackbar('Brightness', 'Image Adjustment', 0, 2 * 127, lambda x: None)
    cv2.createTrackbar('Contrast', 'Image Adjustment', 127, 2 * 127, lambda x: None)

    # 主循环
    while True:
        # 获取轨迹条的值
        brightness = cv2.getTrackbarPos('Brightness', 'Image Adjustment') - 127
        contrast = cv2.getTrackbarPos('Contrast', 'Image Adjustment') - 127

        # 调整图像的亮度和对比度
        adjusted_image = adjust_brightness_contrast(image, brightness, contrast)

        # 显示调整后的图像
        cv2.imshow('Image Adjustment', adjusted_image)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()
    # trackbar name
    # windows name
    # getTrackBar createTrackBarBas('R','trackbar',0,255,callback)

    # RGB  opencv is BGR HSV HSB HSL
    # HSV： HUE Saturation Value
    # YUV主要用于视频
    paint = np.zeros((480, 640, 3), np.uint8)

    #


def mat():
    pass
    # dim width height color-depth data char type

    img = np.full((606, 800, 3), 1, np.uint8)

    img[100:200, 300:400] = 240
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    print(img.shape)
    print(img.size)
    while True:
        cv2.imshow('test', img)
        if (cv2.waitKey(1) == ord('q')):
            break

    cv2.destroyAllWindows()


def channel():
    image = cv2.imread('../data/img_align_celeba/000001.jpg')
    # 读取图像

    # 拆分图像通道
    b, g, r = cv2.split(image)

    b[10:110, 10:110] = 1

    # 交换红色和蓝色通道
    custom_merged_image = cv2.merge([r, g, b])

    # 显示原始图像和拆分后的通道
    cv2.imshow('Original Image', image)
    cv2.imshow('Blue Channel', b)
    cv2.imshow('Green Channel', g)
    cv2.imshow('Red Channel', r)
    cv2.imshow('custom_merged_image Channel', custom_merged_image)

    # 等待按键，按任意键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # split 通道的分割
    # merge 通道的合并


def toHSV():
    # 加载图像
    image = cv2.imread('../data/img_align_celeba/000001.jpg')

    # 将图像从 BGR 转换为 HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 显示原始图像和 HSV 图像
    cv2.imshow('Original Image', image)
    cv2.imshow('HSV Image', hsv_image)

    # 等待按键按下以关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def paint_shape():
    # 创建一个黑色背景的图像
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # 绘制一条从(50, 50)到(450, 450)的红色直线
    cv2.line(image, (50, 50), (450, 450), (0, 0, 255), thickness=2)

    cv2.rectangle(image, (100, 100), (400, 400), (255, 0, 0), thickness=3)

    # 定义多边形的顶点
    pts = np.array([[150, 150], [250, 150], [250, 250], [150, 250]], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 绘制一个绿色的多边形
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 绘制一个半径为50的黄色圆
    cv2.circle(image, (256, 256), 50, (0, 255, 255), thickness=-1)

    cv2.ellipse(image, (256, 256), (100, 50), 45, 0, 360, (255, 0, 255), thickness=2)

    # 显示图像
    cv2.imshow('Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_operate():
    # sub add addWeighted

    # bit opr bitwise_not bitwise_and bitwise_or bitwise_xor

    image = cv2.imread('../data/img_align_celeba/000001.jpg')
    image2 = cv2.imread('../data/img_align_celeba/000002.jpg')
    img_add = cv2.add(image2, image)
    img_weight = cv2.addWeighted(image2, 0.7, image, 0.3, 0)

    center = (100, 100)  # 裁剪区域的中心点
    size = (30, 30)  # 裁剪区域的大小
    cropped_image = cv2.getRectSubPix(image, size, center)
    print(cropped_image.shape)

    # 显示裁剪后的图像
    cv2.imshow('Cropped Image using getRectSubPix', cropped_image)
    cv2.imshow('add2', img_add)
    cv2.imshow('img_weight', img_weight)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 读取图像


def img_opr():
    # resize 缩放算法
    # flip
    # rotate
    # warpAffine
    # getRotationMatrix2D
    # getAffineTransform
    # warpPerspectiveTransform
    image = cv2.imread('../data/img_align_celeba/000001.jpg')

    # 缩放到新尺寸
    resized_image = cv2.resize(image, (800, 600))

    # 或者按比例缩放
    resized_image_by_factor = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("resize img", resized_image)
    cv2.imshow("resize dimg", resized_image_by_factor)

    # 水平翻转
    horizontal_flip = cv2.flip(image, 1)

    # 垂直翻转
    vertical_flip = cv2.flip(image, 0)

    # 同时水平和垂直翻转
    both_flip = cv2.flip(image, -1)

    # 显示原始图像和翻转后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Horizontal Flip', horizontal_flip)
    cv2.imshow('Vertical Flip', vertical_flip)
    cv2.imshow('Both Flip', both_flip)
    # 获取图像中心点
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    angle = 45  # 旋转角度
    scale = 1.0  # 缩放比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 应用仿射变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # 显示原始图像和旋转后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def filter():
    # 低通滤波可以去除噪音或平滑图像
    # 高通滤波可以帮助查找图像的边缘
    # filter2D
    # kernel
    # anchor
    image = cv2.imread('../data/img_align_celeba/000001.jpg')
    # 定义一个3x3的均值滤波器
    kernel = np.ones((3, 3), np.float32) / 9

    # 应用滤波器
    filtered_image = cv2.filter2D(image, -1, kernel)

    # 显示原始图像和滤波后的图像
    cv2.imshow('Filtered Image', filtered_image)
    print(image.shape, filtered_image.shape)

    # 定义一个3x3的高斯模糊滤波器
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32) / 16

    # 应用高斯模糊滤波器
    gaussian_filtered_image = cv2.filter2D(image, -1, gaussian_kernel)

    # 显示原始图像和高斯模糊后的图像
    cv2.imshow('Gaussian Filtered Image', gaussian_filtered_image)

    # 定义一个3x3的Sobel边缘检测滤波器
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)

    # 应用Sobel边缘检测滤波器
    sobel_filtered_image_x = cv2.filter2D(image, -1, sobel_kernel_x)

    # 显示原始图像和Sobel边缘检测后的图像
    cv2.imshow('Sobel Edge Detection (X)', sobel_filtered_image_x)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filter_one_dim():
    import cv2
    import numpy as np

    # 加载图像并转换为灰度图像
    image = cv2.imread('../data/img_align_celeba/000001.jpg')

    # 计算水平方向的梯度
    G_x = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 1]]))

    # 计算垂直方向的梯度
    G_y = cv2.filter2D(image, cv2.CV_64F, np.array([[-1], [1]]))

    # 计算梯度的幅度
    magnitude = np.sqrt(G_x ** 2 + G_y ** 2)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Gradient Magnitude', magnitude / magnitude.max())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def boxFilter():
    # 加载图像
    image = cv2.imread('../data/img_align_celeba/000001.jpg', cv2.IMREAD_GRAYSCALE)

    # 应用盒状滤波
    filtered_image = cv2.boxFilter(image, -1, (3, 3))
    blur_image = cv2.blur(image, (5, 5))
    medianBlur = cv2.medianBlur(image, 5)

    GaussianBlur = cv2.GaussianBlur(image, (5, 5), 0)

    # 应用双边滤波
    bilateralFilter = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # 使用拉普拉斯滤波器
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # 使用Sobel算子
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 使用Scharr滤波器计算水平方向的梯度
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)

    # 使用Scharr滤波器计算垂直方向的梯度
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)

    # 计算梯度的幅度
    magnitude = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    # 应用Canny边缘检测
    Canny = cv2.Canny(image, threshold1=50, threshold2=150)

    # 使用Mean-C方法进行自适应阈值处理
    mean_c = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 使用Gaussian-C方法进行自适应阈值处理
    gaussian_c = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 显示结果
    cv2.imshow('Mean-C Adaptive Threshold', mean_c)
    cv2.imshow('Gaussian-C Adaptive Threshold', gaussian_c)

    # 显示结果
    cv2.imshow('Canny', Canny)

    # 显示结果
    cv2.imshow('Scharr X Gradient', np.abs(scharr_x).astype(np.uint8))
    cv2.imshow('blur_image', blur_image)
    cv2.imshow('Scharr Y Gradient', np.abs(scharr_y).astype(np.uint8))
    cv2.imshow('Gradient Magnitude', np.abs(magnitude).astype(np.uint8))
    # 显示结果
    cv2.imshow('Laplacian Filtered Image', np.abs(laplacian).astype(np.uint8))
    cv2.imshow('Sobel Filtered Image', np.abs(sobel).astype(np.uint8))

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.imshow('medianBlur', medianBlur)
    cv2.imshow('GaussianBlur', GaussianBlur)
    cv2.imshow('bilateralFilter', bilateralFilter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morphology():
    image = cv2.imread('../data/img_align_celeba/000001.jpg', cv2.IMREAD_GRAYSCALE)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 进行腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    # 开运算：先腐蚀再膨胀
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # 闭运算：先膨胀再腐蚀
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # 进行顶帽操作
    tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    # 进行黑帽操作
    blackhat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    # 显示原始图像和腐蚀后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Eroded Image', eroded_image)
    cv2.imshow('Dilated Image', dilated_image)

    cv2.imshow('Opened Image', opened_image)
    cv2.imshow('Closed Image', closed_image)
    cv2.imshow('Top-Hat Image', tophat_image)
    cv2.imshow('Black-Hat Image', blackhat_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contours():
    import cv2
    import numpy as np

    # 加载图像并转换为灰度图像
    image = cv2.imread('../data/img_align_celeba/000001.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行二值化处理
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制所有轮廓
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # 处理每个轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        print(f'Contour Area: {area}')

        # 近似轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)

        # 计算凸包
        hull = cv2.convexHull(contour)
        cv2.drawContours(image, [hull], -1, (255, 0, 0), 2)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (255, 255, 0), 2)

        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 显示结果
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def feature_detect():
    # 读取图像并转换为灰度图像
    image = cv2.imread('../data/img_align_celeba/000001.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 使用 Harris 角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 归一化结果
    dst = cv2.dilate(dst, None)

    # 阈值处理，标记角点
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    # 显示结果
    cv2.imshow('Harris Corner Detection', image)

    # 使用 Shi-Tomasi 角点检测
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # 将角点坐标转换为整数
    corners = np.int0(corners)

    # 绘制检测到的角点
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    # 显示结果
    cv2.imshow('Shi-Tomasi Corner Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 以图搜图:特征匹配+单应性矩阵 findHomography(srcPts )
    # 图像拼接 角点、
    # 拼图游戏

    # harris角点 connerHarris
    #  goodFeaturesToTrack
    # SIFT 图片放大后，harrris不一定能检测出来，可以用SIFT
    # SURF 描述子
    # OBR

    # FLANN 特征匹配
    # drawMatchesKnn


# hstack

if __name__ == '__main__':
    feature_detect()
