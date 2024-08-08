import cv2
if __name__ == '__main__':

    # 打开摄像头。参数0表示第一个摄像头设备，如果有多个摄像头，可以使用1, 2等
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 逐帧捕获
        ret, frame = cap.read()

        # 如果读取帧失败，则退出循环
        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 显示帧
        cv2.imshow('摄像头', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
