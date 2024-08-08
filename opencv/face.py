import cv2

if __name__ == '__main__':

    # 加载 Haar 级联分类器
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取帧
        ret, frame = cap.read()

        # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 在检测到的人脸周围绘制矩形
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 显示结果帧
        cv2.imshow('Face Tracking', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
