import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def compute_integral_image(image):
    """ 计算积分图 """
    integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return integral


def get_haar_like_features(integral, x, y, w, h, feature_type):
    """ 计算Haar-like特征 """
    if feature_type == 'edge':
        left = sum_region(integral, x, y, w // 2, h)
        right = sum_region(integral, x + w // 2, y, w // 2, h)
        return left - right
    elif feature_type == 'line':
        top = sum_region(integral, x, y, w, h // 2)
        bottom = sum_region(integral, x, y + h // 2, w, h // 2)
        return top - bottom
    elif feature_type == 'center_surround':
        center = sum_region(integral, x + w // 4, y + h // 4, w // 2, h // 2)
        surround = sum_region(integral, x, y, w, h) - center
        return center - surround
    else:
        raise ValueError("Unsupported feature type")


def sum_region(integral, x, y, w, h):
    """ 计算矩形区域的像素和 """
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    result = integral[y2, x2] + integral[y1, x1] - integral[y1, x2] - integral[y2, x1]
    return result


def train_weak_classifiers(features, labels, num_classifiers):
    """ 使用AdaBoost训练弱分类器 """
    base_classifier = DecisionTreeClassifier(max_depth=1)
    adaboost = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=num_classifiers)
    adaboost.fit(features, labels)
    return adaboost

    # 构建强分类器


def build_strong_classifier(weak_classifiers):
    def strong_classifier(image):
        integral = compute_integral_image(image)
        feature = get_haar_like_features(integral, 0, 0, image.shape[1]-1, image.shape[0]-1, feature_type)
        predictions = [clf.predict([[feature]]) for clf in weak_classifiers]
        return np.mean(predictions) > 0.5

    return strong_classifier


def build_cascade_classifier(strong_classifiers, stages):
    """ 构建级联分类器 """

    def cascade_classifier(image):
        for stage in range(stages):
            if not strong_classifiers[stage](image):
                return False
        return True

    return cascade_classifier


def load_data():
    import os
    import cv2
    import numpy as np

    # 数据集路径
    dataset_path = '../data'
    face_dir = os.path.join(dataset_path, 'PennFudanPed/PNGImages')
    non_face_dir = os.path.join(dataset_path, 'PennFudanPed/PedMasks')

    # 图像大小
    image_size = (24, 24)

    # 加载图像数据
    def load_images_from_folder(folder, label):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
        return images, labels

    # 加载人脸图像
    face_images, face_labels = load_images_from_folder(face_dir, 1)

    # 加载非人脸图像
    non_face_images, non_face_labels = load_images_from_folder(non_face_dir, 0)

    # 合并数据
    X = face_images + non_face_images
    y = face_labels + non_face_labels

    # 将图像数据转换为NumPy数组
    X = np.array(X)
    y = np.array(y)

    # 打乱数据
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 保存数据
    np.save('X.npy', X)
    np.save('y.npy', y)

    # 提取特征


def extract_features(images, feature_type):
    features = []
    for image in images:
        integral = compute_integral_image(image)
        feature = get_haar_like_features(integral, 0, 0, image.shape[1]-1, image.shape[0]-1, feature_type)
        features.append(feature)
    return np.array(features).reshape(-1, 1)


if __name__ == '__main__':

    # 加载数据
    X = np.load('X.npy')
    y = np.load('y.npy')

    # 训练弱分类器
    num_classifiers = 10
    feature_type = 'edge'  # 选择一个Haar-like特征类型
    X_features = extract_features(X, feature_type)
    weak_classifiers = train_weak_classifiers(X_features, y, num_classifiers)

    strong_classifier = build_strong_classifier(weak_classifiers)

    # 测试分类器
    test_image = cv2.imread('/Users/deipss/workspace/ai/torch_learning/data/img_align_celeba/000001.jpg', cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (24, 24))
    if strong_classifier(test_image):
        print("Face detected!")
    else:
        print("No face detected.")
