from torchvision.transforms import v2 as T
import torch
import cv2
import numpy as np

import kornia


def multiple_channel():
    transforms = []
    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())

    transforms = T.Compose(transforms)

    img = torch.randint(0, 255, (1, 7, 3, 3), dtype=torch.int32)
    print(img)

    t_img = transforms(img)
    print(t_img.shape)
    print(t_img)


class CvConcatTransform(T.Transform):
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, image):
        print(image.shape)
        gray_image = kornia.color.rgb_to_grayscale(image)
        numpy_gray = gray_image.squeeze().numpy()
        laplacian = cv2.Laplacian(numpy_gray, cv2.CV_64F)
        sobelx = cv2.Sobel(numpy_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(numpy_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        Canny = cv2.Canny(numpy_gray, threshold1=50, threshold2=150)
        mean_c = cv2.adaptiveThreshold(numpy_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        gaussian_c = cv2.adaptiveThreshold(numpy_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

        # 将所有图像转换为张量并拼接
        laplacian = torch.from_numpy(laplacian).unsqueeze(0)
        sobel = torch.from_numpy(sobel).unsqueeze(0)
        Canny = torch.from_numpy(Canny).unsqueeze(0)
        mean_c = torch.from_numpy(mean_c).unsqueeze(0)
        gaussian_c = torch.from_numpy(gaussian_c).unsqueeze(0)

        return torch.cat([image, gray_image, laplacian, sobel, Canny, mean_c, gaussian_c],
                         dim=self.dim)


if __name__ == '__main__':
    # 示例使用

    tensor1 = torch.randint(0, 4, (3, 4, 4), dtype=torch.uint8)
    tensor2 = torch.randint(5, 8, (3, 4, 4), dtype=torch.uint8)

    concat_transform = CvConcatTransform(dim=-3)

    print(tensor1)
    print(tensor2)
    result = concat_transform(tensor1)

    print(result.shape)  # 输出拼接后的张量形状
    print(result)
