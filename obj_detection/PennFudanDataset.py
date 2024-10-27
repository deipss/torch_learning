import matplotlib.pyplot as plt
import os
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image
import torch
import cv2
import numpy as np
import kornia

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def concat_opencv(image):
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
                     dim=-3)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(concat_opencv(img))
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def show_img():
    image = read_image("../data/PennFudanPed/PNGImages/FudanPed00064.png")
    mask = read_image("../data/PennFudanPed/PedMasks/FudanPed00064_mask.png")
    # image.size()  torch.Size([3, 438, 567])
    # mask.size()  torch.Size([1, 438, 567])
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("Image")
    # 在PyTorch中，image.permute(1, 2, 0)是一个常用的操作，用于改变张量的维度顺序。
    # 这种操作通常用于图像数据，因为图像张量通常以(C, H, W)的格式存储，
    # 其中C代表通道数（如RGB图像的3个颜色通道），H代表高度，W代表宽度。
    # 当你对一个图像张量使用permute(1, 2, 0)方法时，
    # 你将得到一个新的张量，其维度顺序变为(H, W, C)。这种变换有几种用途：
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask.permute(1, 2, 0))
    plt.colorbar()
    plt.show()




if __name__ == '__main__':
    show_img()
