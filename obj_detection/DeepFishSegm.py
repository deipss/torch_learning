import matplotlib.pyplot as plt

import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class DeepFishSegm(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        #root = /data/ai_data/DeepFish/Segmentation
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "/images/valid"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "/masks/valid"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images/valid", self.imgs[idx])
        mask_path = os.path.join(self.root, "/mask/valid", self.masks[idx])
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
        img = tv_tensors.Image(img)

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
    image = read_image("/media/deipss/cbe791b9-6e15-4287-b9c3-18c0125a33db/ai_data/DeepFish/Segmentation/images/valid/7398_F6_f000080.jpg")
    mask = read_image("/media/deipss/cbe791b9-6e15-4287-b9c3-18c0125a33db/ai_data/DeepFish/Segmentation/masks/valid/7398_F6_f000080.png")
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
