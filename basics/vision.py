import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image
def anchor_demo():
    # 定义锚框生成器
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  # 不同尺度的锚框
        aspect_ratios=((0.5, 1.0, 2.0),)   # 不同长宽比的锚框
    )
    image_path = "/Users/deipss/workspace/ai/torch_learning/data/PennFudanPed/PNGImages/FudanPed00007.png"
    image = read_image(image_path)
    # 假设有一个特征图
    feature_map = torch.rand(1, 256, 32, 32)  # (batch_size, channels, height, width)

    # 生成锚框
    anchors = anchor_generator(image,feature_map)

    output_image = draw_bounding_boxes(image, anchors, colors="red")

    # 打印生成的锚框
    print(anchors)

    # 显示图像
    plt.imshow(output_image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


if __name__ == '__main__':
    anchor_demo()