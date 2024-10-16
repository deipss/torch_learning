import utils
import torchvision
from PennFudanDataset import  PennFudanDataset
from torchvision.transforms import v2 as T
import torch


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    dataset = PennFudanDataset('../data/PennFudanPed', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    # For Training iter把数据集加载器中每个batch都串起来，变成一个迭代器，next取迭代器中第一个元素。
    # 相当于取的第一个batch中的数据
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)

    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions
    print(predictions[0])
