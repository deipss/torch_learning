import utils
from PennFudanDataset import PennFudanDataset
from DeepFishSegm import DeepFishSegm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.io import read_image
import torch.nn as nn
from torchvision.transforms import v2 as T
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406,0.485, 0.456, 0.406,0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225,0.229, 0.224, 0.225,0.229, 0.224, 0.225]))
    return T.Compose(transforms)

class CustomFasterRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=3)
        self.relu = nn.ReLU(inplace=True)
        self.inner = get_model_instance_segmentation(num_classes)
    def forward(self, images, targets=None):
        inner_img = []
        for i in images:
            x = self.conv1(i)
            x=x.unsqueeze(0)
            x = self.bn1(x)
            x = self.relu(x)
            inner_img.append(x.squeeze())
        return self.inner(inner_img,targets)


def get_model_detection_customized_segmentation(in_channels, num_classes):
    return CustomFasterRCNNPredictor(in_channels, num_classes)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation cnn_model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def trainPennFudanDataset():
    from engine import train_one_epoch, evaluate

    # train on the GPU or on the CPU, if a GPU is not available

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('../data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('../data/PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    test_len = len(indices) // 10
    print(f'test_len={test_len}')
    dataset = torch.utils.data.Subset(dataset, indices[:-test_len])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_len:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    # get the cnn_model using our helper function
    model = get_model_detection_customized_segmentation(in_channels=9, num_classes=num_classes)

    # move cnn_model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    torch.save(model.state_dict(), "../data/model_state_dict_penn.pth")


def trainDeepFishSegm():
    from engine import train_one_epoch, evaluate

    # train on the GPU or on the CPU, if a GPU is not available

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = DeepFishSegm('/data/ai_data/DeepFish/Segmentation', get_transform(train=True))
    dataset_test = DeepFishSegm('/data/ai_data/DeepFish/Segmentation', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    test_len = len(indices) // 10
    print(f'test_len={test_len}')
    dataset = torch.utils.data.Subset(dataset, indices[:-test_len])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_len:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    # get the cnn_model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move cnn_model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    torch.save(model.state_dict(), "../data/model_state_dict_deep_fish.pth")


def show():
    import matplotlib.pyplot as plt

    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load("../data/model_state_dict_deep_fish.pth"))
    model.to(device)

    # image = read_image("/data/ai_data/PennFudanPed/PNGImages/FudanPed00046.png")
    image = read_image(
        "/data/ai_data/DeepFish/Segmentation/images/valid/7117_Lutjanus_argentimaculatus_adult_2_f000060.jpg")
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    trainPennFudanDataset()
