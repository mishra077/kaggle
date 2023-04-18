import sys
sys.path.append('./dataloader/')
sys.path.append('./network/')
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from dataloader import GraphsImagesDataLoaderTest # type: ignore
from net import *
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

SAVE_DIR = './weights/'
dataset_dir = './datasets/test/images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

image_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
test_dataset = GraphsImagesDataLoaderTest(dataset_dir, transform=image_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class_name = ['vertical_bar', 'horizontal_bar', 'dot', 'line', 'scatter']


# load the model
model = InceptionV3(num_classes=5, pretrained=False, model_checkpoint='./weights/inception_v3_wt.pth')
model = model.get_model()
checkpoint = torch.load('./weights/model6.ckpt')
model.load_state_dict(checkpoint)
model.to(device)

# test the model
model.eval()
correct_count = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        images = data['image']
        images = images.to(device)

        # Forward pass
        outputs= model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(class_name[predicted.item()])