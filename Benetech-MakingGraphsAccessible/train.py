import sys
sys.path.append('./dataloader/')
sys.path.append('./network/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from dataloader import GraphsImagesDataLoader # type: ignore
from net import *
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split

SAVE_DIR = './weights/'
dataset_dir = './datasets/train/images'
annotations_dir = './datasets/train/'
model_checkpoint = '/mnt/c/Users/mishr/Desktop/kaggle/Benetech-MakingGraphsAccessible/weights/inception_v3_wt.pth'

# Define the device to be used for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Define the dataset
image_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
dataset = GraphsImagesDataLoader(dataset_dir, annotations_dir, 1, transform=image_transform)
train_len = int(len(dataset) * 0.8)
test_len = len(dataset) - train_len

# Split the dataset into train and test
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

# Define the dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the hyperparameters
num_epochs = 10
learning_rate = 0.00001

# Define the model
model = InceptionV3(num_classes=5, pretrained=True, model_checkpoint=model_checkpoint)
model = model.get_model()

# Define the loss function and optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train the model
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for i, data in enumerate(train_loader):
        images = data['image']
        labels = data['class']
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs, aux_outputs = model(images)
        # print(outputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
        
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images = data['image']
            labels = data['class']
            images = images.to(device)
            labels = labels.to(device)
            outputs, aux_outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
    # Save the model checkpoint
    torch.save(model.state_dict(), SAVE_DIR + 'mdoel' + str(epoch + 1) + '.ckpt')


        
            



