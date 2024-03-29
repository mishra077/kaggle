from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class GraphsImagesDataLoader(Dataset):
    """Graphs Images Dataset"""

    def __init__(self, dataset_dir, annotation_dir, batch_size, transform=None):
        """
        Args:
            dataset_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.annotation_dir = annotation_dir
        self.annotation_info = pd.read_csv(self.annotation_dir + 'train.csv')
        self.batch_size = batch_size
        self.transform = transform
        self.class_name = ['vertical_bar', 'horizontal_bar', 'dot', 'line', 'scatter']

    def __len__(self):
        return len(os.listdir(self.dataset_dir))
    
    def read_csv_batch(self):
        for chunk in pd.read_csv(self.annotation_dir + 'train.csv', chunksize=self.batch_size):
            yield chunk
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, str(self.annotation_info.loc[idx, 'ID'])) + '.jpg'
        image = cv2.imread(img_name)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        graph_type = str(self.annotation_info.loc[idx, 'chart-type'])
        graph_type = self.class_name.index(graph_type)


        if self.transform:
            image = self.transform(image)
            graph_type = torch.tensor(graph_type)

        sample = {'image': image, 'class': graph_type}

        return sample
    
class GraphsImagesDataLoaderTest(Dataset):
    """Graphs Images Dataset"""

    def __init__(self, dataset_dir, transform=None):
        """
        Args:
            dataset_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.class_name = ['vertical_bar', 'horizontal_bar', 'dot', 'line', 'scatter']
        self.files = os.listdir(self.dataset_dir)

    def __len__(self):
        print(len(os.listdir(self.dataset_dir)))
        return len(os.listdir(self.dataset_dir))
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, str(self.files[idx]))
        image = cv2.imread(img_name)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        graph_type = 0


        if self.transform:
            image = self.transform(image)
            graph_type = torch.tensor(graph_type)

        sample = {'image': image, 'class': graph_type}

        return sample
    

# if __name__ == '__main__':
#     dataset_dir = '../datasets/train/images'
#     dataset_test_dir = '../datasets/test/images'
#     # annotations_dir = '../datasets/train/'
#     image_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
#     # dataloader = GraphsImagesDataLoader(dataset_dir, annotations_dir, 1, transform=image_transform)
#     test_dataloader = GraphsImagesDataLoaderTest(dataset_test_dir, transform=image_transform)
#     # for i in range(len(dataloader)):
#     #     sample = dataloader[i]
#     #     print(sample['image'].size(), sample['class'])
#     #     break
#     for i in range(len(test_dataloader)):
#         sample = test_dataloader[i]
#         print(sample['image'].size())
#         break

