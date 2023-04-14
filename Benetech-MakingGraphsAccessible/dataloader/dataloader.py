from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


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

    def __len__(self):
        return len(os.listdir(self.dataset_dir))
    
    def read_csv_batch(self):
        for chunk in pd.read_csv(self.annotation_dir + 'train.csv', chunksize=self.batch_size):
            yield chunk
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_dir, str(self.annotation_info.loc[idx, 'ID'])) + '.jpg'
        image = cv2.imread(img_name)
        graph_type = self.annotation_info.loc[idx, 'chart-type']
        sample = {'image': image, 'class': graph_type}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

if __name__ == '__main__':
    dataset_dir = '../datasets/train/images'
    annotations_dir = '../datasets/train/'
    dataloader = GraphsImagesDataLoader(dataset_dir, annotations_dir, 1)
    for i in range(len(dataloader)):
        sample = dataloader[i]
        print(sample['image'].shape, sample['class'])
        break
