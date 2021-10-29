from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import pandas as pd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        txt_file = "Caltech101/" + split + ".txt" #path of the correct txt file
        data = pd.read_csv(txt_file, header=None)
        self.data = data[(data[0].str.contains("BACKGROUND"))==False] #filtered the background images
        
        
        
        self.labels = {} #save a dictionary of labels
        targets = []
        for index, line in data.iterrows():
          folder = line[0].split("/")[0]
          targets.append(folder)
          if folder not in self.labels:
            self.labels[folder]= len(self.labels)
        
        
        self.data.loc[:,1] = targets
        

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int
        img_name = os.path.join(self.root,
                                self.data.iloc[index, 0])

        image =pil_loader(img_name)
        label = self.labels.get(self.data.iloc[index, 1])

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
