from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from PIL import Image
import torch
import os

import argparse


class PairDataset(Dataset):
    
    def __init__(self,imageFolder,transform=None):
        self.imageFolder = imageFolder 
        self.transform = transform
        
    def __getitem__(self,index):
        folders = os.listdir(self.imageFolder)
        folder = random.choice(folders)
        while not(os.path.isdir(self.imageFolder + folder)) and folder[:5] == 'other':
            folder = random.choice(folders)
        folderpath = os.path.join(self.imageFolder,folder)
        imgname = random.choice(os.listdir(folderpath))
        while not(imgname[-1] =='g'):
            imgname = random.choice(os.listdir(folderpath))
        img0 = Image.open(os.path.join(folderpath,imgname))

        not_same_class = random.uniform(0,1) 
        folder2=folder
        if not_same_class < 0.5:
            folder2 = random.choice(folders)
            while not(os.path.isdir(self.imageFolder + folder2)) and not(folder == folder2):
                folder2 = random.choice(folders)
            folderpath2 = os.path.join(self.imageFolder,folder2)
        else:
            folderpath2 = folderpath
        imgname = random.choice(os.listdir(folderpath2))
        while not(imgname[-1] =='g'):
            imgname = random.choice(os.listdir(folderpath2))
        img1 = Image.open(os.path.join(folderpath2,imgname))

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(folder!=folder2)],dtype=np.float32))

    
    def __len__(self):
        folders = os.listdir(self.imageFolder)
        count =0 
        for i in folders :
            count += len(os.listdir(os.path.join(self.imageFolder,i)))
        return count



class SimplePairDataset(Dataset):
    
    def __init__(self,imageFolder,transform=None):
        self.imageFolder = imageFolder 
        self.transform = transform
        
    def __getitem__(self,index):
        folders = os.listdir(self.imageFolder)
        folder = random.choice(folders)
        while not(os.path.isdir(self.imageFolder + folder)):
            folder = random.choice(folders)
        folderpath = os.path.join(self.imageFolder,folder)
        imgname = random.choice(os.listdir(folderpath))
        while not(imgname[-1] =='g'):
            imgname = random.choice(os.listdir(folderpath))
        img0 = Image.open(os.path.join(folderpath,imgname))

        not_same_class = random.uniform(0,1) 
        if not_same_class < 0.5:
            folder2 = random.choice(folders)
            while not(os.path.isdir(self.imageFolder + folder2)):
                folder2 = random.choice(folders)
            folderpath2 = os.path.join(self.imageFolder,folder2)
            imgname = random.choice(os.listdir(folderpath2))
            while not(imgname[-1] =='g'):
                imgname = random.choice(os.listdir(folderpath2))
            img1 = Image.open(os.path.join(folderpath2,imgname))
        else:
            img1 =img0
        

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(not_same_class < 0.5)],dtype=np.float32))

    
    def __len__(self):
        folders = os.listdir(self.imageFolder)
        count =0 
        for i in folders :
            count += len(os.listdir(os.path.join(self.imageFolder,i)))
        return count


class SingleImage(Dataset):
    
    def __init__(self,imageFolder,enumdict,transform=None):
        self.imageFolder = imageFolder 
        self.enumdict = enumdict
        self.transform = transform
        
    def __getitem__(self,index):
        folders = os.listdir(self.imageFolder)
        folder = random.choice(folders)
        while not(os.path.isdir(self.imageFolder + folder)):
            folder = random.choice(folders)
        folderpath = os.path.join(self.imageFolder,folder)
        imgname = random.choice(os.listdir(folderpath))
        while not(imgname[-1] =='g'):
            imgname = random.choice(os.listdir(folderpath))
        img0 = Image.open(os.path.join(folderpath,imgname))

        if self.transform is not None:
            img0 = self.transform(img0)

        return img0, self.enumdict[folder]

    
    def __len__(self):
        folders = os.listdir(self.imageFolder)
        count =0 
        for i in folders :
            if os.path.isdir(os.path.join(self.imageFolder,i)):
                count += len(os.listdir(os.path.join(self.imageFolder,i)))
        return count