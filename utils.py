from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from PIL import Image
import torch
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import argparse

def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class PairDataset(Dataset):
    
    def __init__(self,imageFolder,transform):
        self.imageFolder = imageFolder 
        self.transform = transform
        
    def __getitem__(self,index):
        folders = os.listdir(self.imageFolder)
        folder = random.choice(folders)
        while not(os.path.isdir(self.imageFolder + folder)) and folder[:5] == 'other':
            folder = random.choice(folders)
        folderpath = os.path.join(self.imageFolder,folder)
        imgname = random.choice(os.listdir(folderpath))
        img0 = self.transform(Image.open(os.path.join(folderpath,imgname)))
        # while not(imgname[-1] =='g') :
        #     imgname = random.choice(os.listdir(folderpath))
        #     img0 = self.transform(Image.open(os.path.join(folderpath,imgname)))
            # a,_,_=img0.size()
            # print(a)

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
        img1 = self.transform(Image.open(os.path.join(folderpath2,imgname)))

        # while not(imgname[-1] =='g'):
        #     imgname = random.choice(os.listdir(folderpath2))
        #     img1 = self.transform(Image.open(os.path.join(folderpath2,imgname)))
        # print(img1.size())
        # print(folderpath2,imgname)

        return img0, img1 , torch.from_numpy(np.array([int(folder!=folder2)],dtype=np.float32))

    
    def __len__(self):
        folders = os.listdir(self.imageFolder)
        count =[]
        for i in folders :
            count.append(len(os.listdir(os.path.join(self.imageFolder,i))))
        s=0
        cs= sum(count)
        for i in count:
            s+= i*(cs-i)
        return s/1000



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