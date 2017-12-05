import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

import argparse

from models import SiameseNetwork2, DotProduct #Deconv,
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--netG', type=str, default='', help="path to netG (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass


training_dir = "./datadiv/training/"
testing_dir = "./datadiv/testing/"


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
        if not_same_class < 0.8:
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

        return img0, img1 , torch.from_numpy(np.array([int(folder==folder2)],dtype=np.float32))

    
    def __len__(self):
        folders = os.listdir(self.imageFolder)
        count =0 
        for i in folders :
            count += len(os.listdir(os.path.join(self.imageFolder,i)))
        return count

        # count=[]
        # countother=[]
        # for i in folders :
        #     if os.path.isdir(os.path.join(self.imageFolder,i)) and not(i[:5] == 'other'):
        #         count.append(len(os.listdir(os.path.join(self.imageFolder,i))))
        #     elif os.path.isdir(os.path.join(self.imageFolder,i)):
        #         countother.append(len(os.listdir(os.path.join(self.imageFolder,i))))
        # val=0
        # couns= sum(count)
        # ocouns = sum(countother)
        # for i in count:
        #     val += i * (couns-i + ocouns) + i * (i-1)/2
        # return val

configure('logs/genimage-' + str(opt.out), flush_secs=5)

transform =transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
                              ])



convnet = SiameseNetwork2().cuda()
criterion = DotProduct().cuda()
optimizer = optim.Adam(convnet.parameters(),lr = 0.0005 )

if opt.netG != '':
    convnet.load_state_dict(torch.load(opt.netG))

# singledata = dset.ImageFolder(root=training_dir,transform=transform)
# dataloader = DataLoader(singledata, batch_size=train_batch_size, shuffle=True, num_workers=8)
# l2criterion = nn.MSELoss().cuda()
# deconvnet = Deconv().cuda()
# decoptimizer = optim.Adam(deconvnet.parameters(),lr = 0.0005 )
# for epoch in range(5):
#     for i, data in enumerate(dataloader):
#         inputs, _ = data
#         inputs = Variable(inputs).cuda()

#         convnet.zero_grad()
#         deconvnet.zero_grad()
#         # convout = convnet(inputs)
#         # deconvout = deconvnet(convout)
#         # loss = l2criterion(deconvout,inputs)
#         # loss.backward()
#         [convout,_] = convnet(inputs,inputs)
#         deconvout = deconvnet(convout)
#         loss = l2criterion(deconvout,inputs)
#         loss.backward()
#         optimizer.step()
#         decoptimizer.step()

#         if i %100 == 0 :
#             print("[%d/%d][%d/%d] Loss: %.4f"%(epoch, train_number_epochs,i,len(dataloader) ,loss.data[0]))
#     torch.save(convnet.state_dict(), '%s/pre_netconv%d.pth' % ('./savemodel/', epoch))
#     torch.save(deconvnet.state_dict(), '%s/netdconv%d.pth' % ('./savemodel/', epoch))



siamese_dataset = PairDataset(imageFolder=training_dir,
                            transform=transform)

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=opt.batchsize)
optimizer = optim.Adam(convnet.parameters(),lr = 0.0005 )

iteration_number= 0

for epoch in range(0,opt.nEpochs):
    for i, data in enumerate(train_dataloader,0):
        img0, img1 , label = data
        img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
        output1,output2 = convnet(img0,img1)
        convnet.zero_grad()
        loss= criterion(output1,output2,label)
        loss.backward()
        optimizer.step()
        if i %100 == 0 :
            print("[%d/%d][%d/%d] Main Loss: %.4f"%(epoch, opt.nEpochs,i,len(train_dataloader) ,loss.data[0]))
            iteration_number +=1
            log_value('Netloss', loss.data[0], iteration_number)
    torch.save(convnet.state_dict(), '%s/netconv%d.pth' % (opt.out, epoch))

