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

from models import SiameseNetwork2, DotProduct, Neuralloss #Deconv,
from tensorboard_logger import configure, log_value

from utils import PairDataset, SingleImage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--netG', type=str, default='', help="path to netG (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
parser.add_argument('--train', type=int, default=1, help='training 1/ testing 0')
parser.add_argument('--losstype', type=int, default=1, help='MSE 1/ BCE 0')
parser.add_argument('--dataset', type=int, default=1, help='all class 0/ oxford class 1')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass


if opt.dataset:
    training_dir = "./newdata/training/"
    testing_dir = "./newdata/testing/"
else:
    training_dir = "./datadiv/training/"
    testing_dir = "./datadiv/testing/"



configure('logs/genimage-' + str(opt.out), flush_secs=5)

transform =transforms.Compose([transforms.Resize((224,224)),
                              transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
                              ])



convnet = SiameseNetwork2().cuda()
criterion = Neuralloss(opt.losstype).cuda()


if opt.netG != '':
    convnet.load_state_dict(torch.load(opt.netG))

# optimizer = optim.Adam(convnet.parameters(),lr = 0.0005 )
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


if opt.train:
    siamese_dataset = PairDataset(imageFolder=training_dir,
                                transform=transform)

    train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=opt.batchsize)
    optimizer = optim.Adam(convnet.parameters(),lr = 0.0005 )
    optimizerloss = optim.Adam(criterion.parameters(),lr = 0.0005 )
    iteration_number= 0

    for epoch in range(0,opt.nEpochs):
        for i, data in enumerate(train_dataloader,0):
            convnet.zero_grad()
            img0, img1 , label = data
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            output1,output2 = convnet(img0),convnet(img1)
            loss= criterion(output1,output2,label)
            loss.backward()
            optimizer.step()

            criterion.zero_grad()
            img0, img1 , label = data
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            output1,output2 = convnet(img0.detach()),convnet(img1.detach())
            lossc= criterion(output1,output2,label)
            lossc.backward()
            optimizerloss.step()

            if i %100 == 0 :
                print("[%d/%d][%d/%d] Main Loss: %.4f"%(epoch, opt.nEpochs,i,len(train_dataloader) ,loss.data[0]))
                iteration_number +=1
                log_value('Netloss', loss.data[0], iteration_number)
        torch.save(convnet.state_dict(), '%s/netconv%d.pth' % (opt.out, epoch))
else:
    folderenum={}
    count=1
    folders = os.listdir(training_dir)
    target_names=['other']
    for folder in folders:
        if os.path.isdir(training_dir + folder) and not(folder[:5] == 'other'):
            folderenum[folder] = count
            count+=1
            target_names.append(folder)
        elif os.path.isdir(training_dir + folder):
            folderenum[folder] = 0
    print(folderenum)
    print('Enumeration done')
    single_dataset = SingleImage(imageFolder=training_dir, enumdict = folderenum, transform=transform)

    train_dataloader = DataLoader(single_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=opt.batchsize)
    images=[]
    labels=[]
    for i, data in enumerate(train_dataloader,0):
        img0, label = data
        img0 = Variable(img0).cuda()
        output = convnet(img0)
        for j in range(len(label)):
            # print(output[j].data.cpu().numpy())
            # print(label[j])
            images.append(output[j].data.cpu().numpy())
            labels.append(label[j])
        if i%100==0:
            print('Data creation done for %d/%d'%(i,len(train_dataloader)))

    print('Image and labels done')
    neigh = KNeighborsClassifier(n_neighbors=count)
    neigh.fit(images, labels)

    print('Nearest neighbours Classifier trained')

    single_dataset = SingleImage(imageFolder=testing_dir, enumdict = folderenum, transform=transform)

    test_dataloader = DataLoader(single_dataset,
                shuffle=True,
                num_workers=8,
                batch_size=opt.batchsize)

    act=[]
    pred=[]
    for i, data in enumerate(test_dataloader,0):
        img0, label = data
        img0= Variable(img0).cuda()
        output = convnet(img0)
        for j in range(len(label)):
            act.append(label[j])
            x=neigh.predict([output[j].data.cpu().numpy()])
            pred.append(x[0])
        if i%100==0:
            print('Prediction done for %d/%d'%(i,len(train_dataloader)))
    print(classification_report(act, pred, target_names=target_names))
    print(accuracy_score(act, pred))



