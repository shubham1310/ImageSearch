import torchvision
import torch
from torch.autograd import Variable    
import torch.nn as nn
import torch.nn.functional as F

fina_size = 128

class SiameseNetwork2(nn.Module):
    def __init__(self,pretrain):
        super(SiameseNetwork2, self).__init__()

        self.vgg = torchvision.models.vgg16(pretrained=bool(pretrain==1)) #
        self.vgg.features = nn.Sequential(*(self.vgg.features[i] for i in range(31)))

        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, fina_size))

    def forward(self, x):
        output = self.vgg.features(x);# print(x.size())
        output = output.view(output.size()[0], -1);# print(output.size())
        output = self.fc(output)
        return output


class DotProduct(torch.nn.Module):

    def __init__(self,losstype):
        super(DotProduct, self).__init__()
        self.losstype =losstype

    def forward(self, output1, output2, label):
        output1 =  F.normalize(output1)
        output2 =  F.normalize(output2)
        if self.losstype==1:
            logis_criterion = nn.MSELoss().cuda()
            y = Variable(torch.Tensor([1]).float()).cuda()
        else:
            logis_criterion = nn.BCELoss().cuda()
            y = Variable(torch.Tensor([0.999999]).float()).cuda()
            
        dot = torch.bmm(output1.view(-1, 1, fina_size), output2.view(-1, fina_size, 1))        
        z = Variable(torch.Tensor([0.5]).float()).cuda()
        normdot = dot + y.expand_as(dot)
        finaldot = normdot * z.expand_as(normdot)
        loss = torch.mean( logis_criterion( finaldot ,1-label))
        return loss


class Neuralloss(torch.nn.Module):

    def __init__(self,losstype):
        super(Neuralloss, self).__init__()
        self.losstype =losstype

        self.fc = nn.Sequential(
            nn.Linear(3*fina_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1))

    def forward(self, output1, output2, label):
        output1 =  F.normalize(output1)
        output2 =  F.normalize(output2)
        if self.losstype==1:
            logis_criterion = nn.MSELoss().cuda()
            y = Variable(torch.Tensor([1]).float()).cuda()
        else:
            logis_criterion = nn.BCELoss().cuda()
            y = Variable(torch.Tensor([0.999999]).float()).cuda()
            
        # dot = torch.bmm(output1.view(-1, 1, fina_size), output2.view(-1, fina_size, 1))   
        x = torch.cat((output1,output2, torch.abs( output1-output2)),1)
        pred = self.fc(x)  
        pred = F.sigmoid(pred)
        z = Variable(torch.Tensor([0.5]).float()).cuda()
        normdot = pred + y.expand_as(pred)
        finaldot = normdot * z.expand_as(normdot)
        loss = torch.mean( logis_criterion( finaldot ,label))
        return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
