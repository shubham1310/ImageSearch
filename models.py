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

    # def forward(self, input1, input2):
    #     output1 = self.forward_once(input1)
    #     output2 = self.forward_once(input2)
    #     return output1, output2


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
        x = torch.cat((output1,output2,output1-output2),1)
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

# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(p=.2),
            
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.Dropout2d(p=.2),
#             nn.MaxPool2d((2,2), stride=(2,2)),


#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.Dropout2d(p=.2),

#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.Dropout2d(p=.2),
#             nn.MaxPool2d((2,2), stride=(2,2)),


#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),
#             nn.Dropout2d(p=.2),

#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),
#             nn.Dropout2d(p=.2),
#             nn.MaxPool2d((2,2), stride=(2,2)),


#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512),
#             nn.Dropout2d(p=.2),

#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512),
#             nn.Dropout2d(p=.2),
#             nn.MaxPool2d((2,2), stride=(2,2)),


#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512),
#             nn.Dropout2d(p=.2),

#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512),
#             nn.Dropout2d(p=.2),
#             nn.MaxPool2d((2,2), stride=(2,2)),
            
#         )


#         self.fc = nn.Sequential(
#             nn.Linear(512*7*7, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 256),
#             nn.ReLU(inplace=True),

#             nn.Linear(256, 128))

#     def forward_once(self, x):

#         output = self.cnn(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         return output

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         return output1, output2





# class Deconv(nn.Module):
#     def __init__(self):
#         super(Deconv, self).__init__()

#         self.dconv1 = nn.ConvTranspose2d(128, 64, 3, stride=3,output_padding=1)
#         self.conv1 = nn.Sequential( 
#                         nn.Conv2d(64, 64, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(64),
#                         nn.Dropout2d(p=.2))
#         self.conv2 =   nn.Sequential(   
#                         nn.Conv2d(64, 32, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(32),
#                         nn.Dropout2d(p=.2))

#         self.dconv2 =  nn.ConvTranspose2d(32, 32, 3, stride=3, padding=1)

#         self.conv3 = nn.Sequential(
#                         nn.Conv2d(32, 16, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(16),
#                         nn.Dropout2d(p=.2),
#                         )
#         self.conv4 = nn.Sequential(
#                         nn.Conv2d(16, 16, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(16),
#                         nn.Dropout2d(p=.2),
#                         )

#         self.dconv3 =  nn.ConvTranspose2d(16, 16, 4,stride=3)

#         self.conv5 =  nn.Sequential(
#                         nn.Conv2d(16, 8, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(8),
#                         nn.Dropout2d(p=.2),
#                         )
#         self.conv6 =  nn.Sequential(
#                         nn.Conv2d(8, 8, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(8),
#                         nn.Dropout2d(p=.2),
#                         )

#         self.dconv4 = nn.ConvTranspose2d(8, 8, 3, stride=4, padding = 2)

#         self.conv7 = nn.Sequential(
#                         nn.Conv2d(8, 4, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(4),
#                         nn.Dropout2d(p=.2))
#         self.conv8 = nn.Sequential(
#                         nn.Conv2d(4, 4, 3, padding=1),
#                         nn.ReLU(inplace=True),
#                         nn.BatchNorm2d(4),
#                         nn.Dropout2d(p=.2))

#         self.dconv5  = nn.ConvTranspose2d(4, 4, 3, stride=2, padding =6,output_padding=1)

#         self.conv9 = nn.Sequential(

#             nn.Conv2d(4, 3, 3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(3),
#             nn.Dropout2d(p=.2),
#         )

#         self.conv10 = nn.Sequential(

#             nn.Conv2d(3, 3, 3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(3),
#             nn.Dropout2d(p=.2),
#         )

#     def forward(self, input):
#         # print(input.size())
#         output = input.view(input.size()[0],128,1,1);#print(output.size())
#         output = self.dconv1(output);#print(output.size())
#         output = self.conv1(output);#print(output.size())
#         output = self.conv2(output);#print(output.size())

#         output = self.dconv2(output);#print(output.size())
#         output = self.conv3(output);#print(output.size())
#         output = self.conv4(output);#print(output.size())

#         output = self.dconv3(output);#print(output.size())
#         output = self.conv5(output);#print(output.size())
#         output = self.conv6(output);#print(output.size())

#         output = self.dconv4(output);#print(output.size())
#         output = self.conv7(output);#print(output.size())
#         output = self.conv8(output);#print(output.size())

#         output = self.dconv5(output);#print(output.size())
#         output = self.conv9(output);#print(output.size())
#         output = self.conv10(output);#print(output.size())
#         return output


