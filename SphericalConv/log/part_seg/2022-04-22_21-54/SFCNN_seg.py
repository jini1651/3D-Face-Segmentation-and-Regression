
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SFCNN_utils import *
# from models.SFCNN_utils import EncodingLayer


class get_model(nn.Module):
    def __init__(self, num_class, SF_list, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 1
        self.normal_channel = normal_channel
        self.SF1 = SF_list[0]
        self.SF2 = SF_list[1]
        self.SF3 = SF_list[2]

        self.layer0 = EncodingLayer(self.SF3, in_channel, [32, 32, 64]) 
        self.layer1 = EncodingLayer(self.SF3, 64, [64, 64, 128])   
        self.layer2 = EncodingLayer(self.SF2, 128, [128, 128, 256])   
        self.layer3 = EncodingLayer(self.SF1, 256, [256, 512, 1024])   

        self.smp1 = SphericalMaxPooling(self.SF3, self.SF2)
        self.smp2 = SphericalMaxPooling(self.SF2, self.SF1)

        self.layer4 = DecodingLayer(self.SF2, 1024+256, [256, 256]) 
        self.layer5 = DecodingLayer(self.SF3, 256+128, [256, 128])   

        self.layer6 = LastLayer(self.SF3, 128+16+3, [128, 128]) 

        self.upsampling1 = UpSampling(self.SF1, self.SF2)
        self.upsampling2 = UpSampling(self.SF2, self.SF3)

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, xyz, projected, cls_label):
        B, _, N = xyz.shape      #Bx1x642
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        #     norm = None
        # print(xyz.shape)

        _, l0_feature = self.layer0(projected)       #Bx64x642
        # SphericalMaxPooling(l0_feature, self.SF3, self.SF3)
        print(l0_feature.shape)

        _, l1_feature = self.layer1(l0_feature)    #Bx128x642
        l1_out = self.smp1(l1_feature)     #Bx128x162
        print(l1_feature.shape)
        
        _, l2_feature = self.layer2(l1_out)     #Bx256x162
        l2_out = self.smp2(l2_feature)     #Bx256x42
        print(l2_feature.shape)
        
        _, l3_feature = self.layer3(l2_out)     #Bx1024x42
        print(l3_feature.shape)

        l3_feature = self.upsampling1(l3_feature)       #Bx1024x162
        l2_feature = torch.cat((l3_feature, l2_feature), 1)     #Bx(1024+256)x162
        l2_feature = self.layer4(l2_feature)            #Bx256x162
        print(l2_feature.shape)

        l2_feature = self.upsampling2(l2_feature)       #Bx256x642
        l1_feature = torch.cat((l2_feature, l1_feature), 1)     #Bx(256+128)x642
        l1_feature = self.layer5(l1_feature)            #Bx128x642
        print(l1_feature.shape)

        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        # l0_feature = torch.cat([cls_label_one_hot,l1_feature],1)

        l0_feature = self.layer6(torch.cat([cls_label_one_hot, xyz],1), l1_feature)
        print(l0_feature.shape)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_feature)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        print(x.shape)

        # # print(out_feature.shape)
        # x = out_feature.view(B, -1)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))
        # x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        return x, l3_feature


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss