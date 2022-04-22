
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SFCNN_utils import EncodingLayer, SphericalMaxPooling
# from models.SFCNN_utils import EncodingLayer


class get_model(nn.Module):
    def __init__(self, num_class, SF_list, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 1
        self.normal_channel = normal_channel
        self.SF1 = SF_list[0]
        self.SF2 = SF_list[1]
        self.SF3 = SF_list[2]

        self.layer0 = EncodingLayer(self.SF3, in_channel, [16, 16]) 
        self.layer1 = EncodingLayer(self.SF3, 16, [32, 32])   
        self.layer2 = EncodingLayer(self.SF2, 32, [64, 64])   
        self.layer3 = EncodingLayer(self.SF1, 64, [128, 128])   

        self.smp1 = SphericalMaxPooling(self.SF3, self.SF2)
        self.smp2 = SphericalMaxPooling(self.SF2, self.SF1)

        self.fc1 = nn.Linear(480, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.8)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape      #Bx1x642
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        #     norm = None
        # print(xyz.shape)

        

        l0_inner, l0_outer, l0_feature = self.layer0(xyz)       #Bx16x642
        # SphericalMaxPooling(l0_feature, self.SF3, self.SF3)
        # print(l0_inner.shape, l0_feature.shape)

        l1_inner, l1_outer, l1_feature = self.layer1(l0_feature)    #Bx32x642
        l1_out = self.smp1(l1_feature)     #Bx32x162
        # print(l1_inner.shape, l1_out.shape)
        
        l2_inner, l2_outer, l2_feature = self.layer2(l1_out)     #Bx64x162
        l2_out = self.smp2(l2_feature)     #Bx64x42
        # print(l2_inner.shape, l2_out.shape)
        
        l3_inner, l3_outer, l3_feature = self.layer3(l2_out)     #Bx128x42
        # print(l3_inner.shape, l3_feature.shape)

        # l0_inner = torch.max(l0_inner, 2)[0]
        # l0_feature = torch.max(l0_feature, 2)[0]
        # l1_inner = torch.max(l1_inner, 2)[0]
        # l1_feature = torch.max(l1_feature, 2)[0]
        # l2_inner = torch.max(l2_inner, 2)[0]
        # l2_feature = torch.max(l2_feature, 2)[0]
        # l3_inner = torch.max(l3_inner, 2)[0]
        # l3_feature = torch.max(l3_feature, 2)[0]

        out_feature = torch.cat((l0_inner, l0_outer, l1_inner, l1_outer, l2_inner, l2_outer, l3_inner, l3_outer), 1)

        # print(out_feature.shape)
        x = out_feature.view(B, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_feature


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss