import torch
import torch.nn as nn
import torch.nn.functional as F

def makeInputData(input_feature, SF):
    #feature에서 근점 점 6개 인덱스 뽑아서 MLP 입력 블럭으로 만들
    #output : B x N x 6 x 2C (기준 점x6 로 늘려서 근접 점이랑 붙)
    '''
    input : B x N x C
    SF : N x (xyz + p1~p6)

    input*6 + p1~p6 (concat) => output

    output : B x N x 6 x 2C
    '''
    #임시로 해놓음
    # input_feature = input_feature.numpy()
    # print(input_feature.shape)
    device = input_feature.device
    B, N, C = input_feature.shape
    vertices, near_idx, triangles = SF         #near_idx : Nx6
    # features = np.reshape(input_feature, (B, N, 1, C))
    # input_feature = input_feature.permute(0, 2, 1)
    features = input_feature.view(B, N, 1, C)
    # features = np.repeat(features, 6, axis=2)   #BxNx6xC
    features = features.repeat(1, 1, 6, 1)    #BxNx6xC

    # outofidx_6 = np.zeros((B, N, C)) - 1
    outofidx_6 = torch.zeros((B, N, C)).to(device) - 1
    # input_feature = np.concatenate((input_feature, outofidx_6), axis=1) #Bx(N+1)xC

    input_feature =  torch.cat((input_feature, outofidx_6), 1)    #Bx(N+1)xC

    near_features = input_feature[:, near_idx, :]   #BxNx6xC

    # feature_block = np.concatenate((features, near_features), axis=-1)  #BxNx6x2C
    feature_block = torch.cat((features, near_features), 3)#.cuda()  #BxNx6x2C

    # feature_block = torch.Tensor(feature_block)

    return feature_block    #BxNx6x2C

def SphericalMaxPooling(input, in_SF, out_SF):
    '''
    input : B x C x N0
    SF0 : (xyz, p01~p05 idx) - (N0 x 3), (N0 x 6)
    SF1 : (xyz, p10~p15 idx) - (N1 x 3), (N1 x 6)

    output : B x N1 x C

    for SF1:
        for SF2:
            SF2-xyz중에 SF1이랑 같은거 찾아서 p01~p06 인덱스 input에서 d 꺼내와서 pooling
            그 값 ouput에 추가
    '''
    B, C, N = input.shape
    input = input.permute(0, 2, 1)

    in_v, in_ni, _ = in_SF
    out_v, out_ni, _ = out_SF

    in_nvertices = in_v.shape[0]
    out_nvertices = out_v.shape[0]

    output = torch.zeros((B, out_nvertices, C))
    
    for i in range(out_nvertices):
        out_xyz = out_v[i]
        for j in range(in_nvertices):
            if out_xyz.all() == in_v[j].all():
                near_idx = in_ni[j]  #Bx6
                temp = input[:, near_idx, :]   #Bx6xC
                output[:, i, :] = torch.max(temp, 1)[0]     #BxC

    input = input.permute(0, 2, 1)

    return output   #BxCxN'


class SphericalMaxPooling(nn.Module):
    def __init__(self, in_SF, out_SF):
        super(SphericalMaxPooling, self).__init__()
        self.in_SF = in_SF
        self.out_SF = out_SF
    '''
    input : B x C x N0
    SF0 : (xyz, p01~p05 idx) - (N0 x 3), (N0 x 6)
    SF1 : (xyz, p10~p15 idx) - (N1 x 3), (N1 x 6)

    output : B x N1 x C

    for SF1:
        for SF2:
            SF2-xyz중에 SF1이랑 같은거 찾아서 p01~p06 인덱스 input에서 d 꺼내와서 pooling
            그 값 ouput에 추가
    '''

    def forward(self, input):
        device = input.device
        B, C, N = input.shape
        input = input.permute(0, 2, 1)

        in_v, in_ni, _ = self.in_SF
        out_v, out_ni, _ = self.out_SF

        in_nvertices = in_v.shape[0]
        out_nvertices = out_v.shape[0]

        output = torch.zeros((B, out_nvertices, C)).to(device)

        for i in range(out_nvertices):
            near_idx = in_ni[i]  #Bx6
            temp = input[:, near_idx, :]   #Bx6xC
            output[:, i, :] = torch.max(temp, 1)[0]     #BxC

        
        # for i in range(out_nvertices):
        #     out_xyz = out_v[i]
        #     for j in range(in_nvertices):
        #         if out_xyz.all() == in_v[j].all():
        #             near_idx = in_ni[j]  #Bx6
        #             temp = input[:, near_idx, :]   #Bx6xC
        #             output[:, i, :] = torch.max(temp, 1)[0]     #BxC

        output = output.permute(0, 2, 1)

        return output   #BxCxN'


class UpSampling(nn.Module):
    def __init__(self, in_SF, out_SF):
        super(UpSampling, self).__init__()
        self.in_SF = in_SF
        self.out_SF = out_SF
    '''
    input -
        data : BxCxN0
        SF0 : (xyz, p01~p05 idx, triangles) - (N0 x 3), (N0 x 6), (len(f)x3)  
        SF1 : (xyz, p10~p15 idx, triangles) - (N1 x 3), (N1 x 6), (len(f)x3)  
            (len(face) = (N-2)*2)

    output
        output : BxCxN1
    '''

    def forward(self, data):
        
        _, _, triangles = self.in_SF
        out_v, _, _ = self.out_SF
        out_n, _ = out_v

        B, C, N = data

        edge = []
        output = np.zeros((B, C, out_n))
        cnt = 0

        for i, t in enumerate(triangles):
            p0, p1, p2 = t

            if (p0, p1) not in edge and (p1, p2) not in edge:
                edge.append((p0, p1))
                output[:, :, N+cnt] = (data[:, :, p0] + data[:, :, p1]) / 2
                cnt += 1

            elif (p1, p2) not in edge and (p2, p1) not in edge:
                edge.append((p1, p2))
                output[:, :, N+cnt] = (data[:, :, p1] + data[:, :, p2]) / 2
                cnt += 1

            elif (p2, p0) not in edge and (p0, p2) not in edge:
                edge.append((p2, p0))
                output[:, :, N+cnt] = (data[:, :, p2] + data[:, :, p0]) / 2
                cnt += 1

        if out_n != cnt-1:
            raise upsamplingError

        return output



class EncodingLayer(nn.Module):
    def __init__(self, SF, in_channel, mlp, residual=False):
        super(EncodingLayer, self).__init__()
        self.SF = SF
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel*2

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))      #왜 2d로 함? - kernel=1
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, points):
        """
        Input:
            points: input points data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C', N]
        """
        # print(points.shape)
        points = points.permute(0, 2, 1)    #BxNxC
        # v, i = self.SF
        # print(v.shape)

        new_feature = makeInputData(points, self.SF)      #BxNx6x2C
        new_feature = new_feature.permute(0, 3, 2, 1)    #Bx2Cx6xN

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature =  F.relu(bn(conv(new_feature)))        #BxC'x6xN

            # if i==0:
            #     inner_feature = new_feature         
        
        new_feature = torch.max(new_feature, 2)[0]           #[B, C', N] - maxpooling of conv (1 kernel)
        # inner_feature = torch.max(inner_feature, 1)[0]

        # inner_feature = torch.max(inner_feature, 2)[0]
        out_feature = torch.max(new_feature, 1)[0]      #[B, N]

        return out_feature, new_feature

class DecodingLayer(nn.Module):
    def __init__(self, SF, in_channel, mlp, residual=False):
        super(DecodingLayer, self).__init__()
        self.SF = SF
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel*2

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))      #왜 2d로 함? - kernel=1
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, points):
        """
        Input:
            points: input points data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C', N]
        """
        # print(points.shape)
        points = points.permute(0, 2, 1)    #BxNxC
        # v, i = self.SF
        # print(v.shape)

        new_feature = makeInputData(points, self.SF)      #BxNx6x2C
        new_feature = new_feature.permute(0, 3, 2, 1)    #Bx2Cx6xN

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature =  F.relu(bn(conv(new_feature)))        #BxC'x6xN

            # if i==0:
            #     inner_feature = new_feature         
        
        new_feature = torch.max(new_feature, 2)[0]           #[B, C', N] - maxpooling of conv (1 kernel)
        # inner_feature = torch.max(inner_feature, 1)[0]

        # inner_feature = torch.max(inner_feature, 2)[0]
        # out_feature = torch.max(new_feature, 1)[0]      #[B, N]


def SphericalProjection(xyz, fractal_vertex):
    # xyz = pc[:, :, :3]
    # features = pc[:, :, 3:]
    B, N, _ = xyz.shape

    theta, pi = calcSphericalCoordinate(xyz)

    r = np.sqrt(np.sum(fractal_vertex[0]**2))          #Nx(3(xyz)+6(인접point))
    
    x, y, z = cvtCoord(r, theta, pi)

    x = np.reshape(x, (B, N, 1))
    y = np.reshape(y, (B, N, 1))
    z = np.reshape(z, (B, N, 1))

    projected_xyz = np.concatenate((x, y, z), axis=2)

    return projected_xyz


def SFtoPC(projected, SF, xyz):   #mapping Projected points to original points
    B, N, C = xyz.shape
    _, N0, C0 = projected.shape

    xyz = xyz[:, :, :3]
    # features = pc[:, :, 3:]
    vertex, near_idx, triangles = SF
    nvertex, _ = vertex.shape

    near_point_idx = int(near_idx[0, 0])
    p1 = vertex[0]
    p2 = vertex[near_point_idx]
    threshold = np.sqrt(np.sum((p1-p2)**2))

    val = np.zeros((B, N, C0))

    projected_xyz = SphericalProjection(xyz, vertex)
    # print(projected_xyz.shape, depth.shape)


    for b in range(B):
        for j in range(N):
            for i in range(nvertex):
                d = np.sqrt(np.sum((vertex[i]-projected_xyz[b, j, :])**2))

                if d<=threshold:
                    val[b, j, :] = projected[b, i, :]
                    break

    # new_pc = np.concatenate((xyz, val), axis=2) #BxNx(3+len(features))

    # for i in range(B):
    #     new_pc[i] = discreteToFractal(projected_pc[i], SF)     #Nx(3+1)

    return val           #BxNx(3+1)


class LastLayer(nn.Module):
    def __init__(self, SF, in_channel, mlp, residual=False):
        super(LastLayer, self).__init__()
        self.SF = SF
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))      #왜 2d로 함? - kernel=1
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, original, projected):
        """
        Input:
            original: output points data templit + class_label, [B, 16+3, N]
            projected : feature, [B, C0, N0]
        Return:
            new_xyz: sampled points position data, [B, C, N]
        """
        # print(points.shape)
        original = original.permute(0, 2, 1)    #BxNxC
        projected = projected.permute(0, 2, 1)    #BxN0xC0

        feature = SFtoPC(projected, self.SF, original)  #BxNxC0

        new_feature = torch.concat((original, feature), 2)      #BxNx(3+16+C0)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature =  F.relu(bn(conv(new_feature)))        

            # if i==0:
            #     inner_feature = new_feature         
        
        # new_feature = torch.max(new_feature, 2)[0]           #[B, C', N] - maxpooling of conv (1 kernel)
        # inner_feature = torch.max(inner_feature, 1)[0]

        # inner_feature = torch.max(inner_feature, 2)[0]
        # out_feature = torch.max(new_feature, 1)[0]      #[B, N]

        return new_feature      #BxNxC
