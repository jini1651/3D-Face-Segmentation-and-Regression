import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import math

def calcDistFromCenter(xyz):       #Calc Projection Value
    # B, N, _ = xyz.shape
    depth = np.sqrt(np.sum(xyz**2, axis=2))

    return depth

def calcSphericalCoordinate(xyz):
    # B, N, _ = xyz.shape

    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]

    d_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan(d_xy / z)
    pi = np.arctan(x / y)

    return theta, pi

def cvtCoord(r, theta, pi):
    z = r * np.cos(pi)
    y = r * np.sin(theta) * np.cos(pi)
    x = r * np.sin(pi) * np.cos(theta)

    return x, y, z

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

def discreteToFractal(projected_pc, SF):
    vertex, near_vertex, triangles = SF
    nvertex, _ = vertex.shape
    npoints, _ = projected_pc.shape

    # vertex = SF[:, :3]
    # near_vertex = SF[:, 3:]
    xyz = projected_pc[:, :3]
    feature = projected_pc[:, 3]
    new_features = np.zeros((nvertex, 1))

    near_point_idx = int(near_vertex[0, 0])
    p1 = vertex[0]
    p2 = vertex[near_point_idx]
    threshold = np.sqrt(np.sum((p1-p2)**2))

    for i in range(nvertex):
        features = []
        
        for j in range(npoints):
            # if i==j:
            #     pass
            d = np.sqrt(np.sum((vertex[i]-xyz[j])**2))

            if d<=threshold:
                features.append(feature[j])
        
        if len(features)==0:
            new_features[i] = new_features[i-1]
        else:
            new_features[i] = sum(features)/len(features)
        
    new_pc = np.concatenate((vertex, new_features), axis=1)

    return new_pc

def PCtoSF(pc, SF):   #mapping Point Cloud to Spherical Fractal
    B, N, C = pc.shape

    xyz = pc[:, :, :3]
    # features = pc[:, :, 3:]
    fractal_vertex, near_vertex, triangles = SF

    new_pc = np.zeros((B, fractal_vertex.shape[0], 3+1))

    depth = calcDistFromCenter(xyz)
    projected_xyz = SphericalProjection(xyz, fractal_vertex)
    # print(projected_xyz.shape, depth.shape)

    depth = np.reshape(depth, (B, N, 1))
    projected_pc = np.concatenate((projected_xyz, depth), axis=2) #BxNx(3+1+len(features))

    for i in range(B):
        new_pc[i] = discreteToFractal(projected_pc[i], SF)     #Nx(3+1)

    return new_pc           #BxNx(3+1)


    

def cvtIcosahedron(in_path, out_path):      #Icosahedron .obj 읽어서 nearest point 6개 찾아서 파일로 저장
    textured_mesh = o3d.io.read_triangle_mesh(in_path)
    vertices = np.asarray(textured_mesh.vertices)
    triangle = np.asarray(textured_mesh.triangles)

    edge = [[] for i in range(len(vertices))]

    for t in triangle:
        for idx in t:
            for i in t:
                if idx!=i and i not in edge[idx]:
                    edge[idx].append(i)

    with open(out_path, 'w') as f:
        for v in vertices:
            f.write('v')
            for p in v:
                f.write(' %f' % p)
            f.write("\n")

        for t in triangle:
            f.write('f')
            for idx in t:
                f.write(' %d' % idx)
            f.write("\n")

        for nnp in edge:
            f.write('np')
            for p in nnp:
                f.write(' %d' % int(p))
            f.write("\n")

def readIcosahedron(file_path, n_vertices):       #정리해놓은 Icosahedron 파일 읽어서 SF return
    vertices = np.zeros((n_vertices, 3))
    near_idx = np.zeros((n_vertices, 6)) - 1
    faces = np.zeros(((n_vertices-2)*2, 3))

    cnt_v = 0
    cnt_n = 0
    cnt_f = 0   

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split()

            if temp[0] == 'v':
                vertices[cnt_v, 0] = temp[1]
                vertices[cnt_v, 1] = temp[2]
                vertices[cnt_v, 2] = temp[3]
                cnt_v += 1

            if temp[0] == 'f':
                faces[cnt_f, 0] = temp[1]
                faces[cnt_f, 1] = temp[2]
                faces[cnt_f, 2] = temp[3]
                cnt_f += 1

            if temp[0] == 'np':
                near_idx[cnt_n, 0] = temp[1]
                near_idx[cnt_n, 1] = temp[2]
                near_idx[cnt_n, 2] = temp[3]
                near_idx[cnt_n, 3] = temp[4]
                near_idx[cnt_n, 4] = temp[5]
                if len(temp)>=7:
                    near_idx[cnt_n, 5] = temp[6]

                cnt_n += 1

    return vertices, near_idx, faces
