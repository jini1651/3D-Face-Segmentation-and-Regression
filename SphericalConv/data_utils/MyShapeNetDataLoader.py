# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
from . import projection_utils
import provider
from tqdm import tqdm

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel
        self.SF = projection_utils.readIcosahedron('/home/rmclab102/FaceSegmentation/SphericalConv/data_utils/SF3.txt', 642)                          #chuga


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


        self.save_path = os.path.join(root, 'shapenet_%s_%dpts.dat' % (split, self.npoints))

        if not os.path.exists(self.save_path): 
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_projected = [None] * len(self.datapath)
            self.list_of_cls = [None] * len(self.datapath)
            self.list_of_seg = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                if not self.normal_channel:
                    point_set = data[:, 0:3]
                else:
                    point_set = data[:, 0:6]
                seg = data[:, -1].astype(np.int32)

                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                # resample
                choice = np.random.choice(len(seg), self.npoints, replace=True)
                point_set = point_set[choice, :]
                seg = seg[choice]

                point_set = np.reshape(point_set[:, 0:3], (1, self.npoints, -1))                 #chuga

                point_set[:, :, 0:3] = provider.random_scale_point_cloud(point_set[:, :, 0:3])
                point_set[:, :, 0:3] = provider.shift_point_cloud(point_set[:, :, 0:3])

                projected_set = projection_utils.PCtoSF(point_set, self.SF)                               #chuga
                projected_set = np.reshape(projected_set, (642, -1))
                point_set = np.reshape(point_set, (self.npoints, -1))

                self.list_of_points[index] = point_set
                self.list_of_projected[index] = projected_set
                self.list_of_cls[index] = cls
                self.list_of_seg[index] = seg

            with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_projected, self.list_of_cls, self.list_of_seg], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_projected, self.list_of_cls, self.list_of_seg = pickle.load(f)
                # print(self.list_of_points.shape)


    def __getitem__(self, index):
        if index in self.cache:
            point_set, projected_set, cls, seg = self.cache[index]
        else:
            point_set, projected_set, cls, seg = self.list_of_points[index], self.list_of_projected[index], self.list_of_cls[index], self.list_of_seg[index]
            # point_set = np.reshape(point_set, (-1, 1))
            # fn = self.datapath[index]
            # cat = self.datapath[index][0]
            # cls = self.classes[cat]
            # cls = np.array([cls]).astype(np.int32)
            # data = np.loadtxt(fn[1]).astype(np.float32)
            # if not self.normal_channel:
            #     point_set = data[:, 0:3]
            # else:
            #     point_set = data[:, 0:6]
            # seg = data[:, -1].astype(np.int32)

            # # resample
            # choice = np.random.choice(len(seg), self.npoints, replace=True)
            # point_set = point_set[choice, :]
            # seg = seg[choice]

            # point_set = np.reshape(point_set[:, 0:3], (1, self.npoints, -1))                 #chuga

            # point_set[:, :, 0:3] = provider.random_scale_point_cloud(point_set[:, :, 0:3])
            # point_set[:, :, 0:3] = provider.shift_point_cloud(point_set[:, :, 0:3])

            # projected_set = projection_utils.PCtoSF(point_set, self.SF)                               #chuga
            # projected_set = np.reshape(projected_set, (642, -1))
            # point_set = np.reshape(point_set, (self.npoints, -1))

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, projected_set, cls, seg)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # resample
        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        # point_set = point_set[choice, :]
        # seg = seg[choice]

        return point_set, projected_set, cls, seg

    def __len__(self):
        return len(self.datapath)



