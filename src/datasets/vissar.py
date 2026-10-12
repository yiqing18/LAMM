from os import path as osp
from typing import Dict
from unicodedata import name

import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
from src.utils.dataset import read_vissar_gray
import scipy.io as scio
import os
import cv2
import random
import torch.nn.functional as F


def norm(points_int, width, height):

	points_int_clone = torch.from_numpy(points_int).detach().float()
	x = ((points_int_clone * 2)[..., 0] / (width - 1) - 1)
	y = ((points_int_clone * 2)[..., 1] / (height - 1) - 1)
	return torch.stack([x, y], dim=-1).contiguous().view(-1, 2)


class VISSARDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 npz_name,
                 pair_dir,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 **kwargs):

        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.npz_path = npz_path

        self.data_names = os.listdir(osp.join(self.root_dir,'VIS').replace('\\', '/'))

        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)
    
    
    def random_perspective_transform(self, height, width, max_trans=100):
    
        src_points = np.array([
            [np.random.randint(0, max_trans), np.random.randint(0, max_trans)],
            [width - np.random.randint(0, max_trans), np.random.randint(0, max_trans)],
            [np.random.randint(0, max_trans), height - np.random.randint(0, max_trans)],
            [width - np.random.randint(0, max_trans), height - np.random.randint(0, max_trans)]
        ], dtype=np.float32)

        
        dst_points = np.array([
            [np.random.randint(0, max_trans), np.random.randint(0, max_trans)],
            [width - np.random.randint(0, max_trans), np.random.randint(0, max_trans)],
            [np.random.randint(0, max_trans), height - np.random.randint(0, max_trans)],
            [width - np.random.randint(0, max_trans), height - np.random.randint(0, max_trans)]
        ], dtype=np.float32)

        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        return perspective_matrix,src_points,dst_points



    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        name = data_name.split('.')[0]

        img_name0 = osp.join(self.root_dir, 'VIS', data_name)

        img_name1 = osp.join(self.root_dir, 'SIM', data_name)


        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = read_vissar_gray(img_name0, resize=(640, 480), augment_fn=None)
                                
        image1 = read_vissar_gray(img_name1, resize=(640, 480), augment_fn=None)

        if self.mode in ['train']:
            T_name = random.randint(1, 1624)
           
            try:
                T = scio.loadmat(osp.join(self.npz_path, f'{T_name}.21.mat'))
                T_0to1 = T['H']
                T_1to0 = np.linalg.inv(T_0to1)
                image1 = cv2.warpPerspective(image1, T_0to1, [image1.shape[1], image1.shape[0]])
                T_name = '21'
            except:
                T = scio.loadmat(osp.join(self.npz_path, f'{T_name}.12.mat'))
                T_1to0 = T['H']
                image0 = cv2.warpPerspective(image0, T_1to0, [image0.shape[1], image0.shape[0]])
                T_0to1 = np.linalg.inv(T_1to0)
                T_name='12'

        else:
            try:
                T = scio.loadmat(osp.join(self.npz_path, f'{name}.21.mat'))
                T_0to1 = T['H']
                T_1to0 = np.linalg.inv(T_0to1)
                T_name = '21'
            except:
                T = scio.loadmat(osp.join(self.npz_path, f'{name}.12.mat'))
                T_1to0 = T['H']
                T_0to1 = np.linalg.inv(T_1to0)
                T_name='12'
            

        image0 = torch.from_numpy(image0).float()[None] / 255
        image1 = torch.from_numpy(image1).float()[None] / 255
        data = {
            'image0': image0,   
            'image1': image1,
            'T_0to1': T_0to1,   
            'T_1to0': T_1to0,
            'dataset_name': 'VISSAR',
            'pair_id': idx,
            'T_name': T_name,
            'mode': self.mode,
            'pair_names': (osp.join(f'{data_name}'),
                           osp.join(f'{data_name}'))
        }


        return data
