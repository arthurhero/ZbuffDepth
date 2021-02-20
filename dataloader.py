# Copyright (C) Ziwen Chen, Zixuan Guo
# This file is part of ZbuffDepth <https://github.com/arthurhero/ZbuffDepth>.
#
# ZbuffDepth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZbuffDepth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZbuffDepth.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import random
from torch.utils.data.dataloader import Sampler
from torch.utils.data import Dataset, DataLoader

from data_process import list_folder, integ_sequence, get_stereo_matrices

root_folder = os.path.dirname(os.path.abspath(__file__))


class MySampler(Sampler):
    def __init__(self, data, cur_step=0, batch_size=1, seed=0):
        self.seed = seed
        self.seq = list(range(len(data)))
        random.seed(seed)
        random.shuffle(self.seq)
        self.seq = self.seq[cur_step * batch_size:]

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

class StereoKitti(Dataset):
    def __init__(self, dataset_path, ih, iw):
        '''
        Generate eigen training stereo pairs
        :param dataset_path:
        :param ih:
        :param iw:
        '''
        self.dataset_path = dataset_path
        self.left_img_lst = []
        self.right_img_lst = []
        self.ih = ih
        self.iw = iw

        self.img_path = os.path.join(root_folder, 'eigen_train_files.txt')

        with open(self.img_path) as file:
            for line in file.readlines():
                line  = line.strip()
                left,right = line.split(' ')
                self.left_img_lst.append(left[:-3]+'png')
                self.right_img_lst.append(right[:-3]+'png')

    def __len__(self):
        return len(self.left_img_lst)

    def __getitem__(self, index):
        image_trans = transforms.Compose([
            transforms.ColorJitter(brightness=(0.8, 1.2),
                                   contrast=(0.8, 1.2),
                                   saturation=(0.8, 1.2),
                                   hue=(-0.1, 0.1)),
            transforms.ToTensor()
        ])
        image_trans_ = transforms.Compose([
            transforms.ToTensor()
        ])

        left = self.left_img_lst[index]
        right = self.right_img_lst[index]

        left_orig = Image.open(os.path.join(self.dataset_path,left))
        right_orig = Image.open(os.path.join(self.dataset_path,right))
        left_img = image_trans(left_orig)
        right_img = image_trans(right_orig)
        left_orig = image_trans_(left_orig)
        right_orig = image_trans_(right_orig)

        left_img = left_img[:, 0:self.ih, 0:self.iw]
        left_orig = left_orig[:, 0:self.ih, 0:self.iw]
        right_img = right_img[:, 0:self.ih, 0:self.iw]
        right_orig = right_orig[:, 0:self.ih, 0:self.iw]

        vc,proj_left,proj_right = get_stereo_matrices(left,self.dataset_path)
        vc = torch.from_numpy(vc).float()
        proj_left = torch.from_numpy(proj_left).float()
        proj_right= torch.from_numpy(proj_right).float()

        return left_orig, right_orig, left_img, right_img, vc, proj_left, proj_right

class KITTIDataset(Dataset):
    def __init__(self, dataset_path, ih=None, iw=None, mode='train'):
        '''
        A general KITTI dataloader given image and depth sequences
        :param dataset_path:
        :param ih:
        :param iw:
        :param mode: train, valid, or test
        '''
        self.img_lst = []
        self.depth_lst = []
        self.ih = ih
        self.iw = iw
        self.mode = mode

        if mode == 'train':
            self.img_path = os.path.join(root_folder, 'data/image_sequence.txt')
            self.depth_path = os.path.join(root_folder, 'data/depth_sequence.txt')
        elif mode == 'test':
            self.img_path = os.path.join(root_folder, 'data/image_sequence_test.txt')
            self.depth_path = os.path.join(root_folder, 'data/depth_sequence_test.txt')
        else:
            self.img_path = os.path.join(root_folder, 'data/image_sequence_valid.txt')
            self.depth_path = os.path.join(root_folder, 'data/depth_sequence_valid.txt')

        with open(self.img_path) as file:
            for line in file.readlines():
                self.img_lst.append(os.path.join(dataset_path, line.strip()))
        with open(self.depth_path) as file:
            for line in file.readlines():
                self.depth_lst.append(os.path.join(dataset_path, line.strip()))

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, index):
        ih = self.ih
        iw = self.iw
        if self.mode == 'train':
            image_trans = transforms.Compose([
                transforms.Resize((ih, iw)),
                transforms.ColorJitter(brightness=(0.8, 1.2),
                                       contrast=(0.8, 1.2),
                                       saturation=(0.8, 1.2),
                                       hue=(-0.1, 0.1)),
                transforms.ToTensor()
            ])
        else:
            # transforms.Resize((ih, iw)),
            image_trans = transforms.Compose([
                transforms.ToTensor()
            ])

        image = Image.open(self.img_lst[index])
        image = image_trans(image)

        depth = cv2.imread(self.depth_lst[index], -1)
        if self.mode == 'train':
            depth = cv2.resize(depth, dsize=(iw, ih), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(float)
        depth /= 256.0

        depth = torch.from_numpy(depth)

        return image, depth
