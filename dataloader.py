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


class StaticKitti(Dataset):
    def __init__(self, dataset_path, subseq_file, seq_len, skip, ih, iw, crop=True, separate=True):
        '''
        Generate static video segments with egomotion
        '''
        self.path = dataset_path
        subseq_path = os.path.join(root_folder, subseq_file)
        f = open(subseq_path, 'r')
        seqs = f.read().splitlines()
        f.close()
        entries = list()
        lengths = list()
        self.seq_len = (seq_len - 1) * (skip) + seq_len
        for entry in seqs:
            s, start, end = entry.split()
            start, end = int(start), int(end)
            if end - start + 1 < self.seq_len:
                continue
            entries.append((s, start, end))
            lengths.append(end - start + 1)
        self.seqs = entries
        self.lengths = lengths
        self.real_len = seq_len
        self.skip = skip
        self.ih = ih
        self.iw = iw
        self.crop = crop
        self.separate = separate

    def __len__(self):
        total = 0
        for l in self.lengths:
            total += (l - self.seq_len + 1)
        return total

    def __getitem__(self, idx):
        seq_idx = 0
        for i in range(len(self.lengths)):
            l = self.lengths[i]
            if idx < l - self.seq_len + 1:
                seq_idx = i
                break
            else:
                idx -= (l - self.seq_len + 1)
        seq, start, end = self.seqs[seq_idx]
        start += idx
        end = start + self.seq_len - 1
        if self.separate:
            img, ego, proj, vc = integ_sequence(seq, start, end, self.path, separate=self.separate)
            img, ego, proj, vc = torch.from_numpy(img).float(), torch.from_numpy(ego).float(), torch.from_numpy(
                proj).float(), torch.from_numpy(vc).float()
        else:
            img, ego, proj = integ_sequence(seq, start, end, self.path, separate=self.separate)
            img, ego, proj = torch.from_numpy(img).float(), torch.from_numpy(ego).float(), torch.from_numpy(
                proj).float()
            vc = proj.clone()
        _, orig_h, orig_w, _ = img.shape
        imgs = list()
        imgs_orig = list()
        egos = list()
        image_trans = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((self.ih, self.iw)),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                   saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
            transforms.ToTensor()])
        for i in range(self.real_len):
            image = img[i * (self.skip + 1)]
            image = image.permute(2, 0, 1)
            if self.crop:
                image = image[:, 0:self.ih, 0:self.iw]
            else:
                image = F.interpolate(image.unsqueeze(0), (self.ih, self.iw), mode='bilinear', align_corners=False)
            imgs_orig.append(image)
            image = image_trans(image.squeeze())
            imgs.append(image.unsqueeze(0))
            if i != self.real_len - 1:
                cur_ego = ego[i * (self.skip + 1)]
                for j in range(self.skip):
                    cur_ego = torch.matmul(ego[i * (self.skip + 1) + j + 1], cur_ego)
                egos.append(cur_ego.unsqueeze(0))
        img = torch.cat(imgs, dim=0)
        img_orig = torch.cat(imgs_orig, dim=0)

        ego = torch.cat(egos, dim=0)

        if self.crop:
            return (img, img_orig, ego, proj, vc, float(self.ih), float(self.iw))
        else:
            return (img, img_orig, ego, proj, vc, float(orig_h), float(orig_w))


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
