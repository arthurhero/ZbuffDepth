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
import copy

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from data_process import *
from dataloader import KITTIDataset 
from dataloader import MySampler

from ops import *
from loss import *
from utils import *
import argparse
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(dataset_path, eval_input,model=None,mode='test'):
    dataset = KITTIDataset(dataset_path, mode=mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    if isinstance(eval_input, str):
        if model == 'orig':
            from bts_orig import BtsModel
        elif model == 'neighbor':
            from bts_neighbor import BtsModel
        elif model == 'mask':
            from maskbts import BtsModel
        else:
            import sys
            sys.exit('Unknown model type. Choose either orig, neighbor or mask.')
        eval_path = eval_input
        egodepth = BtsModel().to(device)
        egodepth.eval()
        if os.path.isfile(eval_path):
            load_ckpt(eval_path, egodepth, eval_mode=True)
            print("loaded checkpoint!")
        else:
            print("no checkpoint!")
            return
    else:
        egodepth = eval_input
        egodepth.eval()
    results = torch.zeros((7,)).to(device)
    count = 0.0
    with torch.no_grad():
        for i, (img_batch, gt_batch) in enumerate(dataloader):
            '''
            img_batch - B x 3 x h x w
            gt_batch - B x 1 x h x w
            '''
            img_batch, gt_batch = img_batch.to(device), gt_batch.to(device)
            b, _, h, w = img_batch.shape
            depth_batch = egodepth(img_batch)  # b x 1 x h x w
            # do eigen crop
            h_low = int(0.3324324 * h)
            h_high = int(0.91351351 * h)
            w_low = int(0.0359477 * w)
            w_high = int(0.96405229 * w)
            depth_batch = depth_batch[:, :, h_low:h_high, w_low:w_high]
            gt_batch = gt_batch[:, h_low:h_high, w_low:w_high]
            depth = depth_batch.reshape(-1)
            gt = gt_batch.reshape(-1)
            mask = (gt <= 80).float() * (gt > 0).float()
            idx = mask.nonzero().view(-1)
            depth = depth.index_select(0, idx)
            gt = gt.index_select(0, idx)
            results += evaluation_errors(depth, gt)
            count += 1
    results /= count
    print("abs_rel:", results[0].item())
    print("sq_rel:", results[1].item())
    print("rmse:", results[2].item())
    print("rmse_log:", results[3].item())
    print("< 1.25:", results[4].item())
    print("< 1.25**2:", results[5].item())
    print("< 1.25**3:", results[6].item())
    return results[4].item()

