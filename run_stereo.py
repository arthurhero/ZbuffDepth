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
from dataloader import StereoKitti
from dataloader import MySampler

from ops import *
from loss import *
from utils import *
import argparse
from datetime import datetime

from validate import evaluate 

# use gpu
#device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    '''
    parse cmd line args
    '''
    parser = argparse.ArgumentParser(description='Unsupervised depth estimation')

    # choose model
    parser.add_argument('--model', type=str, help='choose which model to use; orig, neighbor or mask', required=True)

    # Log and save
    parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', required=True)
    parser.add_argument('--description', type=str, help='description of the experiment', required=True)
    parser.add_argument('--experiment_directory', type=str, help='directory particular experiment to continue', 
            default=datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    parser.add_argument('--summary_path', type=str, help='path to TensorBoard', default='runs')

    # Dataset
    parser.add_argument('--kitti_path', type=str, help='path to kitti dataset',
                        default='/data/raw/robotics/kitti/raw_sequences-20200224133836/data/')
    parser.add_argument('--shuffle_seed', type=int, help='seed for shuffling the dataset', default=0)

    # Training
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.00001)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--log_rate', type=int, help='number of step to log', default=100)

    # Ablation
    parser.add_argument('--smooth_lambda', type=float, help='lambda of smooth loss', default=0.0)
    parser.add_argument('--ssim_lambda', type=float, help='lambda of ssim loss', default=2.0)
    parser.add_argument('--nd_lambda', type=float, help='lambda of neg dep loss', default=2.0)
    parser.add_argument('--include_nd', help='inlcude points with neg depth', action='store_true')
    parser.add_argument('--zbuffer', help='use zbuffer or not', action='store_true')
    parser.add_argument('--zbuffer_late', type=int, help='delay zbuffer for x epochs', default=10)
    parser.add_argument('--gordon_zbuffer', help='use zbuffer by gordon et al.', action='store_true')
    parser.add_argument('--skymask_distance', type=float, help='the thereshold of skymask', default=100.0)
    parser.add_argument('--sky_only_smooth', help='only smooth things far away', action='store_true')

    # Preprocessing
    parser.add_argument('--image_height', type=int, help='height of input image', default=352)
    parser.add_argument('--image_width', type=int, help='width of input image', default=1216)

    # load arguments
    args = parser.parse_args()
    return args

def lr_lamb(ep,model):
    return 1.0
    '''
    if model=='orig':
        return 1.0
    if ep < 1:
        return 10.0
    if ep < 3:
        return 3.0
    if ep < 20:
        return 1.0
    else:
        return 0.3
        '''

def train(visualize=False):
    print("Using", device)
    args=parse_args()
    print("Args:", args)
    if args.model == 'orig':
        from bts_orig import BtsModel
    else:
        import sys
        sys.exit('Unknown model type. Choose either orig, neighbor or mask.')

    # hyperparameters
    hyper = {'description': args.description,
             'batch_size': args.batch_size,
             'learning_rate': args.learning_rate,
             'num_epochs': args.num_epochs,
             'image_height': args.image_height,
             'image_width': args.image_width}

    writer = SummaryWriter(log_dir=args.summary_path)
    if visualize:
        chunk = 1
    else:
        chunk = args.log_rate
    egodepth = BtsModel().to(device)
    egodepth.train()
    optimizer = torch.optim.Adam(egodepth.parameters(), lr=hyper['learning_rate'], betas=(0.9, 0.999)) # create optimizer with learning_rate from argument
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:lr_lamb(x,args.model), last_epoch=-1)

    cur_epoch, last_step = 0, -1
    best_res, seed = 0, args.shuffle_seed

    experiment_directory = os.path.join(args.log_directory, args.experiment_directory)

    if os.path.isfile(os.path.join(experiment_directory,'current.ckpt')):
        cur_epoch, last_step, seed = load_ckpt(experiment_directory, egodepth, hyper, optimizer, scheduler)
        print("loaded checkpoint!")
        # update learning rate from the checkpoint
        for g in optimizer.param_groups:
            g['lr'] = hyper['learning_rate']
        if os.path.isfile(os.path.join(experiment_directory, 'best.ckpt')):
            best_res = evaluate(args.kitti_path, os.path.join(experiment_directory, 'best.ckpt'),model=args.model,
                    mode='valid')
        else:
            best_res = evaluate(args.kitti_path, copy.deepcopy(egodepth),mode='valid')
    else:
        if not os.path.isdir(experiment_directory):
            os.mkdir(experiment_directory)
        # create human readable description
        with open(os.path.join(experiment_directory, 'DESC'), 'w') as f:
            for _, (key, value) in enumerate(hyper.items()):
                f.write(key + ": " + str(value) + '\n')

    multiple_gpu = torch.cuda.device_count() > 1
    if multiple_gpu:
        egodepth = torch.nn.DataParallel(egodepth)

    kitti_path = args.kitti_path
    if kitti_path[-1] != '/':
        kitti_path+='/'
    num_epochs = hyper['num_epochs']
    if args.num_epochs > num_epochs:
        num_epochs = args.num_epochs
    for e in range(cur_epoch, num_epochs):
        step = 0
        if visualize:
            dataset = StereoKitti(dataset_path=kitti_path, 
                                  ih=hyper['image_height'], iw=hyper['image_width'])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        elif e == cur_epoch and last_step >= 0:
            dataset = StereoKitti(dataset_path=kitti_path,
                                  ih=hyper['image_height'], iw=hyper['image_width'])
            if (last_step + 1) * hyper['batch_size'] >= len(dataset):
                continue
            sampler = MySampler(dataset, last_step + 1, hyper['batch_size'], seed)
            dataloader = DataLoader(dataset, batch_size=hyper['batch_size'], sampler=sampler, shuffle=False,
                                    num_workers=hyper['batch_size'])
            step = last_step + 1
        else:
            dataset = StereoKitti(dataset_path=kitti_path,
                                  ih=hyper['image_height'], iw=hyper['image_width'])
            sampler = MySampler(dataset, 0, hyper['batch_size'], seed)
            dataloader = DataLoader(dataset, batch_size=hyper['batch_size'], sampler=sampler, shuffle=False,
                                    num_workers=hyper['batch_size'])
        l_ssim_sum, l_smooth_sum, l_recon_sum, l_match_sum, l_nd_sum = 0, 0, 0, 0, 0
        for i, (lo_batch, ro_batch, l_batch, r_batch, vc_batch, l_proj, r_proj) in enumerate(dataloader):
            '''
            lo_batch - B x 3 x h x w
            vc_batch - B x 4 x 4
            proj_batch - B x 4 x 4
            '''
            lo_batch, ro_batch, l_batch, r_batch, vc_batch, l_proj, r_proj= \
                    lo_batch.to(device), ro_batch.to(device), l_batch.to(device), \
                    r_batch.to(device), vc_batch.to(device), l_proj.to(device), r_proj.to(device)
            if visualize:
                print("l proj", l_proj[0])
                print("img shape", l_batch.shape)
                visualize_img(l_batch[0])
                visualize_img(lo_batch[0])
                visualize_img(r_batch[0])
                visualize_img(ro_batch[0])
            b, _, h, w = l_batch.shape
            l_depth = egodepth(l_batch)  # b x 1 x h x w
            r_depth = egodepth(r_batch)  # b x 1 x h x w
            l_skymask = (l_depth <= args.skymask_distance).float() # max depth is 100
            r_skymask = (r_depth <= args.skymask_distance).float()
            if visualize:
                print("depth", l_depth[0][0][100:110, 100:110])
                visualize_depth(l_depth[0])
                visualize_depth(l_skymask[0] * 100.0, window='skymask')

            l_pc = depth2pc(l_depth, l_proj, vc_batch, skymask=l_skymask)  # b x 3 x h*w
            r_pc = depth2pc(r_depth, r_proj, vc_batch, skymask=r_skymask)  # b x 3 x h*w

            loss_ssim = l_pc.new((1,)).zero_()
            loss_m = l_pc.new((1,)).zero_()
            loss_r = l_pc.new((1,)).zero_()
            loss_neg_dep = l_pc.new((1,)).zero_()

            if e < args.zbuffer_late:
                zbuf_ablation=True
            else:
                zbuf_ablation = (not(args.zbuffer))
            for j in range(b):
                l_sky = l_skymask[j]
                r_sky = r_skymask[j]
                l_p = l_pc[j]  # 3 x h*w
                r_p = r_pc[j]  # 3 x h*w
                vc = vc_batch[j]

                l_img = lo_batch[j]  # j-th img in the batch 
                r_img = ro_batch[j]  # j-th img in the batch 

                pc_pixel_f, idx_orig, loss_nd = register_pc(l_p,r_proj[j],vc,l_sky, ablation = zbuf_ablation, gordon=args.gordon_zbuffer, target_depth = r_depth[j], include_nd = args.include_nd)
                loss_neg_dep += loss_nd
                if pc_pixel_f is not None:
                    pixels_l = img_sampling(l_img, idx_orig)
                    pixels_r = img_sampling_bilinear(r_img, pc_pixel_f)
                    l_pc_orig = img_sampling(l_pc[j].view(3,h,w), idx_orig)
                    #r2l_pc_trans = img_sampling_bilinear(r_pc[j].view(3,h,w), pc_pixel_f)
                    r2l_pc_trans = img_sampling(r_pc[j].view(3,h,w), pc_pixel_f[1].floor()*w+pc_pixel_f[0].floor())
                    loss_m += match_loss(r2l_pc_trans, l_pc_orig)
                    selected = scatter_pixel(pixels_l, idx_orig, h, w).float()
                    selected2 = scatter_pixel(pixels_r, idx_orig, h, w).float()
                    loss_ssim += ssim(selected.unsqueeze(0), selected2.unsqueeze(0))
                    if visualize:
                        print("idx orig shape:", idx_orig.shape)
                        visualize_img(torch.cat([l_img, r_img], dim=1), window='img to select')
                        visualize_img(torch.cat([selected, selected2], dim=1), window='selected')
                    loss_r += recon_loss(pixels_l, pixels_r)

                pc_pixel_f, idx_orig, loss_nd = register_pc(r_p,l_proj[j],vc,r_sky, ablation = zbuf_ablation, gordon = args.gordon_zbuffer, target_depth = l_depth[j], include_nd = args.include_nd)
                loss_neg_dep += loss_nd
                if pc_pixel_f is not None:
                    pixels_r = img_sampling(r_img, idx_orig)
                    pixels_l = img_sampling_bilinear(l_img, pc_pixel_f)
                    r_pc_orig = img_sampling(r_pc[j].view(3,h,w), idx_orig)
                    #l2r_pc_trans = img_sampling_bilinear(l_pc[j].view(3,h,w), pc_pixel_f)
                    l2r_pc_trans = img_sampling(l_pc[j].view(3,h,w), pc_pixel_f[1].floor()*w+pc_pixel_f[0].floor())
                    loss_m += match_loss(l2r_pc_trans, r_pc_orig)
                    selected = scatter_pixel(pixels_r, idx_orig, h, w).float()
                    selected2 = scatter_pixel(pixels_l, idx_orig, h, w).float()
                    loss_ssim += ssim(selected.unsqueeze(0), selected2.unsqueeze(0))
                    if visualize:
                        print("idx orig shape:", idx_orig.shape)
                        visualize_img(torch.cat([r_img, l_img], dim=1), window='img to select')
                        visualize_img(torch.cat([selected, selected2], dim=1), window='selected')
                    loss_r += recon_loss(pixels_r, pixels_l)

            if args.sky_only_smooth:
                loss_s=smooth_loss(l_depth,l_batch,l_skymask)+smooth_loss(r_depth,r_batch,r_skymask)
            else:
                loss_s=smooth_loss(l_depth,l_batch)+smooth_loss(r_depth,r_batch)
            #loss_s = gradient_smooth_loss(l_depth, l_batch)
            # loss=loss_r+loss_neg_dep
            if visualize:
                print("loss r", loss_r)
                print("loss_nd", loss_neg_dep)

            l_smooth_sum += loss_s.item()
            l_match_sum += loss_m.item()
            l_recon_sum += loss_r.item()
            l_nd_sum += loss_neg_dep.item()
            l_ssim_sum += loss_ssim.item()

            optimizer.zero_grad()
            loss = 0.005 * loss_m + 10 * loss_r + args.nd_lambda* loss_neg_dep + args.ssim_lambda * loss_ssim + args.smooth_lambda * loss_s

            loss.backward()
            optimizer.step()

            if step % chunk == chunk - 1:
                batch = torch.cat([l_batch,r_batch])
                depth = torch.cat([l_depth,r_depth])
                rgb = batch.clone()
                rgb[:, 0] = batch[:, 2]
                rgb[:, 2] = batch[:, 0]
                writer.add_images('img', rgb, step)
                writer.add_images('depth', convert_to_colormap_batch(depth), step)
                writer.add_scalar('img recons loss', l_recon_sum / chunk, step)
                writer.add_scalar('pc match loss', l_match_sum / chunk, step)
                writer.add_scalar('neg depth loss', l_nd_sum / chunk, step)
                writer.add_scalar('ssim loss', l_ssim_sum / chunk, step)
                writer.add_scalar('smooth loss', l_smooth_sum / chunk, step)
                if not visualize:
                    is_best = False
                    #cur_res = 0.0 
                    cur_res = evaluate(args.kitti_path, copy.deepcopy(egodepth),mode='valid')
                    writer.add_scalar('less than 1.25', cur_res, step)
                    if cur_res > best_res:
                        is_best = True
                        best_res = cur_res
                        print("updated best result! current:", best_res)
                    else:
                        print("current best:", best_res)
                    '''
                    print("sampler length:", len(sampler))
                    print("sampler last ten:", sampler.seq[-10:])
                    print("sampler seed:", sampler.seed)
                    '''
                    save_ckpt(experiment_directory, egodepth, hyper, optimizer, scheduler, is_best, multiple_gpu, \
                              epoch_num=e, step_num=step, seed=sampler.seed)
                # print(egodepth.conv7b.conv.weight)
                if visualize:
                    for i in range(b):
                        visualize_img_depth(l_batch[i].float(), l_depth[i].float(), l_skymask[i].float())
                        visualize_img_depth(r_batch[i].float(), r_depth[i].float(), r_skymask[i].float())
                '''
                print(egodepth.conv7b.conv.weight.grad)
                print("min",depth_batch.min())
                print("max",depth_batch.max())
                '''
                print(l_depth[0])
                print(
                    'Epoch [{}/{}] , Step {}, l_nd: {:.4f}, l_smooth: {:.4f}, l_ssim: {:.4f}, l_recon: {:.4f}, l_match: {:.4f}'
                    .format(e + 1, num_epochs, step, l_nd_sum / chunk, l_smooth_sum / chunk, l_ssim_sum / chunk,
                            l_recon_sum / chunk, l_match_sum / chunk))
                l_smooth_sum, l_ssim_sum, l_recon_sum, l_match_sum, l_nd_sum = 0, 0, 0, 0, 0
            step += 1
        scheduler.step()
    # writer.close()


if __name__ == '__main__':
    train(visualize=False)
