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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def match_loss(R_,R):
    '''
    R_,R - 3 x h x w
    calculate l_match
    invalid pixels have 0 for x-coord
    R_ is the registration from the transformed pc
    '''
    R_x=R_[0] # h x w
    Rx=R[0] # h x w
    mask_=(R_x>0).double()
    mask=(Rx>0).double()
    mask*=mask_ # combine the two masks
    mask=mask.unsqueeze(0) # 1 x h x w
    
    loss=((R_-R).abs()*mask).sum()

    if mask.sum() == 0.0:
        return loss*0.0
    loss=loss/mask.sum()
    return loss

def gradient_x(img):
    '''
    img - b x c x h x w
    '''
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    '''
    img - b x c x h x w
    '''
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def smooth_loss(D,I,mask=None):
    '''
    D,depth - B x 1 x h x w
    I,image - B x 3 x h x w
    mask - B x 1 x h x w
    calculate depth smoothness loss
    '''
    depth_dx=gradient_x(D)
    depth_dy=gradient_y(D)
    img_dx=gradient_x(I)
    img_dy=gradient_y(I)
    weights_x=((img_dx.abs().mean(1,keepdim=True))*(-1)).exp()
    weights_y=((img_dy.abs().mean(1,keepdim=True))*(-1)).exp()
    smooth_x=depth_dx*weights_x
    smooth_y=depth_dy*weights_y
    loss = D.new((0,)).zero_()
    if mask is not None:
        mask = 1-mask # make sky points one
        mask_x = mask[:,:,:,:-1]
        if mask_x.sum() != 0:
            loss = (smooth_x.abs()*mask_x).sum()/mask_x.sum()
        mask_y = mask[:,:,:-1,:]
        if mask_y.sum() != 0:
            loss += (smooth_y.abs()*mask_y).sum()/mask_y.sum()
    else:
        loss=smooth_x.abs().mean()+smooth_y.abs().mean()
    return loss

def gradient_smooth_loss(D,I):
    '''
    D,depth - B x 1 x h x w
    I,image - B x 3 x h x w
    calculate depth gradient smoothness loss
    '''
    depth_dx=gradient_x(D)
    depth_dxx=gradient_x(depth_dx)
    depth_dy=gradient_y(D)
    depth_dyy=gradient_y(depth_dy)
    img_dx=gradient_x(I)[:,:,:,:-1]
    img_dy=gradient_y(I)[:,:,:-1,:]
    img_dx=img_dx.pow(2).sum(1,keepdim=True).pow(0.5)
    img_dx_mask=(img_dx<0.05).double()
    img_dy=img_dy.pow(2).sum(1,keepdim=True).pow(0.5)
    img_dy_mask=(img_dy<0.05).double()
    weights_x=((-img_dx).exp())*img_dx_mask
    weights_y=((-img_dy).exp())*img_dy_mask
    smooth_x=depth_dxx*weights_x
    smooth_y=depth_dyy*weights_y
    loss=smooth_x.abs().mean()+smooth_y.abs().mean()
    return loss

def unsmooth_loss(D,I):
    '''
    D,depth - B x 1 x h x w
    I,image - B x 3 x h x w
    punish same depth when the color of image changes 
    '''
    depth_dx=gradient_x(D)
    depth_dy=gradient_y(D)
    img_dx=gradient_x(I)
    img_dy=gradient_y(I)
    weights_x=img_dx.abs().mean(1,keepdim=True).pow(2)
    weights_y=img_dy.abs().mean(1,keepdim=True).pow(2)
    unsmooth_x=depth_dx*weights_x
    unsmooth_y=depth_dy*weights_y
    loss=unsmooth_x.abs().mean()+unsmooth_y.abs().mean()
    loss*=-1
    return loss

def inf_loss(mask):
    '''
    mask - any shape
    calculate infinity loss
    '''
    loss=(-mask).mean()
    return loss

def recon_loss(pixel1,pixel2):
    '''
    pixel array - 3 x N
    '''
    return (pixel1-pixel2).abs().mean()

def ssim(x, y):
    '''
    computes a differentiable structured image similarity measure
    x , y - b x 3 x h x w
    '''
    maxpool = nn.MaxPool2d(3,1)
    mask = ((maxpool(-x.sum(1,keepdim=True))*maxpool(-y.sum(1,keepdim=True))) != 0).float()
    pool = nn.AvgPool2d(3,1)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pool(x) 
    mu_y = pool(y) 
    sigma_x = pool(x**2) - mu_x**2 # var of x
    sigma_y = pool(y**2) - mu_y**2 # var of y
    sigma_xy = pool(x * y) - mu_x * mu_y # covar of x,y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    ssim = ((1 - ssim) / 2).clamp(0, 1)
    ssim *= mask
    ssim = ssim.sum()
    if mask.sum() != 0:
        return ssim / mask.sum()
    else:
        return 0.0

def evaluation_errors(pred,gt):
    '''
    calculate testing error abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    pred, selected predicted pixel depth - N
    gt, grount-truth depth - N
    '''
    thres=torch.max(pred/gt,gt/pred)
    a1 = (thres < 1.25 ).float().mean()
    a2 = (thres < 1.25 ** 2).float().mean()
    a3 = (thres < 1.25 ** 3).float().mean()

    rmse = (gt - pred).pow(2)
    rmse = rmse.mean().pow(0.5)

    rmse_log = (gt.log() - pred.log()).pow(2)
    rmse_log = rmse_log.mean().pow(0.5)

    abs_rel = ((gt - pred) / gt).abs().mean()

    sq_rel = (((gt - pred).pow(2)) / gt).mean()
    results = torch.Tensor([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
    results_ = a1.new(results.shape).zero_()
    results_[:]=results
    return results_
