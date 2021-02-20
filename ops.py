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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def ego_transform(pcs,egos,skymask=None):
    '''
    pcs - B x 3 x (hxw)
    egos - B x 4 x 4
    skymask - B x 1 x h x w
    return transformed pcs
    '''
    b,_,n=pcs.shape
    pcs_hom=pcs.new(b,4,n).zero_()
    pcs_hom[:,:3]=pcs
    pcs_hom[:,3]=1
    if skymask is not None:
        skymask=skymask.view(b,-1)
        pcs_hom[:,3]=skymask
    pcs_trans=egos.bmm(pcs_hom) # b x 4 x (hxw)
    pcs_trans=pcs_trans[:,:3]
    return pcs_trans

def get_depth(inv_depths):
    '''
    inv_depths - B x 1 x h x w
    get normal depths
    of the same size of input
    '''
    depths=1.0/inv_depths
    return depths

def depth2pc(depths,proj,vc,orig_h=None,orig_w=None,skymask=None):
    '''
    depths - B x 1 x h x w
    proj (ci) - B x 4 x 4
    vc - B x 4 x 4
    orig_h - B
    orig_w - B
    skymask - B x 1 x h x w
    return pcs - B x 3 x (hxw), invalid pixel gets x-coord 0
    '''
    b,_,h,w=depths.shape
    h_ratio=1.0
    w_ratio=1.0
    if orig_h is not None:
        h_ratio=(orig_h/h).unsqueeze(1).cpu()
        w_ratio=(orig_w/w).unsqueeze(1).cpu()
    proj_inv=proj.inverse()
    pcs=depths.new(b,4,h*w).zero_()
    y_idxs=torch.arange(h).float().repeat_interleave(w).unsqueeze(0).expand(b,-1)*h_ratio # 0000...1111...2222....
    x_idxs=torch.arange(w).float().repeat(h).unsqueeze(0).expand(b,-1)*w_ratio #0123...0123...0123...
    pcs[:,0]=x_idxs
    pcs[:,1]=y_idxs
    pcs[:,2]=1
    depths=depths.view(b,1,h*w)
    pcs=pcs.clone()*depths
    pcs[:,3]=1
    if vc is None and skymask is not None:
        skymask=skymask.view(b,-1)
        pcs[:,3]=skymask
    pcs=proj_inv.bmm(pcs) # in camera frame now
    if vc is not None:
        if skymask is not None:
            skymask=skymask.view(b,-1)
            pcs[:,3]=skymask
        vc_inv=vc.inverse()
        pcs=vc_inv.bmm(pcs)
    return pcs[:,:3]

def img_sampling(img,idx):
    '''
    img - c x h x w
    idx_abs - N
    sample all the pixels in the idx array
    return pixel array - c x N 
    '''
    c,h,w=img.shape
    idx = idx.long().clamp(0,h*w-1)
    img = img.view(c,-1) # c x h*w
    img_ret=img.index_select(1,idx) # c x N
    return img_ret

def img_sampling_bilinear(img,idx_f):
    '''
    Bilinear sampling to make result differentiable wrt idx
    img - c x h x w
    idx_f - 2 x N
    sample all the pixels in the idx array
    return pixel array - c x N 
    '''
    c,h,w=img.shape
    img = img.view(c,-1) # c x h*w
    x_f=idx_f[0]
    y_f=idx_f[1]
    x0=x_f.floor().clamp(0,w-2).long()
    x1=x0.clone()+1
    y0=y_f.floor().clamp(0,h-2).long()
    y1=y0.clone()+1

    upperleft=y0*w+x0
    upperright=y0*w+x1
    lowerleft=y1*w+x0
    lowerright=y1*w+x1

    ul=img.index_select(1,upperleft) # 3 x N
    ur=img.index_select(1,upperright)
    ll=img.index_select(1,lowerleft)
    lr=img.index_select(1,lowerright)

    x0=x0.clone().float()
    y0=y0.clone().float()
    x1=x1.clone().float()
    y1=y1.clone().float()

    wul=((x1-x_f)*(y1-y_f)).unsqueeze(0)
    wur=((x_f-x0)*(y1-y_f)).unsqueeze(0)
    wll=((x1-x_f)*(y_f-y0)).unsqueeze(0)
    wlr=((x_f-x0)*(y_f-y0)).unsqueeze(0) # 1 x N

    pixel=ul*wul+ur*wur+ll*wll+lr*wlr # 3 x N

    return pixel

def register_pc(pc,proj,vc,mask,orig_h=None,orig_w=None,h=None,w=None,ablation=False,gordon=False,target_depth=None, include_nd=False):
    '''
    pc - 3 x (hxw)
    proj - 4 x 4
    vc - 4 x 4
    mask - 1 x h x w
    orig_h - scalar
    orig_w - scalar
    target_depth, used for gordon zbuffer - 1 x h x w
    return registration - 3 x h x w
    corresponding new idx for each point - 2 x h*w
    corresponding original idx for each point - h*w
    '''
    start=int(time.time()*1000.0)
    if h is None:
        _,h,w=mask.shape
    h_ratio = 1.0
    w_ratio = 1.0
    if orig_h is not None:
        h_ratio=orig_h/h
        w_ratio=orig_w/w
    mask=mask.view(-1) # h*w
    reg=pc.new(h*w).zero_() # (h*w)

    #project the point cloud to image frame to get the i,j index for each point
    pc_hom=pc.new(4,pc.shape[1]).zero_()
    pc_hom[:3]=pc.clone()
    pc_hom[3]=mask.clone()
    if vc is not None:
        pc_hom=vc.matmul(pc_hom) # in camera frame
        pc_hom[3]=1
    pc_trans=proj.matmul(pc_hom) # in image frame
    pc_trans=pc_trans[:3].clone()
    depths=pc_trans[2]
    dz=(depths==0).float()
    pc_pixel=pc_trans[:2]/(depths+dz) # 2 x (hxw)
    pc_pixel[0]=pc_pixel[0].clone()/w_ratio
    pc_pixel[1]=pc_pixel[1].clone()/h_ratio
    pc_pixel_f=pc_pixel.clone()
    pc_pixel=pc_pixel.floor()

    #calculate the absolute index of the points
    # i.e., where each point should go in reg
    y,x=pc_pixel[1],pc_pixel[0]
    idx_abs=y*w+x # y*w+x
    idx_orig=idx_abs.new(pc.shape[1]).zero_().long()
    idx_orig[:]=torch.arange(pc.shape[1]) #01234...

    #get out of bound idx and get rid of those points
    in_bound=(x>=0).long()*(y>=0).long()*(x<w).long()*(y<h).long()
    invalid_oob=in_bound*((depths<=0).long()) # points inside frame but get neg depth
    invalid_depth = depths*invalid_oob
    if invalid_oob.sum() != 0:
        loss_neg_depth=invalid_depth.sum()/invalid_oob.sum()*(-1)
    else:
        loss_neg_depth=invalid_depth.sum()*0.0
    if not include_nd:
        pos_depth=(depths>0).long()
        in_bound=in_bound*pos_depth
    if (not ablation) and gordon:
        fetched_target_depth = img_sampling_bilinear(target_depth, pc_pixel_f)
        smaller_than_target = (depths <= fetched_target_depth).long()
        in_bound=in_bound*smaller_than_target
    in_bound_idx=in_bound.nonzero().view(-1)

    #concatenate pc, pixel_f and orig_idx for easier operation
    pc = torch.cat([pc.clone(),pc_pixel_f,idx_orig.unsqueeze(0).float()],dim=0) # 6 x n
    pc=pc.index_select(1,in_bound_idx)
    idx_abs=idx_abs.index_select(0,in_bound_idx)

    if ablation or gordon:
        if pc.shape[1]==0:
            print("no valid point")
            return None, None, loss_neg_depth
        else:
            return pc[3:5],pc[5],loss_neg_depth

    if pc.shape[1]==0:
        print("no valid point")
        return None,None,loss_neg_depth

    # the main zbuffer code 
    idx_abs = idx_abs.long()
    idx_abs_orig=idx_abs.clone()

    #since the rows are not operated simultaneously, we only copy the depth row
    depth_row = pc[0].clone() # x's
    reg.index_copy_(0,idx_abs,depth_row)

    while True:
        depth_back = reg.index_select(0,idx_abs) # get back the points from the R
        keep = (depth_row-depth_back < 0).long() # keep the point whose depth is smaller then the registered point
        if keep.sum()==0:
            break
        keep_idx = keep.nonzero()
        keep_idx = keep_idx.view(-1)
        depth_row=depth_row.index_select(0,keep_idx)

        idx_abs=idx_abs.index_select(0,keep_idx)
        reg.index_copy_(0,idx_abs,depth_row)

    # get the depth back from reg using idx_abs_orig
    depth_back = reg.index_select(0,idx_abs_orig)
    keep = (pc[0]-depth_back == 0).long() # keep points with depth <= reg depth
    keep_idx = keep.nonzero().view(-1)
    pc = pc.index_select(1,keep_idx) # select valid points

    pc_pixel_f = pc[3:5]
    idx_orig = pc[5]

    end=int(time.time()*1000.0)
    return pc_pixel_f,idx_orig,loss_neg_depth
