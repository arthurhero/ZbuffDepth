#Adapted from Monodepth (https://github.com/mrharicot/monodepth)
# Copyright © Niantic, Inc. 2018. All rights reserved.
#
# In particular, the code is a derivative of
#   https://github.com/mrharicot/monodepth/tree/master/utils/evaluation_utils.py
# which is released under the terms of the following license:
#   https://github.com/mrharicot/monodepth/blob/master/LICENSE
# having the specific clause:
#   3.  Redistribution and modifications
#    3.1 The Licensee may reproduce and distribute copies of the Software only to
#        this same GitHub repository with or without modifications, in source format
#        only and provided that any and every distribution is accompanied by an
#        unmodified copy of this Licence and that the following copyright notice is
#        always displayed in an obvious manner: Copyright © Niantic, Inc. 2018. All
#        rights reserved.
#
#    3.2 In the case where the Software has been modified, any distribution must
#        include prominent notices indicating which files have been changed.
#
#    3.3 The Licensee shall cause any work that it distributes or publishes, that
#        in whole or in part contains or is derived from the Software or any part
#        thereof (“Work based on the Software”), to be licensed as a whole at no
#        charge to all third parties under the terms of this Licence.

import numpy as np
import cv2
import argparse
from evaluation_utils import *
from data_process import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_eigen(dataset_path,eval_input,model,crop_type='eigen'):
    min_depth = 1e-3
    max_depth = 80.0
    if isinstance(eval_input, str):
        if model == 'orig':
            from bts_orig import BtsModel
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

    num_samples = 697
    test_files = read_text_lines('eigen_test_files.txt')
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, dataset_path)

    num_test = len(im_files)
    gt_depths = []
    pred_depths = []
    for t_id in range(num_samples):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
        gt_depths.append(depth.astype(np.float32))
        
        img = load_img(im_files[t_id])
        img = torch.Tensor(img).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            depth = egodepth(img)
            gt=torch.Tensor(gt_depths[t_id].copy()).unsqueeze(0)
            #visualize_img_depth(img[0].float(), depth[0].float(), save=True, fname=str(t_id)+'.jpg')
            depth = depth.squeeze().cpu().detach().numpy()  # h x w x 1

        depth_pred = cv2.resize(depth, (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_NEAREST)
        depth_pred[np.isinf(depth_pred)] = 0

        pred_depths.append(depth_pred)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):
        
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        
        gt_height, gt_width = gt_depth.shape

        # crop used by Garg ECCV16
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        if crop_type == 'garg':
            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                             0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
        # crop we found by trial and error to reproduce Eigen NIPS14 results
        elif crop_type == 'eigen':
            crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                             0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))


if __name__ == '__main__':
    eval_eigen('/PATH/TO/kitti',\
            '/PATH/TO/best.ckpt',model='orig',crop_type='garg')
