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
import argparse
import random
import sys

def parse_args():
    '''
    parse cmd line args
    '''
    parser = argparse.ArgumentParser(description='Generate validation set')

    # Dataset
    parser.add_argument('--kitti_path', type=str, help='path to kitti dataset',
                        default='/data/raw/robotics/kitti/raw_sequences-20200224133836/data/')
    parser.add_argument('--eigen_val_path', type=str,
                        help='path to a text file that indicates eigen validation set',
                        default='eigen_val_files.txt')

    # arguments
    parser.add_argument('--max_size', type=int, help='max number of images that validation set contains', default=1000)
    parser.add_argument('--seed', type=int, help='random seed used for generating validation set', default=42)

    # path
    parser.add_argument('--image_path', type=str, help='path to output a list of image',
                        default='data/image_subsequence_valid.txt')
    parser.add_argument('--depth_path', type=str, help='path to output a list of depth',
                        default='data/depth_subsequence_valid.txt')

    # load arguments
    args = parser.parse_args()
    return args


def convert_image_to_depth_path(image_path):
    parts = image_path.split(os.path.sep)
    return os.path.join(*parts[:-3], 'proj_depth', 'groundtruth', parts[-3], parts[-1])


def generate():
    args = parse_args()

    # generate a set of image path from the eigen val file
    image_list = list()
    depth_list = list()
    root_folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_folder, args.eigen_val_path)) as eigen_val_file:
        for line in eigen_val_file.readlines():
            left, right = line.strip().split(' ')
            left = left[:-3]+'png'
            right = right[:-3]+'png'
            left_depth = convert_image_to_depth_path(left)
            right_depth = convert_image_to_depth_path(right)
            if os.path.exists(os.path.join(args.kitti_path, left_depth)):
                image_list.append(left)
                depth_list.append(left_depth)
            if os.path.exists(os.path.join(args.kitti_path, right_depth)):
                image_list.append(right)
                depth_list.append(right_depth)

    # Fisher–Yates shuffle
    random.seed(args.seed) # set random seed
    for index in range(len(image_list)-1, 0, -1):
        index_to_swap = random.randint(0, index+1)
        image_list[index], image_list[index_to_swap] = image_list[index_to_swap], image_list[index]
        depth_list[index], depth_list[index_to_swap] = depth_list[index_to_swap], depth_list[index]

    if args.max_size < len(image_list):
        image_list = image_list[:args.max_size]
        depth_list = depth_list[:args.max_size]

    # Write
    with open(os.path.join(root_folder, args.image_path), 'w') as image_file:
        for image_path in image_list:
            image_file.write(image_path+'\n')
    with open(os.path.join(root_folder, args.depth_path), 'w') as depth_file:
        for depth_path in depth_list:
            depth_file.write(depth_path+'\n')


if __name__ == '__main__':
    generate()
