# ZbuffDepth 
An self-supervised monocular depth learning method utilizing image 
reconstruction loss, with the point occlusion issue solved by the novel z-buffer.

This repo trains and tests the model on stereo pairs using the Eigen splits.

# Author(s) 

Ziwen Chen (github: arthurhero)
Zixuan Guo (github: Olament)

## Usage

### Preparing
1. Create a blank folder.
```sh
mkdir raw_kitti && cd raw_kitti
```
2. Download the raw kitti dataset.
```sh
wget https://raw.githubusercontent.com/mrharicot/monodepth/master/utils/kitti_archives_to_download.txt 
wget -i kitti_archives_to_download.txt
```
3. Unzip the data.
```sh
unzip "*.zip"
```
4. Download KITTI's annotated depth map data set (14G) at [KITTI's website](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)
5. Merge the annotated depth data with the raw kitti
```sh
unzip data_depth_annotated.zip
mv -R data_depth_annotated/* raw_kitti
```

### Training
1. Create folder `mkdir logs`
2. For training on stereo pairs, edit the parameters in `train_stereo.sh`, and run `./train_stereo.sh`

### Evaluating On Eigen
1. Open `evaluate_kitti.py` and scroll to bottom.
2. Modify first argument to KITTI data path and second argument to checkpoint path from training
2. Run `./evaluate_kitti.py`


## Files

*train_\*.sh*
:   the training wrapper script for stereo and egomotion, with cmd args

*run_\*.py*
:   the training code

*loss.py*
:   code for all the losses used in the method, e.g., ssim loss and reconstruction loss

*ops.py*
:   code for all the operators used in the method, e.g., depth-to-point projection, z-buffering, etc.

*data_process.py*
:   code for all the data preprocessing and viewing methods, e.g., calculating egomotion matrix from gps coord

*dataloader.py*
:   code for the PyTorch dataloaders

*utils.py*
:   some basic utilities for PyTorch

*eigen_\*_files.txt*
:   the standard eigen splits

*generate_valid.py*
:   reproducibly generate a validation split by randomly sampling from eigen validation split 

*evaluate_kitti.py*
:   main evaluation code on eigen test split, adapted from Monodepth

*evaluation_utils.py*
:   utils for the evaluation, adapted from Monodepth

*bts_orig.py*
:   the encoder-decoder architecture, adapted from BTS (Lee et. al.)


## Acknowledgments

