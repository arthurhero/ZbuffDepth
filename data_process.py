'''
contains methods for loading and processing data

exec_cmd(cmd)
download_kitti()
process_kitti_raw()
integ_sequence(sequence)
process_kitti_integ()
load_img(filename)
load_calib(filename)
load_proj(filename)
load_pos(filename)
load_timediff(filename)
load_pc(filename)
visualize_pc(pc)
load_odometry(filename)
euc2hom(eu)
hom2euc(hom)
padRT(R,T)
laser2cam(laser,vcR,vcT)
cam2laser(cam,vcR,vcT)
get_rot_mat(roll,pit,yaw)
get_trans_mat(x,y,z)
get_ego_matrix(pos1,pos2,gvR,gvT,meter=False,velocity=True)
load_depth(filename)
visualize_depth(depth)
visualize_img_depth(img,depth)
depth2color_map(depth_batch)
scatter_pixel(pixel,idx,h,w)
convert_to_colormap(tensor)
'''

import subprocess
import os

import cv2
import numpy as np
from math import cos
from math import sin 
from math import pi 
from statistics import mean
from statistics import variance as var 
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

#from open3d import *
import torch


kitti_path='datasets/kitti/'
kitti_raw_path=kitti_path+'raw/'
kitti_depth_path=kitti_path+'depth/'
kitti_calib_path=kitti_path+'calib/'
#https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
#https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip

def exec_cmd(cmd):
    '''
    execute a bash cmd
    '''
    #print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error

def download_kitti():
    kp=kitti_path
    f = open(kitti_path+'seq_names.txt','r')
    seqs = f.read().splitlines()
    f.close()
    krp=kitti_raw_path
    kcp=kitti_calib_path
    dates=set()
    for s in seqs:
        date=s[:10]
        dates.add(date)
        cmd='wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'+s+'/'+s+'_sync.zip -P '+krp
        exec_cmd(cmd)
    for d in dates:
        cmd='wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'+d+'_calib.zip -P '+kcp
        exec_cmd(cmd)

def list_folder(folder):
    files,_=exec_cmd('ls '+folder)
    files=files.decode("utf-8").split()
    return files

def process_kitti_raw():
    '''
    process raw kitti into a sorted kitti
    '''
    kp=kitti_path
    krp=kitti_raw_path
    #kdp=kitti_depth_path
    kcp=kitti_calib_path
    f = open(kitti_path+'seq_names.txt','r')
    seqs = f.read().splitlines()
    f.close()

    calibs=list_folder(kcp)
    print(calibs)
    for c in calibs:
        cmd='unzip '+kcp+c+' -d '+kcp
        exec_cmd(cmd)
        cmd='rm '+kcp+c
        exec_cmd(cmd)

    for s in seqs:
        print(s)
        cmd='unzip '+krp+s+'_sync.zip -d '+kp
        exec_cmd(cmd)

        date=s[:10]
        cmd='mv '+kp+date+'/'+s+'_sync '+kp
        exec_cmd(cmd)

        cmd='rm -rf '+kp+s+'_sync/image_00'
        exec_cmd(cmd)

        cmd='rm -rf '+kp+s+'_sync/image_01'
        exec_cmd(cmd)

        cmd='rm -rf '+kp+s+'_sync/image_03'
        exec_cmd(cmd)

        cmd='rm -rf '+kp+s+'_sync/velodyne_points'
        exec_cmd(cmd)

        cmd='mv '+kp+s+'_sync/image_02 '+kp+s+'_sync/image'
        exec_cmd(cmd)

        cmd='rm -rf '+krp+s+'_sync.zip'
        exec_cmd(cmd)

        cmd='rm -rf '+kp+date
        exec_cmd(cmd)

        cmd='cp -R '+kcp+date+' '+kp+s+'_sync/calib/'
        exec_cmd(cmd)

    cmd='rm -rf '+kcp
    exec_cmd(cmd)
    cmd='rm -rf '+krp
    exec_cmd(cmd)

def get_stereo_matrices(left_img_path, raw_kitti_path):
    '''
    left_img_path: path to the left image after raw kitti path
    '''
    date = left_img_path[:10]

    opj = os.path.join
    calib_folder = opj(raw_kitti_path,date)
    vcR,vcT = load_calib(opj(calib_folder,'calib_velo_to_cam.txt'))
    vc = padRT(vcR,vcT)

    proj=load_proj(opj(calib_folder,'calib_cam_to_cam.txt'),cam=2)
    proj_=np.zeros((4,4),np.float64)
    proj_[:3]=proj
    proj_[3,3]=1
    proj_left=proj_
    
    proj=load_proj(opj(calib_folder,'calib_cam_to_cam.txt'),cam=3)
    proj_=np.zeros((4,4),np.float64)
    proj_[:3]=proj
    proj_[3,3]=1
    proj_right=proj_
    
    return vc,proj_left,proj_right

def integ_sequence(sequence, start_idx, end_idx, raw_kitti_path=None, separate=False):
    '''
    integrate a sequence's images and egomotion into one tensor
    return img_tensor, N x h x w x 3
    egomotion tensor, N-1 x 4 x 4
    vci (velo to camera to image) tensor, 4 x 4
    separate - separate proj and vc or not
    '''
    #print("seq:",sequence, start_idx)
    kp=raw_kitti_path
    date=sequence[:10]
    if kp is None:
        folder=kitti_path+sequence+'_sync/'
        img_folder=folder+'image/data/'
    else:
        folder=kp+date+'/'+sequence+'_sync/'
        img_folder=folder+'image_02/data/'
    imgs,_=exec_cmd('ls '+img_folder)
    imgs=imgs.decode("utf-8").split()
    img_tensor=list()
    for i in range(len(imgs)):
        if i>=start_idx and i<=end_idx:
            img_file=img_folder+imgs[i]
            img=load_img(img_file)
            #img = cv2.resize(img,(1242,375))
            #print(sequence,img.shape)
            img_tensor.append(img)
    img_tensor=np.stack(img_tensor)
    #print("just loaded h w",img_tensor.shape)

    pos_folder=folder+'oxts/data/'
    poses,_=exec_cmd('ls '+pos_folder)
    poses=poses.decode("utf-8").split()
    pos_tensor=list()
    for i in range(len(poses)):
        if i>=start_idx and i<=end_idx:
            p=poses[i]
            pos=load_pos(pos_folder+p)
            pos_tensor.append(pos)
    pos_tensor=np.stack(pos_tensor)

    if kp is None:
        calib_folder=folder+'calib/'
    else:
        calib_folder=kp+date+'/'
    gvR,gvT=load_calib(calib_folder+'calib_imu_to_velo.txt') # gps coord to pc coord
    ego_tensor=list()
    for i in range(len(pos_tensor)-1):
        mat=get_ego_matrix(pos_tensor[i],pos_tensor[i+1],gvR,gvT,meter=False,velocity=False)
        ego_tensor.append(mat)
    ego_tensor=np.stack(ego_tensor)
    #print(ego_tensor.shape)

    vcR,vcT=load_calib(calib_folder+'calib_velo_to_cam.txt')
    vc=padRT(vcR,vcT)
    proj=load_proj(calib_folder+'calib_cam_to_cam.txt')
    proj_=np.zeros((4,4),np.float64)
    proj_[:3]=proj
    proj_[3,3]=1
    proj=proj_
    
    if separate:
        return img_tensor,ego_tensor,proj,vc
    else:
        vci=np.matmul(proj,vc) # 4x4
        return img_tensor,ego_tensor,vci

def process_kitti_integ():
    kp=kitti_path
    cur_folders=list_folder(kp)

    f = open(kitti_path+'seq_names2.txt','r')
    seqs = f.read().splitlines()
    f.close()
    vcis=list()
    img_folder=kp+'images/'
    ego_folder=kp+'egos/'
    cmd='mkdir '+img_folder
    exec_cmd(cmd)
    cmd='mkdir '+ego_folder
    exec_cmd(cmd)

    for i in range(len(seqs)):
        entry=seqs[i]
        print(entry)
        s,start,end=entry.split()
        start,end=int(start),int(end)
        img,ego,vci=integ_sequence(s, start, end)
        vcis.append(vci)
        np.savez(img_folder+str(i)+'.npz',img)
        np.savez(ego_folder+str(i)+'.npz',ego)
        print()
    vcis=np.stack(vcis)
    np.savez(kp+'projections.npz',vcis)

    for cur_f in cur_folders:
        cmd='rm -rf '+kp+cur_f
        exec_cmd(cmd)

def load_img(filename):
    '''
    load the rgb image
    '''
    #print(filename)
    img=cv2.imread(filename,-1)
    img=img.astype(np.float64)
    img/=256.0
    '''
    print(img.dtype)
    cv2.imshow('lol',img)
    cv2.waitKey(0)
    '''
    return img

def load_calib(filename):
    '''
    load the velo_to_cam matrix
    or the imu_to_velo matrix
    '''
    f = open(filename,'r')
    RT=f.read().splitlines()[1:3]
    f.close()
    R=[float(num) for num in RT[0].split()[1:]]
    T=[float(num) for num in RT[1].split()[1:]]
    R=np.asarray(R,dtype=np.float64).reshape((3,3))
    T=np.asarray(T,dtype=np.float64).reshape((3,))
    return R,T

def load_proj(filename,cam=2):
    '''
    load 3 x 4 projection matrix for cam x
    fixed bug: added rotation from rect0
    '''
    f = open(filename,'r')
    lines=f.read().splitlines()
    f.close()
    rot=lines[8] # rect0 matrix
    r=[float(num) for num in rot.split()[1:]]
    r=np.asarray(r,dtype=np.float64).reshape((3,3))
    r_=np.zeros((4,4),dtype=np.float64)
    r_[:3,:3]=r
    r_[3,3]=1
    r=r_
    projs=[9,17,25,33] # lines for proj matrices
    proj=lines[projs[cam]]
    m=[float(num) for num in proj.split()[1:]]
    m=np.asarray(m,dtype=np.float64).reshape((3,4))
    m=np.matmul(m,r)
    return m

def load_pos(filename):
    '''
    load 6 dof position 
    lat,lon,alt,roll,pitch,yaw
    deg,deg,m,rad,rad,rad
    and velocities - north,east,forward,left,up
    and rotation velocities - roll,pitch,yaw
    '''
    f = open(filename,'r')
    content = f.read().split()[:23]
    content=[float(c) for c in content]
    content=np.concatenate((content[:11],content[20:23]))
    f.close()
    return content

def load_timediff(filename):
    '''
    load time difference in secs
    first is 0
    '''
    f = open(filename,'r')
    ts=f.read().splitlines()
    ts=[line.split()[1] for line in ts]
    secs=[0]
    for i in range(len(ts)-1):
        tp=ts[i]
        t=ts[i+1]
        hp,mp,sp=tp.split(':')
        h,m,s=t.split(':')
        hp,mp,sp=int(hp),int(mp),float(sp)
        h,m,s=int(h),int(m),float(s)
        diff=3600*h+60*m+s-(3600*hp+60*mp+sp)
        secs.append(diff)
    return secs

def load_pc(filename):
    '''
    load the point cloud
    return nx3 numpy array
    '''
    data=np.fromfile(filename, dtype=np.float32)
    #print(data.shape)
    data=np.reshape(data,(-1,4))
    '''
    for i in range(data.shape[0]):
        print(data[i][0])
        '''
    data=data.astype(np.float64)
    return data[:,:3]

'''
def visualize_pc(pc):
    #pc - nx3 numpy
    if pc.shape[0]==3:
        pc=pc.permute(1,0).cpu().detach().numpy()
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(pc)
    visualization.draw_geometries([pcd])
    '''

def load_odometry(filename):
    '''
    load the odometry matrices
    '''
    f = open(filename,'r')
    matrices = f.read().splitlines()
    f.close()
    mats=list()
    for m in matrices:
        m = [float(num) for num in m.split()]
        m=np.asarray(m,dtype=np.float64)
        m=np.reshape(m,(3,4))
        m_=np.zeros((4,4),dtype=np.float64)
        m_[:3]=m
        m_[3,3]=1
        mats.append(m_)
    return mats

def euc2hom(eu):
    '''
    euclidean to homogeneous
    pad a 1 column at the end
    '''
    r,c=eu.shape
    hom=np.ones((r,c+1),dtype=np.float64)
    hom[:,:c]=eu
    return hom

def hom2euc(hom):
    '''
    homogeneous to euclidean
    '''
    r,c=hom.shape
    last=hom[:,c-1:]
    hom=hom/last
    eu=hom[:,:c-1]
    return eu

def padRT(R,T):
    '''
    pad [0,0,0,1] as the last raw to R|T
    '''
    mat=np.zeros((4,4),dtype=np.float64)
    mat[:3,:3]=R
    mat[:3,3]=T
    mat[3,3]=1
    return mat

def laser2cam(laser,vcR,vcT):
    '''
    laser - nx3
    convert laser coord to camera coord
    '''
    laser=euc2hom(laser)
    cam=np.zeros_like(laser)
    vc=padRT(vcR,vcT)
    for i in range(laser.shape[0]):
        cam[i]=np.matmul(vc,laser[i])
    cam=hom2euc(cam)
    '''
    cam = np.zeros_like(laser,dtype=np.float64)
    cam[:,0]=-laser[:,1]
    cam[:,1]=-laser[:,2]
    cam[:,2]=laser[:,0]
    '''
    return cam

def cam2laser(cam,vcR,vcT):
    '''
    cam - nx3 numpy array
    convert camera coord to laser coord
    '''
    cam=euc2hom(cam)
    laser=np.zeros_like(cam)
    vc=padRT(vcR,vcT)
    cv=np.linalg.inv(vc)
    for i in range(laser.shape[0]):
        laser[i]=np.matmul(cv,cam[i])
    laser=hom2euc(laser)
    '''
    laser = np.zeros_like(cam,dtype=np.float64)
    laser[:,0]=cam[:,2]
    laser[:,1]=-cam[:,0]
    laser[:,2]=-cam[:,1]
    '''
    return laser

def get_rot_mat(roll,pit,yaw):
    '''
    get 4x4 rotation matrix given roll, pitch, yaw
    '''
    mat=np.zeros((4,4),dtype=np.float64)

    mat_r=np.zeros((3,3),dtype=np.float64)
    mat_r[0][0]=1
    mat_r[1][1]=cos(roll)
    mat_r[1][2]=-sin(roll)
    mat_r[2][1]=sin(roll)
    mat_r[2][2]=cos(roll)

    mat_p=np.zeros((3,3),dtype=np.float64)
    mat_p[1][1]=1
    mat_p[0][0]=cos(pit)
    mat_p[0][2]=sin(pit)
    mat_p[2][0]=-sin(pit)
    mat_p[2][2]=cos(pit)

    mat_y=np.zeros((3,3),dtype=np.float64)
    mat_y[2][2]=1
    mat_y[0][0]=cos(yaw)
    mat_y[0][1]=-sin(yaw)
    mat_y[1][0]=sin(yaw)
    mat_y[1][1]=cos(yaw)

    mat_small=np.matmul(mat_y,np.matmul(mat_p,mat_r))
    mat[:3,:3]=mat_small
    mat[3][3]=1
    return mat

def get_trans_mat(x,y,z):
    '''
    get 4x4 translation matrix given 3d translation vector
    '''
    mat=np.eye(4,dtype=np.float64)
    mat[0][3]=x
    mat[1][3]=y
    mat[2][3]=z
    return mat

def get_ego_matrix(pos1,pos2,gvR,gvT,meter=False,velocity=False):
    '''
    calculate egomotion matrix given position at t-1 and t
    meter - pos in meter instead of lat and lon
    velocity - compensate for velocity or not
    '''
    if meter:
        lat_m=pos2[0]-pos1[0]
        lon_m=pos2[1]-pos1[1]
    else:
        lat1=pos1[0]
        lat2=pos2[0]
        latmid=(lat1+lat2)/2.0
        latmid=latmid/180.0*pi
        m_one_lat=111132.92-559.82*cos(2*latmid)+1.175*cos(4*latmid)-0.0023*cos(6*latmid)
        m_one_lon=111412.84*cos(latmid)-93.5*cos(3*latmid)+0.118*cos(5*latmid)
        '''
        # according to RT3000 User Manual
        m_one_lat=6378137*pi/180.0
        m_one_lon=6378137*pi/180.0*cos(lat1*pi/180.0)
        '''
        lat_m=m_one_lat*(lat2-lat1) # if positive, going north
        lon_m=m_one_lon*(pos2[1]-pos1[1]) # if positive, going east
    alt_m=pos2[2]-pos1[2]
    tx,ty,tz=lon_m,lat_m,alt_m
    r1,p1,y1=pos1[3],pos1[4],pos1[5]
    r2,p2,y2=pos2[3],pos2[4],pos2[5]
    #account for velocity
    if velocity:
        t=0.005
        vn=(pos1[6]+pos2[6])/2.0
        ve=(pos1[7]+pos2[7])/2.0
        vu=(pos1[10]+pos2[10])/2.0
        tx+=(ve*t)
        ty+=(vn*t)+0.008
        tz+=(vu*t)+0.01
    '''
    car coord w/ 0 rotation - facing east, level, east is x, north is y, up is z
    cam coord w/ 0 rotation - east is z, south is x, down is y
    '''
    mat_trans=get_trans_mat(-tx,-ty,-tz) # get tranlation matrix

    mat_rot1=get_rot_mat(r1,p1,y1) # get rotation matrix bw pos1 and world (suppose the robot is the origin)
    mat_rot2=get_rot_mat(r2,p2,y2) # get rotation matrix bw pos2 and world
    #convert pos1 coord to world coord wrt rotation using mat_rot1
    # then translation from pos1 to pos2 using mat_trans 
    # finally convert world coord to pos2 coord wrt rotation using inverse of mat_rot2

    mat=np.matmul(np.linalg.inv(mat_rot2),np.matmul(mat_trans,mat_rot1))

    mat_gv=padRT(gvR,gvT)
    mat_vg=np.linalg.inv(mat_gv)
    # translate points from pc coord to gps coord then back to pc coord
    mat=np.matmul(mat_gv,np.matmul(mat,mat_vg))
    return mat

def load_depth(filename):
    '''
    load depth file, convert to meters
    '''
    depth=cv2.imread(filename,-1)
    depth=depth.astype(float)
    depth/=256.0
    return depth

def visualize_depth(depth, window=None):
    '''
    h x w numpy
    '''
    if depth.shape[0]==1:
        depth=depth.cpu().detach().squeeze().numpy()
    if window is None:
        window = 'depth'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.imshow(window,convert_to_colormap(depth,False))
    cv2.waitKey(0)

def visualize_img_depth(img,depth,skymask=None,gt=None, save=False,fname = None):
    if skymask is not None:
        depth*=skymask
    img = img.cpu().detach()
    depth = depth.cpu().detach()
    #gt = gt.cpu().detach()
    d=convert_to_colormap(depth,ret_tensor=True)
    if gt is not None:
        gt_mask = (gt==0.0).float()
        gt=convert_to_colormap(gt+gt_mask,ret_tensor=True)
        gt=gt*(1-gt_mask)
        gt_=gt.clone()
        gt[0]=gt[2]
        gt[2]=gt[0]
        gt[1:2]+=gt_mask
    '''
    d=depth/100.0
    d=d.expand(3,-1,-1)
    if skymask is not None:
        _,h,w=skymask.shape
        s=skymask.new(3,h,w).zero_()
        s[0]=((-skymask)+1)
        d=d+s
    '''
    d_=d.clone()
    d[0]=d_[2]
    d[2]=d_[0]
    if gt is None:
        di=torch.cat([img,d],dim=1)
    else:
        di=torch.cat([img,d,gt],dim=1)
    di=di.permute(1,2,0).numpy()
    if save is True:
        cv2.imwrite(fname,di*255)
    cv2.namedWindow('di',cv2.WINDOW_NORMAL)
    cv2.imshow('di',di)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_img(img,window=None):
    '''
    img - 3 x h x w
    '''
    if img.shape[0]<=3:
        img=img.permute(1,2,0).cpu().detach().numpy()
    if window is None:
        window = 'img'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.imshow(window,img)
    cv2.waitKey(0)


def depth2color_map(depth):
    '''
    depth - B x 1 x h x w
    '''
    depth=depth.clamp(0,100.0)
    depth=depth/100.0
    depth_full=depth.expand(-1,3,-1,-1).clone()
    depth_full[:,1]=0
    depth_full[:,2:]=1-depth
    return depth_full

def get_diff_means(idx,pc_idx=0,velocity=True):
    '''
    0-indexed start idx
    '''
    root='/data/raw/robotics/kitti/raw_sequences-20200224133836/data/'
    vcR,vcT=load_calib(root+"2011_09_30/calib_velo_to_cam.txt")
    gvR,gvT=load_calib(root+"2011_09_30/calib_imu_to_velo.txt")
    pos1=load_pos(root+"2011_09_30/2011_09_30_drive_0020_sync/oxts/data/"+'{:010d}'.format(idx)+".txt")
    pos2=load_pos(root+"2011_09_30/2011_09_30_drive_0020_sync/oxts/data/"+'{:010d}'.format(idx+1)+".txt")
    mat=get_ego_matrix(pos1,pos2,gvR,gvT,velocity=velocity)
    #print(mat)
    #print(np.linalg.inv(mat))
    mat21=load_odometry("/home/chenziwe/robotics/egoDepth/dataset/dataset/poses/06.txt")[idx]
    mat22=load_odometry("/home/chenziwe/robotics/egoDepth/dataset/dataset/poses/06.txt")[idx+1]
    mat2=np.matmul(np.linalg.inv(mat22),mat21)
    #print(mat2)
    pc=load_pc(root+"2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data/"+'{:010d}'.format(pc_idx)+".bin")

    pc_h=euc2hom(pc)
    pc1=np.zeros_like(pc_h)
    for i in range(pc_h.shape[0]):
        #print(pc[i])
        pc1[i]=np.matmul(mat,pc_h[i])
    pc1=hom2euc(pc1)

    pc_c=laser2cam(pc,vcR,vcT)
    pc_ch=euc2hom(pc_c)
    pc2=np.zeros_like(pc_ch)
    for i in range(pc_ch.shape[0]):
        pc2[i]=np.matmul(mat2,pc_ch[i])
    pc2=hom2euc(pc2)
    pc2=cam2laser(pc2,vcR,vcT)
    
    #diff=np.abs(pc1-pc2)
    diff=pc1-pc2
    x_diff=diff[:,0]
    y_diff=diff[:,1]
    z_diff=diff[:,2]
    '''
    print(np.mean(x_diff))
    print(np.mean(y_diff))
    print(np.mean(z_diff))
    print()
    '''
    return np.mean(x_diff),np.mean(y_diff),np.mean(z_diff)

def test_projection(idx):
    '''
    load a point cloud and the corresponding depth map
    and the calib matrix
    test whether the matrix is working
    '''
    pc=load_pc("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/velodyne_points/data/"+'{:010d}'.format(idx)+".bin")
    vcR,vcT=load_calib("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/calib/calib_velo_to_cam.txt")
    vc=padRT(vcR,vcT)
    proj=load_proj("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/calib/calib_cam_to_cam.txt")
    depth=load_depth("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/depth/image_02/"+'{:010d}'.format(idx)+".png")
    img=load_img("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/image/data/"+'{:010d}'.format(idx)+".png")
    '''
    visualize_pc(pc)
    cv2.imshow('lol',depth/100.0)
    cv2.waitKey(0)
    neg_pc=list()
    pos_pc=list()
    zero_pc=list()
    neg_count=0
    pos_count=0
    zero_count=0
    for i in range(pc.shape[0]):
        if pc[i][0]<0:
            neg_count+=1
            neg_pc.append(pc[i])
        elif pc[i][0]>0:
            pos_count+=1
            pos_pc.append(pc[i])
        else:
            zero_count+=1
            zero_pc.append(pc[i])
    print("neg",neg_count)
    print("pos",pos_count)
    print("zero",zero_count)
    neg_pc=np.stack(neg_pc)
    pos_pc=np.stack(pos_pc)
    zero_pc=np.stack(zero_pc)
    visualize_pc(neg_pc)
    visualize_pc(pos_pc)
    visualize_pc(zero_pc)
    '''

    pc_c=list()
    pc=euc2hom(pc) # nx4
    h,w=depth.shape[0],depth.shape[1]
    for i in range(pc.shape[0]):
        #print(pc[i])
        pc[i]=np.matmul(vc,pc[i])
        if pc[i][2]>0:
            #if pc[i][2] != 0:
            cp=pc[i]
            d=cp[2]
            ip=np.matmul(proj,cp)
            d=ip[2]
            ip=ip/ip[2]
            ip[2]=d
            #ip[2]=pc[i][2]
            pc_c.append(ip)
    pc_c=np.stack(pc_c)
    print(pc_c.shape)
    count=0
    for r in depth:
        for e in r:
            if e>0:
                count+=1
    print(count)
    print(np.min(depth))
    #pc_corig=pc_c.copy()
    #pc_c=hom2euc(pc_c) # nx2

    count=0
    diffs=list()
    depth2=np.zeros_like(depth)
    for i in range(pc_c.shape[0]):
        p=pc_c[i]
        #print(p)
        x,y=int(round(p[0])),int(round(p[1]))
        #print(x,y)
        if 0<=x and 0<=y and x<w and y<h:
            depth2[y,x]=p[2]
        if 2<=x and 2<=y and x<w-2 and y<h-2:
            if np.max(depth[y-2:y+3,x-2:x+3])==0:
                continue
            #dep=np.sum(depth[y-1:y+2,x-1:x+2])/float(np.count_nonzero(depth[y-1:y+2,x-1:x+2]))
            d=np.min(np.abs(depth[y-2:y+3,x-2:x+3]-p[2]))
            #if abs(d)>2 and abs(d-p[2])<0.001:
            if abs(d)>5:
                count+=1
                depth2[y,x]=512
                #print(d,p[2],depth[y-2:y+3,x-2:x+3])
            diffs.append(d)
            #print("diff:",abs(d))
        #print()
    '''
    '''
    print(len(diffs))
    print(count)
    print(min(diffs))
    print(max(diffs))
    print(mean(diffs))
    print(var(diffs))
    cv2.imshow('img',img)
    cv2.imshow('lol2',depth2/200.0)
    cv2.waitKey(0)

def test_reverse(idx):
    pc=load_pc("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/velodyne_points/data/"+'{:010d}'.format(idx)+".bin")
    vcR,vcT=load_calib("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/calib/calib_velo_to_cam.txt")
    vc=padRT(vcR,vcT)
    proj=load_proj("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/calib/calib_cam_to_cam.txt")
    proj_=np.zeros((4,4),np.float64)
    proj_[:3]=proj
    proj_[3,3]=1
    proj=proj_
    vci=np.matmul(proj,vc)

    point=np.ones((4,),dtype=np.float64)
    point[:3]=pc[0]
    print(point)
    point_c=np.matmul(vc,point)
    print("point_c",point_c)
    depth=point_c[2]
    print("depth:",depth)
    point_i=np.matmul(proj,point_c)
    print("point_i",point_i)
    point_i/=point_i[2]
    print("point_i",point_i)
    point_i*=depth
    point_i[3]=1
    print("point_i",point_i)
    point_i=np.matmul(np.linalg.inv(proj),point_i)
    print("point_i",point_i)
    point_i=np.matmul(np.linalg.inv(vc),point_i)
    print("point_i",point_i)
    print()

    point_i=np.matmul(vci,point)
    print("point_i",point_i)
    point_i/=point_i[2]
    print("point_i",point_i)
    print(depth)
    point_i*=depth
    point_i[3]=1
    print("point_i",point_i)
    point_i=np.matmul(np.linalg.inv(vci),point_i)
    print(point_i)


def scatter_pixel(pixel,idx,h,w):
    '''
    pixel - 3 x N
    return img - 3 x h x w
    '''
    if idx.shape[0]==2:
        idx=idx.floor().long()
        idx=idx[0]+idx[1]*w
    else:
        idx=idx.view(-1).long()
    idx=idx.unsqueeze(0).expand(3,-1)
    img=pixel.new(3,h*w).zero_()
    img.scatter_(1,idx,pixel)
    img=img.view(3,h,w)
    return img

def convert_to_colormap(tensor,ret_tensor=True):
    '''
    tensor - 1 x h x w
    '''
    if tensor.shape[0]==1:
        tensor = tensor.cpu().detach().squeeze().numpy()
    if tensor.min()<=0:
        print("min <=0",tensor.min())
    tensor = np.log10(tensor)

    vmin = tensor.min()
    vmax = tensor.max()
    if vmin==vmax:
        print("vminmax",vmin,vmax)

    tensor = (tensor - vmin) / (vmax - vmin)

    cmapper = mpl.cm.get_cmap('magma')
    tensor = cmapper(tensor, bytes=True)

    img = tensor[:, :, :3] # h x w x 3
    if ret_tensor:
        img = transforms.ToTensor()(img) # 3 x h x w

    return img

def convert_to_colormap_batch(tensor):
    '''
    tensor - b x 1 x h x w
    '''
    ret = []
    for t in tensor:
        ret.append(convert_to_colormap(t).unsqueeze(0))
    ret=torch.cat(ret)
    return ret


if __name__ == '__main__':
    #download_kitti()
    #process_kitti_raw()
    #process_kitti_integ()
    #visualize_depth("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/depth/image_02/0000000136.png")
    #load_img("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/image/data/0000000136.png")
    #diffs=load_timediff("/home/chenziwe/robotics/egoDepth/datasets/kitti/2011_09_26_drive_0022_sync/image/timestamps.txt")
    '''
    '''
    dx=list()
    dy=list()
    dz=list()
    for i in range(100):
        i+250
        x,y,z=get_diff_means(i,i,velocity=False)
        dx.append(x)
        dy.append(y)
        dz.append(z)
    print(mean(dx),mean(dy),mean(dz))
    print(var(dx),var(dy),var(dz))
    #integ_sequence('2011_09_26_drive_0022')
    #x,y,z=get_diff_means(0,400,velocity=True)
    #test_projection(5)
    #test_reverse(5)
