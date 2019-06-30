#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:59:52 2019

@author: yuriy
"""

import shutil
import os
import cv2
import struct
import scipy.misc
import numpy as np
from scipy.io import loadmat
from transforms3d import axangles

# Specify the FPS value and name of the port for GT. 
# Convention: /port_name/object_name:o
fps = 5
port_name = "object-tracking-ground-truth"
depth_factor_ycb = 10000.0

# Choose if you want to resize images and set new dimensions
resize_img = True
if resize_img:
    width = 320
    height = 240
    resize_dim = (width, height)

vid_num = 1
dataset = 'data/ycb_video/full_videos'
# One video in YCB Video dataset to be converted
video = os.path.join(dataset, "vid{}/".format(vid_num))
dest = '/home/yuriy/robot-code/data/vid{}/'.format(vid_num)

# Rewrite the destination directory
if not os.path.exists(dest):
    os.mkdir(dest)
else:
    shutil.rmtree(dest)
    os.mkdir(dest)

rgb_dir = os.path.join(dest, "rgb/")
depth_dir = os.path.join(dest, "depth/")

if not os.path.exists(rgb_dir):
    os.mkdir(rgb_dir)
if not os.path.exists(depth_dir):
    os.mkdir(depth_dir)

n_images = 0
for file_name in os.listdir(video):
    if file_name.endswith("color.png"):
        n_images += 1   


# Create info.log
rgb_info_file_path = rgb_dir + "info.log"
rgb_info_file = open(rgb_info_file_path,"w") 
rgb_info_file.write("Type: Image;\n")
rgb_info_file.write("[0.0] /depthCamera/rgbImage:o [connected]")
rgb_info_file.close()

depth_info_file_path = depth_dir + "info.log"
depth_info_file = open(depth_info_file_path,"w") 
depth_info_file.write("Type: Image;\n")
depth_info_file.write("[0.0] /depthCamera/depthImage:o [connected]")
depth_info_file.close()


# Create data.log
rgb_data_log_path = rgb_dir + "data.log"
rgb_data_log = open(rgb_data_log_path,"w")

depth_data_log_path = depth_dir + "data.log"
depth_data_log = open(depth_data_log_path,"w")


# Create GT directories for every object
text_file = open(video + '000001-box.txt')
line = text_file.readline()
object_gt_dirs = []
data_log_files = []
data_log_paths = []
i = 0
while line:
    object_name = line.split(" ")[0][4:]
    object_gt_dirs.append(os.path.join(dest, "gt_" + object_name))
    if not os.path.exists(object_gt_dirs[i]):
        os.mkdir(object_gt_dirs[i])
    data_log_paths.append(object_gt_dirs[i] + "/data.log")
    data_log_files.append(open(data_log_paths[i],"w"))
    
    info_log_path = object_gt_dirs[i] + "/info.log"
    object_info_file = open(info_log_path,"w") 
    object_info_file.write("Type: Bottle;\n")
    object_info_file.write("[0.0] /{}/{}:o [connected]".format(port_name, object_name))
    object_info_file.close()
    
    i = i + 1
    line = text_file.readline()

# Move color images to the output folder
for num in range(1,n_images + 1):
    image_name = "00000{}-color.png".format(num)
    depth_name = "00000{}-depth.png".format(num)
    new_name = "00000{}.png".format(num)
    new_name_depth = "00000{}.float".format(num)
    fileobj = open(depth_dir + new_name_depth[-12:], mode='wb')
    image_path = os.path.join(video, image_name[-16:])
    depth_path = os.path.join(video, depth_name[-16:])
    
    if resize_img:
        
        r = cv2.imread(image_path)
        resized_rgb = cv2.resize(r,resize_dim)
        cv2.imwrite(rgb_dir + new_name[-10:], resized_rgb)
        
        d = scipy.misc.imread(depth_path,mode='F')
        resized_depth = np.zeros((height, width),dtype='float32')
        
        # Resize depth by skipping every second element
        for i in range(0, height):
            for j in range(0, width):
                resized_depth[i,j] = d[i*2,j*2]
        
        depth_float = (resized_depth/depth_factor_ycb).reshape(1,-1)
        
        fileobj.write(struct.pack("Q", width))
        fileobj.write(struct.pack("Q", height))
        s = struct.pack('f'*len(depth_float[0,:]), *depth_float[0,:])
        fileobj.write(s)
        fileobj.close()
    
    else:
        shutil.copy2(image_path, rgb_dir)
        os.rename(rgb_dir+image_name[-16:], rgb_dir+new_name[-10:])
        
        d = scipy.misc.imread(depth_path,mode='F')
        depth_float = ((d.astype('float32'))/depth_factor_ycb).reshape(1,-1)
        fileobj.write(struct.pack("Q", d.shape[1]))
        fileobj.write(struct.pack("Q", d.shape[0]))
        s = struct.pack('f'*len(depth_float[0,:]), *depth_float[0,:])
        fileobj.write(s)
        fileobj.close()
    
    # Fill the data.log line
    time_step = (num-1)/fps
    rgb_data_log.write("{} {} {} [rgb]\n".format(num-1,time_step,new_name[-10:]))
    depth_data_log.write("{} {} {} [dec]\n".format(num-1,time_step,new_name_depth[-12:]))
    
    # Compute ground truth and fill data.log files
    mat_file = "00000{}-meta.mat".format(num)
    gt_meta = loadmat(video + mat_file[-15:])
    poses = gt_meta['poses']
    for i in range(poses.shape[-1]):
        rot_mat = poses[:,:3,i]
        axis_angles = axangles.mat2axangle(rot_mat)
        x = poses[0,3,i]
        y = poses[1,3,i]
        z = poses[2,3,i]
        ax = axis_angles[0][0]
        ay = axis_angles[0][1]
        az = axis_angles[0][2]
        theta =  axis_angles[1]
        data_log_files[i].write("{} {} {} {} {} {} {} {} {}\n".format(num-1,time_step,x,y,z,ax,ay,az,theta))


rgb_data_log.close()
depth_data_log.close()
for i in range(poses.shape[-1]):
    data_log_files[i].close()