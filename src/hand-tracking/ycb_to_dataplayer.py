#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:37:22 2019

@author: yuriy
"""

import shutil
import os
import cv2

# Specify the FPS value
fps = 5

vid_num = 1
dataset = 'data/ycb_video/full_videos'
# One video in YCB Video dataset to be converted
video = os.path.join(dataset, "vid{}/".format(vid_num))
dest = '/home/yuriy/robot-code/data/vid{}/'.format(vid_num)

if not os.path.exists(dest):
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

# Move color images to the output folder
for num in range(1,n_images + 1):
    image_name = "00000{}-color.png".format(num)
    depth_name = "00000{}-depth.png".format(num)
    new_name = "00000{}.png".format(num)
    image_path = os.path.join(video, image_name[-16:])
    shutil.copy2(image_path, rgb_dir)
    os.rename(rgb_dir+image_name[-16:], rgb_dir+new_name[-10:])
    
    depth_path = os.path.join(video, depth_name[-16:])
    d = cv2.imread(depth_path)
    cv2.imwrite(depth_dir + new_name[-10:], d)
    
    # Fill the data.log line
    time_step = (num-1)/fps
    rgb_data_log.write("{} {} {} [rgb]\n".format(num-1,time_step,new_name[-10:]))

rgb_data_log.close()
shutil.copy2(rgb_data_log_path, depth_dir)