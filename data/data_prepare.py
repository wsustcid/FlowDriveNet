'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-09-07 21:08:55
@LastEditTime: 2020-09-09 15:19:37
@Description:  
'''
import os, sys
import time
import numpy as np
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '../tools/'))

from label_loading import load_image_path, load_image_path_all, load_clouds_path
from data_processor import ImageProcessor, CloudProcessor

def create_optical_flow(data_root, set_type, dtype):
    '''
    Processing Procedure:
     1. Load two consecutive image with grayscale format (uint8,480,640)
     2. compute the optical flow (float32,480,640)
     3. save the optical flow to bin file
    '''
    # load image path
    if set_type=='train':
        data_path = os.path.join(data_root, 'CH2_002')
        images_path = load_image_path_all(data_path, dtype)
    elif set_type == 'test':
        HMB_path = os.path.join(data_root, "CH2_001")
        images_path = load_image_path(HMB_path, dtype)
    
    imp = ImageProcessor()
    start = time.time()
    for i in range(len(images_path)):
        # load image
        if i == 0:
            pre_img_path = images_path[0]
            pre_img = imp.load_image(pre_img_path)

        end_img_path = images_path[i]
        end_img = imp.load_image(end_img_path) 

        # compute flow
        flow = imp.compute_flow(pre_img, end_img) 
        
        # save to bin
        flow_name = end_img_path.split('/')[-1]
        bin_path = os.path.join(os.path.dirname(end_img_path), '../'+dtype+'_bin', flow_name[:-4]+'.bin')
        imp.save_flow(flow, bin_path)
        
        pre_img = end_img
        
    end = time.time()
    print("Procedure Duration: {} s".format(end-start))
    

def create_point_flow(HMB_path):
    """
    Input: the HMB path 
    Pipeline:
      1. load point cloud from pcd file
      2. cloud process
      3. compute flow and save to .bin file
    """
    start = time.time()
    
    # load cloud path
    clouds_path = load_clouds_path(HMB_path)
    
    clp = CloudProcessor()
    
    for i in range(len(clouds_path)):
        if i == 0:
            cloud_pre_path = clouds_path[0]
            cloud_pre = o3d.io.read_point_cloud(cloud_pre_path)
            cloud_pre = clp.cloud_process(cloud_pre)

        cloud_path = clouds_path[i]
        cloud = o3d.io.read_point_cloud(cloud_path)
        cloud = clp.cloud_process(cloud)
        
        # save to bin
        cloud_name = cloud_path.split('/')[-1]
        bin_path = os.path.join(os.path.dirname(cloud_path), '../points_bin',cloud_name[:-4]+'.bin')
        clp.save_cloud_flow(cloud, cloud_pre, bin_path)

        cloud_pre = o3d.geometry.PointCloud()
        cloud_pre.points = cloud.points
    
    end = time.time()
    print("Procedure Duration: {} s".format(end-start))

if __name__ == '__main__':
    data_root = '/media/ubuntu16/Documents/Datasets/Udacity/CH2'

    create_optical_flow(data_root, set_type='train', dtype='right')

    
