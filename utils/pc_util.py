'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-06-18 23:32:49
@LastEditTime: 2020-08-07 21:40:49
@Description:  
'''

import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import glob
import numpy as np 
import open3d as o3d
import pcl
import pcl.pcl_visualization
import pandas as pd

from pypcd import pypcd
import pprint # for pretty print the meta data of the PCD
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle


# ----------------------------------------
# Load pcd file and visualize clouds
# The time used for loadng 1000 cloud:
#   - pypcd: 165s
#   - open3d: 18s
#   - np.fromfile: 5.6s
# ----------------------------------------

def load_pcd_pypcd(filepath):
    """
    load point cloud from pcd file use pypcd
    show point cloud in scatter

    - The getadata of the cloud saved in pcd file:
        {'count': [1, 1, 1, 1, 1, 1],
        'data': 'ascii',
        'fields': ['x', 'y', 'z', 'intensity', 'ring', 'time'],
        'height': 1,
        'points': 23633,
        'size': [4, 4, 4, 4, 2, 4],
        'type': ['F', 'F', 'F', 'F', 'U', 'F'],
        'version': '0.7',
        'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'width': 23633}
    """
    cloud = pypcd.PointCloud.from_path(filepath)

    pprint.pprint(cloud.get_metadata())

    # access the point cloud as a numpy structured array
    print("Raw shape: {}".format(cloud.points))
    print(cloud.pc_data[:5])
    
    new_cloud = np.stack([cloud.pc_data['x'], cloud.pc_data['y'], cloud.pc_data['z']], axis=1)

    print("New shape: {}".format(new_cloud.shape))
    print(new_cloud[:5])
    

    fig = plt.figure(figsize=(30,30))  # create a figure object
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(new_cloud[:,0],new_cloud[:,1], new_cloud[:,2])
    ax.axis('scaled')
    ax.set_zlabel('Z') 
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()

def load_pcd_o3d(file_path):
    """ load and visualize pcd file use o3d

    Hint:
      1. some useful methods in o3d.geometry (http://www.open3d.org/docs/release/python_api/open3d.geometry.html)
        - o3d.geometry.PointCloud.rotate() scale, transform, translate
        - o3d.geometry.voxel_down_sample()
        - o3d.geometry.compute_point_cloud_mean_and_covariance
        - o3d.geometry.crop_point_cloud()
    """

    cloud = o3d.io.read_point_cloud(file_path) # return a o3d.geometry.PointCloud
    print(cloud.get_max_bound())
    print(cloud.get_min_bound())
    o3d.visualization.draw_geometries([cloud])
    
    cloud_np = np.asarray(cloud.points, np.float32)
    print(cloud_np.shape)
    print(np.max(cloud_np, axis=0))
    print(np.min(cloud_np, axis=0))

def load_pcd_pcl(file_path):
    # load
    cloud_pcl = pcl.load(file_path)
    cloud_np = np.asarray(cloud_pcl, np.float32)
    print(cloud_np.shape)
    #  or convert narray to pcl cloud
    #cloud_pcl = pcl.PointCloud()
    #cloud_pcl.from_array(cloud_np)
    
    # show cloud
    visual = pcl.pcl_visualization.CloudViewing()
    #visual.ShowGrayCloud(cloud_pcl, b'cloud')
    #visual.ShowColorACloud()
    #visual.ShowColorCloud()
    for i in range(1):
        visual.ShowMonochromeCloud(cloud_pcl, b'cloud')
        time.sleep(100)
    #if visual.WasStopped():
    
def load_cloud_seq(label_path):
    data = pd.read_csv(label_path)
    clouds_name = data['point_filename'].values
    num_cloud = len(clouds_name)
    print("Total cloud: {}".format(num_cloud))

    path_prefix = os.path.dirname(label_path)
    visual = pcl.pcl_visualization.CloudViewing()
    for i in range(num_cloud):
        cloud_path = os.path.join(path_prefix, 'points', clouds_name[i])
        cloud_pcl = pcl.load(cloud_path)
        visual.ShowMonochromeCloud(cloud_pcl, b'cloud')
        print(cloud_path)

        cloud_np  = np.asarray(cloud_pcl, np.float32)
        #print('Max range: {}; MIn range: {}'.format(np.max(cloud_np, axis=0),
        #                                            np.min(cloud_np, axis=0)))

        
        time.sleep(0.01)
        #if visual.WasStopped():

def load_cloud_bin_seq(label_path):
    data = pd.read_csv(label_path)
    clouds_name = data['point_filename'].values
    num_cloud = len(clouds_name)
    print("Total cloud: {}".format(num_cloud))

    path_prefix = os.path.dirname(label_path)
    cloud_pcl = pcl.PointCloud()
    visual = pcl.pcl_visualization.CloudViewing()
    for i in range(num_cloud):
        cloud_path = os.path.join(path_prefix, 'points_bin', clouds_name[i][:-3]+'bin')
        cloud_np = np.fromfile(cloud_path, np.float32).reshape(-1,3)
        cloud_pcl.from_array(cloud_np)
        visual.ShowMonochromeCloud(cloud_pcl, b'cloud')
        print(cloud_path)

        time.sleep(0.1)
        #if visual.WasStopped():

def cloud_crop(file_path):
    """ Crop the pointcloud and visulaize it
     
    Hint: o3d.geometry.crop_point_cloud()
    Args:
        input (open3d.geometry.PointCloud): The input point cloud.  
        min_bound (numpy.ndarray[float64[3, 1]]): Minimum bound for point coordinate  
        max_bound (numpy.ndarray[float64[3, 1]]): Maximum bound for point coordinate  
    Returns: 
        open3d.geometry.PointCloud
    """
    # show origin cloud
    cloud = o3d.io.read_point_cloud(file_path)
    print("Origin Points: ", cloud.dimension)
    print("Origin Bound: ", cloud.get_min_bound(), cloud.get_max_bound())
    o3d.visualization.draw_geometries([cloud])
    
    # crop
    min_bound = np.array([-70, -40, -5])
    max_bound = np.array([70,40,15])
    cloud_crop = o3d.geometry.crop_point_cloud(cloud, min_bound,max_bound)
    
    # show cropped cloud
    cloud_np = np.asarray(cloud_crop.points, np.float32)
    print("Cropped Points: ", cloud_np.shape)
    print("Cropped Bound: ", np.min(cloud_np, axis=0), np.max(cloud_np, axis=0))
    o3d.visualization.draw_geometries([cloud_crop])

def clouds_info(train_path, test_path,
                show_distribute=False, 
                play_clouds=False):
    """ plot pointclouds distribution in train set and test set

    1. plot original point set distribution
    2. get the proper crop bounds
    3. crop the point and visualize the sequential cloud
    """
    # load train and test clouds file path
    train_files = glob.glob(os.path.join(train_path, '*/center.csv'))
    test_file   = os.path.join(test_path, 'center.csv')
    
    clouds_path_train = []
    for train_file in train_files:     
        clouds_name = pd.read_csv(train_file)['point_filename'].values
        path_prefix = os.path.dirname(train_file)
        for cloud_name in clouds_name:
            cloud_path = os.path.join(path_prefix, 'points', cloud_name)
            clouds_path_train.append(cloud_path)
    num_train = len(clouds_path_train)

    clouds_path_test = []
    clouds_name = pd.read_csv(test_file)['point_filename'].values
    path_prefix = os.path.dirname(test_file)
    for cloud_name in clouds_name:
        cloud_path = os.path.join(path_prefix, 'points', cloud_name)
        clouds_path_test.append(cloud_path)
    num_test = len(clouds_path_test)

    print("Total train: ", num_train)
    print("Total test: ", num_test)

    # get clouds bound distribution
    if show_distribute == True:
        x_bound, y_bound, z_bound = [], [], []
        for i in range(num_test):
            cloud = o3d.io.read_point_cloud(clouds_path_test[i])
            x_bound.append(cloud.get_min_bound()[0])
            x_bound.append(cloud.get_max_bound()[0])

            y_bound.append(cloud.get_min_bound()[1])
            y_bound.append(cloud.get_max_bound()[1])

            z_bound.append(cloud.get_min_bound()[2])
            z_bound.append(cloud.get_max_bound()[2])
        
        fig, axes = plt.subplots(1,3,figsize=(12,2),dpi=300)
        axes[0].hist(x_bound, rwidth=0.8)
        axes[1].hist(y_bound, rwidth=0.8)
        axes[2].hist(z_bound, rwidth=0.8)
        plt.tight_layout()
        #plt.show()
        fig.savefig(os.path.join(BASE_DIR,'../docs/figs/clouds_bound_test.png'))

        x_bound, y_bound, z_bound = [], [], []
        for i in range(num_train):
            cloud = o3d.io.read_point_cloud(clouds_path_train[i])
            x_bound.append(cloud.get_min_bound()[0])
            x_bound.append(cloud.get_max_bound()[0])

            y_bound.append(cloud.get_min_bound()[1])
            y_bound.append(cloud.get_max_bound()[1])

            z_bound.append(cloud.get_min_bound()[2])
            z_bound.append(cloud.get_max_bound()[2])
        
        fig, axes = plt.subplots(1,3,figsize=(12,2),dpi=300)
        axes[0].hist(x_bound, rwidth=0.8)
        axes[1].hist(y_bound, rwidth=0.8)
        axes[2].hist(z_bound, rwidth=0.8)
        plt.tight_layout()
        #plt.show()
        fig.savefig(os.path.join(BASE_DIR,'../docs/figs/clouds_bound_train.png')) 

    # paly clouds
    # crop # [-100,-75,-10] [100,75,30]
    min_bound = np.array([-70, -40, -10])
    max_bound = np.array([70,40,20])
    
    if play_clouds == True:
        visual = pcl.pcl_visualization.CloudViewing()
        cloud_pcl = pcl.PointCloud()
        for i in range(num_train):
            cloud = o3d.io.read_point_cloud(clouds_path_train[i])
            cloud_crop = o3d.geometry.crop_point_cloud(cloud, min_bound,max_bound)
            
            cloud_np = np.asarray(cloud.points, np.float32)
            
            cloud_pcl.from_array(cloud_np)
            visual.ShowMonochromeCloud(cloud_pcl, b'cloud')
            
            print('{}: Max range: {}; MIn range: {}'.format(cloud_np.shape[0], np.max(cloud_np, axis=0),
                                                        np.min(cloud_np, axis=0)))

            
            time.sleep(0.5)
    
def cloud_show(cloud):
    fig, axes = plt.subplots(1,2, figsize=(100,25))  # create a figure object
    axes[0].axis('equal')
    axes[0].scatter(cloud[:,0],cloud[:,1]) # BEV
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    axes[1].axis('equal')
    axes[1].scatter(cloud[:,0],cloud[:,2]) # side view
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    #fig.savefig(path)
    plt.show()

def cloud_show_3d(cloud):
    fig = plt.figure(figsize=(100,30))  # create a figure object
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(cloud[:,0],cloud[:,1], cloud[:,2])
    plt.show()


def cloud_plot_circle(cloud, radius=0.5):
    fig, axes = plt.subplots(1,2, figsize=(100,25))  # create a figure object
    axes[0].axis('equal')
    axes[1].axis('equal')
    
    axes[0].scatter(cloud[:,0],cloud[:,1]) # BEV
    axes[1].scatter(cloud[:,0],cloud[:,2]) # side view
    for i in range(len(cloud)):
        cir1 = Circle((cloud[i][0], cloud[i][1]), radius, color='r', fill=False)
        axes[0].add_patch(cir1)

        cir2 = Circle((cloud[i][0], cloud[i][2]), radius, color='r', fill=False)
        axes[1].add_patch(cir2)
    
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    #fig.savefig(path)
    plt.show()

def cloud_plot_circle_center(cloud, path, radius=10):
    fig, axes = plt.subplots(1,2, figsize=(100,25))  # create a figure object
    axes[0].axis('equal')
    axes[1].axis('equal')
    
    axes[0].scatter(cloud[:,0],cloud[:,1]) # BEV
    axes[1].scatter(cloud[:,0],cloud[:,2]) # side view
    
    cir1 = Circle((0,0), radius, color='r', fill=False)
    axes[0].add_patch(cir1)

    cir2 = Circle((0,0), radius, color='r', fill=False)
    axes[1].add_patch(cir2)

    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    fig.savefig(path)

if __name__ == '__main__':
    train_path = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_002'
    test_path = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_001'
    
    file_name = 'points/1479425440130861.pcd'
    #load_pcd_pypcd(os.path.join(test_path,file_name))
    #load_pcd_o3d(os.path.join(data_path,file_name))
    #load_pcd_pcl(os.path.join(data_path,file_name))
    #load_cloud_seq(os.path.join(data_path, 'center.csv'))
    #cloud_crop(os.path.join(data_path,file_name))

    #clouds_info(train_path, test_path, show_distribute=False, play_clouds=True)

    load_cloud_bin_seq(os.path.join(test_path, 'center.csv'))
    