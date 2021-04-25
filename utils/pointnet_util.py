'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-06-18 23:32:49
@LastEditTime: 2020-08-07 22:47:58
@Description:  
'''

""" 
Find the proper parameters of the PointNet++ Layers applied on the Udacity data
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tools'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import open3d as o3d 
import pandas as pd
import matplotlib.pyplot as plt 

from pc_util import cloud_show, cloud_plot_circle, cloud_show_3d

from cloud_visualizer import Visualizer

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format) 
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def sa_module_visualize(xyz, npoint=1024, radius=2.5, nsample=32):
    """ 
    One Cloud Pipeline:
    Show origin points -> FPS -> show points -> group points -> show balls -> find proper npoint and radius
    
    #Two clouds Pipeline: 
    Find the center ball -> show ball -> find proper radius -> compute mean shift within the ball -> mean shift diff -> estimated velocity in x and y
    
    第一组sa参数的确定：
    1. npoint的确定：根据肉眼观察和经验, 设置为1024; 
    2. radius的确定：同过在采样点上画圆，看哪种尺度各圈能够相交：
       - 每个采样点在稠密点云内的体积为 120x60x15/1024=105 m³ 及占据4.7为边的立方体
       - 因此选取 radius=2.5的采样半径
       - 半径合适的话，每个采样球内的点不会有太多重复
    """
    vis = Visualizer(play_mode='manual')
    cloud = o3d.geometry.PointCloud()
    sess = tf.Session()
    # save origin cloud
    #cloud_show(xyz)
    
    # add batch dim
    xyz = np.reshape(xyz, (1,*xyz.shape))
    
    # FPS
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) #(batch, npoint, 3)

    # save
    #new_xyz_np = sess.run(new_xyz)
    #cloud_show(new_xyz_np[0])
    #cloud.points = o3d.utility.Vector3dVector(new_xyz_np[0])
    #vis.show([cloud])
    
    # plot circle
    #cloud_plot_circle(new_xyz_np[0], radius=radius)

    # group balls
    #idx, _ = query_ball_point(radius, nsample, xyz, new_xyz)
    #grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    #grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation
    
    npoint_2 = 1024
    new_new_xyz = gather_point(new_xyz, farthest_point_sample(npoint_2, new_xyz))
    idx_new, _ = query_ball_point(5.0, 64, new_xyz, new_new_xyz)
    grouped_new_xyz = group_point(new_xyz, idx_new)

    grouped_new_xyz_np = sess.run(grouped_new_xyz)

    for i in range(npoint_2):
        local_xyz = grouped_new_xyz_np[0,i,:,:]
        #cloud_show_3d(local_xyz)
        cloud.points = o3d.utility.Vector3dVector(local_xyz)
        vis.show([cloud])
    



if __name__ == '__main__':
    bin_file = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_002/HMB_1/points_bin/1479424215883691.bin'
    cloud = np.fromfile(bin_file).reshape((-1,4))
    xyz = cloud[:,:3]

    sa_module_visualize(xyz, npoint=4096, radius=2.5, nsample=32)














#########################################################
'''

def evalute_point_flow():
    """ 
    1. 验证FPS后的相邻两帧点云是否类似（因为及时点云的空间结构类似，点的顺序也有可能不同
       - 对应位置做差后相差很大
    2. 尝试先计算所有采样球质心的和向量，然后做差
       - 也不行，因为有可能有个球离得非常远，导致和向量很大，向量差为：[[-314.7649   501.73926  253.10706]]
    3. 计算车体15米半径范围内的球，得到的和向量应该比较能准确反应点的分布
       - 画出速度曲线
       - 利用均值向量差计算速度
       - 画出估计速度曲线 （x 方向是速度，y方向是转向）
       - 不看具体值，看趋势是否一致就可以
       - 结果：不太明显，跳动太大
    4. 可能是因为在选取球内点时还是随机的，增大了不确定性；采用KNN
    2. 在第二个SA层做差：采样点更少，效率更高；抽象层级更高，导航特征更为具体；已经有了feature,信息更丰富；
    3. 将 相邻两帧对应球间距离作为额外特征拼接到new_points 中，这样每个点既有球内距离，又有球的运动信息
    """
    npoint=1024
    radius=2.5
    nsample=32 
    center_radius=15
    center_point = np.array([[[0,0,0]]])

    # load steer and speed
    test_path = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_001/'
    test_file = os.path.join(test_path, 'center.csv')
    
    steers = pd.read_csv(test_file)['angle'].values
    speeds = pd.read_csv(test_file)['speed'].values
    clouds_name = pd.read_csv(test_file)['point_filename'].values

    clouds_path_test = []
    for cloud_name in clouds_name:
        cloud_path = os.path.join(test_path, 'points', cloud_name)
        clouds_path_test.append(cloud_path)
    num_test = len(clouds_path_test)
    

    sess = tf.Session()
    steers_es = []
    speeds_es = []
    num_show = 250
    for i in range(num_show):
        file_path_pre = clouds_path_test[i]
        file_path_next = clouds_path_test[i+1]

        cloud_pre = o3d.io.read_point_cloud(file_path_pre)
        xyz_0 = np.asarray(cloud_pre.points, np.float32) # (19391,3)
        xyz_0 = np.reshape(xyz_0, (1,*xyz_0.shape))
        new_xyz_0 = gather_point(xyz_0, farthest_point_sample(npoint, xyz_0))
        idx_0_center, pts_cnt_0_center = query_ball_point(center_radius, 2*nsample, new_xyz_0, center_point)
        grouped_xyz_0_center = group_point(new_xyz_0, idx_0_center) # (1, 1, nsample, 3)
        grouped_xyz_0_center_np = sess.run(grouped_xyz_0_center)
        mean_shift_0 = np.sum(grouped_xyz_0_center_np, axis=2)

        cloud_next = o3d.io.read_point_cloud(file_path_next)
        xyz_1 = np.asarray(cloud_next.points, np.float32) # (18735,3)
        xyz_1 = np.reshape(xyz_1, (1,*xyz_1.shape))
        new_xyz_1 = gather_point(xyz_1, farthest_point_sample(npoint, xyz_1))
        idx_1_center, pts_cnt_1_center = query_ball_point(center_radius, 2*nsample, new_xyz_1, center_point)
        grouped_xyz_1_center = group_point(new_xyz_1, idx_1_center) # (1, 1, nsample, 3)
        grouped_xyz_1_center_np = sess.run(grouped_xyz_1_center)
        mean_shift_1 = np.sum(grouped_xyz_1_center_np, axis=2)

        mean_shift_diff = mean_shift_0 - mean_shift_1

        steers_es.append(mean_shift_diff[0,0,0])
        speeds_es.append(mean_shift_diff[0,0,1])

    # plot
    fig1, (ax1,ax2) = plt.subplots(2,1, sharex=True, figsize=(15,4))
    ax1.plot(np.arange(num_show), steers[:num_show])
    ax2.plot(np.arange(num_show), steers_es)
    fig1.savefig(os.path.join(BASE_DIR, '../docs/figs/point_flow_steer.png')) 

    fig2, (ax11,ax22) = plt.subplots(2,1, sharex=True, figsize=(15,4))
    ax11.plot(np.arange(num_show), speeds[:num_show])
    ax22.plot(np.arange(num_show), speeds_es)
    fig2.savefig(os.path.join(BASE_DIR, '../docs/figs/point_flow_speed.png')) 

def evalute_point_flow_v2():
    """ 
    1. 验证FPS后的相邻两帧点云是否类似（因为及时点云的空间结构类似，点的顺序也有可能不同
       - 对应位置做差后相差很大
    2. 尝试先计算所有采样球质心的和向量，然后做差
       - 也不行，因为有可能有个球离得非常远，导致和向量很大，向量差为：[[-314.7649   501.73926  253.10706]]
    3. 计算车体10米半径范围内的球，得到的和向量应该比较能准确反应点的分布
       - 画出速度曲线
       - 利用均值向量差计算速度
       - 画出估计速度曲线 （x 方向是速度，y方向是转向）
       - 不看具体值，看趋势是否一致就可以
       - 结果：不太明显，跳动太大
    4. 可能是因为在选取球内点时还是随机的，增大了不确定性；采用KNN
    2. 在第二个SA层做差：采样点更少，效率更高；抽象层级更高，导航特征更为具体；已经有了feature,信息更丰富；
    3. 将 相邻两帧对应球间距离作为额外特征拼接到new_points 中，这样每个点既有球内距离，又有球的运动信息
    """
    npoint=1024
    radius=2.5
    nsample=32 
    center_radius=10
    center_point = tf.constant(0, dtype=tf.float32, shape=[1,1,3])

    # load steer and speed
    test_path = '/media/ubuntu16/Documents/Datasets/Udacity/CH2/CH2_001/'
    test_file = os.path.join(test_path, 'center.csv')
    
    steers = pd.read_csv(test_file)['angle'].values
    speeds = pd.read_csv(test_file)['speed'].values
    clouds_name = pd.read_csv(test_file)['point_filename'].values

    clouds_path_test = []
    for cloud_name in clouds_name:
        cloud_path = os.path.join(test_path, 'points', cloud_name)
        clouds_path_test.append(cloud_path)
    num_test = len(clouds_path_test)
    

    sess = tf.Session()
    steers_es = []
    speeds_es = []
    num_show = 250
    for i in range(num_show):
        file_path_pre = clouds_path_test[i]
        file_path_next = clouds_path_test[i+1]

        cloud_pre = o3d.io.read_point_cloud(file_path_pre)
        xyz_0 = np.asarray(cloud_pre.points, np.float32) # (19391,3)
        xyz_0 = np.reshape(xyz_0, (1,*xyz_0.shape))
        xyz_0 = tf.constant(xyz_0, tf.float32)
        _, idx_0_center = knn_point(3*nsample, xyz_0, center_point)
        grouped_xyz_0_center = group_point(xyz_0, idx_0_center) # (1, 1, nsample, 3)
        grouped_xyz_0_center_np = sess.run(grouped_xyz_0_center)
        mean_shift_0 = np.mean(grouped_xyz_0_center_np, axis=2)

        cloud_next = o3d.io.read_point_cloud(file_path_next)
        xyz_1 = np.asarray(cloud_next.points, np.float32) # (18735,3)
        xyz_1 = np.reshape(xyz_1, (1,*xyz_1.shape))
        xyz_1 = tf.constant(xyz_1, tf.float32)
        _, idx_1_center = knn_point(3*nsample, xyz_1, center_point)
        grouped_xyz_1_center = group_point(xyz_1, idx_1_center) # (1, 1, nsample, 3)
        grouped_xyz_1_center_np = sess.run(grouped_xyz_1_center)
        mean_shift_1 = np.mean(grouped_xyz_1_center_np, axis=2)

        mean_shift_diff = mean_shift_0 - mean_shift_1

        steers_es.append(mean_shift_diff[0,0,0])
        speeds_es.append(mean_shift_diff[0,0,1])

    # plot
    fig1, (ax1,ax2) = plt.subplots(2,1, sharex=True, figsize=(15,4))
    ax1.plot(np.arange(num_show), steers[:num_show])
    ax2.plot(np.arange(num_show), steers_es)
    fig1.savefig(os.path.join(BASE_DIR, '../docs/figs/point_flow_steer_v2.png')) 

    fig2, (ax11,ax22) = plt.subplots(2,1, sharex=True, figsize=(15,4))
    ax11.plot(np.arange(num_show), speeds[:num_show])
    ax22.plot(np.arange(num_show), speeds_es)
    fig2.savefig(os.path.join(BASE_DIR, '../docs/figs/point_flow_speed_v2.png')) 
'''