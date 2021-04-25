'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-07-30 17:01:12
@LastEditTime: 2020-10-06 21:54:58
@Description:  
'''

import os
import sys
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(os.path.join(root_dir, 'utils'))
sys.path.append(os.path.join(root_dir, 'tf_ops/sampling'))
sys.path.append(os.path.join(root_dir, 'tf_ops/grouping'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point

import tf_util


class FlowDriveNet(object):
    """ An End-to-End driving model that learns driving policies 
        from both optical flow and point flow.
    """
    def __init__(self, input_cfg, model_cfg, height, width, seq_len, num_point):
        '''
         - supported input type: GRAY, GRAYF, GRAYF-T, XYZ, XYZF, XYZF-T, GRAYF-XYZF-T
         - supported model type: VFE, VFE-TFP, PFE, PFE-TFP, VFE-PFE-TFP
        '''
        self.input_cfg = input_cfg # to create input placeholder
        self.model_cfg = model_cfg # defines module combination

        # Args for input placeholder
        self.image_height = height
        self.image_width = width
        self.seq_len = seq_len
        self.num_point = num_point
        self.output_dim = 2

        # Args for VFE
        self.__n_filters = 32 # num. of filters for 3x3 conv
        self.__reduction = 0.5 # reduction rate in transition layer
        self.__n_blocks = [4,4,4] # num. of layers in each dense_block
        #self.__vfe_drop_rate = 0.0

        # Args for PFE
        self.__sa1_npoint = 1024
        self.__sa1_radius = 2.5
        self.__sa1_nsample= 32
        self.__sa1_mlp = [64,64,128]
        
        self.__sa2_npoint = 256
        self.__sa2_radius = 5.0
        self.__sa2_nsample= 64
        self.__sa2_mlp = [128, 128, 256]

        self.__sa3_mlp = [256, 512] # adjust by experiments
        #self.pfe_drop_rate = 0.5 
        
    
    def VFE_Module(self, input_image, is_training):
        ''' A Densely Connected Visual Feature Extraction Module
        '''
        def bottleneck_layer(X, scope):
            ''' The bottleneck layer used in dense_block
            '''
            with tf.name_scope(scope):
                X = tf_util.Batch_Normalization(X, is_training=is_training, scope=scope+'_batch1')
                X = tf.nn.relu(X)
                X = tf.layers.conv2d(X, filters=4*self.__n_filters, kernel_size=[1,1], 
                                        strides=1, padding='SAME', name=scope+'_conv1')
                X = tf_util.Batch_Normalization(X, is_training=is_training, scope=scope+'_batch2')
                X = tf.nn.relu(X)
                X = tf.layers.conv2d(X, filters=self.__n_filters, kernel_size=[3,3], 
                                        strides=1, padding='SAME', name=scope+'_conv2')

                return X

        def dense_block(X, n_layers, layer_name):
            with tf.name_scope(layer_name):
                layers_concat = list()
                layers_concat.append(X)

                X = bottleneck_layer(X, scope=layer_name + '_bottleN_' + str(0))
                layers_concat.append(X)

                for i in range(n_layers - 1):
                    X = tf.concat(layers_concat, axis=3)
                    X = bottleneck_layer(X, scope=layer_name + '_bottleN_' + str(i + 1))
                    layers_concat.append(X)

                X = tf.concat(layers_concat,axis=3) #  (B,H,W,C)

                return X

        def transition_layer(X, scope):
            with tf.name_scope(scope):
                X = tf_util.Batch_Normalization(X, is_training=is_training, scope=scope+'_batch1')
                X = tf.nn.relu(X)
                in_channel = X.get_shape().as_list()[-1]
                X = tf.layers.conv2d(X, filters=in_channel*self.__reduction, 
                                        kernel_size=[1,1], strides=1, padding='VALID',
                                        name=scope+'_conv1')
                X = tf.layers.average_pooling2d(X, pool_size=[2,2], 
                                                   padding='VALID', strides=2)

                return X

        ## input -> [dense_block - transition_layer] - [dense_block - transition_layer] 
        X = tf.layers.conv2d(input_image, 
                             filters=2*self.__n_filters, 
                             kernel_size=[7,7], strides=2, 
                             padding='VALID', name='conv0')
        X = tf.layers.max_pooling2d(X, pool_size=[2,2], padding='VALID',strides=2)
        # Dense_Net
        for i in range(len(self.__n_blocks)-1) :
            X = dense_block(X, n_layers=self.__n_blocks[i], layer_name='dense_'+str(i))
            X = transition_layer(X, scope='trans_'+str(i))

        X = dense_block(X, n_layers=self.__n_blocks[-1], layer_name='dense_final')
        X = tf_util.Batch_Normalization(X, is_training=is_training,
                                           scope='final_batch')
        X = tf.nn.relu(X) # (B, 12,12,240)
        #print(X.get_shape().as_list())
        X = tf.layers.average_pooling2d(X, pool_size=X.get_shape().as_list()[1:3],
                                           strides=1, name='global_average_pooling')
        # (B,1,1,240)
        
        return X

    def PFE_Module(self, point_cloud, is_training, bn_decay=None):
        ''' PointNet-based Point Feature Extraction) Module
        '''
        def sample_and_group(npoint, radius, nsample, xyz, points, 
                             knn=False, use_xyz=True):
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
                new_points = grouped_xyz # if points is None, use grouped_xyz as feature

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


        def pointnet_sa_module(xyz, points, npoint, radius, nsample, 
                              mlp, mlp2, group_all, is_training, bn_decay, 
                              scope, bn=True, pooling='max', knn=False, 
                              use_xyz=True, use_nchw=False):
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

                # final size for new_points is (B, 1, 1024)
                return new_xyz, new_points
        
        ## (B,N,3) or (B,N,4)
        if point_cloud.get_shape().as_list()[-1] == 3:
            l0_xyz = point_cloud
            l0_points = None
        elif point_cloud.get_shape().as_list()[-1] == 4:
            l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
            l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,1])

        # Set Abstraction layers
        l1_xyz, l1_points = pointnet_sa_module(l0_xyz, l0_points, npoint=self.__sa1_npoint,radius=self.__sa1_radius, nsample=self.__sa1_nsample, mlp=self.__sa1_mlp, mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points = pointnet_sa_module(l1_xyz, l1_points, npoint=self.__sa2_npoint,radius=self.__sa2_radius, nsample=self.__sa2_nsample, mlp=self.__sa2_mlp, mlp2=None,group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=self.__sa3_mlp, mlp2=None, group_all=True, is_training=is_training,bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        X = tf.reshape(l3_points, (-1, self.__sa3_mlp[-1])) # (B,1,C) -> (B,C)

        X = tf.layers.dense(X, 240, activation=tf.nn.relu, name='PFE_fc1')

        return X

    def TFP_Module(self, X, hidden_size, scope):
        ''' LSTM-based Temporal Feature Fusion Module
         - input: (B, T, C)
         - output: (B,H)
        '''
        X = tf_util.lstm(X, hidden_size=hidden_size, scope=scope)

        return X


    def get_inputs_pl(self, batch_size):
        ''' Create and return input and label placeholders
        return: image_pl, points_pl, label_pl
        '''

        if self.input_cfg in ['GRAY']:
            image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.image_height,self.image_width,1))
            points_pl = tf.placeholder(tf.float32, shape=(0,0,0))
            
        elif self.input_cfg in ['GRAYF']:
            image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.image_height,self.image_width, 3))
            points_pl = tf.placeholder(tf.float32, shape=(0,0,0))
            
        elif self.input_cfg in ['GRAYF-T']:
            image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.seq_len, self.image_height, self.image_width, 3))
            points_pl = tf.placeholder(tf.float32, shape=(0,0,0))
            
        elif self.input_cfg in ['XYZ']:
            image_pl = tf.placeholder(tf.float32, shape=(0,0,0))
            points_pl = tf.placeholder(tf.float32, shape=(batch_size, self.num_point,3))
        
        elif self.input_cfg in ['XYZF']:
            image_pl = tf.placeholder(tf.float32, shape=(0,0,0))
            points_pl = tf.placeholder(tf.float32, shape=(batch_size, self.num_point,4))
        
        elif self.input_cfg in ['XYZF-T']:
            image_pl = tf.placeholder(tf.float32, shape=(0,0,0))
            points_pl = tf.placeholder(tf.float32, shape=(batch_size, self.seq_len, self.num_point, 4))
            
        elif self.input_cfg in ['GRAYF-XYZF-T']:
            image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.seq_len, self.image_height, self.image_width, 3))
            points_pl = tf.placeholder(tf.float32, shape=(batch_size, self.seq_len, self.num_point, 4))
            
        label_pl = tf.placeholder(tf.float32, shape=(batch_size, self.output_dim))
                                                         
        return image_pl, points_pl, label_pl


    def get_model(self, image_pl, points_pl, is_training, bn_decay):
        ''' Create and return final driving model
        - return: model output
        '''
        # base model
        if self.model_cfg == ['VFE']:
            X = self.VFE_Module(image_pl, is_training) # (B,1,1,240)
            X = tf.squeeze(X, [1,2]) # (B,240)
        
        elif self.model_cfg == 'VFE-TFP':
            img_shape = image_pl.get_shape().as_list()[2:]
            X = tf.reshape(image_pl, (-1,*img_shape)) # (B*T,H,W,C)
            X = self.VFE_Module(X, is_training)
            X = tf.squeeze(X, [1,2]) # (B*T,240)
            X = tf.reshape(X, (-1, self.seq_len, X.get_shape().as_list()[-1])) # (B,T,240)
            X = self.TFP_Module(X, 128, scope='TFP') # (B,128)
            
        elif self.model_cfg == 'PFE':
            X = self.PFE_Module(points_pl, is_training, bn_decay)
        
        elif self.model_cfg == 'PFE-TFP':
            X = tf.reshape(points_pl, (-1, self.num_point, 4)) # (B*T,N,4)
            X = self.PFE_Module(X, is_training, bn_decay) # (B*T,C)
            
            X = tf.reshape(X, (-1, self.seq_len, X.get_shape().as_list()[-1]))
            X = self.TFP_Module(X, 128, scope='TFP')
           
        elif self.model_cfg == 'VFE-PFE-TFP':
            img_shape = image_pl.get_shape().as_list()[2:]
            X_img = tf.reshape(image_pl, (-1,*img_shape)) # (B*T,H,W,C)
            X_img = self.VFE_Module(X_img, is_training) # (B*T, 1,1,240)
            X_img = tf.squeeze(X_img, [1,2]) # (B*T,240)
            X_img = tf.reshape(X_img, (-1, self.seq_len, X_img.get_shape().as_list()[-1])) # (B,T,240)
            X_img = self.TFP_Module(X_img, 128, scope='img_tfp') # (B, 128)
            
            X_points = tf.reshape(points_pl, (-1, self.num_point, 4))
            X_points = self.PFE_Module(X_points, is_training, bn_decay) # (B*T, 240)   
            X_points = tf.reshape(X_points, (-1, self.seq_len, X_points.get_shape().as_list()[-1]))
            X_points = self.TFP_Module(X_points, 128, scope='point_tfp') # (B, 128)
                
            X = tf.concat([X_img, X_points], axis=-1) # (B, 256)
            X = tf.layers.dense(X, 128, activation=tf.nn.relu, name='fc1')
            X = tf.layers.dense(X, 32, activation=tf.nn.relu, name='fc2')
        else:
            raise TypeError
            
        # Prediction Module
        output = tf.layers.dense(X, units=self.output_dim, name='output')

        return output
        '''
        elif self.model_cfg == 'VFE-PFE-TFP-0':
            img_shape = image_pl.get_shape().as_list()[2:]
            X_img = tf.reshape(image_pl, (-1,*img_shape)) # (B*T,H,W,C)
            X_img = self.VFE_Module(X_img, is_training)
            X_img = tf.squeeze(X_img, [1,2]) # (B*T,240)
            #X_img = tf.reshape(X_img, (-1, self.seq_len, X_img.get_shape().as_list()[-1])) # (B,T,240)
            
            X_points = tf.reshape(points_pl, (-1, self.num_point, 4))
            X_points = self.PFE_Module(X_points, is_training, bn_decay) # (B*T, 240)   
            #X_points = tf.reshape(X_points, (-1, self.seq_len, X_points.get_shape().as_list()[-1]))
                
            X = tf.concat([X_img, X_points], axis=-1) # (B*T, 480)
            #print(X.get_shape().as_list())
            X = tf.layers.dense(X, 240, activation=tf.nn.relu, name='p_fusion')
            X = tf.reshape(X, (-1, self.seq_len, 240))
            X = self.TFP_Module(X, 128, scope='TFP')

        elif self.model_cfg == 'VFE-PFE-TFP-2':
            img_shape = image_pl.get_shape().as_list()[2:]
            X_img = tf.reshape(image_pl, (-1,*img_shape)) # (B*T,H,W,C)
            X_img = self.VFE_Module(X_img, is_training) # (B*T, 1,1,240)
            X_img = tf.squeeze(X_img, [1]) # (B*T,1, 240)
            X_img = tf.expand_dims(X_img, axis=-1) # (B*T, 1,240,1)

            X_points = tf.reshape(points_pl, (-1, self.num_point, 4))
            X_points = self.PFE_Module(X_points, is_training, bn_decay) # (B*T, 240)   
            X_points = tf.reshape(X_points, (-1, 1, 240, 1))

            X = tf.concat([X_img, X_points], axis=-1) # (B*T, 1, 240,2)
            X = tf.layers.conv2d(X, 16, [1,3], strides=[1,1], padding='valid', activation=tf.nn.relu, name='spf1')
            X = tf.layers.conv2d(X, 1, [1,3], strides=[1,1], padding='valid', activation=tf.nn.relu, name='spf2') # (B*T, 1, 236,1)
            X = tf.reshape(X, (-1, self.seq_len, X.get_shape().as_list()[2]))
            X = self.TFP_Module(X, 128, scope='point_tfp') # (B, 128)
            X = tf.layers.dense(X, 32, activation=tf.nn.relu, name='fc2')
        ''' 


    def get_loss(self, pred, label, batch, cfg, name='Loss'):
        ''' Return loss for a batch of data
        cfg: weighted; exp; step;
        train_vars = tf.trainable_variables()
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in train_vars[1:] if 'biases' not in v.name])
        
        loss_angle = tf.reduce_mean(tf.square(tf.subtract(pred[:,0], label[:,0])))
        loss_speed = tf.reduce_mean(tf.square(tf.subtract(pred[:,1], label[:,1])))

        if cfg == 'weighted':
            alpha = 1.0
            beta = 1.0
            loss = alpha*loss_angle + beta*loss_speed
            
        elif cfg == 'step':
            n_samples = 16100
            epoch0 = 15
            epoch = batch * self.batch_size/n_samples

            decay_step = 0.05
            decay_exp = 0.7
            beta = tf.maximum(tf.cast(0, tf.float32), tf.cast(1-tf.exp(-decay_step*(epoch-epoch0)), tf.float32))
            loss = loss_angle + beta*loss_speed
        
        elif cfg == 'exp':
            beta = 1.0-tf.pow(tf.cast(decay_exp, tf.float32), tf.cast(epoch/epoch0, tf.float32))
            loss = loss_angle + beta*loss_speed
        tf.summary.scalar(name+'_angle', loss_angle)
        tf.summary.scalar(name+'_speed', loss_speed)
        tf.summary.scalar(name+'_beta', beta)
        #tf.summary.scalar(name+'_l2', l2_weight*loss_l2)
        '''
        
        if cfg == "MSE":
            loss = tf.reduce_mean(tf.square(tf.subtract(pred, label)))

        tf.summary.scalar(name, loss)
        
        return loss


    def get_rmse(self, pred, label, name='rmse'):
        """Return rmse as evalation metrics for a batch of data
        """        
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, label)), axis=0))
        
        tf.summary.scalar(name+'_angle', rmse[0])
        tf.summary.scalar(name+'_speed', rmse[1])

        return rmse[0], rmse[1]

    def get_acc(pred, label, delta_a=5.0, delta_s=1.0, name='acc'):
        """
        - delta_a: tolerance of angle: 5 degrees
        - delta_s: tolerance of speed: 1 ft/s = 0.3048 m/s = 1.09728 km/h
        """
        acc_a = tf.abs(tf.subtract(pred[:, 0], label[:, 0])) < (delta_a/180*np.pi)
        acc_a = tf.reduce_mean(tf.cast(acc_a,tf.float32))
        
        acc_s = tf.abs(tf.subtract(pred[:, 1], label[:, 1])) < delta_s
        acc_s = tf.reduce_mean(tf.cast(acc_s, tf.float32))

        tf.summary.scalar(name+'_angle', acc_a)
        tf.summary.scalar(name+'_speed', acc_s)
        
        return acc_a, acc_s

    

if __name__ == '__main__':
    import argparse

    ## ==== Model evaluation Procedure ==== ## 
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', default='1', help='set 1-7')

    Flags = parser.parse_args()
    with tf.Graph().as_default():
        is_training = tf.cast(True, tf.bool)
        batch_size = 32

        if Flags.test_mode == '1':
            model = FlowDriveNet(input_cfg='GRAY', model_cfg='VFE', 
                                  height=200, width=200, seq_len=1)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 712402
            # Total bytes of variables: 2849608
        elif Flags.test_mode == '2':
            model = FlowDriveNet(input_cfg='GRAYF', model_cfg='VFE', 
                                  height=200, width=200, seq_len=1)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 718674
            # Total bytes of variables: 2874696
        elif Flags.test_mode == '3':
            model = FlowDriveNet(input_cfg='GRAYF-T', model_cfg='VFE-TFP', 
                                  height=200, width=200, seq_len=5)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 907378
            # Total bytes of variables: 3629512
        elif Flags.test_mode == '4':
            model = FlowDriveNet(input_cfg='XYZ', model_cfg='PFE', 
                                  height=200, width=200, seq_len=1)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 403986
            # Total bytes of variables: 1615944
        elif Flags.test_mode == '5':
            model = FlowDriveNet(input_cfg='XYZF', model_cfg='PFE', 
                                  height=200, width=200, seq_len=1)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 404050
            # Total bytes of variables: 1616200
        elif Flags.test_mode == '6':
            model = FlowDriveNet(input_cfg='XYZF-T', model_cfg='PFE-TFP', 
                                  height=200, width=200, seq_len=5)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 592754;  2371016
        elif Flags.test_mode == '7':
            model = FlowDriveNet(input_cfg='GRAYF-XYZF-T', model_cfg='VFE-PFE-TFP', 
                                  height=200, width=200, seq_len=5, num_point=10000)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # 1536706; 6146824
        '''
        elif Flags.test_mode == '7':
            model = FlowDriveNet(input_cfg='GRAYF-XYZF-T', model_cfg='VFE-PFE-TFP', 
                                  height=200, width=200, seq_len=5)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # Total size of variables: 1426386 5705544
        elif Flags.test_mode == '9':
            model = FlowDriveNet(input_cfg='GRAYF-XYZF-T', model_cfg='VFE-PFE-TFP-2', 
                                  height=200, width=200, seq_len=5, num_point=10000)
            image_pl, points_pl, label_pl = model.get_inputs_pl(batch_size)
            pred = model.get_model(image_pl, points_pl, is_training, bn_decay=None)
            # 1312995; 5251980
        '''
        tf_util.model_summary()

