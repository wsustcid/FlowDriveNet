'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-06-28 11:01:02
@LastEditTime: 2020-09-28 22:06:02
@Description:  
'''

'''
PointNet version 1 Model
Reference: https://github.com/charlesq34/pointnet
'''

import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


class PointNet(object):
    def __init__(self, num_point):

        self.num_point = num_point
        self.output_dim = 1


    def get_inputs_pl(self, batch_size):
        points_pl = tf.placeholder(tf.float32, shape=(batch_size, self.num_point, 3))
        label_pl = tf.placeholder(tf.float32, shape=(batch_size, self.output_dim))

        return points_pl, label_pl


    def get_model(self, points_pl, is_training, bn_decay=None):
        """ Regression PointNet, input is BxNx3, output Bx1 """
        input_image = tf.expand_dims(points_pl, -1) # (B,N,3,1)
        
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [self.num_point, 1],
                                padding='VALID', scope='maxpool') # (B,1,1,1024)
        
        # MLP on global point cloud vector
        #net = tf.reshape(net, [batch_size, -1])
        net = tf.squeeze(net, axis=[1,2]) # (B,1024)
        
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                    scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                    scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp1')
        net = tf_util.fully_connected(net, self.output_dim, activation_fn=None, scope='fc3')

        return net


    def get_loss(self, pred, label, name='Loss'):
        """ Return mse loss for a batch of data
        """
        loss = tf.reduce_mean(tf.square(tf.subtract(pred, label)))
        
        tf.summary.scalar(name, loss)
        
        return loss


    def get_rmse(self,pred, label, name='rmse'):
        """Return rmse as evalation metrics for a batch of data
        """
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, label))))
        
        tf.summary.scalar(name, rmse)

        return rmse

if __name__=='__main__':
    with tf.Graph().as_default():
        batch_size = 16
        is_training = tf.constant(True)
        
        model = PointNet(num_point=20000)

        points_pl, label_pl = model.get_inputs_pl(batch_size)
        pred = model.get_model(points_pl, is_training)
        loss = model.get_loss(pred, label_pl)
        print(pred)
        print(loss)
        tf_util.model_summary()
        # Total size of variables: 809601
        # Total bytes of variables: 3238404
