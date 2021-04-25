'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-28 22:19:51
@LastEditTime: 2020-09-15 23:32:47
'''

import os
import sys
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..'))
from utils import tf_util

class PilotNet(object):
    def __init__(self, height=66, width=200):
        self.height = height
        self.width = width
        self.output_dim = 1
        
        self.drop_rate = 0.5

    def get_inputs_pl(self, batch_size):
        """ Create placeholders for the PilotNet
        """
        image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.height, self.width, 3))
        label_pl   = tf.placeholder(tf.float32, shape=(batch_size, self.output_dim))
        
        return image_pl, label_pl

        
    def get_model(self, image_pl, is_training, bn_decay=None):
        """ Nvidia PilotNet model
        Args:
        - input: input Tensor: (B,66,200,3)
        - is_training: A bool flag used for dropout
        Return:
        - output: output Tensor (B,1)
        """
        X = tf.layers.conv2d(image_pl, 24, [5,5], strides=[2,2], 
                             activation=tf.nn.relu,
                             padding='VALID', name='conv1') # (B,31,98,24)
        X = tf.layers.conv2d(X, 36, [5,5], strides=[2,2],
                             activation=tf.nn.relu,
                             padding='VALID', name='conv2') # (B,14,47,36)
        X = tf.layers.conv2d(X, 48, [5,5], strides=[2,2],
                             activation=tf.nn.relu, 
                             padding='VALID', name='conv3') # (B,5,22,48)
        X = tf.layers.conv2d(X, 64, [3,3], strides=[1,1],
                             activation=tf.nn.relu, 
                             padding='VALID', name='conv4') # (B,3,20,64)
        X = tf.layers.conv2d(X, 64, [3,3], strides=[1,1],
                             activation=tf.nn.relu, 
                             padding='VALID', name='conv5') # (B,1,18,64)
        
        X = tf.contrib.layers.flatten(X) # (B, 1152)

        X = tf.layers.dense(X, 1164, activation=tf.nn.relu, name='fc1')
        X = tf.layers.dropout(X, rate=self.drop_rate, training=is_training, name='dp1')

        X = tf.layers.dense(X, 100, activation=tf.nn.relu, name='fc2')
        X = tf.layers.dropout(X, rate=self.drop_rate, training=is_training, name='dp2')

        X = tf.layers.dense(X, 50, activation=tf.nn.relu, name='fc3')
        X = tf.layers.dropout(X, rate=self.drop_rate, training=is_training, name='dp4')

        X = tf.layers.dense(X, 10, activation=tf.nn.relu, name='fc4')
        X = tf.layers.dropout(X, rate=self.drop_rate, training=is_training, name='dp4')

        output = tf.layers.dense(X, self.output_dim, activation=None, name='output')
        
        # y = tf.mul(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2, name='y') # [-pi,pi]

        return output

    def get_loss(self, pred, label, name='Loss'):
        """ Return loss for a batch of data
        Args:
        Return: MSE loss
        """
        
        loss = tf.reduce_mean(tf.square(tf.subtract(pred, label)))
        
        tf.summary.scalar(name, loss)

        return loss

    def get_rmse(self, pred, label, name='rmse'):
        """Return rmse as evalation metrics for a batch of data
        """
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, label))))
        
        tf.summary.scalar(name, rmse)
        
        return rmse

    def get_acc(self, pred, label, delta_a=5.0, delta_s=1.0, name='acc'):
        """
        TODO: Tolerance is to large! angle < 0.01;  
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
    with tf.Graph().as_default():
        is_training = tf.cast(True, tf.bool)
        batch_size = 16
        
        model = PilotNet()

        image_pl, label_pl = model.get_inputs_pl(batch_size)
        pred = model.get_model(image_pl, is_training)
        loss = model.get_loss(pred, label_pl)
        rmse = model.get_rmse(pred, label_pl)

        tf_util.model_summary()
                
        print('loss:', loss)
        print('rmse:', rmse)
        # Total size of variables: 1595511
        # Total bytes of variables: 6382044
