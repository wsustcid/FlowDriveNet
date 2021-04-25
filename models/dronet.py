'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-09-15 19:34:29
@LastEditTime: 2020-09-28 22:17:54
'''

import os
import sys
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..'))
from utils import tf_util

class DroNet(object):
    def __init__(self, height=200, width=200, channels=1):
        self.height = height
        self.width = width
        self.channels = channels
        self.output_dim = 2
        
        self.drop_rate = 0.5

    def get_inputs_pl(self, batch_size):
       
        image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.height, self.width, self.channels))
        label_pl   = tf.placeholder(tf.float32, shape=(batch_size, self.output_dim))
        
        return image_pl, label_pl

        
    def get_model(self, image_pl, is_training, bn_decay=None):
        """ DroNet model (ResNet8)
        Args:
        - input: input Tensor: (B,200,200,1)
        - is_training: A bool flag used for dropout
        Return:
        - output: output Tensor (B,1)
        TODO: in conv2d set
        kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4)
        """
        # Input
        X1 = tf.layers.conv2d(image_pl, 32, [5,5], strides=[2,2], 
                             padding='same', name='conv1')
        X1 = tf.layers.max_pooling2d(X1, pool_size=(3,3), strides=[2,2])
        
        # First residual block
        X2 = tf.layers.batch_normalization(X1, training=is_training, name='res1_bn1')
        X2 = tf.nn.relu(X2, name='res1_relu1')
        X2 = tf.layers.conv2d(X2, 32, [3,3], strides=[2,2],
                             padding='same', name='res1_conv1')
        
        X2 = tf.layers.batch_normalization(X2, training=is_training, name='res1_bn2')
        X2 = tf.nn.relu(X2, name='res1_relu2')
        X2 = tf.layers.conv2d(X2, 32, [3,3], strides=[1,1],
                             padding='same', name='res1_conv2')

        X1 = tf.layers.conv2d(X1, 32, [1,1], strides=[2,2],
                             padding='same', name='res1_skip')
        X3 = tf.add(X1, X2, name='res1_add')

        # Second residual block
        X4 = tf.layers.batch_normalization(X3, training=is_training, name='res2_bn1')
        X4 = tf.nn.relu(X4, name='res2_relu1')
        X4 = tf.layers.conv2d(X4, 64, [3,3], strides=[2,2],
                             padding='same', name='res2_conv1')
        
        X4 = tf.layers.batch_normalization(X4, training=is_training, name='res2_bn2')
        X4 = tf.nn.relu(X4, name='res2_relu2')
        X4 = tf.layers.conv2d(X4, 64, [3,3], strides=[1,1],
                             padding='same', name='res2_conv2')

        X3 = tf.layers.conv2d(X3, 64, [1,1], strides=[2,2],
                             padding='same', name='res2_skip')
        X5 = tf.add(X3, X4, name='res2_add')

        # Third residual block
        X6 = tf.layers.batch_normalization(X5, training=is_training, name='res3_bn1')
        X6 = tf.nn.relu(X6, name='res3_relu1')
        X6 = tf.layers.conv2d(X6, 128, [3,3], strides=[2,2],
                             padding='same', name='res3_conv1')
        
        X6 = tf.layers.batch_normalization(X6, training=is_training, name='res3_bn2')
        X6 = tf.nn.relu(X6, name='res3_relu2')
        X6 = tf.layers.conv2d(X6, 128, [3,3], strides=[1,1],
                             padding='same', name='res3_conv2')

        X5 = tf.layers.conv2d(X5, 128, [1,1], strides=[2,2],
                             padding='same', name='res3_skip')
        X7 = tf.add(X5, X6, name='res3_add')

        X = tf.contrib.layers.flatten(X7) # (B, 1152)
        X = tf.nn.relu(X)
        X = tf.layers.dropout(X, rate=self.drop_rate)

        output = tf.layers.dense(X, self.output_dim, activation=None, name='output')

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
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(pred, label)), axis=0))
        
        tf.summary.scalar(name+'_angle', rmse[0])
        tf.summary.scalar(name+'_speed', rmse[1])

        return rmse[0], rmse[1]

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
        
        model = DroNet()

        image_pl, label_pl = model.get_inputs_pl(batch_size)
        pred = model.get_model(image_pl, is_training)
        loss = model.get_loss(pred, label_pl)

        tf_util.model_summary()
        # Total size of variables: 320930
        # Total bytes of variables: 1283720

