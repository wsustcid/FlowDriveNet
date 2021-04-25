'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-28 22:19:51
@LastEditTime: 2020-09-28 22:02:40
'''

import os
import sys
import numpy as np
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from utils import tf_util


class BMWNet(object):
    def __init__(self, height=66, width=200, seq_len=10):
        self.height = height
        self.width = width
        self.seq_len = seq_len
        
        self.output_dim = 1

        self.drop_rate = 0.5


    def get_inputs_pl(self, batch_size):
        image_pl = tf.placeholder(tf.float32, shape=(batch_size, self.seq_len, self.height, self.width, 3))
        label_pl = tf.placeholder(tf.float32, shape=(batch_size, self.output_dim))
        
        return image_pl, label_pl

        
    def get_model(self, image_pl, is_training, bn_decay=None):
        """ BMWNet model
        Args:
        - input: input Tensor: (B,T,66,200,3)
        - is_training: A bool flag used for dropout
        Return:
        - output: output Tensor (B,2)
        """
        
        img_shape = image_pl.get_shape().as_list()[2:]
        input_img = tf.reshape(image_pl, (-1,*img_shape)) # (B*T,H,W,C)

        X = tf.layers.conv2d(input_img, 24, [5,5], strides=[2,2], 
                             activation=tf.nn.relu,
                             padding='VALID', name='conv1') # (B*T,31,98,24)
        X = tf.layers.conv2d(X, 36, [5,5], strides=[2,2],
                             activation=tf.nn.relu,
                             padding='VALID', name='conv2') # (B*T,14,47,36)
        X = tf.layers.conv2d(X, 48, [5,5], strides=[2,2],
                             activation=tf.nn.relu, 
                             padding='VALID', name='conv3') # (B*T,5,22,48)
        X = tf.layers.conv2d(X, 64, [3,3], strides=[1,1],
                             activation=tf.nn.relu, 
                             padding='VALID', name='conv4') # (B*T,3,20,64)
        X = tf.layers.conv2d(X, 64, [3,3], strides=[1,1],
                             activation=tf.nn.relu, 
                             padding='VALID', name='conv5') # (B*T,1,18,64)
        
        X = tf.contrib.layers.flatten(X) # (B*T, 1152)

        X = tf.layers.dense(X, 1152, activation=tf.nn.relu, name='fc1')
        X = tf.layers.dropout(X, rate=self.drop_rate, training=is_training, name='dp1')

        X = tf.layers.dense(X, 512, activation=tf.nn.relu, name='fc2')
        X = tf.layers.dropout(X, rate=self.drop_rate, training=is_training, name='dp2')
        
        # add lstm
        X = tf.reshape(X, (-1, self.seq_len, X.get_shape().as_list()[-1])) # (B,T,D)
        X = tf_util.lstm(X, hidden_size=128, scope='lstm')
        
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


if __name__ == '__main__':
    with tf.Graph().as_default():
        is_training_pl = tf.placeholder(tf.bool, shape=())

        model = BMWNet()

        img_pl, label_pl = model.get_inputs_pl(batch_size=32)
        outputs = model.get_model(img_pl, is_training=is_training_pl)
        
        tf_util.model_summary()
        #Total size of variables: 2378261
        # Total bytes of variables: 9513044