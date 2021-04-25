'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-09-11 23:42:23
@LastEditTime: 2020-10-13 22:32:20
'''

import os
import sys
import argparse
from datetime import datetime
import time
from tqdm import tqdm
import time

import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from utils.tf_util import log_string

from data_gen import DataLoader
from models.flowdrivenet import FlowDriveNet

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/media/ubuntu16/Documents/Datasets/Udacity/CH2',
                    help='data_root path [default: local path]')
parser.add_argument('--input_cfg', default='GRAY', 
                    help='Input type: GRAY, GRAYF, GRAYF-T, XYZ, XYZF, XYZF-T, GRAYF-XYZF-T')
parser.add_argument('--model_cfg', default='VFE',
                    help='Model type: VFE, VFE-TFP, PFE, PFE-TFP, VFE-PFE-TFP')
parser.add_argument('--height', type=int, default=200, help='img height')
parser.add_argument('--width', type=int, default=200, help='img width')
parser.add_argument('--seq_len', type=int, default=5, help='sel length')
parser.add_argument('--aug_cfg', default='None', help='None, IA, RP, SC, BA, BS')
#parser.add_argument('--use_side_cam', default=False, action='store_true')
parser.add_argument('--num_point', type=int, default=10000, help='Point N')
parser.add_argument('--log_dir', default='test',
                    help='Log dir [default: test]')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch Size during training [default: 16]')
parser.add_argument('--decay_steps', type=int, default=300000,
                    help='Decay step for lr decay [default: 200000]') # decay_steps = n_train * epochs
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--model_file', default='/media/ubuntu16/F/FlowDriveNet/logs/VFE/gray_base/model_best.ckpt',
                    help='the model path to be evaluated')


FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size

log_dir  = os.path.join(base_dir, 'logs', FLAGS.log_dir)
os.makedirs(log_dir, exist_ok=True)
test_log_dir = os.path.join(log_dir, 'log_test.txt')
log_string(test_log_dir, str(FLAGS)+'\n')

# 
dataloader = DataLoader(FLAGS.data_root, FLAGS.input_cfg, 
                        FLAGS.height, FLAGS.width,
                        FLAGS.seq_len, 
                        FLAGS.num_point,
                        FLAGS.aug_cfg)
model = FlowDriveNet(FLAGS.input_cfg, FLAGS.model_cfg, 
                      FLAGS.height, FLAGS.width, FLAGS.seq_len, FLAGS.num_point)

def get_bn_decay(batch):
        bn_momentum = tf.train.exponential_decay(
                        0.5,
                        batch*BATCH_SIZE,
                        float(FLAGS.decay_steps),
                        0.5,
                        staircase=True)
        bn_decay = tf.minimum(0.99, 1 - bn_momentum)
        return bn_decay

def eval():
    with tf.Graph().as_default():
        image_pl, points_pl, _ = model.get_inputs_pl(BATCH_SIZE)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # define global_step; optimizer will increase it in every training loop
        batch = tf.get_variable('batch', [], 
                                initializer=tf.constant_initializer(0),
                                trainable=False)
        bn_decay = get_bn_decay(batch) 
        
        pred = model.get_model(image_pl, points_pl, is_training_pl, bn_decay)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # restore model
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_file)

        # save all tensor
        ops = {'image_pl': image_pl,
            'points_pl': points_pl,
            'is_training_pl': is_training_pl,
            'pred': pred}

        ## evaluation    
        is_training = False
        num_batches = dataloader.num_test // BATCH_SIZE
        rmse_angle_sum = 0.0
        rmse_speed_sum = 0.0
        result_all = np.zeros((0,4)) # pred_a, pred_s, label_a, label_s
        
        time_sum = 0.0
        for i in tqdm(range(num_batches)):
            X_image_batch, X_cloud_batch, y_batch = dataloader.load_test_batch(BATCH_SIZE)
            
            feed_dict = {ops['image_pl']: X_image_batch,
                        ops['points_pl']: X_cloud_batch,
                        ops['is_training_pl']: is_training}
            t1 = time.time()
            pred_batch = sess.run(ops['pred'],feed_dict=feed_dict)
            t2 = time.time()
            time_sum += (t2-t1) 
            result_batch = np.hstack((pred_batch, y_batch))
            result_all = np.concatenate((result_all, result_batch), axis=0)
    
        
        np.savetxt(os.path.join(log_dir, 'results.csv'), result_all, delimiter=",")
        # b = np.loadtxt("temp.csv", delimiter=",")

        rmse_angle = np.sqrt(np.mean(np.square(result_all[:,0] - result_all[:,2])))
        rmse_speed = np.sqrt(np.mean(np.square(result_all[:,1] - result_all[:,3])))
        log_string(test_log_dir, 'Test rmse_angle: %f' % (rmse_angle))
        log_string(test_log_dir, 'Test rmse_speed: %f' % (rmse_speed))
        log_string(test_log_dir, 'Test rmse_average: %f' % ((rmse_angle+rmse_speed)/2))
        log_string(test_log_dir, 'Test FPS: %f' % (1/(time_sum/num_batches)))


if __name__ == "__main__":
    eval()
