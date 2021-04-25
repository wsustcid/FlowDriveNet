'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-06-03 16:56:56
@LastEditTime: 2020-09-10 23:38:05
'''

import os
import sys
import argparse
from datetime import datetime
import time

import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
from utils.tf_util import log_string

from data_gen import DataLoader
from models.flowdrivenet import FlowDriveNet

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--data_root', default='/gdata/wangshuai/Udacity/CH2',
                    help='data_root path [default: local path]')
parser.add_argument('--input_cfg', default='GRAY', 
                    help='Input type: GRAY, GRAYF, GRAYF-T, XYZ, XYZF, XYZF-T, GRAYF-XYZF-T')
parser.add_argument('--model_cfg', default='VFE',
                    help='Model type: VFE, VFE-TFP, PFE, PFE-TFP, VFE-PFE-TFP')
parser.add_argument('--loss_cfg', default='MSE',
                    help='loss type: weighted, step, exp')
parser.add_argument('--height', type=int, default=200, help='img height')
parser.add_argument('--width', type=int, default=200, help='img width')
parser.add_argument('--seq_len', type=int, default=5, help='sel length')
parser.add_argument('--use_side_cam', default=False, action='store_true')
parser.add_argument('--num_point', type=int, default=20000, help='Point N')
parser.add_argument('--log_dir', default='test',
                    help='Log dir [default: test]')
parser.add_argument('--max_epoch', type=int, default=300,
                    help='Epoch to run [default: 1000]')
parser.add_argument('--early_stop', type=int, default=20,
                    help='stop training when loss stop decreasing [default: 20]')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch Size during training [default: 1]')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Learning rate during training [default: 0.001]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_steps', type=int, default=300000,
                    help='Decay step for lr decay [default: 200000]') # decay_steps = n_train * epochs
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.num_gpus * FLAGS.batch_size
DEVICE_BATCH_SIZE = FLAGS.batch_size

log_dir  = os.path.join(base_dir, 'logs', FLAGS.log_dir)
os.makedirs(log_dir, exist_ok=True)
train_log_dir = os.path.join(log_dir, 'log_train.txt')
log_string(train_log_dir, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
log_string(train_log_dir, str(FLAGS)+'\n')

model_file = os.path.join(base_dir, 'models/flowdrivenet.py')
train_file = os.path.join(base_dir,'train.py')
os.system('cp %s %s' % (model_file, log_dir))
os.system('cp %s %s' % (train_file, log_dir))

# 
dataloader = DataLoader(FLAGS.data_root, FLAGS.input_cfg, 
                        FLAGS.height, FLAGS.width, FLAGS.use_side_cam)
model = FlowDriveNet(FLAGS.input_cfg, FLAGS.model_cfg, 
                      FLAGS.height, FLAGS.width, FLAGS.seq_len)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #for g, _ in grad_and_vars:
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def get_bn_decay(batch):
        bn_momentum = tf.train.exponential_decay(
                        0.5,
                        batch*BATCH_SIZE,
                        float(FLAGS.decay_steps),
                        0.5,
                        staircase=True)
        bn_decay = tf.minimum(0.99, 1 - bn_momentum)
        return bn_decay

#def get_lr(batch):
#    lr = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
#                                    global_step=batch*BATCH_SIZE,
#                                    decay_steps=FLAGS.decay_steps,
#                                    decay_rate=FLAGS.decay_rate,
#                                    staircase=True)
#    lr = tf.maximum(lr, 0.00001)
#    return lr

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'): 
            image_pl, points_pl, label_pl = model.get_inputs_pl(BATCH_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # define global_step; optimizer will increase it in every training loop
            batch = tf.get_variable('batch', [], 
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
            bn_decay = get_bn_decay(batch) 
            tf.summary.scalar('bn_decay', bn_decay)

            if FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            tf.summary.scalar('learning_rate', FLAGS.learning_rate)

            # Allocating variables on CPU first to accelerate multi-gpu training.
            model.get_model(image_pl, points_pl, is_training_pl, bn_decay)
            
            # -------------------------------------------
            # Get model and loss on multiple GPU devices
            # Data Parallel
            # -------------------------------------------
            tower_grads = []
            pred_gpus = []
            loss_gpus = []
            for i in range(FLAGS.num_gpus):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(i)), tf.name_scope('gpu_%d'%(i)) as scope:
                        # split data to each gpu
                        image_pl_gpu = image_pl[i*DEVICE_BATCH_SIZE: (i+1)*DEVICE_BATCH_SIZE]
                        points_pl_gpu = points_pl[i*DEVICE_BATCH_SIZE: (i+1)*DEVICE_BATCH_SIZE]
                        label_pl_gpu = label_pl[i*DEVICE_BATCH_SIZE: (i+1)*DEVICE_BATCH_SIZE]

                        pred_gpu = model.get_model(image_pl_gpu, points_pl_gpu,
                                               is_training_pl, bn_decay)
                        loss_gpu = model.get_loss(pred_gpu, label_pl_gpu, 
                                                    batch, FLAGS.loss_cfg, name=scope)

                        grads_gpu = optimizer.compute_gradients(loss_gpu)
                        tower_grads.append(grads_gpu)
                        pred_gpus.append(pred_gpu)
                        loss_gpus.append(loss_gpu)

            # merge pred and losses from multiple gpus
            grads = average_gradients(tower_grads)
            pred = tf.concat(pred_gpus, 0)
            loss = tf.reduce_mean(loss_gpus)
            
            train_op = optimizer.apply_gradients(grads, global_step=batch)
            
            rmse_angle, rmse_speed = model.get_rmse(pred, label_pl) 

            
         # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict={is_training_pl: True})
        
        # save all tensor
        ops = {'image_pl': image_pl,
            'points_pl': points_pl,
            'label_pl': label_pl,
            'is_training_pl': is_training_pl,
            'train_op': train_op,
            'loss': loss,
            'rmse_angle': rmse_angle,
            'rmse_speed':rmse_speed,
            'merged': merged,
            'batch': batch}

        test_err_min = 100000
        for epoch in range(FLAGS.max_epoch):
            log_string(train_log_dir, '**** EPOCH %03d ****' % (epoch))
            
            train_one_epoch(sess, ops, train_writer)
            test_err = test_one_epoch(sess, ops, test_writer)
            # save best
            if test_err < test_err_min:
                es_count = 0
                test_err_min = test_err
                save_path = saver.save(sess, os.path.join(log_dir, "model_best.ckpt"))
                log_string(train_log_dir, "Best model saved in : %s" % save_path)
            else:
                es_count +=1

            #if epoch % 10 == 0:
            #    save_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"))
            #    log_string(train_log_dir, "Model saved in file: %s" % save_path)
            
            # Early Stopping
            if es_count >= FLAGS.early_stop:
                break


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    # shuffle data
    #data_loader.Xs_train, data_loader.y_train = shuffle(data_loader.Xs_train, data_loader.y_train)
    is_training = True
    num_batches = dataloader.num_train // BATCH_SIZE
    loss_sum = 0.0
    rmse_angle_sum = 0.0
    rmse_speed_sum = 0.0

    for i in range(num_batches):
        #t1 = time.time()
        X_image_batch, X_cloud_batch, y_batch = dataloader.load_train_batch(BATCH_SIZE)
        #t2 = time.time()

        feed_dict = {ops['image_pl']: X_image_batch,
                     ops['points_pl']: X_cloud_batch,
                     ops['label_pl']: y_batch,
                     ops['is_training_pl']: is_training}
        
        summary, step, _, loss_batch, rmse_angle_batch, rmse_speed_batch = sess.run([ops['merged'], ops['batch'], ops['train_op'], ops['loss'], ops['rmse_angle'], ops['rmse_speed']], feed_dict=feed_dict)
        #t3 = time.time()
        #print("data time: {}; train time: {}".format(t2-t1, t3-t2))

        train_writer.add_summary(summary, step)
        
        loss_sum += loss_batch
        rmse_angle_sum += rmse_angle_batch
        rmse_speed_sum += rmse_speed_batch

    log_string(train_log_dir, 'Train loss: %f' % (loss_sum / num_batches))
    log_string(train_log_dir, 'Train rmse_angle: %f' % (rmse_angle_sum / num_batches))
    log_string(train_log_dir, 'Train rmse_speed: %f' % (rmse_speed_sum / num_batches))
    log_string(train_log_dir, 'Train rmse_average: %f' % ((rmse_angle_sum+rmse_speed_sum)/ num_batches/2))

def test_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    
    is_training = False
    num_batches = dataloader.num_val // BATCH_SIZE
    loss_sum = 0.0
    rmse_angle_sum = 0.0
    rmse_speed_sum = 0.0

    for i in range(num_batches):
        X_image_batch, X_cloud_batch, y_batch = dataloader.load_val_batch(BATCH_SIZE)
        
        feed_dict = {ops['image_pl']: X_image_batch,
                     ops['points_pl']: X_cloud_batch,
                     ops['label_pl']: y_batch,
                     ops['is_training_pl']: is_training}
        
        summary, step, loss_batch, rmse_angle_batch, rmse_speed_batch = sess.run([ops['merged'], ops['batch'], ops['loss'], ops['rmse_angle'], ops['rmse_speed']],feed_dict=feed_dict)
        
        test_writer.add_summary(summary, step)
        
        loss_sum += loss_batch
        rmse_angle_sum += rmse_angle_batch
        rmse_speed_sum += rmse_speed_batch

    log_string(train_log_dir, 'Val loss: %f' % (loss_sum / num_batches))
    log_string(train_log_dir, 'Val rmse_angle: %f' % (rmse_angle_sum / num_batches))
    log_string(train_log_dir, 'Val rmse_speed: %f' % (rmse_speed_sum / num_batches))
    log_string(train_log_dir, 'Val rmse_average: %f' % ((rmse_angle_sum+rmse_speed_sum)/ num_batches/2))

    return (rmse_angle_sum+rmse_speed_sum)/num_batches/2

if __name__ == "__main__":
    train()
