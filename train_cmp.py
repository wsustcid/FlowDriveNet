'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-09-15 21:58:38
@LastEditTime: 2020-09-16 20:57:19
@Description: Training the comparision models
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

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/gdata/wangshuai/Udacity/CH2',
                    help='data_root path [default: local path]')
parser.add_argument('--input_cfg', default='BGR', 
                    help='Input type: BGR, GRAYF-T, XYZ, GRAY')
parser.add_argument('--model_cfg', default='PilotNet',
                    help='Model type: PilotNet, BMWNet, PointNet, DroNet')
parser.add_argument('--use_side_cam', default=False, action='store_true')
parser.add_argument('--log_dir', default='test',
                    help='Log dir [default: test]')
parser.add_argument('--max_epoch', type=int, default=300,
                    help='Epoch to run [default: 1000]')
parser.add_argument('--early_stop', type=int, default=20,
                    help='stop training when loss stop decreasing [default: 20]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Learning rate during training [default: 0.001]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--decay_steps', type=int, default=300000,
                    help='Decay step for lr decay [default: 200000]') # decay_steps = n_train * epochs
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size

log_dir  = os.path.join(base_dir, 'logs', FLAGS.log_dir)
os.makedirs(log_dir, exist_ok=True)
train_log_dir = os.path.join(log_dir, 'log_train.txt')
log_string(train_log_dir, str(FLAGS)+'\n')


if FLAGS.model_cfg == 'PilotNet':
    from models.pilotnet import PilotNet
    dataloader = DataLoader(FLAGS.data_root, "BGR", 
                            height=66, width=200, 
                            seq_len=None, 
                            num_point=None,
                            use_side_cam=FLAGS.use_side_cam)
    model = PilotNet()
elif FLAGS.model_cfg == 'BMWNet':
    from models.bmwnet import BMWNet
    # TODO add seq_len
    dataloader = DataLoader(FLAGS.data_root, 'GRAYF-T', 
                            height=66, width=200, 
                            seq_len=10, 
                            num_point=None,
                            use_side_cam=FLAGS.use_side_cam)
    model = BMWNet()
elif FLAGS.model_cfg == 'PointNet':
    from models.pointnet import PointNet
    dataloader = DataLoader(FLAGS.data_root, 'XYZ', 
                            height=None, width=None, 
                            seq_len=None, 
                            num_point=10000,
                            use_side_cam=FLAGS.use_side_cam)
    model = PointNet(num_point=10000)
elif FLAGS.model_cfg == 'DroNet':
    from models.dronet import DroNet
    dataloader = DataLoader(FLAGS.data_root, 'GRAY', 
                            height=200, width=200, 
                            seq_len=None, 
                            num_point=None,
                            use_side_cam=FLAGS.use_side_cam)
    model = DroNet()
else:
    raise TypeError

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
        feature_pl, label_pl = model.get_inputs_pl(BATCH_SIZE)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # define global_step; optimizer will increase it in every training loop
        batch = tf.get_variable('batch', [], 
                                initializer=tf.constant_initializer(0),
                                trainable=False)
        bn_decay = get_bn_decay(batch) 
        
        pred = model.get_model(feature_pl, is_training_pl, bn_decay)
        loss = model.get_loss(pred, label_pl)

        #learning_rate = get_lr(batch)
        if FLAGS.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        tf.summary.scalar('learning_rate', FLAGS.learning_rate)
        
        train_op = optimizer.minimize(loss, global_step=batch)
        
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
        sess.run(init)
        
        # save all tensor
        ops = {'feature_pl': feature_pl,
            'label_pl': label_pl,
            'is_training_pl': is_training_pl,
            'train_op': train_op,
            'loss': loss,
            'pred': pred,
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
    log_string(train_log_dir, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    num_batches = dataloader.num_train // BATCH_SIZE
    loss_sum = 0.0
    rmse_angle_sum = 0.0
    rmse_speed_sum = 0.0

    for i in range(num_batches):
        #t1 = time.time()
        if FLAGS.model_cfg in ['PilotNet', 'BMWNet']:
            X_batch, y = dataloader.load_image_train_batch(BATCH_SIZE)
            y_batch = y[:,0:1]
        elif FLAGS.model_cfg == 'PointNet':
            X_batch, y = dataloader.load_cloud_train_batch(BATCH_SIZE)
            y_batch = y[:,1:2]
        elif FLAGS.model_cfg == 'DroNet':
            X_batch, y_batch = dataloader.load_image_train_batch(BATCH_SIZE)
        else:
            raise TypeError
        #t2 = time.time()

        feed_dict = {ops['feature_pl']: X_batch,
                     ops['label_pl']: y_batch,
                     ops['is_training_pl']: is_training}
        
        summary, step, _, loss_batch, pred_batch = sess.run([ops['merged'], ops['batch'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        #t3 = time.time()
        #print("data time: {}; train time: {}".format(t2-t1, t3-t2))

        train_writer.add_summary(summary, step)
        
        loss_sum += loss_batch
        if FLAGS.model_cfg in ['PilotNet', 'BMWNet']:
            rmse_angle_batch = np.sqrt(np.mean(np.square(pred_batch-y_batch)))
            rmse_angle_sum += rmse_angle_batch
        elif FLAGS.model_cfg == 'PointNet':
            rmse_speed_batch = np.sqrt(np.mean(np.square(pred_batch-y_batch)))
            rmse_speed_sum += rmse_speed_batch
        elif FLAGS.model_cfg == 'DroNet':
            rmse_batch = np.sqrt(np.mean(np.square(pred_batch-y_batch), axis=0))
            rmse_angle_sum += rmse_batch[0]
            rmse_speed_sum += rmse_batch[1]
        else:
            raise TypeError
        
    log_string(train_log_dir, 'Train loss: %f' % (loss_sum / num_batches))
    if FLAGS.model_cfg in ['PilotNet', 'BMWNet']:
        log_string(train_log_dir, 'Train rmse_angle: %f' % (rmse_angle_sum / num_batches))
    elif FLAGS.model_cfg == 'PointNet':
        log_string(train_log_dir, 'Train rmse_speed: %f' % (rmse_speed_sum / num_batches))
    elif FLAGS.model_cfg == 'DroNet':
        log_string(train_log_dir, 'Train rmse_angle: %f' % (rmse_angle_sum / num_batches))
        log_string(train_log_dir, 'Train rmse_speed: %f' % (rmse_speed_sum / num_batches))
    else:
        raise TypeError

def test_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    
    is_training = True
    log_string(train_log_dir, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    num_batches = dataloader.num_val // BATCH_SIZE
    loss_sum = 0.0
    rmse_angle_sum = 0.0
    rmse_speed_sum = 0.0

    for i in range(num_batches):
        #t1 = time.time()
        if FLAGS.model_cfg in ['PilotNet', 'BMWNet']:
            X_batch, y = dataloader.load_image_val_batch(BATCH_SIZE)
            y_batch = y[:,0:1]
        elif FLAGS.model_cfg == 'PointNet':
            X_batch, y = dataloader.load_cloud_val_batch(BATCH_SIZE)
            y_batch = y[:,1:2]
        elif FLAGS.model_cfg == 'DroNet':
            X_batch, y_batch = dataloader.load_image_val_batch(BATCH_SIZE)
        else:
            raise TypeError
        #t2 = time.time()

        feed_dict = {ops['feature_pl']: X_batch,
                     ops['label_pl']: y_batch,
                     ops['is_training_pl']: is_training}
        
        summary, step, _, loss_batch, pred_batch = sess.run([ops['merged'], ops['batch'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        #t3 = time.time()
        #print("data time: {}; train time: {}".format(t2-t1, t3-t2))

        test_writer.add_summary(summary, step)
        
        loss_sum += loss_batch
        if FLAGS.model_cfg in ['PilotNet', 'BMWNet']:
            rmse_angle_batch = np.sqrt(np.mean(np.square(pred_batch-y_batch)))
            rmse_angle_sum += rmse_angle_batch
        elif FLAGS.model_cfg == 'PointNet':
            rmse_speed_batch = np.sqrt(np.mean(np.square(pred_batch-y_batch)))
            rmse_speed_sum += rmse_speed_batch
        elif FLAGS.model_cfg == 'DroNet':
            rmse_batch = np.sqrt(np.mean(np.square(pred_batch-y_batch), axis=0))
            rmse_angle_sum += rmse_batch[0]
            rmse_speed_sum += rmse_batch[1]
        else:
            raise TypeError
        
    log_string(train_log_dir, 'Val loss: %f' % (loss_sum / num_batches))
    if FLAGS.model_cfg in ['PilotNet', 'BMWNet']:
        log_string(train_log_dir, 'Val rmse_angle: %f' % (rmse_angle_sum / num_batches))
        return rmse_angle_sum/num_batches
    elif FLAGS.model_cfg == 'PointNet':
        log_string(train_log_dir, 'Val rmse_speed: %f' % (rmse_speed_sum / num_batches))
        return rmse_speed_sum/num_batches
    elif FLAGS.model_cfg == 'DroNet':
        log_string(train_log_dir, 'Val rmse_angle: %f' % (rmse_angle_sum / num_batches))
        log_string(train_log_dir, 'Val rmse_speed: %f' % (rmse_speed_sum / num_batches))
        return (rmse_angle_sum+rmse_speed_sum)/num_batches/2
    else:
        raise TypeError

    

if __name__ == "__main__":
    train()
