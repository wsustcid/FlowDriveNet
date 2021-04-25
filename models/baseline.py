'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-08-04 19:43:43
@LastEditTime: 2020-09-13 22:39:59
@Description:  
'''
import os 
import sys
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '../'))
 
from data_gen import DataLoader


if __name__ == '__main__':
    data_root='/media/ubuntu16/Documents/Datasets/Udacity/CH2'
    dataloader = DataLoader(data_root, input_cfg='GRAY', height=200, width=200, use_side_cam=False)
    
    label_train = dataloader.y_train
    label_val = dataloader.y_val
    label_test = dataloader.y_test
    
    pred_train = np.zeros(label_train.shape, dtype=np.float32)
    pred_val = np.zeros(label_val.shape, dtype=np.float32)
    pred_test = np.zeros(label_test.shape, dtype=np.float32)

    rmse_train = np.sqrt(np.mean(np.square(label_train-pred_train), axis=0))
    rmse_val = np.sqrt(np.mean(np.square(label_val-pred_val), axis=0))
    rmse_test = np.sqrt(np.mean(np.square(label_test-pred_test), axis=0))
    
    print("Train: rmse_angle: {}; rmse_speed: {}; rmse_all: {}".format(rmse_train[0], rmse_train[1], np.mean(rmse_train)))
    print("Val: rmse_angle: {}; rmse_speed: {}; rmse_all: {}".format(rmse_val[0], rmse_val[1], np.mean(rmse_val)))
    print("Test: rmse_angle: {}; rmse_speed: {}; rmse_all: {}".format(rmse_test[0], rmse_test[1], np.mean(rmse_test)))