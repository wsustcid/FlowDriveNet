'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-31 21:40:59
@LastEditTime: 2020-10-13 22:43:48
@Description: Generate data for train, validation and testing.
'''

import os 
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

class DataLoader(object):
    """ Loading images and point clouds from Udacity CH2 dataset
    """
    def __init__(self, data_root, input_cfg, 
                 height, width, seq_len, num_point,
                 aug_cfg):
        """
        Args:
          - data_root: The Udacity dataset path
          - supported input type: GRAY, GRAYF, GRAYF-T, XYZ, XYZF, XYZF-T, GRAYF-XYZF-T
        """
        self.input_cfg = input_cfg
        if self.input_cfg == 'BGR':
            self.height, self.width = height, width
            self.seq_len, self.use_optical_flow  = 1, False
            self.colorspace = 'BGR'
        elif self.input_cfg == 'GRAY':
            self.height, self.width = height, width
            self.seq_len, self.use_optical_flow  = 1, False
            self.colorspace = 'GRAY'
        elif self.input_cfg == 'GRAYF':
            self.height, self.width = height, width
            self.seq_len, self.use_optical_flow  = 1, True
        elif self.input_cfg == 'GRAYF-T':
            self.height, self.width = height, width
            self.seq_len, self.use_optical_flow  = seq_len, True
        elif self.input_cfg == 'XYZ':
            self.height, self.width, self.use_optical_flow = None, None, None
            self.seq_len = 1
            self.num_point = num_point
        elif self.input_cfg == 'XYZF':
            self.height, self.width, self.use_optical_flow = None, None, None
            self.seq_len = 1
            self.num_point = num_point
        elif self.input_cfg == 'XYZF-T':
            self.height, self.width, self.use_optical_flow = None, None, None
            self.seq_len = seq_len
            self.num_point = num_point
        elif self.input_cfg == 'GRAYF-XYZF-T':
            self.height, self.width = height, width
            self.seq_len, self.use_optical_flow  = seq_len, True
            self.num_point = num_point
    
        else:
            raise TypeError
        
        self.data_dir_train  = os.path.join(data_root, 'CH2_002')
        self.data_dir_test  = os.path.join(data_root, 'CH2_001')
        self.val_ratio = 0.2
        
        if aug_cfg == 'None':
            self.random_point, self.use_side_cam = False, False
            self.balance_angle, self.balance_speed = False, False
            self.image_aug = False
        elif aug_cfg == 'IA':
            self.random_point, self.use_side_cam = False, False
            self.balance_angle, self.balance_speed = False, False
            self.image_aug = True
        elif aug_cfg == 'RP':
            self.random_point, self.use_side_cam = True, False
            self.balance_angle, self.balance_speed = False, False
            self.image_aug = False
        elif aug_cfg == 'SC':
            self.random_point, self.use_side_cam = False, True
            self.balance_angle, self.balance_speed = False, False
            self.image_aug = False
        elif aug_cfg == 'BA':
            self.random_point, self.use_side_cam = False, False
            self.balance_angle, self.balance_speed = True, False
            self.image_aug = False
        elif aug_cfg == 'BS':
            self.random_point, self.use_side_cam = False, False
            self.balance_angle, self.balance_speed = False, True
            self.image_aug = False
        elif aug_cfg == 'IA-RP':
            self.random_point, self.use_side_cam = True, False
            self.balance_angle, self.balance_speed = False, False
            self.image_aug = True
        elif aug_cfg == 'IA-RP-SC':
            self.random_point, self.use_side_cam = True, True
            self.balance_angle, self.balance_speed = False, False
            self.image_aug = True
        elif aug_cfg == 'IA-RP-SC-BS':
            self.random_point, self.use_side_cam = True, True
            self.balance_angle, self.balance_speed = False, True
            self.image_aug = True
        else:
            raise TypeError    

        self.__load_train()
        self.__load_test()

        self.__train_batch_pointer = 0
        self.__val_batch_pointer = 0
        self.__test_batch_pointer = 0

        print('===== Input Configuration =====')
        print('Image: ({}, {}); Use_flow: {}; Seq_len: {}; aug_cfg: {}'.format(self.height, self.width, self.use_optical_flow, self.seq_len, aug_cfg))
        print("Train set: {}, Val set: {}, Test set: {}".format(self.num_train, self.num_val,self.num_test))

        
    def __load_train(self):
        """ Load image paths & point paths and labels from multiple csv files      
        Return:
          - self.Xs: saves paths of sequential data: 
                    [[[i1,p1, flag], ...[it,pt,flag]],
                      [[],[]],  
                    ]; size is (N, T, 3)
            FLAG: left: -1; center: 0; right: 1
          - self.y: saves scaled labels:
                   [[angle, speed],
                    [,],
                   ]; size si (N,2)
        """
        # Loading from all csv files;  * is HMB
        file_paths = glob.glob(os.path.join(self.data_dir_train,'*/center.csv'))
        Xs = np.zeros((0, self.seq_len, 3)) # (N,T,3)
        y = np.zeros((0, 2))  # (N,2)

        for file_path in sorted(file_paths):
            path_prefix = os.path.dirname(file_path)
            # load label
            data  = pd.read_csv(file_path)
            img_names = data['filename'].values # relative path
            point_names = data['point_filename'].values
            assert len(img_names) == len(point_names)
            
            # combine sequential image path and point path
            xs = [] # (Ni,T,2)
            for i in range(len(img_names)):
                if i < (self.seq_len-1):
                    continue
                xt = [] # (T,2)
                for t in reversed(range(self.seq_len)):
                    xt.append([os.path.join(path_prefix, img_names[i-t]), 
                               os.path.join(path_prefix, 'points_bin', point_names[i-t][:-3]+'bin'),
                               0.0]) # CAM_FLAG=0 
                xs.append(xt)
            
            # scale label
            angle = data['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
            speed = data['speed'].values[self.seq_len-1:]
            angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
            speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
            ys = np.stack([angle_s, speed_s], axis=1)
            
            # concatenate all data
            assert len(xs) == len(ys) == (len(img_names)-self.seq_len+1)
            Xs = np.concatenate((Xs,xs), axis=0)
            y  = np.concatenate((y, ys), axis=0)
            print("Loading data from {}: {}".format(file_path, len(xs)))

        if self.use_side_cam:
            file_paths_left = glob.glob(os.path.join(self.data_dir_train,'*/left.csv'))
            
            for file_path_left in sorted(file_paths_left):
                path_prefix_left = os.path.dirname(file_path_left)
                # load label
                data_left  = pd.read_csv(file_path_left)
                img_names_left = data_left['filename'].values # relative path
                point_names_left = data_left['point_filename'].values
                assert len(img_names_left) == len(point_names_left)
                
                # combine sequential image path and point path
                xs_left = [] # (Ni,T,3)
                for i in range(len(img_names_left)):
                    if i < (self.seq_len-1):
                        continue
                    xt_left = [] # (T,3)
                    for t in reversed(range(self.seq_len)):
                        xt_left.append([os.path.join(path_prefix_left, img_names_left[i-t]), 
                                os.path.join(path_prefix_left, 'points_bin', point_names_left[i-t][:-3]+'bin'),
                                -1.0])  
                    xs_left.append(xt_left)
                
                # scale label
                angle_left = data_left['angle'].values[self.seq_len-1:]  # n-(self.seq_len-1)
                speed_left = data_left['speed'].values[self.seq_len-1:]
                angle_left_adj = self.__camera_adjust(angle_left, speed_left, camera='left')

                angle_left_s = self.scale_label(angle_left_adj, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
                speed_left_s = self.scale_label(speed_left, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
                ys_left = np.stack([angle_left_s, speed_left_s], axis=1)
                
                # concatenate all data
                assert len(xs_left) == len(ys_left) == (len(img_names_left)-self.seq_len+1)
                Xs = np.concatenate((Xs,xs_left), axis=0)
                y  = np.concatenate((y, ys_left), axis=0)
                print("Loading data from {}: {}".format(file_path_left, len(xs_left)))
            
            ## Load right camera data
            file_paths_right = glob.glob(os.path.join(self.data_dir_train,'*/right.csv'))
            
            for file_path_right in sorted(file_paths_right):
                path_prefix_right = os.path.dirname(file_path_right)
                # load label
                data_right  = pd.read_csv(file_path_right)
                img_names_right = data_right['filename'].values # relative path
                point_names_right = data_right['point_filename'].values
                assert len(img_names_right) == len(point_names_right)
                
                # combine sequential image path and point path
                xs_right = [] # (Ni,T,2)
                for i in range(len(img_names_right)):
                    if i < (self.seq_len-1):
                        continue
                    xt_right = [] # (T,2)
                    for t in reversed(range(self.seq_len)):
                        xt_right.append([os.path.join(path_prefix_right, img_names_right[i-t]), 
                                os.path.join(path_prefix_right, 'points_bin', point_names_right[i-t][:-3]+'bin'),
                                1.0])  
                    xs_right.append(xt_right)
                
                # scale label
                angle_right = data_right['angle'].values[self.seq_len-1:]
                speed_right = data_right['speed'].values[self.seq_len-1:]
                angle_right_adj = self.__camera_adjust(angle_right, speed_right, camera='right')

                angle_right_s = self.scale_label(angle_right_adj, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
                speed_right_s = self.scale_label(speed_right, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
                ys_right = np.stack([angle_right_s, speed_right_s], axis=1)
                
                # concatenate all data
                assert len(xs_right) == len(ys_right) == (len(img_names_right)-self.seq_len+1)
                Xs = np.concatenate((Xs,xs_right), axis=0)
                y  = np.concatenate((y, ys_right), axis=0)
                print("Loading data from {}: {}".format(file_path_right, len(xs_right)))

        if self.balance_angle or self.balance_speed:
            Xs, y = self.balance_data(Xs, y, 
                                      self.balance_angle, 
                                      self.balance_speed,
                                      bin_count=20,
                                      fix_times=1)
        # visualize label distribution
        #self.label_distribution(y)

        # split data
        self.Xs_train, self.Xs_val, self.y_train, self.y_val = train_test_split(Xs, y, test_size=self.val_ratio, random_state=10, shuffle=True)

        self.num_train = len(self.Xs_train)
        self.num_val = len(self.y_val)
        print("Train set: {}; Val set: {}".format(self.num_train, self.num_val))

    def __load_test(self):
        """ Load image paths and labels from test csv files
        DO NOT USE SIDE CAMERA WHILE TESTING    
        """
        # Reading center camera images; 
        file_path = os.path.join(self.data_dir_test,'center.csv')
        path_prefix = os.path.dirname(file_path) # path prefix
        data  = pd.read_csv(file_path)
        img_names = data['filename'].values # relative path
        point_names = data['point_filename'].values
        assert len(img_names) == len(point_names)
        
        self.Xs_test = [] # (N,T,2)
        for i in range(len(img_names)):
            if i < (self.seq_len-1):
                continue
            xt = [] # (T,2)
            for t in reversed(range(self.seq_len)):
                xt.append([os.path.join(path_prefix, img_names[i-t]),
                           os.path.join(path_prefix, 'points_bin', point_names[i-t][:-3]+'bin')])
            self.Xs_test.append(xt) 
        
        self.Xs_test = np.array(self.Xs_test)
        angle = data['angle'].values[self.seq_len-1:]
        speed = data['speed'].values[self.seq_len-1:]
        angle_s = self.scale_label(angle, y_min=-2.0, y_max=2.0, a=-1.0, b=1.0)
        speed_s = self.scale_label(speed, y_min=0.0, y_max=30.0, a=-1.0, b=1.0)
        self.y_test = np.stack([angle_s, speed_s], axis=1)
        
        assert len(self.Xs_test) == len(self.y_test) == (len(img_names)-self.seq_len+1)
        self.num_test = len(self.Xs_test)
        print("Loading data from {}: {}".format(file_path, self.num_test))

    def load_image_train_batch(self, batch_size):
        """ Load batch of the train data 
        Return:
         - X_batch: (B,T,H,W,C) or (B,H,W,C)
         - y_batch: (B,2)
        Training Preprocedure:
          1. Load image with grayscale image and reshape to (480,640,1); float32 (255.0)
          2. Load the optical flow from the bin file and reshape to (480,640,2); float32
          3. concate to iflow (480,640,3)
          4. crop and resize flow
          Note: we can also normalize image and flow to (-1,1) in step 1 & 2 while training
        """
        if not self.use_optical_flow:
            X_batch = [] # (B,T,H,W,C)
            y_batch = []
            for i in range(batch_size):
                train_idx = (self.__train_batch_pointer + i) % self.num_train
                imgs = [] # (T,H,W,C)
                for j in range(self.seq_len):
                    img_path = self.Xs_train[train_idx,j,0]
                    if self.colorspace == 'GRAY':
                        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[100:], (self.width, self.height))
                        img = img.reshape((*img.shape, 1))
                    elif self.colorspace == 'BGR':
                        img = cv2.resize(cv2.imread(img_path)[100:], (self.width, self.height))
                    
                    if self.image_aug:
                        img = self.image_process(img) 
                    img = self.normalize(img)
                    imgs.append(img)
                    
                X_batch.append(imgs)
                y_batch.append(self.y_train[train_idx])
            
            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)

            self.__train_batch_pointer += batch_size
        
        else:
            X_batch = [] # (B,T,H,W,C)
            y_batch = [] # (B,2)
            for i in range(batch_size):
                train_idx = (self.__train_batch_pointer + i) % self.num_train
                imgs = []
                for j in range(self.seq_len):
                    end_img_path = self.Xs_train[train_idx,j,0]
                    end_img = cv2.imread(end_img_path, cv2.IMREAD_GRAYSCALE)
                    end_img = np.reshape(end_img, (480,640,1))
                    if self.image_aug:
                        end_img = self.image_process(end_img) 
                    end_img = self.normalize(end_img)
                    # load flow
                    path_prefix = os.path.dirname(end_img_path)
                    img_name = end_img_path.split('/')[-1]
                    flow_path = os.path.join(path_prefix+'_bin', img_name[:-3]+'bin')
                    flow = np.fromfile(flow_path, dtype=np.float16).reshape((480,640,2))
                    flow = np.asarray(flow, dtype=np.float32)
                    flow = self.normalize(flow)
                    
                    iflow = np.concatenate((end_img, flow), axis=2)
                    iflow = cv2.resize(iflow[100:], (self.width, self.height))

                    imgs.append(iflow)

                X_batch.append(imgs)
                y_batch.append(self.y_train[train_idx])

            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
    
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)
            
            self.__train_batch_pointer += batch_size

        return X_batch, y_batch

    def load_image_val_batch(self, batch_size):
        ''' Load batch of the Val data '''

        if not self.use_optical_flow:
            X_batch = [] # (B,T,H,W,C)
            y_batch = []
            for i in range(batch_size):
                val_idx = (self.__val_batch_pointer + i) % self.num_val
                imgs = [] # (T,H,W,C)
                for j in range(self.seq_len):
                    img_path = self.Xs_val[val_idx,j,0]
                    if self.colorspace == 'GRAY':
                        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[100:], (self.width, self.height))
                        img = img.reshape((*img.shape, 1))
                    elif self.colorspace == 'BGR':
                        img = cv2.resize(cv2.imread(img_path)[100:], (self.width, self.height))
            
                    img = self.normalize(img)
                    imgs.append(img)
                    
                X_batch.append(imgs)
                y_batch.append(self.y_val[val_idx])
            
            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)

            self.__val_batch_pointer += batch_size
        
        else:
            X_batch = [] # (B,T,H,W,C)
            y_batch = [] # (B,2)
            for i in range(batch_size):
                val_idx = (self.__val_batch_pointer + i) % self.num_val
                imgs = []
                for j in range(self.seq_len):
                    end_img_path = self.Xs_val[val_idx,j,0]
                    end_img = cv2.imread(end_img_path, cv2.IMREAD_GRAYSCALE)
                    end_img = np.reshape(end_img, (480,640,1))
                    end_img = self.normalize(end_img)
                    # load flow
                    path_prefix = os.path.dirname(end_img_path)
                    img_name = end_img_path.split('/')[-1]
                    flow_path = os.path.join(path_prefix+'_bin', img_name[:-3]+'bin')
                    flow = np.fromfile(flow_path, dtype=np.float16).reshape((480,640,2))
                    flow = np.asarray(flow, dtype=np.float32)
                    flow = self.normalize(flow)
                    
                    iflow = np.concatenate((end_img, flow), axis=2)
                    iflow = cv2.resize(iflow[100:], (self.width, self.height))

                    imgs.append(iflow)

                X_batch.append(imgs)
                y_batch.append(self.y_val[val_idx])

            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
    
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)
            
            self.__val_batch_pointer += batch_size

        return X_batch, y_batch

    def load_image_test_batch(self, batch_size):
        ''' Load batch of the test data '''
         
        if not self.use_optical_flow:
            X_batch = [] # (B,T,H,W,C)
            y_batch = []
            for i in range(batch_size):
                test_idx = (self.__test_batch_pointer + i) % self.num_test
                imgs = [] # (T,H,W,C)
                for j in range(self.seq_len):
                    img_path = self.Xs_test[test_idx,j,0]
                    if self.colorspace == 'GRAY':
                        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[100:], (self.width, self.height))
                        img = img.reshape((*img.shape, 1))
                    elif self.colorspace == 'BGR':
                        img = cv2.resize(cv2.imread(img_path)[100:], (self.width, self.height))
            
                    img = self.normalize(img)
                    imgs.append(img)
                    
                X_batch.append(imgs)
                y_batch.append(self.y_test[test_idx])
            
            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)

            self.__test_batch_pointer += batch_size
        
        else:
            X_batch = [] # (B,T,H,W,C)
            y_batch = [] # (B,2)
            for i in range(batch_size):
                test_idx = (self.__test_batch_pointer + i) % self.num_test
                imgs = []
                for j in range(self.seq_len):
                    end_img_path = self.Xs_test[test_idx,j,0]
                    end_img = cv2.imread(end_img_path, cv2.IMREAD_GRAYSCALE)
                    end_img = np.reshape(end_img, (480,640,1))
                    end_img = self.normalize(end_img)
                    # load flow
                    path_prefix = os.path.dirname(end_img_path)
                    img_name = end_img_path.split('/')[-1]
                    flow_path = os.path.join(path_prefix+'_bin', img_name[:-3]+'bin')
                    flow = np.fromfile(flow_path, dtype=np.float16).reshape((480,640,2))
                    flow = np.asarray(flow, dtype=np.float32)
                    flow = self.normalize(flow)
                    
                    iflow = np.concatenate((end_img, flow), axis=2)
                    iflow = cv2.resize(iflow[100:], (self.width, self.height))

                    imgs.append(iflow)

                X_batch.append(imgs)
                y_batch.append(self.y_test[test_idx])

            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
    
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)
            
            self.__test_batch_pointer += batch_size

        return X_batch, y_batch

    
    def load_image_visual_batch(self):
        ''' Load one of the test data by index'''
        batch_size = 1
        if not self.use_optical_flow:
            X_batch = [] # (B,T,H,W,C)
            y_batch = []
            for i in range(batch_size):
                test_idx = (self.__test_batch_pointer + i) % self.num_test
                imgs = [] # (T,H,W,C)
                for j in range(self.seq_len):
                    img_path = self.Xs_test[test_idx,j,0]
                    if self.colorspace == 'GRAY':
                        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[100:], (self.width, self.height))
                        img = img.reshape((*img.shape, 1))
                    elif self.colorspace == 'BGR':
                        img = cv2.resize(cv2.imread(img_path)[100:], (self.width, self.height))
            
                    img = self.normalize(img)
                    imgs.append(img)
                    
                X_batch.append(imgs)
                y_batch.append(self.y_test[test_idx])
                # save the last image path
                self.vis_img_path = img_path

            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)
        
        else:
            X_batch = [] # (B,T,H,W,C)
            y_batch = [] # (B,2)
            for i in range(batch_size):
                test_idx = (self.__test_batch_pointer + i) % self.num_test
                imgs = []
                for j in range(self.seq_len):
                    end_img_path = self.Xs_test[test_idx,j,0]
                    end_img = cv2.imread(end_img_path, cv2.IMREAD_GRAYSCALE)
                    end_img = np.reshape(end_img, (480,640,1))
                    end_img = self.normalize(end_img)
                    # load flow
                    path_prefix = os.path.dirname(end_img_path)
                    img_name = end_img_path.split('/')[-1]
                    flow_path = os.path.join(path_prefix+'_bin', img_name[:-3]+'bin')
                    flow = np.fromfile(flow_path, dtype=np.float16).reshape((480,640,2))
                    flow = np.asarray(flow, dtype=np.float32)
                    flow = self.normalize(flow)
                    
                    iflow = np.concatenate((end_img, flow), axis=2)
                    iflow = cv2.resize(iflow[100:], (self.width, self.height))

                    imgs.append(iflow)

                X_batch.append(imgs)
                y_batch.append(self.y_test[test_idx])
                self.vis_img_path = end_img_path

            X_batch = np.array(X_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
    
            if self.seq_len == 1:
                X_batch = np.squeeze(X_batch, axis=1)

        return X_batch, y_batch

    def load_cloud_train_batch(self, batch_size):
        """ Load batch of the train point cloud data from bin file
        Return:
         - X_batch: (B,N,3), (B,N,4) or (B,T,N,4) 
         - y_batch: (B,2)
        Note: When use side cam, the translation of lidar in y is 0.508 m
        """

        X_batch = [] # (B,T,N,4)
        y_batch = []
        for i in range(batch_size):
            train_idx = (self.__train_batch_pointer + i) % self.num_train
            clouds = [] # (T,N,4)
            for j in range(self.seq_len):
                cloud_path = self.Xs_train[train_idx,j,1]
                trans_flag = float(self.Xs_train[train_idx,j,2])
                if self.random_point:
                    index = np.random.choice(20000, self.num_point, replace=False)
                else:
                    np.random.seed(10)
                    index = np.random.choice(20000, self.num_point, replace=False)
                if self.input_cfg in ['XYZ']:
                    cloud = np.fromfile(cloud_path).reshape(-1,4)
                    cloud = cloud[index,0:3]
                    cloud = cloud + np.array([0, trans_flag*0.508, 0]) # trans correction
                elif self.input_cfg in ['XYZF', 'XYZF-T', 'GRAYF-XYZF-T']:
                    cloud = np.fromfile(cloud_path).reshape(-1,4)
                    cloud = cloud[index] + np.array([0, trans_flag*0.508, 0, 0]) # trans correction
                else:
                    raise TypeError
                clouds.append(cloud)
                
            X_batch.append(clouds)
            y_batch.append(self.y_train[train_idx])
        
        X_batch = np.array(X_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)
        if self.seq_len == 1: X_batch = np.squeeze(X_batch, axis=1)

        self.__train_batch_pointer += batch_size

        return X_batch, y_batch


    def load_cloud_val_batch(self, batch_size):
        """ Load batch of the val point cloud from bin file
        Return:
         - X_batch: (B,T,N,4) or (B,N,4)
         - y_batch: (B,2)
        """

        X_batch = [] # (B,T,N,4)
        y_batch = []
        for i in range(batch_size):
            val_idx = (self.__val_batch_pointer + i) % self.num_val
            clouds = [] # (T,N,3)
            for j in range(self.seq_len):
                cloud_path = self.Xs_val[val_idx,j,1]
                trans_flag = float(self.Xs_val[val_idx,j,2])
                np.random.seed(10)
                index = np.random.choice(20000, self.num_point, replace=False)
                if self.input_cfg in ['XYZ']:
                    cloud = np.fromfile(cloud_path).reshape(-1,4)
                    cloud = cloud[index,0:3]
                    cloud = cloud + np.array([0, trans_flag*0.508, 0]) # trans correction
                elif self.input_cfg in ['XYZF', 'XYZF-T', 'GRAYF-XYZF-T']:
                    cloud = np.fromfile(cloud_path).reshape(-1,4)
                    cloud = cloud[index] + np.array([0, trans_flag*0.508, 0, 0]) # trans correction
                else:
                    raise TypeError
                clouds.append(cloud)
                
            X_batch.append(clouds)
            y_batch.append(self.y_val[val_idx])
        
        X_batch = np.array(X_batch, np.float32)
        y_batch = np.array(y_batch, np.float32) 
        if self.seq_len == 1: X_batch = np.squeeze(X_batch, axis=1)
        
        self.__val_batch_pointer += batch_size

        return X_batch, y_batch


    def load_cloud_test_batch(self, batch_size):
        """ Load batch of the test cloud data from bin file
        Return:
         - X_batch: (B,T,N,3) or (B,N,4)
         - y_batch: (B,2)
        """

        X_batch = [] # (B,T,N,3)
        y_batch = []
        for i in range(batch_size):
            test_idx = (self.__test_batch_pointer + i) % self.num_test
            clouds = [] # (T,N,3)
            for j in range(self.seq_len):
                cloud_path = self.Xs_test[test_idx,j,1]
                cloud = np.fromfile(cloud_path).reshape(-1,4)
                np.random.seed(10)
                index = np.random.choice(20000, self.num_point, replace=False)
                if self.input_cfg in ['XYZ']:
                    cloud = cloud[index,0:3]
                elif self.input_cfg in ['XYZF', 'XYZF-T', 'GRAYF-XYZF-T']:
                    cloud = cloud[index]
                else:
                    raise TypeError
                clouds.append(cloud)
                
            X_batch.append(clouds)
            y_batch.append(self.y_test[test_idx])
        
        X_batch = np.array(X_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)
        if self.seq_len == 1: X_batch = np.squeeze(X_batch, axis=1)
        
        self.__test_batch_pointer += batch_size

        return X_batch, y_batch
    
    def load_cloud_visual_batch(self):
        """ Load batch of the test cloud data from bin file
        Return:
         - X_batch: (B,T,N,3) or (B,N,4)
         - y_batch: (B,2)
        """
        batch_size = 1
        X_batch = [] # (B,T,N,3)
        y_batch = []
        for i in range(batch_size):
            test_idx = (self.__test_batch_pointer + i) % self.num_test
            clouds = [] # (T,N,3)
            for j in range(self.seq_len):
                cloud_path = self.Xs_test[test_idx,j,1]
                cloud = np.fromfile(cloud_path).reshape(-1,4)
                index = np.random.choice(20000, self.num_point, replace=False)
                if self.input_cfg in ['XYZ']:
                    cloud = cloud[index,0:3]
                elif self.input_cfg in ['XYZF', 'XYZF-T', 'GRAYF-XYZF-T']:
                    cloud = cloud[index]
                else:
                    raise TypeError
                clouds.append(cloud)
                
            X_batch.append(clouds)
            y_batch.append(self.y_test[test_idx])
        
        X_batch = np.array(X_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)
        if self.seq_len == 1: X_batch = np.squeeze(X_batch, axis=1)

        return X_batch, y_batch


    def load_train_batch(self, batch_size):
        """ Load image batch and point batch for train
        """
        if self.input_cfg in ['BGR', 'GRAY', 'GRAYF', 'GRAYF-T']:
            X_image_batch, y_batch = self.load_image_train_batch(batch_size)
            X_cloud_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
        elif self.input_cfg in ['XYZ', 'XYZF', 'XYZF-T']:
            X_image_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
            X_cloud_batch, y_batch = self.load_cloud_train_batch(batch_size)
        elif self.input_cfg in ['GRAYF-XYZF-T']:
            X_image_batch, y_batch = self.load_image_train_batch(batch_size)
            self.__train_batch_pointer -= batch_size
            X_cloud_batch, _ = self.load_cloud_train_batch(batch_size)

        return X_image_batch, X_cloud_batch, y_batch

    def load_val_batch(self, batch_size):
        if self.input_cfg in ['BGR','GRAY', 'GRAYF', 'GRAYF-T']:
            X_image_batch, y_batch = self.load_image_val_batch(batch_size)
            X_cloud_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
        elif self.input_cfg in ['XYZ', 'XYZF', 'XYZF-T']:
            X_image_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
            X_cloud_batch, y_batch = self.load_cloud_val_batch(batch_size)
        elif self.input_cfg in ['GRAYF-XYZF-T']:
            X_image_batch, y_batch = self.load_image_val_batch(batch_size)
            self.__val_batch_pointer -= batch_size
            X_cloud_batch, _ = self.load_cloud_val_batch(batch_size)

        return X_image_batch, X_cloud_batch, y_batch

    def load_test_batch(self, batch_size):
        if self.input_cfg in ['BGR','GRAY', 'GRAYF', 'GRAYF-T']:
            X_image_batch, y_batch = self.load_image_test_batch(batch_size)
            X_cloud_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
        elif self.input_cfg in ['XYZ', 'XYZF', 'XYZF-T']:
            X_image_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
            X_cloud_batch, y_batch = self.load_cloud_test_batch(batch_size)
        elif self.input_cfg in ['GRAYF-XYZF-T']:
            X_image_batch, y_batch = self.load_image_test_batch(batch_size)
            self.__test_batch_pointer -= batch_size
            X_cloud_batch, _ = self.load_cloud_test_batch(batch_size)

        return X_image_batch, X_cloud_batch, y_batch

    def load_visual_batch(self, index):
        ''' Load model test inputs by the index with the batch size of 1 
        '''
        self.__test_batch_pointer = index
        self.vis_img_path = None
        if self.input_cfg in ['BGR','GRAY', 'GRAYF', 'GRAYF-T']:
            X_image_batch, y_batch = self.load_image_visual_batch()
            X_cloud_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
        elif self.input_cfg in ['XYZ', 'XYZF', 'XYZF-T']:
            X_image_batch = np.zeros(shape=(0,0,0),dtype=np.float32)
            X_cloud_batch, y_batch = self.load_cloud_visual_batch()
        elif self.input_cfg in ['GRAYF-XYZF-T']:
            X_image_batch, y_batch = self.load_image_visual_batch()
            X_cloud_batch, _ = self.load_cloud_visual_batch()

        return X_image_batch, X_cloud_batch, y_batch, self.vis_img_path

    def compute_flow(self, pre_img, end_img):
        flow = cv2.calcOpticalFlowFarneback(pre_img, end_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        return flow
    
    def normalize(self, img):
        """ normalizing and centering data [-1,1]
        (img-127.5)/127.5
        """
        #
        return (img - np.mean(img)) / np.std(img)
        
    def scale_label(self, y, y_min, y_max, a, b):
        """Sacle labels to a fixed interval [a,b]
        """
        return a + (b-a)*(y-y_min)/(y_max-y_min)
    
    def draw_flow(self, im, flow, step=8):
        '''
         - im: the grayscale channel (0) for GRAYF image
         - flow: the flow channels (1:3) for GRAYF image
        Note:
         1. DO NOT normalize im and flow to show more intuitive visualization 
         2. To show: plt.imshow(draw_flow(im, flow))
        '''
        h,w = im.shape[:2]
        y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
        fx,fy = flow[y,x].T
    
        # create line endpoints
        lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
        lines = np.int32(lines)
    
        # create image and draw
        vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        for (x1,y1),(x2,y2) in lines:
            cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.circle(vis,(x1,y1),1,(0,255,0), -1)

        return vis

    def image_process(self, image):
        ## random brightness
        brightness_change = np.random.randint(0,100,dtype=np.uint8)
        if (np.random.uniform(0,1) > 0.5):
            image[image>(255-brightness_change)] = 255
            image[image<=(255-brightness_change)] += brightness_change
        else:
            image[image<(brightness_change)] = 0
            image[image>=(brightness_change)] -= brightness_change
    
        return image

    def __camera_adjust(self, angles, speeds, camera):
        """ Change angle labels according speed and camera frame
        Hints:
          1. Left camera -20 inches, right camera +20 inches (x-direction)
          2. Steering should be correction+current steering for center camera
          3. Reaction time: Time to return to center. (The literature seems to prefer 2.0s (probably really depends on speed)
        Note:
         1. The unit of angle is radians, the counterclockwise is positive (left turn)
         2. The unit of speed is ft/s  1 ft = 0.3048 m; 1 ft = 12 inch
         3. 1 inch = 0.0254 m; 
        """
        angles_adj = []
        for i in range(len(angles)):
            angle = angles[i]
            speed = speeds[i]
            if speed < 1.0:
                reaction_time = 0
                angle = angle
            else:
                reaction_time = 1.0 # Seconds
                # Trig to find angle to steer to get to center of lane in 2s
                opposite = 20.0 # inches = 0.508 m
                adjacent = speed*reaction_time*12.0 #(ft/s)*s*(12 in/ft) inches: y-direction
                angle_adj = np.arctan(float(opposite)/adjacent) # radians
                
                # Adjust based on camera being used and steering angle for center camera
                if camera == 'left':
                    angle_adj = -angle_adj
                angle = angle_adj + angle
            
            angles_adj.append(angle)

        return np.asarray(angles_adj, dtype=np.float32)

    def balance_data(self, Xs, y, 
                     balance_steer, 
                     balance_speed, 
                     bin_count,
                     fix_times):
        """ balance steer angle or speed distribution
        Args: 
        - Xs: (N, T, 3) [img_path, point_path, cam_flag]
        - y: (N,2) # [steer, speed]
        - bin_count: bins of distribution, determine by experiments
        - fix_times: expand dataset
        """
        # read labels to be fixed
        
        if balance_steer and not balance_speed:
            labels = [item[0] for item in y]
        elif not balance_steer and balance_speed:
            labels = [item[1] for item in y]
        else:
            raise TypeError
            
        # Getting bin counts of the labels
        hist, bins = np.histogram(labels, bins=bin_count, range=(min(labels)-0.0001, max(labels)+0.0001))
        desired_hist = int(np.mean(hist)*fix_times) # 扩充倍数可根据实际情况调整
        
        aug_Xs, aug_y = self.augment_data(Xs, y, hist, bins, desired_hist, balance_steer)
        
        trim_Xs, trim_y = self.trim_data(Xs, y, hist, bins, desired_hist, balance_steer)
        
        Xs = np.concatenate((aug_Xs, trim_Xs), axis=0)
        y  = np.concatenate((aug_y, trim_y), axis=0)
        
        return Xs, y
        
    def augment_data(self, Xs, y, hist, bins, desired_hist, balance_steer):
        """ augment IDs and labels
        """
        copy_times = np.float32((desired_hist-hist)/hist) # 小于0的倍数不会被复制
        copy_times_accum = np.zeros_like(copy_times)
        
        aug_Xs = []
        aug_y = []
        assert len(Xs)==len(y)
        for i in range(len(Xs)):
            xs = Xs[i]
            label = y[i]
            
            if balance_steer:
                index = np.digitize(label[0], bins)-1 # 返回数据在bin中的位置
            else:
                index = np.digitize(label[1], bins)-1
        
            copy_times_accum[index] += copy_times[index] # 有可能是小数，累加到1时复制1次，同时accum 减掉1，恢复累加
            copy_times_integer = np.int32(copy_times_accum[index]) # 向下取整
            copy_times_accum[index] -= copy_times_integer # 累积丢弃的小数倍
            
            for j in range(copy_times_integer): # 负倍数不会被复制
                # 涉及二维列表时，使用deepcopy
                aug_Xs.append(xs)
                aug_y.append(label)
                
        return aug_Xs, aug_y

    def trim_data(self, Xs, y, hist, bins, desired_hist, balance_steer):
        """  trim IDs and labels
        """
        keep_ratio = np.float32(desired_hist/hist)
        trim_Xs=[]
        trim_y=[]
        for i in range(len(Xs)):
            xs = Xs[i]
            label = y[i]
            
            if balance_steer:
                prob_to_keep = keep_ratio[np.digitize(label[0], bins)-1]
            else:
                prob_to_keep = keep_ratio[np.digitize(label[1], bins)-1]
            
            random_prob = np.random.uniform(0,1)
            if random_prob <= prob_to_keep:
                trim_Xs.append(xs)
                trim_y.append(label)
                
        return trim_Xs, trim_y

    def label_distribution(self, label):
        steer = label[:,0]
        speed = label[:,1]

        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,3), dpi=300)
        ax1.hist(steer, rwidth=0.5)
        ax2.hist(speed, rwidth=0.5)
        fig.savefig('dist')



if __name__ == '__main__':
    import argparse
    import time 
    import matplotlib.pyplot as plt
    import open3d as o3d
    from tools.cloud_visualizer import Visualizer

    ## ===  Evaluation Procedure === ## 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/ubuntu16/Documents/Datasets/Udacity/CH2', help='dataset path')
    parser.add_argument('--test_mode', default='7', help='set 1-7 to test all input cfg')
    parser.add_argument('--aug_cfg', default='None', help='None, IA, RP, SC, BA, BS')

    Flags = parser.parse_args()

    if Flags.test_mode == '0':
        dataloader = DataLoader(Flags.data_root, input_cfg='BGR', 
                                height=66, width=200,
                                seq_len = None,
                                num_point= None,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        X_image_batch, _, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            fig = plt.figure()
            plt.imshow(X_image_batch[i])
            plt.show()
    elif Flags.test_mode == '1':
        dataloader = DataLoader(Flags.data_root, input_cfg='GRAY', 
                                height=200, width=200, 
                                seq_len = None,
                                num_point= None,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        X_image_batch, _, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            fig = plt.figure()
            plt.imshow(np.squeeze(X_image_batch[i], axis=-1), cmap='gray')
            plt.show()
            
    elif Flags.test_mode == '2':
        dataloader = DataLoader(Flags.data_root, input_cfg='GRAYF', 
                                height=200, width=200, 
                                seq_len = None,
                                num_point= None,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        X_image_batch, _, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            im = X_image_batch[i,:,:,0]
            #im = np.asarray(im, np.uint8)
            flow = X_image_batch[i,:,:,1:3]*5
        
            fig = plt.figure()
            plt.imshow(dataloader.draw_flow(im, flow))
            plt.show()

    elif Flags.test_mode == '3':
        dataloader = DataLoader(Flags.data_root, input_cfg='GRAYF-T', 
                                height=200, width=200, 
                                seq_len = 5,
                                num_point= None,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        seq_len = 5
        X_image_batch, _, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            for t in range(seq_len):    
                im = X_image_batch[i,t,:,:,0]
                flow = X_image_batch[i,t,:,:,1:3]*5
            
                fig = plt.figure()
                plt.imshow(dataloader.draw_flow(im, flow))
                plt.show()

    elif Flags.test_mode == '4':
        dataloader = DataLoader(Flags.data_root, input_cfg='XYZ', 
                                height=200, width=200,
                                seq_len = None,
                                num_point= Flags.num_point, 
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        _, X_cloud_batch, y_batch = dataloader.load_val_batch(batch_size)
        print(X_cloud_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)
        
        vis = Visualizer(play_mode='manual')
        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i])
            vis.show([cloud])

    elif Flags.test_mode == '5':
        dataloader = DataLoader(Flags.data_root, input_cfg='XYZF', 
                                height=200, width=200, 
                                seq_len = None,
                                num_point= Flags.num_point,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 16
        for k in range(3): # test batch
            _, X_cloud_batch, y_batch = dataloader.load_val_batch(batch_size)
            print(X_cloud_batch.shape, y_batch.shape)

            angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
            speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)
            
            vis = Visualizer(play_mode='manual')
            for i in range(batch_size):
                print('Angle: ', angle[i])
                print('Speed: ', speed[i])
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i][:,:3])
                vis.show([cloud])

    elif Flags.test_mode == '6':
        dataloader = DataLoader(Flags.data_root, input_cfg='XYZF-T', 
                                height=200, width=200, 
                                seq_len = 5,
                                num_point= Flags.num_point,
                                use_side_cam=True)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        batch_size = 16
        seq_len = 5
        _, X_cloud_batch, y_batch = dataloader.load_train_batch(batch_size)
        print(X_cloud_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)
        
        vis = Visualizer(play_mode='manual')
        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            cloud = o3d.geometry.PointCloud()
            for t in range(seq_len):
                cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i,t,:,:3])
                vis.show([cloud])

    elif Flags.test_mode == '7':
        dataloader = DataLoader(Flags.data_root, input_cfg='GRAYF-XYZF-T', 
                                height=200, width=200,
                                seq_len = 5,
                                num_point= 10000,
                                aug_cfg=Flags.aug_cfg)
        print('Train label: ', dataloader.Xs_train.shape, dataloader.y_train.shape)
        print('Val label: ',   dataloader.Xs_val.shape, dataloader.y_val.shape)
        print('Test label: ',  dataloader.Xs_test.shape, dataloader.y_test.shape)
        
        batch_size = 5
        seq_len = 5
        X_image_batch, X_cloud_batch, y_batch = dataloader.load_train_batch(batch_size)
        print(X_image_batch.shape, X_cloud_batch.shape, y_batch.shape)

        angle = dataloader.scale_label(y_batch[:,0],-1,1,-2,2)
        speed = dataloader.scale_label(y_batch[:,1],-1,1,0,30)

        vis = Visualizer(play_mode='manual')
        for i in range(batch_size):
            print('Angle: ', angle[i])
            print('Speed: ', speed[i])
            cloud = o3d.geometry.PointCloud()
            for t in range(seq_len):
                fig = plt.figure()
                plt.imshow(X_image_batch[i,t,:,:,0], cmap='gray')
                plt.show()
                cloud.points = o3d.utility.Vector3dVector(X_cloud_batch[i,t,:,:3])
                vis.show([cloud])
