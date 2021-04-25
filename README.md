<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-05-10 09:29:01
 * @LastEditTime: 2021-04-26 00:12:10
 -->
# FlowDriveNet

## 1. Introduction
<div align=center> <img src=./docs/cover.png /></div>

Learning driving policies using an end-to-end network has been proved a promising solution for autonomous driving. Due to the lack of a benchmark driver behavior dataset that contains both the visual and the LiDAR data, existing works solely focus on learning driving from visual sensors. Besides, most works are limited to predict steering angle yet neglect the more challenging vehicle speed control problem. 

In this project, we propose release our __Code__ and __Dataset__ for training our FlowDriveNet, which is a novel end-to-end network and takes advantages of sequential visual data and LiDAR data jointly to predict steering angle and vehicle speed. 

## 2. Requirements
* Python 3.x
* Tensorflow 1.x.x
* Python Libraries: numpy, scipy and __laspy__

## 3. Dataset
### 3.1 Udacity CH2-LiDAR
The dataset used in this project is created from the Udacity `CH2` dataset, which is a popular open source driving dataset and can be used for the vision-based driving learning methods. However, the original dataset does not contain LiDAR data, then we extract the raw point cloud data and remove distortions from the original ROS bag file.

**Pipeline for creating this data:**
 - Extracting images (center, right, left) and corresponding label file from bag file. (6311,6311,6288)
 - Extracting Point Cloud data from bag file and save to seperate pcd file. (3096)
   - fix calibration file
   - covert raw velodyne point packet topic to PointCloud2 topic
   - save to pcd file using ros node
 - Registering the point cloud with images and labels using timestamps.(3096)

**Related tools:**
 - Data extraction tool: [udacity launch](https://github.com/wsustcid/self-driving-car/tree/master/datasets/udacity_launch)
 - Optical Flow and Point Flow extraction tool: './data/data_prepare.py' for more implementation details.


## 4. Train & Evaluation
```python
# single GPU
python train.py --data_root xx --input_config xx ...

# Multiple GPU
python train_multi-gpus.py

# Evaluation 
python eval.py
```

## 5. Citation
```
@article{wang,
  title={FlowDriveNet: An End-to-End Network for Learning Driving Policies from Image Optical Flow and LiDAR Point Flow},
  author={Shuai Wang, Jiahu Qin, Menglin Li and Yaonan Wang},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

## 6. License
This Project is released under the [Apache licenes](LICENSE).

