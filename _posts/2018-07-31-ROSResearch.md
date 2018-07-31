---
title: "ROS Research Topic"
date: 2018-07-31
classes: wide
use_math: true
tags: ROS Gazebo research pcl
category: ros
---


# An Introduction to Robot Operating System (ROS)
[An Introduction to Robot Operating System (ROS)](https://www.allaboutcircuits.com/technical-articles/an-introduction-to-robot-operating-system-ros/)  
![ROS goodbasic](https://www.allaboutcircuits.com/uploads/articles/Gazebo_moving_turt_thumb.jpg){:height="40%" width="40%"}  


# ROS 기반 다중 객체 비전 인식 (pcl)
![ROS pclplane](../../pictures/ros/pclplane.png){:height="40%" width="40%"}  
[ROS 기반 다중 객체 비전 인식 (pcl)](http://daddynkidsmakers.blogspot.com/2015/08/ros_27.html)

# Extracting indices from a PointCloud
![planepcl](../../pictures/ros/planepcl.png){:height="40%" width="40%"}  
[Extracting indices from a PointCloud](http://pointclouds.org/documentation/tutorials/extract_indices.php#extract-indices)

# ROS Controller for autonomous vehicle, using opencv for the neural network input. Using Kinect and Interface controller 
![ROSdriving](../../pictures/ros/rosdriving.png){:height="40%" width="40%"}  
[ROS Controller for autonomous vehicle, using opencv for the neural network input. Using Kinect and Interface controller ](https://github.com/dsapandora/SIMUL_DSA)


# How to display bag file in rviz
```
$ rosrun velodyne_driver velodyne_node _mol:=VLP16 _pcap:=/home/soowon/Documents/County_Fair.pcap _read_once:=true

On new terminal, to record pcap data into bag file:

$ rosrun rosbag record -O Countyfair.bag /velodyne_packets

ON new terminal, I used rosbag to play bag file that I just made:

$ rosbag play Countyfair.bag

Another new terminal, to open rviz:

$ rosrun rviz rviz
```

# ROS의 RViz로 세계적인 수술용 로봇 daVinci 다빈치를 만나보자
![rosdavinci](../../pictures/ros/rosdavinci.png){:height="40%" width="40%"}  
[ROS의 RViz로 세계적인 수술용 로봇 daVinci 다빈치를 만나보자](http://pinkwink.kr/906)

# ROS기반 비전 기반 평면 객체 인식 
[ROS기반 비전 기반 평면 객체 인식 ](http://daddynkidsmakers.blogspot.com/2015/08/blog-post_59.html)

# TK1 기반 ROS, 센서, 무선 및 배터리 설치 
[TK1 기반 ROS, 센서, 무선 및 배터리 설치 ](http://daddynkidsmakers.blogspot.com/2015/08/tk1-ros.html)

# ROS Tutorial 1: Use ROS to get image with an USB camera
[ROS Tutorial 1: Use ROS to get image with an USB camera](http://ninghang.blogspot.com/2011/11/tutorial-1-use-ros-to-get-image-with.html)

# Pixhawk와 ROS를 이용한 자율주행
![pixhawk](../../pictures/ros/pixhawk.png){:height="40%" width="40%"}  
[Pixhawk와 ROS를 이용한 자율주행](http://www.modulabs.co.kr/board_GDCH80/2870)

# Laser Odometry and Mapping (Velodyne version)
[Laser Odometry and Mapping (Velodyne version)](http://wiki.ros.org/loam_velodyne)  
![Laser Odometry and Mapping (Velodyne version)](http://wiki.ros.org/loam_velodyne?action=AttachFile&do=get&target=maps.jpg)

# rosbridge-csharp-connection
[rosbridge-csharp-connection](https://github.com/horverno/rosbridge-csharp-connection/)

# How to Calibrate a Monocular Camera
[How to Calibrate a Monocular Camera](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration)

# Writing a ROS Python Makefile
[Writing a ROS Python Makefile](http://wiki.ros.org/rospy_tutorials/Tutorials/Makefile)

