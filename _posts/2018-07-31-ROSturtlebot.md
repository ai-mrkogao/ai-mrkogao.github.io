---
title: "ROS basic and turtlebot example"
date: 2018-07-31
classes: wide
use_math: true
tags: ROS Gazebo turtlebot
category: ros
---

# ROS Architecture 
![ROS architecture](../../pictures/ros/rosbasicarchitecture.png){:height="50%" width="50%"}

- ROS works on SBC(single board computer) like Rasberry Pi,Intel Edison, BeagleBone

![ROS architecture2](../../pictures/ros/rosbasicarchitecture2.png){:height="50%" width="50%"}

- ROS nodes send and receives msgs which includes topic
- ROS nodes are publisher or subscriber 
- ROS master has each nodes information and some kind of server

![ROS node](../../pictures/ros/nodearchitecture.png){:height="50%" width="50%"}  
![ROS node](../../pictures/ros/nodearchitecture2.png){:height="50%" width="50%"}  
![ROS node](../../pictures/ros/nodearchitecture3.png){:height="50%" width="50%"}  

# ROS Install
- setting sources.list 
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
- setting key 
```
sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net --recv-key 0xB01FA116
```

- package update
```
sudo apt-get update
```

- install ROS
```
sudo apt-get install ros-indigo-desktop-full
```
```
sudo rosdep init
rosdep update
```


- catkin: ROS build system

# ROS setting working environment

## ROS bash shell run
```
source /opt/ros/indigo/setup.bash
```

## create working directory and init
```
mkdir -p ~/catkin_ws/src  

cd ~/catkin_ws/src        

catkin_init_workspace     
```

## build
```
cd ~/catkin_ws/  

$ catkin_make    
```
- check build, devel, src in the catkin_ws folder after building

```
source ~/catkin_ws/devel/setup.bash # catkin command bash shell
```

## parameter setting in bashrc
```
gedit ~/.bashrc 
source /opt/ros/indigo/setup.bash   

source ~/catkin_ws/devel/setup.bash 
export ROS_MASTER_URI=http://XXX.XXX.XXX.XXX:11311  -> master 주소를 설정합니다. XXX.XXX.XXX.XXX에 ip를 입력합니다.

export ROS_HOSTNAME=XXX.XXX.XXX.XXX  -> host ip를 입력합니다. Master와 host robot이 다른 경우엔 각각의 ip를, 같은 경우엔 master와 같은 ip를 입력합니다.


alias cw='cd ~/catkin_ws'   ->  cw 명령어는 catkin workspace 디렉토리로 이동합니다.

alias cs='cd ~/catkin_ws/src'->  cs 명령어는 catkin workspace의 source 디렉토리로 이동합니다.

alias cm='cd ~/catkin_ws && catkin_make' ->  cm 명령어는 catkin workspace로 이동한 뒤 ROS 패키지를 build합니다.
```

## Example
```
roscore -> run ros master 
#start new command window 
rosrun turtlesim turtlesim_node      ->  turtlesim package의 turtlesim_node 노드 실행
#start new command window
rosrun turtlesim turtle_teleop_key  -> turtlesim package의 turtle_teleop_key 노드 실행
```
![ROS turtlesim](../../pictures/ros/turtlesimexample.png){:height="80%" width="80%"}  

```
#start new command window
rosrun rqt_graph rqt_graph -> show nodes and messages relationship
```
![ROS rqtgraph](../../pictures/ros/rqtgraph.png){:height="80%" width="80%"}  
 

# ROS 설치하기 (몹시 초보자용)
- ROS install command 
- create package with catking ws 

[ROS 설치하기 (몹시 초보자용)](http://pinkwink.kr/880) 


# ROS를 사용할 때 자주 사용할 기초 명령 및 기능 구경하기
[ROS를 사용할 때 자주 사용할 기초 명령 및 기능 구경하기](http://pinkwink.kr/886?category=558361)
```
roscore
rosnode list
rosnode info /rosout
rosrun turtlesim turtlesim_node
rosrun turtlesim turtlesim_node __name:=my_turtle
rosnode ping my_turtle
rosrun turtlesim turtle_teleop_key
rosrun rqt_graph rqt_graph
rostopic echo /turtle1/cmd_vel
rostopic list -v
rosmsg show geometry_msgs/Twist
rostopic pub -1 /turtle1/cmd_vel geometry_msgs/Twist -- '[2.0, 0.0, 0.0]' '[0.0, 0.0, 1.8]'
rosrun rqt_plot rqt_plot
```

# Reference sites
[ROS(Robot Operating System) 개념과 활용 - 2. ROS의 동작 구조와 적용 사례](http://enssionaut.com/xe/index.php?mid=board_robotics&page=2&document_srl=421)

