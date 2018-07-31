---
title: "ROS pixhawk"
date: 2018-07-31
classes: wide
use_math: true
tags: ROS Gazebo turtlebot pixhawk
category: ros
---

# Pixhawk와 ROS를 이용한 자율주행

# 1. Install Ubuntu on External Disk

## PX4 Introduction
![pixhawk](../../pictures/pixhawk/pixhawk.png){:height="50%" width="50%"}

- PX4 Architecture
[PX4 dev site](http://dev.px4.io/kr/concept/architecture.html)

- PX4 Autopilot
The PX4 flight stack is part of the Dronecode platform, an end-to-end solution for building and managing Unmanned Aerial Vehicles (AUV).

The platform has several parts:
    - PX4 Flight Stack: The flight control system (autopilot).
    - MAVLink: A highly efficient, lightweight and blazing fast robotics communication toolkit.
    - QGroundControl: Modern, mobile and desktop user interface to configure the system and execute flights.
[PX4 autopilot](http://px4.io/technology/)

- MAVLINK 

![mavlink](../../pictures/pixhawk/mavlink.png){:height="50%" width="50%"}

## Ubuntu Install
- Ubuntu Download  
[Ubuntu Download](http://www.ubuntu.com/download/alternative-downloads)
- Bootable USB 
[Create USB](http://www.ubuntu.com/download/desktop/create-a-usb-stick-on-windows)
- Ubuntu boot in external device mode


- Install

여기서 중요한 게 저희는 기타로 가줘야합니다!
그 다음페이지에서 sda가 있고 sdc가 있는데 순서대로 a,b,c, 이런식으로 이름이 붙는 것 같은데 usb가 꽂혀 있으므로 HDD는 c로 인식하지 않았나 싶습니다. 캡쳐를 안해놔서 사진이 없네요. 이 sdc를 포맷하고 파티션을 새로 짜줘야합니다. 저는 다음 홈페이지를 참조하였습니다.
[setting partition](http://deviantcj.tistory.com/434)

여기서 말하듯이 파티션은 세가지로 나눠줘야하는데
   (1) 주 파티션하나(/)
   (2) swap영역하나
   (3) 논리파티션(/home)이 잇는데
  
   (1)과 (3)에 마운트위치를 까먹으시면 안되요!

주 파티션에 저는 40기가를 할당했고 swap영역에는 보통 ram의 두 배를 사용한다합니다. 따라서 저는 4기가를 할당했고 나머지는 /home에 몰아줍니다.

## 윈도우에서 우분투 듀얼부팅하기(멀티부팅)

[multi boot](http://palpit.tistory.com/765)

위 페이지에 들어가보시면 아주 잘 설명이 되어있습니다.

앞에서의 외장하드 설치와는 다른 것은 현재 하드디스크의 파티션을 나눠주고 나눠준 파티션에 우분투를 설치해준다는 것입니다. 그 파티션에서 우분투의 파티션을 정하는 것과 USB로 부팅하는 것등은 다 동일하다고 보시면 될 것 같습니다.

단, 위 페이지에서 설명하지 않은 것은 파티션을 나누는 것은 4개의 파티션이 최대라서 그 이상으로 만들지 못합니다. 따라서 현재 파티션이 네 개인데 하나의 디스크(예를 들면 D드라이브)의 용량을 줄이고 우분투를 만들 파티션을 만들려고 하면 오류가 발생한다고 합니다. 그러한 경우에는 하나를 없애줘야한다고 합니다.

또한 우분투의 파티션을 나누는 것은 설치할 때 설치화면에서 나눠도 되지만 설치하는 첫 화면에서 Try Ubuntu

without installing을 선택해서 우분투에 Gparted를 설치하셔서 하면 더 잘 된다 합니다. 제가 아는 지인도 이런식으로 문제를 해결했었습니다. Ubuntu에서 Gparted설치는 터미널창에서 다음을 실행시키면 됩니다.

sudo apt-get install gparted

이렇게 외장하드 설치와 듀얼부팅으로 설치하는 법을 살펴보았습니다. 우분투 설치하는 것이 쉽지는 않은 것 같습니다. 우분투 설치에 대한 설명하는 블로그들이 워낙 많지만 사람마다 조금씩 다른 것 같긴하네요. 저는 어려웠는데(ssd로 바꾸는 것부터 너무 생소한 지라...) 여러분은 쉬우셨기를!!


# 2. ROS install on Ubuntu
(1) Setup your sources.list
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
(2) Setup your key
```
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net --recv-key 0xB01FA116
```

(3)  Installation
```
sudo apt-get update

sudo apt-get upgrade

sudo apt-get install ros-indigo-desktop-full

위 명령어로만 설치가 안 될 경우에 다음을 실행해보세요
sudo apt-get install libsdformat1
```
(4)  Initialize rosdep
```
sudo rosdep init
rosdep update
```
(5)  getting rosinstall
```
sudo apt-get install python-rosinstall
```

(6) make workspace for ros
```
mkdir -p ~/catkin_ws/src

cd ~/catkin_ws/src

catkin_init_workspace
```

(7) catkin make test
```
cd ~/catkin_ws/

catkin_make
```
## setting ROS Indigo working environment 
```
gedit ~/.bashrc
```
![rosworkingenvironment](../../pictures/pixhawk/rosworkingenvironment.png){:height="70%" width="70%"}

## install ROS development tool 
(1) 설치
```
sudo apt-get install qtcreator
```
(2) 실행
```
qtcreator
```
![qtcreator](../../pictures/pixhawk/qtcreator.png){:height="70%" width="70%"}

## ROS test
```
roscore

rosrun turtlesim turtlesim_node
rosrun turtlesim turtle_teleop_key

```

# 3. install Gazebo6 
[gazebo homepage](http://gazebosim.org/)

1.  Setup your computer to accept software from packages.osrfoundation.org.
```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
```
2. Setup keys
```
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
```
3. eliminate previous gazebo and install gazebo 6
이전에 gazebo가 설치되어있다면 gazebo6가 설치되지 않습니다. 따라서 현재 ros를 설치하면서
같이 설치된 gazebo 구버전을 지워주셔야합니다.
```
sudo apt-get remove .*gazebo.* && sudo apt-get update && sudo apt-get install gazebo6
```
4. Run gazebo
gazebo

5. install extra package
  For developers that work on top of Gazebo, one extra package
    ```
	    sudo apt-get install libgazebo6-dev
    ```

6. Possible error
(1) 우분투와 가제보가 서로 버전이 맞아야합니다. 저는 ubuntu 14.04에 gazebo6를 설치하였는데 만약 ubuntu 12.04에 gazebo7을 설치하려하면 다음과 같은 에러가 발생합니다.
```
sudo apt-get install gazebo7
Reading package lists... Done
Building dependency tree       
Reading state information... Done
E: Unable to locate package gazebo7
```
이 같은 에러는 현재 ubuntu에서 사용할 수 없는 package를 설치하려 할 때 발생합니다. 앞으로 이러한 에러(unable to locate package)가 발생하였을 경우에는 버전이 호환이 되는지, 호환이 안되면 가능한 버전은 무엇인지 알아보시고 다시 명령을 주면 됩니다. 


# 4. install MAVROS and PX4
이번에는 ROS indigo와 gazebo6의 연동에 대해서 살펴보겠습니다.
임재영님이 만들어놓으신 gitpage를 참조하시면 좋을 것 같습니다.
(idigo and gazebo6 connection)[https://github.com/Jaeyoung-Lim/modudculab_ros]

1. gazebo_ros_pkgs설치
	terminal창을 키고

	```
	sudo apt-get install ros-indigo-gazebo6-ros-pkgs
	```
	을 입력해주시면 설치가 됩니다.

2. MAVROS 설치

	우선 apt-get을 업데이트 시켜줍니다.
	```
	sudo apt-get update
	```
	그리고 MAVROS를 설치해줍니다
	```
	sudo apt-get install ros-indigo-mavros ros-indigo-mavros-extras
	```
	위 문장을 터미널 창에 입력하면 설치가 됩니다.
3. PX4 설치

    이제 Pixhawk의 firmware인 PX4를 설치해줍니다
    ````
    sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
    sudo apt-get update
    sudo apt-get install python-argparse git-core wget zip \
              python-empy qtcreator cmake build-essential genromfs -y
    ````

    # simulation tools
    ```
    sudo apt-get install ant protobuf-compiler libeigen3-dev libopencv-dev openjdk-8-jdk openjdk-8-jre clang-3.5 lldb-3.5 -y
    ```

    이 명령은 에러가 많이 발생합니다. 이 부분에서 에러가 발생하는 것은 openjdko-8-jdk, openjdk-8-jre이 설치되지 않기 때문입니다.  그래서 이 두 개 빼고 나머지 를 설치해주시고 두 개는 따로 설치해주도록 했습니다.
    ```
    sudo apt-get install ant protobuf-compiler libeigen3-dev libopencv-dev clang-3.5 lldb-3.5 -y
    ```
    openjdko-8-jdk, openjdk-8-jre가 설치가 되지 않는 이유는 Ubuntu14.04에서 호환되는 패키지가 아니기 때문입니다. 검색해본 결과 Ubuntu14.04에서는  7버전까지 지원이 되어서 아래와 같이 바꿔서 명령을 주게되면 Java가 설치가 됩니다.
    ```
    sudo apt-get install openjdk-7-jdk openjdk-7-jre
    ```
    
    혹은 다음과 같은 방법으로 Java 8을 설치할 수 있습니다.

    http://tecadmin.net/install-oracle-java-8-jdk-8-ubuntu-via-ppa/#

    위 페이지에 따라서 다른 방법으로 자바를 설치했습니다.
    ```
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer
    ```
     
    하지만 저는 마지막 줄이 실행이 안되어서 다음 문장으로 대체했습니다.
    ```
    sudo apt-get install oracle-java8-set-default
    ```

    또한 다음과 같은 명령어로 설치가 된 자바의 버전을 확인할 수 있습니다.
    ```
    java -version
    ```
    ``` 
    cd ~/catkin_ws/src

    git clone https://github.com/PX4/Firmware.git

    cd Firmware

    git submodule update --init --recursive
    ```
    이렇게 차례대로 실행해주시면 설치가 완료됩니다

4. Iris drone gazebo test

    PX4가 설치가 되었다면 PX4안의 iris drone의 model이 들어있고 그 model을 gazebo로 실행시켜봅니다. gazebo안에 iris drone이 보이면 됩니다.
    ```
    cd ~/catkin_ws/src/Firmware

    make posix_sitl_default gazebo    
    ```
    ![](http://www.modulabs.co.kr/files/attach/images/214/886/001/fe15ff6b7c4b56a149c8c5a1ac6ed9ef.png){:height="50%" width="50%"}

    ![](http://www.modulabs.co.kr/files/attach/images/214/886/001/c2ad0afb95779960e8923477908be28b.png){:height="50%" width="50%"}


# 5. Software In the Loop

```
cd ~/catkin_ws/src
git clone https://github.com/Jaeyoung-Lim/modudculab_ros.git
catkin build modudculab_ros
```
```
roscore
```
roscore를 구동시킬 때 다음과 같은 문제가 발생할 수 있습니다.

![package](http://www.modulabs.co.kr/files/attach/images/214/767/002/0c9fe687a80f8f7c264a89828b777714.png){:height="50%" width="50%"}

이 경우에는 처음에 설정해줬던 IP주소가 달라졌기 때문인데 터미널 창에서 지금 IP를 검색해서 설정파일에서 고쳐주고 다시 실행해주면 실행이 됩니다. 관련 내용은 제 2번째 포스팅을 참고 하시길 바랍니다.

http://www.modulabs.co.kr/board_GDCH80/1562

roscore를 실행한 후에 다른 터미널 창을 킨 다음에 gazebo를 실행시킵니다. 저희는 gazebo에 default로 등록이 되어있는 iris drone을 사용합니다. 3DR사의 드론으로 링크는 https://store.3dr.com/products/IRIS.

gazebo의 좋은 점은 실재 iris 드론의 물성치들이 다 입력이 되어있어서 좀 더 실재와 가까운 시뮬레이션이
가능하다는 것입니다. 가제보를 default로 실행시킵니다.
```
cd ~/catkin_ws/src/Firmware
make posix_sitl_default gazebo
```
그 다음에 처음에 다운받은 패키지의 launch파일을 실행시켜줘야 합니다.
```
roslaunch modudculab_ros ctrl_pos_gazebo.launch
```

하지만 그냥 이대로 실행시키면 문제가 발생합니다. 패키지 내부의 파일에 고쳐줘야할 것이 있습니다.
ctrl_pos_gazebo.launch파일의 위치를 찾아주세요.
![](http://www.modulabs.co.kr/files/attach/images/214/767/002/0d23d81ea87fdcf72ee8ed36b851cf69.png){:height="50%" width="50%"}

즉 터미널 창에서 다음을 입력
```
cd ~/catkin_ws/src/modudculab_ros/launch
```
nano라는 edit프로그램을 사용해서 내용을 살펴봅시다.
```
nano ctrl_pos_gazebo.launch
```

![](http://www.modulabs.co.kr/files/attach/images/214/767/002/7ed9141c0441a3d660467a7581e8c969.png){:height="50%" width="50%"}

밑으로 내려서 node name="pub_setpoints" pkg="modudculab_ros" type="pub_setpoints_pos"이 문장에서 pkg내용을 고쳐줘야하는데 원래는 이상하게 되어있는데 "modudculab_ros"로 고쳐주셔야 launch파일이 정상적으로 작동할 수 있습니다

## Simulation of iris drone default 

다시 launch파일을 실행시켜줍니다.
```
roslaunch modudculab_ros ctrl_pos_gazebo.launch
```
![](http://www.modulabs.co.kr/files/attach/images/214/870/002/f1bfd4a4521ec3129131f01dfdeea8ce.png){:height="50%" width="50%"}

다른 터미널 창을 열고 이제 iris drone을 arming시켜줍니다.
```
rosrun mavros mavsafety arm
```
 
arming이 완료되었으면 가상 보드를 작동시킵니다.
```
rosrun mavros mavsys mode -c OFFBOARD
```
미션은 hovering으로서 (0,0,1)위치에서 떠있으라는 명령을 주었습니다. 명령은 /src/modudculab_ros/src/pub_setpoint_pos.cpp에 다음과 같이 적혀있습니다.

![](http://www.modulabs.co.kr/files/attach/images/214/870/002/e5189d3d02f0e7c60867668040bd733c.png){:height="50%" width="50%"}

이 명령을 받으면 iris drone이 다음과 같이 작동합니다.

![](http://www.modulabs.co.kr/files/attach/images/214/870/002/64a9ac3701ac0326821862b638fb8bdf.png){:height="50%" width="50%"}

gazebo가 상당히 좋은 점이 가상 보드로 사용하고 있는 픽스호크는 시뮬레이션에서 기압센서로 고도를 측정중인데 그 센서의 drift현상까지 표현하고 있다는 점입니다. 따라서 위와 같이 실행해놓고 시간이 좀 지나서 보면 더 위로 드론이 올라가 있는 것을 볼 수 있습니다.


## Altitude data comparison
실재로 데이터를 보면서 얼마나 차이가 있는지 보려고 합니다. 앞으로도 센서 데이터를 읽어오는 작업은 꼭 필요할 것이기 때문에 이 과정을 하고 넘어가도록 하겠습니다.

다시 처음으로 돌아가서 다음 명령어들을 차례로 실행시켜주고
```
roscore

make posix_sitl_default gazebo

roslaunch modudculab_ros ctrl_pos_gazebo.launch

rosrun mavros mavsafety arm
```
 
[ROS recording](http://wiki.ros.org/ROS/Tutorials/Recording%20and%20playing%20back%20data)

다음 명령어를 실행시켜주시면 현재  node사이에서 통신하고 있는 topic의 list를 볼 수 있습니다.

```
rostopic list -v
```
![](http://www.modulabs.co.kr/files/attach/images/214/870/002/4c228519945e954e135f9437357c4141.png){:height="50%" width="50%"}

제가 기록하고자 하는 topic은 /mavros/local_position/pose 입니다. 파일이름은 -O 뒤에 적어주시고 기록하고자 하는 topic을 적어줍니다. 다음과 같은 명령어를 새로운 터미널 창에서 실행해주셔야 합니다.
```
mkdir ~/bagfiles
cd ~/bagfiles
rosbag record -O iris_default_1 /mavros/local_position/pose
```
그 다음에 임무를 수행합니다.
```
rosrun mavros mavsys mode -c OFFBOARD
```
 
적당한 시간이 지난 후 record를 진행하던 터미널 창에서 ctrl+c로 record를 중지합니다. 그리고 다음 명령어를 실행해줍니다. info뒤에는 bagfiles폴더안에 기록된 파일의 이름을 적어주셔야 합니다.
```
rosbag info iris_default_1.bag
```

그 뒤에 다음 명령어를 실행시켜주었습니다.
```
rqt_bag
```
![](http://www.modulabs.co.kr/files/attach/images/214/870/002/8afed3fbf4a26934a77c9a24d270005c.png){:height="50%" width="50%"}

왼쪽 위에 빨간 점 옆의 열기버튼을 눌러서 기록한 파일을 열어주시기 바랍니다.

![](http://www.modulabs.co.kr/files/attach/images/214/870/002/b8ce6b237defb5bf61698c32e7768998.png){:height="50%" width="50%"}

열기버튼 밑의 mavros/local_position/pos를 마우스 오른쪽 버튼으로 누르고 publish를 누르고 view의 plot을 눌러줍니다 그럼 위와 같은 화면이 나오는데 오른쪽 pose의 position중에서 z을 선택해주면 그래프가 plot이 됩니다. 이 것을 저장해주시기 바랍니다.

![](http://www.modulabs.co.kr/files/attach/images/214/870/002/97cedadcd696d92a2a085e61b3ad6b27.png){:height="50%" width="50%"}

이제 optical flow sensor가 달린 iris drone도 마찬가지로 기록해줍니다.
```
roscore

cd ~/catkin_ws/src/Firmware
make posix gazebo_iris_opt_flow

roslaunch modudculab_ros ctrl_pos_gazebo.launch

rosrun mavros mavsafety arm

cd ~/bagfiles
rosbag record -O iris_optical_1 /mavros/local_position/pos
rosrun mavros mavsys mode -c OFFBOARD

rqt_bag
```
![](http://www.modulabs.co.kr/files/attach/images/214/870/002/16722947c6fd24ad665dafe6e85e9ffb.png){:height="50%" width="50%"}


# 6. SITL and MAVROS
1. SITL의 구조와 MAVROS

    저번 글에서는 SITL로 일정한 고도에 호버링하는 간단한 임무를 수행해보았습니다. 더 나아가기에 앞서 SITL이 어떻게 작동하는 지에 대해서 살펴 보려고 합니다. 다음은 PX4 documentation 페이지입니다.
    [simulation sitl](http://dev.px4.io/simulation-sitl.html)

    ![](http://www.modulabs.co.kr/files/attach/images/214/961/002/b63b508d0a72c647d14fbf468ab7a7c4.png){:height="50%" width="50%"}
    
    위 그림과 같이 SITL은 MAVLINK라는 통신 프로토콜을 통해서 simulator에 연결합니다. 이것이 기본적인 세팅이고 SITL은 PX4에 들어있는 기능 중에 하나이기 때문에 위 그림에서 SITL을 PX4로 생각하고 simulator를 gazebo라고 생각하면 더 이해하기 쉬울 것 같습니다.
    첫 번째 글에서 PX4와  mavlink와 ros와 mavros에 대한 개념을 살펴봤는데 그 상관관계는 다음과 같습니다.

    ![](http://www.modulabs.co.kr/files/attach/images/214/961/002/debe120ba9f689827a6c4e6d4ca6e90d.png){:height="50%" width="50%"}
    
    간단하게 ROS가 Control center라고 생각하면 됩니다. attitude나 local position, acceleration 등의 명령을 주면 그것이 mavros라는 어떠한 다리를 통해 mavlink의 형태의 매시지로 px4에 전달이 되고 실재로 px4가 모터들을 컨트롤하고 또 반대로 센서데이터들은 mavros를 통해서 ros로 전달됩니다. 이렇게 control이 되는 것입니다. 

    ![](http://www.modulabs.co.kr/files/attach/images/214/961/002/e9b003f79ddcda0ab48f4df31fc7eed4.png){:height="50%" width="50%"}
    
    따라서 현재는 노트북을 Companion Computer로 ROS를 실행시키지만 이후에는 라즈베리파이를 companion computer로 이용하여 pixhawk와 라즈베이를 같이 드론에 탑재해서 컨트롤 할 예정입니다.
    이러한 상관관계속에서 원래라면 Pixhawk에서 PX4가 작동해야하는데 그러한 보드없이 가상보드에서 작동하는 것처럼 시뮬레이션하는 것이 SITL입니다. 즉 ROS에서 mavros를 통해서 명령을 px4에 전달하면 px4는 보드없이 gazebo라는 세상에서 ros의 명령을 실행하는 것입니다. 

    다음 그림은 SITL이 어떻게 통신하는 지를 보여주는 그림입니다. 

    ![](http://www.modulabs.co.kr/files/attach/images/214/961/002/91c7de9282e05ae8d4dc8e74a195638d.png){:height="50%" width="50%"}

    SITL은 PX4자체의 앱중의 하나로서 처음에 PX4를 설치할 때 함께 설치됩니다. 저희는 JMAVSim대신에 Gazebo를 사용했습니다. 이 그림에서는 ROS가 따로 표시되지는 않았습니다. 들어간다면 PX4와 Mavros로 연결되는 형태로 들어갈 것입니다. qGroundConrtol은 뒤에서 Pixhawk 보드 설정할 때 다룰건데 PX4를 픽스호크 보드에 업로드하고 센서 칼리브레이션 등 여러가지 필요한 것들을 간편하게 할 수 있도록 해줍니다. Controller은 픽스호크보드에 연결된 리시버를 통해 들어오는 조종기의 값들을 의미하는 것이고 그래서 "RC channels"라고 적혀있는 것을 알 수 있습니다. 이 통신은 Serial 통신을 이용하고 있습니다.

    현재 ROS는 TCP/IP를 통해서 노드끼리 통신하는데 PX4내부에서는 UDP로 통신하는 것을 볼 수 있습니다. UDP란 User Datagram Protocol을 지칭하는 말인데 TCP/IP는 상대방과 페어링이 되어야 통신이 되는 반면, UDP는 그러지 않아도 되기 때문에 데이터의 유실이 가능한 방식입니다. 자세한 설명은 위키피디아를 참조하시면 될 것 같습니다.(이 부분은 아직 잘 모릅니다) ROS 2에서는 UDP를 사용할 거라고 합니다.

2. Trajectory control

    저번 글에서 SITL에서 position control을 했었는데 trajectory control도 한 번 해보겠습니다.
    /home/src/modudculab_ros/src/pub_setpoints_traj.cpp를 열어주시면
    다음과 같이 trajectory가 적혀있습니다.

    ![](http://www.modulabs.co.kr/files/attach/images/214/870/002/ce2c07e0f53e328b8cac98cf54bbadec.png){:height="50%" width="50%"}
    ```
    roscore
    cd ~/src/Firmware
    make posix gazebo_iris_opt_flow
    roslaunch modudculab_ros ctrl_pos_gazebo.launch
    rosrun mavros mavsafety arm
    ```
    위 과정에서 launch하는 과정만 바꿔주면 됩니다.
    ```
    roslaunch modudculab_ros ctrl_traj_gazebo.launch
    ```
    ![](http://www.modulabs.co.kr/files/attach/images/214/870/002/2c8f8e307d741eb99d40cae26ffd0000.png){:height="50%" width="50%"}

    실행시켜보시면 위 사진과 같이 높이 2미터에서 원운동을 합니다. 단순히 원의 방정식만 명령으로 줬을 뿐인데 드론이 원의 궤도를 따라 움직이는 것을 볼 수 있는데 바로 이것이 ROS의 장점이라고 할 수 있습니다.이 이외에도 여러가지 다른 trajectory를 해 볼수 있겠지만 이 글에서는 간단하게 이 정도만 다루고 넘어가도록 하겠습니다.
    SITL의 구조를 간단하게 살펴보았고 trajectory control까지 진도를 나가봤습니다. 여기서 더 나가기 전에 앞으로도 계속 사용할 ROS와 PX4의 구조에 대해서 알 필요가 있을 것 같습니다.




# 7. ROS and PX4 Sturcture

1. ROS의 구조

    지금까지 Ubuntu를 설치하고 ROS를 설치하고 Gazebo를 설치하고 PX4를 설치하고 그리고 SITL까지 해보았습니다. SITL까지 해봐서 Gazebo안에서 드론을 날리는 것까지는 해보았지만 사실 아직 ROS와 PX4에 대해서 이해가 되거나 그 구조가 그려지지 않을 것입니다. 따라서 HITL로 넘어가기 전에 숨 돌리는 겸 그 구조에 대해서 살펴보려고 합니다.


    ROS의 구조에 대해서 오로카 홈페이지를 바탕으로 작성하였습니다.

    [ROS architecture](http://cafe.naver.com/openrt/2468)

    기본적으로 ROS는 마스터와 노드들, 그리고 그 사이의 통신으로 이루어져 있습니다. 노드란 프로그램을 실행
    하는 기본 단위이고 마스터는 그 노드들 사이의 커뮤니케이션을 도와주는 역할을 합니다. 바로 그 노드들 사이의 커뮤니케이션(메시지통신)이 ROS의 핵심개념입니다. 다음 그림을 참고하면 이해하기 쉽습니다

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/5c2e988d042cdf455fdd55504a1e8e8d.png){:height="50%" width="50%"}

    처음에 마스터가 노드1과 노드2를 연결시켜주고 서로 커뮤니케이션을 하게 하면 이제 마스터는 빠지고 노드끼리 통신하며 작업을 수행하게 됩니다.(개인적으로는 마스터가 주선자같다는 생각이 드네요) 그렇다면 마스터가 노드1과 노드2를 어떻게 연결시켜줄까요?

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/ac6b45153513821f8e2f7e47756981b4.png){:height="50%" width="50%"}

    우리의 기억을 되짚어 보면 항상 처음에 roscore라는 명령어를 실행시켰다는 것이 기억나실텐데 그 명령어가 바로 이 마스터를 구동시키는 것입니다. 위 그림에서 나오는 XMLRPC에 대해서는 다음 위키피디아를 참고해주시기 바랍니다. 통신 프로토콜의 일종인 것 같습니다.

    https://ko.wikipedia.org/wiki/XML-RPC

    마스터는 노드로부터 4가지의 정보를 받습니다. 노드의 이름, 토픽 or 서비스의 이름, 메시지 타입, URI 주소 or 포트입니다. 그림은 다음과 같습니다. 구독자란 토픽에 해당하는 정보를 받아볼 노드를 의미합니다. 마스터와 노드는 위에서 언급했던 XMLRPC로 통신을 합니다. 

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/5c08fb39b910a0ebd902af8e7bf3241b.png){:height="50%" width="50%"}

    발행자 노드로부터도 동일하게 4가지의 정보를 XMLRPC로 받습니다. 

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/85a8795c8c17aca30efd2c978d3befb8.png){:height="50%" width="50%"}

    이렇게 구독자 노드와 발행자 노드가 자신의 정보를 보내는 것은 우리가 앞에서 많이 해봤던 "rosrun"혹은 "roslaunch"라는 명령어로 패키지안의 노드를 실행시켰을 때입니다.

    이제 마스터는 그 발행자 노드의 정보를 구독자 노드에게 전달하면 구독자 노드가 마스터를 통하지 않고 노드1에 접속을 시도하게 됩니다. (소개팅에서 마스터가 번호와 사진 정보를 전달하면 그때부터 서로 연락하는 방식과 비슷하군요) 노드1이 그 접속시도에 응답을 하게 되면(이 과정까지는 XMLRPC로 통신합니다)

    노드 1과 노드 2가 통신을 직접 하게 됩니다. 그림은 다음과 같습니다. 발행자 노드인 노드 1이 TCPROS의 서버가 되고 구독자 노드인 노드 2가 클라이언트가 되어서 노드 2가 구독을 요청했던 토픽을 정해진 메시지의 형태로 받아볼 수 있게 됩니다.

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/b1ac0a29daf5157ee819ae3ad179677b.png){:height="50%" width="50%"}

    앞서 ROS indigo설치 글에서 설치를 확인하는 과정에서 사용했던 turtlesim예제에서 이러한 관계를 살펴보겠습니다. 처음에 roscore명령어를 통해서 마스터를 작동시켜줍니다. 이제 노드를 2개 실행시켜주는데 거북이가 움직이는 것을 화면으로 표시하는 turtlesim_node와 키보드로 조종하는 명령을 받아오는 turtle_teleop_key노드입니다. 키보드를 통해서 turtlesim_node에서 turtle을 조종하려면 turtlesim_node가 turtle_teleop_key노드의 정보를 받아와서 화면으로 거북이가 이동하는 것을 표시해줘야합니다. 

    처음 turtlesim_node를 rosrun으로 실행시켜주면 자신의 노드이름과 받고자 하는 토픽과 그 형식을 마스터에게 전달합니다. 그리고 turtlesim_teleop_key를 rosrun으로 실행시켜주면 똑같이 자신의 이름과 발행하고자 하는 토픽과 그 형태를 마스터에게 전달합니다. 여기서 그 토픽은 /turtle1/cmd_vel입니다. cmd_vel란 키보드로 들어오는 명령을 뜻할 것입니다. 마스터는 두 노드로부터 받아온 정보를 바탕으로 turtlesim_node에게 turtlesim_teleop_key의 정보를 전달하고 turtlesim_node는 turtlesim_teleop_key노드에게 XMLRPC통신으로 접속을 시도하고 상대방이 응답하면 그때부터 turtlesim_node는 키보드로부터 들어오는 command값을 받아서 자신의 노드안의 기능을 거쳐서 화면으로 표현합니다. 

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/fce44ba3d87618f1f36bd7a90624ceb3.png){:height="50%" width="50%"}

    이 내용을 SITL에서 사용했던 ROS의 경우에 적용시켜봅시다. 쉽게 생각하면 turtlesim_node는 PX4가 되는 것이고 turtle_teleop_Key는 ROS가 되는 것입니다. (iris drone의 SITL의 경우에 한정해서) 따라서 저희가 했던 position control이나 trajectory control같은 경우는 해당 노드에서 사용자가 입력한 명령들을 토픽으로서 발행을 하고 PX4에서는 그러한 명령들을 받아서 실재로 모터를 돌리는 역할을 수행합니다. 밑의 그림에서 네모상자는 토픽을 이야기하고 동그라미인 /mavros는 노드로서 mavlink라는 통신 프로토콜을 사용해서 그러한 토픽들을 ROS안의 node들로부터 받아와서(구독) PX4에 건내주는 역할을 합니다. 또한 오른쪽에 있는 네모들에 해당하는 topic들을 PX4를 통해 가져와서 ROS에게 건내주는 역할(발행)도 합니다. 즉, PX4와의 통신에 사용되는 ROS안의 node인 것입니다. 

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/bd78e33fa4e538c308d2ce2d27e60252.png){:height="50%" width="50%"}

2. PX4의 구조    
    지금까지 ROS의 구조와 MAVROS에 대해서 살펴보았습니다. mavros는 하나의 노드로서 ROS와 PX4사이에서 다리 역할을 해서 서로 여러가지 토픽의 정보를 주고 받도록 하였습니다. ROS와 MAVROS는 어떻게 작동하는 지 알았는데 그러면 PX4 자체는 어떤 구조로 이루어져 있을까요? 

    PX4 consists of two main layers: The PX4 flight stack, an autopilot software solution and the PX4 middleware, a general robotics middleware which can support any type of autonomous robot.

    [dev px4](http://dev.px4.io/concept-architecture.html)

    PX4 홈페이지에서는 위에서와 같이 설명합니다. PX4는 두 개의 층으로 이루어져 있는데, PX4 flight stack과 PX4 middleware입니다. PX4 flight stack은 오토파일럿 소프트웨어이며 PX4 middleware는 일반적인 robotics에 사용되는 middeware입니다.    

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/da511c49e6962eef93c0eaad616b61a3.png){:height="50%" width="50%"}
    
    즉, PX4내부에서 PX4 Flight Stack은 드론의 비행에 관련된 부분입니다. 앞에서 PX4가 architecture가 잘 되어있어서 사용한다고 말했었는데 그것이 바로 PX4 Flight Stack의 architecture를 언급한 것입니다. 

    위 그림은 그 PX4 Flight Stack의 구조를 나타내고 있는데 어떻게 보면 박스 하나 하나는 ROS안에서의 node의 개념과 비슷한 것 같습니다. 각 센서 데이터 값으로부터 자신의 위치와 자세를 추정하여 controller에 들어가면 controller에서는 명령을 mixer로 주게 됩니다. 이 mixer라는 것이 PX4 Flight Stack의 특징 중의 하나입니다. http://dev.px4.io/concept-mixing.html

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/b0e44f3c9734a43f05daf27de94f6b0e.png){:height="50%" width="50%"}

    이 Mixer 또한 하나의 노드처럼 생각하며 될 것 같은데 control 명령(힘이나 토크)을 받아서 각 모터에 그 명령을 수행하기 위한 신호를 주는 역할을 합니다. 즉, 쿼드콥터가 고도를 높여야해서 위로 100N의 힘을 주어야할 때 그 100N이 mixer로 들어가면 mixer는 해당 기체가 쿼드콥터인 것을 알고 각 네 개의 모터에 PWM신호를 주게 됩니다. 이러한 mixer는 기체별로 따로 정의가 되어있기때문에 사용자는 단지 정의만 해주면 된다는 편의성이 있습니다.
    그렇다면 ROS로부터 오는 명령은 어디로 들어가는 것일까요?

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/2af4009ebd3fd062ed976dc1f4289929.png){:height="50%" width="50%"}

    ROS로부터 오는 high-level mission은 위 그림의 네모인 commander와 navigator로 들어가게 됩니다. 이전의 기억을 되살려보면 이 commader와 navigator는 mavlink를 통해 그 정보를 받아옵니다. 또한 position과 attitude의 측정정보, mixer의 값이 mavlink를 통해 ROS로 전달되게 됩니다.

    ![](http://www.modulabs.co.kr/files/attach/images/214/319/003/a67c49d1d7fc4429287f292b8224c2e6.png){:height="50%" width="50%"}
    
    PX4의 나머지 한 부분인 PX4 middleware에 대해서는 아는 것이 없기 때문에 다음 글을 참고해주시면 감사하겠습니다.
    PX4 Flight Stack에서 PID계수를 보거나 Tuning을 해주고 싶으면 Q ground control을 설치하고 그 안에서 해당 내용을 확인 및 변경할 수 있습니다.
    [pixhawk parameter guide](https://pixhawk.org/users/parameter_guide)
    제어자체의 알고리즘에 대해서는 나중에 다루도록 하겠습니다.     










# Reference sites
[Pixhawk와 ROS를 이용한 자율주행](http://www.modulabs.co.kr/board_GDCH80/1543)


[로봇 운영체제 강좌](https://cafe.naver.com/openrt/2360)
```

	01. 로봇 운영 체제

	 로봇 운영체제 강좌: 왜 로봇 소프트웨어 플랫폼을 써야 하는가
	 로봇 운영체제 강좌: 로봇 소프트웨어 플랫폼
	 로봇 운영체제 강좌: ROS 소개
	 로봇 운영체제 강좌: ROS 역사
	 로봇 운영체제 강좌: ROS 버전

	02. ROS 설치 (일반PC)

	 로봇 운영체제 강좌: ROS 간단 설치 (Indigo)
	 로봇 운영체제 강좌: ROS 설치 (Indigo)
	 로봇 운영체제 강좌: ROS 설치 (요약)
	 로봇 운영체제 강좌: ROS 설치 (Hydro)
	 로봇 운영체제 강좌: ROS 개발 환경 구축 (환경설정)
	 로봇 운영체제 강좌: ROS 개발 환경 구축 (IDE)
	 로봇 운영체제 강좌: ROS 동작 테스트

	03. ROS 설치 (SBC or VirtualBox)

	 Raspberry Pi + Raspbian Wheezy + ROS Indigo by Noirmist님
	 Raspberry Pi + Raspbian Wheezy + ROS Groovy
	 Raspberry Pi + Raspbian Wheezy + ROS Groovy by GroundZer님
	 Raspberry Pi + Raspbian Wheezy + ROS Groovy by 나무꾼님
	 Raspberry Pi + Raspbian Wheezy + ROS Hydro by GroundZer님
	 BeagleBone Black + Ubuntu 13.04 + ROS Hydro by Baram님
	 Odroid U3 + Ubuntu 14.04 + ROS Indigo by 김성준님
	 Marsboard RK3066 + Ubuntu 14.04 + ROS Indigo by 수야님
	 Marsboard RK3066 + Ubuntu 14.04 + ROS Indigo (이미지 공유) by 수야님
	 VirtualBox + Ubuntu 12.04 + ROS Hydro by 버섯돌이님

	04. ROS 개념 정리

	 로봇 운영체제 강좌: ROS 용어 정리
	 로봇 운영체제 강좌: ROS 개념 정리
	 로봇 운영체제 강좌: ROS 파일 시스템
	 로봇 운영체제 강좌: ROS 빌드 시스템

	05. ROS 명령어

	 로봇 운영체제 강좌: ROS 명령어
	 로봇 운영체제 강좌: ROS 쉘 명령어
	 로봇 운영체제 강좌: ROS 실행 명령어
	 로봇 운영체제 강좌: ROS 정보 명령어 (rosnode)
	 로봇 운영체제 강좌: ROS 정보 명령어 (rostopic)
	 로봇 운영체제 강좌: ROS 정보 명령어 (rosservice)
	 로봇 운영체제 강좌: ROS 정보 명령어 (rosparam)
	 로봇 운영체제 강좌: ROS 정보 명령어 (rosmsg)
	 로봇 운영체제 강좌: ROS 정보 명령어 (rossrv)
	 로봇 운영체제 강좌: ROS 정보 명령어 (rosbag)

	06. ROS 도구

	 로봇 운영체제 강좌: ROS 도구 (RViz)
	 로봇 운영체제 강좌: ROS 도구 (rqt)

	07. ROS 메시지 통신과 실행

	 로봇 운영체제 강좌: 메시지, 토픽, 서비스, 매개변수
	 로봇 운영체제 강좌: 메시지 발행자 노드와 구독자 노드 작성 및 실행
	 로봇 운영체제 강좌: 서비스 서버 노드와 클라이언트 노드 작성 및 실행
	 로봇 운영체제 강좌: 매개변수 사용법
	 로봇 운영체제 강좌: 로스런치 사용법

	08. ROS 패키지 이용 방법

	 로봇 운영체제 강좌: 로봇 패키지
	 로봇 운영체제 강좌: 센서 패키지
	 로봇 운영체제 강좌: 공개 패키지 사용법

	09. 센서 정보 취득

	 로봇 운영체제 강좌: USB 카메라
	 로봇 운영체제 강좌: 레이저 레인지 파인더 (LRF)
	 로봇 운영체제 강좌: Depth Camera
	 로봇 운영체제 강좌: Structure Sensor
	 로봇 운영체제 강좌: IMU, AHRS

	10. 모바일 로봇

	 로봇 운영체제 강좌: ROS 지원 로봇
	 로봇 운영체제 강좌: Kobuki의 하드웨어
	 로봇 운영체제 강좌: Kobuki의 소프트웨어
	 로봇 운영체제 강좌: Kobuki의 ROS 패키지
	 로봇 운영체제 강좌: Kobuki의 개발환경
	 로봇 운영체제 강좌: Kobuki의 원격제어
	 로봇 운영체제 강좌: Kobuki의 토픽(Topic)
	 로봇 운영체제 강좌: Kobuki의 진단툴, 계기판, CUI 및 GUI 기능 테스트
	 로봇 운영체제 강좌: Kobuki의 펌웨어 업그레이드
	 로봇 운영체제 강좌: Kobuki의 자동 도킹
	 로봇 운영체제 강좌: Kobuki의 시뮬레이션 (RViz)
	 로봇 운영체제 강좌: Kobuki의 시뮬레이션 (Gazebo)

	11. SLAM과 내비게이션

	 로봇 운영체제 강좌: SLAM과 내비게이션
	 로봇 운영체제 강좌: SLAM 실습편
	 로봇 운영체제 강좌: SLAM 응용편
	 로봇 운영체제 강좌: SLAM 이론편
	 로봇 운영체제 강좌: 내비게이션 실습편
	 로봇 운영체제 강좌: 내비게이션 응용편(1)
	 로봇 운영체제 강좌: 내비게이션 응용편(2)
	 로봇 운영체제 강좌: 내비게이션 이론편

	12. 매니퓰레이터

	 로봇 운영체제 강좌: 로봇암을 위한 기구학 및 역기구학
	 로봇 운영체제 강좌: 가상 3축 로봇암 시뮬레이션
	 로봇 운영체제 강좌: 가상 6축 로봇암 시뮬레이션
	 로봇 운영체제 강좌: 실제 6축 로봇암 개발
	 로봇 운영체제 강좌: 물체 파지 노드 개발

	13. 가상 로봇

	 로봇 운영체제 강좌: 시뮬레이션의 필요성
	 로봇 운영체제 강좌: 3D 로봇 모델링-1
	 로봇 운영체제 강좌: 3D 로봇 모델링-2

	14. 마이크로 컨트롤러

	 로봇 운영체제 강좌: rosserial for Arduino
	 로봇 운영체제 강좌: 아두이노에서 모터 제어
	 로봇 운영체제 강좌: 아두이노에서 센서 데이터 전송
	 로봇 운영체제 강좌: 아두이노에서 엔코더 데이터 전송

	15. 안드로이드 연계

	 로봇 운영체제 강좌: ROSJAVA
	 로봇 운영체제 강좌: ROS용 안드로이드 어플 개발 환경
	 로봇 운영체제 강좌: 원격 제어 노드 어플
	 로봇 운영체제 강좌: 원격 로봇 모니터링 어플

	16. 영상처리

	 로봇 운영체제 강좌: 얼굴 인식 및 추적
	 로봇 운영체제 강좌: 물체 인식 및 추적

	17. 음성처리

	 로봇 운영체제 강좌: TTS를 이용한 음성 출력
	 로봇 운영체제 강좌: 음성 인식
	 로봇 운영체제 강좌: 로봇 제어용 음성 인식 및 합성

	99. 기타

	 로봇 운영체제 강좌: ROS 2.0
	 윈도우에 ROS설치하기
	 ROS 치트키
	 ROS 공식 위키에 자신의 노드 공개하기

	[출처] 로봇 운영체제 ROS 강좌 목차 (오픈소스 소프트웨어 & 하드웨어: 로봇 기술 공유 카페 (오로카)) 
```
