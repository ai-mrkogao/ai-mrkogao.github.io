---
title: "ROS tutorial korean"
date: 2018-08-01
classes: wide
use_math: true
tags: ROS tutorial Kalman Particle Graph Bundle adjustment
category: ros
---

# 로봇 운영체제 강좌
[로봇 운영체제 강좌](https://cafe.naver.com/openrt/2360)

## 간단설치
2) 소스 리스트 추가
: "sources.list"에 ROS 저장소 주소를 추가하자. 새로운 커맨드 창을 열고 아래와 같이 입력한다. 
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'
```

3) 키 설정
: ROS 저장소로부터 패키지를 다운로드 받기위해 공개키를 추가하자. 아래와 같이 입력한다.
```
wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add -
```

4) 패키지 인덱스 업데이트
: 소스 리스트에 ROS 저장소 주소를 넣었으니 패키지 리스트를 다시 재인덱싱을 한다.
```
sudo apt-get update
```

5) ROS Indigo Igloo 설치
: 이 명령어로 데스크톱용 기본적인 ROS 패키지들을 설치하게 된다. 여기에는 ROS, rqt, rviz, 로봇 관련 라이브러리, 시뮬레이션, 네이게이션 등이 포함되어 있다.
```
sudo apt-get install ros-indigo-desktop-full
```

6) rosdep 초기화
: ros를 사용하기전에 rosdep 를 초기화 해주어야만 한다. rosdep 는 ros의 핵심 컴포넌트 들을 사용하거나 컴파일 할때 의존성 패키지를 쉽게 설치하여 사용자 편의성을 높인 기능이다.

```
sudo rosdep init

rosdep update
```

7) rosinstall 설치
: ros의 다양한 패키지를 인스톨하는 프로그램이다. 빈번하게 사용할 정도로 유용한 툴인만큼 꼭 설치하도록 하자 

```
sudo apt-get install python-rosinstall
```

8) 환경설정 파일 불러오기
: 환경설정이 설정되어 있는 파일을 불러온다. ROS_ROOT, ROS_PACKAGE_PATH 등의 환경 변수들이 정의되어 있다

```
source /opt/ros/indigo/setup.bash
```

9) 작업폴더 생성 및 초기화
: ROS에서는 catkin 이라는 ROS 전용 빌드 시스템을 사용하고 있다. 이를 사용하기위해서는 아래와 같이 catkin 작업 폴더 및 작업 폴더 초기화 설정을 해주어야 한다. (아래의 설정은 ROS를 사용함에 있어서 한 번만 해주면 된다)

```
mkdir -p ~/catkin_ws/src

cd ~/catkin_ws/src

catkin_init_workspace
```

catkin 작업 폴더를 생성하였으면 컴파일을 하자. 현재의 catkin 작업 폴더에는 src폴더 및 그 안의 CMakeLists.txt 이외에 아무런 파일이 없지만 테스트삼아 아래와 같이 "catking_make"명령어를 이용하여 빌드하여 보자

```
cd ~/catkin_ws/

catkin_make
```

문제없이 빌드를 마치게 되면 아래와 같이 "ls" 명령어를 실행해보자. 유저가 직접 생성하였던 "src" 폴더 이외의 없었던 "build" 및 "devel"폴더가 새로 생성되었다. catkin 빌드 시스템의 빌드 관련 파일은 "build" 폴더에, 빌드 후 실행관련 파일은 "devel" 에 저장되게 된다

```
ls

build  devel  src
```
마지막으로, catkin 빌드 시스템과 관련된 환경 파일을 불러오자 

```
source ~/catkin_ws/devel/setup.bash
```

## 환경 설정

위 설치과정에서 사용된 아래 명렁어처럼 환경 설정 파일을 불러오는 것은 새로운 터미널 창을 열때마다 매번 실행해줘야 한다. 이러한 번거로운 작업을 없애기 위하여 새로운 터미널 창을 열때마다 정해진 환경 설정 파일을 읽어오도록 설정해주도록 하자. 또한, ROS 네트워크 설정 및 자주 사용하는 명령어를 단축 명령어로 설정하도록 하자

```
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash
```
우선, gedit 프로그램과 같은 문서편집 프로그램을 사용하여 bashrc 파일을 수정하도록 하자. 아래의 명령어로 bashrc 파일을 불러오자

```
gedit ~/.bashrc
```

```
# Set ROS Indigo
source /opt/ros/indigo/setup.bash
source ~/catkin_ws/devel/setup.bash

# Set ROS Network
export ROS_MASTER_URI=http://xxx.xxx.xxx.xxx:11311
export ROS_HOSTNAME=xxx.xxx.xxx.xxx

# set ROS alias command
alias cw='cd ~/catkin_ws'
alias cs='cd ~/catkin_ws/src'
alias cm='cd ~/catkin_ws && catkin_make'
```

## ROS install command summary

```
sudo apt-get install chrony

sudo ntpdate ntp.ubuntu.com

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt-key adv --keyserver hkp://pool.sks-keyservers.net --recv-key 0xB01FA116

sudo apt-get update

sudo apt-get upgrade

sudo apt-get install ros-indigo-desktop-full

sudo apt-get install ros-indigo-rqt-*

sudo rosdep init

rosdep update

echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc

source ~/.bashrc

sudo apt-get install python-rosinstall

mkdir -p ~/catkin_ws/src

cd ~/catkin_ws/src

catkin_init_workspace

cd ~/catkin_ws/

catkin_make

echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

echo "export ROS_MASTER_URI=http://xxx.xxx.xxx.xxx:11311" >> ~/.bashrc

echo "export ROS_IP=xxx.xxx.xxx.xxx" >> ~/.bashrc

source ~/.bashrc
```

## 통합 개발 환경 (Integrated Development Environment, IDE)

IDE는 코딩, 디버그, 컴파일, 배포 등 프로그램 개발에 관련된 모든 작업을 하나의 프로그램 안에서 처리하는 환경을 제공하는 소프트웨어를 말한다. 아마 많은 개발자들이 자신만이 즐겨쓰는 IDE는 한 두개쯤 있을 것이라고 생각한다. 

ROS 또한 여러 IDE를 사용할 수 있는데, 많이 사용되는 IDE로는 Eclipse, CodeBlocks, Emacs, Vim, NetBeans, QtCreator 등이 있다. 필자의 경우, 처음에는 Eclipse 를 사용하였으나 Eclipse가 최근 버전에 와서는 매우 무겁게 느껴졌으며, ROS의 catkin 빌드 시스템 사용에 있어서 많은 불편을 느꼈다. 그래서, 다른 IDE 를 구축하고자 여러 IDE 를 검토해 봐다. 그 중 가장 적합한 툴로는 QtCreator가 아닐까 싶다. ROS 의 개발/디버깅 툴인 rqt 같은 경우나, RViz 같은 경우에도 QT로 만들어져 있고, 사용자는 QT 플러그인으로 기존 툴과 플러그인을 개발도 가능하다는 점에서 QT의 편집기인 QtCreator 는 매우 유용하다고 볼 수 있다. 또한 QT 를 이용하지 않더라도 범용적인 에디터로써의 기능도 충분히 갖추고 있을뿐 아니라, 프로젝트를 CMakeLists.txt 를 통하여 바로 열수가 있어서 catkin_make 에도 매우 편리하다.

아래의 내용은 QtCreator 로 ROS 개발 환경을 꾸미는 내용을 담고 있다. 반드시 QtCreator 를 이용하여 개발할 필요는 없으므로 QtCreator 를 IDE로 사용하지 않더라도 본 강좌 이외의 후속 강좌들을 사용하는데에는 아무런 문제가 없음을 미리 밝혀둔다.
 
1) QtCreator 설치

```
sudo apt-get install qtcreator
```

2) QtCreator 구동
QtCreator를 아이콘으로 실행시켜도 구동에는 문제 없으나 우리가 bashrc 에 기술해둔 ROS 경로 등의 설정을 QtCreator 에도 적용하기 위해서는 새로운 터미널 창을 열어 아래와 같이 실행해줘야 한다

```
qtcreator
```

3) ROS 패키지를 프로젝트로 불러오기
앞서 말한바와 같이 QtCreator는 CMakeLists.txt 를 사용한다. 그렇기에 ROS 패키지는 단순히 아래와 화면의 OpenProject 버튼을 클릭하여 해당 ROS 패키지의 CMakeLists.txt 를 선택하면 손쉽게 프로젝트로 불러올 수 있다.


4) ***ROS 패키지를 프로젝트로 불러오기***  
컴파일의 경우에는 단순히 Ctrl + B 로 컴파일 하면 catkin_make 가 실행된다. 단, 빌드 관련은 해당 패키지와 같은 위치의 폴더에 새로운 폴더로 생성된다. 예를 들어 tms_rp_action 이라는 패키지를 컴파일하면 "build-tms_rp_action-Desktop-Default" 이라는 폴더에 빌드 관련이 모두 놓이게 된다. 즉,  원래는 ~/catkin_ws/build 와 ~/catkin_ws/devel 에 보관되어야 할 파일들이 따로 컴파일 되어 새로운 장소에 놓이게 되므로 실행을 위해서는 나중에 다시 한번 catkin_make 를 해줘야 한다. 이는 매번 해줄 필요는 없고 개발도중에는 QtCreator 에서 개발, 디버깅을 한 후에 완료되어 실행할때만 따로 catkin_make 를 해준다고 보면된다


## Ubuntu 12.04 LTS환경에 ROS 설치하기

[Ubuntu 12.04 LTS환경에 ROS 설치하기](https://cafe.naver.com/openrt/4127)


## 용어 정리

- 로스(ROS)

  로스는 로봇 응용 프로그램을 개발을 위한 운영체제와 같은 로봇 플랫폼이다. 로스는 로봇 응용프로그램을 개발할때 필요한 하드웨어 추상화, 하위 디바이스 제어, 일반적으로 사용되는 기능의 구현, 프로세스간의 메시지 파싱, 패키지 관리, 개발환경에 필요한 라이브러리와 다양한 개발 및 디버깅 도구를 제공한다.

- 마스터(master)

  마스터는 노드와 노드사이의 연결 및 메시지 통신을 위한 네임 서버와 같은 역할을 한다. 로스코어(roscore)가 실행 명령어 이며, 마스터를 실행하게되면 각 노드들의 이름을 등록하고 필요에 따라 정보를 받을 수 있다. 마스터가 없이는 노드간의 접속, 토픽과 서비스와 같은 메시지 통신을 할 수 없다. 

  마스터는 마스터에 접속하는 슬레이브들과의 접속 상태를 유지하지 않는 HTTP 기반의 프로토콜인 XMLRPC를 이용하여 슬레이브들과 통신하게 된다. 즉, 슬레이브인 노드들이 필요할때만 접속하여 자신의 정보를 등록하거나 다른 노드의 정보를 요청하여 수신받을 수 있다. 평상시에는 서로간의 접속 상태를 체크하지 않는다. 이러한 기능으로 매우 크고, 복잡한 환경에서도 적용가능하다. 또한, XMLRPC는 매우 가볍고, 다양한 프로그래밍 언어를 지원하고 있기때문에 다기종 하드웨어, 다언어를 지원하는 로스에 매우 적합하다.

  로스를 구동하게 되면 사용자가 정해놓은 ROS_MASTER_URI 변수에 기재되어 있는 URI 주소 및 포트를 갖는다. 사용자가 설정해놓지 않은 경우에는 URI 주소로 현재의 로컬 IP 를 사용하고, 11311 포트를 이용하게 된다.

- 노드(node)

  로스에서 최소 단위의 실행 프로세서를 가르키는 용어이다. 하나의 실행 가능한 프로그램으로 생각하면 된다. 

  로스에서는 하나의 목적에 하나의 노드를 작성하길 권하고 있으며 재사용이 쉽도록 구성하여 만들도록 권고하고 있다. 예를들어, 모바일 로봇의 경우, 로봇을 구동하기 위하여 각 프로그램을 세분화 시킨다. 예를들어, 센서 드라이브, 센서 데이타를 이용한 변환, 장애물 판단, 모터 구동, 엔코더 입력, 네이게이션 등 세부화된 작은 노드들을 이용한다.

  노드는 생성과 함께 마스터에 노드이름, 발행자이름, 구독자이름, 토픽이름, 서비스이름, 메시지형태, URI 주소 및 포트를 등록한다. 이 정보들을 기반으로 각 노드는 노드끼리 토픽 및 서비스를 이용하여 메시지를 주고 받을 수 있다.

  노드는 마스터와 통신할 때, XMLRPC를 이용하며, 노드간의 통신에서는 XMLRPC 및 TCP/IP 통신 계열의 TCPROS를 이용하고 있다. 노드간의 접속 요청 및 응답은  XMLRPC를 사용하며, 메시지 통신은 마스터와는 관계없이 노드와 노드간의 직접적인 통신으로 TCPROS를 이용하고 있다. URI 주소 및 포트는 현재 노드가 실행중인 컴퓨터에 저장된 ROS_HOSTNAME 라는 환경 변수값을 URI 주소로 사용하며, 포트는 임의적 고유의 값으로 설정되게 된다.

- 패키지(package)

  로스를 구성하는 기본 단위로써 실행 가능한 노드를 포함하고 있다. 로스는 패키지를 단위로 각각의 응용 프로그램들이 개발된다. 패키지는 최소한 하나 이상의 노드를 포함하고 있다. ROS Hydro의 경우 공식적으로 약 700개 의 패키지를 제공하고 있으며, 유저들이 개발하여 공개된 패키지가 대략 3300개에 달하고 있다.

- 메타패키지(metapackage)

  공통된 목적을 가지는 패키지들을 모아둔 패키지들의 집합을 말한다. 복수의 패키지를 포함하고 있다.


- 메시지(message,msg)

  노드는 메시지를 통해 노드간의 데이터를 주고받게 된다. 메시지는 integer, floating point, boolean 와 같은 변수형태이다. 또한, 메시지안에 메시지를 품고 있는 간단한 데이터 구조 및 메시지들의 배열과 같은 구조도 사용할 수 있다.

  메세지를 이용한 통신방법으로는 TCPROS, UDPROS 방식등이 있으며, 단방향 메시지 송/수신 방식의 토픽과 양방향 메시지 요청/응답 방식의 서비스를 이용하고 있다.

- 토픽(topic)

  토픽은 "이야깃거리"이다. 발행자 노드가 하나의 이야깃거리에서 대해서 토픽이라는 이름으로 마스터에 등록한 후, 이야깃거리에 대한 이야기를 메시지 형태로 발행한다. 이 이야깃거리를 수신 받기를 원하는 구독자 노드는 마스터에 등록된 토픽의 이름에 해당되는 발행자 노드의 정보를 받는다. 이 정보를 기반으로 구독자 노드는 발행자 노드와 직접적으로 연결하여 메시지를 송/수신 또는 요청/응답 받게 된다. 

- 발행(publish) 및 발행자(Publisher) 

  발행은 토픽의 내용에 해당되는 메시지 형태의 데이터를 송신하는 것을 말한다.

  발행자 노드는 발행을 수행하기 위하여 토픽을 포함한 자신의 정보들을 마스터에 등록하고, 구독을 원하는 구독자 노드에게 메시지를 보내게 된다. 발행자는 이를 실행하는 개체로써 노드에서 선언하게 된다. 발행자는 하나의 노드에서 복수로 선언이 가능하다.

- 구독(subscribe) 및 구독자(Subscriber)

  구독은 토픽의 내용에 해당되는 메시지 형태의 데이터를 수신하는 것을 말한다. 

  구독자 노드는 구독을 수행하기 위하여 토픽을 포함한 자신의 정보들을 마스터에 등록하고, 구독하고자 하는 토픽을 발행하는 발행자 노드의 정보를 마스터로부터 받는다. 이 정보를 기반으로 구독자 노드는 발행자 노드와 직접적으로 접속하여 발행자 노드로부터 메시지를 받게 된다. 구독자는 이를 실행하는 개체로써 노드에서 선언하게 된다. 구독자는 하나의 노드에서 복수로 선언이 가능하다.

- 서비스(service)

  발행과 구독 개념의 토픽 통신 방식은 ***비동기 방식***이라 필요에 따라서 주어진 데이터를 전송하고 받기에 매우 훌륭한 방법이다. 또한, 한번의 접속으로 지속적인 메시지를 송/수신하기 때문에 지속적으로 메시지를 발송해야하는 센서 데이터에 적합하여 많이 사용되고 있다.

  하지만, 경우에 따라서는 요청과 응답이 함께 사용되는 동기 방식의 메시지 교환 방식도 필요하다. 이에따라, 로스에서는 ***서비스라는 이름으로 메시지 동기 방식을 제공하고 있다.***

  서비스는 요청이 있을 경우에 응답을 하는 서비스 서버와 요청을 하고 응답을 받는 서비스 클라이언트로 나뉘어 있다. 서비스는 토픽과는 달리 1회성 메시지 통신이다. 서비스의 요청과 응답이 완료되면 연결된 두 노드의 접속은 끊기게 된다. 

- 서비스 서버 (service server)

  서비스 서버는 요청을 입력으로 받고, 응답을 출력으로 하는 서비스 메시지 통신의 서버 역할을 말한다. 요청과 응답은 모두 메시지로 되어 있으며, 서비스 요청에 의하여 주어진 서비스를 수행 후에 그 결과를 서비스 클라이언트에게 전달한다. 서비스 메시지 방식은 동기식이기에 정해진 명령을 지시 및 수행하는 노드에 사용하는 경우가 많다.

- 서비스 클라이언트 (service client)

  서비스 클라이언트는 요청을 출력으로 하고, 응답을 입력으로 받는 서비스 메시지 통신의 클라이언트 역할을 말한다. 요청과 응답은 모두 메시지로 되어 있으며, 서비스 요청을 서비스 서버에 전달하고 그 결과 값을 서비스 서버로 부터 받는다. 서비스 메시지 방식은 동기식이기에 정해진 명령을 지시 및 수행하는 노드에 사용하는 경우가 많다.

- 캐킨(catkin)

  로스의 빌드 시스템을 말한다. 로스의 빌드 시스템은 기본적으로 CMake(Cross Platform Make) 를 이용하고 있어서 패키지 폴더에 CMakeLists.txt 라는 파일에 빌드 환경을 기술하고 있다. 로스에서는 CMake 를 로스에 맞도록 수정하여 로스에 특화된 캐킨 빌드 시스템을 만들었다. 캐킨 빌드 시스템은 로스와 관련된 빌드, 패키지 관리, 패키지간의 의존관계 등을 편리하게 사용할 수 있도록 하고 있다. 

- 로스코어(roscore)

  로스 마스터를 구동하는 명령어이다. 같은 네트웍이라면 다른 컴퓨터에서 실행하여도 된다. 로스를 구동하게 되면 사용자가 정해놓은 ROS_MASTER_URI 변수에 기재되어 있는 URI 주소 및 포트를 갖는다. 사용자가 설정해놓지 않은 경우에는 URI 주소로 현재의 로컬 IP 를 사용하고, 11311 포트를 이용하게 된다

- 로스런(rosrun)

  로스의 기본적인 실행 명령어이다. 패키지에서 하나의 노드를 실행하는데 사용된다. 노드가 사용하는 URI 주소 및 포트는 현재 노드가 실행중인 컴퓨터에 저장된 ROS_HOSTNAME 라는 환경 변수값을 URI 주소로 사용하며, 포트는 임의적 고유의 값으로 설정되게 된다.

- 로스런치(roslaunch)

  로스런(rosrun)이 하나의 노드를 실행하는 명령어라면 로스런치(roslaunch)는 복 수개의 노드를 실행하는 개념이다. 이 명령어를 통해 정해진 단일 혹은 복수의 노드를 실행시킬 수 있다. 

  그 이외의 기능으로 실행시에 패키지의 매개변수를 변경, 노드 명의 변경, 노드 네임 스페이스 설정, ROS_ROOT 및 ROS_PACKAGE_PATH 설정, 이름 변경, 환경 변수 변경 등의 실행시 변경할 수 있는 많은 옵션들을 갖춘 노드 실행에 특화된 로스 명령어이다. 

  로스런치는 launch 라는 로스런치파일을 사용하여 실행 노드에 대한 설정을 해주는데 이는 XML 기반으로 되어 있으며, 태그별 옵션을 제공하고 있다. 실행 명령어로는 "roslaunch 패키지명 로스런치파일" 이다.

- 배그(bag)

  로스에서 주고받는 메시지의 데이터를 저장할 수 있는 있는데 이를 배그라고 한다. ROS에서는 이 배그를 이용하여 메시지를 저장하고 필요로 할 때 이를 재생하여 이전 상황을 그대로 재현할 수있는 기능을 갖추고 있다. 예를들어, 센서를 이용한 로봇 실험을 실행할 때, 센서 값을 배그를 이용하여 메시지 형태로 저장한다. 이 저장된 메시지는 같은 실험을 수행하지 않아도 저장해둔 배그 파일을 재생하는 것으로 그 당시의 센서값을 반복 사용가능하다. 특히, 기록, 재생의 기능을 활용하여 반복되는 프로그램 수정이 많은 알고리즘 개발에 매우 유용하다.  

- 그래프(graph)

  위에서 설명한 노드, 토픽, 발행자, 구독자 관계를 그래프를 통해 나타나게 하는 것이다. 현재 실행중인 메시지 통신을 그래프화 시킨 것으로 1회성인 서비스에 대한 그래프는 작성할 수 없다. 실행은 rqt_graph 패키지의 rqt_graph 노드를 실행하면 된다. "rqt_graph" 또는 "rosrun rqt_graph rqt_graph" 명령어를 이용하면 된다.

- CMakeLists.txt

  로스의 빌드 시스템인 캐킨은 기본적으로 CMake를 이용하고 있어서 패키지 폴더에 CMakeLists.txt 라는 파일에 빌드 환경을 기술하고 있다.

- package.xml

  패키지의 정보를 담은 XML 파일로써 패키지의 이름, 저작자, 라이선스, 의존성 패키지 등을 기술하고 있다.

## ROS의 파일 시스템

로스의 파일 시스템에 대해서 설명하고자 한다. 로스는 크게 "로스 설치 폴더"와 "사용자 작업 폴더"로 구분된다. 

로스 설치 폴더는 로스의 데스크톱 버전을 설치하게 되면 /opt 폴더에 /ros 라는 이름으로 폴더가 생성되고 그 안에 roscore를 포함한 핵심 유틸리티 및 rqt, rviz, 로봇 관련 라이브러리, 시뮬레이션, 네이게이션 등이 설치된다. 사용자는 이 부분의 파일들을 건들일은 거의 없다. 

사용자 작업 폴더는 사용자가 원하는 곳에 폴더를 생성가능한데 필자는 리눅스 사용자 폴더인 "~/catkin_ws/" (~/ 은 리눅스에서 /home/사용자명에 해당되는 폴더를 의미한다.) 에 설치할 것은 추천한다. 

## 로스 설치 폴더

1) 설치 경로
"/opt/ros/[버전명]" 폴더에 설치된다. 예를들어, Indigo 버전을 설치하였을 경우 "/opt/ros/indigo"가 로스 설치 경로이다.

2) 파일 구성
다음의 그림과 같이 "/opt/ros/indigo" 의 폴더아래에 "bin", "etc", "include", "lib", "share" 폴더 및 환경설정 파일들로 구성되어 있다. 

![](https://cafeptthumb-phinf.pstatic.net/20140825_180/passionvirus_1408923102036IUIa3_PNG/indigofolder.png){:height="50%" width="50%"}

3) 세부 내용
이 폴더에는 로스 설치시에 선택한 패키지를 포함한 로스 구동 제반 프로그램을 포함하고 있다. 세부 내용은 아래와 같다.

    /bin 실행가능한 바이너리 파일  
    /etc ros 및 catkin 관련 설정파일  
    /include 헤더파일  
    /lib 라이브러리 파일  
    /share ros 패키지  
    env.* 환경설정 파일  
    setup.* 환경설정 파일  

## 사용자 작업 폴더

1) 사용자 작업 폴더 경로
사용자 작업 폴더는 사용자가 원하는 곳에 폴더를 생성가능하나 강좌의 원활한 진행을 위하여 리눅스 사용자 폴더인 "~/catkin_ws/" (~/ 은 리눅스에서 /home/사용자명에 해당되는 폴더를 의미한다.) 에 설치할 것은 추천한다. 즉, "/home/사용자명/catkin폴더명"으로 설치하면된다. 예를들어 사용자명이 oroca라는 아이디이고 catkin폴더명은 catkin_ws라고 설정하였다면 "/home/oroca/catkin_ws/" 폴더가 된다. 자세한 내용은"로봇 운영체제 강좌 : ROS Indigo 설치" 를 참조하도록 하자. 

2) 파일 구성
다음의 그림과 같이 "/home/사용자명/" 폴더의 아래에 "catkin_ws" 라는 폴더가 있고, "build", "devel", "src" 폴더로 구성되어 있다. 

![](https://cafeptthumb-phinf.pstatic.net/20130921_169/passionvirus_1379732693539kfMOh_PNG/catkin_ws.png){:height="50%" width="50%"}

3) 세부 내용
이 폴더에는 사용자 작업 폴더로 사용자가 작성한 패키지 및 공개된 다른 개발자의 패키지를 저장하고 빌드하는 공간이다. 사용자는 이 폴더를 작업 폴더로 이용하며 로스 관련된 내부분의 작업을 이 폴더안에서 하게된다. 세부 내용은 아래와 같다.

    /build catkin 빌드 시스템의 빌드 환경 파일  
    /devel catkin 빌드 시스템에 의해 빌드된 msg, srv 헤더파일 및 사용자 패키지 라이브러리 및  실행파일  
    /src 사용자 패키지  

4) 사용자 패키지
"/catkin_ws/src" 폴더에는 사용자가 사용하는 공간이다. 이 폴더에 사용자가 개발한 로스 패키지 및 다른 개발자가 개발한 패키지를 저장하고 빌드하여 실행 파일을 생성해 낼 수 있다. 로스 빌드 시스템은 다음 강좌를 통해 더 자세히 알아 보도록 하고 이번 강좌에는 "이런 폴더와 파일로 구성되어 있구나" 라고 알아두고 넘어가기로 하자. 아래의 예제는 필자가 작성한 "oroca_ros_tutorial" 이라는 패키지를 작성한 후의 상태이다. 자세한 내용은 "ROS 강좌  ROS 빌드 시스템" 및 "메시지 송신 노드와 수신 노드 작성 및 실행"에서 다루어 보기로 하자

![](https://cafeptthumb-phinf.pstatic.net/20130921_275/passionvirus_13797326933283w65c_JPEG/src.png){:height="50%" width="50%"}

## 로스 빌드 시스템

로스의 빌드 시스템은 기본적으로 CMake(Cross Platform Make)[2] 를 이용하고 있고, 패키지 폴더에 CMakeLists.txt 라는 파일에 빌드 환경을 기술하고 있다. 로스에서는 CMake를 로스에 맞도록 수정하여 로스에 특화된 캐킨 빌드 시스템을 만들었다. 

로스에서 CMake를 이용하고 있는 이유는 멀티플랫폼에서 로스 패키지를 빌드할 수 있도록 위함이다. Make[3]가 유닉스계열만 지원하는 것과 달리, CMake는 유닉스 계열인 리눅스, BSD, OS X 뿐만 아니라 윈도우 계열도 지원하기 때문이다. 또한, 마이크로소프트 비주얼 스튜디오도 지원하고 QT개발에도 쉽게 적용될 수 있다. 

더욱이, 캐킨 빌드 시스템은 로스와 관련된 빌드, 패키지 관리, 패키지간의 의존관계 등을 편리하게 사용할 수 있도록 하고 있다. 

## 패키지 생성

로스 패키지를 생성하기 위해서는 다음과 같은 명령어를 이용한다. "catkin_create_pkg" 는 사용자가 패키지를 작성할때 캐킨 빌드 시스템에 꼭 필요한 CMakeLists.txt 와 package.xml 를 포함한 패키지 폴더를 생성한다. 실제로 간단한 패키지를 작성해 보자.

```
catkin_create_pkg <패키지이름> [의존하는패키지1] [의존하는패키지2] [의존하는패키지3]
```

1) 작업 폴더로 이동 

```
$ cd ~/catkin_ws/src
```

2) 패키지 생성

"my_first_ros_pkg" 라는 이름의 패키지를 생성할 것이다. 

로스에서는 패키지 이름에는 모두 소문자를 사용하며, 스페이스바와 같은 공백이 있으면 안된다. 그리고 일반적으로 하이픈 - 대신에 밑줄 _ 을 사용하여 각 단어를 이어붙이는 것을 관례로 하고 있다. 
그리고 이번에는 의존하는 패키지로 "std_msgs"와 "roscpp"를 옵션으로 달아주었다. 로스의 표준 메시지 패키지인 std_msgs 와 로스에서 c/c++을 사용하기 위하여 클라이언트라이브러인 roscpp를 사용하겠다는 것으로 패키지 생성에 앞어서 미리 설치해야한다는 의미이다. 이러한 의존하는 패키지의 설정은 패키지 생성할 때 지정할 수도 있지만, 생성 후 package.xml 에서 직접 입력하여도 된다.

```
$ catkin_create_pkg my_first_ros_pkg std_msgs roscpp
```

위와 같이 패키지를 생성하였으면 " /catkin_ws/src "에 "my_first_ros_pkg" 라는 패키지 폴더 및 로스 패키지가 갖추어야할 기본 내부 폴더 및 CMakeLists.txt 와 package.xml가 생성된다. 다음과 같이 명령어로 ls 를 입력하여 내용을 보던가 윈도우의 탐색기와 같은 역할을 하는 GUI기반의 Nautilus를 이용하여 패키지 내부를 살펴보도록 하자

![](../../pictures/ros/packagefolder.png){:height="50%" width="50%"}

## 패키지 설정 파일 (package.xml) 수정

로스의 필수 설정 파일 중 하나인 package.xml 은 패키지의 정보를 담은 XML 파일로써 패키지의 이름, 저작자, 라이선스, 의존성 패키지 등을 기술하고 있다. 처음에 아무런 수정을 가하지 않은 원본 파일은 다음과 같다.

```xml
    <?xml version="1.0"?>
    <package>
      <name>my_first_ros_pkg</name>
      <version>0.0.0</version>
      <description>The my_first_ros_pkg package</description>
     
      <maintainer email="rt@todo.todo">rt</maintainer>
          
      <license>TODO</license>
     
   
      <buildtool_depend>catkin</buildtool_depend>
      <build_depend>std_msgs</build_depend>
      <build_depend>roscpp</build_depend>
      <run_depend>std_msgs</run_depend>
      <run_depend>roscpp</run_depend>
     
     
      <!-- The export tag contains other, unspecified, tags -->
      <export>
        <!-- You can specify that this package is a metapackage here: -->
        <!-- <metapackage/> -->
     
        <!-- Other tools can request additional information be placed here -->
     
      </export>
    </package>

<buildtool_depend> : 빌드 시스템의 의존성을 기술한다. 지금은 캐킨 빌드 시스템을 이용하고 있기 때문에 catkin 를 입력하면 된다.

<build_depend> : 패키지를 빌드할 때 의존하는 패키지명을 적어준다.

<run_depend> : 패키지를 실행할 때 의존하는 패키지명을 적어준다.

<test_depend> : 패키지를 테스트할때 의존하는 패키지명을 적어준다. 테스트이외에는 사용하지 않는다.

<export> : 로스에서 명시하지 않은 태그명을 사용할때 쓰인다. 일반적인 경우 쓸일이 없다.

<metapackage/> : export 태그 안에서 사용하는 공식적인 태그로 현재의 패키지가 메타패키지의 경우 이를 선언한다.
```

## 빌드 설정 파일 (CMakeLists.txt) 수정

로스의 빌드 시스템인 캐킨은 기본적으로 CMake를 이용하고 있어서 패키지 폴더에 CMakeLists.txt 라는 파일에 빌드 환경을 기술하고 있다. 이는 실행 파일 생성, 의존성 패키지 우선 빌드, 링크 생성 등을 설정하게 되어 있다. 처음에 아무런 수정을 가하지 않은 원본 파일은 다음과 같다.

```c
    cmake_minimum_required(VERSION 2.8.3)
    project(my_first_ros_pkg)
     
    ## Find catkin macros and libraries
    ## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
    ## is used, also find other catkin packages
    find_package(catkin REQUIRED COMPONENTS
      roscpp
      std_msgs
    )
     
    ## System dependencies are found with CMake's conventions
    # find_package(Boost REQUIRED COMPONENTS system)
     
     
    ## Uncomment this if the package has a setup.py. This macro ensures
    ## modules and global scripts declared therein get installed
    ## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
    # catkin_python_setup()
     
    #######################################
    ## Declare ROS messages and services ##
    #######################################
     
    ## Generate messages in the 'msg' folder
    # add_message_files(
    #   FILES
    #   Message1.msg
    #   Message2.msg
    # )
     
    ## Generate services in the 'srv' folder
    # add_service_files(
    #   FILES
    #   Service1.srv
    #   Service2.srv
    # )
     
    ## Generate added messages and services with any dependencies listed here
    # generate_messages(
    #   DEPENDENCIES
    #   std_msgs
    # )
     
    ###################################
    ## catkin specific configuration ##
    ###################################
    ## The catkin_package macro generates cmake config files for your package
    ## Declare things to be passed to dependent projects
    ## INCLUDE_DIRS: uncomment this if you package contains header files
    ## LIBRARIES: libraries you create in this project that dependent projects also need
    ## CATKIN_DEPENDS: catkin_packages dependent projects also need
    ## DEPENDS: system dependencies of this project that dependent projects also need
    catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES my_first_ros_pkg
    #  CATKIN_DEPENDS roscpp std_msgs
    #  DEPENDS system_lib
    )
     
    ###########
    ## Build ##
    ###########
     
    ## Specify additional locations of header files
    ## Your package locations should be listed before other locations
    # include_directories(include)
    include_directories(
      ${catkin_INCLUDE_DIRS}
    )
     
    ## Declare a cpp library
    # add_library(my_first_ros_pkg
    #   src/${PROJECT_NAME}/my_first_ros_pkg.cpp
    # )
     
    ## Declare a cpp executable
    # add_executable(my_first_ros_pkg_node src/my_first_ros_pkg_node.cpp)
     
    ## Add cmake target dependencies of the executable/library
    ## as an example, message headers may need to be generated before nodes
    # add_dependencies(my_first_ros_pkg_node my_first_ros_pkg_generate_messages_cpp)
     
    ## Specify libraries to link a library or executable target against
    # target_link_libraries(my_first_ros_pkg_node
    #   ${catkin_LIBRARIES}
    # )
     
    #############
    ## Install ##
    #############
     
    # all install targets should use catkin DESTINATION variables
    # See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html
     
    ## Mark executable scripts (Python etc.) for installation
    ## in contrast to setup.py, you can choose the destination
    # install(PROGRAMS
    #   scripts/my_python_script
    #   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    # )
     
    ## Mark executables and/or libraries for installation
    # install(TARGETS my_first_ros_pkg my_first_ros_pkg_node
    #   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    #   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    #   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    # )
     
    ## Mark cpp header files for installation
    # install(DIRECTORY include/${PROJECT_NAME}/
    #   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    #   FILES_MATCHING PATTERN "*.h"
    #   PATTERN ".svn" EXCLUDE
    # )
     
    ## Mark other files for installation (e.g. launch and bag files, etc.)
    # install(FILES
    #   # myfile1
    #   # myfile2
    #   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
    # )
     
    #############
    ## Testing ##
    #############
     
    ## Add gtest based cpp test target and link libraries
    # catkin_add_gtest(${PROJECT_NAME}-test test/test_my_first_ros_pkg.cpp)
    # if(TARGET ${PROJECT_NAME}-test)
    #   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
    # endif()
     
    ## Add folders to be run by python nosetests
    # catkin_add_nosetests(test)



```    

```python
// 운영체제에 설치되어 있는 cmake의 최소한의 버전이다. 현재에는 2.8.3 버전으로 명시되어 있다. 이 보다 낮은 cmake를 사용하는 경우에는 버전 업데이트를 해줘야 한다.
cmake_minimum_required(VERSION 2.8.3)

// 패키지의 이름이다. package.xml 에서 입력한 패키지 이름을 그대로 사용하자.
project(my_first_ros_pkg)

// 캐킨 빌드를 할 때 요구되는 구성요소 패키지이다. 현재 의존성 패키지로 roscpp 및 std_msgs가 추가되어 있다. 여기에 입력된 패키지가 없는 경우 캐킨 빌드할 때 사용자에게 에러가 표시된다. 즉, 사용자가 만든 패키지가 의존하는 패키지를 먼저 설치하게 만드는 옵션이다. 
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

// 로스 이외의 패키지를 사용하는 경우에 사용되는 방법이다. 예를들어 다음의 경우, Boost를 사용할때 system 이라는 패키지가 설되어 있어야 한다. 기능은 위에서 설명한 의존하는 패키지를 먼저 설치하게 만드는 옵션이다.
find_package(Boost REQUIRED COMPONENTS system)

// 파이썬을 사용할 때 설정하는 옵션이다. 파이썬은 cmake를 사용할 필요없는 스크립트 언어이지만 패키지의 호환성을 위해 아래와 같이 독자적인 설정을 하게 되어 있다.
catkin_python_setup()

// 사용하는 메시지 파일을 추가하는 옵션이다. FILES를 사용하면 패키지 폴더의 "msg" 안의 .msg 파일들을 참조하게 된다. 다음의 예제에서는 Message1.msg 및 Message2.msg 의 메시지 파일을 이용하겠다는 옵션이다.
add_message_files(
  FILES
  Message1.msg
  Message2.msg
)

// 사용하는 서비스 파일을 추가하는 옵션이다. FILES를 사용하면 패키지 폴더의 "srv" 안의 .srv 파일들을 참조하게 된다. 다음의 예제에서는 Service1.srv 및 Service2.srv 의 서비스 파일을 이용하겠다는 옵션이다.
add_service_files(
  FILES
  Service1.srv
  Service2.srv
)

// 의존하는 메시지를 사용하겠다는 옵션이다. 다음의 예제에서는 DEPENDENCIES 옵션에 의하여 std_msgs 라는 메시지 패키지를 사용하겠다는 설정이다.
generate_messages(
  DEPENDENCIES
  std_msgs
)

// 캐킨 빌드 옵션이다.
// "INCLUDE_DIRS"는 뒤에 설정한 패키지 내부 폴더인 "include"의 헤더파일을 사용하겠다는 설정이다.
// "LIBRARIES"는 뒤에 설정한 패키지의 라이브러리를 사용하겠다는 설정이다.
//  "CATKIN_DEPENDS" 캐킨 빌드할 때 의존하는 패키지들이다. 현재  roscpp 및 std_msgs가 의존하고 있다는 설정이다.
// "DEPENDS" 시스템 의존 패키지를 기술하는 설정이다.
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES my_first_ros_pkg
 CATKIN_DEPENDS roscpp std_msgs
 DEPENDS system_lib
)

// 인클루드 폴더를 지정할 수 있는 옵션이다. 현재 ${catkin_INCLUDE_DIRS} 라고 설정되어 있는데 이는 각 패키지안의 "include" 폴더를 의미하고 이안의 헤더파일을 이용하겠다는 설정이다.
include_directories(
  ${catkin_INCLUDE_DIRS}
)

// cpp 라이브러리를 선언한다. src/${PROJECT_NAME}/my_first_ros_pkg.cpp 파일을 참조하여 my_first_ros_pkg 라는 라이브러리를 생성하게 된다.
add_library(my_first_ros_pkg
  src/${PROJECT_NAME}/my_first_ros_pkg.cpp
)

// cpp 실행 파일을 선언한다. src/my_first_ros_pkg_node.cpp 파일을 참조하여 my_first_ros_pkg_node 라는 실행파일을 생성한다.
add_executable(my_first_ros_pkg_node src/my_first_ros_pkg_node.cpp)

// 패키지를 빌드하기 앞서서 생성해야할 메시지 헤더파일이 있을 경우 빌드전에 우선적으로 메시지를 생성하라는 설정이다. 현재 my_first_ros_pkg_generate_messages_cpp 를 우선적으로 빌드하고 my_first_ros_pkg_node 를 빌드하게 하는 설정이다.
add_dependencies(my_first_ros_pkg_node my_first_ros_pkg_generate_messages_cpp)

// my_first_ros_pkg_node 를 생성하기 앞서서 링크해야하는 라이브러리 및 실행파일을 링크해주는 옵션이다.
target_link_libraries(my_first_ros_pkg_node
  ${catkin_LIBRARIES}
)


```

```python
    cmake_minimum_required(VERSION 2.8.3)
    project(my_first_ros_pkg)
     
    find_package(catkin REQUIRED COMPONENTS
      roscpp
      std_msgs
    )
     
    catkin_package(
      INCLUDE_DIRS include
      CATKIN_DEPENDS roscpp std_msgs
      DEPENDS system_lib
    )
     
    include_directories(
      ${catkin_INCLUDE_DIRS}
    )
     
    add_executable(hello_world_node src/hello_world_node.cpp)
    add_dependencies(hello_world_node my_first_ros_pkg_generate_messages_cpp)
    target_link_libraries(hello_world_node ${catkin_LIBRARIES})
     
```

## 소스코드 작성

위에서 필자가 작성한 CMakelists.txt 파일을 참고하길 바란다. 실행파일 생성 부분에서 다음과 같이 설정해 놓았다. 즉, 패키지의 "src" 폴더에 있는 "hello_world_node.cpp" 소스코드를 참고하여 "hello_world_node" 라는 실행파일을 생성하라는 설정이다. 여기서, "hello_world_node.cpp" 소스코드가 없기에 간단한 예제로 하나 작성해 보자. 다음의 예제에서는 nano라는 에디터를 사용하였으나 vi, gedit, qtcreator 등 자신이 원하는 편집기를 이용하면 된다.

"add_executable(hello_world_node src/hello_world_node.cpp)"
```
$ cd src     (여기서 src 는 자신의 패키지 폴더안의 src 라는 소스코드를 담는 폴더를 말한다.)
$ nano hello_world_node.cpp
```

```c
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <sstream>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hello_world_node");
  ros::NodeHandle nh;
  ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("say_hello_world", 1000);

  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
    std_msgs::String msg;
    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();
    ROS_INFO("%s", msg.data.c_str());
    chatter_pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
    ++count;
  }
  return 0;
}
```

## 패키지 빌드

이제 패키지 빌드를 위한 모든 작업이 완료되었다.  빌드에 앞서서 다음의 명령어로 로스 패키지의 프로파일을 갱신시켜주자. 앞서 제작한 우리의 패키지를 로스 패키지 목록에 반영시켜주는 명령어이다. 필수 사항은 아니지만 새로 패키지를 생성한 후에 갱신해주면 이용하기 편하다.

```
$ rospack profile
```
다음은 캐킨 빌드 이다. 캐킨 작업 폴더로 이용하여 캐킨 빌드를 해주자.

```
$ cd ~/catkin_ws && catkin_make
또는
$ cm
```
이전 강좌인 "ROS 강좌  06. ROS 개발 환경 구축"에서 언급했듯이 bashrc 파일에 alias cm='cd /catkin_ws && catkin_make' 라고 설정해두면 터미널 창에서 "cm" 이라는 간단한 명령어로 위의 명령어를 대체할 수 있다. 유용한 만큼 이전 강좌를 보고 꼭 설정해 두도록 하자

##  노드 실행

에러 없이 빌드가 완료되었으면 "/catkin_ws/devel/lib/my_first_ros_pkg" 에 "hello_world_node" 라는 파일이 생성되었을 것이다. 한번 확인해 보자.

다음 단계는 노드를 실행하는 것인데 노드 실행에 앞서서 roscore를 구동하자. 로스의 모든 노드는 roscore를 구동한 후에 이용할 수 있다.

```
$ roscore
```
마지막으로 새로운 터미널창을 열어 아래의 명령어로 노드를 실행해보자. my_first_ros_pkg 라는 패키지의 hello_world_node 라는 노드를 실행하라는 명령어이다.

```
$ rosrun my_first_ros_pkg hello_world_node 

[ INFO] [1380598894.131775283]: hello world 0
[ INFO] [1380598894.231826916]: hello world 1
[ INFO] [1380598894.331798085]: hello world 2
[ INFO] [1380598894.431796634]: hello world 3
[ INFO] [1380598894.531808660]: hello world 4
[ INFO] [1380598894.631800431]: hello world 5
[ INFO] [1380598894.731805683]: hello world 6
```

노드를 실행하게 되면 위와 같이 hello world 1, 2 ,3 과 같은 메시지가 발행되는 것을 볼 수 있을 것이다. 이번 강좌는 로스의 빌드 시스템을 설명하기 위한 것이니 노드의 소스코드에 관해서는 다음 강좌를 통해 알아보도록 하자.

## ROS 명령어

[ROS 쉘 명령어]

    roscd (★★★) ros+cd(changes directory) : ROS 패키지 또는 스택의 디렉토리 변경 명령어
    rospd (☆☆☆) ros+pushd : ROS 디렉토리 인덱스에 디렉토리 추가
    rosd (☆☆☆) ros+directory : ROS 디렉토리 인덱스 확인 명령어
    rosls (★☆☆) ros+ls(lists files) : ROS 패키지의 파일 리스트를 확인하는 명령어
    rosed (★☆☆) ros+ed(editor) : ROS 패키지의 파일을 편집하는 명령어
    roscp (☆☆☆) ros+cp(copies files) : ROS 패키지의 파일 복사하는 명령어


[ROS 실행 명령어]

    roscore (★★★) ros+core : master (ROS 네임 서비스) + rosout (stdout/stderr) + parameter server (매개변수관리)
    rosrun (★★★) ros+run : 패키지의 노드를 실행하는 명령어
    roslaunch (★★★) ros+launch : 패키지의 노드를 복수개 실행하는 명령어
    rosclean (★☆☆) ros+clean : ros 로그 파일을 체크하거나 삭제하는 명령어


[ROS 정보 명령어]

    rostopic (★★★) ros+topic : ROS 토픽 정보를 확인하는 명령어
    rosservice (★★★) ros+service : ROS 서비스 정보를 확인하는 명령어
    rosnode (★★★) ros+node : ROS의 노드 정보를 얻는 명령어
    rosparam (★★★) ros+param(parameter) : ROS 파라미터 정보를 확인, 수정 가능한 명령어
    rosmsg (★★☆) ros+msg : ROS 메세지 선언 정보를 확인하는 명령어
    rossrv (★★☆) ros+srv : ROS 서비스 선언 정보를 확인하는 명령어
    roswtf (☆☆☆) ros+wtf : ROS 시스템을 검사하는 명령어
    rosversion (★☆☆) ros+version : ros 패키및 배포 릴리즈 버전의 정보를 확인하는 명령어
    rosbag (★★★) ros+bag : ROS 메세지를 기록, 재생하는 명령어

[ROS 캐킨 명령어]

"로봇 운영체제 강좌 : 11. ROS 빌드 시스템"에서 일부 설명하였음

    catkin_create_pkg (★★★) 캐킨 빌드 시스템에 의한 패키지 자동 생성
    catkin_eclipse (★★☆) 캐킨 빌드 시스템에 의해 생성된 패키지를 이클립스에서 사용할 수 있도록 변경하는 명령어
    catkin_find (☆☆☆) 캐킨 검색
    catkin_generate_changelog (☆☆☆) 캐킨 변경로그 생성
    catkin_init_workspace (★☆☆) 캐킨 빌드 시스템의 작업폴더 초기화
    catkin_make (★★★) 캐킨 빌드 시스템을 기반으로한 빌드 명령어


[ROS 패키지 명령어] 

"로봇 운영체제 강좌 : 11. ROS 빌드 시스템"에서 일부 설명하였음

    rosmake (☆☆☆) ros+make : ROS package 를 빌드한다. (구 ROS 빌드 시스템에서 사용됨)
    rosinstall (★☆☆) ros+install : ROS 추가 패키지 설치 명령어
    roslocate (☆☆☆) ros+locate : ROS 패키지 정보 관련 명령어 
    roscreate-pkg (☆☆☆) ros+create-pkg : ROS 패키지를 자동 생성하는 명령어 (구 ROS 빌드 시스템에서 사용됨)
    rosdep (★☆☆) ros+dep(endencies) : 해당 패키지의 의존성 파일들을 설치하는 명령어
    rospack (★★☆) ros+pack(age) : ROS 패키지와 관련된 정보를 알아보는 명령어


## ROS 도구

    RVIZ : 3D visualization tool / 3차원 시각화 툴  
    rqt_bag : Logging and Visualization Sensor Data, rosbag gui tool / 메시지 기록 GUI 유틸리티 툴  
    rqt_plot : Data Plot Tool / 2차원 데이터 플롯 툴   
    rqt_graph : GUI plugin for visualizing the ROS computation graph / 노드 및 메시지간의 상관 관계 툴  
    rqt : Qt-based framework for GUI development / ROS GUI 개발 툴  


## RViz 설치 및 실행

ROS 설치시에 기본 설치라고도 부를 수 있는 "Desktop-Full Install" 를 설치하게되면 RViz는 기본적으로 설치되어 있다. 만약에 "Desktop-Full Install" 으로 설치하지 않았거나, RViz 가 설치되어 있지 않을 경우에는 아래의 명령어로 설치할 수있다.

```
sudo apt-get install ros-indigo-rviz
```
RViz 의 실행 명령어는 아래와 같다. (단, roscore 가 실행되어 있어야 한다.)

```
rosrun rviz rviz
```

1) 3D 뷰 (3D view)
: 위 화면의 가운데의 검정색 부분을 가르킨다. 각종 데이타를 3차원으로 볼 수 있는 메인 화면이다.

2) 디스플레이(Displays) 
: 왼쪽에 있는 디스플레이 화면은 각종 토픽으로부터 사용자가 원하는 데이타의 뷰를 선택하는 화면이다.

3) 메뉴 (Menu)
: 메뉴는 상단에 놓여져 있다. 현재의 뷰 상태를 저장하거나 읽어오는 명령, 각종 패널의 뷰 옵션을 체크할 수 있다.

4) 툴 (Tools)
: 대부분 네비게이션에 필요한 툴들이 놓여져 있다. 상세한 설명은 네비게이션을 다룰때 설명하도록 하겠다.

5) 뷰 (Views)
: 3D 뷰의 시점을 변경한다.

6) 시간 (Time)
: 현재 시간과 ROS Time 을 실시간으로 보여준다.

## rqt 설치

ROS 설치시에 기본 설치라고도 부를 수 있는 "Desktop-Full Install" 를 설치하게되면 rqt는 기본적으로 설치되어 있다. 만약에 "Desktop-Full Install" 으로 설치하지 않았거나, rqt 가 설치되어 있지 않을 경우에는 아래의 명령어로 설치할 수있다.

```
sudo apt-get install ros-indigo-rqt ros-indigo-rqt-common-plugins
```
추가로, rqt_graph 에서는 그래프 생성을 위하여 추가적으로 설치해야할 파일이 있다. rqt_graph 에서는 PyQtGraph, MatPlot, QwtPlot 을 지원하는데 우리는 rqt_graph 가 추천하는 PyQtGraph 을 사용하도록 하자.

http://www.pyqtgraph.org/downloads/python-pyqtgraph_0.9.8-1_all.deb 에서 deb 파일을 받고 클릭하여 설치하도록 하자. 그 뒤 아래와 같이 rqt_graph를 실행한 후, 프로그램의 오른쪽 상단에 있는 옵션을 의미하는 "기어" 모양의 아이콘을 클릭하면 아래의 첨부 그림과 같이 옵션을 선택할 수 있는데 PyQtGraph 를 선택해주면 된다. PyQtGraph 이외에도 MatPlot, QwtPlot 도 이용 가능하니 원하는 그래프 관련 라이브러리를 이용하면 된다.

```
rqt_graph
```

## rqt_plot

rqt_plot 은 2차원 데이터 플롯 툴이다. 플롯이라하면 좌료를 그리다라는 의미이다. 즉, ROS 메시지를 받아서 이를 좌표에 뿌리게 되는 것을 의미한다. 예를들어 turtlesim 노드 pose 메시지의 x 좌표와  y좌표를 좌표에 표기해보도록 하자.

우선, turtlesim 패키지의 turtlesim_node 을 구동하자.

```
rosrun turtlesim turtlesim_node 
```

다음으로, rqt_plot 을 아래의 조건으로 구동하여 좌표를 작성한다. (원래는 rqt 를 구동후에 Plot 플러그인을 불러와서 GUI환경에서 토픽을 설정하면 되야하지만, 현재 버전에서는 이상하게 구동하지 않는다. 그러므로 아래의 명령어로 대채하여 설명한다.)

```
rqt_plot /turtle1/pose/
```

다음으로, turtlesim 패키지의 turtle_teleop_key 을 구동하여, 화면속의 거북이를 이리저리 움직여보자.

```
rosrun turtlesim turtle_teleop_key
```

![](../../pictures/ros/pyqtgraph.png){:height="50%" width="50%"}

## Image View

카메라의 이미지 데이터를 표시하는 플러그인이다. 이미지 처리 프로세스는 아니지만, 단순히 영상을 확인하는 용도로는 매우 간단하기에 유용하다.

일반 USB CAM의 경우, UVC을 지원하기에 ROS의 "uvc_camera" 패키지를 이용하면 된다. 우선, 아래의 명령어로 "uvc_camera" 패키지를 설치하도록 하자.

```
sudo apt-get install ros-indigo-uvc-camera 
```

USB CAM을 컴퓨터의 USB에 연결하고, 아래의 명령어로 uvc_camera 패키지의 uvc_camera_node 노드를 구동하자.

```
rosrun uvc_camera uvc_camera_node
```

그 후, 아래의 명령어로 rqt를 구동후, 플러그인(Plugins) 메뉴에서 이미지 뷰(Image View)를 선택한다. 그 뒤 왼쪽 상단의 메시지 선택란을 "/image_raw"를 선택하면 아래의 화면처럼 영상을 확인할 수 있다. 

```
rqt
```

![](../../pictures/ros/rqt.png){:height="50%" width="50%"}

## rqt_bag

메시지 기록을 시각화한 GUI 툴이다. "로봇 운영체제 강좌 : ROS 정보 명령어 (rosbag)" 에서 다룬 내용을 시각화하면 편집 가능한 툴로 이미지 값등과 함께 편집할 때 매우 유용한 툴이다.

이를 테스트하기 위해서 위헤서 다룬 rqt_graph 및 Image View 에서 다룬 turtlesim 및 uvc camera 관련의 노드들을 전부 실행해 주자. 그 뒤, 아래의 명령어로 카메라의 "/image_raw " 와 터틀시뮬레이션의 "/turtle1/cmd_vel" 값을 bag 파일로 생성하자.

```
rosbag record /image_raw /turtle1/cmd_vel
```

그 후, 아래의 명령어로 rqt를 구동후, 플러그인(Plugins) 메뉴에서 Bag를 선택한다. 그 뒤 왼쪽 상단의 폴더 모양(Load Bag)의 아이콘을 선택하여 방금전에 기록해둔 .bag 파일을 불러오도록 하자. 그러면 아래의 화면처럼 영상 및 cmd_vel 값을 확인할 수 있다. 또한, 이를 확대, 재생, 시간별 데이터 수 등을 확인할 수 있으며, 오른쪽 마우스를 누르면 Publish 라는 옵션이 있는데 이를 통해 메시지를 다시 발행할 수도 있다.

```
rqt
```

![](../../pictures/ros/rqtbag.png){:height="50%" width="50%"}

## publisher and subscriber imple 패키지 생성

1) 작업 폴더로 이동

```
$ cd ~/catkin_ws/src
```

2) 패키지 생성
: 아래의 명령어는 "oroca_ros_tutorials" 라는 패키지를 생성하는 명령어이다. 이 패키지는 의존하는 패키지로 "std_msgs"와 "roscpp"를 옵션으로 달아주었다. 로스의 표준 메시지 패키지인 std_msgs 와 로스에서 c/c++을 사용하기 위하여 클라이언트라이브러인 roscpp를 사용하겠다는 것으로 패키지 생성에 앞어서 미리 설치해야한다는 의미이다. 이러한 의존하는 패키지의 설정은 패키지 생성할 때 지정할 수도 있지만, 생성 후 package.xml 에서 직접 입력하여도 된다.

```
$ catkin_create_pkg oroca_ros_tutorials std_msgs roscpp
```

위와 같이 패키지를 생성하였으면 "/catkin_ws/src"에 "oroca_ros_tutorials" 라는 패키지 폴더 및 ROS 패키지가 갖추어야할 기본 내부 폴더 및 CMakeLists.txt 와 package.xml가 생성된다. 다음은 아래와 같이 ls 명령어를 입력하여 내용을 보던가 윈도우의 탐색기와 같은 역할을 하는 GUI기반의 Nautilus를 이용하여 패키지 내부를 살펴보도록 하자.  
```
$ ls

include ...................... 인클루드 폴더
src ............................. 소스코드 폴더
CMakeLists.txt .......... 빌드 설정 파일
package.xml .............. 패키지 설정 파일
```

3)패키지 설정 파일 (package.xml) 수정

ROS의 필수 설정 파일 중 하나인 package.xml 은 패키지 정보를 담은 XML 파일로써 패키지의 이름, 저작자, 라이선스, 의존성 패키지 등을 기술하고 있다. 아래의 명령어로 gedit 툴을 이용하여 파일을 열고 현재의 노드에 맞도록 수정해보자.

```
$ gedit package.xml 
```

4) 빌드 설정 파일 (CMakeLists.txt) 수정

ROS의 빌드 시스템인 캐킨(cakin)은 기본적으로 CMake를 이용하고 있어서 패키지 폴더에 CMakeLists.txt 라는 파일에 빌드 환경을 기술하고 있다. 이는 실행 파일 생성, 의존성 패키지 우선 빌드, 링크 생성 등을 설정하게 되어 있다.
```
$ gedit CMakeLists.txt 
```

```python
cmake_minimum_required(VERSION 2.8.3)
project(oroca_ros_tutorials)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs message_generation)

## Declare ROS messages and services
add_message_files(FILES msgTutorial.msg)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES oroca_ros_tutorials
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS system_lib
)

## Build node
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(ros_tutorial_msg_publisher src/ros_tutorial_msg_publisher.cpp)
target_link_libraries(ros_tutorial_msg_publisher ${catkin_LIBRARIES})
add_dependencies(ros_tutorial_msg_publisher oroca_ros_tutorials_generate_messages_cpp)

add_executable(ros_tutorial_msg_subscriber src/ros_tutorial_msg_subscriber.cpp)
target_link_libraries(ros_tutorial_msg_subscriber ${catkin_LIBRARIES})
add_dependencies(ros_tutorial_msg_subscriber oroca_ros_tutorials_generate_messages_cpp)

```

5) 메시지 파일 작성

CMakeLists.txt 에 라는 파일에 "add_message_files(FILES msgTutorial.msg)" 라는 옵션을 넣었다. 이는 이번 노드에서 사용할 메시지인 msgTutorial.msg 를 빌드할때 포함하라는 이야기이다. 현재, msgTutorial.msg 는 생성하지 않았기에 아래와 같은 순서로 생성해주도록 하자.
```
$ cd ~/catkin_ws/src/oroca_ros_tutorials/     (패키지 폴더로 이동한다.)
$ mkdir msg     (패키지에 msg 라는 메시지 폴더를 신규 작성한다.)
$ cd msg     (작성한 msg 폴더로 이동)
$ gedit msgTutorial.msg    (msgTutorial.msg 파일 신규 작성 및 내용 수정)
```
내용으로는 아래와 같이 int32 메시지 형식에 data라는 이름의 메시지를 만들어주자.
```
int32 data
```

6) 발행자 노드 작성

add_executable(ros_tutorial_msg_publisher src/ros_tutorial_msg_publisher.cpp)

CMakeLists.txt 에 위와 같은 실행 파일을 생성하는 옵션을 주었다. 즉, "ros_tutorial_msg_publisher.cpp"라는 파일을 빌드하여 "ros_tutorial_msg_publisher"라는 실행 파일을 만들라는 이야기이다. 아래의 순서대로 발행자 노드 기능을 수행하는 "ros_tutorial_msg_publisher.cpp" 소스를 작성해 보자. 
```
$ cd ~/catkin_ws/src/oroca_ros_tutorials/     (패키지 폴더로 이동한다.)
$ cd src     (노드의 소스코드 폴더인 src 폴더로 이동)
$ gedit ros_tutorial_msg_publisher.cpp    (ros_tutorial_msg_publisher.cpp 파일 신규 작성 및 내용 수정)
```

```c
#include "ros/ros.h"                                    // ROS 기본 헤더파일
#include "oroca_ros_tutorials/msgTutorial.h"            // msgTutorial 메시지 파일 헤더 (빌드후 자동 생성됨)

int main(int argc, char **argv)                         // 노드 메인 함수
{
  ros::init(argc, argv, "ros_tutorial_msg_publisher");  // 노드명 초기화
  ros::NodeHandle nh;                                   // ROS 시스템과 통신을 위한 노드 핸들 선언

  // 발행자 선언, oroca_ros_tutorials 패키지의 msgTutorial 메시지 파일을 이용한
  // 발행자 ros_tutorial_pub 를 작성한다. 토픽명은 "ros_tutorial_msg" 이며,
  // 발행자 큐(queue) 사이즈를 100개로 설정한다는 것이다
  ros::Publisher ros_tutorial_pub = nh.advertise<oroca_ros_tutorials::msgTutorial>("ros_tutorial_msg", 100);

  // 루프 주기를 설정한다. "10" 이라는 것은 10Hz를 말하는 것으로 0.1초 간격으로 반복된다
  ros::Rate loop_rate(10);

  int count = 0;    // 메시지에 사용될 변수 선언

  while (ros::ok())
  {
    oroca_ros_tutorials::msgTutorial msg;      // msgTutorial 메시지 파일 형식으로 msg 라는 메시지를 선언
    msg.data = count;                   // count 라는 변수를 이용하여 메시지 값을 정한다

    ROS_INFO("send msg = %d", count);   // ROS_INFO 라는 ROS 함수를 이용하여 count 변수를 표시한다

    ros_tutorial_pub.publish(msg);      // 메시지를 발행한다. 약 0.1초 간격으로 발행된다

    loop_rate.sleep();                  // 위에서 정한 루프 주기에 따라 슬립에 들어간다

    ++count;                            // count 변수 1씩 증가
  }

  return 0;
}

```

7) 구독자 노드 작성

add_executable(ros_tutorial_msg_subscriber src/ros_tutorial_msg_subscriber.cpp)

CMakeLists.txt 에 위와 같은 실행 파일을 생성하는 옵션을 주었다. 즉, "ros_tutorial_msg_subscriber.cpp"라는 파일을 빌드하여 "ros_tutorial_msg_subscriber"라는 실행 파일을 만들라는 이야기이다. 아래의 순서대로 구독자 노드 기능을 수행하는 "ros_tutorial_msg_subscriber.cpp" 소스를 작성해 보자. 
```
$ cd ~/catkin_ws/src/oroca_ros_tutorials/     (패키지 폴더로 이동한다.)
$ cd src     (노드의 소스코드 폴더인 src 폴더로 이동)
$ gedit ros_tutorial_msg_subscriber.cpp    (ros_tutorial_msg_subscriber.cpp 파일 신규 작성 및 내용 수정)
```
```c
#include "ros/ros.h"                                    // ROS 기본 헤더파일
#include "oroca_ros_tutorials/msgTutorial.h"            // msgTutorial 메시지 파일 헤더 (빌드후 자동 생성됨)

// 메시지 콜백함수로써, 밑에서 설정한 ros_tutorial_sub 구독자에 해당되는 메시지를
// 수신하였을때 동작하는 함수이다
// 입력 메시지로는 oroca_ros_tutorial 패키지의 msgTutorial 메시지를 받도록 되어있다
void msgCallback(const oroca_ros_tutorials::msgTutorial::ConstPtr& msg)
{
  ROS_INFO("recieve msg: %d", msg->data);   // 수신된 메시지를 표시하는 함수
}

int main(int argc, char **argv)                         // 노드 메인 함수
{
  ros::init(argc, argv, "ros_tutorial_msg_subscriber"); // 노드명 초기화

  ros::NodeHandle nh;                                   // ROS 시스템과 통신을 위한 노드 핸들 선언

  // 구독자 선언, oroca_ros_tutorials 패키지의 msgTutorial 메시지 파일을 이용한
  // 구독자 ros_tutorial_sub 를 작성한다. 토픽명은 "ros_tutorial_msg" 이며,
  // 구독자 큐(queue) 사이즈를 100개로 설정한다는 것이다
  ros::Subscriber ros_tutorial_sub = nh.subscribe("ros_tutorial_msg", 100, msgCallback);

  // 콜백함수 호출을 위한 함수로써, 메시지가 수신되기를 대기, 수신되었을 경우 콜백함수를 실행한다
  ros::spin();

  return 0;
}
```

8)ROS 노드 빌드
```
$ cd ~/catkin_ws     (catkin 폴더로 이동)
$ catkin_make     (catkin 빌드 실행)
```
위의 명령어로 oroca_ros_tutorials 패키지의 메지시 파일, 발행자 노드, 구독자 노드가 빌드되었다. 

oroca_ros_tutorials 패키지의 소스는 ~/catkin_ws/src/oroca_ros_tutorials/src 에 존재하고,
oroca_ros_tutorials 패키지의 메시지 파일은 ~/catkin_ws/src/oroca_ros_tutorials/msg 에 존재한다.

이를 기반으로 빌드된 결과물은 ~/catkin_ws/build 및 ~/catkin_ws/devel 에 각각 생성된다.
/catkin_ws/build 에는 캐킨 빌드에서 사용된 설정 내용이 저장되며,
/catkin_ws/devel/lib/oroca_ros_tutorials 에는 실행 파일이,
/catkin_ws/devel/include/oroca_ros_tutorials 에는 메시지 파일로부터 자동 생성된 메시지 헤더파일이 저장된다.
각 생성 결과물이 궁금하다면 이 경로에 생성된 결과물을 확인해 보자.

9) 발행자 실행

(※ 주의! 노드 실행에 앞서서 roscore를 실행해주는 것을 잊지 말자!)
```
$ rosrun oroca_ros_tutorials ros_tutorial_msg_publisher
```
ROS 노드 실행 명령어인 rosrun 을 이용하여, oroca_ros_tutorials 패키지의 ros_tutorial_msg_publisher 노드를 구동하라는 명령어이다. 이를 실행하게 되면 아래와 같은 출력 화면을 볼 수 있다. 내부에 선언된 count 값이 표시되고 있으며, ROS 메시지로 발부되고 있다.

```
$ rostopic list

/ros_tutorial_msg
/rosout
/rosout_agg

$ rostopic echo /ros_tutorial_msg
```

10) 구독자 실행
```
$ rosrun oroca_ros_tutorials ros_tutorial_msg_subscriber 
```
ROS 노드 실행 명령어인 rosrun 을 이용하여, oroca_ros_tutorials 패키지의 ros_tutorial_msg_subscriber  노드를 구동하라는 명령어이다. 이를 실행하게 되면 아래와 같은 출력 화면을 볼 수 있다. 발행자에서 발행된 "ros_tutorial_msg" 토픽의 메시지를 수신받아 값이 표시되고 있다

11) ROS Graph

ROS Graph의 단일 수행 명령어인 rqt_graph 를 실행하던가,
```
$ rqt_graph

또는

$ rqt
```

## 서비스 서버 노드와 클라이언트 노드 작성 및 실행
[서비스 서버 노드와 클라이언트 노드 작성 및 실행](https://cafe.naver.com/openrt/3044)

```python
cmake_minimum_required(VERSION 2.8.3)
project(oroca_ros_tutorials)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs message_generation)

## Declare ROS messages and services
add_message_files(FILES msgTutorial.msg)
add_service_files(FILES srvTutorial.srv)

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES oroca_ros_tutorials
  CATKIN_DEPENDS roscpp std_msgs
  DEPENDS system_lib
)

## Build node
include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(ros_tutorial_msg_publisher src/ros_tutorial_msg_publisher.cpp)
target_link_libraries(ros_tutorial_msg_publisher ${catkin_LIBRARIES})
add_dependencies(ros_tutorial_msg_publisher oroca_ros_tutorials_generate_messages_cpp)

add_executable(ros_tutorial_msg_subscriber src/ros_tutorial_msg_subscriber.cpp)
target_link_libraries(ros_tutorial_msg_subscriber ${catkin_LIBRARIES})
add_dependencies(ros_tutorial_msg_subscriber oroca_ros_tutorials_generate_messages_cpp)

add_executable(ros_tutorial_srv_server src/ros_tutorial_srv_server.cpp)
target_link_libraries(ros_tutorial_srv_server ${catkin_LIBRARIES})
add_dependencies(ros_tutorial_srv_server oroca_ros_tutorials_generate_messages_cpp)

add_executable(ros_tutorial_srv_client src/ros_tutorial_srv_client.cpp)
target_link_libraries(ros_tutorial_srv_client ${catkin_LIBRARIES})
add_dependencies(ros_tutorial_srv_client oroca_ros_tutorials_generate_messages_cpp)
 
```

## 매개변수(parameter)

1) 매개변수(parameter)

노드에서 사용되는 매개변수를 말한다. 흔히, 윈도우즈 프로그램에서 .ini 설정파일과 같다고 생각하면 된다. 디폴트로 설정 값들이 지정되어 있고, 필요에 의해서 외부에서 이 매개변수를 읽기, 쓰기가 가능하다. 특히, 상황에 맞추어 이 매개변수를 외부에서 쓰기기능을 이용하여 설정값을 실시간으로 바꿀수 있기에 매우 유용한 방법이다. 예를들어 접속하는 USB포트 및 카메라 캘리브레이션 값, 속도 및 명령어들의 최대/최저 값 등의 설정등을 지정할 수 있다.

2) 매개변수를 활용한 노드 작성

이번 강좌에서는 이전 강좌 "로봇 운영체제 강좌 : 16. 서비스 서버 노드와 클라이언트 노드 작성 및 실행" 에서 다룬 "ros_tutorial_srv_server.cpp" 의 소스를 수정하여 서비스 요청으로 입력된 "a" 와 "b" 를 단순히 덧셈하는 것이 아니라, 사칙연산을 할 수있도록 매개변수를 활용해 볼 것이다.

아래의 순서대로 이전 강좌에서 작성해둔 "ros_tutorial_srv_server.cpp" 소스를 수정하도록 하자.
```
$ roscd oroca_ros_tutorials     (패키지 폴더로 이동한다.)

$ cd src     (노드의 소스코드 폴더인 src 폴더로 이동)

$ gedit ros_tutorial_srv_server.cpp    (ros_tutorial_srv_server.cpp 파일 신규 작성 및 내용 수정)
```

```c
#include "ros/ros.h"                         // ROS 기본 헤더파일
#include "oroca_ros_tutorials/srvTutorial.h" // srvTutorial 서비스 파일 헤더 (빌드후 자동 생성됨)
 
#define PLUS           1    // 덧셈
#define MINUS          2    // 빼기
#define MULTIPLICATION 3    // 곱하기
#define DIVISION       4    // 나누기
 
int g_operator = PLUS;
 
// 서비스 요청이 있을 경우, 아래의 처리를 수행한다
// 서비스 요청은 res, 서비스 응답은 req로 설정하였다
bool calculation(oroca_ros_tutorials::srvTutorial::Request  &req,
                 oroca_ros_tutorials::srvTutorial::Response &res)
{
  // 서비스 요청시 받은 a와 b 값을 파라미터값에 따라 연산자를 달리한다.
  // 계산한 후 서비스 응답값에 저장한다
  switch(g_operator){
    case PLUS:
         res.result = req.a + req.b; break;
    case MINUS:
         res.result = req.a - req.b; break;
    case MULTIPLICATION:
         res.result = req.a * req.b; break;  
    case DIVISION:
         if(req.b == 0){
           res.result = 0; break;
         }  
         else{
           res.result = req.a / req.b; break;  
         }
    default:
         res.result = req.a + req.b; break;
  }
 
  // 서비스 요청에 사용된 a, b값의 표시 및 서비스 응답에 해당되는 result 값을 출력한다
  ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
  ROS_INFO("sending back response: [%ld]", (long int)res.result);
 
  return true;
}
 
int main(int argc, char **argv)                     // 노드 메인 함수
{
  ros::init(argc, argv, "ros_tutorial_srv_server"); // 노드명 초기화
 
  ros::NodeHandle nh;                     // ROS 시스템과 통신을 위한 노드 핸들 선언
 
  nh.setParam("calculation_method", PLUS); // 매개변수 초기설정
 
  // 서비스 서버 선언, oroca_ros_tutorials 패키지의 srvTutorial 서비스 파일을 이용한
  // 서비스 서버 ros_tutorial_service_server 를 작성한다. 서비스명은 "ros_tutorial_srv" 이며,
  // 서비스 요청이 있을경우, calculation 라는 함수를 실행하라는 설정이다
  ros::ServiceServer ros_tutorial_service_server = nh.advertiseService("ros_tutorial_srv", calculation);
 
  ROS_INFO("ready srv server!");
   
  ros::Rate r(10); // 10 hz
 
  while (1)
  {
    nh.getParam("calculation_method", g_operator);  // 연산자를 매개변수로부터 받은 값으로 변경한다
    ros::spinOnce();  // 콜백함수 처리루틴
    r.sleep();        // 루틴 반복을 위한 sleep 처리
  }
 
  return 0;
}
```

3) 매개변수 설정

아래의 소스는 "calculation_method" 라는 이름의 매개변수를 PLUS 라는 값으로 설정한다는 것이다. PLUS 는 위 소스에서 1 이므로 "calculation_method" 매개변수는 1이 되고, 위 소스에서 서비스 요청으로 받은 값을 덧셈하여 서비스 응답을 하게 된다.
```
  nh.setParam("calculation_method", PLUS); // 매개변수 초기설정
```
참로고 매개변수는 integers, floats, boolean, string, dictionaries, list 등으로 설정할 수 있다. 간단히 예를 들자면, 1 은 integer, 1.0은 floats, internetofthings은 string, true는 boolean, [1,2,3]은 integers 의 list, {a: b, c: d}은 dictionary이다. 

4) 매개변수 읽기

아래의 소스는 "calculation_method" 라는 이름의 매개변수를 불러와서 g_operator 의 값으로 설정한다는 것이다. 이에 따라서 위 소스에서의 g_operator 는 매 0.1초마다 매개변수의 값을 확인하여 서비스 요청으로 받은 값을 사칙연사중 어떤 계산을 하여 처리할 지 결정하게 된다.
```
  nh.getParam("calculation_method", g_operator);  // 연산자를 매개변수로부터 받은 값으로 변경한다
```

5) 노드 빌드 및 실행
```
$ cd ~/catkin_ws     (catkin 폴더로 이동)

$ catkin_make     (catkin 빌드 실행)

위의 명령어로 oroca_ros_tutorials 패키지의 서비스 서버 노드가 빌드되었다. 

$ rosrun oroca_ros_tutorials ros_tutorial_srv_server 

[ INFO] [1385278089.933322980]: ready srv server!
```
위 명령어를 실행하면 서비스 서버는 서비스 요청 대기를 하게 된다.

6) 매개변수 리스트 보기
```
$ rosparam list

/calculation_method
/rosdistro
/roslaunch/uris/host_192_168_4_185__60432
/rosversion
/run_id
```
"rosparam list" 명령어로 현재 ROS 네트워크에 사용된 매개변수의 목록을 확인할 수 있다. 위에 출력된 목록중 "/calculation_method" 가 우리가 사용한 매개변수이다.

7) 매개변수 사용예

아래의 명령어 대로 매개변수를 설정해보고, 매번 같은 서비스 요청을 하여 서비스 처리가 달라짐을 확인해보자.
```
$ rosservice call /ros_tutorial_srv 10 5
result: 15
$ rosparam set /calculation_method 2
$ rosservice call /ros_tutorial_srv 10 5
result: 5
$ rosparam set /calculation_method 3
$ rosservice call /ros_tutorial_srv 10 5
result: 50
$ rosparam set /calculation_method 4
$ rosservice call /ros_tutorial_srv 10 5
result: 2 
```

## 로스런치

1) 로스런치(roslaunch)

로스런(rosrun)이 하나의 노드를 실행하는 명령어라면 로스런치(roslaunch)는 복 수개의 노드를 실행하는 개념이다. 이 명령어를 통해 정해진 단일 혹은 복수의 노드를 실행시킬 수 있다. 

그 이외의 기능으로 실행시에 패키지의 매개변수를 변경, 노드 명의 변경, 노드 네임 스페이스 설정, ROS_ROOT 및 ROS_PACKAGE_PATH 설정, 이름 변경, 환경 변수 변경 등의 실행시 변경할 수 있는 많은 옵션들을 갖춘 노드 실행에 특화된 로스 명령어이다.

로스런치는 ".launch" 라는 로스런치파일을 사용하여 실행 노드에 대한 설정을 해주는데 이는 XML 기반으로 되어 있으며, 태그별 옵션을 제공하고 있다. 실행 명령어로는 "roslaunch 패키지명 로스런치파일" 이다.

2) 로스런치의 활용

로스런치의 활용으로 이전 강좌인 "로봇 운영체제 강좌 : 15. 메시지 발행자 노드와 구독자 노드 작성 및 실행" 에서 작성한 ros_tutorial_msg_publisher 와  ros_tutorial_msg_subscriber 를 이름을 바꾸어서 실행해보자. 그냥 이름을 바꾸어 의미가 없으니, 발신자 노드와 구독자 노드를 각각 두 개씩 구동하여 서로간에 별도로 메시지 통신을 해보도록 하겠다.

우선, .launch 파일을 작성하자. 로스런치 파일은 .launch 이라는 파일명을 가지고 있으며, 노드 폴더에 로스런치를 저장할 launch 라는 폴더를 생성해줘야 한다. 아래의 명령어대로 폴더를 생성하고 새롭게 union.launch 이라는 파일으로 로스런치 파일을 생성해보자.

```
$ cd ~/catkin_ws/src/oroca_ros_tutorials
$ mkdir launch
$ cd launch
$ gedit union.launch
```
내용으로는 아래의 내용대로 작성해주도록 하자.
```xml
<launch>
  <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_publisher"   name="msg_publisher1"/>
  <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_subscriber"  name="msg_subscriber1"/>

  <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_publisher"  name="msg_publisher2"/>
  <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_subscriber"  name="msg_subscriber2"/>
</launch>

<launch> 는 로스런치 태그로써 이 태그안에는 로스런치에 필요한 태그들이 기술된다.
<node> 는 로스런치로 실행할 노드를 기술하게 된다. 옵션으로는 pkg, type, name 이 있다. pkg는 패키지의 이름, type는 실제 실행할 노드의 이름, name은 type를 실행하되 실행할때 붙여지는 이름이다.
```

로스런치파일의 작성을 마쳤으면 아래와 같이 로스런치를 실행해주자.
```
$ roslaunch oroca_ros_tutorials union.launch
```

실행 후, 결과가 어떻게 되었을까? 우선 아래와 같이 "rosnode list" 명령어로 현재 실행중인 노드를 살펴보자. 결과적으로  ros_tutorial_msg_publisher 노드가 msg_publisher1 및 msg_publisher2 로 이름이 바뀌어 두 개의 노드가 실행되었으며, ros_tutorial_msg_subscriber 노드도  msg_subscriber1 및 msg_subscriber2 로 이름이 바뀌어 실행되었다.  
```
$ rosnode list

/msg_publisher1
/msg_publisher2
/msg_subscriber1
/msg_subscriber2
/rosout
```

문제는,"발신자 노드와 구독자 노드를 각각 두 개씩 구동하여 서로간에 별도로 메시지 통신" 하게한다는 첫 의도와는 다르게 rqt_graph 를 통해 보면 서로간의 메시지를 모두 구독하고 있다는 것이다. 이는 단순히 실행되는 노드의 이름만을 변경해 주었을뿐 사용되는 메시지의 이름을 바꿔주지 않았기 때문이다. 이 문제를 다른 로스런치 태그를 사용하여 해결해보자.
```
$ rqt_graph
```
![](../../pictures/ros/roslaunchexample.png){:height="50%" width="50%"}

```xml
<launch>

  <group ns="ns1">
    <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_publisher"   name="msg_publisher"/>
    <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_subscriber"  name="msg_subscriber"/>
  </group>

  <group ns="ns2">
    <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_publisher"  name="msg_publisher"/>
    <node pkg="oroca_ros_tutorials" type="ros_tutorial_msg_subscriber"  name="msg_subscriber"/>
  </group>

</launch>

<group> 는 지정된 노드를 그룹으로 묶어주는 태그이다. 옵션으로는 ns 가 있다.
이는 네임스페이스(namespace)로써 그룹의 이름을 지칭하며, 
그룹에 속한 노드의 이름 및 메시지 등도 모두 ns로 지정한 이름에 포함되게 된다
```

![](../../pictures/ros/roslaunchexample2.png){:height="50%" width="50%"}


## 로봇패키지 사용 방법

2) 로봇패키지 사용 방법

만약, 사용하고자 하는 로봇 패키지가 ROS 공식 패키지라면 설치 방법은 매우 간단하다. 우선, 자신이 사용하고자 하는 로봇 패키지가 공개 되었는지 ROS wiki robot (http://wiki.ros.org/Robots ) 에서 확인하거나 아래의 명령어로 전체의 ROS 패키지로부터 찾아 볼 수 있다.
```
$ apt-cache search ros-hydro
```

예를들어 PR2인 경우는 아래의 명령어로 일괄 설치된다.
```
$ sudo apt-get install ros-hydro-pr2-desktop
```
터틀봇2의 경우는 아래와 같으며, 설치관련 wiki 페이지를 보고 몇가지 설정해주면된다.
(http://wiki.ros.org/turtlebot/Tutorials/hydro/Installation)
```
$ sudo apt-get install ros-hydro-turtlebot ros-hydro-turtlebot-apps ros-hydro-turtlebot-viz ros-hydro-turtlebot-simulator ros-hydro-kobuki-ftdi
```
나오의 경우는 아래와 같다. (http://wiki.ros.org/nao/Installation)
```
$ sudo apt-get install ros-hydro-nao-robot ros-hydro-nao-pose ros-hydro-nao-msgs ros-hydro-nao-driver ros-hydro-nao-description ros-hydro-nao-bringup ros-hydro-humanoid-nav-msg
```

만약에 해당 로봇 패키지가 공식적으로 제공되지 않더라도 로봇 패키지 위키에는 설치 방법등을 따로 설명해주고 있다. 예를들어 모바일 로봇으로 유명한 파이오니어(Pioneer)의 경우, 아래와 같이 캐킨 빌드 시스템의 사용자 소스 폴더로 이동한 후, 위키에 적혀진 소스 리포지토리로부터 최신의 로봇 패키지를 다운로드 받으면 된다. 
```
$ cd ~/catkin_ws/src     (캐킨 빌드 시스템의 사용자 소스 폴더로 이동)

$ hg clone http://code.google.com/p/amor-ros-pkg/     (위키에 적혀져 있는 소스 리포지토리로부터 소스 다운로드)
```

## 센서 패키지를 이용한 예제

3) 센서 패키지를 이용한 예제

일반 USB CAM의 경우, UVC을 지원하기에 ROS의 "uvc_camera" 패키지를 이용하면 된다. 우선, 아래의 명령어로 "uvc_camera" 패키지를 설치하도록 하자.
```
sudo apt-get install ros-indigo-uvc-camera 
```
USB CAM을 컴퓨터의 USB에 연결하고, 아래의 명령어로 uvc_camera 패키지의 uvc_camera_node 노드를 구동하자.
```
rosrun uvc_camera uvc_camera_node
```
그 후, 아래의 명령어로 rqt를 구동후, 플러그인(Plugins) 메뉴에서 이미지 뷰(Image View)를 선택한다. 그 뒤 왼쪽 상단의 메시지 선택란을 "/image_raw"를 선택하면 아래의 화면처럼 영상을 확인할 수 있다. 
```
rqt
```

## IMU ,AHRS

4) 설치

4.1 소스 설치
myahrs_driver 패키지를 만들어 두었다. 다음과 같이 설치하도록 하자.
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/robotpilot/myahrs_driver.git
$ cd ~/catkin_ws && catkin_make
```

4.2 바이너리 설치
myahrs_driver 패키지를 바이너리 파일로 설치하려면 다음의 명령으로 설치하도록 하자. 바이너리 파일 공식 등록은 차후에 강좌로 만들어 보기로 하겠다.
```
$ sudo apt-get install ros-indigo-myahrs-driver
```

5) 실행

다음과 같이 실행 가능하다. 뒤에 붙은 _ port 라는 파라미터는 myAHRS 장비를 PC에 접속하였을때 자신의 제어 포트를 말한다. ls /dev/ACM 명령어로 직접 검색해보고 알맞게 입력해 주도록 하자.
```
$ rosrun myahrs_driver myahrs_driver _port:=/dev/ttyACM0
```
더블어 다음과 같이 roslaunch  를 실행해주면 시각화 툴인 RViz도 함께 실행되어 현재 myAHRS 장비의 AHRS 정보를 다음 동영상과 같이 직접 눈으로 확인해 볼 수 있다.
```
$ roslaunch myahrs_driver myahrs_driver.launch
```

## Kobuki의 개발환경

2) 개발환경 

본 강좌의 진행에 앞서서 필자의 개발 환경을 미리 알리고 시작하도록 하겠다. 꼭 반드시 동일할 필요는 없지만, 원할한 강좌 진행을 위해서는 아래와 같이 개발 환경을 권장한다. 이 이외에 환경에서의 질문에 대해서는 질응답은 하지만 실제 필자가 테스트해 볼 수 있는 상황이 아니기 때문에 대처하기 어려울 수도 있다. 필자의 소프트웨어 개발환경으로는 아래와 같다. ROS Indigo의 설치는 "로봇 운영체제 강좌 : Indigo 설치" 를 참고하기 바란다.


● PC 1 (Desktop) 
 - Kobuki 조종, 외/내부 센서 데이터 처리, 네비게이션을 담당한다. 모든 개발은 이 컴퓨터에서 진행되게 된다.
 - Ubuntu 14.04 LTS 64bit (Trusty Tahr)
 - ROS - Indigo
 - ROS 패키지 : sudo apt-get install ros-indigo-kobuki* (거북이 관련 모든 패키지)

● PC 2 (Laptop)
 -  Kobuki 에 직접 탑재할 PC이며 직접적인 제어 및 센서 데이터를 받아 데스트톱으로 전송하게 된다.
 -  Ubuntu 14.04 LTS 64bit (Trusty Tahr)
 -  ROS - Indigo
 -  설치할 ROS 패키지 : ros-indigo-kobuki 와 ros-indigo-kobuki-core

3) 거북이 패키지 설치하기

앞으로의 강좌는 모두 데스크톱에서 진행할 예정이다. 다만, 거북이의 직접적인 제어 및 센서 데이터를 받아 데스크톱으로 전송하기 위하여 거북이용 랩톱에 거북이 관련 패키지를 설치하도록 하자. 아래의 설명은 랩톱과 데스크톱으로 나누어 설명하였으므로 참고하기 바란다.

  - 1) 우선, 랩톱에 "로봇 운영체제 강좌 : Indigo 설치" 를 참고하여 ROS Indigo 를 설치하자. 

  - 2) "로봇 운영체제 강좌 : Indigo 설치" 의 "4. 환경 설정" 에서 랩톱의 경우에는 아래와 같이 설정하도록 하자.
  ROS_MASTER_URI 는 데스크톱의 IP를 와 ROS_IP 에는 랩톱 자기 자신의  IP 로 설정해주도록 하자. 각각의 아이피는 새로운 터미널 창에서 ifconfig 명령어로 확인 할 수 있다.

    ```  
    export ROS_MASTER_URI=http://데스크톱의아이피:11311

    export ROS_IP=랩톱의아이피
    ```  

  - 3) 다음과 같이 거북이의 직접적인 제어를 위한 필수 패키지를 설치하도록 하자.
    ```
    sudo apt-get install ros-indigo-kobuki ros-indigo-kobuki-core
    ```
  - 4) ROS 설치 이후 부터는 SSH(Secure Shell)를 이용하여 데스크톱에서 사용할 예정이기에  ssh를 설치하도록 하자.
    ```
    sudo apt-get install ssh
    ```
  - 5) 데스크톱에 "로봇 운영체제 강좌 : Indigo 설치" 를 참고하여 ROS Indigo 를 설치하자. 

  - 6) "로봇 운영체제 강좌 : Indigo 설치" 의 "4. 환경 설정" 에서 데스크톱의 경우에는 ROS_MASTER_  URI 와 ROS_IP 모두 데스크톱의 IP로 설정해주도록 하자. 
    ```
    export ROS_MASTER_URI=http://데스크톱의아이피:11311

    export ROS_IP=데스크톱의아이피
    ```
  - 7) 거북이 관련 모든 패키지를 설치하도록 하자. (시뮬레이션 관련은 추후에 추가로 설명한다.)
    ```
    sudo apt-get install ros-indigo-kobuki*
    ```

## Kobuki의 원격제어

거북이 구동 테스트

1) ROSCORE 구동
데스크톱에서 터미널창을 하나 열어, roscore 를 구동한다. 
참고로 밑에서 설명하는 각 명령어는 각각 새로운 창을 열어 구동해줘야 한다.
```
roscore
```
2) SSH를 이용한 데스크톱에서 랩톱 접속하기
거북이와 랩톱을 usb 케이블로 연결 후에 전원 스위치를 켜준다. 그 후, 데스크톱에서 ssh 로 랩톱에 접속한다. 예를 들어, 터미널에서 " ssh kobuki@192.168.4.21 " 와 같이 입력해주면 랩톱에 연결할 수 있고, 쉘 명령어로 제어 가능하다.
```
ssh 유저명@랩톱의아이피
```

3) 거북이 포트 생성
랩톱에 전속된 후, 아래와 같이 create_udev_rules 노드를 실행해 주면 /dev/kobuki 포트가 생성된다. 
```
rosrun kobuki_ftdi create_udev_rules
```
실행후, usb 케이블을 연결 해제 후에 다시 꼽아 주도록 하자. 그러면 ls -l 명령어로 해당 포트를 확인할 수 있다.
```
ls -l /dev/kobuki

lrwxrwxrwx 1 root root 7 Aug 24 14:26 /dev/kobuki -> ttyUSB0
```
필자의 경우, /dev/kobuki 의 포트는 본래 ttyUSB0 로 링크되어 있다는 것을 확인할 수 있다. ttyUSB0,1,2 등과 같이 연결 상황에 따라 달라지는 것을 create_udev_rules 노드를 통해 /dev/kobuki 이라는 통일된 포트로 사용할 수 있도록 돕고 있다.

4) 거북이 구동
아래의 명령어를 구동학 되면 비프음으로 소리가 나며, 초기 설정을 마친 후 거북이는 구동 상태로 바뀐다. 랩톱에서 실행해야하는 노드는 이것이 전부이다. 종료는 Ctrl + c 를 이용하면 된다. SSH 사용은 이것으로 마무리 되었다.
```
roslaunch kobuki_node minimal.launch
```
참고로, 아래와 같이 "--screen" 옵션을 붙여주게 되면 구동시에 발생되는 일부 숨겨진 메시지들까지 모두 확인할 수 있다. 거북이 사용중 원인 모를 에러가 있을 경우에는 아래처럼 "--screen" 옵션으로 확인해 보자. 참고로, 필자는 항상 "--screen" 옵션을 붙여주고 실행한다.
```
roslaunch kobuki_node minimal.launch --screen
```
5) 키보드 제어 노드 구동
데스크톱에서 아래와 같이 명령어를 입력해주면 거북이를 원격 제어 할 수 있는 상태가 된다. 사용가능한 키는 밑에 정리해두겠다. 범퍼가 장착되어 있는 부분이 로봇의 앞 부분으로 전진 방향에 주의하도록 하자. 아래에 전진 방향의 그림을 첨부하니 참고하기 바란다.
```
roslaunch kobuki_keyop keyop.launch

방향키 ↑ : 리니어 속도 지정으로 전진 방향으로 이동한다. (0.5씩, 단위 = m/sec) 

방향키 ↓ : 리니어 속도 지정으로 후진 방향으로 이동한다. (0.5씩, 단위 = m/sec) 

방향키 ← : 회전 속도 지정으로 시계 반대 방향으로 회전한다. (0.33씩, 단위 = rad/sec) 

방향키 → : 회전 속도 지정으로 시계 방향으로 회전한다. (0.33씩, 단위 = rad/sec) 

스페이스바 : 리니어 속도 및 회전 속도를 초기화 한다.

d : 모터를 비활성화 한다. (구동 불가능 상태)

e : 모터를 활성화 한다. (구동 가능 상태)

q : 종료
```

## Kobuki의 토픽(Topic)
[Kobuki의 토픽(Topic)](https://cafe.naver.com/openrt/6192)

## Kobuki의 진단툴, 계기판, CUI 및 GUI 기능 테스트
[Kobuki의 진단툴, 계기판, CUI 및 GUI 기능 테스트](https://cafe.naver.com/openrt/6197)

![](../../pictures/ros/kobukigui.png){:height="50%" width="50%"}

## Kobuki의 시뮬레이션 (RViz)

1) 가상 시뮬레이션

거북이는 로봇 본체 없이 가상 로봇을 이용하여 프로그램밍하고 가상 시뮬레이션을 통해 개발하는 것을 지원하고 있다. 그 방법으로는 2가지가 있는데, 하나는 ROS의 3차원 시각화 툴인 RViz 를 이용하는 것이고, 또 다른 하나는 3차원 로봇 시뮬레이터 Gazebo를 이용하는 것이다. 이번 강좌에서는 그 중 첫 번째인 RViz를 이용하는 방법에 대해 알아 보도록 하겠다. 거북이 본체없이 거북이의 구동, 네비게이션을 시험해보기 원하는 유저에게는 강력 추천한다. 로봇 본체 없이도 시뮬레이션을 할 수 있도록 kobuki_soft 를 공개해준 개발자 Jihoon Lee 님에게 이 자리를 빌어 감사하다는 말을 전하고 싶다.


2) kobuki_soft 메타 패키지

kobuki_soft 는 많이 알려지지는 않았지만 ROS의 시각화툴인 RViz 환경에서 거북이의 동작을 시뮬레이션 할 수 있는 가상 시뮬레이션 환경을 제공해주는 메타 패키지이다. 포함된 패키지로는 가상 거북이 구동과 관련된 kobuki_softnode 와 네비게이션 정보를 담고 있는 kobuki_softapps 가 있다. 

kobuki_soft (메타 패키지)
위키: http://wiki.ros.org/kobuki_soft

kobuki_softnode
위키: http://wiki.ros.org/kobuki_softnode
기능: 거북이 시뮬레이션과 관련된 가상 로봇 시뮬레이션 기능의 패키지이다. RViz 에서 실행 가능하다.

kobuki_softapps
위키: http://wiki.ros.org/kobuki_softapps
기능: kobuki_softnode 와 연관되는 패키지로 시뮬레이션 관련 애플리케이션을 가지고 있으며 주로 네비게이션을 다루고 있다. 


가상시뮬레이션을 이용하기 위해서는 kobuki_soft 패키지를 아래와 같이 설치하면 된다.
```
sudo apt-get install ros-indigo-kobuki-soft
```

3) 가상 로봇 실행

우선, 아래와 같이 거북이 소프트노드를 구동한다. 이 런치 파일을 실행하게 되면 kobuki_description 패키지에서 거북이의 3차원 모델을 불러오고, 거북이 노드와 동일한 mobile base nodelet manager, mobile base, diagnostic aggregator, robot state publisher 노드들을 가상으로 동작하게 끔 만들어둔 kobuki_softnode를 실행하게 된다.

이는 내부적으로 turtle_teleop_key 노드에서 발행하는 velocity, motor_power 등의 구동 명령어를 받아서 가상의 구동을 하게 된다. 예를 들어, 속도 명령을 받아서 오드메트리(odometry) 정보를 만들어 토픽으로 발행하게 된다. 마찬가지로 joint states 및 tf 도 발행하게 하여 RViz 에서 거북이의 움직임을 확인할 수 있도록 해준다.

단, 센서 정보는 RViz 에서 사용할 수 없기 때문에 이를 위해서는 물리엔진이 포함된 3차원 시뮬레이터 Gazebo 를 이용해야 한다.  이 부분에서 대해서는 다음 강좌에서 설명하기로 하고 이번 강좌에서는 간단한 이동과 주어진 지도상에서 네비게이션하는 내용만 담도록 하겠다.
``` 
roslaunch kobuki_softnode full.launch
```
그 다음,RViz 를 실행하고, 추가로 왼쪽 디스플레이창(Displays) 의 Global Options 의 fixed frame을 "/odom" 으로 바꿔준다. 그리고 난 후, 디스플레이창의 왼쪽 하단부분의 "Add" 버튼을 눌러, 디스플레이 중 "RobotModel" 을 클릭하여 추가하도록 하자. 이미 full.launch 런치 파일에서 거북이 3차원 모델을 불러온 상태이기 때문에 아래의 그림과 이 거북이의 3차원 모델이 중앙에 표시 될 것이다.
```
rosrun rviz rviz
```

![](../../pictures/ros/kobukirviz.png){:height="50%" width="50%"}

자! 그럼 다음에는 가상의 로봇을 구동시켜 보자. 지금까지 사용하였던 keyop.launch 런치파일을 실행하여 키보드로 직접 구동시켜 보자. 
```
roslaunch kobuki_keyop keyop.launch
```

4) odom 토픽 확인

구동을 확인했으니 오도메트리(odometry) 정보가 제대로 생성되고 발행되는지 확인해보자. 아래와 같이 rostopic 명령어로 확인할 수 도 있을 것이지만, 이번에는 RViz 를 하고 있기때문에 시각적으로 확인해보자. RViz 의 왼쪽 하단의 "Add"를 클릭 후, 나오는 시각화 생성 화면에서 "By Topic" 탭을 클릭하고, "Odometry" 를 선택하여 추가하도록 하자. 그러면 화면의 거북이의 전진방향으로 오도메트리를 나타내는 빨간색 화살표가 나타난다. 화살표 초기 값이 매우 크기 때문에 왼쪽 디스플레이창의 "Odometry"의 세부 옵션인 "Length" 의 값을 0.5로 설정하도록 하자.
```
rostopic echo /odom
```
![](../../pictures/ros/kobukirvizodomtopic.png){:height="50%" width="50%"}

이제 다시 "keyop.launch" 노드를 이용하여 가상의 거북이를 움직여보자. 좀 전과는 달리 빨간 화살표가 로봇의 궤적에 따라 표시됨을 확인할 수 있을 것이다. 이 오도메트리 정보는 이동 로봇에 있어서는 자기 자신이 어디에 있는지에 대한 정보의 기본이 되기 때문에 매우 기본적인 요소이다. 일단, 오도메트리 정보가 제대로 표시되는지만 체크하였다.

![](../../pictures/ros/kobukirvizodomtopic2.png){:height="50%" width="50%"}

5) tf 토픽 확인

거북이 구성 요소들의 상대 좌표 정보를 담은 tf의 경우 이전처럼 rostopic 명령어로 확인할 수 있지만 odom 과 마찬가지로 RViz 로 확인해보고, 계층 구조에 대해서는 rqt_tf_tree 를 통하여 살펴보도록 하자. 이번에는 RViz 의 왼쪽 하단의 "Add"를 클릭 후, 나오는 시각화 생성 화면에서 "TF" 를 선택하여 추가하도록 하자. 그러면 화면의 거북이 모델에 odom, base_footprint, gyro_link, wheel_left_link, wheel_right_link 등이 표시된다. 여기서, 다시 "keyop.launch" 노드를 이용하여 가상의 거북이를 움직여보자. 거북이가 움직이면서 wheel_left_link, wheel_right_link 의 tf 마크가 회전하는 것을 아래의 동영상처럼 확인할 수 있을 것이다.

이번에는 아래 명령어로 rqt_tf_tree 를 실행하도록 하자. 그러면 아래의 그림과 같이, 각 부분의 요소들이 tf 로 상대 위치가 변환되어 연관되어 있음을 확인할 수 있다. 이를 통해 나중에는 로봇위에 장착하는 센서들의 위치등도 표현 가능하다. 이 부분에 대한 자세한 사항은 네비게이션 강좌로 넘기기로 하겠다.
```
rosrun rqt_tf_tree rqt_tf_tree 
```

6) 가상 네비게이션

kobuki_soft 패키지에는 가상 네비게이션이 포함되어 있다. 이를 사용하기 위해서는 추가적으로 몇가지 패키지를 더 설치해 주도록 하자. 이번에 설치하는 패키지는 ROS 의 대표적인 패키지인 navigation 패키지와 유진로봇사의 맵이다. 설치 명령어는 아래와 같다.
```
sudo apt-get install ros-indigo-navigation ros-indigo-yujin-maps
```
필수 패키지를 설치하였으면, 지금까지 실행된 모든 노드를 종료하고 아래와 같이 네비게이션 데모만 실행하도록 하자. 실행하면 아래의 첨부 그림과 같이 RViz 가 자동으로 구동되며 네비게이션에 필요한 모든 환경을 구축해 두었을 것이다. 여기서 네비게이션은 SLAM 으로 지도를 작성 후에  작성된 맵 위에서 지정된 위치로 로봇을 이동시키는 것을 의미한다. 이 부분에 자세한 내용은 차후에 네비게이션 강좌에서 더 자세히 다루기로 하겠다. 일단, RViz 의 중앙 상단의 "2D Nav Goal" 을 클릭하여 지도상의 임의의 위치에 클릭 하고 드래그하여 위치와 방향을 설정하자. 그러면 로봇은 장애물을 피하고 이동하게 된다. 

단, 개발자에게 문의한 결과, RViz 환경만을 쓴 시뮬레이션이라 로봇의 Odometry와 모터의 시뮬레이션만을 목적으로 하였고 외부 정보를 센싱할 수 없어서 벽등에 충돌한다고 밝혔다. 이를 위해서는 거북이의 또다른 시뮬레이션인 stage 및 gazebo 에서 다루어 지고 있다고 한다. 이에 대한 내용은 다음 강좌에 더 자세히 다루도록 하자.
```
roslaunch kobuki_softapps nav_demo.launch
```
![](../../pictures/ros/rosrviznavigation.png){:height="50%" width="50%"}

## Kobuki의 시뮬레이션 (Gazebo)

1) Gazebo

가제보(Gazebo)는 로봇 개발에 필요한 3차원 시뮬레이션을 위한 로봇, 센서, 환경 모델 등을 지원하고 물리 엔진을 탑재하여 실제와 근사한 결과를 얻을 수 있는 3차원 시뮬레이터이다. 

로봇을 위한 2/3차원 시뮬레이터로서는 그 동안 오픈 진영에는  RoKiSim, Robo Analyzer, OpenRAVE, ARS, breve, EZPhysics, Gazebo, Khepera Simulator, Klamp't, LpzRobots, miniBloq, Moby, MORSE, OpenHRP3, Choreonoid, OpenSim, ORCASim, Robotics Toolbox for MATLAB, Simbad 3D Robot Simulator, SimRobot, Stage, STDR Simulator, UCHILSIM, v-rep, UWSim 등이 있었고 현재도 다양한 시뮬레이터들이 등장하고 있다.

또한, 상업용으로도 Actin, anyKode Marilou, Cogmation RobotSim, Microsoft Robotics Developer Studio (MRDS), SimplyCube, V-REP PRO, Visual Components, Webots, WorkCellSimulator, Workspace5 등 많은 시뮬레이터 등이 등장했고 또는 없어지기 했었다. 

현재 소개하는 Gazebo 는 최근에 나온 오픈 진영 시뮬레이터 중 가장 좋은 평가를 받고 있고, 미국 DARPA Robotics Challenge 챌린지의 공식 시뮬레이터로 선정되어 개발에 더욱 박차를 가하고 있는 상황이다. 더욱이 ROS 에서는 그 태생이 Player/Stage, Gazebo 를 기본 시뮬레이터로 사용하고 있어서 ROS와의 호완도 매우 좋다.

  - (1) 동역학 시뮬레이션: 처음에는 ODE만 지원했었지만 3.0 버전에 들어오면서 다양한 유저들의 요구에 충족하기 위하여 Bullet, Simbody, DART 등 다양한 물리 엔진을 사용하고 있다.

  - (2) 3차원 그래픽: 가제보에서는 OGRE (open-source graphics rendering engines)를 이용하여 빛, 그림자, 감촉등을 실감나게 표현하고 있다.

  - (3) 센서와 노이즈 지원: 레이저 레인지 파인더(LRF), 2/3D 카메라, Kinect, 접촉 센서, 힘-토크 센서 등을 가상으로 지원하며 센싱된 데이터에 실제 환경과 비슷하게 노이즈를 포함시킬 수 도 있다.

  - (4) 플러그인 추가 가능: 사용자는 로봇, 센서, 환경 제어 등을 스스로 플러그인 형태로 제작할 수 있도록 API를 지원한다.

  - (5) 로봇 모델: PR2, Pioneer2 DX, iRobot Create, TurtleBot 등이 이미 가제보의 모델 파일인 SDF형태로 지원되고 있으며 사용자는 자신의 로봇을 SDF 형태로 추가할 수도 있다.

  - (6) TCP/IP 데이터 전송: 시뮬레이션은 원격 서버에서도 동작 가능하며 이는 소켓 베이스의 메시지 패싱인 구글 프로토버퍼(Protobufs)를 사용하여 실현하고 있다.

  - (7) 클라우드 시뮬레이션: Gazebo를 Amazon, Softlayer, OpenStack 등에서 사용하기 위하여 CloudSim를 사용하여 클라우드 시뮬레이션을 실현하고 있다.

  - (8) 커맨드 라인 툴: 다양한 커맨드 라인 툴을 이용하여 시뮬레이션의 상태 파악 및 제어를 할 수 있다.

2) 가제보 설치

강좌를 작성하는 이 시점에서 가제보의 버전은 4.0 이다. 불과 1년전만 해도 1.9 버전 이였는데 어느새 4.0 버전까지 나왔다. 현재 버전인 4.0 버전을 테스트해본 결과 이를 설치해도 ROS Indigo 에서 따로 패키지로 지원하고 있었서 사용함에 문제가 없었다. 그러나, 일반적으로 ROS 커뮤니티에서는 2.2 버전을 추천하고 있다. 우리는 추천 사항에 따라 2.2 버전을 설치할 것이다. 나중에 4.0을 많이 사용하게 되면 그때가서 강좌를 업데이트 하도록 하겠다. 그럼 우선, 아래와 같이 로봇 모델 양식인 sdformat과 gazebo2 를 설치하자. (gazebo2를 설치하면 2.2버전이 인스톨 된다.)
```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu trusty main" > /etc/apt/sources.list.d/gazebo-latest.list'

wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

sudo apt-get update

sudo apt-get install libsdformat1 libsdformat-dev gazebo2 
```

가제보를 테스트 하기 위해 아래와 같이 실행해보자. 문제가 없다면 아래의 첨부 그림처럼 가제보가 실행됨을 확인 할 수 있을 것이다. 현재까지는 ROS 와는 상관이 없는 별도의 시뮬레이터라고 볼 수 있다.
```
gazebo
```
이제는 거북이를 가제보 시뮬레이터상에서 동작하기 위하여 관련된 패키지들을 설치해주자. 설치할 패키지로는 가제보와 ROS와의 연동을 위한 gazebo_ros_pkgs 메타 패키지와 거북이의 3차원 시뮬레이션 관련 패키지인 kobuki_desktop 메타 패키지이다.
```
sudo apt-get install ros-indigo-gazebo-ros ros-indigo-gazebo-plugins ros-indigo-kobuki-desktop
```

3) 거북이 시뮬레이션

아래의 런치파일을 실행하자. 그러면 gazebo, gazebo_gui, mobile_base_nodelet_manager, robot_state_publisher, spawn_mobile_base 노드들이 함께 실행되며 아래의 그림과 같이 거북이 가제보 화면에 나타남을 확인 할 수 있다.
```
roslaunch kobuki_gazebo kobuki_empty_world.launch
```

![](../../pictures/ros/gazebosimul.png){:height="50%" width="50%"}

위의 상태에서 아래의 원격 제어 런치 파일을 실행하면 가제보 환경 속 가상의 거북이를 제어 가능하다.
```
roslaunch kobuki_keyop keyop.launch
```
그 다음으로,  환경 모델로 기본 설정인 ground_plane 및 거북이 모델인 mobile_base 이외에 추가로 테이블 6개, 콘크리트 불럭 3개를 환경 모델로 추가하여 보도록 하자. 관련된 가제보의 world 파일은 아래의 위치에서 에서 찾아 볼 수 있다.  이를 참고하여 자신만의 환경 모델도 만들어 보도록 하자. 굳이 world 파일 직접 수정하지 않아도 가제보에서 직접 모델을 추가해도 된다. 더 자세한 설명은 나중에 가제보에 대한 별도의 특별 강좌를 만들어 보도록 하겠다.
 (world 파일 위치: /opt/ros/indigo/share/kobuki_gazebo/worlds/playground.world)
```
roslaunch kobuki_gazebo kobuki_playground.launch
```

![](../../pictures/ros/gazebosimul2.png){:height="50%" width="50%"}

가제보 환경 속의 가상 거북이 로봇은 외형만을 갖춘게 아니라 위 캡쳐의 좌측 하단의 특성 창에서 확인할 수 있는 것 처럼 각각의 몸체는 충돌을 체크할 수 있고, 위치를 계측하고, 범퍼 3개, 절벽 감지 센서, IMU 센서가 가상으로 사용 가능할 수 있도록 설정되어 있다. 이를 이용한 예제가 아래의 런치 파일이다. 이를 실행하게 되면 가상의 거북이는 테이블 위에서 랜덤하게 이동하면서 절벽을 감지하거나 벽에 부딛쳤을 경우, 그 상황을 회피하여 테이블 위에서 돌아다니게 되어 있다. 학습에 매우 좋은 예제이니 꼭 참고해 보기 바란다.
```
roslaunch kobuki_gazebo safe_random_walker_app.launch
```

## SLAM과 Navigation

2) 땔래야 땔 수 없는 SLAM과 내비게이션(navigation)

![](../../pictures/ros/slam.png){:height="70%" width="70%"}

## SLAM 실습편

2) SLAM을 위한 로봇 하드웨어

  - 2.1 제약사항
   본 강좌에서는 SLAM 중 오픈으로 공개 되어 있는 Gmapping (G. Grisetti, C. Stachniss, W, Burgard, CC BY-NC-SA 3.0))을 이용할 예정이다. 이는 몇 가지 하드웨어 제약을 가지고 있다. 일반적인 모바일 로봇의 경우에는 문제 없는 상황이지만 참고하길 바란다.

    - 1) 이동 명령
    두 축의 구동 모터로 이루어진 형태로 주로 좌/우축 바퀴가 따로 구동 가능한 차동 구동형 모바일 로봇(differential drive mobile robot) 또는, 옴니 휠 3개 이상의 구동축을 가지고 있는 전 방향 이동 로봇 (omni-wheel robot)과 같이 x, y, theta 속도 명령어로 받아 동작 가능해야 한다.

    - 2) 주행기록계 (Odometry)
    오도메트리 정보를 얻을 수 있어야 한다. 즉, 자신이 이동한 거리, 회전량을 추측 항법(dead reckoning, 데드레커닝)으로 계산 가능하거나, 이에 준하는 이동량을 구하기 위하여 IMU 센서 등의 관성 정보를 이용하여 위치 보상, 혹은, 단독으로 IMU 센서로 선속도(linear velocity)와 각속도(angular velocity)를 측정하여 로봇 스스로 자신의 위치를 계측/추청 가능해야 한다.

    - 3) 계측 센서
    SLAM과 내비게이션을 위하여 로봇은 X-Y 평면상의 장애물을 계측 가능한 LRF(Laser Range Finder), LiDAR 및 Kinect, Xtion 과 같은 Depth camera 의 경우에는 3차원 정보를 X-Y 평면상의 정보로 변환하여 사용 가능하다. 즉, 2차 평면 계측 가능 센서를 탑재하고 있어야 한다. 
    (특별히 요구되지 않는 이상 초음파 센서 및 PSD 센서 등도 고려 대상이나 본 강좌에서는 다루지 않는다. 기타 카메라를 이용한 비쥬얼 SLAM 등은 본 강좌에서 다루지 않는다.)

    - 4) 로봇 형태
    직사각형 및 원형의 로봇만을 대상으로 한다. 한 쪽 축으로 길게 나온 변형의 로봇, 방문 사이로 지나가지 못할 정도로 너무 큰 로봇, 2족 휴머노이드 로봇, 다관절 로봇, 비행 로봇은 제외한다.


  - 2.2 사용되는 로봇과 센서
  본 강좌에서는 이전 강좌에 이어서 거북이 플랫폼을 사용할 예정이다. 단, SLAM을 위하여 아래와 같이 기존 거북이 본체에 터틀봇2의 패키지로 제공하는 스택(검은색 판)과 폴(은색 스택 받침)을 2단으로 쌓았다. 1단에는 랩톱이 들어가 있으며, 그 위의 2단에는 Hokuyo LRF인 UTM-30LX, Depth camera 인 Kinect 과 Xtion 이 각각 올라간다. 이는 조금 전에 언급한 SLAM 제약 사항을 모두 만족하는 조건이다. 본 강좌에서는 LRF 센서를 이용한 아래의 좌측 그림과 같은 로봇을 이용하여 설명할 예정이다. 내비게이션 강좌가 모두 끝나면 Kinect, Xtion 등도 추가로 설명하도록 하겠다.


3) SLAM 계측 대상 환경

SLAM 이 가능한 환경은 딱히 특정짓지는 않지만, Gmapping 의 알고리즘 상 특징 요소가 매우 적은 1) 장애물 하나도 없는 정사각 형태의 환경, 2) 장애물 하나 없이 양 벽이 평행하게 길게 이어진 복도, 3) 레이저 및 적외선이 반사되지 못하는 유리창 및 4) 산란되는 거울, 5) 호수, 바닷가 등의 경우 및 사용하는 센서의 특성에 따라 장애물 정보를 취득 못하는 환경은 제외된다.

이번 강좌에서는 필자의 연구실에 마련된 실험 공간을 타겟 환경으로 정했다. 이 환경은 일반 가정집을 가정하여 침대, 책상, 테이블, 선반, 책장, 냉장고, TV, 소파 등으로 구성되어 있다. 아래에 실험 환경의 실제 사진과 3차원 정보를 첨부 사진으로 올렸으니 참고하기 바란다. 

![](../../pictures/ros/slam2.png){:height="50%" width="50%"}

4) SLAM을 위한 ROS 패키지

본 강좌에서 사용할 SLAM 관련 ROS 패키지는 kobuki 메타 패키지와 slam_gmappig 메타 패키지의 gmapping 패키지, navigation 메타 패키지의 map_server 패키지 이다. 아래와 같이 미리 모두 설치해 두도록 하자. 본 강좌는 따라하기 강좌이기에 때문에 실행 방법만을 기술할 예정이다. 각 패키지의 설명은 다음 강좌에서 매우 자세히 다루도록 하겠다. 

이번 강좌 부터는 작업 혼란을 막기 위하여, 모든 패키지를 거북이에 설치하고 진행하도록 하겠다. 패키지의 설치, 각 노드, 런치파일의 실행은 모두 거북이 본체와 연결된 랩톱에서 실행해야 한다.
```
sudo apt-get install ros-indigo-kobuki*
sudo apt-get install ros-indigo-gmapping
sudo apt-get install ros-indigo-navigation
```
센서 패키지로는 사용하는 센서에 맞도록 관련 패키지를 아래와 같이 설치하도록 하자.

1) Hokuyo LRF (URG-04LX 및 UTM-30LX 시리즈)
```
sudo apt-get install ros-indigo-urg-node 
```
2) Kinect
```
sudo apt-get install ros-indigo-openni-camera ros-indigo-openni-launch
```
3) Xtion
```
sudo apt-get install ros-indigo-openni2-camera ros-indigo-openni2-launch
```

5) SLAM 실행

현재 아래의 강좌는 거북이와 LRF 를 이용한 강좌이다. Kinect 및 Xtion 의 경우에는 추후에 추가하도록 하겠다.

  - 1) 소스 다운로드 및 컴파일
  우선, 오로카 Github 주소에서 관련 패키지를 다운로드 받는다. 그 후, 컴파일을 해준다.
    ```
    cd ~/catkin_ws/src

    git clone https://github.com/oroca/rosbook_kobuki.git

    cd ~/catkin_ws && catkin_make
    ```
  책에 맞추어 https://github.com/oroca/rosbook_kobuki.git 으로 변경됨.
  - 2) 거북이 노드 실행
  roscore 를 실행한 후, 거북이 노드를 실행한다.
    ```
    roscore

    roslaunch kobuki_node minimal.launch --screen
    ```

  - 3) kobuki_slam 실행
  kobuki_slam 패키지는 단순히 런치 파일 하나로만 구성되어 있다. 이 런치 파일은 LRF의 드라이버인 urg_node 노드, 좌표 변환을 위한 tf 를 활용한 kobuki_tf 노드, 맵 작성을 위해 slam_gmapping 노드를 포함하여 총 3개의 노드가 함께 실행된다.
    ```
    sudo chmod a+rw /dev/ttyACM0

    roslaunch kobuki_slam kobuki_slam.launch
    ```

  - 4) RViz 실행
  SLAM 도중 결과를 눈으로 확인 할 수 있도록 ROS 의 시각화툴인 RViz를 구동하도록 하자. 구동 시에 아래와 같이 옵션을 붙여주면 디스플레이 플러그인들이 처음부터 추가되어 매우 편리하다.
    ```
    rosrun rviz rviz -d `rospack find kobuki_slam`/rviz/kobuki_slam.rviz
    ```
  - 5) 토픽 데이터 저장
  6번에서 유저가 직접 로봇을 조정하며 SLAM 작업을 하게 되는데 이때에 kobuki 와 kobuki_slam 패키지에서 발행하는 /scan과 /tf 토픽을 scan_data 이라는 파일명의 bag 파일로 저장하게 된다. 나중에 이 파일을 가지고 맵을 만들 수도 있고 실험할때의 작업을 반복하지 않아도 맵핑작업시에 실험 당시의 /scan과 /tf 토픽 을 재현할 수 있다. 실험을 녹음한다고 생각하면 된다.
    ```
    rosbag record -O scan_data /scan /tf
    ```
  - 6) 로봇 조종
  아래의 명령어로 유저가 직접 로봇을 조정하며 SLAM 작업을 한다. 여기서 중요한 점은 로봇의 속도를 너무 급하게 바꾸거나 너무 빠른 속도로 전/후진, 회전하지 않도록 주의한다. 그리고, 로봇을 이동시킬 때 계측할 환경의 구석 구석을 로봇이 돌아다니며 스캔할 필요가 있다. 이 부분은 경험이 필요한 부분이니 위 SLAM 작업을 많이 해보며 경험을 쌓아보도록 하자. 
    ```
    roslaunch kobuki_keyop safe_keyop.launch
    ```

  - 7) 로봇 조종
  로봇을 이동시키면 로봇의 오도메트리, tf 정보, 센서의 스캔 정보를 기반으로 맵이 작성된다. 이는 위에서 실행한 RViz 에서 확인 가능하다. 모든 작업이 완료 되었으며 map_saver 노드를 실행하여 맵을 작성하자. 작성된 맵은 map_saver를 동작시킨 디렉토리에 저장되며 파일명은 특명히 지정해주지 않는 이상 실제 맵인 map.pgm 파일명과 맵 정보가 포함된 map.yaml 파일명으로 저장된다.
    ```
    rosrun map_server map_saver
    ```
  1번에서 7번의 과정을 통하여 맵을 작성할 수 있다. 그 과정을 아래의 그림으로 나타내며, 그 결과물인 지도를 아래에 함께 첨부한다. 위 에서 언급한 실험 환경의 맵이 제대로 작성되었음을 확인 할 수 있다.

  ![](../../pictures/ros/slam3.png){:height="50%" width="50%"}
  ![](../../pictures/ros/slam4.png){:height="50%" width="50%"}

6) 미리 준비된 bag 파일을 이용한 SLAM

아래의 내용은 거북이 및 LRF 센서 없이도 SLAM 을 접해 볼 수 있도록 위에서 녹화해둔 bag 파일로 직접 해보기로 하겠다. 우선, 본 강좌의 첨부파일을 다운로드 하자. 압축되어 있으니 이를 해제한 후 자신의 / 폴더에 넣도록 하자. 그 다음에 내용은 위에서 SLAM 실행 방법과 일치한다. 다만, rosbag 만 저장이 아닌 재생(play) 를 하여 실제 실험하는 것과 동일하게 동작 할 것이다. 다만, 여기서 주의 할점은 파라매터 중 use_sim_time 를 활성화 시켜서 현재 시간이 아닌 bag 파일이 저장되었던 시점의 시간을 이용해야 문제가 없다. 직접 해보기를 추천한다.
```
roscore

rosparam set use_sim_time true

roslaunch kobuki_slam kobuki_slam_demo.launch

rosrun rviz rviz -d `rospack find kobuki_slam`/rviz/kobuki_slam.rviz

rosbag play ./scan_data.bag

rosrun map_server map_saver
```


## SLAM 응용편

1) 강좌에 앞서서...

이전 강좌 "로봇 운영체제 강좌: SLAM 실습편"이 단순히 따라해보는 강좌였다면 이번 강좌에서는 SLAM 에서 사용되는 ROS 패키지를 자세히 살펴보고 어떻게 작성, 설정하는지에 대해서 알아 보는 응용편이라고 할 수 있다. 즉, kobuki 메타 패키지와 slam_gmappig 메타 패키지의 gmapping 패키지, navigation 메타 패키지의 map_server 패키지 를 자세히 살펴볼 예정이다. 이는 앞서 진행한 "로봇 운영체제 강좌: SLAM 실습편" 을 자신의 로봇에 적용해 볼 수 있는 응용편이라고 볼 수 있다. SLAM 자체에 대한 이론적인 설명은 "로봇 운영체제 강좌: SLAM 이론편" 에서 설명하도록 하겠다.

이 과정에서 본 강좌는 거북이라는 플랫폼과 LRF 센서를 기반으로 설명하겠지만, 이를 응용하면 특정 로봇 플랫폼, 특정 센서에 국한되지 않고 자신만의 로봇으로 SLAM 이 가능하게 될 것이다. 자신만의 로봇 플랫폼을 만들거나 거북이 로봇 플랫폼 위에 자신만의 스타일로 새로운 로봇을 구성하고 싶다면 이 강좌가 도움이 될 것이다.

2) 지도 (map)

이번 강좌에서는 ROS 커뮤니티에서 일반적으로 많이 사용되는 2차원 점유 격자 지도(OGM, Occupancy Grid Map)를 이용할 것이다. 이 전 강좌에서 얻을 수 있었던 아래의 그림과 같은 지도를 말하는 것으로 흰색은 로봇이 이동 가능한 자유 영역 (free area), 검은색은 이동 불가능한 점유 영역 (occupied area), 회색은 확인되지 않은 미지 영역 (unknown area) 으로 표현된다. 이 영역들은 0에서 255의 값으로 표현하는 그레이스케일(gray scale) 값으로 나타낸다. 이 값은 점유 상태(occupancy state)를 표현한 점유 확률(occupancy probability)을 베이즈(Bayes)정리의 사후 확률(posterior probability)을 통해 구하게 된다. 점유 확률 occ 는 occ = (255 - color_avg) / 255.0 으로 표현된다. 만약, 이미지가 24비트라면 color_avg = ( 한 셀의 그레이스케일 값 / 0xFFFFFF * 255 ) 이 된다. 이 occ 가 1에 가까울 수록 점유되었을 확률이 높아지고 0 에 가까울 수록 비 점유되었다는 것을 의미한다.

ROS 의 메시지(nav_msgs/OccupancyGrid)로 발행될 때는 이를 다시 재정의하여 점유도를 정수 [0 ~ 100] 까지로 표현하고 있다. "0"에 가까울 수록 이동 가능한 자유 영역 (free area), "100"에 가까울 수록 이동 불가능한 점유 영역 (occupied area), 그리고 "-1" 은 특별히 확인되지 않은 미지 영역 (unknown area)으로 정의하고 있다.

ROS 에서는 지도 정보를 portable graymap format 이라고 하는 .pgm 파일 형태를 저장/이용하고 있다. 또한, .yaml 을 함께 포함하고 있어서 여기에 맵의 정보를 기재한다. 예를들어, 우리가 이전 강좌에서 작성한 지도의 정보를 확인하면 아래와 같은데 image 은 파일명, resolution은 지도의 해상도로 meters / pixel 단위이다.

즉 아래의 경우, 각 픽셀은 5cm 를 의미한다. origin은 지도의 원점으로 각 숫자는 각각 x, y, yaw 를 의미한다. 즉, 위 지도의 왼쪽 하단이 x = -10미터, y = -10미터 이다. negate 은 흑/백을 반전하게 된다. 그리고, 각 필셀의 흰색/흑색의 결정은 점유 확률(occupancy probability)이 occupied_thresh 한계치를 넘으면 검은색인 이동 불가능한 점유 공간 (occupied area)으로 표현하며, free_thresh 보다 작으면 반대로 흰색인 이동 가능한 자유 공간 (free area) 으로 표현한다.
```xml
image: map.pgm
resolution: 0.050000
origin: [-10.000000, -10.000000, 0.000000]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```
[ 이전 강좌에서 작성한 map.yaml 의 내용 ]

3) SLAM 에 필요한 정보

지도에 대해 알아봤으니 이 지도 작성을 위한 재료들에 대해서 알아 보기로 하자! 우선, 각자 생각해 보기로 하는데 아래의 물음에 대답할 수 있는가?. 

"지도를 작성할 때 뭐가 필요할까?" 
답부터 말하면 제일 먼저 필요한 것은 1) 거리 값이다. "나를 중심으로 저기에 있는 소파는 2미터 떨어져 있다"라고 판단할 수 있는 거리 값을 의미한다. 이는 LRF, Depth camera 등의 센서를 이용하여 X-Y평면상을 스캔한 값이라고 볼 수 있다.

두 번째는 나의 2) 위치 값이다. "나"라고 하면 여기서는 "센서"를 말하고, 이 센서의 위치는 로봇에 고정되어 있기 때문에 로봇이 움직이면 센서도 함께 움직인다. 그러니 센서의 위치 값은 로봇의 이동량인 오도메트리(odometry)에 의존하게 된다. 이를 계산하여 위치 값으로 제공할 필요가 있다.

여기서 언급한 거리 값은 ROS 에서는 scan 이라는 이름으로 부르며, 위치 정보는 의존 관계에 따라 바뀌기 때문에 tf 라는 이름으로 부른다. (참고로 tf는 변환이라는 의미로 위치 정보 변환,  연결축 변환 등에 사용된다.) 이 scan과 tf 의 2 가지 정보를 기반으로 SLAM 실행하게 되고 우리가 원하는 지도를 작성할 수 있게 된다. 

![](../../pictures/ros/slammsg.png){:height="50%" width="50%"}

4) kobuki_slam의 기능과 처리 과정

필자는 지도를 작성하기 위하여 kobuki 노드 이외에 추가적으로 SLAM을 위하여 kobuki_slam 패키지를 만들었다. 이 패키지는 소스 파일은 없지만 slam 에 필요한 패키지를 launch 파일로 묶어서 실행하고 있다. 이 과정을 그림을 설명하면 아래의 그림과 같다.  

① kobuki_urg_node
LRF 센서를 실행하여 SLAM 에 필요한 scan 정보를 slam_gmapping 노드에게 보낸다. 이는 단 한번으로 끝나지 않고 로봇이 움직이더라도 지속적으로 scan 정보를 보내게 된다.

② kobuki_keyop
키보드 입력 값을 받아서 로봇을 조종 가능한 노드이다. 거북이 노드에게 이동 속도, 회전 속도 명령을 보낸다.

③ kobuki_node
거북이 노드는 유저의 명령을 받고 이동하게 된다. 이 때에 내부적으로는 자신의 위치를 계측/측정한 위치 정보인 odom 정보를 전송하는가 동시에 같은 이름으로 odom 의 위치 변환 정보를 tf 로 내보내며, 이와 연결된 로봇의 중앙 부분의 base_footprint 위치도 tf로 내보내게 된다.

④ kobuki_tf
kobiki_tf 에서는 센서의 위치인 base_scan 을 tf 형태로 SLAM에게 넘기기 위하여 odom → base_footprint → base_link → base_scan 의 변환을 거쳐서 내보낸다.

⑤ slam_gmapping
slam_gmapping 노드에서는 센서가 측정한 1) 거리 값 인 scan 정보와 센서의 2) 위치 값 인 tf 정보를 기반으로 지도를 작성하게 된다. 

⑥ map_saver
map_server 패키지의 map_saver 노드는 이 지도 정보를 가지고 저장 가능한 map.pgm 파일과 이에 대한 정보 파일인 map.yaml 파일을 생성하게 된다.

![](../../pictures/ros/slammsgarchitecture.png){:height="50%" width="50%"}

5) 로봇 각 부분의 상대 위치 변환 정보 (tf)

상대 위치 변환 정보를 시각화를 통하여 알아보면 아래의 그림과 같이 나타낼 수 있다. 그림이 상당히 크게 나왔는데 클릭하여 큰 원본 파일을 볼 수 있도록 하였다. 이 정보를 알고 있어야 뒤에 있는 kobuki_tf 및 kobuki_slam 패키지를 만들 수 있다. 위 그림은 "rosrun rqt_tf_tree rqt_tf_tree" 통해 확인 할 수 있는 tf 의 tree 뷰어이고, 아래의 그림은 "rosrun rviz rviz" 에서 "tf" 플러그인을 추가하면 확인한 tf 정보이다. 

즉, odom → base_footprint → base_link → base_scan 순으로 위치 정보가 연관되어 있다. 우리가 SLAM 에서는 이 모든 tf 정보가 필요하다.

![](../../pictures/ros/slamtfmsg.png){:height="50%" width="50%"}  
![](../../pictures/ros/slamtfmsg2.png){:height="50%" width="50%"}

## SLAM 이론편

2) 다양한 위치 추정(localization) 방법론 

위치 추정 방법은 로봇 공학에서 매우 중요한 연구 분야로 지금도 활발히 연구되고 있는 부분이다. 로봇의 위치 추정만 제대로 이우러 진다면 그 위치를 기반으로한 지도 작성인 SLAM같은 문제도 쉽게 해결 될 수 있을 것이다. 하지만 위치 추정은 센서 관측정보가 불확실하다는 점과 실제 우리 환경에서 동작하기 위하여 실시간성을 확보해야 한다는 점 등 많은 문제점을 가지고 있다. 이를 해결하기 위해서 다양한 위치 추정 방법이 연구되고 있다. 본 강좌에서는 Kalman filter, Particle filter, Graph, Bundle adjustment 등의 방법론에 대해서 알아 보도록 하자.

2.1 칼만 필터 (Kalman filter)

위치 추정에서 미국 나사의 아폴로 프로젝트에 실제 사용되어 유명해진 루돌프 칼만(Rudolf E. Kalman)이 개발한 칼만 필터(Kalman filter)가 많이 사용되어 왔는데, 이는 잡음(노이즈, noise)이 포함되어 있는 선형 시스템에서 대상체의 상태를 추적하는 재귀 필터를 말한다. 이는 기본적으로 베이즈 확률을 기반으로 한 것으로, 모델을 상정하고 이 모델을 이용하여 이전 상태로부터 현재 시점의 상태를 예측(Prediction)한다. 그 뒤 앞단계의 예측 값과 외부 계측기로 얻은 실제 측정 값 간의 오차를 이용하여 더 정확한 상태의 상태 값을 추정하는 보정(update) 단계를 거치게 된다. 이는 지속적으로 재귀 반복되어 정확도를 높여간다. 이 과정을 아래의 그림에서 매우 간단히 나타내었다. 참고하기 바란다.


![](../../pictures/ros/kalman.png){:height="50%" width="50%"}  

단, 칼만 필터는 선형 시스템에만 해당, 적용된다. 우리 로봇 및 센서는 대부분 비선형 시스템인 경우가 많은데 이를 위해 칼만 필터를 수정한 확장 칼만필터, EKF (Extended Kalman Filter) 가 널리 이용된다. 이 이외에도 EKF 의 정확성을 보완한 무향 칼만 필터, UKF (Unscented Kalman Filter), 속도를 개선한 Fast Kalman filter 등 많은 KF 변종이 있으며 지금도 많이 연구되고 있다. 또한, 파티클 필터와 함께 사용하는 RBPF (Rao-Blackwellized Particle Filter) 등 다른 알고리즘과 함께 사용되는 경우도 흔히 찾아 볼 수 있다.


2.2 파티클 필터(Particle filter)

파티클 필터(Particle Filter)는 물체 추적에 있어서 최근에 가장 많이 사용되고 있는 알고리즘이다. 그 대표적으로는 파티클 필터을 이용한 몬테카를로 위치추정 (Monte Carlo Localization) 가 있다. 이 전에 설명한 칼만 필터의 경우 선형 시스템과 가우시안 잡음(Gaussian Noise)가 있는 시스템의 경우에는 그 정확도가 보장되지만 그렇지 못한 경우에는 정확도가 보장되지 못하다는 문제가 있다. 우리 주변의 현실 세계의 문제는 대부분 비선형 시스템이라는 점이 문제가 된다.

로봇과 센서도 마찬가지이여서 위치 추정에 파티클 필터가 많이 사용된다. 칼만 필터가 대상체를 선형 시스템을 가정하고 선형 운동으로 파라미터를 찾아가는 해석적 방법이라고 한다면, 파티클 필터는 시행 착오(try-and-error)법을 기반으로한 시뮬레이션을 통하여 예측하는 기술으로 대상 시스템에 확률 분포로 임의로 생성된 추정값을 파티클(입자) 형태로 나타낸다고 하여서 파티클 필터라는 이름이 붙었다. 이는 SMC(Sequential Monte Carlo) 방법 또는 몬테카를로 방법이라고도 불리운다.

파티클 필터는 여타 위치 추정 알고리즘과 마찬가지로 연속적으로 들어오는 정보 중에 오차가 포함되어 있다고 가정하고 대상체의 위치를 추정하게 된다. SLAM 에서 사용할 때도 로봇의 오도메트리 값과 거리 센서를 이용한 환경 계측 값 등이 관측 값으로 사용되어 로봇의 현재 위치를 추정하게 된다. 

파티클 필터 방법에서는 위치 불확실성을 샘플이라 불리우는 파티클(입자)의 무리로 이를 묘사한다. 그 입자를 로봇의 운동 모델과 확률에 근거하여 새로운 추정 위치로 이동해 가며 실제 계측 값에 따라 각 입자에 가중치(weight)를 주면서 점점 정확한 위치로 잡음을 줄이며 추정해 나가는 과정을 거치게 된다. 여기서 각 파티클(입자)를 이용하게 되는데, particle = pose(x,y,t), weight 와 같이 각 파티클은 로봇의 추정 위치를 나타내는 임의의 작은 입자로 로봇의 x, y theta 좌표와 각 파티클의 가중치(weight) 로 표현된다. 

이 파티클 필터는 아래의 5가지 과정을 거치며 1번 초기화를 제외하고 2번 5번은 반복적으로 수행하며 로봇의 위치 값을 추정하게 된다. 그 과정은 다음과 같다. 즉, X Y 좌표 평면상에 로봇의 위치를 확률로 나태난 입자의 분포를 계측값을 기반으로 갱신해 나아가며 로봇의 위치를 추측하는 방식이다. 

자세한 파티클의 공식 관련은 로봇 공학에서 확률 관련 분야의 교과서로 불리우는 그 이름도 유명한 Sebastian Thrun (스탠포드 교수, 구글 펠로우, 유다시티 창업자)의 저서인 "Probabilistic Robotics" 이라는 책이 있다. 필자는 로봇 공학을 공부하고 싶다는 사람이 있다면 이 책을 적극 추천하는 바이다. 그리고 우리 나라 Open Robotics 분야에서 많은 활동을 하고 계시는 KITECH의 양광웅 연구원님의 블로그와 카페,  로봇 공학 관련 블로그를 운영중이신 황병훈님의 파티클 필터관련 글도 도움이 될 것이다. 그리고 앞서 이야기한 "Probabilistic Robotics" 과 관련하여 유다시티의 강좌가 있다는 것을 줄리앙님을 통해서 알게되었다. 필자도 꼭 수강하고 싶은 내용이 즐비하다. 관심 있는 사람은 "Artificial Intelligence for Robotics" 온라인 강좌를 참고하도록 하자.

1) 초기화(initialization)
전역 위치 추정(Global localization)면에서 처음에는 로봇 위치 및 방향을 알 수 없기 때문에 N개의 입자 (particle_i = pose(x_i,y_i,t_i)) 를 임의로 뿌리게 된다. 이는 가장 처음에만 수행하는 것으로 입자의 가중치는 모두 같으며(1/N) 그 합은 1이 된다. 

2) 예측(prediction)
로봇의 움직임을 기술한 시스템 모델(system model)에 기반하여 로봇의 이동량에 잡음(noise)을 포함하여 각 입자들을 이동시킨다.

3) 보정(update)
계측된 센서 정보들을 기반으로 각 입자가 존재할 확률을 계산하고, 이를 반영하여 각 입자의 가중치가 1이 되도록 가중치의 값을 갱신한다. 이 갱신후의 입자 값은 초기화에서 주어졌던 particle_i = pose(x_i,y_i,t_i), weight_i (i=1,...,N) 이 예측과 갱신을 거쳐 새로운 상태가 된다. 

4) 위치 추정(pose estimation)
N개의 모든 각 입자의 위치 (x,y,t) 와 가중치 (weight)를 곱하여 로봇의 추정 위치를 계산한다.

5) 재추출(Resampling)
새로운 입자를 생성하는 단계로 가중치가 작은 입자를 없애고 가중치가 높은 입자를 중심으로 기존의 입자의 특성인 입자의 위치정보를 물려받은 새로운 입자를 추가로 생성한다. 여기서 입자 수 N은 그대로 유지해야 한다.

![](https://image.slidesharecdn.com/particlefilterinpython-110827011808-phpapp02/95/particle-filter-tracking-in-python-5-728.jpg?cb=1330336520){:height="50%" width="50%"}
![](https://image.slidesharecdn.com/particlefilter-101002221021-phpapp01/95/particle-filter-29-728.jpg?cb=1286058328){:height="50%" width="50%"}



추가로, 파티클 필터는 샘플의 개수가 충분하다면 칼만 필터의 개선한 EKF나 UKF보다 위치 추정이 정확하지만 그 개수가 충분하지 않으면 정확하지 않을 수 있다. 이러한 부분을 해결하기 위한 접근법으로 파티클 필터와 와 칼만 필터를 동시에 사용하는 방법인 RBPF (Rao-Blackwellized Particle Filter) 기반의 SLAM도 매우 일반적으로 사용되고 있다. 궁금하면 관련 자료를 찾아봐도 좋을 듯 싶다.

2.3 Graph 및 Bundle adjustment 를 이용한 SLAM

칼만 필터와 그 확장 개념들의 유사 칸만 필터 시리즈, 그리고 파티클 필터라고해서 모든 것에 만능은 아니다 예를들어, 무인 자동차와 같이 매우 넓은 미지의 구역을 SLAM 해야하는 경우에는 그 연산량이 매우 증가하게 되는데 이 경우 연상량으로 실시간성이 떨어지게 마련이다. 이러한 문제점을 보완하기 위해서 최근에는 목적에 따라서 그래프(Graph) 기반 SLAM 방법 및 Bundle adjustment 방법이 많이 연구되고 있다. 그래프 기반 SLAM (GraphSLAM)의 경우에는 로봇과 계측한 데이터의 특징들의 위치 관계를 구속조건들로 정의하여 그래프를 작성하고, 그 그래프가 교차하는 지점에서 전체 그래프에 누적된 오차를 최소화하는 방법을 채택한 방법으로 1997년에 처음 F. Lu 와 E. Milios 에 의해 고안되어 광범위 SLAM 에서 많이 연구되고 있는 방법이다. 그리고, 카메라와 특징들의 위치를 동시에 보정하는 용도로 Bundle Adjustment 방법도 최근에 극 부상하고 있는 방법이다. 이 들의 방법에 대해서는 나중에 좀 더 자세히 다루어 보기로 한다.


3) OpenSLAM과 Gmapping

SLAM 분야는 앞서 설명하였듯이 로봇 공학에서 매우 많이 연구되고 있는 분야이다. 이러한 정보는 최신 학술지 및 학회 발표 자료를 통해 찾아 볼 수 있는데, 이 들의 연구 중 오픈소스로 공개된 부분이 상당히 많다. 이들의 정보는 OpenSLAM 이라는 그룹이 이를 모두 정리하였고, OpenSLAM.org 이라는 사이트에서 확인 할 수 있다. 우리가 꼭 방문해봐야 할 사이트라고 할 수 있다. 꼭 방문해 보길 바란다.

예를들어, 2D-I-SLSJF, CAS-Toolbox, CEKF-SLAM, DP-SLAM, EKFMonoSLAM, FLIRTLib, G2O, GMapping, GridSLAM, HOG-Man, iSAM, Linear SLAM, Max-Mixture, MTK, OpenRatSLAM, OpenSeqSLAM, ParallaxBA, Pkg. of T.Bailey, RGBDSlam, Robomap Studio, RobotVision, ro-slam, SLAM6D, SLOM, SSA2D, tinySLAM, TJTF for SLAM, TORO, TreeMap, vertigo 등 30여가지의 SLAM 소스와 관련 툴을 공개하고 있다.

우리가 전 강좌에서 사용한 gmapping 또한 이곳에 소개 되고 있고, ROS 커뮤니티에서는 이를 SLAM 에서 많이 사용하고 있다. gmapping 에 관련해서는 아래의 2가지 논문이 소개되고 있다. 하나는 ICRA 2005에서 발표된 것과 또 다른 하나는 2007년 Robotics, IEEE Transactions on 논문지에 발표된 논문이다. 아래의 그 링크를 소개하지만 IEEE 회원 여부에 따라서는 접근 못하는 경우가 있을 수도 있을 것이다. 저자는 따로 편집한 논문을 자신의 페이지에 링크 하고 있으니 이를 참고하기 바란다. 

이 논문들은 어떻게 하면 입자수를 최소한으로 줄여서 연산량을 줄이고 실시간성을 낼 수 있을까에 대한 논문으로 주요 접근 방법으로는 위에서 설명한 Rao-Blackwellized particle filter 를 사용하였다. 자세한 사항은 논문을 참조하기 바라며, 개략적인 설명은 위 파티클 필터의 설명으로 이해할 수 있을 것이다.

## 내비게이션 실습편

3) 내비게이션을 위한 ROS 패키지

본 강좌에서 사용할 내비게이션 관련 ROS 패키지는 kobuki 메타 패키지와 이전 SLAM 강좌에서 작성한 kobuki_tf 패키지 , navigation 메타 패키지의 move_base, amcl, map_server 패키지 등이 이다. 아래와 같이 미리 모두 설치해 두도록 하자. 본 강좌는 따라하기 강좌이기에 때문에 실행 방법만을 기술할 예정이다. 각 패키지의 설명은 다음 강좌에서 매우 자세히 다루도록 하겠다. 

이번 강좌 부터는 작업 혼란을 막기 위하여, 모든 패키지를 거북이에 설치하고 진행하도록 하겠다. 패키지의 설치, 각 노드, 런치파일의 실행은 모두 거북이 본체와 연결된 랩톱에서 실행해야 한다. 단, RViz 및 기타 시각화 관련해서는 데스크톱에서 실행해도 무방하다.

```
sudo apt-get install ros-indigo-kobuki*

sudo apt-get install ros-indigo-navigation
```

센서 패키지로는 사용하는 센서에 맞도록 관련 패키지를 아래와 같이 설치하도록 하자. 이번 강좌에서는 1) LRF 를 이용하여 설명하도록 하고 다른 센서에 대해서는 별도로 다루도록 하겠다.

  - 1) Hokuyo LRF (URG-04LX 및 UTM-30LX 시리즈)
    ```
    sudo apt-get install ros-indigo-urg-node 
    ```
  - 2) Kinect
    ```
    sudo apt-get install ros-indigo-openni-camera ros-indigo-openni-launch
    ```
  - 3) Xtion
    ```
    sudo apt-get install ros-indigo-openni2-camera ros-indigo-openni2-launch
    ```


4) 내비게이션 실행

  - 1) 소스 다운로드 및 컴파일

    우선, 오로카 Github 주소에서 관련 패키지를 다운로드 받는다. 그 후, 컴파일을 해준다.
    ```
    cd ~/catkin_ws/src

    git clone https://github.com/oroca/rosbook_kobuki.git

    cd ~/catkin_ws && catkin_make
    ```

  - 2) 거북이 노드 실행
    ```
    roscore 를 실행한 후, 거북이 노드를 실행한다.

    roscore

    roslaunch kobuki_node minimal.launch --screen
    ```

  - 3) kobuki_navigation 실행

  kobuki_navigation 패키지는 복수의 런치 파일로 구성되어 있다. 아래의 런치 파일을 실행하게 되면 LRF의 드라이버인 urg_node 노드, 좌표 변환을 위한 tf 를 활용한 kobuki_tf 노드, kobuki 3차원 모델 정보, 이전 작성해둔 지도를 불러오는 map_server 노드, AMCL(Adaptive Monte Carlo Localization) 노드, move_base 노드가 함께 실행된다
   
    ```
    sudo chmod a+rw /dev/ttyACM0

    roslaunch kobuki_navigation kobuki_navigation.launch
    ```

  아래의 내용은 kobuki_navigation.launch 의 각 설정 값이다. 사용하는 로봇, 센서에 따라 많은 설정값이 있다. 우선, 필자는 이전에 SLAM 강좌에서 다루었던 거북이 로봇과 LRF 를 이용하여 설정하여 보았다. 추후에 이 런치 파일을 각자 로봇에 적용하는 방법도 별도의 강좌를 통해서 설명하도록 하겠다.

  - 4) RViz 실행

  내비게이션에서 목적지 지정 명령 및 그 결과를 눈으로 확인 할 수 있도록 ROS 의 시각화툴인 RViz를 구동하도록 하자. 구동 시에 아래와 같이 옵션을 붙여주면 디스플레이 플러그인들이 처음부터 추가되어 매우 편리하다.

    ```
    rosrun rviz rviz -d `rospack find kobuki_navigation`/rviz/kobuki_nav.rviz
    ```

  이를 실행시켜주면 아래와 같은 화면을 볼 수 있을 것이다. 우측 지도에 녹색 화살표가 잔뜩 보일텐데 이는 SLAM 이론편 강좌에서 설명하였던 파티클 필터의 각 입자들이다. 나중에 다시 설명하겠지만 내비게이션 또한 파티클 필터를 이용하기 때문이다. 그 녹색 화살표의 가운데 쯤에 있는 것이 거북이 로봇임을 확인 할 수 있을 것이다.

![](../../pictures/ros/particle.png){:height="50%" width="50%"}  

  - 5) 초기 위치 추정

  제일 먼저 로봇의 초기 위치 추정 작업을 거쳐야 한다. RViz 의 상단의 메뉴바 중에 "2D Pose Estimate" 를 누르면 매우 큰 녹색 화살표가 나타날 것이다. 이를 대략적으로 로봇 중앙에 위치하여 클릭하고, 마우스 버튼을 놓치 않은 상태로 로봇이 정면 방향을 가르키도록 녹색 화살표를 드래그 한다. 이 것은 초기에 대략 적인 로봇의 위치를 추정하기 위한 일종의 명령어이다. 이 과정을 거치면 로봇은 지정된 위치(x,y) 와 방향(θ) 를 가지고 스스로 자신의 위치를 추정하여(x, y, θ) 셋팅 될 것이다. 글로 표현하기 어려우니 아래에 첨부한 동영상의 처음 부분을 참고하도록 하자.


  - 6) 목적지 설정 및 로봇 이동

  모든 준비가 완료되었으면 내비게이션 명령을 내려보자. RViz 의 상단의 메뉴바 중에 "2D Nav Goal" 를 누르면 마찬가지로 매우 큰 녹색 화살표가 나타날 것이다. 이를 로봇을 이동시키고자 하는 위치에 클릭하고, 드래그 하여 방향도 설정해주도록 하자. 그러면 로봇은 작성된 지도를 기반으로 목적지까지 장애물을 피하여 이동할 것이다. 이는 아래의 동영상으로 설명을 대체 하기로 한다.

## 내비게이션 응용편(1)

![](http://wiki.ros.org/navigation/Tutorials/RobotSetup?action=AttachFile&do=get&target=overview_tf_small.png){:height="80%" width="80%"}  

