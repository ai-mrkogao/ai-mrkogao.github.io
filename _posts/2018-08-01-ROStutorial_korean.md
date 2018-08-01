---
title: "ROS tutorial korean"
date: 2018-08-01
classes: wide
use_math: true
tags: ROS tutorial 
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
