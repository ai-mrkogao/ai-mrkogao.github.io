---
title: "ROS virtualenv"
date: 2018-08-01
classes: wide
use_math: true
tags: ROS virtualenv 
category: ros
---

# Virtualenv creation
```shell
virtualenv venv --python=$(which python)
virtualenv roskineticenv --python=python3.4 or pythonj3.5

. /roskinecticenv/bin/activate

deactivate
```

```
workon tensorflow-gpu
deactivate
```

- - - 

# Setup virtualenv and virtualenvwrapper
## workon , rmvirtualenv, deactivate

[Setup virtualenv and virtualenvwrapper](https://askubuntu.com/questions/244641/how-to-set-up-and-use-a-virtual-python-environment-in-ubuntu)

First we export the WORKON_HOME variable which contains the directory in which our virtual environments are to be stored
***Let's make this ~/.virtualenvs***


```
export WORKON_HOME=~/.virtualenvs
```

now also create this directory

```
mkdir $WORKON_HOME
```
and put this export in our ~/.bashrc file so this variable gets automatically defined
```
echo "export WORKON_HOME=$WORKON_HOME" >> ~/.bashrc
```
We can also add some extra tricks like the following, which makes sure that if pip creates an extra virtual environment, it is also placed in our WORKON_HOME directory:
```
echo "export PIP_VIRTUALENV_BASE=$WORKON_HOME" >> ~/.bashrc 
```
Source ~/.bashrc to load the changes
```
source ~/.bashrc
```
Test if it works
Now we create our first virtual environment. The -p argument is optional, it is used to set the Python version to use; it can also be python3 for example.
```
mkvirtualenv -p python2.7 test
```
You will see that the environment will be set up, and your prompt now includes the name of your active environment in parentheses. Also if you now run
```
python -c "import sys; print sys.path"
```
you should see a lot of /home/user/.virtualenv/... because it now doesn't use your system site-packages.
You can deactivate your environment by running
```
deactivate
```
and if you want to work on it again, simply type
```
workon test
```
Finally, if you want to delete your environment, type
```
rmvirtualenv test
```
- - - 

# Installing Virtualenv on Ubuntu for Tensorflow
[Installing Virtualenv on Ubuntu for Tensorflow](http://www.pradeepadiga.me/blog/2017/03/24/installing-virtualenv-on-ubuntu-for-tensorflow/)

```
mkvirtualenv myfirstproject

workon mysecondproject

deactivate
```

- - -

# Installation guide for ROS-Kinetic with Python 3.5 on Ubuntu 16.04
[Installation guide for ROS-Kinetic with Python 3.5 on Ubuntu 16.04](https://stackoverflow.com/questions/49758578/installation-guide-for-ros-kinetic-with-python-3-5-on-ubuntu-16-04)

1) Installed ROS-Kinetic-desktop-full

2) pip3 install rospkg catkin_pkg

3) export PYTHONPATH = /usr/local/lib/python3.5/dist-packages
```
sudo apt-get install python3-yaml  # you'll also need this
sudo pip3 install rospkg catkin_pkg
or
pip3 install --user rospkg catkin_pkg
```

- - -

# Ubuntu install of ROS Kinetic
[Ubuntu install of ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)

- - -

# How do you install ROS on Ubuntu using Python 3
[How do you install ROS on Ubuntu using Python 3](https://www.reddit.com/r/robotics/comments/3d48au/how_do_you_install_ros_on_ubuntu_using_python_3/)

- - - 

# ImportError: No module named rospkg (python3) 
[ImportError: No module named rospkg (python3) ](https://answers.ros.org/question/245967/importerror-no-module-named-rospkg-python3-solved/)

```
sudo apt-get install python-catkin-pkg
sudo apt-get install python3-catkin-pkg-modules

sudo apt-get install python3-catkin-pkg-modules
sudo apt-get install python3-rospkg-modules
```

# How To Solve ” sub process usr bin dpkg returned an error code 1″ Error?
[How To Solve ” sub process usr bin dpkg returned an error code 1″ Error?](https://www.poftut.com/solve-sub-process-usr-bin-dpkg-returned-error-code-1-error/)

```
sudo dpkg --configure -a
sudo apt-get install -f
sudo apt-get autoremove
cat /etc/apt/sources.list
```

# Opencv install
[OpenCV install](https://askubuntu.com/questions/783956/how-to-install-opencv-3-1-for-python-3-5-on-ubuntu-16-04-lts)

[OpenCV install](http://www.techgazet.com/install-opencv/)

[Opencv 3 with python3](https://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/)

# Installing OpenCV in Ubuntu for Python 3
[Installing OpenCV in Ubuntu for Python 3](http://cyaninfinite.com/tutorials/installing-opencv-in-ubuntu-for-python-3/)


# Gazebo install
[Gazabo install](http://gazebosim.org/tutorials?tut=install_ubuntu)

# How to remove system dependency packages for a ROS package?
[How to remove system dependency packages for a ROS package?](https://askubuntu.com/questions/885734/how-to-remove-system-dependency-packages-for-a-ros-package)

```
sudo apt-get purge ros-*

sudo apt-get purge python-ros*

sudo apt-get autoremove
```
