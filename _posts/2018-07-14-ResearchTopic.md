---
title: "Research Topic"
date: 2018-07-14
classes: wide
use_math: true
tags: reinforcement_learning tensorflow topic ROS robot autonomous_driving slam 3d
category: reinforcement learning
---

## Research Topic Summarization

- ### [keras RL tutorial](https://github.com/keras-rl/keras-rl) includes below topics
  > ![keras tutorial](../../pictures/topic/keras_RL_tutorial.png){:height="80%" width="80%"} 

- ### [A3C Doom simulation](https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb)
  > This iPython notebook includes an implementation of the A3C algorithm. In it we use A3C to solve a simple 3D Doom challenge using the VizDoom engine. For more information on A3C, see the accompanying Medium post.  
  ![A3C Doom game](../../pictures/topic/A3C_doom.png){:height="80%" width="80%"}

- ### [Behavioral Cloning in autonomous driving](https://github.com/JunshengFu/driving-behavioral-cloning)
  > Coded a Deep Neural Network to Steer a Car in a game simulator. Neural Network directly predicts the steering angles from the image of front camera in the car. The training data is only collected in track 1 by manually driving two laps and the neural network learns to drive the car on different tracks.  
  ![behavioral driving](../../pictures/topic/clone_driving.png){:height="80%" width="80%"}

- ### [2D unity car simulation](https://github.com/ArztSamuel/Applying_EANNs)
  > A 2D Unity simulation in which cars learn to navigate themselves through different courses. The cars are steered by a feedforward neural network. The weights of the network are trained using a modified genetic algorithm.  
  ![2D car driving](../../pictures/topic/2d_driving.png){:height="80%" width="80%"}

- ### [Robotic and Deep Learning](https://huangying-zhan.github.io/2016/08/24/robotic-and-deep-learning.html)
  > ![robot_topic](../../pictures/topic/robot_topic.png){:height="80%" width="80%"}

- ### [Using Python programming to Play Grand Theft Auto 5](https://github.com/sentdex/pygta5)
  > Explorations of Using Python to play Grand Theft Auto 5, mainly for the purposes of creating self-driving cars and other vehicles.  
  ![Grand Auto driving](../../pictures/topic/grand_driving.png){:height="80%" width="80%"}    

- ### [Using OpenAI with ROS](http://www.theconstructsim.com/using-openai-ros/)
  > The drone training example
  In this example, we are going to train a ROS based drone to be able to go to a location of the space moving as low as possible (may be to avoid being detected), but avoiding obstacles in its way.  
  ![ROS_drone](../../pictures/topic/ROS_drone.png){:height="80%" width="80%"}

- ### [Reinforcement Learning with ROS and Gazebo](https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial7/README.md)
  > ![ROS_RL](../../pictures/topic/ROS_RL.png){:height="80%" width="80%"}

- ### [Guided Policy Search¶](http://rll.berkeley.edu/gps/)
  > This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work. It includes a complete robot controller and sensor interface for the PR2 robot via ROS, and an interface for simulated agents in Box2D and MuJoCo. Source code is available on GitHub.

  While the core functionality is fully implemented and tested, the codebase is a work in progress. See the FAQ for information on planned future additions to the code.  
  ![GPS](../../pictures/topic/Guided_policy_search.png){:height="80%" width="80%"}

- ### [Imitation Learning Driving](https://www.clipzui.com/video/54g326e4b4q5r3e4b4j5q3.html)
  > End-to-end Driving via Conditional Imitation Learning  
  ![Imitation Learning Driving](../../pictures/topic/Imitation_driving.png){:height="80%" width="80%"}  
  Conditional Imitation Learning at CARLA  
  [Carla simulator](https://github.com/carla-simulator/carla)  
  ![CARLA](../../pictures/topic/carla_simulator.png){:height="80%" width="80%"}
  Repository to store the conditional imitation learning based AI that runs on carla. The trained model is the one used on "CARLA: An Open Urban Driving Simulator" paper.
  [Conditional Imitation Learning at CARLA github](https://github.com/carla-simulator/imitation-learning)  
  [End-to-end  Driving  via  Conditional  Imitation  Learning paper](vladlen.info/papers/conditional-imitation.pdf)
  ![Imitation Learning Driving](../../pictures/topic/Imitation_driving_paper.png){:height="80%" width="80%"}

- ### [Inverse Reinforcement Learning](https://jangirrishabh.github.io/2016/07/09/virtual-car-IRL/)
  > ![Inverse RL](../../pictures/topic/InverseRL.png){:height="80%" width="80%"}

- ### [ATLAS AI training environment](https://becominghuman.ai/building-intelligence-by-learning-to-act-4b2ca0351e25)
  > ![ATLAS](../../pictures/topic/ATLAS.png){:height="80%" width="80%"}

- ### [Deep Reinforcement : Imitation Learning](https://medium.com/@parthasen/deep-reinforcement-learning-imitation-learning-5173267b22fa)  
  > ![ImitationDrivingPaper](../../pictures/topic/ImitationDrivingPaper.png){:height="80%" width="80%"}
  [Imitation pdf](http://www.yisongyue.com/courses/cs159/lectures/imitation-learning-3.pdf)
- ### [A Course in Machine Learning](http://ciml.info/)
  > ![MLcourse](../../pictures/topic/MLcourse.png){:height="30%" width="30%"}
  [MLcourse pdf](http://ciml.info/dl/v0_99/ciml-v0_99-all.pdf)

- ### [Imitation Learning by Coaching paper](https://papers.nips.cc/paper/4545-imitation-learning-by-coaching.pdf)
  > Imitation Learning has been shown to be successful in solving many challenging
	real-world problems.  Some recent approaches give strong performance guaran-
	tees by training the policy iteratively.  However, it is important to note that these
	guarantees depend on how well the policy we found can imitate the oracle on the
	training data.   When there is a substantial difference between the oracle’s abil-
	ity and the learner’s policy space, we may fail to find a policy that has low error
	on the training set.  In such cases, we propose to use a coach that demonstrates
	easy-to-learn actions for the learner and gradually approaches the oracle.  By a
	reduction of learning by demonstration to online learning, we prove that coach-
	ing can yield a lower regret bound than using the oracle. We apply our algorithm
	to cost-sensitive dynamic feature selection, a hard decision problem that consid-
	ers a user-specified accuracy-cost trade-off. Experimental results on UCI datasets
	show that our method outperforms state-of-the-art imitation learning methods in
	dynamic feature selection and two static feature selection methods  
    > ![ILcoaching](../../pictures/topic/ILcoaching.png){:height="100%" width="100%"}  
    [above picture](https://www.cs.jhu.edu/~jason/papers/he+al.nips12.poster.pdf)

- ### [Just Another DAgger Implementation](https://github.com/jj-zhu/jadagger)    
  > ![DAgger_human](../../pictures/topic/DAgger_human.png){:height="30%" width="30%"}


- ### [Imitation Learning with Dataset Aggregation (DAGGER) on Torcs Env](https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs)
  > ![DAgger_Torch](../../pictures/topic/DAgger_Torch.png){:height="50%" width="50%"}

- ### [Research tools for autonomous systems in Python ](https://github.com/spillai/pybot)
  - structure from motion and other vision

- ### [Open source Structure from Motion pipeline ](https://github.com/mapillary/OpenSfM)

- ### [Structure-from-motion-python](https://github.com/aferral/Structure-from-motion-python)

- ### [COLMAP - Structure-from-Motion and Multi-View Stereo](https://github.com/colmap/colmap)


- ### [CVPR 2017 Tutorial Geometric and Semantic 3D Reconstruction](https://people.eecs.berkeley.edu/~chaene/cvpr17tut/)
Schedule:
  
  >Srikumar:
      Feature-based and deep learning techniques for single-view problems
      Depth estimation
      Semantic segmentation
      Semantic boundary labeling
  
  
  > Jakob:
      Geometric visual-SLAM: Feature-based and Direct methods
      Sparse, Dense, Semi-dense methods
      Stereo, and RGB-D vSLAM
      Semantic SLAM
  
  > Sudipta:
      Semi-global stereo matching (SGM) and variants
      Discrete and continuous optimization in stereo
      Deep learning in stereo
      Efficient scene flow estimation from stereoscopic video
  
  > Christian:
      Volumetric Reconstruction, Depth Map Fusion
      Semantic 3D Reconstruction
      3D Object Shape Priors
  
  > Christian:
      3D Prediction using ConvNets
  
  > Various Visual-SLAM demos (if time permitted)

- ### [3D Machine Learning](https://github.com/timzhang642/3D-Machine-Learning)
  
    Table of Contents
    - [Courses](#courses)
    - [Datasets](#datasets)
      - [3D Models](#3d_models)
      - [3D Scenes](#3d_scenes)
    - [3D Pose Estimation](#pose_estimation)
    - [Single Object Classification](#single_classification)
    - [Multiple Objects Detection](#multiple_detection)
    - [Scene/Object Semantic Segmentation](#segmentation)
    - [3D Model Synthesis/Reconstruction](#3d_synthesis)
      - [Parametric Morphable Model-based methods](#3d_synthesis_model_based)
      - [Part-based Template Learning methods](#3d_synthesis_template_based)
      - [Deep Learning Methods](#3d_synthesis_dl_based)
    - [Style Transfer](#style_transfer)
    - [Scene Synthesis/Reconstruction](#scene_synthesis)
    - [Scene Understanding](#scene_understanding)

- ### [Vision Project](http://www.zeeshanzia.com/research.htm)  
![](http://www.zeeshanzia.com/research_teasers/Saleh_arxiv.png){:height="50%" width="50%"}  
![](http://www.zeeshanzia.com/research_teasers/cvpr2017.png){:height="50%" width="50%"}

![](http://www.zeeshanzia.com/research_teasers/SLAMBench.png){:height="50%" width="50%"}    
![](http://www.zeeshanzia.com/research_teasers/ijcv_teaser.png){:height="50%" width="50%"}    


## Self driving car in GTAV with Deep Learning 
[Self driving car in GTAV with Deep Learning ](https://github.com/ikergarcia1996/GTAV-Self-driving-car)

![](https://github.com/eritzyg/GTAV-Self-driving-car/raw/master/images/video1.JPG?raw=true){:height="50%" width="50%"}    

