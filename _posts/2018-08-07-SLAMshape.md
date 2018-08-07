---
title: "Monocular Reconstruction of Vehicles: Combining SLAM with Shape Priors"
date: 2018-08-07
classes: wide
use_math: true
tags: slam monocular vision
category: slam
---


# Monocular Reconstruction of Vehicles: Combining SLAM with Shape Priors
- [Monocular Reconstruction of Vehicles: Combining SLAM with Shape Priors](http://robotics.iiit.ac.in/people/falak.chhaya/Monocular_Reconstruction_of_Vehicles.html)

- ![](http://www.zeeshanzia.com/research_teasers/icra2016_iiit.png){:height="80%" width="80%"}

- [Monocular Reconstruction of Vehicles: Combining SLAM with Shape Priors Paper](http://www.zeeshanzia.com/pdf_files/chayya16icra.pdf)

## Abstract
- Current  approaches  leverage  two  kinds  of  information  to deal  with  the  
  vehicle  detection  and  tracking  problem
  - (1) 3D representations  (eg.  wireframe  models  or  voxel  based  or  CAD
        models) for diverse vehicle skeletal structures learnt from data,and
  - (2) classifiers trained to detect vehicles or vehicle parts in single  images  
        built  on  top  of  a  basic  feature  extraction  step
- First, we extend detection to a multiple view setting
- Secondly, we  can  also  leverage  3D information from the scene generated using a 
  unique structure from motion algorithm. 

## Introduction
- In  the  present  paper,  we  tightly  integrate  these  deformable 3D object models 
  with state-of-the-art multibody SfM methods,
- Approximating the visible surfaces of a  vehicle  by  planar  segments,  supported  
  by  discriminative part  detectors  allows  us  to  obtain  more  stable  and  
  accurate 3D  reconstruction  of  moving  objects  as  compared  to  state-of-the-art SLAM pipelines

- Summarily, we list the contributions of the present paper in the following

1. We propose a novel piece-wise planar approximation to vehicle surfaces and use it 
   for robust camera trajectory estimation.  The  object  presents  itself  as  a  plane  to the  moving  camera.  By  segmenting  the  car  into  its constituent  planes  by  RANSAC  with  Homography  as the  model  we  obtain  superior  reconstruction  of  the moving object
2. We   extend   the   single-view   deformable   wireframe model fitting to 
   multiple   views, which  stabilizes  the  estimation  of  object  location  and shape
- ![](../../pictures/slamshape/systemoverview.png){:height="80%" width="80%"}

3. We  experimentally  demonstrate  improvements  in  3D shape estimation and 
   localization on several sequences in KITTI dataset [13] resulting from the the 
   tight integration between SfM cues and object shape modeling

   



