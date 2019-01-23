---
title: "Tensorflow Initializer"
date: 2019-01-23
classes: wide
use_math: true
tags: reinforcement_learning actor_critic policy_gradient DDPG
category: reinforcement learning
---

## Tensorflow Initializer

[Tensor He, Xavier Init](https://adventuresinmachinelearning.com/weight-initialization-tutorial-tensorflow/)

The three arguments used in this function are:

- The factor argument, which is a multiplicative factor that is applied to the scaling. This is 1.0 for Xavier weight initialization, and 2.0 for He weight initialization
- The mode argument: this defines which is on the denominator of the variance calculation. If ‘FAN_IN’, the variance scaling is based solely on the number of inputs to the node. If ‘FAN_OUT’ it is based solely on the number of outputs. If it is ‘FAN_AVG’, it is based on an averaging calculation, i.e. Xavier initialization. For He initialization, use ‘FAN_IN’
- The uniform argument: this defines whether to use a uniform distribution or a normal distribution to sample the weights from during initialization. For both Xavier and He weight initialization, you can use a normal distribution, so set this argument to False
