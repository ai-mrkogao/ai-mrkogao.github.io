---
title: "Policy Gradient Tutorial"
date: 2018-06-21 13:00:00 -0400
---


## Learning Goals

### Understand Actor-Critic (AC) algorithms
 - Learned Value Function
 - Learned Policy 
 
Monte Carlo Policy Gradient sill has high variance so critic estimates the action-value function
 - critic updates action-value function parameters w
 - actor updates policy parameter


cliff walk figure

> The cliff-walking task. The results are from a single run, but smoothed by averaging the reward sums from 10 successive episodes.

![cliff walk](../pictures/cliffwalk.png)

```python
from lib.envs.cliff_walking import CliffWalkingEnv 
#this example test cliff walking
from lib import plotting

#create openai gym 
env = CliffWalkingEnv()

```


