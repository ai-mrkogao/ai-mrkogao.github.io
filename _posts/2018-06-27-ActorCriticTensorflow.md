---
title: "Actor-Critic Policy Gradient in Tensorflow"
date: 2018-06-27
classes: wide
use_math: true
tags: reinforcement_learning actor_critic policy_gradient DDPG
category: reinforcement learning
---

[refer to this link](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
## Intorduction
After Deep Q-Network became a hit,people realized that deep learning methods could be used to solve a high-dimensional problems.
one of challenges in reinforcement learning is how to deal with continuous action spaces. ***for example,robotic control, stock prediction***

Deepmind has devised a solid algorithm for solving the continuous action space problem. ***policy gradient actor-critic*** algorithm called Deep Deterministic Policy Gradients(DDPG) that is ***off-policy*** and ***model-free*** that were introduced along with Deep Q-Networks.

## Example 
In this blog, I'm introducing how to implement this algorithm using Tensorflow and tflearn and then evaluate it with OpenAI Gym on the pendulum environment.

## Policy-Gradient Methods
Policy Gradient optimizes a policy end to end by computing noisy estimates of the gradient of the expected reward of the policy and then updating the policy in the gradient direction.
PG methods have assumed a stochastic policy ![stochastic policy](../../pictures/policy_gradient/stochastic_policy.png){:height="10%" width="10%"}, which gives a probability distribution over actions. this algorithms sees lots of training examples of high rewards from the good actions and negative rewards from bad actions. 
then it can increase the probability of the good actions.

## Actor-Critic Algorithms
![actor_critic_architecture](../../pictures/policy_gradient/actor_critic_architecture.png){:height="50%" width="50%"}

the policy function is known as the actor, and the value function is referred to as the critic.
The actor produces an action given the current state of the environment, and the critic produces a TD error signal given the state and resultant reward.
If the critic is estimating the action-value function, it will also need the output of the actor. ***critic uses next state value(td target) in which is generated from current action*** . The output of the critic drives learning in both the actor and the critic.

## Off-policy Vs. On-Policy
RL algorithms which are chracterized as off-policy generally employ a separate behavior policy. the behavior policy is used to simulate a trajectories. A key benefit of this separation is that the behavior policy can operate by sampling all actions, whereas the estimation policy can be deterministic(e greedy).
On-policy algorithms directly use the policy that is being estimated to sample trajectories during the training.

## Model-free Algorithms
Model-free RL makes no effort to learn the underlying dynamics that govern how an agent interacts with the environments.
Model-free algorithms directly estimate the optimal policy or value function through algorithms such as policy interation or value iteration. This is much more computationally efficient.
But Model-free methods generally require a large number of training examples.

## DDPG(Deep Deterministic Policy Gradient)
***policy gradient actor-critic***
DDPG is a plicy gradient algorithm that uses a stochastic behavior policy for good exploration but estimates a deterministic target policy.
Policy gradient algorithms utilize a form of policy iteration;***they evaluate the policy, and then follow the policy gradient to maximize performance.***  

- DDPG is 
  - off-policy
  - uses a deterministic target policy
  - actor-critic algorithms
  - primarily uses two neural network(one for actor and one for critic)
  - these networks compute action prediction for the current state
  - generate TD error each time step
  - the input of the action network is the current state and the output is a single real value representing an action chosen from a ***continuous action space***
  - the critic's output is the estimated Q-value of the current state and the action given by the actor
  - the deterministic policy gradient theorem provides the update rule for the weights of the actor network
  - the critic network is updated from the gradient obtained from the TD error signal

- Key Characteristics
  - In general, temporally-correlated trajectories leads to the introduction enormous amounts of variance. 
  - Use replay buffer to store the experience of the agent during training, and then randomly sample experiences to use for learning in order to break up the temporal correlations ***experience reply***
  - directly updating actor and critic network with gradient from TD error causes divergence.
  - using a set of target network to generate the target for your TD error and increases stability

here are the equations for the TD target ![y_i](../../pictures/policy_gradient/y_i.png){:height="2%" width="2%"} and the losss function for the critic network:

![td_target_y_i](../../pictures/policy_gradient/td_target_y_i.png){:height="60%" width="60%"}

a minibatch of size N has been sampled from the replay buffer, with i index referring to the i th sample. tha TD target ![y_i](../../pictures/policy_gradient/y_i.png){:height="2%" width="2%"} is computed from target actor and critic network having weights.

the weights of the critic network can be updated with the gradients obtained from the loss function in Eq.2. Also, the actor network is updated with the Deterministic Policy Gradient.  
![actor_policy_gradient](../../pictures/policy_gradient/actor_policy_gradient.png){:height="80%" width="80%"}  
![actor_policy_gradient2](../../pictures/policy_gradient/actor_policy_gradient2.png){:height="10%" width="10%"}

***All you need is the gradient of the output of the critic network with respect to the actions, multiplied by the gradient of the output of the actor network with respect to the its parameters, averaged over a minibatch.***

![DDPG_theorem](../../pictures/policy_gradient/DDPG_theorem.png){:height="90%" width="90%"}

Eq.6 is exactly waht we want.