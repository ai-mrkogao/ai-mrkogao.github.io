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

## Pendulum Example
Pendulum has a low dimensional state space and a single continuous action within [-2,2]
the goal is to swing up and balance the pendulum

- set up a data structure to represent your replay buffer
  - deque from python's collection library
  - the replay buffer will return a randomly chosen batch of experience when queried
  

```python
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    .....

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch    
```

- actor-critic network
  - tflearn to condense the boilerplate code


```python
import tflearn

class ActorNetwork(object):
    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

class CriticNetwork(object):
    
    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # net is (?,400)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)
        # t1 is (400,300)
        # t2 is (400,300)

        # tf.matmul(net, t1.W) is (?,300): (?,400)*(400 X 300)
        # net is (?,300)
        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # out is (?,1)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out        
```

[tflearn.input_data](../../tensorflow/tflearninputdata)
[tflearn.fullyconnected](../../tensorflow/tflearnfullyconnected)
[tflearn.layers.normalization.batch_normalization](../../tensorflow/tflearnlayernormalizationbatchnormalization)
[tflearn.activations.relu](../../tensorflow/tflearnactivationsrelu)
[tflearn.initalizations.uniform](../../tensorflow/tflearninitalizationsuniform)
[tflearn.activation](../../tensorflow/tflearnactivation)

- the actor network, the output is a tanh layer scaled to be between ![actor_bound](../../pictures/policy_gradient/actor_bound.png){:height="10%" width="10%"}. This is useful when your action space is on the real line but is bounded and closed, as is the case for the pendulum task.

- critic network takes both the state and the action as inputs; however the action input skips the first layer. This is a design decision that has experimentally worked well.

### Critic network
  - critic network has two input_data(state,action)-> inputs,action
  - inputs -> 400 fully connected layer -> batch_normalization-> relu output:net
  - relu output -> 300 fully connected layer -> t1
  - action -> 300 fully connected layer -> t2
  - net updated : activation relu( matmul(net(relu output),t1.W) + matmul(action,t2.W)+ t2.b)
  - w_init(-0.003,0.003)
  - out = fully_connected(net,1,weights_init=w_init)  
  final output is ***estimated current state-action value***

### Actor network
  - inputs : [None,self.s_dim] -> 400 fully connected layer -> batch_normalization -> relu -> 300 fully connected layer -> batch_normalization -> relu output
  - w_init(-0.003,0.003)
  - out : tanh(relu output -> self.a_dim(action dimension),weights_init=w_init)
  - scaled_out = (out * self.action_bound)  
  final output is ***action probabilities***

### Creation methods twice
- once to create the actor and critic networks that will be used for training, and again to create your target actor and critic network

- update the target network parameters like below

```python
self.network_params = tf.trainable_variables()

self.target_network_params = tf.trainable_variables()[len(self.network_params):]

# Op for periodically updating target network with online network weights
self.update_target_network_params = \
    [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
        tf.mul(self.target_network_params[i], 1. - self.tau))
        for i in range(len(self.target_network_params))]
```        
***update_target_network_params*** that will copy the parameters of the online network with a mixing factor ![tau](../../pictures/policy_gradient/tau.png){:height="2%" width="2%"}. This param is defined in both actor and critic network

### The Gradient computaion and optimization Tensorflow operations
- this is sort of replaced SGD
  ### Actor network  
    - tf.gradients() implements Deterministic Policy Gradient Equation(Eq.4)  
    ![eq_4](../../pictures/policy_gradient/eq_4.png){:height="60%" width="60%"}

[tf.gradients](../../tensorflow/tfgradients)
[tf.Adamoptimizer.apply_gradients](../../tensorflow/tfAdamoptimizerapplygradients)
    
```python
# This gradient will be provided by the critic network
self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

# Combine the gradients, dividing by the batch size to 
# account for the fact that the gradients are summed over the 
# batch by tf.gradients 
self.unnormalized_actor_gradients = tf.gradients(
    self.scaled_out, self.network_params, -self.action_gradient)
self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

# Optimization Op
self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
    apply_gradients(zip(self.actor_gradients, self.network_params))
```

### Critic network
  - This is exactly Eq.2  
  ![td_target_y_i](../../pictures/policy_gradient/td_target_y_i.png){:height="60%" width="60%"}
  - The action-value gradients at the end to pass to the policy network for gradient computation.

```python
# Network target (y_i)
# Obtained from the target networks
self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

# Define loss and optimization Op
self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

# Get the gradient of the net w.r.t. the action
self.action_grads = tf.gradients(self.out, self.action)
```



