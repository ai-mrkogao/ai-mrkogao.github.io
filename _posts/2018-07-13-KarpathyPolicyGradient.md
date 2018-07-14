---
title: "Karpathy Policy Gradient Analysis"
date: 2018-07-13
classes: wide
use_math: true
tags: reinforcement_learning karpathy policy_gradient tensorflow
category: reinforcement learning
---

## Introduction
RL research Area
- ATARI games
- Alpha Go 
- robots learning how to perform complex manipulation tasks
- etc


RL education
- Richard Sutton
- David Silver course
- John Schulmann's lectures

Four separate factors about AI
- Compute(the obvious one: Moore's Law,GPU,ASICs)
- Data(in a nice form)
- Algorithms (research and ideas, CNN,LSTM)
- Infrastructure(software -Linux,ROS,TCP/IP,Git,AWS,Tensorflow,etc)

Short history about ATARI and AlexNet , Alpha Go Development
- 2012 AlexNet envolving from 1990's ConvNets
- 2013 ATARI Deep Q learning paper is a standard implementation of Q learning with function approximation in sutton's book
- AlphaGo uses policy gradients with Monte Carlo Tree Search(MCTS)

Policy Gradients Algorithm in RL
- PG is preferred because it is end to end: there's an explicit policy 
- A principled approach that directly optimizes the expected reward

Pong Game under PG
- an image frame(210 X 160 X 3) byte array(integers from 0 to 255 giving pixel values)
- the paddle UP/DOWN(binary choice)
- the game simulator gives us reward after executing the action

### Policy Network
- The network takes the state of the game and decides the action(UP/DOWN)
- This is a stochastic policy

![PG network](../../pictures/policy_gradient/Karpathy_PG1.png){:height="50%" width="50%"}
- python implementation of above policy network
```python
h = np.dot(W1, x) # compute hidden layer neuron activations
h[h<0] = 0 # ReLU nonlinearity: threshold at zero
logp = np.dot(W2, h) # compute log probability of going up
p = 1.0 / (1.0 + np.exp(-logp)) # sigmoid function (gives probability of going up)
```
- sigmoid non-linearity makes output probability to the range [0,1]
- W1 detects various game scenarios
- W2 decides actions(UP/DOWN)
- the only problem is to find W1 and W2 

### It sounds kind of impossible
- 100,800(210x160x3) and forward our policy network(W1 and W2)
- we could repeat this training process until we get any non-zero reward ->  ***credit assignment problem***

### Supervised Learning
- In ordinary supervised learning we would feed an image to the network and get some probabilities(two classes UP and DOWN)
- out network would now be slightly more likely to predict UP when it sees a very similar image in the future
- we need correct label(UP/DOWN)  
![PG network](../../pictures/policy_gradient/Karpathy_PG2.png){:height="70%" width="70%"}

### Policy Gradients
- If we don't have correct labels, what do we do?
- The answer is Policy Gradients solution
- policy network calculated probabilities of going UP as 30%(logprob -1.2) and DOWN as 70%(logprob -0.36) and select the DOWN action 
- Finally, we get -1 reward(+1 if we won or -1 if we lost)
- so backpropagation fills the network params which makes the same input in the future not to take DOWN action
![PG network](../../pictures/policy_gradient/Karpathy_PG3.png){:height="70%" width="70%"}


### Training protocol
- how the training works in detail
- the policy network with W1,W2 and play 100 games
- 200 frames and 20,000 decisions for each game
- All that remains now is to label every decision as good or bad
- won 12 games and lost 88 games
- 200 frame * 12 won games = 2400 decisions -> positive updates
- +1.0 in the gradients -> doing backprop -> parameter updates encouraging the action
- 200 frame * 88 lost games = 17600 decisions -> negative updates
![PG network](../../pictures/policy_gradient/Karpathy_PG4.png){:height="70%" width="70%"}

 
## Deriving Policy Gradients
- Policy gradients are a special case of a more general score function gradient esimator
- ![PG network](../../pictures/policy_gradient/PG_formular1.png){:height="15%" width="15%"} :
the expectation of some scalar values score function ![PG network](../../pictures/policy_gradient/PG_fx.png){:height="5%" width="5%"} under some probability distribution ![PG network](../../pictures/policy_gradient/PG_dist.png){:height="7%" width="7%"}
- score function ![PG network](../../pictures/policy_gradient/PG_fx.png){:height="5%" width="5%"} become our ***reward function or advantage function***
- ![PG network](../../pictures/policy_gradient/PG_px.png){:height="5%" width="5%"} is our policy network
- ![PG network](../../pictures/policy_gradient/PG_px.png){:height="5%" width="5%"} is a model for ![PG network](../../pictures/policy_gradient/PG_ai.png){:height="7%" width="7%"}, giving a distribution over actions for any Image I

- ### What we must do is how to shift the distribution(through parameters ![PG network](../../pictures/policy_gradient/PG_theta.png){:height="1%" width="1%"}) to increase the scores of its samples  
- ### how do we change the network's parameters so that action samples get higher rewards
![PG network](../../pictures/policy_gradient/Karpathy_PG_eq.png){:height="70%" width="70%"}

- ![PG network](../../pictures/policy_gradient/PG_px.png){:height="5%" width="5%"} distribution have some samples x(this could be a gaussian)
- For each sample we can evaluate the score function ![PG network](../../pictures/policy_gradient/PG_fx.png){:height="5%" width="5%"} which takes the sample and gives us some scalar-valued score
- This equation is telling us how we shift the distribution(through its parameter ![PG network](../../pictures/policy_gradient/PG_theta.png){:height="1%" width="1%"})
- ![PG network](../../pictures/policy_gradient/PG_secondterm.png){:height="10%" width="10%"} is a vector
  - the gradient gives us the direction in the parameter space
  - if we were to nudge ![PG network](../../pictures/policy_gradient/PG_theta.png){:height="1%" width="1%"} in the direction of ![PG network](../../pictures/policy_gradient/PG_secondterm.png){:height="10%" width="10%"} we would see the new probability   
![PG network](../../pictures/policy_gradient/Karpathy_PG_summary.png){:height="80%" width="80%"}


## Non-differentiable computation in Neural Networks
- introduced and popularized by [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) under the name hard attention
- model processes an image with a sequence of low-resolution foveal glances
- at every iteration RNN receives a small piece of the image and sample a location to look at next
  - RNN look at position(5,30) then decide to look at (24,50) next
  - this operation is non-differentiable because we don't know what would have happened if we sampled a different location  
![PG network](../../pictures/policy_gradient/Karpathy_nondiff.png){:height="80%" width="80%"}
- ***non-differentiable computation(red) can't be backprop through***
- during training we will produce several samples and then we'll encourage samples that eventually led to good outcomes
  - the parameters involved in the blue arrows are updated with backprop as usual
  - ***the parameters involved in the red arrows are updated independently using policy gradients which encouraging samples that led to low loss***  
  ![PG network](../../pictures/policy_gradient/Karpathy_nondiff2.png){:height="80%" width="80%"}





### Reference sites
[Karpathy policy gradient blog](http://karpathy.github.io/2016/05/31/rl/)

[deterministic policy gradients from silver, deepmind](http://proceedings.mlr.press/v32/silver14.pdf)

[Guided Policy Search](http://arxiv.org/abs/1504.00702)

trajectory optimization, robot teleoperation from [Karpathy policy gradient blog](http://karpathy.github.io/2016/05/31/rl/)