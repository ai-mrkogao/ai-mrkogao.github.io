---
title: "Policy Gradient Basic"
date: 2018-07-12
classes: wide
use_math: true
tags: reinforcement_learning actor_critic policy_gradient tensorflow
category: reinforcement learning
---

## Tensorflow gradient explaination

TF uses automatic differentiation and more specifically reverse-mode auto differentiation.

- There are 3 popular methods to calculate the derivative:
  - Numerical differentiation  
  ***Numerical differentiation*** relies on the definition of the derivative: ![numeric differentiation](../../pictures/policy_gradient/numeric_diff.png){:height="20%" width="20%"}, where you put a very small h and evaluate function in two places. This is the most basic formula and on practice people use other formulas which give smaller estimation error. This way of calculating a derivative is suitable mostly if you do not know your function and can only sample it. Also it requires a lot of computation for a high-dim function.
  - Symbolic differentiation  
  ***Symbolic differentiation*** manipulates mathematical expressions. If you ever used matlab or mathematica, then you saw something like this ![numeric differentiation](../../pictures/policy_gradient/symbolic_diff.png){:height="50%" width="50%"}

Here for every math expression they know the derivative and use various rules (product rule, chain rule) to calculate the resulting derivative. Then they simplify the end expression to obtain the resulting expression.

  - Automatic differentiation => ***tensorflow gradient optimizer***   
  ***Automatic differentiation*** manipulates blocks of computer programs. A differentiator has the rules for taking the derivative of each element of a program (when you define any op in core TF, you need to register a gradient for this op). It also uses chain rule to break complex expressions into simpler ones. Here is a good example how it works in real TF programs with some explanation.
[tensorflow gradient explain](https://stackoverflow.com/questions/44342432/is-gradient-in-the-tensorflows-graph-calculated-incorrectly)  
  > ### for example below tensorflow code shows us simple policy gradient implementation
  > ![tensorflow policy gradient impl](../../pictures/policy_gradient/tensorflow_gradient.png){:height="50%" width="50%"}  

  > ### Tensorflow just need J function and optimizer function executes the gradient like below
  ![tensorflow_gradient_theory](../../pictures/policy_gradient/tensorflow_gradient_theory.png){:height="70%" width="70%"}


## Neural Network

Neural network

The core of our new agent is a neural network that decides what to do in a given situation. There are two sets of outputs – the policy itself and the value function.

Neural network architecture

It is defined very easily with Keras:
```python 
l_input = Input( batch_shape=(None, NUM_STATE) )
l_dense = Dense(16, activation='relu')(l_input)
 
out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
out_value   = Dense(1, activation='linear')(l_dense)
 
model = Model(inputs=[l_input], outputs=[out_actions, out_value])
```

***The policy output goes through softmax activation to make it correct probability distribution. The ouput for value function is linear, as we need all values to be possible.***

- - - 
- - - 

![policy gradient basic](../../pictures/policy_gradient/policy_gradient_basic1.png){:height="70%" width="70%"}

- - - 
- - - 

![policy gradient basic](../../pictures/policy_gradient/policy_gradient_basic2.png){:height="70%" width="70%"}

- - - 
- - - 

![policy gradient basic](../../pictures/policy_gradient/policy_gradient_basic3.png){:height="70%" width="70%"}





### reference links

[tensorflow differentiation explain](https://stackoverflow.com/questions/36370129/does-tensorflow-use-automatic-or-symbolic-gradients)

[tensorflow gradient explain](https://stackoverflow.com/questions/44342432/is-gradient-in-the-tensorflows-graph-calculated-incorrectly)

[stanford policy gradient tutorial](https://web.stanford.edu/class/cs20si/2017/lectures/slides_14.pdf)

[policy gradient tensorflow impl](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/)

[cartpole tensorflow impl](http://kvfrans.com/simple-algoritms-for-solving-cartpole/)

[Implementing Deep Reinforcement Learning Models with Tensorflow + OpenAI Gym](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html)

[Deep Reinforcement Learning — Policy Gradients ](https://medium.com/@gabogarza/deep-reinforcement-learning-policy-gradients-8f6df70404e6)

[policy gradient](http://christianherta.de/lehre/dataScience/machineLearning/reinforcementLearning/Policy_Gradient_Introduction.slides.php)

