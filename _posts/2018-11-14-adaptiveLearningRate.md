---
title: "How to set adaptive learning rate for GradientDescentOptimizer?"
date: 2018-11-14
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning 
category: reinforcement learning
---


## How to set adaptive learning rate for GradientDescentOptimizer?


```python
learning_rate = tf.placeholder(tf.float32, shape=[])
# ...
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(mse)

sess = tf.Session()

# Feed different values for learning rate to each training step.
sess.run(train_step, feed_dict={learning_rate: 0.1})
sess.run(train_step, feed_dict={learning_rate: 0.1})
sess.run(train_step, feed_dict={learning_rate: 0.01})
sess.run(train_step, feed_dict={learning_rate: 0.01})
```

![How to set adaptive learning rate for GradientDescentOptimizer?](https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer)