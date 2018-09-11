---
title: "33 Experiments (new training policy development part 3)"
date: 2018-09-10
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning svm lstm regression multiple_model logging quandl
category: stock
---


## New training policy
- network depth and size makes a big effects on the final result
- I think second network size should be double than the first network 
- action channel also makes some effects 


```python
net = tf.nn.dropout(net, self.keep_per)
net = tf.layers.dense(inputs = net, units=2500, kernel_initializer=init,activation=tf.nn.relu)
net = tf.nn.dropout(net, self.keep_per)
net = tf.layers.dense(inputs = net, units=h_size, kernel_initializer=init,activation=tf.nn.relu)
net = tf.nn.dropout(net, self.keep_per)


if a == 0:
	if _prevtrainstatus == '' and _status == '':

	if signal_5ma ...

	else:

if a == 1:

if a == 2 or a == 3:


```