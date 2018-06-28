---
title: "Tensorflow tflearn fully_connected"
date: 2018-06-28
classes: wide
tags: tensorflow tflearn
category: tensorflow
---

[tflearn fully_connected](http://tflearn.org/layers/core/)

> tflearn.layers.core.fully_connected (incoming, n_units, activation='linear', bias=True, weights_init='truncated_normal', bias_init='zeros', regularizer=None, weight_decay=0.001, trainable=True, restore=True, reuse=False, scope=None, name='FullyConnected')


- Input
  (2+)-D Tensor [samples, input dim]. If not 2D, input will be flatten.

- Output
  2D Tensor [samples, n_units].

- Arguments
  
    - incoming: Tensor. Incoming (2+)D Tensor.
    - n_units: int, number of units for this layer.
    - activation: str (name) or function (returning a Tensor). Activation applied to this layer (see tflearn.activations). Default: 'linear'.
    - bias: bool. If True, a bias is used.
    - weights_init: str (name) or Tensor. Weights initialization. (see tflearn.initializations) Default: 'truncated_normal'.
    - bias_init: str (name) or Tensor. Bias initialization. (see tflearn.initializations) Default: 'zeros'.
    - regularizer: str (name) or Tensor. Add a regularizer to this layer weights (see tflearn.regularizers). Default: None.
    - weight_decay: float. Regularizer decay parameter. Default: 0.001.
    - trainable: bool. If True, weights will be trainable.
    - restore: bool. If True, this layer weights will be restored when loading a model.
    - reuse: bool. If True and 'scope' is provided, this layer variables will be reused (shared).
    - scope: str. Define this layer scope (optional). A scope can be used to share variables between layers. Note that scope will override name.
    - name: A name for this layer (optional). Default: 'FullyConnected'.
