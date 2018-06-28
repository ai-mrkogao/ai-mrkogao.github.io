---
title: "Tensorflow tflearn layers.normalization.batch_normalization"
date: 2018-06-28
classes: wide
tags: tensorflow tflearn
category: tensorflow
---

[tflearn layers.normalization.batch_normalization](http://tflearn.org/layers/normalization/#batch-normalization)

> tflearn.layers.normalization.batch_normalization (incoming, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002, trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization')

Normalize activations of the previous layer at each batch.

- Arguments 

```
	incoming: Tensor. Incoming Tensor.
	beta: float. Default: 0.0.
	gamma: float. Default: 1.0.
	epsilon: float. Defalut: 1e-5.
	decay: float. Default: 0.9.
	stddev: float. Standard deviation for weights initialization.
	trainable: bool. If True, weights will be trainable.
	restore: bool. If True, this layer weights will be restored when loading a model.
	reuse: bool. If True and 'scope' is provided, this layer variables will be reused (shared).
	scope: str. Define this layer scope (optional). A scope can be used to share variables between layers. Note that scope will override name.
	name: str. A name for this layer (optional).
```

- References

  Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shif. Sergey Ioffe, Christian Szegedy. 2015.

-Links

[1502.03167v3.pdf](http://arxiv.org/pdf/1502.03167v3.pdf)