---
title: "Tensorflow tflearn initializations uniform"
date: 2018-06-28
classes: wide
tags: tensorflow tflearn
category: tensorflow
---

[tflearn.initializations.uniform](http://tflearn.org/initializations/#uniform)


### Uniform

>tflearn.initializations.uniform (shape=None, minval=0, maxval=None, dtype=tf.float32, seed=None)

Initialization with random values from a uniform distribution.

The generated values follow a uniform distribution in the range [minval, maxval). The lower bound minval is included in the range, while the upper bound maxval is excluded.

For floats, the default range is [0, 1). For ints, at least maxval must be specified explicitly.

In the integer case, the random integers are slightly biased unless maxval - minval is an exact power of two. The bias is small for values of maxval - minval significantly smaller than the range of the output (either 2**32 or 2**64).

- Arguments
```
    shape: List of int. A shape to initialize a Tensor (optional).
    dtype: The tensor data type. Only float are supported.
    seed: int. Used to create a random seed for the distribution.
```
- Returns
  The Initializer, or an initialized Tensor if shape is specified.