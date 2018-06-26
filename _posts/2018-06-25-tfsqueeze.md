---
title: "Tensorflow squeeze"
date: 2018-06-25
classes: wide
tag: tensorflow
category: tensorflow
---


```python
tf.squeeze(
    input,
    axis=None,
    name=None,
    squeeze_dims=None
)

#Given a tensor input, this operation returns a tensor of the same type with all
#dimensions of size 1 removed. If you don't want to remove all size 1 dimensions,
#you can   remove specific size 1 dimensions by specifying axis.

For example:
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t))  # [2, 3]

# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
```
