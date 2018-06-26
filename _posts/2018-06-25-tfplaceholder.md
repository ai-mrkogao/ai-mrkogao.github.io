---
title: "Tensorflow placeholder"
date: 2018-06-25
classes: wide
tag: tensorflow
category: tensorflow
---


```python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)

x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  #print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.


array1 = np.array([1, 2, 3,4,5,6,7,8,9,10])
array1.shape
(10,)
np.array([[1, 2], [3, 4]]).shape
(2,2)
```
