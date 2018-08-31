---
title: "27 Experiments (LSTM prediction training with result signal from targetDQN signalDQN)"
date: 2018-08-31
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning svm lstm regression 
category: stock
---

## LSTM prediction training
- actions signals from traget DQN


## Run multiple pre-trained Tensorflow nets at the same time
[Run multiple pre-trained Tensorflow nets at the same time](https://stackoverflow.com/questions/39175945/run-multiple-pre-trained-tensorflow-nets-at-the-same-time)


```python

with tf.Graph().as_default() as mimic_graph:
    q_net = mimicDQN(sess, INPUT_SIZE, OUTPUT_SIZE)
    target_net = mimicDQN(sess, INPUT_SIZE, OUTPUT_SIZE)

with tf.Graph().as_default() as signal_graph:
    signal_net = targetDQN(sess, INPUT_SIZE, OUTPUT_SIZE)

with tf.Session(graph=mimic_graph) as sess:    
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables,tau)

    sess.run(init)

    ckpt_dir = target_net.logs_dir
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        target_net.saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    updateTarget(targetOps,sess)       
    print('load mimic network')

with tf.Session(graph=signal_graph) as sess:         
    # load signal net
    ckpt_sig_dir = signal_net.logs_dir
    ckpt_sig = tf.train.get_checkpoint_state(ckpt_sig_dir)
    if ckpt_sig and ckpt_sig.model_checkpoint_path:
        print(ckpt_sig.model_checkpoint_path)
        signal_net.saver.restore(sess, ckpt_sig.model_checkpoint_path) # restore all variables

    print('load signal network')     

```

```python
graph1 = Graph()
with graph1.as_default():
    session1 = Session()
    with session1.as_default():
        with open('model1_arch.json') as arch_file:
            model1 = model_from_json(arch_file.read())
        model1.load_weights('model1_weights.h5')
        # K.get_session() is session1

# do the same for graph2, session2, model2

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    cnn1 = CNN(..., restore_file='snapshot-model1-10000',..........) 
with g2.as_default():
    cnn2 = CNN(..., restore_file='snapshot-model2-10000',..........)
```


```python
# Build a graph containing `net1`.
with tf.Graph().as_default() as net1_graph:
  net1 = CreateAlexNet()
  saver1 = tf.train.Saver(...)
sess1 = tf.Session(graph=net1_graph)
saver1.restore(sess1, 'epoch_10.ckpt')

# Build a separate graph containing `net2`.
with tf.Graph().as_default() as net2_graph:
  net2 = CreateAlexNet()
  saver2 = tf.train.Saver(...)
sess2 = tf.Session(graph=net1_graph)
saver2.restore(sess2, 'epoch_50.ckpt')



with tf.name_scope("net1"):
  net1 = CreateAlexNet()
with tf.name_scope("net2"):
  net2 = CreateAlexNet()

# Strip off the "net1/" prefix to get the names of the variables in the checkpoint.
net1_varlist = {v.name.lstrip("net1/"): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="net1/")}
net1_saver = tf.train.Saver(var_list=net1_varlist)

# Strip off the "net2/" prefix to get the names of the variables in the checkpoint.
net2_varlist = {v.name.lstrip("net2/"): v
                for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="net2/")}
net2_saver = tf.train.Saver(var_list=net2_varlist)

# ...
net1_saver.restore(sess, "epoch_10.ckpt")
net2_saver.restore(sess, "epoch_50.ckpt")

```

```python
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default() as g:
    with g.name_scope( "g1" ) as g1_scope:
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul( matrix1, matrix2, name = "product")

tf.reset_default_graph()

g2 = tf.Graph()
with g2.as_default() as g:
    with g.name_scope( "g2" ) as g2_scope:
        matrix1 = tf.constant([[4., 4.]])
        matrix2 = tf.constant([[5.],[5.]])
        product = tf.matmul( matrix1, matrix2, name = "product" )

tf.reset_default_graph()

use_g1 = False

if ( use_g1 ):
    g = g1
    scope = g1_scope
else:
    g = g2
    scope = g2_scope

with tf.Session( graph = g ) as sess:
    tf.initialize_all_variables()
    result = sess.run( sess.graph.get_tensor_by_name( scope + "product:0" ) )
    print( result )
```

[graph and session](http://easy-tensorflow.com/tf-tutorials/basics/graph-and-session)    

![](http://easy-tensorflow.com/files/1_1.gif)

## graph

![](http://easy-tensorflow.com/files/1_2.png)

```python
import tensorflow as tf
x = 2
y = 3
add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')
useless_op = tf.multiply(x, add_op, name='Useless')

with tf.Session() as sess:
    pow_out, useless_out = sess.run([pow_op, useless_op])
 
```
