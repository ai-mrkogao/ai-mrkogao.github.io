---
title: "27 Experiments (LSTM prediction training with result signal from targetDQN signalDQN)"
date: 2018-08-31
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning svm lstm regression multiple_model logging
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

- The checkpoint file is just a bookkeeping file that you can use in combination of high-level helper for loading different time saved chkp files.
- The .meta file holds the compressed Protobufs graph of your model and all the metadata associated (collections, learning rate, operations, etc.)
- The .index file holds an immutable key-value table linking a serialised tensor name and where to find its data in the chkp.data files
- The .data files hold the data (weights) itself (this one is usually quite big in size). There can be many data files because they can be sharded and/or created on multiple timestep while training.
- Finally, the events file store everything you need to visualise your model and all the data measured while you were training using summaries. This has nothing to do with saving/restoring your models itself.

## Importing TensorFlow Model
```python
import tensorflow as tf
    ### Linear Regression ###
    # Input placeholders
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    # Model parameters
    W1 = tf.Variable([0.1], tf.float32)
    W2 = tf.Variable([0.1], tf.float32)
    W3 = tf.Variable([0.1], tf.float32)
    b = tf.Variable([0.1], tf.float32)
    
    # Output
    linear_model = tf.identity(W1 * x + W2 * x**2 + W3 * x**3 + b,
                               name='activation_opt')
    
    # Loss
    loss = tf.reduce_sum(tf.square(linear_model - y), name='loss')
    # Optimizer and training step
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss, name='train_step')
    
    # Remember output operation for later aplication
    # Adding it to a collections for easy acces
    # This is not required if you NAME your output operation
    tf.add_to_collection("activation", linear_model)
    
    ## Start the session ##
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #  CREATE SAVER
    saver = tf.train.Saver()
    
    # Training loop
    for i in range(10000):
        sess.run(train, {x: data, y: expected})
        if i % 1000 == 0:
            # You can also save checkpoints using global_step variable
            saver.save(sess, "models/model_name", global_step=i)
    
    # SAVE TensorFlow graph into path models/model_name
    saver.save(sess, "models/model_name")
```

## Multiple Models (Graphs)
```python
import tensorflow as tf
    
class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.activation = tf.get_collection('activation')[0]
              # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})
      
      
### Using the class ###
data = 50         # random data
model = ImportGraph('models/model_name')
result = model.run(data)
print(result)
```

```python
class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)

            saver.restore(self.sess, loc)

            
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.activation = tf.get_collection('activation')[0]
              # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})



```


## tensorflow graph information search

```python

graph = tf.Graph()
sess = tf.Session(graph=self.graph)
loc = './26Experiments/26Experiments.model-46903'
with graph.as_default():
    # Import saved model from location 'loc' into local graph
    saver = tf.train.import_meta_graph(loc + '.meta',
                                       clear_devices=True)

    saver.restore(self.sess, loc)

    for op in tf.get_default_graph().get_operations():
        print str(op.name) 

```



## tensorflow loggin level
```python
tf.logging.set_verbosity(tf.logging.ERROR)
```
