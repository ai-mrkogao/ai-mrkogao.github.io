---
title: "An introduction to TensorFlow queuing and threading"
date: 2018-12-11
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning machine_learning threading
category: reinforcement learning
---

[TensorFlow queuing and threads – introductory concepts](http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/)
[Parallel threads with TensorFlow Dataset API and flat_map](https://stackoverflow.com/questions/47411383/parallel-threads-with-tensorflow-dataset-api-and-flat-map)
[code Multi-Threading-mnist-classifier](https://github.com/andrewliao11/Tensorflow-Multi-Threading-Classifier)



## TensorFlow queuing and threads – introductory concepts

- One of the great things about Tensorflow is its ability to handle multiple threads and therefore allow asynchronous operations
- This functionality is especially handy
- The particular queuing operations/objects 
- Tensorflow Dataset API

- our CPU will get stuck waiting for the completion of a single task..
- Tensorflow has released a performance guide when putting data to our training processes
- Their method of threading is called Queuing

```python
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```
- Here data is fed into the final training operation via the feed_dict argument
- They are data storage objects which can be loaded and deloaded which information asynchronously using threads
- This allows us to stream data into our training algorithms more seamlessly

## FIFOQueue 
- first, I created a random normal tensor of size 3, I created a printing operation

```python
dummy_input = tf.random_normal([3], mean=0, stddev=1)
dummy_input = tf.Print(dummy_input, data=[dummy_input],
                           message='New dummy inputs have been created: ', summarize=6)
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
enqueue_op = q.enqueue_many(dummy_input)
data = q.dequeue()
data = tf.Print(data, data=[q.size()], message='This is how many items are left in q: ')
# create a fake graph that we can call upon
fg = data + 1
```
- set up FIFOQueue with capacity= 3 
- I enqueue all three values of the random tensor in the enqueue_op

```python
with tf.Session() as sess:
    # first load up the queue
    sess.run(enqueue_op)
    # now dequeue a few times, and we should see the number of items
    # in the queue decrease
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)
    # by this stage the queue will be emtpy, if we run the next time, the queue
    # will block waiting for new data
    sess.run(fg)
    # this will never print:
    print("We're here!")
```

- running the enqueue_op which loads up our queue to capacity
- the queue will be empty(dequeu)

```python
New dummy inputs have been created: [0.73847228 0.086355612 0.56138796]
This is how many items are left in q: [3]
This is how many items are left in q: [2]
This is how many items are left in q: [1]
```

## QueueRunners and the Coordinator
- A QueueRunner will control the asynchronous execution of enqueue operations
- it can create multiple threads of enqueue operations
- all of which will handle in an asynchronous 

```python
dummy_input = tf.random_normal([5], mean=0, stddev=1)
dummy_input = tf.Print(dummy_input, data=[dummy_input],
                           message='New dummy inputs have been created: ', summarize=6)
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
enqueue_op = q.enqueue_many(dummy_input)
# now setup a queue runner to handle enqueue_op outside of the main thread asynchronously
qr = tf.train.QueueRunner(q, [enqueue_op] * 1)
tf.train.add_queue_runner(qr)

data = q.dequeue()
data = tf.Print(data, data=[q.size(), data], message='This is how many items are left in q: ')
# create a fake graph that we can call upon
fg = data + 1
```

- qr = tf.train.QueueRunner(q, [enqueue_op] * 1)
- the first argument in this definition is the queue we want to run
- the next argument is a list argument
- ***this specifies how many enqueue operation threads we want to create ***
- below create 10 threads 

```python
qr = tf.train.QueueRunner(q, [enqueue_op] * 10)
```

- A coordinator object helps to make sure that all the threads we create stop together
- this is important at any point in our program where we want to bring all the multiple threads together and rejoin the main thread
- it is also important if an exception occurs on one of any threads we want this exception broadcast to all of the threads so they all stop

```python
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # now dequeue a few times, and we should see the number of items
    # in the queue decrease
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)
    # previously the main thread blocked / hung at this point, as it was waiting
    # for the queue to be filled.  However, it won't this time around, as we
    # now have a queue runner on another thread making sure the queue is
    # filled asynchronously
    sess.run(fg)
    sess.run(fg)
    sess.run(fg)
    # this will print, but not necessarily after the 6th call of sess.run(fg)
    # due to the asynchronous operations
    print("We're here!")
    # we have to request all threads now stop, then we can join the queue runner
    # thread back to the main thread and finish up
    coord.request_stop()
    coord.join(threads)
```

```python
New dummy inputs have been created: [-0.81459045 -1.9739552 -0.9398123 1.0848273 1.0323733]
This is how many items are left in q: [0][-0.81459045]
This is how many items are left in q: [3][-1.9739552]
New dummy inputs have been created: [-0.03232909 -0.34122062 0.85883951 -0.95554483 1.1082178]
This is how many items are left in q: [3][-0.9398123]
We're here!
This is how many items are left in q: [3][1.0848273]
This is how many items are left in q: [3][1.0323733]
This is how many items are left in q: [3][-0.03232909]
```

## A more practical example – reading the CIFAR-10 dataset


1. Create a list of filenames which hold the CIFAR-10 data
2. Create a FIFOQueue to hold the randomly shuffled filenames, and associated enqueuing
3. Dequeue files and extract image data
4. perform image processing
5. Enqueue processed image data into a RandomShuffleQueue
6. Dequeue data batches for classifier training (the classifier training won’t be covered in this tutorial – that’s for a future post)

```python
def cifar_shuffle_batch():
    batch_size = 128
    num_threads = 16
    # create a list of all our filenames
    filename_list = [data_path + 'data_batch_{}.bin'.format(i + 1) for i in range(5)]
    # create a filename queue
    file_q = cifar_filename_queue(filename_list)
    # read the data - this contains a FixedLengthRecordReader object which handles the
    # de-queueing of the files.  It returns a processed image and label, with shapes
    # ready for a convolutional neural network
    image, label = read_data(file_q)
    # setup minimum number of examples that can remain in the queue after dequeuing before blocking
    # occurs (i.e. enqueuing is forced) - the higher the number the better the mixing but
    # longer initial load time
    min_after_dequeue = 10000
    # setup the capacity of the queue - this is based on recommendations by TensorFlow to ensure
    # good mixing
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    image_batch, label_batch = cifar_shuffle_queue_batch(image, label, batch_size, num_threads)
    # now run the training
    cifar_run(image_batch, label_batch)
```

```python
def cifar_filename_queue(filename_list):
    # convert the list to a tensor
    string_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
    # randomize the tensor
    tf.random_shuffle(string_tensor)
    # create the queue
    fq = tf.FIFOQueue(capacity=10, dtypes=tf.string)
    # create our enqueue_op for this q
    fq_enqueue_op = fq.enqueue_many([string_tensor])
    # create a QueueRunner and add to queue runner list
    # we only need one thread for this simple queue
    tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_enqueue_op] * 1))
    return fq
```

```python
def cifar_shuffle_queue_batch(image, label, batch_size, capacity, min_after_dequeue, threads):
    tensor_list = [image, label]
    dtypes = [tf.float32, tf.int32]
    shapes = [image.get_shape(), label.get_shape()]
    q = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue,
                              dtypes=dtypes, shapes=shapes)
    enqueue_op = q.enqueue(tensor_list)
    # add to the queue runner
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * threads))
    # now extract the batch
    image_batch, label_batch = q.dequeue_many(batch_size)
    return image_batch, label_batch
```

```python
def cifar_run(image, label):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            image_batch, label_batch = sess.run([image, label])
            print(image_batch.shape, label_batch.shape)

        coord.request_stop()
        coord.join(threads)
```

        

