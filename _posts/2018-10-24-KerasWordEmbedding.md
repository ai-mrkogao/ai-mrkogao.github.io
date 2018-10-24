---
title: "Keras Word Embedding"
date: 2018-10-24
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning turi machine_learning platform image caption embedding
category: reinforcement learning
---

## Keras Word Embedding Tutorial

- Keras Embedding Layer

- Keras offers an Embedding layer that can be used for neural networks on text data.

- It requires that the input data be integer encoded, so that each word is represented by a unique integer. This data preparation step can be performed using the Tokenizer API also provided with Keras.

For example, below we define an Embedding layer with a vocabulary of 200 (e.g. integer encoded words from 0 to 199, inclusive), a vector space of 32 dimensions in which words will be embedded, and input documents that have 50 words each.

```python
e = Embedding(200, 32, input_length=50)
```

```python
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

[Keras word embedding](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)


## One-Hot layer in Keras's Sequential API

```python
from keras.layers import Lambda
# We will use `one_hot` as implemented by one of the backends
from keras import backend as K

def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))

X = np.array([
    [5, 2, 4, 25, 17], # Instance 1
    [15, 54, 13, 2, 98] # Instance 2
])

print(X.shape) # prints (2, 5)


model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE,
                    output_dim=EMBEDDING_SIZE,
                    input_length=MAX_SEQUENCE_LENGTH))

model.compile(loss='mse', optimizer='sgd')
print(model.predict(X, batch_size=32).shape)  # prints (2, 5, 25)

model = Sequential()
model.add(OneHot(input_dim=VOCAB_SIZE,
                         input_length=MAX_SEQUENCE_LENGTH))

model.compile(loss='mse', optimizer='sgd')
print(model.predict(X, batch_size=32).shape) # prints (2, 5, 100)

model = Sequential()
model.add(OneHot(input_dim=VOCAB_SIZE,
                 input_length=MAX_SEQUENCE_LENGTH))
model.add(TimeDistributed(Dense(EMBEDDING_SIZE)))

model.compile(loss='mse', optimizer='sgd')
print(model.predict(X, batch_size=32).shape) # prints (2, 5, 25)    

```


## Tensorflow RNN tutorial

```python

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# Create input data
batch_size=3
sequence_length=5
input_dim=3

x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)  # batch, sequence_length, input_dim

with tf.variable_scope('generated_data') as scope:
    # One cell RNN input_dim (3) -> output_dim (5). sequence: 5, batch: 3
    cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
    print('cell.output_size {}, cell.state_size {}'.format(cell.output_size, cell.state_size))
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    print('outputs.shape {}'.format(outputs.shape))
    pp.pprint(outputs.eval())
```


```python
# Create input data
batch_size=3
sequence_length=5
input_dim=3

x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)  # batch, sequence_length, input_dim
print('x_data {}'.format(x_data.shape))

# flattern based softmax

hidden_size=3
sequence_length=5
batch_size=3
num_classes=5

pp.pprint('x_data {} {}'.format(x_data.shape,x_data)) # hidden_size=3, sequence_length=4, batch_size=2
x_data = x_data.reshape(-1, hidden_size)
pp.pprint('hidden_size {} x_data {} {}'.format(hidden_size,x_data.shape,x_data))

softmax_w = np.arange(15, dtype=np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data, softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes) # batch, seq, class
pp.pprint('output {} {}'.format(outputs.shape,outputs))


# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())


```


```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
sgd = opt.SGD(lr=0.005, decay=1e-6, momentum=0., nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

```