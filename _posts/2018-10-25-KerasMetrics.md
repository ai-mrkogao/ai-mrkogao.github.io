---
title: "Keras Metrics Option"
date: 2018-10-25
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning turi machine_learning platform image caption embedding metrics
category: reinforcement learning
---

## Keras Metrics Option

[Keras Metrics](https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/)

### Keras Regression Metrics

Below is a list of the metrics that you can use in Keras on regression problems.

- Mean Squared Error: mean_squared_error, MSE or mse
- Mean Absolute Error: mean_absolute_error, MAE, mae
- Mean Absolute Percentage Error: mean_absolute_percentage_error, MAPE, mape
- Cosine Proximity: cosine_proximity, cosine

```python
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# create model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
# train model
history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.plot(history.history['cosine_proximity'])
pyplot.show()
```

### Keras Classification Metrics

Below is a list of the metrics that you can use in Keras on classification problems.

- Binary Accuracy: binary_accuracy, acc
- Categorical Accuracy: categorical_accuracy, acc
- Sparse Categorical Accuracy: sparse_categorical_accuracy
- Top k Categorical Accuracy: top_k_categorical_accuracy (requires you specify a k parameter)
- Sparse Top k Categorical Accuracy: sparse_top_k_categorical_accuracy (requires you specify a k parameter)

Accuracy is special.

```python
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# create model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train model
history = model.fit(X, y, epochs=400, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['acc'])
pyplot.show()
```


