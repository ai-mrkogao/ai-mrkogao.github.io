---
title: "RL reward tutorial review "
date: 2018-09-03
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning svm lstm regression multiple_model logging
category: stock
---


## Class Agent
```python
class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.action_size = 3 # sit, buy, sell
	...
	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))
	...


## state definition
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])

```	
