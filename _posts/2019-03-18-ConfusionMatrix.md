---
title: "Confusion Matrix"
date: 2019-03-18
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning machine_learning  GAN XGboost confusion_matrix 
category: reinforcement learning
---

## Confusion Matrix  

```python
from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
# fig = plt.figure()
# cax = ax.matshow(cm)
# plt.show()

array([[2, 0, 0],
       [0, 0, 1],
       [1, 0, 2]])
```


[confusion matrix](https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix)

