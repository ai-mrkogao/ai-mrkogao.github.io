---
title: "python api numpy randint"
date: 2018-08-09
tags: python numpy randint
categories: python_api
---

Return random integers from low (inclusive) to high (exclusive).

[randint](http://devdocs.io/numpy~1.13/generated/numpy.random.randint)

```python
>>> np.random.randint(2, size=10)
array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
>>> np.random.randint(1, size=10)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Generate a 2 x 4 array of ints between 0 and 4, inclusive:

>>> np.random.randint(5, size=(2, 4))
array([[4, 0, 2, 1],
       [3, 2, 2, 0]])
```
