---
title: "python api itertools"
date: 2018-06-25
tag: python
categories: python_api
---

### itertools API
First, let’s get the boring part out of the way:
```python
import itertools

letters = ['a', 'b', 'c', 'd', 'e', 'f']
booleans = [1, 0, 1, 0, 0, 1]
numbers = [23, 20, 44, 32, 7, 12]
decimals = [0.1, 0.7, 0.4, 0.4, 0.5]
```

chain()

chain() does exactly what you’d expect it to do: give it a list of lists/tuples/iterables and it chains them together for you. Remember making links of paper with tape as a kid? This is that, but in Python.

Let’s try it out!
```python
print (itertools.chain(letters, booleans, decimals))

>>> <itertools.chain object at 0x2c7ff0>
```

```python
print (list(itertools.chain(letters, booleans, decimals)))

>>> ['a', 'b', 'c', 'd', 'e', 'f', 1, 0, 1, 0, 0, 1, 0.1, 0.7, 0.4, 0.4, 0.5]
```

```python
print (list(itertools.chain(letters, letters[3:])))

>>> ['a', 'b', 'c', 'd', 'e', 'f', 'd', 'e', 'f']
```

count()
start a count and keep counting without an end point
```python
for x in itertools.count(10, 20):
    print (x)

for x in itertools.count(10, 20):
    print (x)
    if x > 1000:
        break
```
