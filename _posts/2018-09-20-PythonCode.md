---
title: "Python Code snippet"
date: 2018-09-20
classes: wide
use_math: true
tags: python stock utils keras tensorflow pandas numpy 
category: python_api
---


## Word Counting and vocabrary 

```python
word_reviews = []
all_words = []
for review in reviews_processed:
    word_reviews.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())
    
counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

vocab_to_int
>>
{'warns': 938,
 'funny': 457,
 'against': 616,
 'acne': 211,...
}

```

## Numpy save and load
```python
In  [8]: np.save('test3.npy', a)    # .npy extension is added if not given
In  [9]: d = np.load('test3.npy')
In [10]: a == d
Out[10]: array([ True,  True,  True,  True], dtype=bool)
```


## Python numpy insert
```python
alist = np.array([1,2,3,4,5])
alist = np.insert(alist,0,-1)
alist = np.append(alist,-9)
alist


>>> a = np.array([[1, 1], [2, 2], [3, 3]])
>>> a
array([[1, 1],
       [2, 2],
       [3, 3]])
>>> np.insert(a, 1, 5)
array([1, 5, 1, 2, 2, 3, 3])
>>> np.insert(a, 1, 5, axis=1)
array([[1, 5, 1],
       [2, 5, 2],
       [3, 5, 3]])

Difference between sequence and scalars:
>>>

>>> np.insert(a, [1], [[1],[2],[3]], axis=1)
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])
>>> np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),
...                np.insert(a, [1], [[1],[2],[3]], axis=1))
True

>>>

>>> b = a.flatten()
>>> b
array([1, 1, 2, 2, 3, 3])
>>> np.insert(b, [2, 2], [5, 6])
array([1, 1, 5, 6, 2, 2, 3, 3])

>>>

>>> np.insert(b, slice(2, 4), [5, 6])
array([1, 1, 5, 2, 6, 2, 3, 3])

>>>

>>> np.insert(b, [2, 2], [7.13, False]) # type casting
array([1, 1, 7, 0, 2, 2, 3, 3])

>>>

>>> x = np.arange(8).reshape(2, 4)
>>> idx = (1, 3)
>>> np.insert(x, idx, 999, axis=1)
array([[  0, 999,   1,   2, 999,   3],
       [  4, 999,   5,   6, 999,   7]])

```
