---
title: "MergeSort"
date: 2019-02-12
classes: wide
use_math: true
tags: python algorithm 
category: algorithm
---


[merge sort code](http://interactivepython.org/courselib/static/pythonds/SortSearch/TheMergeSort.html)
[merge sort code](https://pythonandr.com/2015/07/05/the-merge-sort-python-code/)


```python

def merge(a,b):
    """ Function to merge two arrays """
    c = []
    while len(a) != 0 and len(b) != 0:
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])
    if len(a) == 0:
        c += b
    else:
        c += a
    return c

# Code for merge sort

def mergesort(x):
    """ Function to sort an array using merge sort algorithm """
    if len(x) == 0 or len(x) == 1:
        return x
    else:
        middle = int(len(x)/2)
        a = mergesort(x[:middle])
        b = mergesort(x[middle:])
        return merge(a,b)

    
alist = [10,3,6,7,1,2,5,4,8,9]
mergesort(alist)

```