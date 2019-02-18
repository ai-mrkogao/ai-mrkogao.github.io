---
title: "String Permutation"
date: 2019-02-18
classes: wide
use_math: true
tags: python algorithm string permutation
category: algorithm
---

```python
import os,sys

def permutation(src,tar):
    if len(src) != len(tar):
        return False
    src = ''.join(sorted(src))
    tar = ''.join(sorted(tar))
    if src == tar:
        return True
    else:
        return False

src = 'abc'
tar = 'afb'
permutation(src,tar)


import os,sys

def permutation_hash(src,tar):
    if len(src) != len(tar):
        return False
    
    letters = [None]*128
    for c in src:
        letters[ord(c)] = 1
    
    for c in tar:
        if letters[ord(c)] == None:
            return False
    return True
    
src = 'abc'
tar = 'fba'
permutation_hash(src,tar)    
```

