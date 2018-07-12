---
title: "Python map function"
date: 2018-07-12
classes: wide
tags: python map
category: python_api
---

[python map function](https://stackoverflow.com/questions/10973766/understanding-the-map-function)


map isn't particularly pythonic. I would recommend using list comprehensions instead:

### map(f, iterable)

is basically equivalent to:

> [f(x) for x in iterable]

map on its own can't do a Cartesian product, because the length of its output list is always the same as its input list. You can trivially do a Cartesian product with a list comprehension though:

> [(a, b) for a in iterable_a for b in iterable_b]

The syntax is a little confusing -- that's basically equivalent to:

```python
result = []
for a in iterable_a:
    for b in iterable_b:
        result.append((a, b))
```

