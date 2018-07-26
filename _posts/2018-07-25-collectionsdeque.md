---
title: "python api deque"
date: 2018-06-25
tag: python deque collections
categories: python_api
---

[An intro to Deque module](https://pythontips.com/2014/07/02/an-intro-to-deque-module/)

## Examples

```python
from collections import deque
d = deque()
d.append('1')
d.append('2')
d.append('3')
len(d)
print(d[0],d[-1])
>> 1 3
```

```python
d = deque('12345')
len(d)
print(d.popleft(),d.pop(),d)
>> 1 5 deque(['2', '3', '4'])
```

```python
d = deque(maxlen=30)
d = deque([1,2,3,4,5])
d.extendleft([0])
d.extend([6,7,8])
d
>> deque([0, 1, 2, 3, 4, 5, 6, 7, 8])
```

