---
title: "python api pandas to_pickle"
date: 2018-08-09
tags: python pandas to_pickle
categories: python_api
---

### pandas.to_pickle

DataFrame.to_pickle(path, compression='infer', protocol=4)[source]

Pickle (serialize) object to file.

[pandas.to_pickle](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_pickle.html)

```python
>>> original_df = pd.DataFrame({"foo": range(5), "bar": range(5, 10)})
>>> original_df
   foo  bar
0    0    5
1    1    6
2    2    7
3    3    8
4    4    9
>>> original_df.to_pickle("./dummy.pkl")

>>> unpickled_df = pd.read_pickle("./dummy.pkl")
>>> unpickled_df
   foo  bar
0    0    5
1    1    6
2    2    7
3    3    8
4    4    9

>>> import os
>>> os.remove("./dummy.pkl")

```
