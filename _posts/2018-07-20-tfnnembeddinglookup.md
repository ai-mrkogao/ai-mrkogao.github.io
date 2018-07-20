---
title: "Tensorflow tf nn.embedding_lookup"
date: 2018-07-20
classes: wide
tags: tensorflow nn.embedding_lookup
category: tensorflow
---

[tf.nn.embedding_lookup](https://devdocs.io/tensorflow~python/tf/nn/embedding_lookup)


### tf.nn.embedding_lookup

```python
tf.nn.embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
)
```


Defined in tensorflow/python/ops/embedding_ops.py.

See the guide: Neural Network > Embeddings

Looks up ids in a list of embedding tensors.

This function is used to perform parallel lookups on the list of tensors in params. It is a generalization of tf.gather, where params is interpreted as a partitioning of a large embedding tensor. params may be a PartitionedVariable as returned by using tf.get_variable() with a partitioner.

If len(params) > 1, each element id of ids is partitioned between the elements of params according to the partition_strategy. In all strategies, if the id space does not evenly divide the number of partitions, each of the first (max_id + 1) % len(params) partitions will be assigned one more id.

If partition_strategy is "mod", we assign each id to partition p = id % len(params). For instance, 13 ids are split across 5 partitions as: [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]

If partition_strategy is "div", we assign ids to partitions in a contiguous manner. In this case, 13 ids are split across 5 partitions as: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]

The results of the lookup are concatenated into a dense tensor. The returned tensor has shape shape(ids) + shape(params)[1:].

[stackoverflow](https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do)

```python
tf.InteractiveSession()
params = tf.constant([10,20,30,40])
ids = tf.constant([0,1,2,3])
print (tf.nn.embedding_lookup(params,ids).eval())
```

would return [10 20 30 40], because the first element (index 0) of params is 10, the second element of params (index 1) is 20, etc


