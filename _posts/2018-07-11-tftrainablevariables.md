---
title: "Tensorflow tf trainable_variables"
date: 2018-07-11
classes: wide
tags: tensorflow tflearn
category: tensorflow
---

[tf trainable_variables](https://devdocs.io/tensorflow~python/tf/trainable_variables)

> tf.trainable_variables(scope=None)

Defined in tensorflow/python/ops/variables.py.

See the guide: Variables > Variable helper functions

Returns all variables created with trainable=True.

When passed trainable=True, the Variable() constructor automatically adds new variables to the graph collection GraphKeys.TRAINABLE_VARIABLES. This convenience function returns the contents of that collection.

> Args:

   scope: (Optional.) A string. If supplied, the resulting list is filtered to include
   only items whose name attribute matches scope using re.match. Items without a name
   attribute are never returned if a scope is supplied. The choice of re.match means
   that a scope without special tokens filters by prefix.

> Returns:

A list of Variable objects.