---
title: "Tensorflow tf AdamOptimizer"
date: 2018-07-12
classes: wide
tags: tensorflow AdamOptimizer
category: tensorflow
---

[tf AdamOptimizer](https://devdocs.io/tensorflow~python/tf/train/adamoptimizer)

    apply_gradients


>     apply_gradients(
    grads_and_vars,
    global_step=None,
    name=None
    )

Apply gradients to variables.

This is the second part of minimize(). It returns an Operation that applies gradients.
### Args:

- grads_and_vars: List of (gradient, variable) pairs as returned by compute_gradients().
- global_step: Optional Variable to increment by one after the variables have been updated.
- name: Optional name for the returned operation. Default to the name passed to the Optimizer constructor.

### Returns:

An Operation that applies the specified gradients. If global_step was not None, that operation also increments global_step.
### Raises:

- TypeError: If grads_and_vars is malformed.
- ValueError: If none of the variables have gradients.
- RuntimeError: If you should use _distributed_apply() instead.
