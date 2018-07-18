---
title: "python api keras fit_generator"
date: 2018-07-16
tags: python keras fit_generator
categories: keras
---

### fit_generator

```python
fit_generator(self, generator, steps_per_epoch=None, epochs=1, \
	verbose=1,callbacks=None, validation_data=None, validation_steps=None, \
	class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, \
	shuffle=True, initial_epoch=0)
```

Trains the model on data generated batch-by-batch by a Python generator (or an instance of Sequence).

The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.

- Arguments

    - generator: A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing. The output of the generator must be either
        - a tuple (inputs, targets)
        - a tuple (inputs, targets, sample_weights).

      - This tuple (a single output of the generator) makes a single batch. Therefore, all arrays in this tuple must have the same length (equal to the size of this batch). Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size. The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.

    - steps_per_epoch: Integer. Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to the number of samples of your dataset divided by the batch size. Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
    
    - epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire data provided, as defined by steps_per_epoch. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
    
    - verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    
    - callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training. See callbacks.

    - validation_data: This can be either
        - a generator or a Sequence object for the validation data
        - tuple (x_val, y_val)
        - tuple (x_val, y_val, val_sample_weights)

      on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.

    - validation_steps: Only relevant if validation_data is a generator. Total number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch. It should typically be equal to the number of samples of your validation dataset divided by the batch size. Optional for Sequence: if unspecified, will use the len(validation_data) as a number of steps.
    - class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
    - max_queue_size: Integer. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
    - workers: Integer. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
    - use_multiprocessing: Boolean. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
    - shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch. Only used with instances of Sequence (keras.utils.Sequence). Has no effect when steps_per_epoch is not None.
    - initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).

- Returns

  A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

### Example
```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # create numpy arrays of input data
                # and labels, from each line in the file
                x1, x2, y = process_line(line)
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
```



[keras fit_generator](https://keras.io/models/sequential/)

## Tutorial
[Image Classification using Convolutional Neural Networks in Keras](https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/) 


[Keras Tutorial: The Ultimate Beginner’s Guide to Deep Learning in Python](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)

[A detailed example of how to use data generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html)
