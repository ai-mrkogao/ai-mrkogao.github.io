---
title: "Matplotlib tutorial"
date: 2018-08-22
classes: wide
use_math: true
tags: python matplotlib subplot plot
category: python_api
---


## Matplotlib Subplot

[Matplotlib Subplot](https://pythonspot.com/matplotlib-subplot/)

```python
from pylab import *
 
t = arange(0.0, 20.0, 1)
s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
 
subplot(2,1,1)
xticks([]), yticks([])
title('subplot(2,1,1)')
plot(t,s)
 
subplot(2,1,2)
xticks([]), yticks([])
title('subplot(2,1,2)')
plot(t,s,'r-')
 
show()
```

```python
from pylab import *
 
t = arange(0.0, 20.0, 1)
s = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
 
subplot(2,2,1)
xticks([]), yticks([])
title('subplot(2,2,1)')
plot(t,s)
 
subplot(2,2,2)
xticks([]), yticks([])
title('subplot(2,2,2)')
plot(t,s,'r-')
 
subplot(2,2,3)
xticks([]), yticks([])
title('subplot(2,2,3)')
plot(t,s,'g-')
 
subplot(2,2,4)
xticks([]), yticks([])
title('subplot(2,2,4)')
plot(t,s,'y-')
 
show()
```

```python
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
```

