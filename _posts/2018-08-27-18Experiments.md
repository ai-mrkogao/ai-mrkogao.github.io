---
title: "18 Experiments (RL matplotlib plot graph to verify)"
date: 2018-08-27
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning
category: stock
---

## Matplotlib multiple ax plot
- debug purpose and result comparison

```python
self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
for ax in self.axes:
	# disable scientific indicators
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)

x = np.arange(len(chart_data))
y = np.array()
self.axes[0].plot(x, y, )  

self.axes[1].fill_between(x, pvs, pvs_base,
                                  where=pvs > pvs_base, facecolor='r', alpha=0.1)
self.axes[1].fill_between(x, pvs, pvs_base,
                          where=pvs < pvs_base, facecolor='b', alpha=0.1)
self.axes[1].plot(x, pvs, '-k')


for ax in self.axes[1:]:
    ax.cla()  
    ax.relim() 
    ax.autoscale()

self.axes[1].set_ylabel('')
self.axes[2].set_ylabel('')
self.axes[3].set_ylabel('')
for ax in self.axes:
    ax.set_xlim(xlim)  
    ax.get_xaxis().get_major_formatter().set_scientific(False) 
    ax.get_yaxis().get_major_formatter().set_scientific(False) 
    ax.ticklabel_format(useOffset=False) 
    
```


## Predict
- input states makes output which expresses the turning points in the future

## Training
- up,down patterns
