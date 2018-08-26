---
title: "17 Experiments (RL DQN 20 training set with signals including yaw,ck,ma)"
date: 2018-08-25
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning
category: stock
---

## Hyper parameter
```python
SEQ_SIZE = 2
def builddataset(df_train,df_test,SEQ_SIZE):

    x = df_train[['Open','High','Low','Volume','Close','5MA','20MA','60MA','60MASTD','curvature_60ma']].values

def _build_network(self, h_size=128, l_rate=0.001) -> None:

REPLAY_MEMORY = 1500

def main_period(trainmodel,df_data,np_data):
    np_target = np_data
    
    tf.reset_default_graph()
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)



SEQ_SIZE = 2
np_train,np_test = builddataset(df_train,df_test,SEQ_SIZE)

INPUT_SIZE = np_train.shape[1]
OUTPUT_SIZE = 4

DISCOUNT_RATE = 0.99
MAX_EPISODE = 20
BATCH_SIZE = 32
MIN_E = 0.01
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.1

```
