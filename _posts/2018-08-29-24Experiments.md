---
title: "24 Experiments (LSTM prediction training)"
date: 2018-08-28
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning svm lstm regression 
category: stock
---


## LSTM prediction training


## build dataset
```python
def builddataset(df_train,df_test,SEQ_SIZE):

    seq_length = SEQ_SIZE
    TRAINLEN = len(df_train)- 2*SEQ_SIZE
    df_orgtrain = copy.deepcopy(df_train)
    
    x = df_orgtrain[['Open']].values
    print('x {}'.format(x[:12]))
    # build a dataset
    dataX = []
    for i in range(seq_length, len(x) - seq_length-1):
        _all = []
        _x = []
        _x1 = x[i-seq_length:i]
        _x1 = _x1.reshape((1,5))
        _x.append(_x1[0].tolist())
#         print('_x1 {}_x1 {}'.format(_x1,_x1.shape))
        _x2 = x[seq_length]
        _x.append(_x2)
#         print('_x2 {}'.format(_x2))
        _x3 = x[i+1:i+seq_length+1]
        _x3 = _x3.reshape((1,5))
#         print('_x3 {} '.format(_x3))
        _x.append(_x3[0].tolist())
#         print('_x {} '.format(_x))
        for idx in range(len(_x)):
            _all.extend(_x[idx])
        print('_all {}'.format(_all))

```

## Keras load model
```python
try:
    # load json and create model
    json_file = open(save_dir+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(filename)
    print("Loaded model from disk")
except :
    PrintException()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
```        