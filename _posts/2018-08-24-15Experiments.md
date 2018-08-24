---
title: "15 Experiments (RL DQN consecutive points state)"
date: 2018-08-24
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning
category: stock
---

## Points
- 3 consecutive points to express current state

## Hyper Parameters
SEQ_SIZE = 2
INPUT_SIZE = 10
OUTPUT_SIZE = 4
DISCOUNT_RATE = 0.9
REPLAY_MEMORY = 3500
MAX_EPISODE = 30
BATCH_SIZE = 64
MIN_E = 0.0
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.4
h_size=256, l_rate=0.001
init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

## Result
- rough training makes better result around all data
- tighten training makes worse result 
- let's try additional training data (12 Experimnets can generate the signals)
- 97.2% success


## Predict and Verification
- to predict the future, I will use additional data(economic indices) with current stock movement
- tremendous number of stocks are selected and trained with tomorrow gain on RL 
- a various policy applied (randon,bayesian,mc)
- all stocks are classified into several class types(period 1: portfolio 1, period 2:portfolio 2,...)
- current stocks -> into portfolio N

## short-term, long-term 
- if I controls point frequency in short, trade frequency is also decreased?
- Let's test
