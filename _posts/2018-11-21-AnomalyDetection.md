---
title: "Time Series Anomaly Detection & RL time series"
date: 2018-11-21
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning machine_learning anomaly pnn
category: reinforcement learning
---


[Prediction of Stock Moving Direction](https://github.com/aravanshad/Stock_Movement_Prediction)

[Detecting Stock Market Anomalies ](https://slicematrix.github.io/stock_market_anomalies.html)

[Python API for SliceMatrix-IO ](https://github.com/tynano/slicematrixIO-python)

[From Financial Compliance to Fraud Detection with Conditional Variational Autoencoders (CVAE) and Tensorflow](https://amunategui.github.io/cvae-in-finance/index.html)

[CVAE-Financial-Anomaly-Detection](https://github.com/amunategui/CVAE-Financial-Anomaly-Detection/blob/master/Financial%20Compliance%20and%20Fraud%20Detection%20with%20Conditional%20Variational%20Autoencoders%20%28CVAE%29%20and%20Tensorflow.ipynb)

[Probabilistic reasoning and statistical analysis in TensorFlow](https://github.com/tensorflow/probability)

[ RL & SL Methods and Envs For Quantitative Trading](https://github.com/ceruleanacg/Personae)

# CVAE

## CVAE (Conditional Variation Autoencoder)
- we are going to see how CVAE can learn and generate the behavior of a particular stock price action 
- CVAE generates millions of points and whenever real price action veers too far away from the bounds of these generated patterns, we know that something is different

![](https://amunategui.github.io/cvae-in-finance/img/autoencoder-schema.png){:height="50%" width="50%"}  

## The Autoencoder can reconstruct Data
- The autoencoder is an unsupervised neural network that combines a data encoder and decoder
- The encoder reduces data into a lower dimensional space known as the latent space representation
- The decoder will take this reduced representation and blow it back up to its original size
- ***This is also used in anomaly detection. You train a model, feed new data into the encoder,compress it, then observe how well it rebuilds it***
- ***If the reconstruction error is abnormally high, that means the model strugged to rebuild the data and you may have an anomaly on your hands***

## The Variatoinal autoencoder can generate Data
- The variational autoencoder adds the ability to generate new synthetic data from this compressed representation
- It does so by learning the probability distribution of the data and we can thus generate new data by using different latent variables used as input

## The Conditional Variational Autoencoder(CVAE) Can generate Data by Lable
- With CVAE, we can ask the model to recreate data(synthetic data) for a particular label
- we can ask it to recreate data for a particular stock symbol
- we ask the decoder to generate new data down to the granularity of labels





