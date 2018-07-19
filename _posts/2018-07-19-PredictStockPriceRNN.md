---
title: "Predict Stock Price using RNN"
date: 2018-07-19
tags: python keras image classification tensorflow RNN LSTM
categories: stock
---

## Introduction
- This tutorial is for how to build a recurrent neural network using Tensorflow to predict stock market prices
- Part 1 focuses on the prediction of S$P 500 index
- This motivation is demonstrating how to build and train on RNN model in Tensorflow and less on solve the stock prediction problem

## Recurrent Neural Network
- A sequence model is usually designed to transform an input sequence into an output sequence that lives in a different domain
- RNN shows us tremendous improvement in handwriting recognition,speech recognition, and machine translation
- A RNN model is born with the capability to process long sequential data and to tackle with context spreading in time
- Imagine the case when an RNN model reads all the Wikipedia articles, character by character, and then it can predict the following words given the context
- ![rnngeneralmodel](../../pictures/stock/rnngeneralmodel.png){:height="70%" width="70%"}

- However,simple perception neurons that linearly combine the current input and the last unit state may easily lose the long-term dependencies
  - For example, we start a sentence with "Alice is working at..." and later after a whole paragraph, we want to start the next sentence with "She" or "He" correctly
  - If the model forgets the character's name "Alice",we can never know
  - To resolve the issue, researchers created a special neuron with a much more complicated internal structure for memorizing long-ternm context, named ***Long short tern memory(LSTM)*** 
  - It is smart enough to learn for ***how long it should memorize the old information, when to forget, when to make sure of the new data, and how to combine the old memory with new input***
  - ![lstmmodel](../../pictures/stock/lstmmodel.png){:height="40%" width="40%"}

## LSTM networks
- LSTM is a special kind of RNN
- LSTM are explicitly designed to avoid the long-term dependency problem
- Remembering information for long periods of time is practically their default behavior
- ![rnnnetwork](../../pictures/stock/rnnnetwork.png){:height="70%" width="70%"} 
  - LSTM also have this chain structure but the repeating module has a different structure
  - Instead of having a single neural network layer, there are ***four structure***

- ### LSTM repeating module contains four interacting layers
- ![lstmcomponents](../../pictures/stock/lstmcomponents.png){:height="70%" width="70%"} 
- each line carries an entire vector: from the output of one node to the input of others
- The Pink circles represent pointwise operation like vector addition
- the yellow boxes are learned neural network layers
- Line merging denote concatenation
- a line forking denote a content being copied and the copies going to different locations

## Core Idea Behind LSTM
- The key to LSTMs is the cell state, the horizonal line running through the top of the diagram
- ![lstmcomponents_1](../../pictures/stock/lstmcomponents_1.png){:height="50%" width="50%"} 
- The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates
- ![lstmcomponents_sigmoid](../../pictures/stock/lstmcomponents_sigmoid.png){:height="20%" width="20%"}
- Gates are a way to optionally let information through
  - They are composed out of a sigmoid neural net layer between zero and one and a pointwise multiplication operation
  - sigmoid output[0,1] means how much of each component should be let through
  - A zero means ***let nothing through***
  - A one means ***let everything through***
- A LSTM has threee gates to protect and control the cell state

## Step by Step LSTM Walk Through
- ***The first step*** in our LSTM is to decide what information we're going to throw away from the cell state
  - This decision is made by a sigmoid layer called the "forget gate layer"
  - It looks at ![ht_1](../../pictures/stock/ht_1.png){:height="5%" width="5%"} and ![xt](../../pictures/stock/xt.png){:height="4%" width="4%"} and outputs a number between 0 and 1 for each number in the cell state ![ct_1](../../pictures/stock/ct_1.png){:height="5%" width="5%"}
  - 1 represents ***completely keep this*** while ***0 represents get rid of this***
  - Let's go back to our example of a language model trying to predict the next word based on all the previous ones
  - In such a problem, the cell state might include the gender of the present subject, so that the correct pronouns can be used
  - When we see a new subject, we want to forget the gender of the old subject
  - ![lstm_firstgate](../../pictures/stock/lstm_firstgate.png){:height="90%" width="90%"}

- ***The Next step*** is to decide what new information we're going to store in the cell state
  - This has two parts
  - First, a sigmoid layer called "input gate layer" decides which values we'll update
  - Next, a tanh layer creates a vector of new candidate values,![ct_tilt](../../pictures/stock/ct_tilt.png){:height="4%" width="4%"} that could be added to the state
  - In the next step, we'll combine these two to create an update to the state
  - In the example of our language model, we'd want to add the gender of the new subject to the cell state, to replace the old one we're forgetting
-![lstm_secondgate](../../pictures/stock/lstm_secondgate.png){:height="90%" width="90%"}  
  - It's now time to update the old cell state, ![ct_minus](../../pictures/stock/ct_minus.png){:height="5%" width="5%"}, into the new cell state ![ct](../../pictures/stock/ct.png){:height="4%" width="4%"}
  - We multiply the old state by ![ft](../../pictures/stock/ft.png){:height="3%" width="3%"}, forgetting the things we decided to forget earlier
  - Then we add ![itmultiplyct_tilt](../../pictures/stock/itmultiplyct_tilt.png){:height="9%" width="9%"}, This is the new candidate values, scaled by how much we decided to update each state value
  - In the case of language model, this is where we'd actually drop the information about the old subject's gender and add the new information
  - ![lstmsecondcal](../../pictures/stock/lstmsecondcal.png){:height="90%" width="90%"}  

- ***Finally***, we need to decide what we're goint to output
  - This output will be based on our cell state, but will be a filtered version
    - First, we run a sigmoid layer which decides what parts of the cell state we're going to output
    - Then, we put the cell state through tanh(to push the values to be between -1 and 1) and multiply it by the output of the sigmoid gate,so that we only output the parts we decided to
    - for language model,since it just saw a subject, it might want to output information relevant to a verb, in case that's what is coming next
    - For example, it might output whether the subject is singular or plural,so that we know what form a verb should be conjugated into if that's what follows next
    - ![lstmfinalgate](../../pictures/stock/lstmfinalgate.png){:height="110%" width="110%"}  













## Tutorial
[Predict Stock Prices Using RNN: Part 1](https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html) 

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)