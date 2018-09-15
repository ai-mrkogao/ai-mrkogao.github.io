---
title: "Image Captioning"
date: 2018-09-15
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning turi machine_learning platform image caption
category: reinforcement learning
---


![](../../pictures/hvass/22_image_captioning_flowchart.png){:height="70%" width="70%"}

## Load Coco dataset
```python
_, filenames_train, captions_train = coco.load_records(train=True)
```


## Pre-trained Image Model
```python
image_model = VGG16(include_top=True, weights='imagenet')

transfer_layer = image_model.get_layer('fc2')
```

## Training
```python
%%time
transfer_values_train = process_images_train()
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)

# Use the pre-trained image-model to process the image.
# Note that the last batch may have a different size,
# so we only use the relevant images.
transfer_values_batch = \
    image_model_transfer.predict(image_batch[0:current_batch_size])
```

## Tokenizer
Neural Networks cannot work directly on text-data. We use a two-step process to convert text into numbers that can be used in a neural network. The first step is to convert text-words into so-called integer-tokens. The second step is to convert integer-tokens into vectors of floating-point numbers using a so-called embedding-layer

```python
mark_start = 'ssss '
mark_end = ' eeee'

captions_train
captions_train_marked[0]
['ssss Closeup of bins of food that include broccoli and bread. eeee',
 'ssss A meal is presented in brightly colored plastic trays. eeee',
 'ssss there are containers filled with different kinds of foods eeee',
 'ssss Colorful dishes holding meat, vegetables, fruit, and bread. eeee',
 'ssss A bunch of trays that have different food. eeee']

captions_train[0]
['Closeup of bins of food that include broccoli and bread.',
 'A meal is presented in brightly colored plastic trays.',
 'there are containers filled with different kinds of foods',
 'Colorful dishes holding meat, vegetables, fruit, and bread.',
 'A bunch of trays that have different food.']

captions_train_flat = flatten(captions_train_marked)
captions_train_flat
['ssss Closeup of bins of food that include broccoli and bread. eeee',
 'ssss A meal is presented in brightly colored plastic trays. eeee',
 'ssss there are containers filled with different kinds of foods eeee'
 ...
 ]

tokenizer = TokenizerWrap(texts=captions_train_flat,
                          num_words=num_words)

tokens_train = tokenizer.captions_to_tokens(captions_train_marked)

tokens_train[0] ==> captions_train_marked[0]
[[2, 841, 5, 2864, 5, 61, 26, 1984, 238, 9, 433, 3],
 [2, 1, 429, 10, 3310, 7, 1025, 390, 501, 1110, 3],
 [2, 63, 19, 993, 143, 8, 190, 958, 5, 743, 3],
 [2, 299, 725, 25, 343, 208, 264, 9, 433, 3],
 [2, 1, 170, 5, 1110, 26, 446, 190, 61, 3]]                            


batch_x['transfer_values_input'][0] == > 4096 vector size transfer_values

batch_x['decoder_input'][0] ==> padded with max_tokens size 

batch_y['decoder_output'][0]

num_captions_train = [len(captions) for captions in captions_train]
[5,
 5,
 5,
 ...]

total_num_captions_train = np.sum(num_captions_train)


```

## Create Network
```python
state_size = 512
embedding_size = 128

transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')

decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')

decoder_input = Input(shape=(None, ), name='decoder_input')

num_words = 10000 # most frequent 10000 words only used in token

decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')


decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)

decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(transfer_values):
    # Map the transfer-values so the dimensionality matches
    # the internal state of the GRU layers. This means
    # we can use the mapped transfer-values as the initial state
    # of the GRU layers.
    initial_state = decoder_transfer_map(transfer_values)

    # Start the decoder-network with its input-layer.
    net = decoder_input
    
    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output


decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])
                          
```