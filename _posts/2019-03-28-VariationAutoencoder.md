title: "Variation Autoencoder in Tensorflow"
date: 2019-03-28
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning machine_learning  VA autoencoder
category: reinforcement learning
---


### Variation Autoencoder Implementation in Tensorflow  
```python
class VariationAutoencoder:

	def __init__(self,InputDim,HiddenDim):
		# Input Dimension
		self.X = tf.placeholder(tf.float32,shape=(None,InputDim))

		###################
		# encoder 
		###################
		self.encoder_layers = []

		DimIn = InputDim
		MIn = self.X 

		for DimOut in HiddenDim[:-1]:
			hidden_weight = tf.Variable(tf.random_normal(shape=(DimIn,DimOut)) * 2 / np.sqrt(DimIn))
			hidden_bias = tf.Variable(np.zeros(DimOut).astype(np.float32))

			h = tf.nn.relu(tf.matmul(MIn,hidden_weight)+ hidden_bias)

			self.encoder_layers.append(h)
			MIn = h


        # we need 2 times hidden units 
        # 2 times MOut means , MOut variances
        
```

[Variation Autoencoder](http://kvfrans.com/variational-autoencoders-explained/)  

### Mean Vector, Standard Deviation Vector : double size?  


