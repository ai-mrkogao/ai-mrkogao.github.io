title: "GAN predict next state"
date: 2019-03-15
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning machine_learning  GAN DCGAN
category: reinforcement learning
---

## Predict next stock state  
```python

def gan_predict(self):
    tf.reset_default_graph()
    gan = GAN(num_features=5, num_historical_days=self.num_historical_days,
                    generator_input_size=200, is_train=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.gan_model)
        clf = joblib.load(self.xgb_model)
        
        for sym, date, data in self.data:
            # gan.features input X placeholder : num_historical_days * num_features(5)
            features = sess.run(gan.features, feed_dict={gan.X:[data]})
            print('features {}'.format(features))
            _features.append(features)
            features = xgb.DMatrix(features)
            print('{} {} {}'.format(str(date).split(' ')[0], sym, clf.predict(features)[0][1] > 0.5))

```