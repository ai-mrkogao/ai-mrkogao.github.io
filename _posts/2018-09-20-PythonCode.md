---
title: "Python Code snippet"
date: 2018-09-20
classes: wide
use_math: true
tags: python stock utils keras tensorflow pandas numpy 
category: python_api
---


## Word Counting and vocabrary 

```python
word_reviews = []
all_words = []
for review in reviews_processed:
    word_reviews.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())
    
counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

vocab_to_int
>>
{'warns': 938,
 'funny': 457,
 'against': 616,
 'acne': 211,...
}

```



