---
title: "Recursive"
date: 2019-02-27
classes: wide
use_math: true
tags: python algorithm recursive
category: algorithm
---

```python
def word_split(phrase,list_of_words, output = None):
    '''
    Note: This is a very "python-y" solution.
    ''' 
    
    # Checks to see if any output has been initiated.
    # If you default output=[], it would be overwritten for every recursion!
    if output is None:
        output = []
    
    if phrase is not None:
        print("phrase {} list_of_words {}".format(phrase,list_of_words))
    # For every word in list
    for word in list_of_words:
        
        # If the current phrase begins with the word, we have a split point!
        if phrase.startswith(word):
            print("phrase startswith {}".format(word))
            # Add the word to the output
            output.append(word)
            
            # Recursively call the split function on the remaining portion of the phrase--- phrase[len(word):]
            # Remember to pass along the output and list of words
            return word_split(phrase[len(word):],list_of_words,output)
    
    # Finally return output if no phrase.startswith(word) returns True
    return output        

def word_split(phrase,list_of_words, output = None):
    '''
    Note: This is a very "python-y" solution.
    ''' 
    
    # Checks to see if any output has been initiated.
    # If you default output=[], it would be overwritten for every recursion!
    if output is None:
        output = []
    
    if phrase is not None:
        print("phrase {} list_of_words {}".format(phrase,list_of_words))
    # For every word in list
    for word in list_of_words:
        
        # If the current phrase begins with the word, we have a split point!
        if word in phrase:
            print("{} in phrase ".format(word))
            # Add the word to the output
            output.append(word)
            
            # Recursively call the split function on the remaining portion of the phrase--- phrase[len(word):]
            # Remember to pass along the output and list of words
            return word_split(phrase.replace(word,''),list_of_words,output)
    
    # Finally return output if no phrase.startswith(word) returns True
    return output        

word_split('themanran',['the','ran','man'])
word_split('themanran',['clown','ran','man'])
>>phrase themanran list_of_words ['clown', 'ran', 'man']
>>ran in phrase 
>>phrase theman list_of_words ['clown', 'ran', 'man']
>>man in phrase 
>>phrase the list_of_words ['clown', 'ran', 'man']
>>['ran', 'man']
```

