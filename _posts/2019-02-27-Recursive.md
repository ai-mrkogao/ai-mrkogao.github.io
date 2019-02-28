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

## reverse string

```python
def reverse(s):
    
    # Base Case
    if len(s) <= 1:
        return s

    # Recursion
    return reverse(s[1:]) + s[0]
def reverse(s):
    
    # Base Case
    if len(s) <= 1:
        return s

    # Recursion
    return s[-1]+reverse(s[:-1])
reverse('hello world')
```

## Permutation
```python
def permute(s):
    out = []
    
    # Base Case
    if len(s) == 1:
        out = [s]
        print("last {}".format(s))
    else:
        # For every letter in string
        print("s {}".format(s))
        for i, let in enumerate(s):
            print("let {}".format(let))
            # For every permutation resulting from Step 2 and 3 described above
            for perm in permute(s[:i] + s[i+1:]):
                print("perm {}".format(perm))
                # Add it to output
                out += [let + perm]

    return out
permute('abc')
s abc
let a
s bc
let b
last c
perm c
out ['bc']
let c
last b
perm b
out ['bc', 'cb']
perm bc
out ['abc']
perm cb
out ['abc', 'acb']
let b
s ac
let a
last c
perm c
out ['ac']
let c
last a
perm a
out ['ac', 'ca']
perm ac
out ['abc', 'acb', 'bac']
perm ca
out ['abc', 'acb', 'bac', 'bca']
let c
s ab
let a
last b
perm b
out ['ab']
let b
last a
perm a
out ['ab', 'ba']
perm ab
out ['abc', 'acb', 'bac', 'bca', 'cab']
perm ba
out ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']

['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
```
