---
title: "Array quiz"
date: 2019-03-08
classes: wide
use_math: true
tags: python algorithm string bst binary search tree
category: algorithm
---

## Find the missing Element

```python
def finder(arr1,arr2):
    
    # Sort the arrays
    arr1.sort()
    arr2.sort()
    
    # Compare elements in the sorted arrays
    for num1, num2 in zip(arr1,arr2):
        if num1!= num2:
            return num1
    
    # Otherwise return last element
    return arr1[-1]

arr1 = [1,2,3,4,5,6,7]
arr2 = [3,7,2,1,4,6]
finder(arr1,arr2)


import collections

def finder2(arr1, arr2): 
    
    # Using default dict to avoid key errors
    d=collections.defaultdict(int) 
    
    # Add a count for every instance in Array 1
    for num in arr2:
        d[num]+=1 
    
    # Check if num not in dictionary
    for num in arr1: 
        if d[num]==0: 
            return num 
        
        # Otherwise, subtract a count
        else: d[num]-=1 

arr1 = [5,5,7,7]
arr2 = [5,7,7]

arr1+arr2
[5, 5, 7, 7, 5, 7, 7]

def finder3(arr1, arr2): 
    result=0 
    
    # Perform an XOR between the numbers in the arrays
    for num in arr1+arr2: 
        result^=num 
        print result
        
    return result 

```


## Largest continuous sum  
```python
def large_cont_sum(arr): 
    
    # Check to see if array is length 0
    if len(arr)==0: 
        return 0
    
    # Start the max and current sum at the first element
    max_sum=current_sum=arr[0] 
    
    # For every element in array
    for num in arr[1:]: 
        
        print("current_sum+num:{} num:{}".format(current_sum+num, num))
        # Set current sum as the higher of the two
        current_sum=max(current_sum+num, num)
        print("current_sum:{} max_sum:{}".format(current_sum,max_sum))
        # Set max as the higher between the currentSum and the current max
        max_sum=max(current_sum, max_sum) 
        
    return max_sum 

large_cont_sum([-1,-2,10,3,4,10,10,-10,-1])    
```    

### Reverse sentence  
```python
def rev_word3(s):
    """
    Manually doing the splits on the spaces.
    """
    
    words = []
    length = len(s)
    spaces = [' ']
    
    # Index Tracker
    i = 0
    
    # While index is less than length of string
    while i < length:
        
        # If element isn't a space
        if s[i] not in spaces:
            
            # The word starts at this index
            word_start = i
            
            while i < length and s[i] not in spaces:
                
                # Get index where word ends
                i += 1
            # Append that word to the list
            words.append(s[word_start:i])
        # Add to index
        i += 1
        
    # Join the reversed words
    return " ".join(reversed(words))
```


### String compression  
```python
def compress(s):
    """
    This solution compresses without checking. Known as the RunLength Compression algorithm.
    """
    
    # Begin Run as empty string
    r = ""
    l = len(s)
    
    # Check for length 0
    if l == 0:
        return ""
    
    # Check for length 1
    if l == 1:
        return s + "1"
    
    #Intialize Values
    last = s[0]
    cnt = 1
    i = 1
    
    while i < l:
        
        # Check to see if it is the same letter
        if s[i] == s[i - 1]: 
            # Add a count if same as previous
            cnt += 1
        else:
            # Otherwise store the previous data
            r = r + s[i - 1] + str(cnt)
            cnt = 1
            
        # Add to index count to terminate while loop
        i += 1
    
    print("r {} s[i-1]:{}".format(r,s[i-1]))
    # Put everything back into run
    r = r + s[i - 1] + str(cnt)
    
    return r

compress('AAAAABBBBCCCC')
```        

### Unique Character  
```python
def uni_char(s):
    return len(set(s)) == len(s)

def uni_char2(s):
    chars = set()
    for let in s:
        # Check if in set
        if let in chars:
            return False
        else:
            #Add it to the set
            chars.add(let)
    return True

```    