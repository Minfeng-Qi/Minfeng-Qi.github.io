---
layout: post
title: "Learn Tensorflow Through Code"
date: 2020-09-22 
description: "Tensorflow"
tag: Machine Learning
---

## 1 Create, access, update tensor

`import tensorflow as tf` 

### 1.1 Constant() method => create a tensor

```python
#  (axis = 0 => col; axis = 1 => row;)
# create integer tensor
tf.constant(1)
# create float tensor
tf.constant(1.)
# create a tensor and assign a type
tf.constant(1., dtype=tf.double)
# create a tensor through passing a list
tf.constant([[1.,2.,3.],[4.,5.,6.]])
# convert to a tensor
tf.convert_to_tensor(np.ones([2, 3]))
```

### 1.2 Create a tensor assigned values

```python
# zeros() and ones()
a = tf.zeros([2, 3, 3])
b = tf.ones([2, 3, 3])

# zeros_like()与ones_like
tf.zeros_like(b)
tf.ones_like(a)

# Fill()
tf.fill([2,3,3],5)	 # value = 5
```

### 1.3 Random initialization

- **tf.random.normal()** 	# Assign value from Normal distribution

  `tf.random.normal([2, 3, 3], mean=1, stddev=1)`

- **tf.random.uniform()** 	# Assign value from Uniform distribution

  `tf.random.uniform([2, 3], minval=1, maxval=2)`

### 1.4 Index

```python
a = tf.convert_to_tensor(np.arange(80).reshape(2,2,4,5))
# pick up basic index by dimensions
a[0][1][3][3]
# pick up numpy index 
a[1,1,3,3]
# slice 
a[1, :, 0:2, 0:4] 
# slice and step
a[1, :, 0:2, 0:4:2]
# ...
a[1,...,0:4] # equal to a[1, : , : ,0:4]
```

### 1.5 gather and gather_nd

```python
b = np.arange(20).reshape(4,5)
# gather for one dimension
tf.gather(b, axis=0, indices=[0, 2, 3])  # select row 1,3,4
tf.gather(b, axis=1, indices=[0, 2, 3])  # select col 1,3,4
# gather for multiple dimension
tf.gather_nd(b, [[0, 2],[3, 3]])  # select row 1 col 3, and row 4 col 4
```

### 1.6 condition index

```python
a = tf.random.uniform([3,3],minval=-10,maxval=10,dtype=tf.int32)
cond = a < 0
# boolwan_mask() to fetch value 
tf.boolean_mask(a,cond) 
# gather_nd() to fetch value 
m_index = tf.where(cond)
tf.gather_nd(a,m_index)
```

### 1.7 reshape() and transpose()

```python
# reshape()
a = tf.ones([2,3,4])
b = tf.reshape(a, [2,2,6])
# transpose()
c = tf.transpose(a)
d = tf.transpose(a, perm=[0, 2, 1])  # first dim don't change
```

### 1.8 expand_dims()  => add dimensions

 ```python
a = tf.random.uniform([2,3])   # shape=(2,3)
tf.expand_dims(a, axis=0)  # shape=(1,2,3)
tf.expand_dims(a, axis=1)  # shape=(2,1,3)
tf.expand_dims(a, axis=2)  # shape=(2,3,1)
tf.expand_dims(a, axis=-1)   # shape=(2,3,1)
 ```

### 1.9 squeeze()  => delete dims = 1

```python
a = tf.random.uniform([1,2,1,3])  # shape=(1,2,1,3)
tf.squeeze(a)   # shape=(2,3)
```

## 2 Math Operation

### 2.1 Basic operation: +、-、*、/、//、%

```python
a = tf.random.uniform([2, 3], minval=1, maxval=6,dtype=tf.int32)
b = tf.random.uniform([2, 3], minval=1, maxval=6,dtype=tf.int32)
# add +
tf.add(a,b)  // 
a + b
# subtract - 
tf.subtract(a,b)  // 
a - b
# multiply * 
tf.multiply(a,b)  // 
a * b
# divide /
tf.divide(a,b)   // 
a / b
# floor divide //
tf.floor_div(a,b)  //
a // b
# mod %
tf.mod(b,a)  //
b % a
```

### 2.2 Advanced operation: log, pow, sqrt

```python
# tf.math.log() => log
# eg.1
e = 2.71828183
a = tf.constant([e, e*e, e*e*e])
tf.math.log(a)
# eg.2
f = tf.constant([[1., 9.], [16., 100.]])
g = tf.constant([[2., 3.], [2., 10.]])
tf.math.log(f) / tf.math.log(g)

# tf.pow()  => pow
g = tf.constant([[2, 3], [2, 10]])
tf.pow(g, 2)   //
g ** 2

# tf.sqrt()  => sqrt
f = tf.constant([[1., 9.], [16., 100.]])
tf.sqrt(f)
```

### 2.3 Matrix multiply

```python
# tf.matmul() two dims
a = tf.constant(np.arange(6),shape=(2,3))
b = tf.constant(np.arange(6),shape=(3,2))
tf.matmul(a, b)  //
a @ b  # => shape=(2,2)
# three dims
a = tf.constant(np.arange(12),shape=(2,2,3))  
b = tf.constant(np.arange(12),shape=(2,3,2))
a @ b  # => shape=(2,2,2)
```

### 2.4 Broadcasting

```python
a = tf.constant([1,2,3])  # shape=(3,)
b = tf.constant(np.arange(12),shape=(2,2,3))
a + b 
a * b

# process of broadcasting
step0: a shape = (3,) 
step1: a shape = (1,1,3) 
step2: a shape = (1,2,3) 

# the condition to meet broadcasting: 1. equal 2. one tensor = 1
# More example:
[1] A：(2d array): 5 x 4
[2] B：(1d array): 1
[3] Result：(2d array): 5 x 4
  
[1] A：(2d array): 5 x 4
[2] B：(1d array): 4
[3] Result：(2d array): 5 x 4
  
[1] A：(3d array): 15 x 3 x 5
[2] B：(3d array): 15 x 1 x 5
[3] Result：(3d array): 15 x 3 x 5
  
[1] A：(3d array): 15 x 3 x 5
[2] B：(2d array): 3 x 5
[3] Result：(3d array): 15 x 3 x 5
  
[1] A：(3d array): 15 x 3 x 5
[2] B：(2d array): 3 x 1
[3] Result：(3d array): 15 x 3 x 5
```

### 2.5 Norm

```python
a = tf.constant([[1.,2.],[1.,2.]])
# Norm 1
tf.norm(a, ord=1) 
# Norm 2
tf.norm(a, ord=2)
# fault Norm = 2
tf.norm(a) == tf.sqrt(tf.reduce_sum(tf.square(a)))
# assign dims
tf.norm(a, ord=2, axis=0)
tf.norm(a, ord=2, axis=1)
```

## 3 Sort and Min, Max, Mean

### 3.1 Sort

```python
# dim = 1
a = tf.random.shuffle(tf.range(6))
## ascending
tf.sort(a) = tf.sort(a, direction='ASCENDING')
## descending
tf.sort(a, direction='DESCENDING')
# dim > 1
b = tf.random.uniform([3, 3], minval=1, maxval=10,dtype=tf.int32)
tf.sort(b,axis=0)  # sort by col

# tf.argsort() => return index after sorting
tf.argsort(a, direction='ASCENDING')

# top_k()  => fetch the top elements after sorting
top_2 = tf.math.top_k(a, 2)  # fetch the first two
top_2.values  # fetch element value
top_2.indices   # fetch element index
```

### 3.2 reduce_min(), reduce_max(), and reduce_mean()

```python
a = tf.random.uniform([3, 3], minval=1, maxval=10, dtype=tf.int32)
# min 
tf.reduce_min(a)
tf.reduce_min(a, axis=0) 

# max
tf.reduce_max(a)
tf.reduce_max(a, axis=-1)

# mean
tf.reduce_mean(a)
tf.reduce_mean(a, axis=0)
tf.reduce_mean(tf.cast(a, tf.float32), axis=0)

# argmin() => fault dim = 0
tf.argmin(a) == tf.argmin(a, axis=0)

# argmax() => fault dim = 0
tf.argmax(a) == tf.argmax(a, axis=0)

```

## 4 Pad and Tile

### 4.1 Pad (fill tensor)

**pad(tensor, paddings, mode="CONSTANT", name=None, constant_values=0)**

- tensor:
- paddings: [[x,y]] when dim =1, [[x,y], [x,y]] when dim =2, x before, y after
- mode: CONSTANT (default);   REFLECT ; SYMMETRIC
- name:
- constant_values：CONSTANT (def = 0)

```python
# dim = 2
a = tf.random.uniform([3, 4], minval=1, maxval=10, dtype=tf.int32)
tf.pad(a, [[1,1],[3,0]], constant_values=3)

# dim = 3
b = a.reshape(a, [2,2,3])
tf.pad(b, [[1,0],[1,1],[1,0]])

```

### 4.2 Tile (copy tensor)

**tile(input, multiples, name=None)**

- input：tensor
- multiples: 0 delete，1 no copy，2 copy once , 3 copy twice ...

```python
# dim = 1
a = tf.range(12)
tf.tile(a,[2])

# dim = 2
b = tf.reshape(a, [3,4])
tf.tile(b, [2,3]) 

# dim = 3
c = tf.reshape(a, [2,2,3])
tf.tile(c, [2,1,2])
```

