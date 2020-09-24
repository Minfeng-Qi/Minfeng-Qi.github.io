---
layout: post
title: "Learn Tensorflow Through Code"
date: 2020-09-22 
description: "Tensorflow"
tag: Machine Learning
---

### Import tensorflow

`import tensorflow as tf`

### 1.1 Constant() method => create a tensor

```python
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

- **zeros() and ones()**

  `a = tf.zeros([2, 3, 3])`

  `b = tf.ones([2, 3, 3])`

- **zeros_like()与ones_like**

  `tf.zeros_like(b)`

  `tf.ones_like(a)`

- **Fill()**

  `tf.fill([2,3,3],5)`	 # value = 5

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



