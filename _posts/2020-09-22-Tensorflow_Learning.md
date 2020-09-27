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

## 5 Tensor Value Limit

### 5.1 maximum() and minimum()

```python
# maximum() is replace the minimal value the designated value
a = tf.range(10)   # output:  numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tf.maximum(a, 4)	 # output:  numpy=array([4, 4, 4, 4, 4, 5, 6, 7, 8, 9]

# minimum() is replace the maximal value the designated value
tf.minimum(a, 6)   # output:  numpy=array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6]

# combine them together
tf.minimum(tf.maximum(a,4),6)    # output:  numpy=array([4, 4, 4, 4, 4, 5, 6, 6, 6, 6]

```

### 5.2 clip_by_value()

clip_by_value() is equal to combine maximum() and minimum()

```python
b = tf.random.uniform([3,4], minval=1, maxval=10, dtype=tf.int32)
tf.clip_by_value(b,4,6) # == tf.minimum(tf.maximum(b,4),6)
```

### 5.3 relu()

Relu() limits to minimal value to 0, equal to tf.maximum(a, 0)

`tf.nn.relu(a)`

### 5.4 clip_by_norm()

clip_by_norm() is based on L2 Norm

```python
a = tf.random.normal([2,3],mean=10)
tf.clip_by_norm(a,10)    	# limit to 0-10

# the process could be like this:
n = tf.norm(a)  # Norm L2
a1 = a / n      # scale to 0-1
a2 = a1 * 10    # limit to 0-10
```

### 5.5 clip_by_global_norm()

clip_by_global_norm() is used to correct the gradient value and control the problem of gradient explosion

```python
t1 = tf.random.normal([3],mean=10)
t2 = tf.random.normal([3],mean=10)
t3 = tf.random.normal([3],mean=10)
t_list = [t1,t2,t3]
tf.clip_by_global_norm(t_list,25)

# the process could be like this:
global_norm = tf.norm([tf.norm(t) for t in t_list])  # Norm L2 for global
[t*25/global_norm for t in t_list]   # limit to 0-25
```

## 6 Dataset create operation

### 6.1 create dataset object

```python
# (1) use data.Dataset.range()
# range(begin)、range(begin, end)、range（begin, end, step)

dataset1 = tf.data.Dataset.range(5)
for i in dataset1:
    print(i)
    print(i.numpy())
    
# result 
# --- begin---
tf.Tensor(0, shape=(), dtype=int64)
0
tf.Tensor(1, shape=(), dtype=int64)
1
tf.Tensor(2, shape=(), dtype=int64)
2
tf.Tensor(3, shape=(), dtype=int64)
3
tf.Tensor(4, shape=(), dtype=int64)
4
# --- end ---
# each element is a tensor, and the value can be accessed by call numpy()


# (2)  data.Dataset.from_generator()
def count(stop):
  i = 0
  while i<stop:
    print('i')
    yield i
    i += 1

dataset2 = tf.data.Dataset.from_generator(count, args=[3], output_types=tf.int32, output_shapes = (), )


# (3)  from_tensors()
dataset2 = tf.data.Dataset.from_tensors([a,b])
dataset2_n = tf.data.Dataset.from_tensors(np.array([a,b]))
dataset2_t = tf.data.Dataset.from_tensors(tf.constant([a,b]))
next(iter(dataset2))


# (4) from_tensor_slices() **** most used one ****
a = [0,1,2,3,4]
b = [5,6,7,8,9]
dataset3 = tf.data.Dataset.from_tensor_slices([a,b])
for i,elem in enumerate(dataset3):
    print(i, '-->', elem)
    
dataset3 = tf.data.Dataset.from_tensor_slices((a,b))  # most useful
for i in dataset3:
    print(i)
    
```

### 6.2 dataset function

```python
# take()  to return a subdataset
dataset = tf.data.Dataset.range(10)
dataset_take = dataset.take(5)
for i in dataset_take:
    print(i)
    
    
# batch(batch_size, drop_ramainder)  to batch dataset
dataset_batch = dataset.batch(3)
dataset_batch = dataset.batch(3,drop_remainder=True)


# padded_batch(batch_size, padded_shapes, padding_values, drop_remainder)
dataset_padded = dataset.padded_batch(4,　padded_shapes(10,),　padding_values=tf.constant(9,dtype=tf.int64))


# map(_func)
def change_dtype(t):  # change type to int32
    return tf.cast(t,dtype=tf.int32)
 
dataset = tf.data.Dataset.range(3)
dataset_map = dataset.map(change_dtype)


#filter(_func)
dataset = tf.data.Dataset.range(5)
def filter_func(t):  # filter elements equal to even
    if t % 2 == 0:
        return True
    else:
        return False
      
dataset_filter = dataset.filter(filter_func)


# shuffle(buffer_size, seed, rushuffle_each_interation)
dataset = tf.data.Dataset.range(5)
dataset_s = dataset.shuffle(5) # when the size equals to the max, totally shuffle


# repeat()
dataset = tf.data.Dataset.range(3)
dataset_repeat = dataset.repeat(3) # repeat three times
```

## 7 Activation_function

### 7.1 sigmoid()

$$
f(x) = \frac{1}{1+e^{-x}}
$$



<img src='/Minfeng-Qi.github.io/images/posts/tensorflow/sigmoid.png' style='zoom:50%'>

```python
x = tf.linspace(-5., 5.,6)

# two ways to call sigmoid function 
tf.keras.activations.sigmoid(x) 
tf.sigmoid(x)

```

### 7.2 relu()

tf.keras.activations.relu( x, alpha=0.0, max_value=None, threshold=0 )	
$$
f(x) = max(0,x)
$$
<img src='/Minfeng-Qi.github.io/images/posts/tensorflow/relu.png' style='zoom:50%'>

```python
x = tf.linspace(-5., 5.,6)

tf.keras.activations.relu(x,alpha=2., max_value=10., threshold=3.5)
```

### 7.3 softmax()

Softmax() is often used to classify objects. Probability belongs to [0,1] and sum = 1.
$$
f(x_i) = \frac{e^{x_i}}{\sum{e^{x_i}}}
$$

```python
tf.keras.activations.softmax(tf.constant([[1.5,4.4,2.0]]))
```

### 7.4 tanh()

\$ result \in [-1,1] \$
$$
f(x) = \frac{\sinh{x}}{\cosh{x}} = \frac{1-e^{-2x}}{1+e^{-2x}}
$$
<img src='/Minfeng-Qi.github.io/images/posts/tensorflow/tanh.png' style='zoom:50%'>

```python
x = tf.linspace(-5., 5.,6)
tf.keras.activations.tanh(x)
```

## 8 Loss_function

### 8.1 Mean Square Error (MSE)

Used for regression

\$loss = \frac{1}{N}\sum{(y-pred)^{2}}\$

```python
loss_mse_1 = tf.losses.MSE(y,pred)
loss_mse_2 = tf.reduce_mean(loss_mse_1)
```

### 8.2 Cross Entropy

Used for classification

Measure the difference information between two probability distributions. The smaller the cross entropy, the smaller the difference between the two. When the cross entropy is equal to 0, the best state is reached, that is, the predicted value is completely consistent with the true value
$$
H(p,q) = 	- \sum{p(x) \log(q(x))}
$$
𝑝(𝑥) is the probability of the true distribution, 𝑞(𝑥) is the probability estimate calculated by the model through the data.

```python
x = tf.random.normal([1,784])
w = tf.random.normal([784,2])
b = tf.zeros([2])

logits = x@w + b
prob = tf.math.softmax(logits, axis=1)
tf.losses.categorical_crossentropy([0,1],prob)
```



