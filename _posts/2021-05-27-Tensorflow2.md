---
layout: post
title: "Tensorflow2"
date: 2021-05-27 
description: "The tutorial of tensorflow"
tag: tensorflow
---

## 1. 张量数据结构

程序 = 数据结构+算法。

TensorFlow程序 = 张量数据结构 + 计算图算法语言

张量和计算图是 TensorFlow的核心概念。

Tensorflow的基本数据结构是张量Tensor。张量即多维数组。Tensorflow的张量和numpy中的array很类似。

从行为特性来看，有两种类型的张量，常量constant和变量Variable.

常量的值在计算图中不可以被重新赋值，变量可以在计算图中用assign等算子重新赋值。

### 1.1 常量张量

张量的数据类型和numpy.array基本一一对应。

```
import numpy as np
import tensorflow as tf

i = tf.constant(1) # tf.int32 类型常量
l = tf.constant(1,dtype = tf.int64) # tf.int64 类型常量
f = tf.constant(1.23) #tf.float32 类型常量
d = tf.constant(3.14,dtype = tf.double) # tf.double 类型常量
s = tf.constant("hello world") # tf.string类型常量
b = tf.constant(True) #tf.bool类型常量


print(tf.int64 == np.int64) 
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode) # tf.string类型和np.unicode类型不等价
True
True
True
False
```

不同类型的数据可以用不同维度(rank)的张量来表示。

标量为0维张量，向量为1维张量，矩阵为2维张量。

彩色图像有rgb三个通道，可以表示为3维张量。

视频还有时间维，可以表示为4维张量。

可以简单地总结为：有几层中括号，就是多少维的张量。

```
scalar = tf.constant(True)  #标量，0维张量

print(tf.rank(scalar))
print(scalar.numpy().ndim)  # tf.rank的作用和numpy的ndim方法相同
tf.Tensor(0, shape=(), dtype=int32)
0
vector = tf.constant([1.0,2.0,3.0,4.0]) #向量，1维张量

print(tf.rank(vector))
print(np.ndim(vector.numpy()))
tf.Tensor(1, shape=(), dtype=int32)
1
matrix = tf.constant([[1.0,2.0],[3.0,4.0]]) #矩阵, 2维张量

print(tf.rank(matrix).numpy())
print(np.ndim(matrix))
2
2
tensor3 = tf.constant([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])  # 3维张量
print(tensor3)
print(tf.rank(tensor3))
tf.Tensor(
[[[1. 2.]
  [3. 4.]]

 [[5. 6.]
  [7. 8.]]], shape=(2, 2, 2), dtype=float32)
tf.Tensor(3, shape=(), dtype=int32)
tensor4 = tf.constant([[[[1.0,1.0],[2.0,2.0]],[[3.0,3.0],[4.0,4.0]]],
                        [[[5.0,5.0],[6.0,6.0]],[[7.0,7.0],[8.0,8.0]]]])  # 4维张量
print(tensor4)
print(tf.rank(tensor4))
tf.Tensor(
[[[[1. 1.]
   [2. 2.]]

  [[3. 3.]
   [4. 4.]]]


 [[[5. 5.]
   [6. 6.]]

  [[7. 7.]
   [8. 8.]]]], shape=(2, 2, 2, 2), dtype=float32)
tf.Tensor(4, shape=(), dtype=int32)
```

可以用tf.cast改变张量的数据类型。

可以用numpy方法将tensorflow中的张量转化成numpy中的张量。

可以用shape方法查看张量的尺寸。

```
h = tf.constant([123,456],dtype = tf.int32)
f = tf.cast(h,tf.float32)
print(h.dtype, f.dtype)
<dtype: 'int32'> <dtype: 'float32'>
y = tf.constant([[1.0,2.0],[3.0,4.0]])
print(y.numpy()) #转换成np.array
print(y.shape)
[[1. 2.]
 [3. 4.]]
(2, 2)
u = tf.constant(u"你好 世界")
print(u.numpy())  
print(u.numpy().decode("utf-8"))
b'\xe4\xbd\xa0\xe5\xa5\xbd \xe4\xb8\x96\xe7\x95\x8c'
你好 世界
```

### 1.2 变量张量

模型中需要被训练的参数一般被设置成变量。

```
# 常量值不可以改变，常量的重新赋值相当于创造新的内存空间
c = tf.constant([1.0,2.0])
print(c)
print(id(c))
c = c + tf.constant([1.0,1.0])
print(c)
print(id(c))
tf.Tensor([1. 2.], shape=(2,), dtype=float32)
5276289568
tf.Tensor([2. 3.], shape=(2,), dtype=float32)
5276290240
# 变量的值可以改变，可以通过assign, assign_add等方法给变量重新赋值
v = tf.Variable([1.0,2.0],name = "v")
print(v)
print(id(v))
v.assign_add([1.0,1.0])
print(v)
print(id(v))
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
5276259888
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>
5276259888
```

## 2. 三种计算图

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph.

在TensorFlow1.0时代，采用的是静态计算图，需要先使用TensorFlow的各种算子创建计算图，然后再开启一个会话Session，显式执行计算图。

而在TensorFlow2.0时代，采用的是动态计算图，即每使用一个算子后，该算子会被动态加入到隐含的默认计算图中立即执行得到结果，而无需开启Session。

使用动态计算图即Eager Excution的好处是方便调试程序，它会让TensorFlow代码的表现和Python原生代码的表现一样，写起来就像写numpy一样，各种日志打印，控制流全部都是可以使用的。

使用动态计算图的缺点是运行效率相对会低一些。因为使用动态图会有许多次Python进程和TensorFlow的C++进程之间的通信。而静态计算图构建完成之后几乎全部在TensorFlow内核上使用C++代码执行，效率更高。此外静态图会对计算步骤进行一定的优化，剪去和结果无关的计算步骤。

如果需要在TensorFlow2.0中使用静态图，可以使用@tf.function装饰器将普通Python函数转换成对应的TensorFlow计算图构建代码。运行该函数就相当于在TensorFlow1.0中用Session执行代码。使用tf.function构建静态图的方式叫做 Autograph.

### 2.1 计算图简介

计算图由节点(nodes)和线(edges)组成。

节点表示操作符Operator，或者称之为算子，线表示计算间的依赖。

实线表示有数据传递依赖，传递的数据即张量。

虚线通常可以表示控制依赖，即执行先后顺序。

[![img](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/raw/master/data/strjoin_graph.png)](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/data/strjoin_graph.png)

### 2.2 静态计算图

在TensorFlow1.0中，使用静态计算图分两步，第一步定义计算图，第二步在会话中执行计算图。

**TensorFlow 1.0静态计算图范例**

```
import tensorflow as tf

#定义计算图
g = tf.Graph()
with g.as_default():
    #placeholder为占位符，执行会话时候指定填充对象
    x = tf.placeholder(name='x', shape=[], dtype=tf.string)  
    y = tf.placeholder(name='y', shape=[], dtype=tf.string)
    z = tf.string_join([x,y],name = 'join',separator=' ')

#执行计算图
with tf.Session(graph = g) as sess:
    print(sess.run(fetches = z,feed_dict = {x:"hello",y:"world"}))
   
```

**TensorFlow2.0 怀旧版静态计算图**

TensorFlow2.0为了确保对老版本tensorflow项目的兼容性，在tf.compat.v1子模块中保留了对TensorFlow1.0那种静态计算图构建风格的支持。

可称之为怀旧版静态计算图，已经不推荐使用了。

```
import tensorflow as tf

g = tf.compat.v1.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(name='x', shape=[], dtype=tf.string)
    y = tf.compat.v1.placeholder(name='y', shape=[], dtype=tf.string)
    z = tf.strings.join([x,y],name = "join",separator = " ")

with tf.compat.v1.Session(graph = g) as sess:
    # fetches的结果非常像一个函数的返回值，而feed_dict中的占位符相当于函数的参数序列。
    result = sess.run(fetches = z,feed_dict = {x:"hello",y:"world"})
    print(result)
b'hello world'
```

### 2.3 动态计算图

在TensorFlow2.0中，使用的是动态计算图和Autograph.

在TensorFlow1.0中，使用静态计算图分两步，第一步定义计算图，第二步在会话中执行计算图。

动态计算图已经不区分计算图的定义和执行了，而是定义后立即执行。因此称之为 Eager Excution.

Eager这个英文单词的原意是"迫不及待的"，也就是立即执行的意思。

```
# 动态计算图在每个算子处都进行构建，构建后立即执行

x = tf.constant("hello")
y = tf.constant("world")
z = tf.strings.join([x,y],separator=" ")

tf.print(z)
hello world
# 可以将动态计算图代码的输入和输出关系封装成函数

def strjoin(x,y):
    z =  tf.strings.join([x,y],separator = " ")
    tf.print(z)
    return z

result = strjoin(tf.constant("hello"),tf.constant("world"))
print(result)
hello world
tf.Tensor(b'hello world', shape=(), dtype=string)
```

### 2.4 TensorFlow2.0的Autograph

动态计算图运行效率相对较低。

可以用@tf.function装饰器将普通Python函数转换成和TensorFlow1.0对应的静态计算图构建代码。

在TensorFlow1.0中，使用计算图分两步，第一步定义计算图，第二步在会话中执行计算图。

在TensorFlow2.0中，如果采用Autograph的方式使用计算图，第一步定义计算图变成了定义函数，第二步执行计算图变成了调用函数。

不需要使用会话了，一些都像原始的Python语法一样自然。

实践中，我们一般会先用动态计算图调试代码，然后在需要提高性能的的地方利用@tf.function切换成Autograph获得更高的效率。

当然，@tf.function的使用需要遵循一定的规范，我们后面章节将重点介绍。

```
import tensorflow as tf

# 使用autograph构建静态图

@tf.function
def strjoin(x,y):
    z =  tf.strings.join([x,y],separator = " ")
    tf.print(z)
    return z

result = strjoin(tf.constant("hello"),tf.constant("world"))

print(result)
hello world
tf.Tensor(b'hello world', shape=(), dtype=string)
import datetime

# 创建日志
import os
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('data', 'autograph', stamp)

## 在 Python3 下建议使用 pathlib 修正各操作系统的路径
# from pathlib import Path
# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = str(Path('./data/autograph/' + stamp))

writer = tf.summary.create_file_writer(logdir)

#开启autograph跟踪
tf.summary.trace_on(graph=True, profiler=True) 

#执行autograph
result = strjoin("hello","world")

#将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name="autograph",
        step=0,
        profiler_outdir=logdir)
#启动 tensorboard在jupyter中的魔法命令
%load_ext tensorboard
#启动tensorboard
%tensorboard --logdir ./data/autograph/
```

[![img](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/raw/master/data/2-2-tensorboard%E8%AE%A1%E7%AE%97%E5%9B%BE.jpg)](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/data/2-2-tensorboard计算图.jpg)

## 3. 自动微分机制

神经网络通常依赖反向传播求梯度来更新网络参数，求梯度过程通常是一件非常复杂而容易出错的事情。

而深度学习框架可以帮助我们自动地完成这种求梯度运算。

Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值。

这种利用tf.GradientTape求微分的方法叫做Tensorflow的自动微分机制。

### 3.1 利用梯度磁带求导数

```
import tensorflow as tf
import numpy as np 

# f(x) = a*x**2 + b*x + c的导数

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a*tf.pow(x,2) + b*x + c
    
dy_dx = tape.gradient(y,x)
print(dy_dx)
tf.Tensor(-2.0, shape=(), dtype=float32)
# 对常量张量也可以求导，需要增加watch

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a*tf.pow(x,2) + b*x + c
    
dy_dx,dy_da,dy_db,dy_dc = tape.gradient(y,[x,a,b,c])
print(dy_da)
print(dy_dc)
tf.Tensor(0.0, shape=(), dtype=float32)
tf.Tensor(1.0, shape=(), dtype=float32)
# 可以求二阶导数
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:   
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape1.gradient(y,x)   
dy2_dx2 = tape2.gradient(dy_dx,x)

print(dy2_dx2)
tf.Tensor(2.0, shape=(), dtype=float32)
# 可以在autograph中使用

@tf.function
def f(x):   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    # 自变量转换成tf.float32
    x = tf.cast(x,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a*tf.pow(x,2)+b*x+c
    dy_dx = tape.gradient(y,x) 
    
    return((dy_dx,y))

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))
(-2, 1)
(0, 0)
```

### 3.2 利用梯度磁带和优化器求最小值

```
# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
    
tf.print("y =",y,"; x =",x)
y = 0 ; x = 0.999998569
# 求f(x) = a*x**2 + b*x + c的最小值
# 使用optimizer.minimize
# optimizer.minimize相当于先用tape求gradient,再apply_gradient

x = tf.Variable(0.0,name = "x",dtype = tf.float32)

#注意f()无参数
def f():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   
for _ in range(1000):
    optimizer.minimize(f,[x])   
    
tf.print("y =",f(),"; x =",x)
y = 0 ; x = 0.999998569
# 在autograph中完成最小值求解
# 使用optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    for _ in tf.range(1000): #注意autograph时使用tf.range(1000)而不是range(1000)
        with tf.GradientTape() as tape:
            y = a*tf.pow(x,2) + b*x + c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
        
    y = a*tf.pow(x,2) + b*x + c
    return y

tf.print(minimizef())
tf.print(x)
0
0.999998569
# 在autograph中完成最小值求解
# 使用optimizer.minimize

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   

@tf.function
def f():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

@tf.function
def train(epoch):  
    for _ in tf.range(epoch):  
        optimizer.minimize(f,[x])
    return(f())


tf.print(train(1000))
tf.print(x)
0
0.999998569
```
