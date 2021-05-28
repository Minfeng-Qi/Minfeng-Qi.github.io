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

## 4. API示范

下面的范例使用TensorFlow的中阶API实现线性回归模型和和DNN二分类模型。

TensorFlow的中阶API主要包括各种模型层，损失函数，优化器，数据管道，特征列等等。

TensorFlow的高阶API主要为tf.keras.models提供的模型的类接口。

使用Keras接口有以下3种方式构建模型：使用Sequential按层顺序构建模型，使用函数式API构建任意结构模型，继承Model基类构建自定义模型。

```
import tensorflow as tf

#打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)

    
```

### 4.1 线性回归模型

**1) 准备数据**

```
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import tensorflow as tf
from tensorflow.keras import layers,losses,metrics,optimizers

#样本数量
n = 400

# 生成测试用数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10) 
w0 = tf.constant([[2.0],[-3.0]])
b0 = tf.constant([[3.0]])
Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @表示矩阵乘法,增加正态扰动
# 数据可视化
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b")
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)

ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g")
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)
plt.show()
```

[![img](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/raw/master/data/3-2-01-%E5%9B%9E%E5%BD%92%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.png)](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/data/3-2-01-回归数据可视化.png)

```
#构建输入数据管道
ds = tf.data.Dataset.from_tensor_slices((X,Y)) \
     .shuffle(buffer_size = 100).batch(10) \
     .prefetch(tf.data.experimental.AUTOTUNE)  
```

**2) 定义模型**

```
# way1
model = layers.Dense(units = 1) 
model.build(input_shape = (2,)) #用build方法创建variables
model.loss_func = losses.mean_squared_error
model.optimizer = optimizers.SGD(learning_rate=0.001)

# way2
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(1,input_shape =(2,)))
model.summary()
```

**3) 训练模型**

```
#使用autograph机制转换成静态图加速

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
    grads = tape.gradient(loss,model.variables)
    model.optimizer.apply_gradients(zip(grads,model.variables))
    return loss

# 测试train_step效果
features,labels = next(ds.as_numpy_iterator())
train_step(model,features,labels)
def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        loss = tf.constant(0.0)
        for features, labels in ds:
            loss = train_step(model,features,labels)
        if epoch%50==0:
            printbar()
            tf.print("epoch =",epoch,"loss = ",loss)
            tf.print("w =",model.variables[0])
            tf.print("b =",model.variables[1])
train_model(model,epochs = 200)
================================================================================17:01:48
epoch = 50 loss =  2.56481647
w = [[1.99355531]
 [-2.99061537]]
b = [3.09484935]
================================================================================17:01:51
epoch = 100 loss =  5.96198225
w = [[1.98028314]
 [-2.96975136]]
b = [3.09501529]
================================================================================17:01:54
epoch = 150 loss =  4.79625702
w = [[2.00056171]
 [-2.98774862]]
b = [3.09567738]
================================================================================17:01:58
epoch = 200 loss =  8.26704407
w = [[2.00282311]
 [-2.99300027]]
b = [3.09406662]

# way2
model.compile(optimizer="adam",loss="mse",metrics=["mae"])
model.fit(X,Y,batch_size = 10,epochs = 200)  

tf.print("w = ",model.layers[0].kernel)
tf.print("b = ",model.layers[0].bias)


# 结果可视化

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

w,b = model.variables

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0],Y[:,0], c = "b",label = "samples")
ax1.plot(X[:,0],w[0]*X[:,0]+b[0],"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)



ax2 = plt.subplot(122)
ax2.scatter(X[:,1],Y[:,0], c = "g",label = "samples")
ax2.plot(X[:,1],w[1]*X[:,1]+b[0],"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()
```

[![img](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/raw/master/data/3-2-02-%E5%9B%9E%E5%BD%92%E7%BB%93%E6%9E%9C%E5%8F%AF%E8%A7%86%E5%8C%96.png)](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/data/3-2-02-回归结果可视化.png)

### 4.2  DNN二分类模型

**1) 准备数据**

```
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers,losses,metrics,optimizers
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + tf.random.truncated_normal([n_positive,1],0.0,1.0)
theta_p = tf.random.uniform([n_positive,1],0.0,2*np.pi) 
Xp = tf.concat([r_p*tf.cos(theta_p),r_p*tf.sin(theta_p)],axis = 1)
Yp = tf.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + tf.random.truncated_normal([n_negative,1],0.0,1.0)
theta_n = tf.random.uniform([n_negative,1],0.0,2*np.pi) 
Xn = tf.concat([r_n*tf.cos(theta_n),r_n*tf.sin(theta_n)],axis = 1)
Yn = tf.zeros_like(r_n)

#汇总样本
X = tf.concat([Xp,Xn],axis = 0)
Y = tf.concat([Yp,Yn],axis = 0)


#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
plt.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
plt.legend(["positive","negative"]);
```

[![img](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/raw/master/data/3-1-03-%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.png)](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/data/3-1-03-分类数据可视化.png)

```
#构建输入数据管道
ds = tf.data.Dataset.from_tensor_slices((X,Y)) \
     .shuffle(buffer_size = 4000).batch(100) \
     .prefetch(tf.data.experimental.AUTOTUNE) 
```

**2) 定义模型**

```
class DNNModel(tf.Module):
    def __init__(self,name = None):
        super(DNNModel, self).__init__(name=name)
        self.dense1 = layers.Dense(4,activation = "relu") 
        self.dense2 = layers.Dense(8,activation = "relu")
        self.dense3 = layers.Dense(1,activation = "sigmoid")

     
    # 正向传播
    @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def __call__(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y
    
model = DNNModel()
model.loss_func = losses.binary_crossentropy
model.metric_func = metrics.binary_accuracy
model.optimizer = optimizers.Adam(learning_rate=0.001)
# 测试模型结构
(features,labels) = next(ds.as_numpy_iterator())

predictions = model(features)

loss = model.loss_func(tf.reshape(labels,[-1]),tf.reshape(predictions,[-1]))
metric = model.metric_func(tf.reshape(labels,[-1]),tf.reshape(predictions,[-1]))

tf.print("init loss:",loss)
tf.print("init metric",metric)
init loss: 1.13653195
init metric 0.5
```

**3) 训练模型**

```
#使用autograph机制转换成静态图加速

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = model.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
    grads = tape.gradient(loss,model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads,model.trainable_variables))
    
    metric = model.metric_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
    
    return loss,metric

# 测试train_step效果
features,labels = next(ds.as_numpy_iterator())
train_step(model,features,labels)
(<tf.Tensor: shape=(), dtype=float32, numpy=1.2033114>,
 <tf.Tensor: shape=(), dtype=float32, numpy=0.47>)
def train_model(model,epochs):
    for epoch in tf.range(1,epochs+1):
        loss, metric = tf.constant(0.0),tf.constant(0.0)
        for features, labels in ds:
            loss,metric = train_step(model,features,labels)
        if epoch%10==0:
            printbar()
            tf.print("epoch =",epoch,"loss = ",loss, "accuracy = ",metric)
train_model(model,epochs = 60)
================================================================================17:07:36
epoch = 10 loss =  0.556449413 accuracy =  0.79
================================================================================17:07:38
epoch = 20 loss =  0.439187407 accuracy =  0.86
================================================================================17:07:40
epoch = 30 loss =  0.259921253 accuracy =  0.95
================================================================================17:07:42
epoch = 40 loss =  0.244920313 accuracy =  0.9
================================================================================17:07:43
epoch = 50 loss =  0.19839409 accuracy =  0.92
================================================================================17:07:45
epoch = 60 loss =  0.126151696 accuracy =  0.95
# 结果可视化
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
ax1.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = tf.boolean_mask(X,tf.squeeze(model(X)>=0.5),axis = 0)
Xn_pred = tf.boolean_mask(X,tf.squeeze(model(X)<0.5),axis = 0)

ax2.scatter(Xp_pred[:,0].numpy(),Xp_pred[:,1].numpy(),c = "r")
ax2.scatter(Xn_pred[:,0].numpy(),Xn_pred[:,1].numpy(),c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred");
```

[![img](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/raw/master/data/3-2-04-%E5%88%86%E7%B1%BB%E7%BB%93%E6%9E%9C%E5%8F%AF%E8%A7%86%E5%8C%96.png)](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/blob/master/data/3-2-04-分类结果可视化.png)

## 5. 张量的结构操作

张量的操作主要包括张量的结构操作和张量的数学运算。

张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。

本篇我们介绍张量的结构操作。

### 5.1 创建张量

张量创建的许多方法和numpy中创建array的方法很像。

```
import tensorflow as tf
import numpy as np 
a = tf.constant([1,2,3],dtype = tf.float32)
tf.print(a)
[1 2 3]
b = tf.range(1,10,delta = 2)
tf.print(b)
[1 3 5 7 9]
c = tf.linspace(0.0,2*3.14,100)
tf.print(c)
[0 0.0634343475 0.126868695 ... 6.15313148 6.21656609 6.28]
d = tf.zeros([3,3])
tf.print(d)
[[0 0 0]
 [0 0 0]
 [0 0 0]]
a = tf.ones([3,3])
b = tf.zeros_like(a,dtype= tf.float32)
tf.print(a)
tf.print(b)
[[1 1 1]
 [1 1 1]
 [1 1 1]]
[[0 0 0]
 [0 0 0]
 [0 0 0]]
b = tf.fill([3,2],5)
tf.print(b)
[[5 5]
 [5 5]
 [5 5]]
#均匀分布随机
tf.random.set_seed(1.0)
a = tf.random.uniform([5],minval=0,maxval=10)
tf.print(a)
[1.65130854 9.01481247 6.30974197 4.34546089 2.9193902]
#正态分布随机
b = tf.random.normal([3,3],mean=0.0,stddev=1.0)
tf.print(b)
[[0.403087884 -1.0880208 -0.0630953535]
 [1.33655667 0.711760104 -0.489286453]
 [-0.764221311 -1.03724861 -1.25193381]]
#正态分布随机，剔除2倍方差以外数据重新生成
c = tf.random.truncated_normal((5,5), mean=0.0, stddev=1.0, dtype=tf.float32)
tf.print(c)
[[-0.457012236 -0.406867266 0.728577733 -0.892977774 -0.369404584]
 [0.323488563 1.19383323 0.888299048 1.25985599 -1.95951891]
 [-0.202244401 0.294496894 -0.468728036 1.29494202 1.48142183]
 [0.0810953453 1.63843894 0.556645 0.977199793 -1.17777884]
 [1.67368948 0.0647980496 -0.705142677 -0.281972528 0.126546144]]
# 特殊矩阵
I = tf.eye(3,3) #单位矩阵
tf.print(I)
tf.print(" ")
t = tf.linalg.diag([1,2,3]) #对角阵
tf.print(t)
[[1 0 0]
 [0 1 0]
 [0 0 1]]
 
[[1 0 0]
 [0 2 0]
 [0 0 3]]
```

### 5.2 索引切片

张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。

对于tf.Variable,可以通过索引和切片对部分元素进行修改。

对于提取张量的连续子区域，也可以使用tf.slice.

此外，对于不规则的切片提取,可以使用tf.gather,tf.gather_nd,tf.boolean_mask。

tf.boolean_mask功能最为强大，它可以实现tf.gather,tf.gather_nd的功能，并且tf.boolean_mask还可以实现布尔索引。

如果要通过修改张量的某些元素得到新的张量，可以使用tf.where，tf.scatter_nd。

```
tf.random.set_seed(3)
t = tf.random.uniform([5,5],minval=0,maxval=10,dtype=tf.int32)
tf.print(t)
[[4 7 4 2 9]
 [9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]
 [3 7 0 0 3]]
#第0行
tf.print(t[0])
[4 7 4 2 9]
#倒数第一行
tf.print(t[-1])
[3 7 0 0 3]
#第1行第3列
tf.print(t[1,3])
tf.print(t[1][3])
4
4
#第1行至第3行
tf.print(t[1:4,:])
tf.print(tf.slice(t,[1,0],[3,5])) #tf.slice(input,begin_vector,size_vector)
[[9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]]
[[9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]]
#第1行至最后一行，第0列到最后一列每隔两列取一列
tf.print(t[1:4,:4:2])
[[9 2]
 [7 7]
 [9 9]]
#对变量来说，还可以使用索引和切片修改部分元素
x = tf.Variable([[1,2],[3,4]],dtype = tf.float32)
x[1,:].assign(tf.constant([0.0,0.0]))
tf.print(x)
[[1 2]
 [0 0]]
a = tf.random.uniform([3,3,3],minval=0,maxval=10,dtype=tf.int32)
tf.print(a)
[[[7 3 9]
  [9 0 7]
  [9 6 7]]

 [[1 3 3]
  [0 8 1]
  [3 1 0]]

 [[4 0 6]
  [6 2 2]
  [7 9 5]]]
#省略号可以表示多个冒号
tf.print(a[...,1])
[[3 0 6]
 [3 8 1]
 [0 2 9]]
```

以上切片方式相对规则，对于不规则的切片提取,可以使用tf.gather,tf.gather_nd,tf.boolean_mask。

考虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4×10×7的张量来表示。

```
scores = tf.random.uniform((4,10,7),minval=0,maxval=100,dtype=tf.int32)
tf.print(scores)
[[[52 82 66 ... 17 86 14]
  [8 36 94 ... 13 78 41]
  [77 53 51 ... 22 91 56]
  ...
  [11 19 26 ... 89 86 68]
  [60 72 0 ... 11 26 15]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [83 36 31 ... 75 38 85]
  [54 26 67 ... 60 68 98]
  ...
  [20 5 18 ... 32 45 3]
  [72 52 81 ... 88 41 20]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [78 71 54 ... 43 98 81]
  [21 66 53 ... 97 75 77]
  ...
  [6 74 3 ... 53 65 43]
  [98 36 72 ... 33 36 81]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [35 8 82 ... 11 59 97]
  [44 6 99 ... 81 60 27]
  ...
  [76 26 35 ... 51 8 17]
  [33 52 53 ... 78 37 31]
  [71 27 44 ... 0 52 16]]]
#抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
p = tf.gather(scores,[0,5,9],axis=1)
tf.print(p)
[[[52 82 66 ... 17 86 14]
  [24 80 70 ... 72 63 96]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [46 10 94 ... 23 18 92]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [19 12 23 ... 87 86 25]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [6 41 79 ... 97 43 13]
  [71 27 44 ... 0 52 16]]]
#抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = tf.gather(tf.gather(scores,[0,5,9],axis=1),[1,3,6],axis=2)
tf.print(q)
[[[82 55 14]
  [80 46 96]
  [99 58 74]]

 [[73 48 81]
  [10 38 92]
  [21 86 90]]

 [[80 57 60]
  [12 34 25]
  [78 71 21]]

 [[57 75 3]
  [41 47 13]
  [27 96 16]]]
# 抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
#indices的长度为采样样本的个数，每个元素为采样位置的坐标
s = tf.gather_nd(scores,indices = [(0,0),(2,4),(3,6)])
s
<tf.Tensor: shape=(3, 7), dtype=int32, numpy=
array([[52, 82, 66, 55, 17, 86, 14],
       [99, 94, 46, 70,  1, 63, 41],
       [46, 83, 70, 80, 90, 85, 17]], dtype=int32)>
```

以上tf.gather和tf.gather_nd的功能也可以用tf.boolean_mask来实现。

```
#抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
p = tf.boolean_mask(scores,[True,False,False,False,False,
                            True,False,False,False,True],axis=1)
tf.print(p)
[[[52 82 66 ... 17 86 14]
  [24 80 70 ... 72 63 96]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [46 10 94 ... 23 18 92]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [19 12 23 ... 87 86 25]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [6 41 79 ... 97 43 13]
  [71 27 44 ... 0 52 16]]]
#抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
s = tf.boolean_mask(scores,
    [[True,False,False,False,False,False,False,False,False,False],
     [False,False,False,False,False,False,False,False,False,False],
     [False,False,False,False,True,False,False,False,False,False],
     [False,False,False,False,False,False,True,False,False,False]])
tf.print(s)
[[52 82 66 ... 17 86 14]
 [99 94 46 ... 1 63 41]
 [46 83 70 ... 90 85 17]]
#利用tf.boolean_mask可以实现布尔索引

#找到矩阵中小于0的元素
c = tf.constant([[-1,1,-1],[2,2,-2],[3,-3,3]],dtype=tf.float32)
tf.print(c,"\n")

tf.print(tf.boolean_mask(c,c<0),"\n") 
tf.print(c[c<0]) #布尔索引，为boolean_mask的语法糖形式
[[-1 1 -1]
 [2 2 -2]
 [3 -3 3]] 

[-1 -1 -2 -3] 

[-1 -1 -2 -3]
```

以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。

如果要通过修改张量的部分元素值得到新的张量，可以使用tf.where和tf.scatter_nd。

tf.where可以理解为if的张量版本，此外它还可以用于找到满足条件的所有元素的位置坐标。

tf.scatter_nd的作用和tf.gather_nd有些相反，tf.gather_nd用于收集张量的给定位置的元素，

而tf.scatter_nd可以将某些值插入到一个给定shape的全0的张量的指定位置处。

```
#找到张量中小于0的元素,将其换成np.nan得到新的张量
#tf.where和np.where作用类似，可以理解为if的张量版本

c = tf.constant([[-1,1,-1],[2,2,-2],[3,-3,3]],dtype=tf.float32)
d = tf.where(c<0,tf.fill(c.shape,np.nan),c) 
d
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[nan,  1., nan],
       [ 2.,  2., nan],
       [ 3., nan,  3.]], dtype=float32)>
#如果where只有一个参数，将返回所有满足条件的位置坐标
indices = tf.where(c<0)
indices
<tf.Tensor: shape=(4, 2), dtype=int64, numpy=
array([[0, 0],
       [0, 2],
       [1, 2],
       [2, 1]])>
#将张量的第[0,0]和[2,1]两个位置元素替换为0得到新的张量
d = c - tf.scatter_nd([[0,0],[2,1]],[c[0,0],c[2,1]],c.shape)
d
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 0.,  1., -1.],
       [ 2.,  2., -2.],
       [ 3.,  0.,  3.]], dtype=float32)>
#scatter_nd的作用和gather_nd有些相反
#可以将某些值插入到一个给定shape的全0的张量的指定位置处。
indices = tf.where(c<0)
tf.scatter_nd(indices,tf.gather_nd(c,indices),c.shape)
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[-1.,  0., -1.],
       [ 0.,  0., -2.],
       [ 0., -3.,  0.]], dtype=float32)>
```

### 5.3 维度变换

维度变换相关函数主要有 tf.reshape, tf.squeeze, tf.expand_dims, tf.transpose.

tf.reshape 可以改变张量的形状。

tf.squeeze 可以减少维度。

tf.expand_dims 可以增加维度。

tf.transpose 可以交换维度。

tf.reshape可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序，所以，该操作实际上非常迅速，并且是可逆的。

```
a = tf.random.uniform(shape=[1,3,3,2],
                      minval=0,maxval=255,dtype=tf.int32)
tf.print(a.shape)
tf.print(a)
TensorShape([1, 3, 3, 2])
[[[[135 178]
   [26 116]
   [29 224]]

  [[179 219]
   [153 209]
   [111 215]]

  [[39 7]
   [138 129]
   [59 205]]]]
# 改成 （3,6）形状的张量
b = tf.reshape(a,[3,6])
tf.print(b.shape)
tf.print(b)
TensorShape([3, 6])
[[135 178 26 116 29 224]
 [179 219 153 209 111 215]
 [39 7 138 129 59 205]]
# 改回成 [1,3,3,2] 形状的张量
c = tf.reshape(b,[1,3,3,2])
tf.print(c)
[[[[135 178]
   [26 116]
   [29 224]]

  [[179 219]
   [153 209]
   [111 215]]

  [[39 7]
   [138 129]
   [59 205]]]]
```

如果张量在某个维度上只有一个元素，利用tf.squeeze可以消除这个维度。

和tf.reshape相似，它本质上不会改变张量元素的存储顺序。

张量的各个元素在内存中是线性存储的，其一般规律是，同一层级中的相邻元素的物理地址也相邻。

```
s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)
TensorShape([3, 3, 2])
[[[135 178]
  [26 116]
  [29 224]]

 [[179 219]
  [153 209]
  [111 215]]

 [[39 7]
  [138 129]
  [59 205]]]
d = tf.expand_dims(s,axis=0) #在第0维插入长度为1的一个维度
d
<tf.Tensor: shape=(1, 3, 3, 2), dtype=int32, numpy=
array([[[[135, 178],
         [ 26, 116],
         [ 29, 224]],

        [[179, 219],
         [153, 209],
         [111, 215]],

        [[ 39,   7],
         [138, 129],
         [ 59, 205]]]], dtype=int32)>
```

tf.transpose可以交换张量的维度，与tf.reshape不同，它会改变张量元素的存储顺序。

tf.transpose常用于图片存储格式的变换上。

```
# Batch,Height,Width,Channel
a = tf.random.uniform(shape=[100,600,600,4],minval=0,maxval=255,dtype=tf.int32)
tf.print(a.shape)

# 转换成 Channel,Height,Width,Batch
s= tf.transpose(a,perm=[3,1,2,0])
tf.print(s.shape)
TensorShape([100, 600, 600, 4])
TensorShape([4, 600, 600, 100])
```

### 5.4 合并分割

和numpy类似，可以用tf.concat和tf.stack方法对多个张量进行合并，可以用tf.split方法把一个张量分割成多个张量。

tf.concat和tf.stack有略微的区别，tf.concat是连接，不会增加维度，而tf.stack是堆叠，会增加维度。

```
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
c = tf.constant([[9.0,10.0],[11.0,12.0]])

tf.concat([a,b,c],axis = 0)
<tf.Tensor: shape=(6, 2), dtype=float32, numpy=
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.],
       [ 7.,  8.],
       [ 9., 10.],
       [11., 12.]], dtype=float32)>
tf.concat([a,b,c],axis = 1)
<tf.Tensor: shape=(2, 6), dtype=float32, numpy=
array([[ 1.,  2.,  5.,  6.,  9., 10.],
       [ 3.,  4.,  7.,  8., 11., 12.]], dtype=float32)>
tf.stack([a,b,c])
<tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]],

       [[ 9., 10.],
        [11., 12.]]], dtype=float32)>
tf.stack([a,b,c],axis=1)
<tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 5.,  6.],
        [ 9., 10.]],

       [[ 3.,  4.],
        [ 7.,  8.],
        [11., 12.]]], dtype=float32)>
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
c = tf.constant([[9.0,10.0],[11.0,12.0]])

c = tf.concat([a,b,c],axis = 0)
```

tf.split是tf.concat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。

```
#tf.split(value,num_or_size_splits,axis)
tf.split(c,3,axis = 0)  #指定分割份数，平均分割
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
tf.split(c,[2,2,2],axis = 0) #指定每份的记录数量
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
```
