---
layout: post
title: "Python Advanced"
date: 2021-02-05
description: "Python Advanced"
tag: Python
---

### `*args` 和 `**kwargs`

`*args`是用来发送一个非键值对的可变数量的参数列表给一个函数

```python
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')
```

`**kwargs` 允许你将不定长度的**键值对**, 作为参数传递给一个函数

```python
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))

>>> greet_me(name="yasoob")
name == yasoob
```

### 生成器(Generators)

生成器也是一种迭代器，但是你只能对其迭代一次。这是因为它们并没有把所有的值存在内存中，而是在运行时生成值。你通过遍历来使用它们，要么用一个“for”循环，要么将它们传递给任意可以进行迭代的函数和结构。大多数时候生成器是以函数来实现的。然而，它们并不返回一个值，而是`yield`(暂且译作“生出”)一个值。

```python
def fibon(n):
    a = b = 1
    for i in range(n):
        yield a
        a, b = b, a + b

# way1
gen = fibon(1000000)
print(next(gen))

# way2
for x in fibon(1000000):
    print(x)
  
# `iter`将根据一个可迭代对象返回一个迭代器对象
my_string = "Yasoob"
my_iter = iter(my_string)
next(my_iter)
```

### Map, Filter and Reduce

`Map`会将一个函数映射到一个输入列表的所有元素上, 规范：map(function_to_apply, list_of_inputs)

```python
# eg.1
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))

# eg.2
def multiply(x):
        return (x*x)
def add(x):
        return (x+x)

funcs = [multiply, add]
for i in range(5):
    value = map(lambda x: x(i), funcs)
    print(list(value))
```

`filter`过滤列表中的元素，并且返回一个由所有符合要求的元素所构成的列表，`符合要求`即函数映射到该元素时返回值为True

```python
number_list = range(-5, 5)
less_than_zero = filter(lambda x: x < 0, number_list)
print(list(less_than_zero))
```

当需要对一个列表进行一些计算并返回结果时，`Reduce` 是个非常有用的函数。举个例子，当你需要计算一个整数列表的乘积时

```python
from functools import reduce
product = reduce( (lambda x, y: x * y), [1, 2, 3, 4] )
```

### `set`(集合)数据结构

`set`(集合)是一个非常有用的数据结构。它与列表(`list`)的行为类似，区别在于`set`不能包含重复的值。

```python
# 重复的值
some_list = ['a', 'b', 'c', 'b', 'd', 'm', 'n', 'n']
duplicates = set([x for x in some_list if some_list.count(x) > 1])

# 交集
valid = set(['yellow', 'red', 'blue', 'green', 'black'])
input_set = set(['red', 'brown'])

# 差集
valid = set(['yellow', 'red', 'blue', 'green', 'black'])
input_set = set(['red', 'brown'])
print(input_set.difference(valid))
```

### 三元运算符

三元运算符通常在Python里被称为条件表达式，这些表达式基于真(true)/假(not)的条件判断。伪代码：

```python
# 如果条件为真，返回真 否则返回假
# condition_is_true if condition else condition_is_false

is_fat = True 
state = "fat" if is_fat else "not fat"
```

### 装饰器

```python
from functools import wraps
def a_new_decorator(a_func):
  	@wraps(a_func)
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        a_func()
        print("I am doing some boring work after executing a_func()")
    return wrapTheFunction
  
@a_new_decorator
def a_function_requiring_decoration():
    """Hey you! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")

a_function_requiring_decoration()
#outputs: I am doing some boring work before executing a_func()
#         I am the function which needs some decoration to remove my foul smell
#         I am doing some boring work after executing a_func()

#the @a_new_decorator is just a short way of saying:
a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)

print(a_function_requiring_decoration.__name__)
# Output: a_function_requiring_decoration
```

### `__slots__`魔法

使用`__slots__`来告诉Python不要使用字典，而且只给一个固定集合的属性分配空间

这里是一个使用与不使用`__slots__`的例子：

- 不使用 `__slots__`:

```python
class MyClass(object):
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = identifier
        self.set_up()
    # ...
```

- 使用 `__slots__`:

```python
class MyClass(object):
    __slots__ = ['name', 'identifier']
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = identifier
        self.set_up()
    # ...
```

第二段代码会为你的内存减轻负担。通过这个技巧，有些人已经看到内存占用率几乎40%~50%的减少。

### 虚拟环境(virtualenv)

使用`virtualenv`！针对每个程序创建独立（隔离）的Python环境，而不是在全局安装所依赖的模块。

要安装它，只需要在命令行中输入以下命令：

```
$ pip install virtualenv
```

最重要的命令是：

```
$ virtualenv myproject
$ source bin/activate
```

执行第一个命令在`myproject`文件夹创建一个隔离的virtualenv环境，第二个命令激活这个隔离的环境(`virtualenv`)。

在创建virtualenv时，你必须做出决定：这个virtualenv是使用系统全局的模块呢？还是只使用这个virtualenv内的模块。 默认情况下，virtualenv不会使用系统全局模块。

如果你想让你的virtualenv使用系统全局模块，请使用`--system-site-packages`参数创建你的virtualenv，例如：

```
virtualenv --system-site-packages mycoolproject
```

使用以下命令可以退出这个virtualenv:

```
$ deactivate
```

运行之后将恢复使用你系统全局的Python模块。

你可以使用`smartcd`来帮助你管理你的环境，当你切换目录时，它可以帮助你激活（activate）和退出（deactivate）你的virtualenv。

### 容器(`Collections`)

- defaultdict

与`dict`类型不同，你不需要检查**key**是否存在，所以我们能这样做：

```python
from collections import defaultdict

colours = (
    ('Yasoob', 'Yellow'),
    ('Ali', 'Blue'),
    ('Arham', 'Green'),
    ('Ali', 'Black'),
    ('Yasoob', 'Red'),
    ('Ahmed', 'Silver'),
)

favourite_colours = defaultdict(list)

for name, colour in colours:
    favourite_colours[name].append(colour)

print(favourite_colours)
```

- counter

Counter是一个计数器，它可以帮助我们针对某项数据进行计数。比如它可以用来计算每个人喜欢多少种颜色：

```python
from collections import Counter

colours = (
    ('Yasoob', 'Yellow'),
    ('Ali', 'Blue'),
    ('Arham', 'Green'),
    ('Ali', 'Black'),
    ('Yasoob', 'Red'),
    ('Ahmed', 'Silver'),
)

favs = Counter(name for name, colour in colours)
print(favs)

## 输出:
## Counter({
##     'Yasoob': 2,
##     'Ali': 2,
##     'Arham': 1,
##     'Ahmed': 1
##  })
```

- deque

deque提供了一个双端队列，你可以从头/尾两端添加或删除元素。

```python
from collections import deque
d = deque()

d = deque()
d.append('1')
d.append('2')
d.append('3')

# 从两端取出(pop)数据
d.popleft()
d.pop()

# 我们也可以限制这个列表的大小，当超出你设定的限制时, 最左边一端的数据将从队列中删除
d = deque(maxlen=30)

# 还可以从任一端扩展这个队列中的数据
d.extendleft([0])
d.extend([6,7,8])
```

- namedtuple

它把元组变成一个针对简单任务的容器。你不必使用整数索引来访问一个`namedtuples`的数据。你可以像字典(`dict`)一样访问`namedtuples`，但`namedtuples`是不可变的

```python
from collections import namedtuple

Animal = namedtuple('Animal', 'name age type')
perry = Animal(name="perry", age=31, type="cat")

print(perry)
## 输出: Animal(name='perry', age=31, type='cat')

print(perry[0])
print(perry.name)
## 输出: 'perry' 你可以既使用整数索引，也可以使用名称来访问`namedtuple`：
```

一个命名元组(`namedtuple`)有两个必需的参数。它们是元组名称和字段名称。

在上面的例子中，我们的元组名称是`Animal`，字段名称是'name'，'age'和'type'。

- enum.Enum （枚举类型）

Enums基本上是一种组织各种东西的方式。

```python
from collections import namedtuple
from enum import Enum

class Species(Enum):
    cat = 1
    dog = 2
    horse = 3
    aardvark = 4
    butterfly = 5
    owl = 6
    platypus = 7
    dragon = 8
    unicorn = 9
    # 依次类推

    # 但我们并不想关心同一物种的年龄，所以我们可以使用一个别名
    kitten = 1  # (译者注：幼小的猫咪)
    puppy = 2   # (译者注：幼小的狗狗)

Animal = namedtuple('Animal', 'name age type')
perry = Animal(name="Perry", age=31, type=Species.cat)
drogon = Animal(name="Drogon", age=4, type=Species.dragon)
tom = Animal(name="Tom", age=75, type=Species.cat)
charlie = Animal(name="Charlie", age=2, type=Species.kitten)

# 有三种方法访问枚举数据，例如以下方法都可以获取到'cat'的值：
Species(1)
Species['cat']
Species.cat
```

### 枚举

枚举(`enumerate`)是Python内置函数。它的用处很难在简单的一行中说明，但是大多数的新人，甚至一些高级程序员都没有意识到它。它允许我们遍历数据并自动计数，

下面是一个例子：

```python
for counter, value in enumerate(some_list):
    print(counter, value)
```

不只如此，`enumerate`也接受一些可选参数，这使它更有用。

```python
my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)

# 输出:
(1, 'apple')
(2, 'banana')
(3, 'grapes')
(4, 'pear')
```

上面这个可选参数允许我们定制从哪个数字开始枚举。
你还可以用来创建包含索引的元组列表， 例如：

```python
my_list = ['apple', 'banana', 'grapes', 'pear']
counter_list = list(enumerate(my_list, 1))
print(counter_list)
# 输出: [(1, 'apple'), (2, 'banana'), (3, 'grapes'), (4, 'pear')]
```

### 对象自省

自省(introspection)，在计算机编程领域里，是指在运行时来判断一个对象的类型的能力。

- `dir`

它是用于自省的最重要的函数之一。它返回一个列表，列出了一个对象所拥有的属性和方法。这里是一个例子：

```python
my_list = [1, 2, 3]
dir(my_list)
# Output: ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__',
# '__delslice__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
# '__getitem__', '__getslice__', '__gt__', '__hash__', '__iadd__', '__imul__',
# '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__',
# '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__',
# '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__str__',
# '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop',
# 'remove', 'reverse', 'sort']
```

- `type`和`id`

`type`函数返回一个对象的类型。举个例子：

```python
print(type(''))
# Output: <type 'str'>

print(type([]))
# Output: <type 'list'>

print(type({}))
# Output: <type 'dict'>

print(type(dict))
# Output: <type 'type'>

print(type(3))
# Output: <type 'int'>
```

`id()`函数返回任意不同种类对象的唯一ID，举个例子：

```python
name = "Yasoob"
print(id(name))
# Output: 139972439030304
```

- `inspect`模块

`inspect`模块也提供了许多有用的函数，来获取活跃对象的信息。比方说，你可以查看一个对象的成员，只需运行：

```python
import inspect
print(inspect.getmembers(str))
# Output: [('__add__', <slot wrapper '__add__' of ... ...
```

### 列表推导式（`list` comprehensions）

列表推导式（又称列表解析式）提供了一种简明扼要的方法来创建列表。
它的结构是在一个中括号里包含一个表达式，然后是一个`for`语句，然后是0个或多个`for`或者`if`语句。

```python
multiples = [i for i in range(30) if i % 3 is 0]
print(multiples)
# Output: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

# 需要使用for循环来生成一个新列表
squared = [x**2 for x in range(10)]
```

### 处理多个异常

我们可以使用三种方法来处理多个异常。

- 第一种方法需要把所有可能发生的异常放到一个元组里。像这样：

```python
try:
    file = open('test.txt', 'rb')
except (IOError, EOFError) as e:
    print("An error occurred. {}".format(e.args[-1]))
```

- 另外一种方式是对每个单独的异常在单独的`except`语句块中处理。我们想要多少个`except`语句块都可以。这里是个例子：

```python
try:
    file = open('test.txt', 'rb')
except EOFError as e:
    print("An EOF error occurred.")
    raise e
except IOError as e:
    print("An error occurred.")
    raise e
```

上面这个方式中，如果异常没有被第一个`except`语句块处理，那么它也许被下一个语句块处理，或者根本不会被处理。

- 现在，最后一种方式会捕获**所有**异常：

```python
try:
    file = open('test.txt', 'rb')
except Exception:
    # 打印一些异常日志，如果你想要的话
    raise
```

当你不知道你的程序会抛出什么样的异常时，上面的方式可能非常有帮助。

### `for - else`从句

`for`循环还有一个`else`从句，我们大多数人并不熟悉。这个`else`从句会在循环正常结束时执行。这意味着，循环没有遇到任何`break`. 一旦你掌握了何时何地使用它

有个常见的构造是跑一个循环，并查找一个元素。如果这个元素被找到了，我们使用`break`来中断这个循环。有两个场景会让循环停下来。 - 第一个是当一个元素被找到，`break`被触发。 - 第二个场景是循环结束。

现在我们也许想知道其中哪一个，才是导致循环完成的原因。一个方法是先设置一个标记，然后在循环结束时打上标记。另一个是使用`else`从句。

这就是`for/else`循环的基本结构：

```python
for item in container:
    if search_something(item):
        # Found it!
        process(item)
        break
else:
    # Didn't find anything..
    not_found_in_container()
```

### `open`函数

```python
import io

with open('photo.jpg', 'r+') as f:
    jpgdata = f.read()
    
with open('photo.jpg', 'rb') as inf:
    jpgdata = inf.read()

if jpgdata.startswith(b'\xff\xd8'):
    text = u'This is a JPEG file (%d bytes long)\n'
else:
    text = u'This is a random file (%d bytes long)\n'

with io.open('summary.txt', 'w', encoding='utf-8') as outf:
    outf.write(text % len(jpgdata))
```

`open`的第一个参数是文件名。第二个(`mode` 打开模式)决定了这个文件如何被打开。

- 如果你想读取文件，传入`r`
- 如果你想读取并写入文件，传入`r+`
- 如果你想覆盖写入文件，传入`w`
- 如果你想在文件末尾附加内容，传入`a`

一般来说，如果文件格式是由人写的，那么它更可能是文本模式。jpg图像文件一般不是人写的（而且其实不是人直接可读的），因此你应该以二进制模式来打开它们，方法是在`mode`字符串后加一个`b`(你可以看看开头的例子里，正确的方式应该是`rb`)。如果你以文本模式打开一些东西（比如，加一个`t`). 当你写入一个文件，你可以选一个你喜欢的编码（utf-8）

### 函数缓存 (Function caching)

函数缓存允许我们将一个函数对于给定参数的返回值缓存起来

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

>>> print([fib(n) for n in range(10)])
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

那个`maxsize`参数是告诉`lru_cache`，最多缓存最近多少个返回值。

我们也可以轻松地对返回值清空缓存，通过这样：

```python
fib.cache_clear()
```

### 上下文管理器(Context managers)

上下文管理器允许你在有需要的时候，精确地分配和释放资源。

使用上下文管理器最广泛的案例就是`with`语句了。
想象下你有两个需要结对执行的相关操作，然后还要在它们中间放置一段代码。
上下文管理器就是专门让你做这种事情的

```python
with open('some_file', 'w') as opened_file:
    opened_file.write('Hola!')
```

我们还可以用装饰器(decorators)和生成器(generators)来实现上下文管理器。
Python有个`contextlib`模块专门用于这个目的。我们可以使用一个生成器函数来实现一个上下文管理器，而不是使用一个类。

```python
from contextlib import contextmanager

@contextmanager
def open_file(name):
    f = open(name, 'w')
    yield f
    f.close()
```

让我们小小地剖析下这个方法。 1. Python解释器遇到了`yield`关键字。因为这个缘故它创建了一个生成器而不是一个普通的函数。 2. 因为这个装饰器，`contextmanager`会被调用并传入函数名（`open_file`）作为参数。 3. `contextmanager`函数返回一个以`GeneratorContextManager`对象封装过的生成器。 4. 这个`GeneratorContextManager`被赋值给`open_file`函数，我们实际上是在调用`GeneratorContextManager`对象。

那现在我们既然知道了所有这些，我们可以用这个新生成的上下文管理器了，像这样：

```python
with open_file('some_file') as f:
    f.write('hola!')
```
