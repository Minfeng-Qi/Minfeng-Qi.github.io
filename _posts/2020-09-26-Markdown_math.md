---
layout: post
title: "Markdown Math"
date: 2020-09-26 
description: "Markdown_math"
tag: Markdown
---

## Markdown 数学公式

### 数学公式起始和结尾标志

数学公式以 `$` 开头和结尾，例如: `\$f(x) = x^2 + 1\$` 显示为: \$f(x) = x^2 + 1\$

如果需要独占一行的话，则以 `$$` 开头和结尾。 例如: `$$f(x) = a + bx$$` 显示为:
$$
f(x) = a + bx
$$


------

### 符号上标和下标

上表用 `^` 表示，下标用 `_` 表示。

例如 `\$f(x) = a_0 + a_1 * x + a_2 * x^2\$` 显示为: 
$$
f(x) = a_0 + a_1 * x + a_2 * x^2
$$


------

### 大括号

使用 `\left` 和 `\right` 命令作为 () ，[] 以及的前缀，可以显示大括号，效果如下:
$$
f(x)=x2+\left(y2+\frac{a}{b}\right)
$$


### 多行公式

用 \begin 和 \end 把公式包围起来（支持嵌套），每行 `\\`结尾，每个元素 `&` 分隔。

公式对齐写法如下:

```
$$
\begin{align}
  f(x) = a + b \\
       = c + d  \\
\end{align}
$$
```

效果如下:
$$
\begin{align}
  f(x) = a + b \\
       = c + d  \\
\end{align}
$$

### 多值函数

使用 `cases` 块表达式，每行 `\\`结尾，每个元素 `&` 分隔。

```
$$
p(x) = 
\begin{cases}
  p, & x = 1 \\
  1 - p, & x = 0
\end{cases}
$$
```

$$
p(x) = 
\begin{cases}
  p, & x = 1 \\
  1 - p, & x = 0
\end{cases}
$$



### 矩阵

使用 `\begin{matrix}`开头及`\end{matrix}`结尾，每行 `\\`结尾，每个元素 `&`分隔。
$$
\begin{matrix}
  1 & 0 & 0 \\
  0 & 1 & 0 \\
  0 & 0 & 1 \\
\end{matrix}
$$


```
$$
\begin{matrix}
  1 & 0 & 0 \\
  0 & 1 & 0 \\
  0 & 0 & 1 \\
\end{matrix}
$$
```

------

- 常用符号

| 写法             |         符号         | 备注           |
| :--------------- | :------------------: | :------------- |
| \sin(x)          |     \$\sin(x)\$      | 正弦函数       |
| \log(x)          |     \$\log(x)\$      | 对数函数       |
| \sum_{i=0}^n     |   \$\sum_{i=0}^n\$   | 累加和         |
| \prod_{i=0}^n    |  \$\prod_{i=0}^n\$   | 累积乘         |
| \displaystyle    |  \$\displaystyle\$   | 块显示         |
| \ldots           |     \$\ldots\\$      | 底部省略号     |
| \cdots           |      \$\cdots$       | 中部省略号     |
| \int_a^b         |     \$\int_a^b\$     | 积分符号       |
| \lim             |       \$\lim\$       | 极限函数       |
| \to              |          →           | 箭头           |
| \vec{a}          |          a⃗           | 矢量a          |
| 90^\circ         |     \$90^\circ\$     | 度数的圆圈     |
| \uparrow         |          ↑           | 上箭头         |
| \Uparrow         |          ⇑           | 双上箭头       |
| \partial y       |          ∂y          | 导数/偏导      |
| \infty           |          ∞           | 无穷           |
| \Pi              |         Πi=0         | 累乘           |
| \sqrt{x}         |     \$\sqrt{x}\$     | 求平方根       |
| \overline{a+b}   |  \$\overline{a+b}\$  | 上划线         |
| \underline{a+b}  | \$\underline{a+b}\$  | 下划线         |
| \overbrace{a+b}  | \$\overbrace{a+b}\$  | 上括号         |
| \underbrace{a+b} | \$\underbrace{a+b}\$ | 下括号         |
| \pm{a}{b}        |         ±ab          | 正负号         |
| \mp{a}{b}        |         ∓ab          | 负正号         |
| \times           |          ×           | 乘法           |
| \cdot            |          ⋅           | 点乘           |
| \ast             |          ∗           | 星乘           |
| \div             |          ÷           | 除法           |
| \frac{1}{5}      |   \$\frac{1}{5}\$    | 分数           |
| \drac{1}{5}      |   \$\drac{1}{5}\$    | 分数，字体更大 |
| \leq             |          ≤           | 小于等于       |
| \not             |          ⧸           | 非             |
| \geq             |          ≥           | 大于等于       |
| \neq             |          ≠           | 不等于         |
| \nleq            |          ≰           | 不小于等于     |
| \ngeq            |          ≱           | 不大于等于     |
| \sim             |          ∼           | 相关符号       |
| \approx          |          ≈           | 约等于         |
| \equiv           |          ≡           | 常等于/横等于  |
| \bigodot         |          ⨀           | 加运算符       |
| \bigotimes       |          ⨂           | 乘运算符       |

------

### 集合符号

| 写法         | 符号 | 备注   |
| ------------ | :--: | ------ |
| \in          |  ∈   | 属于   |
| \notin       |  ∉   | 不属于 |
| \subset      |  ⊂   | 真子集 |
| \not \subset |  ⊄   | 非子集 |
| \subseteq    |  ⊆   | 子集   |
| \supset      |  ⊃   | 超集   |
| \supseteq    |  ⊇   | 超集   |
| \cup         |  ∪   | 并集   |
| \cap         |  ∩   | 交集   |
| \mathbb{R}   |  ℝ   | 实数集 |
| \emptyset    |  ∅   | 空集   |

### 希腊符号

| 写法          | 符号 |
| :------------ | :--- |
| \alpha        | α    |
| \beta         | β    |
| \gamma        | γ    |
| \Gamma        | Γ    |
| \theta        | θ    |
| \Theta        | Θ    |
| \delta        | δ    |
| \Delta        | Δ    |
| \triangledown | ▽    |
| \epsilon      | ϵ    |
| \zeta         | ζ    |
| \eta          | η    |
| \kappa        | κ    |
| \lambda       | λ    |
| \mu           | μ    |
| \nu           | ν    |
| \xi           | ξ    |
| \pi           | π    |
| \sigma        | σ    |
| \tau          | τ    |
| \upsilon      | υ    |
| \phi          | ϕ    |
| \omega        | ω    |