---
layout: post
title: "Rstudio Tutorial"
date: 2021-08-08
description: "R Tutorial"
tag: R
---

dplyr 1.0版本增加了`across()`函数，这个函数集中体现了dplyr宏包的强大和简约，今天我用企鹅数据，来领略它的美。

```
library(tidyverse)
library(palmerpenguins)
penguins
## # A tibble: 344 x 8
##   species island    bill_length_mm bill_depth_mm
##   <fct>   <fct>              <dbl>         <dbl>
## 1 Adelie  Torgersen           39.1          18.7
## 2 Adelie  Torgersen           39.5          17.4
## 3 Adelie  Torgersen           40.3          18  
## 4 Adelie  Torgersen           NA            NA  
## 5 Adelie  Torgersen           36.7          19.3
## 6 Adelie  Torgersen           39.3          20.6
## # ... with 338 more rows, and 4 more variables:
## #   flipper_length_mm <int>, body_mass_g <int>,
## #   sex <fct>, year <int>
```

看到数据框里有很多缺失值，需要统计每一列缺失值的数量，按照常规的写法

```
penguins %>%
  summarise(
    na_in_species = sum(is.na(species)),
    na_in_island  = sum(is.na(island)),
    na_in_length  = sum(is.na(bill_length_mm)),
    na_in_depth   = sum(is.na(bill_depth_mm)),
    na_in_flipper = sum(is.na(flipper_length_mm)),
    na_in_body    = sum(is.na(body_mass_g)),
    na_in_sex     = sum(is.na(sex)),
    na_in_year    = sum(is.na(year))
  )
## # A tibble: 1 x 8
##   na_in_species na_in_island na_in_length na_in_depth
##           <int>        <int>        <int>       <int>
## 1             0            0            2           2
## # ... with 4 more variables: na_in_flipper <int>,
## #   na_in_body <int>, na_in_sex <int>,
## #   na_in_year <int>
```

幸亏数据框的列数不够多，只有8列，如果数据框有几百列，那就成体力活了，同时代码复制粘贴也容易出错。想偷懒，我们自然想到用`summarise_all()`，

```
penguins %>%
  summarise_all(
    ~ sum(is.na(.))
  )
## # A tibble: 1 x 8
##   species island bill_length_mm bill_depth_mm
##     <int>  <int>          <int>         <int>
## 1       0      0              2             2
## # ... with 4 more variables: flipper_length_mm <int>,
## #   body_mass_g <int>, sex <int>, year <int>
```

挺好。接着探索，我们想先按企鹅类型分组，然后统计出各体征数据的均值，这个好说，直接写代码

```
penguins %>%
  group_by(species) %>%
  summarise(
    mean_length   = mean(bill_length_mm, na.rm = TRUE),
    mean_depth    = mean(bill_depth_mm, na.rm = TRUE),
    mean_flipper  = mean(flipper_length_mm, na.rm = TRUE),
    mean_body     = mean(body_mass_g, na.rm = TRUE)
  )
## # A tibble: 3 x 5
##   species mean_length mean_depth mean_flipper mean_body
##   <fct>         <dbl>      <dbl>        <dbl>     <dbl>
## 1 Adelie         38.8       18.3         190.     3701.
## 2 Chinst~        48.8       18.4         196.     3733.
## 3 Gentoo         47.5       15.0         217.     5076.
```

或者用`summarise_if()`偷懒

```
d1 <- penguins %>%
  group_by(species) %>%
  summarise_if(is.numeric, mean, na.rm = TRUE)
d1
## # A tibble: 3 x 6
##   species bill_length_mm bill_depth_mm flipper_length_~
##   <fct>            <dbl>         <dbl>            <dbl>
## 1 Adelie            38.8          18.3             190.
## 2 Chinst~           48.8          18.4             196.
## 3 Gentoo            47.5          15.0             217.
## # ... with 2 more variables: body_mass_g <dbl>,
## #   year <dbl>
```

方法不错，从语义上还算很好理解。 但多了一列`year`, 我想在`summarise_if()`中用 `is.numeric & !year`去掉`year`，却没成功。人类的欲望是无穷的，我们还需要统计每组下企鹅的个数，然后合并到一起。因此，我们再接再厉

```
d2 <- penguins %>%
  group_by(species) %>%
  summarise(
    n = n()
  )
d2
## # A tibble: 3 x 2
##   species       n
##   <fct>     <int>
## 1 Adelie      152
## 2 Chinstrap    68
## 3 Gentoo      124
```

最后合并

```
d1 %>% left_join(d2, by = "species")
## # A tibble: 3 x 7
##   species bill_length_mm bill_depth_mm flipper_length_~
##   <fct>            <dbl>         <dbl>            <dbl>
## 1 Adelie            38.8          18.3             190.
## 2 Chinst~           48.8          18.4             196.
## 3 Gentoo            47.5          15.0             217.
## # ... with 3 more variables: body_mass_g <dbl>,
## #   year <dbl>, n <int>
```

结果应该没问题，然鹅，总让人感觉怪怪的，过程有点折腾，希望不这么麻烦。

## 1.1 across()横空出世

`across()`的出现，让这一切变得简单和清晰，上面三步完成的动作，一步搞定

![img](https://bookdown.org/wangminjie/R4DS/images/across_cover.jpg)

```
penguins %>%
  group_by(species) %>%
  summarise(
    across(where(is.numeric) & !year, mean, na.rm = TRUE),
    n = n()
  )
## # A tibble: 3 x 6
##   species bill_length_mm bill_depth_mm flipper_length_~
##   <fct>            <dbl>         <dbl>            <dbl>
## 1 Adelie            38.8          18.3             190.
## 2 Chinst~           48.8          18.4             196.
## 3 Gentoo            47.5          15.0             217.
## # ... with 2 more variables: body_mass_g <dbl>,
## #   n <int>
```

是不是很强大。大爱Hadley Wickham !!!

## 1.2 across()函数形式

`across()`函数，它有三个主要的参数：

```
across(.cols = , .fns = , .names = )
```

- 第一个参数.cols = ，选取我们要需要的若干列，选取多列的语法与`select()`的语法一致，选择方法非常丰富和人性化
  - 基本语法
    - `:`，变量在位置上是连续的，可以使用类似 `1:3` 或者`species:island`
    - `!`，变量名前加!，意思是求这个变量的补集，等价于去掉这个变量，比如`!species`
    - `&` 与 `|`，两组变量集的交集和并集，比如 `is.numeric & !year`, 就是选取数值类型变量，但不包括`year`; 再比如 `is.numeric | is.factor`就是选取数值型变量和因子型变量
    - `c()`，选取变量的组合，比如`c(a, b, x)`
  - 通过人性化的语句
    - `everything()`: 选取所有的变量
    - `last_col()`: 选取最后一列，也就说倒数第一列，也可以`last_col(offset = 1L)` 就是倒数第二列
  - 通过变量名的特征
    - `starts_with()`: 指定一组变量名的前缀，也就把选取具有这一前缀的变量，`starts_with("bill_")`
    - `ends_with()`: 指定一组变量名的后缀，也就选取具有这一后缀的变量，`ends_with("_mm")`
    - `contains()`: 指定变量名含有特定的字符串，也就是选取含有指定字符串的变量，`ends_with("length")`
    - `matches()`: 同上，字符串可以是正则表达式
  - 通过字符串向量
    - `all_of()`: 选取字符串向量对应的变量名，比如`all_of(c("species", "sex", "year"))`，当然前提是，数据框中要有这些变量，否则会报错。
    - `any_of()`: 同`all_of()`，只不过数据框中没有字符串向量对应的变量，也不会报错，比如数据框中没有people这一列，代码`any_of(c("species", "sex", "year", "people"))`也正常运行，挺人性化的
  - 通过函数
    - 常见的有数据类型函数 `where(is.numeric), where(is.factor), where(is.character), where(is.date)`
- 第二个参数`.fns =`，我们要执行的函数（或者多个函数），函数的语法有三种形式可选：
  - A function, e.g. `mean`.
  - A purrr-style lambda, e.g. `~ mean(.x, na.rm = TRUE)`
  - A list of functions/lambdas, e.g. `list(mean = mean, n_miss = ~ sum(is.na(.x))`
- 第三个参数`.names =`, 如果`.fns`是单个函数就默认保留原来数据列的名称，即`"{.col}"` ；如果`.fns`是多个函数，就在数据列的列名后面跟上函数名，比如`"{.col}_{.fn}"`；当然，我们也可以简单调整列名和函数之间的顺序或者增加一个标识的字符串，比如弄成`"{.fn}_{.col}"`，`"{.col}_{.fn}_aa"`

## 1.3 across()应用举例

下面通过一些小案例，继续呈现`across()`函数的功能

### 1.3.1 求每一列的缺失值数量

就是本章开始的需求

```
penguins %>%
  summarise(
    na_in_species = sum(is.na(species)),
    na_in_island  = sum(is.na(island)),
    na_in_length  = sum(is.na(bill_length_mm)),
    na_in_depth   = sum(is.na(bill_depth_mm)),
    na_in_flipper = sum(is.na(flipper_length_mm)),
    na_in_body    = sum(is.na(body_mass_g)),
    na_in_sex     = sum(is.na(sex)),
    na_in_year    = sum(is.na(year))
  )
```



```
# using across()
penguins %>%
  summarise(
    across(everything(), function(x) sum(is.na(x)))
  )
## # A tibble: 1 x 8
##   species island bill_length_mm bill_depth_mm
##     <int>  <int>          <int>         <int>
## 1       0      0              2             2
## # ... with 4 more variables: flipper_length_mm <int>,
## #   body_mass_g <int>, sex <int>, year <int>
```



```
# or
penguins %>%
  summarise(
    across(everything(), ~ sum(is.na(.)))
  )
## # A tibble: 1 x 8
##   species island bill_length_mm bill_depth_mm
##     <int>  <int>          <int>         <int>
## 1       0      0              2             2
## # ... with 4 more variables: flipper_length_mm <int>,
## #   body_mass_g <int>, sex <int>, year <int>
```

### 1.3.2 每个类型变量下有多少组？

```
penguins %>%
  summarise(
    distinct_species = n_distinct(species),
    distinct_island  = n_distinct(island),
    distinct_sex     = n_distinct(sex)
  )
## # A tibble: 1 x 3
##   distinct_species distinct_island distinct_sex
##              <int>           <int>        <int>
## 1                3               3            3
```



```
# using across()
penguins %>%
  summarise(
    across(c(species, island, sex), n_distinct)
  )
## # A tibble: 1 x 3
##   species island   sex
##     <int>  <int> <int>
## 1       3      3     3
```

### 1.3.3 多列多个统计函数

```
penguins %>%
  group_by(species) %>%
  summarise(
    length_mean  = mean(bill_length_mm, na.rm = TRUE),
    length_sd    = sd(bill_length_mm, na.rm = TRUE),
    depth_mean   = mean(bill_depth_mm, na.rm = TRUE),
    depth_sd     = sd(bill_depth_mm, na.rm = TRUE),
    flipper_mean = mean(flipper_length_mm, na.rm = TRUE),
    flipper_sd   = sd(flipper_length_mm, na.rm = TRUE),
    n            = n()
  )
## # A tibble: 3 x 8
##   species   length_mean length_sd depth_mean depth_sd
##   <fct>           <dbl>     <dbl>      <dbl>    <dbl>
## 1 Adelie           38.8      2.66       18.3    1.22 
## 2 Chinstrap        48.8      3.34       18.4    1.14 
## 3 Gentoo           47.5      3.08       15.0    0.981
## # ... with 3 more variables: flipper_mean <dbl>,
## #   flipper_sd <dbl>, n <int>
```



```
# using across()
penguins %>%
  group_by(species) %>%
  summarise(
    across(ends_with("_mm"), list(mean = mean, sd = sd), na.rm = TRUE),
    n = n()
  )
## # A tibble: 3 x 8
##   species   bill_length_mm_mean bill_length_mm_sd
##   <fct>                   <dbl>             <dbl>
## 1 Adelie                   38.8              2.66
## 2 Chinstrap                48.8              3.34
## 3 Gentoo                   47.5              3.08
## # ... with 5 more variables: bill_depth_mm_mean <dbl>,
## #   bill_depth_mm_sd <dbl>,
## #   flipper_length_mm_mean <dbl>,
## #   flipper_length_mm_sd <dbl>, n <int>
```

### 1.3.4 不同分组下数据变量的多个分位数

事实上，这里是`across()`与`summarise()`的强大结合起来

```
penguins %>%
  group_by(species, island) %>%
  summarise(
    prob    = c(.25, .75),
    length  = quantile(bill_length_mm, prob, na.rm = TRUE),
    depth   = quantile(bill_depth_mm, prob, na.rm = TRUE),
    flipper = quantile(flipper_length_mm, prob, na.rm = TRUE)
  )
## # A tibble: 10 x 6
## # Groups:   species, island [5]
##   species island     prob length depth flipper
##   <fct>   <fct>     <dbl>  <dbl> <dbl>   <dbl>
## 1 Adelie  Biscoe     0.25   37.7  17.6    185.
## 2 Adelie  Biscoe     0.75   40.7  19.0    193 
## 3 Adelie  Dream      0.25   36.8  17.5    185 
## 4 Adelie  Dream      0.75   40.4  18.8    193 
## 5 Adelie  Torgersen  0.25   36.7  17.4    187 
## 6 Adelie  Torgersen  0.75   41.1  19.2    195 
## # ... with 4 more rows
```



```
# using across()
penguins %>%
  group_by(species, island) %>%
  summarise(
    prob = c(.25, .75),
    across(
      c(bill_length_mm, bill_depth_mm, flipper_length_mm),
      ~ quantile(., prob, na.rm = TRUE)
    )
  )
## # A tibble: 10 x 6
## # Groups:   species, island [5]
##   species island     prob bill_length_mm bill_depth_mm
##   <fct>   <fct>     <dbl>          <dbl>         <dbl>
## 1 Adelie  Biscoe     0.25           37.7          17.6
## 2 Adelie  Biscoe     0.75           40.7          19.0
## 3 Adelie  Dream      0.25           36.8          17.5
## 4 Adelie  Dream      0.75           40.4          18.8
## 5 Adelie  Torgersen  0.25           36.7          17.4
## 6 Adelie  Torgersen  0.75           41.1          19.2
## # ... with 4 more rows, and 1 more variable:
## #   flipper_length_mm <dbl>
```



```
# or
penguins %>%
  group_by(species, island) %>%
  summarise(
    prob = c(.25, .75),
    across(where(is.numeric) & !year, ~ quantile(., prob, na.rm = TRUE))
  )
## # A tibble: 10 x 7
## # Groups:   species, island [5]
##   species island     prob bill_length_mm bill_depth_mm
##   <fct>   <fct>     <dbl>          <dbl>         <dbl>
## 1 Adelie  Biscoe    0.375           37.7          17.6
## 2 Adelie  Biscoe    0.625           40.7          19.0
## 3 Adelie  Dream     0.375           36.8          17.5
## 4 Adelie  Dream     0.625           40.4          18.8
## 5 Adelie  Torgersen 0.375           36.7          17.4
## 6 Adelie  Torgersen 0.625           41.1          19.2
## # ... with 4 more rows, and 2 more variables:
## #   flipper_length_mm <dbl>, body_mass_g <dbl>
```

### 1.3.5 不同分组下更复杂的统计

```
# using across()
penguins %>%
  group_by(species) %>%
  summarise(
    n = n(),
    across(starts_with("bill_"), mean, na.rm = TRUE),
    Area = mean(bill_length_mm * bill_depth_mm, na.rm = TRUE),
    across(ends_with("_g"), mean, na.rm = TRUE),
  )
## # A tibble: 3 x 6
##   species       n bill_length_mm bill_depth_mm  Area
##   <fct>     <int>          <dbl>         <dbl> <dbl>
## 1 Adelie      152           38.8          18.3  712.
## 2 Chinstrap    68           48.8          18.4  900.
## 3 Gentoo      124           47.5          15.0  712.
## # ... with 1 more variable: body_mass_g <dbl>
```

### 1.3.6 数据标准化处理

```
std <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

# using across()
penguins %>%
  summarise(
    across(where(is.numeric), std),
    across(where(is.character), as.factor)
  )
## # A tibble: 344 x 5
##   bill_length_mm bill_depth_mm flipper_length_mm
##            <dbl>         <dbl>             <dbl>
## 1         -0.883         0.784            -1.42 
## 2         -0.810         0.126            -1.06 
## 3         -0.663         0.430            -0.421
## 4         NA            NA                NA    
## 5         -1.32          1.09             -0.563
## 6         -0.847         1.75             -0.776
## # ... with 338 more rows, and 2 more variables:
## #   body_mass_g <dbl>, year <dbl>
```



```
# using across() and purrr style
penguins %>%
  drop_na() %>% 
  summarise(
    across(starts_with("bill_"), ~ (.x - mean(.x)) / sd(.x))
  )
## # A tibble: 333 x 2
##   bill_length_mm bill_depth_mm
##            <dbl>         <dbl>
## 1         -0.895         0.780
## 2         -0.822         0.119
## 3         -0.675         0.424
## 4         -1.33          1.08 
## 5         -0.858         1.74 
## 6         -0.931         0.323
## # ... with 327 more rows
```

### 1.3.7 数据对数化处理

```
# using across()
penguins %>%
  drop_na() %>%
  mutate(
    across(where(is.numeric), log),
    across(where(is.character), as.factor)
  )
## # A tibble: 333 x 8
##   species island    bill_length_mm bill_depth_mm
##   <fct>   <fct>              <dbl>         <dbl>
## 1 Adelie  Torgersen           3.67          2.93
## 2 Adelie  Torgersen           3.68          2.86
## 3 Adelie  Torgersen           3.70          2.89
## 4 Adelie  Torgersen           3.60          2.96
## 5 Adelie  Torgersen           3.67          3.03
## 6 Adelie  Torgersen           3.66          2.88
## # ... with 327 more rows, and 4 more variables:
## #   flipper_length_mm <dbl>, body_mass_g <dbl>,
## #   sex <fct>, year <dbl>
```



```
# using across()
penguins %>%
  drop_na() %>%
  mutate(
    across(where(is.numeric), .fns = list(log = log), .names = "{.fn}_{.col}"),
    across(where(is.character), as.factor)
  )
## # A tibble: 333 x 13
##   species island    bill_length_mm bill_depth_mm
##   <fct>   <fct>              <dbl>         <dbl>
## 1 Adelie  Torgersen           39.1          18.7
## 2 Adelie  Torgersen           39.5          17.4
## 3 Adelie  Torgersen           40.3          18  
## 4 Adelie  Torgersen           36.7          19.3
## 5 Adelie  Torgersen           39.3          20.6
## 6 Adelie  Torgersen           38.9          17.8
## # ... with 327 more rows, and 9 more variables:
## #   flipper_length_mm <int>, body_mass_g <int>,
## #   sex <fct>, year <int>, log_bill_length_mm <dbl>,
## #   log_bill_depth_mm <dbl>,
## #   log_flipper_length_mm <dbl>,
## #   log_body_mass_g <dbl>, log_year <dbl>
```

### 1.3.8 在分组建模中与`cur_data()`配合使用

```
penguins %>%
  group_by(species) %>%
  summarise(
    broom::tidy(lm(bill_length_mm ~ bill_depth_mm, data = cur_data()))
  )
## # A tibble: 6 x 6
## # Groups:   species [3]
##   species  term   estimate std.error statistic  p.value
##   <fct>    <chr>     <dbl>     <dbl>     <dbl>    <dbl>
## 1 Adelie   (Inte~   23.1       3.03       7.60 3.01e-12
## 2 Adelie   bill_~    0.857     0.165      5.19 6.67e- 7
## 3 Chinstr~ (Inte~   13.4       5.06       2.66 9.92e- 3
## 4 Chinstr~ bill_~    1.92      0.274      7.01 1.53e- 9
## 5 Gentoo   (Inte~   17.2       3.28       5.25 6.60e- 7
## 6 Gentoo   bill_~    2.02      0.219      9.24 1.02e-15
```



```
penguins %>%
  group_by(species) %>%
  summarise(
    broom::tidy(lm(bill_length_mm ~ ., data = cur_data() %>% select(is.numeric)))
  )
## # A tibble: 15 x 6
## # Groups:   species [3]
##   species  term    estimate std.error statistic p.value
##   <fct>    <chr>      <dbl>     <dbl>     <dbl>   <dbl>
## 1 Adelie   (Inter~ -2.75e+2   5.09e+2    -0.539 5.90e-1
## 2 Adelie   bill_d~  2.70e-1   1.92e-1     1.40  1.63e-1
## 3 Adelie   flippe~  2.51e-2   3.50e-2     0.717 4.74e-1
## 4 Adelie   body_m~  2.62e-3   5.25e-4     4.98  1.74e-6
## 5 Adelie   year     1.47e-1   2.55e-1     0.576 5.66e-1
## 6 Chinstr~ (Inter~ -4.20e+2   8.24e+2    -0.509 6.12e-1
## # ... with 9 more rows
```



```
penguins %>%
  group_by(species) %>%
  summarise(
    broom::tidy(lm(bill_length_mm ~ .,
                data = cur_data() %>% transmute(across(is.numeric))
    ))
  )
## # A tibble: 15 x 6
## # Groups:   species [3]
##   species  term    estimate std.error statistic p.value
##   <fct>    <chr>      <dbl>     <dbl>     <dbl>   <dbl>
## 1 Adelie   (Inter~ -2.75e+2   5.09e+2    -0.539 5.90e-1
## 2 Adelie   bill_d~  2.70e-1   1.92e-1     1.40  1.63e-1
## 3 Adelie   flippe~  2.51e-2   3.50e-2     0.717 4.74e-1
## 4 Adelie   body_m~  2.62e-3   5.25e-4     4.98  1.74e-6
## 5 Adelie   year     1.47e-1   2.55e-1     0.576 5.66e-1
## 6 Chinstr~ (Inter~ -4.20e+2   8.24e+2    -0.509 6.12e-1
## # ... with 9 more rows
```



```
penguins %>%
  group_by(species) %>%
  summarise(
    broom::tidy(lm(bill_length_mm ~ ., data = across(is.numeric)))
  )
## # A tibble: 15 x 6
## # Groups:   species [3]
##   species  term    estimate std.error statistic p.value
##   <fct>    <chr>      <dbl>     <dbl>     <dbl>   <dbl>
## 1 Adelie   (Inter~ -2.75e+2   5.09e+2    -0.539 5.90e-1
## 2 Adelie   bill_d~  2.70e-1   1.92e-1     1.40  1.63e-1
## 3 Adelie   flippe~  2.51e-2   3.50e-2     0.717 4.74e-1
## 4 Adelie   body_m~  2.62e-3   5.25e-4     4.98  1.74e-6
## 5 Adelie   year     1.47e-1   2.55e-1     0.576 5.66e-1
## 6 Chinstr~ (Inter~ -4.20e+2   8.24e+2    -0.509 6.12e-1
## # ... with 9 more rows
```

### 1.3.9 与`cur_column()`配合使用

```
# 每一列乘以各自的系数
df   <- tibble(x = 1:3, y = 3:5, z = 5:7)
mult <- list(x = 1, y = 10, z = 100)

df %>% 
  mutate(across(all_of(names(mult)), ~ .x * mult[[cur_column()]]))
## # A tibble: 3 x 3
##       x     y     z
##   <dbl> <dbl> <dbl>
## 1     1    30   500
## 2     2    40   600
## 3     3    50   700
```



```
# 每一列乘以各自的权重
df      <- tibble(x = 1:3, y = 3:5, z = 5:7)
weights <- list(x = 0.2, y = 0.3, z = 0.5)

df %>%
  mutate(
    across(all_of(names(weights)),
           list(wt = ~ .x * weights[[cur_column()]]),
          .names = "{col}.{fn}"
    )
  )
## # A tibble: 3 x 6
##       x     y     z  x.wt  y.wt  z.wt
##   <int> <int> <int> <dbl> <dbl> <dbl>
## 1     1     3     5   0.2   0.9   2.5
## 2     2     4     6   0.4   1.2   3  
## 3     3     5     7   0.6   1.5   3.5
```



```
# 每一列有各自的阈值，如果在阈值之上为1，否则为 0
df      <- tibble(x = 1:3, y = 3:5, z = 5:7)
cutoffs <- list(x = 2, y = 3, z = 7)

df %>% mutate(
  across(all_of(names(cutoffs)), ~ if_else(.x > cutoffs[[cur_column()]], 1, 0))
)
## # A tibble: 3 x 3
##       x     y     z
##   <dbl> <dbl> <dbl>
## 1     0     0     0
## 2     0     1     0
## 3     1     1     0
```

### 1.3.10 与`c_across()`配合也挺默契

在一行中的占比

```
df <- tibble(x = 1:3, y = 3:5, z = 5:7)

df %>%
  rowwise() %>%
  mutate(total = sum(c_across(x:z))) %>%
  ungroup() %>%
  mutate(across(x:z, ~ . / total))
## # A tibble: 3 x 4
##       x     y     z total
##   <dbl> <dbl> <dbl> <int>
## 1 0.111 0.333 0.556     9
## 2 0.167 0.333 0.5      12
## 3 0.2   0.333 0.467    15
```

更神奇的方法，请看第 [28](https://bookdown.org/wangminjie/R4DS/beauty-of-across3.html#beauty-of-across3) 章。

看一行中哪个最大，最大的变为1，其余的变为0

```
replace_col_max <- function(vec) {
  if (!is.vector(vec)) {
    stop("input of replace_col_max must be vector.")
  }

  if_else(vec == max(vec), 1L, 0L)
}


df %>%
  rowwise() %>%
  mutate(
    new = list(replace_col_max(c_across(everything())))
  ) %>%
  unnest_wider(new, names_sep = "_")
## # A tibble: 3 x 6
##       x     y     z new_1 new_2 new_3
##   <int> <int> <int> <int> <int> <int>
## 1     1     3     5     0     0     1
## 2     2     4     6     0     0     1
## 3     3     5     7     0     0     1
```

## 1.4 across()总结

我们看到了，`across()`函数在`summarise()/mutate()/transmute()/condense()`中使用，它能实现以下几个功能：

- 数据框中的多列执行相同操作
- 不同性质的操作，有时可以一起写出，不用再`left_join()`

![across()函数总结图](https://bookdown.org/wangminjie/R4DS/images/across.png)

