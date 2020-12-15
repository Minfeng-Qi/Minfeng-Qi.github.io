---
layout: post
title: "Wechat Program Tutorial"
date: 2020-12-16
description: "Wechat Program Tutorial"
tag: Wechat
---

## 一、小程序是什么？

学习小程序之前，先简单说一下，它到底是什么。

字面上讲，小程序就是微信里面的应用程序，外部代码通过小程序这种形式，在微信这个手机 App 里面运行。

但是，更准确的说法是， **小程序可以视为只能用微信打开和浏览的网站。** 小程序和网页的技术模型是一样的，用到的 JavaScript 语言和 CSS 样式也是一样的，只是网页的 HTML 标签被稍微修改成了 WXML 标签。所以，小程序页面本质上就是网页。

小程序的特殊之处在于，虽然是网页，但是它不支持浏览器，所有浏览器的 API 都不能使用，只能用微信提供的 API。这也是为什么小程序只能用微信打开的原因，因为底层全变了。

## 二、小程序的优势

小程序最大的优势，就是它基于微信。

微信 App 的功能（比如拍照、扫描、支付等等），小程序大部分都能使用。微信提供了各种封装好的 API，开发者不用自己实现，也不用考虑 iOS 和安卓的平台差异，只要一行代码就可以调用。

而且，开发者也不用考虑用户的注册和登录，直接使用微信的注册和登录，微信的用户自动成为你的用户。

## 三、知识准备

由于小程序基于网页技术，所以学习之前，最好懂一点网页开发。具体来说，下面两方面的知识是必需的。

（1）JavaScript 语言：懂基本语法，会写简单的 JS 脚本程序。

（2）CSS 样式：理解如何使用 CSS 控制网页元素的外观。

此外，虽然 HTML 标签和浏览器 API 不是必备知识，但是了解浏览器怎么渲染网页，对于理解小程序模型有很大的帮助。

总的来说，先学网页开发，再学小程序，是比较合理的学习途径，而且网页开发的资料比较多，遇到问题容易查到解决方法。但是，网页开发要学的东西太多，不是短期能掌握的，如果想快速上手，先学小程序，遇到不懂的地方再去查资料，也未尝不可。

## 四、开发准备

小程序开发的第一步，是去[微信公众平台](https://mp.weixin.qq.com/)注册，申请一个 AppID，这是免费的。

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092910.jpg)

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092911.jpg)

申请完成以后，你会得到一个 AppID（小程序编号） 和 AppSecret（小程序密钥），后面都会用到。

然后，下载微信提供的[小程序开发工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)。这个工具是必需的，因为只有它才能运行和调试小程序源码。

开发者工具支持 Windows 和 MacOS 两个平台。我装的是 Windows （64位）的版本，这个教程的内容也是基于该版本的，但是 MacOS 版本的操作应该是完全一样的。

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092914.jpg)

安装好打开这个软件，会要求你使用微信扫描二维码登录。

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092915.jpg)

登录后，进入新建项目的页面，可以新建不同的项目，默认是新建小程序项目。

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092916.jpg)

点击右侧的`+`号，就跳出了新建小程序的页面。

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092918.jpg)

如果直接新建小程序，会生成一个完整的项目脚手架。对于初学者来说，这样反而不利于掌握各个文件的作用。更好的学习方法是，自己从头手写每一行代码，然后切换到"导入项目"的选项，将其导入到开发者工具。

![img](https://www.wangbase.com/blogimg/asset/202009/bg2020092919.jpg)

导入时，需要给小程序起一个名字，并且填写项目代码所在的目录，以及前面申请的 AppID。

## 五、 hello world 示例

下面，就请大家动手，跟着写一个最简单的小程序，只要五分钟就能完成。

第一步，新建一个小程序的项目目录。名字可以随便起，这里称为`wechat-miniprogram-demo`。

你可以在资源管理器里面，新建目录。如果熟悉命令行操作，也可以打开 Windows Terminal（没有的话，需要安装），在里面执行下面的命令，新建并进入该目录。

> ```bash
> > mkdir wechat-miniprogram-demo
> > cd wechat-miniprogram-demo
> ```

第二步，在该目录里面，新建一个脚本文件`app.js`。这个脚本用于对整个小程序进行初始化。

`app.js`内容只有一行代码。

> ```javascript
> App({});
> ```

上面代码中，`App()`由小程序原生提供，它是一个函数，表示新建一个小程序实例。它的参数是一个配置对象，用于设置小程序实例的行为属性。这个例子不需要任何配置，所以使用空对象即可。

第三步，新建一个配置文件`app.json`，记录项目的一些静态配置。

`app.json`采用 JSON 格式。JSON 是基于 JavaScript 语言的一种数据交换格式，只有五条语法规则，非常简单，不熟悉 JSON 的同学可以参考[这篇教程](https://wangdoc.com/javascript/stdlib/json.html#json-格式)。

`app.json`文件的内容，至少必须有一个`pages`属性，指明小程序包含哪些页面。

> ```javascript
> {
>   "pages": [
>     "pages/home/home"
>   ]
> }
> ```

上面代码中，`pages`属性是一个数组，数组的每一项就是一个页面。这个示例中，小程序只有一个页面，所以数组只有一项`pages/home/home`。

`pages/home/home`是一个三层的文件路径。

1. 所有页面都放在`pages`子目录里面。
2. 每个页面有一个自己的目录，这里是`pages`下面的`home`子目录，表示这个页面叫做`home`。页面的名字可以随便起，只要对应的目录确实存在即可。
3. 小程序会加载页面目录`pages/home`里面的`home.js`文件，`.js`后缀名可以省略，所以完整的加载路径为`pages/home/home`。`home.js`这个脚本的文件名也可以随便起，但是习惯上跟页面目录同名。

第四步，新建`pages/home`子目录。

> ```bash
> $ mkdir -p pages/home
> ```

然后，在这个目录里面新建一个脚本文件`home.js`。该文件的内容如下。

> ```javascript
> Page({});
> ```

上面代码中，`Page()`由小程序原生提供，它是一个函数，用于初始化一个页面实例。它的参数是一个配置对象，用于设置当前页面的行为属性。这里是一个空对象，表示不设置任何属性。

第五步，在`pages/home`目录新建一个`home.wxml`文件。WXML 是微信页面标签语言，类似于 HTML 语言，用于描述小程序的页面。

`home.wxml`的内容很简单，就写一行`hello world`。

> ```markup
> hello world
> ```

到这一步，就算基本完成了。现在，打开小程序开发工具，导入项目目录`wechat-miniprogram-demo`。如果一切正常，就可以在开发者工具里面，看到运行结果了。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100108.jpg)

点击工具栏的"预览"或"真机调试"按钮，还可以在你的手机上面，查看真机运行结果。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100109.jpg)

这个示例的完整代码，可以到[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/01.hello-world)查看。

## 六、WXML 标签语言

上一节的`home.wxml`文件，只写了最简单的一行`hello world`。实际开发中，不会这样写，而是要加上各种标签，以便后面添加样式和效果。

小程序的 WXML 语言提供各种页面标签。下面，对`home.wxml`改造一下，加上两个最常用的标签。

> ```markup
> <view>
>   <text>hello world</text>
> </view>
> ```

上面的代码用到了两个标签：`<view>`和`<text>`。

`<view>`标签表示一个区块，用于跟其他区块分隔，类似 HTML 语言的`<div>`标签。`<text>`表示一段行内文本，类似于 HTML 语言的`<span>`标签，多个`<text>`标签之间不会产生分行。

注意，每个标签都是成对使用，需要有闭合标记，即标签名前加斜杠表示闭合，比如`<view>`的闭合标记是`</view>`。如果缺少闭合标记，小程序编译时会报错。

由于我们还没有为页面添加任何样式，所以页面的渲染效果跟上一节是一样的。后面添加样式时，大家就可以看到标签的巨大作用。

## 七、小程序的项目结构

总结一下，这个示例一共有4个文件，项目结构如下。

> ```bash
> |- app.json
> |- app.js
> |- pages
>    |- home
>       |- home.wxml
>       |- home.js
> ```

这就是最简单、最基本的小程序结构。所有的小程序项目都是这个结构，在上面不断添加其他内容。

这个结构分成两层：描述整体程序的顶层 app 脚本，以及描述各个页面的 page 脚本。

## 八、项目配置文件 app.json

顶层的`app.json`文件用于整个项目的配置，对于所有页面都有效。

除了前面提到的必需的`pages`属性，`app.json`文件还有一个[`window`属性](https://developers.weixin.qq.com/miniprogram/dev/reference/configuration/app.html#window)，用来设置小程序的窗口。`window`属性的值是一个对象，其中有三个属性很常用。

> - `navigationBarBackgroundColor`：导航栏的颜色，默认为`#000000`（黑色）。
> - `navigationBarTextStyle`：导航栏的文字颜色，只支持`black`（黑色）或`white`（白色），默认为`white`。
> - `navigationBarTitleText`：导航栏的文字，默认为空。

下面，改一下前面的`app.json`，加入`window`属性。

> ```javascript
> {
>   "pages": [
>     "pages/home/home"
>   ],
>   "window": {
>     "navigationBarBackgroundColor": "#ff0000",
>     "navigationBarTextStyle": "white",
>     "navigationBarTitleText": "小程序 Demo"     
>   }
> }
> ```

上面代码中，`window`属性设置导航栏的背景颜色为红色（`#ff0000`），文本颜色为白色（`white`），标题文字为"小程序 Demo"。

开发者工具导入项目代码，就可以看到导航栏变掉了。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100110.jpg)

这个示例的完整代码，可以到[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/02.app-json)查看。

除了窗口的样式，很多小程序的顶部或尾部，还有选项栏，可以切换到不同的选项卡。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100111.jpg)

这个选项栏，也是在`app.json`里面设置，使用[`tabBar`属性](https://developers.weixin.qq.com/miniprogram/dev/reference/configuration/app.html#tabBar)，这里就不展开了。

## 九、总体样式

微信小程序允许在顶层放置一个`app.wxss`文件，里面采用 CSS 语法设置页面样式。这个文件的设置，对所有页面都有效。

注意，小程序虽然使用 CSS 样式，但是样式文件的后缀名一律要写成`.wxss`。

打开上一篇教程的示例，在项目顶层新建一个`app.wxss`文件，内容如下。

> ```css
> page {
>   background-color: pink;
> }
> 
> text {
>   font-size: 24pt;
>   color: blue;
> }
> ```

上面代码将整个页面的背景色设为粉红，然后将`<text>`标签的字体大小设为 24 磅，字体颜色设为蓝色。

开发者工具导入代码之后，得到了下面的渲染结果。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100303.jpg)

可以看到，页面的背景色变成粉红，文本字体变大了，字体颜色变成了蓝色。

实际开发中，直接对`<text>`标签设置样式，会影响到所有的文本。一般不这样用，而是通过`class`属性区分不同类型的文本，然后再对每种`class`设置样式。

打开`pages/home/home.wxml`文件，把页面代码改成下面这样。

> ```markup
> <view>
>   <text class="title">hello world</text>
> </view>
> ```

上面代码中，我们为`<text>`标签加上了一个`class`属性，值为`title`。

然后，将顶层的`app.wxss`文件改掉，不再直接对`<text>`设置样式，改成对`class`设置样式。

> ```css
> page {
>   background-color: pink;
> }
> 
> .title {
>   font-size: 24pt;
>   color: blue;
> }
> ```

上面代码中，样式设置在 class 上面（`.title`），这样就可以让不同的`class`呈现不同的样式。修改之后，页面的渲染结果并不会有变化。

## 十、Flex 布局

各种页面元素的位置关系，称为布局（layout），小程序官方推荐使用 Flex 布局。不熟悉这种布局的同学，可以看看[《Flex 布局教程》](http://www.ruanyifeng.com/blog/2015/07/flex-grammar.html)。

下面演示如何通过 Flex 布局，将上面示例的文本放置到页面中央。

首先，在`pages/home`目录里面，新建一个`home.wxss`文件，这个文件设置的样式，只对 home 页面生效。这是因为每个页面通常有不一样的布局，所以页面布局一般不写在全局的`app.wxss`里面。

然后，`home.wxss`文件写入下面的内容。

> ```css
> page {
>   height: 100%;
>   width: 750rpx;
>   display: flex;
>   justify-content: center;
>   align-items: center;
> }
> ```

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100304.jpg)

下面解释一下上面这段 WXSS 代码，还是很简单的。

（1）`height: 100%;`：页面高度为整个屏幕高度。

（2）`width: 750rpx;`：页面宽度为整个屏幕宽度。

注意，这里单位是`rpx`，而不是`px`。`rpx`是小程序为适应不同宽度的手机屏幕，而发明的一种长度单位。不管什么手机屏幕，宽度一律为`750rpx`。它的好处是换算简单，如果一个元素的宽度是页面的一半，只要写成`width: 375rpx;`即可。

（3）`display: flex;`：整个页面（page）采用 Flex 布局。

（4）`justify-content: center;`：页面的一级子元素（这个示例是`<view>`）水平居中。

（5）`align-items: center;`：页面的一级子元素（这个示例是`<view>`）垂直居中。一个元素同时水平居中和垂直中央，就相当于处在页面的中央了。

这个示例的完整代码，可以到[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/04.flex)查看。

## 十一、WeUI

如果页面的所有样式都自己写，还是挺麻烦的，也没有这个必要。腾讯封装了一套 UI 框架 [WeUI](https://github.com/Tencent/weui)，可以拿来用。

手机访问 [weui.io](https://weui.io/)，可以看到这套 UI 框架的效果。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100305.jpg)

这一节就来看看，怎么使用这个框架的小程序版本 [WeUI-WXSS](https://github.com/Tencent/weui-wxss/)，为我们的页面加上官方的样式。

首先，进入它的 [GitHub 仓库](https://github.com/Tencent/weui-wxss/)，在`dist/style`目录下面，找到[`weui.wxss`](https://github.com/Tencent/weui-wxss/blob/master/dist/style/weui.wxss)这个文件，将[源码](https://raw.githubusercontent.com/Tencent/weui-wxss/master/dist/style/weui.wxss)全部复制到你的`app.wxss`文件的头部。

然后，将`page/home/home.wxml`文件改成下面这样。

> ```markup
> <view>
>   <button class="weui-btn weui-btn_primary">
>     主操作
>   </button>
>   <button class="weui-btn weui-btn_primary weui-btn_loading">
>     <i class="weui-loading"></i>正在加载
>   </button>
>   <button class="weui-btn weui-btn_primary weui-btn_disabled">
>     禁止点击
>   </button>
> </view>
> ```

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100306.jpg)

可以看到，加入 WeUI 框架以后，只要为按钮添加不同的 class，就能自动出现框架提供的样式。你可以根据需要，为页面选择不同的按钮。

这个示例中，`<button>`元素使用了下面的`class`。

> - `weui-btn`：按钮样式的基类
> - `weui-btn_primary`：主按钮的样式。如果是次要按钮，就使用`weui-btn_default`。
> - `weui-btn_loading`：按钮点击后，操作正在进行中的样式。该类内部需要用`<i>`元素，加上表示正在加载的图标。
> - `weui-btn_disabled`：按钮禁止点击的样式。

WeUI 提供了大量的元素样式，完整的清单可以查看[这里](https://github.com/Tencent/weui-wxss)。

这个示例的完整代码，可以到[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/05.weui)查看。

## 十二、加入图片

美观的页面不能光有文字，还必须有图片。小程序的`<image>`组件就用来加载图片。

打开`home.wxml`文件，将其改为如下代码。

> ```markup
> <view>
>   <image src="https://picsum.photos/200"></image>
> </view>
> ```

开发者工具加载项目代码，页面的渲染结果如下，可以显示图片了。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100309.jpg)

`<image>`组件有[很多属性](https://developers.weixin.qq.com/miniprogram/dev/component/image.html)，比如可以通过`style`属性指定样式。

> ```markup
> <view>
>   <image
>     src="https://picsum.photos/200"
>     style="height: 375rpx; width: 375rpx;"
>   ></image>
> </view>
> ```

默认情况下，图片会占满整个容器的宽度（这个例子是`<view>`的宽度），上面代码通过`style`属性指定图片的高度和宽度（占据页面宽度的一半），渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100310.jpg)

当然，图片样式不一定写在`<image>`组件里面，也可以写在 WXSS 样式文件里面。

这个示例的完整代码，可以到[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/06.image)查看。

## 十三、图片轮播

小程序原生的[``组件](https://developers.weixin.qq.com/miniprogram/dev/component/swiper.html)可以提供图片轮播效果。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100311.jpg)

上面页面的图片上面，有三个提示点，表示一共有三张图片，可以切换显示。

它的代码很简单，只需要改一下`home.wxml`文件即可。

> ```markup
> <view>
>   <swiper
>     indicator-dots="{{true}}"  
>     autoplay="{{true}}"
>     style="width: 750rpx;">
>     <swiper-item>
>       <image src="https://picsum.photos/200"></image>
>     </swiper-item>
>     <swiper-item>
>       <image src="https://picsum.photos/250"></image>
>     </swiper-item>
>     <swiper-item>
>       <image src="https://picsum.photos/300"></image>
>     </swiper-item>
>   </swiper>
> </view>
> ```

上面代码中，`<swiper>`组件就是轮播组件，里面放置了三个[``组件](https://developers.weixin.qq.com/miniprogram/dev/component/swiper-item.html)，表示有三个轮播项目，每个项目就是一个`<image>`组件。

`<swiper>`组件的`indicator-dots`属性设置是否显示轮播点，`autoplay`属性设置是否自动播放轮播。它们的属性值都是一个布尔值，这里要写成`{{true}}`。这种`{{...}}`的语法，表示里面放置的是 JavaScript 代码。

## 十四、数据绑定

前面的所有示例，小程序的页面都是写死的，也就是页面内容不会变。但是，页面数据其实可以通过脚本传入，通过脚本改变页面，实现动态效果。

小程序提供了一种特别的方法，让页面可以更方便地使用脚本数据，叫做"数据绑定"（data binding）。

所谓"数据绑定"，指的是脚本里面的某些数据，会自动成为页面可以读取的全局变量，两者会同步变动。也就是说，脚本里面修改这个变量的值，页面会随之变化；反过来，页面上修改了这段内容，对应的脚本变量也会随之变化。这也叫做 MVVM 模式。

下面看一个例子。打开`home.js`文件，改成下面这样。

> ```javascript
> Page({
>   data: {
>     name: '张三'
>   }
> });
> ```

上面代码中，`Page()`方法的配置对象有一个`data`属性。这个属性的值也是一个对象，有一个`name`属性。数据绑定机制规定，`data`对象的所有属性，自动成为当前页面可以读取的全局变量。也就是说，`home`页面可以自动读取`name`变量。

接着，修改`home.wxml`文件，加入`name`变量。

> ```markup
> <view>
>   <text class="title">hello {{name}}</text>
> </view>
> ```

上面代码中，`name`变量写在`{{...}}`里面。这是小程序特有的语法，两重大括号表示，内部不是文本，而是 JavaScript 代码，它的执行结果会写入页面。因此，`{{name}}`表示读取全局变量`name`的值，将这个值写入网页。

注意，变量名区分大小写，`name`和`Name`是两个不同的变量名。

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100407.jpg)

可以看到，`name`变量已经自动替换成了变量值"张三"。

这个示例的完整代码，可以查看[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/08.data-binding)。

页面和脚本对于变量`name`是数据绑定关系，无论哪一方改变了`name`的值，另一方也会自动跟着改变。后面讲解到事件时，会有双方联动的例子。

## 十五、全局数据

数据绑定只对当前页面有效，如果某些数据要在多个页面共享，就需要写到全局配置对象里面。

打开`app.js`，改写如下。

> ```javascript
> App({
>   globalData: {
>     now: (new Date()).toLocaleString()
>   }
> });
> ```

上面代码中，`App()`方法的参数配置对象有一个`globalData`属性，这个属性就是我们要在多个页面之间分享的值。事实上，配置对象的任何一个属性都可以共享，这里起名为`globalData`只是为了便于识别。

然后，打开`home.js`，改成下面的内容，在页面脚本里面获取全局对象。

> ```javascript
> const app = getApp();
> 
> Page({
>   data: {
>     now: app.globalData.now
>   }
> });
> ```

上面代码中，`getApp()`函数是小程序原生提供的函数方法，用于从页面获取 App 实例对象。拿到实例对象以后，就能从它上面拿到全局配置对象的`globalData`属性，从而把`app.globalData.now`赋值给页面脚本的`now`属性，进而通过数据绑定机制，变成页面的全局变量`now`。

最后，修改一下页面代码`home.wxml`。

> ```markup
> <view>
>   <text class="title">现在是 {{now}}</text>
> </view>
> ```

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100408.jpg)

可以看到，页面读到了全局配置对象`app.js`里面的数据。

这个示例的完整代码，可以查看[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/09.global-data)。

## 十六、事件

事件是小程序跟用户互动的主要手段。小程序通过接收各种用户事件，执行回调函数，做出反应。

小程序的[常见事件](https://developers.weixin.qq.com/miniprogram/dev/framework/view/wxml/event.html)有下面这些。

> - `tap`：触摸后马上离开。
> - `longpress`：触摸后，超过 350ms 再离开。如果指定了该事件的回调函数并触发了该事件，`tap`事件将不被触发。
> - `touchstart`：触摸开始。
> - `touchmove`：触摸后移动。
> - `touchcancel`：触摸动作被打断，如来电提醒，弹窗等。
> - `touchend`：触摸结束。

上面这些事件，在传播上分成两个阶段：先是捕获阶段（由上层元素向下层元素传播），然后是冒泡阶段（由下层元素向上层元素传播）。所以，同一个事件在同一个元素上面其实会触发两次：捕获阶段一次，冒泡阶段一次。详细的介绍，请参考我写的[事件模型解释](https://wangdoc.com/javascript/events/model.html#事件的传播)。

小程序允许页面元素，通过属性指定各种事件的回调函数，并且还能够指定是哪个阶段触发回调函数。具体方法是为事件属性名加上不同的前缀。小程序提供四种前缀。

> - `capture-bind`：捕获阶段触发。
> - `capture-catch`：捕获阶段触发，并中断事件，不再向下传播，即中断捕获阶段，并取消随后的冒泡阶段。
> - `bind`：冒泡阶段触发。
> - `catch`：冒泡阶段触发，并取消事件进一步向上冒泡。

下面通过一个例子，来看如何为事件指定回调函数。打开`home.wxml`文件，改成下面的代码。

> ```markup
> <view>
>   <text class="title">hello {{name}}</text>
>   <button bind:tap="buttonHandler">点击</button>
> </view>
> ```

上面代码中，我们为页面加上了一个按钮，并为这个按钮指定了触摸事件（`tap`）的回调函数`buttonHandler`，`bind:`前缀表示这个回调函数会在冒泡阶段触发（前缀里面的冒号可以省略，即写成`bindtap`也可以）。

回调函数必须在页面脚本中定义。打开`home.js`文件，改成下面的代码。

> ```javascript
> Page({
>   data: {
>     name: '张三'
>   },
>   buttonHandler(event) {
>     this.setData({
>       name: '李四'
>     });
>   }
> });
> ```

上面代码中，`Page()`方法的参数配置对象里面，定义了`buttonHandler()`，这就是`<button>`元素的回调函数。它有几个地方需要注意。

（1）事件回调函数的参数是事件对象`event`，可以从它上面获取[事件信息](https://developers.weixin.qq.com/miniprogram/dev/framework/view/wxml/event.html)，比如事件类型、发生时间、发生节点、当前节点等等。

（2）事件回调函数内部的`this`，指向页面实例。

（3）页面实例的`this.setData()`方法，可以更改配置对象的`data`属性，进而通过数据绑定机制，导致页面上的全局变量发生变化。

开发者工具导入项目代码，点击按钮后，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100409.jpg)

可以看到，点击按钮以后，页面的文字从"hello 张三"变成了"hello 李四"。

这个示例的完整代码，可以查看[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/10.events)。

这里要强调一下，修改页面配置对象的`data`属性时，不要直接修改`this.data`，这不仅无法触发事件绑定机制去变更页面，还会造成数据不一致，所以一定要通过`this.setData()`去修改。`this.setData()`是一个很重要的方法，有很多细节，详细介绍可以读一下[官方文档](https://developers.weixin.qq.com/miniprogram/dev/reference/api/Page.html#Page-prototype-setData-Object-data-Function-callback)。

## 十七、动态提示 Toast

小程序提供了很多组件和方法，用来增强互动效果。比如，每次操作后，都显示一个动态提示，告诉用户操作的结果，这种效果叫做 Toast。

打开`home.js`文件，为`this.setData()`加上第二个参数。

> ```javascript
> Page({
>   data: {
>     name: '张三'
>   },
>   buttonHandler(event) {
>     this.setData({
>       name: '李四'
>     }, function () {
>       wx.showToast({
>         title: '操作完成',
>         duration: 700
>       });
>     }),
>   }
> });
> ```

上面代码中，`this.setData()`方法加入了第二个参数，这是一个函数，它会在页面变更完毕后（即`this.setData()`执行完）自动调用。

这个参数函数内部，调用了`wx.showToast()`方法，`wx`是小程序提供的原生对象，所有客户端 API 都定义在这个对象上面，`wx.showToast()`会展示微信内置的动态提示框，它的参数对象的`title`属性指定提示内容，`duration`属性指定提示框的展示时间，单位为毫秒。

开发者工具导入项目代码，点击按钮后，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100410.jpg)

过了700毫秒，提示框就会自动消失。

这个示例的完整代码，可以查看[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/11.toast)。

## 十八、对话框 Modal

下面，我们再用小程序提供的`wx.showModal()`方法，制作一个对话框，即用户可以选择"确定"或"取消"。

打开`home.js`文件，修改如下。

> ```javascript
> Page({
>   data: {
>     name: '张三'
>   },
>   buttonHandler(event) {
>     const that = this;
>     wx.showModal({
>       title: '操作确认',
>       content: '你确认要修改吗？',
>       success (res) {      
>         if (res.confirm) {
>           that.setData({
>             name: '李四'
>           }, function () {
>              wx.showToast({
>                title: '操作完成',
>                duration: 700
>              });
>           });
>         } else if (res.cancel) {
>           console.log('用户点击取消');
>         }
>       }
>     });
>   }
> });
> ```

上面代码中，用户点击按钮以后，回调函数`buttonHandler()`里面会调用`wx.showModal()`方法，显示一个对话框，效果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100411.jpg)

`wx.showModal()`方法的参数是一个配置对象。`title`属性表示对话框的标题（本例为"操作确认"），`content`属性表示对话框的内容（本例为"你确认要修改吗？"），`success`属性指定对话框成功显示后的回调函数，`fail`属性指定显示失败时的回调函数。

`success`回调函数里面，需要判断一下用户到底点击的是哪一个按钮。如果参数对象的`confirm`属性为`true`，点击的就是"确定"按钮，`cancel`属性为`true`，点击的就是"取消"按钮。

这个例子中，用户点击"取消"按钮后，对话框会消失，控制台会输出一行提示信息。点击"确定"按钮后，对话框也会消失，并且还会去调用`that.setData()`那些逻辑。

注意，上面代码写的是`that.setData()`，而不是`this.setData()`。这是因为`setData()`方法定义在页面实例上面，但是由于`success()`回调函数不是直接定义在`Page()`的配置对象下面，`this`不会指向页面实例，导致`this.setData()`会报错。解决方法就是在`buttonHandler()`的开头，将`this`赋值给变量`that`，然后在`success()`回调函数里面使用`that.setData()`去调用。关于`this`更详细的解释，可以参考[这篇教程](https://wangdoc.com/javascript/oop/this.html)。

## 十九、WXML 渲染语法

前面说过，小程序的页面结构使用 WXML 语言进行描述。

WXML 的全称是微信页面标签语言（Weixin Markup Language），它不仅提供了许多功能标签，还有一套自己的语法，可以设置页面渲染的生效条件，以及进行循环处理。

微信 API 提供的数据，就通过 WXML 的渲染语法展现在页面上。比如，`home.js`里面的数据源是一个数组。

> ```javascript
> Page({
>   data: {
>     items: ['事项 A', '事项 B', '事项 C']
>   }
> });
> ```

上面代码中，`Page()`的参数配置对象的`data.items`属性是一个数组。通过数据绑定机制，页面可以读取全局变量`items`，拿到这个数组。

拿到数组以后，怎样将每一个数组成员展现在页面上呢？WXML 的数组循环语法，就是一个很简便的方法。

打开`home.wxml`，改成下面的代码。

> ```markup
> <view>
>   <text class="title" wx:for="{{items}}">
>     {{index}}、 {{item}}
>    </text>
> </view>
> ```

上面代码中，`<text>`标签的`wx:for`属性，表示当前标签（`<text>`）启用数组循环，处理`items`数组。数组有多少个成员，就会生成多少个`<text>`。渲染后的页面结构如下。

> ```markup
> <view>
>   <text>...</text>
>   <text>...</text>
>   <text>...</text>
> </view>
> ```

在循环体内，当前数组成员的位置序号（从`0`开始）绑定变量`index`，成员的值绑定变量`item`。

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100507.jpg)

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/13.wxml)。

WXML 的其他渲染语法（主要是条件判断和对象处理），请查看[官方文档](https://developers.weixin.qq.com/miniprogram/dev/reference/wxml/list.html)。

## 二十、客户端数据储存

页面渲染用到的外部数据，如果每次都从服务器或 API 获取，有时可能会比较慢，用户体验不好。

小程序允许将一部分数据保存在客户端（即微信 App）的本地储存里面（其实就是自定义的缓存）。下次需要用到这些数据的时候，就直接从本地读取，这样就大大加快了渲染。本节介绍怎么使用客户端数据储存。

打开`home.wxml`，改成下面的代码。

> ```markup
> <view>
>   <text class="title" wx:for="{{items}}">
>     {{index}}、 {{item}}
>    </text>
>    <input placeholder="输入新增事项" bind:input="inputHandler"/>
>    <button bind:tap="buttonHandler">确定</button>
> </view>
> ```

上面代码除了展示数组`items`，还新增了一个输入框和一个按钮，用来接受用户的输入。背后的意图是，用户通过输入框，为`items`数组加入新成员。

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100509.jpg)

注意，输入框有一个`input`事件的监听函数`inputHandler`（输入内容改变时触发），按钮有一个`tap`事件的监听函数`buttonHandler`（点击按钮时触发）。这两个监听函数负责处理用户的输入。

然后，打开`home.js`，代码修改如下。

> ```javascript
> Page({
>   data: {
>     items: [],
>     inputValue: ''
>   },
>   inputHandler(event) {
>     this.setData({
>       inputValue: event.detail.value || ''
>     });
>   },
>   buttonHandler(event) {
>     const newItem = this.data.inputValue.trim();
>     if (!newItem) return;
>     const itemArr = [...this.data.items, newItem];
>     wx.setStorageSync('items', itemArr);
>     this.setData({ items: itemArr });
>   },
>   onLoad() {
>     const itemArr = wx.getStorageSync('items') || []; 
>     this.setData({ items: itemArr });
>   }
> });
> ```

上面代码中，输入框监听函数`inputHandler()`只做了一件事，就是每当用户的输入发生变化时，先从事件对象`event`的`detail.value`属性上拿到输入的内容，然后将其写入全局变量`inputValue`。如果用户删除了输入框里面的内容，`inputValue`就设为空字符串。

按钮监听函数`buttonHandler()`是每当用户点击提交按钮，就会执行。它先从`inputValue`拿到用户输入的内容，确定非空以后，就将其加入`items`数组。然后，使用微信提供的`wx.setStorageSync()`方法，将`items`数组存储在客户端。最后使用`this.setData()`方法更新一下全局变量`items`，进而触发页面的重新渲染。

`wx.setStorageSync()`方法属于小程序的客户端数据储存 API，用于将数据写入客户端储存。它接受两个参数，分别是键名和键值。与之配套的，还有一个`wx.getStorageSync()`方法，用于读取客户端储存的数据。它只有一个参数，就是键名。这两个方法都是同步的，小程序也提供异步版本，请参考[官方文档](https://developers.weixin.qq.com/miniprogram/dev/api/storage/wx.setStorage.html)。

最后，上面代码中，`Page()`的参数配置对象还有一个`onLoad()`方法。该方法属于页面的生命周期方法，页面加载后会自动执行该方法。它只执行一次，用于页面初始化，这里的意图是每次用户打开页面，都通过`wx.getStorageSync()`方法，从客户端取出以前存储的数据，显示在页面上。

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/14.storage)。

必须牢记的是，客户端储存是不可靠的，随时可能消失（比如用户清理缓存）。用户换了一台手机，或者本机重装微信，原来的数据就丢失了。所以，它只适合保存一些不重要的临时数据，最常见的用途一般就是作为缓存，加快页面显示。

## 二十一、远程数据请求

小程序可以从外部服务器读取数据，也可以向服务器发送数据。本节就来看看怎么使用小程序的网络能力。

微信规定，只有后台登记过的服务器域名，才可以进行通信。不过，开发者工具允许开发时放松这个限制。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100510.jpg)

按照上图，点击开发者工具右上角的三条横线（"详情"），选中"不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书" 。这样的话，小程序在开发时，就可以跟服务器进行通信了。

下面，我们在本地启动一个开发服务器。为了简单起见，我选用了 [json-server](https://www.npmjs.com/package/json-server) 作为本地服务器，它的好处是只要有一个 JSON 数据文件，就能自动生成 RESTful 接口。

首先，新建一个数据文件`db.json`，内容如下。

> ```javascript
> {
>   "items": ["事项 A", "事项 B", "事项 C"]
> }
> ```

然后，确认本机安装了 Node.js 以后，进入`db.json`所在的目录，在命令行执行下面命令，启动服务器。

> ```bash
> npx json-server db.json
> ```

正常情况下，这时你打开浏览器访问`localhost:3000/items`这个网址，就能看到返回了一个数组`["事项 A", "事项 B", "事项 C"]`。

接着，打开`home.js`，代码修改如下。

> ```javascript
> Page({
>   data: { items: [] },
>   onLoad() {
>     const that = this;
>     wx.request({
>       url: 'http://localhost:3000/items',
>       success(res) {
>         that.setData({ items: res.data });
>       }
>     });
>   }
> });
> ```

上面代码中，生命周期方法`onLoad()`会在页面加载后自动执行，这时就会执行`wx.request()`方法去请求远程数据。如果请求成功，就会执行回调函数`succcess()`，更新页面全局变量`items`，从而让远程数据显示在页面上。

`wx.request()`方法就是小程序的网络请求 API，通过它可以发送 HTTP 请求。它的参数配置对象最少需要指定`url`属性（请求的网址）和`succcess()`方法（服务器返回数据的处理函数）。其他参数请参考[官方文档](https://developers.weixin.qq.com/miniprogram/dev/api/network/request/wx.request.html)。

开发者工具导入项目代码，页面渲染结果如下。它的初始数据是从服务器拿到的。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100509.jpg)

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/15.web-request)。

这个例子只实现了远程数据获取，json-server 实际上还支持数据的新增和删改，大家可以作为练习，自己来实现。

## 二十二、`<open-data>`组件

如果要在页面上展示当前用户的身份信息，可以使用小程序提供的[``组件](https://developers.weixin.qq.com/miniprogram/dev/component/open-data.html)。

打开`home.wxml`文件，代码修改如下。

> ```markup
> <view>
>   <open-data type="userAvatarUrl"></open-data>
>   <open-data type="userNickName"></open-data>
> </view>
> ```

上面代码中，`<open-data>`组件的`type`属性指定所要展示的信息类型，`userAvatarUrl`表示展示用户头像，`userNickName`表示用户昵称。

开发者工具导入项目代码，页面渲染结果如下，显示你的头像和用户昵称。

![img](https://www.wangbase.com/blogimg/asset/202011/bg2020110205.jpg)

`<open-data>`支持的用户信息如下。

> - `userNickName`：用户昵称
> - `userAvatarUrl`：用户头像
> - `userGender`：用户性别
> - `userCity`：用户所在城市
> - `userProvince`：用户所在省份
> - `userCountry`：用户所在国家
> - `userLanguage`：用户的语言

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/16.open-data)。

`<open-data>`不需要用户授权，也不需要登录，所以用起来很方便。但也是因为这个原因，小程序不允许用户脚本读取`<open-data>`返回的信息。

## 二十三、获取用户个人信息

如果想拿到用户的个人信息，必须得到授权。官方建议，通过按钮方式获取授权。

打开`home.wxml`文件，代码修改如下。

> ```markup
> <view>
>   <text class="title">hello {{name}}</text>
>   <button open-type="getUserInfo" bind:getuserinfo="buttonHandler">
>     授权获取用户个人信息
>   </button>
> </view>
> ```

上面代码中，`<button>`标签的`open-type`属性，指定按钮用于获取用户信息，`bind:getuserinfo`属性表示点击按钮会触发`getuserinfo`事件，即跳出对话框，询问用户是否同意授权。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100603.jpg)

用户点击"允许"，脚本就可以得到用户信息。

`home.js`文件的脚本代码如下。

> ```javascript
> Page({
>   data: { name: '' },
>   buttonHandler(event) {
>     if (!event.detail.userInfo) return;
>     this.setData({
>       name: event.detail.userInfo.nickName
>     });
>   }
> });
> ```

上面代码中，`buttonHandler()`是按钮点击的监听函数，不管用户点击"拒绝"或"允许"，都会执行这个函数。我们可以通过事件对象`event`有没有`detail.userInfo`属性，来判断用户点击了哪个按钮。如果能拿到`event.detail.userInfo`属性，就表示用户允许读取个人信息。这个属性是一个对象，里面就是各种用户信息，比如头像、昵称等等。

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/17.user-info)。

实际开发中，可以先用`wx.getSetting()`方法判断一下，用户是否已经授权过。如果已经授权过，就不用再次请求授权，而是直接用`wx.getUserInfo()`方法获取用户信息。

注意，这种方法返回的用户信息之中，不包括能够真正识别唯一用户的`openid`属性。这个属性需要用到保密的小程序密钥去请求，所以不能放在前端获取，而要放在后端。这里就不涉及了。

## 二十四、多页面的跳转

真正的小程序不会只有一个页面，而是多个页面，所以必须能在页面之间实现跳转。

`app.json`配置文件的`pages`属性就用来指定小程序有多少个页面。

> ```javascript
> {
>   "pages": [
>     "pages/home/home",
>     "pages/second/second"
>   ],
>   "window": ...
> }
> ```

上面代码中，`pages`数组包含两个页面。以后每新增一个页面，都必须把页面路径写在`pages`数组里面，否则就是无效页面。排在第一位的页面，就是小程序打开时，默认展示的页面。

新建第二个页面的步骤如下。

第一步，新建`pages/second`目录。

第二步，在该目录里面，新建文件`second.js`，代码如下。

> ```javascript
> Page({});
> ```

第三步，新建第二页的页面文件`second.wxml`，代码如下。

> ```markup
> <view>
>   <text class="title">这是第二页</text>
>   <navigator url="../home/home">前往首页</navigator>
> </view>
> ```

上面代码中，`<navigator>`就是链接标签，相当于网页标签`<a>`，只要用户点击就可以跳转到`url`属性指定的页面（这里是第一页的位置）。

第四步，修改第一页的页面文件`home.wxml`，让用户能够点击进入第二页。

> ```markup
> <view>
>   <text class="title">这是首页</text>
>   <navigator url="../second/second">前往第二页</navigator>
> </view>
> ```

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100604.jpg)

用户点击"前往第二页"，就会看到第二个页面。

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/18.multi-pages)。

## 二十五、wx.navigateTo()

除了使用`<navigator>`组件进行页面跳转，小程序也提供了页面跳转的脚本方法`wx.navigateTo()`。

首先，打开`home.wxml`文件，代码修改如下。

> ```markup
> <view>
>   <text class="title">这是首页</text>
>   <button bind:tap="buttonHandler">前往第二页</button>
> </view>
> ```

开发者工具导入项目代码，页面渲染结果如下。

![img](https://www.wangbase.com/blogimg/asset/202010/bg2020100605.jpg)

然后，打开`home.js`文件，代码修改如下。

> ```javascript
> Page({
>   buttonHandler(event) {
>     wx.navigateTo({
>       url: '../second/second'
>     });
>   }
> });
> ```

上面代码中，`buttonHandler()`是按钮点击的监听函数，只要用户点击按钮，就会调用`wx.navigateTo()`方法。该方法的参数是一个配置对象，该对象的`url`属性指定了跳转目标的位置，自动跳转到那个页面。

这个示例的完整代码，可以参考[代码仓库](https://github.com/ruanyf/wechat-miniprogram-demos/tree/master/demos/19.wx-navigateto)。