---
layout: post
title: "Linux Operation"
date: 2020-09-04 
description: "Learn Linux from today"
tag: Linux
---

### Funny command: 

- 召唤眼睛：$ nohup xeyes & 
- 数字雨：$ sudo apt-get install cmatrix； cmatrix -C red
- 改变字体：Sudo apt-get install sysvbanner / toilet / figlet
- 火炉：sudo apt-get install libaa-bin； aafire
- 动物说话： sudo apt install -y cowsay； cowsay hello
### Usual command: 

- 添加新用户： sudo adduser <username>

- 删除用户： sudo deluser <username> - -remove-home

- 删除用户组： sudo grouped <groupname>

- 切换用户： su -l <username>

- 查看用户：who am I / whoami /

- 为用户添加密码：sudo passwd <username>

- 退出当前用户： exit / ctrl+ D

- 查看用户组方法1：groups <username> 

- 查看用户组方法2: cat /etc/group | sort / | grep - E “username”

- 为用户添加超级用户组：sudo usermod -G sudo <username> (拥有root权限的才可操作)

- 变更文件所有者 sudo chown <username> <filename>

- 修改文件权限  sudo chmod 600 <filename> / $ chmod go-rw <filename>    [g、o 还有 u 分别表示 group（用户组）、others（其他用户） 和 user（用户），+ 和 - 分别表示增加和去掉相应的权限]

- 创建文件：touch <filename>

- 批量创建文件：touch <filename>{1..5}.txt

- 创建目录：mkdir <dirname>  使用 -p 参数，同时创建父目录（如果不存在该父目录)

- 复制文件: cp <filename> <dirname> 成功复制目录需要加上 -r 或者 -R 参数，表示递归复制

- 删除文件： rm <filename> 可以使用 -f 参数强制删除

- 删除目录： rm -r <dirname>

- 移动文件： mv <filename> <dirname>

- 重命名： mv <oldfilename> <newfilename> 

- 批量重命名：rename + perl 正则表达式

- 查看文件： 使用 cat，tac 和 nl 命令查看文件 前两个命令都是用来打印文件内容到标准输出（终端），其中 cat 为正序显示，tac 为倒序显示, 可以加上 -n 参数显示行号。 nl 命令，添加行号并打印，这是个比 cat -n 更专业的行号打印命令。

- 分页查看文件:  more/less <filename> 可以使用 Enter 键向下滚动一行，使用 Space 键向下滚动一屏，按下 h 显示帮助，q 退出

- 查看文件头尾： tail / head -n 行数 <filename>

- 查看文件类型： file <filename>

- 声明变量：declare <var>

- 查看shell：cat /etc/shells

- 搜索文件： 与搜索相关的命令常用的有 whereis，which，find 和 locate. 

  - Whereis <filename> whereis 只能搜索二进制文件(-b)，man 帮助文件(-m)和源代码文件(-s).  

  - Locate <filename> 在这之前要先更新数据库，如果想只统计数目可以加上 -c 参数，-i 参数可以忽略大小写进行查找
    $ sudo apt-get update
    $ sudo apt-get install locate
     	$ sudo updatedb

  - which 本身是 Shell 内建的一个命令，我们通常使用 which 来确定是否安装了某个指定的程序，因为它只从 PATH 环境变量指定的路径中去搜索命令并且返回第一个搜索到的结果。 Which man

  - find 命令的路径是作为第一个参数的， 基本命令格式为 find [path][option] [action]    find <dirname> -name <filename>

- 打包文件：zip -r  -9 -q -o <zipfilename.zip> <dirname/filename> -x ~/*.zip   -r 参数表示递归打包包含子目录的全部内容，-q 参数表示为安静模式，即不向屏幕输出信息，-o，表示输出文件，需在其后紧跟打包输出文件名。这里添加了一个参数用于设置压缩级别 -[1-9]，1 表示最快压缩但体积大，9 表示体积最小但耗时最久。最后那个 -x 是为了所有zip 文件。-e 参数可以创建加密压缩包。
- 查看所有压缩文件大小并排序： du -h -d 0 *.zip | sort 
- 使用安静模式，将文件解压到指定目录: unzip -q <zipfile> -d <dirname>;  -O（英文字母，大写 o）参数指定编码类型 -O GBK
- 查看压缩包内容：unzip -l <zipfile> 
- 创建tar包：tar -P -v -cf <newtarname.tar> <dirname>/<filename>   -P 保留绝对路径符，-c 表示创建一个 tar 包文件，-f 用于指定创建的文件名，注意文件名必须紧跟在 -f 参数之后; -v 参数以可视的的方式输出打包的文件。
- 解包tar包：tar -xf <tarfile> -C <existingdriname> 解包一个文件（-x 参数）到指定路径的已存在目录（-C 参数); 只查看不解包文件 -t 参数.
- 创建不同的压缩格式的文件: 只需要在创建 tar 文件的基础上添加 -z -J -j参数，使用 gzip 来压缩文件. Tar -czf <newfilename.tar.gz> <dirname>
- 查看磁盘的容量: df -h 
- 查看目录的容量：du -h -d 0 <dirname> -d 表示目录的深度
- 使用 dd 命令创建虚拟镜像文件: $ dd if=/dev/zero of=virtual.img bs=1M count=256.    bs（block size）用于指定块大小（缺省单位为 Byte，也可为其指定如'K'，'M'，'G'等单位），count用于指定块数量
- 使用 mkfs 命令格式化磁盘: $ sudo mkfs.ext4 virtual.img
- 挂载： mount [-o [操作选项]] [-t 文件系统类型] [-w|--rw|--ro] [文件系统源] [挂载点]
- 启动 crontab： sudo cron -f &
- 为当前用户添加计划任务: crontab -e
- 为root用户添加计划任务: sudo crontab -e
- 打印/etc/passwd文件中以:为分隔符的第 1 个字段和第 6 个字段分别表示用户名和其家目录: $ cut /etc/passwd -d ':' -f 1,6
- 打印/etc/passwd文件中每一行的前 N 个字符： 

  - 前五个（包含第五个）$ cut /etc/passwd -c -5

  - 前五个之后的（包含第五个）$ cut /etc/passwd -c 5-

  - 第五个 $ cut /etc/passwd -c 5

  - 2到5之间的（包含第五个）$ cut /etc/passwd -c 2-5

- 分别只输出行数、单词数、字节数、字符数和输入文本中最长一行的字节数：

  - 行数 $ wc -l /etc/passwd
  - 单词数 $ wc -w /etc/passwd
  - 字节数 $ wc -c /etc/passwd
  - 字符数 $ wc -m /etc/passwd
  - 最长行字节数 $ wc -L /etc/passwd

- 按特定字段排序：$ cat /etc/passwd | sort -t':' -k 3 -n; 上面的-t参数用于指定字段的分隔符，这里是以":"作为分隔符；-k 字段号用于指定对哪一个字段进行排序。这里/etc/passwd文件的第三个字段为数字，默认情况下是以字典序排序的，如果要按照数字排序就要加上-n参数：

- 删除一段文本信息中的某些文字 : tr [option]...SET1 [SET2] 
选项	说明
-d	删除和 set1 匹配的字符，注意不是全词匹配也不是按字符顺序匹配
-s	去除 set1 指定的在输入文本中连续并重复的字符
- 将输入文本，全部转换为大写或小写输出: tr '[:lower:]' '[:upper:]'
- col 命令可以将Tab换成对等数量的空格键，或反转这个操作: col -x/ -h.  -x将Tab转换为空格, -h将空格转换为Tab（默认选项）
- 两个文件中包含相同内容的那一行合并在一起: join [option]... file1 file2 
选项	说明
-t	指定分隔符，默认为空格
-i	忽略大小写的差异
-1	指明第一个文件要用哪个字段来对比，默认对比第一个字段
-2	指明第二个文件要用哪个字段来对比，默认对比第一个字段

- 不对比数据的情况下，简单地将多个文件合并一起，以Tab隔开: paste [option] file...
选项	说明
-d	指定合并的分隔符，默认为 Tab
-s	不合并到一行，每个文件为一行


- Sudo apt-get reinstall install <pkg>
- Sudo apt-get remove <pkg>
- Sudo apt-get search <pkg>


- 使用较长格式列出文件 ls -l
  ￼
  ￼每个文件有三组固定的权限，分别对应拥有者，所属用户组，其他用户，记住这个顺序是固定的。文件的读写执行对应字母 rwx，以二进制表示就是 111，用十进制表示就是 7, 相对应的有6（不能执行）， 4（不能写入和执行）。 777， 666，600。。。
  ￼
- 环境变量和进程
  - set	显示当前 Shell 所有变量，包括其内建环境变量（与 Shell 外观等相关），用户自定义变量及导出的环境变量。
  - env	显示与当前用户相关的环境变量，还可以让命令在指定环境中运行。
  - export	显示从 Shell 中导出成环境变量的变量，也能通过它将自定义变量导出为环境变量。
- 变量修改
  变量设置方式	说明
  ${变量名#匹配字串}	从头向后开始匹配，删除符合匹配字串的最短数据
  ${变量名##匹配字串}	从头向后开始匹配，删除符合匹配字串的最长数据
  ${变量名%匹配字串}	从尾向前开始匹配，删除符合匹配字串的最短数据
  ${变量名%%匹配字串}	从尾向前开始匹配，删除符合匹配字串的最长数据
  ${变量名/旧的字串/新的字串}	将符合旧字串的第一个字串替换为新的字串
  ${变量名//旧的字串/新的字串}	将符合旧字串的全部字串替换为新的字串
  unset 命令删除一个环境变量
  Source 让环境变量立即生效 source 命令还有一个别名就是 .

与搜索相关的命令常用的有 whereis，which，find 和 locate

/etc/group
————————
/etc/group 的内容包括用户组（Group）、用户组口令、GID（组 ID） 及该用户组所包含的用户（User），每个用户组一条记录.

格式：
group_name:password:GID:user_list

如果用户的 GID 等于用户组的 GID，那么最后一个字段 user_list 就是空的
————————

### Linux 的目录结构

>> 表示将标准输出以追加的方式重定向到一个文件中，注意前面用到的 > 是以覆盖的方式重定向到一个文件中，使用的时候一定要注意分辨


文件打包和解压缩
文件后缀名	说明
*.zip	zip 程序打包压缩的文件
*.rar	rar 程序压缩的文件
*.7z	7zip 程序压缩的文件
*.tar	tar 程序打包，未压缩的文件
*.gz	gzip 程序（GNU zip）压缩的文件
*.xz	xz 程序压缩的文件
*.bz2	bzip2 程序压缩的文件
*.tar.gz	tar 打包，gzip 程序压缩的文件
*.tar.xz	tar 打包，xz 程序压缩的文件
*tar.bz2	tar 打包，bzip2 程序压缩的文件
*.tar.7z	tar 打包，7z 程序压缩的文件

压缩文件格式	参数
*.tar.gz	-z
*.tar.xz	-J
*tar.bz2	-j

Man 命令
章节数	说明
1	Standard commands （标准命令）
2	System calls （系统调用）
3	Library functions （库函数）
4	Special devices （设备说明）
5	File formats （文件格式）
6	Games and toys （游戏和娱乐）
7	Miscellaneous （杂项）
8	Administrative Commands （管理员命令）
9	其他（Linux 特定的）， 用来存放内核例行程序的文档。

文件描述符	设备文件	说明
0	/dev/stdin	标准输入
1	/dev/stdout	标准输出
2	/dev/stderr	标准错误

### 正则表达式

* |竖直分隔符表示选择，例如"boy|girl"可以匹配"boy"或者"girl"
* +表示前面的字符必须出现至少一次(1 次或多次)，例如，"goo+gle",可以匹配"gooogle","goooogle"等；
* ?表示前面的字符最多出现一次(0 次或 1 次)，例如，"colou?r",可以匹配"color"或者"colour";
* *星号代表前面的字符可以不出现，也可以出现一次或者多次（0 次、或 1 次、或多次），例如，“0*42”可以匹配 42、042、0042、00042 等。
* ()圆括号可以用来定义模式字符串的范围和优先级，这可以简单的理解为是否将括号内的模式串作为一个整体。例如，"gr(a|e)y"等价于"gray|grey"

  字符	描述
  \	将下一个字符标记为一个特殊字符、或一个原义字符。例如，“n”匹配字符“n”。“\n”匹配一个换行符。序列“\\”匹配“\”而“\(”则匹配“(”。
  ^	匹配输入字符串的开始位置。
  $	匹配输入字符串的结束位置。
  {n}	n 是一个非负整数。匹配确定的 n 次。例如，“o{2}”不能匹配“Bob”中的“o”，但是能匹配“food”中的两个 o。
  {n,}	n 是一个非负整数。至少匹配 n 次。例如，“o{2,}”不能匹配“Bob”中的“o”，但能匹配“foooood”中的所有 o。“o{1,}”等价于“o+”。“o{0,}”则等价于“o*”。
  {n,m}	m 和 n 均为非负整数，其中 n<=m。最少匹配 n 次且最多匹配 m 次。例如，“o{1,3}”将匹配“fooooood”中的前三个 o。“o{0,1}”等价于“o?”。请注意在逗号和两个数之间不能有空格。

* 匹配前面的子表达式零次或多次。例如，zo*能匹配“z”、“zo”以及“zoo”。*等价于{0,}。

+	匹配前面的子表达式一次或多次。例如，“zo+”能匹配“zo”以及“zoo”，但不能匹配“z”。+等价于{1,}。
?	匹配前面的子表达式零次或一次。例如，“do(es)?”可以匹配“do”或“does”中的“do”。?等价于{0,1}。
?	当该字符紧跟在任何一个其他限制符（*,+,?，{n}，{n,}，{n,m}）后面时，匹配模式是非贪婪的。非贪婪模式尽可能少的匹配所搜索的字符串，而默认的贪婪模式则尽可能多的匹配所搜索的字符串。例如，对于字符串“oooo”，“o+?”将匹配单个“o”，而“o+”将匹配所有“o”。
.	匹配除“\n”之外的任何单个字符。要匹配包括“\n”在内的任何字符，请使用像“(.｜\n)”的模式。
(pattern)	匹配 pattern 并获取这一匹配的子字符串。该子字符串用于向后引用。要匹配圆括号字符，请使用“\(”或“\)”。
x ｜ y	匹配 x 或 y。例如，“z ｜ food”能匹配“z”或“food”。“(z ｜ f)ood”则匹配“zood”或“food”。
[xyz]	字符集合（character class）。匹配所包含的任意一个字符。例如，“[abc]”可以匹配“plain”中的“a”。其中特殊字符仅有反斜线\保持特殊含义，用于转义字符。其它特殊字符如星号、加号、各种括号等均作为普通字符。脱字符^如果出现在首位则表示负值字符集合；如果出现在字符串中间就仅作为普通字符。连字符 - 如果出现在字符串中间表示字符范围描述；如果出现在首位则仅作为普通字符。
[^xyz]	排除型（negate）字符集合。匹配未列出的任意字符。例如，“[^abc]”可以匹配“plain”中的“plin”。
[a-z]	字符范围。匹配指定范围内的任意字符。例如，“[a-z]”可以匹配“a”到“z”范围内的任意小写字母字符。
[^a-z]	排除型的字符范围。匹配任何不在指定范围内的任意字符。例如，“[^a-z]”可以匹配任何不在“a”到“z”范围内的任意字符。

在通过grep命令使用正则表达式之前，先介绍一下它的常用参数
参数	说明
-b	将二进制文件作为文本来进行匹配
-c	统计以模式匹配的数目
-i	忽略大小写
-n	显示匹配文本所在行的行号
-v	反选，输出不匹配行的内容
-r	递归匹配查找
-A n	n 为正整数，表示 after 的意思，除了列出匹配行之外，还列出后面的 n 行
-B n	n 为正整数，表示 before 的意思，除了列出匹配行之外，还列出前面的 n 行
--color=auto	将输出中的匹配项设置为自动颜色显示

特殊符号	说明
[:alnum:]	代表英文大小写字母及数字，亦即 0-9, A-Z, a-z
[:alpha:]	代表任何英文大小写字母，亦即 A-Z, a-z
[:blank:]	代表空白键与 [Tab] 按键两者
[:cntrl:]	代表键盘上面的控制按键，亦即包括 CR, LF, Tab, Del.. 等等
[:digit:]	代表数字而已，亦即 0-9
[:graph:]	除了空白字节 (空白键与 [Tab] 按键) 外的其他所有按键
[:lower:]	代表小写字母，亦即 a-z
[:print:]	代表任何可以被列印出来的字符
[:punct:]	代表标点符号 (punctuation symbol)，亦即：" ' ? ! ; : # $...
[:upper:]	代表大写字母，亦即 A-Z
[:space:]	任何会产生空白的字符，包括空白键, [Tab], CR 等等
[:xdigit:]	代表 16 进位的数字类型，因此包括： 0-9, A-F, a-f 的数字与字节

### sed 命令基本格式：

[n1][,n2]command
[n1][~step]command

其中一些命令可以在后面加上作用范围，形如：

$ sed -i 's/sad/happy/g' test # g表示全局范围
$ sed -i 's/sad/happy/4' test # 4表示指定行中的第四个匹配字符串
命令	说明
s	行内替换
c	整行替换
a	插入到指定行的后面
i	插入到指定行的前面
p	打印指定行，通常与-n参数配合使用
d	删除指定行

参数	说明
-n	安静模式，只打印受影响的行，默认打印输入数据的全部内容
-e	用于在脚本中添加多个执行命令一次执行，在命令行中执行多个命令通常不需要加该参数
-f filename	指定执行 filename 文件中的命令
-r	使用扩展正则表达式，默认为标准正则表达式
-i	将直接修改输入文件内容，而不是打印到标准输出设备


ps 也是我们最常用的查看进程的工具之一
- Ps aux
- Ps axjf
- ps -afxo user,ppid,pid,pgid,command
内容	解释
F	进程的标志（process flags），当 flags 值为 1 则表示此子程序只是 fork 但没有执行 exec，为 4 表示此程序使用超级管理员 root 权限
USER	进程的拥有用户
PID	进程的 ID
PPID	其父进程的 PID
SID	session 的 ID
TPGID	前台进程组的 ID
%CPU	进程占用的 CPU 百分比
%MEM	占用内存的百分比
NI	进程的 NICE 值
VSZ	进程使用虚拟内存大小
RSS	驻留内存中页的大小
TTY	终端 ID
S or STAT	进程状态
WCHAN	正在等待的进程资源
START	启动进程的时间
TIME	进程消耗 CPU 的时间
COMMAND	命令的名称和参数
TPGID栏写着-1 的都是没有控制终端的进程，也就是守护进程
STAT表示进程的状态，而进程的状态有很多，如下表所示
状态	解释
R	Running.运行中
S	Interruptible Sleep.等待调用
D	Uninterruptible Sleep.不可中断睡眠
T	Stoped.暂停或者跟踪状态
X	Dead.即将被撤销
Z	Zombie.僵尸进程
W	Paging.内存交换
N	优先级低的进程
<	优先级高的进程
s	进程的领导者
L	锁定状态
l	多线程状态
+	前台进程

pstree -up
#参数选择：
#-A  ：各程序树之间以 ASCII 字元來連接；
#-p  ：同时列出每个 process 的 PID；
#-u  ：同时列出每个 process 的所屬账户名称。

### 常见的日志

日志名称	记录信息
alternatives.log	系统的一些更新替代信息记录
apport.log	应用程序崩溃信息记录
apt/history.log	使用 apt-get 安装卸载软件的信息记录
apt/term.log	使用 apt-get 时的具体操作，如 package 的下载、打开等
auth.log	登录认证的信息记录
boot.log	系统启动时的程序服务的日志信息
btmp	错误的信息记录
Consolekit/history	控制台的信息记录
dist-upgrade	dist-upgrade 这种更新方式的信息记录
dmesg	启动时，显示屏幕上内核缓冲信息,与硬件有关的信息
dpkg.log	dpkg 命令管理包的日志。
faillog	用户登录失败详细信息记录
fontconfig.log	与字体配置有关的信息记录
kern.log	内核产生的信息记录，在自己修改内核时有很大帮助
lastlog	用户的最近信息记录
wtmp	登录信息的记录。wtmp 可以找出谁正在进入系统，谁使用命令显示这个文件或信息等
syslog	系统信息记录


￼
