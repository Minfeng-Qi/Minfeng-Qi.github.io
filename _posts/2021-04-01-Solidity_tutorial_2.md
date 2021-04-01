---
layout: post
title: "Solidity Tutorial (2)"
date: 2021-04-01
description: "Solidity Tutorial_2"
tag: Blockchain
---

```solidity
pragma solidity ^0.4.4;
/*
pragma：版本声明
Solidity：开发语言
0.4.4：当前合约的版本，0.4代表主版本，4代表修复bug的升级版本
^：代表向上兼容，0.4.4 ～ 0.4.9可以对我们当前的代码进行编译
*/
 
contract Person {
  uint _age;
  uint _height;
  address _owner; // 合约的拥有者
 
  constructor () public{
    _age = 29;
    _height = 180;
    _owner = msg.sender;
  }
 
  function owner() public constant returns (address){
    return _owner;
  }
 
  function getAge() public constant returns(uint) {
    return _age;
  }
 
  function getHeight() public constant returns(uint) {
    return _height;
  }
 
  function setAge(uint age) public {
    _age = age;
  }
 
  function setHeight(uint height) public {
    _height = height;
  }
 
  function kill() public {
    if(_owner == msg.sender){
      selfdestruct(_owner); // 摧毁拥有者的合约
    }
  }

```

### 注释1：Contract

Contract是一个内置的对象，上面的语法就类似于class Person（子类） extends Contract（父类），意味着Person继承于Contract。
对象内有其构造函数constructor，用以对实例进行初始化。
当合约部署的时候，就是产生一个合约的实例，来自同一个地址的相同合约只能部署一次，只有一个专有的合约地址。

### 注释2：状态变量

_age , _height都是状态变量，在Contract中就相当于其属性变量；
在 Solidity 中，有两个地方可以存储变量 —— storage以及memory。
Storage 变量是指永久存储在区块链中的变量。 Memory 变量则是临时的，当外部函数对某合约调用完成时，内存型变量即被移除。
状态变量（在函数之外声明的变量）默认为“storage”形式，并永久写入区块链；而在函数内部声明的变量默认是“memory”型的，它们函数调用结束后消失。
Storage 拿到的是引用/句柄/指针， memory 拿到的是一份拷贝。

### 注释3：函数和状态变量的可见性

因为Solidity有两种函数调用：

内部调用：不创建一个真实的EVM调用(也称为“消息调用”)；
外部的调用：要创建一个真实的EMV调用,
在智能合约中，函数和状态变量的可见性可以分为四种， public ， private ， internal 和 external ，函数默认可见性是 public ，状态变量的默认可见性是 internal 。

public - （任意访问，作为合约接口）可以通过内部调用或通过消息调用。对公共状态变量而言，会有的自动访问限制符的函数生成。

private - （仅当前合约内）私有函数和状态变量仅仅在定义该合约中可见， 在派生的合约中不可见。

internal - （仅当前合约及所继承的合约）

这些函数和状态变量只能内部访问(即在当前合约或由它派生的合约),而不使用（关键字）this 。
external - （仅外部访问，也是合约接口）它们可以从其他合约调用, 也可以通过事务调用。外部函数f不能被内部调用（在内部也只能用外部访问方式访问，即 f()不执行,但this.f()执行）。

### 注释4：函数的限制访问

在Solidity中  constant 、 view 、 pure 三个函数修饰词的作用是告诉编译器，函数不改变/不读取状态变量，这样函数执行就可以不消耗gas了，因为不需要矿工来验证。

在Solidity v4.17之前，只有constant，后续版本将constant拆成了view和pure。view的作用和constant一模一样，可以读取状态变量但是不能改；pure则更为严格，pure修饰的函数不能改也不能读状态变量，只能操作函数内部变量，否则编译通不过。

### 注释5：msg.sender 和 selfdestruct()

在Contract中有一些全局变量和函数，在我们编写智能合约的过程中可以直接调用，比如上面的 msg.sender 和 selfdestruct()  

msg的所有成员包括：

msg.sender  ：储存消息的发送者，即部署智能合约的账户地址
msg.value ：发送的消息的数量
msg.gas ：剩余的gas
msg.data：完整的calldata
msg.sig ：呼叫数据的前4个字节 

合约相关的方法：

selfdestruct ( address recipient ) ：摧毁目前的合同，将资金送到给定的地址
suicide ( address recipient )：同上，是别名
this：指当前合约，明确转换为地址

全部的全局变量和函数有很多，包括：

- Ether单元
- 时间单位
- 块相关
- msg相关
- tx相关
- 当前时间戳
- 错误处理
- 数学和加密功能
- 地址相关
- 合约相关

### 继承

状态变量的继承：可以继承public和internal，但不能继承private

函数的继承：只能继承Public，不能继承internal和private

### 调用外部函数以及关键字payable

```

pragma solidity ^0.4.0;

contract InfoFeed {

    function info() payable returns (uint ret) { // 如果这里有payable，说明该函数外部调用的时候必须发送ether和gas
        return msg.value;
    }
}

contract Consumer {

    function deposit() payable returns (uint){
        return msg.value;
    }

    function left() constant returns (uint){
        return this.balance;
    }

    function callFeed(address addr) returns (uint) {
        return InfoFeed(addr).info.value(1).gas(8000)();  
        // 调用InfoFeed合约中的info函数，使用value()和fas()方法向InfoFeed合约发送ether
    }

```

首先调用deposit()为Consumer合约存入一定量的ether。然后调用callFeed()通过value(1)的方式，向InfoFeed合约的info()函数发送1ether。如果不先充值，由于合约余额为0，余额不足会报错Invalid opcode。

InfoFeed.info()函数，必须使用payable关键字，否则不能通过value()选项来接收ether。

如果被调用的合约不存在，或者是不包代码的帐户，或调用的合约产生了异常，或者gas不足，均会造成函数调用发生异常。
.info.value(1).gas(8000)只是本地设置发送的数额和gas值，真正执行调用的是其后的括号.info.value(1).gas(8000)()。

代码InfoFeed(addr)进行了一个显示的类型转换，声明了我们确定知道给定的地址是InfoFeed类型。所以这里并不会执行构造器的初始化。显示的类型强制转换，需要极度小心，不要尝试调用一个你不知道类型的合约。

如果被调用的合约源码并不事前知道，和它们交互会有潜在的风险。当前合约会将自己的控制权交给被调用的合约，而对方几乎可以做任何事。即使被调用的合约是继承自一个已知的父合约，但继承的子合约仅仅被要求正确实现了接口。合约的实现，可以是任意的内容，由此会有风险。另外，准备好处理调用你自己系统中的其它合约，可能在第一调用结果未返回之前就返回了调用的合约。某种程度上意味着，被调用的合约可以改变调用合约的状态变量(state variable)来标记当前的状态。如，写一个函数，只有当状态变量(state variables)的值有对应的改变时，才调用外部函数，这样你的合约就不会有可重入性漏洞。

### Storage(引用传递)和Memory(值传递)

Solidity中有两种类型：值类型和引用类型
Solidity是静态类型的语言，有值类型和引用类型的区别。

如果一个变量是值类型，那么当把它的值传给另一个变量时，是复制值，对新变量的操作不会影响原来的变量；如果该变量是引用类型，那么当它传值给另一个变量时，则是把该值的地址复制给新的变量。这样操作新变量也会导致旧变量的改变。

值类型：
布尔类型（bool）、整型（int）、地址类型（address）、定长字节数组（bytes）、枚举类型（enums）、函数类型（function）；

如果一个变量是值类型，给它赋值时永远是值传递！

引用类型：
字符串（string）、数组（array）、结构体（structs）、字典（mapping）、不定长字节数组（bytes）

如果一个变量是引用类型，给它赋值时可以是值，也可以是引用，这决定于该变量是Storage类型还是Memory类型。

关键字：Storage 和 Memory
Storage 是把变量永久储存在区块链中，Memory 则是把变量临时放在内存中，当外部函数对某合约调用完成时，内存型变量即被移除。 你可以把它想象成存储在你电脑的硬盘或是RAM中数据的关系。

大多数时候你都用不到这些关键字，默认情况下 Solidity 会自动处理它们。

状态变量（在函数之外声明的变量）默认为“存储”形式，并永久写入区块链；而在函数内部声明的变量是“内存”型的，它们函数调用结束后消失。

通过指定引用类型变量的关键字，可以人为设置变量为storage或memory。

函数的引用类型参数是storage时，是引用传递；函数的引用类型参数是Memory时，是值传递；函数值类型参数永远是值传递。

```

contract SandwichFactory {
  struct Sandwich {
    string name;
    string status;
  }
  Sandwich[] sandwiches;
  function eatSandwich(uint _index) public {
    // Sandwich mySandwich = sandwiches[_index];
    /*
       看上去很直接，不过 Solidity 将会给出警告,告诉你应该明确在这里定义 `storage` 或者 `memory`。
       所以你应该明确定义 `storage`:
    */
    Sandwich storage mySandwich = sandwiches[_index];
    // 这样 `mySandwich` 是指向 `sandwiches[_index]`的指针在存储里，另外...
    mySandwich.status = "Eaten!";
    // 这将永久把 `sandwiches[_index]` 变为区块链上的存储，如果你只想要一个副本，可以使用`memory`:
    Sandwich memory anotherSandwich = sandwiches[_index + 1];
    // 这样 `anotherSandwich` 就仅仅是一个内存里的副本了
    // 另外
    anotherSandwich.status = "Eaten!";
    // 将仅仅修改临时变量，对 `sandwiches[_index + 1]` 没有任何影响
    // 不过你可以这样做:
    sandwiches[_index + 1] = anotherSandwich;
    // 如果你想把副本的改动保存回区块链存储
  }
```

### 接口和抽象合约

接口的存在就是为了合约之间的通信。

有两种实现方式：抽象合约 和 接口

1. 抽象合约

抽象函数是没有函数体的的函数。如下：

```
pragma solidity ^0.4.0;

contract Feline {
    function utterance() returns (bytes32);
}
```

这样的合约不能通过编译，即使合约内也包含一些正常的函数。但它们可以做为基合约被继承。

```
pragma solidity ^0.4.0;

contract Feline {
    function utterance()
        returns (bytes32);
    function getContractName() returns (string){
        return "Feline";
    }
}

contract Cat is Feline {
    function utterance() returns (bytes32) {
        return "miaow";
    }
}
```


如果一个合约从一个抽象合约里继承，但却没实现所有函数，那么它也是一个抽象合约。

如何通过抽象合约实现接口功能？

如果contract B要使用contract A的方法或数据，本质上：

先定义一个抽象合约，让contract A继承于这个抽象合约；
把contract A中已经实现了的方法放入抽象合约中，solidity会自动把这个抽象合约视作接口；
contract B通过contract A的地址来创建连接到contract A的接口实例；
调用contract A中的方法或读取数据；

2. 接口

接口与抽象合约类似，与之不同的是，接口内没有任何函数是已实现的，同时还有如下限制：

不能继承其它合约，或接口。
不能定义构造器
不能定义变量
不能定义结构体
不能定义枚举类
接口基本上限制为合约ABI定义可以表示的内容，ABI和接口定义之间的转换应该是可能的，不会有任何信息丢失。

注意：
1、在两个.sol文件中都声明接口，或者两个合约写到一个.sol文件里，那就只要声明一次；
2、在一个合约中实现METHOD_A，该合同必须继承自接口interfaceContract；
3、在另一个合约中创建一个interfaceContract实例，该实例接受实现接口的合约的地址；
4、通过这个实例调用目标合约的方法，获取目标合约的数据；

实例：

被调用合约 InterfaceImpContract

```
pragma solidity ^0.4.16;

interface interfaceContract {
    function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData);
}

contract InterfaceImplContract is interfaceContract {
    event Receive(address from, uint256 value, address token, bytes extraData);
    function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) {
        Receive(_from,_value,_token,_extraData);
    }
}
```


调用合约 RemoteContract

```
pragma solidity ^0.4.16;

interface interfaceContract {
    function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData);
}

contract RemoteContract {
    function func(address _addr, uint _value) {
        //注意这里的_addr参数，需要填写tokenRecipient合约的地址。这里加载已经存在的智能合约。如何合约不存在会报错回滚。
        interfaceContract _interfaceContract = interfaceContract(_addr);
        _interfaceContract.receiveApproval(msg.sender, _value, address(this), "这是一些信息");
    }
}
```

addr，在下面的代码中，用来加载合约interfaceContract _interfaceContract = interfaceContract(_addr); 所以addr必须传递合约地址。并且这个合约地址是interfaceContract的实现类的合约地址。也就是第一步创建的InterfaceImplContract 合约的地址。

如果传递的_addr参数错误，调用失败。它将回滚所有已执行的功能。也就是这个方法会回滚。
这里部署时，只需部署RemoteContract 即可。不用管接口。接口只是为了声明。

### 事件、日志与交互（含实例）

事件是以太坊EVM提供的一种日志基础设施。事件可以用来做操作记录，存储为日志。也可以用来实现一些交互功能，比如通知UI，返回函数调用结果等。

**总的来说：事件就是当区块链某个函数被调用或执行的时候，被触发从而被前端获取或者记录到日志中的对象。**

#### 1. **事件的实现**

事件的实现是在合约对象中，分两步：
1、定义事件类型
2、实例化事件对象

代码：

```javascript

pragma solidity ^0.4.19;
contract ZombieFactory {
    // 定义事件类型
    event NewZombie(uint zombieId, string name, uint dna);
    uint dnaDigits = 16;
    uint dnaModulus = 10 ** dnaDigits;
    struct Zombie {
        string name;
        uint dna;
    }
    Zombie[] public zombies;
    function _createZombie(string _name, uint _dna) private {
        uint id = zombies.push(Zombie(_name, _dna)) - 1;
        // 实例化事件对象
        NewZombie(id, _name, _dna);
    }

```

#### 2. **事件与交互**

我们在前端使用web3.js来与区块链进行交互。当智能合约中的函数被调用而更改了区块链中的数据后，前端如何实时进行相应的行为？

1、调用合约，生成一个可以访问公共函数和事件的合约对象；
2、监听事件，调用事件方法，异步获取事件返回的值，error或者result；
3、判断并执行相应的前端函数；
4、注意：在操作执行完成后，我们要记得调用event.stopWatching();来终止监听。

代码：

```javascript
// 下面是调用合约的方式:
var abi = /* abi是由编译器生成的 */
var ZombieFactoryContract = web3.eth.contract(abi)
var contractAddress = /* 发布之后在以太坊上生成的合约地址 */
var ZombieFactory = ZombieFactoryContract.at(contractAddress)
// `ZombieFactory` 能访问公共的函数以及事件

// 监听 `NewZombie` 事件, 并且更新UI
var event = ZombieFactory.NewZombie(function(error, result) {
  if (error) return
  generateZombie(result.zombieId, result.name, result.dna)
})

// 获取 Zombie 的 dna, 更新图像
function generateZombie(id, name, dna) {
  // 新建一个Zombie的图像
  ...

```

#### **3. 事件与日志**

如上所说，事件是以太坊EVM提供的一种日志基础设施，日志是区块链中的一种特殊数据结构。

当定义的事件触发时，我们可以将事件存储到EVM的交易日志中，日志与合约关联，与合约的存储合并存入区块链中。只要某个区块可以访问，其相关的日志就可以访问。但在合约中，我们不能直接访问日志和事件数据（即便是创建日志的合约）。

web3.js监听事件，实际上是对EVM的交易日志的监听。

所以，当我们需要对事件日志进行条件性的过滤，即只在满足某些条件的情况下才执行前端的函数，要如何进行？

**检索日志：indexed属性的使用**

可以在事件参数上增加indexed属性，最多可以对三个参数增加这样的属性。加上这个属性，可以允许你在web3.js中通过对加了这个属性的参数进行值过滤，方式如下：

```javascript
var event = myContract.transfer({value: "100"});
```

上面实现的是对value值为100的日志，过滤后的返回。

如果你想同时匹配多个值，还可以传入一个要匹配的数组。

```javascript
var event = myContract.transfer({value: ["99","100","101"]});
```

增加了indexed的参数值会存到日志结构的Topic部分，便于快速查找。
未加indexed的参数值会存在data部分，成为原始日志。

需要注意的是，如果增加indexed属性的是数组类型（包括string和bytes），那么只会在Topic存储对应的数据的web3.sha3哈希值，将不会再存原始数据。因为Topic是用于快速查找的，不能存任意长度的数据，所以通过Topic实际存的是数组这种非固定长度数据哈希结果。要查找时，是将要查找内容哈希后与Topic内容进行匹配，但我们不能反推哈希结果，从而得不到原始值。

所以如果你要实现过滤还要获得原始值，那就不要把indexed加到string和bytes类型参数前面。

#### **使用web3.js读取事件的完整例子**

下面是一个使用以太坊提供的工具包web3.js访问事件的完整例子：

```javascript

let Web3 = require('web3');
let web3;

if (typeof web3 !== 'undefined') {
    web3 = new Web3(web3.currentProvider);
} else {
    // set the provider you want from Web3.providers
    web3 = new Web3(new Web3.providers.HttpProvider("http://localhost:8545"));
}

let from = web3.eth.accounts[0];

//编译合约
let source = "pragma solidity ^0.4.0;contract Transfer{ event transfer(address indexed _from, address indexed _to, uint indexed value); function deposit() payable { address current = this; uint value = msg.value; transfer(msg.sender, current, value); } function getBanlance() constant returns(uint) { return this.balance; } /* fallback function */ function(){}}";

let transferCompiled = web3.eth.compile.solidity(source);
console.log(transferCompiled);
console.log("ABI definition:");
console.log(transferCompiled["info"]["abiDefinition"]);

//得到合约对象
let abiDefinition = transferCompiled["info"]["abiDefinition"];
let transferContract = web3.eth.contract(abiDefinition);
 
//2. 部署合约
//2.1 获取合约的代码，部署时传递的就是合约编译后的二进制码
let deployCode = transferCompiled["code"];
//2.2 部署者的地址，当前取默认账户的第一个地址。
let deployeAddr = web3.eth.accounts[0];

//2.3 异步方式，部署合约
//警告，你不应该每次都部署合约，这里只是为了提供一个可以完全跑通的例子！！！
transferContract.new({
    data: deployCode,
    from: deployeAddr,
    gas: 1000000
}, function(err, myContract) {
    if (!err) {
        // 注意：这个回调会触发两次
        //一次是合约的交易哈希属性完成
        //另一次是在某个地址上完成部署
        // 通过判断是否有地址，来确认是第一次调用，还是第二次调用。
        if (!myContract.address) {
            console.log("contract deploy transaction hash: " + myContract.transactionHash) //部署合约的交易哈希值
            // 合约发布成功后，才能调用后续的方法
        } else {
            console.log("contract deploy address: " + myContract.address) // 合约的部署地址
            console.log("Current balance: " + myContract.getBanlance());
            var event = myContract.transfer();
            // 监听
            event.watch(function(error, result){
              console.log("Event are as following:-------");
              for(let key in result){
                console.log(key + " : " + result[key]);
              }
              console.log("Event ending-------");
            });
            //使用transaction方式调用，写入到区块链上
            myContract.deposit.sendTransaction({
                from: deployeAddr,
                value: 100,
                gas: 1000000
            }, function(err, result){
              console.log("Deposit status: " + err + " result: " + result);
              console.log("After deposit balance: " + myContract.getBanlance());
              //终止监听，注意这里要在回调里面，因为是异步执行的。
              event.stopWatching();
            });
        }
    }

```