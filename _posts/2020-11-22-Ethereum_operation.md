---
layout: post
title: "Ethereum Tutorial"
date: 2020-11-22 
description: "Ethereum"
tag: ethereum
---

## Ethereum Tutorial

In this tutorial we will:

- Setup a blockchain with multiple nodes
- Setup mining nodes
- Connect the multiple nodes and setup the blockchain network
- Test the blockchain network by mining blocks and verifying that blocks are propagated to all the nodes
- Verify that the local copy of blockchain on all the nodes is updated

### Setup and Prerequisite Software

[Go Etheruem](https://geth.ethereum.org/) (Geth) is a command line client interface tool that allows you to interact with your private Ethereum blockchain.

If you need to install [Homebrew](https://brew.sh/) on your Mac, place the following line in your terminal command prompt:

```
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Once Homebrew is installed, [install](https://github.com/ethereum/go-ethereum/wiki/Installation-Instructions-for-Mac) Geth:

```
$ brew tap ethereum/ethereum
$ brew install ethereum
```

### Configure the Genesis Block

Create a new project directory. Within it create the `**genesis.json**` file using any editor of your choice. I use [Visual Studio code on Mac](https://code.visualstudio.com/docs/setup/mac), which is a free editor:

```
$ mkdir project3
$ cd project3
```

Copy the following JSON into your `**genesis.json**` file, and save the file in your project directory.

```json
\\genesis.json file
{"config": {"chainId": 4321,"homesteadBlock": 0,"eip155Block": 0,"eip158Block": 0},"alloc": {},"difficulty" : "0x20000","gasLimit"   : "0x8880000"}
```

### Set up the first node (Node 1)

Once the `**genesis.json**` file is saved, you are ready to create your first node. To create your first node, open a new terminal window and navigate to your project folder, and type in the following command:

```
$ geth --datadir blkchain1 init genesis.json
```

This command initializes a new blockchain node using the configuration specified in the `**genesis.json**` file. By using `— -datadir` we specify the name of the directory where the local copy of the blockchain will be stored on the node.

This command will create a directory called `**blkchain1**` in your project directory. This directory contains the `**geth**` and `**keystore**` directories. The blockchain data will be stored in the local database in the `**geth**` directory.

You can verify this by navigating into the `**blkchain1**` directory in your command terminal and typing in `ls -l`. Also navigate to the `**geth**` sub-directory and type in `ls -l`.

Now that your Node 1 is initialized, let’s start Node 1 using Geth. Within your terminal window, navigate to your project folder (where you saved your `**genesis.json**` file) and type in the following:

```
$ geth --datadir blkchain1 --nodiscover --networkid 1234 console
```

This command will start your first node and bring up the Geth console, where you can type in commands to interact with the Node1 blockchain.

Let’s break down this “geth console” command and understand it:

`— -datadir blkchain1`:
specifies the data directory of the blockchain. If you do not specify this, it will default to the main Ethereum blockchain.

`--nodiscover`:
disables the peer discovery mechanism and enables manual peer addition.

`--networkid 1234`**:**

identity of your Ethereum network, other peer nodes will also have the same network identifier. **The network id can be any random integer value.**

Now that you are in the Geth Console, to get more information about the node we will use the **admin** command. Type in the following command in the Geth console:

```
> admin.nodeInfo
```

Notice the following:

- by default the `listener` port is 30303
- the `enode id` is your node id
- the `discovery` port is ‘0’, since we set it to `--nodiscover`.

To get a list of all admin commands, type in “admin.” and press <tab>

Now, let’s setup an account where the [mined ethers](https://www.ethereum.org/ether) will be collected. Type in the following in the Geth console:

```
> personal.newAccount()
```

You can put in any passphrase — this is your password. The command returns the account id.

To get a list of all the accounts:

```
> personal.listAccounts
```

### Set up the second node (Node 2)

Now let’s setup a second node in the blockchain network. The process will be similar to setting up Node1.

Open a new terminal window and navigate to the project folder that contains the `**genesis.json**` file.

Initialize the new node with the following command:

```
$ geth --datadir blkchain2 init genesis.json
```

**Note**: Since we want this node to be part of the same blockchain, we use the same genesis block.

This will create a new node whose data will be stored in a new directory called `**blkchain2** `(this will contain the local copy of the blockchain database for the node 2).

To get the node 2 up and running with a Geth console, type in the following:

```
$ geth --datadir blkchain2 --nodiscover --networkid 1234 --port 30304 console
```

This will start up Node 2 and bring up the Geth console connected to Node 2.

- we specify a port number 30304
  The default port 30303 is already being used by the first node, so if we do not specify a separate port it will throw an error.
- we give the same networkid as we did for node 1. This is important, since we want both the nodes to be part of the network
- we specify that the `— -datadir` for node2 will be in `**blkchain2**`

Run the `admin.nodeInfo` command in Geth Console for Node2:

```
> admin.nodeInfo
```

Similar to Node 1, we can setup an account on Node 2 by typing in `personal.newAccount()` in the Geth console for Node2.

You can use any passphrase for the account.

```
> personal.newAccount()
```

Node 2 is now ready!

Keep open both the terminal windows, one running the Geth Console (Node1) and the second running the Geth Console (Node2), side by side.

In the next step we will connect Node1 and Node2 and create the blockchain network!

### Connect the Nodes

Congratulations! You have 2 blockchain nodes running.

The next step is to connect them to each other and create a network. We will do this by adding one of the nodes to the other as a “peer”.

Run the following command on the Geth Consoles of both Node1 and Node2:

```
> admin.peers
```

Notice that both return an empty array, since the nodes are not connected to any other nodes.

Let’s add Node1 as a peer to Node2.

1. Run the `admin.nodeInfo` command on Node1

2.  Copy the `enode id` for Node 1:

3. In the Geth Console for Node2, add the Node1 as a peer to Node2 using the command `admin.addPeer(“//enode id”):`

   ```
   >admin.addPeer("enode://549468e6d00e135128af33e03a6d27b0ee5fda7fbd0154b2e83fe68afdfda869eb6ace6ccaefe84ed7a5b804529dcef49f0b5d64be97da87b1e28ddecfca227a@[::]:30303?discport=0")
   ```

   Now when you run the `admin.peers` command in either of the Geth Console you will see the other Node listed as a peer!

   Hurray! You now have a connect blockchain network!

   But how do we know for sure?

   Let’s put it to the test, shall we? Let’s update something in Node1 and see if that change is propagated through the blockchain network to Node2.

### Verify that you have a connected blockchain and a distributed database

Ok, thus far you should have:

- Two terminal windows open
  One running Geth Console for Node1, and the second running Get Console for Node2
- Node 1 saves its local copy of the blockchain in the `**blkchain1**` folder
- Node 2 saves its local copy of the blockchain in the `**blkchain2**` folder
- Also, in the previous step you connected the two nodes using `admin.addPeer()`

Thus, if the new blocks are mined and added in Node 1, this new block should propagate over the blockchain network, and the local blockchain copy of node2 should update automatically.

Let us verify if this happens.

In the Geth Console(Node1) let’s begin mining:

```
> miner.start(1)
```

To check how many blocks have been mined at Node1 at any point, run the following in the Geth Console (Node1):

```
> eth.blockNumber
```

This will return the number of the current block mined (or the blockchain height).

After a few blocks have been mined, go to the Geth Console (Node 2) and run the `eth.blockNumber` command.

You will notice that the block height in Node 2 matches with the block height in Node 1! This verifies that the Blocks that were mined in Node 1 were propagated over the Blockchain to Node 2.

To kick it up a notch, you can verify the data in the actual blocks on Node 1 and Node 2.

Wait until about 10 blocks have been mined and then run the following command in Geth Console(Node1) and Geth Console(Node2):

```
> eth.getBlock(3)
```

This command displays the contents of Block 3 from the blockchain. Notice that the data displayed in Geth Console (Node1) and Geth Console(Node2) is identical, indicating that the Block3 that was mined by Node1 was propagated over the blockchain network and is also part of the local blockchain data in Node 2.

