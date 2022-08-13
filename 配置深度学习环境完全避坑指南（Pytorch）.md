# 配置深度学习环境完全避坑指南（Pytorch）

[TOC]



## 0. 前言

目前为止，LWL因为自身~~笨蛋一个~~原因以及工作原因，已经反复配置了Pytorch版本深度学习环境很多次很多次很多次……了。在自己的电脑上配置了三四次，在公司的主机上配置了两次，甚至在树莓派Linux小车上也配置了一次（

可以说，把各种坑都全踩了个遍，把麻烦的配置方法和摸索中探索到的方法都探究了个遍。~~尊一声配置环境踩坑大师应该不过分~~

所以，在这里，我将会手把手地，图文并茂地，耐下性子地给大家讲一遍最便捷，最不踩坑的配置深度学习Pytorch环境的完全教程。

**请务必！务必！务必！严格按照教程一步一步做！不要随便跳过或者更改什么步骤！不要嫌麻烦！已经很详细很便捷了！再删掉什么步骤等着一群ERROR吧！**你也可以自己先完全按照教程做一遍，再加点别的什么想法。但是很容易会一片偏红的ERROR……谨慎对待！



## 1. 你需要准备什么

现在我们假设，你的电脑关于深度学习的东西，什么anaconda啊，cuda啊，甚至python都没安装。

* 如果你先前安装过自己的python，请**删掉自己电脑的本来的python**，因为之后会干扰到安装。删除的教程见下面Tips。

  在之后安装anaconda的教程中，anaconda会自己带一个最适配的python并自己的python IDE会识别出来，所以我们先不要用自己python。

* 如果你是曾经配置环境失败过才看到本教程，现在请必须去检查一下自己**残余文件已清理干净**：找到自己曾经的安装目录或者全都搜索一遍，关于anaconda的文件夹全部直接移进回收站然后**清空回收站**。

> **Tips：**
>
> **在哪里查看并删掉本来的Python？**
>
> 1. 搜索“控制面板”。
>
> <img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715140608078.png" alt="image-20220715140608078" style="zoom: 67%;" />
>
> 2. 点击“程序”->“卸载程序”，找到Python x.x.x (64-bit)这个东西，然后右键“卸载”。所有的都要卸载。
>
> <img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715140821349.png" alt="image-20220715140821349" style="zoom: 67%;" />



## 2. 开始配环境！

### 2.0. 前置知识

在深度学习环境配置中，我们要清楚：**并不是版本越新越好！要用低一点，稳定一点的旧版本**。有些坑往往是版本太新造成的~~（比如万恶的python3.9）~~，而且现在很多网上下载的项目，基本python版本都是3.6，3.7。所以直接一劳永逸，我们就安装旧一点的。

深度学习必不可少的基本包是这样的（2022.06.24安装时我的选择）：

> anaconda3：5.3.0
>
> python：3.7.7(安装anaconda时会配好)
>
> numpy：1.21.6(安装anaconda时会配好)
>
> Jupyter Notebook：anaconda自带
>
> torch：1.7.1+cu110
>
> torchvision：0.8.2+cu110(安装torch时会配好)

接下来，我们按照上面的版本来安装。



### 2.1. 安装Anaconda3

>**Tips：**
>
>* Anaconda是一个集成了conda和python的大环境，俗称”大蟒蛇“。
>
>* Anaconda方便管理不同版本的软件包和方便切换不同环境（Pytorch、Tensorflow等）
>
>* **自带**有一些重要的基础包，类似**numpy**、**Jupyter**[^ 关于深度学习工具 ] 
>
>  等。
>
>* 关于深度学习需要的包都可以直接下载到anaconda的site_packages并方便调用。
>
>* **Anaconda必不可少。**

#### 步骤

1. 点击下面标签可以直接下载。或者网上搜索”清华镜像源”，进去后点击进入Anaconda->archive，然后找到Anaconda3-5.3.0-Windows-x86_64.exe下载

   | [Anaconda3-5.3.0-Windows-x86_64.exe](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.0-Windows-x86_64.exe) | 631.4 MiB | 2018-09-28 06:46 |
   | ------------------------------------------------------------ | --------- | ---------------- |

   **注意：千万不要从anaconda官网下载！要不然会配置很多额外东西而且还容易出错。踩坑好多次了，从官网下载太麻烦了。**

2. 点击下载好的Anaconda3 Setup进入安装。前面三个选项无脑点击下一步。

3. 路径选择，你可以自己选择默认的或者装到别的盘里。**一定要记住你的安装路径**，因为之后在安装包的时候很多次会到anaconda里的site_packages进行操作。

   <img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715145044433.png" alt="image-20220715145044433" style="zoom:67%;" />

   

3. **Advanced Options一定要两项都要勾选！**

   第一个选项即使标红也不用理他直接勾选。如果没有勾选第一项之后要手动环境变量去配置，因为必须把Anaconda加到PATH里面才能直接访问一些文件。

   第二个选项就可以自动给你安装python3.7.7或者python3.7系列的（如果不是3.7.7也没关系，但是一定是3.7系列的。同3.7系列下的一些配置基本相同）

   <img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715145306537.png" alt="image-20220715145306537" style="zoom:67%;" />

4. 然后我们点击Install，等待安装完毕。

#### Tips：关于深度学习工具

> [^ 关于深度学习工具 ]: 此章节结束后介绍
>
> * **Jupyter**：适合小模型的训练。（如果你的显卡好也适合中等大小模型）
>
>   * 优点：
>
>     方便快捷。
>
>   * 缺点：
>
>     依靠自己电脑的GPU性能。
>
> * **Google Colab**：适合中等大小模型训练。
>
>   * 优点：
>
>     不依靠自己电脑的GPU性能，在Google服务器上面运行，性能大概是RTX 3060。
>
>   * 缺点：
>
>     需要~~科学上网~~翻墙到国外。
>
>     需要一个GoogleDrive才能加载自己的数据集，这意味着你需要一个Google账号，要不然只能每次本地下载，很慢。
>
>     如果长期不动动鼠标，他会回收你的使用权，意味着你得守着他。
>
>     你需要付款才能多次使用，要不然一天只能使用一次。
>
> * **服务器**：适合大型模型训练。（比如华为云，腾讯云等）
>
>   * 优点：
>
>     快！训练是真的快！高！性能是真的高！
>
>   * 缺点：
>
>     贵！是真~~tm~~贵！其实作为个人使用也还好，华为云我看了是30几块钱一小时。



### 2.2. 安装Pytorch深度学习环境

#### 2.2.1. 安装torch

（先别急着安装，请**务必！！**仔细阅读下文，**尤其是Q&A第三条开始的**。看完之后会有链接让你直接操作。

每个python版本都对应不同的torch版本，不同的torch版本对应不同的torchvision版本。并且torch有CPU和GPU之分。

**如果不在意对应的版本，那么无法使用torch。**

网上解释的很乱，那就每个可能遇到的问题都解释一下。

> 1. **Q：我怎么知道哪个torch对应哪个python？**
>
>    A：<img src="https://www.yht7.com/upload/image/2022/05/11/202205110936571.jpg" alt="img" style="zoom:67%;" />
>
> 2. **Q：我怎么知道哪个torchvision对应哪个torch？**
>
>    **A：**在pytorch官网安装torch的时候，会给你对应好的torchvision版本的命令。也就是说，你安装好torch，torchvision也会顺带安装好。
>
> 3. **Q：torch有CPU和GPU之分是怎么回事？我该装哪个，都应该装吗？**
>
>    A：在深度学习训练的时候，torch用的到底是CPU还是GPU在训练时间上有天壤之别。GPU要比CPU运算快得多。
>
>    **如果安装了CPU版本的torch则是无法安装GPU版本的。**如果有GPU，则最好使用GPU版本的torch，否则只能下载CPU版本的。如果之前自己瞎装了一个，请把它先在命令提示符（cmd）里卸载，代码：`pip uninstall torch`。（如何下载哪个版本的之后详述）
>
> 4. **Q：那如何判断是否有GPU呢？有几个GPU？**
>
>    A：右键<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715163126538.png" alt="image-20220715163126538" style="zoom:50%;" />图标，选择“任务管理器”，然后选择“性能”。下滑到最后看看有没有GPU，几个GPU。
>
>    我电脑是这样的，说明有两个GPU。<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715163420602.png" alt="image-20220715163420602" style="zoom:67%;" />
>
>    而公司主机是这样的，说明有一个GPU。<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715163636484.png" alt="image-20220715163636484" style="zoom:67%;" />
>
> 5. **Q：我应该怎么选择安装哪个版本的torch？**
>
>    A：等会儿点开给出的链接时，里面基本是这样的
>
>    <img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715171232534.png" alt="image-20220715171232534" style="zoom:67%;" />
>
>    其中，代码是放在命令提示符（cmd）里执行的。
>
>    `CUDA xx.x`是Windows + 有GPU用户可以执行的命令行，`CPU Only`是Windows/Linux + 只有CPU用户可以执行的命令行。
>
>    如果有GPU的用户，请复制`CUDA xx.x`下方代码并在cmd里执行。否则只有CPU的用户请复制`CPU Only`下方代码并在cmd里执行。

本教程对应的torch链接：**https://pytorch.org/get-started/previous-versions/#linux-and-windows-14**。此链接已经对应到python3.7系列对应版本，且较稳定。

~~如果懒得打开，~~可以直接复制下面代码在cmd里执行。

CUDA 11.0（有GPU的用户）

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

CPU Only （仅有CPU的用户）

```
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

然后等待安装完毕。

#### 2.2.2. 测试是否安装成功

打开命令提示符（cmd），输入以下代码。

```
python
import numpy
import torch
import torchvision
numpy.__version__
torch.__version__
torchvision.__version__
exit()
```

如果输出为下，则安装成功。

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715174015319.png" alt="image-20220715174015319" style="zoom:67%;" />



然后我们可以打开JupyterNotebook这个深度学习工具，试试看是否工具能运行。

在cmd里输入`Jupyter Notebook`，或打开win图标搜索jupyter notebook并点击这个<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715174231326.png" alt="image-20220715174231326" style="zoom: 50%;" />

**！！切记！打开JupyterNotebook后，千万不要关闭后台的这个黑框窗口，要不然会导致在本地运行的jupyter无法运行！！**

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715174052085.png" alt="image-20220715174052085" style="zoom:67%;" />

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715174318110.png" alt="image-20220715174318110" style="zoom:67%;" />

接下来会自动跳转到浏览器里面，出现此界面，说明jupyter安装也成功了。

![image-20220715174557078](C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220715174557078.png)

## 3. 后记

至此，跟着教程一步步做应该没有问题。如果有问题请咨询我QQ：3493617871，可能遗忘掉一些当时遇到的bug。

之后可能会再出一篇教程，记录在安装包和运行程序出现的坑。