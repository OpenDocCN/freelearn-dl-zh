# 前言

如果你对机器学习、人工智能或深度学习等术语有所了解，你可能知道什么是神经网络。你是否曾想过它们是如何高效地帮助解决复杂的计算问题，或者如何训练高效的神经网络？本书将教你这两方面的内容，以及更多。

你将首先快速了解流行的 TensorFlow 库，并了解它是如何用于训练不同的神经网络的。你将深入理解神经网络的基本原理和数学基础，并了解为什么 TensorFlow 是一个流行的选择。接下来，你将实现一个简单的前馈神经网络。然后，你将掌握使用 TensorFlow 进行神经网络优化的技巧和算法。之后，你将学习如何实现一些更复杂的神经网络类型，例如**卷积神经网络**（**CNNs**）、**递归神经网络**（**RNNs**）和**深度置信网络**（**DBNs**）。在本书的过程中，你将使用真实世界的数据集来动手理解神经网络编程。你还将学习如何训练生成模型，并了解自动编码器的应用。

到本书结尾时，你将较为清楚地了解如何利用 TensorFlow 的强大功能，轻松训练不同复杂度的神经网络。

# 本书所需内容

本书将引导你安装所有必要的工具，以便你能跟随示例进行操作：

+   Python 3.4 或以上版本

+   TensorFlow 1.14 或以上版本

# 本书适合的人群

本书适合具有统计学背景的开发人员，他们希望使用神经网络。虽然我们将使用 TensorFlow 作为神经网络的底层库，但本书也可以作为一个通用资源，帮助填补数学与深度学习实现之间的空白。如果你对 TensorFlow 和 Python 有一定了解，并希望学习更深入的内容，而不仅仅是 API 语法层面的知识，那么这本书适合你。

# 约定

在本书中，你会看到几种文本样式，用来区分不同种类的信息。以下是这些样式的示例及其含义。代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号等以以下方式显示：“下一行代码读取链接并将其分配给`BeautifulSoup`函数。”代码块的显示方式如下：

```py
#import packages into the project 
from bs4 import BeautifulSoup 
from urllib.request import urlopen 
import pandas as pd
```

当我们希望引起你注意某个代码块的特定部分时，相关的行或项会以粗体显示：

```py
 [default] exten 
=> s,1,Dial(Zap/1|30) exten 
=> s,2,Voicemail(u100) exten 
=> s,102,Voicemail(b100) exten 
=> i,1,Voicemail(s0) 
```

任何命令行输入或输出都以如下方式书写：

```py
C:\Python34\Scripts> pip install -upgrade pip
C:\Python34\Scripts> pip install pandas
```

**新术语**和**重要词汇**以粗体显示。屏幕上出现的单词，例如在菜单或对话框中，通常会以如下方式显示：“为了下载新模块，我们将进入文件 | 设置 | 项目名称 | 项目解释器。”

警告或重要提示将以如下方式显示。

提示和技巧以如下方式显示。

# 读者反馈

我们始终欢迎读者的反馈。请告诉我们你对本书的看法——你喜欢或不喜欢的内容。读者反馈对我们非常重要，因为它帮助我们开发出你真正能够最大化利用的书籍。如果你有任何建议或反馈，请通过邮件发送至`feedback@packtpub.com`，并在邮件主题中注明书名。如果你在某个领域有专业知识，并且有兴趣写作或参与书籍的编写，欢迎查看我们的作者指南：[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

既然你已经成为了《Packt》书籍的骄傲拥有者，我们为你提供了一些资源，帮助你充分利用这次购买。

# 下载示例代码

你可以从你的账户中下载本书的示例代码文件，网址为[`www.packtpub.com`](http://www.packtpub.com)。如果你是在其他地方购买的本书，可以访问[`www.packtpub.com/support`](http://www.packtpub.com/support)并注册以便直接将文件通过电子邮件发送给你。你可以通过以下步骤下载代码文件：

1.  使用你的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的“支持”标签上。

1.  点击“代码下载 & 勘误”。

1.  在搜索框中输入书名。

1.  选择你想下载代码文件的书籍。

1.  从下拉菜单中选择你购买本书的地点。

1.  点击“代码下载”。

下载完文件后，请确保使用以下最新版本的工具来解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Neural-Network-Programming-with-TensorFlow`](https://github.com/PacktPublishing/Neural-Network-Programming-with-TensorFlow)。我们还提供了来自我们丰富书籍和视频目录中的其他代码包，地址为[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。赶紧去看看吧！

# 下载本书的彩色图片

我们还为你提供了一份包含本书中截图/图表的彩色图片的 PDF 文件。这些彩色图片将帮助你更好地理解输出结果的变化。你可以从[`www.packtpub.com/sites/default/files/downloads/NeuralNetworkProgrammingwithTensorFlow_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/NeuralNetworkProgrammingwithTensorFlow_ColorImages.pdf)下载此文件。

# 勘误

虽然我们已尽力确保内容的准确性，但难免会出现错误。如果您在我们的书籍中发现任何错误——可能是文本或代码中的错误——我们将非常感激您能报告此问题。这样，您可以帮助其他读者避免困扰，并帮助我们改进该书的后续版本。如果您发现任何勘误，请访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入勘误的详细信息。一旦您的勘误被验证，您的提交将被接受，勘误将上传至我们的网站或添加到该书的勘误列表中。要查看已提交的勘误，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索框中输入书名。所需信息将显示在勘误部分。

# 盗版

网络上侵犯版权的行为在所有媒体中都是一个持续存在的问题。在 Packt，我们非常重视版权和许可证的保护。如果您在互联网上遇到任何形式的我们作品的非法复制，请立即向我们提供其所在的地址或网站名称，以便我们采取相应的措施。请通过`copyright@packtpub.com`与我们联系，并提供涉嫌盗版材料的链接。感谢您在保护我们的作者和我们提供有价值内容的能力方面给予的帮助。

# 问题

如果您对本书的任何方面有问题，可以通过`questions@packtpub.com`与我们联系，我们将尽力解决问题。
