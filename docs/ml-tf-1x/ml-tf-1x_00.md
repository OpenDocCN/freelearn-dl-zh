# 前言

机器学习已经彻底改变了现代世界。许多机器学习算法，特别是深度学习，已经被广泛应用于全球范围，从移动设备到基于云的服务。TensorFlow 是领先的开源软件库之一，帮助你构建、训练和部署各种应用的机器学习系统。本实用书籍旨在为你带来 TensorFlow 的精华，帮助你构建真实世界的机器学习系统。

在本书结束时，你将对 TensorFlow 有深入的了解，并能够将机器学习技术应用到你的应用程序中。

# 本书涵盖内容

第一章，*TensorFlow 入门*，展示了如何在 Ubuntu、macOS 和 Windows 上安装 TensorFlow 并开始使用。

第二章，*你的第一个分类器*，带你走进手写识别器的第一次旅程。

第三章，*TensorFlow 工具箱*，概述了 TensorFlow 提供的工具，帮助你更加高效、轻松地工作。

第四章，*猫与狗*，教你如何使用 TensorFlow 中的卷积神经网络构建一个图像分类器。

第五章，*序列到序列模型—Parlez-vous Français?*，讨论了如何使用序列到序列模型构建一个从英语到法语的翻译器。

第六章，*寻找意义*，探索通过情感分析、实体提取、关键词提取和词语关系提取来寻找文本中的意义的方法。

第七章，*用机器学习赚钱*，深入探讨了一个数据量庞大的领域：金融世界。你将学习如何处理时间序列数据来解决金融问题。

第八章，*医生马上就诊*，探讨了如何利用深度神经网络解决一个*企业级*问题——医疗诊断。

第九章，*巡航控制 - 自动化*，教你如何创建一个生产系统，从训练到服务模型。该系统还能接收用户反馈并每天自动进行训练。

第十章，*上线并扩展*，带你进入亚马逊 Web 服务的世界，并展示如何在亚马逊服务器上利用多个 GPU 系统。

第十一章，*更进一步 - 21 个问题*，介绍了 21 个现实生活中的问题，阅读本书后，你可以利用深度学习—TensorFlow 来解决这些问题。

附录*，高级安装*，讨论了 GPU，并重点介绍了逐步 CUDA 设置和基于 GPU 的 TensorFlow 安装。

# 本书所需的条件

对于软件，本书完全基于 TensorFlow。你可以使用 Linux、Windows 或 macOS。

对于硬件，你将需要一台运行 Ubuntu、macOS 或 Windows 的计算机或笔记本电脑。作为作者，我们建议如果你打算使用深度神经网络，特别是在处理大规模数据集时，最好拥有一块 NVIDIA 显卡。

# 本书适合谁阅读

本书非常适合那些有志于构建智能且实用的机器学习系统，能够应用于真实世界的用户。你应该对机器学习概念、Python 编程、集成开发环境（IDEs）和命令行操作感到熟悉。本书对于那些作为职业程序员、科学家和工程师的人非常有用，特别是当他们需要学习机器学习和 TensorFlow 来支持他们的工作时。

# 约定

在本书中，你会发现一些不同的文本样式，用来区分不同类型的信息。以下是这些样式的几个示例及其含义的解释。

书中提到的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号等，显示方式如下：“我们可以通过使用 `include` 指令来包含其他上下文。”

代码块的显示方式如下：

```py
batch_size = 128 
  num_steps = 10000 
  learning_rate = 0.3 
  data_showing_step = 500
```

当我们希望你注意某一部分代码时，相关的行或项会以粗体显示：

```py
Layer 1 CONV (32, 28, 28, 4) 
Layer 2 CONV (32, 14, 14, 4) 
Layer 3 CONV (32, 7, 7, 4)
```

任何命令行输入或输出的显示方式如下：

```py
sudo apt-get install python-pip python-dev
```

**新术语** 和 **重要词汇** 以粗体显示。

警告或重要注意事项如下所示。

提示和技巧如下所示。

# 读者反馈

我们总是欢迎读者的反馈。请告诉我们你对本书的看法——你喜欢或不喜欢的部分。读者反馈对我们非常重要，它帮助我们开发出能够让你真正受益的书籍。

如果你想向我们提供一般反馈，只需发送电子邮件至 feedback@packtpub.com，并在邮件主题中提到本书的标题。

如果你有某个领域的专业知识，并且有兴趣撰写或为书籍贡献内容，请查看我们的作者指南：[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，你已经成为 Packt 书籍的骄傲拥有者，我们提供了许多帮助你充分利用购买内容的资源。

# 下载示例代码

你可以通过你的账户从 [`www.packtpub.com`](http://www.packtpub.com) 下载本书的示例代码文件。如果你是从其他地方购买的本书，你可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support)，并注册以便将文件直接发送到你的邮箱。

你可以按照以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的 SUPPORT 标签上。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名。

1.  选择您希望下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书的地方。

1.  点击“代码下载”。

下载文件后，请确保使用最新版本的以下工具解压或提取文件：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书的代码包也托管在 GitHub 上，链接为[`github.com/PacktPublishing/Machine-Learning-with-TensorFlow-1.x`](https://github.com/PacktPublishing/Machine-Learning-with-TensorFlow-1.x)。我们还有其他来自我们丰富图书和视频目录的代码包，链接为[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快来看看吧！

# 下载本书的彩色图片

我们还为您提供了一个包含本书中使用的截图/图表彩色图片的 PDF 文件。这些彩色图片将帮助您更好地理解输出中的变化。您可以从[`www.packtpub.com/sites/default/files/downloads/MachineLearningwithTensorFlow1.x_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/MachineLearningwithTensorFlow1.x_ColorImages.pdf)下载该文件。

# 勘误

尽管我们已尽力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——例如文本或代码中的错误——我们将非常感激您向我们报告此问题。这样，您可以帮助其他读者避免困扰，并帮助我们改进书籍的后续版本。如果您发现任何勘误，请通过访问[`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)并选择您的书籍，点击“勘误提交表格”链接，输入勘误的详细信息来报告勘误。您的勘误经验证后，将被接受并上传到我们的网站，或添加到该书的勘误部分。

若要查看之前提交的勘误信息，请访问[`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索框中输入书名。所需信息将显示在勘误部分。

# 盗版

互联网盗版版权材料的问题在所有媒体中持续存在。在 Packt，我们非常重视保护我们的版权和许可证。如果您在互联网上发现任何形式的非法复制品，请立即提供相关网址或网站名称，以便我们采取措施。

如发现盗版内容，请通过 copyright@packtpub.com 与我们联系，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供宝贵内容方面的帮助。

# 问题

如果您对本书的任何部分有疑问，您可以通过 questions@packtpub.com 与我们联系，我们将尽力解决问题。
