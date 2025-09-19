# 前言

**自然语言处理（NLP**）是使用机器来操作自然语言。本书通过代码和相关的案例研究，教你如何使用Python构建NLP应用程序。本书将介绍构建NLP应用程序的基本词汇和推荐的工作流程，帮助你开始进行诸如情感分析、实体识别、词性标注、词干提取和词嵌入等流行的NLP任务。

# 本书面向对象

本书面向希望构建能够解释语言的系统的程序员，并且对Python编程有所了解。熟悉NLP词汇和基础知识以及机器学习将有所帮助，但不是必需的。

# 本书涵盖内容

[第一章](5625152b-6870-44b1-a39f-5a79bcc675d9.xhtml)，*开始文本分类之旅*，向读者介绍了自然语言处理（NLP）以及一个良好的NLP工作流程是什么样的。你还将学习如何使用scikit-learn为机器学习准备文本。

[第二章](d88069c9-ebc2-45c1-945f-e3b4450547c2.xhtml)，*整理你的文本*，讨论了一些最常见的文本预处理想法。你将了解spaCy，并学习如何使用它进行分词、句子提取和词形还原。

[第三章](676f965d-1260-4c5d-bbc6-95fabf139386.xhtml)，*利用语言学*，探讨了一个简单的用例，并检查我们如何解决它。然后，我们再次执行这个任务，但是在一个略有不同的文本语料库上。

[第四章](ce6effcc-a0af-4013-8d1e-ffbef39fcea8.xhtml)，*文本表示 – 从单词到数字*，向读者介绍了Gensim API。我们还将学习如何加载预训练的GloVe向量，并在任何机器学习模型中使用这些向量表示而不是TD-IDF。

[第五章](6f75a9b6-8050-461a-91f7-dd0293bdcf78.xhtml)，*现代分类方法*，探讨了关于机器学习的几个新想法。这里的目的是展示一些最常见的分类器。我们还将了解诸如情感分析、简单分类器以及如何针对你的数据集和集成方法进行优化的概念。

[第六章](a8967c79-902c-4de0-ba42-8989df111b18.xhtml)，*深度学习在NLP中的应用*，涵盖了深度学习是什么，它与我们所看到的不同之处，以及任何深度学习模型中的关键思想。我们还将探讨一些与PyTorch相关的话题，如何标记文本，以及什么是循环神经网络。

[第七章](90e6299f-f92d-412c-8438-2ecf2ee30e01.xhtml)，*构建自己的聊天机器人*，解释了为什么应该构建聊天机器人，并确定正确的用户意图。我们还将详细了解*意图*、*响应*、*模板*和*实体*。

[第八章](c86e5871-562b-4f13-b6d2-ce2992526135.xhtml)，*Web部署*，解释了如何训练模型并编写一些用于数据I/O的更简洁的实用工具。我们将构建一个预测函数，并通过Flask REST端点公开它。

# 为了充分利用本书

+   你需要安装Python 3.6或更高版本的conda

+   需要具备对 Python 编程语言的基本理解

+   NLP 或机器学习经验将有所帮助，但不是强制性的

# 下载示例代码文件

您可以从您的账户中下载本书的示例代码文件，网址为 [www.packt.com](http://www.packt.com)。如果您在其他地方购买了本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，以便将文件直接发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择“支持”选项卡。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名，并按照屏幕上的说明操作。

文件下载完成后，请确保使用最新版本的以下软件解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

书籍的代码包也托管在 GitHub 上，网址为 [https://github.com/PacktPublishing/Natural-Language-Processing-with-Python-Quick-Start-Guide](https://github.com/PacktPublishing/Natural-Language-Processing-with-Python-Quick-Start-Guide)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包可供选择，请访问**[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**。查看它们！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[http://www.packtpub.com/sites/default/files/downloads/9781789130386_ColorImages.pdf](http://www.packtpub.com/sites/default/files/downloads/9781789130386_ColorImages.pdf)。

# 使用的约定

本书使用了一些文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个示例：“我使用了 `sed` 语法。”

代码块设置如下：

```py
url = 'http://www.gutenberg.org/ebooks/1661.txt.utf-8'
file_name = 'sherlock.txt'
```

任何命令行输入或输出都如下所示：

```py
import pandas as pd
import numpy as np
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的文字如下所示。以下是一个示例：“预测：pos 实际上是来自我之前上传到本页面的文件的输出。”

警告或重要注意事项看起来像这样。

小技巧和技巧看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并给我们发送电子邮件至 `customercare@packtpub.com`。

**勘误**: 尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果你在这本书中发现了错误，我们将不胜感激，如果你能向我们报告这一点。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果你在互联网上以任何形式遇到我们作品的非法副本，如果你能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并附上材料的链接。

**如果你有兴趣成为作者**: 如果你精通某个主题，并且你对撰写或为书籍做出贡献感兴趣，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。一旦你阅读并使用了这本书，为何不在你购买它的网站上留下评论呢？潜在读者可以查看并使用你的客观意见来做出购买决定，Packt公司可以了解你对我们的产品有何看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于Packt的信息，请访问[packt.com](http://www.packt.com/)。
