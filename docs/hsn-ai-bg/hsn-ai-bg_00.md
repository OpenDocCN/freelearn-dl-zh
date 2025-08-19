# 前言

像 Alexa 和 Siri 这样的虚拟助手处理我们的请求，谷歌的汽车已经开始读取地址，亚马逊的价格和 Netflix 的推荐视频由人工智能决定。人工智能是最令人兴奋的技术之一，并且在现代世界中变得越来越重要。

*初学者的动手人工智能*将教你什么是人工智能，以及如何设计和构建智能应用程序。本书将教你如何利用像 TensorFlow 这样的包来创建强大的 AI 系统。你将首先回顾人工智能的最新变化，并了解**人工神经网络**（**ANNs**）如何使人工智能变得更加智能。你将探索前馈、递归、卷积和生成神经网络（FFNNs，...）

# 本书适合人群

本书面向 AI 初学者、想成为 AI 开发者的人以及对利用各种算法构建强大 AI 应用感兴趣的机器学习爱好者。

# 为了最大限度地发挥本书的作用

本章中的代码可以直接在 Jupyter 和 Python 中执行。本书的代码文件可以在以下章节中提供的 GitHub 链接找到。

# 下载示例代码文件

你可以从你的账户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果你从其他地方购买了本书，可以访问[www.packt.com/support](http://www.packt.com/support)，注册后将文件直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  登录或注册到[www.packt.com](http://www.packt.com)。

1.  选择 SUPPORT 选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用最新版本解压或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，链接为[`github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Beginners`](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Beginners)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还提供了来自我们丰富的书籍和视频目录中的其他代码包，访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。赶快去看看吧！

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名。示例如下：“`tf.add`函数将一个层添加到我们的网络中。”

代码块的设置如下：

```py
import tensorflow as tffrom tensorflow.examples.tutorials.mnist import input_datamnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```

任何命令行输入或输出都如下所示：

```py
tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_func)
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词汇。例如，菜单或对话框中的词语会显示为...

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中注明书名，并通过`customercare@packtpub.com`与我们联系。

**勘误**：虽然我们已尽全力确保内容的准确性，但错误仍然会发生。如果您在本书中发现错误，我们将不胜感激，如果您能将此报告给我们。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入相关详情。

**盗版**：如果您在互联网上发现我们作品的任何非法复制品，请您提供相关位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上该素材的链接。

**如果您有兴趣成为作者**：如果您在某个领域拥有专业知识，并且有兴趣撰写或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。在您阅读并使用了本书后，为什么不在您购买书籍的网站上留下评论呢？潜在读者可以看到并参考您客观的意见来做出购买决定，我们在 Packt 也可以了解您对我们产品的看法，我们的作者可以看到您对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
