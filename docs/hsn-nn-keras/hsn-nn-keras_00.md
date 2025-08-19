# 前言

神经网络是一种数学函数，广泛应用于**人工智能**（**AI**）和深度学习的各个领域，用来解决各种问题。《动手实践 Keras 神经网络》将从介绍神经网络的核心概念开始，让你深入理解各种神经网络模型的组合应用，并结合真实世界的用例，帮助你更好地理解预测建模和函数逼近的价值。接下来，你将熟悉多种最著名的架构，包括但不限于**卷积神经网络**（**CNNs**）、**递归神经网络**（**RNNs**）、**长短期记忆**（**LSTM**）网络、**自编码器**和**生成对抗网络**（**GANs**），并使用真实的训练数据集进行实践。

我们将探索计算机视觉和**自然语言处理**（**NLP**）等认知任务背后的基本理念和实现细节，采用最先进的神经网络架构。我们将学习如何将这些任务结合起来，设计出更强大的推理系统，极大地提高各种个人和商业环境中的生产力。本书从理论和技术角度出发，帮助你直观理解神经网络的内部工作原理。它将涵盖各种常见的用例，包括监督学习、无监督学习和自监督学习任务。在本书的学习过程中，你将学会使用多种网络架构，包括用于图像识别的 CNN、用于自然语言处理的 LSTM、用于强化学习的 Q 网络等。我们将深入研究这些具体架构，并使用行业级框架进行动手实践。

到本书的最后，你将熟悉所有著名的深度学习模型和框架，以及你在将深度学习应用于现实场景、将 AI 融入组织核心的过程中所需的所有选项。

# 本书适合谁阅读

本书适用于**机器学习**（**ML**）从业者、深度学习研究人员和希望通过 Keras 熟悉不同神经网络架构的 AI 爱好者。掌握 Python 编程语言的基本知识是必需的。

# 为了从本书中获得最大收获

具有一定的 Python 知识将大有裨益。

# 下载示例代码文件

你可以从 [www.packt.com](http://www.packt.com) 的账户中下载本书的示例代码文件。如果你是在其他地方购买的本书，可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，文件将直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  登录或在 [www.packt.com](http://www.packt.com) 注册。

1.  选择“SUPPORT”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并按照屏幕上的说明操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Neural-Networks-with-Keras`](https://github.com/PacktPublishing/Hands-On-Neural-Networks-with-Keras)。如果代码有更新，它将会更新到现有的 GitHub 仓库中。

我们还提供了来自我们丰富图书和视频目录的其他代码包，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看！

# 下载彩色图片

我们还提供了一个 PDF 文件，包含了本书中使用的截图/图示的彩色图片。您可以在此下载：`www.packtpub.com/sites/default/files/downloads/9781789536089_ColorImages.pdf`。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如：“这指的是张量中存储的数据类型，可以通过调用张量的`type()`方法进行检查。”

代码块如下所示：

```py
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项会用粗体显示：

```py
keras.utils.print_summary(model, line_length=None, positions=None,    
                          print_fn=None)
```

任何命令行输入或输出如下所示：

```py
! pip install keras-vis
```

**粗体**：表示新术语、重要单词或屏幕上显示的单词。

警告或重要说明如下所示。

提示和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何部分有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：尽管我们已经尽力确保内容的准确性，但难免会有错误。如果您在本书中发现错误，我们将非常感激您向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入相关细节。

**盗版**：如果您在互联网上发现我们作品的任何非法复制版本，我们将非常感激您提供该位置地址或网站名称。请通过`copyright@packt.com`联系我们，并提供相关材料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上有专业知识，并且有兴趣撰写或参与一本书的编写，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 书评

请留下评论。阅读并使用本书后，为什么不在您购买书籍的网站上留下评论呢？潜在的读者可以看到并参考您公正的意见来做出购买决策，我们在 Packt 也能了解您对我们产品的看法，而我们的作者也能看到您对他们书籍的反馈。感谢您！

欲了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
