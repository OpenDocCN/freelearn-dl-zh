# 前言

**深度学习**（**DL**）是一个日益流行的话题，吸引了各大公司以及各种开发者的关注。在过去的五年里，这一领域经历了巨大的进步，最终让我们将 DL 视为一种具有巨大潜力的颠覆性技术。虚拟助手、语音识别和语言翻译只是 DL 技术直接应用的几个例子。与图像识别或物体检测相比，这些应用使用的是顺序数据，其中每个结果的性质都依赖于前一个结果。例如，要将英语句子翻译成西班牙语，你无法不从头到尾追踪每个单词的变化。对于这些问题，正在使用一种特定类型的模型——**递归神经网络**（**RNN**）。本书将介绍 RNN 的基础知识，并重点讲解如何利用流行的深度学习库 TensorFlow 进行一些实际的实现。所有示例都附有深入的理论解释，帮助你理解这个强大但稍微复杂的模型背后的基本概念。阅读本书后，你将对 RNN 有信心，并能够在你的具体应用场景中使用这个模型。

# 本书适用对象

本书面向那些希望通过实际案例了解 RNN 的机器学习工程师和数据科学家。

# 本书内容概览

第一章，*递归神经网络简介*，将为你简要介绍 RNN 的基础知识，并将该模型与其他流行的模型进行比较，展示为什么 RNN 是最好的。随后，本章将通过一个示例来说明 RNN 的应用，同时还会让你了解 RNN 面临的一些问题。

第二章，*使用 TensorFlow 构建你的第一个 RNN*，将探讨如何构建一个简单的 RNN 来解决序列奇偶性识别问题。你还将简要了解 TensorFlow 库以及它如何用于构建深度学习模型。阅读完这一章后，你应该能够完全理解如何在 Python 中使用 TensorFlow，并体会到构建神经网络是多么简单直接。

第三章，*生成你自己的书籍章节*，将介绍一种更强大的 RNN 模型——**门控递归单元**（**GRU**）。你将了解它是如何工作的，以及为什么我们选择它而不是简单的 RNN。你还将逐步了解生成书籍章节的过程。通过这一章的学习，你将获得理论和实践上的双重知识，这将使你能够自由地尝试解决中等难度的任何问题。

第四章，*创建西班牙语到英语的翻译器*，将引导你通过使用 TensorFlow 库实现的序列到序列模型构建一个相当复杂的神经网络模型。你将构建一个简单的西班牙语到英语的翻译器，它可以接受西班牙语句子并输出相应的英语翻译。

第五章，*构建个人助手*，将探讨 RNN 的实际应用，并指导你构建一个对话聊天机器人。本章展示了一个完整实现的聊天机器人系统，能够构建一个简短的对话。然后，你将创建一个端到端模型，旨在产生有意义的结果。你将使用一个基于 TensorFlow 的高阶库——TensorLayer。

第六章，*提升 RNN 的性能*，将介绍一些提高 RNN 性能的技巧。本章将专注于通过数据和调优来提升 RNN 的性能。你还将探索如何优化 TensorFlow 库以获得更好的结果。

# 充分利用本书

你需要具备基本的 Python 3.6.x 知识和 Linux 命令的基础知识。此前使用过 TensorFlow 的经验将会有帮助，但并非必须。

# 下载示例代码文件

你可以从[www.packt.com](http://www.packt.com)账户下载本书的示例代码文件。如果你是从其他地方购买的本书，可以访问[www.packt.com/support](http://www.packt.com/support)，注册后将文件直接发送到你的邮箱。

你可以通过以下步骤下载代码文件：

1.  登录或注册到[www.packt.com](http://www.packt.com)。

1.  选择“SUPPORT”标签。

1.  点击“Code Downloads & Errata”。

1.  在搜索框中输入书名，按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，地址是[`github.com/PacktPublishing/Recurrent-Neural-Networks-with-Python-Quick-Start-Guide`](https://github.com/PacktPublishing/Recurrent-Neural-Networks-with-Python-Quick-Start-Guide)。如果代码有更新，它将会在现有的 GitHub 仓库中同步更新。

我们还有其他来自丰富书籍和视频目录的代码包，访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。赶紧去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，包含本书中使用的截图/图表的彩色图片。你可以在此下载：[`www.packtpub.com/sites/default/files/downloads/9781789132335_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789132335_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名。例如：“将下载的 `WebStorm-10*.dmg` 磁盘映像文件挂载为系统中的另一个磁盘。”

代码块以如下方式显示：

```py
def generate_data():
    inputs = input_values()
    return inputs, output_values(inputs)
```

当我们希望你特别注意某段代码时，相关行或项目会以粗体显示：

```py
 loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,    
    logits=prediction)
 total_loss = tf.reduce_mean(loss)
```

任何命令行输入或输出都以如下方式书写：

```py
import tensorflow as tf
import random
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词语。例如，菜单或对话框中的词语会像这样出现在文本中。这里有一个例子：“从管理面板中选择系统信息。”

警告或重要说明通常以这种方式显示。

提示和技巧通常以这种方式显示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果你对本书的任何内容有疑问，请在邮件的主题中提及书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：虽然我们已经尽力确保内容的准确性，但难免会出现错误。如果你发现本书中的错误，我们非常感激你能向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书籍，点击“勘误提交表单”链接，并填写相关信息。

**盗版**：如果你在互联网上发现我们的作品的非法复制品，无论是何种形式，我们非常感激你能提供该材料的链接地址或网站名称。请通过`copyright@packt.com`与我们联系，并提供该资料的链接。

**如果你有兴趣成为作者**：如果你在某个领域有专长，且有意写作或参与编写书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 书评

请留下书评。阅读并使用本书后，不妨在你购买本书的网站上留下评价。潜在读者可以查看并根据你的公正意见做出购买决策，我们也可以了解你对我们产品的看法，我们的作者则能看到你对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
