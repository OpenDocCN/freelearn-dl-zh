# 序言

Caffe2 是一个流行的深度学习框架，专注于可伸缩性、高性能和可移植性。它用 C++编写，提供了 C++ API 和 Python API。本书是你快速入门 Caffe2 的指南。内容包括安装 Caffe2，使用其操作符组合网络，训练模型以及将模型部署到推理引擎、边缘设备和云端。我们还将展示如何使用 ONNX 交换格式在 Caffe2 和其他深度学习框架之间进行工作。

# 本书适合读者

数据科学家和机器学习工程师希望在 Caffe2 中创建快速且可扩展的深度学习模型，将会发现本书非常有用。

# 本书内容

第一章，*介绍和安装*，介绍了 Caffe2，并探讨了如何构建和安装它。

第二章，*网络组合*，教你如何使用 Caffe2 操作符以及如何组合它们来构建简单的计算图和神经网络，用于识别手写数字。

第三章，*网络训练*，介绍了如何使用 Caffe2 来组合一个用于训练的网络，以及如何训练解决 MNIST 问题的网络。

第四章，*与 Caffe 一起工作*，探索了 Caffe 与 Caffe2 之间的关系，以及如何处理在 Caffe 中训练的模型。

第五章，*与其他框架一起工作*，探讨了 TensorFlow 和 PyTorch 等当代深度学习框架，以及如何在 Caffe2 和这些框架之间交换模型。

第六章，*将模型部署到加速器进行推理*，讨论了推理引擎及其在训练后将 Caffe2 模型部署到加速器上的关键工具。我们专注于两种流行的加速器类型：NVIDIA GPU 和 Intel CPU。我们将探讨如何安装和使用 TensorRT 来在 NVIDIA GPU 上部署我们的 Caffe2 模型。同时，我们还将介绍如何安装和使用 OpenVINO 来在 Intel CPU 和加速器上部署我们的 Caffe2 模型。

第七章，*边缘设备和云中的 Caffe2*，展示了 Caffe2 的两个应用场景，以展示其扩展能力。作为 Caffe2 在边缘设备中的应用，我们将介绍如何在树莓派单板计算机上构建 Caffe2，并在其上运行 Caffe2 应用程序。作为 Caffe2 在云中的应用，我们将探讨如何在 Docker 容器中使用 Caffe2。

# 为了充分利用本书

一些基本机器学习概念的理解和对 C++和 Python 等编程语言的先前接触将会很有帮助。

# 下载示例代码文件

您可以从您的帐户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果您是在其他地方购买的本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便直接通过邮件收到文件。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”选项卡。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Caffe2-Quick-Start-Guide`](https://github.com/PacktPublishing/Caffe2-Quick-Start-Guide)。如果代码有更新，它将在现有的 GitHub 仓库中进行更新。

我们还有来自我们丰富书籍和视频目录的其他代码包，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到它们。赶紧去看看吧！

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如：“`SoftMax`函数的输出值具有良好的性质。”

一块代码按如下方式呈现：

```py
model = model_helper.ModelHelper("MatMul model")
model.net.MatMul(["A", "B"], "C")
```

任何命令行输入或输出都按如下方式书写：

```py
$ sudo apt-get update
```

**粗体**：表示一个新术语、重要单词或您在屏幕上看到的词语。例如，菜单或对话框中的词语会以这种方式出现在文本中。举个例子：“例如，在 Ubuntu 上，它给了我一个下载可自定义包或单一大包的选项。”

警告或重要说明以这种方式出现。

小贴士和技巧以这种方式出现。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何部分有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：尽管我们已经尽力确保内容的准确性，但错误还是会发生。如果您在本书中发现错误，我们将非常感激您向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入详细信息。

**盗版**：如果您在互联网上发现任何形式的我们作品的非法复制品，我们将非常感激您提供该位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上材料链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣撰写或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 书评

请留下书评。在阅读并使用本书之后，为什么不在你购买该书的网站上留下评论呢？潜在的读者可以看到并参考你客观的意见来做出购买决策，我们在 Packt 可以了解你对我们产品的看法，而我们的作者也可以看到你对他们书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。
