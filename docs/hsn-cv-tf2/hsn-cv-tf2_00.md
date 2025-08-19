# 前言

由于深度学习方法如**卷积神经网络**（**CNN**）的应用，计算机视觉在健康、汽车、社交媒体和机器人等领域达到了新的高度。无论是自动化复杂任务、辅助专家工作，还是帮助艺术家进行创作，越来越多的公司正在整合计算机视觉解决方案。

本书将探索 TensorFlow 2，这是谷歌开源机器学习框架的全新版本。我们将介绍其主要功能以及最先进的解决方案，并演示如何高效地构建、训练和部署 CNN，用于多种实际任务。

# 本书读者对象

本书适合具有一定 Python 编程和图像处理背景的读者（例如，知道如何读取和写入图像文件，以及如何编辑其像素值）。本书有一个逐步深入的学习曲线，既适合深度学习的初学者，也适合对 TensorFlow 2 新特性感兴趣的专家。

虽然一些理论解释需要一定的代数和微积分知识，但为关注实际应用的学习者提供了具体的例子。你将一步一步解决现实中的任务，例如自动驾驶汽车和智能手机应用的视觉识别。

# 本书内容简介

第一章，*计算机视觉与神经网络*，为你介绍计算机视觉和深度学习，提供一些理论背景，并教你如何从零开始实现并训练一个用于视觉识别的神经网络。

第二章，*TensorFlow 基础与模型训练*，讲解了与计算机视觉相关的 TensorFlow 2 概念，以及一些更高级的概念。它介绍了 Keras——现在是 TensorFlow 的一个子模块——并描述了如何使用这些框架训练一个简单的识别方法。

第三章，*现代神经网络*，介绍了卷积神经网络（CNN）并解释了它们如何革新计算机视觉。本章还介绍了正则化工具和现代优化算法，这些工具和算法可用于训练更强大的识别系统。

第四章，*影响力分类工具*，提供了理论细节和实用代码，帮助你熟练地应用最先进的解决方案——如 Inception 和 ResNet——进行图像分类。本章还解释了什么是迁移学习，为什么它是机器学习中的关键概念，并讲解了如何使用 TensorFlow 2 执行迁移学习。

第五章，*物体检测模型*，介绍了两种方法的架构，用于检测图像中的特定物体——以速度著称的 You Only Look Once 和以精度著称的 Faster R-CNN。

第六章，*增强与图像分割*，介绍了自编码器以及如何应用 U-Net 和 FCN 等网络进行图像去噪、语义分割等任务。

第七章，*在复杂和稀缺数据集上训练*，专注于高效收集和预处理数据集以应用于深度学习。书中介绍了构建优化数据管道的 TensorFlow 工具，以及多种补偿数据稀缺性的解决方案（图像渲染、领域适应、生成网络如 VAE 和 GAN）。

第八章，*视频和递归神经网络*，介绍了递归神经网络，并展示了更先进的版本——长短时记忆（LSTM）架构。它提供了实际代码，应用 LSTM 进行视频中的动作识别。

第九章，*优化模型并在移动设备上部署*，详细介绍了模型优化方面的内容，包括速度、磁盘空间和计算性能。书中通过一个实际例子讲解了如何在移动设备和浏览器上部署 TensorFlow 解决方案。

附录，*从 TensorFlow 1 到 TensorFlow 2 的迁移*，提供了一些关于 TensorFlow 1 的信息，重点介绍了 TensorFlow 2 引入的关键变化。同时也提供了将旧项目迁移到最新版本的指南。最后，为了那些想要深入了解的读者，每章参考文献也一并列出。

# 为了最大化利用本书

以下部分包含一些信息和建议，旨在帮助读者更好地阅读本书，并充分利用附加材料。

# 下载并运行示例代码文件

练习成就完美。因此，本书不仅提供了对 TensorFlow 2 和最先进计算机视觉方法的深入解释，还附带了每章的许多实际示例和完整实现。

# 下载代码文件

您可以从您的 [www.packt.com](http://www.packt.com) 账户中下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packtpub.com/support](https://www.packtpub.com/support) 并注册，以便直接通过电子邮件获取文件。

您可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用以下最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，地址为 **[`github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2`](https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2)**。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还提供了其他代码包，来自我们丰富的书籍和视频目录，均可在 **[`github.com/PacktPublishing`](https://github.com/PacktPublishing/)** 查找。赶快去看看吧！

# 学习并运行实验

**Jupyter Notebook**（[`jupyter.org`](https://jupyter.org)）是一个开源的 Web 应用，用于创建和分享 Python 脚本，结合文本信息、视觉结果、公式等更多内容。我们将 *Jupyter 笔记本* 称为本书中提供的文档，包含详细的代码、预期结果以及补充说明。每个 Jupyter 笔记本都专注于一个具体的计算机视觉任务。例如，一个笔记本解释了如何训练 CNN 来检测图像中的动物，另一个详细描述了构建自动驾驶汽车识别系统的所有步骤，等等。

正如我们将在本节中看到的，这些文档可以直接学习，也可以作为代码示例运行，重现书中展示的实验。

# 在线学习 Jupyter 笔记本

如果你只是想浏览提供的代码和结果，可以直接在线访问本书的 *GitHub* 仓库中的内容。事实上，GitHub 可以渲染 Jupyter 笔记本并将其显示为静态网页。

然而，GitHub 查看器会忽略一些样式格式和交互内容。为了获得最佳的在线查看体验，我们推荐使用 *Jupyter nbviewer*（[`nbviewer.jupyter.org`](https://nbviewer.jupyter.org)），这是一个官方的在线平台，你可以用它来读取上传到网上的 Jupyter 笔记本。该网站可以查询并渲染存储在 GitHub 仓库中的笔记本。因此，提供的 Jupyter 笔记本也可以通过以下地址进行阅读：[`nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2`](https://nbviewer.jupyter.org/github/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2)。

# 在你的机器上运行 Jupyter 笔记本

要在你的机器上阅读或运行这些文档，你应该首先安装 Jupyter Notebook。对于已经使用 *Anaconda*（[`www.anaconda.com`](https://www.anaconda.com)）来管理和部署 Python 环境的用户（如我们在本书中推荐的那样），Jupyter Notebook 应该已经可以直接使用（因为它与 Anaconda 一起安装）。对于使用其他 Python 发行版的用户，或者不熟悉 Jupyter Notebook 的用户，我们推荐查看文档，它提供了安装说明和教程（[`jupyter.org/documentation`](https://jupyter.org/documentation)）。

一旦 Jupyter Notebook 安装在您的机器上，导航到包含书籍代码文件的目录，打开终端并执行以下命令：

```py
$ jupyter notebook
```

网页界面应在您的默认浏览器中打开。此时，您应该能够浏览目录并打开提供的 Jupyter 笔记本，进行阅读、执行或编辑。

一些文档包含高级实验，这些实验可能需要大量计算资源（例如，使用大型数据集训练识别算法）。没有适当的加速硬件（即没有兼容的 NVIDIA GPU，如第二章《TensorFlow 基础与模型训练》所述），这些脚本可能需要数小时甚至数天的时间（即使有兼容的 GPU，最先进的示例仍然可能需要很长时间）。

# 在 Google Colab 中运行 Jupyter 笔记本

对于那些希望自行运行 Jupyter 笔记本——或进行新的实验——但没有足够强大机器的用户，我们推荐使用**Google Colab**，也叫做**Colaboratory** ([`colab.research.google.com`](https://colab.research.google.com))。它是 Google 提供的基于云的 Jupyter 环境，用户可以在强大的机器上运行计算密集型脚本。有关此服务的更多详情，请查阅 GitHub 仓库。

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含书中使用的截图/图表的彩色图像。您可以在此下载：[`www.packtpub.com/sites/default/files/downloads/9781788830645_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788830645_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL 以及用户输入。例如：“`Model`对象的`.fit()`方法启动训练过程。”

代码块的设置如下：

```py
import tensorflow as tf

x1 = tf.constant([[0, 1], [2, 3]])
x2 = tf.constant(10)
x = x1 * x2
```

当我们希望您注意代码块中的特定部分时，相关的行或项会用粗体显示：

```py
neural_network = tf.keras.Sequential(
    [tf.keras.layers.Dense(64),
     tf.keras.layers.Dense(10, activation="softmax")])
```

任何命令行输入或输出如下所示：

```py
$ tensorboard --logdir ./logs
```

**粗体**：表示新术语、重要词汇或您在屏幕上看到的词汇。例如，菜单或对话框中的词汇会以这种方式显示。这里是一个例子：“您可以在 TensorBoard 的 Scalars 页面上观察解决方案的表现。”

警告或重要提示如下所示。

提示和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中注明书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：尽管我们已经尽力确保内容的准确性，但错误还是难免。如果您在本书中发现了错误，我们将不胜感激，您可以将错误报告给我们。请访问 [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击“勘误提交表单”链接，并填写相关细节。

**盗版**：如果您在互联网上发现任何我们作品的非法复制品，请您提供该内容的地址或网站名称，我们将不胜感激。请通过 `copyright@packt.com` 联系我们，并提供该材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专长，并且有兴趣写作或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下评价。在您阅读并使用了本书后，为什么不在您购买书籍的网站上留下评论呢？潜在读者可以看到并使用您的客观意见来做出购买决策，我们 Packt 可以了解您对我们产品的看法，我们的作者也能看到您对他们书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
