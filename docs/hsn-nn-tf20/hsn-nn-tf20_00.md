# 序言

科技领袖们正在采用神经网络来增强他们的产品，使其更智能，或者用营销术语来说，就是赋予 AI 能力。本书是一本实用的 TensorFlow 指南，涵盖了其内部结构、2.0 版本的新特性以及如何利用这些特性来创建基于神经网络的应用程序。读完本书，您将对 TensorFlow 架构及其新特性有深入了解，并能轻松使用神经网络的力量解决机器学习问题。

本书从机器学习和神经网络的理论概述开始，然后描述了 TensorFlow 库的 1.x 和 2.0 版本。在阅读本书时，您将通过易于理解的示例深入掌握理解神经网络工作原理所需的理论。接下来，您将学习如何掌握优化技术和算法，利用 TensorFlow 2.0 提供的新模块构建各种神经网络架构。此外，在分析完 TensorFlow 结构后，您将学会如何实现更复杂的神经网络架构，如用于分类的 CNN、语义分割网络、生成对抗网络等，以便在您的研究工作和项目中使用。

本书结束时，您将掌握 TensorFlow 结构，并能够利用这个机器学习框架的强大功能，轻松训练和使用各种复杂度的神经网络。

# 本书适用人群

本书面向数据科学家、机器学习开发人员、深度学习研究人员以及具有基本统计学背景的开发人员，适合那些希望使用神经网络并探索 TensorFlow 结构及其新特性的人。为了最大程度地从本书中受益，您需要具备一定的 Python 编程语言基础。

# 本书内容简介

第一章，*什么是机器学习？*，介绍机器学习的基本概念：监督学习、无监督学习和半监督学习的定义及其重要性。此外，您将开始了解如何创建数据管道，如何衡量算法性能，以及如何验证您的结果。

第二章，*神经网络与深度学习*，重点介绍神经网络。您将了解机器学习模型的优势，如何让网络进行学习，以及在实际中如何执行模型参数更新。阅读完本章后，您将理解反向传播和网络参数更新背后的直觉。此外，您还将了解为何深度神经网络架构对于解决具有挑战性的任务是必需的。

第三章，*TensorFlow 图架构*，介绍了 TensorFlow 的结构——这是 1.x 和 2.x 版本之间共享的结构。

第四章，*TensorFlow 2.0 架构*，展示了 TensorFlow 1.x 与 TensorFlow 2.x 之间的区别。你将开始使用这两个版本开发一些简单的机器学习模型，同时深入了解这两个版本的所有常见功能。

第五章，*高效数据输入管道与估算器 API*，展示了如何使用`tf.data` API 定义完整的数据输入管道，并结合`tf.estimator` API 定义实验。通过本章的学习，你将能够创建复杂且高效的输入管道，充分利用`tf.data`和`tf.io.gfile` API 的强大功能。

第六章，*使用 TensorFlow Hub 进行图像分类*，介绍了如何通过利用 TensorFlow Hub 与 Keras API 的紧密集成，轻松实现迁移学习与微调。

第七章，*目标检测入门*，展示了如何扩展你的分类器，将其转变为一个目标检测器，回归边界框的坐标，并为你介绍更复杂的目标检测架构。

第八章，*语义分割与自定义数据集构建器*，介绍了如何实现语义分割网络，如何为此类任务准备数据集，以及如何训练和衡量模型的性能。你将通过 U-Net 解决语义分割问题。

第九章，*生成对抗网络*，从理论和实践的角度介绍了 GAN。你将了解生成模型的结构，以及如何使用 TensorFlow 2.0 轻松实现对抗训练。

第十章，*将模型投入生产*，展示了如何将训练好的模型转化为一个完整的应用。本章还介绍了如何将训练好的模型导出为指定的表示形式（SavedModel）并在完整应用中使用。通过本章的学习，你将能够导出训练好的模型，并在 Python、TensorFlow.js 和使用 tfgo 库的 Go 语言中使用它。

# 为了最大限度地利用本书

你需要具备基本的神经网络知识，但这并非强制要求，因为这些内容将从理论和实践两个角度进行讲解。如果你具备基本的机器学习算法知识会更有帮助。此外，你需要具备良好的 Python 3 使用能力。

你应该已经知道如何使用 `pip` 安装包，如何设置工作环境以便与 TensorFlow 配合使用，以及如何启用（如果可用）GPU 加速。此外，还需要有一定的编程概念基础，例如命令式语言与描述性语言以及面向对象编程的知识。

环境设置将在第三章中介绍，*TensorFlow 图架构*将在机器学习和神经网络理论的前两章之后讲解。

# 下载示例代码文件。

你可以通过你的账户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果你是在其他地方购买的本书，你可以访问[www.packtpub.com/support](http://www.packtpub.com/support)，并注册以直接将文件通过电子邮件发送给你。

你可以按照以下步骤下载代码文件：

1.  登录或注册到 [www.packt.com](http://www.packt.com)。

1.  选择“支持”标签。

1.  点击“代码下载”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

文件下载后，请确保使用最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Neural-Networks-with-TensorFlow-2.0`](https://github.com/PacktPublishing/Hands-On-Neural-Networks-with-TensorFlow-2.0)。如果代码有更新，将会更新到现有的 GitHub 仓库。

我们还提供了其他代码包，来自我们丰富的书籍和视频目录，可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。快去看看吧！

# 下载彩色图片。

我们还提供了一个 PDF 文件，其中包含本书中使用的截图/图表的彩色图片。你可以在这里下载：[`static.packt-cdn.com/downloads/9781789615555_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781789615555_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名。举个例子：“将下载的 `WebStorm-10*.dmg` 磁盘映像文件挂载为系统中的另一个磁盘。”

一块代码如下所示：

```py
writer = tf.summary.FileWriter("log/two_graphs/g1", g1)
writer = tf.summary.FileWriter("log/two_graphs/g2", g2)
writer.close()
```

任何命令行输入或输出均按如下格式书写：

```py
# create the virtualenv in the current folder (tf2)
pipenv --python 3.7
# run a new shell that uses the just created virtualenv
pipenv shell
# install, in the current virtualenv, tensorflow
pip install tensorflow==2.0
#or for GPU support: pip install tensorflow-gpu==2.0
```

**粗体**：表示一个新术语、一个重要的单词，或屏幕上出现的单词。例如，菜单或对话框中的单词会像这样出现在文本中。举个例子：“`tf.Graph` 结构的第二个特点是它的 **图集合**。”

警告或重要提示会以这种方式出现。

提示和技巧会以这种方式出现。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果你对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`与我们联系。

**勘误**：虽然我们已经尽力确保内容的准确性，但错误仍然可能发生。如果你在本书中发现错误，我们将非常感谢你向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书籍，点击“勘误提交表单”链接，并填写相关信息。

**盗版**：如果你在互联网上发现任何形式的我们作品的非法副本，我们将非常感谢你提供该位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并提供相关链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣写作或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。在阅读并使用本书后，为什么不在你购买本书的网站上留下评论呢？潜在读者可以看到并参考你公正的意见来做出购买决策，我们 Packt 也能了解你对我们产品的看法，而我们的作者也可以看到你对他们书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
