# 前言

卷积神经网络（CNN）正在彻底改变多个应用领域，如视觉识别系统、自动驾驶汽车、医学发现、创新电子商务等。本书将带你从 CNN 的基本构建块开始，同时引导你通过最佳实践来实现实际的 CNN 模型和解决方案。你将学习如何为图像和视频分析创造创新的解决方案，以解决复杂的机器学习和计算机视觉问题。

本书从深度神经网络的概述开始，举了一个图像分类的例子，并引导你构建第一个 CNN 模型。你将学习到迁移学习和自编码器等概念，这些概念将帮助你构建非常强大的模型，即使是使用有限的监督（标注图像）训练数据。

后续我们将在这些学习的基础上，构建更高级的与视觉相关的算法和解决方案，用于目标检测、实例分割、生成（对抗）网络、图像描述、注意力机制以及视觉中的递归注意力模型。

除了让你获得与最具挑战性的视觉模型和架构的实践经验外，本书还深入探讨了卷积神经网络和计算机视觉领域的前沿研究。这样，读者可以预测该领域的未来，并通过先进的 CNN 解决方案快速启动创新之旅。

本书结束时，你应该能够在专业项目或个人计划中实施先进、有效且高效的 CNN 模型，尤其是在处理复杂的图像和视频数据集时。

# 本书适合人群

本书适用于数据科学家、机器学习与深度学习从业者，以及认知与人工智能爱好者，他们希望在构建 CNN 模型上更进一步。通过对极大数据集和不同 CNN 架构的实践，你将构建高效且智能的卷积神经网络模型。本书假设你已有一定的深度学习概念基础和 Python 编程语言知识。

# 本书内容概述

第一章，*深度神经网络概述*，快速回顾了深度神经网络的科学原理和不同的框架，这些框架可以用来实现这些网络，并介绍了它们背后的数学原理。

第二章，*卷积神经网络简介*，它将向读者介绍卷积神经网络，并展示如何利用深度学习从图像中提取见解。

第三章，*构建第一个 CNN 并进行性能优化*，从零开始构建一个简单的 CNN 图像分类模型，并解释如何调整超参数、优化训练时间和提升 CNN 的效率与准确性。

第四章，*流行的 CNN 模型架构*，展示了不同流行（并获奖）CNN 架构的优势和工作原理，分析它们的差异，并讲解如何使用它们。

第五章，*迁移学习*，教你如何使用已有的预训练网络，并将其适应到新的数据集上。书中还有一个使用名为**迁移学习**的技术来解决实际应用的自定义分类问题。

第六章，*卷积神经网络的自编码器*，介绍了一种名为**自编码器**的无监督学习技术。我们将通过不同的自编码器应用案例来讲解其在 CNN 中的应用，例如图像压缩。

第七章，*使用 CNN 进行目标检测与实例分割*，讲解了目标检测、实例分割和图像分类之间的区别。接着，我们学习了使用 CNN 进行目标检测与实例分割的多种技术。

第八章，*生成对抗网络（GAN）—使用 CNN 生成新图像*，探索了生成性 CNN 网络，并将它们与我们学习到的判别性 CNN 网络结合，利用 CNN/GAN 生成新图像。

第九章，*CNN 和视觉模型中的注意力机制*，讲解了深度学习中注意力机制的直觉，并学习了基于注意力的模型如何用于实现一些高级解决方案（如图像描述和 RAM）。我们还理解了不同类型的注意力及强化学习在硬注意力机制中的作用。

# 最大化利用本书

本书专注于使用 Python 编程语言构建 CNN。我们使用 Python 2.7 版本（2x）构建了多种应用程序和开源、企业级专业软件，使用的工具包括 Python、Spyder、Anaconda 和 PyCharm。许多示例也兼容 Python 3x。作为良好的实践，我们鼓励用户在实现这些代码时使用 Python 虚拟环境。

我们专注于如何最好地利用各种 Python 和深度学习库（Keras、TensorFlow 和 Caffe）来构建真实世界的应用程序。在这方面，我们尽力使所有代码尽可能友好和易读。我们认为，这将帮助读者轻松理解代码，并在不同的场景中灵活运用。

# 下载示例代码文件

你可以从你的账户下载本书的示例代码文件，网址为 [www.packtpub.com](http://www.packtpub.com)。如果你在其他地方购买了本书，可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便直接通过邮件获得文件。

你可以按照以下步骤下载代码文件：

1.  登录或注册，访问 [www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用最新版本的工具来解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Practical-Convolutional-Neural-Networks`](https://github.com/PacktPublishing/Practical-Convolutional-Neural-Networks)。如果代码有更新，GitHub 上的现有仓库将会进行更新。

我们还有其他的代码包，来自我们丰富的书籍和视频目录，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到它们。快来看看吧！

# 下载彩色图像

我们还提供了一份 PDF 文件，其中包含了本书中使用的截图/图表的彩色图像。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/PracticalConvolutionalNeuralNetworks_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/PracticalConvolutionalNeuralNetworks_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码字、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。例如："将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为您系统中的另一个磁盘。"

代码块如下所示：

```py
import tensorflow as tf

#Creating TensorFlow object 
hello_constant = tf.constant('Hello World!', name = 'hello_constant')
#Creating a session object for execution of the computational graph
with tf.Session() as sess:
```

当我们希望您关注代码块中的某一部分时，相关行或项目会以粗体显示：

```py
x = tf.subtract(1, 2,name=None) # -1
y = tf.multiply(2, 5,name=None) # 10
```

**粗体**：表示新术语、重要词汇或您在屏幕上看到的词汇。例如，菜单或对话框中的词汇将在文本中以此方式显示。举个例子：“从管理面板中选择系统信息。”

警告或重要提示将以此格式出现。

提示和技巧将以此格式显示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：通过电子邮件`feedback@packtpub.com`联系我们，并在邮件主题中注明书名。如果您对本书的任何部分有疑问，请通过`questions@packtpub.com`与我们联系。

**勘误**：虽然我们已尽全力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现任何错误，我们将非常感激您能向我们报告。请访问[www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们的任何作品的非法副本，请提供相关的地址或网站名称。请通过`copyright@packtpub.com`联系我们，并附上该资料的链接。

**如果你有兴趣成为作者**：如果你在某个领域具有专业知识，并且有兴趣撰写或参与编写书籍，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下评价。在你阅读并使用本书后，为什么不在你购买该书的网站上留下评价呢？潜在读者可以通过你的公正意见做出购买决策，我们在 Packt 也能了解你对我们产品的看法，而我们的作者也能看到你对其书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。
