# 前言

Python 是一种编程语言，在数据科学领域提供了多种功能。在本书中，我们将接触到两个 Python 库：scikit-learn 和 TensorFlow。我们将学习集成方法的各种实现，了解它们如何与实际数据集结合使用，以及它们如何提高分类和回归问题中的预测准确性。

本书从学习集成方法及其特点开始。我们将看看 scikit-learn 如何提供合适的工具来选择模型的超参数。从这里，我们将深入研究预测分析，探索其各种特性和特点。我们将了解人工神经网络、TensorFlow 以及用于构建神经网络的核心概念。

在最后一部分，我们将考虑计算能力、改进方法和软件增强等因素，以提高预测分析的效率。你将熟练掌握使用深度神经网络（DNN）来解决常见挑战。

# 本书适用对象

本书适用于数据分析师、软件工程师和机器学习开发者，特别是那些有兴趣使用 Python 实现高级预测分析的人。商业智能专家也会发现本书不可或缺，因为它将教会他们如何从基础预测模型开始，逐步构建更高级的模型，并做出更好的预测。假设读者具备 Python 知识并熟悉预测分析的概念。

# 本书内容概述

第一章，*回归与分类的集成方法*，讲述了集成方法或算法在模型预测中的应用。我们将通过回归和分类问题中的集成方法应用进行学习。

第二章，*交叉验证与参数调优*，探讨了组合和构建更好模型的各种技术。我们将学习不同的交叉验证方法，包括保留法交叉验证和 k 折交叉验证。我们还将讨论什么是超参数调优。

第三章，*特征工程*，探讨了特征选择方法、降维、主成分分析（PCA）和特征工程。我们还将学习如何通过特征工程来改善模型。

第四章，*人工神经网络与 TensorFlow 简介*，是对人工神经网络（ANNs）和 TensorFlow 的介绍。我们将探讨网络中的各种元素及其功能。我们还将在其中学习 TensorFlow 的基本概念。

第五章，*使用 TensorFlow 和深度神经网络进行预测分析*，通过 TensorFlow 和深度学习的帮助探索预测分析。我们将学习 MNIST 数据集及其分类模型的应用。我们将了解 DNN（深度神经网络）、它们的功能以及 DNN 在 MNIST 数据集中的应用。

# 最大限度地发挥本书的价值

本书介绍了一些最先进的预测分析工具、模型和技术。主要目标是展示如何通过构建更复杂的模型来提高预测模型的性能，其次，通过使用相关技术大幅提高预测模型的质量。

# 下载示例代码文件

你可以从你的帐户在 [www.packt.com](http://www.packt.com) 下载本书的示例代码文件。如果你是从其他地方购买的本书，可以访问 [www.packt.com/support](http://www.packt.com/support)，并注册以便直接通过电子邮件接收文件。

你可以按照以下步骤下载代码文件：

1.  登录或注册 [www.packt.com](http://www.packt.com)。

1.  选择 SUPPORT 标签。

1.  点击代码下载与勘误。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用以下最新版本解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，链接为**[`github.com/PacktPublishing/Mastering-Predictive-Analytics-with-scikit-learn-and-TensorFlow`](https://github.com/PacktPublishing)**。如果代码有更新，它将会在现有的 GitHub 仓库中进行更新。

我们还提供了其他来自我们丰富的图书和视频目录中的代码包，链接为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快来看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含本书中使用的截图/图表的彩色图像。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789617740_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789617740_ColorImages.pdf)。

# 使用的规范

本书中使用了多种文本规范。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名。示例：“以下截图显示了用于导入`train_test_split`函数和`RobustScaler`方法的代码行。”

一段代码如下所示：

```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
```

**粗体**：表示新术语、重要单词或你在屏幕上看到的词语。例如，菜单或对话框中的词语会像这样出现在文本中。以下是一个例子：“用于选择最佳估计器的算法，或选择所有超参数的最佳值，称为**超参数调优**。”

警告或重要提示以这种方式出现。

小贴士和技巧以这种方式出现。

# 联系我们

我们非常欢迎读者的反馈。

**一般反馈**：如果你对本书的任何方面有疑问，请在邮件主题中注明书名，并通过`customercare@packtpub.com`联系我们。

**勘误表**：虽然我们已经尽力确保内容的准确性，但错误有时会发生。如果你在本书中发现错误，我们将不胜感激你向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书籍，点击勘误提交表单链接，并输入相关详情。

**盗版**：如果你在互联网上发现我们的作品有任何非法复制形式，我们将不胜感激你提供相关地址或网站名称。请通过`copyright@packt.com`与我们联系，并提供该材料的链接。

**如果你有兴趣成为作者**：如果你在某个领域具有专业知识，并且有兴趣写书或为书籍做贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在购买地点留下评论呢？潜在读者可以看到并参考你的客观意见做出购买决策，我们在 Packt 也能了解你对我们产品的看法，而我们的作者可以看到你对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
