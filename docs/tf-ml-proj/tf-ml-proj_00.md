# 前言

TensorFlow 已经彻底改变了人们对机器学习的认知。*TensorFlow 机器学习项目*将教您如何利用 TensorFlow 在各种实际项目中发挥其优势——简洁性、效率和灵活性。在本书的帮助下，您不仅将学习如何使用不同的数据集构建高级项目，还能通过 TensorFlow 生态系统中的多个库解决常见的挑战。

首先，您将掌握使用 TensorFlow 进行机器学习项目的基本方法。您将通过使用 TensorForest 和 TensorBoard 进行系外行星检测、使用 TensorFlow.js 进行情感分析以及使用 TensorFlow Lite 进行数字分类，来探索一系列项目。

在阅读本书的过程中，您将构建涉及**自然语言处理**（**NLP**）、高斯过程、自编码器、推荐系统和贝叶斯神经网络等多个实际领域的项目，还将涉及诸如**生成对抗网络**（**GANs**）、胶囊网络和强化学习等热门领域。您将学习如何结合 Spark API 使用 TensorFlow，并探索利用 TensorFlow 进行 GPU 加速计算以进行物体检测，接着了解如何训练和开发**递归神经网络**（**RNN**）模型来生成书籍脚本。

在本书结束时，您将获得足够的专业知识，能够在工作中构建完善的机器学习项目。

# 本书适合谁阅读

如果您是数据分析师、数据科学家、机器学习专家或具有基本 TensorFlow 知识的深度学习爱好者，那么《*TensorFlow 机器学习项目*》非常适合您。如果您想在机器学习领域使用监督学习、无监督学习和强化学习技术构建端到端的项目，这本书同样适合您。

# 本书涵盖的内容

第一章，*TensorFlow 与机器学习概述*，解释了 TensorFlow 的基本概念，并让您构建一个使用逻辑回归分类手写数字的机器学习模型。

第二章，*使用机器学习检测外太空中的系外行星*，介绍了如何使用基于决策树的集成方法来检测外太空中的系外行星。

第三章，*在浏览器中使用 TensorFlow.js 进行情感分析*，介绍了如何在您的网页浏览器上使用 TensorFlow.js 训练和构建模型。我们将使用电影评论数据集构建一个情感分析模型，并将其部署到浏览器中进行预测。

第四章，*使用 TensorFlow Lite 进行数字分类*，重点介绍了构建深度学习模型来分类手写数字，并将其转换为适合移动设备的格式，使用 TensorFlow Lite。我们还将学习 TensorFlow Lite 的架构以及如何使用 TensorBoard 来可视化神经网络。

第五章，*使用 NLP 进行语音转文本和主题提取*，重点介绍了通过 TensorFlow 学习各种语音转文本选项以及 Google 在 TensorFlow 中提供的预构建模型，使用 Google 语音命令数据集。

第六章，*使用高斯过程回归预测股票价格*，解释了一种流行的预测模型——贝叶斯统计中的高斯过程。我们使用`GpFlow`库中构建的高斯过程，它是基于 TensorFlow 开发的，来开发股票价格预测模型。

第七章，*使用自编码器进行信用卡欺诈检测*，介绍了一种名为自编码器的降维技术。我们通过使用 TensorFlow 和 Keras 构建自编码器，识别信用卡数据集中的欺诈交易。

第八章，*使用贝叶斯神经网络生成交通标志分类器的不确定性*，解释了贝叶斯神经网络，它帮助我们量化预测中的不确定性。我们将使用 TensorFlow 构建贝叶斯神经网络，以对德国交通标志进行分类。

第九章，*使用 DiscoGAN 从鞋子图像生成匹配的鞋袋*，介绍了一种新的 GAN 类型——**发现 GAN（DiscoGANs）**。我们了解了它的架构与标准 GAN 的区别，以及它如何应用于风格迁移问题。最后，我们在 TensorFlow 中构建了一个 DiscoGAN 模型，从鞋子图像生成匹配的鞋袋，反之亦然。

第十章，*使用胶囊网络对服装图像进行分类*，实现了一个非常新的图像分类模型——胶囊网络。我们将了解它的架构，并解释在 TensorFlow 中实现时的细节。我们使用 Fashion MNIST 数据集，通过该模型对服装图像进行分类。

第十一章，*使用 TensorFlow 进行优质产品推荐*，介绍了诸如矩阵分解（SVD++）、学习排序以及用于推荐任务的卷积神经网络变体等技术。

第十二章，*使用 TensorFlow 进行大规模目标检测*，探索了 Yahoo 的`TensorFlowOnSpark`框架，用于在 Spark 集群上进行分布式深度学习。然后，我们将`TensorFlowOnSpark`应用于大规模图像数据集，并训练网络进行目标检测。

第十三章，*使用 LSTM 生成书籍脚本*，解释了 LSTM 在生成新文本方面的用途。我们使用 Packt 出版的书籍中的书籍脚本，构建了一个基于 LSTM 的深度学习模型，能够自动生成书籍脚本。

第十四章，*使用深度强化学习玩吃豆人*，解释了如何利用强化学习训练模型玩吃豆人，并在此过程中教你强化学习的相关知识。

第十五章，*接下来是什么？*，介绍了 TensorFlow 生态系统中的其他组件，这些组件对于在生产环境中部署模型非常有用。我们还将学习 AI 在各个行业中的应用，深度学习的局限性，以及 AI 中的伦理问题。

# 为了充分利用本书

为了充分利用本书，请从 GitHub 仓库下载本书代码，并在 Jupyter Notebooks 中练习代码。同时，练习修改作者已经提供的实现。

# 下载示例代码文件

你可以从你的账户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果你是从其他地方购买的本书，可以访问[www.packt.com/support](http://www.packt.com/support)并注册，直接将文件通过电子邮件发送给你。

你可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择 SUPPORT 标签。

1.  点击代码下载和勘误。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用以下最新版本解压或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/TensorFlow-Machine-Learning-Projects`](https://github.com/PacktPublishing/TensorFlow-Machine-Learning-Projects)。如果代码有更新，它将被更新到现有的 GitHub 仓库中。

我们还提供了来自丰富书籍和视频目录的其他代码包，访问网址为**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快来查看吧！

# 下载彩色图片

我们还提供了一份 PDF 文件，包含本书中使用的屏幕截图/图表的彩色图片。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789132212_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781789132212_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码字、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个示例：“通过定义占位符并将值传递给 `session.run()`。”

一段代码设置如下：

```py
tf.constant(
  value,
  dtype=None,
  shape=None,
  name='const_name',
  verify_shape=False
  )
```

任何命令行输入或输出如下所示：

```py
const1 (x):  Tensor("x:0", shape=(), dtype=int32)
const2 (y):  Tensor("y:0", shape=(), dtype=float32)
const3 (z):  Tensor("z:0", shape=(), dtype=float16)
```

**粗体**：表示新术语、重要单词或屏幕上出现的文字。例如，菜单或对话框中的词语在文本中通常是这样的格式。以下是一个示例：“在提供的框中输入评论并点击 **提交** 以查看模型预测的分数。”

警告或重要提示通常这样显示。

小贴士和技巧通常这样呈现。

# 获取联系方式

我们欢迎读者的反馈。

**一般反馈**：如果你对本书的任何部分有疑问，请在邮件主题中提及书名，并通过电子邮件联系我们，邮箱地址是 `customercare@packtpub.com`。

**勘误**：虽然我们已尽最大努力确保内容的准确性，但错误偶尔会发生。如果你在本书中发现错误，我们将非常感激你能向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择你的书籍，点击“勘误提交表格”链接，并输入相关细节。

**盗版**：如果你在互联网上发现我们作品的任何非法复制品，请提供相关位置地址或网站名称。请通过 `copyright@packt.com` 联系我们，并附上链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣写书或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。当你阅读并使用完本书后，为什么不在你购买书籍的网站上留下评论呢？潜在读者可以参考你的意见来做出购买决策，我们也可以了解你对我们产品的看法，而作者们也能看到你对他们书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
