# 前言

深度学习是机器学习的一个子集，基于多层神经网络，能够解决自然语言处理、图像分类等领域中的特别难以处理的大规模问题。本书讲解了技术和分析部分的复杂性，以及如何在 Apache Spark 上实现深度学习解决方案的速度。

本书从解释 Apache Spark 和深度学习的基本原理开始（如何为深度学习设置 Spark、分布式建模的原则以及不同类型的神经网络）。然后，讲解在 Spark 上实现一些深度学习模型，如 CNN、RNN 和 LSTM。读者将通过实践体验所需的步骤，并对所面临的复杂性有一个总体的了解。在本书的过程中，将使用流行的深度学习框架，如 DeepLearning4J（主要）、Keras 和 TensorFlow，来实现和训练分布式模型。

本书的使命如下：

+   创建一本关于实现可扩展、高效的 Scala（在某些情况下也包括 Python）深度学习解决方案的实践指南

+   通过多个代码示例让读者对使用 Spark 感到自信

+   解释如何选择最适合特定深度学习问题或场景的模型

# 本书的目标读者

如果你是 Scala 开发人员、数据科学家或数据分析师，想要学习如何使用 Spark 实现高效的深度学习模型，那么这本书适合你。了解核心机器学习概念并具备一定的 Spark 使用经验将有所帮助。

# 本书内容

第一章，*Apache Spark 生态系统*，提供了 Apache Spark 各模块及其不同部署模式的全面概述。

第二章，*深度学习基础*，介绍了深度学习的基本概念。

第三章，*提取、转换、加载*，介绍了 DL4J 框架，并展示了来自多种来源的训练数据 ETL 示例。

第四章，*流式处理*，展示了使用 Spark 和 DL4J DataVec 进行数据流处理的示例。

第五章，*卷积神经网络*，深入探讨了 CNN 的理论及通过 DL4J 实现模型。

第六章，*循环神经网络*，深入探讨了 RNN 的理论及通过 DL4J 实现模型。

第七章，*在 Spark 中训练神经网络*，解释了如何使用 DL4J 和 Spark 训练 CNN 和 RNN。

第八章，*监控和调试神经网络训练*，讲解了 DL4J 提供的在训练时监控和调整神经网络的功能。

第九章，*解释神经网络输出*，介绍了一些评估模型准确性的技术。

第十章，*在分布式系统上部署*，讲解了在配置 Spark 集群时需要考虑的一些事项，以及在 DL4J 中导入和运行预训练 Python 模型的可能性。

第十一章，*NLP 基础*，介绍了**自然语言处理**（**NLP**）的核心概念。

第十二章，*文本分析与深度学习*，涵盖了通过 DL4J、Keras 和 TensorFlow 进行 NLP 实现的一些示例。

第十三章，*卷积*，讨论了卷积和物体识别策略。

第十四章，*图像分类*，深入讲解了一个端到端图像分类 Web 应用程序的实现。

第十五章，*深度学习的未来*，尝试概述未来深度学习的前景。

# 为了最大程度地利用本书

本书的实践部分需要具备 Scala 编程语言的基础知识。具备机器学习的基本知识也有助于更好地理解深度学习的理论。对于 Apache Spark 的初步知识或经验并不是必要的，因为第一章涵盖了关于 Spark 生态系统的所有内容。只有在理解可以在 DL4J 中导入的 Keras 和 TensorFlow 模型时，需要对 Python 有较好的了解。

为了构建和执行本书中的代码示例，需要 Scala 2.11.x、Java 8、Apache Maven 和你选择的 IDE。

# 下载示例代码文件

你可以通过你的账户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果你在其他地方购买了本书，你可以访问[www.packt.com/support](http://www.packt.com/support)并注册，代码文件将通过邮件直接发送给你。

你可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的以下工具解压或提取文件：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Deep-Learning-with-Apache-Spark`](https://github.com/PacktPublishing/Hands-On-Deep-Learning-with-Apache-Spark)。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还有来自丰富书籍和视频目录的其他代码包，您可以访问 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快去看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，其中包含本书中使用的截图/图表的彩色版本。您可以在此下载：[`www.packtpub.com/sites/default/files/downloads/9781788994613_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788994613_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。示例如下：“将下载的 `WebStorm-10*.dmg` 磁盘镜像文件挂载为系统中的另一个磁盘。”

一段代码的设置如下：

```py
val spark = SparkSession
     .builder
       .appName("StructuredNetworkWordCount")
       .master(master)
       .getOrCreate()
```

当我们希望引起您对某一代码块中特定部分的注意时，相关的行或项会以粗体显示：

```py
-------------------------------------------
 Time: 1527457655000 ms
 -------------------------------------------
 (consumer,1)
 (Yet,1)
 (another,1)
 (message,2)
 (for,1)
 (the,1)
```

任何命令行输入或输出如下所示：

```py
$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties
```

**粗体**：表示一个新术语、重要单词或您在屏幕上看到的词汇。例如，菜单或对话框中的单词会像这样出现在文本中。示例如下：“从管理面板中选择系统信息。”

警告或重要提示以这种方式出现。

提示和技巧以这种方式出现。

# 与我们联系

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在您的邮件主题中注明书名，并通过`customercare@packtpub.com`联系我们。

**勘误**：虽然我们已经尽力确保内容的准确性，但错误还是会发生。如果您在本书中发现错误，我们将不胜感激，恳请您向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并填写相关信息。

**盗版**：如果您在互联网上遇到任何非法的我们作品的复制品，无论其形式如何，我们将不胜感激，若您能提供具体位置或网站名称。请通过`copyright@packt.com`与我们联系，并附上相关链接。

**如果您有兴趣成为作者**：如果您在某个领域有专长并且有兴趣撰写或贡献一本书，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在您购买书籍的网站上留下评价呢？潜在读者可以看到并利用您的公正意见来做出购买决策，Packt 可以了解您对我们产品的看法，作者也能看到您对他们书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
