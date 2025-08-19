# 前言

数据的持续增长以及对基于这些数据做出越来越复杂决策的需求，正在带来巨大的障碍，阻碍组织通过传统的分析方法及时获取见解。

为了寻找有意义的价值和见解，深度学习得以发展，深度学习是基于学习多个抽象层次的机器学习算法分支。神经网络作为深度学习的核心，被广泛应用于预测分析、计算机视觉、自然语言处理、时间序列预测以及执行大量其他复杂任务。

至今，大多数深度学习书籍都是以 Python 编写的。然而，本书是为开发者、数据科学家、机器学习从业者和深度学习爱好者设计的，旨在帮助他们利用 Deeplearning4j（一个基于 JVM 的深度学习框架）的强大功能构建强大、稳健和准确的预测模型，并结合其他开源 Java API。

在本书中，你将学习如何使用前馈神经网络、卷积神经网络、递归神经网络、自编码器和因子分解机开发实际的 AI 应用。此外，你还将学习如何在分布式环境下通过 GPU 进行深度学习编程。

完成本书后，你将熟悉机器学习技术，特别是使用 Java 进行深度学习，并能够在研究或商业项目中应用所学知识。总之，本书并非从头到尾逐章阅读。你可以跳到某一章，选择与你所要完成的任务相关的内容，或者是一个激发你兴趣的章节。

祝阅读愉快！

# 本书的读者群体

本书对于希望通过利用基于 JVM 的 Deeplearning4j (DL4J)、Spark、RankSys 以及其他开源库的强大功能来开发实际深度学习项目的开发者、数据科学家、机器学习从业者和深度学习爱好者非常有用。需要具备一定的 Java 基础知识。然而，如果你有一些关于 Spark、DL4J 和基于 Maven 的项目管理的基本经验，将有助于更快速地掌握这些概念。

# 本书内容

第一章，*深度学习入门*，解释了机器学习和人工神经网络作为深度学习核心的一些基本概念。然后简要讨论了现有的和新兴的神经网络架构。接着，介绍了深度学习框架和库的各种功能。随后，展示了如何使用基于 Spark 的多层感知器（MLP）解决泰坦尼克号生还预测问题。最后，讨论了与本项目及深度学习领域相关的一些常见问题。

第二章，*使用递归类型网络进行癌症类型预测*，展示了如何开发一个深度学习应用程序，用于从高维基因表达数据集中进行癌症类型分类。首先，它执行必要的特征工程，以便数据集能够输入到长短期记忆（LSTM）网络中。最后，讨论了一些与该项目和 DL4J 超参数/网络调整相关的常见问题。

第三章，*使用卷积神经网络进行多标签图像分类*，演示了如何在 DL4J 框架上，使用 CNN 处理多标签图像分类问题的端到端项目。它讨论了如何调整超参数以获得更好的分类结果。

第四章，*使用 Word2Vec 和 LSTM 网络进行情感分析*，展示了如何开发一个实际的深度学习项目，将评论文本分类为正面或负面情感。将使用大规模电影评论数据集来训练 LSTM 模型，并且 Word2Vec 将作为神经网络嵌入。最后，展示了其他评论数据集的示例预测。

第五章，*图像分类的迁移学习*，展示了如何开发一个端到端项目，利用预训练的 VGG-16 模型解决猫狗图像分类问题。我们将所有内容整合到一个 Java JFrame 和 JPanel 应用程序中，以便让整个流程更容易理解，进行示例的对象检测。

第六章，*使用 YOLO、JavaCV 和 DL4J 进行实时对象检测*，展示了如何开发一个端到端项目，在视频片段连续播放时，从视频帧中检测对象。预训练的 YOLO v2 模型将作为迁移学习使用，而 JavaCV API 将用于视频帧处理，基于 DL4J 进行开发。

第七章，*使用 LSTM 网络进行股价预测*，展示了如何开发一个实际的股票开盘、收盘、最低、最高价格或交易量预测项目，使用 LSTM 在 DL4J 框架上进行训练。将使用来自实际股市数据集的时间序列来训练 LSTM 模型，并且模型仅预测 1 天后的股价。

第八章，*云端分布式深度学习——使用卷积 LSTM 网络进行视频分类*，展示了如何开发一个端到端项目，使用结合 CNN 和 LSTM 网络在 DL4J 上准确分类大量视频片段（例如 UCF101）。训练将在 Amazon EC2 GPU 计算集群上进行。最终，这个端到端项目可以作为从视频中进行人体活动识别的入门项目。

第九章，使用深度强化学习玩*GridWorld 游戏*，专注于设计一个由批评和奖励驱动的机器学习系统。接着，展示了如何使用 DL4J、RL4J 和神经网络 Q 学习开发一个 GridWorld 游戏，Q 函数由该网络担任。

第十章，*使用因式分解机开发电影推荐系统*，介绍了使用因式分解机开发一个样例项目，用于预测电影的评分和排名。接着，讨论了基于矩阵因式分解和协同过滤的推荐系统的理论背景，然后深入讲解基于 RankSys 库的因式分解机的项目实现。

第十一章，*讨论、当前趋势与展望*，总结了所有内容，讨论了已完成的项目以及一些抽象的收获。然后提供了一些改进建议。此外，还涵盖了其他现实生活中的深度学习项目的扩展指南。

# 为了最大限度地发挥本书的价值

所有示例都已使用 Deeplearning4j 和一些 Java 开源库实现。具体来说，以下 API/工具是必需的：

+   Java/JDK 版本 1.8

+   Spark 版本 2.3.0

+   Spark csv_2.11 版本 1.3.0

+   ND4j 后端版本为 nd4j-cuda-9.0-platform（用于 GPU），否则为 nd4j-native

+   ND4j 版本 >=1.0.0-alpha

+   DL4j 版本 >=1.0.0-alpha

+   Datavec 版本 >=1.0.0-alpha

+   Arbiter 版本 >=1.0.0-alpha

+   Logback 版本 1.2.3

+   JavaCV 平台版本 1.4.1

+   HTTP 客户端版本 4.3.5

+   Jfreechart 1.0.13

+   Jcodec 0.2.3

+   Eclipse Mars 或 Luna（最新版本）或 IntelliJ IDEA

+   Maven Eclipse 插件（2.9 或更高版本）

+   Eclipse 的 Maven 编译插件（2.3.2 或更高版本）

+   Eclipse 的 Maven assembly 插件（2.4.1 或更高版本）

**关于操作系统**：推荐使用 Linux 发行版（包括 Debian、Ubuntu、Fedora、RHEL、CentOS）。具体来说，例如，对于 Ubuntu，建议安装 14.04（LTS）64 位（或更高版本）的完整安装，或使用 VMWare player 12 或 Virtual box。你也可以在 Windows（XP/7/8/10）或 Mac OS X（10.4.7 及以上版本）上运行 Spark 作业。

**关于硬件配置**：需要一台配备 Core i5 处理器、约 100GB 磁盘空间和至少 16GB 内存的机器或服务器。此外，如果你希望在 GPU 上进行训练，还需要安装 Nvidia GPU 驱动程序，并配置 CUDA 和 CuDNN。如果要运行大型作业，需要足够的存储空间（具体取决于你处理的数据集大小），最好至少有 50GB 的空闲磁盘存储（用于独立作业和 SQL 数据仓库）。

# 下载示例代码文件

你可以从你在[www.packtpub.com](http://www.packtpub.com)的账户下载本书的示例代码文件。如果你在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册以直接将文件通过电子邮件发送给你。

您可以按照以下步骤下载代码文件：

1.  登录或注册 [www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”选项卡。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

文件下载完成后，请确保使用最新版本的工具解压或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Java-Deep-Learning-Projects`](https://github.com/PacktPublishing/Java-Deep-Learning-Projects)。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，您可以在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上查看。快去看看吧！

# 下载彩色图片

我们还提供了一个包含本书中使用的屏幕截图/图表的彩色图片的 PDF 文件。您可以在此处下载：[`www.packtpub.com/sites/default/files/downloads/JavaDeepLearningProjects_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/JavaDeepLearningProjects_ColorImages.pdf)。

# 使用的约定

本书中使用了一些文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。例如：“然后，我解压并将每个 `.csv` 文件复制到一个名为 `label` 的文件夹中。”

代码块如下所示：

```py
<properties>
  <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  <java.version>1.8</java.version>
</properties>
```

当我们希望引起您对代码块中特定部分的注意时，相关行或项将以粗体显示：

```py
<properties>
  <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  <java.version>1.8</java.version>
</properties>
```

**粗体**：表示新术语、重要词汇或在屏幕上看到的单词。例如，菜单或对话框中的词汇会以这种方式出现在文本中。示例如下：“我们随后读取并处理图像，生成 PhotoID | 向量地图。”

警告或重要说明将以如下形式显示。

提示和技巧会以如下形式出现。

# 与我们联系

我们始终欢迎读者的反馈。

**一般反馈**：请发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中注明书名。如果您对本书的任何方面有疑问，请通过 `questions@packtpub.com` 与我们联系。

**勘误**：虽然我们已尽一切努力确保内容的准确性，但仍可能会有错误。如果您在本书中发现错误，请向我们报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入详细信息。

**盗版**：如果您在互联网上发现任何形式的非法复制品，我们将非常感激您提供该位置地址或网站名称。请通过 `copyright@packtpub.com` 与我们联系，并附上链接。

**如果您有兴趣成为作者**：如果您在某个领域具有专长，并且有兴趣写作或参与编写一本书，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。阅读并使用本书后，为什么不在您购买本书的网站上留下评论呢？潜在读者可以看到并参考您的公正意见来做出购买决策，我们 Packt 可以了解您对我们产品的看法，作者们也可以看到您对其书籍的反馈。谢谢！

欲了解有关 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/)。
