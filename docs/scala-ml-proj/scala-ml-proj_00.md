# 前言

机器学习通过将数据转化为可操作的智能，已经对学术界和工业界产生了巨大的影响。另一方面，Scala 在过去几年中在数据科学和分析领域的应用稳步增长。本书是为那些具备复杂数值计算背景并希望学习更多实践机器学习应用开发的资料科学家、数据工程师和深度学习爱好者编写的。

所以，如果您精通机器学习概念，并希望通过深入实际应用，利用 Scala 的强大功能扩展您的知识，那么这本书正是您所需要的！通过 11 个完整的项目，您将熟悉如 Spark ML、H2O、Zeppelin、DeepLearning4j 和 MXNet 等流行的机器学习库。

阅读完本书并实践所有项目后，您将能够掌握数值计算、深度学习和函数式编程，执行复杂的数值任务。因此，您可以在生产环境中开发、构建并部署研究和商业项目。

本书并不需要从头到尾阅读。您可以翻阅到您正在尝试实现的目标相关的章节，或者那些激发您兴趣的章节。任何改进的反馈都是欢迎的。

祝阅读愉快！

# 本书的适用人群

如果您想利用 Scala 和开源库（如 Spark ML、Deeplearning4j、H2O、MXNet 和 Zeppelin）的强大功能来理解大数据，那么本书适合您。建议对 Scala 和 Scala Play 框架有较强的理解，基本的机器学习技术知识将是额外的优势。

# 本书涵盖的内容

第一章，*分析保险严重性索赔*，展示了如何使用一些广泛使用的回归技术开发预测模型来分析保险严重性索赔。我们将演示如何将此模型部署到生产环境中。

第二章，*分析与预测电信流失*，使用 Orange Telecoms 流失数据集，其中包含清洗后的客户活动和流失标签，指定客户是否取消了订阅，来开发一个实际的预测模型。

第三章，*基于历史数据和实时数据的高频比特币价格预测*，展示了如何开发一个收集历史数据和实时数据的实际项目。我们预测未来几周、几个月等的比特币价格。此外，我们演示了如何为比特币在线交易生成简单的信号。最后，本章将整个应用程序作为 Web 应用程序，使用 Scala Play 框架进行包装。

第四章，*大规模聚类与种族预测*，使用来自 1000 基因组计划的基因组变异数据，应用 K-means 聚类方法对可扩展的基因组数据进行分析。目的是对种群规模的基因型变异进行聚类。最后，我们训练深度神经网络和随机森林模型来预测种族。

第五章，*自然语言处理中的主题建模——对大规模文本的更好洞察*，展示了如何利用基于 Spark 的 LDA 算法和斯坦福 NLP 开发主题建模应用，处理大规模原始文本。

第六章，*开发基于模型的电影推荐引擎*，展示了如何通过奇异值分解、ALS 和矩阵分解的相互操作，开发一个可扩展的电影推荐引擎。本章将使用电影镜头数据集进行端到端项目。

第七章，*使用 Q 学习和 Scala Play 框架进行期权交易*，在现实的 IBM 股票数据集上应用强化 Q 学习算法，并设计一个由反馈和奖励驱动的机器学习系统。目标是开发一个名为**期权交易**的实际应用。最后，本章将整个应用作为 Web 应用封装，使用 Scala Play 框架。

第八章，*使用深度神经网络进行银行电话营销的客户订阅评估*，是一个端到端项目，展示了如何解决一个名为**客户订阅评估**的现实问题。将使用银行电话营销数据集训练一个 H2O 深度神经网络。最后，本章评估该预测模型的性能。

第九章，*使用自编码器和异常检测进行欺诈分析*，使用自编码器和异常检测技术进行欺诈分析。所用数据集是由 Worldline 与**ULB**（**布鲁塞尔自由大学**）机器学习小组在研究合作期间收集和分析的欺诈检测数据集。

第十章，*使用递归神经网络进行人体活动识别*，包括另一个端到端项目，展示了如何使用名为 LSTM 的 RNN 实现进行人体活动识别，使用智能手机传感器数据集。

第十一章，*使用卷积神经网络进行图像分类*，展示了如何开发预测分析应用，如图像分类，使用卷积神经网络对名为 Yelp 的真实图像数据集进行处理。

# 为了最大限度地利用本书

本书面向开发人员、数据分析师和深度学习爱好者，适合那些对复杂数值计算没有太多背景知识，但希望了解深度学习是什么的人。建议具备扎实的 Scala 编程基础及其函数式编程概念。对 Spark ML、H2O、Zeppelin、DeepLearning4j 和 MXNet 的基本了解及高层次知识将有助于理解本书。此外，假设读者具备基本的构建工具（如 Maven 和 SBT）知识。

所有示例都使用 Scala 在 Ubuntu 16.04 LTS 64 位和 Windows 10 64 位系统上实现。你还需要以下内容（最好是最新版本）：

+   Apache Spark 2.0.0（或更高版本）

+   MXNet、Zeppelin、DeepLearning4j 和 H2O（请参见章节和提供的`pom.xml`文件中的详细信息）

+   Hadoop 2.7（或更高版本）

+   Java（JDK 和 JRE）1.7+/1.8+

+   Scala 2.11.x（或更高版本）

+   Eclipse Mars 或 Luna（最新版本），带有 Maven 插件（2.9+）、Maven 编译插件（2.3.2+）和 Maven 组装插件（2.4.1+）

+   IntelliJ IDE

+   安装 SBT 插件和 Scala Play 框架

需要一台至少配备 Core i3 处理器的计算机，建议使用 Core i5，或者使用 Core i7 以获得最佳效果。然而，多核处理将提供更快的数据处理和可扩展性。对于独立模式，建议至少有 8GB RAM；对于单个虚拟机，使用至少 32GB RAM，对于集群则需要更高配置。你应该有足够的存储空间来运行大型作业（具体取决于你将处理的数据集大小）；最好有至少 50GB 的空闲硬盘存储空间（独立模式和 SQL 数据仓库均适用）。

推荐使用 Linux 发行版（包括 Debian、Ubuntu、Fedora、RHEL、CentOS 等）。更具体地说，例如，对于 Ubuntu，建议使用 14.04（LTS）64 位（或更高版本）的完整安装，VMWare Player 12 或 VirtualBox。你可以在 Windows（XP/7/8/10）或 Mac OS X（10.4.7+）上运行 Spark 作业。

# 下载示例代码文件

你可以从[www.packtpub.com](http://www.packtpub.com)的账户中下载本书的示例代码文件。如果你从其他地方购买了本书，你可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择 SUPPORT 标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

下载文件后，请确保使用以下最新版本的工具解压或提取文件夹：

+   Windows 的 WinRAR/7-Zip

+   Zipeg/iZip/UnRarX（适用于 Mac）

+   Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址是 [`github.com/PacktPublishing/Scala-Machine-Learning-Projects`](https://github.com/PacktPublishing/Scala-Machine-Learning-Projects)。我们还提供了来自我们丰富书籍和视频目录的其他代码包，地址是 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快来看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，里面包含本书中使用的截图/图表的彩色图像。你可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/ScalaMachineLearningProjects_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/ScalaMachineLearningProjects_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如：“将下载的 `WebStorm-10*.dmg` 磁盘镜像文件挂载为系统中的另一个磁盘。”

代码块的设置如下：

```py
val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)
```

Scala 功能代码块如下所示：

```py
 def variantId(genotype: Genotype): String = {
      val name = genotype.getVariant.getContigName
      val start = genotype.getVariant.getStart
      val end = genotype.getVariant.getEnd
      s"$name:$start:$end"
  }
```

当我们希望特别提醒你注意某个代码块的部分内容时，相关的行或项会以粗体显示：

```py
var paramGrid = new ParamGridBuilder()
      .addGrid(dTree.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(dTree.maxBins, 3 :: 5 :: 9 :: 15 :: 23 :: 31 :: Nil)
      .addGrid(dTree.maxDepth, 5 :: 10 :: 15 :: 20 :: 25 :: 30 :: Nil)
      .build()
```

任何命令行输入或输出都按如下方式书写：

```py
$ sudo mkdir Bitcoin
$ cd Bitcoin
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词汇。例如，菜单或对话框中的单词会像这样显示在文本中。示例：“在管理面板中选择系统信息。”

警告或重要提示会像这样显示。

提示和技巧会像这样显示。

# 与我们联系

我们始终欢迎读者的反馈。

**一般反馈**：通过电子邮件 `feedback@packtpub.com` 联系我们，并在邮件主题中注明书名。如果你对本书的任何部分有疑问，请通过 `questions@packtpub.com` 联系我们。

**勘误**：尽管我们已尽力确保内容的准确性，但难免会有错误。如果你发现本书中的错误，请向我们报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择你的书籍，点击“勘误提交表单”链接，输入详细信息。

**盗版**：如果你在互联网上发现任何非法复制的我们的作品，请提供该位置地址或网站名称，我们将不胜感激。请通过 `copyright@packtpub.com` 联系我们，并附上相关材料的链接。

**如果你有兴趣成为一名作者**：如果你对某个领域有专长，并且有兴趣写书或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下评论。在阅读并使用本书后，何不在您购买书籍的网站上留下评论？潜在读者可以看到并参考您的客观意见做出购买决定，我们在 Packt 也能了解您对我们产品的看法，而我们的作者也能看到您对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问[packtpub.com](https://www.packtpub.com/)。
