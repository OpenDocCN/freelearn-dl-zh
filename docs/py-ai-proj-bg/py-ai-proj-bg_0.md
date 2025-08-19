# 前言

**人工智能**（**AI**）是各行各业、各个领域中最新的、颠覆性的技术。本书展示了在 Python 中实现的 AI 项目，涵盖了构成 AI 世界的现代技术。

本书从使用流行的 Python 库 scikit-learn 构建第一个预测模型开始。你将理解如何使用有效的机器学习技术（如随机森林和决策树）构建分类器。通过关于鸟类物种预测、学生表现数据分析、歌曲流派识别和垃圾邮件检测的精彩项目，你将学习到智能应用开发的基础知识以及各种算法和技术。你还将通过使用 Keras 库的项目理解深度学习和神经网络的机制。

在本书结束时，你将能够自信地使用 Python 构建自己的 AI 项目，并准备好迎接更高级的内容。

# 本书适合人群

本书适用于那些希望通过易于跟随的项目迈出人工智能领域第一步的 Python 开发者。要求具备基本的 Python 编程知识，以便能灵活操作代码。

# 本书内容

第一章，*构建你自己的预测模型*，介绍了分类和评估技术，接着解释了决策树，并通过一个编码项目构建了一个用于学生表现预测的模型。

第二章，*使用随机森林进行预测*，讲解了随机森林，并通过一个编码项目使用它来对鸟类物种进行分类。

第三章，*评论分类应用*，介绍了文本处理和词袋模型技术。接着，展示了如何使用这一技术构建 YouTube 评论的垃圾邮件检测器。随后，你将学习更复杂的 Word2Vec 模型，并通过一个编码项目实践它，检测产品、餐厅和电影评论中的正面和负面情感。

第四章，*神经网络*，简要介绍了神经网络，接着讲解了前馈神经网络，并展示了一个使用神经网络识别歌曲流派的程序。最后，你将修改早期的垃圾邮件检测器，使其能够使用神经网络。

第五章，*深度学习*，讨论了深度学习和卷积神经网络（CNN）。你将通过两个项目实践卷积神经网络和深度学习。首先，你将构建一个可以识别手写数学符号的系统，然后回顾鸟类物种识别项目，并将实现方式修改为使用深度卷积神经网络，使其准确度大大提升。

# 为了充分利用本书

1.  您需要具备 Python 及其科学计算库的基础知识。

1.  安装 Jupyter Notebook，最好通过 Anaconda 安装。

# 下载示例代码文件

您可以通过您的帐户在 [www.packtpub.com](http://www.packtpub.com) 下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册以直接将文件通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在 [www.packtpub.com](http://www.packtpub.com/support) 上登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用最新版本的工具解压或提取文件夹：

+   Windows 上使用 WinRAR/7-Zip

+   Mac 上使用 Zipeg/iZip/UnRarX

+   Linux 上使用 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址为 [`github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners`](https://github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners)。如果代码有任何更新，它将更新到现有的 GitHub 仓库中。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，您可以在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上查看它们！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含书中使用的截图/图表的彩色图像。您可以在此下载：[`www.packtpub.com/sites/default/files/downloads/PythonArtificialIntelligenceProjectsforBeginners_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/PythonArtificialIntelligenceProjectsforBeginners_ColorImages.pdf)。

# 使用的约定

本书中使用了一些文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号名。例如：“`classes.txt` 文件展示了带有鸟类物种名称的类 ID。”

**粗体**：表示新术语、重要词汇或在屏幕上看到的文字。例如，菜单或对话框中的词汇会像这样出现在文本中。

警告或重要提示以此方式出现。

小贴士和技巧以此方式显示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书名。如果您对本书的任何内容有疑问，请通过电子邮件联系我们：`questions@packtpub.com`。

**勘误**：尽管我们已尽一切努力确保内容的准确性，错误偶尔也会发生。如果您在本书中发现错误，请向我们报告，我们将不胜感激。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，我们将不胜感激您提供地址或网站名称。请通过`copyright@packtpub.com`与我们联系，并提供材料链接。

**如果您有意成为作者**：如果您在某个专业领域有专长，并且有意撰写或贡献一本书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。在阅读和使用本书后，为什么不在购买它的网站上留下您的评价呢？潜在读者可以看到并使用您公正的意见来做出购买决策，我们在 Packt 可以了解您对我们产品的看法，我们的作者也可以看到您对他们书籍的反馈。谢谢！

欲了解更多有关 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
