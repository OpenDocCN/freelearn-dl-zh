# 前言

本书将帮助用户学习如何在 C#中开发和编程神经网络，以及如何将这项激动人心且强大的技术添加到自己的应用程序中。我们将使用许多开源包以及定制软件，从简单概念和理论到每个人都可以使用的强大技术，逐步进行。

# 本书面向对象

本书面向 C# .NET 开发者，旨在学习如何将神经网络技术和技巧添加到他们的应用程序中。

# 本书涵盖内容

第一章，*快速回顾*，为你提供神经网络的基本复习。

第二章，*一起构建我们的第一个神经网络*，展示了激活是什么，它们的目的，以及它们如何以视觉形式出现。我们还将展示一个小型 C#应用程序，使用开源包如 Encog、Aforge 和 Accord 来可视化每个部分。

第三章，决策树和随机森林，帮助你理解决策树和随机森林是什么，以及它们如何被使用。

第四章，*人脸和动作检测*，将指导你使用 Accord.Net 机器学习框架连接到你的本地视频录制设备，并捕获摄像头视野内的实时图像。视野内的任何人脸都将随后被跟踪。

第五章，*使用 ConvNetSharp 训练 CNNs*，将专注于如何使用开源包 ConvNetSharp 来训练 CNNs。将通过示例来阐述用户的概念。

第六章，*使用 RNNSharp 训练自动编码器*，将指导你使用开源包 RNNSharp 中的自动编码器来解析和处理各种文本语料库。

第七章，*用 PSO 替换反向传播*，展示了粒子群优化如何替换用于训练神经网络的反向传播等神经网络训练方法。

第八章，*函数优化：如何以及为什么*，介绍了函数优化，这是每个神经网络不可或缺的一部分。

第九章，*寻找最优参数*，将展示你如何使用数值和启发式优化技术轻松找到神经网络函数的最优参数。

第十章，*使用 TensorFlowSharp 进行目标检测*，将向读者介绍开源包 TensorFlowSharp。

第十一章，*使用 CNTK 进行时间序列预测和 LSTM*，将展示你如何使用微软认知工具包（CNTK），以及长短期记忆（LSTM）来完成时间序列预测。

第十二章，*GRUs 与 LSTMs、RNNs 和前馈网络的比较*，讨论了门控循环单元（GRUs），包括它们与其他类型神经网络的比较。

附录 A，*激活函数时间表*，展示了不同的激活函数及其相应的图表。

附录 B，*函数优化参考*，包括不同的优化函数。

# 要充分利用本书

在本书中，我们假设读者具备基本的 C# .NET 软件开发知识和熟悉度，并且知道如何在 Microsoft Visual Studio 中操作。

# 下载示例代码文件

您可以从[www.packt.com](http://www.packt.com)的账户下载本书的示例代码文件。如果您在其他地方购买了本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择 SUPPORT 标签页。

1.  点击 Code Downloads & Errata。

1.  在搜索框中输入书名，并遵循屏幕上的说明。

文件下载完成后，请确保使用最新版本解压缩或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

该书的代码包也托管在 GitHub 上，网址为[`github.com/PacktPublishing/Hands-On-Neural-Network-Programming-with-CSharp`](https://github.com/PacktPublishing/Hands-On-Neural-Network-Programming-with-CSharp)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还有其他来自我们丰富图书和视频目录的代码包，可在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789612011_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789612011_ColorImages.pdf)。

# 代码实战

访问以下链接查看代码运行的视频：[`bit.ly/2DlRfgO`](http://bit.ly/2DlRfgO)。

# 使用的约定

本书使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 昵称。以下是一个示例：“将下载的`WebStorm-10*.dmg`磁盘映像文件作为系统中的另一个磁盘挂载。”

代码块设置如下：

```py
m_va.Copy(vtmp, m_bestVectors[i])
m_va.Sub(vtmp, particlePosition);
m_va.MulRand(vtmp, m_c1);
m_va.Add(m_velocities[i], vtmp);
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
BasicNetworknetwork = EncogUtility.SimpleFeedForward(2, 2, 0, 1, false);
///Create a scoring/fitness object
ICalculateScore score = new TrainingSetScore(trainingSet);
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中会这样显示。以下是一个例子：“从管理面板中选择“系统信息”。”

警告或重要提示会像这样显示。

小贴士和技巧会像这样显示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`给我们发送邮件。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，我们将非常感激您能提供位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。一旦您阅读并使用了这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
