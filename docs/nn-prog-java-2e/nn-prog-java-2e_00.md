# 前言

程序员的生活可以描述为一条持续不断的、永无止境的学习之路。程序员总是面临新技术或新方法带来的挑战。通常，在我们的一生中，尽管我们习惯了重复的事物，但我们总是要面对新事物。学习的过程是科学中最有趣的话题之一，有许多尝试去描述或重现人类的学习过程。

本书写作的指导思想是面对新内容并掌握它所带来的挑战。虽然“神经网络”这个名字可能听起来很陌生，甚至可能让人联想到这本书是关于神经学的，但我们努力通过关注您购买这本书的原因来简化这些细微差别。我们旨在构建一个框架，向您展示神经网络实际上很简单，很容易理解，并且完全不需要对该主题有任何先前的知识，才能完全理解我们在这里提出的概念。

因此，我们鼓励您充分利用本书的内容，在解决大问题时领略神经网络的强大力量，但始终以初学者的视角来看待。本书中涉及的所有概念都使用通俗易懂的语言进行解释，并附带技术背景。本书的使命是让您深入了解可以使用简单语言编写的智能应用。

# 本书涵盖内容

[第1章](ch01.xhtml "第1章. 神经网络入门"), *神经网络入门*，介绍了神经网络的概念，并展示了最基本的神经元结构（单层感知器、Adaline），激活函数、权重和学习算法。此外，本章还展示了从零开始到完成创建基本神经网络的Java实现过程。

[第2章](ch02.xhtml "第2章. 让神经网络学习"), *让神经网络学习*，详细介绍了神经网络的学习过程。介绍了有用的概念，如训练、测试和验证。我们展示了如何实现训练和验证算法。本章还展示了错误评估的方法。

[第3章](ch03.xhtml "第3章. 感知器和监督学习"), *感知器和监督学习*，让您熟悉感知器和监督学习特性。我们展示了这些类型神经网络的训练算法。读者将学习如何在Java中实现这些特性。

[第4章](ch04.xhtml "第4章. 自组织映射"), *自组织映射*，介绍了使用自组织映射的无监督学习，即Kohonen神经网络，以及它们的应用，特别是在分类和聚类问题上。

[第5章](ch05.xhtml "第5章. 天气预报")，*天气预报*，涵盖了一个使用神经网络的实用问题，即天气预报。你将得到来自不同地区的历史天气记录的时间序列数据集，并学习在将它们呈现给神经网络之前如何进行预处理。

[第6章](ch06.xhtml "第6章. 疾病诊断分类")，*疾病诊断分类*，介绍了一个分类问题，它也包含在监督学习中。使用患者记录数据，构建了一个神经网络作为专家系统，能够根据患者和症状给出诊断。

[第7章](ch07.xhtml "第7章. 客户画像聚类")，*客户画像聚类*，将教你如何使用神经网络进行聚类，以及应用无监督学习算法来实现这一目标。

[第8章](ch08.xhtml "第8章. 文本识别")，*文本识别*，介绍了另一个涉及神经网络的常见任务，即光学字符识别（OCR），这是一个非常有用且令人印象深刻的任务，真正展示了神经网络强大的学习能力。

[第9章](ch09.xhtml "第9章. 优化和调整神经网络")，*优化和调整神经网络*，展示了帮助优化神经网络的技巧，例如输入选择、更好地将数据集分离为训练、验证和测试，以及数据过滤和选择隐藏神经元数量的选择。

[第10章](ch10.xhtml "第10章. 神经网络当前趋势")，*神经网络当前趋势*，将让你了解神经网络领域的当前前沿状态，使你能够理解和设计新的策略来解决更复杂的问题。

*附录A*，*设置Netbeans开发环境*，本附录展示了读者如何逐步设置Netbeans IDE的开发环境的步骤。

*附录B*，*设置Eclipse开发环境*，本附录展示了读者如果想要使用Eclipse IDE，如何设置开发环境的逐步过程。

这些附录在书中没有提供，但可以从以下链接下载：[https://www.packtpub.com/sites/default/files/downloads/Neural_Network_Programming_with_Java_SecondEdition_Appendices.pdf](https://www.packtpub.com/sites/default/files/downloads/Neural_Network_Programming_with_Java_SecondEdition_Appendices.pdf)

# 你需要这本书的什么

你需要Netbeans ([www.netbeans.org](http://www.netbeans.org)) 或 Eclipse ([www.eclipse.org](http://www.eclipse.org))。这两个都是免费的，可以从它们的网站上下载。

# 这本书是为谁而写的

本书是为想要了解如何利用神经网络的力量开发更智能应用程序的Java开发者而编写的。那些处理大量复杂数据并希望在日常应用程序中有效使用这些数据的人会发现这本书非常有用。预期您有一些统计计算的基本经验。

# 惯例

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter昵称如下所示："层被初始化和计算，以及神经元；它们还实现了`init()`和`calc()`方法"。

代码块设置如下：

```py
public abstract class NeuralLayer {
  protected int numberOfNeuronsInLayer;
  private ArrayList<Neuron> neuron;
  protected IActivationFunction activationFnc;
  protected NeuralLayer previousLayer;
  protected NeuralLayer nextLayer;
  protected ArrayList<Double> input;
  protected ArrayList<Double> output;
  protected int numberOfInputs;
}
```

**新术语**和**重要词汇**以粗体显示。屏幕上看到的单词，例如在菜单或对话框中，在文本中如下所示："选择参数后，通过点击**开始训练**按钮开始训练"。

### 注意

警告或重要注意事项以如下框中的形式出现。

### 小贴士

小技巧和技巧以如下形式出现。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中获得最大收益的标题。

要向我们发送一般反馈，请简单地发送电子邮件至`<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`，并在邮件主题中提及书籍标题。

如果您在某个主题上具有专业知识，并且您对撰写或为书籍做出贡献感兴趣，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经是Packt书籍的骄傲拥有者，我们有一些事情可以帮助您从购买中获得最大收益。

## 下载示例代码

您可以从您的账户中下载本书的示例代码文件[http://www.packtpub.com](http://www.packtpub.com)。如果您在其他地方购买了这本书，您可以访问[http://www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  使用您的电子邮件地址和密码登录或注册我们的网站。

1.  将鼠标指针悬停在顶部的**支持**选项卡上。

1.  点击**代码下载与勘误**。

1.  在**搜索**框中输入书籍名称。

1.  选择您想要下载代码文件的书籍。

1.  从下拉菜单中选择您购买此书籍的地方。

1.  点击**代码下载**。

您也可以通过点击Packt Publishing网站书籍网页上的**代码文件**按钮下载代码文件。您可以通过在**搜索**框中输入书籍名称来访问此页面。请注意，您需要登录到您的Packt账户。

文件下载完成后，请确保您使用最新版本解压缩或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书代码包也托管在GitHub上，网址为 [https://github.com/PacktPublishing/Neural-Network-Programming-with-Java-SecondEdition](https://github.com/PacktPublishing/Neural-Network-Programming-with-Java-SecondEdition)。我们还有其他来自我们丰富图书和视频目录的代码包，可在 [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/) 找到。查看它们吧！

## 勘误

尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在我们的书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这个问题，我们将不胜感激。这样做可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何勘误，请通过访问 [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**勘误提交表单**链接，并输入您的勘误详情来报告它们。一旦您的勘误得到验证，您的提交将被接受，勘误将被上传到我们的网站或添加到该标题的勘误部分下的现有勘误列表中。

要查看之前提交的勘误表，请访问 [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**勘误**部分下。

## 盗版

互联网上版权材料的盗版是一个跨所有媒体的持续问题。在Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现任何形式的我们作品的非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` 联系我们，并提供涉嫌盗版材料的链接。

我们感谢您在保护我们作者和我们为您提供有价值内容的能力方面的帮助。

## 问题

如果您在这本书的任何方面遇到问题，您可以通过 `<[questions@packtpub.com](mailto:questions@packtpub.com)>` 联系我们，我们将尽力解决问题。
