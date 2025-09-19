# 前言

本书深入介绍了**自然语言处理**（**NLP**）技术，从**机器学习**（**ML**）的数学基础开始，逐步深入到高级NLP应用，如**大型语言模型**（**LLMs**）和人工智能应用。作为学习体验的一部分，你将掌握线性代数、优化、概率和统计学，这些对于理解和实现ML和NLP算法至关重要。你还将探索一般的ML技术，了解它们与NLP的关系。在了解如何进行文本分类之前，你将学习文本数据的预处理，包括清洗和准备文本以供分析的方法。本书的后期将讨论LLMs的理论、设计和应用的高级主题，以及NLP的未来趋势，其中将包含对该领域未来的专家意见。为了加强你的实践技能，你还将解决模拟的真实世界NLP业务问题和解决方案。

# 本书面向对象

本书面向技术人士，包括深度学习、ML研究人员、实战NLP实践者以及ML/NLP教育工作者，以及STEM学生。那些将文本作为项目一部分的专业人士和现有的NLP实践者也会在本书中找到大量有用的信息。具备入门级ML知识和基本的Python工作知识将帮助你从本书中获得最佳效果。

# 本书涵盖内容

[*第一章*](B18949_01.xhtml#_idTextAnchor015)，*导航NLP领域：全面介绍*，解释了本书的内容，我们将涵盖哪些主题，以及谁可以使用本书。本章将帮助你决定本书是否适合你。

[*第二章*](B18949_02_split_000.xhtml#_idTextAnchor026)，*掌握机器学习和自然语言处理所需的线性代数、概率和统计学*，分为三个部分。在第一部分，我们将回顾本书不同部分所需的线性代数基础知识。在下一部分，我们将回顾统计学的基础知识，最后，我们将介绍基本的统计估计量。

[*第三章*](B18949_03_split_000.xhtml#_idTextAnchor045)，*释放自然语言处理中的机器学习潜力*，讨论了可用于解决NLP问题的不同ML概念和方法。我们将讨论一般的特征选择和分类技术。我们将涵盖ML问题的一般方面，如训练/测试/验证选择，以及处理不平衡数据集。我们还将讨论用于评估在NLP问题中使用的ML模型的性能指标。我们将解释方法背后的理论，以及如何在代码中使用它们。

[*第4章*](B18949_04.xhtml#_idTextAnchor113), *优化NLP性能的文本预处理技术*，讨论了在现实世界问题背景下的各种文本预处理步骤。我们将根据要解决的问题场景，解释哪些步骤适合哪些需求。本章将展示并审查一个完整的Python管道。

[*第5章*](B18949_05_split_000.xhtml#_idTextAnchor130), *赋予文本分类能力：利用传统机器学习技术*，解释了如何进行文本分类。理论和实现也将被解释。本章将涵盖一个综合的Python笔记本，作为案例研究。

[*第6章*](B18949_06.xhtml#_idTextAnchor209), *重新构想文本分类：深入探究深度学习语言模型*，涵盖了可以使用深度学习神经网络解决的问题。这一类别中的不同问题将向您介绍，以便您学习如何高效地解决它们。这里将解释这些方法的理论，并涵盖一个综合的Python笔记本作为案例研究。

[*第7章*](B18949_07.xhtml#_idTextAnchor365), *揭秘大型语言模型：理论、设计和Langchain实现*，概述了LLM开发和使用的动机，以及它们在创建过程中面临的挑战。通过考察最先进的模型设计，您将深入了解LLM的理论基础和实际应用。

[*第8章*](B18949_08.xhtml#_idTextAnchor440), *利用大型语言模型的力量：高级设置和与RAG的集成*，指导您设置基于API和开源的LLM应用，并深入探讨通过LangChain的提示工程和RAG。我们将通过代码审查实际应用。

[*第9章*](B18949_09.xhtml#_idTextAnchor506), *探索前沿：由LLM驱动的先进应用和创新*，深入探讨使用RAG增强LLM性能，探索高级方法、自动网络源检索、提示压缩、API成本降低和协作多智能体LLM团队，推动当前LLM应用的边界。在这里，您将审查多个Python笔记本，每个笔记本处理不同的高级解决方案以解决实际用例。

[*第10章*](B18949_10.xhtml#_idTextAnchor525), *乘风破浪：分析由LLM和AI塑造的过去、现在和未来趋势*，深入探讨LLM和AI对技术、文化和社会产生的变革性影响，探索关键趋势、计算进步、大型数据集的重要性，以及LLM在商业及其他领域的演变、目的和社会影响。 

[*第11章*](B18949_11.xhtml#_idTextAnchor551)，*独家行业洞察：来自世界级专家的视角和预测*，通过与法律、研究和执行角色中的专家进行对话，深入探讨了未来自然语言处理和大型语言模型趋势，探索了挑战、机遇以及大型语言模型与专业实践和伦理考量的交汇点。

# 要充分利用本书

本书展示的所有代码均以 Jupyter 笔记本的形式呈现。所有代码均使用 Python 3.10.X 开发，并预期在后续版本中也能正常工作。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |

| 通过以下方式之一访问 Python 环境： |

+   通过任何设备上的任何浏览器的免费且易于访问的 Google Colab 访问（推荐）

+   具有安装公共包和访问 OpenAI API 的能力的本地/云 Python 开发环境

| Windows、macOS 或 Linux |
| --- |

| 需要足够的计算资源，如下所示：

+   之前推荐的免费访问 Google Colab 包括免费的 GPU 实例

+   如果选择避免使用 Google Colab，本地/云环境应具有 GPU 以运行几个代码示例

|  |
| --- |

由于本书中的代码示例具有多样化的用例，对于一些高级大型语言模型解决方案，您将需要一个 OpenAI 账户，这将允许您获得 API 密钥。

**如果您使用的是本书的数字版，我们建议您亲自输入代码或从本书的 GitHub 仓库（下一节中提供链接）获取代码。这样做将帮助您避免与代码的复制和粘贴相关的任何潜在错误。**

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件：[https://github.com/PacktPublishing/Mastering-NLP-from-Foundations-to-LLMs](https://github.com/PacktPublishing/Mastering-NLP-from-Foundations-to-LLMs)。如果代码有更新，它将在 GitHub 仓库中更新。

在本书中，我们回顾了代表专业行业级别解决方案的完整代码笔记本：

| **章节** | **笔记本名称** |
| --- | --- |
| 4 | Ch4_Preprocessing_Pipeline.ipynbCh4_NER_and_POS.ipynb |
| 5 | Ch5_Text_Classification_Traditional_ML.ipynb |
| 6 | Ch6_Text_Classification_DL.ipynb |
| 8 | Ch8_Setting_Up_Close_Source_and_Open_Source_LLMs.ipynbCh8_Setting_Up_LangChain_Configurations_and_Pipeline.ipynb |
| 9 | Ch9_Advanced_LangChain_Configurations_and_Pipeline.ipynbCh9_Advanced_Methods_with_Chains.ipynbCh9_Completing_a_Complex_Analysis_with_a_Team_of_LLM_Agents.ipynbCh9_RAGLlamaIndex_Prompt_Compression.ipynbCh9_Retrieve_Content_from_a_YouTube_Video_and_Summarize.ipynb |

我们还有其他来自我们丰富的图书和视频目录的代码包，可在 [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/) 获取。查看它们吧！

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。以下是一个示例：“现在，我们添加了一个实现语法的功能。我们定义了 `output_parser` 变量，并使用不同的函数来生成输出，`predict_and_parse()`。”

代码块设置如下：

```py
import pandas as pd
import matplotlib.pyplot as plt
# Load the record dict from URL
import requests
import pickle
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
qa_engineer (to manager_0):
exitcode: 0 (execution succeeded)
Code output:
Figure(640x480)
programmer (to manager_0):
TERMINATE
```

**粗体**：表示新术语、重要单词或屏幕上出现的单词。例如，菜单或对话框中的单词以 **粗体** 显示。以下是一个示例：“虽然我们选择了一个特定的数据库，但您可以通过参考 **Vector Store** 页面来了解更多关于不同选择的信息。”

小贴士或重要注意事项

它看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请通过电子邮件发送给我们 [customercare@packtpub.com](mailto:customercare@packtpub.com)，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将非常感激您能向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表格。

**盗版**：如果您在互联网上以任何形式遇到我们作品的非法副本，我们将非常感激您能提供位置地址或网站名称。请通过电子邮件发送给我们 [copyright@packt.com](mailto:copyright@packt.com)，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为本书做出贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 评论

请留下评论。一旦您阅读并使用过这本书，为什么不在您购买它的网站上留下评论呢？潜在读者可以查看并使用您的客观意见来做出购买决定，Packt 可以了解您对我们产品的看法，我们的作者也可以看到他们对书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问 [www.packtpub.com](http://www.packtpub.com)。

# 分享您的想法

读完 *Mastering NLP from Foundations to LLMs* 后，我们非常乐意听到您的想法！请 [点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1-804-61918-3) 并分享您的反馈。

您的审阅对我们和科技社区非常重要，并将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 版本

感谢您购买此书！

您喜欢在移动中阅读，但无法携带您的印刷书籍到处走吗？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在每购买一本Packt图书，你都可以免费获得该书的DRM免费PDF版本。

在任何地方、任何设备上阅读。直接从你最喜欢的技术书籍中搜索、复制和粘贴代码到你的应用程序中。

优惠不仅限于此，你还可以获得独家折扣、时事通讯和每日免费内容的每日邮箱访问权限。

按照以下简单步骤获取优惠：

1.  扫描下面的二维码或访问以下链接

![下载此书的免费PDF副本](img/B18949_QR_Free_PDF.jpg)

[https://packt.link/free-ebook/978-1-80461-918-6](https://packt.link/free-ebook/978-1-80461-918-6)

1.  提交你的购买证明

1.  就这样！我们将直接将免费的PDF和其他优惠发送到你的邮箱
