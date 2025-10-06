# 前言

生成式 AI 自从 ChatGPT 发布以来就成为了人们关注的焦点。全球各地的人们对其潜力感到惊叹，各行业也在寻求利用生成式 AI 进行创新和解决商业问题的方法。

2023 年 4 月，亚马逊正式宣布其新的生成式 AI 服务 Amazon Bedrock，该服务简化了生成式 AI 应用的构建和扩展，无需管理基础设施。

本书带你踏上使用 Amazon Bedrock 的生成式 AI 之旅，并赋予你以无缝的方式加速多个生成式 AI 用例的开发和集成。你将探索提示工程、检索增强、微调生成模型以及使用代理编排任务等技术。本书的后半部分涵盖了如何在 Amazon Bedrock 中有效地监控和确保安全和隐私。本书从中级到高级主题逐步展开，并投入了大量努力，使其易于跟随，并辅以实际示例。

在本书结束时，你将深刻理解如何使用 Amazon Bedrock 构建和扩展生成式 AI 应用，并了解几个架构模式和安全性最佳实践，这些将帮助你解决多个商业问题，并能够在你的组织中进行创新。

# 本书面向的对象

本书面向的是通用应用工程师、解决方案工程师和架构师、技术经理、**机器学习**（**ML**）倡导者、数据工程师和数据科学家，他们希望在自己的组织中创新或使用生成式 AI 解决商业用例。你应具备 AWS API 和 ML 核心 AWS 服务的基本理解。

# 本书涵盖的内容

*第一章*，*探索 Amazon Bedrock*，介绍了 Amazon Bedrock，从探索生成式 AI 的格局开始，介绍 Amazon Bedrock 提供的基座模型，选择正确模型的指南，额外的生成式 AI 功能，以及潜在的应用场景。

*第二章*，*访问和使用 Amazon Bedrock 中的模型*，提供了访问和使用 Amazon Bedrock 及其功能的不同方法，涵盖了不同的接口、核心 API、代码片段，Bedrock 与 LangChain 的集成以构建定制管道，多模型链以及关于 Amazon Bedrock 的称为 PartyRock 的游乐场的见解。

*第三章*，*有效模型使用的提示工程*，探讨了提示工程的艺术，其各种技术和构建有效提示以利用 Amazon Bedrock 上生成式 AI 模型力量的最佳实践。它使你对提示工程原则有全面的了解，从而能够设计出能够从 Bedrock 模型中获得预期结果的提示。

*第四章*, *定制模型以提升性能*，提供了使用 Amazon Bedrock 定制**基础模型**的全面指南，以增强其在特定领域应用中的性能。它涵盖了模型定制的理由，数据准备技术，创建定制模型的过程，结果分析以及成功模型定制的最佳实践。

*第五章*, *利用 RAG 的力量*，探讨了**检索增强生成**（**RAG**）方法，该方法通过结合外部数据源来缓解幻觉问题，从而增强语言模型。它深入探讨了 RAG 与 Amazon Bedrock 的集成，包括知识库的实现，并提供了使用 RAG API 和实际场景的动手示例。此外，本章还涵盖了实现 RAG 的替代方法，例如使用 LangChain 编排和其他生成 AI 系统，并讨论了在 RAG 背景下使用 Amazon Bedrock 的当前限制和未来研究方向。

*第六章*, *使用 Amazon Bedrock 生成和总结文本*，深入探讨了架构模式，你将学习如何利用 Amazon Bedrock 的能力生成高质量的文本内容并总结长文档，并探讨了各种实际应用场景。

*第七章*, *构建问答系统和对话界面*，涵盖了在小型和大型文档上进行问答的架构模式，对话记忆，嵌入，提示工程技术和上下文感知技术，以构建智能和吸引人的聊天机器人和问答系统。

*第八章*, *使用 Amazon Bedrock 提取实体和生成代码*，探讨了实体提取在各个领域的应用，提供了使用 Amazon Bedrock 实现它的见解，并研究了代码生成背后的生成 AI 原理和方法，使开发者能够简化工作流程并提高生产力。

*第九章*, *使用 Amazon Bedrock 生成和转换图像*，深入探讨了使用 Amazon Bedrock 上可用的生成 AI 模型进行图像生成的世界。它探讨了图像生成的实际应用，Amazon Bedrock 内可用的多模态模型，多模态系统的设计模式，以及 Amazon Bedrock 提供的伦理考虑和安全保障。

*第十章*, *使用 Amazon Bedrock 开发智能代理*，为您提供了对代理、其优势以及如何利用 LangChain 等工具构建和部署针对 Amazon Bedrock 定制的代理的全面理解，使您能够在实际工业用例中利用生成式 AI 的力量。

*第十一章*, *使用 Amazon Bedrock 评估和监控模型*，提供了如何有效地评估和监控 Amazon Bedrock 的生成式 AI 模型的指导。它涵盖了自动和人工评估方法、用于模型评估的开源工具，以及利用 CloudWatch、CloudTrail 和 EventBridge 等服务进行实时监控、审计和自动化生成式 AI 生命周期的自动化。

*第十二章*, *在 Amazon Bedrock 中确保安全和隐私*，探讨了 Amazon Bedrock 实施的强大安全和隐私措施，确保您的数据得到保护，并使负责任的 AI 实践成为可能。它涵盖了数据本地化、隔离、加密、通过 AWS **身份和访问管理**（**IAM**）进行访问控制，以及实施内容过滤的护栏和防止滥用，以及与安全负责任的 AI 政策保持一致。

# 为了充分利用本书

您需要具备基本的 Python 和 AWS 知识。对生成式 AI 和 ML 工作流程的基本理解将是一个优势。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Python | 基于 Linux 的操作系统 |
| Amazon Web Services |  |
| 基于 Jupyter 的笔记本，例如 Amazon SageMaker |  |

本书要求您拥有访问**Amazon Web Services**（**AWS**）账户的权限。如果您还没有，可以访问[`aws.amazon.com/getting-started/`](https://aws.amazon.com/getting-started/)并创建一个 AWS 账户。

其次，您需要在创建账户后安装和配置 AWS **命令行界面**（**CLI**）([`aws.amazon.com/cli/`](https://aws.amazon.com/cli/))，这将用于从您的本地机器访问 Amazon Bedrock 基础模型。

第三，由于我们将要执行的多数代码单元都是基于 Python 的，因此需要设置 AWS Python SDK（Boto3）([`docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html`](https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html))。您可以通过以下方式执行 Python 设置：在本地机器上安装它，使用 AWS Cloud9，利用 AWS Lambda，或利用 Amazon SageMaker。

**如果您正在使用本书的数字版，我们建议您亲自输入代码或从本书的 GitHub 仓库（下一节中提供链接）获取代码。这样做将帮助您避免与代码复制和粘贴相关的任何潜在错误**。

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件[`github.com/PacktPublishing/Generative-AI-with-Amazon-Bedrock`](https://github.com/PacktPublishing/Generative-AI-with-Amazon-Bedrock)。如果代码有更新，它将在 GitHub 仓库中更新。

我们还有其他来自我们丰富的图书和视频目录的代码包可供在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)获取。查看它们！

# 使用的约定

本书使用了多种文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter/X 用户名。以下是一个示例：“您可以在`ChunkingConfiguration`对象中指定分块策略。”

代码块设置如下：

```py
#import the main packages and libraries
import os
import boto3
import botocore
```

当我们希望您注意代码块中的特定部分时，相关的行或项目将以粗体显示：

```py
Entity Types: Company, Product, Location
```

任何命令行输入或输出都应如下编写：

```py
[Person: Michael Jordan], [Organization: Chicago Bulls], [Location: NBA]
```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词会以粗体显示。以下是一个示例：“在**模型**选项卡中，您可以选择**创建** **微调作业**。”

小贴士或重要提示

它看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请通过 customercare@packtpub.com 给我们发邮件，并在邮件主题中提及书名。

**勘误**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，我们将不胜感激。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)并填写表格。

**盗版**：如果您在互联网上以任何形式发现我们作品的非法副本，我们将不胜感激，如果您能提供位置地址或网站名称，我们将不胜感激。请通过 copyright@packt.com 与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读了《使用 Amazon Bedrock 的生成式 AI》，我们很乐意听听您的想法！请[点击此处直接进入此书的亚马逊评论页面](https://packt.link/r/1803247282)并分享您的反馈。

您的评论对我们和科技社区都非常重要，它将帮助我们确保我们提供高质量的内容。

# 下载本书的免费 PDF 副本

感谢您购买这本书！

您喜欢在路上阅读，但又无法携带您的印刷书籍到处走吗？

您的电子书购买是否与您选择的设备不兼容？

别担心，现在每购买一本 Packt 书籍，您都可以免费获得该书的 DRM 免费 PDF 版本。

在任何地方、任何设备上阅读。从您最喜欢的技术书籍中搜索、复制和粘贴代码到您的应用程序中。

福利远不止于此，您还可以获得独家折扣、时事通讯和每日免费内容的专属访问权限

按照以下简单步骤获取福利：

1.  扫描下面的二维码或访问以下链接

![](img/B22045_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781803247281`](https://packt.link/free-ebook/9781803247281)

1.  提交您的购买证明

1.  就这些！我们将直接将您的免费 PDF 和其他福利发送到您的邮箱

# 第一部分：Amazon Bedrock 基础

本部分建立了有效利用 Amazon Bedrock 的基本原则和实践。我们首先探索 Amazon Bedrock 提供的系列基础模型，提供对其功能和最佳用例的见解。接着，本书进入提示工程的高级技术，这是最大化大型语言模型潜力的关键技能。我们探讨了模型定制的策略，使用户能够根据其特定需求和领域定制这些工具。我们还考察了**RAG**的实施，这是一种通过整合外部知识源显著提升模型性能的尖端方法。

本部分包含以下章节：

+   *第一章*, *探索 Amazon Bedrock*

+   *第二章*, *在 Amazon Bedrock 中访问和使用模型*

+   *第三章*, *有效使用模型的工程提示*

+   *第四章*, *定制模型以提升性能*

+   *第五章*, *利用 RAG 的力量*
