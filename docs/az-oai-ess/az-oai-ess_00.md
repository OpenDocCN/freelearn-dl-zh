# 前言

在过去的十年里，我有幸参与了一些颠覆性的技术项目，这些技术正在重新定义企业的运营和创新方式。其中，**生成式人工智能**（**GenAI**）作为一种突破性进展，能够以前所未有的速度重塑各个行业。围绕 GenAI 的兴奋之情是显而易见的，全球各地的组织都在竞相理解它的潜力，并有效地加以应用。然而，对于许多人来说，从好奇心到实际实施的旅程充满了复杂性和不确定性。

在我们在微软的职业旅程中，我们与各种团队合作，从小型企业到财富 500 强医疗公司，探索并部署 AI 驱动的解决方案。这段经历揭示了一个反复出现的主题：尽管 GenAI 的潜力巨大，但在将概念理解与实际应用之间架起桥梁方面存在着显著的知识差距。组织往往难以在这个快速发展的领域中找到方向，迫切寻求可行的洞察力，以将愿景转化为现实。

本书正是我们试图弥补这一知识空白。它是一本全面指南，介绍 Azure OpenAI 这一 GenAI 领域领先平台，融合了基础知识和实践技巧。无论你是开发者、数据科学家还是决策者，本书都旨在帮助你了解 Azure OpenAI 的潜力，将其融入工作流，并充分发挥其能力。

本书将带你从生成式 AI 和**大型语言模型**（**LLMs**）的基础知识开始，一直到操作化、隐私、安全性和提示工程等高级话题。书中穿插了实际案例和逐步指南，确保你不仅能获得理论知识，还能掌握实际技能，有效地实施 Azure OpenAI 解决方案。

这个领域的一个独特之处是它的快速发展。在写这本书的过程中，我目睹了新技术在几周内迅速涌现，迫使我们不断更新内容以反映最前沿的技术。尽管如此，GenAI 和 LLMs 的核心原理仍然至关重要，它们构成了下一波创新的基础。

本书中的示例和用例主要利用 Azure OpenAI，考虑到其企业级功能以及与 Azure 服务的无缝集成。然而，书中的概念和技巧同样适用于其他平台，确保无论你选择哪个提供商都能具有实际意义。实现示例使用 Python，提供了一种简单而强大的方式与 AI 模型进行交互。

欢迎加入探索 Azure OpenAI Essentials 的旅程。本书旨在赋予你知识和信心，帮助你在项目中利用 GenAI 的强大功能。希望本书既能成为你的灵感来源，又能作为实践资源，助你迈向这个变革性的领域。

让我们一起建设未来！

# 本书适用人群

本书面向希望利用 Azure OpenAI 实现实际应用的专业人士和爱好者，无论其技术背景如何。它适用于开发人员、数据科学家、AI 工程师和 IT 决策者，帮助他们将 GenAI 集成到工作流或商业战略中。尽管对编程概念，尤其是 Python，具有基本理解将有益，但本书提供了清晰的解释和实践示例，以支持初学者。此外，商业领导者和非技术人员将从概念性概述和实际应用案例中受益，这有助于弥合技术与现实世界影响之间的鸿沟。无论你是首次探索 GenAI，还是希望深入提升自己的专业知识，本书都提供了一种结构化且易于理解的方式来掌握 Azure OpenAI。

# 本书内容涵盖

*第一章*，*大语言模型简介*，介绍了 LLM 的基础概念，包括其架构、关键示例如 ChatGPT，以及它们对商业和日常生活的变革性影响。还涵盖了基础模型的概念，并探讨了 LLM 在解决复杂问题中的各种实际应用。

*第二章*，*Azure OpenAI 基础*，深入介绍了 Azure OpenAI 服务。本章探讨了微软与 OpenAI 的合作关系、可用的模型类型、如何部署这些模型以及定价结构。本章将使你具备有效访问、创建和使用 Azure OpenAI 资源的知识。

*第三章*，*Azure OpenAI 高级主题*，介绍了 Azure OpenAI 的高级功能，如理解上下文窗口、不同的嵌入模型以及集成向量数据库。本章还探索了前沿特性，如 Azure OpenAI On Your Data、多模态模型、功能调用、助理 API、批处理 API 和微调，帮助你构建复杂的 AI 应用。

*第四章*，*开发企业文档问答解决方案*，解释了如何设计和构建查询非结构化企业文档的解决方案。主要内容包括嵌入概念、利用 Azure Cognitive Search，以及将 Azure OpenAI 与 LangChain 和 Semantic Kernel 集成。

*第五章*，*构建联络中心分析解决方案*，解释了如何使用 Azure OpenAI 和 Azure Communication Services 开发联络中心分析平台。本章涵盖了识别挑战、理解技术需求、设计架构以及实施解决方案以增强客户互动。

*第六章*，*从结构化数据库查询数据*，探索 SQL GPT，这是一种通过自然语言简化查询结构化数据的工具。本章还展示了架构设计和创建便于访问数据库洞察的用户界面，帮助各类专业背景的用户。

*第七章*，*代码生成与文档编写*，探讨了 Azure OpenAI 如何帮助学习者和专业人士生成和解释代码。本章还重点介绍了构建生成代码片段和文档的工具，使编程更加便捷和高效。

*第八章*，*使用 Azure OpenAI 创建基础推荐系统*，讲解了如何构建一个由聊天机器人驱动的个性化推荐系统，如电影或产品推荐。本章还将引导您设计和实施一种解决方案，以提升用户体验。

*第九章*，*将文本转化为视频*，揭示了如何利用 Azure OpenAI 和 Azure Cognitive Services 将文本提示转换为视频。您将学习架构设计、从文本生成图像，并整合音频，创建教育或宣传视频。

*第十章*，*使用 Azure OpenAI 助理 API 创建多模态多代理框架*，探讨了如何构建一个系统，其中多个智能代理共同协作完成任务，如图像生成和优化。本章还涉及多代理框架，展示了其在生成 AI（GenAI）应用中的潜力。

*第十一章*，*隐私与安全*，聚焦于通过强大的隐私和安全措施保护您的 Azure OpenAI 应用。主题包括遵守 Azure OpenAI 服务标准、确保数据隐私、利用内容过滤以及实施托管身份认证。此外，还深入探讨了配置虚拟网络、私有端点、数据加密，以及采用负责任的 AI 实践以确保安全和道德的 AI 使用。

*第十二章*，*将 Azure OpenAI 转化为实际应用*，讲解如何有效部署、管理和扩展 Azure OpenAI 服务。内容涵盖了基本的操作实践，如日志记录和监控、了解服务配额和限制、管理配额、配置吞吐量单元，以及实施可扩展的策略，以高效应对日益增长的工作负载。

*第十三章*，*高级提示工程*，探讨了掌握提示工程的艺术，以优化 AI 响应的行为和质量。本章深入探讨了有效提示的元素和策略，比较了提示工程与微调的区别，并介绍了提高大语言模型准确性的技术。它还讨论了重要的注意事项，如缓解提示注入攻击和调整 AI 输出以满足特定要求。

# 为了从本书中获得最大收益

为了充分利用本书，你应该安装或准备好以下前提条件：

+   **笔记本电脑或电脑**：需要一台运行 Windows、Linux 或 macOS 的设备来设置工具并跟随示例和练习。

+   **Python 3.9、3.10 或 3.11**：兼容的 Python 版本对运行书中提供的各种脚本和示例至关重要。你可以从[`www.python.org/downloads`](https://www.python.org/downloads)下载所需版本。

+   **Azure 开发者 CLI (azd)**：Azure 开发者 CLI 通过提供模板、工具和最佳实践，简化了云原生应用程序开发。按照[`learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd?tabs=winget-windows%2Cbrew-mac%2Cscript-linux&pivots=os-windows`](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd?tabs=winget-windows%2Cbrew-mac%2Cscript-linux&pivots=os-windows)上的说明进行安装。

+   **Node.js 14 或更高版本**：Node.js 是处理某些后端操作和运行示例中使用的工具所必需的。确保你已经安装了 14 或更高版本，可以从[`nodejs.org/en/download`](https://nodejs.org/en/download)下载。

+   **Git**：Git 是版本控制和管理源代码仓库所必需的工具。你可以从[`git-scm.com/downloads`](https://git-scm.com/downloads)下载安装 Git。

+   **PowerShell 7 或更高版本 (pwsh)**：PowerShell 用于运行脚本并自动化与 Azure 环境相关的任务。你可以在[`github.com/powershell/powershell`](https://github.com/powershell/powershell)下载最新版本。

+   **Azure 账户**：需要一个 Azure 账户才能探索书中讨论的基于云的示例和服务。如果你是 Azure 新手，可以注册一个免费账户并获得初始积分来开始使用。

+   **具有 Azure OpenAI 服务访问权限的 Azure 订阅**：为了使用 Azure OpenAI 示例，请确保你的 Azure 订阅已启用访问权限。你可以通过提交[`learn.microsoft.com/en-in/legal/cognitive-services/openai/limited-access`](https://learn.microsoft.com/en-in/legal/cognitive-services/openai/limited-access)上的表单来申请访问权限。

这些前提条件将确保你具备必要的工具和环境，以有效跟随内容并实现动手练习。

**如果您使用的是本书的电子版，建议您自己输入代码，或者从本书的 GitHub 仓库获取代码（链接在下一节中提供）。这样做可以帮助您避免与复制和粘贴代码相关的潜在错误。**

# 下载示例代码文件

您可以从 GitHub 下载本书的示例代码文件，链接是 [`github.com/PacktPublishing/Azure-OpenAI-Essentials`](https://github.com/PacktPublishing/Azure-OpenAI-Essentials)。如果代码有更新，它将会在 GitHub 仓库中同步更新。

我们还有其他代码包，您可以在我们的丰富书籍和视频目录中找到，网址是 [`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快去看看吧！

# 使用的约定

本书中使用了许多文本约定。

`文本中的代码`：表示文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。例如：“每个消息对象由一个 `role` 类型（可以是 `system`、`user` 或 `assistant`）和 `content` 组成。”

代码块如下所示：

```py
} response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    messages=
        {"role": "system", "content": "You are a helpful assistant that helps people find information"},
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目将以粗体显示：

```py
The final match of the ICC World Cup 2011 was played at the Wankhede Stadium in Mumbai, India.
```

任何命令行输入或输出如下所示：

```py
!pip install -r requirements.txt
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词语。例如，菜单或对话框中的词汇通常以**粗体**显示。示例：“一旦新资源可用，Azure 门户将通知您。点击**转到资源**以访问新创建的资源。”

提示或重要说明

以这样的形式出现。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何内容有疑问，请通过电子邮件联系我们，邮箱地址是 [customercare@packtpub.com，并在邮件主题中注明书名。

**勘误**：虽然我们已尽最大努力确保内容的准确性，但错误难免。如果您在本书中发现错误，我们将非常感激您向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表格。

**盗版**：如果您在互联网上发现我们作品的任何非法复制版本，请提供该网址或网站名称，我们将不胜感激。请通过 copyright@packt.com 联系我们，并提供链接。

**如果您有兴趣成为作者**：如果您在某个领域具有专业知识，并且有兴趣编写或贡献一本书，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

# 分享您的想法

一旦您阅读完*Azure OpenAI Essentials*，我们非常希望听到您的反馈！请[点击这里直接进入亚马逊评论页面](https://packt.link/r/1-805-12506-0)并分享您的想法。

您的评价对我们以及技术社区非常重要，将帮助我们确保提供优质的内容。

# 下载本书的免费 PDF 副本

感谢您购买本书！

喜欢随时随地阅读，但无法随身携带纸质书籍？

您的电子书购买无法与所选设备兼容吗？

不用担心，现在每本 Packt 书籍都能免费获得一份没有数字版权管理（DRM）的 PDF 版本。

在任何地方、任何设备上阅读。搜索、复制并粘贴您喜爱的技术书籍中的代码，直接应用到您的项目中。

优惠不仅如此，您还可以独享折扣、新闻通讯以及每天送到邮箱的优质免费内容

按照以下简单步骤获取优惠：

1.  扫描二维码或访问以下链接

![](img/B21019_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781805125068`](https://packt.link/free-ebook/9781805125068)

1.  提交您的购买凭证

1.  就是这样！我们会将您的免费 PDF 和其他福利直接发送到您的邮箱

# 第一部分：生成式人工智能和 Azure OpenAI 基础

本部分作为理解**大型语言模型**（**LLMs**）及其企业级应用的全面基础。我们首先介绍 LLMs，探讨它们的快速采用，并为理解其在商业和技术中的广泛影响奠定基础。接着，我们深入讲解 Azure OpenAI 服务的基本概念，详细说明与 OpenAI 的合作关系、模型部署过程以及定价策略。最后，我们深入探讨嵌入模型、多模态能力和微调等高级概念，使您更深入地理解 Azure OpenAI 的强大功能。

本部分包含以下章节：

+   *第一章*，*大型语言模型简介*

+   *第二章*，*Azure OpenAI 基础知识*

+   *第三章*，*Azure OpenAI 高级主题*
