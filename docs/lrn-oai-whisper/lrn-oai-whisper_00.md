# 前言

欢迎来到**自动语音识别**（**ASR**）和 OpenAI 开创性的 Whisper 技术的世界！在这本书中，*学习 OpenAI Whisper*，我们将进行一场全面的探索之旅，掌握当今最先进的 ASR 系统之一。

OpenAI 的 Whisper 代表了语音识别的重大进步，提供了无与伦比的准确性、多功能性和易用性。无论您是开发人员、研究人员还是爱好者，本书都将为您提供利用 Whisper 并发挥其全部潜力所需的知识和技能。

在本书的各章中，我们将深入探讨 Whisper 的核心概念、基础架构和实际应用。从*第一部分*介绍 ASR 的基础知识和 Whisper 在关键特性上的应用开始，我们将为理解这一前沿技术奠定坚实基础。

在*第二部分*，我们将探讨 Whisper 的架构的复杂细节，包括变压器模型、多任务能力和训练技术。您将获得调整 Whisper 以适应特定领域和语言的实际经验，使您能够根据自己的需求定制模型。

*第三部分*是真正的激动人心之处，因为我们深入探讨了 Whisper 在各种实际应用和使用案例中的广泛应用。从转录服务和语音助手到无障碍功能和高级技术，例如说话者辨识和个性化语音合成，您将学习如何在各个领域利用 Whisper 的能力。

随着您逐步阅读各章节，您将获得技术技能，并深入了解塑造 ASR 和语音技术领域风貌的道德考量和未来趋势。通过本书的学习，您将具备足够的能力来应对这个快速发展领域中的挑战和机遇。

无论您是想增强现有应用程序、开发创新解决方案还是扩展 ASR 知识，*学习 OpenAI Whisper*都是您的综合指南。本书将全面介绍 Whisper 及其应用，确保您全面理解 Whisper 及其应用。准备好开始与 OpenAI Whisper 一起进行令人兴奋的发现、掌握和创新之旅吧！

# 本书的目标读者

*学习 OpenAI Whisper*专为开发人员、数据科学家、研究人员和业务专业人士设计，他们希望深入了解如何利用 OpenAI 的 Whisper 进行 ASR 任务。

本书的目标读者主要包括以下三个角色：

+   **ASR 爱好者**：对探索先进语音识别技术潜力充满热情，并希望了解该领域最新发展的个人

+   **开发人员和数据科学家**：希望将 Whisper 集成到其项目中、利用语音识别能力增强现有应用程序或从头开始构建新解决方案的专业人士

+   **研究人员和学术人员**：有意研究 Whisper 内部机制、进行实验并推动 ASR 技术边界的学术界或研究机构的个人。

在本书的整个过程中，读者将学习如何设置 Whisper，为特定领域和语言进行微调，并将其应用于现实场景。读者将全面理解 Whisper 的架构、特性，并掌握其有效实施的最佳实践。

# 本书涵盖内容

*第一章*，*揭开 Whisper 的面纱 – 介绍 OpenAI 的 Whisper*，概述了 Whisper 的主要功能和特性，帮助读者掌握其核心功能。你还将亲手操作初始设置和基本使用示例。

*第二章*，*理解 Whisper 的核心机制*，深入探讨了 Whisper 的 ASR 系统的原理。它解释了系统的关键组件和功能，阐明了该技术如何解读和处理人类语言。

*第三章**，深入探索架构*，全面解释了 OpenAI Whisper 的核心——变换器模型。你将探索 Whisper 架构的复杂性，包括编码器-解码器机制，学习变换器模型如何驱动高效的语音识别。

*第四章**，为领域和语言特定需求微调 Whisper*，带领读者亲自实践，为特定领域和语言需求微调 OpenAI 的 Whisper 模型。读者将学习如何设置一个强大的 Python 环境，整合多样的数据集，并根据目标应用调整 Whisper 的预测，同时确保在各类人群中均衡的表现。

*第五章**，在不同场景中应用 Whisper*，探讨了 OpenAI 的 Whisper 在将语音转化为文字方面的卓越能力，包括转录服务、语音助手、聊天机器人和辅助功能等应用。

*第六章**，扩展 Whisper 应用*，探讨了将 OpenAI 的 Whisper 应用扩展到诸如精确的多语言转录、为提升可发现性而建立内容索引、以及将转录用于 SEO 和内容营销等任务*。

*第七章**，探索先进的语音功能*，深入介绍了提升 OpenAI Whisper 性能的高级技术，例如量化，并探讨了其在实时语音识别中的潜力。

*第八章*，使用 WhisperX 和 NVIDIA 的 NeMo 进行发言人分离*，专注于使用 WhisperX 和 NVIDIA 的 NeMo 框架进行发言人分离。你将学习如何整合这些工具，准确地识别并将音频录音中的语音段落归属给不同的说话人。

*第九章**，《利用 Whisper 进行个性化语音合成》* 探讨了如何利用 OpenAI 的 Whisper 进行语音合成，帮助读者创建个性化的语音模型，捕捉目标语音的独特特征。

*第十章**，《使用 Whisper 塑造未来》* 提供了一个面向未来的视角，探讨了自动语音识别（ASR）领域的发展以及 Whisper 的角色。本章深入分析了即将到来的趋势、预期的功能以及语音技术的发展方向。同时，还讨论了伦理问题，提供了全面的视角。

接下来的部分将讨论为了充分利用本书所需的技术要求和设置。它涵盖了软件、硬件和操作系统的前提条件，以及运行代码示例所推荐的环境。此外，本部分还指导你如何访问本书的 GitHub 仓库中的示例代码文件和其他资源。按照这些说明，你将能够为深入 OpenAI 的 Whisper 世界并充分利用书中的实践示例和练习做好准备。

# 为了充分利用本书

在本书的大部分内容中，你只需要一个 Google 账号和互联网连接，就能在 Google Colaboratory (Colab) 中运行 Whisper AI 代码。使用 Colab 的免费版本和 GPU 无需付费订阅。熟悉 Python 的人可以在本地环境中运行这个代码示例，而不必使用 Colab。

| **本书涵盖的软件/硬件** | **操作系统要求** |
| --- | --- |
| Google Colaboratory (Colab) | Windows、macOS 或 Linux 上的 Web 浏览器 |
| Google Drive |
| YouTube |
| RSS |
| GitHub |
| Python |
| Hugging Face |
| Gradio |
| 基础模型：Google 的 gTTS、StableLM、Zephyr 3B – GGUFLlaVA |
| Intel 的 OpenVINO |
| NVIDIA 的 NeMo |
| 麦克风和扬声器 |

Whisper 的小型模型需要至少 12GB 的 GPU 内存。因此，让我们尽量为我们的 Colab 确保一个不错的 GPU！不幸的是，使用 Google Colab 免费版（例如，Tesla T4 16GB）的好 GPU 变得越来越困难。然而，通过 Google Colab Pro，我们应该不会遇到分配 V100 或 P100 GPU 的问题。

**如果你使用的是本书的数字版，我们建议你亲自输入代码或从本书的 GitHub 仓库获取代码（下节提供了链接）。这样可以帮助你避免因复制粘贴代码而产生的潜在错误。**

在*第四章*中微调 Whisper 至少需要一小时。因此，您必须定期监控您在 Colab 中运行的笔记本。有些笔记本实现了一个带有语音录制和音频播放的 Gradio 应用。连接到计算机的麦克风和扬声器可能有助于您体验交互式语音功能。另一种选择是打开 Gradio 在运行时提供的 URL 链接并在手机上查看；您可以通过手机的麦克风来录制声音。

通过满足这些技术要求，您将能够在不同的环境中探索 Whisper，同时享受 Google Colab 提供的流畅体验和 GitHub 上的丰富资源。

# 下载示例代码文件

您可以从 GitHub 上下载本书的示例代码文件，网址是[`github.com/PacktPublishing/Learn-OpenAI-Whisper/`](https://github.com/PacktPublishing/Learn-OpenAI-Whisper/)。如果代码有更新，它将在 GitHub 仓库中进行更新。

我们的丰富书籍和视频目录中还有其他代码包，您可以访问[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)进行查看。

# Code in Action

本书的《Code in Action》视频可以在[`packt.link/gGv9a`](https://packt.link/gGv9a)观看。

# 使用的约定

本书中使用了几种文本约定。

`文本中的代码`：表示文本中的代码字、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如，“用户甚至可以提供如 `.mp4` 之类的视听格式作为输入，因为 Whisper 会提取音频流进行处理。”

代码块的格式如下：

```py
from datasets import load_dataset, DatasetDict
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)
print(common_voice)
```

当我们希望引起您对代码块中特定部分的注意时，相关的行或项目会用粗体显示：

```py
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

任何命令行输入或输出将如下所示：

```py
!pip install --upgrade pip
!pip install --upgrade datasets transformers accelerate soundfile librosa evaluate jiwer tensorboard gradio
```

**粗体**：表示一个新术语、重要的单词或屏幕上出现的单词。例如，菜单或对话框中的单词会以**粗体**显示。示例：“要获取 GPU，请在 Google Colab 的主菜单中点击 **Runtime** | **Change runtime type**，然后将 **Hardware accelerator** 从 **None** 更改为 **GPU**。”

提示或重要说明

如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何部分有疑问，请通过电子邮件联系我们：[customercare@packtpub.com](http://customercare@packtpub.com)，并在邮件主题中提及书名。

**勘误**：虽然我们已尽力确保内容的准确性，但错误总会发生。如果您在本书中发现错误，我们将非常感谢您向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) 并填写表格。

**盗版**：如果你在互联网上发现我们作品的任何非法复制品，我们将非常感激你提供相关地址或网站名称。请通过 copyright@packt.com 与我们联系，并附上该材料的链接。

**如果你有兴趣成为作者**：如果你在某个领域拥有专业知识，并且有兴趣写书或为书籍贡献内容，请访问 authors.packtpub.com。

# 分享你的想法

阅读完*学习 OpenAI Whisper*后，我们很希望听到你的想法！[请点击此处直接前往 Amazon 书评页面并分享你的反馈](https://packt.link/r/1-835-08592-X)。

你的评论对我们和技术社区非常重要，能帮助我们确保提供优质内容。

# 下载这本书的免费 PDF 副本

感谢购买本书！

你喜欢随时阅读，但无法随身携带纸质书籍吗？

你的电子书购买是否与你选择的设备不兼容？

别担心，现在每本 Packt 书籍都可以免费获得无 DRM 限制的 PDF 版本。

随时随地，在任何设备上阅读。从你喜爱的技术书籍中搜索、复制并将代码直接粘贴到你的应用程序中。

福利不止于此，你还可以每天通过邮箱获取独家折扣、新闻通讯和精彩的免费内容。

按照以下简单步骤获得福利：

1.  扫描二维码或访问以下链接

![](img/B21020_QR_Free_PDF.jpg)

[`packt.link/free-ebook/9781835085929`](https://packt.link/free-ebook/9781835085929)

1.  提交你的购买证明。

1.  就这样！我们会将你的免费 PDF 和其他福利直接发送到你的邮箱。

# 第一部分：介绍 OpenAI 的 Whisper

本部分将介绍 OpenAI 的**Whisper**，一项尖端的自动语音识别（ASR）技术。你将了解 Whisper 的基本特性和功能，包括其关键能力和设置过程。这些基础知识将为更深入探讨该技术及其在现实场景中的应用奠定基础。

本部分包括以下章节：

+   *第一章*，*揭开 Whisper 的面纱 - 介绍 OpenAI 的 Whisper*

+   *第二章*，*理解 Whisper 的核心机制*
