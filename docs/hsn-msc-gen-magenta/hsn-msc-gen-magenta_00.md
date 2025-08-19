# 前言

由于近年来该领域的进展，机器学习在艺术中的地位日益牢固。Magenta 处于这一创新的前沿。本书提供了关于音乐生成的机器学习模型的实用方法，并演示如何将其集成到现有的音乐制作工作流程中。通过实践例子和理论背景的解释，本书是探索音乐生成的完美起点。

在《**使用 Magenta 进行音乐生成实战**》一书中，你将学习如何使用 Magenta 中的模型生成打击乐序列、单音和多音旋律的 MIDI 文件以及原始音频中的乐器音效。我们将看到大量的实践例子，并深入解释机器学习模型，如**递归神经网络**（**RNNs**）、**变分自编码器**（**VAEs**）和**生成对抗网络**（**GANs**）。借助这些知识，我们将创建并训练自己的模型，处理高级音乐生成的用例，并准备新的数据集。最后，我们还将探讨如何将 Magenta 与其他技术集成，如**数字音频工作站**（**DAWs**），以及使用 Magenta.js 在浏览器中分发音乐生成应用程序。

到本书结束时，你将掌握 Magenta 的所有功能，并具备足够的知识来用自己的风格进行音乐生成。

# 本书适合谁阅读

本书将吸引既有技术倾向的艺术家，也有音乐倾向的计算机科学家。它面向任何想要获取关于构建使用深度学习的生成音乐应用程序的实际知识的读者。它不要求你具备任何音乐或技术上的专业能力，除了对 Python 编程语言的基本知识。

# 本书涵盖内容

第一章，*Magenta 与生成艺术简介*，将向你展示生成音乐的基础知识以及现有的生成艺术。你将了解生成艺术的新技术，如机器学习，以及这些技术如何应用于创作音乐和艺术。还将介绍 Google 的 Magenta 开源研究平台，以及 Google 的开源机器学习平台 TensorFlow，概述其不同部分，并指导本书所需软件的安装。最后，我们将在命令行生成一个简单的 MIDI 文件来完成安装。

第二章，*使用 Drums RNN 生成鼓序列*，将向你展示许多人认为是音乐基础的内容——打击乐。我们将展示 RNN 在音乐生成中的重要性。然后，你将学习如何使用预训练的鼓组模型，通过在命令行窗口和 Python 中直接调用它，来生成鼓序列。我们将介绍不同的模型参数，包括模型的 MIDI 编码，并展示如何解读模型的输出。

第三章，*生成复调旋律*，将展示**长短期记忆**（**LSTM**）网络在生成较长序列中的重要性。我们将学习如何使用一个单旋律的 Magenta 模型——旋律 RNN，它是一个带有回馈和注意力机制的 LSTM 网络。你还将学习使用两个复调模型——复调 RNN 和表演 RNN，它们都是使用特定编码的 LSTM 网络，其中后者支持音符的力度和表达性时间。

第四章，*使用 MusicVAE 进行潜在空间插值*，将展示 VAE 的连续潜在空间在音乐生成中的重要性，并与标准**自编码器**（**AEs**）进行比较。我们将使用 Magenta 中的 MusicVAE 模型，一个分层的递归 VAE，从中采样序列，然后在它们之间进行插值，平滑地从一个序列过渡到另一个序列。接着，我们将看到如何使用 GrooVAE 模型为现有的序列添加节奏感或人性化效果。最后，我们将查看用于构建 VAE 模型的 TensorFlow 代码。

第五章，*使用 NSynth 和 GANSynth 进行音频生成*，将展示音频生成。我们首先提供 WaveNet 的概述，这是一个现有的音频生成模型，尤其在文本转语音应用中高效。在 Magenta 中，我们将使用 NSynth，一个基于 WaveNet 的自编码器模型，来生成小的音频片段，这些片段可以作为伴奏 MIDI 曲谱的乐器。NSynth 还支持音频转换，如缩放、时间拉伸和插值。我们还将使用 GANSynth，一种基于 GAN 的更快速方法。

第六章，*训练数据准备*，将展示为什么训练我们自己的模型至关重要，因为它可以让我们生成特定风格的音乐、生成特定的结构或乐器。构建和准备数据集是训练我们自己模型的第一步。为此，我们首先查看现有的数据集和 API，帮助我们找到有意义的数据。然后，我们为特定风格（舞曲和爵士）构建两个 MIDI 数据集。最后，我们使用数据转换和管道准备 MIDI 文件以进行训练。

第七章，*训练 Magenta 模型*，将展示如何调整超参数，如批量大小、学习率和网络大小，以优化网络性能和训练时间。我们还将展示常见的训练问题，如过拟合和模型无法收敛。一旦模型的训练完成，我们将展示如何使用训练好的模型生成新的序列。最后，我们将展示如何使用 Google Cloud Platform 在云端更快地训练模型。

第八章，*在浏览器中使用 Magenta.js 展示 Magenta*，将展示 Magenta 的 JavaScript 实现，Magenta 因其易用性而广受欢迎，因为它运行在浏览器中，并且可以作为网页共享。我们将介绍 Magenta.js 所依赖的技术 TensorFlow.js，并展示 Magenta.js 中可用的模型，包括如何转换我们之前训练过的模型。接着，我们将使用 GANSynth 和 MusicVAE 创建小型 Web 应用，分别用于音频和序列的采样。最后，我们将看到 Magenta.js 如何与其他应用互动，使用 Web MIDI API 和 Node.js。

第九章，*使 Magenta 与音乐应用互动*，将展示 Magenta 如何在更广阔的背景下运作，展示如何使其与其他音乐应用（如数字音频工作站（DAW）和合成器）互动。我们将解释如何通过 MIDI 接口将 MIDI 序列从 Magenta 发送到 FluidSynth 和 DAW。通过这样做，我们将学习如何在所有平台上处理 MIDI 端口，以及如何在 Magenta 中循环 MIDI 序列。我们将展示如何使用 MIDI 时钟和传输信息来同步多个应用。最后，我们将介绍 Magenta Studio，这是一种基于 Magenta.js 的独立打包版本，也可以作为插件集成到 Ableton Live 中。

# 为了充分利用本书

本书不要求具备任何关于音乐或机器学习的特定知识，因为我们将在整本书中覆盖这两个主题的所有技术细节。然而，我们假设你具备一定的 Python 编程知识。我们提供的代码都有详细的注释和解释，方便新手使用和理解。

提供的代码和内容适用于所有平台，包括 Linux、macOS 和 Windows。我们将在过程中设置开发环境，因此在开始之前不需要任何特定的设置。如果你已经在使用**集成开发环境**（**IDE**）和 DAW，你可以在本书的学习过程中继续使用它们。

# 下载示例代码文件

你可以从[www.packt.com](http://www.packt.com)上的账户下载本书的示例代码文件。如果你是在其他地方购买的本书，可以访问[www.packtpub.com/support](https://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给你。

您可以按照以下步骤下载代码文件：

1.  在[www.packt.com](http://www.packt.com)登录或注册。

1.  选择“支持”标签。

1.  点击“代码下载”。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

下载文件后，请确保使用最新版本的以下工具解压或提取文件夹：

+   Windows 版本的 WinRAR/7-Zip

+   Mac 版本的 Zipeg/iZip/UnRarX

+   Linux 版本的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址是[`github.com/PacktPublishing/Hands-On-Music-Generation-with-Magenta`](https://github.com/PacktPublishing/Hands-On-Music-Generation-with-Magenta)。如果代码有任何更新，GitHub 上的现有仓库将进行更新。

我们还提供了来自我们丰富书籍和视频目录中的其他代码包，您可以访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**进行查看！

# 下载彩色图片

我们还提供了一个 PDF 文件，包含本书中使用的屏幕截图/图表的彩色图像。您可以在此下载：[`www.packtpub.com/sites/default/files/downloads/9781838824419_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781838824419_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。举个例子：“每次调用 step 操作时，RNN 需要更新其状态，即隐藏向量`h`。”

代码块格式如下所示：

```py
import os
import magenta.music as mm

mm.notebook_utils.download_bundle("drum_kit_rnn.mag", "bundles")
bundle = mm.sequence_generator_bundle.read_bundle_file(
    os.path.join("bundles", "drum_kit_rnn.mag"))
```

当我们希望您注意代码块中特定部分时，相关的行或项目会以粗体显示：

```py
> drums_rnn_generate --helpfull

    USAGE: drums_rnn_generate [flags]
    ...

magenta.models.drums_rnn.drums_rnn_config_flags:
    ...

magenta.models.drums_rnn.drums_rnn_generate:
    ...
```

任何命令行输入或输出格式如下所示：

```py
> drums_rnn_generate --bundle_file=bundles/drum_kit_rnn.mag --output_dir output
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词汇。例如，菜单或对话框中的词汇会以这种方式出现在文本中。举个例子：“坚持使用**bar**的主要原因是为了遵循 Magenta 的代码约定，其中 bar 比**measure**更为一致地使用。”

警告或重要提示将以这种方式显示。

提示和技巧如下所示。

# 动作中的代码

访问以下链接查看代码运行的视频：

[`bit.ly/2uHplI4`](http://bit.ly/2uHplI4)

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何内容有疑问，请在邮件主题中注明书名，并通过`customercare@packtpub.com`联系我们。

**勘误表**：尽管我们已经尽一切努力确保内容的准确性，但错误还是难免会发生。如果您在本书中发现了错误，我们将不胜感激您能向我们报告。请访问[www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击勘误表提交表格链接，并填写详细信息。

**盗版**：如果您在互联网上发现我们作品的任何形式的非法副本，请您提供其位置地址或网站名称，我们将不胜感激。请通过 `copyright@packt.com` 向我们发送链接。

**如果您有意成为作者**：如果您在某个专题有专业知识，并且有意撰写或为书籍贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。一旦您阅读并使用了本书，为什么不在购买网站上留下您的评论呢？潜在读者可以看到并使用您的客观意见来做出购买决策，我们在 Packt 可以了解您对我们产品的看法，而我们的作者也可以看到您对他们书籍的反馈。谢谢！

欲了解更多 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
