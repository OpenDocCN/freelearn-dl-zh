# 序言

随着研究与开发的持续进展，**生成对抗网络**（**GANs**）已成为深度学习领域的下一个重要突破。本书重点介绍了 GAN 相对于传统生成模型的主要改进，并通过动手实例向你展示如何最大化 GAN 的优势。

本书将帮助你理解如何利用 PyTorch 工作的 GAN 架构。你将熟悉这个最灵活的深度学习工具包，并用它将想法转化为实际可运行的代码。你将通过样本生成方法将 GAN 模型应用于计算机视觉、多媒体和自然语言处理等领域。

# 本书适合的人群

本书面向那些希望通过 PyTorch 1.0 实现 GAN 模型的机器学习从业人员和深度学习研究人员。你将通过实际案例熟悉最先进的 GAN 架构。掌握 Python 编程语言的基础知识对于理解本书中的概念是必要的。

# 本书内容概述

第一章，*生成对抗网络基础*，利用了 PyTorch 的新特性。你还将学习如何用 NumPy 构建一个简单的 GAN 来生成正弦信号。

第二章，*开始使用 PyTorch 1.3*，介绍了如何安装 CUDA，以便利用 GPU 加速训练和评估。我们还将详细介绍在 Windows 和 Ubuntu 上安装 PyTorch 的步骤，并从源代码构建 PyTorch。

第三章，*模型设计与训练最佳实践*，探讨了模型架构的整体设计以及选择所需卷积操作时需要遵循的步骤。

第四章，*用 PyTorch 构建你的第一个 GAN*，介绍了一种经典且高效的 GAN 模型——DCGAN，用于生成 2D 图像。你还将了解 DCGAN 的架构，并学习如何训练和评估它们。接下来，你将学习如何使用 DCGAN 生成手写数字和人脸，并了解如何利用自编码器进行对抗学习。你还将学习如何高效地组织源代码，以便于调整和扩展。

第五章，*基于标签信息生成图像*，展示了如何使用 CGAN 根据给定标签生成图像，以及如何通过自编码器实现对抗学习。

第六章，*图像到图像的翻译及其应用*，展示了如何使用逐像素标签信息，利用 pix2pix 进行图像到图像的翻译，以及如何使用 pix2pixHD 翻译高分辨率图像。你还将学习如何灵活设计模型架构来实现你的目标，包括生成更大的图像和在不同类型的图像之间转移纹理。

第七章，*使用 GAN 进行图像修复*，向你展示如何使用 SRGAN 进行图像超分辨率，将低分辨率图像生成高分辨率图像，以及如何使用数据预取器加速数据加载，提高 GPU 训练效率。你还将学习如何训练 GAN 模型来进行图像修复，并填补图像中缺失的部分。

第八章，*训练你的 GAN 打破不同模型*，探讨了对抗样本的基本原理以及如何使用**FGSM**（**快速梯度符号法**）攻击并混淆卷积神经网络（CNN）模型。之后，我们将了解如何使用 accimage 库进一步加速图像加载，并训练一个 GAN 模型来生成对抗样本，欺骗图像分类器。

第九章，*从描述文本生成图像*，提供了关于词嵌入的基本知识以及它们在自然语言处理（NLP）领域中的应用。你还将学习如何设计一个文本到图像的生成对抗网络（GAN）模型，根据一段描述文本生成图像。

第十章，*使用 GAN 进行序列合成*，涵盖了自然语言处理领域常用的技术，如循环神经网络（RNN）和长短期记忆网络（LSTM）。你还将学习强化学习的一些基本概念，并了解它与监督学习（如基于 SGD 的 CNN）有何不同。你还将学习如何使用 SEGAN 去除背景噪声并提升语音音频的质量。

第十一章，*使用 GAN 重建 3D 模型*，展示了**计算机图形学**（**CG**）中如何表示 3D 物体。我们还将探讨计算机图形学的基本概念，包括相机和投影矩阵。你将学习如何构建一个具有 3D 卷积的 3D-GAN 模型，并训练它来生成 3D 物体。

# 为了从本书中获得最大收益

你应该具备基本的 Python 和 PyTorch 知识。

# 下载示例代码文件

你可以从 [www.packt.com](http://www.packt.com) 账户下载本书的示例代码文件。如果你是从其他地方购买本书的，可以访问 [www.packtpub.com/support](https://www.packtpub.com/support) 并注册，将文件直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  在 [www.packt.com](http://www.packt.com) 登录或注册。

1.  选择支持标签。

1.  点击代码下载。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

文件下载完成后，请确保使用最新版本的工具解压或提取文件夹：

+   Windows 版 WinRAR/7-Zip

+   Mac 版 Zipeg/iZip/UnRarX

+   Linux 版 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，链接为 [`github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x`](https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x)。如果代码有更新，将会更新至现有的 GitHub 库。

我们还提供来自丰富书籍和视频目录的其他代码包，访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**。快来看看吧！

# 下载彩色图片

我们还提供了包含本书中使用的截图/图表的彩色图片的 PDF 文件。你可以在此下载：[`www.packtpub.com/sites/default/files/downloads/9781789530513_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789530513_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。例如：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

代码块设置如下：

```py
    # Derivative with respect to w3
    d_w3 = np.matmul(np.transpose(self.x2), delta)
    # Derivative with respect to b3
    d_b3 = delta.copy()
```

任何命令行输入或输出如下所示：

```py
$ python -m torch.distributed.launch --nproc_per_node=NUM_GPUS YOUR_SCRIPT.py --YOUR_ARGUMENTS
```

**粗体**：表示一个新术语、重要词汇或你在屏幕上看到的词汇。例如，菜单或对话框中的文字会以这种方式出现在文本中。示例：“从管理面板中选择系统信息。”

警告或重要提示如下所示。

提示和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果你对本书的任何方面有疑问，请在邮件主题中提到书名，并通过`customercare@packtpub.com`联系我们。

**勘误表**：尽管我们已尽最大努力确保内容的准确性，但错误仍然会发生。如果你在本书中发现错误，我们将感激你向我们报告。请访问 [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择你的书籍，点击勘误表提交表单链接，并输入详细信息。

**盗版**：如果你在互联网上遇到我们作品的任何非法复制形式，我们将感激你提供该位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上相关材料的链接。

**如果你有兴趣成为作者**：如果你在某个领域有专业知识，并且有兴趣撰写或参与书籍的编写，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评论。一旦您阅读并使用了本书，为什么不在购买书籍的网站上留下一条评论呢？潜在的读者可以看到并使用您的客观意见来做出购买决策，我们在 Packt 可以了解您对我们产品的看法，我们的作者也可以看到您对他们书籍的反馈。谢谢！

关于 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。
