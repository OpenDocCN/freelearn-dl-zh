# 前言

TensorFlow 是最受欢迎的机器学习框架之一，最近也广泛应用于深度学习。它提供了一个快速高效的框架，用于训练不同种类的深度学习模型，且具有非常高的准确性。本书是你掌握 TensorFlow 深度学习的指南，通过 12 个实际项目帮助你完成学习。

*TensorFlow 深度学习项目*从设置适合深度学习的 TensorFlow 环境开始。你将学习如何使用 TensorFlow 训练不同类型的深度学习模型，包括 CNN、RNN、LSTM 和生成对抗网络。在此过程中，你将构建端到端的深度学习解决方案，解决图像处理、企业 AI 和自然语言处理等实际问题。你将训练高性能模型，自动生成图像标题、预测股票表现，并创建智能聊天机器人。本书还涵盖了一些高级内容，如推荐系统和强化学习。

到本书结束时，你将掌握深度学习的所有概念及其在 TensorFlow 中的实现，并能够使用 TensorFlow 构建和训练你自己的深度学习模型，解决任何类型的问题。

# 本书适合谁阅读

本书面向数据科学家、机器学习和深度学习从业者，以及那些希望通过测试自身知识和专业技能，打造真实世界智能系统的 AI 爱好者。如果你希望通过实现 TensorFlow 中的实际项目，掌握与深度学习相关的各种概念和算法，那么本书就是你所需要的！

# 本书内容

第一章，*使用 ConvNets 识别交通标志*，展示了如何通过所有必要的预处理步骤从图像中提取合适的特征。对于我们的卷积神经网络，我们将使用使用 matplotlib 生成的简单形状。在我们的图像预处理练习中，我们将使用耶鲁面部数据库。

第二章，*使用目标检测 API 标注图像*，详细介绍了如何构建一个实时目标检测应用，能够使用 TensorFlow 的最新目标检测 API（配备预训练卷积网络，也称为 TensorFlow 检测模型库）和 OpenCV，标注图像、视频和摄像头捕捉到的图像。

第三章，*图像标题生成*，使读者能够学习如何在有无预训练模型的情况下进行标题生成。

第四章，*构建用于条件图像创建的 GANs*，逐步引导你构建选择性 GAN，以重现所需类型的新图像。GAN 将重现的使用数据集将是手写字符（包括 Chars74K 中的数字和字母）。

第五章，*使用 LSTM 进行股票价格预测*，探讨了如何预测单维度信号——股票价格的未来。根据其过去的走势，我们将学习如何使用 LSTM 架构来预测未来，并使我们的预测变得越来越准确。

第六章，*创建和训练机器翻译系统*，展示了如何使用 TensorFlow 创建和训练一个前沿的机器翻译系统。

第七章，*训练并设置一个能够像人类一样讨论的聊天机器人*，告诉您如何从零开始构建一个智能聊天机器人，并如何与它*讨论*。

第八章，*检测重复的 Quora 问题*，讨论了使用 Quora 数据集检测重复问题的方法。当然，这些方法也可以用于其他类似的数据集。

第九章，*构建 TensorFlow 推荐系统*，涵盖了大型应用的实际示例。我们将学习如何在 AWS 上实现云 GPU 计算能力，并提供非常明确的指导。我们还将利用 H2O 的出色 API 进行大规模的深度网络操作。

第十章，*通过强化学习的电子游戏*，详细介绍了一个项目，您将构建一个能够独立玩*Lunar Lander*的 AI。该项目围绕现有的 OpenAI Gym 项目展开，并使用 TensorFlow 进行集成。OpenAI Gym 是一个提供不同游戏环境的项目，用于探索如何使用可以由包括 TensorFlow 神经网络模型在内的算法驱动的 AI 代理。

# 为了从本书中获得最大的收益

本书中涵盖的示例可以在 Windows、Ubuntu 或 Mac 上运行。所有安装说明都已涵盖。您需要具备基本的 Python、机器学习和深度学习知识，并且熟悉 TensorFlow。

# 下载示例代码文件

您可以从您的帐户中下载本书的示例代码文件，访问[www.packtpub.com](http://www.packtpub.com)。如果您在其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  登录或注册到[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择 SUPPORT 标签。

1.  点击代码下载和勘误。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用最新版本的工具解压或提取文件夹：

+   适用于 Windows 的 WinRAR/7-Zip

+   适用于 Mac 的 Zipeg/iZip/UnRarX

+   适用于 Linux 的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，网址是 [`github.com/PacktPublishing/TensorFlow-Deep-Learning-Projects`](https://github.com/PacktPublishing/TensorFlow-Deep-Learning-Projects)。我们还在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上提供了其他来自我们丰富书籍和视频目录的代码包。快去看看吧！

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：指示文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。例如："类 `TqdmUpTo` 只是一个 `tqdm` 包装器，它使得下载也能显示进度条。"

一块代码如下所示：

```py
import numpy as np
import urllib.request
import tarfile
import os
import zipfile
import gzip
import os
from glob import glob
from tqdm import tqdm
```

任何命令行输入或输出如下所示：

```py
epoch 01: precision: 0.064
epoch 02: precision: 0.086
epoch 03: precision: 0.106
epoch 04: precision: 0.127
epoch 05: precision: 0.138
epoch 06: precision: 0.145
epoch 07: precision: 0.150
epoch 08: precision: 0.149
epoch 09: precision: 0.151
epoch 10: precision: 0.152
```

**粗体**：表示新术语、重要单词或在屏幕上看到的单词。例如，菜单或对话框中的单词在文本中会以这种形式出现。示例如下：“从管理面板中选择系统信息。”

警告或重要提示如下所示。

小贴士和技巧如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：发送电子邮件至 `feedback@packtpub.com`，并在邮件主题中提及书名。如果您对本书的任何部分有疑问，请通过电子邮件联系我们，地址是 `questions@packtpub.com`。

**勘误**：尽管我们已尽一切努力确保内容的准确性，但错误仍然会发生。如果您在本书中发现错误，请联系我们并报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表格”链接，并输入详细信息。

**盗版**：如果您在互联网上发现我们作品的任何非法复制形式，我们将非常感激您提供相关位置或网站名称。请通过 `copyright@packtpub.com` 联系我们，并附上链接。

**如果您有意成为作者**：如果您在某个领域有专业知识，并且有兴趣撰写或参与编写书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。在您阅读并使用本书后，不妨在购买本书的网站上留下评论？潜在的读者可以参考并使用您的公正意见来做出购买决策，我们 Packt 可以了解您对我们产品的看法，我们的作者也可以看到您对其书籍的反馈。谢谢！

若想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
