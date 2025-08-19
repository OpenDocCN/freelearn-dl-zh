# 前言

本书提供了几种不同的**强化学习**（**RL**）算法的总结，包括算法中的理论以及如何使用 Python 和 TensorFlow 编写这些算法。具体来说，本书涉及的算法有 Q 学习、SARSA、DQN、DDPG、A3C、TRPO 和 PPO。这些 RL 算法的应用包括 OpenAI Gym 中的计算机游戏和使用 TORCS 赛车模拟器进行的自动驾驶。

# 本书的目标读者

本书面向**机器学习**（**ML**）从业人员，特别是有兴趣学习强化学习的读者。它将帮助机器学习工程师、数据科学家和研究生。读者应具备基本的机器学习知识，并有 Python 和 TensorFlow 编程经验，以便能够顺利完成本书。

# 本书内容概述

第一章，*强化学习入门*，概述了 RL 的基本概念，如代理、环境以及它们之间的关系。还涉及了奖励函数、折扣奖励、价值函数和优势函数等主题。读者还将熟悉 Bellman 方程、策略性算法和非策略性算法，以及无模型和基于模型的 RL 算法。

第二章，*时序差分学习、SARSA 与 Q 学习*，向读者介绍了时序差分学习、SARSA 和 Q 学习。它还总结了如何在 Python 中编码这些算法，并在两个经典的 RL 问题——GridWorld 和 Cliff Walking——上进行训练和测试。

第三章，*深度 Q 网络*，向读者介绍了本书中的第一个深度 RL 算法——DQN。它还将讨论如何在 Python 和 TensorFlow 中编写此算法。然后，代码将用于训练 RL 代理玩*Atari Breakout*。

第四章，*双重 DQN、对抗性架构与彩虹算法*，在前一章的基础上进行了扩展，介绍了双重 DQN。它还讨论了涉及价值流和优势流的对抗性网络架构。这些扩展将在 Python 和 TensorFlow 中编码，并用于训练 RL 代理玩*Atari Breakout*。最后，将介绍 Google 的多巴胺代码，并用于训练 Rainbow DQN 代理。

第五章，*深度确定性策略梯度*，是本书中的第一个演员-评论家算法，也是第一个用于连续控制的 RL 算法。它向读者介绍了策略梯度，并讨论了如何使用它来训练演员的策略。本章将使用 Python 和 TensorFlow 编写此算法，并用其训练一个代理来解决倒立摆问题。

第六章，*异步方法——A3C 和 A2C*，向读者介绍了 A3C 算法，这是一种异步强化学习算法，其中一个主处理器将更新策略网络，多个工作处理器将使用该网络收集经验样本，这些样本将用于计算策略梯度，并传递给主处理器。本章中还将使用 A3C 训练 RL 代理来玩 OpenAI Gym 中的*CartPole*和*LunarLander*。最后，还简要介绍了 A2C。

第七章，*信赖域策略优化和近端策略优化*，讨论了两种基于策略分布比率的强化学习算法——TRPO 和 PPO。本章还讨论了如何使用 Python 和 TensorFlow 编码 PPO，并用其训练一个 RL 代理来解决 OpenAI Gym 中的 MountainCar 问题。

第八章，*深度强化学习在自动驾驶中的应用*，向读者介绍了 TORCS 赛车模拟器，编写 DDPG 算法来训练一个代理以自主驾驶汽车。本章的代码文件还包括用于相同 TORCS 问题的 PPO 算法，并作为练习提供给读者。

# 为了从本书中获得最大的收获

读者应具备机器学习算法的良好知识，如深度神经网络、卷积神经网络、随机梯度下降法和 Adam 优化。读者还应具备 Python 和 TensorFlow 的实践编程经验。

# 下载示例代码文件

您可以通过您的账户从[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果您在其他地方购买了此书，可以访问[www.packt.com/support](http://www.packt.com/support)，注册后将文件直接通过电子邮件发送给您。

您可以按照以下步骤下载代码文件：

1.  登录或注册 [www.packt.com](http://www.packt.com)。

1.  选择“支持”标签。

1.  点击“代码下载和勘误”。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

下载文件后，请确保使用以下最新版解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，地址为**[`github.com/PacktPublishing/TensorFlow-Reinforcement-Learning-Quick-Start-Guide`](https://github.com/PacktPublishing/TensorFlow-Reinforcement-Learning-Quick-Start-Guide)**。如果代码有更新，将会在现有的 GitHub 仓库中更新。

我们还提供了来自我们丰富书籍和视频目录的其他代码包，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看。赶紧去看看吧！

# 下载彩色图像

我们还提供了一个 PDF 文件，其中包含了本书中使用的带颜色的截图/图表。您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789533583_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789533583_ColorImages.pdf)。

# 使用的约定

本书中使用了许多文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。举个例子：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为您系统中的另一个磁盘。”

一段代码的格式如下：

```py
import numpy as np 
import sys 
import matplotlib.pyplot as plt
```

当我们希望您注意代码块中的特定部分时，相关行或项会以**粗体**显示：

```py
def random_action():
    # a = 0 : top/north
    # a = 1 : right/east
    # a = 2 : bottom/south
    # a = 3 : left/west
    a = np.random.randint(nact)
    return a
```

所有命令行输入或输出如下所示：

```py
sudo apt-get install python-numpy python-scipy python-matplotlib
```

**粗体**：表示新术语、重要单词或您在屏幕上看到的词语。例如，菜单或对话框中的词语会以这种形式出现在文本中。举个例子：“从管理面板中选择系统信息。”

警告或重要说明如下所示。

提示和技巧如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中注明书名，并通过电子邮件联系我们 `customercare@packtpub.com`。

**勘误**：虽然我们已尽力确保内容的准确性，但错误是难免的。如果您在本书中发现错误，我们将感激您向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接并输入相关细节。

**盗版**：如果您在互联网上发现任何非法的我们作品的副本，无论其形式如何，我们将感激您提供该材料的网址或网站名称。请通过`copyright@packt.com`联系我们，并附上相关链接。

**如果您有兴趣成为作者**：如果您对某个领域有专业知识，并且有兴趣写书或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。当您阅读并使用了本书后，为什么不在您购买本书的网站上留下评论呢？潜在读者可以看到并参考您公正的意见来做出购买决定，我们在 Packt 可以了解您对我们产品的看法，而我们的作者也能看到您对他们书籍的反馈。谢谢！

想了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
