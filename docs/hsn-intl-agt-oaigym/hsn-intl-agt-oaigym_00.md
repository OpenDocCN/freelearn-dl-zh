# 前言

本书将指导你实现自己的智能体，解决离散和连续值的顺序决策问题，并提供所有必要的构建块，以便在各种学习环境中开发、调试、训练、可视化、定制和测试你的智能体实现，涵盖从 Mountain Car 和 Cart Pole 问题到 Atari 游戏和 CARLA——一个高级自动驾驶模拟器。

# 本书适用对象

如果你是学生、游戏/机器学习开发者，或者是希望开始构建智能体和算法来解决多种问题的 AI 爱好者，并且希望使用 OpenAI Gym 接口中的学习环境，本书适合你。如果你希望学习如何构建基于深度强化学习的人工智能智能体来解决你感兴趣领域中的问题，你也会觉得这本书有用。虽然本书涵盖了你需要了解的所有基本概念，但如果你有一定的 Python 知识，会让你更好地理解书中的内容。

# 本书涵盖内容

第一章，*智能体和学习环境概述*，该章节使得多个 AI 系统的开发成为可能。它揭示了工具包的重要特性，提供了无限的机会，让你创建自动智能体来解决多种算法任务、游戏和控制任务。到本章结束时，你将能够使用 Python 创建一个 Gym 环境实例。

第二章，*强化学习和深度强化学习*，提供了一个简明的

解释强化学习中的基本术语和概念。本章

将帮助你很好地理解基本的强化学习框架，

开发 AI 智能体。本章还将介绍深度强化学习和

为你提供一种算法能够解决的高级问题类型的概念。

解决这些问题。

第三章，*OpenAI Gym 和深度强化学习入门*，直接开始并准备好你的开发机器/计算机，进行所需的安装和配置，以便使用学习环境以及 PyTorch 来开发深度学习算法。

第四章，*探索 Gym 及其特性*，带你了解 Gym 库中可用的学习环境清单，从环境如何分类和命名的概述开始，这将帮助你从 700 多个学习环境中选择正确的版本和类型。接下来，你将学习如何探索 Gym，测试任何你想要的环境，理解不同环境的接口和描述。

第五章，*实现你的第一个学习智能体 – 解决 Mountain Car 问题*，解释如何使用强化学习实现一个 AI 智能体来解决 Mountain Car 问题。

汽车问题。你将实现代理，训练它，并看到它自主改进。

实现细节将帮助你应用这些概念来开发和训练智能体。

用来解决其他各种任务和/或游戏。

第六章，*使用深度 Q 学习实现智能代理的最优控制*，涵盖了改进 Q 学习的各种方法，包括使用深度神经网络的动作-价值函数近似、经验回放、目标网络，以及用于训练和测试深度强化学习代理的必要工具和构建模块。你将实现一个基于 DQN 的智能代理，以采取最优的离散控制动作，并训练它玩多个 Atari 游戏，观察代理的表现。

第七章，*创建自定义 OpenAI Gym 环境——Carla 驾驶模拟器*，将教你如何将现实问题转化为与 OpenAI Gym 兼容的学习环境。你将学习 Gym 环境的结构，并基于 Carla 模拟器创建一个自定义的学习环境，能够注册到 Gym 并用于训练我们开发的代理。

第八章，*使用深度演员-评论家算法实现智能与自主驾驶代理*，将教你基于策略梯度的强化学习算法的基础，并帮助你直观理解深度 n 步优势演员-评论家算法。然后你将学习如何实现一个超级智能的代理，使其能够在 Carla 模拟器中自主驾驶汽车，使用同步和异步实现的深度 n 步优势演员-评论家算法。

第九章，*探索学习环境的全景——Roboschool，Gym-Retro，StarCraft-II，DeepMindLab*，将带你超越 Gym，展示一套你可以用来训练智能代理的其他成熟学习环境。你将了解并学习使用各种 Roboschool 环境、Gym Retro 环境、广受欢迎的 StarCraft II 环境和 DeepMind Lab 环境。

第十章，*探索学习算法的全景——DDPG（演员-评论家），PPO（策略梯度），Rainbow（基于价值）*，提供了最新深度强化学习算法的洞察，并基于你在本书前几章学到的知识，揭示了它们的基本原理。你将快速理解三种不同类型的深度强化学习算法背后的核心概念，分别是：基于演员-评论家的深度确定性策略梯度（DDPG）算法、基于策略梯度的近端策略优化（PPO）和基于价值的 Rainbow 算法。

# 为了最大化本书的价值

以下内容将是必需的：

+   需要具备一定的 Python 编程基础，以理解语法、模块导入和库安装。

+   一些使用 Linux 或 macOS X 命令行的基础任务经验，例如浏览文件系统和运行 Python 脚本。

# 下载示例代码文件

你可以从[www.packtpub.com](http://www.packtpub.com)账户下载本书的示例代码文件。如果你是在其他地方购买的本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，将文件直接通过电子邮件发送给你。

你可以按照以下步骤下载代码文件：

1.  登录或注册到[www.packtpub.com](http://www.packtpub.com/support)。

1.  选择“支持”标签。

1.  点击“代码下载与勘误”。

1.  在搜索框中输入书名并按照屏幕上的指示操作。

文件下载完成后，请确保使用最新版本的工具解压或提取文件夹：

+   Windows 使用 WinRAR/7-Zip。

+   Mac 使用 Zipeg/iZip/UnRarX。

+   Linux 使用 7-Zip/PeaZip。

本书的代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym`](https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym)。如果代码有更新，将会更新现有的 GitHub 仓库。

我们还在丰富的书籍和视频目录中提供了其他代码包，可以访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看！

# 下载彩色图片

我们还提供了一个 PDF 文件，包含本书中使用的截图/图表的彩色图像。你可以在此下载：[`www.packtpub.com/sites/default/files/downloads/HandsOnIntelligentAgentswithOpenAIGym_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/HandsOnIntelligentAgentswithOpenAIGym_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词语、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 用户名。例如：“将下载的`WebStorm-10*.dmg`磁盘映像文件挂载为系统中的另一个磁盘。”

代码块的设置方式如下：

```py
#!/usr/bin/env python
import gym
env = gym.make("Qbert-v0")
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
```

当我们希望引起你对代码块中特定部分的注意时，相关的行或项目会以粗体显示：

```py
for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
```

任何命令行输入或输出都将如下所示：

```py
$ python get_observation_action_space.py 'MountainCar-v0'
```

**粗体**：表示新术语、重要词汇或在屏幕上显示的词语。例如，菜单或对话框中的词语会以这种方式显示。示例：“从管理面板中选择系统信息。”

警告或重要说明会以这种方式出现。

提示和技巧以这种方式出现。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：通过电子邮件发送至`feedback@packtpub.com`并在邮件主题中注明书名。如果你对本书的任何方面有疑问，请发送邮件至`questions@packtpub.com`。

**勘误**：尽管我们已尽力确保内容的准确性，但难免会有错误。如果您在本书中发现错误，欢迎您向我们反馈。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表格”链接并填写相关信息。

**盗版**：如果您在互联网上发现任何我们作品的非法复制品，若您能提供相关网址或网站名称，我们将不胜感激。请通过`copyright@packtpub.com`联系并提供该资料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且有兴趣写书或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在您购买书籍的网站上留下评论呢？潜在的读者可以通过您的公正意见来做出购买决策，我们 Packt 也能了解您对我们产品的看法，而我们的作者也能看到您对他们书籍的反馈。谢谢！

若想了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
