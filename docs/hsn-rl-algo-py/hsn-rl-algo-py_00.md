# 前言

**强化学习**（**RL**）是人工智能的一个热门且前景广阔的分支，涉及到构建更智能的模型和智能体，使其能够根据不断变化的需求自动确定理想行为。《用 Python 实现强化学习算法》将帮助你掌握 RL 算法，并理解它们的实现，帮助你构建自学习的智能体。

本书从介绍在 RL 环境中工作所需的工具、库和设置开始，涵盖了 RL 的基本构建模块，并深入探讨了基于值的方法，如 Q-learning 和 SARSA 算法的应用。你将学习如何将 Q-learning 与神经网络结合使用，解决复杂问题。此外，你将研究策略梯度方法、TRPO 和 PPO，以提高性能和稳定性，然后再深入到 DDPG 和 TD3 的确定性算法。本书还将介绍模仿学习技术的原理，以及如何通过 Dagger 教一个智能体飞行。你将了解进化策略和黑箱优化技术。最后，你将掌握如 UCB 和 UCB1 等探索方法，并开发出一种名为 ESBAS 的元算法。

到本书结尾时，你将通过与关键的强化学习（RL）算法的实践，克服现实世界应用中的挑战，并且你将成为 RL 研究社区的一员。

# 本书适合的读者

如果你是 AI 研究人员、深度学习用户，或任何希望从零开始学习 RL 的人，本书非常适合你。如果你希望了解该领域的最新进展，这本 RL 书也会对你有所帮助。需要具备一定的 Python 基础。

# 本书内容概述

第一章，*强化学习的全景*，为你提供了关于 RL 的深刻洞察。它描述了 RL 擅长解决的问题以及 RL 算法已经应用的领域。还介绍了完成后续章节项目所需的工具、库和设置。

第二章，*实现 RL 循环与 OpenAI Gym*，描述了 RL 算法的主要循环、用于开发算法的工具包，以及不同类型的环境。你将能够通过 OpenAI Gym 接口开发一个随机智能体，使用随机动作玩 CartPole。你还将学习如何使用 OpenAI Gym 接口运行其他环境。

第三章，*通过动态规划解决问题*，向你介绍 RL 的核心思想、术语和方法。你将学习 RL 的主要模块，并对如何构建 RL 算法来解决问题有一个大致的了解。你还将了解基于模型和无模型算法之间的区别，以及强化学习算法的分类。动态规划将被用来解决 FrozenLake 游戏。

第四章，*Q 学习与 SARSA 应用（Q-Learning and SARSA Applications）*，讨论了基于价值的方法，特别是 Q 学习和 SARSA，这两种算法与动态规划不同，并且在大规模问题上具有良好的扩展性。为了深入掌握这些算法，你将把它们应用于 FrozenLake 游戏，并研究它们与动态规划的差异。

第五章，*深度 Q 网络（Deep Q-Networks）*，描述了神经网络，特别是**卷积神经网络（CNNs）**，如何应用于 Q 学习。你将学习为什么 Q 学习和神经网络的结合能产生惊人的结果，并且它的使用能够解决更广泛的问题。此外，你还将利用 OpenAI Gym 接口将 DQN 应用于 Atari 游戏。

第六章，*学习随机优化与策略梯度优化（Learning Stochastic and PG Optimization）*，介绍了一类新的无模型算法：策略梯度方法。你将学习策略梯度方法和基于价值的方法之间的区别，并了解它们的优缺点。接着，你将实现 REINFORCE 和 Actor-Critic 算法，解决一款名为 LunarLander 的新游戏。

第七章，*TRPO 和 PPO 实现（TRPO and PPO Implementation）*，提出了一种修改策略梯度方法的新机制，用以控制策略的改进。这些机制被用来提高策略梯度算法的稳定性和收敛性。特别是，你将学习并实现两种主要的策略梯度方法，这些方法运用了这些新技术，分别是 TRPO 和 PPO。你将通过在 RoboSchool 环境中实现它们，探索具有连续动作空间的环境。

第八章，*DDPG 和 TD3 应用（DDPG and TD3 Applications）*，介绍了一类新的算法——确定性策略算法，这些算法结合了策略梯度和 Q 学习。你将了解其背后的概念，并在一个新的环境中实现 DDPG 和 TD3 这两种深度确定性算法。

第九章，*基于模型的强化学习（Model-Based RL）*，介绍了学习环境模型以规划未来动作或学习策略的强化学习算法。你将学习它们的工作原理、优点，以及为何它们在许多情况下更受青睐。为了掌握它们，你将实现一个基于模型的算法，并在 Roboschool 环境中进行实验。

第十章，*通过 DAgger 算法进行模仿学习（Imitation Learning with the DAgger Algorithm）*，解释了模仿学习如何工作，以及如何将其应用和调整到具体问题上。你将了解最著名的模仿学习算法——DAgger。为了深入理解它，你将通过在 FlappyBird 中实施该算法来加速智能体的学习过程。

第十一章，*理解黑箱优化算法*，探讨了进化算法，这是一类不依赖反向传播的黑箱优化算法。由于其快速的训练速度和易于在数百或数千个核心上并行化，这些算法正在受到越来越多的关注。本章通过特别关注进化策略算法这一进化算法的类型，提供了这些算法的理论和实践背景。

第十二章，*开发 ESBAS 算法*，介绍了强化学习中特有的重要探索-利用困境。通过多臂老虎机问题演示了这一困境，并使用如 UCB 和 UCB1 等方法解决。接着，你将了解算法选择问题，并开发一种名为 ESBAS 的元算法。该算法使用 UCB1 来为每种情况选择最合适的强化学习算法。

第十三章，*解决强化学习挑战的实践实现*，探讨了该领域的主要挑战，并解释了克服这些挑战的一些实践和方法。你还将了解将强化学习应用于现实问题的挑战、深度强化学习的未来发展，以及其对世界的社会影响。

# 要充分利用本书

需要具备一定的 Python 工作知识。了解强化学习及其相关工具也将是有益的。

# 下载示例代码文件

您可以从您的帐户在[www.packt.com](http://www.packt.com)下载本书的示例代码文件。如果您在其他地方购买了此书，您可以访问[www.packtpub.com/support](https://www.packtpub.com/support)并注册，以便将文件直接通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  登录或注册[www.packt.com](http://www.packt.com)。

1.  选择“支持”标签。

1.  点击“代码下载”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用最新版本的工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，链接为[`github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python`](https://github.com/PacktPublishing/Reinforcement-Learning-Algorithms-with-Python)。如果代码有更新，它将被更新到现有的 GitHub 仓库中。

我们还提供了其他来自丰富书籍和视频目录的代码包，您可以在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看。快去看看吧！

# 下载彩色图片

我们还提供了一份包含书中所有截图/图表的彩色图像的 PDF 文件。您可以在此下载：[`www.packtpub.com/sites/default/files/downloads/9781789131116_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789131116_ColorImages.pdf)

# 使用的约定

本书中使用了若干文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。以下是一个示例：“在本书中，我们使用的是 Python 3.7，但 3.5 以上的所有版本应该都可以使用。我们还假设您已经安装了`numpy`和`matplotlib`。”

一段代码块的格式如下：

```py
import gym

# create the environment 
env = gym.make("CartPole-v1")
# reset the environment before starting
env.reset()

# loop 10 times
for i in range(10):
    # take a random action
    env.step(env.action_space.sample())
    # render the game
   env.render()

# close the environment
env.close()
```

所有命令行输入或输出如下所示：

```py
$ git clone https://github.com/pybox2d/pybox2d [](https://github.com/pybox2d/pybox2d) $ cd pybox2d
$ pip install -e .
```

**粗体**：表示新术语、重要词汇或屏幕上看到的词汇。例如，菜单或对话框中的词汇在文中通常是这样呈现的。以下是一个示例：“在**强化学习**（**RL**）中，算法被称为代理，它通过环境提供的数据进行学习。”

警告或重要提示如下所示。

小贴士和技巧如下所示。

# 保持联系

我们非常欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`联系我们。

**勘误表**：尽管我们已经尽力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现了错误，恳请您向我们报告。请访问[www.packtpub.com/support/errata](https://www.packtpub.com/support/errata)，选择您的书籍，点击“勘误提交表格”链接，并填写相关细节。

**盗版**：如果您在互联网上发现任何我们作品的非法复制品，恳请您提供该位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上相关资料的链接。

**如果您有兴趣成为作者**：如果您在某个主题上拥有专业知识，并且有兴趣撰写或参与书籍的编写，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 书评

请留下评论。当您阅读并使用完本书后，不妨在购买平台上留下评论。潜在读者可以通过您的公正评价做出购买决策，我们 Packt 可以了解您的意见，我们的作者也能看到您对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问[packt.com](http://www.packt.com/)。
