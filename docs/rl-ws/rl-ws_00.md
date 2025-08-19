# 前言

# 关于本书

各种智能应用，如视频游戏、库存管理软件、仓库机器人和翻译工具，利用**强化学习**（**RL**）做出决策并执行动作，以最大化期望结果的概率。本书将帮助你掌握在机器学习模型中实现强化学习的技术和算法。

从强化学习的介绍开始，你将被引导了解不同的强化学习环境和框架。你将学习如何实现自己的自定义环境，并使用 OpenAI 基线运行强化学习算法。在你探索了经典的强化学习技术，如动态规划、蒙特卡洛方法和时序差分学习后，你将理解在强化学习中何时应用不同的深度学习方法，并进阶到深度 Q 学习。书中甚至会通过在热门视频游戏《Breakout》上使用 DARQN，帮助你理解基于机器的问题解决不同阶段。最后，你将了解在何时使用基于策略的方法来解决强化学习问题。

完成*强化学习工作坊*后，你将掌握利用强化学习解决具有挑战性的机器学习问题所需的知识和技能。

## 读者对象

如果你是数据科学家、机器学习爱好者，或者是想学习从基础到高级的深度强化学习算法的 Python 开发者，本工作坊适合你。需要具备基础的 Python 语言理解。

## 关于各章

*第一章*，*强化学习简介*，将带你了解强化学习，这是机器学习和人工智能中最令人兴奋的领域之一。

*第二章*，*马尔可夫决策过程与贝尔曼方程*，教你关于马尔可夫链、马尔可夫奖励过程和马尔可夫决策过程的知识。你将学习状态值和动作值，并使用贝尔曼方程计算这些量。

*第三章*，*使用 TensorFlow 2 进行深度学习实践*，介绍了 TensorFlow 和 Keras，概述了它们的关键特性、应用及其如何协同工作。

*第四章*，*开始使用 OpenAI 和 TensorFlow 进行强化学习*，带你了解两个流行的 OpenAI 工具，Gym 和 Universe。你将学习如何将这些环境的接口形式化，如何与它们交互，以及如何为特定问题创建自定义环境。

*第五章*，*动态规划*，教你如何使用动态规划解决强化学习中的问题。你将学习策略评估、策略迭代和价值迭代的概念，并学习如何实现它们。

*第六章*，*蒙特卡洛方法*，教你如何实现各种类型的蒙特卡洛方法，包括“首次访问”和“每次访问”技术。你将学习如何使用这些蒙特卡洛方法解决冰湖问题。

*第七章*，*时序差分学习*，为你准备实现 TD(0)、SARSA 和 TD(λ) Q 学习算法，适用于随机和确定性环境。

*第八章*，*多臂老虎机问题*，介绍了流行的多臂老虎机问题，并展示了几种最常用的算法来解决该问题。

*第九章*，*什么是深度 Q 学习？*，向你讲解深度 Q 学习，并介绍一些深度 Q 学习的高级变体实现，如双重深度 Q 学习，使用 PyTorch 实现。

*第十章*，*使用深度循环 Q 网络玩 Atari 游戏*，介绍了 **深度循环 Q 网络** 及其变体。你将通过实际操作，训练强化学习代理程序来玩 Atari 游戏。

*第十一章*，*基于策略的方法进行强化学习*，教你如何实现不同的强化学习策略方法，如策略梯度、深度确定性策略梯度、信任区域策略优化和近端策略优化。

*第十二章*，*进化策略与强化学习*，将进化策略与传统的机器学习方法结合，特别是在神经网络超参数选择方面。你还将识别这些进化方法的局限性。

注意

《强化学习研讨会》的互动版本包含了一个额外章节，*近期进展* 和 *下一步*。本章教授了强化学习算法的新方法，重点探讨进一步探索的领域，如单次学习和可迁移的领域先验。你可以在这里找到互动版本：[courses.packtpub.com](http://courses.packtpub.com)。

## 约定

文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号如下所示：“请记住，一个算法类的实现需要两个特定的方法与 bandit API 进行交互，`decide()` 和 `update()`，后者更简单且已经实现。”

屏幕上显示的单词（例如，菜单或对话框中的内容）也以这种方式出现在文本中：“`DISTRIBUTIONS` 标签提供了模型参数在各个 epoch 之间如何分布的概述。”

一段代码设置如下：

```py
class Greedy:
    def __init__(self, n_arms=2):
        self.n_arms = n_arms
        self.reward_history = [[] for _ in range(n_arms)]
```

新术语和重要单词以此方式显示：“它的架构允许用户在各种硬件上运行，从 CPU 到 **张量处理单元** (**TPUs**)，包括 GPU 以及移动和嵌入式平台。”

## 代码呈现

跨越多行的代码使用反斜杠 (`\`) 进行分割。当代码执行时，Python 会忽略反斜杠，并将下一行的代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                    validation_split=0.2, shuffle=False)
```

注释被添加到代码中，以帮助解释特定的逻辑部分。单行注释使用 `#` 符号表示，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释被三引号包围，如下所示：

```py
"""
Define a seed for the random number generator to ensure the 
result will be reproducible
"""
seed = 1
np.random.seed(seed)
random.set_seed(seed)
```

## 设置你的环境

在我们详细探索本书之前，需要先设置特定的软件和工具。在接下来的部分，我们将展示如何操作。

## 为 Jupyter Notebook 安装 Anaconda

安装 Anaconda 后，你可以使用 Jupyter notebooks。可以按照 [`docs.anaconda.com/anaconda/install/windows/`](https://docs.anaconda.com/anaconda/install/windows/) 上的步骤在 Windows 系统上安装 Anaconda。

对于其他系统，请访问 [`docs.anaconda.com/anaconda/install/`](https://docs.anaconda.com/anaconda/install/) 获取相应的安装指南。

## 安装虚拟环境

通常来说，在安装 Python 模块时使用独立的虚拟环境是个好习惯，以确保不同项目的依赖项不会发生冲突。因此，建议你在执行这些指令之前采用这种方法。

由于我们在这里使用 Anaconda，强烈建议使用基于 conda 的环境管理。请在 Anaconda Prompt 中运行以下命令以创建并激活环境：

```py
conda create --name [insert environment name here]
conda activate [insert environment name here]
```

## 安装 Gym

要安装 Gym，请确保你的系统中已安装 Python 3.5+。你可以通过 `pip` 简单地安装 Gym。按以下代码片段中的步骤在 Anaconda Prompt 中运行代码：

```py
pip install gym
```

你也可以通过直接克隆 Gym Git 仓库来从源代码构建 Gym 安装。当需要修改 Gym 或添加环境时，这种安装方式非常有用。使用以下代码从源代码安装 Gym：

```py
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

运行以下代码以完成 Gym 的完整安装。此安装可能需要你安装其他依赖项，包括`cmake`和最新版本的`pip`：

```py
pip install -e .[all]
```

在*第十一章，基于策略的强化学习方法*中，你将使用 Gym 中提供的 `Box2D` 环境。你可以通过以下命令安装 `Box2D` 环境：

```py
pip install gym "gym[box2d]"
```

## 安装 TensorFlow 2

要安装 TensorFlow 2，请在 Anaconda Prompt 中运行以下命令：

```py
pip install tensorflow
```

如果你正在使用 GPU，可以使用以下命令：

```py
pip install tensorflow-gpu
```

## 安装 PyTorch

可以按照 [`pytorch.org/`](https://pytorch.org/) 上的步骤在 Windows 上安装 PyTorch。

如果你的系统没有 GPU，可以通过在 Anaconda Prompt 中运行以下代码安装 PyTorch 的 CPU 版本：

```py
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## 安装 OpenAI Baselines

可以按照 [`github.com/openai/baselines`](https://github.com/openai/baselines) 上的说明安装 OpenAI Baselines。

下载 OpenAI Baselines 仓库，切换到 TensorFlow 2 分支，然后按照以下步骤安装：

```py
git clone https://github.com/openai/baselines.git
cd baselines
git checkout tf2
pip install -e .
```

我们在*第一章 强化学习简介*和*第四章 与 OpenAI 及 TensorFlow 一起入门*中使用了 OpenAI Baselines 进行强化学习。由于 OpenAI Baselines 使用的 Gym 版本不是最新版本`0.14`，您可能会遇到如下错误：

```py
AttributeError: 'EnvSpec' object has no attribute '_entry_point'
```

解决此 bug 的方法是将`baselines/run.py`中的两个`env.entry_point`属性改回`env._entry_point`。

详细的解决方案请参见[`github.com/openai/baselines/issues/977#issuecomment-518569750`](https://github.com/openai/baselines/issues/977#issuecomment-518569750)。

另外，您也可以使用以下命令来升级该环境中的 Gym 安装：

```py
pip install --upgrade gym
```

## 安装 Pillow

在 Anaconda 提示符中使用以下命令安装 Pillow：

```py
conda install -c anaconda pillow
```

另外，您也可以运行以下命令使用`pip`：

```py
pip install pillow
```

您可以在[`pypi.org/project/Pillow/2.2.1/`](https://pypi.org/project/Pillow/2.2.1/)了解更多关于 Pillow 的信息。

## 安装 Torch

使用以下命令通过`pip`安装`torch`：

```py
pip install torch==0.4.1 -f https://download.pytorch.org/whl/torch_stable.html
```

请注意，您将只在*第十一章 强化学习的基于策略的方法*中使用版本`0.4.1`的`torch`。对于其他章节，您可以通过*安装 PyTorch*部分中的命令恢复到 PyTorch 的更新版本。

## 安装其他库

`pip`在 Anaconda 中是预装的。安装好 Anaconda 后，所有必需的库可以通过`pip`安装，例如，`pip install numpy`。另外，您也可以使用`pip install –r requirements.txt`安装所有必需的库。您可以在[`packt.live/311jlIu`](https://packt.live/311jlIu)找到`requirements.txt`文件。

练习和活动将通过 Jupyter Notebook 执行。Jupyter 是一个 Python 库，可以像其他 Python 库一样安装——也就是使用`pip install jupyter`，但幸运的是，它已经预装在 Anaconda 中。要打开一个 Notebook，只需在终端或命令提示符中运行命令`jupyter notebook`。

## 访问代码文件

您可以在[`packt.live/2V1MwHi`](https://packt.live/2V1MwHi)找到本书的完整代码文件。

我们尽力为所有活动和练习提供互动版本的支持，但我们建议您也进行本地安装，以便在无法使用该支持时可以正常进行。

如果您在安装过程中遇到任何问题或有任何疑问，请通过电子邮件联系我们：`workshops@packt.com`。
