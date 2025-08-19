# 前言

# 关于本书

新的经历可能令人畏惧，但这次不会！这本深度学习初学者指南将帮助你从零开始探索深度学习，使用 Keras 并开始训练你的第一个神经网络。

Keras 与其他深度学习框架的最大不同之处在于其简单性。Keras 拥有超过二十万的用户，在行业和研究界的应用程度超过了任何其他深度学习框架。

*Keras 深度学习工作坊*首先介绍了使用 scikit-learn 包的机器学习基本概念。在学习如何执行构建神经网络所需的线性变换之后，你将使用 Keras 库构建你的第一个神经网络。随着学习的深入，你将学会如何构建多层神经网络，并识别模型是否对训练数据出现欠拟合或过拟合。通过实际操作，你将学习如何使用交叉验证技术评估模型，然后选择最佳超参数以优化模型表现。最后，你将探索递归神经网络，并学习如何训练它们以预测序列数据中的值。

在本书的结尾，你将掌握自信训练自己神经网络模型所需的技能。

## 读者对象

如果你已经了解数据科学和机器学习的基础，并希望开始学习人工神经网络和深度学习等先进的机器学习技术，那么本书非常适合你。为了更有效地掌握本书中解释的深度学习概念，具有 Python 编程经验以及一定的统计学和逻辑回归知识是必需的。

## 关于章节

*第一章*，*使用 Keras 进行机器学习简介*，将通过使用 scikit-learn 包向你介绍机器学习的基本概念。你将学习如何为模型构建准备数据，并使用一个真实世界的数据集训练一个逻辑回归模型。

*第二章*，*机器学习与深度学习的区别*，将介绍传统机器学习算法与深度学习算法之间的区别。你将学习构建神经网络所需的线性变换，并使用 Keras 库构建你的第一个神经网络。

*第三章*，*使用 Keras 进行深度学习*，将扩展你对神经网络构建的知识。你将学习如何构建多层神经网络，并能够识别模型是否对训练数据出现欠拟合或过拟合的情况。

*第四章*，*使用 Keras 包装器进行交叉验证评估模型*，将教你如何使用 Keras 包装器与 scikit-learn 配合，将 Keras 模型纳入 scikit-learn 工作流中。你将应用交叉验证来评估模型，并使用该技术选择最佳的超参数。

*第五章*，*提升模型准确度*，将介绍各种正则化技术，以防止模型过拟合训练数据。你将学习不同的方法来搜索最优的超参数，从而获得最高的模型准确度。

*第六章*，*模型评估*，将演示多种方法来评估你的模型。除了准确度之外，你还将学习更多的模型评估指标，包括灵敏度、特异性、精确度、假阳性率、ROC 曲线和 AUC 分数，以了解你的模型表现如何。

*第七章*，*卷积神经网络的计算机视觉*，将介绍如何使用卷积神经网络构建图像分类器。你将学习卷积神经网络架构的所有组成部分，并构建图像处理应用程序来进行图像分类。

*第八章*，*迁移学习与预训练模型*，将向你介绍如何将一个模型的学习迁移到其他应用中。你将通过使用不同的预训练模型，并稍作修改，以适应不同的应用场景。

*第九章*，*使用循环神经网络的序列建模*，将教你如何构建使用序列数据的模型。你将了解循环神经网络的架构，并学习如何训练它们，以预测序列数据中的后续值。你将通过预测各种股票价格的未来值来测试你的知识。

## 约定

代码中的单词、数据库表名、文件夹名称、文件名、文件扩展名、路径名称、虚拟 URL、用户输入和 Twitter 用户名如下所示：

"`sklearn` 有一个名为 `train_test_split` 的类，它提供了用于拆分数据的功能。"

屏幕上显示的文字，例如菜单或对话框中的内容，也以相同的格式显示。

一段代码如下所示：

```py
# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
```

新术语和重要词汇将以如下方式展示：

"字典包含多个元素，类似于**列表**，但每个元素作为键值对进行组织。"

## 代码展示

跨多行的代码通过反斜杠（ `\` ）进行拆分。执行代码时，Python 会忽略反斜杠，并将下一行的代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                   validation_split=0.2, shuffle=False)
```

注释是添加到代码中以帮助解释特定逻辑的部分。单行注释使用 `#` 符号表示，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释由三个引号括起来，如下所示：

```py
"""
Define a seed for the random number generator to ensure the 
result will be reproducible
"""
seed = 1
np.random.seed(seed)
random.set_seed(seed)
```

## 设置你的开发环境

在我们详细探讨这本书之前，我们需要安装特定的软件和工具。在接下来的部分中，我们将看到如何进行设置。

## 安装 Anaconda

本课程中，我们将使用 Anaconda；它是一个 Python 发行版，内置了包管理器，并预安装了常用于机器学习和科学计算的包。

要安装 Anaconda，请在 [`docs.anaconda.com/anaconda/install/`](https://docs.anaconda.com/anaconda/install/) 的官方安装页面上找到你需要的版本（Windows、macOS 或 Linux）。按照适用于你操作系统的安装说明进行操作。

一旦安装了 Anaconda，你可以通过 Anaconda Navigator 或 Anaconda Prompt 与它进行交互。有关如何使用这些工具的说明，请访问 [`docs.anaconda.com/anaconda/user-guide/getting-started/`](https://docs.anaconda.com/anaconda/user-guide/getting-started/)。

要验证安装是否正确，可以在 CMD / Terminal 中执行 `anaconda-navigator` 命令。如果安装正确，这将打开 Anaconda Navigator。

## 安装库

`pip` 已预安装在 Anaconda 中。一旦 Anaconda 安装在你的计算机上，所有必需的库都可以通过 `pip` 安装，例如，`pip install numpy`。或者，你可以使用 `pip install –r requirements.txt` 安装所有必需的库。你可以在 [`packt.live/3hhZ2v9`](https://packt.live/3hhZ2v9) 找到 `requirements.txt` 文件。

练习和活动将在 Jupyter Notebooks 中执行。Jupyter 是一个 Python 库，可以像安装其他 Python 库一样安装——也就是说，通过 `pip install jupyter`，但幸运的是，它在 Anaconda 中已预安装。

## 运行 Jupyter Notebook

你可以通过 Anaconda Navigator 中的适当链接启动 Jupyter，或者在 Anaconda Prompt / CMD / Terminal 中执行命令 `jupyter notebook`。

Jupyter 将在你的浏览器中打开，你可以在其中导航到你的工作目录，并创建、编辑和运行你的代码文件。

## 访问代码文件

你可以在 [`packt.live/2OL5E9t`](https://packt.live/2OL5E9t) 找到本书的完整代码文件。你也可以通过使用互动实验环境 [`packt.live/2CXyFLS`](https://packt.live/2CXyFLS) 直接在你的网页浏览器中运行许多活动和练习。

我们尽力支持所有活动和练习的互动版本，但我们也建议你进行本地安装，以防这种支持不可用的情况。

本书中使用的高质量彩色图像可以在 [`packt.live/2u9Tno4`](https://packt.live/2u9Tno4) 找到。

如果你在安装过程中遇到任何问题或有任何疑问，请通过电子邮件联系我们：`workshops@packt.com`。
