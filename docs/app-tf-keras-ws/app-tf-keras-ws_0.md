# 前言

# 关于本书

机器学习赋予计算机像人类一样学习的能力。它正以多种形式逐渐对各行业产生变革性影响，成为为未来数字经济做准备的关键技能。

作为初学者，通过学习所需的技巧，你将开启一个机会的世界，能够在机器学习、深度学习和现代数据分析领域做出贡献，使用最新的尖端工具。

*应用 TensorFlow 和 Keras 工作坊*首先向你展示神经网络的工作原理。在你理解了基础知识后，你将通过调整超参数训练一些网络。为了进一步提高技能，你将学习如何选择最合适的模型来解决当前问题。在处理高级概念时，你将发现如何通过将构建基本深度学习系统所需的所有关键元素——数据、模型和预测——结合起来，来组装一个深度学习系统。最后，你将探索如何评估模型的表现，并通过模型评估和超参数优化等技术来改进模型。

在本书结束时，你将学会如何构建一个预测未来价格的比特币应用程序，并能够为其他项目构建自己的模型。

## 读者群体

如果你是一名数据科学家，或者是机器学习和深度学习的爱好者，正在寻求将 TensorFlow 和 Keras 模型设计、训练并部署到实际应用中，那么这个工作坊适合你。了解计算机科学和机器学习的概念，以及具备数据分析经验，将帮助你轻松理解本书中解释的主题。

## 关于各章节

*第一章*，*神经网络和深度学习简介*，引导我们通过使用 TensorBoard 选择一个经过 TensorFlow 训练的神经网络，并通过变化的迭代次数和学习率训练神经网络。这将为你提供如何训练一个高性能神经网络的实践经验，并让你探索其一些局限性。

*第二章*，*现实世界中的深度学习：预测比特币价格*，教我们如何从数据输入到预测，组装一个完整的深度学习系统。创建的模型将作为基准，我们可以从中进行改进。

*第三章*，*现实世界中的深度学习：评估比特币模型*，重点讲解如何评估神经网络模型。通过超参数调优，我们将提高网络的表现。然而，在改变任何参数之前，我们需要先衡量模型的表现。在本章结束时，你将能够使用不同的功能和技术来评估一个模型。

*第四章*，*产品化*，讲解了如何处理新数据。我们将创建一个能够从展示给它的模式中不断学习并做出更好预测的模型。我们将使用一个 Web 应用程序作为示例，展示如何部署深度学习模型。

## 约定

文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 句柄如下所示：

“在激活虚拟环境后，确保通过对 `requirements.txt` 文件执行 `pip` 来安装正确的组件。”

屏幕上看到的词汇（例如在菜单或对话框中）以相同的格式出现。

一块代码的设置如下：

```py
$ python -m venv venv
$ venv/bin/activate
```

新术语和重要单词如下所示：

“在机器学习中，通常会定义两个不同的术语：**参数** 和 **超参数**。”

## 代码展示

跨多行的代码使用反斜杠（ `\` ）分割。当代码执行时，Python 会忽略反斜杠，并将下一行的代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                   validation_split=0.2, shuffle=False)
```

注释是添加到代码中的，以帮助解释特定的逻辑。单行注释使用 `#` 符号，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释使用三引号括起来，如下所示：

```py
"""
Define a seed for the random number generator to ensure the 
result will be reproducible
"""
seed = 1
np.random.seed(seed)
random.set_seed(seed)
```

## 设置您的环境

在我们详细探讨本书之前，我们需要设置特定的软件和工具。在接下来的部分，我们将展示如何做到这一点。

## 安装

以下部分将帮助您在 Windows、macOS 和 Linux 系统上安装 Python。

### 在 Windows 上安装 Python

Python 在 Windows 上的安装方法如下：

1.  确保从官方安装页面的下载页 [`www.anaconda.com/distribution/#windows`](https://www.anaconda.com/distribution/#windows) 选择 Python 3.7（与 TensorFlow 2.0 兼容）。

1.  确保安装适合您计算机系统的正确架构；即 32 位或 64 位。您可以在操作系统的 **系统属性** 窗口中找到此信息。

1.  下载完安装程序后，只需双击该文件并按照屏幕上的用户友好提示进行操作。

### 在 Linux 上安装 Python

要在 Linux 上安装 Python，您有几个不错的选择；即命令提示符和 Anaconda。

如下使用命令提示符：

1.  打开命令提示符并通过运行 `python3 --version` 验证 `p\Python 3` 是否已安装。

1.  要安装 Python 3，请运行以下命令：

    ```py
    sudo apt-get update
    sudo apt-get install python3.7
    ```

1.  如果遇到问题，网上有许多资源可以帮助您排查问题。

或者，您可以通过从 [`www.anaconda.com/distribution/#linux`](https://www.anaconda.com/distribution/#linux) 下载安装程序并按照说明进行操作来安装 Anaconda Linux。

### 在 macOS 上安装 Python

与 Linux 类似，您在 Mac 上安装 Python 有几种方法。安装 Python 在 macOS X 上的一种方法如下：

1.  通过按 *CMD* + *Spacebar* 打开 Mac 的终端，在打开的搜索框中输入 `terminal`，然后按 *Enter*。

1.  通过运行 `xcode-select --install` 在命令行安装 Xcode。

1.  安装 Python 3 的最简单方法是使用 Homebrew，可以通过命令行运行`ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`来安装。

1.  将 Homebrew 添加到你的 `$PATH` 环境变量中。通过运行`sudo nano ~/.profile`打开命令行中的配置文件，并在文件底部插入`export PATH="/usr/local/opt/python/libexec/bin:$PATH"`。

1.  最后一步是安装 Python。在命令行中运行`brew install python`。

你也可以通过 Anaconda 安装程序来安装 Python，安装包可以从[`www.anaconda.com/distribution/#macos`](https://www.anaconda.com/distribution/#macos)下载。

### 安装 pip

通过 Anaconda 安装程序安装 Python 时，`pip`（Python 的包管理器）会预安装。但是，如果你是直接安装的 Python，你需要手动安装`pip`。安装 `pip` 的步骤如下：

1.  访问[`bootstrap.pypa.io/get-pip.py`](https://bootstrap.pypa.io/get-pip.py)并将文件保存为[`get-pip.py`](http://get-pip.py)。

1.  访问你保存[`get-pip.py`](http://get-pip.py)的文件夹。在该文件夹中打开命令行（Linux 用户使用 Bash，Mac 用户使用 Terminal）。

1.  在命令行中执行以下命令：

    ```py
    python get-pip.py
    ```

1.  请注意，在执行此命令之前，你应该已安装 Python。

1.  一旦 `pip` 安装完成，你可以安装所需的库。要安装 pandas，只需执行`pip install pandas`。要安装特定版本的库，例如 pandas 的 0.24.2 版本，可以执行`pip install pandas=0.24.2`。

## Jupyter Notebook

如果你没有通过 Anaconda 安装程序安装 Python，你需要手动安装 Jupyter。请参考[`jupyter.readthedocs.io/en/latest/install.html#id4`](https://jupyter.readthedocs.io/en/latest/install.html#id4)中*经验丰富的 Python 用户的替代方法：使用 pip 安装 Jupyter 部分*。

## JupyterLab

Anaconda 分发版包含 JupyterLab，它允许你运行 Jupyter Notebooks。Jupyter Notebooks 可以通过浏览器访问，并允许你在集成环境中交互式地运行代码，以及嵌入图像和文本。

## 安装库

`pip` 已随 Anaconda 一起预安装。一旦 Anaconda 安装到你的计算机上，所有必需的库都可以使用`pip`安装，例如，`pip install numpy`。或者，你也可以使用`pip install –r requirements.txt`安装所有必需的库。你可以在[`packt.live/3haRJp0`](https://packt.live/3haRJp0)找到`requirements.txt`文件。

练习和活动将在 Jupyter Notebooks 中执行。Jupyter 是一个 Python 库，可以像安装其他 Python 库一样安装 —— 即使用`pip install jupyter`，幸运的是，Anaconda 已经预装了它。要打开一个 notebook，只需在 Terminal 或命令提示符中运行命令`jupyter notebook`。

## 访问代码文件

您可以在[`packt.live/2DnXRLS`](https://packt.live/2DnXRLS)找到本书的完整代码文件。您还可以通过使用[`packt.live/39dH7ml`](https://packt.live/39dH7ml)上的互动实验环境，直接在网页浏览器中运行许多活动和练习。

我们已尽力支持所有活动和练习的互动版本，但我们仍然推荐进行本地安装，以防在某些情况下无法使用此支持。

如果您在安装过程中遇到任何问题或有任何疑问，请通过电子邮件联系我们 `workshops@packt.com`。
