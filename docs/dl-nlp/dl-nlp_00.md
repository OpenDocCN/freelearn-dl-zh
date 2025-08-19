# 第一章：*前言*

## 关于

本节简要介绍了作者、本书的覆盖范围、入门所需的技术技能，以及完成书中所有活动和练习所需的硬件和软件要求。

## 关于本书

将深度学习方法应用于各种 NLP 任务，可以使你的计算算法在速度和准确性方面达到全新的水平。《自然语言处理中的深度学习》一书首先介绍了自然语言处理领域的基本构建块。接着，本书介绍了你可以用最先进的神经网络模型解决的各种问题。深入探讨不同的神经网络架构及其具体应用领域，将帮助你理解如何选择最适合你需求的模型。在你逐步深入本书的过程中，你将学习卷积神经网络、循环神经网络和递归神经网络，同时还将涉及长短期记忆网络（LSTM）。在后续章节中，你将能够使用 NLP 技术（如注意力模型和束搜索）开发应用程序。

在本书结束时，你不仅将掌握自然语言处理的基本知识，还能选择最佳的文本预处理和神经网络模型，解决一系列 NLP 问题。

### 关于作者

**Karthiek Reddy Bokka** 是一位语音和音频机器学习工程师，毕业于南加州大学，目前在波特兰的 Bi-amp Systems 工作。他的兴趣包括深度学习、数字信号和音频处理、自然语言处理和计算机视觉。他有设计、构建和部署人工智能应用程序的经验，致力于利用人工智能解决现实世界中与各种实用数据相关的问题，包括图像、语音、音乐、非结构化原始数据等。

**Shubhangi Hora** 是一位 Python 开发者、人工智能爱好者和作家。她拥有计算机科学和心理学的背景，尤其对与心理健康相关的 AI 感兴趣。她目前生活在印度浦那，热衷于通过机器学习和深度学习推动自然语言处理的发展。除此之外，她还喜欢表演艺术，并是一名受过训练的音乐家。

**Tanuj Jain** 是一位数据科学家，目前在一家总部位于德国的公司工作。他一直在开发深度学习模型，并将其投入到商业生产中。自然语言处理是他特别感兴趣的领域，他已将自己的专业知识应用于分类和情感评分任务。他拥有电气工程硕士学位，专注于统计模式识别。

**Monicah Wambugu** 是一家金融科技公司的首席数据科学家，该公司通过利用数据、机器学习和分析技术执行替代性信用评分，提供小额贷款。她是加州大学伯克利分校信息学院信息管理与系统硕士项目的研究生。Monicah 特别感兴趣的是数据科学和机器学习如何被用来设计能够响应目标受众行为和社会经济需求的产品和应用。

### 描述

本书将从自然语言处理领域的基本构建模块开始。它将介绍可以通过最先进的神经网络模型解决的问题。它将深入讨论文本处理任务中所需的必要预处理。本书将涵盖 NLP 领域的一些热门话题，包括卷积神经网络、递归神经网络和长短期记忆网络。读者将理解文本预处理和超参数调优的重要性。

### 学习目标

+   学习自然语言处理的基本原理。

+   理解深度学习问题的各种预处理技术。

+   使用 word2vec 和 Glove 开发文本的向量表示。

+   理解命名实体识别。

+   使用机器学习进行词性标注。

+   训练和部署可扩展的模型。

+   理解几种神经网络架构。

### 读者对象

有志成为数据科学家和工程师的人，希望在自然语言处理领域入门深度学习。

读者将从自然语言处理概念的基础开始，逐渐深入理解神经网络的概念及其在文本处理问题中的应用。他们将学习不同的神经网络架构及其应用领域。要求具备扎实的 Python 和线性代数知识。

### 方法

面向自然语言处理的深度学习将从自然语言处理的基本概念开始。一旦基本概念介绍完毕，听众将逐步了解 NLP 技术在现实世界中可应用的领域和问题。一旦用户理解了问题领域，解决方案的开发方法将会被介绍。作为基于解决方案的方法的一部分，讨论神经网络的基本构建模块。最终，将详细阐述各种神经网络的现代架构及其对应的应用领域，并给出实例。

### 硬件要求

为了获得最佳体验，我们推荐以下硬件配置：

+   处理器：Intel Core i5 或同等处理器

+   内存：4 GB RAM

+   存储：5 GB 可用空间

### 软件要求

我们还建议您提前安装以下软件：

+   操作系统：Windows 7 SP1 64 位、Windows 8.1 64 位或 Windows 10 64 位、Linux（Ubuntu、Debian、Red Hat 或 Suse）或最新版本的 OS X

+   Python（3.6.5 或更高版本，建议使用 3.7；可通过[`www.python.org/downloads/release/python-371/`](https://www.python.org/downloads/release/python-371/)下载）

+   Jupyter（前往[`jupyter.org/install`](https://jupyter.org/install)，并按照说明安装）。另外，您也可以使用 Anaconda 安装 Jupyter。

+   Keras（[`keras.io/#installation`](https://keras.io/#installation)）

+   Google Colab：它是一个免费的 Jupyter 笔记本环境，运行在云基础设施上。强烈推荐使用，因为它无需任何设置，并且预先安装了流行的 Python 包和库（[`colab.research.google.com/notebooks/welcome.ipynb`](https://colab.research.google.com/notebooks/welcome.ipynb)）

### 规范

文本中的代码字、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 句柄如下所示：

代码块设置如下：

```py
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

新术语和重要单词以粗体显示。屏幕上看到的词语，例如菜单或对话框中的内容，文本中会像这样显示：“接下来，点击**生成文件**，然后点击**立即下载**，并将下载的文件命名为**model.h5**。”

### 安装与设置

每一段伟大的旅程都始于谦逊的第一步，我们即将展开的数据处理之旅也不例外。在我们用数据做出令人惊叹的事情之前，我们需要准备好最具生产力的环境。在这篇简短的说明中，我们将展示如何做到这一点。

### 在 Windows 上安装 Python

1.  在官方安装页面[`www.python.org/downloads/windows/`](https://www.python.org/downloads/windows/)找到所需的 Python 版本。

1.  请确保安装正确的“-bit”版本，取决于您的计算机系统，是 32 位还是 64 位。您可以在操作系统的系统属性窗口中找到此信息。

    下载安装程序后，只需双击文件并按照屏幕上的用户友好提示进行操作。

### 在 Linux 上安装 Python

在 Linux 上安装 Python，请执行以下操作：

1.  打开命令提示符，并通过运行 `python3 --version` 来确认是否已安装 Python 3。

1.  要安装 Python 3，请运行以下命令：

    ```py
    sudo apt-get update
    sudo apt-get install python3.6
    ```

1.  如果遇到问题，网上有许多资源可以帮助您排查并解决问题。

### 在 macOS X 上安装 Python

要在 macOS X 上安装 Python，请执行以下操作：

1.  通过按住命令和空格键（*CMD* + *Space*），在打开的搜索框中输入 terminal，按下回车键打开终端。

1.  通过运行命令 `xcode-select --install`，在命令行中安装 Xcode。

1.  安装 Python 3 的最简单方法是使用 homebrew，可以通过命令行运行 `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"` 安装。

1.  将 homebrew 添加到你的 PATH 环境变量中。通过运行`sudo nano ~/.profile`打开命令行中的配置文件，并在底部插入`export PATH="/usr/local/opt/python/libexec/bin:$PATH"`。

1.  最后一步是安装 Python。在命令行中运行`brew install python`。

1.  请注意，如果你安装了 Anaconda，最新版本的 Python 将会自动安装。

### **安装 Keras**

安装 Keras，请执行以下步骤：

1.  由于**Keras**需要另一个深度学习框架作为后端，因此你需要先下载另一个框架，推荐使用**TensorFlow**。

    要为你的平台安装**TensorFlow**，请点击[`www.tensorflow.org/install/`](https://www.tensorflow.org/install/)。

1.  一旦后端安装完成，你可以运行`sudo pip install keras`进行安装。

    另外，你也可以从 GitHub 源代码安装，使用以下命令克隆`Keras`：

    ```py
    git clone https://github.com/keras-team/keras.git
    ```

1.  安装`cd keras` `sudo python setup.py install`

    现在需要配置后端。有关更多信息，请参考以下链接：**(**[`keras.io/backend/`](https://keras.io/backend/))

### 额外资源

本书的代码包也托管在 GitHub 上，网址为：[`github.com/TrainingByPackt/Deep-Learning-for-Natural-Language-Processing`](https://github.com/TrainingByPackt/Deep-Learning-for-Natural-Language-Processing)。我们还有来自丰富书籍和视频目录的其他代码包，地址是[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快来看看吧！

你可以从这里下载本书的图形包：

[`www.packtpub.com/sites/default/files/downloads/9781838558024_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781838558024_ColorImages.pdf)
