# 前言

# 关于本书

你是否对像Alexa和Siri这样的应用感兴趣，它们如何在几秒钟内准确处理信息并返回准确的结果？你是否在寻找一本实用的指南，教你如何构建能够改变人工智能世界的智能应用？*《应用人工智能与自然语言处理工作坊》*将带你进行一次实践之旅，在那里你将学习如何使用**亚马逊网络服务**（**AWS**）构建**人工智能**（**AI**）和**自然语言处理**（**NLP**）应用。

从人工智能和机器学习的介绍开始，本书将解释Amazon S3，或称Amazon Simple Storage Service的工作原理。然后，你将集成AI与AWS来构建无服务器服务，并使用亚马逊的NLP服务Comprehend对文档进行文本分析。随着你的进步，本书将帮助你掌握主题建模，以从一组未知主题的文档中提取和分析共同主题。你还将与Amazon Lex合作，创建和定制用于任务自动化的聊天机器人，并使用Amazon Rekognition检测图像中的对象、场景和文本。

在*《应用人工智能与自然语言处理工作坊》*结束时，你将具备构建具有AWS的可扩展智能应用所需的知识和技能。

## 听众

如果你是一位希望探索AWS人工智能和机器学习能力的机器学习爱好者、数据科学家或程序员，这本书适合你。虽然不是必需的，但基本了解AI和NLP将有助于快速掌握关键主题。

## 关于章节

*第一章*，*AWS简介*，介绍了AWS界面。你将学习如何使用亚马逊的简单存储服务，以及使用亚马逊Comprehend API测试NLP接口。

*第二章*，*使用自然语言处理分析文档和文本*，介绍了AWS AI服务集和新兴的计算范式——无服务器计算。然后，你将应用NLP和亚马逊Comprehend服务来分析文档。

*第三章*，*主题建模与主题提取*，描述了主题建模分析的基础，你将学习如何使用Amazon Comprehend进行主题建模来提取和分析共同主题。

*第四章*，*对话式人工智能*，讨论了设计对话式人工智能的最佳实践，然后继续展示如何使用Amazon Lex开发聊天机器人。

*第五章*，*在聊天机器人中使用语音*，教你亚马逊Connect的基础知识。你将为语音交互编程聊天机器人，并使用亚马逊Connect和你的电话号码创建一个个人呼叫中心，以与你的机器人互动。

**第6章**，**计算机视觉和图像处理**，向您介绍使用计算机视觉进行图像分析的Rekognition服务。您将学习如何分析人脸和识别图像中的名人。您还将能够比较不同图像中的面孔，以查看它们彼此之间有多接近。

## 习惯用法

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟URL、用户输入和Twitter标签以如下方式显示：“在此，所选的存储桶名称为`known-tm-analysis`，但您需要创建一个唯一的名称。”

代码块设置如下：

```py
filename = str(text_file_obj['s3']['object']['key'])
print("filename: ", filename)
```

您在屏幕上看到的单词，例如在菜单或对话框中，也以这种方式出现在文本中：“从屏幕左侧的菜单面板中选择`路由`菜单。”

新术语和重要单词以如下方式显示：“亚马逊Comprehend使用的机器学习算法来执行主题建模被称为**潜在狄利克雷分配**（**LDA**）。”

## 代码展示

涵盖多行的代码行使用反斜杠（`\`）分隔。当代码执行时，Python将忽略反斜杠，并将下一行的代码视为当前行的直接延续。

例如：

```py
history = model.fit(X, y, epochs=100, batch_size=5, verbose=1, \
                    validation_split=0.2, shuffle=False)
```

为了帮助解释特定的逻辑部分，代码中添加了注释。单行注释使用`#`符号表示，如下所示：

```py
# Print the sizes of the dataset
print("Number of Examples in the Dataset = ", X.shape[0])
print("Number of Features for each example = ", X.shape[1])
```

多行注释由三个引号包围，如下所示：

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

在我们详细探索这本书之前，我们需要设置特定的软件和工具。在以下部分，我们将看到如何做到这一点。

## 软件要求

您还需要提前安装以下软件：

1.  操作系统：Windows 7 SP1 64位、Windows 8.1 64位或Windows 10 64位、macOS或Linux

1.  浏览器：最新版本的Google Chrome

1.  AWS免费层账户

1.  Python 3.6或更高版本

1.  Jupyter Notebook

## 安装和设置

在您开始这本书之前，您需要一个AWS账户。您还需要设置AWS **命令行界面**（**CLI**），相关步骤如下。您还需要Python 3.6或更高版本、pip以及AWS Rekognition账户来完成这本书。

## AWS账户

对于AWS免费层账户，您需要一个个人电子邮件地址、一张信用卡或借记卡，以及一部可以接收短信的手机，以便您验证您的账户。要创建新账户，请点击此链接：[https://aws.amazon.com/free/](https://aws.amazon.com/free/).

## 关于AWS区域的说明

AWS服务器分布在全球AWS所说的区域中。自从AWS开始以来，区域数量已经增长，您可以在[https://aws.amazon.com/about-aws/global-infrastructure/regions_az/](https://aws.amazon.com/about-aws/global-infrastructure/regions_az/)找到所有区域的列表。当您创建AWS账户时，您还需要选择一个区域。您可以通过访问[aws.amazon.com](http://aws.amazon.com)并选择`AWS管理控制台`来找到您的区域：

![图 0.1：我的账户下拉菜单

![img/B16061_00_01.jpg]

图 0.1：我的账户下拉菜单

在 AWS 管理控制台中，你的区域将显示在右上角。你可以点击它并更改区域：

![图 0.2：AWS 区域列表

![img/B16061_00_02.jpg]

图 0.2：AWS 区域列表

改变区域的一个原因是因为并非所有 AWS 服务都在所有区域可用。在 [https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) 的区域表中列出了每个区域当前可用的服务。因此，如果你想要访问的服务在你的区域不可用，你可以更改你的区域。但请注意区域之间（如果有）的收费差异。此外，你在一个区域创建的工件可能不在另一个区域可用，例如 S3 存储桶。如果你在想，亚马逊不自动使 S3 数据跨区域可用的一个原因是合规性和法规。你必须明确复制或重新创建 S3 存储桶和文件。虽然一开始管理 AWS 服务和区域可能看起来很繁琐，但很快就会习惯。正如我们提到的，亚马逊这样做有原因。

注意

根据你的位置，仅通过更改区域可能无法访问 AWS 服务。例如，Amazon Connect 并非在所有地方都可用，仅通过从下拉菜单更改区域并不能使用 Amazon Connect，因为这涉及到本地号码分配。为了使用 Amazon Connect，我们需要在注册 AWS 时指定 Amazon Connect 可用的地址。在撰写本书时（2020 年 4 月），Amazon Connect 可在美国、英国、澳大利亚、日本、德国和新加坡使用。但好消息是亚马逊正在不断扩展其服务。所以，当你阅读这本书的时候，Amazon Connect 可能会在你所在的地方可用。

## AWS CLI 设置

按照此 URL 的说明安装 AWS CLI（版本 2）：[https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)。AWS 文档描述了如何在各种操作系统上安装 CLI。要验证安装是否成功，请打开命令提示符并输入 `aws --version`。

## AWS CLI 的配置和凭证文件

AWS CLI 文档清楚地描述了配置和凭证文件设置。更多信息，请访问 [https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html)。

## Amazon Rekognition 账户

您需要创建一个新的Amazon Rekognition免费层账户，客户可以使用该账户在前12个月内每月免费分析高达5,000张图片。要创建免费账户，请访问[https://aws.amazon.com/rekognition/](https://aws.amazon.com/rekognition/)。

注意

界面和结果可能与章节中显示的图片略有不同，因为亚马逊定期更新和简化其界面并重新训练模型。

## 安装Python和Anaconda

以下部分将帮助您在Windows、macOS和Linux系统上安装Python和Anaconda。

### 在Windows上安装Python和Anaconda

在Windows上安装Python的步骤如下：

1.  在官方安装页面[https://www.anaconda.com/distribution/#windows](https://www.anaconda.com/distribution/#windows)上找到您想要的Anaconda版本。

1.  确保您从下载页面选择Python 3.7。

1.  确保您安装了适合您计算机系统的正确架构；即32位或64位。您可以在操作系统的**系统属性**窗口中找到此信息。

1.  下载安装程序后，只需双击文件并遵循屏幕上的用户友好提示。

### 在Linux上安装Python和Anaconda

要在Linux上安装Python，您有几个不错的选择：

1.  打开命令提示符，通过运行`python3 --version`来验证`Python 3`是否已经安装。

1.  要安装Python 3，请运行以下命令：

    ```py
    sudo apt-get update
    sudo apt-get install python3.7
    ```

1.  如果您遇到问题，网上有众多资源可以帮助您解决问题。

1.  您也可以使用Anaconda安装Python。通过从[https://www.anaconda.com/distribution/#linux](https://www.anaconda.com/distribution/#linux)下载安装程序并按照说明进行操作来安装Anaconda for Linux。

## 在macOS上安装Python和Anaconda

与Linux类似，您在Mac上安装Python有几个方法。要在macOS上安装Python，请按照以下步骤操作：

1.  通过按*CMD + Spacebar*打开Mac的终端，在打开的搜索框中输入`terminal`，然后按*Enter*。

1.  通过在命令行中运行`xcode-select --install`来安装Xcode。

1.  安装Python 3的最简单方法是使用Homebrew，通过在命令行中运行`ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`来安装。

1.  将Homebrew添加到您的`$PATH`环境变量中。通过在命令行中运行`sudo nano ~/.profile`并插入`export PATH="/usr/local/opt/python/libexec/bin:$PATH"`在底部来打开您的配置文件。

1.  最后一步是安装Python。在命令行中，运行`brew install python`。

1.  同样，您也可以通过Anaconda安装程序安装Python，该安装程序可在[https://www.anaconda.com/distribution/#macos](https://www.anaconda.com/distribution/#macos)找到。

## Project Jupyter

Project Jupyter 是一个开源的免费软件，它允许你从特殊的笔记本中交互式地运行用 Python 编写的代码和一些其他语言，类似于浏览器界面。它于 2014 年从 **IPython** 项目中诞生，并已成为整个数据科学工作者的默认选择。

要安装 Jupyter Notebook，请访问：[https://jupyter.org/install](https://jupyter.org/install)。

在 [https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html)，你可以找到启动 Jupyter Notebook 服务器所需的所有详细信息。在这本书中，我们使用经典笔记本界面。

通常，我们使用 `jupyter notebook` 命令从命令行启动笔记本。

在以下 *安装代码包* 部分中，从你下载代码文件的目录启动笔记本。

例如，在我们的案例中，我们将文件安装在了以下目录 `/Users/ksankar/Documents/aws_book/Artificial-Intelligence-and-Natural-Language-Processing-with-AWS.`

在 CLI 中，输入 `cd /Users/ksankar/Documents/aws_book/Artificial-Intelligence-and-Natural-Language-Processing-with-AWS` 然后输入 `jupyter notebook` 命令。Jupyter 服务器将启动，你将看到 Jupyter 浏览器控制台：

![图 0.3：Jupyter 浏览器控制台

](img/B16061_00_03.jpg)

图 0.3：Jupyter 浏览器控制台

一旦你启动了 Jupyter 服务器，点击 `New` 并选择 `Python 3`。一个新标签页将打开，一个新的空笔记本。重命名 Jupyter 文件：

![图 0.4：Jupyter 服务器界面

](img/B16061_00_04.jpg)

图 0.4：Jupyter 服务器界面

Jupyter 笔记本的主要构建块是单元格。有两种类型的单元格：`In`（代表输入）和`Out`（代表输出）。你可以在 `In` 单元格中编写代码、普通文本和 Markdown，按 *Shift* + *Enter*（或 *Shift* + *Return*），该特定 `In` 单元格中编写的代码将被执行。结果将在一个 `Out` 单元格中显示，你将进入一个新的 `In` 单元格，准备编写下一块代码。一旦你习惯了这种界面，你将逐渐发现它提供的强大功能和灵活性。

当你开始一个新的单元格时，默认情况下，它假定你将在其中编写代码。然而，如果你想写文本，你必须更改类型。你可以使用以下键序列来完成：*Esc* | `M` | *Enter*:

![图 0.5：Jupyter Notebook

](img/B16061_00_05.jpg)

图 0.5：Jupyter Notebook

当你写完一些文本后，使用 *Shift* + *Enter* 执行它。与代码单元格的情况不同，编译后的 Markdown 的结果将在与 `In` 单元格相同的位置显示。

要获取 Jupyter 中所有便捷的快捷键的“速查表”，请访问[https://gist.github.com/kidpixo/f4318f8c8143adee5b40](https://gist.github.com/kidpixo/f4318f8c8143adee5b40)。有了这个基本介绍，我们就准备好开始一段激动人心且富有启发的旅程了。

## 安装库

`pip` 随 Anaconda 预先安装。一旦 Anaconda 在您的机器上安装，所有必需的库都可以使用 `pip` 安装，例如，`pip install numpy`。或者，您也可以使用 `pip install –r requirements.txt` 命令安装所有必需的库。您可以在[https://packt.live/30ddspf](https://packt.live/30ddspf)找到 `requirements.txt` 文件。

练习和活动将在 Jupyter 笔记本中执行。Jupyter 是一个 Python 库，可以像其他 Python 库一样安装——即使用 `pip install jupyter`，但幸运的是，它随 Anaconda 预先安装。要打开笔记本，只需在终端或命令提示符中运行 `jupyter notebook` 命令即可。

## 访问代码文件

您可以在[https://packt.live/2O67hxH](https://packt.live/2O67hxH)找到这本书的完整代码文件。

如果您在安装过程中遇到任何问题或疑问，请通过电子邮件发送给我们[workshops@packt.com](mailto:workshops@packt.com)。
