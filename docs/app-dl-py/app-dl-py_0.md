# 前言

本学习路径采取逐步教学法，教你如何入门数据科学、机器学习和深度学习。每个模块都是在上一章节的基础上进行扩展的。本书包含多个演示，使用真实的商业场景，让你在高度相关的背景下练习并应用你所学的新技能。

在学习路径的第一部分，你将学习入门级的数据科学。你将了解常用的 Anaconda 发行版中的库，并利用真实数据集探索机器学习模型，培养你在实际工作中所需的技能和经验。

在第二部分，你将接触到神经网络和深度学习。接着，你将学习如何训练、评估和部署 TensorFlow 和 Keras 模型，将它们作为真实世界的 Web 应用。读完本书后，你将具备在深度学习环境中构建应用程序的知识，并能够创建复杂的数据可视化和预测。

# 本书适用人群

如果你是一个 Python 程序员，准备踏入数据科学的世界，那么这是一个正确的入门方式。对于有经验的开发者、分析师或数据科学家，想要使用 TensorFlow 和 Keras 时，这也是一个理想的选择。我们假设你已经熟悉 Python、Web 应用开发、Docker 命令，以及线性代数、概率和统计的基本概念。

# 本书内容概述

*第一章*，*Jupyter 基础*，介绍了 Jupyter 中数据分析的基础知识。我们将从 Jupyter 的使用说明和功能开始，如魔法函数和自动补全功能。接着，我们将转向数据科学相关的内容。我们将在 Jupyter Notebook 中进行探索性分析，利用散点图、直方图和小提琴图等可视化工具，帮助我们更深入地理解数据。同时，我们还会进行简单的预测建模。

*第二章*，*数据清理与高级机器学习*，展示了如何在 Jupyter Notebooks 中训练预测模型。我们将讨论如何规划机器学习策略。本章还解释了机器学习的术语，如监督学习、无监督学习、分类和回归。我们将讨论使用 scikit-learn 和 pandas 进行数据预处理的方法。

*第三章*，*网页抓取与交互式可视化*，解释了如何抓取网页表格，并使用交互式可视化来研究数据。我们将从 HTTP 请求的工作原理开始，重点讲解 GET 请求及其响应状态码。然后，我们将进入 Jupyter Notebook，使用 Requests 库通过 Python 发起 HTTP 请求。我们将看到如何利用 Jupyter 渲染 HTML，并与实际的网页进行交互。在发出请求后，我们将学习如何使用 Beautiful Soup 解析 HTML 中的文本，并使用该库抓取表格数据。

*第四章*，*神经网络与深度学习简介*，帮助你设置和配置深度学习环境，并开始查看单独的模型和案例研究。它还讨论了神经网络及其理念，以及它们的起源并探索它们的强大功能。

*第五章*，*模型架构*，展示了如何使用深度学习模型预测比特币价格。

*第六章*，*模型评估与优化*，介绍如何评估神经网络模型。我们将修改网络的超参数以提高其性能。

*第七章*，*产品化*，解释如何将深度学习模型转换为可运行的应用程序。我们将部署我们的比特币预测模型，并创建一个新的模型来处理新的数据。

# 为了充分利用本书

本书最适合那些对数据分析感兴趣，并希望在使用 TensorFlow 和 Keras 开发应用程序领域提高知识的专业人士和学生。为了获得最佳体验，你应该具备编程基础知识，并有一定的 Python 使用经验。特别是，熟悉 Pandas、matplotlib 和 scikit-learn 等 Python 库将大有帮助。

# 下载示例代码文件

你可以从 [www.packtpub.com](http://www.packtpub.com) 下载本书的示例代码文件。如果你是从其他地方购买本书，你可以访问 [www.packtpub.com/support](http://www.packtpub.com/support) 并注册，以便将文件直接发送到你的邮箱。

你可以按照以下步骤下载代码文件：

1.  登录或注册 [www.packtpub.com](http://www.packtpub.com/support)。

1.  选择 SUPPORT 选项卡。

1.  点击“代码下载 & 勘误”。

1.  在搜索框中输入书名，并按照屏幕上的提示操作。

一旦文件下载完成，请确保使用最新版本的以下工具解压或提取文件夹：

+   WinRAR/7-Zip for Windows

+   Zipeg/iZip/UnRarX for Mac

+   7-Zip/PeaZip for Linux

本书的代码包也托管在 GitHub 上，链接为 [`github.com/TrainingByPackt/Applied-Deep-Learning-with-Python`](https://github.com/TrainingByPackt/Applied-Deep-Learning-with-Python)。如果代码有任何更新，将会在现有的 GitHub 仓库中进行更新。

我们还有其他来自丰富书籍和视频目录的代码包，均可在 [`github.com/TrainingByPackt/Applied-Deep-Learning-with-Python`](https://github.com/TrainingByPackt/Applied-Deep-Learning-with-Python) 查看！快来看看吧！

# 使用的规范

本书中使用了多种文本规范。

`CodeInText`：指示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号名。以下是一个例子：“我们可以看到`NotebookApp`在本地服务器上运行。”

代码块如下所示：

```py
fig, ax = plt.subplots(1, 2)
sns.regplot('RM', 'MEDV', df, ax=ax[0],
scatter_kws={'alpha': 0.4}))
sns.regplot('LSTAT', 'MEDV', df, ax=ax[1],
scatter_kws={'alpha': 0.4}))
```

当我们希望特别强调某个代码块的部分时，相关的行或项目会用粗体显示：

```py
    cat chapter-1/requirements.txt
    matplotlib==2.0.2
 numpy==1.13.1
 pandas==0.20.3
 requests==2.18.4
```

任何命令行输入或输出都以如下方式呈现：

```py
pip install version_information 
pip install ipython-sql
```

**粗体**：表示一个新术语、一个重要单词或您在屏幕上看到的单词。例如，菜单或对话框中的文字会像这样出现在文本中。以下是一个示例：“注意如何使用**白色裙子**的价格来填补**缺失值**。”

警告或重要提示会以这种方式呈现。

提示和技巧会以这种方式呈现。

# 与我们联系

我们始终欢迎读者的反馈。

**一般反馈**：请通过电子邮件 `feedback@packtpub.com` 并在邮件主题中提及书籍标题。如果您对本书的任何内容有疑问，请通过 `questions@packtpub.com` 与我们联系。

**勘误表**：尽管我们已经尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将非常感谢您向我们报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入相关详情。

**盗版**：如果您在互联网上发现任何非法复制的我们作品的形式，我们将非常感谢您提供相关地址或网站名称。请通过电子邮件 `copyright@packtpub.com` 与我们联系，并附上材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域拥有专业知识，并且有兴趣撰写或参与书籍的创作，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。阅读并使用本书后，为什么不在您购买书籍的网站上留下评论呢？潜在读者可以看到并利用您的公正意见来做出购买决策，我们 Packt 也能了解您对我们产品的看法，而我们的作者也能看到您对其书籍的反馈。感谢您的支持！

如需了解更多关于 Packt 的信息，请访问 [packtpub.com](https://www.packtpub.com/)。
