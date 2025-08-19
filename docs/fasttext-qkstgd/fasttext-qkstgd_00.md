# 前言

FastText 是一个先进的工具，可用于执行文本分类和构建高效的词向量表示。它是开源的，设计于**Facebook 人工智能研究** (**FAIR**) 实验室。它使用 C++ 编写，并且还提供了 Python 的封装接口。

本书的目标非常宏大，旨在涵盖你在构建实际 NLP 应用时所需的所有技术和知识。它还将介绍 fastText 构建的算法，帮助你清晰理解在什么样的上下文中，能够从 fastText 获得最佳效果。

# 本书的读者对象

如果你是软件开发人员或机器学习工程师，试图了解 NLP 领域的最新技术，本书将对你有所帮助。本书的很大一部分内容涉及如何创建 NLP 管道的实际问题和考虑事项。如果你是 NLP 研究人员，这本书也很有价值，因为你将了解开发 fastText 软件时所采用的内部算法和开发考虑。所有的代码示例都使用 Jupyter Notebooks 编写。我强烈建议你自己输入这些代码，进行修改和实验。保留这些代码，以便以后在实际项目中使用。

# 本书的内容

第一章，*介绍 FastText*，介绍了 fastText 及其在 NLP 中的应用背景。本章将探讨创建该库的动机及其设计目标，阐明库的创建者希望为 NLP 及计算语言学领域带来的用途和好处。还将提供关于如何在工作计算机上安装 fastText 的具体说明。完成本章后，你将能够在自己的计算机上安装并运行 fastText。

第二章，*使用 FastText 命令行创建模型*，讨论了 fastText 库提供的强大命令行功能。本章介绍了默认的命令行选项，并展示了如何使用它创建模型。如果你只是对 fastText 有一个浅显的了解，那么阅读到本章应该已经足够。

第三章，*FastText 中的词表示*，解释了如何在 fastText 中创建无监督的词嵌入。

第四章，*FastText 中的句子分类*，介绍了支持 fastText 句子分类的算法。你还将了解 fastText 如何将大型模型压缩为较小的模型，从而可以部署到内存较小的设备上。

第五章，*Python 中的 FastText*，介绍了如何通过使用 fastText 官方的 Python 封装或使用 gensim 库来在 Python 中创建模型。gensim 是一个广受欢迎的 Python NLP 库。

第六章，*机器学习与深度学习模型*，解释了如何将 fastText 集成到你的 NLP 管道中，前提是你有预构建的管道，使用了统计机器学习或深度学习模型。对于统计机器学习，本章使用了 scikit-learn 库；而在深度学习方面，考虑了 Keras、TensorFlow 和 PyTorch。

第七章，*将模型部署到移动设备和 Web*，主要讲解如何将 fastText 模型集成到实时生产级客户应用中。

# 如何充分利用本书

理想情况下，你应该具备基本的 Python 代码编写和结构知识。如果你不熟悉 Python 或对编程语言的工作原理不清楚，那么请参考一本关于 Python 的书籍。从数据科学视角出发的 Python 书籍会对你更为适合。

如果你已经对 NLP 和机器学习有一定的基本了解，那么本书应该很容易理解。如果你是 NLP 初学者，只要愿意深入研究本书涉及的数学内容，这应该不会成为太大问题。我已经尽力解释本书涉及的数学概念，但如果你觉得还是太难，请随时联系我们并告知我们。

假设读者有深入探索并尝试所有代码的意愿。

# 下载示例代码文件

你可以从你的账户在[www.packtpub.com](http://www.packtpub.com)下载本书的示例代码文件。如果你从其他地方购买了本书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，文件将直接通过电子邮件发送给你。

你可以通过以下步骤下载代码文件：

1.  在[www.packtpub.com](http://www.packtpub.com/support)登录或注册。

1.  选择“SUPPORT”标签。

1.  点击“Code Downloads & Errata”。

1.  在搜索框中输入书名，并按照屏幕上的指示操作。

一旦文件下载完成，请确保使用以下最新版本的工具解压或提取文件夹：

+   Windows 平台的 WinRAR/7-Zip

+   Mac 平台的 Zipeg/iZip/UnRarX

+   Linux 平台的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址为[`github.com/PacktPublishing/fastText-Quick-Start-Guide`](https://github.com/PacktPublishing/fastText-Quick-Start-Guide)。如果代码有更新，它将会在现有的 GitHub 仓库中更新。

我们的其他书籍和视频代码包也可以通过**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到，欢迎查看！

# 本书中使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如：“诸如 `cat`、`grep`、`sed` 和 `awk` 等命令已经存在很长时间，其行为在互联网上有详细的文档。”

代码块按如下方式设置：

```py
import csv
import sys
w = csv.writer(sys.stdout)
for row in csv.DictReader(sys.stdin):
 w.writerow([row['stars'], row['text'].replace('\n', '')])
```

当我们希望您注意某个代码块中的特定部分时，相关的行或项会以粗体显示：

```py
import csv
import sys
w = csv.writer(sys.stdout)
for row in csv.DictReader(sys.stdin):
    w.writerow([row['stars'], row['text'].replace('\n', '')])
```

任何命令行输入或输出都按如下方式书写：

```py
$ cat data/yelp/yelp_review.csv | \
 python parse_yelp_dataset.py \
 > data/yelp/yelp_review.v1.csv
```

**粗体**：表示一个新术语、一个重要的词汇或您在屏幕上看到的词。例如，菜单或对话框中的词语会以这样的形式出现在文本中。这里是一个例子：“从管理面板中选择系统信息。”

警告或重要说明如下所示。

提示和技巧如下所示。

# 联系我们

我们欢迎读者的反馈。

**一般反馈**：请通过电子邮件 `feedback@packtpub.com` 并在邮件主题中提及书名。如果您对本书的任何方面有疑问，请通过电子邮件联系我们：`questions@packtpub.com`。

**勘误**：尽管我们已经尽力确保内容的准确性，但错误仍然可能发生。如果您发现本书中有错误，我们将非常感谢您向我们报告。请访问 [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接，并输入详细信息。

**盗版**：如果您在互联网上发现任何非法复制的我们作品的形式，我们将非常感谢您提供其位置地址或网站名称。请通过电子邮件 `copyright@packtpub.com` 联系我们，并提供该材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域具有专业知识，并且有兴趣撰写或为书籍贡献内容，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下评论。在您阅读并使用本书后，为什么不在您购买书籍的网站上留下评论呢？潜在读者可以看到并参考您的无偏见意见来做出购买决定，我们在 Packt 也能了解您对我们产品的看法，作者也可以看到您对其书籍的反馈。谢谢！

欲了解有关 Packt 的更多信息，请访问 [packtpub.com](https://www.packtpub.com/)。
