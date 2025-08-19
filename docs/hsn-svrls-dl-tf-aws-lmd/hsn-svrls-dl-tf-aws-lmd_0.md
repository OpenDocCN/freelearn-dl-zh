# 前言

本书将帮助你使用自己训练的模型与 AWS Lambda 配合，实现一种简化的无服务器计算方法，无需花费大量时间和金钱。到本书结尾时，你将能够实现一个展示 AWS Lambda 在提供 TensorFlow 模型服务中应用的项目。

此外，我们还将介绍深度学习和 TensorFlow 框架。我们将探讨如何训练神经网络，但更重要的是，我们将讲解如何在应用中使用预训练的神经网络以及如何找到它们。之后，我们将深入了解如何使用无服务器方法部署深度学习应用程序。我们还将讨论它们的优点、缺点、可能的限制和最佳实践。

接下来，我们将构建多个应用程序，利用无服务器深度学习方法。我们将创建一个规划 API，了解 AWS API Gateway 服务，并探讨如何以方便的方式部署所有内容。在后续阶段，我们将创建一个深度学习管道和 AWS 简单查询服务。我们将探讨如何与 AWS Lambda 一起使用它，并展示如何部署该应用程序。

# 本书适合的读者

本课程将帮助数据科学家学习如何轻松部署模型，也适合那些希望了解如何将应用部署到云端的初学者。无需具备 TensorFlow 或 AWS 的先前知识。

# 本书涵盖的内容

第一章，*从无服务器计算与 AWS Lambda 开始*，介绍了我们计划要看的所有示例。我们还将描述无服务器概念，以及它如何改变当前的云基础设施环境。最后，我们将看到无服务器深度学习如何使我们能够比传统部署技术更容易实现项目，同时具有相同的可扩展性和低成本。

第二章，*使用 AWS Lambda 函数开始部署*，介绍了 AWS Lambda，并解释了如何创建 AWS 账户。我们将创建第一个 Lambda，并了解如何使用 Serverless Framework 轻松部署它。

第三章，*部署 TensorFlow 模型*，介绍了 TensorFlow 框架，并展示了几个如何训练和导出模型的示例。此外，我们将查看任何人都可以用于自己任务的预训练模型库。最后，我们将展示如何导入项目中所需的预训练模型。

第四章，*在 AWS Lambda 上使用 TensorFlow*，深入探讨了如何开始使用无服务器 TensorFlow。我们还将了解无服务器 TensorFlow 在成本、规模和速度方面与传统部署的小细节差异。我们还将查看如何使用标准的 AWS UI 开始，并了解如何使用 Serverless Framework 完成相同的工作。

第五章，*创建深度学习 API*，解释了如何制作深度学习 REST API。然后我们将介绍 AWS API Gateway 并学习如何使用两种方法制作应用程序：AWS UI 和 Serverless Framework。

第六章，*创建深度学习管道*，解释了如何制作深度学习管道应用程序。我们将介绍 AWS SQS，并说明如何使用两种方法制作应用程序：AWS UI 和 Serverless Framework。

第七章，*创建深度学习工作流程*，解释了如何制作复杂的深度学习算法应用程序。我们将介绍 AWS Step Functions，并说明如何使用两种方法制作应用程序：AWS UI 和 Serverless Framework。

# 要充分利用这本书

需要基本的 AWS Lambda 和 Python 知识才能充分利用这本书。

# 下载示例代码文件

您可以从您在 [www.packt.com](http://www.packt.com) 的帐户下载本书的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [www.packt.com/support](http://www.packt.com/support) 并注册，直接将文件发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册 [www.packt.com](http://www.packt.com)。

1.  选择支持选项卡。

1.  点击下载代码和勘误。

1.  在搜索框中输入书名并按照屏幕上的说明操作。

一旦文件下载完成，请确保使用最新版本的解压或提取文件夹：

+   WinRAR/7-Zip 适用于 Windows

+   Zipeg/iZip/UnRarX 适用于 Mac

+   7-Zip/PeaZip 适用于 Linux

本书的代码包也托管在 GitHub 上，网址为 [`github.com/PacktPublishing/Hands-On-Serverless-Deep-Learning-with-TensorFlow-and-AWS-Lambda`](https://github.com/PacktPublishing/Hands-On-Serverless-Deep-Learning-with-TensorFlow-and-AWS-Lambda)。如果代码有更新，将在现有的 GitHub 仓库中更新。

我们还提供来自丰富书籍和视频目录的其他代码包，都可在 **[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)** 上查看！

# 下载彩色图片

我们还提供了一个 PDF 文件，包含本书中使用的带有彩色截图/图表的版本。您可以在此下载： [`www.packtpub.com/sites/default/files/downloads/9781838551605_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781838551605_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码单词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号。这里有一个例子：“在 `serverless.yml` 版本中，包含了函数的名称、可用资源和区域。”

一块代码设置如下：

```py
model.fit(x_train, y_train, epochs=2)
print('Evaluation:')
print(model.evaluate(x_test, y_test))
```

任何命令行输入或输出的格式如下：

```py
npm install -g Serverless
serverless --version

```

**粗体**：表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词通常以这种方式显示。这里有一个例子：“接下来，进入用户页面并点击添加用户。”

警告或重要提示通常显示为这样的格式。

提示和技巧通常显示为这样的格式。

# 与我们联系

我们非常欢迎读者的反馈。

**一般反馈**：如果您对本书的任何方面有疑问，请在邮件主题中注明书名，并通过电子邮件与我们联系，地址是 `customercare@packtpub.com`。

**勘误表**：尽管我们已尽力确保内容的准确性，但错误仍可能发生。如果您在本书中发现任何错误，我们将非常感激您向我们报告。请访问 [www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并填写相关细节。

**盗版**：如果您在互联网上发现我们作品的任何非法复制品，我们将非常感激您提供其地址或网站名称。请通过电子邮件联系我们，地址是 `copyright@packt.com`，并附上该材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域拥有专长并且有意写作或为书籍做贡献，请访问 [authors.packtpub.com](http://authors.packtpub.com/)。

# 评价

请留下您的评价。阅读并使用本书后，您不妨在购买书籍的网站上留下评价。潜在读者可以查看您的公正意见，帮助他们做出购买决定，我们 Packt 也能了解您对我们的产品的看法，而我们的作者也可以看到您对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
