# 前言

TensorFlow 作为一个 **机器学习** (**ML**) 库，已经发展成为一个成熟的生产就绪生态系统。本书通过实际示例，帮助你使用最佳设置构建和部署 TensorFlow 模型，确保长期支持，而无需担心库的弃用或在修复漏洞和解决问题时被落下。

本书首先展示了如何优化你的 TensorFlow 项目并为企业级部署做好准备。接下来，你将学习如何选择 TensorFlow 的版本。随着学习的深入，你将了解如何通过遵循 TensorFlow Enterprise 提供的推荐实践，在稳定和可靠的环境中构建和部署模型。本书还将教你如何更好地管理你的服务，并提升 **人工智能** (**AI**) 应用的性能和可靠性。你将发现如何使用各种企业级服务来加速在 Google Cloud 上的 ML 和 AI 工作流。最后，你将学习如何扩展你的 ML 模型，并在 CPU、GPU 和云 TPUs 上处理大量工作负载。

在本书的学习结束时，你将掌握 TensorFlow Enterprise 模型开发、数据管道、训练和部署所需的模式。

# 本书的读者

本书面向数据科学家、机器学习开发者或工程师，以及希望从零开始学习和实现 TensorFlow Enterprise 提供的各种服务和功能的云计算从业者。具备基本的机器学习（ML）开发流程知识将非常有帮助。

# 本书内容

*第一章*，*TensorFlow Enterprise 概述*，介绍了如何在 **Google Cloud Platform** (**GCP**) 环境中设置和运行 TensorFlow Enterprise。这将让你获得初步的实践经验，了解 TensorFlow Enterprise 如何与 GCP 中的其他数据服务集成。

*第二章*，*在 Google AI 平台上运行 TensorFlow Enterprise*，介绍了如何使用 GCP 设置和运行 TensorFlow Enterprise。作为一个独特的 TensorFlow 发行版，TensorFlow Enterprise 可以在多个（但不是所有）GCP 平台上找到。为了确保正确的发行版得到配置，使用这些平台非常重要。

*第三章*，*数据准备与处理技术*，介绍了如何处理原始数据并将其格式化，使其能够独特地适应 TensorFlow 模型训练过程。我们将探讨一些关键的 TensorFlow Enterprise API，它们可以将原始数据转换为 Protobuf 格式以便高效流式传输，这是一种推荐的将数据输入训练过程的工作流。

*第四章*，*可重用模型和可扩展数据管道*，描述了 TensorFlow Enterprise 模型的构建或重用的不同方式。这些选项提供了灵活性，以适应构建、训练和部署 TensorFlow 模型的不同情境需求。掌握这些知识后，你将能够做出明智的选择，并理解不同模型开发策略之间的权衡。

*第五章*，*大规模训练*，阐述了使用 TensorFlow Enterprise 分布式训练策略将模型训练扩展到集群（无论是 GPU 还是 TPU）。这将使你能够构建一个稳健的模型开发和训练过程，并充分利用所有可用的硬件资源。

*第六章*，*超参数调优*，重点讲解了超参数调优，因为这是模型训练中必不可少的一部分，尤其是在构建自己的模型时。TensorFlow Enterprise 现在提供了高级 API 来支持先进的超参数空间搜索算法。通过本章内容，你将学习如何利用分布式计算能力，减少超参数调优所需的训练时间。

*第七章*，*模型优化*，探讨了你的模型是否足够精简高效。你的模型运行是否尽可能高效？如果你的使用场景需要模型在资源有限的情况下运行（如内存、模型大小或数据类型），例如在边缘设备或移动设备上的应用，那么就该考虑模型运行时优化了。本章讨论了通过 TensorFlow Lite 框架进行模型优化的最新方法。完成本章后，你将能够将训练好的 TensorFlow Enterprise 模型优化到尽可能轻量，以便进行推理。

*第八章*，*模型训练和性能的最佳实践*，聚焦于模型训练中两个普遍存在的方面：数据摄取和过拟合。首先，有必要建立一个数据摄取管道，无论训练数据的大小和复杂度如何，它都能正常工作。本章中，介绍和演示了使用 TensorFlow Enterprise 数据预处理管道的最佳实践和建议。其次，在处理过拟合时，讨论了正则化的标准做法，以及 TensorFlow 团队最近发布的一些正则化方法。

*第九章*，*部署 TensorFlow 模型*，介绍了将模型作为 Web 服务进行推理的基本知识。你将学习如何通过构建模型的 Docker 镜像，使用 TensorFlow Serving 来服务 TensorFlow 模型。在本章中，你将首先学习如何在本地环境中使用保存的模型。然后，你将以 TensorFlow Serving 为基础镜像，构建该模型的 Docker 镜像。最后，你将通过 Docker 容器暴露的 RESTful API，将此模型作为 Web 服务提供。

# 充分利用本书

拥有 Keras API 的基础知识和经验会非常有帮助，因为本书基于 2.x 以上版本的 TensorFlow，在该版本中，Keras API 已被官方支持并作为 `tf.keras` API 采用。此外，了解图像分类技术（卷积和多类分类）也会有所帮助，因为本书将图像分类问题作为介绍和解释 TensorFlow Enterprise 2 新功能的载体。另一个有用的工具是 GitHub。具备克隆 GitHub 仓库并浏览文件结构的基础经验，将有助于下载本书中的源代码。

从机器学习的角度来看，理解模型架构、特征工程过程和超参数优化的基本知识将非常有帮助。本书假设你已熟悉基本的 Python 数据结构，包括 NumPy 数组、元组和字典。

**如果你使用的是本书的数字版，我们建议你自己输入代码，或者通过 GitHub 仓库访问代码（链接在下一部分提供）。这样做有助于避免由于复制/粘贴代码而产生的潜在错误。**

# 下载示例代码文件

你可以通过 GitHub 下载本书的示例代码文件，地址：[`github.com/PacktPublishing/learn-tensorflow-enterprise/`](https://github.com/PacktPublishing/learn-tensorflow-enterprise/)。如果代码有更新，将会在现有的 GitHub 仓库中进行更新。

我们还提供了来自我们丰富的书籍和视频目录的其他代码包，访问地址：[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)。快来看看吧！

# 下载彩色图片

我们还提供了一个 PDF 文件，包含本书中使用的屏幕截图/图表的彩色图像。你可以从这里下载：`static.packt-cdn.com/downloads/9781800209145_ColorImages.pdf`

# 使用的约定

本书中使用了若干文本约定。

`Code in text`：表示文本中的代码词汇、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号。例如：‘就像 `lxterminal` 一样，我们也可以从这里运行 Linux 命令。’

代码块的设置如下：

```py
p2 = Person()
```

```py
p2.name = 'Jane'
```

```py
p2.age = 20
```

```py
print(p2.name)
```

```py
print(p2.age)
```

任何命令行输入或输出都按如下方式写出：

```py
sudo apt-get install xrdp -y
```

**粗体**：表示新术语、重要词汇或屏幕上显示的词语。例如，菜单或对话框中的词语会以这种方式出现在文本中。举个例子：“打开 Windows PC 上的**远程桌面连接**应用程序。”

提示或重要说明

以这种形式出现。

# 联系我们

我们始终欢迎读者的反馈。

`customercare@packtpub.com`。

**勘误**：虽然我们已经尽力确保内容的准确性，但错误是难以避免的。如果您在本书中发现了错误，欢迎向我们报告。请访问 [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择您的书籍，点击“勘误提交表单”链接并输入详细信息。

`copyright@packt.com`，并附带链接到该资料。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且有兴趣撰写或参与编写书籍，请访问 [authors.packtpub.com](http://authors.packtpub.com)。

评论

请留下评论。当您阅读并使用完本书后，不妨在您购买书籍的站点上留下评论。潜在读者可以根据您的公正评价做出购买决定，Packt 也能了解您对我们产品的看法，我们的作者也能看到您对他们书籍的反馈。谢谢！

欲了解更多关于 Packt 的信息，请访问 [packt.com](http://packt.com)。
