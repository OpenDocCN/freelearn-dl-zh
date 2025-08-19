# 前言

《*人工智能实例*》第二版将带领你了解当前**人工智能**（**AI**）的主要方面，甚至超越这些内容！

本书包含了对第一版中描述的 AI 关键内容的许多修订和新增：

+   机器学习和深度学习的理论，包括混合算法和集成算法。

+   主要 AI 算法的数学表示，包括自然语言解释，使其更容易理解。

+   真实案例研究，带领读者深入了解电子商务的核心：制造、服务、仓储和配送。

+   介绍将物联网、**卷积神经网络**（**CNN**）和**马尔可夫决策过程**（**MDP**）结合的 AI 解决方案。

+   许多开源 Python 程序，特别关注 TensorFlow 2.x、TensorBoard 和 Keras 的新功能。使用了许多模块，如 scikit-learn、pandas 等。

+   云平台：Google Colaboratory 及其免费的虚拟机、Google Translate、Google Dialogflow、IBM Q 量子计算平台等。

+   利用**限制玻尔兹曼机**（**RBM**）和**主成分分析**（**PCA**）的强大功能，生成数据以创建有意义的聊天机器人。

+   补偿聊天机器人情感缺陷的解决方案。

+   遗传算法，在特定情况下比经典算法运行更快，并且在混合深度学习神经网络中使用的遗传算法。

+   神经形态计算，通过选择性脉冲神经元群体的模型再现我们的大脑活动，这些模型模拟了我们的生物反应。

+   量子计算，将带你深入了解量子比特的巨大计算能力和认知表示实验。

《*人工智能实例*》第二版将带你走在 AI 的最前沿，并通过创新提升现有解决方案。本书将使你不仅成为 AI 专家，还能成为一个有远见的人。你将发现如何作为顾问、开发人员、教授、好奇心驱动者或任何参与人工智能的人，提升你的 AI 技能。

# 本书适合谁阅读

本书提供了 AI 的广泛视角，AI 正在扩展到我们生活的各个领域。

主要的机器学习和深度学习算法通过来自数百个 AI 项目和实现的真实 Python 示例来讲解。

每个 AI 实现都通过一个开源程序进行说明，该程序可以在 GitHub 和 Google Colaboratory 等云平台上获取。

《*人工智能实例*第二版》适合那些希望构建坚实的机器学习程序的开发人员，这些程序将优化生产站点、服务、物联网等。

项目经理和顾问将学习如何构建输入数据集，帮助读者应对现实生活中的 AI 挑战。

教师和学生将全面了解 AI 的关键方面，并结合许多教育示例。

*人工智能实例*，第二版将帮助任何对 AI 感兴趣的人理解如何构建稳固、高效的 Python 程序。

# 本书内容概述

*第一章*，*通过强化学习入门下一代人工智能*，讲解了基于 MDP 的 Bellman 方程的强化学习。案例研究描述了如何解决带有人类驾驶员和自动驾驶车辆的送货路线问题。本章展示了如何在 Python 中从零开始构建 MDP。

*第二章*，*构建奖励矩阵—设计你的数据集*，展示了神经网络的架构，从 McCulloch-Pitts 神经元开始。案例研究描述了如何使用神经网络构建 Bellman 方程在仓库环境中的奖励矩阵。此过程将在 Python 中使用逻辑回归、softmax 和 one-hot 函数进行开发。

*第三章*，*机器智能——评估函数与数值收敛*，展示了机器评估能力如何超越人类决策。案例研究描述了一个国际象棋棋局，以及如何将 AI 程序的结果应用于决策优先级。Python 中决策树的介绍展示了如何管理决策过程。

*第四章*，*通过 K-Means 聚类优化解决方案*，介绍了一个基于 Lloyd 算法的 k-means 聚类程序，以及如何将其应用于自动引导车的优化。k-means 聚类程序的模型将被训练并保存。

*第五章*，*如何使用决策树增强 K-Means 聚类*，从 k-means 聚类的无监督学习开始。k-means 聚类算法的输出将为监督式决策树算法提供标签。将介绍随机森林算法。

*第六章*，*通过 Google 翻译创新 AI*，解释了革命性创新与颠覆性创新的区别。将描述 Google 翻译，并通过一个创新的基于 k 近邻的 Python 程序进行增强。

*第七章*，*通过朴素贝叶斯优化区块链*，讲述了区块链挖掘，并描述了区块链的工作原理。我们使用朴素贝叶斯方法通过预测交易来优化**供应链管理**（**SCM**）区块链的区块，以预测存储水平。

*第八章*，*用前馈神经网络解决 XOR 问题*，讲述了从零开始构建**前馈神经网络**（**FNN**）以解决 XOR 线性可分性问题。案例研究描述了如何为工厂分组订单。

*第九章*，*使用卷积神经网络（CNN）进行抽象图像分类*，详细描述了 CNN：卷积核、形状、激活函数、池化、展平和全连接层。案例研究展示了在食品加工公司中使用 Webcam 通过 CNN 进行传送带上的物品识别。

*第十章*，*概念表示学习*，解释了**概念表示学习**（**CRL**），这是一种通过 CNN 转换为**CRL 元模型**（**CRLMM**）来解决生产流程的创新方法。案例研究展示了如何利用 CRLMM 进行转移学习和领域学习，将模型扩展到其他应用。

*第十一章*，*结合强化学习和深度学习*，将 CNN 与 MDP 结合起来构建解决方案，用于自动规划和调度优化器与基于规则的系统。

这种解决方案应用于服装制造业，展示了如何将 AI 应用于现实生活系统。

*第十二章*，*人工智能和物联网（IoT）*，探索了一个与 CNN 组合的**支持向量机**（**SVM**）。案例研究展示了自动驾驶车辆如何自动找到可用的停车位。

*第十三章*，*使用 TensorFlow 2.x 和 TensorBoard 可视化网络*，提取了 CNN 每一层的信息，并展示了神经网络所采取的中间步骤。每一层的输出包含了应用的转换图像。

*第十四章*，*使用受限玻尔兹曼机（RBM）和主成分分析（PCA）准备聊天机器人输入*，解释了如何利用 RBM 和 PCA 生成宝贵信息，将原始数据转化为聊天机器人输入数据。

*第十五章*，*建立认知 NLP UI/CUI 聊天机器人*，描述了如何从零开始构建一个 Google Dialogflow 聊天机器人，利用 RBM 和 PCA 算法提供的信息。该聊天机器人将包含实体、意图和有意义的响应。

*第十六章*，*改进聊天机器人情感智能不足*，解释了聊天机器人在处理人类情绪时的局限性。Dialogflow 的**情感**选项将被激活，同时结合**Small Talk**，使聊天机器人更加友好。

*第十七章*，*混合神经网络中的遗传算法*，深入探讨了我们的染色体，找到了我们的基因，并帮助你理解我们的繁殖过程如何工作。从这里开始，展示了如何在 Python 中实现进化算法，一个**遗传算法**（**GA**）。混合神经网络将展示如何使用遗传算法优化神经网络。

*第十八章*，*神经形态计算*，描述了神经形态计算的概念，然后探讨了 Nengo，一个独特的神经形态框架，提供了坚实的教程和文档。

这种神经形态的概述将带你进入我们大脑结构解决复杂问题的强大力量。

*第十九章*，*量子计算*，将展示量子计算机优于经典计算机，介绍了量子比特是什么，如何使用它，以及如何构建量子电路。量子门和示例程序的介绍将带你进入量子力学未来世界。

*附录*，*问题的答案*，提供了所有章节中*问题*部分列出的问题的答案。

# 要从本书中获得最大收益

人工智能项目依赖于三个因素：

+   **理解人工智能项目应用的主题**。为此，可以通读一章，掌握关键思想。一旦理解了书中案例研究的关键概念，尝试思考如何将 AI 解决方案应用于你周围的现实生活例子。

+   **AI 算法的数学基础**。如果你有精力，别跳过数学公式。人工智能深深依赖数学。许多优秀的网站解释了本书中使用的数学。

+   **开发**。人工智能解决方案可以直接在 Google 等在线云平台的机器学习网站上使用。我们可以通过 API 访问这些平台。在本书中，多次使用了 Google Cloud。尝试创建你自己的帐户，探索多个云平台，了解它们的潜力和局限性。开发对于 AI 项目依然至关重要。

即使使用云平台，脚本和服务也是必需的。此外，有时编写算法是必须的，因为现成的在线算法无法解决某些特定问题。探索本书附带的程序。它们是开源的，且免费。

## 技术要求

以下是运行本书代码所需的部分技术要求的非详尽列表。欲了解更详细的章节要求，请参考此链接：[`github.com/PacktPublishing/Artificial-Intelligence-By-Example-Second-Edition/blob/master/Technical%20Requirements.csv`](https://github.com/PacktPublishing/Artificial-Intelligence-By-Example-Second-Edition/blob/master/Technical%20Requirements.csv)。

| **Package** | **Website** |
| --- | --- |
| Python | [`www.python.org/`](https://www.python.org/) |
| NumPy | [`pypi.org/project/numpy/`](https://pypi.org/project/numpy/) |
| Matplotlib | [`pypi.org/project/matplotlib/`](https://pypi.org/project/matplotlib/) |
| pandas | [`pypi.org/project/pandas/`](https://pypi.org/project/pandas/) |
| SciPy | [`pypi.org/project/scipy/`](https://pypi.org/project/scipy/) |
| scikit-learn | [`pypi.org/project/scikit-learn/`](https://pypi.org/project/scikit-learn/) |
| PyDotPlus | [`pypi.org/project/pydotplus/`](https://pypi.org/project/pydotplus/) |
| Google API | [`developers.google.com/docs/api/quickstart/python`](https://developers.google.com/docs/api/quickstart/python) |
| html | [`pypi.org/project/html/`](https://pypi.org/project/html/) |
| TensorFlow 2 | [`pypi.org/project/tensorflow/`](https://pypi.org/project/tensorflow/) |
| Keras | [`pypi.org/project/Keras/`](https://pypi.org/project/Keras/) |
| Pillow | [`pypi.org/project/Pillow/`](https://pypi.org/project/Pillow/) |
| Imageio | [`pypi.org/project/imageio/`](https://pypi.org/project/imageio/) |
| Pathlib | [`pypi.org/project/pathlib/`](https://pypi.org/project/pathlib/) |
| OpenCV-Python | [`pypi.org/project/opencv-python/`](https://pypi.org/project/opencv-python/) |
| Google Dialogflow | [`dialogflow.com/`](https://dialogflow.com/) |
| DEAP | [`pypi.org/project/deap/`](https://pypi.org/project/deap/) |
| bitstring | [`pypi.org/project/bitstring/`](https://pypi.org/project/bitstring/) |
| nengo | [`pypi.org/project/nengo/`](https://pypi.org/project/nengo/) |
| nengo-gui | [`pypi.org/project/nengo-gui/`](https://pypi.org/project/nengo-gui/) |
| IBM Q | [`www.research.ibm.com/ibm-q/`](https://www.research.ibm.com/ibm-q/) |
| Quirk | [`algassert.com/2016/05/22/quirk.html`](http://algassert.com/2016/05/22/quirk.html) |

## 下载示例代码文件

您可以通过在[www.packt.com/](http://www.packt.com/)的账户中下载本书的示例代码文件。如果您在其他地方购买了此书，可以访问[www.packtpub.com/support](http://www.packtpub.com/support)并注册，以便直接将文件通过电子邮件发送给您。

您可以通过以下步骤下载代码文件：

1.  登录或注册到[`www.packt.com`](http://www.packt.com)。

1.  选择**支持**标签。

1.  点击**代码下载**。

1.  在**搜索**框中输入书名并按照屏幕上的说明操作。

下载文件后，请确保使用最新版本的以下工具解压或提取文件夹：

+   WinRAR / 7-Zip for Windows

+   Zipeg / iZip / UnRarX for Mac

+   7-Zip / PeaZip for Linux

本书的代码包也托管在 GitHub 上，网址是[`github.com/PacktPublishing/Artificial-Intelligence-By-Example-Second-Edition`](https://github.com/PacktPublishing/Artificial-Intelligence-By-Example-Second-Edition)。如果代码有更新，它将在现有的 GitHub 仓库中更新。

我们还拥有来自我们丰富书籍和视频目录中的其他代码包，您可以在[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)查看。快来看看吧！

## 下载彩色图像

我们还提供了一个 PDF 文件，里面包含了本书中使用的截图/图表的彩色图像。您可以在此下载：[`static.packt-cdn.com/downloads/9781839211539_ColorImages.pdf`](https://static.packt-cdn.com/downloads/9781839211539_ColorImages.pdf)。

## 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 用户名。例如：“决策树程序`decision_tree.py`读取 KMC 预测结果`ckmc.csv`的输出。”

代码块设置如下：

```py
# load dataset
col_names = ['f1', 'f2','label']
df = pd.read_csv("ckmc.csv", header=None, names=col_names)
if pp==1:
    print(df.head()) 
```

当我们希望特别提醒您注意代码块中的某一部分时，相关的行或项会以粗体显示：

```py
for i in range(0,1000):
    xf1=dataset.at[i,'Distance']
    xf2=dataset.at[i,'location']
    X_DL = [[xf1,xf2]]
    prediction = kmeans.predict(X_DL) 
```

所有命令行输入或输出按如下方式编写：

```py
Selection: BnVYkFcRK Fittest: 0 This generation Fitness: 0 Time Difference: 0:00:00.000198 
```

**粗体**：表示一个新术语、重要的词汇，或者在屏幕上看到的词汇，例如菜单或对话框中的内容，也会以这种形式出现在文本中。例如：“当您点击**保存**时，**情感**进度条会跳动。”

警告或重要说明如下所示。

小贴士和技巧如下所示。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何部分有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`与我们联系。

**勘误**：虽然我们已尽最大努力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，感谢您向我们报告。请访问[www.packtpub.com/support/errata](http://www.packtpub.com/support/errata)，选择您的书籍，点击勘误提交表单链接，并填写详细信息。

**盗版**：如果您在互联网上遇到我们作品的任何非法副本，感谢您提供相关位置地址或网站名称。请通过`copyright@packt.com`与我们联系，并附上相关材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域具有专业知识，且有兴趣撰写或参与编写书籍，请访问[authors.packtpub.com](http://authors.packtpub.com)。

## 评论

请留下评论。阅读并使用本书后，您可以在购买网站上留下评论。潜在读者可以根据您的客观意见做出购买决策，我们也可以了解您对我们产品的看法，而我们的作者则可以看到您对其书籍的反馈。谢谢！

有关 Packt 的更多信息，请访问[packt.com](http://packt.com)。
