# 序言

当今各组织在全球范围内花费数十亿美元用于网络安全。**人工智能**（**AI**）已成为构建更智能、更安全的安全系统的伟大解决方案，使你能够预测和检测网络中的可疑活动，例如钓鱼攻击或未经授权的入侵。

本书展示并介绍了流行且成功的 AI 方法和模型，你可以采用这些方法来检测潜在的攻击并保护企业系统。你将理解**机器学习**（**ML**）、**神经网络**（**NNs**）和深度学习在网络安全中的作用，并学习如何在构建智能防御机制时注入 AI 能力。随着学习的深入，你将能够将这些策略应用于多种应用场景，包括垃圾邮件过滤、网络入侵检测、僵尸网络检测和安全身份验证。

本书结束时，你将准备好开发能够检测异常和可疑模式及攻击的智能系统，从而利用人工智能（AI）建立强大的网络安全防御。

# 本书适合谁阅读

如果你是网络安全专家或伦理黑客，想利用机器学习（ML）和人工智能（AI）的力量构建智能系统，你会发现这本书非常有用。

# 本书涵盖的内容

第一章，*网络安全专业人员的 AI 简介*，介绍了 AI 的各个分支，重点介绍了网络安全领域中自动化学习的各种方法的优缺点。本章还讨论了学习算法及其优化的不同策略。AI 的主要概念将在 Jupyter Notebooks 中展示并应用。本章使用的工具包括 Jupyter Notebooks、NumPy 和 scikit-learn，使用的数据集为 scikit-learn 数据集和 CSV 样本。

第二章，*为你的网络安全武器库设置 AI*，介绍了主要的软件需求及其配置。我们将学习如何通过恶意代码样本喂入知识库，以供 AI 算法处理。还将介绍 Jupyter Notebooks，方便交互式执行 Python 工具和命令。本章使用的工具有 Anaconda 和 Jupyter Notebooks，不涉及数据集。

第三章，***垃圾邮件还是正常邮件？利用 AI 检测电子邮件网络安全威胁*，涵盖了利用电子邮件作为攻击向量来检测电子邮件安全威胁的内容。从线性分类器和贝叶斯过滤器到更复杂的解决方案（如决策树、逻辑回归和**自然语言处理**（**NLP**）等），将会展示不同的检测策略。示例将使用 Jupyter Notebooks 以便读者更好地与不同解决方案进行互动。本章使用的工具包括 Jupyter Notebooks、scikit-learn 和 NLTK。本章所使用的数据集包括 Kaggle 垃圾邮件数据集、CSV 垃圾邮件样本和蜜罐钓鱼样本。

第四章，*恶意软件威胁检测*，介绍了恶意软件和勒索软件代码的广泛传播，以及同一类威胁在不同变种（多态性和形态变化恶意软件）中的快速多态变异，这使得基于特征码和图像文件哈希的传统检测解决方案已不再有效。常见的杀毒软件就是基于这些技术的。示例将展示使用机器学习算法的不同恶意软件分析策略。本章使用的工具包括 Jupyter Notebooks、scikit-learn 和 TensorFlow。本章所使用的数据集/样本包括 theZoo 恶意软件样本。

第五章，*基于 AI 的网络异常检测*，解释了当前不同设备之间的互联程度已经达到如此复杂的程度，以至于传统的概念，如边界安全性，的有效性产生了严重怀疑。事实上，在网络空间中，攻击面呈指数增长，因此，必须拥有自动化工具来检测网络异常并了解新的潜在威胁。本章使用的工具包括 Jupyter Notebooks、pandas、scikit-learn 和 Keras。本章所使用的数据集包括 Kaggle 数据集、KDD 1990、CIDDS、CICIDS2017、服务和 IDS 日志文件。

第六章，*用户身份验证安全*，介绍了人工智能在网络安全领域的应用，它在保护敏感的与用户相关的信息（包括用于访问其网络账户和应用程序的凭据）方面发挥着越来越重要的作用，以防止滥用，如身份盗窃。

第七章，*利用云 AI 解决方案预防欺诈*，介绍了企业遭受的许多安全攻击和数据泄露。这些泄露的目标是侵犯敏感信息，如客户的信用卡信息。这些攻击通常在隐蔽模式下进行，意味着通过传统方法很难检测到此类威胁。本章使用的工具有 IBM Watson Studio、IBM Cloud Object Storage、Jupyter Notebooks、scikit-learn、Apache Spark。此处使用的数据集是 Kaggle 信用卡欺诈检测数据集。

第八章，*GANs–攻击与防御*，介绍了**生成对抗网络**（**GANs**），这代表了深度学习为我们提供的最先进的神经网络示例。在网络安全的背景下，GANs 可以用于合法用途，如身份验证程序，但也可以被利用来侵犯这些程序。本章使用的工具有 CleverHans、**对抗机器学习**（**AML**）库、EvadeML-Zoo、TensorFlow 和 Keras。使用的数据集是完全由 GAN 生成的面部示例图像。

第九章，*评估算法*，展示了如何使用适当的分析指标评估各种替代解决方案的有效性。本章使用的工具有 scikit-learn、NumPy 和 Matplotlib。此处使用的 scikit 数据集。

第十章，*评估您的 AI 武器库*，涵盖了攻击者利用的技巧来避开工具。只有通过这种方式，才能获得解决方案有效性和可靠性的真实图景。此外，必须考虑与解决方案可扩展性相关的各个方面，并且要持续监控以保证可靠性。本章使用的工具有 scikit-learn、Foolbox、EvadeML、Deep-pwning、TensorFlow 和 Keras。此处使用的 MNIST 和 scikit 数据集。

# 为了充分利用这本书

为了最大限度地发挥这本书的作用，您需要熟悉网络安全概念，并掌握 Python 编程知识。

# 下载示例代码文件

您可以从您的[www.packt.com](http://www.packt.com)帐户下载本书的示例代码文件。如果您是在其他地方购买的这本书，您可以访问[www.packt.com/support](http://www.packt.com/support)并注册，将文件直接发送到您的邮箱。

您可以按照以下步骤下载代码文件：

1.  登录或注册到[www.packt.com](http://www.packt.com)。

1.  选择支持标签。

1.  点击代码下载和勘误。

1.  在搜索框中输入书名，并按照屏幕上的说明操作。

文件下载完成后，请确保使用最新版本的工具解压或提取文件夹：

+   Windows 上的 WinRAR/7-Zip

+   Mac 上的 Zipeg/iZip/UnRarX

+   Linux 上的 7-Zip/PeaZip

本书的代码包也托管在 GitHub 上，地址是[`github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity`](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)。如果代码有更新，现有的 GitHub 仓库将会进行更新。

我们还有来自丰富书籍和视频目录中的其他代码包可供下载，访问**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**查看！

# 下载彩色图片

我们还提供了包含本书中截图/图表的彩色图片的 PDF 文件，您可以在这里下载：[`www.packtpub.com/sites/default/files/downloads/9781789804027_ColorImages.pdf`](http://www.packtpub.com/sites/default/files/downloads/9781789804027_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`：表示文本中的代码词汇、数据库表名、文件夹名称、文件名、文件扩展名、路径名、虚拟网址、用户输入和 Twitter 账号名。以下是一个例子：“我们将使用的降维技术被称为主成分分析（PCA），并且可以在`scikit-learn`库中找到。”

代码块如下所示：

```py
import numpy as np
np_array = np.array( [0, 1, 2, 3] )

# Creating an array with ten elements initialized as zero
np_zero_array = np.zeros(10)
```

当我们希望您关注代码块的某一部分时，相关的行或项目会以粗体显示：

```py
[default]
exten => s,1,Dial(Zap/1|30)
exten => s,2,Voicemail(u100)
exten => s,102,Voicemail(b100)
exten => i,1,Voicemail(s0)
```

警告或重要提示如下所示。

提示和技巧以如下方式出现。

# 获取联系

我们始终欢迎读者的反馈。

**一般反馈**：如果您对本书的任何部分有疑问，请在邮件主题中提到书名，并通过`customercare@packtpub.com`与我们联系。

**勘误**：虽然我们已尽力确保内容的准确性，但错误仍然可能发生。如果您在本书中发现错误，我们将非常感激您向我们报告。请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击“勘误提交表单”链接并输入详细信息。

**盗版**：如果您在互联网上遇到任何形式的我们作品的非法复制品，我们将非常感激您提供该位置地址或网站名称。请通过`copyright@packt.com`联系我们，并附上相关材料的链接。

**如果您有兴趣成为作者**：如果您在某个领域有专业知识，并且有兴趣撰写或参与书籍的创作，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下评论。当您阅读并使用完本书后，为什么不在您购买本书的网站上留下评论呢？潜在读者可以看到并参考您公正的意见做出购买决定，我们在 Packt 也能了解您对我们产品的看法，我们的作者也能看到您对他们书籍的反馈。谢谢！

关于 Packt 的更多信息，请访问 [packt.com](http://www.packt.com/)。
