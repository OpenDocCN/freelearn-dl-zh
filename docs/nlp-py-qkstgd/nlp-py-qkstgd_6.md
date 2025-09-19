# 第六章：深度学习在自然语言处理中的应用

在上一章中，我们使用经典的机器学习技术来构建我们的文本分类器。在本章中，我们将通过使用**循环神经网络**（**RNN**）来替换这些技术。

特别是，我们将使用一个相对简单的双向LSTM模型。如果你对此不熟悉，请继续阅读——如果你已经熟悉，请随意跳过！

批变量数据集属性应指向`torchtext.data.TabularData`类型的`trn`变量。这是理解训练深度学习模型中数据流差异的有用检查点。

让我们先谈谈被过度炒作的术语，即深度学习中的“深度”和深度神经网络中的“神经”。在我们这样做之前，让我们花点时间解释为什么我使用PyTorch，并将其与Tensorflow和Keras等其他流行的深度学习框架进行比较。

为了演示目的，我将构建尽可能简单的架构。让我们假设大家对循环神经网络（RNN）有一般的了解，不再重复介绍。

在本章中，我们将回答以下问题：

+   深度学习是什么？它与我们所看到的不同在哪里？

+   任何深度学习模型中的关键思想是什么？

+   为什么选择PyTorch？

+   我们如何使用torchtext对文本进行标记化并设置数据加载器？

+   什么是循环网络，我们如何使用它们进行文本分类？

# 深度学习是什么？

深度学习是机器学习的一个子集：一种从数据中学习的新方法，它强调学习越来越有意义的表示的连续层。但深度学习中的“深度”究竟是什么意思呢？

<q>"深度学习中的‘深度’并不是指通过这种方法获得的任何更深层次的理解；相反，它代表的是这种连续层表示的想法。”</q>

– Keras的主要开发者F. Chollet

模型的深度表明我们使用了多少层这样的表示。F Chollet建议将分层表示学习和层次表示学习作为更好的名称。另一个可能的名称是可微分编程。

“可微分编程”这个术语是由Yann LeCun提出的，源于我们的深度学习方法共有的不是更多的层——而是所有这些模型都通过某种形式微分计算来学习——通常是随机梯度下降。

# 现代机器学习方法的差异

我们所研究的现代机器学习方法主要在20世纪90年代成为主流。它们之间的联系在于它们都使用一层表示。例如，决策树只创建一组规则并应用它们。即使你添加集成方法，集成通常也很浅，只是直接结合几个机器学习模型。

这里是对这些差异的更好表述：

“现代深度学习通常涉及十几个甚至上百个连续的表示层——而且它们都是通过接触训练数据自动学习的。与此同时，其他机器学习的方法倾向于只学习数据的一层或两层表示；因此，它们有时被称为浅层学习。”

– F Chollet

让我们看看深度学习背后的关键术语，因为这样我们可能会遇到一些关键思想。

# 理解深度学习

以宽松的方式来说，机器学习是关于将输入（如图像或*电影评论*）映射到目标（如标签猫或*正面*）。模型通过查看（或从多个输入和目标对进行训练）来完成这项工作。

深度神经网络通过一系列简单数据转换（层）来实现从输入到目标的映射。这个序列的长度被称为网络的深度。从输入到目标的整个序列被称为学习数据的模型。这些数据转换是通过重复观察示例来学习的。让我们看看这种学习是如何发生的。

# 拼图碎片

我们正在研究一个特定的子类挑战，我们想要学习一个输入到目标的映射。这个子类通常被称为监督机器学习。这个词监督表示我们为每个输入都有一个目标。无监督机器学习包括尝试聚类文本等挑战，我们并没有目标。

要进行任何监督机器学习，我们需要以下条件：

+   **输入数据**：从过去的股票表现到你的度假照片

+   **目标**：期望输出的示例

+   **衡量算法是否做得好的方法**：这是确定算法当前输出与其期望输出之间距离所必需的

上述组件对于任何监督方法都是通用的，无论是机器学习还是深度学习。特别是深度学习有其自己的一套令人困惑的因素：

+   模型本身

+   损失函数

+   优化器

由于这些演员是新来的，让我们花一分钟时间了解他们做什么。

# 模型

每个模型由几个层组成。每个层是一个数据转换。这种转换通过一串数字来捕捉，称为层权重。但这并不是完全的真理，因为大多数层通常与一个数学运算相关联，例如卷积或仿射变换。一个更精确的观点是，层是通过其权重**参数化**的。因此，我们交替使用术语*层参数*和*层权重*。

所有层权重共同的状态构成了模型状态，即模型权重。一个模型可能有几千到几百万个参数。

让我们尝试理解在这个背景下模型**学习**的概念：学习意味着找到网络中所有层的权重值，以便网络能够正确地将示例输入映射到其相关目标。

注意，这个值集是针对一个地方的*所有层*。这个细微差别很重要，因为改变一个层的权重可能会改变整个模型的行为和预测。

# 损失函数

在设置机器学习任务时使用的组件之一是评估模型的表现。最简单的答案就是衡量模型的概念性准确度。虽然准确度有几个缺点：

+   准确度是一个与验证数据相关联的代理指标，而不是训练数据。

+   准确度衡量我们有多正确。在训练过程中，我们希望衡量我们的模型预测与目标有多远。

这些差异意味着我们需要一个不同的函数来满足我们之前的标准。在深度学习的背景下，这由损失函数来实现。这有时也被称为目标函数。

<q>"损失函数将网络的预测和真实目标（你希望网络输出的内容）计算出一个距离分数，捕捉网络在这个特定例子上的表现如何。</q>"

<q>- 来自F Chollet的《Python深度学习》</q>

这种距离测量被称为损失分数，或简单地称为损失。

# 优化器

这个损失值自动用作反馈信号来调整算法的工作方式。这个调整步骤就是我们所说的学习。

这种在模型权重上的自动调整是深度学习特有的。每次权重调整或*更新*都是朝着降低当前训练对（输入，目标）的损失的方向进行的。

这种调整是优化器的任务，它实现了所谓的反向传播算法：深度学习的核心算法。

优化器和损失函数是所有深度学习方法的共同点——即使我们没有输入/目标对。所有优化器都基于微分计算，如**随机梯度下降**（**SGD**）、Adam等。因此，在我看来，*可微分编程*是深度学习的一个更精确的名称。

# 将所有这些放在一起——训练循环

我们现在有一个共享的词汇表。你对诸如层、模型权重、损失函数和优化器等术语有一个概念性的理解。但它们是如何协同工作的？我们如何对任意数据进行训练？我们可以训练它们，使它们能够识别猫的图片或亚马逊上的欺诈评论。

这里是训练循环内部发生步骤的大致轮廓：

+   初始化：

    +   网络或模型权重被分配随机值，通常形式为(-1, 1)或(0, 1)。

    +   模型与目标相差甚远。这是因为它只是在执行一系列随机变换。

    +   损失值非常高。

+   在网络处理每个示例时，都会发生以下情况：

    +   权重在正确的方向上略有调整

    +   损失得分降低

这就是训练循环，它会被重复多次。整个训练集的每一次遍历通常被称为**一个epoch**。适用于深度学习的每个训练集通常应该有数千个示例。模型有时会训练数千个epoch，或者也可以说是数百万次的**迭代**。

在训练设置（模型、优化器、循环）中，前面的循环更新了最小化损失函数的权重值。一个训练好的网络是在整个训练和验证数据上具有可能最小损失得分的网络。

这是一个简单的机制，当经常重复时，就像魔法一样起作用。

# Kaggle – 文本分类挑战

在这个特定的部分，我们将访问熟悉的文本分类任务，但使用不同的数据集。我们将尝试解决[Jigsaw有毒评论分类挑战](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)。

# 获取数据

注意，您需要接受比赛的条款和条件以及数据使用条款才能获取此数据集。

对于直接下载，您可以从挑战网站上的[数据标签](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)获取训练和测试数据。

或者，您也可以使用官方的Kaggle API ([github链接](https://github.com/Kaggle/kaggle-api))通过终端或Python程序下载数据。

在直接下载和Kaggle API的情况下，您必须将训练数据分割成更小的训练和验证分割，以便在这个笔记本中使用。

您可以使用以下方法创建训练数据的训练和验证分割：

`sklearn.model_selection.train_test_split` 工具。或者，您也可以直接从本书的配套代码仓库中下载。

# 探索数据

如果您有任何缺失的包，您可以通过以下命令从笔记本本身安装它们：

```py
# !conda install -y pandas
# !conda install -y numpy
```

让我们把导入的部分先放一放：

```py
import pandas as pd
import numpy as np
```

然后，将训练文件读取到pandas DataFrame中：

```py
train_df = pd.read_csv("data/train.csv")
train_df.head()
```

我们得到了以下输出：

|  | id | `comment_text` | `toxic` | `severe_toxic` | `obscene` | `threat` | `insult` | `identity_hate` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0000997932d777bf | 解释`\r\n为什么`在我使用下进行的编辑... | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 000103f0d9cfb60f | D'aww! 他匹配了这个背景颜色我正在用的... | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 000113f07ec002fd | 嘿，伙计，我真的不是在尝试编辑战争。我... | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0001b41b1c6bb37e | `\r\n更多\r\n` 我无法提出任何真正的建议... | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0001d958c54c6e35 | 先生，您是我的英雄。您还记得...吗？ | 0 | 0 | 0 | 0 | 0 | 0 |

让我们读取验证数据并预览一下：

```py
val_df = pd.read_csv("data/valid.csv")
val_df.head()
```

我们得到了以下输出：

|  | id | `comment_text` | `toxic` | `severe_toxic` | `obscene` | `threat` | `insult` | `identity_hate` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 000eefc67a2c930f | Radial symmetry `\r\n\r\n` Several now extinct li... | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 000f35deef84dc4a | There's no need to apologize. A Wikipedia arti... | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 000ffab30195c5e1 | Yes, because the mother of the child in the ca... | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0010307a3a50a353 | `\r\nOk`. But it will take a bit of work but I ... | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0010833a96e1f886 | `== A barnstar` for you! `==\r\n\r\n` The Real L... | 0 | 0 | 0 | 0 | 0 | 0 |

# 多个目标数据集

这个数据集有趣的地方在于每个评论可以有多个标签。例如，一个评论可能是侮辱性和有害的，或者它可能是淫秽的，并包含`identity_hate`元素。

因此，我们在这里通过尝试一次预测多个标签（例如正面或负面）而不是一个标签来提升水平。对于每个标签，我们将预测一个介于0和1之间的值，以表示它属于该类别的可能性。

这不是一个贝叶斯意义上的概率值，但表示了相同的意图。

我建议尝试使用这个数据集之前看到的模型，并重新实现这段代码以适应我们最喜欢的IMDb数据集。

让我们用同样的想法预览一下测试数据集：

```py
test_df = pd.read_csv("data/test.csv")
test_df.head()
```

我们得到了以下输出：

|  | `id` | `comment_text` |
| --- | --- | --- |
| 0 | 00001cee341fdb12 | Yo bitch Ja Rule is more succesful then you'll... |
| 1 | 0000247867823ef7 | `== From RfC == \r\n\r\n` The title is fine as i... |
| 2 | 00013b17ad220c46 | `\r\n\r\n == Sources == \r\n\r\n *` Zawe Ashto... |
| 3 | 00017563c3f7919a | If you have a look back at the source, the in... |
| 4 | 00017695ad8997eb | I don't anonymously edit articles at all. |

这个预览确认了我们面临的是一个文本挑战。这里的重点是文本的语义分类。测试数据集没有为目标列提供空标题或列，但我们可以从训练数据框中推断它们。

# 为什么选择PyTorch？

PyTorch是Facebook的一个深度学习框架，类似于Google的TensorFlow。

由于有Google的支持，数千美元被用于TensorFlow的市场营销、开发和文档。它几乎一年前就达到了稳定的1.0版本，而PyTorch最近才达到0.4.1。这意味着通常更容易找到TensorFlow的解决方案，你也可以从互联网上复制粘贴代码。

另一方面，PyTorch对程序员友好。它在语义上与NumPy和深度学习操作相似。这意味着我可以使用我已经熟悉的Python调试工具。

**Pythonic**：TensorFlow在某种程度上像C程序一样工作，因为代码都是在一次会话中编写的，编译后执行，从而完全破坏了其Python风格。这已经被TensorFlow的Eager Execution功能发布所解决，该功能很快将稳定到足以用于大多数原型设计工作。

**训练循环可视化**：直到不久前，TensorFlow有一个很好的可视化工具TensorBoard，用于理解训练和验证性能（以及其他特性），而PyTorch没有。现在，tensorboardX使得TensorBoard与PyTorch的使用变得简单。

简而言之，我推荐使用PyTorch，因为它更容易调试，更符合Python风格，并且对程序员更友好。

# PyTorch和torchtext

你可以通过conda或pip在你的目标机器上安装Pytorch的最新版本([网站](https://pytorch.org/))。我在这台带有GPU的Windows笔记本电脑上运行此代码。

我使用`conda install pytorch cuda92 -c pytorch`安装了torch。

对于安装`torchtext`，我建议直接从他们的GitHub仓库使用pip进行安装，以获取最新的修复，而不是使用PyPi，因为PyPi更新并不频繁。在第一次运行此笔记本时取消注释该行：

```py
# !pip install --upgrade git+https://github.com/pytorch/text
```

让我们设置`torch`、`torch.nn`（用于建模）和`torchtext`的导入：

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
```

如果你在一台带有GPU的机器上运行此代码，请将`use_gpu`标志设置为`True`；否则，设置为`False`。

如果你设置`use_gpu=True`，我们将使用`torch.cuda.is_available()`实用程序检查GPU是否可供PyTorch使用：

```py
use_gpu = True
if use_gpu:
    assert torch.cuda.is_available(), 'You either do not have a GPU or is not accessible to PyTorch'
```

让我们看看这台机器上有多少个GPU设备可供PyTorch使用：

```py
torch.cuda.device_count()
> 1
```

# 使用torchtext的数据加载器

在大多数深度学习应用中，编写良好的数据加载器是最繁琐的部分。这一步骤通常结合了我们在前面看到的预处理、文本清理和向量化任务。

此外，它还将我们的静态数据对象包装成迭代器或生成器。这在处理比GPU内存大得多的数据大小时非常有帮助——这种情况相当常见。这是通过分割数据来实现的，这样你就可以制作出适合你GPU内存的批次样本。

批次大小通常是2的幂，例如32、64、512等等。这种约定存在是因为它有助于在指令集级别上进行向量操作。据我所知，使用不同于2的幂的批次大小并没有帮助或损害我的处理速度。

# 规范和风格

我们将要使用的代码、迭代器和包装器来自[实用torchtext](https://github.com/keitakurita/practical-torchtext/)。这是一个由Keita Kurita创建的`torchtext`教程，他是`torchtext`的前五名贡献者之一。

命名规范和风格是受前面工作和基于PyTorch本身的深度学习框架fastai的启发。

让我们先设置所需的变量占位符：

```py
from torchtext.data import Field
```

`Field` 类确定数据如何进行预处理并转换为数值格式。`Field` 类是 `torchtext` 的基本数据结构，值得深入研究。`Field` 类模拟常见的文本处理并将它们设置为数值化（或向量化）：

```py
LABEL = Field(sequential=False, use_vocab=False)
```

默认情况下，所有字段都接受单词字符串作为输入，然后字段在之后构建从单词到整数的映射。这个映射称为词汇表，实际上是标记的one-hot编码。

我们看到在我们的案例中，每个标签已经是一个标记为0或1的整数。因此，我们不会进行one-hot编码——我们将告诉 `Field` 类这已经是一组one-hot编码且非序列的，分别通过设置 `use_vocab=False` 和 `sequential=False`：

```py
tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
```

这里发生了一些事情，所以让我们稍微展开一下：

+   `lower=True`：所有输入都被转换为小写。

+   `sequential=True`：如果为 `False`，则不应用分词。

+   `tokenizer`：我们定义了一个自定义的tokenize函数，它只是简单地根据空格分割字符串。你应该用spaCy分词器（设置 `tokenize="spacy"`) 替换它，看看这会不会改变损失曲线或最终模型的表现。

# 了解字段

除了我们之前提到的关键字参数之外，`Field` 类还将允许用户指定特殊标记（`unk_token` 用于词汇表外的未知单词，`pad_token` 用于填充，`eos_token` 用于句子的结束，以及可选的 `init_token` 用于句子的开始）。

预处理和后处理参数接受它接收到的任何 `torchtext.data.Pipeline`。预处理在分词之后但在数值化之前应用。后处理在数值化之后但在将它们转换为Tensor之前应用。

`Field` 类的文档字符串写得相当好，所以如果你需要一些高级预处理，你应该查阅它们以获取更多信息：

```py
from torchtext.data import TabularDataset
```

`TabularDataset` 是我们用来读取 `.csv`、`.tsv` 或 `.json` 文件的类。你可以在API中直接指定你正在读取的文件类型，即 `.tsv` 或 `.json`，这既强大又方便。

初看之下，你可能觉得这个类有点放错位置，因为通用的文件I/O+处理器API应该直接在PyTorch中可用，而不是在专门用于文本处理的包中。让我们看看为什么它被放在那里。

`TabularData` 有一个有趣的 `fields` 输入参数。对于CSV数据格式，`fields` 是一个元组的列表。每个元组反过来是列名和我们要与之关联的 `torchtext` 变量。字段应该与CSV或TSV文件中的列顺序相同。

在这里，我们只定义了两个字段：TEXT和LABEL。因此，每一列都被标记为其中之一。如果我们想完全忽略某一列，我们可以简单地将其标记为None。这就是我们如何标记我们的列作为模型学习的输入（TEXT）和目标（LABEL）。

字段参数与`TabularData`的这种紧密耦合是为什么它是`torchtext`的一部分而不是PyTorch的原因：

```py
tv_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]
```

这定义了我们的输入列表。我在这里手动做了这件事，但您也可以通过代码读取`train_df`的列标题，并相应地分配TEXT或LABEL。

作为提醒，我们将不得不为我们的测试数据定义另一个字段列表，因为它有不同的标题。它没有`LABEL`字段。

`TabularDataset`支持两个API：`split`和`splits`。我们将使用带有额外`s`的`splits`。splits API很简单：

+   `path`：这是文件名的前缀

+   `train`、`validation`：这是对应数据集的文件名

+   `format`：如前所述，可以是`.csv`、`.tsv`或`.json`；这里设置为`.csv`

+   `skip_header`：如果您的`.csv`文件中有列标题，就像我们的一样，则设置为`True`

+   `fields`：我们传递我们之前设置的字段列表：

```py
trn, vld = TabularDataset.splits(
        path="data", # the root directory where the data lies
        train='train.csv', validation="valid.csv",
        format='csv',
        skip_header=True, # make sure to pass this to ensure header doesn't get proceesed as data!
        fields=tv_datafields)
```

现在我们也重复同样的步骤来处理测试数据。我们再次删除`id`列，并将`comment_text`设置为我们的标签：

```py
tst_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT)
                 ]
```

我们直接将整个相对文件路径传递到`path`中，而不是在这里使用`path`和`test`变量的组合。我们在设置`trn`和`vld`变量时使用了`path`和`train`的组合。

作为备注，这些文件名与Keita在`torchtext`教程中使用的一致：

```py
tst = TabularDataset(
        path="data/test.csv", # the file path
        format='csv',
        skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=tst_datafields)
```

# 探索数据集对象

让我们看看数据集对象，即`trn`、`vld`和`tst`：

```py
trn, vld, tst

> (<torchtext.data.dataset.TabularDataset at 0x1d6c86f1320>,
 <torchtext.data.dataset.TabularDataset at 0x1d6c86f1908>,
 <torchtext.data.dataset.TabularDataset at 0x1d6c86f16d8>)
```

它们都是同一类的对象。我们的数据集对象可以像正常列表一样索引和迭代，所以让我们看看第一个元素的样子：

```py
trn[0], vld[0], tst[0]
> (<torchtext.data.example.Example at 0x1d6c86f1940>,
 <torchtext.data.example.Example at 0x1d6c86fed30>,
 <torchtext.data.example.Example at 0x1d6c86fecc0>)
```

我们所有的元素都是`example.Example`类的对象。每个示例将每一列存储为一个属性。但我们的文本和标签去哪里了？

```py
trn[0].__dict__.keys()
> dict_keys(['comment_text', 'toxic', 'severe_toxic', 'threat', 'obscene', 'insult', 'identity_hate']
```

`Example`对象将单个数据点的属性捆绑在一起。我们的`comment_text`和`labels`现在都是这些示例对象组成的字典的一部分。我们通过在`example.Example`对象上调用`__dict__.keys()`找到了所有这些对象：

```py
trn[0].__dict__['comment_text'][:5]
> ['explanation', 'why', 'the', 'edits', 'made']
```

文本已经被我们分词了，但还没有被向量化或数值化。我们将使用独热编码来处理训练语料库中存在的所有标记。这将把我们的单词转换成整数。

我们可以通过调用我们的`TEXT`字段的`build_vocab`属性来完成这个操作：

```py
TEXT.build_vocab(trn)
```

这个语句处理整个训练数据——特别是`comment_text`字段。单词被注册到词汇表中。

为了处理词汇表，`torchtext`有自己的类。`Vocab`类也可以接受如`max_size`和`min_freq`等选项，这些选项可以让我们知道词汇表中有多少单词，或者一个单词需要出现多少次才能被注册到词汇表中。

不在词汇表中的单词将被转换为 `<unk>`，表示 *未知* 的标记。过于罕见的单词也会被分配 `<unk>` 标记以简化处理。这可能会损害或帮助模型的性能，具体取决于我们失去了多少单词给 `<unk>` 标记：

```py
TEXT.vocab
> <torchtext.vocab.Vocab at 0x1d6c65615c0>
```

`TEXT` 字段现在有一个词汇属性，它是 `Vocab` 类的特定实例。我们可以利用这一点来查找词汇对象的属性。例如，我们可以找到训练语料库中任何单词的频率。`TEXT.vocab.freqs` 对象实际上是 `type collections.Counter` 的对象：

```py
type(TEXT.vocab.freqs)
> collections.Counter
```

这意味着它将支持所有功能，包括按频率排序单词的 `most_common` API，并为我们找到出现频率最高的前 k 个单词。让我们来看看它们：

```py
TEXT.vocab.freqs.most_common(5)
> [('the', 78), ('to', 41), ('you', 33), ('of', 30), ('and', 26)]
```

`Vocab` 类在其 `stoi` 属性中持有从单词到 `id` 的映射，并在其 `itos` 属性中持有反向映射。让我们看看这些属性：

```py
type(TEX

T.vocab.itos), type(TEXT.vocab.stoi), len(TEXT.vocab.itos), len(TEXT.vocab.stoi.keys())
> (list, collections.defaultdict, 784, 784)
```

**itos**，或整数到字符串映射，是一个单词列表。列表中每个单词的索引是其整数映射。例如，索引为 7 的单词将是 *and*，因为它的整数映射是 7。

**stoi**，或字符串到整数映射，是一个单词的字典。每个键是训练语料库中的一个单词，其值是一个整数。例如，“and”这个单词可能有一个整数映射，可以在该字典中以 O(1) 的时间复杂度查找。

注意，此约定自动处理由 Python 中的零索引引起的偏移量问题：

```py
TEXT.vocab.stoi['and'], TEXT.vocab.itos[7]
> (7, 'and')
```

# Iterators

`torchtext` 对 PyTorch 和 torchvision 中的 `DataLoader` 对象进行了重命名和扩展。本质上，它执行相同的三个任务：

+   批量加载数据

+   打乱数据

+   使用 `multiprocessing` 工作者并行加载数据

这种批量加载数据使我们能够处理比 GPU RAM 大得多的数据集。`Iterators` 扩展并专门化 `DataLoader` 以用于 NLP/文本处理应用。

我们将在这里使用 `Iterator` 和它的表亲 `BucketIterator`：

```py
from torchtext.data import Iterator, BucketIterator
```

# BucketIterator

`BucketIterator` 自动打乱并将输入序列桶化为相似长度的序列。

为了启用批量处理，我们需要具有相同长度的输入序列的批次。这是通过将较短的输入序列填充到批次中最长序列的长度来完成的。查看以下代码：

```py
[ [3, 15, 2, 7], 
  [4, 1], 
  [5, 5, 6, 8, 1] ]
```

这需要填充以成为以下内容：

```py
[ [3, 15, 2, 7, 0],
  [4, 1, 0, 0, 0],
  [5, 5, 6, 8, 1] ]
```

此外，当序列长度相似时，填充操作效率最高。`BucketIterator` 在幕后完成所有这些。这就是它成为文本处理中极其强大的抽象的原因。

我们希望桶排序基于 `comment_text` 字段的长度，因此我们将它作为关键字参数传递。

让我们继续初始化训练数据和验证数据的迭代器：

```py
train_iter, val_iter = BucketIterator.splits(
        (trn, vld), # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(32, 32),
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
```

让我们快速看一下传递给此函数的参数：

+   `batch_size`: 我们在训练和验证中都使用较小的批处理大小 32。这是因为我在使用只有 3 GB 内存 GTX 1060。

+   `sort_key`: `BucketIterator` 被告知使用 `comment_text` 中的标记数量作为排序的键在任何示例中。

+   `sort_within_batch`: 当设置为 `True` 时，它根据 `sort_key` 以降序对每个小批处理中的数据进行排序。

+   `repeat`: 当设置为 True 时，它允许我们循环遍历并再次看到之前看到的样本。我们在这里将其设置为 `False`，因为我们正在使用我们将在一分钟内编写的抽象进行重复。

同时，让我们花一分钟时间探索我们刚刚创建的新变量：

```py
train_iter

> <torchtext.data.iterator.BucketIterator at 0x1d6c8776518>

batch = next(train_iter.__iter__())
batch

> [torchtext.data.batch.Batch of size 25]
        [.comment_text]:[torch.LongTensor of size 494x25]
        [.toxic]:[torch.LongTensor of size 25]
        [.severe_toxic]:[torch.LongTensor of size 25]
        [.threat]:[torch.LongTensor of size 25]
        [.obscene]:[torch.LongTensor of size 25]
        [.insult]:[torch.LongTensor of size 25]
        [.identity_hate]:[torch.LongTensor of size 25]
```

现在，每个批处理都只有大小完全相同的 torch 张量（这里的尺寸是向量的向量的向量的向量的向量的向量的向量的长度）。这些张量还没有被移动到 GPU 上，但这没关系。

`batch` 实际上是已经熟悉的示例对象的包装器，它将所有与批处理相关的属性捆绑在一个变量字典中：

```py
batch.__dict__.keys()
> dict_keys(['batch_size', 'dataset', 'fields', 'comment_text', 'toxic', 'severe_toxic', 'threat', 'obscene', 'insult', 'identity_hate'])
```

如果我们的先前的理解是正确的，并且我们知道 Python 的对象传递是如何工作的，那么批处理变量的数据集属性应该指向 `torchtext.data.TabularData` 类型的 `trn` 变量。让我们检查这一点：

```py
batch.__dict__['dataset'], trn, batch.__dict__['dataset']==trn
```

哈哈！我们做对了。

对于测试迭代器，由于我们不需要洗牌，我们将使用普通的 `torchtext` `Iterator`：

```py
test_iter = Iterator(tst, batch_size=64, sort=False, sort_within_batch=False, repeat=False)
```

让我们看看这个迭代器：

```py
next(test_iter.__iter__())
> [torchtext.data.batch.Batch of size 33]
  [.comment_text]:[torch.LongTensor of size 158x33]
```

这里 `33` 的序列长度与输入的 `25` 不同。这没关系。我们可以看到这现在也是一个 torch 张量。

接下来，让我们为批处理对象编写一个包装器。

# BatchWrapper

在我们深入探讨 `BatchWrapper` 之前，让我告诉你批处理对象的问题所在。我们的批处理迭代器返回一个自定义数据类型，`torchtext.data.Batch`。这类似于多个 `example.Example`。它返回每个字段的批处理数据作为属性。这种自定义数据类型使得代码重用变得困难，因为每次列名更改时，我们需要修改代码。这也使得 `torchtext` 难以与其他库如 torchsample 和 fastai 一起使用。

那么，我们如何解决这个问题呢？

我们将批处理转换为形式为 (x, y) 的元组。x 是模型的输入，y 是目标 – 或者更传统地，x 是自变量，而 y 是因变量。一种思考方式是，模型将学习从 x 到 y 的函数映射。

BatchWrapper 帮助我们在不同数据集之间重用建模、训练和其他代码函数：

```py
class BatchWrapper: 
  def __init__(self, dl, x_var, y_vars): 
      self.dl, self.x_var, self.y_vars = dl, x_var, y_vars # we pass in the list of attributes for x and y

  def __iter__(self): 
      for batch in self.dl: 
          x = getattr(batch, self.x_var) # we assume only one input in this wrapper 
          if self.y_vars is not None: 
                # we will concatenate y into a single tensor 
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
                 else: y = torch.zeros((1)) if use_gpu: yield (x.cuda(), y.cuda()) else: yield (x, y) 

   def __len__(self): return len(self.dl)

```

`BatchWrapper` 类在初始化期间接受迭代器变量本身、变量 x 名称和变量 y 名称。它产生张量 x 和 y。x 和 y 的值通过 `getattr` 在 `self.dl` 中的 `batch` 中查找。

如果 GPU 可用，这个类将使用 `x.cuda()` 和 `y.cuda()` 将这些张量移动到 GPU 上，使其准备好被模型消费。

让我们快速使用这个新类包装我们的`train`、`val`和`test iter`对象：

```py
train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

valid_dl = BatchWrapper(val_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

test_dl = BatchWrapper(test_iter, "comment_text", None)
```

这返回了最简单的迭代器，准备好进行模型处理。请注意，在这种情况下，张量有一个设置为`cuda:0`的"device"属性。让我们预览一下：

```py
next(train_dl.__iter__())

> (tensor([[ 453,   63,   15,  ...,  454,  660,  778],
         [ 523,    4,  601,  ...,   78,   11,  650],
         ...,
         [   1,    1,    1,  ...,    1,    1,    1],
         [   1,    1,    1,  ...,    1,    1,    1]], device='cuda:0'),
 tensor([[ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         ...,
         [ 0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.]], device='cuda:0'))
```

# 训练文本分类器

我们现在准备好训练我们的文本分类器模型了。让我们从简单的事情开始：现在我们将这个模型视为一个黑盒。

模型架构的更好解释来自其他来源，包括斯坦福大学的CS224n等YouTube视频（[http://web.stanford.edu/class/cs224n/](http://web.stanford.edu/class/cs224n/)）。我建议您探索并将其与您已有的知识相结合：

```py
class SimpleLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=2):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=num_linear, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds
```

所有PyTorch模型都继承自`torch.nn.Module`。它们都必须实现`forward`函数，该函数在模型做出预测时执行。相应的用于训练的`backward`函数是自动计算的。

# 初始化模型

任何Pytorch模型都是像Python对象一样实例化的。与TensorFlow不同，其中没有严格的会话对象概念，代码是在其中编译然后运行的。模型类就像我们之前写的那样。

前一个类的`init`函数接受一些参数：

+   `hidden_dim`：这些是隐藏层维度，即隐藏层的向量长度

+   `emb_dim=300`：这是一个嵌入维度，即LSTM的第一个输入*步*的向量长度

+   `num_linear=2`：其他两个dropout参数：

    +   `spatial_dropout=0.05`

    +   `recurrent_dropout=0.1`

两个dropout参数都充当正则化器。它们有助于防止模型过拟合，即模型最终学习的是训练集中的样本，而不是可以用于做出预测的更通用的模式。

关于dropout之间的差异的一种思考方式是，其中一个作用于输入本身。另一个在反向传播或权重更新步骤中起作用，如前所述：

```py
em_sz = 300
nh = 500
model = SimpleLSTMBaseline(nh, emb_dim=em_sz)
print(model)

SimpleLSTMBaseline(
  (embedding): Embedding(784, 300)
  (encoder): LSTM(300, 500, num_layers=2, dropout=0.1)
  (linear_layers): ModuleList(
    (0): Linear(in_features=500, out_features=500, bias=True)
  )
  (predictor): Linear(in_features=500, out_features=6, bias=True)
)
```

我们可以打印任何PyTorch模型来查看类的架构。它是从`forward`函数实现中计算出来的，这正是我们所期望的。这在调试模型时非常有用。

让我们编写一个小工具函数来计算任何PyTorch模型的大小。在这里，我们所说的“大小”是指可以在训练期间更新的模型参数数量，以学习输入到目标映射。

虽然这个函数是在Keras中实现的，但它足够简单，可以再次编写：

```py
def model_size(model: torch.nn)->int:
    """
    Calculates the number of trainable parameters in any model

    Returns:
        params (int): the total count of all model weights
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     model_parameters = model.parameters()
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

print(f'{model_size(model)/10**6} million parameters')
> 4.096706 million parameters
```

我们可以看到，即使是我们的简单基线模型也有超过400万个参数。相比之下，一个典型的决策树可能只有几百个决策分支，最多。

接下来，我们将使用熟悉的`.cuda()`语法将模型权重移动到GPU上：

```py
if use_gpu:
    model = model.cuda()
```

# 再次将各个部分组合在一起

这些是我们查看的部分，让我们快速总结一下：

+   **损失函数**：二元交叉熵与 Logit 损失。它作为预测值与真实值之间距离的质量指标。

    +   **优化器**：我们使用默认参数的 Adam 优化器，学习率设置为 1e-2 或 0.01：

这是我们如何在 PyTorch 中看到这两个组件的方式：

```py
from torch import optim
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss().cuda()
```

我们在这里设置模型需要训练的周期数：

```py
epochs = 3
```

这被设置为一个非常小的值，因为整个笔记本、模型和训练循环只是为了演示目的。

# 训练循环

训练循环在逻辑上分为两部分：`model.train()` 和 `model.eval()`。注意以下代码的放置：

```py
from tqdm import tqdm
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # turn on training mode
    for x, y in tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
        opt.zero_grad()
        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(trn)

    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval() # turn on evaluation mode
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(preds, y)
        val_loss += loss.item() * x.size(0)

    val_loss /= len(vld)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
```

前半部分是实际的学习循环。这是循环内的步骤序列：

1.  将优化器的梯度设置为零

1.  在 `preds` 中对这个训练批次进行模型预测

1.  使用 `loss_func` 查找损失

1.  使用 `loss.backward()` 更新模型权重

1.  使用 `opt.step()` 更新优化器状态

整个反向传播的麻烦都在一行代码中处理：

```py
loss.backward()
```

这种抽象级别暴露了模型的内部结构，而不必担心微分学的方面，这就是为什么像 PyTorch 这样的框架如此方便和有用。

第二个循环是评估循环。这是在数据的验证分割上运行的。我们将模型设置为 *eval* 模式，这会锁定模型权重。只要 `model.eval()` 没有被设置回 `model.train()`，权重就不会意外更新。

在这个第二个循环中，我们只做两件简单的事情：

+   在验证分割上进行预测

+   计算此分割的损失

在每个周期结束时，打印出所有验证批次的总损失，以及运行训练损失。

一个训练循环看起来可能如下所示：

```py
100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2.34it/s]

Epoch: 1, Training Loss: 13.5037, Validation Loss: 4.6498
100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 4.58it/s]

Epoch: 2, Training Loss: 7.8243, Validation Loss: 24.5401

100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 3.35it/s]

Epoch: 3, Training Loss: 57.4577, Validation Loss: 4.0107
```

我们可以看到，训练循环以低验证损失和高训练损失结束。这可能表明模型或训练和验证数据分割存在问题。没有简单的方法来调试这个问题。

前进的好方法通常是训练模型几个更多个周期，直到观察到损失不再变化。

# 预测模式

让我们使用我们训练的模型在测试数据上进行一些预测：

```py
test_preds = []
model.eval()
for x, y in tqdm(test_dl):
    preds = model(x)
    # if you're data is on the GPU, you need to move the data back to the cpu
    preds = preds.data.cpu().numpy()
    # the actual outputs of the model are logits, so we need to pass these values to the sigmoid function
    preds = 1 / (1 + np.exp(-preds))
    test_preds.append(preds)
test_preds = np.hstack(test_preds)
```

整个循环现在处于评估模式，我们使用它来锁定模型权重。或者，我们也可以将 `model.train(False)` 设置为同样。

我们迭代地从测试迭代器中取出批大小样本，进行预测，并将它们追加到一个列表中。最后，我们将它们堆叠起来。

# 将预测转换为 pandas DataFrame

这有助于我们将预测结果转换为更易理解的格式。让我们读取测试数据框，并将预测值插入到正确的列中：

```py
test_df = pd.read_csv("data/test.csv")
for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    test_df[col] = test_preds[:, i]
```

现在，我们可以预览 DataFrame 的几行：

```py
test_df.head(3)
```

我们得到以下输出：

|  | id | `comment_text` | `toxic` | `severe_toxic` | `obscene` | `threat` | `insult` | `identity_hate` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 00001cee341fdb12 | Yo bitch Ja Rule is more succesful then you'll... | 0.629146 | 0.116721 | 0.438606 | 0.156848 | 0.139696 | 0.388736 |
| 1 | 0000247867823ef7 | `== From RfC == \r\n\r\n` 标题是好的，正如我... | 0.629146 | 0.116721 | 0.438606 | 0.156848 | 0.139696 | 0.388736 |
| 2 | 00013b17ad220c46 | "`\r\n\r\n == Sources == \r\n\r\n *`. Zawe Ashto... | 0.629146 | 0.116721 | 0.438606 | 0.156848 | 0.139696 | 0.388736 |

# 摘要

这是我们第一次接触深度学习在NLP中的应用。这是一个对`torchtext`的全面介绍，以及我们如何利用Pytorch来利用它。我们还对深度学习作为一个只有两到三个广泛组成部分的谜团有了非常广泛的了解：模型、优化器和损失函数。这无论你使用什么框架或数据集都是正确的。

为了保持简短，我们在模型架构解释上有所简化。我们将避免使用在其他部分未解释的概念。

当我们使用现代集成方法工作时，我们并不总是知道某个特定预测是如何被做出的。对我们来说，这是一个黑盒，就像所有深度学习模型的预测一样。

在下一章中，我们将探讨一些工具和技术，这些工具和技术将帮助我们窥视这些盒子——至少是更多一点。
