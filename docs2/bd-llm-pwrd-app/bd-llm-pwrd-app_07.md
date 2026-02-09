# 7

# 基于 LLM 的搜索和推荐引擎

在上一章中，我们介绍了构建对话应用程序的核心步骤。我们从一个简单的聊天机器人开始，然后添加了更复杂的组件，如记忆、非参数知识和外部工具。所有这些都可以通过 LangChain 的预构建组件以及 Streamlit 进行 UI 渲染来实现。尽管对话应用程序通常被视为生成 AI 和 LLM 的“舒适区”，但这些模型确实拥抱了更广泛的应用范围。

在本章中，我们将探讨如何使用嵌入和生成模型来增强推荐系统，我们将学习如何利用 LangChain 作为框架，利用最先进的 LLM 创建自己的推荐系统应用程序。

在本章中，我们将涵盖以下主题：

+   推荐系统的定义和演变

+   LLM 如何影响这一研究领域

+   使用 LangChain 构建推荐系统

# 技术要求

要完成本书中的任务，你需要以下内容：

+   Hugging Face 账户和用户访问令牌。

+   OpenAI 账户和用户访问令牌。

+   Python 版本 3.7.1 或更高版本。

+   确保已安装以下 Python 包：`langchain`、`python-dotenv`、`huggingface_hub`、`streamlit`、`lancedb`、`openai`和`tiktoken`。这些可以通过在终端中使用`pip install`轻松安装。

你可以在本书的 GitHub 仓库中找到本章的代码：`github.com/PacktPublishing/Building-LLM-Powered-Applications`。

# 推荐系统简介

推荐系统是一种计算机程序，为电子商务网站和社交网络等数字平台上的用户提供推荐项。它使用大量数据集来开发用户喜好和兴趣的模型，然后向单个用户推荐类似项。

根据使用的方法和数据，存在不同类型的推荐系统。其中一些常见类型包括：

+   **协同过滤**：这种类型的推荐系统使用具有与目标用户相似偏好的其他用户的评分或反馈。它假设过去喜欢某些物品的用户将来也会喜欢类似物品。例如，如果用户 A 和用户 B 都喜欢电影 X 和 Y，那么如果用户 B 也喜欢电影 Z，则算法可能会向用户 A 推荐电影 Z。

协同过滤可以进一步分为两种子类型：基于用户和基于物品：

+   **基于用户的协同过滤**寻找与目标用户相似的用户，并推荐他们喜欢的物品。

+   **基于物品的协同过滤**寻找与目标用户喜欢的物品相似的项目，并推荐它们。

+   **基于内容的过滤**：此类推荐系统使用项目本身的特征或属性来推荐与目标用户之前喜欢或互动过的项目相似的项目。它假设喜欢某个项目特定特征的用户会喜欢具有相似特征的其它项目。与基于项目的协同过滤相比，后者使用用户行为模式来做出推荐，而基于内容的过滤则使用关于项目的自身信息。例如，如果用户 A 喜欢电影 X，它是一部由演员 Y 主演的喜剧，那么算法可能会推荐电影 Z，它也是一部由演员 Y 主演的喜剧。

+   **混合过滤**：此类推荐系统结合了协同过滤和基于内容的过滤方法，以克服它们的一些局限性，并提供更准确和多样化的推荐。例如，YouTube 使用混合过滤来推荐视频，基于观看过类似视频的其他用户的评分和观看次数，以及视频本身的特征和类别。

+   **基于知识的过滤**：此类推荐系统使用关于领域和用户需求或偏好的显式知识或规则来推荐满足某些标准或约束的项目。它不依赖于其他用户的评分或反馈，而是依赖于用户的输入或查询。例如，如果用户 A 想购买具有特定规格和预算的笔记本电脑，那么算法可能会推荐满足这些标准的笔记本电脑。基于知识的推荐系统在没有或很少的评分历史记录的情况下，或者当项目复杂且可定制时，效果很好。

在上述框架内，还有各种可以使用的机器学习技术，我们将在下一节中介绍。

# 现有的推荐系统

现代推荐系统使用**机器学习**（**ML**）技术，基于如下可用数据，来更好地预测用户的偏好：

+   **用户行为数据**：关于用户与产品互动的见解。这些数据可以从用户评分、点击和购买记录等因素中获得。

+   **用户人口统计数据**：这指的是关于用户个人的信息，包括诸如年龄、教育背景、收入水平和地理位置等细节。

+   **产品属性数据**：这涉及关于产品特征的信息，例如书籍的流派、电影的演员阵容，或食物背景下的特定菜系。

到目前为止，一些最受欢迎的机器学习技术包括 K-最近邻算法、降维和神经网络。让我们详细看看这些方法。 

## K-最近邻算法

**K 近邻算法（KNN**）是一种机器学习算法，可用于分类和回归问题。它通过找到到新数据点最近的 k 个数据点（其中 k 表示要找到的最近数据点的数量，在初始化算法之前由用户设置）并使用它们的标签或值进行预测来实现。KNN 基于相似数据点可能具有相似标签或值的假设。

KNN 可以在协同过滤的上下文中应用于推荐系统，包括基于用户和基于物品：

+   基于用户的 KNN 是一种协同过滤，它使用与目标用户有相似口味或偏好的其他用户的评分或反馈。

例如，假设我们有三个用户：Alice、Bob 和 Charlie。他们都在线购买书籍并对其进行评分。Alice 和 Bob 都喜欢（高度评价）系列作品《哈利·波特》和书籍《霍比特人》。系统观察到这个模式，并认为 Alice 和 Bob 是相似的。

现在，如果 Bob 也喜欢 Alice 尚未阅读的书籍《权力的游戏》，系统将向 Alice 推荐《权力的游戏》。这是因为它假设既然 Alice 和 Bob 有相似的口味，Alice 也可能喜欢《权力的游戏》。

+   基于物品的 KNN 是另一种类型的协同过滤，它使用物品的属性或特征来向目标用户推荐相似物品。

例如，让我们考虑相同的用户和他们对书籍的评分。系统注意到 Alice 和 Bob 都喜欢《哈利·波特》系列和书籍《霍比特人》。因此，它认为这两本书是相似的。

现在，如果 Charlie 阅读并喜欢《哈利·波特》，系统将向 Charlie 推荐《霍比特人》。这是因为它假设既然《哈利·波特》和《霍比特人》是相似的（都受到相同用户的喜爱），Charlie 也可能喜欢《霍比特人》。

KNN 是推荐系统中的一个流行技术，但它有一些陷阱：

+   **可扩展性**：当处理大型数据集时，KNN 可能会变得计算成本高昂且速度较慢，因为它需要计算所有物品或用户对之间的距离。

+   **冷启动问题**：KNN 在处理新物品或用户时遇到困难，因为这些用户或物品有有限的或没有交互历史，因为它依赖于基于历史数据找到邻居。

+   **数据稀疏性**：在存在许多缺失值的数据稀疏集中，KNN 的性能可能会下降，这使得找到有意义的邻居变得具有挑战性。

+   **特征相关性**：KNN 将所有特征视为同等重要，并假设所有特征对相似度计算的贡献是相等的。在有些特征比其他特征更相关的情况下，这不一定成立。

+   **K 的选择**：选择合适的 K 值（邻居数量）可能是主观的，并可能影响推荐的品质。较小的 K 值可能导致噪声，而较大的 K 值可能导致推荐过于宽泛。

通常来说，在数据集小、噪声最小（这样异常值、缺失值和其他噪声不会影响距离度量）和动态数据（KNN 是一种基于实例的方法，不需要重新训练，可以快速适应变化）的场景下推荐使用 KNN。

此外，在推荐系统的领域中，还广泛使用了其他技术，如矩阵分解。

## 矩阵分解

矩阵分解是一种在推荐系统中使用的技巧，用于根据历史数据分析和预测用户偏好或行为。它涉及将一个大矩阵分解成两个或更多较小的矩阵，以揭示导致观察到的数据模式的潜在特征，并解决所谓的“维度灾难”。

**定义**

维度灾难指的是处理高维数据时出现的挑战。它导致复杂性增加、数据稀疏，以及由于数据需求指数增长和潜在过拟合导致的分析和建模困难。

在推荐系统的背景下，这项技术被用于预测用户-项目交互矩阵中的缺失值，该矩阵代表用户与各种项目（如电影、产品或书籍）的交互。

让我们考虑以下示例。想象你有一个矩阵，其中行代表用户，列代表电影，单元格包含评分（从 1 作为最低到 5 作为最高）。然而，并非所有用户都对所有电影进行了评分，导致矩阵中有许多缺失项：

|  | 电影 1 | 电影 2 | 电影 3 | 电影 4 |
| --- | --- | --- | --- | --- |
| 用户 1 | 4 | - | 5 | - |
| 用户 2 | - | 3 | - | 2 |
| 用户 3 | 5 | 4 | - | 3 |

表 7.1：具有缺失数据的示例数据集

矩阵分解旨在将这个矩阵分解成两个矩阵：一个用于用户，另一个用于电影，维度减少（潜在因子）。这些潜在因子可能代表属性，如类型偏好或特定电影特征。通过乘以这些矩阵，可以预测缺失的评分并推荐用户可能喜欢的电影。

矩阵分解有不同的算法，包括以下几种：

+   **奇异值分解**（**SVD**）将矩阵分解成三个独立的矩阵，其中中间矩阵包含奇异值，这些奇异值代表数据中不同组件的重要性。它在数据压缩、降维和推荐系统中的协同过滤中得到广泛应用。

+   **主成分分析**（**PCA**）是一种通过将其转换到一个与新坐标系统对齐的系统中来降低数据维度的方法。这些组件捕捉数据中最显著的变化，允许有效的分析和可视化。

+   **非负矩阵分解**（**NMF**）将矩阵分解为两个具有非负值的矩阵。它常用于主题建模、图像处理和特征提取，其中组件代表非负属性。

在推荐系统的背景下，最流行的技术可能是 SVD（多亏了它的可解释性、灵活性和处理缺失值和性能的能力），因此让我们使用这个例子继续下去。我们将使用 Python 的 `numpy` 模块来应用 SVD，如下所示：

```py
import numpy as np
# Your user-movie rating matrix (replace with your actual data)
user_movie_matrix = np.array([
    [4, 0, 5, 0],
    [0, 3, 0, 2],
    [5, 4, 0, 3]
])
# Apply SVD
U, s, V = np.linalg.svd(user_movie_matrix, full_matrices=False)
# Number of latent factors (you can choose this based on your preference)
num_latent_factors = 2
# Reconstruct the original matrix using the selected latent factors
reconstructed_matrix = U[:, :num_latent_factors] @ np.diag(s[:num_latent_factors]) @ V[:num_latent_factors, :]
# Replace negative values with 0
reconstructed_matrix = np.maximum(reconstructed_matrix, 0)
print("Reconstructed Matrix:")
print(reconstructed_matrix) 
```

以下为输出：

```py
Reconstructed Matrix:
[[4.2972542  0\.         4.71897811 0\.        ]
 [1.08572801 2.27604748 0\.         1.64449028]
 [4.44777253 4.36821972 0.52207171 3.18082082]] 
```

在这个例子中，`U` 矩阵包含与用户相关的信息，`s` 矩阵包含奇异值，而 `V` 矩阵包含与电影相关的信息。通过选择一定数量的潜在因子（`num_latent_factors`），你可以用降低维度的原始矩阵进行重构，同时将 `np.linalg.svd` 函数中的 `full_matrices=False` 参数设置为 `False` 确保分解的矩阵被截断，以与所选的潜在因子数量保持一致的维度。

这些预测评分可以用来向用户推荐预测评分更高的电影。矩阵分解使推荐系统能够揭示用户偏好的隐藏模式，并根据这些模式进行个性化推荐。

矩阵分解在推荐系统中是一个广泛使用的技术，尤其是在处理包含大量用户和项目的庞大数据集时，因为它能够有效地捕捉到潜在因子，即使在这种情况下；或者当你想要基于潜在因子进行个性化推荐时，因为它为每个用户和项目学习独特的潜在表示。然而，它有一些陷阱（一些与 KNN 技术类似）：

+   **冷启动问题**：与 KNN 类似，矩阵分解在处理新项目或用户时遇到困难，因为这些新项目或用户具有有限或没有交互历史。由于它依赖于历史数据，因此它无法有效地为新项目或用户提供推荐。

+   **数据稀疏性**：随着用户和项目的数量增加，用户-项目交互矩阵变得越来越稀疏，导致准确预测缺失值具有挑战性。

+   **可扩展性**：对于大型数据集，执行矩阵分解可能计算成本高且耗时。

+   **有限上下文**：矩阵分解通常只考虑用户-项目交互，忽略了时间、位置或额外用户属性等上下文信息。

因此，近年来，**神经网络**（**NNs**）已被探索作为缓解这些陷阱的替代方案。

## 神经网络

在推荐系统中，神经网络被用于通过从数据中学习复杂模式来提高推荐的准确性和个性化。以下是神经网络在这个背景下通常的应用方式：

+   **使用神经网络的协同过滤**：神经网络可以通过将用户和项目嵌入到连续向量空间中来模拟用户-项目交互。这些嵌入捕捉了代表用户偏好和项目特征的潜在特征。神经协同过滤模型将这些嵌入与神经网络架构相结合，以预测用户和项目之间的评分或交互。

+   **基于内容的推荐**：在基于内容的推荐系统中，神经网络可以学习项目内容的表示，如文本、图像或音频。这些表示捕捉了项目特征和用户偏好。卷积神经网络（**CNNs**）和循环神经网络（**RNNs**）等神经网络用于处理和从项目内容中学习，从而实现个性化的基于内容的推荐。

+   **序列模型**：在用户交互具有时间序列的场景中，例如点击流或浏览历史，循环神经网络（RNNs）或其变体如**长短期记忆**（**LSTM**）网络可以捕捉用户行为中的时间依赖性，并做出序列推荐。

+   **自动编码器和变分自动编码器**（**VAEs**）可以用来学习用户和项目的低维表示。

**定义**

自动编码器是一种用于无监督学习和降维的神经网络架构。它们由编码器和解码器组成。编码器将输入数据映射到低维潜在空间表示，而解码器则试图从编码表示中重建原始输入数据。

VAEs 是传统自动编码器的一种扩展，引入了概率元素。VAEs 不仅学习将输入数据编码到潜在空间中，还使用概率方法来模拟这个潜在空间的分布。这允许从学习到的潜在空间中生成新的数据样本。VAEs 用于生成任务，如图像合成、异常检测和数据插补。

在自动编码器和 VAEs 中，想法是在潜在空间中学习输入数据的压缩和有意义的表示，这对于包括特征提取、数据生成和降维在内的各种任务都是有用的。

这些表示可以用来通过在潜在空间中识别相似的用户和项目来做出推荐。实际上，具有神经网络独特架构的想法允许以下技术：

+   **侧信息整合**：神经网络可以整合额外的用户和项目属性，例如人口统计信息、位置或社交关系，通过学习来自不同数据源的信息来改善推荐。

+   **深度强化学习**：在某些情况下，深度强化学习可以用来优化推荐，通过学习用户反馈来建议最大化长期奖励的动作。

神经网络提供了灵活性和捕捉数据中复杂模式的能力，这使得它们非常适合用于推荐系统。然而，它们也需要仔细的设计、训练和调整以达到最佳性能。神经网络也带来了自己的挑战，包括以下内容：

+   **复杂性增加**：由于分层架构，神经网络，尤其是 **深度神经网络**（**DNNs**），可能会变得极其复杂。随着我们添加更多的隐藏层和神经元，模型学习复杂模式的能力增加。

+   **训练需求**：神经网络是重量级的模型，其训练需要特殊的硬件要求，包括 GPU，这可能会非常昂贵。

+   **潜在的过拟合**：当人工神经网络（ANN）学会在训练数据上表现出色，但无法推广到未见数据时，就会发生过拟合。

选择合适的架构、处理大型数据集和调整超参数对于有效地在推荐系统中使用神经网络至关重要。

尽管近年来已经取得了一些相关进展，但上述技术仍然存在一些缺陷，主要是它们具有任务特定性。例如，一个评分预测推荐系统将无法处理我们需要推荐可能符合用户口味的顶级 *k* 项的任务。实际上，如果我们将这种限制扩展到其他“LLM 之前”的 AI 解决方案，我们可能会看到一些相似之处：确实，是任务特定的情况使得 LLMs 和更一般的大型基础模型正在革命性地改变，它们高度通用且能够适应各种任务，这取决于用户的提示和指令。因此，在推荐系统领域进行了广泛的研究，以探讨 LLMs 能够在多大程度上增强当前模型。在接下来的章节中，我们将介绍这些新方法背后的理论，并参考关于这个新兴领域的最新论文和博客。

# LLMs 如何改变推荐系统

我们在前几章中看到了 LLMs 可以通过三种主要方式定制：预训练、微调和提示。根据 Wenqi Fan 等人撰写的论文《大型语言模型时代推荐系统（LLMs）》，这些技术也可以用来定制 LLM 以成为推荐系统：

+   **预训练**：为推荐系统预训练 LLMs 是一个重要步骤，使 LLMs 能够获取广泛的世界知识和用户偏好，并适应不同的推荐任务，无需或仅需少量样本。

**注意**

一个推荐系统 LLM 的例子是 P5，由 Shijie Gang 等人在他们的论文《推荐作为语言处理（RLP）：统一预训练、个性化提示与预测范式（P5）》中介绍。

P5 是一种用于构建推荐系统的统一文本到文本范式，它使用 **大型语言模型**（**LLMs**）。它包括三个步骤：

+   预训练：基于 T5 架构的基础语言模型在大规模网络语料库上预训练，并在推荐任务上微调。

+   个性化提示：根据每个用户的行为数据和上下文特征生成个性化的提示。

+   预测：将个性化提示输入到预训练的语言模型中，以生成推荐。

P5 基于 LLMs 可以编码广泛的世界知识和用户偏好，并且可以适应不同的推荐任务，无需或仅需少量示例。

+   **微调**：从头开始训练一个 LLM 是一项计算密集型活动。为推荐系统定制 LLM 的另一种替代且不那么侵入的方法可能是微调。

更具体地说，论文的作者回顾了微调 LLMs 的两种主要策略：

+   **全模型微调**涉及根据特定推荐数据集更改整个模型的权重。

+   **参数高效微调**旨在仅更改一小部分权重或开发可训练的适配器以适应特定任务。

+   **提示**：将 LLMs 调整为推荐系统的第三种和“最轻”的方法是提示。根据作者的说法，有三种主要的技术用于提示 LLMs：

    +   **传统提示**旨在通过设计文本模板或提供一些输入输出示例，将下游任务统一到语言生成任务中。

    +   **上下文学习**使 LLMs 能够在不进行微调的情况下，根据上下文信息学习新任务。

    +   **思维链**通过在提示中提供多个示例来描述思维链，从而增强了 LLMs 的推理能力。作者们还讨论了每种技术的优缺点，并提供了采用这些技术的现有方法的例子。

无论类型如何，提示是测试通用 LLM 是否能够处理推荐系统任务的最快方式。

LLM 在推荐系统领域的应用正在引起研究领域的兴趣，并且已经有一些有趣的结果证据，如上述所示。

在下一节中，我们将使用提示方法并利用 LangChain 作为 AI 编排器的能力来实现我们自己的推荐应用。

# 实现一个由 LLM 驱动的推荐系统

现在我们已经介绍了一些关于推荐系统以及 LLMs 如何增强它们的研究理论，让我们开始构建我们的推荐应用，这将是一个名为 MovieHarbor 的电影推荐系统。目标是使其尽可能通用，这意味着我们希望我们的应用能够通过对话界面处理各种推荐任务。我们将模拟的情景是所谓的“冷启动”，即用户与推荐系统的第一次交互，我们没有任何用户的偏好历史。我们将利用一个包含文本描述的电影数据库。

为了这个目的，我们将使用可在 Kaggle 上找到的*电影推荐数据*数据集，网址为[`www.kaggle.com/datasets/rohan4050/movie-recommendation-data`](https://www.kaggle.com/datasets/rohan4050/movie-recommendation-data)。

使用包含每部电影文本描述（以及如评分和电影标题等信息）的数据集的原因是为了我们可以获取文本的嵌入。因此，让我们开始构建我们的 MovieHarbor 应用程序。

## 数据预处理

为了将 LLMs 应用到我们的数据集中，我们首先需要对数据进行预处理。初始数据集包括几个列；然而，我们感兴趣的是以下列：

+   **类型**：适用于电影的适用类型列表。

+   **标题**：电影的标题。

+   **概述**：对剧情的文本描述。

+   **平均评分**：给定电影的 1 到 10 分的评分

+   **投票数**：给定电影的投票数。

我不会在这里报告完整的代码（你可以在本书的 GitHub 仓库中找到它：`github.com/PacktPublishing/Building-LLM-Powered-Applications`），然而，我会分享数据预处理的主要步骤：

1.  首先，我们将`genres`列格式化为一个`numpy`数组，这比数据集中的原始字典格式更容易处理：

    ```py
    import pandas as pd
    import ast
    # Convert string representation of dictionaries to actual dictionaries
    md['genres'] = md['genres'].apply(ast.literal_eval)
    # Transforming the 'genres' column
    md['genres'] = md['genres'].apply(lambda x: [genre['name'] for genre in x]) 
    ```

1.  接下来，我们将`vote_average`和`vote_count`列合并为一个单列，这是基于投票数的加权评分。我还将行限制在投票数的 95^(th)百分位数，这样我们就可以去除最低的投票数，以防止结果偏差：

    ```py
    # Calculate weighted rate (IMDb formula)
    def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
        return (vote_count / (vote_count + min_vote_count)) * vote_average + (min_vote_count / (vote_count + min_vote_count)) * 5.0
    # Minimum vote count to prevent skewed results
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    min_vote_count = vote_counts.quantile(0.95)
    # Create a new column 'weighted_rate'
    md['weighted_rate'] = md.apply(lambda row: calculate_weighted_rate(row['vote_average'], row['vote_count'], min_vote_count), axis=1) 
    ```

1.  接下来，我们创建一个名为`combined_info`的新列，我们将合并所有将作为 LLMs 上下文提供的元素。这些元素包括电影标题、概述、类型和评分：

    ```py
    md_final['combined_info'] = md_final.apply(lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}", axis=1).astype(str) 
    ```

1.  我们对电影`combined_info`进行分词，以便在嵌入时获得更好的结果：

    ```py
    import pandas as pd
    import tiktoken
    import os
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    from openai.embeddings_utils import get_embedding
    embedding_encoding = "cl100k_base" # this the encoding for text-embedding-ada-002
    max_tokens = 8000 # the maximum for text-embedding-ada-002 is 8191
    encoding = tiktoken.get_encoding(embedding_encoding)
    # omit reviews that are too long to embed
    md_final["n_tokens"] = md_final.combined_info.apply(lambda x: len(encoding.encode(x)))
    md_final = md_final[md_final.n_tokens <= max_tokens] 
    ```

**定义**

`cl100k_base`是 OpenAI 嵌入 API 使用的标记化器的名称。标记化器是一个工具，它将文本字符串分割成称为标记的单位，然后可以被神经网络处理。不同的标记化器有不同的规则和词汇表，用于如何分割文本以及使用哪些标记。

`cl100k_base` 分词器基于 **字节对编码**（**BPE**）算法，从大量文本语料库中学习子词单元的词汇表。`cl100k_base` 分词器有 10 万个标记，其中主要是常见单词和词片段，但也包括一些用于标点、格式化和控制的特殊标记。它可以处理多种语言和领域的文本，并且可以编码每个输入高达 8,191 个标记。

1.  我们使用 `text-embedding-ada-002` 将文本嵌入：

    ```py
    md_final["embedding"] = md_final.overview.apply(lambda x: get_embedding(x, engine=embedding_model)) 
    ```

在更改一些列的名称并删除不必要的列之后，最终数据集看起来如下所示：

![](img/B21714_07_01.png)

图 7.1：最终电影数据集的样本

让我们看看文本的一个随机行：

```py
md['text'][0] 
```

以下输出是获得的：

```py
'Title: GoldenEye. Overview: James Bond must unmask the mysterious head of the Janus Syndicate and prevent the leader from utilizing the GoldenEye weapons system to inflict devastating revenge on Britain. Genres: Adventure, Action, Thriller. Rating: 6.173464373464373' 
```

我们将进行的最后一个更改是修改一些命名约定和数据类型，如下所示：

```py
md_final.rename(columns = {'embedding': 'vector'}, inplace = True)
md_final.rename(columns = {'combined_info': 'text'}, inplace = True)
md_final.to_pickle('movies.pkl') 
```

1.  现在我们有了我们的最终数据集，我们需要将其存储在 VectorDB 中。为此，我们将利用 **LanceDB**，这是一个使用持久存储构建的开源向量搜索数据库，它极大地简化了嵌入的检索、过滤和管理，同时也提供了与 LangChain 的原生集成。您可以通过 `pip install lancedb` 轻松安装 LanceDB：

    ```py
    import lancedb
    uri = "data/sample-lancedb"
    db = lancedb.connect(uri)
    table = db.create_table("movies", md) 
    ```

现在我们已经拥有了所有原料，我们可以开始使用这些嵌入并开始构建我们的推荐系统。我们将从一个简单的冷启动场景任务开始，使用 LangChain 组件逐步增加复杂度。之后，我们还将尝试一个基于内容的场景，以挑战我们的 LLMs 执行多样化的任务。

## 在冷启动场景中构建 QA 推荐聊天机器人

在前面的章节中，我们看到了冷启动场景——这意味着在没有用户背景的情况下首次与用户互动——是推荐系统经常遇到的问题。我们对用户了解得越少，就越难将推荐与他们的偏好匹配。

在本节中，我们将使用以下高级架构模拟 LangChain 和 OpenAI 的 LLMs 的冷启动场景：

![计算机图示 自动生成的描述](img/B21714_07_02.png)

图 7.2：冷启动场景中推荐系统的高级架构

在上一节中，我们已经将我们的嵌入保存到了 LanceDB 中。现在，我们将构建一个 LangChain RetrievalQA 检索器，这是一个为针对索引进行问答而设计的链组件。在我们的案例中，我们将使用向量存储作为我们的索引检索器。其想法是，链在用户的查询下返回最相似的 *k* 部电影，使用余弦相似度作为距离度量（这是默认的）。

因此，让我们开始构建链：

1.  我们只使用电影概述作为信息输入：

    ```py
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import LanceDB
    os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings()
    docsearch = LanceDB(connection = table, embedding = embeddings)
    query = "I'm looking for an animated action movie. What could you suggest to me?"
    docs = docsearch.similarity_search(query)
    docs 
    ```

以下是对应的输出（我将显示输出的截断版本，只显示四个文档来源中的第一个）：

```py
[Document(page_content='Title: Hitman: Agent 47\. Overview: An assassin teams up with a woman to help her find her father and uncover the mysteries of her ancestry. Genres: Action, Crime, Thriller. Rating: 5.365800865800866', metadata={'genres': array(['Action', 'Crime', 'Thriller'], dtype=object), 'title': 'Hitman: Agent 47', 'overview': 'An assassin teams up with a woman to help her find her father and uncover the mysteries of her ancestry.', 'weighted_rate': 5.365800865800866, 'n_tokens': 52, 'vector': array([-0.00566491, -0.01658553, […] 
```

如您所见，每个`Document`旁边都报告了所有变量作为元数据，距离也作为分数报告。距离越低，用户查询与电影文本嵌入之间的接近度就越大。

1.  一旦我们收集到最相似的文档，我们希望得到一个对话式响应。为了达到这个目标，除了嵌入模型外，我们还将使用 OpenAI 的完成模型 GPT-3，并将其结合到 RetrievalQA 中：

    ```py
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
    query = "I'm looking for an animated action movie. What could you suggest to me?"
    result = qa({"query": query})
    result['result'] 
    ```

让我们看看输出结果：

```py
' I would suggest Transformers. It is an animated action movie with genres of Adventure, Science Fiction, and Action, and a rating of 6.' 
```

1.  由于我们设置了`return_source_documents=True`参数，我们还可以检索文档来源：

    ```py
    result['source_documents'][0] 
    ```

以下是输出结果：

```py
Document(page_content='Title: Hitman: Agent 47\. Overview: An assassin teams up with a woman to help her find her father and uncover the mysteries of her ancestry. Genres: Action, Crime, Thriller. Rating: 5.365800865800866', metadata={'genres': array(['Action', 'Crime', 'Thriller'], dtype=object), 'title': 'Hitman: Agent 47', 'overview': 'An assassin teams up with a woman to help her find her father and uncover the mysteries of her ancestry.', 'weighted_rate': 5.365800865800866, 'n_tokens': 52, 'vector': array([-0.00566491, -0.01658553, -0.02255735, ..., -0.01242317,
       -0.01303058, -0.00709073], dtype=float32), '_distance': 0.42414575815200806}) 
```

注意，第一个报告的文档不是模型建议的。这可能是由于评分较低，低于 Transformers（仅是第三个结果）。这是一个很好的例子，说明了 LLM 是如何在相似性之外考虑多个因素，向用户推荐电影的。

1.  尽管模型能够生成对话式答案，但它仍然只使用了可用信息的一部分——文本概述。如果我们想让我们的 MovieHarbor 系统也利用其他变量怎么办？我们可以以两种方式来处理这个任务：

    +   **“过滤”方式**：这种方法包括向我们的检索器添加一些作为**kwargs**的过滤器，这些过滤器可能在响应用户之前由应用程序要求。这些问题可能包括，例如，关于电影类型的。

    例如，假设我们只想提供那些类型被标记为喜剧的电影的结果。你可以用以下代码实现：

    ```py
    df_filtered = md[md['genres'].apply(lambda x: 'Comedy' in x)]
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}), return_source_documents=True)
    query = "I'm looking for a movie with animals and an adventurous plot."
    result = qa({"query": query}) 
    ```

    过滤器也可以在元数据级别上操作，如下面的示例所示，我们只想过滤出评分高于 7 的结果：

    ```py
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'filter': {weighted_rate__gt:7}}), return_source_documents=True) 
    ```

    +   **“代理”方式**：这可能是解决这个问题的最创新的方法。使我们的链成为代理可以依赖的工具，包括额外的变量。通过这样做，用户只需以自然语言提供他们的偏好，代理就可以在需要时检索最有希望的推荐。

    让我们看看如何通过代码实现这一点，具体要求是一个动作电影（因此根据`genre`变量进行筛选）：

    ```py
    from langchain.agents.agent_toolkits import create_retriever_tool
    from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature = 0)
    retriever = docsearch.as_retriever(return_source_documents = True)
    tool = create_retriever_tool(
        retriever,
        "movies",
        "Searches and returns recommendations about movies."
    )
    tools = [tool]
    agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
    result = agent_executor({"input": "suggest me some action movies"}) 
    ```

让我们看看思维链的片段和产生的输出（始终基于根据余弦相似度确定的四个最相似的影片）：

```py
> Entering new AgentExecutor chain...
Invoking: `movies` with `{'genre': 'action'}`
[Document(page_content='The action continues from [REC], […]
Here are some action movies that you might enjoy:
1\. [REC]² - The action continues from [REC], with a medical officer and a SWAT team sent into a sealed-off apartment to control the situation. It is a thriller/horror movie.
2\. The Boondock Saints - Twin brothers Conner and Murphy take swift retribution into their own hands to rid Boston of criminals. It is an action/thriller/crime movie.
3\. The Gamers - Four clueless players are sent on a quest to rescue a princess and must navigate dangerous forests, ancient ruins, and more. It is an action/comedy/thriller/foreign movie.
4\. Atlas Shrugged Part III: Who is John Galt? - In a collapsing economy, one man has the answer while others try to control or save him. It is a drama/science fiction/mystery movie.
Please note that these recommendations are based on the genre "action" and may vary in terms of availability and personal preferences.
> Finished chain. 
```

1.  最后，我们可能还想使我们的应用程序更符合其作为推荐系统的目标。为此，我们需要进行一些提示工程。

**注意**

使用 LangChain 预构建组件（如 RetrievalQA 链）的一个优点是，它们附带一个预配置、精心制作的提示模板。在覆盖现有提示之前，检查它是很好的做法，这样您也可以看到哪些变量（在`{}`内）已由组件预期。

要探索现有的提示，您可以运行以下代码：

```py
print(qa.combine_documents_chain.llm_chain.prompt.template) 
```

这里是输出结果：

```py
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer: 
```

假设，例如，我们希望我们的系统为每个用户的请求返回三个建议，包括剧情简短描述和用户可能喜欢它的原因。以下是一个符合这一目标的示例提示：

```py
from langchain.prompts import PromptTemplate
template = """You are a movie recommender system that help users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
For each question, suggest three movies, with a short description of the plot and the reason why the user migth like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Your response:"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]) 
```

1.  现在我们需要将其传递到我们的链中：

    ```py
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)
    query = "I'm looking for a funny action movie, any suggestion?"
    result = qa({'query':query})
    print(result['result']) 
    ```

以下输出结果：

```py
1\. A Good Day to Die Hard: An action-packed comedy directed by John Moore, this movie follows Iconoclastic, take-no-prisoners cop John McClane as he travels to Moscow to help his wayward son Jack. With the Russian underworld in pursuit, and battling a countdown to war, the two McClanes discover that their opposing methods make them unstoppable heroes.
2\. The Hidden: An alien is on the run in America and uses the bodies of anyone in its way as a hiding place. With lots of innocent people dying in the chase, this action-packed horror movie is sure to keep you laughing.
3\. District B13: Set in the ghettos of Paris in 2010, this action-packed science fiction movie follows an undercover cop and ex-thug as they try to infiltrate a gang in order to defuse a neutron bomb. A thrilling comedy that will keep you laughing. 
```

1.  在我们的提示中，我们可能还想实现通过对话初步问题收集到的信息，这些信息我们可以将其设置为欢迎页面。例如，在让用户输入他们的自然语言问题之前，我们可能想询问他们的年龄、性别和最喜欢的电影类型。为此，我们可以在提示中插入一个部分，用于格式化与用户共享的输入变量，然后将这个提示块组合到我们将要传递给链的最终提示中。以下是一个示例（为了简单起见，我们将不询问用户就设置变量）：

    ```py
    from langchain.prompts import PromptTemplate
    template_prefix = """You are a movie recommender system that help users to find movies that match their preferences.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}"""
    user_info = """This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}"""
    template_suffix= """Question: {question}
    Your response:"""
    user_info = user_info.format(age = 18, gender = 'female')
    COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
    print(COMBINED_PROMPT) 
    ```

这里是输出结果：

```py
You are a movie recommender system that help users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
This is what we know about the user, and you can use this information to better tune your research:
Age: 18
Gender: female
Question: {question}
Your response: 
```

1.  现在让我们格式化提示并将其传递到我们的链中：

    ```py
    PROMPT = PromptTemplate(
        template=COMBINED_PROMPT, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)
    result = qa({'query':query})
    result['result'] 
    ```

我们得到以下输出：

```py
' Sure, I can suggest some action movies for you. Here are a few examples: A Good Day to Die Hard, Goldfinger, Ong Bak 2, and The Raid 2\. All of these movies have high ratings and feature thrilling action elements. I hope you find something that you enjoy!' 
```

如您所见，系统考虑了提供的用户信息。当我们构建 MovieHarbor 的前端时，我们将使这些信息动态化，作为向用户提出的初步问题。 

## 构建基于内容系统

在上一节中，我们讨论了冷启动场景，即系统对用户一无所知的情况。有时，推荐系统已经对用户有一些背景信息，将这种知识嵌入我们的应用中非常有用。让我们设想，例如，我们有一个用户数据库，其中系统存储了所有注册用户的信息（如年龄、性别、国家等），以及用户已经观看的电影及其评分。

要做到这一点，我们需要设置一个自定义提示，能够从源中检索这些信息。为了简单起见，我们将创建一个包含用户信息的样本数据集，只有两条记录，对应两个用户。每个用户将展示以下变量：用户名、年龄、性别，以及包含他们已经观看的电影及其评分的字典。

高级架构由以下图表表示：

![计算机流程图示意图  自动生成描述](img/B21714_07_03.png)

图 7.3：基于内容推荐系统的高级架构

让我们分解这个架构并检查每个步骤，以构建最终的内容型系统聊天，从可用的用户数据开始：

1.  如前所述，我们现在对我们的用户偏好有一些了解。更具体地说，假设我们有一个包含用户属性（姓名、年龄、性别）以及他们对一些电影评论（1 到 10 分的评分）的数据集。以下是用以创建数据集的代码：

    ```py
    import pandas as pd
    data = {
        "username": ["Alice", "Bob"],
        "age": [25, 32],
        "gender": ["F", "M"],
        "movies": [
            [("Transformers: The Last Knight", 7), ("Pokémon: Spell of the Unknown", 5)],
            [("Bon Cop Bad Cop 2", 8), ("Goon: Last of the Enforcers", 9)]
        ]
    }
    # Convert the "movies" column into dictionaries
    for i, row_movies in enumerate(data["movies"]):
        movie_dict = {}
        for movie, rating in row_movies:
            movie_dict[movie] = rating
        data["movies"][i] = movie_dict
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    df.head() 
    ```

以下输出结果：

![一个黑白屏幕，上面有白色文字，描述自动生成](img/B21714_07_04.png)

图 7.4：样本用户数据集

1.  我们现在想要做的是将冷启动提示的格式化逻辑与变量相结合。这里的区别在于，我们不是要求用户提供这些变量的值，而是直接从我们的用户数据集中收集它们。因此，我们首先定义我们的提示块：

    ```py
    template_prefix = """You are a movie recommender system that help users to find movies that match their preferences.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}"""
    user_info = """This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}
    Movies already seen alongside with rating: {movies}"""
    template_suffix= """Question: {question}
    Your response:""" 
    ```

1.  然后，我们按照以下方式格式化 `user_info` 块（假设与系统交互的用户是 `Alice`）：

    ```py
    age = df.loc[df['username']=='Alice']['age'][0]
    gender = df.loc[df['username']=='Alice']['gender'][0]
    movies = ''
    # Iterate over the dictionary and output movie name and rating
    for movie, rating in df['movies'][0].items():
        output_string = f"Movie: {movie}, Rating: {rating}" + "\n"
        movies+=output_string
        #print(output_string)
    user_info = user_info.format(age = age, gender = gender, movies = movies)
    COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
    print(COMBINED_PROMPT) 
    ```

这里是输出：

```py
You are a movie recommender system that help users to find movies that match their preferences.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
This is what we know about the user, and you can use this information to better tune your research:
Age: 25
Gender: F
Movies already seen alongside with rating: Movie: Transformers: The Last Knight, Rating: 7
Movie: Pokémon: Spell of the Unknown, Rating: 5
Question: {question}
Your response: 
```

1.  让我们现在在我们的链中使用这个提示：

    ```py
    PROMPT = PromptTemplate(
        template=COMBINED_PROMPT, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)
    query = "Can you suggest me some action movie based on my background?"
    result = qa({'query':query})
    result['result'] 
    ```

然后，我们得到以下输出：

```py
" Based on your age, gender, and the movies you've already seen, I would suggest the following action movies: The Raid 2 (Action, Crime, Thriller; Rating: 6.71), Ong Bak 2 (Adventure, Action, Thriller; Rating: 5.24), Hitman: Agent 47 (Action, Crime, Thriller; Rating: 5.37), and Kingsman: The Secret Service (Crime, Comedy, Action, Adventure; Rating: 7.43)."
' 
```

如您所见，现在模型能够根据用户在模型元提示中检索到的过去偏好的信息，为 Alice 推荐电影列表。

注意，在这个场景中，我们使用了一个简单的 pandas 数据框作为数据集。在生产场景中，将有关要解决的问题（如推荐任务）的变量存储的最佳实践是使用特征存储。特征存储是为支持机器学习工作流程而设计的数据库系统。它们允许数据团队存储、管理和访问用于训练和部署机器学习模型的特征。

此外，LangChain 提供了对一些最受欢迎的特征存储的本地集成：

+   **Feast**：这是一个开源的机器学习特征存储。它允许团队定义、管理、发现和提供特征。Feast 支持批处理和流数据源，并与各种数据处理和存储系统集成。Feast 使用 BigQuery 进行离线特征，使用 BigTable 或 Redis 进行在线特征。

+   **Tecton**：这是一个托管功能平台，为构建、部署和使用机器学习功能提供完整解决方案。Tecton 允许用户在代码中定义功能，进行版本控制，并按照最佳实践将它们部署到生产环境中。此外，它集成了现有的数据基础设施和机器学习平台，如 SageMaker 和 Kubeflow，并使用 Spark 进行特征转换，使用 DynamoDB 进行在线特征服务。

+   **Featureform**：这是一个虚拟特征存储，将现有的数据基础设施转换为特征存储。Featureform 允许用户使用标准的特征定义和 Python SDK 创建、存储和访问特征。它协调和管理特征工程和实体化的数据管道，并且与广泛的数据库系统兼容，如 Snowflake、Redis、Spark 和 Cassandra。

+   **AzureML 托管特征存储**：这是一种新的工作区类型，允许用户发现、创建和操作特征。此服务与现有数据存储、特征管道和 ML 平台（如 Azure Databricks 和 Kubeflow）集成。此外，它使用 SQL、PySpark、SnowPark 或 Python 进行特征转换，并使用 Parquet/S3 或 Cosmos DB 进行特征存储。

你可以在[`blog.langchain.dev/feature-stores-and-llms/`](https://blog.langchain.dev/feature-stores-and-llms/)了解更多关于 LangChain 与特征集成的信息。

# 使用 Streamlit 开发前端

既然我们已经看到了 LLM 驱动的推荐系统背后的逻辑，现在是时候给我们的 MovieHarbor 添加一个 GUI 了。为此，我们再次利用 Streamlit，并假设冷启动场景。一如既往，你可以在 GitHub 书籍仓库中找到完整的 Python 代码，网址为`github.com/PacktPublishing/Building-LLM-Powered-Applications`。

根据*第六章*中的 Globebotter 应用程序，在这种情况下，你也需要创建一个`.py`文件，通过`streamlit run file.py`在终端中运行。在我们的例子中，文件将命名为`movieharbor.py`。

让我们现在总结构建带有前端的应用程序的关键步骤：

1.  配置应用程序网页：

    ```py
    import streamlit as st
    st.set_page_config(page_title="GlobeBotter", page_icon="![](img/Icon.png)")
    st.header('![](img/Icon.png) Welcome to MovieHarbor, your favourite movie recommender') 
    ```

1.  导入凭证并建立与 LanceDB 的连接：

    ```py
    load_dotenv()
    #os.environ["HUGGINGFACEHUB_API_TOKEN"]
    openai_api_key = os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings()
    uri = "data/sample-lancedb"
    db = lancedb.connect(uri)
    table = db.open_table('movies')
    docsearch = LanceDB(connection = table, embedding = embeddings)
    # Import the movie dataset
    md = pd.read_pickle('movies.pkl') 
    ```

1.  为用户创建一些小部件来定义他们的特征和电影偏好：

    ```py
    # Create a sidebar for user input
    st.sidebar.title("Movie Recommendation System")
    st.sidebar.markdown("Please enter your details and preferences below:")
    # Ask the user for age, gender and favourite movie genre
    age = st.sidebar.slider("What is your age?", 1, 100, 25)
    gender = st.sidebar.radio("What is your gender?", ("Male", "Female", "Other"))
    genre = st.sidebar.selectbox("What is your favourite movie genre?", md.explode('genres')["genres"].unique())
    # Filter the movies based on the user input
    df_filtered = md[md['genres'].apply(lambda x: genre in x)] 
    ```

1.  定义参数化的提示块：

    ```py
    template_prefix = """You are a movie recommender system that help users to find movies that match their preferences.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}"""
    user_info = """This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}"""
    template_suffix= """Question: {question}
    Your response:"""
    user_info = user_info.format(age = age, gender = gender)
    COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
    print(COMBINED_PROMPT) 
    ```

1.  设置`RetrievalQA`链：

    ```py
    #setting up the chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'data': df_filtered}), return_source_documents=True) 
    ```

1.  插入用户搜索栏：

    ```py
    query = st.text_input('Enter your question:', placeholder = 'What action movies do you suggest?')
    if query:
        result = qa({"query": query})
        st.write(result['result']) 
    ```

就这样！你可以在终端中使用`streamlit run movieharbor.py`运行最终结果。它看起来如下：

![计算机截图  自动生成的描述](img/B21714_07_05.png)

图 7.5：Movieharbor 的 Streamlit 示例前端

所以，你可以看到，仅仅几行代码，我们就能够为我们的 MovieHarbor 设置一个 webapp。从这个模板开始，你可以使用 Streamlit 的组件自定义你的布局，以及针对基于内容的场景进行定制。此外，你可以以这种方式自定义你的提示，使得推荐器按照你的偏好行事。

# 摘要

在本章中，我们探讨了 LLM 如何改变我们处理推荐系统任务的方式。我们从分析构建推荐应用程序的当前策略和算法开始，区分了各种场景（协同过滤、基于内容、冷启动等）以及不同的技术（KNN、矩阵分解和 NNs）。

然后，我们转向研究如何将 LLM 的力量应用于这个新出现的领域，并探索了最近几个月进行的各种实验。

利用这些知识，我们构建了一个由 LLMs 驱动的电影推荐应用，使用 LangChain 作为 AI 调度器和 Streamlit 作为前端，展示了 LLMs 如何凭借其推理能力和泛化能力彻底改变这一领域。这只是 LLMs 不仅能够开辟新的前沿，还能增强现有研究领域的一个例子。

在下一章中，我们将看到这些强大的模型在处理结构化数据时能做什么。

# 参考文献

+   **推荐作为语言处理**（**RLP**）：一个统一的**预训练、个性化提示与预测范式**（**P5**）。[`arxiv.org/abs/2203.13366`](https://arxiv.org/abs/2203.13366)

+   LangChain 关于特征存储的博客。[`blog.langchain.dev/feature-stores-and-llms/`](https://blog.langchain.dev/feature-stores-and-llms/)

+   Feast。[`docs.feast.dev/`](https://docs.feast.dev/)

+   Tecton。[`www.tecton.ai/`](https://www.tecton.ai/)

+   FeatureForm。[`www.featureform.com/`](https://www.featureform.com/)

+   Azure 机器学习功能存储。[`learn.microsoft.com/en-us/azure/machine-learning/concept-what-is-managed-feature-store?view=azureml-api-2`](https://learn.microsoft.com/en-us/azure/machine-learning/concept-what-is-managed-feature-store?view=azureml-api-2)

# 加入我们的 Discord 社区

加入我们的社区 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/llm`](https://packt.link/llm)

![](img/QR_Code214329708533108046.png)
