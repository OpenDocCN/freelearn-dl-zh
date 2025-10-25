

# 第九章：以量和可视化评估 RAG

评估在构建和维护 **检索增强生成** (**RAG**) 管道中发挥着关键作用。 在构建管道时，你可以使用评估来识别改进领域，优化系统性能，并系统地衡量改进的影响。 当你的 RAG 系统部署后，评估可以帮助确保系统的有效性、可靠性和性能 。

在本章中，我们将涵盖以下主题： 以下内容：

+   在构建 RAG 应用时进行评估

+   部署后评估 RAG 应用

+   标准化 评估框架

+   真实值

+   代码实验室 9.1 – ragas

+   针对 RAG 系统的额外评估技术

让我们先谈谈评估如何帮助你在构建你的 RAG 系统的初期阶段。

# 技术要求

本章的代码放置在以下 GitHub 仓库中： [https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_09](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_09)

# 构建时评估

评估在整个 RAG 管道开发过程中发挥着关键作用。 在构建系统时，通过持续评估你的系统，你可以确定需要改进的领域，优化系统的性能，并系统地衡量你做出的任何修改或增强的影响 。

评估对于理解 RAG 管道中不同方法的权衡和限制至关重要。 RAG 管道通常涉及各种技术选择，例如向量存储、检索算法和语言生成模型。 这些组件中的每一个都可能对系统的整体性能产生重大影响。 通过系统地评估这些组件的不同组合，你可以获得宝贵的见解，了解哪些方法对你的特定任务和领域产生最佳结果 。

例如，你可能尝试不同的嵌入模型，例如可以免费下载的本地开源模型或每次将文本转换为嵌入时收费的云服务 API。 你可能需要了解云 API 服务是否优于免费模型，如果是的话，它是否足够好以抵消额外的成本。 同样，你可以评估各种语言生成模型的表现，例如 ChatGPT、Llama 和 Claude。

这个迭代评估过程帮助你做出关于最适合你的 RAG 管道架构和组件的明智决策。 通过考虑效率、可扩展性和泛化能力等因素，你可以微调你的系统以实现最佳性能，同时最小化计算成本并确保在不同场景下的鲁棒性。 这至关重要。

评估对于理解 RAG 管道中不同方法的权衡和限制至关重要。 但评估在部署后也可能很有用，我们将在下一节中讨论。

# 部署后进行评估

一旦你的 RAG 系统部署完成，评估仍然是确保其持续有效性、可靠性和性能的关键方面。 对已部署的 RAG 管道进行持续监控和评估对于保持其质量以及识别任何潜在问题或随时间退化至关重要。 随着时间的推移，这至关重要。

有众多原因可能导致 RAG 系统在部署后性能下降。 例如，用于检索的数据可能随着新信息的出现而变得过时或不相关。 语言生成模型可能难以适应不断变化的患者查询或目标领域的变更。 此外，底层基础设施，如硬件或软件组件，可能会遇到性能问题 或故障。

想象一下，你是一家金融财富管理公司，该公司有一个基于 RAG 的应用程序，帮助用户了解可能影响其金融投资组合的最可能因素。 你的数据源可能包括过去五年内由主要金融公司发布的所有分析，涵盖了你的客户群所代表的全部金融资产。

然而，在金融市场，全球范围内的重大（宏观）事件可以对过去五年数据中未捕捉到的投资组合产生重大影响。重大灾难、政治不稳定，甚至某些股票的区域性事件都可能为它们的业绩设定全新的轨迹。 对于您的 RAG 应用来说，这代表着数据可以为您的最终用户提供的价值的变化，而且如果没有适当的更新，这种价值可能会随着时间的推移而迅速下降。 用户可能会开始询问 RAG 应用无法处理的具体事件，例如 *“刚刚发生的五级飓风将在下一年对我的投资组合产生什么影响？”* 但是，通过持续的更新和监控，尤其是关于飓风影响的最新报告，这些问题很可能会得到妥善解决。

为了减轻这些风险，持续监控您的 RAG 系统至关重要，尤其是在常见的故障点。 通过持续评估您 RAG 管道的这些关键组件，您可以主动识别并解决任何性能下降。 这可能包括用新鲜和相关的数据更新检索语料库，在新数据上微调语言生成模型，或者优化系统的基础设施以处理增加的负载或解决 性能瓶颈。

此外，建立允许用户报告任何问题或提供改进建议的反馈循环至关重要。 通过积极征求并整合用户反馈，您可以持续改进和增强您的 RAG 系统，更好地满足用户的需求。 这也可以包括监控用户界面使用、响应时间以及用户视角下生成的输出的相关性有用性等方面。 进行用户调查、分析用户交互日志和监控用户满意度指标可以提供有关您的 RAG 系统是否满足其预期目的的有价值见解。 您如何利用这些信息在很大程度上取决于您开发了哪种类型的 RAG 应用，但一般来说，这些是部署的 RAG 应用持续改进中最常见的监控领域。

通过定期评估您部署的 RAG 系统，您可以确保其长期的有效性、可靠性和性能。持续监控、主动问题检测以及持续改进的承诺是维护高质量 RAG 管道的关键，该管道能够随着时间的推移为用户带来价值 。

# 评估帮助你变得更好

为什么评估如此重要？ 简单来说，如果你不衡量你目前的位置，然后在改进之后再次衡量，那么将很难理解 如何或是什么改进（或损害）了你的 RAG 系统。

当出现问题时，如果没有客观标准进行比较，理解出了什么问题也很困难。 是你的检索机制出了问题吗？ 是提示出了问题吗？ 是你的 LLM 响应出了问题吗？ 这些问题是一个好的评估系统可以帮助回答的。

评估提供了一个系统性和客观的方式来衡量你的管道性能，确定需要改进的领域，并跟踪你做出的任何更改或改进的影响。 没有强大的评估框架，理解你的 RAG 系统进展如何以及它需要 进一步改进的地方变得具有挑战性。

通过将评估视为开发过程的一个基本部分，你可以持续改进和优化你的 RAG 管道，确保它提供最佳可能的结果，并满足其用户不断变化的需求。

在 RAG 系统开发过程的早期，你必须开始决定你将考虑哪些技术组件。 在这个时候，你甚至还没有安装 任何东西，所以你还不能评估你的代码，但你仍然可以使用 **标准化的评估框架** 来缩小你考虑的范围。 让我们讨论一些最常见的 RAG 系统元素的标准化评估框架。

# 标准化的评估框架

你的 RAG 系统的关键技术组件包括创建嵌入的嵌入模型、向量存储、向量搜索和 LLM。 当你查看每个技术组件的不同选项时，每个组件都有一些标准化的指标可供选择，这些指标可以帮助你比较它们。 以下是每个类别的一些常见指标。 这里有一些常见指标。

## 嵌入模型基准

**大规模文本嵌入基准** (**MTEB**) 检索排行榜评估了 不同数据集上各种检索任务中嵌入模型的性能。 MTEB 排行榜根据模型在多个嵌入和 检索相关任务上的平均性能进行排名。 您可以通过此 链接访问排行榜： [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

访问此网页时，点击 **检索** 和 **带说明的检索** 选项卡以获取 特定检索的嵌入评分。 为了评估排行榜上的每个模型，使用涵盖广泛领域的多个数据集测试模型的输出，例如 以下内容：

+   论点 检索（`ArguAna`）

+   气候事实 检索（`ClimateFEVER`）

+   重复问题 检索（`CQADupstackRetrieval`）

+   实体 检索（`DBPedia`）

+   事实抽取和 验证（`FEVER`）

+   金融 问答（`FiQA2018`）

+   多跳 问答（`HotpotQA`）

+   段落和文档 排序（`MSMARCO`）

+   事实核查（`NFCorpus`）

+   开放域 问答（`NQ`）

+   重复问题 检测（`QuoraRetrieval`）

+   科学文档 检索（`SCIDOCS`）

+   科学声明 验证（`SciFact`）

+   论点 检索（`Touche2020`）

+   与 COVID-19 相关的信息 检索（`TRECCOVID`）

排行榜 根据 这些任务的平均性能对嵌入模型进行排名，从而全面展示它们的优缺点。 您还可以点击任何指标，按该指标对排行榜进行排序。 例如，如果您对更关注金融问答的指标感兴趣，请查看在 FiQA2018 数据集上得分最高的模型。

## 向量存储和向量搜索基准测试

**近似最近邻基准测试** 是一个 基准测试工具，用于评估 **近似最近邻** (**ANN**) 算法的性能，我们已在 *第八章*中详细讨论。 ANN-Benchmarks 评估了不同向量搜索工具在各种数据集上的搜索准确性、速度和内存使用情况，包括 我们在 *第八章*中提到的向量搜索工具—**Facebook AI 相似度搜索** (**FAISS**)，**近似最近邻哦，是的** (**ANNOY**)，以及**分层可导航小世界** **（HNSW）**。

**信息检索基准测试** (**BEIR**) 是另一个 有用的 资源，用于评估向量存储和搜索算法。 它提供了一个异构基准，用于跨多个领域（包括问答、事实核查和实体检索）对信息检索模型进行零样本评估。 我们将在 *第十三章* 中进一步讨论 *零样本* 的含义 *，但基本上，这意味着没有包含任何示例的问题/用户查询，这在 RAG 中是一种常见情况。 BEIR 提供了一个标准化的评估框架，包括以下流行数据集：

+   `MSMARCO`：一个从现实世界的查询和答案中提取的大规模数据集，用于评估搜索和问答中的深度学习模型 和问答

+   `HotpotQA`：一个具有自然、多跳问题的问答数据集，对支持事实进行强监督，以支持更可解释的 问答系统

+   `CQADupStack`：一个用于 **社区问答** (**cQA**) 研究的基准数据集，从 12 个 Stack Exchange 子论坛中提取，并标注了重复 问题信息

这些数据集，以及 BEIR 基准中的其他数据集，涵盖了广泛的领域和信息检索任务，使您能够评估您的检索系统在不同环境中的性能，并将其与 最先进的方法进行比较。

## LLM 基准

人工分析 LLM 性能排行榜是一个全面的资源，用于评估 开源和专有语言模型，如 ChatGPT、Claude 和 Llama。 它 评估了模型在广泛任务上的性能。 为了进行质量比较，它使用了一系列子排行榜：

+   **通用能力**： 聊天机器人竞技场

+   **推理** **和知识**：

    +   **大规模多任务语言** **理解** (**MMLU**)

    +   **多轮基准** (**MT Bench**)

他们还跟踪速度和价格，并提供分析，以便您比较这些领域的平衡。 通过根据这些任务上的性能对模型进行排名，排行榜提供了对它们能力的全面看法。

它可以在以下位置找到： [ https://artificialanalysis.ai/](https://artificialanalysis.ai/)

除了通用的 LLM 排行榜，还有专注于 LLM 性能特定方面的专业排行榜。《人工分析 LLM 性能排行榜 》评估 LLM 的技术方面，例如推理速度、内存消耗和可扩展性。 它包括吞吐量（每秒处理的令牌数）、延迟（生成响应的时间）、内存占用和扩展效率等指标。 这些指标有助于您了解不同 LLM 的计算需求和性能特征。 。

开放 LLM 排行榜跟踪开源语言模型在各种 自然语言理解和生成任务上的性能。 它包括基准测试，如**AI2 推理挑战** (**ARC**)用于复杂科学推理，HellaSwag 用于常识推理，MMLU 用于特定领域的性能，TruthfulQA 用于生成真实和有信息量的响应，WinoGrande 通过代词消歧进行常识推理，以及**小学数学 8K** (**GSM8K**)用于数学 推理能力。

## 关于标准化评估框架的最终思考

使用标准化评估框架和基准测试为比较您 RAG 管道中不同组件的性能提供了一个有价值的起点。 它们涵盖了广泛的任务和领域，使您能够评估各种方法的优缺点。 通过考虑这些基准测试的结果，以及其他因素如计算效率和易于集成，您可以在选择最适合您特定 RAG 应用的最佳组件时缩小选择范围并做出更明智的决策。

然而，需要注意的是，尽管这些标准化评估指标对于初始组件选择有帮助，但它们可能无法完全捕捉到您特定 RAG 管道与独特输入和输出的性能。 为了真正了解您的 RAG 系统在特定用例中的表现，您需要设置自己的评估框架，以适应您特定的需求。 这个定制化的评估 系统将为您的 RAG 管道提供最准确和相关的见解。

接下来，我们需要讨论 RAG 评估中最重要且经常被忽视的一个方面，那就是您的 真实数据。

# 什么是基准数据？

简单来说，基准数据是代表如果你 RAG 系统处于 最佳性能时你期望的理想响应的数据。

作为一个实际例子，如果你有一个专注于允许某人询问关于 犬类兽医医学最新癌症研究的 RAG 系统，你的数据源是所有提交给 PubMed 的关于该主题的最新研究论文，你的基准数据很可能是可以对该数据提出和回答的问题和答案。 你希望使用目标受众真正会提出的问题，并且答案应该是你认为从 LLM 期望的理想答案。 这可能有一定的客观性，但无论如何，拥有可以与你的 RAG 系统的输入和输出进行比较的基准数据集是帮助比较你做出的更改的影响并最终使系统更有效的一种关键方式。 更有效。

## 如何使用基准数据？

基准数据作为衡量 RAG 系统性能的基准。 通过比较 RAG 系统生成的输出与基准数据，你可以评估系统检索相关信息和生成准确、连贯响应的能力。 基准数据有助于量化不同 RAG 方法的有效性并确定改进领域。

## 生成基准数据

手动创建基准数据 可能耗时。 如果你的公司已经有一组针对特定查询或提示的理想响应的数据集，那将是一个宝贵的资源。 然而，如果此类数据不可用，我们将在下一部分探讨获取基准数据的替代方法。

## 人工标注

你可以 雇佣人工标注员手动为一系列查询或提示创建理想响应。 这确保了高质量的基准数据，但可能成本高昂且耗时，尤其是对于 大规模评估。

## 专家知识

在某些 领域，你可能可以访问 **领域专家** (**SMEs**) 他们可以根据他们的 专业知识 提供基准响应。 这在需要准确信息的专业或技术领域尤其有用。

一种常见的帮助此方法的方法称为 **基于规则生成**。使用基于规则生成，对于特定领域或任务，您可以定义一组规则或模板来生成合成地面实况，并利用您的 SMEs 来填写模板。 通过利用领域知识和预定义的模式，您可以创建与预期格式和内容相符的响应。

例如，如果您正在构建一个用于支持手机的客户支持聊天机器人，您可能有一个这样的模板： `要解决[问题]，您可以尝试[解决方案]`。您的 SMEs 可以在可能的问题-解决方案方法中填写各种问题-解决方案，其中问题可能是 *电池耗尽* ，解决方案是 *降低屏幕亮度和关闭后台应用*。这将输入到模板中（我们称之为“加水”），最终输出将是这样的： `要解决[电池耗尽]，您可以尝试[降低屏幕亮度和关闭` `后台应用]`。

## 众包

平台 如 Amazon Mechanical Turk 和 Figure Eight 允许您将创建地面实况数据的任务外包给大量工作者。 通过提供清晰的指示和质量控制措施，您可以获得多样化的响应集。

## 合成地面实况

在获取真实地面实况数据具有挑战性或不切实际的情况下，生成合成地面实况可以是一个可行的替代方案。合成地面实况涉及使用现有的 LLM 或技术自动生成合理的响应。 以下是一些方法：

+   **微调语言模型**：您可以在较小的高质量响应数据集上微调 LLM。 通过向模型提供理想响应的示例，它可以学习为新查询或提示生成类似的响应。 生成的响应可以用作合成 地面实况。

+   **基于检索的方法**：如果您拥有大量高质量的文本数据，您可以使用基于检索的方法来找到与查询或提示紧密匹配的相关段落或句子。 这些检索到的段落可以用作地面实况响应的代理。

获取真实信息是构建您的 RAG 系统中的一个具有挑战性的步骤，但一旦您获得了它，您将为有效的 RAG 评估打下坚实的基础。 在下一节中，我们有一个代码实验室，我们将生成合成真实信息数据，然后整合一个有用的评估平台到我们的 RAG 系统中，这将告诉我们上一章中使用的混合搜索对我们结果的影响。

# 代码实验室 9.1 – ragas

**检索增强生成评估** (**ragas**) 是一个专门为 RAG 设计的评估平台。 在本代码实验室中，我们将逐步实现 ragas 在您的代码中的实现，生成合成真实信息，然后建立一个全面的指标集，您可以将其集成到您的 RAG 系统中。 但评估系统是用来评估某物的，对吧？ 在我们的 代码实验室中，我们将评估什么？

如果您还记得 *第八章*，我们介绍了一种新的检索阶段搜索方法，称为 **混合搜索**。在本 代码实验室中，我们将实现基于密集向量语义的原始搜索，然后使用 ragas 来评估使用混合搜索方法的影响。 这将为您提供一个真实世界的实际工作示例，说明如何在自己的代码中实现一个全面的评估系统！

在我们深入探讨如何使用 ragas 之前，重要的是要注意它是一个高度发展的项目。 随着新版本的发布，新功能和 API 变更经常发生，因此在使用代码 示例时，务必参考文档网站： [https://docs.ragas.io/](https://docs.ragas.io/)

本代码实验室从上一章我们添加的 `EnsembleRetriever` （*代码* *实验室 8.3*）继续进行：

1.  让我们从一些需要安装的新软件包开始： 开始：

    ```py
    $ pip install ragas
    
    $ pip install tqdm -q –user
    
    tqdm package, which is used by ragas, is a popular Python library used for creating progress bars and displaying progress information for iterative processes. You have probably come across the matplotlib package before, as it is a widely used plotting library for Python. We will be using it to provide visualizations for our evaluation metric results.
    ```

1.  接下来，我们需要添加一些与我们刚刚安装的 相关的一些导入：

    ```py
    import tqdm as notebook_tqdm
    
    import pandas as pd
    
    import matplotlib.pyplot as plt
    
    from datasets import Dataset
    
    from ragas import evaluate
    
    from ragas.testset.generator import TestsetGenerator
    
    from ragas.testset.evolutions import (
    
     simple, reasoning, multi_context)
    
    from ragas.metrics import (
    
     answer_relevancy,
    
     faithfulness,
    
     context_recall,
    
     context_precision,
    
    **answer_correctness,**
    
     **answer_similarity**
    
    `tqdm` will give our ragas platform the ability to use progress bars during the time-consuming processing tasks it implements. We are going to use the popular pandas data manipulation and analysis library to pull our data into DataFrames as part of our analysis. The `matplotlib.pyplot as plt` import gives us the ability to add visualizations (charts in this case) for our metric results. We also import `Dataset` from `datasets`. The `datasets` library is an open source library developed and maintained by Hugging Face. The `datasets` library provides a standardized interface for accessing and manipulating a wide variety of datasets, typically focused on the field of `from ragas import evaluate`: The `evaluate` function takes a dataset in the ragas format, along with optional metrics, language models, embeddings, and other configurations, and runs the evaluation on the RAG pipeline. The `evaluate` function returns a `Result` object containing the scores for each metric, providing a convenient way to assess the performance of RAG pipelines using various metrics and configurations.
    ```

1.  **`from ragas.testset.generator import TestsetGenerator`: The `TestsetGenerator` class is used to generate synthetic ground-truth datasets for evaluating RAG pipelines. It takes a set of documents and generates question-answer pairs along with the corresponding contexts. One key aspect of `TestsetGenerator` is that it allows the customization of the test data distribution by specifying the proportions of different question types (e.g., simple, multi-context, or reasoning) using the `distributions` parameter. It supports generating test sets using both LangChain and LlamaIndex document loaders.****

1.  **`from ragas.testset.evolutions import simple, reasoning, multi_context`: These imports represent different types of question evolutions used in the test dataset generation process. These evolutions help create a diverse and comprehensive test dataset that covers various types of questions encountered in real-world scenarios:**

    +   **`from ragas.metrics import…()`: This `import` statement brings in various evaluation metrics provided by the ragas library. The metrics imported include `answer_relevancy`, `faithfulness`, `context_recall`, `context_precision`, `answer_correctness`, and `answer_similarity`. There are currently two more component-wise metrics (context relevancy and context entity recall) that we can import, but to reduce the complexity of this, we will skip over them here. We will talk about additional metrics you can use toward the end of the code lab. These metrics assess different aspects of the RAG pipeline’s performance that relate to the retrieval and generation and, overall, all the end-to-end stages of the active pipeline.****

**`Overall, these imports from the ragas library provide a comprehensive set of tools for generating synthetic test datasets, evaluating RAG pipelines using various metrics, and analyzing the performance results.**

## **`Setting up LLMs/embedding models**

现在，我们将升级我们处理 LLM 和嵌入服务的方式。 使用 ragas，我们在使用的 LLM 数量上引入了更多复杂性；我们希望通过提前设置嵌入服务和 LLM 服务来更好地管理这一点。 让我们看看 代码：

```py
 embedding_ada = "text-embedding-ada-002"
model_gpt35="gpt-3.5-turbo"
model_gpt4="gpt-4o-mini"
embedding_function = OpenAIEmbeddings(
    model=embedding_ada, openai_api_key=openai.api_key)
llm = ChatOpenAI(model=model_gpt35,
    openai_api_key=openai.api_key, temperature=0.0)
generator_llm = ChatOpenAI(model=model_gpt35,
    openai_api_key=openai.api_key, temperature=0.0)
critic_llm = ChatOpenAI(model=model_gpt4,
    openai_api_key=openai.api_key, temperature=0.0)
```

请注意，尽管我们仍然只使用一个嵌入服务，但我们现在有两个不同的 LLM 可以调用。 然而，这个主要目标是建立我们想要直接用于 LLM 的 *主要 LLM，然后是两个额外的 LLM，它们被指定用于评估过程（`llm`)，以及 `generator_llm` 和 `critic_llm`）。

我们有一个好处，那就是有一个更先进的 LLM 可用，ChatGPT-4o-mini，我们可以将其用作评论 LLM，理论上这意味着它可以更有效地评估我们输入给它的内容。 这并不总是如此，或者你可能有一个专门针对评估任务微调的 LLM。 无论如何，将这些 LLM 分离成专门的设计表明了不同的 LLM 可以在 RAG 系统中用于不同的 目的。 你可以从之前初始化 LLM 对象的代码中删除以下行，我们最初使用的是： 

```py
 llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

接下来，我们将添加一个新的 RAG 链来运行相似度搜索（这是我们最初仅使用密集嵌入运行的内容）：

```py
rag_chain_similarity = RunnableParallel(
        {"context": dense_retriever,
        "question": RunnablePassthrough()
}).assign(answer=rag_chain_from_docs)
```

为了使事情更清晰，我们将使用以下名称更新混合 RAG 链： 

```py
rag_chain_hybrid = RunnableParallel(
        {"context": ensemble_retriever,
        "question": RunnablePassthrough()
}).assign(answer=rag_chain_from_docs)
```

请注意，粗体显示的变量发生了变化，它曾经是 `rag_chain_with_source`。现在它被称为 `rag_chain_hybrid`，代表混合 搜索方面。

现在我们将更新我们提交用户查询的原始代码，但这次我们将使用相似度和混合 搜索版本。

创建相似度版本： 

```py
 user_query = "What are Google's environmental initiatives?" result = rag_chain_similarity.invoke(user_query)
retrieved_docs = result['context']
print(f"Original Question to Similarity Search: {user_query}\n")
print(f"Relevance Score: {result['answer']['relevance_score']}\n")
print(f"Final Answer:\n{result['answer']['final_answer']}\n\n")
print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Document {i}: Document ID: {doc.metadata['id']}
        source: {doc.metadata['source']}")
    print(f"Content:\n{doc.page_content}\n")
```

现在，创建 混合版本：

```py
 user_query = "What are Google's environmental initiatives?" result = rag_chain_hybrid.invoke(user_query)
retrieved_docs = result['context']
print(f"Original Question to Dense Search:: {user_query}\n")
print(f"Relevance Score: {result['answer']['relevance_score']}\n")
print(f"Final Answer:\n{result['answer']['final_answer']}\n\n")
print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Document {i}: Document ID: {doc.metadata['id']}
        source: {doc.metadata['source']}")
    print(f"Content:\n{doc.page_content}\n")
```

这两组代码之间的主要区别在于它们展示了我们创建的不同 RAG 链的使用， `rag_chain_similarity` 和 `rag_chain_hybrid`。

首先，让我们看看相似度搜索的输出： 

```py
 Google's environmental initiatives include empowering individuals to take action, working together with partners and customers, operating sustainably, achieving net-zero carbon emissions, water stewardship, and promoting a circular economy. They have implemented sustainability features in products like Google Maps, Google Nest thermostats, and Google Flights to help individuals make more sustainable choices. Google also supports various environmental organizations and initiatives, such as the iMasons Climate Accord, ReFED, and The Nature Conservancy, to accelerate climate action and address environmental challenges. Additionally, Google is involved in public policy advocacy and is committed to reducing its environmental impact through its operations and value chain.
```

接下来是混合搜索的输出：

```py
 Google's environmental initiatives include empowering individuals to take action, working together with partners and customers, operating sustainably, achieving net-zero carbon emissions, focusing on water stewardship, promoting a circular economy, engaging with suppliers to reduce energy consumption and greenhouse gas emissions, and reporting environmental data. They also support public policy and advocacy for low-carbon economies, participate in initiatives like the iMasons Climate Accord and ReFED, and support projects with organizations like The Nature Conservancy. Additionally, Google is involved in initiatives with the World Business Council for Sustainable Development and the World Resources Institute to improve well-being for people and the planet. They are also working on using technology and platforms to organize information about the planet and make it actionable to help partners and customers create a positive impact.
```

说哪个更好可能是主观的，但如果你回顾一下 *第八章*中的代码，每个链的检索机制都会为 LLM 返回一组不同的数据，作为回答用户查询的基础。 你可以在前面的响应中看到这些差异的反映，每个响应都有略微不同的信息和突出显示该信息的不同方面。

到目前为止，我们已经设置了我们的 RAG 系统，使其能够使用两个不同的 RAG 链，一个专注于仅使用相似性/密集搜索，另一个使用混合搜索。这为将拉加斯应用于我们的代码以建立评估我们从中获得的结果的更客观方法奠定了基础。

## 生成合成真实情况

如前所述，**真实情况**是我们进行此评估分析的关键要素。但我们没有真实情况——哦不！ 没问题，我们可以使用拉加斯来生成用于此目的的合成数据。

警告

拉加斯库广泛使用了您的 LLM API。拉加斯提供的分析是 LLM 辅助评估，这意味着每次生成或评估一个真实情况示例时，都会调用一个 LLM（有时为一个指标调用多次），并产生 API 费用。 如果您生成 100 个真实情况示例，包括问题和答案的生成，然后运行六个不同的评估指标，您进行的 LLM API 调用次数将大幅增加，达到数千次。 建议您在使用之前谨慎使用，直到您对调用频率有很好的了解。 这些是产生费用的 API 调用，它们有可能使您的 LLM API 账单大幅增加！ 在撰写本文时，我每次运行整个代码实验室（仅使用 10 个真实示例和 6 个指标）时，费用约为 2 到 2.50 美元。 如果您有一个更大的数据集或设置了 `test_size` ，用于您的 `testset` 生成器以生成超过 10 个示例，费用将大幅增加。

我们首先 创建一个我们将用于生成我们的 真实情况数据集的生成器实例：

```py
 generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embedding_function
)
documents = [Document(page_content=chunk) for chunk in splits]
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=10,
    distributions={
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25
    }
)
testset_df = testset.to_pandas()
testset_df.to_csv(
    os.path.join('testset_data.csv'), index=False)
print("testset DataFrame saved successfully in the local directory.")
```

`如你在代码中所见，我们正在使用` `generator_llm` `和` `critic_llm` `，以及` `embedding_function` `。正如之前` `警告` `框中所述，请注意这一点！` `这是三个不同的 API，如果不小心设置此代码中的设置，它们可能会产生大量成本。` `在此代码中，我们还对我们的早期代码中生成的数据拆分进行预处理，以便更有效地与 ragas 一起工作。` `拆分中的每个` `块` `被假定为表示文档一部分的字符串。` `类` `Document` `来自 LangChain 库，是一种方便的方式，用其内容表示文档。`

`testset` `使用` `generator_with_langchain_docs` `函数` 从我们的生成器对象中生成一个合成测试。`此函数接受文档列表作为输入。` `参数` `test_size` `设置要生成的期望测试题数量（在本例中为 10）。` `参数` `distributions` `定义了问题类型的分布，其中简单问题占数据集的 50%，推理问题占 25%，多上下文问题占 25%，在本例中。` `然后` 将 `testset` `转换为 pandas DataFrame，我们可以用它来查看结果，并将其保存为文件。` `鉴于我们刚才提到的成本，将数据保存到 CSV 文件中，以便在文件目录中持久化，提供了只需运行此` `代码一次` 的额外便利！

`现在让我们将保存的数据集拉回来并查看` `它！`

```py
 saved_testset_df = pd.read_csv(os.path.join('testset_data.csv'))
print("testset DataFrame loaded successfully from local directory.")
saved_testset_df.head(5)
```

`输出应该看起来像你在` `图 9` `.1` `中看到的那样`：

![图 9.1 – 显示合成真实数据的 DataFrame](img/B22475_09_01.jpg)

`图 9.1 – 显示合成真实数据的 DataFrame`

在这个数据集中，你可以看到由`ground_truth`) 生成的 `generator_llm` 实例你之前初始化的。 你现在有了你的基准答案了！ LLM 将尝试为我们的基准答案生成 10 个不同的问答对，但在某些情况下，可能会发生失败，这限制了这种生成。 这将导致 基准答案的例子数量少于你在 `test_size` 变量中设置的。 在这种情况下，生成结果是 7 个例子，而不是 10 个。 总的来说，你可能希望为彻底测试你的 RAG 系统生成超过 10 个例子。 尽管如此，我们在这个简单示例中只接受 7 个例子，主要是为了降低你的 API 成本！

接下来，让我们准备 相似度数据集：

```py
 saved_testing_data = \
    saved_testset_df.astype(str).to_dict(orient='list')
saved_testing_dataset = Dataset.from_dict(saved_testing_data)
saved_testing_dataset_sm = saved_testing_dataset.remove_columns(
    ["evolution_type", "episode_done"])
```

在这里，我们正在进行一些更多的数据转换，以使格式与其他代码部分兼容（在这种情况下是 ragas 输入）。 我们将使用 `saved_testset_df` DataFrame 通过 `to_dict()` 方法转换为字典格式，使用 `orient='list'`，在将所有列转换为字符串类型后使用 `astype(str)`。生成的 `saved_testing_data` 字典随后用于使用 `from_dict()` 方法从 `datasets` 库创建一个名为 `saved_testing_dataset` 的 `Dataset` 对象。 我们创建一个新的数据集 `saved_testing_dataset_sm` ，它代表数据的一个较小部分，只包含我们需要的列。

在这种情况下，我们使用 `remove_columns()` 方法删除了 `evolution_type` 和 `episode_done` 列。 让我们通过在单独的单元中添加此代码来查看：

```py
 saved_testing_dataset_sm
```

输出应该看起来 像这样：

```py
 Dataset({
    features: ['question', 'contexts', 'ground_truth', 'metadata'],
    num_rows: 7
})
```

如果您 有更多的 ground-truth 示例， `num_rows` 变量将反映这一点，但其余部分应该相同。 `Dataset` 对象表示我们拥有的“特征”，代表我们传递给它的列，然后这表明我们有七行 的数据。

接下来，我们将设置一个函数来运行我们传递给它的 RAG 链，然后添加一些额外的格式化，使其能够与 ragas 一起工作： with ragas:

```py
 def generate_answer(question, ground_truth, rag_chain):
    result = rag_chain.invoke(question)
    return {
        "question": question,
        "answer": result["answer"]["final_answer"],
        "contexts": [doc.page_content for doc in result["context"]],
        "ground_truth": ground_truth
    }
```

此块定义了一个 `generate_answer()` 函数，它接受一个问题、 `ground_truth` 数据，以及 `rag_chain` 作为输入。 此函数非常灵活，因为它接受我们提供的任意一条链，这在我们需要生成相似性和混合链的分析时将非常有用。 此函数的第一步是调用 `rag_chain` 输入，该输入已通过给定的问题传递给它，并检索结果。 第二步是返回一个包含问题、结果中的最终答案、从结果中提取的上下文，以及 `ground truth` 的字典。

现在我们已准备好进一步准备我们的数据集以与 ragas 一起工作： with ragas:

```py
 testing_dataset_similarity = saved_testing_dataset_sm.map(
    lambda x: generate_answer(x["question"],
        x["ground_truth"], rag_chain_similarity),
    remove_columns=saved_testing_dataset_sm.column_names)
testing_dataset_hybrid = saved_testing_dataset_sm.map(
    lambda x: generate_answer(x["question"],
        x["ground_truth"], rag_chain_hybrid),
    remove_columns=saved_testing_dataset_sm.column_names)
```

在此代码中，我们通过将 `generate_answer()` 函数应用于每个 RAG 链（相似性和混合）的 `saved_testing_dataset_sm` 行的每一行来创建两个新的数据集， `testing_dataset_similarity` 和 `testing_dataset_hybrid`，使用 `map()` 方法。 `rag_chain_similarity` 和 `rag_chain_hybrid` 分别用作相应数据集创建中的 `rag_chain` 参数。 使用 `remove_columns=saved_testing_dataset_sm.column_names`移除了 `saved_testing_dataset_sm` 的原始列。

最后，让我们在两个数据集上运行 ragas。 以下是应用 ragas 到我们的相似性 RAG 链 的代码：

```py
 score_similarity = evaluate(
    testing_dataset_similarity,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    ]
)
similarity_df = score_similarity.to_pandas()
```

在这里，我们 应用 ragas 来评估 `testing_dataset_similarity` ，使用 ragas 库中的 `evaluate()` 函数。 评估使用指定的指标进行，包括 `faithfulness`、 `answer_relevancy`、 `context_precision`、 `context_recall`、 `answer_correctness`和 `answer_similarity`。评估结果存储在 `score_similarity` 变量中，然后使用 `to_pandas()` 方法将其转换为 pandas DataFrame， `similarity_df`。

我们将对 混合数据集 进行相同的处理：

```py
 score_hybrid = evaluate(
    testing_dataset_hybrid,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    ]
)
hybrid_df = score_hybrid.to_pandas()
```

一旦 你达到这一点，ragas 的使用就完成了！ 我们现在已经使用 ragas 对两条链进行了全面评估，并且在这两个 DataFrame 中， `similarity_df` 和 `hybrid_df`，我们有了所有的指标数据。 我们剩下要做的就是分析 ragas 提供的数据。

## 分析 ragas 的结果

我们将在这段代码实验室的剩余部分中格式化数据，以便我们首先将其保存和持久化（因为，同样，这可能是我们 RAG 系统成本较高的部分）。 这段代码的其余部分可以在将来重用，以从 `.csv` 文件中提取数据，如果您保存它们，这将防止您不得不重新运行这个可能成本较高的 评估过程。

让我们先设置一些重要的变量，然后将我们收集到的数据保存到 `csv` 文件中：

```py
 key_columns = [
    'faithfulness',
    'answer_relevancy',
    'context_precision',
    'context_recall',
    'answer_correctness',
    'answer_similarity'
]
similarity_means = similarity_df[key_columns].mean()
hybrid_means = hybrid_df[key_columns].mean()
comparison_df = pd.DataFrame(
        {'Similarity Run': similarity_means,
        'Hybrid Run': hybrid_means})
comparison_df['Difference'] = comparison_df['Similarity Run'] \
        - comparison_df['Hybrid Run']
similarity_df.to_csv(
    os.path.join('similarity_run_data.csv'), index=False)
hybrid_df.to_csv(
    os.path.join('hybrid_run_data.csv'), index=False)
comparison_df.to_csv(os.path.join('comparison_data.csv'), index=True)
print("Dataframes saved successfully in the local directory.")
```

在这段 代码中，我们首先定义了一个 `key_columns` 列表，包含用于比较的关键列的名称。 然后，我们使用 `mean()` 方法计算 `similarity_df` 和 `hybrid_df` 中每个关键列的平均分数，并将它们分别存储在 `similarity_means` 和 `hybrid_means`中。

接下来，我们创建一个新的 DataFrame，称为 `comparison_df` ，用于比较相似性运行和混合运行的均值分数。 在 `Difference` 列被添加到 `comparison_df`中，计算为相似性运行和混合运行均值分数之间的差异。 最后，我们将 `similarity_df`，`hybrid_df`，和 `comparison_df` DataFrame 保存为 `.csv` 文件。 我们将再次保存文件，我们可以在未来从这些文件中工作，而无需返回并重新生成一切。

此外，请记住，这只是进行分析的一种方式。 这就是你想要发挥创意并调整此代码以进行关注你特定 RAG 系统中重要方面的分析的地方。 例如，你可能只专注于改进你的检索机制。 或者，你可以将此应用于从部署环境中流出的数据，在这种情况下，你可能没有真实数据，并希望关注可以在没有真实数据的情况下工作的指标（有关该概念的更多信息，请参阅本章后面的 *ragas 创始人见解* 部分）。

尽管如此，我们继续分析，现在我们希望将我们保存的文件拉回来以完成我们的分析，然后打印出我们对 RAG 系统两个不同链的每个阶段的分析的总结： ：

```py
 sem_df = pd.read_csv(os.path.join('similarity_run_data.csv'))
rec_df = pd.read_csv(os.path.join('hybrid_run_data.csv'))
comparison_df = pd.read_csv(
    os.path.join('comparison_data.csv'), index_col=0)
print("Dataframes loaded successfully from the local directory.")
print("Performance Comparison:")
print("\n**Retrieval**:")
print(comparison_df.loc[['context_precision', 'context_recall']])
print("\n**Generation**:")
print(comparison_df.loc[['faithfulness', 'answer_relevancy']])
print("\n**End-to-end evaluation**:")
print(comparison_df.loc[['answer_correctness', 'answer_similarity']])
```

此代码段将生成一系列我们将进一步检查的指标。 我们首先从我们在前面的代码块中生成的 CSV 文件中加载 DataFrame。 然后，我们应用一个分析，将所有内容整合为 更易于阅读的分数。

我们继续前进，使用我们在前面的代码块中定义的变量来帮助生成 使用 `matplotlib`:

```py
 fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
bar_width = 0.35
categories = ['Retrieval', 'Generation', 'End-to-end evaluation']
metrics = [
    ['context_precision', 'context_recall'],
    ['faithfulness', 'answer_relevancy'],
    ['answer_correctness', 'answer_similarity']
]
```

在这里，我们 正在为每个类别创建子图，并 增加间距。

接下来，我们将遍历这些类别中的每一个，并绘制 相应的指标：

```py
 for i, (category, metric_list) in enumerate(zip(categories, metrics)):
    ax = axes[i]
    x = range(len(metric_list))
    similarity_bars = ax.bar(
    x, comparison_df.loc[metric_list, 'Similarity Run'],
    width=bar_width, label='Similarity Run',
    color='#D51900')
    for bar in similarity_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height, f'{height:.1%}', ha='center',
            va='bottom', fontsize=10)
    hybrid_bars = ax.bar(
        [i + bar_width for i in x],
        comparison_df.loc[metric_list, 'Hybrid Run'],
        width=bar_width, label='Hybrid Run',
        color='#992111')
    for bar in hybrid_bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height, f'{height:.1%}', ha='center',
            va='bottom', fontsize=10)
    ax.set_title(category, fontsize=14, pad=20)
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(metric_list, rotation=45,
        ha='right', fontsize=12)
    ax.legend(fontsize=12, loc='lower right',
        bbox_to_anchor=(1, 0))
```

大部分 代码都集中在格式化我们的可视化上，包括为相似性和混合运行绘制条形图，以及向这些条形图中添加值。 我们给条形图添加了一些颜色，甚至添加了一些散列来提高对 视觉障碍者的可访问性。

我们只需对 可视化 进行一些更多的改进：

```py
 fig.text(0.04, 0.5, 'Scores', va='center',
    rotation='vertical', fontsize=14)
fig.suptitle('Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.6, top=0.92)
plt.show()
```

在这段代码中，我们添加了标签和标题到我们的可视化中。 我们还调整了子图之间的间距并增加了顶部边距。 然后，最后，我们 `使用 plt.show()` 在笔记本界面中显示可视化。

总的来说，本节中的代码将生成一个基于文本的分析，显示你从两个链中获取的结果，然后以条形图的形式生成一个比较结果的可视化。 虽然代码会一起生成所有这些内容，但我们将将其拆分，并讨论输出中的每个部分，以及它与我们的 RAG 系统主要阶段的相关性。 我们将讨论每个部分与我们的 RAG 系统主要阶段的相关性。 RAG 系统。

正如我们在前面的章节中讨论的，当 RAG 被激活时，它有两个主要的行为阶段：检索和生成。 在评估一个 RAG 系统时，你也可以根据这两个类别来分解你的评估。 让我们首先来谈谈 如何评估检索。

## 检索评估

Ragas 提供了用于评估 RAG 管道每个阶段的指标。 对于检索，ragas 有两个指标，称为 **上下文精确度** 和 **上下文召回率**。你 可以在输出部分的这里和图表中看到这一点： 和图表：

```py
 Performance Comparison:
**Retrieval**:
                   Similarity Run  Hybrid Run  Difference
context_precision        0.906113    0.841267    0.064846
context_recall           0.950000    0.925000    0.025000
```

你可以在 *图 9**.2**中看到检索指标的图表：

![图 9.2 – 显示相似性搜索和混合搜索之间检索性能比较的图表](img/B22475_09_02.jpg)

图 9.2 – 显示相似性搜索和混合搜索之间检索性能比较的图表

检索评估 侧重于评估检索到的文档的准确性和相关性。 我们使用 ragas 的这两个指标来完成这项工作，如 ragas 文档网站 上所述：

+   `context_precision` 是一个指标，用于评估上下文中是否所有真实相关项都被排名更高。 理想情况下，所有相关片段都必须出现在顶部排名。 此指标使用问题， `ground_truth`，和 `contexts`来计算，其值介于 `0` 和 `1`之间，更高的分数表示 更好的精确度。

+   `context_recall` 衡量检索到的上下文与作为真实情况的标注答案的一致性程度。 它是基于真实情况和检索到的上下文计算的，其值介于 `0` 和 `1`之间，更高的值表示 更好的性能。

如果你来自传统的数据科学或信息检索背景，你可能已经认识到了术语 *精确度* 和 *召回度* ，并想知道这些术语之间是否有任何关系。 ragas 中使用的上下文精确度和召回度指标在概念上与传统精确度和召回度指标相似。

在传统意义上，精确度衡量检索到的相关项目的比例，而召回度衡量被检索到的相关项目的比例。 。

类似地，上下文精确度通过评估是否真实相关项被排名更高来评估检索到的上下文的相关性，而上下文召回度衡量检索到的上下文覆盖所需回答问题的相关信息的程度。 。

然而，也有一些关键的区别 需要注意。

传统的精确度和召回度通常基于每个项目的二元相关性判断（相关或不相关）来计算，而 ragas 中的上下文精确度和召回度考虑了检索到的上下文与真实答案的排名和一致性。 此外，上下文精确度和召回度专门设计来评估问答任务中的检索性能，考虑到检索相关信息以回答一个 特定问题的具体要求。

在查看我们的分析结果时，我们需要记住，我们使用的小数据集作为我们的基准。 事实上，我们整个 RAG 系统基于的原始数据集很小，这也可能影响我们的结果。 因此，我不会太在意您在这里看到的数字。 但这一点确实向您展示了如何使用 ragas 进行分析，并提供关于我们 RAG 系统检索阶段发生情况的非常有信息量的表示。 这个代码实验室主要是为了演示您在构建 RAG 系统时可能会遇到的真实世界挑战，在您的特定用例中，您必须考虑不同的指标，这些不同指标之间的权衡，以及必须决定哪种方法以最有效的方式满足您的需求。 最有效的方式。

接下来，我们将回顾我们 RAG 系统生成阶段的类似分析。

## 生成评估

如所述，ragas 为评估 RAG 管道每个阶段的独立指标提供了度量标准。对于生成阶段，ragas 有两个指标，称为忠实度 和 答案相关性，正如您在这里的输出部分和以下图表中看到的那样：

```py
 **Generation**:
                  Similarity Run  Hybrid Run  Difference
faithfulness            0.977500    0.945833    0.031667
answer_relevancy        0.968222    0.965247    0.002976
```

生成指标可以在*图 9**.3**中的图表中看到：

![图 9.3 – 比较相似性搜索和混合搜索生成性能的图表](img/B22475_09_03.jpg)

图 9.3 – 比较相似性搜索和混合搜索生成性能的图表

生成评估衡量的是当提供上下文时，系统生成的响应的适当性。 我们使用 ragas 通过以下两个指标来完成这项工作，如 ragas 文档中所述： ragas 文档：

+   `忠实度`：生成的答案在事实上的准确性如何？ 这衡量的是生成的答案与给定上下文的事实一致性。 它从答案和检索到的上下文中计算得出。 答案被缩放到（`0-1`）范围内，分数越高表示越好。

+   `answer_relevancy`：生成的答案与问题有多相关？ 答案相关性侧重于评估生成的答案与给定提示的相关性。 对于不完整或包含冗余信息的答案，将分配较低的分数，而较高的分数表示更好的相关性。 此指标使用问题、上下文和 答案来计算。

再次，我想 重申，我们正在使用一个小数据集作为我们的真实值和数据集，这可能会使这些结果不太可靠。 但您可以看到这里这些结果如何成为为我们 RAG 系统的生成阶段提供非常有信息量的表示的基础。 RAG 系统。

这使我们来到了下一组指标，即端到端评估指标，我们将在下文中进行讨论。 讨论。

# 端到端评估

除了提供评估 RAG 管道每个阶段的指标外，ragas 还提供了整个 RAG 系统的指标，称为端到端 评估。 对于生成阶段，ragas 有两个 指标，称为 **答案正确性** 和 **答案相似度**，正如您在输出 和图表的最后一部分所看到的：

```py
 **End-to-end evaluation**:
                    Similarity Run  Hybrid Run  Difference
answer_correctness        0.776018    0.717365    0.058653
answer_similarity         0.969899    0.969170    0.000729
```

图*9**.4* 中的图表显示了这些结果的 可视化：

![图 9.4 – 展示相似性搜索和混合搜索之间端到端性能比较的图表](img/B22475_09_04.jpg)

图 9.4 – 展示相似性搜索和混合搜索之间端到端性能比较的图表

端到端 指标用于评估管道的端到端性能，衡量使用管道的整体体验。 结合这些指标提供了对 RAG 管道的全面评估。 我们使用 ragas 通过以下两个指标来完成这项工作，如 ragas 文档中所述： ragas 文档：

+   `answer_correctness`：衡量生成的答案与真实值相比的准确性。 答案正确性的评估涉及衡量生成的答案与真实值相比的准确性。 这种评估依赖于真实值和答案，分数范围从 0 到 1。 更高的分数表示生成的答案与真实值之间的接近程度更高，意味着 更好的正确性。

+   `answer_similarity`：评估生成的答案与 ground truth 之间的语义相似度。 答案语义相似度的概念涉及对生成的答案与 ground truth 之间的语义相似度的评估。 这种评估基于 ground truth 和答案，其值在`0` 到`1`之间。更高的分数表示生成的答案与 ground truth 之间的对齐更好。

评估 管道的端到端性能也非常关键，因为它直接影响用户体验，并有助于确保 全面的评估。

为了使这个代码实验室保持简单，我们省略了一些你可能也在分析中考虑的更多指标。接下来，让我们谈谈这些指标。

# 其他分项评估

分项评估涉及评估管道的各个组成部分，例如检索和生成阶段，以了解它们的有效性并确定改进领域。 我们已经分享了这些阶段中每个阶段的两个指标，但这里还有一些在 ragas 平台中可用的指标：

+   `(0-1)`，更高的值表示 更好的相关性。

+   `ground_truth` 数据和`contexts` 数据相对于`ground_truth` 数据中存在的实体数量。 简单来说，这是一个衡量从`ground_truth` 数据中召回的实体比例的指标。 这个指标在基于事实的使用案例中特别有用，例如旅游帮助台和历史问答。 这个指标可以帮助评估实体检索机制，通过与`ground_truth` 数据中的实体进行比较，因为在这种情况下，实体很重要，我们需要涵盖它们的上下文。

+   **方面批评**：方面批评旨在根据预定义的方面（如无害性和正确性）评估提交内容。 此外，用户可以根据其特定标准定义自己的方面来评估提交内容。 方面批评的输出是二进制的，表示提交是否与定义的方面一致。 此评估使用“*答案*” 作为输入。

这些额外的按组件评估指标提供了进一步细化以评估检索到的上下文和生成的答案。 为了完成这个代码实验室，我们将 引入一些创始人直接提供的见解，以帮助您进行 RAG 评估。

创始人视角

为了准备这一章，我们有机会与 ragas 的创始人之一 Shahul Es 交谈，以获得对平台以及如何更好地利用它进行 RAG 开发和评估的额外见解。 Ragas 是一个年轻平台，但正如您在代码实验室中看到的，它已经建立了一个坚实的指标基础，您可以使用这些指标来评估您的 RAG 系统。 但这也意味着 ragas 有很大的成长空间，这个专门为 RAG 实现构建的平台将继续发展。 Shahul 提供了一些有用的提示和见解，我们将在此总结并分享给您。 我们将在以下部分分享那次讨论的笔记。

## Ragas 创始人见解

以下 是从与 ragas 联合创始人 Shahul Es 的讨论中摘录的笔记，讨论了如何使用 ragas 进行 RAG 评估：

+   **合成数据生成**：人们在 RAG 评估中通常遇到的第一大障碍是没有足够的测试真实数据。 Ragas 的主要重点是创建一个算法，可以创建一个覆盖广泛问题类型的测试数据集，从而产生其合成数据生成能力。 一旦您使用 ragas 合成您的真实数据，检查生成的真实数据并挑选出任何不属于的任何问题将是有帮助的。

+   **反馈指标**：他们在开发中目前强调的是将各种反馈循环从性能和用户反馈中纳入评估，其中存在明确的指标（出了问题）和隐含指标（满意度水平、点赞/踩和类似机制）。 与用户的任何互动都可能具有隐含性。 隐含反馈可能嘈杂（从数据角度来看），但如果 使用得当，仍然可以是有用的。

+   **参考和非参考指标**：Shahul 将这些指标分为参考指标和非参考指标，其中参考意味着它需要真实数据进行处理。 ragas 团队在其工作中强调构建非参考指标，您可以在 ragas 论文中了解更多信息（[https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)）。 对于许多领域，由于难以收集真实数据，这是一个重要的点，因为这至少使得 一些评估仍然成为可能。 Shahul 提到了忠实度和答案相关性是非参考指标 。

+   **部署评估**：非参考评估指标也适用于部署评估，在这种情况下，您不太可能有一个可用的真实 数据。

这些都是一些关键见解，看到未来 ragas 的发展将如何帮助我们所有人不断改进我们的 RAG 系统将会非常令人兴奋。 您可以在以下位置找到最新的 ragas 文档 ： [https://docs.ragas.io/en/stable/](https://docs.ragas.io/en/stable/)

这就是我们使用 ragas 进行的评估代码实验室的结束。 但 ragas 并不是用于 RAG 评估的唯一工具；还有更多！ 接下来，我们将讨论一些您 可以考虑的其他方法。

# 额外的评估技术

Ragas 只是众多评估工具和技术中的一种，可用于评估您的 RAG 系统。 这不是一个详尽的列表，但在接下来的小节中，我们将讨论一些更受欢迎的技术，您可以使用这些技术来评估您的 RAG 系统性能，一旦您获得或生成了 真实数据。

## 双语评估助理（BLEU）

BLEU 衡量生成响应和真实响应之间的 n-gram 重叠 。 它提供了一个表示两者之间相似度的分数。 在 RAG 的背景下，BLEU 可以通过将生成的答案与真实答案进行比较来评估生成答案的质量。 通过计算 n-gram 重叠，BLEU 评估生成的答案在词汇选择和措辞方面与参考答案的匹配程度。 然而，需要注意的是，BLEU 更关注表面相似性，可能无法捕捉到生成答案的语义意义或相关性。 。

## 用于摘要评估的召回率导向的辅助研究（ROUGE）

ROUGE 通过比较生成的响应与真实值在召回率方面的表现来评估生成响应的质量 来衡量真实值中有多少被捕获在生成的响应中。 它衡量真实值中有多少被捕获在生成的响应中。 在 RAG 评估中，ROUGE 可以用来评估生成答案的覆盖率和完整性。 通过计算生成的答案和真实答案之间的召回率，ROUGE 评估生成的答案在多大程度上捕获了参考答案中的关键信息和细节。 当真实答案更长或更详细时，ROUGE 特别有用，因为它关注的是信息的重叠，而不是精确的 单词匹配。

## 语义相似度

指标 如余弦相似度或 **语义文本相似度** (**STS**)可以用来评估生成的响应与真实值之间的语义相关性。 这些指标捕捉了超出精确单词匹配的意义和上下文。 在 RAG 评估中，语义相似度指标可以用来评估生成答案的语义一致性和相关性。 通过比较生成的答案和真实答案的语义表示，这些指标评估生成的答案在多大程度上捕捉了参考答案的潜在意义和上下文。 当生成的答案可能使用不同的单词或措辞，但仍然传达与真实值相同的意义时，语义相似度指标特别有用。

## 人工评估

虽然自动指标提供了定量评估，但人类评估在评估生成的响应与事实真相相比的连贯性、流畅性和整体质量方面仍然很重要。 在 RAG 的背景下，人类评估涉及让人类评分者根据各种标准评估生成的答案。 这些标准可能包括与问题的相关性、事实的正确性、答案的清晰度以及整体连贯性。 人类评估者可以提供自动化指标可能无法捕捉到的定性反馈和见解，例如答案语气的适当性、是否存在任何不一致或矛盾，以及整体用户体验。 人类评估可以通过提供对 RAG 系统性能的更全面和细致的评估来补充自动化指标。

在评估 RAG 系统时，通常有益于结合使用这些评估技术，以获得对系统性能的整体看法。 每种技术都有其优势和局限性，使用多个指标可以提供更稳健和全面的评估。 此外，在选择适当的评估技术时，考虑您 RAG 应用程序的具体需求和目标也很重要。 某些应用程序可能优先考虑事实的正确性，而其他应用程序可能更关注生成的答案的流畅性和连贯性。 通过将评估技术与您的具体需求对齐，您可以有效地评估 RAG 系统的性能并确定改进领域。

# 总结

在本章中，我们探讨了评估在构建和维护 RAG 管道中的关键作用。 我们讨论了评估如何帮助开发者识别改进领域、优化系统性能，并在整个开发过程中衡量修改的影响。 我们还强调了在部署后评估系统的重要性，以确保持续的效力、可靠性和性能。

我们介绍了 RAG 管道各个组件的标准化评估框架，例如嵌入模型、向量存储、向量搜索和 LLMs。 这些框架为比较不同模型和组件的性能提供了有价值的基准。 我们强调了在 RAG 评估中真实数据的重要性，并讨论了获取或生成真实数据的方法，包括人工标注、专家知识、众包和合成真实数据生成。

本章包含了一个动手代码实验室，我们将 ragas 评估平台集成到我们的 RAG 系统中。我们生成了合成的真实数据，并建立了一套全面的指标来评估使用混合搜索与原始密集向量语义搜索相比的影响。我们探讨了 RAG 评估的不同阶段，包括检索评估、生成评估和端到端评估，并分析了我们的评估结果。代码实验室提供了一个在 RAG 管道中实施全面评估系统的真实世界示例，展示了开发者如何利用评估指标来获得洞察力，并做出数据驱动的决策以改进他们的 RAG 管道。我们还能够分享 ragas 创始人之一的关键见解，以进一步帮助您的 RAG 评估工作。

在下一章中，我们将开始讨论如何以最有效的方式利用 LangChain 与 RAG 系统的关键组件。

# 参考文献

**MSMARCO**：[`microsoft.github.io/msmarco/`](https://microsoft.github.io/msmarco/)

**HotpotQA**：[`hotpotqa.github.io/`](https://hotpotqa.github.io/)

**CQADupStack**：[`nlp.cis.unimelb.edu.au/resources/cqadupstack/`](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/)

**Chatbot** **Arena**：[`chat.lmsys.org/?leaderboard`](https://chat.lmsys.org/?leaderboard)

**MMLU**：[`arxiv.org/abs/2009.03300`](https://arxiv.org/abs/2009.03300)

**MT** **Bench**：[`arxiv.org/pdf/2402.14762`](https://arxiv.org/pdf/2402.14762)
