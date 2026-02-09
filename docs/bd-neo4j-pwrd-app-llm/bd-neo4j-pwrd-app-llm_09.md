

# 第七章：介绍用于构建推荐系统的 Neo4j Spring AI 和 LangChain4j 框架

在前面的章节中，我们探讨了基于 Haystack 和 Python 的智能应用。虽然 Python 是数据科学家偏爱的语言框架，但在某些场景中，我们可能需要其他框架来构建解决方案。另一个值得考虑的流行语言框架是 Java。Java 比 Python 运行速度快，能够以无缝的方式集成到各种数据源中，并且是构建基于 Web 的应用程序以及 Spring 框架中最常用的语言。为此，在接下来的几章中，我们将探讨如何基于**大型语言模型**（**LLMs**）和 Neo4j 构建智能应用。

此外，我们一直专注于利用 LLMs 的能力来构建智能搜索应用。尽管这只是其中一个方面；LLMs 在构建和使用知识图谱以增强推荐系统方面也可以是伟大的工具。在本章中，我们将了解推荐系统以及为什么个性化推荐很重要。我们将简要介绍推荐系统的传统基于规则的途径，并讨论其一些不足之处。然后，我们将向您介绍 LangChain4j 和 Spring AI 框架，以及它们如何支持您构建智能推荐系统。

在本章中，我们将涵盖以下主要主题：

+   理解扩展的 Neo4j 能力以构建智能应用

+   个性化推荐

+   介绍 Neo4j 的 LangChain4j 和 Spring AI 框架

+   Neo4j GenAI 生态系统中智能推荐系统的概述

# 技术要求

虽然本章重点在于个性化推荐并介绍了 LangChain4j 和 Spring AI 框架，但本节没有特定的技术要求。

然而，如果您对 Spring 应用还不熟悉，可以参考[`spring.io/guides/gs/spring-boot`](https://spring.io/guides/gs/spring-boot)上的文档来熟悉 Spring Boot。在接下来的章节中，我们将使用一个带有内置 Web 框架的 Spring Boot 应用程序。您还需要在系统上安装 Java。推荐使用 Java 17 或 19。

# 理解扩展的 Neo4j 能力以构建智能应用

在前面的章节中，我们探讨了如何使用 LLMs 和 Neo4j 构建优秀的搜索应用。虽然知识图谱为构建智能搜索应用提供了很好的上下文，但它们也可以是构建个性化推荐应用的坚实基础。

为了从数据中提取智能并构建超越基本流程分析更好、更智能的应用，我们需要比图数据库功能更多的能力。这正是 Neo4j 作为数据库的能力可以帮助构建更好应用的地方。

其中一些能力如下所示：

+   **可扩展性**：Neo4j 使我们能够构建大型图，使用分片构建联邦图以处理大型数据集。它能够扩展以满足数据增长和业务需求，同时最小化成本。您可以在[`neo4j.com/docs/operations-manual/current/database-administration/composite-databases/concepts/`](https://neo4j.com/docs/operations-manual/current/database-administration/composite-databases/concepts/)了解更多关于这些功能的信息。

+   **安全性**：通过利用角色，Neo4j 实现了数据安全。存在一些角色可以提供高级别的安全性，例如谁可以读取或写入数据库。它还提供了更细粒度的安全控制，根据角色定义可以读取哪些数据。采用这种方法，一个用户可能正在查看图的一部分，而另一个用户根据分配的角色查看图的不同部分。您可以在[`neo4j.com/docs/operations-manual/current/authentication-authorization/`](https://neo4j.com/docs/operations-manual/current/authentication-authorization/)了解更多关于这些功能的信息。

+   **灵活的部署架构**：Neo4j 的集群架构提供了多种部署选项，可以水平扩展以处理更高的读取量，并将读取本地化到不同的服务器，以最小化数据增长时的拥有成本。您可以在[`neo4j.com/docs/operations-manual/current/clustering/introduction/`](https://neo4j.com/docs/operations-manual/current/clustering/introduction/)了解更多关于 Neo4j 集群功能的信息。

+   **图数据科学算法**：Neo4j 图数据科学算法能够从连接数据中解锁隐藏的洞察。这些算法涵盖了路径查找、节点相似度、中心性和社区检测，以及机器学习方面的链接预测和节点分类。您可以在[`neo4j.com/docs/graph-data-science/current/`](https://neo4j.com/docs/graph-data-science/current/)了解更多关于 Neo4j 图数据科学功能的信息。

+   **向量索引**：Neo4j 提供了向量索引功能，以索引嵌入，以便能够查找相似的节点，然后利用图遍历提供更准确的结果。您可以在[`neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/`](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)了解更多关于其向量索引功能的信息。

作为图数据库的 Neo4j 使得轻松处理连接数据变得容易，上述功能超越了连接数据，帮助我们构建可扩展且复杂的智能应用。

**注意**

如果您想了解更多关于搜索和推荐系统

这些文章可能有所帮助：

搜索和推荐有何区别：[`medium.com/understanding-recommenders/whats-the-difference-between-search-and-recommendation-c32937506a29`](https://medium.com/understanding-recommenders/whats-the-difference-between-search-and-recommendation-c32937506a29)

搜索和推荐有何相同之处，有何不同之处？[`gist.github.com/veekaybee/2cf54ebcbd72aa73bfe482f20866c6ef`](https://gist.github.com/veekaybee/2cf54ebcbd72aa73bfe482f20866c6ef)

我们将在接下来的章节中利用 Neo4j 的能力来构建智能推荐系统。在此之前，让我们讨论一下推荐引擎是什么，以及个性化如何帮助创建智能推荐系统。

# 个性化推荐

**推荐系统**是一个基于用户的购买和搜索偏好向用户推荐产品的应用程序。这一方面不仅限于产品定位，还用于医疗诊断和治疗。例如，推荐可以帮助理解患者对药物的反应以及哪种治疗顺序更有效。

随着数据的增长和可用产品的增加，理解用户行为并提供最个性化的推荐变得越来越重要。

这些策略可以用来构建个性化体验。以下是一些提到的策略：

+   **构建用户档案**：我们可以通过理解用户行为来构建自定义用户档案。行为模式可以包括用户在特定时间段内进行的交易顺序或事件的结果，以及其他属性，如年龄、种族和性别。我们可以使用这些方面将用户分成不同的组，并为每个组创建档案。

+   **提供上下文支持**：一旦有了用户档案，我们就应该能够提供更有意义和上下文相关的支持给用户。这可能包括基于最后一次购买的产品推荐购买产品，或者基于当前治疗水平和当前症状的下一剂药物。这些推荐不仅考虑了最近发生的事件，还可以考虑其他用户属性，以提供更直接的支持。

+   **提供自助体验**：除了根据需要提供上下文支持外，还可以使用推荐来提供更令人满意的自我服务体验。用户应该能够更改用于推荐的特性，从而提供一个能够根据用户事件调整其响应方式的系统。

+   **整合反馈**：使用所有上述策略，可以整合正面和负面反馈，以便系统能够根据需要适应个别用户的要求。

个性化推荐提供了许多优势，包括基于当前视图建议下一个产品、根据用户行为提供激励、提升品牌声誉、优化患者治疗方案、更有效地推广新药、改进供应链流程以及确定最佳配送路线。这些定制化建议使企业能够向客户提供更相关和有影响力的体验。

这些是一些推荐可以使用的方法。推荐系统的一些其他有趣的用例可能包括提升销售额([`neo4j.com/developer-blog/graphs-acceleration-frameworks-recommendations/`](https://neo4j.com/developer-blog/graphs-acceleration-frameworks-recommendations/))、管理供应链([`neo4j.com/developer-blog/supply-chain-neo4j-gds-bloom/`](https://neo4j.com/developer-blog/supply-chain-neo4j-gds-bloom/))以及执行患者旅程映射([`www.graphable.ai/blog/patient-journey-mapping/`](https://www.graphable.ai/blog/patient-journey-mapping/))。

让我们来看看传统的基于规则的推荐系统方法以及为什么这种方法对于构建智能和个性化的推荐系统来说是不够的。

## 传统方法的局限性

传统上，推荐系统使用**基于规则的系统**。基于规则的系统是指决策是通过执行基于提供的数据输入的一组规则来做出的。逻辑可以是简单的，也可以根据需要非常复杂。例如，在特定地区，任何超过 1000 美元的信用卡交易都会自动拒绝。一个稍微复杂一些的规则可能是，当一个小交易成功执行后，然后尝试进行大交易时，会拒绝该交易。

基于规则的系统通常应用两种类型的规则：

+   **静态规则**：在这里，规则是手动配置的。一旦这些规则被设置好，它们可以非常高效地工作，系统可以忠实执行它们。当您需要快速响应且资源消耗最少时，它们是好的。它们可以像基于输入返回值的 case 语句一样简单。

+   **动态规则**：这些是复杂的规则引擎。在这些情况下，下一个决策可以依赖于当前决策树所处的状态和下一个数据输入。

使用基于规则的系统的一些好处如下：

+   **一致性**：它们的行为是一致的，并保证对于给定的输入或输入集，输出是相同的。

+   **可扩展性**：这些系统可以很好地扩展以轻松处理数据和复杂性。

+   **高效**：在资源消耗和系统成本方面，它们非常高效。

+   **维护和管理**：这些规则更容易构建和维护。这反过来使得管理这些系统变得容易。

通常，这些系统的用例是欺诈预防和网络安全。虽然这些系统简单且易于构建，但它们存在局限性。以下是一些例子：

+   **复杂性**：如果处理不当，随着业务需求的增加，它们可能会变得相当复杂。随着复杂性的增加，大多数好处将逐渐开始消失。

+   **僵化性**：系统过于僵化，难以适应新的数据类型和场景。即使我们识别出新的场景，编码和配置它们可能也需要太长时间才能有效。

+   **业务需求适应性**：适应不断增长的业务需求和要求的这些系统可能需要太多的努力。

正如我们所见，随着业务需求的发展，当我们依赖于基于规则的系统时，我们会陷入有限的选项。构建一个能够适应新的数据点和数据复杂性，以提供更好的上下文并给出良好建议的智能应用变得越来越重要。这些系统应该能够快速适应不断变化的环境、数据和新的要求。

正是这里，Neo4j 作为图数据库及其周围的技术堆栈帮助我们构建智能推荐系统。让我们来看看如何实现。

# 介绍 Neo4j 的 LangChain4j 和 Spring AI 框架

要构建智能应用，我们可以利用围绕 Neo4j 可用的多个框架。对于智能推荐系统的特定用例，我们将探讨 Java 框架 Spring AI 和 LangChain4j。

## LangChain4j

**LangChain4j** ([`github.com/langchain4j/`](https://github.com/langchain4j/)) 是一个受流行的 Python LangChain 框架启发的 Java 框架，用于在 Java 中构建 LLM 应用程序。其目标是简化将 LLM API 集成到 Java 应用程序中。为此，它构建了一个结合了 LangChain、Haystack、LlamaIndex 和其他概念的 API，并为构建复杂应用增添了独特的风味。这就是它实现这些目标的方式。

以下列表帮助我们了解它是如何实现这些目标的：

+   **统一 API**：所有大型语言模型（LLM）提供商，如 Open AI 和 Google Gemini，都有自己的专有 API 来构建应用程序。像 Neo4j、Pinecone 和 Milvus 这样的向量存储也提供自己的 API 来存储和检索嵌入。LangChain4j 提供了一个统一的 API，以隐藏所有这些 API 的复杂性，使开发更加容易。

+   **全面的工具箱**：LangChain 社区已经识别出各种模式、抽象和技术，以构建大量的 LLM 应用程序和示例，并提供现成的包以加速开发。其工具箱包括低级提示模板、聊天内存管理、AI 服务和 RAG 的示例。其中大部分示例都易于集成到其他应用程序中。

LangChain4j 提供以下功能，帮助我们构建智能应用：

+   **超过 15 个 LLM 提供商**：LangChain4j 提供了一个简单的 API，将 LLM 提供商集成到应用中并轻松使用。您可以在[`docs.langchain4j.dev/category/language-models`](https://docs.langchain4j.dev/category/language-models)上了解更多关于语言模型集成的内容。

+   **超过 20 个向量存储**：向量存储 API 允许存储生成的嵌入并查询它们。以下是您要查看的向量存储 API：[`docs.langchain4j.dev/tutorials/embedding-stores`](https://docs.langchain4j.dev/tutorials/embedding-stores)。

+   **AI 服务**：LangChain4j 提供了低级 API，例如直接与 LLM 提供商和向量存储交互的 API。但对于某些场景来说，这可能太底层了。为了简化操作，它还提供了更高级的 API 流程来集成 LLM、向量存储、嵌入模型和 RAG 作为管道。这些被称为 AI 服务([`docs.langchain4j.dev/tutorials/ai-services`](https://docs.langchain4j.dev/tutorials/ai-services))。我们将在接下来的章节中使用 AI 服务。

+   **RAG**：LangChain4j 提供了对 RAG 索引和检索阶段的支持。它有一个简单的**Easy RAG**功能，使得开始使用 RAG 功能变得容易。您可以在[`docs.langchain4j.dev/tutorials/rag`](https://docs.langchain4j.dev/tutorials/rag)上了解更多关于 LangChain4j 提供的 RAG 能力。

LangChain4j 与 Spring 框架有良好的集成。但 Apache Spring 框架也构建了一个类似于 LangChain4j 的独立 AI 集成框架，称为 Spring AI。我们将在下一节中查看这个框架。

## Spring AI

**Spring AI**受到 LangChain4j 和 LlamaIndex 的启发。虽然 LangChain4j 支持简单的 Java 应用以及 Spring 应用，但 Spring AI 针对与 Spring 框架协同工作进行了优化。这意味着熟悉 Spring 框架的开发者可以更快、更轻松地开发 LLM 应用。

由于 Spring 框架提供了多个模块来连接各种数据库和许多开发者定义和使用的良好编码模式，这个新特性使得开发者能够非常容易地采用并快速构建 AI 应用。以下是一些 Spring AI 能力，它们可以帮助我们构建智能应用：

+   **LLM 提示模板**：LLM 提示模板提供了一个简单的 API，以便轻松集成 LLM。

+   **嵌入模型**：Spring AI 可以通过配置集成各种嵌入模型引擎，以生成向量嵌入。

+   **向量存储**：Spring AI 还提供了简单的 API 来存储和查询向量存储。它提供了基于配置的简单集成，以便连接到各种向量存储，如 Neo4j、Pinecone 和 Milvus。

+   **RAG**：您还可以使用 Spring AI 将 LLM 提示模板、嵌入模型和向量存储链接起来，构建有效的 RAG 应用。

LangChain4j 和 Spring AI 框架都提供了核心 API，用于与 LLM 聊天模型、提示模板、嵌入模型和向量存储集成。除了提供与这些系统通信的低级 API 外，它们还使使用高级 API（如 RAG 框架 API）构建更复杂的应用程序变得容易。

## 为什么选择基于 Java 的框架？

在 Python 中有很多框架可以与 Neo4j 协同工作。但是，有很多应用程序使用 Java 框架。这些框架提供了一种连接到各种数据源的方式，利用各种可用的包来构建复杂的应用程序。

这些框架支持各种向量存储，如 Neo4j，以及多个 LLM 提供商，如 Amazon Bedrock、Azure OpenAI、Google Gemini、Hugging Face 和 OpenAI。它们提供高级 AI 功能，从简单的任务，如为 LLM 格式化输入和解析输出，到更复杂的功能，如聊天记忆、工具和 RAG。

通过将这些能力与 Neo4j 结合，这些框架使构建更复杂的应用程序变得更容易，例如使用 LLM 生成图特征（路径等）的嵌入，这可以作为使用相似性和社区检测算法将节点分组到段的基础，以增强图谱。这种分段可以提供下一级推荐和其他方面的基础。您可以在[`neo4j.com/labs/genai-ecosystem/`](https://neo4j.com/labs/genai-ecosystem/)了解更多关于 Neo4j 的 GenAI 生态系统信息。

# Neo4j GenAI 生态系统中的智能推荐系统概述

让我们看看基于 LLM/RAG 原则构建的推荐系统在 Neo4j GenAI 生态系统中的运作方式（*图 7.1*）。

![图 7.1 — Neo4j RAG 推荐架构](img/B31107_07_1.png)

图 7.1 — Neo4j RAG 推荐架构

我们可以利用这些框架的特性来构建基于知识图谱的 RAG 应用程序。在这个架构中，我们利用 Spring AI 应用程序来增强图谱，以便能够提供更多个性化的推荐。

此外，对于 RAG，这个架构可以利用向量索引以及图遍历来增强响应，以获得两者的最佳效果，从而获得更准确的响应。这个概念被称为**G****raph RAG**。知识图谱可以为 AI 模型交互带来更准确的响应、丰富的上下文和可解释性。Neo4j 可以集成到 LangChain4j 和 Spring AI 中，作为向量存储以及图数据库，以增强 LLM 的响应。

# 摘要

在本章中，我们探讨了 Neo4j 帮助我们构建智能应用程序的能力，为什么这些应用程序可以提供的个性化是有用的，以及它们与现有的基于规则的应用程序有何不同。我们探讨了 Spring AI 和 LangChain4j 是什么，以及它们构建智能应用程序的能力。

在下一章，*第八章*，我们将使用 H&M 数据集构建一个图数据模型，以支持智能和个性化的推荐，并了解如何将此类数据加载到图数据模型中，目的是提供推荐。本书的*第九章*将使您能够将这个智能推荐系统集成到 Spring AI 和 LangChain4j 框架中。
