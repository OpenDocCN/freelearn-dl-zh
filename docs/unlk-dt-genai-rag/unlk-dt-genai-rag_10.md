

# 第十章：LangChain 中的关键 RAG 组件

本章深入探讨了与**LangChain** 和**检索增强生成** (**RAG**)相关的关键技术组件。作为复习，我们 RAG 系统的关键技术组件，按照使用顺序排列，包括**向量存储**、**检索器**和**大型语言模型** (**LLMs**)。我们将逐步介绍我们代码的最新版本，这是在***第八章**中最后看到的，*代码实验室 8.3*。我们将关注每个核心组件，并使用 LangChain 在代码中展示每个组件的各种选项。*自然地，这次讨论将突出每个选项之间的差异，并讨论在不同场景下某个选项可能比另一个选项更好的情况。

我们从一份代码实验室开始，概述了您的 向量存储选项。

# 技术要求

本章的代码放置在以下 GitHub 仓库中： [https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_10](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_10)

# 代码实验室 10.1 – LangChain 向量存储

所有这些代码 实验室的目标是帮助您更熟悉 LangChain 平台内提供的每个关键组件选项如何增强您的 RAG 系统。 我们将深入研究每个组件的功能、可用函数、影响参数，以及最终，您可以利用的所有更好的 RAG 实现选项。 从**代码实验室 8.3**开始，（跳过***第九章**的评估代码），我们将按照代码中出现的顺序逐步介绍这些元素，从向量存储开始。 您可以在 GitHub 上找到完整的代码，在***第十章**的代码文件夹中，也标记为**10.1`。

## 向量存储、LangChain 和 RAG

**向量存储** 在 RAG 系统中起着至关重要的作用，通过高效地存储和 索引知识库文档的向量表示。 LangChain 提供了与各种向量存储 实现的无缝集成，例如 **Chroma**, **Weaviate**, **FAISS** (**Facebook AI Similarity Search**), **pgvector**, 和 **Pinecone**。对于这个 代码实验室，我们将展示如何将数据添加到 Chroma、Weaviate 和 FAISS 中，为你能够集成 LangChain 提供的众多向量存储中的任何一个打下基础。 这些向量存储提供了高性能的相似度搜索功能，能够根据 查询向量快速检索相关文档。

LangChain 的向量存储类作为与不同向量存储后端交互的统一接口。 它提供了向向量存储添加文档、执行相似度搜索和检索存储文档的方法。 这种抽象允许开发者轻松地在不同的向量存储实现之间切换，而无需修改核心 检索逻辑。

当使用 LangChain 构建 RAG 系统时，你可以利用向量存储类来高效地存储和检索文档向量。 向量存储的选择取决于可扩展性、搜索性能和部署需求等因素。 例如，Pinecone 提供了一种具有高可扩展性和实时搜索能力的完全托管向量数据库，使其适用于生产级 RAG 系统。 另一方面，FAISS 提供了一个用于高效相似度搜索的开源库，可用于本地开发和实验。 Chroma 因其易用性和与 LangChain 的有效集成而成为开发者构建第一个 RAG 管道时的热门选择。

如果你查看我们在前几章中讨论的代码，我们已经在使用 Chroma。 以下是该代码片段，展示了我们使用 Chroma 的方式，你可以在本代码实验室的代码中找到它： ：

```py
 chroma_client = chromadb.Client()
collection_name = "google_environmental_report"
vectorstore = Chroma.from_documents(
               documents=dense_documents,
               embedding=embedding_function,
               collection_name=collection_name,
               client=chroma_client
)
```

LangChain 称这为 **集成** ，因为它与第三方 Chroma 进行了集成。 LangChain 还有许多其他可用的集成。

在 LangChain**网站**上，目前有一个**集成** 链接位于网站页面顶部的**主网站导航**中。 如果您点击它，您将看到一个沿着左侧伸展得很远的菜单，其中包括**提供者** 和**组件**的主要类别。 正如您可能已经猜到的，这使您能够通过提供者或组件来查看所有集成。 如果您点击**提供者**，您将首先看到**合作伙伴包** 和**特色社区提供者**。 Chroma 目前不在这两个列表中，但如果您想了解更多关于 Chroma 作为提供者的信息，请点击页面末尾的链接，该链接说**点击此处查看所有提供者**。列表按字母顺序排列。 向下滚动到 C 处找到 Chroma。 这将显示与 Chroma 相关的有用 LangChain 文档，尤其是在创建向量存储和**检索器**。

另一种有用的方法是点击**向量存储** 下的 **组件**。目前有 49 个向量存储选项！ 当前链接为版本 0.2.0，但也要关注未来的版本 ：

[https://python.langchain.com/v0.2/docs/integrations/vectorstores/](https://python.langchain.com/v0.2/docs/integrations/vectorstores/ )

我们强烈推荐您查看的另一个领域是 LangChain 向量存储**文档**： 

[https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.vectorstores](https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.vectorstores )

在过去的章节中，我们已经深入讨论了我们的当前向量存储和 Chroma，但让我们回顾一下 Chroma 并讨论它在哪里**最有用**。 

### Chroma

**Chroma** 是一个开源的 AI 原生向量数据库，旨在提高开发者的生产力和**易用性**。 它提供快速的搜索性能，并通过其 Python SDK 与 LangChain 无缝集成。 Chroma 支持多种部署模式，包括内存中、持久存储和 Docker 容器化部署 。

Chroma 的一个关键优势是其简单性和开发者友好的 API。 它提供了添加、更新、删除和查询向量存储中文档的直接方法。 Chroma 还支持基于元数据的动态过滤集合，从而可以进行更有针对性的搜索。 此外，Chroma 还提供了内置的文档分块和索引功能，使得处理大型文本数据集变得方便。 文本数据集。

Chroma 的架构由一个用于快速向量检索的索引层、一个用于高效数据管理的存储层和一个用于实时操作的处理层组成。 Chroma 与 LangChain 无缝集成，使开发者能够在 LangChain 生态系统中利用其功能。 Chroma 客户端可以轻松实例化并传递给 LangChain，从而实现高效的文档向量存储和检索。 Chroma 还支持高级检索选项，例如**最大边际相关性**（**MMR**）和元数据过滤，以细化搜索结果。

总的来说，Chroma 是 一个不错的选择 ，对于寻求一个开源、易于使用且与 LangChain 良好集成的向量数据库的开发者来说。 其简单性、快速搜索性能和内置的文档处理功能使其成为构建 RAG 应用的吸引人选择。 事实上，这也是我们为什么在本书的几个章节中突出介绍 Chroma 的原因之一。 然而，评估您的具体需求并将 Chroma 与其他向量存储替代品进行比较，以确定最适合您项目的方案是很重要的。 让我们查看代码并讨论一些其他可用的选项，从 FAISS 开始。 考虑将 Chroma 作为 RAG 应用的向量存储时，评估其架构和选择标准很重要。

### FAISS

让我们从如何修改我们的代码 开始，如果我们想使用 FAISS 作为我们的向量存储。 您首先需要安装 FAISS：

```py
 %pip install faiss-cpu
```

在您重新启动内核（因为您安装了新包）后，运行所有代码直到向量存储相关的单元格，并将与 Chroma 相关的代码替换为 FAISS 向量存储实例化：

```py
 from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
               documents=dense_documents,
               embedding=embedding_function
)
```

`Chroma.from_documents()` 方法调用已被替换为 `FAISS.from_documents()`。 `collection_name` 和 `client` 参数对 FAISS 不适用，因此已从方法调用中删除。 `我们重复了一些与 Chroma 向量存储相关的代码，例如文档生成，这使我们能够展示两种向量存储选项之间的代码精确等效。` 通过这些更改，代码现在可以使用 FAISS 作为向量存储而不是 Chroma。

**FAISS** 是由 Facebook AI 开发的开源库。 FAISS 提供高性能的搜索功能，可以处理可能无法完全适应内存的大型数据集。 与其他在此处提到的向量存储类似，FAISS 的架构包括一个索引层，用于组织向量以实现快速检索，一个存储层，用于高效的数据管理，以及一个可选的处理层，用于实时操作。 FAISS 提供了各种索引技术，如聚类和量化，以优化搜索性能和内存使用。 它还支持 GPU 加速，以实现更快的相似性搜索。

如果您有可用的 GPU，您 可以安装此包而不是我们之前安装的包：

```py
 %pip install faiss-gpu
```

使用 FAISS 的 GPU 版本可以显著加快相似性搜索过程，特别是对于大规模数据集。 GPU 可以并行处理大量向量比较，从而在 RAG 应用中更快地检索相关文档。 如果您在一个处理大量数据且需要比我们之前使用（Chroma）更高的性能的环境工作，您绝对应该测试 FAISS GPU 并看看它对您的影响。 。

FAISS LangChain 文档提供了如何在 LangChain 框架中使用 FAISS 的详细示例和指南。 它涵盖了诸如摄取文档、查询向量存储、保存和加载索引以及执行高级操作（如过滤和合并）等主题。 文档还突出了 FAISS 特有的功能，例如带有分数的相似性搜索和索引的序列化/反序列化。 。

总的来说，FAISS 是 一个强大且高效的向量存储选项，用于构建与 LangChain 结合的 RAG 应用。 它的高性能搜索能力、可扩展性和与 LangChain 的无缝集成使其成为寻求强大且可定制解决方案的开发者的一个有吸引力的选择，用于存储和检索 文档 向量。

这些是满足你的向量存储需求的有力选择。 接下来，我们将展示并讨论 Weaviate 向量 存储选项。

### Weaviate

关于如何使用和访问 Weaviate，有多种 选择。 我们将展示 嵌入版本，它从你的应用程序代码而不是从独立的 Weaviate 服务器安装中运行 Weaviate 实例。

当嵌入式 Weaviate 首次启动时，它会在 `persistence_data_path`设置的路径中创建一个永久数据存储。 当你的客户端退出时，嵌入式 Weaviate 实例也会退出，但数据会持续保留。 下次客户端运行时，它将启动一个新的嵌入式 Weaviate 实例。 新的嵌入式 Weaviate 实例使用存储在 数据存储中的数据。

如果你熟悉 **GraphQL**，当你开始查看代码时，你可能认识到它对 Weaviate 产生的影响。 查询语言和 API 受到了 GraphQL 的启发，但 Weaviate 并不直接使用 GraphQL。 Weaviate 使用 RESTful API，其查询语言在结构和功能上与 GraphQL 相似。 Weaviate 在模式定义中使用预定义的数据类型来表示属性，类似于 GraphQL 的标量类型。 Weaviate 中可用的数据类型包括字符串、整数、数字、布尔值、日期等。

Weaviate 的一个优势是它支持批量操作，可以在单个请求中创建、更新或删除多个数据对象。 这与 GraphQL 的突变操作类似，你可以在单个请求中执行多个更改。 Weaviate 使用 `client.batch` 上下文管理器将多个操作组合成一个批次，我们将在 稍后演示。

让我们从如何使用 Weaviate 作为我们的向量存储开始。 你首先需要 安装 FAISS：

```py
 %pip install weaviate-client
%pip install langchain-weaviate
```

在你重启内核（因为你安装了新的包）之后，你运行所有与向量存储相关的代码，并更新代码以使用 FAISS 向量 存储实例化：

```py
 import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.embedded import EmbeddedOptions
from langchain.vectorstores import Weaviate
from tqdm import tqdm
```

正如你所见，为了使用 Weaviate，你需要导入许多额外的包。 我们还安装了 `tqdm`，这不是 Weaviate 特有的，但它是必需的，因为 Weaviate 使用 `tqdm` 来显示加载时的进度条。

我们必须首先声明 `weaviate_client` 作为 Weaviate 客户端：

```py
 weaviate_client = weaviate.Client(
    embedded_options=EmbeddedOptions())
```

我们原始的 Chroma 向量存储代码和使用 Weaviate 之间的差异比我们迄今为止采取的其他方法更复杂。 使用 Weaviate，我们使用 `WeaviateClient` 客户端和嵌入选项初始化，以启用嵌入式模式，正如你 之前所看到的。

在我们继续之前，我们需要确保已经没有 Weaviate 客户端的实例存在，否则我们的代码 将会失败：

```py
 try:
     weaviate_client.schema.delete_class(collection_name)
except:
        pass
```

对于 Weaviate，你必须 确保清除过去迭代中遗留的任何模式，因为它们可能会在 后台持续存在。

然后我们使用 `weaviate` 客户端，通过类似于 GraphQL 的 定义模式来建立我们的数据库：

```py
 weaviate_client.schema.create_class({
               "class": collection_name,
               "description": "Google Environmental
                               Report",
               "properties": [
                             {
                                  "name": "text",
                                  "dataType": ["text"],
                                  "description": "Text
                                   content of the document"
                             },
                             {
                                  "name": "doc_id",
                                  "dataType": ["string"],
                                  "description": "Document
                                       ID"
                             },
                             {
                                  "name": "source",
                                  "dataType": ["string"],
                                  "description": "Document
                                       source"
                             }
               ]
})
```

这提供了一个完整的模式类，你稍后将其作为 `weviate_client` 对象的一部分传递给向量存储定义。 你需要使用 `client.collections.create()` 方法为你的集合定义此模式。 模式定义包括指定类名、属性及其数据类型。 属性可以有不同的数据类型，例如字符串、整数和布尔值。 正如你所见，与我们在之前的实验室中使用 Chroma 所用的相比，Weaviate 执行了更严格的模式验证。

虽然这种类似于 GraphQL 的模式在建立你的向量存储时增加了一些复杂性，但它也以有用且强大的方式为你提供了更多对数据库的控制。 特别是，你对自己的模式定义有了更细粒度的控制。

您可能认出下面的代码，因为它看起来很像我们之前定义的 `dense_documents` 和 `sparse_documents` 变量，但如果你仔细看，有一个重要的 差异对 Weaviate 来说很重要：

```py
 dense_documents = [Document(page_content=text,
metadata={"doc_id": str(i), "source": "dense"}) for i,
          text in enumerate(splits)]
sparse_documents = [Document(page_content=text, metadata={"doc_id": str(i), "source": "sparse"}) for i,
          text in enumerate(splits)]
```

当我们使用元数据预处理文档时，这些定义对 Weaviate 会有轻微的变化。 我们使用 `'doc_id'` 而不是 `'id'` 作为 Weaviate。 这是因为 `'id'` 在内部使用，并且不可用于我们。 在代码的后续部分，当您从元数据结果中提取 ID 时，您将需要更新该代码以使用 `'doc_id'` 。

接下来，我们定义我们的向量存储，类似于我们过去在 Chroma 和 FAISS 中做过的事情，但使用 Weaviate 特定的参数：

```py
 vectorstore = Weaviate(
               client=weaviate_client,
               embedding=embedding_function,
               index_name=collection_name,
               text_key="text",
               attributes=["doc_id", "source"],
               by_text=False
)
```

对于向量存储初始化，Chroma 使用 `from_documents` 方法直接从文档创建向量存储，而对于 Weaviate，我们创建向量存储然后添加文档。 Weaviate 还需要额外的配置，例如 `text_key`, `attributes`, 和 `by_text`。一个主要区别是 Weaviate 使用 一个模式。

最后，我们将我们的实际内容加载到Weaviate 向量存储实例中，这也适用于 过程中的嵌入函数：

```py
 weaviate_client.batch.configure(batch_size=100)
with weaviate_client.batch as batch:
    for doc in tqdm(dense_documents, desc="Processing
        documents"):
                    properties = {
                                  "text": doc.page_content,
                                  "doc_id":doc.metadata[
                                               "doc_id"],
                                  "source": doc.metadata[
                                                "source"]
                             }
                    vector=embedding_function.embed_query(
                               doc.page_content)
                           batch.add_data_object(
                               data_object=properties,
                               class_name=collection_name,
                               vector=vector
                           )
```

总的来说，Chroma 提供了一种更简单、更灵活的数据模式定义方法，并专注于嵌入存储和检索。 它可以轻松地嵌入到您的应用程序中。 另一方面，Weaviate 提供了一个结构更清晰、功能更丰富的向量数据库解决方案，具有显式的模式定义、多个存储后端和内置对各种嵌入模型的支持。 它可以作为独立服务器部署或托管在云中。 Chroma、Weaviate 或其他向量存储之间的选择取决于您的具体需求，例如模式灵活性的水平、部署偏好以及除嵌入存储之外需要额外功能的需求： 嵌入 存储。

注意，您可以使用这些向量存储中的任何一个，并且剩余的代码将适用于加载到它们中的数据。这是使用 LangChain 的一个优势，它允许您在组件之间进行交换。这在生成式 AI 的世界中尤其必要，因为新的和显著改进的技术不断推出。使用这种方法，如果您遇到一种新的、更好的向量存储技术，它会在您的 RAG 管道中产生差异，您可以相对快速且容易地进行这种更改。接下来，让我们谈谈 LangChain 武器库中的另一个关键组件，它是 RAG 应用程序的核心：**检索器**。

# 代码实验室 10.2 – LangChain 检索器

在这个代码实验室中，我们将介绍检索过程中最重要的组件的一些示例：**LangChain 检索器**。与 LangChain 向量存储一样，这里列出的 LangChain 检索器选项太多。我们将关注一些特别适用于 RAG 应用程序的流行选择，并鼓励您查看所有其他选项，看看是否有更适合您特定情况的选项。就像我们讨论向量存储时一样，LangChain 网站上有很多文档可以帮助您找到最佳解决方案：[`python.langchain.com/v0.2/docs/integrations/retrievers/`](https://python.langchain.com/v0.2/docs/integrations/retrievers/)

检索器包的文档可以在以下位置找到：[`api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.retrievers`](https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.retrievers)

现在，让我们开始为检索器编写代码！

## 检索器、LangChain 和 RAG

**检索器** 负责根据输入查询查询向量存储并检索最相关的文档。LangChain 提供了一系列的检索器实现，这些实现可以与不同的向量存储和查询编码器一起使用。

在我们目前的代码中，我们已经看到了检索器的三个版本；让我们首先回顾它们，因为它们与基于 Chroma 的原始向量存储相关联。

### 基本检索器（密集嵌入）

我们从 密集检索器 **** 开始。这是我们在此点之前在几个代码实验室中使用的代码：

```py
 dense_retriever = vectorstore.as_retriever(
                      search_kwargs={"k": 10})
```

密集检索器是通过使用 `vectorstore.as_retriever` 函数创建的，指定要检索的顶部结果数量（`k=10`）。 在这个检索器的底层，Chroma 使用文档的密集向量表示，并使用余弦距离或欧几里得距离进行相似度搜索，根据查询嵌入检索最相关的文档。

这是使用最简单的检索器类型，即向量存储检索器，它为每段文本创建嵌入，并使用这些嵌入进行检索。 检索器本质上是对向量存储的包装。 使用这种方法，您可以使用 LangChain 生态系统中的集成和接口访问向量存储的内置检索/搜索功能。 它是对向量存储类的轻量级包装，为 LangChain 中的所有检索器选项提供了一致的接口。 因此，一旦构建了向量存储，构建检索器就非常容易。 如果您需要 更改向量存储或检索器，这也同样容易做到 。

这些类型的检索器提供了两种主要的搜索功能，直接源于它所包装的向量存储：相似度搜索和 MMR。

### 相似度分数阈值检索

默认情况下，检索器使用相似度搜索。如果您想设置一个相似度阈值，则只需将搜索类型设置为 `similarity_score_threshold` 并在传递给检索器对象的 `kwargs` 函数中设置该相似度分数阈值。 代码看起来像这样：

```py
 dense_retriever = vectorstore.as_retriever(
               search_type="similarity_score_threshold",
               search_kwargs={"score_threshold": 0.5}
)
```

这是对默认相似度搜索的有用升级，在许多 RAG 应用中可能很有用。 然而，相似度搜索并不是这些检索器支持的唯一搜索类型；还有 MMR。

### MMR

`search_type="mmr"` 作为定义检索器时的一个参数，如下所示：

```py
 dense_retriever = vectorstore.as_retriever(
                      search_type="mmr"
)
```

将此添加到任何基于向量存储的检索器中，将使其利用 MMR 类型的搜索。

相似性搜索和 MMR 可以由支持这些搜索技术的任何向量存储支持。 接下来，让我们谈谈我们在 *第八章*中引入的稀疏搜索机制，即 BM25 检索器。

### BM25 检索器

`BM25Retriever` 是 LangChain 对 BM25 的表示，可用于稀疏文本 检索目的。

您也见过这个检索器，因为我们曾用它将我们的基本搜索转换为混合搜索，在 *第八章*。我们在代码中通过以下设置看到这一点： 这些设置：

```py
 sparse_retriever = BM25Retriever.from_documents(
    sparse_documents, k=10)
```

调用 `BM25Retriever.from_documents()` 方法从稀疏文档创建稀疏检索器，指定要检索的 top 结果数量（`k=10`）。

BM25 通过根据查询词和文档的 **词频和逆文档频率** (**TF-IDF**) 计算每个文档的相关性得分。 它使用一个 概率模型来估计文档与给定查询的相关性。 检索器返回具有最高 BM25 分数的 top-k 文档。

### 集成检索器

一个 **集成检索器** 结合 多种检索方法，并使用一个额外的 算法将它们的结果合并成一个集合。 这种类型检索器的理想用途是在您想结合密集和稀疏检索器以支持混合检索方法时，例如我们在 *第八章*的 *代码 *实验室 8.3 *中创建的：

```py
 ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5], c=0, k=10)
```

在我们的案例中，集成检索器结合了 Chroma 密集检索器和 BM25 稀疏检索器，以实现更好的检索性能。 它是使用 `EnsembleRetriever` 类创建的，该类接受检索器的列表及其相应的权重。 在这种情况下，密集检索器和稀疏检索器以相等的权重 `0.5` 传递。

在集成检索器中， `c` 参数是一个重新排序参数，它控制原始检索分数和重新排序分数之间的平衡。 它用于调整重新排序步骤对最终检索结果的影响。 在这种情况下， `c` 参数设置为 `0`，这意味着不执行重新排序。 当 `c` 设置为非零值时，集成检索器对检索到的文档执行额外的重新排序步骤。 重新排序步骤根据单独的重新排序模型或函数重新评分检索到的文档。 重新排序模型可以考虑到额外的特征或标准来评估文档与 查询的相关性。

在 RAG 应用中，检索到的文档的质量和相关性直接影响生成的输出。 通过利用 `c` 参数和合适的重新排序模型，您可以增强检索结果以更好地满足您 RAG 应用的具体要求。 例如，您可以设计一个重新排序模型，该模型考虑了诸如文档相关性、与查询的一致性或特定领域标准等因素。 通过为 `c`设置适当的值，您可以在原始检索分数和重新排序分数之间取得平衡，在需要时给予重新排序模型更多的权重。 这可以帮助优先考虑对 RAG 任务更相关和更有信息量的文档，从而提高 生成的输出质量。

当查询传递给集成检索器时，它会将查询发送给密集和稀疏检索器。 集成检索器随后根据分配的权重将两个检索器的结果结合起来，并返回前 k 个文档。 在底层，集成检索器利用了密集和稀疏检索方法的优势。 密集检索通过密集向量表示捕获语义相似性，而稀疏检索则依赖于关键词匹配和词频。 通过结合它们的结果，集成检索器旨在提供更准确和全面的 搜索结果。

代码片段中使用的特定类和方法可能因所使用的库或框架而异。 然而，使用向量相似度搜索进行密集检索、使用 BM25 进行稀疏检索以及结合多个检索器的集成检索的一般概念 仍然是相同的。

这涵盖了我们在之前的代码中已经看到的检索器，所有这些都是从我们在索引阶段访问和处理的数据中提取的。 还有许多其他类型的检索器可以与您从文档中提取的数据一起使用，您可以在 LangChain 网站上探索这些检索器以满足您的需求。 然而，并非所有检索器都旨在从您正在处理的文档中提取数据。 接下来，我们将回顾一个基于公开数据源（维基百科）构建的检索器的示例。

### 维基百科检索器

正如维基百科检索器 的制作者在 LangChain 网站上所描述的（[https://www.langchain.com/](https://www.langchain.com/)）：

维基百科是历史上最大、最受欢迎的参考工具，它是一个由志愿者社区编写和维护的多语言免费在线百科全书。

这听起来像是一个很好的资源，可以用来在你的 RAG 应用中获取有用的知识！ 我们将在现有的检索器单元之后添加一个新的单元，我们将使用这个维基百科检索器从 `wikipedia.org` 检索维基页面到 `文档` 格式，这是 下游使用的。

我们首先需要安装几个 新的包：

```py
 %pip install langchain_core
%pip install --upgrade --quiet        wikipedia
```

一如既往，当你安装新的包时，别忘了重启 你的内核！

有了 WikipediaRetriever 检索器，我们现在有一个机制 可以从维基百科获取与我们所传递的用户查询相关的数据，类似于我们之前使用的其他检索器，但使用的是维基百科的全部数据 （：

```py
 from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever(load_max_docs=10)
docs = retriever.get_relevant_documents(query=
    "What defines the golden age of piracy in the
     Caribbean?")
metadata_title = docs[0].metadata['title']
metadata_summary = docs[0].metadata['summary']
metadata_source = docs[0].metadata['source']
page_content = docs[0].page_content
print(f"First document returned:\n")
print(f"Title: {metadata_title}\n")
print(f"Summary: {metadata_summary}\n")
print(f"Source: {metadata_source}\n")
print(f"Page content:\n\n{page_content}\n")
```

在此代码中，我们从 `langchain_community.retrievers` 模块中导入 `WikipediaRetriever` 类。 `WikipediaRetriever` 是一个专门设计用于根据给定查询从 Wikipedia 检索相关文档的检索器类。 然后我们使用 `WikipediaRetriever` 类实例化一个接收器实例，并将其分配给变量 retriever。 `load_max_docs` 参数设置为 `10`，表示检索器应加载最多 10 个相关文档。 此处的用户查询是 `什么是加勒比海盗黄金时代的定义？`，我们可以查看响应以了解 Wikipedia 文章是如何被检索出来以帮助回答 此问题。

我们调用检索器对象的 `get_relevant_documents` 方法，传入一个查询字符串作为参数，并在响应中接收第一个文档：

```py
 First document returned:
Title: Golden Age of Piracy
Summary: The Golden Age of Piracy is a common designation for the period between the 1650s and the 1730s, when maritime piracy was a significant factor in the histories of the North Atlantic and Indian Oceans. Histories of piracy often subdivide the Golden Age of Piracy into three periods:
The buccaneering period (approximately 1650 to 1680)…
```

您可以在以下链接中查看匹配的内容： 此链接：

[海盗黄金时代](https://en.wikipedia.org/wiki/Golden_Age_of_Piracy)

此链接是由 检索器提供的来源。

总的来说，此代码演示了如何使用 `WikipediaRetriever` 类从 `langchain_community.retrievers` 模块中检索基于给定查询的 Wikipedia 相关文档。 然后它提取并打印特定的元数据信息（标题、摘要、来源）以及检索到的第一份文档的内容。

`WikipediaRetriever` 内部处理查询维基百科 API 或搜索功能的过程，检索相关文档，并将它们作为 `Document` 对象列表返回。 每个 `Document` 对象都包含元数据和实际页面内容，可以根据需要访问和使用。 还有许多其他检索器可以访问类似此类但专注于特定领域的公共数据源。 对于科学研究，有 `PubMedRetriever`。对于其他研究领域，如数学和计算机科学，有 `ArxivRetreiver`，它可以从关于这些主题的超过 200 万篇开放获取档案的数据中获取数据。 在 金融领域，有一个名为 `KayAiRetriever` 的检索器，可以访问 **证券交易委员会** (**SEC**) 的文件，这些文件包含上市公司必须提交给 美国证券交易委员会 的财务报表。

对于处理非大规模数据的项目的检索器，我们还有一个要强调： kNN 检索器。

### kNN 检索器

我们 一直 在使用的 最近邻算法，负责找到与用户查询最相关的内容的算法，一直 基于 **近似最近邻** (**ANN**)。 尽管有一个更*传统** 和*古老** 的算法可以作为 ANN 的替代方案，那就是 k-最近邻 **（kNN）**。 但 kNN 基于一个可以追溯到 1951 年的算法；为什么我们有像 ANN 这样更复杂、更强大的算法可用时还要使用这个呢？ 因为 kNN *仍然 比之后的任何东西都要好。 这不是一个错误。 kNN 仍然是 *最有效** 的方法来找到最近邻。 它比 ANN 更好，ANN 被数据库、向量数据库和信息检索公司吹捧为 *解决方案** 。 ANN 可以接近，但 kNN 仍然被认为是更好的。

为什么人工神经网络（ANN）被誉为 *解决方案* 呢？ 因为 kNN 无法扩展到这些供应商针对的大型企业所看到的水平。 但这都是相对的。 你可能有一百万个数据点，这听起来很多，但与 1,536 维向量相比，在全球企业舞台上仍然相当小。 kNN 可以轻松处理这一点！ 许多在领域内使用 ANN 的小型项目可能从使用 kNN 中受益。 kNN 的理论极限将取决于许多因素，例如你的开发环境、你的数据、数据的维度、如果使用 API，则还取决于互联网连接，等等。 因此，我们无法给出具体的数据点数量。 你需要进行测试。 但如果它小于我刚才描述的项目，即 1 百万个数据点，1,536 维向量，在一个相对强大的开发环境中，你真的应该考虑 kNN！ 在某个时候，你会注意到处理时间的显著增加，当等待时间变得过长以至于你的应用程序的实用性降低时，切换到 ANN。 但在此期间，务必充分利用 kNN 的优越搜索能力 。

幸运的是，kNN 可以通过一个易于设置的检索器**KNNRetriever**获得。这个检索器将利用我们与其他算法一起使用的相同密集嵌入，因此我们将用基于 kNN 的**KNNRetriever**替换**dense_retriever**。以下是实现此功能的代码，在定义了之前版本的我们的**dense_retriever**检索器对象之后很好地融入其中：

```py
 from langchain_community.retrievers import KNNRetriever
dense_retriever = KNNRetriever.from_texts(splits,
    OpenAIEmbeddings(), k=10)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5], c=0, k=10)
```

运行代码实验室中的剩余代码，看看它如何取代我们之前的**dense_retriever**并在此处执行。 在这种情况下，由于数据集非常有限，很难评估它是否比我们之前使用的基于 ANN 的算法表现更好。 但是，随着你的项目规模扩大，我们强烈建议你利用这种方法，直到其扩展问题变得过于沉重。

这标志着我们对支持 RAG 的检索器的探索结束。 还有其他类型的检索器，以及与支持这些检索器的向量存储的显著集成，可以在 LangChain 网站上查看。 例如，有一个时间加权向量存储检索器，允许你在检索过程中结合近期性。 还有一个名为 Long-Context Reorder 的检索器，专注于改进难以关注检索文档中间信息的长上下文模型的结果。 务必查看可用的内容，因为它们可能对你的 RAG 应用产生重大影响。 我们现在将转向讨论操作和生成阶段的**大脑**：LLMs。 LLMs。

# 代码实验室 10.3 – LangChain LLMs

我们现在将注意力转向 RAG 的最后一个关键组件：LLM。 就像检索阶段中的检索器一样，如果没有生成阶段的 LLM，就没有 RAG。 检索阶段只是从我们的数据源检索数据，通常是 LLM 不知道的数据。 但这并不意味着 LLM 在我们的 RAG 实现中没有发挥至关重要的作用。 通过向 LLM 提供检索到的数据，我们迅速让 LLM 了解我们希望它讨论的内容，这使得 LLM 能够发挥其真正擅长的能力，根据这些数据提供基于数据的响应来回答用户提出的原始问题。

LLMs 和 RAG 系统之间的协同作用源于这两种技术的互补优势。RAG 系统通过整合外部知识源来增强 LLMs 的能力，从而生成不仅与上下文相关，而且事实准确且最新的响应。 反过来，LLMs 通过提供对查询上下文的复杂理解，促进从知识库中更有效地检索相关信息。 这种共生关系显著提高了 AI 系统在需要深度语言理解和广泛事实信息访问的任务中的性能，利用每个组件的优势，创建一个更强大和**多才多艺**的系统。

在这个代码实验室中，我们将介绍生成阶段最重要的组件之一：**LangChain LLM**。

## **LLMs**、LangChain 和 RAG

与之前的关键组件一样，我们首先提供与这个主要组件相关的 LangChain 文档链接，即**LLMs**：[`python.langchain.com/v0.2/docs/integrations/llms/`](https://python.langchain.com/v0.2/docs/integrations/llms/)

这里是第二个有用的信息来源，它结合了 LLMs 和 LangChain 的 API 文档：[`api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.llms`](https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.llms)

让我们从我们一直在使用的 API 开始：**OpenAI**。

### **OpenAI**

我们已经有了这段代码，但让我们通过回顾实验室中使这个组件工作的关键区域来刷新这段代码的内部工作原理：

1.  首先，我们必须安装**langchain-openai**包：

    ```py
    langchain-openai library provides integration between OpenAI’s language models and LangChain.
    ```

1.  接下来，我们导入**openai**库，这是与 OpenAI API 交互的官方 Python 库，在本代码中主要用于将 API 密钥应用于模型，以便我们可以访问付费 API。然后，我们导入 `ChatOpenAI` 和 `OpenAIEmbeddings` 类，它们来自 `langchain_openai` 库：

    ```py
    import openai
    
    ChatOpenAI is used to interact with OpenAI’s chat models, and OpenAIEmbeddings is used for generating embeddings from text.
    ```

1.  在下一行，我们使用**load_dotenv**函数从名为**env.txt**的文件中加载环境变量：

    ```py
    env.txt file to store sensitive information (an API key) in a way that we can hide it from our versioning system, practicing better and more secure secret management.
    ```

1.  然后 我们将该 API 密钥通过以下代码传递给 OpenAI：

    ```py
    os.environ['OPENAI_API_KEY'] = os.getenv(
    
     'OPENAI_API_KEY')
    
    OPENAI_API_KEY. Then, we set the OpenAI API key for the openai library using the retrieved value from the environment variable. At this point, we can use the OpenAI integration with LangChain to call the LLM that is hosted at OpenAI with the proper access.
    ```

1.  在代码的后续部分，我们定义了我们想要 使用的 LLM：

    ```py
    llm = ChatOpenAI(model_name="gpt-4o-mini",
    
     temperature=0)
    ```

这一行创建了一个 `ChatOpenAI` 类的实例，指定模型名称为 `gpt-4o-mini` ，并将 `温度` 变量设置为 `0`。温度控制生成响应的随机性，较低的值会产生更集中和确定性的输出。 目前， `gpt-4o-mini` 是最新且能力最强的模型，同时也是 GPT4 系列中最具成本效益的模型。 但即使是这个模型，其成本也比 `gpt-3.5-turbo`高 10 倍，而实际上 `gpt-3.5-turbo`是一个相对 有能力的模型。

OpenAI 最昂贵的模型 `gpt-4-32k`，其速度和能力并不如 `gpt-4o-mini` ，且其上下文窗口大小是其 4 倍。 未来可能会出现更新的模型，包括 `gpt-5`，这些模型可能成本更低且能力更强。 从所有这些中，你可以得出的结论是，你不应该仅仅假设最新的模型将是成本最高的，而且总会有更强大且成本效益更高的替代版本不断推出。 持续关注模型的最新发布，并对每个版本，权衡 成本、LLM 能力以及其他相关 属性，以决定是否需要进行更改 。

但在这一努力中，你不需要将自己限制在仅使用 OpenAI。 使用 LangChain 可以轻松切换 LLM，并扩大你在 LangChain 社区内寻找最佳解决方案的搜索范围。 让我们逐一探讨一些你可能考虑的其他选项。

### Together AI

**Together AI** 提供了一套面向开发者的服务，让你可以访问众多模型。 他们 `托管 LLM 的定价难以匹敌，并且经常提供 5.00 美元的免费信用额度来测试 `不同的模型。`

如果您是 Together API 的新用户，可以使用此链接设置您的 API 密钥并将其添加到您的 `env.txt` 文件中，就像我们过去使用 OpenAI API 密钥一样： [https://api.together.ai/settings/api-keys](https://api.together.ai/settings/api-keys )

当您到达这个网页时，它目前提供 5.00 美元的信用额度，点击 **开始使用** 按钮后即可使用。 您无需提供信用卡即可访问这5.00 美元的信用额度。

请确保将您的新 API 密钥添加到您的 `env.txt` 文件中 作为 `TOGETHER_API_KEY`。

一旦您登录，您就可以在这里看到每个 LLM 的当前成本： 这里： [https://api.together.ai/models](https://api.together.ai/models )

例如，Meta Llama 3 70B instruct (Llama-3-70b-chat-hf) 目前列出的成本为每 100 万个令牌 0.90 美元。 这是一个已被证明可以与 ChatGPT 4 相媲美的模型，但 Together AI 将 以比 OpenAI 收费显著低得多的推理成本运行。 另一个高度强大的模型，Mixtral 专家混合模型，每 100 万个令牌成本为 1.20 美元。按照以下步骤设置和使用 Together AI：

1.  我们首先安装使用 Together API 所需的包：

    ```py
    %pip install --upgrade langchain-together
    ```

1.  这使我们能够使用 Together API 和 LangChain 之间的集成：

    ```py
    from langchain_together import ChatTogether
    
    ChatTogether integration and loads the API key (don’t forget to add it to the env.txt file before running this line of code!).
    ```

1.  就像我们过去使用 OpenAI API 密钥一样，我们将导入 `TOGETHER_API_KEY` 以便它可以访问 您的账户：

    ```py
    os.environ['TOGETHER_API_KEY'] = os.getenv(
    
     'TOGETHER_API_KEY')
    ```

    我们将使用 Llama 3 Chat 模型和 Mistral 的 Mixtral 8X22B Instruct 模型，但您可以从 50 多个模型中选择 这里： [https://docs.together.ai/docs/inference-models](https://docs.together.ai/docs/inference-models)

    您可能会找到更适合您 特定需求的模型！

1.  在这里，我们正在定义 模型：

    ```py
    llama3llm = ChatTogether(
    
     together_api_key=os.environ['TOGETHER_API_KEY'],
    
    **model="meta-llama/Llama-3-70b-chat-hf",**
    
    **)**
    
    **mistralexpertsllm = ChatTogether(**
    
     **together_api_key=os.environ['TOGETHER_API_KEY'],**
    
     **model="mistralai/Mixtral-8x22B-Instruct-v0.1",**
    
    **)**
    the results.
    ```

***   在这里，我们 更新了使用 Llama 3 模型的最终代码：

    ```py
    llama3_rag_chain_from_docs = (
    
     RunnablePassthrough.assign(context=(lambda x:
    
     format_docs(x["context"])))
    
     | RunnableParallel(
    
     {"relevance_score": (
    
     RunnablePassthrough()
    
     | (lambda x: relevance_prompt_template.
    
     format(
    
     question=x['question'],
    
     retrieved_context=x['context']))
    
     | llama3llm
    
     | StrOutputParser()
    
     ), "answer": (
    
     RunnablePassthrough()
    
     | prompt
    
     | llama3llm
    
     | StrOutputParser()
    
     )}
    
     )
    
     | RunnablePassthrough().assign(
    
     final_answer=conditional_answer)
    
     )
    ```

    这应该看起来很熟悉，因为它是我们过去使用的 RAG 链，但现在运行的是 Llama 3 LLM。 3 LLM。

    ```py
    llama3_rag_chain_with_source = RunnableParallel(
    
     {"context": ensemble_retriever,
    
     "question": RunnablePassthrough()}
    
    ).assign(answer=llama3_rag_chain_from_docs)
    ```

    这是我们使用的最终 RAG 链，已更新为之前的以 Llama 3 为重点的 RAG 链。

    +   接下来，我们希望 运行与过去运行类似的代码，该代码调用并运行 RAG 管道，用 Llama 3 LLM 替换 ChatGPT-4o-mini 模型：

    ```py
    llama3_result = llama3_rag_chain_with_source.invoke(
    
    **user_query)**
    
    **llama3_retrieved_docs = llama3_result['context']**
    
    **print(f"Original Question: {user_query}\n")**
    
    **print(f"Relevance Score:**
    
     **{llama3_result['answer']['relevance_score']}\n")**
    
    **print(f"Final Answer:**
    
     **\n{llama3_result['answer']['final_answer']}\n\n")**
    
    **print("Retrieved Documents:")**
    
    **for i, doc in enumerate(llama3_retrieved_docs,**
    
     **start=1):**
    
    **print(f"Document {i}: Document ID:**
    
     **{doc.metadata['id']} source:**
    
     **{doc.metadata['source']}")**
    
    `What are Google's environmental initiatives?` is as follows:

    ```

    谷歌的环境倡议包括：

    ```py

    ```

    1\. 赋能个人采取行动：在谷歌产品中提供可持续性功能，例如谷歌地图中的环保路线，谷歌 Nest 恒温器中的节能功能，以及谷歌航班中的碳排放信息…

    ```py

    ```

    [TRUNCATED]

    ```py

    ```

    10\. 与外部目标和倡议互动：参与行业范围内的倡议和伙伴关系，以促进可持续性，例如 RE-Source 平台、iMasons 气候协定和世界可持续发展商业理事会。

    ```py 
    ```

    ***   让我们看看如果我们使用专家混合模型会是什么样子： 专家模型：

    ```py
    mistralexperts_rag_chain_from_docs = (
    
    **RunnablePassthrough.assign(context=(lambda x:**
    
     **format_docs(x["context"])))**
    
     **| RunnableParallel(**
    
     **{"relevance_score": (RunnablePassthrough()**
    
     **| (lambda x: relevance_prompt_template.format(**
    
     **question=x['question'],**
    
     **retrieved_context=x['context']))**
    
     **| mistralexpertsllm**
    
     **| StrOutputParser()**
    
     **), "answer": (**
    
    ****RunnablePassthrough()****
    
     ****| prompt****
    
     ****| mistralexpertsllm****
    
     ****| StrOutputParser()****
    
     ****)}****
    
     ****)****
    
     ****| RunnablePassthrough().assign(****
    
     ****final_answer=conditional_answer)****
    
    ****)****
    ```

    ****再次，这应该看起来很熟悉，因为我们过去使用的是 RAG 链，但 这次运行的是专家 LLM 混合模型。****

    ```py
    ****mistralexperts_rag_chain_with_source = RunnableParallel(****
    
     ****{"context": ensemble_retriever, "question": RunnablePassthrough()}****
    
    ****).assign(answer=mistralexperts_rag_chain_from_docs)****
    ```

    ****就像我们之前做的那样： 我们更新了最终的 RAG 管道，使用之前的 专家混合模型重点的 RAG 链。****

    ****此代码将让我们看到用专家混合模型替换 ChatGPT-4o-mini 模型的结果： ChatGPT-4o-mini 模型：****

    ```py
    ****mistralexperts_result = mistralexperts_rag_chain_with_source.invoke(user_query)****
    
    ****mistralexperts_retrieved_docs = mistralexperts_result[****
    
     ****'context']****
    
    ****print(f"Original Question: {user_query}\n")****
    
    ****print(f"Relevance Score: {mistralexperts_result['answer']['relevance_score']}\n")****
    
    ****print(f"Final Answer:\n{mistralexperts_result['answer']['final_answer']}\n\n")****
    
    ****print("Retrieved Documents:")****
    
    ****for i, doc in enumerate(mistralexperts_retrieved_docs, start=1):****
    
     ****print(f"Document {i}: Document ID:****
    
    ****{doc.metadata['id']} source: {doc.metadata['source']}")****
    
    **`What are Google's environmental initiatives?` Is the following:

    ```

    谷歌的环境倡议围绕三个关键支柱组织：赋能个人采取行动，与合作伙伴和客户合作，以及可持续运营其业务。

    ```py

    ```

    1\. 赋能个人：谷歌在谷歌地图中提供环保路线功能，在谷歌 Nest 恒温器中提供节能功能，在谷歌航班中提供碳排放信息。他们的目标是帮助个人、城市和其他合作伙伴到 2030 年共同减少 10 亿吨碳当量排放。

    ```py

    ```

    [TRUNCATED]

    ```py

    ```

    此外，谷歌倡导采取强有力的公共政策行动以创造低碳经济，他们与联合国气候变化框架公约（UNFCCC）合作，支持巴黎协定目标，即保持全球温度上升幅度远低于工业化前水平 2°C。他们还与联盟和可持续性倡议如 RE-Source 平台和谷歌.org 气候创新挑战赛合作。

    ```py

    Compare this to the original response we saw in previous chapters:

    ```

    谷歌的环境倡议包括赋权个人采取行动、与合作伙伴和客户合作、可持续运营、实现净零碳排放、关注水资源管理、参与循环经济，以及支持公共商品的可持续消费。他们还与供应商合作以减少能源消耗和温室气体排放、报告环境数据，并评估环境标准。谷歌参与了各种可持续性倡议，如 iMasons 气候协议、ReFED，以及与大自然保护协会支持的项目。他们还与像 RE-Source 平台和世界可持续发展商业理事会这样的联盟合作。此外，谷歌投资于突破性创新，并与初创公司合作应对可持续性挑战。他们还专注于可再生能源，并使用数据分析工具推动更智能的供应链。

    ```py** 
    ```****

****Llama 3 和专家混合模型的新响应显示了扩展的响应，与原始响应相比似乎更相似，如果不是更稳健，而且成本比 OpenAI 更昂贵但更 gpt-4o-mini 模型低得多。

## 扩展 LLM 功能

这些 LLM 对象的一些方面可以在你的 RAG 应用程序中得到更好的利用。 如 LangChain LLM 文档（[https://python.langchain.com/v0.1/docs/modules/model_io/llms/streaming_llm/](https://python.langchain.com/v0.1/docs/modules/model_io/llms/streaming_llm/)）中所述：

所有大型语言模型（LLM）都实现了 Runnable 接口，该接口提供了所有方法的默认实现，即。 ainvoke, batch, abatch, stream, astream。 这为所有 LLM 提供了基本的异步、流式和批量支持。

这些是关键特性，可以显著加快你的 RAG 应用程序的处理速度，尤其是当你同时处理多个 LLM 调用时。 在接下来的小节中，我们将探讨关键方法以及它们如何 帮助你。

### 异步

默认情况下，异步支持 在单独的线程中运行常规同步方法。 这允许你的异步程序的其他部分在语言模型 工作时继续运行。

### 流

流支持 通常返回 `迭代器` (或 `异步迭代器` 用于异步流) 仅包含一个项目：语言模型最终的结果。 这并不提供逐词流，但它确保你的代码可以 与任何期望流令牌的 LangChain 语言模型集成工作。 流

### 批量处理

批量支持处理 同时处理多个输入。 对于同步批量，它使用多个线程。 对于异步批量，它使用 `asyncio.gather`。您可以使用 `max_concurrency` 设置 在 `RunnableConfig`中 控制 simultaneously running tasks.

尽管如此，并非所有 LLM 都原生支持所有这些功能。 对于我们已经讨论的两个实现以及许多其他实现，LangChain 提供了一个深入的图表，可以在以下位置找到： [https://python.langchain.com/v0.2/docs/integrations/llms/](https://python.langchain.com/v0.2/docs/integrations/llms/)

# 摘要

本章在 LangChain 的背景下探讨了 RAG 系统的关键技术组件：向量存储、检索器和 LLM。 它深入探讨了每个组件的各种选项，并讨论了它们的优缺点以及在某些情况下一个选项可能比另一个选项更好的场景。 尽管并非所有 LLM 都原生支持所有这些功能。

本章首先检查了向量存储，这在高效存储和索引知识库文档的向量表示中起着至关重要的作用。 LangChain 与各种向量存储实现集成，例如 Pinecone、Weaviate、FAISS 和具有向量扩展的 PostgreSQL。 向量存储的选择取决于可扩展性、搜索性能和部署要求等因素。 然后，本章转向讨论检索器，它们负责查询向量存储并根据输入查询检索最相关的文档。 LangChain 提供了一系列检索器实现，包括密集检索器、稀疏检索器（如 BM25）和组合多个检索器结果的集成检索器。

最后，本章讨论了 LLMs 在 RAG 系统中的作用。 LLMs 通过提供对查询上下文的深入理解，并促进从知识库中更有效地检索相关信息，从而为 RAG 做出贡献。 本章展示了 LangChain 与各种 LLM 提供商（如 OpenAI 和 Together AI）的集成，并强调了不同模型的性能和成本考虑。 它还讨论了 LLMs 在 LangChain 中的扩展功能，如异步、流式和批量支持，并提供了不同 LLM 集成提供的原生实现比较。 LLM 集成。

在下一章中，我们将继续讨论如何利用 LangChain 构建一个功能强大的 RAG 应用程序，现在重点关注可以支持我们刚才在本章中讨论的关键组件的较小组件。 这一章。****
