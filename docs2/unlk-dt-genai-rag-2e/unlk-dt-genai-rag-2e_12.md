# 10

# LangChain 中的关键 RAG 组件

本章深入探讨了与**LangChain**和**检索增强生成**（**RAG**）相关的关键技术组件。作为复习，我们 RAG 系统的关键技术组件，按照它们的使用顺序，是**向量存储**、**检索器**和**大型语言模型**（**LLMs**）。我们将逐步介绍我们代码的最新版本，即上一次在*第八章*中看到的*代码实验室-8.3*。我们将关注每个核心组件，并使用 LangChain 在代码中展示每个组件的各种选项。自然地，很多讨论将突出选项之间的差异，并讨论在不同场景下某个选项可能比另一个选项更好的不同情况。

我们从一个概述向量存储选项的代码实验室开始。

# 技术要求

本章的代码放置在以下 GitHub 仓库中：[`github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_10`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_10)。

# 代码实验室 10.1 – LangChain 向量存储

所有这些代码实验室的目标是帮助您更熟悉 LangChain 平台内每个关键组件提供的选项如何增强您的 RAG 系统。我们将深入探讨每个组件的功能、可用函数、有区别的参数，以及最终，您可以利用的所有选项以实现更好的 RAG 实现。从*代码实验室 8.3*开始，（跳过*第九章*的评估代码），我们将按照它们在代码中出现的顺序逐步介绍这些元素，首先是向量存储。您可以在 GitHub 上的*第十章*代码文件夹中找到这些代码的完整内容，也标记为 10.1。

## 向量存储、LangChain 和 RAG

**向量存储**在 RAG 系统中起着至关重要的作用，通过有效地存储和索引知识库文档的向量表示。LangChain 提供了与各种向量存储实现的无缝集成，例如**Chroma**、**Weaviate**、**FAISS**（**Facebook AI Similarity Search**）、**pgvector**和**Pinecone**。对于这个代码实验室，我们将展示将数据添加到 Chroma、Weaviate 和 FAISS 的代码，为您打下能够集成 LangChain 提供的众多向量存储中的任何一种的基础。这些向量存储提供高性能的相似度搜索功能，能够根据查询向量快速检索相关文档。

LangChain 的向量存储类作为与不同向量存储后端交互的统一接口。它提供了向向量存储添加文档、执行相似性搜索和检索存储文档的方法。这种抽象允许开发者轻松地在不同的向量存储实现之间切换，而无需修改核心检索逻辑。

当使用 LangChain 构建 RAG 系统时，可以利用向量存储类来高效地存储和检索文档向量。向量存储的选择取决于诸如可扩展性、搜索性能和部署要求等因素。例如，Pinecone 提供了一个具有高可扩展性和实时搜索能力的完全托管向量数据库，使其适用于生产级 RAG 系统。另一方面，FAISS 提供了一个用于高效相似性搜索的开源库，可用于本地开发和实验。Chroma 由于其易用性和与 LangChain 的有效集成，是开发者在构建他们的第一个 RAG 管道时首选的地方。

如果您查看我们在前几章讨论的代码，我们已经在使用 Chroma。以下是展示我们使用 Chroma 的代码片段，您也可以在本次代码实验室的代码中找到：

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

LangChain 将其称为 **集成**，因为它与第三方 Chroma 进行集成。LangChain 有许多其他集成可用。

在 LangChain 网站上，目前的状态是，在网站页面顶部的网站导航中有 **集成** 链接。如果您点击它，您将看到一个沿着左侧伸展得很远的菜单，其中包括 **提供商** 和 **组件** 的主要类别。正如您可能已经猜到的，这使您能够通过提供商或组件查看所有集成。如果您点击 **提供商**，您将首先看到 **合作伙伴包** 和 **特色社区提供商**。Chroma 目前不在这些列表中，但如果您想了解更多关于 Chroma 作为提供商的信息，请点击页面末尾的链接，该链接说 **点击此处查看所有提供商**。列表按字母顺序排列。向下滚动到 Cs 并找到 Chroma。这将显示与 Chroma 相关的有用 LangChain 文档，特别是关于创建向量存储和检索器的文档。

另一个有用的方法是点击 **组件** 下的 **向量存储**。目前有 49 种向量存储选项！当前链接是针对 0.2.0 版本的，但也要关注未来的版本：

[`python.langchain.com/v0.2/docs/integrations/vectorstores/`](https://python.langchain.com/v0.2/docs/integrations/vectorstores/ )

另一个我们强烈建议您查看的领域是 LangChain 向量存储文档：

[`api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.vectorstores`](https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.vectorstores )

我们已经在过去的章节中深入讨论了我们的当前向量存储和 Chroma 的一般情况，但让我们回顾一下 Chroma 并讨论它在哪里最有用。

### Chroma

**Chroma** 是一个开源的 AI 原生向量数据库，旨在提高开发者的生产力和易用性。它提供了快速的搜索性能，并通过其 Python SDK 与 LangChain 实现无缝集成。Chroma 支持多种部署模式，包括内存中、持久存储以及使用 Docker 容器化的部署。

Chroma 的一个关键优势是其简洁性和对开发者友好的 API。它提供了添加、更新、删除和查询向量存储中文档的直接方法。Chroma 还支持基于元数据的集合动态过滤，允许进行更有针对性的搜索。此外，Chroma 提供了内置的文档分块和索引功能，使得处理大型文本数据集变得方便。

当考虑将 Chroma 作为 RAG 应用的向量存储时，评估其架构和选择标准是很重要的。Chroma 的架构包括一个用于快速向量检索的索引层，一个用于高效数据管理的存储层，以及一个用于实时操作的处理层。Chroma 与 LangChain 集成顺畅，允许开发者在其 LangChain 生态系统中利用其功能。Chroma 客户端可以轻松实例化并传递给 LangChain，从而实现文档向量的有效存储和检索。Chroma 还支持高级检索选项，如**最大边际相关度**（**MMR**）和元数据过滤，以细化搜索结果。

总体而言，Chroma 是开发者寻求一个开源、易于使用的向量数据库，并且与 LangChain 集成良好的一个不错的选择。它的简洁性、快速的搜索性能以及内置的文档处理功能使其成为构建 RAG 应用程序的一个有吸引力的选项。实际上，这也是我们选择在本书的几个章节中突出介绍 Chroma 的原因之一。然而，评估您的具体需求并将 Chroma 与其他向量存储替代方案进行比较，以确定最适合您项目的方案是很重要的。让我们查看代码并讨论一些其他可用的选项，从 FAISS 开始。

### FAISS

让我们首先看看如果我们想使用 FAISS 作为我们的向量存储，我们的代码将如何改变。您首先需要安装 FAISS：

```py
%pip install faiss-cpu 
```

在您重新启动内核（因为您安装了一个新包）后，运行所有代码直到与向量存储相关的单元格，并将与 Chroma 相关的代码替换为 FAISS 向量存储实例化：

```py
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
               documents=dense_documents,
               embedding=embedding_function
) 
```

`Chroma.from_documents()` 方法调用已被替换为 `FAISS.from_documents()`。`collection_name` 和 `client` 参数对 FAISS 不适用，因此已从方法调用中删除。我们重复了一些与 Chroma 向量存储相关的代码，例如文档生成，这使我们能够展示两种向量存储选项之间的代码精确等效。这些更改使得代码现在可以使用 FAISS 而不是 Chroma 作为向量存储。

**FAISS** 是由 Facebook AI 开发的开源库。FAISS 提供高性能搜索能力，可以处理可能无法完全装入内存的大型数据集。与这里提到的其他向量存储类似，FAISS 的架构包括一个索引层，用于组织向量以实现快速检索，一个存储层用于高效的数据管理，以及一个可选的处理层，用于实时操作。FAISS 提供各种索引技术，如聚类和量化，以优化搜索性能和内存使用。它还支持 GPU 加速，以实现更快的相似度搜索。

如果你拥有 NVIDIA GPU，你可以安装 GPU 加速版本，而不是我们之前安装的 CPU 版本：

```py
%pip install faiss-gpu 
```

注意，`faiss-gpu` 需要 NVIDIA GPU 并支持 CUDA。此软件包在 Apple Silicon Mac、AMD GPU 或没有兼容 NVIDIA 显卡的机器上无法工作。如果你不确定你的系统是否有兼容的 GPU，请继续使用 `faiss-cpu`。

使用 FAISS 的 GPU 版本可以显著加快相似度搜索过程，尤其是在处理大规模数据集时。GPU 可以并行处理大量向量比较，从而在 RAG 应用中更快地检索相关文档。如果你在一个处理大量数据且需要比我们之前所做的工作（Chroma）有显著性能提升的环境中工作，你绝对应该测试 FAISS GPU 并看看它对你能产生什么影响。

FAISS LangChain 文档提供了如何在 LangChain 框架中使用 FAISS 的详细示例和指南。它涵盖了诸如摄取文档、查询向量存储、保存和加载索引以及执行高级操作（如过滤和合并）等主题。文档还突出了 FAISS 特有的功能，例如带有分数的相似度搜索和索引的序列化/反序列化。

总体而言，FAISS 是构建 LangChain 的 RAG 应用时的强大且高效的向量存储选项。它的高性能搜索能力、可扩展性和与 LangChain 的无缝集成使其成为寻求强大且可定制解决方案的开发者的一个有吸引力的选择，用于存储和检索文档向量。

这些是满足你的向量存储需求的有力选项。接下来，我们将展示并讨论 Weaviate 向量存储选项。

### Weaviate

您可以使用多种方式来使用和访问 Weaviate。我们将展示嵌入版本，它从您的应用程序代码而不是从独立的 Weaviate 服务器安装中运行 Weaviate 实例。

当嵌入式 Weaviate 首次启动时，它会在 `persistence_data_path` 设置的位置创建一个永久数据存储。当您的客户端退出时，嵌入式 Weaviate 实例也会退出，但数据会持续存在。下次客户端运行时，它将启动嵌入式 Weaviate 的新实例。新的嵌入式 Weaviate 实例使用存储在数据存储中的数据。

如果您熟悉 **GraphQL**，您可能会在开始查看代码时认识到它对 Weaviate 产生的影响。查询语言和 API 受 GraphQL 启发，但 Weaviate 并不直接使用 GraphQL。Weaviate 使用类似 GraphQL 在结构和功能方面的查询语言的 RESTful API。Weaviate 在模式定义中为属性使用预定义的数据类型，类似于 GraphQL 的标量类型。Weaviate 中的可用数据类型包括字符串、整数、数字、布尔值、日期等。

Weaviate 的一项优势是它支持在单个请求中批量操作创建、更新或删除多个数据对象。这与 GraphQL 的突变操作类似，您可以在单个请求中执行多个更改。Weaviate 使用 `client.batch` 上下文管理器将多个操作组合成一个批次，我们将在稍后演示。

让我们首先看看如果我们想将 Weaviate 作为我们的向量存储使用，我们的代码将如何改变。您首先需要安装 Weaviate 及其依赖项。请注意，Weaviate 有特定的依赖项要求，可能与某些系统安装的包冲突，尤其是在 macOS 上使用 Homebrew Python：

```py
%pip install weaviate-client==3.26.0 --no-deps
%pip install requests validators
%pip install authlib --no-deps
%pip install cryptography --no-deps
%pip install langchain-weaviate 
```

在您重新启动内核（因为您安装了新包）后，您需要运行所有代码直到向量存储相关的单元格，并更新代码以使用 FAISS 向量存储实例化：

```py
import weaviate
from langchain_community.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions
from tqdm import tqdm 
```

如您所见，还有许多其他包需要导入以用于 Weaviate。我们还安装了 `tqdm`，这不是 Weaviate 特有的，但它需要，因为 Weaviate 使用 `tqdm` 在加载时显示进度条。

我们必须首先声明 `weaviate_client` 作为 Weaviate 客户端：

```py
weaviate_client = weaviate.Client(
    embedded_options=EmbeddedOptions()) 
```

我们原始的 Chroma 向量存储代码与使用 Weaviate 的区别比我们迄今为止采取的其他方法更复杂。使用 Weaviate，我们使用 `weaviate.Client` 客户端和嵌入选项初始化以启用嵌入式模式，正如您之前所看到的。

在我们继续之前，我们需要确保没有现有的 `weaviate_client` 客户端实例，否则我们的代码将失败：

```py
try:
     weaviate_client.schema.delete_class(collection_name)
except:
        pass 
```

对于 Weaviate，您必须确保清除任何过去迭代中遗留的架构，因为它们可能会在后台持续存在。

然后，我们使用 `weaviate_client` 通过类似 GraphQL 的定义模式建立我们的数据库：

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

这提供了一个完整的模式类，您稍后将其作为`weviate_client`对象的一部分传递给向量存储定义。您需要使用`client.collections.create()`方法为您的集合定义此模式。模式定义包括指定类名、属性及其数据类型。属性可以有不同的数据类型，例如字符串、整数和布尔值。正如您所看到的，与我们在之前使用 Chroma 的实验中使用的相比，Weaviate 强制执行了更严格的模式验证。

虽然这种类似于 GraphQL 的模式在建立向量存储时增加了一些复杂性，但它也以有益和强大的方式为您提供了对数据库的更多控制。特别是，您对如何定义您的模式有了更细粒度的控制。

您可能会认出下面的代码，因为它看起来与我们过去定义的`dense_documents`和`sparse_documents`变量非常相似，但如果你仔细观察，会发现一个在 Weaviate 中很重要的细微差别：

```py
dense_documents = [
    Document(page_content=text,
             metadata={"doc_id": str(i), "source": "dense"})
    for i, text in enumerate(splits)
]
sparse_documents = [
    Document(page_content=text,
             metadata={"doc_id": str(i), "source": "sparse"})
    for i, text in enumerate(splits)
] 
```

当我们使用元数据预处理文档时，这些定义对 Weaviate 有所改变。我们使用`"doc_id"`而不是`"id"`，因为`"id"`是内部使用的，并且不可用于我们的用途。在代码的后续部分，当您从元数据结果中提取 ID 时，您将希望更新该代码以使用`"doc_id"`。

接下来，我们定义我们的向量存储，类似于我们过去使用 Chroma 和 FAISS 所做的那样，但使用 Weaviate 特定的参数：

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

对于向量存储初始化，Chroma 使用`from_documents`方法直接从文档创建向量存储，而 Weaviate 则是先创建向量存储，然后添加文档。Weaviate 还需要额外的配置，例如`text_key`、`attributes`和`by_text`。一个主要区别是 Weaviate 对模式的使用。

最后，我们将实际内容加载到 Weaviate 向量存储实例中，在此过程中也应用了`embedding`函数：

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

总结来说，Chroma 提供了一种更简单、更灵活的数据模式定义方法，并专注于嵌入存储和检索。它可以轻松地嵌入到您的应用程序中。另一方面，Weaviate 提供了一个结构更严谨、功能更丰富的向量数据库解决方案，具有显式的模式定义、多个存储后端以及内置对各种嵌入模型的支持。它可以作为独立服务器部署或托管在云端。在 Chroma、Weaviate 或其他任何向量存储之间的选择取决于您的具体需求，例如模式灵活度、部署偏好以及是否需要除嵌入存储之外的其他功能。

注意，你可以使用这些向量存储中的任何一个，剩余的代码将能够与加载到其中的数据进行工作。这是使用 LangChain 的一个优势，它允许你交换组件。这在生成式 AI 的世界中尤其必要，因为新的和显著改进的技术正在不断推出。使用这种方法，如果你遇到一种更新、更好的向量存储技术，它对你的 RAG 管道产生了影响，你可以相对快速且容易地做出这种改变。

接下来，让我们讨论 LangChain 武器库中的另一个关键组件，它是 RAG 应用程序的核心：检索器。

# 代码实验室 10.2 – LangChain 检索器

在这个代码实验室中，我们将介绍检索过程中最重要的组件的一些示例：**LangChain 检索器**。与 LangChain 向量存储一样，LangChain 检索器的选项太多，无法在此列出。我们将关注一些特别适用于 RAG 应用的流行选择，并鼓励您查看所有其他选项，看看是否有更适合您特定情况的选项。就像我们讨论向量存储一样，LangChain 网站上有大量的文档可以帮助您找到最佳解决方案：[`python.langchain.com/v0.2/docs/integrations/retrievers/`](https://python.langchain.com/v0.2/docs/integrations/retrievers/)

检索器包的文档可以在这里找到：[`api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.retrievers`](https://api.python.langchain.com/en/latest/core_api_reference.html#module-langchain_core.retrievers)

现在，让我们开始为检索器编写代码！

## 检索器、LangChain 和 RAG

**检索器**负责查询向量存储并基于输入查询检索最相关的文档。LangChain 提供了一系列的检索器实现，可以与不同的向量存储和查询编码器一起使用。

在我们目前的代码中，我们已经看到了三个版本的检索器；让我们首先回顾它们，因为它们与原始基于 Chroma 的向量存储相关。

### 基本检索器（稠密嵌入）

我们从 **稠密检索器** 开始。这是我们到目前为止在几个代码实验室中使用的代码：

```py
dense_retriever = vectorstore.as_retriever(
                      search_kwargs={"k": 10}) 
```

稠密检索器是通过 `vectorstore.as_retriever` 函数创建的，指定要检索的顶部结果数量（`“k”:10`）。在这个检索器的底层，Chroma 使用文档的稠密向量表示，并使用余弦距离或欧几里得距离进行相似度搜索，根据查询嵌入检索最相关的文档。

这是在使用最简单的检索器类型，即向量存储检索器，它为每段文本创建嵌入，并使用这些嵌入进行检索。检索器本质上是对向量存储的一个包装。使用这种方法可以让你访问向量存储内置的检索/搜索功能，同时与 LangChain 生态系统集成和接口。它是一个轻量级的包装器，围绕向量存储类，为 LangChain 中所有检索器选项提供一致的接口。正因为如此，一旦你构建了一个向量存储，构建检索器就非常容易。如果你需要更改你的向量存储或检索器，这也非常容易做到。

这些类型的检索器有两个主要的搜索能力，直接源于它们所包装的向量存储：相似性搜索和 MMR。

### 相似度分数阈值检索

默认情况下，检索器使用相似性搜索。但是，如果你想设置一个相似度阈值，你只需将搜索类型设置为`similarity_score_threshold`，并在传递给`retriever`对象的`kwargs`函数中设置该相似度分数阈值。代码如下所示：

```py
dense_retriever = vectorstore.as_retriever(
               search_type="similarity_score_threshold",
               search_kwargs={"score_threshold": 0.5}
) 
```

这是对默认相似性搜索的一个有用升级，在许多 RAG 应用中可能很有用。然而，相似性搜索并不是这些检索器可以支持的唯一搜索类型；还有 MMR。

### MMR

**MMR**是一种在检索相关项目时避免冗余的技术。它平衡了检索到的项目的相关性和多样性，而不是简单地检索最相关的项目，这些项目可能相似。MMR 常用于信息检索，并且可以用来通过计算文本部分之间的相似性来总结文档。为了设置你的检索器使用这种类型的搜索，而不是相似性搜索，你可以在定义检索器时添加`search_type="mmr"`作为参数，如下所示：

```py
dense_retriever = vectorstore.as_retriever(
                      search_type="mmr"
) 
```

将此添加到任何基于向量存储的检索器中，将使其利用 MMR 类型的搜索。

相似性搜索和 MMR 可以由支持这些搜索技术的任何向量存储支持。接下来，让我们谈谈我们在*第八章*中引入的稀疏搜索机制，即 BM25 检索器。

### BM25 检索器

**BM25**是一种用于稀疏文本检索的排名函数，`BM25Retriever`是 LangChain 对 BM25 的表示，可用于稀疏文本检索目的。

你也见过这个检索器，因为我们用它将我们的基本搜索转变为混合搜索，见*第八章*。我们在代码中看到这些设置：

```py
sparse_retriever = BM25Retriever.from_documents(
    sparse_documents, k=10) 
```

调用`BM25Retriever.from_documents()`方法从稀疏文档中创建一个稀疏检索器，指定要检索的前 N 个结果数（`k=10`）。

BM25 通过计算每个文档的与查询术语相关的得分，基于文档的**词频和逆文档频率**（**TF-IDF**）来实现。它使用一个概率模型来估计文档与给定查询的相关性。检索器返回具有最高 BM25 得分的 top-k 文档。

### 集成检索器

**集成检索器**结合了多种检索方法，并使用一个额外的算法将它们的结果组合成一个集合。这种类型检索器的理想用途是在您想结合密集和稀疏检索器以支持混合检索方法时，例如我们在*第八章*的*代码实验室 8.3*中创建的：

```py
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5], c=0, k=10) 
```

在我们的案例中，集成检索器结合了 Chroma 密集检索器和 BM25 稀疏检索器，以实现更好的检索性能。它是使用`EnsembleRetriever`类创建的，该类接受检索器的列表及其相应的权重。在这种情况下，密集检索器和稀疏检索器以相等的权重 0.5 传递。

在集成检索器中，`c`参数是一个重新排序参数，它控制原始检索得分和重新排序得分之间的平衡。它用于调整重新排序步骤对最终检索结果的影响。在这种情况下，`c`参数设置为`0`，这意味着不执行重新排序。当`c`设置为非零值时，集成检索器对检索到的文档执行额外的重新排序步骤。重新排序步骤根据单独的重新排序模型或函数重新评分检索到的文档。重新排序模型可以考虑到额外的特征或标准来评估文档与查询的相关性。

在 RAG 应用中，检索到的文档的质量和相关性直接影响生成的输出。通过利用`c`参数和合适的重新排序模型，您可以增强检索结果，以更好地满足您 RAG 应用的具体要求。例如，您可以设计一个重新排序模型，考虑到诸如文档相关性、与查询的一致性或特定领域标准等因素。通过为`c`设置适当的值，您可以在原始检索得分和重新排序得分之间取得平衡，在需要时给予重新排序模型更多的权重。这可以帮助优先考虑对 RAG 任务更相关、更有信息量的文档，从而提高生成的输出质量。

当查询传递给集成检索器时，它会将查询发送给密集和稀疏检索器。然后，集成检索器根据分配的权重结合两个检索器的结果，并返回前 k 个文档。在底层，集成检索器利用密集和稀疏检索方法的优势。密集检索通过密集向量表示捕获语义相似性，而稀疏检索则依赖于关键词匹配和词频。通过结合它们的结果，集成检索器旨在提供更准确和全面的搜索结果。

在代码片段中使用的特定类和方法可能因所使用的库或框架而异。然而，使用向量相似性搜索进行密集检索、使用 BM25 进行稀疏检索以及结合多个检索器的集成检索的一般概念保持不变。

这涵盖了我们在之前的代码中看到的检索器，所有这些都是从我们在索引阶段访问和处理的数据中提取的。在 LangChain 网站上，你可以探索许多其他类型的检索器，以满足你的需求。然而，并非所有检索器都旨在从你正在处理的数据中提取。接下来，我们将回顾一个基于公共数据源（维基百科）构建的检索器示例。

### 维基百科检索器

如 LangChain 网站上的维基百科检索器创建者所述（[`www.langchain.com/`](https://www.langchain.com/))：

*维基百科是历史上最大、阅读量最高的参考工具，它是一个由志愿者社区编写和维护的多语言免费在线百科全书*。

这听起来像是一个在您的 RAG 应用中挖掘有用知识的绝佳资源！我们将在现有的检索器单元之后添加一个新的单元，我们将使用这个维基百科检索器从 wikipedia.org 检索维基页面到下游使用的文档格式。

我们首先需要安装几个新的包：

```py
%pip install langchain_core
%pip install --upgrade --quiet wikipedia==1.4.0 
```

像往常一样，当你安装新包时，别忘了重启你的内核！

使用`WikipediaRetriever`检索器，我们现在有一个机制可以获取与用户查询相关的维基百科数据，类似于我们使用的其他检索器，但使用背后的整个维基百科数据：

```py
from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever(load_max_docs=10)
docs = retriever.get_relevant_documents(
    query= “What defines the golden age of piracy in the Caribbean?”)
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

在这段代码中，我们从`langchain_community.retrievers`模块中导入`WikipediaRetriever`类。`WikipediaRetriever`是一个专门设计用来根据给定查询从维基百科检索相关文档的检索器类。然后我们使用`WikipediaRetriever`类实例化一个接收器实例，并将其分配给变量`retriever`。`load_max_docs`参数设置为 10，表示检索器应加载最多 10 个相关文档。这里的用户查询是`"What defines the golden age of piracy in the Caribbean?"`，我们可以查看响应以了解维基百科文章是如何被检索出来帮助回答这个问题的。

我们调用检索器对象的`get_relevant_documents`方法，传入一个查询字符串作为参数，并在响应中接收这个作为第一份文档：

```py
First document returned:
Title: Golden Age of Piracy
Summary: The Golden Age of Piracy is a common designation for the period between the 1650s and the 1730s, when maritime piracy was a significant factor in the histories of the North Atlantic and Indian Oceans.
Histories of piracy often subdivide the Golden Age of Piracy into three periods:
The buccaneering period (approximately 1650 to 1680)… 
```

你可以在以下链接中看到匹配的内容：

[`en.wikipedia.org/wiki/Golden_Age_of_Piracy`](https://en.wikipedia.org/wiki/Golden_Age_of_Piracy)

这个链接是由检索器提供的源。

总结来说，这段代码展示了如何使用`langchain_community.retrievers`模块中的`WikipediaRetriever`类根据给定查询从维基百科检索相关文档。然后它提取并打印特定的元数据信息（标题、摘要、来源）以及检索到的第一份文档的内容。

`WikipediaRetriever`内部处理查询维基百科 API 或搜索功能、检索相关文档并将它们作为`Document`对象列表返回的过程。每个`Document`对象包含元数据和实际页面内容，可以根据需要访问和使用。还有许多其他检索器可以访问类似此类公共数据源，但专注于特定领域。在科学研究中，有`PubMedRetriever`。在其他研究领域，如数学和计算机科学，有`ArxivRetreiver`，它访问关于这些主题的超过 200 万篇开放获取档案的数据。在金融领域，有一个名为`KayAiRetriever`的检索器，可以访问**证券交易委员会**（**SEC**）的文件，这些文件包含上市公司必须提交给美国 SEC 的财务报表。

对于处理非大规模数据的项目的检索器，我们还有一个要强调的：kNN 检索器。

### kNN 检索器

到目前为止，我们一直在使用的最近邻算法，即负责找到与用户查询最相关内容的算法，是基于**近似最近邻**（**ANN**）。然而，存在一个更**传统**且更**古老**的算法，可以作为 ANN 的替代方案，那就是**k 最近邻**（**kNN**）。但 kNN 基于一个可以追溯到 1951 年的算法；为什么我们还要使用这个算法，当我们有更复杂、更强大的算法，如 ANN 可用时？因为 kNN 仍然**优于**其后出现的任何算法。这不是一个错误。kNN 仍然是找到最近邻的**最有效**方法。它比 ANN 更好，ANN 被数据库、向量数据库和信息检索领域的所有公司吹捧为**解决方案**。ANN 接近这个水平，但 kNN 仍然被认为更好。

那么，为什么 ANN 被吹捧为**解决方案**呢？因为 kNN 无法扩展到这些供应商针对的大型企业所看到的水平。但这都是相对的。你可能有一百万个数据点，这听起来很多，有 1536 维向量，但在全球企业舞台上这仍然被认为相当小。kNN 可以轻松处理这一点！许多在领域中使用 ANN 的小型项目可能从使用 kNN 中受益。kNN 的理论极限将取决于许多因素，例如你的开发环境、你的数据、数据的维度、如果使用 API，则还有互联网连接性等等。因此，我们无法给出具体的数据点数。你需要进行测试。但如果它小于我刚才描述的项目（1 百万数据点，1536 维向量），在一个相对强大的开发环境中，你真的应该考虑 kNN！在某个时候，你会注意到处理时间的显著增加，当等待时间变得过长，以至于你的应用程序的实用性降低时，切换到 ANN。但在同时，务必充分利用 kNN 的优越搜索能力。

幸运的是，kNN 可以通过一个易于设置的检索器`KNNRetriever`获得。这个检索器将利用我们与其他算法一起使用的相同密集嵌入，因此我们将用基于 kNN 的`KNNRetriever`替换`dense_retriever`。以下是实现这一点的代码，它很好地放置在我们定义了之前的`dense_retriever`检索器对象之后：

```py
from langchain_community.retrievers import KNNRetriever
dense_retriever = KNNRetriever.from_texts(splits,
    OpenAIEmbeddings(), k=10)
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5], c=0, k=10) 
```

在代码实验室中运行剩余的代码，以查看它取代我们之前的`dense_retriever`检索器并执行其功能。在这种情况下，由于数据集非常有限，很难评估它是否比我们之前使用的基于 ANN 的算法做得更好。但是，随着你的项目扩展，我们强烈建议你利用这种方法，直到其扩展问题变得过于沉重。

这就结束了我们对支持 RAG 的检索器的探索。LangChain 网站上还有其他类型的检索器，以及与支持这些检索器的向量存储的显著集成，可以查阅。例如，有一个时间加权向量存储检索器，允许你在检索过程中纳入近期性。还有一个名为“长上下文重排”的检索器，专注于改善难以关注检索文档中间信息的长上下文模型的结果。务必查看可用的内容，因为它们有可能对您的 RAG 应用产生重大影响。我们现在将转向讨论操作和生成阶段的“大脑”：LLMs。

# 代码实验室 10.3 – LangChain LLMs

现在，我们将注意力转向 RAG 的最后一个关键组件：LLM。就像检索阶段的检索器一样，如果没有生成阶段的 LLM，就没有 RAG。检索阶段只是从我们的数据源中检索数据——通常是 LLM 不知道的数据。然而，这并不意味着 LLM 在我们的 RAG 实现中不起重要作用。通过向 LLM 提供检索到的数据，我们迅速让 LLM 了解我们希望它讨论的内容，这使得 LLM 能够发挥其真正擅长的能力：根据这些数据提供响应来回答用户提出的原始问题。

LLMs 和 RAG 系统之间的协同作用源于这两种技术的互补优势。RAG 系统通过整合外部知识源，增强了 LLMs 的能力，使其能够生成不仅与上下文相关，而且事实准确且最新的响应。反过来，LLMs 通过提供对查询上下文的复杂理解，促进了从知识库中更有效地检索相关信息。这种共生关系显著提高了 AI 系统在需要深度语言理解和广泛事实信息访问的任务中的性能，利用每个组件的优势，创建了一个更强大、更通用的系统。

在这个代码实验室中，我们将介绍生成阶段最重要的组件的一些示例：LangChain LLM。

## LLMs，LangChain，和 RAG

与之前的关键组件一样，我们首先提供与 LLMs 相关的 LangChain 文档链接，这是这个主要组件：[`python.langchain.com/v0.2/docs/integrations/llms/`](https://python.langchain.com/v0.2/docs/integrations/llms/)

第二个有用的信息来源是将 LLMs 与 LangChain 结合的 API 文档：[`api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.llms`](https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.llms)

让我们从我们已经使用的 API 开始：OpenAI。

### OpenAI

我们已经有了这段代码，但让我们通过逐步检查我们实验室中使该组件工作的关键区域来刷新这段代码的内部工作原理：

1.  首先，我们必须安装 `langchain-openai` 包：

    ```py
    %pip install langchain-openai 
    ```

1.  `langchain-openai` 库提供了 OpenAI 的语言模型与 LangChain 之间的集成。

1.  接下来，我们导入 `openai` 库，这是官方的 Python 库，用于与 OpenAI 的 API 交互，并将主要用于将 API 密钥应用于模型，以便我们可以访问付费 API。然后，我们从 `langchain_openai` 库中导入 `ChatOpenAI` 和 `OpenAIEmbeddings` 类：

    ```py
    import openai
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
    ```

`ChatOpenAI` 用于与 OpenAI 的聊天模型交互，而 `OpenAIEmbeddings` 用于从文本生成嵌入。

1.  在下一行中，我们使用 `load_dotenv` 函数从名为 `env.txt` 的文件中加载环境变量：

    ```py
    _ = load_dotenv(dotenv_path='env.txt') 
    ```

1.  我们使用 `env.txt` 文件以这种方式存储敏感信息（一个 API 密钥），这样我们就可以将其隐藏在我们的版本控制系统之外，实践更好的和更安全的秘密管理。

1.  然后，我们使用以下代码将 API 密钥传递给 OpenAI：

    ```py
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    openai.api_key = os.environ['OPENAI_API_KEY'] 
    ```

1.  我们首先将 API 密钥设置为一个名为 `OPENAI_API_KEY` 的环境变量。然后，我们使用从环境变量中检索到的值来设置 OpenAI 库的 OpenAI API 密钥。此时，我们可以使用 LangChain 与 OpenAI 的集成来调用托管在 OpenAI 上的 LLM，并使用适当的访问权限。

1.  在代码的后面部分，我们定义了我们想要使用的 LLM：

    ```py
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) 
    ```

这行代码创建了一个 `ChatOpenAI` 类的实例，指定模型名称为 gpt-4o-mini，并将温度变量设置为 0。温度控制生成响应的随机性，较低值会产生更专注和确定性的输出。目前，gpt-4o-mini 是最新且功能最强大的模型，同时也是 GPT4 系列中最具成本效益的模型。但即使是这个模型，其成本也比 gpt-3.5-turbo 高 10 倍，而 gpt-3.5-turbo 实际上是一个相对强大的模型。

OpenAI 中最昂贵的模型 gpt-4-32k，其速度和能力不如 gpt-4o-mini，其上下文窗口是其大小的 4 倍。很可能很快就会有新的模型出现，包括 gpt-5，这些模型可能成本更低且功能更强大。从所有这些中，你可以吸取的教训是，你不应该仅仅假设最新的模型将是成本最高的，而且总会有更强大且更具成本效益的替代版本出现。要勤于关注模型的最新发布，并且对于每个发布，权衡成本、LLM 功能和其他相关属性的好处，以决定是否需要进行更改。

但在这个努力中，你不需要将自己限制在仅使用 OpenAI。使用 LangChain 可以轻松切换 LLM，并扩大你在 LangChain 社区中寻找最佳解决方案的搜索范围。让我们来看看你可能考虑的其他一些选项。

### Together AI

**Together AI**提供了一个开发者友好的平台，通过简单的 API 访问 200 多个 OSS 和特殊模型，以及专用端点和 GPU 集群的选项。

创建一个账户，然后导航到**设置** **→** **API 密钥**以生成 API 密钥。将其设置为`TOGETHER_API_KEY`环境变量。如果你是 Together API 的新用户，你可以使用此链接设置你的 API 密钥并将其添加到你的`env.txt`文件中，就像我们过去使用 OpenAI API 密钥时做的那样：[`api.together.ai/settings/api-keys`](https://api.together.ai/settings/api-keys)

一旦登录，你就可以在这里看到每个 LLM 的当前费用：[`api.together.ai/models`](https://api.together.ai/models)

例如，Meta Llama 3.3 70B Instruct-Turbo 目前列出的价格是每 1M 个 token 0.88 美元，而 Mixtral 8X7B Instruct v0.1 的价格是每 1M 个 token 0.60 美元。按照以下步骤设置和使用 Together AI：

1.  我们首先安装使用 Together API 所需的包：

    ```py
    %pip install --upgrade langchain-together 
    ```

1.  这为我们使用 Together API 和 LangChain 之间的集成做好了准备：

    ```py
    from langchain_together import ChatTogether
    _ = load_dotenv(dotenv_path='env.txt') 
    ```

1.  这导入了我们在 LangChain 中需要使用`ChatTogether`集成的包，并加载了 API 密钥（在运行此行代码之前别忘了将其添加到`env.txt`文件中！）。

1.  就像我们之前使用 OpenAI API 密钥做的那样，我们将拉入`TOGETHER_API_KEY`以便它可以访问你的账户：

    ```py
    os.environ['TOGETHER_API_KEY'] = os.getenv(
        'TOGETHER_API_KEY') 
    ```

1.  我们将使用 Llama 3 Chat 模型和 Mistral 的 Mixtral 8X22B Instruct 模型，但你可以在这里选择 50 多个模型：[`docs.together.ai/docs/inference-models`](https://docs.together.ai/docs/inference-models)

你可能会找到更适合你特定需求的更好模型！

在这里，我们正在定义模型：

```py
llama3llm = ChatTogether(
    together_api_key=os.environ['TOGETHER_API_KEY'],
    model="meta-llama/Llama-3-70b-chat-hf",
)
mistralexpertsllm = ChatTogether(
    together_api_key=os.environ['TOGETHER_API_KEY'],
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
) 
```

1.  在前面的代码片段中，我们正在建立两个不同的 LLM，我们可以在剩余的代码中运行并查看结果。

在这里，我们已更新了使用 Llama 3 模型的最终代码：

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

1.  这应该看起来很熟悉，因为它是我们过去使用的 RAG 链，但现在使用的是 Llama 3 LLM。

    ```py
    llama3_rag_chain_with_source = RunnableParallel(
        {"context": ensemble_retriever,
         "question": RunnablePassthrough()}
    ).assign(answer=llama3_rag_chain_from_docs) 
    ```

这是我们的最终 RAG 链，已更新为之前的以 Llama 3 为重点的 RAG 链。

1.  接下来，我们想要运行与过去类似的代码，调用并运行 RAG 管道，用 Llama 3 LLM 替换 ChatGPT-4o-mini 模型：

    ```py
    llama3_result = llama3_rag_chain_with_source.invoke(
        user_query)
    llama3_retrieved_docs = llama3_result['context']
    print(f"Original Question: {user_query}\n")
    print(f"Relevance Score:
        {llama3_result['answer']['relevance_score']}\n")
    print(f"Final Answer:
        \n{llama3_result['answer']['final_answer']}\n\n")
    print("Retrieved Documents:")
    for i, doc in enumerate(llama3_retrieved_docs, start=1):
        print(f"Document {i}: Document ID:
            {doc.metadata['id']} source:
            {doc.metadata['source']}")
            print(f"Content:\n{doc.page_content}\n") 
    ```

1.  对问题“谷歌的环境倡议是什么？”的最终响应如下：

    ```py
    Google's environmental initiatives include:
    1\. Empowering individuals to take action: Offering sustainability features in Google products, such as eco-friendly routing in Google Maps, energy efficiency features in Google Nest thermostats, and carbon emissions information in Google Flights…
    [TRUNCATED]
    10\. Engagement with external targets and initiatives: Participating in industry-wide initiatives and partnerships to promote sustainability, such as the RE-Source Platform, iMasons Climate Accord, and World Business Council for Sustainable Development. 
    ```

1.  让我们看看如果我们使用专家混合模型会是什么样子：

    ```py
    mistralexperts_rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=(lambda x:format_docs(x[“context”])))
        | RunnableParallel(
            {“relevance_score”:
                (RunnablePassthrough()
                | (lambda x: relevance_prompt_template.format(
                    question=x[‘question’],
                    retrieved_context=x[‘context’]))
                | mistralexpertsllm
                | StrOutputParser()
            ), “answer”: (
                RunnablePassthrough()
                | prompt
                | mistralexpertsllm
                | StrOutputParser()
            )}
        )
        | RunnablePassthrough().assign(final_answer=conditional_answer)
    ) 
    ```

1.  再次强调，这应该看起来很熟悉，因为它是我们过去使用的 RAG 链，但这次使用的是专家混合 LLM。

    ```py
    mistralexperts_rag_chain_with_source = RunnableParallel(
        {"context": ensemble_retriever,
         "question": RunnablePassthrough()}
    ).assign(answer=mistralexperts_rag_chain_from_docs) 
    ```

就像我们之前做的那样，我们更新了最终的 RAG 管道，使用之前以混合专家为重点的 RAG 链。

1.  这段代码将让我们看到专家混合模型替换 ChatGPT-4o-mini 模型的结果：

    ```py
    mistralexperts_result = mistralexperts_rag_chain_with_source.invoke(
        user_query
    )
    mistralexperts_retrieved_docs = mistralexperts_result[‘context’]
    print(f”Original Question: {user_query}\n”)
    print(
        f”Relevance Score: {mistralexperts_result[‘answer’]”
        f”[‘relevance_score’]}\n”
    )
    print(f”Final Answer:\n”
        f”{mistralexperts_result[‘answer’][‘final_”
        f”answer’]}\n\n”)
    print(“Retrieved Documents:”)
    for i, doc in enumerate(mistralexperts_retrieved_docs, start=1):
        print(f”Document {i}: Document ID:\n”
            f”{doc.metadata[‘id’]} source: {doc.metadata[‘source’]}”)
        print(f”Content:\n{doc.page_content}\n”) 
    ```

1.  对 `“Google 的环境倡议是什么？”` 的响应如下：

    ```py
    Google's environmental initiatives are organized around three key pillars: empowering individuals to take action, working together with partners and customers, and operating their business sustainably.
    1\. Empowering individuals: Google provides sustainability features like eco-friendly routing in Google Maps, energy efficiency features in Google Nest thermostats, and carbon emissions information in Google Flights. Their goal is to help individuals, cities, and other partners collectively reduce 1 gigaton of carbon equivalent emissions annually by 2030.
    [TRUNCATED]
    Additionally, Google advocates for strong public policy action to create low-carbon economies, they work with the United Nations Framework Convention on Climate Change (UNFCCC) and support the Paris Agreement's goal to keep global temperature rise well below 2°C above pre-industrial levels. They also engage with coalitions and sustainability initiatives like the RE-Source Platform and the Google.org Impact Challenge on Climate Innovation. 
    ```

1.  与之前章节中看到的原始响应进行比较：

    ```py
    Google's environmental initiatives include empowering individuals to take action, working together with partners and customers, operating sustainably, achieving net-zero carbon emissions, focusing on water stewardship, engaging in a circular economy, and supporting sustainable consumption of public goods. They also engage with suppliers to reduce energy consumption and greenhouse gas emissions, report environmental data, and assess environmental criteria. Google is involved in various sustainability initiatives, such as the iMasons Climate Accord, ReFED, and supporting projects with The Nature Conservancy. They also work with coalitions like the RE-Source Platform and the World Business Council for Sustainable Development. Additionally, Google invests in breakthrough innovation and collaborates with startups to tackle sustainability challenges. They also focus on renewable energy and use data analytics tools to drive more intelligent supply chains. 
    ```

Llama 3 的新响应和专家混合模型的组合显示，与使用 OpenAI 的 gpt-4o-mini 模型所能实现的原始响应相比，响应似乎更加丰富，如果不说更稳健，而且成本远低于 OpenAI 更昂贵但功能更强大的模型。

## 扩展 LLM 功能

如 LangChain LLM 文档（[`python.langchain.com/v0.1/docs/modules/model_io/llms/streaming_llm/`](https://python.langchain.com/v0.1/docs/modules/model_io/llms/streaming_llm/)）中所述，这些 LLM 对象的某些方面可以在您的 RAG 应用程序中得到更好的利用。

所有 LLM 都实现了 Runnable 接口，该接口提供了所有方法的默认实现，即 `ainvoke`、`batch`、`abatch`、`stream`、`astream`。这为所有 LLM 提供了基本的异步、流和批量支持。

这些是可以在您的 RAG 应用程序中显著加快处理速度的关键特性，尤其是如果您同时处理多个 LLM 调用。在接下来的小节中，我们将探讨关键方法以及它们如何帮助您。

### 异步

默认情况下，异步支持在单独的线程中运行常规 `sync` 方法。这允许您的异步程序的其他部分在语言模型工作时继续运行。

### 流

流支持通常返回 `Iterator`（或异步流中的 `AsyncIterator`）仅包含一个项目：语言模型的最终结果。这并不提供逐词流，但它确保您的代码可以与任何期望流令牌的 LangChain 语言模型集成一起工作。

### 批量

批量支持同时处理多个输入。对于同步批量，它使用多个线程。对于异步批量，它使用 `asyncio.gather`。您可以使用 `RunnableConfig` 中的 `max_concurrency` 设置来控制同时运行的任务数量。

虽然并非所有 LLM 都原生支持所有这些功能。对于我们已经讨论的两个实现以及许多其他实现，LangChain 提供了一个深入的分析图表，您可以在以下链接中找到：[`python.langchain.com/v0.2/docs/integrations/llms/`](https://python.langchain.com/v0.2/docs/integrations/llms/)

# 摘要

本章在 LangChain 的背景下探讨了 RAG 系统的关键技术组件：向量存储、检索器和 LLM。它深入探讨了每个组件的各种选项，并讨论了它们的优缺点以及在某些情况下一个选项可能比另一个选项更好的场景。

本章首先检查了向量存储，它在高效存储和索引知识库文档的向量表示中起着至关重要的作用。LangChain 与各种向量存储实现集成，例如 Pinecone、Weaviate、FAISS 和具有向量扩展的 PostgreSQL。向量存储的选择取决于可扩展性、搜索性能和部署需求等因素。然后，本章继续讨论检索器，它们负责查询向量存储并根据输入查询检索最相关的文档。LangChain 提供了一系列检索器实现，包括密集检索器、稀疏检索器（如 BM25）以及结合多个检索器结果的集成检索器。

最后，本章讨论了 LLMs 在 RAG 系统中的作用。LLMs 通过提供对查询上下文的深入理解，并促进从知识库中更有效地检索相关信息，从而为 RAG 做出贡献。本章展示了 LangChain 与各种 LLM 提供商（如 OpenAI 和 Together AI）的集成，并强调了不同模型的性能和成本考虑。它还讨论了 LLMs 在 LangChain 中的扩展功能，如异步、流式和批量支持，并提供了不同 LLM 集成提供的本地实现比较。

在下一章中，我们将继续讨论如何利用 LangChain 构建一个功能强大的 RAG 应用程序，重点关注可以支持我们刚才在本章中讨论的关键组件的较小组件。

# 免费订阅电子书

新框架、演进的架构、研究进展、生产分解——AI_Distilled 将噪音过滤成每周简报，供那些与 LLMs 和 GenAI 系统实际操作工程师和研究人员阅读。现在订阅，即可获得免费电子书，以及每周的洞察力，帮助您保持专注并获取信息。

在[`packt.link/8Oz6Y`](https://packt.link/8Oz6Y)订阅或扫描下面的二维码。

![白色背景上的二维码  AI 生成的内容可能不正确。](img/B34736_Free_eBook.png)
