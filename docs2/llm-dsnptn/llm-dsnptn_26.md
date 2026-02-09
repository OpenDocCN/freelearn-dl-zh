# 26

# 检索增强生成

**检索增强生成**（**RAG**）是一种增强 AI 模型性能的技术，尤其是在需要模型预训练参数中不包含的知识或数据的任务中。它结合了基于检索的模型和生成模型的优点。检索组件从外部来源，如数据库、文档或网络内容中检索相关信息，而生成组件则使用这些信息生成更准确、更具情境丰富性的响应。

RAG 是通过将检索机制与语言模型集成来实现的。这个过程从查询知识库或外部资源以获取相关文档或片段开始。然后，这些检索到的信息被输入到语言模型中，通过结合提示和检索到的数据生成响应。这种方法提高了模型回答问题或解决问题的能力，特别是对于它原本缺乏的更新或特定领域的信息。

在本章中，我们将向您介绍 RAG。您将学习如何实现一个简单的 RAG 系统，该系统可以使用相关的外部信息增强 LLM 的输出。

RAG 的关键优势包括增强的事实准确性、获取最新信息、改进的特定领域知识，以及减少 LLM 输出中的幻觉。

在本章中，我们将介绍嵌入和索引技术，用于高效检索向量数据库，查询制定策略，以及将检索信息与 LLM 生成集成的方法。到本章结束时，你将能够实现基本的 RAG 系统，以增强你的 LLM 外部知识。

在本章中，我们将涵盖以下主题：

+   为 LLM 构建简单的 RAG 系统

+   LLM 检索的嵌入和索引技术

+   基于 LLM 的 RAG 查询制定策略

+   将检索信息与 LLM 生成集成

+   LLM 中 RAG 的挑战和机遇

# 为 LLM 构建简单的 RAG 系统

本节提供了一个简单 RAG 系统的实际示例，利用**SerpApi**强大的搜索能力、句子嵌入的语义理解以及 OpenAI 的 GPT-4o 模型的生成能力。SerpApi 是一个网络抓取 API，提供对搜索引擎结果的实时访问，为 Google、Bing 和其他平台提供结构化数据，无需手动抓取。

通过这个例子，我们将探讨 RAG 系统的基本组件，包括基于查询的网页搜索、片段提取和排名，以及最终使用最先进的 LLM 生成全面答案的过程，以逐步方式突出这些元素之间的相互作用。

我们将要构建的简单 RAG 系统的代码包含以下内容：

+   **SerpApi**：根据用户的查询找到相关的网页。

+   **句子嵌入**: 通过使用句子嵌入和余弦相似度来从搜索结果中提取最相关的片段。句子嵌入是文本的密集数值表示，通过将单词、短语或整个句子映射到高维向量空间来捕获语义意义，其中相似的意义被放置得更近。余弦相似度衡量这些嵌入向量之间的角度（范围从-1 到 1），而不是它们的幅度，这使得它成为评估语义相似性的有效方法，无论文本长度如何；当两个嵌入的余弦相似度接近 1 时，它们在意义上高度相似，而接近 0 的值表示无关内容，负值则表示相反的意义。这种技术的组合为许多现代**自然语言处理**（**NLP**）应用提供了动力，从搜索引擎和推荐系统到语言翻译和内容聚类。

+   **OpenAI 的 GPT-4o**: 基于检索到的片段（上下文）和原始查询生成全面且连贯的答案。

首先，让我们安装以下依赖项：

```py
pip install google-search-results sentence-transformers openai
```

在前面的命令中，我们安装了`serpapi`用于搜索，`sentence_transformers`用于嵌入，以及`openai`用于访问 GPT-4o。

接下来，让我们看看如何使用搜索 API、嵌入和 LLM 实现一个完整的 RAG 系统：

1.  我们首先导入安装的库以及`torch`进行张量操作。代码片段还设置了 SerpApi 和 OpenAI 的 API 密钥。请记住用你实际的 API 密钥替换占位符：

    ```py
    from serpapi import GoogleSearch
    from sentence_transformers import SentenceTransformer, util
    import torch
    import openai
    SERPAPI_KEY = "YOUR_SERPAPI_KEY"  # Replace with your SerpAPI key
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI key
    openai.api_key = OPENAI_API_KEY
    ```

1.  然后，我们初始化搜索引擎和句子转换器。以下代码定义了搜索函数，使用 SerpApi 执行 Google 搜索，并初始化句子转换器模型（`all-mpnet-base-v2`）以创建句子嵌入：

    ```py
        def search(query):
        params = {
            "q": query,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com",
            "api_key": SERPAPI_KEY,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    model = SentenceTransformer('all-mpnet-base-v2')
    ```

1.  接下来，我们检索相关片段。我们定义了`retrieve_snippets`函数，它接受搜索结果，提取片段，计算它们的嵌入，并计算查询嵌入与每个片段嵌入之间的余弦相似度。然后，它返回与查询最相似的顶部*k*个片段：

    ```py
       def retrieve_snippets(query, results, top_k=3):
        snippets = [
            result.get("snippet", "")
            for result in results.get("organic_results", [])
        ]
        if not snippets:
            return []
        query_embedding = model.encode(query,
            convert_to_tensor=True)
        snippet_embeddings = model.encode(snippets,
            convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(
            query_embedding, snippet_embeddings
        )[0]
        top_results = torch.topk(cosine_scores, k=top_k)
        return [snippets[i] for i in top_results.indices]
    ```

1.  然后，我们定义了`generate_answer`函数，用于使用 GPT-4o 生成答案。这是我们的 RAG 系统生成部分的核心：

    ```py
        def generate_answer(query, context):
        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable expert. Answer the user's query based only on the information provided in the context. "
                           "If the answer is not in the context, say 'I couldn't find an answer to your question in the provided context.'",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuery: {query}",
            },
        ]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=256
        )
        return response.choices[0].message.content
    ```

    此函数构建一个结构化的提示，用于 LLM 生成一个严格受给定上下文约束的答案。它将对话格式化为系统-用户消息对，指示模型充当主题专家，并将答案限制在提供的信息内，明确避免推测。如果信息不存在，系统将指导返回一个回退消息，表明答案无法找到。查询和上下文直接嵌入到用户消息中，LLM（在本例中为 `gpt-4o`）通过 `temperature=0.7` 和 `256` 令牌的响应长度上限进行适度创造性的查询。这种设计使得该函数在基于上下文的问答任务中可靠，尤其是在 RAG 管道或文档问答或合规工具等约束回答环境中。

1.  这里是主要的 RAG 函数及其示例用法：

    ```py
       def rag_system(query):
        search_results = search(query)
        relevant_snippets = retrieve_snippets(query, search_results)
        if not relevant_snippets:
            return "Could not find any information related to your query"
        context = " ".join(relevant_snippets)
        answer = generate_answer(query, context)
        return answer
    # Example usage
    query = "What are the latest advancements in quantum computing?"
    answer = rag_system(query)
    print(answer)
    ```

    此代码定义了 `rag_system` 函数，该函数协调整个流程：搜索、检索片段和生成答案。然后它演示了如何使用示例查询 `rag_system`，并将生成的答案打印到控制台。

    `rag_system` 函数通过首先使用 `search(query)` 搜索相关信息，然后通过调用 `retrieve_snippets(query, search_results)` API 提取相关片段来回答查询。如果没有找到片段，它将返回一条消息，表明没有找到信息。如果可用片段，它们将被组合成一个单一上下文字符串，并通过 `generate_answer(query, context)` 生成答案。最后，该函数根据上下文返回生成的答案。在示例用法中，该函数使用查询 `"What are the latest advancements in quantum computing?"` 被调用，并将根据相关搜索结果返回一个生成的响应。在实际生产系统中，我们应该在 `retrieve_snippets` API 调用周围实现重试和错误处理。

在我们进入下一节之前，这里有一些事情需要记住：

+   **API 密钥**：确保你有有效的 SerpApi 和 OpenAI API 密钥，并在代码中替换了占位符。

+   **OpenAI 成本**：注意 OpenAI API 使用成本。GPT-4o 可能比其他模型更昂贵。

+   **提示工程**：生成的答案质量很大程度上取决于提供给 GPT-4o 的提示。你可能需要尝试不同的提示以获得最佳结果。考虑添加有关所需答案格式、长度或样式的说明。

+   使用 `try-except` 块来处理潜在问题，例如网络问题、API 错误或无效输入。

+   **高级技术**：这是一个基本的 RAG 系统。你可以通过以下方式进一步改进它：

    +   **更好的片段选择**：考虑因素包括来源多样性、事实性和片段长度

    +   **迭代检索**：如果初始答案不满意，则检索更多上下文

    +   **微调**：在您的特定领域上对较小的、更专业的语言模型进行微调，以实现更好的性能和更低的成本

我们已经成功构建了一个简单的 RAG 系统，涵盖了检索和生成的核心组件。现在我们有了功能性的 RAG 系统，让我们更深入地探讨那些能够从大量数据集中高效检索的关键技术：嵌入和索引。我们将探讨不同的方法来表示文本的语义，并组织这些表示以实现快速相似性搜索。

# LLM 应用中的检索嵌入和索引

嵌入和索引技术为基于 RAG 的 LLM 应用提供了高效和有效的检索。它们允许 LLM 快速找到并利用大量数据中的相关信息。以下小节提供了常见技术的分解。

## 嵌入

嵌入是数据的数值向量表示，例如文本、图像或音频，它将复杂的高维数据映射到一个连续的向量空间中，其中相似的项目彼此靠近。这些向量捕捉数据的潜在模式、关系和语义属性，使得机器学习模型更容易理解和处理。例如，对于文本，词嵌入将单词或短语转换成密集的向量，以反映它们在向量空间中的语义关系，如同义词彼此更接近。嵌入通常通过神经网络等技术从大型数据集中学习，它们是信息检索、分类、聚类和推荐系统等任务的基础。通过降低数据的维度同时保留重要特征，嵌入使得模型能够更好地泛化并有效地处理各种输入数据。

对于 LLM，文本嵌入最为相关。它们是通过将文本传递到神经网络（如我们在上一节中使用的 Sentence Transformer 模型）来生成的。

### 我们为什么需要嵌入？

嵌入对于 RAG 应用的重要性如下：

+   **语义搜索**：嵌入使得语义搜索成为可能，您可以根据意义而不是仅仅根据关键词匹配来查找信息

+   **上下文理解**：LLM 可以使用嵌入来理解不同信息片段之间的关系，提高其推理和生成相关响应的能力

+   **高效检索**：当与适当的索引结合时，嵌入允许从大型数据集中快速检索相关信息

### 常见的嵌入技术

在 RAG 系统中，常用的嵌入技术多种多样，它们在底层模型、方法和适用于不同应用方面有所不同。以下是 RAG 中一些突出的嵌入技术：

+   **预训练的基于 transformer 的嵌入（例如，BERT、RoBERTa 和 T5）**：像**双向编码器表示（BERT）**及其变体（如 RoBERTa 和 T5）这样的 Transformer 模型已被广泛用于生成文本的密集、上下文嵌入。这些模型在大语料库上进行微调，并捕捉到丰富的语言语义理解。在 RAG 系统中，这些嵌入可用于根据语义相似性从文档存储中检索相关段落。这些嵌入通常是高维的，并且通过将文本输入到 Transformer 模型中生成固定大小的向量。

+   **Sentence-BERT（SBERT）**：SBERT 是 BERT 的一种变体，专为句子级嵌入设计，它专注于优化模型以执行诸如语义文本相似性和聚类等任务。它使用 Siamese 网络架构将句子映射到密集向量空间，其中语义相似的句子彼此更接近。这使得它在 RAG 中的信息检索任务中特别有效，在这些任务中，从大型语料库中检索与语义相关的段落至关重要。

+   **Facebook AI 相似性搜索（Faiss）**：Faiss 是由 Facebook AI Research 开发的库，它通过**近似最近邻（ANN）搜索**提供高效的相似性搜索。Faiss 本身不是一种嵌入技术，但它与各种嵌入模型协同工作，以索引和搜索大量向量集合。在 RAG 中使用时，Faiss 通过比较其嵌入与查询嵌入之间的相似性，实现了快速检索相关文档或段落。

+   **密集检索模型（例如，DPR 和 ColBERT）**：**密集段落检索（DPR）**是一种信息检索方法，它使用两个独立的编码器（通常是基于 BERT 的模型）将查询和段落编码为密集向量。DPR 通过利用密集嵌入中编码的上下文知识，优于传统的稀疏检索方法。另一方面，ColBERT 是另一种平衡密集检索效率和传统方法有效性的密集检索模型。这些模型在 RAG 中检索与查询语义相关的优质段落时特别有用。

+   **对比语言-图像预训练（CLIP）**：虽然最初是为多模态应用（文本和图像）设计的，但 CLIP 也被适应用于仅文本的任务。它通过在共享向量空间中对齐文本和图像数据来学习嵌入。尽管 CLIP 主要用于多模态任务，但它在图像空间中共同表示语言的能力提供了一个灵活的嵌入框架，该框架可用于 RAG，尤其是在处理多模态数据时。

+   **深度语义相似性模型（例如，USE 和 InferSent）**：如**通用句子编码器（USE**）和 InferSent 之类的模型通过捕捉更深层的语义意义来生成句子嵌入，可用于各种 NLP 任务，包括文档检索。这些模型产生固定大小的向量表示，可以用于比较相似性，当与检索系统结合使用时，它们对 RAG 非常有用。

+   **Doc2Vec**：Word2Vec 的扩展，Doc2Vec 为整个文档生成嵌入，而不是单个单词。它将可变长度的文本映射到固定大小的向量，可用于检索语义相似的文档或段落。虽然在语义丰富性方面不如基于 transformer 的模型强大，但 Doc2Vec 仍然是 RAG 应用中用于更轻量级检索任务的有效工具。

+   **基于嵌入的搜索引擎（例如，使用密集向量的 Elasticsearch）**：一些现代搜索引擎，如 Elasticsearch，已经集成了对密集向量的支持，同时保留了传统的基于关键词的索引。Elasticsearch 可以存储和检索文本嵌入，允许进行更灵活和语义感知的搜索。当与 RAG 结合使用时，这些嵌入可以用于根据查询的相关性对文档进行排序，从而提高检索性能。

+   **OpenAI 嵌入（例如，基于 GPT 的模型）**：OpenAI 的嵌入，如 GPT-3 模型，也用于 RAG 任务。这些嵌入基于语言模型生成高质量文本表示的能力，可以在大型语料库中进行索引和搜索。虽然它们在检索方面的特定调整不如某些其他模型（如 DPR），但它们非常灵活，可以用于通用 RAG 应用。

这些嵌入技术根据 RAG 系统的具体需求提供各种优势，例如检索速度、模型准确性和处理数据的规模。每个技术都可以针对特定用例进行微调和优化，嵌入技术的选择将取决于诸如检索文档的性质、计算资源和延迟要求等因素。

## 索引

索引是将嵌入组织到一种数据结构中的过程，该数据结构允许进行快速相似性搜索。想象一下书的索引，但针对的是向量而不是单词。

使用更详细的 LLM 术语描述，向量索引技术通过创建专门的数据结构来优化嵌入的存储和检索，这些数据结构根据其相似性关系组织高维向量，而不是按顺序排列。这些结构——无论是基于图（通过可导航路径连接相似向量）、基于树（递归划分向量空间）还是基于量化（压缩向量同时保留相似性）——都服务于将原本代价高昂的全面搜索转化为可管理过程的根本目的，通过策略性地限制搜索空间，使向量数据库能够以亚秒级的查询时间处理数十亿个嵌入，同时保持速度、内存效率和结果精度之间的可接受权衡。

### 为什么索引很重要？

索引对于 LLM 之所以重要，有以下原因：

+   **速度**：没有索引，您必须将查询嵌入与数据集中的每个嵌入进行比较，这计算成本高且速度慢。

+   **可扩展性**：索引允许 LLM 应用扩展以处理包含数百万甚至数十亿数据点的庞大数据集。

### 常见的索引技术

让我们来看看 LLM 的一些常见索引技术。

对于这些索引技术的可视化图表，我建议您查看以下网站：[`kdb.ai/learning-hub/articles/indexing-basics/`](https://kdb.ai/learning-hub/articles/indexing-basics/)

+   **平面索引（暴力法）**：

    +   **工作原理**：将所有嵌入存储在简单的列表或数组中。在搜索过程中，它计算查询嵌入与索引中每个嵌入之间的距离（例如，余弦相似度）。

    +   **优点**：简单易实现且完美精度（找到真正的最近邻）。

    +   **缺点**：对于大数据集来说，速度慢且计算成本高，因为它需要进行全面搜索。

    +   **适用场景**：非常小的数据集或当完美精度是绝对要求时。

+   **倒排文件索引（IVF）**：

    +   **工作原理**：

        +   **聚类**：使用如 k-means 等算法将嵌入空间划分为簇。

        +   **倒排索引**：创建一个倒排索引，将每个簇中心映射到属于该簇的嵌入列表。

        +   **搜索**：

            1.  找到与查询嵌入最近的簇中心。

            1.  只在那些簇内进行搜索，显著减少了搜索空间。

    +   **优点**：比平面索引快；相对简单易实现。

    +   **缺点**：近似（可能无法始终找到真正的最近邻）；精度取决于簇的数量。

    +   **适用场景**：需要速度和精度之间良好平衡的中等大小数据集。

+   **分层可导航小世界（HNSW）**：

    +   **工作原理**：

        +   **基于图**：构建一个分层图，其中每个节点代表一个嵌入。

        +   **层**：图有多个层，顶层有长距离连接（用于更快地遍历），底层有短距离连接（用于准确搜索）。

        +   **搜索**：从顶层随机节点开始，通过探索连接贪婪地向查询嵌入移动。搜索在层中向下进行，逐步细化结果。

    +   **优点**：非常快且准确；通常被认为是近似最近邻搜索的当前最佳水平

    +   **缺点**：比 IVF 实现更复杂，并且由于图结构，内存开销更大

    +   **适用范围**：对于速度和准确度都至关重要的大型数据集

+   **产品量化 (PQ)**：

    +   **工作原理**：

        +   **子向量**：将每个嵌入分割成多个子向量。

        +   **码本**：使用聚类为每个子向量创建单独的码本。每个码本包含一组代表性的子向量（中心点）。

        +   **编码**：通过用对应码本中最接近的中心点替换其子向量来编码每个嵌入。这创建了一个嵌入的压缩表示。

        +   **搜索**：通过使用查询子向量和码本中心之间的预计算距离，计算查询与编码嵌入之间的近似距离。

    +   **优点**：通过压缩嵌入显著减少内存使用；快速搜索。

    +   **缺点**：近似，准确度取决于子向量的数量和码本的大小。

    +   **适用范围**：非常适合内存效率是主要关注点的非常大的数据集。

+   **局部敏感哈希 (LSH)**：

    +   **工作原理**：使用哈希函数以高概率将相似的嵌入映射到同一个“桶”中

    +   **优点**：相对简单；可以跨多台机器分布式部署

    +   **缺点**：近似，性能取决于哈希函数的选择和桶的数量

    +   **适用范围**：非常适合非常大的、高维数据集

现在我们已经介绍了不同的索引方法，让我们介绍一些流行的库和工具，它们实现了这些索引技术，使得在实际中使用它们变得更加容易。这将提供一个实际的角度，了解如何在你的 RAG 应用中利用这些技术。

以下是一些实现索引的库和工具：

+   **Faiss**：Facebook AI 开发的高度优化的库，用于高效地搜索和聚类密集向量。它实现了之前提到的许多索引技术（平面、IVF、HNSW 和 PQ）。

+   **近似最近邻 Oh Yeah (Annoy)**：另一个流行的近似最近邻搜索库，以其易用性和良好的性能而闻名。它采用基于树的方法。

+   **Scalable Nearest Neighbors (ScaNN)**：由谷歌开发的库，旨在用于大规模、高维数据集。

+   **Vespa.ai**：提供查询、组织和在向量、张量、文本和结构化数据中进行推理的工具。它被[`www.perplexity.ai/`](https://www.perplexity.ai/)使用。

+   **Pinecone, Weaviate, Milvus, Qdrant**：专门设计用于存储和搜索嵌入的向量数据库。它们处理索引、扩展和其他基础设施问题。

适用于你的 LLM 应用的最佳嵌入和索引技术将取决于几个因素：

+   **数据集大小**：对于小型数据集，平面索引可能足够。对于大型数据集，考虑 HNSW、IVF 或 PQ。

+   **速度要求**：如果低延迟至关重要，HNSW 通常是速度最快的选项。

+   **精度要求**：如果需要完美的精度，平面索引是唯一的选择，但它不可扩展。HNSW 在近似方法中通常提供最佳的精度。

+   **内存限制**：如果内存有限，PQ 可以显著减少存储需求。

+   **开发工作量**：Faiss 和 Annoy 在性能和易于实现之间提供了良好的平衡。向量数据库简化了基础设施管理。

通过仔细考虑这些因素，并理解每种技术和库的优缺点，你可以选择最合适的嵌入和索引方法来构建高效且有效的 LLM 应用。

我们现在将演示一个使用 Faiss 的嵌入、索引和搜索示例，Faiss 是一个用于高效相似性搜索的强大库。我将使用`all-mpnet-base-v2` Sentence Transformer 模型来生成嵌入。由于代码将超过 20 行，我将将其分解为带有每个块前解释的代码块。

## 示例代码演示嵌入、索引和搜索

在本节中，我们将展示使用嵌入和索引实现快速文本文档集合中相似性搜索的典型工作流程代码：以下是它的功能：

1.  **加载 Sentence Transformer 模型**：初始化用于生成句子嵌入的预训练模型。

1.  **创建样本数据**：定义一个示例句子列表（你将用实际数据替换这部分）。

1.  使用`SentenceTransformer`为每个句子创建嵌入。

1.  在此示例中，使用`IndexFlatL2`为平面 L2 距离索引存储嵌入。

1.  **将嵌入添加到索引中**：将生成的嵌入添加到 Faiss 索引中。

1.  **定义搜索查询**：设置一个我们想要找到相似句子的示例查询。

1.  **编码查询**：使用相同的 Sentence Transformer 模型为搜索查询创建嵌入。

1.  **执行搜索**：使用 Faiss 索引搜索与查询嵌入最相似的*k*个嵌入。

1.  **打印结果**：显示在索引中找到的最近的*k*个邻居的索引和距离。

在我们查看代码之前，让我们安装以下依赖项：

```py
pip install faiss-cpu sentence-transformers
# Use faiss-gpu if you have a compatible GPU
```

让我们看看代码示例：

1.  首先，我们导入必要的库——`sentence_transformers`用于创建嵌入，`faiss`用于索引和搜索——并加载`all-mpnet-base-v2` Sentence Transformer 模型：

    ```py
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-mpnet-base-v2')
    ```

1.  然后，我们通过定义一些示例句子（你可以用你的实际数据替换这些句子）来准备数据，然后使用 Sentence Transformer 模型为每个句子生成嵌入（嵌入被转换为 float32，这是 Faiss 所需要的）：

    ```py
    # Sample sentences
    text_data = [
        "A man is walking his dog in the park.",
        "Children are playing with toys indoors.",
        "An artist is painting a landscape on canvas.",
        "The sun sets behind the mountain ridge.",
        "Birds are singing outside the window."
    ]
    # Generate vector representations using a SentenceTransformer model
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your model if different
    vectors = model.encode(text_data, convert_to_tensor=True)
    # Ensure compatibility with Faiss by converting to 32-bit floating point and moving to CPU
    vectors = vectors.detach().cpu().numpy().astype(np.float32)
    ```

1.  然后，我们创建一个 Faiss 索引并将嵌入添加到其中：

    ```py
    # Get the dimensionality of the embeddings
    dimension = embeddings.shape[1]
    # Create a Faiss index (flat L2 distance)
    index = faiss.IndexFlatL2(dimension)
    # Add the embeddings to the index
    index.add(embeddings)
    ```

    这里，我们使用 IndexFlatL2，这是一个使用 L2 距离（欧几里得距离）进行相似度比较的平面索引。这种类型的索引可以提供准确的结果，但可能对于非常大的数据集来说比较慢。索引是根据正确的维度性创建的（对于这个 Sentence Transformer 模型是 `768`）。

1.  接下来，我们定义一个示例搜索查询，并使用相同的 Sentence Transformer 模型将其编码成嵌入。查询嵌入也被转换为 float32：

    ```py
    # Define a search query
    query = "What is the dog doing?"
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = \
        query_embedding.cpu().numpy().astype('float32')
    ```

1.  最后，我们使用 `index.search()` 方法执行相似度搜索。我们搜索两个最相似的句子（*k*=2）。该方法返回最近邻的距离和索引。然后我们打印出找到的最近邻的索引和距离：

    ```py
    # Search for the k nearest neighbors
    k = 2
    distances, indices = index.search(query_embedding, k)
    # Print the results
    print("Nearest neighbors:")
    for i, idx in enumerate(indices[0]):
        print(f"  Index: {idx}, Distance: {distances[0][i]},
        Sentence: {sentences[idx]}")
    ```

以下是从运行前面的代码块可能得到的示例输出：

```py
Nearest neighbors:
  Index: 0, Distance: 0.634912312, Sentence: A man is walking his dog in the park.
  Index: 1, Distance: 1.237844944, Sentence: Children are playing with toys indoors.
```

这展示了如何使用 Sentence Transformers 和 Faiss 进行语义相似度搜索。请注意，实际数字将根据硬件、模型版本和运行条件而变化。

下面是发生的事情。

查询 `"What is the dog doing?"` 被嵌入并与列表中的所有嵌入句子进行比较。Faiss 根据嵌入空间中的欧几里得（L2）距离检索出两个最语义相似的句子。最小的距离表示最高的相似度。在这个例子中，关于男人遛狗的句子与查询在语义上最接近，这是有意义的。

如果你在自己的机器上运行这个程序，由于模型初始化的非确定性和浮点精度，你的值可能会有所不同，但最接近的句子应该始终是与查询最语义相关的句子。

重要

使用 Faiss 的 `IndexIVFFlat` 或 `IndexHNSWFlat` 可以提高搜索速度。

使用 `faiss-gpu` 可以显著加快索引和搜索的速度。

**数据预处理**: 对于实际应用，你可能需要执行额外的数据预处理步骤，例如小写化、去除标点符号或词干提取/词形还原，具体取决于你的具体需求和数据性质。

**距离度量**: Faiss 支持不同的距离度量。这里我们使用了 L2 距离，但你也可以使用内积（IndexFlatIP）或其他度量，具体取决于你的嵌入是如何生成的以及你想要测量哪种相似度。

**向量数据库**：对于生产级系统，考虑使用专门的向量数据库，如 Pinecone、Weaviate 或 Milvus，以更有效地管理您的嵌入和索引。它们通常提供自动索引、扩展和数据管理等功能，这些功能简化了相似性搜索应用的部署。

我们已经涵盖了使用 Faiss 进行嵌入、索引和搜索的基础知识，以及现实世界实施的重要考虑因素。现在，让我们将注意力转向 RAG 的另一个关键方面：查询公式。我们将探讨各种策略来精炼和扩展用户查询，最终导致从知识库中检索更有效的信息。

# 基于 LLM 的 RAG 查询公式策略

基于 LLM 的 RAG 系统中的查询公式策略旨在通过提高用户查询的表达性和覆盖范围来增强检索。常见的扩展策略包括以下内容：

+   **同义词和释义扩展**：这涉及到使用 LLM 或词汇资源生成语义等效的替代方案。例如，将“气候变化影响”扩展到包括“全球变暖的影响”或“气候变化的环境后果”可以帮助匹配更广泛的文档。

+   **上下文重构**：LLM 可以通过推断对话或文档上下文中的意图来重新解释查询。这有助于调整查询以更好地与知识库中可能表达的信息相匹配。

+   **伪相关性反馈**：也称为盲相关性反馈，这种策略涉及运行初始查询，分析排名靠前的文档中的显著术语，并使用这些术语来扩展查询。虽然有效，但需要防止主题漂移。

+   **基于模板的增强**：在结构化领域中很有用，这种方法使用特定领域的模板或模式来系统地生成变体。例如，关于“高血压治疗”的医疗查询也可能包括“高血压疗法”或“管理高血压”。

+   **实体和概念链接**：在查询中识别命名实体和领域概念，并用它们的别名、定义或层次关系替换或增强。这通常由本体或知识图指导。

+   **基于提示的查询重写**：使用 LLM，可以精心制作提示来明确指示模型生成重构后的查询。这在多语言或多领域 RAG 系统中特别有用，其中查询需要适应目标语料库的风格和词汇。

每种策略都对召回率和精确度有不同的贡献。选择或组合它们取决于底层知识库的结构和可变性。

在以下代码中，`QueryExpansionRAG`实现使用了一个由预训练的序列到序列语言模型（具体来说，是 T5-small）驱动的基于提示的查询重写策略。这种方法指示模型通过在提示前加上`"expand query:"`来生成输入查询的替代表述。生成的扩展反映了释义性改写，其中模型综合了语义相关的变体以增加检索覆盖范围：

```py
from transformers import pipeline
class QueryExpansionRAG(AdvancedRAG):
    def __init__(
        self, model_name, knowledge_base,
        query_expansion_model="t5-small"
    ):
        super().__init__(model_name, knowledge_base)
        self.query_expander = pipeline(
            "text2text-generation", model=query_expansion_model
        )
    def expand_query(self, query):
        expanded = self.query_expander(
            f"expand query: {query}", max_length=50,
            num_return_sequences=3
        )
        return [query] + [e['generated_text'] for e in expanded]
    def retrieve(self, query, k=5):
        expanded_queries = self.expand_query(query)
        all_retrieved = []
        for q in expanded_queries:
            all_retrieved.extend(super().retrieve(q, k))
        # Remove duplicates and return top k
        unique_retrieved = list(dict.fromkeys(all_retrieved))
        return unique_retrieved[:k]
# Example usage
rag_system = QueryExpansionRAG(model_name, knowledge_base)
retrieved_docs = rag_system.retrieve(query)
print("Retrieved documents:", retrieved_docs)
```

此代码定义了一个`QueryExpansionRAG`类，它通过结合使用预训练的 T5 模型来扩展 RAG 框架，实现了查询扩展。当用户提交查询时，`expand_query`方法通过文本到文本生成管道使用 T5 模型，生成查询的多个替代表述，然后将这些表述与原始查询结合。`retrieve`方法遍历这些扩展查询，为每个查询检索文档，并聚合结果同时去除重复项。这种方法通过扩大原始查询的词汇和语义范围，增加了检索相关内容的机会，当知识库以多种方式表达信息时，这种方法尤其有效。

请记住，扩展不当的查询可能会引入噪声并降低检索精确率。在此实现中，T5 模型生成的扩展与原始查询结合，增加了覆盖范围。然而，为了保持平衡，考虑使用相似度分数重新排序结果，或者在检索期间为生成的扩展分配较低的权重。这有助于确保扩展提高了召回率，同时不会损害与原始意图的对齐。

我们已经看到了查询扩展如何增强 RAG 系统中的检索，但管理召回率和精确率之间的权衡是至关重要的。现在，让我们将我们的重点转向 RAG 管道的另一端：将检索到的信息与 LLM 整合以生成最终答案。我们将探讨如何构建能够有效利用检索到的上下文的提示。

# 将检索到的信息与 LLM 生成整合

要将检索到的信息与 LLM 生成整合，我们可以创建一个包含检索到的文档的提示：

```py
from transformers import AutoModelForCausalLM
class GenerativeRAG(QueryExpansionRAG):
    def __init__(
        self, retriever_model, generator_model, knowledge_base
    ):
        super().__init__(retriever_model, knowledge_base)
        self.generator = \
            AutoModelForCausalLM.from_pretrained(generator_model)
        self.generator_tokenizer = \
            AutoTokenizer.from_pretrained(generator_model)
    def generate_response(self, query, max_length=100):
        retrieved_docs = self.retrieve(query)
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        inputs = self.generator_tokenizer(prompt, return_tensors="pt")
        outputs = self.generator.generate(inputs,
            max_length=max_length)
        return self.generator_tokenizer.decode(
            outputs[0], skip_special_tokens=True)
# Example usage
retriever_model = "all-MiniLM-L6-v2"
generator_model = "gpt2-medium"
rag_system = GenerativeRAG(
    retriever_model, generator_model, knowledge_base
)
response = rag_system.generate_response(query)
print("Generated response:", response)
```

在前面的代码片段中，`GenerativeRAG`类通过集成用于答案生成的因果语言模型扩展了 RAG 管道。它继承自`QueryExpansionRAG`，后者已经提供了检索功能，并添加了一个使用 Hugging Face 的`AutoModelForCausalLM`的生成器组件。在构造函数中，它根据给定的模型名称初始化生成器模型和分词器。`generate_response`方法首先检索给定查询的相关文档，将它们连接成一个单一上下文字符串，并构建一个将此上下文与问题结合的提示。然后，该提示被分词并传递给语言模型，该模型生成作为答案的文本续写。最终输出是通过将生成的标记解码成字符串获得的。这种模块化结构将检索和生成步骤分开，使得根据任务或模型性能要求轻松扩展或替换单个组件。

在介绍了 RAG 系统的基本知识后，我们将现在关注现实世界的挑战，例如可扩展性、动态更新和多语言检索。具体来说，我们将讨论如何通过分片索引架构提高大规模检索效率，突出其在数据密集型环境中的性能影响。

# LLMs 在 RAG 中的挑战和机遇

RAG 中的一些关键挑战和机遇包括以下内容：

+   **可扩展性**: 高效地处理非常大的知识库。

+   **动态知识更新**: 保持知识库的时效性。

+   **跨语言 RAG**: 在多种语言中进行检索和生成。

+   **多模态 RAG**: 在检索和生成过程中结合非文本信息。

    请记住，跨语言和多模态 RAG 需要专门的检索管道或适配器，因为标准的检索方法通常在跨语言或模态的语义匹配上遇到困难，需要专门的组件来正确编码、对齐和检索相关信息，无论源语言或格式如何，同时保持上下文理解和相关性。

+   **可解释 RAG**: 在检索和生成过程中提供透明度。

为了使本章内容不过于冗长，在本节中，我们将仅展示一个示例，说明如何通过实现分片索引来解决可扩展性挑战。分片索引指的是将索引分割成多个较小的、可管理的片段，称为分片，每个分片独立存储和维护在不同的节点或存储单元上。这种方法可以实现并行处理，减少查找时间，并缓解与集中式索引相关的瓶颈，使其适用于处理在 AI 应用中常见的大规模数据集或高查询量：

```py
class ShardedRAG(GenerativeRAG):
    def __init__(
        self, retriever_model, generator_model,
        knowledge_base, num_shards=5
    ):
        super().__init__(retriever_model, generator_model,
            knowledge_base)
        self.num_shards = num_shards
        self.sharded_indexes = self.build_sharded_index()
    def build_sharded_index(self):
        embeddings = self.get_embeddings(self.knowledge_base)
        sharded_indexes = []
        shard_size = len(embeddings) // self.num_shards
        for i in range(self.num_shards):
            start = i * shard_size
            end = start + shard_size if i < self.num_shards - 1
                else len(embeddings)
            shard_index = faiss.IndexFlatL2(embeddings.shape[1])
            shard_index.add(embeddings[start:end])
            sharded_indexes.append(shard_index)
        return sharded_indexes
    def retrieve(self, query, k=5):
        query_embedding = self.get_embeddings([query])[0]
        all_retrieved = []
        for shard_index in self.sharded_indexes:
            _, indices = shard_index.search(
                np.array([query_embedding]), k)
            all_retrieved.extend([self.knowledge_base[i]
                for i in indices[0]])
        # Remove duplicates and return top k
        unique_retrieved = list(dict.fromkeys(all_retrieved))
        return unique_retrieved[:k]
# Example usage
sharded_rag = ShardedRAG(retriever_model, generator_model,
    knowledge_base)
response = sharded_rag.generate_response(query)
print("Generated response:", response)
```

在前面的代码中，可扩展性是通过将知识库划分为多个较小的索引或分片来处理的，每个分片包含整体数据的一部分。这种方法减轻了单个索引的计算和内存负担，并允许检索操作在数据集增长的情况下保持高效。在查询过程中，系统将查询嵌入一次，独立地对所有分片进行搜索，然后合并结果。这种设计避免了搜索单个大型索引时可能出现的瓶颈，并使得扩展到更大的知识库成为可能。它还为进一步的优化奠定了基础，例如并行化分片查询或将它们分布到多台机器上。

# 摘要

RAG 是一种强大的技术，用于通过外部知识增强大型语言模型（LLMs）。通过实施本章讨论的策略和技术，你可以创建更明智、更准确的语言模型，这些模型能够访问和利用大量信息。

随着我们继续前进，下一章将探讨基于图的大型语言模型（LLMs）的 RAG，这扩展了 RAG 概念以利用结构化知识表示。这将进一步增强 LLMs 在复杂关系上进行推理和生成更符合上下文响应的能力。
