

# 第六章：使用 Neo4j 探索高级知识图谱功能

通过在前一章建立的基础知识，其中我们介绍了基本的搜索功能，我们现在将探索更复杂的知识探索、图推理和性能优化技术。在本章中，我们将利用 Neo4j 的高级功能，重点关注将这些功能与 Haystack 集成，以创建一个更智能、AI 驱动的搜索系统。

到本章结束时，您将能够从您的知识图谱中解锁更深入的见解，利用高级搜索功能，并确保您的 AI 驱动的搜索系统既高效又可持续。

在本章中，我们将涵盖以下主要主题：

+   探索高级 Haystack 功能以进行知识探索

+   使用 Haystack 进行图推理

+   扩展您的 Haystack 和 Neo4j 集成

+   维护和监控您的 AI 驱动的搜索系统的最佳实践

# 技术要求

在深入本章内容之前，请确保您的开发环境已设置好必要的技术和工具。此外，您的 Neo4j 实例应加载了来自 `Ch4` 的数据和来自 `Ch5` 的嵌入。以下是本章的技术要求：

+   **Neo4j (v5.x 或更高版本)**: 您需要在您的本地机器或服务器上安装并运行 Neo4j。您可以从 [`neo4j.com/download/`](https://neo4j.com/download/) 下载它。

+   **Haystack (v1.x)**: 我们将使用 Haystack 框架来集成 AI 驱动的搜索功能。请确保按照 [`docs.haystack.deepset.ai/docs/installation`](https://docs.haystack.deepset.ai/docs/installation) 中的说明安装 Haystack。

+   **Python (v3.8 或更高版本)**: 确保您已安装 Python。您可以从 [`www.python.org/downloads/`](https://www.python.org/downloads/) 下载它。

+   **OpenAI API 密钥**: 要成功使用基于 GPT 的模型生成嵌入，您需要一个 OpenAI API 密钥：

    +   如果您还没有，请在 OpenAI ([`platform.openai.com/signup`](https://platform.openai.com/signup)) 上注册账户以获取 API 密钥

**注意**

免费层 API 密钥在本项目的多数用例中都不会工作。您需要一个有效的付费 OpenAI 订阅来访问必要的端点和使用限制。

+   登录后，导航到您的 OpenAI 控制台中的 API 密钥部分 ([`platform.openai.com/api-keys`](https://platform.openai.com/api-keys)) 并生成一个新的 API 密钥

如果您已经遵循了前几章的设置，您可以跳过这些要求，因为它们已经安装好了。

本章的所有代码都可在以下 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/tree/main/ch6`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/tree/main/ch6)。

此文件夹包含实现 Neo4j 和 Haystack 集成以及高级知识图谱功能所需的所有必要脚本、文件和配置。

确保克隆或下载存储库，以便您可以在本章中跟随代码示例。

# 探索高级 Haystack 功能以进行知识探索

在本节中，我们将深入探讨使用 Haystack 的更高级搜索功能。您在*第五章*中集成了嵌入到 Neo4j 图中。现在是时候探索如何超越基本的相似度匹配来增强搜索了。这里的目的是从简单的基于检索的嵌入转向对图中知识的更细致、多层次的探索。

我们将探讨诸如基于上下文的推理和针对特定用例优化搜索功能以提供高度相关和智能结果的技术。

让我们先谈谈基于上下文的推理。

## 基于上下文的搜索

现在，我们将在前一章的*将 Haystack 连接到 Neo4j 进行高级向量搜索*部分的基础上构建基于嵌入的方法，通过将多跳推理集成到 Neo4j 图和 Haystack 的相似度搜索功能中。这种方法允许搜索引擎在利用基于 AI 的高级检索方法的同时遍历节点之间的多个关系。我们不会仅仅根据直接匹配检索节点或文档，而是利用 Haystack 探索相关节点之间的路径，增加层次化的上下文并揭示更深入的见解。这种基于图推理和相似度理解的结合使得搜索结果更加智能和相关性更强。

```py
Inception.
```

**注意**

标题作为`title`变量在主程序中的值传递。

在检索到这些相关电影后，Haystack 用于根据相似查询分析和排名结果，展示了结合基于图的关系和高级相似度检索的多跳搜索：

```py
def fetch_multi_hop_related_movies(title):
    query = """
    MATCH (m:Movie {title: $title})<-[:DIRECTED]-(d:Director)-[:DIRECTED]->(related:Movie)
    RETURN related.title AS related_movie, related.overview AS overview
    """
    with driver.session() as session:
        result = session.run(query, title=title)
        documents = [
            {
                "content": record["overview"],
                "meta": {"title": record["related_movie"]}
            }
            for record in result
        ]
    return documents
def perform_similarity_search_with_multi_hop(query, movie_title):
    # Fetch multi-hop related movies from Neo4j
    multi_hop_docs = fetch_multi_hop_related_movies(movie_title)
    if not multi_hop_docs:
        print(f"No related movies found for {movie_title}")
        return
    # Write these documents to the document store
    document_store.write_documents(multi_hop_docs)
    # Generate embedding for the search query (e.g., "time travel")
    query_embedding = text_embedder.run(query).get("embedding")
    if query_embedding is None:
        print("Query embedding not created successfully.")
        return
    # Perform vector search only on the multi-hop related movies
    similar_docs = document_store.query_by_embedding(
        query_embedding, top_k=3
    )
    if not similar_docs:
        print("No similar documents found.")
        return
    for doc in similar_docs:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = doc.score
        print(
            f"Title: {title}\nOverview: {overview}\n"
            f"Score: {score:.2f}\n{'-'*40}"
        )
    print("\n\n") 
```

然而，由于我们只导入了一小部分原始数据集，它没有一对一的关系。当一个导演指导了多部电影时，搜索可能会产生类似`Inception 没有找到相关电影`的输出。

您可以尝试更新脚本以导入整个数据集（在从 AuraDB Free 升级到 AuraDB Professional 或 AuraDB Business Critical 或在 Neo4j Desktop 版本中）并查看多跳推理是如何执行的。

## 动态搜索查询与灵活的搜索过滤器

知识图谱的一个优势是在搜索查询期间动态应用过滤器。

在以下代码片段中，我们将演示如何将过滤器和约束条件纳入您的 Haystack 查询中，使用户能够根据特定参数（例如时间范围、类别或实体之间的关系）细化搜索结果。这种灵活性对于构建更互动和上下文丰富的搜索系统至关重要：

```py
def perform_filtered_search(query):
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", text_embedder)
    # pipeline.add_component("retriever", retriever)
    pipeline.add_component(
        "retriever", 
        Neo4jEmbeddingRetriever(document_store=document_store)
    )
    pipeline.connect(
        "query_embedder.embedding", "retriever.query_embedding"
    )
    result = pipeline.run(
        data={
            "query_embedder": {"text": query},
            "retriever": {
                "top_k": 5,
                "filters": {
                    "field": "release_date", "operator": ">=", 
                    "value": "1995-11-17"
                },
            },
        }
    )
    # Extracting documents from the retriever results
    documents = result["retriever"]["documents"]
    for doc in documents:
        # Extract title and overview from document metadata
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        # Extract score from the document (not from meta)
        score = getattr(doc, "score", None)
        # Format score if it exists, else show "N/A"
        score_display = f"{score:.2f}" if score is not None else "N/A"
        # Print the title, overview, and score (or N/A for missing score)
        print(
            f"Title: {title}\nOverview: {overview}\n"
            f"Score: {score_display}\n{'-'*40}\n"
        ) 
demonstrates how to apply dynamic filters, such as release_date, to refine search results. By incorporating these filters, you can add constraints on specific fields—for instance, showing only documents from a certain date onward or filtering by specific attributes such as category or rating. This capability allows you to narrow down results to what is most relevant to them, effectively enhancing the search functionality. Using this approach, you can easily extend or modify filters to suit different needs, offering a flexible and powerful way to interact with data in the knowledge graph.
```

## 搜索优化：针对特定用例定制搜索

并非所有搜索系统都是相同的。无论您是在构建推荐引擎还是特定领域的搜索工具，都需要不同的优化。在本节中，我们将探讨如何根据您的独特用例定制 Haystack 的搜索配置，确保针对您特定数据的最优性能和相关性。我们还将讨论调整模型和索引以适应高规模环境的重要性。

请查看以下代码块：

```py
def perform_optimized_search(query, top_k):
       optimized_results = document_store.query_by_embedding(
            query_embedding=text_embedder.run(query).get("embedding"), 
            top_k=top_k
        )
    for doc in optimized_results:
        title = doc.meta["title"]
        overview = doc.meta.get("overview", "N/A")
        print(f"Title: {title}\nOverview: {overview}\n{'-'*40}") 
```

此代码展示了如何调整参数，如`top_k`，以微调搜索查询返回的前 N 个结果的数量——而不是模型本身。`top_k`参数决定了基于向量相似度检索多少个前 N 个结果。

**注意**

这些只是代码片段。完整版本可在 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch6/beyond_basic_search.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch6/beyond_basic_search.py)。

利用 Haystack 的相似度检索能力（如上下文感知搜索方法和动态过滤），您现在可以创建更精确的 AI 驱动搜索系统，并实现更好的搜索优化。然而，*搜索*只是开始。

在下一节中，我们将利用 Haystack 的推理能力和 Neo4j 知识图中的关系，将推理扩展到基于图的方法。

# 基于 Haystack 的图推理

在本节中，我们将探讨如何通过将 Haystack 与 Neo4j 强大的图推理功能集成，扩展 Haystack 的能力，使其超越基本搜索。虽然传统搜索方法基于文本相似度检索结果，但图推理允许您通过利用知识图中实体之间丰富的关联来揭示更深入的洞察。通过结合 Haystack 的相似度理解和 Neo4j 中的结构化数据，您可以执行更复杂的查询，遍历多个连接，揭示隐藏的模式，并解锁上下文丰富的洞察。

本节将指导您构建这些高级推理能力，将您的搜索系统转变为一个智能的、知识驱动的工具。

## 通过遍历多个关系来揭示隐藏的洞察

当前的图遍历有助于发现实体之间的联系，但跨多个关系和不同类型的关系遍历可以揭示你的知识图谱中的隐藏模式。通过在 Neo4j 中跨越各种路径——无论是电影、演员、导演还是类型之间——你可以生成超越直接关系的更深入见解。这种多步遍历允许你以基本搜索无法实现的方式探索数据，揭示可能被忽视的联系。

我们现在将探讨如何使用多种关系类型和多跳查询来检索更复杂的结果。然后我们将结合 Haystack 的相似度搜索功能进行精炼和排序。

这里有一个例子；你想找到既有与《侏罗纪公园》相同的演员又有相同导演的电影，这样你不仅可以发现直接合作，还可以发现间接联系：

```py
def fetch_multi_hop_related_movies(title):
    query = """
    MATCH (m:Movie {title: $title})<-[:ACTED_IN|DIRECTED]-(p)-
        [:ACTED_IN|DIRECTED]->(related:Movie)
    WITH related.title AS related_movie, p.name AS person,
         CASE
            WHEN (p)-[:ACTED_IN]->(m) AND (p)-[:ACTED_IN]->(related) THEN 'Actor'
            WHEN (p)-[:DIRECTED]->(m) AND (p)-[:DIRECTED]->(related) THEN 'Director'
            ELSE 'Unknown Role'
         END AS role,
         related.overview AS overview, related.embedding AS embedding
     RETURN related_movie, person, role, overview, embedding
    """
    with driver.session() as session:
        result = session.run(query, title=title)
        documents = []
        for record in result:
            documents.append(
                Document(
                    content=record.get("overview", "No overview available"),  # Store overview in content
                    meta={
                        "title": record.get("related_movie", "Unknown Movie"),  # Movie title
                        "person": record.get("person", "Unknown Person"),       # Actor/Director's name
                        "role": record.get("role", "Unknown Role"),              # Actor or Director
                        "embedding": record.get("embedding", "No embedding available")  # Retrieve the precomputed embedding
                    },
                )
            )
    return documents 
```

## 通过路径查询解锁见解

图推理的另一个强大功能是能够查询节点之间的特定路径。例如，通过一系列合作找出两部电影是如何连接的，可以揭示令人惊讶的见解。

看看下面的查询：

```py
MATCH path = (m1:Movie {title: "Inception"})-[:ACTED_IN*3]-(m2:Movie)
RETURN m1.title, m2.title, path 
```

这个查询找到了《盗梦空间》和另一部电影通过共享演员连接，跨越了三个层次的关系。

![图 6.1 — 电影图中三跳路径遍历的插图](img/B31107_06_01.png)

图 6.1 — 电影图中三跳路径遍历的插图

本图中的插图显示了一个电影图中的三跳路径遍历，从电影《盗梦空间》开始，通过一系列演员合作链达到《电影 C》。这条路径是使用重复三次的`ACTED_IN`关系探索电影之间连接的 Cypher 查询的结果。在所描述的示例中，《盗梦空间》通过演员 A 与《电影 B》相连，而《电影 B》又通过演员 B 进一步与《电影 C》相连。每跳代表从电影到演员或反之的转换，形成一个三跳的无向遍历。这种可视化突出了 Neo4j 中的多跳推理如何揭示更深层次的间接关系——这对于内容发现、推荐系统和协作网络分析等应用非常有价值。

**注意**

这些只是代码片段。完整版本可在 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch6/graph_reasoning.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch6/graph_reasoning.py)。

通过结合 Neo4j 的图推理和 Haystack 的相似度理解，我们已经能够捕捉到数据中的有意义联系，例如电影和演员之间的关系、理解多跳导演合作，以及揭示实体之间的复杂路径。

接下来，我们将探讨如何优化这些过程，以确保随着图在复杂性和规模上的增长，保持高性能。

# 扩展您的 Haystack 和 Neo4j 集成

随着您的系统扩展，对 Haystack 和 Neo4j 的需求也会增加。优化性能变得至关重要，尤其是在处理大型数据集、更复杂的图结构和高级搜索功能时。

在本节中，我们将重点关注最佳实践和技术，以确保您的 Haystack 和 Neo4j 集成能够高效地处理增加的负载。我们将探讨查询优化、缓存策略、索引改进以及扩展基础设施的技术，以满足性能需求，同时不牺牲速度或准确性，以下各小节将进行详细说明。

## 优化大型图上的 Neo4j 查询

随着您的 Neo4j 图的大小和复杂性增加，查询性能可能会下降，尤其是在遍历多个关系或处理大型数据集时。以下是一些提高 Neo4j 查询性能的技术：

+   **使用索引和约束**：确保经常查询的属性，如`title`和`name`，被索引。索引可以加快节点查找速度，并使遍历更高效：

    ```py
    CREATE INDEX FOR (m:Movie) ON (m.title);
    CREATE INDEX FOR (p:Person) ON (p.name); 
    ```

+   **配置和优化查询**：使用 Neo4j 的`PROFILE`或`EXPLAIN`关键字来分析查询的性能。这有助于您了解查询的哪些部分正在减慢速度，以及您可以在哪里进行优化：

    ```py
    PROFILE MATCH (m:Movie {title: "Inception"}) RETURN m; 
    ```

+   **尽早限制结果数量**：如果您正在处理大型结果集，请在查询的早期阶段限制返回的节点数量，以避免过度获取数据：

    ```py
    MATCH (m:Movie)-[:ACTED_IN]->(a:Actor) RETURN m.title LIMIT 10; 
    ```

## 缓存嵌入和查询结果

当扩展 Haystack 和 Neo4j 时，缓存可以帮助减少冗余计算和网络调用，显著提高性能。通过缓存嵌入和查询结果，您可以提高搜索系统的效率，尤其是在处理大量查询时。以下是这些缓存策略如何产生差异的示例：

+   **缓存嵌入**：将 Haystack 生成的嵌入存储在 Neo4j 或单独的缓存层（如 Redis）中。通过缓存嵌入，您可以避免为频繁询问的查询重新计算它们：

    ```py
    # Example of caching embeddings
    embedding_cache = {}  # Simple in-memory cache, replace with Redis for larger setups
    def get_cached_embedding(query):
        if query in embedding_cache:
            return embedding_cache[query]
        else:
            embedding = text_embedder.run(query).get("embedding")
            embedding_cache[query] = embedding
            return embedding 
    ```

+   **缓存查询结果**：对于频繁执行的 Neo4j 查询，考虑在内存中缓存查询结果或使用缓存（如 Redis 或 Memcached）。这通过为常用查询返回缓存结果来减少对 Neo4j 的负载：

    ```py
    # Example using a Redis cache for Neo4j query results
    import redis
    cache = redis.Redis()
    def get_cached_query_result(query):
        cached_result = cache.get(query)
        if cached_result:
            return cached_result
        else:
            # Run the query against Neo4j
            result = run_neo4j_query(query)
            cache.set(query, result)
            return result 
    ```

## 高效使用向量索引

随着基于向量的搜索能力扩展，优化 Neo4j 中的向量索引对于保持性能至关重要。您可以这样做：

+   **配置向量索引以实现高性能**：确保您的 Neo4j 中的向量索引根据嵌入维度和搜索需求进行优化配置：

    ```py
    CREATE VECTOR INDEX overview_embeddings IF NOT EXISTS
    FOR (m:Movie) ON (m.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536, 
            `vector.similarity_function`: 'cosine'
        }
    } 
    ```

+   **批量写入操作**：当将许多嵌入写入 Neo4j 时，使用批量操作以减少单个写入的开销：

    ```py
    document_store.write_documents(embeddings_list, batch_size=100)  
    # Batch size optimized for performance 
    ```

## 负载均衡和水平扩展

为了处理 Haystack 和 Neo4j 上的增加的交通和负载，水平扩展和负载均衡是必不可少的。通过实施负载均衡和水平扩展，你可以确保在大量交通下，你的系统保持响应性和弹性。以下是每种方法如何贡献于可扩展性的说明：

+   **扩展 Neo4j**：利用 Neo4j AuraDB 或 Neo4j 集群，你可以将你的数据库工作负载分配到多个实例，增强读写能力。这对于需要快速数据检索和大规模处理的应用程序特别有益。

+   **负载均衡 Haystack**：通过负载均衡器将传入的搜索查询分配到多个 Haystack 实例，可以防止任何单个实例过载。这种方法保持了一致的性能，并确保了高可用性，即使需求增长也是如此。

+   **使用 Kubernetes**：通过容器化实例在 Kubernetes 上部署 Haystack 允许你根据流量调整副本数量，轻松地进行扩展。Kubernetes 动态编排这些副本，确保资源与需求相匹配，并且你的系统可以有效地处理使用高峰。以下是一个 Kubernetes 部署配置示例，用于扩展 Haystack，其中创建了多个副本以有效地处理增加的交通：

    ```py
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: haystack-deployment
    spec:
      replicas: 3  # Number of replicas to scale based on traffic
      selector:
        matchLabels:
          app: haystack
      template:
        metadata:
          labels:
            app: haystack
        spec:
          containers:
          - name: haystack
            image: haystack:latest 
    ```

通过实施这些优化策略，你可以确保随着你的数据和查询复杂性的增长，Haystack 和 Neo4j 的集成保持高性能和可扩展性。无论是通过缓存、高效的索引还是水平扩展基础设施，这些技术都将帮助你保持速度和准确性，即使在不断增长的负载下。随着你的系统增长，优化性能是至关重要的，但维护和监控你的 AI 驱动的搜索系统的健康状况同样关键。

**注意**

想了解 Neo4j 如何实现行业领先的速度和可扩展性，尤其是在你的数据和查询复杂性增长时？请探索这篇博客文章：*使用 Neo4j 实现无与伦比的速度和可扩展性* ([`neo4j.com/blog/machine-learning/achieve-unrivaled-speed-and-scalability-neo4j/`](https://neo4j.com/blog/machine-learning/achieve-unrivaled-speed-and-scalability-neo4j/))。

在下一节中，我们将探讨保持你的系统长期平稳运行的最佳实践，重点关注如何监控性能、设置警报，并确保代码之外的长远稳定性和可靠性。

# 维护和监控你的 AI 驱动的搜索系统的最佳实践

构建一个强大的 AI 驱动的搜索系统只是开始。为了确保其长期成功，你需要超越初始设置，并专注于随着时间的推移维护和监控你的系统。定期的性能检查、主动监控和坚实的日志策略对于识别瓶颈、防止系统故障和优化资源使用至关重要。

现在我们来谈谈保持 Haystack 和 Neo4j 集成平稳运行的最佳实践，包括监控关键性能指标、设置关键问题的警报以及实施可持续的维护流程，以确保您的搜索系统即使在扩展时也能保持可靠和高效。

性能优化不是一个一次性活动。我们需要持续监控和收集指标，以识别瓶颈和改进区域。让我们看看我们如何实现这一点。

## 监控 Neo4j 和 Haystack 性能

定期跟踪查询响应时间、数据库性能和整体系统健康对于维护 AI 驱动的搜索系统至关重要。为 Neo4j 和 Haystack 设置监控以跟踪关键指标、识别瓶颈并确保平稳运行：

+   **Neo4j 监控**：利用 Neo4j 内置的指标以及与 Prometheus 和 Grafana 等工具的集成，可视化查询性能并监控系统负载。

+   **Haystack 监控**：使用 Grafana 和 Prometheus 监控 Haystack 中的查询吞吐量、延迟和响应时间。

这里是一个监控查询响应时间的示例：

```py
# Example: Monitor response time of a query in Haystack
import time
start_time = time.time()
result = retriever.retrieve(query)
end_time = time.time()
response_time = end_time - start_time
print(f"Query response time: {response_time} seconds") 
```

## 设置关键问题的警报

设置自动警报可以确保在性能或系统故障发生时您会收到通知。通过使用 Prometheus 与 Alertmanager 或 Grafana，您可以设置基于阈值的警报，用于慢查询、失败的搜索或增加的负载。

例如，您可以为当 Neo4j 查询响应时间超过某个阈值或当 Haystack 的搜索延迟超出可接受范围时触发的警报。

您可以在 [`neo4j.com/docs/operations-manual/current/monitoring/`](https://neo4j.com/docs/operations-manual/current/monitoring/) 上了解更多关于 Neo4j 监控和警报的信息。

## 实施日志策略

详细日志有助于排查问题并了解失败或性能下降的根本原因。在 Haystack 和 Neo4j 中实施日志记录，包括记录查询执行时间、失败和系统资源使用情况。

在 [`neo4j.com/docs/operations-manual/current/logging/`](https://neo4j.com/docs/operations-manual/current/logging/) 上了解更多关于 Neo4j 日志的信息。有关 Haystack 日志和调试的更多信息，请访问 [`docs.haystack.deepset.ai/docs/debug`](https://docs.haystack.deepset.ai/docs/debug)。

## 建立定期的维护流程

定期安排的维护确保您的 AI 驱动的搜索系统随着时间的推移继续以最佳性能运行。这包括以下内容：

+   **Neo4j**：执行定期的索引重建、数据一致性检查和磁盘空间监控。有关 Neo4j 维护的更多信息，请参阅 [`neo4j.com/docs/operations-manual/current/backup-restore/maintenance/`](https://neo4j.com/docs/operations-manual/current/backup-restore/maintenance/)。

+   **Haystack**：监控嵌入质量，根据需要更新模型，并管理文档存储增长以避免性能下降。有关 Haystack 优化和维护的更多信息，请参阅[`docs.haystack.deepset.ai/docs/pipelineoptimization`](https://docs.haystack.deepset.ai/docs/pipelineoptimization)。

通过实施这些最佳实践，你确保你的 AI 驱动搜索系统保持稳健、可靠，并能适应不断变化的需求。主动监控、有效的日志记录和定期的维护使你能够在问题影响性能之前发现它们，并确保随着数据和查询负载的增长，系统运行平稳。这些策略不仅防止了停机和不效率，还使你的系统能够无缝地发展和扩展。随着你继续构建和改进你的 AI 驱动搜索，对监控和维护的持续关注将是维持其长期成功的关键。

# 摘要

在本章中，我们探讨了如何优化你的 Haystack 和 Neo4j 集成，并确立了维护和监控你的 AI 驱动搜索系统的最佳实践。你学习了关于缓存、高效索引、查询优化以及扩展你的基础设施以处理增长的数据和查询负载的关键策略。我们还强调了监控系统性能、设置警报以及实施坚实的日志策略以保持系统长期平稳运行的重要性。随着数据和复杂性的增加，这些知识是创建快速、可靠和可扩展的搜索系统的关键第一步。

随着我们结束 Haystack 的这部分旅程，本书的下一部分将转向将 Spring AI 框架和 LangChain4j 与 Neo4j 集成。在接下来的章节中，你将探索这些技术如何结合在一起来构建复杂的推荐系统，进一步增强你的 AI 驱动应用程序的功能。
