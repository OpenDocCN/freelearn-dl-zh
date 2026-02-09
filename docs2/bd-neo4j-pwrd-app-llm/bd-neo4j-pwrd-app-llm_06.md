# 5

# 使用 Neo4j 和 Haystack 实现强大的搜索功能

在本章中，我们开始将 Haystack 与 Neo4j 集成，结合 LLM 和图数据库的能力来构建一个由 AI 驱动的搜索系统。**Haystack** 是一个开源框架，使开发者能够通过利用现代 NLP 技术、机器学习模型和基于图的数据来创建 AI 驱动的应用程序。对于我们的智能搜索，Haystack 将作为一个统一的平台来协调 LLM、搜索引擎和数据库，提供高度上下文化和相关的搜索结果。

在前一章的工作基础上——我们清理并结构化 Neo4j 数据——我们将首先使用 OpenAI 的 GPT 模型生成嵌入。这些嵌入将丰富图结构，使其更强大，并能够处理细微的、上下文感知的搜索查询。Haystack 将作为 OpenAI 模型和 Neo4j 图数据库之间的桥梁，使我们能够结合两者的优势。

在本章中，您将学习如何设置和配置 Haystack 以实现与 Neo4j 的无缝集成。我们将引导您构建强大的搜索功能，并最终使用 Hugging Face Spaces 上的 Gradio 部署这个完全功能化的搜索系统。

在本章中，我们将涵盖以下主要主题：

+   使用 Haystack 生成嵌入以增强您的 Neo4j 图

+   将 Haystack 连接到 Neo4j 进行高级向量搜索

+   构建强大的搜索体验

+   微调您的 Haystack 集成

# 技术要求

要成功实现 Haystack 和 Neo4j 的集成，并构建一个由 AI 驱动的搜索系统，您需要确保您的环境已正确设置。以下是本章的技术要求列表：

+   **Python**：您需要在您的系统上安装 Python `3.11`。Python 用于脚本编写和与 Neo4j 数据库交互。您可以从官方 Python 网站下载 Python：[`www.python.org/downloads/`](https://www.python.org/downloads/)。

+   **Neo4j AuraDB 或本地 Neo4j 实例**：您需要访问一个 Neo4j 数据库来存储和查询您的图数据。这可以是本地安装的 Neo4j 实例或云托管的 Neo4j AuraDB 实例。如果您正在跟随前一章的内容，其中我们讨论了 `graph_build.py` 脚本 ([`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch4/graph_build.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch4/graph_build.py))，您可以继续使用已设置并填充数据的相同 Neo4j 实例。这确保了连续性，并允许您在已导入的结构化数据之上构建。

+   **Cypher 查询语言**：熟悉 Cypher 查询语言是必要的，因为我们将在创建和查询图时广泛使用 Cypher。你可以在 Cypher 查询语言文档中了解更多关于 Cypher 语法的细节：[`neo4j.com/docs/cypher/`](https://neo4j.com/docs/cypher/)。

+   **Neo4j Python 驱动程序**：安装 Neo4j Python 驱动程序以使用 Python 连接到 Neo4j 数据库。你可以通过`pip`安装它：

    ```py
    pip install neo4j 
    ```

+   **Haystack**：我们将使用 Haystack v2.5.0。

使用 pip 安装 Haystack：

```py
pip install haystack-ai 
```

+   **OpenAI API 密钥**：要成功使用基于 GPT 的模型生成嵌入，你需要一个 OpenAI API 密钥。

如果你还没有账户，请在 OpenAI（[`platform.openai.com/signup`](https://platform.openai.com/signup)）注册以获取 API 密钥。

**注意**

免费层 API 密钥在本项目的多数用例中都不会工作。你需要一个活跃的付费 OpenAI 订阅才能访问必要的端点和使用限制。

登录后，导航到你的 OpenAI 仪表板中的**API 密钥**部分（[`platform.openai.com/api-keys`](https://platform.openai.com/api-keys)）并生成一个新的 API 密钥。

你还需要使用 pip 安装 OpenAI 包。在你的终端中运行以下命令：

```py
pip install openai 
```

+   **Gradio**：我们将使用 Gradio 创建一个用户友好的聊天机器人界面。使用 pip 安装 Gradio：

    ```py
    pip install gradio 
    ```

+   **Hugging Face 账户**：要将你的聊天机器人托管在 Hugging Face Spaces 上，你需要一个 Hugging Face 账户。如果你还没有账户，请在 Hugging Face 网站上注册：[`huggingface.co/`](https://huggingface.co/)。

+   **Google Cloud Storage（可选）**：如果你将 CSV 文件存储在 Google Cloud Storage 上，请确保在脚本中正确配置了文件路径。

+   **python-dotenv 包**：确保安装`python-dotenv`包以管理项目中的环境变量：

    ```py
    pip install python-dotenv 
    ```

本章的所有代码都可在以下 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs)。

在此仓库中，导航到名为`ch6`的文件夹以访问与本章相关的代码示例和资源。此文件夹包含实现 Neo4j 和 Haystack 集成以及使用电影数据集构建 AI 驱动的搜索系统所需的所有必要脚本、文件和配置。

确保克隆或下载仓库，这样你就可以跟随本章中的代码示例进行操作。

使用 Haystack 生成嵌入以增强你的 Neo4j 图

在本节中，我们将专注于生成上一章中添加到我们的 Neo4j 图中的电影剧情的嵌入。**嵌入**是现代搜索系统的一个关键部分，因为它们将文本转换为高维向量，从而实现相似度搜索。这使得搜索引擎能够理解单词和短语之间的上下文关系，提高搜索结果的准确性和相关性。

我们将集成 Haystack 与 OpenAI 的基于 GPT 的模型以生成这些嵌入，并将它们存储在你的 Neo4j 图中。这将启用更准确和上下文感知的搜索功能。

## 初始化 Haystack 和 OpenAI 以生成嵌入

在生成嵌入之前，你需要确保 Haystack 已设置并集成到 OpenAI 的 API 中，以从其基于 GPT 的模型中检索嵌入。按照以下步骤设置 Haystack：

1.  通过以下命令安装所需的库（如果你还没有安装）：

    ```py
    pip install haystack haystack-ai openai neo4j-haystack 
    ```

1.  接下来，配置你的 OpenAI API 密钥，并确保它在你的`.env`文件中设置：

    ```py
    makefile
    OPENAI_API_KEY=your_openai_api_key_here 
    ```

1.  通过创建一个初始化 Haystack 并连接到 OpenAI 以生成嵌入的 Python 脚本来初始化 Haystack 使用 OpenAI 嵌入：

    ```py
    # Initialize Haystack with OpenAI for text embeddings
    def initialize_haystack():
        # Initialize document store (In-memory for now, but you can configure other stores)
        document_store = InMemoryDocumentStore()
        # Initialize OpenAITextEmbedder to generate text embeddings
        embedder = OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        return embedder 
    ```

此配置初始化 Haystack 使用内存中的文档存储，并使用 OpenAI 嵌入设置检索器。

## 为电影剧情生成嵌入

接下来，我们将为存储在 Neo4j 图中的电影剧情生成嵌入。目标是检索剧情描述，为它们生成嵌入，并将这些嵌入链接回相应的电影节点：

1.  **从 Neo4j 查询电影剧情**：首先，你需要从 Neo4j 查询电影剧情。使用以下 Cypher 查询检索电影标题和剧情摘要：

    ```py
    # Retrieve movie plots and titles from Neo4j
    def retrieve_movie_plots():
        # The query retrieves the "title", "overview", and "tmdbId" properties of each Movie node
        query = """
        MATCH (m:Movie)
        WHERE m.embedding IS NULL
        RETURN m.tmdbId AS tmdbId, m.title AS title, m.overview AS overview
        """
        with driver.session() as session:
            results = session.run(query)
            # Each movie's title, plot (overview), and ID are retrieved and stored in the movies list
            movies = [
                {
                    "tmdbId": row["tmdbId"],
                    "title": row["title"],
                    "overview": row["overview"]
                }
                for row in results
            ]
        return movies 
    ```

这将返回图中的每个电影的`tmdbId`值和概述（即剧情摘要）。

1.  **使用 OpenAI 和 Haystack 生成嵌入**：一旦检索到剧情摘要，就可以使用 Haystack 的`OpenAITextEmbedder`生成嵌入：

    ```py
    #Parallel embedding generation with ThreadPoolExecutor
    def generate_and_store_embeddings(embedder, movies, max_workers=10): 
        results_to_store = []
        def process_movie(movie):
            title = movie.get("title", "Unknown Title")
            overview = str(movie.get("overview", "")).strip()
            tmdbId = movie.get("tmdbId")
            if not overview:
                print(f"Skipping {title} — No overview available.")
                return None
            try:
                print(f"Generating embedding for: {title}")
                embedding_result = embedder.run(overview)
                embedding = embedding_result.get("embedding")
                if embedding:
                    return (tmdbId, embedding)
                else:
                    print(f"No embedding generated for: {title}")
            except Exception as e:
                print(f"Error processing {title}: {e}")
            return None 
    ```

1.  **在 Neo4j 中存储嵌入**：生成嵌入后，下一步是将它们存储在你的 Neo4j 图中。每个电影节点都将更新一个属性，以存储其嵌入：

    ```py
    # Store the embeddings back in Neo4j
    def store_embedding_in_neo4j(tmdbId, embedding):
        query = """
        MATCH (m:Movie {tmdbId: $tmdbId})
        SET m.embedding = $embedding
        """
        with driver.session() as session:
            session.run(query, tmdbId=tmdbId, embedding=embedding)
        print(f"Stored embedding for TMDB ID: {tmdbId}") 
    ```

这将在 Neo4j 图中的每个`Movie`节点中存储名为`embedding`的属性。

1.  **验证 Neo4j 中的嵌入存储**：一旦嵌入存储，你可以通过查询几个节点来检查`embedding`属性以验证它们的存在：

    ```py
    # Verify embeddings stored in Neo4j
    def verify_embeddings():
        query = """
        MATCH (m:Movie)
        WHERE exists(m.embedding)
        RETURN m.title, m.embedding
        LIMIT 10
        """
        with driver.session() as session:
            results = session.run(query)
            for record in results:
                title = record["title"]
                embedding = np.array(record["embedding"])[:5]
                print(f" {title}: {embedding}...") 
    ```

此查询将返回一些电影的标题和嵌入，以便你可以验证嵌入是否已成功存储。

**注意**

这些只是代码片段。完整版本可在 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch5/generate_embeddings.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch5/generate_embeddings.py)。

我们现在已经用这些嵌入丰富了我们的图，从而添加了相似性搜索，这将使我们能够执行更具有上下文意识和智能的查询。这一步对于增强搜索体验和基于文本意义的先进检索操作至关重要，而不是简单的关键词匹配。

现在我们已经用向量嵌入丰富了我们的 Neo4j 图，下一步是将 Haystack 连接到 Neo4j 以进行高级向量搜索。在接下来的章节中，我们将重点介绍如何使用这些嵌入在 Neo4j 中执行高效且准确的向量搜索，使我们能够根据它们的向量相似性检索电影或节点。

# 将 Haystack 连接到 Neo4j 以进行高级向量搜索

现在电影嵌入已存储在 Neo4j 中，我们需要在 `embedding` 属性上配置一个向量索引，这将使我们能够根据它们的向量相似性高效地搜索电影。通过在 Neo4j 中创建向量索引，我们能够快速检索在高维嵌入空间中彼此接近的节点，这使得执行复杂的查询成为可能，例如找到具有相似剧情摘要的电影。

一旦创建了向量索引，它将与 Haystack 集成以从 Neo4j 执行基于向量的检索。此搜索将基于向量相似性机制，如余弦相似性。

## 在 Neo4j 中创建向量搜索索引

您首先需要删除嵌入属性上的任何现有向量索引（如果存在），然后创建一个新的索引以执行向量搜索。这是您如何在 Python 脚本中使用 Cypher 查询来完成此操作的示例：

```py
def create_or_reset_vector_index():
    with driver.session() as session:
        try:
            # Drop the existing vector index if it exists
            session.run("DROP INDEX overview_embeddings IF EXISTS ")
            print("Old index dropped")
        except:
            print("No index to drop")
        # Create a new vector index on the embedding property
        print("Creating new vector index")
        query_index = """
        CREATE VECTOR INDEX overview_embeddings IF NOT EXISTS
        FOR (m:Movie) ON (m.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'}}
        """
        session.run(query_index)
        print("Vector index created successfully") 
```

## 使用 Haystack 和 Neo4j 向量索引执行相似性搜索

在 Neo4j 图上创建向量索引后，您可以利用 Haystack 执行基于电影剧情嵌入的相似性搜索查询。这种方法允许您比较给定电影剧情或任何文本查询与现有电影概述之间的相似性，根据它们的嵌入返回最相关的结果。在这个示例中，我们使用 Haystack 库中的 `OpenAITextEmbedder` 模型将文本查询转换为嵌入，然后使用它来搜索具有相似剧情的 Neo4j 图中的电影。

这就是您生成查询嵌入并执行相似性搜索的方法：

```py
text_embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )
    # Step 1: Create embedding for the query
    query_embedding = text_embedder.run(query).get("embedding")

    if query_embedding is None:
        print("Query embedding not created successfully.")
        return

    print("Query embedding created successfully.") 
```

## 使用 Haystack 和 Neo4j 运行向量搜索查询

一旦创建了向量索引并将嵌入存储在 Neo4j 中，您就可以通过传递查询或样本电影剧情来执行基于向量的搜索。系统将为查询生成一个嵌入，将其与存储在 Neo4j 中的嵌入进行比较，并返回最相关的结果。

这里是一个使用 Haystack 进行向量搜索的示例，它显示了最相似的电影剧情，而不使用 Cypher：

```py
# Step 2: Search for similar documents using the query embedding
    similar_documents = document_store.query_by_embedding(
        query_embedding, top_k=3
    )
    if not similar_documents:
        print("No similar documents found.")
        return
    print(f"Found {len(similar_documents)} similar documents.")
    print("\n\n")
    # Step 3: Displaying results
    for doc in similar_documents:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = doc.score
        print(
             f"Title: {title}\nOverview: {overview}\n"
             f"Score: {score:.2f}\n{'-'*40}"
        )
    print("\n\n") 
```

现在，我们将集成 Neo4j Cypher 查询与 Haystack 以运行向量搜索，从而实现类似剧情的检索。

## 使用 Cypher 和 Haystack 运行向量搜索查询

要运行向量搜索，我们将使用 Cypher 的图查询功能，同时使用由`OpenAITextEmbedder`生成的向量嵌入进行相似度搜索。

与直接使用 Haystack 查询向量索引不同，这种方法结合了 Cypher 的灵活性，可以返回更复杂的数据，例如电影元数据（例如，演员和类型），同时仍然保持向量相似度搜索的效率。

这里涉及到的步骤如下：

1.  **使用 OpenAITextEmbedder 嵌入查询**：将用户的文本查询（例如，电影剧情）转换为高维向量嵌入。

1.  **使用 Neo4j 和 Cypher 进行搜索**：使用 Cypher 通过比较查询嵌入与存储在 Neo4j 向量索引中的电影剧情嵌入来检索相似电影。

1.  **返回丰富数据**：为每个结果检索额外的电影信息，例如标题、概述、演员、类型和评分（相似度）。

这就是实现向量搜索的方法：

1.  **定义 Cypher 查询**：我们首先定义一个 Cypher 查询，该查询搜索 Neo4j 向量索引（`overview_embeddings`），以检索基于查询嵌入和电影嵌入之间的余弦相似度的`top_k`最相似的电影：

    ```py
    cypher_query = """
        CALL db.index.vector.queryNodes("overview_embeddings", $top_k, $query_embedding)
        YIELD node AS movie, score
        MATCH (movie:Movie)
        RETURN movie.title AS title, movie.overview AS overview, score
    """ 
    ```

1.  **生成查询嵌入**：使用`OpenAITextEmbedder`，我们将用户的输入查询（例如，电影剧情）转换为嵌入。此嵌入将被传递到 Neo4j 向量索引，以便与存储的电影嵌入进行比较：

    ```py
    text_embedder = OpenAITextEmbedder(
        api_key= Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    ) 
    ```

1.  **使用 Haystack 管道运行向量搜索**：我们设置 Haystack 管道来管理 Haystack 组件：

    +   `query_embedder`从用户查询生成嵌入

    +   `retriever`在 Neo4j 上使用查询嵌入运行 Cypher 查询，并返回最相似的电影：

        ```py
        retriever = Neo4jDynamicDocumentRetriever(
            client_config=client_config,
            runtime_parameters=["query_embedding"],
            compose_doc_from_result=True,
            verify_connectivity=True,
        )
        pipeline = Pipeline()
        pipeline.add_component("query_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.connect(
            "query_embedder.embedding", "retriever.query_embedding"
        )
        result = pipeline.run(
            {
                "query_embedder": {"text": query},
                "retriever": {
                    "query": cypher_query,
                    "parameters": {
                        "index": "overview_embeddings", "top_k": 3
                    },
                },
            }
        ) 
        ```

1.  **显示结果**：一旦搜索完成，我们从 Neo4j 图中提取结果，并显示电影标题、概述和相似度分数：

    ```py
    # Extracting documents from the retriever results
    documents = result["retriever"]["documents"]
    for doc in documents:
        # Extract title and overview from document metadata
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        # Extract score from the document
        score = getattr(doc, "score", None)
        score_display = f"{score:.2f}" if score is not None else "N/A"
        # Print the title, overview, and score (or N/A for missing score)
        print(
             f"Title: {title}\nOverview: {overview}\n"
             f"Score: {score_display}\n{'-'*40}\n"
        ) 
    ```

使用 Cypher 和 Haystack 提供了以下好处：

+   **Cypher 的灵活性**：通过结合 Cypher 和 Haystack，我们不仅可以查询嵌入，还可以检索基于图的其他信息，例如演员、类型和实体之间的关系。

+   **丰富结果**：除了检索最相似的电影外，您还可以轻松扩展查询以检索相关元数据（例如，演员、类型、评分）或使用额外的过滤条件（例如，上映年份、类型）来细化搜索。

+   **针对大型图优化**：Neo4j 的向量索引允许高效查询具有复杂关系的大型数据集，而 Haystack 的嵌入模型提供了对电影剧情的准确理解。

让我们看看下一个示例用例。

## 示例用例

考虑寻找剧情类似于*一个英雄必须拯救世界免于毁灭*的电影。通过使用我们刚刚创建的管道，您可以检索相关结果：

```py
Title: The Matrix
Overview: A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.
Score: 0.98
----------------------------------------
Title: Inception
Overview: A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.
Score: 0.96
----------------------------------------
Title: The Dark Knight
Overview: Batman raises the stakes in his war on crime, with the help of Lieutenant Jim Gordon and District Attorney Harvey Dent.
Score: 0.94
---------------------------------------- 
```

此管道结合了两者之长——通过向量嵌入进行相似度搜索和通过 Cypher 进行图查询的丰富数据功能——允许在大型数据集（如电影）上进行强大且灵活的搜索。

**注意**

这些只是代码片段。完整版本可在 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch5/vector_search.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch5/vector_search.py)。

我们现在已经将 Haystack 连接到 Neo4j 并启用了高级向量搜索功能。有了向量索引，Neo4j 现在可以高效地根据嵌入相似度搜索类似的电影节点。Haystack 的集成允许您无缝地使用 `Neo4jDynamicDocumentRetriever` 执行这些搜索。此检索器通过利用向量嵌入和 Neo4j 的图功能在您的图中搜索类似项。

在下一节中，我们将探讨如何构建一个利用 Haystack 和 Neo4j 的强大功能来提供丰富、上下文感知响应的搜索驱动聊天机器人。使用 Gradio，我们将创建一个直观的聊天机器人界面，可以与用户交互并通过自然语言查询执行高级搜索。这将结合 LLMs、向量搜索和 Neo4j 的优势，创建一个用户友好、AI 驱动的搜索体验。

# 使用 Gradio 和 Haystack 构建 search-driven 聊天机器人

在本节中，我们将集成 Gradio 来构建一个由 Haystack 和 Neo4j 驱动的交互式聊天机器人界面。Gradio 使得创建一个用于与聊天机器人交互的基于网页的界面变得简单。聊天机器人将允许用户输入查询，然后触发对存储在 Neo4j 中的电影嵌入的基于向量的搜索。聊天机器人将返回详细的响应，包括电影标题、概述和相似度分数，提供信息丰富且用户友好的体验。

## 设置 Gradio 界面

如果您尚未安装 Gradio，请通过运行以下命令进行安装：

```py
pip install gradio 
```

**注意**

本章中的脚本与 Gradio v `5.23.1` 兼容。

接下来，我们将设置一个基本的 Gradio 界面，该界面触发我们的搜索管道并显示结果：

```py
import gradio as gr
# Define the Gradio chatbot interface
def chatbot(user_input):
    return perform_vector_search_cypher(user_input)
# Create Gradio interface
chat_interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        placeholder="What kind of movie would you like to watch?",
        lines=3,
        label="Your movie preference"
    ),
    outputs=gr.Textbox(
        label="Recommendations",
        lines=12
    ),
    title="AI Movie Recommendation System",
    description="Ask me about movies! I can recommend movies based on your preferences.",
    examples=[
        ["I want to watch a sci-fi movie with time travel"],
        ["Recommend me a romantic comedy with a happy ending"],
        ["I'm in the mood for something with superheroes but not too serious"],
        ["I want a thriller that keeps me on the edge of my seat"],
        ["Show me movies about artificial intelligence taking over the world"]
    ],
    flagging_mode="never" 
```

此界面允许用户输入文本查询，聊天机器人将使用 `perform_vector_search_cypher()` 函数搜索最相关的电影。

## 与 Haystack 和 Neo4j 集成

为了为聊天机器人提供动力，我们将将其连接到 Haystack 的嵌入生成和 Neo4j 的向量搜索功能。我们将使用 `OpenAITextEmbedder` 为查询和存储在 Neo4j 中的电影情节生成嵌入。电影嵌入存储在 Neo4j 内部的向量索引中，我们将查询最相似的电影。

这就是如何将我们的聊天机器人与之前的 Haystack 设置集成：

```py
# Conversational chatbot handler using Cypher-powered search and Haystack
def perform_vector_search(query):
    print("MESSAGES RECEIVED:", user_input)
    cypher_query = """
        CALL db.index.vector.queryNodes("overview_embeddings", $top_k, $query_embedding)
        YIELD node AS movie, score
        MATCH (movie:Movie)
        RETURN movie.title AS title, movie.overview AS overview, score
    """
    # Embedder
    embedder = OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )
    # Retriever
    retriever = Neo4jDynamicDocumentRetriever(
        client_config=client_config,
        runtime_parameters=["query_embedding"],
        compose_doc_from_result=True,
        verify_connectivity=True,
    )
    # Pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect(
        "query_embedder.embedding", "retriever.query_embedding"
    ) 
```

## 将 Gradio 连接到完整管道

现在，将这个 Gradio 聊天机器人连接到您已经设置的 Haystack 和 Neo4j 管道。Gradio 接口将调用 `perform_vector_search_cypher()` 函数，该函数反过来利用 `Neo4jDynamicDocumentRetriever` 根据用户的查询搜索类似的电影。

更新 `main()` 函数以初始化聊天机器人：

```py
# Main function to orchestrate the entire process
def main():
    # Step 1: Create or reset vector index in Neo4j AuraDB
    create_or_reset_vector_index()
    # Step 2: Launch Gradio chatbot interface
    chat_interface.launch()
if __name__ == "__main__":
    main() 
```

## 运行聊天机器人

要运行聊天机器人，只需执行您的 Python 脚本。Gradio 接口将在您的浏览器中启动，让您能够实时与聊天机器人互动：

```py
python search_chatbot.py 
```

在您的浏览器中将会启动一个 Gradio 接口，让您能够实时与聊天机器人互动。您可以输入如下查询：

```py
"Tell me about a hero who saves the world." 
```

聊天机器人将根据向量搜索返回与该查询相似的剧情。

**注意**

这些只是代码片段。完整版本可在 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch5/search_chatbot.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch5/search_chatbot.py)。

当我们接近本节的结尾时，我们已经使用 Gradio、Haystack 和 Neo4j 构建了一个功能齐全的搜索驱动聊天机器人。该聊天机器人利用存储在 Neo4j 中的嵌入来执行高级基于向量的搜索，通过从 Neo4j 中检索有意义的电影标题和演员来以用户查询的形式向用户返回上下文相关的结果。

然而，这仅仅是开始。在下一节中，我们将更深入地探讨如何微调您的 Haystack 集成，并探索高级技术，例如优化搜索性能、调整检索模型以及改进聊天机器人的响应，以创建一个更加无缝和高效的搜索驱动体验。

# 微调您的 Haystack 集成

现在是时候探索如何微调此集成以提升性能和用户体验了。虽然当前的设置提供了丰富且上下文感知的响应，但您还可以实施一些高级技术来优化搜索过程、提高检索准确性，并使聊天机器人的交互更加流畅。

在本节中，我们将专注于调整 Haystack 的关键组件，包括尝试不同的嵌入模型、优化 Neo4j 查询以获得更快的速度，以及改进聊天机器人显示其响应的方式。这些改进将帮助您扩展聊天机器人以处理更复杂的查询，提高响应时间，并呈现更加相关的搜索结果。

## 尝试不同的嵌入模型

目前，我们正在使用 OpenAI 的 `text-embedding-ada-002` 模型来生成嵌入。虽然这个模型自发布以来一直作为各种任务的可靠和高效选择，但值得注意的是，OpenAI 最近推出了新的模型——例如 `text-embedding-3-small` 和 `text-embedding-3-large`——它们在性能和成本效益方面都取得了显著改进。例如，`text-embedding-3-small` 在多语言和英语任务中实现了更好的结果，同时比 `text-embedding-ada-002` 至少节省五倍的成本。尽管我们在这个项目中没有切换模型以保持一致性，但正在实施类似管道的读者可以考虑使用 `text-embedding-3-small` 来提高效率，同时不牺牲性能——特别是如果嵌入生成是频繁或大规模操作的话。

然而，Haystack 支持各种其他模型，并且你可以尝试不同的模型以查看哪个为你特定的用例提供了最准确或最相关的结果。例如，你可以切换到一个更复杂的 OpenAI 模型，具有更高的维度，或者尝试 Haystack 支持的另一个嵌入服务。

这就是你可以轻松切换到不同模型的方法：

```py
embedder = OpenAITextEmbedder(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model="text-embedding-babbage-001"  # Experiment with different models
) 
```

你还可以探索 OpenAI 的其他模型，甚至集成不同的嵌入服务，以查看哪个对你的电影聊天机器人表现最佳。

## 优化 Neo4j 以实现更快的查询

虽然 Neo4j 已经在处理基于图查询方面非常高效，但你还可以应用一些优化，特别是对于大型数据集。你可以索引额外的属性以提高查询性能。

### 索引额外属性

除了嵌入属性上的向量索引之外，你还可以索引其他频繁查询的属性，例如 `title` 或 `tmdbId`，以加快检索速度。这将确保每次你根据这些属性过滤或检索电影时，搜索都更快、更高效：

```py
def create_additional_indexes():
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS movie_title_index FOR (m:Movie) ON (m.title)")
        session.run("CREATE INDEX IF NOT EXISTS movie_tmdbId_index FOR (m:Movie) ON (m.tmdbId)")
        print("Additional indexes created successfully") 
```

通过索引这些属性，你可以在搜索不仅基于嵌入时优化查找，例如在按标题过滤或检索特定电影时。

为了持续改进聊天机器人的搜索体验，你可以记录用户查询并随着时间的推移进行分析。让我们详细谈谈这一点。

### 记录和分析查询

记录可以帮助你跟踪最常见的搜索模式。基于用户查询的日志及其分析，你可以调整索引策略，优化检索器，或调整嵌入模型以获得更好的准确性。

这就是实现简单日志记录机制的方法：

```py
import logging
logging.basicConfig(filename='chatbot_queries.log', level=logging.INFO)
def log_query(query):
    logging.info(f"User query: {query}") 
```

每当用户输入一个查询时，它将被记录以供将来分析。然后你可以分析这些日志，对系统进行有根据的调整，确保它随着时间的推移变得更加响应和准确。

这些技术可以帮助您显著提升搜索驱动的聊天机器人的性能、准确性和用户体验。无论是尝试不同的嵌入模型、优化 Neo4j 查询，还是改进结果格式，每一次调整都让您更接近无缝且强大的用户交互。

这些高级技术使您的聊天机器人能够有效扩展，处理更复杂的查询，并返回更加相关和吸引人的结果。

# 摘要

在本章中，我们通过整合 Gradio、Haystack 和 Neo4j 成功构建了一个功能齐全的搜索驱动的聊天机器人。我们首先通过 OpenAI 的模型生成的电影嵌入丰富了我们的 Neo4j 图，从而实现了高级的基于向量的搜索功能。从那里，我们将 Haystack 连接到 Neo4j，使我们能够在图中存储的嵌入上执行相似度搜索。最后，我们通过创建一个用户友好的聊天机器人界面（使用 Gradio），根据用户查询动态检索电影详情，如标题和演员，来完成整个构建过程。

在下一章中，我们将重点关注 Haystack 的高级搜索能力和搜索优化。我们还将讨论大型图的查询优化。
