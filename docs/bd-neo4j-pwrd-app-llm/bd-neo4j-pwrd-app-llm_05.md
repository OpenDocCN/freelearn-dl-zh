

# 第四章：使用电影数据集构建您的 Neo4j 图

在前面的章节中，我们学习了知识图谱如何成为一项变革性工具，它提供了一种结构化的方式来连接不同的数据点，使各种领域中的智能搜索、推荐和推理能力成为可能。

知识图谱擅长捕捉实体之间的复杂关系，对于需要深度上下文理解的应用程序来说，它们是不可或缺的。

基于其最先进的图数据库技术，Neo4j 在构建和管理知识图谱方面脱颖而出，成为领先的平台。正如我们在上一章中看到的，与传统的关系数据库不同，Neo4j 被设计用来轻松处理高度连接的数据，这使得查询更加直观，并能够更快地检索洞察。这使得它成为希望将原始的非结构化数据转化为有意义洞察的开发人员和数据科学家的理想选择。

在本章中，我们将涵盖以下主要主题：

+   为高效搜索设计的 Neo4j 图设计考虑

+   利用电影数据集

+   使用代码示例构建您的电影知识图谱

+   超越基础：用于复杂图结构的先进 Cypher 技术

# 技术要求

要成功完成本章的练习，您需要以下工具：

+   **Neo4j AuraDB**：您可以使用 Neo4j AuraDB，这是 Neo4j 的云版本，可在[`neo4j.com/aura`](https://neo4j.com/aura)找到。

+   **Cypher 查询语言**：熟悉 Cypher 查询语言是必要的，因为我们将广泛使用 Cypher 来创建和查询图。您可以在 Cypher 查询语言文档中找到有关 Cypher 语法的更多信息：[`neo4j.com/docs/cypher/`](https://neo4j.com/docs/cypher/)。

+   **Python**：您需要在系统上安装 Python 3.x。Python 用于脚本编写和与 Neo4j 数据库交互。您可以从官方 Python 网站下载 Python：[`www.python.org/downloads/`](https://www.python.org/downloads/)。

+   **Python 库**：

    +   **Python 的 Neo4j 驱动程序**：使用 Python 连接到 Neo4j 数据库，请安装 Neo4j Python 驱动程序。您可以通过`pip`安装它：

        ```py
        pip install neo4j 
        ```

    +   **pandas**：此库将用于数据处理和分析。您可以通过`pip`安装它：

        ```py
        pip install pandas 
        ```

+   **集成开发环境（IDE）**：推荐使用 PyCharm、VS Code 或 Jupyter Notebook 等 IDE 来高效地编写和管理您的 Python 代码。

+   **Git 和 GitHub**：需要基本的 Git 知识来进行版本控制。您还需要一个 GitHub 账户来访问本章的代码仓库。

+   **电影数据集**：**The Movie Database**（**TMDb**）是必需的，可在 Kaggle 上找到：[`www.kaggle.com/datasets/rounakbanik/the-movies-dataset/`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/)。

+   此数据集是**Movie Lens Datasets**（F. Maxwell Harper 和 Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. [`doi.org/10.1145/2827872`](https://doi.org/10.1145/2827872)）的衍生品。

+   由于存储限制，某些数据文件，如`credits.csv`和`ratings.csv`，可能不在 GitHub 上可用。但是，您可以从 GCS 存储桶中访问所有原始数据文件。

本章的所有代码均可在以下 GitHub 仓库中找到：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/tree/main/ch4`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/tree/main/ch4)。

该文件夹包含所有必要的文件和脚本，以帮助您使用电影数据集和 Cypher 代码构建 Neo4j 图。

确保克隆或下载该仓库，以跟随本章提供的代码示例。

GitHub 仓库包含访问原始数据文件的 GCS 路径。

# 为高效搜索设计的 Neo4j 图设计考虑

一个设计良好的 Neo4j 图确保您的搜索功能不仅准确，而且高效，能够快速检索相关信息。数据在图中的组织方式直接影响搜索结果的表现力和相关性，因此理解有效图建模的原则至关重要。

本节将深入探讨正确结构化您的 Neo4j 图的重要性，它如何影响搜索过程，以及在设计图模型时您需要牢记的关键考虑因素。

## 定义节点和关系类型时的注意事项

回想一下第三章，任何 Neo4j 图的基础都是建立在**节点**和**关系**之上的。节点代表实体，如电影或人物（例如，演员或导演），而关系定义了这些实体如何连接。您选择的节点和关系类型在确定搜索查询的有效性方面起着至关重要的作用。

在电影数据集中，节点可以传统地表示不同的实体，如`Movies`、`Actors`、`Directors`和`Genres`。关系随后定义这些节点如何交互，如`ACTED_IN`、`DIRECTED`或`BELONGS_TO`。然而，有一种替代方法，通常更有效——将相似实体合并为单个节点类型。

你不需要为`Actors`和`Directors`创建单独的节点，你可以创建一个单一的`Person`节点。每个`Person`节点的特征——无论是演员、导演还是两者都是——由它与`Movie`节点的关系类型定义。例如，通过`ACTED_IN`关系连接到`Movie`节点的`Person`节点表示该人是该电影中的演员。同样，`DIRECTED`关系表示该人执导了该电影。我们将在接下来的章节中创建完整的图。

但首先，让我们谈谈为什么这种方法更好。正如我们在*第三章*中展示的那样，这种方法导致以下结果：

+   **简化的数据模型**: 通过使用单个`Person`节点来表示演员和导演，你的数据模型变得更加精简。这降低了图的复杂性，使其更容易理解和维护。

+   **增强查询性能**: 由于节点类型较少，图数据库在查询期间可以更有效地遍历关系。这是因为数据库引擎有更少的独特实体需要区分，从而缩短查询执行时间。

+   **减少冗余**: 通过统一的`Person`节点，消除了信息重复的需求。在一个人既是演员又是导演的情况下，你可以避免创建两个具有重叠数据的独立节点，从而最小化冗余，节省存储空间。

+   **灵活的关系定义**: 这种方法允许更灵活和细粒度的关系定义。如果一个人在多部电影中扮演多个角色（例如，在一部电影中担任演员，在另一部电影中担任导演），关系可以清楚地区分这些角色，而无需创建多个节点。

+   **易于维护和扩展**: 随着你的数据集增长，维护更简单的节点结构变得越来越重要。当你使用统一的节点类型工作时，添加新的角色或关系变得更加直接。

通过仔细选择和定义这些类型和关系，你创建了一个反映现实世界联系的图结构。这使得你的搜索查询更加直观，结果更有意义，整个系统更加高效。

## 应用索引和约束对搜索性能的影响

随着你的 Neo4j 图数据库增长，**索引**和**约束**的应用变得至关重要。**索引**允许 Neo4j 快速定位查询的起点，极大地提高了搜索性能，尤其是在大型数据集中。然而，约束通过防止创建重复节点或无效关系来确保数据完整性。

在我们的电影数据集的背景下，我们使用一个统一的`Person`节点来表示演员和导演，索引变得尤为重要。你可以根据诸如`person_name`或`role`等属性来**索引**节点，确保对特定人物或他们在电影中的角色的搜索能够迅速返回结果。例如，你可以对关系（例如，`ACTED_IN`或`DIRECTED`）上的角色属性进行索引，以便快速过滤参与特定电影的人物。

约束对于维护图形的完整性也是必不可少的。让我们看看这些约束中的一些。这些约束应根据数据集的性质和应用需求仔细设计——它们不是一刀切解决方案。

以下是一些示例语句，展示了如何为电影数据集创建定制的约束和索引。这些示例包括确保人员标识符的唯一性和优化节点和关系属性上的搜索性能的常见场景。根据你的具体用例和数据质量，你可以调整这些模式以强制执行数据完整性和提高查询速度：

+   对`person_name`的唯一约束（对于简化用例）。在许多情况下——例如我们的电影数据集，我们假设每个人都有一个独特的名字——你可能会对`person_name`属性施加唯一约束，以确保即使他们在不同的电影中扮演多个角色（例如，演员和导演），每个个体也只由一个节点表示。以下是你可以这样做的示例：

    ```py
    CREATE CONSTRAINT unique_person_name IF NOT EXISTS
    FOR (p:Person)
    REQUIRE p.person_name IS UNIQUE; 
    ```

这有助于防止意外创建重复节点，并保持你的图形干净高效。

+   对更可靠的 ID（例如，`person_id`）的唯一约束。在前面的场景中，唯一约束是基于对数据的假设。在现实世界的场景中，遇到具有相同名字的不同个体是很常见的。

在这种情况下，你应该使用更可靠的标识符，例如来自外部源（例如，**互联网电影数据库**（**IMDb**）或**TMDb**）的`person_id`值，以确保唯一性。以下 Cypher 代码展示了如何实现这一点：

```py
CREATE CONSTRAINT unique_person_id IF NOT EXISTS
FOR (p:Person)
REQUIRE p.person_id IS UNIQUE; 
```

+   在`person_name`上建立索引（如果未强制执行唯一性，则用于更快地查找）。如果你没有强制执行唯一性，但仍然经常按名称搜索人物，对`person_name`属性建立索引可以显著提高查询性能。这允许 Neo4j 根据其名称快速定位`Person`节点：

    ```py
    CREATE INDEX person_name_index IF NOT EXISTS
    FOR (p:Person)
    ON (p.person_name); 
    ```

+   在`Movie`的`title`属性上建立索引。电影通常按标题查询——特别是在推荐系统或搜索功能中。对`title`属性建立索引确保当用户搜索特定电影时能够快速查找：

    ```py
    CREATE INDEX movie_title_index IF NOT EXISTS
    FOR (m:Movie)
    ON (m.title); 
    ```

+   在`ACTED_IN`关系中的`role`属性上建立索引。如果你的应用程序需要通过电影中演员的具体角色进行过滤（例如，主角或客串），在`ACTED_IN`关系上对`role`属性建立索引可以帮助加快这些查询，避免对所有关系进行全扫描：

    ```py
    CREATE INDEX acted_in_role_index IF NOT EXISTS
    FOR ()-[r:ACTED_IN]-()
    ON (r.role); 
    ```

    **注意**

    Neo4j 仅支持版本`5.x`及以上版本的关系属性索引。

正确实现的索引和约束使你的图更加健壮，搜索过程更快、更可靠。这不仅提升了用户体验，还减少了系统上的计算负载，允许实现更可扩展的解决方案。

在下一节中，我们将探讨如何通过利用电影数据集来构建你的图来发挥开放数据的力量。

# 利用电影数据集

在本节中，我们将专注于利用**TMDb**，这是一个在 Kaggle 上提供的综合元数据集合：[`www.kaggle.com/datasets/rounakbanik/the-movies-dataset/`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/)。这个数据集包含了关于电影的各种信息，如标题、类型、演员阵容、制作团队、上映日期和评分。这个数据集包含超过 45,000 部电影及其制作人员的详细信息，为构建一个能够捕捉电影行业复杂关系的 Neo4j 图提供了一个坚实的基础。

你将使用这个数据集来将数据建模为知识图谱，在一个实际的应用场景中学习数据集成。你将学习如何获取、准备并将这些数据导入 Neo4j。

当处理像 TMDb 这样的大型数据集时，在将其集成到你的 Neo4j 图之前确保数据是清洁的、一致的并且结构良好至关重要。原始数据虽然信息丰富，但往往包含不一致性、冗余和复杂的结构，这些可能会阻碍知识图谱的性能和准确性。这就是数据规范化和清理发挥作用的地方。

## 为什么需要规范化和清理数据？

在构建 Neo4j 图时，保持数据集的清洁和规范化至关重要，因为它直接影响到应用程序的质量和性能。通过规范化和清理数据，你确保了数据的一致性，提高了效率，并为分析创建了一个可扩展的基础。以下是为什么每个步骤都很重要的原因：

+   **一致性**：原始数据可能存在记录相似信息的方式上的变化。例如，电影类型可能以不同的格式列出或包含重复项。规范化数据确保相似的数据点以一致格式记录，这使得查询和分析更加容易。然而，在现实世界的数据集中处理这些问题可能具有挑战性。Neo4j 通过强大的功能如 Cypher 模式匹配、用于合并节点和清理重复项的 APOC 过程以及包含节点相似度算法以识别和合并相关实体的 Graph Data Science 库，帮助解决实体链接和去重等问题。这些功能使您能够构建一个干净、可靠的图，反映数据的真实结构。

+   **效率**：规范化数据减少了冗余，这可以提高 Neo4j 图的效率。通过将数据组织成标准化的格式，您可以最小化存储需求并优化查询性能。

+   **准确性**：清理数据涉及删除或纠正不准确记录。这一步骤对于确保从您的图中得出的见解基于准确和可靠的数据至关重要。

+   **可扩展性**：一个干净且规范化的数据集更容易进行扩展。随着数据集的增长，保持标准化的结构可以确保图在不断增加的负载下保持可管理并表现良好。

让我们继续清理和规范化 CSV 文件。

## 清理和规范化 CSV 文件

现在，我们将清理和规范化 TMDb 中包含的每个 CSV 文件。我们数据集中的可用 CSV 文件如下：

+   `credits.csv`：此文件包含关于我们数据集中每部电影演员和制作团队的详细信息，以字符串化的 JSON 对象形式呈现。就我们的目的而言，我们将专注于提取与角色、演员、导演和制片人相关的相关细节：

    ```py
    # Load the CSV file
    df = pd.read_csv('./raw_data/credits.csv')
    # Function to extract relevant cast information
    def extract_cast(cast_str):
        cast_list = ast.literal_eval(cast_str)
        return [
            {
                'actor_id': c['id'],
                'name': c['name'],
                'character': c['character'],
                'cast_id': c['cast_id']
            }
            for c in cast_list
        ]
    # Function to extract relevant crew information
    def extract_crew(crew_str):
        crew_list = ast.literal_eval(crew_str)
        relevant_jobs = ['Director', 'Producer']
        return [
            {
                'crew_id': c['id'],
                'name': c['name'],
                'job': c['job']
            }
            for c in crew_list if c['job'] in relevant_jobs
        ]
    # Apply the extraction functions to each row
    df['cast'] = df['cast'].apply(extract_cast)
    df['crew'] = df['crew'].apply(extract_crew)
    # Explode the lists into separate rows
    df_cast = df.explode('cast').dropna(subset=['cast'])
    df_crew = df.explode('crew').dropna(subset=['crew'])
    # Normalize the exploded data
    df_cast_normalized = pd.json_normalize(df_cast['cast'])
    df_crew_normalized = pd.json_normalize(df_crew['crew'])
    # Reset index to avoid duplicate indices
    df_cast_normalized = df_cast_normalized.reset_index(drop=True)
    df_crew_normalized = df_crew_normalized.reset_index(drop=True)
    # Drop duplicate rows if any
    df_cast_normalized = df_cast_normalized.drop_duplicates()
    df_crew_normalized = df_crew_normalized.drop_duplicates()
    # Add the movie ID back to the normalized DataFrames
    df_cast_normalized['tmdbId'] = df_cast.reset_index(drop=True)['id']
    df_crew_normalized['tmdbId'] = df_crew.reset_index(drop=True)['id']
    # Save the normalized data with the updated column names
    df_cast_normalized.to_csv(
        os.path.join(output_dir, 'normalized_cast.csv'),
        index=False
    )
    df_crew_normalized.to_csv(
        os.path.join(output_dir, 'normalized_crew.csv'),
        index=False
    )
    # Display a sample of the output for verification
    print("Sample of normalized cast data:")
    print(df_cast_normalized.head())
    print("Sample of normalized crew data:")
    print(df_crew_normalized.head()) 
    ```

+   `keywords.csv`：此文件包含数据集中每部电影的剧情关键词。这些关键词对于对电影中的主题元素进行分类和识别至关重要，可用于各种目的，例如搜索、推荐和内容分析：

    ```py
    # Load the CSV file
    df = pd.read_csv('./raw_data/keywords.csv')  # Update the path as necessary
    # Function to extract and normalize keywords
    def normalize_keywords(keyword_str):
        if pd.isna(keyword_str) or not isinstance(keyword_str, str):  # Check if the value is NaN or not a string
            return []
        # Convert the stringified JSON object into a list of dictionaries
        keyword_list = ast.literal_eval(keyword_str)
        # Extract the 'name' of each keyword and return them as a list
        return [kw['name'] for kw in keyword_list]
    # Apply the normalization function to the 'keywords' column
    df['keywords'] = df['keywords'].apply(normalize_keywords)
    # Combine all keywords for each tmdbId into a single row
    df_keywords_aggregated = df.groupby('id', as_index=False).agg({
        'keywords': lambda x: ', '.join(sum(x, []))
    })
    # Rename the 'id' column to 'tmdbId'
    df_keywords_aggregated.rename(
        columns={'id': 'tmdbId'}, inplace=True
    )
    # Save the aggregated DataFrame to a new CSV file
    df_keywords_aggregated.to_csv(
        os.path.join(output_dir, 'normalized_keywords.csv'),
        index=False
    )
    # Display the first few rows of the aggregated DataFrame for verification
    print(df_keywords_aggregated.head()) 
    ```

+   `links.csv`：此文件包含将全**MovieLens 数据集**中的每部电影与其在 TMDb 和 IMDB 中的对应条目链接的必要元数据。此文件作为连接 MovieLens 数据集与外部电影数据库的关键桥梁，实现了数据集成和进一步分析的丰富化。然而，对于此用例，我们跳过了处理`links.csv`文件，因为它对我们当前的分析不是必需的。我们的重点将保持在其他与我们的项目目标更直接相关的 CSV 文件上。`links.csv`中的数据对于需要与外部数据库集成的未来项目仍然可能是有用的，但在此实例中不会使用。

+   `links_small.csv`: 这个文件包含来自完整 MovieLens 数据集的 9,000 部电影的 TMDb 和 IMDb IDs 的子集。虽然这个文件为较小电影集合提供了简化的链接版本，但我们不会使用这个文件，因为我们已经使用了 Kaggle 提供的完整数据集，其中包含所有可用的电影。这个文件通常在需要更易于管理的较小数据集的场景中很有用，但出于我们的目的，完整的数据库更适合进行综合分析和集成。

+   `movies_metadata.csv`: 这个文件是一个包含 45,000 部电影详细信息的全面数据集，这些电影出现在完整的 MovieLens 数据集中。该文件包括各种功能，如海报、背景、预算、收入、上映日期、语言、制作国家和公司等。为了有效地组织和分析这些数据，我们将 `movies_metadata.csv` 文件归一化成多个 CSV 文件，每个文件代表数据集中一个相关的节点。这些节点包括流派、制作公司、制作国家和配音语言。通过将这些数据分解成单独的文件，我们可以更轻松地管理和利用数据集中包含的丰富信息。让我们看看如何操作。

    1.  开始必要的导入。

        ```py
        import pandas as pd
        import ast
        # Load the CSV file
        df = pd.read_csv('./raw_data/movies_metadata.csv')  # Update the path as necessary 
        ```

    1.  提取并归一化流派、制作公司、国家和配音语言。我们将为流派和制作公司演示这一步骤。其余的代码可在 [`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/tree/main/ch4`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/tree/main/ch4) 上找到。

        ```py
        # Function to extract and normalize genres
        def extract_genres(genres_str):
            if pd.isna(genres_str) or not isinstance(
                genres_str, str
            ):
                return []
            genres_list = ast.literal_eval(genres_str)
            return [
                {'genre_id': int(g['id']), 'genre_name': g['name']}
                for g in genres_list
            ]
        # Function to extract and normalize production companies
        def extract_production_companies(companies_str):
            if pd.isna(companies_str) or not isinstance(
                companies_str, str
            ):
                return []
            companies_list = ast.literal_eval(companies_str)
            if isinstance(companies_list, list):
                return [
                    {'company_id': int(c['id']),
                        'company_name': c['name']
                    }
                    for c in companies_list
                ]
            return [] 
        ```

    1.  应用提取函数。

        ```py
        df['genres'] = df['genres'].apply(extract_genres)
        df['production_companies'] = \
            df['production_companies'].apply(
                extract_production_companies
            )
        df['production_countries'] = \
            df['production_countries'].apply(
                extract_production_countries
            )
        df['spoken_languages'] = df['spoken_languages'].apply(
            extract_spoken_languages
        )
        # Explode lists into rows
        df_genres = df.explode('genres').dropna(subset=['genres'])
        df_companies = df.explode('production_companies').dropna(
            subset=['production_companies']
        )
        df_countries = df.explode('production_countries').dropna(
            subset=['production_countries']
        )
        df_languages = df.explode('spoken_languages').dropna(
            subset=['spoken_languages']
        ) 
        ```

    1.  归一化展开后的数据。让我们先对流派进行操作。

        ```py
        df_genres_normalized = pd.json_normalize(df_genres['genres'])
        # Reset index to avoid duplicate indices
        df_genres_normalized = \
            df_genres_normalized.reset_index(drop=True)
        # Add the movie ID back to the normalized DataFrames as 'tmdbId'
        df_genres_normalized['tmdbId'] = df_genres.reset_index(
            drop=True
        )['id']
        # Ensure that 'company_id' and similar fields are treated as integers
        df_genres_normalized['genre_id'] = \
            df_genres_normalized['genre_id'].astype(int)
        # Save the normalized data with the updated column names
        df_genres_normalized.to_csv(
            os.path.join(output_dir, 'normalized_genres.csv'),
            index=False
        ) 
        ```

    1.  接下来，提取集合名称。

        ```py
        # For the movies, including "Belongs to Collection" within the same CSV
        # Extract only the "name" from "belongs_to_collection" and include additional fields
        def extract_collection_name(collection_str):
            if isinstance(collection_str, str):
                try:
                    collection_dict = \
                        ast.literal_eval(collection_str)
                    if isinstance(collection_dict, dict):
                        return collection_dict.get('name', "None")
                except (ValueError, SyntaxError):  # Handle cases where string parsing fails
                    return "None"
            return "None"
        df_movies = df[
            [
                'id', 'original_title', 'adult', 'budget', 'imdb_id',
                'original_language', 'revenue', 'tagline', 'title',
                'release_date', 'runtime', 'overview',
                'belongs_to_collection'
            ]
        ].copy()
        df_movies['belongs_to_collection'] = \
            df_movies['belongs_to_collection'].apply(
                extract_collection_name
            )
        df_movies['adult'] = df_movies['adult'].apply(
            lambda x: 1 if x == 'TRUE' else 0
        )  # Convert 'adult' to integer
        # Rename 'id' to 'tmdbId'
        df_movies.rename(columns={'id': 'tmdbId'}, inplace=True)  # Rename 'id' to 'tmdbId'
        # Save the movies to a separate CSV, including the extracted fields
        df_movies.to_csv(
            './normalized_data/normalized_movies.csv', index=False
        ) 
        ```

+   `ratings.csv`: 这个文件是完整的 MovieLens 数据集，包含 2600 万条评分和 75 万个标签应用，来自 27 万名用户对数据集中所有 45,000 部电影的评分。这个全面的数据集提供了详细的用户交互数据，我们将直接使用这些数据，无需进行归一化处理。然而，对于这个用例，我们决定跳过处理 `ratings.csv` 文件。虽然它提供了广泛的用户交互数据，但对于我们当前的分析和目标来说并非必需。我们正在关注其他与我们的项目更直接相关的 CSV 文件。`ratings.csv` 中的数据对于未来需要深入挖掘用户评分和交互的项目仍然有价值，但在这个实例中不会使用。

+   `ratings_small.csv`: 这个文件是 `ratings.csv` 文件的较小子集，包含 700 名用户对 9,000 部电影的 10 万条评分。我们将使用 `ratings_small.csv` 而不是关注 `ratings.csv` 中提供的完整数据集。

通过这个过程，我们学习了如何将原始的非结构化数据转换为干净、规范化的数据集，这些数据集现在已准备好集成到您的 Neo4j 图中。这种准备为构建一个强大、高效和有效的 AI 驱动的搜索和推荐系统铺平了道路。在下一节中，我们将使用这些规范化的 CSV 文件，并通过 Cypher 代码构建知识图谱，释放我们数据集的全部潜力。

# 使用代码示例构建您的电影知识图谱

在本节中，我们将导入您的标准化数据集到 Neo4j，并将它们转换成完全功能的知识图谱。

## 设置您的 AuraDB 免费实例

要开始使用 Neo4j 构建您的知识图谱，您首先需要设置一个 AuraDB Free 实例。AuraDB Free 是一个云托管的 Neo4j 数据库，它允许您快速开始，无需担心本地安装或基础设施管理。

按照以下步骤创建您的实例：

1.  访问 [`console.neo4j.io`](https://console.neo4j.io)。

1.  使用您的 Google 账户或电子邮件登录。

1.  点击**创建免费实例**。

1.  在实例配置过程中，将出现一个弹出窗口，显示您的数据库连接凭据。

确保从弹出窗口中下载并安全保存以下详细信息——这些信息对于将您的应用程序连接到 Neo4j 是必不可少的：

```py
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-generated-password>
AURA_INSTANCEID=<your-instance-id>
AURA_INSTANCENAME=<your-instance-name> 
```

在您的 AuraDB Free 实例设置完成后，您现在可以导入您的标准化数据集，并开始使用 Cypher 代码构建您的知识图谱。在下一节中，我们将指导您导入数据并在您的图中构建关系。

## 将数据导入 AuraDB

现在，您的 AuraDB Free 实例已经启动并运行，是时候导入您的标准化数据集并构建您的知识图谱了。在本节中，我们将通过一个 Python 脚本指导您准备 CSV 文件、设置索引和约束、导入数据以及创建关系。

1.  准备您的 CSV 文件以供导入。

1.  确保您生成的 CSV 文件（例如，`normalized_movies.csv`、`normalized_genres.csv` 等）已准备好导入。这些文件应该是干净的、结构良好的，并且托管在可访问的 URL 上。在这种情况下，`graph_build.py` 脚本从公共云存储（例如，[`storage.googleapis.com/movies-packt/normalized_movies.csv`](https://storage.googleapis.com/movies-packt/normalized_movies.csv)）获取文件，因此您不需要手动上传它们到任何地方。

1.  添加索引和约束以优化图查询检索。

在加载数据之前，创建唯一约束和索引对于确保完整性和优化查询性能至关重要。该脚本包括用于以下操作的 Cypher 命令：

+   确保 `tmdbId`、`movieId` 和 `company_id` 等 ID 的唯一性

+   在 `actor_id`、`crew_id` 和 `user_id` 等属性上创建索引

下面是如何创建索引和约束的说明：

```py
"CREATE CONSTRAINT unique_tmdb_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.tmdbId IS UNIQUE;",
"CREATE CONSTRAINT unique_movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE;",
"CREATE CONSTRAINT unique_prod_id IF NOT EXISTS FOR (p:ProductionCompany) REQUIRE p.company_id IS UNIQUE;",
"CREATE CONSTRAINT unique_genre_id IF NOT EXISTS FOR (g:Genre) REQUIRE g.genre_id IS UNIQUE;",
"CREATE CONSTRAINT unique_lang_id IF NOT EXISTS FOR (l:SpokenLanguage) REQUIRE l.language_code IS UNIQUE;",
"CREATE CONSTRAINT unique_country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.country_code IS UNIQUE;",
"CREATE INDEX actor_id IF NOT EXISTS FOR (p:Person) ON (p.actor_id);",
"CREATE INDEX crew_id IF NOT EXISTS FOR (p:Person) ON (p.crew_id);",
"CREATE INDEX movieId IF NOT EXISTS FOR (m:Movie) ON (m.movieId);",
"CREATE INDEX user_id IF NOT EXISTS FOR (p:Person) ON (p.user_id);" 
```

1.  导入数据并创建节点。

在添加约束和索引后，脚本从各自的 CSV 文件中加载节点：

+   `load_movies()` 添加所有电影元数据。

+   `load_genres()`、`load_production_companies()`、`load_countries()` 等创建相关的节点，例如 `Genre`、`ProductionCompany`、`Country` 和 `SpokenLanguage`。

+   使用 `load_person_actors()` 和 `load_person_crew()` 添加与人物相关的数据。

通过 `load_links()`、`load_keywords()` 和 `load_ratings()` 添加额外的属性。

以下是一个示例：

```py
graph.load_movies('https://storage.googleapis.com/movies-packt/normalized_movies.csv', movie_limit) 
```

1.  创建关系。

随着每个加载函数的运行，它不仅创建节点，还建立有意义的关联：

+   `HAS_GENRE` 在 `Movie` 和 `Genre` 之间。

+   `PRODUCED_BY` 在 `Movie` 和 `ProductionCompany` 之间。

+   `HAS_LANGUAGE` 在 `Movie` 和 `SpokenLanguage` 之间，`PRODUCED_IN` 在 `Movie` 和 `Country` 之间，`ACTED_IN`、`DIRECTED`、`PRODUCED` 在 `Movie` 和 `Person` 之间，以及 `RATED` 在 `Movie` 和 `User` 之间，等等。

1.  运行完整脚本。

在运行脚本之前，请确保您已安装 Neo4j Python 驱动程序。您可以使用 `pip` 安装它。

```py
 pip install neo4j 
```

要运行整个图构建过程，只需执行以下操作：

```py
python graph_build.py 
```

此脚本按以下顺序执行以下操作：

+   使用 `.env` 文件中的凭据连接到您的 AuraDB 实例。

+   清理数据库。

+   添加索引和约束。

+   使用托管 CSV 文件批量加载所有节点数据和关系。

请参阅此处提供的完整脚本：[`github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch4/graph_build.py`](https://github.com/PacktPublishing/Building-Neo4j-Powered-Applications-with-LLMs/blob/main/ch4/graph_build.py)。

完成后，使用 Neo4j 浏览器验证您的导入：

```py
MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
RETURN m.title, g.genre_name
LIMIT 10; 
```

*图 4.1* 展示了一个包含超过 90K 个节点和 320K+ 关联的连接电影图。例如 `Movie`、`Genre`、`Person` 和 `ProductionCompany` 这样的节点用不同的颜色表示，而例如 `ACTED_IN`、`HAS_GENRE` 和 `PRODUCED_BY` 这样的关系展示了相互关联的元数据网络。

![图 4.1 — 电影数据集的 Neo4j 图](img/B31107_04_01.png)

图 4.1 — 电影数据集的 Neo4j 图。

使用 Python 和 Cypher 成功导入数据并构建完知识图谱后，您现在可以开始构建一个由 GenAI 驱动的搜索应用了。在下一章中，我们将深入探讨高级 Cypher 技术，这些技术能帮助您处理复杂的关系并从数据中获得更深入的见解。

# 除此之外：复杂图结构的 Cypher 高级技术。

随着您的知识图谱在规模和复杂性上的增长，对您的查询和数据管理能力的需求也在增加。Cypher，Neo4j 强大的查询语言，提供了一系列高级功能，旨在处理复杂的图结构并实现更复杂的数据分析。在本节中，我们将探索这些高级 Cypher 技术，包括**路径模式**、**可变长度关系**、子查询和图算法。理解这些技术将帮助您有效地管理复杂的关系，进行更深入的分析，并释放您的知识图谱在高级用例中的全部潜力。

让我们探索这些关键的 Cypher 高级技术：

+   **可变长度的关系**：Cypher 中的可变长度关系允许您在节点之间匹配不同长度的路径。这在探索层次结构或具有多个分离度的网络时特别有用。例如，找到与特定演员在三个分离度范围内的所有电影：

    ```py
    MATCH (a:Actor {name: 'Tom Hanks'})-[:ACTED_IN*1..3]-(m:Movie)
    RETURN DISTINCT m.title; 
    ```

    +   在这里，`*1..3`指定了关系路径的长度可以在 1 到 3 步之间。

    +   **用例**：可变长度关系非常适合社交网络分析等场景，您想要找到在特定连接度内的人，或者在具有多级父子关系的层次数据集中探索父子关系。

+   **使用路径模式进行模式匹配**：您可以在 Neo4j 中创建**命名路径模式**以及链式路径。

    +   **定义路径模式**：Cypher 允许您定义可重用的命名路径模式，这些模式可以在查询中多次使用。这使得您的查询更易于阅读，并允许您将复杂的关系封装在单个模式中。以下是一个示例：

        ```py
        MATCH path = (a:Actor)-[:ACTED_IN]->(m:Movie)
        RETURN path; 
        ```

    在这里，`path`是一个命名路径模式，可以在后续操作或子查询中重复使用。

    +   **链式路径模式**：Cypher 允许您组合多个路径模式，在图内执行复杂的遍历。这在试图揭示间接关系或发现满足特定标准的多条路径时特别有用。

    一个例子是探索电影数据集中的合作情况。

    假设我们想要找到演员与导演合作过的电影，而这些导演之前可能通过另一部电影与他们合作过。这涉及到从演员到电影，再到导演的路径链，并查看是否存在另一部电影连接相同的演员-导演对：

    ```py
    MATCH (a:Actor {name: "Tom Hanks"})-[:ACTED_IN]->(m1:Movie)<-[:DIRECTED_BY]-(d:Director) MATCH (a)-[:ACTED_IN]->(m2:Movie)<-[:DIRECTED_BY]-(d)
    WHERE m1 <> m2
    RETURN a.name AS actor, d.name AS director, collect(DISTINCT m1.title) + collect(DISTINCT m2.title) AS movies 
    ```

    这种模式链在识别专业关系、重复合作或分析网络中的间接影响方面非常有帮助。

+   **子查询**和**过程逻辑**：您可以使用子查询和过程来处理复杂查询。以下是操作方法：

    +   **使用子查询进行模块化查询**：Cypher 中的子查询允许您将复杂的查询分解成模块化、可重用的组件。这在处理大型图或需要对同一数据集执行多个操作时特别有用。以下是一个示例：

        ```py
        CALL {
          MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: 'Action'})
          RETURN m
        }
        MATCH (m)-[:DIRECTED_BY]->(d:Director)
        RETURN d.name, COUNT(m) AS action_movies_directed; 
        ```

    在这里，子查询检索所有动作电影，而外部查询将这些电影与它们的导演匹配。

    +   **使用 CALL 执行过程逻辑**：Cypher 中的 `CALL` 子句允许您调用过程并在后续查询中使用这些结果。这对于高级数据处理至关重要，例如运行图算法或调用自定义过程。

    我们已经在 `graph_build.py` 文件中的实现中应用了这种方法，特别是在 `load_ratings()` 函数中。在这里，我们使用 `CALL { ... } IN TRANSACTIONS` 模式，通过以 50,000 行的块来处理数据，有效地加载大量数据集：

    ```py
    LOAD CSV WITH HEADERS FROM $csvFile AS row
    CALL (row) {
      MATCH (m:Movie {movieId: toInteger(row.movieId)})
      WITH m, row
      MERGE (p:Person {user_id: toInteger(row.userId)})
      ON CREATE SET p.role = 'user'
      MERGE (p)-[r:RATED]->(m)
      ON CREATE SET r.rating = toFloat(row.rating), r.timestamp = toInteger(row.timestamp)
    } IN TRANSACTIONS OF 50000 ROWS; 
    ```

    这种方法使我们能够在保持性能和事务完整性的同时处理大量的 CSV 导入——这是 `CALL` 在现实世界图应用中的许多强大用例之一。

+   **处理嵌套查询**：在复杂的图结构中，您可能需要**组合多个查询的结果**。Cypher 允许您嵌套查询，将一个查询的结果传递给另一个查询，这对于基于多个标准过滤或细化结果非常有用。以下是一个示例：

    ```py
    MATCH (m:Movie)
    WHERE m.revenue > 100000000
    CALL {
      WITH m
      MATCH (m)-[:HAS_GENRE]->(g:Genre)
      RETURN g.name AS genre
    }
    RETURN m.title, genre; 
    ```

在这里，嵌套查询通过根据收入过滤电影来细化结果，然后找到它们相关的类型。

这些 Cypher 技巧使您能够应对复杂的图结构，实现更深入的洞察和更复杂的分析。您可以参考[`neo4j.com/docs/cypher-manual/current/appendix/tutorials/advanced-query-tuning/`](https://neo4j.com/docs/cypher-manual/current/appendix/tutorials/advanced-query-tuning/)以进一步探索这些技巧。

# 摘要

在本章中，我们致力于将原始的半结构化数据转换为干净、规范化的数据集，以便将其集成到我们的知识图谱中。然后，我们探讨了图建模的最佳实践，重点关注如何构建节点和关系以增强搜索效率，并确保您的图保持可扩展和高效。在此之后，我们探讨了其他 Cypher 技巧，为您提供处理可变长度关系、模式匹配、子查询和图算法的技能。您现在已准备好构建一个由知识图谱驱动的搜索，它可以处理甚至最复杂的数据关系。

在下一章中，我们将进一步探索如何将 Haystack 集成到 Neo4j 中。这本实用指南将向您展示如何在您的知识图谱中构建强大的搜索功能，让您能够充分利用 Neo4j 和 Haystack 的全部潜力，以实现智能搜索解决方案。
