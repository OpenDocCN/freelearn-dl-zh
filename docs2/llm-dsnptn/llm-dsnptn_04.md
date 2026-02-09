# 4

# 处理 LLM 训练中的大规模数据集

在本章中，你将学习管理和处理大规模数据集的高级技术，这对于训练最先进的 LLMs 至关重要。我们将探讨大规模语言数据集带来的独特挑战，并提供解决这些挑战的实际方案。

本章的目标是让你掌握处理大规模数据的知识和工具，从而能够训练更强大、更有效的 LLMs。

本章我们将涵盖以下主题：

+   大数据集的挑战

+   数据采样技术

+   分布式数据处理

+   数据分片和并行化策略

+   高效的数据存储格式

+   流式数据处理以实现持续的 LLM 训练

+   内存高效的数据加载技术

# 大数据集的挑战

训练 LLMs 需要巨大的数据集，通常在千兆或甚至太字节范围内。这种规模引入了几个挑战：

+   **存储需求**：数据集可能超过单台机器的容量，需要分布式存储解决方案。

+   **输入/输出（I/O）瓶颈**：读取大量数据可能成为显著的瓶颈，限制训练速度。

+   **预处理开销**：由于处理大量文本数据需要通过多个顺序操作的计算开销，分词和其他预处理步骤在规模上可能耗时。挑战来自于需要对每段文本执行多个步骤——分词、规范化、清理、语言检测以及其他转换——这些步骤在数百万或数十亿个文本样本上成倍增加。这个过程本质上是顺序的（每一步都依赖于前一步），需要 CPU/内存资源，并可能涉及复杂的操作，如**正则表达式**（**regexes**）、字典查找和特定语言规则。当处理多语言或代码混合数据时，复杂性进一步增加，因为需要应用不同的语言规则，并且需要对每个文本段进行额外的步骤，如脚本规范化或语言检测，这使得预处理管道成为大规模**自然语言处理**（**NLP**）系统的一个显著瓶颈。

+   **内存限制**：将整个数据集加载到内存中通常是不切实际的，需要流式或批处理方法。

+   **数据质量和多样性**：随着数据集规模的增加，确保数据集质量和代表性变得更加困难。

为了应对这些挑战，我们需要采用复杂的数据处理技术。让我们通过使用 Hugging Face 的**Datasets**库的 Python 实现来探索这些技术，该库旨在高效地处理大规模数据集：

```py
from datasets import load_dataset, Dataset
import psutil
def load_and_process_large_dataset(dataset_name, num_proc):
    # Load the dataset
    dataset = load_dataset(dataset_name, streaming=True)
    # Define a preprocessing function
    def preprocess_function(examples):
        # Implement your preprocessing logic here
        return examples
    # Apply preprocessing in parallel
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset["train"].column_names
    )
    return processed_dataset
#Determine the number of CPU cores for parallel processing
num_cores = psutil.cpu_count(logical=False)
#Load and process a large dataset (e.g., C4 dataset)
large_dataset = load_and_process_large_dataset("c4", 
    num_proc=num_cores)
#Print the first few examples
for example in large_dataset["train"].take(5):
    print(example)
```

在此代码中，我们使用 Datasets 库高效地加载和处理大型数据集（在这种情况下，是 `C4` 数据集）。`num_proc` 参数指定用于数据集映射操作中的并行处理所使用的处理器核心数。在预处理大型数据集时，通过并行处理使用多个 CPU 核心可以显著加快操作速度。例如，如果 `num_proc=4`，则预处理函数将在四个处理器核心上同时执行，并行处理不同的数据批次，而不是顺序处理。

为了更好地理解大型数据集的使用上下文，探索一个具体的例子很有帮助。前述代码片段中使用的一个此类数据集是 **Colossal Clean Crawled Corpus** （**C4**）数据集，它在现代 LLM 的训练中发挥着重要作用。

**C4** 数据集是由 Google 创建的一个庞大的、经过清洗的网页爬取文本语料库，用于训练 LLM。包含大约 750 GB 的英语文本，C4 是从 Common Crawl 数据中提取的，并经过广泛的过滤以去除重复内容、非英语内容和冒犯性材料。它有几个变体，包括标准清洗版本、未过滤版本以及专注于新闻类内容的子集。虽然公开可用，但访问 C4 需要一些努力，通常通过 Google Cloud Storage 或 Hugging Face datasets 等库进行。尽管经过清洗过程，C4 在内容质量和潜在偏差方面仍存在一些局限性，研究人员在使用它进行模型训练时应予以考虑。尽管如此，它仍然是 NLP 任务中的一个宝贵资源，并在训练像 **Text-to-Text Transfer Transformer** （**T5**）和 **Language Model for Dialogue Applications** （**LaMDA**）这样的突出模型中发挥了关键作用。

我们采用流式处理以避免一次性将整个数据集加载到内存中。`num_proc` 参数设置为物理 CPU 核心数，以最大化并行处理效率。

`preprocess_function` 函数是您实现特定数据集预处理逻辑的地方。此函数在数据集上并行应用，显著加快了大型数据集的预处理速度。

您也可以使用 GPU 来完成这项任务。请参阅以下代码示例（请注意，虽然基于 GPU 的预处理在诸如标记化、嵌入生成等操作中特别有益，但它可能不会显著加速简单的文本操作）：

```py
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
def load_and_process_dataset(dataset_name, batch_size):
    dataset = load_dataset(dataset_name, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def preprocess(examples):
        return tokenizer(
            examples["text"], padding="max_length",
            truncation=True, return_tensors="pt"
        )
    def process_batch(batch):
        return {k: v.to(device) for k, v in preprocess(batch).items()}
    return DataLoader(
        dataset["train"].map(process_batch),
        batch_size=batch_size, num_workers=2,
        pin_memory=True
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = load_and_process_dataset("c4", batch_size=32)
for i, batch in enumerate(dataloader):
    if i >= 5: break
    print(f"Batch {i}:", {k: v.shape for k, v in batch.items()})
```

此代码使用 PyTorch 和 Hugging Face 库以 GPU 加速处理数据集（例如，C4）。它使用数据加载器进行高效的批量处理，将数据移动到 GPU 内存，并使用预训练的标记化器。主要的 GPU 优势来自并行批量处理和 GPU 加速的标记化。虽然这种设置使得 GPU 的使用成为可能，但最大的 GPU 优势通常出现在模型训练或推理期间，而不是预处理期间。

# 数据采样技术

数据采样是一种在不牺牲代表性情况下减少大型数据集大小的实用方法。存在几种技术，每种技术都有特定的用例和权衡。**随机采样**从数据集中均匀随机选择数据点。当数据独立同分布时，它简单有效，但如果数据不平衡，可能会错过重要的子群体。**系统采样**在随机起点后从列表中选择每*k*个项。它比随机采样更有结构，当数据以有意义的方式排序时可能很有用，但如果排序与隐藏的周期性模式一致，则可能引入偏差。**蓄水池采样**是为未知大小数据集的流式传输设计的。它在顺序遍历数据的同时维护一个固定大小的样本，并确保每个项目都有相等的机会被包含。这在数据以连续流形式到达的在线或增量学习场景中特别有用。

由于本章篇幅限制，我们仅关注**分层采样**，这是一种在数据集中保持子群体比例表示的技术。它特别适用于某些属性（如标签类别、句子长度或元数据类别）已知会影响模型性能并需要在采样子集中保持的情况。在 NLP 中，文本长度是一个常见的分层变量，因为它对模型输入动态有影响。

以下实现演示了如何根据文本长度应用分层采样。它将数据集划分为基于百分比的层级，并从每个层级按比例采样以创建一个子集，该子集保留了整个数据集的长度分布：

```py
import numpy as np
from datasets import Dataset
def stratified_length_sampling(
    dataset, num_samples, num_strata=10
):
    # Calculate text lengths
    lengths = [len(example['text']) for example in dataset]
    # Create strata based on text length
    strata_bounds = np.percentile(
        lengths, np.linspace(0, 100, num_strata + 1)
    )
    sampled_data = []
    for i in range(num_strata):
        stratum = [
            example for example in dataset
            if strata_bounds[i] <= len(example['text']) < \
                strata_bounds[i+1]
        ]
        stratum_samples = np.random.choice(
            stratum,
            size=num_samples // num_strata,
            replace=False
        )
        sampled_data.extend(stratum_samples)
    return Dataset.from_dict({
        key: [example[key] for example in sampled_data]
        for key in dataset[0].keys()
    })
#Usage
sampled_dataset = stratified_length_sampling(large_dataset, 
    num_samples=100000)
```

这种分层采样技术确保我们在采样数据集中保持文本长度的代表性分布。我们使用 10 个层级（`num_strata=10`）来平衡粒度和计算效率。根据您特定的数据集特征和采样要求调整此值。

随着数据集规模和复杂性的增长，单机处理在速度和可扩展性方面都成为瓶颈。数据采样等技术可以提供部分缓解，但它们不能解决集中式架构固有的计算限制。为了解决这些限制，下一节介绍了分布式数据处理，其中计算分布在多个机器或节点上以提高吞吐量、降低延迟并支持大规模 LLM 训练管道所需的并行工作流程。

# 分布式数据处理

对于真正庞大的数据集，分布式处理变得必要。以下是一个使用**Dask**的例子，这是一个用于 Python 并行计算的灵活库（[`www.dask.org/`](https://www.dask.org/))。

Dask 和 Apache Spark 都是分布式计算框架，但它们的主要区别在于其架构和用例。Spark 围绕**弹性分布式数据集**（**RDDs**）的概念构建，需要集群设置，使其非常适合大规模生产数据处理。另一方面，Dask 旨在无缝集成到 Python 生态系统，可以从单个笔记本电脑扩展到集群，使用与 NumPy、pandas 和 scikit-learn 相似的 API。虽然 Spark 在处理大规模数据集的批处理方面表现出色，但 Dask 在交互式计算和科学工作流程方面更加灵活，尤其是在使用 Python 原生库或需要以最小修改扩展现有 Python 代码时。

让我们回到我们的代码：

```py
import dask.dataframe as dd
from dask.distributed import Client
def distributed_preprocessing(data_path, num_partitions):
    # Initialize Dask client
    client = Client()
    # Read the dataset into a Dask DataFrame
    df = dd.read_csv(data_path, blocksize="64MB")
    # Repartition the data for better distribution
    df = df.repartition(npartitions=num_partitions)
    # Define preprocessing function
    def preprocess(text):
        # Implement your preprocessing logic here
        return processed_text
    # Apply preprocessing in parallel
    df['processed_text'] = df['text'].map(preprocess)
    # Trigger computation and return results
    result = df.compute()
    client.close()
    return result
#Usage
processed_data = distributed_preprocessing(
    "path/to/large/dataset.csv", num_partitions=100
)
```

在这个例子中，我们使用 Dask 将预处理工作负载分布到多台机器或核心上。`num_partitions`参数（设置为`100`）决定了并行化级别，应根据您的可用计算资源和数据集大小进行调整。

# 数据分片和并行化策略

**数据分片**是指将大型数据集分解成更小、更易于管理的块，称为“分片”，然后分布到多台机器或存储系统中的技术。每个分片可以独立处理，这使得处理大型数据集变得更容易，尤其是那些不适合单台机器内存的数据集。这种方法在机器学习中广泛使用，用于分配大型数据集的处理，从而允许训练更大的模型或进行更快的计算。

数据分片使得计算资源的使用更加高效，因为每个分片可以独立处理，并且结果可以在之后进行汇总。

然而，必须仔细考虑确保分片策略保持所有分片间数据分布的完整性和代表性，以避免训练模型中的偏差或不一致性。

这里是一个分片策略的例子：

```py
import hashlib
def shard_data(dataset, num_shards):
    shards = [[] for _ in range(num_shards)]
    for item in dataset:
        # Use a hash function to determine the shard
        shard_index = int(
            hashlib.md5(
                item['id'].encode()
            ).hexdigest(), 16
        ) % num_shards
        shards[shard_index].append(item)
    return shards
#Usage
sharded_data = shard_data(large_dataset, num_shards=10)
```

这种分片策略使用哈希函数将数据项分布到各个分片中。`num_shards`参数（设置为`10`）应根据您的基础设施和并行化需求进行调整。

`shard_data`函数通过应用基于每个项目唯一标识符的一致性哈希方案，将数据集中的项目分配到指定的分片数量中。它初始化一个空列表的列表，每个列表代表一个分片，对于输入数据集中的每个项目，它使用`'id'`字段计算一个分片索引。哈希输出被转换为整数，并取模以确保在分片之间均匀分布。这种方法保证了具有相同 ID 的项目在执行过程中始终映射到相同的分片，这对于分布式存储或并行处理等任务很有用，在这些任务中，确定性和平衡很重要。

分片策略的选择基于数据的性质和预期的查询模式，每种方法在可扩展性、性能和复杂性方面都提供了不同的权衡：

+   **哈希分片**：通过哈希函数映射键以均匀分布负载，适用于均匀分布的数据

+   **范围分片**：适用于有序数据集，如时间序列日志，其中每个分片包含连续的数据值范围

+   **地理分片**：通过根据地理区域分区数据来优化基于位置的查询

+   **键值分片**：通过将特定的键范围或值分配给定义的分片，允许手动控制热点

+   **基于目录的分片**：通过使用查找服务来确定数据放置以支持动态分片分配，适应数据分布的变化

+   **一致性哈希**：当分片数量变化时最小化数据移动，保持稳定性并减少重新平衡开销

+   **轮询分片**：按顺序将数据分配到各个分片中，提供简单性但范围查询性能较差

+   **基于工作负载的分片**：通过根据观察到的查询模式将高流量数据分配到单独的分片中来平衡访问负载

+   **复合分片**：结合多种策略以支持具有多种数据类型和查询需求复杂系统的支持

+   **基于标签的分片**：根据用户角色或数据类别等标签对数据进行分类，支持特定领域的分区策略

对于前面的代码块，我们还可以定义以下函数作为主要协调器来处理和聚合分片：

```py
def process_with_sharding(
    dataset: List[Dict], num_shards: int
) -> List[Dict]:
    # Step 1: Shard the data
    shards = shard_data(dataset, num_shards)
    # Step 2: Process shards in parallel
    with ProcessPoolExecutor(max_workers=num_shards) as executor:
        processed_shards = list(executor.map(process_shard, shards))
    # Step 3: Aggregate results
    aggregated_results = []
    for shard_results in processed_shards:
        aggregated_results.extend(shard_results)
```

`process_with_sharding` 函数接收一个表示为字典列表的数据集，并使用 `shard_data` 函数将其划分为指定数量的分片。然后，它使用 `ProcessPoolExecutor` 并行处理每个分片，每个分片使用 `process_shard` 函数。处理完所有分片后，通过遍历处理过的分片并将它们的内容扩展到最终结果列表中，将每个分片的单个结果聚合到一个列表中。

一旦数据被有效地分区并分配以进行并行处理，就必须关注其物理存储和访问方式——这引出了高效存储格式的选择。

# 高效的数据存储格式

选择正确的存储格式可以显著影响数据加载和处理速度。

例如，我们可以使用 **Apache Parquet** ([`parquet.apache.org/`](https://parquet.apache.org/))，这是一种特别适用于大型数据集的列式存储格式。

下面是一个比较不同列格式及其在存储大型语言数据集特性方面的表格：

| **特性** | **CSV** | **JSON** | **Apache Parquet** | **Apache Arrow** |
| --- | --- | --- | --- | --- |
| **存储类型** | 行式 | 行式 | 列式 | 列式 |
| **压缩** | 基础 | 差 | 优秀 | 优秀 |
| **查询速度** | 慢 | 慢 | 快 | 非常快 |
| **嵌套结构** | 否 | 是 | 是 | 是 |
| **模式支持** | 否 | 有限 | 是 | 是 |
| **随机访问** | 差 | 差 | 好 | 优秀 |
| **内存效率** | 差 | 差 | 好 | 优秀 |
| **Python 集成** | 简单 | 简单 | 好（通过 PyArrow） | 原生 |
| **典型** **用例** | 小数据集 | API 响应 | 大数据分析 | 内存处理 |
| **加载速度** | 慢 | 中等 | 快 | 非常快 |
| **NLP** **特征支持** | 基础 | 好 | 优秀 | 优秀 |
| **跨平台** | 是 | 是 | 是 | 是 |
| **元数据支持** | 不支持 | 有限 | 支持 | 支持 |

表 4.1 – 不同列格式的特性

此表突出了为什么 Parquet 由于其列式存储格式、高效的压缩和强大的对 NLP 任务中常见的复杂数据结构支持，通常被首选用于 LLM 数据集。

下面是一个示例，说明数据通常如何在 Apache Parquet 列中为 NLP 数据集结构化：

| **列名** | **数据类型** | **示例值** |
| --- | --- | --- |
| `text_id` | 整数 | `1, 2,` `3, 4` |
| `文本` | 字符串 | `"This is sample text", "``Another example"` |
| `标记` | 字符串列表 | `["This", "is", "sample", "text"], ["``Another", "example"]` |
| `嵌入` | 浮点数列表 | `[0.1, 0.2, 0.3], [0.4,` `0.5, 0.6]` |
| `元数据` | 结构体 | `{"lang": "en", "source": "web"}, {"lang": "fr", "``source": "news"}` |
| `标签` | 整数 | `1, 0,` `1, 0` |
| `时间戳` | 时间戳 | `2024-01-01 10:30:00,` `2024-01-01 10:31:00` |
| `language_score` | 浮点数 | `0.95,` `0.87, 0.92` |
| `实体` | 结构体列表 | `[{"text": "Google", "type": "ORG"}, {"text": "New York", "``type": "LOC"}]` |
| `doc_stats` | 结构体 | `{"word_count": 150, "char_count": 750, "``sentence_count": 8}` |

表 4.2 – Apache Parquet 列中的数据结构

每一列数据都单独存储，并且可以高效地压缩和独立访问，这对于大规模自然语言处理尤其有用。

以下代码片段使用 PyArrow 库将表示为 Python 字典列表的数据集转换为 Parquet 文件，并将其读回：

```py
import pyarrow as pa
import pyarrow.parquet as pq
def convert_to_parquet(dataset, output_path):
    # Convert dataset to Arrow Table
    table = pa.Table.from_pydict(dataset[0])
    # Write to Parquet file
    pq.write_table(table, output_path)
def read_from_parquet(file_path):
    # Read Parquet file
    table = pq.read_table(file_path)
    # Convert back to dictionary
    return table.to_pydict()
#Usage
convert_to_parquet(large_dataset, "large_dataset.parquet")
loaded_dataset = read_from_parquet("large_dataset.parquet")
```

在前面的代码片段中，`convert_to_parquet` 函数接收一个数据集和一个输出文件路径，使用 `pa.Table.from_pydict` 将数据集中的第一个字典转换为 PyArrow 表，并使用 `pq.write_table` 将其写入 Parquet 文件。`read_from_parquet` 函数使用 `pq.read_table` 从指定路径读取 Parquet 文件到 PyArrow 表，然后使用 `table.to_pydict` 将其转换回 Python 字典。在用法示例中，一个变量 `large_dataset` 被序列化为 `"large_dataset.parquet"`，然后反序列化回 `loaded_dataset`。

Parquet 为 LLM 数据集提供了几个优势：

+   列存储以实现高效查询

+   压缩以减少存储需求

+   支持在 NLP 数据中常见的复杂嵌套结构

虽然前面的部分已经讨论了通过采样、分布式计算和优化存储策略来管理大规模静态数据集的方法，但这些方法假设语料库是有限且定义良好的。然而，训练场景越来越多地涉及数据的持续流入，如用户交互、实时遥测或不断变化的内容流。这些动态环境需要从传统的数据管道转向能够处理实时摄取和处理的架构。下一节介绍了流数据处理作为在 LLM 中维持长期、自适应训练机制所必需的演变。

# 用于持续 LLM 训练的流数据处理

对于不断生成新数据的情况，流处理允许持续更新模型。以下是一个使用 **Apache Kafka** ([`kafka.apache.org/`](https://kafka.apache.org/)) 和 **Faust** ([`faust.readthedocs.io/en/latest/`](https://faust.readthedocs.io/en/latest/)) 的示例。

Apache Kafka 是一个分布式流平台，它是构建实时数据管道和流应用程序的基础。它使用 **发布-订阅** （**pub-sub**） 模型，其中数据生产者向主题发送消息，消费者从这些主题中读取，允许在多个代理之间进行可扩展、容错的数据分发。当与异步处理结合使用时，这些技术使系统能够在不阻塞操作的情况下实时处理大量数据。Kafka 中的多个代理提供冗余和负载均衡，确保高可用性和吞吐量。这种架构在需要实时数据处理的情况下特别有用，例如日志聚合、指标收集、流处理和事件溯源。

另一方面，Faust 是一个基于 Python 的流处理库，旨在通过将数据视为连续的事件流来处理实时数据处理任务。与 Kafka Streams 类似，但用 Python 编写，Faust 允许开发者构建能够实时处理、转换和分析数据的流应用程序。它提供了用于处理流的高级抽象，使得实现复杂的流工作流程变得更容易，同时保持了 Python 的简单性和表达性。Faust 内部使用现代 Python 功能，如 `async/await`，并利用 Python 的 asyncio 库高效地处理并发操作。

以下代码定义了一个使用 Faust 的简单实时数据处理应用程序，Faust 是一个建立在 Kafka 之上的 Python 流处理库。它演示了如何从 Kafka 主题中消费消息，应用预处理逻辑，并为下游任务（如训练 LM）准备数据：

```py
import faust
class Text(faust.Record):
    content: str
app = faust.App('llm-training', broker='kafka://localhost:9092')
topic = app.topic('raw-text', value_type=Text)
@app.agent(topic)
async def process(stream):
    async for text in stream:
        processed_text = preprocess(text.content)
        # Here you would typically send the processed text to your LLM training pipeline
        print(f"Processed: {processed_text}")
if __name__ == '__main__':
    app.main()
```

首先，代码使用`faust.Record`定义了一个`Text`类，用于表示包含单个名为`content`的字符串字段的传入 Kafka 消息。然后，使用`'llm-training'`标识符创建 Faust 应用程序，并连接到运行在`kafka://localhost:9092`的本地 Kafka 代理。应用程序订阅名为`'raw-text'`的主题，并将传入的消息反序列化为`Text`对象。

核心处理逻辑在`process`函数中实现，该函数使用`@app.agent(topic)`装饰器，使其成为处理来自`raw-text`主题事件的 Faust 代理。该函数异步遍历流中的每条消息，对`content`字段应用`preprocess`函数，并打印结果。尽管当前代码打印处理后的文本，但在实际设置中，这通常是传递输出到语言模型训练管道或进一步处理阶段的典型位置。

最后，脚本包括一个标准的 Python 入口点，当脚本直接运行时，用于启动 Faust 应用程序。请注意，`preprocess`函数假定在完整实现的其他地方定义，因为它不包括在提供的代码片段中。

这种设置允许您持续处理传入的文本数据，然后可以实时或近实时地更新您的 LLM。`preprocess`函数将包含您的特定预处理逻辑。

# 内存高效的数据加载技术

对于太大而无法放入内存的数据集，我们可以使用**内存映射**或**分块**技术。

内存映射利用操作系统级别的功能，将大文件直接映射到内存中，而不需要加载整个文件。这使得对文件部分的随机访问成为可能，使其适用于需要频繁但非顺序访问的场景。对于大型、结构化数据集（例如嵌入或分词文本文件）来说，这种方法很快，但对于小而分散的读取，可能会有更高的开销。

另一方面，分块将数据分成更小的、顺序处理的块。这对于将大型、顺序访问的数据集（例如文本或日志）流式传输到内存受限环境中非常有效。虽然分块比内存映射简单且更易于移植，但在随机访问模式中，分块可能比内存映射慢。

这里有一个使用 NumPy 的`memmap`功能的示例，它创建类似于数组的对象，映射到磁盘上的文件，允许在不将整个数组加载到内存的情况下进行高效的读写操作。`memmap`功能利用操作系统的虚拟内存功能，在最小化内存使用的同时提供无缝的数组操作：

```py
import numpy as np
def create_memmap_dataset(dataset, output_file):
    # Determine the shape of the dataset
    num_samples = len(dataset)
    sample_shape = dataset[0]['input'].shape
    # Create a memory-mapped array
    mmap = np.memmap(
        output_file, dtype='float32', mode='w+',
        shape=(num_samples, *sample_shape)
    )
    # Write data to the memory-mapped array
    for i, sample in enumerate(dataset):
        mmap[i] = sample['input']
    # Flush to disk
    mmap.flush()
def load_memmap_dataset(file_path, shape):
    # Load the memory-mapped array
    return np.memmap(file_path, dtype='float32',
        mode='r', shape=shape)
#Usage
create_memmap_dataset(large_dataset, "large_dataset.mmap")
mmap_dataset = load_memmap_dataset(
    "large_dataset.mmap", shape=(len(large_dataset),
    *large_dataset[0]['input'].shape)
)
```

这种技术允许您通过将大部分数据保留在磁盘上，并在需要时仅将必要的部分加载到内存中，来处理比可用 RAM 更大的数据集。

这里是一个分块技术的示例，这在处理必须按顺序处理但一次无法全部装入内存的大型数据集时特别有用。与允许随机访问的内存映射不同，分块明确地按顺序加载和顺序处理固定大小的数据块。这在处理大型 CSV 文件、文本语料库或流日志时是一个常见的模式。在以下示例中，使用 pandas 以分块方式处理大型 CSV 文件，pandas 内部将行块读入内存，最小化峰值内存占用：

```py
import pandas as pd
def process_chunk(chunk):
    # Placeholder: process or transform the chunk here
    # For example, compute the mean of a column
    return chunk['value'].mean()
def process_large_csv(file_path, chunk_size=10000):
    results = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        result = process_chunk(chunk)
        results.append(result)
    return results
# Usage
file_path = 'large_dataset.csv'
aggregated_results = process_large_csv(file_path)
print("Processed chunk-level results:", aggregated_results)
```

在本例中，CSV 文件以每次 10,000 行为单位进行读取。每个块被传递到处理函数中，中间结果（在这种情况下，名为`'value'`的列的平均值）被存储以供进一步聚合或分析。这种方法是灵活的，并且可以轻松扩展到过滤、转换或将分块输出写入新文件等任务。

分块特别适用于线性访问数据且每个块相互独立的情况。然而，如果需要随机访问单个记录或跨块记录，内存映射或索引数据库解决方案可能更有效率。

# 摘要

在本节中，我们探讨了用于 LLM 训练的大型数据集管理和处理的高级技术。你了解了大型数据集的挑战、数据采样技术、分布式处理、高效的存储格式、流处理、数据分片和内存高效加载。

这些技术对于将 LLM 训练扩展到大规模数据集同时保持效率和数据质量至关重要，每种技术都对处理 LLM 的大数据集有自己的贡献：

+   **数据采样技术**：通过关注高影响或具有代表性的数据，它们减少了计算负担，提高效率并确保质量，而无需处理整个数据集

+   **分布式处理**：通过在机器间并行化任务，加快数据准备和训练速度，为大规模数据集提供可扩展性

+   **高效的存储格式**：它们提高了数据检索速度并减少了存储大小，简化了对大型数据集的访问并提高了 I/O 效率

+   **流处理**：通过增量处理数据，最小化内存使用，支持实时更新和连续数据流的有效处理

+   **数据分片**：通过将数据分割成更小的块，平衡工作负载并减少延迟，实现并行性和无缝扩展

+   **内存高效加载**：通过以可管理的部分加载数据，限制内存使用，确保处理超出内存容量的数据集的效率

在下一章中，我们将介绍另一种模式：LLM 开发的版本控制。
