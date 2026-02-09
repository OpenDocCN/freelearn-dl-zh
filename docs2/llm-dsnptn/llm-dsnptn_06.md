# 6

# 数据集标注和标签化

**数据集标注**是丰富数据集中原始数据的过程，通过添加信息性元数据或标签，使其对监督机器学习模型可理解和使用。这些元数据根据数据类型和预期任务而变化。对于文本数据，标注可能涉及为整个文档或特定的文本片段分配标签或类别，识别和标记实体，建立实体之间的关系，突出关键信息，以及添加语义解释。标注的目的是提供结构化信息，使模型能够学习模式并做出准确的预测或生成相关的输出。

**数据集标注**是一种专注于为单个数据点分配预定义的类别标签或类别的特定数据集标注类型。这通常用于分类任务，其目标是将数据分类到不同的组别中。在文本数据的背景下，标注可能涉及根据情感、主题或体裁对文档进行分类。

虽然标注为分类模型提供了至关重要的监督信号，但标注是一个更广泛的术语，它包括比简单分类更复杂的数据丰富形式。有效的数据集标注，包括适当的标注策略，对于开发能够处理各种复杂语言任务的性能优异的语言模型至关重要。

数据集标注和标签化是开发高性能模型的过程。在本章中，我们将探讨创建良好标注数据集的高级技术，这些技术可以显著影响您的大型语言模型（LLM）在各种任务上的性能。

在本章中，我们将涵盖以下主题：

+   质量标注的重要性

+   不同任务的标注策略

+   大规模文本标注的工具和平台

+   管理标注质量

+   群众外包标注 - 利益与挑战

+   半自动化标注技术

+   扩展大规模语言数据集的标注流程

# 质量标注的重要性

高质量的标注对于 LLM 训练的成功至关重要。它们提供了指导模型学习过程的真实信息，使模型能够理解语言的细微差别并准确执行特定任务。低质量的标注可能导致有偏见或不准确的模型，而高质量的标注可以显著提高 LLM 的性能和泛化能力。

那么，什么是高质量的标注？

高质量注释的特点是相似实例之间标签的一致性，对数据集中所有相关元素的完整覆盖，没有遗漏，并且与事实或既定标准的准确对齐——这意味着标签必须精确反映数据的真实性质，严格遵循预定的注释指南，并在边缘情况或模糊情况下保持可靠性。

让我们通过使用 spaCy 库的 **命名实体识别**（**NER**）任务来说明注释质量的影响。NER 是一种 **自然语言处理**（**NLP**）技术，它将文本中的关键信息（实体）识别和分类到预定义的类别中，例如人名、组织、地点、时间、数量、货币价值等。SpaCy 是一个流行的开源库，用于 Python 中的高级 NLP，以其效率和准确性而闻名。它提供了预训练模型，可以执行各种 NLP 任务，包括 NER、词性标注、依存句法分析等，使开发者更容易将复杂的语言处理能力集成到他们的应用程序中。

以下 Python 代码片段演示了如何以编程方式创建 spaCy 格式的 NER 任务训练数据：

```py
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
def create_training_data(texts, annotations):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in zip(texts, annotations):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db
texts = [
    "Apple Inc. is planning to open a new store in New York.",
    "Microsoft CEO Satya Nadella announced new AI features."
]
annotations = [
    [(0, 9, "ORG"), (41, 49, "GPE")],
    [(0, 9, "ORG"), (14, 27, "PERSON")]
]
training_data = create_training_data(texts, annotations)
training_data.to_disk("./train.spacy")
```

此代码使用 spaCy 创建 NER 的训练数据集。让我们分解一下：

1.  我们从 spaCy 导入必要的模块，包括用于高效存储训练数据的 `DocBin`。

1.  `create_training_data` 函数将原始文本和注释转换为 spaCy 的训练格式：

    1.  它创建了一个空白英语语言模型作为起点。

    1.  初始化一个 `DocBin` 对象来高效地存储处理后的文档。

    1.  对于每个文本及其注释，我们创建一个 spaCy `Doc` 对象，并根据提供的注释添加实体跨度。

1.  我们提供了两个带有相应 NER 注释的示例句子。

1.  在此代码中，`doc.char_span()` 通过将注释中的字符级 `start` 和 `end` 位置映射到 spaCy `Doc` 对象的实际标记边界来创建实体跨度。它将原始字符索引（例如 `Apple Inc.` 的 `0` 到 `9`）转换为与标记边界对齐的适当的 spaCy `Span` 对象，确保实体标签正确地附加到文档中它们所代表的精确文本序列。

1.  训练数据以 spaCy 的二进制格式保存到磁盘。

这些注释的质量直接影响模型正确识别和分类实体的能力。例如，如果 `Apple Inc.` 被错误地标记为个人而不是组织，模型就会学会错误地将公司名称分类为个人。

# 不同任务的注释策略

不同的 LLM 任务需要特定的注释策略。让我们探讨一些常见任务及其注释方法：

+   `datasets` 库：

    ```py
    from datasets import Dataset
    texts = [
        "This movie was fantastic!",
        "The service was terrible.",
        "The weather is nice today."
    ]
    labels = [1, 0, 2]  # 1: positive, 0: negative, 2: neutral
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    print(dataset[0])
    # Output: {'text': 'This movie was fantastic!', 'label': 1}
    ```

    以下代码创建了一个简单的数据集用于情感分析。每个文本都与一个表示其情感的标签相关联。

+   **NER**：对于命名实体识别（NER），我们使用实体标签标注特定的文本跨度。这里介绍一种使用**BIO**标签方案的方法。

BIO 标签方案

使用`"B-"`标记实体的起始单词，`"I-"`标记属于同一实体的任何后续单词，以及`"O"`标记不属于任何实体的单词。这种方法解决了区分相邻实体和处理多词实体的难题——例如，帮助模型理解`《纽约时报》`是一个单一的组织实体，或者在句子`Steve Jobs met Steve Wozniak`中，存在两个不同的人物实体，而不是一个或四个单独的实体。这种标签系统的简洁性和有效性使其成为机器学习识别和分类文本中命名实体的标准选择。

以下代码演示了如何使用分词器直接将文本编码成适合 transformer 模型的格式：

```py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Apple Inc. was founded by Steve Jobs"
labels = ["B-ORG", "I-ORG", "O", "O", "O", "B-PER", "I-PER"]
tokens = tokenizer.tokenize(text)
inputs = tokenizer(text, return_tensors="pt")
print(list(zip(tokens, labels)))
```

此示例演示了如何为 NER 任务创建 BIO 标签。`B-`前缀表示实体的开始，`I-`表示实体的延续，而`O`表示任何实体的外部标记。

+   答案在上下文中的`start`和`end`位置：

    ```py
    context = "The capital of France is Paris. It is known for its iconic Eiffel Tower."
    question = "What is the capital of France?"
    answer = "Paris"
    start_idx = context.index(answer)
    end_idx = start_idx + len(answer)
    print(f"Answer: {context[start_idx:end_idx]}")
    print(f"Start index: {start_idx}, End index: {end_idx}")
    ```

    以下代码演示了如何为问答任务标注答案跨度。

现在，让我们来看看一些用于执行大规模文本标注的工具和平台。

# 大规模文本标注的工具和平台

数据标注是许多机器学习项目的基石，为训练和评估模型提供所需的标记数据。然而，手动标注，尤其是在大规模上，既耗时又容易出错，难以管理。这就是专业标注工具变得至关重要的地方。它们简化了流程，提高了数据质量，并提供自动化、协作以及与机器学习工作流程集成的功能，最终使大规模标注项目变得可行且高效。

**Prodigy**，来自 spaCy 创建者的强大商业工具，因其主动学习功能而脱颖而出。它智能地建议下一个需要标注的最具信息量的示例，显著减少了标注工作量。Prodigy 的优势在于其可定制性，允许用户使用 Python 代码定义标注工作流程，并将其无缝集成到机器学习模型中，尤其是在 spaCy 生态系统中。对于需要复杂标注任务、有预算购买高级工具且重视主动学习效率提升的项目来说，它是一个极佳的选择。

**Label Studio** 是一个多功能的开源选项，适用于多种数据类型，包括文本、图像、音频和视频。它友好的可视化界面和可定制的标注配置使其对所有级别的标注者都易于使用。Label Studio 还支持协作，并提供多种导出格式，使其与各种机器学习平台兼容。对于需要灵活、免费解决方案且支持多种数据类型并需要协作标注环境的项目来说，它是一个强有力的竞争者。

**Doccano** 是一个专门为机器学习中的文本标注设计的开源工具。它在序列标注、文本分类和序列到序列标注等任务上表现出色。Doccano 具有简单直观的界面，支持多用户，并提供 API 以与机器学习管道集成。对于仅关注文本标注且需要简单、免费解决方案并希望与现有机器学习工作流程无缝集成的项目来说，它是首选选择。

下面是一个示例，说明如何将 Doccano 的标注集成到 Python 工作流程中：

```py
import json
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification)
def load_doccano_ner(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data
doccano_data = load_doccano_ner('doccano_export.jsonl')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased")
for item in doccano_data:
    text = item['text']
    labels = item['labels']
    # Process annotations and prepare for model input
    tokens = tokenizer.tokenize(text)
    ner_tags = ['O'] * len(tokens)
    for start, end, label in labels:
        start_token = len(tokenizer.tokenize(text[:start]))
        end_token = len(tokenizer.tokenize(text[:end]))
        ner_tags[start_token] = f'B-{label}'
        for i in range(start_token + 1, end_token):
            ner_tags[i] = f'I-{label}'
    # Now you can use tokens and ner_tags for model training or inference
```

此代码从 Doccano 导出文件中加载 NER 标注，并将它们处理成适合训练基于 BERT 的标记分类模型的格式。以下示例中的标记和 `ner_tags` 显示了样本格式：

```py
text = "The majestic Bengal tiger prowled through the Sundarbans, a habitat it shares with spotted deer."
labels = [[13, 25, "ANIMAL"], [47, 57, "GPE"], [81, 93, "ANIMAL"]]
tokens = ['The', 'majestic', 'Bengal', 'tiger', 'prowled', 'through', 
    'the', 'Sundarbans', ',', 'a', 'habitat', 'it', 'shares', 'with',
    'spotted', 'deer', '.']
ner_tags = ['O', 'O', 'B-ANIMAL', 'I-ANIMAL', 'O', 'O', 'O', 'B-GPE',
    'O', 'O', 'O', 'O', 'O', 'O', 'B-ANIMAL', 'I-ANIMAL', 'O']
```

此示例演示了在文本中识别和分类动物名称的命名实体识别（NER）。文本包含关于孟加拉虎和斑点鹿在恒河三角洲的句子。`labels` 列表提供了动物实体（`"Bengal tiger"`，`"spotted deer"`）的起始和结束索引以及它们对应的类型（`"ANIMAL"`），以及地缘政治实体，即 `"Sundarbans"`（`"GPE"`）。`tokens` 列表是文本的词级分割。最后，`ner_tags` 列表表示 BIO（Begin-Inside-Outside）格式的 NER 标注，其中 `"B-ANIMAL"` 标记动物实体的开始，`"I-ANIMAL"` 标记同一动物实体内的后续单词，`"B-GPE"` 标记地缘政治实体的开始，而 `"O"` 表示不属于任何命名实体的标记。

# 管理标注质量

为了确保高质量的标注，我们需要实施一个强大的质量保证流程。

让我们看看一些衡量标注质量的方法：

+   `-1` 和 `1`，其中 `1` 表示完全一致，`0` 表示与机会一致，负值表示低于机会的一致性。

    以下代码计算 Cohen 的 Kappa 系数，以量化两组分类评级之间的一致性：

    ```py
    from sklearn.metrics import cohen_kappa_score
    annotator1 = [0, 1, 2, 0, 1]
    annotator2 = [0, 1, 1, 0, 1]
    kappa = cohen_kappa_score(annotator1, annotator2)
    print(f"Cohen's Kappa: {kappa}")
    ```

+   `calculate_accuracy` 函数计算一组真实标签（即 `gold_standard`）与一组预测或标注标签（即标注）之间的协议：

    ```py
    def calculate_accuracy(gold_standard, annotations):
        return sum(
            g == a for g, a in zip(
                gold_standard, annotations
            )
        ) / len(gold_standard)
    gold_standard = [0, 1, 2, 0, 1]
    annotator_result = [0, 1, 1, 0, 1]
    accuracy = calculate_accuracy(gold_standard, annotator_result)
    print(f"Accuracy: {accuracy}")
    ```

    虽然 Cohen 的 Kappa 和与黄金标准的准确性是基础，但其他指标可以更深入地了解标注质量。例如，Krippendorff 的 Alpha 提供了一种灵活的方法，适应各种数据类型并处理缺失数据，使其适合复杂的标注任务。在涉及多个标注者的场景中，Fleiss 的 Kappa 扩展了 Cohen 的 Kappa，提供了整个群体的一致性总体评估。

    对于诸如目标检测或图像分割等任务，**交并比**（**IoU**）变得至关重要，它量化了预测和真实边界框或掩模之间的重叠。此外，特别是在处理不平衡数据集或成本更高的特定错误类型时，精确度、召回率和 F1 分数提供了细微的评价，特别适用于诸如命名实体识别（NER）等任务。

+   **敏感度和特异性**：这些指标，常用于医学诊断或二元分类，对于标注质量评估也很有价值。敏感性（也称为召回率或真正率）衡量的是实际正例中被正确识别的比例，而特异性（真正负率）衡量的是实际负例中被正确识别的比例。

+   **均方根误差**（**RMSE**）和**平均绝对误差**（**MAE**）：对于涉及数值或连续标注的任务（例如，评分量表、边界框坐标等），RMSE 和 MAE 可以量化标注值与真实值之间的差异。RMSE 对较大误差赋予更高的权重，而 MAE 对所有误差同等对待。

+   **基于时间的指标**：除了标签的质量外，标注过程的效率也很重要。跟踪每条标注花费的时间，尤其是当与准确性或一致性评分相关联时，可以揭示流程改进的领域或识别可能需要额外培训的标注者。此外，分析标注时间的分布可以帮助识别异常困难或模糊的实例。

最终，对标注质量的全面方法涉及考虑一系列相关指标的组合，这些指标针对特定的任务和项目目标量身定制。定期监控、反馈循环以及指南和培训的迭代改进对于在整个标注过程中保持高标准至关重要。记住，指标的选择应与数据的性质、任务的复杂性和机器学习项目的预期结果相一致。

**众包是扩展标注工作的有效替代方案**。

# **众包标注——优势和挑战**

**众包可以是一种有效的扩展标注工作的方法**。例如，Amazon Mechanical Turk 或 Appen（前身为 Figure Eight）等平台提供了对大量工作力的访问。然而，确保质量可能具有挑战性。以下是一个如何汇总众包标注的例子：

```py
from collections import Counter
def aggregate_annotations(annotations):
    return Counter(annotations).most_common(1)[0][0]
crowd_annotations = [
    ['PERSON', 'PERSON', 'ORG', 'PERSON'],
    ['PERSON', 'ORG', 'ORG', 'PERSON'],
    ['PERSON', 'PERSON', 'ORG', 'LOC']
]
aggregated = [aggregate_annotations(annot) 
    for annot in zip(*crowd_annotations)]
print(f"Aggregated annotations: {aggregated}")
```

此代码使用简单的多数投票方案来汇总多个标注者的标注。虽然这种方法在许多情况下都有效，但在票数相等的情况下需要平局处理，并且可以采用基于标注者可靠性分配权重或利用基于机器学习的协调模型等额外策略来进一步提高质量。

接下来，我们将深入了解半自动化标注技术，其中机器学习模型协助人工标注者加速标注任务。

# 半自动化标注技术

半自动化标注结合机器学习与人工验证以加快标注过程。以下是一个使用 spaCy 的简单示例：

```py
import spacy
nlp = spacy.load("en_core_web_sm")
def semi_automated_ner(text):
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_)
    for ent in doc.ents]
text = "Apple Inc. was founded by Steve Jobs in Cupertino."
auto_annotations = semi_automated_ner(text)
print(f"Auto-generated annotations: {auto_annotations}")
# Human annotator would then verify and correct these annotations
```

此代码使用预训练的 spaCy 模型生成初始 NER 标注，然后可以由人工标注者进行验证和纠正。

接下来，我们将探讨一些策略，以扩展标注工作流程以处理大规模语言数据集。

# 扩展大规模语言数据集的标注过程

对于大规模数据集，考虑以下策略：

+   **分布式处理**：使用如 **Dask** 或 **PySpark** 这样的库进行分布式标注处理。Dask 和 PySpark 是强大的库，可用于分布式数据标注处理，使团队能够高效地处理大规模标注任务。这些库允许您在多个核心或甚至计算机集群上并行化标注工作流程，显著加快大规模数据集的处理速度。使用 Dask，您可以扩展现有的基于 Python 的标注脚本来在分布式系统中运行，而 PySpark 在 Apache Spark 生态系统内提供强大的数据处理能力。这两个库都提供了熟悉的 API，使得从本地标注管道过渡到分布式管道变得更加容易，允许标注团队处理和管理单个机器无法处理的大型数据集。

+   **主动学习**：这项技术涉及根据模型不确定性或预期影响，迭代选择最具信息量的样本进行人工标注。从一个小的、已标注的数据集开始，训练一个模型，使用它来识别有价值的未标注样本，然后由人工进行标注，并更新模型。这个周期重复进行，优化标注努力并有效地提高模型性能。

    这里有一个简单的主动学习示例：

    ```py
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from modAL.models import ActiveLearner
    # Simulated unlabeled dataset
    X_pool = np.random.rand(1000, 10)
    # Initialize active learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_pool[:10],
        y_training=np.random.randint(0, 2, 10)
    )
    # Active learning loop
    n_queries = 100
    for _ in range(n_queries):
        query_idx, query_inst = learner.query(X_pool)
        # In real scenario, get human annotation here
        y_new = np.random.randint(0, 2, 1)
        learner.teach(X_pool[query_idx], y_new)
        X_pool = np.delete(X_pool, query_idx, axis=0)
    print(
        f"Model accuracy after active learning: "
        f"{learner.score(
            X_pool, np.random.randint(0, 2, len(X_pool)))}"
    )
    ```

    这个例子演示了一个基本的主动学习循环，其中模型选择最具信息量的样本进行标注，这可能会减少所需的总标注数量。

现在我们已经了解了某些标注技术，让我们来看看在执行标注过程中可能出现的偏差以及如何避免它们。

# 标注偏差及缓解策略

标注偏差是在标注过程中可能渗透到标注数据集中的系统性错误或偏见。这些偏差可能会显著影响基于这些数据训练的机器学习模型的性能和公平性，导致模型不准确或表现出歧视行为。识别和减轻这些偏差对于构建稳健和道德的 AI 系统至关重要。

标注偏差的类型包括以下：

+   **选择偏差**：当用于标注的数据不能代表模型在现实世界中遇到的真实数据分布时，就会发生这种情况。例如，如果用于面部识别的数据集主要包含肤色较浅的人的图像，那么在它上面训练的模型在处理肤色较深的人时可能会表现不佳。

+   **标注偏差**：这源于标注者的主观解释、文化背景或个人信仰。例如，在情感分析中，来自不同文化的标注者可能对相同的文本进行不同的情感极性标注。同样，标注者的个人偏见可能导致他们对某些群体或个人进行更负面或更正面的标注，而其他人则不然。

+   **确认偏差**：标注者可能无意识地倾向于确认他们关于数据的先入为主的信念或假设。

+   **自动化偏差**：过度依赖预训练模型或主动学习系统的建议可能导致标注者在没有足够审查的情况下接受错误的标签。

+   **指南中的模糊性**：如果标注指南不明确或不完整，可能会导致标注者之间标注不一致，将噪声和偏差引入数据集。

这里有一些减轻偏差的策略：

+   **多样性和代表性数据**：确保用于标注的数据既多样化又能够代表目标人群和使用案例。这可能涉及对代表性不足的群体进行过度采样或从多个来源收集数据。

+   **明确和全面的指南**：制定详细的标注指南，明确定义标注标准，并为每个标签提供示例。在指南中解决潜在的模糊性和边缘情况。根据标注者的反馈和新兴问题定期审查和更新指南。

+   **标注者培训和校准**：对标注者进行关于任务、指南以及他们应该意识到的潜在偏差的全面培训。进行校准会议，让标注者对相同的数据进行标注并讨论任何差异，以确保一致性。

+   **多个标注者和标注者间一致性**：对每个数据点使用多个标注者，并使用如 Cohen 的 Kappa 或 Fleiss 的 Kappa 等指标来衡量**标注者间一致性**（IAA）。高 IAA 表明一致性良好，而低 IAA 则表明指南、培训或任务本身存在问题。

+   **裁决过程**：建立一个解决标注员之间分歧的程序。这可能涉及让资深标注员或专家审查并做出最终决定。

+   **具有偏差意识的主动学习**：在使用主动学习时，要留意模型建议中可能存在的偏差。鼓励标注员批判性地评估建议，而不是盲目接受。

+   **偏差审计和评估**：定期审计标记数据和训练模型以识别潜在的偏差。评估模型在不同人口群体或类别中的性能，以识别任何差异。

+   **多元化的标注团队**：组建具有不同背景、观点和经验的标注团队，以减轻个人偏差的影响。

通过实施这些缓解策略，您可以显著减少标注偏差的影响，从而实现更准确、公平和可靠的机器学习模型。重要的是要记住，偏差缓解是一个持续的过程，需要在整个机器学习生命周期中持续监控、评估和改进。

# 摘要

从这个设计模式中，您了解了 LLM 开发中数据集标注和标记的高级技术。您现在理解了高质量标注在提高模型性能和泛化能力中的关键重要性。您对各种 LLM 任务的标注策略有了深入了解，包括文本分类、命名实体识别和问答。

在本章中，我们向您介绍了用于大规模文本标注的工具和平台、管理标注质量的方法以及众包标注的优缺点。您还了解了半自动化标注技术和用于大规模语言数据集标注过程扩展的策略，例如分布式处理和主动学习。我们通过使用 spaCy、transformers 和 scikit-learn 等库提供了实际示例，这有助于您掌握关键概念和实现方法。

在下一章中，您将探索如何构建用于训练 LLM 的高效和可扩展的管道。这包括探索数据预处理的最佳实践、设计模型架构的关键考虑因素以及优化性能和可扩展性的策略。

# 第二部分：大型语言模型的训练和优化

本部分深入探讨了有效训练和优化大型语言模型所需的过程。我们将引导您设计既模块化又可扩展的稳健训练流程。您将学习如何调整超参数以最大化性能，实施正则化技术以稳定训练，并集成高效的检查点和恢复方法以支持长时间运行的训练会话。此外，我们还将探讨高级主题，如剪枝和量化，这些技术可以帮助您在不牺牲性能的情况下减小模型大小和计算需求。此外，还将详细介绍微调技术，这些技术用于将预训练模型适应特定任务或领域。到本部分结束时，您将具备构建、训练和优化能够应对现实应用挑战的大型语言模型的能力。

本部分包含以下章节：

+   *第七章*, *训练流程*

+   *第八章*, *超参数调整*

+   *第九章*, *正则化*

+   *第十章*, *检查点和恢复*

+   *第十一章*, *微调*

+   *第十二章*, *模型剪枝*

+   *第十三章*, *量化*
