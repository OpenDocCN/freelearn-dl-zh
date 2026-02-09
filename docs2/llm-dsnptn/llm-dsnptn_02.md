# 2

# LLM 训练的数据清洁

在本章中，我们将深入探讨 LLM 训练中的**数据清洁**模式。

清洁、高质量的数据是构建稳健和可靠的语言模型的基础。我们将探讨常见的数据质量问题、预处理技术和处理不同数据类型的策略。*图 2.1*展示了专门设计用于在用于训练语言模型之前处理原始文本数据的数据清洁流程。

![图 2.1 – 数据清洁流程](img/B31249_02_01.jpg)

图 2.1 – 数据清洁流程

该过程从初步的数据质量检查开始，以评估原始数据的适用性。随后，应用文本预处理和去重步骤以精炼和简化数据集。如果在任何点上数据未能达到所需标准，它将通过自动化清洁流程进行额外处理。成功完成此阶段后，进行数据验证以确保数据集的完整性和符合训练标准。如果数据通过验证，则标记为清洁并准备好用于语言模型训练，确保为有效模型开发提供高质量输入。

到本章结束时，你将具备用于为 LLM 训练清洁数据的实用工具和技术。

本章将涵盖以下主题：

+   理解清洁数据的重要性

+   语言数据集中常见的质量问题

+   适用于 LLM 的文本预处理技术

+   处理多语言和代码混合数据

+   大型文本语料库的去重策略

+   自动化数据清洁流程

+   数据验证和质量保证

# 理解清洁数据的重要性

用于训练 LLM 的数据质量直接影响其性能和可靠性。当我们使用嘈杂或不一致的数据训练 LLM 时，我们可能会将偏差、错误和不一致性引入模型的学习表示和输出中。

为了说明数据质量对 LLM 性能的影响，我们可以使用一个简单的 Python 脚本来比较在清洁和嘈杂数据上训练的模型的混淆度得分。

1.  首先，安装必要的包并导入它们：

    ```py
    pip install torch
    pip install transformers
    import torch
    torch) is a powerful deep learning framework that provides dynamic computational graphs, GPU acceleration, and extensive neural network building blocks, making it popular for machine learning research and development. The transformers package, developed by Hugging Face, complements PyTorch by providing a comprehensive library of pre-trained transformer models (such as BER, GPT, and T5) and tools for natural language processing tasks. Together, these packages offer a robust ecosystem in which torch provides the foundational deep learning operations, tensor computations, and automatic differentiation capabilities, while transformers provides high-level abstractions for working with state-of-the-art language models, including functions for tokenization, model fine-tuning, and inference.
    ```

1.  然后，定义函数的初始部分：

    ```py
    def calculate_perplexity(model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
        outputs = model(inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    model = GPT4LMHeadModel.from_pretrained("GPT4")
    tokenizer = GPT4Tokenizer.from_pretrained("GPT4")
    ```

    `calculate_perplexity`函数使用提供的分词器将输入文本分词成 PyTorch 张量。然后，它将分词后的输入传递给模型，其中`input_ids`也用作标签，允许模型计算表示预测错误的损失。该损失被指数化以推导出一个标量混淆度得分，并以 Python 浮点数的形式返回。

    代码的第二部分初始化了一个语言模型和分词器，使用`GPT4LMHeadModel.from_pretrained("GPT4")`和`GPT4Tokenizer.from_pretrained("GPT4")`，从标识为`"GPT4"`的预训练源加载模型和分词器权重。

混淆度

**混淆度**是用于评估语言模型的一个度量。它量化了一个概率模型预测样本的能力。

较低的困惑度表明模型对其预测更有信心，并认为文本更可能或“正常”。较高的困惑度表明模型认为文本更令人惊讶或不同寻常。

1.  这里有一些示例文本：

    ```py
    clean_text = "The quick brown fox jumps over the lazy dog."
    noisy_text = "Th3 qu1ck br0wn f0x jumps 0ver th3 l@zy d0g."
    clean_text and noisy_text. clean_text holds a standard English sentence, while noisy_text contains the same sentence with deliberate character substitutions, making it “noisy” or corrupted. The clean_text and noisy_text examples are used to evaluate a language model’s perplexity, where clean_text provides a baseline for ideal text prediction and noisy_text assesses the model’s robustness to real-world data corruption; by comparing the perplexity scores, we determine how well the model handles noisy input and its suitability for applications where text data is not always perfectly formatted.
    ```

1.  最后，计算困惑度并打印结果：

    ```py
    clean_perplexity = calculate_perplexity(model, tokenizer,
        clean_text)
    noisy_perplexity = calculate_perplexity(model, tokenizer,
        noisy_text)
    print(f"Clean text perplexity: {clean_perplexity:.2f}")
    print(f"Noisy text perplexity: {noisy_perplexity:.2f}")
    ```

此脚本演示了输入数据中的微小噪声如何显著影响模型的困惑度。

困惑度评分是交叉熵损失的指数。在此代码中，它使用`torch.exp(outputs.loss).item()`进行计算。

这里是我们的可能结果：

+   `The quick brown fox jumps over the lazy dog`是一个常见的、语法正确的英语句子。干净文本的困惑度可能类似于`10.25`。

+   `Th3 qu1ck br0wn f0x jumps 0ver th3 l@zy d0g`中包含数字和符号代替字母，使其不那么常见，并且对模型预测来说更困难。噪声文本的困惑度可能类似于`52.87`。

具体的数字将取决于所使用的特定模型和分词器，但噪声文本的困惑度评分应该始终高于干净文本。

这种分数差异展示了模型区分标准、易于预测的文本和异常、难以预测的文本的能力。这对于检测机器生成或篡改的文本等任务非常有用，因为此类文本的困惑度评分通常高于人类撰写的文本。

# 语言数据集中常见的数据质量问题

语言数据集通常包含各种质量问题，可能会对 LLM 训练产生负面影响：

+   拼写和语法错误可能会在学习的表示中引入噪声和不一致性。

+   不一致的格式可能导致模型学习到的模式中出现不必要的复杂性。

+   冗余数据可能导致模型过度拟合到重复项中存在的特定模式或偏差。

+   不相关或低质量的内容会稀释数据集中有用的信息。

+   不完整或截断的句子可能导致模型学习到不完整的语言结构。

+   代码切换和混合语言可能会使针对特定语言训练的模型感到困惑。

+   **个人身份信息**（**PII**）引发隐私问题，并可能导致敏感数据的记忆化。

为了检测这些问题，我们可以使用各种 Python 库和技术。以下是一个使用 spaCy 进行基本文本质量检查的示例：

1.  提供导入语句和整体函数定义：

    ```py
    import spacy
    from collections import Counter
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    def analyze_text_quality(text):
        doc = nlp(text)
    ```

1.  检查拼写错误（使用 spaCy 内置的拼写检查器）：

    ```py
        misspelled = [
            token.text for token in doc if token._.is_misspelled
        ]
    ```

1.  检查语法问题（使用词性（pos）标签的简单方法）：

    ```py
    pos_counts = Counter(token.pos_ for token in doc)
    grammar_score = pos_counts['NOUN'] + pos_counts['VERB'] 
        + pos_counts['ADJ'] + pos_counts['ADV']
    ```

    **词性**（**POS**）标签是指分配给句子中每个单词的标签，以指示其语法角色。这些标签帮助系统理解句子的句法结构，并在解析、机器翻译、情感分析和信息提取等任务中使用。每个标签对应一个词性，如名词、动词或形容词，通常有更细粒度的区分来捕捉时态、数或功能。

1.  检查句子完整性：

    ```py
    incomplete_sentences = [
        sent.text for sent in doc.sents if len(sent) < 3
    ]
        return {
            "misspelled_words": misspelled,
            "grammar_score": grammar_score,
            "incomplete_sentences": incomplete_sentences
        }
    ```

1.  下面是代码的一个示例用法：

    ```py
    text = "This iz a smple txt with sum issues. Incomplet"
    quality_report = analyze_text_quality(text)
    print(quality_report)
    ```

在步骤 1 到 5 中提供的脚本展示了识别一些常见文本质量问题的基本框架。我们将在接下来的章节中讨论其他质量相关问题。

# LLM 的文本预处理技术

有效的文本预处理对于为 LLM 训练准备数据至关重要。我们采用各种技术，包括小写化、标点处理、空白字符标准化、特殊字符处理、**分词**、数字标准化和缩写词扩展。分词是将文本分解成更小单元以进行进一步分析或处理的过程。在自然语言处理中，标记是文本的最小有意义的单元。它们可以是单词，但也可以包括标点、数字或其他元素，具体取决于分词策略。

此外，**子词分词**是一种高级文本处理技术，它将单词分解成更小的有意义的单元（子词），使得在自然语言处理任务中更有效地处理罕见词、复合词和形态变化。与传统词级分词不同，子词分词可以识别常见的词首、词尾和词根，使模型能够通过识别其熟悉的组件来理解和处理之前未见过的单词。

以“unbelievably”这个词为例。传统的词级分词会将它视为一个单独的标记。如果模型之前从未见过这个单词，它可能难以正确解释它。相比之下，子词分词会将它分解成更小的组件，如“un”、“believ”和“ably”。这些子词在不同上下文中更有可能出现——“un-”在“unlikely”中，“believ”在“believe”中，“ably”在“capably”中——即使模型第一次遇到“unbelievably”，也能从中推导出意义。这种分解增强了泛化能力，减少了词汇量，并提高了模型处理罕见或形态复杂单词的能力。

流行的子词分词算法包括**字节对编码**（**BPE**）、WordPiece 和 SentencePiece，这些算法学习识别训练语料库中频繁出现的字符序列，并创建一个子词标记词汇表。这种方法对于处理形态丰富的语言特别有价值，可以在保持语义意义的同时减少词汇量，并且已成为现代语言模型如 Gemini、Claude、GPT 和其他基于 transformer 架构的基本组成部分。

这些方法有助于清理和标准化文本数据，减少噪声并提高模型泛化的能力。下面是一个演示这些预处理技术的 Python 脚本：

1.  首先，导入必要的 Python 包：

    ```py
    import unicodedata
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

1.  然后，定义整体预处理函数：

    ```py
    def preprocess_text(text):
        # Lowercase the text
        text = text.lower()
        # Normalize unicode characters
        text = unicodedata.normalize(
            'NFKD', text
        ).encode(
            'ascii', 'ignore'
        ).decode('utf-8')
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Tokenize :
       tokens = word_tokenize(text)
    ```

1.  移除停用词（停用词是像“the”、“is”和“at”这样的常见词，它们在文本处理中通常被移除，因为它们语义意义很小）：

    ```py
        stop_words = set(stopwords.words('english'))
        tokens = [
            token for token in tokens if token not in stop_words
        ]
        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    ```

1.  下面是一个代码示例的使用方法：

    ```py
    raw_text = "This is an EXAMPLE of text preprocessing... It's quite useful!"
    cleaned_text = preprocess_text(raw_text)
    print(f"Original: {raw_text}")
    print(f"Preprocessed: {cleaned_text}")
    ```

此脚本演示了基本的文本预处理技术。对于 LLM 训练，我们可能需要根据模型和数据集的具体要求调整这些技术。

# 处理多语言和代码混合数据

大型语言模型（LLMs）经常遇到多语言和代码混合数据，这种数据是在单个句子或对话中混合两种或更多语言。这对 LLMs 来说是一个挑战，因为它们必须解释跨多种语言的语用学细微差别、语法和语义联系。为了处理代码混合数据，LLMs 需要学习语言切换、词汇和句法变化，并保持连贯的回应，这要求强大的语言建模和多语言训练数据。

我们需要实施策略来有效处理这些场景。以下步骤是必要的，因为它们创建更干净、更一致的训练数据，有助于 LLMs 更好地理解和处理不同语言和混合语言场景中的文本，最终提高它们在实际应用中语言混合常见场景下的性能。

对于多语言数据，某些任务至关重要：

+   **语言识别**：检测每个文本样本的主要语言

+   **脚本规范化**：将文本转换为一致的脚本（例如，转写）

+   **特定语言预处理**：应用特定语言的标记化和规范化

同时，对于代码混合数据，你应该执行以下步骤：

+   **标记级语言识别**：识别单个标记的语言

+   **一致性执行**：确保一致地处理代码切换模式

下面是一个演示语言检测和脚本规范化的 Python 脚本。

1.  让我们提供导入和整体函数定义：

    ```py
    from langdetect import detect
    from unidecode import unidecode
    from nltk import word_tokenize
    import nltk
    # Download required NLTK data
    nltk.download('punkt')
    def handle_multilingual_text(text):
        # Detect language
        try:
            lang = detect(text)
        except:
            lang = 'unknown'
        # Transliterate non-ASCII characters
        transliterated_text = unidecode(text)
    ```

1.  标记化（为了简单起见使用 NLTK，但请考虑特定语言的标记化器）：

    ```py
        tokens = word_tokenize(transliterated_text)
        return {
            'original': text,
            'language': lang,
            'transliterated': transliterated_text,
            'tokens': tokens
        }
    ```

1.  下面是一个示例用法：

    ```py
    texts = [
        "This is English text.",
        "Dies ist deutscher Text.",
        "これは日本語のテキストです。",
        "This is mixed language text avec un peu de français."
    ]
    for text in texts:
        result = handle_multilingual_text(text)
        print(f"Original: {result['original']}")
        print(f"Detected Language: {result['language']}")
        print(f"Transliterated: {result['transliterated']}")
        print(f"Tokens: {result['tokens']}\n")
    ```

    此代码遍历一个包含英语、德语、日语和代码混合示例的多语言文本字符串列表，并对每个字符串调用一个`handle_multilingual_text`函数（可能定义在其他地方）来处理文本，返回一个包含原始文本、检测到的语言、转写文本（如果适用）和分词单词的字典，然后打印到控制台。

将前面的三个代码块合并，我们提供了一个处理多语言文本的基本框架。对于更高级的场景，我们会使用专门的库，如 Polyglot 进行特定语言的处理和代码混合分析，当同一对话中使用多种语言时([`dl.acm.org/doi/10.1145/3544548.3581445`](https://dl.acm.org/doi/10.1145/3544548.3581445))。

例如，Polyglot 包含内置的语言检测、命名实体识别、情感分析和跨多种语言的转写功能，同时与较大的多语言框架相比，保持了相对轻量级的性能。该库对于处理国际文本数据的项目尤其有价值，因为它提供了跨语言的统一 API，并附带预训练模型，使其成为无需管理多个特定语言工具的复杂性的多语言文本分析任务的效率选择。

# 大型文本语料库的去重策略

去重是准备大型文本语料库进行 LLM 训练的关键步骤。重复内容可能导致模型偏差和计算资源的浪费。我们采用各种策略来高效地识别和删除重复项：

+   **精确匹配去重**: 删除完全相同的文本样本。

+   **近似重复检测**: 识别并删除高度相似的文字样本。

+   **Shingling**: 创建用于比较的小重叠单词序列。

+   **局部敏感哈希**: 在大型数据集中高效地找到相似项。

以下部分展示了每种策略的示例。

## 精确匹配去重

**场景**: 你有一份客户地址列表：

+   **数据**:

    +   “123 Main St, Anytown, CA 91234”

    +   “456 Oak Ave, Somecity, NY 56789”

    +   “123 Main St, Anytown, CA 91234”

+   **结果**: 第三条记录“123 Main St, Anytown, CA 91234”被删除，因为它与第一条记录完全相同。

+   **剩余数据**:

    +   “123 Main St, Anytown, CA 91234”

    +   “456 Oak Ave, Somecity, NY 56789”

## 近似重复检测

**场景**: 你有一系列新闻文章：

+   **数据**:

    +   文章 1: “公司报告了季度利润的显著增长。”

    +   文章 2: “公司报告季度利润大幅增长。”

+   **结果**: 近似重复检测算法确定这些文章在内容上高度相似，尽管措辞略有不同。基于相似度阈值，删除了一篇文章。

+   **剩余数据**: “公司报告了季度利润的显著增长。”

## Shingling

**场景**：您想比较文本文档的相似度：

+   **数据**：

    +   文档 1：“The quick brown fox jumps over the lazy dog。”

    +   k=3 词 shingle。

+   **结果**：生成的 shingles 如下：

    +   “The quick brown”

    +   “quick brown fox”

    +   “brown fox jumps”

    +   “fox jumps over”

    +   “jumps over the”

    +   “over the lazy”

    +   “the lazy dog”

    然后文档被表示为那些 shingles 的集合。然后另一个文档可以被转换成 shingles，shingles 的集合可以进行比较。

## 局部敏感哈希（LSH）

**场景**：您有一个非常大的在线产品描述数据库：

+   **过程**：

    +   LSH 用于对产品描述进行哈希处理。

    +   相似的产品描述更有可能被哈希到相同的“桶”中。

    +   然后只比较同一桶内的描述，以详细查找近似重复项。

+   **结果**：LSH 不是将每个产品描述与其他每个描述进行比较，而是将比较缩小到同一桶内的描述，大大提高了查找近似重复项的效率。

注意

去重计算成本非常高，因此可以使用 minhashing 或并行处理等技术来扩展去重，以适应语料库数据的增加。

Minhashing 通过使用更小、更易于管理的表示来有效地近似文档之间的相似度，从而减少计算负载。并行处理进一步将去重任务分配到多个处理器或机器上，允许同时比较，从而显著加快整体过程，从而实现大规模语料库的有效去重。

这里有一个 Python 脚本演示了基本去重技术：

1.  首先，定义整体函数：

    ```py
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    def deduplicate_corpus(corpus, similarity_threshold=0.9):
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(tfidf_matrix)
    TfidfVectorizer to convert a text corpus into a numerical cosine_similarity to calculate the pairwise similarity between all documents in the corpus, providing a matrix of similarity scores that can be used to identify near-duplicate texts based on a specified threshold.
    ```

1.  然后，查找重复项：

    ```py
        duplicates = set()
        for i in range(len(corpus)):
            for j in range(i + 1, len(corpus)):
                if similarity_matrix[i, j] > similarity_threshold:
                    duplicates.add(j)
    ```

1.  创建去重语料库：

    ```py
        deduplicated_corpus = [
            doc for i, doc in enumerate(corpus) 
            if i not in duplicates
        ]
        return deduplicated_corpus
    ```

1.  这里有一个例子：

    ```py
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above the sleepy canine.",
        "The quick brown fox jumps over the lazy dog.",
        "An entirely different sentence about cats.",
    ]
    deduplicated = deduplicate_corpus(corpus)
    print(f"Original corpus size: {len(corpus)}")
    print(f"Deduplicated corpus size: {len(deduplicated)}")
    print("Deduplicated corpus:")
    for doc in deduplicated:
        print(f"- {doc}")
    ```

此脚本演示了使用 TF-IDF 和**余弦相似度**的基本近似重复检测方法。TF-IDF 是一种数值统计，用于反映集合中文档中单词的重要性。它结合了单词在文档中出现的频率（TF）以及在整个文档中其独特性（IDF）。TF-IDF 将文本转换为数值向量，使得可以在文档之间进行数学比较，这对于去重过程中使用的相似度计算至关重要。对于大规模去重，我们会使用更高效的算法和分布式计算技术。

在这里，去重函数代码中使用的相似度阈值`0.9`决定了文档必须有多相似才能被认为是重复的，默认要求 90%的相似度。此值可以根据具体用例进行调整——更高的阈值（例如，`0.95`或`1`，即最大值）更严格，减少了误报，而较低的阈值（例如，`0`即最小值或`0.8`）更宽松，可以捕获更多潜在的重复项。

接下来，让我们讨论自动化数据清洗管道。

# 自动化数据清洗管道

为了处理 LLM 训练所需的庞大数据集，我们需要实现自动化数据清洗流程。这些流程应该是可扩展的、高效的，并且能够处理各种数据质量问题。

自动化数据清洗流程的关键组件如下：

+   **数据摄取**：高效地加载和解析大型文本语料库。

+   **质量评估**：自动检测并标记数据质量问题。

+   **预处理**：应用文本清洗和规范化技术。

+   **去重**：移除完全重复和近似重复的内容。

+   **过滤**：根据预定义的标准移除低质量或不相关的样本。

+   **验证**：确保清洗后的数据符合质量标准。

+   **输出**：将清洗后的数据保存为 LLM 训练的适当格式。

下面是一个概述基本自动化数据清洗流程的 Python 脚本：

1.  我们首先定义整体类结构：

    ```py
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    # Download required NLTK data
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    class DataCleaningPipeline:
        def __init__(
            self, similarity_threshold=0.9, min_length=10,
            max_length=1000
        ):
            self.similarity_threshold = similarity_threshold
            self.min_length = min_length
            self.max_length = max_length
            self.vectorizer = TfidfVectorizer(stop_words='english')
    DataCleaningPipeline class that encapsulates text preprocessing, length filtering, and near-duplicate removal functionalities. It initializes with configurable parameters such as similarity threshold and text length constraints, leverages NLTK for stop word removal, and employs scikit-learn’s TfidfVectorizer and cosine_similarity to identify and eliminate similar text entries from a pandas DataFrame.
    ```

1.  然后，我们将定义一个预处理函数：

    ```py
        def preprocess(self, text):
            # Basic preprocessing
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            tokens = [
                word for word in text.split()
                if word not in stop_words
            ]
            return ' '.join(tokens)
        def filter_by_length(self, df):
            return df[
                (df['text'].str.len() >= self.min_length) &
                (df['text'].str.len() <= self.max_length)
            ]
     methods within a class for text processing.
    ```

    +   `preprocess`：此方法接收一个文本字符串作为输入，将其转换为小写，删除标点符号，将其拆分为单词，过滤掉常见的停用词，然后将剩余的单词连接成一个字符串，从而有效地清洗和规范化文本。

    +   `filter_by_length`：此方法接收一个包含`text`列的 pandas DataFrame，并过滤 DataFrame，仅包括`text`列长度在指定最小和最大长度范围内的行，从而允许选择所需字符范围内的文本样本。

1.  然后，我们定义去重函数：

    ```py
    def deduplicate(self, df):
        tfidf_matrix = self.vectorizer.fit_transform(df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)

        duplicates = set()
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarity_matrix[i, j] > \
                        self.similarity_threshold:
                    duplicates.add(j)

        return df.drop(df.index[list(duplicates)])
    ```

    这个`deduplicate`方法接收一个 pandas DataFrame 作为输入，并根据它们的相似性移除近似重复的文本条目。它首先使用向量器将 DataFrame 的`text`列转换为 TF-IDF 矩阵，将每个文本样本表示为一个数值向量。然后，它使用 TF-IDF 矩阵计算所有文本样本对之间的余弦相似度，从而得到一个相似度矩阵。代码遍历相似度矩阵，如果两个文本样本之间的相似度超过定义的`similarity_threshold`，则第二个样本的索引被添加到一个重复集。最后，它从 DataFrame 中删除对应于已识别重复索引的行，并返回去重后的 DataFrame。

1.  将所有函数组合起来，我们现在可以定义一个`clean`函数：

    ```py
        def clean(self, input_file, output_file):
            # Read data
            df = pd.read_csv(input_file)
            # Preprocess
            df['text'] = df['text'].apply(self.preprocess)
            # Filter by length
            df = self.filter_by_length(df)
            # Deduplicate
            df = self.deduplicate(df)
            # Save cleaned data
            df.to_csv(output_file, index=False)
            print(f"Cleaned data saved to {output_file}")
    ```

    这种`clean`方法在 CSV 文件上执行一系列数据清洗步骤。它首先将输入的 CSV 文件读取到一个 pandas DataFrame 中。然后，对`text`列中的每个文本条目应用`preprocess`方法，对文本进行归一化和清洗。随后，使用`filter_by_length`方法过滤 DataFrame，仅保留指定长度范围内的文本条目。长度过滤后，使用`deduplicate`方法移除近似重复的条目。最后，将清洗后的 DataFrame 保存到由`output_file`指定的新的 CSV 文件中，排除索引，并打印一个确认消息，指示输出文件的存储位置。本质上，此方法执行了一个完整的文本清洗流程，包括预处理、长度过滤和去重。

1.  以下是一个示例用法：

    ```py
    pipeline = DataCleaningPipeline()
    pipeline.clean('input_data.csv', 'cleaned_data.csv')
    ```

总体而言，此脚本提供了一个自动化数据清洗流程的基本框架。在实际应用中，我们会扩展此流程，以包含更复杂的清洗技术、错误处理和并行处理能力，以有效地处理大规模数据集。

代码中的值`10`和`1000`代表数据清洗流程中文本文档允许的最小和最大长度：

+   `min_length=10`：这设置了文档必须具有的最小字符数，才能包含在清洗后的数据集中。它有助于过滤掉可能不包含有意义信息的非常短的文本，例如单个单词或简短的短语。

+   `max_length=1000`：这确定了文档允许的最大字符数。它排除了可能不典型或可能对处理造成问题的极长文本，例如整本书或非常大的文档，这些文档可能会扭曲分析。

这些长度约束有助于确保清洗后的数据集包含合理且一致的文档大小范围，这可以提高后续文本分析或机器学习任务的质量和效率。您可以根据您的用例调整长度。

# 数据验证和质量保证

清洗数据后，您需要验证结果并确保清洗后的数据集符合 LLM 训练所需的质量标准。我们实施各种验证检查和质量保证措施，以验证我们清洗过程的有效性。

关键方面包括执行统计分析、抽样和人工审查、自动测试、一致性验证和性能影响评估。

下面是一个演示基本数据验证技术的 Python 脚本：

1.  首先，定义基本函数：

    ```py
    def validate_cleaned_data(file_path, sample_size=100):
        df = pd.read_csv(file_path)
        # Basic statistics
        print(f"Total samples: {len(df)}")    
        print(
            f"Average text length: "
            f"{df['text'].str.len().mean():.2f}"
        )

        print(f"Unique samples: {df['text'].nunique()}")
    ```

1.  然后，检查空或非常短的文本：

    ```py
    short_texts = df[df['text'].str.len() < 10]
    print(
        f"Texts shorter than 10 characters: "
        f"{len(short_texts)}"
    )
    ```

1.  进行人工审查的抽样：

    ```py
        sample = df.sample(n=min(sample_size, len(df)))
        print("\nSample for manual review:")
        print(sample['text'].head())
        # Check for common issues
      common_issues = {
            'special_chars': df['text'].str.contains(
                r'[^a-zA-Z0-9\s]'
            ),
            'numbers': df['text'].str.contains(r'\d'),
            'all_caps': df['text'].str.isupper()
        }
        for issue, mask in common_issues.items():
            print(f"Samples with {issue}: {mask.sum()}")
    ```

1.  评估模型困惑度的影响：

    ```py
        model = GPT4LMHeadModel.from_pretrained('GPT4')
        tokenizer = GPT4Tokenizer.from_pretrained('GPT4')
        def calculate_perplexity(text):
            inputs = tokenizer(
                text, return_tensors='pt', truncation=True, 
                    max_length=1024
            )
            with torch.no_grad():
                outputs = model(inputs, labels=inputs['input_ids'])
            return torch.exp(outputs.loss).item()
        sample_perplexities = sample['text'].apply(
            calculate_perplexity)
        print(
            f"\nAverage perplexity on sample: "
            f"{sample_perplexities.mean():.2f}"
        )
    ```

1.  让我们看看一个例子：

    ```py
    validate_cleaned_data('cleaned_data.csv')
    ```

该脚本定义了一个名为 `validate_cleaned_data` 的函数，该函数旨在对存储在 CSV 文件中的文本数据集（假设在初始清理步骤之后）进行基本质量评估。它加载数据，计算一些基本统计数据，检查文本内容中的特定潜在问题，提供样本以供人工检查，并使用预训练的语言模型（假设为 GPT-4）通过困惑度评估文本样本的自然度或质量。

检查以下问题：

+   数据集大小和基本属性：

    +   `len(df)`: 检查 CSV 中的样本总数（行数）。

    +   `df['text'].str.len().mean()`: 计算文本条目的平均长度，这有助于判断文本是普遍较长还是较短。

    +   `df['text'].nunique()`: 统计唯一文本条目的数量。与样本总数相比，低数值可能表明存在许多重复项。

+   `df[df['text'].str.len() < 10]`: 过滤 DataFrame 以找到`text`列中字符串长度小于 10 个字符的行*   `len(short_texts)`: 计算找到的此类短文本的数量*   `df['text'].str.contains(r'[^a-zA-Z0-9\s]')`: 使用 pandas 的`.str.contains()`方法和正则表达式（`r'[^a-zA-Z0-9\s]'`）。正则表达式模式`[^...]`匹配不在指定集合（a-z, A-Z, 0-9, 空白字符\s）中的任何字符.*   `mask.sum()`: 将结果布尔序列（true=`1`，false=0）求和，以计算包含至少一个此类特殊字符的文本数量.*   `df['text'].str.contains(r'\d')`: 使用`.str.contains()`和正则表达式`\d`（匹配任何数字）*   `mask.sum()`: 计算包含至少一个数字的文本数量*   `df['text'].str.isupper()`: 使用 pandas 的`.str.isupper()`字符串方法，如果字符串中的所有大小写字符都是大写并且至少有一个字母字符是大写的（即字母），则返回`True`。如果字符串全部是非字母字符（如数字或标点符号），则返回`False`——即使这些字符也不是小写的.*   `mask.sum()`: 计算完全为大写的文本数量*   `df.sample(...)`). 惊奇度计算可能很昂贵，因此通常在代表性样本上而不是整个数据集上进行计算.*   `GPT4LMHeadModel`)及其对应的分词器(`GPT4Tokenizer`)被加载。（注意：这里的`'GPT4'`是示例性的；你会使用实际的模型标识符，例如来自 Hugging Face Transformers 库的`'gpt2'`或`'bert-base-uncased'`。）*   `calculate_perplexity`函数对文本进行分词，将其输入到模型中，获取损失（衡量模型对文本感到惊讶的程度的一个指标），并使用`torch.exp(outputs.loss)`计算惊奇度.*   `sample_perplexities.mean()`)以获得一个代表样本平均质量的单一分数.*   `sample = df.sample(...)`: 从数据中随机抽取样本*   `print(sample['text'].head())`: 打印随机样本中的前几个文本条目，使用户运行脚本时可以快速查看一些示例

为了确保全面的质量保证，你可以执行以下操作：

+   实施针对你特定数据特征和清洗规则的更复杂的自动化测试。

+   制定一个系统的手动审查流程，包括为人类标注者提供评估数据质量的一致性指南。

+   使用已知存在问题的合成数据集来基准测试和评估管道的性能。

+   将清洗后的数据集与原始数据集进行比较，以验证在清洗过程中是否发生了意外的数据丢失或更改。

+   定期审计你的数据清洗管道，以识别在清洗过程中出现的任何新兴问题或偏差。

+   记录详细的清洁过程日志，包括做出的任何决策及其依据，以确保可重复性和便于未来的改进。

通过实施这些措施，你可以确保你的清洗数据集具有高质量且适合训练鲁棒的 LLMs。

# 摘要

在本章中，我们探讨了 LLM 训练中数据清洗的关键过程。我们讨论了清洁数据在开发鲁棒和可靠的语言模型中的重要性，并涵盖了针对语言数据集的常见数据质量问题。我们提供了解决这些问题的技术，包括文本预处理、处理多语言和代码混合数据以及大型文本语料库的去重策略。

我们还深入探讨了自动化数据清洗管道的实施，这对于处理 LLM 训练中使用的海量数据集至关重要。最后，我们讨论了数据验证和质量保证措施，以确保清洗过程的有效性。

在下一章中，我们将重点关注 LLMs 的数据增强模式。
