# 3

# 数据增强

数据增强在增强 LLMs（大型语言模型）的性能和泛化能力方面发挥着关键作用。通过人工扩大训练数据集，我们可以让我们的模型接触到更广泛的语言变化和上下文，提高它们处理各种输入和生成更连贯、上下文相关输出的能力。

在 LLMs 的背景下，数据增强面临着独特的挑战和机遇。与**图像数据**不同，图像数据可以通过简单的变换（如旋转或翻转）来创建有效的新的样本，**文本数据**需要更细致的方法来保持语义完整性和语言连贯性。LLMs 数据增强的主要目标包括增加数据集大小和多样性，解决数据不平衡和偏差问题，提高模型对输入变化的鲁棒性，以及增强对未见数据的泛化能力。

在*图 3.1*中，我展示了数据增强的关键方面。

![图 3.1 – 数据增强的关键元素](img/B31249_03_001.jpg)

图 3.1 – 数据增强的关键元素

有三个主要组成部分，即**技术**、**考虑因素**和**评估**。每个部分都有具体的子组件，我们将在本章中详细讨论。

到本章结束时，您将深入了解数据增强模式，从增加训练数据集的多样性到保持其完整性：

+   文本数据增强技术

+   利用现有 LLMs 进行数据生成

+   多语言数据增强策略

+   文本增强中的语义保留

+   平衡增强和数据质量

+   评估数据增强的影响

# 文本数据增强技术

文本数据增强包括一系列技术，从简单的单词级别操作到更复杂的语义转换。

## 同义词替换

这种技术涉及用同义词替换原始文本中的单词。我们可以使用**WordNet**，一个英语词汇数据库，来查找同义词：

```py
def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(
        set([word for word in words if word.isalnum()])
    )
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [
                synonym if word == random_word else word
                for word in new_words
            ]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)
```

`synonym_replacement`函数接受一个文本输入，并用同义词替换指定数量的单词（默认为 1）。选择默认值 1 是为了最小化文本修改，保留意义和可读性，同时允许轻松实验。如果您想进行更多替换，可以增加这个数字。

函数将文本拆分为单词，创建一个唯一的字母数字单词列表，然后打乱这个列表，并遍历它。对于每个单词，它尝试使用一个未定义的`get_synonyms`函数来查找同义词。如果找到了同义词，它将随机选择一个并替换文本中所有原始单词的出现。该函数会跟踪已替换的单词数量，并在达到指定数量时停止。最后，它将修改后的单词重新组合成一个字符串并返回。

## 反向翻译

此方法涉及将文本翻译成另一种语言，然后再翻译回原始语言。这对于引入句子结构和词汇选择中的自然变化特别有效：

```py
def back_translation(text, target_lang='fr'):
    translator = Translator()
    translated = translator.translate(text, dest=target_lang)
    back_translated = translator.translate(translated.text, dest='en')
    return back_translated.text
```

## 使用 T5 进行文本生成

由谷歌研究开发的**文本到文本迁移转换器**（**T5**）模型是一个基于转换器架构的多功能**自然语言处理**（**NLP**）模型。其关键创新是将所有 NLP 任务框架化为文本到文本问题，这使得它能够处理多个任务而无需特定于任务的架构。使用“跨度损坏”目标在大型网络文本语料库上预训练，T5 有多种尺寸，并在广泛的 NLP 任务中展示了强大的性能。

T5 通过将所有基于文本的任务框架化为文本到文本问题来处理广泛的文本任务。这意味着无论任务是什么，无论是摘要、翻译、问答还是分类，输入和输出都被视为文本。这种统一的方法使得 T5 能够在无需特定于任务的修改的情况下执行各种任务，使其高度适应不同的用例。

当谈到数据增强时，T5 通过生成现有文本数据的变体，在扩展和多样化数据集方面发挥着关键作用。数据增强在训练机器学习模型时尤其有价值，因为它通过让模型接触到更广泛的示例来帮助它们更好地泛化，减少过拟合并提高鲁棒性。以下是 T5 如何帮助数据增强的说明：

+   **释义**：T5 可以在保持原意的同时重新表述句子。例如，如果输入是“这部电影很无聊”，T5 可以生成一个释义版本，如“这部电影很乏味。”这种表达方式的多样性为模型提供了额外的学习示例，有助于它更好地泛化到不同的表述方式。

+   **同义词替换**：T5 可以用同义词替换单词，在保留整体情感或上下文的同时，创造轻微的意义变化。例如，从“这部电影很长且无聊”中，T5 可能会生成“这部电影很冗长且乏味。”这种简单的修改增加了数据集的多样性，为依赖于理解语言微小变化的模型提供了更多的训练示例。

+   **基于情感的转换**：T5 还可以转换句子的情感。例如，给定一个负面句子，如“这部电影非常令人失望”，T5 可以生成一个中立或正面的版本，如“这部电影开始得很慢，但后来有所改进。”这种能力允许在不同情感类别中创建多个示例，这在如情感分析等任务中特别有用，在这些任务中，模型需要区分积极、中立和消极的情感。

+   **文本扩展**：T5 可以接受简短的句子并通过添加更多上下文、细节或描述来扩展它。例如，从句子“事件很棒”中，T5 可以生成一个更详细的版本，如“事件很棒，有出色的演讲和引人入胜的讨论。”通过添加更多上下文，T5 提供了句子的额外变体，有助于训练模型处理更复杂的输入。

我们可以使用预训练的 T5 模型生成输入文本的变体。这种方法特别强大，因为它可以产生更多样化和上下文丰富的增强。让我们看看这个例子：

```py
def t5_augmentation(text, model, tokenizer, num_return_sequences=1):
    input_ids = tokenizer.encode(
        f"paraphrase: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model.generate(
        input_ids=input_ids,
        max_length=150,
        num_return_sequences=num_return_sequences,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
    )
    return [
        tokenizer.decode(
            output, skip_special_tokens=True
        ) for output in outputs
    ]
```

此函数接受文本输入、预训练的 T5 模型、其分词器以及要生成的释义数量（默认为 1）。默认的 1 个返回序列是为了简单起见，但你可以通过增加此值来请求多个释义。

函数使用`"paraphrase:"`前缀对输入文本进行编码，限制其长度为`512`个标记。然后使用模型生成最大长度为`150`个标记的释义。生成过程使用 5 个 beam 的 beam 搜索，防止 2-gram 重复，并应用`50`) 和 `0.95`) `512`, `150`, `5`, `2`, `50`, `0.95`)，这些参数也可以根据具体用例进行调整，以控制生成释义的长度、多样性和质量。

函数解码并返回生成的释义，跳过在过程中添加的任何特殊标记。

在语言生成系统中使用温度控制作为额外的参数，允许微调创造性和连贯性之间的平衡。温度是一个介于 0 到 1 之间的标量值，它在生成过程中影响下一个标记的概率分布。低值（接近 0）使分布集中，使模型更确定性和连贯，但可能重复或保守。高值（接近 1）使分布平坦，增加随机性和多样性，但牺牲了连贯性。

# 利用现有 LLMs 进行数据生成

对于 LLMs 的数据增强，最强大的方法之一是使用现有模型生成新的训练示例。这种技术通常被称为**自监督学习**或**基于模型的 数据增强**，它使我们能够创建大量多样化、高质量的训练数据。

我们将探讨如何使用**GPT-4o**和**OpenAI API**进行数据生成：

```py
def gpt4o_data_generation(prompt, num_samples=5):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        n=num_samples,
        temperature=0.7,
    )
    return [choice.message.content.strip() 
        for choice in response.choices
    ]
```

此函数发送包含提供的提示的单个用户消息，用于聊天完成请求。它将响应限制在最多 `150` 个标记，这平衡了获得实质性响应和控制输出长度的需求。`n` 参数，设置为 `num_samples`，决定了要生成的替代完成内容的数量。使用 `0.7` 的温度，这为生成的文本提供了创造性和连贯性之间的平衡：高值增加随机性，而低值会使输出更确定。然后函数提取并返回每个生成的完成内容的文本，去除任何前导或尾随空白。这些参数（`150` 个标记，`0.7` 温度）可以根据输出长度和创造性的具体需求进行调整。

当使用这种方法时，我们需要考虑以下因素：

+   **提示工程**：需要精心设计提示以生成相关和多样化的样本。

+   **质量控制**：实施过滤机制以确保生成数据符合您的质量标准。

+   **多样性**：使用温度和 top-p 采样来控制生成样本的随机性和多样性。

我们已经探讨了使用 GPT-4o 的数据增强技术并检查了基本考虑因素。现在，让我们将注意力转向多语言数据增强的策略。

# 多语言数据增强策略

对于旨在处理多种语言的 LLM，多语言数据增强是必不可少的。我们可以调整我们之前的技术以跨语言工作。

## 跨语言回译

在将其翻译回原始语言之前，将文本翻译成多种语言：

```py
def cross_lingual_back_translation(text, 
    target_langs=['fr', 'de', 'es']
):
    translator = Translator()
    augmented_texts = []
    for lang in target_langs:
        translated = translator.translate(text, dest=lang)
        back_translated = translator.translate(
            translated.text, dest='en'
        )
        augmented_texts.append(back_translated.text)
    return augmented_texts
```

`cross_lingual_back_translation` 函数接收一个文本输入，通过首先将其翻译成多种目标语言（默认为法语、德语和西班牙语），然后将其翻译回英语来生成其增强版本。该函数使用 `Translator` 对象执行这些翻译，将每个回译版本存储在一个列表中，并将其作为输出返回。

## 多语言 T5 增强

您可以使用多语言 T5 模型在不同语言中生成释义：

```py
def multilingual_t5_augmentation(
    text, model, tokenizer, target_langs=['fr', 'de', 'es']
):
    augmented_texts = []
    for lang in target_langs:
        input_ids = tokenizer.encode(
            f"translate English to {lang}: {text}",
            return_tensors="pt", max_length=512,
            truncation=True
        )
        outputs = model.generate(input_ids=input_ids, max_length=150)
        translated = tokenizer.decode(outputs[0],
            skip_special_tokens=True)
        augmented_texts.append(translated)
    return augmented_texts
```

`multilingual_t5_augmentation` 函数使用 T5 模型通过将其翻译成多种目标语言（默认为法语、德语和西班牙语）来增强给定的文本。对于每种目标语言，它使用翻译提示对文本进行编码，使用模型生成翻译输出，并解码结果。翻译的文本被收集在一个列表中，并作为原始文本的增强版本返回。

# 文本增强中的语义保留

在为 LLM 增强数据时保持语义完整性至关重要。我们必须确保我们的技术不会改变文本的原始含义。

## 句子嵌入的使用

通过比较原始文本和增强文本的 **句子嵌入**，您可以确保 **语义相似性**：

```py
def semantic_similarity(original, augmented, model):
    original_embedding = model.encode(original)
    augmented_embedding = model.encode(augmented)
    similarity = cosine_similarity(
        [original_embedding], [augmented_embedding]
    )[0][0]
    return similarity
def filter_by_semantic_similarity(
    original, augmented_list, model, threshold=0.8
):
    return [
        aug for aug in augmented_list
        if semantic_similarity(original, aug, model) >= threshold
    ]
```

我们定义了两个用于根据语义相似度测量和过滤文本的函数：

+   `semantic_similarity(original, augmented, model)` 使用两个文本嵌入的余弦相似度计算两个文本之间的语义相似度。它使用提供的模型（可能是句子嵌入模型）将原始文本和增强文本编码为向量表示。然后计算这些向量之间的余弦相似度，得到一个介于 -1 和 1 之间的值，其中 1 表示完美相似。

+   `filter_by_semantic_similarity(original, augmented_list, model, threshold=0.8)` 根据与原始文本的语义相似度过滤增强文本列表。`semantic_similarity` 函数将每个增强文本与原始文本进行比较。默认阈值设置为 `0.8`：默认情况下，它将仅保留与原始文本相似度达到 `0.8` 或更高的增强文本。此阈值在 NLP 任务中常用，因为它通常表示强语义相似度，同时允许一些变化。可以根据您希望过滤有多严格或多宽松来调整此阈值：更高的阈值将导致更多相似（但可能更少）的增强；更低的阈值将允许更多样化（但可能不太相关）的增强。

## 用于同义词替换的上下文词嵌入

您可以使用 **上下文词嵌入** 来根据上下文找到更合适的同义词。上下文词嵌入是指使用语言模型生成的词表示，这些表示捕获了单词在其特定句子或段落中的意义，而不是将单词视为具有固定意义。与传统的静态嵌入不同，其中单词的向量无论在什么上下文中都相同，上下文嵌入根据其周围的单词为相同的单词分配不同的向量。这允许进行更准确的同义词替换，因为所选的同义词不仅与词典意义相符，而且与单词在特定上下文中的使用方式相符。例如，“bank”在“river bank”与“savings bank”中的表示就不同，这会导致上下文适当的同义词建议，如“shore”或“financial institution”。以下代码片段显示了它是如何工作的：

```py
def contextual_synonym_replacement(text, model, tokenizer, n=1):
    words = text.split()
    new_words = words.copy()
    for i in range(n):
        word_index = random.randint(0, len(words) - 1)
        original_word = words[word_index]
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs)
        word_embedding = outputs.last_hidden_state[0, word_index]
        similar_words = find_similar_words(
            word_embedding, model, tokenizer
        )
        if similar_words:
            new_words[word_index] = random.choice(similar_words)
    return ' '.join(new_words)
```

此函数使用语言模型进行上下文感知的单词替换：

1.  它接受文本输入、预训练的语言模型、其分词器和要替换的单词数量（默认为 1）。

1.  将文本拆分为单词，并创建一个用于修改的副本。

1.  函数迭代 `n` 次（默认为 1）。每次执行以下操作：

    1.  随机选择一个单词索引

    1.  对整个文本进行分词

    1.  将其通过模型运行以获取上下文嵌入

    1.  提取所选单词的嵌入

    1.  根据此嵌入查找相似单词（使用未定义的 `find_similar_words` 函数）

    1.  如果找到相似单词，则随机选择一个来替换原始单词

1.  最后，它将修改后的单词重新组合成一个字符串并返回。

默认的`n`=1 是为了在引入变化的同时做出最小的改变。这保留了大部分原始意义和结构。您可以增加`n`以获得更多的替换，但更高的值可能会更显著地改变文本的意义。

与简单的同义词替换相比，这种方法更注重上下文，因为它在寻找替换词时考虑了单词在全文中的使用情况。确切的行为将取决于所使用的模型和分词器，以及`find_similar_words`函数的实现。

# 平衡增强和数据质量

虽然数据增强可以显著提高大型语言模型（LLM）的性能，但我们需要在数量和质量之间取得平衡。

您应该限制训练集中增强数据的比例。一种常见的做法是开始时以原始数据与增强数据 1:1 的比例开始，并根据模型性能进行调整。

## 质量过滤

您可以实施质量检查以过滤掉低质量的增强样本：

```py
def quality_filter(
    augmented_texts, original_text,
    similarity_threshold=0.8, perplexity_threshold=100
):
    filtered_texts = []
    for aug_text in augmented_texts:
        if (
            semantic_similarity(
                original_text, aug_text, similarity_model
            ) >= similarity_threshold and
            calculate_perplexity(
                aug_text, perplexity_model
            ) <= perplexity_threshold
        ):
            filtered_texts.append(aug_text)
    return filtered_texts
```

## 人工参与循环验证

对于关键应用，将人工验证纳入您的增强流程中。

**人工参与循环**（**HITL**）验证是一种在人工智能流程中使用的控制机制，其中人类被故意插入到自动化工作流程中，以确保正确性，尤其是在涉及主观判断、敏感内容或关键决策的任务中。这在数据质量直接影响安全、公平或合规性的应用中尤为重要——例如，医疗诊断、法律文件分析或自主系统。在数据增强的背景下，其目标是通过对现有样本生成变体来扩展训练数据集，HITL 用于验证生成的样本是否连贯、准确，并与预期的标签或任务保持一致：

```py
def human_validation(augmented_texts):
    validated_texts = []
    for text in augmented_texts:
        if input(
            f"Is this text valid? (y/n)\n{text}\n"
        ).lower() == 'y':
            validated_texts.append(text)
    return validated_texts
```

此函数旨在通过从人类操作员那里获取二元反馈（是或否）来手动验证一系列增强文本样本。它在增强流程中的存在承认了并非所有自动生成数据都可以仅凭表面价值信赖。保留或丢弃给定样本的决定是交互式做出的，这加强了在语义完整性不可协商的任务中的人类监督。

函数循环的每一迭代代表一个决策点。人类验证者会看到生成的文本，并被要求评估它是否符合预期的标准。这些标准通常基于特定任务的要求，如语法正确性、与原始数据的语义等效性、语气适当性或领域一致性。例如，在医疗文本分类任务中，改写的句子必须保留所有关键的临床实体。如果不在验证期间捕捉到，增强技术引入的术语上的微小变化可能会误导模型。这就是人类评估变得不可或缺的地方。

将输入转换为小写的逻辑是为了处理不一致的用户输入。无论用户输入`Y`、`y`或任何其他大小写，比较都变得不区分大小写。只有当输入等同于`y`时，函数才接受样本。这种二进制检查故意严格，以防止模糊的批准。被拒绝的样本被静默丢弃，不记录或返回，这意味着任何进一步检查或更正被拒绝样本都需要单独实现。

函数通过返回一个仅包含明确验证的样本列表来结束。然后可以使用这些输出以更高的信心扩展训练数据集。重要的是，这种方法并不取代自动质量检查，而是在高风险应用中补充它们。在部署模型的环境中使用 HITL 验证特别有用，在这些环境中，假阳性或假阴性具有高昂的成本，例如法律推荐系统、欺诈检测或自主导航。人工验证过程有助于减轻过度依赖缺乏明确语义保证的生成增强方法所带来的风险。

在一个更大的系统中，这类功能通常会嵌入到一个更广泛的流程中，其中自动过滤器首先筛选出明显低质量或不相关的增强。人工验证员只会评估边缘或高影响案例。为了提高操作效率，交互通常通过网页界面或集成注释工具而不是命令行提示来处理。然而，这个功能以最简单的方式展示了原理：在将增强数据纳入模型训练之前，人类判断被用作质量最终裁决者。

# 评估数据增强的影响

为了评估我们数据增强技术的有效性，我们需要评估它们对大型语言模型（LLM）性能的影响。

## 感疑度

您可以在数据增强前后，在保留的测试集上测量模型的可疑度（见*第二章*），以评估它是否提高了模型预测未见文本的能力：

```py
def evaluate_perplexity(model, tokenizer, test_data):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in test_data:
            inputs = tokenizer(
                text, return_tensors="pt"
            ).to(model.device)
            outputs = model(inputs, labels=inputs["input_ids"])
            total_loss += (
                outputs.loss.item() * inputs["input_ids"].size(1)
            )
            total_tokens += inputs["input_ids"].size(1)
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity
```

这个函数`evaluate_perplexity`计算给定测试数据集上语言模型的疑惑度。以下是分解：

1.  它接受一个预训练的语言模型、其分词器和测试数据集作为输入。

1.  模型被设置为评估模式以禁用 dropout 和其他特定于训练的行为。

1.  它初始化变量以跟踪总损失和总处理令牌数。

1.  对于测试数据中的每个文本，执行以下操作：

    1.  文本被分词并转换为张量。

    1.  模型处理输入，计算损失。

    1.  损失被累积，并按输入中令牌的数量加权。

1.  处理完所有文本后，它使用以下公式计算疑惑度：`exp(total_loss / total_tokens)`。

此实现以零样本方式使用模型，将每个输入视为上下文和预测的目标。使用`torch.no_grad()`确保不计算梯度，使评估更高效。

此函数假设模型和数据兼容（即模型可以处理数据的最大序列长度）。在实际应用中，您可能需要添加检查或截断以处理非常长的序列。

## 特定任务指标

您可以对与您的用例相关的下游任务进行模型评估，例如文本分类或问答：

```py
def evaluate_classification(
    model, tokenizer, test_data, test_labels
):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text in test_data:
            inputs = tokenizer(
                text, return_tensors="pt"
            ).to(model.device)
            outputs = model(inputs)
            predictions.append(torch.argmax(outputs.logits).item())
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='weighted')
    return accuracy, f1
```

此函数评估分类模型在测试数据集上的性能：

1.  它接受一个预训练的分类模型、其分词器、测试数据（文本）和相应的测试标签作为输入。

1.  模型设置为评估模式以禁用 dropout 和其他特定于训练的行为。

1.  它处理测试数据中的每个文本，对其进行分词，并使用模型进行预测。

1.  处理完所有文本后，它计算两个评估指标：

    +   **准确率**：所有预测中正确预测的比例。

    +   **F1 分数**：模型精确率和召回率的平衡度量。F1 分数是**精确率**（所有正预测中真正预测的比例）和**召回率**（所有实际正实例中真正预测的比例）的调和平均数。

    F1 分数的公式是 *F1 = 2 * (精确率 * 召回率) / (精确率 + 召回率)*。

    F1 分数的范围从 0 到 1，其中 1 表示完美的精确度和召回率。对于仅准确率可能具有误导性的不平衡数据集，它特别有用。加权平均计算每个类的 F1 分数，并按每个类中实例的数量加权平均。

1.  函数返回准确率和 F1 分数，提供了对模型在可能不平衡的类别上的性能的更全面评估。

此实现还使用`torch.no_grad()`以提高效率，并假设已导入必要的 scikit-learn 指标。在实际应用中，您可能需要添加错误处理以处理意外的模型输出或预测/标签计数不匹配。

## 多样性指标

评估增强数据集的多样性很重要：

```py
def calculate_diversity_metrics(texts):
    all_words = [word for text in texts for word in text.split()]
    vocab_size = len(set(all_words))
    all_trigrams = [text[i:i+3] for text in texts 
        for i in range(len(text)-2)]
    unique_trigrams = len(set(all_trigrams))
    return {
        "vocabulary_size": vocab_size,
        "unique_trigrams": unique_trigrams
    }
```

此函数接受一组文本作为输入，并计算**多样性指标**。一旦完成，此函数将返回一个包含这两个指标的字典：

+   **词汇量大小**（范围从 1 到总单词数）：这可以给出词汇多样性的概念。高数值表明文本中使用了多样化的词汇。此指标将每个文本拆分为单词，然后将所有文本中的所有单词合并，并使用**集合**来计算唯一单词的数量。在此上下文中，集合指的是一种数据结构，它自动删除重复元素。

+   **独特的三元组**（范围从 1 到三元组的总数）：这些指标表示字符级别的多样性。高数值表明字符序列多样化，可能表明句子结构或词汇选择多样化。此指标通过从每个文本中创建三元组（三个字符的序列）并使用仅包含唯一元素的集合来计算独特三元组的数量。

这些指标可用于比较原始文本与增强文本之间的多样性，或评估数据集中的多样性。然而，结果应在特定背景下进行解读，因为高多样性可能表明数据中的不连贯性或噪声。

通过系统地应用这些技术，我们可以量化我们的数据增强策略对 LLM 性能的影响，并就使用哪些技术和如何微调我们的增强流程做出明智的决定。

# 摘要

在本章中，我们探讨了针对 LLM 的高级数据增强技术，包括文本操作方法、利用现有模型进行数据生成、多语言策略、语义保留、质量控制以及多个指标。我们还讨论了平衡增强与数据质量的重要性，并提供了各种技术的实用 Python 实现。

在下一章中，我们将专注于处理 LLM 训练的大型数据集。
