

# 第十五章：交叉验证

**交叉验证**是一种统计技术，用于评估机器学习模型对未见数据的泛化能力。它涉及将数据集划分为多个子集或“折”，在这些子集上训练模型，同时在剩余的子集上进行测试。这个过程会重复进行，以确保可靠的性能估计。这有助于检测过拟合，并提供比单一训练-测试分割更稳健的评估。在 LLM 的背景下，交叉验证必须适应解决预训练、微调、少样本学习和领域泛化的复杂性，使其成为评估模型在多种任务和数据分布上性能的必要工具。

在本章中，你将探索专门为 LLM 设计的交叉验证策略。我们将深入研究创建适当的预训练和微调数据分割的方法，以及少样本和零样本评估的策略。你将学习如何评估 LLM 中的领域和任务泛化，并处理 LLM 背景下交叉验证的独特挑战。

到本章结束时，你将掌握强大的交叉验证技术，以可靠地评估你的 LLM 在各种领域和任务上的性能和泛化能力。

在本章中，我们将探讨以下主题：

+   预训练和微调数据分割

+   少样本和零样本评估策略

+   领域和任务泛化

+   持续学习评估

+   交叉验证的挑战和最佳实践

# 预训练和微调数据分割

在 LLM 中，数据分割指的是将数据集划分为训练集、验证集和测试集，以确保模型学习可泛化的模式，而不是记住数据。这对于公平地评估性能、调整模型参数和防止数据泄露至关重要。在 LLM 中，适当的分割尤为重要，因为它们的规模很大，任务多样性高，并且需要评估领域和任务泛化。

## 预训练数据的分层抽样

**分层抽样**是一种抽样方法，首先根据共享特征将总体划分为更小的子组（**层**），然后从每个层中随机抽样以确保在最终样本中所有组按比例代表。这在处理不平衡数据集时特别有用。

在创建预训练数据分割时，确保每个分割代表整个数据集的多样性非常重要。以下是一个你可能用于预训练数据的**分层抽样**示例：

```py
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
def stratified_pretraining_split(
    data, text_column, label_column, test_size=0.1, random_state=42
):
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(
        data[text_column], data[label_column]
    ):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
    return train_data, test_data
# Example usage
data = pd.read_csv('your_pretraining_data.csv')
train_data, test_data = stratified_pretraining_split(
    data, 'text', 'domain')
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
```

此代码使用`StratifiedShuffleSplit`创建预训练数据的分层分割，确保训练集和测试集中领域的分布（或任何其他相关分类变量）相似。

## 基于时间的微调数据分割

对于涉及时间敏感数据的微调任务，使用**基于时间的分割**通常是有益的。

基于时间的分割是一种数据分区策略，其中数据集根据时间顺序进行划分，确保早期数据用于训练，而后期数据用于验证或测试。这种方法对于涉及时间敏感数据的微调任务尤为重要——例如金融预测、用户行为建模或事件预测——在这些任务中，未来的信息不应影响过去的训练。通过保留自然的时间序列，基于时间的分割有助于评估模型对未来未见场景的泛化能力，紧密模拟现实世界的部署。

这种方法有助于评估模型对未来数据的泛化能力：

```py
import pandas as pd
def time_based_finetuning_split(data, timestamp_column, split_date):
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    train_data = data[data[timestamp_column] < split_date]
    test_data = data[data[timestamp_column] >= split_date]
    return train_data, test_data
# Example usage
data = pd.read_csv('your_finetuning_data.csv')
split_date = '2023-01-01'
train_data, test_data = time_based_finetuning_split(
    data, 'timestamp', split_date)
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")
```

此函数根据指定的日期分割数据，这对于模型需要泛化到未来事件或趋势的任务特别有用。

## 数据平衡的过采样和加权技术

当处理类别分布不均的数据集时，例如不平衡领域或标签频率，**过采样**和**加权技术**可以帮助确保模型从所有类别中有效地学习。过采样涉及复制来自代表性不足类别的示例，以增加其在训练数据中的存在，防止模型忽略它们。这可以通过随机过采样或合成数据生成方法（例如，SMOTE 用于结构化数据）来完成。另一方面，加权技术通过为代表性不足的类别分配更高的重要性来调整损失函数，因此模型可以从它们中学习，而无需 necessarily 增加数据集的大小。两种方法都有助于减轻偏差，提高模型在所有类别中泛化的能力，而不是偏向最频繁的类别。

下面是一个简短的代码示例，展示了使用 PyTorch 和 sklearn 的过采样和类加权技术，应用于文本分类任务：

```py
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
# Example class distribution (e.g., from dataset labels)
labels = [0, 0, 0, 1, 1, 2]  # Class 2 is underrepresented
# --- 1\. Class Weighting ---
# Compute weights inversely proportional to class frequencies
class_weights = compute_class_weight(
    'balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)
# Pass weights to loss function
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
# --- 2\. Oversampling with Weighted Sampler ---
# Create sample weights: inverse of class frequency for each label
label_counts = np.bincount(labels)
sample_weights = [1.0 / label_counts[label] for label in labels]
# Create sampler for DataLoader
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(labels), replacement=True
)
# Use the sampler in your DataLoader
# Assuming `train_dataset` is a PyTorch Dataset object
train_loader = DataLoader(train_dataset, sampler=sampler,
    batch_size=4)
```

这段代码演示了两种解决分类任务中类别不平衡的常见技术：类别权重和过采样。首先，它使用 `sklearn` 的 `compute_class_weight` 来计算与类别频率成反比的权重，将更高的重要性分配给代表性不足的类别（例如，出现频率较低的类别 2）。这些权重被传递到 PyTorch 的 `CrossEntropyLoss`，因此在训练过程中，对稀有类别的误分类比常见类别的误分类对模型的惩罚更大。其次，它通过根据每个样本类别的逆频率计算每个样本的权重来进行过采样，这确保了在训练过程中，来自少数类别的样本有更高的概率被选中。这些样本权重用于初始化 PyTorch 的 `WeightedRandomSampler`，这使得 `DataLoader` 能够在类之间以平衡的方式采样训练数据，而无需实际复制数据。这些技术共同帮助模型学会公平地对待所有类别，从而提高其在不平衡数据集上的泛化能力。

# 少样本和零样本评估策略

少样本和零样本评估策略使大型语言模型（LLMs）能够在不进行大量重新训练的情况下跨任务泛化。零样本学习在无标签示例可用的情况下很有用，而少样本学习通过提供有限的指导来提高性能。这些方法是使 LLMs 能够适应现实世界应用并可扩展的关键。

下面是这两种策略的比较：

| **方面** | **零样本** | **少样本** |
| --- | --- | --- |
| **描述** | 无示例；模型必须仅从提示中推断任务 | 在提示中提供少量标签示例 |
| **优点** | 不需要标签数据，高度灵活 | 准确率更高，对任务理解更好 |
| **弱点** | 准确率较低，存在歧义风险 | 需要仔细选择示例，仍然不如微调有效 |
| **用例** | 开放式问答，常识推理，通用知识任务 | 文本分类，翻译，摘要，代码生成 |

表 15.1 – 少样本与零样本

让我们看看如何实现这些策略中的每一个。

## 少样本评估

在**少样本评估**中，我们在要求模型执行任务之前，向模型提供少量示例。以下是一个实现少样本评估的示例：

```py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
def few_shot_evaluate(
    model, tokenizer, task_description, examples, test_instance
):
    prompt = f"{task_description}\n\nExamples:\n"
    for example in examples:
        prompt += (
            f"Input: {example['input']}\n"
            f"Output: {example['output']}\n\n"
        )
    prompt += f"Input: {test_instance}\nOutput:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100,
            num_return_sequences=1)
    generated_text = tokenizer.decode(output[0],
        skip_special_tokens=True)
    return generated_text.split("Output:")[-1].strip()
# Example usage
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
task_description = "Classify the sentiment of the following movie reviews as positive or negative."
examples = [
    {"input": "This movie was fantastic!", "output": "Positive"},
    {"input": "I hated every minute of it.", "output": "Negative"}
]
test_instance = "The acting was superb, but the plot was confusing."
result = few_shot_evaluate(
    model, tokenizer, task_description, examples, test_instance
)
print(f"Few-shot evaluation result: {result}")
```

这段代码演示了如何使用预训练的 GPT-2 模型在情感分析任务上执行少样本评估。

`few_shot_evaluate` 函数接收一个 GPT-2 模型、分词器、任务描述、示例和一个测试实例作为输入。它构建一个 `tokenizer.encode`，将其转换为适合模型处理的数值标记。然后，该函数在 `torch.no_grad()` 块中使用 `model.generate` 来生成文本，不计算梯度，使推理更高效。模型生成的响应最长为 `100` 个标记，确保其简洁。随后，使用 `tokenizer.decode` 对生成的文本进行解码，通过设置 `skip_special_tokens=True` 来移除不需要的标记。最后，该函数提取响应中最后一个 `"Output:"` 发生之后的部分，以隔离模型生成的答案，并修剪任何额外的空白字符。这种方法有效地实现了 **少样本学习**，其中模型利用提供的示例来做出更明智的预测。

## 零样本评估

**零样本评估** 测试模型在没有特定示例的情况下执行任务的能力。以下是实现零样本评估的方法：

```py
def zero_shot_evaluate(
    model, tokenizer, task_description, test_instance
):
    prompt = f"{task_description}\n\nInput: {test_instance}\nOutput:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=100, num_return_sequences=1
        )
    generated_text = tokenizer.decode(output[0],
        skip_special_tokens=True)
    return generated_text.split("Output:")[-1].strip()
# Example usage
task_description = "Classify the following text into one of these categories: Science, Politics, Sports, Entertainment."
test_instance = "NASA's Mars rover has discovered traces of ancient microbial life."
result = zero_shot_evaluate(
    model, tokenizer, task_description, test_instance
)
print(f"Zero-shot evaluation result: {result}")
```

此函数演示了在文本分类任务上的零样本评估。

`zero_shot_evaluate` 函数使用 `test_instance` 执行 `task_description`，确保模型理解任务及其需要分类的内容。短语 `"Output:"` 被附加以指示模型应生成响应的位置。然后，使用分词器.encode 对提示进行标记化，将其转换为模型可以处理的数值输入张量。函数使用 `torch.no_grad()` 禁用梯度计算，使推理更高效。`model.generate` 函数接收标记化的提示，并生成一个最大长度为 `100` 个标记的输出序列，同时只返回一个序列。生成的输出随后使用 `tokenizer.decode` 解码回文本，确保移除任何特殊标记。最后，函数提取并返回出现在 `"Output:"` 之后的部分，这代表了模型的预测分类。在示例用法中，该函数应用于一个分类任务，模型被要求将给定的文本片段`——“NASA 的火星探测器发现了古代微生物生命的痕迹。”`分类到预定义的类别之一：`科学`、`政治`、`体育`或`娱乐`。模型没有看到任何标记的示例，根据其先验知识推断出正确的类别。然后输出被打印出来，展示了模型的零样本分类能力。

# 领域和任务泛化

评估一个大型语言模型（LLM）在不同领域和任务上的泛化能力对于理解其真实能力至关重要。让我们探讨一些用于此目的的技术。

## 评估领域自适应

为了评估 **领域自适应**，我们可以测试模型在训练领域之外的数据上的表现。以下是一个示例：

```py
def evaluate_domain_adaptation(
    model, tokenizer, source_domain_data, target_domain_data
):
    def predict(text):
        inputs = tokenizer(
            text, return_tensors='pt', truncation=True, padding=True
        )
        outputs = model(inputs)
        return torch.argmax(outputs.logits, dim=1).item()
    # Evaluate on source domain
    source_predictions = [
        predict(text) for text in source_domain_data['text']
    ]
    source_accuracy = accuracy_score(
        source_domain_data['label'], source_predictions
)
    # Evaluate on target domain
    target_predictions = [
        predict(text) for text in target_domain_data['text']
    ]
    target_accuracy = accuracy_score(
        target_domain_data['label'], target_predictions
)
    return {
        'source_accuracy': source_accuracy,
        'target_accuracy': target_accuracy,
        'adaptation_drop': source_accuracy - target_accuracy
    }
```

以下是我们评估领域自适应并输出结果的方法：

```py
source_domain_data = load_source_domain_data()  # Replace with actual data loading
target_domain_data = load_target_domain_data()  # Replace with actual data loading
results = evaluate_domain_adaptation(
    model, tokenizer, source_domain_data, target_domain_data
)
print(f"Source domain accuracy: {results['source_accuracy']:.2f}")
print(f"Target domain accuracy: {results['target_accuracy']:.2f}")
print(f"Adaptation drop: {results['adaptation_drop']:.2f}")
```

将前面的代码整合起来，我们可以使用它来评估模型在源域（它所训练的域）和目标域上的性能，计算性能下降作为领域自适应的度量。

## 评估任务泛化

为了评估**任务泛化能力**，我们可以评估模型在它没有特定微调的各种任务上的表现。以下是一个使用 GLUE 基准（我们在*第十四章*中讨论过）的例子：

```py
def evaluate_task_generalization(
    model_name, tasks=['mnli', 'qqp', 'qnli', 'sst2']
):
    results = {}
    for task in tasks:
        model = \
            AutoModelForSequenceClassification.from_pretrained(
            model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = load_dataset('glue', task)
        def tokenize_function(examples):
            return tokenizer(
                examples['sentence'], truncation=True, padding=True)
        tokenized_datasets = dataset.map(tokenize_function,
            batched=True)
        training_args = TrainingArguments(
            output_dir=f"./results_{task}",
            evaluation_strategy="epoch",
            num_train_epochs=1,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
        )
        eval_results = trainer.evaluate()
        results[task] = eval_results['eval_accuracy']
    return results
```

以下是基于先前定义的函数运行评估的方法：

```py
model_name = "bert-base-uncased"  # Replace with your model
generalization_results = evaluate_task_generalization(model_name)
for task, accuracy in generalization_results.items():
    print(f"{task} accuracy: {accuracy:.2f}")
```

将前面的代码整合起来，我们可以评估模型在多个 GLUE 任务上的表现，以评估其跨不同 NLP 任务的泛化能力。

# 持续学习评估

**持续学习**是指模型在不会忘记先前学习的内容的情况下学习新任务的能力。以下是如何在 LLMs 中评估持续学习的例子：

1.  通过初始化模型、分词器和主要函数结构来设置我们的持续学习框架：

    ```py
    def evaluate_continual_learning(
        model_name, tasks=['sst2', 'qnli', 'qqp'], num_epochs=3
    ):
        model = \
            AutoModelForSequenceClassification.from_
            pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        results = {}
    ```

1.  定义预处理函数，以处理各种 GLUE 任务的不同输入格式：

    ```py
    def preprocess_function(examples, task):
        # Different tasks have different input formats
        if task == 'qqp':
            texts = (examples['question1'], examples['question2'])
        elif task == 'qnli':
            texts = (examples['question'], examples['sentence'])
        else:  # sst2
            texts = (examples['sentence'], None)
        tokenized = tokenizer(*texts, padding=True, truncation=True)
        tokenized['labels'] = examples['label']
        return tokenized
    ```

1.  预处理并准备每个任务的训练数据集：

    ```py
    for task in tasks:
        dataset = load_dataset('glue', task)
        tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, task),
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        model.config.num_labels = 3 if task == 'mnli' else 2
    ```

1.  提供每个任务的训练设置和执行：

    ```py
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./results_{task}",
            num_train_epochs=num_epochs,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch"
        ),
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation']
    )
    trainer.train()
    ```

1.  在所有先前看到的任务上执行评估：

    ```py
    task_results = {}
    for eval_task in tasks[:tasks.index(task)+1]:
        eval_dataset = load_dataset('glue', eval_task)['validation']
        eval_tokenized = eval_dataset.map(
            lambda x: preprocess_function(x, eval_task),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        eval_results = trainer.evaluate(eval_dataset=eval_tokenized)
        task_results[eval_task] = eval_results['eval_accuracy']
    results[task] = task_results
    ```

1.  运行评估并显示结果：

    ```py
    model_name = "bert-base-uncased"  # Replace with your model
    cl_results = evaluate_continual_learning(model_name)
    for task, task_results in cl_results.items():
        print(f"\nAfter training on {task}:")
        for eval_task, accuracy in task_results.items():
            print(f"  {eval_task} accuracy: {accuracy:.2f}")
    ```

将前面的代码块整合起来，我们展示了如何对一系列任务进行微调，并在每次微调步骤之后评估模型在所有先前看到的任务上的性能，从而评估它保留早期任务知识的能力。

# 交叉验证的挑战和最佳实践

由于 LLMs 的规模和训练数据的性质，它们在交叉验证中面临独特的挑战。以下是一些关键挑战：

+   **数据污染**：由于 LLM 训练所依赖的互联网数据非常庞大且多样化，避免测试集与预训练数据重叠变得困难，这使得确保一个真正未见过的验证集变得很困难

+   **计算成本**：由于需要巨大的计算资源，传统的 k 折交叉验证方法通常不可行

+   **领域偏移**：当 LLM 接触到来自代表性不足或全新的领域的数据时，可能会表现出不一致的性能，这复杂了泛化能力的评估

+   **提示敏感性**：LLMs 的性能可能会根据提示措辞的微妙差异而有很大差异，这为验证过程增加了另一层可变性

基于这些挑战，以下是 LLM 交叉验证的一些最佳实践：

+   **减轻数据污染**：使用严格的数据去重方法来识别和删除预训练语料库和验证数据集之间的重叠。例如，MinHash 或 Bloom 过滤器可以在大型数据集中有效地检测近重复项。

MinHash

**MinHash**是一种概率技术，通过将大型集合转换为较小的、代表性的指纹（**散列**），快速估计两个集合的相似度，其中散列冲突的概率与原始集合之间的相似度成比例，这使得它在检测大型数据集中的近似重复内容时特别有用。

**MinHashLSH**基于 MinHash 和**局部敏感哈希**（**LSH**），它将相似项分组到相同的“桶”中，以实现快速查找和比较。

以下代码示例演示了使用 MinHash 和 MinHashLSH 进行数据去重，以检测数据集中的近似重复项：

```py
from datasketch import MinHash, MinHashLSH
import numpy as np
def deduplicate_data(texts, threshold=0.8):
    # Initialize LSH index for fast similarity search
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    for idx, text in enumerate(texts):
        minhash = MinHash(num_perm=128)
        for ngram in get_ngrams(text):
            minhash.update(ngram.encode('utf8'))
        if not lsh.query(minhash):  # Check if similar text exists
            lsh.insert(str(idx), minhash)
            unique_texts.append(text)
    return unique_texts
```

+   **降低计算成本**：使用分层抽样或单一分割验证方法（例如，训练-验证-测试）以最小化计算开销。或者，在扩展之前，在实验中使用较小的模型检查点或 LLM 的蒸馏版本。

    以下代码示例展示了用于高效验证的分层抽样：

    ```py
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict
    def create_efficient_splits(data, labels, n_splits=5):
        # Group data by domain
        domain_data = defaultdict(list)
        for text, domain in zip(data, labels):
            domain_data[domain].append(text)
        # Create stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        splits = []
        for train_idx, val_idx in skf.split(data, labels):
            splits.append((train_idx, val_idx))
        return splits
    ```

+   **处理领域偏移**：构建具有来自不同领域显式表示的验证数据集。使用具有代表性的领域特定数据微调模型，以减少在代表性不足区域中的性能差距。

    此代码示例演示了通过领域特定验证处理领域偏移：

    ```py
    def evaluate_domain_performance(model, tokenizer, eval_data):
        domain_scores = defaultdict(list)
        for text, domain in eval_data:
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(inputs)
                score = outputs.logits.mean().item()
                domain_scores[domain].append(score)
        # Calculate domain-specific metrics
        return {domain: np.mean(scores)
            for domain, scores in domain_scores.items()}
    ```

+   **解决提示敏感性**：系统性地进行提示工程。使用提示释义、指令调整或跨多个提示的集成评估等技术，以确保鲁棒性并最小化提示变化引入的变异性。

    以下代码示例展示了使用多个变体的系统提示工程：

    ```py
    def evaluate_with_prompt_ensemble(
        model, tokenizer, text, base_prompt
    ):
        prompt_variants = [
            f"{base_prompt}: {text}",
            f"Please {base_prompt.lower()}: {text}",
            f"I want you to {base_prompt.lower()}: {text}"
        ]
        responses = []
        for prompt in prompt_variants:
            inputs = tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                output = model.generate(inputs, max_length=100)
                responses.append(tokenizer.decode(output[0]))
        # Aggregate responses (e.g., by voting or averaging)
        return aggregate_responses(responses)
    ```

以下代码示例展示了如何将这些方法组合成一个单一的评估流程：

```py
def robust_evaluation_pipeline(model, data, domains):
    # First deduplicate the data
    clean_data = deduplicate_data(data)
    # Create efficient splits
    splits = create_efficient_splits(clean_data, domains)
    # Evaluate across domains with prompt ensembles
    results = defaultdict(dict)
    for domain in domains:
        domain_data = [d for d, dom in zip(clean_data, domains)
            if dom == domain]
        scores = evaluate_with_prompt_ensemble(model, tokenizer,
            domain_data, "analyze")
        results[domain] = scores
    return results
```

# 摘要

对于 LLM 的交叉验证需要仔细考虑它们的独特特性和能力。通过实施这些高级技术和最佳实践，您可以获得对 LLM 在各个领域和任务中性能的更稳健和全面的评估。

随着我们继续前进，下一章将深入探讨 LLM 中解释性的关键主题。我们将探讨理解和解释 LLM 输出和行为的技术。
