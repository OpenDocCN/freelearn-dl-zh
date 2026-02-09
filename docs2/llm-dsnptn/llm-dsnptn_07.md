

# 第七章：训练管道

在本章中，我们将探讨 LLM 训练管道的关键组成部分，从数据摄取和预处理到模型架构和优化策略。

你将深入了解实施有效的监控和记录系统，确保你可以在整个训练过程中跟踪你的模型进度并做出数据驱动的决策。

在本章中，我们将涵盖以下主题：

+   训练管道的组成部分

+   数据输入和预处理

+   LLM 架构设计考虑因素

+   损失函数和优化策略

+   记录

+   管道模块化和可重用性

+   扩展你的训练管道以适应更大的模型

# 训练管道的组成部分

LLM 训练管道由几个相互关联的步骤组成，每个步骤在模型的发展中扮演着角色。我们将在这里展示一个基本管道，并在本章的后续部分深入探讨许多这些组件：

+   **数据集创建**：将预处理数据构建成适合训练的格式，通常涉及洗牌和分批处理。

+   **模型架构**：定义了 LLM 的结构，包括层数、注意力机制和其他架构选择。

+   **训练循环**：管道的核心，模型通过正向和反向传递从数据中学习。

+   **优化**：根据计算出的梯度和选择的优化策略处理参数更新。

+   **评估**：定期评估模型在验证数据上的性能，以跟踪进度并防止过拟合。我们将在 *第十四章* 中更详细地讨论这个主题。

+   **检查点**：定期保存模型状态以恢复训练或用于推理。我们将在 *第十章* 中详细讨论这个主题。

+   **记录和监控**：持续跟踪训练指标和资源利用率。

我们将使用 PyTorch 和 Transformers 库实现一个基本的 LLM 训练管道：

```py
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AdamW,
    get_linear_schedule_with_warmup
from datasets import load_dataset
import torch
from torch.nn import functional as F
import wandb
```

PyTorch 是一个流行的深度学习框架，它通过动态计算图允许构建神经网络，而 Transformers 库实现了我们在 *第一章* 中讨论的流行变压器架构。

以下代码块展示了使用预训练的 GPT-2 分词器加载维基百科数据集并对其文本内容进行分词的过程：

```py
# Dataset Creation: Ingestion and Preprocessing
dataset = load_dataset("wikipedia", "20220301.en", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,
        max_length=512, padding="max_length")
tokenized_dataset = dataset.map(preprocess_function,
    batched=True, remove_columns=dataset.column_names)
```

在前面的代码块中，我们正在设置管道的数据摄取和预处理组件。我们使用 Hugging Face Datasets 库加载维基百科数据集，它提供了一个适合训练 LLM 的大型文本语料库。然后，我们初始化一个基于 **GPT-2 模型**的分词器，它将被用于预处理我们的文本数据。

上文定义的`preprocess_function`函数将原始文本示例进行分词，截断到最大 512 个 token 的长度，并将较短的序列填充到这个长度。这确保了所有输入序列的长度相同，这对于高效的批量处理是必要的。我们选择`max_length`值为`512`，这是在上下文长度和内存效率之间的平衡。较长的序列提供更多的上下文，但需要更多的内存和计算。一些最近的 LLM 模型，如**Gemini 1.5 Pro**，其内容长度可以达到多达 200 万个 token（[`cloud.google.com/vertex-ai/generative-ai/docs/long-context`](https://cloud.google.com/vertex-ai/generative-ai/docs/long-context)）。

接下来，我们创建我们的训练数据加载器，它将在训练过程中处理数据集的批处理和打乱：

```py
# Dataset Creation: Loading
train_dataloader = DataLoader(
    tokenized_dataset, shuffle=True, batch_size=8)
```

我们将批大小设置为`8`，这是在内存使用和训练效率之间做出的平衡选择。更大的批大小可以加快训练速度，但需要更多的 GPU 内存。对于具有大量参数的 LLM，通常需要较小的批大小才能将模型和数据放入 GPU 内存中。

然后，我们使用预训练的 GPT-2 模型初始化我们的模型架构。这为我们 LLM 提供了一个强大的起点，利用了预训练权重中已经捕获的知识。使用预训练模型作为起点是迁移学习中的一种常见做法，使我们能够从模型在大量文本语料库上学习到的通用语言理解中受益。以下代码展示了这一过程：

```py
# Model Architecture
model = AutoModelForCausalLM.from_pretrained("gpt2")
# Optimization
optimizer = AdamW(model.parameters(), lr=5e-5)
```

如前述代码所示，为了优化，我们将学习率`lr`设置为`5e-5`，这是微调预训练模型时的一个常见选择。学习率是一个超参数，它决定了在训练过程中对模型权重进行调整的大小，影响着模型学习的速度和效率。

这个学习率在学习和稳定性之间提供了良好的平衡。它足够小，可以允许对预训练权重进行精细的更新，但又足够大，以允许有意义的学习发生。

下面的代码块概述了训练语言模型的基本阶段，包括设置训练过程、初始化日志工具、执行带有正向和反向传递的主训练循环、执行评估以评估模型性能，以及在训练过程中保存模型参数的检查点。

1.  我们首先设置训练循环：

    ```py
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100,
        num_training_steps=num_training_steps)
    ```

1.  然后，我们初始化 Weights & Biases (`wandb`)库以进行实验跟踪和训练指标的日志记录：

    ```py
    wandb.init(project="llm_training", name="gpt2_finetune")
    device = torch.device("cuda" if torch.cuda.is_available() 
        else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            wandb.log({"loss": loss.item()})
    ```

1.  接下来，我们实现一个评估阶段来评估模型在训练数据上的性能：

    ```py
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in train_dataloader:  # Using training data for simplicity
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                eval_loss += outputs.loss.item()
        eval_loss /= len(train_dataloader)
        wandb.log({"eval_loss": eval_loss})
    ```

1.  最后，在每个 epoch 结束时，我们保存模型状态字典的检查点：

    ```py
        torch.save(model.state_dict(), 
            f"model_checkpoint_epoch_{epoch}.pt")
    wandb.finish()
    ```

这些代码片段实现了我们流水线中的训练循环、评估、检查点和日志记录组件：

我们将训练时代数设置为`3`，这意味着模型将在训练期间遍历整个数据集三次。这个超参数可以根据你的具体需求进行调整——如果模型欠拟合，增加它可能会导致更好的模型性能，而减少它可以帮助防止过拟合并减少训练时间。在训练期间监控验证损失，以确定特定数据集和模型架构的最佳时代数。

学习率调度器实现了一个带有预热阶段的线性衰减，这有助于在训练的早期阶段稳定训练，然后逐渐降低学习率以更精确地微调模型。学习率控制模型在训练期间调整其内部参数的程度——较高的速率意味着更大的调整但可能存在过度调整的风险，而较低的速率意味着更精确但学习速度较慢。

我们使用`wandb`进行记录，这允许我们实时跟踪我们的训练进度并比较不同的运行([`wandb.ai/site`](https://wandb.ai/site))。这对于监控训练过程和在超参数调整和模型架构更改方面做出明智的决定至关重要。

训练循环遍历指定数量的时代数据。在每次迭代中，我们执行以下操作：

1.  将批次移动到适当的设备（如果有 GPU 则使用 GPU）

1.  通过模型执行正向传递

1.  计算损失

1.  执行反向传播

1.  更新模型参数

1.  更新学习率调度器

1.  记录训练损失

在每个时代之后，我们执行对训练数据的简单评估（在实际场景中，你会使用一个单独的验证集），记录评估损失，并保存模型的检查点。检查点对于长时间运行的训练过程是必需的，允许我们在需要时从保存的状态恢复训练。

正如我们所见，训练管道涉及几个基本步骤。然而，在模型架构和训练循环可以有效地运行之前，我们必须解决数据输入和预处理问题，我们将在下一节中讨论。

# 数据输入和预处理

高效的数据处理对于 LLM 训练至关重要，正如我们在本书的*第一部分*中讨论的那样。在这里，让我们探讨数据输入和预处理的高级技术：

1.  导入所需的 Python 包：

    ```py
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import numpy as np
    ```

1.  加载和组合多个数据集：

    ```py
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
    books_dataset = load_dataset("bookcorpus", split="train")
    # Combine datasets
    combined_dataset = concatenate_
        datasets([wiki_dataset, books_dataset])
    ```

1.  初始化分词器并执行`preprocess`：

    ```py
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    def preprocess_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"], truncation=True, max_length=1024)
        # Create input_ids and attention_mask
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        # Create labels for causal language modeling
        labels = [
            ids[1:] + [tokenizer.eos_token_id] for ids in input_ids]
        return {"input_ids": input_ids, 
            "attention_mask": attention_mask, "labels": labels}
    # Apply preprocessing
    tokenized_dataset = combined_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=combined_dataset.column_names,
        num_proc=4  # Adjust based on your CPU cores
    )
    ```

1.  创建数据加载器：

    ```py
    train_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        batch_size=16,
        collate_fn=lambda x: {k: np.stack([xi[k] for xi in x]) 
            for k in x[0]}
    )
    ```

在这个增强的预处理管道中，我们正在加载多个数据集以增加我们训练数据的多样性。这对于 LLM 是必要的，因为多样化的数据集有助于模型学习更广泛的语言模式和知识。

我们使用较长的`max_length`值为`1024`个标记，为模型提供更多上下文。这种增加的上下文长度允许模型捕获文本中的长距离依赖关系，这对许多语言理解任务可能有益。然而，这也增加了内存使用和计算需求，因此需要权衡考虑。

`preprocess_function`现在通过移位输入序列为因果语言建模创建标签。这是训练语言模型的一种常见方法，其中模型的任务是预测给定前一个标记的下一个标记。在预处理过程中，处理边缘情况，如表情符号、URL 和非标准字符，可以提高模型性能。表情符号可以传达细微的情感和上下文，需要适当的编码或标记化以保留其含义而不引入噪声。URL 通常包含有价值的信息，但结构可能差异很大，因此可能用占位符标记替换以保持一致性，同时防止模型过度拟合到特定链接。非标准字符，包括来自不同语言的符号或特殊标点符号，需要仔细归一化或删除以减少复杂性和避免训练时的混淆。通过采用归一化、标记替换和选择性过滤等策略解决这些边缘情况，预处理管道可以更好地准备多样化和复杂的数据，从而提高结果语言模型的鲁棒性和准确性。

我们使用多进程（`num_proc=4`）来加速预处理。进程数应根据您的 CPU 核心数和可用内存进行调整。多进程可以显著减少预处理时间，特别是对于大型数据集。

批处理大小增加到`16`，这更适合较大的 GPU 内存。DataLoader 中的自定义`collate_fn`确保了我们的预处理数据的正确批处理。此函数将批处理中每个键的数组堆叠，创建出可以被 PyTorch 高效处理的张量结构。

在数据适当准备后，我们现在将注意力转向 LLM 架构设计考虑因素，这些因素决定了模型有效学习并理解数据输入的能力。

# LLM 架构设计考虑因素

在设计 LLM 的架构时，有几个因素需要考虑。

影响 LLM 架构的关键因素如下：

+   **词汇量大小**：决定了输入和输出嵌入层的大小

+   **最大序列长度（上下文大小）**：定义了模型可以考虑的先前文本的数量

+   **嵌入维度**：指定每个标记的向量表示的大小，影响模型捕获信息的能力

+   **transformer 层数量**：代表网络的深度，影响模型可以学习的模式复杂性

+   **注意力头数量**：允许模型同时关注输入的不同部分

+   **模型大小（参数数量）**：模型的整体容量，受嵌入维度、层数和注意力头数的影响

+   **数据集大小**：训练数据的数量和多样性

+   **训练步数数量**：优化过程的持续时间

+   **计算资源**：影响模型大小、训练速度和整体可行性的硬件限制。

+   **过拟合风险**：随着模型规模增大和数据集减小而增加

+   **数据质量**：训练数据的清洁度和相关性

+   **模型架构效率**：可以在不大幅增加模型大小的情况下提高性能的设计选择

+   **训练算法**：优化技术和策略

+   **数据整理实践**：选择和准备训练数据的方法

+   **推理时间计算资源**：推理过程中可用的计算资源

在以下代码块中，我们提供了使用 GPT-2 风格的语言模型配置一些这些因素的示例，指定关键架构参数。

```py
from transformers import GPT2Config, GPT2LMHeadModel
# Define custom model configuration
config = GPT2Config(
    vocab_size=50257,  # GPT-2 vocabulary size
    n_positions=1024,  # Maximum sequence length
    n_ctx=1024,        # Context size
    n_embd=768,        # Embedding dimension
    n_layer=12,        # Number of transformer layers
    n_head=12          # Number of attention heads
)
# Initialize the model with custom configuration
model = GPT2LMHeadModel(config)
print(f"Model parameters: {model.num_parameters():,}")
```

此配置创建了一个具有 `12` 层和 `12` 个注意力头的 GPT-2 风格模型。让我们分解关键参数：

+   `vocab_size`：设置为 `50257`，这是原始 GPT-2 模型的词汇量。这决定了嵌入层和输出层的大小。

+   `n_positions` 和 `n_ctx`：两者都设置为 `1024`，与我们的预处理步骤相匹配。这定义了模型可以处理的最大序列长度。

+   `n_embd`：嵌入维度，设置为 `768`。这决定了模型中隐藏状态的大小。

+   `n_layer`：transformer 层数的数量，设置为 `12`。更多的层可以捕捉更复杂的模式，但会增加计算需求。

+   `n_head`：注意力头的数量，设置为 `12`。多个注意力头允许模型同时关注输入的不同方面。

`768` 的嵌入维度和 `12` 层提供了在模型容量和计算效率之间的平衡折衷。这种配置产生了一个大约有 1.24 亿个参数的模型，这相当大，但仍然可以在常见的 GPU 硬件上训练。

对于更大的模型，你可能需要增加 `n_layer`、`n_embd` 和 `n_head`。然而，这也会增加计算需求以及过拟合的风险，尤其是在较小的数据集上。在扩展规模时，考虑使用梯度累积、混合精度训练和分布式训练等技术来管理增加的计算负载。

在更广泛的范围内，可以考虑**扩展定律**。LLM 的扩展定律描述了随着三个关键因素的提高，性能如何可预测地提升：模型大小（参数数量）、数据集大小（训练数据量）和训练步数（优化迭代次数）。具体来说，更大的模型倾向于捕捉更复杂的模式并表现出更好的泛化能力，更大的数据集为学习提供了更多样化的信息，更多的训练步数允许模型细化其理解并减少错误。为了获得最佳性能，这些因素应该成比例扩展——例如，增加模型大小应该与数据集大小和训练步数的相应增加相匹配。这种平衡扩展确保每个组件都支持其他组件，防止了诸如在庞大的数据集上过度拟合较小模型或用不足的数据训练大型模型等瓶颈问题。

然而，最近的发展和实际挑战表明，仅仅扩展这些因素并不总是足以实现持续的性能提升。例如，边际效益递减的问题，即每个额外的参数或数据点对整体性能的贡献越来越少，变得更加明显。此外，训练越来越大的模型所需的巨大计算和能源资源引发了可持续性和可访问性的担忧。数据质量也成为了一个关键因素，因为更大的数据集可能会引入更多的噪声和偏差，从而降低模型性能。关于这方面的更多细节，请参阅[`www.pcgamer.com/software/ai/open-ai-co-founder-reckons-ai-training-has-hit-a-wall-forcing-ai-labs-to-train-their-models-smarter-not-just-bigger/`](https://www.pcgamer.com/software/ai/open-ai-co-founder-reckons-ai-training-has-hit-a-wall-forcing-ai-labs-to-train-their-models-smarter-not-just-bigger/)上的文章。

为了应对这些挑战，研究人员正在探索更有效的模型架构、改进的训练算法、更好的数据整理实践以及测试时间计算。有关测试时间计算的更多细节，请参阅我的 Medium 文章：[`kenhuangus.medium.com/test-time-compute-3633a4c55716`](https://kenhuangus.medium.com/test-time-compute-3633a4c55716)。

2025 年初，DeepSeek（一家中国的人工智能初创公司）通过引入一系列旨在显著提高效率和降低成本的同时增强模型推理能力的技术的套件，宣布了一些模型训练创新（[`arxiv.org/abs/2501.12948`](https://arxiv.org/abs/2501.12948)）。与严重依赖大量计算资源和人工监督微调的传统方法不同，DeepSeek 利用针对推理任务的大规模强化学习，使用自动奖励系统而不是人类反馈。关键创新包括多标记预测，这使得模型能够一次学习多个未来的标记，从而提高样本效率并加快训练速度。DeepSeek 还采用专家混合架构，只为每个任务激活相关的子网络，从而减少计算负载。通过优化算法和硬件，DeepSeek 已经能够以竞争对手所需成本和时间的一小部分训练出高度能干的模型，为开放、高效和强大的 AI 开发设定了新的标准。

在探讨了 LLMs 的架构设计考虑因素和模型训练创新，以及一个演示如何配置模型训练参数的代码示例之后，我们现在准备检查这些架构选择在训练过程中是如何实际学习的。在接下来的这一节中，我们将讨论损失函数和优化策略，它们是推动模型根据训练数据和我们所定义的架构调整其内部参数的引擎。

# 损失函数和优化策略

LLMs 通常使用 **交叉熵损失** 进行训练。这种方法衡量模型预测的单词概率分布与训练数据中实际分布之间的差异。通过最小化这种损失，LLMs 学习生成更准确和上下文相关的文本。由于交叉熵损失能够处理文本数据的高维性和离散性，因此它特别适合语言任务。

让我们结合一些高级优化技术来实现这个功能：

1.  首先，我们导入所需的 PyTorch 库以及来自 Transformers 库的特定模块以进行优化：

    ```py
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    ```

1.  接下来，我们配置 AdamW 优化器，指定学习率和权重衰减：

    ```py
    optimizer = AdamW(model.parameters(), lr=5e-5, 
        weight_decay=0.01)
    ```

1.  然后，我们定义一个具有预热期的线性学习率调度器：

    ```py
    num_epochs = 3
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    ```

1.  接着，我们设置训练设备和启动主训练循环：

    ```py
    device = torch.device("cuda" if torch.cuda.is_available() 
        else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: torch.tensor(v).to(device) 
                for k, v in batch.items()
            }
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
    ```

1.  最后，我们实现梯度裁剪以防止训练过程中的梯度爆炸：

    ```py
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    ```

在这个优化设置中，我们使用学习率为`5e-5`和权重衰减为`0.01`的`AdamW`优化器。该算法根据梯度的第一和第二矩来调整每个参数的学习率，使其能够有效地处理稀疏梯度。这使得`AdamW`对于训练大型神经网络特别有用。

权重衰减`0.01`向损失函数中添加了一个小的正则化项，这可以通过惩罚大的权重值来帮助防止过拟合。

我们实现了一个`预热`阶段。预热阶段通过逐渐增加学习率从一个非常小的值来帮助在早期阶段稳定训练。在预热阶段之后，学习率线性下降。这种时间表可以帮助模型收敛到一个更好的最优解。

在训练循环中，我们实现了`max_norm`值为`1.0`。梯度裁剪通过缩小超过某个阈值的梯度值来防止梯度爆炸。这对于 LLM 尤其重要，因为 LLM 由于其深度和捕获的长距离依赖性，可能会出现不稳定的梯度。

在本节中，我们学习了关于 AdamW 优化、带有预热的学习率调度以及梯度裁剪以稳定 LLM 训练的内容。接下来，我们将讨论记录训练过程，这对于监控进度和使用如**TensorBoard**等工具以获得改进的见解至关重要。

# 日志记录

有效的日志记录对于跟踪 LLM 训练的进度非常有用。

以下代码块演示了如何使用 PyTorch 在训练 LLM 时集成 TensorBoard 以进行有效的日志记录。让我们分解每个部分。

1.  我们首先初始化 TensorBoard 的`SummaryWriter`以记录训练进度：

    ```py
    from torch.utils.tensorboard import SummaryWriter
    import time
    # Initialize TensorBoard writer
    writer = SummaryWriter()
    ```

1.  然后，我们将模型设置为训练模式，初始化跟踪损失的变量，定义日志间隔，并记录开始时间以监控训练性能：

    ```py
    model.train()
    total_loss = 0
    log_interval = 100
    start_time = time.time()
    ```

1.  然后，我们进入训练循环。我们通过将数据移动到适当的设备，执行正向和反向传递，应用梯度裁剪，并使用优化器和调度器更新模型的参数来处理每个批次：

    ```py
    for i, batch in enumerate(train_dataloader):
        batch = {k: torch.tensor(v).to(device) 
            for k, v in batch.items()}
        outputs = model(batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
            max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    ```

1.  我们在指定的间隔将训练指标记录到 TensorBoard 中，计算平均损失，测量经过的时间，将进度打印到控制台，并为下一个间隔重置跟踪变量：

    ```py
        if (i + 1) % log_interval == 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            writer.add_scalar(
                'training_loss', cur_loss, global_step=i
            )
            writer.add_scalar(
                'learning_rate', scheduler.get_last_lr()[0], 
                global_step=i
            )
            print(
                f'| epoch {epoch:3d} '
                f'| {i:5d}/{len(train_dataloader):5d} batches | '
                f'lr {scheduler.get_last_lr()[0]:02.2f} | '
                f'ms/batch {elapsed * 1000 / log_interval:5.2f} | '
                f'loss {cur_loss:5.2f}'
            )
            total_loss = 0
            start_time = time.time()
    writer.close()
    ```

    这个增强的训练循环使用 TensorBoard 来记录训练损失和学习率。TensorBoard 是一个强大的工具，用于可视化训练进度和比较不同的运行。我们记录以下指标：

    +   `log_interval`批次。此指标呈下降趋势表明模型正在学习。

    +   **学习率**：我们记录当前的学习率，以可视化由于我们的学习率调度器而随时间变化的情况。

我们将`log_interval`设置为`100`，这意味着我们每 100 个批次记录和打印进度信息。这个间隔在获取频繁更新和不过度减慢训练速度之间取得了平衡。您可能需要根据您的数据集大小和训练速度进行调整。

输出或日志信息包括以下内容：

+   当前周期和批次号

+   当前学习率

+   每批次的耗时（以毫秒为单位）

+   当前损失

这种详细的日志记录允许您密切监控训练过程，帮助您识别不稳定损失、学习率问题或意外缓慢的训练等问题。

# 管道模块化和可重用性

**模块化**和**可重用性**是构建高效管道的基本原则，因为它们使代码更易于维护、适应和可靠。通过将管道分解为独立的、可重用的模块（如数据预处理、模型训练和评估组件），开发者可以轻松修改单个部分而不影响其他部分，单独测试每个组件，并在不同项目之间重用经过验证的代码。

这种方法不仅节省了开发时间，还确保了操作的连续性，减少了出错的机会，并使团队在维护组件之间清晰接口的同时，通过在单独的模块上工作而更容易协作。在训练管道的情况下，将过程封装在可重用类中允许灵活配置，与不同数据集的无缝集成，以及在不同项目之间轻松共享标准化实现。

为了使我们的管道更加模块化和可重用，让我们将我们的训练过程封装在一个类中：

1.  我们从类定义开始：

    ```py
    class LLMTrainer:
        def __init__(self, model, train_dataloader, optimizer,
        scheduler, device
        ):
            self.model = model
            self.train_dataloader = train_dataloader
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.device = device
            self.writer = SummaryWriter()
    ```

1.  然后，我们定义训练周期函数。该函数将模型设置为训练模式，并遍历训练数据，通过计算损失、执行梯度裁剪的逆向传播以及使用优化器和调度器更新模型参数来处理每个批次：

    ```py
        def train_epoch(self):
            self.model.train()
            total_loss = 0
            log_interval = 100
            start_time = time.time()
            for i, batch in enumerate(self.train_dataloader):
                batch = {k: torch.tensor(v).to(self.device)
                    for k, v in batch.items()
                }
                outputs = self.model(batch)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
    ```

1.  接下来，我们通过检查当前批次索引是否是`log_interval`的倍数来定期将训练进度记录到 TensorBoard 和控制台；如果是，我们计算平均损失和自上次日志以来的经过时间，使用`SummaryWriter`将训练损失和学习率记录到 TensorBoard，打印包括批次号、学习率、每批次的毫秒数和当前损失等信息的格式化进度更新到控制台，然后重置下一次日志间隔的累积`total_loss`和`start_time`：

    ```py
                if (i + 1) % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    self.writer.add_scalar(
                        'training_loss', cur_loss, global_step=i
                    )
                    self.writer.add_scalar(
                        'learning_rate', 
                        self.scheduler.get_last_lr()[0], 
                        global_step=i
                    )
                        print(
                            f'| {i:5d}/'
                            f'{len(self.train_dataloader):5d} '
                            f'batches | '
                            f'lr '
                            f'{self.scheduler.get_last_lr()'
                            f'[0]:02.2f} | '
                            f'ms/batch '
                            f'{elapsed * 1000 / log_interval:5.2f} '
                            f'| '
                            f'loss '
                            f'{cur_loss:5.2f}'
                        )
                    total_loss = 0
                    start_time = time.time()
    ```

1.  接下来，`train` 函数通过遍历指定的周期数来协调训练过程，在每个周期的开始打印一条消息，调用 `train_epoch` 方法执行该周期的训练，并在所有周期完成后关闭写入器。它是训练的主要入口点，提供了一个结构，其中可以按需集成如验证和检查点等附加功能：

    ```py
        def train(self, num_epochs):
            for epoch in range(num_epochs):
                print(f'Starting epoch {epoch+1}')
                self.train_epoch()
                # Here you could add validation, checkpointing, etc.
            self.writer.close()
    ```

1.  最后，我们使用指定的模型、训练数据加载器、优化器、调度器和设备实例化 `LLMTrainer` 类。然后，通过调用 `train` 方法来执行三个完整的训练周期，从而启动并管理模型的学习周期：

    ```py
    trainer = LLMTrainer(model, train_dataloader,
        optimizer, scheduler, device)
    trainer.train(num_epochs=3)
    ```

这种模块化设计提供了几个优点：

+   `LLMTrainer` 类，使其更容易管理和理解。

+   **可重用性**：你可以通过创建具有不同参数的新实例，轻松地将此训练器用于不同的模型或数据集。

+   **可扩展性**：类结构使得添加新功能变得容易。例如，你可以添加用于验证、检查点或早期停止的方法。

+   **关注点分离**：训练逻辑与模型定义和数据准备分离，遵循良好的软件工程原则。

以下日志演示了在 3 个周期内进行的训练过程，每隔 100 批次进行一次定期记录。每个日志条目包括当前批次号、总批次数、学习率、每批次的毫秒数和平均损失：

```py
Starting epoch 1
|   100/1000 batches | lr 0.01 | ms/batch 45.67 | loss 2.35
|   200/1000 batches | lr 0.01 | ms/batch 44.89 | loss 2.10
|   300/1000 batches | lr 0.01 | ms/batch 46.12 | loss 1.95
|   400/1000 batches | lr 0.01 | ms/batch 45.50 | loss 1.80
|   500/1000 batches | lr 0.01 | ms/batch 44.75 | loss 1.65
|   600/1000 batches | lr 0.009 | ms/batch 45.30 | loss 1.50
|   700/1000 batches | lr 0.009 | ms/batch 44.95 | loss 1.40
|   800/1000 batches | lr 0.009 | ms/batch 45.10 | loss 1.30
|   900/1000 batches | lr 0.009 | ms/batch 45.00 | loss 1.25
|  1000/1000 batches | lr 0.009 | ms/batch 44.80 | loss 1.20
Starting epoch 2
|   100/1000 batches | lr 0.009 | ms/batch 44.60 | loss 1.18
|   200/1000 batches | lr 0.009 | ms/batch 44.70 | loss 1.15
|   300/1000 batches | lr 0.009 | ms/batch 44.80 | loss 1.12
|   400/1000 batches | lr 0.008 | ms/batch 44.50 | loss 1.10
|   500/1000 batches | lr 0.008 | ms/batch 44.60 | loss 1.08
|   600/1000 batches | lr 0.008 | ms/batch 44.55 | loss 1.05
|   700/1000 batches | lr 0.008 | ms/batch 44.65 | loss 1.03
|   800/1000 batches | lr 0.007 | ms/batch 44.50 | loss 1.00
|   900/1000 batches | lr 0.007 | ms/batch 44.60 | loss 0.98
|  1000/1000 batches | lr 0.007 | ms/batch 44.55 | loss 0.95
Starting epoch 3
|   100/1000 batches | lr 0.007 | ms/batch 44.50 | loss 0.93
|   200/1000 batches | lr 0.007 | ms/batch 44.60 | loss 0.90
|   300/1000 batches | lr 0.006 | ms/batch 44.55 | loss 0.88
|   400/1000 batches | lr 0.006 | ms/batch 44.50 | loss 0.85
|   500/1000 batches | lr 0.006 | ms/batch 44.60 | loss 0.83
|   600/1000 batches | lr 0.006 | ms/batch 44.55 | loss 0.80
|   700/1000 batches | lr 0.005 | ms/batch 44.50 | loss 0.78
|   800/1000 batches | lr 0.005 | ms/batch 44.60 | loss 0.75
|   900/1000 batches | lr 0.005 | ms/batch 44.55 | loss 0.73
|  1000/1000 batches | lr 0.005 | ms/batch 44.50 | loss 0.70
Training completed. Writer closed.
```

这里是对上述模拟日志的解释：

+   `开始第 1 个周期`，表示新的训练周期的开始

+   `100/1000 批次`

+   `lr 0.01`

+   `ms/批次 45.67`

+   `损失 2.35`

+   **学习率调度**：注意学习率在周期内是如何降低的，这反映了调度器为促进更好的收敛所做的调整*   `训练完成。写入器已关闭。`) 表示训练过程的结束和日志写入器的关闭

日志提供了对训练动态的清晰概述，允许开发人员和研究人员监控模型的学习进度，如有必要，调整超参数，并确保训练按预期进行。

# 扩展您的训练流程以适应更大的模型

为了训练更大的模型，我们需要采用梯度累积和混合精度训练等技术。

为了训练可能不适合单个 GPU 的大规模语言模型，以下代码引入了一个特殊的 `LargeScaleLLMTrainer`。它使用两个主要技巧来处理这个问题：

首先，梯度累积允许我们模拟访问更大的 GPU。我们不是在每处理一小批数据后更新模型的参数，而是在处理几个小批量的过程中累积它们的梯度。只有在预定义的批次数之后，我们才对模型的参数进行实际更新。这种技术使模型能够像看到了一个更大的数据批次一样学习，而不需要极端大 GPU 的内存容量。

其次，它采用混合精度训练，这是一种计算机使用较小的、低精度数字（需要更少的内存且计算速度更快）进行许多计算的技术，同时为精度至关重要的场合保留高精度数字。这种方法加速了训练并减少了整体内存使用。为了减轻使用低精度值可能出现的潜在问题，GradScaler 在反向传播过程中保持数值稳定性。

以下代码定义了这种特殊训练器的工作方式，包括如何处理数据、计算损失以及使用这些技巧更新模型的参数、学习率和梯度缩放器。它还包括确保梯度（模型应该如何改变）不会太大以及记录进度以便我们可以看到训练进展的重要步骤。最后，它展示了如何使用这个特殊训练器的简单示例。现在，让我们将其分解成几个部分：

1.  让我们先导入相关的 Python 包并定义类：

    ```py
    import torch.cuda.amp as amp
    class LargeScaleLLMTrainer(LLMTrainer):
        def __init__(self, model, train_dataloader,
            optimizer, scheduler, device, accumulation_steps=4
        ):
            super().__init__(model, train_dataloader,
                optimizer, scheduler, device)
            self.accumulation_steps = accumulation_steps
            self.scaler = amp.GradScaler()
    ```

1.  然后，我们可以定义训练的迭代次数：

    ```py
        def train_epoch(self):
            self.model.train()
            total_loss = 0
            log_interval = 100
            start_time = time.time()
            for i, batch in enumerate(self.train_dataloader):
                batch = {
                    k: torch.tensor(v).to(self.device)
                    for k, v in batch.items()
                }
                with amp.autocast():
                    outputs = self.model(batch)
                    loss = outputs.loss / self.accumulation_steps
                self.scaler.scale(loss).backward()
    ```

1.  然后，我们实现以下代码块，它仅在处理了定义数量的批次（`accumulation_steps`）之后更新模型的参数、学习率和梯度缩放器，有效地模拟更大的批次数同时管理内存限制：

    ```py
                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                total_loss += loss.item() * self.accumulation_steps
    ```

1.  然后，我们定期计算并记录平均训练损失和学习率到 TensorBoard，同时按照`log_interval`定义的时间间隔在控制台上打印当前训练进度的摘要：

    ```py
                if (i + 1) % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() start_time
                    self.writer.add_scalar('training_loss',
                        cur_loss, global_step=i)
                    self.writer.add_scalar('learning_rate',
                        self.scheduler.get_last_lr()[0],
                        global_step=i)
                    print(
                        f'| {i:5d}/{len(self.train_dataloader):5d}
                            batches | '
                        f'lr {self.scheduler.get_last_lr()[0]:02.2f}
                            | '
                        f'ms/batch {elapsed * 1000 /
                            log_interval:5.2f} | '
                        f'loss {cur_loss:5.2f}'
                     )
                    total_loss = 0
                    start_time = time.time()
    ```

1.  我们展示了大规模语言模型训练过程的初始化和执行：

    ```py
    large_trainer = LargeScaleLLMTrainer(
        model, train_dataloader, optimizer, scheduler, device)
    large_trainer.train(num_epochs=3)
    ```

    此增强型训练器使用两种关键技术来扩展到更大的模型：

    +   (`accumulation_steps`)。这允许我们有效地增加批次数而不增加内存使用，这对于在有限的 GPU 内存上训练大型模型是有效的。我们将损失除以`accumulation_steps`以保持相同的有效学习率。

    +   在可能的情况下使用`float16`，同时保留`float32`的主权重。这可以显著加快训练速度并减少内存使用，尤其是在具有张量核心的现代 GPU 上。

`GradScaler`用于防止`float16`计算中的下溢。它将损失缩放以防止小的梯度值，然后在优化器步骤之前进行缩放。

我们仍然应用梯度裁剪，但现在是在将梯度缩放回原始尺度之后进行的，以确保我们裁剪的是真实的梯度值。

对于更大的模型，你可能需要考虑诸如**模型并行**（将模型分割到多个 GPU 上）、**流水线并行**（将模型分割成阶段）或使用如**DeepSpeed**或**Megatron-LM**等专用库等技术。这些高级技术允许在多个 GPU 甚至多台机器上训练具有数十亿参数的模型。当 GPU 内存不足以处理大量数据和模型参数时，内存卸载可以是一个好的替代方案。内存卸载涉及将模型数据或计算的部分转移到替代内存存储，例如**非易失性内存表达式**（**NVMe**）SSD。通过利用 NVMe 内存，它提供与传统存储相比的高速数据访问，系统可以有效地管理和存储超出 GPU 内存容量的中间激活、梯度和模型状态。这种方法允许在不要求立即扩展 GPU 内存的情况下训练更大的模型或使用更大的批量大小。然而，由于 GPU 和 NVMe 存储之间的数据传输，它引入了额外的延迟，这可能会影响训练速度。优化数据访问模式并利用有效的卸载策略可以在采用内存卸载技术时最小化性能开销并保持有效的训练工作流程。

# 摘要

在本章中，你了解到了训练 LLM 的实际管道设计模式。你学习了如何创建高效的数据预处理工作流程，实现模型架构，并应用高级优化策略。你现在理解了如何设置有效的日志系统来跟踪你的模型进度。你还探索了构建模块化和可重用管道的技术，并发现了扩展你的训练过程以适应更大模型的方法。有了这些技能，你将能够高效且有效地训练最先进的语言模型。

在下一章中，我们将探讨超参数调整模式。
