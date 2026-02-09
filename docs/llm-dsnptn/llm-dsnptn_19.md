

# 第十九章：从人类反馈中进行强化学习

在本章中，我们将深入探讨**从人类反馈中进行强化学习**（**RLHF**），这是一种将 LLM 与人类偏好对齐的强大技术。RLHF 结合了强化学习和人类反馈来微调语言模型。它的目标是使模型的输出与人类偏好对齐，提高生成文本的质量和安全。

RLHF 与标准监督微调不同，它优化的是人类偏好而不是预定义的正确答案。虽然监督学习最小化对标记示例的损失，但 RLHF 从模型输出之间的人类比较中创建一个奖励模型，然后使用这个奖励函数（通常使用**近端策略优化**（**PPO**））来更新模型的政策。这个过程通常采用一个发散惩罚来防止过度偏离初始模型分布。

RLHF 的关键好处如下：

+   模型与人类价值观和偏好的改进对齐

+   对模型输出的增强控制

+   减少有害或偏见的内容

+   优化特定任务性能的能力

到本章结束时，你将能够实现 RLHF 技术来提高你 LLM 的对齐和输出质量。

在本章中，我们将涵盖以下主题：

+   RLHF 系统的组成部分

+   扩展 RLHF

+   RLHF 在语言建模中的局限性

+   RLHF 的应用

# RLHF 系统的组成部分

LLMs 的典型 RLHF 系统由三个主要组件组成：

+   **基础语言模型**：待微调的预训练 LLM

+   **奖励模型**：一个基于人类偏好进行训练以提供反馈的模型

+   **策略优化**：使用奖励信号更新基础模型的过程

基础语言模型是起点。这是一个已经在大规模语料库上使用如下一个标记预测等自监督目标进行广泛预训练的通用大型语言模型。在这个阶段，模型能够生成连贯的语言并展示广泛的语言能力。然而，它缺乏与人类偏好、特定任务目标或实际部署中期望的上下文相关行为的对齐。这个预训练模型是后续微调的基础。其架构、训练方式和扩展已经在文献中得到了很好的记录，并且由于 RLHF 在它没有改变其基本结构的基础上构建，因此在这里进一步详细说明是不必要的。

相反，让我们关注奖励模型和政策优化组件，它们共同工作，根据人类对齐的标准来引导和重塑基础模型的输出分布。这两部分引入了反馈驱动的适应和强化调整的核心机制，将在以下章节中进行探讨。

## 奖励模型

让我们为奖励模型实现一个基本结构：

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class RLHFSystem:
    def __init__(self, base_model_name, reward_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name)
        self.reward_model = \
            AutoModelForSequenceClassification.from_pretrained(
            reward_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name)
    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.base_model.generate(inputs, max_length=100)
        return self.tokenizer.decode(outputs[0],
            skip_special_tokens=True)
    def get_reward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.reward_model(inputs)
        return outputs.logits.item()
```

此类设置 RLHF 系统的基本结构，包括基本语言模型和奖励模型。`generate_text`方法从给定的提示生成文本，而`get_reward`方法使用奖励模型估计给定文本的奖励。

奖励模型是 RLHF 过程的核心，因为它将人类偏好转化为可学习的信号。在由模型输出之间的人类比较组成的数据集上训练——评估者选择两个响应中较好的一项——它学会预测人类可能会如何评估任何给定的响应。在强化学习阶段，此奖励模型作为人类判断的自动化代理，使基本模型能够立即获得数千个生成输出的反馈。策略模型（正在优化的语言模型）随后通过 PPO 等技术学习最大化这些预测奖励分数，逐步将其行为调整为生成与人类偏好更好地对齐的响应，同时通过发散约束保持连贯性和能力。这创建了一个强大的反馈循环，使持续与人类价值观保持一致成为可能，这在静态监督数据集的情况下是不可能的。

这里是一个奖励模型训练的简单实现：

```py
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments
class FeedbackDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}
    def train_reward_model(model, tokenizer, texts, labels):
    dataset = FeedbackDataset(texts, labels)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length",
            truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    return model
```

此代码设置了一个包含人类反馈的数据集，并使用 Hugging Face Trainer API 训练奖励模型。奖励模型学会根据提供的标签预测人类偏好。

## 策略优化

策略优化是使用奖励模型中的奖励来更新基本语言模型的过程。一种常见的方法是 PPO，它在实现简便性、样本效率和可靠性能之间取得平衡。PPO 中的“近端”一词指的是其关键创新：限制策略在每个训练步骤中可以改变的程度，以防止有害的大更新。它是通过使用“剪裁”目标函数来实现的，该函数会阻止将策略移动得太远，从而远离其先前版本。PPO 因其比其他策略梯度方法更稳定而特别受到 AI 对齐和 RLHF 的欢迎——它有助于避免模型更新变得过于激进并破坏先前学习到的良好行为。当用于语言模型时，PPO 有助于逐步调整模型的输出，使其更好地匹配人类偏好，同时保持连贯流畅的文本生成。

这里是 LLMs 的 PPO 简化实现：

```py
def ppo_step(
    base_model, reward_model, optimizer, prompt, num_iterations=5
):
    for _ in range(num_iterations):
        # Generate text
        outputs = base_model.generate(prompt, max_length=100,
            return_dict_in_generate=True, output_scores=True
        )
        generated_text = tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        # Get reward
        reward = reward_model(generated_text)
        # Compute policy loss
        log_probs = outputs.scores[0].log_softmax(dim=-1)
        policy_loss = -log_probs * reward
        # Update model
        optimizer.zero_grad()
        policy_loss.mean().backward()
        optimizer.step()
    return base_model
```

此函数执行 PPO 的单步操作，生成文本，计算奖励，并更新基本模型的参数以最大化预期奖励。请注意，此 PPO 代码仅用于说明；实际实现可能需要更多的奖励和安全性检查。

**直接偏好优化**（**DPO**）是 RLHF（强化学习与人类反馈）中的另一种方法，它通过直接优化首选结果来关注使模型与人类偏好对齐。与传统的 RL 方法不同，后者通常依赖于奖励模型来指导学习，DPO 通过使用首选和不受欢迎的输出对来调整模型的行为，从而简化了过程。这种方法提高了训练模型的效率和效果，使它们生成的输出更接近人类的期望。

当计算效率和实现简单性是重点时，DPO 可能比 PPO 更受欢迎。这是因为 DPO 消除了单独训练奖励模型和复杂的强化学习优化循环的需求。它通过直接从偏好数据更新策略参数提供了一种更简化的方法，这在资源有限或 PPO 训练表现出不稳定或奖励黑客行为的情况下尤其有价值。DPO 还可以在没有奖励建模的中间步骤的情况下更好地利用有限的人类偏好数据集。此外，它提供了一个更清晰的实验设置，用于研究偏好如何直接影响模型行为，而没有引入由单独的奖励模型和强化学习优化引入的混杂因素。

下面是一个简短的代码示例，展示了如何使用 Python 实现 DPO：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
# Load a pre-trained language model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Define the dataset containing human preference pairs
# Each entry in the dataset is a tuple (prompt, preferred_completion, dispreferred_completion)
dataset = [
    ("Prompt 1", "Preferred Completion 1", "Dispreferred Completion 1"),
    ("Prompt 2", "Preferred Completion 2", "Dispreferred Completion 2"),
    # Add more data as needed
]
# Initialize the DPO Trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    beta=0.1  # Hyperparameter controlling the strength of preference optimization
)
# Train the model using DPO
trainer.train()
# Save the fine-tuned model
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")
```

这段代码片段展示了如何使用 DPO（分布式训练优化）来设置和训练一个语言模型，使其能够通过直接优化首选完成情况来更好地与人类反馈对齐。

在讨论了 PPO 和 DPO 之后，接下来我们将探讨关于大规模模型的 RLHF 的扩展策略。

# 扩展 RLHF

将 RLHF 扩展到大型模型面临着计算需求带来的挑战。以下是一些可以实施的战略：

+   **分布式训练**：这涉及到通过采用数据并行性、模型并行性或流水线并行性，将训练工作负载分配到多个设备上——通常是 GPU 或 TPU。在数据并行性中，相同的模型在设备上被复制，每个副本处理不同的数据小批量。在每个步骤之后，梯度被平均并同步。另一方面，模型并行性将模型本身分割到多个设备上，使得可以训练那些无法适应单个设备的架构。最后，流水线并行性进一步将模型分割成设备上的顺序阶段，然后以流水线方式训练以提高吞吐量。DeepSpeed 和 Megatron-LM 等框架提供了管理这些复杂并行化方案和优化通信开销的基础设施。

+   `torch.utils.checkpoint`或 TensorFlow 的重新计算包装器使得可以在不重写模型架构的情况下应用这项技术。

+   **混合精度训练**：这种方法使用 16 位浮点数（FP16 或 BF16）格式而不是标准的 32 位（FP32）格式进行大多数计算。这减少了内存占用并提高了吞吐量，因为算术运算更快，内存带宽使用更低。为了保持模型精度和数值稳定性，权重的主副本保持在 FP32 格式，并且通常使用动态损失缩放来防止梯度下溢。NVIDIA 的 Apex 库或 PyTorch 和 TensorFlow 的本地支持使得自动混合精度训练成为可能。这种方法在 NVIDIA 的 Tensor Cores 或 Google 的 TPUs 等现代硬件上特别有效，这些硬件针对低精度计算进行了优化。

*图 19*.1 总结了这些策略：

![图 19.1 – 扩展 RLHF 的策略](img/B31249_19_01.jpg)

图 19.1 – 扩展 RLHF 的策略

下面是如何实现梯度检查点的示例：

```py
from transformers import GPT2LMHeadModel
def enable_gradient_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        model.base_model.gradient_checkpointing_enable()
    return model
base_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
base_model = enable_gradient_checkpointing(base_model)
```

此功能为模型启用梯度检查点，这可以在训练期间显著减少内存使用，从而允许使用更大的批量大小或模型大小。

# RLHF 在语言模型中的局限性

虽然 RLHF 功能强大，但它面临几个挑战：

+   **奖励黑客攻击**：模型可能会利用奖励函数中的漏洞

+   **有限的反馈**：人类反馈可能无法涵盖所有可能的场景

+   **次优局部最优解**：优化过程可能陷入次优解

+   **扩展问题**：以规模获取高质量的人类反馈具有挑战性

为了解决奖励黑客攻击问题，考虑实施约束优化方法：

```py
def constrained_ppo_step(
    base_model, reward_model, constraint_model,
    optimizer, prompt, constraint_threshold=0.5
):
    outputs = base_model.generate(prompt, max_length=100,
        return_dict_in_generate=True, output_scores=True
    )
    generated_text = tokenizer.decode(
        outputs.sequences[0], skip_special_tokens=True
    )
    reward = reward_model(generated_text)
    constraint_value = constraint_model(generated_text)
    if constraint_value > constraint_threshold:
        return base_model  # Skip update if constraint is violated
    # Compute and apply policy update (similar to previous ppo_step)
    # ...
    return base_model
```

此功能在更新模型之前添加一个约束检查，通过确保生成的文本满足某些标准来帮助防止奖励黑客攻击。

此方法通过评估生成的输出不仅与奖励对齐，还与外部约束模型的一致性来修改标准的训练流程。该过程从使用给定提示从基础模型生成响应开始。生成的文本通过奖励模型和约束模型。奖励模型根据其与期望行为或目标的对齐情况分配标量奖励值。同时，约束模型评估输出是否满足指定的限制，例如避免有害内容、保持事实界限或遵守法律或伦理过滤器。

约束模型返回一个标量值，该值量化了违反约束的程度。此值与预定义的阈值进行比较。如果值超过阈值，表明输出违反了约束，则对该样本的训练步骤被终止。不计算梯度，模型参数保持不变。这种选择性更新机制确保只有既符合人类偏好又满足安全或策略约束的输出才对学习做出贡献。这种设计将约束信号与奖励函数解耦，保持学习目标和约束执行之间的清晰界限。因此，它保留了两个组件的完整性，并使系统更具可解释性和模块化。

# RLHF 的应用

RLHF 可以应用于各种 LLM 任务，包括以下内容：

+   开放式文本生成

+   对话系统

+   内容审核

+   摘要

+   代码生成

下面是应用 RLHF 到摘要任务的一个示例：

```py
def rlhf_summarization(
    base_model, reward_model, text, num_iterations=5
):
    prompt = f"Summarize the following text:\n{text}\n\nSummary:"
    for _ in range(num_iterations):
        summary = base_model.generate(prompt, max_length=100)
        reward = reward_model(summary)
        # Update base_model using PPO or another RL algorithm
        # ...
    return summary
# Example usage
long_text = "..."  # Long text to summarize
summary = rlhf_summarization(base_model, reward_model, long_text)
print(summary)
```

此函数将 RLHF 应用于文本摘要任务，通过根据奖励模型提供的奖励，迭代地改进摘要。

关键步骤包括使用基础模型生成摘要，从奖励模型接收反馈，并迭代地更新基础模型以随着时间的推移改进摘要。

下面是如何在这段代码中实现摘要的分解：

1.  `摘要以下文本:\n{text}\n\n 摘要:`。此提示发送到基础模型，以便生成摘要。

1.  使用`base_model.generate`函数从提示生成摘要。生成的摘要长度限制为 100 个标记（`max_length=100`）。摘要基于输入文本，是第一次尝试摘要。

1.  **奖励模型反馈**：在基础模型生成摘要后，奖励模型评估摘要的质量。奖励模型是一个独立的模型，它衡量生成的摘要与期望质量（如准确性、简洁性或连贯性）的匹配程度。奖励函数为摘要分配一个分数，该分数反映了其质量，基于模型的内部标准。

1.  `num_iterations`次（在这种情况下，默认为五次）。每次迭代包括生成新的摘要，从奖励模型接收反馈，并可能更新基础模型以在未来的迭代中改进摘要。

1.  `# 使用 PPO 或其他 RL 算法更新 base_model`，表示在每次迭代后，应使用强化学习算法（如 PPO）更新基础模型。此更新将调整基础模型的参数，以便根据奖励模型提供的反馈生成更好的摘要。然而，此处未提供模型更新的实际代码，通常涉及强化学习技术，根据接收到的奖励对基础模型进行微调。

1.  **最终输出**：在完成指定次数的迭代后，函数返回由基础模型生成的最终摘要。这个摘要预计是基于在迭代过程中从奖励模型收到的反馈进行多次改进的结果。

# 摘要

RLHF 是一种被许多前沿模型提供商（如 OpenAI 和 Anthropic）用于微调预训练模型的有力技术。本章讨论了这种模式背后的基本思想。由于人类参与了训练奖励模型的过程，因此 RLHF 仍然存在局限性，并且扩展性不佳。最近，一些公司如 DeepSeek 测试了无需人类反馈的更通用的强化学习。然而，这超出了本书的范围。您可以参考以下 DeepSeek 的研究论文以获取更多信息：[`arxiv.org/pdf/2501.12948`](https://arxiv.org/pdf/2501.12948)。

随着我们继续前进，我们将探讨 LLMs 的高级提示工程技术。在下一章中，我们将深入探讨通过精心设计的提示来引导 LLM 行为和输出的复杂方法，这些方法基于我们在这里讨论的对齐技术。这些高级提示策略将使您能够充分利用 LLMs 的潜力，同时保持对其输出的精细控制。

# 第四部分：高级提示工程技术

在本部分中，我们通过创新的提示策略和推理方法探索增强 LLMs 能力的高级技术。您将学习如何使用思维链和思维树提示来引导模型通过复杂的推理过程。我们还涵盖了无需直接观察的推理技术，使 LLMs 能够处理假设情景和抽象问题。反思技术将向您展示如何提示 LLMs 进行迭代自我改进，而自动多步推理和工具使用的方法将教会您如何将 LLMs 扩展到复杂的多功能系统。通过掌握这些高级方法，您将获得解锁 LLMs 全部潜力的能力，使它们能够解决甚至最具有挑战性的问题。

本部分包含以下章节：

+   *第二十章*，*思维链提示*

+   *第二十一章*，*思维树提示*

+   *第二十二章*，*推理与行动*

+   *第二十三章*，*无观察推理*

+   *第二十四章*，*反思技术*

+   *第二十五章*，*自动多步推理与工具使用*
