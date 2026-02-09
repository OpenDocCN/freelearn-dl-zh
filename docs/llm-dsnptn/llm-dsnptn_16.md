

# 第十六章：可解释性

**LLM 中的可解释性**指的是模型理解并解释模型如何处理输入和生成输出的能力。

可解释性对于 LLM 的几个原因：

+   **信任与透明度**：理解 LLM 如何得出其输出可以建立用户和利益相关者的信任

+   **调试和改进**：可解释性技术可以帮助识别模型弱点并指导改进

+   **伦理考量**：可解释的模型允许更好地评估潜在的偏见和公平性问题

+   **合规性监管**：在某些领域，可解释的人工智能模型可能需要满足监管合规性要求

在本章中，我们将探讨理解和解释 LLM 输出和行为的高级技术。我们将讨论如何将这些技术应用于基于 Transformer 的 LLM，并检查模型性能与可解释性之间的权衡。

在本章中，我们将讨论以下主题：

+   注意力可视化技术

+   探测方法

+   使用归因方法解释 LLM 的预测

+   基于 Transformer 的 LLM 的可解释性

+   机制可解释性

+   可解释性与性能之间的权衡

# 注意力可视化技术

**注意力机制**是基于 Transformer 的 LLM 的关键组成部分（参见*第一章*）。可视化注意力模式可以提供模型如何处理和关注输入的不同部分的见解。

下面是一个如何在基于 Transformer 的模型中可视化注意力的示例：

```py
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
def visualize_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(inputs, output_attentions=True)
    attention = outputs.attentions[-1].squeeze().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=tokens,
        yticklabels=tokens, cmap="YlGnBu")
    plt.title("Attention Visualization")
    plt.show()
# Example usage
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
text = "The cat sat on the mat."
visualize_attention(model, tokenizer, text)
```

此代码提供了一个简单的方法来可视化 BERT 模型在处理给定输入句子时的注意力机制。它首先导入必要的库：PyTorch 用于模型处理，Hugging Face 的`transformers`库用于加载 BERT 模型和分词器，以及 Matplotlib 和 Seaborn 用于可视化。`visualize_attention`函数接受一个 BERT 模型、分词器和输入文本。它首先使用分词器对输入进行分词，然后将分词后的输入通过`output_attentions=True`传递给模型以检索注意力权重。从返回的输出中，它提取最后一层的注意力矩阵（即`outputs.attentions[-1]`），将其从计算图中分离出来，并将其转换为 NumPy 数组。这个矩阵表示序列中每个标记对其他每个标记的关注程度。然后将标记 ID 转换回可读的标记，用于标记热图的轴。使用 Seaborn 的`heatmap`，注意力分数被可视化为一个彩色编码的矩阵，这使得解释模型在处理每个标记时关注哪些单词变得更加容易。最后，代码加载预训练的 BERT 基础模型和分词器，定义一个示例句子，并调用可视化函数以显示注意力图，从而提供对 BERT 内部工作的见解。

请记住，在 LLMs 中，注意力图并不总是与模型推理相关。虽然它们显示了模型关注的区域，但它们并不一定解释了为什么做出某个决定。注意力可能分散、不一致或误导，有时会突出无关的标记，同时仍然产生正确的输出。由于 LLMs 以分布式表示编码信息，推理通常发生在直接注意力之外，涉及跨层的深层潜在变换。研究还表明，注意力图可以在不改变模型行为的情况下被操纵，这证明了它们不是推理的最终解释。为了更好的可解释性，它们应该与基于梯度的方法、探查技术和因果分析相结合。

# 探查方法

**探查**涉及在 LLM 的内部表示上训练简单模型，以评估在不同层中捕获了哪些语言属性。

变换器中的不同层专门处理不同的语言属性。底层捕获句法和标记身份；中间层处理语法和句子结构；高层专注于语义、推理和事实回忆。这种层次结构在训练过程中自然出现，底层在句法任务上表现出色，而高层在语义推理上表现出色。探查研究证实了这种专业化，有助于可解释性、微调和模型压缩以进行特定任务的优化。

这里有一个如何实现探查任务的例子：

```py
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def probe_bert_layers(model, tokenizer, texts, labels, layer_nums):
    # Get BERT embeddings for each layer
    def get_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt",
            padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
        return outputs.hidden_states
    results = {}
    for layer in layer_nums:
        embeddings = [
            get_embeddings(text)[layer]
            .squeeze()
            .mean(dim=0)
            .numpy() for text in texts
        ]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
        # Train and evaluate probe
        probe = LogisticRegression(random_state=42)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[f"Layer_{layer}"] = accuracy
    return results
# Example usage
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
texts = ["The cat sat on the mat.", "The dog chased the ball.", ...]  # Add more examples
labels = [0, 1, ...]  # Corresponding labels (e.g., 0 for simple, 1 for complex sentences)
layer_nums = [1, 6, 12]  # Layers to probe
probe_results = probe_bert_layers(model, tokenizer, texts, labels,
    layer_nums)
for layer, accuracy in probe_results.items():
    print(f"{layer} Accuracy: {accuracy:.2f}")
```

这段代码实现了一个简单的探查任务，以评估 BERT 模型的不同层如何捕捉特定的语言属性（在这种情况下，句子复杂性）。

# 使用归因方法解释 LLM 预测

**归因**方法旨在确定哪些输入特征对模型的预测贡献最大。

我们需要讨论归因方法，因为理解模型为何产生特定预测对于现实应用中的可解释性和可靠性至关重要。归因方法提供了一种系统的方式来追踪特定输入标记对模型输出的影响，这在 LLMs（大型语言模型）中尤为重要，因为预测通常来自复杂、高维的标记嵌入以及多个注意力层之间的非线性交互。没有归因，用户和开发者将面临一个黑盒模型，该模型产生输出而不提供任何透明的理由，这使得验证决策、调试行为或确保与预期用例一致变得困难。

一种流行的归因方法是**集成梯度**。

**集成梯度**是一种归因方法，用于通过量化每个输入特征对模型输出的贡献来解释神经网络的预测。它通过沿从基线到实际输入的直线路径，将模型输出的梯度与输入进行积分来计算特征归因。

请记住，LLM 中的基于梯度的方法可能会因为对输入扰动的敏感性、小批量方差和梯度饱和而变得嘈杂，这会影响训练稳定性和可解释性。在优化中，噪声可能导致振荡或次优收敛，而在可解释性中，如集成梯度等方法可能在不同的运行中产生不一致的归因。这种不稳定性降低了模型洞察力的可信度，特别是对于相似输入。梯度平滑、平均和二阶优化等技术有助于减轻噪声，但会增加计算开销，在 LLM 开发中在效率和精度之间形成权衡。

下面是一个如何实现基于 Transformer 模型的集成梯度的示例：

```py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
def integrated_gradients(
    model, tokenizer, text, target_class, steps=50
):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    baseline_ids = torch.zeros_like(input_ids)
    alphas = torch.linspace(0, 1, steps)
    delta = input_ids - baseline_ids
    accumulated_grads = 0
    for alpha in alphas:
        interpolated_ids = baseline_ids + alpha * delta
        interpolated_ids.requires_grad_()
        outputs = model(interpolated_ids)
        pred = outputs.logits[:, target_class]
        model.zero_grad()
        pred.backward()
        accumulated_grads += interpolated_ids.grad
    attributions = \
        (input_ids - baseline_ids) * accumulated_grads / steps
    return attributions.squeeze().detach().numpy()
# Example usage
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
text = "This movie was fantastic!"
target_class = 1  # Assuming 1 is the positive sentiment class
attributions = integrated_gradients(model, tokenizer, text,
    target_class)
# Visualize attributions
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
plt.figure(figsize=(10, 5))
plt.bar(range(len(tokens)), attributions)
plt.xticks(range(len(tokens)), tokens, rotation=45)
plt.title("Integrated Gradients Attribution")
plt.show()
```

这段代码演示了如何使用集成梯度方法通过将模型的预测归因于单个输入标记来解释基于 BERT 的序列分类模型。`integrated_gradients`函数首先使用分词器将输入文本编码为标记 ID，并创建一个形状相同的基线输入，其中填充了零。然后，它在基线和实际输入之间进行小步插值（默认为`50`），以计算沿此路径的梯度。对于每个插值输入，它计算模型针对指定目标类的输出，执行反向传播以获取关于输入的梯度，并累积这些梯度。最后，它计算这些梯度的平均值，并将其乘以输入差异（*输入 – 基线*）以获得归因——这量化了每个输入标记对预测的贡献。在定义模型和分词器后，代码在示例文本上运行归因方法，并将结果以条形图的形式显示出来，其中每个条形对应于一个标记及其对目标预测的重要性。这种技术提供了一种更原则性和模型感知的方式来理解输入的哪些部分最具影响力，使其成为可解释性和对模型预测信任的有力工具。

# 基于 Transformer 的 LLM 的可解释性

基于 Transformer 的 LLM 在可解释性方面面临着独特的挑战和机遇。以下是一些需要考虑的关键领域：

+   **多头注意力**: 分析单个注意力头以揭示专用功能

+   **位置嵌入**: 理解模型如何使用位置信息

+   **层分析**: 检查不同语言特征如何在各层中被捕捉

下面是一个分析多头注意力的示例：

```py
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
def analyze_multihead_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(inputs, output_attentions=True)
    attention = outputs.attentions[-1].squeeze().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    num_heads = attention.shape[0]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(num_heads):
        sns.heatmap(attention[i], xticklabels=tokens,
            yticklabels=tokens, ax=axs[i], cmap="YlGnBu")
        axs[i].set_title(f"Head {i+1}")
    plt.tight_layout()
    plt.show()
# Example usage
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
text = "The president of the United States visited Paris last week."
analyze_multihead_attention(model, tokenizer, text)
```

这段代码可视化了 BERT 模型最后一层不同头的注意力模式，允许比较它们的专用功能。

# 机制可解释性

**机制可解释性**（**MI**）是一个新兴领域，旨在从详细、组件级别理解神经网络如何处理信息——类似于我们可能如何逆向工程一个机械装置。MI 不仅仅观察输入和输出，而是试图追踪信息在网络中的流动，识别特定的计算模式，并理解网络的各个部分（如单个神经元或注意力头）如何贡献于模型的行为。

MI 之所以重要，是因为它超越了表面解释，揭示了神经网络，尤其是像 LLM 这样的复杂模型，实际工作的内部机制。通过分析特定组件（如神经元、层或注意力头）如何处理和转换信息，MI 帮助研究人员建立对模型行为的更深入、更原则性的理解。这种洞察力对于几个原因至关重要：它通过使模型更透明来增强信任；它使精确调试和有针对性的改进成为可能；它有助于发现和减轻隐藏的偏差或漏洞；它支持开发更安全、更可控的 AI 系统。最终，MI 使我们更接近于将神经网络视为不是黑盒，而是可以更有信心地分析、解释和改进的可理解系统。

让我们一步步构建：

1.  首先，让我们创建一个简单的可解释模型结构：

    ```py
    import torch
    import torch.nn as nn
    class InterpretableTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, nhead, num_layers):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer,
                num_layers)
            self.fc = nn.Linear(d_model, vocab_size)
    ```

1.  现在，让我们添加一个提取注意力模式的方法，这对于理解模型如何处理标记之间的关系至关重要：

    ```py
    def get_attention_patterns(self, x):
        """Extract attention weights from each layer"""
        x = self.embedding(x)
        attention_patterns = []
        for layer in self.transformer.layers:
            # Register a hook to capture attention weights
            attention_weights = None
            def hook(module, input, output):
                nonlocal attention_weights
                attention_weights = output[1]  # attention weights
            handle = layer.self_attn.register_forward_hook(hook)
            x = layer(x)
            attention_patterns.append(attention_weights)
            handle.remove()
        return attention_patterns
    ```

1.  让我们添加一个神经元激活分析来了解哪些神经元对于特定输入最为活跃：

    ```py
    def analyze_neuron_activations(self, x, layer_idx):
        """Analyze individual neuron activations in a specific layer"""
        activations = []
        def hook(module, input, output):
            activations.append(output.detach())
        # Register hook on specific layer
        handle = list(self.transformer.layers)[layer_idx]\
            .register_forward_hook(hook)
        # Forward pass
        with torch.no_grad():
            self(x)
        handle.remove()
        layer_activations = activations[0]
        # Find most active neurons
        mean_activation = layer_activations.mean(dim=(0,1))  # Average across batch and sequence
        top_neurons = torch.topk(mean_activation, k=10)
        return top_neurons.indices, top_neurons.values
    ```

1.  我们可以添加一种因果干预的方法——暂时修改特定的神经元，看看它如何影响输出：

    ```py
    def intervention_study(self, x, layer_idx, neuron_idx):
        """Study how zeroing out specific neurons affects the output"""
        original_output = None
        modified_output = None
        def hook_original(module, input, output):
            nonlocal original_output
            original_output = output.detach()
        def hook_modified(module, input, output):
            nonlocal modified_output
            modified = output.clone()
            modified[:,:,neuron_idx] = 0  # Zero out specific neuron
            modified_output = modified
            return modified
        layer = list(self.transformer.layers)[layer_idx]
        # Get original output
        handle = layer.register_forward_hook(hook_original)
        self(x)
        handle.remove()
        # Get modified output
        handle = layer.register_forward_hook(hook_modified)
        self(x)
        handle.remove()
        return original_output, modified_output
    ```

1.  最后，让我们添加一个可视化辅助工具：

    ```py
    import matplotlib.pyplot as plt
    def visualize_attention(attention_weights, tokens=None):
        """Visualize attention patterns"""
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights[0].cpu(), cmap='viridis')
        if tokens is not None:
            plt.xticks(range(len(tokens)), tokens, rotation=45)
            plt.yticks(range(len(tokens)), tokens)
        plt.colorbar()
        plt.title('Attention Pattern')
        plt.show()
    ```

这是如何使用这些工具一起使用的方法：

```py
# Initialize model
model = InterpretableTransformer(vocab_size=1000,
    d_model=256, nhead=8, num_layers=4)
# Sample input
input_ids = torch.randint(0, 1000, (1, 20))  # Batch size 1, sequence length 20
# Get attention patterns
attention_patterns = model.get_attention_patterns(input_ids)
# Analyze neuron activations
top_neurons, activation_values = model.analyze_neuron_activations(
    input_ids, layer_idx=0
)
# Perform intervention study
original, modified = model.intervention_study(input_ids,
    layer_idx=0, neuron_idx=42)
# Visualize attention
visualize_attention(attention_patterns[0])  # Visualize first layer's attention
```

每个组件帮助我们理解模型的不同方面：

+   注意力模式显示了模型如何将不同的标记相互关联

+   神经元激活分析揭示了哪些神经元对于处理特定输入最为重要

+   因果干预通过观察当我们修改它们时输出如何变化，帮助我们理解特定神经元的作用

+   可视化工具帮助我们更直观地解释这些模式

这是一个基本的实现——真正的 MI 研究通常涉及更复杂的技术，如电路分析、激活修补以及如何在网络中实现特定能力（如归纳或否定）的详细研究。

# 可解释性和性能之间的权衡

模型性能和可解释性之间往往存在紧张关系。更复杂的模型往往表现更好，但更难解释。以下是一些平衡这种权衡的方法：

+   **蒸馏**：训练更小、更可解释的模型来模仿更大的 LLM

+   **稀疏模型**：鼓励模型权重或激活的稀疏性以更容易进行解释

+   **模块化架构**：设计具有可解释组件的模型

这里有一个模型蒸馏的简单示例：

```py
import torch
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizer)
def distill_bert(
    teacher_model, student_model, tokenizer, texts, temperature=2.0
):
    teacher_model.eval()
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
            teacher_logits = teacher_outputs.logits / temperature
        student_outputs = student_model(inputs)
        student_logits = student_outputs.logits / temperature
        loss = loss_fn(torch.log_softmax(student_logits, dim=-1),
                       torch.softmax(teacher_logits, dim=-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return student_model
# Example usage
teacher_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased")
student_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["This movie was great!", "I didn't like the book.", ...]  # Add more examples
distilled_model = distill_bert(
    teacher_model, student_model, tokenizer, texts
)
```

此代码演示了一个简单的蒸馏过程，其中较小的 DistilBERT 模型学习模仿较大的 BERT 模型的行为。

此外，我们还需要牢记压缩和可解释性之间的权衡，这涉及到在效率、准确性和透明度之间取得平衡。量化、剪枝和知识蒸馏等压缩技术显著减少了模型大小和推理延迟，使得 LLMs 能够在边缘设备上运行或以更低的计算成本运行。然而，这些方法可能会降低性能，尤其是在长上下文推理、罕见标记预测或特定领域任务中，在这些任务中保留复杂的权重结构至关重要。此外，高度压缩的模型通常变得不太可解释，因为移除神经元或注意力头或降低精度会掩盖模型的内部表示，使得分析为什么产生某些输出变得更加困难。

相反，可解释性技术，如特征归因、注意力可视化和探针，帮助研究人员和用户了解 LLMs 如何处理信息、检测偏差或调试故障，但它们通常需要访问完整且未修改的模型。较大的、未压缩的模型保留了更多的内部知识和细微的表示，这使得它们更容易分析但更难高效部署。此外，高度可解释的架构有时会对模型灵活性施加约束，限制它们在多样化任务中泛化的能力。

关键挑战是找到最佳平衡——例如，**低秩自适应**（**LoRA**）允许在不修改完整模型权重的情况下进行微调，有助于保持某些可解释性同时实现高效部署。随着 LLMs 的扩展，开发者必须权衡压缩带来的效率提升与降低透明度的风险，尤其是在医疗保健、法律和 AI 安全等高风险应用中，理解模型决策与性能一样关键。

# 摘要

在本章中，我们为您提供了可解释性技术工具包，以深入了解您的 LLMs 的决策过程，这对于开发更透明和值得信赖的 AI 系统至关重要。

随着 LLMs 在规模和能力上的持续增长，可解释性研究将在确保这些强大的模型可理解、可信赖且安全地部署在实际应用中发挥关键作用。可解释性中的关键挑战和未来方向包括为大型模型扩展这些技术、理解因果关系、实现交互式探索以及开发针对特定下游任务的技术。

在下一章中，我们将探讨评估和减轻大型语言模型（LLMs）中公平性和偏差的技术。这是负责任的人工智能开发的关键方面，基于我们讨论的解释方法，以确保 LLMs 不仅强大且可解释，而且在输出和决策过程中公平且无偏见。
