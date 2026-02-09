# 9

# 正则化

**正则化**是一组方法，它约束或修改学习过程，以防止模型过于精确地记住训练数据，鼓励它学习更稳健和可泛化的模式。

正则化是训练 LLMs（大型语言模型）的一个关键方面，用于防止过拟合并提高泛化能力。过拟合是有害的，因为它会导致模型在训练数据上表现异常出色，而在新的、未见过的数据上却表现糟糕。当模型过拟合时，它实际上记住了训练数据集中的噪声和特殊性，而不是学习可泛化的模式和关系。这会在开发阶段产生高准确率的错觉，但会导致现实世界的表现不佳，使模型无法有效地用于其旨在对新颖输入进行准确预测的目的。

在本章中，您将了解针对 LLMs 特别定制的不同正则化技术。我们将探讨分层自适应正则化、微调中的正则化和多种技术的组合等方法。您将深入了解这些策略的实施及其对模型性能的影响。

在本章中，我们将涵盖以下主题：

+   L2 正则化（岭回归）

+   Dropout

+   分层自适应正则化

+   梯度裁剪和噪声注入

+   在迁移学习和微调场景中的正则化

+   针对下一代 LLMs 的新兴正则化技术

# L2 正则化（岭回归）

**L2 正则化**，也称为岭回归或权重衰减，是一种用于防止机器学习模型过拟合的技术。它通过向损失函数添加一个惩罚项来实现，该惩罚项与模型权重的平方成正比。这个惩罚项阻止模型将大权重分配给单个特征，从而得到一个更简单、更通用的模型。通过最小化包含原始损失和惩罚项的合并损失函数，模型在拟合训练数据的同时保持权重较小，最终提高其泛化到新、未见过的数据的能力。

这里是如何使用它的：

```py
from torch.optim import AdamW
def train_with_weight_decay(
    model, train_dataloader, weight_decay=0.01, lr=5e-5, epochs=3
):
    optimizer = AdamW(model.parameters(), lr=lr,
        weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss / len(train_dataloader):.4f}"
        )
# Assuming you have a train_dataloader
# train_with_weight_decay(model, train_dataloader)
```

在这个实现中，我们使用了我们在*第七章*中讨论的 AdamW 优化器，它正确地实现了权重衰减。`weight_decay`参数控制正则化的强度。一个典型的值是`0.01`，但您可能需要根据您的特定模型和数据集进行调整。

# Dropout

**Dropout**是另一种强大的正则化技术，它在训练过程中随机“丢弃”一部分神经元。

Dropout 通过在每次训练迭代中随机停用一部分神经元来帮助对抗过拟合。这迫使网络发展冗余的信息流路径。这种技术通过在单个网络内创建一种集成学习形式，防止神经元过度依赖彼此，其中不同的子网络处理类似任务。结果是，一个更健壮的模型，它依赖于分布式表示而不是记忆特定模式，最终在推理期间所有神经元都活跃时，提高了对未见数据的泛化能力。

它在大型神经网络（如 LLM）中特别有效。以下是如何在基于变换器的 LLM 中实现 dropout 的方法：

```py
class TransformerWithDropout(nn.Module):
    def __init__(
    self, vocab_size, d_model, nhead, num_layers, dropout=0.1
):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)  # Simplified positional encoding
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead,
                dim_feedforward=4*d_model, dropout=dropout),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder(
            torch.arange(x.size(1), device=x.device))
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Transform to shape expected by transformer
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Transform back
        return self.fc_out(x)
model = TransformerWithDropout(vocab_size=50257,
    d_model=768, nhead=12, num_layers=12, dropout=0.1)
print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)
```

在这个实现中，dropout 在嵌入层之后以及每个变换器层内应用。`0.1`的 dropout 率是典型的，但根据您的具体用例，您可能需要调整这个值。

请记住，dropout 仅在训练期间应用，不在推理期间（当模型被用于做出预测时）应用。

在训练期间，神经元以指定的概率随机“停用”（失活）（例如，`0.5`表示每个神经元有 50%的机会在该训练批次中被关闭）。这迫使网络学习更鲁棒的特征，因为它不能依赖于任何单个神经元始终存在。

在推理（测试、评估或部署）期间，dropout 被禁用，所有神经元都是活跃的。然而，权重通常按 dropout 率进行缩放，以考虑到训练期间活跃的神经元比推理期间更多。这种缩放确保了期望的输出幅度保持一致。

这种仅在训练中应用的 dropout 是使其作为正则化技术有效的关键部分——它在训练期间创建了一种集成学习的形式，同时在实际使用时仍允许网络发挥全部能力。

# 层级自适应正则化

层级自适应正则化涉及对模型的不同层应用不同的正则化强度。这对于 LLM 尤其有效，其中较低层可能从较少的正则化中受益以捕捉基本模式，而较高层可能需要更强的正则化以防止过拟合。

以下 Python 代码定义了一个`LayerwiseAdaptiveRegularization`类，这是一个 PyTorch `nn.Module`，旨在封装一个基础变换器模型并应用一个随着模型层深度线性增加的 dropout 率：

```py
class LayerwiseAdaptiveRegularization(nn.Module):
    def __init__(
        self, base_model, num_layers, base_dropout=0.1,
        dropout_increase_per_layer=0.02
    ):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers
        self.base_dropout = base_dropout
        self.dropout_increase_per_layer = dropout_increase_per_layer
        self.set_layerwise_dropout()
    def set_layerwise_dropout(self):
        for i, layer in enumerate(self.base_model.transformer.h):
            dropout = self.base_dropout
                + i * self.dropout_increase_per_layer
            layer.attn.dropout.p = dropout
            layer.mlp.dropout.p = dropout
    def forward(self, *args, kwargs):
        return self.base_model(*args, kwargs)
base_model = create_lm_model()
model = LayerwiseAdaptiveRegularization(base_model, num_layers=12)
```

`LayerwiseAdaptiveRegularization`类使用基础模型、层数、起始 dropout 概率以及后续每层的增量进行初始化。然后，它配置变换器块中注意力和 MLP 子层的 dropout 概率。最后，其前向方法简单地通过封装的基础模型传递输入。其使用示例是通过将`create_lm_model()`与这个层级 dropout 正则化封装来展示的。

此实现包装了一个基本的 GPT-2 模型，并将递增的 dropout 率应用于更高层。基本 dropout 率是`0.1`，后续每层增加`0.02`。

# 梯度裁剪和噪声注入

梯度裁剪和噪声注入是用于提高大型语言模型（LLMs）训练稳定性和泛化的技术。

梯度裁剪，虽然主要用于优化稳定性（参见*第七章*），但可以间接地促进正则化。通过限制梯度的幅度，它可以约束模型参数的更新，可能带来更平滑的优化路径并防止过拟合。在某些情况下，梯度裁剪可以有效地减少某些参数的影响，尤其是当这些参数的梯度持续被裁剪时。这可能导致一种隐式稀疏性，即不那么重要的参数被有效地降低权重。

噪声注入是一种常用的正则化技术，用于提高机器学习模型的泛化能力。通过向输入数据、权重或激活函数添加少量噪声，噪声注入有助于防止过拟合。该技术迫使模型对训练数据中的特定模式依赖性降低，鼓励它学习更稳健、更通用的特征，这些特征适用于不同的数据集。这种方法在神经网络中特别有用，以下噪声可以在各个阶段注入：

+   **输入噪声**：直接向输入数据添加噪声，帮助模型对输入的变异性更加鲁棒

+   **权重噪声**：在训练过程中扰动权重，鼓励模型更好地泛化

+   **激活噪声**：向激活函数添加噪声，导致决策边界更加平滑并减少过拟合

这些方法有助于防止过拟合，减少异常值的影响，并鼓励模型探索更广泛的解决方案，最终导致更稳健和可靠的语言模型。

下面是如何实现梯度裁剪和噪声注入的方法：

```py
import torch.nn.functional as F
def train_with_grad_clip_and_noise(
    model, train_dataloader, grad_clip=1.0,
    noise_factor=0.01, lr=5e-5, epochs=3
):
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            # Add noise to inputs
            input_ids = batch['input_ids']
            noise = torch.randn_like(
                input_ids, dtype=torch.float) * noise_factor
            noisy_inputs = input_ids.float() + noise
            noisy_inputs = noisy_inputs.long().clamp(
                min=0, max=model.config.vocab_size - 1)
            outputs = model(input_ids=noisy_inputs, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss / len(train_dataloader):.4f}"
        )
# Assuming you have a train_dataloader
# train_with_grad_clip_and_noise(model, train_dataloader)
```

此实现应用梯度裁剪以防止梯度爆炸，并向输入添加少量噪声以提高鲁棒性。`noise_factor`控制添加噪声的量；您可能需要根据您的特定用例进行调整。

函数初始化一个**AdamW 优化器**，并对数据集进行指定数量的轮次迭代。在每次训练步骤中，它清除旧梯度，向输入标记添加噪声（确保值保持在词汇范围内），并将带噪声的输入送入模型进行正向和反向传播。**梯度裁剪**防止梯度爆炸，确保稳定训练。优化器更新模型参数，并跟踪损失以监控进度。最后，函数打印每轮的平均损失。

接下来，让我们探讨转移学习和微调场景中的规范化。

# 转移学习和微调场景中的规范化

在微调预训练的 LLM 时，仔细调整规范化以避免阻碍特定任务的适应同时仍然防止过拟合是很重要的。以下是一种使用自适应规范化的微调方法：

```py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
def fine_tune_with_adaptive_regularization(
    pretrained_model_name, train_dataloader,
    initial_dropout=0.1, epochs=3
):
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        current_dropout = initial_dropout * (1 - epoch / epochs)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss / len(train_dataloader):.4f}, "
            f"Dropout: {current_dropout:.4f}"
        )
# Assuming you have a train_dataloader
# fine_tune_with_adaptive_regularization('gpt2', train_dataloader)
```

此实现从更高的 dropout 率开始，并在微调过程中逐渐降低它。这允许模型适应新任务，同时仍然保持一些规范化以防止过拟合。这种方法也称为自适应 dropout。

自适应 dropout 之所以有效，是因为它根据神经元的重要性动态调整 dropout 率，而不是在整个网络中应用均匀的 dropout。通过选择性更频繁地丢弃不那么关键的神经元，同时保留重要的特征检测器，自适应 dropout 在规范化和信息保留之间创造了一个最佳平衡。这种有针对性的方法比标准 dropout 更有效地防止过拟合，因为它通过重要的神经元保持网络的复杂模式学习能力，同时积极规范化冗余或噪声敏感的部分，从而产生泛化能力更强且在关键特征上性能损失较小的模型。

# 出现的规范化技术

近年来，出现了解决现代深度学习架构复杂挑战的复杂技术。这些新方法不仅超越了简单地防止过拟合，它们旨在提高模型鲁棒性，在损失景观中找到更好的极值，并通过创新的训练策略增强泛化。从几何启发方法如**削弱度感知最小化**（**SAM**）到高级优化策略如**随机权重平均**（**SWA**），这些新兴的规范化技术正在重塑我们处理模型训练和泛化的方式。

## 随机权重平均

SWA 是一种通过平均优化轨迹上的多个点的权重来提高神经网络泛化能力的技巧，有效地找到更平坦、更稳健的极小值，这些极小值在未见过的数据上的表现优于传统优化方法通常找到的尖锐极小值。**随机梯度下降**（**SGD**）是一种基本的优化算法，通过跟踪在随机选择的训练数据小批次上计算的损失函数的负梯度来更新模型参数，通过近似全梯度计算同时引入有益的噪声来帮助逃离不良局部极小值，从而实现大型模型（如神经网络）的高效训练。

WA 涉及使用修改后的学习率计划对 SGD 轨迹上的多个点进行平均。它通过找到更广泛的极值来提高泛化能力。以下是一个代码示例：

```py
from torch.optim.swa_utils import AveragedModel, SWALR
# Create SWA model and scheduler
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
# Training loop with SWA
for epoch in range(100):
    if epoch > 75:  # Start SWA after epoch 75
        swa_model.update_parameters(model)
        swa_scheduler.step()
```

## 削弱度感知最小化

SAM 寻求位于具有均匀低损失值邻域中的参数，从而实现更好的泛化。其关键特性如下：

+   寻找“平坦”的极小值而不是尖锐的极小值

+   提高了对输入扰动的鲁棒性

+   通常比标准的 SGD 提供更好的泛化能力

让我们在下面的 Python 代码中实现`SAM`类：

```py
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        self.rho = rho
        self.base_optimizer = base_optimizer(params)
    def step(self):
        # First forward-backward pass
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        # Perturb weights
        for group in self.param_groups:
            for p in group['params']:
                e_w = p.grad * scale
                p.add_(e_w)
        # Second forward-backward pass
        self.base_optimizer.step()
```

## 基于差分隐私的正则化

**差分隐私**（**DP**）是一种技术，通过对数据或计算添加精心校准的噪声来保护个人隐私，同时仍然允许有用的见解，确保任何单个数据点的包含或排除都不会显著影响模型性能。

基于 DP 的正则化是一种技术，通过向模型的训练过程中添加噪声来增强模型隐私，从而保护个人数据点不被暴露在模型输出或学习表示中。通过引入受控的随机性，基于 DP 的正则化限制了模型对任何特定数据样本的依赖，从而降低了过拟合的风险，并使模型对个人数据点的变化不那么敏感。这种方法在需要数据保密的应用中特别有价值，因为它确保模型可以在不泄露训练数据具体信息的情况下学习可泛化的模式，使其在医疗保健、金融和其他需要数据保密的领域非常有用。

以下代码片段实现了`DPOptimizer`类：

```py
class DPOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params, noise_multiplier=1.0, max_grad_norm=1.0
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
    def step(self):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'],
                                     self.max_grad_norm)
        # Add noise
        for p in self.param_groups[0]['params']:
            noise = torch.randn_like(p.grad) * self.noise_multiplier
            p.grad.add_(noise)
```

## 快速梯度符号方法

**快速梯度符号方法**（**FGSM**）是一种通过向输入数据添加小的、有针对性的扰动来创建对抗性样本的技术，使模型误分类。它通过计算损失函数相对于输入的梯度并应用一个轻微调整，以最大化模型错误的方向来实现。输入数据通过一个称为ϵ的因子控制的微小量进行轻微改变，以创建一个可以欺骗机器学习模型的“对抗性示例”。FGSM 通常用于测试模型鲁棒性和对抗性训练，其中模型在对抗性示例上进行训练以增强安全性。然而，FGSM 的单步特性使其快速但对抗强大防御的效果较差，与实现更高攻击成功率的迭代方法不同。

让我们看看它是如何在这里实现的：

```py
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
```

## 看前优化器

预视优化器是一种创新的优化技术，通过维持两组参数：快速权重和慢速权重，来增强传统优化器（如 Adam 或 SGD）的训练稳定性和收敛性。快速权重通过使用标准优化器频繁更新，而慢速权重通过与其同步更新来较少地更新。这种方法允许优化器更好地探索损失景观，因为优化器可以逃离局部最小值并平滑优化轨迹中的振荡。通过利用基础优化器和预视机制的优势，这种优化器实现了更快的收敛和更好的泛化，使其成为深度学习模型训练中的一个宝贵补充。

以下代码片段展示了如何实现预视优化器：

```py
class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0
        self.slow_weights = [
            [p.clone().detach() for p in group['params']]

            for group in optimizer.param_groups
        ]
    def step(self):
        self.step_counter += 1
        self.optimizer.step()
        if self.step_counter % self.k == 0:
            for group, slow_weights in zip(
                self.optimizer.param_groups, self.slow_weights
            ):
                for p, q in zip(group['params'], slow_weights):
                    p.data.mul_(self.alpha).add_(
                        q, alpha=1.0 - self.alpha)
                    q.data.copy_(p.data)
```

# 摘要

在本章中，我们介绍了诸如权重衰减和 L2 正则化、dropout 方法、逐层自适应正则化以及结合多种正则化方法等基本概念。我们还讨论了迁移学习和微调场景下的正则化策略，以及增强模型稳定性的技术，例如梯度裁剪和噪声注入。此外，我们还介绍了各种新兴的正则化方法。

在下一章中，我们将探讨检查点和恢复技术，并研究为什么这些技术对于管理长时间运行的训练过程至关重要。
