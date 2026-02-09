

# 第十三章：量化

在本章中，我们将深入探讨**量化**方法，这些方法可以优化 LLM 以在资源受限的设备上部署，例如移动电话、嵌入式系统或边缘计算环境。

量化是一种降低数值表示精度的技术，从而缩小模型的大小并提高其推理速度，而不会严重损害其性能。

量化在以下场景中特别有益：

+   **资源受限的部署**：当在内存、存储或计算能力有限的设备上部署模型时，例如移动电话、物联网设备或边缘计算平台

+   **对延迟敏感的应用**：当需要实时或近实时响应时，量化可以显著减少推理时间

+   **大规模部署**：当大规模部署模型时，即使模型大小和推理时间的适度减少也可以转化为基础设施和能源消耗的显著成本节约

+   **带宽受限的场景**：当模型需要通过有限的带宽连接下载到设备时，较小的量化模型可以减少传输时间和数据使用量

+   **具有冗余精度的模型**：当许多 LLM 被训练以比良好性能所需的精度更高的精度时，它们成为量化极佳的候选者。

然而，在某些情况下，量化可能不适用，例如以下情况：

+   **对精度高度敏感的任务**：对于即使精度略有下降也不可接受的应用，例如某些医疗诊断或关键金融模型

+   **已针对低精度优化的模型**：如果一个模型被专门设计或训练以在较低精度下高效运行，进一步的量化可能会导致性能显著下降

+   **小型模型**：对于已经紧凑的模型，量化操作的开销在某些硬件配置中可能超过其带来的好处

+   **开发和微调阶段**：在积极开发和实验期间，使用全精度模型通常更可取，以获得最大灵活性并避免掩盖潜在问题

+   **硬件兼容性**：目标硬件可能缺乏对您计划使用的特定量化格式的有效支持（例如，某些设备可能没有优化 INT8 或 INT4 计算能力）

+   **具有不同敏感性的复杂架构**：LLM 架构的某些部分（例如注意力机制）可能比其他部分对量化更敏感，需要更复杂的混合精度方法，而不是简单的量化

通过理解这些考虑因素，您可以就是否以及如何将量化技术应用于您的 LLM 部署做出明智的决定，在性能需求和资源限制之间取得平衡。

在本章中，你将了解不同的量化策略，到本章结束时，你将能够应用量化方法使你的 LLM 更高效，同时确保任何精度降低对模型性能的影响最小。

在本章中，我们将涵盖以下主题：

+   理解基础知识

+   混合精度量化

+   硬件特定考虑

+   比较量化策略

+   将量化与其他优化技术相结合

# 理解基础知识

量化是指降低模型权重和激活的精度，通常从 **32 位浮点**（**FP32**）降低到更低的精度格式，如 **16 位**（**FP16**）或甚至 **8 位整数**（**INT8**）。目标是减少内存使用，加快计算速度，并使模型在计算能力有限的硬件上更容易部署。虽然量化可能导致性能下降，但精心调整的量化方案通常只会导致精度损失很小，特别是对于具有稳健架构的 LLM。

有两种主要的量化方法：**动态量化**和**静态量化**。

+   使用 `torch.quantization.quantize_dynamic` 对预训练的 LLM 的线性层进行动态量化：

    ```py
    import torch
    from torch.quantization import quantize_dynamic
    # Assume 'model' is a pre-trained LLM (e.g., transformer-based model)
    model = ...
    # Apply dynamic quantization on linear layers for INT8 precision
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Check size reduction
    print(f"Original model size: {torch.cuda.memory_allocated()} bytes")
    print(f"Quantized model size: {torch.cuda.memory_allocated()} bytes")
    ```

    这立即降低了内存需求并提高了推理速度。

+   `torch.quantization.prepare` 和 `torch.quantization.convert`:

    ```py
    import torch
    import torch.nn as nn
    import torch.quantization
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 256)
            self.relu = nn.ReLU()
            self.out = nn.Linear(256, 10)
        def forward(self, x):
            x = self.relu(self.fc(x))
            return self.out(x)
    # Create and prepare model for static quantization
    model_fp32 = SimpleModel()
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig(
        'fbgemm')
    prepared_model = torch.quantization.prepare(model_fp32)
    # Calibration step: run representative data through the model
    # (This example uses random data; replace with real samples)
    for _ in range(100):
        sample_input = torch.randn(1, 784)
        prepared_model(sample_input)
    # Convert to quantized version
    quantized_model = torch.quantization.convert(prepared_model)
    # Model is now statically quantized and ready for inference
    print(quantized_model
    ```

    此静态量化模型使用每个量化张量固定的缩放和零点参数，允许硬件加速器实现更高的推理效率。

    与动态量化不同，静态量化在推理之前需要一个具有代表性数据的校准阶段。在这个阶段，模型以评估模式运行以收集激活统计信息，然后使用这些统计信息来计算量化参数。权重和激活量在推理前进行量化并保持固定，从而实现更快的执行和更可预测的性能。

根据何时应用量化，也存在两种主要的量化方法：

+   **后训练量化（PTQ）**：在模型完全训练后应用量化，最小化或无需额外训练。可以实施为静态（带有校准）或动态。

+   **量化感知训练（QAT）**：通过在正向传递中添加模拟量化操作的假量化操作，在训练期间模拟量化效果，同时保持梯度以全精度。通常导致部署时的静态量化。

## PTQ

PTQ 是量化最直接的形式，在模型完全训练后应用。它不需要模型重新训练，通过将高精度权重和激活转换为低精度格式（通常是 INT8）来实现。PTQ 对于重新训练昂贵或不切实际的情况非常理想，并且对于对精度损失不太敏感的任务效果最佳。

请记住，一些 PTQ 方法通常需要在代表性数据集上执行校准步骤，以确定最佳量化参数，如缩放因子和零点，捕获推理期间的激活分布，并最小化原始模型输出和量化模型输出之间的误差。这个过程有助于量化算法理解网络中权重和激活的数值范围和分布，从而实现从更高精度格式（如 FP32）到更低精度格式（如 INT8 或 INT4）的更准确映射，最终在减少内存占用和部署的计算需求的同时保持模型精度。

此示例演示了静态 PTQ：

```py
import torch
import torch.quantization as quant
# Load pre-trained model
model = ...
# Convert model to quantization-ready state
model.eval()
model.qconfig = torch.quantization.default_qconfig
# Prepare for static quantization
model_prepared = quant.prepare(model)
# Apply quantization
model_quantized = quant.convert(model_prepared)
```

模型首先使用`.eval()`方法置于评估模式，然后使用`.prepare()`方法准备量化，最后转换为量化模型。这种方法为在低功耗设备上高效部署 LLM 提供了一种有效手段。

## QAT

QAT 通过将量化效果纳入训练过程本身，超越了简单的 PTQ。这允许模型学习如何补偿量化引起的噪声，通常比 PTQ 有更好的性能，尤其是在更复杂的任务中。

在量化加速训练（QAT）期间，训练过程中权重和激活都使用较低的精度进行模拟，但在梯度计算时保持较高的精度。这种方法在应用需要高性能且量化程度较大的情况下特别有用。

在以下示例中，我们使用`get_default_qat_qconfig()`配置模型进行 QAT，该配置在训练阶段模拟量化行为：

```py
import torch.quantization as quant
# Set up QAT
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# Prepare for QAT
model_prepared = quant.prepare_qat(model)
# Training loop (for simplicity, only showing initialization)
for epoch in range(num_epochs):
    train_one_epoch(model_prepared, train_loader, optimizer)
    validate(model_prepared, val_loader)
# Convert to quantized version
model_quantized = quant.convert(model_prepared.eval())
```

模型训练完成后，将其转换为适合部署的量化版本。与 PTQ 相比，QAT 通常能带来更好的模型精度，尤其是在更复杂或关键的应用中。

# 混合精度量化

**混合精度量化**是一种更灵活的方法，它利用单个模型内的多个数值精度级别。例如，模型中不那么关键的层可以使用 INT8，而更敏感的层则保持在 FP16 或 FP32。这允许在性能和精度之间有更大的控制权。使用混合精度量化可以显著减小模型大小和推理时间，同时保持 LLM 的关键特性。

以下代码演示了量化示例，以优化 LLM 训练或推理中的内存使用和速度：

```py
from torch.cuda.amp import autocast
# Mixed precision in LLM training or inference
model = ...
# Use FP16 where possible, fall back to FP32 for sensitive computations
with autocast():
    output = model(input_data)
```

在此示例中，我们使用 PyTorch 的**自动混合精度**（**AMP**）库中的`autocast()`函数，在模型中精度不那么关键的部位启用 FP16 计算，而保留 FP32 用于更敏感的层。这种方法有助于减少内存使用和推理时间，同时不会严重影响性能。

# 硬件特定考虑因素

不同的硬件平台——例如 GPU、CPU 或专门的加速器如 TPU——在处理量化模型时可能具有截然不同的能力和性能特征。例如，某些硬件可能原生支持 INT8 操作，而其他硬件则针对 FP16 进行了优化。

理解目标部署硬件对于选择合适的量化技术至关重要。例如，NVIDIA GPU 因其支持混合精度训练和推理而非常适合 FP16 计算，而 CPU 通常因为硬件加速的整数操作而在 INT8 量化方面表现更佳。

当在生产中部署 LLM 时，重要的是要尝试针对特定硬件量身定制的量化策略，并确保您的模型利用平台的优势。

# 比较量化策略

当比较不同的量化策略时，每种方法都提供了独特的优势和挑战，这些可以通过实现复杂度、精度保持、性能影响和资源需求等因素来衡量。

在实现复杂度方面，PTQ 是最简单的，只需在原始模型训练之外做最少的工作。动态量化更复杂，因为它涉及到更多的运行时考虑，因为需要动态处理激活。混合精度量化引入了更多的复杂性，因为它需要逐层评估精度敏感性，并可能需要为优化执行开发定制的内核。QAT 被认为是最复杂的，需要将伪量化节点集成到训练图中，并延长训练时间以考虑量化引入的噪声。

在精度保持方面，QAT 表现最佳，将精度保持在浮点性能的小范围内，尤其是在针对激进的量化（小于 8 位）时。混合精度量化在精度保持方面也排名很高，因为它允许关键层保持更高的精度，从而在性能和精度之间取得良好的平衡。PTQ 通常在可接受的精度范围内保持精度，尽管更复杂的架构可能会遭受更高的精度损失。动态量化通常在基于 RNN 的模型中比 PTQ 保持更好的精度，但在 CNN 架构中表现不佳，尤其是在激活对输入分布变化敏感时。

在资源需求方面，PTQ（Post-Training Quantization，训练后量化）需要的资源最少，使其成为计算能力有限且需要快速部署的场景的理想选择。动态量化在资源消耗上略高，因为它在运行时处理激活量化，尽管这减少了内存和存储的负担。混合精度量化由于敏感性分析，在实现过程中需要更多资源，但在推理阶段可以更高效，尤其是在支持多种精度的硬件上。QAT（Quantization-Aware Training，量化感知训练）是资源消耗最多的，因为它需要额外的训练时间，训练期间的内存使用更高，以及更多的计算资源来适应量化。

从性能角度来看，PTQ 在内存节省和计算加速方面提供了显著的改进，通常可以减少 75%的存储，并在兼容硬件上实现 2–4 倍的加速。然而，QAT 虽然压缩比相似，但在训练期间增加了开销，但通过产生可以处理更激进量化而不会造成显著性能损失的模式来补偿。动态量化提供了与 PTQ 相似的内存节省，但由于运行时开销，其计算加速通常较低。混合精度量化可以提供接近浮点性能，其加速取决于硬件执行具有不同精度级别的模型的有效性。

选择最佳量化策略的决策框架取决于具体的项目需求。当快速部署是首要任务，模型架构相对简单，且可以接受轻微的精度损失时，PTQ 是合适的。当精度至关重要，有可用的重新训练资源，并且需要激进量化时，QAT 是最佳选择。动态量化适合需要运行时灵活性和处理不同输入分布的场景，尤其是在基于 RNN 的架构中。混合精度量化对于需要不同精度需求的复杂模型是最优的，在这些模型中，需要高精度和性能，并且硬件可以有效地管理多种精度格式。

每种量化策略基于精度、复杂性、性能和资源之间的权衡，服务于不同的目的，使用户能够根据其部署环境的特定需求定制其方法。

*表 13.1* 对每种策略进行了比较。

| **策略** | **精度** | **复杂性** | **性能** | **资源** |
| --- | --- | --- | --- | --- |
| PTQ | 适用于简单模型；随着复杂度的增加而下降 | 低；最小设置 | 75%的存储减少；2–4 倍的加速 | 低；所需计算最小 |
| QAT | 最高；适用于小于 8 位的子集 | 高；需要扩展训练 | 高压缩率，最佳精度 | 高；需要密集训练 |
| 动态 | 适合 RNN；对 CNN 较弱 | 中；运行时开销 | 良好的内存节省；较慢的计算 | 中；运行时处理 |
| 混合精度 | 高；灵活的精度选项 | 中高；层特定调整 | 硬件依赖的速度提升 | 中高在设置期间 |

表 13.1 – 量化策略比较

在实践中，某些场景可能从结合策略中受益。例如，您可能最初应用 PTQ 以实现快速部署，然后有选择地在精度敏感层上使用 QAT。另一种方法可能涉及对特定层使用混合精度，同时为激活应用动态量化以平衡运行时灵活性和性能。

# 将量化与其他优化技术相结合

量化可以与其他优化技术相结合，例如剪枝和知识蒸馏，以创建适用于在资源受限设备上部署的高度高效模型。通过利用多种方法，您可以显著减小模型大小，同时保持或最小化对性能的影响。这在将 LLM 部署在边缘设备或移动平台上特别有用，因为这些平台计算和内存资源有限。

## 剪枝和量化

最有效的组合之一是**剪枝**后跟量化。首先，剪枝从模型中移除冗余权重，减少参数数量。然后量化降低剩余权重的精度，这进一步减小模型大小并提高推理速度。以下是一个示例：

```py
import torch
import torch.nn.utils.prune as prune
import torch.quantization as quant
# Step 1: Prune the model
model = ...  # Pre-trained LLM model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.5)  
        # Prune 50% of the weights
        prune.remove(module, 'weight')
# Step 2: Apply dynamic quantization to the pruned model
quantized_model = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8  # Convert to INT8 precision
)
# Check size reduction
print("Original model size:", torch.cuda.memory_allocated())
print("Quantized model size:", torch.cuda.memory_allocated())
```

在此示例中，剪枝应用于移除所有线性层中 50%的权重，动态量化将剩余权重的精度降低到 INT8 以进一步减小尺寸。

结果是一个紧凑、高度优化的模型，消耗更少的计算资源，使其适合在硬件能力有限的设备上部署。

## 知识蒸馏和量化

另一种强大的组合是先知识蒸馏后量化。在这种情况下，一个较小的学生模型被训练以复制较大教师模型的行为。一旦学生模型训练完成，量化就被应用于进一步优化学生模型以进行部署。这种组合在您需要以最小的计算开销保持高性能时特别有用。

让我们一步一步地看看一个例子：

1.  定义教师和学生模型：

    ```py
    import torch
    import torch.nn.functional as F
    teacher_model = ...  # Larger, fully trained model
    student_model = ...  # Smaller model to be trained through distillation
    ```

1.  定义知识蒸馏损失函数：

    ```py
    def distillation_loss(
        student_outputs, teacher_outputs, temperature=2.0
    ):
        teacher_probs = F.softmax(
            teacher_outputs / temperature, dim=1)
        student_probs = F.log_softmax(
            student_outputs / temperature, dim=1)
        return F.kl_div(student_probs, teacher_probs,
            reduction='batchmean')
    ```

1.  为知识蒸馏添加训练循环：

    ```py
    optimizer = torch.optim.Adam(student_model.parameters(),
        lr=1e-4)
    for batch in train_loader:
        inputs, _ = batch
        optimizer.zero_grad()
    ```

1.  通过教师和学生模型进行前向传递：

    ```py
        teacher_outputs = teacher_model(inputs)
        student_outputs = student_model(inputs)
    ```

    通过教师模型和学生模型对相同输入数据进行正向传播，生成它们各自的对数输出。这一并行推理步骤是计算蒸馏损失所必需的，该损失量化了学生模型复制教师模型行为的多接近程度。通过比较这些输出，训练过程可以引导学生内部化教师的知识，而无需原始标签。

1.  计算蒸馏损失：

    ```py
        loss = distillation_loss(student_outputs, teacher_outputs)
        loss.backward()
        optimizer.step()
    ```

    计算蒸馏损失允许学生模型通过最小化它们输出分布之间的差异来从教师模型中学习。这引导学生近似更大、更准确的教师模型的行为，同时保持其自身的紧凑结构。通过反向传播此损失并通过优化更新模型参数，学生模型逐渐使其预测与教师模型对齐，从而在降低模型复杂性的同时提高性能。

1.  量化蒸馏的学生模型：

    ```py
    quantized_student_model = quant.quantize_dynamic(
        student_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ```

1.  检查大小和效率改进：

    ```py
    print("Quantized student model size:",
        torch.cuda.memory_allocated())
    ```

    知识蒸馏用于训练一个较小的学生模型，该模型模仿较大教师模型的行为，并将量化应用于学生模型，降低其权重的精度，以进一步优化其部署。

此方法有助于在大幅减少模型尺寸的同时保持性能，使其非常适合低功耗或实时应用。

通过将量化与剪枝和知识蒸馏相结合，你可以实现高度优化的模型，这些模型在大小、效率和性能之间取得平衡。这些模型特别适用于部署在边缘设备或资源受限的环境中。

# 摘要

在本章中，我们探讨了优化 LLMs 的不同量化技术，包括 PTQ、QAT 和混合精度量化。我们还涵盖了针对特定硬件的考虑因素和评估量化模型的方法。通过将量化与其他优化方法相结合，如剪枝或知识蒸馏，LLMs 可以变得既高效又强大，适用于现实世界的应用。

在下一章中，我们将深入探讨评估 LLMs 的过程，重点关注文本生成、语言理解和对话系统的指标。理解这些评估方法是确保你的优化模型在多样化的任务中按预期表现的关键。

# 第三部分：大型语言模型的评估和解释

在本部分，我们专注于评估和解释 LLMs 的方法，以确保它们满足性能预期并与预期用例保持一致。您将学习如何使用针对各种 NLP 任务的评估指标，并应用交叉验证技术来可靠地评估您的模型。我们探讨了允许您理解 LLMs 内部工作原理的解释方法，以及识别和解决其输出中偏差的技术。对抗鲁棒性是另一个关键领域，有助于您防御模型受到的攻击。此外，我们介绍了从人类反馈中进行强化学习（RLHF）作为一种将 LLMs 与用户偏好对齐的强大方法。通过掌握这些评估和解释技术，您将能够微调您的模型以实现透明度、公平性和可靠性。

本部分包含以下章节：

+   *第十四章*, *评估指标*

+   *第十五章*, *交叉验证*

+   *第十六章*, *可解释性*

+   *第十七章*, *公平性与偏差检测*

+   *第十八章*, *对抗鲁棒性*

+   *第十九章*, *从人类反馈中进行强化学习*
