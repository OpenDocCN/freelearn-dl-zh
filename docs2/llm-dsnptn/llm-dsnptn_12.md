

# 第十二章：模型剪枝

在本章中，我们将探讨旨在在保持性能的同时减少模型大小的**模型剪枝**技术。

模型剪枝是指在保持性能的同时，从神经网络中系统地消除不必要的参数。对于 LLMs 来说，这通常涉及根据幅度、敏感性分析或基于梯度的重要性等标准识别和移除冗余或不重要的权重、神经元或注意力头。

你将学习如何实现各种剪枝方法，从基于幅度的剪枝到迭代技术，以及大小缩减与性能之间的权衡。此外，本章将帮助你决定是在训练期间还是训练后进行剪枝，以确保你的 LLMs 保持高效和有效。

在本章中，我们将涵盖以下主题：

+   基于幅度的剪枝

+   结构化与非结构化剪枝

+   迭代剪枝技术

+   训练期间剪枝与训练后剪枝

+   平衡剪枝和模型性能

+   将剪枝与其他压缩技术结合

# 基于幅度的剪枝

**基于幅度的剪枝**是最简单且最广泛使用的剪枝技术之一。这种方法背后的思想是移除对神经网络整体功能贡献最小的权重，通常，这些是幅度最小（绝对值）的权重。通过剪枝这些权重，模型变得更加紧凑和快速，对准确性的影响最小：

```py
import torch
import torch.nn.utils.prune as prune
# Assume model is an instance of a pre-trained LLM
model = ...  # Load or define your LLM model
# Prune 30% of the lowest magnitude weights in all Linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
# Remove the pruning reparameterization
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, 'weight')
```

在此代码示例中，基于幅度的剪枝移除了 LLM 所有线性层中幅度最低的 30%的权重。`prune.l1_unstructured`函数指定了具有最小 L1 范数的权重将被剪枝。

以下代码片段实现了在 PyTorch 模块中对给定参数张量进行无结构 L1 范数剪枝的`prune.l1_unstructured`函数，通过将绝对值最小的权重置零来实现：

```py
    def prune.l1_unstructured(module, name, amount):
    """Prunes weights with lowest L1 norm magnitude in a module's tensor"""
    # Get the parameter to prune
    tensor = getattr(module, name)
    # Calculate number of parameters to prune
    n_params_to_prune = int(amount * tensor.numel())
    # Get magnitude threshold (kth smallest absolute value)
    threshold = torch.kthvalue(
        tensor.abs().view(-1), n_params_to_prune
    ).values
    # Create and apply mask (zeros out weights below threshold)
    mask = tensor.abs() > threshold
    pruned_tensor = tensor.clone() * mask
    # Update parameter and register mask
    setattr(module, name, torch.nn.Parameter(pruned_tensor))
    module.register_buffer(f'{name}_mask', mask)
    # Add hook to maintain pruning during updates
    module.register_forward_pre_hook(
        lambda m, _: setattr(
            m, name,
            torch.nn.Parameter(
                getattr(m, name) * getattr(m, f'{name}_mask')
            )
        )
    )
    return mask
```

在这里，函数首先从模块中提取目标张量，并确定根据指定的比例`amount`应该剪枝多少个元素。它通过计算张量中第`k`小的绝对值来确定剪枝阈值，其中`k`对应于要剪枝的参数数量。然后创建一个二进制掩码，其中高于阈值的值被保留，而低于阈值的值被设置为零。这个掩码被应用于生成张量的剪枝版本，它替换了模块中的原始参数。掩码作为缓冲区存储，以在模型操作之间持久化，并注册了一个前向预钩子，以确保在每次前向传递之前强制执行剪枝，即使在训练过程中底层权重被更新，也能保持稀疏模式。

在模型剪枝中，使用 L1 范数通过求其成分的绝对值之和来评估模型中权重或参数的重要性，通常 L1 范数值较低表示不那么重要的参数，可以移除以减小模型大小同时保持性能。

剪枝后，调用`prune.remove`方法来移除剪枝重新参数化并使更改永久化。

基于幅度的剪枝对于具有许多小权重且对整体性能贡献不大的模型特别有效，但单独应用时可能不足以进行大规模剪枝。

# 结构化剪枝与无结构化剪枝

在剪枝 LLMs 时，你可以单独剪除权重（无结构化剪枝）或移除整个结构，如滤波器、通道或注意力头（结构化剪枝）：

+   之前描述的`prune.l1_unstructured`函数。

+   **结构化剪枝**：剪枝整个模型的部分，如神经元、通道或层。这种方法在现代硬件上更容易实现，并且通常会导致推理时间上的更好加速，尽管它可能对模型性能有更大的即时影响。

在 LLMs 中，可以使用 PyTorch 的内置工具实现结构化剪枝，如下面的代码所示。在这里，我们应用 L2 范数结构化剪枝来移除线性层中 30%的神经元，目标是整个权重矩阵的行，以有效地消除完整的神经元而不是仅仅单个连接：

```py
import torch.nn.utils.prune as prune
# Structured pruning of entire neurons in a layer
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.ln_structured(
            module, name='weight', amount=0.3, n=2, dim=0
        )
```

在这个结构化剪枝示例中，`ln_structured`函数根据给定维度的所有权重中的 L2 范数从线性层中移除整个神经元。结构化剪枝的选择可以显著降低计算复杂度，同时使模型更适合部署在标准硬件架构上。

接下来，我们将看到如何通过在多个训练步骤中每次剪除一小部分权重，而不是一次性剪除模型的大部分内容。

# 迭代剪枝技术

在这里，我们将讨论**迭代剪枝**，它允许你在多个训练步骤中每次剪除一小部分权重。这种方法降低了性能急剧下降的风险，并为模型提供了更多恢复和适应剪枝的机会。

迭代方法还允许在每个剪枝步骤后进行微调，使模型能够从权重减少中“恢复”：

```py
# Iteratively prune 10% of the model after every 10 epochs
for epoch in range(1, num_epochs+1):
    train(model, train_loader, optimizer)  # Regular training step
    if epoch % 10 == 0:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight',
                    amount=0.1)
                prune.remove(module, 'weight')  # Remove pruning mask after each step
    validate(model, val_loader)
```

在这个例子中，每 10 个 epoch 后剪除 10%的权重。逐渐移除权重确保模型有足够的时间在每次剪枝步骤之间进行调整。迭代剪枝与验证步骤相结合可以帮助找到模型大小和性能之间的更优平衡。

# 训练期间剪枝与训练后剪枝

应用剪枝时的一个关键决策是在训练期间还是训练完成后进行剪枝：

+   **训练过程中的剪枝**：这种方法允许模型通过迭代地剪枝权重来逐渐适应剪枝结构。模型可以补偿剪枝的权重，从而可能带来更好的最终性能。然而，这需要更多的计算资源和训练时间。

    这里是这种方法的一个例子：

    ```py
    import torch
    import torch.nn.utils.prune as prune
    # Assuming model is a pre-trained LLM
    model = ...  # Load or define your LLM model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    def train(model, train_loader, optimizer):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    # Prune 20% of the weights every 5 epochs during training
    for epoch in range(1, 20):
        train(model, train_loader, optimizer)
        # Apply pruning every 5 epochs
        if epoch % 5 == 0:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight',
                        amount=0.2)
                    prune.remove(module, 'weight')  # Remove reparameterization after each pruning
    ```

+   **训练后剪枝**：在这种方法中，剪枝是在模型完全训练后进行的。这种方法计算效率高，因为它不需要在训练过程中进行修改，并且你可以选择在之后微调模型。然而，与训练过程中的剪枝相比，它可能会导致更大的准确性下降。

    让我们看看训练后剪枝的一个例子：

    ```py
    # Assuming the model has already been fully trained
    model = ...  # Load or define your trained LLM model
    # Prune 30% of the weights in all Linear layers after training
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
    # Optionally, fine-tune the model after pruning
    fine_tune_epochs = 3
    for epoch in range(fine_tune_epochs):
        train(model, train_loader, optimizer)  # Fine-tuning the pruned model
    ```

这两种方法的选择取决于你的性能约束和可用资源。训练过程中的剪枝通常会导致更稳定的模型，而训练后的剪枝更快且更高效。

# 平衡剪枝和模型性能

在剪枝和模型性能之间找到合适的平衡至关重要。过于激进的剪枝可能导致性能显著下降，而剪枝不足可能不会带来足够的收益。关键在于识别哪些模型部分可以被剪枝而不会对准确性产生太大影响。这需要在每次剪枝步骤后进行仔细验证，并密切监控关键性能指标。这些指标包括参数减少率、推理速度提升、内存占用减少、困惑度变化以及特定任务的性能。在整个过程中，平衡准确性-效率权衡至关重要，以确保剪枝模型在参数减少的情况下仍能保持可接受的性能。

一种常见的策略是在剪枝后进行微调以恢复一些丢失的性能。微调允许模型适应剪枝结构并恢复其原始能力：

```py
import torch.nn.utils.prune as prune
# Assuming model has been trained and pruned
model = ...  # Pruned LLM model
# Apply fine-tuning to restore performance after pruning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate for fine-tuning
fine_tune_epochs = 5
for epoch in range(fine_tune_epochs):
    train(model, train_loader, optimizer)  # Reuse the train function from earlier
    validate(model, val_loader)  # Validation step to monitor performance
```

在这个例子中，在剪枝部分权重之后，模型使用较低的学习率进行微调以恢复性能。较低的学习率允许模型逐渐适应新的剪枝结构，防止学习特征的不稳定。在每次微调步骤后进行验证，以监控模型的进展并确保剪枝没有显著降低性能。

让我们看看如何将剪枝与其他模型压缩技术相结合。

# 将剪枝与其他压缩技术相结合

剪枝可以与其他模型压缩技术相结合，如量化或蒸馏，以实现模型大小和复杂性的更大减少。结合这些技术通常会产生更紧凑的模型，同时保持高性能。

## 剪枝和量化

在剪枝后进行**量化**可以显著减少模型大小并加快推理速度，尤其是在资源受限的环境中：

```py
import torch
import torch.nn.utils.prune as prune
import torch.quantization as quant
# Prune the model first
model = ...  # Pre-trained LLM
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)
        prune.remove(module, 'weight')
# Apply dynamic quantization after pruning
quantized_model = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# Check the size reduction
print("Original model size:", torch.cuda.memory_allocated())
print("Quantized model size:", torch.cuda.memory_allocated())
```

## 剪枝和知识蒸馏

你还可以将剪枝与**知识蒸馏**相结合，其中较小的、剪枝的**学生模型**被训练来模仿较大的、训练良好的**教师模型**的行为：

```py
# Teacher and student models for knowledge distillation
teacher_model = ...  # Larger, fully trained model
student_model = ...  # Smaller model to be distilled and pruned
def distillation_loss(student_outputs, teacher_outputs, temperature):
    return torch.nn.KLDivLoss()(
        torch.nn.functional.log_softmax(
            student_outputs / temperature
        ),
        torch.nn.functional.softmax(
            teacher_outputs / temperature
        )
    )
# Train the smaller, pruned model using knowledge distillation
temperature = 2.0
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
for batch in train_loader:
    inputs, _ = batch
    teacher_outputs = teacher_model(inputs)
    student_outputs = student_model(inputs)
    loss = distillation_loss(
        student_outputs, teacher_outputs, temperature
    )
    loss.backward()
    optimizer.step()
```

这种方法允许学生模型在更少的参数下实现高性能。知识蒸馏有助于通过从未剪枝的教师模型中传递高级表示来补偿剪枝造成的精度损失。

这些示例说明了剪枝如何在训练期间或之后应用，平衡性能要求，并结合其他压缩技术，如量化和知识蒸馏，以创建更高效的 LLMs。

# 摘要

在本章中，我们探讨了 LLMs 的各种模型剪枝技术，包括基于幅度的剪枝、结构化与非结构化剪枝以及迭代剪枝方法。我们讨论了在训练期间与训练后剪枝所涉及的权衡，以及剪枝后微调以恢复丢失性能的重要性。通过结合剪枝与其他压缩技术，如量化和蒸馏，你可以创建更适合在资源受限环境中部署的更高效的 LLMs。

在下一章中，我们将探讨用于 LLMs 的量化技术，重点关注降低数值精度以提高模型效率，同时保持性能。你将学习如何应用训练后和量化感知训练来进一步优化你的 LLMs。
