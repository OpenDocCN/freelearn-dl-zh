# 11

# 微调

在这个设计模式中，你将学习到微调预训练语言模型的有效策略**。

微调大型语言模型（LLMs）解决了迁移学习中的一个基本优化问题：在大数据集上预训练有助于 LLMs 学习通用的语言技能和知识，但预训练数据与特定任务数据之间的差异可能会降低性能。微调使用较小且精心选择的任务数据集来更新模型，使其更适合任务需求。这个过程保留了预训练中的有用知识，同时提高了模型在目标任务上有效执行的能力。

在本章中，我们将涵盖以下主题：

+   实现迁移学习和微调

+   层冻结和解冻策略

+   学习率调度

+   领域特定技术

+   少样本和零样本微调

+   持续微调和灾难性遗忘

# 实现迁移学习和微调

我们将使用以下代码块来演示使用 GPT-2 的迁移学习，包括模型初始化、数据处理和微调工作流程。我们将使用 Transformers 库和 WikiText 数据集来微调预训练语言模型：

1.  首先，我们使用配置的填充加载并初始化 GPT-2 模型和标记器：

    ```py
    def load_model_and_tokenizer(model_name="gpt2"):
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    ```

1.  然后，以下代码块使用`512`的序列长度管理数据集加载和文本标记化：

    ```py
    def prepare_dataset(dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1"
    ):
        dataset = load_dataset(dataset_name, dataset_config)
        return dataset
    def tokenize_function(examples, tokenizer):
        return tokenizer(
            examples["text"], truncation=True,
            padding="max_length", max_length=512)
    ```

1.  最后，我们设置训练配置，初始化训练器，并执行微调：

    ```py
    def fine_tune_lm(model, tokenizer,
        dataset, output_dir="./fine_tuned_model"
    ):
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
        )
        trainer.train()
        trainer.save_model()
    ```

    代码设置了一个`fine_tune_lm`函数，用于准备和执行语言模型微调。它首先使用批量处理对数据集进行标记化，然后配置包括 epoch、批大小、预热步骤和权重衰减的训练参数。接下来，它使用模型、参数和数据集初始化训练器，运行训练过程，并最终保存微调后的模型。

批大小对训练稳定性和性能都有显著影响。较大的批大小允许更多的并行化，并在强大的硬件上更快地训练，但需要更多的内存。它们可以通过平均更多示例来提供更稳定的梯度估计，从而可能实现更好的收敛。然而，与较小的批大小相比，非常大的批大小可能泛化较差，因为它们可能导致模型收敛到更尖锐的局部最小值。较小的批大小在梯度更新中引入更多噪声，这有助于逃离局部最小值，并可能找到更好的解决方案，但训练时间更长。找到最佳批大小需要平衡特定模型和数据集的硬件约束、收敛稳定性和泛化性能。

当微调 LLMs 时，我们通常不需要更新所有模型的参数。选择性地**冻结**和**解冻**层可以导致更高效和有效的微调。

# 层冻结和解冻策略

选择性冻结和解冻层的理念源于知识在深度神经网络中的结构和分布方式。LLM 中的底层倾向于捕获更多通用语言表示，例如句法、词性和形态，而高层则更专业且与任务相关。这种层次结构允许我们利用早期层中已经编码的通用语言知识，同时仅微调网络的任务特定部分。

通过冻结底层，我们保留了它们的预训练能力并防止了灾难性遗忘，这在整个模型在狭窄领域数据集上无差别更新时可能会发生。这也大大减少了可训练参数的数量，从而降低了内存使用并加快了收敛速度。同时，选择性解冻上层允许模型在不干扰其核心语言理解能力的情况下，为新任务或领域调整其表示。

让我们看看我们如何实现这一点：

1.  首先，我们通过禁用除指定数量的最终层之外的所有层的梯度来实现选择性层冻结：

    ```py
    def freeze_layers(model, num_layers_to_freeze):
        for param in model.base_model.parameters():
            param.requires_grad = False
        for i, layer in enumerate(model.base_model.transformer.h):
            if i >= len(model.base_model.transformer.h) -\
                num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = True
    ```

1.  然后，我们在训练周期中管理渐进层解冻：

    ```py
    def gradual_unfreeze(model, trainer, num_epochs, total_layers):
        layers_per_epoch = total_layers // num_epochs
        for epoch in range(num_epochs):
            freeze_layers(model, (epoch + 1) * layers_per_epoch)
            trainer.train(resume_from_checkpoint=True)
    ```

1.  最后，我们为渐进解冻过程配置优化的训练参数：

    ```py
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=5,  # Increased epochs for better learning
        per_device_train_batch_size=16,  # Larger batch size
        per_device_eval_batch_size=16,
        warmup_steps=1000,  # More warmup steps
        learning_rate=2e-5,  # Added learning rate
        weight_decay=0.1,  # Increased weight decay
        logging_dir="./logs",
        save_steps=500,  # Added save frequency
        eval_steps=500   # Added evaluation frequency
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    ```

此实现引入了两种关键策略：

+   `freeze_layers`：此函数冻结所有层，除了最后`num_layers_to_freeze`层

+   `gradual_unfreeze`：此函数在训练过程中逐步解冻层

渐进解冻方法允许模型首先适应其高级特征，然后逐步微调低级特征。这可以提高性能并帮助防止灾难性遗忘。

由于以下原因，灾难性遗忘得到了减少：

+   层冻结通过禁用早期层的梯度更新来保留这些层中的知识，在预训练期间保持学习到的基本表示，同时仅适应特定任务的后期层。这保留了模型的一般知识，同时允许对新任务进行适应。

+   渐进解冻实施了一种分阶段的方法，其中训练开始时仅解冻最终层（其中包含更多特定于任务的表示），然后逐步解冻早期层。这允许模型首先适应高级特征，然后再进行更根本的变化，提供一种温和的过渡，有助于保持之前学习到的模式。

+   训练配置通过精心平衡的学习率、增加预热步骤和更高的权重衰减来支持这些方法，进一步防止参数发生剧烈变化。增加的周期允许更渐进的适应，同时保存和评估检查点提供监控，以防止解冻过程中的过拟合。

这些技术共同创造了一个更受控制的微调过程，在适应新任务的同时保留了一般知识。

通过应用适当的 学习率调度，可以显著提高微调性能，我们将在下一节中探讨。

# 学习率调度

如前所述，适当的 **学习率调度** 通常用于有效的微调。以下代码演示了 LLM 微调的常见学习率调度技术，提供了**线性**和**余弦预热**策略以优化训练：

1.  首先，我们使用所需的导入和函数初始化设置调度框架：

    ```py
    from transformers import (
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup)
    def fine_tune_with_lr_scheduling(
        model, tokenizer, dataset, scheduler_type="linear",
        num_epochs=3
    ):
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True)
    ```

1.  接下来，我们使用改进的默认值配置优化训练参数：

    ```py
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=32,  # Increased batch size
        per_device_eval_batch_size=32,
        weight_decay=0.1,  # Increased weight decay
        logging_dir="./logs",
        learning_rate=2e-5,  # Adjusted learning rate
        warmup_ratio=0.1,   # Added warmup ratio
        eval_steps=100,     # Added evaluation frequency
        save_steps=100      # Added save frequency
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    ```

1.  最后，我们实现具有动态预热步骤计算的 学习率调度：

    ```py
    num_training_steps = len(tokenized_dataset["train"]) //
        training_args.per_device_train_batch_size * num_epochs
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=num_training_steps // 10,  # 10% warmup
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=num_training_steps // 10,  # 10% warmup
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError("Unsupported scheduler type")
    ```

    此实现提供了两种常见的学习率调度策略：

    +   在预热期间，从`0`线性减少到初始`lr`。我们曾在*第七章*的*损失函数和优化策略*部分讨论过这一点。然而，需要注意的是，我们还需要在微调中使用相同的预热调度。预热有助于防止训练早期突然的权重更新，从而确保更平滑的收敛。

    +   **带预热余弦调度**：类似于线性调度，但在此情况下，下降遵循余弦曲线。

这些调度策略可以帮助稳定训练并可能带来更好的收敛。

# 领域特定微调技术

当为特定领域微调 LLM 时，我们通常需要调整我们的方法。让我们看看一个针对科学语料库的领域特定微调示例。以下代码使用自定义数据集准备和训练配置实现了科学文本的领域特定微调：

1.  首先，我们使用指定的块大小和语言模型整理器设置科学文本数据集的准备：

    ```py
    import torch
    from transformers import (
        TextDataset, DataCollatorForLanguageModeling )
    def prepare_scientific_dataset(file_path, tokenizer):
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=128,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
        return dataset, data_collator
    ```

1.  接下来，我们处理训练和评估的数据集准备：

    ```py
    def fine_tune_for_scientific_domain(
        model, tokenizer, train_file, eval_file,
        output_dir="./scientific_model"
    ):
        train_dataset, data_collator =
            prepare_scientific_dataset(train_file, tokenizer)
        eval_dataset, _ = prepare_scientific_dataset(
            eval_file, tokenizer)
    ```

1.  最后，我们为科学领域适应配置优化训练参数：

    ```py
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,            # Reduced epochs
        per_device_train_batch_size=8, # Increased batch size
        per_device_eval_batch_size=8,
        warmup_steps=1000,            # Increased warmup
        weight_decay=0.1,             # Increased weight decay
        learning_rate=3e-5,           # Added learning rate
        logging_dir="./logs",
        evaluation_strategy="steps",   # Changed to steps
        eval_steps=500,               # Added eval frequency
        save_steps=500,               # Added save frequency
        gradient_accumulation_steps=4  # Added gradient accumulation
    )
    ```

此实现包括几个领域特定考虑因素：

+   `TextDataset`处理领域特定文本文件

+   **较小的批量大小**：科学文本通常有更长的序列，因此我们减少批量大小

+   **更多 epoch**：领域适应可能需要更多的训练迭代

+   **定期评估**：在每个 epoch 之后，我们评估模型以跟踪验证损失和关键领域特定指标，确保适当的适应。

当为特定领域进行微调时，请考虑以下步骤：

+   适应领域特定术语的词汇表

+   使用领域特定评估指标

+   可能修改模型架构以适应领域特定特征

在下一节中，我们将探讨几种用于微调模型且来自目标领域几乎没有标记数据的策略。

# 少样本和零样本微调

**少样本**和**零样本学习**是强大的技术，可以用于将 LLMs 适应新任务，而无需或仅需最少量的特定任务训练数据。让我们实现一个少样本微调方法：

1.  我们创建一个包含任务几个示例的提示：

    ```py
    def prepare_few_shot_dataset(examples, tokenizer, num_shots=5):
        few_shot_examples = examples[:num_shots]
        prompt = "\n\n".join(
            [
                f"Input: {ex['input']}\n"
                f"Output: {ex['output']}"
                for ex in few_shot_examples
            ]
        )
        prompt += "\n\nInput: {input}\nOutput:"
        def tokenize_function(example):
            full_prompt = prompt.format(input=example['input'])
            tokenized_prompt = tokenizer(full_prompt,
                truncation=True,
                padding="max_length", max_length=512)
            tokenized_output = tokenizer(
                example['output'], truncation=True,
                padding="max_length", max_length=512)
            tokenized_prompt['labels'] = \
                [-100] * len(tokenized_prompt['input_ids'])
                + tokenized_output['input_ids']
            return tokenized_prompt
        return examples.map(tokenize_function)
    ```

1.  模型随后在基于提示的数据集上进行微调：

    ```py
    def few_shot_fine_tune(
        model, tokenizer, dataset, num_shots=5, num_epochs=3
    ):
        few_shot_dataset = prepare_few_shot_dataset(dataset,
            tokenizer, num_shots)
        training_args = TrainingArguments(
            output_dir="./few_shot_model",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=few_shot_dataset,
        )
        trainer.train()
        return trainer
    ```

    `few_shot_fine_tune`函数实现了少样本微调，它使用最少量的示例将预训练模型适应新任务。它接受一个模型、分词器、数据集和配置参数（`num_shots=5`，`num_epochs=3`），然后使用`prepare_few_shot_dataset`准备数据的小子集，使用`TrainingArguments`配置训练（指定输出位置、批大小和优化参数），使用这些组件初始化一个`Trainer`对象，通过`trainer.train()`执行训练过程，并最终返回包裹在`Trainer`对象中的训练模型——所有这些操作都使用 Hugging Face Transformers 库框架，该框架是语言模型中常用的一种。

1.  微调后的模型可以推广到新任务的新实例：

    ```py
    # Usage
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset("your_dataset")  # Load your few-shot dataset
    few_shot_trainer = few_shot_fine_tune(model, tokenizer, dataset)
    ```

此实现展示了少样本微调。

对于零样本学习，你通常会依赖预训练模型理解任务描述的能力，无需任何特定任务的示例或微调。

# 持续微调和灾难性遗忘

**持续微调**涉及在保留先前任务性能的同时适应新任务。然而，这可能导致**灾难性遗忘**。在 LLMs 中，灾难性遗忘指的是在没有适当机制来保留先前知识的情况下，模型在针对新任务或数据微调时丢失先前学习的信息。

让我们实施一个简单的策略来减轻这一点：

1.  首先，我们计算参数重要性并实现**弹性权重巩固**（**EWC**）损失以保留关键权重：

    ```py
    import copy
    def ewc_loss(model, old_model, importance, loss):
        ewc_lambda = 0.01
        for n, p in model.named_parameters():
            if n in importance:
                loss += ewc_lambda * importance[n]
                    * (p - old_model[n]).pow(2).sum()
        return loss
    def compute_importance(model, dataset):
        importance = {}
        model.eval()
        for batch in dataset:
            model.zero_grad()
            output = model(batch)
            loss = output.loss
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    if n not in importance:
                        importance[n] = p.grad.data.clone().pow(2)
                    else:
                        importance[n] += p.grad.data.clone().pow(2)
        return importance
    ```

1.  我们接下来实现以下代码，该代码在多个任务上管理顺序训练的同时保持先前知识：

    ```py
    def continual_fine_tune(
            model, tokenizer, datasets, num_epochs=3
    ):
        old_model = None
        importance = None
        for i, dataset in enumerate(datasets):
            if old_model is not None:
                importance = compute_importance(
                    old_model, datasets[i-1])
            old_model = copy.deepcopy(model)
            tokenized_dataset = dataset.map(
                lambda examples: tokenize_function(examples,
                    tokenizer),
                batched=True)
    ```

1.  最后，我们为持续学习定义优化训练参数：

    ```py
    training_args = TrainingArguments(
        output_dir=f"./continual_fine_tuned_model_task_{i+1}",
        num_train_epochs=8,                # Increased epochs
        per_device_train_batch_size=20,    # Increased batch size
        per_device_eval_batch_size=20,
        warmup_steps=2000,                 # Increased warmup
        weight_decay=0.2,                  # Increased weight decay
        learning_rate=2e-5,                # Added learning rate
        logging_dir="./logs",
        evaluation_strategy="steps",    # Added evaluation strategy
        eval_steps=1000,                # Added evaluation frequency
        save_steps=1000                    # Added save frequency
    )
    ```

此实现引入了持续微调的几个关键概念：

+   **EWC**：我们实现 EWC 的简化版本，它将惩罚项添加到损失函数中，以防止对先前任务的重要参数产生剧烈变化

+   **重要性计算**：我们根据先前任务上梯度的幅度计算每个参数的重要性

+   **持续微调循环**：我们按顺序在每个任务上微调模型，使用 EWC 来减轻遗忘

+   **对所有任务的评估**：在针对每个新任务进行微调后，我们评估模型在所有先前任务上的性能以监控遗忘

持续微调的关键考虑因素如下：

+   **可塑性和稳定性之间的平衡**：EWC 有助于维持这种平衡，使模型能够在学习新任务的同时保留对先前任务的知识

+   **计算开销**：计算重要性和应用 EWC 增加了训练的计算成本

+   **任务相似性**：持续微调的有效性可能取决于任务之间的相似性

考虑到缓解灾难性遗忘的策略，以下是一些额外的策略：

+   **梯度周期性记忆（GEM）**：在此方法中，存储并使用来自先前任务的小周期性数据记忆来约束新任务上的梯度更新，如下所示：

    ```py
    def project(gradient, memories):
        for memory in memories:
            if torch.dot(gradient, memory) < 0:
                gradient -= (
                    torch.dot(gradient, memory) / torch.dot(
                        memory, memory)
                ) * memory
        return gradient
    # This would be integrated into the training loop
    ```

+   **渐进式神经网络**：在这里，为每个新任务创建一个新的“列”层，同时保持到先前学习特征的横向连接。

+   **无遗忘学习（LwF）**：在此方法中，采用知识蒸馏来保留模型在先前任务上的性能：

    ```py
    def lwf_loss(
        model, old_model, new_data, old_data, temperature=2
    ):
        # Compute standard loss on new data
        new_loss = compute_loss(model, new_data)
        # Compute distillation loss on old data
        old_outputs = old_model(old_data)
        new_outputs = model(old_data)
        distillation_loss = F.kl_div(
            F.log_softmax(new_outputs / temperature, dim=1),
            F.softmax(old_outputs / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature  2)
        return new_loss + distillation_loss
    # This would replace the standard loss in the training loop
    ```

这些高级技术当在多样化的任务或领域微调 LLM 时特别有用。

# 摘要

对于大型语言模型（LLM）的微调模式包括一系列技术，从基本的迁移学习到高级的持续学习策略。通过掌握这些模式，您可以有效地将预训练模型适应于新任务和领域，优化性能，并缓解灾难性遗忘等问题。随着 LLM 领域的持续发展，跟上最新的微调技术对于开发针对特定应用的尖端语言模型至关重要。

本章的关键要点如下：

+   **微调适应预训练 LLM**：微调是将通用、预训练的 LLM 适应于特定任务和数据集的关键过程，它弥合了通用语言理解与特定性能之间的差距

+   **层管理至关重要**：战略性地冻结和解冻层（尤其是逐步解冻）对于在保留预训练知识与新任务适应之间取得平衡至关重要

+   **学习率调度稳定训练**：使用带有预热（线性或余弦）的学习率调度对于稳定和有效的微调至关重要，可以防止剧烈的早期更新并促进收敛

+   **领域/任务特定性很重要**：诸如领域特定词汇适应、自定义数据处理以及少样本/零样本方法等技术对于在特定任务上最大化性能至关重要

+   **必须解决灾难性遗忘问题**：在持续学习场景中，EWC、GEM 等技术在训练新任务时防止模型丢失先前学习的信息是必要的

我们将在下一章探讨模型剪枝。模型剪枝系统地从 LLM 中移除冗余或不太重要的神经连接，同时保留核心功能，本质上创建了一个更轻、更高效的版本，它保持了相似的性能但需要更少的计算资源。
