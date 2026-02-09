# 8

# 超参数调整

在本章中，你将了解 LLM 中的超参数以及优化它们的策略。我们将探讨手动和自动调整方法，包括网格搜索、随机搜索以及更高级的方法，如贝叶斯优化和基于群体的训练。你还将深入了解在 LLM 开发中常见的多目标优化场景的处理。

到最后，你将掌握实用的工具和技术，以优化你的 LLM 在各种任务和领域中的表现。

在本章中，我们将涵盖以下主题：

+   理解超参数

+   手动与自动调整

+   网格和随机搜索

+   贝叶斯优化

+   基于群体的方法

+   多目标超参数优化

+   规模化超参数调整——挑战与解决方案

# 理解超参数

超参数是在机器学习训练过程开始之前设置的参数，它们不是从数据中学习的。它们控制学习算法本身的各个方面，例如模型的复杂性、学习速率以及整体训练过程。数据科学家手动选择和调整这些超参数以优化模型的表现。

LLM 中的超参数可以大致分为三类：架构、优化和正则化超参数：

+   **架构超参数**：这些定义了模型的设计和结构，决定了模型如何处理和表示数据。它们至关重要，因为它们直接影响模型学习数据中的复杂模式和关系的能力。正确的架构在计算效率和性能之间取得平衡，使模型能够很好地泛化到未见数据。

    本类别中的参数包括以下内容：

    +   层数数量

    +   隐藏层大小

    +   注意力头数

    +   前馈维度

    +   词汇量大小

+   **优化超参数**：这些通过调整参数以最小化损失函数来控制模型在训练过程中的学习方式。它们很重要，因为它们控制更新速率和方式，影响收敛速度、稳定性和模型达到最优解的能力。适当的调整确保了高效的训练，避免了发散或欠拟合。

    本类别中的参数包括以下内容（我们在*第七章*中讨论过）：

    +   学习速率

    +   批处理大小

    +   训练步数

    +   预热步骤

    +   学习速率调度

+   **正则化超参数**：这些引入机制以防止模型过度拟合训练数据，确保其泛化到新数据。它们至关重要，因为高容量模型可以轻易记住训练数据，导致在未见数据上的表现不佳。正则化技术强制执行约束，鼓励简单性和鲁棒性。

    此类别中的参数包括以下内容（更多内容请参阅*第九章*）：

    +   Dropout 率

    +   权重衰减

    +   标签平滑

让我们实现一个函数来创建具有可配置超参数的 LLM：

```py
from transformers import GPT2Config, GPT2LMHeadModel
def create_llm(
    num_layers, hidden_size, num_heads, ff_dim, vocab_size
):
    config = GPT2Config(
        n_layer=num_layers,
        n_embd=hidden_size,
        n_head=num_heads,
        n_inner=ff_dim,
        vocab_size=vocab_size
    )
    model = GPT2LMHeadModel(config)
    return model
# Example usage
model = create_llm(num_layers=12, hidden_size=768,
    num_heads=12, ff_dim=3072, vocab_size=50257)
print(f"Model parameters: {model.num_parameters():,}")
```

在此代码中，我们定义了一个函数`create_llm`，它允许我们轻松地创建具有不同架构超参数的 LLMs。该函数接受以下参数：

+   `num_layers`：模型中 transformer 层的数量。更多的层可以捕捉更复杂的模式，但它们会增加计算需求。

+   `hidden_size`：模型中隐藏状态的维度。这影响模型捕捉信息的能力。

+   `num_heads`：每个层中注意力头的数量。多个头允许模型同时关注输入的不同方面。

+   `ff_dim`：每个 transformer 块中前馈层的维度。这通常设置为`hidden_size`的四倍。

+   `vocab_size`：模型词汇表的大小。这决定了嵌入层和输出层的大小。

我们使用这些参数来创建一个`GPT2Config`对象，然后使用该对象初始化一个`GPT2LMHeadModel`。这种方法允许我们轻松地实验不同的模型架构。

# 手动与自动化调整

**手动调整**涉及根据直觉、经验和逐步实验调整超参数。手动调整允许您利用领域知识系统地探索定制配置，但它是时间密集型的，容易产生次优结果，并且在探索大超参数空间方面效率低下。

**自动化调整**，另一方面，使用算法系统地探索超参数空间。自动化调整通过算法优化性能，有效地探索大超参数空间，与手动调整相比，可以节省时间和精力，但可能计算成本较高，并且可能需要专业知识来正确配置。

当领域知识或直觉可以引导一个小型、有针对性的搜索空间时，手动调整很有用，尤其是在资源受限的环境或简单的模型中。自动化调整更适合大型、复杂的超参数空间，因为需要系统探索和优化，尽管计算成本较高，但它可以更有效地找到更好的配置。

让我们实现两种方法。

## 手动调整

首先，我们将实现手动调整：

1.  开始导入：

    ```py
    import numpy as np
    from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    ```

1.  加载一个样本数据集：

    ```py
    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=512)
    tokenized_dataset = dataset.map(tokenize_function,
        batched=True, remove_columns=dataset.column_names)
    ```

1.  设置手动调整的超参数：

    ```py
    manual_hyperparameters = [
        {"num_layers": 6, "hidden_size": 512, "num_heads": 8, "ff_dim": 2048},
        {"num_layers": 12, "hidden_size": 768, "num_heads": 12, "ff_dim": 3072},
        {"num_layers": 24, "hidden_size": 1024, "num_heads": 16, "ff_dim": 4096}
    ]
    ```

1.  使用`manual_hyperparameters`进行训练：

    ```py
    for hp in manual_hyperparameters:
        model = create_llm(hp, vocab_size=50257)
        training_args = TrainingArguments(
            output_dir=(
                f"./results_{hp['num_layers']}_"
                f"{hp['hidden_size']}"
            ),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir=(
                f"./logs_{hp['num_layers']}_"
                f"{hp['hidden_size']}"
            ),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
    ```

1.  评估模型：

    ```py
        eval_results = trainer.evaluate()
        print(f"Hyperparameters: {hp}")
        print(f"Evaluation results: {eval_results}")
    ```

在这个手动调整示例中，我们定义了一个要尝试的超参数配置列表。然后我们遍历这些配置，为每个配置创建一个模型，对其进行训练，并评估其性能。这种方法允许我们系统地探索不同的模型大小和架构。

手动调优过程可以由领域知识和直觉指导。例如，我们可能从一个小的模型（6 层，512 隐藏大小）开始，逐渐增加大小以查看它如何影响性能。我们选择这些特定的配置是基于基于转换器模型的常见实践：

+   最小配置（6 层，512 隐藏大小）代表一个紧凑的模型，适合快速训练和部署

+   中等配置（12 层，768 隐藏大小）与已知的在许多任务上表现良好的基础 GPT-2 模型相似

+   最大配置（24 层，1,024 隐藏大小）代表一个更强大的模型，可能能够捕捉更复杂的模式，但需要更多的计算资源

## 自动调优

现在，让我们使用随机搜索实现一个简单的自动化调优方法（我们将在下一节中展示更高级的随机搜索）：

1.  添加`import`语句并设置随机参数：

    ```py
    import random
    def random_hp_search(num_trials=10):
        best_eval_loss = float('inf')
        best_hp = None
        for _ in range(num_trials):
            hp = {
                "num_layers": random.choice([6, 12, 24]),
                "hidden_size": random.choice([512, 768, 1024]),
                "num_heads": random.choice([8, 12, 16]),
                "ff_dim": random.choice([2048, 3072, 4096])
            }
    ```

1.  进行训练：

    ```py
            model = create_llm(hp, vocab_size=50257)
            training_args = TrainingArguments(
                output_dir=f"./results_random_{_}",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                logging_dir=f"./logs_random_{_}",
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            trainer.train()
    ```

1.  评估并打印结果：

    ```py
            eval_results = trainer.evaluate()
            eval_loss = eval_results['eval_loss']
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_hp = hp
            print(
                f"Trial {_ + 1}: "
                f"Hyperparameters: {hp}, "
                f"Eval Loss: {eval_loss}"
            )
        print(
            f"Best Hyperparameters: {best_hp}, "
            f"Best Eval Loss: {best_eval_loss}"
        )
    random_hp_search()
    ```

这种随机搜索实现从预定义的选项中随机选择每个试验的超参数（试验期间没有手动干预）。预定义选项是指搜索过程中从指定的范围、集合或分布中抽取超参数随机值的指定范围、集合或分布。例如，离散超参数（如层数）可能从集合`[6, 12, 24]`中选择，而连续超参数（如学习率）可能从均匀分布或对数均匀分布中抽取，例如从`10^(-5)`到`10^(-3)`。这些选项定义了每个超参数的边界和可能值，指导随机抽样过程。

我们选择在每个超参数上搜索一组离散值以限制搜索空间，并确保我们正在探索已知对转换器模型表现良好的配置。试验次数（本例中为 10 次）是探索和计算资源之间的平衡。更多的试验增加了找到良好配置的机会，但也增加了计算成本。

在接下来的章节中，我们将介绍其他自动化调优技术，例如网格搜索和更高级的随机搜索、贝叶斯优化、基于群体的方法和多目标超参数优化

# 网格搜索和随机搜索

**网格搜索**和**随机搜索**是两种常见的超参数调优方法。我们在上一节中介绍了随机搜索。在本节中，我们实现网格搜索和更高级的随机搜索版本。

1.  添加导入并设置网格搜索参数：

    ```py
    import itertools
    def grid_search():
        hp_grid = {
            "num_layers": [6, 12, 24],
            "hidden_size": [512, 768, 1024],
            "num_heads": [8, 12, 16],
            "ff_dim": [2048, 3072, 4096]
        }
        best_eval_loss = float('inf')
        best_hp = None
    ```

1.  使用定义的超参数训练模型：

    ```py
    for hp in itertools.product(*hp_grid.values()):
            hp_dict = dict(zip(hp_grid.keys(),hp))
            model = create_llm(
                hp_dict["num_layers"],
                hp_dict["hidden_size"],
                hp_dict["num_heads"],
                hp_dict["ff_dim"],
                vocab_size=50257
            )
            training_args = TrainingArguments(
                output_dir=(
                    f"./results_grid_{hp_dict['num_layers']}_"
                    f"{hp_dict['hidden_size']}"
                ),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                logging_dir=(
                    f"./logs_grid_{hp_dict['num_layers']}_"
                    f"{hp_dict['hidden_size']}"
                ),
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            trainer.train()
    ```

1.  评估并打印结果：

    ```py
            eval_results = trainer.evaluate()
            eval_loss = eval_results['eval_loss']
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_hp = hp_dict
            print(
                f"Hyperparameters: {hp_dict}, "
                f"Eval Loss: {eval_loss}"
            )
        print(
            f"Best Hyperparameters: {best_hp}, "
            f"Best Eval Loss: {best_eval_loss}"
        )
    grid_search()
    ```

网格搜索全面探索所有超参数组合。这种方法非常彻底，但可能计算成本高昂，尤其是对于具有许多超参数的 LLM。在本实现中，我们正在探索`3⁴` = `81`种不同的配置，这可能需要大量的时间和资源。

超参数范围的选择旨在覆盖合理的模型大小空间，从相对较小（6 层，512 个隐藏层大小）到相当大（24 层，1,024 个隐藏层大小）。这使我们能够探索模型大小和性能之间的权衡。

现在，让我们实现一个更复杂的随机搜索，它还包括优化超参数：

1.  添加`import`语句并设置`advanced_random_search`超参数：

    ```py
    import random
    def advanced_random_search(num_trials=20):
        best_eval_loss = float('inf')
        best_hp = None
        for _ in range(num_trials):
            hp = {
                "num_layers": random.choice([6, 12, 24]),
                "hidden_size": random.choice([512, 768, 1024]),
                "num_heads": random.choice([8, 12, 16]),
                "ff_dim": random.choice([2048, 3072, 4096]),
                "learning_rate": 10random.uniform(-5, -3),
                "batch_size": random.choice([8, 16, 32]),
                "num_epochs": random.randint(2, 5),
                "warmup_steps": random.randint(100, 1000),
                "weight_decay": random.uniform(0, 0.2)
            }
    ```

1.  进行训练：

    ```py
            model = create_llm(
                num_layers=hp['num_layers'],
                    hidden_size=hp['hidden_size'],
                num_heads=hp['num_heads'], ff_dim=hp['ff_dim'],
                    vocab_size=50257)
            training_args = TrainingArguments(
                output_dir=f"./results_advanced_random_{_}",
                num_train_epochs=hp['num_epochs'],
                per_device_train_batch_size=hp['batch_size'],
                learning_rate=hp['learning_rate'],
                warmup_steps=hp['warmup_steps'],
                weight_decay=hp['weight_decay'],
                logging_dir=f"./logs_advanced_random_{_}",
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            trainer.train()
    ```

1.  评估并打印结果：

    ```py
            eval_results = trainer.evaluate()
            eval_loss = eval_results['eval_loss']
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_hp = hp
            print(
                f"Trial {_ + 1}: Hyperparameters: {hp}, "
                f"Eval Loss: {eval_loss}"
            )

        print(
            f"Best Hyperparameters: {best_hp}, "
            f"Best Eval Loss: {best_eval_loss}"
        )
    ```

这种高级随机搜索包括架构和优化超参数。我们使用`random.uniform`来设置`0.001`、`0.0015`或`0.002`等，例如学习率和权重衰减，以及`random.choice`或`random.randint`来设置`32`、`64`和`128`作为批量大小或从一组固定的选项中进行选择）。

每个超参数的范围是基于 LLM 训练中的常见做法选择的（也请参阅*第七章*）：

+   `1e-5`和`1e-3`，因为 LLM 的学习率通常在这个范围内

+   `8`、`16`和`32`，这些是常见的批量大小，在计算效率和稳定性之间取得平衡

+   `2`到`5`个周期，因为 LLM 通常在大数据集上只需要几个周期就会收敛

+   `100`步和`1,000`步，这有助于稳定早期训练

+   `0`和`0.2`，因为少量的权重衰减可以帮助防止过拟合

高级随机搜索比网格搜索更好，因为它通过随机采样而不是全面评估每个可能的组合来更有效地探索超参数空间。这种灵活性允许它专注于对性能有显著影响的键超参数，防止对影响较小的超参数进行冗余评估。它可以直接通过从分布中采样来处理连续参数，而网格搜索则需要离散化，并且随着参数空间的增加，计算成本呈指数增长。通过将试验次数限制在预定义的预算内，高级随机搜索可以更快地发现有效的配置，并且计算成本更低，使其更适合大型和复杂模型。

# 贝叶斯优化

**贝叶斯优化**是一种更高级的超参数调整方法，对于 LLM 特别有效。它使用概率模型来预测不同超参数配置的性能，并智能地选择下一个要尝试的配置。

让我们使用 `optuna` 库实现贝叶斯优化。**Optuna** 是一个开源的超参数优化框架，用于自动化寻找算法和模型最优参数的过程。它采用先进的贝叶斯优化技术，主要使用 **树结构帕累托估计器**（**TPE**）算法，以有效地搜索复杂的参数空间：

1.  导入 optuna 并设置超参数：

    ```py
    import optuna
    from transformers import Trainer, TrainingArguments
    import torch
    def objective(trial):
        # Define the hyperparameters to optimize
        hp = {
            "num_layers": trial.suggest_int("num_layers", 6, 24),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [512, 768, 1024]
            ,
            "num_heads": trial.suggest_categorical(
                "num_heads", [8, 12, 16]
            ),
            "ff_dim": trial.suggest_categorical(
                "ff_dim", [2048, 3072, 4096]
            ),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-5, 1e-3
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [8, 16, 32]
            ),
            "num_epochs": trial.suggest_int("num_epochs", 2, 5),
            "warmup_steps": trial.suggest_int(
            "warmup_steps", 100, 1000),
            "weight_decay": trial.suggest_uniform(        "weight_decay", 0, 0.2)
        }
        model = create_llm(
            num_layers=hp['num_layers'],
            hidden_size=hp['hidden_size'],
            num_heads=hp['num_heads'], ff_dim=hp['ff_dim'],
            vocab_size=50257
        )
    ```

1.  进行训练：

    ```py
        training_args = TrainingArguments(
            output_dir=f"./results_bayesian_{trial.number}",
            num_train_epochs=hp['num_epochs'],
            per_device_train_batch_size=hp['batch_size'],
            learning_rate=hp['learning_rate'],
            warmup_steps=hp['warmup_steps'],
            weight_decay=hp['weight_decay'],
            logging_dir=f"./logs_bayesian_{trial.number}",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results['eval_loss']
    ```

1.  运行优化：

    ```py
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    ```

在此实现中，我们定义了一个 `objective` 函数，Optuna 将对其进行优化。该函数使用 Optuna 建议的超参数创建和训练模型，然后返回评估损失。

我们使用 Optuna 的建议方法来定义搜索空间：

+   `suggest_int` 用于整数超参数，如 `num_layers` 和 `num_epochs`

+   `suggest_categorical` 用于具有离散选项的超参数，例如 `hidden_size` 和 `num_heads`

+   `suggest_loguniform` 用于学习率，因为我们想以对数方式搜索这个空间

+   `suggest_uniform` 用于权重衰减，因为我们想在这个空间内均匀搜索

每个超参数的范围与我们在基于随机搜索的实现中使用的范围类似，基于 LLM 训练中的常见实践。

贝叶斯优化可能比网格搜索或随机搜索更有效，尤其是在评估成本高昂的函数，如训练 LLM 时。它使用先前试验的结果来指导未来试验的选择，可能更快地找到好的配置。

# 基于群体的方法

**基于群体的训练**（**PBT**）是一种强大的技术，它将并行搜索与训练过程中的自适应超参数调整相结合。PBT 特别适用于可以高效暂停和恢复训练的问题。这是因为 PBT 定期评估和更新群体中的超参数和模型权重，需要无缝的暂停和恢复功能。这种适应性确保了计算资源的最佳利用，使 PBT 成为神经架构搜索、强化学习和超参数调整等任务的理想选择，在这些任务中，迭代优化计算密集。

在这里，我们将实现 PBT 的简化版本，以说明其核心概念和功能。

我们将首先创建一个 `SimplePBT` 类，它封装了 PBT 算法的核心功能。让我们分解实现过程：

1.  首先，初始化类：

    ```py
    import random
    import copy
    class SimplePBT:
        def __init__(self, population_size=4, num_generations=5):
            self.population_size = population_size
            self.num_generations = num_generations
            self.population = []
    ```

    `SimplePBT` 类使用两个主要参数进行初始化：

    +   `population_size`: 维护的不同超参数配置的数量（默认为 `4`）

    +   `num_generations`: PBT 算法将运行的迭代次数（默认为 `5`）

    `population` 列表将存储代表群体中每个个体的字典，包含超参数及其相应的性能分数。

1.  初始化人口：`initialize_population` 方法创建初始的超参数配置集：

    ```py
    def initialize_population(self):
        for _ in range(self.population_size):
            hp = {
                "num_layers": random.choice([6, 12, 24]),
                "hidden_size": random.choice([512, 768, 1024]),
                "num_heads": random.choice([8, 12, 16]),
                "ff_dim": random.choice([2048, 3072, 4096]),
                "learning_rate": 10random.uniform(-5, -3),
                "batch_size": random.choice([8, 16, 32]),
                "weight_decay": random.uniform(0, 0.2)
            }
            self.population.append({"hp": hp, "score": None})
    ```

    对于种群中的每个个体，执行以下操作：

    +   `num_layers` 和 `hidden_size` 从预定义选项中随机选择。这些超参数是分类的，因为它们代表的是离散的、个体的选择，而不是连续值。

    +   `learning_rate` 和 `weight_decay` 从指定的范围内采样。

    每个配置都添加到 `population` 列表中，初始得分为 `None`。

1.  训练和评估：`train_and_evaluate` 方法负责创建具有给定超参数的 LLM，设置训练参数，使用模型和参数初始化训练器，训练模型，评估模型，并返回评估损失：

    ```py
    def train_and_evaluate(self, hp):
        model = create_llm(num_layers=hp['num_layers'],
            hidden_size=hp['hidden_size'],
            num_heads=hp['num_heads'],
            ff_dim=hp['ff_dim'], vocab_size=50257)
        training_args = TrainingArguments(
            output_dir=f"./results_pbt_{random.randint(0, 1000)}",
            num_train_epochs=3,
            per_device_train_batch_size=hp['batch_size'],
            learning_rate=hp['learning_rate'],
            weight_decay=hp['weight_decay'],
            logging_dir=f"./logs_pbt_{random.randint(0, 1000)}",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
        eval_results = trainer.evaluate()
        return eval_results['eval_loss']
    ```

    此方法假设存在 `create_llm`、`TrainingArguments` 和 `Trainer` 类，这些类通常由深度学习框架（如 Hugging Face Transformers）提供。

1.  利用和探索：`exploit_and_explore` 方法实现了核心的 PBT 算法：

    ```py
    def exploit_and_explore(self):
        # Sort population by score
        self.population.sort(key=lambda x: x['score'])
        # Replace bottom half with mutated versions of top half
        for i in range(self.population_size // 2):
            self.population[i + self.population_size // 2]['hp'] =\
                self.mutate(
                    copy.deepcopy(self.population[i]['hp'])
                )
    ```

    它根据得分对人口进行排序（得分越低表示损失越小）。表现最差的种群下半部分被表现最好的种群变异版本所取代。这种方法平衡了 `mutate` 方法在超参数中引入的变化：

    ```py
    def mutate(self, hp):
        # Randomly mutate one hyperparameter
        param_to_mutate = random.choice(list(hp.keys()))
        if param_to_mutate in [
            'num_layers', 'hidden_size', 'num_heads', 'ff_dim',
            'batch_size'
        ]:
            hp[param_to_mutate] = random.choice(
                [6, 12, 24] 
                if param_to_mutate == "num_layers" else
                [512, 768, 1024] 
                if param_to_mutate == "hidden_size" else
                [8, 12, 16] 
                if param_to_mutate == "num_heads" else
                [2048, 3072, 4096] 
                if param_to_mutate == "ff_dim" else
                [8, 16, 32]
            )
        elif param_to_mutate == 'learning_rate':
            hp[param_to_mutate] *= random.uniform(0.8, 1.2)
        elif param_to_mutate == 'weight_decay':
            hp[param_to_mutate] = min(
                max(hp[param_to_mutate]
                    + random.uniform(-0.05, 0.05), 0), 0.2
                )
        return hp
    ```

    它随机选择一个超参数进行变异。对于分类参数，它从预定义选项中选择一个新值。对于像学习率这样的连续参数，它在一定范围内扰动当前值。对于权重衰减，它在保持其在 `[0, 0.2]` 范围内的同时添加一个小的随机值。

    这种变异策略允许对超参数进行小到大的变化，从而促进对超参数空间的多样化探索。

1.  运行 PBT 过程：

    ```py
    def run(self):
        self.initialize_population()
        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}")
            for i, individual in enumerate(self.population):
                individual['score'] = \
                    self.train_and_evaluate(individual['hp'])
                print(
                    f"Individual {i + 1}:
                    Score = {individual['score']}"
                )
            self.exploit_and_explore()
        best_individual = min(self.population,
            key=lambda x: x['score'])
        print("\nBest Hyperparameters:")
        print(best_individual['hp'])
        print(f"Best Score: {best_individual['score']}")
    ```

    `run` 方法协调整个 PBT 过程：

    1.  它初始化人口。

    1.  对于每一代，它训练和评估种群中的每个个体，并执行利用和探索以更新种群。

    1.  在所有代数完成后，它打印出找到的最佳超参数和得分。

1.  使用 `SimplePBT` 类：要使用 `SimplePBT` 类，您可以简单地创建一个实例并运行它：

    ```py
    # Run PBT
    pbt = SimplePBT()
    pbt.run()
    ```

这将启动 PBT 过程，默认人口大小为 `4` 和 `5` 代。您可以在创建 `SimplePBT` 实例时调整这些参数以适应您的特定需求。

# 多目标超参数优化

在 LLM 开发中，我们经常需要平衡多个目标，例如模型性能、推理速度和模型大小。让我们使用 Optuna 实现多目标优化：

1.  添加 `import` 语句并设置超参数：

    ```py
    import optuna
    def objective(trial):
        hp = {
            "num_layers": trial.suggest_int("num_layers", 6, 24),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [512, 768, 1024]),
            "num_heads": trial.suggest_categorical(
                "num_heads", [8, 12, 16]),
            "ff_dim": trial.suggest_categorical(
                "ff_dim", [2048, 3072, 4096]),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical(
                "batch_size", [8, 16, 32]),
            "weight_decay": trial.suggest_uniform(
                "weight_decay", 0, 0.2)
        }
        model = create_llm(
            num_layers=hp['num_layers'],
            hidden_size=hp['hidden_size'],
            num_heads=hp['num_heads'],
            ff_dim=hp['ff_dim'],
            vocab_size=50257
        )
    ```

1.  进行训练：

    ```py
        training_args = TrainingArguments(
            output_dir=f"./results_multi_objective_{trial.number}",
            num_train_epochs=3,
            per_device_train_batch_size=hp['batch_size'],
            learning_rate=hp['learning_rate'],
            weight_decay=hp['weight_decay'],
            logging_dir=f"./logs_multi_objective_{trial.number}",
        )
           trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
    ```

1.  执行评估：

    ```py
        eval_results = trainer.evaluate()
        eval_loss = eval_results['eval_loss']
        # Calculate model size in MB
        model_size = sum(p.numel() for p in model.parameters())
            * 4 / 1024 / 1024  # assuming float32
        # Simulate inference time (this would be more accurate if actually measured)
        inference_time = 0.001 * hp['num_layers']
            * (hp['hidden_size'] / 512)  2
        return eval_loss, model_size, inference_time
    ```

1.  运行多目标优化：

    ```py
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=50)
    print("Pareto front:")
    for trial in study.best_trials:
        print(f"Trial {trial.number}")
        print(f"  Value: Loss={trial.values[0]:.4f},
            Size={trial.values[1]:.2f}MB,
            Inference Time={trial.values[2]:.4f}s")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    ```

在这个多目标优化中，我们试图同时最小化三个目标：

+   评估损失（模型性能）

+   模型大小（以 MB 计）。

+   推理时间（基于模型架构模拟）

我们通过在`create_study`中指定多个方向来使用 Optuna 的多目标优化能力。优化过程将尝试找到这些目标配置的**帕累托前沿**（任何改进一个目标都需要至少降低另一个目标的一组解决方案），其中提高一个目标必然会导致另一个目标恶化。

现在`objective`函数返回三个值，对应我们的三个目标。对于模型大小，我们计算参数总数并将其转换为 MB。对于推理时间，我们使用基于模型架构的简单启发式方法，在实际场景中，你可能会想测量这个值。

这种方法使我们能够探索模型性能、大小和速度之间的权衡。对于 LLM 开发尤其有用，因为我们经常需要在不同部署场景中平衡这些因素。

# 超参数调整的规模——挑战和解决方案

当调整 LLM 的超参数时，我们面临几个挑战：

+   **计算成本**：训练 LLM 很昂贵，限制了我们可以运行的试验数量

+   **长训练时间**：每个试验可能需要几天或几周，使整个过程非常耗时

+   **大搜索空间**：LLM 有很多超参数，创建了一个庞大的搜索空间

+   **对初始化的敏感性**：LLM 的性能可能因不同的随机种子而有很大差异

为了应对这些挑战，我们可以采用几种策略：

+   **使用较小的代理任务**：不是在完整任务上调整，而是使用较小的数据集或更少的训练步骤来快速估计性能

+   **利用预训练模型**：从预训练权重开始，专注于调整微调超参数

+   **使用多保真度优化**：从低保真度评估（例如，少量训练步骤）开始，并逐渐增加有希望配置的保真度

+   **分布式超参数调整**：使用多台机器并行探索不同的超参数

让我们实现一个简单的多保真度优化方法：

1.  添加`import`语句并设置超参数：

    ```py
    import optuna
    def objective(trial):
        hp = {
            "num_layers": trial.suggest_int("num_layers", 6, 24),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [512, 768, 1024]),
            "num_heads": trial.suggest_categorical(
                "num_heads", [8, 12, 16]),
            "ff_dim": trial.suggest_categorical(
                "ff_dim", [2048, 3072, 4096]),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical(
                "batch_size", [8, 16, 32]),
            "weight_decay": trial.suggest_uniform(
                "weight_decay", 0, 0.2)
        }
        model = create_llm(
            num_layers=hp['num_layers'],
            hidden_size=hp['hidden_size'],
            num_heads=hp['num_heads'], ff_dim=hp['ff_dim'],
            vocab_size=50257)
    ```

1.  使用多保真度策略进行训练，从少量步骤开始：

    ```py
        for steps in [100, 500, 2000]:
            training_args = TrainingArguments(
                output_dir= \
                    f"./results_multi_fidelity_{trial.number}_
                    {steps}",
                max_steps=steps,
                per_device_train_batch_size=hp['batch_size'],
                learning_rate=hp['learning_rate'],
                weight_decay=hp['weight_decay'],
                logging_dir=\
                    f"./logs_multi_fidelity_{trial.number}_{steps}",
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            trainer.train()
    ```

1.  进行评估：

    ```py
            eval_results = trainer.evaluate()
            eval_loss =
            eval_results = trainer.evaluate()
            eval_loss = eval_results['eval_loss']
            trial.report(eval_loss, step=steps)
    ```

1.  剪枝没有希望的试验：

    ```py
            if trial.should_prune():
                raise optuna.TrialPruned()
        return eval_loss
    ```

1.  运行多保真度优化：

    ```py
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    ```

我们应该关注这个多保真度方法的一些方面：

+   我们首先只对每个模型配置进行`100`步的训练，这给出了性能的快速初始估计

+   然后，我们将训练步骤的数量增加到`500`，然后对于有希望的配置增加到`2,000`

+   我们使用 Optuna 的剪枝机制来提前终止没有希望的试验，节省计算资源

`MedianPruner`如果试验的性能低于同一步骤之前试验的中位数，则停止试验。这使我们能够将计算资源集中在最有希望的参数配置上。

这种方法有助于解决大规模超参数调整的挑战：

+   它通过快速消除不良配置来降低计算成本

+   它通过使用较短的训练运行进行初始评估来缩短整体调整时间

+   它通过在相同的时间内运行更多试验来允许我们探索更大的搜索空间

然而，这种方法仍然存在局限性。经过少量步骤后的性能可能并不总是与最终性能很好地相关，特别是对于需要长时间训练才能收敛的 LLMs（大型语言模型）。

为了进一步改进大规模超参数调整，考虑以下高级技术：

+   **分布式超参数调整**：这种设置允许多台机器共同参与同一超参数搜索，大大加快了过程：

    ```py
    import optuna
    def objective(trial):
        # ... (same as before) ...
    # Create a study object with MySQL storage for distributed optimization
    storage = optuna.storages.RDBStorage(
        "mysql://user:password@host/database",
        engine_kwargs={"pool_size": 20, "max_overflow": 0}
    )
    study = optuna.create_study(
        storage=storage, pruner=optuna.pruners.MedianPruner())
    # This can be run on multiple machines
    study.optimize(objective, n_trials=10)
    ```

+   **利用预训练模型**：这种方法从预训练模型开始，专注于调整微调的超参数和模型大小，这比从头开始训练更有效率：

    ```py
    from transformers import AutoModelForCausalLM, AutoTokenizer
    def create_pretrained_llm(model_name, num_layers=None):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if num_layers is not None:
            # Adjust the number of layers (this is a simplified approach)
            model.transformer.h = model.transformer.h[:num_layers]
        return model
    def objective(trial):
        hp = {
            "model_name": trial.suggest_categorical(
                "model_name",
                ["gpt2", "gpt2-medium", "gpt2-large"]),
            "num_layers": trial.suggest_int("num_layers", 6, 24),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical(
                "batch_size", [8, 16, 32]),
            "weight_decay": trial.suggest_uniform(
                "weight_decay", 0, 0.2)
        }
        model = create_pretrained_llm(
            hp['model_name'], hp['num_layers'])
        # ... (rest of the objective function) ...
    study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)
    ```

+   **基于高斯过程的贝叶斯优化**：对于只能进行少量试验的问题，基于高斯过程的贝叶斯优化比基于树的 TPE（Optuna 的默认方法）等方法更具有样本效率：

    ```py
    import optuna
    sampler = optuna.samplers.GPSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=50)
    ```

    这种方法对于 LLM 调整特别有用，因为每次试验的成本都非常高。

+   **异步逐次减半算法**（**ASHA**）：ASHA 是一种基于 bandit 的算法，其效率可能高于简单的剪枝方法：

    ```py
    from optuna.pruners import SuccessiveHalvingPruner
    pruner = SuccessiveHalvingPruner(
        min_resource=100, reduction_factor=3,
            min_early_stopping_rate=0)
    study = optuna.create_study(pruner=pruner)
    study.optimize(objective, n_trials=100)
    ```

    ASHA 特别适合大规模超参数优化，因为它可以有效地处理异步并行优化。

# 摘要

由于 LLMs 的规模和复杂性，超参数调整面临着独特的挑战。通过利用多保真优化、分布式调整和贝叶斯优化、ASHA 等高级算法，我们可以使这个过程更加高效和有效。然而，重要的是要记住，通常没有一种适合所有情况的解决方案，最佳方法可能取决于您的具体用例、可用资源和您的 LLM 任务的特性。

在下一章中，我们将重点关注 LLM 正则化。
