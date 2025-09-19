

# 微调生成模型以适应特定任务

在我们的StyleSprint叙事中，我们描述了使用预训练的生成AI模型来创建引人入胜的产品描述。虽然这个模型在生成多样化内容方面表现出色，但StyleSprint不断变化的需求要求我们转变关注点。新的挑战不仅在于产生内容，还在于参与特定、面向任务的交互，例如自动回答客户对所描述产品的具体问题。

在本章中，我们介绍了微调的概念，这是将预训练模型适应特定下游任务的关键步骤。对于StyleSprint来说，这意味着将模型从多才多艺的内容生成器转变为能够提供准确和详细回答客户问题的专业工具。

我们将探索和定义一系列可扩展的微调技术，并将它们与其他方法如上下文学习进行比较。我们将展示高级微调方法，包括参数高效的微调和提示微调，以展示它们如何能够微调模型在特定任务如问答上的能力。

到本章结束时，我们将训练一个语言模型来回答问题，并以符合StyleSprint品牌指南的方式回答。然而，在我们探讨微调的机制及其在我们应用中的重要性之前，我们将回顾LLMs背景下微调的历史。

# 基础和相关性——微调简介

微调是将在大数据集上预训练的模型继续在较小、特定任务的数据集上进行训练的过程，以提高其在该任务上的性能。它还可能涉及适应新领域细微差别的额外训练。后者被称为领域自适应，我们将在[*第6章*](B21773_06.xhtml#_idTextAnchor211)中介绍。前者通常被称为特定任务微调，它可以执行多个任务，包括问答、摘要、分类等。对于本章，我们将专注于特定任务微调，以提高通用模型在回答问题时表现的能力。

对于StyleSprint来说，将模型微调以处理特定任务，如回答客户关于产品的询问，引入了独特的挑战。与主要涉及使用现成的预训练模型进行语言生成的产品描述不同，回答客户问题要求模型对产品特定数据有广泛的理解，并且应该有一个品牌意识的声音。具体来说，模型必须准确解释和回答有关产品功能、尺寸、可用性、用户评论和其他许多细节的问题。它还应该产生与StyleSprint独特的品牌语调一致的答案。这项任务需要从预训练中获得的一般自然语言能力以及关于产品元数据和客户反馈的稳健知识，这些是通过微调实现的。

模型如GPT最初通过无监督学习过程学习预测文本，这个过程涉及在广泛和庞大的数据集上进行训练。这个预训练阶段使模型接触到各种文本，使其能够获得对语言的广泛理解，包括语法、语法和上下文，而不需要任何特定任务导向的指导。然而，微调通过任务导向的监督学习来细化模型的能力以完成特定任务——具体来说，是半监督学习，正如Radford等人（2018年）所描述的，它涉及通过使模型接触到包含输入序列（`x1`，...，`xm`）和相应标签（`y`）的数据集来调整模型以适应特定的监督任务。

在本章中，我们将详细介绍微调过程，包括如何选择性地在产品相关信息和客户互动的精选数据集上训练模型，使其能够以客户期望的知情、品牌一致的方式做出回应。然而，微调具有数十亿参数的LLM通常需要大量的资源和时间。这就是高级技术如**参数高效微调**（**PEFT**）在使微调变得可访问方面变得特别有价值。

# PEFT

随着模型规模的增大，传统的微调方法变得越来越不切实际，因为训练和更新所有模型参数需要巨大的计算资源和时间。对于大多数企业，包括大型组织，传统的微调方法成本高昂，实际上无法启动。

另一种方法是仅修改模型参数的小子集，在保持最先进性能的同时减少计算负担。这种方法在将大型模型适应特定任务而不进行大量重新训练时具有优势。

有一种PEFT方法叫做**低秩自适应**（**LoRA**），由Hu等人（2021年）开发。

## LoRA

LoRA方法专注于选择性地微调Transformer架构中的特定组件，以增强LLMS的效率和有效性。LoRA针对Transformer的自注意力模块中发现的权重矩阵，正如在第[*3章*](B21773_03.xhtml#_idTextAnchor081)中讨论的那样，这些矩阵是其功能的关键，包括四个矩阵：wq（查询）、wk（键）、wv（值）和wo（输出）。尽管这些矩阵在多头注意力设置中可以被分为多个头——其中每个*头*代表几个并行注意力机制之一，这些机制独立处理输入——但LoRA将它们视为单个矩阵，简化了调整过程。

LoRA的方法仅涉及调整下游任务的注意力权重，而Transformer的另一个组件——**前馈网络**（**FFN**）中的权重保持不变。专注于仅调整注意力权重并冻结FFN的决定是为了简化并提高参数效率。通过这样做，LoRA确保了一个更易于管理和资源高效的微调过程，避免了重新训练整个网络的复杂性和需求。

这种选择性的微调策略使得LoRA能够有效地调整模型以适应特定任务，同时保持预训练模型的整体结构和优势。这使得LoRA成为将LLMs适应新任务的一种实用解决方案，减少了计算负担，无需在整个模型中进行全面的参数更新（刘等人，2021）。

建立在LoRA基础之上，**自适应低秩调整**（**AdaLoRA**），如刘等人（2022）在研究中介绍的那样，代表了PEFT方法的一个进一步发展。LoRA和AdaLoRA之间的关键区别在于（正如其名称所暗示的）其适应性。虽然LoRA在模型微调过程中应用了一致的低秩方法，但AdaLoRA则根据每一层的需要调整更新，提供了一种更灵活且可能更有效的微调大型模型以适应特定任务的方法。

## AdaLoRA

AdaLoRA的关键创新在于其对预训练模型权重矩阵中**参数预算**的自适应分配。许多PEFT方法倾向于将参数预算平均分配到所有预训练权重矩阵中，可能忽略了不同权重参数的不同重要性。AdaLoRA通过为这些权重矩阵分配重要性分数并相应地分配参数预算来克服这一点。在AdaLoRA的上下文中，**重要性分数**是用于确定模型中不同权重参数的重要性（或重要性）的指标，在微调期间更有效地指导参数预算的分配。

注意

**参数预算**是指在预训练模型微调过程中可以引入的额外参数数量的预定义限制。此预算的设置是为了确保模型的复杂性不会显著增加，这可能导致诸如过拟合、增加计算成本和更长的训练时间等挑战。

此外，AdaLoRA将**奇异值分解**（**SVD**）应用于在模型微调过程中进行的增量更新的有效组织。SVD允许有效地剪枝与不太重要的更新相关的奇异值，从而减少微调所需的总体参数预算。值得注意的是，这种方法还避免了需要计算密集型精确计算的需求，使得微调过程更加高效。

AdaLoRA已在包括自然语言处理、问答和自然语言生成在内的多个领域进行了实证测试。广泛的实验已经证明了它在提高模型性能方面的有效性，尤其是在问答任务中。AdaLoRA的适应性和效率使其成为需要精确和高效模型调整以完成复杂任务的理想选择。

在StyleSprint的情况下，AdaLoRA提供了一个机会，可以在不产生传统微调所必需的大量开销的情况下微调其语言模型以回答客户问题，传统微调需要调整所有模型参数。通过采用AdaLoRA，StyleSprint可以高效地通过调整显著较少的参数来适应其模型以处理细微的客户询问。具体来说，AdaLoRA对参数预算的适应性分配意味着StyleSprint可以在不使用大量计算资源的情况下优化其模型以适应客户查询的具体细微差别。

到本章结束时，我们将使用AdaLoRA微调一个LLM以完成我们的问答任务。然而，我们首先应该决定微调是否真的是正确的做法。基于提示的LLM提供了一个可行的替代方案，称为上下文学习，其中模型可以从提示中给出的示例中学习，这意味着提示将包含客户的提问以及一些关键的历史示例，说明其他问题的回答方式。模型可以从这些示例中推断出如何以与示例一致的方式回答当前的问题。在下一节中，我们将探讨上下文学习的利弊，以帮助我们确定微调是否是使模型能够回答非常具体问题的最佳方法。

# 在上下文学习

在上下文学习是一种技术，其中模型根据输入提示中提供的少量示例生成响应。这种方法利用了模型的预训练知识和提示中包含的特定上下文或示例，以执行任务而无需参数更新或重新训练。布朗等人（2020年）在《语言模型是少量学习者》中详细描述了这些模型的广泛预训练如何使它们能够根据嵌入在提示中的指令和少量示例执行任务和生成响应。与传统方法不同，传统方法需要对每个特定任务进行微调，上下文学习允许模型根据推理时提供的附加上下文进行适应和响应。

在上下文学习中，核心概念是少量提示，这对于使模型能够适应和执行任务而无需额外训练数据至关重要，而是依赖于它们预训练的知识和输入提示中提供的上下文。为了说明这一点，我们将描述一个LLM通常的工作方式，即零样本方法，并将其与使用少量方法的上下文学习进行对比：

+   `x`。模型计算潜在输出序列`y`的可能性，表示为`P(y|x)`。这种计算在没有特定于任务的先前示例的情况下进行，完全依赖于模型的通用预训练。这意味着零样本方法除了其通用知识外没有特定上下文。例如，如果我们问*Are winter coats available in children’s sizes?*，模型无法提供关于StyleSprint库存的具体答案。它只能提供一些通用的答案。

+   `x`)来形成一个扩展输入序列。因此，我们的问题*Are winter coats available in children’s sizes?*可能与以下示例配对：

    +   `Do you sell anything in` `children’s sizes?`

    `Any items for children are specifically listed on the “StyleSprint for` `Kids” page`.

    +   `What do you offer` `for kids?`

    `StyleSprint offers a variety of children’s fashions` `on its “StyleSprint for` `Kids” page`.

LLM随后计算在给定扩展输入序列`x`的情况下生成特定输出序列`y`的概率。从数学上讲，这可以理解为模型估计`y`和`x`（其中`x`包括之前演示的提示和少量示例）的联合概率分布。模型使用这个联合概率分布来生成与输入序列中给出的示例配对的指令一致的响应。

在这两种情况下，模型根据给定上下文调整其输出的能力，无论是零个示例还是少量示例，都展示了其底层架构和训练的灵活性和复杂性。然而，少量方法允许LLM从提供的非常具体的示例中学习。

让我们考虑 StyleSprint 如何应用情境学习来回答客户查询。使用情境学习（或少量示例方法）的性能始终显示出相对于零样本行为的显著提升（Brown 等人，2020 年）。我们可以将先前的例子扩展到客户询问特定产品的可用性。同样，StyleSprint 团队可以系统地在每个提示中添加几个示例，如下所示。

这里是提示：`请回答以下关于产品可用性的{问题}。`

这些是一些示例：

+   示例 1:

    +   `你携带黑色皮革手提包吗？`

    +   `请给我一点时间，我需要检索关于那个特定物品的信息。`

+   示例 2:

    +   `你有蓝色的丝绸围巾吗？`

    +   `让我在我们的库存中搜索蓝色丝绸围巾。`

StyleSprint 可以提供有效的示例，帮助模型理解查询的本质，并生成既具有信息性又符合公司政策和产品提供的信息响应。在这个例子中，我们看到响应旨在与搜索组件配对。这是一种常见的方法，可以使用称为 **检索增强生成**（**RAG**）的技术来实现，这是一个促进实时数据检索以告知生成响应的组件。将少量示例情境学习方法与 RAG 结合使用可以确保系统提供逻辑性和具体的答案。

使用少量示例的情境学习允许模型快速适应各种客户查询，同时使用有限数量的示例。当与 RAG 结合使用时，StyleSprint 可能能够满足其用例并减少微调所需的时间和资源。然而，这种方法必须权衡专业化的深度和特定任务微调的一致性，正如所描述的，这也可能产生高度准确且符合品牌语调的答案。

在下一节中，我们将制定有助于我们直接比较的指标，以指导 StyleSprint 做出最适合其客户服务目标和运营框架的明智决策。

# 微调与情境学习

我们了解到情境学习如何使StyleSprint的模型能够处理各种客户查询，而无需进行大量重新训练。具体来说，少量方法与RAG的结合可以促进对新查询的快速适应，因为模型可以根据几个示例生成响应。然而，情境学习的效果在很大程度上取决于提示中提供的示例的质量和相关性。其成功也取决于RAG的实施。此外，没有微调，响应可能缺乏一致性，或者可能不会严格遵循StyleSprint的品牌语气和客户服务政策。最后，完全依赖生成模型而不进行微调可能会无意中引入偏差，如第4章所述[*](B21773_04.xhtml#_idTextAnchor123)。

在实践中，我们有两种非常相似且可行的方案。然而，为了做出明智的决定，我们应首先使用定量方法进行更深入的比较。

为了公正地评估情境学习与微调相比的效力，我们可以衡量生成响应的质量和一致性。我们可以使用既定且可靠的指标来比较每种方法的结果。与之前的评估一样，我们希望应用以下关键维度的定量和定性方法：

+   **与人类判断的一致性**：我们可以再次应用语义相似性，以提供基于人类编写的参考答案的定量指标，衡量模型响应的正确性或相关性。

    StyleSprint的品牌传播专家可以审查一部分响应，以提供对响应准确性和与品牌语气和声音一致性的定性评估。

+   **一致性和稳定性**：重要的是要衡量每次回答问题时，尽管提问方式略有不同，但问题回答的一致程度。同样，当输入保持不变时，我们可以利用语义相似性来比较每个新输出与先前的输出。

除了评估每种方法的模型响应质量外，我们还可以直接比较每种方法所需的操作和计算开销。

对于微调，我们需要了解训练模型所涉及的开销。虽然PEFT方法将显著减少训练工作量，但与情境学习相比，可能存在相当多的基础设施相关成本，因为情境学习不需要额外的训练。另一方面，对于情境学习，如OpenAI的GPT-4这样的通用化模型有一个按令牌计费的成本模型。StyleSprint还必须考虑在提示中嵌入足够数量的少量示例所需的令牌成本。

在这两种情况下，StyleSprint将承担一些运营成本，以创建由人类编写的最佳示例，这些示例可以用作在少样本方法或额外模型训练中的“黄金标准”。

通过进行这些比较测试并分析结果，StyleSprint将获得宝贵的见解，了解哪种方法——情境学习或微调——最能与其运营目标和客户服务标准相匹配。这种数据驱动的评估将指导他们决定最佳的AI策略，以增强客户服务体验。我们将在接下来的实践项目中实施这些比较。

# 实践项目：使用PEFT进行问答微调

对于我们的实践项目，我们将尝试使用AdaLoRA高效地微调一个用于客户查询的模型，并将其与使用情境学习的**最先进**（**SOTA**）模型的输出直接比较。像上一章一样，我们可以依赖一个原型环境，如Google Colab，来完成两种方法的评估和比较。我们将展示如何配置模型训练以使用AdaLoRA作为我们的PEFT方法。

## 关于问答微调的背景

我们的项目利用了Hugging Face训练管道库，这是机器学习社区中广为人知的资源。这个库提供了各种预构建的管道，包括一个用于问答的管道，这使得我们能够以最小的设置微调预训练模型。Hugging Face管道抽象了模型训练中涉及的大部分复杂性，使得开发者能够直接且高效地实现高级自然语言处理任务。特别是，这个管道作为一个接口，连接到具有特定问答任务头的transformer模型。回想一下，当我们微调一个transformer模型时，我们保持模型的架构——包括自注意力机制和transformer层——但我们仅在特定任务上训练模型的参数，在这种情况下，结果是针对问答任务进行了优化的模型。回想一下我们在[*第三章*](B21773_03.xhtml#_idTextAnchor081)中的实践项目，其中生成的模型是一个翻译器；我们使用翻译器头来完成从英语到法语的语言翻译。对于这个项目，“头”被调整为学习问答数据中的模式。

然而，当使用问答训练管道时，重要的是要理解模型不仅仅是记住问答对，它还学习问题与答案之间的联系。此外，为了给出适当的答案，模型不能完全依赖训练。它还需要额外的上下文作为输入来组成一个相关的答案。为了进一步理解这一点，我们将模型推理步骤分解如下：

1.  当向模型输入问题时，我们还必须包括与主题相关的上下文。

1.  模型随后确定上下文中回答问题的最相关部分。它是通过为上下文中的每个标记（单词或子词）分配概率分数来做到这一点的。

1.  模型“认为”上下文是答案的潜在来源，并为每个标记分配两个分数：一个分数用于作为答案的**开始**，另一个分数用于作为答案的**结束**。

1.  然后选择具有最高“开始”分数和“结束”分数的标记来形成答案**跨度**。跨度就是向用户展示的内容。

为了提供一个具体的例子，如果我们向模型提问，“StyleSprint有没有任何皮夹克？”并提供上下文“StyleSprint销售各种外套、夹克和外衣”，模型将处理这个上下文并确定最可能的答案是类似“是的，StyleSprint销售各种外衣”。然而，如果问题的答案不包含在提供的上下文中，模型无法生成可靠的答案。此外，如果上下文过于不具体，模型可能会提供一个更通用的答案。就像上下文学习一样，问答的微调方法也需要相关的上下文。这意味着在实践中，模型必须与能够检索与每个问题相关额外上下文的搜索组件集成。

以我们的皮夹克例子为例。当接收到一个问题，系统可以对它的知识库进行搜索并检索与皮夹克相关的任何上下文信息（例如，关于外套的段落）。同样，由于模型被训练成以与品牌调性一致的方式回答问题，它将从提供的上下文中提取相关信息来制定适当的答案。不仅与搜索的集成将为模型提供所需的上下文，而且它还将允许模型拥有最新和实时信息。

此外，我们可能还会引入一个置信度阈值，只有当模型为开始和结束标记分配足够高的概率时，它才会给出答案。如果最高概率低于这个阈值，我们可能会说模型不知道，或者请求更多信息。总的来说，模型的有效性在很大程度上依赖于训练数据的质量和大小，以及上下文与提出的问题的相关性。

现在我们已经更好地理解了问答微调的工作原理以及在使用Hugging Face的问答管道时可以期待什么，我们可以开始编写我们的实现代码。

## Python中的实现

首先，我们安装所需的库：

```py
!pip install transformers peft sentence-transformers
```

然后，我们从transformers库中导入问答模块。对于我们的项目，我们将使用谷歌的**Flan T5（小型**），这被认为是GPT 3.5的SOTA替代品。我们的一个目标继续是衡量性能与效率之间的权衡，因此我们从Flan T5的最小版本开始，它有80M个参数。这将使训练更快，迭代更迅速。然而，请注意，即使在少量epoch上训练的小型模型也需要高RAM的运行环境：

```py
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer)
model_name = " google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```

在预训练模型实例化后，我们现在可以配置模型以适应其训练过程，使用AdaLoRA，正如我们所学的，它专门设计用于在微调过程中高效地分配参数预算：

```py
from peft import AdaLoraConfig
# Example configuration; adjust parameters as needed
adapter_config = AdaLoraConfig(target_r=16)
model.add_adapter(adapter_config)
```

正如讨论的那样，微调在很大程度上依赖于训练数据的质量和大小。在StyleSprint场景中，公司可以从其FAQ页面、社交媒体和客户服务记录中聚合问答对。为此练习，我们将构建一个类似于以下的数据集：

```py
demo_data = [{
"question": "What are the latest streetwear trends available at Stylesprint?",
  "answer": "Stylesprint's latest streetwear collection includes hoodies, and graphic tees, all inspired by the latest hip-hop fashion trends."
...
}]
```

然而，为了将我们的数据集与问答管道集成，我们首先应该了解`Trainer`类。Hugging Face transformers库中的`Trainer`类期望训练和评估数据集以特定格式提供，通常是一个PyTorch `Dataset`对象，而不仅仅是简单的字典列表。此外，数据集中的每个条目都需要进行标记化，并使用必要的字段结构化，如`input_ids`、`attention_mask`，对于问答任务，还需要`start_positions`和`end_positions`。让我们更详细地探讨这些内容：

+   `input_ids`：这是一个表示模型中输入句子的整数序列。句子中的每个单词或子词都被转换成唯一的整数或ID。回想一下，在早期章节中，这个过程被称为`[101,` `354, 2459]`。

+   `attention_mask`：注意力掩码是一个二进制值序列，其中1表示真实标记，0表示填充标记。换句话说，在1存在的地方，模型将理解这些地方需要注意力，而0存在的地方将被模型忽略。这在处理不同长度的句子和处理训练模型中的句子批次时至关重要。

+   `start_positions`和`end_positions`：这些用于问答任务。它们代表答案在上下文标记化形式中的起始和结束标记的索引。例如，在上下文*巴黎是法国的首都*中，如果问题是*法国的首都是什么？*，给出的答案是*巴黎*，在标记化后，`start_position`和`end_position`将对应于上下文中*巴黎*的索引。

有这样的理解后，我们可以创建一个类，使我们的数据集适应训练器的期望，如下所示：

```py
from torch.utils.data import Dataset
class StylesprintDataset(Dataset):
   def __init__(self, tokenizer, data):
       tokenizer.pad_token = tokenizer.eos_token
       self.tokenizer = tokenizer
       self.data = data
```

要查看完整的自定义数据集类代码，请访问此书的GitHub仓库：[https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python](https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python)。

在准备完训练集并将我们的管道配置为应用AdaLoRA方法后，我们最终可以进入训练步骤。对于这个项目，我们将配置训练只运行几个周期，但在StyleSprint场景中，需要一个更加稳健的训练过程：

```py
from transformers import Trainer, TrainingArguments
# Split the mock dataset into training and evaluation sets (50/50)
train_data = StylesprintDataset(
    tokenizer, demo_data[:len(demo_data)//2])
eval_data = StylesprintDataset(
    tokenizer, demo_data[len(demo_data)//2:])
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data
)
# Start training
trainer.train()
```

对于我们的简单实验，我们并不期望模型有很高的性能；然而，我们可以学习如何解释训练输出，它描述了模型在评估样本上的表现。`Trainer`类将输出一个包含损失指标的训练摘要。

### 训练损失

训练损失是衡量模型表现好坏的一个指标；损失值越低，表示表现越好。在许多深度学习模型中，尤其是处理复杂任务（如语言理解）的模型，通常开始时损失值会相对较高。预期这个值应该随着训练的进行而降低。

在训练的早期阶段，高损失值并不是一个需要警觉的原因，因为它通常会在模型继续学习的过程中降低。然而，如果损失值保持较高，这表明可能需要额外的训练。如果在长时间训练后损失值仍然很高，那么学习率和其他超参数可能需要调整，因为不适当的学习率可能会影响模型的学习效率。此外，应该评估训练数据的质量和数量，因为数据不足可能会阻碍训练。例如，由于我们只为实验使用了几个示例，我们预期损失值会相对较高。

下一步是使用我们新微调的模型进行推理或预测。我们还应该确保我们的训练模型参数安全，这样我们就可以在不重新训练的情况下重用它：

```py
import torch
# save parameters
model.save_pretrained("./stylesprint_qa_model")
def ask_question(model, question, context):
   # Tokenize the question and context
   inputs = tokenizer.encode_plus(question, context,
        add_special_tokens=True, return_tensors="pt")
   # Get model predictions
   with torch.no_grad():
       outputs = model(**inputs)
   # Get the start and end positions
   answer_start_scores = outputs.start_logits
   answer_end_scores = outputs.end_logits
   # Find the tokens with the highest `start` and `end` scores
   answer_start = torch.argmax(answer_start_scores)
   answer_end = torch.argmax(answer_end_scores) + 1
   # Convert the tokens to the answer string
   answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end]
            )
        )
   return answer
question = "What is the return policy for online purchases?"
context = """Excerpt from return policy returned from search."""
answer = ask_question(model, question, context)
print(answer)
```

如前所述，我们将上下文和问题一起引入模型，以便它可以识别哪个上下文片段最恰当地响应查询。因此，我们可能希望考虑集成一个向量搜索系统（如RAG），根据与查询的语义相似性自动从大型数据集中识别相关文档。这些搜索结果可能不会提供具体的答案，但训练好的QA模型可以从结果中提取更精确的答案。

使用这种混合方法，向量搜索系统首先检索与查询在语义上相关的文档或文本片段。然后，QA模型分析这个上下文以确定与StyleSprint的指南和期望相符的精确答案。

## 结果评估

为了评估我们的模型结果，StyleSprint可能会应用我们在本章中已经讨论过的定性和定量方法。为了我们的实验目的，我们可以使用一个简单的语义相似度度量来衡量模型输出的黄金标准响应：

```py
from sentence_transformers import SentenceTransformer, util
import pandas as pd
# Example of a gold standard answer written by a human
gs = "Our policy at Stylesprint is to accept returns on online purchases within 30 days, with the condition that the items are unused and remain in their original condition."
# Example of answer using GPT 3.5 with in-context learning reusing a relevant subset of the training data examples
gpt_35 = "Stylesprint accepts returns within 30 days of purchase, provided the items are unworn and in their original condition."
# Load your dataset
dataset = pd.DataFrame([
   (gs, gpt_35, answer)
])# pd.read_csv("dataset.csv")
dataset.columns = ['gold_standard_response',
    'in_context_response', 'fine_tuned_response']
# Load a pre-trained sentence transformer model
eval_model = SentenceTransformer('all-MiniLM-L6-v2')
# Function to calculate semantic similarity
def calculate_semantic_similarity(model, response, gold_standard):
    response_embedding = model.encode(
        response, convert_to_tensor=True)
    gold_standard_embedding = model.encode(gold_standard,
        convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding,
        gold_standard_embedding).item()
# Measure semantic similarity
dataset['in_context_similarity'] = dataset.apply(
    lambda row:calculate_semantic_similarity(
        eval_model, row['in_context_response'],
        row['gold_standard_response']
    ), axis=1)
dataset['fine_tuned_similarity'] = dataset.apply(
    lambda row:calculate_semantic_similarity(
        eval_model, row['fine_tuned_response'],
        row['gold_standard_response']
    ), axis=1)
# Print semantic similarity
print("Semantic similarity for in-context learning:", 
    dataset['in_context_similarity'])
print("Semantic similarity for fine-tuned model:", 
    dataset['fine_tuned_similarity'])
```

我们评估的结果如下：

|  | PEFT Flan T5 | GPT 3.5T |
| --- | --- | --- |
|  | 微调 | 上下文 |
| 语义相似度 | 0.543 | 0.91 |

表5.1：微调后的Flan和GPT 3.5 Turbo的语义相似度分数

无疑，上下文学习得出的答案与我们的黄金标准参考非常接近。然而，微调模型并不落后。这告诉我们，通过更健壮的训练数据集和相当多的epoch，微调模型可以与GPT 3.5相媲美。通过更多的迭代和实验，StyleSprint可以拥有一个非常健壮的微调模型来回答客户的具体问题。

# 摘要

在本章中，我们专注于StyleSprint的AI驱动客户服务系统中微调与上下文学习之间的战略决策过程。虽然上下文学习，尤其是少样本学习，提供了适应性和资源效率，但它可能并不始终与StyleSprint的品牌调性和客户服务指南保持一致。这种方法高度依赖于提示中提供的示例的质量和相关性，需要精心设计以确保最佳结果。

另一方面，PEFT方法如AdaLoRA，提供了一种更专注的方法来适应预训练模型以满足客户服务查询的特定需求。PEFT方法仅修改模型参数的一小部分，减少了计算负担，同时仍然实现高性能。这种效率对于现实世界应用至关重要，在这些应用中，计算资源和响应准确性都是关键考虑因素。

最终，在上下文学习与微调之间的选择，不仅仅是一个技术决策，也是一个战略决策，它与公司的运营目标、资源配置以及期望的客户体验紧密相连。本章建议进行对比测试，以评估两种方法的有效性，通过可靠的指标评估大规模的结果。这种数据驱动的评估将指导StyleSprint在提升客户服务体验方面做出最佳AI策略的决定。

总结来说，我们现在对LLM中微调与上下文学习的含义有了更全面的理解，特别是在客户服务的背景下。它强调了像StyleSprint这样的公司做出明智战略决策的需要，在微调提供的深度专业化和一致性以及上下文学习的适应性和效率之间取得平衡。

在下一章中，我们将探讨用于领域自适应的PEFT，其中我们的训练结果是一个经过微调以理解高度特定领域（如金融或法律）的通用模型。

# 参考文献

本参考文献部分作为本书中引用的资料的存储库；您可以探索这些资源，以进一步加深对主题内容的理解和知识：

+   Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. (2018). *通过生成性预训练改进语言理解*。OpenAI.

+   Hu, E. J., Shen, Y., Wallis, P., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). *LoRA: 大型语言模型的低秩自适应*。ArXiv. /abs/2106.09685

+   张琪，陈明，布哈林，何平，程宇，陈伟，赵天 (2023). *参数高效微调的自适应预算分配*。ArXiv. /abs/2303.10512

+   Brown TB, Mann B, Ryder N, et al. 2020\. *语言模型是少样本学习者*。ArXiv:2005.14165.
