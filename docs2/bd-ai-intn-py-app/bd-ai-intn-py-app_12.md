# 12

# 修正和优化您的生成式AI应用

到目前为止，您已经阅读了如何构建**生成式AI**（**GenAI**）应用、其各种组件以及它们如何相互配合的内容。您已经对它们如何工作（以及如何不工作）有了坚实的理解。您也意识到了一些GenAI应用的挑战以及如何识别它们。

在本章中，您将开始揭开如何*改进*您的GenAI应用之谜，一旦您确定了其不足之处。您还将了解优化和微调您的GenAI应用，使其成为一个可靠、有效且稳定的机器，为您服务，而不是一个带来混乱的恶意行为者。

本章将讨论几种提高您的GenAI应用效果的经典技术，以便您对自己的成品充满信心。理想情况下，您将执行所有这些技术。本章将定义这些技术并解释它们如何改进您的应用。然后，您将通过活动完成这些技术的稳健示例。到本章结束时，您将有许多关于如何改进您应用的想法。

本章将涵盖以下主题：

+   基准测试

+   训练和评估数据集

+   少样本提示

+   检索和重排

+   晚期交互策略，包括应用内反馈和用户反馈循环

+   查询重写

+   测试和红队行动

+   信息后处理

# 技术要求

本章不包含任何编码。然而，它建立在所有前面的章节之上，描述了各种改进和优化您的GenAI应用输出的方法。要重现一些示例，您只需使用您最喜欢的**大型语言模型**（**LLM**）提供商并自行尝试即可。本章使用ChatGPT。

# 基准测试

在GenAI的背景下，**基准测试**指的是为AI模型定义一个标准或参考输出，以便比较未来的输出。这个标准作为评估模型性能、一致性和随时间改进的关键基准。通过建立基准，开发人员和利益相关者可以客观地衡量AI相对于预定义的预期表现，确保模型达到并维持期望的标准。

在GenAI中，基准测试对于以下几个原因至关重要。首先，它为评估AI模型的质量和性能提供了一个明确的指标。其次，它有助于跟踪模型随时间的进展和改进。最后，基准测试是帮助确保模型输出一致性的工具，通过检测输出变异性来实现。所有这些对于维护AI系统的可靠性和信任至关重要。

可以进行基准测试的AI模型方面众多，且高度依赖于具体的应用及其目标。以下是一些可能进行基准测试的常见元素：

+   **准确性**：这涉及到衡量模型输出的正确性。例如，在语言模型中，准确性可以通过生成的文本与预期文本的匹配程度或提供正确信息的频率来衡量。

+   **响应速度**：这指的是模型在接收到输入后生成输出所需的时间。通常，更快的响应时间更受欢迎，尤其是在实时应用中。

+   **有效性**：这可以衡量AI实现其预期目的的程度。例如，在推荐系统中，有效性可能通过提供的推荐的相关性和个性化来评估。

+   **用户满意度**：这个主观指标可以通过用户反馈和调查来衡量，反映用户对AI性能和输出的满意度。

在您的当前性能旁边建立基线标准也有助于您——工程师——确定您是否随着时间的推移在提高结果。这种知识对于确保您的应用程序性能没有下降至关重要。在某些行业中，基线性能指标可能需要满足行业或监管标准，并且可能是您应用程序或组织的报告要求。

评估您应用程序的初始性能后，您将希望记录这些结果。随后，确保您在每个训练和更新周期中持续将模型的输出与基线进行比较。全面的文档提供了可以用于比较未来输出并识别模型性能中的趋势或问题的参考。

定期评估模型输出与基线之间的差异也是至关重要的。在随后的训练和更新迭代中，这些评估有助于检测与预期（基线）性能的偏差。如果模型的性能低于基线，这可能表明需要解决的问题，例如数据漂移、用户行为变化或训练数据集问题。

## 训练和评估数据集

要创建您的基线，您需要创建一个**评估数据集**。评估数据集是一系列针对您的应用程序提出的问题，以确定它是否满足您已确定的标准。请注意，评估数据集不应与**训练数据集**混淆，训练数据集是您用于**训练**模型的数据。评估数据集应是一组完全不同的问题和答案。实际上，训练数据集类似于您提供给学生的笔记和资料，而评估数据集则类似于期末考试。您不希望考试太容易！

### 训练数据集

正如其名所示，训练数据集是用于教授或**训练**机器学习模型的 数据集合。它包含输入-输出对，其中输入数据被输入到模型中，模型学习产生正确的输出。这个过程涉及调整模型的参数，以便它能够很好地泛化到新的、未见过的数据。训练数据集的质量和多样性直接影响训练模型的性能和准确性。

高质量训练数据确保模型能够识别模式并做出准确的预测或生成适当的响应。因此，您的训练数据集应代表问题域，涵盖模型在现实世界应用中预期会遇到的各种场景。这有助于减少偏差并提高模型的泛化能力。

训练数据集中的数据类型可能包括以下内容：

+   **标记数据**：这是监督学习中使用的主要数据类型。每个数据点都包含一个输入和一个相应的正确输出，或标签。例如，在文本分类任务中，标记数据可能包括句子及其相应的类别。

+   **未标记数据**：在无监督学习中使用，这种数据没有预定义的标签。模型试图在数据中找到模式和结构。例如，聚类算法使用未标记数据将相似的数据点分组在一起。

+   **混合数据**：半监督学习结合了标记和未标记数据。这种方法利用了大量的未标记数据，同时从较小的标记数据集中受益，以指导学习过程。

+   **多样化数据**：包括多样化数据确保模型可以处理各种输入。这可能包括不同的语言、方言、格式和上下文。对于某些类型的应用，这可能包括既有人可读的文档又包括代码库的训练数据。

尽管如此，您可能还希望包括**补充训练数据**。补充训练数据是指用于微调或增强已训练模型性能的附加数据。有许多原因要做这件事，但让我们谈谈其中三个特别有说服力的原因：

+   补充数据可以帮助将通用模型适应特定领域。例如，在通用文本上训练的语言模型可以通过医学文献进行微调，以在医疗应用中表现更好。

+   补充训练数据可以用来增强模型在可能较弱的特定领域的功能。例如，添加更多与金融交易相关的数据可以帮助欺诈检测模型变得更加准确。

+   随着新信息的出现，补充训练数据可以用来更新模型的知识。这对于需要最新信息的应用尤其相关，例如新闻生成或行业快速发展的领域（如技术）。

### 评估数据集

除了您的训练数据和补充数据之外，您还需要一个评估数据集。评估数据集至关重要，因为它们提供了一种受控和一致的方式来衡量AI模型的表现。它们作为比较的基准，确保模型输出可以客观地与预定义的标准进行比较。通过使用标准数据集，您可以可靠地跟踪改进、识别弱点，并随着时间的推移保持模型的质量。这有助于验证模型不仅在开发阶段表现良好，而且能够有效地推广到新的、未见过的数据。

评估数据集的内容取决于具体的应用及其目标。通常，它应包括以下内容：

+   **代表性查询**：AI在现实世界使用中可能遇到的各种问题或输入。这些应该涵盖不同的场景和边缘情况，以确保全面的评估。

+   **预期输出**：对应于每个查询的正确或理想响应，这些响应将用于与AI的响应进行比较。

+   **多样化数据**：反映模型将面临的各种输入的数据，包括语言、格式和上下文的变化。这有助于评估模型的鲁棒性和处理不同类型输入的能力。

例如，MongoDB文档聊天机器人的评估数据集包括针对前250个搜索术语、前250个按数量排序的支持问题和关于MongoDB的一些最常见问题的问答。这可以采取简单的关键词或完整的句子格式，如下所示：

```py
Mongodb install
Install mongodb ubuntu
Mongodb connection string
$in mongodb
How to create a collection in mongodb
What is an aggregation pipeline
Cannot deserialize a 'String' from BsonType 'ObjectId' in C#
```

这些术语和问题是从多个来源检索的，具体取决于您的基础设施。对于MongoDB，这些基础设施来自[mongodb.com](http://mongodb.com)的Google搜索控制台，以及支持聊天、社区论坛和Stack Overflow。

确定合适的评估数据量需要平衡彻底性和实用性。您应该有足够的数据来覆盖广泛的场景，并确保您的GenAI应用的输出始终准确可靠。通常，这涉及数百甚至数千个数据点，具体取决于应用的复杂性。

话虽如此，虽然更多的数据可以提供更全面的评估，但存在一个收益递减的点，即额外的数据不会显著提高评估，但会增加复杂性和资源需求。**过度评估**也可能导致评估数据集过度拟合，而不是提高整体性能。回到之前的学生/考试类比，你不想你的评估考试是训练材料的精确复制品，因为那时你只是在测试学生是否能够记住一个问题及其回答。你不会测试学生掌握材料的情况。

总结来说，确保高质量的、准确的和全面的训练和评估数据集可以降低模型学习错误模式的可能性。这需要在部署GenAI应用之前投入大量努力，但可以显著提高你的GenAI准确性和响应的深度，以及确保其响应质量。

## Few-shot提示

在许多情况下，你会要求你的GenAI应用生成新内容或总结现有内容，在这种情况下，你需要做的是向现有应用提供正确设计的提示。大多数时候，用户简单地请求他们需要的内容就足够了。但在输出复杂的情况下，你会发现使用一种称为**few-shot提示**的技术可以改善GenAI应用的响应质量和准确性。Few-shot提示是指你将示例作为LLM输入的一部分提供，以便它可以看到你确切需要什么类型的语法和响应。如果你认为输入可能是一个LLM不熟悉的术语，或者你正在使用特定于业务的术语，你还可以在示例中包含一个定义。

让我们用一个GPT-4的例子来试一试。

**示例 1**：让我们看看LLM对未使用few-shot提示的请求如何响应。

这是用户输入：

```py
Please generate a linkedin post less than 500 words about the company OpenAI.
```

这里是输出：

```py
🚀 Pioneering the Future with OpenAI 🚀
In the dynamic world of technology, OpenAI stands as a beacon of innovation and progress. Their mission to advance artificial general intelligence (AGI) is not just envisioning the future; they are actively shaping it.
OpenAI operates under the guidance of visionary leaders like CEO Sam Altman. Their cutting-edge research is pushing the boundaries of what's possible in AI, crafting a future where technology enhances human capabilities and opens new realms of possibility.
From healthcare to finance, and education to entertainment, OpenAI's innovations are leaving an indelible mark. Their models, such as GPT-4, are revolutionizing how we interact with technology. Businesses are leveraging these tools to enhance customer experiences, streamline operations, and achieve unprecedented levels of productivity.
OpenAI thrives on collaboration, partnering with organizations worldwide to accelerate innovation. Their open research culture demonstrates a belief in collective progress. By sharing insights, tools, and findings, OpenAI is democratizing access to AI, empowering developers, researchers, and businesses to push the boundaries of what's possible. Through initiatives like the OpenAI Scholars and Fellows programs, they are nurturing the next generation of AI talent. These programs provide mentorship, resources, and opportunities to individuals from diverse backgrounds, ensuring a wealth of perspectives and ideas in AI development.
Join the conversation and be part of the future OpenAI is creating.
#AI #OpenAI #Innovation #Technology #Future #Collaboration #Leadership
```

**示例 2**：现在，让我们尝试使用few-shot提示的例子。

这是用户输入：

```py
Please generate a linkedin post focusing on the OpenAI company. It should be less than 500 words in length. Also, please match the tone, syntax, word choices, and style of this passage:"There's a way to do it better. Find it." — Thomas Edison
"Lies. We are being lied to. We are told that technology takes our jobs, reduces our wages, increases inequality, threatens our health, ruins the environment, degrades our society, corrupts our children, impairs our humanity, threatens our future, and is ever on the verge of ruining everything.
We are told to be angry, bitter, and resentful about technology. We are told to be pessimistic. The myth of Prometheus – in various updated forms like Frankenstein, Oppenheimer, and Terminator – haunts our nightmares. We are told to denounce our birthright – our intelligence, our control over nature, our ability to build a better world. We are told to be miserable about the future." –Marc Andreessen
```

这里是输出：

```py
We are told that artificial intelligence is a threat, that it steals our jobs, that it widens the chasm of inequality, that it's an invisible force eroding our humanity. We are told that AI is an uncontrollable beast, ready to plunge us into dystopia.
OpenAI is here to shatter these myths.
Founded on the principle of advancing digital intelligence for the collective good, OpenAI isn't just another tech company. It's a bold proclamation that the future is ours to shape. OpenAI stands as a beacon against the darkness of misinformation and fearmongering. It's a rallying cry for innovation, for the relentless pursuit of better.
Imagine a world where AI enhances every aspect of our lives—where it amplifies human potential rather than diminishes it. OpenAI's groundbreaking models, like GPT-4, are not instruments of destruction but tools of empowerment. From revolutionizing healthcare to transforming education, from pioneering new frontiers in entertainment to optimizing complex financial systems, OpenAI's innovations are the keystones of a new era.
OpenAI is driven by visionaries who refuse to bow to the pessimism that pervades our discourse. That we are a testament to what can be achieved when we refuse to accept the status quo. We are the architects of tomorrow, wielding AI not as a weapon, but as a bridge.
```

你可以看到，通过提供示例文本，你可以增强输出以匹配你偏好的任何语法。额外的提示信息也不需要特别困难来生成。如果你能向你的GenAI应用提供示例输出，其结果将更接近你期望的。

## 检索和重排序

检索和重排序是用于增强LLM性能和准确性的关键技术。首先，要明白通过检索相关上下文或文档，LLM可以提供更准确和上下文相关的响应。这在模型训练数据未涵盖查询的具体内容或需要最新信息时尤其有用。

在 LLM 的背景下，**检索**可能涉及在大量文档、知识库或其他数据源中进行搜索，以找到与特定查询或任务相关的信息片段。让我们看看两种不同的检索类型：

+   在你的查询中包含 `cars`，它会返回包含单词 *cars* 的文档。

+   **基于嵌入的检索**：这使用向量嵌入来找到匹配的文档。查询和文档都被转换成高维空间中的向量。检索涉及找到接近查询向量的向量（即文档）。

**重新排序**是重新排列检索到的文档或信息片段的过程，以优先考虑最相关的内容。在初始检索之后，文档根据其与查询的相关性进行排名。检索到的文档最初根据其在嵌入空间中的相似性（例如，使用余弦相似度）与查询进行排名。然而，一个更复杂的模型可以通过考虑额外的特征和上下文来重新排序这些最初检索到的文档。

让我们看看以下示例。

**示例 1**：使用 GenAI 应用推荐餐厅。

你已经构建了一个 GenAI 应用程序，该程序提供餐厅推荐。用户请求附近的当前营业的餐厅。在检查提供给用户的潜在餐厅时，应用程序会查看用户当前位置或提供的地址与当前本地时间和营业时间之间的距离。

然后，它会根据结果进行排名，使得最近的餐厅首先显示给用户。这是一个完全可行的解决方案。但你可能希望得到更智能的结果，这些结果可以根据其他标准动态重新排序，例如餐厅的用户评分。你可能希望首先显示一个距离三英里远的评分较高的餐厅，而不是一个距离一英里远的评分为一星的餐厅。随着用户对结果的反馈，你可能希望动态重新排序，随着你获得更多关于用户偏好的信息（包括，比如说，菜系或氛围），扩大你的餐厅选择范围。

通过重新排序结果，最相关和有用的信息被优先考虑，从而提高了 LLM 输出的整体质量。它有助于过滤掉不那么相关或冗余的信息，确保响应精确且有用。

当结合检索和重新排序时，可以显著增强 LLM 的输出，具体如下：

+   模型可以访问并利用其训练数据中可能不存在的相关信息，提供更准确和上下文相关的答案。

+   通过通过重新排序来关注最相关的信息，模型的响应变得更加精确，减少了错误和不相关的内容。

+   检索可以从更新的来源中提取最新信息，使模型的响应更加及时。

+   这些技术使模型能够高效地处理特定、详细的查询，而无需频繁重新训练整个模型。

**示例 2**：总结量子计算的最新研究。

这里还有一个实际例子。假设您询问LLM关于量子计算的最新研究。输出步骤如下：

1.  **检索**：该模型通过搜索大量科学论文和文章的数据库来找到关于量子计算的相关文档。

1.  **重排序**：最初检索到的文档随后被重新排序，最新的和最相关的研究被置于顶部。

1.  **响应生成**：LLM使用排名最高的文档来生成关于量子计算最新研究趋势的详细和准确响应。

通过结合检索和重排序，LLM可以提供信息丰富、最新和上下文准确的答案，极大地改善了用户体验。

## 后期交互策略

现在您准备将应用程序投入生产，还有一些事情可以做来帮助改善用户体验并创建反馈循环，以便更好地了解您的GenAI应用程序的行为。这些建议的下一组重点在于**后期交互策略**，有时也称为**BERT上的上下文化后期交互**（**ColBERT**）。

首先，让我们定义**交互**。交互是指通过比较查询和文档的表示来评估查询与文档之间相关性的过程。后期处理策略是指在处理过程中较晚发生查询和文档表示之间的交互，通常是在它们被独立编码之后。早期交互模型是在查询和文档嵌入在早期阶段交互，通常是在模型编码之前或编码期间。

其次，让我们深入了解一下其内部工作原理。当用户与GenAI应用程序交互时，他们输入一个查询，该查询被编码成一个密集向量表示。潜在响应，通常是文档或段落，也被编码成密集向量表示。系统在查询和文档嵌入之间执行相似度匹配，返回相似度得分最高的文档作为最佳匹配。

为了提高相关性，您不会向用户返回所有匹配结果。相反，您旨在提供最相关的结果或结果集的总结版本。ColBERT等后期交互模型通过专注于最有希望的查询-文档对，而不是考虑所有可能的组合，从而提高效率，产生更精确的结果和更好的用户体验。这种选择性的方法可以提供更精确和相关的结果，从而提升用户体验。

如果你需要专注于提高搜索结果，考虑实现ColBERT或类似技术来增强检索性能，并为用户查询提供更相关的结果。

## 查询重写

**查询重写**，或**查询重构**，是一种用于提高LLM（大型语言模型）提供答案质量的技巧。这个过程涉及修改原始查询，使其更清晰、更具体或更详细，这有助于模型生成更好的响应。LLMs不会在后台明确重写查询，因此除非你实现了在处理之前评估和重写用户查询的工作流程，否则这项工作将是手动的。

重写查询可以使它更清晰、更精确，减少歧义，并确保模型确切地理解了被询问的内容。向查询中添加相关上下文或细节可以帮助模型提供更准确和上下文相关的答案，并有助于消除具有多个含义的术语的歧义，确保响应与预期意义一致。此外，重构查询以包含更多相关细节可以导致更全面的答案。

查询重写是如何工作的？了解你的GenAI（生成式人工智能）应用程序的用户意图非常重要。你的应用程序的**目的**是什么，它将尝试回答哪些类型的问题？了解用户期望得到的响应与你的应用程序可能提供的响应之间的差异是关键。之后，你可以进行以下活动，这些活动不是相互排斥的，这意味着你可以执行一些、一个或全部这些活动。

例如，基于**意图**，用户查询可以通过添加额外的**上下文**和细节来增强。这项活动极大地扩展了用户查询（并增加了每个查询的令牌计数），但通常会得到更好的结果。

以一个简单的例子来说明，假设你的应用程序生成图像。用户请求“一张小猫的图片”，这是一个相当简单的查询，可能有无穷无尽的结果。

为了帮助用户获得更好的结果，你可以在用户界面中添加三个按钮，以便用户可以选择“一只小猫的图片”，查询被修改为以下内容：

```py
An image of a kitten, in anime style, large eyes, chikai, chibi-style, pixel-style, anime illustration, cute, in the style of Akira Toriyama.
```

在这里，对于每种按钮样式，你可以在提交前添加增强用户查询的术语。

作为另一个例子，考虑以下用户查询：

```py
"What's the average revenue?"
```

有意义的重写可能如下所示：

```py
"What's the average revenue for [May 2024] for [sales sku 123]?"
```

这个添加了额外上下文的重写查询有助于系统理解用户正在请求特定产品和时间段，从而得到更准确和有用的响应。

最终，在进行查询重写时，你希望**简化语言**。复杂的查询可以被简化或分解成更简单的部分，使模型更容易处理并准确响应。这种方法涉及将大型查询分解为其组成部分（通常通过一系列输入字段/表单实现），然后将每个数据条目统一为一个提交的查询。这指导用户构建一个没有专业知识的良好格式查询。

例如，假设你的用户只有一个输入字段来输入他们的查询。在这种情况下，他们可能会遗漏相关信息或提供可能影响准确性或增加幻觉可能性的无关信息。相反，如果你为用户提供一系列字段，每个字段都有明确的说明，然后将输入的信息组装成一个查询，该查询被输入到GenAI应用中，你会得到比自由文本输入更好的结果。

对于实际实施，你可以考虑一个工作流程，其中系统本身分析查询的意图和上下文，审查查询的复杂性，然后将查询重写得更清晰、更具体或更详细。重写的查询可以用来生成响应。

# 测试和红队行动

测试AI系统对于确保它们的准确性、可靠性和整体性能至关重要。通常，在软件工程中，自动化测试被用作软件开发过程的一部分。GenAI应用也不例外。你希望定期和定期测试输出，以确保输出质量没有发生重大变化。

## 测试

就像你典型的软件工程特性一样，你希望在测试计划中包括单元测试、集成测试、性能测试和用户验收测试的阶段。然而，具体做法因用例而异。

在GenAI应用的背景下，**单元测试**仍然遵循相同的基本原则，涉及测试应用程序的各个组件或模块以确保它们能正确运行。然而，在GenAI应用的情况下，你的单元测试还需要包括以下步骤：

+   **输入验证**：确保应用程序正确处理和验证各种输入类型、格式和范围。测试边缘情况，如空输入、过大输入或不规范数据。

+   **预处理**：验证任何预处理步骤，如分词、归一化或特征提取，是否正确执行。

+   **模型加载**：测试模型是否正确从其存储位置加载，并验证是否使用了正确的版本。

+   **模型推理**: 确保模型在给定有效输入的情况下能够生成无错误的输出。通过受控输入测试推理函数以验证预期的行为，例如对于某些提示或场景的确定性响应。

+   **输出格式**: 验证生成的输出是否符合预期的格式和结构。这包括检查输出是否完整、格式正确，并遵守任何长度或内容限制。

+   **后处理**: 测试任何修改或增强模型输出的后处理步骤，例如清理文本、转换格式或应用额外的业务逻辑。

+   **正常功能**: 输出应该正常工作。如果你的 GenAI 应用程序输出代码，你需要测试代码本身是否能够编译并且按照预期行为执行。

这些只是你应该包括在单元测试你的 GenAI 应用程序中的几个项目。

**集成测试**侧重于验证你的 GenAI 系统的组件是否按需协同工作。这意味着你将测试组件之间的交互，以检查以下内容：

+   是否你的数据摄取管道拉取了正确的数据

+   如何向用户展示推荐（例如，格式化，如果这是由另一个库或工具完成的）

+   如果你使用的是 OpenAI 或 Anthropic 等其他 LLM，进行 API 负载测试。

你将通过**性能测试**评估处理时间、效率和可扩展性。这可能包括以下活动：

+   测试你的应用程序如何处理大量并发查询的负载。

+   评估自托管模型在不同硬件配置下的推理时间。

+   测量输入和输出应该设置多少令牌限制以控制成本和处理时间。

+   测量模型生成输出所需的时间，并确保它符合性能要求。这对于具有实时约束的应用程序尤为重要。

除了这个常规测试之外，你还需要为你的测试套件添加更多内容。一般来说，也建议 GenAI 应用程序进行以下**额外测试**：

+   **偏差和公平性**: 如果你的模型做出的推荐会影响生活和生计，你将需要仔细考虑不同人口群体的训练数据偏差。

+   **鲁棒性**: 为了确保你的 GenAI 应用程序能够抵抗变化和噪声，你需要使用对抗性示例和边缘情况来测试，以评估其处理意外输入的能力。

一旦你完成了所有这些，你将想要考虑**用户验收测试**，这是流程中最激动人心的部分，你将在下一节中看到。

## 红队攻击

如果你的GenAI应用将接受来自人类的自然语言提示和输入，那么强烈推荐进行**红队测试**。红队测试涉及模拟现实世界、具有挑战性或对抗性的情况，以识别你的GenAI应用中的漏洞和弱点。这种方法借鉴了网络安全实践，对于确保你的GenAI应用满足用户期望尤为重要。

这涉及到拥有一个大量用户池，他们会提出现实世界的问题，但他们不受**脚本**的限制，不知道可以问什么。进行红队测试的原因是GenAI应用可以，并且经常会产生不同的输出，即使输入相似或相同，输出也会有很大的差异。不仅如此，生成的输出质量往往是主观的，取决于人类判断。因此，虽然传统软件应用会产生可预测和一致的结果，但GenAI并非如此。让我们通过一个例子来看看这是如何工作的。

对于聊天机器人应用，你可能会有常规的自动化测试，它会向你的GenAI应用提出最常问的200个问题，然后评估它们的正确性。使用红队，你会让50个用户提出他们想问的任何问题，并记录下提出的问题和回答。这可能会产生以下见解：

+   如果用户以类似的方式提出问题，但不是用完全相同的措辞，他们就会收到不正确或不那么正确的答案。

+   一些用户会提出恶意问题，而GenAI应用会做出较差的回应。

+   其他用户提出的问题不在训练数据中，GenAI应用会胡编乱造答案（或者根本不回答），从而识别出需要扩展训练数据的必要性。

+   当用户连续提出许多问题时，应用会卡住。

+   当用户提出特定类型的问题时，他们会对输出感到不满意，因为应用缺乏高质量的训练数据，或者回复的格式不理想。

+   当得到适当的提示时，GenAI应用会分享其他用户会话的细节，从而识别出安全问题。

为了启用红队测试阶段，建议你记录下每个用户提出的每个问题以及每个给出的回答，然后要求测试者对回答进行评分并附上备注。虽然这种详细程度的用户测试在软件开发中既费力又罕见，但在产品发布前看到你的应用在现实场景中、与真实人类互动的表现，却具有极高的价值。

由于某些人工智能系统的规模和范围，全面测试每个组件是不可能的。有效的测试和红队行动依赖于对系统哪些部分风险最高的判断。偶尔给出不太准确的建议可能是一个无足轻重的事件。然而，单个幻觉的潜在危害可能相当高。您将需要考虑危害的严重性、不准确性的可能性以及撤回或纠正不准确性的能力，作为您衡量风险的标准措施。使用这些简单但主观的措施可以帮助您确定测试系统每个方面的程度以及您红队的大小。

为了让您对将要测试哪些类型的危害和事件有一个概念——这些事件太多，无法一一列举——您会发现查看[https://incidentdatabase.ai/](https://incidentdatabase.ai/)上的AI事件数据库很有帮助。在审查此工具后，您可能会发现您特定的用例（或类似用例）以及已经报告的事件，这样您就可以测试并思考不准确性可能带来的后果。

例如，这里详细描述的一个事件涉及一个提供人员配备级别建议的应用程序。然而，基于算法的建议导致设施人员不足，导致忽视、伤害和死亡的关键事件。这些事件随后引发了针对使用人工智能的医疗保健提供者的诉讼甚至立法。

# 信息后处理

您可能知道，生成式人工智能与之前的人工智能或分析形式的主要区别在于它能够高效地生成新内容。但您知道这些内容通常是*非结构化*形式，例如书面文本或图像吗？当您看到格式良好、以项目符号列表、多种字体等形式呈现的输出时，这是一种**信息后处理**的形式。

信息后处理指的是在人工智能模型生成初步响应之后，但在将响应发送给用户之前所采取的一系列步骤。这一关键步骤增强了生成式人工智能模型的输出，将原始响应精炼得更加有用、准确和符合上下文。它可能采取多种形式，因此本章将仅讨论其中一些最有用的形式，并附带如何实施它们的信息：

+   **事实核查**：验证提供信息的准确性。这可能涉及将事实与可靠来源或数据库进行核对。

+   **格式化**：以清晰易读的格式组织信息，例如项目符号、段落或表格。这还可能包括风格变化，如粗体、文字颜色或字体，以增强可读性和强调。

+   **语法、风格和语气检查**：有时，GenAI应用提供的文本结果不符合标准或与预期的精确信息、语气和风格不一致。后处理工具和供应商可以显著改善生成的文本输出，使其更具可读性，使其符合读者的期望。

信息后处理是GenAI输出生命周期中的一个重要组成部分。它弥合了原始模型输出和经过打磨、用户准备好的响应之间的差距，提高了准确性、可读性、相关性和整体用户满意度。通过实施有效的后处理策略，AI系统可以提供更高质量和更可靠的输出。

围绕GenAI流程中这一宝贵步骤，出现了整个服务，因此工程师不必自己构建它。

# 其他补救措施

一些其他的技术补救措施甚至比本章详细描述的更容易实施。其中一些可能会提高你的GenAI应用的准确性和性能，尽管所需的努力程度不同。例如，在MongoDB对GPT的测试中，发现相同问题集的准确率在GPT-3.5和GPT-4之间提高了7%。通过提示、检索增强或后期交互策略获得这样的准确率提升是可能的，但会非常困难。

因此，调查所有潜在的改进途径都是值得的，包括硬件升级、代码优化、并发管理、数据库查询优化，甚至仅仅是升级你的软件。所有这些都可以提高你的GenAI应用的结果，并且应该独立进行调查：

+   **硬件和软件升级**：升级计算资源，例如使用更强大的GPU、通过更多服务器进行横向扩展，或更新到软件的最新版本，以对准确性和性能产生巨大影响。

+   **代码优化**：重构和优化代码以提高效率，减少计算负载，并更有效地处理数据。

+   **网络优化**：通过优化数据传输、缓存响应和最小化API调用开销来降低网络延迟。

+   **并发管理**：实现并发和并行处理技术以高效处理多个请求。

+   **数据库优化**：优化数据库查询和交互以减少I/O开销。

# 摘要

实施纠正和优化你的GenAI应用的机制可以有多种形式，可以在生成答案之前、期间和之后实施。为了获得最佳性能，你希望用高质量的数据训练你的GenAI模型，用你的特定用例数据补充现有模型，并拥有详尽的评估数据集，记录模型的表现以建立准确性的基线。

然而，一旦有了这个基准，你就可以立即开始使用本章讨论的技术来改进它。其中一种技术是一或少量提示。这涉及到向GenAI模型提供一个示例或提示来引导其响应，使模型能够在最少训练数据的情况下生成相关且上下文适当的输出。你也可以尝试根据用户的查询检索和重新排序相关文档或数据点，然后在生成最终响应之前重新排序这些结果，以优先显示最相关和有用的信息。查询重写是另一种可以提高清晰度、具体性或上下文的技术，有助于AI模型更准确地理解和响应用户的请求。

通过结构化和以清晰、有序和可读的方式呈现AI生成的内容来格式化GenAI响应可以增强整体用户体验并确保信息易于消化。同样，实施如ColBERT之类的后期交互策略可以提高检索信息的关联性和准确性。通过测试、红队攻击和记录你的结果，你可以跟踪你在提高性能、安全性和响应质量方面的进展。

GenAI技术正在改变（并将继续改变）软件行业的面貌。有了这些优化策略，你的GenAI应用将能够适应并在这个不断变化的环境中脱颖而出。

# 附录：进一步阅读

除了章节内提供的链接外，这里还有一些资源可以帮助你继续学习之旅。

[**第1章**](B22495_01.xhtml#_idTextAnchor009)**, 开始使用生成式AI**

+   Gryka, Maciej. “Invest in RAG” in “Building reliable systems out of unreliable agents.” *The Rainforest Blog*, April 3, 2024\. [https://www.rainforestqa.com/blog/building-reliable-systems-out-of-unreliable-agents#Invest_in_RAG](https://www.rainforestqa.com/blog/building-reliable-systems-out-of-unreliable-agents#Invest_in_RAG).

+   “The Black Box: Even AI’s creators don’t understand it.” July 2023\. *Unexplainable*. Produced by Vox Creative. Podcast, Spotify, 36:15\. [https://open.spotify.com/episode/3npjXNCtUSGRUjVR4EYb4Y?si=-XpudYVzSEKfhD0-2NBjEQ](https://open.spotify.com/episode/3npjXNCtUSGRUjVR4EYb4Y?si=-XpudYVzSEKfhD0-2NBjEQ).

[**第2章**](B22495_02.xhtml#_idTextAnchor021)**, 智能应用构建模块**

+   Naveed et al. “大型语言模型的全面概述。” arXiv，2023年7月12日\[https://arxiv.org/abs/2307.06435](https://arxiv.org/abs/2307.06435).

[**第3章**](B22495_03.xhtml#_idTextAnchor041)**, 大型语言模型**

+   “语音和语言处理，” n.d., [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/).

+   Hochreiter, Sepp，和 Jürgen Schmidhuber。“长短期记忆。” *Neural Computation* 9，第 8 期（1997年11月1日）：1735–80\[https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735).

+   Vaswani, Ashish，Noam Shazeer，Niki Parmar，Jakob Uszkoreit，Llion Jones，Aidan N. Gomez，Lukasz Kaiser，和 Illia Polosukhin。“Attention Is All You Need。” *arXiv (Cornell University)*，2017年1月1日\[https://doi.org/10.48550/arxiv.1706.03762](https://doi.org/10.48550/arxiv.1706.03762).

+   “提示工程指南 – Nextra”，未注明日期\[https://www.promptingguide.ai/](https://www.promptingguide.ai/).

[**第四章**](B22495_04.xhtml#_idTextAnchor061)**,** **嵌入模型**

+   A. Aruna Gladys 和 V. Vetriselvi，“关于情感识别的多模态方法综述”，*Neurocomputing* 556（2023年11月1日）：126693\[https://doi.org/10.1016/j.neucom.2023.126693](https://doi.org/10.1016/j.neucom.2023.126693).

+   Sumit Kumar，“语义搜索中表示学习的正负采样策略”，Sumit 的日记，2023年3月22日\[https://blog.reachsumit.com/posts/2023/03/pairing-for-representation](https://blog.reachsumit.com/posts/2023/03/pairing-for-representation).

+   Tomas Mikolov 等人，“在向量空间中高效估计词表示”，arXiv.org，2013年1月16日，[https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781).

+   OpenAI，“GPT-4”。GPT-4 研究，2023年3月14日\[https://openai.com/index/gpt-4-research](https://openai.com/index/gpt-4-research).

+   Jeffrey Pennington，“GloVe：全局词向量表示”，未注明日期\[https://nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove).

+   Jacob Devlin 等人，“BERT：用于语言理解的深度双向变换器预训练”，arXiv.org，2018年10月11日\[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

+   “fastText”，未注明日期\[https://fasttext.cc/](https://fasttext.cc/).

+   Peters, M. E.，Neumann, M.，Iyyer, M.，Gardner, M.，Clark, C.，Lee, K.，和 Zettlemoyer, L.，“深度上下文化词表示”，arXiv:1802.05365，2018年3月22日\[https://arxiv.org/pdf/1802.05365](https://arxiv.org/pdf/1802.05365).

+   Karen Simonyan 和 Andrew Zisserman，“用于大规模图像识别的非常深卷积网络”，arXiv.org，2014年9月4日\[https://arxiv.org/abs/1409.1556v6](https://arxiv.org/abs/1409.1556v6).

+   Kaiming He 等人，“用于图像识别的深度残差学习”，arXiv.org，2015年12月10日\[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385).

+   Aurora Cramer，Ho-Hsiang Wu，Justin Salamon，和 Juan Pablo Bello，“OpenL3 — OpenL3 0.4.2 文档”，未注明日期\[https://openl3.readthedocs.io/en/latest/#](https://openl3.readthedocs.io/en/latest/#).

+   “Google | vggish | Kaggle”，未注明日期\[https://www.kaggle.com/models/google/vggish](https://www.kaggle.com/models/google/vggish).

+   Tran, D.，Bourdev, L.，Fergus, R.，Torresani, L.，和 Paluri, M.，“使用 3D 卷积网络学习时空特征。”arXiv:1412.0767，2015年10月7日。[https://arxiv.org/pdf/1412.0767](https://arxiv.org/pdf/1412.0767).

+   Grover, A.，和 Leskovec, J.，“Node2Vec：网络的缩放特征学习方法。”*第 22 届 ACM SIGKDD 国际知识发现和数据挖掘会议论文集*，2016年。[https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf).

+   Bryan Perozzi，Rami Al-Rfou，和 Steven Skiena，“DeepWalk，”2014年8月24日，[https://doi.org/10.1145/2623330.2623732](https://doi.org/10.1145/2623330.2623732).

+   Zhang, S.，和 Xu, Y.，“Json2Vec：JSON 数据的表示学习方法。”arXiv:2002.05707，2020年2月13日。[https://arxiv.org/pdf/2002.05707](https://arxiv.org/pdf/2002.05707).

+   Alec Radford 等人，“从自然语言监督中学习可迁移的视觉模型，”arXiv.org，2021年2月26日，[https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020).

[**第五章**](B22495_05.xhtml#_idTextAnchor115)**，向量数据库**

+   Yu. A. Malkov 和 D. A. Yashunin，“使用分层可导航小世界图进行高效且鲁棒的近似最近邻搜索，”arXiv.org，2016年3月30日，[http://arxiv.org/abs/1603.09320](http://arxiv.org/abs/1603.09320).

+   Yikun Han，Chunjiang Liu，和 Pengfei Wang，“向量数据库全面综述：存储和检索技术，挑战，”arXiv.org，2023年10月18日，[http://arxiv.org/abs/2310.11703](http://arxiv.org/abs/2310.11703).

+   Zhi Jing 等人，“大型语言模型与向量数据库的相遇：综述，”arXiv.org，2024年1月30日，[http://arxiv.org/abs/2402.01763](http://arxiv.org/abs/2402.01763).

+   Doug Turnbull，“什么是判断列表？”，Doug Turnbull 的博客，2021年2月21日，[https://softwaredoug.com/blog/2021/02/21/what-is-a-judgment-list](https://softwaredoug.com/blog/2021/02/21/what-is-a-judgment-list).

+   “构建基于 RAG 的 LLM 应用程序以用于生产，”Anyscale，未注明日期，[https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1).

+   “如何执行混合搜索 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/).

+   “审查部署选项 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/atlas-vector-search/deployment-options/](https://www.mongodb.com/docs/atlas/atlas-vector-search/deployment-options/).

[**第六章**](B22495_06.xhtml#_idTextAnchor137)**，AI/ML 应用设计**

+   “如何在向量搜索中索引字段 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/#considerations](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/#considerations).

+   Lauren Schaefer Daniel Coupal，“膨胀的文档 | MongoDB，”2022年5月31日，[https://www.mongodb.com/developer/products/mongodb/schema-design-anti-pattern-bloated-documents/](https://www.mongodb.com/developer/products/mongodb/schema-design-anti-pattern-bloated-documents/).

+   Daniel Coupal，“使用模式构建：扩展引用模式，”MongoDB，2019年3月19日，[https://www.mongodb.com/blog/post/building-with-patterns-the-extended-reference-pattern](https://www.mongodb.com/blog/post/building-with-patterns-the-extended-reference-pattern).

+   “Atlas 集群大小和层级选择 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/sizing-tier-selection/](https://www.mongodb.com/docs/atlas/sizing-tier-selection/).

+   “自定义集群存储 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/customize-storage/](https://www.mongodb.com/docs/atlas/customize-storage/).

+   “Amazon EBS 卷类型 - Amazon EBS，”未注明日期，[https://docs.aws.amazon.com/ebs/latest/userguide/ebs-volume-types.html#gp3-ebs-volume-type](https://docs.aws.amazon.com/ebs/latest/userguide/ebs-volume-types.html#gp3-ebs-volume-type).

+   “自定义集群存储 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/customize-storage/](https://www.mongodb.com/docs/atlas/customize-storage/).

[**第7章**](B22495_07.xhtml#_idTextAnchor162)**，有用的框架、库和 API**

+   “MongoDB Atlas，”LangChain，未注明日期，[https://python.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/](https://python.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/).

+   “如何在向量搜索中索引字段 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/atlas-vector-search/manage-indexes/](https://www.mongodb.com/docs/atlas/atlas-vector-search/manage-indexes/).

+   “使用 LangChain 集成开始 - MongoDB Atlas，”未注明日期，[https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/).

+   “使用 Python 的 MongoDB - MongoDB 文档，”未注明日期，[https://www.mongodb.com/docs/languages/python/#integrations](https://www.mongodb.com/docs/languages/python/#integrations).

+   “Transformers，”未注明日期，[https://huggingface.co/docs/transformers/en/index](https://huggingface.co/docs/transformers/en/index).

+   “OpenAI 开发者平台，”OpenAI 平台，未注明日期，[https://platform.openai.com/docs/overview](https://platform.openai.com/docs/overview).

[**第8章**](B22495_08.xhtml#_idTextAnchor180)**，在 AI 应用中实现向量搜索**

+   Yunfan Gao等人，“大型语言模型的检索增强生成：综述，” arXiv.org，2023年12月18日，[https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997).

+   Rupak Roy, “利用CommaSeparatedListOutputParser、PydanticOutputParser等LLM输出解析器构建结构化AI | by Rupak (Bob) Roy - II | Medium | Medium，” *Medium*，2024年8月14日，[https://bobrupakroy.medium.com/harness-llm-output-parsers-for-a-structured-ai-7b456d231834](https://bobrupakroy.medium.com/harness-llm-output-parsers-for-a-structured-ai-7b456d231834).

+   Mirjam Minor和Eduard Kaucher，“使用LLM进行检索增强生成以解释业务流程模型，” 在 *Lecture Notes in Computer Science*，2024年，第175-190页，[https://doi.org/10.1007/978-3-031-63646-2_12](https://doi.org/10.1007/978-3-031-63646-2_12).

[**第9章**](B22495_09.xhtml#_idTextAnchor193)**, LLM输出评估**

+   “Papers with Code - 测量大规模多任务语言理解，” 2020年9月7日，[https://paperswithcode.com/paper/measuring-massive-multitask-language](https://paperswithcode.com/paper/measuring-massive-multitask-language).

+   “Papers with Code - HellaSwag: 一台机器真的能完成你的句子吗？，” 2019年5月19日，[https://paperswithcode.com/paper/hellaswag-can-a-machine-really-finish-your](https://paperswithcode.com/paper/hellaswag-can-a-machine-really-finish-your).

+   “Papers with Code - 评估在代码上训练的大型语言模型，” 2021年7月7日，[https://paperswithcode.com/paper/evaluating-large-language-models-trained-on](https://paperswithcode.com/paper/evaluating-large-language-models-trained-on).

+   “介绍 | Ragas，” 未注明日期，[https://docs.ragas.io/en/stable/index.html](https://docs.ragas.io/en/stable/index.html).

[**第10章**](B22495_10.xhtml#_idTextAnchor214)**, 精炼语义数据模型以提高准确性**

+   “SentenceTransformers 文档 — Sentence Transformers 文档，” 未注明日期，[https://sbert.net/](https://sbert.net/).

+   “训练和微调Sentence Transformers模型，” 未注明日期，[https://huggingface.co/blog/how-to-train-sentence-transformers](https://huggingface.co/blog/how-to-train-sentence-transformers).

+   “运行向量搜索查询 - MongoDB Atlas，” 未注明日期，[https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter).

+   “知识图谱RAG查询引擎 - LlamaIndex，” 未注明日期，[https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine/](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine/).

[**第11章**](B22495_11.xhtml#_idTextAnchor232)**, 生成式AI的常见失败**

+   Lance Eliot， “医生依赖生成式AI来总结医疗记录可能会不知不觉地承担重大风险，” *Forbes*，2024年7月2日，[https://www.forbes.com/sites/lanceeliot/2024/02/05/doctors-relying-on-generative-ai-to-summarize-medical-notes-might-unknowingly-be-taking-big-risks/](https://www.forbes.com/sites/lanceeliot/2024/02/05/doctors-relying-on-generative-ai-to-summarize-medical-notes-might-unknowingly-be-taking-big-risks/).

+   Markman, Ofer. “是时候制定战略了：85%的数据是垃圾或孤岛化的。” *Filo Focus*，2024年2月11日。 [https://www.filo.systems/blog/85-percent-of-data-is-not-actionable-time-to-restrategize](https://www.filo.systems/blog/85-percent-of-data-is-not-actionable-time-to-restrategize).

+   Neeman, Ella, Roee Aharoni, Or Honovich, 等人. “DisentQA：通过反事实问答分解参数知识和上下文知识。” arXiv.org，2022年11月10日。 [https://arxiv.org/pdf/2211.05655](https://arxiv.org/pdf/2211.05655).

+   Sharma, Mrinank, Meg Tong, Tomasz Korbak, 等人. “理解语言模型中的谄媚。” arXiv.org，2023年10月20日。 [https://arxiv.org/abs/2310.13548](https://arxiv.org/abs/2310.13548).

+   Sparkes, Matthew. “随着AI聊天机器人变得更加高级，它们变得越来越谄媚。” *New Scientist*，2023年8月17日。 [https://www.newscientist.com/article/2386915-ai-chatbots-become-more-sycophantic-as-they-get-more-advanced/](https://www.newscientist.com/article/2386915-ai-chatbots-become-more-sycophantic-as-they-get-more-advanced/).

+   Wei, Jerry, Da Huang, Yifeng Lu, 等人. “简单的合成数据可以减少大型语言模型中的谄媚。” arXiv.org，2023年8月7日。 [https://arxiv.org/abs/2308.03958](https://arxiv.org/abs/2308.03958).

[**第12章**](B22495_12.xhtml#_idTextAnchor253)**，纠正和优化您的生成式AI应用**

+   Chui, Michael, Roger Roberts, Tanya Rodchenko, 等人. “每位CEO都应该了解的生成式AI。” McKinsey Digital, 2023年5月12日。 [https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/what-every-ceo-should-know-about-generative-ai](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/what-every-ceo-should-know-about-generative-ai).

+   Xiao, Han. “什么是ColBERT和晚期交互，以及为什么它们在搜索中很重要？”，Jina AI，2024年2月20日。 [https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/).
