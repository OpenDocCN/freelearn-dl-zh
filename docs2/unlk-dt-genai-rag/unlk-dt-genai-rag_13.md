

# 使用提示工程来提升 RAG 努力

快速问答，您使用什么从大型语言 **模型** **（**LLM**）** 中生成内容？

一个提示！

显然，提示是任何生成式 AI 应用的关键元素，因此任何 **检索增强生成** (**RAG**) 应用。 RAG 系统结合了信息检索和生成语言模型的能力，以提升生成文本的质量和相关性。 在此背景下，提示工程涉及战略性地制定和优化输入提示，以改善相关信息的检索，从而提高生成过程。 提示是生成式 AI 世界中另一个可以写满整本书的领域。 有许多策略专注于提示的不同领域，可以用来改善您 LLM 使用的成果。 然而，我们将专注于更具体于 RAG 应用 的策略。

在本章中，我们将集中精力探讨以下主题：

+   关键提示工程概念 和参数

+   针对 RAG 应用的具体提示设计和提示工程的基本原理

+   适应不同 LLM 的提示，而不仅仅是 OpenAI 模型

+   代码实验室 13.1 – 创建自定义 提示模板

+   代码实验室 13.2 – 提示选项

到本章结束时，您将具备 RAG 提示工程的坚实基础，并掌握优化提示以检索相关信息、生成高质量文本以及适应特定用例的实用技术。 我们将从介绍提示世界中的关键概念开始，首先是 提示参数。

# 技术要求

本章的代码放置在以下 GitHub 仓库中： [https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_13](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_13 )

# 提示参数

在大多数 LLM 中存在许多共同参数，但我们将讨论一个可能对您的 RAG 努力产生影响的较小子集：温度、top-p，以及种子。 。

## 温度

如果你将你的 输出视为一系列 **标记**，那么在基本意义上，一个 LLM（大型语言模型）是根据你提供的数据和它已经生成的先前标记来预测下一个单词（或标记）的。 LLM 预测的下一个单词是表示所有潜在单词及其概率的概率分布的结果。 LLM 预测的下一个单词是表示所有潜在单词及其概率的概率分布的结果。

在许多情况下，某些单词的概率可能远高于其他大多数单词，但 LLM 仍然有一定概率选择其中不太可能出现的单词。 温度是决定模型选择概率分布中较后单词的可能性大小的设置。 换句话说，这允许你使用温度来设置模型输出的随机程度。 你可以将温度作为一个参数传递给你的 LLM 定义。 这是可选的。 如果你不使用它，默认值是 `1`。你可以设置温度值在 `0` 和 `2`之间。 较高的值会使输出更加随机，这意味着它将强烈考虑概率分布中较后的单词，而较低的值则会做 相反的事情。

简单的温度示例

让我们回顾一个简单的 *下一个单词* 概率分布的例子，以说明温度是如何工作的。 假设你有一个句子 `The dog ran` ，并且你正在等待模型预测下一个 *单词*。假设基于这个模型的训练和它考虑的所有其他数据，这个预测的简单条件概率分布如下： 如下：

`P("next word" | "The dog ran") = {"down": 0.4, "to" : 0.3, "with": 0.2, "``away": 0.1}`

总概率加起来是 `1`。最可能的词是 `down` ，其次是 `to`。然而，这并不意味着 `away` 永远不会出现在推理中。 模型将对这个选择应用概率模型，有时，随机地，不太可能的词会被选中。 在某些场景中，这可能是您 RAG 应用的优势，但在其他情况下，这可能是劣势。 如果您将温度设置为 `0`，它将只使用最可能的词。 如果您将其设置为 `2`，则更有可能查看所有选项，并且大多数情况下会随机选择不太可能的词。 换句话说，您可以通过增加温度来增加模型的随机性。

我们从一开始就在使用温度，将其设置为零。 以下是添加的行： （此处省略了代码行）

```py
 llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

这里的目的是使我们的代码实验室的结果更可预测，以便当您运行它们时，您可以得到类似的结果。 您的结果可能会有所不同，但至少在 `0` 温度下，它们有更大的可能性 相似。

您可能并不总是想使用 `0` 温度。 考虑以下场景，例如当您希望从 LLM 获得更 *有创意* 的输出时，您可能想利用温度在您的 RAG 应用中。

温度 和 top-p 在某种程度上是相关的，因为它们都管理 LLM 输出的随机性。 然而，它们之间有差异。 让我们讨论 top-p，并谈谈这些 差异是什么。

## Top-p

与温度类似，top-p 也可以帮助您将随机性引入模型输出。 然而，温度处理的是对输入随机性的总体强调，而 top-p 可以帮助您通过那种随机性针对概率分布的特定部分。 在提供的简单示例中，我们讨论了 概率分布：

```py
 P("next word" | "The dog ran") = {"down": 0.4, "to" : 0.3,
    "with": 0.2, "away": 0.1}
```

记住，我们提到这里表示的总概率加起来是 `1.0`。使用 top-p，你可以指定你希望包含的概率部分。 例如，如果你将 top-p 设置为 `0.7`，它将只考虑这个概率分布中的前两个词，这些词的总和是概率分布中的第一个 `0.7` (1.0 中的)。 使用温度时，你没有这种针对性的控制。 Top-p 也是可选的。 如果你不使用它，默认值是 `1`，这意味着它考虑了 所有选项。

你可能想同时使用温度和 top-p，但这可能会变得非常复杂且不可预测。 因此，通常建议使用其中之一，而不是同时使用。 同时使用。

| **LLM 参数** | **结果** |
| --- | --- |
| 温度 | 一般随机性 |
| Top-p | 针对性随机性 |
| 温度 + top-p | 不可预测的复杂性 |

表 13.1 – 展示每个 LLM 参数的结果类型

接下来，我们将学习如何 使用 top-p 与你的模型结合。 这与我们之前使用的其他参数略有不同，因为你必须将其作为 `model_kwargs` 变量的一部分传递，这看起来像这样：

```py
 llm = ChatOpenAI(model_name="gpt-4o-mini",
    model_kwargs={"top_p": 0.5})
```

`model_kwargs` 变量是传递那些没有直接集成到 LangChain 但存在于 LLM 底层 API 中的参数的便捷方式。 top-p 是这个 ChatGPT 模型的参数，但其他模型可能叫法不同或不存在。 务必检查你使用的每个 API 的文档，并使用正确的引用来访问该模型的参数。 现在我们已经了解了帮助定义我们输出中随机性的参数，让我们学习种子设置，这是 旨在帮助我们控制 不可控的随机性。

## 种子

LLM 响应默认是非确定性的 ，这意味着推理结果可能因请求而异。 然而，作为数据科学家，我们通常需要更确定性和可重复的结果。 这些细节似乎相互矛盾，但情况并不一定如此。 OpenAI 和其他人最近已经努力提供一些控制，以便通过提供对种子参数和 `system_fingerprint` 响应字段的访问来实现。

种子是许多涉及生成随机数或随机数据序列的软件应用中的常见设置。 通过使用种子，你仍然可以生成随机序列，但你可以每次都生成相同的随机序列。 这让你能够控制通过 API 调用接收（主要是）确定性的输出。 你可以将种子参数设置为任何你选择的整数，并在你希望获得确定性输出的请求中使用相同的值。 此外，如果你使用种子，即使与其他随机设置（如温度或 top-p）一起使用，你仍然可以（主要是）依赖接收相同的 确切响应。

需要注意的是，即使使用了种子，你的结果仍然可能不同，因为你正在使用一个连接到服务的 API，该服务正在不断进行更改。 这些更改可能导致随着时间的推移结果不同。 例如 ChatGPT 这样的模型在其输出中提供了一个 `system_fingerprint` 字段，你可以将其相互比较，作为系统变化可能引起响应差异的指示。 如果你上一次调用那个 LLM API 时 `system_fingerprint` 值发生了变化，而你当时使用了相同的种子，那么你仍然可能会看到不同的输出，这是由于 OpenAI 对其系统所做的更改造成的。

种子参数也是可选的，并且不在 LangChain 的 LLM 参数集中。 因此，再次强调，就像 `top-p` 参数一样，我们必须通过 `model_kwargs` 参数传递它：

```py
 optional_params = {
  "top_p": 0.5, "seed": 42
}
llm = ChatOpenAI(model_name="gpt-4o-mini", model_kwargs=optional_params)
```

在这里，我们将种子参数与 `top-p` 参数一起添加到我们将传递给 `model_kwargs` 参数的参数字典中。

你可以探索许多其他模型参数，我们鼓励你这样做，但这些参数可能对你的 RAG 应用影响最大。

我们将要讨论的下一个以提示为导向的关键概念是 **镜头** 概念，重点关注你提供给 LLM 的背景信息量。

# 尝试你的镜头

**无镜头**、**单镜头**、**少镜头**和**多镜头** 是在讨论你的提示策略时经常听到的术语。 它们都源自同一个概念，即一个镜头是你给 LLM 的一个例子，以帮助它确定如何回应你的查询。 如果这还不清楚，我可以给你一个 例子来说明我所说的内容。 哦，等等，这正是 镜头概念背后的想法！ 你可以提供没有例子（无镜头）、一个例子（单镜头）或多个 例子（少镜头或多镜头）。 每个镜头都是一个例子；每个例子都是一个镜头。 以下是你对 LLM 可能说的话的例子（我们可以称之为单镜头，因为我只提供了一个 例子）：

```py
 "Give me a joke that uses an animal and some action that animal takes that is funny. Use this example to guide the joke you provide:
Joke-question: Why did the chicken cross the road? Joke-answer: To get to the other side."
```

这里的假设是，通过提供那个例子，你正在帮助引导 LLM 如何 回应。

在 RAG 应用中，你通常会提供上下文中的例子。 这并不总是如此，因为有时上下文只是额外的（但重要的）数据。 然而，如果你在上下文中提供实际的问题和答案的例子，目的是指导 LLM 以类似的方式回答新的用户查询，那么你就是在使用镜头方法。 你会发现一些 RAG 应用更严格地遵循多镜头模式，但这实际上取决于你应用的目标和可用的数据。

在提示中，例子和镜头并不是唯一需要理解的概念，因为你还需要了解指代你如何处理提示的术语之间的差异。 我们将在下一节中讨论这些 方法。

# 提示、提示设计和提示工程回顾

在词汇部分 *第一章*中，我们讨论了这三个概念及其相互作用。 为了复习，我们提供了以下要点：

+   **提示** 是指 发送一个查询或 *提示 到 一个 LLM。

+   **提示设计** 指的是你采取的策略来 *设计 你将发送给 LLM 的提示。 许多不同的提示设计 策略在不同的场景下都有效。

+   **提示工程** 更关注 围绕你使用的提示的技术方面，以改进 LLM 的输出。 例如，你可能将一个复杂的查询分解成两个或三个不同的 LLM 交互， *工程化 它以实现 更优的结果。

我们曾承诺在 *第十三章 *中重新审视这些主题，所以我们现在来履行这个承诺！ 我们不仅将重新审视这些主题，还会向你展示如何在代码中实际执行这些操作。 提示是一个相对直接的概念，所以我们将会关注其他两个主题：设计和工程。

# 提示设计对比工程方法

当我们讨论了在“*射击 *方法 *中”的不同 *射击 *方法，这属于提示设计。 然而，当我们用从 RAG 系统的其他部分提取的问题和上下文数据填写提示模板时，我们也实施了提示工程。 当我们用来自系统其他部分的数据填写这个提示时，你可能记得这被称为“水化”，这是一种特定的提示工程方法。 提示设计和提示工程有显著的交集，因此你经常会听到这两个术语被交替使用。 在我们的案例中，我们将一起讨论它们，特别是它们如何被用来改进我们的 RAG 应用。

在过去几年中，我看到了这些概念以许多不同的方式被描述，因此似乎我们的领域还没有形成对每个概念的完整定义，也没有在它们之间划清界限。 为了理解这本书中的这些概念，我会描述提示设计和提示工程之间的区别是：提示工程是一个更广泛的概念，它不仅包括提示的设计，还包括用户与语言模型之间整个交互的优化和微调。

理论上，有无数种提示设计技术，都可以用来改进你的 RAG 应用。 跟踪你拥有的选项并了解每种方法最适合哪种场景很重要。 需要通过不同提示设计方法的实验来确定哪种最适合你的应用。 提示设计没有一劳永逸的解决方案。 我们将提供一些示例列表，但我们强烈建议你从其他来源了解更多关于提示设计的信息，并注意哪些方法可能有助于你的特定应用： 

+   **镜头设计**：

    +   任何提示设计 思维过程的起点 

    +   涉及精心设计初始提示，使用示例帮助引导 AI 模型达到期望的输出 

    +   可以与其他设计模式结合使用，以增强生成内容的质量和相关性 

+   **思维链提示**：

    +   将复杂问题分解成更小、更易于管理的步骤，在每个步骤中提示 LLM 进行中间推理 

    +   通过提供清晰、逐步的思维过程，提高 LLM 生成答案的质量，确保更好的理解和更准确的响应 

+   **角色（****角色提示）**

    +   涉及创建一个基于用户群体或群体的代表性部分的虚构 角色，包括姓名、职业、人口统计、个人故事、痛点 和挑战

    +   确保输出与目标受众的需求和偏好相关、有用且一致，给内容增添更多个性和风格 

    +   是开发符合用户需求的有效语言模型的有力工具 

+   **密度链** **（摘要）**：

    +   确保 LLM 已经正确地总结了内容，检查是否有重要信息被遗漏，并且摘要是否足够简洁 

    +   在 LLM 遍历摘要时使用实体密度，确保包含最重要的实体 

+   **思维树（探索** **思维）**

    +   从一个初始提示开始，它 生成多个思考选项，并迭代选择最佳选项以生成下一轮 的思考

    +   允许更全面和多样化的探索想法和概念，直到生成 所需的输出文本 。

+   **图提示**

    +   一个专门为处理 图结构数据 而设计的新的提示框架

    +   使 LLM 能够根据图中的实体关系理解和生成内容 。

+   **知识增强**

    +   涉及通过添加额外的、相关的信息来增强提示，以提高 生成内容的 质量和准确性

    +   可以通过将外部知识纳入 提示 的技术，如 RAG 来实现

+   **展示给我而不是告诉我** **提示**

    +   两种向生成式 AI 模型提供指令的不同方法： *展示给我* 涉及提供示例或演示，而 *告诉我* 涉及提供明确的指令 或文档

    +   结合这两种方法提供灵活性，并可能根据特定任务的具体上下文和复杂性提高生成式 AI 响应的准确性 。

这份列表只是触及了表面，因为还有许多其他方法可以用来提高提示工程和生成式 AI 的性能。 随着提示工程领域的持续发展，可能会出现新的创新技术，进一步增强生成式 AI 模型的能力。 。

接下来，让我们谈谈有助于 RAG 应用的提示设计的基本原则。 。

# 提示设计的基本原则

在设计 RAG 应用的提示时，必须牢记以下基本要点 ：

+   `请分析给定上下文并回答问题，考虑到所有相关信息和细节` 可能不如说 `根据提供的上下文，回答以下问题：[``具体问题`。

+   `总结上下文的主要观点，识别提到的关键实体，然后回答给定的问题`，即你同时要求的多项任务。 如果你将这个问题拆分成多个提示，并说类似以下的内容，你可能会得到更好的结果 这样的：

    +   `总结以下` `上下文的主要观点：[上下文]`

    +   `识别以下总结中提到的关键实体：[来自` `先前提示的总结]`

    +   `使用上下文和识别的实体回答以下问题：[``具体问题]`

+   `根据上下文，对主题表达的情感是什么？`，如果你这样说，你可能会得到更好的结果 `根据上下文，将对主题表达的情感分类为正面、负面` `或中性`。

+   `根据提供的上下文回答以下问题`，你可能会想这样说 `以下列出的示例作为指导，根据提供的上下文回答以下问题：示例 1：[问题] [上下文] [答案] 示例 2：[问题] [上下文] [答案] 当前问题：[问题]` `上下文：[上下文]`。

+   `总结文章的主要观点，识别关键实体，并回答以下问题：[问题]。 提供示例，并使用以下格式回答：[格式]。 文章：[长篇文章文本]` 提示可能不如以下多次迭代提示有效： 以下：

    +   `总结以下文章的主要观点：[``文章文本]`

    +   `总结以下文章的主要观点并识别关键实体：[``文章文本]`

    +   `基于总结和关键实体，回答以下问题：[问题] 文章：[``文章文本]`

+   `###` 用于区分指令和上下文部分。 这有助于 AI 模型更好地理解和遵循给定的指令。 例如，与以下提示相比 `[上下文]请使用上述上下文回答以下问题：[问题]。 以简洁的方式提供您的答案` ，您将获得更少的成功 。 `指令：使用以下提供的上下文，以简洁的方式回答问题。 上下文：[上下文]` `问题：[问题]`。

虽然提示设计的根本原则为创建有效的提示提供了坚实的基础，但重要的是要记住，不同的语言模型可能需要特定的调整以达到最佳效果。 让我们接下来讨论这个 话题。

# 为不同的 LLM 调整提示

随着 AI 领域的不断发展，人们不再仅仅依赖于 OpenAI 来满足他们的语言建模需求。 其他玩家，如 Anthropic 及其 Claude 模型，因其处理长上下文窗口的能力而受到欢迎。 Google 也在发布（并将继续发布）强大的模型。 此外，开源模型社区正在迎头赶上，Llama 等模型已被证明是 可行的替代品。

然而，需要注意的是，提示（prompts）并不能无缝地从一种大型语言模型（LLM）转移到另一种。 每种 LLM 可能都有最适合其架构的特定技巧和技术。 例如，Claude-3 在提示时更喜欢使用 XML 编码，而 Llama3 在标记提示的不同部分（如 SYS 和 INST）时使用特定的语法。 以下是一个使用 SYS 和 INST 标签的 Llama 模型的示例提示： 。

+   `<SYS>您是一个 AI 助手，旨在为用户提供有帮助和信息的回答。 问题。 </SYS>`

+   `<INST>分析以下用户的问题，并使用您的知识库提供清晰、简洁的答案。 如果问题不清楚，请` `要求澄清。`

+   用户问题： `"与化石燃料相比，使用可再生能源的主要优势是什么?" ` </INST>

在这个例子中， `SYS` 标签简要确立了 AI 作为辅助工具的角色，旨在提供有用的响应。 `INST` 标签提供了回答用户问题的具体指令，这些指令包含在 `INST` 块中。 **SYS** 用作**系统** 或**系统消息**的简称，而**INST** 用于代替**指令**。

在设计用于 RAG 应用的提示时，考虑与所选 LLM 相关的具体要求和最佳实践至关重要，以确保最佳性能和结果。 所有最著名的模型都有提示文档，可以解释如果你使用它们时需要做什么。 所有最著名的模型都有提示文档，可以解释如果你使用它们时需要做什么。 如果你使用它们。

现在，让我们将本章第一部分所涵盖的所有概念通过一个 代码实验室付诸实践！

# 代码实验室 13.1 – 自定义提示模板

提示模板是一个表示在 LangChain 中管理和使用提示的机制的类。 与大多数模板一样，提供了文本，以及代表模板输入的变量。 使用`PromptTemplate` 包来管理你的提示，确保它在 LangChain 生态系统中运行良好。 此代码基于我们在 *第八章*的 *8.3 代码实验室*中完成的代码，可以在 GitHub 仓库的 `CHAPTER13` 目录中找到，作为`CHAPTER13-1_PROMPT_TEMPLATES.ipynb`。

作为复习，这是我们使用最多的模板：

```py
 prompt = hub.pull("jclemens24/rag-prompt")
```

打印这个提示看起来 像这样：

```py
 You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Question: {question}
Context: {context}
Answer:
```

这存储在你可以打印出来的 `PromptTemplate` 对象中。 如果你这样做，你会看到类似这样的内容：

```py
 ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'jclemens24', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '1a1f3ccb9a5a92363310e3b130843dfb2540239366ebe712ddd94982acc06734'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
```

所以，正如我们 在这里可以看到的，完整的 `PromptTemplate` 对象不仅仅是文本和变量。 首先，我们可以这样说，这是一个名为 `PromptTemplate` 对象的具体版本，称为 `ChatPromptTemplate` 对象，这表明它被设计成在聊天场景中最有用。 输入变量是 `context` 和 `question`，它们在模板字符串本身中稍后出现。 稍后，我们将设计一个自定义模板，但这个特定的模板来自 LangChain hub。 你可以在这里看到与 hub 相关的元数据，包括所有者、仓库和提交哈希。

让我们通过用我们自己的定制模板替换这个提示模板来开始我们的代码实验室。 ：

我们已经导入 LangChain 包用于以下目的： ：

```py
 from langchain_core.prompts import PromptTemplate
```

对于这个，没有必要添加更多的导入！ 我们将替换以下代码： ：

```py
 prompt = hub.pull("jclemens24/rag-prompt")
```

这里是我们将要替换的代码 我们将用以下内容替换它：

```py
 prompt = PromptTemplate.from_template(
    """
    You are an environment expert assisting others in
    understanding what large companies are doing to
    improve the environment. Use the following pieces
    of retrieved context with information about what
    a particular company is doing to improve the
    environment to answer the question. If you don't know the answer, just say that you don't know. Question: {question}
    Context: {context}
    Answer:"""
)
```

正如你所见，我们已经定制了这个提示模板，使其专注于我们的数据（谷歌环境报告）所关注的主题。 我们使用角色提示设计模式来建立一个我们希望 LLM 扮演的角色，我们希望这会使它更符合我们的 特定主题。

提示模板接受一个字典作为输入，其中每个键代表提示模板中的一个变量，用于填充。 `PromptTemplate` 对象输出的结果是 `PromptValue` 变量，可以直接传递给 LLM 或 ChatModel 实例，或者作为 LCEL 链中的一个步骤。

使用以下代码打印出提示对象： ：

```py
 print(prompt)
```

输出将 是 以下内容：

```py
 input_variables=['context', 'question'] template="\n    You are an environment expert assisting others in \n    understanding what large companies are doing to \n    improve the environment. Use the following pieces \n    of retrieved context with information about what \n    a particular company is doing to improve the \n    environment to answer the question. \n    \n    If you don't know the answer, just say that you don't know.\n    \n    Question: {question} \n    Context: {context} \n    \n    Answer:"
```

我们看到它已经捕获了输入变量，而无需我们明确地 声明它们：

```py
 input_variables=['context', 'question']
```

我们可以使用这条命令来打印出文本： ：

```py
 print(prompt.template)
```

这会给你一个比我们之前展示的输出更易于阅读的版本，但只包含 提示本身。

你将注意到，就在这个下面，我们已经有了一个针对确定我们的 相关性分数的定制提示模板：

```py
 relevance_prompt_template = PromptTemplate.from_template(
    """
    Given the following question and retrieved context, determine if the context is relevant to the question. Provide a score from 1 to 5, where 1 is not at all relevant and 5 is highly relevant. Return ONLY the numeric score, without any additional text or explanation. Question: {question}
    Retrieved Context: {retrieved_context}
    Relevance Score:"""
)
```

如果你运行剩余的代码，你可以看到它对 RAG 应用最终结果的影响。 这个提示模板被认为是字符串提示模板，这意味着它是使用包含提示文本和动态内容占位符的普通字符串创建的（例如， `{question}` 和 `{retrieved_context}`）。 你还可以使用 `ChatPromptTemplate` 对象进行格式化，该对象用于格式化消息列表。 它由一系列 模板本身组成。

提示模板 在最大化 RAG 系统性能中扮演着关键支持角色。 在本章剩余的代码实验室中，我们将使用提示模板作为主要元素。 然而，我们现在将重点转移到我们在编写提示时需要记住的一系列概念上。 我们的下一个代码实验室将专注于所有这些概念，从 提示格式化开始。

# 代码实验室 13.2 – 提示选项

此代码可在 的 `CHAPTER13-2_PROMPT_OPTIONS.ipynb` 文件中找到，位于 `CHAPTER13` 目录下的 GitHub 仓库。

通常，当你接近设计你的提示时，会有许多一般概念是你希望记住的。 这些包括迭代、总结、转换和扩展。 这些概念各有不同的用途，并且它们通常可以以各种方式组合。 当你改进你的 RAG 应用时，了解如何设计你的提示的基础知识将非常有用。 我们将通过一个真实场景来介绍不同的提示方法，在这个场景中，你正在帮助公司市场部门编写提示。 我们将从 迭代开始。

## 迭代

这个概念简单来说就是迭代 你的提示以获得更好的结果。 你的第一个提示很少会是最佳和最终的提示。 本节将重点介绍一些基本技巧和概念，以帮助您快速迭代提示，使其更适合您的 RAG 应用。

## 迭代语气

你的老板刚刚打电话。 他们告诉你营销人员表示他们想要在他们的营销材料中使用 RAG 应用程序的输出，但必须以更营销事实表的形式提供。 没问题；我们可以在那里直接设计提示设计！

在第一个提示之后 添加第二个提示：

```py
 prompt2 = PromptTemplate.from_template(
    """Your task is to help a marketing team create a
    description for the website about the environmental
    initiatives our clients are promoting. Write a marketing description based on the information
    provided in the context delimited by triple backticks. If you don't know the answer, just say that you don't know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

然后你需要将 `rag_chain_from_docs` 链中的提示更改为 `prompt2`。查看 `RunnablePassthrough()` 行之后的代码：

```py
 rag_chain_from_docs = (
    …
             "answer": (
                RunnablePassthrough()
                | prompt2 # <- update here | llm
                | str_output_parser
            )
        …
)
```

然后，重新运行 从 `prompt2` 向下到 这个结果：

```py
 Google is at the forefront of environmental sustainability, leveraging its technological prowess to drive impactful initiatives across various domains. Here are some of the key environmental initiatives that Google is promoting: Empowering Individuals: Google aims to help individuals make more sustainable choices through its products. In 2022, Google reached its goal of assisting 1 billion people in making eco-friendly decisions. This was achieved through features like eco-friendly routing in Google Maps, energy-efficient settings in Google Nest thermostats, and carbon emissions information in Google Flights. By 2030, Google aspires to help reduce 1 gigaton of carbon equivalent emissions annually. …[TRUNCATED FOR BREVITY]… By organizing information about the planet and making it actionable through technology, Google is helping to create a more sustainable future. The company's efforts span from individual empowerment to global partnerships, all aimed at reducing environmental impact and fostering a healthier planet.
```

如果你阅读 完整的输出，你会注意到这确实更偏向于营销导向。 这可能正是营销团队所寻找的。 然而，你刚刚记得你的老板也说过这将被放在网站上只能容纳 50 个单词的小方块里 最多！

## 缩短长度

对于 `prompt3`，我们只需要添加 这个小的片段： `最多使用 50 个单词`。看起来是这样的：

```py
 prompt3 = PromptTemplate.from_template(
    """Your task is to help a marketing team create a
    description for the website about the environmental
    initiatives our clients are promoting. Write a marketing description based on the information
    provided in the context delimited by triple backticks. If you don't know the answer, just say that you don't know. Use at most 50 words. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

更新链中的提示为 `prompt3`。运行剩余的代码，你将得到 以下输出：

```py
 Google's environmental initiatives include promoting electric vehicles, sustainable agriculture, net-zero carbon operations, water stewardship, and a circular economy. They aim to help individuals and partners reduce carbon emissions, optimize resource use, and support climate action through technology and data-driven solutions.
```

营销团队 喜欢你的工作！ 一切都很顺利！ 做得好！ 时间 过得真快…

一个月后，决定与其关注你所有客户的环保努力，不如专注于 与技术相关的努力。

## 改变焦点

我们希望 LLM 关注更具体于技术的方面。 这是为了小册子，所以它可以再次变长。 让我们设计 `prompt4` 来指导 LLM 改变其焦点，我们将取消 长度限制：

```py
 prompt4 = PromptTemplate.from_template(
    """Your task is to help a marketing team create a
    description for the website about the environmental
    initiatives our clients are promoting. Write a marketing description based on the information
    provided in the context delimited by triple backticks. The description is intended for a technology audience,
    so this should focus on only the aspects of the
    company's efforts that relate to using technology. If
    you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

再次提醒 更新链中的提示为 `prompt4` ，然后运行剩余的代码以获取 以下输出：

```py
 Google is at the forefront of leveraging technology to drive environmental sustainability. Here are some of the key initiatives that highlight their commitment to using technology for a greener future: Eco-Friendly Product Features: Google Maps: Introduced eco-friendly routing, which has helped prevent over 1.2 million metric tons of carbon emissions since its launch. Google Nest: Energy efficiency features in Google Nest thermostats help users reduce their energy consumption. Google Flights: Provides carbon emissions information to help travelers make more sustainable choices. …TRUNCATED FOR BREVITY…
Sustainability-Focused Accelerators: Google for Startups Accelerator: Supports early-stage innovations aimed at tackling sustainability challenges, fostering the growth of technologies that can positively impact the planet. Google's comprehensive approach to environmental sustainability leverages their technological expertise to create significant positive impacts. By integrating sustainability features into their products, optimizing their operations, and collaborating with partners, Google is driving forward a more sustainable future.
```

同样，我们在这里的书中不得不缩短它，但如果你在代码中查看，结果令人印象深刻。 显然，对他们的环境方面的技术方面有更高的关注。 你的营销团队 印象深刻！

这是一个有趣的例子，但这并不远离构建这些类型系统时发生的事情。 在现实世界中，你可能会迭代更多次，但采用迭代方法来设计提示将帮助你达到更优的 RAG 应用，就像你的 RAG 系统的任何其他部分一样。

接下来，让我们 谈谈如何将大量数据压缩成更小的数据量，这通常被称为摘要。

## 摘要

摘要 是 RAG（检索增强生成）的一个流行用途。 将公司内部的大量数据消化成更小、更简洁的信息，可以是一种快速简单的方法来提高生产力。 这对于依赖信息或需要跟上快速变化信息的职位尤其有用。 我们已经看到了如何设计一个提示来使用字数限制，这在 `prompt3`中。 然而，在这种情况下，我们将更多地关注 LLM 的摘要内容，而不是试图成为一个专家或撰写营销文章。 代码如下： 如下：

```py
 prompt5 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. Summarize the retrieved context below, delimited by
    triple backticks, in at most 30 words. If you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

更新链中的 `prompt5` ，然后再次运行剩余的代码。 结果如下：

```py
 Google's environmental initiatives include achieving net-zero carbon, promoting water stewardship, supporting a circular economy, and leveraging technology to help partners reduce emissions.
```

好的，这很好，简短，并且是摘要。 只有事实！

下一个例子是 另一种我们可以关注 LLM 的情况，但增加了摘要的努力。

## 专注于摘要的摘要

对于 `prompt6`，我们将 保留之前提示中的大部分内容。 然而，我们将尝试将 LLM 特别集中在他们产品的环保方面：

```py
 prompt6 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. Summarize the retrieved context below, delimited by
    triple backticks, in at most 30 words, and focusing
    on any aspects that mention the eco-friendliness of
    their products. If you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

将链中的提示更新为 `prompt6`，然后运行代码以获得 以下输出：

```py
 Google's environmental initiatives include eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions information in Google Flights.
```

这是简短的，如果您将其与更冗长的描述进行比较，它似乎确实专注于 PDF 中展示的产品。 这是一个相当好的结果，但通常当您请求摘要时，即使您将 LLM 聚焦于特定方面，LLM 仍然可能包含您不希望包含的信息。 为了应对这种情况，我们 转向 *extract* 方法。

## extract instead of summarize

如果您遇到常见的总结包含过多不必要信息的问题，尝试使用单词 *extract* 而不是 *summarize*。这看起来可能只是细微的差别，但它对 LLM 来说可能有很大的影响。 *Extract* 给人一种您正在提取特定信息的印象，而不是仅仅试图捕捉整个文本中的整体数据。 LLM 不会错过这个细微差别，这可以是一个很好的技巧，帮助您避免总结有时带来的挑战。 我们将考虑到这个变化来设计 `prompt7` ：

```py
 prompt7 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. If you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{context}```py
    Answer:"""
)
```

将链中的提示更新为 `prompt7`，然后运行代码以获取 以下输出：

```py
 Google's environmental initiatives include eco-friendly routing in Google Maps, energy efficiency features in Google Nest thermostats, and carbon emissions information in Google Flights to help users make sustainable choices.
```

这与 `prompt6`的响应略有不同，但我们已经得到了一个很好的聚焦结果。 当您的总结响应包含不必要的数据时，尝试这个技巧来帮助提高 您的结果。

迭代和总结并不是提高提示努力所需要理解的唯一概念。 我们将接下来讨论如何利用您的 RAG 应用从您的 现有数据中推断信息。

## 推理

在推理的根源，您是 要求模型查看您的数据并提供某种额外的分析。 这通常涉及提取标签、名称和主题，甚至确定文本的情感。 这些能力对 RAG 应用具有深远的影响，因为它们使那些不久前被认为仅属于人类读者领域的工作成为可能。 让我们从一个简单的布尔式情感分析开始，其中我们考虑文本是积极的 还是消极的：

```py
 prompt8 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. After this summary, determine what the sentiment
    of context is, providing your answer as a single word,
    either "positive" or "negative". If you don't know the
    answer, just say that you don't know. Question: {question}
    Context: ```{context}```py
    Answer:"""
)
```

在这段代码中，我们基于前一个提示的总结，但增加了一个 *分析* LLM 正在消化的数据的情感 在这种情况下，它确定情感为 积极 ：

```py
 Google is enhancing eco-friendliness through features like eco-friendly routing in Maps, energy-efficient Nest thermostats, and carbon emissions data in Flights, aiming to reduce emissions significantly. Sentiment: positive
```

在类似的分析领域中，另一个常见的应用是从 上下文中 提取特定数据。

## 提取关键数据

作为一个参考点，你现在被要求识别客户在其与环境努力相关的文档中提到的任何特定产品。 在这种情况下，谷歌（客户）有许多产品，但在这个文档中只提到了其中的一小部分。 你将如何快速提取这些产品并识别它们？ 让我们用 我们的提示 来试试：

```py
 prompt9 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. After this summary, determine any specific products
    that are identified in the context below, delimited
    by triple backticks. Indicate that this is a list
    of related products with the words 'Related products: '
    and then list those product names after those words. If you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

在这段代码中，我们继续基于之前的提示进行构建，但不是要求情感分析，而是要求与我们所检索到的文本相关的产品列表。 我们使用的 GPT-4o-mini 模型成功地遵循了这些指示，列出了文本中特别命名的每个 产品：

```py
 Google is enhancing eco-friendliness through products like eco-friendly routing in Google Maps, energy efficiency features in Google Nest thermostats, and carbon emissions information in Google Flights. Related products: Google Maps, Google Nest thermostats, Google Flights
```

再次，LLM 能够处理我们所要求的一切。 然而，有时我们只想对主题有一个整体的感觉。 我们将使用 LLM 来讨论推理的概念。

## 推断主题

你可能认为这是一个极端的 总结案例。 在这个例子中，我们正在将数千个单词总结成一组简短的主题。 这能行吗？ 让我们试试！ 我们将从 以下代码 开始：

```py
 prompt10 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. After this summary, determine eight topics that are
    being discussed in the context below delimited
    by triple backticks. Make each item one or two words long. Indicate that this is a list of related topics
    with the words 'Related topics: '
    and then list those topics after those words. If you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

在这里，我们使用与之前提示类似的方法，但不是要求产品列表，而是要求与我们所检索到的文本相关的至少八个主题列表。 再次，我们使用的 GPT-4o mini 模型成功地遵循了这些指示，列出了八个 与文本特别相关的、高度相关的主题：

```py
 Google is enhancing eco-friendliness through products like eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions information in Google Flights. Related topics:
1\. Electric vehicles
2\. Net-zero carbon
3\. Water stewardship
4\. Circular economy
5\. Supplier engagement
6\. Climate resilience
7\. Renewable energy
8\. AI for sustainability
```

我们已经涵盖了迭代、总结和推理，这些都显示出极大的潜力来提高你的提示效果。 我们将要介绍的概念还有 转换。

## 转换

转换 是将您当前的数据转换 成不同的状态或格式。 一个非常常见的例子是语言翻译，但还有许多其他情况，包括将数据放入某种编码格式，如 JSON 或 HTML。 您还可以应用转换，如检查拼写或 语法错误。

我们将从 语言翻译开始。

### 语言转换（翻译）

营销部门又 打电话来了。 你到目前为止所做的工作非常出色，但现在事情正在加速，我们将走向国际市场了！ 我们选择的第一批国际市场包括西班牙语和法语的使用者。 我们公司的一位新投资者也是任何与海盗有关的东西的大粉丝，所以是的，我们也将涵盖这种方言！ 既然我们在谈论 *转换*，我们称之为 语言转换，但在这个上下文中，看到术语 *翻译* 也很常见。 让我们 开始吧：

```py
 prompt11 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. Translate the summary into three additional languages,
    Spanish, French, and English Pirate:
    labeling each language with a format like this:
    English: [summary]
    Spanish: [summary]
    French: [summary]
    English pirate: [summary]
    If you don't know the answer, just say that you don't
    know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

在这段代码中，我们基于之前的提示，但随后要求 LLM 生成四种不同版本的简短摘要：英语、西班牙语、法语和英语海盗语。 显然，我们应该更经常地说海盗话，因为这是 最 有趣的：

```py
 English: Google enhances eco-friendliness through features like eco-friendly routing in Maps, energy-efficient Nest thermostats, and carbon emissions info in Flights, helping reduce carbon emissions significantly. Spanish: Google mejora la eco-amigabilidad con funciones como rutas ecológicas en Maps, termostatos Nest eficientes en energía e información de emisiones de carbono en Flights, ayudando a reducir significativamente las emisiones de carbono. French: Google améliore l'éco-responsabilité avec des fonctionnalités telles que les itinéraires écologiques dans Maps, les thermostats Nest économes en énergie et les informations sur les émissions de carbone dans Flights, aidant à réduire significativement les émissions de carbone. English pirate: Google be makin' things greener with eco-routes in Maps, energy-savin' Nest thermostats, and carbon info in Flights, helpin' to cut down on carbon emissions mightily.
```

语言翻译是 RAG 的一个流行用途，但还有其他转换可能很有用。 让我们回顾一下向摘要添加不同语气的例子。

### 语气转换

我们到目前为止的努力 在撰写摘要甚至 营销文案方面都取得了成功，但现在我们需要扩展到其他渠道，如电子邮件，并给我们的摘要一个更友好的语气以适应这种格式。 为此，我们将应用 语气转换：

```py
 prompt12 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. After providing the summary, translate the summary
    into an email format with a more friendly and
    casual tone. If you don't know the answer, just say
    that you don't know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

在这里，我们继续原来的 总结，但随后我们要求 LLM 使用相同的信息写一封 轻松 语气的电子邮件：

```py
 Google is enhancing eco-friendliness through features like eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions data in Google Flights. Email Format: Subject: Exciting Eco-Friendly Features from Google! Hi [Recipient's Name],
I hope you're doing well! I wanted to share some cool updates from Google about their efforts to help the environment. They've introduced some awesome features like eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and even carbon emissions data in Google Flights. It's great to see such big steps towards a greener future! Best, [Your Name]
```

正如我们通过这些例子所看到的，LLM 可以以多种方式用于改进 RAG 应用。 在我们的例子中不太适用（但在其他场景中仍然非常有价值）的其他选项包括翻译成特定的编码格式，或者从一个编码格式转换到另一个编码格式。 拼写和语法检查也是可以应用的热门转换。 这些。 可以。 应用。

我们已经涵盖了迭代、总结、推理和转换。 为了结束这个代码实验，我们还将介绍一个概念。 扩展。

## 扩展

从许多方面来说，扩展的目标可以被视为我们之前介绍的概念（如总结）目标的一种反向。 在总结中，我们正在将大量数据合并成更小的数据量，同时试图保留数据的含义。 扩展试图做的是相反的，它将一小部分数据扩展到更大的信息集。 让我们通过一个可以实施这种扩展的场景来了解一下：简短总结的扩展。 扩展 它 让我们通过一个可以实施这种扩展的场景来了解一下：简短总结的扩展。 简短总结的扩展。

### 扩展简短文本

我们的努力持续增长！ 我们最新的任务是向我们的客户推广我们一直在讨论的环境概念的好处。 在接下来的提示中，我们将根据我们得到的简短总结，暂时假设这就是我们拥有的全部内容。 然后我们将要求 LLM 在此基础上进行扩展，重点关注它如何吸引投资者。 让我们看看它能提出什么： ：

```py
 prompt13 = PromptTemplate.from_template(
    """Your task is to generate a short summary of what a
    company is doing to improve the environment. From the retrieved context below, delimited by
    triple backticks, extract the information focusing
    on any aspects that mention the eco-friendliness of
    their products. Limit to 30 words. After providing the summary, provide a much longer
    description of what the company is doing to improve
    the environment, using only the summary you have
    generated as the basis for this description. If you
    don't know the answer, just say that you don't know. Question: {question}
    Context: ```{上下文}```py
    Answer:"""
)
```

在现实世界的场景中，当你希望 LLM 扩展你的数据时，你很可能大部分情况下都没有原始的上下文。 因此，通过告诉 LLM 将其扩展的来源仅限于总结中提供的内容，我们更准确地模拟了这种场景。 我们的 LLM 没有让我们失望，因为它提供了对原始总结中提到的环境努力的扩展描述，并针对投资者进行了调整。 然后，它针对投资者进行了调整：

```py
Summary: Google offers eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions information in Google Flights to help users make sustainable choices. Broader Description: Google is actively enhancing the eco-friendliness of its products by integrating features that promote sustainability. For instance, Google Maps now includes eco-friendly routing options, Google Nest thermostats are designed for energy efficiency, and Google Flights provides carbon emissions information. These initiatives not only help users make more environmentally conscious decisions but also demonstrate Google's commitment to reducing its carbon footprint. For investors, this focus on sustainability can be a significant value proposition, as it aligns with the growing consumer demand for eco-friendly products and can lead to long-term cost savings and regulatory advantages. Additionally, it positions Google as a leader in environmental responsibility, potentially enhancing its brand reputation and market share.
```

这只是扩展概念可以应用的一个例子。 考虑一下在你的 RAG 应用中如何以及何时利用数据扩展。 以及。

这总结了如何改进你的提示设计的关键概念：迭代、总结、推断、转换和扩展。 这些概念构成了许多更深入和复杂概念的基础，可以使你的 RAG 应用更加有效。 把这视为你在该领域的知识起点，并继续跟踪该领域的进步和新技术，随着它们的被发现。

# 摘要

在本章中，我们探讨了提示工程在增强 RAG 系统性能和有效性中的关键作用。 通过战略性地设计和精炼输入提示，我们可以提高相关信息的检索，从而提高生成文本的质量。 我们讨论了各种提示设计技术，例如射击设计、思维链提示、角色扮演和知识增强，这些技术可以应用于优化 RAG 应用。

在本章中，我们讨论了提示设计的根本概念，包括简洁、具体和明确的重要性，以及逐步迭代和使用清晰分隔符的必要性。 我们还强调了不同 LLM 需要不同的提示，以及适应特定模型使用的提示的重要性。

通过一系列代码实验室，我们学习了如何使用 LangChain 中的`PromptTemplate` 类创建自定义提示模板，以及如何将各种提示概念应用于提高我们的 RAG 工作。 这些概念包括迭代以精炼提示、总结以浓缩信息、推断以提取额外见解、将数据转换为不同的格式或语气，以及扩展简短摘要以生成更全面的描述。 我们还探讨了使用提示参数，如温度、top-p 和种子，来控制 LLM 输出的随机性和确定性。

通过利用本章介绍的技术和概念，我们可以显著提高 RAG 应用的表现，使它们在检索相关信息、生成高质量文本和适应特定用例方面更加有效。 随着提示工程领域的不断发展，跟上最新的技术和最佳实践对于在各个领域最大化 RAG 系统的潜力至关重要。

在我们下一章和最后一章中，我们将讨论一些更高级的技术，你可以使用这些技术对你的 RAG 应用进行潜在的显著改进！
