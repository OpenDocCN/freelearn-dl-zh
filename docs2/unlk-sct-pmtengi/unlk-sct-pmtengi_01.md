# 1

# 理解提示和提示技术

本书首先考察**大型语言模型**（**LLMs**）的基础，包括其结构和响应生成机制。然后，它将探讨各种提示工程技术，如对初始想法进行精炼和迭代，以制作出能够从 LLM 中诱发出期望输出的有效提示。本书还将讨论使用 LLMs 的伦理考量，包括减轻偏见和确保公平、透明和问责的方法。

本书展示了 LLMs 在各个领域的创新应用，以展示它们在改善生活和重塑行业方面的巨大潜力。我们将通过观察新兴趋势、潜在突破以及合作在推动进步中的作用来思考 LLMs 的未来。

在本书中，将提供实际案例、案例研究和动手练习，以提供关于提示工程的全面学习体验。目标是让你成为一个熟练的提示工程师，能够利用 LLMs 的力量创造有意义的、积极的变化。

本章旨在提供对 AI 提示及其重要性的清晰理解。通过探索 LLMs（如 GPT-4）等内部工作原理，结合提示，你将获得有效利用其力量的实用知识和技巧。

在本章中，你将做以下事情：

+   **发现提示的内部机制**：本章将揭示驱动 LLM 提示的组件，从输入提示到上下文和响应的作用。通过检查每个元素，你将了解它们如何塑造语言模型生成的输出。

+   **探索丰富的提示类型**：本章将深入探讨提示技术的多样化领域，提供关于不同类型提示的见解。实际案例将展示如何在不同场景中使用提示来实现期望的结果。

+   **导航提示的挑战和限制**：虽然 LLM 提示具有巨大潜力，但也伴随着挑战和限制。本章将讨论常见障碍并提供克服它们的见解。通过理解这些限制，你将更好地准备在实施提示工程策略时做出明智的决定。

在本章结束时，你将拥有提示工程的良好基础。凭借所获得的知识和技能，你将准备好在后续章节探索高级技术和应用时，充分利用 LLM 提示的潜力。

本章将涵盖以下主题：

+   介绍 LLM 提示

+   LLM 提示的工作原理

+   LLM 提示的类型

+   LLM 提示的组成部分

+   角色提示

+   语音定义

+   使用模式来增强提示的有效性

+   探索一些结合提示工程技术的示例

+   探索 LLM 参数

+   如何进行提示工程（实验）

+   使用 LLM 提示的挑战和限制

# 技术要求

要能够在这个章节中玩转提示，你需要使用以下工具之一或多个创建一个账户：

+   **OpenAI** **ChatGPT**: [`chat.openai.com/`](https://chat.openai.com/).

    一旦你创建了账户（如果你还没有创建的话），你可以升级到 ChatGPT Plus，每月 20 美元。这不是利用这本书所必需的。

+   **Google** **Bard**: [`bard.google.com/`](https://bard.google.com/).

    如果你已经登录了 Google Workspace 账户，你的管理员可能没有启用对 Bard 的访问权限。如果是这种情况，请要求你的管理员启用访问权限。

+   **Anthropic** **Claude**: [`console.anthropic.com/`](https://console.anthropic.com/)

我们将从 LLM 提示的介绍开始，了解它们是如何工作的，以及我们可以使用哪种类型的提示来产生最佳结果。

# 介绍 LLM 提示

LLMs 已经彻底改变了我们与技术互动的方式，塑造了我们的数字景观，并改变了我们沟通、学习和创新的方式。在这场革命的最前沿是像 OpenAI GPT-4、Google LaMDA 和 Anthropic 这样的大规模 LLM。这些语言模型可以通过它们的 API 或通过它们的聊天界面程序化地使用。这些界面分别被称为 OpenAI ChatGPT、Google Bard 和 Anthropic Claude。

提示工程是为 LLM 设计和完善输入提示以实现所需输出的过程。虽然它可能最初看起来微不足道或纯粹是技术性的，但实际上，它是一个多学科的努力。它要求对语言学、认知科学、**人工智能**（**AI**）、用户体验设计和伦理学等多个领域的全面理解。本章的目标是提供一个全面的框架来理解和掌握提示工程的复杂性，并为你提供利用 LLMs 全部潜力的必要工具。

# LLM 提示是如何工作的

大规模 LLM 是一种专注于理解和生成人类语言的 AI 形式。它们使用复杂的机器学习算法，主要是神经网络，来处理和分析大量文本数据。LLM 的主要目标是针对给定的输入提示产生连贯、上下文相关且类似人类的响应。要理解 LLM 是如何工作的，讨论其底层架构和训练过程至关重要。使用一些类比来解释这些概念将使它们更容易理解。

## 架构

LLMs，如 OpenAI 的 GPT-4，是使用一种称为 Transformer 的特殊类型的神经网络制作的。Transformer 具有一种特殊的结构，有助于它们很好地处理文本。

Transformer 中的一个重要特点是自注意力。这意味着模型可以关注句子的不同部分，并决定在特定上下文中哪些词更重要。这就像关注最重要的词。

另一个 Transformer 使用的是位置编码。这有助于模型跟踪每个词在句子中的位置。这就像给每个词一个特殊的标签，以便模型知道它在序列中的位置。

带有这些特性，LLM 可以很好地处理和理解长篇文本。它们可以根据它们出现的上下文理解单词的意义，并记住句子中单词的顺序。

## LLM 训练

LLM 的训练过程包括两个主要阶段：预训练和微调。LLM 像极富语言技巧的学生。在训练过程中，它们会经历这两个主要阶段。

### 预训练

在这个第一阶段，LLM 被暴露于来自书籍、文章、网站等的大量文本。这就像它有机会阅读一个充满各种信息的巨大图书馆。

当 LLM 审查所有这些文本时，它开始注意到语言结构的模式。LLM 学习以下内容：

+   哪些单词倾向于相互跟随（“dog”后面跟“bark”的概率）

+   不同语言的语法和句子结构（动词在句子中的位置）

+   某些单词相关的主题和概念（学习“dog”和“puppy”与动物、宠物等相关联）

为了处理所有这些文本，LLM 将其分解成更小的、易于消化的部分，有点像将语言咀嚼成一口大小的块。这个过程称为分块。

LLM 将句子分割成更小的部分，称为标记。标记可以是单个单词、部分单词，甚至是特殊字符，如标点符号。

在将文本分割成块之后，LLM 将每个标记嵌入或编码成一个数值向量，这就像给每个标记一个数学表示——例如，将 *dog* 翻译成类似 *[0.51, 0.72, 0.33,...]* 的形式，以便计算机处理。这个过程称为嵌入。

这就像将句子从英语翻译成数字。不是用单词，每个标记现在都有一个对应的数字向量，计算机可以理解。

这个嵌入过程基于 LLM 从其广泛的预训练中学习到的模式，捕捉每个标记的意义信息。具有相似意义的标记在向量空间中嵌入得更近。

所有这些数值标记向量都存储在 LLM 的向量数据库中，它可以在以后查找标记并分析它们与其他标记之间的关系。这个向量数据库就像是 LLM 的数学图书馆索引。

因此，在预训练中，LLM 通过分析大量文本并在其复杂的神经网络大脑中存储模式，在单词和概念之间建立联系。因此，LLM 存储了“*狗*”和“*小狗*”具有相似的向量表示，因为它们具有相关的意义和上下文。

然而，“*狗*”和“*自行车*”在语义上不同，因此它们之间的距离更远。向量空间根据它们的相似性和差异性组织单词。

### 微调

在预训练之后，LLM 进入微调阶段。在这里，它会在与特定任务相关的小型数据集上接受额外的训练。

这就像 LLM 在完成通识教育后专注于特定的研究领域——例如，在掌握科学基础知识后学习高级生物学课程。

在微调过程中，大型语言模型（LLM）根据标记的示例数据练习生成特定任务的输出。标记数据是指已经用标签进行注释的数据，这些标签将内容分类或描述。这些标签通过提供预期输出的示例来帮助训练模型。

当你后来向 LLM 提供新的提示时，它使用在预训练和微调中学习的模式来分析提示并生成合适的响应。

LLM 并不真正像人类一样理解语言。但通过从其训练中的大量示例中识别模式，它可以模仿类似人类的响应，并成为一个高度胜任的语言学习者。

此外，这些向量表示可以用于各种自然语言处理任务，如情感分析、主题建模和文档分类。通过比较单词或短语的向量，算法可以确定它们所代表的概念的相似性或相关性，这对于许多高级语言理解和生成任务至关重要。

对于所有模型来说，一个关键因素是上下文窗口——模型一次可以考虑的文本量——这会影响交互中的连贯性和深度。特别是，上下文窗口对于以下原因很重要：

+   **连贯性和相关性**：更大的上下文窗口允许模型保持对话或文档的线索，从而产生更连贯和上下文相关的响应。

+   **文本生成**：对于撰写文章、故事或代码等任务，更大的上下文窗口使模型能够生成与前一部分内容一致的内容。

+   **对话深度**：在对话系统中，更大的上下文窗口允许 AI 记住并引用对话的早期部分，从而创建更吸引人和自然的交互。

+   **知识检索**：对于需要参考大量文本或从文档的多个部分提取信息的任务，更大的上下文窗口允许模型更有效地交叉引用和综合信息。

然而，存在权衡，因为更大的上下文窗口需要更多的计算能力和内存来处理。这可能会影响响应时间和成本。这是 LLM 之间差异的关键领域，因为上下文窗口的改进可以显著提高模型在复杂任务中的可用性和适应性。

Claude 2 的上下文窗口为 100,000 个标记，而新的 GPT-4-turbo-1106- 评测的上下文窗口为 128,000 个标记。在英语中，每 1,000 个标记的平均单词数约为 750。研究人员预测到 2024 年将出现一百万以上标记的模型。

## 从提示到回复的旅程——推理如何帮助 LLM 填补空白

一旦 LLM 完成训练，它就准备好开始生成对用户提供的提示的响应。

当用户输入一个提示时，该提示被输入到 LLM 的神经网络大脑中。LLM 在其大脑架构中具有特殊组件，有助于分析提示。

其中一部分特别关注上下文中最相关的单词，有点像我们在阅读时关注关键词一样。

另一个组件记得单词的顺序以及它们在提示中的位置，这对于正确获取上下文很重要。

使用其大脑组件，LLM 生成一个可能逻辑上接下来在响应中出现的单词列表。它为每个可能的下一个单词分配一个概率分数。

然后，LLM 使用一种称为解码的技术来选择最佳单词选项，并将它们转换为其最终响应。

它可能会贪婪地选择最有可能的下一个单词。或者它可能从几个最可能的候选人中随机选择，以使响应不那么重复，听起来更像人类。

因此，总的来说，LLM 的特殊大脑架构帮助它关注正确的单词，记住它们的顺序，并为下一个单词分配概率。然后，它将最佳选择解码成符合提示的响应。

这个过程使得 LLM 能够生成非常类似人类的响应，根据用户提供的初始提示合理地继续对话。

LLM 的一个关键优势是它们能够执行 *少量样本* 或 *零样本* 学习。这意味着它们可以从预训练阶段的知识中推广，并快速适应新的任务或领域，而无需额外的训练数据。在少量样本学习中，模型被提供少量示例来学习，而在零样本学习中，模型仅依赖于其预存在的知识和给定的提示来生成响应。

LLM 在自然语言理解和生成任务上取得了显著的进步，应用范围涵盖多个领域，如对话 AI、内容创作、翻译、问答系统等。然而，必须认识到 LLM 并非没有局限性。它们有时可能会产生错误或不合逻辑的答案，对输入措辞的微小变化敏感，或表现出训练数据中的偏差。因此，提示工程在减轻这些局限性并确保 LLM 为特定任务或应用生成所需输出方面发挥着至关重要的作用。

在下一节中，我们将探讨不同类型的 LLM 提示。了解这些不同类型的提示将为您提供宝贵的见解，了解如何有效地与语言模型互动，使您能够为所需的特定应用或任务生成更准确和定制的响应。

# LLM 提示类型

LLM 提示可以根据多个方面进行分类，例如它们的目的、格式和提供的详细程度。

将它们进行分类的一种方式是根据所提示信息的类型：

+   **零样本提示**：这是一种允许 LLM 对它未特别训练过的任务生成响应的技术。在这个技术中，LLM 被提供了一段输入文本和一个用自然语言描述模型预期输出的提示。然后 LLM 使用其知识生成一个与提示一致的响应。

    例如，如果你向 LLM 提供输入文本“写一首关于爱的诗”，以及提示“这首诗应该是美丽而浪漫的”，LLM 可能会生成以下响应：

    ```py
    Love is a many splendored thing,
    It's the April rose that only grows in the early spring.
    Love is nature's way of giving,
    A reason to be living.
    Love is a many splendored thing,
    It's the golden crown that makes a man a king.
    Love is the April rose that only grows in the early spring.
    Love is nature's way of giving,
    A reason to be living.
    ```

+   **微调**：在微调中，LLM 被训练在一个小型标记示例数据集上，用于它被要求执行的任务。这可能是一种更有效的方法来生成高质量的响应，但它需要一个更大的标记示例数据集。

    例如，假设输入是一篇关于治疗癌症的新研究的文章，而训练数据是带有摘要标签的癌症研究文章。

    这里是这个提示。

    提示：

    ```py
    Summarize the key findings of the research described in the input article.
    A short summarization highlighting the key points of the input article.
    ```

+   **数据增强**：在数据增强中，LLM 被训练在一个通过向现有数据添加噪声或变化来人工扩展的数据集上。这可能是一种提高 LLM 在有限标记数据可用任务上的性能的方法。

    例如，假设原始训练数据由关于天气的 1,000 个句子组成，而增强数据由通过添加同义词、改写和引入原句中的错别字创建的 2,000 个额外句子组成。

    这里是提示内容：

    ```py
    Classify each sentence as describing sunny, rainy, or snowy weather.
    Weather classification predictions on the expanded training set.
    ```

+   **主动学习**：在主动学习中，也称为“少样本学习”，LLM 被提供少量标记示例，然后被要求识别最有信息量的示例进行标记。这可能是一种更有效的训练 LLM 的方法，因为它专注于标记那些将最有帮助于提高模型性能的示例。

    微调是在所有对话中全局进行的。在主动学习中，用户在提示时提供一些示例，以获得遵循某种模式的输出。

    例如，如果输出是“这个人的名字是 Peter，他 23 岁”，用户可能希望得到一个响应为*{“name”: “Peter”, “age”: 23 }*。在这种情况下，用户将提供一些原始响应的示例，并明确要求 LLM 以 JSON 格式生成输出，如之前所示。

+   **迁移学习**：在迁移学习中，LLM 被训练在与其被要求执行的任务相似的任务上。这可能是一种更有效的方法来生成高质量的响应，因为它允许 LLM 从更大的数据集中学习。

    例如，假设原始任务是电影评论的情感分析，而新任务是产品评论的情感分析。

    这是提示：

    ```py
    Classify the sentiment of these new product reviews, transferring knowledge from movie review sentiment analysis.
    ```

    这里是输出结果：

    ```py
    Sentiment classification predictions on product reviews, leveraging the capabilities gained from the movie review dataset.
    ```

使用的最佳提示类型将取决于 LLM 被要求执行的具体任务以及可用的标记数据量。

另一种按功能分类提示类型的方法是：

+   **指令提示**：这些提示明确指示模型执行特定任务，例如总结文本、翻译句子或回答问题。指令提示通常以明确的指令开始，例如“将以下句子翻译成法语：”或“总结以下段落：”。

+   **会话提示**：这些提示旨在使模型参与自然、类似人类的对话。它们可以以问题或陈述的形式呈现，通常涉及用户和模型之间的来回对话。会话提示可以涵盖广泛的主题，从闲聊到更专注于特定主题的讨论。让我们看看一个例子。

    提示：

    ```py
    Hello! How are you today?
    I'm doing well, thanks for asking. How about yourself?
    ```

    提示：

    ```py
    I'm a bit tired but can't complain. What are your plans for the weekend?
    ```

    这种来回交流允许进行自然、流畅的对话。会话提示对于开发亲切、有帮助的人工智能助手非常重要。

+   **上下文提示**：这些提示提供背景或上下文信息以指导模型的响应。上下文提示有助于模型理解用户的意图、期望的输出格式或生成响应时应考虑的任何约束。它们可能包括示例、解释或描述，以帮助阐明任务。以下是一个示例：

    *您是 OpenAI 创建的 AI 助手，旨在提供帮助、无害和诚实。用户要求您总结以下段落中的关键事件：[**插入段落]*

+   **创造性提示**：这些提示邀请模型生成原创内容或想法，例如写一首诗、创作一个故事或为问题头脑风暴解决方案。创造性提示通常为模型的想象力留出空间，并且可能需要多次迭代和改进以达到期望的输出。例如，*写一首关于海洋的俳句诗*。

+   **事实性提示**：这些提示要求模型提供准确和具体的信息或知识，通常以问题的形式出现。例如，关于历史事件、科学概念或一般常识的查询。事实性提示依赖于模型在预训练阶段学习并综合信息的能力。例如，*独立宣言是在哪一年签署的？*

+   **逐步提示**：这些提示要求模型提供完成特定任务的步骤序列或程序，例如食谱、教程或算法。逐步提示需要模型生成连贯、有序和逻辑的指令——例如，*提供从零开始烘焙蛋糕的步骤，分解每个步骤*。逐步提示通常用于要求 LLM 解决数学问题。让我们看看一个例子。

    提示：

    ```py
    How much is 5+3*8+5?
    The answer is 43.
    ```

    这不是正确的答案。因此，我们需要使用逐步提示来获得正确的答案。通过逐步推理来解决这个问题

    响应：

    ```py
    Okay, let's go through this step-by-step:
    Step 1) 3 * 8 = 24
    (I should perform the multiplication first since it has higher precedence than addition)
    Step 2) 24 + 5 = 29
    (Now I can add the 5 that comes before the multiplication)
    Step 3) 29 + 5 = 34
    (Finally, I add the last 5)
    Step 4) Therefore, the full solution is:
    5 + 3 * 8 + 5 = 34
    ```

通过将问题分解为清晰的顺序步骤，我能够按照正确的运算顺序评估操作。这种逐步的方法防止我在乘法之前进行加法，故意逐步推理使我能够正确解决问题：

+   **基于意见的提示**：这些提示要求模型对给定主题提供意见、观点或建议。虽然 LLM 没有个人信念或偏好，但它们可以根据从训练数据中提取的模式和见解生成响应。基于意见的提示通常需要仔细考虑模型输出中可能存在的潜在偏差——例如，*自动驾驶汽车技术的优缺点是什么？支持你的观点*。

+   **多模态提示**：这些提示涉及来自多个模态的数据输入，如文本结合图像、音频或视频。多模态提示要求模型处理和整合来自各种来源的信息以生成适当的响应。虽然一些 LLM 主要设计用于基于文本的交互，但 AI 的最新进展导致了能够处理多模态输入的模型的发展，例如 OpenAI 的 DALL-E 和 CLIP。例如，*描述这张图片中的场景：[**插入图片**]*。

+   **系统化提示符**：这些提示符旨在引发遵循特定结构、模式或格式的响应。例如，生成项目列表、创建文章或演示文稿的大纲，或提供给定主题的结构化分析。系统化提示符通常需要模型有组织地、逻辑地组织和呈现信息。这里有一个例子：

    *将以下动物分类到* *分类组中：*

    *[动物列表]*

+   **提示链，也称为思维链**：这些提示符涉及一系列相互关联的输入和输出，其中模型对一个提示符的响应作为下一个提示符的输入。提示链可用于复杂问题解决、多步骤任务或在与模型的对话中保持连续性。让我们看看一个例子。

    提示：

    ```py
    What is the capital of France?
    The capital of France is Paris.
    ```

    提示：

    ```py
    What is the population of Paris?
    ```

    LLM 在提示符之间保持上下文和一致性。

在设计针对大型语言模型（LLMs）的提示符时，考虑预期应用的特定需求和目标是至关重要的。通过理解和结合不同类型的提示符，提示符工程师可以调整他们的方法，以从模型中获得最准确、最相关和最有用的响应。此外，有效的提示符工程通常涉及迭代优化和实验，包括用户反馈并调整提示符结构以优化模型的表现。随着 LLMs 领域的持续发展，提示符工程的艺术和科学将在塑造我们与这些强大 AI 工具互动的方式中发挥关键作用，解锁它们在广泛的应用和领域中的全部潜力。

# LLM 提示符的组成部分

一个 LLM 提示符作为大规模 LLM 的输入，指导其响应生成过程。制作一个有效的提示符对于获得准确、相关和有用的输出至关重要。LLM 提示符的组成部分可能因任务、应用和期望的结果而异。然而，在精心设计的提示符中通常存在几个关键元素：

+   **任务描述**:

    一个提示符的基本组成部分是对模型预期执行的任务的清晰简洁描述。这可以是一条指令、一个问题或一个声明，用于指定交互的目的，例如总结一段文字、翻译文本或回答查询。

    这里有一个例子：

    ```py
    Summarize the following passage in two sentences:
    ```

+   **上下文**:

    在提示符中提供上下文有助于模型理解任务的范围、约束和相关的背景信息。这可以包括特定领域的术语、示例或解释，以指导模型生成响应。上下文信息在消除歧义、细化模型焦点或确保输出遵循特定格式或风格时特别有用。

    这里有一个例子：

    ```py
    In the context of a scientific research paper, provide a concise summary of the main findings.
    ```

+   **输入数据**：

    输入数据是模型将处理并据此生成响应的实际内容。这可能是一句话、一个段落，或一系列问题或陈述。输入数据应与任务相关，并以便于模型理解和分析的方式表述。

    这里有一个例子：

    ```py
    Climate change has led to increased global temperatures, melting polar ice, and rising sea levels. The consequences for marine ecosystems are severe, with many species struggling to adapt to the rapidly changing environment. Coral reefs, in particular, are under threat due to ocean acidification and bleaching events that result from warmer water temperatures.
    ```

+   **占位符标记**：

    在某些情况下，提示可能包含占位符标记（变量），指示模型应在何处插入其响应。这些标记可以帮助保持输出结构和格式，尤其是在需要按特定顺序或布局生成多个输出或响应时。让我们来看一个例子。

    提示：

    ```py
    Imagine you are visiting a [City_Name] for the first time. You have [Number_of_Days] days to explore the city and experience its culture. What are the top [Number_of_Attractions] attractions or activities you would like to do while you're there? Provide a brief description of each attraction and explain why it interests you.
    city name: Boston
    number of days: 3
    number of attractions: 3
    ```

    响应：

    ```py
    1\. Freedom Trail: A 2.5-mile-long path through downtown Boston that passes by 16 historically significant sites, including the Massachusetts State House and Paul Revere's House. I'm interested in learning about the city's rich history and its role in the American Revolution.
    2\. Fenway Park: As a baseball fan, I would love to catch a Red Sox game at this iconic ballpark, which opened in 1912 and is one of the oldest Major League Baseball stadiums still in use.
    3\. New England Aquarium: This popular attraction features a variety of marine life, including sea turtles, penguins, and a giant ocean tank. Visiting the aquarium would be a great way to learn …
    ```

    由于初始提示包含变量（`[]`中包含的术语），用户现在可以通过提供以下提示再次提问：

    ```py
    city name: Paris
    number of days: 4
    number of attractions: 10
    ```

    LLM 将提供相应的列表。用户可以继续添加类似于此的提示，以获取他们想要的任何城市的推荐景点。

+   **示例**：

    在提示中包含示例可能有益，尤其是在处理少样本学习场景时。这些示例作为所需输出格式的演示，可以帮助模型更有效地理解任务。当任务复杂或需要模型以特定结构生成输出时，示例尤其有价值。让我们来看一个例子。

    提示：

    ```py
    Please categorize the following product review as either positive, negative, or neutral.
    The speakers produced very crisp and clear sound with deep bass. The set up was quick and easy. Overall an excellent product I would highly recommend.
    ```

    输出：

    ```py
    Positive
    ```

    输入：

    ```py
    These headphones are lightweight and comfortable, but the audio quality is tinny and lacks bass. The noise cancellation feature is mediocre at best. Output:
    Please categorize the following product review as either positive, negative, or neutral.
    The speakers produced very crisp and clear sound with deep bass. The set up was quick and easy. Overall an excellent product I would highly recommend.
    ```

    输出：

    ```py
    Positive
    These headphones are lightweight and comfortable, but the audio quality is tinny and lacks bass. The noise cancellation feature is mediocre at best. Output:
    ```

    响应：

    ```py
    Negative
    ```

    这些示例可以是用户希望从响应中获得的那种语气和词汇类型。

+   **限制**：

    有时，为了确保模型响应满足特定要求、遵守指南或避免问题内容，有必要对模型的响应施加限制。这些限制可以在提示中明确表达，或者通过精心设计任务的描述和上下文隐含表达。让我们来看一个例子。

    提示：

    ```py
    Compose a short 4-line poem about the sunrise with an AABB rhyme scheme.
    ```

+   **语气** **和风格**：

    提示的语气和风格可以影响模型的响应。指定所需的语气，如正式、随意或说服性，可以帮助生成与预期目的和受众相一致的结果。

    以下示例指定了讽刺的语气。

    提示：

    ```py
    Write a 100-word product review mocking the useless features and flimsy design of a silly kitchen gadget. Use an overly sarcastic tone.
    ```

    以下示例指定了海盗的语气。

    提示：

    ```py
    Compose a pirate's journal entry recounting a day searching for treasure on the high seas. Use pirate slang and language.
    ```

在设计 LLM 提示时，提示工程师必须考虑这些组件之间的相互作用，以创建一个有效的输入，从而从模型中获得预期的输出。这个过程通常涉及迭代优化、实验和结合用户反馈，以优化提示的结构和内容。通过理解和巧妙地结合这些组件，提示工程师可以充分发挥大规模 LLM 的潜力，并确保其响应在广泛的任务和应用中准确、相关且有价值。

将所有这些放在一起，以下是一个使用提示组件之间相互作用的例子。完整的提示如下：

```py
"Summarize the following passage in two sentences. In the context of a scientific research paper, provide a concise summary of the main findings. The passage is: 'Climate change has led to increased global temperatures, melting polar ice, and rising sea levels. The consequences for marine ecosystems are severe, with many species struggling to adapt to the rapidly changing environment. Coral reefs, in particular, are under threat due to ocean acidification and bleaching events that result from warmer water temperatures.' Make sure the summary does not exceed two sentences and avoids using overly technical terms. Provide a summary that is clear, concise, and suitable for a general audience. Example 1: Input: 'The economy has experienced significant growth due to advancements in technology and globalization. However, the distribution of wealth remains unequal, with a small percentage of the population controlling a large proportion of resources.' Output: 'Economic growth has been driven by technology and globalization, but wealth distribution remains unequal.' Example 2: Input: '[Input Data]'. Output: '[summary]' . Input: [Input Data]. Output:"
```

这个提示有效地结合了所有必要的组件，以引导 LLM 生成给定段落的简洁和准确摘要。通过提供清晰的任务描述、上下文、输入数据和语气和风格，模型被引导生成期望的输出。示例展示了预期的格式，而约束确保了摘要简短且适合一般受众。占位符标记有助于保持输出结构，使其更容易提取和处理模型的响应。通过深思熟虑地整合这些组件，这个提示成为实际操作中提示工程的有效示例。

另一种可以融入提示工程的技术是角色提示，其中用户和/或系统采用特定的角色或视角来引导语言模型的响应。

# 承担任何人格 – 定制交互的角色提示

角色提示是提示工程中的一种技术，其中用户和/或系统（LLM）承担特定的角色或人格，通常具有独特的知识或专长，以引导 LLM 生成更准确、相关和上下文适当的响应。通过明确定义角色或用户与模型之间的关系，角色提示可以帮助创建更吸引人和互动的体验，从而产生高质量的输出。

角色提示可以采取多种形式。其中一些列在这里：

+   **专家角色**：用户可以假装成为特定领域或领域的专家，例如科学家、历史学家或专业人士，以从模型中获得更具体和有见地的响应。这种方法还可以鼓励模型提供更详细和细微的信息，利用其广泛的预训练知识。

    您也可以要求系统假装成为专家或扮演某个人。让我们来看一个例子。

    提示：

    ```py
    As an experienced software engineer, I recommend using Python for your web scraping project due to its simplicity and the availability of powerful libraries like BeautifulSoup and Scrapy. What are the pros and cons of using Python for web scraping?
    ```

+   **虚构角色**：用户可以扮演虚构角色，如侦探或探险家，以与模型创建更沉浸和创造性的互动。这特别适用于生成故事、对话或角色扮演场景。让我们来看一个例子。

    提示：

    ```py
    As Sherlock Holmes, I've deduced that the stolen painting must be hidden in the abandoned warehouse at the docks. Can you, as Dr. Watson, provide a detailed plan to recover the painting?
    ```

+   **引导角色**：用户可以承担引导或指导模型的角色，例如教师或教练。这种方法鼓励模型更深入地思考主题，探索不同的视角，或完善对复杂概念的理解。让我们来看一个例子。

    提示：

    ```py
    As your biology tutor, I'd like to review the process of photosynthesis with you. Can you explain the light-dependent and light-independent reactions in your own words?
    ```

+   **协作角色**：用户可以采用强调与模型协作或伙伴关系的角色，例如队友或合著者。这种方法可以导致更动态的互动、相互学习和协同解决问题的效果。让我们来看一个例子。

    提示：

    ```py
    As your co-author for our article on sustainable agriculture, I've drafted the introduction and the first section. Can you provide a well-researched and informative section on the benefits of crop rotation?
    ```

角色提示的有效性取决于角色的清晰度和具体性，以及模型理解和适应给定上下文的能力。通过采用一个定义良好的角色，用户可以设定期望并引导模型生成与期望的个性或专业知识相一致的反应。这种方法可以导致与 LLM 的互动更加吸引人、信息丰富且准确。

必须牢记的是，角色提示并不总是保证完美的结果，因为模型的输出取决于各种因素，例如训练数据的质量和模型泛化知识的能力。然而，角色提示可以是任何提示工程师工具箱中的宝贵工具，它能够使 LLM 在广泛的应用和领域中实现更有效和上下文感知的交互。

让我们看看几个例子，以便我们能够理解这些信息。

## 少样本学习——使用示例提示训练模型

在训练过程中使用提示示例，模型被引导生成与特定任务所需的格式、风格或内容相一致的反应。这些示例被称为“少样本学习”，它们展示了模型从预训练中泛化其知识并快速适应新任务的能力，而无需额外的训练数据。

这里有一些使用提示示例来完成不同任务的示例：

+   **文本摘要**：

    提示：

    ```py
    Summarize the following text in one sentence. Example: Input: The quick brown fox jumps over the lazy dog. The dog doesn't seem to mind. Output: A fox jumps over a lazy dog, who remains unfazed.
    Input: Jane went to the store to buy groceries. She purchased fruits, vegetables, and some snacks for her family. Output:
    Jane buys groceries, including fruits, vegetables, and snacks for her family.
    ```

+   **语言翻译**：

    提示：

    ```py
    Translate the following English text to French. Example: Input: Hello, how are you? Output: Bonjour, comment ça va?
    Input: What time is it? Output:
     Quelle heure est -il?
    ```

+   **情感分析**：

    提示：

    ```py
    Determine the sentiment of the following statement as positive, negative, or neutral. Example: Input: I love this product. Output: positive
    Input: The movie was boring and predictable. Output
    negative
    ```

+   **文本分类**：

    提示：

    ```py
    Categorize the following animal as a mammal, bird, reptile, or fish. Example: Input: lion Output: mammal
    Input: eagle. Output:
    bird
    ```

+   **类比**：

    提示：

    ```py
    Complete the following analogy. Example: Input: hot:cold::happy:? Output: sad
    Input: day:night::awake:? Output:
    ```

    反应：

    ```py
    asleep
    ```

在每个示例中，提示从明确的任务描述开始，接着是一个示例输入-输出对，展示了预期的格式和所需反应。通过提供这些提示示例，模型可以更好地理解任务并生成更准确、上下文相关的输出。

必须牢记的是，少样本学习的效果取决于模型泛化其预训练知识的能力以及提供的示例质量。随着 LLM 领域的持续发展，少样本学习和有效的提示工程等技术将在解锁这些 AI 模型的全部潜力以及增强它们在广泛任务和应用中的性能方面发挥至关重要的作用。

接下来，我们将讨论声音定义。这起着至关重要的作用，因为它指的是使某人表达方式独特的特殊品质和风格。定义所需的声调对于制作能够从语言模型中引发吸引人、自然反应的提示至关重要，这些反应与预期的语气和个性相一致。通过将声音定义纳入提示，用户可以塑造模型的反应，使其更具相关性、与品牌身份一致，并总体上更有效，具体取决于应用。

# 寻找你的声音——在提示中定义个性

声音定义指的是区分个人或实体沟通方式的独特特征、风格和语气。在写作的背景下，声音是内容个性化的一个基本元素，使其引人入胜、相关且难忘。一个定义良好的声音有助于有效地传达意图信息，与目标受众产生共鸣，并建立作者的个性或品牌。

发展独特的声音需要考虑语言的各个方面，如下所示：

+   **语气**：

    写作作品的语气反映了作者希望传达的态度或情感。根据背景和目的，语气可以是正式的、非正式的、对话式的、权威的、有说服力的或轻松的，等等。

+   **词汇**：

    词语、短语和表达方式的选择对文章的声音有重大影响。独特的词汇可以反映作者的个性、专业知识和文化背景，同时满足目标受众的偏好和期望。

+   **句子结构和句法**：

    句子的构建方式，包括它们的长度、复杂性和节奏，可以影响文章的声音。多样化的句子结构可以创造一个动态和吸引人的阅读体验，而一致的句法可以建立可识别的声音。

+   **视角**：

    一篇文章的写作视角，例如第一人称、第二人称或第三人称，可以影响声音和与读者建立的联系。选择正确的视角可以帮助创造一个更沉浸式和相关的体验，为观众。

+   **意象和** **比喻语言**：

    使用生动的意象、隐喻、明喻和其他比喻语言元素可以增强声音，使内容更具吸引力、唤起性和难忘性。

当创作内容时，作家必须考虑这些方面，以发展一个一致且吸引人的声音，使其与目标相符，与受众产生共鸣，并使他们与其他人区别开来。一个定义良好的声音可以帮助建立作者或品牌的身份，在读者中建立信任和熟悉感，并最终有助于他们沟通工作的成功和影响力。

在人工智能生成内容的背景下，通过提示工程技术，如向模型提供有关语气、风格和词汇的具体指令，可以发展出独特的声音。通过调整提示以激发所需的声音，如 LLM 等人工智能模型可以生成与作者或品牌独特沟通风格一致的内容，进一步增强了人工智能生成内容的价值和有效性。

这里有一些不同类型写作和情境下声音定义的例子：

+   **专业和** **权威的声音**：

    这种声音以正式的语调、精确的词汇和结构良好的句子为特点。它通常用于商业报告、学术论文或法律文件。

    这里有一个例子：

    *对市场趋势的综合分析表明，可再生能源在未来十年具有巨大的增长潜力。本报告详细探讨了推动这一增长的因素，并提供了利用新兴机会的战略建议。*

+   **对话式和** **友好式**：

    这种声音采用非正式的语调、日常词汇和随意的句子结构。它适合博客文章、社交媒体内容或个人论文。

    这里有一个例子：

    *嘿，大家好！我刚刚尝试了这个令人惊叹的新巧克力曲奇饼食谱，我迫不及待想和你们分享。它们超级容易做，味道绝对美味。试试看，并告诉我你的想法！*

+   **启发性和** **激励性的语调**：

    这种声音使用振奋人心的语调、生动的意象和情感化的语言来吸引和激励读者。它通常出现在励志演讲、自助书籍或个人叙述中。

    这里有一个例子：

    *每一次旅程都始于一小步，而迈出这步信仰的跳跃取决于你。拥抱未知，克服恐惧，释放你的真正潜能。记住，存在的唯一限制是你为自己设定的限制。*

+   **幽默和** **机智的语调**：

    这个声音融合了轻松的语调、巧妙的文字游戏和幽默，以娱乐和逗笑读者。它可以用于讽刺性文章、喜剧剧本或幽默文章。

    这里有一个例子：

    *如果拖延是一种奥林匹克运动，我可能最终会赢得金牌...当然，我首先得看完最新的电视剧，整理我的袜子抽屉，然后思考生活的意义。*

+   **说服性和** **引人入胜的语调**：

    这种声音使用有说服力的语调、强有力的论点和针对性的语言来说服读者接受特定的观点或采取行动。它通常用于观点社论、销售文案或政治演讲。

    这里有一个例子：

    *投资教育不仅是一种道德上的必要，也是一种经济上的必需。通过赋予我们的青年 21 世纪所需的知识和技能，我们为所有人建立一个繁荣和可持续的未来奠定基础。*

每种声音定义都有其独特的特点、语调和风格，适合不同的环境和目的。通过理解和掌握这些声音定义，作家可以调整他们的内容，有效地与目标受众产生共鸣，并实现他们的沟通目标。

当与 AI 生成的内容一起工作时，提供清晰的指令和所需声音定义的示例可以帮助指导 LLM 生成与预期语气、风格和词汇相符的内容。通过将声音定义纳入提示工程，AI 模型可以创建更具吸引力、相关性和影响力的内容，从而进一步增加 AI 生成内容在各种应用和领域的价值。声音定义可以通过两种方式融入。第一种方式是在提示中指定您希望 LLM 在响应中复制某人的风格、语调和其他特征。如果我们要使用像莎士比亚或蒂姆·埃利斯这样的知名作家的声音定义，这很有用。第二种方式是在提示中提供目标人物写作的示例。

现在我们已经探讨了声音定义的概念及其在书面沟通中的重要性，让我们将注意力转向另一个重要方面：模式。正如声音为内容增添个性一样，模式为写作增添结构和节奏，使写作流畅且引人入胜。

# 使用模式来增强提示的有效性

在提示工程的背景下，模式指的是从给定输入提示中元素的组织和重复中出现的可识别的结构、序列和关系。理解和利用这些模式可以提高提示的有效性，并有助于从 LLM 中获得更准确、相关和上下文适当的响应。

提示工程中的模式可以涵盖以下几个方面：

+   **语言模式**：这包括在构建提示时使用的语法、语法和词汇。通过理解语言模式，提示工程师可以创建更有效和连贯的提示，引导 LLM 以所需的格式、风格和语调生成响应。

+   **任务模式**：某些任务或应用可能需要遵循特定的模式或惯例。例如，摘要任务可能要求模型在保留关键点的同时压缩信息。另一方面，翻译任务则涉及在保留意义和结构的同时转换文本。在提示中识别和结合这些模式有助于模型为给定任务生成更合适的输出。

+   **上下文模式**：这些模式涉及提示中元素之间的关系和依赖性，例如提供给模型的上下文、输入数据和所需输出。通过理解这些模式，提示工程师可以创建更好的提示，以更好地引导模型生成上下文相关和准确的响应。

+   **响应模式**：这指的是模型生成的输出中的模式，它可以受到提示的结构、措辞和语气的影响。分析 LLM 响应中的模式可以帮助提示工程师迭代优化他们的提示，以优化模型性能并减少不准确或偏差。

在提示工程中，识别和利用模式对于以下几个原因至关重要：

+   **提高模型性能**：理解模式使得提示工程师能够设计出更有效的提示，引导模型生成准确和上下文相关的响应。

+   **减少歧义**：通过结合使任务、上下文或所需输出更明确的模式，模型产生模糊、无意义或不相关响应的可能性降低。

+   **适应新任务**：识别模式的能力可以帮助提示工程师快速调整他们的提示以适应新的任务或领域，从而在多样化的应用中更有效地使用 LLM。

+   **减轻偏差**：识别和解决模型生成模式中的偏差可以帮助提示工程师创建生成更公平、无偏见和负责任的输出的提示。

输出模式指的是 LLM 生成的响应应遵循的具体结构、格式或约定。通过在提示中提供明确的指令和示例，您可以引导模型生成遵循所需模式的输出。在生成格式为 JSON 数组的列表项的情况下，提示可以包括一个示例，展示预期的输出格式。

这里是一个如何构建此类提示的示例：

提示：

```py
Given the following list of fruits – apple, banana, orange – create a JSON array of the items. Use this format as a reference.
```

输入：

```py
dog, cat, fish
```

输出：

```py
'[{animal: "dog"}, {animal: "cat"}, {animal:"fish"}]'"
```

输入：

```py
apple, banana, orange
```

输出：

```py
[{fruit: "apple"}, {fruit: "banana"}, {fruit: "orange"}]
```

在这个提示中，任务描述清楚地说明了创建项目 JSON 数组的目标。提示中提供的示例有助于模型理解在生成响应时应遵循的模式。

为了确保输出模式的一致性，您还可以在提示中指定额外的约束或格式指南。例如，您可以提到输出应包括每个项目周围的引号，或者数组应保持输入项目的顺序。

当处理 JSON 数组或其他结构化数据格式时，确保生成的输出遵循该格式的正确语法和约定至关重要。通过精心设计包含明确指令、示例和约束的提示，您可以引导 LLM 生成遵循所需输出模式的响应，从而为各种任务和应用生成更准确、有用和格式良好的输出。

如我们所探讨的，提示工程中有各种有效技术，如角色提示、使用示例和应用输出模式。每种技术都有其优势，但结合它们可以产生更强大的效果。不同技术融合的协同效应使得提示工程师能够制作出能够从语言模型中引发高度准确、自然和上下文适当的响应的提示。

# 混合搭配——增强提示的战略组合

通过深思熟虑地整合互补的方法，提示工程师可以开发出引人入胜、自然且高度有效的提示。在本节中，我们将探讨一些提示技术协同工作以从语言模型中引发准确、上下文相关的响应的示例。通过角色提示、约束和其他技术协同工作，提示工程师可以引导模型生成完成任务所需的精确内容。

这里有一些示例，展示了这些技术的结合：

+   **结合角色提示和示例**：

    假设我们想在导师-学生关系的背景下询问模型关于时间管理的建议。我们可以结合角色提示和示例来创建一个有效的提示。

    提示：

    ```py
    As my time management mentor, can you give me three tips to improve my productivity? For instance, an example of a tip might be: Set specific and achievable goals for each day.
    Certainly! Here are three tips to enhance your productivity: 1\. Break large tasks into smaller, manageable subtasks. 2\. Prioritize your tasks by importance and urgency using the Eisenhower Matrix. 3\. Take regular breaks to refresh your mind and stay focused throughout the day.
    ```

+   **结合角色提示、示例和输出模式**：

    任务：为新闻频道写一段简洁的天气预报。

    提示：

    ```py
    As a professional weather reporter, provide a brief and informative weather report for New York City today, including temperature, chance of precipitation, and any weather advisories. Example: Good morning, NYC! Today, expect a high of 72°F with a 40% chance of afternoon showers. Wind advisory in effect until 6 PM.
    Good morning, New York City! As your local weather reporter, here is your forecast for today. We're looking at a high temperature of 68°F under partly cloudy skies. There is a 30% chance of light rain showers this evening, so keep those umbrellas handy. A wind advisory is in effect from noon until 8PM tonight, with gusts up to 25 mph expected. Other than the potential for wet weather later on, it looks to be a mild October day in the Big Apple. Enjoy, and stay tuned for more updates as needed!
    ```

    在这种情况下，提示提供了一个输出示例，同时也作为响应应如何提供的模式。

+   **结合角色提示、示例和输出模式**：

    任务：为面向家庭的博客写一篇简短的电影评论。

    提示：

    ```py
    As a family-friendly movie critic, write a short review for the animated movie Inside Out, focusing on its message, entertainment value, and age-appropriateness. Example: Inside Out is a heartwarming and entertaining film that teaches valuable life lessons to both kids and adults. The animation is top-notch, and the voice acting is superb. Perfect for ages 6 and up.
    As a parent and movie critic, I highly recommend Inside Out for family viewing. This clever animated film creatively explores emotional intelligence and human psychology through colorful characters representing Joy, Sadness, Anger, Disgust, and Fear. Kids will enjoy the fun humor and adventures while also learning about complex emotions we all experience. The top-notch animation is visually appealing and the talented voice cast shines. Inside Out is heartfelt and entertaining for viewers of all ages 6 and up. It teaches an important message about embracing our feelings, both positive and negative. This is a thoughtful movie the whole family can enjoy together.
    ```

    提示提供了一个输出示例，同时也作为响应应如何提供的模式。

+   **结合角色提示、输出模式和示例**：

    任务：创建一句励志名言来激励年轻企业家。

    提示：

    ```py
    As a successful entrepreneur, come up with a motivational quote that inspires young entrepreneurs to chase their dreams and never give up. Example: Success is not final, failure is not fatal; it's the courage to continue that truly counts. Keep pushing, young dreamers!
    Every setback brings you closer to success if you keep believing. Stay determined, stay resilient - your dreams are always within reach.
    ```

+   **结合角色提示、示例和输出模式**：

    任务：为素食意面菜谱写一段简短的描述。

    提示：

    ```py
    As a vegan cookbook author, describe a simple and delicious vegan pasta recipe, including the main ingredients and preparation method. Example: Creamy avocado pesto pasta is a delightful vegan dish that combines perfectly cooked pasta with a rich, smooth avocado-based sauce. Blend ripe avocados, fresh basil, garlic, lemon juice, and pine nuts to create a luscious pesto that will leave everyone craving more.
    For a quick and easy vegan pasta, sauté minced garlic in olive oil over medium heat. Add chopped zucchini and sauté for 3-4 minutes. Add marinara or tomato pasta sauce and simmer for 2-3 minutes. Meanwhile, cook pasta according to package instructions. Top the hot pasta with the vegetable sauce mixture. Finish with a garnish of toasted pine nuts, fresh basil, and grated vegan Parmesan for a flavorsome plant-based pasta dish. The savory vegetables, aromatic garlic and basil, and umami Parmesan create a simple yet delicious vegan pasta in minutes.
    ```

+   **结合角色提示、示例和输出模式**：

    任务：为在家工作的人写一条简洁的健身建议。

    提示：

    ```py
    As a fitness expert, provide a short and helpful fitness tip for people who are working from home and want to stay active. Example: Don't forget to take breaks! Set a timer for every hour, stand up, stretch, and take a quick 5-minute walk to boost your energy and productivity.
    ```

    输出：

    ```py
    Try doing bodyweight exercises during work breaks. Pushups, squats, lunges, and planks can be done anywhere without equipment. Just a few minutes every hour keeps your blood pumping and muscles activated. Staying physically active throughout the workday is key for energy, focus, and wellbeing when working from home.
    ```

现在我们已经介绍了各种制作有效提示的技术，让我们将注意力转向探索可以调整以控制和细化 LLM 输出的关键参数。

# 探索 LLM 参数

LLMs，如 OpenAI 的 GPT-4，由几个可以调整以控制和微调其行为和性能的参数组成。理解和操作这些参数可以帮助用户获得更准确、相关和上下文适当的输出。以下列出了一些最重要的 LLM 参数：

+   **模型大小**: LLM 的大小通常指的是它拥有的神经元或参数数量。较大的模型可能更强大，能够生成更准确和连贯的响应。然而，它们可能也需要更多的计算资源和处理时间。用户可能需要根据具体需求在模型大小和计算效率之间进行权衡。

+   **温度**: 温度参数控制 LLM 生成的输出的随机性。较高的温度值（例如，0.8）会产生更多样化和创造性的响应，而较低的温度值（例如，0.2）则会导致更专注和确定性的输出。调整温度可以帮助用户微调模型响应中的创造性和一致性之间的平衡。

+   **Top-k**: Top-k 参数是控制 LLM（大型语言模型）输出随机性和多样性的另一种方式。此参数限制模型在生成响应的每一步中仅考虑前“k”个最可能的标记。例如，如果 top-k 设置为 5，模型将从一个最有可能的五个选项中选择下一个标记。通过调整 top-k 值，用户可以管理响应多样性和连贯性之间的权衡。较小的 top-k 值通常会导致更专注和确定性的输出，而较大的 top-k 值则允许有更多样化和创造性的响应。

+   **最大令牌数**: 最大令牌数参数设置了生成输出中允许的最大令牌数（单词或子词）。通过调整此参数，用户可以控制 LLM 提供的响应长度。设置较低的 max tokens 值可以帮助确保简洁的答案，而较高的值则允许有更详细和详尽的响应。

+   **提示长度**: 虽然提示长度不是 LLM 的直接参数，但它会影响模型的表现。较长的、更详细的提示可以为 LLM 提供更多的上下文和指导，从而产生更准确和相关的响应。然而，用户应意识到非常长的提示可能会消耗大量令牌限制，从而可能截断模型的输出。

通过理解这些 LLM 参数并根据具体需求和需求进行调整，用户可以优化与模型的交互，并获得更准确、相关和上下文适当的输出。平衡这些参数并根据手头的任务进行定制是提示工程的关键方面，这可以显著提高 LLM 的整体有效性。

重要的是要注意，不同的任务可能需要不同的参数设置才能达到最佳效果。用户应尝试各种参数组合，并考虑创造力、一致性、响应长度和计算需求等因素之间的权衡。这种测试和改进参数设置的迭代过程将帮助用户充分发挥 GPT-4、Claude 和 Google Bard 等 LLM 的潜力。

通过尝试不同的参数和技术，可以帮助你了解每种情况的最佳方案。下一节将深入探讨在处理提示时如何采用这种实验心态。

# 如何处理提示工程（实验）

接近提示工程涉及一个系统性和迭代的实验过程，以从 LLM 中获得期望的输出。关键是根据模型的响应来精炼和调整你的提示，不断改进其有效性。以下是通过实验方法接近提示工程的步骤：

1.  **定义目标**: 明确说明与 LLM 交互的目标。确定所需的具体信息、格式和上下文，以生成期望的输出。

1.  **制定初始提示**: 使用提示的组成部分，如上下文、指令、角色提示、示例和输出模式，创建一个清晰简洁的提示，向 LLM 传达你的期望和要求。

1.  **调整 LLM 参数**: 根据你的输出偏好，如创造力、确定性和响应长度，设置 LLM 参数的初始值，如温度、top-k 和最大标记数。

1.  **测试和评估**: 将提示提交给 LLM，并分析生成的输出。评估响应与你的期望如何相符，考虑因素包括相关性、连贯性、格式和语气。

1.  **精炼提示**: 根据输出评估，确定改进区域，并相应地修改提示。这可能包括澄清指令、添加示例、调整输出模式或改变角色提示。如有必要，还可以考虑精炼 LLM 参数，调整如温度或 top-k 等值，以影响响应的创造力和确定性。

1.  **迭代**: 重复测试、评估和改进过程，直到 LLM 生成满足你标准的满意输出。这种迭代方法有助于微调提示工程流程，并适应各种任务和需求。

1.  **记录成功与失败**: 保留成功和未达到预期效果的提示工程技术和参数设置的记录。这份文档将在未来的实验中作为宝贵的参考资料，帮助你基于以往的经验构建，并简化提示工程流程。

1.  **分享发现并合作**：与更广泛的 LLM 用户社区互动，分享见解，从他人的经验中学习，并共同制定提示工程的最佳实践。知识和思想的交流可以帮助提高提示工程过程的整体效果和效率。

1.  **应用可转移的技术**：随着你在提示工程中积累经验，识别可以应用于各种任务和领域的技巧和策略。这些可转移的方法可以帮助你快速适应新的挑战，并最大化 LLM 交互的有效性。

1.  **关注 LLM 的进步**：随着大规模 LLM 的持续发展，了解新的发展、功能和最佳实践至关重要。定期审查 LLM 开发者（如 OpenAI）的更新、研究和资源，以确保你的提示工程技巧保持有效和相关性。

1.  **探索创意应用**：提示工程不仅限于传统任务和输出。尝试创新和创意的 LLM 应用，推动这些模型所能达到的边界。这种探索性方法可以带来新颖的解决方案、见解和应用，展示大规模 LLM 的真实潜力。通过系统实验的方法来处理提示工程，用户可以逐步改进他们的提示和 LLM 交互，确保输出更加准确、相关和符合上下文。

尽管提示工程非常强大，但它可能导致错误的结果。因此，在下一节中，我们将探讨一些限制以及如何尝试减轻它们。

# 使用 LLM 提示的挑战和限制

尽管 LLM 如 GPT-4 在生成类似人类的响应方面表现出惊人的能力，但在制定有效的提示时，它们也带来了一组挑战和限制。以下是一些挑战和限制：

+   **冗长**：LLM 倾向于生成冗长的输出，经常提供比必要更多的信息或重复观点。构建鼓励简洁响应的提示可能具有挑战性，可能需要迭代提示并设置适当的约束。

+   **歧义**：LLM 可能难以处理模糊或定义不明确的提示，导致输出不符合用户的期望。用户必须投入时间和精力来创建清晰和具体的提示，以最大限度地减少歧义。

+   **不一致性**：LLM 有时会生成包含相互矛盾信息或在不同运行中质量不同的响应。确保输出的连贯性可能需要微调参数和提示工程技术。

+   **缺乏常识**：尽管大型语言模型（LLMs）拥有庞大的知识库，但它们有时会产生缺乏常识或做出错误假设的输出。用户可能需要尝试不同的提示技巧来获得更准确和合理的响应。

+   **偏见**：LLMs 可能会无意中表现出其训练数据中存在的偏见。因此，LLMs 可能会无意中学习和延续这些偏见和歧视性信念，导致观点偏颇和不公平的结果。这可能会产生严重的影响，尤其是在 LLMs 被用于招聘、教育和决策等领域时。因此，识别和解决 LLMs 中的这些偏见至关重要，确保这些强大的工具被负责任和道德地使用。

+   **幻觉**：LLMs 的幻觉指的是 LLMs 生成与输入上下文不符、不合逻辑或无关的文本的情况。这种现象发生是因为 LLMs 从大量的训练数据中学习模式和关联，但它们不具备对世界的内在理解或像人类一样推理的能力。因此，它们有时可能会产生看似合理但实际上不准确或不合逻辑的输出。当用户依赖 LLMs 获取事实信息、进行决策或生成内容时，幻觉可能会特别令人担忧。为了减轻幻觉的影响，用户必须投资于改进模型、创建更好的评估指标和实施用户反馈循环，以增强 LLMs 的性能和可靠性。

随着本章的结束，我们开始全面探索 AI 提示，深入其组成部分、类型和实际应用。有了这些基础知识，我们现在可以更深入地探索提示工程的世界，并在接下来的章节中揭示高级技术。

# 摘要

在本章的介绍中，你开始了探索 AI 提示世界的旅程。本章提供了 LLM 提示的全面概述，包括其组成部分、类型及其工作方式。通过实际示例和用例，你见证了 LLM 提示的力量和多功能性，例如生成产品描述和翻译文本。

通过探索不同类型的提示及其在引导 LLMs 中的作用，你在提示工程方面打下了坚实的基础。你熟悉了提示的关键组成部分，如输入提示、上下文和响应，并理解了它们如何塑造语言模型的输出。

本章强调了在创建引人入胜和有效内容时，声音定义和模式的重要性。你发现了这些元素如何赋予你的写作独特的个性，并保持流畅，使你的信息具有影响力和记忆点。

在探索 LLM 提示语的潜力时，本章还讨论了与它们使用相关的挑战和限制。通过理解这些障碍，你获得了如何应对它们并在你的提示语工程实践中做出明智决策的见解。

随着本章的结束，你现在拥有了本书其余部分的综合基础。通过对 LLM 提示语的深入理解，你准备好探索高级技术，结合不同的提示语工程方法，尝试各种参数，并克服挑战以开启新的可能性。在整个旅程中，你将利用 LLM 提示语的强大功能来创造非凡的体验并实现你的预期目标。

在建立了提示语工程的坚实基础之后，我们现在将目光转向将这些技能应用于释放 AI 自动内容创建的潜力。这将在接下来的章节中深入探讨。
