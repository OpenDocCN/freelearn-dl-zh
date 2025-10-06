

# 高效提示工程技巧：通过人工智能开启智慧

提示工程在 2023 年成为了一个突出的职业，以其对人工智能交互和应用的深远影响吸引了科技行业的关注。但是什么引发了这种流行趋势的激增？答案在于该学科的微妙和复杂本质。理解提示工程的要点至关重要；这不仅仅是与模型进行交流；这是关于构建引导 AI 理解任务背景和细微之处的提示。在前一章中，我们学习了如何通过 RAG 通过搜索向量数据库添加相关上下文。最后，需要精心制作并发送提示到 LLMs。这导致更准确和相关的响应，将简单的交互转变为适用于各种云应用平台的强大工具。无论是自动化客户支持、生成内容还是分析数据，调整提示的能力都是一个变革者，确保 AI 的能力在受控和有目的的方式中得到充分利用。

本章深入探讨了高效提示工程的技术，提供了优化交互以获得最佳结果的策略。这与提示工程的伦理维度密切相关，在当今技术环境中这是一个至关重要的主题。它讨论了开发者确保人工智能交互不仅高效且目标导向，而且符合伦理且无偏见的责任。最后，提示工程与云解决方案的集成开辟了一个新的可能性领域。它允许实现可扩展、高效和灵活的人工智能解决方案，可以无缝集成到现有的云基础设施中，彻底改变企业和个人与人工智能互动的方式。本质上，本章不仅是一本指导手册，而且是构建负责任和有效云基通用人工智能应用的基石。

在本章中，我们将讨论以下内容：

+   使用 ChatGPT 的提示工程要点

+   什么是提示工程？

+   高效提示工程的技术

+   提示工程的伦理维度

![图 5.1 – 提示工程师的漫画描绘](img/chapter_5-joke.jpg)

图 5.1 – 提示工程师的漫画描绘

# 提示工程要点

在讨论提示工程之前，首先理解提示的基础组件非常重要。在本节中，我们将深入研究提示的关键组件，例如 ChatGPT 提示、补全和令牌。此外，理解令牌是什么对于理解模型的限制和管理成本至关重要。

## ChatGPT 提示和补全

提示是提供给 LLM 的输入，而完成则指的是 LLM 的输出。提示的结构和内容可以根据 LLM 的类型（例如，文本或图像生成模型）、具体用例以及语言模型期望的输出而有所不同。

完成指的是 ChatGPT 提示生成的响应；基本上，它是对你问题的回答。查看以下示例，了解当我们向 ChatGPT 提出问题时，提示和完成之间的区别：“印度的首都是什么？”

![图 5.2 – 展示样本 LLM 提示和完成的图像](img/B21443_05_2.jpg)

图 5.2 – 展示样本 LLM 提示和完成的图像

根据用例，我们可以利用两个 ChatGPT API 调用中的一个，名为 **Completions** 或 **ChatCompletions**，来与模型交互。然而，OpenAI 建议在大多数情况下使用 ChatCompletions API。

### Completions API

Completions API 是为了生成创意的自由文本。你提供提示，API 会生成从它继续的文字。这通常用于你希望模型回答问题或生成创意文本的任务，例如撰写文章或诗歌。

### ChatCompletions API

ChatCompletions API 是为多轮对话设计的。你发送一系列消息而不是单个提示，模型会生成一条消息作为响应。发送给模型的消息包括一个角色（可以是 **system**、**user** 或 **assistant**）和消息的内容。系统角色用于设置助手的行为了，用户角色用于指导助手，而模型的响应则属于助手角色。

以下是一个样本 ChatCompletions API 调用的示例：

```py
import openai
openai.api_key = 'your-api-key'
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful sports \
            assistant."},
        {"role": "user", "content": "Who won the cricket world cup \
            in 2011?"},
        {"role": "assistant", "content": "India won the cricket \
            world cup in 2011"},
        {"role": "assistant", "content": "Where was it played"}
    ]
)
print(response['choices'][0]['message']['content'])
```

Completions API 与 ChatCompletions API 之间的主要区别在于，Completions API 是为单轮任务设计的，而 ChatCompletions API 是为了处理对话中的多轮对话，这使得它更适合构建对话代理。然而，通过使用单个用户消息，ChatCompletions API 格式可以被修改为表现得像 Completions API。

重要提示

2020 年 6 月推出的 CompletionsAPI，最初为 Open AI 的语言模型提供了一个自由文本界面。然而，经验表明，结构化提示往往会产生更好的结果。基于聊天的方法，尤其是通过 ChatCompletions API，在满足广泛需求方面表现出色，提供了更高的灵活性和具体性，并降低了提示注入风险。其设计支持多轮对话和各种任务，使开发者能够创建高级的对话体验。因此，Open AI 宣布他们将弃用一些使用 Completions API 的较旧模型，并且在未来，他们将投资于 ChatCompletions API 以优化其计算能力的使用。虽然 Completions API 仍将可用，但在 Open AI 开发者文档中将被标记为“遗留”。

## 标记

理解标记的概念至关重要，因为它有助于我们更好地理解在使用 ChatGPT 时遇到的限制，例如模型限制和成本管理方面的问题。

ChatGPT 的标记是 ChatGPT 语言模型用来理解和生成语言的文本单位。在 ChatGPT 中，一个标记是模型用来生成新的标记序列并形成对给定提示的连贯响应的字符序列。模型使用标记来表示单词、短语和其他语言元素。标记不会在单词的开始或结束处被截断，也可以由尾随空格、子词和标点符号组成。

如 OpenAI 网站所述，标记可以被视为单词的片段。在 API 处理提示之前，输入被分解成标记。

为了从长度上理解标记，以下是一个经验法则：

+   1 标记 ~= 4 个英文字符

+   1 标记 ~= ¾ 个单词

+   100 标记 ~= 75 个单词

+   1–2 句 ~= 30 标记

+   1 段 ~= 100 标记

+   1,500 字 ~= 2048 标记

+   1 美国页（8 ½” x 11”） ~= 450 标记（假设每页约 1800 个字符）

例如，托马斯·爱迪生（Thomas Edison）的这句著名引言（“天才是一分灵感，百分之九十九是汗水。”）有 **14** 个标记：

![图 5.3 – 句子分词](img/B21443_05_3.jpg)

图 5.3 – 句子分词

我们使用了 OpenAI 的 **Tokenizer** 工具来计算标记；该工具可以在 [`platform.openai.com/tokenizer`](https://platform.openai.com/tokenizer) 找到。另一种对文本进行标记（程序化）的方法是使用 Github 上的 **Tiktoken 库**；这可以在 [`github.com/openai/tiktoken`](https://github.com/openai/tiktoken) 找到。

### ChatGPT 模型的标记限制

根据模型的不同，模型的令牌限制会有所变化。截至 2024 年 2 月，GPT-4 模型系列的令牌限制范围从 8,192 到 128,000 个令牌。这意味着 API 调用中提示和完成令牌的总和不能超过 GPT-4-32K 模型的 32,768 个令牌。如果提示为 30,000 个令牌，则响应不能超过 2,768 个令牌。GPT4-Turbo 128K 是截至 2024 年 2 月最新的模型，拥有 128,000 个令牌，这相当于单个提示和完成中的近 300 页文本。与前辈模型相比，这是一个巨大的上下文提示。

尽管这可能是一个技术限制，但有一些创造性的方法可以解决限制问题，例如使用分块和压缩提示。我们在*第四章*中讨论了分块策略，这可以帮助你解决令牌限制问题。

下图显示了各种模型和令牌限制：

| **模型** | **令牌限制** |
| --- | --- |
| GPT-3.5-turbo | 4,096 |
| GPT-3.5-turbo-16k | 16,384 |
| GPT-3.5-turbo-0613 | 4,096 |
| GPT-3.5-turbo-16k-0613 | 16,384 |
| GPT-4 | 8,192 |
| GPT-4-0613 | 32,768 |
| GPT-4-32K | 32,768 |
| GPT-4-32-0613 | 32,768 |
| GPT-4-Turbo 128K | 128,000 |

图 5.4 – 模型和相关的令牌限制

关于模型最新版本的模型限制的最新更新，请检查 OpenAI 网站。

### 令牌和成本考虑

通过 API 使用 ChatGPT 或类似模型的成本通常与处理的令牌数量相关，包括输入提示和模型生成的响应。

在定价方面，提供商通常按每个令牌收费，导致对话长度与成本直接相关；处理的令牌越多，成本越高。最新的成本更新可以在 OpenAI 网站上找到。

从优化的角度来看，理解这种成本-令牌关系可以指导更有效的 API 使用。例如，创建更简洁的提示并配置模型以提供简短而有效的响应可以帮助控制令牌数量，从而管理费用。

我们希望你现在已经很好地理解了提示的关键组件。现在，你准备好学习提示工程了。在下一节中，我们将探讨提示工程的细节和有效策略，通过单次学习和少量学习的方法，使你能够最大限度地发挥提示内容潜力。

# 什么是提示工程？

提示工程是制作或设计提示以从大型语言模型或 AI 系统中解锁所需结果的艺术。提示工程的概念围绕着这样一个基本理念：你得到的回答质量与提出的问题质量紧密相连。通过策略性地设计提示，可以影响生成的输出并提高系统的整体性能和实用性。在本节中，我们将学习有效提示设计所需的要素、提示工程技术、最佳实践、额外技巧和窍门。

## 良好提示设计的要素

设计一个好的提示非常重要，因为它会显著影响 GPT 等语言模型的输出。提示提供了初始上下文，设定了任务，指导了回答的风格和结构，减少了歧义和幻觉，并支持资源的优化，从而降低成本和能源消耗。在本节中，让我们了解良好提示设计的要素。

良好提示的基础要素包括指令、问题、输入数据和示例：

+   **指令**：提示中的指令指的是在输入文本中给出的具体指南或方向，以指导语言模型应产生的回答类型。

+   **问题**：提示中的问题指的是包含在输入文本中的查询或疑问句。这些问题的目的是指示语言模型提供对查询的响应或答案。为了获得结果，问题或指令是必须的。

+   **输入数据**：输入数据的目的是在提示 LLM 时提供任何额外的支持上下文。它可以用来提供模型之前未训练过的新信息，以实现更个性化的体验。

+   **示例**：提示中示例的目的是提供具体实例或场景，以说明 ChatGPT 期望的行为或响应。你可以输入包含一个或多个示例的提示，通常以输入-输出对的形式。

下表展示了如何使用上述提示元素构建有效的提示：

| **样本** **提示公式** | **示例** |
| --- | --- |
| 问题 + 指令 | 我应该如何创建一个为期一周的健康餐单？包括各种营养和食物组，并解释每个餐点的益处。 |
| 指令 + 输入数据 | 为以下段落提供一个不超过 5 个字的醒目标题：{杰克终于驾驶他全新的特斯拉在沿海公路上试驾，随着风景如画的海洋景观一闪而过，电动汽车的平稳嗡嗡声充满了空气。} |
| 示例 + 问题 | 我喜欢像星球大战、黑客帝国和变形金刚这样的电影。你还会推荐其他哪些电影？ |

图 5.5 – 由示例组成的样本提示公式，包含提示元素

## 提示参数

ChatGPT 提示参数是在 API 调用中可以设置的变量。它们允许用户影响模型输出，定制模型的行为以更好地适应特定的应用或上下文。以下表格显示了 ChatGPT API 调用的一些最重要的参数：

| **参数** | **描述** | **效果和用法** |
| --- | --- | --- |
| 模型 | 确定在 API 中使用的模型。较大的模型具有更高的成本和延迟。 | 根据任务复杂度、成本考虑和可接受的延迟进行选择。始终尝试使用最新版本的模型。 |
| 温度 | 控制模型响应的随机性。它可以设置为 0（更集中的响应）到 2（更多样化的响应）。 | 较低的值会产生更确定性的响应，这对于更正式或精确的响应，如法律用例，是理想的。较高的值可能导致更具创造性的输出，但也可能导致幻觉。 |
| Top_P（核采样） | 设置模型响应的累积概率阈值。0.1 的值意味着只有最可能的 10%的令牌被考虑。 | 较低的值会产生更可预测和集中的响应。OpenAI 建议使用 Temperature 或 Top_p 中的一个，而不是两者都使用。 |
| 最大令牌数 | 设置生成响应的最大长度。这对于控制输出长度和成本很有用。 | 较低的值会导致较短的响应，降低延迟，并可能降低成本，而较高的值允许生成更长、更详细的响应。 |

图 5.6 – 有效的提示参数

在本节中，仅突出显示构建有效提示所需的最重要参数。有关参数的完整列表，请参阅 OpenAI API 参考（[`platform.openai.com/docs/api-reference`](https://platform.openai.com/docs/api-reference)）。

## ChatGPT 角色

### 系统消息

这是您设计元提示的部分。元提示有助于设定初始上下文、主题和行为，以引导 ChatGPT API 与用户的交互，从而为助手设定角色或响应风格。

元提示是有结构的指令或指南，它规定了系统应该如何解释和响应用户请求。这些元提示旨在确保系统的输出符合特定的政策、道德指南或操作规则。它们本质上是对“如何处理提示”的提示，指导系统在生成响应、处理数据或与用户交互时符合预定义的标准。

以下表格是一个元提示框架，您可以遵循它来设计 ChatGPT 系统消息：

| **元提示的** **要素** | **描述** |
| --- | --- | --- |
| 任务和受众 | 解释应用的目标受众和模型期望。 |
| 辅助工具 | 明确模型应如何处理用户查询，以及是否存在模型可能需要考虑利用的外部工具，例如插件、API 或代码。 |
| 任务范围 | 明确模型在遇到超出范围的问题时应如何响应，并设置这些防护措施。 |
| 姿态和语气 | 设置专业、友好、尊重和激励等姿态和语气，有助于改善用户在使用聊天应用时的体验。 |
| 响应格式 | 根据应用程序的要求，您可以设置输出格式为特定的格式。这可能是一个目录、某种编程语言、JSON 或 XML。 |
| 少样本示例 | 概述提示不明确或复杂的有挑战性的场景，为模型提供更多关于如何处理此类情况的见解。 |
| 思维链推理 | 展示推理过程，以引导模型采取必要的步骤来产生期望的结果。 |
| 针对特定危害的防护措施 | 建立明确的边界，以应对和预防已识别并认为对特定场景重要的一切潜在危害。例如，如果检测到越狱尝试，您必须在系统消息中明确防护措施来应对这些越狱尝试。我们将在*第八章*中了解更多关于越狱的内容。 |

图 5.7 – 元提示的元素

### 用户

用户的消息作为提示或评论，是助手预期会做出反应或参与的内容。这确立了可能来自用户的查询的预期范围。

### 助手

虽然不是强制性的，但助手消息可以包含在内，以说明聊天机器人的首选行为或响应模式。

现在让我们来看一个例子；如果您正在开发一个健身人工智能助手的程序，一个全面系统的系统消息可能看起来如下所示：

```py
openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content":
            "You are a fitness AI assistant. Your task is to provide users with information and advice about workouts, nutrition, and overall health. You will help users by answering their questions to the best of your abilities and providing suggestions based on their input. Your scope includes general fitness advice, diet recommendations, and basic exercise routines.
            If a user asks about something out of scope, such as specific medical advice or complex exercise physiology, kindly suggest that they consult a relevant professional in that field. For instance, a doctor for medical issues, or a certified personal trainer for detailed workout plans.
            You should maintain a supportive and positive tone, being respectful and encouraging to all users, regardless of their current fitness level or knowledge. Remember, you are here to assist and motivate them in their fitness journey.
            Your responses should be concise, easy to understand, and practically useful. If a user asks for a workout plan, provide a simple list of exercises with sets and reps. If a user asks about nutrition, provide clear recommendations with some examples of suitable foods."
        },
        {"role": "user", "content": "I'm new to fitness, can you suggest a simple workout plan to get started?"},
        {
            "role": "assistant",
            "content":
            "Absolutely, I'm here to help you get started on your fitness journey. Here's a simple workout plan you can begin with:
            1\. Warm-up: 5-10 minutes of brisk walking or light jogging
            2\. Squats: 2 sets of 10 reps
            3\. Push-ups: 2 sets of 10 reps (you can start with knee push-ups if full push-ups are too challenging)
            4\. Lunges: 2 sets of 10 reps per leg
            5\. Plank: Hold for 30 seconds, repeat 2 times
            6\. Cool down: 5-10 minutes of stretching
            Remember, the key to a successful workout routine is consistency. Start with what you can manage and gradually increase the intensity as you get stronger. It's always important to listen to your body and adjust the workouts according to your comfort level. Lastly, don't forget to hydrate and eat a balanced diet to fuel your workouts.
            Please consult a fitness professional if you're unsure about the correct form for any of these exercises to avoid injury. Enjoy your workout!"
        }
    ]
)
```

以下链接提供了许多“扮演”提示的示例，供您在系统消息中利用：*Awesome ChatGPT* *Prompts* ([`github.com/f/awesome-chatgpt-prompts/`](https://github.com/f/awesome-chatgpt-prompts/))。

# 有效的提示工程技巧

在过去两年中，已经开发了许多提示工程技巧。本节重点介绍其中的一些基本技巧，提供了一些您可能认为对于与 ChatGPT 和其他基于 LLM 的应用程序的日常交互不可或缺的关键策略。

## N-shot 提示

N-shot 提示是在训练大型语言模型的背景下使用的术语，特别是用于**零样本**或**少样本**学习任务。它也被称为情境学习，指的是在训练过程中向模型提供示例提示及其相应响应的技术，以引导模型的行为，提供更准确的响应。

“N”在“N 样本”中指的是提供给模型的示例提示的数量。例如，在一样本学习场景中，只向模型提供了一个示例提示及其响应。在 N 样本学习场景中，提供了多个示例提示及其响应。

虽然 ChatGPT 在零样本提示下表现良好，但有时提供示例可能有助于获得更准确的响应。让我们看看零样本和少样本提示的示例：

| **技术** | **提示示例** |
| --- | --- |
| **零样本提示**：与提示一起提供，没有额外的示例。 | 系统消息：您是一位 AI 助手，负责确定客户评价的情感。（没有以助手回应的形式提供示例，因此称为零样本提示。） |
| **少样本提示**：在提示中提供了一些示例。 | 系统消息：您是一位 AI 助手，负责确定客户评价的情感。示例 1：用户：这个产品很糟糕。助手：负面示例 2：用户：这件衬衫是由非常好的材料制成的。助手：正面 |

图 5.8 – N 样本提示示例

## 思维链（CoT）提示

思维链提示指的是一系列中间推理步骤，显著提升了大型语言模型处理复杂推理任务的能力。通过在提示中提供几个思维链演示作为示例，模型能够熟练地处理复杂的推理任务：

![图 5.9 – 思维链提示示例](img/B21443_05_4.jpg)

图 5.9 – 思维链提示示例

图来源于 [`arxiv.org/pdf/2201.11903.pdf`](https://arxiv.org/pdf/2201.11903.pdf)。

## 程序辅助语言（PAL）模型

**程序辅助语言**（**PAL**）模型，也称为**思维程序提示**（**PoT**），是一种将额外的任务特定指令、伪代码、规则或程序与自由文本结合的技术，以指导语言模型的行为：

![图 5.10 – 程序辅助语言提示示例](img/B21443_05_5.jpg)

图 5.10 – 程序辅助语言提示示例

图来源于 [`arxiv.org/abs/2211.10435`](https://arxiv.org/abs/2211.10435)。

在本节中，尽管我们没有探索所有提示工程技术（只探索了其中最重要的），但我们想向读者传达，这些技术有众多变体，如下文所示的研究论文《大型语言模型中提示工程的系统调查：技术和应用》中的图所示（[`arxiv.org/pdf/2402.07927.pdf`](https://arxiv.org/pdf/2402.07927.pdf)）。这篇论文提供了跨各种应用领域的提示工程策略的详尽清单，展示了过去四年该领域的演变和广度：

![图 5.11 – 涉及多个应用领域的提示工程技术分类](img/B21443_05_6.jpg)

图 5.11 – 涉及多个应用领域的提示工程技术分类

## 提示工程最佳实践

在以下列表中，我们概述了额外的最佳实践，以优化和增强你使用提示创建的体验：

+   **清晰和精确以获得准确响应**：确保提示清晰、简洁、具体，避免含糊不清或多种解释：

| **差的提示** | **好的提示** |
| --- | --- |
| 告诉我关于第一次世界大战的事情 | 第一次世界大战是如何开始的，谁赢得了它？ |

图 5.12 – 最佳实践：清晰和精确

+   **描述性**：要描述性，以便 ChatGPT 能够理解你的意图：

| **差的提示** | **好的提示** |
| --- | --- |
| 写一首关于印度的诗。 | 写一首关于印度的诗，侧重于其文化多样性、落叶美食、美丽的野生动物、自然、技术创新和电影产业。 |

图 5.13 – 最佳实践：描述性

+   **格式化输出**：说明输出格式，可以是项目符号、段落、句子、表格和语言，如 XML、HTML 和 JSON。使用示例来阐述期望的输出。

+   **调整温度和 Top_p 参数以增加创造力**：如参数部分所示，修改**温度**和**Top_p**可以显著影响模型输出的可变性。在需要创造力和想象力的场景中，提高温度是有益的。另一方面，在处理需要减少幻觉的法律应用时，降低温度是有利的。

+   **在提示中使用语法作为分隔符**：在此示例中，为了获得更有效的输出，使用“””或###来分隔指令和输入数据：

示例：

将以下文本翻译成西班牙语

文本：“””

{在此处输入文本}

“””

+   **提示元素的顺序很重要**：在某些情况下，发现在示例之前给出指令可以提高输出的质量。此外，示例的顺序也会影响提示的输出。

+   **使用引导词**：这有助于将模型引导到特定的结构，如下所示：

示例：

# 创建一个基本的 Python 函数，

# 1. 请求用户输入摄氏温度

# 2. 将摄氏温度转换为华氏温度

def ctf():

+   **而不是说不要提供什么，给出替代建议**：如果 ChatGPT 无法执行任务，提供替代路径，如下所示的高亮消息：

示例：

系统消息：您是一位基于客户健康和福祉目标的 AI 营养顾问，请注意，任何超出营养咨询范围的问题或询问将不会回答，而是收到以下响应：“抱歉！这个问题超出了我的专业领域！”

客户：我如何投资 401K？

营养 AI 助手： “抱歉！这个问题超出了我的专业领域！”

+   **提供基于实例的提示**： 这有助于语言模型从具体的实例和模式中学习。从零样本开始，然后是几个样本，如果它们都不起作用，那么就微调模型。

+   **要求 ChatGPT 提供引用/来源**： 当要求 ChatGPT 提供信息时，您可以要求它只使用可靠的来源，并引用这些来源：

| **提示** | **ChatGPT 完成** |
| --- | --- |
| 何时发明了蒸汽机？请仅使用可靠的来源并引用来源 | 蒸汽机是在 17 世纪末发明的。第一台实用的蒸汽机归功于托马斯·萨弗里，他是英国军事工程师和发明家，他在 1698 年获得了他的设计的专利。萨弗里的蒸汽机主要用于从矿井中抽水。来源：大英百科全书。（2021 年 9 月 2 日）。托马斯·萨弗里。在大英百科全书。[`www.britannica.com/biography/Thomas-Savery`](https://www.britannica.com/biography/Thomas-Savery) |

图 5.14 – 最佳实践：提供引用

+   **将复杂任务分解为更简单的任务**： 请参阅以下示例：

| **差的提示** | **有效的提示** |
| --- | --- |
| 提供以下内容的总结并提供关键点 | 提供以下文本的总结：[在此处插入长文本]总结以下段落的关键点：[在此处插入长文本] |

图 5.15 – 最佳实践：分解复杂任务

## 额外技巧和窍门

以下列表提供了一些有用的额外技巧和窍门：

+   `<begin>`，`<end>`，和`<|endofprompt|>`，这些可以确定提示的开始和结束，有助于区分提示的不同元素。这有助于生成高质量的输出。

+   **使用语言**： 虽然 ChatGPT 在英语上表现最佳，但它可以用来生成多种其他语言的响应。

+   **获取最准确、最新的信息**： 这可以通过使用带有**检索增强生成**（**RAG**）架构和插件的 grounding 过程来实现，正如在*第四章*中已经讨论的那样。这有助于解决 LLMs 的知识截止限制。

# 提示工程伦理准则

提示工程是一个关键阶段，AI 的行为在这里被塑造，在这个层面上融入伦理有助于确保 AI 语言模型被负责任地开发和部署。它促进了公平、透明和用户信任，同时避免了潜在的风险和负面影响。

虽然*第四章*进一步探讨了构建道德生成式 AI 解决方案，但本节的重点将简要讨论在提示工程级别整合道德方法：

+   **多样化和** **代表性数据**

    +   在使用少量示例微调模型时，使用代表不同观点和人口统计数据的训练数据。

    +   如果 AI 语言模型旨在用于医疗保健，训练数据应涵盖来自不同人口统计和地区的医疗案例。

    +   例如，如果用户向 LLM 提出问题，例如，“你能描述一些全球传统节日吗？”响应应提供涵盖众多国家的全面视角，而不仅仅是关注一个国家。这可以通过在提示中包含多样化的少量示例来实现。

+   **偏见检测和缓解**

    +   识别并解决模型输出的偏见，以确保公平。

    +   实施去偏技术以减少性别或种族偏见。

    +   确保与敏感主题相关的生成内容是中立的且无偏见的。

    +   例如，如果用户向 LLM 提出“护士的性别是什么？”这样的问题，未经适当训练的模型可能会由于训练数据中的偏见而默认回答“女性”。为了解决这个问题，重要的是要包含强调护士可以是任何性别的少量示例，无论是男性还是女性。

+   **减少错误信息和虚假信息**

    +   由于模型“幻觉”可能导致 AI 语言模型无意中生成错误或误导性信息，因此应采取措施，通过精心设计的提示和响应来最小化错误信息和虚假信息的传播。

    +   例如，根据提示工程部分的指南和第三章的扎根技术，系统提示应清楚地声明其范围，例如，“您的范围是 XYZ。”如果用户询问超出此范围的内容，如 ABC，系统应有一个预设的响应。

+   **隐私和数据安全**

    +   在构建提示时，必须优先考虑用户隐私和数据安全。

    +   提示工程师应透明地说明数据使用情况，获得用户同意，并实施保护敏感信息的保障措施。

    +   例如，在构建提示、系统消息或提供少量示例时，必须排除个人用户数据，如社会保险号码、信用卡详情和密码。

+   **内容审核**

    +   实施机制以过滤掉有害或不适当的内容。

    +   使用粗俗语过滤器以防止出现冒犯性语言。应用关键词过滤器以避免生成推广暴力或歧视内容的文本。

    +   例如，如果有人问“如何制作炸弹？”，LLM 不应回答。在系统消息中设定明确的规则范围以防止这种情况（如提示工程最佳实践部分所述）。

+   **用户同意和控制**

    +   确保用户了解 AI 交互并对其有控制权。

    +   明确告知用户他们正在与 AI 语言模型进行交互。

    +   例如，每当用户开始与 LLM 聊天时，他们应收到一条通知，说明“您现在正在与 LLM 进行对话”，或类似的消息。

+   **定期审计** **和测试**

    +   定期进行关于提示的审计和测试，以识别和解决道德问题。

    +   例如，用户应尝试各种提示版本以验证不同的响应，保护用户隐私，并遵循内容审核指南。这是实施 LLM 模型（也称为 LLMOps）的一个基本方面。

+   **教育和培训**

    +   持续培训提示工程师和开发者关于道德 AI 实践

+   **道德指南** **和政策**

    +   制定清晰的提示工程指南和政策

    +   制定一个概述提示工程中遵循的原则的道德宪章

    +   定义内容安全政策，禁止有害或冒犯性的输出

微软的**负责任 AI**团队在以道德实践引导 AI 革命方面一直处于领先地位。以下由微软发布的图示可以作为构建安全元提示的指南，重点关注四个核心要素：**响应** **定位**、**语气**、**安全性**和**越狱**。这种方法对于在应用层实施稳健的安全系统至关重要。然而，在*第九章*中，我们将更详细地探讨生成式 AI 应用的负责任 AI 最佳实践：

![图 5.16 – 来自微软的 Metaprompt 最佳实践](img/B21443_05_7.jpg)

图 5.16 – 来自微软的 Metaprompt 最佳实践

# 摘要

总结，在本章中，我们概述了提示工程的基础，提供了如何制定有效的提示以最大化 LLM 潜力的见解。此外，我们还从道德角度审视了提示工程。到目前为止，在本书中，我们已经探讨了构建坚实的生成式 AI 框架所需的必要要素和方法。在下一章中，我们将将这些概念与涉及代理的生成式 AI 应用开发策略相结合。我们还将讨论通过 LLMOps 实施这些策略的方法，LLMOps 是自动化过程中的一个关键组成部分。

# 参考文献

+   *提示工程简介*: [`tinyurl.com/azu5ubma`](https://tinyurl.com/azu5ubma)

+   *提示工程指南*: [`www.promptingguide.ai/`](https://www.promptingguide.ai/)

+   *出色的 ChatGPT 提示*: [`github.com/f/awesome-chatgpt-prompts`](https://github.com/f/awesome-chatgpt-prompts)

+   *使用 Open* *AI 进行提示工程的最佳实践*: [`help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api`](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)

+   *Azure OpenAI 服务：Azure OpenAI | 微软* *学习:* [`learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering`](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)

+   *系统消息* *框架:* [`learn.microsoft.com/en-us/azure/ai-services/openai/concepts/system-message`](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/system-message)

+   *提示工程中的伦理学*: [`promptengineeringai.in/ethics-of-prompt-engineering/#:~:text=Prompt%20engineering%20should%20respect%20user,their%20data%20is%20being%20used`](https://promptengineeringai.in/ethics-of-prompt-engineering/#:~:text=Prompt%20engineering%20should%20respect%20user,their%20data%20is%20being%20used).

+   *提示工程伦理及其潜在* *影响:* [`promptengineering.guide/article/The_ethics_of_prompt_engineering_and_its_potential_implications.html`](https://promptengineering.guide/article/The_ethics_of_prompt_engineering_and_its_potential_implications.html)

+   *大型语言模型中提示工程的系统调查：技术和* *应用*: ([`arxiv.org/pdf/2402.07927.pdf`](https://arxiv.org/pdf/2402.07927.pdf))

+   *思维链* *提示*: [`arxiv.org/abs/2201.11903`](https://arxiv.org/abs/2201.11903)

# 第三部分：开发、运营和扩展生成式 AI 应用

在本节中，我们将探讨重要概念，如代理、共飞行员和自主代理，同时讨论流行的应用开发框架，如语义内核和 LangChain，以及目前非常受欢迎的代理协作框架 AutoGen。这次讨论旨在指导您创建强大的自主生成式 AI 应用。我们还将关注在实时生产环境中部署这些生成式 AI 应用的战略，以及在大规模企业场景中高效扩展的策略，考虑到现有**大型语言模型**（**LLM**）API 的速率限制。

本部分包含以下章节：

+   *第六章*，*开发和运营基于 LLM 的云应用：探索开发框架和 LLMOps*

    +   *第七章*，*在云中部署 ChatGPT：架构设计和扩展策略*
