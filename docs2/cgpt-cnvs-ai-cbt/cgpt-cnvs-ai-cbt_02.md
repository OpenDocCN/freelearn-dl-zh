# 2

# 使用 ChatGPT 进行对话设计

在本章中，我们将深入探讨 ChatGPT 与对话设计的交汇点。我们将探讨对话设计师的角色，并探讨**大型语言模型**（**LLMs**）如 ChatGPT 如何影响对话设计。我们还将检查 ChatGPT 在对话设计中的实际应用，包括其在模拟对话和创建角色中的应用。最后，我们将讨论测试和迭代设计在创建有效和引人入胜的对话人工智能系统中的重要性。到本章结束时，您将全面了解如何利用 ChatGPT 进行对话设计。

一些提示结果被列出，而其他结果则因空间限制而被省略。我们鼓励您亲自尝试这些提示。

在本章中，我们将涵盖以下主要主题：

+   理解对话设计

+   与 ChatGPT 在对话设计中的实际应用合作

+   模拟对话

+   使用 ChatGPT 创建角色

+   使用 ChatGPT 进行对话设计的测试和迭代

# 技术要求

在本章中，我们将使用 ChatGPT。我们建议您在 OpenAI 上注册。

# 理解对话设计

对话设计是一个复杂且重要的角色，对于创建一个有用、自然并且因此成功的聊天机器人或语音助手至关重要。对话设计领域还有其他角色，尤其是 AI 训练师及其变体。为了本书的目的，在讨论对话设计时，我们将考虑这些其他角色，因为许多任务的执行在这些角色之间有很大的重叠。

对话设计利用用户体验设计、语言学、认知心理学和人工智能的原则，通过语音或文字媒介创造人类与机器之间引人入胜的互动体验。在核心上，对话设计师理解沟通、对话人工智能的技术格局、向用户部署的媒介，以及如何满足用户需求。

对话设计的目的是创建一个流畅、直观、自然的用户界面，使用户感觉像是在进行类似人类的对话，同时利用现有技术来模拟这种对话。

## 探索对话设计师的角色

对话设计师执行几个重要的任务。这些任务很重要，这样我们才能看到 LLMs 和 ChatGPT 如何被用来简化并改进这个过程。让我们看看创建对话设计所涉及的过程。这些设计将在项目所选的对话人工智能平台上实施。

### 理解用户需求

对话设计的第一步是理解用户的需求和行为，以及对话发生的背景。

对话设计师的角色是理解组织的运营、流程、挑战和目标。一个好的设计师可以将业务需求转化为客户旅程、用例和自然流程。设计师需要了解目标用户的角色，包括他们的年龄、人口统计、需求、期望和语言。他们还需要了解组织或应用中的当前交互。对现有查询和现有非结构化或结构化数据的彻底理解在这里尤为重要，对对话体验将部署到的渠道的深入了解也同样重要。

### 设计角色

代理的角色不仅决定了其语气，还指导了其对话方式。这需要仔细考虑以匹配用户需求。

还需要考虑现有的品牌和内容指南。

例如，部署在企业软件平台上处理客户支持查询的聊天机器人，其语气和角色将与设计在新生入学周期间帮助学生在 WhatsApp 上查询的代理不同。

### 设计对话

对话流程是用户与对话代理之间交互的序列。

对话设计师需要优先考虑现有用例，以了解代理需要执行的任务以及如何构建对话来实现这一目标。

#### 意图澄清、话语创建和实体澄清

意图代表了用户的意图。例如，意图可能是查询订单状态或预订航班。每个意图都与一组话语相关联，这些话语是用户可能用来表达该意图的短语。例如，对于检查航班状态的意图，话语可能是“*我的航班什么时候起飞？*”或“*我的航班* *准时吗？*”。

实体是聊天机器人需要用来实现意图的具体信息片段。例如，对于预订航班的意图，实体可能是出发城市、目的地城市和旅行日期。理解每种类型的实体及其不同条目是对话设计师的工作。常见的实体通常很简单，因为它们由对话 AI 模型支持。然而，如果需要自定义实体，每个都需要单独获取。例如，一个狗训练聊天机器人可能需要理解所有不同的狗品种。

#### 对话设计

在对话的用户旅程被规划出来后，对话设计师的任务是创建交互的细节。这包括设计用户与对话代理之间的对话流程。它包括定义聊天机器人将用于引导对话的提示、用户响应和代理的回复。对话应该设计得自然且引人入胜，并引导用户实现他们的目标。

设计师会仔细编写对话和响应的不同路径。这些路径基于可能的用户输入，其中需要收集信息以使用户能够传达他们的需求，并使代理收集回答查询所需的值。设计师从通常所说的“快乐路径”开始，这是实现目标的最简单对话路径，然后扩展以处理对话周围的边缘情况。

设计需要支持其他情境边缘情况，例如未处理的意图、不匹配的意图、输入时以及用户尝试引导对话的任何其他方式。这里有很多需要记录的内容。随着对话的许多转折，这些设计可能会变得复杂，因为对话通常不是线性的，用户可能会改变话题或提出意外的问题。这是对话设计师在设计中包含这一点的任务。

对话设计通过以下几种不同的方法在项目利益相关者之间共享：

+   流程图

+   电子表格

+   模拟伪代码

+   对话管理软件，如 Voiceflow

根据项目的大小、团队和预算，以及个人偏好，没有一种适合所有对话设计工具的方法。只要它们易于迭代和协作，这些设计通常会在签署前被多个不同的团队触及。然而，一些更受欢迎的对话管理工具提供了比设计和协作更多的功能，并包括一些利用 LLMs 的非常有用的功能。这将在下一节中介绍。

### 写作

最后一步是编写聊天机器人的实际回复文本。这需要使用有说服力和引人入胜的语言引导用户通过对话。文案应反映聊天机器人的个性，并与品牌的语气和风格保持一致。对于拥有数百甚至数千个回复的大型代理，挑战通常在于确保一致性，尤其是在有多个对话设计师同时或随着时间的推移在一个项目上工作时。随着对话人工智能项目的成熟，这种情况发生的可能性更大。由于每个参与者都会为项目带来自己的风格，因此语气可能会偏离代理的原始个性和品牌指南。

在我们介绍了对话设计师的角色以及构成对话设计过程的特定任务，并在实施对话式人工智能解决方案之前，现在是时候看看 ChatGPT 在这些特定工作流程中可以如何被利用了。

# 在对话设计中使用 ChatGPT 的实际应用

从我们对需要执行对话设计任务的了解来看，很容易看出在某些领域，我们可以利用 ChatGPT 来帮助完成这些任务。一些最受欢迎的对话设计工具也提供了作为其功能一部分执行这些任务的能力。

小贴士

记住，ChatGPT 是一个对话设计工具，而不是对话设计最佳实践的替代品。

## 使用 ChatGPT 进行意图聚类

意图是大多数传统对话式人工智能代理的核心概念。意图的存在是为了定义用户的意图，每个意图都需要有一组话语来触发这个意图。许多对话式人工智能实现都是从大量的实时聊天数据、语音转录或其他非结构化文本数据开始的，这些数据将形成聊天机器人或语音界面需要支持的任务或意图的基础。

### 意图聚类有什么用？

一个对话式人工智能项目可能还处于起步阶段。它也可能是一个在重压下使用或主题内容经常变化的成熟项目。通常，对话设计师或 AI 训练师的工作是利用这些数据通过聚类非结构化训练数据或话语来理解和优先排序意图。

例如，如果你正打算从实时聊天过渡到聊天机器人自动化，你可能有一个大量的历史聊天语料库。你将需要查看这些数据并决定哪些常见问题或意图被问得最多。同样，处理特定流动主题内容（如假日）的聊天机器人或语音助手可能需要定期更新以回答由外部因素引起的新问题。这些新问题可以通过聚类来突出显示。

### 什么是意图聚类？

意图聚类本质上意味着查看大量数据并寻找语义相似句子的分组。因此，对于我们的用例，我们将查看我们的非结构化聊天数据，并根据语义相似的意义进行分组，以便我们可以开始浮现意图。

在过去，通过聚类进行意图浮现的过程涉及使用嵌入的复杂过程，或者使用如 HumanFirst 之类的专有解决方案。这个过程将遵循以下步骤：

1.  **数据收集**：组装你想要分析的文本数据语料库。这可能包括从科学文章的集合到用户生成内容的数据库。

1.  **嵌入生成**：使用 TensorFlow 等机器学习库、Word2Vec 等服务或**双向编码器表示的 Transformer**(**BERT**)生成文本数据的相似性嵌入。这些嵌入是高维向量表示，捕获文本的语义内容。因此，如果一个文本与另一个文本具有相似的意义，那么嵌入将更接近。

1.  **聚类算法**：将无监督机器学习算法（如 K-means）应用于生成的嵌入。这些算法根据点之间的密度和距离将高维嵌入空间分离成簇。

1.  **聚类分析**：一旦完成，应该有一组包含语义相似文本的话语簇。分析这些簇可以揭示意图或问题，然后可以用来训练基于意图的机器学习模型。

在这个背景下，聚类是一种无监督技术，它使用语义嵌入将相似文本分组在一起，帮助发现大量非结构化文本数据集中的潜在意图。

### 学习如何使用 ChatGPT 创建意图簇

当你遵循简单的提示工程规则时，使用 ChatGPT 实现这一点很简单：指令、上下文、输入数据和输出指示符。我们将在*第四章*中更详细地介绍更多提示工程示例。

对于我们的意图，提示可能看起来像以下这样。重要的是要注意，设计提示是一个迭代过程，有时需要实验才能得到你想要的结果。

对于我们的第一个示例，我们正在创建一个用于我们大学网站的聊天机器人，该机器人将回答有关大学的问题。我们有一组从实时聊天实现中收集的非结构化话语。

对于这个示例，我们在提示中使用指令、输入数据和输出指示符。指令和输入数据处理告诉 ChatGPT 我们想要什么以及使用哪些话语。输出指示符还显示了我们想要的输出格式。在这种情况下，JSON 将最有用，因为我们正在考虑将意图或话语作为自动化过程的一部分使用。该模型还将处理其他输出格式，如 CSV。它看起来像这样：

```py
Look at the following utterances from livechat transcripts and cluster them into intents
Utterances:   "can you tell me where is the college located", "where's the college", "where is college located", "how can I reach college","whats is the college address", "whats the Syllabus for IT", "what is the Information Technology syllabus"," "Whats my next lecture"
Output: Return as JSON in the following format-{"intent": {name},"utterance":{utterances}}
```

结果应该看起来像以下这样：

```py
[
  {
    "intent": "Get_College_Location",
    "utterances": ["can you tell me where is the college located", "where's the college", "where is college located", "how can I reach college"]
  },{
    "intent": "Get_College_Address",
    "utterances": ["whats is the college address"]
  }
]
```

除了通过 ChatGPT 界面手动创建意图外，还可以通过直接使用 ChatGPT API 进一步自动化此过程。

### 将示例扩展以创建意图

考虑到我们的聊天机器人用例，我们可能希望通过创建更多意图来构建我们的意图集。例如，如果你的实时聊天数据有限，并且你想扩展到其他意图，你可以使用 ChatGPT 执行一个任务，通过使用一些现有的意图来创建更多意图和话语。

让我们通过使用我们之前示例的输出，提示意图聚类来创建我们的提示，您可以在提示中看到`{``input_data}`占位符：

```py
Look at the following intents and utterances for a chatbot which will answer questions about a college, these JSON objects consist of an array of intent objects, each of which includes a name and a list of corresponding utterances.
Create another 5 examples of intents and ensure that each intent has 10 utterances:
Intents:
{input_data}
Output: Return as JSON in the following format-{"intent": {name},"utterance":{utterances}}
```

如果我们运行这个提示，我们可以看到这很有效。这是输出：

```py
[
  {
    "intent": "Get_College_Events",
    "utterances": ["Whats happening on Tuesday","What are the upcoming events?", "Can you tell me about any college events?", "What's happening on campus?", "What events are scheduled this month?", "Is there a college event this week?", "Are there any sports events coming up?", "When is the next cultural fest?", "Are there any seminars or workshops?", "What's the college event calendar?", "Are there any academic events happening soon?"]
  },
  {
    "intent": "Inquire_About_College_Facilities",
    "utterances": ["What facilities are available in the college?", "Does the college have a library?", "Can you tell me about the sports facilities?", "What are the lab facilities in the college?", "Are there any on-campus cafeterias?", "Does the college have a gym?", "Are there any hostel facilities?", "Does the college have a swimming pool?", "Tell me about the medical facilities", "Are there any recreational facilities in the college?"]
  },
...
 ]
```

## 理解语句和实体生成

在现代对话式人工智能系统中，意图和实体是至关重要的组成部分，它们使系统能够理解和处理人类语言，从而实现更自然、更有效的对话。创建这些元素是对话设计师最常见的任务之一。这些元素随后可以用于训练我们用于对话式人工智能平台的机器学习模型。

### 创建语句

在前面的*如何使用 ChatGPT 创建意图聚类*部分，我们探讨了如何从非结构化训练数据中聚类语句以提取意图。ChatGPT 也可以用来创建语句。

如果我们没有足够多的语句来满足特定的意图，我们可以提示 ChatGPT 为特定意图创建一组语句，如下所示：

```py
Consider a "Get_College_Events" intent for a chatbot which will answer questions about college events.
Create another 10 examples of utterances for each intent:
Output: Return as JSON in the following format-{"intent": {name},"utterance":{utterances}}
```

结果令人惊讶地好：

```py
[
  {
    "intent": "Get_College_Events",
    "utterances": ["What's happening in college this week?", "Tell me about the upcoming college events.", "Are there any events in college next month?", "Is there a sports meet this year?", "What are the details of the cultural fest?", "When is the next science fair?", "Is there any guest lecture scheduled soon?", "When is the annual college fest?", "Are there any workshops I can attend?", "Can you give me a list of events for this semester?"]
  }
]
```

我们还可以更具体地说明语句的类型。例如，如果我们想要更广泛的语句范围，我们可以提示 ChatGPT 提供更长的尾部示例。让我们以“获取大学位置”的意图用例为例，向现有语句中添加长尾语句：

```py
Look at the following intent for a chatbot that will answer questions about a college, these JSON objects consist of an array of intent objects, each of which includes a name and a list of corresponding utterances.
Create another 5 examples of long tail utterances for each intent and add to the array of utterances
Utterances:
[
  {
    "intent": "Get_College_Location",
    "utterances": ["can you tell me where is the college located", "where's the college", "where is college located", "how can I reach college"]
  }
]
Output: Return as JSON in the following format-{"intent": {name},"utterance":{utterances}}
```

这使得我们的 NLU 训练数据集的语句范围更广：

```py
[
  {
    "intent": "Get_College_Location",
    "utterances": ["can you tell me where is the college located", "where's the college", "where is college located", "how can I reach college", "I am new to this city, can you guide me to the college?", "Can you provide the exact location of the college?", "I am not familiar with the area, could you help me find the college?", "I would like to visit the college, can you tell me its location?", "I need directions to the college, can you help?", "Could you give me the location details of the college?"]
  }
]
```

每个创建的语句都符合我们的标准，具有更对话式和复杂的（长尾）格式。

小贴士

记住，人工训练数据，即使是 LLM 创建的，也不能替代真实使用数据。这一点在创建语音语句时尤其正确，因为 ChatGPT 是在文本数据上训练的。

### 创建实体

ChatGPT 非常擅长的一项生成任务是创建实体。

实体是与意图相关的具体信息片段。它们是提供用户请求详细信息的变量。

例如，考虑我们的`Get_College_Events`意图。为了正确回答问题，我们需要了解一些事情：事件的日期和类型。虽然系统日期通常由对话式人工智能平台支持，但为了训练 NLU，通常需要创建自定义实体。对于这个特定用例，事件类型就是我们需要填充的自定义实体。我们可以请 ChatGPT 帮助我们。记住，提示要尽可能具体。在这种情况下，我们要求输出格式正确，以便导入到我们的对话式人工智能平台：

```py
Consider a "Event_Type" entity which needs to be extracted from utterances for the Get_College_Events for a chatbot which will answer questions about college events.
Create 5 examples of entity_val and 5 synonyms  for each entity type:
Output: Return as JSON in the following format-{"entity_val": 
{entity_val},"synonyms  ": [synonyms]  }
```

结果是一套全面的实体类型及其对应同义词：

```py
[
  {
    "entity_val": "seminar",
    "synonyms": ["workshop", "conference", "symposium", "meeting", "forum"]
  },
  {
    "entity_val": "sports meet",
    "synonyms": ["athletic meet", "sports event", "sports day", "athletic event", "games"]
  },…
]
```

通过正确地查找和处理实体，ChatGPT 可以快速创建精确且有用的响应。

在没有客户对话数据的情况下，ChatGPT 的语句可以作为启动聊天机器人进行进一步迭代的好方法。

## 使用 ChatGPT 帮助编写您的对话

编写引人入胜的聊天和语音对话是对话设计师角色的重要部分。那么，ChatGPT 如何帮助，您甚至应该考虑使用它吗？

### 理解编写对话的挑战

作为对话设计师，我们工作中最吸引人但最具挑战性的方面之一是制作聊天机器人或语音助手的回应。毕竟，这是您展示自己的机会。好的对话是平淡的对话体验和真正吸引用户的对话体验之间的区别。

创建示例对话，确保对话流中各部分之间的平稳过渡，并为高度具体的环境制定精确的回应，这些都需要时间和仔细的考虑。这一切都是在应对满足截止日期的压力、保持利益相关者知情并采取反馈行动的同时完成的。以下是一些需要考虑的事项：

+   保持与您的 AI 助手角色相呼应的声音和语调

+   遵守公司内容指南

+   使用与您可能需要提供给用户的内含内容相一致的语言

+   能够熟练且一致地处理错误情况

+   在说得太多和过于冗长之间找到平衡

+   确保精心制作的回应在各种模态中都是有效的，而不仅仅是理论上

+   解决可能影响我们使用语言的内隐偏见，并可能对最终用户产生影响

+   仔细遵循法律和合规指南，以确保我们使用的语言是正确的，不会让您陷入法律麻烦

从本质上讲，编辑对话性对话是一系列复杂的任务，需要创造力和精确性的微妙平衡。

### 使用 ChatGPT 编辑您的对话

ChatGPT 可以执行多项任务来增强您的对话。让我们深入探讨一些例子，例如重新编写以提高可读性、检查内容指南或为更好的对话体验创建副本的不同版本。

#### 重新编写以提高可读性

如果您有不适合您用例的副本，但仍然需要传达包含的信息，您可以要求 ChatGPT 为您重新编写。以下是一个执行此操作的提示示例。这是一个临床试验患者信息表的副本：

```py
I'm writing a chatbot response for my "what is the purpose of this drug trial" intent. Revise this dialogue to make it more understandable for clinical trial patients and ensure it's under 100 words:
Chronic kidney disease (CKD), affecting 1 in 10 UK adults, is a progressive loss of kidney function that can lead to complete kidney failure, requiring dialysis or transplantation. High blood pressure treatment, especially with drugs like Angiotensin Converting Enzyme inhibitors (ACEi) or Angiotensin Receptor Blockers (ARBs), can slow CKD progression. However, recent research suggests that stopping these drugs in advanced CKD (stage 4 or 5) can stabilize kidney function and delay dialysis, indicating these drugs might contribute to kidney function decline. To confirm this, the ACEi trial will recruit 500 CKD stage 4 or 5 patients on ACEi/ARBs treatment, randomly allocating half to continue and half to stop their treatment. The trial will also examine other effects of stopping these drugs, such as heart effects and participant quality of life.
```

从结果来看，您可以看到 ChatGPT 在总结内容方面做得非常出色：

```py
The STOP-ACEi trial is for CKD patients, a kidney condition affecting 1 in 10 UK adults. We're studying if stopping certain blood pressure drugs (ACEi/ARBs) can help stabilize kidney function and delay the need for dialysis. We'll have two groups: one continues the medication, the other stops. We aim to understand the best approach for CKD treatment
```

用您自己的示例文本尝试这个方法。不断迭代您的提示，直到您对结果满意为止。

另一个良好的对话创建示例是为特定目的重新编写内容。例如，如果我们想从描述性副本中创建一组简洁的说明，我们可能会使用以下方法：

```py
I'm writing copy for my dog training chatbot intent "how do i teach my dog to sit". Rewrite these instructions into a set of steps in the style of Barbara Woodhouse. Each step can only be 150 characters. Also write an introductory sentence:
Training a puppy to sit is one of the most basic commands and is useful in various situations. Start by holding a treat close to your puppy's nose and then move your hand up, allowing their head to follow the treat and causing their bottom to lower. Once they're in sitting position, say "sit", give them the treat and share affection. Remember, repetition is key. Practice this command a few times each day and be sure to reward your puppy for sitting even without command, to reinforce this behavior.
```

提供具有上下文、输入和输出指令的简洁说明，可以得到良好的结果，并且是我们可以工作的内容。用您自己的例子尝试这个方法。

认为我们可以自动化响应创建是很诱人的。然而，这样做并不明智。ChatGPT 创建的内容只是一个起点。确保 ChatGPT 提供正确的信息，同时不改变意义，并遵守你的对话设计要求是至关重要的。

# 使用 ChatGPT 创建用户画像

用户画像在以用户为中心的设计中是关键组成部分，因为它们使组织能够了解他们的目标受众，并构建满足他们需求的解决方案。本节将深入探讨如何使用 ChatGPT 生成复杂和智能的用户画像。

真实世界案例

认识 LuxeLocks，这是我们虚构的公司，专门销售奢华护发产品。LuxeLocks 热衷于为其客户提供高端、高质量的产品，将他们的头发变成荣耀的象征。为了将客户服务提升到下一个层次，并随时回答查询，LuxeLocks 决定实施聊天机器人。

这个聊天机器人将被设计成以各种方式协助客户，从帮助他们根据头发类型和问题选择完美的产品，到回答关于产品使用和成分的常见问题，甚至协助订单跟踪和退货。

通过提出问题和呈现场景，你可以深入了解用户偏好、行为和痛点。

画像创建基于硬事实和特定用户数据。使用 ChatGPT 不是真实研究的捷径，但它可以帮助。

## 使用 ChatGPT 进行用户研究

在本书中概述通过调查、访谈和市场研究收集关于目标受众信息的过程超出了范围。这也无法替代。以下是一些外部资源，以了解更多关于此过程的信息：

+   [研究流程步骤](https://www.questionpro.com/blog/research-process-steps/)

+   我推荐的每本 UX 书籍及其原因

+   [为什么你只需要测试 5 个用户](https://www.nngroup.com/articles/why-you-only-need-to-test-with-5-users/)

+   [用户体验](https://www.usability.gov/what-and-why/user-experience.html)

然而，使用 ChatGPT 帮助你提出一些进行研究的想法是可行的。你可以通过以下提示来完成：

```py
Create an outline for market research, including surveys, interviews, and additional market research methods for a luxury haircare brand:
```

至少，输出可以形成一个基础，开始你的研究。这可能看起来是这样的：

```py
A. Customer Interviews:
Purpose: Gain deeper insights into the customer experience.
Sample Questions:
Describe your haircare routine.
What led you to try our brand?
Are there any challenges you have faced with our products?
B. Expert Interviews:
Purpose: Gather insights from industry experts, such as hairstylists or dermatologists.
Sample Questions:
What trends are you currently noticing in the luxury haircare market?
How do you think our brand is perceived in the market?
IV. Additional Market Research:
A. Competitor Analysis:
Purpose: Understand the competitive landscape.
Identify key competitors in the luxury haircare market.
Analyze their product range, pricing strategy, marketing efforts, and customer reviews. Identify their strengths and weaknesses.
B. Focus Groups:
Purpose: Gather qualitative data on potential new products or marketing strategies.
Assemble a diverse group of individuals who are part of the target market. Present them
with new product concepts, packaging designs, or marketing materials.
Facilitate a discussion to gather their feedback and impressions.
C. Market Trend Analysis:
Purpose: Identify current trends in the luxury haircare market.
Review industry reports and publications for data on emerging trends.
Analyze social media and influencer content to identify popular products and ingredients.
…..
```

一旦你有了详细信息，你还可以使用 ChatGPT 来分析数据，并揭示一些关于用户群体的细节：

```py
Based on the following data {data}, please describe the main demographic characteristics and preferences of our target audience.
Preferences:
Characteristics:
```

LuxeLocks 用户的偏好和特征可以帮助我们构建用户画像，并为我们的聊天机器人画像提供指导。

## 创建我们的用户和聊天机器人画像

您可以使用从早期提示中创建的人口统计特征来帮助您创建用户和聊天机器人角色。

### 创建我们的用户角色

让我们提示 ChatGPT 创建一个角色，并使用我们之前提示中的偏好和特征结果。在以下提示中，将`*characteristics*`和`*preferences*`替换为这些值：

```py
Create a user persona description for a luxury haircare brand. Users tend to have these Characteristics: *characteristics*
Preferences: *preferences*
The persona is to be the user of our luxury haircare brand over multiple channels. The brand sells high end haircare products which are targeted at specific hair types:
Use the following template:
Name: {first_name}
Age: {age}
Relationship status: {relationship_status}
Occupation: {occupation}
Location: {location}
Interests: {interests}
Bio: {bio}
{first_name}'s motivations:
1\. {motivation 1}
2\. {motivation 2}
3\. {motivation 3}
{first_name}'s goals:
1\. {goal 1}
2\. {goal 2}
3\. {goal 3}
{first_name}'s frustrations:
1\. {frustration 1}
2\. {frustration 2}
3\. {frustration 3}
{first_name}'s Key concern/barrier to purchase:
1\. {barrier 1}
2\. {barrier 2}
3\. {barrier 3}
A quote from {first_name}'s: {quote}
```

我们可以在后续问题中用这个角色描述来询问 ChatGPT。

### 创建我们的聊天机器人角色

您还可以使用 ChatGPT 来创建您的聊天机器人或虚拟助手的角色。与任何提示一样，您可以调整包含的信息量，但我发现更详细的提示会产生更好的结果。尝试使用类似以下的内容：

```py
Create our luxury haircare chatbot persona:
Name and Title:
What is the chatbot's name?
Does the chatbot have a title or role (e.g., Customer Care Assistant, Haircare Expert)?
Background and Purpose:
Why was this chatbot created?
What is its primary role in interacting with customers?
```

我们现在可以使用用户和聊天机器人的角色来指导 ChatGPT 在创建对话和交互相关的进一步任务中。

# 模拟对话

ChatGPT 可以用来与具有特定角色的潜在用户创建样本对话，这可以帮助创建您的对话流程。我再次强调，这不应取代对话流程设计过程，而是一种有效查看特定用例的方法。

## 样例对话是什么？

样例对话包括用户和代理之间特定多轮对话的例子。

对于参与对话体验设计和开发的人来说，这些对话片段是必不可少的工具。它们使我们能够探索对话可能遵循的潜在路径，并评估哪些分支需要考虑并整合到最终的对话设计中。

```py
User: What's the luggage limit for my flight to Paris?
Chatbot: May I know the airline you are flying with?
User: It's AirExcellence.
Chatbot: For AirExcellence, you're allowed one carry-on up to 10 kg and one checked bag up to 23 kg. Carry-on size: 55 cm x 40 cm x 23 cm.
User: Thanks!
Chatbot: You're welcome! Have a great flight.
```

现在我们将来看如何创建我们自己的聊天机器人对话。

## 创建示例对话

在以下提示中，我们提供有关语气、语言、澄清和对话元素的信息。我们希望在这种情况下提供这些信息，以确保我们已经解决了问题：

```py
Generate a conversation between a user and a customer support chatbot. The user is trying to troubleshoot a problem with their internet connection. The chatbot should follow these content guidelines:
1\. Maintain a polite and professional tone throughout the conversation.
2\. Use simple, clear language that is easy for the user to understand.
3\. Ask clarifying questions to accurately diagnose the problem.
4\. Provide step-by-step instructions to resolve the issue.
5\. Confirm that the user's problem has been resolved before ending the conversation.
6\. If the chatbot cannot resolve the issue, it should suggest the user to contact a human support representative.
User: "Hello, I'm having trouble with my internet connection. It's really slow."
```

这产生了一个好的对话示例。尝试不同的变体是值得的，这样您可以建立起对对话可能走向的理解。

在这个阶段，我们正在设计我们的流程，但稍后我们也会看到，我们可以将这种技术作为我们测试过程的一部分。

## 创建带有角色的示例对话

我们可以通过将更多关于用户和代理角色的信息传递到提示中，来构建基于样本对话片段的创建，这些片段到目前为止主要集中在特定的用例上：

```py
We are going to create some sample dialogues between our luxury haircare chatbot and a user:
User persona:
Name: Sophia
Age: 42
Relationship status: Single
Occupation: Corporate Wellness Coach
Location: London, UK
Interests: Pilates, Culinary Arts, Wine Tasting, Sustainable Fashion, Volunteering
...
```

这里的技巧是向 ChatGPT 提供您的用户和聊天机器人角色的详细信息。一旦您向 ChatGPT 提供了这些信息，您就可以给出后续提示，要求进一步提供对话或用例的示例。

当查看示例用例时，这种技术同样适用。在下一个提示中，我们将在此基础上继续，并要求另一个对话示例：

```py
Using the persona descriptions of Sophia and Lila, create a conversation between them in which Sophia is asking about the best shampoo for her hair type.. Lila asks whether Sophie would like to do her quiz so she can learn about her hair.
```

ChatGPT 的回应在这里很令人印象深刻，因为它更进一步，创建了一些测验问题，尽管最好的选择是包括您的测验细节。

使用详细提示来创建对话示例的技术也可以用于测试和进一步的迭代，我们将在下一节中介绍。

# 使用 ChatGPT 进行对话设计和迭代测试

对话设计是一个迭代的过程。自动化智能体总是在变化和改进。在本节中，我们将探讨使用 ChatGPT 测试和改进我们的智能体。

## 使用 ChatGPT 进行测试

在前面的章节中，我们看到了 ChatGPT 可以通过创建对话来帮助我们设计，从而在简化设计过程中成为一个有价值的工具。在部署之前测试我们的智能体时，我们可以采取类似的方法。

我们通常根据我们认为他们会被问到的问题来创建智能体。如果我们是从零开始启动一个对话人工智能项目，几乎没有训练数据，这种情况就更加可能。如果是这种情况，在将智能体发布到野外之前测试您的智能体就更加重要。

即使对话人工智能项目有内部测试人员，或者您在 alpha 或 beta 发布之前可以向朋友或家人展示您的智能体，但在将其交给人类用户之前尽可能彻底地进行测试仍然很有价值。这就是 ChatGPT 可以发挥作用的地方。

### 测试的对话模拟

这个过程与设计过程类似。我们可以创建模拟，其中 ChatGPT 扮演用户角色。然后，您可以构建一系列测试用户交互脚本，以便在无需真实用户参与的情况下与聊天机器人进行交互。

我们将在我们的奢华护发聊天机器人示例的上下文中考虑这一点。就像 ChatGPT 被用来创建实体和话语一样，我们也可以用它来生成反映现实对话的额外训练数据。

小贴士

如果您继续使用之前的 ChatGPT 会话，我们已经在提示中设置了角色。如果您开始一个新的会话，请记住将这些添加到您的提示中，以设置上下文。

因此，我们将简单地要求 ChatGPT 创建以问题形式呈现的训练数据结果，这些问题涵盖了与护发、产品推荐、可持续性和个性化流程相关的各种主题，反映了 Sophia 的角色和兴趣：

```py
Create 10 questions which Sophia could ask Lila
```

我们还可以提示更多长尾示例，以便我们可以用我们的智能体进行测试：

```py
Create 10 long tail questions which Sophia could ask Lila
```

一旦我们有一组训练问题，您就可以手动测试我们的智能体，或者如果您的聊天机器人提供 API，您可以创建自动化。

### 创建更具体的测试脚本

对于更复杂的对话交互，ChatGPT 也可以用来创建相同的流程示例，以便您可以使用它们来测试您的智能体。

考虑一个用户对其订单状态感兴趣的具体询问。这可以用作提示 ChatGPT 创建具有不同对话转折的其他示例：

```py
Consider the following enquiry from Sophie about her order:
####
Sophia: Hi, placed an order for the Curl Harmony Shampoo yesterday. Can you tell me when it will be shipped?
Lila: Welcome back, Sophia! I'm glad to assist you. Your order for the Curl Harmony Shampoo is scheduled to be shipped tomorrow. You'll receive an email confirmation with the tracking details once it's on its way. Is there anything else I can help you with?
Sophia:
#####
Create 3 examples of this conversation when Sophie asks follow up questions
```

然后，您可以通过这些脚本示例运行您的聊天机器人，以查看其表现。

### 测试失败和边缘情况

对于同一用例的其他对话结果也可以考虑并使用 ChatGPT 进行处理。例如，你可以使用 ChatGPT 来覆盖设计中需要考虑的边缘情况。

一些例子可能涉及用户偏离了快乐路径，或者以下例子，其中存在 Lila 提供答案所依赖的服务问题：

```py
Give 1 example where Lila has technical issues so she can't access any order details.
```

你也可以提示 ChatGPT 在这里提供自己的建议：

```py
Provide 3 examples of what else could go wrong with the conversation
```

这些结果可以用来检查你在设计和实施过程中是否遗漏了任何内容。

### 使用 ChatGPT 迭代你的聊天机器人

一旦你的代理有真实用户进行测试，你就可以使用转录和交互日志来查看如何通过向 ChatGPT 提供示例来改进你的对话设计。

```py
Look at the following chat transcript:
{transcript}
Provide an example of how to improve this conversation
```

这尤其有用，如果你想要查看失败的对话、以不满意结果结束的对话或以不满意的用户结束的对话。它让你能够查看你可以改进对话流程的方式。

# 摘要

在本章中，我们重点介绍了如何将 ChatGPT 应用于对话设计。我们探讨了对话设计师的多重角色，并更详细地审视了构成对话设计过程的许多任务。我们了解到，其中一些任务可以在 ChatGPT 的帮助下完成。

我们深入探讨了涉及的实际任务，例如理解用户需求、设计角色、创建对话流程和编写对话。

我们还了解到，理解 ChatGPT 并非是优秀对话设计实践的替代品，而是一个功能极其强大的工具。

我们将 ChatGPT 作为对话设计助手的使用核心是有效实施提示工程。我们开始研究提示工程所涉及的原则。在*第四章*中，我们将更详细地探讨提示工程，以深入了解这一新兴领域。在下一章中，我们将探讨使用 ChatGPT 的不同方式。

# 进一步阅读

以下链接是帮助你在本章中使用的资源：

+   [`chat.openai.com/`](https://chat.openai.com/)

+   [`www.deepset.ai/blog/the-beginners-guide-to-text-embeddings`](https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings)

# 第二部分：使用 ChatGPT、提示工程和探索 LangChain

本部分侧重于 ChatGPT 的实际应用、提示工程的复杂性以及深入探讨 LangChain。你将探索与 ChatGPT 交互的不同方式，无论是通过网页界面、API 还是官方库。你还将深入了解提示工程，学习如何制作有效的提示。最后，本节介绍了 LangChain，指导你了解其基本和高级用法，包括调试技术、内存管理和利用代理增强功能。

本部分包含以下章节：

+   *第三章*, *ChatGPT 精通 – 解锁其全部潜力*

+   *第四章*, *使用 ChatGPT 进行提示工程*

+   *第五章*, *LangChain 入门*

+   *第六章*, *使用 LangChain 进行高级调试、监控和检索*
