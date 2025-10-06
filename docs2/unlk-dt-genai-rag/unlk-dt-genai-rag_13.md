# <st c="0">13</st>

# <st c="3">使用提示工程来提升 RAG 努力</st>

<st c="51">快速问答，您使用什么从大型语言</st> **<st c="105">模型</st>** **<st c="120">（</st>**<st c="127">LLM</st>**<st c="130">）** 中生成内容？</st>

<st c="133">一个提示！</st>

<st c="143">显然，提示是任何生成式 AI 应用的关键元素，因此任何</st> **<st c="234">检索增强生成</st>** <st c="264">(</st>**<st c="266">RAG</st>**<st c="269">) 应用。</st> <st c="285">RAG 系统结合了信息检索和生成语言模型的能力，以提升生成文本的质量和相关性。</st> <st c="432">在此背景下，提示工程涉及战略性地制定和优化输入提示，以改善相关信息的检索，从而提高生成过程。</st> <st c="639">提示是生成式 AI 世界中另一个可以写满整本书的领域。</st> <st c="739">有许多策略专注于提示的不同领域，可以用来改善您 LLM 使用的成果。</st> <st c="873">然而，我们将专注于更具体于</st> <st c="961">RAG 应用</st> 的策略。

<st c="978">在本章中，我们将集中精力探讨以下主题：</st>

+   <st c="1054">关键提示工程概念</st> <st c="1087">和参数</st>

+   <st c="1101">针对 RAG 应用的具体提示设计和提示工程的基本原理</st>

+   <st c="1192">适应不同 LLM 的提示，而不仅仅是</st> <st c="1241">OpenAI 模型</st>

+   <st c="1254">代码实验室 13.1 – 创建自定义</st> <st c="1287">提示模板</st>

+   <st c="1303">代码实验室 13.2 –</st> <st c="1320">提示选项</st>

<st c="1037">到本章结束时，您将具备 RAG 提示工程的坚实基础，并掌握优化提示以检索相关信息、生成高质量文本以及适应特定用例的实用技术。</st> <st c="1598">我们将从介绍提示世界中的关键概念开始，首先是</st> <st c="1709">提示参数</st>。

# <st c="1727">技术要求</st>

<st c="1750">本章的代码放置在以下 GitHub</st> <st c="1811">仓库中：</st> [<st c="1823">https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_13</st>](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_13 )

# <st c="1920">提示参数</st>

<st c="1938">在大多数 LLM 中存在许多共同参数，但我们将讨论一个可能对您的 RAG 努力产生影响的较小子集：温度、top-p，以及种子。</st> <st c="2124">。</st>

## <st c="2133">温度</st>

<st c="2145">如果你将你的</st> <st c="2166">输出视为一系列</st> **<st c="2189">标记</st>**<st c="2195">，那么在基本意义上，一个 LLM（大型语言模型）是根据你提供的数据和它已经生成的先前标记来预测下一个单词（或标记）的。</st> <st c="2241">LLM 预测的下一个单词是表示所有潜在单词及其概率的概率分布的结果。</st> <st c="2462">LLM 预测的下一个单词是表示所有潜在单词及其概率的概率分布的结果。</st>

<st c="2482">在许多情况下，某些单词的概率可能远高于其他大多数单词，但 LLM 仍然有一定概率选择其中不太可能出现的单词。</st> <st c="2669">温度是决定模型选择概率分布中较后单词的可能性大小的设置。</st> <st c="2801">换句话说，这允许你使用温度来设置模型输出的随机程度。</st> <st c="2907">你可以将温度作为一个参数传递给你的 LLM 定义。</st> <st c="2973">这是可选的。</st> <st c="2989">如果你不使用它，默认值是</st> `<st c="3026">1</st>`<st c="3027">。你可以设置温度值在</st> `<st c="3071">0</st>` <st c="3072">和</st> `<st c="3077">2</st>`<st c="3078">之间。</st> 较高的值会使输出更加随机，这意味着它将强烈考虑概率分布中较后的单词，而较低的值则会做</st> <st c="3238">相反的事情。</st>

<st c="3251">简单的温度示例</st>

<st c="3278">让我们回顾一个简单的</st> *<st c="3319">下一个单词</st>* <st c="3328">概率分布的例子，以说明温度是如何工作的。</st> <st c="3391">假设你有一个句子</st> `<st c="3423">The dog ran</st>` <st c="3434">，并且你正在等待模型预测下一个</st> *<st c="3484">单词</st>*<st c="3493">。假设基于这个模型的训练和它考虑的所有其他数据，这个预测的简单条件概率分布如下：</st> <st c="3693">如下：</st>

`<st c="3704">P("next word" | "The dog ran") = {"down": 0.4, "to" : 0.3, "with": 0.2, "</st>``<st c="3778">away": 0.1}</st>`

<st c="3790">总概率加起来是</st> `<st c="3825">1</st>`<st c="3826">。最可能的词是</st> `<st c="3852">down</st>` <st c="3856">，其次是</st> `<st c="3892">to</st>`<st c="3894">。然而，这并不意味着</st> `<st c="3929">away</st>` <st c="3933">永远不会出现在推理中。</st> <st c="3970">模型将对这个选择应用概率模型，有时，随机地，不太可能的词会被选中。</st> <st c="4095">在某些场景中，这可能是您 RAG 应用的优势，但在其他情况下，这可能是劣势。</st> <st c="4204">如果您将温度设置为</st> `<st c="4234">0</st>`<st c="4235">，它将只使用最可能的词。</st> <st c="4276">如果您将其设置为</st> `<st c="4293">2</st>`<st c="4294">，则更有可能查看所有选项，并且大多数情况下会随机选择不太可能的词。</st> <st c="4403">换句话说，您可以通过增加温度来增加模型的随机性。</st>

<st c="4497">我们从一开始就在使用温度，将其设置为零。</st> <st c="4570">以下是添加的行：</st> <st c="4595">（此处省略了代码行）</st>

```py
 llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

<st c="4664">这里的目的是使我们的代码实验室的结果更可预测，以便当您运行它们时，您可以得到类似的结果。</st> <st c="4796">您的结果可能会有所不同，但至少在</st> `<st c="4858">0</st>` <st c="4859">温度下，它们有更大的可能性</st> <st c="4902">相似。</st>

<st c="4916">您可能并不总是想使用</st> `<st c="4950">0</st>` <st c="4951">温度。</st> <st c="4965">考虑以下场景，例如当您希望从 LLM 获得更</st> *<st c="5014">有创意</st>* <st c="5022">的输出时，您可能想利用温度在您的</st> <st c="5114">RAG 应用</st>中。</st>

<st c="5130">温度</st> <st c="5142">和 top-p 在某种程度上是相关的，因为它们都管理 LLM 输出的随机性。</st> <st c="5232">然而，它们之间有差异。</st> <st c="5264">让我们讨论 top-p，并谈谈这些</st> <st c="5310">差异是什么。</st>

## <st c="5326">Top-p</st>

<st c="5332">与温度类似，top-p 也可以帮助您将随机性引入模型输出。</st> <st c="5428">然而，温度处理的是对输入随机性的总体强调，而 top-p 可以帮助您通过那种随机性针对概率分布的特定部分。</st> <st c="5630">在提供的简单示例中，我们讨论了</st> <st c="5679">概率分布：</st>

```py
 P("next word" | "The dog ran") = {"down": 0.4, "to" : 0.3,
    "with": 0.2, "away": 0.1}
```

<st c="5789">记住，我们提到这里表示的总概率加起来是</st> `<st c="5868">1.0</st>`<st c="5871">。使用 top-p，你可以指定你希望包含的概率部分。</st> <st c="5958">例如，如果你将 top-p 设置为</st> `<st c="5998">0.7</st>`<st c="6001">，它将只考虑这个概率分布中的前两个词，这些词的总和是概率分布中的第一个</st> `<st c="6105">0.7</st>` <st c="6108">(1.0 中的)。</st> <st c="6155">使用温度时，你没有这种针对性的控制。</st> <st c="6219">Top-p 也是可选的。</st> <st c="6243">如果你不使用它，默认值是</st> `<st c="6280">1</st>`<st c="6281">，这意味着它考虑了</st> <st c="6304">所有选项。</st>

<st c="6316">你可能想同时使用温度和 top-p，但这可能会变得非常复杂且不可预测。</st> <st c="6424">因此，通常建议使用其中之一，而不是同时使用。</st> <st c="6497">同时使用。</st>

| **<st c="6507">LLM 参数</st>** | **<st c="6521">结果</st>** |
| --- | --- |
| <st c="6529">温度</st> | <st c="6541">一般随机性</st> |
| <st c="6560">Top-p</st> | <st c="6566">针对性随机性</st> |
| <st c="6585">温度 +</st> <st c="6600">top-p</st> | <st c="6605">不可预测的复杂性</st> |

<st c="6630">表 13.1 – 展示每个 LLM 参数的结果类型</st>

<st c="6695">接下来，我们将学习如何</st> <st c="6721">使用 top-p 与你的模型结合。</st> <st c="6748">这与我们之前使用的其他参数略有不同，因为你必须将其作为</st> `<st c="6855">model_kwargs</st>` <st c="6867">变量的一部分传递，这看起来像这样：</st>

```py
 llm = ChatOpenAI(model_name="gpt-4o-mini",
    model_kwargs={"top_p": 0.5})
```

<st c="6973">`<st c="6978">model_kwargs</st>` <st c="6990">变量是传递那些没有直接集成到 LangChain 但存在于 LLM 底层 API 中的参数的便捷方式。</st> <st c="7128">top-p 是这个 ChatGPT 模型的参数，但其他模型可能叫法不同或不存在。</st> <st c="7249">务必检查你使用的每个 API 的文档，并使用正确的引用来访问该模型的参数。</st> <st c="7372">现在我们已经了解了帮助定义我们输出中随机性的参数，让我们学习种子设置，这是</st> <st c="7508">旨在帮助我们控制</st> <st c="7537">不可控的随机性。</st>

## <st c="7563">种子</st>

<st c="7568">LLM 响应默认是非确定性的</st> <st c="7605">，这意味着推理结果可能因请求而异。</st> <st c="7679">然而，作为数据科学家，我们通常需要更确定性和可重复的结果。</st> <st c="7779">这些细节似乎相互矛盾，但情况并不一定如此。</st> <st c="7868">OpenAI 和其他人最近已经努力提供一些控制，以便通过提供对种子参数和</st> `<st c="8016">system_fingerprint</st>` <st c="8034">响应字段的访问来实现。</st>

<st c="8050">种子是许多涉及生成随机数或随机数据序列的软件应用中的常见设置。</st> <st c="8175">通过使用种子，你仍然可以生成随机序列，但你可以每次都生成相同的随机序列。</st> <st c="8290">这让你能够控制通过 API 调用接收（主要是）确定性的输出。</st> <st c="8377">你可以将种子参数设置为任何你选择的整数，并在你希望获得确定性输出的请求中使用相同的值。</st> <st c="8515">此外，如果你使用种子，即使与其他随机设置（如温度或 top-p）一起使用，你仍然可以（主要是）依赖接收相同的</st> <st c="8659">确切响应。</st>

<st c="8674">需要注意的是，即使使用了种子，你的结果仍然可能不同，因为你正在使用一个连接到服务的 API，该服务正在不断进行更改。</st> <st c="8865">这些更改可能导致随着时间的推移结果不同。</st> <st c="8918">例如 ChatGPT 这样的模型在其输出中提供了一个</st> `<st c="8951">system_fingerprint</st>` <st c="8969">字段，你可以将其相互比较，作为系统变化可能引起响应差异的指示。</st> <st c="9109">如果你上一次调用那个 LLM API 时</st> `<st c="9116">system_fingerprint</st>` <st c="9134">值发生了变化，而你当时使用了相同的种子，那么你仍然可能会看到不同的输出，这是由于 OpenAI 对其系统所做的更改造成的。</st>

<st c="9309">种子参数也是可选的，并且不在 LangChain 的 LLM 参数集中。</st> <st c="9405">因此，再次强调，就像</st> `<st c="9430">top-p</st>` <st c="9435">参数一样，我们必须通过</st> `<st c="9475">model_kwargs</st>` <st c="9487">参数传递它：</st>

```py
 optional_params = {
  "top_p": 0.5, "seed": 42
}
llm = ChatOpenAI(model_name="gpt-4o-mini", model_kwargs=optional_params)
```

<st c="9628">在这里，我们将种子参数与</st> `<st c="9675">top-p</st>` <st c="9680">参数一起添加到我们将传递给</st> `<st c="9746">model_kwargs</st>` <st c="9758">参数的参数字典中。</st>

<st c="9769">你可以探索许多其他模型参数，我们鼓励你这样做，但这些参数可能对你的</st> <st c="9879">RAG 应用影响最大。</st>

<st c="9955">我们将要讨论的下一个以提示为导向的关键概念是</st> **<st c="10017">镜头</st>** <st c="10021">概念，重点关注你提供给</st> <st c="10095">LLM</st> 的背景信息量。

# <st c="10103">尝试你的镜头</st>

**<st c="10118">无镜头</st>**<st c="10126">、**<st c="10128">单镜头</st>**<st c="10139">、**<st c="10141">少镜头</st>**<st c="10149">和**<st c="10155">多镜头</st>** <st c="10165">是在讨论你的提示策略时经常听到的术语。</st> <st c="10241">它们都源自同一个概念，即一个镜头是你给 LLM 的一个例子，以帮助它确定如何回应你的查询。</st> <st c="10378">如果这还不清楚，我可以给你一个</st> <st c="10425">例子来说明我所说的内容。</st> <st c="10461">哦，等等，这正是</st> <st c="10485">镜头概念背后的想法！</st> <st c="10520">你可以提供没有例子（无镜头）、一个例子（单镜头）或多个</st> <st c="10591">例子（少镜头或多镜头）。</st> <st c="10634">每个镜头都是一个例子；每个例子都是一个镜头。</st> <st c="10683">以下是你对 LLM 可能说的话的例子（我们可以称之为单镜头，因为我只提供了一个</st> <st c="10793">例子）：</st>

```py
 "Give me a joke that uses an animal and some action that animal takes that is funny. Use this example to guide the joke you provide:
Joke-question: Why did the chicken cross the road? Joke-answer: To get to the other side."
```

<st c="11030">这里的假设是，通过提供那个例子，你正在帮助引导 LLM 如何</st> <st c="11123">回应。</st>

<st c="11135">在 RAG 应用中，你通常会提供上下文中的例子。</st> <st c="11207">这并不总是如此，因为有时上下文只是额外的（但重要的）数据。</st> <st c="11298">然而，如果你在上下文中提供实际的问题和答案的例子，目的是指导 LLM 以类似的方式回答新的用户查询，那么你就是在使用镜头方法。</st> <st c="11510">你会发现一些 RAG 应用更严格地遵循多镜头模式，但这实际上取决于你应用的目标和可用的数据。</st>

<st c="11686">在提示中，例子和镜头并不是唯一需要理解的概念，因为你还需要了解指代你如何处理提示的术语之间的差异。</st> <st c="11880">我们将在下一节中讨论这些</st> <st c="11905">方法。</st>

# <st c="11921">提示、提示设计和提示工程回顾</st>

<st c="11980">在词汇部分</st> *<st c="12010">第一章</st>*<st c="12019">中，我们讨论了这三个概念及其相互作用。</st> <st c="12079">为了复习，我们提供了以下要点：</st>

+   **<st c="12121">提示**</st> <st c="12131">是指</st> <st c="12146">发送一个查询或</st> *<st c="12165">提示</st> <st c="12171">到</st> <st c="12175">一个 LLM。</st>

+   **<st c="12182">提示设计</st>** <st c="12196">指的是你采取的策略来</st> *<st c="12232">设计</st> <st c="12238">你将发送给 LLM 的提示。</st> <st c="12276">许多不同的提示设计</st> <st c="12304">策略在不同的场景下都有效。</st>

+   **<st c="12344">提示工程</st>** <st c="12363">更关注</st> <st c="12371">围绕你使用的提示的技术方面，以改进 LLM 的输出。</st> <st c="12475">例如，你可能将一个复杂的查询分解成两个或三个不同的 LLM 交互，</st> *<st c="12567">工程化</st> <st c="12578">它以实现</st> <st c="12600">更优的结果。</st>

<st c="12617">我们曾承诺在</st> *<st c="12661">第十三章</st> *<st c="12671">中重新审视这些主题，所以我们现在来履行这个承诺！</st> <st c="12720">我们不仅将重新审视这些主题，还会向你展示如何在代码中实际执行这些操作。</st> <st c="12816">提示是一个相对直接的概念，所以我们将会关注其他两个主题：设计和工程。</st>

# <st c="12932">提示设计对比工程方法</st>

<st c="12976">当我们讨论了在“*<st c="13009">射击</st> *<st c="13013">方法</st> *<st c="13032">中”的不同</st> *<st c="13046">射击</st> *<st c="13087">方法，这属于提示设计。</st> <st c="13087">然而，当我们用从 RAG 系统的其他部分提取的问题和上下文数据填写提示模板时，我们也实施了提示工程。</st> <st c="13254">当我们用来自系统其他部分的数据填写这个提示时，你可能记得这被称为“水化”，这是一种特定的提示工程方法。</st> <st c="13418">提示设计和提示工程有显著的交集，因此你经常会听到这两个术语被交替使用。</st> <st c="13539">在我们的案例中，我们将一起讨论它们，特别是它们如何被用来改进我们的</st> <st c="13643">RAG 应用。</st>

<st c="13659">在过去几年中，我看到了这些概念以许多不同的方式被描述，因此似乎我们的领域还没有形成对每个概念的完整定义，也没有在它们之间划清界限。</st> <st c="13863">为了理解这本书中的这些概念，我会描述提示设计和提示工程之间的区别是：提示工程是一个更广泛的概念，它不仅包括提示的设计，还包括用户与语言模型之间整个交互的优化和微调。</st>

<st c="14218">理论上，有无数种提示设计技术，都可以用来改进你的 RAG 应用。</st> <st c="14332">跟踪你拥有的选项并了解每种方法最适合哪种场景很重要。</st> <st c="14454">需要通过不同提示设计方法的实验来确定哪种最适合你的应用。</st> <st c="14577">提示设计没有一劳永逸的解决方案。</st> <st c="14635">我们将提供一些示例列表，但我们强烈建议你从其他来源了解更多关于提示设计的信息，并注意哪些方法可能有助于你的特定应用：</st> <st c="14817"></st>

+   **<st c="14834">镜头设计</st>**<st c="14846">：</st>

    +   <st c="14848">任何提示设计</st> <st c="14860">思维过程的起点</st> <st c="14889"></st>

    +   <st c="14904">涉及精心设计初始提示，使用示例帮助引导 AI 模型达到期望的输出</st> <st c="15006"></st>

    +   <st c="15020">可以与其他设计模式结合使用，以增强生成内容的质量和相关性</st> <st c="15120"></st>

+   **<st c="15137">思维链提示</st>**<st c="15164">：</st>

    +   <st c="15166">将复杂问题分解成更小、更易于管理的步骤，在每个步骤中提示 LLM 进行中间推理</st> <st c="15194"></st>

    +   <st c="15289">通过提供清晰、逐步的思维过程，提高 LLM 生成答案的质量，确保更好的理解和更准确的响应</st> <st c="15427"></st>

+   **<st c="15445">角色（</st>****<st c="15456">角色提示）</st>**

    +   <st c="15474">涉及创建一个基于用户群体或群体的代表性部分的虚构</st> <st c="15503">角色，包括姓名、职业、人口统计、个人故事、痛点</st> <st c="15669">和挑战</st>

    +   <st c="15683">确保输出与目标受众的需求和偏好相关、有用且一致，给内容增添更多个性和风格</st> <st c="15835"></st>

    +   <st c="15844">是开发符合用户需求的有效语言模型的有力工具</st> <st c="15928"></st>

+   **<st c="15936">密度链</st>** **<st c="15946">（摘要）</st>**<st c="15969">：</st>

    +   <st c="15971">确保 LLM 已经正确地总结了内容，检查是否有重要信息被遗漏，并且摘要是否足够简洁</st> <st c="16000"></st>

    +   <st c="16138">在 LLM 遍历摘要时使用实体密度，确保包含最重要的实体</st> <st c="16240"></st>

+   **<st c="16252">思维树（探索</st>** **<st c="16283">思维）</st>**

    +   <st c="16297">从一个初始提示开始，它</st> <st c="16334">生成多个思考选项，并迭代选择最佳选项以生成下一轮</st> <st c="16439">的思考</st>

    +   <st c="16450">允许更全面和多样化的探索想法和概念，直到生成</st> <st c="16539">所需的输出文本</st> <st c="16559">。</st>

+   **<st c="16571">图提示</st>**

    +   <st c="16587">一个专门为处理</st> <st c="16613">图结构数据</st> <st c="16653">而设计的新的提示框架</st>

    +   <st c="16674">使 LLM 能够根据图中的实体关系理解和生成内容</st> <st c="16773">。</st>

+   **<st c="16780">知识增强</st>**

    +   <st c="16803">涉及通过添加额外的、相关的信息来增强提示，以提高</st> <st c="16831">生成内容的</st> <st c="16908">质量和准确性</st>

    +   <st c="16925">可以通过将外部知识纳入</st> <st c="17017">提示</st> <st c="17017">的技术，如 RAG 来实现</st>

+   **<st c="17027">展示给我而不是告诉我</st>** **<st c="17048">提示</st>**

    +   <st c="17058">两种向生成式 AI 模型提供指令的不同方法：</st> *<st c="17135">展示给我</st>* <st c="17142">涉及提供示例或演示，而</st> *<st c="17196">告诉我</st>* <st c="17203">涉及提供明确的指令</st> <st c="17245">或文档</st>

    +   <st c="17261">结合这两种方法提供灵活性，并可能根据特定任务的具体上下文和复杂性提高生成式 AI 响应的准确性</st> <st c="17426">。</st>

<st c="17434">这份列表只是触及了表面，因为还有许多其他方法可以用来提高提示工程和生成式 AI 的性能。</st> <st c="17598">随着提示工程领域的持续发展，可能会出现新的创新技术，进一步增强生成式 AI 模型的能力。</st> <st c="17755">。</st>

<st c="17765">接下来，让我们谈谈有助于 RAG 应用的提示设计的基本原则。</st> <st c="17840">。</st>

# <st c="17858">提示设计的基本原则</st>

<st c="17888">在设计 RAG 应用的提示时，必须牢记以下基本要点</st> <st c="17984">：</st>

+   `<st c="18174">请分析给定上下文并回答问题，考虑到所有相关信息和细节</st>` <st c="18306">可能不如说</st> `<st c="18354">根据提供的上下文，回答以下问题：[</st>``<st c="18417">具体问题</st>`<st c="18436">。</st>

+   `<st c="18626">总结上下文的主要观点，识别提到的关键实体，然后回答给定的问题</st>`<st c="18739">，即你同时要求的多项任务。</st> <st c="18801">如果你将这个问题拆分成多个提示，并说类似以下的内容，你可能会得到更好的结果</st> <st c="18905">这样的：</st>

    +   `<st c="18913">总结以下</st>` `<st c="18957">上下文的主要观点：[上下文]</st>`

    +   `<st c="18975">识别以下总结中提到的关键实体：[来自</st>` `<st c="19052">先前提示的总结]</st>`

    +   `<st c="19068">使用上下文和识别的实体回答以下问题：[</st>``<st c="19144">具体问题]</st>`

+   `<st c="19427">根据上下文，对主题表达的情感是什么？</st>`<st c="19499">，如果你这样说，你可能会得到更好的结果</st> `<st c="19550">根据上下文，将对主题表达的情感分类为正面、负面</st>` `<st c="19653">或中性</st>`<st c="19663">。</st>

+   `<st c="19967">根据提供的上下文回答以下问题</st>`<st c="20026">，你可能会想这样说</st> `<st c="20071">以下列出的示例作为指导，根据提供的上下文回答以下问题：示例 1：[问题] [上下文] [答案] 示例 2：[问题] [上下文] [答案] 当前问题：[问题]</st>` `<st c="20280">上下文：[上下文]</st>`<st c="20298">。</st>

+   `<st c="20548">总结文章的主要观点，识别关键实体，并回答以下问题：[问题]。</st> <st c="20660">提供示例，并使用以下格式回答：[格式]。</st> <st c="20733">文章：[长篇文章文本]</st>` <st c="20764">提示可能不如以下多次迭代提示有效：</st> <st c="20850">以下：</st>

    +   `<st c="20879">总结以下文章的主要观点：[</st>``<st c="20932">文章文本]</st>`

    +   `<st c="20961">总结以下文章的主要观点并识别关键实体：[</st>``<st c="21040">文章文本]</st>`

    +   `<st c="21069">基于总结和关键实体，回答以下问题：[问题] 文章：[</st>``<st c="21160">文章文本]</st>`

+   `<st c="21319">###</st>` <st c="21322">用于区分指令和上下文部分。</st> <st c="21384">这有助于 AI 模型更好地理解和遵循给定的指令。</st> <st c="21461">例如，与以下提示相比</st> `<st c="21523">[上下文]请使用上述上下文回答以下问题：[问题]。</st> <st c="21608">以简洁的方式提供您的答案</st>` <st c="21647">，您将获得更少的成功</st> <st c="21664">。</st> `<st c="21673">指令：使用以下提供的上下文，以简洁的方式回答问题。</st> <st c="21762">上下文：[上下文]</st>` `<st c="21781">问题：[问题]</st>`<st c="21801">。</st>

<st c="21802">虽然提示设计的根本原则为创建有效的提示提供了坚实的基础，但重要的是要记住，不同的语言模型可能需要特定的调整以达到最佳效果。</st> <st c="22021">让我们接下来讨论这个</st> <st c="22040">话题。</st>

# <st c="22051">为不同的 LLM 调整提示</st>

<st c="22087">随着</st> <st c="22095">AI 领域的不断发展，人们不再仅仅依赖于</st> <st c="22155">OpenAI 来满足他们的语言建模需求。</st> <st c="22198">其他玩家，如 Anthropic 及其 Claude 模型，因其处理长上下文窗口的能力而受到欢迎。</st> <st c="22333">Google 也在发布（并将继续发布）强大的模型。</st> <st c="22406">此外，开源模型社区正在迎头赶上，Llama 等模型已被证明是</st> <st c="22504">可行的替代品。</st>

<st c="22524">然而，需要注意的是，提示（prompts）并不能无缝地从一种大型语言模型（LLM）转移到另一种。</st> <st c="22622">每种 LLM 可能都有最适合其架构的特定技巧和技术。</st> <st c="22708">例如，Claude-3 在提示时更喜欢使用 XML 编码，而 Llama3 在标记提示的不同部分（如 SYS 和 INST）时使用特定的语法。</st> <st c="22871">以下是一个使用 SYS 和 INST 标签的 Llama 模型的示例提示：</st> <st c="22932">。</st>

+   `<st c="22942"><SYS>您是一个 AI 助手，旨在为用户提供有帮助和信息的回答。</st> <st c="23034">问题。</st> <st c="23045"></SYS></st>`

+   `<st c="23051"><INST>分析以下用户的问题，并使用您的知识库提供清晰、简洁的答案。</st> <st c="23156">如果问题不清楚，请</st>` `<st c="23188">要求澄清。</st>`

+   <st c="23206">用户问题：</st> `<st c="23223">"与化石燃料相比，使用可再生能源的主要优势是什么?"</st> <st c="23307"></st>` <st c="23315"></INST></st>

<st c="23322">在这个例子中，</st> `<st c="23344">SYS</st>` <st c="23347">标签简要确立了 AI 作为辅助工具的角色，旨在提供有用的响应。</st> <st c="23439">`INST` <st c="23443">标签提供了回答用户问题的具体指令，这些指令包含在</st> `<st c="23547">INST</st>` <st c="23551">块中。</st> **<st c="23559">SYS</st>** <st c="23562">用作**<st c="23590">系统</st>** <st c="23596">或**<st c="23600">系统消息</st>**的简称，而**<st c="23622">INST</st>** <st c="23626">用于代替**<st c="23651">指令</st>**<st c="23663">。</st>

<st c="23664">在设计用于 RAG 应用的提示时，考虑与所选 LLM 相关的具体要求和最佳实践至关重要，以确保最佳性能和结果。</st> <st c="23715">所有最著名的模型都有提示文档，可以解释如果你使用它们时需要做什么。</st> <st c="23854">所有最著名的模型都有提示文档，可以解释如果你使用它们时需要做什么。</st> <st c="23961">如果你使用它们。</st>

<st c="23970">现在，让我们将本章第一部分所涵盖的所有概念通过一个</st> <st c="24075">代码实验室</st>付诸实践！

# <st c="24084">代码实验室 13.1 – 自定义提示模板</st>

<st c="24123">提示模板是一个表示在 LangChain 中管理和使用提示的机制的类。</st> <st c="24222">与大多数模板一样，提供了文本，以及代表模板输入的变量。</st> <st c="24330">使用`<st c="24340">PromptTemplate</st>` <st c="24354">包来管理你的提示，确保它在 LangChain 生态系统中运行良好。</st> <st c="24445">此代码基于我们在</st> *<st c="24491">第八章</st>*<st c="24500">的</st> *<st c="24504">8.3 代码实验室</st>*<st c="24516">中完成的代码，可以在 GitHub 仓库的</st> `<st c="24542">CHAPTER13</st>` <st c="24551">目录中找到，作为`<st c="24583">CHAPTER13-1_PROMPT_TEMPLATES.ipynb</st>`<st c="24618">。</st>

<st c="24619">作为复习，这是我们使用最多的模板：</st>

```py
 prompt = hub.pull("jclemens24/rag-prompt")
```

<st c="24722">打印这个提示看起来</st> <st c="24754">像这样：</st>

```py
 You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Question: {question}
Context: {context}
Answer:
```

<st c="24993">这存储在你可以打印出来的</st> `<st c="25016">PromptTemplate</st>` <st c="25030">对象中。</st> <st c="25062">如果你这样做，你会看到类似这样的内容：</st>

```py
 ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'jclemens24', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '1a1f3ccb9a5a92363310e3b130843dfb2540239366ebe712ddd94982acc06734'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\nQuestion: {question} \nContext: {context} \nAnswer:"))])
```

<st c="25675">所以，正如我们</st> <st c="25690">在这里可以看到的，完整的</st> `<st c="25726">PromptTemplate</st>` <st c="25740">对象不仅仅是文本和变量。</st> <st c="25782">首先，我们可以这样说，这是一个名为</st> `<st c="25839">PromptTemplate</st>` <st c="25853">对象的具体版本，称为</st> `<st c="25870">ChatPromptTemplate</st>` <st c="25888">对象，这表明它被设计成在聊天场景中最有用。</st> <st c="25969">输入变量是</st> `<st c="25993">context</st>` <st c="26000">和</st> `<st c="26005">question</st>`<st c="26013">，它们在模板字符串本身中稍后出现。</st> <st c="26066">稍后，我们将设计一个自定义模板，但这个特定的模板来自 LangChain hub。</st> <st c="26172">你可以在这里看到与 hub 相关的元数据，包括所有者、仓库和提交哈希。</st>

<st c="26266">让我们通过用我们自己的定制模板替换这个提示模板来开始我们的代码实验室。</st> <st c="26339">：</st>

<st c="26359">我们已经导入 LangChain 包用于以下目的：</st> <st c="26407">：</st>

```py
 from langchain_core.prompts import PromptTemplate
```

<st c="26466">对于这个，没有必要添加更多的导入！</st> <st c="26518">我们将替换以下代码：</st> <st c="26542">：</st>

```py
 prompt = hub.pull("jclemens24/rag-prompt")
```

<st c="26595">这里是我们将要替换的代码</st> <st c="26613">我们将用以下内容替换它：</st>

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

<st c="27076">正如你所见，我们已经定制了这个提示模板，使其专注于我们的数据（谷歌环境报告）所关注的主题。</st> <st c="27231">我们使用角色提示设计模式来建立一个我们希望 LLM 扮演的角色，我们希望这会使它更符合我们的</st> <st c="27370">特定主题。</st>

<st c="27388">提示模板接受一个字典作为输入，其中每个键代表提示模板中的一个变量，用于填充。</st> <st c="27506">`PromptTemplate` <st c="27524">对象输出的结果是</st> `<st c="27551">PromptValue</st>` <st c="27562">变量，可以直接传递给 LLM 或 ChatModel 实例，或者作为 LCEL 链中的一个步骤。</st>

<st c="27671">使用以下代码打印出提示对象：</st> <st c="27706">：</st>

```py
 print(prompt)
```

<st c="27730">输出将</st> <st c="27747">是</st> <st c="27750">以下内容：</st>

```py
 input_variables=['context', 'question'] template="\n    You are an environment expert assisting others in \n    understanding what large companies are doing to \n    improve the environment. Use the following pieces \n    of retrieved context with information about what \n    a particular company is doing to improve the \n    environment to answer the question. \n    \n    If you don't know the answer, just say that you don't know.\n    \n    Question: {question} \n    Context: {context} \n    \n    Answer:"
```

<st c="28236">我们看到它已经捕获了输入变量，而无需我们明确地</st> <st c="28317">声明它们：</st>

```py
 input_variables=['context', 'question']
```

<st c="28368">我们可以使用这条命令来打印出文本：</st> <st c="28406">：</st>

```py
 print(prompt.template)
```

<st c="28439">这会给你一个比我们之前展示的输出更易于阅读的版本，但只包含</st> <st c="28532">提示本身。</st>

<st c="28546">你将注意到，就在这个下面，我们已经有了一个针对确定我们的</st> <st c="28657">相关性分数</st>的定制提示模板：</st>

```py
 relevance_prompt_template = PromptTemplate.from_template(
    """
    Given the following question and retrieved context, determine if the context is relevant to the question. Provide a score from 1 to 5, where 1 is not at all relevant and 5 is highly relevant. Return ONLY the numeric score, without any additional text or explanation. Question: {question}
    Retrieved Context: {retrieved_context}
    Relevance Score:"""
)
```

<st c="29084">如果你运行剩余的代码，你可以看到它对 RAG 应用最终结果的影响。</st> <st c="29204">这个提示模板被认为是字符串提示模板，这意味着它是使用包含提示文本和动态内容占位符的普通字符串创建的（例如，</st> `<st c="29389">{question}</st>` <st c="29399">和</st> `<st c="29404">{retrieved_context}</st>`<st c="29423">）。</st> <st c="29427">你还可以使用</st> `<st c="29456">ChatPromptTemplate</st>` <st c="29474">对象进行格式化，该对象用于格式化消息列表。</st> <st c="29527">它由一系列</st> <st c="29552">模板本身组成。</st>

<st c="29569">提示模板</st> <st c="29587">在最大化 RAG 系统性能中扮演着关键支持角色。</st> <st c="29660">在本章剩余的代码实验室中，我们将使用提示模板作为主要元素。</st> <st c="29756">然而，我们现在将重点转移到我们在编写提示时需要记住的一系列概念上。</st> <st c="29860">我们的下一个代码实验室将专注于所有这些概念，从</st> <st c="29926">提示格式化</st>开始。

# <st c="29944">代码实验室 13.2 – 提示选项</st>

<st c="29978">此代码可在</st> <st c="30004">的</st> `<st c="30009">CHAPTER13-2_PROMPT_OPTIONS.ipynb</st>` <st c="30041">文件中找到，位于</st> `<st c="30054">CHAPTER13</st>` <st c="30063">目录下的</st> <st c="30081">GitHub 仓库。</st>

<st c="30099">通常，当你接近设计你的提示时，会有许多一般概念是你希望记住的。</st> <st c="30232">这些包括迭代、总结、转换和扩展。</st> <st c="30299">这些概念各有不同的用途，并且它们通常可以以各种方式组合。</st> <st c="30398">当你改进你的 RAG 应用时，了解如何设计你的提示的基础知识将非常有用。</st> <st c="30528">我们将通过一个真实场景来介绍不同的提示方法，在这个场景中，你正在帮助公司市场部门编写提示。</st> <st c="30692">我们将从</st> <st c="30706">迭代</st>开始。

## <st c="30721">迭代</st>

<st c="30731">这个概念简单来说就是迭代</st> <st c="30775">你的提示以获得更好的结果。</st> <st c="30811">你的第一个提示很少会是最佳和最终的提示。</st> <st c="30894">本节将重点介绍一些基本技巧和概念，以帮助您快速迭代提示，使其更适合您的</st> <st c="31043">RAG 应用。</st>

## <st c="31059">迭代语气</st>

<st c="31078">你的老板刚刚打电话。</st> <st c="31102">他们告诉你营销人员表示他们想要在他们的营销材料中使用 RAG 应用程序的输出，但必须以更营销事实表的形式提供。</st> <st c="31319">没问题；我们可以在那里直接设计提示设计！</st>

<st c="31372">在第一个提示之后</st> <st c="31411">添加第二个提示：</st>

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

<st c="31834">然后你需要将</st> `<st c="31877">rag_chain_from_docs</st>` <st c="31896">链中的提示更改为</st> `<st c="31906">prompt2</st>`<st c="31913">。查看</st> `<st c="31934">RunnablePassthrough()</st>` <st c="31955">行之后的代码：</st>

```py
 rag_chain_from_docs = (
    …
             "answer": (
                RunnablePassthrough()
                | prompt2 <st c="32032"># <- update here</st> | llm
                | str_output_parser
            )
        …
)
```

<st c="32080">然后，重新运行</st> <st c="32097">从</st> `<st c="32107">prompt2</st>` <st c="32114">向下到</st> <st c="32124">这个结果：</st>

```py
 Google is at the forefront of environmental sustainability, leveraging its technological prowess to drive impactful initiatives across various domains. Here are some of the key environmental initiatives that Google is promoting: <st c="32366">Empowering Individuals</st>: Google aims to help individuals make more sustainable choices through its products. In 2022, Google reached its goal of assisting 1 billion people in making eco-friendly decisions. This was achieved through features like eco-friendly routing in Google Maps, energy-efficient settings in Google Nest thermostats, and carbon emissions information in Google Flights. By 2030, Google aspires to help reduce 1 gigaton of carbon equivalent emissions annually. <st c="32845">…[TRUNCATED FOR BREVITY]…</st> By organizing information about the planet and making it actionable through technology, Google is helping to create a more sustainable future. The company's efforts span from individual empowerment to global partnerships, all aimed at reducing environmental impact and fostering a healthier planet.
```

<st c="33169">如果你阅读</st> <st c="33181">完整的输出，你会注意到这确实更偏向于营销导向。</st> <st c="33272">这可能正是营销团队所寻找的。</st> <st c="33324">然而，你刚刚记得你的老板也说过这将被放在网站上只能容纳 50 个单词的小方块里</st> <st c="33465">最多！</st>

## <st c="33473">缩短长度</st>

<st c="33492">对于</st> `<st c="33497">prompt3</st>`<st c="33504">，我们只需要添加</st> <st c="33526">这个小的片段：</st> `<st c="33547">最多使用 50 个单词</st>`<st c="33567">。看起来是这样的：</st>

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

<st c="34022">更新链中的提示为</st> `<st c="34057">prompt3</st>`<st c="34064">。运行剩余的代码，你将得到</st> <st c="34108">以下输出：</st>

```py
 Google's environmental initiatives include promoting electric vehicles, sustainable agriculture, net-zero carbon operations, water stewardship, and a circular economy. They aim to help individuals and partners reduce carbon emissions, optimize resource use, and support climate action through technology and data-driven solutions.
```

<st c="34451">营销团队</st> <st c="34470">喜欢你的工作！</st> <st c="34488">一切都很顺利！</st> <st c="34506">做得好！</st> <st c="34516">时间</st> <st c="34521">过得真快…</st>

<st c="34529">一个月后，决定与其关注你所有客户的环保努力，不如专注于</st> <st c="34661">与技术相关的努力。</st>

## <st c="34688">改变焦点</st>

<st c="34707">我们希望 LLM 关注更具体于技术的方面。</st> <st c="34744">这是为了小册子，所以它可以再次变长。</st> <st c="34833">让我们设计</st> `<st c="34846">prompt4</st>` <st c="34853">来指导 LLM 改变其焦点，我们将取消</st> <st c="34917">长度限制：</st>

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

<st c="35499">再次提醒</st> <st c="35515">更新链中的提示为</st> `<st c="35553">prompt4</st>` <st c="35560">，然后运行剩余的代码以获取</st> <st c="35602">以下输出：</st>

```py
 Google is at the forefront of leveraging technology to drive environmental sustainability. Here are some of the key initiatives that highlight their commitment to using technology for a greener future: <st c="35816">Eco-Friendly Product Features</st>: <st c="35848">Google Maps</st>: Introduced eco-friendly routing, which has helped prevent over 1.2 million metric tons of carbon emissions since its launch. <st c="35987">Google Nest</st>: Energy efficiency features in Google Nest thermostats help users reduce their energy consumption. <st c="36099">Google Flights</st>: Provides carbon emissions information to help travelers make more sustainable choices. <st c="36203">…TRUNCATED FOR BREVITY…</st>
<st c="36226">Sustainability-Focused Accelerators</st>: <st c="36265">Google for Startups Accelerator</st>: Supports early-stage innovations aimed at tackling sustainability challenges, fostering the growth of technologies that can positively impact the planet. Google's comprehensive approach to environmental sustainability leverages their technological expertise to create significant positive impacts. By integrating sustainability features into their products, optimizing their operations, and collaborating with partners, Google is driving forward a more sustainable future.
```

<st c="36771">同样，我们在这里的书中不得不缩短它，但如果你在代码中查看，结果令人印象深刻。</st> <st c="36883">显然，对他们的环境方面的技术方面有更高的关注。</st> <st c="36976">你的营销团队</st> <st c="36996">印象深刻！</st>

<st c="37009">这是一个有趣的例子，但这并不远离构建这些类型系统时发生的事情。</st> <st c="37114">在现实世界中，你可能会迭代更多次，但采用迭代方法来设计提示将帮助你达到更优的 RAG 应用，就像你的 RAG 系统的任何其他部分一样。</st>

<st c="37340">接下来，让我们</st> <st c="37353">谈谈如何将大量数据压缩成更小的数据量，这通常被称为摘要。</st>

## <st c="37461">摘要</st>

<st c="37473">摘要</st> <st c="37487">是 RAG（检索增强生成）的一个流行用途。</st> <st c="37513">将公司内部的大量数据消化成更小、更简洁的信息，可以是一种快速简单的方法来提高生产力。</st> <st c="37681">这对于依赖信息或需要跟上快速变化信息的职位尤其有用。</st> <st c="37789">我们已经看到了如何设计一个提示来使用字数限制，这在</st> `<st c="37865">prompt3</st>`<st c="37872">中。</st> 然而，在这种情况下，我们将更多地关注 LLM 的摘要内容，而不是试图成为一个专家或撰写营销文章。</st> <st c="38023">代码如下：</st> <st c="38035">如下：</st>

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

<st c="38395">更新链中的</st> `<st c="38406">prompt5</st>` <st c="38413">，然后再次运行剩余的代码。</st> <st c="38469">结果如下：</st>

```py
 Google's environmental initiatives include achieving net-zero carbon, promoting water stewardship, supporting a circular economy, and leveraging technology to help partners reduce emissions.
```

<st c="38687">好的，这很好，简短，并且是摘要。</st> <st c="38731">只有事实！</st>

<st c="38753">下一个例子是</st> <st c="38773">另一种我们可以关注 LLM 的情况，但增加了摘要的努力。</st>

## <st c="38861">专注于摘要的摘要</st>

<st c="38886">对于</st> `<st c="38891">prompt6</st>`<st c="38898">，我们将</st> <st c="38915">保留之前提示中的大部分内容。</st> <st c="38969">然而，我们将尝试将 LLM 特别集中在他们产品的环保方面：</st>

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

<st c="39496">将链中的提示更新为</st> `<st c="39531">prompt6</st>`<st c="39538">，然后运行代码以获得</st> <st c="39569">以下输出：</st>

```py
 Google's environmental initiatives include eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions information in Google Flights.
```

<st c="39755">这是简短的，如果您将其与更冗长的描述进行比较，它似乎确实专注于 PDF 中展示的产品。</st> <st c="39912">这是一个相当好的结果，但通常当您请求摘要时，即使您将 LLM 聚焦于特定方面，LLM 仍然可能包含您不希望包含的信息。</st> <st c="40101">为了应对这种情况，我们</st> <st c="40119">转向</st> *<st c="40132">extract</st>* <st c="40139">方法。</st>

## <st c="40147">extract instead of summarize</st>

<st c="40176">如果您遇到常见的总结包含过多不必要信息的问题，尝试使用单词</st> *<st c="40289">extract</st>* <st c="40296">而不是</st> *<st c="40309">summarize</st>*<st c="40318">。这看起来可能只是细微的差别，但它对 LLM 来说可能有很大的影响。</st> *<st c="40407">Extract</st>* <st c="40414">给人一种您正在提取特定信息的印象，而不是仅仅试图捕捉整个文本中的整体数据。</st> <st c="40555">LLM 不会错过这个细微差别，这可以是一个很好的技巧，帮助您避免总结有时带来的挑战。</st> <st c="40692">我们将考虑到这个变化来设计</st> `<st c="40707">prompt7</st>` <st c="40714">：</st>

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

<st c="41183">将链中的提示更新为</st> `<st c="41218">prompt7</st>`<st c="41225">，然后运行代码以获取</st> <st c="41256">以下输出：</st>

```py
 Google's environmental initiatives include eco-friendly routing in Google Maps, energy efficiency features in Google Nest thermostats, and carbon emissions information in Google Flights to help users make sustainable choices.
```

<st c="41494">这与</st> `<st c="41545">prompt6</st>`<st c="41552">的响应略有不同，但我们已经得到了一个很好的聚焦结果。</st> <st c="41596">当您的总结响应包含不必要的数据时，尝试这个技巧来帮助提高</st> <st c="41682">您的结果。</st>

<st c="41695">迭代和总结并不是提高提示努力所需要理解的唯一概念。</st> <st c="41709">我们将接下来讨论如何利用您的 RAG 应用从您的</st> <st c="41899">现有数据中推断信息。</st>

## <st c="41913">推理

<st c="41923">在推理的根源，您是</st> <st c="41957">要求模型查看您的数据并提供某种额外的分析。</st> <st c="42042">这通常涉及提取标签、名称和主题，甚至确定文本的情感。</st> <st c="42147">这些能力对 RAG 应用具有深远的影响，因为它们使那些不久前被认为仅属于人类读者领域的工作成为可能。</st> <st c="42327">让我们从一个简单的布尔式情感分析开始，其中我们考虑文本是积极的</st> <st c="42434">还是消极的：</st>

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

<st c="43026">在这段代码中，我们基于前一个提示的总结，但增加了一个</st> *<st c="43102">分析</st>* <st c="43110">LLM 正在消化的数据的情感</st> <st c="43175">在这种情况下，它确定情感为</st> <st c="43220">积极</st> <st c="43222">：</st>

```py
 Google is enhancing eco-friendliness through features like eco-friendly routing in Maps, energy-efficient Nest thermostats, and carbon emissions data in Flights, aiming to reduce emissions significantly. Sentiment: positive
```

<st c="43456">在类似的分析领域中，另一个常见的应用是从</st> <st c="43552">上下文中</st> 提取特定数据。

## <st c="43564">提取关键数据</st>

<st c="43584">作为一个参考点，你现在被要求识别客户在其与环境努力相关的文档中提到的任何特定产品。</st> <st c="43628">在这种情况下，谷歌（客户）有许多产品，但在这个文档中只提到了其中的一小部分。</st> <st c="43750">你将如何快速提取这些产品并识别它们？</st> <st c="43926">让我们用</st> <st c="43946">我们的提示</st> 来试试：

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

<st c="44663">在这段代码中，我们继续基于之前的提示进行构建，但不是要求情感分析，而是要求与我们所检索到的文本相关的产品列表。</st> <st c="44842">我们使用的 GPT-4o-mini 模型成功地遵循了这些指示，列出了文本中特别命名的每个</st> <st c="44937">产品：</st>

```py
 Google is enhancing eco-friendliness through products like eco-friendly routing in Google Maps, energy efficiency features in Google Nest thermostats, and carbon emissions information in Google Flights. Related products: Google Maps, Google Nest thermostats, Google Flights
```

<st c="45258">再次，LLM 能够处理我们所要求的一切。</st> <st c="45322">然而，有时我们只想对主题有一个整体的感觉。</st> <st c="45392">我们将使用 LLM 来讨论推理的概念。</st>

## <st c="45452">推断主题</st>

<st c="45469">你可能认为这是一个极端的</st> <st c="45508">总结案例。</st> <st c="45531">在这个例子中，我们正在将数千个单词总结成一组简短的主题。</st> <st c="45632">这能行吗？</st> <st c="45650">让我们试试！</st> <st c="45661">我们将从</st> <st c="45684">以下代码</st> 开始：

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

<st c="46423">在这里，我们使用与之前提示类似的方法，但不是要求产品列表，而是要求与我们所检索到的文本相关的至少八个主题列表。</st> <st c="46610">再次，我们使用的 GPT-4o mini 模型成功地遵循了这些指示，列出了八个</st> <st c="46711">与文本特别相关的、高度相关的主题：</st>

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

<st c="47145">我们已经涵盖了迭代、总结和推理，这些都显示出极大的潜力来提高你的提示效果。</st> <st c="47270">我们将要介绍的概念还有</st> <st c="47304">转换</st>。

## <st c="47322">转换</st>

<st c="47337">转换</st> <st c="47352">是将您当前的数据转换</st> <st c="47403">成不同的状态或格式。</st> <st c="47441">一个非常常见的例子是语言翻译，但还有许多其他情况，包括将数据放入某种编码格式，如 JSON 或 HTML。</st> <st c="47589">您还可以应用转换，如检查拼写或</st> <st c="47657">语法错误。</st>

<st c="47672">我们将从</st> <st c="47692">语言翻译</st>开始。</st>

### <st c="47713">语言转换（翻译）</st>

<st c="47751">营销部门又</st> <st c="47768">打电话来了。</st> <st c="47776">你到目前为止所做的工作非常出色，但现在事情正在加速，我们将走向国际市场了！</st> <st c="47885">我们选择的第一批国际市场包括西班牙语和法语的使用者。</st> <st c="47972">我们公司的一位新投资者也是任何与海盗有关的东西的大粉丝，所以是的，我们也将涵盖这种方言！</st> <st c="48110">既然我们在谈论</st> *<st c="48137">转换</st>*<st c="48152">，我们称之为</st> <st c="48167">语言转换，但在这个上下文中，看到术语</st> *<st c="48235">翻译</st>* <st c="48246">也很常见。</st> <st c="48269">让我们</st> <st c="48275">开始吧：</st>

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

<st c="48953">在这段代码中，我们基于之前的提示，但随后要求 LLM 生成四种不同版本的简短摘要：英语、西班牙语、法语和英语海盗语。</st> <st c="49123">显然，我们应该更经常地说海盗话，因为这是</st> <st c="49186">最</st> <st c="49191">有趣的：</st>

```py
 English: Google enhances eco-friendliness through features like eco-friendly routing in Maps, energy-efficient Nest thermostats, and carbon emissions info in Flights, helping reduce carbon emissions significantly. Spanish: Google mejora la eco-amigabilidad con funciones como rutas ecológicas en Maps, termostatos Nest eficientes en energía e información de emisiones de carbono en Flights, ayudando a reducir significativamente las emisiones de carbono. French: Google améliore l'éco-responsabilité avec des fonctionnalités telles que les itinéraires écologiques dans Maps, les thermostats Nest économes en énergie et les informations sur les émissions de carbone dans Flights, aidant à réduire significativement les émissions de carbone. English pirate: Google be makin' things greener with eco-routes in Maps, energy-savin' Nest thermostats, and carbon info in Flights, helpin' to cut down on carbon emissions mightily.
```

<st c="50132">语言翻译是 RAG 的一个流行用途，但还有其他转换可能很有用。</st> <st c="50237">让我们回顾一下向摘要添加不同语气的例子。</st>

### <st c="50301">语气转换</st>

<st c="50321">我们到目前为止的努力</st> <st c="50341">在撰写摘要甚至</st> <st c="50391">营销文案方面都取得了成功，但现在我们需要扩展到其他渠道，如电子邮件，并给我们的摘要一个更友好的语气以适应这种格式。</st> <st c="50534">为此，我们将应用</st> <st c="50558">语气转换：</st>

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

<st c="51132">在这里，我们继续原来的</st> <st c="51164">总结，但随后我们要求 LLM 使用相同的信息写一封</st> <st c="51260">轻松</st> <st c="51266">语气</st>的电子邮件：</st>

```py
 Google is enhancing eco-friendliness through features like eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions data in Google Flights. <st c="51456">Email Format:</st> Subject: Exciting Eco-Friendly Features from Google! Hi [Recipient's Name],
I hope you're doing well! I wanted to share some cool updates from Google about their efforts to help the environment. They've introduced some awesome features like eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and even carbon emissions data in Google Flights. It's great to see such big steps towards a greener future! Best, [Your Name]
```

<st c="51916">正如我们通过这些例子所看到的，LLM 可以以多种方式用于改进 RAG 应用。</st> <st c="52029">在我们的例子中不太适用（但在其他场景中仍然非常有价值）的其他选项包括翻译成特定的编码格式，或者从一个编码格式转换到另一个编码格式。</st> <st c="52217">拼写和语法检查也是可以应用的热门转换。</st> <st c="52277">这些。</st> <st c="52283">可以。</st> <st c="52287">应用。</st>

<st c="52298">我们已经涵盖了迭代、总结、推理和转换。</st> <st c="52372">为了结束这个代码实验，我们还将介绍一个概念。</st> <st c="52434">扩展。</st>

## <st c="52449">扩展</st>

<st c="52459">从许多方面来说，扩展的目标可以被视为我们之前介绍的概念（如总结）目标的一种反向。</st> <st c="52485">在总结中，我们正在将大量数据合并成更小的数据量，同时试图保留数据的含义。</st> <st c="52597">扩展试图做的是相反的，它将一小部分数据扩展到更大的信息集。</st> <st c="52746">让我们通过一个可以实施这种扩展的场景来了解一下：简短总结的扩展。</st> <st c="52816">扩展</st> <st c="52825">它</st> <st c="52861">让我们通过一个可以实施这种扩展的场景来了解一下：简短总结的扩展。</st> <st c="52941">简短总结的扩展。</st>

### <st c="52955">扩展简短文本</st>

<st c="52978">我们的努力持续增长！</st> <st c="53009">我们最新的任务是向我们的客户推广我们一直在讨论的环境概念的好处。</st> <st c="53150">在接下来的提示中，我们将根据我们得到的简短总结，暂时假设这就是我们拥有的全部内容。</st> <st c="53298">然后我们将要求 LLM 在此基础上进行扩展，重点关注它如何吸引投资者。</st> <st c="53403">让我们看看它能提出什么：</st> <st c="53430">：</st>

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

<st c="54079">在现实世界的场景中，当你希望 LLM 扩展你的数据时，你很可能大部分情况下都没有原始的上下文。</st> <st c="54109">因此，通过告诉 LLM 将其扩展的来源仅限于总结中提供的内容，我们更准确地模拟了这种场景。</st> <st c="54217">我们的 LLM 没有让我们失望，因为它提供了对原始总结中提到的环境努力的扩展描述，并针对投资者进行了调整。</st> <st c="54375">然后，它针对投资者进行了调整：</st>

```py
<st c="54544">Summary:</st> Google offers eco-friendly routing in Google Maps, energy-efficient Google Nest thermostats, and carbon emissions information in Google Flights to help users make sustainable choices. <st c="54738">Broader Description:</st> Google is actively enhancing the eco-friendliness of its products by integrating features that promote sustainability. For instance, Google Maps now includes eco-friendly routing options, Google Nest thermostats are designed for energy efficiency, and Google Flights provides carbon emissions information. These initiatives not only help users make more environmentally conscious decisions but also demonstrate Google's commitment to reducing its carbon footprint. For investors, this focus on sustainability can be a significant value proposition, as it aligns with the growing consumer demand for eco-friendly products and can lead to long-term cost savings and regulatory advantages. Additionally, it positions Google as a leader in environmental responsibility, potentially enhancing its brand reputation and market share.
```

<st c="55585">这只是扩展概念可以应用的一个例子。</st> <st c="55614">考虑一下在你的 RAG 应用中如何以及何时利用数据扩展。</st> <st c="55657">以及。</st>

<st c="55753">这总结了如何改进你的提示设计的关键概念：迭代、总结、推断、转换和扩展。</st> <st c="55897">这些概念构成了许多更深入和复杂概念的基础，可以使你的 RAG 应用更加有效。</st> <st c="56033">把这视为你在该领域的知识起点，并继续跟踪该领域的进步和新技术，随着它们的被发现。</st>

# <st c="56159">摘要</st>

<st c="56167">在本章中，我们探讨了提示工程在增强 RAG 系统性能和有效性中的关键作用。</st> <st c="56299">通过战略性地设计和精炼输入提示，我们可以提高相关信息的检索，从而提高生成文本的质量。</st> <st c="56463">我们讨论了各种提示设计技术，例如射击设计、思维链提示、角色扮演和知识增强，这些技术可以应用于优化</st> <st c="56630">RAG 应用。</st>

<st c="56647">在本章中，我们讨论了提示设计的根本概念，包括简洁、具体和明确的重要性，以及逐步迭代和使用清晰分隔符的必要性。</st> <st c="56866">我们还强调了不同 LLM 需要不同的提示，以及适应特定模型使用的提示的重要性。</st>

<st c="57021">通过一系列代码实验室，我们学习了如何使用 LangChain 中的`<st c="57112">PromptTemplate</st>` <st c="57126">类创建自定义提示模板，以及如何将各种提示概念应用于提高我们的 RAG 工作。</st> <st c="57226">这些概念包括迭代以精炼提示、总结以浓缩信息、推断以提取额外见解、将数据转换为不同的格式或语气，以及扩展简短摘要以生成更全面的描述。</st> <st c="57487">我们还探讨了使用提示参数，如温度、top-p 和种子，来控制 LLM 输出的随机性和确定性。</st>

<st c="57629">通过利用本章介绍的技术和概念，我们可以显著提高 RAG 应用的表现，使它们在检索相关信息、生成高质量文本和适应特定用例方面更加有效。</st> <st c="57896">随着提示工程领域的不断发展，跟上最新的技术和最佳实践对于在各个领域最大化 RAG 系统的潜力至关重要。</st>

<st c="58098">在我们下一章和最后一章中，我们将讨论一些更高级的技术，你可以使用这些技术对你的 RAG 应用进行潜在的显著改进！</st>
