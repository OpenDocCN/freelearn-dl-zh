

# 第八章：使用 LLMs 编码的限制

本章将帮助您了解**大型语言模型**（**LLMs**）如何缺乏对人类语言细微差别的完美理解，并且在面对复杂编码任务时超出了它们的能力范围。我们将探讨 LLM 聊天机器人中的不一致性和不可预测性。本章还将帮助您将 LLM 生成的代码整合到您的代码库中。希望通过对该领域改进状态的研究，您能获得启发，迈向构建更复杂代码的方向。

在本章中，您将有机会学习以下内容：

+   LLMs 的固有限制

+   将 LLMs 整合到编码工作流中的挑战

+   未来的研究方向以应对这些限制

# 技术要求

本章中，您可能需要具备以下内容：

+   访问 LLM/chatbot，如 GPT-4 或 Gemini；每种都需要登录。对于 GPT-4，您需要一个 OpenAI 帐户，对于 Gemini，您需要一个 Google 帐户。

+   一款互联网浏览器，用于进一步阅读以深入了解。

现在，我们将进入本章的第一部分，讨论所有 LLMs 的固有限制。

# LLMs 的固有限制

大型语言模型（LLMs）在生成代码方面展现出了显著的能力，但它们也具有固有的局限性，这些局限性可能会显著影响输出的质量和可靠性。

## 核心限制

以下是 LLMs 的一些限制：

+   **缺乏真正的理解**：虽然 LLMs 可以生成语法正确的代码，但它们缺乏对底层概念、算法和问题领域的深入理解。这可能导致次优或错误的解决方案。

+   **幻觉**：LLMs 可能会生成看似合理但实际上是错误或荒谬的代码，通常称为“幻觉”。这在关键应用中尤为成问题。

+   **对训练数据的依赖**：LLM 生成的代码质量在很大程度上依赖于训练数据的质量和多样性。训练数据中的偏见或限制可能会反映到生成的代码中。

+   **处理复杂逻辑的困难**：LLMs 在处理需要复杂逻辑推理或问题解决的任务时常常遇到困难，导致生成次优或错误的代码。

+   **缺乏上下文理解**：尽管 LLMs 可以顺序处理信息，但它们通常缺乏对更广泛上下文的全面理解，这可能导致生成的代码出现不一致或错误。

+   **有限的上下文窗口或内存**：在 LLMs 的上下文窗口中，一次提示（查询）中可以输入的信息量或响应中可以传递的信息量是有限的。这些上下文窗口正在迅速增大，但它们目前有较大的硬件要求。

+   **有限的调试能力**：LLMs 通常不擅长调试自己生成的代码，因此需要人工干预以识别和纠正错误。

+   **旧的训练数据**：LLM 无法更新其训练数据，因此其回答可能基于过时的信息。

代码生成中也存在一些特定的局限性，具体如下：

+   **代码质量和效率**：LLM 生成的代码往往在性能和资源利用上不够高效或最优。

+   **安全漏洞**：LLM 由于缺乏安全专长，可能生成包含安全漏洞的代码。

+   **可维护性**：LLM 生成的代码可能由于其潜在的复杂性和不符合编码标准，难以维护。

+   **可复现性**：多次生成相同的代码输出可能具有挑战性，因为 LLM 是随机系统。

[Prompt_Drive]

## LLM 的其他局限性

除了前述内容，LLM 可能还会存在伦理和法律限制。LLM 可能会无意中生成有偏见的代码，或者从训练数据中逐字复制现有的代码片段，从而引发**知识产权**（**IP**）或不当伦理问题。[Parth Santpurkar, 该书的技术审阅人]

关于伦理和偏见的内容请参见*第五章*；*第六章*讨论了法律考虑，*第七章*则尝试通过对策应对大多数安全威胁。

## 评估 LLM 性能

LLM 的输出非常难以评估，但有许多方法可以进行评估。一些方法基于神经网络，一些则是统计分析方法。

可以使用哪些指标来评估 LLM，它们是如何计算的？以下是一些指标的简要介绍：

+   生成的代码在语法和语义上是否与标准答案一致？LLM 生成的代码是否解决了所要求的问题？

+   生成的代码在功能和逻辑上与预期解决方案有多相似？

+   LLM 会生成错误的代码吗？

+   **上下文相关性**：对于**检索增强生成**（**RAG**）模型，LLM 是否能够从提供的上下文中提取并使用最相关的信息？

+   **总结能力**：LLM 是否能够提供简洁且正确的代码片段或文档，并且这些内容是基于源材料的？

+   **CodeBLEU（双语评估替代）**：通过每个匹配的*n*元组（*n*个连续的单词）的精确度来比较输出与标准答案代码。BLEU 本身在代码评估中并不那么有效，因此 CodeBLEU 被用于代码合成。[CodeBLEU]

+   **METEOR**：一种结合了单词匹配、词干提取和同义词处理的指标，用以捕捉语义相似性，这对代码评估可能有帮助。

[confident_ai, Stevens, CodeBLEU]

了解更多关于指标的信息，请访问：[`www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation`](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)。

你还可以看看 Jane Huang 关于 LLM 度量标准和最佳实践的说法：[`medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5`](https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5)。

了解 Ren 等人提出的 CodeBLEU，点击这里：[`arxiv.org/abs/2009.10297`](https://arxiv.org/abs/2009.10297)。

## 克服固有的限制

让我们看看已经可以做些什么来提高 LLM 的工作和结果。

### 搜索网络

大型语言模型（LLMs）是在一定数据集上训练的，然后进行测试和部署。因此，它们的数据总是过时的，随着世界的变化总是需要重新训练。然而，Gemini 会搜索互联网，而 Serper 是一个低成本的 API，帮助像 GPT-4 这样的 LLMs，通过搜索最新的信息来更新它们。可以在这里获取 Serper：[`serper.dev/`](https://serper.dev/)。注册 API 密钥并开始在你的代理中使用它非常快速且容易。Serper 提供 2,500 次免费查询。

### AI 代理

**智能代理**是在环境中操作的实体，具有状态、传感器（具有感知能力），并能够自主执行动作以实现目标。温控器、人类和状态都是智能代理的例子 [Wiki_Agent]。

从 LLMs 创建 AI 代理可以帮助减少一些 LLMs 的固有问题。

当你输入一个提示词来从 GPT-4o 获取一些代码，运行它时发现有错误，然后再向 GPT-4o 输入另一个提示词来修正这个错误时，你的行为就像一个代理，并将反馈出需要改进的弱点。

这可以被自动化：当应用程序表现得像代理时，它们被称为*代理性*。

AI 代理会自己完成整个代码运行、反馈、重新查询过程以及迭代。

通过这种方式，LLM 代理可以自行处理并消除错误。如果是人工检查，这个过程可能更精确，但迭代速度会更慢。

Devin 是一个虚拟的软件工程师代理，可以通过提示词生成代码。不幸的是，截止到 2024 年 7 月，Cognition 尚未向公众发布 Devin；你需要加入等待名单：[`www.cognition.ai/blog/introducing-devin`](https://www.cognition.ai/blog/introducing-devin)。

如果你有一个代理性应用程序，使用几个不同的 LLM 作为代理，那么一个 LLM 的盲点和弱点可以通过其他 LLMs 得到改善。

代理，特别是多代理应用程序，可以大大提升 LLM 的性能。即使是由较弱的 LLMs 驱动的代理，也能表现得比世界上最新最强的 LLM 更好（在任何时刻）。这里有一项关于多代理系统及其如何提升 LLM 性能的研究：[`arxiv.org/html/2402.05120v1`](https://arxiv.org/html/2402.05120v1)。LLMs 也可以进行辩论和投票。

即使是人类推理，在团队中也远远优于单个人独自推理[Mercier]；这就是集体智慧。辩论和被迫提供充分的论据和证据大大提高了揭示真相或避免错误的机会。人类群体和 AI 代理群体共享这些益处：

+   错误修正

+   经验或数据的多样性

+   知识共享：成员可以互相学习

+   更多的计算能力：更快的速度

因此，团队合作可以非常有益；这可能也是人类组成最多可达数十亿人的群体的原因！是否会有由数十亿个 AI 代理组成的团队呢？

代理可以组成小组来编写代码，其中之一被称为**ChatDev**。ChatDev 是一个虚拟软件公司，拥有 CEO、开发者、架构师和项目经理。这些都是一起工作的代理，旨在根据提示帮助用户制作所请求的软件。了解更多关于 ChatDev 的信息请点击这里：[`chatdev.toscl.com/`](https://chatdev.toscl.com/)以及这里：`medium.com/@meirgotroot/chatdev-review-the-good-the-bad-and-the-ugly-469b5cb691d4`。

微软已经开发出一个用于自动化复杂 LLM 工作流的多代理应用程序；它被称为**AutoGen**。

在 AutoGen 和 ChatDev 中，代理彼此之间进行沟通并协作，生成更好的解决方案。查看 ChatDev 和 AutoGen 的对比，请点击这里：[`www.ikangai.com/autogen-vs-chatdev-pioneering-multi-agent-systems-in-the-llm-landscape/`](https://www.ikangai.com/autogen-vs-chatdev-pioneering-multi-agent-systems-in-the-llm-landscape/)。

AutoGPT 是一个开源 AI 代理，你可以向其查询。它会将你给定的目标分解为子任务，并使用互联网等工具进行自动循环（[`en.wikipedia.org/wiki/Auto-GPT`](https://en.wikipedia.org/wiki/Auto-GPT)）。

使用 MetaGPT 创建你自己的 AI 代理：[`play.google.com/store/apps/details?id=com.metagpt.app&hl=en_US&pli=1`](https://play.google.com/store/apps/details?id=com.metagpt.app&hl=en_US&pli=1)。

AI 代理是目前的一个关键研究领域，并且该领域展现出很大的潜力。关于这个话题可以写很多内容，但我们需要继续讨论本章的主题。

我们已经讨论了固有的局限性，接下来我们将学习将代码插入编码工作流中的困难。

# 将 LLM 集成到编码工作流中的挑战

首先，我们应该看看 LLM 生成的代码和工作流中存在哪些挑战。以下是一些挑战：

+   代码质量和可靠性、安全风险以及依赖管理已经被提到。

+   **可解释性**：理解 LLM 生成代码背后的逻辑可能很困难，这使得调试和维护变得具有挑战性

+   **上下文理解**：LLM 能够理解小范围的上下文，但无法理解整个代码库或项目，因此它的使用可能导致与其他代码不兼容，或无法生成与其他代码风格一致的代码。

+   **代码片段长度**：大型语言模型（LLMs）可能在理解和处理长代码片段时遇到困难，导致生成不完整或不准确的响应。

+   **专门领域**：在通用数据集上训练的 LLM 可能缺乏某些编码任务所需的深度领域知识，例如医学影像或金融建模。

+   **修复复杂错误**：虽然 LLMs 擅长发现代码中的错误，但通常不能检测到所有错误，也可能无法识别和修复微妙或复杂的错误。

+   **性能考虑**：LLM 可能不会优先考虑代码效率或优化，可能生成在执行速度或资源使用方面不理想的代码。

+   **算法选择**：选择最适合给定任务的算法和数据结构对 LLMs 来说可能是一个挑战，因为它们可能没有深入理解算法复杂度或权衡。

## 相关工作流示例

一个常见的 LLM 生成代码的工作流是**自动化代码生成用于软件开发**。该工作流包括使用 LLM 根据用户需求或代码注释生成初始代码片段，之后进行人工审查、测试，并将其集成到主代码库中。

这里是有关 LLM 生成代码集成过程的更多细节：

1.  **代码审查和优化**：生成的代码会经过人类专家的严格审查，以确保其符合编码标准、最佳实践和项目需求。这可能包括重构、调试和优化。

1.  **单元测试**：集成后的代码会经过严格的单元测试，以验证其功能性和正确性。这有助于在开发过程中及早识别和解决潜在问题。

1.  **集成测试**：集成后的代码与系统中的其他组件一起进行测试，以确保无缝集成和兼容性。

1.  **版本控制**：集成后的代码通过版本控制系统（如 Git 或 SVN）进行适当管理，以跟踪更改、促进协作，并在必要时支持回滚。

1.  **持续集成和持续交付（CI/CD）**：集成后的代码会被纳入 CI/CD 管道中，以自动化测试、部署和监控。

1.  **监控与维护**：集成后的代码的性能和行为会被密切监控，以识别并解决可能出现的任何问题。

[Liu_2024]

## 安全风险

前述过程没有提到安全性作为一个阶段，因为安全性必须贯穿整个过程。我们将简要提到风险和保护系统的措施，但*第七章*中有更多关于安全性的内容。

存在各种安全风险；以下是 LLM 生成的代码可能带来的安全风险。

### 风险

尽管 LLM 生成的代码提供了潜在的好处，但它也带来了若干风险：

+   **代码质量与** **可靠性风险**：

    +   **不正确或低效的代码**：LLM 可能生成功能不正确、低效或次优的代码

    +   **安全漏洞**：如果没有经过仔细审查，生成的代码可能引入安全漏洞

    +   **不遵守编码标准**：代码可能不符合既定的编码标准，从而导致可维护性问题

+   **操作风险**：

    +   **依赖问题**：生成的代码可能引入与现有环境不兼容的依赖项

    +   **集成挑战**：将 LLM 生成的代码集成到现有系统中可能既复杂又容易出错

### 安全措施

下面是一些针对前述风险保护系统的措施：

+   **集成前安全**：

    +   **LLM 模型安全**：确保使用的 LLM 模型是安全的，并且不会暴露敏感数据。考虑使用具备强大安全措施的模型。

    +   **数据隐私**：保护用于训练 LLM 并生成代码的敏感数据。实施数据匿名化和加密技术。

    +   **代码漏洞扫描**：在集成前对生成的代码进行全面的漏洞扫描，以识别潜在的安全风险。

+   **集成与** **后集成安全**：

    +   **代码审查与安全审计**：聘请安全专家审查集成代码，检查漏洞并确保符合安全标准

    +   **安全编码实践**：在整个集成过程中遵循严格的安全编码实践

    +   **安全测试**：进行全面的安全测试，包括渗透测试，以识别和解决弱点

    +   **监控与威胁检测**：实施*持续*监控和威胁检测机制，以识别并响应潜在的安全事件

+   **具体的** **安全措施**：

    +   **输入验证与清理**：验证并清理所有输入，以防止注入攻击和其他漏洞

    +   **访问控制**：实施严格的访问控制，以保护 LLM 和集成代码免受未经授权的访问

    +   **加密**：对敏感数据进行加密，无论是在静态存储还是在传输过程中

    +   **事件响应计划**：制定全面的事件响应计划，以有效应对安全漏洞

*第七章*包含更多关于安全的信息，因此我们在此不再重复。*第六章*深入探讨了 LLM 代码的法律方面，因此我们在此也不再重复。

## 知识产权问题

关于 LLM 生成代码的知识产权问题复杂且不断发展。以下是一些潜在的问题：

+   **版权问题**：

    +   **版权侵权** ：如果 LLM 在受版权保护的代码上进行了训练，那么生成的代码可能会侵犯这些版权。由于训练数据的性质以及可能的无意复制，这一问题尤为复杂。

    +   **生成代码的所有权** ：谁拥有生成代码的版权？是 LLM 提供者、用户，还是共享所有权模式？这是一个法律先例有限的领域。

+   **专利问题** ：

    +   **专利侵权** ：如果生成的代码实施了某项专利发明，它可能构成专利侵权。

    +   **专利资格** ：生成的代码是否能够获得专利是一个复杂的法律问题，因为它涉及到判断该代码是否代表了一项创造性步骤。

+   **商业秘密问题** ：

    **商业秘密披露** ：如果 LLM 是基于专有代码或数据进行训练的，那么通过生成的代码有可能无意中披露商业秘密。

+   **其他问题** ：

    +   **合理使用** ：在某些情况下，合理使用原则可能适用，但其在 LLM 生成代码中的应用仍然不明确。

    +   **许可** ：了解 LLM 及其底层数据的许可条款对避免知识产权问题至关重要。

需要注意的是，该领域的法律环境正在迅速变化，建议咨询法律专家，以评估具体风险并制定相应的策略以降低这些风险。

再次查看*第六章*，了解更多法律问题。

### 了解更多有关 LLM 生成代码的知识产权问题

查看这些著名的法律网站，了解 LLM 生成代码（即 AI 生成代码）的最新发展：

+   **哈佛法学评论** ：[`harvardlawreview.org/`](https://harvardlawreview.org/)

+   **斯坦福法学评论** ：[`www.stanfordlawreview.org/`](https://www.stanfordlawreview.org/)

+   **哥伦比亚法学评论** ：[`columbialawreview.org/`](https://columbialawreview.org/)

+   **美国知识产权法协会 (AIPLA)** ：[`www.aipla.org/`](https://www.aipla.org/)（例如，[`www.aipla.org/detail/event/2024/04/23/default-calendar/aipla-cle-webinar-copyright-implications-in-generative-ai`](https://www.aipla.org/detail/event/2024/04/23/default-calendar/aipla-cle-webinar-copyright-implications-in-generative-ai)）

+   **国际商标协会 (INTA)** ：[`www.inta.org/`](https://www.inta.org/)

+   **欧洲专利局 (EPO)** ：[`www.epo.org/`](https://www.epo.org/)（例如：[`www.epo.org/en/about-us/statistics/patent-index-2023/insight-artificial-intelligence`](https://www.epo.org/en/about-us/statistics/patent-index-2023/insight-artificial-intelligence)）

+   **美国专利商标局 (USPTO)** ：[`www.uspto.gov/`](https://www.uspto.gov/)

+   **世界知识产权组织（WIPO）**：[`www.wipo.int/`](https://www.wipo.int/)（例如，[`www.wipo.int/export/sites/www/about-ip/en/frontier_technologies/pdf/generative-ai-factsheet.pdf`](https://www.wipo.int/export/sites/www/about-ip/en/frontier_technologies/pdf/generative-ai-factsheet.pdf)）

+   **LexisNexis**：[`www.lexisnexis.com/`](https://www.lexisnexis.com/)

+   **汤姆森** **路透社**：[`legal.thomsonreuters.com/en/search-results#q=LLM%20code&t=Legal&sort=relevancy`](https://legal.thomsonreuters.com/en/search-results#q=LLM%20code&t=Legal&sort=relevancy)

我们不能谈论将 LLM 生成的代码集成到工作流程中的挑战，而不提及依赖管理，它是代码集成的支柱。

## 依赖管理

**依赖管理**是识别、控制和管理软件项目所依赖的外部软件组件（如库、框架或工具）的过程。这些外部组件称为**依赖项**。

如果这些依赖项没有得到妥善管理，并且无法按预期工作，那么整个应用程序可能会停止工作，进而影响到许多或所有用户的使用。这些故障可能非常尴尬，并对业务造成不利影响。如果依赖项较少，风险和维护成本也会相应减少。

### 在 LLM 生成代码集成中的重要性

在集成 LLM 生成的代码时，依赖管理变得更加关键，原因有以下几点：

+   **不可预测的依赖项**：LLM 可能引入未预见的依赖项，导致兼容性问题或安全风险

+   **版本冲突**：不同的依赖项可能有冲突的版本要求，导致构建失败或运行时错误

+   **安全漏洞**：过时或被破坏的依赖项可能会使整个应用程序暴露于安全威胁中

+   **性能影响**：低效或臃肿的依赖树可能会降低应用程序性能

+   **可维护性**：适当的依赖管理对未来理解和修改代码库至关重要

依赖管理的最佳实践包括：

+   **依赖分析**：彻底分析 LLM 生成的代码引入的依赖项，以识别潜在的冲突或问题

+   **版本控制**：使用强大的版本控制系统来追踪依赖项的更改，并在必要时回退到先前的版本

+   **依赖管理工具**：使用 npm、Apache Maven、Gradle 或 pip 等工具来有效管理依赖项

+   **定期更新**：保持依赖项更新至最新版本，以便受益于错误修复和安全补丁

+   **依赖漏洞扫描**：定期扫描依赖项中的已知漏洞，并及时解决

+   **依赖最小化**：努力减少依赖项数量，以降低复杂性和潜在问题

通过遵循前述的实践，您可以减轻与 LLM 生成代码相关的风险，并确保您的应用程序和系统的稳定性与安全性[QwietAI, Sonatype]。

希望用于检查和修正这些问题的自动化工具很快就会被开发出来。也许它们会在其操作中使用 LLM。

关于依赖管理，我们就讲到这里。接下来，我们必须讨论可解释性，因为我们希望确保代码是可以理解的，能够按我们预期的方式运行，并且在需要时能够向他人解释其工作原理。

## 可解释性

合理地说，当前有朝着更具可解释性和透明度的代码发展的趋势，但如果我们采用错误的方法，使用 LLM 生成的代码（通常是黑箱代码）可能会使这变得更加困难。这些 AI 生成的代码片段可能会偏离已建立的编码惯例，产生不可预见的依赖关系，或者做出与现有代码库不兼容的假设。可解释的 AI 代码被称为**XAI**。

无论是 AI 还是人类，整合由不同作者生成的代码时，由于不了解其他脚本、函数、类和装饰器的写法，可能会采取不同的方法并做出不同的假设，从而引入难以理解的复杂性。如果额外的代码与整体软件架构不符，这种情况甚至更糟。

使用 LLM 生成的代码可能会遇到以下问题：

+   **隐藏的假设和偏见**：LLM 可能会将训练数据中的隐藏偏见或假设融入到生成的代码中。这些偏见可能难以识别，并可能导致意外的行为或错误。

+   **缺乏可追溯性**：理解 LLM 生成输出中具体代码片段的来源可能具有挑战性。这使得定位错误源或有效修改代码变得困难。

+   **动态行为**：LLM 能够生成表现出动态行为的代码，这使得难以预测代码在不同条件下的表现。这可能导致意外的结果，并增加调试的挑战。

+   **过度依赖注释**：虽然注释可以提高代码可读性，但过度依赖注释来解释 LLM 生成的代码可能会产生误导。尤其当代码本身复杂或模糊时，注释可能无法准确反映代码的实际行为。

这些挑战凸显了*严格*测试、代码审查和在将 LLM 生成代码整合到软件系统时进行仔细集成的重要性。

这是研究 XAI 的一个好来源：[ACM_DL]。

既然我们已经理解了将 LLM 生成的代码整合到编码工作流中的挑战，那么我们可以开始思考未来，研究人员可能会如何致力于改善 LLM 的局限性。

# 解决局限性的未来研究方向

我们的人工-机器文明能为大语言模型做些什么，以去除和减轻更多的限制并推动技术进步？

让我们在这里考虑一些想法。

## 持续学习

如果我们能够让大语言模型持续吸收新数据并频繁再训练（例如每天），它们就不会长时间过时，并且可以在短时间内经历多次改进的迭代。

## 新颖的架构

探索新的神经网络架构和混合模型可能会在大语言模型的能力上带来突破。

新的硬件设备、编码和测试实践一直对机器学习的进步至关重要，但真正推动人工智能力量的是新的神经网络架构。

神经网络让我们能够训练软件做出自己的决策并更具适应性，而不是将每种场景都编程并硬编码进去。

在深度学习之前，神经网络较为薄弱，无法解决复杂的问题：物体检测、翻译、物体描述等等。

在大语言模型（LLMs）之前，公众无法用自然语言（如英语、法语、日语等）快速查询人工智能，获取知识，生成文本、AI 艺术、AI 音乐、AI 电影和 AI 生成的代码。

每一代新的机器学习架构都为世界带来新的能力。

很可能，一些提出的新机器学习架构将带来世界能够大大受益的进展。

## 计算效率

优化模型大小和计算需求可以让大语言模型（LLMs）更加可访问且具备更好的可扩展性。

为了模拟人类思维并理解查询或话题的上下文，大语言模型需要数十亿个参数（神经网络训练权重）；Meta 的最新大语言模型 Llama 3.1 有一个版本拥有 4050 亿个参数。GPT-4o 则有超过 2000 亿个参数。这些模型对内存的需求巨大（Llama 3.1 400B 需要 800GB），因此普通人无法使用这些最强大的模型。它们消耗了过多的资金、空间、能量和时间来购买硬件。人类大脑，实际上任何动物的大脑，在能量和空间使用上的效率远高于直接使用内存。如果我们能在这些方面提高大语言模型的效率，就能大大实现大语言模型的普及，帮助普通人生活得更好，并加速技术发展。

减轻负担的方法包括使用闪存注意力、更低的精度以及前述的架构创新[HuggingFace_Optimizing, Snowflake]。闪存注意力是一种内存效率更高的注意力算法，并且在使用 GPU 内存时表现更好。

量化或低精度涉及使用不太精确的数字；因此，模型可以使用 8 位数字代替 16 位数字进行存储。8 位数字是 2⁸ = 256 个数字（例如图片中的 RGB 值：0 到 255），而 16 位数字是 2¹⁶，即 65,536 个不同的值。所以，如果你仅使用 8 位精度存储模型，你将节省大量的计算、时间和能源，但模型的精度会降低。这就是为什么会有 Llama 8b 和 Llama 70b 模型的原因；它们更小，可以在更多普通计算机硬件上运行。

剪枝也可以在不显著降低性能的情况下，减少模型的大小和推理时间。

更加专业化的架构，如旋转嵌入、Alibi、分组查询注意力和多查询注意力，能够提升 LLM 的效率。你可以从 [HuggingFace_Optimizing] 了解更多相关信息。这超出了本章的讨论范围，Hugging Face 提供了更多关于架构的信息。

对于 LLM，推理是指你给 LLM 提供一个提示并获得回应 [Gemini, Symbli]。

如果 LLM 能够更加高效地使用能源，它们的训练成本就会降低，从而实现 LLM 训练的民主化。已经有工作致力于开发更轻量的 LLM，这些模型可以在更小和更移动的设备上运行，因此这个问题是已知的。

更高效的 LLM 可能能够更好地理解脚本、类、函数和装饰器的上下文。

如果代码运行得更快，那么漏洞也能更快速地被发现。

更高效的模型和更好的上下文理解也可能有助于依赖关系的理解。

## 专业化训练

如果你希望 LLM 能够深入理解特定问题或应用所需的代码，它会通过针对这些问题和解决方案的特定训练表现得更好。这是因为它会更加熟悉该领域的工作和最佳实践。

希望 LLM 的训练能够变得更加高效，从而降低成本并简化过程。

更多的安全性训练可能对 LLM 及其用户有所帮助。可以通过安全数据集来实现；这些数据集专门设计用于教授漏洞和最佳实践。

LLM 可能能够通过依赖项、所需的代码库和版本以及目标硬件和使用场景进行训练。

以上就是关于持续学习、新架构、效率和专业化训练的内容；现在，是时候进行本章总结了。

# 摘要

本章中我们思考并学习了 LLM 的限制，包括缺乏理解、缺乏上下文、高计算需求、对训练数据的依赖以及安全风险。我们还简要讨论了一些评估 LLM 性能的指标。

我们尝试克服这些限制，并探讨了几条有前景的路径，旨在创造更强大的 LLM。

本章还讨论了 IP 问题、LLM 需要具备可解释性，以及从哪里可以了解更多关于这些问题的信息。

在下一章中，我们将学习关于基于 LLM 的编码中的协作和知识共享，因为这是你改变世界、帮助他人并让自己名声大噪的方式。

# 参考文献

+   *ACM_DL* : “通过基于场景的设计探索生成 AI 解释性的调查”，Jiao Sun，Q. Vera Liao，Michael Muller，Mayank Agarwal，Stephanie Houde，Karthik Talamadulupa 和 Justin D. Weisz，[`dl.acm.org/doi/fullHtml/10.1145/3490099.3511119`](https://dl.acm.org/doi/fullHtml/10.1145/3490099.3511119)

+   *CodeBLEU* : “CodeBLEU：代码合成的自动评估方法”，Shuo Ren，Daya Guo，Shuai Lu，Long Zhou，Shujie Liu，Duyu Tang，Neel Sundaresan，Ming Zhou，Ambrosio Blanco 和 Shuai Ma，[`arxiv.org/abs/2009.10297`](https://arxiv.org/abs/2009.10297)

+   *confident_ai* : “LLM 评估指标：终极 LLM 评估指南”，Jeffrey Ip，[`www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation`](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)

+   *Gemini: Gemini,* *Alphabet* : https://gemini.google.com/

+   **HuggingFace_Optimizing** : “在生产中优化你的 LLM”，Patrick von Platen，https://huggingface.co/blog/optimize-llm

+   *Liu_2024* : “网络中的大型语言模型：工作流程、进展和挑战”，Chang Liu，Xiaohui Xie，Xinggong Zhang 和 Yong Cui，[`arxiv.org/html/2404.12901v1`](https://arxiv.org/html/2404.12901v1)

+   *Mercier* : “理性的谜团”，Hugo Mercier 和 Dan Sperber，https://www.hup.harvard.edu/books/9780674237827

+   *Prompt_Drive* : “大型语言模型（LLMs）的限制是什么？”，Jay，[`promptdrive.ai/llm-limitations/`](https://promptdrive.ai/llm-limitations/)

+   *QwietAI* : “AppSec 101 – 依赖管理”，QweitAI，https://qwiet.ai/appsec-101-dependency-management/#:~:text=Dependency%20Management%20Tools,-The%20software%20development&text=These%20tools%20are%20the%20backbone,ensure%20compatibility%20across%20the%20board .

+   *Snowflake* : “使用 Meta 的 Llama 3.1 405B 和 Snowflake 优化 AI 栈实现低延迟和高吞吐量推理”，Aurick Qiao，Reza Yazdani，Hao Zhang，Jeff Rasley，Flex Wang，Gabriele Oliaro，Yuxiong He 和 Samyam Rajbhandari，[`www.snowflake.com/engineering-blog/optimize-llms-with-llama-snowflake-ai-stack`](https://www.snowflake.com/engineering-blog/optimize-llms-with-llama-snowflake-ai-stack)

+   *Sonatype* : “什么是软件依赖？”，Sonatype，https://www.sonatype.com/resources/articles/what-are-software-dependencies

+   *Stevens* : “什么是语义相似性：在检索增强生成（RAG）背景下的解释”，Ingrid Stevens，[`ai.gopubby.com/what-is-semantic-similarity-an-explanation-in-the-context-of-retrieval-augmented-generation-rag-78d9f293a93b`](https://ai.gopubby.com/what-is-semantic-similarity-an-explanation-in-the-context-of-retrieval-augmented-generation-rag-78d9f293a93b)

+   *Symbli* : “LLM 推理性能监控指南，”Kartik Talamadupula

+   https://symbl.ai/developers/blog/a-guide-to-llm-inference-performance-monitoring

+   *Wiki_Agent* : “智能体，”维基百科，https://en.wikipedia.org/wiki/Intelligent_agent
