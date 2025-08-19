

# 第九章：培养 LLM 增强编程中的协作

本章旨在培养一个协作、开源、透明和共享学习的 LLM 增强编程环境。它提出了建立最佳实践的建议，用于共享 LLM 生成的代码及相关知识。你将找到一些策略来协作，确保 LLM 生成的代码中蕴含的专业知识能够在开发团队中得到有效共享。通过鼓励合作文化，本章概述了如何充分利用 LLM 的潜力，创建一个丰富的共享知识和良好编码实践的生态系统。

在本章中，我们将涵盖以下主要内容：

+   为什么要分享 LLM 生成的代码？

+   代码共享的最佳实践

+   知识管理：捕获和共享专业知识

+   最好地利用协作平台

# 技术要求

对于本章，你需要以下内容：

+   获取本书中的代码：[`github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main`](https://github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main)。如后面所述，你需要一个 GitHub 账户才能正确使用这些代码。

+   你可能想要注册 Kaggle、Codefile、GitLab、Stack Overflow 和/或 Dabblet、Trello、Jira、Monday.com 或 Miro 等平台。

+   你需要访问一个 LLM/聊天机器人，如 GPT-4、Copilot 或 Gemini，每个工具都需要登录。对于 GPT-4，你需要一个 OpenAI 账户；对于 Gemini，你需要一个 Google 账户，这对 Colab 也有帮助；Copilot 则需要一个 Microsoft 账户。

让我们开始本章的内容，首先讨论为什么分享代码（包括 LLM 生成的代码）而不是独自保留。

# 为什么要分享 LLM 生成的代码？

当然，LLM 已经彻底改变了开发者的编程方式。然而，这些工具的潜力远不止于提升个人生产力。分享 LLM 生成的代码创造了一种协作环境，这不仅有助于集体学习，还能加速问题解决并推动创新。

分享代码对所有程序员都有帮助，包括学生、学者、业余爱好者、软件工程师、开发者、数据科学家以及其他编写代码的科学家。

## 分享代码的好处

分享代码有多个好处，可以显著提升你的开发过程。

### 与团队分享代码

与团队分享代码有助于促进更大的透明度。当你分享代码时，每个人都可以看到过程，这有助于在团队内部建立信任和责任感。

它还能提高调试效率。通过多方视角查看代码，识别和修复 bug 变得更加快速和高效。

代码应在公司内部共享，以确保开发过程顺利进行。对于大学里的学生和独立研究人员来说，是否共享代码并不那么明显；然而，实际上，通过更多的眼睛和大脑来审查代码、使其更易用并改进它，确实会有很大帮助。

学术代码通常不够生产就绪，并且可能难以理解、运行或在后续研究项目中进一步开发。

对于提升开发速度和代码质量，**配对编程**非常有帮助。

### 与世界共享代码——开源

更广泛地分享代码有助于建立更强大的社区。共享知识能够促进社区意识，鼓励开发者之间的合作。这最终会导致一个更强大、更充满活力的开发者生态系统，同时也能产生更加稳定、用户友好且具备用户所需功能（没有臃肿软件——用户不需要但软件生产商仍然强行加入的功能）的代码和软件。

这些好处并非仅仅理论上的。许多案例研究表明，采用代码共享的团队表现和士气都有所提高。例如，GitHub Octoverse 的一项研究表明，开源项目通常会因为多样化的贡献而推进得更快。以下是他们报告的链接：[`octoverse.github.com/`](https://octoverse.github.com/)。

共享代码并不总是可行的，因为公司可能希望开发新技术而不将其提供给竞争对手，而学术研究人员则希望将最终的研究成果发布在学术期刊上，并且由于竞争原因也无法分享代码。

所以，我建议你在可以分享代码时尽量分享，尤其是当代码是免费的，很多代码都是免费的。

## 现实世界的例子

**Linux**是最著名的开源操作系统之一。它由 Linus Torvalds 于 1991 年创建，现已发展成一个强大而多功能的平台，广泛应用于个人电脑、服务器甚至智能手机等设备。其开源特性意味着任何人都可以查看、修改和分发源代码，这使得有一个庞大的开发者社区为其改进做出了贡献。

据估计，**Linux**的开发者有大约 13,500 人。

这种协作方式已促成了一个高度安全和稳定的操作系统，它驱动着互联网基础设施中的大部分内容。

坦率地说，Linux 对于一些新用户可能有一定的学习曲线。某些硬件可能存在驱动兼容性问题，且企业级支持需要付费。

然而，Linux 操作系统中的漏洞不断被社区发现并修复。

**Linux**拥有一个庞大且活跃的安全研究人员和开发者社区。这个社区能够迅速识别并修复安全漏洞，这使得 Linux 具备了强大的安全功能，如组权限、防火墙和加密等。

Linux 还有其他优点，如良好的性能、定制化、成本效益和快速补丁。当然，作为开源软件，开发过程是透明的[ *Linux_Foundation* , *CBT_Nuggets* ]。

另一个开源开发如此有效的绝佳例子是**Apache 软件基金会**（**ASF**）。ASF 监管着一系列开源项目，其中最著名的成功之一就是 Apache HTTP Server，通常简称为 Apache。Apache 于 1995 年发布，凭借其可靠性和灵活性，迅速成为最流行的网页服务器软件。

Apache 的开源模式允许来自全球的开发者为其代码库做出贡献。这种协作方式确保软件始终与最新的网络标准和安全实践保持同步。就像 Linux 一样，Apache 也从贡献者的多元化视角和专业知识中受益，这促进了持续的改进和创新。

ASF 还支持众多其他开源项目，培养了社区中的协作和创新文化。像 Hadoop、Cassandra 和 Spark 这样的项目只是 ASF 旗下涌现出的具有深远影响的软件的几个例子。这种广泛的支持有助于构建一个强大的生态系统，让开发者可以分享知识、工具和最佳实践。

此外，ASF 对开源原则的坚持意味着任何人都可以参与其中。无论你是经验丰富的开发者，还是刚刚入门的新人，你都可以为这些项目做出贡献，向他人学习，并推动技术进步。这种包容性不仅加快了开发过程，还建设了一个强大而支持性的社区。

这种方法的好处有很多文献记录。例如，GitHub 的一项研究显示，开源项目通常能更快地发展，这是因为贡献的多样性。开源项目能够利用全球的人才资源，从而更快地修复漏洞、推出更多创新功能，并且整体软件质量更好。

总之，ASF 展示了开源开发的力量。通过营造协作环境并支持广泛的项目，ASF 推动了技术创新，建设了强大而充满活力的社区。

想要了解更多信息，你可以访问以下资源：*Apache_SF* , *GitHub_Octoverse* , *Apache_Projects* , *Wiki_Ken_Coar* , *SpringerLink* , 和 *Wikiwand* 。

Linux 和 Apache 展示了开源开发能做到什么。它们展示了如何通过社区驱动的项目，开发出高效且广泛采用的软件解决方案。通过让任何人都能参与，开源项目从多样化的观点和专业知识中受益，推动了持续的改进和创新。

2023 年一篇来自 Techjury 的文章表示，全球有 3280 万 Linux 用户，"*全球排名前一百万的 Web 服务器中有 96.3% 使用 Linux*"，并且服务器市场份额中，33.9% 是 Red Hat（Linux）[ *Techjury* ]。

W3Techs 表示，Adobe.com、Netflix.com、Spotify.com、Samsung.com、Theguardian.com、Mit.edu 和 eBay.com 等网站都使用 Apache [ *W3Techs* ]。根据 Apache HTTP Server 的 Wikipedia 页面，它是开源的，并由 ASF 管理，该页面指出在 2022 年 3 月，Apache 为全球最繁忙的 100 万个网站提供服务，占 23.04% 的份额 [ *Wiki_Apache* ]。这一份额高于 NGINX、Cloudflare 和 Microsoft Internet Information Services。

通过了解这些实际应用，开发者可以更好地认识到共享 LLM 生成的代码以及它带来的集体利益的价值。

了解并遵循最佳实践总是好的，无论是为了自己，还是为了他人。所以，接下来的部分将讲解最佳实践。

# 代码共享的最佳实践

在你的组织内部创造一种优先考虑代码共享的文化，需要实施确保共享代码有用、易于理解并且易于维护的最佳实践。

最佳实践包括以下内容：

+   良好的文档编写

+   良好地使用版本控制

+   遵循安全最佳实践

+   对相关创作者给予适当的归属或信用

+   彻底测试代码

+   遵守编码标准

+   坚持持续改进

以下是每项内容的详细说明。

## 文档

清晰的文档对于其他人理解你代码的目的和功能是 100% 必须的。否则，他们很难使用或维护你的代码。文档可以包括内联注释、README 文件和使用示例。

有一些文档工具，包括 Sphinx（[`www.sphinx-doc.org/en/master/`](https://www.sphinx-doc.org/en/master/)）和 MkDocs（[`www.mkdocs.org/`](https://www.mkdocs.org/)），它们可以帮助自动化从代码库生成文档，使其更易于维护。

## 一致的编码标准

使用广泛的编码约定（或建立自己的编码约定）。统一的编码风格提高了可读性，减少了多个开发者共同在同一代码库上工作时的摩擦。

像 ESLint（[`eslint.org/`](https://eslint.org/)）用于 JavaScript，或 Pylint（[`pylint.pycqa.org/en/latest/`](https://pylint.pycqa.org/en/latest/)）用于 Python 等工具，可以帮助自动执行编码标准。

ESLint 允许你共享配置、创建自定义规则，并保持与当前最佳实践同步，从而让你作为开发者或程序员的工作更加轻松。

Pylint 与 ESLint 具有类似的功能。它执行静态代码分析，基于 PEP 8（[`peps.python.org/pep-0008/`](https://peps.python.org/pep-0008/)）提供规则，并为你的代码提供评分和报告。Pylint 还会保持与最佳实践的同步。

Linting 工具有助于保持代码的一致性。它们会检查错误，允许定制，并通过自动反馈来帮助学习。

我在*第四章*中写了更多关于如何使代码易读的内容，包括文档编写和编码规范。

## 版本控制

**版本控制系统**（**VCSs**）如 Git、Apache **Subversion**（**SVN**）、Mercurial 和**Concurrent Versions System**（**CVS**）对于跟踪代码的变化至关重要。它们允许开发人员协作而无需担心覆盖彼此的工作。

它们允许不同版本的存在，既包括时间上的版本，也包括并行的多版本用于不同目的（分支和合并）。VCS 使开发人员能够在最新更新无法按预期工作时，返回到早期的版本（回退）。它们还允许可靠的代码备份。

你可以区分不同版本的工作：逐行检查此版本和那个版本之间的差异。

LaTeX 使用 diff。LaTeX 非常适合写科学论文和书籍，通常被计算机科学家、物理学家、数学家、工程师等使用。非数理专业的人通常不会使用。它的发音是“layteck”，因为 X 实际上是希腊字母 Chi 的硬“ch”音（发音为“Kai”）。

如果你还没有备份代码并使用版本控制，请从今天开始！你真的不想失去整个项目（即使是博士项目），然后不得不重新做一遍！这非常重要！

我提到过，你可以使用版本控制系统（VCS）来写 LaTeX；你也可以用它来处理其他文档写作，甚至是任何文件类型：书籍、画作、声音和视频。我不知道你能否对比一幅画或一个视频，但你可以备份你的工作。

实施分支策略（例如 Gitflow）并编写清晰的提交信息可以增强协作。更多内容请见：[`en.wikipedia.org/wiki/List_of_version-control_software`](https://en.wikipedia.org/wiki/List_of_version-control_software)。

查看 GitLab 关于 Git 最佳实践的指南：[`about.gitlab.com/topics/version-control/version-control-best-practices/`](https://about.gitlab.com/topics/version-control/version-control-best-practices/)。

请查看 SVN 最佳实践：[`svn.apache.org/repos/asf/subversion/trunk/doc/user/svn-best-practices.html`](https://svn.apache.org/repos/asf/subversion/trunk/doc/user/svn-best-practices.html)。

## 代码安全最佳实践

确保代码遵循安全最佳实践。检查漏洞（注入攻击、缓冲区溢出、身份验证缺陷、加密弱点），确保代码质量，确保符合法规和标准（如**常见弱点枚举**，或**CWE**），识别性能瓶颈，并进行优化。

你可以使用工具扫描常见的安全问题。

工具包括 **静态应用程序安全测试**（**SAST**），如 SonarQube 或 Fortify 静态代码分析器；**动态应用程序安全测试**（**DAST**），如（OWASP）ZAP 或 Burp Suite；**软件组成分析**（**SCA**），如 Snyk 或 Mend.io（前身为 WhiteSource）；**互动应用程序安全测试**（**IAST**），如 Contrast Security 或 Synopsys 的 Seeker；以及**运行时应用程序自我保护**（**RASP**），如 Imperva RASP 或 Signal Sciences（由 Fastly 拥有）。此外，使用代码审查工具，如 GitHub 代码扫描或 Phabricator（由 Phacility 开发），以及在*一致的编码* *标准*部分提到的 linting 工具。

## 正确的署名

给与应有的荣誉。就像你应该为开发者和作者的贡献给予信用一样，你也应该给你使用的 LLM 工具提供归属[ *Copilot* , *Gemini* ]。

为代码提供正确且完整的归属对于透明度、尊重知识产权以及协作至关重要。

这里是你可以执行的操作。

对于 LLM（大语言模型）生成的代码，按照以下步骤操作：

1.  在 Python 中，你可以这样写注释：

    ```py
    # This code was generated by Microsoft Copilot
    ```

1.  你也可以提供用于生成它的提示：

    ```py
    # Prompt: "Write a function to calculate the factorial of a number"
    ```

对于由人编写的代码，例如 Python **# Author: Jane Smith** 用来标注作者，包括你自己。你可以这样做：**# Additional optimizations by** **Joel Evans.**

你可以提供联系方式，例如以下内容：

**#** **GitHub:** [`github.com/janedoe`](https://github.com/janedoe)

你甚至可以链接到来源：

**#** **Source:** [`github.com/janedoe/project`](https://github.com/janedoe/project)

你可以像这样链接到文档：

**#** **Documentation:** [`www.microsoft.com/en-us/edge/learning-center/how-to-create-citations`](https://www.microsoft.com/en-us/edge/learning-center/how-to-create-citations)

你还应该指定一个许可协议，比如 **# License:** **MIT License**。

你还可以选择 GNU AGPLv3、Apache 2.0 许可协议、Boost 软件许可 1.0，甚至是“The Unlicense” [ *Choose_License* ]。

## 彻底测试代码

对单个组件进行单元测试，进行集成测试以确保所有部分协同工作，系统测试在类似生产环境中进行，验收测试以查看软件是否满足业务需求并准备好部署。

测试可以尽可能地自动化，以确保你在时间使用上高效并保持一致性。然而，确保你的测试覆盖了多种场景：边缘情况、特殊情况和潜在的失败点。

进行用户体验/可用性测试。测试用户是否能轻松使用你的软件，并确保它满足目标用户（或实际使用者）的需求。软件应尽可能对更多人开放，而不是偏向某种能力，但你可能需要在安全性上有所偏向。

测试代码的安全性，正如之前所提到的。

详细记录所有测试（测试计划、测试用例和测试结果），以便相关人员可以理解并重现所需的测试。

进行回归测试。在更新后再次测试软件，以确保一切仍然按预期工作。

参见*第三章*了解更多有关测试代码的内容。

这引导我们进入下一小节：持续改进。

## 持续改进

LLM 从一开始就需要持续改进，因为你的第一个提示可能不会给你理想中的、完全可用的代码。你几乎总是需要在尝试代码和向 LLM 请求稍微更好的代码之间进行迭代。

当你拥有可用代码后，你仍然需要继续开发自己的代码或他人的代码。

定期进行同行评审，以寻找 LLM 生成代码中的问题。继续进行单元测试和集成测试。事实上，你或你的组织可以构建**持续集成/持续部署**（**CI/CD**）管道，进行代码质量检查和自动化测试等操作。通过这种方式，每次更改都会在合并到代码库之前进行测试。

值得注意的是，技术企业家 Gregory Zem 也提供了 LLM 生成代码的最佳实践：`medium.com/@mne/improving-llm-code-generation-my-best-practices-eb88b128303`。

遵循这些最佳实践，团队可以创建一个易于阅读、维护和贡献的共享代码库。

如果你是一个没有工作开发团队的个人，试着在现实生活中或在线（如 LinkedIn、Discord 等）找到朋友，与他们交换想法，进行结对编程，共享代码，进行代码审查和测试。

尽管“孤独的程序员、黑客或科学家能够独自进入一个安静的房间，短时间内快速创造出改变世界的工作”在某些圈子里是一个流行的形象，但独自工作、不向他人寻求建议、反馈或测试，可能不会帮助你编写出最佳代码。因为你无法从他人（无论是客户/终端用户还是代码开发与批评类朋友）那里获得快速的反馈和极为有帮助的意见，甚至包括书籍推荐。

你也无法快速听到该领域的最新发展，比如软件或你选择的应用程序。

对于开源代码，以正确的方式分享代码更加重要。你的项目有可能会发展成被全球数百万甚至数十亿人使用的工具。你可以做出贡献，产生影响，并为自己的成就感到自豪。

# 知识管理——捕获并共享专业知识

有效的知识管理对于最大化 LLM 生成代码的价值至关重要。

## 创建知识库

你可以拥有一个充满教程、代码片段和设计模式的仓库——这就是一个集中式仓库可以提供的功能。像 Confluence ([`www.atlassian.com/software/confluence`](https://www.atlassian.com/software/confluence)) 或 Notion ([`www.notion.so/`](https://www.notion.so/)) 这样的平台非常适合存储代码和文档，使整个团队或朋友小组可以轻松访问 [*Atlassian_Confluence*，*Notion*]。

代码在不断演进，我们的知识也应该如此，因此，定期更新和版本化这些仓库能确保每个人都在使用最新和最好的信息。

## 定期进行知识分享会议

无论你是处于一个开发团队、拥有个人项目的研究小组，还是目前独自一人，分享想法和代码给他人可以获得惊人的反馈并加速工作进程。

提供一段基本的代码，勉强满足你的使用或业务案例需求，是最好的起点。不要在没有获取代码或功能反馈的情况下进行大量工作。除非客户要求特定的完美功能并明确反馈，否则不要追求完美。大学作业不要求完美，只要求良好的成绩和快速的工作。

头脑风暴时间：如果你们有定期的聚会，团队讨论新的发现、洞察和最佳实践，这将是建立团队凝聚力并互相学习的绝佳方式。

团队成员分享他们发现的有趣想法和代码的休闲会议有助于鼓励知识交换的文化，并保持每个人都在学习和激励。

人类在玩耍一些酷炫的玩具或进行实验时最有效。所以，和新点子、代码一起玩耍并讨论它们如何使用是非常有价值的。不要只是坐在办公桌前吃午餐，闭口不言。看看你的同事和朋友在没有压力的情况下说些什么：休息/午餐、饮料等。

## 同行指导——分享智慧

导师计划可以作为促进团队内部知识传递和合作的宝贵工具。通过将经验丰富的开发人员与新加入的学员配对，组织可以创造一个支持性环境，让有经验的专业人士分享他们的专业知识并指导他人。

这对学员显然是有价值的：他们可以了解事情如何运作以及人们如何看待它。然而，导师也能整理自己的思路，并通过教育初级人员并回答他们的问题来测试和提升自己的知识。

一名初级开发人员的年龄可能比该领域中更有经验的专业人士还大。

此外，实施伙伴制度有助于顺利地进行新团队成员的入职过程。通过为每位新员工分配一位导师，组织可以确保从第一天起就分享知识，帮助新员工快速成为团队的高效成员。

如果实施工作影子，这应该以一种不太单调并能充分传授知识的方式进行。如果你是初级成员，要做好工作影子，重要的是要准备充分、积极参与并主动。研究你所影随的角色，并准备好问题。在整个体验过程中要积极观察，做笔记并提问。展现主动性，随时提供帮助。最后，在影子体验结束后，跟导师进行跟进，感谢他们并讨论下一步。

如果你是工作影子过程中的导师，要提供明确的指导，分享你的专业知识，鼓励提问，并提供机会让影子人员参与任务。人们通常是通过实践来学习的。评估他们的表现并提供反馈，帮助他们提升技能。

通过优先考虑知识管理，我们可以充分挖掘 LLM 生成代码的全部潜力。想象一个团队，在这个团队中，专业知识触手可及，大家不断学习和成长。那才是一个技术高超且成功的团队！ [ *You.com* , *Gemini* ]

# 最大化利用协作平台

现代软件开发依赖于协作平台和工具，这些工具简化了团队成员之间的沟通和协调，正如之前提到的那样。这些平台对确保每个人都在同一页面上至关重要，从最初的规划到最终的部署。接下来我们来探讨一些充分利用它们的好方法。

## 代码审查工具

我们之前提到过代码审查、安全性、测试和迭代，现在我们可以更专注于一些用于代码审查的工具。

代码审查是开发过程中的一个关键环节。它为团队成员提供了提供建设性反馈的机会，并确保代码质量。它能帮助你避免将不能正常工作的代码推入生产环境，避免尴尬。这对于开发者来说是尴尬的，但对于他们所工作的组织也是相当危险的。大学的工作更倾向于个人，但对研究小组却不利。

GitHub 和 GitLab 等平台提供了内置的代码审查功能，使得协作变得无缝。这些工具允许开发者对特定代码行发表评论、建议修改并批准更改，所有这些都在同一个界面内进行。这不仅提高了代码质量，还在团队内培养了持续学习和改进的文化。

## 项目管理软件

跟踪进展和管理工作流程对任何开发团队都至关重要。像 Jira 和 Trello 这样的工具旨在帮助团队保持组织性和进度。这些平台提供任务分配、进度跟踪和截止日期管理等功能，确保每个人都知道需要做什么以及何时完成。

通过这些工具实施敏捷方法论，团队可以变得更加灵活和适应变化。它们能够快速响应变化，按时交付高质量的软件。敏捷实践借助这些项目管理工具，鼓励定期检查、迭代开发和持续反馈，所有这些都使得开发过程比其他方法更加高效和有效。

### Jira

Jira（由 Atlassian 提供）在采用敏捷方法论的环境中使用频繁，尤其是在 Scrum 和看板（Kanban）框架中。它提供了强大的功能来跟踪问题、缺陷和任务，非常适合大型团队和更复杂的项目。Jira 的可定制工作流和详细报告功能使团队能够根据其特定流程定制该工具。它还能够与其他开发工具良好集成，因此提供了管理整个软件开发生命周期的多种方式。

### Trello

另一方面，Trello 以其简单性和视觉化的项目管理方法而闻名。它采用了一个卡片和看板的系统，直观且易于使用，非常适合小型团队或不需要 Jira 繁琐功能的项目。Trello 的灵活性使团队能够根据自己的工作流程组织任务，无论是团队中的软件开发还是个人项目。它的拖拽界面和简洁设计使各个技术层级的用户都能轻松使用。

Trello 由 Fog Creek Software 开发，现在由 Atlassian 开发，后者收购了它。

### Miro

Miro（由 ServiceRocket Inc.提供）是一个在线白板平台，能够显著提升软件开发团队的生产力和协作效率。通过提供一个集中的空间来进行头脑风暴、规划和项目管理，Miro 帮助团队可视化想法、跟踪进展并识别潜在瓶颈。其直观的界面和丰富的功能使其成为开发者、设计师和项目经理的高效协作工具。

### Monday.com

Monday.com 是一个相当多功能的工作管理平台，提供可定制的界面、直观的视觉看板和强大的自动化功能。通过为团队提供一个集中的位置来管理任务、跟踪截止日期并有效协作，Monday.com 可以简化工作流程并提高整体效率。其用户友好的界面和拖拽功能使其对各种技术背景的团队都可访问，允许他们快速将平台调整为适应特定需求。无论你是在管理市场营销活动、软件开发冲刺还是创意头脑风暴会议，Monday.com 都可以为保持团队组织性和进度提供有价值的工具。大多数这些平台的使用范围超出了软件开发领域。

Monday.com Dev 是专为软件开发团队打造的。它具有 Git 集成、缺陷跟踪、代码集成、敏捷洞察以及用于软件开发的看板。

Monday.com 也是公司的名称。

请参阅本章*知识管理*部分中提到的 Confluence 和 Notion。

### 所有这些平台

这些平台还拥有大量的模板库和预构建框架，为各种软件开发活动提供坚实的基础，帮助团队节省时间和精力。从敏捷规划到用户故事映射，这些平台提供了各种工具来支持整个软件开发生命周期。

这些项目管理平台提供与多种其他应用程序的集成，增强了其功能，允许团队创建无缝的工作流程。无论你需要 Jira 的详细跟踪和报告、Miro 的视觉头脑风暴功能，还是 Monday.com 的直观组织方式，各种项目管理平台都可以显著改善团队协作和项目管理。

集成意味着你可以实现无缝的工作流程管理和实时更新。集成避免了你需要在多个工具之间切换，从而简化了开发过程，提高了效率：

+   **Jira 集成**：Google、GitHub、Confluence、Bitbucket、CircleCI、Figma、Zoom、Slack 等等

+   **Trello 集成**：Google Drive、MS Teams、Miro、Zapier、Jira Cloud、Slack 等等

+   **Miro 集成**：Google Workspace、GitHub、MS Teams、Figma、Zoom、Slack、Jira、Trello 和 Monday.com

+   **Monday.com 集成**：GitHub、GitLab、Bitbucket、Figma、Jira、Trello、Asana、Google Drive、OneDrive、MS Teams、Slack 等等

来源：[ *Gemini* 、 *Jira* 、 *Miro* 、 *Atlassian* 、 *Monday.com* ]

这将引出沟通渠道的话题，这是下一小节的内容。

## 沟通渠道 – 保持对话畅通

高效的沟通是任何成功团队的命脉。软件世界发展非常迅速，因此选择最佳的沟通渠道对项目结果可能产生重要影响。

像 Slack（[`slack.com/`](https://slack.com/)）、Microsoft Teams（[`www.microsoft.com/en-us/microsoft-teams/group-chat-software`](https://www.microsoft.com/en-us/microsoft-teams/group-chat-software)）、Discord（[`discord.com/`](https://discord.com/)）和 Simpplr（[`www.simpplr.com/`](https://www.simpplr.com/)）等工具提供实时通讯。快速提问、项目更新和头脑风暴会议都可以即时发生，促进了协作感，并帮助团队迅速解决问题。

### 集成

协作平台通常与通信工具集成，模糊了界限并创造了无缝的工作流。想象一下，在项目管理板上讨论一个任务后，能够无缝跳转到与队友的聊天中澄清细节。像这样的集成可以改变游戏规则，提升团队的生产力。

通过选择合适的沟通渠道并有效利用它们的集成功能，团队能够培养一种协作氛围，鼓励信息共享、问题解决，并最终实现项目成功。

来自 crowd.dev 的 Sofia de Mattia 也写了一篇关于开发者最佳沟通工具的文章：[`www.crowd.dev/post/6-best-communication-tools-for-developer-communities`](https://www.crowd.dev/post/6-best-communication-tools-for-developer-communities)。

# 摘要

希望你能意识到，在 LLM 增强的编程中，像任何编程一样，启用并帮助建立协作文化对于最大化这些惊人工具的潜力至关重要。这些工具正迅速适应，从而变得更加强大和有用。共享代码、实施最佳实践、有效管理知识以及利用协作平台，都是这一过程中的关键组成部分。

以下是本章的一些关键要点：

+   鼓励团队分享代码并相互学习，从而提升整体开发环境

+   制定文档、版本控制、测试和编码标准的指导方针，以促进更有效的代码共享

+   创建并维护代码库，组织分享会议，确保专业知识得到记录和传播，从而提升结果

+   使用项目管理工具来传达目标和里程碑、存储文件并展示进度

随着软件开发领域的不断发展，有效协作的能力将成为取得成功并推动进步的决定性因素。将 LLM 集成到开发过程中，为团队提供了一个机会，通过协作使用这些工具，创造一个促进创新和持续学习的动态环境。

在*第十章*中，我们将重点讨论非 LLM 工具、代码补全与生成工具、静态代码分析与代码审查工具，以及测试和调试工具。

# 参考文献

+   *Apache_Projects* : *Apache 项目目录* , Apache Software Foundation Team, [`projects.apache.org/`](https://projects.apache.org/)

+   *Apache_SF* : *Apache HTTP Server 项目* , Brian Behlendorf, Ken Coar, Mark Cox, Lars Eilebrecht, Ralf S. Engelschall, Roy T. Fielding, Dean Gaudet, Ben Hyde, Jim Jagielski, Alexei Kosut, Martin Kraemer, Ben Laurie, Doug MacEachern, Aram M. Mirzadeh, Sameer Parekh, Cliff Skolnick, Marc Slemko, William (Bill) Stoddard, Paul Sutton, Randy Terbush, Dirk-Willem van Gulik, [`httpd.apache.org`](https://httpd.apache.org)

+   *Atlassian_Confluence* : *Confluence* , Atlassian Team, [`www.atlassian.com/software/confluence`](https://www.atlassian.com/software/confluence)

+   *CBT_Nuggets* : *Linux 内核开发：一个全球性的努力* , Graeme Messina, [`www.cbtnuggets.com/blog/technology/programming/linux-kernel-development`](https://www.cbtnuggets.com/blog/technology/programming/linux-kernel-development)

+   *Choose_License* : *许可证* , GitHib Inc., [`choosealicense.com/licenses/`](https://choosealicense.com/licenses/)

+   *ESLint* : *用于 JavaScript 和 JSX 的可插入式代码检查实用工具* , Nicholas C. Zakas 和 ESLint 团队, [`eslint.org/`](https://eslint.org/)

+   *GitHub_Octoverse* : *Octoverse：2023 年开源状态与人工智能的崛起* , Kyle Daigle 和 GitHub Staff, [`github.blog/news-insights/research/the-state-of-open-source-and-ai/`](https://github.blog/news-insights/research/the-state-of-open-source-and-ai/)

+   *Jira* : “项目管理软件”, Atlassian Team, [`www.atlassian.com/software/jira`](https://www.atlassian.com/software/jira)

+   *Linux_Foundation* : *开源指南：参与开源社区* , The Linux Foundation, [`www.linuxfoundation.org/resources/open-source-guides/participating-in-open-source-communities`](https://www.linuxfoundation.org/resources/open-source-guides/participating-in-open-source-communities)

+   *Microsoft Teams* : *Microsoft Teams* , Microsoft Corporation, [`www.microsoft.com/en-us/microsoft-teams/group-chat-software`](https://www.microsoft.com/en-us/microsoft-teams/group-chat-software)

+   *Miro* : *利用 Miro，基于人工智能的视觉工作空间加快创新* , Miro, [`miro.com/`](https://miro.com/)

+   *MkDocs* : *MkDocs 文档* , Tom Christie 和 MkDocs 团队, [`www.mkdocs.org/`](https://www.mkdocs.org/)

+   *Monday.com* : *你的工作平台* , Monday.com, [`monday.com/`](https://monday.com/)

+   *Notion* : *一体化工作空间* , Notion Labs Inc., [`www.notion.so/`](https://www.notion.so/)

+   *Pylint* : *Pylint - Python 代码静态检查器* , Sylvain Thénault 和 Pylint 团队, [`pylint.pycqa.org/en/latest/`](https://pylint.pycqa.org/en/latest/)

+   *Slack* : *Slack：工作场所之所在* , Slack Technologies., [`slack.com`](https://slack.com)

+   *Sphinx* : *Sphinx 文档生成器* , Georg Brandl 和 Sphinx 团队, [`www.sphinx-doc.org/en/master/`](https://www.sphinx-doc.org/en/master/)

+   *SpringerLink* : *Apache Web 服务器* , SpringerLink 团队, [`link.springer.com/`](https://link.springer.com/)

+   *Techjury* : *19 个令人惊讶的 Linux 统计数据，很多人不知道* , Muninder Adavelli, [`techjury.net/blog/linux-statistics`](https://techjury.net/blog/linux-statistics)

+   *Trello* : *Trello - 组织一切* , Trello 团队, [`trello.com`](https://trello.com)

+   *W3Techs* : *Apache 的使用统计* , W3Techs, [`w3techs.com/technologies/details/ws-apache`](https://w3techs.com/technologies/details/ws-apache)

+   *Wiki_Apache* : *Apache HTTP 服务器* , 各种编辑器和作者, [`en.wikipedia.org/wiki/Apache_HTTP_Server`](https://en.wikipedia.org/wiki/Apache_HTTP_Server)

+   *Wiki_Ken_Coar* : *Ken Coar - * *维基百科* , [`en.wikipedia.org/wiki/Ken_Coar`](https://en.wikipedia.org/wiki/Ken_Coar)

+   *Wikiwand* : *Apache HTTP 服务器 - * *维基百科* , [`www.wikiwand.com/en/Apache_HTTP_Server`](https://www.wikiwand.com/en/Apache_HTTP_Server)

+   *You.com* : [`you.com/`](https://you.com/) , 也是安卓应用
