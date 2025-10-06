# 前言

学习如何应用 AI 至关重要，它可以在开发传统、教育或任何其他类型的游戏时将乐趣因素提升到新的水平。虚幻引擎是一个强大的游戏开发引擎，允许你创建 2D 和 3D 游戏。如果你想使用 AI 来延长游戏寿命并使其更具挑战性和趣味性，这本书就是为你准备的。

本书从将**人工智能**（**AI**）分解为简单概念开始，以获得对其的基本理解。通过各种示例，你将实际操作实现，旨在突出与 UE4 中游戏 AI 相关的关键概念和功能。你将学习如何通过内置 AI 框架构建每个流派（例如 RPG、策略、平台、FPS、模拟、街机和教育游戏）的可信角色。你将学习如何为你的 AI 代理配置*导航*、*环境查询*和*感官*系统，并将其与*行为树*结合，所有这些都有实际示例。然后，你将探索引擎如何处理动态人群。在最后一章中，你将学习如何分析、可视化和调试你的 AI 系统，以纠正 AI 逻辑并提高性能。

到本书结束时，你对虚幻引擎内置 AI 系统的 AI 知识将深入且全面，这将使你能够在项目中构建强大的 AI 代理。

# 本书面向对象

如果你是一位在虚幻引擎中有些经验的游戏开发者，现在想要理解和实现虚幻引擎中的可信游戏 AI，这本书就是为你准备的。本书将涵盖蓝图和 C++，让不同背景的人都能享受阅读。无论你是想构建你的第一款游戏，还是想作为游戏 AI 程序员扩展你的知识，你都会找到大量关于游戏 AI 的概念和实现方面的精彩信息和示例，包括如何扩展这些系统的一些内容。

# 本书涵盖内容

第一章，《在 AI 世界迈出第一步》，探讨了成为 AI 游戏开发者的先决条件以及 AI 在游戏开发流程中的应用。

第二章，《行为树和黑板》，介绍了在虚幻 AI 框架中使用的两种主要结构，这些结构用于控制游戏中的大多数 AI 代理。你将学习如何创建*行为树*以及它们如何在*黑板*中存储数据。

第三章，《导航》，教你如何让代理在地图或环境中导航或找到路径。

第四章，《环境查询系统》，帮助你掌握制作*环境查询*，这是虚幻 AI 框架的空间推理子系统。掌握这些是实现在虚幻中可信行为的关键。

第五章，*代理意识*，处理 AI 代理如何感知世界和周围环境。这包括视觉、听觉，以及通过扩展系统可能想象到的任何其他感官。

第六章，*扩展行为树*，通过使用蓝图或 C++扩展行为树，带你完成 Unreal 的任务。你将学习如何编程新的*任务*、*装饰器*和*服务*。

第七章，*人群*，解释了如何在提供一些功能性的 Unreal AI 框架内处理人群。

第八章，*设计行为树 – 第 I 部分*，专注于如何实现行为树，以便 AI 代理可以在游戏中追逐我们的玩家（在蓝图和 C++中）。本章，连同下一章，从设计到实现探讨了这一示例。

第九章，*设计行为树 – 第 II 部分*，是上一章的延续。特别是，在我们下一章构建最终的行为树之前，我们将构建最后缺失的拼图（一个自定义的*服务*）。

第十章，*设计行为树 – 第 III 部分*，是上一章的延续，也是*设计行为树*系列的最后一部分。我们将完成我们开始的工作。特别是，我们将构建最终的*行为树*并使其运行。

第十一章，*AI 调试方法 – 记录日志*，检查我们可以用来调试 AI 系统的一系列方法，包括控制台日志、蓝图中的屏幕消息等等。通过掌握日志记录的艺术，你将能够轻松跟踪你的值以及你正在执行的代码的哪个部分。

第十二章，*AI 调试方法 – 导航、EQS 和性能分析*，探讨了 Unreal 引擎内集成的 AI 系统的一些更具体的工具。我们将看到更多与 AI 代码相关的性能分析工具，以及可视化*环境查询*和*导航信息*的工具。

第十三章，*AI 调试方法 – 游戏调试器*，带你探索最强大的调试工具，也是任何 Unreal AI 开发者的最佳朋友——游戏调试器。本章将更进一步，通过教授如何扩展这个工具来定制它以满足你的需求。

第十四章，*超越*，以一些关于如何探索本书中提出（以及其他）概念的建议以及一些关于 AI 的想法作为总结。

# 要充分利用这本书

熟练使用 Unreal Engine 4 是一个重要的起点。本书的目标是将那些使用这项技术的人带到他们能够足够舒适地掌握所有方面，成为项目中的技术领导者和推动者的水平。

# 下载示例代码文件

您可以从这里下载本书的代码包：[`github.com/PacktPublishing/Hands-On-Artificial-Intelligence-with-Unreal-Engine`](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-with-Unreal-Engine)。

我们还有其他来自我们丰富的图书和视频目录的代码包可供选择，这些资源可在**[`github.com/PacktPublishing/`](https://github.com/PacktPublishing/)**找到。查看它们吧！

# 下载彩色图像

我们还提供了一份包含本书中使用的截图/图表的彩色图像的 PDF 文件。您可以从这里下载：[`www.packtpub.com/sites/default/files/downloads/9781788835657_ColorImages.pdf`](https://www.packtpub.com/sites/default/files/downloads/9781788835657_ColorImages.pdf)。

# 使用的约定

本书中使用了多种文本约定。

`CodeInText`: 表示文本中的代码单词。以下是一个示例：“接下来要实现的事件是`OnBecomRelevant()`，并且只有在*服务*变得相关时才会触发”

代码块设置如下：

```py
void AMyFirstAIController::OnPossess(APawn* InPawn)
{
  Super::OnPossess(InPawn);
  AUnrealAIBookCharacter* Character = Cast<AUnrealAIBookCharacter>(InPawn);
  if (Character != nullptr)
  {
    UBehaviorTree* BehaviorTree = Character->BehaviorTree;
    if (BehaviorTree != nullptr) {
      RunBehaviorTree(BehaviorTree);
    }
  }
}
```

当我们希望您注意代码块的特定部分时，相关的行或项目将以粗体显示：

```py
void AMyFirstAIController::OnPossess(APawn* InPawn)
{
  Super::OnPossess(InPawn);
  AUnrealAIBookCharacter* Character = Cast<AUnrealAIBookCharacter>(InPawn);
  if (Character != nullptr)
  {
 UBehaviorTree* BehaviorTree = Character->BehaviorTree;
 if (BehaviorTree != nullptr) {
 RunBehaviorTree(BehaviorTree);
 }
  }
}
```

**粗体**: 表示新术语、重要单词或屏幕上看到的单词。例如，菜单或对话框中的单词在文本中显示如下。以下是一个示例：“从行为树变量中的下拉菜单中选择 BT_MyFirstBehaviorTree。”

警告或重要注意事项看起来像这样。

小贴士和技巧看起来像这样。

# 联系我们

我们始终欢迎读者的反馈。

**一般反馈**: 如果您对本书的任何方面有疑问，请在邮件主题中提及书名，并通过`customercare@packtpub.com`与我们联系。

**勘误**: 尽管我们已经尽一切努力确保内容的准确性，但错误仍然可能发生。如果您在这本书中发现了错误，我们将不胜感激，如果您能向我们报告，请访问[www.packt.com/submit-errata](http://www.packt.com/submit-errata)，选择您的书籍，点击勘误提交表单链接，并输入详细信息。

**盗版**: 如果您在互联网上以任何形式遇到我们作品的非法副本，如果您能提供位置地址或网站名称，我们将不胜感激。请通过`copyright@packt.com`与我们联系，并提供材料的链接。

**如果您有兴趣成为作者**: 如果您在某个主题上具有专业知识，并且您有兴趣撰写或为书籍做出贡献，请访问[authors.packtpub.com](http://authors.packtpub.com/)。

# 评论

请留下您的评价。一旦您阅读并使用过这本书，为何不在购买它的网站上留下评价呢？潜在读者可以查看并使用您的客观意见来做出购买决定，我们 Packt 公司可以了解您对我们产品的看法，并且我们的作者可以查看他们对书籍的反馈。谢谢！

如需了解更多关于 Packt 的信息，请访问 [packt.com](http://www.packt.com/)。
