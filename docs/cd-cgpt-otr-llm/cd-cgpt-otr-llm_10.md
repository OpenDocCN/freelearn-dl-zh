

# 第十章：扩展编码者的 LLM 工具包：超越 LLM

在软件开发这个快速变化的领域，像 OpenAI 的 GPT 系列和 OpenAI o1、Google 的 Gemini 以及 Meta 的 Llama 3 等 LLM，因其在编码任务中的辅助能力而受到了广泛关注。然而，尽管 LLM 是强大的工具，但它们并不是唯一的选择。还有许多非 LLM AI 工具，旨在补充编码过程，提升生产力和效率。本章将探讨这些工具，讨论它们的能力、局限性，以及如何将它们集成到全面的编码工具包中。

本章包含以下主题：

+   代码补全和生成工具

+   **静态代码分析**（**SCA**）和代码审查工具

+   测试和调试工具

# 技术要求

本章所需的工具：

+   访问浏览器以获取这些 AI 代码工具。

+   一台可安装软件的笔记本电脑或台式机。

+   一个用于 Python 的**集成开发环境**（**IDE**），如 Visual Studio、Spyder、IDLE、PyCharm 或 Atom。

+   一款用于 Javascript 的 IDE，如 Visual Studio、Atom 或 Brackets。在线解释器在这里不够用。以下是一些示例：

    [`onecompiler.com/javascript`](https://onecompiler.com/javascript)

    [`jipsen.github.io/js.html`](https://jipsen.github.io/js.html)

+   一款 Java 集成开发环境（IDE），如 IntelliJ IDEA、Eclipse 或 NetBeans。

+   一些代码示例需要在 Unix **操作系统**（**OSs**）上使用 bash（基础 shell）。

+   获取本章的代码，请点击这里：[`github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main/Chapter10`](https://github.com/PacktPublishing/Coding-with-ChatGPT-and-other-LLMs/tree/main/Chapter10)。

现在，让我们学习如何使用非 LLM 工具使代码生成、分析和测试变得更加简单。

# 代码补全和生成工具

代码补全和生成工具旨在帮助开发人员更高效地编写代码。这些工具利用各种技术，包括语法分析、语义理解和**机器学习**（**ML**）算法，在开发人员输入代码时预测并建议代码片段。这些工具的实用性不容小觑，因为它们能简化编码过程，减少错误，并提升整体生产力。在本节中，我们将探讨几款代码补全和生成工具，介绍它们的特点和实际应用示例，帮助你更好地理解它们如何融入到你的编码工作流程中。让我们来看看一些最受欢迎的工具，包括 Eclipse 的内容辅助、PyCharm 的代码补全工具、NetBeans 的代码补全工具以及**Visual Studio Code**（**VS Code**）的 IntelliSense。

总体而言，这些工具帮助开发人员提高代码准确性，加快开发速度，学习新语法和 API，并使代码更具可读性。

## Eclipse 的内容辅助

Eclipse 是一款多功能的 IDE，具有强大的代码补全插件工具 **内容助手**。该工具通过提供相关建议来提高开发者的工作效率，建议会基于你输入的内容，分析代码的当前上下文和前缀，提供关键字、方法、变量等。这不仅加快了编码速度，还减少了错误。

内容助手的一个显著特点是其与 Eclipse IDE 的无缝集成，无需额外安装。内置的特性意味着开发者可以立即受益于它的代码补全能力，无需额外的设置。拥有这样一个随时可用的工具使开发工作变得更加轻松。

内容助手具有高度的可定制性，允许用户根据自己的特定需求定制建议。开发者可以配置他们想要的建议类型，比如方法名称、变量名称或整个代码片段。你可以根据相关性、类型或可访问性来过滤建议。此外，用户还可以调整触发设置，决定内容助手是自动激活还是手动激活。这种灵活性确保了该工具能够适应不同的编码风格和偏好。

建议是基于你刚输入的字母，建议列表会按照以下顺序出现并显示相关性：

+   字段

+   变量

+   方法

+   函数

+   类

+   结构体

+   联合体

+   命名空间

+   枚举

你可以通过按 *Ctrl* + 空格键来触发代码补全，但这一设置是可定制的，当你输入以下内容时，也会触发代码补全：“ **."** ”，“ **->"** ”或“ **::"** 。

你可以为经常编写的代码创建模板；如果你按下 *Ctrl* + 空格键，依赖于作用域，会出现你的模板列表，你可以选择插入你需要的模板。

示例包括 C 或 C++ 中的 **do while** 循环，Java 中的 main 方法：

```py
public static void main(String[] args) {
    ${cursor}
}
```

或者，在 C++/C 中使用 **包含保护**（也叫 **宏保护**、**头文件保护** 或 **文件保护**）：

```py
#ifndef ${header_guard}
#define ${header_guard}
${cursor}
#endif // ${header_guard}
```

这些能防止你意外地多次包含一个库 [Wiki_Include]。

在 Java 中，你会使用 **for 循环**：

```py
for (int i = 0; i < ${size}; i++) {
     ${cursor}
}
```

或者，你会使用 **try-catch 块**。

对于 Python，示例包括 **函数定义**：

```py
def ${functionName}(${parameters}):
${cursor}
```

它们还包括 **类定义**：

```py
class ${className}:
def __init__(self, ${parameters}):
${cursor}
```

[Eclipse_Help, Gemini]

另一个显著的优点是其广泛的语言支持。由于在 Eclipse 环境中，内容助手支持众多编程语言，包括 Java、C++、Python 和 PHP。这使得内容助手成为多语言开发环境中开发者的好工具。无论是在 JavaScript 中开发 Web 应用程序，在 Java 中开发桌面应用程序，还是在 Python 中编写脚本，内容助手都能提供相关建议，加快编码过程。

然而，它也有一些缺点。性能可能是一个问题，尤其是在大型项目中，代码补全建议可能比专用工具更慢。这种延迟可能打断编码流程并降低生产力。而且，虽然内容辅助提供有用的建议，但它们可能并不总是像专门工具或由机器学习模型驱动的工具那样上下文相关或先进。

尽管存在这些限制，Eclipse 的内容辅助仍然是许多开发者的宝贵工具。它与 Eclipse IDE 的集成，再加上可定制选项和广泛的语言支持，使其成为多种编码任务的实用选择。对于那些优先考虑支持多语言的 IDE 的开发者来说，内容辅助提供了一种便利与功能兼具的平衡。

以下是使用内容辅助的一些优点：

+   不需要额外安装。

+   它提供了多种设置，以便根据你的需求调整建议。

+   它支持广泛的编程语言。

+   它直接集成在 Eclipse IDE 中。

+   它可以配置为自动或手动激活。

以下是使用内容辅助的一些缺点：

+   在大型项目中，它有时可能比专用的代码补全工具更慢。

+   与专门的工具或由机器学习模型驱动的工具相比，它可能并不总是提供最具上下文相关性的建议。

总结来说，Eclipse 的内容辅助是一个对开发者来说非常有价值的工具，适合寻找集成化、可定制且多功能的代码补全功能的开发者。尽管它可能并不总是与专门工具的性能或高级功能相匹配，但其内置特性和广泛的语言支持使其成为许多编码环境中的一个可靠选择。无论你是经验丰富的开发者还是刚刚起步，内容辅助都能帮助简化你的编码过程，提升整体生产力。

有关内容辅助的更详细信息，你可以参考 Eclipse 帮助文档：[`help.eclipse.org/latest/index.jsp?topic=%2Forg.eclipse.cdt.doc.user%2Fconcepts%2Fcdt_c_content_assist.htm`](https://help.eclipse.org/latest/index.jsp?topic=%2Forg.eclipse.cdt.doc.user%2Fconcepts%2Fcdt_c_content_assist.htm)。

## PyCharm 的代码补全。

**PyCharm**，由 JetBrains 开发，是一款广受好评的专门为 Python 开发设计的 IDE。其突出特点之一是智能代码补全，它通过提供超越基本代码补全的上下文感知建议，显著提升了你的编码体验。该功能可以执行静态分析，并使用机器学习提供高度相关的推荐，包括方法调用、变量名称和代码片段。

PyCharm 的代码补全工具与 Python 深度集成，是 Python 开发者的绝佳选择。IDE 在启动时会对整个项目进行索引，从而在你输入时提供准确且具有上下文相关性的建议。这种深度集成确保了代码补全功能能够理解 Python 语法和语义的细微差别，提供具有正确语法且适用于当前上下文的建议。

PyCharm 的代码补全工具的一个关键优势是其上下文感知特性。该工具在你编写代码时会分析当前上下文，提供最相关的建议。例如，如果你在一个类的方法中，它会优先显示该作用域内可访问的方法名称和变量。此上下文感知能力扩展到理解变量的类型，并建议适用于这些类型的方法和属性。这种智能行为有助于减少你需要编写的样板代码，并最小化错误的发生几率。

除了基本的代码补全功能，PyCharm 还提供智能类型匹配补全。此功能会筛选建议列表，仅显示适用于当前上下文的类型。例如，如果你正在为一个变量赋值，它只会建议与该变量类型匹配的值。这个智能筛选有助于保持类型安全，并确保你的代码符合预期的类型约束。

下面是一些使用代码补全的示例。

在 PyCharm 中开始输入一个名称，然后按 *Ctrl* + 空格键。或者，你也可以进入菜单选择 **Code** | **Code Completion**，然后选择 **Basic** 版本。对于 **Basic Completion**，你可以进行方法、方法参数、字典、Django 模板和文件路径的补全 [Python]。

第二次按 *Ctrl* + 空格键或 *Ctrl* + *Alt* + 空格键，你将得到与你输入的字母开头相同的类、函数、模块和变量的名称。

其他类型的代码补全包括智能补全、层级补全、链式补全、文档字符串补全、自定义补全、实时模板补全、后缀补全和类型提示补全 [Jetbrains_Completion]。

智能类型匹配补全会根据当前上下文（光标周围）提供相关类型的列表。例如，异常类型。如果你按 *Ctrl* + *Shift* + 空格键，它会显示相关类型的列表，或者你也可以使用菜单：**Code** | **Code Completion** | **Type-Matching**。

### 重构

PyCharm 也可以进行大量的重构。它的重构工具与代码补全功能协同工作。这些工具允许你轻松地重命名变量、提取方法以及执行其他重构操作。重构工具是基于上下文的，确保变更在你的代码库中得以传播，保持一致性并减少引入错误的风险。重构工具与代码补全的深度集成，使 PyCharm 成为一个非常有用的 IDE，用于维护和提升代码质量。

PyCharm 包含一些组件：项目视图、结构工具窗口、编辑器和 UML 类图。要开始进行重构，有两种方法：

1.  将鼠标悬停在一些代码上。

1.  选择代码，然后从菜单中选择**重构** | **重构此项**，或者按*Ctrl* + *Alt* + *Shift* + *T*。

1.  然后，选择你想要的重构选项。

1.  你会看到一个对话框，你可以在其中输入重构选项，然后点击**确定**或**重构** [Jetbrains_refactoring]。

你可以通过代码自动补全做很多很棒的事情，但也有一些缺点。PyCharm 是一个商业工具，虽然它提供了免费的社区版，但许多高级功能，包括一些代码补全功能，只在专业版中提供，专业版需要购买许可证。这对预算有限的开发者或组织来说可能是一个限制。你可以在这里看到不同版本之间的差异，并查看定价链接：[`www.jetbrains.com/pycharm/editions/`](https://www.jetbrains.com/pycharm/editions/)。

另一个需要考虑的因素是，PyCharm 可能会消耗大量资源，尤其是在处理大型项目时。IDE 的全面索引和分析功能需要相当多的计算资源，这可能会导致在性能较差的机器上运行缓慢。这种资源密集型使用有时会导致代码补全建议的延迟，从而打断编码的流畅性。

尽管存在这些缺点，PyCharm 的代码补全仍然是 Python 开发者非常有价值的工具。它智能的、基于上下文的建议、与 Python 的深度集成以及强大的重构工具，使其成为初学者和经验丰富的开发者的优选。代码补全行为的可定制性意味着你可以定义它如何帮助检查和完成代码：你可以根据自己的具体偏好来调整这个工具。

下面是使用 PyCharm 代码自动补全工具的一些优点：

+   它根据你代码的上下文提供高度相关的代码补全。

+   它提供了无缝的 Python 开发体验。

+   它包括优秀的重构功能，可以提高代码质量。

下面是使用 PyCharm 代码自动补全工具的一些缺点：

+   完全使用需要付费许可证。

+   它可能会消耗大量资源，特别是在大型项目中。

总结来说，PyCharm 的代码自动完成是一个强大而智能的功能，大大提升了 Python 开发体验。它的上下文感知建议、与 Python 的深度集成以及强大的重构工具，使其成为 Python 开发者的极好选择。尽管完全使用该功能需要许可证，并且可能对资源要求较高，但它在提升生产力和代码质量方面的好处，使其对许多开发人员来说是值得投资的。

你可以在这里了解更多关于 PyCharm 代码自动完成功能的信息：[`www.jetbrains.com/help/pycharm/auto-completing-code.html`](https://www.jetbrains.com/help/pycharm/auto-completing-code.html)。

## NetBeans 的代码自动完成

NetBeans 提供了完整的代码自动完成功能。此功能旨在通过提供与关键字、方法、变量等相关的建议，提升编程体验，支持多种编程语言。NetBeans 的代码自动完成功能是提高生产力和减少编码错误的宝贵工具。

NetBeans 的代码自动完成工具非常用户友好。其设计使得开发人员无论经验水平如何，都能轻松使用。代码自动完成功能集成在编辑器中，允许在你输入时自动出现建议。这种集成有助于简化编码过程，使其更加快捷高效。界面设计简单直观，即使是新用户也能快速熟悉该工具。

NetBeans 的编辑器代码自动完成 API 有两个类，**CompletionItem**和**Completion** **Provider**。代码自动完成同样可以通过*Ctrl* + 空格键或通过菜单激活：在 Windows 中，这个路径是：**工具** | **选项** | **编辑器** | **代码自动完成**；在 macOS 中，这个路径是：**NetBeans** | **偏好设置…** | **编辑器** | **代码自动完成**。与其他工具类似，它会根据你所写的内容生成一个建议列表，随着你输入，列表会缩短，最相关的建议会排在列表的顶部。你可以指定代码自动完成的触发方式，但默认情况下是这样的。

还有一种叫做**hippie completion**的代码自动完成版本，它会搜索当前代码的作用域，首先搜索当前文档，如果没有找到期望的结果，则继续搜索其他文档。Windows 系统中使用*Ctrl* + *K*，在 macOS 系统中使用*cmd* + *K*来激活该功能。

如果你正在声明一个对象或变量类型，按*Ctrl* + 空格键将建议该类型的对象，例如，如果你声明一个**int**，第一次按*Ctrl* + 空格键时，它会给你所有 int 类型的选项。如果你再次使用相同的快捷键激活代码自动完成，它会建议所有项，而不仅仅是 int 类型。

如果你使用*Tab*键，该工具将填充最常用的前缀和建议，例如，**print**；如果你输入**System.out.p**，将出现以下内容：

```py
print(Object obj)                                     void
print(String s)                                       void
print(boolean b)                                      void
print(char c)                                         void
print(char[] s)                                       void
```

使用*Tab*键选择建议项。

NetBeans 的代码自动完成工具还会自动完成子词。这取决于你输入的字母，但它会关联到这些字母相关的所有内容，而不仅仅是以这些字母开头的内容。因此，即使你忘记了项目的首字母，它仍然可以正常工作！举个例子，如果你输入 **Binding.prop**，它会提供以下建议：

```py
addPropertyChangeListener (PropertyChangeLi..
addPropertyChangeListener (String propertyN…
getPropertyChangeListeners ()
getPropertyChangeListener (String property..
etc.
```

当你需要一系列命令时，按 *Ctrl* + 空格键两次，所有可用的命令链都会显示出来。它会查找变量、字段和方法。

比如，当你输入字符串 **bindName =** 时，自动完成工具可以显示以下内容：

```py
binding.toString()                                   String
clone().toString()                                   String
getClass().getCanonicalName()                        String
getClass().getName()                                 String
```

NetBeans 的代码自动完成工具还以更多方式帮助你编码。

了解如何在 HTML 文件的上下文中实现编辑器代码自动完成 API 的教程，请访问：[`netbeans.apache.org/tutorial/main/tutorials/nbm-code-completion/`](https://netbeans.apache.org/tutorial/main/tutorials/nbm-code-completion/) [ Netbeans_Completion, Netbeans_SmartCode]。

然而，NetBeans 的代码自动完成工具也存在一些缺点。在处理较大项目时，性能有时会成为问题。与专门设计用于代码自动完成的工具相比，NetBeans 的代码自动完成功能可能会较慢。这在处理大型代码库或项目及其所有依赖项时尤为明显。建议的延迟可能会打乱编码流畅性，降低整体生产力。此外，尽管 NetBeans 提供了一整套功能，但它可能不像一些商业 IDE 那样提供那么多高级功能。例如，商业工具通常拥有机器学习模型，能够提供更复杂和上下文感知的代码自动完成。

这些高级工具能够提供更准确的建议，尤其是在复杂的编码场景中。虽然 NetBeans 非常强大，但可能缺少一些这些前沿功能，这对寻找最先进工具的开发者来说是一个限制。

尽管存在这些局限性，NetBeans 的代码自动完成仍然是许多开发者高度重视的工具。其用户友好的界面、跨平台兼容性和开源特性，使其成为广泛用户的有吸引力选择。能够自定义代码自动完成的行为，使它对于你的编码更加有用，你可以像使用 PyCharm 的工具一样定制它 [NetBeans_Completion]。

使用 NetBeans 代码自动完成功能的一些优点如下：

+   它具有直观的界面，易于使用

+   它支持 Windows、OSX 和 Linux 操作系统

+   它是免费的并且是开源的，使得所有开发者都能使用

使用 NetBeans 代码自动完成功能的一些缺点如下：

+   它有时会比更专业的代码自动完成工具慢

+   它可能不像一些商业 IDE 那样提供那么多高级功能

总结来说，NetBeans 的代码补全功能是一个强大且多功能的工具，可以极大地帮助开发体验。其直观的界面、跨平台支持以及开源可访问性，使其成为开发者的优秀选择。虽然它可能无法与专门的商业工具在性能或高级功能上相媲美，但其全面的功能集和易用性使其成为许多编码环境中的坚实选择。因此，无论你是经验丰富的开发者还是新手，NetBeans 的代码补全都能帮助简化你的编码过程，提高你的工作效率。让工具帮你完成繁重的工作。人们常说，懒惰的程序员是最好的程序员。制作并获取能帮你提高效率的工具，不必事事亲力亲为。

了解更多关于 NetBeans 代码补全工具的信息，请访问：[`netbeans.apache.org/tutorial/main/tutorials/nbm-code-completion/`](https://netbeans.apache.org/tutorial/main/tutorials/nbm-code-completion/)。

## VS Code 的 IntelliSense

VS Code 是由 Microsoft 提供的，其中一个最佳特性就是 IntelliSense。IntelliSense 会分析代码的上下文，提供相关的建议，包括方法调用、变量和关键字，从而加快编码速度并减少出错的可能性。

IntelliSense 的主要优点之一是它的轻量化特性。尽管功能强大，VS Code 仍然保持着快速和高效的性能，即使在处理大型项目时也不例外。这种性能效率对于需要一个响应迅速且可靠的编码环境的开发者来说至关重要。轻量化的设计确保 IntelliSense 能够提供实时建议，而不会导致显著的延迟或性能问题，即使是在庞大的代码库中也是如此。

与 PyCharm 和 NetBeans 的工具类似，IntelliSense 也是高度可定制的，提供广泛的选项来根据你的特定需求调整工具的设置。开发者可以配置 IntelliSense 的各个方面，例如它提供的建议类型以及触发这些建议显示的条件。这种定制化允许你创建一个与你的工作流和偏好相匹配的编码环境。

VS Code 还允许用户定义代码片段，使你能够创建并使用自定义代码模板，这些模板可以迅速插入到你的代码中。这项功能现在已经在许多集成开发环境（IDE）中提供，但它仍然是一个非常实用的工具。

Visual Studio Marketplace 提供了大量插件，可以扩展 IntelliSense 和整个编辑器的功能。这些插件覆盖了各种编程语言、框架和工具，允许开发者通过附加的功能和能力来增强他们的编码环境。无论你需要某个特定语言的支持，还是需要与版本控制系统集成，或是需要调试和测试的工具，市场上都能找到满足你需求的插件。

让我们来看一下它的负面影响。

需要考虑的一点是，尽管 IntelliSense 提供了良好的代码补全功能，但与一些专门的集成开发环境（IDE）相比，它在某些编程语言或框架上可能不够专业。例如，专为 Java 开发设计的 IDE 可能会为 Java 代码提供比 VS Code 更加高级和具备上下文感知的建议。然而，丰富的插件列表通过允许你添加特定语言的扩展，帮助缓解这一限制，增强了 IntelliSense 在你偏好的语言和框架上的能力。

使用 VS Code IntelliSense 的一些优点如下：

+   **轻量级** : 即使是大型项目，IntelliSense 也快速高效。

+   **高度可定制** : 它提供了广泛的定制选项。

+   **.NET 支持** : IntelliSense 对 .NET 语言（如 C#、F# 和 VB.NET）提供了强大的支持，并且在这些语言上表现出色。

使用 VS Code IntelliSense 的一些缺点如下：

+   **变量性能** : IntelliSense 的准确性和完整性取决于语言和项目的设置。较少使用的语言和更复杂的项目可能会导致 IntelliSense 性能不佳。

+   **潜在的性能开销** : 在复杂的项目中，IntelliSense 可能会消耗比某些替代工具更多的系统资源。

总结来说，VS Code 的 IntelliSense 对开发者来说是一个非常有用的工具。它的轻量级设计、高度的可定制性，以及大量插件使得它成为许多不同编码任务的宝贵选择。你可以根据自己的工作方式定制 IntelliSense，使其以适合你工作风格的方式帮助你。

这里是关于 VS Code IntelliSense 的更多信息：[`code.visualstudio.com/docs/editor/intellisense`](https://code.visualstudio.com/docs/editor/intellisense)。

现在，让我们来看看 SCA 和代码审查工具。

SCA 工具帮助你在运行代码之前发现问题，分析大型代码库，并自动执行常规检查。

代码审查工具帮助经验丰富的编码人员提出改进代码的建议，帮助团队更好地协作，并且考虑代码的更广泛环境。

# SCA 和代码审查工具

SCA 和代码审查工具在现代软件开发中已变得不可或缺，它们在确保代码质量和可靠性方面起着关键作用，尤其是在代码执行之前。这些工具会细致地分析源代码，找出潜在的错误、安全漏洞、风格不一致和其他可能影响软件的问题。通过在开发过程早期捕捉这些问题，SCA 工具帮助保持高标准的软件质量，减少生产中的缺陷风险，最终节省时间和资源。

SCA 工具的主要优点之一是它们能够为开发者提供即时反馈。随着代码的编写，这些工具会扫描源代码并突出潜在问题，允许开发者立刻解决。这种实时反馈循环对于维持代码质量并确保始终遵循最佳实践至关重要。此外，SCA 工具通常与流行的 IDE 和 CI/CD 管道无缝集成，使其成为开发工作流的核心部分。

有多种知名的 SCA 工具可供选择，每种工具都有自己的一套功能和能力。例如，SonarQube 是一个广泛使用的工具，支持多种编程语言，并提供关于代码质量、安全漏洞和技术债务的全面报告。它深入分析代码问题，并提出可能的修复建议，帮助开发者随着时间的推移改进代码库。另一个受欢迎的工具是 ESLint，它专为 JavaScript 和 TypeScript 设计。ESLint 允许开发者强制执行编码标准并捕捉常见错误，是前端开发中必不可少的工具。

## SonarQube

**SonarQube** 是一个广泛使用的静态分析工具，支持多种编程语言，包括 Java、C# 和 JavaScript。它提供了一个持续检查代码质量的平台，使团队能够发现错误、漏洞和代码异味。这个工具用于维持高标准的软件质量，并确保代码在投入生产之前是可靠和安全的。

SonarQube 可以检测代码库中的错误和漏洞。通过扫描源代码，SonarQube 能发现可能导致运行时错误或安全漏洞的问题。这种主动的做法使开发者能够在开发过程的早期解决问题，从而减少日后修复的成本。该工具提供了关于错误和漏洞的详细信息，帮助开发者理解根本原因并考虑最佳解决方案。

SonarQube 还提供了许多有助于维持高质量代码的代码度量指标。这些指标包括代码覆盖率、复杂度和重复率。代码覆盖率衡量了自动化测试对代码库的测试程度，从而反映测试过程的健壮性。复杂度指标有助于识别过于复杂的代码，这类代码可能难以维护或容易出错。重复率则突出了代码库中重复出现的相似代码，提示可以进行重构以提高可维护性并减少技术债务。

SonarQube 与 CI/CD 流水线的集成是另一个显著的优势。通过将 SonarQube 集成到 CI/CD 流程中，团队可以确保代码质量检查成为开发工作流的一个不可或缺的部分。这种集成使得每次提交代码时都能自动进行代码分析，提供即时反馈给开发者，并防止在代码库中引入新的问题。与流行的 CI/CD 工具（如 Jenkins、Azure DevOps 和 GitLab）集成，使得将 SonarQube 融入现有开发流程变得容易。

使用 SonarQube 很简单。要分析一个项目，开发者可以使用 SonarQube 扫描器，这是一个命令行工具，它将代码发送到 SonarQube 服务器进行分析。例如，要分析一个 Java 项目，你可以按以下步骤操作：

1.  首先，在以下链接下载：[`www.sonarsource.com/products/sonarqube/`](https://www.sonarsource.com/products/sonarqube/)。

1.  使用以下链接安装 SonarScanner：[`docs.sonarsource.com/sonarqube/9.7/analyzing-source-code/scanners/sonarscanner/`](https://docs.sonarsource.com/sonarqube/9.7/analyzing-source-code/scanners/sonarscanner/)。

1.  通过在 Java 项目的根目录下创建一个**sonar-project.properties**文件来配置 SonarScanner：

    1.  指定项目密钥、名称和源代码目录：

    ```py
    Properties
    sonar.projectKey=my-java-project
    sonar.projectName=My Java Project
    sonar.sources=src/main/java
    ```

1.  然后，你可以运行以下命令来分析项目：

    ```py
    sonar-scanner -Dsonar.projectKey=my_project -Dsonar.sources=.
    ```

此命令指定了项目密钥和要分析的源代码目录。分析结果随后会显示在 SonarQube 仪表盘上，开发者可以查看结果并采取适当的措施。

尽管 SonarQube 有很多优点，但也存在一些挑战。一个潜在的负面因素是误报的可能性，即工具标记了一个实际上并不是问题的情况。这可能导致开发者进行不必要的工作，并可能引发挫败感。此外，尽管 SonarQube 做了相当全面的分析，但它可能无法捕捉到所有类型的问题，尤其是与代码运行时行为相关的问题。因此，重要的是将静态分析与其他测试方法（如单元测试和集成测试）结合使用，以确保全面覆盖。

使用 SonarQube 的一些优点如下：

+   SonarQube 支持多种编程语言，使其能够在各种项目中提供帮助。

+   它提供关于代码质量的详细报告，帮助团队有效地优先处理问题。

+   强大的社区和各种插件增强了 SonarQube 的功能。

+   提供了免费版本和付费版本以获得更多功能。

使用 SonarQube 的一些缺点如下：

+   SonarQube 有时会产生误报，开发者应仔细审查这些问题。

+   运行 SonarQube 可能会消耗较多资源，特别是对于大型代码库，可能需要专门的基础设施。

总之，SonarQube 是一个非常有用的静态代码分析（SCA）工具，能够持续检查代码质量。它检测错误、漏洞和代码异味的能力，再加上全面的代码指标和与 CI/CD 流水线的无缝集成，使其成为开发团队不可或缺的资产。通过将 SonarQube 融入开发流程，团队可以保持高标准的软件质量，减少缺陷风险，并交付可靠且安全的软件解决方案。

更多信息请访问 SonarQube 官方网站：[`www.sonarsource.com/products/sonarqube/`](https://www.sonarsource.com/products/sonarqube/)。

## ESLint

**ESLint** 是专门为 JavaScript 和 JSX 设计的静态分析工具。（JSX 是一种类似 XML 的 JavaScript 扩展。）ESLint 在现代 Web 开发中扮演着重要角色，帮助开发人员遵循编码标准并识别代码中的问题模式。这个工具有助于在项目中保持代码质量和一致性，使它成为开发人员在处理 JavaScript 时的最爱。

与代码自动补全工具类似，ESLint 具有可定制的规则。ESLint 允许用户定义自己的规则并在团队间共享配置，从而促进编码实践的一致性。这种灵活性确保团队可以遵循特定的编码标准，无论项目的要求如何。例如，你可以创建规则，强制使用单引号来表示字符串，或要求语句末尾必须有分号。你还可以指定缩进样式、变量命名约定、函数长度，甚至代码复杂度。这种定制化程度有助于维持统一的代码风格，特别适用于大型团队或开源项目，其中涉及多个贡献者。

ESLint 另一个特点是它能够轻松与 CI/CD 流水线集成。通过将 ESLint 集成到 CI/CD 流程中，开发人员可以确保在部署之前自动进行代码质量检查。这种集成有助于在开发周期的早期发现问题，从而减少了缺陷和不一致性进入生产环境的风险。流行的 CI/CD 工具，如 Jenkins、Travis CI 和 GitHub Actions，都支持 ESLint，使其能够轻松融入现有的工作流。

ESLint 还提供了强大的修复功能。ESLint 检测到的许多问题可以自动修复，从而节省开发人员的时间和精力。例如，如果 ESLint 检测到缺少分号或引号样式不正确，它可以根据定义的规则自动纠正这些问题。这个自动修复功能尤其适用于解决小的代码风格违规，允许开发人员将精力集中在更复杂的任务上。

使用 ESLint 非常简单。开发人员可以通过创建一个配置文件（通常命名为**.eslintrc.js**），定义环境、扩展配置和指定规则来配置 ESLint。以下是 ESLint 配置的示例：

```py
module.exports = {
    "env": {
        "browser": true,
        "es6": true
    },
    "extends": "eslint:recommended",
    "rules": {
        "semi": ["error", "always"],
        "quotes": ["error", "single"]
    }
};
```

+   **env**：这指定了代码将要运行的环境。在这种情况下，它配置了浏览器和 ES6 环境。

+   **extends**：此设置扩展了推荐的 ESLint 规则集，为执行常见编码标准提供了一个良好的起点。

+   **rules**：这一部分允许您定制特定的规则。在这里，您已经强制使用了分号和单引号。

尽管 ESLint 有许多优点，但也存在一些问题。一个可能的挑战是初始设置。配置 ESLint 以适应团队特定需求需要大量时间，尤其是对于可能觉得广泛的配置选项令人不知所措的新用户来说。然而，一旦设置完成，ESLint 在代码质量和一致性方面提供了显著的长期利益。

另一件需要考虑的事情是与 ESLint 相关联的学习曲线。虽然这个工具高度可配置，但新用户可能需要时间来熟悉其功能及如何有效地进行定制。幸运的是，ESLint 有一个活跃的社区，为新用户提供了丰富的插件和共享配置，提供了充足的资源和支持。

以下是使用 ESLint 的一些优点：

+   **高度可配置**：ESLint 的灵活性允许团队有效地执行其编码标准。

+   **活跃社区**：庞大的社区为插件和共享配置提供了丰富的贡献。

以下是使用 ESLint 的一些缺点：

+   **Setup time**：ESLint 需要一开始投入时间来配置符合团队需求的规则。

+   **False positives**：与本章中的其他一些工具类似，ESLint 有时可能会标记一些并非实际错误或违反编码标准的问题；这些是误报。这可能令人沮丧，并意味着您必须为无效的代码变更做出额外的更改。

+   ESLint 可能会有一些限制，强制执行严格的规则，这些规则并不总是适合某些代码风格或使用情况。

总之，ESLint 对 JavaScript 和 JSX 开发非常有帮助。它的可定制规则、与 CI/CD 流水线的集成以及自动修复能力使其成为坚持高代码质量和一致性的强大工具。虽然配置可能需要一开始投入时间和新用户需要一定的学习曲线，但它在提高生产力和代码可靠性方面提供的优势绝对是值得的。无论您是在开发小项目还是大型应用程序，ESLint 都可以帮助确保您的代码遵循最佳实践并且随着时间的推移易于维护。正如您在之前章节中可能已经看到的，代码标准可能非常严格和详细，因此拥有工具来保持这些标准可以提供很大的帮助。

更多信息可以在 ESLint 的官方网站找到：[`eslint.org/`](https://eslint.org/)。

## PMD

**PMD**是一个开源的 SCA 工具，对于开发者识别代码中的潜在问题非常有帮助。它支持几种编程语言，包括 Java、JavaScript 和 XML（并对 C、C++、C#、Python 和 PHP 有一些支持）。PMD 主要用于 Java，专注于发现常见的编程缺陷，如未使用的变量、空的 catch 块和不必要的对象创建。这使得 PMD 成为一个极好的工具，用于保持高标准的代码质量，确保软件的可靠性和效率，尤其是在 Java 开发中。

PMD 的一个特点是基于规则的分析。PMD 使用一组预定义规则来分析代码并识别潜在问题。这些规则涵盖了各种常见的编程错误和最佳实践，帮助开发者在开发过程中尽早发现错误。预定义的规则非常全面，涵盖了从语法错误到更复杂的逻辑问题等多个方面。这种彻底的分析有助于保持代码库的清晰和高效。

除了预定义的规则外，PMD 还允许开发者创建自定义规则，以满足你期望的编码标准和实践。这种可定制性是 PMD 的最佳特点之一，因为它使团队能够强制执行独特的编码指南，并确保项目之间的一致性。自定义规则可以使用 Java 编写或通过 XPath 查询定义，这为定义和实现规则提供了灵活性。根据具体需求调整工具的能力使 PMD 能够高度适应不同的开发环境和需求。

PMD 还与流行的构建工具集成，如 Maven 和 Gradle（PMD Maven 插件可在[`github.com/apache/maven-pmd-plugin`](https://github.com/apache/maven-pmd-plugin)找到，PMD Gradle 插件可在[`docs.gradle.org/current/userguide/pmd_plugin.html`](https://docs.gradle.org/current/userguide/pmd_plugin.html)找到），使其可以轻松融入现有的工作流中。这种集成确保了代码质量检查成为构建过程的一个组成部分，为开发者提供持续反馈，并防止将新问题引入代码库。通过将 PMD 作为构建过程的一部分运行，团队可以早期捕获并解决问题，降低生产环境中缺陷的风险。例如，要在使用 Maven 的 Java 项目上运行 PMD，你可以使用以下命令：

```py
mvn pmd:pmd
```

在 bash 中运行此命令。此命令触发 PMD 分析项目并生成关于发现问题的报告，允许开发者及时审查并解决这些问题。

尽管 PMD 有许多优点，但也存在一些局限性。一个潜在的问题是它有限的语言支持。虽然 PMD 支持几种语言，但它的主要关注点是 Java，这可能不适合所有项目。这对于使用多种编程语言的团队来说可能是一个限制。然而，PMD 也支持其他语言，如 JavaScript、Salesforce Apex 等，这对于许多项目仍然有很大的帮助。

另一个需要考虑的因素是为 PMD 配置自定义规则的复杂性。设置 PMD 来强制执行特定的编码标准可能比较复杂，特别是对于那些不熟悉该工具配置选项的新用户来说。这与本章中的其他一些工具类似。这一初步设置需要投入时间和精力，但长期来看，拥有一个量身定制的静态分析工具是非常值得的。你会发现，这一度过的时间投资真的为每天的反复帮助带来了很大的回报。

以下是使用 PMD 的一些优点：

+   PMD 是免费的，因此适用于各种规模的开发人员和团队

+   创建自定义规则的能力使团队能够强制执行特定的编码标准

以下是使用 PMD 的一些缺点：

+   尽管 PMD 支持多种语言，但它的主要关注点是 Java，这可能不适用于所有项目

+   为 PMD 配置自定义规则对新用户来说可能比较复杂

总结来说，PMD 是一个适用于 Java 和其他语言的 SCA 工具，帮助开发人员保持高标准的代码质量。它基于规则的分析、可定制性以及与构建工具的集成使它成为许多开发团队的良好工具。虽然它在语言支持和配置复杂性方面可能存在一些限制，但在发现潜在问题和执行编码标准方面所带来的好处，使其成为软件开发过程中宝贵的工具。通过将 PMD 融入团队的工作流程，您可以确保代码可靠、高效，并符合最佳实践，最终带来更高质量的软件。

更多信息可以在 PMD 的网站找到：[`pmd.github.io/`](https://pmd.github.io/)。

## Java 的 Checkstyle

由于我们已经讲解了一些工具，我会简要说明。

**Checkstyle** 是一个非常有价值的工具，确保 Java 代码遵循预定义的编码标准。通过自动化代码检查，它保持项目的一致性和质量，使代码更容易阅读、理解和维护。

下面是它的一些关键特性：

+   它根据可定制的规则检查代码，包括命名规范、格式化、设计模式等

+   它与 Eclipse 和 IntelliJ IDEA 无缝集成，提供实时反馈

+   它允许你定义自己的规则，以匹配特定的编码实践

它的工作原理如下：

1.  在构建工具中配置 Checkstyle（例如 Maven、Gradle），以指定所需的规则集。

1.  Checkstyle 会根据这些规则分析你的 Java 代码。

1.  它生成报告，突出显示违规项，并提供改进建议。

这是一个示例（Maven）。

这是 XML 代码：

```py
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-pmd-plugin</artifactId>
  <version>3.19.0</version>
  <configuration>
    </configuration>
</plugin>
```

下面是使用 Checkstyle 的一些优点：

+   **专注于 Java**：Checkstyle 专门为 Java 设计，使其在 Java 项目中非常有效

+   **实时反馈**：与 IDE 的集成使开发人员能够立即获得编码标准的反馈

下面是使用 Checkstyle 的一些缺点：

+   **仅限 Java**：Checkstyle 仅限于 Java，可能不适用于多语言项目

+   **配置开销**：设置和维护 Checkstyle 配置可能会耗费大量时间

通过使用 Checkstyle，您可以确保您的 Java 代码符合高质量标准，促进可读性、可维护性和团队协作。

**来源**：[`github.com/jvalentino/jenkins-agent-maven`](https://github.com/jvalentino/jenkins-agent-maven)。

您可以在这里找到更多信息：[`checkstyle.sourceforge.io/`](https://checkstyle.sourceforge.io/)。

## Fortify 静态代码分析器

**Fortify**，由 OpenText 开发，是一个商业静态分析工具，擅长识别源代码中的安全漏洞。它支持多种编程语言，是多元化开发团队的理想选择。

下面是 Fortify 的一些关键特性：

+   Fortify 的深度分析能力揭示了广泛的安全风险

+   您可以生成具有修复指导的可操作报告，有效解决已识别的漏洞

+   您可以将 Fortify 集成到 CI/CD 流水线中，进行持续的安全检查

这是它的工作方式：

1.  在代码上运行 Fortify，以识别漏洞。

1.  查看详细报告，获取见解和修复指导。

1.  将 Fortify 集成到您的开发工作流中，以便进行持续的安全检查。

下面是使用 Fortify 的一些优点：

+   它非常适合大型项目和组织

+   它能够检测各种漏洞

+   它提供了清晰的修复指导

下面是使用 Fortify 的一些缺点：

+   它需要商业许可证

+   初始设置和配置可能比较复杂

总之，Fortify 是一个对那些希望加强软件安全性的组织非常有价值的工具。尽管它需要一定的投资，但在漏洞检测和风险缓解方面的收益是巨大的。通过将 Fortify 纳入您的开发过程，您可以确保您的代码符合最高的安全标准。

Fortify 的官方网站提供更多信息：[`www.microfocus.com/en-us/products/static-code-analysis-sast/overview`](https://www.microfocus.com/en-us/products/static-code-analysis-sast/overview)。

下面是更多来源：

+   [`www.microfocus.com/documentation/fortify-static-code-analyzer-and-tools/2310/`](https://www.microfocus.com/documentation/fortify-static-code-analyzer-and-tools/2310/)

+   [`www.opentext.com/products/fortify-static-code-analyzer`](https://www.opentext.com/products/fortify-static-code-analyzer)

+   [`bing.com/search?q=Fortify+Static+Code+Analyzer+summary`](https://bing.com/search?q=Fortify+Static+Code+Analyzer+summary)

+   [`www.microfocus.com/media/data-sheet/fortify_static_code_analyzer_static_application_security_testing_ds.pdf`](https://www.microfocus.com/media/data-sheet/fortify_static_code_analyzer_static_application_security_testing_ds.pdf)

+   [`gemini.google.com/`](https://gemini.google.com/)

+   [`copilot.microsoft.com/`](https://copilot.microsoft.com/)

## CodeSonar

**CodeSonar**，由 GrammaTech 提供的静态分析工具，也被称为**静态应用程序安全测试**（**SAST**）工具，是确保代码质量和安全的优秀工具，尤其在汽车、航空航天和医疗设备等关键行业中。

以下是它的一些关键特性：

+   它能发现广泛的问题，从内存泄漏到并发问题

+   它与流行的 IDE 和构建系统配合良好

+   它帮助你通过可视化工具理解复杂的代码结构

其工作原理如下：

1.  CodeSonar 扫描你的代码，查找潜在问题。

1.  详细报告突出显示漏洞并提供洞察。

1.  轻松将 CodeSonar 集成到你的开发工作流中。

以下是使用 CodeSonar 的优点：

+   它在问题变得代价高昂之前就能捕捉到错误

+   它确保代码质量和安全性

+   它理解你的代码结构和依赖关系

以下是使用 CodeSonar 的缺点：

+   它可能需要许可证

+   它可能需要一些时间来掌握其功能

+   初始设置可能需要一些努力

总之，CodeSonar 是维护代码质量和安全的宝贵工具。尽管它可能有一定的学习曲线并需要投资，但它在防止代价高昂的错误和确保可靠软件方面所提供的好处，使得它成为许多开发团队值得考虑的工具。

了解更多信息，请访问 CodeSonar 网站：[`codesecure.com/our-products/codesonar/`](https://codesecure.com/our-products/codesonar/)。

## Coverity

**Coverity**，由 Synopsys 提供，是一款静态分析工具，它帮助企业环境中的开发团队在软件开发生命周期的早期识别并修复缺陷。

以下是它的主要功能：

+   Coverity 扫描代码，查找各种问题，提供详细的报告和修复建议

+   它与流行的 CI/CD 工具无缝集成，进行自动化的代码质量检查

+   你可以通过可定制的仪表板监控代码健康指标，并跟踪进展

以下是使用 Coverity 的优点：

+   它主动检测缺陷和漏洞，增强代码健康和安全性

+   它早期发现问题，减少后期开发过程中代价高昂的漏洞修复

+   你可以通过可定制的仪表板了解代码质量

以下是使用 Coverity 的缺点：

+   它需要商业许可证，对于小团队或个人开发者来说，可能不太可行

+   初始设置可能比较复杂，并需要专门的资源

+   丰富的功能和配置选项对新用户来说有一定的学习曲线

总结来说，对于大型项目，Coverity 的全面缺陷检测、CI/CD 集成和可定制的仪表盘使其成为维护高质量和高安全性代码的宝贵工具。尽管它有一定的费用和学习曲线，但对于许多开发团队来说，其长期利益远远超过这些成本。

流行的 CI/CD 工具，如 Jenkins（[`www.devopsschool.com/blog/what-is-coverity-and-how-it-works-an-overview-and-its-use-cases/`](https://www.devopsschool.com/blog/what-is-coverity-and-how-it-works-an-overview-and-its-use-cases/)）、GitLab 和 Azure DevOps，支持 Coverity，这使得它可以轻松地集成到现有的开发流程中。

更多信息请参见 Coverity 网站：[`www.synopsys.com/software-integrity/security-testing/static-analysis-sast.html`](https://www.synopsys.com/software-integrity/security-testing/static-analysis-sast.html)。

以下是来源：[`www.devopsschool.com/blog/what-is-coverity-and-how-it-works-an-overview-and-its-use-cases/`](https://www.devopsschool.com/blog/what-is-coverity-and-how-it-works-an-overview-and-its-use-cases/)，[`www.trustradius.com/products/synopsys-coverity-static-application-security-testing-sast/reviews?qs=pros-and-cons`](https://www.trustradius.com/products/synopsys-coverity-static-application-security-testing-sast/reviews?qs=pros-and-cons)，[`www.gartner.com/reviews/market/application-security-testing/vendor/synopsys/product/coverity-static-application-security-testing`](https://www.gartner.com/reviews/market/application-security-testing/vendor/synopsys/product/coverity-static-application-security-testing)，[`stackshare.io/coverity-scan`](https://stackshare.io/coverity-scan)，[`www.softwareadvice.com/app-development/coverity-static-analysis-profile/`](https://www.softwareadvice.com/app-development/coverity-static-analysis-profile/)，[`en.wikipedia.org/wiki/Coverity`](https://en.wikipedia.org/wiki/Coverity)，[`gemini.google.com/`](https://gemini.google.com/)，[`copilot.microsoft.com/`](https://copilot.microsoft.com/)

## FindBugs/SpotBugs

**SpotBugs**，继承了 FindBugs，是一款专门为 Java 设计的静态分析工具，擅长检测 Java 代码中的潜在 bug。通过利用一整套 bug 模式，它能够识别常见的编码错误，从而确保代码质量和可靠性更高。SpotBugs 通过分析 Java 字节码来工作，这使得它能够根据预定义的模式发现潜在问题。这种分析方法特别有效于发现那些通过手动代码审查可能不容易察觉的 bug。

SpotBugs 的一大优势是与流行构建工具（如 Maven 和 Gradle）的良好集成。这种集成使它能轻松地融入现有的开发工作流程，成为 CI/CD 管道中的便捷选择。

SpotBugs 以其用户友好的设置和易用性而著称，适合各种技能水平的开发者。无论你是初学者还是经验丰富的专业人士，SpotBugs 都能提供一个简单的方式来增强代码的健壮性。它能够平滑地集成到各种开发环境中，且易于使用，是维护 Java 项目代码质量高标准的宝贵工具。

例如，在 bash 中运行以下命令：

```py
# To run SpotBugs on a Java project, you would use:
mvn spotbugs:check
```

以下是使用 SpotBugs 的优点：

+   **免费且开源**：SpotBugs 免费使用，开发者和团队可以轻松访问。

+   **专注于 Java**：它在 Java 方面的专长意味着它为 Java 项目提供高度相关的分析。

    以下是使用 SpotBugs 的缺点：

+   **语言支持有限**：SpotBugs 专门用于 Java，这可能会限制其在多语言项目中的可用性。

+   **误报**：像许多静态分析工具一样，它可能会产生误报，需要手动审核。

+   总之，SpotBugs 是一个静态分析工具，专为 Java 开发者设计，提供了一个强大的解决方案来识别和修复潜在的错误。它与构建工具的便捷集成和用户友好的设置，使得各级开发者都能轻松使用，从而确保项目中代码质量的高标准。

+   更多信息请访问 SpotBugs 网站：[`spotbugs.github.io/`](https://spotbugs.github.io/)。

## Bandit

**Bandit** 是一个 SCA，或者说是一个 SAST 工具，帮助你识别 Python 代码中的安全漏洞。它扫描常见问题，如硬编码密码和不安全的 API 使用，确保你的 Python 应用程序保持安全。

其主要特点如下：

+   **全面的漏洞检测**：它可以识别多种安全风险。

+   **可定制规则**：根据你的具体需求定制 Bandit。

+   **CI/CD 集成**：自动化安全检查，实现持续监控。

使用示例，在 bash 中运行以下命令：

```py
# To run Bandit on a Python project, you would use:
bandit -r my_project/
```

以下是使用 Bandit 的优点：

+   **专为 Python 设计**：Bandit 专门为 Python 量身定制，因此在识别 Python 代码中的安全问题方面非常有效。

+   **开源**：作为一个开源工具，Bandit 可以免费使用，任何开发者都可以访问。

以下是使用 Bandit 的缺点：

+   **仅限 Python**：Bandit 专门为 Python 设计，因此可能不适用于使用多种语言的项目。

+   **误报**：像许多静态分析工具一样，Bandit 也可能生成误报，需要手动审核。

总之，Bandit 是针对安全性关注的 Python 开发者的宝贵工具。它能够检测多种安全漏洞，再加上可定制的规则和无缝的 CI/CD 集成，使其成为维护安全代码库的重要资产。通过将 Bandit 纳入开发工作流，开发者可以主动应对安全风险，确保他们的应用程序在潜在威胁面前保持安全。

查看 Bandit 的网站：[`bandit.readthedocs.io/`](https://bandit.readthedocs.io/) 。

## HoundCI

**HoundCI** 是一个与 GitHub 集成的工具，用于执行代码质量标准。它为拉取请求提供实时反馈，确保代码干净、一致。

以下是它的关键功能：

+   它在拉取请求过程中识别风格违规和问题

+   它根据您的特定编码标准定制 HoundCI

+   它支持多种编程语言

以下是使用 HoundCI 的优点：

+   HoundCI 为代码风格问题提供即时反馈，帮助团队保持一致的编码实践

+   与 GitHub 的无缝集成使得它容易融入现有工作流

以下是使用 HoundCI 的缺点：

+   HoundCI 主要关注风格和最佳实践，可能无法覆盖更深层次的静态分析需求

+   它专为 GitHub 仓库设计，限制了其在使用其他版本控制系统的团队中的可用性

访问 HoundCI 的官方网站，了解更多：[`houndci.com/`](https://houndci.com/) 。

这里是相关链接：[`github.com/houndci/hound`](https://github.com/houndci/hound) , [`github.com/marketplace/hound`](https://github.com/marketplace/hound) , [`www.houndci.com/configuration`](https://www.houndci.com/configuration)

接下来，我们将看看测试和调试工具。测试和调试工具对于确保软件质量和可靠性也至关重要。它们帮助在开发过程中尽早识别和修复错误，防止潜在问题影响最终用户，节省时间和资源。

# 测试和调试工具

测试和调试工具是软件开发生命周期中的关键组成部分，确保代码正确运行并符合质量标准。这些工具帮助在开发过程中尽早识别问题，通过在问题升级之前解决潜在问题，最终节省时间和资源。在本节中，我们将探讨各种测试和调试工具及其特性、优点和限制，并重点介绍它们如何提升您的开发工作流程。

## Jest

**Jest** 是一个由 Christoph Nakazawa 开发的广泛使用的测试框架，现为 Meta 所有，但它是一个开源项目，拥有许多开发者。它在 JavaScript 开发者中尤其流行，提供了一个全面的 JavaScript 应用测试解决方案，注重简洁性和易用性。Jest 特别适用于测试 React 应用，但也可以与其他框架和库一起使用。

Jest 的一个优点是其快照测试功能。这允许开发者捕捉组件输出的快照，方便验证代码的更改不会引入意外的副作用。当组件渲染时，Jest 会将输出与先前保存的快照进行比较。如果有差异，Jest 会提醒开发者，便于他们回顾更改并确保一切按预期工作。Jest 还具备强大的模拟功能，允许开发者在测试过程中隔离组件。这在测试依赖外部服务或 API 的组件时特别有用。通过模拟这些依赖，开发者可以专注于测试组件的逻辑，而不必担心外部资源的行为。

下面是一个包含 JavaScript 代码的示例：

```py
test('adds 1 + 2 to equal 3', () => {
    expect(1 + 2).toBe(3);
});
```

虽然 Jest 是一个强大的工具，但需要认识到，它可能并不适用于每一种测试场景。例如，在需要更专业测试策略的复杂应用中，比如端到端测试或性能测试，开发者可能需要结合使用专门为这些目的设计的其他工具。

尽管如此，Jest 作为一个出色的单元和集成测试框架，帮助开发者在整个项目中维护高质量的代码。如果你想了解更多细节和资源，可以访问 Jest 的官方网站：[`jestjs.io/`](https://jestjs.io/)。

## Postman

**Postman** 是一个领先的 API 测试工具，提供了一个用户友好的界面，用于发送请求和分析响应。它简化了与 API 交互的过程，使得无论是资深开发者还是新手都能轻松使用。

Postman 的一个关键特点是它能够创建自动化测试。用户可以编写测试脚本，在发送请求后自动运行，验证 API 是否按预期行为返回。这一功能对于确保 API 端点返回正确的状态码、头部和响应体非常宝贵。通过自动化这些测试，开发者可以快速识别问题，并在 API 演变过程中保持其完整性。

Postman 还提供了一个集合运行器，允许用户将请求组织成集合并按顺序执行。这对于涉及多个 API 调用的工作流测试特别有用。例如，如果一个应用要求用户在访问某些端点之前进行身份验证，开发者可以创建一个集合，首先发送登录请求，然后继续调用受保护的端点。这种顺序测试功能有助于模拟现实场景，确保整个流程按预期工作。

这里是一个 JavaScript 示例：

```py
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
```

虽然 Postman 功能丰富且被广泛认为是 API 测试的必备工具，但对于那些不熟悉 API 测试概念的人来说，它有一定的学习曲线。新用户可能需要一些时间来熟悉 Postman 的界面和功能，但这笔投资通常是值得的。

作为支持开发团队协作的工具，Postman 增强了沟通并简化了 API 开发和测试的流程。你可以访问 Postman 的官方网站了解更多信息：[`www.postman.com/`](https://www.postman.com/) 。

## Cypress

**Cypress** 是一个端到端的测试框架，专为现代 Web 应用程序设计。它提供了一个强大且易于使用的平台，用于编写模拟用户交互的测试，帮助开发者确保应用程序从用户的角度正常运行。

Cypress 的一个显著特点是其实时重载功能。这意味着开发者在编写测试时，可以立即在浏览器中看到测试结果，而无需手动刷新。这个即时反馈循环加速了开发过程，并使得早期发现问题变得更加容易。

Cypress 还提供了一个直观的时间旅行功能，允许开发者在任何时候暂停测试执行并检查应用程序。这对于调试失败特别有用，因为开发者可以清楚地看到测试失败时应用程序的状态，从而更容易找出问题的根本原因。

此外，Cypress 还与流行的 CI/CD 工具良好集成，使得开发流程中的自动化测试成为可能。这种集成帮助确保新的代码更改不会破坏现有功能，从而保持应用程序的整体质量。

使用 Cypress 进行测试的示例如下：

1.  安装 Cypress bash：**npm install** **cypress –save-dev** 。

1.  使用 bash 打开：**npx** **cypress open** 。

1.  运行测试：

    1.  点击你创建的测试套件（在这个例子中是 **My App** ）。

    1.  Cypress 会自动运行测试并在 Test Runner UI 中显示结果。

虽然 Cypress 是一个强大的测试工具，但它主要专注于 Web 应用程序。处理移动应用程序或其他环境的开发者可能需要寻找额外的测试解决方案。此外，新的用户学习曲线较陡，特别是对于那些不熟悉 JavaScript 和异步编程的人。

要了解更多关于 Cypress 的信息，可以访问 Cypress 的官方网站：[`www.cypress.io/`](https://www.cypress.io/)。

## Selenium

**Selenium** 是一个成熟的开源测试框架，允许开发者自动化浏览器以测试 Web 应用程序。它支持多种编程语言，包括 Java、C#、Python 和 Ruby，使其成为跨平台测试的多功能选择。

Selenium 的一个优势是能够模拟用户与 Web 应用程序的交互。开发者可以编写脚本来自动化任务，如点击按钮、填写表单和在页面之间导航。这一功能使得 Selenium 特别适用于端到端测试，其目的是验证应用程序的所有组件是否按预期协同工作。

Selenium 还支持多种浏览器，使开发者能够进行跨浏览器测试。这对于确保应用程序在不同的 Web 浏览器和操作系统中表现一致非常重要。随着移动设备的兴起，Selenium 也扩展了其功能，通过工具如 Appium 支持移动浏览器测试。

尽管 Selenium 拥有许多优点，但也存在一些缺点。设置和维护 Selenium 测试可能比较复杂，尤其是对于大型应用程序，类似于其他工具。此外，测试有时可能会脆弱，也就是说，它们可能会因为应用程序 UI 或行为的微小变化而失败，而不是实际的错误。开发者需要花时间创建健壮的测试并管理相关的维护工作。

访问 Selenium 的官网：[`www.selenium.dev/`](https://www.selenium.dev/)。

## Mocha

**Mocha** 是一个灵活的 JavaScript 测试框架，可以在 Node.js 和浏览器中运行。它旨在进行异步测试，使得测试依赖回调或承诺的应用程序变得更加容易。

Mocha 的一个显著特点是其简洁的语法，使得开发者能够编写清晰且富有表现力的测试代码。这种简洁性对于注重代码可读性和可维护性的团队特别有益。Mocha 支持多种断言库，开发者可以自由选择最适合自己需求的库。流行的选择包括 Chai、Should.js 和 Assert。

Mocha 还提供了多种报告选项，允许开发者以不同格式查看测试结果。这种灵活性使得 Mocha 更容易集成到现有的开发工作流和 CI/CD 管道中。

虽然 Mocha 是一个良好且高效的测试工具，但可以注意到它主要侧重于单元测试。因此，开发人员可能需要使用其他工具来补充 Mocha 进行端到端或集成测试。此外，配置 Mocha 可能需要一些前期工作，特别是对于那些刚接触 JavaScript 测试的团队。有关 Mocha 的更多细节，请参见此处：[`mochajs.org/`](https://mochajs.org/)。

## Charles Proxy

**Charles Proxy** 是一个网络调试工具，允许开发人员查看计算机与互联网之间所有 HTTP 和 SSL/HTTPS 流量。它充当代理服务器，使开发人员能够实时检查和分析请求和响应。

Charles Proxy 的主要用途之一是 API 测试和调试。通过捕获网络流量，开发人员可以轻松识别诸如请求参数错误、响应格式异常或身份验证问题等问题。这种可视性对于处理 API 至关重要，因为它使开发人员能够快速有效地排查问题。

Charles Proxy 还支持请求修改和响应模拟等功能。这意味着开发人员可以实时修改请求，从而在不更改实际应用程序代码的情况下测试不同的场景。

此外，Charles Proxy 可以模拟不同的网络条件，使你能够查看应用程序在各种情况下的表现。

从[`www.charlesproxy.com/`](https://www.charlesproxy.com/) 获取工具。

1.  配置 Charles Proxy：

    +   如果需要，可以配置你的系统或浏览器使用 Charles Proxy 作为 HTTP 代理。通常，这涉及在浏览器或操作系统中设置代理。

1.  打开你的应用程序：

    +   启动你要测试的 Web 应用程序。

1.  拦截请求：

    +   在 Charles Proxy 中，启用**断点**功能。

    +   点击**工具**菜单并选择**断点**。

    +   启用**启用** **断点**选项。

1.  测试你的应用程序：

    +   在你的应用程序中执行触发 HTTP 请求的操作。Charles Proxy 将拦截这些请求并在断点处暂停。

1.  检查并修改请求：

    +   在 Charles Proxy 中，检查拦截的请求详情，包括 URL、头部和请求正文。

    +   你可以修改请求参数、头部或正文来模拟不同的场景。

1.  继续请求：

    +   若要继续带有修改参数的请求，请点击 Charles Proxy 中的**执行**按钮。

以下是一些示例：

+   **测试身份验证**：拦截登录请求，修改用户名或密码，并观察应用程序的响应

+   **模拟网络条件**：修改请求头以模拟慢速网络速度或不同的网络类型

+   **调试 API 调用**：检查请求和响应，以识别 API 交互中的问题或错误

尽管 Charles Proxy 是一款优秀的调试工具，但它可能并不适用于所有场景。对于开发大型应用程序的开发者来说，流量量可能变得过于庞大，导致难以定位具体问题。此外，新用户在学习如何操作界面和配置设置时，可能会面临一定的学习曲线。这些工具大多数都需要投入不少精力来掌握，但它们是值得的。

如果你想了解更多关于 Charles Proxy 的信息，可以访问 Charles Proxy 的官方网站：[`www.charlesproxy.com/`](https://www.charlesproxy.com/)。

# 总结

本章为理解可供开发者使用的各种非 LLM AI 工具奠定了基础。通过探索这些工具的功能和最佳实践，你可以更好地装备自己，提升编码工具包和工作流程。欲了解更多信息，阅读本章中链接的官方站点。

在本章中，我们深入探讨了多种非 LLM AI 编程工具，重点介绍了它们的功能、能力和局限性。

我们首先检查了代码补全和生成工具，如 Content Assist 和 PyCharm 以及 NetBeans 的代码补全工具，这些工具通过提供实时建议和自动化重复任务，大大提高了编码效率。

接下来，我们探讨了静态分析工具，如 SonarQube 和 ESLint，这些工具在保持代码质量和在开发过程中及早识别潜在问题方面发挥着至关重要的作用。

最后，我们讨论了测试和调试工具，如 Jest 和 Postman，强调它们在确保应用程序正确运行并满足用户期望方面的重要性。

将这些工具集成到你的编码工作流程中，能够创建一个强大的工具包，从而提升软件开发过程中的各个方面。虽然大语言模型（LLMs）提供了宝贵的帮助，但利用非 LLM 工具可以最大化生产力，并确保你的代码不仅是功能性的，还干净、易维护且高效。通过结合使用这些工具，开发人员能够有效应对挑战，简化工作流程，并提高整体代码质量。

本章帮助没有相关知识的开发者开始理解可供开发者使用的广泛非 LLM AI 工具。通过探索它们的功能和最佳实践，你可以装备自己，掌握提升编码工具包和工作流程所需的知识。

学会并掌握这些工具，将使你能够提升开发实践，从而带来更高效、更有成效的编码体验。

You.com 是本章中软件工具的良好信息来源：[`you.com`](https://you.com) [You.com]。

在*第十一章*中，我们将探讨如何利用 LLM 帮助他人，并最终最大化你的职业生涯：为什么你应该指导他人，更多分享工作的方式，建立网络，以及使用 LLM 的一些新方法。

# 参考文献

除了之前提到的来源，以下是一些更多的参考资料：

+   *Copilot* : 微软，[`copilot.microsoft.com/`](https://copilot.microsoft.com/)，[`copilot.cloud.microsoft/en-GB/prompts`](https://copilot.cloud.microsoft/en-GB/prompts)

+   *Eclipse_Help* : “内容辅助”，Eclipse，[`help.eclipse.org/latest/index.jsp?topic=%2Forg.eclipse.cdt.doc.user%2Fconcepts%2Fcdt_c_content_assist.htm`](https://help.eclipse.org/latest/index.jsp?topic=%2Forg.eclipse.cdt.doc.user%2Fconcepts%2Fcdt_c_content_assist.htm)

+   *Gemini* : [`gemini.google.com/`](https://gemini.google.com/)

+   *Jetbrains_Completion* : “代码补全”，JetBrains，[`www.jetbrains.com/help/pycharm/auto-completing-code.html`](https://www.jetbrains.com/help/pycharm/auto-completing-code.html)

+   *Jetbrains_refactoring* : “重构代码”，JetBrains，[`www.jetbrains.com/help/pycharm/refactoring-source-code.html`](https://www.jetbrains.com/help/pycharm/refactoring-source-code.html)

+   *NetBeans_Completion* : “NetBeans 代码补全教程”，Apache，[`netbeans.apache.org/tutorial/main/tutorials/nbm-code-completion/`](https://netbeans.apache.org/tutorial/main/tutorials/nbm-code-completion/)

+   *Netbeans_SmartCode* : “NetBeans IDE Java 编辑器中的代码辅助：参考指南：智能代码补全”，Apache：[`netbeans.apache.org/tutorial/main/kb/docs/java/editor-codereference/#_smart_code_completion`](https://netbeans.apache.org/tutorial/main/kb/docs/java/editor-codereference/#_smart_code_completion)

+   *Wiki_Include* : “包含保护”，各种，[`en.wikipedia.org/wiki/Include_guard`](https://en.wikipedia.org/wiki/Include_guard)

+   *You.com* : [`you.com`](https://you.com)

# 第四部分：用 LLM 最大化你的潜力：超越基础

本节探讨了如何利用 LLM 促进个人和职业成长。我们将查看各种可以增强 LLM 能力的 AI 工具，创建一个非常强大的 AI 工具包。我们还将涵盖导师策略、社区参与以及在 LLM 驱动的编程领域推进职业发展的方法。最后，我们将了解各种新兴趋势、技术进展以及 LLM 对软件开发的长期影响。

本节覆盖以下章节：

+   *第十一章* *,* *用 LLM 帮助他人并最大化你的职业生涯*

+   *第十二章* *,* *LLM 在软件开发中的未来*
