

# 第二章：从安装到你的第一个 AI 生成文本

现在我们已经完成了第一章，让我们讨论一下**Auto-GPT**的必备要求。

在这一点上，在我们开始之前，是否选择在 OpenAI 注册一个**应用程序接口**（**API**）账户取决于你自己。我首先建议在注册之前尝试安装并启动 Auto-GPT，以防 Auto-GPT 只在 Docker 中运行（这可能会发生，因为它不断变化）；你可能能够或不能运行 Auto-GPT。不过，我们先从没有账户的情况下设置 Auto-GPT 开始。否则，你会有一个 OpenAI 账户，但其实并不需要它。

在本章中，我们将指导你准备机器以运行 Auto-GPT，安装过程，以及你使用 Auto-GPT 的第一步。

我们将涵盖 Auto-GPT 的基本概念、安装和设置说明。最后，我们将解释如何使用 Auto-GPT 执行你的第一个 AI 自动化任务。

Auto-GPT 团队（包括我）正在努力让 Auto-GPT 尽可能易于访问。

最近，Auto-GPT 的一位维护者开发了一个新工具，叫做**Auto-GPT Wizard**。如果你在任何时候遇到设置 Auto-GPT 的困难，这个工具旨在自动化安装并让新手更容易上手 Auto-GPT。

你可以在 [`github.com/Significant-Gravitas/AutoGPT_Wizard`](https://github.com/Significant-Gravitas/AutoGPT_Wizard) 找到这个工具。

在本章中，我们将学习以下内容：

+   系统要求和先决条件

+   安装并设置 Auto-GPT

+   了解基本概念和术语

+   首次运行

以下是一些系统要求和先决条件：

+   安装 VS Code。

+   安装 Python。

+   安装 Poetry。

# 安装 VS Code

我强烈建议安装 VS Code 以提高可用性，或者使用任何你认为合适的 Python IDE。作为 Auto-GPT 的一个 triage catalyst（审阅者、支持者和贡献者角色），我遇到过很多人因为使用文本编辑器，甚至是 Microsoft Word 而卡住。

使用配置得当的高级文本编辑器可能足够用于基本脚本编写或编辑配置文件，因为它们可以配置以避免文本编码和文件扩展名错误的问题。然而，像 VS Code 这样的 IDE 提供了更强大的工具和集成，能够为复杂的项目（如 Auto-GPT）提供无缝的开发体验；但我们需要编辑 JSON 文件、`.env` 文件，有时还需要编辑 markdown（`.md`）文件。使用其他工具编辑这些文件可能会导致错误的文件扩展名（例如，`.env` 和 `settings.json` 可能会变成 `.env.txt` 或 `settings.json.docx`，这些是无法正常工作的）。

作为一个许多开发者都会使用的常见工具，并且它是免费的，我们将重点介绍 VS Code。

为了不偏离为何你还可以使用 VS Code 这个话题，微软写了一篇非常好的文章，说明了为何 VS Code 值得使用。当然，你也可以使用其他的 IDE。我推荐使用 VS Code 的主要原因是它是开源的且免费使用，并且被大多数 Auto-GPT 贡献者使用，这使得它与 Auto-GPT 以及一些 VS Code 的集成项目设置非常容易配合使用。

## 安装 Python 3.10

如果你想直接运行 Auto-GPT 而不使用 Docker，可能需要安装 Python 3.10 或将其启用为终端的 `python` 和 `python3` 别名，以确保 Auto-GPT 不会意外调用其他 Python 版本。

Auto-GPT 是用 Python 开发的，它特别要求使用 Python 版本 3.10.x。3.10.x 中的 *x* 代表任何子版本（例如，3.10.0、3.10.6），该软件与这些子版本兼容。

尽管 Auto-GPT 在文件大小上比较轻量，但根据你启用的选项和插件，它可能会消耗大量资源。因此，确保有一个兼容并且经过优化的环境对于保证 Auto-GPT 及其插件的顺利运行至关重要，因为这些插件都是为 Python 3.10 编写的，并且那些预期的模块也都为 3.10 版本所准备。除了安装 Python 3.10，建议为 Auto-GPT 开发使用虚拟环境。虚拟环境允许你按项目管理依赖项和 Python 版本，确保 Auto-GPT 在一个独立且可控的环境中运行，而不会影响你可能正在开发的其他 Python 项目。这对于维护兼容性并避免不同项目要求之间的冲突至关重要。

## 为什么选择 Python 3.10？

Python 3.10 引入了许多对运行 Auto-GPT 有利的功能和优化。其中一项功能是改进了类型提示的语法。在 Python 3.10 中，你可以使用管道符号 `|` 作为一种更简洁的方式，表示一个变量可以是多种类型。这被称为 **类型** **联合操作符**：

```py
File "L:\00000000ai\Auto-GPT\autogpt\llm\providers\openai.py", line 95, in <module>
    OPEN_AI_MODELS: dict[str, ChatModelInfo | EmbeddingModelInfo | TextModelInfo] = {
TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

在这个示例错误信息中，Auto-GPT 尝试使用这种新的类型联合语法，而该语法在 3.10 之前的 Python 版本中不被支持。这就是为什么使用 Python 3.9 会导致语法错误，因为它无法解析新的语法。

此外，Python 3.10 带来了性能改进、更好的错误信息以及对复杂应用程序（如 Auto-GPT）有利的新功能。

因此，为了避免兼容性问题并利用新功能和优化，正确安装和设置 Python 3.10 在运行 Auto-GPT 之前是至关重要的。

### Python 安装先决条件

在安装 Python 3.10 之前，确保你的系统满足必要的先决条件非常重要。以下是这些先决条件：

+   **足够的磁盘空间**：确保你的系统有足够的空闲磁盘空间来容纳 Python 安装以及你将来可能安装的任何附加包或库。

+   **检查现有的 Python 安装**：如果你的系统中已经安装了先前版本的 Python，建议检查是否存在与 Python 3.10 的潜在冲突或兼容性问题。你可以通过运行适当的版本特定命令或使用操作系统的 Python 版本管理工具来检查。

确保你的系统满足这些先决条件后，你可以自信地继续安装 Python 3.10 并成功设置 Auto-GPT。

### 安装 Python 3.10

Auto-GPT 主要基于 Python 3.10 包；如果你尝试使用 3.9 运行它，举例来说，你只会遇到一些异常，无法执行 Auto-GPT。

运行 Python 3.10 需要一个能够支持此版本编程语言的系统。以下是每个操作系统的系统要求和安装说明：

+   对于 Windows 系统，请参考[`www.digitalocean.com/community/tutorials/install-python-windows-10`](https://www.digitalocean.com/community/tutorials/install-python-windows-10)的文档。

    要验证安装是否成功，请在命令提示符中运行以下命令：

    ```py
    python --version command in the command prompt.
    ```

+   在 Linux（Ubuntu/Debian）上安装 Python 3.10 时，可能需要根据所使用的 Linux 发行版做一些研究。不过，正如人们所说，强大的能力带来的是巨大的责任；你可能需要研究如何在你的机器上启用 Python 3.10。

+   对于 Ubuntu 和 Debian，关于如何安装 3.10 的文档可以在此找到：[`www.linuxcapable.com/how-to-install-python-3-10-on-ubuntu-linux/`](https://www.linuxcapable.com/how-to-install-python-3-10-on-ubuntu-linux/)。

    要验证安装是否成功，请运行以下命令：

    ```py
    python3.10 –version
    ```

    系统应该返回 Python 3.10.x。

注意

具体的命令和步骤可能会根据每个操作系统的版本略有不同。始终参考官方的 Python 文档或操作系统的文档，以获取最准确和最新的信息。

## 安装 Poetry

最近添加了一个新的依赖项，安装起来有点复杂。

如何安装的文档可以在这里找到：[`python-poetry.org/docs/#installing-with-pipx`](https://python-poetry.org/docs/#installing-with-pipx)。

如果你在设置过程中遇到困难（例如在 Windows 上），你也可以尝试这里的向导脚本：[`github.com/Significant-Gravitas/AutoGPT_Wizard`](https://github.com/Significant-Gravitas/AutoGPT_Wizard)。

### 可能出现的额外要求

请检查官方文档：[`docs.agpt.co/autogpt/setup/`](https://docs.agpt.co/autogpt/setup/)，以确保没有遗漏任何内容。

除了在系统上安装兼容版本的 Python 和 poetry 外，还必须确保您的硬件符合运行 Auto-GPT 的特定最低要求：

+   **处理器（CPU）**：推荐使用现代多核处理器（如 Intel Core i5/i7 或 AMD Ryzen），以实现在使用 Auto-GPT 时的最佳性能。

+   **内存（RAM）**：建议至少有 8 GB RAM；然而，拥有更多可用内存将会提高处理大型数据集或复杂任务时的性能。

+   **存储**：确保计算机主存储驱动器（HDD/SSD）上有足够的空闲磁盘空间 — 至少数千兆字节 — 因为在操作期间 Auto-GPT 可能会生成临时文件，并且在存储生成的输出文件时可能需要额外空间。

+   **互联网连接**：Auto-GPT 通信需要稳定且带有合理带宽的互联网连接，以便访问 OpenAI 的 API 来使用 GPT 模型并生成文本。

+   **GPU 支持（可选）**：虽然不是严格要求，但拥有兼容的 NVIDIA 或 AMD GPU 可显著提高某些任务的性能，例如使用 Silero **文本转语音**（**TTS**）引擎。

通过确保您的系统符合这些要求和先决条件，您将能够有效地安装和使用 Auto-GPT。

在接下来的章节中，我们将为您介绍如何在各种操作系统上安装 Auto-GPT，并提供与 Auto-GPT 及其底层技术相关的基本概念和术语概述。

请记住，虽然这些系统要求和先决条件旨在在使用 Auto-GPT 时提供流畅的体验，但具体任务的个体需求可能会有所不同。例如，如果您打算用 Auto-GPT 进行大规模文本生成或复杂的**自然语言处理**（**NLP**）任务，您可能会受益于拥有更强大的 CPU 或额外的可用内存。

无论如何，在使用 Auto-GPT 时，随时监视系统性能并根据需要调整硬件配置都是一个好主意。这将有助于确保您能够充分利用这款强大的基于 AI 的文本生成工具，而不会遇到性能瓶颈或其他问题。

在安装和设置 Auto-GPT 在您的系统上之前，请执行以下操作：

1.  确保您的操作系统（macOS、Linux/Ubuntu/Debian、Windows）满足运行 Python 3.10 的最低要求。

1.  安装 Python 3.10.x，按照每个操作系统提供的指示进行操作。

1.  通过在终端（macOS/Linux）或命令提示符（Windows）中检查其版本，验证 Python 3.10.x 是否已正确安装。

1.  确保您的硬件符合最低要求，如处理器（CPU）、内存（RAM）、存储空间可用性以及互联网连接的稳定性/带宽。

1.  这一步是可选的。如果计划使用资源密集型功能，如语音合成引擎或本地 LLM（如 Vicuna 或 LLAMA），可以考虑 GPU 支持（这是一个有趣的话题，因为大多数 GPU 无法处理与 Auto-GPT 兼容的 LLM）。

通过仔细遵循这些指南，并确保在安装 Auto-GPT 之前系统满足所有要求和前置条件，你将为成功安装并愉快地使用这个强大的 AI 驱动文本生成工具做好充分准备。

在接下来的部分，我们将引导你完成每个步骤，帮助你开始使用这个令人惊叹的软件——从为每个操作系统量身定制的安装程序，到理解 Auto-GPT 功能背后的基本概念和术语。

# 安装和设置 Auto-GPT

以下是安装 Auto-GPT 的步骤：

1.  根据你的经验，你可能想要直接使用 Git 并从[`github.com/Significant-Gravitas/Auto-GPT.git`](https://github.com/Significant-Gravitas/Auto-GPT.git)克隆仓库。或者，如果你对终端不太熟悉，你可以访问[`github.com/Significant-Gravitas/Auto-GPT`](https://github.com/Significant-Gravitas/Auto-GPT)。

1.  在右上角点击`.zip`文件，并将其保存到你希望存放 Auto-GPT 文件夹的任何位置。然后，简单地解压`.zip`文件。

1.  如果你想 100%确保使用的是最稳定的版本，请访问[`github.com/Significant-Gravitas/Auto-GPT/releases/latest`](https://github.com/Significant-Gravitas/Auto-GPT/releases/latest)。

1.  选择最新的版本（在我们的例子中是 0.4.1），在该帖子的**Assets**部分下载`.zip`文件并解压。

1.  我使用的最新版本是 v0.4.7；任何更高版本可能已重新结构化，例如，0.5.0 版本已经将 Auto-GPT 文件夹放在了`Auto-GPT/autogpts/autogpt`中。为了更详细地了解，阅读仓库中的更新版`README`和文档，查看你正在使用的版本。

## 安装 Auto-GPT

对于一个快速发展的项目，Auto-GPT 的安装可能会有所不同，因此如果你在按照以下指南操作时遇到问题，可以查看[`docs.agpt.co/`](https://docs.agpt.co/)了解是否有任何变化。

由于 Auto-GPT 本身包含多种 Python 依赖项，你现在可能想要在终端中导航到你的 Auto-GPT 文件夹。

使用 Docker 运行 Auto-GPT，请按以下步骤操作：

1.  一些开发者直接使用 Dockerfile，但我（作为 Docker 新手）推荐使用`docker-compose.yml`，这是一些人添加的。

1.  确保你已安装 Docker（请回到上一章节的*安装 Docker*部分）。

1.  只需进入 Auto-GPT 文件夹，运行以下命令：

    ```py
    docker-compose build auto-gpt
    docker-compose run –rm auto-gpt –gpt3only
    ```

注意

我提供`–gpt3only`仅仅是为了确保我们现在不花费任何钱，因为我假设你刚创建了 OpenAI 账户，并获得了 5 美元的免费起始奖金。

## 使用 Docker 拉取 Auto-GPT 镜像

在这里，让我们确保你的系统上安装了 Docker。如果不确定，你可以跳到 *第六章*，我会在其中介绍如何在你的机器上设置 Docker，并给出一些关于如何在 Auto-GPT 中使用 Docker 的额外提示。

如果你已经安装了 Docker，请执行以下步骤：

1.  为 Auto-GPT 创建一个项目目录：

    ```py
    mkdir Auto-GPT
    docker-compose.yml with the specified contents provided in the documentation.
    ```

1.  创建必要的配置文件。你可以在仓库中找到模板。

1.  从 Docker Hub 拉取最新镜像：

    ```py
    docker pull significantgravitas/auto-gpt
    ```

1.  按照文档中的指示使用 Docker 运行。

## 使用 Git 克隆 Auto-GPT

假设你已经在系统上安装了 Git（例如 Windows 系统默认不带 Git），我们将在这里介绍如何克隆 Auto-GPT。

让我们确保你的操作系统已安装 Git：

1.  我们首先需要借助以下命令来克隆仓库：

    ```py
    git clone -b stable https://github.com/Significant-Gravitas/Auto-GPT.git
    ```

1.  接下来，我们将导航到你下载仓库的目录：

    ```py
    cd Auto-GPT
    python –m pip install –r ./requirements.txt
    ```

未使用 Git/Docker

1\. 从最新的稳定版本下载源代码（`.zip` 文件）。

2\. 将压缩文件解压到一个文件夹中。

1.  接下来，我们将导航到你下载仓库的目录：

    ```py
    cd Auto-GPT
    python –m pip install –r ./requirements.txt
    ```

注意

从这里开始，根据你可能安装的版本，Auto-GPT 可能位于 `Auto-GPT/autogpts/autogpt` 文件夹中，因为主仓库已被转变为一个框架，用于创建其他 `Auto-GPT` 实例。我们在本书中讨论的 Auto-GPT 项目位于前述文件夹中。

### 配置

下面是我们如何进行配置：

1.  在主 Auto-GPT 文件夹中找到名为 `.env.template` 的文件。

1.  创建 `.env.template` 的副本，并将其重命名为 `.env`。

1.  在文本编辑器中打开 `.env` 文件。如果还没有，建议使用 VS Code 之类的工具，这样你就可以将 Auto-GPT 作为项目打开，并编辑你需要的任何内容。

1.  找到包含 `OPENAI_API_KEY=` 的那一行。

1.  在 `=` 符号后输入你的唯一 OpenAI API 密钥，不要加引号或空格。

1.  如果你使用多个 Auto-GPT 实例（只需另建一个 `auto-gpt` 文件夹即可轻松实现，最好创建多个 API 密钥），你可以确保关注每个实例的费用。

1.  根据你可以访问的 GPT 模型，你现在需要像我们刚才对 API 密钥所做的那样，修改 `FAST_LLM_MODEL` 和 `SMART_LLM_MODEL` 属性。

1.  要了解哪些模型对你可用，请访问 [`platform.openai.com/account/rate-limits`](https://platform.openai.com/account/rate-limits)。

1.  它只列出了你可以使用的项。

截至写这章时，OpenAI 刚刚发布了一个 16 K 模型的 gpt-3.5-turbo-16k。它可以处理比 GPT-4 更多的令牌/单词，但我通常认为其输出仍不如 GPT-4，因为 Auto-GPT 倾向于执行一些它凭空编造的随机任务。

问题出在上下文处理能力上，尽管它可以处理更多的令牌，GPT-4 的参数更多，并且经过了更多优化。

如果设置 GPT-3.5-Turbo 作为模型，则默认的令牌数量为 4,000；如果设置 GPT-4 作为模型，则为 8,000 令牌，但我建议将这些限制稍微调低。

例如，使用 7,000 而不是 8,000 令牌，在`SMART_LLM_MODEL`上会减少内存摘要的空间，同时仍确保没有更多的单词或令牌溢出到 Chat Completion 提示中。

Auto-GPT 引入了自定义选项，比如禁止某些命令或选择你想要使用的文本转语音引擎。

启用语音功能使得 Auto-GPT 可以通过语音与您对话。选择使用哪个 TTS 引擎完全由您决定。我个人更喜欢 Silero TTS，它几乎与 ElevenLabs 一样优秀，而且完全免费；只需要一台具有强大 CPU 和/或 GPU 的计算机（您可以选择是否使用 CPU 或 GPU 来进行 TTS 模型的处理）。

正如你可能已经注意到的，Auto-GPT 带有大量来自人工智能和机器学习领域的术语。接下来我们将介绍一些最常见的术语。

# 基本概念和术语

在我们开始使用 Auto-GPT 之前，让我们回顾一些基本的概念和术语，以帮助我们理解它是如何工作的：

+   **文本生成**：文本生成是根据给定的输入数据或上下文创建自然语言文本的任务。例如，给定一个主题、体裁或提示，文本生成可以生成一段文字、一篇文章、一则故事或一段对话，且与输入相匹配。

+   **模型**：模型是一个系统或过程的数学表示，用来做出预测或决策。在机器学习中，模型是一个将输入映射到输出的函数。例如，一个模型可以将图像作为输入，输出描述图像内容的标签。

+   **思维链**：这一概念着眼于通过系统化和顺序应用思维过程，逐步发展和完善思想或解决方案。在使用像 ChatGPT 这样的工具时，“思维链”方法会将一个查询的输出作为下一个查询的输入，实质上创建一个“链条”般不断演进的回答。

    这种方法允许对一个主题或问题进行深度探索，因为链条中的每个步骤都建立在前一个步骤的基础上，可能会导致更细致、更复杂的结果。这在一些任务中尤其有用，例如开发复杂的叙事、迭代优化模型，或在确定解决方案之前，从多个角度探讨问题。

+   **思维树**：一种用于在文本生成中获得更好结果的策略，例如 ChatGPT，可以通过指示它解决一个问题并提供多个替代方案来实现。这可以通过说“写四个替代方案，评估它们并加以改进”来实现。这个简单的指令告诉模型要具备创造力，生成四个替代方案来替代已给出的解决方案，评估它们，并鼓励模型输出一个改进后的解决方案，而不仅仅是一个答案。

    这样可以产生更准确的输出，并且可以进行多次迭代。例如，我在开发一个新的神经元单元网络原型时，请求 ChatGPT 帮助我设计一个数据转换方法，该方法会接收一个字符串（文本）并将其应用到多个矩阵上。第一次的结果不好，甚至不是正确的 Python 代码，但经过三四次迭代，每次说“写四个可能改进该代码的替代方案并改进其策略，评估它们，给它们打分 1 到 10，排名，然后改进”，最终得到了非常干净的代码，甚至在第二次迭代后，它还给了我一些提高代码性能的改进建议，这些是我如果直接要求它的话，它是不会提供的。

+   **思维森林**：这一概念建立在思维树的原则基础上，但正如其名称所示，它有多个实例，像一群人一样进行思考。我最近观看的这个视频中有一个精彩的解释：[`www.youtube.com/watch?v=y6SVA3aAfco`](https://www.youtube.com/watch?v=y6SVA3aAfco)。

+   **神经网络**：神经网络是一种由互联的单元（称为**神经元**）组成的模型。每个神经元可以对输入执行简单的计算并生成输出。通过在不同层次和配置中组合神经元，神经网络可以从数据中学习复杂的模式和关系。例如，GPT 就有多个神经网络在运行，它们各自有不同的任务，并且由多个神经网络层次组成。

+   **深度学习**：由 OpenAI 开发的**生成预训练变换器 3**（**GPT-3**）是自然语言处理领域的一个里程碑。这一深度学习模型具有惊人的 1750 亿个参数，并拥有 45TB 的庞大训练数据集，以其文本生成能力而著称，能够在各种话题、体裁和风格中提供连贯性和多样性。尽管人们对其继任者 GPT-4 充满期待，GPT-4 承诺增强上下文理解和逻辑处理能力，但 GPT-3 仍然是一个强大的工具，特别适用于较小的任务。最近的升级使其能够处理多达 16K 的 tokens，显著提升了输出质量，尽管建议避免给模型输入过多数据，以免导致混乱。

+   **GPT-3**：GPT-3 是由 OpenAI 开发的一种深度学习自然语言处理模型。它是文本生成领域最大、最强大的模型之一，拥有 1750 亿个参数和 45TB 的训练数据。GPT-3 能够为几乎任何主题、类型或风格生成连贯且多样的文本。OpenAI 不断改进该模型，尽管后继的 GPT-4 在上下文处理能力和逻辑推理方面可能更强大，但它仍然是一个更快速、非常适用于小型任务的模型。它现在可以处理 16K 个 tokens，但我发现这个优势更多体现在输出而非输入数据上。这意味着当一次提供过多信息时，模型会很快感到困惑。

+   **GPT-4**：这是 GPT-3 的继任者，在文本生成方面更强大。它有 170 万亿个参数，是 GPT-3 的近 1,000 倍。这个模型支持所有插件，并具备 Bing 浏览器功能，使其能够自主进行信息搜索。OpenAI 在一些细节上非常保密，目前尚不清楚它具体是如何工作的。一些资源和论文表明，它是递归运作的，并且随着每次输入不断学习。

+   **Auto-GPT**：Auto-GPT 是一个自动化文本生成工具，使用 OpenAI 的聊天完成 API，主要与 GPT-4 一起使用。它允许你指定输入文本和控制输出文本的参数，如长度、语气、格式和关键词。Auto-GPT 随后通过 OpenAI API 将你的输入文本和参数发送给 GPT-3 模型，并获取生成的文本作为回应。Auto-GPT 还提供了一些功能，帮助你编辑和改进生成的文本，例如建议、反馈和重写：

    +   **插件**：可以加载到 Auto-GPT 中的扩展程序，用以增加更多功能。

    +   **无头浏览器**：一种没有图形用户界面的网页浏览器，用于自动化任务。

    +   **工作空间**：Auto-GPT 保存文件和数据的目录。

    +   **API 密钥**：API 密钥是一个用于在 API 请求中验证用户、开发者或调用程序的唯一标识符。这个密钥有助于跟踪和控制 API 的使用，防止滥用并确保安全。它本质上就像一个密码，允许访问特定服务或数据，促进不同软件组件之间的无缝和安全的通信。必须将 API 密钥保密，以防止未经授权的访问和潜在的滥用。

# 在你的机器上首次运行 Auto-GPT

要运行 Auto-GPT，你需要根据操作系统使用不同的命令。Linux 或 macOS 使用`run.sh`，Windows 使用`run.bat`。另外，你也可以直接在控制台中运行以下命令。进入 Auto-GPT 文件夹（不是里面的那个——我知道文件结构有时可能让人误解），然后执行以下命令：

```py
python -m autogpt
```

你也可以在“autogpts/autogpt”文件夹内执行“autogpt.bat”或“autogpt.sh”脚本。

如果你不确定默认的 Python 是否是 Python 3.10，或者如果前面的命令返回错误，可以使用 `python –V` 命令检查。如果返回的是除 Python 3.10 以外的内容，你可以运行以下命令：

```py
python3.10 -m autogpt
```

对于任何操作系统，如果你已安装 Docker，还可以使用 `docker-compose`。

你还可以传递一些参数来定制你的 Auto-GPT 体验，例子包括：

+   `–gpt3only` 使用 GPT-3.5 代替 GPT-4

+   `–speak` 启用语音输出

+   `–continuous` 用于在没有用户授权的情况下运行 Auto-GPT（不推荐）

+   `–debug` 打印调试日志及更多信息

+   你可以使用 `–help` 查看完整的参数列表

你还可以在 `.env` 文件中更改 Auto-GPT 设置，例如 `SMART_LLM_MODEL` 选择语言模型，`DISABLED_COMMAND_CATEGORIES` 禁用如 `auto` 等命令组，等等。你可以在 `.env.template` 文件中找到每个设置的模板和说明。

当你首次启动 Auto-GPT 时，系统会提示你提供名称、AI 角色和目标。默认情况下，这些字段是自动填充的，意味着你可以直接发出命令。

例如，要研究 *Unlocking the Power of Auto-GPT and its Plugins* 的作者 Wladastic，并将结果写入文本文件，你可以发出以下命令：

```py
"Research Wladastic, the author of Unlocking the Power of Auto-GPT and Its Plugins and write the results into a text file."
```

然后，Auto-GPT 会尝试生成 `ai_settings.yaml` 文件；如果失败，你将被要求提供实例的名称、`ai_settings` 的五个主要目标以及影响实例行为的角色。

确保在提示中非常具体和详细。使用 Auto-GPT 时，我倾向于手动编辑 `ai_settings.yaml` 文件，它对于较长的指令以及超过 5 个目标非常有效（这是默认设置，因为它是在只有 GPT-3.5 可用时开发的，GPT-3.5 的令牌限制要低得多）。

可以自由查阅 ChatGPT 提示指南，了解如何让 Auto-GPT 达到最高效率。模糊或“过于简短”的提示可能导致 Auto-GPT 出现幻觉或执行错误的操作，比如“为我的作业做研究”，这可能会导致多个步骤，例如询问用户（你）具体想要什么，这些操作将会在你的 OpenAI 账户上产生费用。

# 概要

在这一章节中，我们深入探讨了如何在各种操作系统上安装和设置 Auto-GPT，包括 Windows、macOS 和 Linux，帮助你掌握启动所需的基本知识。我们首先概述了每个平台的系统要求，并提供了详细的 Python 3.10 安装说明，这对于运行 Auto-GPT 至关重要。我们的指南还包括了获取 Auto-GPT 的不同方法，比如通过 Git 克隆仓库或从 GitHub 下载 ZIP 文件。

一旦你在系统上安装了 Auto-GPT，我们将带你通过使用 Docker（推荐）、Git 或不使用任何工具的方式进行安装。我们还解释了如何配置 `.env` 文件，输入你独特的 OpenAI API 密钥，并在 `FAST_LLM_MODEL` 和 `SMART_LLM_MODEL` 属性中设置 GPT 模型。

在成功设置 Auto-GPT 后，我们介绍了文本生成模型的基本概念，如 OpenAI 的 GPT-3/GPT-4，讨论了神经网络、用于自然语言处理（NLP）的深度学习模型，以及这些模型执行的文本生成任务。

本章进一步探讨了额外的 Auto-GPT 功能，包括增强其功能的插件、用于自动化任务的无头浏览器、用于文件管理的工作区以及用于安全访问 OpenAI 服务的 API 密钥。

最后，我们演示了如何使用 Auto-GPT 执行第一个 AI 生成的任务，突出了其作为工具的易用性和强大功能。我们以为你准备好接下来的章节作为结束，接下来的章节将深入探讨高级 Auto-GPT 功能，比如针对特定需求的定制和与各种插件的协作，以扩展其能力。通过掌握这些方面，并有效利用 AI 生成文本的力量，你将能够胜任一系列任务，从自动化内容创作到根据提示生成引人入胜的叙述。敬请期待我们即将发布的章节，继续探索 Auto-GPT 的全部潜力。

在掌握了安装和配置 Auto-GPT 的基础知识，并了解了文本生成模型后，接下来我们将探索的章节标题是 *掌握提示生成并理解 Auto-GPT 如何生成提示*。本章将加深你对提示生成的理解，这是最大化 Auto-GPT 潜力的关键技能。它将揭开 Auto-GPT 提示生成背后的机制，并提供有效构建提示的指导，从而增强你与这个高级语言模型的互动。
