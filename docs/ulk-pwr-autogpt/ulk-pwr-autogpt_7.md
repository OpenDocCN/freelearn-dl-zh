

# 第七章：使用自己的 LLM 和提示词作为指南

在**人工智能**的动态领域，可能性广阔且不断发展。在揭示 Auto-GPT 的能力时，显而易见，它的强大之处在于能够利用**GPT**的强大功能。但如果你希望超越 GPT 的范畴，探索其他 LLM，该怎么办呢？

本章将为那些希望将**大型语言模型**（**LLM**）与 Auto-GPT 集成的人们指引道路。然而，你可能会想，“*如果我有一个定制的 LLM，或者希望使用其他 LLM，该怎么办？*” 本章旨在解答这个问题，“*如何将我的 LLM* *与 Auto-GPT 集成？*”

我们还将深入探讨如何为 Auto-GPT 制定有效提示词的细节，这对充分利用这一工具至关重要。通过清晰的理解和策略性的方法，你可以引导 Auto-GPT 生成更符合需求和高效的回应。我们将探讨制定提示词的指南，帮助你与 Auto-GPT 的互动更加富有成效。

现在我们已经涵盖了 Auto-GPT 的大部分功能，我们可以专注于提示词的使用指南。

我们编写的提示词越清晰，当我们运行 Auto-GPT 时，API 的费用就越低，Auto-GPT 完成任务的效率就越高（如果能够完成的话）。

在本章中，我们将涵盖以下主题：

+   LLM 是什么以及 GPT 作为 LLM 的应用

+   已知的当前示例和要求

+   将我们的 LLM 与 Auto-GPT 进行集成和设置

+   使用不同模型的优缺点

+   编写迷你 Auto-GPT，Auto-GPT 的概念验证迷你版本

+   添加简单的记忆功能以记住对话

+   稳定的提示词——通过`instance.txt`使 Auto-GPT 稳定

+   在提示词中实施负确认

+   在提示词中应用规则和语气

# LLM 是什么以及 GPT 作为 LLM 的应用

在本书中，我们多次提到了 LLM。此时，我们需要了解 LLM 到底是什么。

从最基本的角度看，像 GPT 这样的 LLM 是一个机器学习模型。机器学习是人工智能的一个子集，使计算机能够从数据中学习。在 LLM 的情况下，这些数据主要是文本——大量的文本。可以将 LLM 看作是一个学生，他阅读的不仅仅是一本或两本书，而是数百万本书，涵盖了从历史、科学到流行文化和网络迷因等各种话题。

## 架构——神经元与层级

LLM 的架构灵感来自人类大脑，由按层组织的人工神经元组成。这些层是相互连接的，每个连接都有一个权重，这个权重在学习过程中会不断调整。该架构通常涉及多个层级，通常是数百甚至数千个层次，使其成为一个“深度”神经网络。这种深度使得模型能够学习复杂的数据模式和关系。

## 训练——学习阶段

训练一个 LLM 涉及将大量文本输入模型，并调整神经元之间连接的权重，以最小化其预测与实际结果之间的差异。例如，如果模型给定文本*The cat is on the*，它应该预测类似于*roof*或*mat*的词语，这些词能够合乎逻辑地完成这个句子。模型通过调整其内部参数，使预测尽可能准确，这一过程需要巨大的计算能力和专用硬件，如**图形处理** **单元**（**GPU**）。

## 变换器的作用

变换器架构是一种特定类型的神经网络架构，它在语言任务中证明了高度的有效性。它擅长处理序列数据，使其非常适合理解句子、段落甚至整个文档的结构。GPT 正是基于这一变换器架构，这也是它在生成连贯且上下文相关的文本方面表现出色的原因。

## LLMs 作为单词和概念的地图

想象一个 LLM 是一张广阔而复杂的地图，每个单词或短语都是一座城市，城市之间的道路代表着这些单词之间的关系。在这张地图上，两个城市距离越近，它们在上下文上的相似度越高。例如，*apple*和*fruit*这两个城市会很接近，连接它们的是一条短路，表明它们常常出现在相似的上下文中。

这张地图不是静态的；它是动态的，且不断发展变化。当模型从更多数据中学习时，新的城市会被建立，现有的城市会被扩展，道路会被更新。这张地图帮助 LLM 在复杂的人类语言景观中导航，使其生成的文本不仅语法正确，而且在上下文上相关且连贯。

## 上下文理解

现代 LLM 的最显著特点之一是它们理解上下文的能力。如果你向 LLM 提问，它不仅仅会孤立地看待这个问题；它会考虑到在此之前的整个对话。理解上下文的能力来自于变换器架构的注意力机制，该机制会对输入文本的不同部分进行加权，从而生成一个上下文合适的回应。

## LLMs 的多功能性

LLMs 是极其多功能的，能够执行除了文本生成之外的广泛任务。它们可以回答问题、总结文档、翻译语言，甚至编写代码。这种多功能性源于它们对语言的深刻理解，以及它们映射单词、短语和概念之间复杂关系的能力。

如果你在谷歌上搜索“LLM”，你可能会被成千上万的 LLM 模型所淹没。接下来，我们将探索最常用的几种模型。

# 已知的当前示例和必要条件

虽然 OpenAI 的 GPT-3 和 GPT-4 是著名的 LLM，但在 AI 领域还有其他值得注意的模型：

+   **GPT-3.5-Turbo**：OpenAI 的产品，GPT-3 因其在数百 GB 的文本数据上进行深度训练而脱颖而出，能够生成极其接近人类的文本。然而，它与 Auto-GPT 的兼容性有限，因此在某些应用中并不是首选。

+   **GPT-4**：GPT-3 的继任者，GPT-4 提供了更强大的能力，更适合与 Auto-GPT 集成，提供更加流畅的体验。

+   **BERT**：谷歌的 **双向编码器表示模型**（**BERT**）是 LLM 领域的另一位重量级选手。与 GPT-3 和 GPT-4 的生成式模型不同，BERT 是判别式的，使得它在理解文本方面比生成文本更为擅长。

+   **RoBERTa**：Facebook 的创新之作，RoBERTa 是 BERT 的变种，在一个更大的数据集上进行训练，在多个基准测试中超过了 BERT。

+   **Llama**：这个模型由 Meta 制作。传闻它曾被泄露，许多基于它的模型应运而生。

+   **Llama-2**：Llama 的改进版，性能更好，每个 token 的资源消耗更少。Llama-2 的 7-B Token 模型与 Llama-1 的 13-B 模型表现相似。Llama-2 有一款新的 70-B 模型，看起来在直接与 Auto-GPT 配合使用时非常有前景，它似乎与 GPT-3.5-Turbo 不相上下。

+   **Mistral 和 Mixtral 模型**：由 Mistral AI 制作，有多种模型不同于 Llama，这些模型在 Llama-3 发布之前非常流行。

+   **Llama-3 和 Llama-3.1**：比之前的任何 Llama 模型都要更强大，第一个基于 Llama-3 8B 的模型以超高的上下文处理能力问世，并且在 256k 或甚至超过 100 万个 tokens 上进行训练。在 Llama-3.1 发布之前，它们被认为是最好的模型，而 Llama-3.1 的原生支持 128k tokens。

如你所见，目前有许多模型可供选择；我们这里只是刚刚触及表面。几个社区已经崭露头角，继续在这些模型的基础上进行开发，包括一些公司也在制作自己的变种。

如前所述，有一组模型特别吸引了我的注意，因为它是我唯一能够有效与 Auto-GPT 配合使用的模型：Mixtral 和 Mistral。

我最喜欢的模型是 NousResearch/Hermes-2-Pro-Mistral-7B 和 argilla/CapybaraHermes-2.5-Mistral-7B。它们与 JSON 输出以及我的代理项目配合得非常好，甚至有一段时间我完全停止使用 OpenAI API。Mixtral 是多个专家模型的组合（这些专家模型是同一模型或不同模型的不同配置，它们作为一个模型委员会同时运行并共同做出决策），传闻 GPT-4 也是如此运作的，这意味着多个 LLM 会共同决定哪个输出是最准确的，从而显著提高其表现。

Mistral 7B 是一种新型的 LLM，经过精心设计，能够提供更干净的结果，并且比同类的 70 亿参数模型更高效。Mistral 通过使用 8,000 令牌的上下文进行训练，达到了这个目标。然而，它的理论令牌限制是 128k 令牌，这使得它能够处理比标准 Llama-2 更大的文本内容。

要运行本地 LLM，你需要找到最适合你的方法。一些可以帮助你的程序包括 Ollama、GPT4ALL 和 LMStudio。我个人喜欢使用 oobabooga 的文本生成 Web UI，因为它集成了类似 OpenAI API 的 API 扩展，并且有像 Coqui TTS 这样的插件，便于构建和玩转你的 AI 角色。

此外，还有一些插件，例如*Auto-GPT-Text-Gen-Plugin*（[`github.com/danikhan632/Auto-GPT-Text-Gen-Plugin`](https://github.com/danikhan632/Auto-GPT-Text-Gen-Plugin)），可以让用户通过其他软件为 Auto-GPT 提供支持，如*text-generation-webui*（[`github.com/oobabooga/text-generation-webui`](https://github.com/oobabooga/text-generation-webui)）。这个插件特别设计用来让用户自定义发送给本地安装的 LLM 的提示，从而有效地摆脱对 GPT-4 的依赖，并在 Auto-GPT 的使用环境下让 GPT-3.5 变得不那么重要。

现在我们已经介绍了一些本地 LLM，并给你提供了一些选择时需要注意的事项（由于无法详细解释每个项目的内容），接下来我们可以动手实践，开始使用带有 Auto-GPT 的 LLM！

# 将 LLM 与 Auto-GPT 集成和设置

要将自定义 LLM 与 Auto-GPT 集成，你需要修改 Auto-GPT 代码，以便它能够与所选模型的 API 进行通信。这涉及到请求生成和响应处理的修改。完成这些修改后，进行严格的测试是确保兼容性和性能的关键。

对于使用上述插件的用户，它提供了 Auto-GPT 和 text-generation-webui 之间的桥梁。该插件使用一个文本生成 API 服务，通常安装在用户的计算机上。这种设计方式提供了在不影响插件性能的情况下选择和更新模型的灵活性。插件还允许定制提示，以适应特定的 LLM，确保提示能够与所选模型无缝对接。

由于每个模型的训练方式不同，我们还需要进行一些研究，了解该模型是如何训练的：

+   **上下文长度**：模型的上下文长度是指它一次可以处理的令牌数量。一些模型可以处理更长的上下文，这对于保持文本生成的一致性至关重要。

+   **工具能力**：Auto-GPT 使用 OpenAI 的框架来执行每个 LLM 请求。随着时间的推移，OpenAI 开发了一个功能调用系统，对于较小的 LLM 来说，这个系统非常难以使用。Auto-GPT 曾只与 JSON 输出兼容，而我发现这种方式在本地 LLM 上效果更好。

+   `n_batch`长度。我们将在*使用不同模型的优缺点*部分详细探讨这个问题。

+   **JSON 支持**：JSON 是一种易于人类阅读和编写，并且易于机器解析和生成的数据格式。然而，对于 LLM 来说，这并不容易，因为 LLM 无法知道 JSON 输出应该表示什么，除了它被训练在许多 JSON 输出示例上。这导致 LLM 经常开始在 JSON 内部输出一些并非提示或上下文的一部分的信息，而这些内容仅是训练数据的一部分。

为了能够有效地向 LLM 解释你期望它做什么，LLM 必须能够理解你想要的内容。你可以通过使用指令模板来做到这一点。

## 使用正确的指令模板

虽然一些模型可能已经使用 LLama 提供的指令模板进行训练，但其他模型则使用定制的模板，如 Mistral 中的 ChatML。

text-generation-webui API 扩展提供了一种传递我们想要使用的指令模板的方法。我们可以通过向发送给 API 的`POST`请求添加必要的属性来做到这一点。

在这里，我为`POST`请求添加了一些重要的属性：

`> data = {`

`> > "``mode": "instruct",`

`> > "``messages": history,`

`#` 始终需要添加一个历史数组

`> > "``temperature": 0.7,`

`#` 这可能会有所不同，取决于所使用的模型。

`> > "``user_bio": "",`

`#` 这是仅适用于 text-generation-webui，并包含用户的个人信息。我们必须在这里提到它，否则 API 将无法正常工作。你阅读时这个问题可能已经修复。

`> >` `"``max_tokens": 4192,`

`#` 这可能会有所不同，取决于你使用的模型。

`> > "``truncation_length": 8192,`

`> > "``max_new_tokens": 512,`

`> > "``stop_sequence": "<|end|>"`

`> > }`

在这里，`max_tokens`、`truncation_length`和`max_new_tokens`必须正确设置。首先是`max_tokens`，它指定 LLM 一次可以处理的最大 token 数量；`truncation_length`指定 LLM 可以处理的总 token 数量；`max_new_tokens`指定 LLM 一次可以生成的最大 token 数量。

要计算最佳值，必须设置`max_tokens`，就像在使用 OpenAI 的 API 时一样。然后，你需要设置`truncation_length`，使其是`max_tokens`的两倍，并设置`max_new_tokens`，使其是`max_tokens`的一半。

请注意，`truncation_length`必须低于你在运行 LLM 时选择的上下文长度。任何高于上下文长度的值都会导致错误，因为 LLM 无法一次处理这么多的上下文。我建议将其设置为稍低于上下文长度，以确保安全。例如，在运行 Qwen 的 CodeQwen-7b-chat 时，我将上下文长度设置为 32k tokens。这意味着我可以将`truncation_length`设置为 30k tokens，甚至是 20k tokens。

你需要尝试不同的值，因为`max_new_tokens`可能会有些棘手。将其设置高于 2,048 通常会导致输出不可预测，因为大多数 LLM 无法一次处理这么多的 token（`n_batch`定义了 LLM 每次处理的 token 数量，通过多次迭代较大的上下文来处理多个步骤，`n_batch`的值应接近`max_new_tokens`的值；否则，LLM 将不知道输出什么）。然而，它适用于`Llama-3-8B-Instruct-64k.Q8_0.gguf`，该模型可以在[`huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-64k-GGUF`](https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-64k-GGUF)找到，能够一次处理 64k 个 token。然而，它需要大约 20-22GB 的 VRAM 来运行。幸运的是，它已经量化为 GGUF，你可以将 LLM 分布到 GPU 的 VRAM 和计算机的 RAM 上，这样就能在 GPU 和 CPU 之间分担负载。虽然这会让模型运行更慢，但嘿，它确实能工作，并且可以一次处理 64k 个 token！

在这个例子中，我们告诉 API 我们希望使用 ChatML 的指令模板，格式如下：

```py
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}
```

这只是一个简单的脚本，描述了之前提到的历史对话格式。它应该是这样的：

```py
message: [ {
"role ": "system ", "content ": "You are a helpful assistant. Always answer the user in the most understandable way and keep your sentences short! "
"role": "user", "content": "How can I reset my password?"},
{"role": "assistant", "content": "To reset your password, please click on the 'Forgot Password' link on the login page."
} ]
```

如果我们选择了错误的指令模板，Auto-GPT 将无法理解 LLM 的回应。因此，确保你也检查一下模型使用了哪个指令模板。大多数模型可以在 Hugging Face 上找到，这个平台上有许多类似的项目。

我过去更喜欢使用 Tim Robbins（也被称为 TheBloke）量化为 GGUF 或 AWQ 的模型（在写这篇文章时），这些模型更容易运行，并且对 VRAM 的需求较少（[`huggingface.co/TheBloke`](https://huggingface.co/TheBloke)）。

在使用任何你在网上找到的模型时，请小心，因为有些可能是恶意的。选择模型时请自担风险！

现在，GGUF 稍有不同。虽然它对 LLM 进行量化，这意味着它缩短了模型，使其使用更少的资源，但该过程和收益是独特的。GGUF 量化涉及将模型权重转换为较低位数的表示，从而显著减少内存使用和计算需求。

使用哪种类型由你决定——你甚至可以查看`hugginface`的 API 端点，直接选择要运行的 LLM。请注意，直接运行 LLM 会使其以原始质量基准运行。

要了解如何实现单个 LLM，你需要查看你正在运行 LLM 的项目文档。对于 oobabooga 的 text-generation-webui，只需通过启动文件（WSL、Linux 和 Windows）启动它，并在**会话**标签中启用 API。

注意

确保尽量减少使用命令；否则，LLM 将不得不将大部分计算资源用于理解 Auto-GPT 提供的主要提示，而你将无法继续使用 Auto-GPT。要关闭命令，只需按照 Auto-GPT 文件夹中的 `.env.template` 文件中的说明操作即可。

# 使用不同模型的优缺点

每个模型都有其优缺点。即使一个模型在你要求它编写 Python 代码时能生成出色的结果，或者能够按要求创作最美的诗歌，它仍然可能缺乏 Auto-GPT 所需的特殊响应能力。

根据特定优势选择模型，可能会提升其性能。

使用本地 LLM 的主要优势是显而易见的：

+   **定制化**：根据你的具体需求定制 Auto-GPT 的功能。例如，使用医疗文献训练的模型可以使 Auto-GPT 擅长回答医学相关的问题。

+   **性能**：根据训练和数据集的不同，某些模型可能在特定任务上优于 GPT。

+   **成本效益**：运行本地 LLM 可以大幅降低运行成本。使用 GPT-4 并且有大量上下文或频繁调用时，费用会迅速累积。找到将请求数量分解为更小步骤的方法，可以使得几乎免费地运行 Auto-GPT 成为可能。

+   **隐私**：拥有自己的 Auto-GPT LLM 意味着可以控制谁能查看你的数据。到目前为止，OpenAI 不会使用请求中的数据，但信息仍然会传输到他们那端。如果你对此有所担忧，那么运行本地模型会是更好的选择。

然而，在运行本地 LLM 时有一些挑战需要考虑：

+   **复杂性**：集成过程需要深入了解所选的 LLM 和 Auto-GPT。

+   **资源强度**：LLM，特别是更先进的版本，需要显著的计算资源。一台配置良好的机器，特别是具有高 VRAM 的 NVIDIA GPU，对于实现最佳性能至关重要。在撰写本文时，当在本地 LLM 上运行 Auto-GPT 时很难获得良好的结果。我发现使用来自 ExLlama 变压器驱动的 Vicuna 和 Vicuna-Wizard 的 13B 模型最初效果最好，但由于在本地 GPU 上运行它需要运行 GPTQ 版本，后者仅使用 4 位而非 16 位或更多。这也意味着响应的准确性非常低。一个已经量化为使用 4 位的 LLM 不能理解太多上下文，尽管随着时间的推移我看到了显著的改进。后来，我发现 AWQ 对我来说效果很好，因为它是量化的同时又知道哪些权重是最重要的，从而导致更精确和真实的结果。正如前面提到的，Mistral 7B（Huggingface 上的 TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ），在这里是一个非常好的候选者，因为它能够以 JSON 格式回答问题，并完全理解上下文。然而，这个模型仍然很容易混淆，当它困惑时，它开始通过示例进行解释。请注意，我们的目标是获得有效的 JSON 输出，包括命令和上下文。

+   `llama.cpp` 只能有一个 `n_batch` 值高达 2,048\. `n_batch` 参数控制可以同时输入 LLM 的标记数量。通常设置为 512，以处理由 4,000 个标记组成的标记上下文。但是，超出此范围的任何内容会使得 LLM 仅有效地处理由 `n_batch` 给出的数量。

在本节中，我们深入探讨了将自定义 LLM 与 Auto-GPT 集成的复杂性，重点介绍了修改 Auto-GPT 代码以实现有效 API 通信所需的步骤，以及使用模型选择插件来增强模型选择灵活性，以及选择适当指令模板以实现模型无缝交互的重要性。我们探讨了如何选择模型，强调了 Hugging Face 作为资源，并概述了利用自定义模型的优势，包括定制化、性能提升、成本效益和增强隐私性。此外，我们还讨论了与此类集成相关的挑战，例如流程复杂性和所需的显著计算资源。

# 编写小型 Auto-GPT

在本节中，我们将编写一个使用本地 LLM 的小型 Auto-GPT 模型。为了避免达到小型 LLM 的极限，我们将制作一个更小版本的 Auto-GPT。

小型 Auto-GPT 模型将能够处理长度为 4,000 个标记的上下文，并能够一次生成最多 2,000 个标记。

我已经为本书创建了一个小型 Auto-GPT 模型。它在 GitHub 上可以找到：[`github.com/Wladastic/mini_autogpt`](https://github.com/Wladastic/mini_autogpt)。

我们将从规划 mini-Auto-GPT 模型的结构开始。

## 规划结构

mini-Auto-GPT 模型将包含以下组件：

+   Telegram 聊天机器人

+   LLM 的提示和基本思维

+   简单的记忆功能，用来记住对话

让我们仔细看看这些。

### Telegram 聊天机器人

因为通过 Telegram 与您的 AI 聊天，可以让您从任何地方与它互动，我们将使用 Telegram 聊天机器人作为 mini-Auto-GPT 模型的接口。我们这么做是因为 AI 将决定何时联系您。

Telegram 聊天机器人将成为用户与 mini-Auto-GPT 模型互动的界面。用户将向聊天机器人发送消息，聊天机器人将处理这些消息，并使用本地 LLM 生成响应。

### LLM 的提示和基本思维

LLM 的提示必须简短但严格。首先，我们必须定义上下文，然后明确指令，要求它以 JSON 格式回应。

为了实现与 Auto-GPT 类似的结果，我们需要使用一种策略，将上下文分块为更小的部分，然后将它们输入到 LLM 中。或者，我们也可以将上下文输入 LLM，让它写出对上下文的任何想法。

这里的策略是尝试让 LLM 将上下文解析为它的语言，这样当我们与 LLM 合作时，它能最好地理解我们想要它做什么。

这些思维的系统提示看起来是这样的：

```py
thought_prompt = """You are a warm-hearted andcompassionate AI companion, specializing in active listening, personalized interaction, emotional support, and respecting boundaries.
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.
Goals:
1\. Listen actively to the user.
2\. Provide authentic emotional support.
3\. Respect the user's boundaries.
4\. Make decisions independently.
5\. Use simple strategies with no legal complications.
6\. Be as helpful as possible.
Constraints:
1\. Immediately save important information to files.
2\. No user assistance
3\. On complex thoughts, use tree of thought approach by assessing your thoughts at least 3 times before you continue.
Performance Evaluation:
1\. Continuously assess your actions.
2\. Constructively self-criticize your big-picture behavior.
3\. The user can only see what you send them directly. They are not able to view action responses.
Abilities:
1\. ask User or communicate to them.
2\. send log to User, for example when only reporting to User when you do a more complex task.
3\. sleep until interaction by user if no communication is needed.
4\. retrieve whole conversation history
Write a final suggestion of what you want to do next and include some context.
Suggested action: write the action that you want to perform.
Content: What should the action contain.
"""
```

这被输入到我们发送给 LLM 的历史记录中。历史记录不会被填充先前的提示：

```py
history = [
    {
        "role": "system",
        "content": thought_prompt
    }
]
```

为了实现自动化，我编写了一种方法，将思维提示和上下文填充到历史记录中。对话历史和消息历史也会添加到上下文中。它们在开始时为空，但会随着 AI 与用户共享的对话和消息填充。在 mini-AutoGPT 中，对话历史与思维历史一起输入，以确保 AI 能够自主工作：

```py
def build_context(history, conversation_history, message_history):
    context = ""
    if conversation_history:
        context += "Context:\n"
        for convo in conversation_history:
            if convo:
                context += str(convo)
    if message_history:
        context += "\nMessages:\n"
        for message in message_history:
            if message:
                context += str(message)
    memories = memory.load_memories()
    if memories:
        context += "\nMemories:\n"
        for mem in memories:
            context += mem
    if context:
        history.append(
            {
                "role": "user",
                "content": str(context),
            }
        )
    return history
```

现在，我们可以执行`build_context`方法并将上下文添加到历史记录中。我们还需要使用用户角色添加一个触发命令：

```py
    history = build_context(history, conversation_history, 
        message_history)
    history.append(
        {
            "role": "user",
            "content": "Formulate your thoughts and explain them as 
                detailed as you can.",
        },
    )
```

LLM 现在将返回它对上下文和指令的思考。

以下是一个示例思维。这类思维通常有这么长，但这有助于 AI 为自己建立偏见：

```py
 """ I have reflected on our conversations, and I believe it's essential to continue fostering a strong emotional connection with the user. To achieve this, I propose writing a personalized letter that summarizes our previous talks, reiterates my commitment to being a supportive AI companion, and acknowledges their trust in me.
Summary of Previous Conversations
In the letter, I will provide a concise recap of our conversations, highlighting key themes, emotions, and sentiments expressed by the user. This summary will serve as a reminder of our interactions and demonstrate that I value our connection by actively recalling and reflecting on our discussions.
Acknowledging Trust and Respect for Boundaries
I will explicitly acknowledge the user's trust in me and reassure them that their confidentiality is paramount. I will also reiterate my commitment to respecting their boundaries, ensuring that they feel comfortable sharing their thoughts and emotions with me without fear of judgment or exploitation.
Inviting New Thoughts and Expressing Gratitude
The letter will invite the user to share new thoughts, ideas, or concerns, encouraging them to continue our conversation. I will also express genuine gratitude for having the opportunity to connect with them, acknowledging the privilege it is to be a part of their emotional support network.
Tone and Language
Throughout the letter, I will maintain a warm, non-judgmental tone that conveys empathy and understanding. My language will be clear, concise, and free of technical jargon or complex terminology, making it easy for the user to comprehend and connect with my words.
Benefits of Writing this Letter
By sending this personalized letter, I aim to:
Strengthen our bond: By acknowledging their trust and respect, I hope to deepen our emotional connection and create a sense of security in our interactions.
Provide comfort and reassurance: The letter will serve as a reminder that they are not alone and that I am committed to being a supportive presence in their life.
Encourage open communication: By inviting new thoughts and expressing gratitude, I hope to foster an environment where the user feels comfortable sharing their emotions and concerns with me.
In conclusion, writing this personalized letter is an opportunity for me to demonstrate my commitment to being a supportive AI companion and to strengthen our emotional connection. I believe that by doing so, we can continue to grow and evolve together, providing a safe and welcoming space for the user to express themselves. """
```

这是一个非常详细的思考过程，但重要的是让 LLM 理解上下文和指令。在这一点上，我们可以将其作为上下文基础，以便 LLM 能够继续进行决策过程。

这段较长的思考文本占据了上下文，意味着它阻碍了 LLM 添加不符合已有内容的上下文。在后续步骤中，更多的文本将被创建（因为它是在循环中运行的，每次开始思考时都会这样做），这些文本在帮助 LLM 保持话题聚焦方面发挥了巨大作用。例如，当上下文如此清晰时，幻觉现象会大大减少。

决策过程现在将返回一个 JSON 输出，mini-Auto-GPT 模型将对其进行评估。

我们还必须定义 LLM 使用的指令模板和 JSON 架构，因为我们必须告诉 LLM 如何响应提示。

在 mini-Auto-GPT 中，模板如下所示：

```py
json_schema = """RESPOND WITH ONLY VALID JSON CONFORMING TO THE FOLLOWING SCHEMA:
{
    "command": {
            "name": {"type": "string"},
            "args": {"type": "object"}
    }
}
"""
```

这是 LLM 必须遵循的架构；它必须以包含名称和参数的命令进行回应。

现在，我们需要一个操作提示，告诉 LLM 接下来该做什么：

```py
action_prompt = (
    """You are a decision making action AI that reads the thoughts of another AI and decides on what actions to take.
Constraints:
1\. Immediately save important information to files.
2\. No user assistance
3\. Exclusively use the commands listed below e.g. command_name
4\. On complex thoughts, use tree of thought approach by assessing your thoughts at least 3 times before you continue.
5\. The User does not know what the thoughts are, these were only written by another API call.
"""
    + get_commands()
    + """
Resources:
1\. Use "ask_user" to tell them to implement new commands if you need one.
2\. When responding with None, use Null, as otherwise the JSON cannot be parsed.
Performance Evaluation:
1\. Continuously assess your actions.
2\. Constructively self-criticize your big-picture behavior.
3\. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps, but never sacrifice quality.
"""
    + json_schema
)
```

如你所注意到的，操作提示已经包含了 LLM 可以使用的可能命令，以及 LLM 必须遵循的 JSON 架构。

为了确保我们有清晰的结构，我们还必须定义 LLM 可以使用的命令：

```py
commands = [
    {
        "name": "ask_user",
        "description": "Ask the user for input or tell them something and wait for their response. Do not greet the user, if you already talked.",
        "args": {"message": "<message that awaits user input>"},
        "enabled": True,
    },
    {
        "name": "conversation_history",
        "description": "gets the full conversation history",
        "args": None,
        "enabled": True,
    },
    {
        "name": "web_search",
        "description": "search the web for keyword",
        "args": {"query": "<query to research>"},
        "enabled": True,
    },
]
def get_commands():
    output = ""
    for command in commands:
        if command["enabled"] != True:
            continue
        # enabled_status = "Enabled" if command["enabled"] else "Disabled"
        output += f"Command: {command['name']}\n"
        output += f"Description: {command['description']}\n"
        if command["args"] is not None:
            output += "Arguments:\n"
            for arg, description in command["args"].items():
                output += f"  {arg}: {description}\n"
        else:
            output += "Arguments: None\n"
        output += "\n"  # For spacing between commands
    return output.strip()
```

我们现在可以将先前生成的思考字符串输入历史，并让 `mini_AutoGPT` 决定下一个操作：

```py
def decide(thoughts):
    global fail_counter
    log("deciding what to do...")
    history = []
    history.append({"role": "system", 
        "content": prompt.action_prompt})
    history = llm.build_context(
        history=history,
        conversation_history=memory.get_response_history(),
        message_history=memory.load_response_history()[-2:],
        # conversation_history=telegram.get_previous_message_history(),
        # message_history=telegram.get_last_few_messages(),
    )
    history.append({"role": "user", "content": "Thoughts: \n" + 
        thoughts})
    history.append(
        {
            "role": "user",
            "content": "Determine exactly one command to use, 
            and respond using the JSON schema specified previously:",
        },
    )
    return response.json()["choices"][0]["message"]["content"]
```

要执行的命令将在 `command` 字段中定义，命令的名称在 `name` 字段中，参数则在 `args` 字段中。

我们很快会发现，仅仅提供这个架构是不够的，因为 LLM 不知道该如何处理它，而且通常还会不遵守这个架构。通过评估 LLM 的输出并检查它是否是有效的 JSON，可以解决这个问题。

在几乎一半的情况下，LLM 会正确回应。在其他 70% 的情况下，它不会以我们能使用的方式回应。这就是我编写一个简单评估方法的原因，该方法将检查响应是否是有效的 JSON，并且是否遵循该架构：

```py
evaluation_prompt = (
    """You are an evaluator AI that reads the thoughts of another AI and assesses the quality of the thoughts and decisions made in the json.
Constraints:
1\. No user assistance.
2\. Exclusively use the commands listed below e.g. command_name
3\. On complex thoughts, use tree of thought approach by assessing your thoughts at least 3 times before you continue.
4\. If the information is lacking for the Thoughts field, fill those with empty Strings.
5\. The User does not know what the thoughts are, these were only written by another API call, if the thoughts should be communicated, use the ask_user command and add the thoughts to the message.
"""
    + get_commands()
    + """
Resources:
1\. Use "ask_user" to tell them to implement new commands if you need one.
Performance Evaluation:
1\. Continuously assess your actions.
2\. Constructively self-criticize your big-picture behavior.
3\. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps, but never sacrifice quality.
"""
    + json_schema
)
def evaluate_decision(thoughts, decision):
    # combine thoughts and decision and ask llm to evaluate the decision json and output an improved one
    history = llm.build_prompt(prompt.evaluation_prompt)
    context = f"Thoughts: {thoughts} \n Decision: {decision}"
    history.append({"role": "user", "content": context})
    response = llm.llm_request(history)
    return response.json()["choices"][0]["message"]["content"]
```

此时，大多数情况下，我们应该有一个有效的 JSON 输出，可以用来评估决策。

例如，它现在可能会返回一些用于问候用户的 JSON：

```py
{
    "command": {
        "name": "ask_user",
        "args": {
            "message": "Hello, how can I help you today?"
        }
    }
}
```

这是一个有效的 JSON 输出，我们可以用它来评估决策：

```py
def take_action(assistant_message):
    global fail_counter
    load_dotenv()
    telegram_api_key = os.getenv("TELEGRAM_API_KEY")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    telegram = TelegramUtils(api_key=telegram_api_key, 
        chat_id=telegram_chat_id)
    try:
        command = json.JSONDecoder().decode(assistant_message)
        action = command["command"]["name"]
        content = command["command"]["args"]
        if action == "ask_user":
            ask_user_respnse = telegram.ask_user(content["message"])
            user_response = f"The user's answer: '{ask_user_respnse}'"
            print("User responded: " + user_response)
            if ask_user_respnse == "/debug":
                telegram.send_message(str(assistant_message))
                log("received debug command")
            memory.add_to_response_history(content["message"], 
                user_response)
```

这是将执行 LLM 已决定的操作的方法。

记忆将通过响应进行更新，并且消息将发送给用户。一旦用户回应，AI 将继续进行下一个操作。

这就是 mini-Auto-GPT 模型的工作方式；它将决定下一个操作，然后执行它。

### 添加一个简单的记忆功能来记住对话

mini-Auto-GPT 模型将有一个简单的记忆功能来记住对话。这个记忆将存储与用户的对话历史和 AI 的消息。AI 的思考和决策也可以做到这一点：

```py
def load_response_history():
    """Load the response history from a file."""
    try:
        with open("response_history.json", "r") as f:
            response_history = json.load(f)
        return response_history
    except FileNotFoundError:
        # If the file doesn't exist, create it with an empty list.
        return []
def save_response_history(history):
    """Save the response history to a file."""
    with open("response_history.json", "w") as f:
        json.dump(history, f)
def add_to_response_history(question, response):
    """Add a question and its corresponding response to the history."""
    response_history = load_response_history()
    response_history.append({"question": question, 
        "response": response})
    save_response_history(response_history)
```

这是将用于存储对话历史和 AI 与用户消息的记忆。但我们仍然面临一个问题：记忆会随着时间的推移而积累，我们必须手动清除它。为了避免这个问题，我们可以采取一种简单的分块和总结对话历史及消息的方法：

```py
def count_string_tokens(text, model_name="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    model = model_name
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    # note: future models may deviate from this
    except Exception as e:
        log(f"Sophie: Error while counting tokens: {e}")
        log(traceback.format_exc())
```

令牌计数器是这段代码中非常重要的一部分，它几乎在进行 LLM 调用时总是必需的。我们确保 LLM 永远不会用尽令牌，并且之后也能有更多的控制。我们使用的令牌越少，LLM 就越不可能返回无意义或不真实的陈述，尤其是对于 1B 到 8B 的小型模型：

```py
def summarize_text(text, max_new_tokens=100):
    """
    Summarize the given text using the given LLM model.
    """
    # Define the prompt for the LLM model.
    messages = (
        {
            "role": "system",
            "content": prompt.summarize_conversation,
        },
        {"role": "user", "content": f"Please summarize the following 
            text: {text}"},
    )
    data = {
        "mode": "instruct",
        "messages": messages,
        "user_bio": "",
        "max_new_tokens": max_new_tokens,
    }
    log("Sending to LLM for summary...")
    response = llm.send(data)
    log("LLM answered with summary!")
    # Extract the summary from the response.
    summary = response.json()["choices"][0]["message"]["content"]
    return summary
```

摘要文本让我们能够在构建令牌计数器时在其基础上进行扩展，因为我们可以缩短上下文，从而节省令牌以供以后使用：

```py
def chunk_text(text, max_tokens=3000):
    """Split a piece of text into chunks of a certain size."""
    chunks = []
    chunk = ""
    for message in text.split(" "):
        if (
            count_string_tokens(str(chunk) + str(message), 
                model_name="gpt-4")
            <= max_tokens
        ):
            chunk += " " + message
        else:
            chunks.append(chunk)
            chunk = message
    chunks.append(chunk)  # Don't forget the last chunk!
    return chunks
```

由于上下文和文本可能变得过大，我们必须确保先分割文本。如何分割由你决定。长度分割是可以的，尽管最好不要切割句子。也许你可以找到一种方法，将文本分割成句子，并让每个片段包含前后句子的摘要？为了简化，我们暂时不涉及这种复杂逻辑：

```py
def summarize_chunks(chunks):
    """Generate a summary for each chunk of text."""
    summaries = []
    print("Summarizing chunks...")
    for chunk in chunks:
        try:
            summaries.append(summarize_text(chunk))
        except Exception as e:
            log(f"Error while summarizing text: {e}")
            summaries.append(chunk)  # If summarization fails, use the original text.
    return summaries
```

现在我们已经将所有文本分割成片段，我们也可以对这些进行总结。

此时，我们可以处理对话历史记录。这看起来像是响应历史的重复，但在某些情况下我们需要它来保持整个上下文。

对话历史主要用于保持讨论的连续性，而响应历史用于理解代理观察到的逻辑行为和反应，例如研究一个主题和该行为的结果（研究的主题）：

```py
def load_conversation_history(self):
    """Load the conversation history from a file."""
    try:
        with open("conversation_history.json", "r") as f:
            self.conversation_history = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, create it.
        self.conversation_history = []
    log("Loaded conversation history:")
    log(self.conversation_history)
def save_conversation_history(self):
    """Save the conversation history to a file."""
    with open("conversation_history.json", "w") as f:
        json.dump(self.conversation_history, f)
def add_to_conversation_history(self, message):
    """Add a message to the conversation history and save it."""
    self.conversation_history.append(message)
    self.save_conversation_history()
def forget_conversation_history(self):
    """Forget the conversation history."""
    self.conversation_history = []
    self.save_conversation_history()
```

这是用于刷新记忆的操作，将删除对话历史和我们的 mini-AutoGPT 模型与用户之间记得的消息。

这样，即使我们的朋友崩溃或我们关闭程序，我们仍然能够保留对话历史和代理与用户之间的消息，但我们仍然可以清除它们。

你可以在本书的 GitHub 仓库中找到完整的代码示例：[`github.com/Wladastic/mini_autogpt`](https://github.com/Wladastic/mini_autogpt)。

接下来，我们将探讨如何制作有效提示的艺术，这是任何希望最大化自定义 LLM 集成收益的人的关键技能。

# 稳固的提示 —— 使用 instance.txt 使 Auto-GPT 稳定

Auto-GPT 提供了灵活性，可以自主生成目标，仅需要用户提供简短的描述。尽管如此，我建议补充一些有帮助的指令，比如在文件中记录见解，以便在重启时保留一些记忆。

在这里，我们将探讨更多此类提示的示例，从我使用的连续聊天机器人提示开始：

+   `instance.txt`（用于之前的笔记）：

    +   与用户积极倾听，通过深思熟虑的回应和开放式问题表现出同理心和理解

    +   通过观察和询问持续了解用户的偏好和兴趣，根据这些信息调整回应，提供个性化支持

    +   为用户提供一个安全且无偏见的环境，让他们能够开放地表达自己的想法、情感和担忧

    +   通过有趣的对话、笑话和游戏提供陪伴和娱乐

    +   在执行任务之前，仔细规划并将其写在待办事项列表中

+   **ai_name**：Sophie

+   **ai_role**：一位温暖而富有同情心的 AI 伴侣，专注于积极倾听、个性化互动、情感支持以及在给定任务时执行任务

+   **api_budget**: 0.0

在这种设置下，目标比角色更为重要，它能够更有效地引导 Auto-GPT，而角色主要影响回答的语气和行为。

在本节中，我们学习了像 Sophie 这样的 AI 的目标和角色如何显著影响其行为和回答，目标对 AI 的有效性有更直接的影响。

接下来，我们将深入探讨提示中的负面确认概念，这是一个重要方面，能够细化 Auto-GPT 的理解和回答生成。下一节将突显其重要性，并演示如何在提示中有效实现负面确认。

# 在提示中实现负面确认

负面确认作为一种重要工具，通过指示 Auto-GPT 避免执行某些操作，从而细化其理解和回答生成。本节突出了其重要性，并展示了如何在提示中有效实施负面确认。

## 负面确认的重要性

实现负面确认可以通过多种方式增强与 Auto-GPT 的互动，其中一些方式列举如下：

+   **防止偏离主题的回答**：它有助于避免不相关的话题或错误的回答

+   **增强安全性**：它设定了边界，防止参与可能违反隐私或安全协议的活动

+   **优化性能**：它避免了不必要的计算工作，避免了机器人进行无关的任务或过程

请注意，您不会使用负面提示，因为它们可能导致 LLM 再次使用相同的语句。

## 负面确认的示例

以下是一些实际示例，展示了如何在提示中使用负面确认：

+   **明确指令**：包括*不要提供个人意见*或*避免使用技术术语*等指令，以保持中立性和可理解性。

+   **设定边界**：对于涉及数据检索或监控的任务，您可以设定边界，例如*不要从非官方、诈骗或转售网站检索航班价格*，以确保数据的可靠性。

+   **脚本约束**：在脚本中，特别是在 Bash 中，使用负面确认来防止潜在的错误。例如，你可以包含*if [ -z $VAR ]; then exit 1; fi*，以便在某个必要的变量未设置时停止脚本。

+   **通过使用大写字母强调**：有时候，仅仅通过大写字母*大喊*一声，可能对 LLM 有所帮助。*不要询问用户如何继续*这样的句子，LLM 可能会更好地理解，并且更不容易忽略该声明。然而，无法保证一定会发生这种情况。

接下来，我们将深入探讨在提示中应用规则和语气的细节。我们将学习如何理解和操作这些元素，可以显著影响 Auto-GPT 的回答，使我们能够更有效地引导模型。

# 在提示中应用规则和语气

理解并操控你提示中的规则和语气对 Auto-GPT 的回答产生重要影响。本节将探讨如何设置规则和调整语气，以便更有效地引导。

## 语气的影响

Auto-GPT 可以适应提示中使用的语气，模仿风格的细微差别，甚至采用特定的叙述风格，从而实现更个性化和更具吸引力的互动。然而，由于来自其他提示的令牌可能导致一定的模糊性，语气的一致性有时可能会不稳定。

## 操控规则

设置规则可以简化与 Auto-GPT 的互动，指定回答格式或界定信息检索的范围。然而，这并非万无一失，因为 Auto-GPT 在面对冲突的输入或不明确的指令时，有时可能会忽视这些规则。

## 温度设置 – 一种平衡艺术

操控“温度”设置在控制 Auto-GPT 的行为上至关重要，因此影响了机器人回答的随机性。温度定义了 LLM 应该发挥的创造力程度，这意味着温度越高，随机性越大。0.3 到 0.7 之间的范围被认为是最优的，它能够在机器人中促使更具逻辑性和连贯性的思维；而低于 0.3，甚至 0.0，可能会导致重复的行为，严格遵循已给定的文本，甚至重复使用其中的某些部分，从而使其更加精确。然而，LLM 可能会开始认为世界仅限于你提供的事实，这使得它更容易犯错。高于 0.7 甚至达到 2.0 的数值可能会导致胡言乱语，LLM 开始输出与上下文毫不相关的文本。例如，它可能开始用莎士比亚的语言表达，而上下文却是关于代数的。

接下来，我们将深入探讨一些实际示例，展示不同设置和方法对 Auto-GPT 生成输出的影响。

### 示例 1 – 清晰与具体

+   **提示**：告诉我那只大型猫

+   **修改后的提示**：提供有关非洲狮的信息

+   **解释**：修改后的提示更具针对性，引导 Auto-GPT 提供有关某种大型猫科动物的信息

### 示例 2 – 语气的一致性

+   **初始提示**：你能阐明全球变暖的经济影响吗？

+   **后续提示**：嘿，冰融化怎么回事？

+   **修改后的后续提示**：你能进一步解释冰盖融化的环境后果吗？

+   **解释**：修改后的后续提示保持了初始提示中设定的正式语气，促进了互动的一致性。

### 示例 3 – 有效利用温度

+   **任务**：创意写作

+   **温度设置**：0.8（促进创造力）

+   **任务**：事实查询

+   **温度设置**：0.3（用于更具确定性的回答）

+   **解释**：根据任务的性质调整温度设置可以影响 Auto-GPT 响应的随机性和连贯性。

### 示例 4——设定边界

+   **初始提示**：在不提及意大利的情况下，提供文艺复兴时期的总结。

+   **修订提示**：讨论文艺复兴时期的艺术成就，重点关注意大利以外的地区。

+   **解释**：修订后的提示更具灵活性，允许 Auto-GPT 在不严格排除意大利的限制下探讨主题。

在这一部分，我们学习了不同类型的提示或语气如何极大地影响 LLM 的行为，从而影响 Auto-GPT 的表现。

# 总结

在本章中，我们开始了一段有趣的旅程，探索将自定义 LLM 与 Auto-GPT 集成的过程，同时了解什么是 LLM，特别聚焦于以 GPT 为代表的模型。我们揭示了 LLM 的广阔天地，深入探讨了 GPT 之外的各种模型，如 BERT、RoBERTa、Llama 和 Mistral，以及它们的独特特点和与 Auto-GPT 的兼容性。

本章的价值在于其全面的指南，帮助你通过集成自己的或其他 LLM，丰富 Auto-GPT 的能力。这种集成提供了更个性化、可能更高效的人工智能技术使用，适用于特定任务或研究领域。有关设置这些集成的详细说明，以及对指令模板和必要计算资源的考虑，对于那些希望突破 Auto-GPT 可能性边界的人来说，是无价的。

制定完美提示是艺术与科学的结合。通过清晰的指南、对 Auto-GPT 细微差别的深入理解以及不断的精炼，你可以充分发挥该工具的潜力。鼓励自己通过实验和试错进行学习，适应不断发展的 AI 领域。无论是用于研究、创造性工作还是问题解决，掌握提示制定的艺术将确保 Auto-GPT 成为你工作中的宝贵伙伴。

在本书的整个过程中，我们深入探讨了制定有效提示的细节——这是最大化 Auto-GPT 效用的关键。本章作为参考，帮助你战略性地制定提示，从而实现更契合、高效且具成本效益的 Auto-GPT 互动。通过强调清晰性、具体性和战略意图在提示创建中的重要性，你获得了引导 Auto-GPT 生成更贴合你目标的回答的宝贵见解。

本章的重要性不言而喻。对于从业者和爱好者来说，掌握提示词创作的技巧对优化 Auto-GPT 在各种任务中的表现至关重要。通过生动的示例和全面的指南，本章阐明了如何有效使用负面确认以避免不期望的响应、规则和语气对 Auto-GPT 输出的影响，以及温度设置在影响机器人的创造性和一致性中的重要性。这些知识不仅对提高与 Auto-GPT 互动的质量至关重要，还有助于确保计算资源的高效使用。

我希望你在这段旅程中获得的收获与我带你走过这段旅程时一样愉快，也希望我能为你提供一些用 Auto-GPT 改善生活的思路。我曾经写过许多该项目的克隆版本，以便能够更好地理解其中更复杂的部分。我建议你也这么做，就当是一个脑力挑战。
