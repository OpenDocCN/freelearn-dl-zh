

# 第八章：使用 WhisperX 和 NVIDIA 的 NeMo 进行语音分段

欢迎来到*第八章*，在这里我们将探索**语音分段**的世界。尽管 Whisper 已被证明是一个强大的语音转录工具，但语音分析中还有另一个至关重要的方面——说话人分段，这可以显著增强其效用。通过增强 Whisper 的能力，识别并将语音片段归属于不同的说话人，我们为分析多说话人对话开辟了新的可能性。本章将探讨如何将 Whisper 与最先进的分段技术结合，以解锁这些能力。

我们将从探索说话人分段系统的演变开始，了解早期方法的局限性，以及变压器模型带来的变革性影响。通过实际的操作示例，我们将预处理音频数据，使用 Whisper 转录语音，并微调转录与原始音频之间的对齐。

在本章中，我们将涵盖以下主要主题：

+   增强 Whisper 的说话人分段能力

+   执行实际的语音分段

本章结束时，您将了解如何将 Whisper 与先进的技术（如语音活动检测、说话人嵌入提取和聚类）集成，从而增强其功能，并实现最先进的分段性能。您还将学习如何利用 NVIDIA 强大的**多尺度分段解码器**（**MSDD**）模型，该模型考虑了说话人嵌入的多个时间分辨率，以提供卓越的准确性。通过掌握本章中介绍的技术，您将能够应对复杂的多说话人音频场景，推动 OpenAI Whisper 的可能性极限。

准备好深入探索说话人分段的激动人心的世界，并从多说话人对话中获得新的见解吧！让我们一起开始这段变革之旅！

# 技术要求

为了利用 OpenAI 的 Whisper 实现高级应用，本章使用 Python 和 Google Colab，便于使用和访问。Python 环境设置包括用于转录任务的 Whisper 库。

**关键要求**：

+   **Google Colab 笔记本**：笔记本设置为使用最低要求的内存和容量运行我们的 Python 代码。如果可用，请选择**T4 GPU**运行类型以获得更好的性能。

+   **Google Colab 笔记本**：笔记本设置为使用最低要求的内存和容量运行我们的 Python 代码。如果可用，请将运行时类型更改为**GPU**以获得更好的性能。

+   **Python 环境**：每个笔记本包含指令，用于加载所需的 Python 库，包括 Whisper 和 Gradio。

+   **Hugging Face 帐户**：某些笔记本需要 Hugging Face 帐户和登录 API 密钥。Colab 笔记本中包含有关此主题的信息。

+   **麦克风和扬声器**：一些笔记本实现了一个带有语音录制和音频播放功能的 Gradio 应用程序。连接到计算机的麦克风和扬声器可以帮助您体验互动语音功能。另一种选择是在运行时打开 Gradio 提供的 URL 链接，在您的手机上使用手机麦克风录制您的声音。

+   **GitHub 仓库访问**：所有 Python 代码，包括示例，都可以在本章的 GitHub 仓库中找到（[`github.com/PacktPublishing/Learn-OpenAI-Whisper/tree/main/Chapter08`](https://github.com/PacktPublishing/Learn-OpenAI-Whisper/tree/main/Chapter08)）。这些 Colab 笔记本已准备好运行，提供了一种实用且动手的学习方法。

通过满足这些技术要求，您将为在不同情境中探索 Whisper 做好准备，同时享受 Google Colab 带来的流畅体验以及 GitHub 上提供的全面资源。

# 使用说话人分离增强 Whisper

说话人分离是将音频流按说话人的身份划分为不同的段落，是多说话人语音处理中的一个强大功能。它解决了*谁在什么时候说话？*的问题。在给定的音频片段中，增强 ASR 系统的功能性和可用性至关重要。说话人分离的起源可以追溯到 1990 年代，当时为基于聚类的分离范式奠定了基础。这些早期的研究主要集中在广播新闻和通信应用，旨在提高 ASR 性能。早期研究中使用的特征大多是手工设计的，其中**Mel 频率倒谱系数**（**MFCCs**）是常见的选择。

随着时间的推移，说话人分离领域取得了显著的进展，特别是在深度学习技术的出现之后。现代分离系统通常利用神经网络和大规模 GPU 计算来提高准确性和效率。分离技术的发展包括早期方法中使用**高斯混合模型**（**GMMs**）和**隐马尔可夫模型**（**HMMs**），以及近年来采用神经嵌入（如*x*-向量和*d*-向量，我们将在本章稍后的*说话人嵌入介绍*部分中详细介绍）和聚类方法。

对该领域最重要的贡献之一是端到端神经分离方法的发展，这些方法通过将分离流程中的不同步骤合并，简化了分离过程。这些方法旨在处理多说话人标注和分离中的挑战，例如处理嘈杂的声学环境、不同的音色和口音差异。

开源项目也为说话人分离能力的演变做出了贡献，工具如 ALIZE、pyannote.audio、pyAudioAnalysis、SHoUT 和 LIUM SpkDiarization 为研究人员和开发人员提供了资源，以便在他们的应用程序中实现和实验说话人分离。大多数早期工具现在已经不再活跃或被遗弃，只有 pyannote.audio（Pyannote）仍在使用。

早期的说话人分离系统虽然在解决音频录音中*谁在何时说话*的问题上具有开创性，但也面临着若干局限性，这些局限性影响了它们的准确性和效率。在下一节中，我们将更详细地探讨早期说话人分离解决方案的根本性障碍。

## 理解说话人分离的局限性和约束

早期说话人分离技术中的许多不足和不准确性源于当时的技术约束、人类语音的复杂性以及应用于音频处理的机器学习技术尚处于初步阶段。理解这些局限性为我们提供了对说话人分离能力演变的宝贵见解，并帮助我们认识到随着时间推移所取得的重大进展：

+   **计算限制**：早期的说话人分离系统受限于当时可用的计算能力。处理大规模音频数据集需要大量的计算资源，这些资源在当时并不像今天这样普遍和强大。这一限制影响了可以在合理时间内运行的算法的复杂度，从而限制了早期说话人分离系统的准确性。

+   **特征提取和建模的局限性**：早期的说话人分离系统中使用的特征提取技术，如 MFCCs，相较于现代系统中使用的复杂嵌入技术，要简单得多。这些早期的特征可能无法有效捕捉不同说话人声音的细微差别，从而导致说话人区分不够准确。

+   **依赖 GMM 和 HMM 进行说话人建模**：尽管这些模型为说话人分离提供了基础，但它们在处理不同说话人和环境下人类语音的变化性和复杂性时存在局限性。

+   **处理说话者变化点**：早期分离系统面临的一个重大挑战是准确检测说话者变化点。这些系统尤其在处理短语音段和接近说话者变化点的语音段时遇到了困难。随着语音段时长的减少以及与变化点的接近，系统的表现会有所下降。例如，在**单一远程麦克风**（**SDM**）和**多重远程麦克风**（**MDM**）的条件下，所有评估的系统中超过 33%和 40%的错误发生在变化点前后 0.5 秒内。SDM 指的是在离说话者一定距离处放置一个麦克风，用来捕捉所有参与者的音频。而 MDM 则是在录音环境中不同位置放置多个麦克风，提供额外的空间信息，从而可以提升分离性能。这些设置下的错误百分比突显了早期分离系统在准确检测说话者变化时，尤其是在变化点附近，所面临的挑战。

+   **可扩展性和灵活性**：早期的分离系统通常是针对特定应用设计的，如广播新闻或会议录音，可能无法快速适应其他类型的音频内容。这种缺乏灵活性限制了分离技术的广泛应用。此外，这些系统在处理大规模或实时分离任务时的可扩展性也是一个重大挑战。

+   **错误分析和改进方向**：对早期分离系统的深入错误分析表明，改善说话者变化点附近的处理可以显著提升整体性能。为解决这些限制，探索了替代最小持续时间约束以及利用最显著和第二大对数似然得分之间的差异进行无监督聚类等改进措施。

尽管早期的说话者分离方法取得了开创性进展，但它们仍面临各种限制，这些限制本可以提高它们的准确性和效率。这些限制来源于技术约束、人类语音的复杂性以及机器学习技术的初期阶段。然而，引入基于变压器的模型彻底改变了这一领域，解决了许多挑战，为更准确高效的解决方案铺平了道路。

## 将变压器引入语音分离

变换器在推动最先进的语音分离技术方面起到了关键作用。它们擅长处理语音的序列性和上下文性，这对于在音频流中区分说话人至关重要。变换器中的自注意力机制使得模型能够权衡输入数据中每个部分的重要性，这对于识别说话人切换点并将语音段归属到正确的说话人至关重要。

如前所述，传统的语音分离方法通常依赖于高斯混合模型（GMMs）和隐马尔可夫模型（HMMs）来建模说话人的特征。这些方法需要改进，以应对人类语音的变化性和复杂性。相比之下，基于变换器的语音分离系统可以同时处理整个数据序列，从而更有效地捕捉语音段之间的上下文和关系。

变换器还启用了嵌入表示，如*x*-向量和*d*-向量，它们提供了更细致的说话人特征表示。这有助于提高语音分离的性能，特别是在具有挑战性的声学环境或有重叠语音的场景中。

超越早期语音分离方法的局限性，我们必须引入一种颠覆性的框架，将变换器（transformers）引入语音分离——NVIDIA 的**神经模块**（**NeMo**）。NeMo 是一个开源工具包，用于构建、训练和微调 GPU 加速的语音和自然语言处理（NLP）模型。它提供了一组预构建的模块和模型，可以快速组合以创建复杂的 AI 应用程序，如自动语音识别（ASR）、自然语言理解和文本到语音合成。NeMo 通过其基于变换器的管道为语音分离提供了更直接的方法，开启了说话人识别和分离的新可能性。

## 引入 NVIDIA 的 NeMo 框架

与传统方法相比，基于变换器的语音分离系统提供了更优的性能，并更适应自然语音的复杂性。NVIDIA 的 NeMo 工具包支持训练和微调说话人语音分离模型。NeMo 利用基于变换器的模型处理各种语音任务，包括语音分离。该工具包提供了一个管道，其中包括**语音活动检测**（**VAD**）、**说话人嵌入提取**和**聚类**模块，这些模块是语音分离系统的关键组成部分。NeMo 的语音分离方法涉及训练能够捕捉未见过说话人特征的模型，并将音频段分配到正确的说话人索引。

从更全面的角度来看，NVIDIA 的 NeMo 提供的功能远超基于变换器的说话人分离。NeMo 是一个端到端的、云原生的框架，用于在各种平台上构建、定制和部署生成性 AI 模型，包括大型语言模型（LLMs）。它为整个生成性 AI 模型开发生命周期提供了全面的解决方案，从数据处理、模型训练到推理。NeMo 特别以其在对话 AI 方面的能力而著称，涵盖了自动语音识别（ASR）、自然语言处理（NLP）和文本转语音合成。

NeMo 以其处理大规模模型的能力脱颖而出，支持训练具有数万亿参数的模型。诸如张量并行、流水线并行和序列并行等先进的并行化技术为此提供了支持，使得模型可以在成千上万的 GPU 上高效扩展。该框架建立在 PyTorch 和 PyTorch Lightning 之上，为研究人员和开发人员提供了一个熟悉的环境，在对话 AI 领域进行创新。

NeMo 的一个关键特点是其模块化架构，在这种架构中，模型由具有强类型输入和输出的神经模块组成。这种设计促进了重用性，并简化了新对话 AI 模型的创建，允许研究人员利用现有代码和预训练模型。

NeMo 作为开源软件可供使用，鼓励社区的贡献，并促进了广泛的采用和定制。它还与 NVIDIA 的 AI 平台集成，包括 NVIDIA Triton 推理服务器，以便在生产环境中部署模型。NVIDIA NeMo 提供了一个强大而灵活的框架，用于开发最先进的对话 AI 模型，提供的工具和资源可以简化将生成性 AI 应用从概念到部署的全过程。

现在我们已经分别探讨了 Whisper 和 NeMo 的能力，接下来我们来考虑将这两种强大工具集成的潜力。将 Whisper 的转录能力与 NeMo 的先进说话人分离功能结合，可以从音频数据中解锁更多的洞见。

## 集成 Whisper 与 NeMo

尽管 Whisper 主要以其转录能力而闻名，但它也可以适应说话人分离任务。然而，Whisper 并不原生支持说话人分离。为了实现 Whisper 的说话人分离，需要借助如 Pyannote 这样的说话人分离工具包，结合 Whisper 的转录结果来识别说话人。

将 NVIDIA 的 NeMo 与 OpenAI 的 Whisper 结合进行说话人分离，涉及到一个创新的流程，利用两个系统的优势来增强分离效果。这种集成在推理和结果解读方面尤其值得注意。

该管道首先由 Whisper 处理音频以生成高精度的转录。Whisper 的主要角色是转录音频，提供详细的口语内容文本输出。然而，Whisper 本身不支持说话人分离（diarization）——识别音频中*谁在什么时候说话*。

为了引入分离，管道集成了 NVIDIA 的 NeMo，特别是它的说话人分离模块。NeMo 的分离系统旨在处理音频录音，通过说话人标签进行分段。它通过多个步骤实现这一目标，包括语音活动检测（VAD）、说话人嵌入提取和聚类。说话人嵌入捕捉独特的声音特征，然后通过聚类区分音频中的不同说话人。

Whisper 和 NeMo 在分离中的集成使你能够将 Whisper 的转录与 NeMo 识别的说话人标签对齐。这意味着输出不仅包括说了什么（来自 Whisper 的转录），还识别了每一部分由哪位说话人说（来自 NeMo 的分离）。结果是对音频内容有更全面的理解，既提供文本转录，又提供说话人的归属。

这种集成在理解对话动态至关重要的场景中非常有用，例如会议、访谈和法律程序。通过为转录添加一层特定于说话人的上下文，它增强了转录的实用性，使得更容易跟随对话并准确归属发言。

Whisper 和 NeMo 的说话人分离集成将 Whisper 的先进转录能力与 NeMo 强大的分离框架结合在一起。这种协同作用通过提供详细的转录和准确的说话人标签，增强了音频内容的可解释性，从而为口语互动提供更丰富的分析。

在深入探讨 Whisper 和 NeMo 的集成之前，了解现代语音处理系统中的一个基本概念——**说话人嵌入**至关重要。这些说话人特征的向量表示对于实现准确的说话人分离至关重要。

## 说话人嵌入简介

说话人嵌入（Speaker embeddings）是从语音信号中提取的向量表示，能够以紧凑的形式 encapsulate 说话人声音的特征。这些嵌入被设计为具有辨别性，意味着它们能够有效地区分说话人，同时对语音内容、通道和环境噪音的变化具有鲁棒性。目标是从可变长度的语音话语中获取固定长度的向量，捕捉说话人声音的独特特征。

说话人嵌入是现代语音处理系统的基础组件，支持从说话人验证到分离等多种应用。它们能够将说话人声音的丰富信息压缩成固定长度的向量，这使得它们在需要识别、区分或追踪不同说话人的音频记录系统中具有不可替代的价值。

从更技术的角度来看，有几种类型的说话人嵌入，每种都有其提取方法和特征：

+   *i*-vectors：这些嵌入在低维空间中捕捉说话人和通道的变异性。它们源自 GMM 框架，并表示给定说话人的发音与一组语音类别之间的平均发音的差异。

+   *d*-vectors：这些向量通过训练一个说话人区分的**深度神经网络**（**DNN**）并从最后的隐藏层提取帧级向量得到。这些向量随后在整个话语中进行平均，产生*d*-vector，代表说话人的身份。

+   *x*-vectors：这种类型的嵌入涉及帧级和段级特征（话语）处理。*X*-vectors 通过一个 DNN 提取，该 DNN 处理一系列声学特征并对它们进行聚合，使用统计池化层生成固定长度的向量。

+   *s*-vectors：也称为序列向量或摘要向量，*s*-vectors 来源于递归神经网络架构，如 RNN 或 LSTM。它们旨在捕捉顺序信息，可以在相当程度上编码口语词汇和单词顺序。

提取说话人嵌入通常涉及训练一个神经网络模型，通过优化编码器并使用鼓励区分学习的损失函数来实现。训练完成后，提取段级网络中隐藏层的前激活值作为说话人嵌入。该网络在一个包含大量说话人的数据集上进行训练，以确保嵌入能够很好地泛化到未见过的说话人。

在说话人分离的背景下，说话人嵌入根据说话人的身份将语音段进行聚类。嵌入提供了一种测量段之间相似性的方法，并将这些段与可能来自同一说话人的段群体进行分组。这是分离过程中的关键步骤，因为它允许你在音频流中准确地将语音归属到正确的说话人。

正如我们所见，增强了 Pyannote 的 Whisper 和 NVIDIA 的 NeMo 都提供了强大的分离能力。然而，理解这些方法之间的关键区别至关重要，这样才能在选择分离解决方案时做出明智的决定。

## 区分 NVIDIA 的 NeMo 能力

分话功能的集成对 ASR 系统的影响受到变换器模型（transformer models）出现的显著推动，特别是在 OpenAI 的 Whisper 和 NVIDIA 的 NeMo 框架的背景下。这些进展提升了 ASR 系统的准确性，并引入了处理分话任务的新方法。我们将深入探讨使用 Pyannote 的 Whisper 分话与使用 NVIDIA NeMo 的分话之间的相似性与差异，重点关注语音活动检测、说话人变化检测和重叠语音检测。了解这两种分话方法之间的差异，对于在选择适合自己特定使用案例的解决方案时做出明智决策至关重要。通过研究每个系统如何处理分话过程中的关键环节，如语音活动检测、说话人变化检测和重叠语音检测，您可以更好地评估哪种方法与您的准确性、效率和集成需求最为契合：

| **分话功能** | **Whisper** **与 Pyannote** | **NVIDIA NeMo** |
| --- | --- | --- |
| **检测** **语音活动** | Whisper 本身并不执行语音活动检测（VAD）作为其分话任务的一部分。然而，当与 Pyannote 结合使用时，来自 Pyannote 工具包的外部 VAD 模型可以在应用分话之前将音频分割为语音和非语音区间。这种方法需要将 Whisper 的语音识别（ASR）能力与 Pyannote 的 VAD 模型相结合，基于深度学习技术和微调，实现精确的语音/非语音分割。 | NeMo 的分话流程包括一个专门的 VAD 模块，该模块是可训练和优化的，作为分话系统的一部分。此 VAD 模型旨在检测语音的存在或缺失，并生成语音活动的时间戳。将 VAD 集成到 NeMo 的分话流程中，可以实现更加简化的过程，直接将 VAD 结果传递到后续的分话步骤中。 |
| **检测** **说话人变化** | Whisper 与 Pyannote 集成以执行分话任务时，依赖于 Pyannote 的说话人变化检测能力。Pyannote 使用神经网络模型来识别音频中发生说话人变化的点。这个过程对于将音频分割成归属于各个说话人的同质段落至关重要。Pyannote 中的说话人变化检测是一个独立模块，与其分话流程一起工作。 | NeMo 的说话人变化检测方法隐式地在其分话流程中处理，包括用于提取和聚类说话人嵌入的模块。虽然 NeMo 没有明确提到独立的说话人变化检测模块，但通过分析说话人嵌入及其在音频中的时间分布，识别说话人变化已集成到整体分话工作流中。 |
| **检测** **重叠语音** | 重叠语音检测是 Pyannote 补充 Whisper 功能的另一个领域。Pyannote 的工具包包括设计用于检测和处理重叠语音的模型，这是说话人分离中一个具有挑战性的方面。这个功能对于准确地分离多个说话人同时发言的对话至关重要。 | 与说话人变化检测类似，NeMo 对重叠语音的处理被集成到其分离管道中，而不是通过单独的模块来解决。该系统处理重叠语音的能力来源于其复杂的说话人嵌入和聚类技术，即使在挑战性的重叠情境下，也能识别和分离说话人。 |
| **将说话人嵌入集成到** **分离管道** | Whisper 与 Pyannote 的结合依赖于外部模块来完成这些任务，提供了灵活性和模块化。而 NeMo 的分离管道直接集成了这些功能，提供了一个简化而连贯的工作流程。这些进展凸显了变换器模型对语音处理的变革性影响，为更精确高效的分离系统铺平了道路。 | NVIDIA 的 NeMo 工具包提供了一种更集成的说话人分离方法。它提供了一个完整的分离管道，包含 VAD、说话人嵌入提取和聚类。NeMo 的说话人嵌入是通过专门训练的模型提取的，这些嵌入随后在同一框架中用于执行分离所需的聚类。 |
| **聚类和分配** **说话人嵌入** | 在提取了说话人嵌入后，Pyannote 使用各种聚类算法，如层次聚类，来对嵌入进行分组并将其分配给相应的说话人。这个聚类过程对于确定哪些音频段属于哪个说话人至关重要。 | NeMo 也使用聚类算法对说话人嵌入进行分组。然而，NeMo 采用了一种多尺度、自调节的光谱聚类方法，据称比 Pyannote 的版本更具抗干扰性。这种方法包括使用不同窗口长度对音频文件进行分段，并计算多个尺度的嵌入，随后将这些嵌入进行聚类，以标记每个段落的说话人。 |

表 8.1 – 不同的分离方法如何处理关键的分离特征

虽然 Whisper 结合 Pyannote 和 NVIDIA 的 NeMo 都使用说话人嵌入作为其分离流程的核心部分，但它们的处理方法有显著的不同。Whisper 需要一个外部工具包（`pyannote.audio`）来执行说话人分离，而 NeMo 则提供了一个集成的解决方案，包括说话人嵌入提取和聚类模块。NeMo 的多尺度聚类方法是一个独特的特点，使其与与 Whisper 结合使用的 Pyannote 实现有所不同。这些差异反映了语音分离研究领域中多样化的方法和创新。

混合 Whisper 和 PyAnnote – WhisperX

WhisperX ([`replicate.com/dinozoiddev/whisperx`](https://replicate.com/dinozoiddev/whisperx)) 提供了快速的自动语音识别（比 OpenAI 的`Whisper large-v2`快 70 倍），并且具备词级时间戳和说话人分离功能，这些是 Whisper 本身不原生支持的特性。WhisperX 在 Whisper 的基础优势上进行扩展，解决了其一些局限性，特别是在时间戳精度和说话人分离方面。尽管 Whisper 提供的是发言级别的时间戳，WhisperX 通过提供词级时间戳来推动这一进步，这对于需要精确文本和音频同步的应用（如字幕和详细的音频分析）至关重要。这一功能通过结合多种技术实现，包括 VAD（语音活动检测）、将音频预分段为可管理的块，以及使用外部音素模型进行强制对齐，从而提供准确的词级时间戳。

WhisperX 的实现支持 Whisper 支持的所有语言的转录，目前英语音频的对齐功能已经可用。它已升级为融合最新的 Whisper 模型和由 Pyannote 支持的分离技术，以进一步提升其性能。在撰写本文时，WhisperX 集成了`whisper-large-v3`，并且通过 Pyannote 增强了说话人分离（更新至 speaker-diarization-3.1）和分段技术（更新至 segmentation-3.0）。WhisperX 在词汇分割的精度和召回率方面展现出了显著改进，同时在词错误率（WER）上有所减少，并且在采用批量转录和 VAD 预处理时，转录速度有了显著提升。

总结来说，WhisperX 是 OpenAI 的 Whisper 的重要进化，提供了通过词级时间戳和说话人分离增强的功能。这些进展使 WhisperX 成为一个强大的工具，适用于需要详细且准确的语音转录和分析的应用。

有了这个坚实的理论基础，是时候将我们的知识付诸实践了。接下来的实践部分将探索一个实际的实现，结合 WhisperX、NeMo 和其他支持的 Python 库，对现实世界的音频数据进行语音分离。

# 执行实践中的语音分离

从语音分离的理论背景过渡到实际实现，让我们深入了解结合 WhisperX、NeMo 以及其他支持 Python 库的实践应用，所有这些都可以在我们信赖的 Google Colaboratory 中完成。我鼓励你访问本书的 GitHub 仓库，找到`LOAIW_ch08_diarizing_speech_with_WhisperX_and_NVIDIA_NeMo.ipynb`笔记本（[`github.com/PacktPublishing/Learn-OpenAI-Whisper/blob/main/Chapter08/LOAIW_ch08_diarizing_speech_with_WhisperX_and_NVIDIA_NeMo.ipynb`](https://github.com/PacktPublishing/Learn-OpenAI-Whisper/blob/main/Chapter08/LOAIW_ch08_diarizing_speech_with_WhisperX_and_NVIDIA_NeMo.ipynb)），并自己运行 Python 代码；可以随意修改参数并观察结果。该笔记本详细介绍了如何将 Whisper 的转录功能与 NeMo 的语音分离框架集成，提供了一个强大的解决方案来分析音频记录中的语音。

该笔记本被结构化为多个关键部分，每个部分专注于语音分离过程中的特定方面。

## 设置环境

笔记本的第一部分介绍了几个 Python 库和工具的安装，这些工具对于语音分离过程至关重要：

```py
!pip install git+https://github.com/m-bain/whisperX.git@78dcfaab51005aa703ee21375f81ed31bc248560
!pip install --no-build-isolation nemo_toolkit[asr]==1.22.0
!pip install --no-deps git+https://github.com/facebookresearch/demucs#egg=demucs
!pip install dora-search "lameenc>=1.2" openunmix
!pip install deepmultilingualpunctuation
!pip install wget pydub
```

让我们回顾一下每个工具，以理解它们在语音分离中的作用：

+   `whisperX`：OpenAI Whisper 模型的扩展，旨在增强功能。特别地，WhisperX 安装了 faster-whisper（[`github.com/SYSTRAN/faster-whisper`](https://github.com/SYSTRAN/faster-whisper)），这是一个使用 CTranslate2（[`github.com/OpenNMT/CTranslate2/`](https://github.com/OpenNMT/CTranslate2/)）重新实现的 OpenAI Whisper 模型。该实现比 OpenAI 的 Whisper 快最多四倍，且在保持相同精度的同时，内存占用更少。通过在 CPU 和 GPU 上使用 8 位量化，可以进一步提高效率。

+   `nemo_toolkit[asr]`：NVIDIA 的 NeMo 工具包，用于自动语音识别（ASR），为说话人分离提供基础。

+   `demucs`：一个用于音乐源分离的库，能够通过将语音与背景音乐隔离来进行音频文件的预处理。

+   `dora-search`、`lameenc` 和 `openunmix`：用于音频处理的工具和库，提升音频数据的质量和兼容性，以便于语音分离任务。

+   `deepmultilingualpunctuation`：一个用于为转录文本添加标点符号的库，改善了生成文本的可读性和结构。

+   `wget 和 pydub`：用于下载和操作音频文件的工具，简化了在 Python 环境中处理音频数据的过程。

这些库共同构成了处理音频文件、转录语音以及执行说话人分离的基础。每个工具都发挥着特定的作用，从准备音频数据到生成准确的转录内容，再到识别音频中的不同说话人。

## 简化语音分离工作流的辅助函数

该笔记本定义了几个辅助函数，以简化使用 Whisper 和 NeMo 进行话者分离的过程。这些函数在管理音频数据、将转录与说话者身份对齐以及优化工作流程方面起着关键作用。以下是每个函数的简要描述：

+   `create_config()`: 初始化并返回配置对象，设置话者分离过程所需的基本参数：

    ```py
    def create_config(output_dir):
        DOMAIN_TYPE = "telephonic"  # Can be meeting, telephonic, or general based on domain type of the audio file
        CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
        if not os.path.exists(MODEL_CONFIG):
            MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)
        config = OmegaConf.load(MODEL_CONFIG)
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        meta = {
            "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")
        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"
        config.num_workers = 0  # Workaround for multiprocessing hanging with ipython issue
        config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
        config.diarizer.out_dir = (
            output_dir  # Directory to store intermediate files and prediction outputs
        )
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.oracle_vad = (
            False  # compute VAD provided with model_path to vad config
        )
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        # Here, we use our in-house pretrained NeMo VAD model
        config.diarizer.vad.model_path = pretrained_vad
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05
        config.diarizer.msdd_model.model_path = (
            "diar_msdd_telephonic"  # Telephonic speaker diarization model
        )
        return config
    ```

+   `get_word_ts_anchor()`: 确定单词的时间戳锚点，确保说话单词与其音频时间戳之间的准确对齐：

    ```py
    def get_word_ts_anchor(s, e, option="start"):
        if option == "end":
            return e
        elif option == "mid":
            return (s + e) / 2
        return s
    ```

+   `get_words_speaker_mapping()`: 根据话者分离结果，将转录中的每个单词映射到相应的说话者，确保每个单词归属于正确的说话者：

    ```py
    def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
        s, e, sp = spk_ts[0]
        wrd_pos, turn_idx = 0, 0
        wrd_spk_mapping = []
        for wrd_dict in wrd_ts:
            ws, we, wrd = (
                int(wrd_dict["start"] * 1000),
                int(wrd_dict["end"] * 1000),
                wrd_dict["word"],
            )
            wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
            while wrd_pos > float(e):
                turn_idx += 1
                turn_idx = min(turn_idx, len(spk_ts) - 1)
                s, e, sp = spk_ts[turn_idx]
                if turn_idx == len(spk_ts) - 1:
                    e = get_word_ts_anchor(ws, we, option="end")
            wrd_spk_mapping.append(
                {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp}
            )
        return wrd_spk_mapping
    ```

+   `get_first_word_idx_of_sentence()`: 找到句子中第一个单词的索引，对于在说话者归属和对齐的上下文中处理句子至关重要：

    ```py
    def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
        is_word_sentence_end = (
            lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
        )
        left_idx = word_idx
        while (
            left_idx > 0
            and word_idx - left_idx < max_words
            and speaker_list[left_idx - 1] == speaker_list[left_idx]
            and not is_word_sentence_end(left_idx - 1)
        ):
            left_idx -= 1
        return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1
    ```

+   `get_last_word_idx_of_sentence()`: 找到句子中最后一个单词的索引，有助于在转录文本中划定句子边界：

    ```py
    def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
        is_word_sentence_end = (
            lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
        )
        right_idx = word_idx
        while (
            right_idx < len(word_list)
            and right_idx - word_idx < max_words
            and not is_word_sentence_end(right_idx)
        ):
            right_idx += 1
        return (
            right_idx
            if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
            else -1
        )
    ```

+   `get_realigned_ws_mapping_with_punctuation()`: 考虑标点符号调整单词到说话者的映射，提高在复杂对话场景中的说话者归属准确性：

    ```py
    def get_realigned_ws_mapping_with_punctuation(
        word_speaker_mapping, max_words_in_sentence=50
    ):
        is_word_sentence_end = (
            lambda x: x >= 0
            and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
        )
        wsp_len = len(word_speaker_mapping)
        words_list, speaker_list = [], []
        for k, line_dict in enumerate(word_speaker_mapping):
            word, speaker = line_dict["word"], line_dict["speaker"]
            words_list.append(word)
            speaker_list.append(speaker)
        k = 0
        while k < len(word_speaker_mapping):
            line_dict = word_speaker_mapping[k]
            if (
                k < wsp_len - 1
                and speaker_list[k] != speaker_list[k + 1]
                and not is_word_sentence_end(k)
            ):
                left_idx = get_first_word_idx_of_sentence(
                    k, words_list, speaker_list, max_words_in_sentence
                )
                right_idx = (
                    get_last_word_idx_of_sentence(
                        k, words_list, max_words_in_sentence - k + left_idx - 1
                    )
                    if left_idx > -1
                    else -1
                )
                if min(left_idx, right_idx) == -1:
                    k += 1
                    continue
                spk_labels = speaker_list[left_idx : right_idx + 1]
                mod_speaker = max(set(spk_labels), key=spk_labels.count)
                if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                    k += 1
                    continue
                speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                    right_idx - left_idx + 1
                )
                k = right_idx
            k += 1
        k, realigned_list = 0, []
        while k < len(word_speaker_mapping):
            line_dict = word_speaker_mapping[k].copy()
            line_dict["speaker"] = speaker_list[k]
            realigned_list.append(line_dict)
            k += 1
        return realigned_list
    ```

+   `get_sentences_speaker_mapping()`: 生成整个句子与说话者的映射，提供说话者在音频中贡献的高层次视图：

    ```py
    def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
        sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
        s, e, spk = spk_ts[0]
        prev_spk = spk
        snts = []
        snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}
        for wrd_dict in word_speaker_mapping:
            wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
            s, e = wrd_dict["start_time"], wrd_dict["end_time"]
            if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
                snts.append(snt)
                snt = {
                    "speaker": f"Speaker {spk}",
                    "start_time": s,
                    "end_time": e,
                    "text": "",
                }
            else:
                snt["end_time"] = e
            snt["text"] += wrd + " "
            prev_spk = spk
        snts.append(snt)
        return snts
    ```

+   `get_speaker_aware_transcript()`: 生成考虑到说话者身份的转录，将文本内容和说话者信息整合为一致的格式：

    ```py
    def get_speaker_aware_transcript(sentences_speaker_mapping, f):
        previous_speaker = sentences_speaker_mapping[0]["speaker"]
        f.write(f"{previous_speaker}: ")
        for sentence_dict in sentences_speaker_mapping:
            speaker = sentence_dict["speaker"]
            sentence = sentence_dict["text"]
            # If this speaker doesn't match the previous one, start a new paragraph
            if speaker != previous_speaker:
                f.write(f"\n\n{speaker}: ")
                previous_speaker = speaker
            # No matter what, write the current sentence
            f.write(sentence + " ")
    ```

+   `format_timestamp()`: 将时间戳转换为人类可读的格式，便于为转录注释准确的时间信息：

    ```py
    def format_timestamp(
        milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
    ):
        assert milliseconds >= 0, "non-negative timestamp expected"
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
        )
    ```

+   `write_srt()`: 以**SubRip Text**（**SRT**）格式输出话者分离结果，适用于字幕或详细分析，包括说话者标签和时间戳：

    ```py
    def write_srt(transcript, file):
        """
        Write a transcript to a file in SRT format.
        """
        for i, segment in enumerate(transcript, start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
                f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )
    ```

+   `find_numeral_symbol_tokens()`: 在转录文本中识别表示数字符号的标记，帮助处理文本中的数字数据：

    ```py
    def find_numeral_symbol_tokens(tokenizer):
        numeral_symbol_tokens = [
            -1,
        ]
        for token, token_id in tokenizer.get_vocab().items():
            has_numeral_symbol = any(c in "0123456789%$£" for c in token)
            if has_numeral_symbol:
                numeral_symbol_tokens.append(token_id)
        return numeral_symbol_tokens
    ```

+   `_get_next_start_timestamp()`: 计算下一个单词的开始时间戳，确保转录中时间戳序列的连续性：

    ```py
    def _get_next_start_timestamp(word_timestamps, current_word_index, final_timestamp):
        # if current word is the last word
        if current_word_index == len(word_timestamps) - 1:
            return word_timestamps[current_word_index]["start"]
        next_word_index = current_word_index + 1
        while current_word_index < len(word_timestamps) - 1:
            if word_timestamps[next_word_index].get("start") is None:
                # if next word doesn't have a start timestamp
                # merge it with the current word and delete it
                word_timestamps[current_word_index]["word"] += (
                    " " + word_timestamps[next_word_index]["word"]
                )
                word_timestamps[next_word_index]["word"] = None
                next_word_index += 1
                if next_word_index == len(word_timestamps):
                    return final_timestamp
            else:
                return word_timestamps[next_word_index]["start"]
    ```

+   `filter_missing_timestamps()`: 过滤并修正转录数据中缺失或不完整的时间戳，保持时间信息的完整性：

    ```py
    def filter_missing_timestamps(
        word_timestamps, initial_timestamp=0, final_timestamp=None
    ):
        # handle the first and last word
        if word_timestamps[0].get("start") is None:
            word_timestamps[0]["start"] = (
                initial_timestamp if initial_timestamp is not None else 0
            )
            word_timestamps[0]["end"] = _get_next_start_timestamp(
                word_timestamps, 0, final_timestamp
            )
        result = [
            word_timestamps[0],
        ]
        for i, ws in enumerate(word_timestamps[1:], start=1):
            # if ws doesn't have a start and end
            # use the previous end as start and next start as end
            if ws.get("start") is None and ws.get("word") is not None:
                ws["start"] = word_timestamps[i - 1]["end"]
                ws["end"] = _get_next_start_timestamp(word_timestamps, i, final_timestamp)
            if ws["word"] is not None:
                result.append(ws)
        return result
    ```

+   `cleanup()`: 清理在话者分离过程中创建的临时文件或目录，确保工作环境整洁：

    ```py
    def cleanup(path: str):
        """path could either be relative or absolute."""
        # check if file or directory exists
        if os.path.isfile(path) or os.path.islink(path):
            # remove file
            os.remove(path)
        elif os.path.isdir(path):
            # remove directory and all its content
            shutil.rmtree(path)
        else:
            raise ValueError("Path {} is not a file or dir.".format(path))
    ```

+   `process_language_arg()`: 处理语言参数，确保与模型兼容，促进不同语言间的准确转录：

    ```py
    def process_language_arg(language: str, model_name: str):
        """
        Process the language argument to make sure it's valid and convert language names to language codes.
        """
        if language is not None:
            language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")
        if model_name.endswith(".en") and language != "en":
            if language is not None:
                logging.warning(
                    f"{model_name} is an English-only model but received '{language}'; using English instead."
                )
            language = "en"
        return language
    ```

+   `transcribe()`: 利用 Whisper 将音频转录为文本，为话者分离过程提供基础文本数据：

    ```py
    def transcribe(
        audio_file: str,
        language: str,
        model_name: str,
        compute_dtype: str,
        suppress_numerals: bool,
        device: str,
    ):
        from faster_whisper import WhisperModel
        from helpers import find_numeral_symbol_tokens, wav2vec2_langs
        # Faster Whisper non-batched
        # Run on GPU with FP16
        whisper_model = WhisperModel(model_name, device=device, compute_type=compute_dtype)
        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")
        if suppress_numerals:
            numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        else:
            numeral_symbol_tokens = None
        if language is not None and language in wav2vec2_langs:
            word_timestamps = False
        else:
            word_timestamps = True
        segments, info = whisper_model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            word_timestamps=word_timestamps,  # TODO: disable this if the language is supported by wav2vec2
            suppress_tokens=numeral_symbol_tokens,
            vad_filter=True,
        )
        whisper_results = []
        for segment in segments:
            whisper_results.append(segment._asdict())
        # clear gpu vram
        del whisper_model
        torch.cuda.empty_cache()
        return whisper_results, language
    ```

+   `transcribe_batched()`：提供批处理能力以转录音频文件，从而优化转录过程的效率和可扩展性：

    ```py
    def transcribe_batched(
        audio_file: str,
        language: str,
        batch_size: int,
        model_name: str,
        compute_dtype: str,
        suppress_numerals: bool,
        device: str,
    ):
        import whisperx
        # Faster Whisper batched
        whisper_model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_dtype,
            asr_options={"suppress_numerals": suppress_numerals},
        )
        audio = whisperx.load_audio(audio_file)
        result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)
        del whisper_model
        torch.cuda.empty_cache()
        return result["segments"], result["language"]
    ```

这些功能共同构成了笔记本的说话人分离工作流程基础，能够无缝集成 Whisper 的转录能力与 NeMo 的高级说话人分离功能。

## 使用 Demucs 将音乐与语音分离

在探索笔记本时，让我们关注预处理步骤，这对于在进行说话人分离之前提升语音清晰度至关重要。本节介绍了**Demucs**，一个用于将音乐源人声与复杂音轨分离的深度学习模型。

将音乐与语音分离非常关键，尤其是在处理包含背景音乐或其他非语音元素的录音时。通过提取人声成分，分离系统可以更有效地分析并将语音归属给正确的说话人，因为其语音信号的频谱和时间特征变得更加明显，不会被音乐干扰：

```py
if enable_stemming:
    # Isolate vocals from the rest of the audio
    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
    )
    if return_code != 0:
        logging.warning("Source splitting failed, using original audio file.")
        vocal_target = audio_path
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(audio_path))[0],
            "vocals.wav",
        )
else:
    vocal_target = audio_path
```

Demucs 通过利用一个神经网络来区分混合音频中的不同音源来工作。当应用于音频文件时，它可以将人声轨道与伴奏乐器分开，从而使得后续工具（如 Whisper 和 NeMo）能够在没有背景音乐干扰的情况下处理语音。

这个分离步骤对说话人分离的准确性以及任何需要清晰语音输入的后续任务（如转录和语音识别）非常有帮助。通过将 Demucs 作为预处理管道的一部分，笔记本确保输入到分离系统的音频得到了优化，从而实现最佳性能。

## 使用 WhisperX 进行音频转录

下一步是利用 WhisperX 来转录音频内容。转录过程包括将音频文件通过 Whisper 处理，生成一组文本段落，每个段落都有时间戳，指示该段落被说出的时间：

```py
compute_type = "float16"
# or run on GPU with INT8
# compute_type = "int8_float16"
# or run on CPU with INT8
# compute_type = "int8"
if batch_size != 0:
    whisper_results, language = transcribe_batched(
        vocal_target,
        language,
        batch_size,
        whisper_model_name,
        compute_type,
        suppress_numerals,
        device,
    )
else:
    whisper_results, language = transcribe(
        vocal_target,
        language,
        whisper_model_name,
        compute_type,
        suppress_numerals,
        device,
    )
```

这一基础步骤提供了进行说话人分离和进一步分析所需的文本内容。我希望你注意到，`transcribe()`和`transcribe_batch()`这两个函数在笔记本中已经定义过了。

## 使用 Wav2Vec2 对转录文本与原始音频进行对齐

在转录之后，笔记本介绍了如何使用**Wav2Vec2**进行强制对齐，这是一个将转录文本与原始音频进行对齐的过程。Wav2Vec2 是一个大型神经网络模型，擅长学习有助于语音识别和对齐任务的语音表示。通过使用 Wav2Vec2，我们演示了如何微调转录段落与音频信号的对齐，确保文本与所说的话语准确同步：

```py
if language in wav2vec2_langs:
    device = "cuda"
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    result_aligned = whisperx.align(
        whisper_results, alignment_model, metadata, vocal_target, device
    )
    word_timestamps = filter_missing_timestamps(
        result_aligned["word_segments"],
        initial_timestamp=whisper_results[0].get("start"),
        final_timestamp=whisper_results[-1].get("end"),
    )
    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()
else:
    assert batch_size == 0, (  # TODO: add a better check for word timestamps existence
        f"Unsupported language: {language}, use --batch_size to 0"
        " to generate word timestamps using whisper directly and fix this error."
    )
    word_timestamps = []
    for segment in whisper_results:
        for word in segment["words"]:
            word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})
```

这一对齐对于分离至关重要，因为它允许根据说话人的变化进行更精确的音频分割。Whisper 和 Wav2Vec2 的结合输出提供了一个完全对齐的转录，这对于说话人分离、情感分析和语言识别等任务非常有帮助。本节强调，如果某个特定语言没有可用的 Wav2Vec2 模型，则将使用 Whisper 生成的单词时间戳，展示了该方法的灵活性。

通过将 Whisper 的转录能力与 Wav2Vec2 的对齐精度结合，我们为准确的说话人分离奠定了基础，提高了分离过程的整体质量和可靠性。

## 使用 NeMo 的 MSDD 模型进行说话人分离

在笔记本的核心部分，重点转向了复杂的说话人分离过程，利用了 NVIDIA NeMo MSDD 的先进能力。该部分非常关键，因为它解决了在音频信号中区分不同说话人的问题，这对于将语音片段准确归属到个人说话人至关重要：

```py
# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cuda")
msdd_model.diarize()
del msdd_model
torch.cuda.empty_cache()
```

NeMo MSDD 模型处于这一过程的最前沿，采用了一种复杂的分离方法，考虑了说话人嵌入的多时间分辨率。这种多尺度策略提高了模型在挑战性音频环境中（如重叠语音或背景噪音）区分说话人的能力。

## 根据时间戳将说话人映射到句子

在成功地将语音与音乐分离、使用 Whisper 转录音频并通过 NeMo MSDD 模型进行说话人分离后，下一个挑战是将转录中的每个句子准确地映射到相应的说话人。这涉及分析转录中每个单词或片段的时间戳以及在分离过程中分配的说话人标签：

```py
speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

前面的代码确保了转录中的每个句子都正确地归属于某个说话人，考虑了口语片段的开始和结束时间。这一细致的映射对于理解对话动态（例如谁在何时说了什么）至关重要。它使得对多说话人对话、会议、访谈和音频内容的分析更加细致。

## 通过基于标点符号的重新对齐提升说话人归属

以下代码片段演示了标点符号如何决定每个句子的主要说话人。它使用一个预训练的标点符号模型`kredor/punctuate-all`，为转录的单词预测标点符号。然后，代码处理这些单词及其预测的标点符号，处理一些特殊情况，例如首字母缩略词（如 USA），以避免错误的标点符号。这种方法确保即使在其他说话人的背景评论或简短插话的情况下，每个句子的说话人归属也能保持一致。这在转录未指示说话人变化的情况下特别有用，例如当一个说话人的发言被另一个人的话语打断或重叠时。通过分析每个句子中每个单词的说话人标签分布，代码能够为整个句子分配一个一致的说话人标签，从而增强对话分段输出的连贯性：

```py
if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    words_list = list(map(lambda x: x["word"], wsm))
    labled_words = punct_model.predict(words_list)
    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word
else:
    logging.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )
wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
```

这种方法还解决了在主要说话人进行独白时，背景评论或简短插话发生的情况。代码有效地将主要的讲话内容归属于主说话人，而忽略其他人的零星评论。这使得语音片段与相应说话人的映射更加准确可靠，确保分段过程反映了对话的实际结构。

## 完成分段过程

在最后一部分，代码执行必要的清理任务，导出分段结果以供进一步使用，并将说话人 ID 替换为对应的名字。主要步骤包括以下内容：

1.  `get_speaker_aware_transcript` 函数生成一个包含文本内容和说话人信息的转录本。该转录本随后被保存为与输入音频文件同名的文件，但扩展名为`.txt`：

    ```py
    with open(f"{os.path.splitext(audio_path)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)
    ```

1.  `write_srt function` 用于将分段结果导出为 SRT 格式。此格式通常用于字幕，并包括每个发言的说话人标签和精确的时间戳。SRT 文件将以与输入音频文件相同的名称保存，但扩展名为`.srt`：

    ```py
    with open(f"{os.path.splitext(audio_path)[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)
    ```

1.  **清理临时文件**：清理功能删除在分段过程中创建的任何临时文件或目录。这一步确保了一个干净、井然有序的工作环境，释放了存储空间并保持系统效率：

    ```py
    cleanup(temp_path)
    ```

1.  将`Speaker 0`、`Speaker 1`和`Speaker 2`替换为实际说话人的名字：

    ```py
    # Open the file
    with open(f"{os.path.splitext(audio_path)[0]}.txt", 'r') as f:
        text = f.read()
    # Replace the speaker IDs with names
    text = text.replace('Speaker 0','Ewa Jasiewicz')
    text = text.replace('Speaker 1','Chris Faulkner')
    text = text.replace('Speaker 2','Matt Frei')
    # Write the file to disk
    with open(audio_path[:-4] + '-with-speakers-names.txt', 'w') as f:
        f.write(text)
    ```

通过完成这些最终步骤，语音分离过程得以完成，结果可以用于进一步分析、后处理或与其他工具和工作流程的集成。导出的包含说话者信息的转录、SRT 文件和映射了说话者名称的转录提供了对音频录音内容和结构的宝贵洞见，为广泛应用提供了可能，如内容分析、说话者识别和字幕生成。

在深入研究该笔记本后，我们发现了一个关于使用前沿 AI 工具进行语音分离的宝贵宝藏。这本笔记本是一本实践指南，详细指导我们如何从复杂的音频文件中分离和转录语音。

第一个教训是如何设置正确的环境。该笔记本强调了安装特定依赖项的必要性，如 Whisper 和 NeMo，它们对于任务的执行至关重要。这一步是基础，为所有后续操作奠定了基础。

随着深入学习，我们了解了辅助函数的实用性。这些函数是默默奉献的英雄，它们简化了工作流程，从处理音频文件到处理时间戳和清理资源。它们体现了编写简洁、可重用代码的原则，大大降低了项目的复杂性。

该笔记本还介绍了使用 Demucs 将音乐从语音中分离的技术。这个步骤展示了预处理在提高分离准确性方面的强大作用。通过隔离人声，我们专注于语音的频谱和时间特征，这对于识别不同的说话者至关重要。

另一个关键收获是集成多个模型以获得更好的结果。该笔记本展示了如何使用 Whisper 进行转录，使用 Wav2Vec2 将转录与原始音频对齐。模型之间的协同作用是一个出色的例子，展示了如何结合不同的 AI 工具来实现更强大的解决方案。

将说话者映射到句子并通过标点符号重新对齐语音片段的过程尤其令人启发。它展示了语音分离的复杂性以及对细节的关注，确保每个说话者在转录中都得到准确表达。

从本质上讲，这本笔记本是一个关于 AI 在语音分离应用中的实用技巧的高级教程。它不仅教会了我们涉及的技术步骤，还传授了关于预处理重要性、结合不同 AI 模型的力量，以及细致后处理以确保最终输出完整性的更广泛教训。

# 总结

在本章中，我们开始了对 OpenAI Whisper 先进语音能力的激动人心的探索。我们深入研究了增强 Whisper 性能的强大技术，例如量化，并发现了它在说话者分离和实时语音识别中的潜力。

我们为 Whisper 增添了说话人分离功能，使其能够识别并将音频录音中的语音片段归属到不同的说话人。通过将 Whisper 与 NVIDIA NeMo 框架整合，我们学会了如何执行精准的说话人分离，为分析多说话人对话开辟了新天地。我们与 WhisperX 和 NVIDIA NeMo 的实操经验展示了将 Whisper 的转录能力与先进的说话人分离技术结合的强大潜力。

在本章中，我们深入理解了优化 Whisper 性能和通过说话人分离扩展其功能的高级技巧。通过动手编码示例和实用的见解，我们掌握了应用这些技巧的知识和技能，推动了 Whisper 的可能性边界。

当我们结束本章时，我们将展望*第九章*，*利用 Whisper 进行个性化语音合成*。在那一章中，我们将获得预处理音频数据、微调语音模型以及使用个人语音合成模型生成逼真语音的知识和技能。动手编码示例和实用的见解将使你能够将这些技巧应用到你的项目中，推动个性化语音合成技术的边界。

跟随我继续与 Whisper 同行，准备迎接语音合成技术这一迅速发展的领域中的激动人心的可能性。
