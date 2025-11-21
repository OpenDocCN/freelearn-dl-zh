

# 第六章：理解大型语言模型的领域自适应

在上一章中，我们探讨了**参数高效微调**（**PEFT**）如何增强**大型语言模型**（**LLMs**）以适应特定任务，例如问答。在本章中，我们将介绍领域自适应，这是一种独特的微调方法。与特定任务微调不同，领域自适应使模型能够解释特定行业或领域的独特语言，解决LLMs在理解专业语言方面的差距。

为了说明这一点，我们将介绍*Proxima投资集团*，这是一个假设的仅数字投资公司，旨在利用内部数据将LLM调整到其特定的金融语言。我们将展示如何修改LLM以处理Proxima环境中典型的特定术语和细微差别，从而增强模型在金融领域的相关性和有效性。

我们还将探讨Proxima可能采取的实用步骤，例如选择相关的内部数据集进行训练，应用**低秩自适应**（**LoRA**）等PEFT方法来高效地调整模型，以及使用掩码技术来细化模型的理解。然后，我们将探讨Proxima如何评估这种领域自适应的成功，评估模型在分析金融趋势、回应客户咨询和生成符合Proxima内部标准和市场定位的报告等任务中的性能。

到本章结束时，我们将清楚地理解领域自适应的理论基础及其在现实世界中的应用，特别是在金融等复杂领域，模型对领域理解的深度可以显著影响业务成果。

让我们从揭秘这个概念开始，探讨其技术基础，并讨论其在实现特定业务目标中的重要性。

# 揭秘领域自适应——了解其历史和重要性

在生成型LLMs的背景下，领域自适应特别针对像**BLOOM**这样的模型进行定制，这些模型在广泛的通用数据集（如新闻文章和维基百科条目）上进行了预训练，以增强对目标行业文本的理解，包括生物医学、法律和金融领域。这种改进可能至关重要，因为尽管LLMs在预训练中具有广泛的数据，但它们可能无法天生捕捉到这些领域固有的复杂细节和专门术语。这种自适应涉及一个有意识的过程，将模型学习到的模式重新对齐到目标领域普遍存在的语言特征、术语和语境细微差别。

**领域自适应**在**迁移学习**的范畴内运作。在这个更广泛的概念中，模型从一个任务中学到的知识被重新用于提高其在相关但不同的任务上的有效性。这种方法利用模型预先学习到的特征，以提高其在后续任务上的效率和准确性，显著减少其对大量特定领域数据和计算资源的依赖。具体来说，我们从一个在广泛数据集上训练好的模型开始，将其用作适应特定领域的起点，从而提高其准确性、相关性和对更针对性用例的适用性。

在实践中，可以采用多种方法来调整模型以适应特定领域，包括以下方法：

+   **持续预训练**：模型在特定领域的语料库上进行额外的预训练，使其参数能够逐步适应目标领域的语言特征，正如 Gururangan 等人在 2020 年的研究中所强调的。

+   **中间任务训练**：在这里，模型在中间任务上进行训练，在微调下游应用之前使用特定领域的数据。这一步骤有助于更稳健地适应领域（Pruksachatkun 等人，2020）。

+   **数据增强**：利用**回译**（Xie 等人，2019）和**标记替换**（Anaby-Tavor 等人，2020）等技术，从有限的实际数据中生成合成的特定领域训练示例：

    +   **回译**涉及将现有文本从一种语言（例如，英语）翻译成另一种语言（例如，法语），然后再将其翻译回原始语言。这个过程生成了原始文本的释义版本，同时保留了其语义。

    +   **标记替换**涉及改变句子中的单个单词以生成新的句子。这种改变通常旨在保留原始句子的语义意义，同时引入变化。

+   **多任务学习**：该框架在适应阶段同时优化模型以处理通用和特定领域的任务，正如 Clark 等人在 2019 年所展示的。

随着领域自适应技术的不断发展，它们在特定领域中的模型性能不断提高，即使是在减少特定领域数据的情况下。如第 [*第4章*](B21773_04.xhtml#_idTextAnchor123) 所述，最近的发展更加关注这些技术的计算效率。如 LoRA 这样的适应方法通过最小的参数变化实现重大的模型调整，而无需全面重新训练。需要注意的是，模型的表现将始终根据数据集的质量、可用的计算资源和其他实现细节等多种因素而变化。

现在我们对领域自适应技术和它们对计算效率的关注有所了解，我们可以将这些概念应用于实践。我们的实践项目将利用BLOOM，一个最先进的开源LLM，来展示金融领域的领域自适应。利用PEFT，我们旨在以最少的计算资源微调BLOOM，展示这些高级自适应方法在增强金融领域模型性能中的实际应用。

# 实践项目：金融领域的迁移学习

本项目旨在对特定文档的精选语料库上的BLOOM进行微调，使其具备解释和阐述Proxima及其产品特定概念的能力。

我们的方法灵感来源于跨多个领域的领域自适应策略，包括生物医学、金融和法律。一项由Cheng等人于2023年进行的值得注意的研究，名为《通过阅读理解调整大型语言模型》*，提出了一种增强LLM在特定任务中能力的新方法。这种方法将大量的预训练语料库重新格式化为有利于阅读理解任务的格式，显著提高了模型在特定领域的功能。在我们的案例中，我们将采用类似但简化的方法，通过使用针对Proxima特定数据集的微调来继续预训练，有效地继续模型的训练。这个过程逐步调整模型参数，以确保模型更好地理解Proxima的产品和提供的独特语言。

## 金融领域自适应的培训方法

为了我们持续的培训策略，我们将采用**因果语言模型**（**CLM**）。这种方法是更广泛的一组培训方法之一，旨在优化模型性能以实现各种目标。在转向实施之前，让我们尝试区分我们选择的方法与其他流行策略，以便更好地理解CLM方法：

+   **掩码语言模型**（**MLM**）：这是基于Transformer的模型（如BERT）的基石，MLM随机掩码输入文本的部分，并挑战模型预测掩码的标记。通过考虑掩码周围的整体上下文（包括掩码之前和之后的内容），MLM使模型能够发展双向的语言理解能力，丰富其对上下文和语义的掌握。

+   **下一句预测**（**NSP**）：这种方法通过训练模型判断两个句子是否逻辑上相互跟随，进一步扩展了模型的叙事理解能力。NSP对于教授模型关于文本结构和连贯性至关重要，使其能够在更大的文本体中构建和理解逻辑序列。

+   **CLM**：我们为BLOOM的适应性选择了一条不同的路径，采用CLM因为它具有专注的、顺序预测的能力。与MLM（它同时查看（在掩码标记之前和之后））不同，CLM采用单向方法，仅根据前面的上下文预测每个后续标记。这种方法与自然语言生成内在一致，使其特别适合在目标领域中构建连贯、上下文丰富的叙述。

在选择CLM对BLOOM进行适应性时，我们将扩展模型的生成能力，以产生不仅逻辑结构良好，而且深深植根于目标领域细微差别的文本序列。CLM的单向性质确保每个生成的标记都由对先前文本的连贯理解所指导，使模型能够生成详细、准确且特定于领域的文本。

一旦微调完成，我们可以根据该领域自适应BLOOM模型在生成与上下文相关且特定于领域的叙述方面的熟练程度来评估其有效性。我们将比较自适应模型与原始模型的表现，特别关注模型的流畅性、准确性和对目标领域的整体理解。

正如我们之前所做的那样，我们将利用Google Colab进行我们的初始原型设计阶段。正如第4章和第5章所描述的，Google Colab提供了一个预配置的环境，简化了在我们考虑将我们的方法推广到生产环境之前测试我们的方法的过程。本章中所有的代码都可在本书GitHub仓库的`Chapter 6`文件夹中找到（[https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python](https://github.com/PacktPublishing/Generative-AI-Foundations-in-Python)）。

我们将从初始设置开始，这涉及到使用Transformers库加载**BLOOM-1b1**的一个较小变体。我们还将导入我们将需要应用PEFT的方法。对于这个例子，我们将依赖一些可以按以下方式安装的库：

```py
pip install sentence-transformers transformers peft datasets
```

安装完成后，我们可以开始导入：

```py
from transformers import (
    AutoTokenizer, AutoModelForCausalLM)
from peft import AdaLoraConfig, get_peft_model
```

下一步是加载分词器和模型：

```py
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b1")
```

如前所述，我们正在引入PEFT以提高适应性：

```py
adapter_config = AdaLoraConfig(target_r=16)
model.add_adapter(adapter_config)
```

PEFT技术，特别是通过`AdaLoraConfig`，允许我们引入一个紧凑、高效的层，这样我们就可以通过显著减少可训练参数的数量来将模型适应新的上下文——在这里，是金融领域：

```py
model = get_peft_model(model, adapter_config)
model.print_trainable_parameters()
```

我们必须集成适配器以完成PEFT模型设置，从而有效地创建一个针对我们特定领域训练优化的模型变体，同时关注效率。我们可以通过检查我们的模型将使用的可训练参数数量来量化这一点：

```py
trainable params: 1,769,760 || all params: 1,067,084,088 || trainable%: 0.1658500974667331
```

上述代码为我们提供了以下信息：

+   **可训练参数**：1,769,760

+   **模型中的总参数**：1,067,084,088

+   **可训练参数百分比**：0.166%

这意味着在BLOOM-1b1模型超过10亿个参数中，只有大约177万个参数被用于金融领域自适应的微调。这个很小的百分比（0.166%）的可训练参数突出了PEFT的效率，允许通过最小的调整实现显著的模型适应性。这对于实际应用至关重要，因为它减少了计算成本和训练时间。

接下来，我们将进入数据准备阶段。我们假设我们已经收集了涵盖Proxima产品及其服务（如**Proxima Passkey**）的广泛知识文本。CLM训练需要区分测试和训练阶段，以评估模型准确预测序列中下一个标记的能力。这确保了模型在训练数据之外也能很好地泛化到未见过的文本。在训练过程中，损失计算衡量模型预测的标记概率与实际标记之间的差异。它指导模型调整其参数以最小化这种损失，通过迭代提高其预测准确性。因此，我们必须定义训练和测试文本作为我们的数据集。本书的GitHub仓库（本章前面已链接）中包含了一个示例数据集。

```py
dataset = load_dataset("text",
    data_files={"train": "./train.txt",
        "test": "./test.txt"}
    )
```

接下来，我们必须应用预处理和分词。文本被清理、标准化，然后转换为数值格式（`512`个标记，以便与模型的架构相匹配）：

```py
def preprocess_function(examples):
    inputs = tokenizer(examples["text"], truncation=True,
        padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs
```

`TrainingArguments`类配置了训练过程，设置如批量大小、训练轮数和保存模型检查点的目录等参数。这种配置对于高效学习和模型评估至关重要。同时，`Trainer`类协调模型的训练过程。再次强调，持续训练逐渐调整模型的参数，以生成和理解与Proxima Passkey相关的文本：

```py
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    prediction_loss_only=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
model.save_pretrained("./proxima_da_model")
```

通常，我们的配置指定了训练参数并初始化`Trainer`类，同时专注于领域自适应。`TrainingArguments`类被定制以高效管理训练过程，包括日志记录和模型保存策略。记住，我们为训练模型选择的批量大小平衡了GPU的内存容量和模型从数据集中学习的速度。较大的批量大小允许一次处理更多数据，从而加快训练速度，但需要更多内存，如果GPU容量有限，这可能会成为限制。相反，较小的批量大小意味着模型使用较少的样本更频繁地更新其权重，这可以促进学习，但会导致通过数据集的整体进度变慢。

训练完成后，我们可以使用自适应模型根据与Proxima Passkey相关的提示生成文本。模型考虑提示，生成表示续写的标记序列，然后将此序列解码回可读的文本：

```py
def predict(model, prompt="The Proxima Passkey is"):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

注意`model.generate()`函数，它接受分词输入并生成一系列标记作为输出。然后，这些标记被解码成文本。

在这个例子中，我们调整了BLOOM语言模型，使其专门针对金融领域。这包括加载预训练模型，应用PEFT适配器以实现高效的领域适应性，并通过标准化和分词准备金融数据集以供模型训练。在用特定领域的数据进行微调BLOOM后，我们使用该模型生成与金融行业相关的文本。最后一步是评估这个调整后的模型与原始预训练版本的性能，重点关注其在准确处理财务语言和概念方面的有效性。

## 评估和结果分析——ROUGE度量标准

定量评估和定性评估对于评估适应后的BLOOM模型与原始模型之间的差异至关重要，尤其是在Proxima语言的环境中。在定量方面，模型的输出通过与使用**ROUGE**度量标准反映Proxima产品语言的参考数据集进行比较。这种比较有助于衡量关键术语和风格的重叠程度。此外，为评估模型在Proxima相关的财务术语和概念方面的熟练程度，开发特定的度量标准是有益的：

```py
from rouge import Rouge
# Example reference text (what we expect the model to generate after training on a complete dataset)
reference = "Proxima's Passkey enables seamless integration of diverse financial portfolios, offering unparalleled access to global investment opportunities and streamlined asset management."
# Example predicted model output
predicted = "The Proxima Passkey provides a unified platform for managing various investment portfolios, granting access to worldwide investment options and efficient asset control."
# Initialize the Rouge metric
rouge = Rouge()
# Compute the Rouge scores
scores = rouge.get_scores(predicted, reference)
print(scores)
```

ROUGE分数将通过比较本例中的两个文本来计算。该分数衡量预测输出与参考文本在**n-gram**（单词序列）方面的重叠。例如，**ROUGE-N**（其中`N`可以是1、2或L）计算预测文本和参考文本之间的n-gram重叠：

+   **ROUGE-1**评估预测文本和参考文本之间单语（单个单词）的重叠

+   **ROUGE-2**评估文本之间双语（两个单词的短语）的重叠

+   **ROUGE-L**关注最长公共子序列，这对于评估句子级结构相似性很有用

ROUGE分数的范围从0到1，量化预测文本与参考文本之间的相似性，为模型输出与预期内容匹配的程度提供见解。接近1的分数表示更高的相似性或重叠，而接近0的分数则表示几乎没有共同性。这些分数分为三个关键组成部分——精确度、召回率和F1分数：

+   **精确度**衡量预测文本中在参考文本中也存在的单词比例。高精确度分数表明模型生成的单词大多数是相关的，并且出现在参考文本中，这表明模型输出的准确性。

+   **召回率**评估参考文本中在模型预测中捕获的单词比例。高召回率意味着模型有效地在其输出中包含了参考文本中的大多数相关内容，表明内容的全面性。

+   **F1分数**是精确率和召回率的调和平均数，平衡了两者。它在理解模型生成既相关（精确率）又全面（召回率）的文本的整体准确性方面特别有用。当在评估模型性能时，精确率和召回率同等重要时，F1分数至关重要。

+   这里是输出：

| **度量** | **召回率（r）** | **精确率（p）** | **F1分数（f）** |
| --- | --- | --- | --- |
| ROUGE-1 | 0.35 | 0.333 | 0.341 |
| ROUGE-2 | 0.053 | 0.048 | 0.05 |
| ROUGE-L | 0.35 | 0.333 | 0.341 |

表6.1：ROUGE度量结果

这些分数表明，文本之间存在中等程度的单语素重叠（ROUGE-1），但二元重叠（ROUGE-2）显著较低。ROUGE-1和ROUGE-L分数之间的相似性表明，模型在一定程度上捕捉了单个关键术语，但可能在较长的短语结构上遇到困难，这指出了模型改进的领域。

总体而言，虽然该模型在关键个体术语方面表现出基本理解（如ROUGE-1和ROUGE-L所示），但它复制参考文本中更复杂结构或短语的能力（如ROUGE-2所示）相当有限。这表明，尽管模型对特定领域的语言有一定理解，但仍需进一步微调才能有效地复制参考文本中更细微和结构化的方面。记住，正如我们在其他章节中看到的，语义相似性也是衡量特定领域语言理解的好指标，并且不像ROUGE那样依赖于词汇重叠。

定性上，领域专家应审查模型的输出，以判断其在Proxima的产品和机构语言背景下的相关性和准确性。这些专家可以提供关于模型性能细微之处的见解，这些细微之处可能无法仅通过量化指标来捕捉。比较他们对原始模型和自适应模型输出的反馈将突出自适应如何使BLOOM与Proxima的具体沟通需求相一致。这种双重方法确保了全面的评估，将统计分析与实际应用和相关性相结合。

# 摘要

在本章中，我们探讨了BLOOM LLM的领域自适应过程，该过程专门针对提高其在金融领域的熟练度，特别是在理解和生成与Proxima的产品提供相关的内容。我们首先介绍了领域自适应的概念，这是在迁移学习更广泛范围内的一个概念，强调了其在微调通用模型以掌握专业领域复杂性的重要性。

该适应过程涉及将PEFT技术集成到BLOOM中，并对金融数据集进行预处理以进行模型训练。这包括通过截断和填充标准化文本长度，并对文本进行标记化以确保模型输入的一致性。然后，使用ROUGE指标对适应后的模型性能进行定量评估，以参考数据集为基准，从而提供了对其捕捉关键金融术语和短语能力的见解。同时，也建议由领域专家进行定性评估，作为衡量模型在实际场景中实际有效性的补充方法。

总体而言，本章详细介绍了针对特定领域微调LLM的常见方法，阐述了方法论以及细微评估的重要性，以确保此类适应的成功。在下一章中，我们将探讨如何使用提示工程来适应LLM而不进行微调。我们将发现如何使模型输出具有上下文并引导其产生与微调模型相似的结果。

# 参考文献

本参考文献部分作为本书中引用的资源的存储库；您可以探索这些资源以进一步加深对主题的理解和知识。

+   Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). *不要停止预训练：将语言模型适应到领域和任务中*。载于arXiv [cs.CL]。[http://arxiv.org/abs/2004.10964/](http://arxiv.org/abs/2004.10964/)。

+   Pruksachatkun, Y., Phang, J., Liu, H., Htut, P. M., Zhang, X., Pang, R. Y., Vania, C., Kann, K., & Bowman, S. R. (2020a). *中间任务迁移学习与预训练语言模型：何时以及为什么它有效？* 第58届计算语言学协会年度会议论文集。

+   Xie, Q., Dai, Z., Hovy, E., Luong, M.-T., & Le, Q. V. (n.d.). *无监督数据增强用于一致性训练*。Arxiv.org。2024年3月16日检索自 [http://arxiv.org/abs/1904.12848](http://arxiv.org/abs/1904.12848)。

+   Anaby-Tavor, A., Carmeli, B., Goldbraich, E., Kantor, A., Kour, G., Shlomov, S., Tepper, N., & Zwerdling, N. (2020). *数据不足？深度学习来拯救!* 第... AAAI人工智能会议论文集。AAAI Conference on Artificial Intelligence, 34(05), 7383–7390。 [https://doi.org/10.1609/aaai.v34i05.6233](https://doi.org/10.1609/aaai.v34i05.6233)。

+   Clark, K., Luong, M.-T., Khandelwal, U., Manning, C. D., & Le, Q. V. (2019). *BAM！自然语言理解中再生的多任务网络*。载于arXiv [cs.CL]。[http://arxiv.org/abs/1907.04829](http://arxiv.org/abs/1907.04829)。
