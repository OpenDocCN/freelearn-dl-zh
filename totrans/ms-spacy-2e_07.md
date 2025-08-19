# 7

# 使用 spacy-llm 增强 NLP 任务

在本章中，我们将基于在*第六章*中获得的知识，探讨如何使用`spacy-llm`库将**大型语言模型**（`LLMs`）集成到 spaCy 管道中。我们将从理解 LLMs 和提示工程的基础知识开始，以及这些强大的模型如何在 spaCy 中执行各种 NLP 任务。我们将演示如何配置和使用预构建的 LLM 任务，如文本摘要，然后进一步创建一个自定义任务，从文本中提取上下文信息。这涉及到使用 Jinja 模板创建提示，并编写自定义 spaCy 组件，以高效地处理复杂的 NLP 任务。到本章结束时，你将更深入地了解如何利用 LLMs 的灵活性和强大功能来增强传统的 NLP 管道。

在本章中，我们将涵盖以下主要主题：

+   LLMs 和提示工程基础知识

+   使用 LLMs 和 spaCy 进行文本摘要

+   使用 Jinja 模板创建自定义 LLM 任务

# 技术要求

在本章中，我们将使用 spaCy 和`spacy-llm`库来创建和运行我们的管道。你可以在此章节中找到使用的代码：[`github.com/PacktPublishing/Mastering-spaCy-Second-Edition`](https://github.com/PacktPublishing/Mastering-spaCy-Second-Edition)。

# LLMs 和提示工程基础知识

正如我们在*第六章*中看到的，**语言建模**是预测给定先前标记序列的下一个标记的任务。我们使用的例子是，给定单词序列**昨天我访问了一个**，语言模型可以预测下一个标记可能是**教堂**、**医院**、**学校**等。传统的语言模型通常以监督方式训练以执行特定任务。**预训练语言模型**（`PLM`）以自监督方式训练，目的是学习语言的通用表示。然后对这些 PLM 模型进行微调以执行特定的下游任务。这种自监督的预训练使 PLM 模型比常规语言模型更强大。

LLMs（大型语言模型）是 PLMs（预训练语言模型）的进化，拥有更多的模型参数和更大的训练数据集。例如，GPT-3 模型拥有 1750 亿个参数。其继任者 GPT3.5 是 2022 年 11 月发布的 ChatGPT 模型的基础。LLMs 可以作为通用工具，能够执行从语言翻译到编码辅助的各种任务。它们理解和生成类似人类文本的能力，在医学、教育、科学、数学、法律等领域产生了重大影响。在医学领域，LLMs 为医生提供基于证据的建议，并增强患者互动。在教育领域，它们定制学习体验，并协助教师创建内容。在科学领域，LLMs 加速研究和科学写作。在法律领域，它们分析法律文件并阐明复杂术语。

我们还可以使用 LLMs 进行常规 NLP 任务，如**命名实体识别**（`NER`）、文本分类和文本摘要。基本上，这些模型可以完成我们要求的几乎所有事情。但这是有代价的，因为训练它们需要大量的计算资源，大量的层和参数使它们产生答案的速度比非 LLM 模型慢得多。LLMs 也可能**产生幻觉**：产生看似合理但实际上不正确或与事实或上下文不一致的响应。这种现象发生是因为模型根据从训练数据中学习的模式生成文本，而不是通过外部来源验证信息。因此，它们可能会创建听起来合理但实际上具有误导性、不准确或完全虚构的陈述。鉴于所有这些，LLMs 是有用的，但我们始终应该分析它们是否是手头项目的最佳解决方案。

要与 LLMs 交互，我们使用提示。**提示**应引导模型生成答案或使模型采取行动。提示通常包含以下元素：

+   **指令**：您希望模型执行的任务

+   **上下文**：对产生更好答案有用的外部信息或附加上下文

+   **输入数据**：我们想要得到答案的输入/问题

+   **输出指示器**：我们希望模型输出的格式类型

使用`spacy-llm`，我们将提示定义为任务。当使用 LLMs 构建 spaCy 管道时，每个 LLM 组件都使用一个**任务**和一个**模型**来定义。**任务**定义了提示和解析结果的提示和功能。**模型**定义了 LLM 模型以及如何连接到它。

现在您已经了解了 LLMs 是什么以及如何与它们交互，让我们在管道中使用一个`spacy-llm`组件。在下一节中，我们将创建一个使用 LLM 来总结文本的管道。

# 使用 LLMs 和 spacy-llm 进行文本摘要

每个 `spacy-llm` 组件都有一个任务定义。spaCy 有一些预定义的任务，我们也可以创建自己的任务。在本节中，我们将使用 `spacy.Summarization.v1` 任务。每个任务都是通过一个提示来定义的。以下是该任务的提示，可在 [`github.com/explosion/spacy-llm/blob/main/spacy_llm/tasks/templates/summarization.v1.jinja`](https://github.com/explosion/spacy-llm/blob/main/spacy_llm/tasks/templates/summarization.v1.jinja) 找到：

```py
You are an expert summarization system. Your task is to accept Text as input and summarize the Text in a concise way.
{%- if max_n_words -%}
{# whitespace #}
The summary must not, under any circumstances, contain more than {{ max_n_words }} words.
{%- endif -%}
{# whitespace #}
{%- if prompt_examples -%}
{# whitespace #}
Below are some examples (only use these as a guide):
{# whitespace #}
{%- for example in prompt_examples -%}
{# whitespace #}
Text:
'''
{{ example.text }}
'''
Summary:
'''
{{ example.summary }}
'''
{# whitespace #}
{%- endfor -%}
{# whitespace #}
{%- endif -%}
{# whitespace #}
Here is the Text that needs to be summarized:
'''
{{ text }}
'''
Summary:
```

`spacy-llm` 使用 `Jinja` 模板来定义指令和示例。Jinja 使用占位符在模板中动态插入数据。最常见的占位符是 `{{ }}` ，`{% %}` 和 `{# #}` 。`{{ }}` 用于添加变量或表达式，`{% %}` 与流程控制语句一起使用，`{# #}` 用于添加注释。让我们看看这些占位符如何在 `spacy.Summarization.v1` 模板中使用。

我们可以要求模型输出具有特定最大单词数的摘要。`max_n_words` 的默认值是 `null`。如果我们在配置中设置了此参数，模板将包括此数字：

```py
{%- if max_n_words -%}
{# whitespace #}
The summary must not, under any circumstances, contain more than {{ max_n_words }} words.
{%- endif -%}
```

`Few-shot prompting` 是一种技术，它包括提供一些期望的输入和输出示例（通常为 1 到 5 个），以展示我们希望模型如何生成结果。这些示例有助于模型更好地理解模式，而无需使用大量示例进行微调。`spacy.Summarization.v1` 任务有一个 `examples` 参数来生成少量示例。

现在我们已经了解了总结模板任务的工作原理，是时候开始处理 `spacy-llm` 的其他元素了，即 **模型**。我们将使用 Anthropic 的 `Claude 2` 模型。为了确保连接到该模型的凭据可用，您可以在计算机上的控制台中运行 `export ANTHROPIC_API_KEY="..."` 命令。现在让我们使用 `python3 -m pip install spacy-llm==0.7.2` 命令安装该包。我们将使用 `config.cfg` 文件来加载管道（如果您需要复习，可以回到 *第六章* 中的 *使用 spaCy config.cfg 文件* 部分）。让我们构建配置文件：

1.  首先，我们将定义 `nlp` 部分，其中我们应该定义我们的管道的语言和组件。我们只使用 `llm` 组件：

    ```py
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    ```

1.  现在，是时候指定 `components` 部分。为了初始化 `llm` 组件，我们将写入 `factory = "llm"`；然后，我们将指定任务和模型：

    ```py
    [components]
    [components.llm]
    factory = "llm"
    [components.llm.task]
    @llm_tasks = "spacy.Summarization.v1"
    examples = null
    max_n_words = null
    [components.llm.model]
    @llm_models = "spacy.Claude-2.v2"
    config = {"max_tokens_to_sample": 1024}
    ```

1.  为了加载此管道，我们将通过 `spacy_llm.util` 的 `assemble` 方法传递此配置文件的路径。让我们要求模型总结本章的 *LLMs 和提示工程基础* 部分：

    ```py
    from spacy_llm.util import assemble
    nlp = assemble("config.cfg")
    content = """
    As we saw on Chapter 6, Language Modeling is the task of predicting the next token given the sequence of previous tokens.
    [...]
    Now that you know what LLMs are and how to interact with them, let's use a spacy-llm component in a pipeline. In the next section we're going to create a pipeline to summarize texts using a LLM.
    """
    doc = nlp(content)
    print(doc._.summary)
    ```

1.  `spacy.Summarization.v1` 任务默认将摘要添加到 `._.summary` 扩展属性中。以下是模型的响应：

    ```py
    'Here is a concise summary of the key points from the text:\n\nLanguage models predict the next token in a sequence. Pre-trained language models (PLMs) are trained in a self-supervised way to learn general representations of language. PLMs are fine-tuned for downstream tasks. Large language models (LLMs) like GPT-3 have billions of parameters and are trained on huge datasets. LLMs can perform a variety of tasks including translation, coding assistance, scientific writing, and legal analysis. However, LLMs require lots of compute resources, are slow, and can sometimes "hallucinate" plausible but incorrect information. We interact with LLMs using prompts that provide instructions, context, input data, and indicate the desired output format. Spacy-llm allows defining LLM components in spaCy pipelines using tasks to specify prompts and models to connect to the LLM. The text then explains we will create a pipeline to summarize text using a LLM component.'
    ```

很好，对吧？你可以检查其他参数来在此处自定义任务 [`spacy.io/api/large-language-models#summarization-v1`](https://spacy.io/api/large-language-models#summarization-v1) 。一些其他可用的 `spacy-llm` 任务包括 `spacy.EntityLinker.v1** , **spacy.NER.v3** , **spacy.SpanCat.v3** , **spacy.TextCat.v3` , 和 `spacy.Sentiment.v1` 。但除了使用这些预构建的任务之外，你还可以创建自己的任务，这不仅增强了 `spacy-llm` 的功能，而且为构建 NLP 管道时遵循最佳实践提供了一种有组织的方式。让我们在下一节中学习如何做到这一点。

# 创建自定义 spacy-llm 任务

在本节中，我们将创建一个任务，给定一个来自 [`dummyjson.com/docs/quotes`](https://dummyjson.com/docs/quotes) 的引用，模型应提供引用的上下文。以下是一个示例：

```py
Quote: We must balance conspicuous consumption with conscious capitalism.
Context: Business ethics.
```

创建自定义 `spacy-llm` 任务的第一个步骤是创建提示并将其保存为 Jinja 模板。以下是此任务的模板：

```py
You are an expert at extracting context from text.
Your tasks is to accept a quote as input and provide the context of the quote.
This context will be used to group the quotes together.
Do not put any other text in your answer and provide the context in 3 words max. The quote should have one context only.
{# whitespace #}
{# whitespace #}
Here is the quote that needs classification
{# whitespace #}
{# whitespace #}
Quote:
'''
{{ text }}
```

我们将这个保存到名为 `templates/quote_context_extract.jinja` 的文件中。下一步是创建任务的类。这个类应该实现两个函数：

+   `generate_prompts(docs: Iterable[Doc]) -> Iterable[str]` : 这个函数将 spaCy `Doc` 对象列表转换为提示列表。

+   `parse_responses(docs: Iterable[Doc], responses: Iterable[str]) -> Iterable[Doc]` : 这个函数将 LLM 输出解析为 spaCy `Doc` 对象。

`generate_prompts()` 方法将使用我们的 Jinja 模板，而 `parse_responses()` 方法将接收模型的响应并添加上下文扩展属性到我们的 `Doc` 对象。让我们创建 `QuoteContextExtractTask` 类：

1.  首先，我们导入所有需要的函数并设置模板的目录：

    ```py
    from pathlib import Path
    from spacy_llm.registry import registry
    import jinja2
    from typing import Iterable
    from spacy.tokens import Doc
    TEMPLATE_DIR = Path("templates")
    ```

1.  现在，让我们创建一个方法来读取 Jinja 模板中的文本：

    ```py
    def read_template(name: str) -> str:
        """Read the text from a Jinja template using pathlib"""
        path = TEMPLATE_DIR / f"{name}.jinja"
        if not path.exists():
            raise ValueError(f"{name} is not a valid template.")
        return path.read_text()
    ```

1.  最后，我们可以开始创建 `QuoteContextExtractTask` 类。让我们从创建 `__init__()` 方法开始。这个类应该使用 Jinja 模板的名称和一个字段字符串来设置任务将添加到 `Doc` 对象的扩展属性名称：

    ```py
    class QuoteContextExtractTask:
        def __init__(self, template: str = "quotecontextextract",
                     field: str = "context"):
            self._template = read_template(template)
            self._field = field
    ```

1.  现在我们将创建一个方法来构建提示。Jinja 使用 `Environment` 对象从文件中加载模板。我们将使用 `from_string()` 方法从文本构建模板并生成它。每次此方法内部运行时，它将渲染模板，用模板中的 `{{text}}` 变量的值替换模板的 `doc.text` :

    ```py
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
            )
            yield prompt
    ```

1.  我们现在可以编写类的最后一个方法。`parse_responses()` 方法将添加模型的响应到 `Doc` 对象。首先，我们创建一个辅助方法来添加扩展属性，如果它不存在。为了设置扩展属性，我们将使用 Python 的 `setattr ()` 方法，这样我们就可以使用类字段变量动态设置属性：

    ```py
      def _check_doc_extension(self):
          """Add extension if need be."""
          if not Doc.has_extension(self._field):
              Doc.set_extension(self._field, default=None)
        def parse_responses(
            self, docs: Iterable[Doc], responses: Iterable[str]
        ) -> Iterable[Doc]:
            self._check_doc_extension()
            for doc, prompt_response in zip(docs, responses):
                try:
                    setattr(
                        doc._,
                        self._field,
                        prompt_response[0].strip(),
                    )
                except ValueError:
                    setattr(doc._, self._field, None)
                yield doc
    ```

1.  要在`config.cfg`文件中使用这个类，我们需要将任务添加到`spacy-llm`的`llm_tasks`注册表中：

    ```py
    @registry.llm_tasks("my_namespace.QuoteContextExtractTask.v1")
    def make_quote_extraction() -> "QuoteContextExtractTask":
        return QuoteContextExtractTask()
    ```

完成了！将此保存到`quote.py`类中。以下是应该在这个脚本中的完整代码：

```py
from pathlib import Path
from spacy_llm.registry import registry
import jinja2
from typing import Iterable
from spacy.tokens import Doc
TEMPLATE_DIR = Path("templates")
def read_template(name: str) -> str:
    """Read the text from a Jinja template using pathlib"""
    path = TEMPLATE_DIR / f"{name}.jinja"
    if not path.exists():
        raise ValueError(f"{name} is not a valid template.")
    return path.read_text()
@registry.llm_tasks("my_namespace.QuoteContextExtractTask.v1")
def make_quote_extraction() -> "QuoteContextExtractTask":
    return QuoteContextExtractTask()
class QuoteContextExtractTask:
    def __init__(self, template: str = "quote_context_extract",
                 field: str = "context"):
        self._template = read_template(template)
        self._field = field
    def generate_prompts(self, 
        docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self_template)
        for doc in docs:
            prompt = _template.render(
                text=doc.text,
            )
            yield prompt
    def _check_doc_extension(self):
        """Add extension if need be."""
        if not Doc.has_extension(self._field):
            Doc.set_extension(self._field, default=None)
  def parse_responses(
      self, docs: Iterable[Doc], responses: Iterable[str]
  ) -> Iterable[Doc]:
        self._check_doc_extension()
        for doc, prompt_response in zip(docs, responses):
            try:
                setattr(
                    doc._,
                    self._field,
                    prompt_response[0].strip(),
                ),
            except ValueError:
                setattr(doc._, self._field, None)
            yield doc
```

1.  现在，让我们通过首先创建`config_custom_task.cfg`文件来测试我们的自定义任务：

    ```py
    [nlp]
    lang = "en"
    pipeline = ["llm"]
    [components]
    [components.llm]
    factory = "llm"
    [components.llm.task]
    @llm_tasks = "my_namespace.QuoteContextExtractTask.v1"
    [components.llm.model]
    @llm_models = "spacy.Claude-2.v2"
    config = {"max_tokens_to_sample": 1024}
    ```

1.  最后，我们可以使用这个文件组装`nlp`对象，并打印出引语的上下文。别忘了从`quote.py`导入`QuoteContextExtractTask`，这样 spaCy 就知道从哪里加载这个任务：

    ```py
    from spacy_llm.util import assemble
    from quote import QuoteContextExtractTask
    nlp = assemble("config_custom_task.cfg")
    quote = "Life isn't about getting and having, it's about giving and being."
    doc = nlp(quote)
    print("Context:", doc._.context)
    >>> Context: self-improvement
    ```

在本节中，你创建了一个自定义的`spacy-llm`任务来从给定的引语中提取上下文。这种方法不仅允许你根据需求定制高度具体的 NLP 任务，而且还提供了一种结构化的方式将软件工程的最佳实践，如模块化和可重用性，集成到你的 NLP 管道中。

# 摘要

在本章中，我们探讨了如何利用**大型语言模型**（`LLMs`）在 spaCy 管道中使用`spacy-llm`库。我们回顾了 LLMs 和提示工程的基础知识，强调它们作为多功能工具在执行各种 NLP 任务中的作用，从文本分类到摘要。然而，我们也指出了 LLMs 的局限性，例如它们的高计算成本和产生幻觉的倾向。然后，我们展示了如何通过定义任务和模型将 LLMs 集成到 spaCy 管道中。具体来说，我们实现了一个摘要任务，随后创建了一个自定义任务来从引语中提取上下文。这个过程涉及到创建提示的 Jinja 模板和定义生成和解析响应的方法。

在下一章中，我们将回到更传统的机器学习，学习如何使用 spaCy 从头开始标注数据和训练管道组件。
