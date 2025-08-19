# 5

# 使用 spaCy 管道提取语义表示

在本章中，我们将把在*第四章*中学到的知识应用到**航空公司旅行信息系统**（`ATIS`）这个知名机票预订系统数据集上。数据包括用户询问信息的句子。首先，我们将提取命名实体，使用`SpanRuler`创建自己的提取模式。然后，我们将使用`DepedencyMatcher`模式确定用户话语的意图。我们还将使用代码提取意图，并创建自己的自定义 spaCy 组件，使用`Language.pipe()`方法以更快的速度处理大型数据集。

在本章中，我们将涵盖以下主要主题：

+   使用`SpanRuler`提取命名实体

+   使用`DependencyMatcher`提取依存关系

+   使用扩展属性创建管道组件

+   使用大数据集运行管道

# 技术要求

在本章中，我们将处理一个数据集。数据集和本章代码可以在[`github.com/PacktPublishing/Mastering-spaCy-Second-Edition`](https://github.com/PacktPublishing/Mastering-spaCy-Second-Edition)找到。

我们将使用`pandas`库来操作我们的数据集。我们还将使用`wget`命令行工具。`pandas`可以通过`pip`安装，而`wget`在许多 Linux 发行版中是预安装的。

# 使用`SpanRuler`提取命名实体

在许多自然语言处理应用中，包括语义解析，我们通过检查实体类型并将实体提取组件放入我们的 NLP 管道中来开始寻找文本中的意义。命名实体在理解用户文本的意义中起着关键作用。

我们还将通过从我们的语料库中提取命名实体来启动语义解析管道。为了了解我们想要提取哪种类型的实体，首先，我们将了解 ATIS 数据集。

## 了解 ATIS 数据集

在本章中，我们将使用 ATIS 语料库。ATIS 是一个知名的数据集；它是意图分类的标准基准数据集之一。该数据集包括想要预订航班和/或获取航班信息（包括航班费用、目的地和时间表）的客户的话语。

无论 NLP 任务是什么，你都应该用肉眼仔细检查你的语料库。我们想要了解我们的语料库，以便将我们对语料库的观察整合到我们的代码中。在查看我们的文本数据时，我们通常会关注以下方面：

+   有哪些类型的语句？是短文本语料库，还是语料库由长文档或中等长度的段落组成？

+   语料库包括哪些类型的实体？例如，包括人名、城市名、国家名、组织名等。我们想要提取哪些？

+   标点符号是如何使用的？文本是否正确使用了标点，或者根本就没有使用标点？

+   用户是否遵循了语法规则？语法规则是如何遵循的？大写是否正确？是否有拼写错误？

在开始任何处理之前，我们将检查我们的语料库。以下是方法：

1.  让我们继续下载数据集：

    ```py
    mkdir data
    wget -P data https://github.com/PacktPublishing/Mastering-spaCy-Second-Edition/blob/main/chapter_05/data/atis_intents.csv
    ```

    数据集是一个两列的 CSV 文件。

重要提示

如果你在一个 Jupyter 笔记本上运行代码，你可以在命令前添加一个 `!` 来在那里运行它们。

1.  接下来，我们将使用 `pandas` 对数据集统计进行一些洞察。`pandas` 是一个流行的数据处理库，常被数据科学家使用。你可以在 [`pandas.pydata.org/docs/getting_started/intro_tutorials/`](https://pandas.pydata.org/docs/getting_started/intro_tutorials/) 上了解更多信息。

    ```py
    import pandas as pd
    df = pd.read_csv("data/atis_intents.csv", header=None, 
                     names=["utterance", "text"])
    df.shape
    ```

    `shape` 属性返回一个表示 `DataFrame` 维度的元组。我们可以看到数据集有两列和 4,978 行。

1.  让我们创建一个条形图来查看数据集中的出口数量：

    ```py
    df["utterance"].value_counts().plot.barh()
    ```

    `value_counts()` 方法返回一个包含唯一值计数的序列。pandas 库在底层使用 Matplotlib 来绘制条形图；这是结果：

![图 5.1 – 出口频率条形图](img/B22441_05_01.jpg)

图 5.1 – 出口频率条形图

大多数用户请求是关于航班的信息，其次是关于机票的请求。然而，在提取出口之前，我们将学习如何提取命名实体。让我们在下一节中这样做。

## 定义位置实体

在本节中，我们的目标是提取 **位置** 实体。`en_core_web_sm` 模型的管道已经有一个 `NER` 组件。让我们看看默认 NER 模型从 `"i want to fly from boston at 838 am and arrive in denver at 1110 in the morning"` 这句话中提取了哪些实体：

1.  首先，我们导入库：

    ```py
    import spacy
    from spacy import displacy
    ```

1.  然后我们加载 spaCy 模型并处理一个句子：

    ```py
    nlp = spacy.load("en_core_web_sm")
    text = "i want to fly from boston at 838 am and arrive in denver at 1110 in the morning"
    doc = nlp(text)
    ```

1.  最后，我们使用 `displacy` 显示实体：

    ```py
    displacy.render(doc, style='ent')
    ```

    我们可以在 *图 5.2* 中看到结果：

![图 5.2 – NER 组件提取的实体](img/B22441_05_02.jpg)

图 5.2 – NER 组件提取的实体

NER 模型为 `boston` 和 `denver` 找到 **全球政治实体` ( **GPE` )，但仅知道这些城市是不够的。我们想知道他们想从哪里飞以及飞往何处。这意味着在这种情况下，**介词**（一个包括介词和后置介词的通用术语）很重要。spaCy 使用通用的 **词性` ( **POS` ) 标签，因此介词标签被命名为 `"ADP"`。你可以在词汇表中看到所有 POS 标签、依存标签或 spaCy 实体类型的描述（[`github.com/explosion/spaCy/blob/master/spacy/glossary.py`](https://github.com/explosion/spaCy/blob/master/spacy/glossary.py)）。

回到之前的句子示例，**从波士顿**和**在丹佛**是我们想要提取的实体。由于我们知道需要哪些 POS 标签和 GPE 实体来创建新的实体，因此实现这种提取的一个好方法就是依赖于 NLP 管道中的`Tagger`和`EntityRecognizer`组件。我们将通过创建基于标签的规则来提取标记。spaCy 使用`SpanRuler`组件使得这一过程变得简单易行。

## 将 SpanRuler 组件添加到我们的处理管道中

使用 spaCy 定制 NLP 管道非常简单。每个管道都是通过 spaCy 组件的组合创建的。一开始可能不太清楚，但当我们加载现成的 spaCy 模型时，它已经包含了几种不同的组件：

```py
nlp.pipe_names
```

`nlp.pipe_names`属性按顺序返回组件名称。*图 5.3*显示了所有这些组件。

![图 5.3 – en_core_web_sm 模型的默认组件](img/B22441_05_03.jpg)

图 5.3 – en_core_web_sm 模型的默认组件

我们可以看到，`en_core_web_sm`模型默认包含`['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']`组件。每个组件返回一个处理后的`Doc`对象，然后传递给下一个组件。

您可以使用`Language.add_pipe()`方法向处理管道添加组件（这里 nlp 是 Language 类的对象，我们将使用它来调用`add_pipe()`）。该方法期望一个包含组件名称的字符串。在底层，此方法负责创建组件，将其添加到管道中，然后返回组件对象。

`SpanRuler`组件是一个现成的基于规则的跨度命名实体识别组件。该组件允许您将跨度添加到`Doc.spans`和/或`Doc.ents`。让我们第一次尝试使用它：

1.  首先，我们调用`add_pipe()`方法：

    ```py
    spanruler_component = nlp.add_pipe("span_ruler")
    ```

1.  然后我们使用`add_patterns()`方法添加模式。它们应该使用包含`"label"`和`"pattern"`键的字典列表来定义：

    ```py
    patterns_location_spanruler = [
        {"label": "LOCATION", 
         "pattern": [{"POS": "ADP"}, {"ENT_TYPE": "GPE"}]}]
    spanruler_component.add_patterns(patterns_location_spanruler)
    ```

    spaCy 使用*Thinc*（[`thinc.ai/docs/api-config#registry`](https://thinc.ai/docs/api-config#registry)）注册表，这是一个将字符串键映射到函数的系统。`"span_ruler"`字符串名称是引用`SpanRuler`组件的字符串。然后我们定义一个名为`LOCATION`的模式，并使用`add_patterns()`方法将其添加到组件中。

1.  与`doc.ents`不同，`doc.spans`允许重叠匹配。默认情况下，`SpanRuler`将匹配项作为`spans`添加到`doc.spans["ruler"]`跨度组中。让我们再次处理文本并检查`SpanRuler`是否完成了其工作。由于组件将 Span 添加到`"ruler"`键，我们需要指定这一点以使用`displacy`渲染跨度：

    ```py
    doc = nlp(text)
    options = {"spans_key": "ruler"}
    displacy.render(doc, style='span', options=options)
    ```

    让我们看看*图 5.4*的结果。

![图 5.4 – 使用 SpanRuler 提取的跨度](img/B22441_05_04.jpg)

图 5.4 – 使用 SpanRuler 提取的跨度

我们可以看到，组件识别了`"from boston"`和`"in denver"`跨度。`SpanRuler`有一些您可以更改的设置。这可以通过`nlp.pipe()`方法上的 config 参数或使用`config.cfg`文件来完成。让我们将跨度添加到`Doc.ents`而不是`doc.spans["ruler"]`：

1.  首先，我们移除了管道中的组件，因为我们只有一个具有相同名称的组件：

    ```py
    nlp.remove_pipe("span_ruler")
    ```

1.  然后我们将组件的`"annotate_ents"`参数设置为`True`。由于我们的模式需要`EntityRecognizer`添加的实体，因此我们还需要将覆盖参数设置为`False`，这样我们就不覆盖它们：

    ```py
    config = {"annotate_ents": True, "overwrite": False}
    ```

1.  现在我们使用此配置再次创建组件，添加之前创建的模式，并再次处理文本。

    ```py
    spanruler_component_v2 = nlp.add_pipe(
        "span_ruler", config=config)
    spanruler_component_v2.add_patterns(patterns_location_spanruler)
    doc = nlp(text)
    displacy.render(doc, style='ent')
    ```

    通过做所有这些，匹配项变成了实体，而不是跨度。

    让我们看看图 5.5 中的新实体：

![图 5.5 – 使用 SpanRuler 提取的实体](img/B22441_05_05.jpg)

图 5.5 – 使用 SpanRuler 提取的实体

您可以看到，`{'GPE', 'boston'}`和`{'GPE', 'denver'}`实体不再存在；现在分别是`{'from boston', 'LOCATION'}`和`{'in denver', 'LOCATION'}`。`Doc.ents`中不允许重叠实体，因此默认使用`util.filter_spans`函数进行过滤。此函数保留较短的跨度中第一个最长的跨度。

您可以覆盖大多数`SpanRuler`设置。一些可用的设置如下：

+   `spans_filter`：一个在将跨度分配给`doc.spans`之前过滤跨度的方法

+   `ents_filter`：一个在将跨度分配给`doc.ents`之前过滤跨度的方法

+   `validate`：一个设置是否应该验证模式或将其作为验证传递给`Matcher`和`PhraseMatcher`的方法

在本节中，我们学习了如何使用`SpanRuler`创建和提取实体。有了提取**位置**实体的模式，我们现在可以继续提取话语的意图。让我们在下一节中使用`DependencyMatcher`来完成这项工作。

# 使用 DependencyMatcher 提取依赖关系

为了提取话语的**意图**，我们需要根据它们之间的语法关系匹配标记。目标是找出用户携带的意图类型——预订航班、在他们已预订的航班上购买餐点、取消航班等。每个意图都包括一个动词（预订）和网页所作用的对象（航班、酒店、餐点等）。

在本节中，我们将从话语中提取及物动词及其直接宾语。我们将通过提取及物动词及其直接宾语来开始我们的意图识别部分。在我们继续提取及物动词及其直接宾语之前，让我们首先快速回顾一下及物动词和直接/间接宾语的概念。

## 语言学入门

让我们探索一些与句子结构相关的语言概念，包括动词和动词宾语关系。动词是句子中非常重要的组成部分，因为它表示句子中的*动作*。句子的宾语是受到动词动作影响的*事物或人*。因此，句子动词和宾语之间存在自然联系。及物性的概念捕捉了动词宾语关系。及物动词是需要作用对象的动词。让我们看看一些例子：

```py
I bought flowers.
He loved his cat.
He borrowed my book.
```

在这些例子句子中，`bought`、`loved`和`borrowed`是及物动词。在第一个句子中，`bought`是及物动词，`flowers`是其宾语，即句子主语所购买的事物。`I Loved` – `his cat`和`borrowed` – `my book`是及物动词宾语例子。我们再次关注第一个句子——如果我们删除`flowers`宾语会发生什么？让我们看看这里的情况：

```py
I bought
```

你买了*什么*？如果没有宾语，这个句子就完全没有意义了。在前面的句子中，每个宾语都完成了动词的意义。这是理解一个动词是否及物的一种方法——删除宾语并检查句子是否在语义上仍然完整。

有些动词是及物的，有些动词是不及物的。不及物动词是及物动词的相反，它不需要作用对象。让我们看看一些例子：

```py
Yesterday I slept for 8 hours.
The cat ran towards me.
When I went out, the sun was shining.
Her cat died 3 days ago.
```

在所有前面的句子中，即使没有宾语，动词也有意义。如果我们删除除了主语和宾语之外的所有单词，这些句子仍然是有意义的：

```py
I slept.
The cat ran.
The sun was shining.
Her cat died.
```

将不及物动词与宾语搭配是没有意义的。你不能跑某人或某物，你不能使某人或某物发光，你当然也不能使某人或某物死亡。

### 句子宾语

正如我们之前提到的，宾语是受到动词动作影响的物体或人。动词所陈述的动作是由句子主语执行的，而句子宾语受到动作的影响。一个句子可以是直接的或间接的。直接宾语回答了*谁*和*什么*的问题。你可以通过问**主语{动词}什么/谁？**来找到直接宾语。以下是一些例子：

```py
I bought flowers.       I bought what?      - flowers
He loved his cat.       He loved who?       - his cat
He borrowed my book.    He borrowed what?   - my book
```

一个*间接宾语*回答了**为了什么**、**为了谁**和/或**给谁**的问题。让我们看看一些例子：

```py
He gave me his book.    He gave his book to whom?   - me
He gave his book to me. He gave his book to whom?   - me
```

间接宾语通常由介词`to`、`for`、`from`等引导。正如你从这些例子中看到的，间接宾语也是一个宾语，并且受到动词动作的影响，但它在句子中的角色略有不同。间接宾语有时被视为直接宾语的接受者。

这就是您需要了解的关于及物/不及物动词和直接/间接宾语的知识，以便消化本章的内容。如果您想了解更多关于句子语法的知识，可以阅读 Emily Bender 的优秀书籍 *自然语言处理的语言学基础* ([`dl.acm.org/doi/book/10.5555/2534456`](https://dl.acm.org/doi/book/10.5555/2534456))。我们已经涵盖了句子语法的基础知识，但这仍然是一个深入了解语法的极好资源。

## 使用 `DependencyMatcher` 组件匹配模式

`DependencyMatcher` 组件使我们能够将模式与提取信息相匹配，但与 `SpanRuler` 模式定义的相邻标记列表不同，`DependencyMatcher` 模式匹配指定它们之间关系的标记。该组件与 `DependencyParser` 组件提取的依存关系一起工作。让我们通过一个示例看看这个组件提取的信息类型：

```py
text = "show me flights from denver to philadelphia on tuesday"
doc = nlp(text)
displacy.render(doc, style='dep')
```

让我们看看 *图 5.6* 的结果： 

![图 5.6 – 句子的依存弧（句子其余部分省略）](img/B22441_05_06.jpg)

图 5.6 – 句子的依存弧（句子其余部分省略）

提取的依存标签是弧线下方的标签。*Show* 是一个及物动词，在这个句子中，它的直接宾语是 *flight*。这个依存关系由 `DependencyParser` 组件提取，并标记为 `dobj`（直接宾语）。

我们的目标是提取意图，因此我们将定义始终寻找 **动词** 和其 `dobj` 依存关系的模式。`DependencyMatcher` 使用 Semgrex 操作符来定义模式。**Semgrex 语法** 可能一开始会让人困惑，所以让我们一步一步来。

`DependencyMatcher` 模式由一系列字典组成。第一个字典使用 `RIGHT_ID` 和 `RIGHT_ATTRS` 定义一个锚点标记。`RIGHT_ID` 是关系右侧节点的唯一名称，`RIGHT_ATTRS` 是要匹配的标记属性。模式格式与 `SpanRuler` 中使用的相同模式。在我们的模式中，锚点标记将是 `dobj` 标记，因此第一个字典定义如下：

```py
pattern = [
    {
        "RIGHT_ID": "direct_object_token",
        "RIGHT_ATTRS": {"DEP": "dobj"}
    }
]
```

如 spaCy 的文档所述 ([`spacy.io/usage/rule-based-matching/#dependencymatcher`](https://spacy.io/usage/rule-based-matching/#dependencymatcher))，在第一个字典之后，模式字典应包含以下键：

+   `LEFT_ID`：关系左侧节点的名称，该名称已在早期节点中定义

+   `REL_OP`：描述两个节点之间关系的操作符

+   `RIGHT_ID`：关系右侧节点的唯一名称

+   `RIGHT_ATTRS`：与 `SpanRuler` 中提供的正则标记模式相同的格式，用于匹配关系右侧节点的标记属性

给定这些键，我们通过指示关系的左侧节点、为新右侧节点定义一个名称以及指示描述两个节点之间关系的运算符来构建模式。回到我们的例子，在将 `direct_object_token` 定义为锚点后，我们将下一个字典的 `RIGHT_ID` 设置为 `VERB` 标记，并将运算符定义为 `direct_object_token < verb_token`，因为直接宾语是动词的 *直接依赖项*。以下是 `DependencyMatcher` 支持的一些其他运算符（您可以在[这里](https://spacy.io/usage/rule-based-matching/#dependencymatcher-operators)查看完整的运算符列表）：

+   `A < B` : A 是 B 的直接依赖项

+   `A > B` : A 是 B 的直接头

+   `A << B` : A 是在 dep → head 路径上跟随 B 的链中的依赖项

+   `A >> B` : A 是在 head → dep 路径上跟随 B 的链中的头

如果这些操作让您有点头疼，那也发生在我身上。这只是其中的一小部分，您可以在[这里](https://spacy.io/usage/rule-based-matching#dependencymatcher-operators)查看完整的运算符列表。好吧，让我们回到我们的例子并定义完整的模式：

```py
pattern = [
    {
        "RIGHT_ID": "direct_object_token",
        "RIGHT_ATTRS": {"DEP": "dobj"}
    },
    {
        "LEFT_ID": "direct_object_token",
        "REL_OP": "<",
        "RIGHT_ID": "verb_token",
        "RIGHT_ATTRS": {"POS": "VERB"}
    }
]
```

现在，我们可以创建 `DependencyMatcher`：

1.  首先，我们需要传递 `vocabulary` 对象（词汇与匹配器操作的文档共享）：

    ```py
    from spacy.matcher import DependencyMatcher
    matcher = DependencyMatcher(nlp.vocab)
    ```

    接下来，我们需要定义一个回调函数，该函数将接受以下参数：`matcher`、`doc`、`i` 和 `matches`。`matcher` 参数指的是匹配器实例，`doc` 是正在分析的文档，`i` 是当前匹配的索引，`matches` 是一个详细说明找到的匹配项的列表。我们将创建一个 `callback` 函数来显示一个单词的意图，例如 `bookFlight`、`cancelFlight`、`bookMeal` 等。该函数将接受匹配项的标记并打印它们的词元：

    ```py
    def show_intent(matcher, doc, i, matches):
        match_id, token_ids = matches[i]
        verb_token = doc[token_ids[1]]
        dobj_token = doc[token_ids[0]]
        intent = verb_token.lemma_ + dobj_token.lemma_.capitalize()
    print("Intent:", intent)
    ```

1.  要向 `matcher` 添加规则，我们指定一个 ID 键、一个或多个模式以及可选的回调函数来处理匹配项。最后，我们再次处理文本并调用 `matcher` 对象，将此 `doc` 作为参数传递：

    ```py
    matcher.add("INTENT", [pattern], on_match=show_intent)
    doc = nlp("show me flights from denver to philadelphia on tuesday")
    matches = matcher(doc)
    ```

    太棒了！代码打印出 `Intent: showFlightIntent` ，所以识别是成功的。在这里，我们识别了一个单一意图，但某些话语可能携带多个意图。例如，考虑以下来自语料库的话语：

    ```py
    show all flights and fares from denver to san francisco
    ```

    在这里，用户想要列出所有航班，同时查看票价信息。一种处理方式是将这些意图视为单一且复杂的意图。处理此类话语的常见方式是用多个意图标记话语。

让我们看看 `DEP` 依赖关系由 `DependencyParser` 提取的：

![图 5.7 – 新句子的依存弧（句子其余部分省略）](img/B22441_05_07.jpg)

图 5.7 – 新句子的依存弧（句子其余部分省略）

在 *图 5.7* 中，我们看到 `dobj` 弧连接了 `show` 和 `flights`。`conj`（连词）弧将 `flights` 和 `fares` 连接起来，表示连词关系。这种关系是通过连词如 `and` 或 `or` 构建的，表示一个名词通过这个连词与另一个名词相连。现在让我们编写代码来识别这两个意图：

1.  将弧关系转换为 `REL_OP` 操作符，`direct_object_token` 将成为这次关系的头，因此我们将使用 `>` 操作符，因为 `direct_object_token` 是新的 `conjunction_token` 的 **直接头**。这是匹配两个意图的新模式：

    ```py
    pattern_two = [
        {
            "RIGHT_ID": "direct_object_token",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        },
        {
            "LEFT_ID": "direct_object_token",
            "REL_OP": "<",
            "RIGHT_ID": "verb_token",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "direct_object_token",
            "REL_OP": ">",
            "RIGHT_ID": "conjunction_token",
            "RIGHT_ATTRS": {"DEP": "conj"}
        }
    ]
    ```

1.  我们还需要更新回调函数，使其能够打印出两个意图：

    ```py
    def show_two_intents(matcher, doc, i, matches):
        match_id, token_ids = matches[i]
        verb_token = doc[token_ids[1]]
        dobj_token = doc[token_ids[0]]
        conj_token = doc[token_ids[2]]
        intent = verb_token.lemma_ + \
            dobj_token.lemma_.capitalize() + ";" + \
            verb_token.lemma_ + conj_token.lemma_.capitalize()
        print("Two intents:", intent)
    ```

1.  现在我们只需要将这个新规则添加到匹配器中。由于模式 ID 已经存在，模式将被扩展：

    ```py
    matcher.add("TWO_INTENTS", [pattern_two], 
                on_match=show_two_intents)
    ```

1.  在设置好所有这些之后，我们现在可以再次找到匹配项：

    ```py
    doc = nlp("show all flights and fares from denver to san francisco")
    matches = matcher(doc)
    ```

现在匹配器找到了两个模式的标记，第一个模式和这个新的模式，它匹配两个意图。到目前为止，我们只是 *打印意图*，但在实际设置中，将此信息存储在 `Doc` 对象上是一个好主意。为此，我们将创建自己的 spaCy 组件。让我们在下一节学习如何做到这一点。

# 使用扩展属性创建管道组件

要创建我们的组件，我们将使用 `@Language.factory` 装饰器。组件工厂是一个可调用对象，它接受设置并返回一个 `pipeline component function`。`@Language.factory` 装饰器还将自定义组件的名称添加到注册表中，使得可以使用 `.add_pipe()` 方法将组件添加到管道中。

spaCy 允许你在 `Doc`、`Span` 和 `Token` 对象上设置任何自定义属性和方法，这些属性和方法将作为 `Doc._.`、`Span._.` 和 `Token._.` 可用。在我们的案例中，我们将向 `Doc` 添加 `Doc._.intent` 属性，利用 spaCy 的数据结构来存储我们的数据。

我们将在一个 Python 类内部实现组件逻辑。spaCy 期望 `__init__()` 方法接受 `nlp` 和 `name` 参数（spaCy 会自动填充它们），而 `__call__()` 方法应该接收并返回 `Doc`。

让我们创建 `IntentComponent` 类：

1.  首先，我们创建类。在 `__init__()` 方法中，我们创建 `DependencyMatcher` 实例，将模式添加到匹配器中，并设置 `intent` 扩展属性：

    ```py
    class IntentComponent:
        def __init__(self, nlp: Language):
            self.matcher = DependencyMatcher(nlp.vocab)
            pattern = [
                {
                    "RIGHT_ID": "direct_object_token",
                    "RIGHT_ATTRS": {"DEP": "dobj"}
                },
                {
                    "LEFT_ID": "direct_object_token",
                    "REL_OP": "<",
                    "RIGHT_ID": "verb_token",
                    "RIGHT_ATTRS": {"POS": "VERB"}
                }
            ]
            pattern_two = [
                {
                    "RIGHT_ID": "direct_object_token",
                    "RIGHT_ATTRS": {"DEP": "dobj"}
                },
                {
                    "LEFT_ID": "direct_object_token",
                    "REL_OP": "<",
                    "RIGHT_ID": "verb_token",
                    "RIGHT_ATTRS": {"POS": "VERB"}
                },
                {
                    "LEFT_ID": "direct_object_token",
                    "REL_OP": ">",
                    "RIGHT_ID": "conjunction_token",
                    "RIGHT_ATTRS": {"DEP": "conj"}
                }
            ]
            self.matcher.add("INTENT", [pattern])
            self.matcher.add("TWO_INTENTS", [pattern_two])
            if not Doc.has_extension("intent"):
                Doc.set_extension("intent", default=None)
    ```

    现在，在 `__call__()` 方法内部，我们找到匹配项并检查是否是 `"TWO_INTENTS"` 匹配。如果是，我们提取该模式的标记并设置 `doc._.intent` 属性；如果不是，则在 `else` 块中，我们提取 `"INTENT"` 匹配的标记：

    ```py
        def __call__(self, doc: Doc) -> Doc:
            matches = self.matcher(doc)
            for match_id, token_ids in matches:
                string_id = nlp.vocab.strings[match_id]
                if string_id == "TWO_INTENTS":
                    verb_token = doc[token_ids[1]]
                    dobj_token = doc[token_ids[0]]
                    conj_token = doc[token_ids[2]]
                    intent = verb_token.lemma_ + \
                        dobj_token.lemma_.capitalize() + \
                        ";" + verb_token.lemma_ + \
                        conj_token.lemma_.capitalize()
                    doc._.intent = intent
                    break
            else:
                for match_id, token_ids in matches:
                    string_id = nlp.vocab.strings[match_id]
                    if string_id == "INTENT":
                        verb_token = doc[token_ids[1]]
                        dobj_token = doc[token_ids[0]]
                        intent = verb_token.lemma_ + \
                            dobj_token.lemma_.capitalize()
                        doc._.intent = intent
            return doc
    ```

    使用这段代码，我们在`Doc`中注册自定义扩展，通过在`__call__()`方法上设置`doc._.intent = intent`来找到匹配项并保存意图。

1.  现在我们有了自定义组件的类，下一步是使用装饰器来注册它：

    ```py
    @Language.factory("intent_component")
    def create_intent_component(nlp: Language, name: str):
        return IntentComponent(nlp)
    ```

重要提示

如果你正在使用 Jupyter Notebook 并且需要重新创建组件，你需要重新启动内核。如果不这样做，spaCy 会给我们一个错误，因为组件名称已经被注册。

就这样，这就是你的第一个自定义组件！恭喜！现在，为了提取意图，我们只需要将组件添加到管道中。如果我们想查看意图，我们可以通过`doc._.intent`来访问它。这是你可以这样做的方式：

```py
nlp.add_pipe("intent_component")
text = "show all flights and fares from denver to san francisco"
doc = nlp(text)
doc._.intent
```

太酷了，对吧？如果你不记得，数据集有 4,978 个语音。这不是一个非常大的数字，但如果它更大呢？spaCy 能帮助我们让它更快吗？是的！在下一节中，我们将学习如何使用`Language.pipe()`方法运行我们的管道。

# 使用大型数据集运行管道

`Language.pipe()`方法将文本作为流处理，并按顺序产生`Doc`对象。它以批量而不是逐个缓冲文本，因为这通常更有效率。如果我们想获取特定的文档，我们需要先调用`list()`，因为该方法返回一个 Python 生成器，它产生`Doc`对象。这是你可以这样做的方式：

```py
utterance_texts = df.text.to_list()
processed_docs = list(nlp.pipe(utterance_texts))
print(processed_docs[0], processed_docs[0]._.intent)
```

在前面的代码中，我们正在从本章开头加载的 DataFrame 中获取文本语音列表，并使用`.pipe()`进行批量处理。让我们通过使用和不使用`.pipe()`方法来比较时间差异：

```py
import timestart_time = time.time()
utterance_texts = df.text.to_list()
processed_docs_vanilla = [nlp(text) for text in utterance_texts]
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
>>> Execution time: 27.12 seconds
```

这给了我们 27.12 秒的时间。现在，让我们使用以下方法：

```py
import timestart_time = time.time()
utterance_texts = df.text.to_list()
processed_docs_pipe = list(nlp.pipe(utterance_texts))
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
>>> Execution time: 5.90 seconds
```

使用`nlp.pipe()`，我们在 5.90 秒内得到了相同的结果。这是一个巨大的差异。我们还可以指定`batch_size`和`n_process`来设置要使用的处理器数量。还有一个选项可以禁用组件，如果你只需要运行`.pipe()`来获取特定组件处理过的文本结果。

太棒了，我们使用自己的自定义组件完成了我们的第一个管道！恭喜！以下是管道的完整代码：

```py
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.matcher import DependencyMatcher
@Language.factory("intent_component")
def create_intent_component(nlp: Language, name: str):
@Language.factory("intent_component")
def create_intent_component(nlp: Language, name: str):
    return IntentComponent(nlp)
class IntentComponent:
    def __init__(self, nlp: Language):
                self.matcher = DependencyMatcher(nlp.vocab)
        pattern = [
            {
                "RIGHT_ID": "direct_object_token",
                "RIGHT_ATTRS": {"DEP": "dobj"}
            },
            {
                "LEFT_ID": "direct_object_token",
                "REL_OP": "<",
                "RIGHT_ID": "verb_token",
                "RIGHT_ATTRS": {"POS": "VERB"}
            }
        ]
        pattern_two = [
            {
                "RIGHT_ID": "direct_object_token",
                "RIGHT_ATTRS": {"DEP": "dobj"}
            },
            {
                "LEFT_ID": "direct_object_token",
                "REL_OP": "<",
                "RIGHT_ID": "verb_token",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "direct_object_token",
                "REL_OP": ">",
                "RIGHT_ID": "conjunction_token",
                "RIGHT_ATTRS": {"DEP": "conj"}
            }
        ]
        self.matcher.add("INTENT", [pattern])
        self.matcher.add("TWO_INTENTS", [pattern_two])
        if not Doc.has_extension("intent"):
            Doc.set_extension("intent", default=None)
    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        for match_id, token_ids in matches:
            string_id = nlp.vocab.strings[match_id]
            if string_id == "TWO_INTENTS":
                verb_token = doc[token_ids[1]]
                dobj_token = doc[token_ids[0]]
                conj_token = doc[token_ids[2]]
                intent = verb_token.lemma_ + \
                    dobj_token.lemma_. capitalize() + ";" + \
                    verb_token.lemma_ + \
                    conj_token.lemma_. capitalize()
                doc._.intent = intent
                break
        else:
            for match_id, token_ids in matches:
                string_id = nlp.vocab.strings[match_id]
                if string_id == "INTENT":
                    verb_token = doc[token_ids[1]]
                    dobj_token = doc[token_ids[0]]
                    intent = verb_token.lemma_ + \
                        dobj_token.lemma_. capitalize()
                    doc._.intent = intent
       return doc
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("intent_component")
text = "show all flights and fares from denver to san francisco"
doc = nlp(text)
doc._.intent
```

spaCy 使管道代码整洁有序，这是我们想要维护代码库时两个至关重要的品质。

# 摘要

在本章中，你学习了如何生成语音的完整语义解析。首先，你添加了一个`SpanRuler`组件来提取与用例上下文相关的 NER 实体。然后，你学习了如何使用`DependencyMatcher`通过分析句子结构来进行意图识别。接下来，你还学习了如何创建自己的自定义 spaCy 组件来提取语音的意图。最后，你看到了如何使用`Language.pipe()`方法更快地处理大型数据集。

`SpanRuler` 和 `DependencyMatcher` 都依赖于我们创建的模式。创建这些模式的过程是一个反复迭代的过程。我们分析结果，然后测试新的模式，然后再分析结果，如此循环。本章的目标是教会你如何使用这些工具，以便你可以在自己的项目中执行这个过程。

在接下来的章节中，我们将更多地转向机器学习方法。*第六章* 将介绍如何使用 spaCy 与 Transformers 结合使用。
