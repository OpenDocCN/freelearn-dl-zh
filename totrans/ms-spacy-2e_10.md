# 10

# 使用 spaCy 训练实体链接器模型

**实体链接**是将文本提及映射到外部知识库中唯一标识符的 NLP 任务。本章将探讨如何使用 spaCy 训练实体链接模型，以及如何创建用于 NLP 训练的高质量数据集的最佳实践。我们还将学习如何使用自定义语料库读取器来训练 spaCy 组件。有了这些知识，你可以自定义任何 spaCy 组件，以便在训练模型时使用。

本章将涵盖以下内容：

+   理解实体链接在 NLP 中的概念和重要性

+   创建用于 NLP 训练的高质量数据集的最佳实践

+   使用 spaCy 训练**实体链接器**组件

+   利用自定义语料库读取器来训练 spaCy 组件

到本章结束时，你将能够开发与外部知识库集成的 NLP 模型，从而提高在现实世界场景中的准确性和适用性。

# 技术要求

本章的所有数据和代码都可以在[`github.com/PacktPublishing/Mastering-spaCy-Second-Edition`](https://github.com/PacktPublishing/Mastering-spaCy-Second-Edition)找到。

# 理解实体链接任务

实体链接是将提到的实体与其在各个知识库中的对应条目进行关联的任务。例如，**华盛顿**实体可以指代人物乔治·华盛顿或美国的一个州。通过实体链接或实体解析，我们的目标是把实体映射到正确的现实世界表示。正如 spaCy 的文档所说，spaCy 的**实体链接器**架构需要三个主要组件：

+   一个用于存储唯一标识符、同义词和先验概率的**知识库**（`KB`）

+   一个用于生成可能标识符的**候选生成步骤**

+   一个用于从候选列表中选择最可能 ID 的**机器学习模型**

在 KB 中，每个文本提及（别名）都表示为一个可能或可能未链接到实体的**候选**对象。每个候选（别名，实体）对都分配一个先验概率。

在 spaCy 的**实体链接器**架构中，首先，我们使用共享的语言**词汇表**和固定大小实体向量的长度初始化一个 KB。然后，我们需要设置模型。`spacy.EntityLinker.v2`类使用`spacy.HashEmbedCNN.v2`模型作为默认的模型架构，这是 spaCy 的标准`tok2vec`层。

`spacy.HashEmbedCNN.v2`架构由一个**多哈希嵌入**嵌入层和一个**最大输出窗口编码器**编码层定义。

`MultiHashEmbed` 嵌入层使用子词特征，并构建一个嵌入层，该层使用哈希嵌入分别嵌入词汇属性（`"NORM"`、`"PREFIX"`、`"SUFFIX"` 等）。哈希嵌入旨在解决嵌入大小的问题。在论文《用于高效词表示的哈希嵌入》([`arxiv.org/pdf/1709.03933`](https://arxiv.org/pdf/1709.03933) )中，Svenstrup、Hansen 和 Winther 指出，即使是中等大小的嵌入尺寸（300 维度），如果词汇量很大（如 Google Word2Vec 中的 300 万个单词和短语），总参数数接近 10 亿。使用哈希嵌入是内存高效的，因为它是一个紧凑的数据结构，所需的存储空间比词袋表示少。

`MaxoutWindowEncoder` 编码层使用卷积来编码上下文。该层接受的两个主要参数是 `window_size` 和 `depth`。`window_size` 参数设置围绕每个标记连接的单词数量，以构建卷积。`depth` 参数设置卷积层的数量。

`spacy.HashEmbedCNN.v2` 和所有其他 spaCy 层都是使用 Thinc API 定义的。让我们看看定义这些层的代码（仅用于学习目的，因为我们只需要在 `config.cfg` 文件中指向这个定义）：

```py
@registry.architectures("spacy.HashEmbedCNN.v2")
def build_hash_embed_cnn_tok2vec(
    *,
    width: int,
    depth: int,
    embed_size: int,
    window_size: int,
    maxout_pieces: int,
    subword_features: bool,
    pretrained_vectors: Optional[bool],
) -> Model[List[Doc], List[Floats2d]]:
    if subword_features:
        attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
        row_sizes = [embed_size, embed_size // 2, 
                     embed_size // 2, embed_size // 2]
    else:
        attrs = ["NORM"]
        row_sizes = [embed_size]
    return build_Tok2Vec_model(
        embed=MultiHashEmbed(
            width=width,
            rows=row_sizes,
            attrs=attrs,
            include_static_vectors=bool(pretrained_vectors),
        ),
        encode=MaxoutWindowEncoder(
            width=width,
            depth=depth,
            window_size=window_size,
            maxout_pieces=maxout_pieces,
        ),
    )
```

我们可以看到，该层正在使用 `MultiHashEmbed` 来嵌入文本，并使用 `MaxoutWindowEncoder` 层来编码嵌入，为模型提供最终的线性输出。接下来，让我们看看 `EntityLinker.v2` 架构本身的代码：

```py
@registry.architectures("spacy.EntityLinker.v2")
def build_nel_encoder(
    tok2vec: Model, nO: Optional[int] = None
) -> Model[List[Doc], Floats2d]:
    with Model.define_operators({">>": chain, "&": tuplify}):
        token_width = tok2vec.maybe_get_dim("nO")
        output_layer = Linear(nO=nO, nI=token_width)
        model = (
            ((tok2vec >> list2ragged()) & build_span_maker())
            >> extract_spans()
            >> reduce_mean()
            >> residual(Maxout(nO=token_width, nI=token_width, 
                               nP=2, dropout=0.0))  # type: ignore
            >> output_layer
        )
        model.set_ref("output_layer", output_layer)
        model.set_ref("tok2vec", tok2vec)
    # flag to show this isn't legacy
    model.attrs["include_span_maker"] = True
    return model
```

`build_nel_encoder()` 方法的参数是 `tok2vec` 模型和由 KB 中每个实体的编码向量的长度确定的输出维度 `nO`。当我们不设置 `nO` 时，它在调用 `initialize` 时会自动设置。我们将在 `config.cfg` 文件中定义 `tok2vec` 模型，如下所示：

```py
[components.entity_linker.model]
@architectures = "spacy.EntityLinker.v2"
nO = null
[components.entity_linker.model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v1"
pretrained_vectors = null
width = 96
depth = 2
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true
```

现在您已经了解了 spaCy 和 Thinc 在底层如何交互以创建 `EntityLinker` 模型架构。如果您需要更改参数或尝试不同的模型，您知道去哪里更改这些设置。

在商业环境中，我们通常从没有可用的数据集开始，迫使我们创建自己的数据集。拥有高质量的数据集是训练性能良好的模型的关键组成部分，因此学习创建良好数据集的基本知识非常重要。让我们在下一节中讨论这个问题。

# 创建良好 NLP 语料库的最佳实践

由于微调技术的存在，幸运的是，我们今天不需要大量的数据来训练模型。然而，好的数据集仍然非常重要，因为它们对于保证和评估我们的 NLP 系统的性能至关重要。例如，如果未经仔细策划，语言中深深嵌入的偏见和文化细微差别可能会无意中塑造 AI 的输出。在文章《如何在不经意间制造出有种族歧视倾向的 AI》中（[`blog.conceptnet.io/posts/2017/how-to-make-a-racist-ai-without-really-trying/`](https://blog.conceptnet.io/posts/2017/how-to-make-a-racist-ai-without-really-trying/)），研究人员罗宾·斯皮尔提供了一个关于情感分析的精彩教程，并展示了我们如何通过简单地重复使用在偏见数据上训练的嵌入来产生有种族歧视倾向的 AI 解决方案。

通常，NLP 任务的标注过程主要包括以下步骤：

1.  定义问题或任务。我们试图解决什么问题？这个问题将指导我们如何选择用于标注的样本，如何一致性地标注数据，等等。

1.  确定并准备一组代表性文本，作为语料库的起始材料。

1.  定义一个标注方案，并对语料库的一部分进行标注，以确定数据解决任务的可行性。

1.  标注语料库的大部分内容。

总是定义标注说明是一个好习惯，以确保数据集的标注一致性。这很重要，因为我们需要这种一致性，以便机器学习算法能够有效地泛化。安德鲁·吴在课程《生产中的机器学习》中讲述了标注过程中一致性的重要性（[`www.coursera.org/learn/introduction-to-machine-learning-in-production`](https://www.coursera.org/learn/introduction-to-machine-learning-in-production)），其中，一个计算机视觉模型没有达到预期的性能，经过一些错误分析后，他们发现是由于标注过程中的不一致性。任务是寻找钢板图像中的缺陷，一些标注将整个有缺陷的区域标注为缺陷，而另一些则单独标注缺陷。

为了构建高质量的 NLP 数据集，我们可以从数据质量维度借用一些原则。一个好的标注数据集的第一个特征是**一致性**。数据在格式、标注和分类方面应该具有一致的结构和统一性。

一个好的数据集也应该代表目标应用领域。如果我们正在构建一个从公司业务文档中提取信息的管道，那么使用法律文本来训练嵌入模型可能不是一个好主意。

一个好的数据集的第三个也是最后一个特征是它**有良好的文档记录**。我们应该明确和透明地说明我们如何收集和选择数据，标签说明是什么，以及谁标注了数据。这种文档记录非常重要，因为它允许透明度、可重复性和偏差管理。

在接下来的章节中，我们创建了一个小数据集，我们创建了一个包含来自新闻网站提及我们想要消歧的实体（Taylor Swift、Taylor Lautner 和 Taylor Fritz）的句子的数据集。标签任务是标记新闻句子中的`Taylor`提及到每个提及的人。不幸的是，在现实生活中，我们通常不会遇到这样的简单场景，这些是我们遵循一致性、代表性和良好文档原则来构建数据集变得更为重要的时刻。

现在我们知道了如何创建语料库，让我们回到 spaCy，学习如何训练我们的**实体链接器**组件。

# 使用 spaCy 训练实体链接器组件

训练模型的第一个步骤是创建知识库。我们想要创建一个管道，用于检测对*Taylor*的引用是指 Taylor Swift（歌手）、Taylor Lautner（演员）还是 Taylor Fritz（网球运动员）。他们每个人在 Wikidata 上都有自己的页面和标识符，因此我们将使用 Wikidata 作为我们的知识库来源。要创建知识库，我们需要创建一个`InMemoryLookupKB`类的实例，传递共享的`Vocab`对象和我们将要用来编码实体的嵌入向量的大小。让我们创建我们的知识库：

1.  首先，我们将选择**语言**对象（`en_core_web_md`）并添加一个`SpanRuler`组件来匹配所有的`taylor`提及（这将用于创建语料库）：

    ```py
    import spacy
    nlp = spacy.load("en_core_web_md")
    ruler = nlp.add_pipe("span_ruler", after="ner")
    patterns = [{"label": "PERSON", "pattern": [{"LOWER": "taylor"}]}]
    ruler.add_patterns(patterns)
    ```

1.  我们将把知识库和模型保存到磁盘上，所以让我们定义这些文件：

    ```py
    kb_loc = "chapter_10/nel_taylor/my_kb"
    nlp_dir = "chapter_10/nel_taylor/my_nlp"
    ```

1.  最后，我们可以开始创建知识库。首先，我们实例化 kb 对象，传递`Vocab`和向量的大小。`en_core_web_md`模型有 300 维的向量，所以我们为此设置实体向量的大小：

    ```py
    import os
    from spacy.kb import InMemoryLookupKB
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=300)
    ```

现在我们有了知识库，我们可以使用`add_entity()`方法添加实体，使用`add_alias()`方法添加提及（在我们的情况下，提及将是`Taylor`）以及每个实体的先验概率。让我们创建完成所有这些的代码：

1.  首先，我们创建两个字典，一个包含我们每个实体的 Wikidata ID，另一个包含它们的描述：

    ```py
    entities = {'Q26876': 'Taylor Swift', 'Q23359': 'Taylor Lautner', 'Q17660516': 'Taylor Fritz'}
    descriptions = {'Q26876': 'American singer-songwriter (born 1989)', 'Q23359': 'American actor', 'Q17660516': 'American tennis player'}
    ```

1.  现在，是时候将实体添加到知识库中。每个 Wikidata QID 将有一个实体描述的向量表示。要添加实体，我们使用`add_entity()`方法，目前我们可以为`freq`参数设置一个任意值（我们将告诉 spaCy 在训练模型的`config.cfg`文件中忽略这个频率）：

    ```py
    for qid, desc in descriptions.items():
        desc_doc = nlp(desc)
        desc_vector = desc_doc.vector
        kb.add_entity(entity=qid, entity_vector=desc_vector,
                      freq=111)
    ```

1.  现在，我们可以将提及（`alias`）添加到知识库中。如果文本中出现`Taylor Swift`实体，我们毫无疑问它指的是歌手泰勒·斯威夫特。同样，如果文本中出现`Taylor Lautner`实体，我们毫无疑问它指的是演员泰勒·洛特纳。我们将通过将这些实体的概率设置为`1`来添加此信息到知识库中：

    ```py
    for qid, name in entities.items():
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])
    ```

1.  当实体的名字和姓氏都存在时，我们毫无疑问知道是谁，但如果文本只提到`Taylor`呢？我们将为所有三个实体设置初始概率相等（每个实体 30%，因为概率之和不能超过 100%）：

    ```py
    qids = entities.keys()
    kb.add_alias(alias="Taylor", entities=qids, 
                 probabilities=[0.3, 0.3, 0.3])
    ```

1.  我们的 KB 还没有完全设置好。让我们打印实体和别名来检查是否一切正常：

    ```py
    print(f"Entities in the KB: {kb.get_entity_strings()}")
    >>> Entities in the KB: ['Q23359', 'Q17660516', 'Q26876']
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    >>> Aliases in the KB: ['Taylor Lautner', 'Taylor', 'Taylor Swift', 'Taylor Fritz']
    ```

1.  现在，是时候将知识库和`nlp`模型保存到磁盘上了，这样我们以后就可以使用它们：

    ```py
    kb.to_disk(kb_loc)
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)
    ```

在`InMemoryLookupKB`配置完成后，我们现在可以为 spaCy 准备训练数据。我们将处理的 CSV 文件包含名为`text`（句子）、`person`和`label`（实体的名称）、`ent_start`和`ent_end`（标记在句子中的位置）以及 Wikidata 的`QID`这些列。数据包含 49 个句子，我们将使用 80%进行训练（我们将将其分为`train`和`dev`集）和 20%进行测试。我们将分两步准备数据：首先，创建`Doc`对象，然后将其添加到`DocBin`的`train`和`dev`对象中。

为了创建`Doc`对象，我们将每个句子包装在一个`Doc`对象中，并使用 CSV 文件中的数据创建`Span`对象。这个`Span`对象有一个`kb_id`参数，我们将用它来设置实体的 Wikidata QID。让我们继续这样做：

1.  首先，我们加载 CSV 文件，并获取 80%的行用于训练，其余的用于测试：

    ```py
    import pandas as pd
    df_labeled = pd.read_csv("https://raw.githubusercontent.com/PacktPublishing/Mastering-spaCy-Second-Edition/main/chapter_10/taylor_labeled_dataset.csv")
    df_train = df_labeled.sample(frac=0.8, random_state=123)
    df_test = df_labeled.drop(df_train.index)
    ```

1.  现在，我们实例化我们之前用于创建知识库的管道，将句子包装在它里面以创建`Doc`对象，并为实体创建`Span`对象。我们将使用两个列表，一个用于存储`docs`，另一个用于存储`QIDs`：

    ```py
    import spacy
    from spacy.tokens import Span
    from collections import Counter
    nlp_dir = "chapter_10/nel_taylor/my_nlp"
    nlp = spacy.load(nlp_dir)
    docs = []
    QIDs = []
    for _,row in df_train.iterrows():
        sentence = row["text"]
        QID = row["QID"]
        span_start = row["ent_start"]
        span_end = row["ent_end"]
        doc = nlp(sentence)
        QIDs.append(QID)
        label_ent = "PERSON"
        ent_span = Span(doc, span_start, span_end, label_ent, 
                        kb_id=QID)
        doc.ents = [ent_span]
        docs.append(doc)
    ```

我们将使用`QIDs`列表将句子分割成`train`和`dev`的`DocBin`对象。为此，我们将获取每个 QID 的索引，使用前八个句子进行`train`，其余的留给`dev`。让我们在代码中这样做：

1.  首先，我们导入对象并创建我们的空`DocBin`对象：

    ```py
    from spacy.tokens import DocBin
    import math
    train_docs = DocBin()
    dev_docs = DocBin()
    ```

1.  现在，我们遍历每个实体 QID，获取它们句子的索引，并将它们添加到每个`DocBin`对象中：

    ```py
    entities = {'Q26876': 'Taylor Swift', 
                'Q23359': 'Taylor Lautner',
                'Q17660516': 'Taylor Fritz'}
    for QID in entities.keys():
        indexes_sentences_qid = [i for i, j in enumerate(QIDs) 
                                 if j == QID]
        for index in indexes_sentences_qid[0:8]:
            train_docs.add(docs[index])
        for index in indexes_sentences_qid[8:]:
            dev_docs.add(docs[index])
    ```

1.  现在我们可以将`DocBin`文件保存到磁盘上：

    ```py
    train_corpus = "chapter_10/nel_taylor/train.spacy"
    dev_corpus = "chapter_10/nel_taylor/dev.spacy"
    train_docs.to_disk(train_corpus)
    dev_docs.to_disk(dev_corpus)
    ```

当准备好`train`和`dev`集后，我们可以继续训练模型。我们将使用 spaCy 的 Nel Emerson 教程中使用的相同配置文件（[`github.com/explosion/projects/tree/v3/tutorials/nel_emerson`](https://github.com/explosion/projects/tree/v3/tutorials/nel_emerson)）。您可以在 GitHub 仓库中获取此文件：[`github.com/PacktPublishing/Mastering-spaCy-Second-Edition/tree/main/chapter_10`](https://github.com/PacktPublishing/Mastering-spaCy-Second-Edition/tree/main/chapter_10)。

如果您需要刷新对 spaCy 训练过程的了解，可以参考*第六章*。为了训练`EntityLinker`组件，我们需要做一些新的事情，即使用一个包含用于训练的额外代码的自定义文件。我们将这样做，因为我们需要以`EntityLinker`需要的方式创建`Example`对象。让我们在下一节中更多关于这一点进行讨论。

# 使用自定义语料库读取器进行训练

spaCy 的`Corpus`类管理用于训练期间数据加载的标注语料库。默认的语料库读取器（`spacy.Corpus.v1`）使用`Language`类的`make_doc()`方法创建`Example`对象。此方法仅对文本进行分词。为了训练`EntityLinker`组件，它需要在文档中具有可用的实体。这就是为什么我们将创建自己的语料库读取器，并将其保存为名为`custom_functions.py`的文件。读取器应接收`DocBin`文件的路径和`nlp`对象作为参数。在方法内部，我们将遍历每个`Doc`以创建示例。让我们继续创建这个方法：

1.  首先，我们禁用管道中的`EntityLinker`组件，然后从`DocBin`文件中获取所有文档：

    ```py
    def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
       with nlp.select_pipes(disable="entity_linker"):
          doc_bin = DocBin().from_disk(file)
          docs = doc_bin.get_docs(nlp.vocab)
    ```

1.  现在，我们将为每个文档创建`Example`对象。`Example`的第一个参数是经过`nlp`对象处理的文本，第二个参数是带有标注实体的`doc`：

    ```py
    # ...
          for doc in docs:
             yield Example(nlp(doc.text), doc)
    ```

1.  为了在`config.cfg`文件中引用此读取器，我们需要使用`@spacy.registry`装饰器进行注册。让我们导入我们将需要的库并注册读取器：

    ```py
    from functools import partial
    from pathlib import Path
    from typing import Iterable, Callable
    import spacy
    from spacy.training import Example
    from spacy.tokens import DocBin
    @spacy.registry.readers("MyCorpus.v1")
    def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
        return partial(read_files, file)
    ```

1.  我们使用`partial`函数，这样当 spaCy 代码在内部使用读取器时，我们只需要传递`nlp`对象。现在，我们可以在`config.cfg`文件中像这样引用这个`MyCorpus.v1`读取器：

    ```py
    #config.cfg snippet
    [corpora]
    [corpora.train]
    @readers = "MyCorpus.v1"
    file = ${paths.train}
    [corpora.dev]
    @readers = "MyCorpus.v1"
    file = ${paths.dev}
    ```

1.  这里是我们的`custom_functions.py`文件的完整源代码：

    ```py
    from functools import partial
    from pathlib import Path
    from typing import Iterable, Callable
    import spacy
    from spacy.training import Example
    from spacy.tokens import DocBin
    @spacy.registry.readers("MyCorpus.v1")
    def create_docbin_reader(file: Path) -> Callable[["Language"],

    Iterable[Example]]:
        return partial(read_files, file)
    def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
        # we run the full pipeline and not just nlp.make_doc to
        # ensure we have entities and sentences
        # which are needed during training of the entity linker
        with nlp.select_pipes(disable="entity_linker"):
            doc_bin = DocBin().from_disk(file)
            docs = doc_bin.get_docs(nlp.vocab)
            for doc in docs:
                yield Example(nlp(doc.text), doc)
    ```

1.  为了在调用`train` CLI 命令时使读取器可用，我们需要提供包含我们创建的 Python 代码的源代码。我们使用`code`参数来完成此操作。首先，我们获取`config.cfg`文件：

    ```py
    !curl https://raw.githubusercontent.com/PacktPublishing/Mastering-spaCy-Second-Edition/main/chapter_10/config.cfg -o config.cfg
    ```

1.  现在，我们可以运行完整的命令来训练我们的`EntityLinker`管道：

    ```py
    python -m spacy train ./config.cfg --output entity_linking_taylor --paths.train ./nel_taylor/train.spacy --paths.dev ./nel_taylor/dev.spacy --paths.kb nel_taylor/my_kb --paths.base_nlp ./nel_taylor/my_nlp --code custom_functions.py
    ```

现在我们有了训练好的模型，我们需要对其进行测试。让我们在下一节中这样做。

## 测试实体链接模型

为了测试模型，我们将使用 `train` 命令从我们保存它的路径加载它，并在我们想要消歧义实体的文档上调用 `nlp` 对象。实体链接器模型将 `kb_id` 添加到实体中，我们可以用它来查看模型预测了哪个 *Taylors*。让我们使用一些 `df_test` 句子来评估模型：

1.  首先，我们加载模型：

    ```py
    nlp = spacy.load("entity_linking_taylor/model-best")
    ```

1.  现在，我们处理句子。`doc.ents` 实体具有包含实体哈希值的 `kb_id` 属性，以及包含纯文本 QID 的 `kb_id_` 属性。让我们处理文本：

    ```py
    text = 'Taylor struggled with chilly temperatures in Edinburgh, pausing the show to warm up her hands and to assist a distressed fan.'
    doc = nlp(text)
    ```

1.  现在，我们可以使用 `displacy` 显示实体：

    ```py
    from spacy import displacy
    displacy.serve(doc, style="ent")
    ```

1.  *图 10.1* 显示了结果。`Q26876` 是泰勒·斯威夫特的 Wikidata ID，所以这次模型也是正确的。

![图 10.1 – 模型正确地消歧义了泰勒·斯威夫特的实体](img/B22441_10_01.jpg)

图 10.1 – 模型正确地消歧义了泰勒·斯威夫特的实体

让我们用另一个句子测试模型：

```py
text = 'Now, Taylor has revealed that he had to re-audition for the part because the producers wanted to go in a different direction.'
doc = nlp(text)
```

1.  *图 10.2* 显示了结果。`Q23359` 是泰勒·洛特纳的 Wikidata ID，所以模型在这点上也是正确的。

![图 10.2 – 模型正确地消歧义了泰勒·洛特纳的实体](img/B22441_10_02.jpg)

图 10.2 – 模型正确地消歧义了泰勒·洛特纳的实体

我们对模型采取轻松的态度，只评估这些简单的句子，因为这里的目的是仅展示如何从模型中获取结果。训练这个组件是一次相当漫长的旅程；恭喜！

# 摘要

在本章中，我们学习了如何使用 spaCy 训练 `EntityLinker` 组件。我们看到了一些实现细节，以了解更多关于 `HashEmbedCNN.v2` 层和 `EntityLinker.v2` 架构的信息。

我们还讨论了用于 NLP 训练的高质量数据集的一些特征，强调了一致性、代表性和详尽文档的重要性。最后，我们看到了如何创建用于训练实体链接模型的定制语料库读取器。有了这些知识，你可以定制任何其他 spaCy 组件。

在下一章和最后一章中，你将学习如何将 spaCy 与其他酷炫的开源库结合使用，以创建出色的 NLP 应用程序。那里见！
