

# 第二章：撒播语法

语法是语言的主要构建块之一。每种人类语言，以及编程语言，都有一个规则集，每个使用它的人都必须遵守，否则可能会不被理解。这些语法规则可以通过NLP揭示，并且对于从句子中提取数据很有用。例如，使用关于文本语法结构的信息，我们可以解析出主语、宾语以及不同实体之间的关系。

在本章中，你将学习如何使用不同的包来揭示单词和句子的语法结构，以及提取句子的某些部分。本章涵盖以下主题：

+   计数名词——复数和单数名词

+   获取依存句法分析

+   提取名词短语

+   提取句子的主语和宾语

+   使用语法信息在文本中寻找模式

# 技术要求

请按照[*第1章*](B18411_01.xhtml#_idTextAnchor013)中给出的安装要求运行本章中的笔记本。

# 计数名词——复数和单数名词

在本食谱中，我们将做两件事：确定一个名词是复数还是单数，并将复数名词转换为单数，反之亦然。

你可能需要这两样东西来完成各种任务。例如，你可能想要统计单词统计信息，为此，你很可能需要一起计算单数和复数名词。为了将复数名词与单数名词一起计数，你需要一种方法来识别一个单词是复数还是单数。

## 准备工作

为了确定一个名词是单数还是复数，我们将通过两种不同的方法使用`spaCy`：通过查看词元和实际单词之间的差异，以及通过查看`morph`属性。为了屈折这些名词，或将单数名词转换为复数或反之亦然，我们将使用`textblob`包。我们还将了解如何通过OpenAI API使用GPT-3确定名词的数量。本节代码位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/tree/main/Chapter02](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/tree/main/Chapter02)。

## 如何做到这一点...

我们将首先使用`spaCy`的词元信息来推断一个名词是单数还是复数。然后，我们将使用`Token`对象的`morph`属性。然后，我们将创建一个函数，使用这些方法之一。最后，我们将使用GPT-3.5来确定名词的数量：

1.  运行文件和语言实用工具笔记本中的代码。如果你遇到一个错误，说小或大模型不存在，你需要打开**lang_utils.ipynb**文件，取消注释，并运行下载模型的语句：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  初始化**text**变量，并使用**spaCy**小型模型进行处理，以获取结果**Doc**对象：

    ```py
    text = "I have five birds"
    doc = small_model(text)
    ```

1.  在这一步，我们遍历 **Doc** 对象。对于对象中的每个标记，我们检查它是否是名词，以及词元是否与单词本身相同。由于词元是单词的基本形式，如果词元与单词不同，则该标记是复数：

    ```py
    for token in doc:
        if (token.pos_ == "NOUN" and token.lemma_ != token.text):
            print(token.text, "plural")
    ```

    结果应该是这样的：

    ```py
    birds plural
    ```

1.  现在，我们将使用不同的方法来检查名词的数量：**Token** 对象的 **morph** 特征。**morph** 特征是单词的形态学特征，如数量、格等。由于我们知道标记 **3** 是一个名词，我们直接访问 **morph** 特征并获取 **Number** 以获得之前相同的结果：

    ```py
    doc = small_model("I have five birds.")
    print(doc[3].morph.get("Number"))
    ```

    下面是结果：

    ```py
    ['Plur']
    ```

1.  在这一步，我们准备定义一个返回元组 **(noun, number)** 的函数。为了更好地编码名词数量，我们使用一个 **Enum** 类，将不同的值分配给数字。我们将 **1** 分配给单数，将 **2** 分配给复数。一旦创建了这个类，我们就可以直接引用名词数量变量为 **Noun_number.SINGULAR** 和 **Noun_number.PLURAL**：

    ```py
    class Noun_number(Enum):
        SINGULAR = 1
        PLURAL = 2
    ```

1.  在这一步，我们定义了一个函数。该函数接受文本、**spaCy** 模型以及确定名词数量的方法作为输入。这两种方法是 **lemma** 和 **morph**，分别与我们之前在 *步骤 3* 和 *步骤 4* 中使用的相同两种方法。该函数输出一个元组列表，每个元组的格式为 **(名词文本, 名词数量**)，其中名词数量使用在 *步骤 5* 中定义的 **Noun_number** 类表示：

    ```py
    def get_nouns_number(text, model, method="lemma"):
        nouns = []
        doc = model(text)
        for token in doc:
            if (token.pos_ == "NOUN"):
                if method == "lemma":
                    if token.lemma_ != token.text:
                        nouns.append((token.text, 
                            Noun_number.PLURAL))
                    else:
                        nouns.append((token.text,
                            Noun_number.SINGULAR))
                elif method == "morph":
                    if token.morph.get("Number") == "Sing":
                        nouns.append((token.text,
                            Noun_number.PLURAL))
                    else:
                        nouns.append((token.text,
                            Noun_number.SINGULAR))
        return nouns
    ```

1.  我们可以使用前面的函数并查看它在不同的 **spaCy** 模型上的性能。在这一步，我们使用我们刚刚定义的函数和小的 **spaCy** 模型。使用两种方法，我们看到 **spaCy** 模型错误地获取了不规则名词 **geese** 的数量：

    ```py
    text = "Three geese crossed the road"
    nouns = get_nouns_number(text, small_model, "morph")
    print(nouns)
    nouns = get_nouns_number(text, small_model)
    print(nouns)
    ```

    结果应该是这样的：

    ```py
    [('geese', <Noun_number.SINGULAR: 1>), ('road', <Noun_number.SINGULAR: 1>)]
    [('geese', <Noun_number.SINGULAR: 1>), ('road', <Noun_number.SINGULAR: 1>)]
    ```

1.  现在，让我们使用大型模型做同样的事情。如果您尚未下载大型模型，请通过运行第一行来下载。否则，您可以将其注释掉。在这里，我们看到尽管 **morph** 方法仍然错误地将 **geese** 分配为单数，但 **lemma** 方法提供了正确的答案：

    ```py
    !python -m spacy download en_core_web_lg
    large_model = spacy.load("en_core_web_lg")
    nouns = get_nouns_number(text, large_model, "morph")
    print(nouns)
    nouns = get_nouns_number(text, large_model)
    print(nouns)
    ```

    结果应该是这样的：

    ```py
    [('geese', <Noun_number.SINGULAR: 1>), ('road', <Noun_number.SINGULAR: 1>)]
    [('geese', <Noun_number.PLURAL: 2>), ('road', <Noun_number.SINGULAR: 1>)]
    ```

1.  现在，让我们使用 GPT-3.5 来获取名词数量。在结果中，我们看到 GPT-3.5 给出了相同的结果，并且正确地识别了 **geese** 和 **road** 的数量：

    ```py
    from openai import OpenAI
    client = OpenAI(api_key=OPEN_AI_KEY)
    prompt="""Decide whether each noun in the following text is singular or plural.
    Return the list in the format of a python tuple: (word, number). Do not provide any additional explanations.
    Sentence: Three geese crossed the road."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": "You are a helpful 
                assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    print(response.choices[0].message.content)
    ```

    结果应该是这样的：

    ```py
    ('geese', 'plural')
    ('road', 'singular')
    ```

## 还有更多…

我们也可以将名词从复数变为单数，反之亦然。我们将使用 `textblob` 包来完成这项工作。该包应通过 Poetry 环境自动安装：

1.  从包中导入 **TextBlob** 类：

    ```py
    from textblob import TextBlob
    ```

1.  初始化一个文本变量列表，并通过列表推导式使用 **TextBlob** 类进行处理：

    ```py
    texts = ["book", "goose", "pen", "point", "deer"]
    blob_objs = [TextBlob(text) for text in texts]
    ```

1.  使用对象的 **pluralize** 函数来获取复数。此函数返回一个列表，我们访问其第一个元素。打印结果：

    ```py
    plurals = [blob_obj.words.pluralize()[0] 
        for blob_obj in blob_objs]
    print(plurals)
    ```

    结果应该是这样的：

    ```py
    ['books', 'geese', 'pens', 'points', 'deer']
    ```

1.  现在，我们将进行相反的操作。我们使用前面的 **复数** 列表将复数名词转换为 **TextBlob** 对象：

    ```py
    blob_objs = [TextBlob(text) for text in plurals]
    ```

1.  使用 **singularize** 函数将名词转换为单数并打印：

    ```py
    singulars = [blob_obj.words.singularize()[0] 
        for blob_obj in blob_objs]
    print(singulars)
    ```

    结果应该与我们在 *步骤 2* 中开始时的列表相同：

    ```py
    ['book', 'goose', 'pen', 'point', 'deer']
    ```

# 获取依存句法分析

依存句法分析是一种显示句子中依存关系的工具。例如，在句子 *The cat wore a hat* 中，句子的根是动词，*wore*，而主语，*the cat*，和宾语，*a hat*，都是依存词。依存句法分析在许多 NLP 任务中非常有用，因为它显示了句子的语法结构，包括主语、主要动词、宾语等。然后它可以用于下游处理。

`spaCy` NLP 引擎将其整体分析的一部分作为依存句法分析。依存句法分析标签解释了句子中每个词的作用。`ROOT` 是所有其他词都依赖的词，通常是动词。

## 准备中

我们将使用 `spaCy` 来创建依存句法分析。所需的包是 Poetry 环境的一部分。

## 如何做…

我们将从 `sherlock_holmes1.txt` 文件中选取几句话来展示依存句法分析。步骤如下：

1.  运行文件和语言实用工具笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  定义我们将要分析的句子：

    ```py
    sentence = 'I have seldom heard him mention her under any other name.'
    ```

1.  定义一个函数，该函数将打印出词、其嵌入在 **dep_** 属性中的语法功能以及该属性的说明。**Token** 对象的 **dep_** 属性显示了词在句子中的语法功能：

    ```py
    def print_dependencies(sentence, model):
        doc = model(sentence)
        for token in doc:
            print(token.text, "\t", token.dep_, "\t", 
                spacy.explain(token.dep_))
    ```

1.  现在，让我们将此函数应用于我们列表中的第一句话。我们可以看到动词 **heard** 是句子的 **ROOT** 词，所有其他词都依赖于它：

    ```py
    print_dependencies(sentence, small_model)
    ```

    结果应该如下所示：

    ```py
    I    nsubj    nominal subject
    have    aux    auxiliary
    seldom    advmod    adverbial modifier
    heard    ROOT    root
    him    nsubj    nominal subject
    mention    ccomp    clausal complement
    her    dobj    direct object
    under    prep    prepositional modifier
    any    det    determiner
    other    amod    adjectival modifier
    name    pobj    object of preposition
    .    punct    punctuation
    ```

1.  要探索依存句法分析结构，我们可以使用 **Token** 类的属性。使用 **ancestors** 和 **children** 属性，我们可以获取此标记所依赖的标记和依赖于它的标记，分别。打印祖先的函数如下：

    ```py
    def print_ancestors(sentence, model):
        doc = model(sentence)
        for token in doc:
            print(token.text, [t.text for t in token.ancestors])
    ```

1.  现在，让我们将此函数应用于我们列表中的第一句话：

    ```py
    print_ancestors(sentence, small_model)
    ```

    输出将如下所示。在结果中，我们看到 `heard` 没有祖先，因为它是在句子中的主要词。所有其他词都依赖于它，实际上，它们的祖先列表中都包含 `heard`。

    通过跟踪每个词的祖先链接，可以看到依存链。例如，如果我们查看单词 `name`，我们看到它的祖先是 `under`、`mention` 和 `heard`。`name` 的直接父词是 `under`，`under` 的父词是 `mention`，`mention` 的父词是 `heard`。依存链始终会引导到句子的根，或主要词：

    ```py
    I ['heard']
    have ['heard']
    seldom ['heard']
    heard []
    him ['mention', 'heard']
    mention ['heard']
    her ['mention', 'heard']
    under ['mention', 'heard']
    any ['name', 'under', 'mention', 'heard']
    other ['name', 'under', 'mention', 'heard']
    name ['under', 'mention', 'heard']
    . ['heard']
    ```

1.  要查看所有子词，请使用以下函数。此函数打印出每个词及其依赖于它的词，其 **children**：

    ```py
    def print_children(sentence, model):
        doc = model(sentence)
        for token in doc:
            print(token.text,[t.text for t in token.children])
    ```

1.  现在，让我们将此函数应用于我们列表中的第一句话：

    ```py
    print_children(sentence, small_model)
    ```

    结果应该是这样的。现在，单词 `heard` 有一个依赖它的单词列表，因为它在句子中是主词：

    ```py
    I []
    have []
    seldom []
    heard ['I', 'have', 'seldom', 'mention', '.']
    him []
    mention ['him', 'her', 'under']
    her []
    under ['name']
    any []
    other []
    name ['any', 'other']
    . []
    ```

1.  我们还可以在单独的列表中看到左右子节点。在以下函数中，我们将子节点打印为两个单独的列表，左和右。这在进行句子语法转换时可能很有用：

    ```py
    def print_lefts_and_rights(sentence, model):
        doc = model(sentence)
        for token in doc:
            print(token.text,
                [t.text for t in token.lefts],
                [t.text for t in token.rights])
    ```

1.  让我们使用这个函数处理我们列表中的第一句话：

    ```py
    print_lefts_and_rights(sentence, small_model)
    ```

    结果应该是这样的：

    ```py
    I [] []
    have [] []
    seldom [] []
    heard ['I', 'have', 'seldom'] ['mention', '.']
    him [] []
    mention ['him'] ['her', 'under']
    her [] []
    under [] ['name']
    any [] []
    other [] []
    name ['any', 'other'] []
    . [] []
    ```

1.  我们还可以通过使用此函数看到标记所在的子树：

    ```py
    def print_subtree(sentence, model):
        doc = model(sentence)
        for token in doc:
            print(token.text, [t.text for t in token.subtree])
    ```

1.  让我们使用这个函数处理我们列表中的第一句话：

    ```py
    print_subtree(sentence, small_model)
    ```

    结果应该是这样的。从每个单词所属的子树中，我们可以看到句子中出现的语法短语，如 `any other name` 和 `under any other name`：

    ```py
    I ['I']
    have ['have']
    seldom ['seldom']
    heard ['I', 'have', 'seldom', 'heard', 'him', 'mention', 'her', 'under', 'any', 'other', 'name', '.']
    him ['him']
    mention ['him', 'mention', 'her', 'under', 'any', 'other', 'name']
    her ['her']
    under ['under', 'any', 'other', 'name']
    any ['any']
    other ['other']
    name ['any', 'other', 'name']
    . ['.']
    ```

## 参见

可以使用 `displaCy` 包图形化地可视化依存句法，它是 `spaCy` 的一部分。请参阅 [*第7章*](B18411_08.xhtml#_idTextAnchor205) *可视化文本数据*，了解如何进行可视化的详细食谱。

# 提取名词短语

在语言学中，名词短语被称为名词短语。它们代表名词以及任何依赖和伴随名词的单词。例如，在句子 *The big red apple fell on the scared cat* 中，名词短语是 *the big red apple* 和 *the scared cat*。提取这些名词短语对于许多其他下游自然语言处理任务至关重要，例如命名实体识别以及处理实体及其关系。在本食谱中，我们将探讨如何从文本中提取命名实体。

## 准备工作

我们将使用 `spaCy` 包，它有一个用于提取名词短语的函数，以及 `sherlock_holmes_1.txt` 文件中的文本作为示例。

## 如何做到这一点...

使用以下步骤从文本中获取名词短语：

1.  运行文件和语言实用工具笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  定义一个函数，该函数将打印出名词短语。名词短语包含在 **doc.noun_chunks** 类变量中：

    ```py
    def print_noun_chunks(text, model):
        doc = model(text)
        for noun_chunk in doc.noun_chunks:
            print(noun_chunk.text)
    ```

1.  从 **sherlock_holmes_1.txt** 文件中读取文本并使用该函数处理结果文本：

    ```py
    sherlock_holmes_part_of_text = read_text_file("../data/sherlock_holmes_1.txt")
    print_noun_chunks(sherlock_holmes_part_of_text, small_model)
    ```

    这是部分结果。请参阅笔记本的输出[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter02/noun_chunks_2.3.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter02/noun_chunks_2.3.ipynb)，以获取完整打印输出。该函数正确地获取了文本中的代词、名词和名词短语：

    ```py
    Sherlock Holmes
    she
    the_ woman
    I
    him
    her
    any other name
    his eyes
    she
    the whole
    …
    ```

## 更多内容…

名词短语是 `spaCy` `Span` 对象，并具有所有属性。请参阅官方文档[https://spacy.io/api/token](https://spacy.io/api/token)。

让我们探索名词短语的一些属性：

1.  我们将定义一个函数，该函数将打印出名词短语的不同属性。它将打印出名词短语的文本，它在 **Doc** 对象中的起始和结束索引，它所属的句子（当有多个句子时很有用），名词短语的根（其主要单词），以及该短语与单词 **emotions** 的相似度。最后，它将打印出整个输入句子与 **emotions** 的相似度：

    ```py
    def explore_properties(sentence, model):
        doc = model(sentence)
        other_span = "emotions"
        other_doc = model(other_span)
        for noun_chunk in doc.noun_chunks:
            print(noun_chunk.text)
            print("Noun chunk start and end", "\t",
                noun_chunk.start, "\t", noun_chunk.end)
            print("Noun chunk sentence:", noun_chunk.sent)
            print("Noun chunk root:", noun_chunk.root.text)
            print(f"Noun chunk similarity to '{other_span}'",
                noun_chunk.similarity(other_doc))
        print(f"Similarity of the sentence '{sentence}' to 
            '{other_span}':",
            doc.similarity(other_doc))
    ```

1.  将句子设置为 **All emotions, and that one particularly, were abhorrent to his cold, precise but admirably** **balanced mind**：

    ```py
    sentence = "All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind."
    ```

1.  使用小模型上的 **explore_properties** 函数：

    ```py
    explore_properties(sentence, small_model)
    ```

    这是结果：

    ```py
    All emotions
    Noun chunk start and end    0    2
    Noun chunk sentence: All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.
    Noun chunk root: emotions
    Noun chunk similarity to 'emotions' 0.4026421588260174
    his cold, precise but admirably balanced mind
    Noun chunk start and end    11    19
    Noun chunk sentence: All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.
    Noun chunk root: mind
    Noun chunk similarity to 'emotions' -0.036891259527462
    Similarity of the sentence 'All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.' to 'emotions': 0.03174900767577446
    ```

    你还会看到类似这样的警告消息，因为小模型没有自己的词向量：

    ```py
    /tmp/ipykernel_1807/2430050149.py:10: UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Span.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.
      print(f"Noun chunk similarity to '{other_span}'", noun_chunk.similarity(other_doc))
    ```

1.  现在，让我们将相同的函数应用于使用大型模型的同一句子：

    ```py
    sentence = "All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind."
    explore_properties(sentence, large_model)
    ```

    大型模型确实包含自己的词向量，不会产生警告：

    ```py
    All emotions
    Noun chunk start and end    0    2
    Noun chunk sentence: All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.
    Noun chunk root: emotions
    Noun chunk similarity to 'emotions' 0.6302678068015664
    his cold, precise but admirably balanced mind
    Noun chunk start and end    11    19
    Noun chunk sentence: All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.
    Noun chunk root: mind
    Noun chunk similarity to 'emotions' 0.5744456705692561
    Similarity of the sentence 'All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.' to 'emotions': 0.640366414527618
    ```

    我们看到，与 `his cold, precise but admirably balanced mind` 这个名词短语相比，`All emotions` 这个名词短语与单词 `emotions` 的相似度较高。

重要提示

一个更大的 **spaCy** 模型，如 **en_core_web_lg**，占用更多空间但更精确。

## 参见

语义相似性的主题将在 [*第3章*](B18411_03.xhtml#_idTextAnchor067) 中更详细地探讨。

# 提取句子的主语和宾语

有时，我们可能需要找到句子的主语和直接宾语，而使用 `spaCy` 包件可以轻松完成。

## 准备工作

我们将使用 `spaCy` 的依赖标签来查找主语和宾语。代码使用 `spaCy` 引擎解析句子。然后，主语函数遍历标记，如果依赖标签包含 `subj`，则返回该标记的子树，一个 `Span` 对象。存在不同的主语标签，包括 `nsubj` 用于普通主语和 `nsubjpass` 用于被动句的主语，因此我们希望寻找两者。

## 如何做到这一点...

我们将使用标记的 `subtree` 属性来找到完整的名词短语，它是动词的主语或直接宾语（参见 *获取依赖分析* 菜谱）。我们将定义函数来查找主语、直接宾语、宾语从句和介词短语：

1.  运行文件和语言实用工具笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  我们将使用两个函数来找到句子的主语和直接宾语。这些函数将遍历标记并分别返回包含 **subj** 或 **dobj** 依赖标签的标记的子树。以下是主语函数。它寻找具有包含 **subj** 依赖标签的标记，然后返回包含该标记的子树。存在几个主语依赖标签，包括 **nsubj** 和 **nsubjpass**（被动句的主语），因此我们寻找最一般的模式：

    ```py
    def get_subject_phrase(doc):
        for token in doc:
            if ("subj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
    ```

1.  这里是直接宾语函数。它的工作方式与**get_subject_phrase**类似，但寻找的是**dobj**依赖标签，而不是包含**subj**的标签。如果句子没有直接宾语，它将返回**None**：

    ```py
    def get_object_phrase(doc):
        for token in doc:
            if ("dobj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
    ```

1.  将句子列表分配给一个变量，遍历它们，并使用前面的函数打印出它们的主题和宾语：

    ```py
    sentences = [
        "The big black cat stared at the small dog.",
        "Jane watched her brother in the evenings.",
        "Laura gave Sam a very interesting book."
    ]
    for sentence in sentences:
        doc = small_model(sentence)
        subject_phrase = get_subject_phrase(doc)
        object_phrase = get_object_phrase(doc)
        print(sentence)
        print("\tSubject:", subject_phrase)
        print("\tDirect object:", object_phrase)
    ```

    结果将如下所示。由于第一个句子没有直接宾语，将打印出`None`。对于句子`The big black cat stared at the small dog`，主语是`the big black cat`，没有直接宾语（`the small dog`是介词`at`的宾语）。对于句子`Jane watched her brother in the evenings`，主语是`Jane`，直接宾语是`her brother`。在句子`Laura gave Sam a very interesting book`中，主语是`Laura`，直接宾语是`a very interesting book`：

    ```py
    The big black cat stared at the small dog.
      Subject: The big black cat
      Direct object: None
    Jane watched her brother in the evenings.
      Subject: Jane
      Direct object: her brother
    Laura gave Sam a very interesting book.
      Subject: Laura
      Direct object: a very interesting book
    ```

## 还有更多…

我们可以寻找其他宾语，例如，动词如*give*的宾格宾语和介词短语宾语。这些函数看起来非常相似，主要区别在于依赖标签：宾格宾语函数的标签是`dative`，介词宾语函数的标签是`pobj`。介词宾语函数将返回一个列表，因为一个句子中可能有多个介词短语：

1.  宾格宾语函数检查标记的**宾格**标签。如果没有宾格宾语，则返回**None**：

    ```py
    def get_dative_phrase(doc):
        for token in doc:
            if ("dative" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
    ```

1.  我们还可以将主题、宾语和宾格函数组合成一个，通过一个参数指定要查找哪种宾语：

    ```py
    def get_phrase(doc, phrase):
        # phrase is one of "subj", "obj", "dative"
        for token in doc:
            if (phrase in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
    ```

1.  现在让我们定义一个带有宾格宾语的句子，并运行所有三种短语类型的函数：

    ```py
    sentence = "Laura gave Sam a very interesting book."
    doc = small_model(sentence)
    subject_phrase = get_phrase(doc, "subj")
    object_phrase = get_phrase(doc, "obj")
    dative_phrase = get_phrase(doc, "dative")
    print(sentence)
    print("\tSubject:", subject_phrase)
    print("\tDirect object:", object_phrase)
    print("\tDative object:", dative_phrase)
    ```

    结果将如下所示。宾格宾语是`Sam`：

    ```py
    Laura gave Sam a very interesting book.
      Subject: Laura
      Direct object: a very interesting book
      Dative object: Sam
    ```

1.  这里是介词宾语函数。它返回介词宾语的列表，如果没有，则列表为空：

    ```py
    def get_prepositional_phrase_objs(doc):
        prep_spans = []
        for token in doc:
            if ("pobj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                prep_spans.append(doc[start:end])
        return prep_spans
    ```

1.  让我们定义一个句子列表，并在它们上运行这两个函数：

    ```py
    sentences = [
        "The big black cat stared at the small dog.",
        "Jane watched her brother in the evenings."
    ]
    for sentence in sentences:
        doc = small_model(sentence)
        subject_phrase = get_phrase(doc, "subj")
        object_phrase = get_phrase(doc, "obj")
        dative_phrase = get_phrase(doc, "dative")
        prepositional_phrase_objs = \
            get_prepositional_phrase_objs(doc)
        print(sentence)
        print("\tSubject:", subject_phrase)
        print("\tDirect object:", object_phrase)
        print("\tPrepositional phrases:", prepositional_phrase_objs)
    ```

    结果将如下所示：

    ```py
    The big black cat stared at the small dog.
      Subject: The big black cat
      Direct object: the small dog
      Prepositional phrases: [the small dog]
    Jane watched her brother in the evenings.
      Subject: Jane
      Direct object: her brother
      Prepositional phrases: [the evenings]
    ```

    每个句子中都有一个介词短语。在句子`The big black cat stared at the small dog`中是`at the small dog`，在句子`Jane watched her brother in the evenings`中是`in the evenings`。

请将实际带有介词的介词短语而不是仅依赖于这些介词的名词短语找出来，这留作练习：

# 使用语法信息在文本中查找模式

在本节中，我们将使用`spaCy` `Matcher`对象在文本中查找模式。我们将使用单词的语法属性来创建这些模式。例如，我们可能正在寻找动词短语而不是名词短语。我们可以指定语法模式来匹配动词短语。

## 准备工作

我们将使用 `spaCy` 的 `Matcher` 对象来指定和查找模式。它可以匹配不同的属性，而不仅仅是语法。你可以在[https://spacy.io/usage/rule-based-matching/](https://spacy.io/usage/rule-based-matching/)的文档中了解更多信息。

## 如何操作...

你的步骤应该格式化如下：

1.  运行文件和语言实用工具笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  导入 **Matcher** 对象并初始化它。我们需要放入词汇对象，这与我们将用于处理文本的模型的词汇相同：

    ```py
    from spacy.matcher import Matcher
    matcher = Matcher(small_model.vocab)
    ```

1.  创建一个模式列表并将其添加到匹配器中。每个模式是一个字典列表，其中每个字典描述一个标记。在我们的模式中，我们只为每个标记指定词性。然后我们将这些模式添加到 **Matcher** 对象中。我们将使用的模式是一个单独的动词（例如，*paints*），一个助动词后面跟一个动词（例如，**was observing**），一个助动词后面跟一个形容词（例如，**were late**），以及一个助动词后面跟一个动词和一个介词（例如，**were staring at**）。这不是一个详尽的列表；请随意提出其他示例：

    ```py
    patterns = [
        [{"POS": "VERB"}],
        [{"POS": "AUX"}, {"POS": "VERB"}],
        [{"POS": "AUX"}, {"POS": "ADJ"}],
        [{"POS": "AUX"}, {"POS": "VERB"}, {"POS": "ADP"}]
    ]
    matcher.add("Verb", patterns)
    ```

1.  在小部分 *福尔摩斯* 文本中阅读并使用小模型进行处理：

    ```py
    sherlock_holmes_part_of_text = read_text_file("../data/sherlock_holmes_1.txt")
    doc = small_model(sherlock_holmes_part_of_text)
    ```

1.  现在，我们使用 **Matcher** 对象和已处理文本来查找匹配项。然后我们遍历匹配项，打印出匹配ID、字符串ID（模式的标识符）、匹配的开始和结束位置以及匹配文本：

    ```py
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = small_model.vocab.strings[match_id]
        span = doc[start:end]
        print(match_id, string_id, start, end, span.text)
    ```

    结果将如下所示：

    ```py
    14677086776663181681 Verb 14 15 heard
    14677086776663181681 Verb 17 18 mention
    14677086776663181681 Verb 28 29 eclipses
    14677086776663181681 Verb 31 32 predominates
    14677086776663181681 Verb 43 44 felt
    14677086776663181681 Verb 49 50 love
    14677086776663181681 Verb 63 65 were abhorrent
    14677086776663181681 Verb 80 81 take
    14677086776663181681 Verb 88 89 observing
    14677086776663181681 Verb 94 96 has seen
    14677086776663181681 Verb 95 96 seen
    14677086776663181681 Verb 103 105 have placed
    14677086776663181681 Verb 104 105 placed
    14677086776663181681 Verb 114 115 spoke
    14677086776663181681 Verb 120 121 save
    14677086776663181681 Verb 130 132 were admirable
    14677086776663181681 Verb 140 141 drawing
    14677086776663181681 Verb 153 154 trained
    14677086776663181681 Verb 157 158 admit
    14677086776663181681 Verb 167 168 adjusted
    14677086776663181681 Verb 171 172 introduce
    14677086776663181681 Verb 173 174 distracting
    14677086776663181681 Verb 178 179 throw
    14677086776663181681 Verb 228 229 was
    ```

代码在文本中找到了一些动词短语。有时，它找到一个部分匹配，它是另一个匹配的一部分。清除这些部分匹配被留作练习。

## 参考以下内容

我们可以使用除了词性以外的其他属性。可以基于文本本身、其长度、是否为字母数字、标点符号、单词的大小写、`dep_` 和 `morph` 属性、词元、实体类型等来匹配。还可以在模式上使用正则表达式。更多详细信息，请参阅 spaCy 文档：[https://spacy.io/usage/rule-based-matching](https://spacy.io/usage/rule-based-matching)。
