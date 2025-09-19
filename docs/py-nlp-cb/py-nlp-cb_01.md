

# 第一章：学习NLP基础知识

在编写这本书的过程中，我们专注于包括对各种NLP项目有用的食谱。它们从简单到复杂，从处理语法到处理可视化，在许多食谱中还包括了除英语之外的语言选项。在新版中，我们包括了使用GPT和其他大型语言模型、可解释人工智能、关于转换器的新章节以及自然语言理解的新主题。我们希望这本书对你有所帮助。

这本书的格式类似于**编程食谱**，其中每个食谱都是一个具有具体目标和需要执行的一系列步骤的短期迷你项目。理论解释很少，重点是实际目标和实现它们所需的工作。

在我们开始真正的NLP工作之前，我们需要为文本处理做好准备。本章将向你展示如何进行。到本章结束时，你将能够得到一个包含文本中单词及其词性、词干或词根的列表，并且移除了非常频繁的单词。

**自然语言工具包**（**NLTK**）和**spaCy**是我们在本章以及整本书中将要使用的重要库。书中还会使用到其他一些库，例如PyTorch和Hugging Face Transformers。我们还将利用OpenAI API和GPT模型。

本章包含的食谱如下：

+   将文本划分为句子

+   将句子划分为单词——分词

+   词性标注

+   结合相似词语——词形还原

+   移除停用词

# 技术要求

在整本书中，我们将使用**Poetry**来管理Python包的安装。你可以使用最新版本的Poetry，因为它保留了之前版本的功能。一旦安装了Poetry，管理要安装的包将会非常容易。整本书我们将使用**Python 3.9**。你还需要安装**Jupyter**以便运行笔记本。

注意

你可以尝试使用Google Colab来运行笔记本，但你需要调整代码以便使其在Colab上工作。

按照以下安装步骤进行：

1.  安装**Git**：[https://github.com/git-guides/install-git](https://github.com/git-guides/install-git)。

1.  安装**Poetry**：[https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)。

1.  安装**Jupyter**：[https://jupyter.org/install](https://jupyter.org/install)。

1.  在终端中输入以下命令以克隆包含本书所有代码的GitHub仓库（[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition)）：

```py
git clone https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition.git
```

1.  在包含**pyproject.toml**文件的目录中，使用终端运行以下命令：

```py
poetry install
poetry shell
```

1.  启动笔记本引擎：

```py
jupyter notebook
```

现在，你应该能够运行你克隆的仓库中的所有笔记本。

如果你不想使用Poetry，你可以使用书中提供的`requirements.txt`文件设置虚拟环境。你可以有两种方法来做这件事。你可以使用`pip`：

```py
pip install -r requirements.txt
```

你也可以使用`conda`：

```py
conda create --name <env_name> --file requirements.txt
```

# 将文本分割成句子

当我们处理文本时，我们可以处理不同尺度的文本单元：文档本身，例如一篇报纸文章，段落，句子或单词。句子是许多NLP任务中的主要处理单元。例如，当我们将数据发送到**大型语言模型**（**LLMs**）时，我们经常想在提示中添加一些上下文。在某些情况下，我们希望这个上下文中包含文本中的句子，以便模型可以从该文本中提取一些重要信息。在本节中，我们将向您展示如何将文本分割成句子。

## 准备工作

对于这部分，我们将使用《福尔摩斯探案集》的文本。你可以在这本书的GitHub文件中找到整个文本（[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes.txt)）。对于这个食谱，我们只需要书的开始部分，可以在文件[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt)中找到。

为了完成这个任务，你需要NLTK包及其句子分词器，它们是Poetry文件的一部分。安装Poetry的说明在*技术要求*部分中描述。

## 如何操作...

现在，我们将分割《福尔摩斯探案集》一小部分的文本，输出句子列表。（参考笔记本：[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter01/dividing_text_into_sentences_1.1.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter01/dividing_text_into_sentences_1.1.ipynb)）。在这里，我们假设你正在运行笔记本，所以路径都是相对于笔记本位置的：

1.  从**util**文件夹中导入文件实用函数（[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/file_utils.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/file_utils.ipynb)）：

    ```py
    %run -i "../util/file_utils.ipynb"
    ```

1.  读取书籍的部分文本：

    ```py
    sherlock_holmes_part_of_text = read_text_file("../data/sherlock_holmes_1.txt")
    ```

    `read_text_file`函数位于我们之前导入的`util`笔记本中。以下是它的源代码：

    ```py
    def read_text_file(filename):
        file = open(filename, "r", encoding="utf-8")
        return file.read()
    ```

1.  打印出结果以确保一切正常并且文件已加载：

    ```py
    print(sherlock_holmes_part_of_text)
    ```

    打印输出的开始部分将看起来像这样：

    ```py
    To Sherlock Holmes she is always _the_ woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and
    predominates the whole of her sex…
    ```

1.  导入**nltk**包：

    ```py
    import nltk
    ```

1.  如果你第一次运行代码，你需要下载分词器数据。之后你不需要再运行此命令：

    ```py
    nltk.download('punkt')
    ```

1.  初始化分词器：

    ```py
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    ```

1.  使用分词器将文本分割成句子。结果将是一个句子列表：

    ```py
    sentences_nltk = tokenizer.tokenize(
        sherlock_holmes_part_of_text)
    ```

1.  打印结果：

    ```py
    print(sentences_nltk)
    ```

    它应该看起来像这样。句子中包含来自书籍格式的换行符，它们不一定是句子的结尾：

    ```py
    ['To Sherlock Holmes she is always _the_ woman.', 'I have seldom heard him\nmention her under any other name.', 'In his eyes she eclipses and\npredominates the whole of her sex.', 'It was not that he felt any emotion\nakin to love for Irene Adler.', 'All emotions, and that one particularly,\nwere abhorrent to his cold, precise but admirably balanced mind.', 'He\nwas, I take it, the most perfect reasoning and observing machine that\nthe world has seen, but as a lover he would have placed himself in a\nfalse position.', 'He never spoke of the softer passions, save with a gibe\nand a sneer.', 'They were admirable things for the observer—excellent for\ndrawing the veil from men's motives and actions.', 'But for the trained\nreasoner to admit such intrusions into his own delicate and finely\nadjusted temperament was to introduce a distracting factor which might\nthrow a doubt upon all his mental results.', 'Grit in a sensitive\ninstrument, or a crack in one of his own high-power lenses, would not\nbe more disturbing than a strong emotion in a nature such as his.', 'And\nyet there was but one woman to him, and that woman was the late Irene\nAdler, of dubious and questionable memory.']
    ```

1.  打印结果中的句子数量；总共有11个句子：

    ```py
    print(len(sentences_nltk))
    ```

    这将给出以下结果：

    ```py
    11
    ```

虽然使用正则表达式在句号处分割文本以将其分为句子可能看起来很简单，但实际上要复杂得多。我们在句子的其他地方也使用句号；例如，在缩写词之后——例如，“Dr. Smith will see you now.” 类似地，虽然英语中的所有句子都以大写字母开头，但我们也会使用大写字母来表示专有名词。`nltk`中使用的这种方法考虑了所有这些因素；它是一个无监督算法的实现，该算法在[https://aclanthology.org/J06-4003.pdf](https://aclanthology.org/J06-4003.pdf)中提出。

## 还有更多...

我们还可以使用不同的策略来将文本解析为句子，采用另一个非常流行的NLP包，**spaCy**。以下是它是如何工作的：

1.  导入spaCy包：

    ```py
    import spacy
    ```

1.  第一次运行笔记本时，你需要下载spaCy模型。该模型是在大量英文文本上训练的，并且可以使用包括句子分词器在内的几个工具。在这里，我正在下载最小的模型，但你也可以尝试其他模型（见[https://spacy.io/usage/models/](https://spacy.io/usage/models/))。

    ```py
    !python -m spacy download en_core_web_sm
    ```

1.  初始化spaCy引擎：

    ```py
    nlp = spacy.load("en_core_web_sm")
    ```

1.  使用spaCy引擎处理文本。这一行假设你已经初始化了**sherlock_holmes_part_of_text**变量。如果没有，你需要运行之前的一个单元格，其中文本被读入这个变量：

    ```py
    doc = nlp(sherlock_holmes_part_of_text)
    ```

1.  从处理后的**doc**对象中获取句子，并打印出结果数组和它的长度：

    ```py
    sentences_spacy = [sentence.text for sentence in doc.sents]
    print(sentences_spacy)
    print(len(sentences_spacy))
    ```

    结果将看起来像这样：

    ```py
    ['To Sherlock Holmes she is always _the_ woman.', 'I have seldom heard him\nmention her under any other name.', 'In his eyes she eclipses and\npredominates the whole of her sex.', 'It was not that he felt any emotion\nakin to love for Irene Adler.', 'All emotions, and that one particularly,\nwere abhorrent to his cold, precise but admirably balanced mind.', 'He\nwas, I take it, the most perfect reasoning and observing machine that\nthe world has seen, but as a lover he would have placed himself in a\nfalse position.', 'He never spoke of the softer passions, save with a gibe\nand a sneer.', 'They were admirable things for the observer—excellent for\ndrawing the veil from men's motives and actions.', 'But for the trained\nreasoner to admit such intrusions into his own delicate and finely\nadjusted temperament was to introduce a distracting factor which might\nthrow a doubt upon all his mental results.', 'Grit in a sensitive\ninstrument, or a crack in one of his own high-power lenses, would not\nbe more disturbing than a strong emotion in a nature such as his.', 'And\nyet there was but one woman to him, and that woman was the late Irene\nAdler, of dubious and questionable memory.']
    11
    ```

spaCy与NLTK之间的重要区别在于完成句子分割过程所需的时间。原因在于spaCy加载了一个语言模型，并使用除了分词器之外的其他工具，而NLTK的分词器只有一个功能：将文本分割成句子。我们可以通过使用`time`包并将分割句子的代码放入`main`函数中来计时：

```py
import time
def split_into_sentences_nltk(text):
    sentences = tokenizer.tokenize(text)
    return sentences
def split_into_sentences_spacy(text):
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sents]
    return sentences
start = time.time()
split_into_sentences_nltk(sherlock_holmes_part_of_text)
print(f"NLTK: {time.time() - start} s")
start = time.time()
split_into_sentences_spacy(sherlock_holmes_part_of_text)
print(f"spaCy: {time.time() - start} s")
```

spaCy算法耗时0.019秒，而NLTK算法耗时0.0002秒。时间是通过从代码块开始设置的时间减去当前时间(`time.time()`)来计算的。你可能会得到略微不同的值。

您可能会使用spaCy的原因是如果您在使用该包进行其他处理的同时，还需要将其分割成句子。spaCy处理器执行许多其他任务，这就是为什么它需要更长的时间。如果您正在使用spaCy的其他功能，就没有必要仅为了句子分割而使用NLTK，最好在整个流程中使用spaCy。

还可以使用spaCy的tokenizer而不使用其他工具。请参阅他们的文档以获取更多信息：[https://spacy.io/usage/processing-pipelines](https://spacy.io/usage/processing-pipelines)。

重要提示

spaCy可能较慢，但它后台执行了许多更多的事情，如果您正在使用它的其他功能，那么在句子分割时也使用spaCy。

## 另请参阅

您可以使用NLTK和spaCy来分割非英语语言的文本。NLTK包括捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、法语、德语、希腊语、意大利语、挪威语、波兰语、葡萄牙语、斯洛文尼亚语、西班牙语、瑞典语和土耳其语的tokenizer模型。为了加载这些模型，请使用语言名称后跟`.pickle`扩展名：

```py
tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")
```

查看NLTK文档以获取更多信息：[https://www.nltk.org/index.html](https://www.nltk.org/index.html)。

同样，spaCy也提供了其他语言的模型：中文、荷兰语、英语、法语、德语、希腊语、意大利语、日语、葡萄牙语、罗马尼亚语、西班牙语以及其他语言。这些模型都是在这些语言的文本上训练的。为了使用这些模型，您需要分别下载它们。例如，对于西班牙语，可以使用以下命令下载模型：

```py
python -m spacy download es_core_news_sm
```

然后，将此行代码放入以使用它：

```py
nlp = spacy.load("es_core_news_sm")
```

查看spaCy文档以获取更多信息：[https://spacy.io/usage/models](https://spacy.io/usage/models)。

# 将句子分割成单词 – 分词

在许多情况下，我们在进行NLP任务时依赖于单个单词。例如，当我们通过依赖单个单词的语义来构建文本的语义模型时，或者当我们寻找具有特定词性的单词时，这种情况就会发生。为了将文本分割成单词，我们可以使用NLTK和spaCy来为我们完成这个任务。

## 准备工作

对于这部分，我们将使用书籍《福尔摩斯探案集》的相同文本。您可以在书籍的GitHub仓库中找到整个文本。对于这个食谱，我们只需要书的开始部分，这部分可以在`sherlock_holmes_1.txt`文件中找到。

为了完成这个任务，您将需要NLTK和spaCy包，它们是Poetry文件的一部分。在*技术* *要求*部分描述了安装Poetry的说明。

（笔记本参考：[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter01/dividing_sentences_into_words_1.2.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter01/dividing_sentences_into_words_1.2.ipynb)）。

## 如何实现

流程如下：

1.  导入**file_utils**笔记本。实际上，我们在一个笔记本中运行**file_utils**笔记本，这样我们就可以访问其定义的函数和变量：

    ```py
    %run -i "../util/file_utils.ipynb"
    ```

1.  读取书籍片段文本：

    ```py
    sherlock_holmes_part_of_text = read_text_file("../data/sherlock_holmes_1.txt")
    print(sherlock_holmes_part_of_text)
    ```

    结果应该看起来像这样：

    ```py
    To Sherlock Holmes she is always _the_ woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and
    predominates the whole of her sex... [Output truncated]
    ```

1.  导入**nltk**包：

    ```py
    import nltk
    ```

1.  将输入分成单词。在这里，我们使用NLTK单词分词器将文本分割成单个单词。该函数的输出是一个包含单词的Python列表：

    ```py
    words = nltk.tokenize.word_tokenize(
        sherlock_holmes_part_of_text)
    print(words)
    print(len(words))
    ```

1.  输出将是文本中的单词列表和**words**列表的长度：

    ```py
    ['To', 'Sherlock', 'Holmes', 'she', 'is', 'always', '_the_', 'woman', '.', 'I', 'have', 'seldom', 'heard', 'him', 'mention', 'her', 'under', 'any', 'other', 'name', '.', 'In', 'his', 'eyes', 'she', 'eclipses', 'and', 'predominates', 'the', 'whole', 'of', 'her', 'sex', '.', 'It', 'was', 'not', 'that', 'he', 'felt', 'any', 'emotion', 'akin', 'to', 'love', 'for', 'Irene', 'Adler', '.', 'All', 'emotions', ',', 'and', 'that', 'one', 'particularly', ',', 'were', 'abhorrent', 'to', 'his', 'cold', ',', 'precise', 'but', 'admirably', 'balanced', 'mind', '.', 'He', 'was', ',', 'I', 'take', 'it', ',', 'the', 'most', 'perfect', 'reasoning', 'and', 'observing', 'machine', 'that', 'the', 'world', 'has', 'seen', ',', 'but', 'as', 'a', 'lover', 'he', 'would', 'have', 'placed', 'himself', 'in', 'a', 'false', 'position', '.', 'He', 'never', 'spoke', 'of', 'the', 'softer', 'passions', ',', 'save', 'with', 'a', 'gibe', 'and', 'a', 'sneer', '.', 'They', 'were', 'admirable', 'things', 'for', 'the', 'observer—excellent', 'for', 'drawing', 'the', 'veil', 'from', 'men', ''', 's', 'motives', 'and', 'actions', '.', 'But', 'for', 'the', 'trained', 'reasoner', 'to', 'admit', 'such', 'intrusions', 'into', 'his', 'own', 'delicate', 'and', 'finely', 'adjusted', 'temperament', 'was', 'to', 'introduce', 'a', 'distracting', 'factor', 'which', 'might', 'throw', 'a', 'doubt', 'upon', 'all', 'his', 'mental', 'results', '.', 'Grit', 'in', 'a', 'sensitive', 'instrument', ',', 'or', 'a', 'crack', 'in', 'one', 'of', 'his', 'own', 'high-power', 'lenses', ',', 'would', 'not', 'be', 'more', 'disturbing', 'than', 'a', 'strong', 'emotion', 'in', 'a', 'nature', 'such', 'as', 'his', '.', 'And', 'yet', 'there', 'was', 'but', 'one', 'woman', 'to', 'him', ',', 'and', 'that', 'woman', 'was', 'the', 'late', 'Irene', 'Adler', ',', 'of', 'dubious', 'and', 'questionable', 'memory', '.']
    230
    ```

输出是一个列表，其中每个标记要么是单词，要么是标点符号。NLTK分词器使用一组规则将文本分割成单词。它分割但不扩展缩写，如*don’t → do n’t*和*men’s → men ’s*，正如前面的例子所示。它将标点和引号视为单独的标记，因此结果包括没有其他标记的单词。

## 还有更多…

有时候，不将某些单词分开，而将它们作为一个整体使用是有用的。这种做法的一个例子可以在[*第3章*](B18411_03.xhtml#_idTextAnchor067)中找到，在*表示短语 – phrase2vec*配方中，我们存储的是短语而不是单个单词。NLTK包允许我们使用其自定义分词器`MWETokenizer`来实现这一点：

1.  导入**MWETokenizer**类：

    ```py
    from nltk.tokenize import MWETokenizer
    ```

1.  初始化分词器并指示单词**dim sum dinner**不应被分割：

    ```py
    tokenizer = MWETokenizer([('dim', 'sum', 'dinner')])
    ```

1.  添加更多应该保留在一起的单词：

    ```py
    tokenizer.add_mwe(('best', 'dim', 'sum'))
    ```

1.  使用分词器分割一个句子：

    ```py
    tokens = tokenizer.tokenize('Last night I went for dinner in an Italian restaurant. The pasta was delicious.'.split())
    print(tokens)
    ```

    结果将包含与之前相同方式的分割标记：

    ```py
    ['Last', 'night', 'I', 'went', 'for', 'dinner', 'in', 'an', 'Italian', 'restaurant.', 'The', 'pasta', 'was', 'delicious.']
    ```

1.  分割不同的句子：

    ```py
    tokens = tokenizer.tokenize('I went out to a dim sum dinner last night. This restaurant has the best dim sum in town.'.split())
    print(tokens)
    ```

    在这种情况下，分词器会将短语组合成一个单元，并用下划线代替空格：

    ```py
    ['I', 'went', 'out', 'to', 'a', 'dim_sum_dinner', 'last', 'night.', 'This', 'restaurant', 'has', 'the_best_dim_sum', 'in', 'town.']
    ```

我们也可以使用spaCy进行分词。单词分词是spaCy在处理文本时完成的一系列任务中的一个任务。

## 还有更多

如果你正在对文本进行进一步处理，使用spaCy是有意义的。以下是它是如何工作的：

1.  导入**spacy**包：

    ```py
    import spacy
    ```

1.  仅在您之前没有执行此命令的情况下执行此命令：

    ```py
    !python -m spacy download en_core_web_sm
    ```

1.  使用英语模型初始化spaCy引擎：

    ```py
    nlp = spacy.load("en_core_web_sm")
    ```

1.  将文本分割成句子：

    ```py
    doc = nlp(sherlock_holmes_part_of_text)
    words = [token.text for token in doc]
    ```

1.  打印结果：

    ```py
    print(words)
    print(len(words))
    ```

    输出将如下所示：

    ```py
    ['To', 'Sherlock', 'Holmes', 'she', 'is', 'always', '_', 'the', '_', 'woman', '.', 'I', 'have', 'seldom', 'heard', 'him', '\n', 'mention', 'her', 'under', 'any', 'other', 'name', '.', 'In', 'his', 'eyes', 'she', 'eclipses', 'and', '\n', 'predominates', 'the', 'whole', 'of', 'her', 'sex', '.', 'It', 'was', 'not', 'that', 'he', 'felt', 'any', 'emotion', '\n', 'akin', 'to', 'love', 'for', 'Irene', 'Adler', '.', 'All', 'emotions', ',', 'and', 'that', 'one', 'particularly', ',', '\n', 'were', 'abhorrent', 'to', 'his', 'cold', ',', 'precise', 'but', 'admirably', 'balanced', 'mind', '.', 'He', '\n', 'was', ',', 'I', 'take', 'it', ',', 'the', 'most', 'perfect', 'reasoning', 'and', 'observing', 'machine', 'that', '\n', 'the', 'world', 'has', 'seen', ',', 'but', 'as', 'a', 'lover', 'he', 'would', 'have', 'placed', 'himself', 'in', 'a', '\n', 'false', 'position', '.', 'He', 'never', 'spoke', 'of', 'the', 'softer', 'passions', ',', 'save', 'with', 'a', 'gibe', '\n', 'and', 'a', 'sneer', '.', 'They', 'were', 'admirable', 'things', 'for', 'the', 'observer', '—', 'excellent', 'for', '\n', 'drawing', 'the', 'veil', 'from', 'men', ''s', 'motives', 'and', 'actions', '.', 'But', 'for', 'the', 'trained', '\n', 'reasoner', 'to', 'admit', 'such', 'intrusions', 'into', 'his', 'own', 'delicate', 'and', 'finely', '\n', 'adjusted', 'temperament', 'was', 'to', 'introduce', 'a', 'distracting', 'factor', 'which', 'might', '\n', 'throw', 'a', 'doubt', 'upon', 'all', 'his', 'mental', 'results', '.', 'Grit', 'in', 'a', 'sensitive', '\n', 'instrument', ',', 'or', 'a', 'crack', 'in', 'one', 'of', 'his', 'own', 'high', '-', 'power', 'lenses', ',', 'would', 'not', '\n', 'be', 'more', 'disturbing', 'than', 'a', 'strong', 'emotion', 'in', 'a', 'nature', 'such', 'as', 'his', '.', 'And', '\n', 'yet', 'there', 'was', 'but', 'one', 'woman', 'to', 'him', ',', 'and', 'that', 'woman', 'was', 'the', 'late', 'Irene', '\n', 'Adler', ',', 'of', 'dubious', 'and', 'questionable', 'memory', '.']
    251
    ```

你会注意到，当使用spaCy时，单词列表的长度比NLTK长。其中一个原因是spaCy保留了换行符，每个换行符都是一个单独的标记。另一个区别是spaCy会分割带有连字符的单词，如*high-power*。您可以通过运行以下行来找到两个列表之间的确切差异：

```py
print(set(words_spacy)-set(words_nltk))
```

这应该产生以下输出：

```py
{'high', 'power', 'observer', '-', '_', '—', 'excellent', ''s', '\n'}
```

重要提示

如果你正在使用spaCy进行其他处理，使用它是有意义的。否则，NLTK单词分词就足够了。

## 参见

NLTK包只为英语提供单词分词。

spaCy有其他语言的模型：中文、荷兰语、英语、法语、德语、希腊语、意大利语、日语、葡萄牙语、罗马尼亚语、西班牙语和其他语言。为了使用这些模型，你需要单独下载它们。例如，对于西班牙语，使用以下命令下载模型：

```py
python -m spacy download es_core_news_sm
```

然后，在代码中添加这一行来使用它：

```py
nlp = spacy.load("es_core_news_sm")
```

查看spaCy文档以获取更多信息：[https://spacy.io/usage/models](https://spacy.io/usage/models)。

# 词性标注

在许多情况下，NLP处理取决于确定文本中单词的词性。例如，当我们想要找出文本中出现的命名实体时，我们需要知道单词的词性。在这个食谱中，我们再次考虑NLTK和spaCy算法。

## 准备工作

对于这部分，我们将使用书籍《福尔摩斯探案集》的相同文本。你可以在这本书的GitHub仓库中找到整个文本。对于这个食谱，我们只需要书的开始部分，这部分可以在文件[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt)中找到。

为了完成这个任务，你需要NLTK和spaCy包，这些包在*技术要求*部分有所描述。

我们还将使用OpenAI API的GPT模型来完成这个任务，以证明它也能像spaCy和NLTK一样完成。为了运行这部分，你需要`openai`包，该包包含在Poetry环境中。你还需要自己的OpenAI API密钥。

## 如何操作…

在这个食谱中，我们将使用spaCy包来标注单词的词性。

流程如下：

1.  导入**util**文件和语言**util**文件。语言**util**文件包含spaCy和NLTK的导入，以及将小的spaCy模型初始化到**small_model**对象中。这些文件还包括从文件中读取文本和使用spaCy和NLTK的标记化函数：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  我们将定义一个函数，该函数将为每个单词输出词性。在这个函数中，我们首先使用spaCy模型处理输入文本，这会产生一个**Document**对象。产生的**Document**对象包含一个带有**Token**对象的迭代器，每个**Token**对象都包含有关词性的信息。

    我们使用这些信息来创建两个列表，一个包含单词，另一个包含它们各自的词性。

    最后，我们将两个列表进行压缩，将单词与词性配对，并返回结果列表的元组。我们这样做是为了能够轻松地打印出带有相应词性的整个列表。当你想在代码中使用词性标注时，你只需遍历标记列表：

    ```py
    def pos_tag_spacy(text, model):
        doc = model(text)
        words = [token.text for token in doc]
        pos = [token.pos_ for token in doc]
        return list(zip(words, pos))
    ```

1.  读取文本：

    ```py
    text = read_text_file("../data/sherlock_holmes_1.txt")
    ```

1.  使用文本和模型作为输入运行前面的函数：

    ```py
    words_with_pos = pos_tag_spacy(text, small_model)
    ```

1.  打印输出：

    ```py
    print(words_with_pos)
    ```

    以下部分结果显示如下；要查看完整的输出，请参阅 Jupyter 笔记本 ([https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter01/part_of_speech_tagging_1.3.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter01/part_of_speech_tagging_1.3.ipynb))：

    ```py
    [('To', 'ADP'),
     ('Sherlock', 'PROPN'),
     ('Holmes', 'PROPN'),
     ('she', 'PRON'),
     ('is', 'AUX'),
     ('always', 'ADV'),
     ('_', 'PUNCT'),
     ('the', 'DET'),
     ('_', 'PROPN'),
     ('woman', 'NOUN'),
     ('.', 'PUNCT'),
     ('I', 'PRON'),
     ('have', 'AUX'),
     ('seldom', 'ADV'),
     ('heard', 'VERB'),
     ('him', 'PRON'),
     ('\n', 'SPACE'),
     ('mention', 'VERB'),
     ('her', 'PRON'),
     ('under', 'ADP'),
     ('any', 'DET'),
     ('other', 'ADJ'),
     ('name', 'NOUN'),
     ('.', 'PUNCT'),…
    ```

结果列表包含单词和词性的元组。词性标签列表可在以下位置找到：[https://universaldependencies.org/u/pos/](https://universaldependencies.org/u/pos/)。

## 还有更多

我们可以将 spaCy 的性能与 NLTK 在此任务中的性能进行比较。以下是使用 NLTK 获取词性的步骤：

1.  我们导入的语言 **util** 文件中已经处理了导入，所以我们首先创建一个函数，该函数输出输入单词的词性。在其中，我们利用也导入自语言 **util** 笔记本的 **word_tokenize_nltk** 函数：

    ```py
    def pos_tag_nltk(text):
        words = word_tokenize_nltk(text)
        words_with_pos = nltk.pos_tag(words)
        return words_with_pos
    ```

1.  接下来，我们将该函数应用于之前读取的文本：

    ```py
    words_with_pos = pos_tag_nltk(text)
    ```

1.  打印出结果：

    ```py
    print(words_with_pos)
    ```

    以下部分输出如下。要查看完整的输出，请参阅 Jupyter 笔记本：

    ```py
    [('To', 'TO'),
     ('Sherlock', 'NNP'),
     ('Holmes', 'NNP'),
     ('she', 'PRP'),
     ('is', 'VBZ'),
     ('always', 'RB'),
     ('_the_', 'JJ'),
     ('woman', 'NN'),
     ('.', '.'),
     ('I', 'PRP'),
     ('have', 'VBP'),
     ('seldom', 'VBN'),
     ('heard', 'RB'),
     ('him', 'PRP'),
     ('mention', 'VB'),
     ('her', 'PRP'),
     ('under', 'IN'),
     ('any', 'DT'),
     ('other', 'JJ'),
     ('name', 'NN'),
     ('.', '.'),…
    ```

NLTK 使用的词性标签列表与 SpaCy 使用的不同，可以通过运行以下命令访问：

```py
python
>>> import nltk
>>> nltk.download('tagsets')
>>> nltk.help.upenn_tagset()
```

比较性能，我们发现 spaCy 需要 0.02 秒，而 NLTK 需要 0.01 秒（你的数字可能不同），因此它们的性能相似，NLTK 略好。然而，词性信息在初始处理完成后已经存在于 spaCy 对象中，所以如果你要进行任何进一步的处理，spaCy 是更好的选择。

重要提示

spaCy 会一次性完成所有处理，并将结果存储在 **Doc** 对象中。通过迭代 **Token** 对象可以获得词性信息。

## 还有更多

我们可以使用 GPT-3.5 和 GPT-4 模型通过 OpenAI API 执行各种任务，包括许多 NLP 任务。在这里，我们展示了如何使用 OpenAI API 获取输入文本的 NLTK 风格的词性。你还可以在提示中指定输出格式和词性标签的风格。为了使此代码正确运行，你需要自己的 OpenAI API 密钥：

1.  导入 **openai** 并使用您的 API 密钥创建 OpenAI 客户端。**OPEN_AI_KEY** 常量变量在 **../****util/file_utils.ipynb** 文件中设置：

    ```py
    from openai import OpenAI
    client = OpenAI(api_key=OPEN_AI_KEY)
    ```

1.  设置提示：

    ```py
    prompt="""Decide what the part of speech tags are for a sentence.
    Preserve original capitalization.
    Return the list in the format of a python tuple: (word, part of speech).
    Sentence: In his eyes she eclipses and predominates the whole of her sex."""
    ```

1.  向 OpenAI API 发送请求。我们发送到 API 的一些重要参数是我们想要使用的模型、温度，这会影响模型响应的变化程度，以及模型应返回的最大令牌数作为补全：

    ```py
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", 
             "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    ```

1.  打印响应：

    ```py
    print(response)
    ```

    输出将如下所示：

    ```py
    ChatCompletion(id='chatcmpl-9hCq34UAzMiNiqNGopt2U8ZmZM5po', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Here are the part of speech tags for the sentence "In his eyes she eclipses and predominates the whole of her sex" in the format of a Python tuple:\n\n[(\'In\', \'IN\'), (\'his\', \'PRP$\'), (\'eyes\', \'NNS\'), (\'she\', \'PRP\'), (\'eclipses\', \'VBZ\'), (\'and\', \'CC\'), (\'predominates\', \'VBZ\'), (\'the\', \'DT\'), (\'whole\', \'JJ\'), (\'of\', \'IN\'), (\'her\', \'PRP$\'), (\'sex\', \'NN\')]', role='assistant', function_call=None, tool_calls=None))], created=1720084483, model='gpt-3.5-turbo-0125', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=120, prompt_tokens=74, total_tokens=194))
    ```

1.  要仅查看 GPT 输出，请执行以下操作：

    ```py
    print(response.choices[0].message.content)
    ```

    输出将如下所示：

    ```py
    Here are the part of speech tags for the sentence "In his eyes she eclipses and predominates the whole of her sex" in the format of a Python tuple:
    [('In', 'IN'), ('his', 'PRP$'), ('eyes', 'NNS'), ('she', 'PRP'), ('eclipses', 'VBZ'), ('and', 'CC'), ('predominates', 'VBZ'), ('the', 'DT'), ('whole', 'JJ'), ('of', 'IN'), ('her', 'PRP$'), ('sex', 'NN')]
    ```

1.  我们可以使用**literal_eval**函数将响应转换为元组。我们要求GPT模型只返回答案，而不附加任何解释，这样答案中就没有自由文本，我们可以自动处理它。我们这样做是为了能够比较OpenAI API的输出与NLTK的输出：

    ```py
    from ast import literal_eval
    def pos_tag_gpt(text, client):
        prompt = f"""Decide what the part of speech tags are for a sentence.
        Preserve original capitalization.
        Return the list in the format of a python tuple: (word, part of speech).
        Do not include any other explanations.
        Sentence: {text}."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", 
                 "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        result = response.choices[0].message.content
        result = result.replace("\n", "")
        result = list(literal_eval(result))
        return result
    ```

1.  现在，让我们计时GPT函数，以便我们可以将其性能与其他先前使用的方法进行比较：

    ```py
    start = time.time()
    first_sentence = "In his eyes she eclipses and predominates the whole of her sex."
    words_with_pos = pos_tag_gpt(first_sentence, OPEN_AI_KEY)
    print(words_with_pos)
    print(f"GPT: {time.time() - start} s")
    ```

    结果看起来像这样：

    ```py
    [('In', 'IN'), ('his', 'PRP$'), ('eyes', 'NNS'), ('she', 'PRP'), ('eclipses', 'VBZ'), ('and', 'CC'), ('predominates', 'VBZ'), ('the', 'DT'), ('whole', 'NN'), ('of', 'IN'), ('her', 'PRP$'), ('sex', 'NN'), ('.', '.')]
    GPT: 2.4942469596862793 s
    ```

1.  GPT的输出与NLTK非常相似，但略有不同：

    ```py
    words_with_pos_nltk = pos_tag_nltk(first_sentence)
    print(words_with_pos == words_with_pos_nltk)
    ```

    这会输出以下内容：

    ```py
    False
    ```

GPT与NLTK的区别在于，GPT将整个单词标记为形容词，而NLTK将其标记为名词。在这种情况下，NLTK是正确的。

我们看到LLM的输出非常相似，但比NLTK慢约400倍。

## 参见

如果你想用另一种语言标记文本，你可以使用spaCy的其他语言模型。例如，我们可以加载西班牙语spaCy模型来处理西班牙语文本：

```py
nlp = spacy.load("es_core_news_sm")
```

如果spaCy没有你正在使用的语言的模型，你可以使用spaCy训练自己的模型。请参阅[https://spacy.io/usage/training#tagger-parser](https://spacy.io/usage/training#tagger-parser)。

# 结合相似单词 - 词元化

我们可以使用**词元化**来找到单词的规范形式。例如，单词*cats*的词元是*cat*，而单词*ran*的词元是*run*。当我们试图匹配某些单词而不想列出所有可能形式时，这很有用。相反，我们只需使用其词元。

## 准备工作

我们将使用spaCy包来完成这个任务。

## 如何做到这一点...

当spaCy处理一段文本时，生成的`Document`对象包含一个迭代器，遍历其中的`Token`对象，正如我们在*词性标注*食谱中看到的。这些`Token`对象包含文本中每个单词的词元信息。

获取词元的过程如下：

1.  导入文件和语言**工具**文件。这将导入spaCy并初始化**小型模型**对象：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  创建一个我们想要词元化的单词列表：

    ```py
    words = ["leaf", "leaves", "booking", "writing", "completed", "stemming"]
    ```

1.  为每个单词创建一个**文档**对象：

    ```py
    docs = [small_model(word) for word in words]
    ```

1.  打印列表中每个单词及其词元：

    ```py
    for doc in docs:
        for token in doc:
            print(token, token.lemma_)
    ```

    结果将如下所示：

    ```py
    leaf leaf
    leaves leave
    booking book
    writing write
    completed complete
    stemming stem
    ```

    结果显示所有单词的正确词元化。然而，有些单词是模糊的。例如，单词*leaves*可以是动词，在这种情况下词元是正确的，或者它是名词，在这种情况下这个词元是错误的。如果我们给spaCy连续文本而不是单个单词，它很可能会正确地消除歧义。

1.  现在，将词元化应用于更长的文本。在这里，我们读取一小部分*福尔摩斯探案集*文本，并对其每个单词进行词元化：

    ```py
    Text = read_text_file(../data/sherlock_holmes_1.txt")
    doc = small_model(text)
    for token in doc:
        print(token, token.lemma_)
    ```

    部分结果将如下所示：

    ```py
    To to
    Sherlock Sherlock
    Holmes Holmes
    she she
    is be
    always always
    _ _
    the the
    _ _
    woman woman
    . ….
    ```

## 更多内容...

我们可以使用spaCy词形还原对象来找出一个单词是否在其基本形式中。我们可能在操纵句子语法时这样做，例如，在将被动句转换为主动句的任务中。我们可以通过操纵spaCy管道来获取词形还原对象，该管道包括应用于文本的各种工具。有关更多信息，请参阅[https://spacy.io/usage/processing-pipelines/](https://spacy.io/usage/processing-pipelines/)。以下是步骤：

1.  管道组件位于一个元组列表中，**（组件名称，组件）**。为了获取词形还原组件，我们需要遍历这个列表：

    ```py
    lemmatizer = None
    for name, proc in small_model.pipeline:
        if name == "lemmatizer":
            lemmatizer = proc
    ```

1.  现在，我们可以将**is_base_form**函数调用应用于《福尔摩斯探案集》中的每个单词：

    ```py
    for token in doc:
        print(f"{token} is in its base form: 
            {lemmatizer.is_base_form(token)}")
    ```

    部分结果如下：

    ```py
    To is in its base form: False
    Sherlock is in its base form: False
    Holmes is in its base form: False
    she is in its base form: False
    is is in its base form: False
    always is in its base form: False
    _ is in its base form: False
    the is in its base form: False
    _ is in its base form: False
    woman is in its base form: True
    . is in its base form: False…
    ```

# 移除停用词

当我们处理单词时，尤其是如果我们正在考虑单词的语义时，我们有时需要排除一些在句子中不带来任何实质性意义的非常频繁的单词（例如*但是*、*可以*、*我们*等）。例如，如果我们想对文本的主题有一个大致的了解，我们可以计算其最频繁的单词。然而，在任何文本中，最频繁的单词将是停用词，因此我们希望在处理之前移除它们。这个菜谱展示了如何做到这一点。我们在这个菜谱中使用的停用词列表来自NLTK包，可能不包括你需要的所有单词。你需要相应地修改列表。

准备工作

我们将使用spaCy和NLTK来移除停用词；这些包是我们之前安装的Poetry环境的一部分。

我们将使用之前提到的*福尔摩斯探案集*文本。对于这个菜谱，我们只需要书的开始部分，可以在文件[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt)中找到。

在**步骤 1**中，我们运行实用工具笔记本。在**步骤 2**中，我们导入`nltk`包及其停用词列表。在**步骤 3**中，如果需要，我们下载停用词数据。在**步骤 4**中，我们打印出停用词列表。在**步骤 5**中，我们读取《福尔摩斯探案集》的一小部分。在**步骤 6**中，我们对文本进行分词并打印其长度，为230。在**步骤 7**中，我们通过列表推导式从原始单词列表中移除停用词。然后，我们打印结果的长度，看到列表长度已减少到105。你会注意到在列表推导式中，我们检查单词的小写形式是否在停用词列表中，因为所有停用词都是小写的。

## 如何做到这一点…

在这个菜谱中，我们将读取文本文件，对文本进行分词，并从列表中移除停用词：

1.  运行文件和语言实用工具笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  导入NLTK停用词列表：

    ```py
    from nltk.corpus import stopwords
    ```

1.  第一次运行笔记本时，下载**停用词**数据。下次运行代码时，无需再次下载停用词：

    ```py
    nltk.download('stopwords')
    ```

注意

这里是一个NLTK支持的停用词语言列表：阿拉伯语、阿塞拜疆语、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、匈牙利语、意大利语、哈萨克语、尼泊尔语、挪威语、葡萄牙语、罗马尼亚语、俄语、西班牙语、瑞典语和土耳其语。

1.  你可以通过打印列表来查看NLTK附带的所有停用词：

    ```py
    print(stopwords.words('english'))
    ```

    结果将如下所示：

    ```py
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    ```

1.  读取文本文件：

    ```py
    text = read_text_file("../data/sherlock_holmes_1.txt")
    ```

1.  将文本分词并打印结果列表的长度：

    ```py
    words = word_tokenize_nltk(text)
    print(len(words))
    ```

    结果将如下所示：

    ```py
    230
    ```

1.  使用列表推导从列表中移除停用词并打印结果长度。你将注意到在列表推导中，我们检查单词的小写版本是否在停用词列表中，因为所有停用词都是小写的。

    ```py
    words = [word for word in words if word not in stopwords.words("english")]
    print(len(words))
    ```

    结果将如下所示：

    ```py
    105
    ```

代码随后会从文本中过滤掉停用词，并且只有当这些词不在停用词列表中时，才会保留文本中的单词。从两个列表的长度来看，一个未过滤，另一个没有停用词，我们移除了超过一半的单词。

重要提示

你可能会发现提供的停用词列表中的某些单词是不必要的或缺失的。你需要相应地修改列表。NLTK的停用词列表是一个Python列表，你可以使用标准的Python列表函数添加和删除元素。

## 还有更多…

我们还可以使用spaCy移除停用词。以下是这样做的方法：

1.  为了方便，将停用词分配给一个变量：

    ```py
    stopwords = small_model.Defaults.stop_words
    ```

1.  将文本分词并打印其长度：

    ```py
    words = word_tokenize_nltk(text)
    print(len(words))
    ```

    它将给出以下结果：

    ```py
    230
    ```

1.  使用列表推导从列表中移除停用词并打印结果长度：

    ```py
    words = [word for word in words if word.lower() not in stopwords]
    print(len(words))
    ```

    结果将非常类似于NLTK：

    ```py
    106
    ```

1.  spaCy中的停用词存储在一个集合中，我们可以向其中添加更多单词：

    ```py
    print(len(stopwords))
    stopwords.add("new")
    print(len(stopwords))
    ```

    结果将如下所示：

    ```py
    327
    328
    ```

    同样，如果需要，我们可以移除单词：

    ```py
    print(len(stopwords))
    stopwords.remove("new")
    print(len(stopwords))
    ```

    结果将如下所示：

    ```py
    328
    327
    ```

我们也可以使用我们正在处理的文本来编译一个停用词列表，并计算其中单词的频率。这为你提供了一个自动移除停用词的方法，无需手动审查。

## 还有更多

在本节中，我们将展示两种实现方式。你需要使用文件[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes.txt)。NLTK包中的`FreqDist`对象计算每个单词的出现次数，这是我们后来用来找到最频繁的单词并将其移除作为停用词的依据：

1.  导入NTLK的**FreqDist**类：

    ```py
    from nltk.probability import FreqDist
    ```

1.  定义一个将编译停用词列表的函数：

    ```py
    def compile_stopwords_list_frequency(text, cut_off=0.02):
        words = word_tokenize_nltk(text)
        freq_dist = FreqDist(word.lower() for word in words)
        words_with_frequencies = [
            (word, freq_dist[word]) for word in freq_dist.keys()]
        sorted_words = sorted(words_with_frequencies, 
            key=lambda tup: tup[1])
        stopwords = []
        if (type(cut_off) is int):
            # First option: use a frequency cutoff
            stopwords = [tuple[0] for tuple in sorted_words 
                if tuple[1] > cut_off]
        elif (type(cut_off) is float):
            # Second option: use a percentage of the words
            length_cutoff = int(cut_off*len(sorted_words))
            stopwords = [tuple[0] for tuple in 
                sorted_words[-length_cutoff:]]
        else:
            raise TypeError("The cut off needs to be either a float (percentage) or an int (frequency cut off)")
        return stopwords
    ```

1.  使用默认设置定义停用词列表，并打印结果及其长度：

    ```py
    text = read_text_file("../data/sherlock_holmes.txt")
    stopwords = compile_stopwords_list_frequency(text)
    print(stopwords)
    print(len(stopwords))
    ```

    结果将如下所示：

    ```py
    ['make', 'myself', 'night', 'until', 'street', 'few', 'why', 'thought', 'take', 'friend', 'lady', 'side', 'small', 'still', 'these', 'find', 'st.', 'every', 'watson', 'too', 'round', 'young', 'father', 'left', 'day', 'yet', 'first', 'once', 'took', 'its', 'eyes', 'long', 'miss', 'through', 'asked', 'most', 'saw', 'oh', 'morning', 'right', 'last', 'like', 'say', 'tell', 't', 'sherlock', 'their', 'go', 'own', 'after', 'away', 'never', 'good', 'nothing', 'case', 'however', 'quite', 'found', 'made', 'house', 'such', 'heard', 'way', 'yes', 'hand', 'much', 'matter', 'where', 'might', 'just', 'room', 'any', 'face', 'here', 'back', 'door', 'how', 'them', 'two', 'other', 'came', 'time', 'did', 'than', 'come', 'before', 'must', 'only', 'know', 'about', 'shall', 'think', 'more', 'over', 'us', 'well', 'am', 'or', 'may', 'they', ';', 'our', 'should', 'now', 'see', 'down', 'can', 'some', 'if', 'will', 'mr.', 'little', 'who', 'into', 'do', 'has', 'could', 'up', 'man', 'out', 'when', 'would', 'an', 'are', 'by', '!', 'were', 's', 'then', 'one', 'all', 'on', 'no', 'what', 'been', 'your', 'very', 'him', 'her', 'she', 'so', ''', 'holmes', 'upon', 'this', 'said', 'from', 'there', 'we', 'me', 'be', 'but', 'not', 'for', '?', 'at', 'which', 'with', 'had', 'as', 'have', 'my', ''', 'is', 'his', 'was', 'you', 'he', 'it', 'that', 'in', '"', 'a', 'of', 'to', '"', 'and', 'i', '.', 'the', ',']
    181
    ```

1.  现在，使用频率截止值为5%的函数（使用最频繁的5%的单词作为停用词）：

    ```py
    text = read_text_file("../data/sherlock_holmes.txt")
    stopwords = compile_stopwords_list_frequency(text, cut_off=0.05)
    print(len(stopwords))
    ```

    结果将如下所示：

    ```py
    452
    ```

1.  现在，使用绝对频率截止值为**100**（选取频率大于100的单词）：

    ```py
    stopwords = compile_stopwords_list_frequency(text, cut_off=100)
    print(stopwords)
    print(len(stopwords))
    ```

    结果如下：

    ```py
    ['away', 'never', 'good', 'nothing', 'case', 'however', 'quite', 'found', 'made', 'house', 'such', 'heard', 'way', 'yes', 'hand', 'much', 'matter', 'where', 'might', 'just', 'room', 'any', 'face', 'here', 'back', 'door', 'how', 'them', 'two', 'other', 'came', 'time', 'did', 'than', 'come', 'before', 'must', 'only', 'know', 'about', 'shall', 'think', 'more', 'over', 'us', 'well', 'am', 'or', 'may', 'they', ';', 'our', 'should', 'now', 'see', 'down', 'can', 'some', 'if', 'will', 'mr.', 'little', 'who', 'into', 'do', 'has', 'could', 'up', 'man', 'out', 'when', 'would', 'an', 'are', 'by', '!', 'were', 's', 'then', 'one', 'all', 'on', 'no', 'what', 'been', 'your', 'very', 'him', 'her', 'she', 'so', ''', 'holmes', 'upon', 'this', 'said', 'from', 'there', 'we', 'me', 'be', 'but', 'not', 'for', '?', 'at', 'which', 'with', 'had', 'as', 'have', 'my', ''', 'is', 'his', 'was', 'you', 'he', 'it', 'that', 'in', '"', 'a', 'of', 'to', '"', 'and', 'i', '.', 'the', ',']
    131
    ```

创建停用词列表的函数接受文本和`cut_off`参数。它可以是表示停用词列表中频率排名单词百分比的浮点数。或者，它也可以是一个表示绝对阈值频率的整数，高于该频率的单词被视为停用词。在函数中，我们首先从书中提取单词，然后创建一个`FreqDist`对象，接着使用频率分布创建一个包含元组（单词，单词频率）的列表。我们使用单词频率对列表进行排序。然后，我们检查`cut_off`参数的类型，如果它不是浮点数或整数，则引发错误。如果是整数，我们返回频率高于参数的所有单词作为停用词。如果是浮点数，我们使用参数作为百分比来计算要返回的单词数量。
