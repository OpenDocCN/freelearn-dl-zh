# 第十一章：使用 Word2Vec 创建和可视化词向量

在本章中，我们将介绍以下内容：

+   获取数据

+   导入必要的库

+   数据准备

+   构建和训练模型

+   进一步可视化

+   进一步分析

# 介绍

在对文本数据进行神经网络训练并使用 LSTM 单元生成文本之前，理解文本数据（例如词语、句子、客户评论或故事）如何首先转换为词向量，并输入到神经网络中是非常重要的。本章将描述如何将文本转换为语料库，并从中生成词向量，使得使用欧几里得距离计算或余弦距离计算等技术将相似词语进行分组变得容易。

# 获取数据

第一步是获取一些数据来进行处理。在本章中，我们将需要大量的文本数据，将其转换为标记（tokens），并可视化以理解神经网络如何基于欧几里得距离和余弦相似度来对词向量进行排名。这是理解不同词语如何相互关联的重要步骤。反过来，这可以用于设计更好、更高效的语言和文本处理模型。

# 准备工作

请考虑以下内容：

+   模型的文本数据需要以`.txt`格式保存文件，并且必须确保这些文件放在当前工作目录中。文本数据可以是任何内容，例如 Twitter 动态、新闻摘要、客户评论、计算机代码或保存在`.txt`格式中的完整书籍。在我们的例子中，我们使用了《权力的游戏》系列书籍作为模型的输入文本。然而，任何文本都可以代替书籍，且相同的模型仍然有效。

+   许多经典的文本已经不再受到版权保护。这意味着你可以免费下载这些书籍的所有文本，并将它们用于实验，例如创建生成模型。获得不再受版权保护的免费书籍的最佳网站是古腾堡计划（[`www.gutenberg.org/`](https://www.gutenberg.org/)）。

# 如何操作...

步骤如下：

1.  首先访问古腾堡计划网站，浏览你感兴趣的书籍。点击书籍，然后点击 UTF-8，这样你就可以以纯文本格式下载该书籍。链接如以下截图所示：

![](img/de2e19ef-378b-4782-b3b1-0393bdb35026.png)

古腾堡计划数据集下载页面

1.  点击“Plain Text UTF-8”后，你应该会看到一个如下所示的页面。右键点击页面并选择“另存为...”，接下来，将文件重命名为你想要的名称，并保存到工作目录中：

![](img/44d32c31-04e8-4ee0-b4e6-2106b0afafc4.png)

1.  现在，你应该能在当前工作目录中看到一个`.txt`文件，文件名已按指定格式保存。

1.  Project Gutenberg 在每本书的开头和结尾添加标准的页眉和页脚；这些并不是原始文本的一部分。请在文本编辑器中打开文件，删除页眉和页脚。

# 它是如何工作的...

该功能如下：

1.  使用以下命令检查当前工作目录：`pwd`。

1.  工作目录可以使用 `cd` 命令更改，如下所示截图：

![](img/4399df8e-ceb1-4d15-aa8f-abd4ab5b8e26.png)

1.  请注意，在我们的案例中，文本文件存放在名为 `USF` 的文件夹中，因此这被设置为工作目录。你也可以类似地将一个或多个 `.txt` 文件存放在工作目录中，作为模型的输入。

1.  UTF-8 指定了文本文件中字符的编码类型。**UTF-8** 代表 **Unicode Transformation Format**。其中的 **8** 表示它使用 **8 位** 块来表示一个字符。

1.  UTF-8 是一种折衷的字符编码，它可以像 ASCII 一样紧凑（如果文件仅为纯英文文本），但也能包含任何 Unicode 字符（尽管文件大小会有所增加）。

1.  文本文件不需要是 UTF-8 格式，因为我们将在后期使用 codecs 库将所有文本编码为 Latin1 编码格式。

# 还有更多...

有关 UTF-8 和 Latin1 编码格式的更多信息，请访问以下链接：

+   [`en.wikipedia.org/wiki/UTF-8`](https://en.wikipedia.org/wiki/UTF-8)

+   [`www.ic.unicamp.br/~stolfi/EXPORT/www/ISO-8859-1-Encoding.html`](http://www.ic.unicamp.br/~stolfi/EXPORT/www/ISO-8859-1-Encoding.html)

# 另见

访问以下链接以更好地理解神经网络中单词向量的需求：

[`medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1`](https://medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1)

以下列出了一些与将单词转换为向量相关的其他有用文章：

[`monkeylearn.com/blog/word-embeddings-transform-text-numbers/`](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/)

[`towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817`](https://towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817)

# 导入必要的库

在我们开始之前，我们需要以下库和依赖项，这些都需要导入到我们的 Python 环境中。这些库将使我们的任务变得更轻松，因为它们提供了现成的函数和模型，可以直接使用，而不需要我们自己实现。这样也使得代码更简洁、可读。

# 准备开始

以下是创建单词向量、绘制图形以及在二维空间中可视化 n 维单词向量所需的库和依赖项：

+   `future`

+   `codecs`

+   `glob`

+   `` `multiprocessing` ``

+   `os`

+   `` `pprint` ``

+   `re`

+   `nltk`

+   `Word2Vec`

+   `sklearn`

+   `numpy`

+   `matplotlib`

+   `pandas`

+   `seaborn`

# 如何实现...

步骤如下：

1.  在你的 Jupyter notebook 中键入以下命令来导入所有必需的库：

```py
from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy
as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%pylab inline
```

1.  你应该看到如下截图中的输出：

![](img/cf709ffc-0cbc-419d-8b2c-b3b55f7c4daa.png)

1.  接下来，使用以下命令导入 `stopwords` 和 `punkt` 库：

```py
nltk.download("punkt")
nltk.download("stopwords")
```

1.  你看到的输出应如下图所示：

![](img/0e52652d-b201-4877-8399-99a018016314.png)

# 它是如何工作的...

本节将描述本配方中使用的每个库的目的。

1.  `future` 库是 Python 2 和 Python 3 之间的桥梁。它充当这两个版本之间的桥梁，允许我们使用这两个版本的语法。

1.  `codecs` 库将用于执行文本文件中所有单词的编码处理。这构成了我们的数据集。

1.  Regex 是用于快速查找或搜索文件的库。`glob` 函数允许快速高效地在大型数据库中搜索所需的文件。

1.  `multiprocessing` 库允许我们执行并发处理，这是通过运行多个线程并让每个线程运行不同的进程来实现的。这是一种通过并行化使程序运行更快的方法。

1.  `os` 库允许与操作系统（如 Mac、Windows 等）轻松交互，并执行诸如读取文件等功能。

1.  `pprint` 库提供了一种漂亮打印任意 Python 数据结构的能力，格式化后可以作为解释器的输入。

1.  `re` 模块提供了类似于 Perl 中的正则表达式匹配操作。

1.  NLTK 是一个自然语言工具包，能够用非常简短的代码对单词进行标记化。当输入一个完整的句子时，`nltk` 函数会将句子拆分并输出每个单词的标记。基于这些标记，单词可以被组织成不同的类别。NLTK 通过将每个单词与一个庞大的预训练词库（称为**词典**）进行比较来实现这一点。

1.  `Word2Vec` 是谷歌的模型，训练于一个巨大的词向量数据集。它将语义相似的词汇彼此靠近。这将是本节最重要的库。

1.  `sklearn.manifold` 通过使用**t-分布随机邻居嵌入**（**t-SNE**）技术来实现数据集的降维。由于每个词向量是多维的，我们需要某种形式的降维技术将这些词的维度降低到较低的空间，以便可以在二维空间中进行可视化。

# 还有更多内容...

`Numpy` 是一个常用的 `math` 库。`Matplotlib` 是我们将使用的 `plotting` 库，`pandas` 通过允许轻松地重塑、切片、索引、子集化和操作数据，提供了很大的灵活性。

`Seaborn`库是另一个统计数据可视化库，我们需要与`matplotlib`一起使用。`Punkt`和`Stopwords`是两个数据处理库，它们简化了任务，比如将语料库中的一段文本拆分为标记（即通过分词）并移除`stopwords`。

# 另见

有关所使用的某些库的更多信息，请访问以下链接：

+   [`docs.python.org/3/library/codecs.html`](https://docs.python.org/3/library/codecs.html)

+   [`docs.python.org/2/library/pprint.html`](https://docs.python.org/2/library/pprint.html)

+   [`docs.python.org/3/library/re.html`](https://docs.python.org/3/library/re.html)

+   [`www.nltk.org/`](https://www.nltk.org/)

+   [`www.tensorflow.org/tutorials/word2vec`](https://www.tensorflow.org/tutorials/word2vec)

+   [`scikit-learn.org/stable/modules/manifold.html`](http://scikit-learn.org/stable/modules/manifold.html)

# 数据准备

在将数据输入模型之前，需要进行一系列数据预处理步骤。本节将描述如何清理数据并准备好输入模型。

# 准备工作

所有来自`.txt`文件的文本首先会被转换成一个大的语料库。通过从每个文件中读取每个句子并将其添加到一个空的语料库中来实现。接着，会执行一系列预处理步骤，以去除不规则项，如空格、拼写错误、`stopwords`等。然后，清理过的文本数据需要被分词，分词后的句子会通过循环添加到一个空的数组中。

# 如何操作...

步骤如下：

1.  输入以下命令，以在工作目录中查找`.txt`文件并打印找到的文件名：

```py
book_names = sorted(glob.glob("./*.txt"))
print("Found books:")
book_names
```

在我们的案例中，有五本书，分别名为`got1`、`got2`、`got3`、`got4`和`got5`，它们保存在工作目录中。

1.  创建一个`corpus`，从第一个文件开始读取每个句子，对其进行编码，并将编码后的字符添加到`corpus`中，使用以下命令：

```py
corpus = u''
for book_name in book_names:
print("Reading '{0}'...".format(book_name))
with codecs.open(book_name,"r","Latin1") as book_file:
corpus += book_file.read()
print("Corpus is now {0} characters long".format(len(corpus)))
print()
```

1.  执行前述步骤中的代码，最终输出应该类似于以下截图所示：

![](img/9dc0f8ab-814f-46ce-9efa-8946d4378b2f.png)

1.  使用以下命令从`punkt`加载英语分词器`tokenizer`：

```py
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
```

1.  使用以下命令将整个`corpus`分词为句子：

```py
raw_sentences = tokenizer.tokenize(corpus)
```

1.  定义一个函数，将句子拆分为其组成的单词，并以以下方式移除不必要的字符：

```py
def sentence_to_wordlist(raw):
     clean = re.sub("[^a-zA-Z]"," ", raw)
     words = clean.split()
     return words
```

1.  将所有原始句子添加到一个新的句子数组中，每个句子的每个单词都已被分词。可以通过以下代码实现：

```py
sentences = []
for raw_sentence in raw_sentences:
  if len(raw_sentence) > 0:
  sentences.append(sentence_to_wordlist(raw_sentence))
```

1.  打印语料库中的一个随机句子，直观地看到`tokenizer`如何拆分句子并从结果中创建单词列表。可以使用以下命令来实现：

```py
print(raw_sentences[50])
print(sentence_to_wordlist(raw_sentences[50]))
```

1.  使用以下命令统计数据集中的所有标记：

```py
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))
```

# 它是如何工作的...

执行分词器并对语料库中的所有句子进行分词，应该会产生如下所示的输出：

![](img/1257f121-c354-454a-99af-54282e394815.png)

接下来，去除不必要的字符，如连字符和特殊字符，可以通过以下方式完成。使用用户定义的`sentence_to_wordlist()`函数拆分所有句子，生成如下所示的输出：

![](img/18472244-abe7-4029-a5aa-3ab539a44964.png)

将原始句子添加到一个名为`sentences[]`的新数组中，生成如下所示的输出：

![](img/c7d44661-9a51-4107-8661-8c09e6610aa4.png)

当打印语料库中的总分词数时，我们注意到整个语料库中有 1,110,288 个分词。这在以下截图中得到了说明：

![](img/a535b630-52be-4d91-8b31-bf9626571f4d.png)

功能如下：

1.  使用 NLTK 提供的预训练`tokenizer`对整个语料库进行分词，通过将每个句子作为一个分词进行计数。每个分词的句子都被添加到变量`raw_sentences`中，该变量存储了所有分词后的句子。

1.  下一步，将去除常见的停用词，并通过将每个句子拆分成单词来清理文本。

1.  随机打印一个句子及其单词列表，以理解这一过程是如何工作的。在我们的例子中，我们选择打印`raw_sentences`数组中的第 50 个句子。

1.  统计并打印句子数组中的总分词数（在我们的例子中是句子）。在我们的案例中，我们看到`tokenizer`创建了 1,110,288 个分词。

# 还有更多…

有关段落和句子分词的更多信息，请访问以下链接：

+   [`textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize`](https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize)

+   [`stackoverflow.com/questions/37605710/tokenize-a-paragraph-into-sentence-and-then-into-words-in-nltk`](https://stackoverflow.com/questions/37605710/tokenize-a-paragraph-into-sentence-and-then-into-words-in-nltk)

+   [`pythonspot.com/tokenizing-words-and-sentences-with-nltk/`](https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/)

# 另见

有关正则表达式工作原理的更多信息，请访问以下链接：

[`stackoverflow.com/questions/13090806/clean-line-of-punctuation-and-split-into-words-python`](https://stackoverflow.com/questions/13090806/clean-line-of-punctuation-and-split-into-words-python)

# 构建和训练模型

一旦我们拥有了以数组形式存储的分词数据，就可以将其输入到模型中。首先，我们需要为模型定义多个超参数。本节将介绍如何执行以下操作：

+   声明模型超参数

+   使用`Word2Vec`构建模型

+   在准备好的数据集上训练模型

+   保存并检查训练过的模型

# 准备就绪

需要声明的一些模型超参数包括以下内容：

+   结果词向量的维度

+   最小词数阈值

+   在训练模型时运行的并行线程数

+   上下文窗口长度

+   下采样（针对频繁出现的词）

+   设置种子

一旦之前提到的超参数被声明，可以使用来自`Gensim`库的`Word2Vec`函数来构建模型。

# 如何操作...

步骤如下：

1.  使用以下命令声明模型的超参数：

```py
num_features = 300
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1
```

1.  使用声明的超参数构建模型，代码如下：

```py
got2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
```

1.  使用分词后的句子并遍历所有的词元来构建模型的词汇表。可以通过以下方式使用`build_vocab`函数来完成：

```py
got2vec.build_vocab(sentences,progress_per=10000, keep_raw_vocab=False, trim_rule=None)
```

1.  使用以下命令训练模型：

```py
got2vec.train(sentences, total_examples=got2vec.corpus_count, total_words=None, epochs=got2vec.iter, start_alpha=None, end_alpha=None, word_count=0, queue_factor=2, report_delay=1.0, compute_loss=False)
```

1.  如果`trained`目录尚未存在，则创建该目录。使用以下命令保存并检查点`trained`模型：

```py
if not os.path.exists("trained"):
     os.makedirs("trained")
got2vec.wv.save(os.path.join("trained", "got2vec.w2v"), ignore=[])
```

1.  要在任何时刻加载保存的模型，使用以下命令：

```py
got2vec = w2v.KeyedVectors.load(os.path.join("trained", "got2vec.w2v"))
```

# 它是如何工作的...

功能如下：

1.  模型参数的声明不会产生任何输出，它仅仅是为存储模型参数的变量在内存中腾出空间。以下截图描述了这个过程：![](img/b01b9b42-aaa9-41a6-91dd-550faff73ec6.png)

1.  模型是使用前述的超参数构建的。在我们的案例中，我们将模型命名为`got2vec`，但模型的名称可以按您的喜好来命名。模型定义如下截图所示：![](img/abe1bb95-6a89-45e2-8060-229f2759473c.png)

1.  在模型上运行`build_vocab`命令后，应该会产生如下截图所示的输出：![](img/54a633e8-d76c-42a8-b40c-aea8e440a79d.png)

1.  训练模型通过定义如下截图中的参数来完成：![](img/a6116087-5efc-474e-9e00-69289485ade7.png)

1.  上述命令会产生如下截图所示的输出：

![](img/9d7dd945-1ed2-497f-9cd1-2e7daa05b4c7.png)

1.  保存、检查点和加载模型的命令会产生如下输出，如截图所示：

![](img/12a49260-afaf-484f-925c-cd73e1682646.png)

# 还有更多...

请考虑以下内容：

+   在我们的案例中，我们注意到`build_vocab`函数从 1,110,288 个词中识别出 23,960 个不同的词类型。然而，这个数字会因文本语料的不同而有所变化。

+   每个词被表示为一个 300 维的向量，因为我们已经将维度声明为 300。增加这个数字会增加模型的训练时间，但也确保模型能更容易地推广到新数据。

+   发现 1e![](img/4b993d4b-623a-44ac-8980-51f3d1ea5c11.png)3 的下采样率是一个不错的比率。该值用来告诉模型何时对频繁出现的词进行下采样，因为这些词在分析中并不重要。此类词的例子包括：this、that、those、them 等等。

+   设置一个种子以确保结果可重复。设置种子也能让调试变得更加容易。

+   训练模型大约需要 30 秒，使用普通 CPU 计算，因为模型不复杂。

+   模型在检查点时被保存在工作目录中的`trained`文件夹下。

# 另见

欲了解更多关于`Word2Vec`模型和 Gensim 库的信息，请访问以下链接：

[`radimrehurek.com/gensim/models/word2vec.html`](https://radimrehurek.com/gensim/models/word2vec.html)

# 进一步可视化

本节将描述如何将所有训练好的词的维度压缩，并将其放入一个巨大的矩阵中进行可视化。由于每个词是一个 300 维的向量，因此需要将其降维到较低的维度，以便我们在二维空间中可视化它。

# 准备工作

在模型保存并检查点训练完成后，像上一节那样开始将模型加载到内存中。本节将使用的库和模块有：

+   `tSNE`

+   `pandas`

+   `Seaborn`

+   `numpy`

# 如何操作...

步骤如下：

1.  使用以下命令压缩 300 维词向量的维度：

```py
 tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
```

1.  将所有词向量放入一个巨大的矩阵（命名为`all_word_vectors_matrix`）中，并使用以下命令查看：

```py
 all_word_vectors_matrix = got2vec.wv.syn0
 print (all_word_vectors_matrix)
```

1.  使用`tsne`技术通过以下命令将所有学习到的表示拟合到二维空间中：

```py
 all_word_vectors_matrix_2d =  tsne.fit_transform(all_word_vectors_matrix)
```

1.  使用以下代码收集所有的词向量以及它们关联的词：

```py
 points = pd.DataFrame(
     [
            (word, coords[0], coords[1])
             for word, coords in [
              (word, all_word_vectors_matrix_2d[got2vec.vocab[word].index])
                for word in got2vec.vocab
         ]
    ],
    columns=["word", "x", "y"]
)
```

1.  可以使用以下命令获取前十个点的`X`和`Y`坐标及其关联的词：

```py
points.head(10)
```

1.  使用以下命令绘制所有点：

```py
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(15, 15))
```

1.  可以放大绘制图中的选定区域进行更细致的检查。通过以下函数切片原始数据来实现这一点：

```py
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
        for i, point in slice.iterrows():
            ax.text(point.x + 0.005, point.y + 0.005, point.word,                                                  fontsize=11)
```

1.  使用以下命令绘制切片后的数据。切片数据可以被视为原始图中所有数据点的放大区域：

```py
plot_region(x_bounds=(20.0, 25.0), y_bounds=(15.5, 20.0))
```

# 它是如何工作的...

功能如下：

1.  t-SNE 算法是一种非线性降维技术。计算机在处理计算时可以轻松处理多个维度。然而，人类只能一次性可视化两到三维。因此，在尝试从数据中提取洞见时，这些降维技术非常有用。

1.  将 t-SNE 应用于 300 维的词向量后，我们能够将其压缩成二维，进行绘图并查看。

1.  通过将`n_components`指定为 2，我们告诉算法必须将数据压缩到二维空间。一旦完成，我们将所有压缩后的向量加入到一个名为`all_word_vectors_matrix`的巨型矩阵中，如下图所示：![](img/325c8759-34c8-4a40-ad78-7e050945cfe2.png)

1.  t-SNE 算法需要对所有这些词向量进行训练。使用普通 CPU 训练大约需要五分钟。

1.  一旦 t-SNE 在所有词向量上完成训练，它会输出每个单词的 2D 向量。这些向量可以通过将它们转换为数据框来作为点进行绘制。具体做法如下图所示：![](img/e3554859-c59a-4361-929c-5aac54dcf654.png)

1.  我们看到，前面的代码生成了多个点，每个点代表一个单词及其 X 和 Y 坐标。检查数据框中的前二十个点时，输出如下图所示：

![](img/a3db5c42-0c48-4990-a657-2a8117e6df71.png)

1.  使用`all_word_vectors_2D`变量绘制所有点时，你应该会看到一个类似于以下截图的输出：![](img/82c9e244-106f-4c01-825f-dbb00d5bd297.png)

1.  上述命令将生成整个文本中所有标记或单词的图表，如下图所示：![](img/7f14da02-37c8-42f0-a62c-019961507c8f.png)

1.  我们可以使用`plot_region`函数来缩放到图表的某个区域，这样就可以实际看到单词及其坐标。此步骤如以下截图所示：![](img/3c916ae2-d6a1-4d0b-b069-5a2a880562af.png)

1.  可以通过设置`x_bounds`和`y_bounds`值来放大或缩小图表的区域，如下图所示：![](img/a0214205-14a3-4165-ae29-22535d292dbb.png)

1.  可以通过更改`x_bounds`和`y_bounds`值来可视化同一图表的不同区域，如下图所示：![](img/ab6375bb-8bea-4697-8109-3e05c12cee31.png)

![](img/1041a93a-3c8e-42f2-a001-c0dea26cbc11.png)

# 另请参见

下面是一些需要注意的附加要点：

+   要了解有关 t-SNE 算法如何工作的更多信息，请访问以下链接：

+   [`www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm`](https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)

+   有关余弦距离相似度和排序的更多信息，请访问以下链接：

    [`code.google.com/archive/p/word2vec/`](https://code.google.com/archive/p/word2vec/)

+   使用以下链接来探索`Seaborn`库的不同功能：

    [`seaborn.pydata.org/`](https://seaborn.pydata.org/)

# 进一步分析

本节将描述在可视化后可以对数据执行的进一步分析。例如，探索不同词向量之间的余弦距离相似度。

# 准备开始

以下链接是一个很好的博客，讲解了余弦距离相似度的原理，并讨论了一些相关的数学内容：

[`blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/`](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)

# 如何做到...

请考虑以下内容：

+   使用`Word2Vec`的不同功能，可以执行各种自然语言处理任务。其中一个任务是给定某个单词后，找出最语义相似的单词（即，具有高余弦相似度或较短欧几里得距离的词向量）。这可以通过使用`Word2Vec`的`most_similar`函数来实现，如以下截图所示：![](img/9ac6fa2d-4380-4100-80be-ec37006e6efd.png)该截图显示了与单词`Lannister`相关的所有最接近的单词：

    ![](img/30dd7d1c-bbed-4cef-9a87-377ff69d022a.png)该截图显示了与单词`Jon`相关的所有词汇：![](img/ec9b90eb-4a33-49a9-ac48-75b56b4f65e6.png)

# 它是如何工作的...

请考虑以下内容：

+   有多种方法可以测量单词之间的语义相似度。本节中我们使用的方法是基于余弦相似度的。我们还可以通过使用以下代码行来探索单词之间的线性关系：

```py
 def nearest_similarity_cosmul(start1, end1, end2):
    similarities = got2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
)
start2 = similarities[0][0]
print("{start1} is related to {end1}, as {start2} is related to         {end2}".format(**locals()))
return start2
```

+   要找出与给定一组单词最接近的单词的余弦相似度，可以使用以下命令：

```py
nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")
```

+   上述过程在以下截图中进行了说明：![](img/09338881-485f-4402-945c-c4ff5abc61b5.png)

+   结果如下：![](img/0af641a8-cf30-4c33-8994-300ef137b545.png)

+   如本节所示，词向量构成了所有自然语言处理（NLP）任务的基础。在深入研究更复杂的 NLP 模型，如**递归神经网络**和**长短期记忆**（**LSTM**）单元之前，理解词向量及其构建数学模型是非常重要的。

# 另见

为了更好地理解余弦距离相似度、聚类和用于排序词向量的其他机器学习技术，可以进一步阅读以下内容。以下是一些关于此主题的有用论文链接：

+   [`s3.amazonaws.com/academia.edu.documents/32952068/pg049_Similarity_Measures_for_Text_Document_Clustering.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1530163881&Signature=YG6YjvJb2z0JjmfHzaYujA2ioIo%3D&response-content-disposition=inline%3B%20filename%3DSimilarity_Measures_for_Text_Document_Cl.pdf`](https://s3.amazonaws.com/academia.edu.documents/32952068/pg049_Similarity_Measures_for_Text_Document_Clustering.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1530163881&Signature=YG6YjvJb2z0JjmfHzaYujA2ioIo%3D&response-content-disposition=inline%3B%20filename%3DSimilarity_Measures_for_Text_Document_Cl.pdf)

+   [`csis.pace.edu/ctappert/dps/d861-12/session4-p2.pdf`](http://csis.pace.edu/ctappert/dps/d861-12/session4-p2.pdf)
