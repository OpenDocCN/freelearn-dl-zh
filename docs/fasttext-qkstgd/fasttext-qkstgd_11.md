# 第十一章：给读者的注意事项

# Windows 和 Linux

我们建议您使用 PowerShell 作为 Windows 命令行工具，因为它比简单的 `cmd` 更强大。

| **任务** | **Windows** | **Linux**/**macOS** |
| --- | --- | --- |
| 创建目录 | `mkdir` | `mkdir` |
| 切换目录 | `cd` | `cd` |
| 移动文件 | `move` | `mv` |
| 解压文件 | 图形界面并双击 | `unzip` |
| 文件顶部 | `get-content` | `head` |
| 文件内容 | `type` | `cat` |
| 管道 | `this pipes objects` | `this pipes text` |
| 文件底部 | 带有 `get-content` 的 `-wait` 参数 | `tail` |

`python` 和 `perl` 命令在 Windows 中的工作方式与在 bash 中相同，因此您可以以类似方式使用这些文件，特别是 `perl` 单行命令。

# Python 2 和 Python 3

fastText 适用于 Python 2 和 Python 3。尽管如此，您应该注意特定 Python 版本之间的一些差异。

1.  `print` 在 Python 2 中是一个语句，而在 Python 3 中是一个函数。这意味着，如果您在 Jupyter Notebook 中查看变量的变化，您需要根据对应的 Python 版本使用适当的 `print` 语句。

1.  fastText 处理文本时使用 Unicode 编码。Python 3 也处理文本为 Unicode，因此如果您使用 Python 3 编写代码，不会增加额外的开销。但如果您使用 Python 2 开发模型，数据不能是字符串实例。您需要将数据作为 Unicode 处理。以下是一个在 Python 2 中，`str` 类和 `unicode` 类的文本实例示例。

```py
>>> text1 = "some text" # this will not work for fastText
>>> type(text1)
<type 'str'>
>>> text2 = unicode("some text") # in fastText you will need to use this.
>>> type(text2)
<type 'unicode'>
>>>
```

# fastText 命令行

以下是您可以与 fastText 命令行一起使用的参数列表：

```py
$ ./fasttext
usage: fasttext <command> <args>

The commands supported by fasttext are:

 supervised train a supervised classifier
 quantize quantize a model to reduce the memory usage
 test evaluate a supervised classifier
 predict predict most likely labels
 predict-prob predict most likely labels with probabilities
 skipgram train a skipgram model
 cbow train a cbow model
 print-word-vectors print word vectors given a trained model
 print-sentence-vectors print sentence vectors given a trained model
 print-ngrams print ngrams given a trained model and word
 nn query for nearest neighbors
 analogies query for analogies
 dump dump arguments,dictionary,input/output vectors
```

`supervised`、`skipgram` 和 `cbow` 命令用于训练模型。`predict`、`predict-prob` 用于在有监督模型上进行预测。`test`、`print-word-vectors`、`print-sentence-vectors`、`print-ngrams`、`nn`、analogies 可用于评估模型。`dump` 命令基本上是用来查找模型的超参数，`quantize` 用于压缩模型。

您可以用于训练的超参数列表稍后会列出。

# fastText 有监督

```py
$ ./fasttext supervised
Empty input or output path.

The following arguments are mandatory:
 -input training file path
 -output output file path

The following arguments are optional:
 -verbose verbosity level [2]

The following arguments for the dictionary are optional:
 -minCount minimal number of word occurences [1]
 -minCountLabel minimal number of label occurences [0]
 -wordNgrams max length of word ngram [1]
 -bucket number of buckets [2000000]
 -minn min length of char ngram [0]
 -maxn max length of char ngram [0]
 -t sampling threshold [0.0001]
 -label labels prefix [__label__]

The following arguments for training are optional:
 -lr learning rate [0.1]
 -lrUpdateRate change the rate of updates for the learning rate [100]
 -dim size of word vectors [100]
 -ws size of the context window [5]
 -epoch number of epochs [5]
 -neg number of negatives sampled [5]
 -loss loss function {ns, hs, softmax} [softmax]
 -thread number of threads [12]
 -pretrainedVectors pretrained word vectors for supervised learning []
 -saveOutput whether output params should be saved [false]

The following arguments for quantization are optional:
 -cutoff number of words and ngrams to retain [0]
 -retrain whether embeddings are finetuned if a cutoff is applied [false]
 -qnorm whether the norm is quantized separately [false]
 -qout whether the classifier is quantized [false]
 -dsub size of each sub-vector [2]
```

# fastText skipgram

```py
$ ./fasttext skipgram
Empty input or output path.

The following arguments are mandatory:
 -input training file path
 -output output file path

The following arguments are optional:
 -verbose verbosity level [2]

The following arguments for the dictionary are optional:
 -minCount minimal number of word occurences [5]
 -minCountLabel minimal number of label occurences [0]
 -wordNgrams max length of word ngram [1]
 -bucket number of buckets [2000000]
 -minn min length of char ngram [3]
 -maxn max length of char ngram [6]
 -t sampling threshold [0.0001]
 -label labels prefix [__label__]

The following arguments for training are optional:
 -lr learning rate [0.05]
 -lrUpdateRate change the rate of updates for the learning rate [100]
 -dim size of word vectors [100]
 -ws size of the context window [5]
 -epoch number of epochs [5]
 -neg number of negatives sampled [5]
 -loss loss function {ns, hs, softmax} [ns]
 -thread number of threads [12]
 -pretrainedVectors pretrained word vectors for supervised learning []
 -saveOutput whether output params should be saved [false]

The following arguments for quantization are optional:
 -cutoff number of words and ngrams to retain [0]
 -retrain whether embeddings are finetuned if a cutoff is applied [false]
 -qnorm whether the norm is quantized separately [false]
 -qout whether the classifier is quantized [false]
 -dsub size of each sub-vector [2]
```

# fastText cbow

```py
$ ./fasttext cbow
Empty input or output path.

The following arguments are mandatory:
 -input training file path
 -output output file path

The following arguments are optional:
 -verbose verbosity level [2]

The following arguments for the dictionary are optional:
 -minCount minimal number of word occurences [5]
 -minCountLabel minimal number of label occurences [0]
 -wordNgrams max length of word ngram [1]
 -bucket number of buckets [2000000]
 -minn min length of char ngram [3]
 -maxn max length of char ngram [6]
 -t sampling threshold [0.0001]
 -label labels prefix [__label__]

The following arguments for training are optional:
 -lr learning rate [0.05]
 -lrUpdateRate change the rate of updates for the learning rate [100]
 -dim size of word vectors [100]
 -ws size of the context window [5]
 -epoch number of epochs [5]
 -neg number of negatives sampled [5]
 -loss loss function {ns, hs, softmax} [ns]
 -thread number of threads [12]
 -pretrainedVectors pretrained word vectors for supervised learning []
 -saveOutput whether output params should be saved [false]

The following arguments for quantization are optional:
 -cutoff number of words and ngrams to retain [0]
 -retrain whether embeddings are finetuned if a cutoff is applied [false]
 -qnorm whether the norm is quantized separately [false]
 -qout whether the classifier is quantized [false]
 -dsub size of each sub-vector [2]
```

# Gensim fastText 参数

Gensim 支持与 fastText 本地实现相同的超参数。您应该能够按如下方式设置它们：

+   `sentences`：这可以是一个包含标记的列表的列表。一般来说，建议使用标记流，如之前提到的 word2vec 模块中的 `LineSentence`。在 Facebook fastText 库中，这由文件路径提供，并通过 `-input` 参数传递。

+   `sg`：可以是 1 或 0。1 表示训练 skip-gram 模型，0 表示训练 CBOW 模型。在 Facebook fastText 库中，等效操作是传递 `skipgram` 和 `cbow` 参数。

+   `size`：词向量的维度，因此必须是整数。与原始实现一致，默认选择 100。这与 Facebook fastText 实现中的`-dim`参数类似。

+   `window`：围绕一个词语考虑的窗口大小。这与原始实现中的`-ws`参数相同。

+   `alpha`：这是初始学习率，类型为浮动数。它与第二章中看到的`-lr`参数相同，*使用 FastText 命令行创建模型*。

+   `min_alpha`：这是训练过程中学习率降至的最小值。

+   `seed`：这是为了可复现性。为了让种子生效，线程数也需要设置为 1。

+   `min_count`：文档中单词的最小频率，低于此频率的单词将被丢弃。类似于命令行中的`-minCount`参数。

+   `max_vocab_size`：用于限制 RAM 大小。如果词汇表中有更多的唯一单词，那么会修剪掉频率较低的单词。这个值需要根据你拥有的 RAM 大小来决定。例如，如果你有 2GB 内存，则`max_vocab_size`需要为 10M * 2 = 2000 万（20 000 000）。

+   `sample`：用于对单词进行下采样。类似于 fasttext 命令行中的"-t"参数。

+   `workers`：训练的线程数，类似于 fastText 命令中的`-thread`参数。

+   `hs`：可以是 0 或 1。如果是 1，则会使用层次化 softmax 作为损失函数。

+   `negative`：如果你想使用负采样作为损失函数，则将`hs`设置为 0，并将 negative 设为非零正数。请注意，损失函数仅支持两种功能：层次化 softmax 和负采样。简单的 softmax 不被支持。这个参数和`hs`一起，等同于`fasttext`命令中的`-loss`参数。

+   `cbow_mean`：这里与 fastText 命令有些不同。在原始实现中，对于`cbow`会取向量的均值。但在这种情况下，你可以选择通过传递 0 来使用和 1 来尝试均值。

+   `hashfxn`：用于随机初始化权重的哈希函数。

+   `iter`：样本的迭代次数或周期数。这与命令行中的`-epoch`参数相同。

+   `trim_rule`：用于指定是否应保留某些词汇或将其修剪掉的函数。

+   `sorted_vocab`：接受的值为 1 或 0。如果为 1，则词汇表将在索引之前进行排序。

+   `batch_words`：这是传递的批次的目标大小。默认值为 10000。这与命令行中的`-lrUpdateRate`有些相似，因为批次数决定了权重何时更新。

+   `min_n`和`max_n`：字符 n-grams 的最小和最大长度。

+   `word_ngrams`：丰富子词信息，以便在训练过程中使用。

+   bucket：字符 n-gram 被哈希到一个固定大小的向量上。默认情况下使用 200 万词的桶大小。

+   `callbacks`：在训练过程中特定阶段执行的回调函数列表。
