# 第六章：创建并训练机器翻译系统

本项目的目标是训练一个**人工智能**（**AI**）模型，使其能够在两种语言之间进行翻译。具体来说，我们将看到一个自动翻译器，它读取德语并生成英语句子；不过，本章中开发的模型和代码足够通用，可以应用于任何语言对。

本章探讨的项目有四个重要部分，如下所示：

+   架构概述

+   预处理语料库

+   训练机器翻译器

+   测试与翻译

它们每个都会描述项目中的一个关键组件，最后，你将对发生的事情有一个清晰的了解。

# 架构概述

一个机器翻译系统接收一种语言的任意字符串作为输入，并生成一个在另一种语言中具有相同含义的字符串作为输出。Google 翻译就是一个例子（但许多其他大型 IT 公司也有自己的系统）。在这里，用户可以进行超过 100 种语言之间的翻译。使用网页非常简单：在左侧输入你想翻译的句子（例如，Hello World），选择其语言（在这个例子中是英语），然后选择你希望翻译成的目标语言。

这是一个例子，我们将句子 Hello World 翻译成法语：

![](img/ccea21d1-123d-41a1-b60d-7cae15f0e6cf.png)

这很容易吗？乍一看，我们可能会认为这只是简单的字典替换。单词被分块，翻译通过特定的英法词典查找，每个单词用其翻译进行替换。不幸的是，事实并非如此。在这个例子中，英语句子有两个单词，而法语句子有三个。更一般来说，想想短语动词（turn up, turn off, turn on, turn down），萨克森属格，语法性别，时态，条件句……它们并不总是有直接的翻译，正确的翻译应当根据句子的上下文来决定。

这就是为什么，在进行机器翻译时，我们需要一些人工智能工具。具体来说，就像许多其他**自然语言处理**（**NLP**）任务一样，我们将使用**循环神经网络**（**RNN**）。我们在上一章介绍了 RNN，主要特点是它们处理序列：给定输入序列，产生输出序列。本章的目标是创建正确的训练流程，以使句子作为输入序列，其翻译作为输出序列。还要记住*没有免费的午餐定理*：这个过程并不简单，更多的解决方案可以用相同的结果创造出来。在本章中，我们将提出一个简单而强大的方法。

首先，我们从语料库开始：这可能是最难找到的部分，因为它应该包含从一种语言到另一种语言的高保真度翻译。幸运的是，NLTK，一个著名的 Python 自然语言处理包，包含了 Comtrans 语料库。**Comtrans**是**机器翻译组合方法**（combination approach to machine translation）的缩写，包含了三种语言的对齐语料库：德语、法语和英语。

在这个项目中，我们将使用这些语料库出于以下几个原因：

1.  它可以很容易地在 Python 中下载和导入。

1.  不需要预处理来从磁盘/互联网读取它。NLTK 已经处理了这部分内容。

1.  它足够小，可以在许多笔记本电脑上使用（只有几万句）。

1.  它可以自由地在互联网上获取。

关于 Comtrans 项目的更多信息，请访问 [`www.fask.uni-mainz.de/user/rapp/comtrans/`](http://www.fask.uni-mainz.de/user/rapp/comtrans/)。

更具体来说，我们将尝试创建一个机器翻译系统，将德语翻译成英语。我们随机选择了这两种语言，作为 Comtrans 语料库中可用语言的其中一对：你可以自由选择交换它们，或者改用法语语料库。我们的项目管道足够通用，可以处理任何组合。

现在，让我们通过输入一些命令来调查语料库的组织结构：

```py
from nltk.corpus import comtrans
print(comtrans.aligned_sents('alignment-de-en.txt')[0])
```

输出如下：

```py
<AlignedSent: 'Wiederaufnahme der S...' -> 'Resumption of the se...'>
```

句子对可以通过函数`aligned_sents`获取。文件名包含源语言和目标语言。在这种情况下，作为项目的后续部分，我们将翻译德语（*de*）到英语（*en*）。返回的对象是类`nltk.translate.api.AlignedSent`的一个实例。从文档中可以看到，第一个语言可以通过属性`words`访问，第二个语言可以通过属性`mots`访问。所以，为了分别提取德语句子和其英语翻译，我们应该运行：

```py
print(comtrans.aligned_sents()[0].words)
print(comtrans.aligned_sents()[0].mots)
```

上面的代码输出：

```py
['Wiederaufnahme', 'der', 'Sitzungsperiode']
['Resumption', 'of', 'the', 'session']
```

真棒！这些句子已经被分词，并且看起来像是序列。实际上，它们将是我们项目中 RNN 的输入和（希望）输出，RNN 将为我们提供德语到英语的机器翻译服务。

此外，如果你想了解语言的动态，Comtrans 还提供了翻译中单词的对齐：

```py
print(comtrans.aligned_sents()[0].alignment)
```

上面的代码输出：

```py
0-0 1-1 1-2 2-3
```

德语中的第一个词被翻译为英语中的第一个词（*Wiederaufnahme*到*Resumption*），第二个词被翻译为第二个词（*der*到*of*和*the*），第三个（索引为 1）被翻译为第四个词（*Sitzungsperiode*到*session*）。

# 语料库的预处理

第一步是获取语料库。我们已经看到过如何做到这一点，但现在让我们将其形式化为一个函数。为了使其足够通用，我们将把这些函数封装在一个名为`corpora_tools.py`的文件中。

1.  让我们导入一些稍后会用到的内容：

```py
import pickle
import re
from collections import Counter
from nltk.corpus import comtrans
```

1.  现在，让我们创建一个函数来获取语料库：

```py
def retrieve_corpora(translated_sentences_l1_l2='alignment-de-en.txt'):
    print("Retrieving corpora: {}".format(translated_sentences_l1_l2))
    als = comtrans.aligned_sents(translated_sentences_l1_l2)
    sentences_l1 = [sent.words for sent in als]
    sentences_l2 = [sent.mots for sent in als]
    return sentences_l1, sentences_l2
```

这个函数有一个参数；包含来自 NLTK Comtrans 语料库的对齐句子的文件。它返回两个句子列表（实际上是词汇列表），一个用于源语言（在我们的例子中是德语），另一个用于目标语言（在我们的例子中是英语）。

1.  在一个单独的 Python REPL 中，我们可以测试这个函数：

```py
sen_l1, sen_l2 = retrieve_corpora()
print("# A sentence in the two languages DE & EN")
print("DE:", sen_l1[0])
print("EN:", sen_l2[0])
print("# Corpora length (i.e. number of sentences)")
print(len(sen_l1))
assert len(sen_l1) == len(sen_l2)
```

1.  上述代码生成了以下输出：

```py
Retrieving corpora: alignment-de-en.txt
# A sentence in the two languages DE & EN
DE: ['Wiederaufnahme', 'der', 'Sitzungsperiode']
EN: ['Resumption', 'of', 'the', 'session']
# Corpora length (i.e. number of sentences)
33334
```

我们还打印了每个语料库中的句子数量（33,000），并确认源语言和目标语言的句子数量相同。

1.  在接下来的步骤中，我们希望清理掉无用的标记。具体来说，我们要对标点符号进行分词处理，并将所有词汇小写。为此，我们可以在`corpora_tools.py`中创建一个新函数。我们将使用`regex`模块来进一步分词：

```py
def clean_sentence(sentence):
    regex_splitter = re.compile("([!?.,:;$\"')( ])")
    clean_words = [re.split(regex_splitter, word.lower()) for word in sentence]
    return [w for words in clean_words for w in words if words if w]
```

1.  再次，在 REPL 中，我们来测试这个函数：

```py
clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
print("# Same sentence as before, but chunked and cleaned")
print("DE:", clean_sen_l1[0])
print("EN:", clean_sen_l2[0])
```

上述代码输出与之前相同的句子，但已分块并清理：

```py
DE: ['wiederaufnahme', 'der', 'sitzungsperiode']
EN: ['resumption', 'of', 'the', 'session']
```

不错！

该项目的下一步是筛选出过长的句子，无法进行处理。由于我们的目标是在本地机器上进行处理，我们应该限制句子的长度在*N*个词以内。在这种情况下，我们将*N*设置为 20，以便在 24 小时内能够训练学习器。如果你有一台强大的机器，可以随意提高这个限制。为了使函数足够通用，还设置了一个下限，默认值为 0，例如一个空的词汇集。

1.  函数的逻辑非常简单：如果句子或其翻译的词汇数大于*N*，那么就将该句子（无论源语言还是目标语言）移除：

```py
def filter_sentence_length(sentences_l1, sentences_l2, min_len=0, max_len=20):
    filtered_sentences_l1 = []
    filtered_sentences_l2 = []
    for i in range(len(sentences_l1)):
        if min_len <= len(sentences_l1[i]) <= max_len and \
                 min_len <= len(sentences_l2[i]) <= max_len:
            filtered_sentences_l1.append(sentences_l1[i])
            filtered_sentences_l2.append(sentences_l2[i])
    return filtered_sentences_l1, filtered_sentences_l2
```

1.  再次，让我们在 REPL 中查看有多少句子通过了这个过滤器。记住，我们起始时有超过 33,000 个句子：

```py
filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, 
          clean_sen_l2)
print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)
```

上述代码打印出以下输出：

```py
# Filtered Corpora length (i.e. number of sentences)
14788
```

大约 15,000 个句子存活下来，也就是语料库的一半。

现在，我们终于从文本转向数字（AI 主要使用这些）。为此，我们将为每种语言创建一个词典。这个词典应该足够大，能够包含大多数词汇，尽管我们可以丢弃一些出现频率很低的词汇。如果某种语言有低频词汇，这是常见做法，就像 tf-idf（文档中词频乘以文档频率的倒数，即该词在多少个文档中出现）一样，极为罕见的词汇会被丢弃，以加速计算并使解决方案更加可扩展和通用。在这里，我们需要在两个词典中分别有四个特殊符号：

1.  一个符号用于填充（稍后我们会看到为什么需要它）

1.  一个符号用于分隔两个句子

1.  一个符号表示句子的结束位置

1.  一个符号用于表示未知词汇（比如那些非常罕见的词）

为此，让我们创建一个新的文件，命名为`data_utils.py`，并包含以下代码行：

```py
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]
```

然后，返回到`corpora_tools.py`文件中，让我们添加以下函数：

```py
import data_utils

def create_indexed_dictionary(sentences, dict_size=10000, storage_path=None):
    count_words = Counter()
    dict_words = {}
    opt_dict_size = len(data_utils.OP_DICT_IDS)
    for sen in sentences:
        for word in sen:
            count_words[word] += 1

    dict_words[data_utils._PAD] = data_utils.PAD_ID
    dict_words[data_utils._GO] = data_utils.GO_ID
    dict_words[data_utils._EOS] = data_utils.EOS_ID
    dict_words[data_utils._UNK] = data_utils.UNK_ID

    for idx, item in enumerate(count_words.most_common(dict_size)):
        dict_words[item[0]] = idx + opt_dict_size
    if storage_path:
        pickle.dump(dict_words, open(storage_path, "wb"))
    return dict_words
```

这个函数的参数包括字典中的条目数和存储字典的路径。记住，字典是在训练算法时创建的：在测试阶段它会被加载，且令牌/符号的关联应与训练中使用的一致。如果唯一令牌的数量大于设定的值，则只选择最常见的那些。最终，字典包含每种语言中令牌及其 ID 之间的关联。

在构建字典之后，我们应该查找令牌并用它们的令牌 ID 进行替换。

为此，我们需要另一个函数：

```py
def sentences_to_indexes(sentences, indexed_dictionary):
    indexed_sentences = []
    not_found_counter = 0
    for sent in sentences:
        idx_sent = []
        for word in sent:
            try:
                idx_sent.append(indexed_dictionary[word])
            except KeyError:
                idx_sent.append(data_utils.UNK_ID)
                not_found_counter += 1
        indexed_sentences.append(idx_sent)

    print('[sentences_to_indexes] Did not find {} words'.format(not_found_counter))
    return indexed_sentences
```

这一步非常简单；令牌会被替换成其 ID。如果令牌不在字典中，则使用未知令牌的 ID。让我们在 REPL 中查看经过这些步骤后的句子：

```py
dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path="/tmp/l1_dict.p")
dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path="/tmp/l2_dict.p")
idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)
print("# Same sentences as before, with their dictionary ID")
print("DE:", list(zip(filt_clean_sen_l1[0], idx_sentences_l1[0])))
```

这段代码打印了两个句子的令牌及其 ID。RNN 中使用的将只是每个元组的第二个元素，也就是整数 ID：

```py
# Same sentences as before, with their dictionary ID
DE: [('wiederaufnahme', 1616), ('der', 7), ('sitzungsperiode', 618)]
EN: [('resumption', 1779), ('of', 8), ('the', 5), ('session', 549)]
```

另外请注意，像英语中的*the*和*of*，德语中的*der*等常见令牌，其 ID 较低。这是因为 ID 是按流行度排序的（见函数`create_indexed_dictionary`的主体）。

即使我们做了过滤以限制句子的最大长度，我们仍然应该创建一个函数来提取最大长度。对于那些拥有非常强大机器的幸运用户，如果没有进行任何过滤，那么现在就是看 RNN 中最长期限句子多长的时刻。这个函数就是：

```py
def extract_max_length(corpora):
    return max([len(sentence) for sentence in corpora])
```

让我们对这些句子应用以下操作：

```py
max_length_l1 = extract_max_length(idx_sentences_l1)
max_length_l2 = extract_max_length(idx_sentences_l2)
print("# Max sentence sizes:")
print("DE:", max_length_l1)
print("EN:", max_length_l2)
```

如预期的那样，输出为：

```py
# Max sentence sizes:
DE: 20
EN: 20
```

最终的预处理步骤是填充。我们需要所有序列具有相同的长度，因此需要填充较短的序列。此外，我们需要插入正确的令牌，指示 RNN 字符串的开始和结束位置。

基本上，这一步应该：

+   填充输入序列，使它们都为 20 个符号长

+   填充输出序列，使其为 20 个符号长

+   在输出序列的开头插入一个`_GO`，在结尾插入一个`_EOS`，用以标识翻译的开始和结束

这是通过这个函数完成的（将其插入到`corpora_tools.py`中）：

```py
def prepare_sentences(sentences_l1, sentences_l2, len_l1, len_l2):
    assert len(sentences_l1) == len(sentences_l2)
    data_set = []
    for i in range(len(sentences_l1)):
        padding_l1 = len_l1 - len(sentences_l1[i])
        pad_sentence_l1 = ([data_utils.PAD_ID]*padding_l1) + sentences_l1[i]
        padding_l2 = len_l2 - len(sentences_l2[i])
        pad_sentence_l2 = [data_utils.GO_ID] + sentences_l2[i] + [data_utils.EOS_ID] + ([data_utils.PAD_ID] * padding_l2)
        data_set.append([pad_sentence_l1, pad_sentence_l2])
    return data_set
```

为了测试它，让我们准备数据集并打印第一句：

```py
data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
print("# Prepared minibatch with paddings and extra stuff")
print("DE:", data_set[0][0])
print("EN:", data_set[0][1])
print("# The sentence pass from X to Y tokens")
print("DE:", len(idx_sentences_l1[0]), "->", len(data_set[0][0]))
print("EN:", len(idx_sentences_l2[0]), "->", len(data_set[0][1]))
```

上述代码输出如下：

```py
# Prepared minibatch with paddings and extra stuff
DE: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1616, 7, 618]
EN: [1, 1779, 8, 5, 549, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# The sentence pass from X to Y tokens
DE: 3 -> 20
EN: 4 -> 22
```

正如你所看到的，输入和输出都通过零进行填充以保持常数长度（在字典中，它们对应于`_PAD`，见`data_utils.py`），输出中包含标记 1 和 2，分别位于句子的开始和结束之前。根据文献证明的有效方法，我们将填充输入句子的开始，并填充输出句子的结束。完成此操作后，所有输入句子的长度都是`20`，输出句子的长度是`22`。

# 训练机器翻译器

到目前为止，我们已经看到了预处理语料库的步骤，但尚未看到使用的模型。实际上，模型已经可以在 TensorFlow Models 仓库中找到，可以从 [`github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py`](https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py) 免费下载。

这段代码采用 Apache 2.0 许可证。我们非常感谢作者开源了如此出色的模型。版权所有 2015 TensorFlow 作者。保留所有权利。根据 Apache 许可证第 2.0 版（许可证）授权；除非符合该许可证，否则不得使用此文件。你可以在此处获得许可证副本：[`www.apache.org/licenses/LICENSE-2.0`](http://www.apache.org/licenses/LICENSE-2.0) 除非适用法律要求或书面同意，否则按“原样”分发软件，且不提供任何形式的保证或条件。有关许可证下特定权限和限制的语言，请参见许可证。

我们将在本节中看到模型的使用。首先，让我们创建一个名为 `train_translator.py` 的新文件，并导入一些库和常量。我们将把字典保存在 `/tmp/` 目录下，以及模型和它的检查点：

```py
import time
import math
import sys
import pickle
import glob
import os
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from corpora_tools import *

path_l1_dict = "/tmp/l1_dict.p"
path_l2_dict = "/tmp/l2_dict.p"
model_dir = "/tmp/translate "
model_checkpoints = model_dir + "/translate.ckpt"
```

现在，让我们在一个函数中使用前面部分创建的所有工具，给定一个布尔标志返回语料库。更具体地说，如果参数是`False`，则从头开始构建字典（并保存）；否则，它使用路径中现有的字典：

```py
def build_dataset(use_stored_dictionary=False):
    sen_l1, sen_l2 = retrieve_corpora()
    clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
    clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
    filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2)

    if not use_stored_dictionary:
        dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path=path_l1_dict)
        dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path=path_l2_dict)
    else:
        dict_l1 = pickle.load(open(path_l1_dict, "rb"))
        dict_l2 = pickle.load(open(path_l2_dict, "rb"))

    dict_l1_length = len(dict_l1)
    dict_l2_length = len(dict_l2)

    idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
    idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)

    max_length_l1 = extract_max_length(idx_sentences_l1)
    max_length_l2 = extract_max_length(idx_sentences_l2)

    data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
    return (filt_clean_sen_l1, filt_clean_sen_l2), \
        data_set, \
        (max_length_l1, max_length_l2), \
        (dict_l1_length, dict_l2_length)
```

这个函数返回清理后的句子、数据集、句子的最大长度以及字典的长度。

此外，我们还需要一个清理模型的函数。每次运行训练例程时，我们需要清理模型目录，因为我们没有提供任何垃圾信息。我们可以通过一个非常简单的函数来实现这一点：

```py
def cleanup_checkpoints(model_dir, model_checkpoints):
    for f in glob.glob(model_checkpoints + "*"):
    os.remove(f)
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
```

最后，让我们以可重用的方式创建模型：

```py
def get_seq2seq_model(session, forward_only, dict_lengths, max_sentence_lengths, model_dir):
    model = Seq2SeqModel(
            source_vocab_size=dict_lengths[0],
            target_vocab_size=dict_lengths[1],
            buckets=[max_sentence_lengths],
            size=256,
            num_layers=2,
            max_gradient_norm=5.0,
            batch_size=64,
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            forward_only=forward_only,
            dtype=tf.float16)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model
```

这个函数调用模型的构造函数，传递以下参数：

+   源语言词汇大小（在我们的示例中是德语）

+   目标词汇大小（在我们的示例中是英语）

+   桶（在我们的示例中只有一个，因为我们已将所有序列填充为单一大小）

+   **长短期记忆**（**LSTM**）内部单元的大小

+   堆叠的 LSTM 层数

+   梯度的最大范数（用于梯度裁剪）

+   小批量大小（即每个训练步骤的观察次数）

+   学习率

+   学习率衰减因子

+   模型的方向

+   数据类型（在我们的示例中，我们将使用 flat16，即使用 2 个字节的浮动类型）

为了加速训练并获得良好的模型表现，我们已经在代码中设置了这些值；你可以自由更改它们并查看效果。

函数中的最终 if/else 语句会从检查点中检索模型（如果模型已经存在）。事实上，这个函数也会在解码器中使用，以在测试集上检索并处理模型。

最后，我们达到了训练机器翻译器的函数。它是这样的：

```py
def train():
    with tf.Session() as sess:
        model = get_seq2seq_model(sess, False, dict_lengths, max_sentence_lengths, model_dir)
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        bucket = 0
        steps_per_checkpoint = 100
        max_steps = 20000
        while current_step < max_steps:
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step {} learning rate {} step-time {} perplexity {}".format(
                model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                sess.run(model.learning_rate_decay_op)
                model.saver.save(sess, model_checkpoints, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, True)
                eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                print(" eval: perplexity {}".format(eval_ppx))
                sys.stdout.flush()    
```

该函数通过创建模型开始。此外，它设置了一些常量，用于确定每个检查点的步骤数和最大步骤数。具体来说，在代码中，我们将在每 100 步保存一次模型，并且最多执行 20,000 步。如果这仍然需要太长时间，可以随时终止程序：每个检查点都包含一个训练好的模型，解码器将使用最新的模型。

到这一点，我们进入了 while 循环。每一步，我们要求模型获取一个大小为 64 的小批量数据（如之前设置的）。`get_batch` 方法返回输入（即源序列）、输出（即目标序列）和模型的权重。通过 `step` 方法，我们执行一步训练。返回的信息之一是当前小批量数据的损失值。这就是所有的训练！

为了每 100 步报告性能并保存模型，我们会打印模型在过去 100 步中的平均困惑度（数值越低越好），并保存检查点。困惑度是与预测不确定性相关的指标：我们对单词的信心越强，输出句子的困惑度就越低。此外，我们重置计数器，并从测试集中的一个小批量数据（在这个案例中是数据集中的一个随机小批量）中提取相同的指标，并打印其性能。然后，训练过程会重新开始。

作为一种改进，每 100 步我们还会将学习率降低一个因子。在这种情况下，我们将其乘以 0.99。这有助于训练的收敛性和稳定性。

现在我们需要将所有函数连接在一起。为了创建一个可以通过命令行调用的脚本，同时也可以被其他脚本导入函数，我们可以创建一个 `main` 函数，如下所示：

```py
if __name__ == "__main__":
    _, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    cleanup_checkpoints(model_dir, model_checkpoints)
    train()
```

在控制台中，你现在可以使用非常简单的命令来训练你的机器翻译系统：

```py
$> python train_translator.py
```

在一台普通的笔记本电脑上，没有 NVIDIA GPU，困惑度降到 10 以下需要一天多的时间（12 个小时以上）。这是输出：

```py
Retrieving corpora: alignment-de-en.txt
[sentences_to_indexes] Did not find 1097 words
[sentences_to_indexes] Did not find 0 words
Created model with fresh parameters.
global step 100 learning rate 0.5 step-time 4.3573073434829713 perplexity 526.6638556683066
eval: perplexity 159.2240770935855
[...]
global step 10500 learning rate 0.180419921875 step-time 4.35106209993362414 perplexity 2.0458043055629487
eval: perplexity 1.8646006006241982
[...]
```

# 测试并翻译

翻译的代码在文件 `test_translator.py` 中。

我们从一些导入和预训练模型的位置开始：

```py
import pickle
import sys
import numpy as np
import tensorflow as tf
import data_utils
from train_translator import (get_seq2seq_model, path_l1_dict, path_l2_dict,
build_dataset)
model_dir = "/tmp/translate"
```

现在，让我们创建一个函数来解码 RNN 生成的输出序列。请注意，序列是多维的，每个维度对应于该单词的概率，因此我们将选择最可能的单词。在反向字典的帮助下，我们可以找出实际的单词是什么。最后，我们将修剪掉标记（填充、开始、结束符号），并打印输出。

在这个例子中，我们将解码训练集中的前五个句子，从原始语料库开始。随时可以插入新的字符串或使用不同的语料库：

```py
def decode():
    with tf.Session() as sess:
        model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths, model_dir)
        model.batch_size = 1
        bucket = 0
        for idx in range(len(data_set))[:5]:
            print("-------------------")
            print("Source sentence: ", sentences[0][idx])
            print("Source tokens: ", data_set[idx][0])
            print("Ideal tokens out: ", data_set[idx][1])
            print("Ideal sentence out: ", sentences[1][idx])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            {bucket: [(data_set[idx][0], [])]}, bucket)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
            target_weights, bucket, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if data_utils.EOS_ID in outputs:
                outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
            print("Model output: ", " ".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs]))
            sys.stdout.flush()
```

在这里，我们再次需要一个`main`来与命令行配合使用，如下所示：

```py
if __name__ == "__main__":
    dict_l2 = pickle.load(open(path_l2_dict, "rb"))
    inv_dict_l2 = {v: k for k, v in dict_l2.items()}
    build_dataset(True)
    sentences, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
    try:
        print("Reading from", model_dir)
        print("Dictionary lengths", dict_lengths)
        print("Bucket size", max_sentence_lengths)
    except NameError:
        print("One or more variables not in scope. Translation not possible")
        exit(-1)
    decode()
```

运行上述代码会生成以下输出：

```py
Reading model parameters from /tmp/translate/translate.ckpt-10500
-------------------
Source sentence: ['wiederaufnahme', 'der', 'sitzungsperiode']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1616, 7, 618]
Ideal tokens out: [1, 1779, 8, 5, 549, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['resumption', 'of', 'the', 'session']
Model output: resumption of the session
-------------------
Source sentence: ['ich', 'bitte', 'sie', ',', 'sich', 'zu', 'einer', 'schweigeminute', 'zu', 'erheben', '.']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 266, 22, 5, 29, 14, 78, 3931, 14, 2414, 4]
Ideal tokens out: [1, 651, 932, 6, 159, 6, 19, 11, 1440, 35, 51, 2639, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['please', 'rise', ',', 'then', ',', 'for', 'this', 'minute', "'", 's', 'silence', '.']
Model output: i ask you to move , on an approach an approach .
-------------------
Source sentence: ['(', 'das', 'parlament', 'erhebt', 'sich', 'zu', 'einer', 'schweigeminute', '.', ')']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 11, 58, 3267, 29, 14, 78, 3931, 4, 51]
Ideal tokens out: [1, 54, 5, 267, 3541, 14, 2095, 12, 1440, 35, 51, 2639, 53, 2, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['(', 'the', 'house', 'rose', 'and', 'observed', 'a', 'minute', "'", 's', 'silence', ')']
Model output: ( the house ( observed and observed a speaker )
-------------------
Source sentence: ['frau', 'präsidentin', ',', 'zur', 'geschäftsordnung', '.']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 151, 5, 49, 488, 4]
Ideal tokens out: [1, 212, 44, 6, 22, 12, 91, 8, 218, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['madam', 'president', ',', 'on', 'a', 'point', 'of', 'order', '.']
Model output: madam president , on a point of order .
-------------------
Source sentence: ['wenn', 'das', 'haus', 'damit', 'einverstanden', 'ist', ',', 'werde', 'ich', 'dem', 'vorschlag', 'von', 'herrn', 'evans', 'folgen', '.']
Source tokens: [0, 0, 0, 0, 85, 11, 603, 113, 831, 9, 5, 243, 13, 39, 141, 18, 116, 1939, 417, 4]
Ideal tokens out: [1, 87, 5, 267, 2096, 6, 16, 213, 47, 29, 27, 1941, 25, 1441, 4, 2, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['if', 'the', 'house', 'agrees', ',', 'i', 'shall', 'do', 'as', 'mr', 'evans', 'has', 'suggested', '.']
Model output: if the house gave this proposal , i would like to hear mr byrne .
```

如你所见，输出结果主要是正确的，尽管仍然存在一些有问题的标记。为了减轻这个问题，我们需要一个更复杂的 RNN、更长的语料库或更多样化的语料库。

# 家庭作业

该模型是在相同的数据集上进行训练和测试的；这对数据科学来说并不理想，但为了有一个可运行的项目，这是必要的。尝试找一个更长的语料库，并将其拆分成两部分，一部分用于训练，另一部分用于测试：

+   更改模型的设置：这会如何影响性能和训练时间？

+   分析`seq2seq_model.py`中的代码。如何在 TensorBoard 中插入损失的图表？

+   NLTK 还包含法语语料库；你能否创建一个系统，将它们一起翻译？

在本章中，我们已经学习了如何基于 RNN 创建一个机器翻译系统。我们了解了如何组织语料库、如何训练它以及如何测试它。在下一章中，我们将看到 RNN 的另一个应用：聊天机器人。
