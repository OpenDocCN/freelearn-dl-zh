# 第七章：摘要

# 训练并设置一个能够像人类一样对话的聊天机器人

本章将向你展示如何训练一个自动聊天机器人，使其能够回答简单且通用的问题，以及如何通过 HTTP 创建一个端点，通过 API 提供答案。更具体地，我们将展示：

+   什么是语料库以及如何预处理语料库

+   如何训练一个聊天机器人以及如何测试它

+   如何创建一个 HTTP 端点来暴露 API

# 项目介绍

聊天机器人正变得越来越普及，成为为用户提供帮助的一种方式。许多公司，包括银行、移动/固话公司以及大型电子商务公司，现在都在使用聊天机器人为客户提供帮助，并在售前阶段帮助用户。如今，单纯的 Q&A 页面已经不够了：每个客户现在都期待得到针对自己问题的答案，而这些问题可能在 Q&A 中没有覆盖或只是部分覆盖。此外，对于那些不需要为琐碎问题提供额外客户服务容量的公司来说，聊天机器人是一项很棒的工具：这真的是一种双赢的局面！

自从深度学习流行以来，聊天机器人已经成为非常流行的工具。得益于深度学习，我们现在能够训练机器人提供更好的个性化问题，并且在最新的实现中，还能够保持每个用户的上下文。

简单来说，主要有两种类型的聊天机器人：第一种是简单的聊天机器人，它尝试理解话题，总是为所有关于同一话题的问题提供相同的答案。例如，在火车网站上，问题*“从 City_A 到 City_B 的时刻表在哪里？”*和*“从 City_A 出发的下一班火车是什么？”*可能会得到相同的答案，可能是*“你好！我们网络上的时刻表可以在这个页面找到：<link>”*。基本上，这种类型的聊天机器人通过分类算法来理解话题（在这个例子中，两个问题都是关于时刻表的话题）。在确定话题后，它们总是提供相同的答案。通常，它们有一个包含 N 个话题和 N 个答案的列表；此外，如果分类出来的话题概率较低（问题太模糊，或是涉及到不在列表中的话题），它们通常会要求用户更具体地说明并重复问题，最终可能会提供其他提问方式（例如发送电子邮件或拨打客服热线）。

第二种类型的聊天机器人更为先进、更智能，但也更复杂。对于这种类型的聊天机器人，答案是通过 RNN（循环神经网络）构建的，方式类似于机器翻译的实现（见前一章）。这些聊天机器人能够提供更个性化的答案，并且它们可能提供更具体的回复。事实上，它们不仅仅是猜测话题，而是通过 RNN 引擎，能够更好地理解用户的问题并提供最佳的答案：实际上，使用这种类型的聊天机器人，两个不同问题得到相同答案的可能性非常小。

在本章中，我们将尝试使用 RNN 构建第二种类型的聊天机器人，类似于我们在上一章中使用机器翻译系统所做的。同时，我们将展示如何将聊天机器人放在 HTTP 端点后面，以便将聊天机器人作为服务从您的网站或更简单地从命令行中使用。

# 输入语料库

不幸的是，我们没有找到任何面向消费者的、开放源代码并且可以自由使用的网络数据集。因此，我们将使用一个更通用的数据集来训练聊天机器人，而不是专注于客户服务的聊天数据集。具体来说，我们将使用康奈尔电影对话语料库（Cornell Movie Dialogs Corpus），该语料库来自康奈尔大学。该语料库包含从原始电影剧本中提取的对话集，因此聊天机器人能够更多地回答虚构性问题而非现实性问题。康奈尔语料库包含来自 617 部电影中的 10,000 多个电影角色之间的 200,000 多个对话交换。

数据集可以在这里获取：[`www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html`](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)。

我们要感谢作者发布了这个语料库：这使得实验、可重复性和知识共享变得更加容易。

数据集以`.zip`归档文件的形式提供。解压后，您将找到其中的几个文件：

+   `README.txt`包含了数据集的描述、语料库文件的格式、收集过程的细节以及作者的联系方式。

+   `Chameleons.pdf`是发布该语料库的原始论文。虽然论文的主要目标并不直接围绕聊天机器人，但它研究了对话中使用的语言，是理解更多内容的好信息来源。

+   `movie_conversations.txt`包含了所有的对话结构。对于每个对话，它包括参与讨论的两个人物 ID、电影 ID 以及按时间顺序排列的句子 ID（或者更准确地说是发言 ID）列表。例如，文件的第一行是：

*u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']*

这意味着用户`u0`在电影`m0`中与用户`u2`进行了对话，该对话包含了 4 个发言：`'L194'`、`'L195'`、`'L196'`和`'L197'`。

+   `movie_lines.txt`包含了每个发言 ID 的实际文本及其发言者。例如，发言`L195`在此列出为：

*L195 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Well, I thought we'd start with pronunciation, if that's okay with you.*

所以，发言`L195`的文本是*Well, I thought we'd start with pronunciation, if that's okay with you.*，并且是由电影`m0`中的角色`u2`（名为`CAMERON`）所发出的。

+   `movie_titles_metadata.txt` 包含有关电影的信息，包括标题、年份、IMDB 评分、IMDB 的投票数和流派。例如，这里描述的电影 `m0` 是这样的：

*m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ 62847 +++$+++ ['喜剧', '爱情']*

因此，电影 ID 为 `m0` 的电影标题是 *10 things i hate about you*，出自 1999 年，是一部喜剧爱情片，IMDB 上获得了近 63,000 票，平均评分为 6.9（满分 10 分）。

+   `movie_characters_metadata.txt` 包含有关电影角色的信息，包括角色名、出现的电影标题、性别（如果已知）和在演职员表中的位置（如果已知）。例如，角色“u2”在这个文件中以此描述：

*u2 +++$+++ CAMERON +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ m +++$+++ 3*

角色 `u2` 的名字是 *CAMERON*，出现在电影 `m0` 中，标题是 *10 things i hate about you*，他是男性，排名第三。

+   `raw_script_urls.txt` 包含可以检索每部电影对话的源 URL。例如，对于电影 `m0`，它是：

*m0 +++$+++ 10 things i hate about you +++$+++ http://www.dailyscript.com/scripts/10Things.html*

正如您注意到的那样，大多数文件使用标记 *+++$+++* 分隔字段。除此之外，该格式看起来相当容易解析。请特别注意解析文件时的格式：它们不是 UTF-8，而是 *ISO-8859-1*。

# 创建训练数据集

现在让我们为聊天机器人创建训练集。我们需要所有角色之间按正确顺序的对话：幸运的是，语料库包含了我们实际需要的以上内容。为了创建数据集，我们将从下载 zip 存档开始（如果尚未在磁盘上）。然后，我们将在临时文件夹解压缩存档（如果您使用 Windows，应该是 `C:\Temp`），并且我们将仅读取 `movie_lines.txt` 和 `movie_conversations.txt` 文件，这些是我们真正需要创建连续话语数据集的文件。

现在让我们一步一步地进行，创建多个函数，每个步骤一个函数，在文件 `corpora_downloader.py` 中。我们需要的第一个函数是，如果磁盘上没有可用，从互联网上检索文件。

```py
def download_and_decompress(url, storage_path, storage_dir):
   import os.path
   directory = storage_path + "/" + storage_dir
   zip_file = directory + ".zip"
   a_file = directory + "/cornell movie-dialogs corpus/README.txt"
   if not os.path.isfile(a_file):
       import urllib.request
       import zipfile
       urllib.request.urlretrieve(url, zip_file)
       with zipfile.ZipFile(zip_file, "r") as zfh:
           zfh.extractall(directory)
   return
```

此函数正是这样做的：它检查本地是否有 “`README.txt`” 文件；如果没有，它将下载文件（感谢 `urllib.request` 模块中的 `urlretrieve` 函数），然后解压缩 zip（使用 `zipfile` 模块）。

下一步是读取对话文件并提取话语 ID 列表。提醒一下，它的格式是：*u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']*，因此我们需要关注的是通过用*+++$+++*分割后的列表中的第四个元素。此外，我们还需要清除方括号和撇号，以获得一个干净的 ID 列表。为此，我们将导入 re 模块，函数将如下所示。

```py
import re
def read_conversations(storage_path, storage_dir):
   filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_conversations.txt"
   with open(filename, "r", encoding="ISO-8859-1") as fh:
       conversations_chunks = [line.split(" +++$+++ ") for line in fh]
   return [re.sub('[\[\]\']', '', el[3].strip()).split(", ") for el in conversations_chunks]
```

如前所述，记得以正确的编码读取文件，否则会出现错误。此函数的输出是一个包含对话中角色话语 ID 序列的列表的列表。下一步是读取并解析`movie_lines.txt`文件，以提取实际的对话文本。提醒一下，文件的格式如下：

*L195 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ 好吧，我想我们从发音开始，如果你没问题的话。*

在这里，我们需要关注的是第一个和最后一个块。

```py
def read_lines(storage_path, storage_dir):
   filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_lines.txt"
   with open(filename, "r", encoding="ISO-8859-1") as fh:
       lines_chunks = [line.split(" +++$+++ ") for line in fh]
   return {line[0]: line[-1].strip() for line in lines_chunks}
```

最后部分涉及到标记化和对齐。我们希望拥有一组观察结果，其中包含两个连续的话语。通过这种方式，我们可以训练聊天机器人，在给定第一个话语的情况下，生成下一个话语。希望这能促使聊天机器人变得智能，能够回答多个问题。以下是这个函数：

```py
def get_tokenized_sequencial_sentences(list_of_lines, line_text):
   for line in list_of_lines:
       for i in range(len(line) - 1):
           yield (line_text[line[i]].split(" "), line_text[line[i+1]].split(" "))
```

它的输出是一个生成器，包含两个话语的元组（右边的那个时间上紧跟在左边的后面）。此外，话语是在空格字符上进行标记化的。

最后，我们可以将所有内容封装到一个函数中，该函数下载文件并解压（如果未缓存），解析对话和行，并将数据集格式化为生成器。默认情况下，我们将文件存储在`/tmp`目录中：

```py
def retrieve_cornell_corpora(storage_path="/tmp", storage_dir="cornell_movie_dialogs_corpus"):
   download_and_decompress("http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",      
                     storage_path,
                           storage_dir)
   conversations = read_conversations(storage_path, storage_dir)
   lines = read_lines(storage_path, storage_dir)
   return tuple(zip(*list(get_tokenized_sequencial_sentences(conversations, lines))))
```

此时，我们的训练集与上一章翻译项目中使用的训练集非常相似。实际上，它不仅相似，它是相同的格式和相同的目标。因此，我们可以使用在上一章中开发的一些代码片段。例如，`corpora_tools.py`文件可以在这里直接使用而不需要任何更改（此外，它还依赖于`data_utils.py`）。

给定该文件，我们可以进一步深入分析语料库，使用一个脚本检查聊天机器人的输入。

要检查语料库，我们可以使用在上一章中编写的`corpora_tools.py`，以及我们之前创建的文件。让我们获取 Cornell 电影对话语料库，格式化语料库并打印一个示例及其长度：

```py
from corpora_tools import *
from corpora_downloader import retrieve_cornell_corpora
sen_l1, sen_l2 = retrieve_cornell_corpora()
print("# Two consecutive sentences in a conversation")
print("Q:", sen_l1[0])
print("A:", sen_l2[0])
print("# Corpora length (i.e. number of sentences)")
print(len(sen_l1))
assert len(sen_l1) == len(sen_l2)
```

这段代码打印了两个标记化的连续话语示例，以及数据集中示例的数量，超过了 220,000 个：

```py
# Two consecutive sentences in a conversation
Q: ['Can', 'we', 'make', 'this', 'quick?', '', 'Roxanne', 'Korrine', 'and', 'Andrew', 'Barrett', 'are', 'having', 'an', 'incredibly', 'horrendous', 'public', 'break-', 'up', 'on', 'the', 'quad.', '', 'Again.']
A: ['Well,', 'I', 'thought', "we'd", 'start', 'with', 'pronunciation,', 'if', "that's", 'okay', 'with', 'you.']
# Corpora length (i.e. number of sentences)
221616
```

现在，让我们清理句子中的标点符号，将其转为小写，并将其长度限制为最多 20 个单词（也就是那些至少有一个句子长度超过 20 个单词的示例会被丢弃）。这是为了标准化标记：

```py
clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2)
print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)
```

这将使我们得到近 140,000 个示例：

```py
# Filtered Corpora length (i.e. number of sentences)
140261
```

Then, let's create the dictionaries for the two sets of sentences. Practically, they should look the same (since the same sentence appears once on the left side, and once in the right side) except there might be some changes introduced by the first and last sentences of a conversation (they appear only once). To make the best out of our corpora, let's build two dictionaries of words and then encode all the words in the corpora with their dictionary indexes:

```py
dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path="/tmp/l1_dict.p")
dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=15000, storage_path="/tmp/l2_dict.p")
idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)
print("# Same sentences as before, with their dictionary ID")
print("Q:", list(zip(filt_clean_sen_l1[0], idx_sentences_l1[0])))
print("A:", list(zip(filt_clean_sen_l2[0], idx_sentences_l2[0])))
```

That prints the following output. We also notice that a dictionary of 15 thousand entries doesn't contain all the words and more than 16 thousand (less popular) of them don't fit into it:

```py
[sentences_to_indexes] Did not find 16823 words
[sentences_to_indexes] Did not find 16649 words
# Same sentences as before, with their dictionary ID
Q: [('well', 68), (',', 8), ('i', 9), ('thought', 141), ('we', 23), ("'", 5), ('d', 83), ('start', 370), ('with', 46), ('pronunciation', 3), (',', 8), ('if', 78), ('that', 18), ("'", 5), ('s', 12), ('okay', 92), ('with', 46), ('you', 7), ('.', 4)]
A: [('not', 31), ('the', 10), ('hacking', 7309), ('and', 23), ('gagging', 8761), ('and', 23), ('spitting', 6354), ('part', 437), ('.', 4), ('please', 145), ('.', 4)]
```

As the final step, let's add paddings and markings to the sentences:

```py
data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
print("# Prepared minibatch with paddings and extra stuff")
print("Q:", data_set[0][0])
print("A:", data_set[0][1])
print("# The sentence pass from X to Y tokens")
print("Q:", len(idx_sentences_l1[0]), "->", len(data_set[0][0]))
print("A:", len(idx_sentences_l2[0]), "->", len(data_set[0][1]))
```

And that, as expected, prints:

```py
# Prepared minibatch with paddings and extra stuff
Q: [0, 68, 8, 9, 141, 23, 5, 83, 370, 46, 3, 8, 78, 18, 5, 12, 92, 46, 7, 4]
A: [1, 31, 10, 7309, 23, 8761, 23, 6354, 437, 4, 145, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# The sentence pass from X to Y tokens
Q: 19 -> 20
A: 11 -> 22
```

# Training the chatbot

After we're done with the corpora, it's now time to work on the model. This project requires again a sequence to sequence model, therefore we can use an RNN. Even more, we can reuse part of the code from the previous project: we'd just need to change how the dataset is built, and the parameters of the model. We can then copy the training script built in the previous chapter, and modify the `build_dataset` function, to use the Cornell dataset.

Mind that the dataset used in this chapter is bigger than the one used in the previous, therefore you may need to limit the corpora to a few dozen thousand lines. On a 4 years old laptop with 8GB RAM, we had to select only the first 30 thousand lines, otherwise, the program ran out of memory and kept swapping. As a side effect of having fewer examples, even the dictionaries are smaller, resulting in less than 10 thousands words each.

```py
def build_dataset(use_stored_dictionary=False):
   sen_l1, sen_l2 = retrieve_cornell_corpora()
   clean_sen_l1 = [clean_sentence(s) for s in sen_l1][:30000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
   clean_sen_l2 = [clean_sentence(s) for s in sen_l2][:30000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
   filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2, max_len=10)
   if not use_stored_dictionary:
       dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=10000, storage_path=path_l1_dict)
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

By inserting this function into the `train_translator.py` file (from the previous chapter) and rename the file as `train_chatbot.py`, we can run the training of the chatbot.

After a few iterations, you can stop the program and you'll see something similar to this output:

```py
[sentences_to_indexes] Did not find 0 words
[sentences_to_indexes] Did not find 0 words
global step 100 learning rate 1.0 step-time 7.708967611789704 perplexity 444.90090078460474
eval: perplexity 57.442316329639176
global step 200 learning rate 0.990234375 step-time 7.700247814655302 perplexity 48.8545568311572
eval: perplexity 42.190180314697045
global step 300 learning rate 0.98046875 step-time 7.69800933599472 perplexity 41.620538109894945
eval: perplexity 31.291903031786116
...
...
...
global step 2400 learning rate 0.79833984375 step-time 7.686293318271639 perplexity 3.7086356605442767
eval: perplexity 2.8348589631663046
global step 2500 learning rate 0.79052734375 step-time 7.689657487869262 perplexity 3.211876894960698
eval: perplexity 2.973809378544393
global step 2600 learning rate 0.78271484375 step-time 7.690396382808681 perplexity 2.878854805600354
eval: perplexity 2.563583924617356
```

Again, if you change the settings, you may end up with a different perplexity. To obtain these results, we set the RNN size to 256 and 2 layers, the batch size of 128 samples, and the learning rate to 1.0.

At this point, the chatbot is ready to be tested. Although you can test the chatbot with the same code as in the `test_translator.py` of the previous chapter, here we would like to do a more elaborate solution, which allows exposing the chatbot as a service with APIs.

# Chatbox API

First of all, we need a web framework to expose the API. In this project, we've chosen Bottle, a lightweight simple framework very easy to use.

To install the package, run `pip install bottle` from the command line. To gather further information and dig into the code, take a look at the project webpage, [`bottlepy.org`](https://bottlepy.org).

现在让我们创建一个函数，用来解析用户作为参数提供的任意句子。所有接下来的代码应该都写在`test_chatbot_aas.py`文件中。我们从一些导入和使用字典来清理、分词并准备句子的函数开始：

```py
import pickle
import sys
import numpy as np
import tensorflow as tf
import data_utils
from corpora_tools import clean_sentence, sentences_to_indexes, prepare_sentences
from train_chatbot import get_seq2seq_model, path_l1_dict, path_l2_dict
model_dir = "/home/abc/chat/chatbot_model"
def prepare_sentence(sentence, dict_l1, max_length):
   sents = [sentence.split(" ")]
   clean_sen_l1 = [clean_sentence(s) for s in sents]
   idx_sentences_l1 = sentences_to_indexes(clean_sen_l1, dict_l1)
   data_set = prepare_sentences(idx_sentences_l1, [[]], max_length, max_length)
   sentences = (clean_sen_l1, [[]])
   return sentences, data_set
```

`prepare_sentence`函数执行以下操作：

+   对输入句子进行分词

+   清理它（转换为小写并清理标点符号）

+   将词元转换为字典 ID

+   添加标记和填充以达到默认长度

接下来，我们需要一个函数，将预测的数字序列转换为由单词组成的实际句子。这是通过`decode`函数完成的，该函数根据输入句子运行预测，并使用 softmax 预测最可能的输出。最后，它返回没有填充和标记的句子（函数的更详细描述见上一章）：

```py
def decode(data_set):
with tf.Session() as sess:
   model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths, model_dir)
   model.batch_size = 1
   bucket = 0
   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
     {bucket: [(data_set[0][0], [])]}, bucket)
   _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket, True)
   outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
   if data_utils.EOS_ID in outputs:
       outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
tf.reset_default_graph()
return " ".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs])
```

最后是主函数，也就是在脚本中运行的函数：

```py
if __name__ == "__main__":
   dict_l1 = pickle.load(open(path_l1_dict, "rb"))
   dict_l1_length = len(dict_l1)
   dict_l2 = pickle.load(open(path_l2_dict, "rb"))
   dict_l2_length = len(dict_l2)
   inv_dict_l2 = {v: k for k, v in dict_l2.items()}
   max_lengths = 10
   dict_lengths = (dict_l1_length, dict_l2_length)
   max_sentence_lengths = (max_lengths, max_lengths)
   from bottle import route, run, request
   @route('/api')
   def api():
       in_sentence = request.query.sentence
     _, data_set = prepare_sentence(in_sentence, dict_l1, max_lengths)
       resp = [{"in": in_sentence, "out": decode(data_set)}]
       return dict(data=resp)
   run(host='127.0.0.1', port=8080, reloader=True, debug=True)
```

初始时，它加载字典并准备反向字典。接着，它使用 Bottle API 创建一个 HTTP GET 端点（在/api URL 下）。路由装饰器设置并增强了当通过 HTTP GET 访问该端点时运行的函数。在这种情况下，运行的是`api()`函数，它首先读取作为 HTTP 参数传递的句子，然后调用上述的`prepare_sentence`函数，最后执行解码步骤。返回的是一个字典，其中包含用户提供的输入句子和聊天机器人的回复。

最后，网页服务器已启动，运行在 localhost 的 8080 端口上。使用 Bottle 实现聊天机器人作为服务是不是非常简单？

现在是时候运行它并检查输出了。要运行它，请从命令行执行：

```py
$> python3 –u test_chatbot_aas.py
```

接着，让我们开始用一些通用问题查询聊天机器人，为此我们可以使用 CURL，这是一个简单的命令行工具；此外，所有浏览器都可以使用，只需记住 URL 应当编码，例如，空格字符应该用它的编码替代，即`%20`。

Curl 让事情变得更容易，它提供了一种简单的方式来编码 URL 请求。以下是几个示例：

```py
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=how are you?"
{"data": [{"out": "i ' m here with you .", "in": "where are you?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=are you here?"
{"data": [{"out": "yes .", "in": "are you here?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=are you a chatbot?"
{"data": [{"out": "you ' for the stuff to be right .", "in": "are you a chatbot?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=what is your name ?"
{"data": [{"out": "we don ' t know .", "in": "what is your name ?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=how are you?"
{"data": [{"out": "that ' s okay .", "in": "how are you?"}]}
```

如果系统在你的浏览器中无法正常工作，请尝试对 URL 进行编码，例如：

`$> curl -X GET http://127.0.0.1:8080/api?sentence=how%20are%20you?` `{"data": [{"out": "that ' s okay .", "in": "how are you?"}]}`

回复相当有趣；始终记得我们训练聊天机器人的数据集是电影，因此回复的风格跟电影有关。

要关闭网页服务器，请使用*Ctrl* + `C`。

# 家庭作业

以下是家庭作业：

+   你能创建一个简单的网页，通过 JS 查询聊天机器人吗？

+   互联网上有许多其他训练集可供选择；尝试查看不同模型之间的回答差异。哪种最适合客户服务机器人？

+   你能否修改模型，使其作为服务进行训练，即通过 HTTP GET/POST 传递句子？

# 摘要

在本章中，我们实现了一个聊天机器人，能够通过 HTTP 端点和 GET API 回答问题。这是我们使用 RNN 能做的又一个精彩示例。在下一章，我们将转向另一个话题：如何使用 Tensorflow 创建推荐系统。
