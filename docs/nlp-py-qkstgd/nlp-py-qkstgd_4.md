# 第四章：文本表示 - 将单词转换为数字

今天的计算机不能直接对单词或文本进行操作。它们需要通过有意义的数字序列来表示。这些长序列的十进制数字被称为向量，这一步骤通常被称为文本的向量化。

那么，这些词向量在哪里使用：

+   在文本分类和摘要任务中

+   在类似词的搜索中，例如同义词

+   在机器翻译中（例如，将文本从英语翻译成德语）

+   当理解类似文本时（例如，Facebook文章）

+   在问答会话和一般任务中（例如，用于预约安排的聊天机器人）

非常频繁地，我们看到词向量在某种形式的分类任务中使用。例如，使用机器学习或深度学习模型进行情感分析，以下是一些文本向量化方法：

+   在sklearn管道中使用逻辑回归的TF-IDF

+   斯坦福大学的GLoVe，通过Gensim查找

+   Facebook的fastText使用预训练的向量

我们已经看到了TF-IDF的示例，并且在这本书的其余部分还将看到更多。本章将介绍其他可以将您的文本语料库或其部分向量的方法。

在本章中，我们将学习以下主题：

+   如何向量化特定数据集

+   如何制作文档嵌入

# 向量化特定数据集

本节几乎完全专注于词向量以及我们如何利用Gensim库来执行它们。

我们在本节中想要回答的一些问题包括这些：

+   我们如何使用原始嵌入，如GloVe？

+   我们如何处理词汇表外的单词？（提示：fastText）

+   我们如何在我们自己的语料库上训练自己的word2vec向量？

+   我们如何训练我们自己的word2vec向量？

+   我们如何训练我们自己的fastText向量？

+   我们如何使用相似词来比较上述两者？

首先，让我们从一些简单的导入开始，如下所示：

```py
import gensim
print(f'gensim: {gensim.__version__}')
> gensim: 3.4.0
```

请确保您的Gensim版本至少为3.4.0。这是一个非常流行的包，主要由RaRe Technologies的文本处理专家维护和开发。他们在自己的企业B2B咨询工作中使用相同的库。Gensim的内部实现的大部分是用Cython编写的，以提高速度。它原生支持多进程。

这里需要注意的是，Gensim因其API的破坏性更改而闻名，因此在使用他们的文档或教程中的代码时，请考虑再次检查API。

如果你使用的是Windows机器，请注意以下类似的警告：

```py
C:\Users\nirantk\Anaconda3\envs\fastai\lib\site-packages\Gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
 warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
```

现在，让我们开始下载预训练的GloVe嵌入。虽然我们可以手动完成这项工作，但在这里我们将使用以下Python代码来下载：

```py
from tqdm import tqdm
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    import os
    from urllib.request import urlretrieve

    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)
```

我们还将重用`get_data` API来下载我们想要在本节中使用的任何任意文件。我们还设置了`tqdm`（阿拉伯语意为进度），它通过将我们的`urlretrieve`可迭代对象包装在其中，为我们提供了一个进度条。

以下文本来自tqdm的README：

tqdm在Linux、Windows、Mac、FreeBSD、NetBSD、Solaris/SunOS等任何平台上工作，在任何控制台或GUI中，并且对IPython/Jupyter笔记本也很友好。

tqdm不需要任何依赖（甚至不是curses！），只需要Python和一个支持回车符\r和换行符\n控制字符的环境。

对了，我们终于可以下载嵌入文件了，不是吗？

```py
embedding_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
get_data(embedding_url, 'data/glove.6B.zip')
```

前面的代码片段将下载一个包含60亿个英语单词的GloVe词表示的大型文件。

让我们快速使用终端或Jupyter笔记本中的命令行语法解压文件。您也可以手动或通过编写代码来完成此操作，如下所示：

```py
# # We need to run this only once, can unzip manually unzip to the data directory too
# !unzip data/glove.6B.zip 
# !mv glove.6B.300d.txt data/glove.6B.300d.txt 
# !mv glove.6B.200d.txt data/glove.6B.200d.txt 
# !mv glove.6B.100d.txt data/glove.6B.100d.txt 
# !mv glove.6B.50d.txt data/glove.6B.50d.txt
```

在这里，我们已经将所有的`.txt`文件移回到`data`目录。这里需要注意的是文件名，`glove.6B.50d.txt`。

`6B`代表60亿个单词或标记。`50d`代表50个维度，这意味着每个单词都由一个由50个数字组成的序列表示，在这种情况下，那就是50个浮点数。

我们现在将稍微偏离一下，给您一些关于词表示的背景信息。

# 词表示

词嵌入中最受欢迎的名字是谷歌的word2vec（Mikolov）和斯坦福大学的GloVe（Pennington、Socher和Manning）。fastText似乎在多语言子词嵌入中相当受欢迎。

我们建议您不要使用word2vec或GloVe。相反，使用fastText向量，它们要好得多，并且来自同一作者。word2vec是由T. Mikolov等人（[https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en)）在谷歌工作时引入的，它在单词相似性和类比任务上表现良好。

GloVe是由斯坦福大学的Pennington、Socher和Manning在2014年引入的，作为一种词嵌入的统计近似。词向量是通过词-词共现矩阵的矩阵分解来创建的。

如果在两个恶之间选择较小的那个，我们推荐使用GloVe而不是word2vec。这是因为GloVe在大多数机器学习任务和学术界的NLP挑战中优于word2vec。

在这里跳过原始的word2vec，我们现在将探讨以下主题：

+   我们如何使用GloVe中的原始嵌入？

+   我们如何处理词汇表外的单词？（提示：fastText）

+   我们如何在自己的语料库上训练自己的word2vec向量？

# 我们如何使用预训练的嵌入？

我们刚刚下载了这些。

word2vec和GloVe使用的文件格式略有不同。我们希望有一个一致的API来查找任何词嵌入，我们可以通过转换嵌入格式来实现这一点。请注意，在词嵌入的存储方式上存在一些细微的差异。

这种格式转换可以使用Gensim的API `glove2word2vec`来完成。我们将使用它将我们的GloVe嵌入信息转换为word2vec格式。

因此，让我们先处理导入，然后设置文件名，如下所示：

```py
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'data/glove.6B.300d.txt'
word2vec_output_file = 'data/glove.6B.300d.word2vec.txt'
```

如果我们已经进行过一次转换，我们不想重复这一步骤。最简单的方法是查看`word2vec_output_file`是否已经存在。只有在文件不存在的情况下，我们才运行以下转换：

```py
import os
if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)
```

前面的片段将在一个与Gensim API堆栈兼容的标准中创建一个新文件。

# KeyedVectors API

我们现在必须执行一个简单的任务，即从文件中加载向量。我们使用Gensim中的`KeyedVectors` API来完成这项工作。我们想要查找的单词是键，该单词的数值表示是相应的值。

让我们先导入API并设置目标文件名，如下所示：

```py
from gensim.models import KeyedVectors
filename = word2vec_output_file
```

我们将把整个文本文件加载到我们的内存中，包括从磁盘读取的时间。在大多数运行过程中，这是一个一次性I/O步骤，并且不会为每次新的数据传递而重复。这将成为我们的Gensim模型，详细说明如下：

```py
%%time
# load the Stanford GloVe model from file, this is Disk I/O and can be slow
pretrained_w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# binary=False format for human readable text (.txt) files, and binary=True for .bin files
```

更快的SSD应该可以显著提高速度，提高一个数量级。

我们可以进行一些单词向量算术运算来组合并展示这种表示不仅捕捉了语义意义。例如，让我们重复以下著名的单词向量示例：

```py
(king - man) + woman = ?
```

现在我们对单词向量执行所提到的算术运算，如下所示：

```py
# calculate: (king - man) + woman = ?
result = pretrained_w2v_model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
```

我们使用`most_similar` API完成了这项工作。在幕后，Gensim为我们做了以下操作：

1.  查询了`woman`、`king`和`man`的向量

1.  添加了`king`和`woman`，并从`man`中减去向量以找到结果向量

1.  在这个模型中的60亿个标记中，按距离对所有单词进行排序并找到了最接近的单词

1.  找到了最接近的单词

我们还添加了`topn=1`来告诉API我们只对最接近的匹配感兴趣。现在的预期输出只是一个单词，`'queen'`，如下面的片段所示：

```py
print(result)
> [('queen', 0.6713277101516724)]
```

我们不仅得到了正确的单词，还得到了一个伴随的十进制数！我们现在暂时忽略这个数字，但请注意，这个数字代表了一个概念，即单词与API为我们计算的结果向量的接近程度或相似程度。

让我们再试几个例子，比如社交网络，如下面的片段所示：

```py
result = pretrained_w2v_model.most_similar(positive=['quora', 'facebook'], negative=['linkedin'], topn=1)
print(result)
```

在这个例子中，我们正在寻找一个比LinkedIn更随意但比Facebook更专注于学习的社交网络，通过添加Quora来实现。正如您在以下输出中可以看到，Twitter似乎完美地符合这一要求：

```py
[('twitter', 0.37966805696487427)]
```

我们同样可以期待Reddit也能符合这一要求。

那么，我们能否使用这种方法简单地探索更大语料库中的相似单词？看起来是这样的。现在让我们查找与`india`最相似的单词，如下面的片段所示。请注意，我们用小写写印度；这是因为模型中只包含小写单词：

```py
pretrained_w2v_model.most_similar('india')
```

值得注意的是，这些结果可能有点偏颇，因为GloVe主要是基于一个名为Gigaword的大型新闻语料库进行训练的：

```py
[('indian', 0.7355823516845703),
 ('pakistan', 0.7285579442977905),
 ('delhi', 0.6846907138824463),
 ('bangladesh', 0.6203191876411438),
 ('lanka', 0.609517514705658),
 ('sri', 0.6011613607406616),
 ('kashmir', 0.5746493935585022),
 ('nepal', 0.5421023368835449),
 ('pradesh', 0.5405811071395874),
 ('maharashtra', 0.518537700176239)]
```

考虑到在外国媒体中，印度经常因其与地理邻国（包括巴基斯坦和克什米尔）的紧张关系而被提及，先前的结果确实是有意义的。孟加拉国、尼泊尔和斯里兰卡是邻国，而马哈拉施特拉邦是印度商业之都孟买的所在地。

# word2vec 和 GloVe 缺少了什么？

无论是 GloVe 还是 word2vec 都无法处理训练过程中没有见过的单词。这些单词在文献中被称为**词汇表外**（OOV）。

如果你尝试查找不常使用的名词，例如一个不常见的名字，就可以看到这种证据。如下面的代码片段所示，模型会抛出一个 `not in vocabulary` 错误：

```py
try: 
  pretrained_w2v_model.wv.most_similar('nirant')
except Exception as e: 
  print(e)  
```

这导致了以下输出：

```py
"word 'nirant' not in vocabulary"
```

这个结果还伴随着一个API警告，有时会声明API将在 gensim v4.0.0 中更改。

# 我们如何处理词汇表外的单词？

word2vec 的作者（Mikolov 等人）将其扩展到 Facebook 上的 fastText。它使用字符 n-gram 而不是整个单词。字符 n-gram 在具有特定形态学特性的语言中非常有效。

我们可以创建自己的 fastText 嵌入，它可以处理 OOV 标记。

# 获取数据集

首先，我们需要从公共数据集中下载几个 TED 演讲的字幕。我们将使用这些字幕以及 word2vec 嵌入进行对比训练，如下所示：

```py
ted_dataset = "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip"
get_data(ted_dataset, "data/ted_en.zip")
```

Python 允许我们访问 `.zip` 文件内部的文件，使用 `zipfile` 包很容易做到这一点。注意，是 `zipfile.zipFile` 语法使得这一点成为可能。

我们还使用 `lxml` 包来解析 ZIP 文件内的 XML 文件。

在这里，我们手动打开文件以找到相关的 `content` 路径，并从中查找 `text()`。在这种情况下，我们只对字幕感兴趣，而不是任何伴随的元数据，如下所示：

```py
import zipfile
import lxml.etree
with zipfile.ZipFile('data/ted_en.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))

```

现在我们预览一下以下 `input_text` 的前500个字符：

```py
input_text[:500]
> "Here are two reasons companies fail: they only do more of the same, or they only do what's new.\nTo me the real, real solution to quality growth is figuring out the balance between two activities: exploration and exploitation. Both are necessary, but it can be too much of a good thing.\nConsider Facit. I'm actually old enough to remember them. Facit was a fantastic company. They were born deep in the Swedish forest, and they made the best mechanical calculators in the world. Everybody used them. A"
```

由于我们使用的是 TED 演讲中的字幕，有一些填充词是没有用的。这些通常是括号中描述声音的单词和演讲者的名字。

让我们使用一些正则表达式删除这些填充词，如下所示：

```py
import re
# remove parenthesis 
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)

# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)
```

注意，我们使用 `.split('\n')` 语法在我们的整个语料库上创建了 `sentence_strings_ted`。作为读者练习，将其替换为更好的句子分词器，例如 spaCy 或 NLTK 中的分词器：

```py
print(sentences_ted[:2])
```

注意，每个 `sentences_ted` 现在都是一个列表的列表。第一个列表的每个元素都是一个句子，每个句子是一个标记（单词）的列表。

这是使用 Gensim 训练文本嵌入的预期结构。我们将把以下代码写入磁盘，以便稍后检索：

```py
import json
# with open('ted_clean_sentences.json', 'w') as fp:
#     json.dump(sentences_ted, fp)

with open('ted_clean_sentences.json', 'r') as fp:
    sentences_ted = json.load(fp)

```

我个人更喜欢 JSON 序列化而不是 Pickle，因为它稍微快一点，跨语言互操作性更强，最重要的是，它对人类来说是可读的。

现在让我们在这个小语料库上训练 fastText 和 word2vec 嵌入。虽然语料库很小，但我们使用的语料库代表了实践中通常看到的数据大小。在行业中，大型标注文本语料库极其罕见。

# 训练 fastText 嵌入

在新的 Gensim API 中设置导入实际上非常简单；只需使用以下代码：

```py
from gensim.models.fasttext import FastText
```

下一步是输入文本并构建我们的文本嵌入模型，如下所示：

```py
fasttext_ted_model = FastText(sentences_ted, size=100, window=5, min_count=5, workers=-1, sg=1)
 # sg = 1 denotes skipgram, else CBOW is used
```

您可能会注意到我们传递给构建模型的参数。以下列表解释了这些参数，正如 Gensim 文档中所述：

+   `min_count (int, 可选)`: 模型忽略所有总频率低于此值的单词

+   `size (int, 可选)`: 这表示单词向量的维度

+   `window (int, 可选)`: 这表示句子中当前单词和预测单词之间的最大距离

+   `workers (int, 可选)`: 使用这些工作线程来训练模型（这可以在多核机器上实现更快的训练；`workers=-1` 表示使用机器中每个可用的核心的一个工作线程）

+   `sg ({1, 0}, 可选)`: 这是一个训练算法，当 `sg=1` 时为 `skip-gram` 或 CBOW

前面的参数实际上是更大列表的一部分，可以通过调整这些参数来改善文本嵌入的质量。我们鼓励您在探索 Gensim API 提供的其他参数的同时，尝试调整这些数字。

现在让我们快速查看这个语料库中按 fastText 嵌入相似度排名的与印度最相似的单词，如下所示：

```py
fasttext_ted_model.wv.most_similar("india")

[('indians', 0.5911639928817749),
 ('indian', 0.5406097769737244),
 ('indiana', 0.4898717999458313),
 ('indicated', 0.4400438070297241),
 ('indicate', 0.4042605757713318),
 ('internal', 0.39166826009750366),
 ('interior', 0.3871103823184967),
 ('byproducts', 0.3752930164337158),
 ('princesses', 0.37265270948410034),
 ('indications', 0.369659960269928)]
```

在这里，我们注意到 fastText 利用子词结构，例如 `ind`、`ian` 和 `dian` 来对单词进行排序。我们在前 3 名中得到了 `indians` 和 `indian`，这相当不错。这是 fastText 有效的原因之一——即使是对于小型的训练文本任务。

现在让我们重复使用 word2vec 进行相同的过程，并查看那里与 `india` 最相似的单词。

# 训练 word2vec 嵌入

导入模型很简单，只需使用以下命令。到现在为止，您应该对 Gensim 模型 API 的结构有了直观的了解：

```py
from gensim.models.word2vec import Word2Vec
```

在这里，我们使用与 fastText 相同的配置来构建 word2vec 模型。这有助于减少比较中的偏差。

鼓励您使用以下方法比较最佳 fastText 模型和最佳 word2vec 模型：

```py
word2vec_ted_model = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=-1, sg=1)
```

对了，现在让我们查看与 `india` 最相似的单词，如下所示：

```py
word2vec_ted_model.wv.most_similar("india")

[('cent', 0.38214215636253357),
 ('dichotomy', 0.37258434295654297),
 ('executing', 0.3550642132759094),
 ('capabilities', 0.3549191951751709),
 ('enormity', 0.3421599268913269),
 ('abbott', 0.34020164608955383),
 ('resented', 0.33033430576324463),
 ('egypt', 0.32998529076576233),
 ('reagan', 0.32638251781463623),
 ('squeezing', 0.32618749141693115)]
```

与 `india` 最相似的单词与原始单词没有实质性的关系。对于这个特定的数据集和 word2vec 的训练配置，模型根本没有捕捉到任何语义或句法信息。这在 word2vec 旨在处理大型文本语料库的情况下并不罕见。

# fastText 与 word2vec 的比较

根据以下 Gensim 的初步比较：

<q>fastText在编码句法信息方面显著优于word2vec。这是预料之中的，因为大多数句法类比都是基于形态学的，而fastText的字符n-gram方法考虑了这种信息。原始的word2vec模型在语义任务上似乎表现更好，因为语义类比中的单词与它们的字符n-gram无关，而无关字符n-gram添加的信息反而恶化了嵌入。</q>

此资料的来源是：*word2vec fasttext比较笔记本* ([https://github.com/RaRe-Technologies/gensim/blob/37e49971efa74310b300468a5b3cf531319c6536/docs/notebooks/Word2Vec_FastText_Comparison.ipynb](https://github.com/RaRe-Technologies/gensim/blob/37e49971efa74310b300468a5b3cf531319c6536/docs/notebooks/Word2Vec_FastText_Comparison.ipynb))。

通常，我们更喜欢fastText，因为它天生具有处理训练中未见过的单词的能力。当处理小数据（如我们所展示的）时，它肯定优于word2vec，并且在大型数据集上至少与word2vec一样好。

fastText在处理充满拼写错误的文本时也非常有用。例如，它可以利用子词相似性在嵌入空间中将`indian`和`indain`拉近。

在大多数下游任务中，如情感分析或文本分类，我们继续推荐GloVe优于word2vec。

以下是我们为文本嵌入应用推荐的经验法则：fastText > GloVe > word2vec。

# 文档嵌入

文档嵌入通常被认为是一种被低估的方法。文档嵌入的关键思想是将整个文档，例如专利或客户评论，压缩成一个单一的向量。这个向量随后可以用于许多下游任务。

实验结果表明，文档向量优于词袋模型以及其他文本表示技术。

其中最有用的下游任务之一是能够聚类文本。文本聚类有几种用途，从数据探索到在线分类管道中传入的文本。

尤其是我们对在小型数据集上使用doc2vec进行文档建模感兴趣。与捕捉在生成句子向量中的单词序列的序列模型（如RNN）不同，doc2vec句子向量与单词顺序无关。这种单词顺序无关性意味着我们可以快速处理大量示例，但这确实意味着捕捉到句子固有的意义较少。

本节大致基于Gensim存储库中的doc2Vec API教程。

让我们先通过以下代码处理掉导入：

```py
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
from pprint import pprint
import multiprocessing
```

现在，让我们按照以下方式从我们之前使用的`doc`变量中提取谈话：

```py
talks = doc.xpath('//content/text()')
```

要训练Doc2Vec模型，每个文本样本都需要一个标签或唯一标识符。为此，可以编写一个如下的小函数：

```py
def read_corpus(talks, tokens_only=False):
    for i, line in enumerate(talks):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
```

在前面的函数中发生了一些事情；具体如下：

+   **重载if条件**：这个函数读取测试语料库并将`tokens_only`设置为`True`。

+   **目标标签**：这个函数分配一个任意的索引变量`i`作为目标标签。

+   `**gensim.utils.simple_preprocess**`：这个函数将文档转换为一系列小写标记，忽略过短或过长的标记，然后产生`TaggedDocument`实例。由于我们产生而不是返回，这个整个函数作为一个生成器在运行。

值得注意的是，这如何改变函数的行为。使用`return`时，当函数被调用时，它会返回一个特定的对象，例如`TaggedDocument`或如果没有指定返回，则返回`None`。另一方面，生成器函数只返回一个`generator`对象。

那么，你期望以下代码行返回什么？

```py
read_corpus(talks)
```

如果你猜对了，你就会知道我们期望它返回一个`generator`对象，如下所示：

```py
<generator object read_corpus at 0x0000024741DBA990>
```

前面的对象意味着我们可以按需逐个读取文本语料库的元素。如果训练语料库的大小超过你的内存大小，这特别有用。

理解Python迭代器和生成器的工作方式。它们使你的代码内存效率更高，更容易阅读。

在这个特定的例子中，我们有一个相当小的训练语料库作为示例，所以让我们将整个语料库作为一个`TaggedDocument`对象的列表读入工作内存，如下所示：

```py
ted_talk_docs = list(read_corpus(talks))
```

`list()`语句遍历整个语料库，直到函数停止产生。我们的变量`ted_talk_docs`应该看起来像以下这样：

```py
ted_talk_docs[0]

TaggedDocument(words=['here', 'are', 'two', 'reasons', 'companies', 'fail', ...., 'you', 'already', 'know', 'don', 'forget', 'the', 'beauty', 'is', 'in', 'the', 'balance', 'thank', 'you', 'applause'], tags=[0])
```

让我们快速看一下这台机器有多少个核心。我们将使用以下代码初始化doc2vec模型：

```py
cores = multiprocessing.cpu_count()
print(cores)
8
```

现在我们去初始化我们的doc2vec模型，来自Gensim。

# 理解doc2vec API

```py
model = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, iter=5, workers=cores)
```

让我们快速了解前面代码中使用的标志：

+   `dm ({1,0}, optional)`：这定义了训练算法；如果`dm=1`，则使用*分布式内存*（PV-DM）；否则，使用分布式词袋（PV-DBOW）。

+   `size (int, optional)`：这是特征向量的维度

+   `window (int, optional)`：这代表句子中当前词和预测词之间的最大距离

+   `negative (int, optional)`：如果大于`0`，则使用负采样（负值的int指定应该抽取多少*噪声词*，这通常在5-20之间）；如果设置为`0`，则不使用负采样。

+   `hs ({1,0}, optional)`：如果设置为`1`，则用于模型训练的层次softmax；如果设置为0且负采样非零，则使用负采样。

+   `iter (int, optional)`：这代表在语料库上的迭代次数（周期）

前面的列表直接来自Gensim文档。考虑到这一点，我们现在将解释这里引入的一些新术语，包括负采样和层次softmax。

# 负采样

负采样最初是一种为了加速训练而采用的技巧，现在已经成为一种被广泛接受的实践。这里的要点是，除了在可能正确的答案上训练你的模型之外，为什么不给它展示一些错误的例子？

特别是，使用负采样通过减少所需的模型更新次数来加速训练。我们不是为每个错误的单词更新模型，而是选择一个较小的数字，通常在5到25之间，并在它们上训练模型。因此，我们将从在大语料库上训练所需的几百万次更新减少到一个更小的数字。这是一个经典的软件编程技巧，在学术界也有效。

# 层次化软最大化

我们通常的softmax中的分母项是通过在大量单词上的求和操作来计算的。这种归一化操作在训练过程中的每次更新都是一个昂贵的操作。

相反，我们可以将其分解为一系列特定的计算，这样可以节省我们计算所有单词上的昂贵归一化的时间。这意味着对于每个单词，我们使用某种近似。

在实践中，这种近似已经非常有效，以至于一些系统在训练和推理时间都使用这种方法。对于训练来说，它可以提供高达50倍的速度（据NLP研究博主Sebastian Ruder所说）。在我的实验中，我看到了大约15-25倍的速度提升。

```py
model.build_vocab(ted_talk_docs)
```

训练doc2vec模型的API略有不同。我们首先使用`build_vocab` API从一系列句子中构建词汇表，如前一个代码片段所示。我们还将我们的内存变量`ted_talk_docs`传递到这里，但也可以传递从`read_corpora`函数中获得的单次生成器流。

现在我们设置一些以下样本句子，以找出我们的模型是否学到了什么：

```py
sentence_1 = 'Modern medicine has changed the way we think about healthcare, life spans and by extension career and marriage'

sentence_2 = 'Modern medicine is not just a boon to the rich, making the raw chemicals behind these is also pollutes the poorest neighborhoods'

sentence_3 = 'Modern medicine has changed the way we think about healthcare, and increased life spans, delaying weddings'
```

Gensim有一个有趣的API，允许我们使用我们刚刚用词汇表更新的模型在两个未见文档之间找到相似度值，如下所示：

```py
model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_3.split())
> -0.18353473068679

model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_2.split())
> -0.08177642293252027
```

前面的输出并不完全合理，对吧？我们写的句子应该有一些合理的相似度，这绝对不是负面的。

哎呀！我们忘记在语料库上训练模型了。现在让我们用以下代码来做这件事，然后重复之前的比较，看看它们是如何变化的：

```py
%time model.train(ted_talk_docs, total_examples=model.corpus_count, epochs=model.epochs)
Wall time: 6.61 s
```

在BLAS设置好的机器上，这一步应该不到几秒钟。

我们实际上可以根据以下模型提取任何特定句子的原始推理向量：

```py
model.infer_vector(sentence_1.split())

array([-0.03805782,  0.09805363, -0.07234333,  0.31308332,  0.09668373,
       -0.01471598, -0.16677614, -0.08661497, -0.20852503, -0.14948   ,
       -0.20959479,  0.17605443,  0.15131783, -0.17354141, -0.20173495,
        0.11115499,  0.38531387, -0.39101505,  0.12799   ,  0.0808568 ,
        0.2573657 ,  0.06932276,  0.00427534, -0.26196653,  0.23503092,
        0.07589306, -0.01828301,  0.38289976, -0.04719075, -0.19283117,
        0.1305226 , -0.1426582 , -0.05023642, -0.11381021,  0.04444459,
       -0.04242943,  0.08780348,  0.02872207, -0.23920575,  0.00984556,
        0.0620702 , -0.07004016,  0.15629964,  0.0664391 ,  0.10215732,
        0.19148728, -0.02945088,  0.00786009, -0.05731675, -0.16740018,
       -0.1270729 ,  0.10185472,  0.16655563,  0.13184668,  0.18476236,
       -0.27073956, -0.04078012, -0.12580603,  0.02078131,  0.23821649,
        0.09743162, -0.1095973 , -0.22433399, -0.00453655,  0.29851952,
       -0.21170728,  0.1928157 , -0.06223159, -0.044757  ,  0.02430432,
        0.22560015, -0.06163954,  0.09602281,  0.09183675, -0.0035969 ,
        0.13212039,  0.03829316,  0.02570504, -0.10459486,  0.07317936,
        0.08702451, -0.11364868, -0.1518436 ,  0.04545208,  0.0309107 ,
       -0.02958601,  0.08201223,  0.26910907, -0.19102073,  0.00368607,
       -0.02754402,  0.3168101 , -0.00713515, -0.03267708, -0.03792975,
        0.06958092, -0.03290432,  0.03928463, -0.10203536,  0.01584929],
      dtype=float32)
```

在这里，`infer_vector` API期望一个标记列表作为输入。这应该解释了为什么我们也可以在这里使用`read_corpora`并设置`tokens_only = True`。

既然我们的模型已经训练好了，让我们再次比较以下句子：

```py
model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_3.split())
0.9010817740272721

model.docvecs.similarity_unseen_docs(model, sentence_1.split(), sentence_2.split())
0.7461058869759862
```

新的前置输出是有意义的。第一句和第三句确实比第一句和第二句更相似。在探索的精神下，现在让我们看看第二句和第三句的相似度，如下所示：

```py
model.docvecs.similarity_unseen_docs(model, sentence_2.split(), sentence_3.split())
0.8189999598358203
```

啊，这样更好。我们的结果现在与我们的预期一致。相似度值大于第一句和第二句，但小于第一句和第三句，它们在意图上几乎相同。

作为一种轶事观察或启发式方法，真正相似的句子在相似度量表上的值大于0.8。

我们已经提到，文档或文本向量通常是一种探索数据语料库的好方法。接下来，我们将以非常浅显的方式探索我们的语料库，然后给你一些继续探索的想法。

# 数据探索和模型评估

评估任何向量化的简单技术是将训练语料库作为测试语料库。当然，我们预计我们的模型会对训练集过度拟合，但这没关系。

我们可以通过以下方式使用训练语料库作为测试语料库：

+   为每份文档学习新的结果或*推理*向量

+   将向量与所有示例进行比较

+   根据相似度分数对文档、句子和段落向量进行排序

让我们用以下代码来做这件事：

```py
ranks = []
for idx in range(len(ted_talk_docs)):
    inferred_vector = model.infer_vector(ted_talk_docs[idx].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(idx)
    ranks.append(rank)
```

我们现在已经弄清楚每个文档在排名中的位置。所以，如果最高排名是文档本身，那就足够了。正如我们所说，我们可能在训练语料库上略微过度拟合，但这仍然是一个很好的合理性测试。我们可以通过以下方式使用`Counter`进行频率计数：

```py
import collections
collections.Counter(ranks) # Results vary due to random seeding + very small corpus
Counter({0: 2079, 1: 2, 4: 1, 5: 2, 2: 1})
```

`Counter`对象告诉我们有多少文档发现自己处于什么排名。所以，2079份文档发现自己排名第一（索引0），但有两份文档分别发现自己排名第二（索引1）和第六（索引5）。有一份文档排名第五（索引4）和第三（索引2）。总的来说，这是一个非常好的训练性能，因为2084份文档中有2079份将自己排名为第一。

这有助于我们理解向量确实以有意义的方式在文档中代表了信息。如果它们没有这样做，我们会看到更多的排名分散。

现在我们快速取一份文档，找到与之最相似、最不相似以及介于两者之间的文档。以下代码可以完成这个任务：

```py
doc_slice = ' '.join(ted_talk_docs[idx].words)[:500]
print(f'Document ({idx}): «{doc_slice}»\n')
print(f'SIMILAR/DISSIMILAR DOCS PER MODEL {model}')
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
      doc_slice = ' '.join(ted_talk_docs[sims[index][0]].words)[:500]
      print(f'{label} {sims[index]}: «{doc_slice}»\n')

```

注意我们是如何选择预览整个文档的一部分以供探索的。你可以自由选择这样做，或者使用一个小型文本摘要工具来动态创建你的预览。

结果如下：

```py
Document (2084): «if you re here today and very happy that you are you've all heard about how sustainable development will save us from ourselves however when we're not at ted we're often told that real sustainability policy agenda is just not feasible especially in large urban areas like new york city and that because most people with decision making powers in both the public and the private sector really don't feel as though they are in danger the reason why here today in part is because of dog an abandoned puppy»

SIMILAR/DISSIMILAR DOCS PER MODEL Doc2Vec(dbow,d100,n5,mc2,s0.001,t8)
 MOST (2084, 0.893369197845459): «if you are here today and very happy that you are you've all heard about how sustainable development will save us from ourselves however when we are not at ted we are often told that real sustainability policy agenda is just not feasible especially in large urban areas like new york city and that because most people with decision making powers in both the public and the private sector really don feel as though they re in danger the reason why here today in part is because of dog an abandoned puppy»

MEDIAN (1823, 0.42069244384765625): «so going to talk today about collecting stories in some unconventional ways this is picture of me from very awkward stage in my life you might enjoy the awkwardly tight cut off pajama bottoms with balloons anyway it was time when was mainly interested in collecting imaginary stories so this is picture of me holding one of the first watercolor paintings ever made and recently I've been much more interested in collecting stories from reality so real stories and specifically interested in collecting »

LEAST (270, 0.12334088981151581): «on june precisely at in balmy winter afternoon in so paulo brazil typical south american winter afternoon this kid this young man that you see celebrating here like he had scored goal juliano pinto years old accomplished magnificent deed despite being paralyzed and not having any sensation from mid chest to the tip of his toes as the result of car crash six years ago that killed his brother and produced complete spinal cord lesion that left juliano in wheelchair juliano rose to the occasion and»
```

# 摘要

这章不仅仅是Gensim API的介绍。我们现在知道如何加载预训练的GloVe向量，你可以在任何机器学习模型中使用这些向量表示，而不是TD-IDF。

我们探讨了为什么fastText向量在小型训练语料库上通常比word2vec向量更好，并了解到你可以将它们用于任何ML模型。

我们学习了如何构建doc2vec模型。现在，你可以将这种doc2vec方法扩展到构建sent2vec或paragraph2vec风格的模型。理想情况下，paragraph2vec将会改变，仅仅是因为每个文档将变成一个段落。

此外，我们现在知道如何在不使用标注测试语料库的情况下快速对doc2vec向量进行合理性检查。我们是通过检查排名分散度指标来做到这一点的。
