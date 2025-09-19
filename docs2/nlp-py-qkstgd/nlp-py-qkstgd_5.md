# 分类现代方法

现在我们知道如何将文本字符串转换为捕获一些意义的数值向量。在本章中，我们将探讨如何使用这些向量与嵌入结合。嵌入是比词向量更常用的术语，也是数值表示。

在本章中，我们仍然遵循第一章节的总体框架，即文本→表示→模型→评估→部署。

我们将继续使用文本分类作为我们的示例任务。这主要是因为它是一个简单的演示任务，但我们也可以将本书中的几乎所有想法扩展到解决其他问题。然而，接下来的主要焦点是文本分类的机器学习。

总结来说，在本章中，我们将探讨以下主题：

+   情感分析作为文本分类的一个特定类别和示例

+   简单分类器和如何优化它们以适应你的数据集

+   集成方法

# 文本机器学习

在社区中至少有10到20种广为人知的机器学习技术，从SVM到多种回归和梯度提升机。我们将从中选择一小部分进行尝试。

![图片](img/290de774-4008-41d8-91dd-3e7027b94f3b.png)

来源：[https://www.kaggle.com/surveys/2017.](https://www.kaggle.com/surveys/2017)

上述图表显示了Kagglers使用最广泛的机器学习技术。

在处理20个新闻组数据集时，我们遇到了**逻辑回归**。我们将重新审视**逻辑回归**并介绍**朴素贝叶斯**、**支持向量机**、**决策树**、**随机森林**和**XgBoost**。**XgBoost**是一种流行的算法，被多位Kaggle获奖者用于获得获奖结果。我们将使用Python中的scikit-learn和XGBoost包来查看之前的代码示例。

# 情感分析作为文本分类

分类器的一个流行用途是情感分析。这里的最终目标是确定文本文档的主观价值，这本质上是指文本文档的内容是积极的还是消极的。这对于快速了解你正在制作的影片或你想要阅读的书籍的语气特别有用。

# 简单分类器

让我们从简单地尝试几个机器学习分类器开始，如逻辑回归、朴素贝叶斯和决策树。然后我们将尝试随机森林和额外树分类器。对于所有这些实现，我们不会使用除scikit-learn以外的任何东西。

# 优化简单分类器

我们可以通过尝试几个略微不同的分类器版本来调整这些简单的分类器以改善它们的性能。为此，最常见的方法是尝试改变分类器的参数。

我们将学习如何使用`GridSearch`和`RandomizedSearch`来自动化这个搜索过程以找到最佳分类器参数。

# 集成方法

拥有一系列不同的分类器意味着我们将使用一组模型。集成是一种非常流行且易于理解的机器学习技术，几乎是每个获胜的Kaggle竞赛的一部分。

尽管最初担心这个过程可能很慢，但一些在商业软件上工作的团队已经开始在生产软件中使用集成方法。这是因为它需要很少的开销，易于并行化，并且允许内置的单个模型回退。

我们将研究一些基于简单多数的简单集成技术，也称为投票集成，然后基于此构建。

总结来说，本节机器学习NLP涵盖了简单的分类器、参数优化和集成方法。

# 获取数据

我们将使用Python的标准内置工具`urlretrieve`从`urllib.request`编程下载数据。以下是从互联网下载的部分：

```py
from pathlib import Path
import pandas as pd
import gzip
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import numpy as np

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)
```

如果你使用的是fastAI环境，所有这些导入都有效。第二个块只是为我们设置Tqdm，以便可视化下载进度。现在让我们使用`urlretrieve`下载数据，如下所示：

```py
def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)
```

让我们下载一些数据，如下所示：

```py
data_url = 'http://files.fast.ai/data/aclImdb.tgz'
get_data(data_url, 'data/imdb.tgz')
```

现在我们提取前面的文件，看看目录包含什么：

```py
data_path = Path(os.getcwd())/'data'/'imdb'/'aclImdb'
assert data_path.exists()
for pathroute in os.walk(data_path):
    next_path = pathroute[1]
    for stop in next_path:
        print(stop)
```

注意，我们更喜欢使用`Path from pathlib`而不是`os.path`功能。这使得它更加平台无关，也更加Pythonic。这个写得非常糟糕的实用工具告诉我们至少有两个文件夹：`train`和`test`。每个文件夹反过来又至少有三个文件夹，如下所示：

```py
Test
 |- all
 |- neg
 |- pos

 Train
 |- all
 |- neg
 |- pos
 |- unsup
```

`pos`和`neg`文件夹包含评论，分别代表正面和负面。`unsup`文件夹代表无监督。这些文件夹对于构建语言模型很有用，特别是对于深度学习，但在这里我们不会使用。同样，`all`文件夹是多余的，因为这些评论在`pos`或`neg`文件夹中已经重复。

# 读取数据

让我们将以下数据读入一个带有适当标签的Pandas `DataFrame`中：

```py
train_path = data_path/'train'
test_path = data_path/'test'

def read_data(dir_path):
    """read data into pandas dataframe"""

    def load_dir_reviews(reviews_path):
        files_list = list(reviews_path.iterdir())
        reviews = []
        for filename in files_list:
            f = open(filename, 'r', encoding='utf-8')
            reviews.append(f.read())
        return pd.DataFrame({'text':reviews})

    pos_path = dir_path/'pos'
    neg_path = dir_path/'neg'

    pos_reviews, neg_reviews = load_dir_reviews(pos_path), load_dir_reviews(neg_path)

    pos_reviews['label'] = 1
    neg_reviews['label'] = 0

    merged = pd.concat([pos_reviews, neg_reviews])
    merged.reset_index(inplace=True)

    return merged
```

此函数读取特定`train`或`test`分割的文件，包括正负样本，对于IMDb数据集。每个分割都是一个包含两列的`DataFrame`：`text`和`label`。`label`列给出了我们的目标值，或`y`，如下所示：

```py
train = read_data(train_path)
test = read_data(test_path)

X_train, y_train = train['text'], train['label']
X_test, y_test = test['text'], test['label']
```

我们现在可以读取相应的`DataFrame`中的数据，然后将其拆分为以下四个变量：`X_train`、`y_train`、`X_test`和`y_test`。

# 简单分类器

为了尝试一些我们的分类器，让我们先导入基本库，如下所示。在这里，我们将根据需要导入其余的分类器。这种在稍后导入事物的能力对于确保我们不会将太多不必要的组件导入内存非常重要：

```py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
```

由于本节仅用于说明目的，我们将使用最简单的特征提取步骤，如下所示：

+   词袋模型

+   TF-IDF

我们鼓励您尝试使用更好的文本向量化（例如，使用直接GloVe或word2vec查找）的代码示例。

# 逻辑回归

现在我们简单地复制我们在[第1章](5625152b-6870-44b1-a39f-5a79bcc675d9.xhtml)“开始文本分类”中做的简单逻辑回归，但是在我们的自定义数据集上，如下所示：

```py
from sklearn.linear_model import LogisticRegression as LR
lr_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',LR())])
```

如您在前面代码片段中看到的，`lr_clf`成为我们的分类器管道。我们在介绍部分看到了管道。管道允许我们在一个单一的Python对象中排队多个操作。

我们能够调用`fit`、`predict`和`fit_transform`等函数在我们的`Pipeline`对象上，因为管道会自动调用列表中最后一个组件的相应函数。

```py
lr_clf.fit(X=X_train, y=y_train) # note that .fit function calls are inplace, and the Pipeline is not re-assigned
```

如前所述，我们正在调用管道上的`predict`函数。测试评论将经过与训练期间相同的预处理步骤，即`CountVectorizer()`和`TfidfTransformer()`，如下面的代码片段所示：

```py
lr_predicted = lr_clf.predict(X_test)
```

这个过程的简便性和简单性使得`Pipeline`成为软件级机器学习中最常用的抽象之一。然而，用户可能更喜欢独立执行每个步骤，或者在研究或实验用例中构建自己的管道等效物：

```py
lr_acc = sum(lr_predicted == y_test)/len(lr_predicted)
lr_acc # 0.88316
```

我们如何找到我们的模型准确率？好吧，让我们快速看一下前面一行发生了什么。

假设我们的预测是[1, 1, 1]，而真实值是[1, 0, 1]。等式将返回一个简单的布尔对象列表，例如`[True, False, True]`。当我们对Python中的布尔列表求和时，它返回`True`案例的数量，这给我们提供了模型正确预测的确切次数。

将此值除以所做预测的总数（或者，同样地，测试评论的数量）可以得到我们的准确率。

让我们将之前的两行逻辑写入一个简单、轻量级的函数来计算准确率，如下面的代码片段所示。这将防止我们重复逻辑：

```py
def imdb_acc(pipeline_clf):
    predictions = pipeline_clf.predict(X_test)
    assert len(y_test) == len(predictions)
    return sum(predictions == y_test)/len(y_test), predictions
```

# 移除停用词

通过简单地传递一个标志到`CountVectorizer`步骤，我们可以移除最常见的停用词。我们将指定要移除的停用词所写的语言。在以下情况下，那是`english`：

```py
lr_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)
lr_acc, lr_predictions = imdb_acc(lr_clf)
lr_acc # 0.879
```

如您所见，这并没有在提高我们的准确率方面起到很大作用。这表明分类器本身正在移除或忽略由停用词添加的噪声。

# 增加ngram范围

现在我们尝试通过包括二元组和三元组来改进分类器可用的信息，如下所示：

```py
lr_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',LR())])
lr_clf.fit(X=X_train, y=y_train)
lr_acc, lr_predictions = imdb_acc(lr_clf)
lr_acc # 0.86596
```

# 多项式朴素贝叶斯

让我们将与我们的逻辑回归分类器相同的方式初始化分类器，如下所示：

```py
from sklearn.naive_bayes import MultinomialNB as MNB
mnb_clf = Pipeline([('vect', CountVectorizer()), ('clf',MNB())])
```

之前的命令将测量以下方面的性能：

```py
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc # 0.81356
```

# 添加TF-IDF

现在，让我们尝试在单词袋（单语元）之后作为另一个步骤使用TF-IDF的先前模型，如下所示：

```py
mnb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc # 0.82956
```

这比我们之前的价值要好，但让我们看看我们还能做些什么来进一步提高这个值。

# 移除停用词

现在我们再次通过将`english`传递给分词器来移除英语中的停用词，如下所示：

```py
mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc # 0.82992 
```

这有助于提高性能，但只是略微提高。我们可能不如简单地保留其他分类器中的停用词。

作为最后的手动实验，让我们尝试添加二元组和一元组，就像我们在逻辑回归中做的那样，如下所示：

```py
mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',MNB())])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc # 0.8572
```

这比之前的多项式朴素贝叶斯性能要好得多，但不如我们的逻辑回归分类器表现好，后者接近88%的准确率。

现在我们尝试一些针对贝叶斯分类器的特定方法。

# 将fit prior改为false

增加`ngram_range`对我们有所帮助，但将`prior`从`uniform`改为拟合（通过将`fit_prior`改为`False`）并没有起到任何作用，如下所示：

```py
mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1,3))), ('tfidf', TfidfTransformer()), ('clf',MNB(fit_prior=False))])
mnb_clf.fit(X=X_train, y=y_train)
mnb_acc, mnb_predictions = imdb_acc(mnb_clf)
mnb_acc # 0.8572
```

我们已经考虑了所有可能提高我们性能的组合。请注意，这种方法很繁琐，而且容易出错，因为它过于依赖人类的直觉。

# 支持向量机

**支持向量机**（**SVM**）继续保持着一种非常受欢迎的机器学习技术，它从工业界进入课堂，然后再回到工业界。除了多种形式的回归之外，SVM是构成数十亿美元在线广告定位产业支柱的技术之一。

在学术界，T Joachim的工作（[https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf](https://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf)）建议使用支持向量分类器进行文本分类。

基于这样的文献，很难估计它对我们是否同样有效，主要是因为数据集和预处理步骤的不同。尽管如此，我们还是试一试：

```py
from sklearn.svm import SVC
svc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',SVC())])
svc_clf.fit(X=X_train, y=y_train)
svc_acc, svc_predictions = imdb_acc(svc_clf)
print(svc_acc) # 0.6562
```

虽然 SVM最适合线性可分的数据（正如我们所见，我们的文本不是线性可分的），但为了完整性，仍然值得一试。

在上一个例子中，SVM表现不佳，并且与其他分类器相比，训练时间也非常长（约150倍）。我们将不再针对这个特定数据集查看SVM。

# 决策树

决策树是分类和回归的简单直观工具。当从视觉上看时，它们常常类似于决策流程图，因此得名决策树。我们将重用我们的管道，简单地使用`DecisionTreeClassifier`作为我们的主要分类技术，如下所示：

```py
from sklearn.tree import DecisionTreeClassifier as DTC
dtc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',DTC())])
dtc_clf.fit(X=X_train, y=y_train)
dtc_acc, dtc_predictions = imdb_acc(dtc_clf)
dtc_acc # 0.7028
```

# 随机森林分类器

现在我们尝试第一个集成分类器。随机森林分类器中的“森林”来源于这个分类器的每个实例都由多个决策树组成。随机森林中的“随机”来源于每个树从所有特征中随机选择有限数量的特征，如下面的代码所示：

```py
from sklearn.ensemble import RandomForestClassifier as RFC
rfc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',RFC())])
rfc_clf.fit(X=X_train, y=y_train)
rfc_acc, rfc_predictions = imdb_acc(rfc_clf)
rfc_acc # 0.7226
```

虽然在大多数机器学习任务中使用时被认为非常强大，但随机森林方法在我们的案例中表现不佳。这部分原因在于我们相当粗糙的特征提取。

决策树、随机森林（RFC）和额外的树分类器在文本等高维空间中表现不佳。

# 额外的树分类器

“额外的树”中的“额外”来源于其极端随机化的想法。虽然随机森林分类器中的树分割是有效确定性的，但在额外的树分类器中是随机的。这改变了高维数据（如我们这里的每个单词都是一个维度或分类器）的偏差-方差权衡。以下代码片段显示了分类器的实际应用：

```py
from sklearn.ensemble import ExtraTreesClassifier as XTC
xtc_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',XTC())])
xtc_clf.fit(X=X_train, y=y_train)
xtc_acc, xtc_predictions = imdb_acc(xtc_clf)
xtc_acc # 0.75024
```

如您所见，这种变化在这里对我们有利，但这并不总是普遍适用的。结果会因数据集以及特征提取管道的不同而有所变化。

# 优化我们的分类器

让我们现在专注于我们表现最好的模型——逻辑回归，看看我们是否能将其性能提升一点。我们基于LR的模型的最佳性能是之前看到的0.88312的准确率。

我们在这里将短语“参数搜索”和“超参数搜索”互换使用。这样做是为了保持与深度学习词汇的一致性。

我们希望选择我们管道的最佳性能配置。每个配置可能在某些小方面有所不同，例如当我们移除停用词、二元词和三元词，或类似的过程时。这样的配置总数可能相当大，有时可能达到数千。除了手动选择一些组合进行尝试外，我们还可以尝试所有这些数千种组合并评估它们。

当然，这个过程对于我们这样的小规模实验来说将过于耗时。在大规模实验中，可能的空间可以达到数百万，需要几天的时间进行计算，这使得成本和时间变得难以承受。

我们建议阅读一篇关于超参数调整的博客（[https://www.oreilly.com/ideas/evaluating-machine-learning-models/page/5/hyperparameter-tuning](https://www.oreilly.com/ideas/evaluating-machine-learning-models/page/5/hyperparameter-tuning)），以更详细地了解这里讨论的词汇和思想。

# 使用随机搜索进行参数调整

Bergstra 和 Bengio 在 2012 年提出了一种替代方法（[http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)）。他们证明了在大超参数空间中进行随机搜索比手动方法更有效，正如我们在多项式朴素贝叶斯中做的那样，并且通常与`GridSearch`一样有效，甚至更有效。

我们在这里如何使用它？

在这里，我们将基于Bergstra 和 Bengio 等人的结果。我们将把我们的参数搜索分为以下两个步骤：

1.  使用 `RandomizedSearch`，在有限的迭代次数中遍历广泛的参数组合空间

1.  使用步骤 1 的结果，在略微狭窄的空间内运行 `GridSearch`

我们可以重复之前的步骤，直到我们不再看到结果中的改进，但在这里我们不会这么做。我们将把这个作为读者的练习。我们的示例在下面的片段中概述：

```py
from sklearn.model_selection import RandomizedSearchCV
param_grid = dict(clf__C=[50, 75, 85, 100], 
                  vect__stop_words=['english', None],
                  vect__ngram_range = [(1, 1), (1, 3)],
                  vect__lowercase = [True, False],
                 )
```

如您所见，`param_grid` 变量定义了我们的搜索空间。在我们的管道中，我们为每个估计器分配名称，例如 `vect`、`clf` 等。`clf` 双下划线（也称为 dunder）的约定表示这个 `C` 是 `clf` 对象的属性。同样，对于 `vect`，我们指定是否要移除停用词。例如，`english` 表示移除英语停用词，其中停用词列表是 `scikit-learn` 内部使用的。您也可以用 spaCy、NLTK 或更接近您任务的命令替换它。

```py
random_search = RandomizedSearchCV(lr_clf, param_distributions=param_grid, n_iter=5, scoring='accuracy', n_jobs=-1, cv=3)
random_search.fit(X_train, y_train)
print(f'Calculated cross-validation accuracy: {random_search.best_score_}')
```

之前的代码给出了交叉验证准确率在 0.87 范围内。这可能会根据随机分割的方式而有所不同。

```py
best_random_clf = random_search.best_estimator*_* best_random_clf.fit(X_train, y_train)
imdb_acc(best_random_clf) # 0.90096
```

如前述片段所示，通过简单地更改几个参数，分类器性能提高了超过 1%。这是一个惊人的进步！

让我们现在看看我们正在使用的参数。为了进行比较，您需要知道所有参数的默认值。或者，我们可以简单地查看我们编写的 `param_grid` 参数，并注意选定的参数值。对于不在网格中的所有内容，将选择默认值并保持不变，如下所示：

```py
print(best_random_clf.steps)

[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
          dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
          lowercase=True, max_df=1.0, max_features=None, min_df=1,
          ngram_range=(1, 3), preprocessor=None, stop_words=None,
          strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
          tokenizer=None, vocabulary=None)),
 ('tfidf',
  TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
 ('clf',
  LogisticRegression(C=75, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
            verbose=0, warm_start=False))]
```

在这里，我们注意到最佳分类器中的这些事情：

+   在 `clf` 中选择的 `C` 值是 `100`

+   `lowercase` 被设置为 `False`

+   移除停用词不是一个好主意

+   添加二元词和三元词有帮助

前面的观察结果非常具体于这个数据集和分类器管道。然而，根据我的经验，这可以并且确实有很大的变化。

让我们也避免假设在运行迭代次数如此之少的 `RandomizedSearch` 时，我们总是能得到最佳值。在这种情况下，经验法则是至少运行 60 次迭代，并且也要使用一个更大的 `param_grid`。

在这里，我们使用了 `RandomizedSearch` 来了解我们想要尝试的参数的广泛布局。我们将其中一些参数的最佳值添加到我们的管道中，并将继续对这些参数的值进行实验。

我们没有提到 `C` 参数代表什么或它如何影响分类器。在理解和执行手动参数搜索时，这绝对很重要。通过尝试不同的值来改变 `C` 可以简单地帮助。

# 网格搜索

我们现在将为我们选择的参数运行 `GridSearch`。在这里，我们选择在运行 `GridSearch` 时包括二元词和三元词，同时针对 `LogisticRegression` 的 `C` 参数进行搜索。

我们在这里的意图是尽可能自动化。我们不是在`RandomizedSearch`期间尝试改变`C`的值，而是在人类学习时间（几个小时）和计算时间（几分钟）之间进行权衡。这种思维方式为我们节省了时间和精力。

```py
from sklearn.model_selection import GridSearchCV
param_grid = dict(clf__C=[85, 100, 125, 150])
grid_search = GridSearchCV(lr_clf, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_.steps
```

在前面的代码行中，我们已经使用新的、更简单的`param_grid`在`lr_clf`上运行了分类器，这个`param_grid`只在`LogisticRegression`的`C`参数上工作。

让我们看看我们最佳估计器的步骤，特别是`C`的值，如下面的代码片段所示：

```py
[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
          dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
          lowercase=True, max_df=1.0, max_features=None, min_df=1,
          ngram_range=(1, 3), preprocessor=None, stop_words=None,
          strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
          tokenizer=None, vocabulary=None)),
 ('tfidf',
  TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)),
 ('clf',
  LogisticRegression(C=150, class_weight=None, dual=False, fit_intercept=True,
            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
            verbose=0, warm_start=False))]
```

让我们直接从我们的对象中获取结果性能。这些对象中每个都有一个名为`best_score_`的属性。该属性存储了我们选择的度量标准的最优值。在以下情况下，我们选择了准确率：

```py
print(f'Calculated cross-validation accuracy: {grid_search.best_score_} while random_search was {random_search.best_score_}')

> Calculated cross-validation accuracy: 0.87684 while random_search was 0.87648

best_grid_clf = grid_search.best_estimator_
best_grid_clf.fit(X_train, y_train)

imdb_acc(best_grid_clf) 
> (0.90208, array([1, 1, 1, ..., 0, 0, 1], dtype=int64))
```

如您在前面的代码中看到的，这几乎是非优化模型的3%的性能提升，尽管我们尝试了很少的参数进行优化。

值得注意的是，我们可以并且必须重复这些步骤（`RandomizedSearch`和`GridSearch`），以进一步提高模型的准确率。

# 集成模型

集成模型是提高各种机器学习任务模型性能的非常强大的技术。

在以下部分，我们引用了由MLWave撰写的Kaggle集成指南（[https://mlwave.com/kaggle-ensembling-guide/](https://mlwave.com/kaggle-ensembling-guide/)）。

我们可以解释为什么集成可以帮助减少错误或提高精度，同时展示在我们选择的任务和数据集上流行的技术。虽然这些技术可能不会在我们的特定数据集上带来性能提升，但它们仍然是心理工具箱中一个强大的工具。

为了确保您理解这些技术，我们强烈建议您在几个数据集上尝试它们。

# 投票集成 - 简单多数（也称为硬投票）

也许是简单的多数投票是最简单的集成技术。这基于直觉，单个模型可能在某个特定预测上出错，但几个不同的模型不太可能犯相同的错误。

让我们看看一个例子。

真实值：11011001

数字1和0代表一个假想的二元分类器的`True`和`False`预测。每个数字是对不同输入的单个真或假预测。

让我们假设在这个例子中有三个模型，只有一个错误；它们如下所示：

+   模型A预测：10011001

+   模型B预测：11011001

+   模型C预测：11011001

多数投票给出了以下正确答案：

+   多数投票：11011001

在模型数量为偶数的情况下，我们可以使用平局决定者。平局决定者可以是简单地随机选择一个结果，或者更复杂地选择更有信心的结果。

为了在我们的数据集上尝试这种方法，我们导入`VotingClassifier`从scikit-learn。`VotingClassifier`不使用预训练模型作为输入。它将对模型或分类器管道调用fit，然后使用所有模型的预测来做出最终预测。

为了反驳其他地方对集成方法的过度炒作，我们可以证明硬投票可能会损害你的准确率性能。如果有人声称集成总是有帮助的，你可以向他们展示以下示例以进行更有建设性的讨论：

```py
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('xtc', xtc_clf), ('rfc', rfc_clf)], voting='hard', n_jobs=-1)
voting_clf.fit(X_train, y_train)
hard_voting_acc, _ = imdb_acc(voting_clf)
hard_voting_acc # 0.71092
```

在前面的示例中，我们仅使用了两个分类器进行演示：Extra Trees 和 Random Forest。单独来看，这些分类器的性能上限大约为74%的准确率。

在这个特定的例子中，投票分类器的性能比单独使用任何一个都要差。

# 投票集成 – 软投票

软投票根据类别概率预测类别标签。为每个分类器计算每个类别的预测概率之和（这在多类别的情况下很重要）。然后，分配的类别是具有最大概率之和的类别或`argmax(p_sum)`。

这对于一组校准良好的分类器是推荐的，如下所示：

校准良好的分类器是概率分类器，其predict_proba方法的输出可以直接解释为置信水平。

- 来自sklearn的校准文档 ([http://scikit-learn.org/stable/modules/calibration.html](http://scikit-learn.org/stable/modules/calibration.html))

我们的代码流程与硬投票分类器相同，只是将参数`voting`传递为`soft`*，如下面的代码片段所示：

```py
voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('mnb', mnb_clf)], voting='soft', n_jobs=-1)
voting_clf.fit(X_train, y_train)
soft_voting_acc, _ = imdb_acc(voting_clf)
soft_voting_acc # 0.88216
```

在这里，我们可以看到软投票为我们带来了1.62%的绝对准确率提升。

# 加权分类器

次级模型要推翻最佳（专家）模型，唯一的办法是它们必须集体且自信地同意一个替代方案。

为了避免这种情况，我们可以使用加权多数投票——但为什么要加权？

通常，我们希望在投票中给予更好的模型更多的权重。实现这一目标最简单但计算效率最低的方法是重复使用不同名称的分类器管道，如下所示：

```py
weighted_voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('lr2', lr_clf),('rf', xtc_clf), ('mnb2', mnb_clf),('mnb', mnb_clf)], voting='soft', n_jobs=-1)
weighted_voting_clf.fit(X_train, y_train)
```

用硬投票而不是软投票重复实验。这将告诉你投票策略如何影响我们集成分类器的准确率，如下所示：

```py

weighted_voting_acc, _ = imdb_acc(weighted_voting_clf)
weighted_voting_acc # 0.88092
```

在这里，我们可以看到加权投票为我们带来了1.50%的绝对准确率提升。

那么，到目前为止我们学到了什么？

+   基于简单多数的投票分类器可能比单个模型表现更差

+   软投票比硬投票更有效

+   通过简单地重复分类器来权衡分类器可以帮助

到目前为止，我们似乎是在随机选择分类器。这并不理想，尤其是在我们为商业应用构建模型时，每0.001%的提升都很重要。

# 移除相关分类器

让我们通过三个简单的模型来观察这个方法在实际中的应用。正如你所见，真实值都是1：

```py
1111111100 = 80% accuracy
 1111111100 = 80% accuracy
 1011111100 = 70% accuracy
```

这些模型在预测上高度相关。当我们进行多数投票时，我们并没有看到任何改进：

```py
1111111100 = 80% accuracy
```

现在，让我们将这个结果与以下三个性能较低但高度不相关的模型进行比较：

```py
1111111100 = 80% accuracy
 0111011101 = 70% accuracy
 1000101111 = 60% accuracy
```

当我们使用多数投票集成这些模型时，我们得到以下结果：

```py
1111111101 = 90% accuracy
```

在这里，我们看到了比我们任何单个模型都要高的改进率。模型预测之间的低相关性可以导致更好的性能。在实践中，这很难做到正确，但仍然值得研究。

我们将以下部分留给你作为练习尝试。

作为一个小提示，你需要找到不同模型预测之间的相关性，并选择那些相互之间相关性较低（理想情况下小于0.5）且作为独立模型有足够好的性能的成对模型。

```py

 np.corrcoef(mnb_predictions, lr_predictions)[0][1] # this is too high a correlation at 0.8442355164021454

corr_voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('mnb', mnb_clf)], voting='soft', n_jobs=-1)
corr_voting_clf.fit(X_train, y_train)
corr_acc, _ = imdb_acc(corr_voting_clf)
 print(corr_acc) # 0.88216 
```

那么，当我们使用来自同一方法的两分类器时，我们会得到什么结果呢？

```py
np.corrcoef(dtc_predictions,xtc_predictions )[0][1] # this is looks like a low correlation # 0.3272698219282598

low_corr_voting_clf = VotingClassifier(estimators=[('dtc', dtc_clf), ('xtc', xtc_clf)], voting='soft', n_jobs=-1)
low_corr_voting_clf.fit(X_train, y_train)
low_corr_acc, _ = imdb_acc(low_corr_voting_clf)
 print(low_corr_acc) # 0.70564
```

正如你所见，前面的结果也不是很鼓舞人心，但请记住，这只是一个提示！我们鼓励你继续尝试这个任务，并使用更多的分类器，包括我们在这里没有讨论过的分类器。

# 摘要

在本章中，我们探讨了关于机器学习的几个新想法。这里的目的是展示一些最常见的分类器。我们通过一个主题思想来探讨如何使用它们：将文本转换为数值表示，然后将这个表示输入到分类器中。

本章只涵盖了可用可能性的一小部分。记住，你可以尝试从使用Tfidf进行更好的特征提取到使用`GridSearch`和`RandomizedSearch`调整分类器，以及集成多个分类器。

本章主要关注特征提取和分类的深度学习之前的预方法。

注意，深度学习方法还允许我们使用一个模型，其中特征提取和分类都是从底层数据分布中学习的。虽然关于计算机视觉中的深度学习已经有很多文献，但我们只提供了自然语言处理中深度学习的一个简介。
