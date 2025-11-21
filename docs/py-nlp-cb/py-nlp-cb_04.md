

# 第四章：文本分类

在本章中，我们将使用不同的方法对文本进行分类。文本分类是经典的NLP问题。这个NLP任务包括为文本分配一个值，例如，一个主题（如体育或商业）或情感，如负面或正面，并且任何此类任务都需要评估。

在阅读本章后，您将能够使用关键词、无监督聚类和两种监督算法（**支持向量机**（**SVMs**）和spaCy框架内训练的**卷积神经网络**（**CNN**）模型）对文本进行预处理和分类。我们还将使用GPT-3.5对文本进行分类。

关于本节中讨论的一些概念的理论背景，请参阅Coelho等人所著的《Building Machine Learning Systems with Python》。这本书将解释构建机器学习项目的基础，例如训练和测试集，以及用于评估此类项目的指标，包括精确度、召回率、F1和准确度。

下面是本章中的食谱列表：

+   准备数据集和评估

+   使用关键词进行基于规则的文本分类

+   使用K-Means聚类句子 – 无监督文本分类

+   使用SVMs进行监督文本分类

+   训练spaCy模型进行监督文本分类

+   使用OpenAI模型进行文本分类

# 技术要求

本章的代码可以在书的GitHub仓库的`Chapter04`文件夹中找到([https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition))。像往常一样，我们将使用`poetry`环境安装必要的包。您也可以使用提供的`requirements.txt`文件安装所需的包。我们将使用Hugging Face的`datasets`包来获取本章中我们将使用的所有数据集。

# 准备数据集和评估

在这个食谱中，我们将加载一个数据集，为其准备处理，并创建一个评估基准。这个食谱基于[*第3章*](B18411_03.xhtml#_idTextAnchor067)中的一些食谱，其中我们使用了不同的工具将文本表示成计算机可读的形式。

## 准备工作

对于这个食谱，我们将使用Rotten Tomatoes评论数据集，该数据集可通过Hugging Face获取。这个数据集包含用户电影评论，可以分类为正面和负面。我们将为机器学习分类准备数据集。在这种情况下，准备过程将涉及加载评论，过滤掉非英语语言评论，将文本分词成单词，并移除停用词。在机器学习算法运行之前，文本评论需要被转换成向量。这个转换过程在[*第3章*](B18411_03.xhtml#_idTextAnchor067)中有详细描述。

笔记本位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.1_data_preparation.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.1_data_preparation.ipynb)。

## 如何做到这一点…

我们将分类输入评论是负面还是正面情绪。我们首先过滤掉非英语文本，然后将其分词成单词并移除停用词和标点符号。最后，我们将查看类别分布并回顾每个类别中最常见的单词。

这里是步骤：

1.  运行简单分类器文件：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    ```

1.  导入必要的类。我们从**langdetect**导入**detect**函数，这将帮助我们确定评论的语言。我们还导入**word_tokenize**函数，我们将用它将评论拆分成单词。NLTK中的**FreqDist**类将帮助我们查看评论中最频繁出现的正面和负面单词。我们将使用来自NLTK的**stopwords**列表来过滤文本中的停用词。最后，来自**string**包的**punctuation**字符串将帮助我们过滤标点符号：

    ```py
    from langdetect import detect
    from nltk import word_tokenize
    from nltk.probability import FreqDist
    from nltk.corpus import stopwords
    from string import punctuation
    ```

1.  使用简单分类器文件中的函数加载训练和测试数据集，并打印出两个数据框。我们看到数据包含一个**text**列和一个**label**列，其中文本列是小写的：

    ```py
    (train_df, test_df) = load_train_test_dataset_pd("train", 
        "test")
    print(train_df)
    print(test_df)
    ```

    输出应该看起来类似于以下内容：

    ```py
                                                       text  label
    0     the rock is destined to be the 21st century's ...      1
    1     the gorgeously elaborate continuation of " the...      1
    ...                                                 ...    ...
    8525  any enjoyment will be hinge from a personal th...      0
    8526  if legendary shlockmeister ed wood had ever ma...      0
    [8530 rows x 2 columns]
                                                       text  label
    0     lovingly photographed in the manner of a golde...      1
    1                 consistently clever and suspenseful .      1
    ...                                                 ...    ...
    1061  a terrible movie that some people will neverth...      0
    1062  there are many definitions of 'time waster' bu...      0
    [1066 rows x 2 columns]
    ```

1.  现在，我们在数据框中创建一个名为**lang**的新列，该列将包含评论的语言。我们使用**detect**函数通过**apply**方法填充此列。然后我们过滤数据框，只包含英语评论。过滤前后训练数据框的最终行数显示，有178行是非英语的。这一步可能需要一分钟才能运行：

    ```py
    train_df["lang"] = train_df["text"].apply(detect)
    train_df = train_df[train_df['lang'] == 'en']
    print(train_df)
    ```

    现在的输出应该看起来像这样：

    ```py
                                                       text  label lang
    0     the rock is destined to be the 21st century's ...      1   en
    1     the gorgeously elaborate continuation of " the...      1   en
    ...                                                 ...
        ...  ...
    8528    interminably bleak , to say nothing of boring .      0   en
    8529  things really get weird , though not particula...      0   en
    [8364 rows x 3 columns]
    ```

1.  现在我们将对测试数据框做同样的处理：

    ```py
    test_df["lang"] = test_df["text"].apply(detect)
    test_df = test_df[test_df['lang'] == 'en']
    ```

1.  现在，我们将文本分词成单词。如果你收到一个错误信息说没有找到**english.pickle**分词器，请在运行其余代码之前运行**nltk.download('punkt')**这一行。此代码也包含在**lang_utils**笔记本中([https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/lang_utils.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/lang_utils.ipynb))：

    ```py
    train_df["tokenized_text"] = train_df["text"].apply(
        word_tokenize)
    print(train_df)
    test_df["tokenized_text"] = test_df["text"].apply(word_tokenize)
    print(test_df)
    ```

    结果将类似于以下内容：

    ```py
                                                       text  label lang  \
    0     the rock is destined to be the 21st century's ...
          1   en
    1     the gorgeously elaborate continuation of " the...      
    1   en
    ...                                                 ...    ...
      ...
    8528    interminably bleak , to say nothing of boring .      
    0   en
    8529  things really get weird , though not particula...      
    0   en
                                             tokenized_text
    0     [the, rock, is, destined, to, be, the, 21st, c...
    1     [the, gorgeously, elaborate, continuation, of,...
    ...                                                 ...
    8528  [interminably, bleak, ,, to, say, nothing, of,...
    8529  [things, really, get, weird, ,, though, not, p...
    [8352 rows x 4 columns]
    ```

1.  在这一步中，我们将删除停用词和标点符号。首先，我们使用NLTK包加载停用词。然后我们将**'s**和**``**添加到停用词列表中。你可以添加你认为也是停用词的其他单词。然后我们定义一个函数，该函数将接受一个单词列表作为输入并对其进行过滤，返回一个不包含停用词或标点的新的单词列表。最后，我们将此函数应用于训练数据和测试数据。从打印输出中，我们可以看到停用词和标点符号已被删除：

    ```py
    stop_words = list(stopwords.words('english'))
    stop_words.append("``")
    stop_words.append("'s")
    def remove_stopwords_and_punct(x):
        new_list = [w for w in x if w not in stop_words and w not in punctuation]
        return new_list
    train_df["tokenized_text"] = train_df["tokenized_text"].apply(
        remove_stopwords_and_punct)
    print(train_df)
    test_df["tokenized_text"] = test_df["tokenized_text"].apply(
        remove_stopwords_and_punct)
    print(test_df)
    ```

    结果将类似于这个：

    ```py
                                                       text  label lang  \
    0     the rock is destined to be the 21st century's ...
          1   en
    1     the gorgeously elaborate continuation of " the...
          1   en
    ...                                                 ...
        ...  ...
    8528    interminably bleak , to say nothing of boring .
          0   en
    8529  things really get weird , though not particula...
          0   en
                                             tokenized_text
    0     [rock, destined, 21st, century, new, conan, go...
    1     [gorgeously, elaborate, continuation, lord, ri...
    ...                                                 ...
    8528        [interminably, bleak, say, nothing, boring]
    8529  [things, really, get, weird, though, particula...
    [8352 rows x 4 columns]
    ```

1.  现在我们将检查两个数据集的类别平衡。每个类别中项目数量大致相同是很重要的，因为如果某个类别占主导地位，模型可以学会总是分配这个主导类别，而不会犯很多错误：

    ```py
    print(train_df.groupby('label').count())
    print(test_df.groupby('label').count())
    ```

    我们看到在训练数据中负面评论略多于正面评论，但并不显著，而在测试数据中这两个数字几乎相等。

    ```py
    text  lang  tokenized_text
    label
    0      4185  4185            4185
    1      4167  4167            4167
           text  lang  tokenized_text
    label
    0       523   523             523
    1       522   522             522
    ```

1.  现在我们将清理后的数据保存到磁盘：

    ```py
    train_df.to_json("../data/rotten_tomatoes_train.json")
    test_df.to_json("../data/rotten_tomatoes_test.json")
    ```

1.  在这一步中，我们定义一个函数，该函数将接受一个单词列表和单词数量作为输入，并返回一个**FreqDist**对象。它还将打印出前`n`个最频繁的单词，其中`n`是传递给函数的参数，默认值为**200**：

    ```py
    def get_stats(word_list, num_words=200):
        freq_dist = FreqDist(word_list)
        print(freq_dist.most_common(num_words))
        return freq_dist
    ```

1.  现在我们使用前面的函数来展示正面和负面评论中最常见的单词，以查看这两个类别之间是否存在显著的词汇差异。我们创建了两个单词列表，一个用于正面评论，一个用于负面评论。我们首先通过标签过滤数据框，然后使用**sum**函数从所有评论中获取单词：

    ```py
    positive_train_words = train_df[
        train_df["label"] == 1].tokenized_text.sum()
    negative_train_words = train_df[
        train_df["label"] == 0].tokenized_text.sum()
    positive_fd = get_stats(positive_train_words)
    negative_fd = get_stats(negative_train_words)
    ```

    在输出中，我们看到单词`film`和`movie`以及一些其他单词在这种情况下也充当停用词，因为它们是两组中最常见的单词。我们可以在第7步将它们添加到停用词列表中，并重新进行清理：

    ```py
    [('film', 683), ('movie', 429), ("n't", 286), ('one', 280), ('--', 271), ('like', 209), ('story', 194), ('comedy', 160), ('good', 150), ('even', 144), ('funny', 137), ('way', 135), ('time', 127), ('best', 126), ('characters', 125), ('make', 124), ('life', 124), ('much', 122), ('us', 122), ('love', 118), ...]
    [('movie', 641), ('film', 557), ("n't", 450), ('like', 354), ('one', 293), ('--', 264), ('story', 189), ('much', 177), ('bad', 173), ('even', 160), ('time', 146), ('good', 143), ('characters', 138), ('little', 137), ('would', 130), ('never', 122), ('comedy', 121), ('enough', 107), ('really', 105), ('nothing', 103), ('way', 102), ('make', 101), ...]
    ```

# 使用基于规则的文本分类使用关键词

在这个菜谱中，我们将使用文本的词汇来对烂番茄评论进行分类。我们将创建一个简单的分类器，该分类器将为每个类别有一个向量器。该向量器将包括该类特有的单词。分类将简单地使用每个向量器对文本进行向量化，然后使用拥有更多单词的类别。

## 准备工作

我们将使用`CountVectorizer`类和`sklearn`中的`classification_report`函数，以及NLTK中的`word_tokenize`方法。所有这些都包含在`poetry`环境中。

笔记本位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.2_rule_based.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.2_rule_based.ipynb)。

## 如何做到这一点…

在这个食谱中，我们将为每个类别创建一个单独的向量器。然后我们将使用这些向量器来计算每个评论中每个类别的单词数量以进行分类：

1.  运行简单的分类器文件：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    ```

1.  执行必要的导入：

    ```py
    from nltk import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import classification_report
    ```

1.  从磁盘加载清洗后的数据。如果在这一步收到**FileNotFoundError**错误，你需要首先运行之前的食谱，*获取数据集和评估准备*，因为那些文件是在数据清洗后创建的：

    ```py
    train_df = pd.read_json("../data/rotten_tomatoes_train.json")
    test_df = pd.read_json("../data/rotten_tomatoes_test.json")
    ```

1.  在这里，我们创建一个包含每个类别独特单词的列表。我们首先将**text**列中的所有单词连接起来，根据相关的**label**值（*`0`*表示负面评论，*`1`*表示正面评论）进行过滤。然后我们在**word_intersection**变量中获取出现在这两个列表中的单词。最后，我们为每个类别创建过滤后的单词列表，这些列表不包含同时出现在两个类别中的单词。基本上，我们从相应的列表中删除了同时出现在正面和负面评论中的所有单词：

    ```py
    positive_train_words = train_df[train_df["label"] 
        == 1].text.sum()
    negative_train_words = train_df[train_df["label"] 
        == 0].text.sum()
    word_intersection = set(positive_train_words) \ 
        & set(negative_train_words)
    positive_filtered = list(set(positive_train_words) 
        - word_intersection)
    negative_filtered = list(set(negative_train_words) 
        - word_intersection)
    ```

1.  接下来，我们定义一个函数来创建向量器，每个类别一个。该函数的输入是一个列表的列表，其中每个列表是只出现在那个类中的单词列表；我们在上一步创建了这些列表。对于每个单词列表，我们创建一个**CountVectorizer**对象，将单词列表作为**vocabulary**参数。提供这个参数确保我们只为分类目的计算这些单词：

    ```py
    def create_vectorizers(word_lists):
        vectorizers = []
        for word_list in word_lists:
            vectorizer = CountVectorizer(vocabulary=word_list)
            vectorizers.append(vectorizer)
        return vectorizers
    ```

1.  使用前面的函数创建向量器：

    ```py
    vectorizers = create_vectorizers([negative_filtered,
        positive_filtered])
    ```

1.  在这一步，我们创建一个**vectorize**函数，该函数接受一个单词列表和一个向量器列表。我们首先从单词列表创建一个字符串，因为向量器期望一个字符串。对于列表中的每个向量器，我们将它应用于文本，然后计算该向量器中单词的总计数。最后，我们将这个总和追加到分数列表中。这将按类别计数输入中的单词。函数结束时返回这个分数列表：

    ```py
    def vectorize(text_list, vectorizers):
        text = " ".join(text_list)
        scores = []
        for vectorizer in vectorizers:
            output = vectorizer.transform([text])
            output_sum = sum(output.todense().tolist()[0])
            scores.append(output_sum)
        return scores
    ```

1.  在这一步，我们定义**classify**函数，该函数接受由**vectorize**函数返回的分数列表。这个函数简单地从列表中选择最大分数，并返回对应于类别标签的分数索引：

    ```py
    def classify(score_list):
        return max(enumerate(score_list),key=lambda x: x[1])[0]
    ```

1.  在这里，我们将前面的函数应用于训练数据。我们首先对文本进行向量化，然后进行分类。我们为结果创建一个名为**prediction**的新列：

    ```py
    train_df["prediction"] = train_df["text"].apply(
        lambda x: classify(vectorize(x, vectorizers)))
    print(train_df)
    ```

    输出将类似于以下内容：

    ```py
                                                       text  label lang  \
    0     [rock, destined, 21st, century, new, conan, go...      
    1   en
    1     [gorgeously, elaborate, continuation, lord, ri...      
    1   en
    ...                                                 ...    ...
      ...
    8528        [interminably, bleak, say, nothing, boring]      
    0   en
    8529  [things, really, get, weird, though, particula...      
    0   en
          prediction
    0              1
    1              1
    ...          ...
    8528           0
    8529           0
    [8364 rows x 4 columns]
    ```

1.  现在我们通过打印分类报告来衡量基于规则的分类器的性能。我们输入分配的标签和预测列。结果是整体准确率为87%：

    ```py
    print(classification_report(train_df['label'], 
        train_df['prediction']))
    ```

    这导致以下结果：

    ```py
                  precision    recall  f1-score   support
               0       0.79      0.99      0.88      4194
               1       0.99      0.74      0.85      4170
        accuracy                           0.87      8364
       macro avg       0.89      0.87      0.86      8364
    weighted avg       0.89      0.87      0.86      8364
    ```

1.  在这里，我们对测试数据做同样的处理，我们看到准确率显著下降，降至62%。这是因为我们用来创建向量器的词汇表只来自训练数据，并不全面。它们会导致未见数据中的错误：

    ```py
    test_df["prediction"] = test_df["text"].apply(
        lambda x: classify(vectorize(x, vectorizers)))
    print(classification_report(test_df['label'], 
        test_df['prediction']))
    ```

    结果如下：

    ```py
                  precision    recall  f1-score   support
               0       0.59      0.81      0.68       523
               1       0.70      0.43      0.53       524
        accuracy                           0.62      1047
       macro avg       0.64      0.62      0.61      1047
    weighted avg       0.64      0.62      0.61      1047
    ```

# 使用K-Means进行句子聚类——无监督文本分类

在这个食谱中，我们将使用BBC新闻数据集。该数据集包含按五个主题排序的新闻文章：政治、科技、商业、体育和娱乐。我们将应用无监督的K-Means算法将数据分类到未标记的类别中。

在阅读完这份食谱后，你将能够创建自己的无监督聚类模型，该模型能够将数据分类到几个类别中。之后，你可以将其应用于任何文本数据，而无需先对其进行标记。

## 准备工作

我们将使用`KMeans`算法创建我们的无监督模型。它是`sklearn`包的一部分，并包含在`poetry`环境中。

我们在这里使用的BBC新闻数据集是由Hugging Face用户上传的，随着时间的推移，链接和数据集可能会发生变化。为了避免任何潜在问题，你可以使用GitHub仓库中提供的CSV文件加载的书籍的BBC数据集。

笔记本位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.3_unsupervised_classification.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.3_unsupervised_classification.ipynb)。

## 如何操作...

在这个食谱中，我们将对数据进行预处理，将其向量化，然后使用K-Means进行聚类。由于无监督建模通常没有正确答案，因此评估模型更困难，但我们将能够查看一些统计数据，以及所有聚类中最常见的单词。

你的步骤应该格式化为如下：

1.  运行简单的分类文件：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  导入必要的函数和包：

    ```py
    from nltk import word_tokenize
    from sklearn.cluster import KMeans
    from nltk.probability import FreqDist
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedShuffleSplit
    ```

1.  我们将加载BBC数据集。我们使用Hugging Face的`datasets`包中的`**load_dataset**`函数。此函数在步骤1中运行的简单分类文件中已导入。在Hugging Face仓库中，数据集通常分为训练集和测试集。我们将加载两者，尽管在无监督学习中，测试集通常不使用：

    ```py
    train_dataset = load_dataset("SetFit/bbc-news", split="train")
    test_dataset = load_dataset("SetFit/bbc-news", split="test")
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    print(train_df)
    print(test_df)
    ```

    结果将类似于以下这样：

    ```py
                                                       text  label
         label_text
    0     wales want rugby league training wales could f...      2
              sport
    1     china aviation seeks rescue deal scandal-hit j...      1
           business
    ...                                                 ...    ...
                ...
    1223  why few targets are better than many the econo...      1
           business
    1224  boothroyd calls for lords speaker betty boothr...      4
           politics
    [1225 rows x 3 columns]
                                                      text  label
         label_text
    0    carry on star patsy rowlands dies actress pats...      3
      entertainment
    1    sydney to host north v south game sydney will ...      2
              sport
    ..                                                 ...    ...
                ...
    998  stormy year for property insurers a string of ...      1
           business
    999  what the election should really be about  a ge...      4
           politics
    [1000 rows x 3 columns]
    ```

1.  现在我们将检查训练数据和测试数据中每个类别的项目分布。在分类中，类别平衡很重要，因为一个不成比例更大的类别将影响最终的分类器：

    ```py
    print(train_df.groupby('label_text').count())
    print(test_df.groupby('label_text').count())
    ```

    我们看到类别分布相当均匀，但在`商业`和`体育`类别中示例更多：

    ```py
                   text  label
    label_text
    business        286    286
    entertainment   210    210
    politics        242    242
    sport           275    275
    tech            212    212
                   text  label
    label_text
    business        224    224
    entertainment   176    176
    politics        175    175
    sport           236    236
    tech            189    189
    ```

1.  由于测试集和训练集中的数据量几乎一样多，我们将合并数据并创建一个更好的训练/测试分割。我们首先连接两个数据框。然后我们创建一个**StratifiedShuffleSplit**，它将创建一个训练/测试分割，并在保持类别平衡的同时进行。我们指定我们只需要一个分割（**n_splits**），并且测试数据需要占整个数据集的20%（**test_size**）。**sss**对象的**split**方法返回一个生成器，其中包含分割的索引。然后我们可以使用这些索引来获取新的训练和测试数据框。为此，我们根据相关索引进行筛选，然后复制结果数据框的切片。如果我们没有复制，那么我们就会在原始数据框上工作。然后我们打印出两个数据框的类别计数，并看到有更多的训练数据和较少的测试数据：

    ```py
    combined_df = pd.concat([train_df, test_df],
        ignore_index=True, sort=False)
    print(combined_df)
    sss = StratifiedShuffleSplit(n_splits=1,
        test_size=0.2, random_state=0)
    train_index, test_index = next(
        sss.split(combined_df["text"], combined_df["label"]))
    train_df = combined_df[combined_df.index.isin(
        train_index)].copy()
    test_df = combined_df[combined_df.index.isin(test_index)].copy()
    print(train_df.groupby('label_text').count())
    print(test_df.groupby('label_text').count())
    ```

    结果应该看起来像这样：

    ```py
                   text  label  text_tokenized  text_clean  cluster
    label_text
    business        408    408             408         408      330
    entertainment   309    309             309         309      253
    politics        333    333             333         333      263
    sport           409    409             409         409      327
    tech            321    321             321         321      262
                   text  label  text_tokenized  text_clean  cluster
    label_text
    business        102    102             102         102       78
    entertainment    77     77              77          77       56
    politics         84     84              84          84       70
    sport           102    102             102         102       82
    tech             80     80              80          80       59
    ```

1.  现在，我们将预处理数据：对其进行分词并去除停用词和标点符号。执行此操作的函数（**tokenize**，**remove_stopword_punct**）在步骤1中运行的**language_utils**文件中导入。如果你收到一个错误，表明找不到**english.pickle**分词器，请在运行其余代码之前运行**nltk.download('punkt')**这一行。此代码也包含在**lang_utils notebook**中：

    ```py
    train_df = tokenize(train_df, "text")
    train_df = remove_stopword_punct(train_df, "text_tokenized")
    test_df = tokenize(test_df, "text")
    test_df = remove_stopword_punct(test_df, "text_tokenized")
    ```

1.  在这一步，我们创建向量器。为此，我们从训练新闻文章中获取所有单词。首先，我们将清洗后的文本保存在一个单独的列中，**text_clean**，然后我们将两个数据框保存到磁盘上。然后我们创建一个TF-IDF向量器，它将计算单语元、双语元和三元语（**ngram_range**参数）。然后我们仅在训练数据上拟合向量器。我们仅在训练数据上拟合它的原因是，如果我们同时在训练和测试数据上拟合它，就会导致数据泄露，我们会得到比实际在未见数据上的性能更好的测试分数：

    ```py
    train_df["text_clean"] = train_df["text_tokenized"].apply(
        lambda x: " ".join(list(x)))
    test_df["text_clean"] = test_df["text_tokenized"].apply(
        lambda x: " ".join(list(x)))
    train_df.to_json("../data/bbc_train.json")
    test_df.to_json("../data/bbc_test.json")
    vec = TfidfVectorizer(ngram_range=(1,3))
    matrix = vec.fit_transform(train_df["text_clean"])
    ```

1.  现在我们可以创建五个簇的**Kmeans**分类器，然后将其拟合到前面代码中使用的向量器生成的矩阵上。我们使用**n_clusters**参数指定簇的数量。我们还指定算法应该运行的次数为10，使用**n_init**参数。对于高维问题，建议进行多次运行。初始化分类器后，我们将其拟合到步骤7中使用的向量器创建的矩阵上。这将创建训练数据的聚类：

注意

在实际项目中，你不会像我们这样事先知道簇的数量。你需要使用肘部方法或其他方法来估计最佳类别数量。

```py
km = KMeans(n_clusters=5, n_init=10)
km.fit(matrix)
```

1.  **get_most_frequent_words**函数将返回一个列表，其中包含列表中最频繁的单词。最频繁单词列表将为我们提供有关文本是关于哪个主题的线索。我们将使用此函数打印出聚类中最频繁的单词，以了解它们指的是哪个主题。该函数接受输入文本，对其进行分词，然后创建一个**FreqDist**对象。我们通过使用其**most_common**函数获取顶级单词频率元组，并最终仅获取没有频率的单词并作为列表返回：

    ```py
    def get_most_frequent_words(text, num_words):
        word_list = word_tokenize(text)
        freq_dist = FreqDist(word_list)
        top_words = freq_dist.most_common(num_words)
        top_words = [word[0] for word in top_words]
        return top_words
    ```

1.  在这一步，我们定义了另一个函数，**print_most_common_words_by_cluster**，它使用我们在上一步定义的**get_most_frequent_words**函数。我们以数据框、**KMeans**模型和聚类数量作为输入参数。然后我们获取分配给每个数据点的聚类列表，并在数据框中创建一个指定分配聚类的列。对于每个聚类，我们过滤数据框以获取仅针对该聚类的文本。我们使用此文本将其传递到**get_most_frequent_words**函数以获取该聚类中最频繁单词的列表。我们打印聚类编号和列表，并返回添加了聚类编号列的输入数据框：

    ```py
    def print_most_common_words_by_cluster(input_df, km, 
        num_clusters):
        clusters = km.labels_.tolist()
        input_df["cluster"] = clusters
        for cluster in range(0, num_clusters):
            this_cluster_text = input_df[
                input_df['cluster'] == cluster]
            all_text = " ".join(
                this_cluster_text['text_clean'].astype(str))
            top_200 = get_most_frequent_words(all_text, 200)
            print(cluster)
            print(top_200)
        return input_df
    ```

1.  在这里，我们将上一步定义的函数应用于训练数据框。我们还传递了拟合的**KMeans**模型和聚类数量，*`5`*。打印输出给我们一个关于哪个聚类对应哪个主题的想法。聚类编号可能不同，但包含**劳动**、**政党**、**选举**作为最频繁单词的聚类是**政治**聚类；包含单词**音乐**、**奖项**和**表演**的聚类是**娱乐**聚类；包含单词**游戏**、**英格兰**、**胜利**、**比赛**和**杯**的聚类是**体育**聚类；包含单词**销售**和**增长**的聚类是**商业**聚类；包含单词**软件**、**网络**和**搜索**的聚类是**技术**聚类。我们还注意到单词**说**和**先生**是明显的停用词，因为它们出现在大多数聚类中接近顶部：

    ```py
    print_most_common_words_by_cluster(train_df, km, 5)
    ```

    每次运行训练时结果都会有所不同，但它们可能看起来像这样（输出已截断）：

    ```py
    0
    ['mr', 'said', 'would', 'labour', 'party', 'election', 'blair', 'government', ...]
    1
    ['film', 'said', 'best', 'also', 'year', 'one', 'us', 'awards', 'music', 'new', 'number', 'award', 'show', ...]
    2
    ['said', 'game', 'england', 'first', 'win', 'world', 'last', 'one', 'two', 'would', 'time', 'play', 'back', 'cup', 'players', ...]
    3
    ['said', 'mr', 'us', 'year', 'people', 'also', 'would', 'new', 'one', 'could', 'uk', 'sales', 'firm', 'growth', ...]
    4
    ['said', 'people', 'software', 'would', 'users', 'mr', 'could', 'new', 'microsoft', 'security', 'net', 'search', 'also', ...]
    ```

1.  在这一步，我们使用拟合模型预测测试示例的聚类。我们使用测试数据框的第1行的文本。它是一个政治示例。我们使用向量器将文本转换为向量，然后使用K-Means模型预测聚类。预测是聚类0，在这种情况下是正确的：

    ```py
    test_example = test_df.iloc[1, test_df.columns.get_loc('text')]
    print(test_example)
    vectorized = vec.transform([test_example])
    prediction = km.predict(vectorized)
    print(prediction)
    ```

    结果可能看起来像这样：

    ```py
    lib dems  new election pr chief the lib dems have appointed a senior figure from bt to be the party s new communications chief for their next general election effort.  sandy walkington will now work with senior figures such as matthew taylor on completing the party manifesto. party chief executive lord rennard said the appointment was a  significant strengthening of the lib dem team . mr walkington said he wanted the party to be ready for any  mischief  rivals or the media tried to throw at it.   my role will be to ensure this new public profile is effectively communicated at all levels   he said.  i also know the party will be put under scrutiny in the media and from the other parties as never before - and we will need to show ourselves ready and prepared to counter the mischief and misrepresentation that all too often comes from the party s opponents.  the party is already demonstrating on every issue that it is the effective opposition.  mr walkington s new job title is director of general election communications.
    [0]
    ```

1.  最后，我们使用**joblib**包的**dump**函数保存模型，然后使用**load**函数再次加载它。我们检查加载模型的预测，它与内存中模型的预测相同。这一步将允许我们在未来重用该模型：

    ```py
    dump(km, '../data/kmeans.joblib')
    km_ = load('../data/kmeans.joblib')
    prediction = km_.predict(vectorized)
    print(prediction)
    ```

    结果可能看起来像这样：

    ```py
    [0]
    ```

# 使用SVM进行监督文本分类

在这个菜谱中，我们将构建一个使用SVM算法的机器学习分类器。到这个菜谱结束时，你将拥有一个可以对新输入进行测试并使用我们在上一节中使用的相同`classification_report`工具进行评估的工作分类器。我们将使用与之前`KMeans`相同的BBC新闻数据集。

## 准备工作

我们将继续使用之前菜谱中已经安装的相同包。需要的包安装在了`poetry`环境中，或者通过安装`requirements.txt`文件。

笔记本位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.4-svm_classification.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.4-svm_classification.ipynb)。

## 如何做……

我们将加载在之前菜谱中保存的清洗后的训练和测试数据。然后我们将创建SVM分类器并对其进行训练。我们将使用BERT编码作为我们的向量器。

你的步骤应该格式化如下：

1.  运行简单分类器文件：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    ```

1.  导入必要的函数和包：

    ```py
    from sklearn.svm import SVC
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import confusion_matrix
    ```

1.  在这里，我们加载训练和测试数据。如果你在这个步骤中遇到**FileNotFoundError**错误，请运行之前菜谱中的步骤1-7，即*使用K-Means聚类句子 – 无监督文本分类*。然后我们使用**sample**函数对训练数据进行洗牌。洗牌确保我们没有长序列的数据，这些数据属于同一类别。最后，我们打印出每个类别的示例数量。我们看到类别大致平衡，这对于训练分类器很重要：

    ```py
    train_df = pd.read_json("../data/bbc_train.json")
    test_df = pd.read_json("../data/bbc_test.json")
    train_df.sample(frac=1)
    print(train_df.groupby('label_text').count())
    print(test_df.groupby('label_text').count())
    ```

    结果将如下所示：

    ```py
                   text  label  text_tokenized  text_clean  cluster
    label_text
    business        231    231             231         231      231
    entertainment   181    181             181         181      181
    politics        182    182             182         182      182
    sport           243    243             243         243      243
    tech            194    194             194         194      194
                   text  label  text_tokenized  text_clean
    label_text
    business         58     58              58          58
    entertainment    45     45              45          45
    politics         45     45              45          45
    sport            61     61              61          61
    tech             49     49              49          49
    ```

1.  在这里，我们加载了为我们提供向量的句子转换器**all-MiniLM-L6-v2**模型。要了解更多关于该模型的信息，请阅读[*第3章*](B18411_03.xhtml#_idTextAnchor067)中的*使用BERT和OpenAI嵌入代替词嵌入*菜谱。然后我们定义**get_sentence_vector**函数，该函数返回文本输入的句子嵌入：

    ```py
    model = SentenceTransformer('all-MiniLM-L6-v2')
    def get_sentence_vector(text, model):
        sentence_embeddings = model.encode([text])
        return sentence_embeddings[0]
    ```

1.  定义一个函数，该函数将创建一个SVM对象并在给定输入数据上对其进行训练。它接受输入向量和金标签，创建一个具有RBF核和正则化参数**0.1**的SVC对象，并在训练数据上对其进行训练。然后它返回训练好的分类器：

    ```py
    def train_classifier(X_train, y_train):
        clf = SVC(C=0.1, kernel='rbf')
        clf = clf.fit(X_train, y_train)
        return clf
    ```

1.  在这个步骤中，我们为分类器和**vectorize**方法创建标签列表。然后我们使用位于简单分类器文件中的**create_train_test_data**方法创建训练和测试数据集。然后我们使用**train_classifier**函数训练分类器并打印训练和测试指标。我们看到测试指标非常好，所有指标都超过90%：

    ```py
    target_names=["tech", "business", "sport", 
        "entertainment", "politics"]
    vectorize = lambda x: get_sentence_vector(x, model)
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize, column_name="text_clean")
    clf = train_classifier(X_train, y_train)
    print(classification_report(train_df["label"],
            y_train, target_names=target_names))
    test_classifier(test_df, clf, target_names=target_names)
    ```

    输出将如下所示：

    ```py
                   precision    recall  f1-score   support
             tech       1.00      1.00      1.00       194
         business       1.00      1.00      1.00       231
            sport       1.00      1.00      1.00       243
    entertainment       1.00      1.00      1.00       181
         politics       1.00      1.00      1.00       182
         accuracy                           1.00      1031
        macro avg       1.00      1.00      1.00      1031
     weighted avg       1.00      1.00      1.00      1031
                   precision    recall  f1-score   support
             tech       0.92      0.98      0.95        49
         business       0.95      0.90      0.92        58
            sport       1.00      1.00      1.00        61
    entertainment       1.00      0.98      0.99        45
         politics       0.96      0.98      0.97        45
         accuracy                           0.97       258
        macro avg       0.97      0.97      0.97       258
     weighted avg       0.97      0.97      0.96       258
    ```

1.  在这一步，我们打印出混淆矩阵以查看分类器在哪些地方犯了错误。行代表正确的标签，列代表预测的标签。我们看到最多的混淆（四个例子）是正确的标签是**商业**但预测为**技术**，以及正确的标签是**商业**而预测为**政治**（两个例子）。我们还看到**商业**被错误地预测为**技术**、**娱乐**和**政治**各一次。这些错误也反映在指标中，我们看到**商业**的召回率和精确率都受到了影响。唯一得分完美的类别是**体育**，它在混淆矩阵的每个地方都是零，除了正确的行和预测的列的交叉点。我们可以使用混淆矩阵来查看哪些类别之间有最多的混淆，并在必要时采取措施纠正：

    ```py
    print(confusion_matrix(test_df["label"], test_df["prediction"]))
    [[48  1  0  0  0]
     [ 4 52  0  0  2]
     [ 0  0 61  0  0]
     [ 0  1  0 44  0]
     [ 0  1  0  0 44]]
    ```

1.  我们将在新的示例上测试分类器。我们首先将文本向量化，然后使用训练好的模型进行预测并打印预测结果。新文章是关于技术的，预测类别为 *`0`*，这确实是**技术**：

    ```py
    new_example = """iPhone 12: Apple makes jump to 5G
    Apple has confirmed its iPhone 12 handsets will be its first to work on faster 5G networks.
    The company has also extended the range to include a new "Mini" model that has a smaller 5.4in screen.
    The US firm bucked a wider industry downturn by increasing its handset sales over the past year.
    But some experts say the new features give Apple its best opportunity for growth since 2014, when it revamped its line-up with the iPhone 6.
    "5G will bring a new level of performance for downloads and uploads, higher quality video streaming, more responsive gaming, real-time interactivity and so much more," said chief executive Tim Cook.
    …"""
    vector = vectorize(new_example)
    prediction = clf.predict([vector])
    print(prediction))
    ```

    结果将如下所示：

    ```py
    [0]
    ```

## 还有更多…

有许多不同的机器学习算法可以用作 SVM 算法的替代。其中一些包括回归、朴素贝叶斯和决策树。你可以尝试它们，看看哪个表现更好。

# 训练 spaCy 模型进行监督文本分类

在这个菜谱中，我们将使用 BBC 数据集训练 spaCy 模型，与之前菜谱中使用的数据集相同，以预测文本类别。

## 准备工作

我们将使用 spaCy 包来训练我们的模型。所有依赖项都由 `poetry` 环境处理。

您需要从书籍的 GitHub 仓库下载配置文件，位于 [https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/spacy_config.cfg](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/spacy_config.cfg)。此文件应位于相对于笔记本的路径 `../data/spacy_config.cfg`。

注意

您可以修改训练配置，或在其 [https://spacy.io/usage/training](https://spacy.io/usage/training) 上生成自己的配置。

笔记本位于 [https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.5-spacy_textcat.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.5-spacy_textcat.ipynb)。

## 如何做…

训练的一般结构类似于普通机器学习模型训练，其中我们清理数据，创建数据集，并将其分为训练集和测试集。然后我们训练一个模型并在未见过的数据上测试它：

1.  运行简单的分类器文件：

    ```py
    %run -i "../util/lang_utils.ipynb"
    ```

1.  导入必要的函数和包：

    ```py
    import pandas as pd
    from spacy.cli.train import train
    from spacy.cli.evaluate import evaluate
    from spacy.cli.debug_data import debug_data
    from spacy.tokens import DocBin
    ```

1.  在这里，我们定义了**preprocess_data_entry**函数，它将接受输入文本、其标签以及所有标签的列表。然后它将在文本上运行小的spaCy模型。这个模型是通过在步骤1中运行语言实用工具文件导入的。在这个步骤中我们使用哪个模型并不重要，因为我们只是想要从文本中创建一个**Doc**对象。这就是为什么我们运行最小的模型，因为它花费的时间更少。然后我们为文本类别创建一个one-hot编码，将类别标签设置为*`1`*，其余设置为*`0`*。然后我们创建一个将类别名称映射到其值的标签字典。我们将**doc.cats**属性设置为这个字典，并返回**Doc**对象。spaCy需要对此数据进行预处理才能训练分类模型：

    ```py
    def preprocess_data_entry(input_text, label, label_list):
        doc = small_model(input_text)
        cats = [0] * len(label_list)
        cats[label] = 1
        final_cats = {}
        for i, label in enumerate(label_list):
            final_cats[label] = cats[i]
        doc.cats = final_cats
        return doc
    ```

1.  现在我们准备训练和测试数据集。我们为spaCy算法所需的训练和测试数据创建了**DocBin**对象。然后我们从磁盘加载保存的数据。这是我们保存到K-Means配方中的数据。如果你在这里遇到**FileNotFoundError**错误，你需要运行*使用K-Means聚类句子 – 无监督文本分类*配方中的步骤1-7。然后我们随机打乱训练数据框。然后我们使用之前定义的函数预处理每个数据点。然后我们将每个数据点添加到**DocBin**对象中。最后，我们将两个数据集保存到磁盘：

    ```py
    train_db = DocBin()
    test_db = DocBin()
    label_list = ["tech", "business", "sport", 
        "entertainment", "politics"]
    train_df = pd.read_json("../data/bbc_train.json")
    test_df = pd.read_json("../data/bbc_test.json")
    train_df.sample(frac=1)
    for idx, row in train_df.iterrows():
        text = row["text"]
        label = row["label"]
        doc = preprocess_data_entry(text, label, label_list)
        train_db.add(doc)
    for idx, row in test_df.iterrows():
        text = row["text"]
        label = row["label"]
        doc = preprocess_data_entry(text, label, label_list)
        test_db.add(doc)
    train_db.to_disk('../data/bbc_train.spacy')
    test_db.to_disk('../data/bbc_test.spacy')
    ```

1.  使用**train**命令训练模型。为了使训练工作，你需要将配置文件下载到**data**文件夹中。这在本配方的*准备就绪*部分有解释。训练配置指定了训练和测试数据集的位置，因此你需要运行前面的步骤才能使训练工作。**train**命令将模型保存到我们在输入中指定的目录的**model_last**子目录中（在本例中为**../models/spacy_textcat_bbc/**）：

    ```py
    train("../data/spacy_config.cfg", output_path="../models/spacy_textcat_bbc")
    ```

    输出结果可能会有所不同，但可能看起来像这样（为了便于阅读而截断）。我们可以看到，我们训练的模型的最终准确率是85%：

    ```py
    ℹ Saving to output directory: ../models/spacy_textcat_bbc
    ℹ Using CPU
    =========================== Initializing pipeline ===========================
    ✔ Initialized pipeline
    4.5-spacy_textcat.ipynb
    ============================= Training pipeline =============================
    ℹ Pipeline: ['tok2vec', 'textcat']
    ℹ Initial learn rate: 0.001
    E    #       LOSS TOK2VEC  LOSS TEXTCAT  CATS_SCORE  SCORE
    ---  ------  ------------  ------------  ----------  ------
      0       0          0.00          0.16        8.48    0.08
      0     200         20.77         37.26       35.58    0.36
      0     400         98.56         35.96       26.90    0.27
      0     600         49.83         37.31       36.60    0.37
    … (truncated)
      4    4800       7571.47          9.64       80.25    0.80
      4    5000      16164.99         10.58       87.71    0.88
      5    5200       8604.43          8.20       84.98    0.85
    ✔ Saved pipeline to output directory
    ../models/spacy_textcat_bbc/model-last
    ```

1.  现在我们对一个未见过的例子进行模型测试。我们首先加载模型，然后从测试数据中获取一个例子。然后我们检查文本及其类别。我们在输入文本上运行模型并打印出结果概率。模型将给出一个包含各自概率得分的类别字典。这些得分表示文本属于相应类别的概率。概率最高的类别是我们应该分配给文本的类别。类别字典在**doc.cats**属性中，就像我们在准备数据时一样，但在这个情况下，模型分配它。在这种情况下，文本是关于政治的，模型正确地将其分类：

    ```py
    nlp = spacy.load("../models/spacy_textcat_bbc/model-last")
    input_text = test_df.iloc[1, test_df.columns.get_loc('text')]
    print(input_text)
    print(test_df["label_text"].iloc[[1]])
    doc = nlp(input_text)
    print("Predicted probabilities: ", doc.cats)
    ```

    输出将看起来类似于这样：

    ```py
    lib dems  new election pr chief the lib dems have appointed a senior figure from bt to be the party s new communications chief for their next general election effort.  sandy walkington will now work with senior figures such as matthew taylor on completing the party manifesto. party chief executive lord rennard said the appointment was a  significant strengthening of the lib dem team . mr walkington said he wanted the party to be ready for any  mischief  rivals or the media tried to throw at it.   my role will be to ensure this new public profile is effectively communicated at all levels   he said.  i also know the party will be put under scrutiny in the media and from the other parties as never before - and we will need to show ourselves ready and prepared to counter the mischief and misrepresentation that all too often comes from the party s opponents.  the party is already demonstrating on every issue that it is the effective opposition.  mr walkington s new job title is director of general election communications.
    8    politics
    Name: label_text, dtype: object
    Predicted probabilities:  {'tech': 3.531841841208916e-08, 'business': 0.000641813559923321, 'sport': 0.00033847044687718153, 'entertainment': 0.00016174423217307776, 'politics': 0.9988579750061035}
    ```

1.  在这一步，我们定义一个**get_prediction**函数，它接受文本、spaCy模型和潜在类别的列表，并输出概率最高的类别。然后我们将此函数应用于测试数据框的**text**列：

    ```py
    def get_prediction(input_text, nlp_model, target_names):
        doc = nlp_model(input_text)
        category = max(doc.cats, key = doc.cats.get)
        return target_names.index(category)
    test_df["prediction"] = test_df["text"].apply(
        lambda x: get_prediction(x, nlp, label_list))
    ```

1.  现在，我们根据之前步骤中生成的测试数据框中的数据打印出分类报告。模型的总体准确率为87%，它之所以有点低，是因为我们没有足够的数据来训练更好的模型：

    ```py
    print(classification_report(test_df["label"],
        test_df["prediction"], target_names=target_names))
    ```

    结果应该看起来像这样：

    ```py
                   precision    recall  f1-score   support
             tech       0.82      0.94      0.87        80
         business       0.94      0.83      0.89       102
            sport       0.89      0.89      0.89       102
    entertainment       0.94      0.87      0.91        77
         politics       0.78      0.83      0.80        84
         accuracy                           0.87       445
        macro avg       0.87      0.87      0.87       445
     weighted avg       0.88      0.87      0.87       445
    ```

1.  在这一步，我们使用spaCy的**evaluate**命令进行相同的评估。此命令接受模型路径和测试数据集路径，并以略微不同的格式输出分数。我们看到这两个步骤的分数是一致的：

    ```py
    evaluate('../models/spacy_textcat_bbc/model-last', '../data/bbc_test.spacy')
    ```

    结果应该看起来像这样：

    ```py
    {'token_acc': 1.0,
     'token_p': 1.0,
     'token_r': 1.0,
     'token_f': 1.0,
     'cats_score': 0.8719339318444819,
     'cats_score_desc': 'macro F',
     'cats_micro_p': 0.8719101123595505,
     'cats_micro_r': 0.8719101123595505,
     'cats_micro_f': 0.8719101123595505,
     'cats_macro_p': 0.8746516896205309,
     'cats_macro_r': 0.8732906799083269,
     'cats_macro_f': 0.8719339318444819,
     'cats_macro_auc': 0.9800144873453936,
     'cats_f_per_type': {'tech': {'p': 0.8152173913043478,
       'r': 0.9375,
       'f': 0.872093023255814},
      'business': {'p': 0.9444444444444444,
       'r': 0.8333333333333334,
       'f': 0.8854166666666667},
      'sport': {'p': 0.8921568627450981,
       'r': 0.8921568627450981,
       'f': 0.8921568627450981},
      'entertainment': {'p': 0.9436619718309859,
       'r': 0.8701298701298701,
       'f': 0.9054054054054054},
      'politics': {'p': 0.7777777777777778,
       'r': 0.8333333333333334,
       'f': 0.8045977011494253}},
     'cats_auc_per_type': {'tech': 0.9842808219178081,
      'business': 0.9824501229063054,
      'sport': 0.9933544846510032,
      'entertainment': 0.9834839073969509,
      'politics': 0.9565030998549005},
     'speed': 6894.989948433934}
    ```

# 使用OpenAI模型进行文本分类

在这个食谱中，我们将要求OpenAI模型对输入文本进行分类。我们将使用之前食谱中相同的BBC数据集。

## 准备工作

要运行此食谱，你需要安装`openai`包，该包作为`poetry`环境的一部分提供，以及`requirements.txt`文件。你还需要一个OpenAI API密钥。将其粘贴到文件实用工具笔记本中提供的字段（[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/file_utils.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/file_utils.ipynb)）中。

笔记本位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.6_openai_classification.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter04/4.6_openai_classification.ipynb)。

注意

OpenAI经常更改和淘汰现有模型，并引入新的模型。我们在这个食谱中使用的**gpt-3.5-turbo**模型在你阅读本文时可能已经过时。在这种情况下，请检查OpenAI文档并选择另一个合适的模型。

## 如何操作…

在这个食谱中，我们将查询OpenAI API并提供一个作为提示的分类请求。然后我们将对结果进行后处理，并评估Open AI模型在此任务上的表现：

1.  运行简单的分类器和文件实用工具笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/util_simple_classifier.ipynb"
    ```

1.  使用API密钥导入必要的函数和包以创建OpenAI客户端：

    ```py
    import re
    from sklearn.metrics import classification_report
    from openai import OpenAI
    client = OpenAI(api_key=OPEN_AI_KEY)
    ```

1.  使用Hugging Face加载训练和测试数据集，无需对类别数量进行预处理，因为我们不会训练新的模型：

    ```py
    train_dataset = load_dataset("SetFit/bbc-news", split="train")
    test_dataset = load_dataset("SetFit/bbc-news", split="test")
    ```

1.  加载并打印数据集中的第一个示例及其类别：

    ```py
    example = test_dataset[0]["text"]
    category = test_dataset[0]["label_text"]
    print(example)
    print(category)
    ```

    结果应该是这样的：

    ```py
    carry on star patsy rowlands dies actress patsy rowlands  known to millions for her roles in the carry on films  has died at the age of 71.  rowlands starred in nine of the popular carry on films  alongside fellow regulars sid james  kenneth williams and barbara windsor. she also carved out a successful television career  appearing for many years in itv s well-loved comedy bless this house....
    entertainment
    ```

1.  在这个示例上运行OpenAI模型。在第5步，我们查询OpenAI API，要求它对这个示例进行分类。我们创建提示并将示例文本附加到它。在提示中，我们指定模型将输入文本分类为五个类别之一，并指定输出格式。如果我们不包括这些输出指令，它可能会添加其他词语并返回类似“这个话题是娱乐”的文本。我们选择**gpt-3.5-turbo**模型并指定提示、温度和其他几个参数。我们将温度设置为*`0`*，以便模型响应没有或最小变化。然后我们打印API返回的响应。输出可能会有所不同，但在大多数情况下，它应该返回“娱乐”，这是正确的：

    ```py
    prompt="""You are classifying texts by topics. There are 5 topics: tech, entertainment, business, politics and sport.
    Output the topic and nothing else. For example, if the topic is business, your output should be "business".
    Give the following text, what is its topic from the above list without any additional explanations: """ + example
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": 
                "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    print(response.choices[0].message.content)
    ```

    结果可能会有所不同，但应该看起来像这样：

    ```py
    entertainment
    ```

1.  创建一个函数，该函数将提供输入文本的分类并返回类别。它接受输入文本并调用我们之前使用的相同提示的OpenAI API。然后它将响应转换为小写，去除额外的空白，并返回它：

    ```py
    def get_gpt_classification(input_text):
        prompt="""You are classifying texts by topics. There are 5 topics: tech, entertainment, business, politics and sport.
    Output the topic and nothing else. For example, if the topic is business, your output should be "business".
    Give the following text, what is its topic from the above list without any additional explanations: """ + input_text
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": 
                    "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        classification = response.choices[0].message.content
        classification = classification.lower().strip()
        return classification
    ```

1.  在这一步，我们加载测试数据。我们从Hugging Face获取测试数据集并将其转换为数据框。然后我们打乱数据框并选择前200个示例。原因是我们要通过OpenAI API降低测试这个分类器的成本。你可以修改你测试此方法的数据量：

    ```py
    test_df = test_dataset.to_pandas()
    test_df.sample(frac=1)
    test_data = test_df[0:200].copy()
    ```

1.  在第8步中，我们使用**get_gpt_classification**函数在测试数据框中创建一个新列。根据你拥有的测试示例数量，运行可能需要几分钟：

    ```py
    test_data["gpt_prediction"] = test_data["text"].apply(
        lambda x: get_gpt_classification(x))
    ```

1.  尽管我们指示OpenAI只提供类别作为答案，但它可能还会添加一些其他词语，因此我们定义了一个函数**get_one_word_match**，用于清理OpenAI的输出。在这个函数中，我们使用正则表达式匹配其中一个类别标签，并从原始字符串中返回该单词。然后我们将此函数应用于测试数据框中的**gpt_prediction**列：

    ```py
    def get_one_word_match(input_text):
        loc = re.search(
            r'tech|entertainment|business|sport|politics',
            input_text).span()
        return input_text[loc[0]:loc[1]]
    test_data["gpt_prediction"] = test_data["gpt_prediction"].apply(
        lambda x: get_one_word_match(x))
    ```

1.  现在我们将标签转换为数值格式：

    ```py
    label_list = ["tech", "business", "sport", 
        "entertainment", "politics"]
    test_data["gpt_label"] = test_data["gpt_prediction"].apply(
        lambda x: label_list.index(x))
    ```

1.  我们打印出结果数据框。我们可以看到我们进行评估所需的所有信息。我们既有正确的标签（**标签**列）也有预测的标签（**gpt_label**列）：

    ```py
    print(test_data)
    ```

    结果应该看起来像这样：

    ```py
                                                      text  label
         label_text  \
    0    carry on star patsy rowlands dies actress pats...      3
      entertainment
    1    sydney to host north v south game sydney will ...      2
              sport
    ..                                                 ...    ...
                ...
    198  xbox power cable  fire fear  microsoft has sai...      0
               tech
    199  prop jones ready for hard graft adam jones say...      2
              sport
        gpt_prediction  gpt_label
    0    entertainment          3
    1            sport          2
    ..             ...        ...
    198           tech          0
    199          sport          2
    ```

1.  现在我们可以打印出评估OpenAI分类的分类报告：

    ```py
    print(classification_report(test_data["label"],
            test_data["gpt_label"], target_names=label_list))
    ```

    结果可能会有所不同。这是一个示例输出。我们看到整体准确率很好，达到90%：

    ```py
                   precision    recall  f1-score   support
             tech       0.97      0.80      0.88        41
         business       0.87      0.89      0.88        44
            sport       1.00      0.96      0.98        48
    entertainment       0.88      0.90      0.89        40
         politics       0.76      0.96      0.85        27
         accuracy                           0.90       200
        macro avg       0.90      0.90      0.90       200
     weighted avg       0.91      0.90      0.90       200
    ```
