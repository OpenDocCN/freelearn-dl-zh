

# 表示文本——捕捉语义

将单词、短语和句子的意义表示成计算机能理解的形式是NLP处理的基础之一。例如，机器学习将每个数据点表示为一个数字列表（固定大小的向量），我们面临的问题是如何将单词和句子转换为这些向量。大多数NLP任务首先将文本表示成某种数值形式，在本章中，我们将展示几种实现这一目标的方法。

首先，我们将创建一个简单的分类器来展示每种编码方法的有效性，然后我们将用它来测试不同的编码方法。我们还将学习如何将诸如“炸鸡”之类的短语转换为向量——也就是说，如何训练短语用`word2vec`模型。最后，我们将看到如何使用基于向量的搜索。

对于本节中讨论的一些概念的理论背景，请参阅Coelho等人所著的《用Python构建机器学习系统》。这本书将解释构建机器学习项目的基础，例如训练集和测试集，以及用于评估此类项目的指标，包括精确度、召回率、F1和准确率。

本章涵盖了以下食谱：

+   创建一个简单的分类器

+   将文档放入词袋中

+   构建一个*N*-gram模型

+   使用TF-IDF表示文本

+   使用词嵌入

+   训练自己的嵌入模型

+   使用BERT和OpenAI嵌入而不是词嵌入

+   使用**检索增强****生成**（**RAG**）

# 技术要求

本章的代码位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/tree/main/Chapter03](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/tree/main/Chapter03)。本章所需的包应通过`poetry`环境自动安装。

此外，我们还将使用以下URL中定位的模型和数据集。谷歌`word2vec`模型是一个将单词表示为向量的模型，IMDB数据集包含电影标题、类型和描述。将它们下载到`root`目录下的`data`文件夹中：

+   **谷歌****word2vec****模型**：[https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)

+   **IMDB电影数据集**：[https://github.com/venusanvi/imdb-movies/blob/main/IMDB-Movie-Data.csv](https://github.com/venusanvi/imdb-movies/blob/main/IMDB-Movie-Data.csv)（本书的GitHub仓库中也有提供）

除了前面的文件外，我们还将使用我们将在第一个菜谱中创建的简单分类器中的各种函数。此文件可在[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/util_simple_classifier.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/util_simple_classifier.ipynb)找到。

# 创建一个简单的分类器

我们需要将文本表示为向量的原因是为了将其转换为计算机可读的形式。计算机不能理解单词，但擅长处理数字。NLP的主要任务之一是文本分类，我们将创建一个用于电影评论的分类器。我们将使用相同的分类器代码，但使用不同的从文本创建向量的方法。

在本节中，我们将创建一个分类器，该分类器将为*烂番茄*评论分配负面或正面情绪，这是一个通过Hugging Face提供的、包含大量开源模型和数据集的数据库。然后我们将使用基线方法，通过计算文本中存在的不同词性数量（动词、名词、专有名词、形容词、副词、助动词、代词、数字和标点符号）来编码文本。

到本菜谱结束时，我们将创建一个单独的文件，其中包含创建数据集和训练分类器的函数。我们将使用此文件在本章中测试不同的编码方法。

## 准备工作

在本菜谱中，我们将创建一个简单的电影评论分类器。它将是一个`sklearn`包。

## 如何操作...

我们将从Hugging Face加载Rotten Tomatoes数据集。我们将只使用数据集的一部分，以便训练时间不会很长：

1.  导入文件和语言**util**笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    %run -i "../util/lang_utils.ipynb"
    ```

1.  从Hugging Face（**datasets**包）加载训练和测试数据集。对于训练集和测试集，我们将选择数据的前15%和后15%，而不是加载完整的数据集。完整的数据集很大，训练模型需要很长时间：

    ```py
    from datasets import load_dataset
    train_dataset = load_dataset("rotten_tomatoes",
        split="train[:15%]+train[-15%:]")
    test_dataset = load_dataset("rotten_tomatoes",
        split="test[:15%]+test[-15%:]")
    ```

1.  打印出每个数据集的长度：

    ```py
    print(len(train_dataset))
    print(len(test_dataset))
    ```

    输出应该是这样的：

    ```py
    2560
    320
    ```

1.  在这里，我们创建了**POS_vectorizer**类。这个类有一个名为**vectorize**的方法，它处理文本并计算动词、名词、专有名词、形容词、副词、助动词、代词、数字和标点符号的数量。该类需要一个**spaCy**模型来处理文本。每段文本被转换成大小为10的向量。向量的第一个元素是文本的长度，其他数字表示该特定词性的文本中的单词数量：

    ```py
    class POS_vectorizer:
        def __init__(self, spacy_model):
            self.model = spacy_model
        def vectorize(self, input_text):
            doc = self.model(input_text)
            vector = []
            vector.append(len(doc))
            pos = {"VERB":0, "NOUN":0, "PROPN":0, "ADJ":0,
                "ADV":0, "AUX":0, "PRON":0, "NUM":0, "PUNCT":0}
            for token in doc:
                if token.pos_ in pos:
                    pos[token.pos_] += 1
            vector_values = list(pos.values())
            vector = vector + vector_values
            return vector
    ```

1.  现在，我们可以测试**POS_vectorizer**类。我们取第一篇评论的文本进行处理，并使用小的**spaCy**模型创建向量器。然后我们使用新创建的类对文本进行向量化：

    ```py
    sample_text = train_dataset[0]["text"]
    vectorizer = POS_vectorizer(small_model)
    vector = vectorizer.vectorize(sample_text)
    ```

1.  让我们打印文本和向量：

    ```py
    print(sample_text)
    print(vector)
    ```

    结果应该看起来像这样。我们可以看到，向量正确地计算了词性。例如，有五个标点符号（两个引号、一个逗号、一个句号和一个破折号）：

    ```py
    the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .
    [38, 3, 8, 3, 4, 1, 3, 1, 0, 5]
    ```

1.  现在，我们将为训练我们的分类器准备数据。我们首先导入**pandas**和**numpy**包，然后创建两个数据框，一个用于训练，另一个用于测试。在每个数据集中，我们创建一个名为**vector**的新列，其中包含该文本的向量。我们使用**apply**方法将文本转换为向量并将它们存储在新列中。在此方法中，我们传递一个lambda函数，该函数接受一段文本并将其应用于**POS_vectorizer**类的**vectorize**方法。然后，我们将向量列和标签列转换为**numpy**数组，以便数据以正确的格式供分类器使用。我们使用**np.stack**方法对向量进行操作，因为它已经是一个列表，而使用**to_numpy**方法对评论标签进行操作，因为它们只是数字：

    ```py
    import pandas as pd
    import numpy as np
    train_df = train_dataset.to_pandas()
    train_df.sample(frac=1)
    test_df = test_dataset.to_pandas()
    train_df["vector"] = train_df["text"].apply(
        lambda x: vectorizer.vectorize(x))
    test_df["vector"] = test_df["text"].apply(
        lambda x: vectorizer.vectorize(x))
    X_train = np.stack(train_df["vector"].values, axis=0)
    X_test = np.stack(test_df["vector"].values, axis=0)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    ```

1.  现在，我们将训练分类器。我们将选择逻辑回归算法，因为它是最简单的算法之一，同时也是最快的算法之一。首先，我们从**sklearn**中导入**LogisticRegression**类和**classification_report**方法。然后，我们创建**LogisticRegression**对象，并最终在之前步骤中的数据上对其进行训练：

    ```py
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    clf = LogisticRegression(C=0.1)
    clf = clf.fit(X_train, y_train)
    ```

1.  我们可以通过将**predict**方法应用于测试数据中的向量并打印出分类报告来测试分类器。我们可以看到，整体准确率很低，略高于随机水平。这是因为我们使用的向量表示非常粗糙。在下一节中，我们将使用其他向量并看看它们如何影响分类器结果：

    ```py
    test_df["prediction"] = test_df["vector"].apply(
        lambda x: clf.predict([x])[0])
    print(classification_report(test_df["label"], 
        test_df["prediction"]))
    ```

    输出应该类似于这个：

    ```py
                  precision    recall  f1-score   support
               0       0.59      0.54      0.56       160
               1       0.57      0.62      0.60       160
        accuracy                           0.58       320
       macro avg       0.58      0.58      0.58       320
    weighted avg       0.58      0.58      0.58       320
    ```

## 还有更多...

现在，我们将前面的代码转换为几个函数，这样我们就可以只改变在构建数据集时使用的向量器。生成的文件位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/util_simple_classifier.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/util/util_simple_classifier.ipynb)。生成的代码将如下所示：

1.  导入必要的包：

    ```py
    from datasets import load_dataset
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    ```

1.  定义一个函数，该函数将创建并返回训练和测试数据框。它将从Hugging Face的Rotten Tomatoes数据集中创建它们：

    ```py
    def load_train_test_dataset_pd():
        train_dataset = load_dataset("rotten_tomatoes",
            split="train[:15%]+train[-15%:]")
        test_dataset = load_dataset("rotten_tomatoes",
            split="test[:15%]+test[-15%:]")
        train_df = train_dataset.to_pandas()
        train_df.sample(frac=1)
        test_df = test_dataset.to_pandas()
        return (train_df, test_df)
    ```

1.  此函数接收数据框和**vectorize**方法，并为训练和测试数据创建**numpy**数组。这将使我们能够使用创建的向量来训练逻辑回归分类器：

    ```py
    def create_train_test_data(train_df, test_df, vectorize):
        train_df["vector"] = train_df["text"].apply(
            lambda x: vectorize(x))
        test_df["vector"] = test_df["text"].apply(
            lambda x: vectorize(x))
        X_train = np.stack(train_df["vector"].values, axis=0)
        X_test = np.stack(test_df["vector"].values, axis=0)
        y_train = train_df["label"].to_numpy()
        y_test = test_df["label"].to_numpy()
        return (X_train, X_test, y_train, y_test)
    ```

1.  此函数在给定的训练数据上训练一个逻辑回归分类器：

    ```py
    def train_classifier(X_train, y_train):
        clf = LogisticRegression(C=0.1)
        clf = clf.fit(X_train, y_train)
        return clf
    ```

1.  此最终函数接收测试数据和训练好的分类器，并打印出分类报告：

    ```py
    def test_classifier(test_df, clf):
        test_df["prediction"] = test_df["vector"].apply(
            lambda x: clf.predict([x])[0])
        print(classification_report(test_df["label"],         test_df["prediction"]))
    ```

在每个展示新向量化方法的后续部分，我们将使用这个文件来预加载必要的函数以测试分类结果。这将使我们能够评估不同的向量化方法。我们将只改变向量化器，保持分类器不变。当分类器表现更好时，这反映了底层向量化器对文本的表示效果。

# 将文档放入词袋中

**词袋**是表示文本的最简单方式。我们将文本视为一组**文档**，其中文档可以是句子、科学文章、博客文章或整本书。由于我们通常将不同的文档相互比较或将它们用于其他文档的更大上下文中，所以我们处理的是文档集合，而不仅仅是一个单独的文档。

词袋方法使用一个“训练”文本，为它提供一个应该考虑的单词列表。在编码新句子时，它计算每个单词在文档中的出现次数，最终向量包括词汇表中每个单词的这些计数。这种表示可以随后输入到机器学习算法中。

这种向量化的方法被称为“词袋”是因为它不考虑单词之间的相互关系，只计算每个单词出现的次数。关于什么代表一个文档的决定权在工程师手中，在许多情况下，这将是显而易见的。例如，如果你正在对属于特定主题的推文进行分类，那么一条单独的推文就是你的文档。相反，如果你想找出哪本书的章节与你已经拥有的书最相似，那么章节就是文档。

在这个菜谱中，我们将为Rotten Tomatoes的评论创建一个词袋。我们的文档将是评论。然后，我们通过构建逻辑回归分类器并使用前一个菜谱中的代码来测试编码。

## 准备工作

对于这个菜谱，我们将使用来自`sklearn`包的`CountVectorizer`类。它包含在`poetry`环境中。`CountVectorizer`类专门设计用来计算文本中每个单词出现的次数。

## 如何做到这一点...

我们的代码将接受一组文档——在这个例子中，是评论——并将它们表示为一个向量矩阵。我们将使用来自Hugging Face的Rotten Tomatoes评论数据集来完成这项任务：

1.  运行简单的分类器**实用程序**文件，然后导入**CountVectorizer**对象和**sys**包。我们需要**sys**包来更改打印选项：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    from sklearn.feature_extraction.text import CountVectorizer
    import sys
    ```

1.  通过使用来自**util_simple_classifier.ipynb**文件的函数来加载训练和测试数据框。我们在之前的菜谱中创建了此函数，即*创建简单分类器*。该函数将Rotten Tomatoes数据集的15%加载到**pandas**数据框中，并随机化其顺序。这可能需要几分钟才能运行：

    ```py
    (train_df, test_df) = load_train_test_dataset_pd()
    ```

1.  创建向量器，将其拟合到训练数据上，并打印出结果矩阵。我们将使用**max_df**参数来指定哪些单词应作为停用词。在这种情况下，我们指定在构建向量器时，出现超过40%的文档中的单词应被忽略。你应该进行实验，看看**max_df**的确切值哪个适合你的用例。然后我们将向量器拟合到**train_df**数据框的**text**列：

    ```py
    vectorizer = CountVectorizer(max_df=0.4)
    X = vectorizer.fit_transform(train_df["text"])
    print(X)
    ```

    生成的矩阵是一个`scipy.sparse._csr.csr_matrix`对象，其打印输出的开头如下。稀疏矩阵的格式是`(行, 列) 值`。在我们的例子中，这意味着（文档索引，单词索引）后面跟着频率。在我们的例子中，第一篇评论，即第一篇文档，是文档编号`0`，它包含索引为`6578`、`4219`等的单词。这些单词的频率分别是`1`和`2`。

    ```py
      (0, 6578)  1
      (0, 4219)  1
      (0, 2106)  1
      (0, 8000)  2
      (0, 717)  1
      (0, 42)  1
      (0, 1280)  1
      (0, 5260)  1
      (0, 1607)  1
      (0, 7889)  1
      (0, 3630)  1
    …
    ```

1.  在大多数情况下，我们使用不同的格式来表示向量，这是一种在实际中更容易使用的密集矩阵。我们不是用数字指定行和列，而是从值的位位置推断它们。现在我们将创建一个密集矩阵并打印它：

    ```py
    dense_matrix = X.todense()
    print(dense_matrix)
    ```

    生成的矩阵是一个NumPy矩阵对象，其中每个评论都是一个向量。你可以看到矩阵中的大多数值都是零，正如预期的那样，因为每个评论只使用了一小部分单词，而向量收集了词汇表中的每个单词或所有评论中的每个单词的计数。任何不在向量器词汇表中的单词将不会包含在向量中：

    ```py
    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]]
    ```

1.  我们可以看到文档集中使用的所有单词和词汇表的大小。这可以用作合理性检查，并查看词汇表中是否存在任何不规则性：

    ```py
    print(vectorizer.get_feature_names_out())
    print(len(vectorizer.get_feature_names_out()))
    ```

    结果将如下。如果你想查看完整的、非截断的列表，请使用在*步骤 8*中使用的`set_printoptions`函数：

    ```py
    ['10' '100' '101' ... 'zone' 'ótimo' 'últimos']
    8856
    ```

1.  我们还可以看到向量器使用的所有停用词：

    ```py
    print(vectorizer.stop_words_)
    ```

    结果是三个单词，`and`、`the`和`of`，它们出现在超过40%的评论中：

    ```py
    {'and', 'the', 'of'}
    ```

1.  现在，我们也可以使用**CountVectorizer**对象来表示原始文档集中未出现的新评论。这是在我们有一个训练好的模型并想在新的、未见过的样本上测试它时进行的。我们将使用测试数据集中的第一篇评论。为了获取测试集中的第一篇评论，我们将使用**pandas**的**iat**函数。

    ```py
    first_review = test_df['text'].iat[0]
    print(first_review)
    ```

    第一次审查看起来如下：

    ```py
    lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without stickiness .
    ```

1.  现在，我们将从第一篇评论创建一个稀疏和一个密集向量。向量器的**transform**方法期望一个字符串列表，所以我们将创建一个列表。我们还设置了**print**选项来打印整个向量而不是只打印部分：

    ```py
    sparse_vector = vectorizer.transform([first_review])
    print(sparse_vector)
    dense_vector = sparse_vector.todense()
    np.set_printoptions(threshold=sys.maxsize)
    print(dense_vector)
    np.set_printoptions(threshold=False)
    ```

    稠密向量非常长，大部分是零，正如预期的那样：

    ```py
      (0, 955)  1
      (0, 3968)  1
      (0, 4451)  1
      (0, 4562)  1
      (0, 4622)  1
      (0, 4688)  1
      (0, 4779)  1
      (0, 4792)  1
      (0, 5764)  1
      (0, 7547)  1
      (0, 7715)  1
      (0, 8000)  1
      (0, 8734)  1
    [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    …]]
    ```

1.  我们可以使用不同的方法来计算停用词。在这里，停用词是通过在单词频率上设置绝对阈值来计算的。在这种情况下，我们使用所有在文档中频率低于300的单词。你可以看到，停用词列表现在更大了。

    ```py
    vectorizer = CountVectorizer(max_df=300)
    X = vectorizer.fit_transform(train_df["text"])
    print(vectorizer.stop_words_)
    ```

    结果如下：

    ```py
    {'but', 'this', 'its', 'as', 'to', 'and', 'the', 'is', 'film', 'for', 'it', 'an', 'of', 'that', 'movie', 'with', 'in'}
    ```

1.  最后，我们可以向矢量化器提供自己的停用词列表。这些单词将被矢量化器忽略，不会在矢量化中表示。如果你有非常具体的单词想要忽略，这很有用：

    ```py
    vectorizer = CountVectorizer(stop_words=['the', 'this',
        'these', 'in', 'at', 'for'])
    X = vectorizer.fit_transform(train_df["text"])
    ```

1.  现在，我们将使用我们在上一个配方中定义的函数测试这个词袋矢量化器对简单分类器的影响。首先，我们创建矢量化器，指定只使用在不到80%的文档中出现的单词。然后，我们加载训练和测试数据框。我们在训练集评论上拟合矢量化器。我们使用矢量化器创建一个矢量化函数，并将其传递给**create_train_test_data**函数，同时传递训练和测试数据框。然后我们训练分类器并在测试数据上测试它。我们可以看到，这种方法给我们带来的结果比我们在上一节中使用的基本词性计数矢量化器要好得多：

    ```py
    vectorizer = CountVectorizer(max_df=0.8)
    (train_df, test_df) = load_train_test_dataset_pd()
    X = vectorizer.fit_transform(train_df["text"])
    vectorize = lambda x: vectorizer.transform([x]).toarray()[0]
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    结果将类似于以下内容：

    ```py
                  precision    recall  f1-score   support
               0       0.74      0.72      0.73       160
               1       0.73      0.75      0.74       160
        accuracy                           0.74       320
       macro avg       0.74      0.74      0.74       320
    weighted avg       0.74      0.74      0.74       320
    ```

# 构建N-gram模型

将文档表示为词袋是有用的，但语义不仅仅是关于孤立单词。为了捕捉词组合，使用**n-gram模型**是有用的。其词汇不仅包括单词，还包括单词序列，或*n*-gram。

在这个配方中，我们将构建一个**bigram模型**，其中bigram是两个单词的序列。

## 准备工作

`CountVectorizer`类非常灵活，允许我们构建*n*-gram模型。我们将在此配方中使用它，并用简单的分类器进行测试。

在这个配方中，我将代码及其结果与*将文档放入词袋*配方中的结果进行比较，因为这两个配方非常相似，但它们有一些不同的特性。

## 如何做到这一点...

1.  运行简单的分类器笔记本并导入**CountVectorizer**类：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    from sklearn.feature_extraction.text import CountVectorizer
    ```

1.  使用来自**util_simple_classifier.ipynb**笔记本的代码创建训练和测试数据框：

    ```py
    (train_df, test_df) = load_train_test_dataset_pd()
    ```

1.  创建一个新的矢量化器类。在这种情况下，我们将使用**ngram_range**参数。当设置**ngram_range**参数时，**CountVectorizer**类不仅计算单个单词，还计算单词组合，组合中单词的数量取决于提供给**ngram_range**参数的数字。我们提供了**ngram_range=(1,2)**作为参数，这意味着组合中单词的数量范围从1到2，因此计算单语和双语：

    ```py
    bigram_vectorizer = CountVectorizer(
        ngram_range=(1, 2), max_df=0.8)
    X = bigram_vectorizer.fit_transform(train_df["text"])
    ```

1.  打印矢量化器的词汇及其长度。正如你所见，词汇的长度比单语矢量化器的长度大得多，因为我们除了单字外还使用了双字组合：

    ```py
    print(bigram_vectorizer.get_feature_names_out())
    print(len(bigram_vectorizer.get_feature_names_out()))
    ```

    结果应该看起来像这样：

    ```py
    ['10' '10 inch' '10 set' ... 'ótimo esforço' 'últimos' 'últimos tiempos']
    40552
    ```

1.  现在，我们从测试数据框中取出第一条评论并获取其密集向量。结果看起来与 *将文档放入词袋* 菜谱中的向量输出非常相似，唯一的区别是现在的输出更长，因为它不仅包括单个单词，还包括二元组，即两个单词的序列：

    ```py
    first_review = test_df['text'].iat[0]
    dense_vector = bigram_vectorizer.transform(
        [first_review]).todense()
    print(dense_vector)
    ```

    打印输出看起来像这样：

    ```py
    [[0 0 0 ... 0 0 0]]
    ```

1.  最后，我们使用新的二元向量器训练一个简单的分类器。其结果准确率略低于上一节中使用单语元向量器的分类器的准确率。这可能有几个原因。一个是现在的向量要长得多，而且大部分是零。另一个原因是我们可以看到并非所有评论都是英文的，因此分类器很难泛化输入数据：

    ```py
    vectorize = \
        lambda x: bigram_vectorizer.transform([x]).toarray()[0]
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    输出将如下所示：

    ```py
                  precision    recall  f1-score   support
               0       0.72      0.75      0.73       160
               1       0.74      0.71      0.72       160
        accuracy                           0.73       320
       macro avg       0.73      0.73      0.73       320
    weighted avg       0.73      0.73      0.73       320
    ```

## 更多内容...

我们可以通过提供相应的元组给 `ngram_range` 参数来在向量器中使用三元组、四元组等。这样做的不利之处是词汇表不断扩展，句子向量也在增长，因为每个句子向量都必须为输入词汇表中的每个单词提供一个条目。

也可以使用 `CountVectorizer` 类来表示字符 *n*-gram。在这种情况下，你会计算字符序列的出现次数而不是单词序列。

# 使用 TF-IDF 表示文本

我们可以更进一步，使用 TF-IDF 算法来计算传入文档中的单词和 *n*-gram。**TF-IDF** 代表 **词频-逆文档频率**，它给独特于文档的单词比在整个文档中频繁重复的单词更多的权重。这允许我们给特定文档的独特特征词更多的权重。

在这个菜谱中，我们将使用一种不同类型的向量器，该向量器可以将 TF-IDF 算法应用于输入文本并构建一个小型分类器。

## 准备工作

我们将使用来自 `sklearn` 包的 `TfidfVectorizer` 类。`TfidfVectorizer` 类的特征应该与之前的两个菜谱 *将文档放入词袋* 和 *构建 N-gram 模型* 熟悉。我们将再次使用来自 Hugging Face 的 Rotten Tomatoes 评论数据集。

## 如何实现...

下面是构建和使用 TF-IDF 向量器的步骤：

1.  运行小分类器笔记本并导入 **TfidfVectorizer** 类：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    from sklearn.feature_extraction.text import TfidfVectorizer
    ```

1.  使用 **load_train_test_dataset_pd()** 函数创建训练和测试数据框：

    ```py
    (train_df, test_df) = load_train_test_dataset_pd()
    ```

1.  创建向量器并在训练文本上拟合。我们将使用 **max_df** 参数来排除停用词——在这种情况下，是指比 300 更频繁的单词：

    ```py
    vectorizer = TfidfVectorizer(max_df=300)
    vectorizer.fit(train_df["text"])
    ```

1.  为了确保结果有意义，我们将打印向量器的词汇表及其长度。由于我们只是使用单语元，词汇表的大小应该与词袋菜谱中的相同：

    ```py
    print(vectorizer.get_feature_names_out())
    print(len(vectorizer.get_feature_names_out()))
    ```

    结果应该是这样的。词汇表长度应该与我们在词袋配方中得到的相同，因为我们没有使用*n*-grams：

    ```py
    ['10' '100' '101' ... 'zone' 'ótimo' 'últimos']
    8842
    ```

1.  现在，让我们取测试数据框中的第一个审查并对其进行向量化。然后我们打印密集向量。要了解更多关于稀疏向量和密集向量之间的区别，请参阅*将文档放入词袋*配方。请注意，向量中的值现在是浮点数而不是整数。这是因为单个值现在是比率而不是计数：

    ```py
    first_review = test_df['text'].iat[0]
    dense_vector = vectorizer.transform([first_review]).todense()
    print(dense_vector)
    ```

    结果应该是这样的：

    ```py
    [[0\. 0\. 0\. ... 0\. 0\. 0.]]
    ```

1.  现在，让我们训练分类器。我们可以看到，分数略高于词袋分类器的分数，无论是单词还是*n*-gram版本：

    ```py
    vectorize = lambda x: vectorizer.transform([x]).toarray()[0]
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    测试分数的打印输出将类似于以下内容：

    ```py
                  precision    recall  f1-score   support
               0       0.76      0.72      0.74       160
               1       0.74      0.78      0.76       160
        accuracy                           0.75       320
       macro avg       0.75      0.75      0.75       320
    weighted avg       0.75      0.75      0.75       320
    ```

## 它是如何工作的……

`TfidfVectorizer`类几乎与`CountVectorizer`类完全相同，只是在计算**词频**的方式上有所不同，所以大多数步骤应该是熟悉的。词频是单词在文档中出现的次数。逆文档频率是包含该单词的文档总数除以文档数。通常，这些频率是按对数缩放的。

这是通过以下公式完成的：

![<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mrow><mi>T</mi><mi>F</mi><mo>=</mo><mfrac><mrow><mi>N</mi><mi>u</mi><mi>m</mi><mi>b</mi><mi>e</mi><mi>r</mi><mi>o</mi><mi>f</mi><mi>t</mi><mi>i</mi><mi>m</mi><mi>e</mi><mi>s</mi><mi>a</mi><mi>t</mi><mi>e</mi><mi>r</mi><mi>r</mi><mi>m</mi><mi>a</mi><mi>p</mi><mi>p</mi><mi>e</mi><mi>a</mi><mi>r</mi><mi>s</mi><mi>i</mi><mi>n</mi><mi>t</mi><mi>h</mi><mi>e</mi><mi>d</mi><mi>o</mi><mi>c</mi><mi>u</mi><mi>m</mi><mi>e</mi><mi>n</mi><mi>t</mi></mrow><mrow><mi>T</mi><mi>o</mi><mi>t</mi><mi>a</mi><mi>l</mi><mi>n</mi><mi>u</mi><mi>m</mi><mi>b</mi><mi>e</mi><mi>r</mi><mi>o</mi><mi>f</mi><mi>w</mi><mi>o</mi><mi>r</mi><mi>d</mi><mi>s</mi><mi>i</mi><mi>n</mi><mi>t</mi><mi>h</mi><mi>e</mi><mi>d</mi><mi>o</mi><mi>c</mi><mi>u</mi><mi>m</mi><mi>e</mi><mi>n</mi><mi>t</mi></mrow></mfrac></mrow></mrow></math>](img/1.png)

![<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mrow><mrow><mi>I</mi><mi>D</mi><mi>F</mi><mo>=</mo><mfrac><mrow><mi>T</mi><mi>o</mi><mi>t</mi><mi>a</mi><mi>l</mi><mi>n</mi><mi>u</mi><mi>m</mi><mi>b</mi><mi>e</mi><mi>r</mi><mi>o</mi><mi>f</mi><mi>d</mi><mi>o</mi><mi>c</mi><mi>u</mi><mi>m</mi><mi>e</mi><mi>n</mi><mi>t</mi><mi>s</mi></mrow><mrow><mi>N</mi><mi>u</mi><mi>m</mi><mi>b</mi><mi>e</mi><mi>r</mi><mi>o</mi><mi>f</mi><mi>d</mi><mi>o</mi><mi>c</mi><mi>u</mi><mi>m</mi><mi>e</mi><mi>n</mi><mi>t</mi><mi>s</mi><mi>w</mi><mi>h</mi><mi>e</mi><mi>r</mi><mi>e</mi><mi>t</mi><mi>e</mi><mi>r</mi><mi>r</mi><mi>m</mi><mi>o</mi><mi>c</mi><mi>u</mi><mi>r</mi><mi>s</mi></mrow></mfrac></mrow></mrow></math>](img/2.png)

![<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" display="block"><mml:mi>T</mml:mi><mml:mi>F</mml:mi><mml:mo>-</mml:mo><mml:mi>I</mml:mi><mml:mi>D</mml:mi><mml:mi>F</mml:mi><mml:mo>=</mml:mo><mml:mi>T</mml:mi><mml:mi>F</mml:mi><mml:mi>*</mml:mi><mml:mi>I</mml:mi><mml:mi>D</mml:mi><mml:mi>F</mml:mi></mml:math>](img/3.png)

## 还有更多…

我们可以构建 `TfidfVectorizer` 并使用 `[t, h, e, w, o, m, a, n, th, he, wo, om, ma, an, the, wom, oma, man]` 集合。在一些实验设置中，基于字符 *n*-gram 的模型比基于单词的 *n*-gram 模型表现更好。

我们将使用小型的夏洛克·福尔摩斯文本文件，`sherlock_holmes_1.txt`，位于 [https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/sherlock_holmes_1.txt)，以及相同的类，`TfidfVectorizer`。由于分析的单位是字符而不是单词，我们不需要标记化函数或停用词列表。创建向量器和分析句子的步骤如下：

1.  创建一个新的使用 **char_wb** 分析器的向量器对象，然后将其拟合到训练文本上：

    ```py
    tfidf_char_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(1,5))
    tfidf_char_vectorizer = tfidf_char_vectorizer.fit(
        train_df["text"])
    ```

1.  打印向量器的词汇表及其长度：

    ```py
    print(list(tfidf_char_vectorizer.get_feature_names_out()))
    print(len(tfidf_char_vectorizer.get_feature_names_out()))
    ```

    部分结果将看起来像这样：

    ```py
    [' ', ' !', ' ! ', ' "', ' " ', ' $', ' $5', ' $50', ' $50-', ' $9', ' $9 ', ' &', ' & ', " '", " ' ", " '5", " '50", " '50'", " '6", " '60", " '60s", " '7", " '70", " '70'", " '70s", " '[", " '[h", " '[ho", " 'a", " 'a ", " 'a'", " 'a' ", " 'ab", " 'aba", " 'ah", " 'ah ", " 'al", " 'alt", " 'an", " 'ana", " 'ar", " 'are", " 'b", " 'ba", " 'bar", " 'be", " 'bee", " 'bes", " 'bl", " 'blu", " 'br", " 'bra", " 'bu", " 'but", " 'c", " 'ch", " 'cha", " 'co", " 'co-", " 'com", " 'd", " 'di", " 'dif", " 'do", " 'dog", " 'du", " 'dum", " 'e", " 'ed", " 'edg", " 'em", " 'em ", " 'ep", " 'epi", " 'f", " 'fa", " 'fac", " 'fat", " 'fu", " 'fun", " 'g", " 'ga", " 'gar", " 'gi", " 'gir", " 'gr", " 'gra", " 'gu", " 'gue", " 'guy", " 'h", " 'ha", " 'hav", " 'ho", " 'hos", " 'how", " 'i", " 'i ", " 'if", " 'if ", " 'in", " 'in ", " 'is",…]
    51270
    ```

1.  使用新的向量器创建 **vectorize** 方法，然后创建训练数据和测试数据。训练分类器然后测试它：

    ```py
    vectorize = lambda x: tfidf_char_vectorizer.transform([
        x]).toarray()[0]
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    结果将类似于以下内容：

    ```py
                  precision    recall  f1-score   support
               0       0.74      0.74      0.74       160
               1       0.74      0.74      0.74       160
        accuracy                           0.74       320
       macro avg       0.74      0.74      0.74       320
    weighted avg       0.74      0.74      0.74       320
    ```

## 参见

+   你可以在 [https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting) 了解更多关于词频逆文档频率（TF-IDF）的词权重信息

+   更多关于 **TfidfVectorizer** 的信息，请参阅 [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

# 使用词嵌入

在这个菜谱中，我们将转换方向，学习如何使用词嵌入来表示*words*，这是因为它们是训练一个预测句子中所有其他单词的神经网络的产物。嵌入也是向量，但通常大小要小得多，200或300。结果向量嵌入对于在相似上下文中出现的单词是相似的。相似度通常通过计算超平面中两个向量之间角度的余弦值来衡量，维度为200或300。我们将使用嵌入来展示这些相似性。

## 准备工作

在这个菜谱中，我们将使用预训练的`word2vec`模型，该模型可在[https://github.com/mmihaltz/word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)找到。下载模型并将其解压缩到数据目录中。现在你应该有一个路径为`…/``data/GoogleNews-vectors-negative300.bin.gz`的文件。

我们还将使用`gensim`包来加载和使用模型。它应该在`poetry`环境中安装。

笔记本位于[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter03/3.5_word_embeddings.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter03/3.5_word_embeddings.ipynb)。

## 如何做到这一点...

我们将加载模型，演示`gensim`包的一些功能，然后使用词嵌入计算一个句子向量：

1.  运行简单的分类器文件：

    ```py
    %run -i "../util/simple_classifier.ipynb"
    ```

1.  导入**gensim**包：

    ```py
    import gensim
    ```

1.  加载预训练模型。如果在这一步出现错误，请确保您已将模型下载到**data**目录中：

    ```py
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '../data/GoogleNews-vectors-negative300.bin.gz',
        binary=True)
    ```

1.  使用预训练模型，我们现在可以加载单个词向量。在这里，我们加载单词*king*的词向量。我们必须将其转换为小写，因为模型中的所有单词都是小写的。结果是表示该单词在**word2vec**模型中的长向量：

    ```py
    vec_king = model['king']
    print(vec_king)
    ```

    结果将如下所示：

    ```py
    [ 1.25976562e-01  2.97851562e-02  8.60595703e-03  1.39648438e-01
     -2.56347656e-02 -3.61328125e-02  1.11816406e-01 -1.98242188e-01
      5.12695312e-02  3.63281250e-01 -2.42187500e-01 -3.02734375e-01
     -1.77734375e-01 -2.49023438e-02 -1.67968750e-01 -1.69921875e-01
      3.46679688e-02  5.21850586e-03  4.63867188e-02  1.28906250e-01
      1.36718750e-01  1.12792969e-01  5.95703125e-02  1.36718750e-01
      1.01074219e-01 -1.76757812e-01 -2.51953125e-01  5.98144531e-02
      3.41796875e-01 -3.11279297e-02  1.04492188e-01  6.17675781e-02  …]
    ```

1.  我们还可以获取与给定单词最相似的单词。例如，让我们打印出与*apple*和*tomato*最相似的单词。输出将打印出最相似的单词（即出现在相似上下文中）及其相似度分数。分数是两个向量之间的余弦距离——在这种情况下，表示一对单词。分数越大，两个单词越相似。结果是有意义的，因为与*apple*最相似的单词大多是水果，与*tomato*最相似的单词大多是蔬菜：

    ```py
    print(model.most_similar(['apple'], topn=15))
    print(model.most_similar(['tomato'], topn=15))
    ```

    结果如下所示：

    ```py
    [('apples', 0.720359742641449), ('pear', 0.6450697183609009), ('fruit', 0.6410146355628967), ('berry', 0.6302295327186584), ('pears', 0.613396167755127), ('strawberry', 0.6058260798454285), ('peach', 0.6025872826576233), ('potato', 0.5960935354232788), ('grape', 0.5935863852500916), ('blueberry', 0.5866668224334717), ('cherries', 0.5784382820129395), ('mango', 0.5751855969429016), ('apricot', 0.5727777481079102), ('melon', 0.5719985365867615), ('almond', 0.5704829692840576)]
    [('tomatoes', 0.8442263007164001), ('lettuce', 0.7069936990737915), ('asparagus', 0.7050934433937073), ('peaches', 0.6938520669937134), ('cherry_tomatoes', 0.6897529363632202), ('strawberry', 0.6888598799705505), ('strawberries', 0.6832595467567444), ('bell_peppers', 0.6813562512397766), ('potato', 0.6784172058105469), ('cantaloupe', 0.6780219078063965), ('celery', 0.675195574760437), ('onion', 0.6740139722824097), ('cucumbers', 0.6706333160400391), ('spinach', 0.6682621240615845), ('cauliflower', 0.6681587100028992)]
    ```

1.  在接下来的两个步骤中，我们通过平均句子中的所有词向量来计算一个句子向量。这种方法的一个挑战是表示模型中不存在的词，在这里，我们简单地跳过这些词。让我们定义一个函数，它将接受一个句子和一个模型，并返回句子词向量的列表。如果模型中不存在词，将返回**KeyError**，在这种情况下，我们捕获错误并继续：

    ```py
    def get_word_vectors(sentence, model):
        word_vectors = []
        for word in sentence:
            try:
                word_vector = model[word.lower()]
                word_vectors.append(word_vector)
            except KeyError:
                continue
        return word_vectors
    ```

1.  现在，让我们定义一个函数，它将接受词向量列表并计算句子向量。为了计算平均值，我们将矩阵表示为一个**numpy**数组，并使用**numpy**的**mean**函数来获取平均向量：

    ```py
    def get_sentence_vector(word_vectors):
        matrix = np.array(word_vectors)
        centroid = np.mean(matrix[:,:], axis=0)
        return centroid
    ```

注意

通过平均词向量来获取句子向量是处理这个任务的一种方法，但并非没有问题。另一种选择是训练一个**doc2vec**模型，其中句子、段落和整个文档都可以作为单位，而不是词。

1.  我们现在可以测试平均词嵌入作为向量器。我们的向量器接受字符串输入，获取每个词的词向量，然后返回我们在**get_sentence_vector**函数中计算的句子向量。然后我们加载训练数据和测试数据，创建数据集。我们训练逻辑回归分类器并对其进行测试：

    ```py
    vectorize = lambda x: get_sentence_vector(
        get_word_vectors(x, model))
    (train_df, test_df) = load_train_test_dataset_pd()
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    我们可以看到，分数比前几节低得多。这可能有几个原因；其中之一是`word2vec`模型仅支持英语，而数据是多语言的。作为一个练习，你可以编写一个脚本来过滤仅支持英语的评论，看看是否可以提高分数：

    ```py
                  precision    recall  f1-score   support
               0       0.54      0.57      0.55       160
               1       0.54      0.51      0.53       160
        accuracy                           0.54       320
       macro avg       0.54      0.54      0.54       320
    weighted avg       0.54      0.54      0.54       320
    ```

## 更多内容…

`gensim`使用预训练模型可以做很多有趣的事情。例如，它可以从一个词表中找到一个异常词，并找到与给定词最相似的词。让我们看看这些：

1.  编译一个包含不匹配词的词表，将**doesnt_match**函数应用于该列表，并打印结果：

    ```py
    words = ['banana', 'apple', 'computer', 'strawberry']
    print(model.doesnt_match(words))
    ```

    结果将如下所示：

    ```py
    computer
    ```

1.  现在，让我们找到一个与另一个词最相似的词。

    ```py
    word = "cup"
    words = ['glass', 'computer', 'pencil', 'watch']
    print(model.most_similar_to_given(word, words))
    ```

    结果将如下所示：

    ```py
    glass
    ```

## 参见

+   有许多其他预训练模型可供选择，包括一些其他语言的模型；参见[http://vectors.nlpl.eu/repository/](http://vectors.nlpl.eu/repository/)。

    一些预训练模型包括词性信息，这在区分词时可能很有帮助。这些模型将词与其`cat_NOUN`等属性连接起来，所以使用它们时请记住这一点。

+   要了解更多关于**word2vec**背后的理论，你可以从这里开始：[https://jalammar.github.io/illustrated-word2vec/](https://jalammar.github.io/illustrated-word2vec/)。

# 训练自己的嵌入模型

我们现在可以在语料库上训练自己的 `word2vec` 模型。这个模型是一个神经网络，当给定一个带有空格的句子时，可以预测一个单词。神经网络训练的副产品是训练词汇表中每个单词的向量表示。对于这个任务，我们将继续使用 Rotten Tomatoes 评论。数据集不是很大，所以结果不如拥有更大集合时那么好。

## 准备工作

我们将使用 `gensim` 包来完成这个任务。它应该作为 `poetry` 环境的一部分安装。

## 如何做到这一点...

我们将创建数据集，然后在数据上训练模型。然后我们将测试其性能：

1.  导入必要的包和函数：

    ```py
    import gensim
    from gensim.models import Word2Vec
    from datasets import load_dataset
    from gensim import utils
    ```

1.  加载训练数据并检查其长度：

    ```py
    train_dataset = load_dataset("rotten_tomatoes", split="train")
    print(len(train_dataset))
    ```

    结果应该是这样的：

    ```py
    8530
    ```

1.  创建 **RottenTomatoesCorpus** 类。**word2vec** 训练算法需要一个具有定义的 **__iter__** 函数的类，这样你就可以遍历数据，这就是为什么我们需要这个类的原因：

    ```py
    class RottenTomatoesCorpus:
        def __init__(self, sentences):
            self.sentences = sentences
        def __iter__(self):
            for review in self.sentences:
                yield utils.simple_preprocess(
                    gensim.parsing.preprocessing.remove_stopwords(
                        review))
    ```

1.  使用加载的训练数据集创建一个 **RottenTomatoesCorpus** 实例。由于 **word2vec** 模型仅在文本上训练（它们是自监督模型），我们不需要评论评分：

    ```py
    sentences = train_dataset["text"]
    corpus = RottenTomatoesCorpus(sentences)
    ```

1.  在这个步骤中，我们初始化 **word2vec** 模型，训练它，然后将其保存到磁盘。唯一必需的参数是单词列表；其他一些重要的参数是 **min_count**、**size**、**window** 和 **workers**。**min_count** 参数指的是一个单词在训练语料库中必须出现的最小次数，默认值为 5。**size** 参数设置单词向量的大小。**window** 限制了句子中预测单词和当前单词之间的最大单词数。**workers** 指的是工作线程的数量；线程越多，训练速度越快。在训练模型时，**epoch** 参数将确定模型将经历的训练迭代次数。在初始化模型对象后，我们在语料库上训练它 100 个 epoch，最后将其保存到磁盘：

    ```py
    model = Word2Vec(sentences=corpus, vector_size=100,
        window=5, min_count=1, workers=4)
    model.train(corpus_iterable=corpus,
        total_examples=model.corpus_count, epochs=100)
    model.save("../data/rotten_tomato_word2vec.model")
    ```

1.  找出与单词 *movie* 相似的 10 个单词。单词 *sequels* 和 *film* 与这个单词搭配合理；其余的则不太相关。这是因为训练语料库的规模较小。你得到的结果将会有所不同，因为每次训练模型时结果都会不同：

    ```py
    w1 = "movie"
    words = model.wv.most_similar(w1, topn=10)
    print(words)
    ```

    这是一个可能的结果：

    ```py
    [('sequels', 0.38357362151145935), ('film', 0.33577531576156616), ('stuffed', 0.2925359606742859), ('quirkily', 0.28789234161376953), ('convict', 0.2810690104961395), ('worse', 0.2789292335510254), ('churn', 0.27702808380126953), ('hellish', 0.27698105573654175), ('hey', 0.27566075325012207), ('happens', 0.27498629689216614)]
    ```

## 还有更多...

有工具可以评估 `word2vec` 模型，尽管其创建是无监督的。`gensim` 包含一个文件，列出了单词类比，例如 *Athens* 对 *Greece* 的关系与 *Moscow* 对 *Russia* 的关系相同。`evaluate_word_analogies` 函数将类比通过模型运行，并计算正确答案的数量。

这里是如何做到这一点的：

1.  使用**evaluate_word_analogies**函数评估我们的训练模型。我们需要**类比**文件，该文件可在GitHub存储库的书中找到，地址为[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/questions-words.txt](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/questions-words.txt)。

    ```py
    (analogy_score, word_list) = model.wv.evaluate_word_analogies(
        '../data/questions-words.txt')
    print(analogy_score)
    ```

    结果应该类似于以下内容：

    ```py
    0.0015881418740074113
    ```

1.  现在我们来评估预训练模型。这些命令可能需要更长的时间来运行：

    ```py
    pretrained_model = \
        gensim.models.KeyedVectors.load_word2vec_format(
            '../data/GoogleNews-vectors-negative300.bin.gz',
            binary=True)
    (analogy_score, word_list) = \
        pretrained_model.evaluate_word_analogies(
            '../data/questions-words.txt')
    print(analogy_score)
    ```

    结果应该类似于以下内容：

    ```py
    0.7401448525607863
    ```

1.  我们在预训练模型和我们的模型案例中使用了不同的**evaluate_word_analogies**函数，因为它们是不同类型的。对于预训练模型，我们只需加载向量（一个**KeyedVectors**类，其中每个由键表示的单词都映射到一个向量），而我们的模型是一个完整的**word2vec**模型对象。我们可以使用以下命令来检查类型：

    ```py
    print(type(pretrained_model))
    print(type(model))
    ```

    结果将如下所示：

    ```py
    <class 'gensim.models.keyedvectors.KeyedVectors'>
    <class 'gensim.models.word2vec.Word2Vec'>
    ```

    预训练模型是在一个更大的语料库上训练的，因此，预测地，它的表现更好。您也可以构建自己的评估文件，其中包含您数据所需的概念。

注意

确保您的评估基于您将在应用程序中使用的类型的数据；否则，您可能会得到误导性的评估结果。

## 参考信息

有一种额外的评估模型性能的方法，即通过比较模型分配给单词对的相似度与人类分配的判断之间的相似度。您可以通过使用`evaluate_word_pairs`函数来完成此操作。更多信息请参阅[https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_pairs](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_pairs)。

# 使用BERT和OpenAI嵌入而不是词嵌入

我们可以使用**双向编码器表示从Transformer**（**BERT**）嵌入而不是词嵌入。BERT模型，就像词嵌入一样，是一个预训练模型，它给出一个向量表示，但它考虑上下文，可以表示整个句子而不是单个单词。

## 准备工作

对于这个食谱，我们可以使用Hugging Face的`sentence_transformers`包将句子表示为向量。我们需要`PyTorch`，它是作为`poetry`环境的一部分安装的。

为了获取向量，我们将使用`all-MiniLM-L6-v2`模型来完成这个食谱。

我们还可以使用来自OpenAI的**大型语言模型**（**LLMs**）的嵌入。

要使用OpenAI嵌入，您需要创建一个账户并从OpenAI获取API密钥。您可以在[https://platform.openai.com/signup](https://platform.openai.com/signup)创建账户。

笔记本位于 [https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter03/3.6_train_own_word2vec.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter03/3.6_train_own_word2vec.ipynb)。

## 如何做到这一点…

Hugging Face 代码使使用 BERT 非常容易。第一次运行代码时，它将下载必要的模型，这可能需要一些时间。下载后，只需使用模型对句子进行编码即可。我们将使用这些嵌入测试简单的分类器：

1.  运行简单的分类器笔记本以导入其函数：

    ```py
    %run -i "../util/util_simple_classifier.ipynb"
    ```

1.  导入 **SentenceTransformer** 类：

    ```py
    from sentence_transformers import SentenceTransformer
    ```

1.  加载句子转换器模型，检索句子 *我爱爵士* 的嵌入，并打印出来。

    ```py
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(["I love jazz"])
    print(embedding)
    ```

    如我们所见，它是一个与之前菜谱中的词嵌入向量相似的向量：

    ```py
    [[ 2.94217980e-03 -7.93536603e-02 -2.82228496e-02 -5.13779782e-02
      -6.44981042e-02  9.83557850e-02  1.09671958e-01 -3.26390602e-02
       4.96566631e-02  2.56580133e-02 -1.08482063e-01  1.88441798e-02
       2.70963665e-02 -3.80690470e-02  2.42502335e-02 -3.65605950e-03
       1.29364491e-01  4.32255343e-02 -6.64561391e-02 -6.93060979e-02
      -1.39410645e-01  4.36719768e-02 -7.85463024e-03  1.68625098e-02
      -1.01160072e-02  1.07926019e-02 -1.05814040e-02  2.57284809e-02
      -1.51516097e-02 -4.53920700e-02  7.12087378e-03  1.17573030e-01… ]]
    ```

1.  现在，我们可以使用 BERT 嵌入来测试我们的分类器。首先，让我们定义一个函数，该函数将返回一个句子向量。这个函数接受输入文本和一个模型。然后，它使用该模型对文本进行编码，并返回结果嵌入。我们需要将文本放入列表中传递给 **encode** 方法，因为它期望一个可迭代对象。同样，我们返回结果中的第一个元素，因为它返回一个嵌入列表。

    ```py
    def get_sentence_vector(text, model):
        sentence_embeddings = model.encode([text])
        return sentence_embeddings[0]
    ```

1.  现在，我们定义 **vectorize** 函数，使用我们创建在 *创建简单分类器* 菜谱中的 **load_train_test_dataset_pd** 函数创建训练和测试数据，训练分类器，并对其进行测试。我们将计时数据集创建步骤，因此包含了 **time** 包命令。我们看到整个数据集（约 85,000 条记录）的向量化大约需要 11 秒。然后我们训练模型并对其进行测试：

    ```py
    import time
    vectorize = lambda x: get_sentence_vector(x, model)
    (train_df, test_df) = load_train_test_dataset_pd()
    start = time.time()
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    print(f"BERT embeddings: {time.time() - start} s")
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    结果是我们迄今为止最好的结果：

    ```py
    BERT embeddings: 11.410213232040405 s
                  precision    recall  f1-score   support
               0       0.77      0.79      0.78       160
               1       0.79      0.76      0.77       160
        accuracy                           0.78       320
       macro avg       0.78      0.78      0.78       320
    weighted avg       0.78      0.78      0.78       320
    ```

## 更多内容…

我们现在可以使用 OpenAI 嵌入来查看它们的性能：

1.  导入 **openai** 包并分配 API 密钥：

    ```py
    import openai
    openai.api_key = OPEN_AI_KEY
    ```

1.  分配我们将使用的模型、句子和创建嵌入。我们将使用的模型是一个特定的嵌入模型，因此它为文本输入返回一个嵌入向量：

    ```py
    model = "text-embedding-ada-002"
    text = "I love jazz"
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    embeddings = response['data'][0]['embedding']
    print(embeddings)
    ```

    部分结果将如下所示：

    ```py
    [-0.028350897133350372, -0.011136125773191452, -0.0021299426443874836, -0.014453398995101452, -0.012048527598381042, 0.018223850056529045, -0.010247894562780857, -0.01806674897670746, -0.014308380894362926, 0.0007220656843855977, -9.998268797062337e-05, 0.010078707709908485,…]
    ```

1.  现在，让我们使用 OpenAI 嵌入来测试我们的分类器。这是将返回句子向量的函数：

    ```py
    def get_sentence_vector(text, model):
        text = "I love jazz"
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        embeddings = response['data'][0]['embedding']
        return embeddings
    ```

1.  现在，定义 **vectorize** 函数，创建训练和测试数据，训练分类器，并对其进行测试。我们将计时向量化步骤：

    ```py
    import time
    vectorize = lambda x: get_sentence_vector(x, model)
    (train_df, test_df) = load_train_test_dataset_pd()
    start = time.time()
    (X_train, X_test, y_train, y_test) = create_train_test_data(
        train_df, test_df, vectorize)
    print(f"OpenAI embeddings: {time.time() - start} s")
    clf = train_classifier(X_train, y_train)
    test_classifier(test_df, clf)
    ```

    结果将如下所示：

    ```py
    OpenAI embeddings: 704.3250799179077 s
                  precision    recall  f1-score   support
               0       0.49      0.82      0.62       160
               1       0.47      0.16      0.23       160
        accuracy                           0.49       320
       macro avg       0.48      0.49      0.43       320
    weighted avg       0.48      0.49      0.43       320
    ```

    注意，从得分来看，结果相当差，处理整个数据集需要超过 10 分钟。在这里，我们只使用了 LLM 嵌入，并在这些嵌入上训练了一个逻辑回归分类器。这与使用 LLM 本身进行分类不同。

## 参见

更多预训练模型，请参阅 [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)。

# 检索增强生成（RAG）

在这个示例中，我们将看到向量嵌入的实际应用。RAG 是一种流行的处理大型语言模型（LLM）的方法。由于这些模型是在广泛可用的互联网数据上预训练的，因此它们无法访问我们的个人数据，我们也不能直接使用该模型来对其提问。一种克服这一限制的方法是使用向量嵌入来表示我们的数据。然后，我们可以计算我们的数据与问题之间的余弦相似度，并将最相似的数据片段连同问题一起包含在内——这就是“检索增强生成”这个名字的由来，因为我们首先通过余弦相似度检索相关数据，然后使用大型语言模型生成文本。

## 准备工作

我们将使用来自 **Kaggle** 的 IMDB 数据集，该数据集可以从 [https://www.kaggle.com/PromptCloudHQ/imdb-data](https://www.kaggle.com/PromptCloudHQ/imdb-data) 下载，也包含在本书的 GitHub 仓库中 [https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/IMDB-Movie-Data.csv](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/data/IMDB-Movie-Data.csv)。下载数据集并解压 CSV 文件。

我们还将使用 OpenAI 嵌入，以及包含在 `poetry` 环境中的 `llama_index` 包。

笔记本位于 [https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter03/3.9_vector_search.ipynb](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/blob/main/Chapter03/3.9_vector_search.ipynb)。

## 如何操作…

我们将加载 IMDB 数据集，然后使用其前10个条目创建一个向量存储，然后使用 `llama_index` 包查询向量存储：

1.  运行 **utilities** 笔记本：

    ```py
    %run -i "../util/file_utils.ipynb"
    ```

1.  导入必要的类和包：

    ```py
    import csv
    import openai
    from llama_index import VectorStoreIndex
    from llama_index import Document
    openai.api_key = OPEN_AI_KEY
    ```

1.  读取 CSV 数据。我们将跳过数据的第一行标题：

    ```py
    with open('../data/IMDB-Movie-Data.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        movies = data[1:]
    ```

1.  在这一步，我们使用刚刚读取的数据的前10行来首先创建一个**Document**对象列表，然后创建一个包含这些**Document**对象的**VectorStoreIndex**对象。索引是一个用于搜索的对象，其中每个记录包含某些信息。向量存储索引存储每个记录的元数据以及向量表示。对于每部电影，我们将描述作为将被嵌入的文本，其余部分作为元数据。我们打印出**document**对象，可以看到每个对象都被分配了一个唯一的ID：

    ```py
    documents = []
    for movie in movies[0:10]:
        doc_id = movie[0]
        title = movie[1]
        genres = movie[2].split(",")
        description = movie[3]
        director = movie[4]
        actors = movie[5].split(",")
        year = movie[6]
        duration = movie[7]
        rating = movie[8]
        revenue = movie[10]
        document = Document(
            text=description,
            metadata={
                "title": title,
                "genres": genres,
                "director": director,
                "actors": actors,
                "year": year,
                "duration": duration,
                "rating": rating,
                "revenue": revenue
            }
        )
        print(document)
        documents.append(document)
    index = VectorStoreIndex.from_documents(documents)
    ```

    部分输出将类似于以下内容：

    ```py
    id_='6e1ef633-f10b-44e3-9b77-f5f7b08dcedd' embedding=None metadata={'title': 'Guardians of the Galaxy', 'genres': ['Action', 'Adventure', 'Sci-Fi'], 'director': 'James Gunn', 'actors': ['Chris Pratt', ' Vin Diesel', ' Bradley Cooper', ' Zoe Saldana'], 'year': '2014', 'duration': '121', 'rating': '8.1', 'revenue': '333.13'} excluded_embed_metadata_keys=[] excluded_llm_metadata_keys=[] relationships={} hash='e18bdce3a36c69d8c1e55a7eb56f05162c68c97151cbaf40
    91814ae3df42dfe8' text='A group of intergalactic criminals are forced to work together to stop a fanatical warrior from taking control of the universe.' start_char_idx=None end_char_idx=None text_template='{metadata_str}\n\n{content}' metadata_template='{key}: {value}' metadata_seperator='\n'
    ```

1.  从我们刚刚创建的索引中创建查询引擎。查询引擎将允许我们向索引中加载的文档发送问题：

    ```py
    query_engine = index.as_query_engine()
    ```

1.  使用引擎回答问题：

    ```py
    response = query_engine.query("""Which movies talk about something gigantic?""")
    print(response.response)
    The answer seems to make sense grammatically, and arguably the Great Wall of China is gigantic. However, it is not clear what is gigantic in the movie Prometheus. So here we have a partially correct answer. The Great Wall and Prometheus both talk about something gigantic. In The Great Wall, the protagonists become embroiled in the defense of the Great Wall of China against a horde of monstrous creatures. In Prometheus, the protagonists find a structure on a distant moon.
    ```
