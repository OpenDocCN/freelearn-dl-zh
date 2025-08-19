# 第六章. 字符串比较与聚类

本章将涵盖以下几种方案：

+   距离和接近度 – 简单编辑距离

+   加权编辑距离

+   Jaccard 距离

+   Tf-Idf 距离

+   使用编辑距离和语言模型进行拼写纠正

+   大小写恢复修正器

+   自动短语完成

+   使用编辑距离的单链和完全链接聚类

+   潜在狄利克雷分配（LDA）用于多主题聚类

# 介绍

本章从使用标准的语言中立技术来比较字符串开始。然后，我们将使用这些技术构建一些常用的应用程序。我们还将探讨基于字符串之间距离的聚类技术。

对于字符串，我们使用标准定义，即字符串是字符的序列。所以，显然，这些技术适用于单词、短语、句子、段落等，你在前几章中已经学会了如何提取这些内容。

# 距离和接近度 – 简单编辑距离

字符串比较是指用于衡量两个字符串相似度的技术。我们将使用距离和接近度来指定任意两个字符串的相似性。两个字符串的相似性越高，它们之间的距离就越小，因此，一个字符串与自身的距离为 0。相反的度量是接近度，意味着两个字符串越相似，它们的接近度就越大。

我们将首先看看简单编辑距离。简单编辑距离通过衡量将一个字符串转换为另一个字符串所需的编辑次数来计算距离。Levenshtein 在 1965 年提出的一种常见距离度量允许删除、插入和替换作为基本操作。加入字符交换后就称为 Damerau-Levenshtein 距离。例如，`foo`和`boo`之间的距离为 1，因为我们是在将`f`替换为`b`。

### 注意

有关距离度量的更多信息，请参考维基百科上的[距离](http://en.wikipedia.org/wiki/Distance)文章。

让我们看一些可编辑操作的更多示例：

+   **删除**：`Bart`和`Bar`

+   **插入**：`Bar`和`Bart`

+   **替换**：`Bar`和`Car`

+   **字符交换**：`Bart`和`Brat`

## 如何做到...

现在，我们将运行一个关于编辑距离的简单示例：

1.  使用命令行或你的 IDE 运行`SimpleEditDistance`类：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter6.SimpleEditDistance

    ```

1.  在命令提示符下，系统将提示你输入两个字符串：

    ```py
    Enter the first string:
    ab
    Enter the second string:
    ba
    Allowing Transposition Distance between: ab and ba is 1.0
    No Transposition Distance between: ab and ba is 2.0

    ```

1.  你将看到允许字符交换和不允许字符交换情况下两个字符串之间的距离。

1.  多做一些示例来感受它是如何工作的——先手动尝试，然后验证你是否得到了最优解。

## 它是如何工作的...

这是一段非常简单的代码，所做的只是创建两个`EditDistance`类的实例：一个允许字符交换，另一个不允许字符交换：

```py
public static void main(String[] args) throws IOException {

  EditDistance dmAllowTrans = new EditDistance(true);
  EditDistance dmNoTrans = new EditDistance(false);
```

剩余的代码将设置输入/输出路由，应用编辑距离并输出结果：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.println("Enter the first string:");
  String text1 = reader.readLine();
  System.out.println("Enter the second string:");
  String text2 = reader.readLine();
  double allowTransDist = dmAllowTrans.distance(text1, text2);
  double noTransDist = dmNoTrans.distance(text1, text2);
  System.out.println("Allowing Transposition Distance " +" between: " + text1 + " and " + text2 + " is " + allowTransDist);
  System.out.println("No Transposition Distance between: " + text1 + " and " + text2 + " is " + noTransDist);
}
}
```

如果我们想要的是接近度而不是距离，我们只需使用`proximity`方法，而不是`distance`方法。

在简单的`EditDistance`中，所有可编辑的操作都有固定的成本 1.0，也就是说，每个可编辑的操作（删除、替换、插入，以及如果允许的话，交换）都被计为成本 1.0。因此，在我们计算`ab`和`ba`之间的距离时，有一个删除操作和一个插入操作，两个操作的成本都是 1.0。因此，如果不允许交换，`ab`和`ba`之间的距离为 2.0；如果允许交换，则为 1.0。请注意，通常将一个字符串编辑成另一个字符串的方法不止一种。

### 注意

虽然`EditDistance`使用起来非常简单，但实现起来却并不容易。关于这个类，Javadoc 是这么说的：

*实现说明：该类使用动态规划实现编辑距离，时间复杂度为 O(n * m)，其中 n 和 m 是正在比较的两个序列的长度。通过使用三个格子片段的滑动窗口，而不是一次性分配整个格子所需的空间，仅为三个整数数组的空间，长度为两个字符序列中较短的那个。*

在接下来的章节中，我们将看到如何为每种编辑操作分配不同的成本。

## 另见

+   更多详情请参阅 LingPipe Javadoc 中的`EditDistance`：[`alias-i.com/lingpipe/docs/api/com/aliasi/spell/EditDistance.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/spell/EditDistance.html)

+   更多关于距离的详情，请参阅 Javadoc：[`alias-i.com/lingpipe/docs/api/com/aliasi/util/Distance.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/util/Distance.html)

+   更多关于接近度的详情，请参阅 Javadoc：[`alias-i.com/lingpipe/docs/api/com/aliasi/util/Proximity.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/util/Proximity.html)

# 加权编辑距离

加权编辑距离本质上是一个简单的编辑距离，只不过编辑操作允许为每种操作分配不同的成本。我们在前面的示例中识别出的编辑操作包括替换、插入、删除和交换。此外，还可以为完全匹配分配成本，以提高匹配的权重——当需要进行编辑时，这可能会用于字符串变异生成器。编辑权重通常以对数概率的形式进行缩放，这样你就可以为编辑操作分配可能性。权重越大，表示该编辑操作越有可能发生。由于概率值介于 0 和 1 之间，因此对数概率或权重将在负无穷大到零之间。更多内容请参阅`WeightedEditDistance`类的 Javadoc：[`alias-i.com/lingpipe/docs/api/com/aliasi/spell/WeightedEditDistance.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/spell/WeightedEditDistance.html)

在对数尺度上，加权编辑距离可以通过将匹配权重设置为 0，将替换、删除和插入的权重设置为-1，且将置换权重设置为-1 或负无穷（如果我们想关闭置换操作），以此方式将简单编辑距离的结果与前一个示例中的结果完全一样。

我们将在其他示例中查看加权编辑距离在拼写检查和中文分词中的应用。

在本节中，我们将使用`FixedWeightEditDistance`实例，并创建扩展了`WeightedEditDistance`抽象类的`CustomWeightEditDistance`类。`FixedWeightEditDistance`类通过为每个编辑操作初始化权重来创建。`CustomWeightEditDistance`类扩展了`WeightedEditDistance`，并为每个编辑操作的权重制定了规则。删除字母数字字符的权重是-1，对于所有其他字符，即标点符号和空格，则为 0。我们将插入权重设置为与删除权重相同。

## 如何操作...

让我们在前面的例子基础上扩展，并看一个同时运行简单编辑距离和加权编辑距离的版本：

1.  在你的 IDE 中运行`SimpleWeightedEditDistance`类，或者在命令行中输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter6.SimpleWeightedEditDistance

    ```

1.  在命令行中，你将被提示输入两个字符串：输入此处显示的示例，或者选择你自己的：如何操作...

1.  如你所见，这里显示了另外两种距离度量：固定权重编辑距离和自定义权重编辑距离。

1.  尝试其他示例，包括标点符号和空格。

## 它是如何工作的...

我们将实例化一个`FixedWeightEditDistance`类，并设置一些权重，这些权重是任意选择的：

```py
double matchWeight = 0;
double deleteWeight = -2;
double insertWeight = -2;
double substituteWeight = -2;
double transposeWeight = Double.NEGATIVE_INFINITY;
WeightedEditDistance wed = new FixedWeightEditDistance(matchWeight,deleteWeight,insertWeight,substituteWeight,transposeWeight);
System.out.println("Fixed Weight Edit Distance: "+ wed.toString());
```

在这个例子中，我们将删除、替换和插入的权重设置为相等。这与标准的编辑距离非常相似，唯一的区别是我们将编辑操作的权重从 1 修改为 2。将置换权重设置为负无穷有效地完全关闭了置换操作。显然，删除、替换和插入的权重不必相等。

我们还将创建一个`CustomWeightEditDistance`类，它将标点符号和空格视为匹配项，也就是说，插入和删除操作的成本为零（对于字母或数字，成本仍为-1）。对于替换操作，如果字符仅在大小写上有所不同，则成本为零；对于所有其他情况，成本为-1。我们还将通过将其成本设置为负无穷来关闭置换操作。这将导致`Abc+`与`abc-`匹配：

```py
public static class CustomWeightedEditDistance extends WeightedEditDistance{

  @Override
  public double deleteWeight(char arg0) {
    return (Character.isDigit(arg0)||Character.isLetter(arg0)) ? -1 : 0;

  }

  @Override
  public double insertWeight(char arg0) {
    return deleteWeight(arg0);
  }

  @Override
  public double matchWeight(char arg0) {
    return 0;
  }

  @Override
  public double substituteWeight(char cDeleted, char cInserted) {
    return Character.toLowerCase(cDeleted) == Character.toLowerCase(cInserted) ? 0 :-1;

  }

  @Override
  public double transposeWeight(char arg0, char arg1) {
    return Double.NEGATIVE_INFINITY;
  }

}
```

这种自定义加权编辑距离特别适用于比较字符串，其中可能会遇到细微的格式更改，比如基因/蛋白质名称从`Serpin A3`变成`serpina3`，但它们指的却是同一样东西。

## 另见

+   有一个 T&T（Tsuruoka 和 Tsujii）编辑距离规范用于比较蛋白质名称，参见 [`alias-i.com/lingpipe/docs/api/com/aliasi/dict/ApproxDictionaryChunker.html#TT_DISTANCE`](http://alias-i.com/lingpipe/docs/api/com/aliasi/dict/ApproxDictionaryChunker.html#TT_DISTANCE)

+   有关 `WeightedEditDistance` 类的更多细节，可以在 Javadoc 页面找到，网址为：[`alias-i.com/lingpipe/docs/api/com/aliasi/spell/WeightedEditDistance.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/spell/WeightedEditDistance.html)

# Jaccard 距离

Jaccard 距离是一种非常流行且高效的字符串比较方法。Jaccard 距离在标记级别进行操作，通过首先对两个字符串进行标记化，然后将共同标记的数量除以总的标记数量来比较两个字符串。在第一章《简单分类器》中的*使用 Jaccard 距离消除近似重复项*示例中，我们应用该距离来消除近似重复的推文。本篇会更详细地介绍，并展示如何计算它。

距离为 0 是完美匹配，也就是说，两个字符串共享所有的词项，而距离为 1 是完美不匹配，也就是说，两个字符串没有共同的词项。请记住，接近度和距离是相互逆的，因此接近度的范围也是从 1 到 0。接近度为 1 是完美匹配，接近度为 0 是完美不匹配：

```py
proximity  = count(common tokens)/count(total tokens)
distance = 1 – proximity
```

标记由 `TokenizerFactory` 生成，在构造时传入。例如，让我们使用 `IndoEuropeanTokenizerFactory`，并看一个具体示例。如果 `string1` 是 `fruit flies like a banana`，`string2` 是 `time flies like an arrow`，那么 `string1` 的标记集为 `{'fruit', 'flies', 'like', 'a', 'banana'}`，`string2` 的标记集为 `{'time', 'flies', 'like', 'an', 'arrow'}`。这两个标记集之间的共同词项（或交集）是 `{'flies', 'like'}`，这些词项的并集是 `{'fruit', 'flies', 'like', 'a', 'banana', 'time', 'an', 'arrow'}`。现在，我们可以通过将共同词项的数量除以词项的总数量来计算 Jaccard 接近度，即 2/8，结果为 0.25。因此，距离是 0.75（1 - 0.25）。显然，通过修改类初始化时使用的标记器，Jaccard 距离是非常可调的。例如，可以使用一个大小写标准化的标记器，使得 `Abc` 和 `abc` 被认为是等效的。同样，使用词干提取标记器时，`runs` 和 `run` 将被认为是等效的。我们将在下一个距离度量——Tf-Idf 距离中看到类似的功能。

## 如何操作...

下面是如何运行 `JaccardDistance` 示例：

1.  在 Eclipse 中，运行 `JaccardDistanceSample` 类，或者在命令行中输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter6.JaccardDistanceSample

    ```

1.  与之前的示例一样，您将被要求输入两个字符串。我们将使用的第一个字符串是`Mimsey Were the Borogroves`，这是一个非常优秀的科幻短篇小说标题，第二个字符串`All mimsy were the borogoves,`是来自*Jabberwocky*的实际诗句，启发了这个标题：

    ```py
    Enter the first string:
    Mimsey Were the Borogroves
    Enter the second string:
    All mimsy were the borogoves,

    IndoEuropean Tokenizer
    Text1 Tokens: {'Mimsey''Were''the'}
    Text2 Tokens: {'All''mimsy''were''the''borogoves'}
    IndoEuropean Jaccard Distance is 0.8888888888888888

    Character Tokenizer
    Text1 Tokens: {'M''i''m''s''e''y''W''e''r''e''t''h''e''B''o''r''o''g''r''o''v''e'}
    Text2 Tokens: {'A''l''l''m''i''m''s''y''w''e''r''e''t''h''e''b''o''r''o''g''o''v''e''s'}
    Character Jaccard Distance between is 0.42105263157894735

    EnglishStopWord Tokenizer
    Text1 Tokens: {'Mimsey''Were'}
    Text2 Tokens: {'All''mimsy''borogoves'}
    English Stopword Jaccard Distance between is 1.0

    ```

1.  输出包含使用三种不同分词器生成的标记和距离。`IndoEuropean`和`EnglishStopWord`分词器非常相似，显示这两行文本相距较远。记住，两个字符串越接近，它们之间的距离就越小。然而，字符分词器显示，这些行在以字符为比较基础的情况下距离较近。分词器在计算字符串间距离时可能会产生很大的差异。

## 它是如何工作的……

代码很简单，我们只会讲解`JaccardDistance`对象的创建。我们将从三个分词器工厂开始：

```py
TokenizerFactory indoEuropeanTf = IndoEuropeanTokenizerFactory.INSTANCE;

TokenizerFactory characterTf = CharacterTokenizerFactory.INSTANCE;

TokenizerFactory englishStopWordTf = new EnglishStopTokenizerFactory(indoEuropeanTf);
```

请注意，`englishStopWordTf`使用基础分词器工厂构建自己。如果有任何疑问，参阅第二章，*查找和处理词语*。

接下来，构建 Jaccard 距离类，并将分词器工厂作为参数：

```py
JaccardDistance jaccardIndoEuropean = new JaccardDistance(indoEuropeanTf);
JaccardDistance jaccardCharacter = new JaccardDistance(characterTf);

JaccardDistance jaccardEnglishStopWord = new JaccardDistance(englishStopWordTf);
```

其余的代码只是我们标准的输入/输出循环和一些打印语句。就是这样！接下来是更复杂的字符串距离度量。

# Tf-Idf 距离

一个非常有用的字符串间距离度量是由`TfIdfDistance`类提供的。它实际上与流行的开源搜索引擎 Lucene/SOLR/Elastic Search 中的距离度量密切相关，其中被比较的字符串是查询与索引中文档的比对。Tf-Idf 代表核心公式，即**词频**（**TF**）乘以**逆文档频率**（**IDF**），用于查询与文档中共享的词。关于这种方法的一个非常酷的地方是，常见词（例如，`the`）在文档中出现频繁，因此其权重被下调，而稀有词则在距离比较中得到上调。这有助于将距离集中在文档集中真正具有区分性的词上。

`TfIdfDistance`不仅对类似搜索引擎的应用非常有用，它对于聚类和任何需要计算文档相似度的问题也非常有用，而无需监督训练数据。它有一个理想的属性；分数被标准化为 0 到 1 之间的分数，并且对于固定的文档`d1`和不同长度的文档`d2`，不会使分配的分数过大。在我们的经验中，如果你想评估一对文档的匹配质量，不同文档对的分数是相当稳健的。

### 注意

请注意，有一系列不同的距离被称为 Tf-Idf 距离。此类中的距离定义为对称的，不像典型的用于信息检索目的的 Tf-Idf 距离。

Javadoc 中有很多值得一看的信息。然而，针对这些食谱，你需要知道的是，Tf-Idf 距离在逐字查找相似文档时非常有用。

## 如何做……

为了让事情稍微有点趣味，我们将使用我们的`TfIdfDistance`类来构建一个非常简单的推文搜索引擎。我们将执行以下步骤：

1.  如果你还没有做过，运行第一章中的`TwitterSearch`类，*简单分类器*，并获取一些推文进行操作，或者使用我们提供的数据。我们将使用通过运行`Disney World`查询找到的推文，它们已经在`data`目录中。

1.  在命令行中输入以下内容——这使用我们的默认设置：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar:lib/opencsv-2.4.jar com.lingpipe.cookbook.chapter6.TfIdfSearch
    Reading search index from data/disney.csv
    Getting IDF data from data/connecticut_yankee_king_arthur.txt
    enter a query:

    ```

1.  输入一个可能有匹配单词的查询：

    ```py
    I want to go to disney world
    0.86 : I want to go to Disneyworld
    0.86 : I want to go to disneyworld
    0.75 : I just want to go to DisneyWorld...
    0.75 : I just want to go to Disneyworld ???
    0.65 : Cause I wanna go to Disneyworld.
    0.56 : I wanna go to disneyworld with Demi
    0.50 : I wanna go back to disneyworld
    0.50 : I so want to go to Disneyland I've never been. I've been to Disneyworld in Florida.
    0.47 : I want to go to #DisneyWorld again... It's so magical!!
    0.45 : I want to go to DisneyWorld.. Never been there :( #jadedchildhood

    ```

1.  就是这样。尝试不同的查询，玩弄一下得分。然后，看看源代码。

## 它是如何工作的……

这段代码是构建搜索引擎的一种非常简单的方法，而不是一种好方法。然而，它是探索字符串距离概念在搜索上下文中如何工作的一个不错的方式。本书后续将基于相同的距离度量进行聚类。可以从`src/com/lingpipe/cookbook/chapter6/TfIdfSearch.java`中的`main()`类开始：

```py
public static void main(String[] args) throws IOException {
  String searchableDocs = args.length > 0 ? args[0] : "data/disneyWorld.csv";
  System.out.println("Reading search index from " + searchableDocs);

  String idfFile = args.length > 1 ? args[1] : "data/connecticut_yankee_king_arthur.txt";
  System.out.println("Getting IDF data from " + idfFile);
```

该程序可以接受命令行传入的`.csv`格式的搜索数据文件和用作训练数据源的文本文件。接下来，我们将设置一个标记器工厂和`TfIdfDistance`。如果你不熟悉标记器工厂，可以参考第二章中的*修改标记器工厂*食谱，以获取解释：

```py
TokenizerFactory tokFact = IndoEuropeanTokenizerFactory.INSTANCE;
TfIdfDistance tfIdfDist = new TfIdfDistance(tokFact);
```

然后，我们将通过按“.”分割训练文本来获取将作为 IDF 组件的数据，这种方式大致上是句子检测——我们本可以像在第五章的*句子检测*食谱中那样进行正式的句子检测，但我们选择尽可能简单地展示这个例子：

```py
String training = Files.readFromFile(new File(idfFile), Strings.UTF8);
for (String line: training.split("\\.")) {
  tfIdfDist.handle(line);
}
```

在`for`循环中，有`handle()`，它通过语料库中的标记分布训练该类，句子即为文档。通常情况下，文档的概念要么小于（句子、段落和单词），要么大于通常所称的`文档`。在这种情况下，文档频率将是该标记所在的句子数。

接下来，我们加载我们要搜索的文档：

```py
List<String[]> docsToSearch = Util.readCsvRemoveHeader(new File(searchableDocs));
```

控制台设置为读取查询：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.println("enter a query: ");
  String query = reader.readLine();
```

接下来，每个文档将使用`TfIdfDistance`与查询进行评分，并放入`ObjectToDoubleMap`中，该映射用于跟踪相似度：

```py
ObjectToDoubleMap<String> scoredMatches = new ObjectToDoubleMap<String>();
for (String [] line : docsToSearch) {
  scoredMatches.put(line[Util.TEXT_OFFSET], tfIdfDist.proximity(line[Util.TEXT_OFFSET], query));
}
```

最后，`scoredMatches`按相似度顺序被检索，并打印出前 10 个示例：

```py
List<String> rankedDocs = scoredMatches.keysOrderedByValueList();
for (int i = 0; i < 10; ++i) {
  System.out.printf("%.2f : ", scoredMatches.get(rankedDocs.get(i)));
  System.out.println(rankedDocs.get(i));
}
}
```

尽管这种方法非常低效，因为每次查询都遍历所有训练数据，进行显式的`TfIdfDistance`比较并存储结果，但它对于玩转小数据集和比较度量指标来说并不是一种坏方法。

## 还有更多内容...

有一些值得强调的`TfIdfDistance`的细节。

### 有监督和无监督训练的区别

当我们训练`TfIdfDistance`时，在训练的使用上有一些重要的区别，这些区别与本书其他部分的使用不同。这里进行的训练是无监督的，这意味着没有人类或其他外部来源标记数据的预期结果。本书中大多数训练使用的是人类标注或监督数据。

### 在测试数据上训练是可以的

由于这是无监督数据，因此没有要求训练数据必须与评估或生产数据不同。

# 使用编辑距离和语言模型进行拼写纠正

拼写纠正接收用户输入的文本并提供纠正后的形式。我们大多数人都熟悉通过智能手机或像 Microsoft Word 这样的编辑器进行的自动拼写纠正。网络上显然有很多有趣的例子，展示了拼写纠正失败的情况。在这个例子中，我们将构建自己的拼写纠正引擎，并看看如何调整它。

LingPipe 的拼写纠正基于噪声信道模型，该模型模拟了用户的错误和预期用户输入（基于数据）。预期用户输入通过字符语言模型进行建模，而错误（或噪声）则通过加权编辑距离建模。拼写纠正是通过`CompiledSpellChecker`类来完成的。该类实现了噪声信道模型，并根据实际收到的消息，提供最可能的消息估计。我们可以通过以下公式来表达这一点：

```py
didYouMean(received) = ArgMaxintended P(intended | received) 
= ArgMaxintended P(intended,received) / P(received) 
= ArgMaxintended P(intended,received) 
= ArgMaxintended P(intended) * P(received | intended)
```

换句话说，我们首先通过创建一个 n-gram 字符语言模型来构建预期消息的模型。语言模型存储了已见短语的统计数据，本质上，它存储了 n-gram 出现的次数。这给我们带来了`P(intended)`。例如，`P(intended)`表示字符序列`the`的可能性。接下来，我们将创建信道模型，这是一个加权编辑距离，它给出了输入错误的概率，即用户输入的错误与预期文本之间的差距。再例如，用户本来打算输入`the`，但错误地输入了`teh`，这种错误的概率是多少。我们将使用加权编辑距离来建模这种可能性，其中权重按对数概率进行缩放。请参考本章前面的*加权编辑距离*配方。

创建一个编译后的拼写检查器的常见方法是通过`TrainSpellChecker`实例。编译拼写检查训练类并将其读取回来后的结果就是一个编译过的拼写检查器。`TrainSpellChecker`通过编译过程创建了基本的模型、加权编辑距离和标记集。然后，我们需要在`CompiledSpellChecker`对象上设置各种参数。

可以选择性地指定一个分词工厂来训练对标记敏感的拼写检查器。通过分词，输入会进一步规范化，在所有未由空格分隔的标记之间插入单个空格。标记会在编译时输出，并在编译后的拼写检查器中读取回来。标记集的输出可能会被修剪，以删除任何低于给定计数阈值的标记。因为在没有标记的情况下我们只有字符，所以阈值在没有标记的情况下没有意义。此外，已知标记集可用于在拼写校正时限制替代拼写的建议，仅包括观察到的标记集中的标记。

这种拼写检查方法相较于纯粹基于字典的解决方案有几个优点：

+   这个上下文得到了有效建模。如果下一个词是`dealership`，则`Frod`可以被纠正为`Ford`；如果下一个词是`Baggins`（《魔戒》三部曲中的角色），则可以纠正为`Frodo`。

+   拼写检查可以对领域敏感。这个方法相较于基于字典的拼写检查还有一个大优点，那就是修正是基于训练语料库中的数据进行的。因此，在法律领域，`trt`将被纠正为`tort`，在烹饪领域，它将被纠正为`tart`，在生物信息学领域，它将被纠正为`TRt`。

## 如何操作...

让我们来看一下运行拼写检查的步骤：

1.  在你的 IDE 中，运行`SpellCheck`类，或者在命令行中输入以下命令—注意我们通过`–Xmx1g`标志分配了 1GB 的堆内存：

    ```py
    java -Xmx1g -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar:lib/opencsv-2.4.jar com.lingpipe.cookbook.chapter6.SpellCheck 

    ```

1.  请耐心等待；拼写检查器需要一到两分钟的时间来训练。

1.  现在，让我们输入一些拼写错误的单词，例如`beleive`：

    ```py
    Enter word, . to quit:
    >beleive
    Query Text: beleive
    Best Alternative: believe
    Nbest: 0: believe Score:-13.97322991490364
    Nbest: 1: believed Score:-17.326215342327487
    Nbest: 2: believes Score:-20.8595682233572
    Nbest: 3: because Score:-21.468056442099623

    ```

1.  如你所见，我们获得了最接近输入文本的最佳替代方案，以及一些其他替代方案。它们按最有可能是最佳替代方案的可能性排序。

1.  现在，我们可以尝试不同的输入，看看这个拼写检查器的表现如何。输入多个单词，看看它的效果：

    ```py
    The rain in Spani falls mainly on the plain.
    Query Text: The rain in Spani falls mainly on the plain.
    Best Alternative: the rain in spain falls mainly on the plain .
    Nbest: 0: the rain in spain falls mainly on the plain . Score:-96.30435947472415
    Nbest: 1: the rain in spain falls mainly on the plan . Score:-100.55447634639404
    Nbest: 2: the rain in spain falls mainly on the place . Score:-101.32592701496742
    Nbest: 3: the rain in spain falls mainly on the plain , Score:-101.81294112237359

    ```

1.  此外，尝试输入一些专有名词，看看它们是如何被评估的。

## 它是如何工作的...

现在，让我们来看一下是什么让这一切运作起来。我们将从设置`TrainSpellChecker`开始，它需要一个`NGramProcessLM`实例、`TokenizerFactory`和一个`EditDistance`对象，用于设置编辑操作的权重，例如删除、插入、替换等：

```py
public static void main(String[] args) throws IOException, ClassNotFoundException {
  double matchWeight = -0.0;
  double deleteWeight = -4.0;
  double insertWeight = -2.5;
  double substituteWeight = -2.5;
  double transposeWeight = -1.0;

  FixedWeightEditDistance fixedEdit = new FixedWeightEditDistance(matchWeight,deleteWeight,insertWeight,substituteWeight,transposeWeight);
  int NGRAM_LENGTH = 6;
  NGramProcessLM lm = new NGramProcessLM(NGRAM_LENGTH);

  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  tokenizerFactory = new com.aliasi.tokenizer.LowerCaseTokenizerFactory(tokenizerFactory);
```

`NGramProcessLM` 需要知道在建模数据时要采样的字符数量。此示例中已经为加权编辑距离提供了合理的值，但可以根据特定数据集的变化进行调整：

```py
TrainSpellChecker sc = new TrainSpellChecker(lm,fixedEdit,tokenizerFactory);
```

`TrainSpellChecker` 现在可以构建，接下来我们将从古腾堡计划中加载 150,000 行书籍。在搜索引擎的上下文中，这些数据将是你的索引中的数据：

```py
File inFile = new File("data/project_gutenberg_books.txt");
String bigEnglish = Files.readFromFile(inFile,Strings.UTF8);
sc.handle(bigEnglish);
```

接下来，我们将从字典中添加条目，以帮助处理罕见单词：

```py
File dict = new File("data/websters_words.txt");
String webster = Files.readFromFile(dict, Strings.UTF8);
sc.handle(webster);
```

接下来，我们将编译 `TrainSpellChecker`，以便我们可以实例化 `CompiledSpellChecker`。通常，`compileTo()` 操作的输出会写入磁盘，并从磁盘读取并实例化 `CompiledSpellChecker`，但这里使用的是内存中的选项：

```py
CompiledSpellChecker csc = (CompiledSpellChecker) AbstractExternalizable.compile(sc);
```

请注意，还有一种方法可以将数据反序列化为 `TrainSpellChecker`，以便以后可能添加更多数据。`CompiledSpellChecker` 不接受进一步的训练实例。

`CompiledSpellChecker` 接受许多微调方法，这些方法在训练期间不相关，但在使用时是相关的。例如，它可以接受一组不进行编辑的字符串；在这种情况下，单个值是 `lingpipe`：

```py
Set<String> dontEdit = new HashSet<String>();
dontEdit.add("lingpipe");
csc.setDoNotEditTokens(dontEdit);
```

如果输入中出现这些标记，它们将不会被考虑进行编辑。这会对运行时间产生巨大影响。这个集合越大，解码器的运行速度就越快。如果执行速度很重要，请将不编辑标记的集合配置得尽可能大。通常，这通过从已编译的拼写检查器中获取对象并保存出现频率较高的标记来实现。

在训练期间，使用了分词器工厂将数据标准化为由单个空格分隔的标记。它不会在编译步骤中序列化，因此，如果需要在不编辑标记中保持标记敏感性，则必须提供：

```py
csc.setTokenizerFactory(tokenizerFactory);
int nBest = 3;
csc.setNBest(64);
```

`nBest` 参数设置了在修改输入时将考虑的假设数量。尽管输出中的 `nBest` 大小设置为 3，但建议在从左到右探索最佳编辑的过程中允许更大的假设空间。此外，类还有方法来控制允许的编辑以及如何评分。有关更多信息，请参阅教程和 Javadoc。

最后，我们将进行一个控制台 I/O 循环以生成拼写变化：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
String query = "";
while (true) {
  System.out.println("Enter word, . to quit:");
  query = reader.readLine();
  if (query.equals(".")){
    break;
  }
  String bestAlternative = csc.didYouMean(query);
  System.out.println("Best Alternative: " + bestAlternative);
  int i = 0;
  Iterator<ScoredObject<String>> iterator = csc.didYouMeanNBest(query);
  while (i < nBest) {
    ScoredObject<String> so = iterator.next();
    System.out.println("Nbest: " + i + ": " + so.getObject() + " Score:" + so.score());
    i++;
  }
}
```

### 提示

我们在这个模型中包含了一个字典，我们将像处理其他数据一样将字典条目输入到训练器中。

通过多次训练字典中的每个单词，可能会使字典得到增强。根据字典的数量，它可能会主导或被源训练数据所主导。

## 另请参阅

+   拼写修正教程更完整，涵盖了在 [`alias-i.com/lingpipe/demos/tutorial/querySpellChecker/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/querySpellChecker/read-me.html) 进行的评估

+   `CompiledSpellChecker` 的 Javadoc 可以在 [`alias-i.com/lingpipe/docs/api/com/aliasi/spell/CompiledSpellChecker.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/spell/CompiledSpellChecker.html) 找到

+   更多关于拼写检查器如何工作的内容，请参见教材《*Speech and Language Processing*》，*Jurafsky*、*Dan* 和 *James H. Martin* 编著，*2000*，*Prentice-Hall*。

# 大小写恢复校正器

大小写恢复拼写校正器，也叫做真大小写校正器，只恢复大小写，不更改其他任何内容，也就是说，它不会纠正拼写错误。当处理转录、自动语音识别输出、聊天记录等低质量文本时，这非常有用，因为这些文本通常包含各种大小写问题。我们通常希望增强这些文本，以构建更好的基于规则或机器学习的系统。例如，新闻和视频转录（如字幕）通常存在错误，这使得使用这些数据训练命名实体识别（NER）变得更加困难。大小写恢复可以作为不同数据源之间的标准化工具，确保所有数据的一致性。

## 如何操作……

1.  在你的 IDE 中运行 `CaseRestore` 类，或者在命令行中输入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter6.CaseRestore 

    ```

1.  现在，让我们输入一些错误大小写或单一大小写的文本：

    ```py
    Enter input, . to quit:
    george washington was the first president of the u.s.a
    Best Alternative: George Washington was the first President of the U.S.A
    Enter input, . to quit:
    ITS RUDE TO SHOUT ON THE WEB
    Best Alternative: its rude to shout on the Web

    ```

1.  如你所见，大小写错误已经被纠正。如果我们使用更现代的文本，例如当前的报纸数据或类似内容，这将直接应用于广播新闻转录或字幕的大小写标准化。

## 它是如何工作的……

该类的工作方式类似于拼写校正，我们有一个由语言模型指定的模型和一个由编辑距离度量指定的通道模型。然而，距离度量只允许大小写更改，也就是说，大小写变体是零成本的，所有其他编辑成本都被设置为 `Double.NEGATIVE_INFINITY`：

我们将重点讨论与前一个方法不同的部分，而不是重复所有源代码。我们将使用来自古腾堡计划的英文文本训练拼写检查器，并使用 `CompiledSpellChecker` 类中的 `CASE_RESTORING` 编辑距离：

```py
int NGRAM_LENGTH = 5;
NGramProcessLM lm = new NGramProcessLM(NGRAM_LENGTH);
TrainSpellChecker sc = new TrainSpellChecker(lm,CompiledSpellChecker.CASE_RESTORING);
```

再次通过调用 `bestAlternative` 方法，我们将获得最好的大小写恢复文本估计：

```py
String bestAlternative = csc.didYouMean(query);
```

就是这样。大小写恢复变得简单。

## 另见

+   Lucian Vlad Lita 等人于 2003 年的论文，[`www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf`](http://www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf)，是关于真大小写恢复的一个很好的参考资料。

# 自动短语补全

自动短语补全与拼写校正不同，它是在用户输入的文本中，从一组固定短语中找到最可能的补全。

显然，自动短语补全在网络上无处不在，例如，在[`google.com`](https://google.com)上。例如，如果我输入 `anaz` 作为查询，谷歌会弹出以下建议：

![自动短语补全](img/4672OS_06_02.jpg)

请注意，应用程序在完成补全的同时也在进行拼写检查。例如，即使查询到目前为止是**anaz**，但顶部的建议是**amazon**。这并不令人惊讶，因为以**anaz**开头的短语的结果数量可能非常少。

接下来，注意到它并不是进行单词建议，而是短语建议。比如一些结果，如**amazon prime**是由两个单词组成的。

自动补全和拼写检查之间的一个重要区别是，自动补全通常是基于一个固定的短语集，必须匹配开头才能完成。这意味着，如果我输入查询`I want to find anaz`，就不会有任何推荐补全。网页搜索的短语来源通常是来自查询日志的高频查询。

在 LingPipe 中，我们使用`AutoCompleter`类，它维护一个包含计数的短语字典，并通过加权编辑距离和短语似然性基于前缀匹配提供建议的补全。

自动补全器为给定的前缀找到得分最高的短语。短语与前缀的得分是短语得分和该前缀与短语任何前缀匹配的最大得分之和。短语的得分就是其最大似然概率估计，即其计数的对数除以所有计数的总和。

谷歌和其他搜索引擎很可能将它们的查询计数作为最佳得分短语的数据。由于我们这里没有查询日志，因此我们将使用美国人口超过 100,000 的城市的美国人口普查数据。短语是城市名称，计数是它们的人口。

## 如何操作...

1.  在你的 IDE 中，运行`AutoComplete`类，或者在命令行中输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter6.AutoComplete 

    ```

1.  输入一些美国城市名称并查看输出。例如，输入`new`将产生以下输出：

    ```py
    Enter word, . to quit:
    new
    |new|
    -13.39 New York,New York
    -17.89 New Orleans,Louisiana
    -18.30 Newark,New Jersey
    -18.92 Newport News,Virginia
    -19.39 New Haven,Connecticut
    If we misspell 'new' and type 'mew' instead, 
    Enter word, . to quit:
    mew 

    |mew |
    -13.39 New York,New York
    -17.89 New Orleans,Louisiana
    -19.39 New Haven,Connecticut

    ```

1.  输入我们初始列表中不存在的城市名称将不会返回任何输出：

    ```py
    Enter word, . to quit:
    Alta,Wyoming
    |Alta,Wyoming|

    ```

## 它是如何工作的...

配置自动补全器与配置拼写检查非常相似，不同之处在于，我们不是训练一个语言模型，而是提供一个固定的短语和计数列表、一个编辑距离度量以及一些配置参数。代码的初始部分只是读取一个文件，并设置一个短语到计数的映射：

```py
File wordsFile = new File("data/city_populations_2012.csv");
String[] lines = FileLineReader.readLineArray(wordsFile,"ISO-8859-1");
ObjectToCounterMap<String> cityPopMap = new ObjectToCounterMap<String>();
int lineCount = 0;
for (String line : lines) {
if(lineCount++ <1) continue;
  int i = line.lastIndexOf(',');
  if (i < 0) continue;
  String phrase = line.substring(0,i);
  String countString = line.substring(i+1);
  Integer count = Integer.valueOf(countString);

  cityPopMap.set(phrase,count);
}
```

下一步是配置编辑距离。此操作将衡量目标短语的前缀与查询前缀的相似度。该类使用固定权重的编辑距离，但一般来说，可以使用任何编辑距离：

```py
double matchWeight = 0.0;
double insertWeight = -10.0;
double substituteWeight = -10.0;
double deleteWeight = -10.0;
double transposeWeight = Double.NEGATIVE_INFINITY;
FixedWeightEditDistance editDistance = new FixedWeightEditDistance(matchWeight,deleteWeight,insertWeight,substituteWeight,transposeWeight);
```

有一些参数可以调整自动补全：编辑距离和搜索参数。编辑距离的调整方式与拼写检查完全相同。返回结果的最大数量更多是应用程序的决定，而不是调整的决策。话虽如此，较小的结果集计算速度更快。最大队列大小表示在被修剪之前，自动补全器内部假设集可以变得多大。在仍能有效执行的情况下，将`maxQueueSize`设置为尽可能小，以提高速度：

```py
int maxResults = 5;
int maxQueueSize = 10000;
double minScore = -25.0;
AutoCompleter completer = new AutoCompleter(cityPopMap, editDistance,maxResults, maxQueueSize, minScore);
```

## 另见

+   查看`AutoCompleter`类的 Javadoc 文档：[`alias-i.com/lingpipe/docs/api/com/aliasi/spell/AutoCompleter.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/spell/AutoCompleter.html)

# 使用编辑距离的单链和完全链聚类

聚类是通过相似性将一组对象分组的过程，也就是说，使用某种距离度量。聚类的核心思想是，聚类内的对象彼此接近，而不同聚类的对象彼此较远。我们可以大致将聚类技术分为层次（或凝聚）和分治两种技术。层次技术从假设每个对象都是自己的聚类开始，然后合并聚类，直到满足停止准则。

例如，一个停止准则可以是每个聚类之间的固定距离。分治技术则恰好相反，首先将所有对象聚集到一个聚类中，然后进行拆分，直到满足停止准则，例如聚类的数量。

我们将在接下来的几个实例中回顾层次聚类技术。LingPipe 中我们将提供的两种聚类实现是单链聚类和完全链聚类；所得的聚类形成输入集的所谓划分。若一组集合是另一个集合的划分，则该集合的每个元素恰好属于划分中的一个集合。从数学角度来说，构成划分的集合是成对不相交的，并且它们的并集是原始集合。

聚类器接收一组对象作为输入，并返回一组对象的集合作为输出。也就是说，在代码中，`Clusterer<String>`有一个`cluster`方法，作用于`Set<String>`并返回`Set<Set<String>>`。

层次聚类器扩展了`Clusterer`接口，同样作用于一组对象，但返回的是`Dendrogram`（树状图），而不是一组对象的集合。树状图是一个二叉树，表示正在聚类的元素，其中每个分支附有距离值，表示两个子分支之间的距离。对于`aa`、`aaa`、`aaaaa`、`bbb`、`bbbb`这些字符串，基于单链的树状图并采用`EditDistance`作为度量看起来是这样的：

```py
3.0
 2.0
 1.0
 aaa
 aa
 aaaaa
 1.0
 bbbb
 bbb

```

上述树状图基于单链聚类，单链聚类将任何两个元素之间的最小距离作为相似性的度量。因此，当`{'aa','aaa'}`与`{'aaaa'}`合并时，得分为 2.0，通过将两个`a`添加到`aaa`中。完全链接聚类则采用任何两个元素之间的最大距离，这将是 3.0，通过将三个`a`添加到`aa`中。单链聚类倾向于形成高度分离的聚类，而完全链接聚类则倾向于形成更紧密的聚类。

从树状图中提取聚类有两种方法。最简单的方法是设置一个距离上限，并保持所有在此上限或以下形成的聚类。另一种构建聚类的方法是继续切割最大距离的聚类，直到获得指定数量的聚类。

在这个示例中，我们将研究使用`EditDistance`作为距离度量的单链和完全链接聚类。我们将尝试通过`EditDistance`对城市名称进行聚类，最大距离为 4。

## 如何操作…

1.  在您的 IDE 中运行`HierarchicalClustering`类，或者在命令行中输入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter6.HierarchicalClustering

    ```

1.  输出是对同一基础集合`Strings`的各种聚类方法。在这个示例中，我们将交替展示源和输出。首先，我们将创建我们的字符串集合：

    ```py
    public static void main(String[] args) throws UnsupportedEncodingException, IOException {

      Set<String> inputSet = new HashSet<String>();
      String [] input = { "aa", "aaa", "aaaaa", "bbb", "bbbb" };
      inputSet.addAll(Arrays.asList(input));
    ```

1.  接下来，我们将设置一个使用`EditDistance`的单链实例，并为前面的集合创建树状图并打印出来：

    ```py
    boolean allowTranspositions = false;
    Distance<CharSequence> editDistance = new EditDistance(allowTranspositions);

    AbstractHierarchicalClusterer<String> slClusterer = new SingleLinkClusterer<String>(editDistance);

    Dendrogram<String> slDendrogram = slClusterer.hierarchicalCluster(inputSet);

    System.out.println("\nSingle Link Dendrogram");
    System.out.println(slDendrogram.prettyPrint());
    ```

1.  输出将如下所示：

    ```py
    Single Link Dendrogram

    3.0
     2.0
     1.0
     aaa
     aa
     aaaaa
     1.0
     bbbb
     bbb

    ```

1.  接下来，我们将创建并打印出相同集合的完全链接处理结果：

    ```py
    AbstractHierarchicalClusterer<String> clClusterer = new CompleteLinkClusterer<String>(editDistance);

    Dendrogram<String> clDendrogram = clClusterer.hierarchicalCluster(inputSet);

    System.out.println("\nComplete Link Dendrogram");
    System.out.println(clDendrogram.prettyPrint());
    ```

1.  这将产生相同的树状图，但具有不同的分数：

    ```py
    Complete Link Dendrogram

    5.0
     3.0
     1.0
     aaa
     aa
     aaaaa
     1.0
     bbbb
     bbb

    ```

1.  接下来，我们将生成控制单链情况聚类数量的聚类：

    ```py
    System.out.println("\nSingle Link Clusterings with k Clusters");
    for (int k = 1; k < 6; ++k ) {
      Set<Set<String>> slKClustering = slDendrogram.partitionK(k);
      System.out.println(k + "  " + slKClustering);
    }
    ```

1.  这将产生如下结果——对于完全链接来说，给定输入集合时，它们将是相同的：

    ```py
    Single Link Clusterings with k Clusters
    1  [[bbbb, aaa, aa, aaaaa, bbb]]
    2  [[aaa, aa, aaaaa], [bbbb, bbb]]
    3  [[aaaaa], [bbbb, bbb], [aaa, aa]]
    4  [[bbbb, bbb], [aa], [aaa], [aaaaa]]
    5  [[bbbb], [aa], [aaa], [aaaaa], [bbb]]

    ```

1.  以下代码片段是没有最大距离的完全链接聚类：

    ```py
    Set<Set<String>> slClustering = slClusterer.cluster(inputSet);
    System.out.println("\nComplete Link Clustering No " + "Max Distance");
    System.out.println(slClustering + "\n");
    ```

1.  输出将是：

    ```py
    Complete Link Clustering No Max Distance
    [[bbbb, aaa, aa, aaaaa, bbb]]

    ```

1.  接下来，我们将控制最大距离：

    ```py
    for(int k = 1; k < 6; ++k ){
      clClusterer.setMaxDistance(k);
      System.out.println("Complete Link Clustering at " + "Max Distance= " + k);

      Set<Set<String>> slClusteringMd = clClusterer.cluster(inputSet);
      System.out.println(slClusteringMd);
    }
    ```

1.  以下是通过最大距离限制的聚类效果，适用于完全链接的情况。请注意，这里的单链输入将在 3 的距离下将所有元素放在同一聚类中：

    ```py
    Complete Link Clustering at Max Distance= 1
    [[bbbb, bbb], [aaa, aa], [aaaaa]]
    Complete Link Clustering at Max Distance= 2
    [[bbbb, bbb], [aaa, aa], [aaaaa]]
    Complete Link Clustering at Max Distance= 3
    [[bbbb, bbb], [aaa, aa, aaaaa]]
    Complete Link Clustering at Max Distance= 4
    [[bbbb, bbb], [aaa, aa, aaaaa]]
    Complete Link Clustering at Max Distance= 5
    [[bbbb, aaa, aa, aaaaa, bbb]] 

    ```

1.  就是这样！我们已经演练了 LingPipe 聚类 API 的很大一部分。

## 还有更多内容…

聚类对用于比较聚类的`Distance`非常敏感。查阅 Javadoc 以获取 10 个实现类的可能变种。`TfIdfDistance`在聚类语言数据时非常有用。

K-means（++）聚类是一种基于特征提取的聚类方法。Javadoc 是这样描述它的：

> *K-means 聚类* *可以视为一种迭代方法，旨在最小化项目与其聚类中心之间的平均平方距离……*

## 另请参见…

+   要查看详细的教程，包括评估的具体细节，请访问[`alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html)

# 潜在狄利克雷分配 (LDA) 用于多主题聚类

**潜在狄利克雷分配** (**LDA**) 是一种基于文档中存在的标记或单词的文档聚类统计技术。像分类这样的聚类通常假设类别是互斥的。LDA 的一个特点是，它允许文档同时属于多个主题，而不仅仅是一个类别。这更好地反映了一个推文可以涉及*迪士尼*和*沃利世界*等多个主题的事实。

LDA 的另一个有趣之处，就像许多聚类技术一样，是它是无监督的，这意味着不需要监督式训练数据！最接近训练数据的是必须提前指定主题的数量。

LDA 可以是探索你不知道的未知数据集的一个很好的方式。它也可能很难调整，但通常它会做出一些有趣的结果。让我们让系统运作起来。

对于每个文档，LDA 根据该文档中的单词分配一个属于某个主题的概率。我们将从转换为标记序列的文档开始。LDA 使用标记的计数，并不关心单词出现的上下文或顺序。LDA 在每个文档上操作的模型被称为“词袋模型”，意味着顺序并不重要。

LDA 模型由固定数量的主题组成，每个主题都被建模为一个单词分布。LDA 下的文档被建模为主题分布。对单词的主题分布和文档的主题分布都存在狄利克雷先验。如果你想了解更多幕后发生的事情，可以查看 Javadoc、参考教程和研究文献。

## 准备工作

我们将继续使用来自推文的`.csv`数据。请参考第一章，*简单分类器*，了解如何获取推文，或使用书中的示例数据。该配方使用`data/gravity_tweets.csv`。

这个教程紧密跟随了[`alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html)中的教程，该教程比我们在这个配方中所做的更为详细。LDA 部分位于教程的最后。

## 如何做到的…

本节将对`src/com/lingpipe/cookbook/chapter6/Lda.java`进行源代码审查，并参考`src/com/lingpipe/cookbook/chapter6/LdaReportingHandler.java`辅助类，在使用其部分内容时进行讨论：

1.  `main()`方法的顶部从标准的`csv reader`获取数据：

    ```py
    File corpusFile = new File(args[0]);
     List<String[]> tweets = Util.readCsvRemoveHeader(corpusFile);
    ```

1.  接下来是一堆我们将逐行处理的配置。`minTokenCount` 会过滤掉在算法中出现次数少于五次的所有标记。随着数据集的增大，这个数字可能会增大。对于 1100 条推文，我们假设至少五次提及有助于减少 Twitter 数据的噪声：

    ```py
    int minTokenCount = 5;
    ```

1.  `numTopics` 参数可能是最关键的配置值，因为它告诉算法要找多少个主题。更改这个数字会产生非常不同的主题。你可以尝试调整它。选择 10 表示这 1100 条推文大致涉及 10 个主题。但这显然是错误的，也许 100 会更接近实际情况。也有可能这 1100 条推文有超过 1100 个主题，因为一条推文可以出现在多个主题中。可以多尝试一下：

    ```py
    short numTopics = 10;
    ```

1.  根据 Javadoc，`documentTopicPrior` 的经验法则是将其设置为 5 除以主题数量（如果主题非常少，则可以设置更小的值；0.1 通常是使用的最大值）：

    ```py
    double documentTopicPrior = 0.1;
    ```

1.  `topicWordPrior` 的一个通用实用值如下：

    ```py
    double topicWordPrior = 0.01;
    ```

1.  `burninEpochs` 参数设置在采样之前运行多少个周期。将其设置为大于 0 会产生一些理想的效果，避免样本之间的相关性。`sampleLag` 控制在烧入阶段完成后，采样的频率，`numSamples` 控制采样的数量。目前将进行 2000 次采样。如果 `burninEpochs` 为 1000，那么将会进行 3000 次采样，样本间隔为 1（每次都采样）。如果 `sampleLag` 为 2，那么将会有 5000 次迭代（1000 次烧入，2000 次每 2 个周期采样，总共 4000 个周期）。更多细节请参见 Javadoc 和教程：

    ```py
    int burninEpochs = 0;
    int sampleLag = 1;
    int numSamples = 2000;
    ```

1.  最后，`randomSeed` 初始化了 `GibbsSampler` 中的随机过程：

    ```py
    long randomSeed = 6474835;
    ```

1.  `SymbolTable` 被构造，它将存储字符串到整数的映射，以便进行高效处理：

    ```py
    SymbolTable symbolTable = new MapSymbolTable();
    ```

1.  接下来是我们的标准分词器：

    ```py
    TokenzierFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
    ```

1.  接下来，打印 LDA 的配置：

    ```py
    System.out.println("Input file=" + corpusFile);
    System.out.println("Minimum token count=" + minTokenCount);
    System.out.println("Number of topics=" + numTopics);
    System.out.println("Topic prior in docs=" + documenttopicPrior);
    System.out.println("Word prior in topics=" + wordPrior);
    System.out.println("Burnin epochs=" + burninEpochs);
    System.out.println("Sample lag=" + sampleLag);
    System.out.println("Number of samples=" + numSamples);
    ```

1.  然后，我们将创建一个文档和标记的矩阵，这些矩阵将作为输入传递给 LDA，并报告有多少标记：

    ```py
    int[][] docTokens = LatentDirichletAllocation.tokenizeDocuments(IdaTexts,tokFactory,symbolTable, minTokenCount);
    System.out.println("Number of unique words above count" + " threshold=" + symbolTable.numSymbols());
    ```

1.  紧接着进行一个合理性检查，报告总的标记数量：

    ```py
    int numTokens = 0;
    for (int[] tokens : docTokens){
      numTokens += tokens.length;
    }
    System.out.println("Tokenized.  #Tokens After Pruning=" + numTokens);
    ```

1.  为了获取有关周期/样本的进度报告，创建了一个处理程序来传递所需的消息。它将 `symbolTable` 作为参数，以便能够在报告中重新创建标记：

    ```py
    LdaReportingHandler handler = new LdaReportingHandler(symbolTable);
    ```

1.  搜索在 `LdaReportingHandler` 中访问的方法如下：

    ```py
    public void handle(LatentDirichletAllocation.GibbsSample sample) {
      System.out.printf("Epoch=%3d   elapsed time=%s\n", sample.epoch(), Strings.msToString(System.currentTimeMillis() - mStartTime));

      if ((sample.epoch() % 10) == 0) {
        double corpusLog2Prob = sample.corpusLog2Probability();
        System.out.println("      log2 p(corpus|phi,theta)=" + corpusLog2Prob + "   token cross" + entropy rate=" + (-corpusLog2Prob/sample.numTokens()));
      }
    }
    ```

1.  在完成所有设置之后，我们将开始运行 LDA：

    ```py
    LatentDirichletAllocation.GibbsSample sample = LatentDirichletAllocation.gibbsSampler(docTokens, numTopics,documentTopicPrior,wordPrior,burninEpochs,sampleLag,numSamples,new Random(randomSeed),handler);
    ```

1.  等一下，还有更多内容！不过，我们快完成了。只需要一个最终报告：

    ```py
    int maxWordsPerTopic = 20;
    int maxTopicsPerDoc = 10;
    boolean reportTokens = true;
    handler.reportTopics(sample,maxWordsPerTopic,maxTopicsPerDoc,reportTokens);
    ```

1.  最后，我们将开始运行这段代码。输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar:lib/opencsv-2.4.jar com.lingpipe.cookbook.chapter6.LDA

    ```

1.  看一下结果输出的样本，确认配置和搜索周期的早期报告：

    ```py
    Input file=data/gravity_tweets.csv
    Minimum token count=1
    Number of topics=10
    Topic prior in docs=0.1
    Word prior in topics=0.01
    Burnin epochs=0
    Sample lag=1
    Number of samples=2000
    Number of unique words above count threshold=1652
    Tokenized.  #Tokens After Pruning=10101
    Epoch=  0   elapsed time=:00
     log2 p(corpus|phi,theta)=-76895.71967475882 
     token cross-entropy rate=7.612683860484983
    Epoch=  1   elapsed time=:00

    ```

1.  完成时，我们将获得一个关于发现的主题的报告。第一个主题从按计数排序的单词列表开始。请注意，该主题没有标题。可以通过扫描具有高计数和高 Z 分数的单词来获取`meaning`主题。在这种情况下，有一个 Z 分数为 4.0 的单词`movie`，`a`得到了 6.0，向下查看列表，我们看到`good`的得分为 5.6。Z 分数反映了该单词与具有较高分数的主题的非独立性，这意味着该单词与主题的关联更紧密。查看`LdaReportingHandler`的源代码以获取确切的定义。

    ```py
    TOPIC 0  (total count=1033)
               WORD    COUNT        Z
    --------------------------------------------------
              movie      109       4.0
            Gravity       73       1.9
                  a       72       6.0
                 is       57       4.9
                  !       52       3.2
                was       45       6.0
                  .       42      -0.4
                  ?       41       5.8
               good       39       5.6
    ```

1.  前述输出相当糟糕，而其他主题看起来也不怎么样。下一个主题显示出了潜力，但由于标记化而出现了一些明显的问题：

    ```py
    TOPIC 1  (total count=1334)
     WORD    COUNT        Z
    --------------------------------------------------
     /      144       2.2
     .      117       2.5
     #       91       3.5
     @       73       4.2
     :       72       1.0
     !       50       2.7
     co       49       1.3
     t       47       0.8
     http       47       1.2

    ```

1.  戴上我们系统调谐者的帽子，我们将调整分词器为`new RegExTokenizerFactory("[^\\s]+")`分词器，这真正清理了聚类，将聚类增加到 25 个，并应用`Util.filterJaccard(tweets, tokFactory, .5)`来去除重复项（从 1100 到 301）。这些步骤并非一次执行，但这是一个配方，因此我们展示了一些实验结果。由于没有评估测试集，所以这是一个逐步调整的过程，看看输出是否更好等等。聚类在这样一个开放性问题上评估和调整是非常困难的。输出看起来好了一些。

1.  在浏览主题时，我们发现仍然有许多低价值的词汇扰乱了主题，但`Topic 18`看起来有些有希望，其中`best`和`ever`的 Z 分数很高：

    ```py
    OPIC 18  (total count=115)
     WORD    COUNT        Z
    --------------------------------------------------
     movie       24       1.0
     the       24       1.3
     of       15       1.7
     best       10       3.0
     ever        9       2.8
     one        9       2.8
     I've        8       2.7
     seen        7       1.8
     most        4       1.4
     it's        3       0.9
     had        1       0.2
     can        1       0.2

    ```

1.  进一步查看输出，我们会看到一些在`Topic 18`上得分很高的文档：

    ```py
    DOC 34
    TOPIC    COUNT    PROB
    ----------------------
     18        3   0.270
     4        2   0.183
     3        1   0.096
     6        1   0.096
     8        1   0.096
     19        1   0.096

    Gravity(4) is(6) the(8) best(18) movie(19) I've(18) seen(18) in(3) a(4)

    DOC 50
    TOPIC    COUNT    PROB
    ----------------------
     18        6   0.394
     17        4   0.265
     5        2   0.135
     7        1   0.071

    The(17) movie(18) Gravity(7) has(17) to(17) be(5) one(18) of(18) the(18) best(18) of(18) all(17) time(5)

    ```

1.  对于`best movie ever`主题，这两者看起来都是合理的。然而，请注意其他主题/文档分配相当糟糕。

诚实地说，我们不能完全宣称在这个数据集上取得了胜利，但我们已经阐明了 LDA 的工作原理及其配置。LDA 在商业上并不是巨大的成功，但它为国家卫生研究院和其他客户提供了有趣的概念级别实现。LDA 是一个调谐者的天堂，有很多方法可以对生成的聚类进行调整。查看教程和 Javadoc，并向我们发送您的成功案例。
