# 第四章. 标注单词和标记

在本章中，我们将介绍以下食谱：

+   有趣短语检测

+   前景或背景驱动的有趣短语检测

+   隐藏马尔可夫模型 (HMM) – 词性标注

+   N-best 单词标注

+   基于置信度的标注

+   训练单词标注

+   单词标注评估

+   条件随机场 (CRF) 用于词/标记标注

+   修改 CRF

# 简介

单词和标记是本章的重点。最常见的提取技术，如命名实体识别，实际上已经编码在本章介绍的概念中，但这一点将留待 [第五章](part0061_split_000.html#page "第五章. 在文本中查找跨度 – 分块")，*在文本中查找跨度 – 分块* 中讨论。我们将从查找有趣的标记集开始，然后转向 HMM，并以 LingPipe 最复杂的组件之一结束。像往常一样，我们将向您展示如何评估标注并训练自己的标注器。

# 有趣短语检测

想象一下，一个程序可以自动从大量文本数据中找到有趣的片段，其中“有趣”意味着单词或短语出现的频率高于预期。它有一个非常好的特性——不需要训练数据，并且适用于我们有标记的任何语言。您最常见到这种情况是在如下所示的标签云中：

![有趣短语检测](img/00010.jpeg)

前面的图显示了为 [lingpipe.com](http://lingpipe.com) 主页生成的标签云。然而，请注意，标签云被认为是杰弗里·泽尔达曼所说的互联网上的“莫霍克”，因此如果您在网站上部署此类功能，可能会处于不稳定的状态。

## 如何做...

要从关于迪士尼的小数据集（推文）中提取有趣的短语，请执行以下步骤：

1.  启动命令行并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar:lib/opencsv-2.4.jar com.lingpipe.cookbook.chapter4.InterestingPhrases

    ```

1.  程序应响应如下：

    ```py
    Score 42768.0 : Crayola Color 
    Score 42768.0 : Bing Rewards 
    Score 42768.0 : PassPorter Moms 
    Score 42768.0 : PRINCESS BATMAN 
    Score 42768.0 : Vinylmation NIB 
    Score 42768.0 : York City 
    Score 42768.0 : eternal damnation 
    Score 42768.0 : ncipes azules 
    Score 42768.0 : diventare realt 
    Score 42768.0 : possono diventare 
    ….
    Score 42768.0 : Pictures Releases 
    Score 42768.0 : SPACE MOUNTAIN 
    Score 42768.0 : DEVANT MOI 
    Score 42768.0 : QUOI DEVANT 
    Score 42768.0 : Lindsay Lohan 
    Score 42768.0 : EPISODE VII 
    Score 42768.0 : STAR WARS 
    Score 42768.0 : Indiana Jones 
    Score 42768.0 : Steve Jobs 
    Score 42768.0 : Smash Mouth

    ```

1.  您也可以提供一个 `.csv` 文件作为参数，以查看不同的数据。

输出往往令人难以置信地无用。令人难以置信地无用意味着一些有用的短语出现了，但伴随着大量您永远不会想在数据有趣总结中的不那么有趣的短语。在有趣的一侧，我们可以看到 `Crayola Color`、`Lindsey Lohan`、`Episode VII` 等等。在垃圾的一侧，我们可以看到 `ncipes azules`、`pictures releases` 等等。有许多方法可以解决垃圾输出——显然的第一步将是使用语言 ID 分类器来丢弃非英语。

## 它是如何工作的...

在这里，我们将从头到尾解释源代码：

```py
package com.lingpipe.cookbook.chapter4;

import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.SortedSet;
import au.com.bytecode.opencsv.CSVReader;
import com.aliasi.lm.TokenizedLM;
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.util.ScoredObject;

public class InterestingPhrases {
  static int TEXT_INDEX = 3;
  public static void main(String[] args) throws IOException {
    String inputCsv = args.length > 0 ? args[0] : "data/disney.csv";
```

在这里，我们看到路径、导入和 `main()` 方法。我们提供默认文件名或从命令行读取的三元运算符是最后一行：

```py
List<String[]> lines = Util.readCsv(new File(inputCsv));
int ngramSize = 3;
TokenizedLM languageModel = new TokenizedLM(IndoEuropeanTokenizerFactory.INSTANCE, ngramSize);
```

在收集输入数据后，第一个有趣的代码构建了一个标记化语言模型，它与[第1章](part0014_split_000.html#page "第1章。简单分类器")中使用的字符语言模型有显著的不同。标记化语言模型在`TokenizerFactory`创建的标记上操作，`ngram`参数指定了使用的标记数而不是字符数。`TokenizedLM`的一个细微之处在于，它还可以使用字符语言模型对它之前未看到的标记进行预测。请参阅*前景或背景驱动的有趣短语检测*配方以了解实际操作；除非在估计时没有未知标记，否则不要使用前面的构造函数。相关的Javadoc提供了更多详细信息。在以下代码片段中，语言模型被训练：

```py
for (String [] line: lines) {
  languageModel.train(line[TEXT_INDEX]);
}
```

下一个相关步骤是创建配对：

```py
int phraseLength = 2;
int minCount = 2;
int maxReturned = 100;
SortedSet<ScoredObject<String[]>> collocations = languageModel.collocationSet(phraseLength, minCount, maxReturned);
```

参数化控制短语在标记中的长度；它还设置短语可以出现的最小次数以及要返回的短语数量。由于我们有一个存储3元组的语言模型，我们可以查看长度为3的短语。接下来，我们将查看结果：

```py
for (ScoredObject<String[]> scoredTokens : collocations) {
  double score = scoredTokens.score();
  StringBuilder sb = new StringBuilder();
  for (String token : scoredTokens.getObject()) {
    sb.append(token + " ");
  }
  System.out.printf("Score %.1f : ", score);
  System.out.println(sb);
}
```

`SortedSet<ScoredObject<String[]>>` 配对从高分数到低分数排序。分数背后的直觉是，当标记一起出现次数超过预期时，会给予更高的分数，这取决于它们在训练数据中的单标记频率。换句话说，短语根据它们与基于标记的独立性假设的差异进行评分。有关确切定义，请参阅[http://alias-i.com/lingpipe/docs/api/com/aliasi/lm/TokenizedLM.html](http://alias-i.com/lingpipe/docs/api/com/aliasi/lm/TokenizedLM.html)中的Javadoc——一个有趣的练习是创建自己的评分标准并与LingPipe中使用的评分标准进行比较。

## 还有更多...

由于此代码接近在网站上使用，因此讨论调整是值得的。调整是查看系统输出并根据系统犯的错误进行更改的过程。我们立即会考虑的一些更改包括：

+   一个语言ID分类器，可以方便地过滤掉非英文文本

+   关于如何更好地标记数据的思考

+   变化的标记长度，包括3元组和单标记在摘要中

+   使用命名实体识别来突出显示专有名词

# 前景或背景驱动的有趣短语检测

与之前的配方类似，这个配方也寻找有趣的短语，但它使用另一个语言模型来确定什么是有趣的。亚马逊的统计不可能短语（**SIP**）就是这样工作的。您可以从他们的网站[http://www.amazon.com/gp/search-inside/sipshelp.html](http://www.amazon.com/gp/search-inside/sipshelp.html)获得清晰的了解：

> *"Amazon.com的统计不可能短语，或"SIPs"，是搜索内部!™计划中书籍文本中最独特的短语。为了识别SIPs，我们的计算机扫描搜索内部!计划中所有书籍的文本。如果它们在特定书籍中相对于所有搜索内部!书籍出现次数很多，那么这个短语就是该书籍中的SIP。*
> 
> SIPs在特定书籍中不一定是不可能的，但相对于搜索内部!中的所有书籍来说是不可能的。

前景模型将是正在处理的书籍，背景模型将是亚马逊搜索内部!™计划中的所有其他书籍。虽然亚马逊可能已经引入了不同的调整，但基本思想是相同的。

## 准备工作

有几个数据来源值得一看，以获取两个独立语言模型中的有趣短语。关键是你希望背景模型作为预期单词/短语分布的来源，这将有助于突出前景模型中的有趣短语。以下是一些例子：

+   **时间分离的Twitter数据**：时间分离的Twitter数据的例子如下：

    +   **背景模型**：这指的是昨天之前一年的关于迪士尼世界的推文。

    +   **前景模型**：今天的推文。

    +   **有趣短语**：今天在Twitter上关于迪士尼世界的新鲜事。

+   **主题分离的Twitter数据**：主题分离的Twitter数据的例子如下：

    +   **背景模型**：关于迪士尼乐园的推文

    +   **前景模型**：关于迪士尼世界的推文

    +   **有趣短语**：关于迪士尼世界所说但没有在迪士尼乐园说的内容

+   **非常相似主题的书籍**：相似主题书籍的例子如下：

    +   **背景模型**：一堆早期的科幻小说

    +   **前景模型**：儒勒·凡尔纳的*世界大战*

    +   **有趣短语**：关于“世界大战”的独特短语和概念

## 如何做到这一点...

在关于迪士尼乐园和迪士尼世界推文的推文中运行前景或背景模型的步骤如下：

1.  在命令行中输入：

    ```py
    java -cp  lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar:lib/opencsv-2.4.jar com.lingpipe.cookbook.chapter4.InterestingPhrasesForegroundBackground

    ```

1.  输出将类似于：

    ```py
    Score 989.621859 : [sleeping, beauty]
    Score 989.621859 : [california, adventure]
    Score 521.568529 : [winter, dreams]
    Score 367.309361 : [disneyland, resort]
    Score 339.429700 : [talking, about]
    Score 256.473825 : [disneyland, during]

    ```

1.  前景模型由搜索词`disneyland`的推文组成，背景模型由搜索词`disneyworld`的推文组成。

1.  最相关的结果是为了加州迪士尼乐园的独特特征，即城堡的名字、睡美人的城堡，以及建在加州迪士尼乐园停车场上的主题公园。

1.  下一个双词组是关于*Winter Dreams*的，它指的是一部电影的预映。

1.  总体来说，区分两个度假村的推文，输出还不错。

## 它是如何工作的...

代码位于`src/com/lingpipe/cookbook/chapter4/InterestingPhrasesForegroundBackground.java`。在加载前景和背景模型的原始`.csv`数据之后，解释开始：

```py
TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
tokenizerFactory = new LowerCaseTokenizerFactory(tokenizerFactory);
int minLength = 5;
tokenizerFactory = new LengthFilterTokenizerFactoryPreserveToken(tokenizerFactory, minLength);
```

```py
Chapter 2, *Finding and Working with Words*, but the third one is a customized factory that bears some examination. The intent behind the LengthFilterTokenizerFactoryPreserveToken class is to filter short tokens but at the same time not lose adjacency information. The goal is to take the phrase, "Disney is my favorite resort", and produce tokens (disney, _234, _235, favorite, resort), because we don't want short words in our interesting phrases—they tend to sneak past simple statistical models and mess up the output. Please refer to src/come/lingpipe/cookbook/chapter4/LengthFilterTokenizerFactoryPreserveToken.java for the source of the third tokenizer. Also, refer to Chapter 2, *Finding and Working with Words* for exposition. Next is the background model:
```

```py
int nGramOrder = 3;
TokenizedLM backgroundLanguageModel = new TokenizedLM(tokenizerFactory, nGramOrder);
for (String [] line: backgroundData) {
  backgroundLanguageModel.train(line[Util.TEXT_OFFSET]);
}
```

正在构建的是用于判断前景模型中短语新颖性的模型。然后，我们将创建并训练前景模型：

```py
TokenizedLM foregroundLanguageModel = new TokenizedLM(tokenizerFactory,nGramOrder);
for (String [] line: foregroundData) {
  foregroundLanguageModel.train(line[Util.TEXT_OFFSET]);
}
```

接下来，我们将从前景模型中访问`newTermSet()`方法。参数和`phraseSize`确定标记序列的长度；`minCount`指定要考虑的短语的最小实例数，而`maxReturned`控制要返回的结果数量：

```py
int phraseSize = 2;
int minCount = 3;
int maxReturned = 100;
SortedSet<ScoredObject<String[]>> suprisinglyNewPhrases
    = foregroundLanguageModel.newTermSet(phraseSize, minCount, maxReturned,backgroundLanguageModel);
for (ScoredObject<String[]> scoredTokens : suprisinglyNewPhrases) {
    double score = scoredTokens.score();
    String[] tokens = scoredTokens.getObject();
    System.out.printf("Score %f : ", score);
    System.out.println(java.util.Arrays.asList(tokens));
}
```

前面的`for`循环按从最令人惊讶的短语到最不令人惊讶的短语的顺序打印短语。

这里发生的事情的细节超出了食谱的范围，但Javadoc再次为我们指明了通往启迪的道路。

使用的精确评分方法是z分数，如`BinomialDistribution.z(double,int,int)`中定义的那样，成功概率由背景模型中的n-gram概率估计定义，成功次数是此模型中n-gram的计数，试验次数是此模型中的总计数。

## 还有更多...

这个食谱是我们第一次遇到未知标记的地方，如果不正确处理，这些标记可能会具有非常糟糕的特性。很容易看出为什么这是一个基于最大似然的语言模型的问题，这是一个语言模型的华丽名称，它通过乘以每个标记的似然来估计一些未见过的标记。每个似然是标记在训练中出现的次数除以在数据中看到的标记数。例如，考虑以下来自《亚瑟王宫廷中的康涅狄格州扬基》的训练数据：

> “本故事中提到的那些不温柔的习俗和历史事件都是历史的，用来说明它们的情节也是历史的。”

这是非常少的训练数据，但对于要说明的观点来说已经足够了。考虑一下我们如何使用我们的语言模型来估计短语“不温柔的岳父”的值。有24个单词包含一次出现的“The”，我们将分配1/24的概率给这个。我们也将1/24的概率分配给“ungentle”。如果我们在这里停止，我们可以说“The ungentle”的可能性是1/24 * 1/24。然而，下一个词是“inlaws”，它不在训练数据中。如果这个标记被分配0/24的值，这将使整个字符串的可能性变为0（1/24 * 1/24 * 0/20）。这意味着每当有一个未见过的标记，其估计很可能是零时，这通常是一个无益的特性。

对此问题的标准响应是用替代值和近似值来代替训练中未见过的数据。有几种解决此问题的方法：

+   为未知标记提供一个低但非零的估计。这是一个非常常见的做法。

+   使用包含未知标记的字符语言模型。类中有这方面的规定——请参阅Javadoc。

+   有许多其他方法和大量的研究文献。好的搜索词是“回退”和“平滑”。

# 隐藏马尔可夫模型（HMM）- 词性

这个配方引入了LingPipe的第一个核心语言能力；它指的是单词或**词性**（**POS**）的语法类别。文本中的动词、名词、形容词等等是什么？

## 如何做到这一点...

让我们直接跳进去，回到那些尴尬的中学英语课或我们等效的年份：

1.  像往常一样，前往你友好的命令提示符并输入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter9.PosTagger 

    ```

1.  系统将响应一个提示，我们将添加一个豪尔赫·路易斯·博尔赫斯（Jorge Luis Borges）的引言：

    ```py
    INPUT> Reality is not always probable, or likely.

    ```

1.  系统将愉快地响应这个引言：

    ```py
    Reality_nn is_bez not_* always_rb probable_jj ,_, or_cc likely_jj ._. 

    ```

每个标记后面都附加了`_`和词性标签；`nn`是名词，`rb`是副词，等等。完整的标签集和标注器的语料库描述可以在[http://en.wikipedia.org/wiki/Brown_Corpus](http://en.wikipedia.org/wiki/Brown_Corpus)找到。稍作尝试。词性标注器是90年代早期NLP中第一个突破性的机器学习应用之一。你可以期待这个应用的准确率超过90%，尽管由于基础语料库是在1961年收集的，因此在Twitter数据上可能会有所下降。

## 它是如何工作的...

正如食谱书所应有的，我们不会透露词性标注器构建的基本原理。有Javadoc、网络和科研文献来帮助你理解底层技术——在训练HMM的配方中，对底层HMM有一个简要的讨论。这是关于如何使用API的展示：

```py
public static void main(String[] args) throws ClassNotFoundException, IOException {
  TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  String hmmModelPath = args.length > 0 ? args[0] : "models/pos-en-general-brown.HiddenMarkovModel";
  HiddenMarkovModel hmm = (HiddenMarkovModel) AbstractExternalizable.readObject(new File(hmmModelPath));
  HmmDecoder decoder = new HmmDecoder(hmm);
  BufferedReader bufReader = new BufferedReader(new InputStreamReader(System.in));
  while (true) {
    System.out.print("\n\nINPUT> ");
    System.out.flush();
    String input = bufReader.readLine();
    Tokenizer tokenizer = tokFactory.tokenizer(input.toCharArray(),0,input.length());
    String[] tokens = tokenizer.tokenize();
    List<String> tokenList = Arrays.asList(tokens);
    firstBest(tokenList,decoder);
  }
}
```

代码首先设置`TokenizerFactory`，这是有道理的，因为我们需要知道将要获得词性的单词。下一行读取一个之前训练好的词性标注器作为`HiddenMarkovModel`。我们不会过多深入细节；你只需要知道HMM将根据其前面的标签分配为token *n* 分配一个词性标签。

这些标签在数据中不是直接观察到的，这使得马尔可夫模型是隐藏的。通常，会查看一个或两个标记之前的标记。HMM中有许多值得理解的事情在进行。

下一行使用`HmmDecoder`解码器将HMM包装在代码中以标注提供的标记。接下来是我们的标准交互式`while`循环，所有有趣的代码都在`firstBest(tokenList,decoder)`方法中。方法如下：

```py
static void firstBest(List<String> tokenList, HmmDecoder decoder) {
  Tagging<String> tagging = decoder.tag(tokenList);
    System.out.println("\nFIRST BEST");
    for (int i = 0; i < tagging.size(); ++i){
      System.out.print(tagging.token(i) + "_" + tagging.tag(i) + " ");
    }
  System.out.println();
}
```

注意`decoder.tag(tokenList)`调用，它产生一个`Tagging<String>`标签。标签没有迭代器或标签/标记对的封装，因此信息是通过增加索引i来访问的。

# N-best词性标注

计算机科学的确定性本质在语言学的不确定性中并未得到体现，在乔姆斯基的助手出现之前，合理的博士们可以至少同意或不同意。这个食谱使用在前面食谱中训练的相同HMM，但为每个单词提供可能的标记排名列表。

这可能在哪里有帮助？想象一个搜索引擎，它搜索单词和标记——不一定是词性。搜索引擎可以索引单词和前*n*-best标记，这将允许匹配到非最佳标记。这可以帮助提高召回率。

## 如何操作...

`N`-best分析推动了NLP开发者的复杂度边界。曾经是单例的现在变成了一个排名列表，但这是下一个性能级别的发生地。让我们通过执行以下步骤开始：

1.  将你的《句法结构》副本面朝下放好，并输入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.NbestPosTagger 

    ```

1.  然后，输入以下内容：

    ```py
    INPUT> Colorless green ideas sleep furiously.

    ```

1.  它产生了以下输出：

    ```py
    N BEST
    #   JointLogProb         Analysis
    0     -91.141  Colorless_jj   green_jj   ideas_nns  sleep_vb   furiously_rb   ._. 
    1     -93.916  Colorless_jj   green_nn   ideas_nns  sleep_vb   furiously_rb   ._. 
    2     -95.494  Colorless_jj   green_jj   ideas_nns  sleep_rb   furiously_rb   ._. 
    3     -96.266  Colorless_jj   green_jj   ideas_nns  sleep_nn   furiously_rb   ._. 
    4     -98.268  Colorless_jj   green_nn   ideas_nns  sleep_rb   furiously_rb   ._.

    ```

输出列表从最有可能到最不可能列出整个标记序列的估计，给定HMM的估计。记住，联合概率是以2为底的对数。要比较联合概率，从-93.9减去-91.1得到2.8的差异。所以，标记器认为选项1比选项0发生的机会低7倍。这种差异的来源在于将`green`分配为名词而不是形容词。

## 它的工作原理...

加载模型和命令I/O的代码与之前的食谱相同。不同之处在于获取和显示标记的方法：

```py
static void nBest(List<String> tokenList, HmmDecoder decoder, int maxNBest) {
  System.out.println("\nN BEST");
  System.out.println("#   JointLogProb         Analysis");
  Iterator<ScoredTagging<String>> nBestIt = decoder.tagNBest(tokenList,maxNBest);
  for (int n = 0; nBestIt.hasNext(); ++n) {
    ScoredTagging<String> scoredTagging = nBestIt.next();
    System.out.printf(n + "   %9.3f  ",scoredTagging.score());
    for (int i = 0; i < tokenList.size(); ++i){
      System.out.print(scoredTagging.token(i) + "_" + pad(scoredTagging.tag(i),5));
    }
    System.out.println();
  }
```

这并没有什么太多的事情，除了在迭代标记时解决格式问题。

# 基于置信度的标记

另一种看待标记概率的方法；这反映了单词层面的概率分配。代码反映了底层的`TagLattice`，并提供了关于标记器是否自信的见解。

## 如何操作...

这个食谱将聚焦于对单个标记的概率估计。执行以下步骤：

1.  在命令行或IDE等效环境中输入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.ConfidenceBasedTagger

    ```

1.  然后，输入以下内容：

    ```py
    INPUT> Colorless green ideas sleep furiously.

    ```

1.  它产生了以下输出：

    ```py
    CONFIDENCE
    #   Token          (Prob:Tag)*
    0   Colorless           0.991:jj       0.006:np$      0.002:np 
    1   green               0.788:jj       0.208:nn       0.002:nns 
    2   ideas               1.000:nns      0.000:rb       0.000:jj 
    3   sleep               0.821:vb       0.101:rb       0.070:nn 
    4   furiously           1.000:rb       0.000:ql       0.000:jjr 
    5   .                   1.000:.        0.000:np       0.000:nn 

    ```

这种数据视图将标记和单词的联合概率分布。我们可以看到，`green`被标记为`nn`或单数名词的概率是`.208`，但正确的分析仍然是`.788`，带有形容词`jj`。

## 它的工作原理…

我们仍然使用与《隐藏马尔可夫模型（HMM）- 词性》食谱中相同的旧HMM，但使用它的不同部分。读取模型的代码完全相同，但在报告结果的方式上有重大差异。`src/com/lingpipe/cookbook/chapter4/ConfidenceBasedTagger.java`中的方法：

```py
static void confidence(List<String> tokenList, HmmDecoder decoder) {
  System.out.println("\nCONFIDENCE");
  System.out.println("#   Token          (Prob:Tag)*");
  TagLattice<String> lattice = decoder.tagMarginal(tokenList);

  for (int tokenIndex = 0; tokenIndex < tokenList.size(); ++tokenIndex) {
    ConditionalClassification tagScores = lattice.tokenClassification(tokenIndex);
    System.out.print(pad(Integer.toString(tokenIndex),4));
    System.out.print(pad(tokenList.get(tokenIndex),15));

    for (int i = 0; i < 3; ++i) {
      double conditionalProb = tagScores.score(i);
      String tag = tagScores.category(i);
      System.out.printf(" %9.3f:" + pad(tag,4),conditionalProb);

    }
    System.out.println();
  }
}
```

该方法明确展示了标记的底层图，这是HMM的核心。通过改变`for`循环的终止条件来查看更多或更少的标记。

# 训练词标记

当你可以创建自己的模型时，词性标注就变得更有趣了。对词性标注语料库进行标注的领域对于一本简单的食谱书来说有点过于广泛——对词性数据的标注非常困难，因为它需要相当的语言学知识才能做好。这个食谱将直接解决基于HMM的句子检测器的机器学习部分。

由于这是一本食谱书，我们将最小化解释HMM是什么。我们一直在使用的标记语言模型在计算当前估计的单词的前一个上下文时，考虑了一些单词/标记的数量。HMM在计算当前标记的标签估计时，会考虑到一些前一个标签的长度。这使得看似不同的邻居，如 `of` 和 `in`，看起来相似，因为它们都是介词。

在 *句子检测* 食谱中，从 [第5章](part0061_split_000.html#page "第5章. 在文本中查找范围 – 分块")，*在文本中查找范围 – 分块*，一个有用但不太灵活的句子检测器基于LingPipe中的 `HeuristicSentenceModel`。我们不会去修改/扩展 `HeuristicSentenceModel`，而是将使用我们标注的数据构建一个基于机器学习的句子系统。

## 如何做到这一点...

以下步骤描述了如何在 `src/com/lingpipe/cookbook/chapter4/HMMTrainer.java` 中运行程序：

1.  要么创建一个句子标注数据的新的语料库，要么使用以下默认数据，该数据位于 `data/connecticut_yankee_EOS.txt`。如果你正在自己生成数据，只需用 `['` 和 `']` 编辑一些文本以标记句子边界。我们的例子如下：

    ```py
    [The ungentle laws and customs touched upon in this tale are
    historical, and the episodes which are used to illustrate them
    are also historical.] [It is not pretended that these laws and
    customs existed in England in the sixth century; no, it is only
    pretended that inasmuch as they existed in the English and other
    civilizations of far later times, it is safe to consider that it is
    no libel upon the sixth century to suppose them to have been in
    practice in that day also.] [One is quite justified in inferring
    that whatever one of these laws or customs was lacking in that
    remote time, its place was competently filled by a worse one.]
    ```

1.  前往命令提示符并使用以下命令运行程序：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter4.HmmTrainer

    ```

1.  它将给出以下输出：

    ```py
    Training The/BOS ungentle/WORD laws/WORD and/WORD customs/WORD touched/WORD…
    done training, token count: 123
    Enter text followed by new line
    > The cat in the hat. The dog in a bog.
    The/BOS cat/WORD in/WORD the/WORD hat/WORD ./EOS The/BOS dog/WORD in/WORD a/WORD bog/WORD ./EOS

    ```

1.  输出是一个标记化文本，包含三个标签之一：`BOS` 表示句子的开始，`EOS` 表示句子的结束，以及 `WORD` 表示所有其他标记。

## 它是如何工作的…

就像许多基于范围的标记一样，`span` 标注被转换成之前在食谱输出中展示的标记级别标注。因此，首要任务是收集标注文本，设置 `TokenizerFactory`，然后调用一个解析子例程将其添加到 `List<Tagging<String>>`：

```py
public static void main(String[] args) throws IOException {
  String inputFile = args.length > 0 ? args[0] : "data/connecticut_yankee_EOS.txt";
  char[] text = Files.readCharsFromFile(new File(inputFile), Strings.UTF8);
  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  List<Tagging<String>> taggingList = new ArrayList<Tagging<String>>();
  addTagging(tokenizerFactory,taggingList,text);
```

解析前述格式的子例程通过首先使用 `IndoEuropeanTokenizer` 对文本进行标记化来实现，该标记化器具有将 `['` 和 `']` 句子分隔符作为单独标记的有益特性。它不会检查句子分隔符是否格式正确——需要更健壮的解决方案来完成这项工作。棘手的部分在于我们希望忽略结果标记流中的这种标记，但同时又想使用它来确保跟随 `['` 的标记是 BOS，而跟随 `']` 的标记是 EOS。其他标记只是 `WORD`。子例程为标签和标记构建了一个并行 `Lists<String>` 实例，然后用于创建 `Tagging<String>` 并添加到 `taggingList` 中。第 2 章（part0027_split_000.html#page "Chapter 2. Finding and Working with Words"）*Finding and Working with Words* 中的标记化配方涵盖了标记化器正在处理的内容。查看以下代码片段：

```py
static void addTagging(TokenizerFactory tokenizerFactory, List<Tagging<String>> taggingList, char[] text) {
  Tokenizer tokenizer = tokenizerFactory.tokenizer(text, 0, text.length);
  List<String> tokens = new ArrayList<String>();
  List<String> tags = new ArrayList<String>();
  boolean bosFound = false;
  for (String token : tokenizer.tokenize()) {
    if (token.equals("[")) {
      bosFound = true;
    }
    else if (token.equals("]")) {
      tags.set(tags.size() - 1,"EOS");
    }
    else {
      tokens.add(token);
      if (bosFound) {
        tags.add("BOS");
        bosFound = false;
      }
      else {
        tags.add("WORD");
      }
    }
  }
  if (tokens.size() > 0) {
    taggingList.add(new Tagging<String>(tokens,tags));
  }
}
```

前述代码有一个细微之处。训练数据被视为单个标记——这将模拟当我们使用句子检测器对新颖数据进行处理时输入将看起来像什么。如果使用多个文档/章节/段落进行训练，那么我们将为每个文本块调用此子例程。

返回到 `main()` 方法，我们将设置 `ListCorpus` 并将标记添加到语料库的训练部分，一次添加一个标记。还有一个 `addTest()` 方法，但这个配方不涉及评估；如果是评估，我们可能无论如何都会使用 `XValidatingCorpus`：

```py
ListCorpus<Tagging<String>> corpus = new ListCorpus<Tagging<String>> ();
for (Tagging<String> tagging : taggingList) {
  System.out.println("Training " + tagging);
  corpus.addTrain(tagging);
}
```

接下来，我们将创建 `HmmCharLmEstimator`，这是我们使用的 HMM。请注意，有一些构造函数允许自定义参数，这些参数会影响性能——请参阅 Javadoc。接下来，估计器将对语料库进行训练，并创建 `HmmDecoder`，它将实际标记标记，如以下代码片段所示：

```py
HmmCharLmEstimator estimator = new HmmCharLmEstimator();
corpus.visitTrain(estimator);
System.out.println("done training, token count: " + estimator.numTrainingTokens());
HmmDecoder decoder = new HmmDecoder(estimator);
```

```py
Note that there is no requirement that the training tokenizer be the same as the production tokenizer, but one must be careful to not tokenize in a radically different way; otherwise, the HMM will not be seeing the tokens it was trained with. The back-off model will then be used, which will likely degrade performance. Have a look at the following code snippet:
```

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.print("Enter text followed by new line\n>");
  String evalText = reader.readLine();
  Tokenizer tokenizer = tokenizerFactory.tokenizer(evalText.toCharArray(),0,evalText.length());
  List<String> evalTokens = Arrays.asList(tokenizer.tokenize());
  Tagging<String> evalTagging = decoder.tag(evalTokens);
  System.out.println(evalTagging);
}
```

就这些了！为了真正将其作为一个合适的句子检测器，我们需要将字符偏移量映射回原始文本，但这在第 5 章（part0061_split_000.html#page "Chapter 5. Finding Spans in Text – Chunking"）*Finding Spans in Text – Chunking* 中有所介绍。这足以展示如何使用 HMM。一个更完整的功能方法将确保每个 BOS 都有一个匹配的 EOS，反之亦然。HMM 没有这样的要求。

## 还有更多...

我们有一个小型且易于使用的词性标注语料库；这使我们能够展示如何训练用于非常不同问题的 HMM 与训练简单分类器（[Chapter 1](part0014_split_000.html#page "Chapter 1. Simple Classifiers"）*Simple Classifiers*）中的配方相同。它就像我们的 *如何对情感进行分类 – 简单版本* 配方，在 [Chapter 1](part0014_split_000.html#page "Chapter 1. Simple Classifiers") *Simple Classifiers*；语言 ID 和情感之间的唯一区别是训练数据。我们将从一个硬编码的语料库开始，以保持简单——它位于 `src/com/lingpipe/cookbook/chapter4/TinyPosCorus.java`：

```py
public class TinyPosCorpus extends Corpus<ObjectHandler<Tagging<String>>> {

  public void visitTrain(ObjectHandler<Tagging<String>> handler) {
    for (String[][] wordsTags : WORDS_TAGSS) {
      String[] words = wordsTags[0];
      String[] tags = wordsTags[1];
      Tagging<String> tagging = new Tagging<String>(Arrays.asList(words),Arrays.asList(tags));
      handler.handle(tagging);
    }
  }

  public void visitTest(ObjectHandler<Tagging<String>> handler) {
    /* no op */
  }

  static final String[][][] WORDS_TAGSS = new String[][][] {
    { { "John", "ran", "." },{ "PN", "IV", "EOS" } },
    { { "Mary", "ran", "." },{ "PN", "IV", "EOS" } },
    { { "John", "jumped", "!" },{ "PN", "IV", "EOS" } },
    { { "The", "dog", "jumped", "!" },{ "DET", "N", "IV", "EOS" } },
    { { "The", "dog", "sat", "." },{ "DET", "N", "IV", "EOS" } },
    { { "Mary", "sat", "!" },{ "PN", "IV", "EOS" } },
    { { "Mary", "likes", "John", "." },{ "PN", "TV", "PN", "EOS" } },
    { { "The", "dog", "likes", "Mary", "." }, { "DET", "N", "TV", "PN", "EOS" } },
    { { "John", "likes", "the", "dog", "." }, { "PN", "TV", "DET", "N", "EOS" } },
    { { "The", "dog", "ran", "." },{ "DET", "N", "IV", "EOS", } },
    { { "The", "dog", "ran", "." },{ "DET", "N", "IV", "EOS", } }
  };
```

语料库手动创建标记以及标记中的静态 `WORDS_TAGS` 中的标记，并为每个句子创建 `Tagging<String>`；在这种情况下，`Tagging<String>` 由两个对齐的 `List<String>` 实例组成。然后，标记被发送到 `Corpus` 超类的 `handle()` 方法。替换这个语料库看起来如下：

```py
/*
List<Tagging<String>> taggingList = new ArrayList<Tagging<String>>();
addTagging(tokenizerFactory,taggingList,text);
ListCorpus<Tagging<String>> corpus = new ListCorpus<Tagging<String>> ();
for (Tagging<String> tagging : taggingList) {
  System.out.println("Training " + tagging);
  corpus.addTrain(tagging);
}
*/

Corpus<ObjectHandler<Tagging<String>>> corpus = new TinyPosCorpus();
HmmCharLmEstimator estimator = new HmmCharLmEstimator();
corpus.visitTrain(estimator);
```

我们只是注释掉了在 `TinyPosCorpus` 中加载带有句子检测和特征的代码，取而代之。由于它不需要添加数据，我们将仅用此训练HMM。为了避免混淆，我们创建了一个单独的类 `HmmTrainerPos.java`。运行它会产生以下结果：

```py
java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar 
done training, token count: 42
Enter text followed by new line
> The cat in the hat is back.
The/DET cat/N in/TV the/DET hat/N is/TV back/PN ./EOS

```

唯一的错误是 `in` 是一个及物动词 `TV`。由于训练数据非常小，所以错误是可以预料的。就像在[第1章](part0014_split_000.html#page "Chapter 1. Simple Classifiers")中语言ID和情感分类的差异一样，HMM通过改变训练数据来学习一个非常不同的现象。

# 词性标注评估

词性标注评估推动了下游技术，如命名实体检测的发展，反过来，又推动了高端应用，如指代消解。您会注意到，大部分评估与我们的分类器评估相似，只是每个标记都像其自己的分类器类别一样进行评估。

这个菜谱应该可以帮助您开始评估，但请注意，我们网站上有一个关于词性标注评估的优秀教程[http://alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html)；这个菜谱更详细地介绍了如何最好地理解标记器的性能。

这个菜谱简短且易于使用，所以你没有不评估你的标记器的借口。

## 准备工作

以下是我们评估器的类源代码，位于 `src/com/lingpipe/cookbook/chapter4/TagEvaluator.java`：

```py
public class TagEvaluator {
  public static void main(String[] args) throws ClassNotFoundException, IOException {
    HmmDecoder decoder = null;
    boolean storeTokens = true;
    TaggerEvaluator<String> evaluator = new TaggerEvaluator<String>(decoder,storeTokens);
    Corpus<ObjectHandler<Tagging<String>>> smallCorpus = new TinyPosCorpus();
    int numFolds = 10;
    XValidatingObjectCorpus<Tagging<String>> xValCorpus = new XValidatingObjectCorpus<Tagging<String>>(numFolds);
    smallCorpus.visitCorpus(xValCorpus);
    for (int i = 0; i < numFolds; ++i) {
      xValCorpus.setFold(i);
      HmmCharLmEstimator estimator = new HmmCharLmEstimator();
      xValCorpus.visitTrain(estimator);
      System.out.println("done training " + estimator.numTrainingTokens());
      decoder = new HmmDecoder(estimator);
      evaluator.setTagger(decoder);
      xValCorpus.visitTest(evaluator);
    }
    BaseClassifierEvaluator<String> classifierEval = evaluator.tokenEval();
    System.out.println(classifierEval);
  }
}
```

## 如何操作...

我们将指出前面代码中的有趣部分：

1.  首先，我们将使用空 `HmmDecoder` 和控制标记是否存储的 `boolean` 来设置 `TaggerEvaluator`。`HmmDecoder` 对象将在代码中的交叉验证部分稍后设置：

    ```py
    HmmDecoder decoder = null;
    boolean storeTokens = true;
    TaggerEvaluator<String> evaluator = new TaggerEvaluator<String>(decoder,storeTokens);
    ```

1.  接下来，我们将从上一个菜谱中加载 `TinyPosCorpus` 并用它来填充 `XValididatingObjectCorpus`——这是一个非常巧妙的技巧，允许轻松地在语料库类型之间进行转换。请注意，我们选择了10个折——语料库只有11个训练示例，所以我们希望最大化每个折的训练数据量。如果您对这个概念不熟悉，请参阅[第1章](part0014_split_000.html#page "Chapter 1. Simple Classifiers")中的*如何使用交叉验证进行训练和评估*菜谱，*简单分类器*。查看以下代码片段：

    ```py
    Corpus<ObjectHandler<Tagging<String>>> smallCorpus = new TinyPosCorpus();
    int numFolds = 10;
    XValidatingObjectCorpus<Tagging<String>> xValCorpus = new XValidatingObjectCorpus<Tagging<String>>(numFolds);
    smallCorpus.visitCorpus(xValCorpus);
    ```

1.  以下代码片段是一个 `for()` 循环，它遍历折的数量。循环的前半部分处理训练：

    ```py
    for (int i = 0; i < numFolds; ++i) {
      xValCorpus.setFold(i);
      HmmCharLmEstimator estimator = new HmmCharLmEstimator();
      xValCorpus.visitTrain(estimator);
      System.out.println("done training " + estimator.numTrainingTokens());
    ```

1.  循环的其余部分首先为HMM创建解码器，将评估器设置为使用此解码器，然后将配置适当的评估器应用于语料库的测试部分：

    ```py
    decoder = new HmmDecoder(estimator);
    evaluator.setTagger(decoder);
    xValCorpus.visitTest(evaluator);
    ```

1.  最后几行在语料库的所有折都用于训练和测试之后应用。注意，评估器是`BaseClassifierEvaluator`！它报告每个标签作为类别：

    ```py
    BaseClassifierEvaluator<String> classifierEval = evaluator.tokenEval();
    System.out.println(classifierEval);
    ```

1.  准备迎接评估的洪流。以下只是其中的一小部分，即你应该从[第1章](part0014_split_000.html#page "Chapter 1. Simple Classifiers")*简单分类器*中熟悉的混淆矩阵：

    ```py
    Confusion Matrix
    reference \ response
      ,DET,PN,N,IV,TV,EOS
      DET,4,2,0,0,0,0
      PN,0,7,0,1,0,0
      N,0,0,4,1,1,0
      IV,0,0,0,8,0,0
      TV,0,1,0,0,2,0
      EOS,0,0,0,0,0,11
    ```

就这样。你有一个与[第1章](part0014_split_000.html#page "Chapter 1. Simple Classifiers")*简单分类器*中的分类器评估紧密相关的评估设置。

## 还有更多...

对于n-best词标注有评估类，即`NBestTaggerEvaluator`和`MarginalTaggerEvaluator`，用于置信度排名。再次提醒，查看关于词性标注的更详细教程，以获得关于评估指标和一些帮助调整HMM的示例软件的全面介绍。

# 用于词/标记标注的条件随机场（CRF）

**条件随机场**（**CRF**）是[第3章](part0036_split_000.html#page "Chapter 3. Advanced Classifiers")*高级分类器*中*逻辑回归*公式的扩展，但应用于词标注。在[第1章](part0014_split_000.html#page "Chapter 1. Simple Classifiers")*简单分类器*的结尾，我们讨论了将问题编码为分类问题的各种方法。CRFs将序列标注问题视为寻找最佳类别，其中每个类别（C）是C*T标签（T）分配给标记之一。

例如，如果我们有标记`The`和`rain`，并且将`d`标记为限定词，将`n`标记为名词，那么CRF分类器的类别集合是：

+   **类别1**：`d d`

+   **类别2**：`n d`

+   **类别3**：`n n`

+   **类别4**：`d d`

为了使这个组合噩梦可计算，已经应用了各种优化，但这是基本思路。疯狂，但有效。

此外，CRFs允许在训练中使用随机特征，这与逻辑回归用于分类的方式完全相同。此外，它具有针对HMM风格观察的优化数据结构。它在词性标注方面的应用并不令人兴奋，因为我们的当前HMMs已经非常接近最先进水平。CRFs真正发挥作用的地方是在诸如命名实体识别等用例中，这些用例在[第5章](part0061_split_000.html#page "Chapter 5. Finding Spans in Text – Chunking")*在文本中查找跨度 - 分块*中有介绍，但我们希望在将分块接口复杂化之前先解决纯CRF实现问题。

在[http://alias-i.com/lingpipe/demos/tutorial/crf/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/crf/read-me.html)有一个关于CRF的优秀详细教程；这道菜谱非常接近这个教程。你将在那里找到更多信息以及适当的引用。

## 如何做到这一点…

我们至今所展示的所有技术都是在上一个千年发明的；这是一个来自新千年的技术。执行以下步骤：

1.  在命令行中，输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.CRFTagger

    ```

1.  控制台继续显示收敛结果，这些结果应该与[第3章](part0036_split_000.html#page "第3章。高级分类器")中的*逻辑回归*菜谱中熟悉的结果一样，我们将得到标准的命令提示符：

    ```py
    Enter text followed by new line
    >The rain in Spain falls mainly on the plain.

    ```

1.  作为对此的回应，我们将得到一些相当混乱的输出：

    ```py
    The/DET rain/N in/TV Spain/PN falls/IV mainly/EOS on/DET the/N plain/IV ./EOS

    ```

1.  这是一个糟糕的输出，但CRF已经在11个句子上进行了训练。所以，我们不要太苛刻——尤其是考虑到这项技术主要在给足训练数据以完成其工作时，在词性标注和跨度标注方面占据主导地位。

## 它是如何工作的…

就像逻辑回归一样，我们需要执行许多配置相关的任务来使这个类运行起来。这道菜谱将解决代码中CRF特定的方面，并参考[第3章](part0036_split_000.html#page "第3章。高级分类器")中的*逻辑回归*菜谱，*高级分类器*中的逻辑回归配置方面。

从`main()`方法的顶部开始，我们将获取我们的语料库，这在前面三道菜谱中已经讨论过：

```py
Corpus<ObjectHandler<Tagging<String>>> corpus = new TinyPosCorpus();
```

接下来是特征提取器，它是CRF训练器的实际输入。它之所以是最终的，唯一的原因是有一个匿名内部类将访问它，以展示在下一道菜谱中特征提取是如何工作的：

```py
final ChainCrfFeatureExtractor<String> featureExtractor
  = new SimpleCrfFeatureExtractor();
```

我们将在菜谱的后面部分讨论这个类的工作方式。

下一个配置块是为底层逻辑回归算法准备的。有关此方面的更多信息，请参阅[第3章](part0036_split_000.html#page "第3章。高级分类器")中的*逻辑回归*菜谱，*高级分类器*。请查看以下代码片段：

```py
boolean addIntercept = true;
int minFeatureCount = 1;
boolean cacheFeatures = false;
boolean allowUnseenTransitions = true;
double priorVariance = 4.0;
boolean uninformativeIntercept = true;
RegressionPrior prior = RegressionPrior.gaussian(priorVariance, uninformativeIntercept);
int priorBlockSize = 3;
double initialLearningRate = 0.05;
double learningRateDecay = 0.995;
AnnealingSchedule annealingSchedule = AnnealingSchedule.exponential(initialLearningRate,
  learningRateDecay);
double minImprovement = 0.00001;
int minEpochs = 2;
int maxEpochs = 2000;
Reporter reporter = Reporters.stdOut().setLevel(LogLevel.INFO);
```

接下来，使用以下内容训练CRF：

```py
System.out.println("\nEstimating");
ChainCrf<String> crf = ChainCrf.estimate(corpus,featureExtractor,addIntercept,minFeatureCount,cacheFeatures,allowUnseenTransitions,prior,priorBlockSize,annealingSchedule,minImprovement,minEpochs,maxEpochs,reporter);
```

代码的其余部分只是使用了标准的I/O循环。有关`tokenizerFactory`的工作方式，请参阅[第2章](part0027_split_000.html#page "第2章。查找和使用单词")，*查找和使用单词*。

```py
TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.print("Enter text followed by new line\n>");
  System.out.flush();
  String text = reader.readLine();
  Tokenizer tokenizer = tokenizerFactory.tokenizer(text.toCharArray(),0,text.length());
  List<String> evalTokens = Arrays.asList(tokenizer.tokenize());
  Tagging<String> evalTagging = crf.tag(evalTokens);
  System.out.println(evalTagging);
```

### SimpleCrfFeatureExtractor

现在，我们将进入特征提取器。提供的实现紧密模仿了标准HMM的特征。在`com/lingpipe/cookbook/chapter4/SimpleCrfFeatureExtractor.java`中的类开始于：

```py
public class SimpleCrfFeatureExtractor implements ChainCrfFeatureExtractor<String> {
  public ChainCrfFeatures<String> extract(List<String> tokens, List<String> tags) {
    return new SimpleChainCrfFeatures(tokens,tags);
  }
```

`ChainCrfFeatureExtractor` 接口需要一个 `extract()` 方法，该方法接受标记和相关标签，这些标签在此情况下被转换为 `ChainCrfFeatures<String>`。这是通过以下 `SimpleChainCrfFeatures` 下的内部类处理的；这个内部类扩展了 `ChainCrfFeatures` 并提供了 `nodeFeatures()` 和 `edgeFeatures()` 抽象方法的实现：

```py
static class SimpleChainCrfFeatures extends ChainCrfFeatures<String> {
```

以下构造函数传递标记和标签到超类，超类将进行账目管理以支持查找 `tags` 和 `tokens`：

```py
public SimpleChainCrfFeatures(List<String> tokens, List<String> tags) {
  super(tokens,tags);
}
```

节点特征的计算如下：

```py
public Map<String,Double> nodeFeatures(int n) {
  ObjectToDoubleMap<String> features = new ObjectToDoubleMap<String>();
  features.increment("TOK_" + token(n),1.0);
  return features;
}
```

标记按其在句子中的位置进行索引。位置 `n` 的单词/标记的节点特征是 `ChainCrfFeatures` 的基类方法 `token(n)` 返回的 `String` 值，带有前缀 `TOK_`。这里的值是 `1.0`。特征值可以有效地调整到其他值，这对于更复杂的 CRF 方法很有用，例如使用其他分类器的置信估计。以下是一个示例。

与 HMMs 一样，有一些特征依赖于输入中的其他位置——这些被称为 **边缘特征**。边缘特征接受两个参数：一个用于为 `n` 和 `k` 生成特征的参数，这将应用于句子中的所有其他位置：

```py
public Map<String,Double> edgeFeatures(int n, int k) {
  ObjectToDoubleMap<String> features = new ObjectToDoubleMap<String>();
  features.increment("TAG_" + tag(k),1.0);
  return features;
}
```

下一个方法将说明如何修改特征提取。

## 还有更多...

Javadoc 中引用了广泛的研究文献，LingPipe 网站上还有更详尽的教程。

# 修改 CRFs

CRFs 的强大和吸引力来自于丰富的特征提取——继续使用一个提供反馈的评估工具来探索。这个方法将详细说明如何创建更复杂的特征。

## 如何操作...

我们将不会训练和运行一个 CRF；相反，我们将打印出特征。用这个特征提取器替换上一个方法中的特征提取器，以查看它们的工作情况。执行以下步骤：

1.  前往命令行并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.ModifiedCrfFeatureExtractor

    ```

1.  特征提取器类为训练数据中的每个标记输出正在使用的真实标签，以用于学习：

    ```py
    -------------------
    Tagging:  John/PN

    ```

1.  这反映了由 `src/com/lingpipe/cookbook/chapter4/TinyPosCorpus.java` 确定的标记 `John` 的训练标签。

1.  节点特征遵循来自我们 Brown 语料库 HMM 标签器的三个最高 POS 标签以及 `TOK_John` 特征：

    ```py
    Node Feats:{nps=2.0251355582754984E-4, np=0.9994337160349874, nn=2.994165140854113E-4, TOK_John=1.0}
    ```

1.  接下来，显示句子 "John ran" 中其他标记的边缘特征：

    ```py
    Edge Feats:{TOKEN_SHAPE_LET-CAP=1.0, TAG_PN=1.0}
    Edge Feats:{TAG_IV=1.0, TOKEN_SHAPE_LET-CAP=1.0}
    Edge Feats:{TOKEN_SHAPE_LET-CAP=1.0, TAG_EOS=1.0}
    ```

1.  输出的其余部分是句子中剩余标记的特征，然后是 `TinyPosCorpus` 中剩余句子的特征。

## 它是如何工作的...

我们的特性提取代码位于 `src/com/lingpipe/cookbook/chapter4/ModifiedCrfFeatureExtractor.java`。我们将从 `main()` 方法开始，该方法加载语料库，将内容传递给特征提取器，并打印出来：

```py
public static void main(String[] args) throws IOException, ClassNotFoundException {

  Corpus <ObjectHandler<Tagging<String>>> corpus = new TinyPosCorpus();
  final ChainCrfFeatureExtractor<String> featureExtractor = new ModifiedCrfFeatureExtractor();
```

我们将使用之前步骤中的`TinyPosCorpus`作为我们的语料库，然后，我们将从包含的类中创建一个特征提取器。在匿名内部类中引用变量时需要使用`final`。

对于匿名内部类表示歉意，但这只是访问语料库中存储内容的简单方法，原因有很多，比如复制和打印。在这种情况下，我们只是生成和打印训练数据中找到的特征：

```py
corpus.visitCorpus(new ObjectHandler<Tagging<String>>() {
  @Override
  public void handle(Tagging<String> tagging) {
    ChainCrfFeatures<String> features = featureExtractor.extract(tagging.tokens(), tagging.tags());
```

语料库包含`Tagging`对象，它们反过来包含一个`List<String>`的标记和标记。然后，通过将`featureExtractor.extract()`方法应用于标记和标记来创建一个`ChainCrfFeatures<String>`对象，这将涉及大量的计算，如下所示。

接下来，我们将使用标记和预期的标记报告训练数据：

```py
for (int i = 0; i < tagging.size(); ++i) {
  System.out.println("---------");
  System.out.println("Tagging:  " + tagging.token(i) + "/" + tagging.tag(i));
```

然后，我们将使用以下节点先前标记的节点来告知CRF模型使用的特征：

```py
System.out.println("Node Feats:" + features.nodeFeatures(i));
```

然后，通过以下迭代相对位置到源节点`i`来生成边特征：

```py
for (int j = 0; j < tagging.size(); ++j) {
  System.out.println("Edge Feats:" 
        + features.edgeFeatures(i, j));
}
```

这就是打印特征的步骤。现在，我们将讨论如何构建特征提取器。我们假设你已经熟悉之前的步骤。首先，引入布朗语料库POS标记器的构造器：

```py
HmmDecoder mDecoder;

public ModifiedCrfFeatureExtractor() throws IOException, ClassNotFoundException {
  File hmmFile = new File("models/pos-en-general-" + "brown.HiddenMarkovModel");
  HiddenMarkovModel hmm = (HiddenMarkovModel)AbstractExternalizable.readObject(hmmFile);
  mDecoder = new HmmDecoder(hmm);
}
```

构造器引入了一些用于特征生成的外部资源，即一个在布朗语料库上训练的POS标记器。为什么要在POS标记器中涉及另一个POS标记器？我们将布朗POS标记器的角色称为“特征标记器”，以区别于我们试图构建的标记器。使用特征标记器的一些原因包括：

+   我们使用一个非常小的语料库进行训练，一个更健壮的通用POS特征标记器将有助于解决问题。"TinyPosCorpus"对于这种好处来说太小了，但有了更多数据，存在一个将`the`、`a`和`some`统一起来的特征`at`将有助于CRF识别`some dog`是`'DET'` `'N'`，即使它从未在训练中看到`some`。

+   我们不得不与不与POS特征标记器对齐的标记集一起工作。CRF可以使用这些观察结果在外国标记集中更好地推理所需的标记。最简单的情况是，来自布朗语料库标记集的`at`在这个标记集中干净地映射到`DET`。

+   可以通过运行在训练数据上训练的不同数据或使用不同技术进行标记的多个标记器来提高性能。CRF可以，希望如此，识别出某个标记器优于其他标记器的上下文，并使用这些信息来指导分析。在那些日子里，我们的MUC-6系统有3个POS标记器，它们为最佳输出投票。让CRF来解决这个问题将是一个更优越的方法。

特征提取的核心可以通过`extract`方法访问：

```py
public ChainCrfFeatures<String> extract(List<String> tokens, List<String> tags) {
  return new ModChainCrfFeatures(tokens,tags);
}
```

`ModChainCrfFeatures` 被创建为一个内部类，目的是将类的数量保持在最小，并且包含它的类非常轻量级：

```py
class ModChainCrfFeatures extends ChainCrfFeatures<String> {

  TagLattice<String> mBrownTaggingLattice;

  public ModChainCrfFeatures(List<String> tokens, List<String> tags) {
    super(tokens,tags);
    mBrownTaggingLattice = mDecoder.tagMarginal(tokens);
  }
```

前一个构造函数将标记和标签传递给超类，该超类处理此数据的管理。然后，将“特征标记器”应用于标记，并将结果输出分配给成员变量 `mBrownTaggingLattice`。代码将逐个访问标记，因此必须现在计算。

特征创建步骤通过两种方法进行：`nodeFeatures` 和 `edgeFeatures`。我们将从对之前食谱中 `edgeFeatures` 的简单增强开始：

```py
public Map<String,? extends Number> edgeFeatures(int n, int k) {
  ObjectToDoubleMap<String> features = newObjectToDoubleMap<String>();
  features.set("TAG_" + tag(k), 1.0d);
  String category = IndoEuropeanTokenCategorizer.CATEGORIZER.categorize(token(n));
  features.set("TOKEN_SHAPE_" + category,1.0d);
  return features;
}
```

代码添加了一个形状为标记的特征，将 `12` 和 `34` 通用于 `2-DIG` 以及许多其他通用的形式。对于 CRF 来说，除非特征提取表明不同，否则 `12` 和 `34` 作为两位数的相似性是不存在的。请参阅 Javadoc 以获取完整的分类器输出。

### 候选边特征

CRF 允许应用随机特征，因此问题是什么特征是有意义的。边特征与节点特征一起使用，因此另一个问题是特征是否应该应用于边或节点。边特征将用于推理当前单词/标记与其周围标记之间的关系。一些可能的边特征包括：

+   前一个标记的形状（全部大写，以数字开头等）与之前所做的相同。

+   需要正确排列重音和非重音音节的扬抑格识别。这需要音节重音标记器。

+   文本中经常包含一种或多种语言——这被称为代码切换。这在推文中很常见。一个合理的边特征将是周围标记的语言；这将更好地模拟下一个单词可能具有与上一个单词相同的语言。

### 节点特征

节点特征在 CRF 中往往是最活跃的部分，并且可以变得非常丰富。在 [第 5 章](part0061_split_000.html#page "第 5 章. 文本中的跨度查找 – 分块") 的 *使用更好的特征的 CRF 命名实体识别* 食谱中，*文本中的跨度查找 – 分块* 是一个例子。我们将在此食谱中添加之前食谱的标记特征：

```py
public Map<String,? extends Number> nodeFeatures(int n) {
  ObjectToDoubleMap<String> features = new ObjectToDoubleMap<String>();
  features.set("TOK_" + token(n), 1);
  ConditionalClassification tagScores = mBrownTaggingLattice.tokenClassification(n);
  for (int i = 0; i < 3; ++ i) {
    double conditionalProb = tagScores.score(i);
    String tag = tagScores.category(i);
    features.increment(tag, conditionalProb);
  }
  return features;
}
```

然后，就像之前的食谱一样，通过以下方式添加标记特征：

```py
features.set("TOK_" + token(n), 1); 
```

这导致标记字符串被 `TOK_` 预先附加，并计数为 `1`。请注意，虽然 `tag(n)` 在训练中可用，但使用此信息没有意义，因为这正是 CRF 试图预测的内容。

接下来，从 POS 特征标记器中提取前三个标签，并添加相关的条件概率。CRF 将能够有效地处理具有不同权重的标签。

## 还有更多...

在生成新特征时，考虑数据的稀疏性是值得思考的。如果日期对于CRF来说可能是一个重要的特征，那么做标准的计算机科学事情，将日期转换为自1970年1月1日GMT以来的毫秒数可能不是一个好主意。原因如下：

+   底层分类器不知道这两个值几乎相同

+   分类器不知道`MILLI_`前缀是相同的——公共前缀只是为了方便人类

+   该特征在训练中不太可能多次出现，并且可能会被最小特征计数剪枝

与将日期规范化为毫秒数不同，考虑对训练数据中可能存在多个实例的日期进行抽象，例如`has_date`特征，它忽略实际日期但记录日期的存在。如果日期很重要，那么计算所有关于日期的重要信息。如果它是星期几，那么映射到星期几。如果时间顺序很重要，那么映射到更粗略的测量，这种测量可能有很多测量值。一般来说，条件随机场（CRFs）及其底层的逻辑回归分类器对无效特征具有鲁棒性，所以请随意发挥创意——添加特征不太可能降低准确性。
