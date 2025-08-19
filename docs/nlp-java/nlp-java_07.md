# 第四章. 词语和标记的标注

本章我们将涵盖以下方法：

+   有趣短语检测

+   前景或背景驱动的有趣短语检测

+   隐马尔可夫模型（HMM）——词性标注

+   N 最佳词标注

+   基于置信度的标注

+   训练词标注

+   词语标注评估

+   条件随机场（CRF）用于词/标记标注

+   修改 CRF

# 介绍

本章的重点是词语和标记。像命名实体识别这样的常见提取技术，实际上已经编码成了这里呈现的概念，但这需要等到第五章，*在文本中找到跨度 - Chunking*时才能讲解。我们将从简单的寻找有趣的标记集开始，然后转向隐马尔可夫模型（HMM），最后介绍 LingPipe 中最复杂的组件之一——条件随机场（CRF）。和往常一样，我们会向你展示如何评估标注并训练你自己的标注器。

# 有趣短语检测

假设一个程序能够自动从一堆文本数据中找到有趣的部分，其中“有趣”意味着某个词或短语出现的频率高于预期。它有一个非常好的特性——不需要训练数据，而且适用于我们有标记的任何语言。你最常在标签云中看到这种情况，如下图所示：

![有趣短语检测](img/4672OS_04_01.jpg)

上图展示了为[lingpipe.com](http://lingpipe.com)主页生成的标签云。然而，正如 Jeffery Zeldman 在[`www.zeldman.com/daily/0405d.shtml`](http://www.zeldman.com/daily/0405d.shtml)中指出的那样，标签云被认为是“互联网的穆雷发型”，因此如果你在网站上部署这样的功能，可能会站不住脚。

## 如何做到这一点……

要从一个包含迪士尼推文的小数据集中提取有趣短语，请执行以下步骤：

1.  启动命令行并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar:lib/opencsv-2.4.jar com.lingpipe.cookbook.chapter4.InterestingPhrases

    ```

1.  程序应该返回类似如下的结果：

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

1.  你还可以提供一个`.csv`文件，按照我们的标准格式作为参数，以查看不同的数据。

输出往往是令人既期待又无用的。所谓“既期待又无用”是指一些有用的短语出现了，但同时也有许多无趣的短语，这些短语你在总结数据中有趣的部分时根本不想要。在有趣的那一侧，我们能看到`Crayola Color`、`Lindsey Lohan`、`Episode VII`等。在垃圾短语的那一侧，我们看到`ncipes azules`、`pictures releases`等。解决垃圾输出有很多方法——最直接的一步是使用语言识别分类器将非英语的内容过滤掉。

## 它是如何工作的……

在这里，我们将完整地浏览源代码，并通过解释性文字进行拆解：

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

在这里，我们看到路径、导入语句和`main()`方法。我们提供默认文件名或从命令行读取的三元运算符是最后一行：

```py
List<String[]> lines = Util.readCsv(new File(inputCsv));
int ngramSize = 3;
TokenizedLM languageModel = new TokenizedLM(IndoEuropeanTokenizerFactory.INSTANCE, ngramSize);
```

在收集输入数据后，第一个有趣的代码构建了一个标记化的语言模型，这与第一章中使用的字符语言模型有显著不同，*简单分类器*。标记化语言模型操作的是由`TokenizerFactory`创建的标记，而`ngram`参数决定了使用的标记数，而不是字符数。`TokenizedLM`的一个微妙之处在于，它还可以使用字符语言模型来为它之前未见过的标记做出预测。请参见*前景或背景驱动的有趣短语检测*食谱，了解这一过程是如何在实践中运作的；除非在估算时没有未知标记，否则不要使用之前的构造器。此外，相关的 Javadoc 提供了更多的细节。在以下代码片段中，语言模型被训练：

```py
for (String [] line: lines) {
  languageModel.train(line[TEXT_INDEX]);
}
```

接下来的相关步骤是创建搭配词：

```py
int phraseLength = 2;
int minCount = 2;
int maxReturned = 100;
SortedSet<ScoredObject<String[]>> collocations = languageModel.collocationSet(phraseLength, minCount, maxReturned);
```

参数化控制短语的长度（以标记为单位）；它还设置了短语出现的最小次数以及返回多少个短语。我们可以查看长度为 3 的短语，因为我们有一个存储 3-gram 的语言模型。接下来，我们将查看结果：

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

`SortedSet<ScoredObject<String[]>>` 搭配词按得分从高到低排序。得分的直观理解是，当标记的共现次数超过其在训练数据中单独出现的频率时，给予更高的得分。换句话说，短语的得分取决于它们如何偏离基于标记的独立假设。请参阅 Javadoc [`alias-i.com/lingpipe/docs/api/com/aliasi/lm/TokenizedLM.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/lm/TokenizedLM.html) 获取准确的定义——一个有趣的练习是创建你自己的得分系统，并与 LingPipe 中的做法进行比较。

## 还有更多……

鉴于此代码接近可在网站上使用，因此值得讨论调优。调优是查看系统输出并根据系统的错误做出修改的过程。一些我们会立即考虑的修改包括：

+   一个语言 ID 分类器，方便用来过滤非英语文本

+   思考如何更好地标记化数据

+   改变标记长度，以便在摘要中包含 3-gram 和 unigram

+   使用命名实体识别来突出专有名词

# 前景或背景驱动的有趣短语检测

和之前的食谱一样，这个食谱也会找到有趣的短语，但它使用了另一种语言模型来判断什么是有趣的。亚马逊的统计不可能短语（**SIP**）就是这样运作的。你可以通过他们的官网[`www.amazon.com/gp/search-inside/sipshelp.html`](http://www.amazon.com/gp/search-inside/sipshelp.html)清晰了解：

> *“亚马逊的统计学上不太可能出现的短语，或称为“SIPs”，是《搜索内容！™》项目中书籍文本中最具辨识度的短语。为了识别 SIPs，我们的计算机扫描所有《搜索内容！》项目中的书籍文本。如果它们发现某个短语在某本书中相对于所有《搜索内容！》书籍出现的频率很高，那么该短语就是该书中的 SIP。”*
> 
> SIPs 在某本书中不一定是不太可能的，但相对于《搜索内容！》中的所有书籍，它们是不太可能的。

前景模型将是正在处理的书籍，而背景模型将是亚马逊《搜索内容！™》项目中的所有其他书籍。虽然亚马逊可能已经引入了一些不同的调整，但基本理念是相同的。

## 准备工作

有几个数据源值得查看，以便通过两个独立的语言模型得到有趣的短语。关键在于，你希望背景模型作为预期单词/短语分布的来源，帮助突出前景模型中的有趣短语。一些示例包括：

+   **时间分隔的推特数据**：时间分隔的推特数据示例如下：

    +   **背景模型**：这指的是直到昨天关于迪士尼世界的一整年的推文。

    +   **前景模型**：今天的推文。

    +   **有趣的短语**：今天关于迪士尼世界在推特上的新内容。

+   **话题分隔的推特数据**：话题分隔的推特数据示例如下：

    +   **背景模型**：关于迪士尼乐园的推文

    +   **前景模型**：关于迪士尼世界的推文

    +   **有趣的短语**：关于迪士尼世界说的而不是关于迪士尼乐园说的

+   **相似主题的书籍**：关于相似主题的书籍示例如下：

    +   **背景模型**：一堆早期的科幻小说

    +   **前景模型**：儒勒·凡尔纳的*世界大战*

    +   **有趣的短语**：《世界大战》的独特短语和概念

## 如何操作……

这是运行一个前景或背景模型来处理关于迪士尼乐园与迪士尼世界推文的步骤：

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

1.  前景模型包括关于搜索词`disneyland`的推文，背景模型包括关于搜索词`disneyworld`的推文。

1.  排名前列的结果是关于加利福尼亚州迪士尼乐园独特的特征，特别是城堡的名字——睡美人城堡，以及在迪士尼乐园停车场建造的主题公园——加州冒险乐园。

1.  下一个二元组是关于*冬季梦想*，它指的是一部电影的首映。

1.  总体而言，输出效果不错，可以区分这两家度假村的推文。

## 它是如何工作的……

代码位于`src/com/lingpipe/cookbook/chapter4/InterestingPhrasesForegroundBackground.java`。当我们加载前景和背景模型的原始`.csv`数据后，展示内容开始：

```py
TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
tokenizerFactory = new LowerCaseTokenizerFactory(tokenizerFactory);
int minLength = 5;
tokenizerFactory = new LengthFilterTokenizerFactoryPreserveToken(tokenizerFactory, minLength);
```

人们可以理解为什么我们把第二章，*查找和使用单词*，完全用来讨论标记化，但是事实证明，大多数 NLP 系统对于字符流如何被拆分成单词或标记非常敏感。在前面的代码片段中，我们看到三个标记化工厂对字符序列进行有效的破坏。前两个在第二章，*查找和使用单词*中已经得到了充分的介绍，但第三个是一个自定义工厂，需要仔细检查。`LengthFilterTokenizerFactoryPreserveToken`类的目的在于过滤短标记，同时不丢失相邻信息。目标是处理短语"Disney is my favorite resort"，并生成标记（`disney`, `_234`, `_235`, `favorite`, `resort`），因为我们不希望在有趣的短语中出现短单词——它们往往能轻易通过简单的统计模型，并破坏输出。有关第三个标记器的源代码，请参见`src/come/lingpipe/cookbook/chapter4/LengthFilterTokenizerFactoryPreserveToken.java`。此外，请参阅第二章，*查找和使用单词*以了解更多说明。接下来是背景模型：

```py
int nGramOrder = 3;
TokenizedLM backgroundLanguageModel = new TokenizedLM(tokenizerFactory, nGramOrder);
for (String [] line: backgroundData) {
  backgroundLanguageModel.train(line[Util.TEXT_OFFSET]);
}
```

这里构建的是用于判断前景模型中短语新颖性的模型。然后，我们将创建并训练前景模型：

```py
TokenizedLM foregroundLanguageModel = new TokenizedLM(tokenizerFactory,nGramOrder);
for (String [] line: foregroundData) {
  foregroundLanguageModel.train(line[Util.TEXT_OFFSET]);
}
```

接下来，我们将从前景模型中访问`newTermSet()`方法。参数和`phraseSize`决定了标记序列的长度；`minCount`指定要考虑的短语的最小出现次数，`maxReturned`控制返回多少结果：

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

上面的`for`循环按最令人惊讶到最不令人惊讶的短语顺序打印出短语。

这里发生的细节超出了食谱的范围，但 Javadoc 再次引导我们走向启蒙之路。

使用的确切评分是 z-score，如`BinomialDistribution.z(double, int, int)`中定义的那样，其中成功概率由背景模型中的 n-gram 概率估计定义，成功的次数是该模型中 n-gram 的计数，试验次数是该模型中的总计数。

## 还有更多……

这个食谱是我们第一次遇到未知标记的地方，如果处理不当，它们可能具有非常不好的属性。很容易理解为什么这对于基于标记的语言模型的最大似然估计来说是个问题，这是一种通过将每个标记的似然性相乘来估计一些未见标记的语言模型的花哨名称。每个似然性是标记在训练中出现的次数除以数据中出现的标记总数。例如，考虑使用来自*《康涅狄格州的亚瑟王》*的数据进行训练：

> *“这个故事中提到的冷酷的法律和习俗是历史性的，用来说明它们的事件也是历史性的。”*

这非常少的训练数据，但足以证明所提的观点。考虑一下我们如何通过语言模型来估计短语“The ungentle inlaws”。在训练数据中，“The”出现一次，共有 24 个单词；我们将给它分配 1/24 的概率。我们也将给“ungentle”分配 1/24 的概率。如果我们在这里停止，可以说“The ungentle”的概率是 1/24 * 1/24。但是，下一个单词是“inlaws”，它在训练数据中不存在。如果这个词元被赋予 0/24 的值，那么整个字符串的可能性将变为 0（1/24 * 1/24 * 0/20）。这意味着每当有一个未见的词元，且其估计值可能为零时，这通常是一个无用的特性。

解决这个问题的标准方法是替代并近似未在训练中看到的数据的值。解决此问题有几种方法：

+   为未知词元提供一个低但非零的估计。这是一种非常常见的方法。

+   使用字符语言模型与未知词元。这在类中有相关的规定——请参考 Javadoc。

+   还有许多其他方法和大量的研究文献。好的搜索词是“back off”和“smoothing”。

# 隐马尔可夫模型（HMM）——词性

这个配方引入了 LingPipe 的第一个核心语言学功能；它指的是单词的语法类别或**词性**（**POS**）。文本中的动词、名词、形容词等是什么？

## 如何操作...

让我们直接进入，回到那些尴尬的中学英语课堂时光，或者是我们相应的经历：

1.  像往常一样，去你的命令提示符并键入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter9.PosTagger 

    ```

1.  系统将响应一个提示，我们将在其中添加一条豪尔赫·路易斯·博尔赫斯的引用：

    ```py
    INPUT> Reality is not always probable, or likely.

    ```

1.  系统将愉快地响应这个引用：

    ```py
    Reality_nn is_bez not_* always_rb probable_jj ,_, or_cc likely_jj ._. 

    ```

每个词元后附加有一个`_`和一个词性标签；`nn`是名词，`rb`是副词，等等。完整的标签集和标注器语料库的描述可以在[`en.wikipedia.org/wiki/Brown_Corpus`](http://en.wikipedia.org/wiki/Brown_Corpus)找到。多玩玩这个。词性标注器是 90 年代 NLP 领域最早的突破性机器学习应用之一。你可以期待它的表现精度超过 90%，尽管它在 Twitter 数据上可能会有点问题，因为底层语料库是 1961 年收集的。

## 它是如何工作的...

适合食谱书的方式是，我们并未透露如何构建词性标注器的基础知识。可以通过 Javadoc、Web 以及研究文献来帮助你理解底层技术——在训练 HMM 的配方中，简要讨论了底层 HMM。这是关于如何使用呈现的 API：

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

代码首先设置 `TokenizerFactory`，这很有意义，因为我们需要知道哪些词将会得到词性标注。接下来的一行读取了一个之前训练过的词性标注器，作为 `HiddenMarkovModel`。我们不会深入讨论细节；你只需要知道 HMM 将词标记 *n* 的词性标记视为先前标注的函数。

由于这些标签在数据中并不是直接观察到的，这使得马尔可夫模型成为隐含的。通常，回看一两个标记。隐马尔可夫模型（HMM）中有许多值得理解的内容。

下一行的 `HmmDecoder decoder` 将 HMM 包装到代码中，用于标注提供的标记。接下来的标准交互式 `while` 循环将进入 `firstBest(tokenList, decoder)` 方法，并且所有有趣的内容都发生在方法的结尾。该方法如下：

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

请注意 `decoder.tag(tokenList)` 调用，它会产生一个 `Tagging<String>` 标注。Tagging 没有迭代器或有用的标签/标记对封装，因此需要通过递增索引 i 来访问信息。

# N-best 单词标注

计算机科学的确定性驱动特性并未体现在语言学的变数上，合理的博士们至少可以同意或不同意，直到乔姆斯基的亲信出现为止。本配方使用了在前一配方中训练的相同 HMM，但为每个单词提供了可能标签的排名列表。

这在什么情况下可能有帮助？想象一个搜索引擎，它不仅搜索单词，还搜索标签——不一定是词性。这个搜索引擎可以索引单词以及最优的 *n* 个标签，这些标签可以让匹配的标签进入非首选标签。这可以帮助提高召回率。

## 如何操作...

`N`-best 分析推动了 NLP 开发者的技术边界。曾经是单一的，现在是一个排名列表，但它是性能提升的下一阶段。让我们开始执行以下步骤：

1.  把你那本《句法结构》放好，翻过来并键入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.NbestPosTagger 

    ```

1.  然后，输入以下内容：

    ```py
    INPUT> Colorless green ideas sleep furiously.

    ```

1.  它将输出以下内容：

    ```py
    N BEST
    #   JointLogProb         Analysis
    0     -91.141  Colorless_jj   green_jj   ideas_nns  sleep_vb   furiously_rb   ._. 
    1     -93.916  Colorless_jj   green_nn   ideas_nns  sleep_vb   furiously_rb   ._. 
    2     -95.494  Colorless_jj   green_jj   ideas_nns  sleep_rb   furiously_rb   ._. 
    3     -96.266  Colorless_jj   green_jj   ideas_nns  sleep_nn   furiously_rb   ._. 
    4     -98.268  Colorless_jj   green_nn   ideas_nns  sleep_rb   furiously_rb   ._.

    ```

输出列表按从最可能到最不可能的顺序列出整个标记序列的估计，基于 HMM 的估计。记住，联合概率是以对数 2 为基数的。为了比较联合概率，将 -93.9 从 -91.1 中减去，差值为 2.8。因此，标注器认为选项 1 的出现几率是选项 0 的 2 ^ 2.8 = 7 倍小。这个差异的来源在于将名词标记为绿色，而不是形容词。

## 它是如何工作的……

加载模型和命令输入输出的代码与之前的配方相同。不同之处在于获取和显示标注所使用的方法：

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

除了在标注迭代过程中解决格式化问题外，没有太多复杂的内容。

# 基于置信度的标注

另一个视图展示了标注概率，这反映了在单词级别的概率分配。代码反映了底层的`TagLattice`，并提供了对标注器是否有信心的洞察。

## 如何实现...

本食谱将把概率估计集中在单个标记上。请执行以下步骤：

1.  在命令行或 IDE 中键入以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.ConfidenceBasedTagger

    ```

1.  然后，输入以下内容：

    ```py
    INPUT> Colorless green ideas sleep furiously.

    ```

1.  它会生成以下输出：

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

这种数据视图分配了标签和词的联合概率。我们可以看到`green`有`.208`的概率应该被标记为`nn`（名词单数），但正确的分析仍然是`.788`，标记为形容词`jj`。

## 它是如何工作的…

我们仍然使用的是*隐藏马尔可夫模型（HMM）——词性*食谱中的旧 HMM，但使用了不同的部分。读取模型的代码完全相同，主要的区别在于我们报告结果的方式。`src/com/lingpipe/cookbook/chapter4/ConfidenceBasedTagger.java`中的方法：

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

该方法明确演示了标记的底层格子到概率的映射，这就是 HMM 的核心。更改`for`循环的终止条件，以查看更多或更少的标签。

# 训练词性标注

当你可以创建自己的模型时，词性标注变得更加有趣。注释词性标注语料库的领域对于一本简单的食谱书来说有些过于复杂——词性数据的注释非常困难，因为它需要相当多的语言学知识才能做得好。本食谱将直接解决基于 HMM 的句子检测器的机器学习部分。

由于这是一本食谱书，我们将简单解释一下什么是 HMM。我们一直在使用的标记语言模型会根据当前估计词汇前面的一些词/标记来进行前文上下文计算。HMM 在计算当前标记的标签估计时，会考虑前面标签的一些长度。这使得看似不同的邻接词，如`of`和`in`，变得相似，因为它们都是介词。

在*句子检测*食谱中，来自第五章，*文本中的跨度 – 分块*，基于`HeuristicSentenceModel`的句子检测器虽然有用，但灵活性不强。与其修改/扩展`HeuristicSentenceModel`，我们将基于我们注释的数据构建一个基于机器学习的句子系统。

## 如何实现...

这里的步骤描述了如何运行`src/com/lingpipe/cookbook/chapter4/HMMTrainer.java`中的程序：

1.  可以创建一个新的句子注释数据集，或使用以下默认数据，该数据位于`data/connecticut_yankee_EOS.txt`。如果你自己处理数据，只需编辑一些文本，并用`[`和`]`标记句子边界。我们的示例如下：

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

1.  打开命令提示符并运行以下命令启动程序：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter4.HmmTrainer

    ```

1.  它将输出如下内容：

    ```py
    Training The/BOS ungentle/WORD laws/WORD and/WORD customs/WORD touched/WORD…
    done training, token count: 123
    Enter text followed by new line
    > The cat in the hat. The dog in a bog.
    The/BOS cat/WORD in/WORD the/WORD hat/WORD ./EOS The/BOS dog/WORD in/WORD a/WORD bog/WORD ./EOS

    ```

1.  输出是一个标记化文本，包含三种标签之一：`BOS`表示句子的开始，`EOS`表示句子的结束，`WORD`表示所有其他的标记。

## 它是如何工作的…

与许多基于跨度的标记一样，`span`注解被转换为标记级别的注解，如配方输出中所示。因此，首先的任务是收集注解文本，设置`TokenizerFactory`，然后调用一个解析子程序将其添加到`List<Tagging<String>>`中：

```py
public static void main(String[] args) throws IOException {
  String inputFile = args.length > 0 ? args[0] : "data/connecticut_yankee_EOS.txt";
  char[] text = Files.readCharsFromFile(new File(inputFile), Strings.UTF8);
  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  List<Tagging<String>> taggingList = new ArrayList<Tagging<String>>();
  addTagging(tokenizerFactory,taggingList,text);
```

解析前述格式的子程序首先通过`IndoEuropeanTokenizer`对文本进行标记化，这个标记化器的优点是将`[`和`]`作为独立的标记处理。它不检查句子分隔符是否格式正确——一个更健壮的解决方案将需要做这件事。难点在于，我们希望在生成的标记流中忽略这些标记，但又希望使用它来使得`[`后面的标记为 BOS，而`]`前面的标记为 EOS。其他标记只是`WORD`。该子程序构建了一个并行的`Lists<String>`实例来存储标记和标记词，然后用它创建`Tagging<String>`并将其添加到`taggingList`中。第二章中的标记化配方，*查找和处理单词*，涵盖了标记化器的工作原理。请看下面的代码片段：

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

前面的代码有一个微妙之处。训练数据被视为单一的标记——这将模拟当我们在新数据上使用句子检测器时，输入的样子。如果训练中使用了多个文档/章节/段落，那么我们将针对每一块文本调用这个子程序。

返回到`main()`方法，我们将设置`ListCorpus`并逐一将标记添加到语料库的训练部分。也有`addTest()`方法，但本例不涉及评估；如果涉及评估，我们很可能会使用`XValidatingCorpus`：

```py
ListCorpus<Tagging<String>> corpus = new ListCorpus<Tagging<String>> ();
for (Tagging<String> tagging : taggingList) {
  System.out.println("Training " + tagging);
  corpus.addTrain(tagging);
}
```

接下来，我们将创建`HmmCharLmEstimator`，这就是我们的 HMM。请注意，有一些构造函数允许定制参数来影响性能——请参见 Javadoc。接下来，估算器将针对语料库进行训练，创建`HmmDecoder`，它将实际标记标记，如下面的代码片段所示：

```py
HmmCharLmEstimator estimator = new HmmCharLmEstimator();
corpus.visitTrain(estimator);
System.out.println("done training, token count: " + estimator.numTrainingTokens());
HmmDecoder decoder = new HmmDecoder(estimator);
```

在下面的代码片段中，我们的标准 I/O 循环会被调用以获取一些用户反馈。一旦我们从用户那获得一些文本，它将通过我们用于训练的相同标记器进行标记化，并且解码器将展示生成的标记。

注意，训练分词器不必与生产分词器相同，但必须小心不要以完全不同的方式进行分词；否则，HMM 将无法看到它训练时使用的标记。接着会使用回退模型，这可能会降低性能。看一下以下的代码片段：

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

就是这样！为了真正将其封装成一个合适的句子检测器，我们需要将其映射回原始文本中的字符偏移量，但这部分在第五章，*在文本中查找跨度——分块*中有讲解。这足以展示如何使用 HMM。一个更完备的方法将确保每个 BOS 都有一个匹配的 EOS，反之亦然。而 HMM 并没有这样的要求。

## 还有更多……

我们有一个小型且易于使用的词性标注语料库；这使我们能够展示如何将 HMM 的训练应用于一个完全不同的问题，并得出相同的结果。这就像我们的*如何分类情感——简单版*的食谱，在第一章，*简单分类器*；语言识别和情感分类之间唯一的区别是训练数据。为了简单起见，我们将从一个硬编码的语料库开始——它位于`src/com/lingpipe/cookbook/chapter4/TinyPosCorus.java`：

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

语料库手动创建了标记和静态`WORDS_TAGS`中标记的每个词的标签，并为每个句子创建了`Tagging<String>`；在这种情况下，`Tagging<String>`由两个对齐的`List<String>`实例组成。然后，这些标注被发送到`Corpus`超类的`handle()`方法。替换这个语料库看起来像这样：

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

我们仅仅注释掉了加载带有句子检测和特征的语料库的代码，并将`TinyPosCorpus`替换进去。它不需要添加数据，所以我们只需使用它来训练 HMM。为了避免混淆，我们创建了一个单独的类`HmmTrainerPos.java`。运行它将得到以下结果：

```py
java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar 
done training, token count: 42
Enter text followed by new line
> The cat in the hat is back.
The/DET cat/N in/TV the/DET hat/N is/TV back/PN ./EOS

```

唯一的错误是`in`是一个及物动词`TV`。训练数据非常小，因此错误是可以预期的。就像第一章，*简单分类器*中语言识别和情感分类的区别一样，通过仅仅改变训练数据，HMM 用来学习一个非常不同的现象。

# 词标注评估

词标注评估推动了下游技术的发展，比如命名实体识别，而这些技术又推动了如共指消解等高端应用。你会注意到，大部分评估与我们分类器的评估相似，唯一的不同是每个标签都像自己的分类器类别一样被评估。

这个食谱应能帮助你开始进行评估，但请注意，我们网站上有一个关于标注评估的非常好的教程，地址是[`alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html)；这个食谱更详细地介绍了如何最佳地理解标注器的表现。

这个食谱简短且易于使用，因此你没有理由不去评估你的标注器。

## 准备工作

以下是我们评估器的类源代码，位于`src/com/lingpipe/cookbook/chapter4/TagEvaluator.java`：

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

## 如何操作…

我们将指出前面代码中的有趣部分：

1.  首先，我们将设置`TaggerEvaluator`，其包含一个空的`HmmDecoder`和一个控制是否存储标记的`boolean`。`HmmDecoder`对象将在后续代码的交叉验证代码中设置：

    ```py
    HmmDecoder decoder = null;
    boolean storeTokens = true;
    TaggerEvaluator<String> evaluator = new TaggerEvaluator<String>(decoder,storeTokens);
    ```

1.  接下来，我们将加载前一个食谱中的`TinyPosCorpus`并使用它填充`XValididatingObjectCorpus`——这是一种非常巧妙的技巧，允许在语料库类型之间轻松转换。注意，我们选择了 10 折——语料库只有 11 个训练示例，因此我们希望最大化每个折叠中的训练数据量。如果你是这个概念的新手，请查看第一章，*简单分类器*中的*如何进行训练和交叉验证评估*食谱。请查看以下代码片段：

    ```py
    Corpus<ObjectHandler<Tagging<String>>> smallCorpus = new TinyPosCorpus();
    int numFolds = 10;
    XValidatingObjectCorpus<Tagging<String>> xValCorpus = new XValidatingObjectCorpus<Tagging<String>>(numFolds);
    smallCorpus.visitCorpus(xValCorpus);
    ```

1.  以下代码片段是一个`for()`循环，它迭代折叠的数量。循环的前半部分处理训练：

    ```py
    for (int i = 0; i < numFolds; ++i) {
      xValCorpus.setFold(i);
      HmmCharLmEstimator estimator = new HmmCharLmEstimator();
      xValCorpus.visitTrain(estimator);
      System.out.println("done training " + estimator.numTrainingTokens());
    ```

1.  循环的其余部分首先为 HMM 创建解码器，将评估器设置为使用该解码器，然后将适当配置的评估器应用于语料库的测试部分：

    ```py
    decoder = new HmmDecoder(estimator);
    evaluator.setTagger(decoder);
    xValCorpus.visitTest(evaluator);
    ```

1.  最后的几行代码应用于所有折叠的语料库已用于训练和测试后。注意，评估器是`BaseClassifierEvaluator`！它将每个标签作为一个类别报告：

    ```py
    BaseClassifierEvaluator<String> classifierEval = evaluator.tokenEval();
    System.out.println(classifierEval);
    ```

1.  为评估的洪流做好准备。以下是其中的一小部分，即你应该从第一章，*简单分类器*中熟悉的混淆矩阵：

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

就这样。你有了一个与第一章，*简单分类器*中的分类器评估密切相关的评估设置。

## 还有更多…

针对 n 最佳词标注，存在评估类，即`NBestTaggerEvaluator`和`MarginalTaggerEvaluator`，用于信心排名。同样，可以查看更详细的词性标注教程，里面有关于评估指标的详细介绍，以及一些示例软件来帮助调整 HMM。

# 条件随机场（CRF）用于词/标记标注

**条件随机场**（**CRF**）是第三章的*逻辑回归*配方的扩展，应用于词标注。在第一章的*简单分类器*中，我们讨论了将问题编码为分类问题的各种方式。CRF 将序列标注问题视为找到最佳类别，其中每个类别（C）是 C*T 标签（T）分配到词元的其中之一。

例如，如果我们有词组`The`和`rain`，并且标签`d`表示限定词，`n`表示名词，那么 CRF 分类器的类别集如下：

+   **类别 1**：`d d`

+   **类别 2**：`n d`

+   **类别 3**：`n n`

+   **类别 4**：`d d`

为了使这个组合计算的噩梦变得可计算，采用了各种优化方法，但这是大致的思路。疯狂，但它有效。

此外，CRF 允许像逻辑回归对分类所做的那样，在训练中使用随机特征。此外，它具有针对上下文优化的 HMM 样式观察的数据结构。它在词性标注中的使用并不令人兴奋，因为我们当前的 HMM 已经接近最先进的技术。CRF 真正有所作为的地方是像命名实体识别这样的使用案例，这些内容在第五章的*在文本中寻找跨度 - 分块*中有所涵盖，但我们希望在通过分块接口使演示更加复杂之前，先讨论纯 CRF 实现。

[`alias-i.com/lingpipe/demos/tutorial/crf/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/crf/read-me.html)上有一篇关于 CRF 的详细优秀教程；这个配方与该教程非常接近。你将在那里找到更多的信息和适当的参考文献。

## 如何做到……

我们到目前为止所展示的所有技术都是在上一个千年发明的；这是一项来自新千年的技术。请按照以下步骤进行操作：

1.  在命令行中输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.CRFTagger

    ```

1.  控制台继续显示收敛结果，这些结果应该与你在第三章的*逻辑回归*配方中见过的非常相似，我们将得到标准的命令行提示符：

    ```py
    Enter text followed by new line
    >The rain in Spain falls mainly on the plain.

    ```

1.  对此，我们将得到一些相当混乱的输出：

    ```py
    The/DET rain/N in/TV Spain/PN falls/IV mainly/EOS on/DET the/N plain/IV ./EOS

    ```

1.  这是一个糟糕的输出，但 CRF 已经在 11 个句子上进行了训练。所以，我们不要过于苛刻——特别是考虑到这项技术在词标注和跨度标注方面表现得尤为出色，只要提供足够的训练数据来完成它的工作。

## 它是如何工作的……

与逻辑回归类似，我们需要执行许多与配置相关的任务，以使这个类能够正常运行。本食谱将处理代码中的 CRF 特定方面，并参考第三章中的*逻辑回归*食谱，了解与配置相关的逻辑回归部分。

从`main()`方法的顶部开始，我们将获取我们的语料库，这部分在前面三个食谱中有讨论：

```py
Corpus<ObjectHandler<Tagging<String>>> corpus = new TinyPosCorpus();
```

接下来是特征提取器，它是 CRF 训练器的实际输入。它之所以是最终的，仅仅是因为一个匿名内部类将访问它，以展示在下一个食谱中如何进行特征提取：

```py
final ChainCrfFeatureExtractor<String> featureExtractor
  = new SimpleCrfFeatureExtractor();
```

我们将在本食谱后面讨论这个类的工作原理。

接下来的配置块是针对底层逻辑回归算法的。有关更多信息，请参考第三章中的*逻辑回归*食谱，看看以下代码片段：

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

接下来，使用以下内容来训练 CRF：

```py
System.out.println("\nEstimating");
ChainCrf<String> crf = ChainCrf.estimate(corpus,featureExtractor,addIntercept,minFeatureCount,cacheFeatures,allowUnseenTransitions,prior,priorBlockSize,annealingSchedule,minImprovement,minEpochs,maxEpochs,reporter);
```

其余的代码只是使用标准的 I/O 循环。有关`tokenizerFactory`如何工作的内容，请参考第二章，*查找和使用单词*：

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

现在，我们将进入特征提取器部分。提供的实现 closely mimics 标准 HMM 的特性。`com/lingpipe/cookbook/chapter4/SimpleCrfFeatureExtractor.java` 类以如下内容开始：

```py
public class SimpleCrfFeatureExtractor implements ChainCrfFeatureExtractor<String> {
  public ChainCrfFeatures<String> extract(List<String> tokens, List<String> tags) {
    return new SimpleChainCrfFeatures(tokens,tags);
  }
```

`ChainCrfFeatureExtractor`接口要求一个`extract()`方法，该方法接收令牌和相关的标签，并将它们转换为`ChainCrfFeatures<String>`，在此案例中是这样的。这个过程由下面的一个内部类`SimpleChainCrfFeatures`处理；该内部类继承自`ChainCrfFeatures`，并提供了抽象方法`nodeFeatures()`和`edgeFeatures()`的实现：

```py
static class SimpleChainCrfFeatures extends ChainCrfFeatures<String> {
```

以下构造函数访问将令牌和标签传递给超类，超类将进行账务处理，以支持查找`tags`和`tokens`：

```py
public SimpleChainCrfFeatures(List<String> tokens, List<String> tags) {
  super(tokens,tags);
}
```

节点特征计算如下：

```py
public Map<String,Double> nodeFeatures(int n) {
  ObjectToDoubleMap<String> features = new ObjectToDoubleMap<String>();
  features.increment("TOK_" + token(n),1.0);
  return features;
}
```

令牌根据它们在句子中的位置进行索引。位置为`n`的单词/令牌的节点特征是通过`ChainCrfFeatures`的基类方法`token(n)`返回的`String`值，前缀为`TOK_`。这里的值是`1.0`。特征值可以有用地调整为 1.0 以外的其他值，这对于更复杂的 CRF 方法非常有用，比如使用其他分类器的置信度估计。看看下面的食谱，以了解如何实现这一点。

与 HMM 类似，有些特征依赖于输入中的其他位置——这些被称为**边缘特征**。边缘特征接受两个参数：一个是生成特征的位置`n`和`k`，它将适用于句子中的所有其他位置：

```py
public Map<String,Double> edgeFeatures(int n, int k) {
  ObjectToDoubleMap<String> features = new ObjectToDoubleMap<String>();
  features.increment("TAG_" + tag(k),1.0);
  return features;
}
```

下一篇食谱将处理如何修改特征提取。

## 还有更多内容……

Javadoc 中引用了大量研究文献，LingPipe 网站上也有一个更加详细的教程。

# 修改 CRF

CRF 的强大和吸引力来源于丰富的特征提取——通过提供反馈的评估工具来进行你的探索。本示例将详细介绍如何创建更复杂的特征。

## 如何操作……

我们不会训练和运行 CRF；相反，我们将打印出特征。将此特征提取器替换为之前示例中的特征提取器，以查看它们的工作效果。执行以下步骤：

1.  打开命令行并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter4.ModifiedCrfFeatureExtractor

    ```

1.  特征提取器类会为训练数据中的每个标记输出真实标签，这些标签用于学习：

    ```py
    -------------------
    Tagging:  John/PN

    ```

1.  这反映了`John`标记的训练标签，它是由`src/com/lingpipe/cookbook/chapter4/TinyPosCorpus.java`文件中确定的。

1.  节点特征遵循我们 Brown 语料库 HMM 标注器的前三个 POS 标签以及`TOK_John`特征：

    ```py
    Node Feats:{nps=2.0251355582754984E-4, np=0.9994337160349874, nn=2.994165140854113E-4, TOK_John=1.0}
    ```

1.  接下来，显示句子“John ran”中其他标记的边特征：

    ```py
    Edge Feats:{TOKEN_SHAPE_LET-CAP=1.0, TAG_PN=1.0}
    Edge Feats:{TAG_IV=1.0, TOKEN_SHAPE_LET-CAP=1.0}
    Edge Feats:{TOKEN_SHAPE_LET-CAP=1.0, TAG_EOS=1.0}
    ```

1.  剩余的输出为句子中其余标记的特征，然后是`TinyPosCorpus`中剩余的句子。

## 它的工作原理是……

我们的特征提取代码位于`src/com/lingpipe/cookbook/chapter4/ModifiedCrfFeatureExtractor.java`。我们将从加载语料库、通过特征提取器处理内容并打印出来的`main()`方法开始：

```py
public static void main(String[] args) throws IOException, ClassNotFoundException {

  Corpus <ObjectHandler<Tagging<String>>> corpus = new TinyPosCorpus();
  final ChainCrfFeatureExtractor<String> featureExtractor = new ModifiedCrfFeatureExtractor();
```

我们将使用之前示例中的`TinyPosCorpus`作为我们的语料库，然后从包含类创建特征提取器。引用变量在后面的匿名内部类中需要使用`final`修饰符。

对于匿名内部类表示歉意，但这是访问语料库中存储内容的最简单方式，原因多种多样，例如复制和打印。在这种情况下，我们只是生成并打印训练数据中找到的特征：

```py
corpus.visitCorpus(new ObjectHandler<Tagging<String>>() {
  @Override
  public void handle(Tagging<String> tagging) {
    ChainCrfFeatures<String> features = featureExtractor.extract(tagging.tokens(), tagging.tags());
```

语料库包含`Tagging`对象，而它们又包含一个`List<String>`的标记和标签。然后，使用这些信息通过应用`featureExtractor.extract()`方法到标记和标签，创建一个`ChainCrfFeatures<String>`对象。这将涉及大量计算，如将展示的那样。

接下来，我们将对训练数据进行报告，包含标记和预期标签：

```py
for (int i = 0; i < tagging.size(); ++i) {
  System.out.println("---------");
  System.out.println("Tagging:  " + tagging.token(i) + "/" + tagging.tag(i));
```

接下来，我们将继续展示将用于通知 CRF 模型，以尝试为节点生成前置标签的特征：

```py
System.out.println("Node Feats:" + features.nodeFeatures(i));
```

然后，通过以下对源节点`i`相对位置的迭代来生成边特征：

```py
for (int j = 0; j < tagging.size(); ++j) {
  System.out.println("Edge Feats:" 
        + features.edgeFeatures(i, j));
}
```

现在我们打印出特征。接下来，我们将介绍如何构建特征提取器。假设你已经熟悉之前的示例。首先，构造函数引入了 Brown 语料库 POS 标注器：

```py
HmmDecoder mDecoder;

public ModifiedCrfFeatureExtractor() throws IOException, ClassNotFoundException {
  File hmmFile = new File("models/pos-en-general-" + "brown.HiddenMarkovModel");
  HiddenMarkovModel hmm = (HiddenMarkovModel)AbstractExternalizable.readObject(hmmFile);
  mDecoder = new HmmDecoder(hmm);
}
```

该构造函数引入了一些外部资源用于特征生成，即一个基于布朗语料库训练的 POS 标注器。为什么要为 POS 标注器引入另一个 POS 标注器呢？我们将布朗 POS 标注器的角色称为“特征标注器”，以将其与我们正在构建的标注器区分开来。使用特征标注器的原因有几个：

+   我们使用的是一个非常小的语料库进行训练，一个更强大的通用 POS 特征标注器将帮助改善结果。`TinyPosCorpus`语料库甚至太小，无法带来这样的好处，但如果有更多的数据，`at`这个特征统一了`the`、`a`和`some`，这将帮助 CRF 识别出`some dog`应该是`'DET'` `'N'`，即便在训练中它从未见过`some`。

+   我们不得不与那些与 POS 特征标注器不一致的标签集一起工作。CRF 可以使用这些外部标签集中的观察结果来更好地推理期望的标注。最简单的情况是，来自布朗语料库标签集中的`at`可以干净地映射到当前标签集中的`DET`。

+   可以通过运行多个标注器来提高性能，这些标注器可以基于不同的数据进行训练，或使用不同的技术进行标注。然后，CRF 可以在希望的情况下识别出一个标注器优于其他标注器的上下文，并利用这些信息来引导分析。在过去，我们的 MUC-6 系统使用了 3 个 POS 标注器，它们投票选出最佳输出。让 CRF 来解决这个问题会是一种更优的方法。

特征提取的核心通过`extract`方法访问：

```py
public ChainCrfFeatures<String> extract(List<String> tokens, List<String> tags) {
  return new ModChainCrfFeatures(tokens,tags);
}
```

`ModChainCrfFeatures`作为一个内部类创建，旨在将类的数量保持在最低限度，且外部类非常轻量：

```py
class ModChainCrfFeatures extends ChainCrfFeatures<String> {

  TagLattice<String> mBrownTaggingLattice;

  public ModChainCrfFeatures(List<String> tokens, List<String> tags) {
    super(tokens,tags);
    mBrownTaggingLattice = mDecoder.tagMarginal(tokens);
  }
```

上述构造函数将令牌和标签交给父类，父类负责处理这些数据的记账工作。然后，“特征标注器”应用于令牌，结果输出被分配给成员变量`mBrownTaggingLattice`。代码将一次访问一个令牌的标注，因此现在必须计算这些标注。

特征创建步骤通过两个方法进行：`nodeFeatures`和`edgeFeatures`。我们将从对前一个配方中`edgeFeatures`的简单增强开始：

```py
public Map<String,? extends Number> edgeFeatures(int n, int k) {
  ObjectToDoubleMap<String> features = newObjectToDoubleMap<String>();
  features.set("TAG_" + tag(k), 1.0d);
  String category = IndoEuropeanTokenCategorizer.CATEGORIZER.categorize(token(n));
  features.set("TOKEN_SHAPE_" + category,1.0d);
  return features;
}
```

代码添加了一个令牌形态特征，将`12`和`34`泛化为`2-DIG`以及其他许多泛化。对于 CRF 而言，除非特征提取另有说明，否则`12`和`34`作为两位数之间的相似性是不存在的。请参阅 Javadoc 获取完整的分类器输出。

### 候选边缘特征

CRF 允许应用随机特征，因此问题是哪些特征是有意义的。边缘特征与节点特征一起使用，因此另一个问题是特征应该应用于边缘还是节点。边缘特征将用于推理当前词/令牌与周围词语的关系。一些可能的边缘特征包括：

+   前一个令牌的形态（全大写、以数字开头等），如前所述。

+   需要正确排序重音和非重音音节的抑扬格五音步识别。这还需要一个音节重音分词器。

+   文本中经常包含一种或多种语言——这叫做代码切换。这在推文中是常见的现象。一个合理的边缘特征将是周围令牌的语言；这种语言可以更好地建模下一词可能与前一词属于同一语言。

### 节点特征

节点特征通常是 CRF 中动作的关键所在，并且它们可以变得非常丰富。在第五章中的*使用 CRF 和更好的特征进行命名实体识别*方法，*Finding Spans in Text – Chunking*，就是一个例子。在这个方法中，我们将为前一个方法的令牌特征添加词性标注：

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

然后，像在前一个方法中一样，通过以下方式添加令牌特征：

```py
features.set("TOK_" + token(n), 1); 
```

这导致令牌字符串前面加上`TOK_`和计数`1`。请注意，虽然`tag(n)`在训练中可用，但使用该信息没有意义，因为 CRF 的目标就是预测这些标签。

接下来，从词性特征标注器中提取出前三个标签，并与相关的条件概率一起添加。CRF 将能够通过这些变化的权重进行有效的工作。

## 还有更多…

在生成新特征时，值得考虑数据的稀疏性。如果日期可能是 CRF 的重要特征，可能不适合做计算机科学中的标准操作——将日期转换为自 1970 年 1 月 1 日格林威治标准时间以来的毫秒数。原因是`MILLI_1000000000`特征将被视为与`MILLI_1000000001`完全不同。原因有几个：

+   底层分类器并不知道这两个值几乎相同。

+   分类器并不知道`MILLI_`前缀是相同的——这个通用前缀仅仅是为了方便人类。

+   该特征在训练中不太可能出现多次，可能会被最小特征计数修剪掉。

而不是将日期标准化为毫秒，考虑使用一个抽象层来表示日期，这个日期在训练数据中可能有很多实例，例如忽略实际日期但记录日期存在性的`has_date`特征。如果日期很重要，那么计算关于日期的所有重要信息。如果它是星期几，那么映射到星期几。如果时间顺序很重要，那么映射到更粗略的度量，这些度量可能有许多测量值。一般来说，CRF 和底层的逻辑回归分类器对于无效特征具有鲁棒性，因此可以大胆尝试创新——添加特征不太可能使准确度更差。
