# 第五章：在文本中查找跨度—分块

本章涵盖以下内容：

+   句子检测

+   句子检测的评估

+   调整句子检测

+   在字符串中标记嵌套的分块—句子分块示例

+   段落检测

+   简单的名词短语和动词短语

+   基于正则表达式的命名实体识别（NER）分块

+   基于词典的 NER 分块

+   在单词标注和分块之间转换—BIO 编解码器

+   基于隐马尔可夫模型（HMM）的 NER

+   混合 NER 数据源

+   用于分块的条件随机场（CRFs）

+   使用更好的特征的条件随机场（CRFs）进行 NER

# 介绍

本章将告诉我们如何处理通常涵盖一个或多个单词/标记的文本跨度。LingPipe API 将这种文本单元表示为分块，并使用相应的分块器生成分块。以下是一些带有字符偏移的文本：

```py
LingPipe is an API. It is written in Java.
012345678901234567890123456789012345678901
          1         2         3         4           
```

将前面的文本分块成句子将会得到如下输出：

```py
Sentence start=0, end=18
Sentence start =20, end=41
```

为命名实体添加分块，增加了 LingPipe 和 Java 的实体：

```py
Organization start=0, end=7
Organization start=37, end=40
```

我们可以根据命名实体的偏移量来定义命名实体分块；这对 LingPipe 没有影响，但对 Java 而言会有所不同：

```py
Organization start=17, end=20
```

这是分块的基本思路。有很多方法可以构建它们。

# 句子检测

书面文本中的句子大致对应于口头表达。它们是工业应用中处理单词的标准单元。在几乎所有成熟的 NLP 应用程序中，即使是推文（可能在限定的 140 字符内有多个句子），句子检测也是处理管道的一部分。

## 如何做到这一点...

1.  和往常一样，我们首先将玩一些数据。请在控制台输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.SentenceDetection

    ```

1.  程序将为您的句子检测实验提供提示。按下回车/换行键结束待分析的文本：

    ```py
    Enter text followed by new line
    >A sentence. Another sentence.
    SENTENCE 1:
    A sentence.
    SENTENCE 2:
    Another sentence.

    ```

1.  值得尝试不同的输入。以下是一些示例，用于探索句子检测器的特性。去掉句子开头的首字母大写；这样就能防止检测到第二个句子：

    ```py
    >A sentence. another sentence.
    SENTENCE 1:
    A sentence. another sentence.

    ```

1.  检测器不需要结束句号—这是可配置的：

    ```py
    >A sentence. Another sentence without a final period
    SENTENCE 1:A sentence.
    SENTENCE 2:Another sentence without a final period

    ```

1.  检测器平衡括号，这样就不会让句子在括号内断开—这也是可配置的：

    ```py
    >(A sentence. Another sentence.)
    SENTENCE 1: (A sentence. Another sentence.)

    ```

## 它是如何工作的...

这个句子检测器是基于启发式或规则的句子检测器。统计句子检测器也是一个合理的方案。我们将遍历整个源代码来运行检测器，稍后我们会讨论修改：

```py
package com.lingpipe.cookbook.chapter5;

import com.aliasi.chunk.Chunk;
import com.aliasi.chunk.Chunker;
import com.aliasi.chunk.Chunking;
import com.aliasi.sentences.IndoEuropeanSentenceModel;
import com.aliasi.sentences.SentenceChunker;
import com.aliasi.sentences.SentenceModel;
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.tokenizer.TokenizerFactory;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Set;

public class SentenceDetection {

public static void main(String[] args) throws IOException {
  boolean endSent = true;
  boolean parenS = true;
  SentenceModel sentenceModel = new IndoEuropeanSentenceModel(endSent,parenS);
```

从`main`类的顶部开始，布尔类型的`endSent`参数控制是否假定被检测的句子字符串以句子结尾，无论如何—这意味着最后一个字符始终是句子边界—它不一定是句号或其他典型的句子结束符号。改变它，试试没有结束句号的句子，结果将是没有检测到句子。

接下来的布尔值`parenS`声明在寻找句子时优先考虑括号，而不是句子标记符。接下来，实际的句子分块器将被设置：

```py
TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
Chunker sentenceChunker = new SentenceChunker(tokFactory,sentenceModel);
```

`tokFactory`应该对你来说并不陌生，来自第二章，*查找和处理单词*。然后可以构建`sentenceChunker`。以下是标准的命令行交互输入/输出代码：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.print("Enter text followed by new line\n>");
  String text = reader.readLine();
```

一旦我们得到了文本，句子检测器就会被应用：

```py
Chunking chunking = sentenceChunker.chunk(text);
Set<Chunk> sentences = chunking.chunkSet();
```

这个分块操作提供了一个`Set<Chunk>`参数，它非正式地提供了`Chunks`的适当排序；它们将根据`ChunkingImpl`的 Javadoc 进行添加。真正偏执的程序员可能会强制执行正确的排序，我们将在本章后面讨论如何处理重叠的分块。

接下来，我们将检查是否找到了任何句子，如果没有找到，我们将向控制台报告：

```py
if (sentences.size() < 1) {
  System.out.println("No sentence chunks found.");
  return;
}
```

以下是书中首次介绍`Chunker`接口，并且有一些评论需要说明。`Chunker`接口生成`Chunk`对象，这些对象是通过`CharSequence`（通常是`String`）上的连续字符序列，带有类型和得分的。`Chunks`可以重叠。`Chunk`对象被存储在`Chunking`中：

```py
String textStored = chunking.charSequence().toString();
for (Chunk sentence : sentences) {
  int start = sentence.start();
  int end = sentence.end();
  System.out.println("SENTENCE :" 
    + textStored.substring(start,end));
  }
}
```

首先，我们恢复了基础文本字符串`textStored`，它是`Chunks`的基础。它与`text`相同，但我们希望说明`Chunking`类中这个可能有用的方法，这个方法在递归或其他上下文中可能会出现，其中`CharSequence`可能不可用。

剩余的`for`循环遍历句子并使用`String`的`substring()`方法将其打印出来。

## 还有更多...

在讲解如何创建自己的句子检测器之前，值得一提的是 LingPipe 有一个`MedlineSentenceModel`，它专门处理医学研究文献中常见的句子类型。它已经处理了大量数据，应该是你在这类数据上进行句子检测的起点。

### 嵌套句子

特别是在文学作品中，句子可能包含嵌套的句子。考虑以下内容：

```py
John said "this is a nested sentence" and then shut up.
```

前述句子将被正确标注为：

```py
[John said "[this is a nested sentence]" and then shut up.]
```

这种嵌套与语言学中嵌套句子的概念不同，后者是基于语法角色的。考虑以下例子：

```py
[[John ate the gorilla] and [Mary ate the burger]].
```

这个句子由两个在语言学上完整的句子通过`and`连接而成。两者的区别在于前者是由标点符号决定的，后者则由语法功能决定。这个区别是否重要可以讨论。然而，前者的情况在编程中更容易识别。

然而，在工业环境中我们很少需要建模嵌套句子，但在我们的 MUC-6 系统和各种共指解析研究系统中，我们已经涉及过此问题。这超出了食谱书的范围，但请注意这个问题。LingPipe 没有开箱即用的嵌套句子检测功能。

# 句子检测的评估

就像我们做的大多数事情一样，我们希望能够评估组件的性能。句子检测也不例外。句子检测是一种跨度注释，区别于我们之前对分类器和分词的评估。由于文本中可能有不属于任何句子的字符，因此存在句子开始和句子结束的概念。一个不属于句子的字符示例是来自 HTML 页面的 JavaScript。

以下示例将引导你完成创建评估数据并通过评估类运行它的步骤。

## 如何操作...

执行以下步骤来评估句子检测：

1.  打开文本编辑器，复制并粘贴一些你想用来评估句子检测的文学作品，或者你可以使用我们提供的默认文本，如果没有提供自己的数据，则会使用此文本。最简单的方法是使用纯文本。

1.  插入平衡的`[`和`]`来标识文本中句子的开始和结束。如果文本中已经包含`[`或`]`，请选择文本中没有的其他字符作为句子分隔符——大括号或斜杠是不错的选择。如果使用不同的分隔符，您需要相应地修改源代码并重新创建 JAR 文件。代码假设使用单字符文本分隔符。以下是来自《银河系漫游指南》的句子注释文本示例——注意并非每个字符都在句子中；一些空格位于句子之间：

    ```py
    [The Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.] [It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.]
    ```

1.  打开命令行并运行以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.EvaluateAnnotatedSentences
    TruePos: 0-83/The Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.:S
    TruePos: 84-233/It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.:S

    ```

1.  对于这些数据，代码将显示两个完美匹配的句子，这些句子与用`[]`注释的句子一致，正如`TruePos`标签所示。

1.  一个好的练习是稍微修改注释以强制产生错误。我们将第一个句子边界向前移动一个字符：

    ```py
    T[he Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.] [It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.]

    ```

1.  保存并重新运行修改后的注释文件后，结果如下：

    ```py
    TruePos: 84-233/It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.:S
    FalsePos: 0-83/The Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.:S
    FalseNeg: 1-83/he Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.:S

    ```

    通过改变真值注释，会产生一个假阴性，因为句子跨度错过了一个字符。此外，由于句子检测器识别了 0-83 字符序列，产生了一个假阳性。

1.  通过与注释和各种数据的交互，了解评估的工作原理以及句子检测器的能力是一个好主意。

## 工作原理...

该类从消化注释文本并将句子块存储到评估对象开始。然后，创建句子检测器，就像我们在前面的示例中所做的那样。代码最后通过将创建的句子检测器应用于文本，并打印结果。

### 解析注释数据

给定带有`[]`注释的文本表示句子边界，这意味着必须恢复句子的正确偏移量，并且必须创建原始的未注释文本，即没有任何`[]`。跨度解析器编写起来可能有些棘手，以下代码为了简化而不是为了效率或正确的编程技巧：

```py
String path = args.length > 0 ? args[0] 
             : "data/hitchHikersGuide.sentDetected";
char[] chars 
  = Files.readCharsFromFile(new File(path), Strings.UTF8);
StringBuilder rawChars = new StringBuilder();
int start = -1;
int end = 0;
Set<Chunk> sentChunks = new HashSet<Chunk>();
```

前面的代码将整个文件作为一个`char[]`数组读取，并使用适当的字符编码。此外，注意对于大文件，使用流式处理方法会更加节省内存。接下来，设置了一个未注释字符的累加器`StringBuilder`对象，并通过`rawChars`变量进行存储。所有遇到的不是`[`或`]`的字符都将被附加到该对象中。剩余的代码设置了用于句子开始和结束的计数器，这些计数器被索引到未注释的字符数组中，并设置了一个用于注释句子片段的`Set<Chunk>`累加器。

以下的`for`循环逐个字符遍历注释过的字符序列：

```py
for (int i=0; i < chars.length; ++i) {
  if (chars[i] == '[') {
    start = rawChars.length();
  }
  else if (chars[i] == ']') {
    end = rawChars.length();

    Chunk chunk = ChunkFactory.createChunk(start,end, SentenceChunker.SENTENCE_CHUNK_TYPE);
    sentChunks.add(chunk);}
  else {
    rawChars.append(chars[i]);
  }
}
String originalText = rawChars.toString();
```

第一个`if (chars[i] == '[')`用于测试注释中句子的开始，并将`start`变量设置为`rawChars`的长度。迭代变量`i`包括由注释添加的长度。相应的`else if (chars[i] == ']')`语句处理句子结束的情况。请注意，这个解析器没有错误检查——这是一个非常糟糕的设计，因为如果使用文本编辑器输入，注释错误非常可能发生。然而，这样做是为了保持代码尽可能简洁。在接下来的章节中，我们将提供一个带有最小错误检查的示例。一旦找到句子的结束，就会使用`ChunkFactory.createChunk`根据偏移量为句子创建一个分块，并且使用标准的 LingPipe 句子类型`SentenceChunker.SENTENCE_CHUNK_TYPE`，这是接下来评估类正确工作的必需条件。

剩下的`else`语句适用于所有非句子边界的字符，它仅仅将字符添加到`rawChars`累加器中。`for`循环外部创建`String unannotatedText`时，可以看到这个累加器的结果。现在，我们已经将句子分块正确地索引到文本字符串中。接下来，我们将创建一个合适的`Chunking`对象：

```py
ChunkingImpl sentChunking = new ChunkingImpl(unannotatedText);
for (Chunk chunk : sentChunks) {
  sentChunking.add(chunk);
}
```

实现类`ChunkingImpl`（`Chunking`是接口）在构造时需要底层文本，这就是为什么我们没有在前面的循环中直接填充它。LingPipe 通常会尝试使对象构造完整。如果可以不使用底层`CharSequence`方法创建`Chunking`，那么调用`charSequence()`方法时会返回什么呢？空字符串会误导用户。或者，返回`null`需要捕获并处理。最好直接强制对象构造以确保其合理性。

接下来，我们将看到上一节中句子分块器的标准配置：

```py
boolean eosIsSentBoundary = false;
boolean balanceParens = true;
SentenceModel sentenceModel = new IndoEuropeanSentenceModel(eosIsSentBoundary, balanceParens);
TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
SentenceChunker sentenceChunker = new SentenceChunker(tokFactory,sentenceModel);
```

有趣的部分紧随其后，评估器将`sentenceChunker`作为待评估的参数：

```py
SentenceEvaluator evaluator = new SentenceEvaluator(sentenceChunker);
```

接下来，`handle(sentChunking)`方法将把我们刚刚解析的文本转化为`Chunking`，并在`sentChunking`中提供的`CharSequence`上运行句子检测器，并设置评估：

```py
evaluator.handle(sentChunking);
```

然后，我们只需要获取评估数据，并通过对比正确的句子检测与系统执行的结果，逐步分析差异：

```py
SentenceEvaluation eval = evaluator.evaluation();
ChunkingEvaluation chunkEval = eval.chunkingEvaluation();
for (ChunkAndCharSeq truePos : chunkEval.truePositiveSet()) {
  System.out.println("TruePos: " + truePos);
}
for (ChunkAndCharSeq falsePos : chunkEval.falsePositiveSet()) {
  System.out.println("FalsePos: " + falsePos);
}
for (ChunkAndCharSeq falseNeg : chunkEval.falseNegativeSet()){
  System.out.println("FalseNeg: " + falseNeg);
}
```

这个食谱并没有涵盖所有评估方法——可以查看 Javadoc——但它确实提供了句子检测调整器可能最需要的内容；这列出了句子检测器正确识别的内容（真阳性）、检测到但错误的句子（假阳性）以及漏掉的句子（假阴性）。注意，在跨度注解中，真阴性没有太大意义，因为它们将是所有可能的跨度集合，但不包含在正确的句子检测中。

# 调整句子检测

很多数据将抵抗`IndoEuropeanSentenceModel`的魅力，所以这个食谱将为修改句子检测以适应新类型的句子提供一个起点。不幸的是，这是一个非常开放的问题，所以我们将专注于技术，而不是句子格式的可能性。

## 如何做……

这个食谱将遵循一个常见的模式：创建评估数据、设置评估并开始动手。我们开始吧：

1.  拿出你最喜欢的文本编辑器并标记一些数据——我们将使用`[`和`]`标记法。以下是一个违反我们标准`IndoEuropeanSentenceModel`的示例：

    ```py
    [All decent people live beyond their incomes nowadays, and those who aren't respectable live beyond other people's.]  [A few gifted individuals manage to do both.]

    ```

1.  我们将把前面的句子放入`data/saki.sentDetected.txt`并运行它：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.EvaluateAnnotatedSentences data/saki.sentDetected 
    FalsePos: 0-159/All decent people live beyond their incomes nowadays, and those who aren't respectable live beyond other people's.  A few gifted individuals manage to do both.:S
    FalseNeg: 0-114/All decent people live beyond their incomes nowadays, and those who aren't respectable live beyond other people's.:S
    FalseNeg: 116-159/A few gifted individuals manage to do both.:S

    ```

## 还有更多……

唯一的假阳性对应的是检测到的一个句子，两个假阴性是我们这里注释的两个未被检测到的句子。发生了什么？句子模型漏掉了`people's.`作为句子结尾。如果删除撇号，句子就能正确检测到——发生了什么？

首先，我们来看一下后台运行的代码。`IndoEuropeanSentenceModel`通过配置来自`HeuristicSentenceModel`的 Javadoc 中的几类标记来扩展`HeuristicSentenceModel`：

+   **可能的停止符**：这些是可以作为句子结尾的标记。这个集合通常包括句尾标点符号标记，比如句号（.）和双引号（"）。

+   **不可能的倒数第二个**：这些是可能不是句子的倒数第二个（倒数第二）标记。这个集合通常由缩写或首字母缩写组成，例如`Mr`。

+   **不可能的开头**：这些是可能不是句子开头的标记。这个集合通常包括应该与前一句连接的标点符号字符，如结束引号（''）。

`IndoEuropeanSentenceModel`不可配置，但从 Javadoc 中可以看出，所有单个字符都被视为不可能的倒数第二个字符。单词`people's`被分词为`people`、`'`、`s`和`.`。单个字符`s`位于`.`的倒数第二位，因此会被阻止。如何修复这个问题？

有几种选择呈现出来：

+   忽略这个错误，假设它不会频繁发生

+   通过创建自定义句子模型来修复

+   通过修改分词器以避免拆分撇号来修复

+   为该接口编写一个完整的句子检测模型

第二种选择，创建一个自定义句子模型，通过将`IndoEuropeanSentenceModel`的源代码复制到一个新类中并进行修改来处理，这是最简单的做法，因为相关的数据结构是私有的。这样做是为了简化类的序列化——几乎不需要将任何配置写入磁盘。在示例类中，有一个`MySentenceModel.java`文件，它通过明显的包名和导入语句的变化来区分：

```py
IMPOSSIBLE_PENULTIMATES.add("R");
//IMPOSSIBLE_PENULTIMATES.add("S"); breaks on "people's."
//IMPOSSIBLE_PENULTIMATES.add("T"); breaks on "didn't."
IMPOSSIBLE_PENULTIMATES.add("U");
```

前面的代码只是注释掉了两种可能的单字母倒数第二个标记的情况，这些情况是单个字符的单词。要查看其效果，请将句子模型更改为`SentenceModel sentenceModel = new MySentenceModel();`，并在`EvaluateAnnotatedSentences.java`类中重新编译并运行。

如果你将前面的代码视为一个合理的平衡，它可以找到以可能的缩写结尾的句子与非句子情况之间的平衡，例如`[Hunter S. Thompson is a famous fellow.]`，它会将`S.`识别为句子边界。

扩展`HeuristicSentenceModel`对于多种类型的数据非常有效。Mitzi Morris 构建了`MedlineSentenceModel.java`，它设计得很好，适用于 MEDLINE 研究索引中提供的摘要。

看待前面问题的一种方式是，缩写不应被拆分为标记用于句子检测。`IndoEuropeanTokenizerFactory`应该进行调整，以将"people's"和其他缩写保持在一起。虽然这初看起来似乎稍微比第一个解决方案好，但它可能会遇到`IndoEuropeanSentenceModel`是针对特定的分词方式进行调整的问题，而在没有评估语料库的情况下，改变的后果是未知的。

另一种选择是编写一个完全新的句子检测类，支持`SentenceModel`接口。面对像 Twitter 流这样的高度新颖的数据集，我们可以考虑使用基于机器学习的跨度注释技术，如 HMMs 或 CRFs，这些内容在第四章，*标注词汇和标记*中以及本章末尾讨论过。

# 标记字符串中的嵌入块——句子块示例

先前食谱中展示块的方法不适用于需要修改底层字符串的应用程序。例如，一个情感分析器可能只想突出显示那些情感强烈的正面句子，而不标记其余句子，同时仍然显示整个文本。在生成标记化文本时的一个小难点是，添加标记会改变底层字符串。这个食谱提供了通过逆序添加块来插入块的工作代码。

## 如何实现...

尽管这个食谱在技术上可能不复杂，但它对于在文本中添加跨度注释非常有用，而无需从零开始编写代码。`src/com/lingpipe/coobook/chapter5/WriteSentDetectedChunks`类中包含了参考代码：

1.  句子块是根据第一个句子检测食谱创建的。以下代码提取块作为`Set<Chunk>`，然后按照`Chunk.LONGEST_MATCH_ORDER_COMPARITOR`进行排序。在 Javadoc 中，该比较器被定义为：

    > *根据文本位置比较两个块。如果一个块比另一个块晚开始，或者它们在相同位置开始但结束得更早，那么前者更大。*

    还有`TEXT_ORDER_COMPARITOR`，如下所示：

    ```py
    String textStored = chunking.charSequence().toString();
    Set<Chunk> chunkSet = chunking.chunkSet();
    System.out.println("size: " + chunkSet.size());
    Chunk[] chunkArray = chunkSet.toArray(new Chunk[0]);
    Arrays.sort(chunkArray,Chunk.LONGEST_MATCH_ORDER_COMPARATOR);
    ```

1.  接下来，我们将按逆序遍历块，这样可以避免为`StringBuilder`对象的变化长度保持偏移量变量。偏移量变量是一个常见的错误来源，因此这个食谱尽可能避免使用它们，但使用了非标准的逆序循环迭代，这可能更糟：

    ```py
    StringBuilder output = new StringBuilder(textStored);
    int sentBoundOffset = 0;
    for (int i = chunkArray.length -1; i >= 0; --i) {
      Chunk chunk = chunkArray[i];
      String sentence = textStored.substring(chunk.start(), chunk.end());
      if (sentence.contains("like")) {
        output.insert(chunk.end(),"}");
        output.insert(chunk.start(),"{");
      }
    }
    System.out.println(output.toString());
    ```

1.  前面的代码通过查找字符串`like`来进行非常简单的情感分析，如果找到则标记该句子为`true`。请注意，这段代码无法处理重叠的块或嵌套的块。它假设一个单一的、不重叠的块集合。一些示例输出如下：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.WriteSentDetectedChunks
    Enter text followed by new line
    >People like to ski. But sometimes it is terrifying. 
    size: 2
    {People like to ski.} But sometimes it is terrifying. 

    ```

1.  要打印嵌套的块，请查看下面的*段落* *检测*食谱。

# 段落检测

一组句子的典型包含结构是段落。它可以在标记语言中显式设置，例如 HTML 中的`<p>`，或者通过两个或更多的换行符来设置，这也是段落通常如何呈现的方式。我们处于自然语言处理的领域，这里没有硬性规定，所以我们为这种含糊其辞表示歉意。我们将在本章中处理一些常见的示例，并将推广的部分留给你来完成。

## 如何实现...

我们从未为段落检测设置过评估工具，但它可以通过类似句子检测的方式进行实现。这个食谱将演示一个简单的段落检测程序，它做了一件非常重要的事情——在进行嵌入句子检测的同时，保持原始文档的偏移量。细节上的关注会在你需要以对句子或文档的其他子跨度（例如命名实体）敏感的方式标记文档时帮助你。请考虑以下示例：

```py
Sentence 1\. Sentence 2
Sentence 3\. Sentence 4.
```

它被转化为以下内容：

```py
{[Sentence 1.] [Sentence 2]}

{[Sentence 3.] [Sentence 4.]
}
```

在前面的代码片段中，`[]` 表示句子，`{}` 表示段落。我们将直接跳入这个配方的代码，位于 `src/com/lingpipe/cookbook/chapter5/ParagraphSentenceDetection.java`：

1.  示例代码在段落检测技术方面几乎没有提供什么。它是一个开放性问题，你必须运用你的聪明才智来解决它。我们的段落检测器是一个可悲的 `split("\n\n")`，在更复杂的方法中，它会考虑上下文、字符和其他特征，这些特征过于独特，无法一一涵盖。以下是读取整个文档作为字符串并将其拆分为数组的代码开头。请注意，`paraSeperatorLength` 是用于段落拆分的字符数——如果拆分长度有所变化，那么该长度将必须与对应段落相关联：

    ```py
    public static void main(String[] args) throws IOException {
      String document = Files.readFromFile(new File(args[0]), Strings.UTF8);
      String[] paragraphs = document.split("\n\n");
      int paraSeparatorLength = 2;
    ```

1.  该配方的真正目的是帮助维护原始文档中字符偏移量的机制，并展示嵌入式处理。这将通过保持两个独立的块进行：一个用于段落，另一个用于句子：

    ```py
    ChunkingImpl paraChunking = new ChunkingImpl(document.toCharArray(),0,document.length());
    ChunkingImpl sentChunking = new ChunkingImpl(paraChunking.charSequence());
    ```

1.  接下来，句子检测器将以与上一配方中相同的方式进行设置：

    ```py
    boolean eosIsSentBoundary = true;
    boolean balanceParens = false;
    SentenceModel sentenceModel = new IndoEuropeanSentenceModel(eosIsSentBoundary, balanceParens);
    SentenceChunker sentenceChunker = new SentenceChunker(IndoEuropeanTokenizerFactory.INSTANCE, sentenceModel);
    ```

1.  块处理会遍历段落数组，并为每个段落构建一个句子块。这个方法中稍显复杂的部分是，句子块的偏移量是相对于段落字符串的，而不是整个文档。因此，变量的开始和结束在代码中会通过文档偏移量进行更新。块本身没有调整开始和结束的方式，因此必须创建一个新的块 `adjustedSentChunk`，并将适当的偏移量应用到段落的开始，并将其添加到 `sentChunking` 中：

    ```py
    int paraStart = 0;
    for (String paragraph : paragraphs) {
      for (Chunk sentChunk : sentenceChunker.chunk(paragraph).chunkSet()) {
        Chunk adjustedSentChunk = ChunkFactory.createChunk(sentChunk.start() + paraStart,sentChunk.end() + paraStart, "S");
        sentChunking.add(adjustedSentChunk);
      }
    ```

1.  循环的其余部分添加段落块，然后用段落的长度加上段落分隔符的长度更新段落的起始位置。这将完成将正确偏移的句子和段落插入到原始文档字符串中的过程：

    ```py
    paraChunking.add(ChunkFactory.createChunk(paraStart, paraStart + paragraph.length(),"P"));
    paraStart += paragraph.length() + paraSeparatorLength;
    }
    ```

1.  程序的其余部分涉及打印出带有一些标记的段落和句子。首先，我们将创建一个同时包含句子和段落块的块：

    ```py
    String underlyingString = paraChunking.charSequence().toString();
    ChunkingImpl displayChunking = new ChunkingImpl(paraChunking.charSequence());
    displayChunking.addAll(sentChunking.chunkSet());
    displayChunking.addAll(paraChunking.chunkSet());
    ```

1.  接下来，`displayChunking` 将通过恢复 `chunkSet` 进行排序，转换为一个块数组，并应用静态比较器：

    ```py
    Set<Chunk> chunkSet = displayChunking.chunkSet();
    Chunk[] chunkArray = chunkSet.toArray(new Chunk[0]);
    Arrays.sort(chunkArray, Chunk.LONGEST_MATCH_ORDER_COMPARATOR);
    ```

1.  我们将使用与 *在字符串中标记嵌入块 - 句子块示例* 配方中相同的技巧，即将标记反向插入字符串中。我们需要保持一个偏移量计数器，因为嵌套的句子会延长结束段落标记的位置。该方法假设没有块重叠，并且句子始终包含在段落内：

    ```py
    StringBuilder output = new StringBuilder(underlyingString);
    int sentBoundOffset = 0;
    for (int i = chunkArray.length -1; i >= 0; --i) {
      Chunk chunk = chunkArray[i];
      if (chunk.type().equals("P")) {
        output.insert(chunk.end() + sentBoundOffset,"}");
        output.insert(chunk.start(),"{");
        sentBoundOffset = 0;
      }
      if (chunk.type().equals("S")) {
        output.insert(chunk.end(),"]");
        output.insert(chunk.start(),"[");
        sentBoundOffset += 2;
      }
    }
    System.out.println(output.toString());
    ```

1.  这就是该配方的全部内容。

# 简单的名词短语和动词短语

本配方将展示如何查找简单的**名词短语**（**NP**）和**动词短语**（**VP**）。这里的“简单”是指短语内没有复杂结构。例如，复杂的 NP "The rain in Spain" 将被分解成两个简单的 NP 块：“The rain”和“Spain”。这些短语也称为“基础短语”。

本配方不会深入探讨如何计算基础 NP/VP，而是介绍如何使用这个类——它非常实用，如果你想了解它如何工作，可以包括源代码。

## 如何实现……

和许多其他的配方一样，我们在这里提供一个命令行交互式界面：

1.  打开命令行并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.PhraseChunker
    INPUT> The rain in Spain falls mainly on the plain.
    The/at rain/nn in/in Spain/np falls/vbz mainly/rb on/in the/at plain/jj ./. 
     noun(0,8) The rain
     noun(12,17) Spain
     verb(18,30) falls mainly
     noun(34,43) the plain

    ```

## 它是如何工作的……

`main()`方法首先反序列化词性标注器，然后创建`tokenizerFactory`：

```py
public static void main(String[] args) throws IOException, ClassNotFoundException {
  File hmmFile = new File("models/pos-en-general-brown.HiddenMarkovModel");
  HiddenMarkovModel posHmm = (HiddenMarkovModel) AbstractExternalizable.readObject(hmmFile);
  HmmDecoder posTagger  = new HmmDecoder(posHmm);
  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
```

接下来，构造`PhraseChunker`，这是一种启发式的方法来解决该问题。查看源代码了解它是如何工作的——它从左到右扫描输入，查找 NP/VP 的开始，并尝试逐步添加到短语中：

```py
PhraseChunker chunker = new PhraseChunker(posTagger,tokenizerFactory);
```

我们的标准控制台输入/输出代码如下：

```py
BufferedReader bufReader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.print("\n\nINPUT> ");
  String input = bufReader.readLine();
```

然后，输入被分词，词性标注，并打印出标记和标签：

```py
Tokenizer tokenizer = tokenizerFactory.tokenizer(input.toCharArray(),0,input.length());
String[] tokens = tokenizer.tokenize();
List<String> tokenList = Arrays.asList(tokens);
Tagging<String> tagging = posTagger.tag(tokenList);
for (int j = 0; j < tokenList.size(); ++j) {
  System.out.print(tokens[j] + "/" + tagging.tag(j) + " ");
}
System.out.println();
```

然后计算并打印 NP/VP 的分块结果：

```py
Chunking chunking = chunker.chunk(input);
CharSequence cs = chunking.charSequence();
for (Chunk chunk : chunking.chunkSet()) {
  String type = chunk.type();
  int start = chunk.start();
  int end = chunk.end();
  CharSequence text = cs.subSequence(start,end);
  System.out.println("  " + type + "(" + start + ","+ end + ") " + text);
  }
```

这里有一个更全面的教程，访问[`alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html)。

# 基于正则表达式的 NER 分块

**命名实体识别**（**NER**）是识别文本中具体事物提及的过程。考虑一个简单的名称；位置命名实体识别器可能会在以下文本中分别找到`Ford Prefect`和`Guildford`作为人名和地名：

```py
Ford Prefect used to live in Guildford before he needed to move.
```

我们将从构建基于规则的 NER 系统开始，逐步过渡到机器学习方法。这里，我们将构建一个能够从文本中提取电子邮件地址的 NER 系统。

## 如何实现……

1.  在命令提示符中输入以下命令：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.RegexNer

    ```

1.  与程序的交互如下进行：

    ```py
    Enter text, . to quit:
    >Hello,my name is Foo and my email is foo@bar.com or you can also contact me at foo.bar@gmail.com.
    input=Hello,my name is Foo and my email is foo@bar.com or you can also contact me at foo.bar@gmail.com.
    chunking=Hello,my name is Foo and my email is foo@bar.com or you can also contact me at foo.bar@gmail.com. : [37-48:email@0.0, 79-96:email@0.0]
     chunk=37-48:email@0.0  text=foo@bar.com
     chunk=79-96:email@0.0  text=foo.bar@gmail.com

    ```

1.  你可以看到`foo@bar.com`和`foo.bar@gmail.com`都被识别为有效的`e-mail`类型块。此外，请注意，句子末尾的句号不是第二个电子邮件地址的一部分。

## 它是如何工作的……

正则表达式分块器查找与给定正则表达式匹配的块。本质上，`java.util.regex.Matcher.find()`方法用于迭代地查找匹配的文本片段，然后将这些片段转换为 Chunk 对象。`RegExChunker`类包装了这些步骤。`src/com/lingpipe/cookbook/chapter5/RegExNer.java`的代码如下所述：

```py
public static void main(String[] args) throws IOException {
  String emailRegex = "A-Za-z0-9*)" + + "@([A-Za-z0-9]+)" + "(([\\.\\-]?[a-zA-Z0-9]+)*)\\.([A-Za-z]{2,})";
  String chunkType = "email";
  double score = 1.0;
  Chunker chunker = new RegExChunker(emailRegex,chunkType,score);
```

所有有趣的工作都在前面的代码行中完成。`emailRegex`是从互联网上获取的——参见以下源代码，其余的部分是在设置`chunkType`和`score`。

其余代码会读取输入并输出分块结果：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
  String input = "";
  while (true) {
    System.out.println("Enter text, . to quit:");
    input = reader.readLine();
    if(input.equals(".")){
      break;
    }
    Chunking chunking = chunker.chunk(input);
    System.out.println("input=" + input);
    System.out.println("chunking=" + chunking);
    Set<Chunk> chunkSet = chunking.chunkSet();
    Iterator<Chunk> it = chunkSet.iterator();
    while (it.hasNext()) {
      Chunk chunk = it.next();
      int start = chunk.start();
      int end = chunk.end();
      String text = input.substring(start,end);
      System.out.println("     chunk=" + chunk + " text=" + text);
    }
  }
}
```

## 另见

+   用于匹配电子邮件地址的正则表达式来自[regexlib.com](http://regexlib.com)，网址为[`regexlib.com/DisplayPatterns.aspx?cattabindex=0&categoryId=1`](http://regexlib.com/DisplayPatterns.aspx?cattabindex=0&categoryId=1)。

# 基于词典的命名实体识别（NER）分块

在许多网站和博客，特别是在网络论坛上，你可能会看到关键词高亮，这些关键词链接到你可以购买产品的页面。同样，新闻网站也提供关于人物、地点和流行事件的专题页面，例如[`www.nytimes.com/pages/topics/`](http://www.nytimes.com/pages/topics/)。

其中许多操作是完全自动化的，通过基于词典的`Chunker`很容易实现。编译实体名称及其类型的列表非常简单。精确的字典分块器根据分词后的字典条目的精确匹配来提取分块。

LingPipe 中基于字典的分块器的实现基于 Aho-Corasick 算法，该算法在线性时间内找到所有与字典匹配的项，无论匹配数量或字典大小如何。这使得它比做子字符串搜索或使用正则表达式的天真方法更高效。

## 如何操作……

1.  在你选择的 IDE 中运行`chapter5`包中的`DictionaryChunker`类，或者在命令行中输入以下命令：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.DictionaryChunker

    ```

1.  由于这个特定的分块器示例强烈偏向于《银河系漫游指南》，我们使用一个涉及一些角色的句子：

    ```py
    Enter text, . to quit:
    Ford and Arthur went up the bridge of the Heart of Gold with Marvin
    CHUNKER overlapping, case sensitive
     phrase=|Ford| start=0 end=4 type=PERSON score=1.0
     phrase=|Arthur| start=9 end=15 type=PERSON score=1.0
     phrase=|Heart| start=42 end=47 type=ORGAN score=1.0
     phrase=|Heart of Gold| start=42 end=55 type=SPACECRAFT score=1.0
     phrase=|Marvin| start=61 end=67 type=ROBOT score=1.0

    ```

1.  请注意，我们有来自`Heart`和`Heart of Gold`的重叠部分。正如我们将看到的，这可以配置为不同的行为方式。

## 它是如何工作的……

基于字典的 NER 驱动了大量的自动链接，针对非结构化文本数据。我们可以使用以下步骤构建一个。

代码的第一步将创建`MapDictionary<String>`来存储字典条目：

```py
static final double CHUNK_SCORE = 1.0;

public static void main(String[] args) throws IOException {
  MapDictionary<String> dictionary = new MapDictionary<String>();
  MapDictionary<String> dictionary = new MapDictionary<String>();
```

接下来，我们将用`DictionaryEntry<String>`填充字典，其中包括类型信息和将用于创建分块的得分：

```py
dictionary.addEntry(new DictionaryEntry<String>("Arthur","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Ford","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Trillian","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Zaphod","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Marvin","ROBOT",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Heart of Gold", "SPACECRAFT",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("HitchhikersGuide", "PRODUCT",CHUNK_SCORE));
```

在`DictionaryEntry`构造函数中，第一个参数是短语，第二个字符串参数是类型，最后一个双精度参数是分块的得分。字典条目始终区分大小写。字典中没有限制不同实体类型的数量。得分将作为分块得分传递到基于字典的分块器中。

接下来，我们将构建`Chunker`：

```py
boolean returnAllMatches = true;
boolean caseSensitive = true;
ExactDictionaryChunker dictionaryChunker = new ExactDictionaryChunker(dictionary, IndoEuropeanTokenizerFactory.INSTANCE, returnAllMatches,caseSensitive);
```

精确的字典分块器可以配置为提取所有匹配的分块，或者通过`returnAllMatches`布尔值将结果限制为一致的非重叠分块。查看 Javadoc 以了解精确的标准。还有一个`caseSensitive`布尔值。分块器需要一个分词器，因为它根据符号匹配分词，并且在匹配过程中会忽略空白字符。

接下来是我们的标准输入/输出代码，用于控制台交互：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
String text = "";
while (true) {
  System.out.println("Enter text, . to quit:");
  text = reader.readLine();
  if(text.equals(".")){
    break;
  }
```

剩余的代码创建了一个分块器，遍历分块，并将它们打印出来：

```py
System.out.println("\nCHUNKER overlapping, case sensitive");
Chunking chunking = dictionaryChunker.chunk(text);
  for (Chunk chunk : chunking.chunkSet()) {
    int start = chunk.start();
    int end = chunk.end();
    String type = chunk.type();
    double score = chunk.score();
    String phrase = text.substring(start,end);
    System.out.println("     phrase=|" + phrase + "|" + " start=" + start + " end=" + end + " type=" + type + " score=" + score);
```

字典块划分器在基于机器学习的系统中也非常有用。通常，总会有一些实体类别，使用这种方式最容易识别。*混合命名实体识别源*食谱介绍了如何处理多个命名实体来源。

# 词语标记与块之间的转换 – BIO 编解码器

在第四章中，*标签词语与词元*，我们使用了 HMM 和 CRF 来为词语/词元添加标签。本食谱讨论了如何通过使用**开始、内含和结束**（**BIO**）标签，从标记中创建块，进而编码可能跨越多个词语/词元的块。这也是现代命名实体识别系统的基础。

## 准备就绪

标准的 BIO 标记方案中，块类型 X 的第一个词元被标记为 B-X（开始），同一块中的所有后续词元被标记为 I-X（内含）。所有不在块中的词元被标记为 O（结束）。例如，具有字符计数的字符串：

```py
John Jones Mary and Mr. Jones
01234567890123456789012345678
0         1         2         
```

它可以被标记为：

```py
John  B_PERSON
Jones  I_PERSON
Mary  B_PERSON
and  O
Mr    B_PERSON
.    I_PERSON
Jones  I_PERSON
```

相应的块将是：

```py
0-10 "John Jones" PERSON
11-15 "Mary" PERSON
20-29 "Mr. Jones" PERSON
```

## 如何做…

程序将展示标记和块的最简单映射关系，反之亦然：

1.  运行以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.BioCodec

    ```

1.  程序首先打印出将被标记的字符串：

    ```py
    Tagging for :The rain in Spain.
    The/B_Weather
    rain/I_Weather
    in/O
    Spain/B_Place
    ./O

    ```

1.  接下来，打印出块：

    ```py
    Chunking from StringTagging
    0-8:Weather@-Infinity
    12-17:Place@-Infinity

    ```

1.  然后，从刚刚显示的块中创建标记：

    ```py
    StringTagging from Chunking
    The/B_Weather
    rain/I_Weather
    in/O
    Spain/B_Place
    ./O

    ```

## 它是如何工作的…

代码首先手动构造`StringTagging`——我们将在 HMM 和 CRF 中看到同样的程序化操作，但这里是显式的。然后它会打印出创建的`StringTagging`：

```py
public static void main(String[] args) {
  List<String> tokens = new ArrayList<String>();
  tokens.add("The");
  tokens.add("rain");
  tokens.add("in");
  tokens.add("Spain");
  tokens.add(".");
  List<String> tags = new ArrayList<String>();
  tags.add("B_Weather");
  tags.add("I_Weather");
  tags.add("O");
  tags.add("B_Place");
  tags.add("O");
  CharSequence cs = "The rain in Spain.";
  //012345678901234567
  int[] tokenStarts = {0,4,9,12,17};
  int[] tokenEnds = {3,8,11,17,17};
  StringTagging tagging = new StringTagging(tokens, tags, cs, tokenStarts, tokenEnds);
  System.out.println("Tagging for :" + cs);
  for (int i = 0; i < tagging.size(); ++i) {
    System.out.println(tagging.token(i) + "/" + tagging.tag(i));
  }
```

接下来，它将构造`BioTagChunkCodec`，并将刚刚打印出来的标记转换为块，然后打印出块：

```py
BioTagChunkCodec codec = new BioTagChunkCodec();
Chunking chunking = codec.toChunking(tagging);
System.out.println("Chunking from StringTagging");
for (Chunk chunk : chunking.chunkSet()) {
  System.out.println(chunk);
}
```

剩余的代码反转了这一过程。首先，创建一个不同的`BioTagChunkCodec`，并使用`boolean`类型的`enforceConsistency`，如果为`true`，它会检查由提供的分词器创建的词元是否完全与块的开始和结束对齐。如果没有对齐，根据使用场景，我们可能会得到标记和块之间无法维持的关系：

```py
boolean enforceConsistency = true;
BioTagChunkCodec codec2 = new BioTagChunkCodec(IndoEuropeanTokenizerFactory.INSTANCE, enforceConsistency);
StringTagging tagging2 = codec2.toStringTagging(chunking);
System.out.println("StringTagging from Chunking");
for (int i = 0; i < tagging2.size(); ++i) {
  System.out.println(tagging2.token(i) + "/" + tagging2.tag(i));
}
```

最后的`for`循环仅仅打印出由`codec2.toStringTagging()`方法返回的标记。

## 还有更多…

本食谱通过最简单的标记与块之间的映射示例进行讲解。`BioTagChunkCodec`还接受`TagLattice<String>`对象，生成 n-best 输出，正如后面将在 HMM 和 CRF 块器中展示的那样。

# 基于 HMM 的命名实体识别（NER）

`HmmChunker`使用 HMM 对标记化的字符序列进行块划分。实例包含用于该模型的 HMM 解码器和分词器工厂。块划分器要求 HMM 的状态符合块的逐个词元编码。它使用分词器工厂将块分解为词元和标签序列。请参考第四章中的*隐马尔可夫模型（HMM） – 词性*食谱，*标签词语与词元*。

我们将讨论如何训练`HmmChunker`并将其用于`CoNLL2002`西班牙语任务。你可以并且应该使用自己的数据，但这个配方假设训练数据将采用`CoNLL2002`格式。

训练是通过一个`ObjectHandler`完成的，`ObjectHandler`提供了训练实例。

## 准备工作

由于我们希望训练这个 chunker，我们需要使用**计算自然语言学习**（**CoNLL**）模式标注一些数据，或者使用公开的模式。为了提高速度，我们选择获取一个在 CoNLL 2002 任务中可用的语料库。

### 注意

ConNLL 是一个年度会议，赞助一个比赛。2002 年，这个比赛涉及了西班牙语和荷兰语的命名实体识别（NER）。

数据可以从[`www.cnts.ua.ac.be/conll2002/ner.tgz`](http://www.cnts.ua.ac.be/conll2002/ner.tgz)下载。

类似于我们在前一个配方中展示的内容；让我们来看一下这些数据的样子：

```py
El       O 
Abogado     B-PER 
General     I-PER 
del     I-PER 
Estado     I-PER 
,       O 
Daryl     B-PER 
Williams     I-PER 
,       O
```

使用这种编码方式，短语*El Abogado General del Estado*和*Daryl Williams*被编码为人物（person），其开始和继续的标记分别为 B-PER 和 I-PER。

### 注意

数据中有一些格式错误，必须修复这些错误，才能让我们的解析器处理它们。在数据目录解压`ner.tgz`后，你需要进入`data/ner/data`，解压以下文件，并按照指示进行修改：

```py
esp.train, line 221619, change I-LOC to B-LOC
esp.testa, line 30882, change I-LOC to B-LOC
esp.testb, line 9291, change I-LOC to B-LOC

```

## 如何操作……

1.  使用命令行，输入以下命令：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.HmmNeChunker

    ```

1.  如果模型不存在，它将对 CoNLL 训练数据进行训练。这可能需要一段时间，所以请耐心等待。训练的输出结果将是：

    ```py
    Training HMM Chunker on data from: data/ner/data/esp.train
    Output written to : models/Conll2002_ESP.RescoringChunker
    Enter text, . to quit:

    ```

1.  一旦提示输入文本，输入来自 CoNLL 测试集的西班牙语文本：

    ```py
    La empresa también tiene participación en Tele Leste Celular , operadora móvil de los estados de Bahía y Sergipe y que es controlada por la española Iberdrola , y además es socia de Portugal Telecom en Telesp Celular , la operadora móvil de Sao Paulo .
    Rank   Conf      Span         Type     Phrase
    0      1.0000   (105, 112)    LOC      Sergipe
    1      1.0000   (149, 158)    ORG      Iberdrola
    2      1.0000   (202, 216)    ORG      Telesp Celular
    3      1.0000   (182, 198)    ORG      Portugal Telecom
    4      1.0000   (97, 102)     LOC      Bahía
    5      1.0000   (241, 250)    LOC      Sao Paulo
    6      0.9907   (163, 169)    PER      además
    7      0.9736   (11, 18)      ORG      también
    8      0.9736   (39, 60)      ORG      en Tele Leste Celular
    9      0.0264   (42, 60)      ORG      Tele Leste Celular

    ```

1.  我们将看到一系列实体、它们的置信度分数、原始句子中的跨度、实体的类型和表示该实体的短语。

1.  要找出正确的标签，请查看标注过的`esp.testa`文件，该文件包含了以下标签：

    ```py
    Tele B-ORG
    Leste I-ORG
    Celular I-ORG
    Bahía B-LOC
    Sergipe B-LOC
    Iberdrola B-ORG
    Portugal B-ORG
    Telecom I-ORG
    Telesp B-ORG
    Celular I-ORG
    Sao B-LOC
    Paulo I-LOC

    ```

1.  这可以这样理解：

    ```py
    Tele Leste Celular      ORG
    Bahía                   LOC
    Sergipe                 LOC
    Iberdrola               ORG
    Portugal Telecom        ORG
    Telesp Celular          ORG
    Sao Paulo               LOC

    ```

1.  所以，我们把所有置信度为 1.000 的实体识别正确，其他的都识别错了。这有助于我们在生产环境中设置阈值。

## 它是如何工作的……

`CharLmRescoringChunker`提供了一个基于长距离字符语言模型的 chunker，通过重新评分包含的字符语言模型 HMM chunker 的输出结果来运行。底层的 chunker 是`CharLmHmmChunker`的一个实例，它根据构造函数中指定的分词器工厂、n-gram 长度、字符数和插值比率进行配置。

让我们从`main()`方法开始；在这里，我们将设置 chunker，如果模型不存在则进行训练，然后允许输入以提取命名实体：

```py
String modelFilename = "models/Conll2002_ESP.RescoringChunker";
String trainFilename = "data/ner/data/esp.train";
```

如果你在数据目录中解压了 CoNLL 数据（`tar –xvzf ner.tgz`），训练文件将位于正确的位置。记得修正`esp.train`文件第 221619 行的标注。如果你使用其他数据，请修改并重新编译类。

接下来的代码段会训练模型（如果模型不存在），然后加载序列化版本的分块器。如果你对反序列化有疑问，请参见第一章中的*反序列化和运行分类器*部分，了解更多内容。以下是代码片段：

```py
File modelFile = new File(modelFilename);
if(!modelFile.exists()){
  System.out.println("Training HMM Chunker on data from: " + trainFilename);
  trainHMMChunker(modelFilename, trainFilename);
  System.out.println("Output written to : " + modelFilename);
}

@SuppressWarnings("unchecked")
RescoringChunker<CharLmRescoringChunker> chunker = (RescoringChunker<CharLmRescoringChunker>) AbstractExternalizable.readObject(modelFile);
```

`trainHMMChunker()`方法首先进行一些`File`文件管理，然后设置`CharLmRescoringChunker`的配置参数：

```py
static void trainHMMChunker(String modelFilename, String trainFilename) throws IOException{
  File modelFile = new File(modelFilename);
  File trainFile = new File(trainFilename);

  int numChunkingsRescored = 64;
  int maxNgram = 12;
  int numChars = 256;
  double lmInterpolation = maxNgram; 
  TokenizerFactory factory
    = IndoEuropeanTokenizerFactory.INSTANCE;

CharLmRescoringChunker chunkerEstimator
  = new CharLmRescoringChunker(factory,numChunkingsRescored,
          maxNgram,numChars,
          lmInterpolation);
```

从第一个参数开始，`numChunkingsRescored`设置来自嵌入式`Chunker`的分块数量，这些分块将重新评分以提高性能。此重新评分的实现可能有所不同，但通常会使用更少的局部信息来改进基本的 HMM 输出，因为它在上下文上有限。`maxNgram`设置每种分块类型的最大字符数，用于重新评分的字符语言模型，而`lmInterpolation`决定模型如何进行插值。一个好的值是字符 n-gram 的大小。最后，创建一个分词器工厂。在这个类中有很多内容，更多信息请查阅 Javadoc。

方法中的下一部分将获取一个解析器，我们将在接下来的代码片段中讨论，它接受`chunkerEstimator`和`setHandler()`方法，然后，`parser.parse()`方法进行实际训练。最后一段代码将模型序列化到磁盘——请参见第一章中的*如何序列化 LingPipe 对象—分类器示例*部分，了解其中发生的情况：

```py
Conll2002ChunkTagParser parser = new Conll2002ChunkTagParser();
parser.setHandler(chunkerEstimator);
parser.parse(trainFile);
AbstractExternalizable.compileTo(chunkerEstimator,modelFile);
```

现在，让我们来看看如何解析 CoNLL 数据。此类的源代码是`src/com/lingpipe/cookbook/chapter5/Conll2002ChunkTagParser`：

```py
public class Conll2002ChunkTagParser extends StringParser<ObjectHandler<Chunking>>
{

  static final String TOKEN_TAG_LINE_REGEX = "(\\S+)\\s(\\S+\\s)?(O|[B|I]-\\S+)";
  static final int TOKEN_GROUP = 1;
  static final int TAG_GROUP = 3;
  static final String IGNORE_LINE_REGEX = "-DOCSTART(.*)";
  static final String EOS_REGEX = "\\A\\Z";
  static final String BEGIN_TAG_PREFIX = "B-";
  static final String IN_TAG_PREFIX = "I-";
  static final String OUT_TAG = "O";
```

静态方法设置`com.aliasi.tag.LineTaggingParser` LingPipe 类的配置。像许多可用的数据集一样，CoNLL 使用每行一个标记/标签的格式，这种格式非常容易解析：

```py
private final LineTaggingParser mParser = new LineTaggingParser(TOKEN_TAG_LINE_REGEX, TOKEN_GROUP, TAG_GROUP, IGNORE_LINE_REGEX, EOS_REGEX);
```

`LineTaggingParser`构造函数需要一个正则表达式，通过分组识别标记和标签字符串。此外，还有一个正则表达式用于忽略的行，最后一个正则表达式用于句子的结束。

接下来，我们设置`TagChunkCodec`；它将处理从 BIO 格式的标记令牌到正确分块的映射。关于这里发生的过程，请参见前一个食谱，*在词标记和分块之间转换—BIO 编解码器*。剩余的参数将标签自定义为与 CoNLL 训练数据的标签相匹配：

```py
private final TagChunkCodec mCodec = new BioTagChunkCodec(null, false, BEGIN_TAG_PREFIX, IN_TAG_PREFIX, OUT_TAG);
```

该类的其余部分提供`parseString()`方法，立即将其传递给`LineTaggingParser`类：

```py
public void parseString(char[] cs, int start, int end) {
  mParser.parseString(cs,start,end);
}
```

接下来，`ObjectHandler`解析器与编解码器和提供的处理器一起正确配置：

```py
public void setHandler(ObjectHandler<Chunking> handler) {

  ObjectHandler<Tagging<String>> taggingHandler = TagChunkCodecAdapters.chunkingToTagging(mCodec, handler);
  mParser.setHandler(taggingHandler);
}

public TagChunkCodec getTagChunkCodec(){
  return mCodec;
}
```

这些代码看起来很奇怪，但实际上它们的作用是设置一个解析器，从输入文件中读取行并从中提取分块。

最后，让我们回到`main`方法，看看输出循环。我们将把`MAX_NBEST`块值设置为 10，然后调用块器的`nBestChunkings`方法。这将提供前 10 个块及其概率分数。根据评估结果，我们可以选择在某个特定分数处进行截断：

```py
char[] cs = text.toCharArray();
Iterator<Chunk> it = chunker.nBestChunks(cs,0,cs.length, MAX_N_BEST_CHUNKS);
System.out.println(text);
System.out.println("Rank          Conf      Span"    + "    Type     Phrase");
DecimalFormat df = new DecimalFormat("0.0000");

for (int n = 0; it.hasNext(); ++n) {

Chunk chunk = it.next();
double conf = chunk.score();
int start = chunk.start();
int end = chunk.end();
String phrase = text.substring(start,end);
System.out.println(n + " "       + "            "   + df.format(conf)     + "       (" + start  + ", " + end  + ")    " + chunk.type()      + "         " + phrase);
}
```

## 还有更多内容……

欲了解如何运行完整评估的更多细节，请参见教程中的评估部分：[`alias-i.com/lingpipe/demos/tutorial/ne/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/ne/read-me.html)。

## 另见

有关`CharLmRescoringChunker`和`HmmChunker`的更多详情，请参见：

+   [`alias-i.com/lingpipe/docs/api/com/aliasi/chunk/AbstractCharLmRescoringChunker.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/chunk/AbstractCharLmRescoringChunker.html)

+   [`alias-i.com/lingpipe/docs/api/com/aliasi/chunk/HmmChunker.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/chunk/HmmChunker.html)

# 混合 NER 源

现在我们已经看过如何构建几种不同类型的命名实体识别（NER），接下来可以看看如何将它们组合起来。在本教程中，我们将结合正则表达式块器、基于词典的块器和基于 HMM 的块器，并将它们的输出合并，看看重叠情况。

我们将以与前几个食谱中相同的方式初始化一些块器，然后将相同的文本传递给这些块器。最简单的情况是每个块器返回唯一的输出。例如，我们考虑一个句子：“总统奥巴马原定于今晚在 G-8 会议上发表演讲”。如果我们有一个人名块器和一个组织块器，我们可能只会得到两个唯一的块。然而，如果我们再加入一个`美国总统`块器，我们将得到三个块：`PERSON`、`ORGANIZATION`和`PRESIDENT`。这个非常简单的食谱将展示一种处理这些情况的方法。

## 如何操作……

1.  使用命令行或 IDE 中的等效命令，输入以下内容：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.MultipleNer

    ```

1.  常见的交互式提示如下：

    ```py
    Enter text, . to quit:
    President Obama is scheduled to arrive in London this evening. He will address the G-8 summit.
    neChunking: [10-15:PERSON@-Infinity, 42-48:LOCATION@-Infinity, 83-86:ORGANIZATION@-Infinity]
    pChunking: [62-66:MALE_PRONOUN@1.0]
    dChunking: [10-15:PRESIDENT@1.0]
    ----Overlaps Allowed

     Combined Chunks:
    [83-86:ORGANIZATION@-Infinity, 10-15:PERSON@-Infinity, 10-15:PRESIDENT@1.0, 42-48:LOCATION@-Infinity, 62-66:MALE_PRONOUN@1.0]

    ----Overlaps Not Allowed

     Unique Chunks:
    [83-86:ORGANIZATION@-Infinity, 42-48:LOCATION@-Infinity, 62-66:MALE_PRONOUN@1.0]

     OverLapped Chunks:
    [10-15:PERSON@-Infinity, 10-15:PRESIDENT@1.0]

    ```

1.  我们看到来自三个块器的输出：`neChunking`是经过训练返回 MUC-6 实体的 HMM 块器的输出，`pChunking`是一个简单的正则表达式，用于识别男性代词，`dChunking`是一个词典块器，用于识别美国总统。

1.  如果允许重叠，我们将在合并的输出中看到`PRESIDENT`和`PERSON`的块。

1.  如果不允许重叠，它们将被添加到重叠块集合中，并从唯一块中移除。

## 它是如何工作的……

我们初始化了三个块器，这些块器应该是您从本章之前的食谱中熟悉的：

```py
Chunker pronounChunker = new RegExChunker(" He | he | Him | him", "MALE_PRONOUN",1.0);
File MODEL_FILE = new File("models/ne-en-news.muc6." + "AbstractCharLmRescoringChunker");
Chunker neChunker = (Chunker) AbstractExternalizable.readObject(MODEL_FILE);

MapDictionary<String> dictionary = new MapDictionary<String>();
dictionary.addEntry(
  new DictionaryEntry<String>("Obama","PRESIDENT",CHUNK_SCORE));
dictionary.addEntry(
  new DictionaryEntry<String>("Bush","PRESIDENT",CHUNK_SCORE));
ExactDictionaryChunker dictionaryChunker = new ExactDictionaryChunker(dictionary, IndoEuropeanTokenizerFactory.INSTANCE);
```

现在，我们将通过所有三个块器对输入文本进行分块，将块合并为一个集合，并将`getCombinedChunks`方法传递给它：

```py
Set<Chunk> neChunking = neChunker.chunk(text).chunkSet();
Set<Chunk> pChunking = pronounChunker.chunk(text).chunkSet();
Set<Chunk> dChunking = dictionaryChunker.chunk(text).chunkSet();
Set<Chunk> allChunks = new HashSet<Chunk>();
allChunks.addAll(neChunking);
allChunks.addAll(pChunking);
allChunks.addAll(dChunking);
getCombinedChunks(allChunks,true);//allow overlaps
getCombinedChunks(allChunks,false);//no overlaps
```

这个食谱的核心在于`getCombinedChunks`方法。我们将遍历所有的块，检查每一对是否在开始和结束时有重叠。如果它们有重叠且不允许重叠，就将它们添加到重叠集；否则，添加到合并集：

```py
static void getCombinedChunks(Set<Chunk> chunkSet, boolean allowOverlap){
  Set<Chunk> combinedChunks = new HashSet<Chunk>();
  Set<Chunk>overLappedChunks = new HashSet<Chunk>();
  for(Chunk c : chunkSet){
    combinedChunks.add(c);
    for(Chunk x : chunkSet){
      if (c.equals(x)){
        continue;
      }
      if (ChunkingImpl.overlap(c,x)) {
        if (allowOverlap){
          combinedChunks.add(x);
        } else {
          overLappedChunks.add(x);
          combinedChunks.remove(c);
        }
      }
    }
  }
}
```

这是添加更多重叠块规则的地方。例如，你可以基于分数进行评分，如果`PRESIDENT`块类型的分数高于基于 HMM 的块类型，你可以选择它。

# 用于分块的 CRF

CRF 最著名的是在命名实体标注方面提供接近最先进的性能。本食谱将告诉我们如何构建这样的系统。该食谱假设你已经阅读、理解并尝试过*条件随机场 – 用于词汇/标记标注*的第四章，该章节涉及了基础技术。与 HMM 类似，CRF 将命名实体识别视为一个词汇标注问题，具有一个解释层，提供分块信息。与 HMM 不同，CRF 使用基于逻辑回归的分类方法，这使得可以包含随机特征。此外，本食谱遵循了一个优秀的 CRF 教程（但省略了细节），教程地址是[`alias-i.com/lingpipe/demos/tutorial/crf/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/crf/read-me.html)。Javadoc 中也有很多信息。

## 准备工作

就像我们之前做的那样，我们将使用一个小型手动编码的语料库作为训练数据。该语料库位于`src/com/lingpipe/cookbook/chapter5/TinyEntityCorpus.java`，开始于：

```py
public class TinyEntityCorpus extends Corpus<ObjectHandler<Chunking>> {

  public void visitTrain(ObjectHandler<Chunking> handler) {
    for (Chunking chunking : CHUNKINGS) handler.handle(chunking);
  }

  public void visitTest(ObjectHandler<Chunking> handler) {
    /* no op */
  }
```

由于我们仅使用此语料库进行训练，`visitTest()`方法没有任何作用。然而，`visitTrain()`方法将处理程序暴露给`CHUNKINGS`常量中存储的所有分块。这看起来像以下内容：

```py
static final Chunking[] CHUNKINGS = new Chunking[] {
  chunking(""), chunking("The"), chunking("John ran.", chunk(0,4,"PER")), chunking("Mary ran.", chunk(0,4,"PER")), chunking("The kid ran."), chunking("John likes Mary.", chunk(0,4,"PER"), chunk(11,15,"PER")), chunking("Tim lives in Washington", chunk(0,3,"PER"), chunk(13,23,"LOC")), chunking("Mary Smith is in New York City", chunk(0,10,"PER"), chunk(17,30,"LOC")), chunking("New York City is fun", chunk(0,13,"LOC")), chunking("Chicago is not like Washington", chunk(0,7,"LOC"), chunk(20,30,"LOC"))
};
```

我们还没有完成。由于`Chunking`的创建相对冗长，存在静态方法来帮助动态创建所需的对象：

```py
static Chunking chunking(String s, Chunk... chunks) {
  ChunkingImpl chunking = new ChunkingImpl(s);
  for (Chunk chunk : chunks) chunking.add(chunk);
  return chunking;
}

static Chunk chunk(int start, int end, String type) {
  return ChunkFactory.createChunk(start,end,type);
}
```

这就是所有的设置；接下来，我们将使用前面的数据训练并运行 CRF。

## 如何操作...

1.  在命令行中键入`TrainAndRunSimplCrf`类，或者在你的 IDE 中运行相应的命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.TrainAndRunSimpleCrf

    ```

1.  这会导致大量的屏幕输出，报告 CRF 的健康状态和进展，主要是来自底层的逻辑回归分类器，它驱动了整个过程。最有趣的部分是我们将收到一个邀请，去体验新的 CRF：

    ```py
    Enter text followed by new line
    >John Smith went to New York.

    ```

1.  分块器报告了第一个最佳输出：

    ```py
    FIRST BEST
    John Smith went to New York. : [0-10:PER@-Infinity, 19-27:LOC@-Infinity]

    ```

1.  前述输出是 CRF 的最优分析，展示了句子中有哪些实体。它认为`John Smith`是`PER`，其输出为`the 0-10:PER@-Infinity`。我们知道它通过从输入文本中提取从 0 到 10 的子字符串来应用于`John Smith`。忽略`–Infinity`，它是为没有分数的片段提供的。最优片段分析没有分数。它认为文本中的另一个实体是`New York`，其类型为`LOC`。

1.  紧接着，条件概率跟随其后：

    ```py
    10 BEST CONDITIONAL
    Rank log p(tags|tokens)  Tagging
    0    -1.66335590 [0-10:PER@-Infinity, 19-27:LOC@-Infinity]
    1    -2.38671498 [0-10:PER@-Infinity, 19-28:LOC@-Infinity]
    2    -2.77341747 [0-10:PER@-Infinity]
    3    -2.85908677 [0-4:PER@-Infinity, 19-27:LOC@-Infinity]
    4    -3.00398856 [0-10:PER@-Infinity, 19-22:LOC@-Infinity]
    5    -3.23050827 [0-10:PER@-Infinity, 16-27:LOC@-Infinity]
    6    -3.49773765 [0-10:PER@-Infinity, 23-27:PER@-Infinity]
    7    -3.58244582 [0-4:PER@-Infinity, 19-28:LOC@-Infinity]
    8    -3.72315571 [0-10:PER@-Infinity, 19-22:PER@-Infinity]
    9    -3.95386735 [0-10:PER@-Infinity, 16-28:LOC@-Infinity]

    ```

1.  前述输出提供了整个短语的 10 个最佳分析结果及其条件（自然对数）概率。在这种情况下，我们会发现系统对任何分析结果都没有特别的信心。例如，最优分析被正确的估计概率为`exp(-1.66)=0.19`。

1.  接下来，在输出中，我们看到个别片段的概率：

    ```py
    MARGINAL CHUNK PROBABILITIES
    Rank Chunk Phrase
    0 0-10:PER@-0.49306887565189683 John Smith
    1 19-27:LOC@-1.1957935770408703 New York
    2 0-4:PER@-1.3270942262839682 John
    3 19-22:LOC@-2.484463373596263 New
    4 23-27:PER@-2.6919267821139776 York
    5 16-27:LOC@-2.881057607295971 to New York
    6 11-15:PER@-3.0868632773744222 went
    7 16-18:PER@-3.1583044940140192 to
    8 19-22:PER@-3.2036305275847825 New
    9 23-27:LOC@-3.536294896211011 York

    ```

1.  与之前的条件输出一样，概率是对数，因此我们可以看到`John Smith`片段的估计概率为`exp(-0.49) = 0.61`，这很有道理，因为在训练时，CRF 看到`John`出现在`PER`的开始位置，`Smith`出现在另一个`PER`的结束位置，而不是直接看到`John Smith`。

1.  前述类型的概率分布如果有足够的资源去考虑广泛的分析范围以及结合证据的方式，以允许选择不太可能的结果，确实能改善系统。最优分析往往会过于保守，适应训练数据的外观。

## 它是如何工作的…

`src/com/lingpipe/cookbook/chapter5/TrainAndRunSimpleCRF.java`中的代码与我们的分类器和 HMM 配方类似，但有一些不同之处。这些不同之处如下所示：

```py
public static void main(String[] args) throws IOException {
  Corpus<ObjectHandler<Chunking>> corpus = new TinyEntityCorpus();

  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  boolean enforceConsistency = true;
  TagChunkCodec tagChunkCodec = new BioTagChunkCodec(tokenizerFactory, enforceConsistency);
```

当我们之前使用 CRF 时，输入数据是`Tagging<String>`类型。回顾`TinyEntityCorpus.java`，数据类型是`Chunking`类型。前述的`BioTagChunkCodec`通过提供的`TokenizerFactory`和`boolean`来帮助将`Chunking`转换为`Tagging`，如果`TokenizerFactory`与`Chunk`的开始和结束不完全匹配，则会引发异常。回顾*在词语标注和片段之间的转换–BIO 编解码器*配方，以更好理解这个类的作用。

让我们看一下以下内容：

```py
John Smith went to New York City. : [0-10:PER@-Infinity, 19-32:LOC@-Infinity]
```

这个编解码器将转化为一个标注：

```py
Tok    Tag
John   B_PER
Smith  I_PER
went  O
to     O
New    B_LOC
York  I_LOC
City  I_LOC
.    O
```

编解码器也将执行相反的操作。Javadoc 值得一看。一旦建立了这种映射，剩下的 CRF 与背后的词性标注案例是相同的，正如我们在*条件随机场 – 用于词语/标记标注*的配方中所展示的那样，参见第四章，*标注词语和标记*。考虑以下代码片段：

```py
ChainCrfFeatureExtractor<String> featureExtractor = new SimpleCrfFeatureExtractor();
```

所有的机制都隐藏在一个新的 `ChainCrfChunker` 类中，它的初始化方式类似于逻辑回归，这是其底层技术。如需了解更多配置信息，请参阅 第三章中的 *逻辑回归* 示例，*高级分类器*：

```py
int minFeatureCount = 1;
boolean cacheFeatures = true;
boolean addIntercept = true;
double priorVariance = 4.0;
boolean uninformativeIntercept = true;
RegressionPrior prior = RegressionPrior.gaussian(priorVariance, uninformativeIntercept);
int priorBlockSize = 3;
double initialLearningRate = 0.05;
double learningRateDecay = 0.995;
AnnealingSchedule annealingSchedule = AnnealingSchedule.exponential(initialLearningRate, learningRateDecay);
double minImprovement = 0.00001;
int minEpochs = 10;
int maxEpochs = 5000;
Reporter reporter = Reporters.stdOut().setLevel(LogLevel.DEBUG);
System.out.println("\nEstimating");
ChainCrfChunker crfChunker = ChainCrfChunker.estimate(corpus, tagChunkCodec, tokenizerFactory, featureExtractor, addIntercept, minFeatureCount, cacheFeatures, prior, priorBlockSize, annealingSchedule, minImprovement, minEpochs, maxEpochs, reporter);
```

这里唯一的新内容是我们刚刚描述的 `tagChunkCodec` 参数。

一旦训练完成，我们将通过以下代码访问分块器的最佳结果：

```py
System.out.println("\nFIRST BEST");
Chunking chunking = crfChunker.chunk(evalText);
System.out.println(chunking);
```

条件分块由以下内容提供：

```py
int maxNBest = 10;
System.out.println("\n" + maxNBest + " BEST CONDITIONAL");
System.out.println("Rank log p(tags|tokens)  Tagging");
Iterator<ScoredObject<Chunking>> it = crfChunker.nBestConditional(evalTextChars,0, evalTextChars.length,maxNBest);

  for (int rank = 0; rank < maxNBest && it.hasNext(); ++rank) {
    ScoredObject<Chunking> scoredChunking = it.next();
    System.out.println(rank + "    " + scoredChunking.score() + " " + scoredChunking.getObject().chunkSet());
  }
```

可以通过以下方式访问各个块：

```py
System.out.println("\nMARGINAL CHUNK PROBABILITIES");
System.out.println("Rank Chunk Phrase");
int maxNBestChunks = 10;
Iterator<Chunk> nBestIt  = crfChunker.nBestChunks(evalTextChars,0, evalTextChars.length,maxNBestChunks);
for (int n = 0; n < maxNBestChunks && nBestIt.hasNext(); ++n) {
  Chunk chunk = nBestChunkIt.next();
  System.out.println(n + " " + chunk + " " + evalText.substring(chunk.start(),chunk.end()));
}
```

就这些。你已经访问了世界上最优秀的分块技术之一。接下来，我们将向你展示如何改进它。

# 使用更好特征的 CRFs 进行命名实体识别（NER）

在这个示例中，我们将展示如何为 CRF 创建一个逼真的、尽管不是最先进的、特征集。这些特征将包括标准化的标记、词性标签、词形特征、位置特征以及标记的前后缀。将其替换为 *CRFs for chunking* 示例中的 `SimpleCrfFeatureExtractor` 进行使用。

## 如何做到……

该示例的源代码位于 `src/com/lingpipe/cookbook/chapter5/FancyCrfFeatureExtractor.java`：

1.  打开你的 IDE 或命令提示符，输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.FancyCrfFeatureExtractor

    ```

1.  准备好迎接控制台中爆炸性的特征输出。用于特征提取的数据是上一个示例中的 `TinyEntityCorpus`。幸运的是，第一部分数据仅仅是句子 `John ran.` 中 "John" 的节点特征：

    ```py
    Tagging:  John/PN
    Node Feats:{PREF_NEXT_ra=1.0, PREF_Jo=1.0, POS_np=1.0, TOK_CAT_LET-CAP=1.0, SUFF_NEXT_an=1.0, PREF_Joh=1.0, PREF_NEXT_r=1.0, SUFF_John=1.0, TOK_John=1.0, PREF_NEXT_ran=1.0, BOS=1.0, TOK_NEXT_ran=1.0, SUFF_NEXT_n=1.0, SUFF_NEXT_ran=1.0, SUFF_ohn=1.0, PREF_J=1.0, POS_NEXT_vbd=1.0, SUFF_hn=1.0, SUFF_n=1.0, TOK_CAT_NEXT_ran=1.0, PREF_John=1.0}

    ```

1.  序列中的下一个词汇添加了边缘特征——我们不会展示节点特征：

    ```py
    Edge Feats:{PREV_TAG_TOKEN_CAT_PN_LET-CAP=1.0, PREV_TAG_PN=1.0}

    ```

## 它是如何工作的……

与其他示例一样，我们不会讨论那些与之前示例非常相似的部分——这里相关的前一个示例是 第四章中的 *Modifying CRFs*，*标记单词和标记*。这完全相同，唯一不同的是我们将添加更多特征——可能来自意想不到的来源。

### 注意

CRFs 的教程涵盖了如何序列化/反序列化这个类。该实现并未覆盖这部分内容。

对象构造方式类似于 第四章中的 `Modifying CRFs` 示例，*标记单词和标记*：

```py
public FancyCrfFeatureExtractor()
  throws ClassNotFoundException, IOException {
  File posHmmFile = new File("models/pos-en-general" + "brown.HiddenMarkovModel");
  @SuppressWarnings("unchecked") HiddenMarkovModel posHmm = (HiddenMarkovModel)
  AbstractExternalizable.readObject(posHmmFile);

  FastCache<String,double[]> emissionCache = new FastCache<String,double[]>(100000);
  mPosTagger = new HmmDecoder(posHmm,null,emissionCache);
}
```

构造函数设置了一个带有缓存的词性标注器，并将其传递给 `mPosTagger` 成员变量。

以下方法几乎不做任何事，只是提供了一个内部的 `ChunkerFeatures` 类：

```py
public ChainCrfFeatures<String> extract(List<String> tokens, List<String> tags) {
  return new ChunkerFeatures(tokens,tags);
}
```

`ChunkerFeatures` 类是更有趣的部分：

```py
class ChunkerFeatures extends ChainCrfFeatures<String> {
  private final Tagging<String> mPosTagging;

  public ChunkerFeatures(List<String> tokens, List<String> tags) {
    super(tokens,tags);
    mPosTagging = mPosTagger.tag(tokens);
  }
```

`mPosTagger` 函数用于为类创建时呈现的标记设置 `Tagging<String>`。这将与 `tag()` 和 `token()` 超类方法对齐，并作为节点特征的来源提供词性标签。

现在，我们可以开始特征提取了。我们将从边缘特征开始，因为它们是最简单的：

```py
public Map<String,? extends Number> edgeFeatures(int n, int k) {
  ObjectToDoubleMap<String> feats = new ObjectToDoubleMap<String>();
  feats.set("PREV_TAG_" + tag(k),1.0);
  feats.set("PREV_TAG_TOKEN_CAT_"  + tag(k) + "_" + tokenCat(n-1), 1.0);
  return feats;
}
```

新的特征以 `PREV_TAG_TOKEN_CAT_` 为前缀，示例如 `PREV_TAG_TOKEN_CAT_PN_LET-CAP=1.0`。`tokenCat()` 方法查看前一个标记的单词形状特征，并将其作为字符串返回。查看 `IndoEuropeanTokenCategorizer` 的 Javadoc 以了解其具体内容。

接下来是节点特征。这里有许多特征；每个特征将依次呈现：

```py
public Map<String,? extends Number> nodeFeatures(int n) {
  ObjectToDoubleMap<String> feats = new ObjectToDoubleMap<String>();
```

前面的代码设置了带有适当返回类型的方法。接下来的两行设置了一些状态，以便知道特征提取器在字符串中的位置：

```py
boolean bos = n == 0;
boolean eos = (n + 1) >= numTokens();
```

接下来，我们将计算当前、前一个和下一个位置的标记类别、标记和词性标注：

```py
String tokenCat = tokenCat(n);
String prevTokenCat = bos ? null : tokenCat(n-1);
String nextTokenCat = eos ? null : tokenCat(n+1);

String token = normedToken(n);
String prevToken = bos ? null : normedToken(n-1);
String nextToken = eos ? null : normedToken(n+1);

String posTag = mPosTagging.tag(n);
String prevPosTag = bos ? null : mPosTagging.tag(n-1);
String nextPosTag = eos ? null : mPosTagging.tag(n+1);
```

上一个和下一个方法检查我们是否处于句子的开始或结束，并相应地返回`null`。词性标注来自构造函数中计算并保存的词性标注。

标记方法提供了一些标记规范化，将所有数字压缩为相同类型的值。此方法如下：

```py
public String normedToken(int n) {
  return token(n).replaceAll("\\d+","*$0*").replaceAll("\\d","D");
}
```

这只是将每个数字序列替换为`*D...D*`。例如，`12/3/08`被转换为`*DD*/*D*/*DD*`。

然后，我们将为前一个、当前和后一个标记设置特征值。首先，一个标志表示它是否开始或结束一个句子或内部节点：

```py
if (bos) {
  feats.set("BOS",1.0);
}
if (eos) {
  feats.set("EOS",1.0);
}
if (!bos && !eos) {
  feats.set("!BOS!EOS",1.0);
}
```

接下来，我们将包括标记、标记类别及其词性：

```py
feats.set("TOK_" + token, 1.0);
if (!bos) {
  feats.set("TOK_PREV_" + prevToken,1.0);
}
if (!eos) {
  feats.set("TOK_NEXT_" + nextToken,1.0);
}
feats.set("TOK_CAT_" + tokenCat, 1.0);
if (!bos) {
  feats.set("TOK_CAT_PREV_" + prevTokenCat, 1.0);
}
if (!eos) {
  feats.set("TOK_CAT_NEXT_" + nextToken, 1.0);
}
feats.set("POS_" + posTag,1.0);
if (!bos) {
  feats.set("POS_PREV_" + prevPosTag,1.0);
}
if (!eos) {
  feats.set("POS_NEXT_" + nextPosTag,1.0);
}
```

最后，我们将添加前缀和后缀特征，这些特征为每个后缀和前缀（最多指定长度）添加特征：

```py
for (String suffix : suffixes(token)) {
  feats.set("SUFF_" + suffix,1.0);
}
if (!bos) {
  for (String suffix : suffixes(prevToken)) {
    feats.set("SUFF_PREV_" + suffix,1.0);
    if (!eos) {
      for (String suffix : suffixes(nextToken)) {
        feats.set("SUFF_NEXT_" + suffix,1.0);
      }
      for (String prefix : prefixes(token)) {
        feats.set("PREF_" + prefix,1.0);
      }
      if (!bos) {
        for (String prefix : prefixes(prevToken)) {
          feats.set("PREF_PREV_" + prefix,1.0);
      }
      if (!eos) {
        for (String prefix : prefixes(nextToken)) {
          feats.set("PREF_NEXT_" + prefix,1.0);
        }
      }
      return feats;
    }
```

之后，我们将返回生成的特征映射。

`prefix` 或 `suffix` 函数简单地用一个列表实现：

```py
static int MAX_PREFIX_LENGTH = 4;
  static List<String> prefixes(String s) {
    int numPrefixes = Math.min(MAX_PREFIX_LENGTH,s.length());
    if (numPrefixes == 0) {
      return Collections.emptyList();
    }
    if (numPrefixes == 1) {
      return Collections.singletonList(s);
    }
    List<String> result = new ArrayList<String>(numPrefixes);
    for (int i = 1; i <= Math.min(MAX_PREFIX_LENGTH,s.length()); ++i) {
      result.add(s.substring(0,i));
    }
    return result;
  }

  static int MAX_SUFFIX_LENGTH = 4;
  static List<String> suffixes(String s) {
    int numSuffixes = Math.min(s.length(), MAX_SUFFIX_LENGTH);
    if (numSuffixes <= 0) {
      return Collections.emptyList();
    }
    if (numSuffixes == 1) {
      return Collections.singletonList(s);
    }
    List<String> result = new ArrayList<String>(numSuffixes);
    for (int i = s.length() - numSuffixes; i < s.length(); ++i) {
      result.add(s.substring(i));
    }
    return result;
  }
```

这是一个很好的特征集，适合你的命名实体检测器。
