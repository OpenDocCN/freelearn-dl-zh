# 第五章：第5章：在文本中查找跨度 – 块

本章涵盖了以下食谱：

+   句子检测

+   句子检测评估

+   调整句子检测

+   在字符串中标记嵌入的块——句子块示例

+   段落检测

+   简单的名词短语和动词短语

+   基于正则表达式的命名实体识别块

+   基于词典的命名实体识别块

+   在词性标注和块之间进行翻译 – BIO 编码器

+   基于HMM的命名实体识别

+   混合命名实体识别源

+   块的CRFs

+   使用CRFs和更好特征的命名实体识别

# 简介

本章将告诉我们如何处理通常覆盖一个或多个单词/标记的文本跨度。LingPipe API将这个文本单元表示为块，并使用相应的块生成器来生成块。以下是一些带有字符偏移的文本：

```py
LingPipe is an API. It is written in Java.
012345678901234567890123456789012345678901
          1         2         3         4           
```

将前面的文本块成句子将给出以下输出：

```py
Sentence start=0, end=18
Sentence start =20, end=41
```

为命名实体添加块，为LingPipe和Java添加实体：

```py
Organization start=0, end=7
Organization start=37, end=40
```

我们可以根据它们与包含它们的句子的偏移量来定义命名实体块；这不会对LingPipe产生影响，但对Java来说会是这样：

```py
Organization start=17, end=20
```

这是块的基本思想。有很多方法可以创建它们。

# 句子检测

书面文本中的句子大致对应于口语表达。它们是工业应用中处理单词的标准单元。在几乎所有成熟的NLP应用中，句子检测都是处理流程的一部分，即使在推文中，推文可以有超过140个字符的多个句子。

## 如何做...

1.  如往常一样，我们首先会玩一些数据。在控制台中输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.SentenceDetection

    ```

1.  程序将为您的句子检测实验提供提示。一个新行/回车符终止要分析的文本：

    ```py
    Enter text followed by new line
    >A sentence. Another sentence.
    SENTENCE 1:
    A sentence.
    SENTENCE 2:
    Another sentence.

    ```

1.  值得尝试不同的输入。以下是一些探索句子检测器特性的示例。删除句子的首字母大写；这将防止检测第二个句子：

    ```py
    >A sentence. another sentence.
    SENTENCE 1:
    A sentence. another sentence.

    ```

1.  检测器不需要句尾的句号——这是可配置的：

    ```py
    >A sentence. Another sentence without a final period
    SENTENCE 1:A sentence.
    SENTENCE 2:Another sentence without a final period

    ```

1.  检测器平衡括号，这不会允许句子在括号内断开——这也是可配置的：

    ```py
    >(A sentence. Another sentence.)
    SENTENCE 1: (A sentence. Another sentence.)

    ```

## 它是如何工作的...

这个句子检测器是基于启发式或基于规则的句子检测器。统计句子检测也是一个合理的方法。我们将运行整个源代码来运行检测器，稍后我们将讨论修改：

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

从`main`类的顶部开始工作，布尔`endSent`参数控制是否假设检测到的句子以句子结束，无论什么情况——这意味着最后一个字符总是句子边界——它不需要是句号或其他典型的句子结束标记。更改它并尝试一个不带句尾句号的句子，结果将是没有检测到句子。

下一个布尔变量`parenS`的声明在查找句子时优先考虑括号而不是句子生成器。接下来，将设置实际的句子分块器：

```py
TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
Chunker sentenceChunker = new SentenceChunker(tokFactory,sentenceModel);
```

`tokFactory`应该对你很熟悉，来自[第2章](part0027_split_000.html#page "第2章. 寻找和使用单词")，*寻找和使用单词*。然后可以构建`sentenceChunker`。以下是为命令行交互的标准I/O代码：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.print("Enter text followed by new line\n>");
  String text = reader.readLine();
```

一旦我们有了文本，然后应用句子检测器：

```py
Chunking chunking = sentenceChunker.chunk(text);
Set<Chunk> sentences = chunking.chunkSet();
```

分块提供了`Set<Chunk>`参数，它将非合同地提供`Chunks`的适当顺序；它们将按照`ChunkingImpl` Javadoc中的方式添加。真正偏执的程序员可能会强制执行正确的排序顺序，我们将在本章后面讨论，当我们必须处理重叠的块时。

接下来，我们将检查是否找到了任何句子，如果没有找到，我们将向控制台报告：

```py
if (sentences.size() < 1) {
  System.out.println("No sentence chunks found.");
  return;
}
```

以下是在书中首次接触`Chunker`接口，有一些评论是必要的。`Chunker`接口生成`Chunk`对象，这些对象是`CharSequence`（通常是`String`）上类型化和评分的连续字符序列。`Chunks`可以重叠。`Chunk`对象存储在`Chunking`中：

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

首先，我们恢复了基于块的基础文本字符串`textStored`。它与`text`相同，但我们想展示`Chunking`类中这种可能有用的方法，该方法可能在分块远离它所使用的`CharSequence`的递归或其他上下文中出现。

剩余的`for`循环遍历句子，并使用`String`的`substring()`方法将它们打印出来。

## 还有更多...

在继续介绍如何自己实现句子检测器之前，值得提一下，LingPipe有`MedlineSentenceModel`，它面向医学研究文献中发现的句子类型。它已经看到了大量数据，应该成为你在这些类型数据上句子检测工作的起点。

### 嵌套句子

句子，尤其是在文学中，可以包含嵌套的句子。考虑以下例子：

```py
John said "this is a nested sentence" and then shut up.
```

前面的句子将被正确标记为：

```py
[John said "[this is a nested sentence]" and then shut up.]
```

这种嵌套与语言学家对嵌套句子的概念不同，后者基于语法角色。考虑以下例子：

```py
[[John ate the gorilla] and [Mary ate the burger]].
```

这个句子由两个通过`and`连接的语言学上完整的句子组成。这两个句子的区别在于，前者由标点符号决定，而后者由语法功能决定。这种区别是否重要可以讨论。然而，前者在程序上更容易识别。

然而，我们在工业环境中很少需要建模嵌套句子，但我们在MUC-6系统和研究环境中的各种核心ference解析系统中承担了这项任务。这超出了食谱书的范围，但请注意这个问题。LingPipe没有现成的嵌套句子检测功能。

# 句子检测评估

就像我们做的许多事情一样，我们希望能够评估我们组件的性能。句子检测也不例外。句子检测是一种跨度注释，与我们之前对分类器和分词的评价不同。由于文本中可能包含不属于任何句子的字符，因此存在句子开始和句子结束的概念。一个不属于句子的字符的例子将来自HTML页面的JavaScript。

以下步骤将指导你创建评估数据并将其传递给评估类。

## 如何进行...

执行以下步骤以评估句子检测：

1.  打开一个文本编辑器，复制粘贴一些你想要用来评估句子检测的文学瑰宝，或者你可以使用我们提供的默认文本，如果你没有提供自己的数据，就会使用这个文本。如果你坚持使用纯文本，这会更容易。

1.  在文本中插入平衡的`[`和`]`来指示句子的开始和结束。如果文本已经包含`[`或`]`，请选择一个不在文本中的字符作为句子分隔符——花括号或斜杠是一个不错的选择。如果你使用不同的分隔符，你将不得不相应地修改源代码并重新创建JAR文件。代码假设使用单字符文本分隔符。以下是从《银河系漫游指南》中摘取的句子注释文本的例子——请注意，并非每个字符都在句子中；句子之间有一些空白：

    ```py
    [The Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.] [It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.]
    ```

1.  打开命令行并运行以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.EvaluateAnnotatedSentences
    TruePos: 0-83/The Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.:S
    TruePos: 84-233/It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.:S

    ```

1.  对于这些数据，代码将显示两个与用`[]`标注的句子完美匹配的句子，正如`TruePos`标签所示。

1.  一个好的练习是稍微修改一下注释以强制出现错误。我们将把第一个句子边界向前移动一个字符：

    ```py
    T[he Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.] [It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.]

    ```

1.  保存修改后的注释文件后重新运行会产生以下结果：

    ```py
    TruePos: 84-233/It says that the effect of a Pan Galactic Gargle Blaster is like having your brains smashed out by a slice of lemon wrapped round a large gold brick.:S
    FalsePos: 0-83/The Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.:S
    FalseNeg: 1-83/he Guide says that the best drink in existence is the Pan Galactic Gargle Blaster.:S

    ```

    通过更改真实标注，产生了一个假阴性，因为句子跨度被遗漏了一个字符。此外，句子检测器还创建了一个假阳性，它识别了0-83字符序列。

1.  试着玩一玩注释和各种类型的数据，以了解评估的工作方式和句子检测器的功能。

## 它是如何工作的...

类首先消化注释文本并将句子块存储在评估对象中。然后创建句子检测器，就像我们在前面的食谱中做的那样。代码最后将创建的句子检测器应用于文本，并打印结果。

### 解析注释数据

给定用`[]`注释句子边界的文本意味着必须恢复句子的正确偏移量，并且必须创建原始未注释的文本，即没有任何`[]`。跨度解析器可能有点难以编码，以下提供的是为了简单而不是效率或正确的编码技术：

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

之前的代码以适当的字符编码将整个文件读入为一个`char[]`数组。另外，请注意，对于大文件，流式方法将更节省内存。接下来，设置一个未注释字符的累加器，作为一个`StringBuilder`对象，使用`rawChars`变量。所有遇到的既不是`[`也不是`]`的字符都将追加到该对象中。剩余的代码设置句子开始和结束的计数器，这些计数器索引到未注释的字符数组中，以及一个用于注释句子段的`Set<Chunk>`累加器。

下面的`for`循环逐个字符地遍历注释字符序列：

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

第一个`if (chars[i] == '[')`测试注释中的句子开头，并将`start`变量设置为`rawChars`的长度。迭代变量`i`包括由注释添加的长度。相应的`else if (chars[i] == ']')`语句处理句子结尾的情况。请注意，这个解析器没有错误检查——这是一个非常糟糕的想法，因为如果使用文本编辑器输入，注释错误的可能性非常高。然而，这是为了保持代码尽可能简单。在后面的食谱中，我们将提供一个带有一些最小错误检查的示例。一旦找到句子结尾，就会使用`ChunkFactory.createChunk`创建一个句子块，带有偏移量和标准LingPipe句子类型`SentenceChunker.SENTENCE_CHUNK_TYPE`，这对于即将到来的评估类正常工作是必需的。

剩余的`else`语句适用于所有不是句子边界的字符，它只是将字符添加到`rawChars`累加器中。当创建`String unannotatedText`时，可以在`for`循环外部看到这个累加器的结果。现在，我们已经将句子块正确地索引到一个文本字符串中。接下来，我们将创建一个合适的`Chunking`对象：

```py
ChunkingImpl sentChunking = new ChunkingImpl(unannotatedText);
for (Chunk chunk : sentChunks) {
  sentChunking.add(chunk);
}
```

实现`ChunkingImpl`类的类（`Chunking`是一个接口）在构造时需要底层的文本，这就是为什么我们没有在先前的循环中直接填充它的原因。LingPipe通常试图使对象构造完整。如果可以在没有底层的`CharSequence`方法的情况下创建`Chunkings`，那么当调用`charSequence()`方法时将返回什么？一个空字符串会误导用户。或者，返回`null`需要被捕获并处理。最好是强制对象在构造时就有意义。

接下来，我们将看到来自先前食谱的标准句子分块器配置：

```py
boolean eosIsSentBoundary = false;
boolean balanceParens = true;
SentenceModel sentenceModel = new IndoEuropeanSentenceModel(eosIsSentBoundary, balanceParens);
TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
SentenceChunker sentenceChunker = new SentenceChunker(tokFactory,sentenceModel);
```

有趣的部分紧随其后，有一个评估器，它将`sentenceChunker`作为评估参数：

```py
SentenceEvaluator evaluator = new SentenceEvaluator(sentenceChunker);
```

接下来，`handle(sentChunking)`方法将接受我们刚刚解析到的`Chunking`文本，并在`sentChunking`提供的`CharSequence`上运行句子检测器，并设置评估：

```py
evaluator.handle(sentChunking);
```

然后，我们只需获取评估数据，逐步分析真实句子检测和系统所做之间的差异：

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

这个配方并不涵盖所有的评估方法——查看Javadoc，但它确实提供了句子检测调整器可能最需要的；这是一个列出句子检测器正确识别的内容（真阳性）、它找到但错误的句子（假阳性）以及它遗漏的句子（假阴性）。请注意，在跨度注释中，真阴性没有太多意义，因为它们将是所有不在真实句子检测中的可能跨度集合。

# 调整句子检测

大量的数据将抵制`IndoEuropeanSentenceModel`的诱惑，因此这个配方将提供一个起点来修改句子检测以适应新的句子类型。不幸的是，这是一个非常开放的系统构建领域，因此我们将关注技术而不是句子可能的格式。

## 如何做到这一点...

这个配方将遵循一个熟悉的模式：创建评估数据，设置评估，然后开始修改。我们开始吧：

1.  拿出你最喜欢的文本编辑器并标记一些数据——我们将坚持使用`[`和`]`标记方法。以下是一个违反我们标准`IndoEuropeanSentenceModel`的例子：

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

## 还有更多...

单个假阳性对应于找到的一个句子，两个假阴性是我们在这里注释的两个未找到的句子。发生了什么？句子模型遗漏了`people's.`作为句子结束。如果去掉撇号，句子就能正确检测到——这是怎么回事？

首先，让我们看看在后台运行的代码。`IndoEuropeanSentenceModel`通过配置来自`HeuristicSentenceModel`的Javadoc中几个类别的标记来扩展`HeuristicSentenceModel`：

+   **可能的停止点**：这些是允许作为句子最后一个标记的标记。这个集合通常包括句子结尾的标点符号，例如句号(.)和双引号(")。

+   **不可能的次末点**：这些可能是句子中不是次末（倒数第二个）的标记。这个集合通常由缩写或首字母缩略词组成，例如`Mr`。

+   **不可能的起始点**：这些可能是句子中不是第一个的标记。这个集合通常包括应该附加到上一个句子的标点符号，例如结束引号('')。

`IndoEuropeanSentenceModel` 不可配置，但从 Javadoc 中可以看出，所有单个字符都被视为不可能的末尾音节。单词 `people's` 被分词为 `people`、`'`、`s` 和 `.`。单个字符 `s` 是 `.` 的前缀音节，因此被阻止。如何解决这个问题？

几种选项呈现在眼前：

+   假设这种情况不会经常发生，忽略这个错误

+   通过创建一个自定义句子模型来修复

+   通过修改分词器来不分离撇号来修复

+   为接口编写一个完整的句子检测模型

第二种选项，创建一个自定义句子模型，最简单的处理方式是将 `IndoEuropeanSentenceModel` 的源代码复制到一个新类中并对其进行修改，因为相关数据结构是私有的。这样做是为了简化类的序列化——写入磁盘的配置非常少。在示例类中，有一个 `MySentenceModel.java` 文件，它与原文件的区别在于包名和导入语句的明显变化：

```py
IMPOSSIBLE_PENULTIMATES.add("R");
//IMPOSSIBLE_PENULTIMATES.add("S"); breaks on "people's."
//IMPOSSIBLE_PENULTIMATES.add("T"); breaks on "didn't."
IMPOSSIBLE_PENULTIMATES.add("U");
```

上述代码只是注释掉了可能的单字符末尾音节的情况，这些音节是单个单词字符。要看到它的工作效果，请在 `EvaluateAnnotatedSentences.java` 类中将句子模型更改为 `SentenceModel sentenceModel = new MySentenceModel();` 并重新编译和运行它。

如果你认为上述代码是找到以可能缩写结尾的句子与诸如 `[Hunter S. Thompson is a famous fellow.]` 这样的非句子情况之间合理平衡的合理方法，这将检测 `S.` 作为句子边界。

扩展 `HeuristicSentenceModel` 对于许多类型的数据都适用。Mitzi Morris 构建了 `MedlineSentenceModel.java`，它旨在与 MEDLINE 研究索引中提供的摘要很好地配合工作。

看待上述问题的一种方式是，为了句子检测的目的，不应该将缩写拆分成标记。`IndoEuropeanTokenizerFactory` 应该调整以将 "people's" 和其他缩写保留在一起。虽然最初似乎第一个解决方案稍微好一些，但它可能会违反 `IndoEuropeanSentenceModel` 是针对特定分词进行调整的事实，而在没有评估语料库的情况下，这种变化的影响是未知的。

另一种选择是编写一个完全新颖的句子检测类，该类支持 `SentenceModel` 接口。面对高度新颖的数据集，例如 Twitter 流，我们将考虑使用机器学习驱动的跨度标注技术，如 HMMs 或 CRFs，这些技术在本章的 [第 4 章](part0051_split_000.html#page "第 4 章。标记单词和标记") 和本章末尾的 “标记单词和标记” 中有所介绍。

# 在字符串中标记嵌入的块 - 句子分块示例

之前配方中显示块的方法并不适合需要修改底层字符串的应用程序。例如，情感分析器可能只想突出显示强烈积极的句子，而不标记其他句子，同时仍然显示整个文本。产生标记文本的轻微复杂性在于添加标记会改变底层字符串。这个配方提供了插入块的工作代码，通过反向添加块来实现。

## 如何做...

虽然这个配方在技术上可能并不复杂，但它有助于将跨度注释添加到文本中，而无需从头编写代码。`src/com/lingpipe/coobook/chapter5/WriteSentDetectedChunks`类包含了引用的代码：

1.  句子块是按照第一个句子检测配方创建的。以下代码将块提取为`Set<Chunk>`，然后按`Chunk.LONGEST_MATCH_ORDER_COMPARITOR`排序。在Javadoc中，比较器被定义为：

    > *根据文本位置比较两个块。如果一个块比另一个块开始晚，或者如果它以相同的位置开始但结束得更早，则该块更大。*

    此外，还有`TEXT_ORDER_COMPARITOR`，如下所示：

    ```py
    String textStored = chunking.charSequence().toString();
    Set<Chunk> chunkSet = chunking.chunkSet();
    System.out.println("size: " + chunkSet.size());
    Chunk[] chunkArray = chunkSet.toArray(new Chunk[0]);
    Arrays.sort(chunkArray,Chunk.LONGEST_MATCH_ORDER_COMPARATOR);
    ```

1.  接下来，我们将以相反的顺序遍历块，这样可以消除为`StringBuilder`对象的改变长度保持偏移变量的需要。偏移变量是常见的错误来源，所以这个配方尽可能地避免了它们，但进行了非标准的反向循环迭代，这可能会更糟：

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

1.  上述代码通过查找句子中的字符串`like`来进行非常简单的情感分析，并在`true`时标记该句子。请注意，此代码无法处理重叠块或嵌套块。它假设一个单一的非重叠块集。以下是一些示例输出：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.WriteSentDetectedChunks
    Enter text followed by new line
    >People like to ski. But sometimes it is terrifying. 
    size: 2
    {People like to ski.} But sometimes it is terrifying. 

    ```

1.  要打印嵌套块，请查看下面的*段落* *检测*配方。

# 段落检测

句子集的典型包含结构是段落。它可以在HTML中的`<p>`等标记语言中显式设置，或者通过两个或更多的新行，这是段落通常的渲染方式。我们处于NLP的这样一个部分，没有硬性规则适用，所以我们对此表示歉意。我们将在本章中处理一些常见示例，并将其留给你自己去推广。

## 如何做...

我们从未为段落检测设置过评估工具，但它可以通过与句子检测类似的方式进行。这个配方将展示一个简单的段落检测程序，它做了一件非常重要的事情——使用嵌入的句子检测来维护对原始文档的偏移量。如果你需要以对句子或其他文档子跨度（如命名实体）敏感的方式标记文档，这种对细节的关注将对你大有裨益。考虑以下示例：

```py
Sentence 1\. Sentence 2
Sentence 3\. Sentence 4.
```

它被转换成以下形式：

```py
{[Sentence 1.] [Sentence 2]}

{[Sentence 3.] [Sentence 4.]
}
```

```py
[]designates sentences, and {} designates paragraphs. We will jump right into the code on this recipe from src/com/lingpipe/cookbook/chapter5/ParagraphSentenceDetection.java:
```

1.  示例代码在段落检测技术方面提供很少的内容。这是一个开放性问题，你将不得不运用你的技巧来解决它。我们的段落检测器是一个可怜的 `split("\n\n")`，在更复杂的方法中，将考虑上下文、字符和其他对我们来说过于特殊而无法涵盖的特征。以下是代码的起始部分，它将整个文档作为字符串读取并将其分割成数组。请注意，`paraSeperatorLength` 是构成段落分割的基础字符数——如果分割的长度变化，那么这个长度将必须与相应的段落相关联：

    ```py
    public static void main(String[] args) throws IOException {
      String document = Files.readFromFile(new File(args[0]), Strings.UTF8);
      String[] paragraphs = document.split("\n\n");
      int paraSeparatorLength = 2;
    ```

1.  菜单的真正目的是帮助处理将字符偏移量维持到原始文档中的机制，并展示嵌入式处理。这将通过保持两个独立的分块来实现：一个用于段落，一个用于句子：

    ```py
    ChunkingImpl paraChunking = new ChunkingImpl(document.toCharArray(),0,document.length());
    ChunkingImpl sentChunking = new ChunkingImpl(paraChunking.charSequence());
    ```

1.  接下来，句子检测器将以与之前菜谱中相同的方式设置：

    ```py
    boolean eosIsSentBoundary = true;
    boolean balanceParens = false;
    SentenceModel sentenceModel = new IndoEuropeanSentenceModel(eosIsSentBoundary, balanceParens);
    SentenceChunker sentenceChunker = new SentenceChunker(IndoEuropeanTokenizerFactory.INSTANCE, sentenceModel);
    ```

1.  分块遍历段落数组并为每个段落构建一个句子分块。这种方法中较为复杂的部分是，句子分块的偏移量是相对于段落字符串的，而不是整个文档。因此，代码中变量的开始和结束位置将使用文档偏移量更新。块没有调整开始和结束的方法，因此必须创建一个新的块 `adjustedSentChunk`，带有适当的段落起始偏移量，并将其添加到 `sentChunking`：

    ```py
    int paraStart = 0;
    for (String paragraph : paragraphs) {
      for (Chunk sentChunk : sentenceChunker.chunk(paragraph).chunkSet()) {
        Chunk adjustedSentChunk = ChunkFactory.createChunk(sentChunk.start() + paraStart,sentChunk.end() + paraStart, "S");
        sentChunking.add(adjustedSentChunk);
      }
    ```

1.  循环的其余部分添加段落块，然后更新段落的开始位置，该位置为段落长度加上段落分隔符的长度。这将完成在原始文档字符串中正确偏移的句子和段落的创建：

    ```py
    paraChunking.add(ChunkFactory.createChunk(paraStart, paraStart + paragraph.length(),"P"));
    paraStart += paragraph.length() + paraSeparatorLength;
    }
    ```

1.  程序的其余部分关注于打印带有一些标记的段落和句子。首先，我们将创建一个同时包含句子和段落分块的块：

    ```py
    String underlyingString = paraChunking.charSequence().toString();
    ChunkingImpl displayChunking = new ChunkingImpl(paraChunking.charSequence());
    displayChunking.addAll(sentChunking.chunkSet());
    displayChunking.addAll(paraChunking.chunkSet());
    ```

1.  接下来，`displayChunking` 将通过恢复 `chunkSet`，将其转换为块数组并应用静态比较器来排序：

    ```py
    Set<Chunk> chunkSet = displayChunking.chunkSet();
    Chunk[] chunkArray = chunkSet.toArray(new Chunk[0]);
    Arrays.sort(chunkArray, Chunk.LONGEST_MATCH_ORDER_COMPARATOR);
    ```

1.  我们将使用与我们在 *在字符串中标记嵌入式块 - 句子分块示例* 菜单中使用的相同技巧，即将标记反向插入到字符串中。我们将需要保留一个偏移计数器，因为嵌套句子将扩展完成段落标记的位置。这种方法假设没有块重叠，并且句子始终包含在段落中：

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

1.  菜单就到这里。

# 简单的名词短语和动词短语

这个菜谱将向您展示如何找到简单的**名词短语**（**NP**）和**动词短语**（**VP**）。这里的“简单”意味着短语内没有复杂结构。例如，复杂的 NP “The rain in Spain” 将被拆分为两个简单的 NP 块 “The rain” 和 “Spain”。这些短语也被称为“基础”。

这个食谱不会详细介绍如何计算基线NP/VP，而是如何使用这个类——它可能很有用，如果你想要弄清楚它是如何工作的，可以包含源代码。

## 如何做…

就像许多食谱一样，我们在这里将提供一个命令行交互式界面：

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

## 它是如何工作的…

`main()`方法首先反序列化一个词性标注器，然后创建`tokenizerFactory`：

```py
public static void main(String[] args) throws IOException, ClassNotFoundException {
  File hmmFile = new File("models/pos-en-general-brown.HiddenMarkovModel");
  HiddenMarkovModel posHmm = (HiddenMarkovModel) AbstractExternalizable.readObject(hmmFile);
  HmmDecoder posTagger  = new HmmDecoder(posHmm);
  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
```

接下来，构建`PhraseChunker`，这是一种针对该问题的启发式方法。查看源代码以了解其工作原理——它从左到右扫描输入以查找NP/VP的开始，并尝试增量地添加到短语中：

```py
PhraseChunker chunker = new PhraseChunker(posTagger,tokenizerFactory);
```

我们的标准控制台I/O代码如下：

```py
BufferedReader bufReader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.print("\n\nINPUT> ");
  String input = bufReader.readLine();
```

然后，对输入进行分词，进行词性标注，并打印出分词和标签：

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

然后计算并打印出NP/VP分块：

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

在[http://alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/posTags/read-me.html)有一个更全面的教程。

# 基于正则表达式的NER分块

**命名实体识别**（**NER**）是找到文本中特定事物提及的过程。考虑一个简单的名字；一个地点命名实体识别器可能会在以下文本中将`Ford Prefect`和`Guildford`分别识别为名字和地点提及：

```py
Ford Prefect used to live in Guildford before he needed to move.
```

我们将首先构建基于规则的NER系统，然后逐步过渡到机器学习方法。在这里，我们将探讨如何构建一个可以从文本中提取电子邮件地址的NER系统。

## 如何做…

1.  在命令提示符中输入以下命令：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.RegexNer

    ```

1.  与程序的交互过程如下：

    ```py
    Enter text, . to quit:
    >Hello,my name is Foo and my email is foo@bar.com or you can also contact me at foo.bar@gmail.com.
    input=Hello,my name is Foo and my email is foo@bar.com or you can also contact me at foo.bar@gmail.com.
    chunking=Hello,my name is Foo and my email is foo@bar.com or you can also contact me at foo.bar@gmail.com. : [37-48:email@0.0, 79-96:email@0.0]
     chunk=37-48:email@0.0  text=foo@bar.com
     chunk=79-96:email@0.0  text=foo.bar@gmail.com

    ```

1.  你可以看到，`foo@bar.com`以及`foo.bar@gmail.com`都被返回为有效的`e-mail`类型分块。此外，请注意，句子中的最后一个句点不是第二个电子邮件的一部分。

## 它是如何工作的…

正则表达式分块器找到与给定正则表达式匹配的分块。本质上，使用`java.util.regex.Matcher.find()`方法迭代地找到匹配的文本段，然后这些段被转换为Chunk对象。`RegExChunker`类封装了这些步骤。`src/com/lingpipe/cookbook/chapter5/RegExNer.java`的代码描述如下：

```py
public static void main(String[] args) throws IOException {
  String emailRegex = "[A-Za-z0-9](([_\\.\\-]?[a-zA-Z0-9]+)*)" + + "@([A-Za-z0-9]+)" + "(([\\.\\-]?[a-zA-Z0-9]+)*)\\.([A-Za-z]{2,})";
  String chunkType = "email";
  double score = 1.0;
  Chunker chunker = new RegExChunker(emailRegex,chunkType,score);
```

所有有趣的工作都在前面的代码行中完成了。`emailRegex`是从互联网上获取的——见下文以获取来源，其余部分是设置`chunkType`和`score`。

代码的其余部分读取输入并打印出分块：

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

## 参见

+   电子邮件地址匹配的正则表达式来自[regexlib.com](http://regexlib.com)，具体链接为[http://regexlib.com/DisplayPatterns.aspx?cattabindex=0&categoryId=1](http://regexlib.com/DisplayPatterns.aspx?cattabindex=0&categoryId=1)。

# 基于词典的NER分块

在许多网站、博客以及当然是在网络论坛上，你可能会看到关键字高亮显示，链接到你可以购买产品的页面。同样，新闻网站也为人物、地点和热门事件提供主题页面，例如[http://www.nytimes.com/pages/topics/](http://www.nytimes.com/pages/topics/)。

这里的许多操作都是完全自动化的，并且使用基于词典的`Chunker`很容易做到。编译实体及其类型的名称列表非常直接。精确的词典分词器根据标记化词典条目的精确匹配提取分词。

LingPipe中基于词典的分词器实现基于Aho-Corasick算法，该算法在字典中找到所有匹配项，其时间复杂度与匹配项的数量或字典的大小无关。这使得它比使用子字符串搜索或正则表达式的朴素方法要高效得多。

## 如何做到这一点...

1.  在你选择的IDE中运行`chapter5`包中的`DictionaryChunker`类，或者使用命令行输入以下内容：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.DictionaryChunker

    ```

1.  由于这个特定的分词示例偏向于《银河系漫游指南》（非常严重），让我们使用一个涉及一些角色的句子：

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

1.  注意，我们从`Heart`和`Heart of Gold`中得到了重叠的分词。正如我们将看到的，这可以配置为以不同的方式行为。

## 它是如何工作的...

基于词典的命名实体识别（NER）在针对非结构化文本数据的大量自动链接中发挥着重要作用。我们可以通过以下步骤构建一个：

代码的第一步将创建`MapDictionary<String>`来存储词典条目：

```py
static final double CHUNK_SCORE = 1.0;

public static void main(String[] args) throws IOException {
  MapDictionary<String> dictionary = new MapDictionary<String>();
  MapDictionary<String> dictionary = new MapDictionary<String>();
```

接下来，我们将使用`DictionaryEntry<String>`填充词典，它包括类型信息和用于创建分词的分数：

```py
dictionary.addEntry(new DictionaryEntry<String>("Arthur","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Ford","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Trillian","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Zaphod","PERSON",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Marvin","ROBOT",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("Heart of Gold", "SPACECRAFT",CHUNK_SCORE));
dictionary.addEntry(new DictionaryEntry<String>("HitchhikersGuide", "PRODUCT",CHUNK_SCORE));
```

在`DictionaryEntry`构造函数中，第一个参数是短语，第二个字符串参数是类型，最后一个双精度参数是分词的分数。词典条目总是区分大小写的。词典中不同实体类型的数量没有限制。分数将简单地作为基于词典的分词器中的分词分数传递。

接下来，我们将构建`Chunker`：

```py
boolean returnAllMatches = true;
boolean caseSensitive = true;
ExactDictionaryChunker dictionaryChunker = new ExactDictionaryChunker(dictionary, IndoEuropeanTokenizerFactory.INSTANCE, returnAllMatches,caseSensitive);
```

一个精确的词典分词器可能被配置为提取所有匹配的分词，通过`returnAllMatches`布尔值将结果限制为一致的非重叠分词集。查看Javadoc以了解确切的标准。还有一个`caseSensitive`布尔值。分词器需要一个分词器，因为它将标记作为符号进行匹配，并且在匹配过程中忽略空白字符。

接下来是我们的标准I/O代码，用于控制台交互：

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

剩余的代码创建分词，遍历分词并打印它们：

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

即使在基于机器学习的系统中，词典分词器也非常有用。通常总有一类实体最适合以这种方式识别。"混合NER来源"配方解决了如何处理多个命名实体来源的问题。

# 词语标记和块之间的翻译 – BIO 编码器

在第 4 章 [“标记词语和标记”](part0051_split_000.html#page "Chapter 4. Tagging Words and Tokens") 中，我们使用了 HMM 和 CRF 来对词语/标记应用标记。这个配方解决了从使用 **开始，内部，和外部** （**BIO**） 标记来编码可以跨越多个词语/标记的块分割的标记中创建块的情况。这反过来又是现代命名实体检测系统的基础。

## 准备工作

标准的 BIO 标记方案将类型为 X 的块中的第一个标记标记为 B-X（开始），而同一块中的所有后续标记标记为 I-X（内部）。所有不在块中的标记标记为 O（外部）。例如，具有字符计数的字符串：

```py
John Jones Mary and Mr. Jones
01234567890123456789012345678
0         1         2         
```

它可以标记为：

```py
John  B_PERSON
Jones  I_PERSON
Mary  B_PERSON
and  O
Mr    B_PERSON
.    I_PERSON
Jones  I_PERSON
```

对应的块将是：

```py
0-10 "John Jones" PERSON
11-15 "Mary" PERSON
20-29 "Mr. Jones" PERSON
```

## 如何做到这一点…

程序将显示标记和块分割之间的最简单映射以及相反的映射：

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

1.  接下来，打印出块分割：

    ```py
    Chunking from StringTagging
    0-8:Weather@-Infinity
    12-17:Place@-Infinity

    ```

1.  然后，从刚刚显示的块分割创建标记：

    ```py
    StringTagging from Chunking
    The/B_Weather
    rain/I_Weather
    in/O
    Spain/B_Place
    ./O

    ```

## 它是如何工作的…

代码首先手动构建 `StringTagging`——我们将看到 HMM 和 CRF 以编程方式执行相同的操作，但在这里是明确的。然后打印出创建的 `StringTagging`：

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

接下来，它将构建 `BioTagChunkCodec` 并将刚刚打印出的标记转换为块分割，然后打印出块分割：

```py
BioTagChunkCodec codec = new BioTagChunkCodec();
Chunking chunking = codec.toChunking(tagging);
System.out.println("Chunking from StringTagging");
for (Chunk chunk : chunking.chunkSet()) {
  System.out.println(chunk);
}
```

剩余的代码将反转此过程。首先，创建一个不同的 `BioTagChunkCodec`，带有 `boolean` `enforceConsistency`，如果为 `true`，则检查由提供的分词器创建的标记与块开始和结束的精确对齐。如果没有对齐，我们最终会得到一个可能无法维持的块和标记之间的关系，这取决于用例：

```py
boolean enforceConsistency = true;
BioTagChunkCodec codec2 = new BioTagChunkCodec(IndoEuropeanTokenizerFactory.INSTANCE, enforceConsistency);
StringTagging tagging2 = codec2.toStringTagging(chunking);
System.out.println("StringTagging from Chunking");
for (int i = 0; i < tagging2.size(); ++i) {
  System.out.println(tagging2.token(i) + "/" + tagging2.tag(i));
}
```

最后的 `for` 循环简单地打印出 `codec2.toStringTagging()` 方法返回的标记。

## 更多内容…

该配方通过标记和块之间映射的最简单示例来工作。`BioTagChunkCodec` 还接受 `TagLattice<String>` 对象以生成 n-best 输出，正如接下来的 HMM 和 CRF 块分割器将展示的那样。

# 基于HMM的命名实体识别

`HmmChunker` 使用 HMM 对标记化的字符序列进行块分割。实例包含模型和分词器工厂的 HMM 解码器。块分割器要求 HMM 的状态符合块分割的按标记编码。它使用分词器工厂将块分解成标记和标记的序列。请参阅第 4 章 [“隐藏马尔可夫模型 (HMM) – 词性”](part0051_split_000.html#page "Chapter 4. Tagging Words and Tokens") 中的配方，*标记词语和标记*。

我们将查看训练 `HmmChunker` 并使用它进行 `CoNLL2002` 西班牙语任务。你可以也应该使用自己的数据，但这个配方假设训练数据将以 `CoNLL2002` 格式。

训练使用 `ObjectHandler` 完成，它提供训练实例。

## 准备工作

由于我们想要训练这个分词器，我们需要使用 **计算自然语言学习**（**CoNLL**）模式标记一些数据，或者使用公开可用的数据。为了提高速度，我们将选择获取 CoNLL 2002 任务中可用的语料库。

### 注意

ConNLL 是一个年度会议，它赞助了一个烘焙比赛。在 2002 年，比赛涉及西班牙语和荷兰语的命名实体识别。

数据可以从 [http://www.cnts.ua.ac.be/conll2002/ner.tgz](http://www.cnts.ua.ac.be/conll2002/ner.tgz) 下载。

与我们之前展示的配方类似；让我们看看这些数据看起来像什么：

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

在这种编码方案中，短语 *El Abogado General del Estado* 和 *Daryl Williams* 被编码为人物，分别用标签 B-PER 和 I-PER 挑选出它们的起始和持续标记。

### 注意

数据中存在一些格式错误，在解析器能够处理它们之前必须修复。在 `data` 目录中解压 `ner.tgz` 后，您将需要转到 `data/ner/data`，解压以下文件，并按指示修改：

```py
esp.train, line 221619, change I-LOC to B-LOC
esp.testa, line 30882, change I-LOC to B-LOC
esp.testb, line 9291, change I-LOC to B-LOC

```

## 如何操作...

1.  使用命令行，键入以下内容：

    ```py
    java –cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter5.HmmNeChunker

    ```

1.  如果模型不存在，它将在 CoNLL 训练数据上运行训练。这可能需要一段时间，所以请耐心等待。训练的输出将如下：

    ```py
    Training HMM Chunker on data from: data/ner/data/esp.train
    Output written to : models/Conll2002_ESP.RescoringChunker
    Enter text, . to quit:

    ```

1.  一旦出现输入文本的提示，请输入 CoNLL 测试集中的西班牙文文本：

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

1.  我们将看到一系列实体，它们的置信度分数，原始句子中的范围，实体的类型，以及代表这个实体的短语。

1.  要找出正确的标签，请查看注释过的 `esp.testa` 文件，其中包含以下标签用于这个句子：

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

1.  这可以读作如下：

    ```py
    Tele Leste Celular      ORG
    Bahía                   LOC
    Sergipe                 LOC
    Iberdrola               ORG
    Portugal Telecom        ORG
    Telesp Celular          ORG
    Sao Paulo               LOC

    ```

1.  因此，我们得到了所有 1.000 置信度正确的那些，其余的都是错误的。这可以帮助我们在生产中设置一个阈值。

## 它是如何工作的...

`CharLmRescoringChunker` 提供了一个基于长距离字符语言模型的分词器，它通过重新评分包含的字符语言模型 HMM 分词器的输出来操作。底层分词器是 `CharLmHmmChunker` 的一个实例，它使用构造函数中指定的分词器工厂、n-gram 长度、字符数和插值比进行配置。

让我们从 `main()` 方法开始；在这里，我们将设置分词器，如果它不存在，则对其进行训练，然后允许输入一些文本以获取命名实体：

```py
String modelFilename = "models/Conll2002_ESP.RescoringChunker";
String trainFilename = "data/ner/data/esp.train";
```

如果你在数据目录中解压 CoNLL 数据（`tar –xvzf ner.tgz`），训练文件将位于正确的位置。请记住，要更正 `esp.train` 文件中第 221619 行的注释。如果你使用其他数据，那么请修改并重新编译类。

下一段代码在模型不存在时训练模型，然后加载分词器的序列化版本。如果你对反序列化有疑问，请参阅第 1 章 *反序列化和运行分类器* 的配方，*简单分类器*。考虑以下代码片段：

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

`trainHMMChunker()` 方法在设置 `CharLmRescoringChunker` 的配置参数之前，进行了一些 `File` 记录：

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

```py
chunkerEstimator with the setHandler() method, and then, the parser.parse() method does the actual training. The last bit of code serializes the model to disk—see the *How to serialize a LingPipe object – classifier example* recipe in Chapter 1, *Simple Classifiers*, to read about what is going on:
```

```py
Conll2002ChunkTagParser parser = new Conll2002ChunkTagParser();
parser.setHandler(chunkerEstimator);
parser.parse(trainFile);
AbstractExternalizable.compileTo(chunkerEstimator,modelFile);
```

现在，让我们看看如何解析 CoNLL 数据。这个类的来源是 `src/com/lingpipe/cookbook/chapter5/Conll2002ChunkTagParser`：

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

静态设置 `com.aliasi.tag.LineTaggingParser` LingPipe 类的配置。CoNLL，像许多可用的数据集一样，使用每行一个标记/标记的格式，这种格式旨在非常容易解析：

```py
private final LineTaggingParser mParser = new LineTaggingParser(TOKEN_TAG_LINE_REGEX, TOKEN_GROUP, TAG_GROUP, IGNORE_LINE_REGEX, EOS_REGEX);
```

`LineTaggingParser` 构造函数需要一个正则表达式，该正则表达式通过分组识别标记和标记字符串。此外，还有一个用于忽略行的正则表达式，最后是一个用于句子结束的正则表达式。

接下来，我们设置 `TagChunkCodec`；这将处理从 BIO 格式的标记标记到正确分块的映射。有关这里发生的事情的更多信息，请参阅之前的菜谱，*在词标记和分块之间转换 – BIO codec*。其余参数定制标记以匹配 CoNLL 训练数据：

```py
private final TagChunkCodec mCodec = new BioTagChunkCodec(null, false, BEGIN_TAG_PREFIX, IN_TAG_PREFIX, OUT_TAG);
```

类的其余部分提供了 `parseString()` 方法，该方法立即发送到 `LineTaggingParser` 类：

```py
public void parseString(char[] cs, int start, int end) {
  mParser.parseString(cs,start,end);
}
```

接下来，使用 codec 和提供的处理器正确配置了 `ObjectHandler` 解析器：

```py
public void setHandler(ObjectHandler<Chunking> handler) {

  ObjectHandler<Tagging<String>> taggingHandler = TagChunkCodecAdapters.chunkingToTagging(mCodec, handler);
  mParser.setHandler(taggingHandler);
}

public TagChunkCodec getTagChunkCodec(){
  return mCodec;
}
```

这段代码看起来很奇怪，但它所做的只是设置一个解析器来读取输入文件的行并从中提取分块。

最后，让我们回到 `main` 方法，看看输出循环。我们将设置 `MAX_NBEST` 分块值为 10，然后对分块器调用 `nBestChunkings` 方法。这提供了前 10 个分块及其概率分数。基于评估，我们可以选择在特定分数处截断：

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

## 还有更多…

关于运行完整评估的更多详细信息，请参阅教程中的评估部分，网址为 [http://alias-i.com/lingpipe/demos/tutorial/ne/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/ne/read-me.html)。

## 参见

关于 `CharLmRescoringChunker` 和 `HmmChunker` 的更多详细信息，请参阅：

+   [http://alias-i.com/lingpipe/docs/api/com/aliasi/chunk/AbstractCharLmRescoringChunker.html](http://alias-i.com/lingpipe/docs/api/com/aliasi/chunk/AbstractCharLmRescoringChunker.html)

+   [http://alias-i.com/lingpipe/docs/api/com/aliasi/chunk/HmmChunker.html](http://alias-i.com/lingpipe/docs/api/com/aliasi/chunk/HmmChunker.html)

# 混合 NER 源

现在我们已经看到了如何构建几种不同类型的 NER，我们可以看看如何将它们组合起来。在这个菜谱中，我们将使用正则表达式分块器、基于字典的分块器和基于 HMM 的分块器，并将它们的输出组合起来，查看重叠部分。

我们将像过去几道菜谱中做的那样初始化几个块分割器，然后将相同的文本通过这些块分割器。最简单的情况是每个块分割器返回一个独特的输出。例如，让我们考虑一个句子，如“美国总统奥巴马今晚将在G-8会议上发表演讲”。如果我们有一个人物块分割器和组织块分割器，我们可能只能得到两个独特的块。然而，如果我们添加一个`Presidents of USA`块分割器，我们将得到三个块：`PERSON`、`ORGANIZATION`和`PRESIDENT`。这个非常简单的菜谱将展示处理这些情况的一种方法。

## 如何做到这一点...

1.  使用命令行或你IDE中的等效工具，输入以下内容：

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

1.  我们看到来自三个块分割器的输出：`neChunking`是一个训练有素以返回MUC-6实体的HMM块分割器的输出，`pChunking`是一个简单的正则表达式，用于识别男性代词，而`dChunking`是一个识别美国总统的字典块分割器。

1.  允许重叠的情况下，我们将在合并输出中看到`PRESIDENT`以及`PERSON`的块。

1.  禁止重叠的情况下，它们将被添加到重叠块集合中，并从唯一块集合中移除。

## 它是如何工作的...

我们初始化了三个块分割器，这些块分割器应该在本章之前的菜谱中对你来说很熟悉：

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

现在，我们将通过所有三个块分割器对输入文本进行块分割，将块组合成一个集合，并将我们的`getCombinedChunks`方法传递给它：

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

这个菜谱的核心在于`getCombinedChunks`方法。我们只需遍历所有块，并检查每一对是否在起始和结束位置重叠。如果它们重叠且不允许重叠，则将它们添加到一个重叠集中；否则，将它们添加到一个组合集中：

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

这里是添加更多重叠块规则的地方。例如，你可以使其基于分数，如果`PRESIDENT`块类型比基于HMM的块类型得分更高，你可以选择它。

# CRFs用于块分割

CRFs最著名的是在命名实体标注方面提供接近最先进性能。这个菜谱将告诉我们如何构建这样的系统。这个菜谱假设你已经阅读、理解并尝试过[第4章](part0051_split_000.html#page "第4章。标注单词和标记")中的*条件随机字段 – CRF用于单词/标记标注*菜谱，它涉及底层技术。与HMMs一样，CRFs将命名实体检测视为一个单词标注问题，有一个解释层提供块分割。与HMMs不同，CRFs使用基于逻辑回归的分类方法，这反过来又允许包括随机特征。此外，这个菜谱紧密遵循（但省略了细节）一个关于CRFs的优秀教程[http://alias-i.com/lingpipe/demos/tutorial/crf/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/crf/read-me.html)。Javadoc中也有大量信息。

## 准备工作

正如我们之前所做的那样，我们将使用一个小型手编语料库作为训练数据。语料库位于 `src/com/lingpipe/cookbook/chapter5/TinyEntityCorpus.java`。它从以下内容开始：

```py
public class TinyEntityCorpus extends Corpus<ObjectHandler<Chunking>> {

  public void visitTrain(ObjectHandler<Chunking> handler) {
    for (Chunking chunking : CHUNKINGS) handler.handle(chunking);
  }

  public void visitTest(ObjectHandler<Chunking> handler) {
    /* no op */
  }
```

由于我们只使用这个语料库进行训练，`visitTest()` 方法不起作用。然而，`visitTrain()` 方法将处理程序暴露给存储在 `CHUNKINGS` 常量中的所有分块。这反过来又看起来像以下这样：

```py
static final Chunking[] CHUNKINGS = new Chunking[] {
  chunking(""), chunking("The"), chunking("John ran.", chunk(0,4,"PER")), chunking("Mary ran.", chunk(0,4,"PER")), chunking("The kid ran."), chunking("John likes Mary.", chunk(0,4,"PER"), chunk(11,15,"PER")), chunking("Tim lives in Washington", chunk(0,3,"PER"), chunk(13,23,"LOC")), chunking("Mary Smith is in New York City", chunk(0,10,"PER"), chunk(17,30,"LOC")), chunking("New York City is fun", chunk(0,13,"LOC")), chunking("Chicago is not like Washington", chunk(0,7,"LOC"), chunk(20,30,"LOC"))
};
```

我们还没有完成。鉴于 `Chunking` 的创建相当冗长，有一些静态方法可以帮助动态创建所需的对象：

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

这就是所有的设置；接下来，我们将对前面的数据进行训练和运行一个条件随机场（CRF）。

## 如何做到这一点…

1.  在命令行中输入 `TrainAndRunSimplCrf` 类或在您的集成开发环境（IDE）中运行等效命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.TrainAndRunSimpleCrf

    ```

1.  这会产生大量的屏幕输出，报告 CRF 的健康和进度，这主要是来自驱动整个过程的底层逻辑回归分类器的信息。有趣的是，我们将收到一个邀请来玩新的 CRF：

    ```py
    Enter text followed by new line
    >John Smith went to New York.

    ```

1.  分块器报告了第一个最佳输出：

    ```py
    FIRST BEST
    John Smith went to New York. : [0-10:PER@-Infinity, 19-27:LOC@-Infinity]

    ```

1.  前面的输出是 CRF 对句子中存在哪些实体类型的第一次最佳分析。它认为 `John Smith` 是 `PER`，输出为 `the 0-10:PER@-Infinity`。我们知道它适用于 `John Smith` 字符串，是通过在输入文本中从 0 到 10 取子字符串来实现的。忽略 `–Infinity`，这是为没有得分的分块提供的。第一次最佳分块没有得分。它认为文本中存在的另一个实体是 `New York` 作为 `LOC`。

1.  立即，条件概率随之而来：

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

1.  前面的输出提供了整个短语的 10 个最佳分析，以及它们的条件（自然对数）概率。在这种情况下，我们将看到系统对其分析并不特别自信。例如，第一次最佳分析正确的估计概率为 `exp(-1.66)=0.19`。

1.  接下来，在输出中，我们可以看到单个分块的概率：

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

1.  与之前的条件输出一样，概率是日志形式，因此我们可以看到 `John Smith` 分块估计的概率为 `exp(-0.49) = 0.61`，这是有道理的，因为在训练 CRF 时，它看到了 `John` 在 `PER` 的开头，`Smith` 在另一个结尾，但没有直接看到 `John Smith`。

1.  前面的这种概率分布如果考虑到足够多的资源来考虑广泛的分析和结合证据的方式，以允许选择不太可能的结果，确实可以改善系统。第一次最佳分析往往过于承诺于符合训练数据看起来那样的保守结果。

## 它是如何工作的…

`src/com/lingpipe/cookbook/chapter5/TrainAndRunSimpleCRF.java` 中的代码与我们的分类器和隐马尔可夫模型（HMM）配方相似，但有几个不同之处。这些差异如下所述：

```py
public static void main(String[] args) throws IOException {
  Corpus<ObjectHandler<Chunking>> corpus = new TinyEntityCorpus();

  TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  boolean enforceConsistency = true;
  TagChunkCodec tagChunkCodec = new BioTagChunkCodec(tokenizerFactory, enforceConsistency);
```

在我们之前玩CRFs时，输入是 `Tagging<String>` 类型。回顾 `TinyEntityCorpus.java`，类型是 `Chunking` 类型。前面的 `BioTagChunkCodec` 通过提供的 `TokenizerFactory` 和 `boolean` 将 `Chunking` 转换为 `Tagging`，如果 `TokenizerFactory` 与 `Chunk` 的开始和结束不完全一致，则会抛出异常。回顾*在单词标记和分块之间转换-BIO编解码器*配方以更好地理解此类的作用。

让我们看看以下内容：

```py
John Smith went to New York City. : [0-10:PER@-Infinity, 19-32:LOC@-Infinity]
```

此编解码器将转换为以下标记：

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

```py
ChainCrfFeatureExtractor<String> featureExtractor = new SimpleCrfFeatureExtractor();
```

所有机制都隐藏在一个新的 `ChainCrfChunker` 类中，并且它的初始化方式类似于逻辑回归，这是其底层技术。有关配置的更多信息，请参阅[第3章](part0036_split_000.html#page "第3章. 高级分类器")的*逻辑回归*配方：

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

这里唯一的新事物是 `tagChunkCodec` 参数，我们刚刚描述了它。

训练完成后，我们将使用以下代码访问分块器以获取最佳结果：

```py
System.out.println("\nFIRST BEST");
Chunking chunking = crfChunker.chunk(evalText);
System.out.println(chunking);
```

条件分块通过以下方式提供：

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

通过以下方式访问单个分块：

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

就这样。您已经可以使用世界上最先进的分块技术之一了。接下来，我们将向您展示如何使其变得更好。

# 使用CRFs进行NER并具有更好的特征

在这个配方中，我们将向您展示如何为CRFs创建一组真实但并非最先进的特征。这些特征将包括归一化标记、词性标记、词形特征、位置特征以及标记的前缀和后缀。将其替换为*CRFs for chunking*配方中的 `SimpleCrfFeatureExtractor` 以使用它。

## 如何做…

此配方的源代码位于 `src/com/lingpipe/cookbook/chapter5/FancyCrfFeatureExtractor.java`：

1.  打开您的IDE或命令提示符，并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter5.FancyCrfFeatureExtractor

    ```

1.  准备好从控制台爆炸般涌现出的功能。用于特征提取的数据是之前配方中的 `TinyEntityCorpus`。幸运的是，第一部分数据只是句子 `John ran.` 中 "John" 的节点特征：

    ```py
    Tagging:  John/PN
    Node Feats:{PREF_NEXT_ra=1.0, PREF_Jo=1.0, POS_np=1.0, TOK_CAT_LET-CAP=1.0, SUFF_NEXT_an=1.0, PREF_Joh=1.0, PREF_NEXT_r=1.0, SUFF_John=1.0, TOK_John=1.0, PREF_NEXT_ran=1.0, BOS=1.0, TOK_NEXT_ran=1.0, SUFF_NEXT_n=1.0, SUFF_NEXT_ran=1.0, SUFF_ohn=1.0, PREF_J=1.0, POS_NEXT_vbd=1.0, SUFF_hn=1.0, SUFF_n=1.0, TOK_CAT_NEXT_ran=1.0, PREF_John=1.0}

    ```

1.  序列中的下一个词添加了边缘特征——我们不会展示节点特征：

    ```py
    Edge Feats:{PREV_TAG_TOKEN_CAT_PN_LET-CAP=1.0, PREV_TAG_PN=1.0}

    ```

## 它是如何工作的…

与其他配方一样，我们不会讨论与之前配方非常相似的部分——这里相关的先前配方是[第4章](part0051_split_000.html#page "第4章. 标记单词和标记")的*修改CRFs*配方，*标记单词和标记*。这完全一样，只是我们将添加更多功能——也许，来自意想不到的来源。

### 注意

CRFs教程涵盖了如何序列化/反序列化此类。此实现不包含它。

对象构造类似于[第4章](part0051_split_000.html#page "第4章. 标记单词和标记")的*修改CRFs*配方，*标记单词和标记*：

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

构造函数使用缓存设置了一个词性标注器，并将其推入`mPosTagger`成员变量中。

以下方法做得很少，除了提供一个内部的`ChunkerFeatures`类：

```py
public ChainCrfFeatures<String> extract(List<String> tokens, List<String> tags) {
  return new ChunkerFeatures(tokens,tags);
}
```

`ChunkerFeatures`类是事情变得更有趣的地方：

```py
class ChunkerFeatures extends ChainCrfFeatures<String> {
  private final Tagging<String> mPosTagging;

  public ChunkerFeatures(List<String> tokens, List<String> tags) {
    super(tokens,tags);
    mPosTagging = mPosTagger.tag(tokens);
  }
```

`mPosTagger`函数用于在类创建时为提供的标记设置`Tagging<String>`。这将与`tag()`和`token()`超类方法对齐，并作为节点特征的词性标签来源。

现在，我们可以继续进行特征提取。我们将从边缘特征开始，因为它们是最简单的：

```py
public Map<String,? extends Number> edgeFeatures(int n, int k) {
  ObjectToDoubleMap<String> feats = new ObjectToDoubleMap<String>();
  feats.set("PREV_TAG_" + tag(k),1.0);
  feats.set("PREV_TAG_TOKEN_CAT_"  + tag(k) + "_" + tokenCat(n-1), 1.0);
  return feats;
}
```

新特征以`PREV_TAG_TOKEN_CAT_`为前缀，例如`PREV_TAG_TOKEN_CAT_PN_LET-CAP=1.0`。`tokenCat()`方法查看前一个标记的词形特征，并将其作为字符串返回。查看`IndoEuropeanTokenCategorizer`的Javadoc以了解发生了什么。

接下来是节点特征。这里有许多这样的特征；每个将依次介绍：

```py
public Map<String,? extends Number> nodeFeatures(int n) {
  ObjectToDoubleMap<String> feats = new ObjectToDoubleMap<String>();
```

之前的代码设置了具有适当返回类型的方法。接下来的两行设置了某些状态，以了解特征提取器在字符串中的位置：

```py
boolean bos = n == 0;
boolean eos = (n + 1) >= numTokens();
```

接下来，我们将计算输入的当前位置、前一个位置和下一个位置的标记类别、标记和词性标签：

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

前一个和下一个方法检查我们是否位于句子的开始或结束，并相应地返回`null`。词性标注是从构造函数中计算出的已保存的词性标注中获取的。

标记方法提供了一些标记的规范化，以将所有数字压缩到同一类型的值。此方法如下：

```py
public String normedToken(int n) {
  return token(n).replaceAll("\\d+","*$0*").replaceAll("\\d","D");
}
```

这只是将每个数字序列替换为`*D...D*`。例如，`12/3/08`被转换为`*DD*/*D*/*DD*`。

然后，我们将为前一个、当前和后续标记设置特征值。首先，一个标志指示它是否是句子或内部节点的开始或结束：

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

最后，我们将添加前缀和后缀特征，这些特征为每个后缀和前缀（最长可达预定义长度）添加特征：

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

之后，我们只需返回生成的特征映射。

`prefix`或`suffix`函数简单地使用列表实现：

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

这是对你的命名实体检测器来说很棒的特征集。
