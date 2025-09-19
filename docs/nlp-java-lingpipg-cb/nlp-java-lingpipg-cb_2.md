# 第二章：查找和使用单词

本章涵盖了以下内容：

+   分词器工厂简介——在字符流中查找单词

+   结合分词器——小写分词器

+   结合分词器——停用词分词器

+   使用Lucene/Solr分词器

+   使用LingPipe与Lucene/Solr分词器

+   使用单元测试评估分词器

+   修改分词器工厂

+   为没有空格的语言查找单词

# 简介

构建NLP系统的一个重要部分是与适当的处理单元合作。本章讨论与处理单词级别相关的抽象层。这被称为分词，其目的是将相邻字符分组为有意义的块，以支持分类、实体识别以及其他NLP任务。

LingPipe提供了一系列的分词需求，这些需求在本书中没有涵盖。请查看分词器的Javadoc，这些分词器可以进行词干提取、Soundex（基于英语单词发音的标记）等。

# 分词器工厂简介——在字符流中查找单词

LingPipe分词器基于一个通用模式，即一个可以单独使用或作为后续过滤分词器源的基部分词器。过滤分词器操作基部分词器提供的标记/空白。本食谱涵盖了我们的最常用分词器`IndoEuropeanTokenizerFactory`，它适用于使用印欧风格标点符号和单词分隔符的语言——例如英语、西班牙语和法语。一如既往，Javadoc提供了有用的信息。

### 注意

`IndoEuropeanTokenizerFactory`创建具有内置对印欧语言中的字母数字、数字和其他常见结构的支持的分词器。

分词规则大致基于MUC-6中使用的规则，但必然更加精细，因为MUC分词器基于词汇和语义信息，例如一个字符串是否为缩写。

MUC-6指的是1995年发起的政府赞助的承包商之间竞争的会议，该会议起源于1995年。非正式术语是*Bake off*，指的是1949年开始的Pillsbury Bake-Off，其中一位作者作为MUC-6的后博士后参与者。MUC推动了NLP系统评估的大部分创新。

LingPipe 标记化器是使用 LingPipe `TokenizerFactory` 接口构建的，该接口提供了一种使用相同接口调用不同类型标记化器的方法。这在创建过滤标记化器时非常有用，过滤标记化器作为标记化器的链构建，并以某种方式修改其输出。`TokenizerFactory` 实例可以是基本标记化器，它在构建时接受简单参数，或者作为过滤标记化器，它接受其他标记化器工厂对象作为参数。在两种情况下，`TokenizerFactory` 实例都有一个单一的 `tokenize()` 方法，该方法接受输入作为字符数组、起始索引和要处理的字符数，并输出一个 `Tokenizer` 对象。`Tokenizer` 对象表示对特定字符串片段进行标记化的状态，并提供标记流。虽然 `TokenizerFactory` 是线程安全且/或可序列化的，但标记化器实例通常既不是线程安全的也不是可序列化的。`Tokenizer` 对象提供方法来遍历字符串中的标记，并提供标记在底层文本中的位置。

## 准备工作

如果您还没有这样做，请下载该书的 JAR 文件和源代码。

## 如何操作...

这一切都很简单。以下是与标记化开始相关的步骤：

1.  进入 `cookbook` 目录并调用以下类：

    ```py
    java -cp "lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar" com.lingpipe.cookbook.chapter2.RunBaseTokenizerFactory

    ```

    这将带我们到一个命令提示符，提示我们输入一些文本：

    ```py
    type a sentence to see tokens and white spaces

    ```

1.  如果我们输入一个句子，例如：`It's no use growing older if you only learn new ways of misbehaving yourself`，我们将得到以下输出：

    ```py
    It's no use growing older if you only learn new ways of misbehaving yourself. 
    Token:'It'
    WhiteSpace:''
    Token:'''
    WhiteSpace:''
    Token:'s'
    WhiteSpace:' '
    Token:'no'
    WhiteSpace:' '
    Token:'use'
    WhiteSpace:' '
    Token:'growing'
    WhiteSpace:' '
    Token:'older'
    WhiteSpace:' '
    Token:'if'
    WhiteSpace:' '
    Token:'you'
    WhiteSpace:' '
    Token:'only'
    WhiteSpace:' '
    Token:'learn'
    WhiteSpace:' '
    Token:'new'
    WhiteSpace:' '
    Token:'ways'
    WhiteSpace:' '
    Token:'of'
    WhiteSpace:' '
    Token:'misbehaving'
    WhiteSpace:' '
    Token:'yourself'
    WhiteSpace:''
    Token:'.'
    WhiteSpace:' '

    ```

1.  检查输出并注意标记和空白。文本来自萨基的短篇小说《巴斯特布尔女士的狂奔》。

## 它是如何工作的...

代码非常简单，可以完整地包含如下：

```py
package com.lingpipe.cookbook.chapter2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;

public class RunBaseTokenizerFactory {

  public static void main(String[] args) throws IOException {
    TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
    BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

    while (true) {
      System.out.println("type a sentence to " + "see the tokens and white spaces");
      String input = reader.readLine();
      Tokenizer tokenizer = tokFactory.tokenizer(input.toCharArray(), 0, input.length());
      String token = null;
      while ((token = tokenizer.nextToken()) != null) {
        System.out.println("Token:'" + token + "'");
        System.out.println("WhiteSpace:'" + tokenizer.nextWhitespace() + "'");

      }
    }
  }
}
```

此配方从 `main()` 方法的第一个语句中创建 `TokenizerFactory tokFactory` 开始。请注意，使用了单例 `IndoEuropeanTokenizerFactory.INSTANCE`。该工厂将为给定的字符串生成标记化器，这在 `Tokenizer tokenizer = tokFactory.tokenizer(input.toCharArray(), 0, input.length())` 这一行中很明显。输入的字符串通过 `input.toCharArray()` 转换为字符数组，作为 `tokenizer` 方法的第一个参数，并将起始和结束偏移量提供给创建的字符数组。

生成的 `tokenizer` 为提供的字符数组片段提供标记，空白和标记在 `while` 循环中打印出来。调用 `tokenizer.nextToken()` 方法执行以下操作：

+   该方法返回下一个标记或 null（如果没有下一个标记）。null 结束循环；否则，循环继续。

+   该方法还增加相应的空白。总是有一个与标记一起的空白，但它可能是空字符串。

`IndoEuropeanTokenizerFactory`假设字符分解的相当标准的抽象，分解如下：

+   从`char`数组的开始到第一个标记之间的字符被忽略，并且不会作为空白字符报告

+   上一个标记的末尾字符到`char`数组末尾之间的字符被报告为下一个空白字符

+   由于两个相邻的标记，空白可以是空字符串——注意输出中的撇号和相应的空白

这意味着如果输入不以标记开始，则不一定能够重建原始字符串。幸运的是，分词器可以很容易地修改以满足定制需求。我们将在本章后面看到这一点。

## 更多...

分词可以是任意复杂的。LingPipe分词器旨在覆盖大多数常见用途，但你可能需要创建自己的分词器以实现细粒度控制，例如，将“Victoria's Secret”中的“Victoria's”作为一个标记。如果需要此类自定义，请参考`IndoEuropeanTokenizerFactory`的源代码，以了解这里是如何进行任意分词的。

# 结合分词器 – 小写分词器

在前面的配方中，我们提到LingPipe分词器可以是基本的或过滤的。基本分词器，如印欧分词器，不需要太多的参数化，实际上根本不需要。然而，过滤分词器需要一个分词器作为参数。我们使用过滤分词器所做的就是在多个分词器中调用，其中基本分词器通常通过过滤器修改以产生不同的分词器。

LingPipe提供了几个基本分词器，例如`IndoEuropeanTokenizerFactory`或`CharacterTokenizerFactory`。完整的列表可以在LingPipe的Javadoc中找到。在本节中，我们将向您展示如何将一个印欧分词器与一个小写分词器结合使用。这是一个相当常见的流程，许多搜索引擎为印欧语言实现此流程。

## 准备工作

您需要下载书籍的JAR文件，并设置Java和Eclipse，以便可以运行示例。

## 如何做...

这与前面的配方工作方式相同。执行以下步骤：

1.  从命令行调用`RunLowerCaseTokenizerFactory`类：

    ```py
    java -cp "lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar" com.lingpipe.cookbook.chapter2.RunLowerCaseTokenizerFactory.

    ```

1.  然后，在命令提示符下，让我们使用以下示例：

    ```py
    type a sentence below to see the tokens and white spaces are:
    This is an UPPERCASE word and these are numbers 1 2 3 4.5.
    Token:'this'
    WhiteSpace:' '
    Token:'is'
    WhiteSpace:' '
    Token:'an'
    WhiteSpace:' '
    Token:'uppercase'
    WhiteSpace:' '
    Token:'word'
    WhiteSpace:' '
    Token:'and'
    WhiteSpace:' '
    Token:'these'
    WhiteSpace:' '
    Token:'are'
    WhiteSpace:' '
    Token:'numbers'
    WhiteSpace:' '
    Token:'1'
    WhiteSpace:' '
    Token:'2'
    WhiteSpace:' '
    Token:'3'
    WhiteSpace:' '
    Token:'4.5'
    WhiteSpace:''
    Token:'.'
    WhiteSpace:''

    ```

## 它是如何工作的...

您可以在前面的输出中看到，所有标记都被转换为小写，包括以大写输入的单词`UPPERCASE`。由于此示例使用印欧分词器作为其基本分词器，您可以看到数字4.5被保留为`4.5`，而不是被拆分为4和5。

我们组合分词器的方式非常简单：

```py
public static void main(String[] args) throws IOException {

  TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  tokFactory = new LowerCaseTokenizerFactory(tokFactory);
  tokFactory = new WhitespaceNormTokenizerFactory(tokFactory);

  BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

  while (true) {
    System.out.println("type a sentence below to see the tokens and white spaces are:");
    String input = reader.readLine();
    Tokenizer tokenizer = tokFactory.tokenizer(input.toCharArray(), 0, input.length());
    String token = null;
    while ((token = tokenizer.nextToken()) != null) {
      System.out.println("Token:'" + token + "'");
      System.out.println("WhiteSpace:'" + tokenizer.nextWhitespace() + "'");
    }
  }
}
```

在这里，我们创建了一个分词器，它返回使用印欧语分词器生成的归一化大小写和空白符标记符。从分词器工厂创建的分词器是一个过滤分词器，它以印欧语基础分词器开始，然后通过`LowerCaseTokenizer`修改以生成小写分词器。然后，它再次通过`WhiteSpaceNormTokenizerFactory`修改，以生成小写、空白符归一化的印欧语分词器。

在这里，我们应用了大小写归一化，因为单词的大小写并不重要；例如，搜索引擎通常在它们的索引中存储大小写归一化的单词。现在，我们将使用大小写归一化的标记符在即将到来的关于分类器的示例中。

## 参见

+   有关如何构建过滤分词器的更多详细信息，请参阅抽象类`ModifiedTokenizerFactory`的Javadoc。

# 结合分词器 - 停用词分词器

类似于我们组合小写和空白符归一化分词器的方式，我们可以使用过滤分词器创建一个过滤掉停用词的分词器。再次以搜索引擎为例，我们可以从我们的输入集中移除常见单词，以便归一化文本。通常被移除的停用词本身传达的信息很少，尽管它们可能在上下文中传达信息。

输入使用设置的任何基础分词器进行分词，然后，通过停用词分词器过滤掉结果标记符，以生成一个在初始化停用词分词器时指定的停用词免于出现的标记符流。

## 准备工作

你需要下载书籍的JAR文件，并设置Java和Eclipse，以便你可以运行示例。

## 如何做到这一点...

正如我们之前所做的那样，我们将通过与分词器交互的步骤：

1.  从命令行调用`RunStopTokenizerFactory`类：

    ```py
    java -cp "lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar" com.lingpipe.cookbook.chapter2.RunStopTokenizerFactory

    ```

1.  然后，在提示中，让我们使用以下示例：

    ```py
    type a sentence below to see the tokens and white spaces:
    the quick brown fox is jumping
    Token:'quick'
    WhiteSpace:' '
    Token:'brown'
    WhiteSpace:' '
    Token:'fox'
    WhiteSpace:' '
    Token:'jumping'
    WhiteSpace:''

    ```

1.  注意，我们丢失了相邻信息。在输入中，我们有`fox is jumping`，但分词结果却是`fox`后面跟着`jumping`，因为`is`被过滤掉了。这可能对需要准确相邻信息的基于分词的过程造成问题。在第4章（part0051_split_000.html#page "Chapter 4. Tagging Words and Tokens"）的*前景或背景驱动的有趣短语检测*配方中，我们将展示一个基于长度的过滤分词器，它可以保留相邻信息。

## 它是如何工作的...

在这个`StopTokenizerFactory`过滤器中使用的停用词只是一个非常短的单词列表，`is`、`of`、`the`和`to`。显然，如果需要，这个列表可以更长。正如你在前面的输出中看到的那样，单词`the`和`is`已经被从分词输出中移除。这是通过一个非常简单的步骤完成的：我们在`src/com/lingpipe/cookbook/chapter2/RunStopTokenizerFactory.java`中实例化`StopTokenizerFactory`。相关的代码是：

```py
TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
tokFactory = new LowerCaseTokenizerFactory(tokFactory);
Set<String> stopWords = new HashSet<String>();
stopWords.add("the");
stopWords.add("of");
stopWords.add("to");
stopWords.add("is");

tokFactory = new StopTokenizerFactory(tokFactory, stopWords);
```

由于我们在分词器工厂中使用 `LowerCaseTokenizerFactory` 作为其中一个过滤器，我们可以忽略只包含小写单词的停用词。如果我们想保留输入标记的大小写并继续移除停用词，我们需要添加大写或混合大小写的版本。

## 参见

+   LingPipe 提供的过滤分词器的完整列表可以在 Javadoc 页面 [http://alias-i.com/lingpipe/docs/api/com/aliasi/tokenizer/ModifyTokenTokenizerFactory.html](http://alias-i.com/lingpipe/docs/api/com/aliasi/tokenizer/ModifyTokenTokenizerFactory.html) 上找到。

# 使用 Lucene/Solr 分词器

非常流行的搜索引擎 Lucene 包含许多分析模块，它们提供通用分词器以及从阿拉伯语到泰语的语言特定分词器。截至 Lucene 4，这些不同的分析器大多可以在单独的 JAR 文件中找到。我们将介绍 Lucene 分词器，因为它们可以用作 LingPipe 分词器，您将在下一个配方中看到。

与 LingPipe 分词器类似，Lucene 分词器也可以分为基本分词器和过滤分词器。基本分词器以读取器作为输入，而过滤分词器以其他分词器作为输入。我们将查看一个使用标准 Lucene 分析器和 lowercase-filtered 分词器的示例。Lucene 分析器本质上将字段映射到标记流。因此，如果您有一个现有的 Lucene 索引，您可以使用分析器而不是原始分词器，正如我们将在本章的后续部分展示的那样。

## 准备工作

您需要下载书籍的 JAR 文件，并设置 Java 和 Eclipse，以便运行示例。示例中使用的某些 Lucene 分析器是 `lib` 目录的一部分。但是，如果您想尝试其他语言分析器，请从 Apache Lucene 网站下载它们：[https://lucene.apache.org](https://lucene.apache.org)。

## 如何操作...

记住，在这个配方中我们没有使用 LingPipe 分词器，而是介绍了 Lucene 分词器类：

1.  从命令行调用 `RunLuceneTokenizer` 类：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lucene-analyzers-common-4.6.0.jar:lib/lucene-core-4.6.0.jar com.lingpipe.cookbook.chapter2.RunLuceneTokenize

    ```

1.  然后，在提示中，让我们使用以下示例：

    ```py
    the quick BROWN fox jumped
    type a sentence below to see the tokens and white spaces:
    The rain in Spain.
    Token:'the' Start: 0 End:3
    Token:'rain' Start: 4 End:8
    Token:'in' Start: 9 End:11
    Token:'spain' Start: 12 End:17

    ```

## 它是如何工作的...

让我们回顾以下代码，看看 Lucene 分词器在调用上与前面的示例有何不同——来自 `src/com/lingpipe/cookbook/chapter2/RunLuceneTokenizer.java` 的相关代码部分是：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

while (true) {
```

```py
BufferedReader from the command line and starts a perpetual while() loop. Next, the prompt is provided, the input is read, and it is used to construct a Reader object:
```

```py
System.out.println("type a sentence below to see the tokens and white spaces:");
String input = reader.readLine();
Reader stringReader = new StringReader(input);
```

所有输入都已封装，现在是构建实际分词器的时候了：

```py
TokenStream tokenStream = new StandardTokenizer(Version.LUCENE_46,stringReader);

tokenStream = new LowerCaseFilter(Version.LUCENE_46,tokenStream);
```

输入文本用于使用 Lucene 的版本控制系统构建 `StandardTokenizer`，这产生了一个 `TokenStream` 实例。然后，我们使用 `LowerCaseFilter` 创建最终的过滤 `tokenStream`，其中将基本 `tokenStream` 作为参数。

在 Lucene 中，我们需要从标记流中附加我们感兴趣的属性；这是通过 `addAttribute` 方法完成的：

```py
CharTermAttribute terms = tokenStream.addAttribute(CharTermAttribute.class);
OffsetAttribute offset = tokenStream.addAttribute(OffsetAttribute.class);
tokenStream.reset();
```

注意，在 Lucene 4 中，一旦分词器被实例化，在使用分词器之前必须调用 `reset()` 方法：

```py
while (tokenStream.incrementToken()) {
  String token = terms.toString();
  int start = offset.startOffset();
  int end = offset.endOffset();
  System.out.println("Token:'" + token + "'" + " Start: " + start + " End:" + end);
}
```

`tokenStream` 被以下内容包装：

```py
tokenStream.end();
tokenStream.close();
```

## 参见

在 *Text Processing with Java*，*Mitzi Morris*，*Colloquial Media Corporation* 中，对 Lucene 的一个很好的介绍，其中我们之前解释的内容比在配方中提供的更清晰。

# 使用 LingPipe 与 Lucene/Solr 分词器

我们可以使用这些 Lucene 分词器与 LingPipe 一起使用；这很有用，因为 Lucene 有如此丰富的分词器集。我们将展示如何通过扩展 `Tokenizer` 抽象类将 Lucene `TokenStream` 包装到 LingPipe `TokenizerFactory` 中。

## How to do it...

我们将做一些不同的尝试，并有一个非交互式的配方。执行以下步骤：

1.  从命令行调用 `LuceneAnalyzerTokenizerFactory` 类：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lucene-analyzers-common-4.6.0.jar:lib/lucene-core-4.6.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter2.LuceneAnalyzerTokenizerFactory

    ```

1.  类中的 `main()` 方法指定了输入：

    ```py
    String text = "Hi how are you? " + "Are the numbers 1 2 3 4.5 all integers?";
    Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
    TokenizerFactory tokFactory = new LuceneAnalyzerTokenizerFactory(analyzer, "DEFAULT");
    Tokenizer tokenizer = tokFactory.tokenizer(text.toCharArray(), 0, text.length());

    String token = null;
    while ((token = tokenizer.nextToken()) != null) {
      String ws = tokenizer.nextWhitespace();
      System.out.println("Token:'" + token + "'");
      System.out.println("WhiteSpace:'" + ws + "'");
    }
    ```

1.  前面的代码片段创建了一个 Lucene `StandardAnalyzer` 并使用它来构建一个 LingPipe `TokenizerFactory`。输出如下——`StandardAnalyzer` 过滤掉停用词，所以标记 `are` 被过滤掉：

    ```py
    Token:'hi'
    WhiteSpace:'default'
    Token:'how'
    WhiteSpace:'default'
    Token:'you'
    WhiteSpace:'default'
    Token:'numbers'
    WhiteSpace:'default'

    ```

1.  空白字符报告为 `default`，因为实现没有准确提供空白字符，而是使用默认值。我们将在 *How it works…* 部分讨论这个限制。

## How it works...

让我们看看 `LuceneAnalyzerTokenizerFactory` 类。这个类通过包装 Lucene 分析器实现了 LingPipe `TokenizerFactory` 接口。我们将从 `src/com/lingpipe/cookbook/chapter2/LuceneAnalyzerTokenizerFactory.java` 中的类定义开始：

```py
public class LuceneAnalyzerTokenizerFactory implements TokenizerFactory, Serializable {

  private static final long serialVersionUID = 8376017491713196935L;
  private Analyzer analyzer;
  private String field;
  public LuceneAnalyzerTokenizerFactory(Analyzer analyzer, String field) {
    super();
    this.analyzer = analyzer;
    this.field = field;
  }
```

构造函数将分析器和字段的名称存储为私有变量。由于这个类实现了 `TokenizerFactory` 接口，我们需要实现 `tokenizer()` 方法：

```py
public Tokenizer tokenizer(char[] charSeq , int start, int length) {
  Reader reader = new CharArrayReader(charSeq,start,length);
  TokenStream tokenStream = analyzer.tokenStream(field,reader);
  return new LuceneTokenStreamTokenizer(tokenStream);
}
```

`tokenizer()` 方法创建一个新的字符数组读取器，并将其传递给 Lucene 分析器以将其转换为 `TokenStream`。基于标记流创建了一个 `LuceneTokenStreamTokenizer` 实例。`LuceneTokenStreamTokenizer` 是一个嵌套的静态类，它扩展了 LingPipe 的 `Tokenizer` 类：

```py
static class LuceneTokenStreamTokenizer extends Tokenizer {
  private TokenStream tokenStream;
  private CharTermAttribute termAttribute;
  private OffsetAttribute offsetAttribute;

  private int lastTokenStartPosition = -1;
  private int lastTokenEndPosition = -1;

  public LuceneTokenStreamTokenizer(TokenStream ts) {
    tokenStream = ts;
    termAttribute = tokenStream.addAttribute(
      CharTermAttribute.class);
    offsetAttribute = tokenStream.addAttribute(OffsetAttribute.class);
  }
```

构造函数存储 `TokenStream` 并附加术语和偏移量属性。在先前的配方中，我们看到了术语和偏移量属性包含标记字符串，以及输入文本中的标记起始和结束偏移量。在找到任何标记之前，标记偏移量也被初始化为 `-1`：

```py
@Override
public String nextToken() {
  try {
    if (tokenStream.incrementToken()){
      lastTokenStartPosition = offsetAttribute.startOffset();
      lastTokenEndPosition = offsetAttribute.endOffset();
      return termAttribute.toString();
    } else {
      endAndClose();
      return null;
    }
  } catch (IOException e) {
    endAndClose();
    return null;
  }
}
```

我们将实现 `nextToken()` 方法，并使用标记流的 `incrementToken()` 方法从标记流中检索任何标记。我们将使用 `OffsetAttribute` 设置标记的起始和结束偏移量。如果标记流已结束或 `incrementToken()` 方法抛出 I/O 异常，我们将结束并关闭 `TokenStream`。

`nextWhitespace()` 方法有一些限制，因为 `offsetAttribute` 专注于当前标记，LingPipe 分词器将输入量化为下一个标记和下一个偏移量。在这里找到一个通用的解决方案将非常具有挑战性，因为标记之间可能没有明确定义的空白——想想字符 n-gram。因此，提供了 `default` 字符串以使其更清晰。该方法如下：

```py
@Override
public String nextWhitespace() {   return "default";
}
```

代码还涵盖了如何序列化分词器，但我们将不在菜谱中涵盖这一点。

# 使用单元测试评估分词器

我们不会像 LingPipe 的其他组件那样，用精确度和召回率等指标来评估印欧语系分词器。相反，我们将通过单元测试来开发它们，因为我们的分词器是启发式构建的，并预期在示例数据上表现完美——如果一个分词器未能分词一个已知案例，那么它是一个错误，而不是性能的降低。为什么是这样？有几个原因：

+   许多分词器非常“机械”，并且易于单元测试框架的刚性。例如，`RegExTokenizerFactory` 显然是单元测试的候选，而不是评估工具。

+   推动大多数分词器的启发式规则非常通用，并且不存在以部署系统为代价过度拟合训练数据的问题。如果你有一个已知的坏案例，你只需去修复分词器并添加一个单元测试即可。

+   标记和空白被认为是语义中性的，这意味着标记不会根据上下文而改变。但我们的印欧语系分词器并非完全如此，因为它会根据上下文不同对待 `.`，例如，如果是十进制的一部分或在句子末尾，例如 `3.14 是 pi.`：

    ```py
    Token:'3.14'
    WhiteSpace:' '
    Token:'is'
    WhiteSpace:' '
    Token:'pi'
    WhiteSpace:''
    Token:'.'
    WhiteSpace:''.

    ```

对于基于统计的分词器，可能需要使用评估指标；这在本章的 *为没有空格的语言寻找单词* 菜谱中有讨论。参见 [第 5 章](part0061_split_000.html#page "第 5 章. 在文本中寻找跨度 – 分块") 的 *句子检测评估* 菜谱，*在文本中寻找跨度 – 分块*，以了解适当的基于跨度的评估技术。

## 如何操作...

我们将跳过运行代码步骤，直接进入源代码来构建一个分词器评估器。源代码位于 `src/com/lingpipe/chapter2/TestTokenizerFactory.java`。执行以下步骤：

1.  以下代码设置了一个基于正则表达式的基分词工厂——如果你不清楚正在构建的内容，请查看该类的 Javadoc：

    ```py
    public static void main(String[] args) {
      String pattern = "[a-zA-Z]+|[0-9]+|\\S";
      TokenizerFactory tokFactory = new RegExTokenizerFactory(pattern);
      String[] tokens = {"Tokenizers","need","unit","tests","."};
      String text = "Tokenizers need unit tests.";
      checkTokens(tokFactory,text,tokens);
      String[] whiteSpaces = {" "," "," ","",""};
      checkTokensAndWhiteSpaces(tokFactory,text,tokens,whiteSpaces);
      System.out.println("All tests passed!");
    }
    ```

1.  `checkTokens` 方法接受 `TokenizerFactory`，一个表示所需分词的 `String` 数组，以及要分词的 `String`。它遵循以下步骤：

    ```py
    static void checkTokens(TokenizerFactory tokFactory, String string, String[] correctTokens) {
      Tokenizer tokenizer = tokFactory.tokenizer(input.toCharArray(),0,input.length());
      String[] tokens = tokenizer.tokenize();
      if (tokens.length != correctTokens.length) {
        System.out.println("Token list lengths do not match");
        System.exit(-1);
      }
      for (int i = 0; i < tokens.length; ++i) {
        if (!correctTokens[i].equals(tokens[i])) {
          System.out.println("Token mismatch: got |" + tokens[i] + "|");
          System.out.println(" expected |" + correctTokens[i] + "|" );
          System.exit(-1);
        }
      }
    ```

1.  该方法对错误非常敏感，因为如果令牌数组长度不同或任何令牌不相等，程序就会退出。JUnit这样的适当单元测试框架将是一个更好的框架，但这超出了本书的范围。您可以查看`lingpipe.4.1.0`/`src/com/aliasi/test`中的LingPipe单元测试，了解JUnit的使用方法。

1.  `checkTokensAndWhiteSpaces()`方法检查空白字符以及令牌。它遵循与`checkTokens()`相同的基本思想，所以我们不再解释。

# 修改分词器工厂

在这个菜谱中，我们将描述一个修改令牌流中令牌的分词器。我们将扩展`ModifyTokenTokenizerFactory`类以返回在英文字母表中旋转了13个位置的文本，也称为rot-13。Rot-13是一个非常简单的替换密码，用字母表中13个位置后的字母替换字母。例如，字母`a`将被替换为字母`n`，字母`z`将被替换为字母`m`。这是一个互逆密码，这意味着应用相同的密码两次可以恢复原始文本。

## 如何做到这一点...

我们将从命令行调用`Rot13TokenizerFactory`类：

```py
java -cp "lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar" com.lingpipe.cookbook.chapter2.Rot13TokenizerFactory

type a sentence below to see the tokens and white spaces:
Move along, nothing to see here.
Token:'zbir'
Token:'nybat'
Token:','
Token:'abguvat'
Token:'gb'
Token:'frr'
Token:'urer'
Token:'.'
Modified Output: zbir nybat, abguvat gb frr urer.
type a sentence below to see the tokens and white spaces:
zbir nybat, abguvat gb frr urer.
Token:'move'
Token:'along'
Token:','
Token:'nothing'
Token:'to'
Token:'see'
Token:'here'
Token:'.'
Modified Output: move along, nothing to see here.

```

您可以看到，输入文本，原本是混合大小写且为正常英语，已经被转换为其Rot-13等价形式。您可以看到，第二次，我们将经过Rot-13修改后的文本作为输入，并得到了原始文本，除了它全部都是小写字母。

## 它是如何工作的...

`Rot13TokenizerFactory`扩展了`ModifyTokenTokenizerFactory`类。我们将重写`modifyToken()`方法，该方法一次操作一个令牌，在这种情况下，将令牌转换为它的Rot-13等价形式。还有一个类似的`modifyWhiteSpace`（String）方法，如果需要，会修改空白字符：

```py
public class Rot13TokenizerFactory extends ModifyTokenTokenizerFactory{

  public Rot13TokenizerFactory(TokenizerFactory f) {
    super(f);
  }

  @Override
  public String modifyToken(String tok) {
    return rot13(tok);
  }

  public static void main(String[] args) throws IOException {

  TokenizerFactory tokFactory = IndoEuropeanTokenizerFactory.INSTANCE;
  tokFactory = new LowerCaseTokenizerFactory(tokFactory);
  tokFactory = new Rot13TokenizerFactory(tokFactory);
```

令牌本身的起始和结束偏移量与底层分词器的相同。在这里，我们将使用一个印欧语系分词器作为我们的基础分词器。首先通过`LowerCaseTokenizer`过滤一次，然后通过`Rot13Tokenizer`过滤。

`rot13`方法如下：

```py
public static String rot13(String input) {
  StringBuilder sb = new StringBuilder();
  for (int i = 0; i < input.length(); i++) {
    char c = input.charAt(i);
    if       (c >= 'a' && c <= 'm') c += 13;
    else if  (c >= 'A' && c <= 'M') c += 13;
    else if  (c >= 'n' && c <= 'z') c -= 13;
    else if  (c >= 'N' && c <= 'Z') c -= 13;
    sb.append(c);
  }
  return sb.toString();
}
```

# 为没有空格的语言寻找单词

例如，像中文这样的语言没有单词边界。例如，木卫三是围绕木星运转的一颗卫星，公转周期约为7天，来自维基百科的这句话在中文中大致翻译为“甘尼德在木星的卫星周围运行，轨道周期约为七天”，这是由[https://translate.google.com](https://translate.google.com)上的机器翻译服务完成的。注意空格的缺失。

在这类数据中查找标记需要一种非常不同的方法，该方法基于字符语言模型和我们的拼写检查类。这个配方通过将未标记的文本视为*拼写错误*的文本来编码查找单词，其中*更正*插入空格以分隔标记。当然，中文、日语、越南语和其他非单词分隔的语系并没有拼写错误，但我们已经在拼写纠正类中对其进行了编码。

## 准备工作

我们将使用去空格的英语来近似非单词分隔的语系。这足以理解配方，并且可以很容易地修改为实际所需的语言。获取大约10万个英语单词，并将它们以UTF-8编码的方式存储到磁盘上。固定编码的原因是输入假定是UTF-8——你可以通过更改编码并重新编译配方来更改它。

我们使用了马克·吐温的《亚瑟王宫廷中的康涅狄格州扬基》，从Project Gutenberg下载（[http://www.gutenberg.org/](http://www.gutenberg.org/)）。Project Gutenberg是公共领域文本的绝佳来源，马克·吐温是一位优秀的作家——我们强烈推荐这本书。将你的选定文本放在食谱目录中或使用我们的默认设置。

## 如何做到这一点...

我们将运行一个程序，稍作玩耍，并使用以下步骤解释它做什么以及它是如何做到的：

1.  输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter2.TokenizeWithoutWhiteSpaces
    Type an Englese sentence (English without spaces like Chinese):
    TheraininSpainfallsmainlyontheplain

    ```

1.  下面的输出是：

    ```py
    The rain in Spain falls mainly on the plain
    ```

1.  你可能不会得到完美的输出。马克·吐温从生成它的Java程序中恢复正确空白的能力如何？让我们来看看：

    ```py
    type an Englese sentence (English without spaces like Chinese)
    NGramProcessLMlm=newNGramProcessLM(nGram);
    NGram Process L Mlm=new NGram Process L M(n Gram);

    ```

1.  前面的方法并不很好，但我们也不是非常公平；让我们使用LingPipe的连接源作为训练数据：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter2.TokenizeWithoutWhiteSpaces data/cookbookSource.txt
    Compiling Spell Checker
    type an Englese sentence (English without spaces like Chinese)
    NGramProcessLMlm=newNGramProcessLM(nGram);
    NGramProcessLM lm = new NGramProcessLM(nGram);

    ```

1.  这就是完美的空格插入。

## 它是如何工作的...

尽管有很多乐趣和游戏，但涉及的代码非常少。酷的地方在于我们正在基于[第1章](part0014_split_000.html#page "Chapter 1. Simple Classifiers")中的字符语言模型，*简单分类器*。源代码位于`src/com/lingpipe/chapter2/TokenizeWithoutWhiteSpaces.java`：

```py
public static void main (String[] args) throws IOException, ClassNotFoundException {
  int nGram = 5;
  NGramProcessLM lm = new NGramProcessLM(nGram);
  WeightedEditDistance spaceInsertingEditDistance
    = CompiledSpellChecker.TOKENIZING;
  TrainSpellChecker trainer = new TrainSpellChecker(lm, spaceInsertingEditDistance);
```

`main()`方法通过创建`NgramProcessLM`来启动。接下来，我们将访问一个用于编辑距离的类，该类旨在仅向字符流中添加空格。就是这样。`Editdistance`通常是一个相当粗略的字符串相似度度量，它评估需要对`string1`进行多少编辑才能使其与`string2`相同。关于这方面的许多信息可以在Javadoc `com.aliasi.spell`中找到。例如，`com.aliasi.spell.EditDistance`对基础知识进行了很好的讨论。

### 注意

`EditDistance`类实现了带有或不带有转置的标准编辑距离概念。不带转置的距离被称为Levenshtein距离，带有转置则称为Damerau-Levenstein距离。

使用LingPipe阅读Javadoc；它包含许多有用的信息，但这些信息在这本书中我们没有足够的空间来介绍。

到目前为止，我们已经配置和构建了一个 `TrainSpellChecker` 类。下一步是自然地对其进行训练：

```py
File trainingFile = new File(args[0]);
String training = Files.readFromFile(trainingFile, Strings.UTF8);
training = training.replaceAll("\\s+", " ");
trainer.handle(training);
```

我们假设文本文件是 UTF-8 编码，将其全部读取；如果不是，则更正字符编码并重新编译。然后，我们将所有多个空白字符替换为单个空白字符。如果多个空白字符具有意义，这可能不是最好的选择。接下来是训练，就像我们在 [第 1 章](part0014_split_000.html#page "第 1 章。简单分类器") 中训练语言模型一样，*简单分类器*。

接下来，我们将编译和配置拼写检查器：

```py
System.out.println("Compiling Spell Checker");
CompiledSpellChecker spellChecker = (CompiledSpellChecker)AbstractExternalizable.compile(trainer);

spellChecker.setAllowInsert(true);
spellChecker.setAllowMatch(true);
spellChecker.setAllowDelete(false);
spellChecker.setAllowSubstitute(false);
spellChecker.setAllowTranspose(false);
spellChecker.setNumConsecutiveInsertionsAllowed(1);
```

下一个有趣的步骤是编译 `spellChecker`，它将底层语言模型中的所有计数转换为预计算的概率，这要快得多。编译步骤可以写入磁盘，因此可以在不进行训练的情况下稍后使用；然而，请参阅 `AbstractExternalizable` 的 Javadoc 了解如何进行此操作。接下来的几行配置 `CompiledSpellChecker` 以仅考虑插入字符的编辑，并检查精确字符串匹配，但禁止删除、替换和转置。最后，只允许一个插入。应该很清楚，我们正在使用 `CompiledSpellChecker` 的非常有限的功能，但这正是所需要的——插入空格或不插入。

最后是我们的标准 I/O 例程：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
while (true) {
  System.out.println("type an Englese sentence (English " + "without spaces like Chinese)"));
  String input = reader.readLine();
  String result = spellChecker.didYouMean(input);
  System.out.println(result);
}
```

`CompiledSpellChecker` 和 `WeightedEditDistance` 类的机制在 Javadoc 或 [第 6 章](part0075_split_000.html#page "第 6 章。字符串比较和聚类") 的 *使用编辑距离和语言模型进行拼写校正* 菜单中描述得更好，*字符串比较和聚类*。然而，基本思想是将输入的字符串与刚刚训练的语言模型进行比较，得到一个分数，显示该字符串与模型匹配得有多好。这个字符串将是一个没有空白字符的巨大单词——但请注意，这里没有分词器在工作，因此拼写检查器开始插入空格并重新评估结果的分数。它保留这些序列，其中插入空格会增加序列的分数。

记住，语言模型是在带有空白字符的文本上训练的。拼写检查器试图在可能的地方插入空格，并保留一组“迄今为止最佳”的空白字符插入。最后，它返回最佳得分的编辑序列。

注意，为了完成分词器，需要将适当的 `TokenizerFactory` 应用到经过空白字符修改的文本上，但这被留作读者的练习。

## 还有更多...

`CompiledSpellChecker` 允许输出 *n*-best 结果；这允许对文本进行多种可能的解析。在像研究搜索引擎这样的高覆盖率/召回率情况下，可能需要允许应用多种分词。此外，可以通过直接扩展 `WeightedEditDistance` 类来调整编辑成本。

## 参见

如果不提供非英语资源，那么这份食谱将不会有所帮助。我们构建并评估了一个中文分词器，使用了网络上可用的资源进行科研。我们的关于中文分词教程详细介绍了这一点。您可以在[http://alias-i.com/lingpipe/demos/tutorial/chineseTokens/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/chineseTokens/read-me.html)找到中文分词教程。
