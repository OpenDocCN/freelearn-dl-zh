# 第二章：查找和处理词语

在本章中，我们介绍以下配方：

+   分词器工厂简介——在字符流中查找单词

+   结合分词器——小写字母分词器

+   结合分词器——停用词分词器

+   使用 Lucene/Solr 分词器

+   使用 Lucene/Solr 分词器与 LingPipe

+   使用单元测试评估分词器

+   修改分词器工厂

+   查找没有空格的语言的单词

# 介绍

构建 NLP 系统的重要部分是使用适当的处理单元。本章讨论的是与词级处理相关的抽象层次。这个过程称为分词，它将相邻字符分组为有意义的块，以支持分类、实体识别和其他 NLP 任务。

LingPipe 提供了广泛的分词器需求，这些需求在本书中没有涵盖。请查阅 Javadoc 以了解执行词干提取、Soundex（基于英语发音的标记）等的分词器。

# 分词器工厂简介——在字符流中查找单词

LingPipe 分词器建立在一个通用的基础分词器模式上，基础分词器可以单独使用，也可以作为后续过滤分词器的来源。过滤分词器会操作由基础分词器提供的标记和空格。本章节涵盖了我们最常用的分词器 `IndoEuropeanTokenizerFactory`，它适用于使用印欧语言风格的标点符号和词汇分隔符的语言——例如英语、西班牙语和法语。和往常一样，Javadoc 中包含了有用的信息。

### 注意

`IndoEuropeanTokenizerFactory` 创建具有内建支持的分词器，支持印欧语言中的字母数字、数字和其他常见构造。

分词规则大致基于 MUC-6 中使用的规则，但由于 MUC 分词器基于词汇和语义信息（例如，字符串是否为缩写），因此这些规则必须更为精细。

MUC-6 指的是 1995 年发起的消息理解会议，它创立了政府资助的承包商之间的竞争形式。非正式的术语是 *Bake off*，指的是 1949 年开始的比尔斯伯里烘焙大赛，且其中一位作者在 MUC-6 中作为博士后参与了该会议。MUC 对自然语言处理系统评估的创新起到了重要推动作用。

LingPipe 标记器是使用 LingPipe 的`TokenizerFactory`接口构建的，该接口提供了一种方法，可以使用相同的接口调用不同类型的标记器。这在创建过滤标记器时非常有用，过滤标记器是通过一系列标记器链构建的，并以某种方式修改其输出。`TokenizerFactory`实例可以作为基本标记器创建，它在构造时接受简单的参数，或者作为过滤标记器创建，后者接受其他标记器工厂对象作为参数。在这两种情况下，`TokenizerFactory`的实例都有一个`tokenize()`方法，该方法接受输入字符数组、起始索引和要处理的字符数，并输出一个`Tokenizer`对象。`Tokenizer`对象表示标记化特定字符串片段的状态，并提供标记符流。虽然`TokenizerFactory`是线程安全和/或可序列化的，但标记器实例通常既不线程安全也不具备序列化功能。`Tokenizer`对象提供了遍历字符串中标记符的方法，并提供标记符在底层文本中的位置。

## 准备工作

如果你还没有下载书籍的 JAR 文件和源代码，请先下载。

## 如何操作...

一切都很简单。以下是开始标记化的步骤：

1.  转到`cookbook`目录并调用以下类：

    ```py
    java -cp "lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar" com.lingpipe.cookbook.chapter2.RunBaseTokenizerFactory

    ```

    这将带我们进入命令提示符，提示我们输入一些文本：

    ```py
    type a sentence to see tokens and white spaces

    ```

1.  如果我们输入如下句子：`It's no use growing older if you only learn new ways of misbehaving yourself`，我们将得到以下输出：

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

1.  查看输出并注意标记符和空格的内容。文本摘自萨基的短篇小说《*巴斯特布尔夫人的冲击*》。

## 它是如何工作的...

代码非常简单，可以完整地如下包含：

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

本示例从在`main()`方法的第一行创建`TokenizerFactory tokFactory`开始。注意使用了单例`IndoEuropeanTokenizerFactory.INSTANCE`。该工厂会为给定的字符串生成标记器，这一点在这一行中有所体现：`Tokenizer tokenizer = tokFactory.tokenizer(input.toCharArray(), 0, input.length())`。输入的字符串通过`input.toCharArray()`转换为字符数组，并作为`tokenizer`方法的第一个参数，起始和结束偏移量传入到生成的字符数组中。

结果`tokenizer`为提供的字符数组片段提供标记符，空格和标记符将在`while`循环中打印出来。调用`tokenizer.nextToken()`方法执行了几个操作：

+   该方法返回下一个标记符，如果没有下一个标记符，则返回 null。此时循环结束；否则，循环继续。

+   该方法还会递增相应的空格。每个标记符后面都会有一个空格，但它可能是空字符串。

`IndoEuropeanTokenizerFactory`假设有一个相当标准的字符抽象，其分解如下：

+   从`char`数组的开头到第一个分词的字符会被忽略，并不会被报告为空格。

+   从上一个分词的末尾到`char`数组末尾的字符被报告为下一个空格。

+   空格可能是空字符串，因为有两个相邻的分词——注意输出中的撇号和相应的空格。

这意味着，如果输入不以分词开始，则可能无法重建原始字符串。幸运的是，分词器很容易根据自定义需求进行修改。我们将在本章后面看到这一点。

## 还有更多内容……

分词可能会非常复杂。LingPipe 分词器旨在覆盖大多数常见用例，但你可能需要创建自己的分词器以进行更精细的控制，例如，将“Victoria's Secret”中的“Victoria's”作为一个分词。如果需要这样的自定义，请查阅`IndoEuropeanTokenizerFactory`的源码，了解这里是如何进行任意分词的。

# 组合分词器——小写分词器

我们在前面的配方中提到过，LingPipe 分词器可以是基本的或过滤的。基本分词器，例如 Indo-European 分词器，不需要太多的参数化，事实上根本不需要。然而，过滤分词器需要一个分词器作为参数。我们使用过滤分词器的做法是调用多个分词器，其中一个基础分词器通常会被过滤器修改，产生一个不同的分词器。

LingPipe 提供了几种基本的分词器，例如`IndoEuropeanTokenizerFactory`或`CharacterTokenizerFactory`。完整的列表可以在 LingPipe 的 Javadoc 中找到。在本节中，我们将向你展示如何将 Indo-European 分词器与小写分词器结合使用。这是许多搜索引擎为印欧语言实现的一个常见过程。

## 准备工作

你需要下载书籍的 JAR 文件，并确保已经设置好 Java 和 Eclipse，以便能够运行示例。

## 如何操作……

这与前面的配方完全相同。请按照以下步骤操作：

1.  从命令行调用`RunLowerCaseTokenizerFactory`类：

    ```py
    java -cp "lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar" com.lingpipe.cookbook.chapter2.RunLowerCaseTokenizerFactory.

    ```

1.  然后，在命令提示符下，我们使用以下示例：

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

## 它是如何工作的……

你可以在前面的输出中看到，所有的分词都被转换成小写，包括大写输入的单词`UPPERCASE`。由于这个示例使用了 Indo-European 分词器作为基础分词器，你可以看到数字 4.5 被保留为`4.5`，而不是分解为 4 和 5。

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

在这里，我们创建了一个分词器，该分词器返回通过印欧语言分词器产生的大小写和空格标准化的标记。通过分词器工厂创建的分词器是一个过滤的分词器，它从印欧基础分词器开始，然后由`LowerCaseTokenizer`修改为小写分词器。接着，它再次通过`WhiteSpaceNormTokenizerFactory`进行修改，生成一个小写且空格标准化的印欧分词器。

在对单词大小写不太重要的地方应用大小写标准化；例如，搜索引擎通常会将大小写标准化的单词存储在索引中。现在，我们将在接下来的分类器示例中使用大小写标准化的标记。

## 另见

+   有关如何构建过滤分词器的更多细节，请参见抽象类`ModifiedTokenizerFactory`的 Javadoc。

# 组合分词器 – 停用词分词器

类似于我们如何构建一个小写和空格标准化的分词器，我们可以使用一个过滤的分词器来创建一个过滤掉停用词的分词器。再次以搜索引擎为例，我们可以从输入集中删除常见的词汇，以便规范化文本。通常被移除的停用词本身传达的信息很少，尽管它们在特定上下文中可能会有意义。

输入会通过所设置的基础分词器进行分词，然后由停用词分词器过滤掉，从而生成一个不包含初始化时指定的停用词的标记流。

## 准备工作

你需要下载书籍的 JAR 文件，并确保已经安装 Java 和 Eclipse，以便能够运行示例。

## 如何做……

如前所述，我们将通过与分词器交互的步骤进行演示：

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

1.  请注意，我们失去了邻接信息。在输入中，我们有`fox is jumping`，但分词后变成了`fox`后跟`jumping`，因为`is`被过滤掉了。对于那些需要准确邻接信息的基于分词的过程，这可能会成为一个问题。在第四章的*前景驱动或背景驱动的有趣短语检测*配方中，我们将展示一个基于长度过滤的分词器，它保留了邻接信息。

## 它是如何工作的……

在这个`StopTokenizerFactory`过滤器中使用的停用词仅是一个非常简短的单词列表，包括`is`、`of`、`the`和`to`。显然，如果需要，这个列表可以更长。如你在前面的输出中看到的，单词`the`和`is`已经从分词输出中移除了。这通过一个非常简单的步骤完成：我们在`src/com/lingpipe/cookbook/chapter2/RunStopTokenizerFactory.java`中实例化了`StopTokenizerFactory`。相关代码如下：

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

由于我们使用`LowerCaseTokenizerFactory`作为分词器工厂中的一个过滤器，我们可以忽略只包含小写字母的停用词。如果我们想保留输入标记的大小写并继续删除停用词，我们还需要添加大写或混合大小写版本。

## 另请参见

+   由 LingPipe 提供的过滤器分词器的完整列表可以在 Javadoc 页面找到，链接为[`alias-i.com/lingpipe/docs/api/com/aliasi/tokenizer/ModifyTokenTokenizerFactory.html`](http://alias-i.com/lingpipe/docs/api/com/aliasi/tokenizer/ModifyTokenTokenizerFactory.html)

# 使用 Lucene/Solr 分词器

备受欢迎的搜索引擎 Lucene 包含许多分析模块，提供通用的分词器以及从阿拉伯语到泰语的语言特定分词器。从 Lucene 4 开始，这些不同的分析器大多可以在单独的 JAR 文件中找到。我们将讲解 Lucene 分词器，因为它们可以像 LingPipe 分词器一样使用，正如你将在下一个配方中看到的那样。

就像 LingPipe 分词器一样，Lucene 分词器也可以分为基础分词器和过滤分词器。基础分词器以读取器为输入，过滤分词器则以其他分词器为输入。我们将看一个示例，演示如何使用标准的 Lucene 分析器和一个小写过滤分词器。Lucene 分析器本质上是将字段映射到一个标记流。因此，如果你有一个现有的 Lucene 索引，你可以使用分析器和字段名称，而不是使用原始的分词器，正如我们在本章后面的部分所展示的那样。

## 准备工作

你需要下载本书的 JAR 文件，并配置 Java 和 Eclipse，以便运行示例。示例中使用的一些 Lucene 分析器是`lib`目录的一部分。然而，如果你想尝试其他语言的分析器，可以从 Apache Lucene 官网[`lucene.apache.org`](https://lucene.apache.org)下载它们。

## 如何实现...

请记住，在这个配方中我们没有使用 LingPipe 分词器，而是介绍了 Lucene 分词器类：

1.  从命令行调用`RunLuceneTokenizer`类：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lucene-analyzers-common-4.6.0.jar:lib/lucene-core-4.6.0.jar com.lingpipe.cookbook.chapter2.RunLuceneTokenize

    ```

1.  然后，在提示中，我们使用以下示例：

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

让我们回顾以下代码，看看 Lucene 分词器如何与前面的示例中的调用不同——`src/com/lingpipe/cookbook/chapter2/RunLuceneTokenizer.java`中相关部分的代码是：

```py
BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

while (true) {
```

上述代码片段从命令行设置`BufferedReader`并启动一个永久的`while()`循环。接下来，提供了提示，读取`input`，并用于构造`Reader`对象：

```py
System.out.println("type a sentence below to see the tokens and white spaces:");
String input = reader.readLine();
Reader stringReader = new StringReader(input);
```

所有输入现在都已封装，可以构造实际的分词器了：

```py
TokenStream tokenStream = new StandardTokenizer(Version.LUCENE_46,stringReader);

tokenStream = new LowerCaseFilter(Version.LUCENE_46,tokenStream);
```

输入文本用于构造`StandardTokenizer`，并提供 Lucene 的版本控制系统——这会生成一个`TokenStream`实例。接着，我们使用`LowerCaseFilter`创建最终的过滤`tokenStream`，并将基础`tokenStream`作为参数传入。

在 Lucene 中，我们需要从 token 流中附加我们感兴趣的属性；这可以通过`addAttribute`方法完成：

```py
CharTermAttribute terms = tokenStream.addAttribute(CharTermAttribute.class);
OffsetAttribute offset = tokenStream.addAttribute(OffsetAttribute.class);
tokenStream.reset();
```

请注意，在 Lucene 4 中，一旦 tokenizer 被实例化，必须在使用 tokenizer 之前调用`reset()`方法：

```py
while (tokenStream.incrementToken()) {
  String token = terms.toString();
  int start = offset.startOffset();
  int end = offset.endOffset();
  System.out.println("Token:'" + token + "'" + " Start: " + start + " End:" + end);
}
```

`tokenStream`用以下方式进行包装：

```py
tokenStream.end();
tokenStream.close();
```

## 另请参见

关于 Lucene 的一个优秀入门书籍是*Text Processing with Java*，*Mitzi Morris*，*Colloquial Media Corporation*，其中我们之前解释的内容比我们在此提供的食谱更清晰易懂。

# 将 Lucene/Solr 的 tokenizers 与 LingPipe 一起使用

我们可以将这些 Lucene 的 tokenizers 与 LingPipe 一起使用；这是非常有用的，因为 Lucene 拥有一套非常丰富的 tokenizers。我们将展示如何通过扩展`Tokenizer`抽象类将 Lucene 的`TokenStream`封装成 LingPipe 的`TokenizerFactory`。

## 如何实现……

我们将稍微改变一下，提供一个非交互式的示例。请执行以下步骤：

1.  从命令行调用`LuceneAnalyzerTokenizerFactory`类：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lucene-analyzers-common-4.6.0.jar:lib/lucene-core-4.6.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter2.LuceneAnalyzerTokenizerFactory

    ```

1.  类中的`main()`方法指定了输入：

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

1.  前面的代码片段创建了一个 Lucene 的`StandardAnalyzer`并用它构建了一个 LingPipe 的`TokenizerFactory`。输出如下——`StandardAnalyzer`过滤了停用词，因此单词`are`被过滤掉了：

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

1.  空格报告为`default`，因为实现没有准确提供空格，而是使用了默认值。我们将在*它是如何工作的……*部分讨论这个限制。

## 它是如何工作的……

让我们来看一下`LuceneAnalyzerTokenizerFactory`类。这个类通过封装一个 Lucene 分析器实现了 LingPipe 的`TokenizerFactory`接口。我们将从`src/com/lingpipe/cookbook/chapter2/LuceneAnalyzerTokenizerFactory.java`中的类定义开始：

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

构造函数将分析器和字段名作为私有变量存储。由于该类实现了`TokenizerFactory`接口，我们需要实现`tokenizer()`方法：

```py
public Tokenizer tokenizer(char[] charSeq , int start, int length) {
  Reader reader = new CharArrayReader(charSeq,start,length);
  TokenStream tokenStream = analyzer.tokenStream(field,reader);
  return new LuceneTokenStreamTokenizer(tokenStream);
}
```

`tokenizer()`方法创建一个新的字符数组读取器，并将其传递给 Lucene 分析器，将其转换为`TokenStream`。根据 token 流创建了一个`LuceneTokenStreamTokenizer`的实例。`LuceneTokenStreamTokenizer`是一个嵌套的静态类，继承自 LingPipe 的`Tokenizer`类：

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

构造函数存储了`TokenStream`并附加了术语和偏移量属性。在前面的食谱中，我们看到术语和偏移量属性包含 token 字符串，以及输入文本中的 token 起始和结束偏移量。token 偏移量在找到任何 tokens 之前也被初始化为`-1`：

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

我们将实现`nextToken()`方法，并使用 token 流的`incrementToken()`方法从 token 流中获取任何 tokens。我们将使用`OffsetAttribute`来设置 token 的起始和结束偏移量。如果 token 流已经结束，或者`incrementToken()`方法抛出 I/O 异常，我们将结束并关闭`TokenStream`。

`nextWhitespace()`方法有一些局限性，因为`offsetAttribute`聚焦于当前标记，而 LingPipe 分词器会将输入量化为下一个标记和下一个偏移量。这里的一个通用解决方案将是相当具有挑战性的，因为标记之间可能没有明确的空格——可以想象字符 n-grams。因此，`default`字符串仅供参考，以确保清楚表达。该方法如下：

```py
@Override
public String nextWhitespace() {   return "default";
}
```

代码还涵盖了如何序列化分词器，但我们在本步骤中不做详细讨论。

# 使用单元测试评估分词器

我们不会像对 LingPipe 的其他组件一样，用精确度和召回率等度量标准来评估印欧语言分词器。相反，我们会通过单元测试来开发它们，因为我们的分词器是启发式构建的，并预计在示例数据上能完美执行——如果分词器未能正确分词已知案例，那就是一个 BUG，而不是性能下降。为什么会这样呢？有几个原因：

+   许多分词器非常“机械化”，适合于单元测试框架的刚性。例如，`RegExTokenizerFactory`显然是一个单元测试的候选对象，而不是评估工具。

+   驱动大多数分词器的启发式规则是非常通用的，并且不存在以牺牲已部署系统为代价的过拟合训练数据的问题。如果你遇到已知的错误案例，你可以直接修复分词器并添加单元测试。

+   假设标记和空格在语义上是中性的，这意味着标记不会根据上下文而变化。对于我们的印欧语言分词器来说，这并不完全正确，因为它会根据上下文的不同（例如，`3.14 is pi.`中的`.`与句末的`.`）处理`.`。

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

对于基于统计的分词器，使用评估指标可能是合适的；这一点在本章的*为没有空格的语言寻找单词*的步骤中进行了讨论。请参阅第五章中的*句子检测评估*步骤，了解适合的基于跨度的评估技术。

## 如何实现...

我们将跳过代码步骤，直接进入源代码，构建分词器评估器。源代码在`src/com/lingpipe/chapter2/TestTokenizerFactory.java`。请执行以下步骤：

1.  以下代码设置了一个基础的分词器工厂，使用正则表达式——如果你对构建的内容不清楚，请查看该类的 Javadoc：

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

1.  `checkTokens`方法接受`TokenizerFactory`、一个期望的分词结果的`String`数组，以及一个待分词的`String`。具体如下：

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

1.  该方法对错误的容忍度很低，因为如果标记数组的长度不相同，或者某些标记不相等，它会退出程序。一个像 JUnit 这样的单元测试框架会是一个更好的框架，但这超出了本书的范围。你可以查看 `lingpipe.4.1.0`/`src/com/aliasi/test` 中的 LingPipe 单元测试，了解如何使用 JUnit。

1.  `checkTokensAndWhiteSpaces()` 方法检查空格以及标记。它遵循与 `checkTokens()` 相同的基本思路，因此我们将其略去不做解释。

# 修改标记器工厂

在本篇中，我们将描述一个修改标记流中标记的标记器。我们将扩展 `ModifyTokenTokenizerFactory` 类，返回一个经过 13 位旋转的英文字符文本，也叫做 rot-13。Rot-13 是一种非常简单的替换密码，它将一个字母替换为向后 13 个位置的字母。例如，字母 `a` 会被替换成字母 `n`，字母 `z` 会被替换成字母 `m`。这是一个互逆密码，也就是说，应用两次同样的密码可以恢复原文。

## 如何实现……

我们将通过命令行调用 `Rot13TokenizerFactory` 类：

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

你可以看到输入的文本，原本是大小写混合并且是正常的英文，已经转变为其 Rot-13 等价物。你可以看到第二次，我们将 Rot-13 修改过的文本作为输入，返回了原始文本，只是它变成了全小写。

## 它的工作原理是……

`Rot13TokenizerFactory` 扩展了 `ModifyTokenTokenizerFactory` 类。我们将重写 `modifyToken()` 方法，它一次处理一个标记，在这个例子中，它将标记转换为其 Rot-13 等价物。还有一个类似的 `modifyWhiteSpace`（字符串）方法，如果需要，它会修改空格：

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

标记的起始和结束偏移量与底层标记器保持一致。在这里，我们将使用印欧语言标记器作为基础标记器。先通过 `LowerCaseTokenizer` 过滤一次，然后通过 `Rot13Tokenizer` 过滤。

`rot13` 方法是：

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

# 为没有空格的语言找到单词

像中文这样的语言没有单词边界。例如，木卫三是围绕木星运转的一颗卫星，公转周期约为 7 天，来自维基百科，这句话大致翻译为“Ganymede is running around Jupiter's moons, orbital period of about seven days”，这是机器翻译服务在 [`translate.google.com`](https://translate.google.com) 上的翻译。注意到没有空格。

在这种数据中找到标记需要一种非常不同的方法，这种方法基于字符语言模型和我们的拼写检查类。这个方法通过将未标记的文本视为*拼写错误*的文本来编码查找单词，其中*修正*操作是在标记之间插入空格。当然，中文、日文、越南语和其他非单词分隔的书写系统并没有拼写错误，但我们已经在我们的拼写修正类中进行了编码。

## 准备工作

我们将通过去除空格来近似非单词分隔的书写系统。这足以理解这个方法，并且在需要时可以轻松修改为实际的语言。获取大约 100,000 个英文单词并将它们存储在 UTF-8 编码的磁盘中。固定编码的原因是输入假定为 UTF-8——你可以通过更改编码并重新编译食谱来修改它。

我们使用了马克·吐温的《康涅狄格州的国王亚瑟宫廷人》（*A Connecticut Yankee in King Arthur's Court*），从古腾堡项目（[`www.gutenberg.org/`](http://www.gutenberg.org/)）下载。古腾堡项目是一个很好的公共领域文本来源，马克·吐温是位杰出的作家——我们强烈推荐这本书。将你选定的文本放在食谱目录中，或者使用我们的默认设置。

## 如何操作...

我们将运行一个程序，稍微玩一下它，并解释它是如何工作的，使用以下步骤：

1.  输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter2.TokenizeWithoutWhiteSpaces
    Type an Englese sentence (English without spaces like Chinese):
    TheraininSpainfallsmainlyontheplain

    ```

1.  以下是输出：

    ```py
    The rain in Spain falls mainly on the plain
    ```

1.  你可能不会得到完美的输出。马克·吐温从生成它的 Java 程序中恢复正确空格的能力有多强呢？我们来看看：

    ```py
    type an Englese sentence (English without spaces like Chinese)
    NGramProcessLMlm=newNGramProcessLM(nGram);
    NGram Process L Mlm=new NGram Process L M(n Gram);

    ```

1.  之前的方法不是很好，但我们并不公平；让我们使用 LingPipe 的连接源作为训练数据：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar com.lingpipe.cookbook.chapter2.TokenizeWithoutWhiteSpaces data/cookbookSource.txt
    Compiling Spell Checker
    type an Englese sentence (English without spaces like Chinese)
    NGramProcessLMlm=newNGramProcessLM(nGram);
    NGramProcessLM lm = new NGramProcessLM(nGram);

    ```

1.  这是完美的空格插入。

## 它是如何工作的...

在所有的有趣操作中，涉及的代码非常少。酷的是，我们在 第一章，*简单分类器* 中的字符语言模型基础上构建。源代码位于 `src/com/lingpipe/chapter2/TokenizeWithoutWhiteSpaces.java`：

```py
public static void main (String[] args) throws IOException, ClassNotFoundException {
  int nGram = 5;
  NGramProcessLM lm = new NGramProcessLM(nGram);
  WeightedEditDistance spaceInsertingEditDistance
    = CompiledSpellChecker.TOKENIZING;
  TrainSpellChecker trainer = new TrainSpellChecker(lm, spaceInsertingEditDistance);
```

`main()` 方法通过创建 `NgramProcessLM` 开始。接下来，我们将访问一个只添加空格到字符流的编辑距离类。就这样。`Editdistance` 通常是衡量字符串相似度的一个粗略指标，它计算将 `string1` 转换为 `string2` 所需的编辑次数。关于这一点的很多信息可以在 Javadoc `com.aliasi.spell` 中找到。例如，`com.aliasi.spell.EditDistance` 对基础概念有很好的讨论。

### 注意

`EditDistance` 类实现了标准的编辑距离概念，支持或不支持交换操作。不支持交换的距离被称为 Levenshtein 距离，支持交换的距离被称为 Damerau-Levenshtein 距离。

阅读 LingPipe 的 Javadoc；它包含了很多有用的信息，这些信息在本书中没有足够的空间介绍。

到目前为止，我们已经配置并构建了 `TrainSpellChecker` 类。下一步自然是对其进行训练：

```py
File trainingFile = new File(args[0]);
String training = Files.readFromFile(trainingFile, Strings.UTF8);
training = training.replaceAll("\\s+", " ");
trainer.handle(training);
```

我们加载了一个文本文件，假设它是 UTF-8 编码；如果不是，就需要纠正字符编码并重新编译。然后，我们将所有多余的空格替换为单一空格。如果多个空格有特殊意义，这可能不是最好的做法。接着，我们进行了训练，正如我们在 第一章、*简单分类器* 中训练语言模型时所做的那样。

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

下一行编译 `spellChecker`，它将基础语言模型中的所有计数转换为预计算的概率，这样会更快。编译步骤可以将数据写入磁盘，以便后续使用而不需要重新训练；不过，访问 Javadoc 中关于 `AbstractExternalizable` 的部分，了解如何操作。接下来的几行配置 `CompiledSpellChecker` 只考虑插入字符的编辑，并检查是否有完全匹配的字符串，但它禁止删除、替换和变换操作。最后，仅允许进行一次插入。显然，我们正在使用 `CompiledSpellChecker` 的一个非常有限的功能集，但这正是我们需要的——要么插入空格，要么不插入。

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

`CompiledSpellChecker` 和 `WeightedEditDistance` 类的具体机制可以在 Javadoc 或者《*使用编辑距离和语言模型进行拼写纠正*》一书中的 第六章、*字符串比较与聚类* 中得到更好的描述。然而，基本思想是：输入的字符串与刚训练好的语言模型进行比较，从而得到一个分数，表明该字符串与模型的契合度。这个字符串将是一个没有空格的大单词——但请注意，这里没有使用分词器，所以拼写检查器会开始插入空格，并重新评估生成序列的分数。它会保留那些插入空格后，分数提高的序列。

请记住，语言模型是在带有空格的文本上训练的。拼写检查器会尽量在每个可能的位置插入空格，并保持一组“当前最佳”的空格插入结果。最终，它会返回得分最高的编辑序列。

请注意，要完成分词器，必须对修改过空格的文本应用合适的 `TokenizerFactory`，但这留给读者作为练习。

## 还有更多……

`CompiledSpellChecker` 也支持 *n* 最优输出；这允许对文本进行多种可能的分析。在高覆盖率/召回率的场景下，比如研究搜索引擎，它可能有助于应用多个分词方式。此外，可以通过直接扩展 `WeightedEditDistance` 类来调整编辑成本，从而调节系统的表现。

## 另见

如果没有实际提供非英语资源来支持这个配方，那么是没有帮助的。我们使用互联网上可用的资源为研究用途构建并评估了一个中文分词器。我们的中文分词教程详细介绍了这一点。你可以在[`alias-i.com/lingpipe/demos/tutorial/chineseTokens/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/chineseTokens/read-me.html)找到中文分词教程。
