# 第7章. 在概念/人物之间寻找指代关系

在本章中，我们将介绍以下食谱：

+   与文档的命名实体指代

+   向指代中添加代词

+   跨文档指代

+   约翰·史密斯问题

# 简介

指代是人类语言中的基本机制，它允许两个句子谈论同一件事。这对人类交流来说意义重大——它在很大程度上与编程语言中变量名的作用相同，只是作用域的定义规则与代码块不同。在商业上，指代的重要性较小——也许这一章能帮助改变这一点。以下是一个例子：

```py
Alice walked into the garden. She was surprised.
```

`Alice`和“她”之间存在指代关系；这些短语谈论的是同一件事。当我们开始询问一个文档中的Alice是否与另一个文档中的Alice相同，事情就变得非常有趣。

语义消歧一样，指代消解是下一代工业能力。指代消解的挑战促使美国国税局坚持要求有一个社会安全号码，该号码可以明确地识别个人，而与他们的名字无关。许多讨论的技术都是为了帮助在文本数据中跟踪个人和组织，这些数据具有不同的成功程度。

# 与文档的命名实体指代

如[第5章](part0061_split_000.html#page "第5章. 在文本中寻找跨度 – 分块")中所述，“在文本中寻找跨度 – 分块”，LingPipe可以使用各种技术来识别与人物、地点、事物、基因等相对应的正确名词。然而，分块并没有完成这项工作，因为它在两个命名实体相同的情况下无法帮助找到实体。能够说约翰·史密斯与史密斯先生、约翰或甚至完全重复的约翰·史密斯是同一实体，这可能非常有用——有用到这种想法成为我们公司作为婴儿防御承包商时的基础。我们的创新贡献是生成按提及的实体索引的句子，这最终证明是总结关于该实体所说内容的极好方法，尤其是如果映射跨越了语言——我们称之为**基于实体的摘要**。

### 注意

实体基础摘要的想法是在巴德温在宾夕法尼亚大学的一次研究生研讨会上发表演讲后产生的。当时的系主任米奇·马库斯认为，显示提及一个实体的所有句子——包括代词——将是对该实体的极好总结。在某种程度上，这个评论就是为什么LingPipe存在的原因。这导致了巴德温领导宾夕法尼亚大学的DARPA项目，然后创建了Alias-i。学到的教训——与所有人谈论你的想法和研究。

这个食谱将带你了解计算指代的基本知识。

## 准备工作

找到一些叙事文本；我们将使用一个我们知道可以工作的简单示例——共指系统通常需要对领域进行大量的调整。你可以自由选择其他内容，但它必须用英语编写。

## 如何做到这一点…

如同往常，我们将带你通过命令行运行代码，然后深入探讨代码的实际功能。我们出发吧。

1.  我们将从一个简单的文本开始，以说明共指。文件位于`data/simpleCoref.txt`，它包含：

    ```py
    John Smith went to Washington. Mr. Smith is a business man.
    ```

1.  打开命令行和一个Java解释器，重新生成以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.NamedEntityCoreference

    ```

1.  这会导致：

    ```py
    Reading in file :data/simpleCoref.txt 
    Sentence Text=John Smith went to Washington.
         mention text=John Smith type=PERSON id=0
         mention text=Washington type=LOCATION id=1
    Sentence Text=Mr. Smith is a business man.
         mention text=Mr. Smith type=PERSON id=0
    ```

1.  找到了三个命名实体。请注意，输出中有一个`ID`字段。`John Smith`和`Mr. Smith`实体具有相同的ID，`id=0`。这意味着这些短语被认为是共指的。剩余的实体`Washington`具有不同的ID，`id=1`，并且与John Smith / Mr. Smith不共指。

1.  创建你自己的文本文件，将其作为命令行参数提供，并查看会计算什么。

## 它是如何工作的…

LingPipe中的共指代码是在句子检测和命名实体识别之上构建的启发式系统。整体流程如下：

1.  分词文本。

1.  在文档中检测句子，对于每个句子，在句子中按从左到右的顺序检测命名实体，并对每个命名实体执行以下任务：

    1.  创建一个提及。提及是命名实体的单个实例。

    1.  提及可以添加到现有的提及链中，或者它们可以开始自己的提及链。

    1.  尝试将提及解析到已创建的提及链中。如果找到唯一匹配项，则将提及添加到提及链中；否则，创建一个新的提及链。

代码位于`src/com/lingpipe/cookbook/chapter7/NamedEntityCoreference.java`。`main()`方法首先设置本食谱的各个部分，从分词工厂开始，然后是句子块处理器，最后是命名实体块处理器：

```py
public static void main(String[] args) 
    throws ClassNotFoundException, IOException {
  String inputDoc = args.length > 0 ? args[0] 
        : "data/simpleCoref.txt";
  System.out.println("Reading in file :" 
      + inputDoc);
  TokenizerFactory mTokenizerFactory
    = IndoEuropeanTokenizerFactory.INSTANCE;
  SentenceModel sentenceModel
    = new IndoEuropeanSentenceModel();
  Chunker sentenceChunker 
    = new SentenceChunker(mTokenizerFactory,sentenceModel);
   File modelFile  
    = new File("models/ne-en-news-"
      + "muc6.AbstractCharLmRescoringChunker");
  Chunker namedEntChunker
    = (Chunker) AbstractExternalizable.readObject(modelFile);
```

现在，我们已经为食谱设置了基本的基础设施。接下来是一个特定的共指类：

```py
MentionFactory mf = new EnglishMentionFactory();
```

`MentionFactory`类从短语和类型创建提及——当前源名为`entities`。接下来，使用`MentionFactory`作为参数创建共指类：

```py
WithinDocCoref coref = new WithinDocCoref(mf);
```

`WithinDocCoref`类封装了计算共指的所有机制。从[第5章](part0061_split_000.html#page "第5章. 文本中的跨度查找 – 块处理")，*文本中的跨度查找 - 块处理*，你应该熟悉获取文档文本、检测句子以及迭代应用命名实体块处理器的句子代码：

```py
File doc = new File(inputDoc);
String text = Files.readFromFile(doc,Strings.UTF8);
Chunking sentenceChunking
  = sentenceChunker.chunk(text);
Iterator sentenceIt 
  = sentenceChunking.chunkSet().iterator();

for (int sentenceNum = 0; sentenceIt.hasNext(); ++sentenceNum) {
  Chunk sentenceChunk = (Chunk) sentenceIt.next();
  String sentenceText 
    = text.substring(sentenceChunk.start(),
          sentenceChunk.end());
  System.out.println("Sentence Text=" + sentenceText);

  Chunking neChunking = namedEntChunker.chunk(sentenceText);
```

在当前句子的上下文中，句子中的命名实体按照从左到右的顺序迭代，就像它们被阅读时的顺序一样。我们知道这一点是因为`ChunkingImpl`类返回的块是按照它们被添加的顺序返回的，而我们的`HMMChunker`按照从左到右的顺序添加它们：

```py
Chunking neChunking = namedEntChunker.chunk(sentenceText);
for (Chunk neChunk : neChunking.chunkSet()) {
```

以下代码从信息块中获取类型和短语信息——但不包括偏移信息，并创建一个提及：

```py
String mentionText
  = sentenceText.substring(neChunk.start(),
          neChunk.end());
String mentionType = neChunk.type();
Mention mention = mf.create(mentionText,mentionType);
```

下一行运行核心词引用与提及及其所在的句子，并返回其 ID：

```py
int mentionId = coref.resolveMention(mention,sentenceNum);

System.out.println("     mention text=" + mentionText
            + " type=" + mentionType
            + " id=" + mentionId);
```

如果提及被解析到现有实体，它将具有该 ID，正如我们在 Mr. Smith 例子中看到的。否则，它将获得一个独特的 ID，并且自身也可以作为后续提及的前体。

这涵盖了在文档内运行核心词引用的机制。接下来的配方将涵盖对这个类的修改。下一个配方将添加代词并提供引用。

# 添加代词到核心词引用

前面的配方处理了命名实体之间的核心词引用。此配方将添加代词到其中。

## 如何做…

此配方将使用交互式版本来帮助您探索核心词算法的特性。系统非常依赖于命名实体检测的质量，因此请使用 HMM 可能会正确处理示例。这是在 90 年代的《华尔街日报》文章上训练的。

1.  将你的控制台准备好，并输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.Coreference

    ```

1.  在生成的命令提示符中，输入以下内容：

    ```py
    Enter text followed by new line
    >John Smith went to Washington. He was a senator.
    Sentence Text=John Smith went to Washington.
    mention text=John Smith type=PERSON id=0
    mention text=Washington type=LOCATION id=1
    Sentence Text= He was a senator.
    mention text=He type=MALE_PRONOUN id=0
    ```

1.  `He` 和 `John Smith` 之间的共享 ID 表明两者之间的核心词引用。接下来将有更多示例，并附有注释。请注意，每个输入都被视为一个独立的文档，具有单独的 ID 空间。

1.  如果代词没有解析到命名实体，它们将得到索引 `-1`，如下所示：

    ```py
    >He went to Washington.
    Sentence Text= He went to Washington.
    mention text=He type=MALE_PRONOUN id=-1
    mention text=Washington type=LOCATION id=0
    ```

1.  以下案例也导致 `id` 的值为 `-1`，因为在先前的上下文中没有一个人，而是两个人。这被称为失败的唯一性预设：

    ```py
    >Jay Smith and Jim Jones went to Washington. He was a senator.
    Sentence Text=Jay Smith and Jim Jones went to Washington.
    mention text=Jay Smith type=PERSON id=0
    mention text=Jim Jones type=PERSON id=1
    mention text=Washington type=LOCATION id=2
    Sentence Text= He was a senator.
    mention text=He type=MALE_PRONOUN id=-1
    ```

1.  以下代码显示 `John Smith` 也可以解析到女性代词。这是因为没有关于哪些名字表示哪些性别数据。它可以添加，但通常情况下，上下文会消除歧义。`John` 可能是一个女性名字。关键在于代词将消除性别歧义，而随后的男性代词将无法匹配：

    ```py
    Frank Smith went to Washington. She was a senator. 
    Sentence Text=Frank Smith went to Washington.
         mention text=Frank Smith type=PERSON id=0
         mention text=Washington type=LOCATION id=1
    Sentence Text=She was a senator.
         mention text=She type=FEMALE_PRONOUN id=0
    ```

1.  性别分配将阻止错误性别引起的引用。以下代码中的 `He` 代词解析到 ID `-1`，因为唯一的人解析到了一个女性代词：

    ```py
    John Smith went to Washington. She was a senator. He is now a lobbyist.
    Sentence Text=John Smith went to Washington.
         mention text=John Smith type=PERSON id=0
         mention text=Washington type=LOCATION id=1
    Sentence Text=She was a senator.
         mention text=She type=FEMALE_PRONOUN id=0
    Sentence Text=He is now a lobbyist.
         mention text=He type=MALE_PRONOUN id=-1
    ```

1.  核心词引用也可以在句子内部发生：

    ```py
    >Jane Smith knows her future.
    Sentence Text=Jane Smith knows her future.
         mention text=Jane Smith type=PERSON id=0
         mention text=her type=FEMALE_PRONOUN id=0
    ```

1.  在解析提及时，提及的顺序（按最近提及排序）很重要。在以下代码中，`He` 被解析到 `James`，而不是 `John`：

    ```py
    John is in this sentence. Another sentence about nothing. James is in this sentence. He is here.
    Sentence Text=John is in this sentence.
         mention text=John type=PERSON id=0
    Sentence Text=Another sentence about nothing.
    Sentence Text=James is in this sentence.
         mention text=James type=PERSON id=1
    Sentence Text=He is here.
         mention text=He type=MALE_PRONOUN id=1
    ```

1.  同样的效果也发生在命名实体提及上。`Mr. Smith` 实体解析到最后一次提及：

    ```py
    John Smith is in this sentence. Random sentence. James Smith is in this sentence. Mr. Smith is mention again here.
    Sentence Text=John Smith is in this sentence.
         mention text=John Smith type=PERSON id=0
    Sentence Text=Random sentence.
         mention text=Random type=ORGANIZATION id=1
    Sentence Text=James Smith is in this sentence.
         mention text=James Smith type=PERSON id=2
    Sentence Text=Mr. Smith is mention again here.
         mention text=Mr. Smith type=PERSON id=2
    ```

1.  如果有太多的间隔句子，`John` 和 `James` 之间的区别就会消失：

    ```py
    John Smith is in this sentence. Random sentence. James Smith is in this sentence. Random sentence. Random sentence. Mr. Smith is here.
    Sentence Text=John Smith is in this sentence.
         mention text=John Smith type=PERSON id=0
    Sentence Text=Random sentence.
         mention text=Random type=ORGANIZATION id=1
    Sentence Text=James Smith is in this sentence.
         mention text=James Smith type=PERSON id=2
    Sentence Text=Random sentence.
         mention text=Random type=ORGANIZATION id=1
    Sentence Text=Random sentence.
         mention text=Random type=ORGANIZATION id=1
    Sentence Text=Mr. Smith is here.
         mention text=Mr. Smith type=PERSON id=3
    ```

前面的例子旨在演示文档内核心词引用系统的特性。

## 它是如何工作的…

添加代词的代码更改很简单。此配方的代码位于 `src/com/lingpipe/cookbook/chapter7/Coreference.java`。此配方假设你已经理解了前面的配方，所以它只涵盖了添加代词提及的部分：

```py
Chunking mentionChunking
  = neChunker.chunk(sentenceText);
Set<Chunk> chunkSet = new TreeSet<Chunk> (Chunk.TEXT_ORDER_COMPARATOR);
chunkSet.addAll(mentionChunking.chunkSet());
```

我们添加了来自多个来源的`Mention`对象，因此不再保证元素顺序。相应地，我们创建了`TreeSet`和适当的比较器，并添加了所有来自`neChunker`的切分。

接下来，我们将添加男性和女性代词：

```py
addRegexMatchingChunks(MALE_EN_PRONOUNS,"MALE_PRONOUN",
        sentenceText,chunkSet);
addRegexMatchingChunks(FEMALE_EN_PRONOUNS,"FEMALE_PRONOUN",
        sentenceText,chunkSet);
```

`MALE_EN_PRONOUNS`常量是一个正则表达式，`Pattern`：

```py
static Pattern MALE_EN_PRONOUNS =   Pattern.compile("\\b(He|he|Him|him)\\b");
```

以下代码行显示了`addRegExMatchingChunks`子例程。它基于正则表达式匹配添加块，并移除重叠的现有HMM派生的块：

```py
static void addRegexMatchingChunks(Pattern pattern, String type, String text, Set<Chunk> chunkSet) {

  java.util.regex.Matcher matcher = pattern.matcher(text);

  while (matcher.find()) {
    Chunk regexChunk 
    = ChunkFactory.createChunk(matcher.start(),
            matcher.end(),
            type);
    for (Chunk chunk : chunkSet) {
    if (ChunkingImpl.overlap(chunk,regexChunk)) {
      chunkSet.remove(chunk);
    }
    }
  chunkSet.add(regexChunk);
  }
```

其中一个复杂的问题是，`MALE_PRONOUN`和`FEMALE_PRONOUN`代词的类型将用于与`PERSON`实体匹配，其结果是解析集确定了解析到的实体的性别。

除了这些，代码应该非常熟悉我们的标准I/O循环，在命令提示符中运行交互。

## 参见

系统背后的算法基于Baldwin的博士论文。该系统被称为CogNIAC，这项工作始于20世纪90年代中期，并不是当前最先进的共指系统。更现代的方法可能会使用机器学习框架来利用Baldwin方法和其他许多特征生成特征，并用于开发性能更好的系统。关于该系统的论文可在[http://www.aclweb.org/anthology/W/W97/W97-1306.pdf](http://www.aclweb.org/anthology/W/W97/W97-1306.pdf)找到。

# 跨文档共指

跨文档共指（XDoc）将单个文档的`id`空间扩展到更大的宇宙。这个宇宙通常包括其他已处理的文档和已知实体的数据库。虽然标注很简单，但只需要将文档范围ID交换为宇宙范围ID。XDoc的计算可能相当困难。

这个配方将告诉我们如何使用在多年部署此类系统过程中开发的轻量级XDoc实现。我们将为可能想要扩展/修改代码的人提供代码概述——但内容很多，配方相当密集。

输入是XML格式，其中每个文件可以包含多个文档：

```py
<doc id="1">
<title/>
<content>
Breck Baldwin and Krishna Dayanidhi wrote a book about LingPipe. 
</content>
</doc>

<doc id="2">
<title/>
<content>
Krishna Dayanidhi is a developer. Breck Baldwin is too. 
</content>
</doc>

<doc id="3">
<title/>
<content>
K-dog likes to cook as does Breckles.
</content>
</doc>
```

目标是生成标注，其中Breck Baldwin的提及在文档中与Krishna共享相同的ID。请注意，在最后一份文档中，两者都提到了他们的昵称。

XDoc的一种非常常见的扩展是将已知实体的**数据库**（**DB**）链接到这些实体的文本提及。这弥合了结构化数据库和非结构化数据（文本）之间的差距，许多人认为这是商业智能/客户声音/企业知识管理领域的下一个大趋势。我们已经构建了将基因/蛋白质数据库链接到MEDLINE摘要和感兴趣人员名单链接到自由文本的系统，等等。数据库还为人编者提供了一个自然的方式来控制XDoc的行为。

## 如何实现...

本菜谱的所有代码都在`com.lingpipe.cookbook.chapter7.tracker`包中。

1.  获取对您的IDE的访问权限并运行`RunTracker`或在命令行中输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.tracker.RunTracker

    ```

1.  屏幕将滚动显示文档的分析，但我们将转到指定的输出文件并检查它。在您最喜欢的文本编辑器中打开`cookbook/data/xDoc/output/docs1.xml`。除非您的编辑器自动格式化XML，否则您将看到示例输出的糟糕格式版本——Firefox网络浏览器在渲染XML方面做得相当不错。输出应该看起来像这样：

    ```py
    <docs>
    <doc id="1">
    <title/>
    <content>
    <s index="0">
    <entity id="1000000001" type="OTHER">Breck Baldwin</entity> and <entity id="1000000002" type="OTHER">Krishna Dayanidhi</entity> wrote a book about <entity id="1000000003" type="OTHER">LingPipe.</entity>
    </s>
    </content>
    </doc>
    <doc id="2">
    <title/>
    <content><s index="0">
    <entity id="1000000002" type="OTHER">Krishna Dayanidhi</entity> is a developer.
    </s>
    <s index="1"><entity id="1000000001" type="OTHER">Breck Baldwin</entity> is too.
    </s>
    </content>
    </doc>
    <doc id="3"><title/><content><s index="0">K-dog likes to cook as does <entity id="1000000004" start="28" type="OTHER">Breckles</entity>.</s></content></doc>
    </docs>
    ```

1.  `Krishna`在前两个文档中被识别为具有共享ID，`1000000002`，但昵称`K-dog`根本未被识别。`Breck`在所有三个文档中都被识别，但由于第三次提及时`Breckles`的ID与前两次提及的ID不同，系统不认为它们是同一实体。

1.  接下来，我们将使用字典形式的数据库来提高通过昵称提及作者时的识别率。在`data/xDoc/author-dictionary.xml`中有一个字典；它看起来像这样：

    ```py
    <dictionary>
    <entity canonical="Breck Baldwin" id="1" speculativeAliases="0" type="MALE">
      <alias xdc="1">Breck Baldwin</alias>
      <alias xdc="1">Breckles</alias>
      <alias xdc="0">Breck</alias>
    </entity>

    <entity canonical="Krishna Dayanidhi" id="2" speculativeAliases="0" type="MALE">
      <alias xdc="1">Krishna Dayanidhi</alias>
      <alias xdc="1">K-Dog</alias>
      <alias xdc="0">Krishna</alias> 
    </entity>
    ```

1.  上述字典包含作者的两个昵称，以及他们的名字。具有`xdc=1`值的别名将用于跨文档链接实体。`xdc=0`值仅适用于文档内部。所有别名都将通过字典查找来识别命名实体。

1.  运行以下命令，指定实体字典或IDE等效项：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.tracker.RunTracker data/xDoc/author-dictionary.xml

    ```

1.  `xDoc/output/docs1.xml`中的输出与上一次运行的结果非常不同。首先，请注意，现在的ID与我们指定的字典文件中的ID相同：`Breck`为`1`，`Krishna`为`2`。这是结构化数据库（如字典的性质）与无结构文本之间的链接。其次，请注意，我们的昵称都被正确识别并分配到了正确的ID。第三，请注意，类型现在是`MALE`而不是`OTHER`：

    ```py
    <docs>
    <doc id="1">
    <title/>
    <content>
    <s index="0">
    <entity id="1" type="MALE">Breck Baldwin</entity> and <entity id="2" type="MALE">Krishna Dayanidhi</entity> wrote a book about <entity id="1000000001" type="OTHER">LingPipe.</entity>
    </s>
    </content>
    </doc>
    <doc id="2">
    <title/>
    <content>
    <s index="0">
    <entity id="2" start="0" type="MALE">K-dog</entity> likes to cook as does <entity id="1" start="28" type="MALE">Breckles</entity>.
    </s>
    </content>
    </doc>
    </docs>
    ```

这是对如何运行XDoc的非常快速的介绍。在下一节中，我们将看到它是如何工作的。

## 它是如何工作的...

到目前为止，我们一直试图保持代码简单、直接且易于理解，无需深入研究大量的源代码。这个菜谱更复杂。支撑这个菜谱的代码无法完全放入分配的解释空间中。本说明假设您将自行探索整个类，并且会参考本书中的其他菜谱进行解释。我们提供这个菜谱是因为XDoc核心参照问题非常有趣，我们现有的基础设施可能有助于他人探索这一现象。欢迎来到泳池的深处。

### 批处理生命周期

整个过程由`RunTracker.java`类控制。`main()`方法的整体流程如下：

1.  通过 `Dictionary` 读取已知实体数据库，这些实体将成为命名实体识别的来源，以及从别名到字典条目的已知映射。别名带有关于是否应通过 `xdc=1` 或 `xdc=0` 标志在文档间匹配实体的说明。

1.  设置 `EntityUniverse`，这是全局数据结构，包含文本中找到的以及从已知实体字典中提到的实体的 ID。

1.  设置文档内核心参照所需的内容——例如分词器、句子检测器和命名实体检测器。在 POS 标记器和一些词频统计方面，这会变得有些复杂。

1.  有一个布尔值控制是否添加推测性实体。如果这个布尔值为 `true`，则意味着我们将更新我们的跨文档实体宇宙，包括我们之前从未见过的那些。将这个集合设置为 `true` 是一个更艰巨的任务，需要可靠地计算。

1.  所有提到的配置都用于创建一个 `Tracker` 对象。

1.  然后，`main()` 方法读取要处理的文档，将它们交给 `Tracker` 对象进行处理，并将它们写入磁盘。`Tracker.processDocuments()` 方法的重大步骤如下：

    1.  从 XML 格式的文档集中提取单个文档。

    1.  对于每个文档，应用 `processDocument()` 方法，该方法在文档内使用字典帮助找到实体以及命名实体检测器，并返回 `MentionChain[]`。然后，将个别提及的链与实体宇宙进行解析，以更新文档级别的 ID 为实体宇宙 ID。最后一步是将文档写入磁盘，带有实体宇宙 ID。

关于 `RunTracker` 的介绍就到这里；其中没有任何内容是你在这个书籍的上下文中无法处理的。在接下来的章节中，我们将讨论 `RunTracker` 使用的各个组件。

#### 设置实体宇宙

实体宇宙 `EntityUniverse.java` 是文档/数据库集中提到的全局实体的内存表示。实体宇宙还包含对这些实体的各种索引，支持对单个文档计算 XDoc。

字典用已知实体初始化 `EntityUniverse` 文件，随后处理的文档对这些实体敏感。XDoc 算法试图在创建新实体之前与现有实体合并，因此字典实体是这些实体提及的强大吸引物。

每个实体由一个唯一的长期 ID、一组划分为四个单独列表的别名以及一个类型（人物、地点等）组成。实体是否在用户定义的字典中，以及是否允许将推测性提及添加到实体中，也都有说明。`toString()` 方法将实体列出如下：

```py
id=556 type=ORGANIZATION userDefined=true allowSpec=false user XDC=[Nokia Corp., Nokia] user non-XDC=[] spec XDC=[] spec non-XDC
=[]
```

全局数据结构如下：

```py
    private long mLastId = FIRST_SYSTEM_ID;
```

实体需要唯一的 ID，我们有一个约定，即 `FIRST_SYSTEM_ID` 的值是一个大整数，例如 `1,000,000`。这为用户提供了空间（ID < 1,000,000），以便在不与系统发现的实体冲突的情况下添加新实体。

我们将为追踪器实例化一个分词器：

```py
    private final TokenizerFactory mTokenizerFactory;
```

存在着一个从唯一实体 ID 到实体的全局映射：

```py
    private final Map<Long,Entity> mIdToEntity
        = new HashMap<Long,Entity>();
```

另一个重要的数据结构是从别名（短语）到具有该别名的实体的映射—`mXdcPhraseToEntitySet`。只有那些可能用于寻找跨文档同指可能匹配的候选短语才会被添加到这里。从字典中，`xdc=1` 的别名会被添加：

```py
private final ObjectToSet<String,Entity> mXdcPhraseToEntitySet
        = new ObjectToSet<String,Entity>();
```

对于推测性发现的别名，如果一个别名至少有两个标记并且尚未出现在另一个实体上，它将被添加到这个集合中。这反映了一种试图尽可能不分割实体的启发式方法。这个逻辑相当复杂，超出了本教程的范围。你可以参考 `EntityUniverse.createEntitySpeculative` 和 `EntityUniverse.addPhraseToEntity` 来查看代码。

为什么有些别名在寻找候选实体时不被使用？考虑一下，`George` 在 `EntityUniverse` 中几乎没有描述性内容来区分实体，但 `George H.W. Bush` 则有更多信息可供利用。

#### ProcessDocuments() 和 ProcessDocument()

有趣的部分开始于 `Tracker.processDocuments()` 方法，它调用每个文档的 XML 解析，然后逐步调用 `processDocument()` 方法。前者的代码很简单，所以我们将继续到 `processDocument()` 方法中更具体的任务工作部分：

```py
public synchronized OutputDocument processDocument(
            InputDocument document) {

       WithinDocCoref coref
            = new WithinDocCoref(mMentionFactory);

        String title = document.title();
        String content = document.content();

        List<String> sentenceTextList = new ArrayList<String>();
        List<Mention[]> sentenceMentionList 
        = new ArrayList<Mention[]>();

        List<int[]> mentionStartList = new ArrayList<int[]>();
        List<int[]> mentionEndList = new ArrayList<int[]>();

        int firstContentSentenceIndex
            = processBlock(title,0,
                           sentenceTextList,
                           sentenceMentionList,
                           mentionStartList,mentionEndList,
                           coref);

        processBlock(content,firstContentSentenceIndex,
                     sentenceTextList,
                     sentenceMentionList,
                     mentionStartList,mentionEndList,
                     coref);

        MentionChain[] chains = coref.mentionChains();
```

我们使用了一种支持区分文档标题和正文的文档格式。如果标题和正文的大小写不同，这是一个好主意，就像新闻稿中常见的那样。`chains` 变量将包含来自文本标题和正文的链，以及它们之间可能存在的同指。`mentionStartList` 和 `mentionEndList` 数组将使得在方法后面的某个时候能够将文档作用域的 ID 与实体宇宙作用域的 ID 对齐：

```py
Entity[] entities  = mXDocCoref.xdocCoref(chains);
```

#### 计算XDoc

XDoc 代码是经过许多小时手动调整算法以适应新闻风格数据的结果。它已经在包含 20,000 篇文档的数据集上运行，并且被设计成非常积极地支持字典条目。代码还试图防止**短路**，这发生在明显不同的实体被合并在一起时。如果你错误地将芭芭拉·布什和乔治·布什在全局数据库中标记为同指，那么你将得到令人尴尬的糟糕结果，用户会看到。

另一种错误是在全局存储中有两个实体，而只需要一个。这是一种类似于 *超人/克拉克·肯特问题* 的情况，也可以适用于同一名称的多个提及。

我们将从顶层代码开始：

```py
    public Entity[] xdocCoref(MentionChain[] chains) { Entity[]
        entities = new Entity[chains.length];

        Map<MentionChain,Entity> chainToEntity
            = new HashMap<MentionChain,Entity>();
        ObjectToSet<Entity,MentionChain> entityToChainSet
            = new ObjectToSet<Entity,MentionChain>();

        for (MentionChain chain : chains)
            resolveMentionChain((TTMentionChain) chain,
                                chainToEntity, entityToChainSet);

        for (int i = 0; i < chains.length; ++i) {
            TTMentionChain chain = (TTMentionChain) chains[i];
            Entity entity = chainToEntity.get(chain);

            if (entity != null) {
                if (Tracker.DEBUG) {
                    System.out.println("XDOC: resolved to" + entity);
         Set chainSetForEntity = entityToChainSet.get(entity);
                    if (chainSetForEntity.size() > 1) 
                        System.out.println("XDOC: multiple chains resolved to same entity " + entity.id());
                }
                entities[i] = entity;
                if (entity.addSpeculativeAliases()) 
                    addMentionChainToEntity(chain,entity);
            } else {
                Entity newEntity = promote(chain);
                entities[i] = newEntity;
            }
        }
        return entities;
    }
```

文档有一个提及链列表，每个提及链要么被添加到现有实体中，要么提及链被提升为新的实体。提及链必须包含一个非代词提及，这在文档内的共指级别进行处理。

每处理一个提及链，都会更新三个数据结构：

+   `Entity[]` 实体由 `xdocCoref` 方法返回，以支持文档的行内注释。

+   `Map<MentionChain,Entity> chainToEntity` 将提及链映射到实体。

+   `ObjectToSet<Entity,MentionChain> entityToChainSet` 是 `chainToEntity` 的逆映射。可能同一文档中的多个链会被映射到同一个实体，因此这个数据结构对这种可能性很敏感。这个版本的代码允许这种情况发生——实际上，XDoc 正在将文档内的解析作为副作用设置起来。

简单来说，如果找到了实体，那么 `addMentionChainToEntity()` 方法会将提及链中的任何新信息添加到实体中。新信息可以包括新的别名和类型变化（即，由于一个消歧代词的引用，一个人被移动到男性或女性）。如果没有找到实体，那么提及链将进入 `promote()`，在实体宇宙中创建一个新实体。我们将从 `promote()` 开始。

#### `promote()` 方法

实体宇宙是一个极简的数据结构，仅跟踪短语、类型和 ID。`TTMentionChain` 类是对特定文档中提及的更复杂表示：

```py
private Entity promote(TTMentionChain chain) {
    Entity entity
        = mEntityUniverse.createEntitySpeculative(
          chain.normalPhrases(),
                            chain.entityType());
        if (Tracker.DEBUG)
            System.out.println("XDOC: promoted " + entity);
        return entity;
    }
```

调用 `mEntityUniverse.createEntitySpeculative` 只需要链的短语（在这种情况下，是已经被转换为小写并且所有空白序列都转换为单个空格的规范化短语）和实体的类型。不会记录提及链来自的文档、计数或其他可能有用的信息。这是为了使内存表示尽可能小。如果需要找到实体被提及的所有句子或文档（这是一个常见任务），那么必须将实体 ID 的映射存储在其他地方。XDoc 运行后生成的文档的 XML 表示是一个开始解决这些需求的自然地方。

#### `createEntitySpeculative()` 方法

创建一个推测性找到的新实体只需要确定其别名中哪些是连接提及链的好候选。那些适合跨文档共指的进入 `xdcPhrases` 集合，其他则进入 `nonXdc` 短语集合：

```py
    public Entity createEntitySpeculative(Set<String> phrases,
                                          String entityType) {
        Set<String> nonXdcPhrases = new HashSet<String>();
        Set<String> xdcPhrases = new HashSet<String>();
        for (String phrase : phrases) {
            if (isXdcPhrase(phrase,hasMultiWordPhrases)) 
                xdcPhrases.add(phrase);
            else
                nonXdcPhrases.add(phrase);
        }
        while (mIdToEntity.containsKey(++mLastId)) ; // move up to next untaken ID
        Entity entity = new Entity(mLastId,entityType,
                                  null,null,xdcPhrases,nonXdcPhrases);
        add(entity);
        return entity;
    }
```

`boolean` 方法 `XdcPhrase()` 在 XDoc 过程中起着关键作用。当前方法支持一个非常保守的关于什么是好的 XDoc 短语的观点。直观地说，在新闻稿的领域，像 `he`、`Bob` 和 `John Smith` 这样的短语是关于谈论的独特个体的较差指标。好的短语可能是 `Breckenridge Baldwin`，因为这很可能是一个独特的名字。关于这里发生的事情有许多复杂的理论，参见刚性指示符 ([http://en.wikipedia.org/wiki/Rigid_designator](http://en.wikipedia.org/wiki/Rigid_designator))。接下来的几行代码对2000年的哲学思想进行了粗暴的践踏：

```py
public boolean isXdcPhrase(String phrase,
          boolean hasMultiWordPhrase) {

    if (mXdcPhraseToEntitySet.containsKey(phrase)) {
        return false;
    }  
    if (phrase.indexOf(' ') == -1 && hasMultiWordPhrase) {
        return false;
    }
    if (PronounChunker.isPronominal(phrase)) {
        return false;
   }
    return true;
}
```

这种方法试图识别 XDoc 的坏短语而不是好短语。推理如下：

+   **短语已经与一个实体相关联**：这强制了一个假设，即世界上只有一个约翰·史密斯。这对于情报收集应用来说效果非常好，分析师们几乎没有困难地分辨出 `John Smith` 的情况。您可以在本章末尾的 *The John Smith problem* 菜单中了解更多关于此的信息。

+   **短语仅由一个单词组成，并且与提及链或实体相关联的短语是多个单词的**：这假设较长的单词更适合 XDoc。请注意，实体创建的不同顺序可能导致具有多词别名的实体上的单词短语 `xdc` 为 `true`。

+   **短语是代词**：这是一个相当安全的假设，除非我们处于宗教文本中，其中句子中间大写的 `He` 或 `Him` 指的是上帝。

一旦知道了 `xdc` 和 `nonXdc` 短语的集合，然后实体就被创建。请参考 `Entity.java` 的源代码来了解实体是如何创建的。

然后，实体被创建，并且 `add` 方法更新 `EntityUniverse` 文件中 `xdc` 短语到实体 ID 的映射：

```py
public void add(Entity e) {
        if (e.id() > mLastId)
            mLastId = e.id();
        mIdToEntity.put(new Long(e.id()),e);
        for (String phrase : e.xdcPhrases()) {
            mXdcPhraseToEntitySet.addMember(phrase,e);
        }
    }
```

`EntityUniverse` 文件中的全局 `mXdcPhraseToEntitySet` 变量是用于在 `xdcEntitiesToPhrase()` 中查找 XDoc 的候选实体的关键。

#### XDocCoref.addMentionChainToEntity() 实体

返回到 `XDocCoref.xdocCoref()` 方法，我们已经介绍了如何通过 `XDocCoref.promote()` 创建一个新的实体。接下来要介绍的是，当提及链解析为现有实体时会发生什么，即 `XDocCoref.addMentionChainToEntity()`。为了添加推测性提及，实体必须允许由 `Entity.allowSpeculativeAliases()` 方法提供的推测性找到的提及。这是在用户定义实体中讨论的用户定义字典实体的一个特性。如果允许推测性实体，那么提及链将被添加到实体中，同时考虑到它们是否是 `xdc` 短语：

```py
private void addMentionChainToEntity(TTMentionChain chain, 
                Entity entity) {
    for (String phrase : chain.normalPhrases()) {
             mEntityUniverse.addPhraseToEntity(normalPhrase,
                entity);
        }
    }
```

添加提及链到实体中唯一可能带来的变化是添加一个新的短语。这些附加短语会按照在提及链提升过程中所做的方式，被分类为是否是`xdc`。

到目前为止，我们已经讨论了从文档中提及链要么提升为推测性实体，要么与 `EntityUniverse` 中的现有实体合并的基本方法。接下来，我们将看看 `XDocCoref.resolveMentionChain()` 中是如何进行解决的。

#### `XDocCoref.resolveMentionChain()` 实体

`XDocCoref.resolveMentionChain()` 方法组装一组可能匹配正在解决的提及链的实体，然后通过调用 `XDocCoref.resolveCandates()` 尝试找到唯一的实体：

```py
private void resolveMentionChain(TTMentionChain chain, Map<MentionChain,Entity> chainToEntity, ObjectToSet<Entity,MentionChain> entityToChainSet) {
        if (Tracker.DEBUG)
            System.out.println("XDOC: resolving mention chain " 
          + chain);
        int maxLengthAliasOnMentionChain = 0;
        int maxLengthAliasResolvedToEntityFromMentionChain = -1;
        Set<String> tokens = new HashSet<String>();
        Set<Entity> candidateEntities = new HashSet<Entity>();
        for (String phrase : chain.normalPhrases()) {
        String[] phraseTokens = mEntityUniverse.normalTokens(phrase);
         String normalPhrase 
      = mEntityUniverse.concatenateNormalTokens(phraseTokens);
         for (int i = 0; i < phraseTokens.length; ++i) {
                    tokens.add(phraseTokens[i]);
    }
         int length = phraseTokens.length;       
         if (length > maxLengthAliasOnMentionChain) {
                maxLengthAliasOnMentionChain = length;
        }
         Set<Entity> matchingEntities
           = mEntityUniverse.xdcEntitiesWithPhrase(phrase);
         for (Entity entity : matchingEntities) {
           if (null != TTMatchers.unifyEntityTypes(
            chain.entityType(),
            entity.type())) {
               if (maxLengthAliasResolvedToEntityFromMentionChain < length) 
                        maxLengthAliasResolvedToEntityFromMentionChain = length;
  candidateEntities.add(entity);
}
}
}   
resolveCandidates(chain,
                  tokens,
                  candidateEntities,
                          maxLengthAliasResolvedToEntityFromMentionChain == maxLengthAliasOnMentionChain,
                          chainToEntity,
                          entityToChainSet);}
```

代码通过使用 `EntityUniverse.xdcEntitiesWithPhrase()` 在实体宇宙中进行查找来组装一组实体。在将实体添加到 `candidateEntities` 之前，必须确保返回的类型与由 `TTMatchers.unifyEntityTypes` 确定的提及链的类型一致。这样，就不会将地点“华盛顿”解析为人物“华盛顿”。为了确定提及链上最长的别名是否匹配了实体，进行了一些记录。

#### `resolveCandidates()` 方法

`resolveCandidates()` 方法捕捉到对于文档内和 XDoc 共指都成立的一个关键假设——这个明确的引用是唯一解决的基础。在文档内的情况下，人类有这种问题的例子是句子，“鲍勃和乔一起工作。他掉进了脱粒机。”谁是指的“他”？一个单数指称词有一个唯一先行词的语言学期望被称为唯一性预设。以下是一个 XDoc 例子：

+   **Doc1**：约翰·史密斯是《波卡洪塔斯》中的一个角色

+   **Doc2**：约翰·史密斯是董事长或总经理

+   **Doc3**：约翰·史密斯受人钦佩

Doc3 中的 `John Smith` 与哪个 `John Smith` 相匹配？也许都不是。这个软件中的算法要求应该有一个单一的实体能够通过匹配标准。如果有多个或零个，则创建一个新的实体。实现方式如下：

```py
        private void resolveCandidates(TTMentionChain chain,
                                   Set<String> tokens,
                                   Set<Entity> candidateEntities,
                               boolean resolvedAtMaxLength,
                               Map<MentionChain,Entity> chainToEntity,
                               ObjectToSet<Entity,MentionChain> entityToChainSet) {
        filterCandidates(chain,tokens,candidateEntities,resolvedAtMaxLength);
        if (candidateEntities.size() == 0)
            return;
        if (candidateEntities.size() == 1) {
            Entity entity = Collections.<Entity>getFirst(candidateEntities);
            chainToEntity.put(chain,entity);
            entityToChainSet.addMember(entity,chain);
            return;
        }
        // BLOWN Uniqueness Presupposition; candidateEntities.size() > 1
        if (Tracker.DEBUG)
            System.out.println("Blown UP; candidateEntities.size()=" + candidateEntities.size());
    }
```

`filterCandidates` 方法消除了所有因各种语义原因而失败的候选实体。在实体宇宙中，只有当存在唯一可能的解决方案时，才会发生与实体的共指。在候选实体过多（多于一个）或过少（零）之间没有区别。在一个更高级的系统里，可以通过`context`尝试进一步消除过多实体（多个实体）的歧义。

这是 XDoc 代码的核心。其余的代码使用 `xdocCoref` 方法返回的与实体宇宙相关的索引标记文档，我们刚刚已经讨论过：

```py
Entity[] entities  = mXDocCoref.xdocCoref(chains);
```

以下`for`循环遍历提及链，这些提及链与`xdocCoref`返回的`Entities[]`对齐。对于每个提及链，提及被映射到其跨文档实体：

```py
Map<Mention,Entity> mentionToEntityMap
     = new HashMap<Mention,Entity>();
for (int i = 0; i < chains.length; ++i){ 
  for (Mention mention : chains[i].mentions()) {
         mentionToEntityMap.put(mention,entities[i]);
  }
}
```

接下来，代码将设置一系列映射来创建反映实体宇宙ID的块：

```py
String[] sentenceTexts
        = sentenceTextList
            .<String>toArray(new String[sentenceTextList.size()])
Mention[][] sentenceMentions
            = sentenceMentionList
            .<Mention[]>toArray(new Mention[sentenceMentionList.size()][]);
int[][] mentionStarts
         = mentionStartList
            .<int[]>toArray(new int[mentionStartList.size()][]);

int[][] mentionEnds
            = mentionEndList
            .<int[]>toArray(new int[mentionEndList.size()][]);
```

实际创建块将在下面发生：

```py
Chunking[] chunkings = new Chunking[sentenceTexts.length];
  for (int i = 0; i < chunkings.length; ++i) {
   ChunkingImpl chunking = new ChunkingImpl(sentenceTexts[i]);
   chunkings[i] = chunking;
   for (int j = 0; j < sentenceMentions[i].length; ++j) {
    Mention mention = sentenceMentions[i][j];
    Entity entity = mentionToEntityMap.get(mention);
    if (entity == null) {
     Chunk chunk = ChunkFactory.createChunk(mentionStarts[i][j],
       mentionEnds[i][j],
       mention.entityType()
       + ":-1");
     //chunking.add(chunk); //uncomment to get unresolved ents as -1 indexed.
    } else {
     Chunk chunk = ChunkFactory.createChunk(mentionStarts[i][j],
       mentionEnds[i][j],
       entity.type()
       + ":" + entity.id());
     chunking.add(chunk);
    }
   }
  }
```

然后使用这些块创建文档的相关部分，并返回`OutputDocument`：

```py
        // needless allocation here and last, but simple
        Chunking[] titleChunkings = new Chunking[firstContentSentenceIndex];
        for (int i = 0; i < titleChunkings.length; ++i)
            titleChunkings[i] = chunkings[i];

        Chunking[] bodyChunkings = new Chunking[chunkings.length - firstContentSentenceIndex];
        for (int i = 0; i < bodyChunkings.length; ++i)
            bodyChunkings[i] = chunkings[firstContentSentenceIndex+i];

        String id = document.id();

        OutputDocument result = new OutputDocument(id,titleChunkings,bodyChunkings);
        return result;
    }
```

因此，这就是我们为XDoc共指提供的起点。希望我们已经解释了更不透明的方法背后的意图。祝你好运！

# 约翰·史密斯问题

不同的人、地点和概念可能有相同的书写形式但却是不同的。世界上有多个“John Smith”、“Paris”和“bank”的例子，一个合适的跨文档共指系统应该能够处理这种情况。对于像“bank”这样的概念（河岸与金融机构），术语是词义消歧。这个方案将展示Baldwin当年与Amit Bagga一起为人物消歧开发的一种解决问题的方法。

## 准备工作

这个方案的代码紧密遵循[http://alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html](http://alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html)中的聚类教程，但将其修改得更接近原始的Bagga-Baldwin工作。代码量相当大，但没有什么特别复杂的。源代码位于`src/com/lingpipe/cookbook/chapter7/JohnSmith.java`。

这个类以标准NLP工具的集合开始，用于分词、句子检测和命名实体检测。如果这个堆栈不熟悉，请参考之前的方案：

```py
public static void main(String[] args) 
      throws ClassNotFoundException, IOException {
    TokenizerFactory tokenizerFactory = IndoEuropeanTokenizerFactory.INSTANCE;
    SentenceModel sentenceModel
    = new IndoEuropeanSentenceModel();
    SENTENCE_CHUNKER 
    = new SentenceChunker(tokenizerFactory,sentenceModel);
    File modelFile
    = new File("models/ne-en-news-muc6.AbstractCharLmRescoringChunker");
    NAMED_ENTITY_CHUNKER 
    = (Chunker) AbstractExternalizable.readObject(modelFile);
```

接下来，我们将重新审视`TfIdfDistance`。然而，这项任务要求我们将类包装在`Documents`上而不是`CharSequences`上，因为我们希望保留文件名，并且能够操作用于后续计算的文本：

```py
TfIdfDocumentDistance tfIdfDist = new TfIdfDocumentDistance(tokenizerFactory);
```

降级到引用类，我们有以下代码：

```py
public class TfIdfDocumentDistance implements Distance<Document> {
  TfIdfDistance mTfIdfDistance;
  public TfIdfDocumentDistance (TokenizerFactory tokenizerFactory) {
  mTfIdfDistance = new TfIdfDistance(tokenizerFactory);
  }

   public void train(CharSequence text) {
      mTfIdfDistance.handle(text);
   }

  @Override
  public double distance(Document doc1, Document doc2) {
    return mTfIdfDistance.distance(doc1.mCoreferentText,
              doc2.mCoreferentText);
  }

}
```

`train`方法与`TfIdfDistance.handle()`方法接口，并提供了一个实现`distance(Document doc1, Document doc2)`方法的实现，该方法将驱动下面讨论的聚类代码。`train`方法所做的只是提取相关文本并将其传递给`TfIdfDistance`类以获取相关值。

引用类`Document`是`JohnSmith`中的一个内部类，它相当简单。它获取具有匹配`.*John Smith.*`模式的句子，并将它们放入`mCoreferentText`变量中：

```py
static class Document {
        final File mFile;
        final CharSequence mText; 
        final CharSequence mCoreferentText;
        Document(File file) throws IOException {
            mFile = file; // includes name
            mText = Files.readFromFile(file,Strings.UTF8);
            Set<String> coreferentSents 
      = getCoreferentSents(".*John "                        + "Smith.*",mText.toString());
            StringBuilder sb = new StringBuilder();
            for (String sentence : coreferentSents) {
              sb.append(sentence);
            }
            mCoreferentText = sb.toString();
        }

        public String toString() {
            return mFile.getParentFile().getName() + "/"  
            + mFile.getName();
        }
    }
```

深入代码，我们现在将访问`getCoreferentSents()`方法：

```py
static final Set<String> getCoreferentSents(String targetPhrase, String text) {
     Chunking sentenceChunking
    = SENTENCE_CHUNKER.chunk(text);
  Iterator<Chunk> sentenceIt 
    = sentenceChunking.chunkSet().iterator();
  int targetId = -2;
  MentionFactory mentionFactory = new EnglishMentionFactory();
  WithinDocCoref coref = new WithinDocCoref(mentionFactory);
  Set<String> matchingSentenceAccumulator 
  = new HashSet<String>();
for (int sentenceNum = 0; sentenceIt.hasNext(); ++sentenceNum) {
  Chunk sentenceChunk = sentenceIt.next();
  String sentenceText 
    = text.substring(sentenceChunk.start(),
          sentenceChunk.end());
  Chunking neChunking
    = NAMED_ENTITY_CHUNKER.chunk(sentenceText);
  Set<Chunk> chunkSet 
    = new TreeSet<Chunk>(Chunk.TEXT_ORDER_COMPARATOR);
  chunkSet.addAll(neChunking.chunkSet());      Coreference.addRegexMatchingChunks(
    Pattern.compile("\\bJohn Smith\\b"),
            "PERSON",sentenceText,chunkSet);
  Iterator<Chunk> neChunkIt = chunkSet.iterator();
  while (neChunkIt.hasNext()) {
    Chunk neChunk = neChunkIt.next();
    String mentionText
        = sentenceText.substring(neChunk.start(),
            neChunk.end());
    String mentionType = neChunk.type();
    Mention mention 
    = mentionFactory.create(mentionText,mentionType);
    int mentionId 
    = coref.resolveMention(mention,sentenceNum);
    if (targetId == -2 && mentionText.matches(targetPhrase)) {
    targetId = mentionId;
    }
    if (mentionId == targetId) {                          matchingSentenceAccumulator.add(sentenceText);
     System.out.println("Adding " + sentenceText);      
     System.out.println("     mention text=" + mentionText
            + " type=" + mentionType
            + " id=" + mentionId);
     }
  }
}
if (targetId == -2) {
  System.out.println("!!!Missed target doc " + text);
}
return matchingSentenceAccumulator;
}
```

查看前述方法的*跨文档共指*配方，了解大多数移动部件。我们将指出一些值得注意的部分。从某种意义上说，我们通过使用正则表达式分块器来查找任何包含`John Smith`子串的字符串并将其添加为`PERSON`实体，我们是在作弊。像大多数作弊一样，如果你的唯一目的是追踪`John Smith`，这非常有用。我们在现实中所做的作弊是使用字典匹配来查找所有高价值情报目标（如`Osama bin Laden`）的变体。最终，我们在MiTAP项目中公开可用的新闻源中搜索了他的40多种版本。

此外，随着每个句子的处理，我们将检查所有提及的`John Smith`的匹配模式，如果匹配，我们将收集任何提及此ID的句子。这意味着如果一个句子用代词指回`John Smith`，它将被包括在内，如果共指正在发挥作用，`Mr. Smith`的情况也是如此。请注意，在我们开始收集上下文信息之前，我们需要看到`John Smith`的匹配，所以我们会错过`He awoke. John Smith was a giant cockroach`的第一句话。此外，如果出现第二个具有不同ID的`John Smith`，它将被忽略——这种情况可能发生。

最后，请注意，有一些错误检查，如果找不到`John Smith`，则将错误报告给`System.out`。

在设置`TfIdfDocumentDistance`之后，如果我们回到`main()`方法中的平凡I/O操作，我们会这样做：

```py
File dir = new File(args[0]);
       Set<Set<Document>> referencePartition
            = new HashSet<Set<Document>>();
        for (File catDir : dir.listFiles()) {
            System.out.println("Category from file=" + catDir);
            Set<Document> docsForCat = new HashSet<Document>();
            referencePartition.add(docsForCat);
            for (File file : catDir.listFiles()) {
                Document doc = new Document(file);
                tfIdfDist.train(doc.mText);
                docsForCat.add(doc);
            }
        }
```

我们尚未讨论这一点，但哪个文档引用了哪个`Mr. Smith`的真实标注编码在数据目录结构中。顶级`johnSmith`目录中的每个子目录都被视为真实簇。因此，`referencePartition`包含真实标注。我们可以将此作为每个子目录的正确分类的分类问题进行包装。我们将把这个任务留给你，用逻辑回归解决方案将其填充到交叉验证语料库中。

接下来，我们将通过将之前的类别展平成一个单一的`Documents`集合来构建测试集。我们本来可以在上一步完成这个操作，但混合任务往往会产生错误，而且额外的`for`循环对执行速度的影响非常小：

```py
        Set<Document> docSet = new HashSet<Document>();
        for (Set<Document> cluster : referencePartition) {
            docSet.addAll(cluster);
        }
```

接下来，我们将准备聚类算法。我们将使用`TfIdfDocumentDistance`运行程序，进行`CompleteLink`和`SingleLink`：

```py

        HierarchicalClusterer<Document> clClusterer
            = new CompleteLinkClusterer<Document>(tfIdfDist);
        Dendrogram<Document> completeLinkDendrogram
            = clClusterer.hierarchicalCluster(docSet);

        HierarchicalClusterer<Document> slClusterer
            = new SingleLinkClusterer<Document>(tfIdfDist);
        Dendrogram<Document> singleLinkDendrogram
            = slClusterer.hierarchicalCluster(docSet);
```

聚类算法的细节在[第5章](part0061_split_000.html#page "第5章. 文本中的跨度查找 – 分块") *文本中的跨度查找 – 分块* 中有所介绍。现在，我们将根据从`1`到输入数量的聚类数量报告性能。一个花哨的部分是，`Cross`类别使用`SingleLinkClusterer`作为参考，`CompleteLinkClusterer`作为响应：

```py
System.out.println();
System.out.println(" -------------------------------------------"
        + "-------------");
System.out.println("|  K  |  Complete      |  Single        | "
        + " Cross         |");
System.out.println("|     |  P    R    F   |  P    R    F   |  P"
        + "     R    F   |");
System.out.println(" -------------------------------------------"
        +"-------------");
for (int k = 1; k <= docSet.size(); ++k) {
   Set<Set<Document>> clResponsePartition
       = completeLinkDendrogram.partitionK(k);
   Set<Set<Document>> slResponsePartition
       = singleLinkDendrogram.partitionK(k);

   ClusterScore<Document> scoreCL
       = new ClusterScore<Document>(referencePartition,
                                    clResponsePartition) PrecisionRecallEvaluation clPrEval 
      = scoreCL.equivalenceEvaluation();
   ClusterScore<Document> scoreSL
       = new ClusterScore<Document>(referencePartition,
                                     slResponsePartition);
PrecisionRecallEvaluation slPrEval 
  = scoreSL.equivalenceEvaluation();

ClusterScore<Document> scoreX
    = new ClusterScore<Document>(clResponsePartition
                                 slResponsePartition);
PrecisionRecallEvaluation xPrEval 
  = scoreX.equivalenceEvaluation();

System.out.printf("| %3d | %3.2f %3.2f %3.2f | %3.2f %3.2f %3.2f" 
      + " | %3.2f %3.2f %3.2f |\n",
                   k,
                   clPrEval.precision(),
                   clPrEval.recall(),
                   clPrEval.fMeasure(),
                   slPrEval.precision(),
                   slPrEval.recall(),
                   slPrEval.fMeasure(),
                   xPrEval.precision(),
                   xPrEval.recall(),
                   xPrEval.fMeasure());
 }
System.out.println(" --------------------------------------------"
         + "------------");
}
```

这就是我们为这个食谱做准备所需做的所有事情。这是一个罕见的现象，这是一个玩具实现，但关键概念应该是显而易见的。

## 如何做...

我们将只运行这个代码，然后稍微修改一下：

1.  打开终端并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.JohnSmith

    ```

1.  结果将是一堆信息，表明正在提取哪些句子用于聚类——记住，真实标注是由文件所在的目录决定的。第一个聚类是`0`：

    ```py
    Category from file=data/johnSmith/0
    ```

1.  代码报告包含对`John Smith`引用的句子：

    ```py
    Adding I thought John Smith marries Pocahontas.''
         mention text=John Smith type=PERSON id=5
    Adding He's bullets , she's arrows.''
         mention text=He type=MALE_PRONOUN id=5
    ```

1.  对`John Smith`的代词引用是第二句被包含的基础。

1.  系统输出继续，最后，我们将得到针对真实情况的单链接聚类结果和完全链接聚类结果。`K`列表示算法允许的聚类数量，并报告了精确度、召回率和F度量。在这种情况下，第一行表示只有一个聚类允许100%的召回率和23%的精确度，无论是完全链接还是单链接。向下查看分数，我们可以看到完全链接报告了最佳的F度量，有11个聚类在`0.60`——实际上有35个聚类。单链接方法将F度量最大化到68个聚类，达到`0.78`，并在不同数量的聚类上显示出更大的鲁棒性。交叉案例显示，单链接和完全链接在直接比较中也很不同。请注意，为了可读性，一些`K`值已被消除：

    ```py
    --------------------------------------------------------
    |  K  |  Complete      |  Single        
    |     |  P    R    F   |  P    R    F   
     --------------------------------------------------------
    |   1 | 0.23 1.00 0.38 | 0.23 1.00 0.38 
    |   2 | 0.28 0.64 0.39 | 0.24 1.00 0.38 
    |   3 | 0.29 0.64 0.40 | 0.24 1.00 0.39 
    |   4 | 0.30 0.64 0.41 | 0.24 1.00 0.39 
    |   5 | 0.44 0.63 0.52 | 0.24 0.99 0.39 
    |   6 | 0.45 0.63 0.52 | 0.25 0.99 0.39 
    |   7 | 0.45 0.63 0.52 | 0.25 0.99 0.40 
    |   8 | 0.49 0.62 0.55 | 0.25 0.99 0.40 
    |   9 | 0.55 0.61 0.58 | 0.25 0.99 0.40 
    |  10 | 0.55 0.61 0.58 | 0.25 0.99 0.41 
    |  11 | 0.59 0.61 0.60 | 0.27 0.99 0.42 
    |  12 | 0.59 0.61 0.60 | 0.27 0.98 0.42 
    |  13 | 0.56 0.41 0.48 | 0.27 0.98 0.43 
    |  14 | 0.71 0.41 0.52 | 0.27 0.98 0.43 
    |  15 | 0.71 0.41 0.52 | 0.28 0.98 0.43 
    |  16 | 0.68 0.34 0.46 | 0.28 0.98 0.44 
    |  17 | 0.68 0.34 0.46 | 0.28 0.98 0.44 
    |  18 | 0.69 0.34 0.46 | 0.29 0.98 0.44 
    |  19 | 0.67 0.32 0.43 | 0.29 0.98 0.45 
    |  20 | 0.69 0.29 0.41 | 0.29 0.98 0.45 
    |  30 | 0.84 0.22 0.35 | 0.33 0.96 0.49 
    |  40 | 0.88 0.18 0.30 | 0.61 0.88 0.72 
    |  50 | 0.89 0.16 0.28 | 0.64 0.86 0.73 
    |  60 | 0.91 0.14 0.24 | 0.66 0.77 0.71 
    |  61 | 0.91 0.14 0.24 | 0.66 0.75 0.70 
    |  62 | 0.93 0.14 0.24 | 0.87 0.75 0.81 
    |  63 | 0.94 0.13 0.23 | 0.87 0.69 0.77 
    |  64 | 0.94 0.13 0.23 | 0.87 0.69 0.77 
    |  65 | 0.94 0.13 0.23 | 0.87 0.68 0.77 
    |  66 | 0.94 0.13 0.23 | 0.87 0.66 0.75 
    |  67 | 0.95 0.13 0.23 | 0.87 0.66 0.75 
    |  68 | 0.95 0.13 0.22 | 0.95 0.66 0.78 
    |  69 | 0.94 0.11 0.20 | 0.95 0.66 0.78 
    |  70 | 0.94 0.11 0.20 | 0.95 0.65 0.77 
    |  80 | 0.98 0.11 0.19 | 0.97 0.43 0.59 
    |  90 | 0.99 0.10 0.17 | 0.97 0.30 0.46 
    | 100 | 0.99 0.08 0.16 | 0.96 0.20 0.34 
    | 110 | 0.99 0.07 0.14 | 1.00 0.11 0.19 
    | 120 | 1.00 0.07 0.12 | 1.00 0.08 0.14 
    | 130 | 1.00 0.06 0.11 | 1.00 0.06 0.12 
    | 140 | 1.00 0.05 0.09 | 1.00 0.05 0.10 
    | 150 | 1.00 0.04 0.08 | 1.00 0.04 0.08 
    | 160 | 1.00 0.04 0.07 | 1.00 0.04 0.07 
    | 170 | 1.00 0.03 0.07 | 1.00 0.03 0.07 
    | 180 | 1.00 0.03 0.06 | 1.00 0.03 0.06 
    | 190 | 1.00 0.02 0.05 | 1.00 0.02 0.05 
    | 197 | 1.00 0.02 0.04 | 1.00 0.02 0.04 
     --------------------------------------------------------
    ```

1.  以下输出通过最大距离阈值来约束聚类，而不是通过聚类大小。输出是针对单链接聚类的，`.05`增加距离，评估是B-立方度指标。输出包括距离、精确度、召回率和最终聚类的规模。在`.80`和`.9`时的性能相当好，但要注意在这种事后设置生产阈值的方式。在生产环境中，我们希望在设置阈值之前看到更多数据：

    ```py
    B-cubed eval
    Dist: 0.00 P: 1.00 R: 0.77 size:189
    Dist: 0.05 P: 1.00 R: 0.80 size:171
    Dist: 0.10 P: 1.00 R: 0.80 size:164
    Dist: 0.15 P: 1.00 R: 0.81 size:157
    Dist: 0.20 P: 1.00 R: 0.81 size:153
    Dist: 0.25 P: 1.00 R: 0.82 size:148
    Dist: 0.30 P: 1.00 R: 0.82 size:144
    Dist: 0.35 P: 1.00 R: 0.83 size:142
    Dist: 0.40 P: 1.00 R: 0.83 size:141
    Dist: 0.45 P: 1.00 R: 0.83 size:141
    Dist: 0.50 P: 1.00 R: 0.83 size:138
    Dist: 0.55 P: 1.00 R: 0.83 size:136
    Dist: 0.60 P: 1.00 R: 0.84 size:128
    Dist: 0.65 P: 1.00 R: 0.84 size:119
    Dist: 0.70 P: 1.00 R: 0.86 size:108
    Dist: 0.75 P: 0.99 R: 0.88 size: 90
    Dist: 0.80 P: 0.99 R: 0.94 size: 60
    Dist: 0.85 P: 0.95 R: 0.97 size: 26
    Dist: 0.90 P: 0.91 R: 0.99 size:  8
    Dist: 0.95 P: 0.23 R: 1.00 size:  1
    Dist: 1.00 P: 0.23 R: 1.00 size:  1
    ```

1.  B-立方度（Bagga, Bierman, 和 Baldwin）评估是为了严重惩罚将大聚类放在一起。它假设将大量关于乔治·W·布什的文档与乔治·H·W·布什放在一起是一个更大的问题，两者都是大聚类，而不是将乔治·布什，那位在数据集中只被提及一次的机械师弄错。其他评分指标将把这两个错误视为同样糟糕。这是文献中用于这种现象的标准评分指标。

## 参见

在研究文献中，关于这个确切问题的研究相当丰富。我们并不是第一个考虑这个问题的人，但我们提出了主导的评价指标，并且发布了一个语料库，供其他团队与我们以及彼此进行比较。我们的贡献是Bagga和Baldwin在*ACL '98第36届计算语言学协会年会和第17届国际计算语言学会议论文集*中提出的*基于实体的跨文档共指消解使用向量空间模型*。自那时以来，已经取得了许多进展——在谷歌学术上有超过400篇关于这个模型的引用；如果这个问题对你很重要，它们值得一看。
