# 第七章：在概念/人物之间寻找共指

本章将涵盖以下内容：

+   与文档中的命名实体共指

+   向共指中添加代词

+   跨文档共指

+   John Smith 问题

# 介绍

共指是人类语言中的一种基本机制，它使得两句话可以指代同一个事物。对人类交流而言，它非常重要——其功能与编程语言中的变量名非常相似，但在细节上，作用范围的定义规则与代码块截然不同。共指在商业上不那么重要——也许本章将帮助改变这一点。这里有一个例子：

```py
Alice walked into the garden. She was surprised.
```

共指存在于`Alice`和`She`之间；这些短语指代的是同一个事物。当我们开始探讨一个文档中的 Alice 是否与另一个文档中的 Alice 相同时，情况变得非常有趣。

共指，就像词义消歧一样，是下一代的工业能力。共指的挑战促使美国国税局（IRS）坚持要求一个能够明确识别个人的社会保障号码，而不依赖于其名字。许多讨论的技术都是为了帮助跟踪文本数据中的个人和组织，尽管成功程度不一。

# 与文档中的命名实体共指

如第五章中所见，*文本中的跨度 – Chunking*，LingPipe 可以使用多种技术来识别与人、地方、事物、基因等相关的专有名词。然而，分块并未完全解决问题，因为它无法帮助在两个命名实体相同的情况下找到一个实体。能够判断 John Smith 和 Mr. Smith、John 甚至完全重复的 John Smith 是同一个实体是非常有用的——它甚至在我们还是一个初创国防承包商时就成为了我们公司成立的基础。我们的创新贡献是生成按实体索引的句子，这种方法证明是总结某个实体所讨论内容的极佳方式，尤其是当这种映射跨越不同语言时——我们称之为**基于实体的摘要化**。

### 注意

基于实体的摘要化的想法源自巴尔温在宾夕法尼亚大学研究生研讨会上的一次讲座。时任系主任的米奇·马库斯认为，展示所有提到某个实体的句子——包括代词——将是对该实体的极佳总结。从某种意义上说，这就是 LingPipe 诞生的原因。这一想法促使巴尔温领导了一个 UPenn DARPA 项目，并最终创立了 Alias-i。经验教训——与每个人交流你的想法和研究。

本教程将带你了解计算共指的基础知识。

## 准备工作

拿到一些叙述性文本，我们将使用一个简单的示例，大家知道它是有效的——共指系统通常需要针对特定领域进行大量调整。你可以自由选择其他文本，但它必须是英文的。

## 如何做...

如常，我们将通过命令行运行代码，然后深入分析代码的实际功能。我们开始吧。

1.  我们将从一个简单的文本开始，以说明共指。文件位于`data/simpleCoref.txt`，它包含：

    ```py
    John Smith went to Washington. Mr. Smith is a business man.
    ```

1.  去命令行和 Java 解释器那里，复制以下内容：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.NamedEntityCoreference

    ```

1.  这会得到以下结果：

    ```py
    Reading in file :data/simpleCoref.txt 
    Sentence Text=John Smith went to Washington.
         mention text=John Smith type=PERSON id=0
         mention text=Washington type=LOCATION id=1
    Sentence Text=Mr. Smith is a business man.
         mention text=Mr. Smith type=PERSON id=0
    ```

1.  找到了三个命名实体。注意，输出中有一个`ID`字段。`John Smith`和`Mr. Smith`实体具有相同的 ID，`id=0`。这意味着这些短语被认为是共指的。剩余的实体`Washington`具有不同的 ID，`id=1`，并且与 John Smith / Mr. Smith 不共指。

1.  创建你自己的文本文件，将其作为参数传递到命令行，看看会计算出什么。

## 它是如何工作的...

LingPipe 中的共指代码是建立在句子检测和命名实体识别之上的启发式系统。总体流程如下：

1.  对文本进行分词。

1.  检测文档中的句子，对于每个句子，按从左到右的顺序检测句子中的命名实体，并对每个命名实体执行以下任务：

    1.  创建一个提到。提到是命名实体的单一实例。

    1.  提到可以被添加到现有的提到链中，或者可以启动它们自己的提到链。

    1.  尝试将提到的实体解析为已创建的提到链。如果找到唯一匹配，则将该提到添加到提到链中；否则，创建一个新的提到链。

代码位于`src/com/lingpipe/cookbook/chapter7/NamedEntityCoreference.java`。`main()`方法首先设置这个配方的各个部分，从分词器工厂、句子分块器，到最后的命名实体分块器：

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

现在，我们已经设置了基本的配方基础设施。接下来是一个共指专用类：

```py
MentionFactory mf = new EnglishMentionFactory();
```

`MentionFactory`类从短语和类型创建提到——当前的源被命名为`entities`。接下来，共指类会以`MentionFactory`作为参数创建：

```py
WithinDocCoref coref = new WithinDocCoref(mf);
```

`WithinDocCoref`类封装了计算共指的所有机制。从第五章，*查找文本中的跨度 – 分块*，你应该熟悉获取文档文本、检测句子，并遍历应用命名实体分块器到每个句子的代码：

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

在当前句子的上下文中，句子中的命名实体会按从左到右的顺序进行迭代，就像它们被阅读的顺序一样。我们知道这一点是因为`ChunkingImpl`类按照它们被添加的顺序返回块，而我们的`HMMChunker`是以从左到右的顺序添加它们的：

```py
Chunking neChunking = namedEntChunker.chunk(sentenceText);
for (Chunk neChunk : neChunking.chunkSet()) {
```

以下代码从分块中获取信息——类型和短语，但*不*包括偏移信息，并创建一个提到：

```py
String mentionText
  = sentenceText.substring(neChunk.start(),
          neChunk.end());
String mentionType = neChunk.type();
Mention mention = mf.create(mentionText,mentionType);
```

下一行与提到的内容进行共指，并返回它所在的句子的 ID：

```py
int mentionId = coref.resolveMention(mention,sentenceNum);

System.out.println("     mention text=" + mentionText
            + " type=" + mentionType
            + " id=" + mentionId);
```

如果提及已解析为现有实体，它将具有该 ID，正如我们在 Mr. Smith 例子中看到的那样。否则，它将获得一个独立的 ID，并且可以作为后续提及的先行词。

这涵盖了在文档内运行共指关系的机制。接下来的配方将介绍如何修改这个类。下一个配方将添加代词并提供引用。

# 向共指关系中添加代词

前面的配方处理了命名实体之间的共指关系。这个配方将把代词添加到其中。

## 如何操作……

这个配方将使用交互式版本帮助你探索共指算法的特性。该系统非常依赖命名实体检测的质量，因此请使用 HMM 可能正确识别的例子。它是在 90 年代的*华尔街日报*文章上进行训练的。

1.  启动你的控制台并键入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.Coreference

    ```

1.  在结果命令提示符中，键入以下内容：

    ```py
    Enter text followed by new line
    >John Smith went to Washington. He was a senator.
    Sentence Text=John Smith went to Washington.
    mention text=John Smith type=PERSON id=0
    mention text=Washington type=LOCATION id=1
    Sentence Text= He was a senator.
    mention text=He type=MALE_PRONOUN id=0
    ```

1.  `He`和`John Smith`之间的共享 ID 表示它们之间的共指关系。接下来会有更多的例子和注释。请注意，每个输入被视为具有独立 ID 空间的不同文档。

1.  如果代词没有解析为命名实体，它们会得到`-1`的索引，如下所示：

    ```py
    >He went to Washington.
    Sentence Text= He went to Washington.
    mention text=He type=MALE_PRONOUN id=-1
    mention text=Washington type=LOCATION id=0
    ```

1.  以下情况也会导致`id`为`-1`，因为在先前的上下文中没有唯一的一个人，而是两个人。这被称为失败的唯一性预设：

    ```py
    >Jay Smith and Jim Jones went to Washington. He was a senator.
    Sentence Text=Jay Smith and Jim Jones went to Washington.
    mention text=Jay Smith type=PERSON id=0
    mention text=Jim Jones type=PERSON id=1
    mention text=Washington type=LOCATION id=2
    Sentence Text= He was a senator.
    mention text=He type=MALE_PRONOUN id=-1
    ```

1.  以下代码显示了`John Smith`也可以解析为女性代词。这是因为没有关于哪些名字表示哪种性别的数据。可以添加这类数据，但通常上下文会消除歧义。`John`也可能是一个女性名字。关键在于，代词会消除性别歧义，后续的男性代词将无法匹配：

    ```py
    Frank Smith went to Washington. She was a senator. 
    Sentence Text=Frank Smith went to Washington.
         mention text=Frank Smith type=PERSON id=0
         mention text=Washington type=LOCATION id=1
    Sentence Text=She was a senator.
         mention text=She type=FEMALE_PRONOUN id=0
    ```

1.  性别分配将阻止错误性别的引用。以下代码中的`He`代词被解析为 ID`-1`，因为唯一的人被解析为女性代词：

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

1.  共指关系也可以发生在句子内部：

    ```py
    >Jane Smith knows her future.
    Sentence Text=Jane Smith knows her future.
         mention text=Jane Smith type=PERSON id=0
         mention text=her type=FEMALE_PRONOUN id=0
    ```

1.  提及的顺序（按最新提及排序）在解析提及时很重要。在以下代码中，`He`被解析为`James`，而不是`John`：

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

1.  命名实体提及也会产生相同的效果。`Mr. Smith`实体解析为最后一次提及：

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

1.  如果插入太多句子，`John`和`James`之间的区别将消失：

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

前面的例子旨在展示文档内共指关系系统的特性。

## 它是如何工作的……

添加代词的代码变化非常直接。此配方的代码位于`src/com/lingpipe/cookbook/chapter7/Coreference.java`。该配方假设你理解了前一个配方，因此这里只涵盖了代词提及的添加：

```py
Chunking mentionChunking
  = neChunker.chunk(sentenceText);
Set<Chunk> chunkSet = new TreeSet<Chunk> (Chunk.TEXT_ORDER_COMPARATOR);
chunkSet.addAll(mentionChunking.chunkSet());
```

我们从多个来源添加了`Mention`对象，因此元素的顺序不再有保证。相应地，我们创建了`TreeSet`和适当的比较器，并将所有来自`neChunker`的分块添加到其中。

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

以下代码行展示了`addRegExMatchingChunks`子程序。它根据正则表达式匹配添加片段，并移除重叠的、已有的 HMM 派生片段：

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

复杂之处在于，`MALE_PRONOUN`和`FEMALE_PRONOUN`代词的类型将用于与`PERSON`实体匹配，结果是解析过程会设置被解析实体的性别。

除此之外，代码应与我们标准的 I/O 循环非常相似，该循环在命令提示符中运行交互。

## 另见

系统背后的算法基于 Baldwin 的博士论文。该系统名为 CogNIAC，工作始于 90 年代中期，并非当前最先进的共指消解系统。更现代的方法很可能会使用机器学习框架，利用 Baldwin 方法生成的特征和其他许多特征来开发一个性能更好的系统。有关该系统的论文可见于[`www.aclweb.org/anthology/W/W97/W97-1306.pdf`](http://www.aclweb.org/anthology/W/W97/W97-1306.pdf)。

# 跨文档共指

跨文档共指（XDoc）将单个文档的`id`空间扩展到更广泛的宇宙。这一宇宙通常包括其他处理过的文档和已知实体的数据库。虽然注解本身非常简单，只需将文档范围内的 ID 替换为宇宙范围内的 ID 即可，但计算 XDoc 可能相当复杂。

本教程将告诉我们如何使用在多年部署此类系统过程中开发的轻量级 XDoc 实现。我们将为那些可能希望扩展/修改代码的人提供代码概述，但内容较为复杂，教程也相当密集。

输入采用 XML 格式，其中每个文件可以包含多个文档：

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

目标是生成注解，其中 Breck Baldwin 的提及在各个文档中共享与 Krishna 相同的 ID。注意，在最后一篇文档中，二者都是以昵称被提及的。

XDoc 的一个常见扩展是将已知实体的**数据库**（**DB**）与文本中提到的这些实体进行链接。这弥合了结构化数据库和非结构化数据（文本）之间的鸿沟，许多人认为这是商业智能/客户声音/企业知识管理中的下一个重要发展方向。我们曾构建过将基因/蛋白质数据库与 MEDLINE 摘要、以及人物关注名单与自由文本等链接的系统。数据库还为人工编辑提供了一种自然的方式来控制 XDoc 的行为。

## 如何实现...

本食谱的所有代码都位于 `com.lingpipe.cookbook.chapter7.tracker` 包中。

1.  访问您的 IDE 并运行 `RunTracker`，或者在命令行中输入以下命令：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.tracker.RunTracker

    ```

1.  屏幕将会滚动，显示文档分析的过程，但我们将转到指定的输出文件并进行查看。用您喜欢的文本编辑器打开 `cookbook/data/xDoc/output/docs1.xml`。您将看到一个格式不佳的示例输出版本，除非您的编辑器能够自动格式化 XML（例如，Firefox 浏览器能较好地呈现 XML）。输出应如下所示：

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

1.  `Krishna` 在前两份文档中被共享 ID `1000000002` 识别，但昵称 `K-dog` 完全没有被识别。`Breck` 在所有三份文档中都被识别，但由于第三次提到的 ID `Breckles` 与前两次提到的不同，系统认为它们不是同一个实体。

1.  接下来，我们将使用字典形式的数据库来提高当作者通过昵称提及时的识别度。`data/xDoc/author-dictionary.xml` 中有一个字典，内容如下：

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

1.  上述字典包含了两位作者的昵称，以及他们的名字。带有 `xdc=1` 值的别名将用于跨文档链接实体。`xdc=0` 值只会在单个文档内应用。所有别名将通过字典查找来识别命名实体。

1.  运行以下命令，指定实体字典或相应的 IDE 等效项：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.tracker.RunTracker data/xDoc/author-dictionary.xml

    ```

1.  `xDoc/output/docs1.xml` 中的输出与上次运行的结果有很大不同。首先，注意现在的 ID 与字典文件中指定的相同：`Breck` 的 ID 为 `1`，`Krishna` 的 ID 为 `2`。这是结构化数据库（如字典的性质）与非结构化文本之间的联系。其次，注意到我们的昵称已经被正确识别并分配到正确的 ID。第三，注意到类型现在是 `MALE`，而不是 `OTHER`：

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

这是对如何运行 XDoc 的简要介绍。在接下来的部分，我们将看到它是如何工作的。

## 它是如何工作的…

在这道食谱之前，我们一直尝试保持代码简单、直观并且易于理解，而不深入探讨大量源代码。这道食谱更为复杂。支撑这个食谱的代码无法完全放入预定的空间中进行解释。此处的阐述假设您会自行探究整个类，并参考本书中的其他食谱进行说明。我们提供这道食谱是因为 XDoc 核心参考是一个非常有趣的问题，我们现有的基础设施可能帮助其他人探索这一现象。欢迎来到游泳池的深水区。

### 批处理生命周期

整个过程由 `RunTracker.java` 类控制。`main()` 方法的总体流程如下：

1.  读取已知实体的数据库，这些实体将通过`Dictionary`进行命名实体识别，并且存在从别名到字典条目的已知映射。别名包含关于是否应该通过`xdc=1`或`xdc=0`标志用于跨文档匹配实体的说明。

1.  设置`EntityUniverse`，它是文本中找到的实体及已知实体字典的全局 ID 数据结构。

1.  设置文档内核心指代所需的内容——例如分词器、句子检测器和命名实体检测器。还会用到一些 POS 标注器和词汇计数器，处理得稍微复杂一些。

1.  有一个布尔值控制是否会添加推测性实体。如果该布尔值为`true`，则表示我们会将从未见过的实体添加到跨文档实体的宇宙中。将该值设置为`true`时，可靠地计算是更具挑战性的任务。

1.  所有提到的配置都用于创建`Tracker`对象。

1.  然后，`main()`方法读取待处理的文档，将其交给`Tracker`对象进行处理，并将处理结果写入磁盘。`Tracker.processDocuments()`方法的主要步骤如下：

    1.  获取一组 XML 格式的文档，并获取单个文档。

    1.  对于每个文档，应用`processDocument()`方法，该方法使用字典进行文档内核心指代分析，帮助查找实体以及命名实体检测器，并返回`MentionChain[]`。然后，将每个提及的链条与实体宇宙进行对比，以更新文档级 ID 为实体宇宙 ID。最后一步是将文档写入磁盘，带上实体宇宙 ID。

以上就是我们要说的关于`RunTracker`的内容，里面没有任何你在本书中无法处理的内容。在接下来的章节中，我们将讨论`RunTracker`使用的各个组成部分。

#### 设置实体宇宙

实体宇宙`EntityUniverse.java`是文档/数据库集合中提到的全局实体的内存表示。实体宇宙还包含指向这些实体的各种索引，支持在单个文档上计算 XDoc。

字典将已知实体填充到`EntityUniverse`文件中，随后处理的文档会对这些实体敏感。XDoc 算法尝试在创建新实体之前与现有实体合并，因此字典中的实体会强烈吸引这些实体的提及。

每个实体由唯一的长整型 ID、一组分为四个单独列表的别名和一个类型（如人、地点等）组成。还会说明该实体是否在用户定义的字典中，以及是否允许添加推测性提及到该实体。`toString()`方法将实体列出为：

```py
id=556 type=ORGANIZATION userDefined=true allowSpec=false user XDC=[Nokia Corp., Nokia] user non-XDC=[] spec XDC=[] spec non-XDC
=[]
```

全局数据结构如下：

```py
    private long mLastId = FIRST_SYSTEM_ID;
```

实体需要唯一的 ID，我们约定`FIRST_SYSTEM_ID`的值是一个大整数，比如`1,000,000`。这样可以为用户提供一个空间（ID < 1,000,000），以便他们在不会与系统已发现的实体发生冲突的情况下添加新实体。

我们将为整个跟踪器实例化一个分词器：

```py
    private final TokenizerFactory mTokenizerFactory;
```

存在一个全局映射，将唯一的实体 ID 映射到实体：

```py
    private final Map<Long,Entity> mIdToEntity
        = new HashMap<Long,Entity>();
```

另一个重要的数据结构是一个将别名（短语）映射到拥有该别名的实体的映射——`mXdcPhraseToEntitySet`。只有那些可以作为跨文档共指候选的短语才会被添加到这里。从字典中，`xdc=1`的别名会被添加进来：

```py
private final ObjectToSet<String,Entity> mXdcPhraseToEntitySet
        = new ObjectToSet<String,Entity>();
```

对于推测性找到的别名，如果该别名至少包含两个标记且尚未关联到其他实体，则将其添加到此集合中。这反映了一种启发式方法，力求不拆分实体。这一逻辑相当复杂，超出了本教程的范畴。你可以参考`EntityUniverse.createEntitySpeculative`和`EntityUniverse.addPhraseToEntity`中的代码。

为什么有些别名在寻找候选实体时不被使用？考虑到`George`对`EntityUniverse`中的实体区分帮助不大，而`George H.W. Bush`则提供了更多的信息用于区分。

#### ProcessDocuments()和 ProcessDocument()

有趣的部分开始出现在`Tracker.processDocuments()`方法中，该方法调用每个文档的 XML 解析，然后逐步调用`processDocument()`方法。前者的代码比较简单，因此我们将跳到更具任务特定性的工作部分，即调用`processDocument()`方法时的逻辑：

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

我们使用了一种文档格式，可以将标题与正文区分开来。如果标题格式与正文格式有所不同，这种做法是一个好主意，就像新闻稿中通常所做的那样。`chains`变量将包含来自标题和正文的链条，其中可能存在相互指代的情况。`mentionStartList`和`mentionEndList`数组将在方法的后续步骤中使得重新对齐文档范围内的 ID 与实体宇宙范围内的 ID 成为可能：

```py
Entity[] entities  = mXDocCoref.xdocCoref(chains);
```

#### 计算 XDoc

XDoc 代码是通过多小时手动调试算法的结果，旨在处理新闻风格的数据。它已经在 20,000 文档范围的数据集上运行，并且设计上非常积极地支持词典条目。该代码还试图避免**短路**，即当显然不同的实体被合并在一起时会发生的错误。如果你错误地将芭芭拉·布什和乔治·布什视为同义词，那么结果将会非常尴尬，用户将看到这些错误。

另一种错误是全局存储中有两个实体，而实际上一个就足够了。这类似于*超人/克拉克·肯特问题*，同样也适用于多次提及同一个名字的情况。

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

一个文档包含一个提及链列表，每个提及链要么被添加到现有实体中，要么被提升为一个新实体。提及链必须包含一个非代词的提及，这在文档内的共指层面上进行处理。

在处理每个提及链时，会更新三种数据结构：

+   `Entity[]` 实体由 `xdocCoref` 方法返回，以支持文档的内联注解。

+   `Map<MentionChain,Entity> chainToEntity` 将提及链映射到实体。

+   `ObjectToSet<Entity,MentionChain> entityToChainSet` 是 `chainToEntity` 的反向映射。可能同一文档中的多个链条映射到同一实体，因此这个数据结构要考虑到这种可能性。该版本的代码允许这种情况发生——实际上，XDoc 正在以副作用的方式设置文档内的共指解析。

很简单，如果找到了实体，`addMentionChainToEntity()` 方法会将提及链中的任何新信息添加到该实体中。新信息可能包括新别名和类型变化（例如，通过消歧代词引用将一个人从无性别转变为男性或女性）。如果没有找到实体，那么提及链会被送到 `promote()`，它会在实体宇宙中创建一个新实体。我们将从 `promote()` 开始。

#### `promote()` 方法

实体宇宙是一个极简的数据结构，仅记录短语、类型和 ID。`TTMentionChain` 类是特定文档中提及的更复杂的表示形式：

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

对 `mEntityUniverse.createEntitySpeculative` 的调用只需要链条的短语（在这种情况下，已归一化为小写且所有空格序列被转换为单个空格的短语）以及实体的类型。不会记录提及链来自的文档、计数或其他潜在有用的信息。这样做是为了尽量减小内存表示。如果需要查找实体被提及的所有句子或文档（这是一个常见任务），那么实体 ID 到文档的映射必须存储在其他地方。XDoc 执行后生成的文档 XML 表示是解决这些需求的一个自然起点。

#### `createEntitySpeculative()` 方法

创建一个推测性找到的新实体只需要确定哪些别名是将提及链连接起来的好候选。适合跨文档共指的那些别名进入 `xdcPhrases` 集合，其他的进入 `nonXdc` 短语集合：

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

`boolean` 方法 `XdcPhrase()` 在 XDoc 过程中扮演着关键角色。当前的方法支持一种非常保守的对什么是好 XDoc 短语的定义。直觉上，在新闻领域，诸如 `he`、`Bob` 和 `John Smith` 这样的短语并不好，无法有效地指示正在讨论的独特个体。好的短语可能是 `Breckenridge Baldwin`，因为那可能是一个独特的名字。有很多复杂的理论解释这里发生了什么，参见刚性指示符（[`en.wikipedia.org/wiki/Rigid_designator`](http://en.wikipedia.org/wiki/Rigid_designator)）。接下来的几行代码几乎抹去了 2,000 年的哲学思想：

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

这种方法试图识别 XDoc 中的不良短语，而不是好的短语。推理如下：

+   **短语已经与一个实体相关联**：这强制假设世界上只有一个 John Smith。这对于情报收集应用程序非常有效，因为分析师在分辨 `John Smith` 案例时几乎没有困难。你可以参考本章末尾的 *The John Smith problem* 配方了解更多内容。

+   **短语只有一个单词，并且与提及链或实体相关联的有多个单词短语**：这假设较长的单词对 XDoc 更有利。请注意，实体创建的不同顺序可能导致单个单词短语在具有多单词别名的实体上，`xdc` 为 `true`。

+   **短语是代词**：这是一个相对安全的假设，除非我们处在宗教文本中，其中句中间大写的 `He` 或 `Him` 表示指向上帝。

一旦知道了 `xdc` 和 `nonXdc` 短语的集合，实体就会被创建。请参阅 `Entity.java` 的源代码，了解实体是如何创建的。

然后，实体被创建，`add` 方法更新 `EntityUniverse` 文件中从 `xdc` 短语到实体 ID 的映射：

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

`EntityUniverse` 文件的全局 `mXdcPhraseToEntitySet` 变量是找到候选实体的关键，正如在 `xdcEntitiesToPhrase()` 中使用的那样。

#### `XDocCoref.addMentionChainToEntity()` 实体

返回到 `XDocCoref.xdocCoref()` 方法，我们已经介绍了如何通过 `XDocCoref.promote()` 创建一个新实体。接下来要讨论的选项是当提及链被解析为现有实体时会发生什么，即 `XDocCoref.addMentionChainToEntity()`。为了添加推测性提及，实体必须允许通过 `Entity.allowSpeculativeAliases()` 方法提供的推测性找到的提及。这是用户定义的字典实体的一个特性，已在用户定义实体中讨论过。如果允许推测性实体，则提及链会被添加到实体中，并且会根据它们是否为 `xdc` 短语来敏感处理：

```py
private void addMentionChainToEntity(TTMentionChain chain, 
                Entity entity) {
    for (String phrase : chain.normalPhrases()) {
             mEntityUniverse.addPhraseToEntity(normalPhrase,
                entity);
        }
    }
```

添加提及链到实体的唯一变化就是增加了一个新的短语。这些附加的短语会像在提及链的提升过程中那样被分类为是否为`xdc`。

到目前为止，我们已经了解了文档中的提及链是如何被提升为猜测实体，或者如何与`EntityUniverse`中的现有实体合并的。接下来，我们将探讨在`XDocCoref.resolveMentionChain()`中解析是如何进行的。

#### `XDocCoref.resolveMentionChain()`实体

`XDocCoref.resolveMentionChain()`方法组装了一个可能与被解析的提及链匹配的实体集合，并通过调用`XDocCoref.resolveCandidates()`尝试找到唯一的实体：

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

该代码通过调用`EntityUniverse.xdcEntitiesWithPhrase()`从实体宇宙中查找实体集合。所有提及链的别名都会被尝试，而不考虑它们是否是有效的 XDoc 别名。在将实体添加到`candidateEntities`之前，返回的类型必须与`TTMatchers.unifyEntityTypes`所确定的提及链类型一致。这样，`华盛顿`（地点）就不会被解析为`华盛顿`（人名）。在此过程中，会做一些记录工作，以确定提及链上最长的别名是否与某个实体匹配。

#### `resolveCandidates()`方法

`resolveCandidates()`方法捕捉了一个关键假设，这一假设适用于文档内和 XDoc 共指的情况——这种不歧义的引用是唯一的解析基础。在文档内的案例中，一个类似的问题是“Bob 和 Joe 一起工作。他掉进了脱粒机。”这里的“他”指的是谁？单一指代词有唯一先行词的语言预期被称为唯一性假设。一个 XDoc 的例子如下：

+   **Doc1**：约翰·史密斯是《风中奇缘》中的一个角色

+   **Doc2**：约翰·史密斯是董事长或总经理

+   **Doc3**：约翰·史密斯受人尊敬

`Doc3`中的`约翰·史密斯`与哪个`约翰·史密斯`相匹配？也许，两者都不是。这个软件中的算法要求在匹配标准下应该有一个唯一的实体得以保留。如果有多个或没有，系统就会创建一个新的实体。其实现方式如下：

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

`filterCandidates`方法会删除因各种语义原因无法通过的所有候选实体。只有当实体宇宙中的一个实体有唯一的匹配时，才会发生共指。这里并没有区分候选实体过多（多个）或过少（零）的情况。在一个更高级的系统中，如果实体过多，可以尝试通过`context`进一步消除歧义。

这是 XDoc 代码的核心。其余的代码使用`xdocCoref`方法返回的实体宇宙相关索引对文档进行标注，这部分我们刚刚已经讲解过：

```py
Entity[] entities  = mXDocCoref.xdocCoref(chains);
```

以下的`for`循环遍历了提到的链，这些链与`xdocCoref`返回的`Entities[]`对齐。对于每一个提到的链，提到的内容会被映射到它的跨文档实体：

```py
Map<Mention,Entity> mentionToEntityMap
     = new HashMap<Mention,Entity>();
for (int i = 0; i < chains.length; ++i){ 
  for (Mention mention : chains[i].mentions()) {
         mentionToEntityMap.put(mention,entities[i]);
  }
}
```

接下来，代码将设置一系列映射，创建反映实体宇宙 ID 的块：

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

实际的块创建在下一步进行：

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

然后，块被用来创建文档的相关部分，并返回`OutputDocument`：

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

这是我们为 XDoc 共指提供的起点。希望我们已经解释了更多晦涩方法背后的意图。祝你好运！

# 约翰·史密斯问题

不同的人、地点和概念可能有相同的书面表示，但却是不同的。世界上有多个“约翰·史密斯”、“巴黎”和“银行”的实例，适当的跨文档共指系统应该能够处理这些情况。对于“银行”这样的概念（例如：河岸和金融银行），术语是词义消歧。本示例将展示巴尔温（Baldwin）和阿米特·巴加（Amit Bagga）当年为人物消歧开发的一个方法。

## 准备工作

这个示例的代码紧跟[`alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html`](http://alias-i.com/lingpipe/demos/tutorial/cluster/read-me.html)的聚类教程，但进行了修改，以更贴合最初的 Bagga-Baldwin 工作。代码量不小，但没有非常复杂的部分。源代码在`src/com/lingpipe/cookbook/chapter7/JohnSmith.java`。

该类首先使用了标准的 NLP 工具包，包括分词、句子检测和命名实体检测。如果这个工具堆栈不熟悉，请参阅前面的示例：

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

接下来，我们将重新访问`TfIdfDistance`。不过，任务要求我们将类封装成处理`Documents`而非`CharSequences`，因为我们希望保留文件名，并能够操作用于后续计算的文本：

```py
TfIdfDocumentDistance tfIdfDist = new TfIdfDocumentDistance(tokenizerFactory);
```

降级到引用的类，我们有以下代码：

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

`train`方法与`TfIdfDistance.handle()`方法接口，并提供了一个`distance(Document doc1, Document doc2)`方法的实现，驱动下面讨论的聚类代码。`train`方法的作用仅仅是提取相关文本，并将其交给`TfIdfDistance`类来计算相关值。

引用类`Document`是`JohnSmith`中的一个内部类，非常简单。它获取包含匹配`.*John Smith.*`模式的实体的句子，并将其放入`mCoreferentText`变量中：

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

深入到代码中，我们现在将访问`getCoreferentSents()`方法：

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

查看*跨文档共指*的配方，了解前面方法的大部分运动部分。我们将挑出一些值得注意的部分。某种意义上，我们通过使用正则表达式分块器来找到任何包含`John Smith`子字符串的字符串，并将其作为`PERSON`实体添加进来，算是作弊。像大多数类型的作弊一样，如果你的人生目标仅仅是追踪`John Smith`，这种方法相当有效。实际上，我们做的作弊是使用字典匹配来找到`Osama bin Laden`等高价值情报目标的所有变种。最终，在 MiTAP 项目中，我们找到了超过 40 个版本的他的名字，遍历公开的新闻来源。

此外，在处理每个句子时，我们会检查所有提及的内容是否匹配`John Smith`的模式，如果匹配，则收集包含该 ID 的句子。这意味着，任何提到`John Smith`的句子，包括用代词指代的句子，如果共指工作正常，`Mr. Smith`的情况也会被包括在内。注意，我们需要看到`John Smith`的匹配才能开始收集上下文信息，所以我们会错过句子`He awoke. John Smith was a giant cockroach`的第一个句子。同时，如果第二个`John Smith`出现并带有不同的 ID，它将被忽略——这种情况是可能发生的。

最后，注意有一些错误检查，如果找不到`John Smith`，系统会向`System.out`报告错误。

如果我们在设置好`TfIdfDocumentDistance`后又回到`main()`方法中的普通 I/O 处理，我们将会有：

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

我们没有讨论这个问题，但关于哪个文档引用了哪个`Mr. Smith`的真实注解编码在数据的目录结构中。`johnSmith`顶级目录中的每个子目录都被视为真实聚类。所以，`referencePartition`包含了真实数据。我们本可以将其包装为一个分类问题，每个子目录对应正确的分类。我们将这个作为练习留给你，要求将其嵌入到交叉验证语料库中，并用逻辑回归解决。

接下来，我们将通过将之前的类别展平为一个`Documents`的集合来构建测试集。我们本可以在前一步完成这个操作，但混合任务往往会产生错误，而且多出的`for`循环对执行速度几乎没有影响：

```py
        Set<Document> docSet = new HashSet<Document>();
        for (Set<Document> cluster : referencePartition) {
            docSet.addAll(cluster);
        }
```

接下来，我们将启动聚类算法。我们将执行`CompleteLink`和`SingleLink`，由`TfIdfDocumentDistance`驱动，后者负责整个过程：

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

聚类算法的细节在第五章中进行了介绍，*文本中的跨度查找 – 分块*。现在，我们将根据聚类数从`1`到输入数量的变化来报告性能。一个特别的地方是，`Cross`类别使用`SingleLinkClusterer`作为参考，而`CompleteLinkClusterer`作为响应：

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

这就是我们为准备这个配方所需做的一切。这是一个罕见的现象要计算，这是一个玩具实现，但关键概念应该是显而易见的。

## 如何做...

我们只需运行这段代码，稍微调整一下：

1.  到终端并输入：

    ```py
    java -cp lingpipe-cookbook.1.0.jar:lib/lingpipe-4.1.0.jar: com.lingpipe.cookbook.chapter7.JohnSmith

    ```

1.  结果将是一堆信息，指示正在提取用于聚类的句子——记住真相注释是由文件所在目录确定的。第一个聚类是`0`：

    ```py
    Category from file=data/johnSmith/0
    ```

1.  代码报告包含对`John Smith`的引用的句子：

    ```py
    Adding I thought John Smith marries Pocahontas.''
         mention text=John Smith type=PERSON id=5
    Adding He's bullets , she's arrows.''
         mention text=He type=MALE_PRONOUN id=5
    ```

1.  对`John Smith`的代词引用是包含第二句的基础。

1.  系统输出继续进行，最后，我们将获得与真相进行单链接聚类和与真相进行完全链接的结果。`K`列指示算法允许多少个聚类，并报告了精确度、召回率和 F-度量。第一行在这种情况下是只允许一个聚类，将允许百分之百的召回率和 23%的精确度，无论是完全链接还是单链接。查看得分，我们可以看到完全链接在`0.60`时报告了最佳的 F-度量——事实上，有 35 个聚类。单链接方法在`0.78`时将 F-度量最大化到 68 个聚类，并在不同数量的聚类上显示出更大的鲁棒性。交叉案例显示单链接和完全链接在直接比较中有很大的不同。请注意，为了可读性，一些`K`值已被消除：

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

1.  下面的输出限制了聚类的方式不是通过聚类大小，而是通过最大距离阈值。输出是对单链接聚类的，增加了`.05`距离，并且评估是 B-cubed 度量。输出是距离、精确度、召回率以及生成聚类的大小。在`.80`和`.9`的表现非常好，但要小心在事后设置生产阈值。在生产环境中，我们将希望在设置阈值之前看到更多数据：

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

1.  B-cubed（Bagga、Bierman 和 Baldwin）评估被设计为严重惩罚将大量文档关联在一起的情况。它假设将关于乔治·W·布什和乔治·H·W·布什这样的大型聚类合并在一起是更大的问题，而不是误将提到数据集中的一次性提到的机械师乔治·布什的情况。其他评分指标将同样认为这两种错误同样糟糕。这是文献中用于此现象的标准评分指标。

## 另请参阅

在研究文献中，关于这个具体问题有相当多的工作。我们并不是第一个考虑这个问题的人，但我们提出了主流的评估指标，并发布了一个语料库供其他团队与我们以及彼此进行比较。我们的贡献是*基于实体的跨文档共指消解，使用向量空间模型*，由 Bagga 和 Baldwin 提出，收录于*ACL '98 第 36 届计算语言学会年会和第 17 届国际计算语言学会议论文集*。自那时以来已经取得了许多进展——Google Scholar 上已有超过 400 次引用，如果这个问题对你来说很重要，它们值得一看。
