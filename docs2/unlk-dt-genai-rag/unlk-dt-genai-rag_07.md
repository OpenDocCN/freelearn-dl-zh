# 7

# 向量和向量存储在 RAG 中扮演的关键角色

**<st c="52">向量</st>** <st c="60">是</st> <st c="66">检索增强生成</st> **<st c="84">（RAG）</st> 的一个关键组成部分，需要理解，因为它们是帮助整个过程顺利进行的秘密成分。</st> <st c="212">在本章中，我们将重新审视前几章的代码，重点关注向量对其的影响。</st> <st c="330">简单来说，本章将讨论向量是什么，向量是如何创建的，以及</st> <st c="426">然后在哪里存储它们。</st> <st c="453">从更技术性的角度来说，我们将讨论向量、</st> **<st c="506">向量化</st>**<st c="519">，以及</st> **<st c="525">向量存储</st>**<st c="538">。本章全部关于向量的创建以及为什么</st> <st c="590">它们很重要。</st> <st c="610">我们将关注向量与 RAG 的关系，但我们鼓励你花更多的时间和精力去深入研究向量，尽可能获得深入的理解。</st> <st c="781">你对向量的理解越深入，你在改进你的</st> <st c="863">RAG 流水线</st> 时就会越有效。

向量讨论的重要性如此之高，以至于我们将它扩展到两个章节中。</st> <st c="967">虽然本章重点讨论向量和向量存储，</st> *<st c="1024">第八章</st> <st c="1033">将重点讨论向量搜索，也就是说向量如何在 RAG 系统中被使用。</st>

在本章中，我们将具体涵盖以下主题：<st c="1122">以下：</st>

+   RAG 中的向量基础

+   向量在你代码中的位置

+   你向量化的文本量很重要！

+   并非所有语义都是平等的！

+   常见的向量化技术

+   选择向量化选项

+   开始使用向量存储

+   向量存储

+   选择一个向量存储

# 技术要求

回顾我们在过去章节中讨论的代码，本章将重点讨论这一行代码：<st c="1596">代码：</st>

```py
 vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings())
```

本章的代码在此：<st c="1689">这里：</st> [<st c="1725">https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_07</st>](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_07 )

文件名 <st c="1822">是</st> `<st c="1839">CHAPTER7-1_COMMON_VECTORIZATION_TECHNIQUES.ipynb</st>`<st c="1887">。</st>

<st c="1888">并且</st> *<st c="1893">第八章</st>* <st c="1902">将专注于这一行</st> <st c="1932">代码：</st>

```py
 retriever = vectorstore.as_retriever()
```

<st c="1979">就这样了吗？</st> <st c="1992">就这两行代码对应两章内容吗？</st> <st c="2039">是的！</st> <st c="2044">这显示了向量对 RAG 系统的重要性。</st> <st c="2104">为了彻底理解向量，我们从基础开始，并在此基础上构建。</st>

<st c="2197">让我们</st> <st c="2204">开始吧！</st>

# <st c="2216">RAG 中向量的基础</st>

<st c="2247">在本节中，我们</st> <st c="2256">将涵盖与向量、嵌入在自然语言处理</st> **<st c="2356">（NLP</st>**<st c="2383">）和 RAG 的相关的重要主题。</st> <st c="2400">我们将首先阐明向量和嵌入之间的关系，解释嵌入是 NLP 中使用的特定类型的向量表示。</st> <st c="2562">然后，我们将讨论向量的属性，如它们的维度和大小，以及这些特征如何影响文本搜索和</st> <st c="2728">相似度比较的精度和有效性。</st>

## <st c="2751">嵌入和向量之间的区别是什么？</st>

<st c="2806">向量和</st> **<st c="2819">嵌入</st>** <st c="2829">是自然语言处理（NLP）和 RAG 系统中的关键概念，在构建语言模型和 RAG 系统中发挥着至关重要的作用。</st> <st c="2923">但它们是什么，它们之间有什么关系呢？</st> <st c="2979">简单来说，你可以将嵌入视为一种特定的向量表示。</st> <st c="3070">当我们谈论 RAG 中使用的</st> **<st c="3100">大型语言模型</st>** <st c="3121">（**<st c="3123">LLMs</st>**<st c="3127">）时，它们是 NLP 这个更大宇宙的一部分，我们使用的向量被称为嵌入。</st> <st c="3243">另一方面，一般来说，向量被广泛应用于各种领域，可以代表许多其他对象，而不仅仅是语言结构（如单词、句子、段落等）。</st> <st c="3439">在谈论 RAG 时，嵌入、向量、向量嵌入和嵌入向量这些词可以</st> <st c="3546">互换使用！</st>

<st c="3567">现在我们已经解决了这个问题，让我们来谈谈向量实际上是什么。</st> <st c="3637">是什么。</st>

## <st c="3649">什么是向量？</st>

<st c="3667">当你听到单词</st> <st c="3676">向量</st> <st c="3734">时，你首先想到的是什么？许多人会说数学。</st> <st c="3764">这将是准确的；向量实际上是我们在数据处理中使用的文本的数学表示，并且它们允许我们以新的和非常</st> <st c="3956">有用的方式对我们的数据进行数学运算。</st>

<st c="3968">单词</st> *<st c="3978">向量</st> <st c="3984">也可能让你想到速度。</st> <st c="4021">这也是准确的；与向量相比，我们可以以比任何先于向量搜索的技术都要快的速度进行文本搜索。</st>

<st c="4175">与单词</st> *<st c="4231">向量</st> <st c="4237">经常相关联的另一个概念是精度。</st> <st c="4252">通过将文本转换为具有语义表示的嵌入，我们可以显著提高我们搜索系统的精度，以便找到我们正在寻找的内容。</st> <st c="4410">我们正在寻找的内容。</st>

<st c="4422">当然，如果你是电影</st> *<st c="4468">《神偷奶爸》</st> <st c="4481">的粉丝，你可能会想到反派角色 Vector，他自称是“</st>*<st c="4564">我名叫……Vector。</st> <st c="4594">这是一个数学术语，一个由箭头表示的具有方向</st>* *<st c="4675">和大小</st>*<st c="4688">。”</st>

<st c="4690">他可能是一个做可疑事情的反派，但他对他名字背后的含义是正确的！从这个描述中我们可以得出的关键点是，向量不仅仅是一堆数字；它是一个数学对象，它代表了大小和方向。</st> <st c="4789">这也是为什么它比仅仅表示数字更能代表你的文本和文本之间的相似性，因为它捕捉了它们更复杂的形式。</st> <st c="4961">这就是为什么它比简单的数字更能代表你的文本和文本之间的相似性，因为它捕捉了它们更复杂的形式。</st> <st c="5104">这是为什么它比简单的数字更能代表你的文本和文本之间的相似性。</st>

<st c="5119">这可能会让你对向量是什么有一个理解，但接下来让我们讨论对 RAG 开发有影响的重要的向量方面，首先是</st> <st c="5296">向量大小。</st>

## <st c="5308">向量维度和大小</st>

<st c="5335">Vector，</st> <st c="5348">来自</st> *<st c="5361">《神偷奶爸》</st>*<st c="5374">的反派角色说，向量是“</st>*<st c="5399">一个由箭头表示的量</st>*<st c="5434">。”但尽管在二维或三维图上思考表示向量的箭头可以更容易地理解向量是什么，重要的是要</st> <st c="5573">理解我们处理的向量通常在两个或三个维度以上表示。</st> <st c="5685">向量的维度数也被称为向量大小。</st> <st c="5764">为了在我们的代码中看到这一点，我们将在定义我们的变量下方添加一个新的单元。</st> <st c="5863">此代码将打印出嵌入向量</st> `<st c="5911">的一部分</st>` <st c="5920">：</st>

```py
 question = "What are the advantages of using RAG?" question_embedding=embedding_function.embed_query(question)first_5_numbers = question_embedding[:5]
print(f"User question embedding (first 5 dimensions):
    {first_5_numbers}")
```

在此代码中，我们采用了我们在代码示例中一直使用的**问题**，《<st c="6237">使用 RAG 的优势是什么？</st>》，并将其使用 OpenAI 的嵌入 API 转换为向量表示。</st> <st c="6354">问题嵌入</st> <st c="6376">变量代表这个嵌入。</st> <st c="6413">使用切片</st> `<st c="6428">[0:5]</st>`<st c="6433">，我们从</st> `<st c="6471">问题嵌入</st>`<st c="6489">中取出前五个数字，这代表向量的前五个维度，并将它们打印出来。</st> <st c="6568">完整的向量包含 1,536 个浮点数，每个数有 17-20 位数字，因此我们将打印的内容最小化，以便更容易阅读。</st> <st c="6720">这个单元格的输出将看起来</st> <st c="6754">像这样：</st>

```py
 User question embedding (first 5 dim): [
-0.006319054113595048, -0.0023517232115089787, 0.015498643243434815, -0.02267445873596028, 0.017820641897159206]
```

<st c="6918">我们在这里只打印出前五个维度，但嵌入的大小远大于这个。</st> <st c="7013">我们将在稍后讨论确定维度总数的一种实用方法，但首先我想将你的注意力引向每个数字的长度。</st>

<st c="7174">这些嵌入中的所有数字都将有 +/-0 的小数点，因此让我们谈谈小数点后有多少位数字。</st> <st c="7319">这里的第一个数字</st> `<st c="7342">-0.006319054113595048</st>`<st c="7363">，小数点后有 18 位数字，第二个数字有 19 位，第四个数字有 17 位。</st> <st c="7460">这些数字长度与 OpenAI 的嵌入模型</st> `<st c="7581">OpenAIEmbeddings</st>`<st c="7597">使用的浮点数表示精度有关。</st> <st c="7734">这个高精度格式提供了 64 位数字（也称为</st> **<st c="7714">双精度</st>**<st c="7730">）。</st> <st c="7734">这种高精度导致</st> <st c="7765">了非常精细的区分和准确表示嵌入模型捕获的语义信息。</st>

<st c="7902">此外，让我们回顾一下在</st> *<st c="7946">第一章</st>*<st c="7955">中提到的一个观点，即前面的输出看起来非常像 Python 的浮点数列表。</st> <st c="8034">实际上，在这种情况下它确实是一个 Python 列表，因为这是 OpenAI 从他们的嵌入 API 返回的内容。</st> <st c="8134">这可能是为了使其与 Python 编码世界更加兼容而做出的决定。</st> <st c="8219">但为了避免混淆，重要的是要理解，在机器学习领域，当你看到这种用于机器学习相关处理的内容时，它通常是 NumPy 数组，尽管数字列表和 NumPy 数组在打印输出时看起来相同，就像我们</st> <st c="8547">刚才做的那样。</st>

<st c="8556">有趣的事实</st>

<st c="8565">如果你与生成式 AI 一起工作，你最终会听到被称为</st> <st c="8600">量化</st> **<st c="8616">的概念</st> <st c="8628">。如果你与生成式 AI 一起工作。</st> <st c="8661">与嵌入类似，量化处理高精度浮点数。</st> <st c="8739">然而，在量化中，概念是将模型参数，如权重和激活，从它们原始的高精度浮点表示转换为低精度格式。</st> <st c="8938">这减少了 LLM 的内存占用和计算需求，这可以应用于使其在预训练、训练和微调 LLM 时更具成本效益。</st> <st c="9111">量化还可以使使用 LLM 进行推理更具成本效益，这就是当你使用 LLM 获取响应时所说的。</st> <st c="9262">当我在这句话中说</st> *<st c="9273">成本效益</st>* <st c="9287">时，我指的是能够在</st> <st c="9357">更小、更便宜的硬件环境中完成这些事情。</st> <st c="9404">然而，有一个权衡；量化是一种</st> **<st c="9452">有损压缩技术</st>**<st c="9479">，这意味着在转换过程中会丢失一些信息。</st> <st c="9561">量化 LLM 的降低精度可能与原始</st> <st c="9663">高精度 LLM 相比导致精度损失。</st>

<st c="9683">当你在使用 RAG 并考虑将文本转换为嵌入的不同算法时，请注意嵌入值的长度，以确保如果你在 RAG 系统中对准确性和响应质量有较高要求，你正在使用高精度浮点格式。</st> <st c="9965">RAG 系统。</st>

<st c="9976">但是这些嵌入表示了多少维度呢？</st> <st c="10038">在前面的例子中，我们只展示了五个，但我们本可以将它们全部打印出来并单独计数。</st> <st c="10152">这当然看起来不太实际。</st> <st c="10188">我们将使用</st> `<st c="10204">len()</st>` <st c="10209">函数来为我们计数。</st> <st c="10246">在下面的代码中，你可以看到这个有用的函数被很好地利用，给出了这个嵌入的总大小：</st> <st c="10344">如下：</st>

```py
 embedding_size = len(question_embedding)
print(f"Embedding size: {embedding_size}")
```

<st c="10443">此代码的输出如下：</st> <st c="10471">如下：</st>

```py
 Embedding size: 1536
```

<st c="10503">这表明这个嵌入是 1,536 维度！</st> <st c="10560">当我们通常最多只考虑 3 维时，在脑海中尝试可视化这很困难，但这些额外的 1,533 维度在如何精确地表示相关文本的嵌入语义表示方面产生了显著差异。</st>

<st c="10809">在大多数现代向量化算法中处理向量时，通常会有数百或数千个维度。</st> <st c="10936">维度的数量等于表示嵌入的浮点数的数量，这意味着一个 1,024 维度的向量由 1,024 个浮点数表示。</st> <st c="11107">嵌入的长度没有硬性限制，但一些现代向量化算法倾向于预设大小。</st> <st c="11236">我们使用的模型，OpenAI 的</st> `<st c="11269">ada</st>` <st c="11272">嵌入模型，默认使用 1,536。</st> <st c="11313">这是因为它是训练来产生特定大小的嵌入，如果你尝试截断该大小，它将改变嵌入中捕获的上下文。</st> <st c="11454">的嵌入。</st>

<st c="11468">然而，这种情况正在改变。</st> <st c="11496">现在有新的向量化工具可用（例如 OpenAI 的</st> `<st c="11550">text-embedding-3-large</st>` <st c="11572">模型），它允许你更改向量大小。</st> <st c="11620">这些嵌入模型被训练来在不同向量维度大小上提供相对相同的内容。</st> <st c="11750">这使一种称为</st> **<st c="11782">自适应检索</st>**<st c="11800">的技术成为可能。</st>

<st c="11801">使用自适应检索，你会在不同大小下生成多组嵌入。</st> <st c="11888">你首先搜索低维向量以帮助你</st> *<st c="11944">接近</st>* <st c="11949">最终结果，因为搜索低维向量比搜索高维向量快得多。</st> <st c="12070">一旦你的低维搜索将你带入与你的输入查询最相似的内容附近，你的搜索</st> *<st c="12190">适应</st>* <st c="12196">到搜索速度较慢、维度较高的嵌入，以定位最相关的内容并最终完成相似度搜索。</st> <st c="12335">总的来说，这可以提高你的搜索速度 30-90%，具体取决于你如何设置搜索。</st> <st c="12432">这种技术生成的嵌入被称为</st> **<st c="12486">套娃嵌入</st>**<st c="12507">，得名于俄罗斯套娃，反映了嵌入，就像娃娃一样，彼此之间相对相同，但在大小上有所不同。</st> <st c="12662">如果你需要在生产环境中优化用于重用场景的 RAG 管道，你将需要考虑</st> <st c="12785">这种技术。</st>

<st c="12800">下一个你需要理解的概念是代码中你的向量所在的位置，这有助于你将你正在学习的关于向量的概念直接应用到你的</st> <st c="12987">RAG 努力中。</st>

# <st c="12999">你的代码中向量潜伏之处</st>

<st c="13031">一种表示 RAG 系统中向量值的方法是向您展示它们被使用的所有地方。<st c="13043">如前所述，你从你的文本数据开始，在向量化过程中将其转换为向量。</st> <st c="13136">这发生在 RAG 系统的索引阶段。</st> <st c="13248">但是，在大多数情况下，你必须有一个地方来存放这些嵌入向量，这就引入了向量存储的概念。</st> <st c="13301">。

在 RAG 系统的检索阶段，你从用户输入的问题开始，在检索开始之前，首先将其转换为嵌入向量。<st c="13425">最后，检索过程使用一个相似度算法来确定问题嵌入与向量存储中所有嵌入之间的接近程度。<st c="13599">还有一个潜在的领域，向量是常见的，那就是当你想要评估你的 RAG 响应时，但我们将在这*<st c="13907">第九章</st>* <st c="13916">当我们介绍评估技术时</st> <st c="13954">进行讨论。现在，让我们更深入地探讨这些其他概念，从向量化开始。</st> <st c="14025">。

## <st c="14044">向量化发生在两个地方</st>。

<st c="14079">在 RAG 过程的非常前端，你通常有一个机制让用户输入一个问题，这个问题被传递给检索器。</st> <st c="14213">我们在我们的代码中看到这个过程正在发生：</st> <st c="14260">。

```py
 rag_chain_with_source = RunnableParallel(
    {"context": <st c="14325">retriever</st>,
     "question":RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)
```

<st c="14406">检索器是一个 LangChain</st> `<st c="14436">检索器</st>` <st c="14445">对象，它简化了基于用户查询的相似度搜索和相关性向量的检索。</st> <st c="14547">因此，当我们谈论向量化时，它实际上在我们的代码中发生在两个地方：</st> <st c="14620">。

+   首先，当我们对将要用于 RAG 系统的原始数据进行向量化时<st c="14629">。

+   <st c="14708">其次，当我们需要向量化用户查询时<st c="14747">。</st>。

这两个单独步骤之间的关系是它们都用于相似度搜索。<st c="14757">不过，在我们谈论搜索之前，让我们先谈谈后者组嵌入，即原始数据嵌入的存储位置：向量存储。</st> <st c="14861">。

## <st c="15030">向量数据库/存储存储和包含向量</st>。

<st c="15080">向量存储</st> <st c="15095">通常是一个向量数据库（但并非总是如此，见以下说明），它针对存储和提供向量进行了优化，并在有效的 RAG 系统中发挥着关键作用。</st> <st c="15272">技术上，你可以不使用向量数据库构建 RAG 系统，但你将错过这些数据存储工具中已经构建的许多优化，这会影响你的内存、计算需求以及搜索</st> <st c="15512">精度，这是不必要的。</st>

<st c="15536">注意</st>

<st c="15541">你经常听到</st> **<st c="15566">向量数据库</st>** <st c="15582">这个术语</st>，当<st c="15587">提到用于存储向量的优化数据库结构时。</st> <st c="15657">然而，有一些工具和其他机制虽然不是数据库，但它们在功能上与向量数据库相同或类似。</st> <st c="15790">因此，我们将它们统称为</st> *<st c="15850">向量存储</st>*<st c="15863">。这与 LangChain 文档中的表述一致，它也将这一组称为向量存储，包括所有存储和提供向量的机制类型。</st> <st c="16043">但你会经常听到这些术语被互换使用，而术语</st> *<st c="16112">向量数据库</st>** <st c="16127">实际上是更常用的术语，用来指代所有这些机制。</st> <st c="16204">为了准确起见，并使我们的术语与 LangChain 文档保持一致，在这本书中，我们将使用术语</st> *<st c="16323">向量存储</st>**<st c="16335">。</st>

<st c="16336">在</st> *<st c="16349">向量在你的代码中隐藏的位置</st>**<st c="16380">方面，向量存储是存储你代码中生成的大多数向量的地方。</st> <st c="16463">当你将数据向量化时，这些嵌入会进入你的向量存储。</st> <st c="16537">当你进行相似度搜索时，用于表示数据的嵌入会从向量存储中提取。</st> <st c="16652">这使得向量存储在 RAG 系统中扮演着关键角色，值得我们关注。</st>

<st c="16736">既然我们已经知道了原始数据嵌入的存储位置，让我们将其与用户</st> <st c="16868">查询嵌入的使用联系起来。</st>

## <st c="16885">向量相似度比较你的向量</st>

<st c="16925">我们有我们的两个主要</st> <st c="16950">向量化事件：</st>

+   <st c="16976">我们</st> <st c="16999">用户查询</st> 的嵌入

+   <st c="17009">代表我们</st> <st c="17065">向量存储中所有数据的</st> 向量嵌入

<st c="17077">让我们</st> <st c="17084">回顾一下这两次发生的事件是如何相互关联的。</st> <st c="17139">当我们进行高度重要的向量相似度搜索，这是我们的检索过程的基础时，我们实际上只是在执行一个数学运算，该运算测量用户查询嵌入和原始</st> <st c="17385">数据嵌入之间的距离。</st>

<st c="17401">可以使用多种数学算法来执行这种距离计算，我们将在后面的</st> *<st c="17515">第八章</st>*<st c="17524">中对其进行回顾。但就目前而言，重要的是要理解这种距离计算确定了与用户查询嵌入最接近的原始数据嵌入，并按距离顺序（从最近到最远）返回这些嵌入的列表。</st> <st c="17781">我们的代码稍微简单一些，因为嵌入以 1:1 的关系表示数据点（块）。</st>

<st c="17900">但在许多应用中，例如在与问答聊天机器人一起使用时，问题或答案可能非常长，并被分成更小的块，你可能会看到这些块有一个外键 ID，它引用回更大的内容。</st> <st c="18152">这使我们能够检索整个内容，而不仅仅是块。</st> <st c="18234">这会根据你的 RAG 系统试图解决的问题而变化，但重要的是要理解，这个检索系统的架构可以根据应用的需求而变化。</st>

<st c="18437">这涵盖了你在你的 RAG 系统中找到向量的最常见地方：它们出现的地方，它们存储的地方，以及它们如何在 RAG 系统的服务中使用。</st> <st c="18603">在下一节中，我们将讨论我们在 RAG 系统的搜索中使用的数据文本的大小是如何变化的。</st> <st c="18724">你最终会在你的代码中做出决定，这将决定那个大小。</st> <st c="18796">但根据你对向量的了解，你可能开始想知道，如果我们将各种大小的内容矢量化，这会如何影响我们比较它们的能力，并最终构建我们能够构建的最有效的检索过程？</st> <st c="19036">你确实有理由感到好奇！</st> <st c="19070">让我们接下来讨论我们将内容转换为</st> <st c="19140">嵌入时内容大小的影响。</st>

# <st c="19156">你矢量化文本的数量很重要！</st>

我们之前展示的向量<st c="19198">来自文本</st> <st c="19227">《</st> <st c="19247">使用 RAG 的优势是什么？</st> <st c="19284">》</st>。这是一段相对较短的文字，这意味着一个 1,536 维度的向量将能够非常彻底地代表该文本中的上下文。</st> <st c="19444">但如果我们回到代码，我们矢量化以表示我们的*<st c="19522">数据</st> <st c="19526">的内容</st> <st c="19533">来自这里：</st>

```py
 loader = WebBaseLoader(
    web_paths=("https://kbourne.github.io/chapter1.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title",
                    "post-header")
        )
    ),
)
docs = loader.load()
```

<st c="19749">这引入了我们在前几章中查看的网页，与问题文本相比，它相对较长。</st> <st c="19868">为了使这些数据更易于管理，我们使用以下代码中的文本分割器将这些内容分割成片段：</st>

```py
 text_splitter = SemanticChunker(embedding_function)
splits = text_splitter.split_documents(docs)
```

<st c="20072">如果你使用</st> `<st c="20119">splits[2]</st>`<st c="20128">提取第三个片段，它看起来会是这样：</st>

```py
 There are also generative models that generate images from text prompts, while others generate video from text prompts. There are other models that generate text descriptions from images. We will talk about these other types of models in Chapter 16, Going Beyond the LLM. But for most of the book, I felt it would keep things simple and let you focus on the core principles of RAG if we focus on the type of model that most RAG pipelines use, the LLM. But I did want to make sure it was clear, that while the book focuses primarily on LLMs, RAG can also be applied to other types of generative models, such as those for images and videos. Some popular examples of LLMs are the OpenAI ChatGPT models, the Meta LLaMA models, Google's PaLM and Gemini models, and Anthropic's Claude models. Foundation model\nA foundation model is the base model for most LLMs. In the case of ChatGPT, the foundation model is based on the GPT (Generative Pre-trained Transformer) architecture, and it was fine-tuned for Chat. The specific model used for ChatGPT is not publicly disclosed. The base GPT model cannot talk with you in chatbot-style like ChatGPT does. It had to get further trained to gain that skill.
```

<st c="21348">我选择第三个片段来展示，因为它相对较短。</st> <st c="21421">大多数片段都大得多。</st> <st c="21457">我们使用的**<st c="21461">语义块分割文本分割器</st>** <st c="21491">试图使用语义来确定如何分割文本，使用嵌入来确定这些语义。</st> <st c="21528">理论上，这应该会给我们提供更好的基于上下文分割数据的片段，而不是仅仅基于任意大小的。</st>

<st c="21747">然而，有一个重要的概念需要理解，当涉及到嵌入时，它将影响你选择的分割器以及你嵌入的大小。</st> <st c="21762">这一切都源于这样一个事实，即无论你传递给矢量化算法的文本有多大，它仍然会给你一个与任何其他嵌入相同大小的嵌入。</st> <st c="21914">在这种情况下，这意味着用户查询嵌入将是 1,536 维，但向量存储中的所有那些长文本段也将是 1,536 维，尽管它们的实际文本长度相当不同。</st> <st c="22123">这可能看起来有些反直觉，但令人惊讶的是，它</st> <st c="22434">效果很好！</st>

<st c="22445">当使用向量存储的用户查询进行搜索时，用户查询嵌入和其他嵌入的数学表示是以一种方式进行的，这样我们仍然能够检测它们之间的大小差异很大的语义相似性。</st> <st c="22737">向量相似性搜索的这一方面是让数学家如此热爱数学的原因之一。</st> <st c="22847">这似乎完全违背了逻辑，你可以将不同大小的文本转换为数字，并且能够检测它们之间的相似性</st> <st c="22974">。</st>

<st c="22987">但是，还有</st> <st c="23001">另一个方面需要考虑——当你只比较你将数据分割成块的结果时，这些块的大小将很重要。</st> <st c="23165">在这种情况下，被向量化内容量越大，嵌入就会越稀释。</st> <st c="23278">另一方面，嵌入表示的内容量越小，你在执行向量相似度搜索时需要匹配的上下文就越少。</st> <st c="23450">对于你的每个 RAG 实现，你都需要在块大小和</st> <st c="23552">上下文表示之间找到一个微妙的平衡。</st>

<st c="23575">理解这一点将帮助你在尝试改进你的 RAG 系统时，做出更好的关于如何分割数据和选择向量化算法的决定。</st> <st c="23740">当我们讨论 LangChain 分割器时，我们将在</st> *<st c="23831">第十一章</st>* <st c="23841">中介绍一些其他技术，以充分利用你的分割/分块策略。</st> <st c="23882">接下来，我们将讨论测试不同</st> <st c="23943">向量化模型的重要性。</st>

# <st c="23964">并非所有语义都是平等的！</st>

在 RAG 应用中常见的错误是选择第一个实现的向量化算法，并仅仅假设这会提供最佳结果。</st> 这些算法将文本的语义意义以数学方式表示。</st> 然而，这些算法本身通常是大型 NLP 模型，它们的能力和质量可以像 LLMs 一样有很大的差异。</st> 正如我们作为人类，常常发现理解文本的复杂性和细微差别具有挑战性一样，这些模型也会面临同样的挑战，它们在把握书面语言固有的复杂性方面具有不同的能力。</st> 例如，过去的模型无法区分</st> `<st c="24687">bark</st>` <st c="24691">(狗叫声) 和</st> `<st c="24710">bark</st>` <st c="24714">(大多数树木的外层)，但新模型可以根据周围的文本和使用的上下文来检测这一点。</st> 这个领域的这个方面正在以与其他领域一样快的速度适应和演变。</st>

<st c="24923">在某些情况下，可能一个特定领域的向量化模型，例如在科学论文上训练的模型，在专注于科学论文的应用程序中可能会比使用通用向量化模型表现得更好。</st> 科学家们的谈话方式非常具体，与你在社交媒体上看到的方式大不相同，因此在一个基于通用网络文本训练的大型模型可能在这个</st> <st c="25330">特定领域表现不佳。</st>

<st c="25346">有趣的事实</st>

<st c="25355">您经常听说如何微调 LLM 以改进您的特定领域结果。</st> <st c="25445">但您知道您也可以微调嵌入模型吗？</st> <st c="25508">微调嵌入模型有可能改善嵌入模型理解您特定领域数据的方式，因此有可能改善您的相似度搜索结果。</st> <st c="25711">这有可能显著改善您整个 RAG 系统在您领域中的表现。</st> <st c="25786">您的领域。</st>

<st c="25798">为了总结本节关于基础知识的部分，向量的多个方面在尝试为您的需求构建最有效的 RAG 应用时可能会帮助您，也可能伤害您。</st> <st c="25967">当然，如果不告诉您有哪些可用的向量化算法，就告诉您向量化算法的重要性，那将是不礼貌的！</st> <st c="26113">为了解决这个问题，在下一节中，让我们列举一些最受欢迎的向量化技术！</st> <st c="26231">我们甚至会用代码来做这件事！</st>

# <st c="26262">代码实验室 7.1 – 常见的向量化技术</st>

<st c="26309">向量化算法</st> <st c="26335">在过去几十年中已经发生了显著的变化。</st> <st c="26389">了解这些变化的原因，将帮助您获得更多关于如何选择最适合您需求的算法的视角。</st> <st c="26528">让我们回顾一些这些向量化算法，从一些最早的算法开始，到最新的、更高级的选项结束。</st> <st c="26683">这远非一个详尽的列表，但这些精选的几个应该足以让您了解这一领域的这一部分是从哪里来的，以及它将走向何方。</st> <st c="26851">在我们开始之前，让我们安装并导入一些在通过向量化技术进行编码之旅中扮演重要角色的新的 Python 包：</st> <st c="26974">这些代码应该放在上一个代码块中，与包安装相同的单元格中。</st>

```py
 %pip install gensim --user
%pip install transformers
%pip install torch
```

<st c="27071">此代码应放在上一个代码块中，与包安装相同的单元格中。</st>

## <st c="27178">词频-逆文档频率 (TF-IDF)</st>

<st c="27229">1972 年可能比您在关于相对较新的技术如 RAG 的书中预期的要早得多，但这就是我们将要讨论的向量化技术的根源。</st> <st c="27323">我们将要讨论的向量化技术。</st> <st c="27411">我们将要讨论的向量化技术。</st> <st c="27438">我们将要讨论的向量化技术。</st>

<st c="27449">凯伦·伊达·博尔特·斯帕克·琼斯是一位自学成才的程序员和开创性的英国计算机科学家，她在 NLP 领域的几篇论文中进行了研究。</st> <st c="27609">1972 年，她做出了她最重要的贡献之一，引入了</st> **<st c="27695">逆文档频率</st>** <st c="27721">(**<st c="27723">IDF</st>**<st c="27726">) 的概念。</st> <st c="27730">她所陈述的基本概念是，“</st>*<st c="27772">一个术语的特异性可以通过它在文档中出现的数量的倒数来量化</st>* *<st c="27876">。”</st>

<st c="27887">作为一个现实世界的例子，考虑将</st> `<st c="27935">df</st>` <st c="27937">(文档频率) 和</st> `<st c="27963">idf</st>` <st c="27966">(逆文档频率) 分数应用于莎士比亚的 37 部戏剧中的某些单词，你会发现单词</st> `<st c="28074">Romeo</st>` <st c="28079">是得分最高的结果。</st> <st c="28111">这是因为它出现的频率非常高，但只在一</st> *<st c="28171">个文档</st>*<st c="28179">中，即</st> `<st c="28185">罗密欧与朱丽叶</st>` <st c="28201">文档。</st> <st c="28212">在这种情况下，</st> `<st c="28226">Romeo</st>` <st c="28231">的</st> `<st c="28248">df</st>`<st c="28249">分数将是</st> `<st c="28254">1</st>` <st c="28249">，因为它出现在 1 个文档中。</st> `<st c="28288">Romeo</st>` <st c="28293">的</st> `<st c="28306">idf</st>`<st c="28310">分数将是</st> `<st c="28315">1.57</st>` <st c="28310">，高于其他任何单词，因为它在那一个文档中的频率很高。</st> <st c="28399">同时，莎士比亚偶尔使用了单词</st> `<st c="28436">sweet</st>` <st c="28441">，但在每一部戏剧中都有出现，给它一个低分。</st> <st c="28504">这使得</st> `<st c="28515">sweet</st>` <st c="28520">的</st> `<st c="28523">df</st>` <st c="28525">分数为</st> `<st c="28535">37</st>`<st c="28537">，而</st> `<st c="28546">idf</st>` <st c="28549">分数为</st> `<st c="28559">0</st>`<st c="28560">。凯伦·琼斯在她的论文中提到的是，当你看到像</st> `<st c="28639">Romeo</st>` <st c="28644">这样的单词只出现在总数很少的戏剧中时，你可以将这些单词出现的戏剧视为非常重要，并且可以预测该戏剧的内容。</st> <st c="28831">相比之下，</st> `<st c="28844">sweet</st>` <st c="28849">产生了相反的效果，因为它在单词的重要性和单词所在的文档方面都没有提供信息。</st>

<st c="28978">但话已说够。</st> <st c="29003">让我们看看这个算法在代码中的样子！</st> <st c="29047">scikit-learn 库有一个函数可以将 TF-IDF 方法应用于文本，以将文本向量化。</st> <st c="29163">以下代码是我们定义的</st> `<st c="29205">splits</st>` <st c="29211">变量，这是我们用作训练模型的</st> <st c="29262">数据：</st>

```py
 from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf_documents = [split.page_content for split in splits]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(
    tfidf_documents)
vocab = tfidf_vectorizer.get_feature_names_out()
tf_values = tfidf_matrix.toarray()
idf_values = tfidf_vectorizer.idf_
word_stats = list(zip(vocab, tf_values.sum(axis=0),
    idf_values))
word_stats.sort(key=lambda x: x[2], reverse=True)
print("Word\t\tTF\t\tIDF")
print("----\t\t--\t\t---")
for word, tf, idf in word_stats[:10]:
         print(f"{word:<12}\t{tf:.2f}\t\t{idf:.2f}")
```

<st c="29927">与 OpenAI 嵌入模型不同，此模型要求您在您的</st> *<st c="29990">训练</st> *<st c="29995">语料库</st> *<st c="30004">数据</st> *<st c="30010">上训练，这是一个术语，指的是您可用于训练的所有文本数据。</st> <st c="30095">此代码主要用于演示与我们的当前 RAG 管道检索器相比如何使用 TD-IDF 模型，因此我们不会逐行审查它。</st> <st c="30244">但我们鼓励您亲自尝试代码并尝试不同的设置。</st>

<st c="30321">需要注意的是，此算法生成的向量被称为</st> <st c="30385">稀疏向量</st> **<st c="30393">**<st c="30407">，而我们之前在之前的代码实验室中使用的向量被称为</st> **<st c="30491">密集向量</st> **<st c="30504">。这是一个重要的区别，我们将在</st> *<st c="30572">第八章</st> *<st c="30581">中详细讨论。</st>

<st c="30582">此模型使用语料库数据来设置环境，然后可以计算您向其引入的新内容的嵌入。</st> <st c="30684">输出应类似于以下表格：</st>

```py
 Word                     TF    IDF
000                      0.16  2.95
1024                     0.04  2.95
123                      0.02  2.95
13                       0.04  2.95
15                       0.01  2.95
16                       0.07  2.95
192                      0.06  2.95
1m                       0.08  2.95
200                      0.08  2.95
2024                     0.01  2.95
```

<st c="30920">在这种情况下，我们至少看到有 10 个文档在</st> `<st c="30972">idf</st>` <st c="30975">最高值上并列（我们只显示了 10 个，所以可能还有更多），而且所有这些都是基于数字的文本。</st> <st c="31083">这似乎并不特别有用，但这主要是因为我们的语料库数据如此之小。</st> <st c="31182">在来自同一作者或领域的更多数据上训练可以帮助您构建一个对底层内容有更好上下文理解的模型。</st>

<st c="31331">现在，回到我们一直在使用的原始问题，</st> `<st c="31398">RAG 的优势是什么？</st>`<st c="31429">，我们想使用 TF-IDF 嵌入来确定哪些是最相关的文档：</st>

```py
 tfidf_user_query = ["What are the advantages of RAG?"]
new_tfidf_matrix = tfidf_vectorizer.transform(
    tfidf_user_query)
tfidf_similarity_scores = cosine_similarity(
    new_tfidf_matrix, tfidf_matrix)
tfidf_top_doc_index = tfidf_similarity_scores.argmax()
print("TF-IDF Top Document:\n",
    tfidf_documents[tfidf_top_doc_index])
```

<st c="31840">这复制了我们在检索器中看到的行为，其中它使用相似性算法通过距离找到最近的嵌入。</st> <st c="31977">在这种情况下，我们使用余弦相似度，我们将在*<st c="32045">第八章</st>*中讨论，但请记住，我们可以使用许多距离算法来计算这个距离。</st> <st c="32022">从这个代码输出的结果是</st> <st c="32191">如下：</st>

```py
 TF-IDF Top Document:
Can you imagine what you could do with all of the benefits mentioned above, but combined with all of the data within your company, about everything your company has ever done, about your customers and all of their interactions, or about all of your products and services combined with a knowledge of what a specific customer's needs are? You do not have to imagine it, that is what RAG does…[TRUNCATED FOR BREVITY]
```

<st c="32638">如果你运行我们原始的代码，该代码使用原始的向量存储和检索器，你会看到</st> <st c="32734">以下输出：</st>

```py
 Retrieved Document:
Can you imagine what you could do with all of the benefits mentioned above, but combined with all of the data within your company, about everything your company has ever done, about your customers and all of their interactions, or about all of your products and services combined with a knowledge of what a specific customer's needs are? You do not have to imagine it, that is what RAG does…[TRUNCATED FOR BREVITY]
```

<st c="33181">它们匹配！</st> <st c="33194">一个 1972 年的小算法，在几秒钟内训练我们自己的数据，和 OpenAI 花费数十亿美元开发的庞大算法一样好！</st> <st c="33380">好吧，让我们放慢速度，这绝对不是情况！</st> <st c="33436">现实是，在现实世界的场景中，你将处理比我们更大的数据集，以及更复杂的用户查询，这将受益于使用更复杂的现代</st> <st c="33644">嵌入技术。</st>

<st c="33665">TF-IDF 在过去的几年中非常有用。</st> <st c="33677">但是，当我们谈论有史以来最先进的生成式 AI 模型时，有必要学习 1972 年的算法吗？</st> <st c="33710">答案是 BM25。</st> <st c="33846">这只是个预告，但你将在下一章中了解更多关于这个非常流行的**<st c="33937">关键词搜索</st>** <st c="33951">算法，它是目前使用最广泛的算法之一。</st> <st c="33966">而且你知道吗？</st> <st c="34049">它是基于 TF-IDF 的！</st> <st c="34072">然而，TF-IDF 的问题在于捕捉上下文和语义，以及我们接下来将要讨论的一些模型。</st> <st c="34202">让我们讨论下一个重大步骤：Word2Vec 和相关算法。</st>

## <st c="34272">Word2Vec、Sentence2Vec 和 Doc2Vec</st>

**<st c="34308">Word2Vec</st>** 和 <st c="34317">类似模型引入了无监督学习的早期应用，这代表了自然语言处理领域的一个重要进步。</st> <st c="34322">存在多个</st> *<st c="34472">vec</st> <st c="34475">模型（单词、文档和句子），它们的训练分别集中在单词、文档或句子上。</st> <st c="34592">这些模型在训练的文本级别上有所不同。</st>

<st c="34653">Word2Vec 专注于学习单个单词的向量表示，捕捉其语义意义和关系。</st> **<st c="34780">Doc2Vec</st>**<st c="34787">，另一方面，学习整个文档的向量表示，使其能够捕捉文档的整体上下文和主题。</st> **<st c="34928">Sentence2Vec</st>** <st c="34940">与 Doc2Vec 类似，但它在句子级别操作，学习单个句子的向量表示。</st> <st c="35057">虽然 Word2Vec 对于单词相似性和类比等任务很有用，但 Doc2Vec 和 Sentence2Vec 更适合文档级别的任务，如文档相似性、分类和检索。</st>

<st c="35256">因为我们正在处理更大的文档，而不仅仅是单词或句子，我们将选择 Doc2Vec 模型而不是 Word2Vec 或 Sentence2Vec，并训练此模型以查看它作为我们的检索器的工作方式。</st> <st c="35466">像 TD-IDF 模型一样，此模型可以用我们的数据进行训练，然后我们向它传递用户查询以查看我们是否可以得到最相似数据块的结果。</st>

<st c="35642">在 TD-IDF 代码单元之后添加此代码：</st> <st c="35688">代码单元：</st>

```py
 from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
doc2vec_documents = [
    split.page_content for split in splits]
doc2vec_tokenized_documents = [
    doc.lower().split() for doc in doc2vec_documents]
doc2vec_tagged_documents = [TaggedDocument(words=doc,
    tags=[str(i)]) for i, doc in enumerate(
    doc2vec_tokenized_documents)]
doc2vec_model = Doc2Vec(doc2vec_tagged_documents,
    vector_size=100, window=5, min_count=1, workers=4)
doc2vec_model.save("doc2vec_model.bin")
```

<st c="36220">与 TD-IDF 模型类似，这段代码主要是为了演示如何使用 Doc2Vec 模型，与我们的当前 RAG 管道检索器相比，所以我们将不会逐行审查，但我们鼓励您自己尝试代码并尝试不同的设置。</st> <st c="36477">此代码专注于训练 Doc2Vec 模型并将其本地保存。</st>

<st c="36547">有趣的事实</st>

<st c="36556">训练语言模型是当今的热门话题，并且可以成为一份收入颇丰的职业。</st> <st c="36643">你曾经训练过语言模型吗？</st> <st c="36683">如果你的答案是*<st c="36702">没有</st>*<st c="36704">，你就错了。</st> <st c="36726">你不仅刚刚训练了一个语言模型，你现在已经训练了两个了！</st> <st c="36801">TF-IDF 和 Doc2Vec 都是你刚刚训练的语言模型。</st> <st c="36868">这些是相对基本的模型训练版本，但你必须从某个地方开始，而你刚刚就做到了！</st>

<st c="36973">在接下来的代码中，我们将使用该模型在我们的数据上：</st> <st c="36996">进行操作：</st>

```py
 loaded_doc2vec_model = Doc2Vec.load("doc2vec_model.bin")
doc2vec_document_vectors = [loaded_doc2vec_model.dv[
    str(i)] for i in range(len(doc2vec_documents))]
doc2vec_user_query = ["What are the advantages of RAG?"]
doc2vec_tokenized_user_query = [content.lower().split() for content in doc2vec_user_query]
doc2vec_user_query_vector = loaded_doc2vec_model.infer_vector(
    doc2vec_tokenized_user_query[0])
doc2vec_similarity_scores = cosine_similarity([
    doc2vec_user_query_vector], doc2vec_document_vectors)
doc2vec_top_doc_index = doc2vec_similarity_scores.argmax()
print("\nDoc2Vec Top Document:\n",
    doc2vec_documents[doc2vec_top_doc_index])
```

<st c="37668">我们将创建和保存模型的代码与模型的使用分离，这样您就可以看到该模型如何被保存和稍后引用。</st> <st c="37823">以下是此代码的输出：</st>

```py
 Doc2Vec Top Document:
Once you have introduced the new knowledge, it will always have it! It is also how the model was originally created, by training with data, right? That sounds right in theory, but in practice, fine-tuning has been more reliable in teaching a model specialized tasks (like teaching a model how to converse in a certain way), and less reliable for factual recall…[TRUNCATED FOR BREVITY]
```

<st c="38264">将此与之前展示的我们原始检索器的结果进行比较，此模型不会返回相同的结果。</st> <st c="38385">然而，这个模型在这个</st> <st c="38451">行</st>中仅设置了 100 维向量：</st>

```py
 doc2vec_model = Doc2Vec(doc2vec_tagged_documents,
    vector_size=100, window=5, min_count=1, workers=4)
```

<st c="38562">当您将此行中的`vector_size`改为 1,536，与 OpenAI 模型的相同向量大小时，会发生什么？</st>

<st c="38672">将</st> `<st c="38684">doc2vec_model</st>` <st c="38697">变量定义</st> <st c="38718">改为：</st>

```py
 doc2vec_model = Doc2Vec(doc2vec_tagged_documents,
    vector_size=1536, window=5, min_count=1, workers=4)
```

<st c="38828">结果</st> <st c="38841">将变为：</st>

```py
 Doc2Vec Top Document:
Can you imagine what you could do with all of the benefits mentioned above, but combined with all of the data within your company, about everything your company has ever done, about your customers and all of their interactions, or about all of your products and services combined with a knowledge of what a specific customer's needs are? You do not have to imagine it, that is what RAG does…[TRUNCATED FOR BREVITY]
```

<st c="39298">这导致了与我们的原始结果相同的结果，使用了 OpenAI 的嵌入。</st> <st c="39382">然而，结果并不一致。</st> <st c="39423">如果您在更多数据上训练这个模型，它很可能会改善</st> <st c="39486">结果。</st>

<st c="39498">从理论上讲，这种模型相较于 TF-IDF 的优势在于它是一种基于神经网络的模型，它考虑了周围的词语，而 TF-IDF 仅仅是一个统计度量，用于评估一个词对文档的相关性（关键词搜索）。</st> <st c="39765">但正如我们之前提到的 TD-IDF 模型，还有比</st> *<st c="39850">vec</st>* <st c="39853">模型更强大的模型，这些模型能够捕捉到更多被输入文本的上下文和语义。</st> <st c="39932">让我们跳到另一代的</st> <st c="39968">模型，即 transformers。</st>

## <st c="39989">从 transformers 中得到的双向编码器表示</st>

<st c="40045">在这个时候，使用**<st c="40066">从 transformers 中得到的双向编码器表示</st>** <st c="40121">(**<st c="40123">BERT</st>**<st c="40127">)，我们已经完全转向使用神经网络来更好地理解语料库的潜在语义，这是 NLP 算法的又一次重大进步。</st> <st c="40281">BERT 也是最早应用特定类型的神经网络之一，即**<st c="40358">transformer</st>**<st c="40369">，这是导致我们今天熟悉的 LLMs 发展的关键步骤之一。</st> <st c="40481">OpenAI 流行的 ChatGPT 模型也是 transformers，但它们是在一个更大的语料库上用不同的技术训练的，与 BERT 不同。</st>

<st c="40616">话虽如此，BERT 仍然是一个非常强大的模型。</st> <st c="40664">您可以将 BERT 作为一个独立的模型导入，避免依赖于 OpenAI 的嵌入服务等 API。</st> <st c="40788">能够在您的代码中使用本地模型在某些网络受限的环境中可能是一个很大的优势，而不是依赖于像 OpenAI 这样的 API 服务。</st> <st c="40943">OpenAI。</st>

<st c="40953">变压器模型的一个定义特征是使用自注意力机制来捕捉文本中词语之间的依赖关系。</st> <st c="41106">BERT 也具有多层变压器，这使得它能够学习更复杂的表示。</st> <st c="41209">与我们的 Doc2Vec 模型相比，BERT 已经在大规模数据上进行了预训练，例如维基百科和 BookCorpus，目的是预测</st> <st c="41364">下一个句子。</st>

<st c="41378">与之前的两个模型类似，我们为您提供了代码来比较使用 BERT 检索的结果</st> <st c="41467">：</st>

```py
 from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
bert_documents = [split.page_content for split in splits]
bert_tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_vector_size = bert_model.config.hidden_size
print(f"Vector size of BERT (base-uncased) embeddings:
    {bert_vector_size}\n")
bert_tokenized_documents = [bert_tokenizer(doc,
    return_tensors='pt', max_length=512, truncation=True)
    for doc in bert_documents]
bert_document_embeddings = []
with torch.no_grad():
    for doc in bert_tokenized_documents:
        bert_outputs = bert_model(**doc)
        bert_doc_embedding =
            bert_outputs.last_hidden_state[0, 0, :].numpy()
        bert_document_embeddings.append(bert_doc_embedding)
bert_user_query = ["What are the advantages of RAG?"]
bert_tokenized_user_query = bert_tokenizer(
    bert_user_query[0], return_tensors='pt',
    max_length=512, truncation=True)
bert_user_query_embedding = []
with torch.no_grad():
    bert_outputs = bert_model(
        **bert_tokenized_user_query)
         bert_user_query_embedding =
             bert_outputs.last_hidden_state[
                 0, 0, :].numpy()
bert_similarity_scores = cosine_similarity([
    bert_user_query_embedding], bert_document_embeddings)
bert_top_doc_index = bert_similarity_scores.argmax()
print("BERT Top Document:\n", bert_documents[
    bert_top_doc_index])
```

<st c="42859">与之前几个模型的使用相比，这段代码有一个非常重要的区别。</st> <st c="42869">在这里，我们不是在自己的数据上调整模型。</st> <st c="42964">这个 BERT 模型已经在大数据集上进行了训练。</st> <st c="43015">我们可以使用我们的数据进一步微调模型，如果您想使用这个模型，这是推荐的。</st> <st c="43076">结果将反映这种训练不足，但我们不会让这阻止我们向您展示它是如何工作的！</st> <st c="43189">结果将反映这种训练不足，但我们不会让这阻止我们向您展示它是如何工作的！</st>

<st c="43300">对于这段代码，我们正在打印出向量大小以供与其他进行比较。</st> <st c="43382">与其他模型一样，我们可以看到检索到的最顶部结果。</st> <st c="43442">以下是</st> <st c="43450">输出：</st>

```py
 Vector size of BERT (base-uncased) embeddings: 768
BERT Top Document:
Or if you are developing in a legal field, you may want it to sound more like a lawyer. Vector Store or Vector Database?
```

<st c="43652">向量大小是可尊敬的</st> `<st c="43686">768</st>`<st c="43689">。我不需要指标就能告诉你，它找到的最顶部文档并不是回答问题的最佳片段</st> `<st c="43805">RAG 的优势是什么</st>` `<st c="43829">。</st>`

<st c="43837">这个</st> <st c="43842">模型功能强大，有可能比之前的模型表现更好，但我们需要做一些额外的工作（微调）才能让它在我们与之前讨论的嵌入模型类型进行比较时做得更好。</st> <st c="44107">这并不适用于所有数据，但在这种专业领域，微调应该被视为嵌入模型的一个选项。</st> <st c="44269">这尤其适用于您使用的是较小的本地模型，而不是像 OpenAI 的</st> <st c="44381">嵌入 API 这样的大型托管 API。</st>

<st c="44396">运行这三个不同的模型说明了嵌入模型在过去 50 年中的变化。</st> <st c="44517">希望这次练习已经向你展示了选择嵌入模型的重要性。</st> <st c="44623">我们将通过回到我们最初使用的原始嵌入模型，即 OpenAI 的 API 服务中的 OpenAI 嵌入模型，来结束我们对嵌入模型的讨论。</st> <st c="44808">我们将讨论 OpenAI 模型，以及它在其他</st> <st c="44872">云服务中的同类模型。</st>

## <st c="44887">OpenAI 和其他类似的大型嵌入服务</st>

<st c="44943">让我们更深入地谈谈我们刚刚使用的 BERT 模型，相对于 OpenAI 的嵌入模型。</st> <st c="45042">这是</st> `<st c="45055">'bert-base-uncased'</st>` <st c="45074">版本，这是一个相当健壮的 1100 万个参数的 Transformer 模型，特别是与之前我们使用的模型相比。</st> <st c="45195">自从 TD-IDF 模型以来，我们已经走了很长的路。</st> <st c="45243">根据你工作的环境，这可能会测试你的计算限制。</st> <st c="45338">这是我的电脑能够运行的 BERT 选项中最大的模型。</st> <st c="45408">但如果你有一个更强大的环境，你可以在这两行</st> <st c="45497">中将模型</st> <st c="45500">'bert-large-uncased'</st>`<st c="45520">:</st>

```py
 tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```

<st c="45641">您可以在</st> <st c="45684">这里</st> <st c="45690">查看 BERT 选项的完整列表：</st> [<st c="45690">https://huggingface.co/google-bert/bert-base-uncased</st>](https://huggingface.co/google-bert/bert-base-uncased )

<st c="45742">The</st> `<st c="45747">'bert-large-uncased'</st>` <st c="45767">模型有 3400 万个参数，是</st> `<st c="45829">'bert-base-uncased'</st>`<st c="45848">的三倍多。如果你的环境无法处理这么大的模型，它将导致你的内核崩溃，你将不得不重新加载所有导入和相关的笔记本单元格。</st> <st c="46006">这仅仅告诉你这些模型可以有多大。</st> <st c="46058">但为了明确起见，这两个</st> <st c="46090">BERT 模型分别有 1100 万个和 3400 万个参数，这是以百万为单位的，</st> <st c="46154">而不是十亿。</st>

<st c="46167">我们一直在使用的 OpenAI 嵌入模型基于</st> **<st c="46235">GPT-3</st>** <st c="46240">架构，该架构拥有 1750 亿个参数。</st> <st c="46289">这是一个</st> *<st c="46299">十亿</st>* <st c="46306">，带有</st> *<st c="46314">B</st>*<st c="46315">。我们将在本章后面讨论他们的较新嵌入模型，这些模型基于</st> **<st c="46417">GPT-4</st>** <st c="46422">架构，并拥有一万亿个参数（带有</st> *<st c="46477">T</st>*<st c="46478">！）。</st> <st c="46482">不用说，这些模型规模巨大，远超我们之前讨论过的任何其他模型。</st> <st c="46577">BERT 和 OpenAI 都是 Transformer，但 BERT 是在 33 亿个单词上训练的，而 GPT-3 的完整语料库估计约为 1700 亿个单词（45 TB 的文本）。</st>

<st c="46753">OpenAI 目前有三种不同的嵌入模型可供选择。</st> <st c="46819">我们一直在使用基于 GPT-3 的较旧模型来节省 API 成本，即</st> `<st c="46888">'text-embedding-ada-002'</st>`<st c="46912">，但它是一个非常强大的嵌入模型。</st> <st c="46956">另外两种基于 GPT-4 的新模型是</st> `<st c="47011">'text-embedding-3-small'</st>` <st c="47035">和</st> `<st c="47040">'text-embedding-3-large'</st>`<st c="47064">。这两个模型都支持我们之前讨论过的马赛克嵌入，这允许你为</st> <st c="47204">你的检索使用自适应检索方法。</st>

<st c="47219">虽然 OpenAI 不是唯一提供文本嵌入 API 的云服务提供商。</st> `<st c="47422">'text-embedding-preview-0409'</st>`<st c="47451">。这个</st> `<st c="47457">'text-embedding-preview-0409'</st>` <st c="47486">模型是除 OpenAI 的较新模型之外，我所知唯一支持马赛克嵌入的大型云托管嵌入模型。</st>

**<st c="47650">亚马逊网络服务</st>** <st c="47670">(</st>**<st c="47672">AWS</st>**<st c="47675">) 提供基于他们</st> **<st c="47714">Titan 模型</st>**<st c="47725">的嵌入模型，以及</st> **<st c="47738">Cohere 的嵌入模型</st>**<st c="47763">。</st> **<st c="47765">Titan Text Embeddings V2</st>** <st c="47789">预计很快将推出，并预计也将支持</st> <st c="47849">马赛克嵌入。</st>

<st c="47870">这就是我们穿越 50 年嵌入生成技术风驰电掣的冒险之旅的结束！</st> <st c="47963">所强调的模型被选中以代表过去 50 年嵌入能力的进步，但这些只是实际生成嵌入方式的微小部分。</st> <st c="48042">现在，你对嵌入能力的了解已经扩展，让我们转向你在实际决策中考虑的因素，选择哪个模型。</st> <st c="48133">来使用。</st>

# <st c="48345">选择向量化选项的因素</st>

<st c="48389">选择</st> <st c="48404">正确的向量化选项是构建 RAG 系统时的一个关键决策。</st> <st c="48481">关键考虑因素包括特定应用中嵌入的质量、相关成本、网络可用性、嵌入生成速度以及嵌入模型之间的兼容性。</st> <st c="48688">在为选择嵌入模型时，还有许多其他选项可供探索，以满足您的特定需求。</st> <st c="48841">让我们回顾</st> <st c="48854">这些考虑因素。</st>

## <st c="48875">嵌入质量</st>

<st c="48900">在考虑嵌入质量时，你不能仅仅依赖你所看到的每个模型的通用指标。</st> <st c="49002">例如，OpenAI 在“text-embedding-ada-002”模型上进行了测试，而“text-embedding-3-large”模型的得分是 64.6%。</st> <st c="49134">'text-embedding-ada-002'</st> <st c="49158">模型，而</st> <st c="49178">'text-embedding-3-large'</st> <st c="49202">模型的得分是 64.6%。</st> <st c="49223">这些指标可能是有用的，尤其是在尝试专注于特定质量的模型时，但这并不意味着该模型将比你的特定模型好 3.6%。</st> <st c="49395">甚至不意味着它一定会更好。</st> <st c="49455">不要完全依赖通用测试。</st> <st c="49496">最终重要的是你的嵌入在你的特定应用中表现如何。</st> <st c="49592">这包括你用自己数据训练的嵌入模型。</st> <st c="49658">如果你从事一个涉及特定领域（如科学、法律或技术）的应用，你很可能可以找到或训练一个与你的特定领域数据表现更好的模型。</st> <st c="49862">当你开始你的项目时，尝试在 RAG 系统中使用多个嵌入模型，然后使用我们在</st> *<st c="49995">第九章</st>* <st c="50004">中分享的评估技术来比较每个模型的使用结果，以确定哪个最适合</st> <st c="50077">你的应用。</st>

## <st c="50094">成本</st>

<st c="50099">这些嵌入服务的成本从免费到相对昂贵不等。</st> <st c="50179">OpenAI 最昂贵的嵌入模型每百万个令牌的成本为 0.13 美元。</st> <st c="50251">这意味着对于一个包含 800 个令牌的页面，它将花费您 0.000104 美元，或者略多于 1 美分的 1%。</st> <st c="50363">这</st> <st c="50368">可能听起来不多，但对于大多数使用嵌入的应用程序，尤其是在企业中，这些成本会迅速增加，即使是小型项目，成本也可能达到 1000 美元或 10000 美元。</st> <st c="50576">但其他嵌入 API 的成本较低，也许同样能满足您的需求。</st> <st c="50648">当然，如果您像我在这章前面描述的那样构建自己的模型，您将只需承担该模型的硬件或托管成本。</st> <st c="50811">这可能会随着时间的推移而大幅降低成本，并且可能满足您的需求</st> <st c="50869">。</st>

## <st c="50877">网络可用性</st>

<st c="50898">在考虑网络可用性方面，您需要考虑各种场景。</st> <st c="50992">几乎所有应用程序都存在网络不可用的情况。</st> <st c="51082">网络可用性会影响用户访问您的应用程序接口，但它也可能影响您从应用程序到其他服务的网络调用。</st> <st c="51248">在后一种情况下，这可能是一种用户可以访问您的应用程序接口，但应用程序无法达到 OpenAI 的嵌入服务以为您用户查询生成嵌入的情况。</st> <st c="51458">在这种情况下，您会怎么做呢？</st> <st c="51489">如果您使用的是您环境内的模型，这可以避免这个问题。</st> <st c="51573">这是关于可用性和它对</st> <st c="51638">您的用户产生的影响的考虑。</st>

<st c="51649">请记住，您不能仅仅切换您用户查询的嵌入模型，以防您认为您可以使用</st> *<st c="51779">回退</st>* <st c="51787">机制，并在网络不可用时使用一个本地嵌入模型作为次要选项。</st> <st c="51896">如果您使用的是仅提供 API 的专有嵌入模型来矢量化您的嵌入，您就承诺了该嵌入模型，并且您的 RAG 系统将依赖于该 API 的可用性。</st> <st c="52088">OpenAI 不提供其嵌入模型以供本地使用。</st> <st c="52149">请参阅即将到来的</st> *<st c="52166">嵌入</st> * *<st c="52176">兼容性</st> * <st c="52189">子节！</st>

## <st c="52201">速度</st>

<st c="52207">生成嵌入的速度是一个重要的考虑因素，因为它可能会影响你应用程序的响应性和用户体验。</st> <st c="52351">当使用如 OpenAI 这样的托管 API 服务时，你正在通过网络调用生成嵌入。</st> <st c="52452">虽然这些网络调用相对较快，但与在本地环境中生成嵌入相比，仍然存在一些延迟。</st> <st c="52607">然而，重要的是要注意，本地嵌入生成并不总是更快，因为速度也取决于所使用的特定</st> <st c="52735">模型。</st> <st c="52753">某些模型可能具有较慢的推理时间，从而抵消了本地处理的好处。</st> <st c="52841">在确定你的嵌入选项的速度时需要考虑的关键方面包括网络延迟、模型推理时间、硬件资源，以及在涉及多个嵌入的情况下，批量生成嵌入并优化</st> <st c="53101">的能力。</st>

## <st c="53111">嵌入兼容性</st>

<st c="53135">请注意；这是关于嵌入的一个非常重要考虑和事实！</st> <st c="53223">在任何比较嵌入的情况下，例如当你检测用户查询嵌入与存储在向量存储中的嵌入之间的相似性时，*<st c="53394">它们必须由相同的嵌入模型创建</st>*<st c="53442">。这些模型生成仅针对该模型的独特向量签名。</st> <st c="53520">即使是同一服务提供商的模型也是如此。</st> <st c="53578">例如，在 OpenAI，所有三个嵌入模型之间都不兼容。</st> <st c="53667">如果你使用 OpenAI 的任何嵌入模型将你的向量存储中的嵌入向量化，你必须调用 OpenAI API 并在向量化用户查询以进行向量搜索时使用相同的模型。</st> <st c="53872">进行向量搜索。</st>

当你的应用程序规模扩大时，更改或更新嵌入模型会产生重大的成本影响，因为这意味着你需要生成所有新的嵌入以使用新的嵌入模型。<st c="54083">这甚至可能迫使你使用本地模型而不是托管 API 服务，因为使用你控制的模型生成新的嵌入通常成本</st> <st c="54242">要低得多。</st>

<st c="54252">虽然通用基准可以提供指导，但评估您特定领域和应用中的多个嵌入模型以确定最佳匹配至关重要。</st> <st c="54424">成本可能会有很大差异，这取决于服务提供商和所需的嵌入量。</st> <st c="54527">网络可用性和速度是重要因素，尤其是在使用托管 API 服务时，因为它们可能会影响您应用程序的响应性和用户体验。</st> <st c="54703">嵌入模型之间的兼容性也非常关键，因为由不同模型生成的嵌入无法直接比较。</st>

<st c="54831">随着您的应用程序增长，更改或更新向量嵌入模型可能会产生重大的成本影响。</st> <st c="54944">本地嵌入生成可以提供更多控制，并可能降低成本，但速度取决于特定模型和可用的硬件资源。</st> <st c="55101">彻底的测试和基准测试对于找到适用于您应用程序的最佳质量、成本、速度和其他相关因素的平衡至关重要。</st> <st c="55251">现在我们已经探讨了选择向量化选项的考虑因素，让我们深入了解它们是如何通过</st> <st c="55388">向量存储进行存储的。</st>

# <st c="55402">开始使用向量存储</st>

<st c="55437">向量存储，结合</st> <st c="55462">其他数据存储（数据库、数据仓库、数据湖以及任何其他数据源）是您 RAG 系统引擎的燃料。</st> <st c="55595">不言而喻，但没有一个地方来存储您专注于 RAG 的数据，这通常涉及向量的创建、管理、过滤和搜索，您将无法构建一个有能力的 RAG 系统。</st> <st c="55810">您所使用的内容及其实现方式将对您整个 RAG 系统的性能产生重大影响，使其成为一个关键的决定和努力。</st> <st c="55967">为了开始本节，让我们首先回顾一下</st> <st c="56037">数据库的原始概念。</st>

## <st c="56048">数据源（除了向量之外）

<st c="56081">在我们的</st> <st c="56088">基本的 RAG 示例中，我们目前保持简单（目前是这样），并且尚未将其连接到额外的数据库资源。</st> <st c="56212">您可以考虑内容提取的网页作为数据库，尽管在这个上下文中最准确的描述可能是将其称为非结构化数据源。</st> <st c="56400">无论如何，您的应用程序很可能发展到需要类似数据库的支持。</st> <st c="56506">这可能以传统 SQL 数据库的形式出现，或者可能是巨大的数据湖（所有类型原始数据的大型存储库），其中数据被预处理成更可用的格式，代表支持您的</st> <st c="56783">RAG 系统数据源。</st>

<st c="56794">您数据存储的架构可能基于一个</st> **<st c="56851">关系型数据库管理系统</st>** <st c="56888">(</st>**<st c="56890">RDBMS</st>**<st c="56895">**<st c="56888">)** <st c="56901">，多种不同类型的 NoSQL，NewSQL（旨在结合前两种方法的优点），或者各种版本的数据仓库和数据湖。</st> <st c="57063">从本书的角度来看，我们将这些系统所代表的数据源视为一个抽象概念，即</st> **<st c="57186">数据源</st>**<st c="57197">。但在此需要考虑的重要问题是，您选择使用哪种向量存储的决定可能会受到您现有数据源架构的高度影响。</st> <st c="57221">您的员工当前的技术技能也可能会在这些决策中扮演关键角色。</st>

<st c="57469">例如，您可能正在</st> <st c="57496">使用</st> **<st c="57502">PostgreSQL</st>** <st c="57512">作为您的 RDBMS，并且拥有一支在充分利用和优化 PostgreSQL 方面具有丰富经验的专家工程师团队。</st> <st c="57637">在这种情况下，您</st> <st c="57655">将希望考虑</st> **<st c="57681">pgvector</st>** <st c="57689">扩展，它将 PostgreSQL 表转换为向量存储，将您的团队熟悉的许多 PostgreSQL 功能扩展到向量世界。</st> <st c="57862">诸如索引和针对 PostgreSQL 特定编写的 SQL 等概念已经熟悉，这将帮助您的团队快速了解如何扩展到 pgvector。</st> <st c="58054">如果您正在从头开始构建整个数据基础设施，这在企业中很少见，那么您可能选择一条针对速度、成本、准确性或所有这些优化的不同路线！</st> <st c="58241">但对于大多数公司来说，您在选择向量存储时需要考虑与现有基础设施的兼容性</st> <st c="58364">决策标准。</st>

<st c="58382">有趣的事实——那么像 SharePoint 这样的应用程序呢？</st>

<st c="58437">SharePoint 通常被认为是</st> <st c="58451">一个内容管理系统</st> <st c="58475">（</st>**<st c="58475">内容管理系统</st>**<st c="58500">）**<st c="58502">（CMS</st>**<st c="58505">）**，可能不完全符合我们之前提到的其他数据源的定义。</st> <st c="58537">但 SharePoint 和类似的应用程序包含大量的非结构化数据存储库，包括 PDF、Word、Excel 和 PowerPoint 文档，这些文档代表了一个公司知识库的很大一部分，尤其是在大型企业环境中。</st> <st c="58855">结合生成式 AI 显示出对非结构化数据有前所未有的倾向，这为 RAG 系统提供了一个令人难以置信的数据来源。</st> <st c="59067">这类应用程序还具有复杂的 API，可以在提取文档时进行数据提取，例如在向量化之前从 Word 文档中提取文本并将其放入数据库中。</st> <st c="59284">在许多大型公司中，由于这些应用程序中数据的高价值以及使用 API 提取数据的相对容易，这已经成为 RAG 系统数据来源之一。</st> <st c="59491">因此，是的，你绝对可以将 SharePoint 和类似的应用程序列入你的潜在</st> <st c="59588">数据源列表！</st>

<st c="59601">我们将在稍后更多地讨论 pgvector 和其他向量存储选项，但了解这些决策可能非常具体于每种情况，并且除了向量存储本身之外的其他考虑因素将在你最终决定与之合作的内容中扮演重要角色。</st> <st c="59897">工作。</st>

<st c="59907">无论你选择什么选项，或者从哪里开始，这都将是一个向你的 RAG 系统提供数据的关键组件。</st> <st c="60037">这引出了向量存储本身，我们接下来可以</st> <st c="60097">讨论。</st>

## <st c="60110">向量存储</st>

<st c="60124">向量存储，也</st> <st c="60144">被称为向量数据库或向量搜索引擎，是一种专门为高效存储、管理和检索数据向量表示而设计的存储系统。</st> <st c="60313">与传统数据库按行和列组织数据不同，向量存储针对高维向量空间中的操作进行了优化。</st> <st c="60460">它们在有效的 RAG 系统中发挥着关键作用，通过实现快速相似性搜索，这对于在向量查询的响应中识别最相关的信息至关重要。</st> <st c="60640">向量化查询。</st>

<st c="60657">向量存储的架构通常由三个</st> <st c="60721">主要组件组成：</st>

+   **<st c="60737">索引层</st>**<st c="60752">：这一层</st> <st c="60760">以加快搜索查询的方式组织向量。</st> <st c="60831">它采用基于树的分区（例如，KD 树）或哈希（例如，局部敏感哈希）等技术，以促进在</st> <st c="61029">向量空间中彼此靠近的向量的快速检索。</st>

+   **<st c="61042">存储层</st>**<st c="61056">：存储层</st> <st c="61076">高效管理磁盘或内存中的数据存储，确保最佳性能</st> <st c="61165">和可扩展性。</st>

+   **<st c="61181">处理层（可选）</st>**<st c="61209">：一些向量存储包括一个处理层来处理</st> <st c="61267">向量转换、相似度计算和其他实时分析操作。</st>

<st c="61361">虽然在技术上可以构建一个不使用向量存储的 RAG 系统，但这样做会导致性能和可扩展性不佳。</st> <st c="61512">向量存储专门设计来处理存储和提供高维向量的独特挑战，提供了显著提高内存使用、计算需求和</st> <st c="61727">搜索精度</st>的优化。

<st c="61744">正如我们之前提到的，需要注意的是，虽然术语</st> *<st c="61821">向量数据库</st>** <st c="61836">和</st> *<st c="61841">向量存储</st>** <st c="61853">经常互换使用，但并非所有向量存储都是数据库。</st> <st c="61935">还有其他工具和机制服务于与向量数据库相同或类似的目的。</st> <st c="62033">为了准确性和与 LangChain 文档的一致性，我们将使用术语</st> *<st c="62125">向量存储</st>** <st c="62137">来指代所有存储和提供向量的机制，包括向量数据库和其他</st> <st c="62232">非数据库解决方案。</st>

<st c="62255">接下来，让我们讨论向量存储选项，以便您更好地了解</st> <st c="62336">可用的内容。</st>

## <st c="62349">常见的向量存储选项</st>

<st c="62377">在选择向量存储时，考虑因素包括可扩展性要求、设置和维护的简便性、性能需求、预算限制以及您对底层基础设施的控制和灵活性水平。</st> <st c="62618">此外，评估集成选项和支持的编程语言，以确保与您现有的</st> <st c="62744">技术栈兼容。</st>

<st c="62761">目前有相当多的向量存储，一些来自知名数据库公司和社区，许多是新成立的初创公司，每天都有更多出现，而且很可能在你阅读这篇文档时，一些公司可能会倒闭。</st> <st c="63016">这是一个非常活跃的领域！</st> <st c="63043">保持警惕，并使用本章中的信息来了解对你特定的 RAG 应用最重要的方面，然后查看当前市场，以确定哪个选项最适合你。</st> <st c="63259">。</st>

<st c="63267">我们将重点关注已经与 LangChain 建立整合的向量存储，即便如此，我们也会将它们筛选到不会让你感到压倒性，同时也会给你足够的选择，以便你能感受到有哪些选项可供选择。</st> <st c="63517">请记住，这些向量存储一直在添加功能和改进。</st> <st c="63606">在选择之前，务必查看它们的最新版本！</st> <st c="63674">这可能会让你改变主意，做出更好的选择！</st>

<st c="63761">在以下小节中，我们将介绍一些与 LangChain 集成的常见向量存储选项，以及在选择过程中你应该考虑的每个选项。</st> <st c="63939">选择过程。</st>

### <st c="63957">Chroma</st>

**<st c="63964">Chroma</st>** <st c="63971">是一个</st> <st c="63978">开源向量数据库。</st> <st c="64007">它提供快速搜索性能，并通过其 Python SDK 轻松集成 LangChain。</st> <st c="64110">Chroma</st> <st c="64117">以其简洁性和易用性而突出，具有直观的 API 和搜索过程中对集合动态过滤的支持。</st> <st c="64255">它还提供内置的文档分块和索引支持，方便处理大型文本数据集。</st> <st c="64382">如果你优先考虑简洁性并希望有一个可以自行托管的开源解决方案，Chroma 是一个不错的选择。</st> <st c="64493">然而，它可能没有像一些其他选项（如分布式搜索、支持多种索引算法和内置的将向量相似性与元数据过滤相结合的混合搜索功能）那样多的高级功能。</st>

### <st c="64732">LanceDB</st>

**<st c="64740">LanceDB</st>** 是一个专为高效相似搜索和检索设计的向量数据库。<st c="64748">它以其混合搜索能力而突出，结合了向量相似搜索和基于关键词的传统搜索。</st> <st c="64754">LanceDB 支持各种距离度量索引算法，包括</st> **<st c="65027">层次可导航小世界</st>** <st c="65061">(**<st c="65063">HNSW</st>**<st c="65067">**)，用于高效的近似最近邻搜索。<st c="65121">它</st> <st c="65124">与 LangChain 集成，并提供快速搜索性能和对各种索引技术的支持。</st> <st c="65230">如果您想要一个性能良好且与 LangChain 集成的专用向量数据库，LanceDB 是一个不错的选择。</st> <st c="65349">然而，与一些其他选项相比，它可能没有这么大的社区或生态系统。</st>

### <st c="65442">Milvus</st>

**<st c="65449">Milvus</st>** 是一个提供可扩展相似搜索并支持各种索引算法的开源向量数据库。<st c="65456">它提供了一种云原生架构，并支持基于 Kubernetes 的部署，以实现可扩展性和高可用性。</st> <st c="65463">Milvus 提供了多向量索引等特性，允许您同时搜索多个向量字段，并提供一个插件系统以扩展其功能。</st> <st c="65504">它与 LangChain 集成良好，提供分布式部署和横向可扩展性。</st> <st c="65574">如果您需要一个可扩展且功能丰富的开源向量存储库，Milvus 是一个不错的选择。</st> <st c="65695">然而，与托管服务相比，它可能需要更多的设置和管理。</st>

### <st c="66142">pgvector</st>

**<st c="66151">pgvector</st>** 是一个为 PostgreSQL 添加向量相似搜索支持并作为向量存储与 LangChain 集成的扩展。<st c="66160">它利用了世界上功能最强大的开源关系型数据库 PostgreSQL 的力量和可靠性，并从 PostgreSQL 成熟的生态系统、广泛的文档和强大的社区支持中受益。</st> <st c="66288">pgvector 无缝地将向量相似搜索与传统的关系型数据库功能集成，实现了混合搜索能力。</st>

<st c="66644">最近的更新提高了 pgvector 的性能水平，使其与其他专门的向量数据库服务保持一致。</st> <st c="66779">鉴于 PostgreSQL 是世界上最受欢迎的数据库（一种经过实战考验的成熟技术，拥有庞大的社区），以及向量扩展 pgvector 为您提供了其他向量数据库的所有功能，这种组合为任何已经</st> <st c="67067">使用 PostgreSQL 的公司提供了一个绝佳的解决方案。</st>

### <st c="67084">Pinecone</st>

**<st c="67093">Pinecone</st>** <st c="67102">是一个</st> <st c="67108">完全托管的向量数据库服务</st> <st c="67145">，提供高性能、可扩展性和与 LangChain 的轻松集成。</st> <st c="67226">它提供完全托管和无服务器体验，抽象出基础设施管理的复杂性。</st> <st c="67345">Pinecone 提供实时索引等特性，允许您以低延迟更新和搜索向量，并支持混合搜索，结合向量相似度和元数据过滤。</st> <st c="67539">它还提供分布式搜索和多种索引算法的支持。</st> <st c="67638">如果您需要一个性能良好且设置简单的托管解决方案，Pinecone 是一个不错的选择。</st> <st c="67736">然而，与自托管选项相比，它可能更昂贵。</st>

### <st c="67802">Weaviate</st>

**<st c="67811">Weaviate</st>** <st c="67820">是一个</st> <st c="67827">开源向量搜索引擎，支持</st> <st c="67873">各种向量索引和相似度搜索算法。</st> <st c="67932">它采用基于模式的方案，允许您为您的向量定义语义数据模型。</st> <st c="68031">Weaviate 支持 CRUD 操作、数据验证和授权机制，并提供用于常见机器学习任务的模块，如文本分类和图像相似度搜索。</st> <st c="68223">它集成了 LangChain，并提供了诸如模式管理、实时索引和 GraphQL API 等特性。</st> <st c="68338">如果您需要一个具有高级功能和灵活性的开源向量搜索引擎，Weaviate 是一个不错的选择。</st> <st c="68449">然而，与托管服务相比，它可能需要更多的设置和配置。</st>

<st c="68531">在前几节中，我们讨论了与 LangChain 集成的各种向量存储选项，概述了它们的特性、优势和选择时的考虑因素。</st> <st c="68721">这</st> <st c="68726">强调了在选择向量存储时评估可扩展性、易用性、性能、预算以及与现有技术栈兼容性等因素的重要性。</st> <st c="68909">虽然这个列表很广泛，但与 LangChain 可集成和作为一般向量存储选项的总数相比，仍然非常短。</st> <st c="69060">。</st>

<st c="69071">所提到的向量存储涵盖了多种功能，包括快速相似性搜索、支持各种索引算法、分布式架构、结合向量相似性和元数据过滤的混合搜索，以及与其他服务和数据库的集成。</st> <st c="69350">鉴于向量存储领域的快速演变，新的选项频繁出现。</st> <st c="69444">请以此信息为基础，但当你准备构建下一个 RAG 系统时，我们强烈建议你访问 LangChain 关于可用向量存储的文档，并考虑当时最适合你需求的选项。</st> <st c="69667">。</st>

<st c="69677">在下一节中，我们将更深入地讨论选择向量存储时需要考虑的因素。</st> <st c="69785">。</st>

# <st c="69796">选择向量存储</st>

<st c="69820">选择</st> <st c="69830">RAG 系统的正确向量存储需要考虑多个因素，包括数据规模、所需的搜索性能（速度和准确性）以及向量操作的复杂性。</st> <st c="70041">对于处理大型数据集的应用，可扩展性至关重要，需要一种能够高效管理和检索从不断增长的语料库中向量的机制。</st> <st c="70204">性能考虑包括评估数据库的搜索速度及其返回高度相关结果的能力。</st>

<st c="70328">此外，与现有 RAG 模型的集成简便性以及支持各种向量操作的灵活性也是至关重要的。</st> <st c="70464">开发者应寻找提供强大 API、全面文档和强大社区或供应商支持的向量存储。</st> <st c="70598">如前所述，有许多流行的向量存储，每个都提供针对不同用例和性能需求定制的独特功能和优化。</st> <st c="70742">。</st>

<st c="70760">在选择向量存储时，确保其选择与 RAG 系统的整体架构和运营需求相一致。</st> <st c="70907">以下是一些</st> <st c="70921">关键考虑因素：</st>

+   **<st c="70940">与现有基础设施的兼容性</st>**<st c="70983">：在评估向量存储时，考虑它们与您现有的数据基础设施（如数据库、数据仓库和数据湖）的集成程度至关重要。</st> <st c="71158">评估向量存储与您当前技术栈和开发团队技能的兼容性。</st> <st c="71279">例如，如果您在特定的数据库系统（如 PostgreSQL）方面有很强的专业知识，向量存储扩展（如 pgvector）</st> <st c="71414">可能是一个合适的选择，因为它允许无缝集成并利用您团队的</st> <st c="71507">现有知识。</st>

+   **<st c="71526">可扩展性和性能</st>**<st c="71554">：向量存储处理您预期数据增长和 RAG 系统性能要求的能力如何？</st> <st c="71688">评估向量存储的索引和搜索能力，确保它能够提供所需的性能和准确性。</st> <st c="71824">如果您预计进行大规模部署，具有向量插件的分布式向量数据库，如 Milvus 或 Elasticsearch，可能更合适，因为它们旨在处理大量数据并提供高效的</st> <st c="72047">搜索吞吐量。</st>

+   **<st c="72065">易用性和维护性</st>**<st c="72093">：考虑到可用的文档、社区支持和供应商支持，与向量存储相关的学习曲线是什么？</st> <st c="72245">了解设置、配置和向量存储的持续维护所需的工作量。</st> <st c="72347">完全托管服务，如 Pinecone，可以简化部署和管理，减轻您团队的操作负担。</st> <st c="72473">另一方面，自托管解决方案，如 Weaviate，提供更多控制和灵活性，允许您进行定制和微调以满足您的</st> <st c="72625">特定需求。</st>

+   **<st c="72647">数据安全和合规性</st>**<st c="72676">：评估向量存储提供的安全功能和访问控制，确保它们符合您所在行业的合规性要求。</st> <st c="72826">如果您处理敏感数据，评估向量存储的加密和数据保护能力。</st> <st c="72935">考虑向量存储满足数据隐私法规和标准的能力，例如根据您的</st> <st c="73060">特定需求，GDPR 或 HIPAA。</st>

+   **<st c="73075">成本和许可</st>**<st c="73094">：矢量存储的定价模式是什么？</st> <st c="73144">它是基于数据量、搜索操作，还是多种因素的组合？</st> <st c="73220">考虑矢量存储的长期成本效益，同时考虑您 RAG 系统的可扩展性和增长预测。</st> <st c="73362">评估与矢量存储相关的许可费用、基础设施成本和维护费用。</st> <st c="73470">开源解决方案可能具有较低的前期成本，但需要更多的内部专业知识和维护资源，而托管服务可能具有更高的订阅费用，但提供简化的管理和支持。</st>

+   **<st c="73689">生态系统和集成</st>**<st c="73716">：在选择矢量存储时，评估它支持的生态系统和集成非常重要。</st> <st c="73821">考虑不同编程语言的客户端库、SDK 和 API 的可用性，因为这可以极大地简化开发过程，并使与现有代码库的无缝集成成为可能。</st> <st c="74035">评估矢量存储与其他在 RAG 系统中常用工具和框架的兼容性，例如 NLP 库或机器学习框架。</st> <st c="74196">支持社区的一般规模也很重要；确保它具有足够的规模以成长和繁荣。</st> <st c="74318">具有强大生态系统和广泛集成的矢量存储可以为您的 RAG 系统提供更多灵活性和扩展功能的机会。</st>

<st c="74482">通过仔细评估这些因素，并将它们与您的具体需求相匹配，您在选择 RAG 系统的矢量存储时可以做出明智的决定。</st> <st c="74656">进行彻底的研究，比较不同的选项，并考虑您选择的长期影响，包括可扩展性、性能和可维护性。</st>

<st c="74843">请记住，矢量存储的选择不是一刀切的决定，它可能会随着您的 RAG 系统的发展以及需求的变化而演变。</st> <st c="74995">定期重新评估您的矢量存储选择，并根据需要调整，以确保最佳性能和与整体系统架构的一致性至关重要。</st> <st c="75144">系统架构。</st>

# <st c="75164">总结</st>

将向量和向量存储集成到 RAG 系统中是提高信息检索和生成任务效率和准确性的基础。<st c="75340">通过仔细选择和优化您的向量化和向量存储方法，您可以显著提高您 RAG 系统的性能。</st> 向量化技术和向量存储只是向量在 RAG 系统中发挥作用的一部分；它们也在我们的检索阶段发挥着重要作用。<st c="75655">在下一章中，我们将讨论向量在检索中扮演的角色，深入探讨向量相似性搜索算法和服务。</st>
