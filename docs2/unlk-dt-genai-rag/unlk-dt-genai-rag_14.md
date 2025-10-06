# 14

# 用于改进结果的 RAG 相关高级技术

在本章的最后，我们探讨了几个高级技术来改进**<st c="131">检索增强生成</st>** <st c="161">(**<st c="163">RAG</st>**<st c="166">)应用。<st c="183">这些技术超越了基本的 RAG 方法，以应对更复杂的挑战并实现更好的结果。</st> <st c="308">我们的起点将是我们在前几章中已经使用过的技术。</st> <st c="389">我们将在此基础上构建，了解它们的不足之处，以便我们可以引入新的技术来弥补差距，并将你的 RAG 工作推进得更远。</st>

在本章中，你将通过一系列代码实验室获得通过实现这些高级技术的实践经验。<st c="697">我们的主题将包括以下内容：</st> <st c="721">以下：</st>

+   简单 RAG 及其局限性

+   混合 RAG/多向量 RAG 以提高检索

+   在混合 RAG 中进行重新排序

+   代码实验室 14.1 – <st c="858">查询扩展</st>

+   代码实验室 14.2 – <st c="890">查询分解</st>

+   代码实验室<st c="909">14.3 –</st> **<st c="926">多模态</st>** **<st c="938">RAG</st>** <st c="941">(**<st c="943">MM-RAG</st>**<st c="949">)</st>

+   其他高级 RAG 技术<st c="981">以供探索</st>

这些技术通过增强查询、将问题分解为子问题以及结合多种数据模态来增强检索和生成。<st c="1151">我们还讨论了一系列其他高级 RAG 技术，包括索引、检索、生成以及整个 RAG 管道。</st> <st c="1279">我们从讨论简单 RAG 开始，这是我们在第<st c="1374">第二章</st> <st c="1383">中回顾的 RAG 的主要方法，你应该现在感到非常熟悉</st> <st c="1428">。</st>

# 技术要求

本章的代码放置在以下 GitHub<st c="1519">存储库</st> <st c="1531">中：</st> [<st c="1531">https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_14</st>](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_14 )

# 简单 RAG 及其局限性

<st c="1658">到目前为止，我们已经研究了三种类型的 RAG 方法，</st> **<st c="1718">朴素 RAG</st>**<st c="1727">，</st> **<st c="1729">混合 RAG</st>**<st c="1739">，以及</st> **<st c="1745">重排序</st>**<st c="1755">。最初，我们</st> <st c="1771">正在使用所谓的朴素 RAG。</st> <st c="1815">这是我们在第二章</st> *<st c="1881">第二章</st>* <st c="1890">以及随后的多个代码实验室中使用的 RAG 基本方法。</st> <st c="1921">朴素 RAG 模型，RAG 技术的最初迭代，为将检索机制与生成模型集成提供了一个基础框架，尽管在灵活性和可扩展性方面存在限制。</st>

<st c="2130">朴素 RAG 检索大量的碎片化上下文块，这些块是我们矢量化并放入 LLM 上下文窗口的文本块。</st> <st c="2261">如果你不使用足够大的文本块，你的上下文将</st> <st c="2326">经历更高的碎片化程度。</st> <st c="2369">这种碎片化导致对上下文和块内语义的理解和捕获减少，从而降低了你的 RAG 应用检索机制的有效性。</st> <st c="2561">在典型的朴素 RAG 应用中，你正在使用某种类型的语义搜索，因此仅使用该类型的搜索就会暴露出这些限制。</st> <st c="2725">因此，我们引入了一种更高级的检索类型：混合搜索。</st>

# <st c="2805">混合 RAG/多向量 RAG 以改进检索</st>

<st c="2856">混合 RAG 通过在检索过程中使用多个向量来扩展朴素 RAG 的概念，而不是依赖于查询和文档的单个向量表示。</st> <st c="3043">我们在</st> <st c="3054">第八章</st> <st c="3090">中深入探讨了混合 RAG，不仅在代码实验室中使用了 LangChain 推荐的机制，而且还自己重新创建了该机制，以便我们可以看到其内部工作原理。</st> <st c="3099">也称为多向量 RAG，混合 RAG 不仅涉及语义和关键词搜索，正如我们在代码实验室中看到的那样，还可以混合任何适合你的</st> <st c="3448">RAG 应用的不同向量检索技术。</st>

我们的混合 RAG 代码实验室引入了关键词搜索，这扩大了我们的搜索能力，导致更有效的检索，尤其是在处理具有较弱上下文的内容（如名称、代码、内部缩写和类似文本）时。<st c="3721">这种多向量方法使我们能够考虑查询和数据库内容更广泛的方面。</st> <st c="3831">这反过来可以提高检索信息的关联性和准确性，以支持生成过程。</st> <st c="3955">这导致生成的文本不仅更相关、更信息丰富，而且与输入查询的细微差别也更加一致。</st> <st c="4095">多向量 RAG 在需要生成内容具有高度精确性和细微差别的应用中特别有用，例如技术写作、学术研究辅助、包含大量内部代码和实体引用的内部公司文档以及复杂的问答系统。</st> <st c="4414">但多向量 RAG 并不是我们在第八章中探索的唯一先进技术；我们还<st c="4505">应用了</st> **<st c="4513">重新排序</st>**<st c="4523">。</st>

# <st c="4524">混合 RAG 中的重新排序</st>

在第八章中，除了我们的混合 RAG 方法，我们还介绍了一种形式的重新排序，这是另一种常见的先进 RAG 技术。<st c="4684">在语义搜索和关键词搜索完成检索后，我们根据它们是否同时出现在两个集合中以及它们最初排名的位置重新排序结果。</st> <st c="4876">最初排名。</st>

<st c="4887">因此，你已经走过了三种 RAG 技术，包括两种先进技术！<st c="4982">但本章的重点是向您介绍三种更多的先进方法：</st> **<st c="5058">查询扩展</st>**<st c="5073">、**<st c="5075">查询分解</st>**<st c="5094">和**<st c="5100">MM-RAG</st>**<st c="5106">。我们还将提供您可以探索的许多其他方法的列表，但<st c="5185">我们筛选并挑选出这些三种先进的 RAG 技术，因为它们在广泛的 RAG 应用中得到了应用。</st>

<st c="5321">在本章的第一个代码实验室中，我们将讨论</st> <st c="5380">查询扩展。</st>

# <st c="5396">代码实验室 14.1 – 查询扩展</st>

<st c="5428">本实验室的代码可以在 GitHub 仓库的<st c="5544">`CHAPTER14`</st> <st c="5526">目录下的<st c="5471">`CHAPTER14-1_QUERY_EXPANSION.ipynb`</st> <st c="5504">文件中找到。</st>

<st c="5562">许多增强 RAG 的技术都集中在改进一个领域，例如检索或生成，但查询扩展有潜力同时改进这两个方面。</st> <st c="5714">我们已经在</st> *<st c="5771">第十三章</st>*<st c="5781">中讨论了扩展的概念，但那主要关注 LLM 的输出。</st> <st c="5823">在这里，我们将这个概念聚焦于模型的输入，通过添加额外的关键词或短语来增强原始提示。</st> <st c="5945">这种方法可以通过向用于检索的用户查询添加更多上下文来提高检索模型的理解，从而增加检索相关文档的机会。</st> <st c="6132">通过改进检索，你已经在帮助提高生成，给它提供了更好的工作上下文，但这种方法也有潜力产生一个更有效的查询，这反过来也有助于 LLM 提供</st> <st c="6370">改进的响应。</st>

<st c="6388">通常，带有答案的查询扩展的工作方式是，你将用户查询立即发送到 LLM，并附带一个专注于获取问题初始答案的提示，即使你还没有向它展示在 RAG 应用中通常展示的任何典型上下文。</st> <st c="6681">从 LLM 的角度来看，这类变化可以帮助扩大搜索范围，同时不失对</st> <st c="6790">原始意图的关注。</st>

<st c="6806">在创建</st> `<st c="6859">rag_chain_from_docs</st>` <st c="6878">链的单元格上方开始一个新的单元格。</st> <st c="6886">我们将介绍一系列提示模板来完成这个任务：</st>

```py
 from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
```

<st c="7071">让我们回顾一下这些提示模板及其</st> <st c="7120">用途：</st>

+   `<st c="7131">ChatPromptTemplate</st>` <st c="7150">类：这提供了一个创建基于聊天的提示的模板，我们可以使用它将我们的其他提示模板组合成一个更</st> <st c="7283">基于聊天的方法。</st>

+   `<st c="7303">HumanMessagePromptTemplate</st>` <st c="7330">类：这提供了一个创建聊天提示中人类消息的提示模板。</st> <st c="7418">`<st c="7422">HumanMessage</st>` <st c="7434">对象代表在语言模型对话中由人类用户发送的消息。</st> <st c="7527">我们将使用来自</st> `<st c="7564">user_query</st>` <st c="7574">字符串的提示来填充这个提示，这个字符串来自</st> *<st c="7604">人类</st>` <st c="7609">在这个场景中！</st>

+   `<st c="7627">SystemMessagePromptTemplate</st>` <st c="7655">类：*<st c="7667">系统</st>` <st c="7673">也获得了一个提示，对于一个基于聊天的 LLM 来说，这与人类生成的提示相比具有不同的意义。</st> <st c="7787">这为我们提供了一个创建聊天提示中的这些系统消息的提示模板。</st>

<st c="7881">接下来，我们想要创建一个</st> <st c="7908">函数，该函数将处理查询扩展，并使用我们刚刚讨论的不同提示模板。</st> `<st c="8023">这是我们将使用的系统消息提示，您需要将其定制为您的 RAG 系统关注的任何领域 - 在这种情况下是环境报告：</st>`

`<st c="8195">"您是一位有用的环境研究专家。</st> <st c="8256">提供一个可能出现在年度环境报告等文档中的问题的示例答案。"</st>` `<st c="8354">环境报告。</st>`

<st c="8376">这将是我们创建的函数的第一个步骤</st> <st c="8421">：</st>

```py
 def augment_query_generated(user_query):
     system_message_prompt = SystemMessagePromptTemplate.from_template(
           "You are a helpful expert environmental research assistant. Provide an example answer to the given question, that might be found in a document like an annual environmental report." )
     human_message_prompt = HumanMessagePromptTemplate.from_template("{query}")
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt, human_message_prompt])
    response = chat_prompt.format_prompt(
        query=user_query).to_messages()
    result = llm(response)
    content = result.content
    return content
```

<st c="9025">在这里，您可以看到我们利用了所有三种类型的提示模板来制定发送给 LLM 的整体消息集。</st> `<st c="9150">最终，这导致 LLM 给出了一个响应，它尽力回答</st> <st c="9236">我们的问题。</st>`

<st c="9249">让我们提供一些调用此函数的代码，这样我们就可以讨论输出，表示</st> *<st c="9354">查询扩展</st>* <st c="9363">中的</st> <st c="9367">扩展</st>：

```py
 original_query = "What are Google's environmental initiatives?" hypothetical_answer = augment_query_generated(
    original_query)
joint_query = f"{original_query} {hypothetical_answer}"
print(joint_query)
```

<st c="9585">在这里，我们调用</st> <st c="9606">我们的用户查询</st> `<st c="9622">original_query</st>`<st c="9636">，表示这是我们即将进行扩展的源查询。</st> `<st c="9718">假设答案</st>` <st c="9741">实例是我们从 LLM 获取的响应字符串。</st> <st c="9800">然后，您将原始用户查询与想象中的答案连接成一个</st> `<st c="9875">联合查询</st>` <st c="9886">字符串，并使用该字符串作为新的查询。</st> `<st c="9925">输出将类似于以下内容（为了简洁而截断）：</st>`

```py
 What are Google's environmental initiatives? In 2022, Google continued to advance its environmental initiatives, focusing on sustainability and reducing its carbon footprint. Key initiatives included:
1\. **Carbon Neutrality and Renewable Energy**: Google has maintained its carbon-neutral status since 2007 and aims to operate on 24/7 carbon-free energy by 2030\. In 2022, Google procured over 7 gigawatts of renewable energy, making significant strides towards this goal. 2\. **Data Center Efficiency**: Google's data centers are among the most energy-efficient in the world. In 2022, the company achieved an average power usage effectiveness (PUE) of 1.10, significantly lower than the industry average. This was accomplished through advanced cooling technologies and AI-driven energy management systems. 3\. **Sustainable Products and Services**…[TRUNCATED]
```

<st c="10849">这是一个更长的答案。</st> `<st c="10878">我们的 LLM 真的尽力彻底回答了它！</st>` `<st c="10924">这个初始答案是您发送给它的原始用户查询的假设或想象中的答案。</st>` `<st c="11033">通常，我们避免使用想象中的答案，但在这里，我们利用了它们，因为它们帮助我们深入了解 LLM 的内部工作原理，并提取出与您将要使用的</st> `<st c="11215">user_query</st>` <st c="11225">字符串相一致的概念。</st>`

<st c="11254">在这个阶段，我们将逐步执行原始代码，但与过去不同，我们将传递一个连接了原始答案和一个想象中的答案到我们之前的</st> `<st c="11477">RAG 管道</st>` <st c="11340">original_query</st> <st c="11354">字符串，而不是我们过去所拥有的</st> <st c="11364">字符串：</st>

```py
 result_alt = rag_chain_with_source.invoke(joint_query)
retrieved_docs_alt = result_alt['context']
print(f"Original Question: {joint_query}\n")
print(f"Relevance Score:
    {result_alt['answer']['relevance_score']}\n")
print(f"Final Answer:\n{
    result_alt['answer']['final_answer']}\n\n")
print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs_alt, start=1):
     print(f"Document {i}: Document ID:
        {doc.metadata['id']} source:
        {doc.metadata['search_source']}")
     print(f"Content:\n{doc.page_content}\n")
```

<st c="11996">您将看到传递给我们的原始 RAG 管道的查询是更长的</st> `<st c="12090">joint_query</st>` <st c="12101">字符串，然后我们看到一组扩展的结果，这些结果将我们提供的数据与 LLM 帮助</st> <st c="12229">添加的扩展结构混合在一起。</st>

<st c="12236">因为 LLM 以 Markdown 版本返回文本，我们可以使用 IPython 以良好的格式打印。</st> <st c="12348">此代码打印出以下内容：</st>

```py
 from IPython.display import Markdown, display
markdown_text_alt = result_alt['answer']['final_answer']
display(Markdown(markdown_text_alt))
```

<st c="12523">以下是</st> <st c="12532">输出：</st>

```py
 Google has implemented a comprehensive set of environmental initiatives aimed at sustainability and reducing its carbon footprint. Here are the key initiatives: <st c="12705">1\.</st> <st c="12708">Carbon Neutrality and Renewable Energy</st>: Google has been carbon-neutral since 2007 and aims to operate on 24/7 carbon-free energy by 2030\. In 2022, Google procured over 7 gigawatts of renewable energy. <st c="12910">2\.</st> <st c="12913">Data Center Efficiency</st>: Google's data centers are among the most energy-efficient globally, achieving an average power usage effectiveness (PUE) of 1.10 in 2022\. This was achieved through advanced cooling technologies and AI-driven energy management systems. <st c="13173">…[TRUNCATED FOR BREVITY]</st>
<st c="13197">3\.</st> <st c="13201">Supplier Engagement</st>: Google works with its suppliers to build an energy-efficient, low-carbon, circular supply chain, focusing on improving environmental performance and integrating sustainability principles. <st c="13411">4\.</st> <st c="13414">Technological Innovations</st>: Google is investing in breakthrough technologies, such as next-generation geothermal power and battery-based backup power systems, to optimize the carbon footprint of its operations. These initiatives reflect Google's commitment to sustainability and its role in addressing global environmental challenges. The company continues to innovate and collaborate to create a more sustainable future. ---- END OF OUTPUT ----
```

<st c="13859">将此与原始查询的结果进行比较，看看您是否认为它改进了答案！</st> <st c="13964">如您所见，您确实会得到不同的响应，您可以确定最适合您的</st> <st c="14073">RAG 应用的最佳方法。</st>

<st c="14090">采用这种方法的一个重要方面是，您将 LLM 引入了检索阶段，而过去我们只在使用 LLM 进行生成阶段时使用它。</st> <st c="14272">然而，当这样做的时候，现在 prompt 工程成为检索阶段的一个关注点，而之前我们只担心它在生成阶段。</st> <st c="14434">然而，这种方法与我们讨论的提示工程章节（</st>*<st c="14523">第十三章</st>*<st c="14534">）中讨论的方法相似，我们在那里讨论了迭代直到我们从</st> <st c="14604">我们的 LLM 中获得更好的结果。</st>

<st c="14612">有关查询扩展的更多信息，您可以在以下位置阅读原始论文：</st> <st c="14686">这里：</st> [<st c="14692">https://arxiv.org/abs/2305.03653</st>](https://arxiv.org/abs/2305.03653 )

<st c="14724">查询扩展只是许多增强原始查询以帮助改进 RAG 输出的方法之一。</st> <st c="14836">我们在本章末尾列出了更多，但在我们下一个代码实验室中，我们将解决一种称为查询分解的方法，因为它特别强调问答，因此在 RAG 场景中非常有用。</st>

# <st c="15066">代码实验室 14.2 – 查询分解</st>

<st c="15102">此实验室的代码可以在</st> `<st c="15145">CHAPTER14-2_DECOMPOSITION.ipynb</st>` <st c="15176">文件中找到，该文件位于</st> `<st c="15189">CHAPTER14</st>` <st c="15198">目录下的</st> <st c="15216">GitHub 仓库中。</st>

<st c="15234">查询分解是一种专注于在 GenAI 空间内改进问答的策略。</st> <st c="15333">它属于</st> <st c="15347">查询翻译类别，这是一组专注于改进 RAG 管道初始阶段的检索的方法。</st> <st c="15488">使用查询分解，我们将</st> *<st c="15522">分解</st> *<st c="15531">或把一个问题分解成更小的问题。</st> <st c="15581">这些小问题可以根据你的需求依次或独立处理，从而在不同场景下使用 RAG 时提供更多灵活性。</st> <st c="15760">在每个问题回答后，都有一个整合步骤，它提供最终响应，通常比使用原始 RAG 的响应有更广阔的视角。</st>

<st c="15933">还有其他查询翻译方法，如 RAG-Fusion 和多查询，它们专注于子问题，但这种方法专注于分解问题。</st> <st c="16097">我们将在本章末尾更多地讨论这些其他技术。</st>

<st c="16173">在提出这种方法的论文中，由谷歌研究人员撰写，他们称之为 Least-to-Most，或分解。</st> <st c="16293">LangChain 在其网站上对此方法有文档，称之为查询分解。</st> <st c="16388">因此，当我们谈论这个</st> <st c="16446">特定方法时，我们处于一个非常不错的公司中！</st>

<st c="16466">我们将介绍更多概念，以帮助我们理解如何实现</st> <st c="16555">查询</st> <st c="16560">分解：</st>

+   <st c="16575">第一个概念是</st> **<st c="16597">思维链</st>** <st c="16613">(</st>**<st c="16615">CoT</st>**<st c="16618">)，这是一种提示工程策略，我们以模仿人类推理的方式结构化</st> <st c="16674">输入提示，目的是提高语言模型在需要逻辑、计算</st> <st c="16821">和决策的任务上的性能。</st>

+   <st c="16841">第二个概念是</st> **<st c="16864">交错检索</st>**<st c="16886">，其中你在由 CoT 驱动的</st> <st c="16937">提示和检索之间来回移动，*<st c="16960">交错</st>*, <st c="16972">与简单地传递用户查询进行检索相比，目的是为了在后续推理步骤中检索到更多相关信息。</st> <st c="17102">这种组合被称为</st> **<st c="17144">交错检索与思维链</st>** <st c="17173">或</st> **<st c="17177">IR-CoT</st>**<st c="17183">。</st>

<st c="17184">将所有这些整合在一起，你最终得到一种方法，它将问题分解为子问题，并通过动态检索过程逐步解决它们。</st> <st c="17341">在将你的原始用户查询分解为子问题之后，你开始处理第一个问题，检索文档，回答它，然后对第二个问题进行检索，将第一个问题的答案添加到结果中，然后使用所有这些数据来回答问题 2。</st> <st c="17654">这个过程会一直持续到你得到最后一个答案，这将是你的</st> <st c="17762">最终答案。</st>

<st c="17775">有了所有这些解释，你可能只想跳入代码看看它是如何工作的，所以让我们</st> <st c="17877">开始吧！</st>

<st c="17889">我们将导入几个</st> <st c="17914">新包：</st>

```py
 from langchain.load import dumps, loads
```

<st c="17967">从</st> `<st c="17972">dumps</st>` <st c="17977">和</st> `<st c="17982">loads</st>` <st c="17987">函数导入自</st> `<st c="18012">langchain.load</st>` <st c="18026">用于将 Python 对象序列化和反序列化为字符串表示。</st> <st c="18137">在我们的代码中，我们将使用它来在去重之前将每个</st> <st c="18181">Document</st> <st c="18189">对象转换为字符串表示，然后再转换回来。</st>

<st c="18268">然后我们跳过检索器定义，添加一个单元格，我们将在这里添加我们的分解</st> <st c="18377">提示、链和代码来运行它。</st> <st c="18412">首先创建一个新的</st> <st c="18438">提示模板：</st>

```py
 prompt_decompose = PromptTemplate.from_template(
     """You are an AI language model assistant. Your task is to generate five different versions of
     the
     given user query to retrieve relevant documents from a
     vector search. By generating multiple perspectives on
     the user question, your goal is to help the user
     overcome some of the limitations of the distance-based
     similarity search. Provide these alternative
     questions
     separated by newlines. Original question: {question}"""
)
```

<st c="18928">阅读这个</st> `<st c="18964">PromptTemplate</st>` <st c="18978">对象中的字符串，我们得到一个提示版本，解释给 LLM 如何执行我们正在寻找的分解。</st> <st c="19088">这是一个非常透明的对 LLM 的请求，解释了我们要克服的问题以及我们需要它做什么！</st> <st c="19208">我们还提示 LLM 以特定格式提供结果。</st> <st c="19275">这可能会有些风险，因为 LLM 有时会返回意外的结果，即使被明确提示以特定格式返回。</st> <st c="19397">在更健壮的应用中，这是一个运行检查以确保你的响应格式正确的好地方。</st> <st c="19513">但在这个简单的例子中，我们使用的 ChatGPT-4o-mini 模型似乎在以正确格式返回它时表现良好。</st> <st c="19618">正确格式。</st>

<st c="19632">接下来，我们设置链，使用我们在链中通常使用的各种元素，但使用提示</st> <st c="19740">来分解：</st>

```py
 decompose_queries_chain = (
     prompt_decompose
     | llm
     | str_output_parser
     | (lambda x: x.split("\n"))
)
```

<st c="19854">这是一个自解释的链；它使用提示模板、代码中之前定义的 LLM、输出解析器，然后应用格式化以获得更</st> <st c="20012">易读的结果。</st>

<st c="20028">要调用此链，我们</st> <st c="20052">实现以下代码：</st>

```py
 decomposed_queries = decompose_queries_chain.invoke(
    {"question": user_query})
print("Five different versions of the user query:")
print(f"Original: {user_query}")
for i, question in enumerate(decomposed_queries, start=1):
     print(f"{question.strip()}")
```

<st c="20333">此代码调用我们设置的链，并为我们提供原始查询，以及我们的分解提示和 LLM 生成的五个新查询：</st>

```py
 Five different versions of the user query:
Original: What are Google's environmental initiatives? What steps is Google taking to address environmental concerns? How is Google contributing to environmental sustainability? Can you list the environmental programs and projects Google is involved in? What actions has Google implemented to reduce its environmental impact? What are the key environmental strategies and goals of Google?
```

<st c="20925">LLM 在将我们的查询分解成一系列相关问题方面做得非常出色，这些问题涵盖了不同方面，有助于回答原始查询。</st>

<st c="22544">但分解概念只完成了一半！接下来，我们将运行所有问题通过检索，与我们在过去的代码实验室中拥有的相比，这将给我们提供一个更健壮的检索上下文集合：</st>

<st c="21318">我们将首先设置一个函数来格式化基于所有这些</st> <st c="21416">新查询检索到的文档：</st>

```py
 def format_retrieved_docs(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents
        for doc in sublist]
    print(f"FLATTENED DOCS: {len(flattened_docs)}")
    deduped_docs = list(set(flattened_docs))
    print(f"DEDUPED DOCS: {len(deduped_docs)}")
    return [loads(doc) for doc in deduped_docs]
```

<st c="21729">此函数将提供一个</st> <st c="21759">列表的列表，代表每个检索到的文档集合列表。</st> <st c="21868">我们平铺这个列表的列表，这意味着我们只将其变成一个长列表。</st> <st c="21943">然后我们使用从 `<st c="21991">LangChain.load</st>` 导入的 `<st c="21959">dumps</st>` <st c="21964">函数将每个</st> `<st c="22022">Document</st>` <st c="22030">对象转换为字符串；我们根据该字符串进行去重，然后将其返回为列表。</st> <st c="22133">我们还打印出在去重前后我们最终拥有的文档数量，以查看我们的去重工作表现如何。</st> <st c="22245">在这个例子中，在运行了</st> `<st c="22282">decompose_queries_chain</st>` <st c="22305">链之后，我们从</st> `<st c="22326">100</st>` <st c="22329">个文档</st> <st c="22340">减少到</st> `<st c="22343">67</st>`<st c="22345">：</st>

```py
 FLATTENED DOCS: 100
DEDUPED DOCS: 67
```

<st c="22384">让我们设置一个链，该链将运行我们之前的分解链、所有新查询的检索以及我们刚刚创建的函数的最终格式化：</st>

```py
 retrieval_chain = (
     decompose_queries_chain
     | ensemble_retriever.map()
     | format_retrieved_docs
)
```

<st c="22977">这一相对简短的代码行完成了许多工作！最终结果是包含所有从原始查询和分解生成的查询的</st> `<st c="22739">67</st>` <st c="22741">个文档的集合。</st> <st c="22842">请注意，我们已经直接将之前的</st> `<st c="22879">decompose_queries_chain</st>` <st c="22902">链添加到其中，因此不需要单独调用该链。</st>

<st c="20478">我们使用这一行代码将此链的结果分配给一个</st> `<st c="23019">docs</st>` <st c="23023">变量：</st>

```py
 docs = retrieval_chain.invoke({"question":user_query})
```

通过调用这个链，我们检索到了与之前方法相比的显著数量的文档（</st>`<st c="23198">67</st>`<st c="23201">），但我们仍然需要使用我们扩展的检索结果来运行我们的最终 RAG 步骤。</st> <st c="23316">在此之后，大部分代码保持不变，但我们用我们刚刚构建的</st> `<st c="23408">retrieval_chain</st>` <st c="23423">链替换了集成链：</st>

```py
 rag_chain_with_source = RunnableParallel(
    {"context": retrieval_chain,
     "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)
```

<st c="23587">这把我们的新代码整合到了之前的 RAG 应用中。</st> <st c="23654">运行这一行将运行我们刚刚添加的所有链，因此没有必要像我们在这个例子中那样单独运行它们。</st> <st c="23797">这是一组大型连贯的代码，它将我们之前的努力与这个新的强大 RAG 技术结合起来。</st> <st c="23909">我们邀请您将此技术当前的结果与过去的代码实验室结果进行比较，以查看细节是如何更好地填充并为我们提供更广泛的关于我们</st> `<st c="24101">original_query</st>` <st c="24115">链</st> <st c="24122">询问的主题的覆盖范围：</st>

```py
 Google has implemented a wide range of environmental initiatives aimed at improving sustainability and reducing its environmental impact. Here are some key initiatives based on the provided context. Here is the beginning of the current results, truncated to just the first couple of bullets: <st c="24426">1\.</st> <st c="24429">Campus and Habitat Restoration</st>:
Google has created and restored more than 40 acres of habitat on its campuses and surrounding urban landscapes, primarily in the Bay Area. This includes planting roughly 4,000 native trees and restoring ecosystems like oak woodlands, willow groves, and wetland habitats. <st c="24733">2\.</st> <st c="24736">Carbon-Free Energy</st>:
Google is working towards achieving net-zero emissions and 24/7 carbon-free energy (CFE) by 2030\. This involves clean energy procurement, technology innovation, and policy advocacy. They have also launched a policy roadmap for 24/7 CFE and are advocating for strong public policies to decarbonize electricity grids worldwide. <st c="25083">3\.</st> <st c="25086">Water Stewardship</st>…[TRUNCATED FOR BREVITY]
```

<st c="25127">像这样的高级技术</st> <st c="25148">根据您 RAG 应用的目标，可以提供非常有希望的结果！</st>

<st c="25237">有关此方法的更多信息，请访问原始</st> <st c="25296">论文：</st> [<st c="25303">https://arxiv.org/abs/2205.10625</st>](https://arxiv.org/abs/2205.10625 )

<st c="25335">对于本书的下一个也是最后一个代码实验室，我们将超越文本的世界，将我们的 RAG 扩展到其他模态，例如图像和视频，使用一种称为 MM-RAG 的技术。</st> <st c="25513">MM-RAG。</st>

# <st c="25527">代码实验室 14.3 – MM-RAG</st>

<st c="25550">此实验室的代码可以在 GitHub 存储库的</st> `<st c="25593">CHAPTER14-3_MM_RAG.ipynb</st>` <st c="25617">文件中找到，位于</st> `<st c="25630">CHAPTER14</st>` <st c="25639">目录下。</st>

<st c="25675">这是一个很好的例子，说明了</st> <st c="25702">当缩写词真的能帮助我们更快地说话时。</st> <st c="25750">试着大声说出</st> *<st c="25761">多模态检索增强再生</st>* <st c="25805">一次，你可能会想从此以后就使用 MM-RAG！</st> <st c="25873">但我跑题了。</st> <st c="25888">这是一个具有突破性的方法，预计在不久的将来会获得很多关注。</st> <st c="25982">它更好地代表了我们作为人类处理信息的方式，所以它肯定很棒，对吧？</st> <st c="26071">让我们首先回顾一下使用</st> <st c="26118">多种模式</st> 的概念。

## <st c="26133">多模态</st>

<st c="26145">到目前为止，我们讨论的一切都集中在文本上：以文本为输入，根据该输入检索文本，然后将检索到的文本传递给 LLM，最终生成文本输出。</st> <st c="26365">那么非文本呢？</st> <st c="26390">随着构建这些 LLM 的公司开始提供强大的多模态功能，我们如何将这些多模态功能整合到我们的</st> <st c="26547">RAG 应用中？</st>

<st c="26564">多模态简单来说就是处理多种“模式”，包括文本、图像、视频、音频以及其他任何类型的输入。</st> <st c="26710">这些多种模式可以体现在输入、输出或两者之中。</st> <st c="26783">例如，你可以传入文本并得到一张图片作为回应，这就是</st> <st c="26851">多模态。</st> <st c="26865">你可以传入一张图片并得到</st> <st c="26897">文本作为回应（称为字幕），这也是</st> <st c="26941">多模态。</st>

<st c="26958">更高级的方法还可以包括传入一个文本提示</st> `<st c="27031">"将此图像转换为一段视频，进一步展示瀑布，并添加瀑布的声音"` <st c="27133">以及一张瀑布的图片，并得到一段视频作为回应，该视频将用户带入图像中的瀑布，并添加了瀑布的声音。</st> <st c="27300">这将代表四种不同的模式：文本、图像、视频和音频。</st> <st c="27375">鉴于现在存在具有类似 API 的模型，这些模型现在具有这些功能，考虑如何将它们应用于我们的 RAG 方法，使用 RAG 再次挖掘我们企业数据宝库中存储的其他类型的内容，这是一个短而合理的步骤。</st> <st c="27683">让我们讨论使用</st> <st c="27721">多模态方法的好处。</st>

## <st c="27742">多模态的益处</st>

<st c="27766">这种方法</st> <st c="27780">利用了 RAG 技术在理解和利用多模态数据源方面的优势，允许创建更具吸引力、信息丰富和上下文丰富的输出。</st> <st c="27963">通过整合多模态数据，这些 RAG 系统可以提供更细微、更全面的答案，生成更丰富的内容，并与用户进行更复杂的交互。</st> <st c="28149">应用范围从能够理解和生成多媒体响应的增强型对话代理，到能够生成复杂的多模态文档和演示的高级内容创作工具。</st> <st c="28367">MM-RAG 代表了在使 RAG 系统更加灵活和能够以类似于人类感官和</st> <st c="28527">认知体验的方式理解世界方面的一项重大进步。</st>

<st c="28549">与我们在第七章和第八章中关于向量的讨论类似，重要的是要认识到向量嵌入在 MM-RAG 中扮演着重要的角色。</st> <st c="28651">the important role vector embeddings play in MM-RAG</st> <st c="28704">as well.</st>

## <st c="28712">多模态向量嵌入</st>

<st c="28742">MM-RAG 被启用，因为向量嵌入不仅能表示文本；它们还能表示你传递给它的任何类型的数据。</st> <st c="28879">某些数据需要做更多准备工作才能将其转换为可以矢量化的事物，但所有类型的数据都有可能被矢量化并供 RAG 应用使用。</st> <st c="29071">如果你还记得，矢量化在本质上是将你的数据转换成数学表示，而数学和向量是</st> **<st c="29234">深度学习</st>** <st c="29247">(**<st c="29249">DL</st>**) 模型的**<st c="29251">主要语言</st>**，这些模型构成了我们所有 RAG 应用的基础。</st>

<st c="29318">你可能记得向量的另一个方面是向量空间的概念，其中相似的概念在向量空间中彼此更接近，而不相似的概念则更远。</st> <st c="29513">当你将多个模式混合在一起时，这一点仍然适用，这意味着像海鸥这样的概念应该以相似的方式表示，无论是单词**<st c="29681">海鸥</st>**，海鸥的图像，海鸥的视频，还是海鸥尖叫声的音频剪辑。</st> <st c="29776">这种多模态嵌入概念，即同一上下文的跨模态表示，被称为**<st c="29877">模态独立性</st>**。这种向量空间概念的扩展是 MM-RAG 像单模态 RAG 一样服务于类似目的但具有多种数据模式的基础。</st> <st c="29934">关键概念是多模态向量嵌入在它们所代表的所有模态中保持语义相似性。</st> <st c="30051">The key concept is that multi-modal vector embeddings preserve semantic similarity across all modalities</st> <st c="30156">they represent.</st>

<st c="30171">当谈到在企业中使用 MM-RAG 时，重要的是要认识到企业中的大量数据存在于多种模式中，所以让我们接下来讨论这一点。</st> <st c="30324">that next.</st>

## <st c="30334">图像不仅仅是“图片”</st>

<st c="30365">图像不仅仅是漂亮的风景画或你在上次度假中拍摄的 500 张照片那么简单！</st> <st c="30496">在企业中，图像可以代表图表、流程图、某些时候被转换为图像的文本，以及更多。</st> <st c="30641">图像是企业的重要数据来源。</st>

<st c="30696">如果您还没有看过我们许多实验室中使用的代表</st> *<st c="30752">Google Environmental Report 2023</st>* <st c="30784">的 PDF 文件，您可能已经开始相信它只是基于文本的。</st> <st c="30880">但是打开它，您会看到我们一直在工作的文本周围的精心设计的图像。</st> <st c="30999">您看到的某些图表，尤其是那些高度设计的图表，实际上是图像。</st> <st c="31078">如果我们有一个想要利用那些图像中数据的 RAG 应用程序怎么办？</st> <st c="31145">让我们开始构建一个吧！</st>

## <st c="31202">在代码中介绍 MM-RAG</st>

<st c="31229">在这个实验室中，我们将进行以下操作：</st>

1.  <st c="31268">从 PDF 中提取文本和图像</st> <st c="31292">使用一个强大的开源包</st> <st c="31341">称为</st> `<st c="31348">unstructured</st>`<st c="31360">。</st>

1.  <st c="31361">使用多模态 LLM 从提取的图像中生成文本摘要。</st>

1.  <st c="31436">使用对原始图像的引用嵌入和检索这些图像摘要（以及我们之前已经使用的文本对象）。</st>

1.  <st c="31568">使用 Chroma 将图像摘要存储在多向量检索器中，Chroma 存储原始文本和图像及其摘要。</st>

1.  <st c="31698">将原始图像和文本块传递给同一个多模态 LLM 进行答案合成。</st>

<st c="31784">我们从安装一些您需要用于</st> <st c="31842">unstructured</st>`<st c="31901">的新包开始：</st>

```py
 %pip install "unstructured[pdf]"
%pip install pillow
%pip install pydantic
%pip install lxml
%pip install matplotlib
%pip install tiktoken
!sudo apt-get -y install poppler-utils
!sudo apt-get -y install tesseract-ocr
```

<st c="32132">以下是我们将在代码中使用的这些包将为我们做什么的列表：</st>

+   `<st c="32206">unstructured[pdf]</st>`<st c="32224">：The</st> `<st c="32231">unstructured</st>` <st c="32243">library is a Python library for extracting structured information from unstructured data, such as PDFs, images, and HTML pages.</st> <st c="32372">This installs only the PDF support from</st> `<st c="32412">unstructured</st>`<st c="32424">. There are many other</st> <st c="32447">documents supported that you can include if using those types of documents, or you can use</st> `<st c="32538">all</st>` <st c="32541">to get support for all documents</st> <st c="32575">they support.</st>

+   `<st c="32588">pillow</st>`<st c="32595">：The</st> `<st c="32602">pillow</st>` <st c="32608">library is a fork</st> <st c="32627">of the</st> `<st c="32669">pillow</st>` <st c="32675">library provides support for opening, manipulating, and saving various image file formats.</st> <st c="32767">在我们的代码中，我们使用</st> `<st c="32818">unstructured</st>`<st c="32830">时正在处理图像，并且</st> `<st c="32836">unstructured</st>` <st c="32848">使用</st> `<st c="32854">pillow</st>` <st c="32860">来帮助</st> <st c="32869">完成这项工作！</st>

+   `<st c="32879">pydantic</st>`<st c="32888">: The</st> `<st c="32895">pydantic</st>` <st c="32903">库是一个使用 Python 类型注解进行数据验证和设置管理的库。</st> <st c="32996">该</st> `<st c="33000">pydantic</st>` <st c="33008">库通常用于定义数据模型和验证</st> <st c="33074">输入数据。</st>

+   `<st c="33085">lxml</st>`<st c="33090">: The</st> `<st c="33097">lxml</st>` <st c="33101">库是一个用于处理 XML 和 HTML 文档的库。</st> <st c="33162">我们使用</st> `<st c="33169">lxml</st>` <st c="33173">与</st> `<st c="33188">非结构化</st>` <st c="33200">库或其他依赖项一起用于解析和从</st> <st c="33275">结构化文档中提取信息。</st>

+   `<st c="33296">matplotlib</st>`<st c="33307">: The</st> `<st c="33314">matplotlib</st>` <st c="33324">库是一个用于在 Python 中创建可视化的知名绘图库。</st>

+   `<st c="33404">tiktoken</st>`<st c="33413">: The</st> `<st c="33420">tiktoken</st>` <st c="33428">库是一个</st> **<st c="33442">字节对编码</st>** <st c="33460">(</st>**<st c="33462">BPE</st>**<st c="33465">) 分词器，用于与 OpenAI 的模型一起使用。</st> <st c="33508">BPE 最初是作为一个用于压缩文本的算法开发的，然后被 OpenAI 用于在预训练</st> <st c="33633">GPT 模型时进行分词。</st>

+   `<st c="33643">poppler-utils</st>`<st c="33657">: The</st> `<st c="33664">poppler</st>` <st c="33671">实用工具是一组用于操作 PDF 文件的命令行工具。</st> <st c="33742">在我们的代码中，</st> `<st c="33755">poppler</st>` <st c="33762">被</st> `<st c="33774">非结构化</st>` <st c="33786">用于从</st> <st c="33820">PDF 文件中提取元素。</st>

+   `<st c="33829">tesseract-ocr</st>`<st c="33843">: The</st> `<st c="33850">tesseract-ocr</st>` <st c="33863">引擎是一个开源的 OCR 引擎，可以从图像中识别和提取文本。</st> <st c="33949">这是另一个由</st> `<st c="33985">非结构化</st>` <st c="33997">库所需的库，用于支持 PDF，从图像中提取文本。</st>

<st c="34039">这些包提供了 langchain 和 unstructured 库及其在代码中使用的相关模块所需的各项功能和相关依赖。</st> <st c="34204">它们使任务如 PDF 解析、图像处理、数据验证、分词和 OCR 成为可能，这些对于处理和分析 PDF 文件以及生成对用户查询的响应是必不可少的。</st>

<st c="34399">我们现在将添加对这些包以及其他包的导入，以便我们可以在</st> <st c="34480">我们的代码中使用：</st>

```py
 from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.runnables import RunnableLambda
from langchain.storage import InMemoryStore
from langchain_core.messages import HumanMessage
import base64
import uuid
from IPython.display import HTML, display
from PIL import Image
import matplotlib.pyplot as plt
```

<st c="34894">这是一个 Python 包的很长列表，所以让我们逐个列出它们：</st>

+   `<st c="34987">MultiVectorRetriever</st>` `<st c="35008">from</st>` `<st c="35014">langchain.retrievers.multi_vector</st>` `<st c="35047">: The</st> `<st c="35054">MultiVectorRetriever</st>` `<st c="35074">package is a retriever that combines multiple vector stores and allows for efficient retrieval of documents based on similarity search.</st> `<st c="35211">In our code,</st>` `<st c="35224">MultiVectorRetriever</st>` `<st c="35244">is used to create a retriever that combines</st> `<st c="35289">vectorstore</st>` `<st c="35300">and</st> `<st c="35305">docstore</st>` `<st c="35313">for retrieving relevant documents based on the</st> `<st c="35361">user’s query.</st>`

+   `<st c="35374">UnstructuredPDFLoader</st>` `<st c="35396">from</st>` `<st c="35402">langchain_community.document_loaders</st>` `<st c="35438">: The</st> `<st c="35445">UnstructuredPDFLoader</st>` `<st c="35466">package is a document loader that extracts elements, including text and images, from a PDF file using the</st> `<st c="35573">unstructured</st>` `<st c="35586">library.</st> `<st c="35595">In our code,</st>` `<st c="35608">UnstructuredPDFLoader</st>` `<st c="35629">is used to load and extract elements from the specified PDF</st> `<st c="35690">file (</st>` `<st c="35696">short_pdf_path</st>` `<st c="35711">).</st>`

+   `<st c="35714">RunnableLambda</st>` `<st c="35729">from</st>` `<st c="35735">langchain_core.runnables</st>` `<st c="35759">: The</st> `<st c="35766">RunnableLambda</st>` `<st c="35780">class is a utility class that allows wrapping a function as a runnable component in</st> `<st c="35864">a LangChain pipeline.</st> `<st c="35887">In our code,</st>` `<st c="35900">RunnableLambda</st>` `<st c="35914">is used to wrap the</st> `<st c="35935">split_image_text_types</st>` `<st c="35957">and</st> `<st c="35962">img_prompt_func</st>` `<st c="35977">functions as runnable components in the</st> `<st c="36018">RAG chain.</st>`

+   `<st c="36028">InMemoryStore</st>` `<st c="36042">from</st>` `<st c="36048">langchain.storage</st>` `<st c="36065">: The</st> `<st c="36072">InMemoryStore</st>` `<st c="36085">class is a simple in-memory storage class that stores key-value pairs.</st> `<st c="36157">In our code,</st>` `<st c="36169">InMemoryStore</st>` `<st c="36183">is used as a document store for storing the actual document content associated with each</st> `<st c="36273">document ID.</st>`

+   `<st c="36285">HumanMessage</st>` `<st c="36298">from</st>` `<st c="36304">langchain_core.messages</st>` `<st c="36327">: We saw this type of prompt in</st> *<st c="36360">Code Lab 14.1</st>* `<st c="36373">already, representing a message sent by a human user in a conversation with the language model.</st> `<st c="36470">In this code lab,</st>` `<st c="36488">HumanMessage</st>` `<st c="36500">is used to construct prompt messages for image summarization</st> `<st c="36562">and description.</st>`

+   `<st c="36578">base64</st>`<st c="36585">: 在我们的代码中，</st> `<st c="36601">base64</st>` <st c="36607">用于将图像编码为</st> `<st c="36636">base64</st>` <st c="36642">字符串以进行存储</st> <st c="36663">和检索。</st>

+   `<st c="36677">uuid</st>`<st c="36682">: The</st> `<st c="36689">uuid</st>` <st c="36693">模块提供生成</st> `<st c="36789">uuid</st>` <st c="36793">的函数，用于为添加到</st> `<st c="36852">vectorstore</st>` <st c="36861">和</st> `<st c="36872">docstore</st>`<st c="36877">的文档生成唯一的文档 ID。</st>

+   `<st c="36886">HTML</st>` <st c="36891">和</st> `<st c="36896">显示</st>` <st c="36903">来自</st> `<st c="36909">IPython.display</st>`<st c="36924">: The</st> `<st c="36931">HTML</st>` <st c="36935">函数用于创建对象的 HTML 表示，而</st> `<st c="37004">显示</st>` <st c="37011">函数用于在 IPython 笔记本中显示对象。</st> <st c="37073">在我们的代码中，</st> `<st c="37086">HTML</st>` <st c="37090">和</st> `<st c="37095">显示</st>` <st c="37102">在</st> `<st c="37119">plt_img_base64</st>` <st c="37133">函数中用于显示</st> `<st c="37154">base64</st>`<st c="37160">编码的图像。</st>

+   `<st c="37177">Image</st>` <st c="37183">from PIL: PIL 提供打开、操作和保存各种图像</st> <st c="37269">文件格式的函数。</st>

+   `<st c="37282">matplotlib.pyplot as plt</st>`<st c="37307">: Matplotlib 是一个提供创建可视化图表函数的绘图库。</st> <st c="37406">在代码中，</st> `<st c="37419">plt</st>` <st c="37422">没有直接使用，但它可能被其他库</st> <st c="37494">或函数隐式使用。</st>

<st c="37507">这些导入的包</st> <st c="37531">和模块提供与文档加载、检索、存储、消息、图像处理和可视化相关的各种功能，这些功能在代码中用于处理和分析 PDF 文件，并生成对</st> <st c="37772">用户查询的响应。</st>

<st c="37785">在我们的导入之后，我们建立了几个在代码中使用的变量。</st> <st c="37881">这里有一些</st> <st c="37899">亮点</st><st c="37912">:</st>

+   **<st c="37914">GPT-4o-mini</st>**<st c="37925">: 我们将使用</st> <st c="37947">GPT-4o-mini，其中最后一个字符，</st> **<st c="37987">o</st>**<st c="37988">，代表</st> **<st c="38001">全功能</st>**<st c="38005">，这也可以说是多模态的另一种说法！</st>

    ```py
     llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    ```

+   **<st c="38111">PDF 的简短版本</st>**<st c="38132">: 注意我们现在使用的是不同的</st> <st c="38165">文件：</st>

    ```py
     short_pdf_path = "google-2023-environmental-report-short.pdf"
    ```

    <st c="38236">整个文件很大，使用整个文件会增加处理成本，而不会在演示方面提供太多价值。</st> <st c="38384">因此，我们鼓励您使用此文件，我们仍然可以演示 MM- RAG 应用程序，但与我们的 LLM 相比，推理成本会显著降低。</st> <st c="38526">。</st>

+   **<st c="38534">OpenAI 嵌入</st>**<st c="38552">：在使用 OpenAI 嵌入时，这里有一个需要注意的关键点，如下所示：</st> <st c="38623">正如所见：</st>

    ```py
     embedding_function = OpenAIEmbeddings()
    ```

<st c="38673">此嵌入模型不支持多模态嵌入，这意味着它不会将海鸥的图像嵌入为与文本单词 *<st c="38817">海鸥</st>* <st c="38824">非常相似，正如真正的多模态嵌入模型应该做的那样。</st> <st c="38871">为了克服这一不足，我们嵌入的是图像的描述而不是图像本身。</st> <st c="38976">这仍然被视为一种多模态方法，但请注意未来可能有助于我们在嵌入级别解决这一问题的多模态嵌入！</st> <st c="39118">。</st> <st c="39139">。</st>

<st c="39147">接下来，我们将使用 `<st c="39193">UnstructuredPDFLoader</st>` <st c="39214">文档加载器来加载 PDF：</st>

```py
 pdfloader = UnstructuredPDFLoader(
     short_pdf_path,
     mode="elements",
     strategy="hi_res",
     extract_image_block_types=["Image","Table"],
     extract_image_block_to_payload=True,
     # converts images to base64 format
)
pdf_data = pdfloader.load()
```

<st c="39465">在这里，我们使用 LangChain 和 `<st c="39521">非结构化</st>`<st c="39533">从 PDF 中提取元素。这需要一点时间，通常在 1-5 分钟之间，具体取决于您的开发环境有多强大。</st> <st c="39650">因此，这是一个休息和阅读有关使此包按需工作的参数的好时机</st> <st c="39767">！</st>

<st c="39775">让我们谈谈我们使用此文档加载器的参数以及它们如何为我们设置接下来的代码实验室：</st>

+   `<st c="39896">short_pdf_path</st>`<st c="39911">：这是代表我们之前定义的 PDF 文件较短版本的文件路径变量。</st>

+   `<st c="40021">mode="elements"</st>`<st c="40037">：此参数设置 `<st c="40086">UnstructuredPDFLoader</st>`<st c="40107"> 的提取模式。通过设置 `<st c="40120">mode="elements"</st>`<st c="40135">，加载器被指示从 PDF 文件中提取单个元素，例如文本块和图像。</st> <st c="40244">与其它模式相比，此模式允许对提取内容有更精细的控制。</st>

+   `<st c="40342">strategy="hi_res"</st>`<st c="40360">: 此参数指定用于从 PDF 文件中提取元素所使用的策略。</st> <st c="40454">其他选项包括</st> `<st c="40476">auto</st>`<st c="40480">,</st> `<st c="40482">fast</st>`<st c="40486">, 和</st> `<st c="40492">ocr_only</st>`<st c="40500">。其中</st> `<st c="40506">"hi_res</st>``<st c="40513">"</st>` <st c="40515">策略识别文档布局并使用它来获取有关文档元素的更多信息。</st> <st c="40631">如果您需要显著加快此过程，请尝试</st> `<st c="40686">fast</st>` <st c="40690">模式，但提取效果将远不如您从</st> `<st c="40774">"hi_res"</st>`<st c="40782">中看到的效果。我们鼓励您尝试所有设置以亲自查看差异</st> <st c="40851">。</st>

+   `<st c="40864">extract_image_block_types=["Image","Table"]</st>`<st c="40908">: `<st c="40915">extract_image_block_types</st>` <st c="40940">参数用于指定在处理图像块时作为</st> `<st c="41035">base64</st>`<st c="41041">编码数据存储在元数据字段中时需要提取的元素类型。</st> <st c="41083">此参数允许您在文档处理过程中针对图像中的特定元素进行定位。</st> <st c="41179">在此，我们针对图像</st> <st c="41202">和表格。</st>

+   `<st c="41213">extract_image_block_to_payload=True</st>`<st c="41249">: `<st c="41256">extract_image_block_to_payload</st>` <st c="41286">参数用于指定提取的图像块是否应包含在有效载荷中作为</st> `<st c="41388">base64</st>`<st c="41394">编码数据。</st> <st c="41410">此参数在处理文档并使用高分辨率策略提取图像块时相关。</st> <st c="41525">我们将它设置为</st> `<st c="41538">True</st>` <st c="41542">，这样我们实际上就不需要存储任何图像作为文件；加载器将提取的图像转换为</st> `<st c="41660">base64</st>` <st c="41666">格式，并将它们包含在相应元素的元数据中。</st>

<st c="41737">当此过程完成后，您将拥有所有从 PDF 加载到</st> `<st c="41824">pdf_data</st>`<st c="41832">中的数据。让我们添加一些代码来帮助我们探索这些已加载的数据：</st>

```py
 texts = [doc for doc in pdf_data if doc.metadata
    ["category"] == NarrativeText"]
images = [doc for doc in pdf_data if doc.metadata[
    "category"] == "Image"]
print(f"TOTAL DOCS USED BEFORE REDUCTION: texts:
    {len(texts)} images: {len(images)}")
categories = set(doc.metadata[
    "category"] for doc in pdf_data)
print(f"CATEGORIES REPRESENTED: {categories}")
```

<st c="42252">在此，我们挑选出代码实验室中最重要的两个元素类别，</st> `<st c="42335">'</st>``<st c="42336">NarrativeText'</st>` <st c="42350">和</st> `<st c="42355">'Image'</st>`<st c="42362">。我们使用列表推导式将这些元素拉入变量中，这些变量将仅包含</st> <st c="42440">这些元素。</st>

<st c="42455">我们即将减少图像数量以节省处理成本，因此我们打印出我们之前有多少个以确保它工作！</st> <st c="42595">我们还想看看数据中有多少种元素类型。</st> <st c="42667">以下是</st> <st c="42675">输出：</st>

```py
 TOTAL DOCS USED BEFORE REDUCTION: texts: 78 images: 17
CATEGORIES REPRESENTED: {'ListItem', 'Title', 'Footer', 'Image', 'Table', 'NarrativeText', 'FigureCaption', 'Header', 'UncategorizedText'}
```

<st c="42880">所以，现在我们有了</st> `<st c="42904">17</st>` <st c="42906">张图片。</st> <st c="42915">我们想减少这个演示的数量，因为我们即将使用 LLM 来总结每一张，三张图片的成本大约是 17 张的六分之一</st> <st c="43066">！</st>

<st c="43074">我们还看到，当我们只使用</st> `<st c="43168">'NarrativeText'</st>`<st c="43183">时，我们的数据中包含许多其他元素。</st> <st c="43259">如果我们想构建一个更健壮的应用程序，我们可以将</st> `<st c="43259">'Title'</st>`<st c="43266">,</st> `<st c="43268">'Footer'</st>`<st c="43276">,</st> `<st c="43278">'Header'</st>`<st c="43286">和其他元素纳入我们发送给 LLM 的上下文中，告诉它相应地强调这些元素。</st> <st c="43396">例如，我们可以告诉它更多地强调</st> `<st c="43449">'Title'</st>`<st c="43456">。这个</st> `<st c="43462">非结构化</st>` <st c="43474">库在以使它</st> <st c="43553">更适合 LLM 的方式提供我们的 PDF 数据方面做得非常出色！</st>

<st c="43571">OK – 所以，正如承诺的那样，我们将减少图像数量以节省你在</st> <st c="43663">处理上的费用：</st>

```py
 if len(images) > 3:
     images = images[:3]
print(f"total documents after reduction: texts:
    {len(texts)} images: {len(images)}")
```

<st c="43803">我们基本上只是切掉了前三张图片，并使用这个列表在</st> `<st c="43878">images</st>` <st c="43884">列表中使用。</st> <st c="43891">我们打印出来，看到我们已经减少到</st> <st c="43939">三张图片：</st>

```py
 total documents after reduction: texts: 78 images: 3
```

<st c="44005">接下来的几个代码块将专注于</st> <st c="44044">图像摘要，从我们将提示应用于图像并获取</st> <st c="44134">摘要的函数开始：</st>

```py
 def apply_prompt(img_base64):
    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
     These summaries will be embedded and used to retrieve the raw image. \
     Give a concise summary of the image that is well optimized for retrieval."""
     return [HumanMessage(content=[
           {"type": "text", "text": prompt},
           {"type": "image_url","image_url": {"url":
                 f"data:image/jpeg;base64,{img_base64}"},},
     ])]
```

<st c="44566">此函数接受一个</st> `<st c="44590">img_base64</st>` <st c="44600">参数，它表示图像的</st> `<st c="44633">base64</st>`<st c="44639">-编码字符串。</st> <st c="44669">函数首先定义一个包含字符串提示的提示变量，指示助手为了检索目的总结图像。</st> <st c="44824">该函数返回一个包含单个</st> `<st c="44872">HumanMessage</st>` <st c="44884">对象的列表，该对象表示图像的摘要。</st> <st c="44931">该</st> `<st c="44935">HumanMessage</st>` <st c="44947">对象有一个</st> `<st c="44961">content</st>` <st c="44968">参数，它是一个包含</st> <st c="45007">两个字典的列表：</st>

+   <st c="45024">第一个字典表示一个文本消息，其提示作为</st> <st c="45091">其值</st>

+   <st c="45100">第二个字典表示一个图像 URL 消息，其中</st> `<st c="45165">image_url</st>` <st c="45175">键包含一个字典，该字典的</st> `<st c="45211">url</st>` <st c="45214">键设置为以适当的 data URI 方案（</st>`<st c="45299">data:image/jpeg;base64</st>`<st c="45322">）为前缀的</st> `<st c="45230">base64</st>`<st c="45236">-编码的图像</st>

<st c="45324">记得当我们使用</st> `<st c="45345">extract_image_block_to_payload</st>` <st c="45375">将</st> `<st c="45379">True</st>` <st c="45383">设置为</st> `<st c="45399">UnstructuredPDFLoader</st>` <st c="45420">文档加载函数时吗？</st> <st c="45447">因此，我们的元数据中已经有了图像，以</st> `<st c="45476">base64</st>` <st c="45482">格式存在，所以我们只需要将这个传递给这个函数！</st> <st c="45568">如果你在其他应用程序中使用这种方法，并且有一个典型的图像文件，比如一个</st> `<st c="45656">.jpg</st>` <st c="45660">或</st> `<st c="45664">.png</st>` <st c="45668">文件，你只需要将其转换为</st> `<st c="45712">base64</st>` <st c="45718">即可使用此功能。</st>

<st c="45740">但是，对于这个应用程序来说，因为我们</st> <st c="45779">将图像提取为</st> `<st c="45806">base64</st>` <st c="45813">表示形式，LLM 与</st> `<st c="45863">base64</st>` <st c="45869">图像一起工作，而这个函数使用它作为参数，所以我们实际上不需要处理图像文件！</st> <st c="45977">你失望地发现你将看不到任何图像吗？</st> <st c="46032">不要失望！</st> <st c="46043">我们将使用之前讨论过的 HTML 函数创建一个辅助函数，将图像从它们的</st> `<st c="46169">base64</st>` <st c="46175">表示形式转换为 HTML 版本，这样我们就可以在我们的</st> `<st c="46234">笔记本</st>》中显示它们了！</st>

<st c="46247">但首先，我们准备我们的文本和图像，并设置列表以收集我们在运行我们</st> <st c="46357">刚刚讨论过的函数时生成的摘要：</st>

```py
 text_summaries = [doc.page_content for doc in texts]
# Store base64 encoded images, image summaries
img_base64_list = []
image_summaries = []
for img_doc in images:
     base64_image = img_doc.metadata["image_base64"]
     img_base64_list.append(base64_image)
     message = llm.invoke(apply_prompt(base64_image))
     image_summaries.append(message.content)
```

<st c="46711">请注意，我们不是在文本上运行摘要；我们只是直接将文本作为摘要。</st> <st c="46825">你也可以对文本进行摘要，这可能会提高检索结果，因为这是改进 RAG 检索的常见方法。</st> <st c="46961">然而，为了节省更多的 LLM 处理成本，我们在这里只专注于对图像进行摘要。</st> <st c="47063">你的钱包会</st> <st c="47080">感谢我们！</st>

<st c="47089">尽管如此，对于图像来说，这就足够了——你刚刚实现了多模态，在你的 LLM 使用中同时使用了文本和图像！</st> <st c="47199">我们目前还不能说我们使用了 MM-RAG，因为我们还没有以多模态的方式检索任何内容。</st> <st c="47296">但我们很快就会达到那里——让我们</st> <st c="47331">继续前进！</st>

<st c="47342">我们的数据准备已经结束；现在我们可以回到添加与 RAG 相关的元素，例如向量存储和检索器！</st> <st c="47368">在这里，我们设置了向量存储：</st>

```py
 vectorstore = Chroma(
     collection_name="mm_rag_google_environmental",
     embedding_function=embedding_function
)
```

<st c="47614">在这里，我们设置了一个新的集合名称，</st> `<st c="47654">mm_rag_google_environment</st>`<st c="47679">，这表明了该向量存储内容的多元模态性质。</st> <st c="47756">我们添加了我们的</st> `<st c="47767">embedding_function</st>` <st c="47785">链，该链将用于嵌入我们的内容，就像我们在代码实验室中多次看到的那样。</st> <st c="47893">然而，在过去，我们通常在设置向量存储时添加文档。</st> <st c="47978">设置它。</st>

<st c="47984">然而，在这种情况下，我们等待添加文档的时间不仅是在设置向量存储之后，也是在设置检索器之后！</st> <st c="48119">我们如何将它们添加到一个检索器中，这是一个检索文档的机制？</st> <st c="48193">嗯，就像我们过去说的那样，LangChain 中的检索器只是一个围绕向量存储的包装器，所以向量存储仍然在其中，我们可以通过检索器以类似的方式添加文档。</st> <st c="48387">以类似的方式。</st>

<st c="48403">但是首先，我们需要设置</st> <st c="48437">多向量检索器：</st>

```py
 store = InMemoryStore()
id_key = "doc_id"
retriever_multi_vector = MultiVectorRetriever(
     vectorstore=vectorstore,
     docstore=store,
     id_key=id_key,
)
```

<st c="48607">在这里，我们应用了这个</st> `<st c="48635">MultiVectorRetriever</st>` <st c="48655">包装器到我们的</st> `<st c="48671">vectorstore</st>` <st c="48682">向量存储。</st> <st c="48697">但是，这个其他元素，</st> `<st c="48729">InMemoryStore</st>`<st c="48742">，是什么？</st> `<st c="48748">InMemoryStore</st>` <st c="48761">元素是一个内存存储类，用于存储键值对。</st> <st c="48829">它用作</st> `<st c="48843">docstore</st>` <st c="48851">对象，用于存储与每个文档 ID 关联的实际文档内容。</st> <st c="48933">我们通过定义</st> `<st c="48966">id_key</st>` <st c="48972">与</st> `<st c="48982">doc_id</st>` <st c="48988">字符串来提供这些。</st>

<st c="48996">在这个阶段，我们将所有内容传递给</st> `<st c="49034">MultiVectorRetriever(...)</st>`<st c="49059">，这是一个结合多个向量存储并允许基于相似性搜索高效检索多种数据类型的检索器。</st> <st c="49196">我们已经多次看到了</st> `<st c="49213">vectorstore</st>` <st c="49224">向量存储，但正如你所见，你可以使用一个</st> `<st c="49284">docstore</st>` <st c="49292">对象来存储和检索文档内容。</st> <st c="49345">它被设置为</st> `<st c="49362">store</st>` <st c="49367">变量（一个</st> `<st c="49393">InMemoryStore</st>`<st c="49406">的实例）</st>，其中</st> `<st c="49418">id_key</st>` <st c="49424">字符串被设置为检索器中的</st> `<st c="49449">id_key</st>` <st c="49455">参数。</st> <st c="49484">这使得使用那个</st> `<st c="49597">id_key</st>` <st c="49603">字符串，就像在关系数据库中跨越两个存储的键值一样，轻松检索与向量存储中的向量相关的内容。</st>

<st c="49677">尽管如此，我们还没有在我们的任何存储中添加任何数据！</st> <st c="49738">让我们构建一个函数，这样我们就可以</st> <st c="49783">添加数据：</st>

```py
 def add_documents(retriever, doc_summaries, doc_contents):
     doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
     summary_docs = [
           Document(page_content=s, metadata={id_key:
               doc_ids[i]})
           for i, s in enumerate(doc_summaries)
     ]
     content_docs = [
           Document(page_content=doc.page_content,
           metadata={id_key: doc_ids[i]})
           for i, doc in enumerate(doc_contents)
     ]
     retriever.vectorstore.add_documents(summary_docs)
     retriever.docstore.mset(list(zip(doc_ids,
         doc_contents)))
```

<st c="50251">这个函数是一个辅助函数，它将文档添加到</st> `<st c="50314">vectorstore</st>` <st c="50325">向量存储和</st> `<st c="50343">docstore</st>` <st c="50351">检索器对象的</st> `<st c="50366">对象</st>` <st c="50375">。</st> <st c="50384">它接受</st> `<st c="50397">retriever</st>` <st c="50406">对象、</st> `<st c="50415">doc_summaries</st>` <st c="50428">列表和</st> `<st c="50439">doc_contents</st>` <st c="50451">列表作为参数。</st> <st c="50471">正如我们之前讨论的，我们为每个类别都有摘要和内容：我们将传递给</st> `<st c="50600">此函数</st>` <st c="50609">的文本和图像。</st>

<st c="50614">此函数使用</st> `<st c="50646">str(uuid.uuid4())</st>` <st c="50700">为每个文档生成唯一的</st> <st c="50720">文档 ID</st> <st c="50732">，然后通过遍历</st> `<st c="50760">doc_summaries</st>` <st c="50773">列表并创建具有摘要作为页面内容以及相应的文档 ID 作为元数据的</st> `<st c="50792">Document</st>` <st c="50800">对象来创建一个</st> `<st c="50720">summary_docs</st>` <st c="50732">列表。</st> <st c="50893">它还通过遍历</st> `<st c="50971">doc_contents</st>` <st c="50983">列表并创建具有文档内容作为页面内容以及相应的文档 ID 作为元数据的</st> `<st c="51002">Document</st>` <st c="51010">对象来创建一个</st> `<st c="50911">content_docs</st>` <st c="50923">列表。</st> <st c="50932">Document</st> <st c="50940">对象。</st> <st c="51112">它使用</st> `<st c="51124">summary_docs</st>` <st c="51136">列表通过</st> `<st c="51161">retriever.vectorstore.add_documents</st>` <st c="51172">函数将其添加到检索器的</st> `<st c="51196">vectorstore</st>` <st c="51231">向量存储中。</st> <st c="51242">它使用</st> `<st c="51254">retriever.docstore.mset</st>` <st c="51277">函数将</st> `<st c="51298">content_docs</st>` <st c="51310">列表添加到检索器的</st> `<st c="51335">docstore</st>` <st c="51343">对象中，将每个文档 ID 与其对应的</st> <st c="51404">文档内容关联起来。</st>

<st c="51421">我们仍然需要应用</st> `<st c="51449">add_document</st>` <st c="51461">函数：</st>

```py
 if text_summaries:
    add_documents(
         retriever_multi_vector, text_summaries, texts)
if image_summaries:
     add_documents(
        retriever_multi_vector, image_summaries, images)
```

<st c="51636">这将添加我们 MM-RAG 管道所需的相关文档和摘要，包括代表文本和</st> <st c="51789">图像摘要的嵌入向量。</st>

<st c="51805">接下来，我们将添加我们最终 MM-RAG 链中需要的最后一批辅助函数，首先是用于分割</st> `<st c="51928">base64</st>`<st c="51934">-编码的图像</st> <st c="51951">和文本的函数：</st>

```py
 def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            if doc.metadata.get("category") == "Image":
                base64_image = doc.metadata["image_base64"]
                b64_images.append(base64_image)
            else:
                     texts.append(doc.page_content)
        else:
                if isinstance(doc, str):
                    texts.append(doc)
     return {"images": b64_images, "texts": texts}
```

<st c="52321">此函数接收我们的图像相关文档列表作为输入，并将它们拆分为</st> `<st c="52403">base64</st>`<st c="52409">编码图像和文本。</st> <st c="52437">它初始化两个空列表：</st> `<st c="52469">b64_images</st>` <st c="52479">和</st> `<st c="52484">texts</st>`<st c="52489">。它遍历</st> `<st c="52533">docs</st>` <st c="52537">列表中的每个</st> `<st c="52513">doc</st>` <st c="52516">变量，检查它是否是</st> `<st c="52581">Document</st>` <st c="52589">类的实例。</st> <st c="52597">如果</st> `<st c="52604">doc</st>` <st c="52607">变量是一个</st> `<st c="52622">Document</st>` <st c="52630">对象并且其元数据有一个</st> `<st c="52661">category</st>` <st c="52669">键，其值为</st> `<st c="52689">Image</st>`<st c="52694">，它从</st> `<st c="52739">doc.metadata["image_base64"]</st>` <st c="52767">中提取</st> `<st c="52712">base64</st>`<st c="52718">编码的图像并将其添加到</st> `<st c="52790">b64_images</st>` <st c="52800">列表中。</st>

<st c="52806">如果</st> `<st c="52814">doc</st>` <st c="52817">变量是一个</st> `<st c="52832">Document</st>` <st c="52840">对象</st> <st c="52847">但没有</st> `<st c="52870">Image</st>` <st c="52875">类别，它将</st> `<st c="52897">doc.page_content</st>` <st c="52913">添加到</st> `<st c="52921">texts</st>` <st c="52926">列表中。</st> <st c="52933">如果</st> `<st c="52940">doc</st>` <st c="52943">变量不是一个</st> `<st c="52962">Document</st>` <st c="52970">对象但是一个字符串，它将</st> `<st c="53010">doc</st>` <st c="53013">变量添加到</st> `<st c="53030">texts</st>` <st c="53035">列表中。</st> <st c="53042">最后，该函数返回一个包含两个键的字典：</st> `<st c="53100">"images"</st>`<st c="53108">，包含一个包含</st> `<st c="53131">base64</st>`<st c="53137">编码图像的列表，以及</st> `<st c="53159">"texts"</st>`<st c="53166">，包含一个文本列表。</st>

<st c="53195">我们还有一个函数用于</st> <st c="53222">生成我们的图像</st> <st c="53242">提示信息：</st>

```py
 def img_prompt_func(data_dict):
    formatted_texts = "\n".join(
        data_dict["context"]["texts"])
    messages = []
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;
                     base64,{image}"}}
                messages.append(image_message)
     text_message = {
           "type": "text",
           "text": (
                 f"""You are are a helpful assistant tasked with describing what is in an image. The user will ask for a picture of something. Provide text that supports what was asked for. Use this information to provide an in-depth description of the aesthetics of the image. Be clear and concise and don't offer any additional commentary. User-provided question: {data_dict['question']}
Text and / or images: {formatted_texts}"""
           ),
     }
     messages.append(text_message)
     return [HumanMessage(content=messages)]
```

<st c="54104">此函数接收</st> `<st c="54125">data_dict</st>` <st c="54134">作为输入</st> <st c="54143">并生成一个用于图像分析的提示信息。</st> <st c="54195">它从</st> `<st c="54218">data_dict["context"]</st>` <st c="54238">中提取文本，并使用</st> `<st c="54299">"\n".join</st>` <st c="54308">将它们合并成一个字符串，</st> `<st c="54276">formatted_texts</st>`<st c="54291">。它初始化一个名为</st> `<st c="54339">messages</st>`<st c="54346">的空列表。</st>

<st c="54355">如果</st> `<st c="54359">data_dict["context"]["images"]</st>` <st c="54389">存在，它将遍历列表中的每个图像。</st> <st c="54439">对于每个图像，它创建一个包含</st> `<st c="54469">image_message</st>` <st c="54482">字典，其中有一个</st> `<st c="54501">"type"</st>` <st c="54507">键设置为</st> `<st c="54519">"image_url"</st>` <st c="54530">和一个</st> `<st c="54538">"image_url"</st>` <st c="54549">键包含一个包含</st> `<st c="54587">base64</st>`<st c="54593">编码的图像 URL 的字典。</st> <st c="54614">它将每个</st> `<st c="54630">image_message</st>` <st c="54643">实例追加到</st> `<st c="54660">messages</st>` <st c="54668">列表中。</st>

<st c="54674">现在，最后的润色——在我们运行我们的 MM-RAG 应用程序之前，我们建立了一个 MM-RAG 链，包括使用我们刚刚</st> <st c="54709">设置的</st> <st c="54817">两个函数：</st>

```py
 chain_multimodal_rag = ({"context": retriever_multi_vector
    | RunnableLambda(split_image_text_types),
    "question": RunnablePassthrough()}
    | RunnableLambda(img_prompt_func)
    | llm
    | str_output_parser
)
```

<st c="55022">这创建了我们自己的 MM-RAG 链，它由以下</st> `<st c="55076">组件组成：</st>`

+   `<st c="55097">{"context": retriever_multi_vector | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}</st>`<st c="55209">: 这与其他我们过去看到的检索组件类似，提供了一个包含两个键的字典：</st> `<st c="55320">"context"</st>` <st c="55329">和</st> `<st c="55334">"question"</st>`<st c="55344">。键“</st> `<st c="55350">"context"</st>` <st c="55359">”被分配给</st> `<st c="55390">retriever_multi_vector | RunnableLambda(split_image_text_types)</st>`<st c="55453">的结果。函数</st> `<st c="55459">retriever_multi_vector</st>` <st c="55481">根据问题检索相关文档，然后这些结果通过</st> `<st c="55585">RunnableLambda(split_image_text_types)</st>`<st c="55623">传递，这是一个围绕</st> `<st c="55655">split_image_text_types</st>` <st c="55677">函数的包装器。</st> <st c="55688">正如我们之前讨论的，函数</st> `<st c="55720">split_image_text_types</st>` <st c="55742">将检索到的文档分割成</st> `<st c="55788">base64</st>`<st c="55794">编码的图像和文本。</st> <st c="55822">键“</st> `<st c="55826">"question"</st>` <st c="55836">”被分配给</st> `<st c="55853">RunnablePassthrough</st>`<st c="55872">，它简单地传递问题而不进行</st> `<st c="55923">任何修改</st>`。</st>

+   `<st c="55940">RunnableLambda(img_prompt_func)</st>`<st c="55972">: 上一个组件（分割图像和文本以及问题）的输出通过</st> `<st c="56083">RunnableLambda(img_prompt_func)</st>`<st c="56114">传递。正如我们之前讨论的，</st> `<st c="56148">img_prompt_func</st>` <st c="56163">函数根据检索到的上下文和问题生成一个用于图像分析的提示信息，因此这就是我们将格式化并传递到下一个</st> `<st c="56330">步骤</st>` `<st c="56336">llm</st>`<st c="56339">的内容。</st>

+   `<st c="56340">llm</st>`<st c="56344">：生成的提示</st> <st c="56367">信息，其中包含一个以</st> `<st c="56404">base64</st>` <st c="56410">格式编码的图像，传递给我们的 LLM 进行处理。</st> <st c="56456">LLM 根据多模态提示信息生成响应，然后将其传递到下一步：输出解析器。</st>

+   `<st c="56580">str_output_parser</st>`<st c="56598">：我们在代码实验室中看到了输出解析器，这是我们在过去表现良好的相同可靠的</st> `<st c="56683">StrOutputParser</st>` <st c="56698">类，它将生成的响应解析为</st> <st c="56776">一个字符串。</st>

<st c="56785">总的来说，这个链代表了一个 MM-RAG 管道，它检索相关文档，将它们分成图像和文本，生成一个提示信息，用 LLM 处理它，并将输出解析为</st> <st c="56985">一个字符串。</st>

<st c="56994">我们调用此链并实现完整的</st> <st c="57035">多模态检索：</st>

```py
 user_query = "Picture of multiple wind turbines in the ocean." chain_multimodal_rag.invoke(user_query)
```

<st c="57160">请注意，我们使用了一个与过去不同的</st> `<st c="57196">user_query</st>` <st c="57206">字符串。</st> <st c="57245">我们将其更改为与我们可用的图像相关的内容。</st>

<st c="57331">以下是我们的 MM-RAG 管道基于此</st> <st c="57390">用户查询</st>的输出：

```py
 'The image shows a vast array of wind turbines situated in the ocean, extending towards the horizon. The turbines are evenly spaced and stand tall above the water, with their large blades capturing the wind to generate clean energy. The ocean is calm and blue, providing a serene backdrop to the white turbines. The sky above is clear with a few scattered clouds, adding to the tranquil and expansive feel of the scene. The overall aesthetic is one of modernity and sustainability, highlighting the use of renewable energy sources in a natural setting.'
```

<st c="57955">响应与</st> `<st c="57985">user_query</st>` <st c="57995">字符串一致，以及我们用来向 LLM 解释如何描述它“看到”的图像的提示。由于我们只有三张图像，因此很容易找到</st> <st c="58157">正在讨论的这张图像，图像</st> *<st c="58198">#2</st>*<st c="58200">，我们可以用这个来检索</st> <st c="58224">：</st>

```py
 def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,
        {img_base64}" />'
    display(HTML(image_html))
plt_img_base64(img_base64_list[1])
```

<st c="58394">这里的函数是我们承诺的辅助函数，帮助你查看图像。</st> <st c="58475">它接受一个</st> `<st c="58486">base64</st>`<st c="58492">编码的图像，</st> `<st c="58509">img_base64</st>`<st c="58519">，作为输入，并使用 HTML 显示它。</st> <st c="58558">它是通过创建一个</st> `<st c="58586">image_html</st>` <st c="58596">HTML 字符串来表示一个</st> `<st c="58628"><img></st>` <st c="58633">标签，其</st> `<st c="58647">src</st>` <st c="58650">属性设置为</st> `<st c="58672">base64</st>`<st c="58678">编码的图像 URL。</st> <st c="58699">它使用</st> `<st c="58711">display()</st>` <st c="58720">函数从 IPython 渲染 HTML 字符串并显示图像。</st> <st c="58792">在你的代码实验室中运行此代码，你将看到从 PDF 中提取的图像，为 MM-RAG 响应提供基础！</st>

<st c="58923">仅作参考，以下是为此图像生成的图像摘要，使用与</st> `<st c="59039">img_base64_list</st>` <st c="59054">列表相同的索引，因为它们匹配：</st>

```py
 image_summaries[1]
```

<st c="59096">摘要应该看起来像这样：</st> <st c="59131">：</st>

```py
 'Offshore wind farm with multiple wind turbines in the ocean, text "What\'s inside" on the left side.'
```

<st c="59244">鉴于来自 MM-RAG 链的输出描述，它对此图像的描述更加稳健和详细，您可以看到 LLM 实际上“看到”了这张图像，并告诉您关于它的情况。</st> <st c="59435">您现在是</st> <st c="59443">官方的多模态了！</st>

<st c="59466">我们选择本章中的三个代码实验室，因为我们认为它们代表了大多数 RAG 应用中潜在改进的最广泛代表性。</st> <st c="59632">但这些都是您特定 RAG 需要可能适用的技术冰山一角。</st> <st c="59748">在</st> <st c="59754">下一节中，我们提供了我们认为只是许多您应该考虑将其纳入您的</st> <st c="59883">RAG 管道的技术之始。</st>

# <st c="59896">其他要探索的先进 RAG 技术</st>

<st c="59937">正如我们之前与 RAG 和 GenAI 讨论的大多数其他事情一样，可用于应用于您的 RAG 应用的高级技术的选项太多，无法一一列出或跟踪。</st> <st c="60138">我们已选择专注于 RAG 特定方面的技术，根据它们将在您的 RAG 应用中产生最大影响的领域对它们进行分类。</st>

<st c="60314">让我们按照我们的 RAG 管道操作相同的顺序来探讨它们，从</st> <st c="60398">索引开始。</st>

## <st c="60412">索引改进</st>

<st c="60434">这些是专注于 RAG 管道索引阶段的先进 RAG 技术：</st>

+   **<st c="60522">深度分块</st>**<st c="60536">：检索结果的质量通常取决于您在检索系统本身存储之前如何分块您的数据。</st> <st c="60644">使用深度分块，您使用深度学习模型，包括变压器，进行最优和</st> <st c="60748">智能分块。</st>

+   **<st c="60769">训练和使用嵌入适配器</st>**<st c="60811">：嵌入适配器是轻量级模块，经过训练以适应预存在的语言模型嵌入，用于特定任务或领域，而无需大量重新训练。</st> <st c="60986">当应用于 RAG 系统时，这些适配器可以调整模型的理解和生成能力，以更好地与提示的细微差别相匹配，从而促进更准确和</st> <st c="61174">相关的检索。</st>

+   **<st c="61194">多表示索引</st>**<st c="61224">：命题索引使用 LLM 生成针对检索优化的文档摘要（命题）。</st>

+   **<st c="61338">用于树状检索的递归抽象处理（RAPTOR）</st>**<st c="61409">：RAG 系统需要处理“低级”问题，这些问题引用单个</st> <st c="61510">文档中找到的特定事实，或者“高级”问题，这些问题提炼了跨越许多文档的思想。</st> <st c="61593">在典型的 kNN 检索中，只能检索有限数量的文档块，处理这两种类型的问题可能是一个挑战。</st> <st c="61728">RAPTOR 通过创建文档摘要来解决这个问题，这些摘要捕获了高级概念。</st> <st c="61817">它嵌入并聚类文档，然后总结每个聚类。</st> <st c="61884">它通过递归地这样做，产生了一个具有越来越高级概念的摘要树。</st> <st c="61979">摘要和起始文档一起索引，覆盖了</st> <st c="62056">用户问题。</st>

+   **<st c="62071">BERT 上的上下文化后期交互（ColBERT）</st>**<st c="62123">：嵌入模型将文本压缩成</st> <st c="62162">固定长度（向量）表示，这些表示捕获了文档的语义内容。</st> <st c="62240">这种压缩对于高效的搜索检索非常有用，但给单个向量表示带来了巨大的负担，以捕捉文档的所有语义细微差别和细节。</st> <st c="62439">在某些情况下，无关或冗余的内容可能会稀释嵌入的语义有用性。</st> <st c="62539">ColBERT 通过使用更高粒度的嵌入来解决这个问题，专注于在文档和</st> <st c="62709">查询之间产生更细粒度的标记相似度评估。</st>

## <st c="62719">检索</st>

<st c="62729">检索是我们最大的高级 RAG 技术类别，反映了检索在 RAG 过程中的重要性。</st> <st c="62767">以下是我们在您的</st> <st c="62851">RAG 应用中推荐您考虑的一些方法：</st> <st c="62911">：</st>

+   **<st c="62927">假设文档嵌入（HyDE）</st>**<st c="62967">：HyDE 是一种检索方法，通过</st> <st c="63025">为传入的查询生成一个假设文档来增强检索。</st> <st c="63084">这些文档来自 LLM 的知识库，被嵌入并用于从索引中检索文档。</st> <st c="63137">其理念是假设文档可能比原始的</st> <st c="63288">用户问题与索引文档更匹配。</st>

+   **<st c="63302">句子窗口检索</st>**<st c="63328">：在句子窗口检索中，您基于更小的句子进行检索，以</st> <st c="63414">更好地匹配相关上下文，然后基于句子周围的扩展上下文窗口进行综合。</st>

+   **<st c="63525">自动合并检索</st>**<st c="63548">：自动合并检索解决了你在简单的 RAG 中看到的问题，即拥有较小的块可能会导致我们的数据碎片化。<st c="63642">它使用自动合并启发式方法将较小的块合并到一个更大的父块中，以确保更</st> <st c="63787">连贯的上下文。</st>

+   **<st c="63804">多查询重写</st>**<st c="63826">：多查询是一种从</st> <st c="63844">多个角度重写问题的方法，对每个重写的问题进行检索，并取所有文档的唯一并集。</st>

+   **<st c="64000">查询翻译回溯步骤</st>**<st c="64028">：回溯提示是一种基于 CoT</st> <st c="64105">推理来改进检索的方法。从一个问题出发，它生成一个回溯（更高层次、更抽象）的问题，该问题可以作为正确回答原始问题的先决条件。<st c="64277">这在需要背景知识或更基本的理解来回答特定问题时特别有用。</st>

+   **<st c="64417">查询结构化</st>**<st c="64435">：查询结构化是将文本转换为 DSL 的过程，其中 DSL 是用于与特定数据库交互的领域特定语言。<st c="64526">这将用户问题转换为</st> <st c="64605">结构化查询。</st>

## <st c="64624">检索/生成后</st>

<st c="64650">这些是高级 RAG</st> <st c="64673">技术，专注于 RAG 管道的生成阶段：</st>

+   **<st c="64740">交叉编码重新排名</st>**<st c="64765">：我们已经在我们的混合 RAG 代码实验室中看到了重新排名可以带来的改进，它应用于检索结果在发送到</st> <st c="64927">LLM 之前。交叉编码重新排名通过使用一个计算量更大的模型来重新评估和重新排序检索到的文档，基于它们与原始提示的相关性，从而进一步利用这一技术。<st c="64936">这种细粒度分析确保了最相关的信息在生成阶段得到优先考虑，从而提高了整体</st> <st c="65284">输出质量。</st>

+   **<st c="65299">RAG-fusion 查询重写</st>**<st c="65326">：RAG-fusion 是一种从多个角度重写问题的方法，对每个重写的问题进行检索，并对每个检索的结果进行相互排名融合，从而得到一个</st> <st c="65536">综合排名。</st>

## <st c="65557">整个 RAG 管道覆盖</st>

<st c="65586">这些高级 RAG 技术专注于整个 RAG 管道，而不仅仅是其中的一个阶段：</st>

+   **<st c="65691">自反式 RAG</st>**<st c="65711">：结合 LangGraph 技术的自反式 RAG 通过引入与 LangGraph 的语用图结构相结合的自反机制，改进了简单的 RAG 模型。</st> <st c="65895">在此方法中，LangGraph 有助于在更深层次上理解上下文和语义，使 RAG 系统能够根据对内容和其相互关系的更细致理解来优化其</st> <st c="66028">响应。</st> <st c="66118">这在内容创作、问答和对话代理等应用中尤其</st> <st c="66142">有用，因为它导致更准确、相关和上下文感知的输出，显著提高了生成文本的质量。</st>

+   **<st c="66360">模块化 RAG</st>**<st c="66372">：模块化 RAG 使用可互换的组件来提供更灵活的架构</st> <st c="66458">，以满足您的 RAG 开发需求。</st> <st c="66506">这种模块化使得研究人员和开发者能够尝试不同的检索机制、生成模型和优化策略，根据特定需求和应用程序定制 RAG 系统。</st> <st c="66717">正如您在本书的代码实验室中所见，LangChain 提供了支持这种方法的机制，在许多情况下，LLMs、检索器、向量存储和其他组件可以轻松替换和切换。</st> <st c="66948">模块化 RAG 的目标是朝着更可定制、更高效、更强大的 RAG 系统迈进，能够以更高的效率处理更广泛的任务。</st>

<st c="67111">随着每天都有新的研究成果出现，这项技术列表正在迅速增长。</st> <st c="67130">新技术的绝佳来源是 Arxiv.org 网站：</st> [<st c="67254">https://arxiv.org/</st>](https://arxiv.org/)<st c="67272">。</st>

<st c="67273">访问此网站并搜索与您的 RAG 应用相关的各种关键词，包括</st> *<st c="67369">RAG</st>**<st c="67372">、**<st c="67374">检索增强生成</st>**<st c="67404">、**<st c="67406">向量搜索</st>**<st c="67419">以及其他</st> <st c="67431">相关术语。</st>

# <st c="67445">总结</st>

<st c="67453">在本章的最后一部分，我们探讨了多种提高 RAG 应用的高级技术，包括查询扩展、查询分解和 MM-RAG。</st> <st c="67606">这些技术通过增强查询、将问题分解为子问题以及结合多种数据模态来提升检索和生成能力。</st> <st c="67765">我们还讨论了一系列其他高级 RAG 技术，涵盖了索引、检索、生成以及整个</st> <st c="67881">RAG 流程。</st>

<st c="67894">与您一同踏上 RAG 之旅，探索 RAG 的世界及其巨大的潜力，这是一件令人愉快的事情。</st> <st c="68005">随着我们结束这本书的撰写，我希望您已经具备了足够的知识和实践经验来应对自己的 RAG 项目。</st> <st c="68138">祝您在未来的 RAG 探索中好运——我坚信您将创造出令人瞩目的应用，推动这一激动人心的</st> <st c="68294">新技术所能达到的边界！</st>
