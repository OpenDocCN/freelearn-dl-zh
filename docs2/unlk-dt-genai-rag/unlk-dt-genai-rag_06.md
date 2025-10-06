# <st c="0">6</st>

# <st c="2">与 RAG 和 Gradio 交互</st>

<st c="33">在几乎所有情况下，**<st c="55">检索增强生成**</st> <st c="85">(**<st c="87">RAG</st>**) 开发涉及创建一个或多个应用程序，或**<st c="159">应用**</st> <st c="163">（简称 apps）。</st> <st c="175">在最初编码 RAG 应用程序时，你通常会创建一个变量，在代码中表示提示或其他类型的输入，这些输入反过来又表示 RAG 管道将基于其进行操作。</st> <st c="370">但未来的用户会怎样使用你正在构建的应用程序呢？</st> <st c="434">你如何使用你的代码与这些用户进行测试？</st> <st c="489">你需要一个界面！</st>

<st c="511">在本章中，我们将提供使用 Gradio 作为用户界面（**<st c="617">Gradio</st>**）的实用指南，以使您的应用程序通过 RAG 交互。</st> <st c="651">它涵盖了设置 Gradio 环境、集成 RAG 模型、创建用户友好的界面，使用户能够像使用典型的网络应用程序一样使用您的 RAG 系统，并在永久且免费的在线空间中托管它。</st> <st c="881">您将学习如何快速原型设计和部署 RAG 驱动的应用程序，使最终用户能够实时与 AI 模型交互。</st>

<st c="1017">关于如何构建界面的书籍已经有很多，你可以在很多地方提供界面，例如在网页浏览器或通过移动应用。</st> <st c="1185">但幸运的是，使用 Gradio，我们可以为您提供一种简单的方法来为您的基于 RAG 的应用程序提供界面，而无需进行大量的网页或移动开发。</st> <st c="1352">这使得共享和演示模型变得更加容易。</st> <st c="1386">模型。</st>

<st c="1398">在本章中，我们将具体介绍以下主题：</st> <st c="1446">以下主题：</st>

+   <st c="1463">为什么选择 Gradio？</st>

+   <st c="1475">使用 Gradio 的好处</st>

+   <st c="1500">使用 Gradio 的局限性</st>

+   <st c="1528">代码实验室 – 添加一个</st> <st c="1549">Gradio 界面</st>

<st c="1565">让我们首先讨论为什么 Gradio 是您 RAG 开发工作的重要组成部分。</st> <st c="1630">努力。</st>

# <st c="1650">技术要求</st>

<st c="1673">本章的代码在此：</st> <st c="1703">这里：</st> [<st c="1709">https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_06</st>](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_06 )

# <st c="1488">为什么选择 Gradio？</st>

<st c="1818">到目前为止，我们一直</st> <st c="1847">关注的是通常被归入</st> <st c="1902">数据科学领域的主题。</st> **<st c="1925">机器学习</st>**<st c="1941">、**<st c="1943">自然语言处理</st>** <st c="1970">(**<st c="1972">NLP</st>**<st c="1975">)、**<st c="1979">生成式人工智能</st>** <st c="2013">(**<st c="2015">生成式 AI</st>**<st c="2028">)、**<st c="2032">大型语言模型</st>** <st c="2053">(**<st c="2055">LLMs</st>**<st c="2059">)和 RAG 都是需要大量**<st c="2113">专业知识**的技术，通常需要投入足够多的**<st c="2148">时间**，以至于我们无法在其他技术领域，如使用网络技术和构建网络前端，建立专业知识。</st> **<st c="2285">Web 开发</st>** <st c="2300">本身是一个高度技术化的领域，需要大量的**<st c="2382">经验和专业知识**才能**<st c="2400">成功实施**。</st>

<st c="3050">然而，对于 RAG 来说，拥有一个 UI 非常有帮助，特别是如果你想测试它或向潜在用户展示它。</st> <st c="3205">如果我们没有时间学习**<st c="2552">网络开发</st>**，我们该如何提供这样的 UI 呢？</st>

<st c="2628">这就是为什么许多数据科学家，包括我自己，都使用**<st c="2703">Gradio</st>**的原因。它允许你非常快速地（相对于构建网络前端）以可共享的格式启动一个用户界面，甚至还有一些基本的身份验证功能。</st> <st c="2712">这不会让任何网络开发者失业，因为如果你想要将你的 RAG 应用变成一个完整的、健壮的网站，Gradio 可能不是一个很好的选择。</st> <st c="2875">但它将允许你，作为一个时间非常有限来构建网站的人，在几分钟内就启动一个非常适合 RAG 应用的 UI！</st>

<st c="3220">因为这里的想法是让你将大部分精力集中在 RAG 开发上，而不是网络开发上，我们将简化我们对 Gradio 的讨论，只讨论那些能帮助你将 RAG 应用部署到网络并使其可共享的组件。</st> <st c="3483">然而，随着你的 RAG 开发继续进行，我们鼓励你进一步调查 Gradio 的功能，看看是否还有其他什么可以帮助你**<st c="3651">特定努力**的地方！</st>

<st c="3668">考虑到这一点，让我们来谈谈使用 Gradio 构建**<st c="3749">RAG 应用**的主要好处。</st>

# <st c="3765">使用 Gradio 的好处</st>

除了对非网页开发者来说非常容易使用之外，Gradio 还有很多优点。<st c="3790">除了</st> <st c="3799">仅仅对非网页开发者来说非常容易使用之外，Gradio 还有很多优点。</st> <st c="3881">Gradio 的核心库是开源的，这意味着开发者可以自由地使用、修改并为项目做出贡献。</st> <st c="3997">Gradio 与广泛使用的机器学习框架集成良好，例如</st> <st c="4071">如下</st> **<st c="4074">TensorFlow</st>**<st c="4084">，**<st c="4086">PyTorch</st>**<st c="4093">，和</st> **<st c="4099">Keras</st>**<st c="4104">。除了开源库之外，Gradio 还提供了一个托管平台，开发者可以在该平台上部署他们的模型接口并管理访问权限。</st> <st c="4109">此外，Gradio 还包含一些有助于机器学习项目团队协作的功能，例如共享接口和</st> <st c="4384">收集反馈。</st>

<st c="4404">Gradio 的另一个令人兴奋的功能是它与**<st c="4472">Hugging Face</st>**<st c="4484">集成得很好。</st> 由 OpenAI 的前员工创立的 Hugging Face 拥有许多旨在支持生成式 AI 社区的资源，例如模型共享和数据集托管。</st> <st c="4650">这些资源之一是能够使用**<st c="4756">Hugging Face Spaces</st>**<st c="4775">在互联网上设置指向您的 Gradio 演示的永久链接。</st> Hugging Face Spaces 提供了免费永久托管您的机器学习模型的必要基础设施！<st c="4828">查看 Hugging Face 网站以了解更多关于</st> <st c="4946">他们的 Spaces 的信息。</st>

<st c="4959">当使用 Gradio 为您的 RAG 应用程序时，也存在一些限制，了解这些限制是很重要的。</st>

# <st c="5102">使用 Gradio 的限制</st>

<st c="5130">在使用 Gradio 时，最重要的是要记住的是，它并不提供足够的支持来构建一个将与其他数百、数千甚至数百万用户交互的生产级应用程序。</st> <st c="5352">在这种情况下，您可能需要雇佣一位在构建大规模生产级应用程序前端方面有专业知识的人。</st> <st c="5484">但对我们所说的**<st c="5507">概念验证</st>** <st c="5523">(</st>**<st c="5525">POC</st>**<st c="5528">)类型的应用程序，或者构建允许您测试具有基本交互性和功能的应用程序，Gradio 做得非常出色。</st> <st c="5672">Gradio 做得非常出色。</st>

<st c="5686">当你使用 Gradio 进行 RAG 应用时可能会遇到的一个限制是，你所能构建的内容缺乏灵活性。</st> <st c="5820">对于许多 RAG 应用，尤其是在构建原型时，这不会成为问题。</st> <st c="5906">但如果你或你的用户开始要求更复杂的 UI 功能，Gradio 将比完整的 Web 开发框架限制得多。</st> <st c="6059">不仅了解这一点对你很重要，而且与你的用户设定这些期望也很重要，帮助他们理解这只是一个简单的</st> <st c="6229">演示</st> <st c="6233">应用程序。</st>

<st c="6246">让我们直接进入代码，了解 Gradio 如何为你的 RAG 应用程序提供它应得的界面。</st> <st c="6350">它应得的。</st>

# <st c="6362">代码实验室 – 添加 Gradio 接口</st>

<st c="6399">此代码从我们</st> <st c="6434">在</st> *<st c="6446">第五章</st>*<st c="6455">停止的地方继续，除了最后一组代表提示探针攻击的行。</st> <st c="6528">正如我们在所有代码实验室的开始一样，我们将从安装一个新的包开始，当然是 Gradio！</st> <st c="6663">我们还将卸载</st> `<st c="6694">uvloop</st>`<st c="6700">，因为它与我们的</st> <st c="6733">其他包</st>存在冲突：</st>

```py
 %pip install gradio
%pip uninstall uvloop -y
```

<st c="6793">这会安装</st> `<st c="6812">gradio</st>` <st c="6818">包并移除冲突的</st> `<st c="6855">uvloop</st>` <st c="6861">包。</st>

<st c="6870">接下来，我们将向导入列表中添加多个包：</st> <st c="6919">导入：</st>

```py
 import asyncio
import nest_asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()
import gradio as gr
```

<st c="7070">这些行导入</st> `<st c="7094">asyncio</st>` <st c="7101">和</st> `<st c="7106">nest_asyncio</st>` <st c="7118">库，并设置事件循环策略。</st> `<st c="7163">asyncio</st>` <st c="7170">是一个用于使用协程和事件循环编写并发代码的库。</st> `<st c="7246">nest_asyncio</st>` <st c="7258">是一个允许 Jupyter 笔记本中嵌套事件循环的库。</st> <st c="7324">`<st c="7328">asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())</st>` <st c="7391">行将事件循环策略设置为默认策略。</st> `<st c="7447">nest_asyncio.apply()</st>` <st c="7467">应用必要的补丁以启用嵌套事件循环。</st> <st c="7528">然后，最后，我们导入</st> `<st c="7556">gradio</st>` <st c="7562">包，并将其分配给</st> `<st c="7589">gr</st>` <st c="7591">别名</st> <st c="7598">以方便使用。</st>

<st c="7614">在添加导入之后，我们</st> <st c="7644">只需要在现有代码的末尾添加此代码来设置我们的</st> <st c="7717">Gradio 接口：</st>

```py
 def process_question(question):
    result = rag_chain_with_source.invoke(question)
    relevance_score = result['answer']['relevance_score']
    final_answer = result['answer']['final_answer']
    sources = [doc.metadata['source'] for doc in result['context']]
    source_list = ", ".join(sources)
    return relevance_score, final_answer, source_list
```

<st c="8063">该</st> `<st c="8068">process_question</st>` <st c="8084">函数是在你点击那个</st> `<st c="8198">gr.Interface</st>` <st c="8210">代码时被调用的函数，但这是被调用并处理的函数。</st> <st c="8272">该</st> `<st c="8276">process_question</st>` <st c="8292">函数接收用户提交的问题作为输入，并使用我们的 RAG 管道进行处理。</st> <st c="8392">它调用</st> `<st c="8407">rag_chain_with_source</st>` <st c="8428">对象，并使用给定的问题检索相关性得分、最终答案和来源。</st> <st c="8538">然后该函数将来源合并成一个以逗号分隔的字符串，并返回相关性得分、最终答案和</st> <st c="8655">来源列表。</st>

<st c="8667">接下来，我们将设置一个</st> <st c="8708">Gradio 界面</st>的实例：

```py
 demo = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(label="Enter your question",
        value="What are the Advantages of using RAG?"),
    outputs=[
        gr.Textbox(label="Relevance Score"),
        gr.Textbox(label="Final Answer"),
        gr.Textbox(label="Sources")
    ],
    title="RAG Question Answering",
    description=" Enter a question about RAG and get an answer, a 
        relevancy score, and sources." )
```

<st c="9101">该</st> `<st c="9105">demo = gr.Interface(...)</st>` <st c="9129">行是 Gradio 魔法发生的地方。</st> <st c="9170">它使用</st> `<st c="9210">gr.Interface</st>` <st c="9222">函数创建一个 Gradio 界面。</st> <st c="9233">该</st> `<st c="9237">fn</st>` <st c="9239">参数指定当用户与界面交互时调用的函数，这正是我们在上一段中提到的，调用</st> `<st c="9388">process_question</st>` <st c="9404">并启动 RAG 管道。</st> <st c="9439">该</st> `<st c="9443">inputs</st>` <st c="9449">参数定义了界面的输入组件，用于输入问题的是</st> `<st c="9515">gr.Textbox</st>` <st c="9525">。</st> <st c="9553">该</st> `<st c="9557">outputs</st>` <st c="9564">参数定义了界面的输出组件，有三个</st> `<st c="9639">gr.Textbox</st>` <st c="9649">组件用于显示相关性得分、最终答案和来源。</st> <st c="9724">该</st> `<st c="9728">title</st>` <st c="9733">和</st> `<st c="9738">description</st>` <st c="9749">参数设置了界面的标题和描述。</st>

<st c="9808">剩下的唯一行动就是启动</st> <st c="9851">界面：</st>

```py
 demo.launch(share=True, debug=True)
```

<st c="9901">这条</st> `<st c="9906">demo.launch(share=True, debug=True)</st>` <st c="9941">行启动了 Gradio 界面。</st> <st c="9978">这个</st> `<st c="9982">share=True</st>` <st c="9992">参数启用了 Gradio 的共享功能，生成一个公开可访问的 URL，你可以与他人分享以访问你的界面。</st> <st c="10136">Gradio 使用隧道服务来提供此功能，允许任何拥有 URL 的人与你的界面交互，而无需在本地运行代码。</st> <st c="10301">这个</st> `<st c="10305">debug=True</st>` <st c="10315">参数启用了调试模式，提供了额外的信息和工具，用于调试和开发。</st> <st c="10420">在调试模式下，如果执行过程中发生错误，Gradio 会在浏览器控制台中显示详细的错误消息。</st>

<st c="10572">我认为</st> `<st c="10584">demo.launch(share=True, debug=True)</st>` <st c="10619">是这本书中所有其他代码中的一条特殊代码行。</st> <st c="10712">这是因为它做了一些你以前没有看到的事情；它调用 Gradio 来启动一个本地 Web 服务器来托管由</st> `<st c="10851">gr.Interface(...)</st>`<st c="10868">定义的界面。当你运行这个单元格时，你会注意到它会持续运行，直到你停止它。</st> <st c="10967">你还会注意到，除非停止它，否则你不能运行任何其他单元格。</st>

<st c="11044">还有一个我们想要让你注意的附加参数：auth 参数。</st> <st c="11132">你可以像这样将其添加到</st> `<st c="11150">demo.launch</st>` <st c="11161">函数</st> <st c="11171">中：</st>

```py
 demo.launch(share=True, debug=True, auth=("admin", "pass1234"))
```

<st c="11245">这将添加一个简单的认证级别，以防你公开分享你的应用程序。</st> <st c="11345">它生成一个额外的界面，需要你添加的用户名（</st>`<st c="11405">admin</st>`<st c="11411">）和密码（</st>`<st c="11428">pass1234</st>`<st c="11437">）。</st> <st c="11460">将</st> `<st c="11467">admin/pass1234</st>` <st c="11481">改为你想要的任何内容，但绝对要更改它！</st> <st c="11530">仅将这些凭据分享给那些你想让他们访问你的 RAG 应用程序的用户。</st> <st c="11623">请记住，这并不非常安全，但它至少提供了一个基本的目的，以限制</st> <st c="11723">用户访问。</st>

<st c="11735">现在，你有一个活跃的 Web</st> <st c="11764">服务器，它可以接收输入，处理它，并根据你为你的 Gradio 界面编写的代码来响应和返回新的界面元素。</st> <st c="11914">这曾经需要显著的 Web 开发专业知识，但现在你可以在几分钟内将其设置并运行！</st> <st c="12046">这使得你可以专注于你想要关注的事情：编写你的</st> <st c="12122">RAG 应用程序的代码！</st>

<st c="12138">一旦你在该单元中运行了 Gradio 代码，界面就会变得交互式，允许用户在输入框中输入问题。</st> <st c="12278">正如我们之前所描述的，当用户提交一个问题，</st> `<st c="12342">process_question</st>` <st c="12358">函数会以用户的问题作为输入被调用。</st> <st c="12413">该函数调用一个 RAG 流程，</st> `<st c="12450">rag_chain_with_source</st>`<st c="12471">，并使用问题检索相关性得分、最终答案和来源。</st> <st c="12569">然后它返回相关性得分、最终答案和来源列表。</st> <st c="12637">Gradio 会用返回的值更新输出文本框，向用户显示相关性得分、最终答案和来源。</st>

<st c="12769">界面保持活跃和响应，直到单元执行完成或直到</st> `<st c="12858">gr.close_all()</st>` <st c="12872">被调用以关闭所有活动的</st> <st c="12903">Gradio 界面。</st>

<st c="12921">最终，当你使用 Gradio 代码运行这个笔记本单元时，你将得到一个看起来像</st> *<st c="13034">图 6</st>**<st c="13042">.1</st>*<st c="13044">. 的界面。你可以在笔记本中直接显示 Gradio 界面，也可以在运行</st> <st c="13184">单元时提供的链接的完整网页上显示。</st>

![图 6.1 – Gradio 界面](img/B22475_06_01.jpg)

<st c="13341">图 6.1 – Gradio 界面</st>

<st c="13370">我们已经预先填充了这个问题：</st> `<st c="13405">使用 RAG 的优点是什么？</st>`<st c="13442">。然而，你可以更改这个问题并询问其他内容。</st> <st c="13506">正如我们在上一章所讨论的，如果它与数据库的内容不相关，LLM 应该响应</st> `<st c="13626">我不知道</st>`<st c="13638">。我们鼓励你尝试使用相关和不相关的问题来测试它！</st> <st c="13716">看看你是否能找到一个按预期工作的场景来提高你的</st> <st c="13791">调试技能。</st>

<st c="13808">在你的笔记本中，这个界面上方你可能会看到类似</st> <st c="13892">以下</st> 的文本：

```py
 Colab notebook detected. This cell will run indefinitely so that you can see errors and logs. To turn off, set debug=False in launch(). Running on public URL: https://pl09q9e4g8989braee.gradio.live
This share link expires in 72 hours.
```

<st c="14135">点击该链接应该会在自己的浏览器窗口中提供界面的视图！</st> <st c="14233">它看起来就像</st> *<st c="14256">图 6</st>**<st c="14264">.1</st>*<st c="14266">，但它将占据整个</st> <st c="14299">浏览器窗口。</st>

<st c="14314">点击</st> `<st c="14450">result = rag_chain_with_source.invoke(question)</st>` <st c="14497">并在等待几秒钟后返回一个响应。</st> <st c="14552">生成的界面应该看起来类似于</st> *<st c="14599">图 6</st>**<st c="14607">.2</st>*<st c="14609">：</st>

![图 6.2 – Gradio 响应界面](img/B22475_06_02.jpg)

<st c="15535">图 6.2 – 带有响应的 Gradio 界面</st>

<st c="15578">当 LLM 返回响应时，让我们谈谈在这个界面中发生的一些事情。</st> <st c="15689">它从相关性得分开始，这是我们添加在</st> *<st c="15751">第五章</st>* <st c="15760">中，当使用 LLM 来确定问题的相关性作为安全措施以阻止提示注入时。</st> <st c="15872">在你向用户展示的应用程序中，这很可能不会显示，但在这里作为展示你 LLM 响应旁边额外信息的示例。</st> <st c="16040">LLM 响应。</st>

<st c="16054">说到 LLM 响应，</st> **<st c="16085">最终答案</st>** <st c="16097">来自 ChatGPT 4 已经格式化为带有标记的方式显示。</st> <st c="16174">Gradio 将自动使用该标记的换行符并相应地显示文本，在这种情况下，将段落分割。</st> <st c="16303">。</st>

最后，来源是一个包含四个来源的列表，表明检索器返回了四个来源。</st> <st c="16424">这来自于我们在</st> *<st c="16462">第三章</st>* <st c="16471">中设置的代码，当时我们增加了在元数据中携带检索结果来源的能力，以便我们在 UI 中显示。</st> <st c="16620">现在我们终于在这里看到了这个努力的成果</st> *<st c="16677">第六章</st>*<st c="16686">，因为我们现在有一个 UI 可以展示了！</st> <st c="16719">你可能已经注意到，所有四个来源都是相同的。</st> <st c="16776">这是由于这是一个小示例，我们只拉入了一个数据来源</st> <st c="16857">。</st>

<st c="16865">在大多数应用程序中，你可能会</st> <st c="16904">将更多的信息来源拉入你的数据中，并且在该列表中会有更多的来源。</st> <st c="17011">如果你向此代码添加更多与所提问题相关的数据来源，你应该会看到它们出现在这个来源</st> <st c="17137">列表中。</st>

# <st c="17148">摘要</st>

<st c="17156">在本章中，我们介绍了一个使用 RAG 和 Gradio 作为 UI 创建交互式应用的实用指南。</st> <st c="17277">我们涵盖了设置 Gradio 环境、集成 RAG 模型以及创建一个用户友好的界面，使用户能够像典型 Web 应用一样与 RAG 系统交互。</st> <st c="17468">开发者可以快速原型设计和部署 RAG 驱动的应用，使最终用户能够实时与 RAG 管道交互。</st> <st c="17591">。</st>

<st c="17601">我们还讨论了使用 Gradio 的好处，例如其开源性质、与流行机器学习框架的集成、协作功能以及 Gradio 与 Hugging Face 的集成，后者为生成式 AI 社区提供资源，包括使用 Hugging Face Spaces 永久和免费托管 Gradio 演示的能力。</st>

<st c="17959">通过代码实验室，我们学习了如何将 Gradio 界面添加到 RAG 应用中。</st> <st c="18042">我们使用</st> `<st c="18080">gr.Interface</st>`<st c="18092">创建 Gradio 界面，指定输入和输出组件、标题和描述。</st> <st c="18162">我们使用</st> `<st c="18193">demo.launch()</st>`<st c="18206">启动界面，该命令启动一个本地 Web 服务器以托管界面。</st> <st c="18263">这涉及到创建一个</st> `<st c="18288">process_question</st>` <st c="18304">函数，该函数调用 RAG 管道处理用户的问题，并从结果中检索相关性得分、最终答案和来源。</st> <st c="18447">这个过程反映了 Gradio 界面，使用户能够输入问题并接收由 RAG 系统返回的相关性得分、最终答案和来源。</st>

<st c="18614">本章还讨论了如何将来源从检索器传递到 UI 中显示，展示了在前面章节中添加此功能所付出的努力。</st>

<st c="18800">这只是对 Gradio 的一个简单介绍。</st> <st c="18849">我们鼓励您访问 Gradio 网站（</st>[<st c="18895">https://www.gradio.app/</st>](https://www.gradio.app/)<st c="18919">）并浏览他们的</st> **<st c="18945">快速入门</st>** <st c="18955">指南和文档，以了解他们平台提供的其他重要功能。</st>

<st c="19057">在下一章中，我们将探讨向量和向量存储在增强 RAG 系统中所扮演的关键角色。</st>
