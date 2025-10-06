# <st c="0">12</st>

# <st c="3">结合 RAG 与 AI 代理和 LangGraph 的力量</st>

<st c="59">一次调用一个</st> **<st c="75">大型语言模型</st>** <st c="95">(</st>**<st c="97">LLM</st>**<st c="100">) 可以非常强大，但将你的逻辑放在一个循环中，以实现更复杂的任务为目标，你就可以将你的</st> **<st c="226">检索增强生成</st>** <st c="256">(</st>**<st c="258">RAG</st>**<st c="261">) 开发提升到全新的水平。</st> <st c="298">这就是</st> **<st c="325">代理</st>**<st c="331">背后的概念。过去一年 LangChain 的开发重点放在了提高对</st> *<st c="432">代理</st>* <st c="439">工作流程的支持上，增加了能够更精确控制代理行为和功能的功能。</st> <st c="544">这一进步的部分成果是</st> **<st c="595">LangGraph</st>**<st c="604">的出现，LangChain 的另一个相对较新的部分。</st> <st c="648">共同来说，代理和 LangGraph 作为提高</st> <st c="725">RAG 应用</st> 的强大方法，配合得很好。

<st c="742">在本章中，我们将专注于深入了解可用于 RAG 的代理元素，然后将它们与你自己的 RAG 工作联系起来，涵盖以下主题：</st> <st c="930">以下内容：</st>

+   <st c="944">AI 代理和 RAG 集成的 fundamentals</st>

+   <st c="990">图，AI 代理，</st> <st c="1010">和 LangGraph</st>

+   <st c="1023">将 LangGraph 检索代理添加到你的</st> <st c="1067">RAG 应用</st>

+   <st c="1082">工具</st> <st c="1089">和工具包</st>

+   <st c="1101">代理状态</st>

+   <st c="1113">图论的核心概念</st>

<st c="1143">到本章结束时，你将牢固掌握 AI 代理和 LangGraph 如何增强你的 RAG 应用。</st> <st c="1266">在下一节中，我们将深入探讨 AI 代理和 RAG 集成的 fundamentals，为后续的概念和代码实验做好准备。</st>

# <st c="1416">技术要求</st>

<st c="1439">本章的代码放置在以下 GitHub</st> <st c="1500">仓库中：</st> [<st c="1512">https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_12</st>](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG/tree/main/Chapter_12)

# <st c="1609">AI 代理和 RAG 集成的 fundamentals</st>

<st c="1655">在与生成式 AI 的新开发者交谈时，我们被告知，AI 代理的概念往往是更难理解的概念之一。</st> <st c="1822">当专家们谈论代理时，他们经常用非常抽象的术语来谈论它们，关注 AI 代理在 RAG 应用中可以负责的所有事情，但未能真正彻底地解释 AI 代理是什么以及它是如何</st> <st c="2056">工作的。</st>

<st c="2065">我发现，通过解释它实际上是什么来消除 AI 代理的神秘感是最容易的，这是一个非常简单的概念。</st> <st c="2123">要构建最基本形式的 AI 代理，你只是将你在这些章节中一直在使用的相同的 LLM 概念添加一个循环，当预期任务完成时循环终止。</st> <st c="2419">就是这样！</st> <st c="2430">这只是个循环而已！</st>

*<st c="2453">图 12</st>**<st c="2463">.1</st>* <st c="2465">表示你将在即将投入使用的代码实验室中与之合作的**<st c="2481">RAG 代理循环</st>** <st c="2495">：</st>

![图 12.1 – 代理控制流程图](img/B22475_12_01.jpg)

<st c="2598">图 12.1 – 代理控制流程图</st>

<st c="2645">这代表了一系列相对简单的逻辑步骤，循环执行，直到代理决定它已经成功完成了你给它分配的任务。</st> <st c="2801">椭圆形框，例如</st> *<st c="2825">代理</st> * <st c="2830">和</st> *<st c="2835">检索</st>*, 被称为**<st c="2856">节点</st>** <st c="2861">，而线条被称为**<st c="2879">边</st>**<st c="2892">。虚线也是边，但它们是特定类型的边，称为**<st c="2963">条件边</st>**<st c="2980">，这些边也是**<st c="2997">决策点</st>**。</st>

<st c="3028">尽管简单，但在 LLM 调用中添加循环的概念确实使它比直接使用 LLM 更强大，因为它更多地利用了 LLM 推理和将任务分解成更简单任务的能力。</st> <st c="3267">这提高了你在追求的任何任务中取得成功的可能性，并且对于更复杂的多步骤**<st c="3401">RAG 任务</st>**将特别有用。</st>

<st c="3411">当你的 LLM 在循环执行代理任务时，你还会向代理提供称为**<st c="3493">工具</st>** <st c="3498">的函数，LLM 将使用其推理能力来确定使用哪个工具，如何使用该工具，以及向其提供什么数据。</st> <st c="3641">这很快就会变得非常复杂。</st> <st c="3695">你可以有多个代理，众多工具，集成的知识图谱来引导你的代理沿着特定路径前进，众多框架提供不同**<st c="3860">风味</st>** <st c="3867">的代理，众多代理架构的方法，等等。</st> <st c="3937">但在这个章节中，我们将专门关注 AI 代理如何帮助改进 RAG 应用。</st> <st c="4047">一旦你看到了使用 AI 代理的力量，我毫不怀疑你将想要在其他生成式 AI 应用中使用它，而且**<st c="4180">你应该这样做</st>**！</st>

## <st c="4191">生活在智能体世界中</st>

<st c="4219">在智能体周围的兴奋情绪中</st> <st c="4243">，你可能会认为 LLM 已经过时了。</st> <st c="4308">但事实远非如此。</st> <st c="4353">与 AI 智能体一起，你实际上是在挖掘一个更强大的 LLM 版本，在这个版本中，LLM 充当智能体的“大脑”，使其能够进行推理并提出超越一次性聊天问题的多步解决方案。</st> <st c="4625">智能体只是在用户和 LLM 之间提供一个层次，推动 LLM 完成可能需要多次查询的任务，但最终，从理论上讲，将得到一个更好的结果。</st> <st c="4819">更好的结果。</st>

<st c="4833">如果你这么想，这更符合现实世界中解决问题的方式，即使简单的决策也可能很复杂。</st> <st c="4970">我们做的许多任务都是基于一系列观察、推理和对新经验的调整。</st> <st c="5077">在现实世界中，我们很少以与在线使用 LLM 相同的方式与人们、任务和事物互动。</st> <st c="5199">通常会有这种理解和知识的构建过程，帮助我们找到最佳解决方案。</st> <st c="5324">AI 智能体更能处理这种类型的解决问题的方法</st> <st c="5382">。</st>

<st c="5401">智能体可以对您的 RAG 工作产生重大影响，但关于 LLM 作为其大脑的概念又如何呢？</st> <st c="5516">让我们进一步探讨</st> <st c="5536">这个概念。</st>

## <st c="5552">作为智能体的大脑</st>

<st c="5579">如果你认为 LLM 是您 AI 智能体的大脑，那么下一个合乎逻辑的步骤是，你很可能希望找到</st> *<st c="5685">最聪明</st>* <st c="5693">的 LLM 来充当这个大脑。</st> <st c="5729">LLM 的能力将影响您的 AI 智能体推理和决策的能力，这无疑将影响您 RAG 应用的查询结果。</st>

<st c="5910">然而，这种 LLM 大脑的隐喻有一个主要的</st> <st c="5929">缺陷，但以一种非常好的方式。</st> <st c="6008">与现实世界中的智能体不同，AI 智能体可以随时更换其 LLM 大脑为另一个 LLM 大脑。</st> <st c="6113">我们甚至可以给它多个 LLM 大脑，这些大脑可以相互检查并确保一切按计划进行。</st> <st c="6238">这为我们提供了更大的灵活性，将有助于我们不断改进智能体的能力。</st>

<st c="6341">那么，LangGraph 或一般意义上的图与 AI 智能体有何关联？</st> <st c="6409">我们将在下一节讨论</st> <st c="6425">这一点。</st>

# <st c="6435">图、AI 智能体和 LangGraph</st>

<st c="6468">LangChain 在</st> <st c="6489">2024 年引入了</st> <st c="6502">LangGraph，因此它仍然相对较新。</st> <st c="6540">它是建立在</st> `<st c="6863">AgentExecutor</st>` <st c="6876">类之上的扩展，仍然存在，LangGraph 现在是</st> *<st c="6919">推荐</st>* <st c="6930">在 LangChain 中构建代理的</st> *<st c="6951">方式</st>*

<st c="6964">LangGraph 增加了两个重要的组件</st> <st c="7004">以支持代理：</st>

+   <st c="7027">轻松定义周期（</st><st c="7065">循环图</st>）

+   <st c="7082">内置内存</st>

<st c="7098">它提供了一个与</st> `<st c="7144">AgentExecutor</st>`<st c="7157">等效的预构建对象，允许开发者使用基于</st> <st c="7209">图的方法来编排代理。</st>

<st c="7230">在过去的几年里，出现了许多将代理构建到 RAG 应用中的论文、概念和方法，例如编排代理、ReAct 代理、自我优化代理和多代理框架。</st> <st c="7452">这些方法中的一个共同主题是表示代理控制流的循环图概念。</st> <st c="7567">虽然许多这些方法从实现的角度来看正在变得过时，但它们的概念仍然非常有用，并且被 LangGraph 的基于图的环境所捕捉。</st> <st c="7741">LangGraph 已经成为支持代理并在 RAG 应用中管理它们的流程和过程的有力工具。</st>

**<st c="7754">LangGraph</st>** <st c="7764">已经成为</st> <st c="7775">支持代理和管理它们在 RAG 应用中的流程和过程的有力工具。</st> <st c="7871">它使开发者能够将单代理和多代理流程描述和表示为图，提供极其可控的</st> *<st c="7990">流程</st>*<st c="7995">。这种可控性对于避免开发者早期创建代理时遇到的陷阱至关重要。</st>

<st c="8111">例如，流行的 ReAct 方法是为构建代理的早期范例。</st> **<st c="8197">ReAct</st>** <st c="8202">代表</st> **<st c="8214">reason + act</st>**<st c="8226">。在这个模式中，一个 LLM</st> <st c="8251">首先思考要做什么，然后决定采取的行动。</st> <st c="8318">然后在这个环境中执行该行动，并返回一个观察结果。</st> <st c="8397">有了这个观察结果，LLM 随后重复这个过程。</st> <st c="8455">它使用推理来思考接下来要做什么，决定另一个要采取的行动，并继续直到确定目标已经达成。</st> <st c="8608">如果你将这个过程绘制出来，它可能看起来就像你在</st> *<st c="8685">图 12</st>**<st c="8694">.2</st>*<st c="8696">中看到的那样：</st>

![图 12.2 – ReAct 循环图表示](img/B22475_12_02.jpg)

<st c="8721">图 12.2 – ReAct 循环图表示</st>

<st c="8770">图 12.2 中的循环集合可以用 LangGraph 中的循环图来表示，每个步骤由节点和边表示。</st> <st c="8791">*<st c="8791">图 12</st>**<st c="8800">.2</st>**</st c="8802">可以用 LangGraph 中的循环图来表示，每个步骤由节点和边表示。</st> <st c="8902">使用这种图形范式，你可以看到像 LangGraph 这样的工具，LangChain 中构建图的工具，如何成为您代理框架的核心。</st> <st c="9062">在我们构建代理框架时，我们可以使用 LangGraph 来表示这些代理循环，这有助于您描述和编排控制流。</st> <st c="9206">这种对控制流的关注对于解决代理的一些早期挑战至关重要，缺乏控制会导致无法完成循环或专注于错误任务的代理。</st>

<st c="9410">LangGraph 内置的另一个关键元素是持久性。</st> <st c="9480">持久性可以用来保持代理的记忆，给它提供所需的信息来反思迄今为止的所有行动，并代表在</st> *<st c="9638">图 12</st>**<st c="9645">.2</st>**<st c="9680">中展示的* <st c="9638">OBSERVE</st> *组件。这非常有帮助，可以同时进行多个对话或记住之前的迭代和行动。</st> <st c="9804">这种持久性还使人类在循环中具有功能，让您在代理行动的关键间隔期间更好地控制其行为。</st>

<st c="9955">介绍 ReAct 方法构建代理的论文可以在以下位置找到：</st> <st c="10032">[<st c="10038">https://arxiv.org/abs/2210.03629</st>](https://arxiv.org/abs/2210.03629)

<st c="10070">让我们直接进入构建代理的代码实验室，并在代码中遇到它们时，更深入地探讨一些关键概念。</st> <st c="10199">。</st>

# <st c="10208">代码实验室 12.1 – 向 RAG 添加 LangGraph 代理</st>

在这个代码实验室中，我们将向现有的 RAG 管道添加一个代理<st c="10256">，它可以决定是否从索引中检索或使用网络搜索。</st> <st c="10295">我们将展示代理在处理数据时的内部想法，这些数据是为了向您提供更全面的回答。</st> <st c="10410">当我们添加代理的代码时，我们将看到新的组件，例如工具、工具包、图表、节点、边，当然还有代理本身。</st> <st c="10575">对于每个组件，我们将更深入地了解该组件如何与您的 RAG 应用程序交互和支持。</st> <st c="10718">我们还将添加代码，使这个功能更像是一个聊天会话，而不是一个</st> <st c="10832">问答会话：</st>

1.  <st c="10929">首先，我们将安装一些新的包来支持我们的</st> <st c="10986">代理开发：</st>

    ```py
    <st c="11004">%pip install tiktoken</st>
    ```

    ```py
    <st c="11084">tiktoken</st> package, which is an OpenAI package used for tokenizing text data before feeding it into language models. Last, we pull in the <st c="11220">langgraph</st> package we have been discussing.
    ```

1.  <st c="11262">接下来，我们添加一个新的 LLM 定义并更新我们的</st> <st c="11312">现有定义：</st>

    ```py
    <st c="11325">llm = ChatOpenAI(model_name="gpt-4o-mini",</st>
    ```

    ```py
     <st c="11368">temperature=0, streaming=True)</st>
    ```

    ```py
    <st c="11399">agent_llm = ChatOpenAI(model_name="gpt-4o-mini",</st>
    ```

    ```py
     <st c="11448">temperature=0, streaming=True)</st>
    ```

<st c="11479">新的</st> `<st c="11488">agent_llm</st>` <st c="11497">LLM 实例将作为我们代理的大脑，处理推理和执行代理任务，而原始的</st> `<st c="11618">llm</st>` <st c="11621">实例仍然存在于我们的通用 LLM 中，执行我们过去使用的相同 LLM 任务。</st> <st c="11730">虽然在我们的示例中，两个 LLM 使用相同的模型和参数定义，但您应该尝试使用不同的 LLM 来完成这些不同的任务，以查看是否有更适合您 RAG 应用的组合。</st> <st c="11975">您甚至可以添加额外的 LLM 来处理特定任务，例如，如果在这个代码中您发现某个 LLM 在那些任务上表现更好，或者您已经为这些特定操作训练或微调了自己的 LLM，那么可以添加</st> `<st c="12048">improve</st>` <st c="12055">或</st> `<st c="12059">score_documents</st>` <st c="12074">函数。</st> <st c="12209">例如，对于简单任务，只要它们能成功完成任务，通常可以使用更快、成本更低的 LLM 来处理。</st> <st c="12344">此代码中内置了大量的灵活性，您可以充分利用这些灵活性！</st> <st c="12427">此外，请注意，我们在 LLM 定义中添加了</st> `<st c="12450">streaming=True</st>` <st c="12464">。</st> <st c="12488">这会开启从 LLM 流式传输数据，这对可能进行多次调用（有时是并行调用）并不断与</st> <st c="12647">LLM 交互的代理更有利。</st>

<st c="12655">现在，我们将跳过检索器定义（</st>`<st c="12723">dense_retriever</st>`<st c="12739">， `<st c="12741">sparse_retriever</st>`<st c="12757">，和</st> `<st c="12763">ensemble_retriever</st>`<st c="12781">）之后的部分，并添加我们的第一个工具。</st> <st c="12808">在代理方面，</st> **<st c="12810">工具</st>** <st c="12814">有一个非常具体且重要的含义；因此，让我们现在来谈谈</st> <st c="12903">它。</st>

## <st c="12912">工具和工具包</st>

<st c="12931">在下面的代码中，我们将添加</st> <st c="12974">一个</st> **<st c="12977">web</st>** **<st c="12981">search</st>** <st c="12987">工具：</st>

```py
 from langchain_community.tools.tavily_search import TavilySearchResults
_ = load_dotenv(dotenv_path='env.txt')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
!export TAVILY_API_KEY=os.environ['TAVILY_API_KEY']
web_search = TavilySearchResults(max_results=4)
web_search_name = web_search.name
```

<st c="13297">您需要获取另一个 API 密钥并将其添加到</st> `<st c="13353">env.txt</st>` <st c="13360">文件中，这是我们过去用于 OpenAI 和 Together API 的文件。</st> <st c="13425">就像那些 API 一样，您需要访问那个网站，设置您的 API 密钥，然后将它复制到您的</st> `<st c="13539">env.txt</st>` <st c="13546">文件中。</st> <st c="13553">Tavily 网站可以通过此</st> <st c="13593">URL</st> 找到：[<st c="13598">https://tavily.com/</st>](https://tavily.com/)

<st c="13617">我们再次运行</st> <st c="13624">从</st> `<st c="13669">env.txt</st>` <st c="13676">文件加载数据的</st>代码，然后我们设置了一个名为</st> `<st c="13705">TavilySearchResults</st>` <st c="13724">的对象，其</st> `<st c="13737">max_results</st>` <st c="13748">为</st> `<st c="13752">4</st>`<st c="13753">，这意味着当我们运行搜索时，我们只想获得最多四个搜索结果。</st> <st c="13832">然后我们将</st> `<st c="13851">web_search.name</st>` <st c="13866">变量分配给一个名为</st> `<st c="13897">web_search_name</st>` <st c="13912">的变量，以便我们稍后当需要告诉代理关于它时可以使用。</st> <st c="13991">您可以直接使用以下代码运行此工具：</st>

```py
 web_search.invoke(user_query)
```

<st c="14068">使用</st> `<st c="14097">user_query</st>` <st c="14107">运行此工具代码将得到如下结果（为了简洁而截断）：</st>

```py
 [{'url': 'http://sustainability.google/',
  'content': "Google Maps\nChoose the most fuel-efficient route\nGoogle Shopping\nShop for more efficient appliances for your home\nGoogle Flights\nFind a flight with lower per-traveler carbon emissions\nGoogle Nest\...[TRUNCATED HERE]"},
…
  'content': "2023 Environmental Report. Google's 2023 Environmental Report outlines how we're driving positive environmental outcomes throughout our business in three key ways: developing products and technology that empower individuals on their journey to a more sustainable life, working together with partners and organizations everywhere to transition to resilient, low-carbon systems, and operating ..."}]
```

<st c="14856">我们截断了这部分内容，以便在书中占用更少的空间，但在代码中尝试这样做，您将看到我们请求的四个结果，并且它们似乎都与</st> `<st c="15038">user_query</st>` <st c="15048">用户询问的主题</st>高度相关。</st> <st c="15066">请注意，您不需要像我们</st> <st c="15141">刚才那样直接在代码中运行此工具。</st>

<st c="15150">到此为止，您已经建立了您的第一个代理工具！</st> <st c="15215">这是一个搜索引擎工具，您的代理可以使用它从互联网上检索更多信息，以帮助它实现回答用户提出的问题的目标。</st> <st c="15385">到它。</st>

<st c="15391">在 LangChain</st> *<st c="15396">工具</st>* <st c="15400">概念以及构建代理时，其灵感来源于您希望使动作对代理可用，以便它可以执行其任务。</st> <st c="15557">工具是实现这一目标的机制。</st> <st c="15609">您定义一个工具，就像我们刚才为网络搜索所做的那样，然后您稍后将其添加到代理可以用来完成任务的工具列表中。</st> <st c="15757">在我们设置该列表之前，我们想要创建另一个对于 RAG 应用至关重要的工具：一个</st> <st c="15864">检索工具：</st>

```py
 from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    ensemble_retriever,
    "retrieve_google_environmental_question_answers",
    "Extensive information about Google environmental
     efforts from 2023.",
)
retriever_tool_name = retriever_tool.name
```

<st c="16164">请注意，使用</st> <st c="16194">网络搜索</st> <st c="16221">工具时，我们从</st> `<st c="16221">langchain_community.tools.tavily_search</st>`<st c="16260">》导入，而使用这个工具时，我们使用</st> `<st c="16293">langchain.tools.retriever</st>`<st c="16318">》。这反映了 Tavily 是一个第三方工具，而我们在这里创建的检索工具是 LangChain 核心功能的一部分。</st> <st c="16465">导入</st> `<st c="16485">create_retriever_tool</st>` <st c="16506">函数后，我们使用它来创建</st> `<st c="16541">retriever_tool</st>` <st c="16555">工具供我们的代理使用。</st> <st c="16576">同样，就像</st> `<st c="16593">web_search_name</st>`<st c="16608">》一样，我们提取出</st> `<st c="16626">retriever_tool.name</st>` <st c="16645">变量，稍后当我们需要为代理引用它时可以引用。</st> <st c="16721">你可能注意到了这个工具将使用的实际检索器名称，即</st> `<st c="16793">ensemble_retriever</st>` <st c="16811">检索器，我们在</st> *<st c="16843">第八章</st>*<st c="16852">》的</st> *<st c="16856">8.3</st>* *<st c="16860">代码实验室</st>*<st c="16868">》中创建的！</st>

<st c="16869">你还应该注意，这个工具的名称，从代理的角度来看，位于第二个字段，我们将其命名为</st> `<st c="17020">retrieve_google_environmental_question_answers</st>`<st c="17066">》。在代码中命名变量时，我们通常尝试使它们更小，但对于代理将使用的工具，提供更详细的名称有助于代理理解可以使用的内容。</st> <st c="17266">完全。</st>

<st c="17277">我们现在为我们的代理有了两个工具！</st> <st c="17315">然而，我们最终还需要告诉代理关于它们的信息；因此，我们将它们打包成一个列表，稍后我们可以与</st> <st c="17443">代理共享：</st>

```py
 tools = [web_search, retriever_tool]
```

<st c="17490">你在这里可以看到我们之前创建的两个工具，即</st> `<st c="17545">web_search</st>` <st c="17555">工具和</st> `<st c="17569">retriever_tool</st>` <st c="17583">工具，被添加到工具列表中。</st> <st c="17623">如果我们有其他想要提供给代理的工具，我们也可以将它们添加到列表中。</st> <st c="17727">在 LangChain</st> <st c="17743">生态系统中有数百种工具</st> <st c="17783">可供使用：</st> [<st c="17794">https://python.langchain.com/v0.2/docs/integrations/tools/</st>](https://python.langchain.com/v0.2/docs/integrations/tools/)

<st c="17852">你想要确保你使用的 LLM 在推理和使用工具方面是“优秀”的。</st> <st c="17942">一般来说，聊天模型通常已经针对工具调用进行了微调，因此在使用工具方面会更好。</st> <st c="18047">未针对聊天进行微调的模型可能无法使用工具，尤其是当工具复杂或需要多次调用时。</st> <st c="18167">使用良好的名称和描述可以在为你的代理 LLM 设定成功方面发挥重要作用。</st> <st c="18277">同样。</st>

<st c="18285">在我们构建的代理中，我们拥有所有需要的工具，但你也会想看看工具包，这些是方便的工具组合。</st> <st c="18433">LangChain 在其网站上提供当前可用的工具包列表</st> <st c="18482">：</st> [<st c="18511">https://python.langchain.com/v0.2/docs/integrations/toolkits/</st>](https://python.langchain.com/v0.2/docs/integrations/toolkits/)

<st c="18572">例如，如果你有一个使用 pandas DataFrames 的数据基础设施，你可以使用 pandas DataFrame 工具包为你提供各种工具，以不同的方式访问这些 DataFrames。</st> <st c="18772">直接从 LangChain 网站引用，工具包被描述如下：</st> <st c="18843">（</st>[<st c="18853">https://python.langchain.com/v0.1/docs/modules/agents/concepts/#toolkits</st>](https://python.langchain.com/v0.1/docs/modules/agents/concepts/#toolkits)<st c="18926">）</st>

<st c="18928">对于许多常见任务，代理将需要一套相关工具。</st> <st c="18994">为此，LangChain 提供了工具包的概念——大约 3-5 个工具的组合，用于完成特定目标。</st> <st c="19117">例如，GitHub 工具包包含用于搜索 GitHub 问题的工具、用于读取文件的工具、用于评论的工具等。</st>

<st c="19251">因此，基本上，如果你正在关注你的代理或 LangChain（例如 Salesforce 集成）的一组常见任务，很可能有一个工具包可以让你一次性获得所有需要的工具。</st> <st c="19493">对于许多常见任务，代理将需要一套相关工具。</st>

<st c="19501">现在我们已经建立了工具，让我们开始构建代理的组件，从</st> <st c="19610">代理状态</st> 开始。

## <st c="19622">代理状态</st>

<st c="19634">用于为你的代理建立“状态”并随时间跟踪的</st> `<st c="19740">AgentState</st>` <st c="19750">类。</st> <st c="19826">此状态是代理的本地机制，你可以将其提供给图的所有部分，并可以存储在持久层中。</st>

<st c="19962">在这里，我们为我们的</st> <st c="19998">RAG 代理</st> 设置此状态：

```py
 from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],
                        add_messages]
```

<st c="20250">这导入设置`<st c="20297">AgentState</st>`<st c="20307">的相关包。例如，`<st c="20322">BaseMessage</st>` <st c="20333">是用于表示用户与 AI 智能体之间对话的消息的基类。</st> <st c="20431">它将用于定义对话状态中消息的结构和属性。</st> <st c="20532">然后它定义了一个图和一个`<st c="20562">"state"</st>` <st c="20569">对象，并将其传递给每个节点。</st> <st c="20613">您可以设置状态为各种类型的对象，以便存储不同类型的数据，但对我们来说，我们设置我们的状态为一个`<st c="20774">"messages"</st>`<st c="20784">列表。</st>

<st c="20785">我们需要导入另一轮包来设置我们的智能体其他部分：</st> <st c="20860">我们的智能体：</st>

```py
 from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import tools_condition
```

<st c="21022">在这段代码中，我们首先导入</st> `<st c="21061">HumanMessage</st>`<st c="21073">。</st> `<st c="21075">HumanMessage</st>` <st c="21087">是一种特定的消息类型，代表由人类用户发送的消息。</st> <st c="21168">它将在构建智能体生成响应的提示时使用。</st> <st c="21248">我们还导入</st> `<st c="21263">BaseModel</st>` <st c="21272">和</st> `<st c="21277">Field</st>`<st c="21282">。</st> `<st c="21284">BaseModel</st>` <st c="21293">是来自`<st c="21314">Pydantic</st>` <st c="21322">库的一个类，用于定义数据模型和验证数据。</st> `<st c="21385">Field</st>` <st c="21390">是来自`<st c="21407">Pydantic</st>` <st c="21415">的一个类，用于定义数据模型中字段的属性和验证规则。</st> <st c="21503">最后，我们导入</st> `<st c="21519">tools_condition</st>`<st c="21534">。`<st c="21540">tools_condition</st>` <st c="21555">函数是`<st c="21605">LangGraph</st>` <st c="21614">库提供的预构建函数。</st> <st c="21624">它用于根据当前对话状态评估智能体是否使用特定工具的决策。</st>

<st c="21746">这些导入的类和函数在代码中用于定义消息的结构、验证数据和根据智能体的决策控制对话流程。</st> <st c="21938">它们为使用`<st c="22053">LangGraph</st>` <st c="22062">库构建语言模型应用程序提供了必要的构建块和实用工具。</st>

<st c="22071">然后我们定义我们的主要提示（表示用户会输入的内容）：</st> <st c="22147">如下所示：</st>

```py
 generation_prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, just say
    that you don't know. Provide a thorough description to
    fully answer the question, utilizing any relevant
    information you find. Question: {question}
    Context: {context}
    Answer:"""
)
```

<st c="22550">这是过去我们在代码实验室中使用的代码的替代品：</st> <st c="22617">代码实验室：</st>

```py
 prompt = hub.pull("jclemens24/rag-prompt")
```

<st c="22670">我们将名称更改为</st> `<st c="22692">generation_prompt</st>` <st c="22709">，以使此提示的使用</st> <st c="22736">更加清晰。</st>

<st c="22747">我们的代码中的图使用即将增加，但首先，我们需要介绍一些基本的图</st> <st c="22842">理论概念。</st>

# <st c="22858">图论的核心概念</st>

<st c="22888">为了更好地理解我们将在接下来的几段代码中使用 LangGraph 的方式，回顾一些</st> <st c="23017">图论</st> <st c="23033">中的关键概念是有帮助的。</st> **<st c="23035">图</st>** <st c="23041">是数学</st> <st c="23058">结构，可以用来表示不同对象之间的关系。</st> <st c="23141">这些对象被称为</st> **<st c="23164">节点</st>** <st c="23169">，它们之间的关系，通常用线表示，被称为</st> **<st c="23246">边</st>**<st c="23251">。您已经在 *<st c="23293">图 12</st>**<st c="23302">.1</st>*<st c="23304"> 中看到了这些概念，但了解它们如何与任何图相关联以及如何在 LangGraph 中使用它们是很重要的。</st>

<st c="23403">在 LangGraph 中，也有表示这些关系不同类型的特定类型的边。</st> <st c="23512">例如，与 *<st c="23564">图 12</st>**<st c="23573">.1</st>*<st c="23575"> 一起提到的“条件边”，表示您需要决定下一步应该访问哪个节点；因此，它们代表决策。</st> <st c="23708">在讨论 ReAct 范式时，这也被称为</st> <st c="23772">**<st c="23777">动作边</st>**<st c="23788">**，因为它是在动作发生的地方，与 ReAct 的</st> *<st c="23845">原因 + 动作</st>** <st c="23860">方法相关。</st> *<st c="23880">图 12</st>**<st c="23889">.3</st>* <st c="23891">显示了由节点</st> <st c="23932">和边组成的</st> 基本图：</st>

![图 12.3 – 表示我们 RAG 应用基本图的图形](img/B22475_12_03.jpg)

<st c="23955">图 12.3 – 表示我们 RAG 应用基本图的图形</st>

<st c="24013">如图 <st c="24046">图 12</st>**<st c="24055">.3</st>*<st c="24057"> 所示的循环图中，您可以看到代表开始、代理、检索工具、生成、观察和结束的节点。</st> <st c="24153">关键边是 LLM 决定使用哪个工具（这里只有检索可用），观察检索到的信息是否足够，然后推动到生成。</st> <st c="24343">如果决定检索到的数据不足，有一条边将观察结果发送回代理，以决定是否再次尝试。</st> <st c="24501">这些决策点是我们讨论的</st> *<st c="24531">条件边</st>* <st c="24548">。</st>

# <st c="24562">我们的智能体中的节点和边</st>

<st c="24591">好的，让我们</st> <st c="24604">回顾一下。</st> <st c="24613">我们提到，一个代理 RAG 图有三个关键组件：我们之前提到的</st> *<st c="24685">状态</st>* <st c="24690">，添加到或更新状态的</st> *<st c="24725">节点</st> <st c="24730">，以及决定下一个要访问哪个节点的</st> *<st c="24775">条件边</st>* <st c="24792">。</st> <st c="24831">我们现在已经到了可以逐个在代码块中逐步通过这些组件，看到这三个组件如何相互作用的程度。</st>

<st c="24968">基于这个背景，我们将首先向代码中添加条件边，这是决策的地方。</st> <st c="25087">在这种情况下，我们将定义一个边，以确定检索到的文档是否与问题相关。</st> <st c="25205">这是将决定是否进入生成阶段或返回并</st> <st c="25304">重试的函数：</st>

1.  <st c="25314">我们将分多步逐步通过此代码，但请记住，这是一个大函数，从</st> <st c="25429">定义</st>开始：</st>

    ```py
     def score_documents(state) -> Literal[
    ```

    ```py
     "generate", "improve"]:
    ```

    <st c="25507">此代码首先定义了一个名为</st> `<st c="25555">score_documents</st>` <st c="25570">的函数，该函数确定检索到的文档是否与给定问题相关。</st> <st c="25655">该函数接受我们一直在讨论的状态作为参数，这是一个收集到的消息集合。</st> <st c="25775">这就是我们使状态</st> `<st c="25805">可用</st>` <st c="25814">于此条件</st> <st c="25835">边函数</st>的方式。

1.  <st c="25849">现在，我们构建</st> <st c="25868">数据模型：</st>

    ```py
     class scoring(BaseModel):
    ```

    ```py
     binary_score: str = Field(
    ```

    ```py
     description="Relevance score 'yes' or 'no'")
    ```

    <st c="25977">这定义了一个名为</st> `<st c="26017">scoring</st>` <st c="26024">的数据模型类，使用</st> `<st c="26031">Pydantic</st>`<st c="26039">的</st> `<st c="26043">BaseModel</st>`<st c="26052">。`<st c="26058">scoring</st>` <st c="26065">类有一个名为</st> `<st c="26098">binary_score</st>`<st c="26110">的单个字段，它是一个表示相关性得分的字符串，可以是</st> `<st c="26173">是</st>` <st c="26176">或</st> `<st c="26180">否</st>`<st c="26182">。</st>

1.  <st c="26183">接下来，我们添加将做出此决定的 LLM：</st>

    ```py
     llm_with_tool = llm.with_structured_output(
    ```

    ```py
     scoring)
    ```

    <st c="26287">这通过调用</st> `<st c="26316">llm_with_tool</st>` <st c="26329">并使用</st> `<st c="26341">llm.with_structured_output(scoring)</st>`<st c="26376">创建了一个实例，将 LLM 与评分数据模型结合用于结构化</st> <st c="26439">输出验证。</st>

1.  <st c="26457">正如我们过去所看到的，我们需要设置一个</st> `<st c="26507">PromptTemplate</st>` <st c="26521">类，并将其传递给</st> <st c="26543">LLM。</st> <st c="26548">以下是</st> <st c="26556">该提示：</st>

    ```py
     prompt = PromptTemplate(
    ```

    ```py
     template="""You are assessing relevance of a retrieved document to a user question with a binary grade. Here is the retrieved document:
    ```

    ```py
     {context}
    ```

    ```py
     Here is the user question: {question}
    ```

    ```py
     If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
    ```

    ```py
     input_variables=["context", "question"],
    ```

    ```py
     )
    ```

    <st c="27034">这使用</st> `<st c="27066">PromptTemplate</st>` <st c="27080">类定义了一个提示，为 LLM 提供根据给定问题对检索到的文档的相关性应用二进制评分的说明。</st>

1.  <st c="27222">然后我们可以使用 LCEL 构建一个链，将提示与刚刚</st> `<st c="27297">llm_with_tool</st>` <st c="27310">设置的工具</st> `<st c="27324">结合：</st>

    ```py
     chain = prompt | llm_with_tool
    ```

    <st c="27362">这个链表示了评分文档的管道。</st> <st c="27425">这定义了链，但我们还没有调用</st> <st c="27472">它。</st>

1.  <st c="27479">首先，我们想要</st> <st c="27498">获取状态。</st> <st c="27517">接下来，我们将状态（</st>`<st c="27542">"messages"</st>`<st c="27553">）拉入函数中，以便我们可以使用它，并取最后一条</st> <st c="27613">消息：</st>

    ```py
     messages = state["messages"]
    ```

    ```py
     last_message = messages[-1]
    ```

    ```py
     question = messages[0].content
    ```

    ```py
     docs = last_message.content
    ```

    <st c="27742">这从</st> `<st c="27792">"state"</st>` <st c="27799">参数中提取必要的信息，然后准备状态/消息作为我们将传递给我们的智能大脑（LLM）的上下文。</st> <st c="27905">这里提取的具体组件包括</st> <st c="27952">以下内容：</st>

    +   `<st c="27966">messages</st>`<st c="27975">: 对话中的消息列表</st>

    +   `<st c="28018">last_message</st>`<st c="28031">: 对话中的最后一条消息</st>

    +   `<st c="28070">question</st>`<st c="28079">: 第一条消息的内容，假设它是</st> <st c="28143">用户的问题</st>

    +   `<st c="28158">docs</st>`<st c="28163">: 最后一条消息的内容，假设它是</st> <st c="28226">检索到的文档</st>

    <st c="28245">然后，最后，我们使用填充的提示调用链（如果你还记得，我们称之为</st> `<st c="28362">question</st>` <st c="28370">和上下文</st> `<st c="28383">docs</st>` <st c="28387">以获取</st> <st c="28399">评分结果：</st>

    ```py
     scored_result = chain.invoke({"question":
    ```

    ```py
     question, "context": docs})
    ```

    ```py
     score = scored_result.binary_score
    ```

    <st c="28518">这从</st> <st c="28532">`<st c="28537">binary_score</st>`</st> <st c="28549">变量从</st> `<st c="28568">scored_result</st>` <st c="28581">对象中提取，并将其分配给</st> `<st c="28611">score</st>` <st c="28616">变量。</st> <st c="28627">`<st c="28631">llm_with_tool</st>`</st> <st c="28644">步骤，这是 LangChain 链中的最后一步，恰当地称为</st> `<st c="28711">chain</st>`<st c="28716">，将根据评分函数的响应返回基于字符串的二进制结果：</st>

    ```py
     if score == "yes":
    ```

    ```py
     print("---DECISION: DOCS RELEVANT---")
    ```

    ```py
     return "generate"
    ```

    ```py
     else:
    ```

    ```py
     print("---DECISION: DOCS NOT RELEVANT---")
    ```

    ```py
     print(score)
    ```

    ```py
     return "improve"
    ```

    <st c="28969">这检查得分的值。</st> <st c="29006">如果</st> `<st c="29013">得分</st>` <st c="29018">值是</st> `<st c="29028">是</st>`<st c="29031">，它将打印一条消息，表明文档是相关的，并从</st> `<st c="29108">generate</st>` <st c="29116">函数返回作为最终输出，这表明下一步是生成一个响应。</st> <st c="29229">如果</st> `<st c="29236">得分</st>` <st c="29241">值是</st> `<st c="29251">否</st>`<st c="29253">，或者技术上讲，任何不是</st> `<st c="29292">是</st>`<st c="29295">的东西，它将打印消息表明文档是不相关的，并返回</st> `<st c="29375">改进</st>`<st c="29382">，这表明下一步是从</st> `<st c="29443">用户</st>` `<st c="29443">那里改进查询。</st>

    <st c="29452">总的来说，这个函数在工作流程中充当决策点，确定检索到的文档是否与问题相关，并根据相关性得分将流程导向生成响应或重写问题。</st> <st c="29681">的。</st>

1.  <st c="29697">现在我们已经定义了我们的条件边，我们将继续定义我们的节点，从</st> <st c="29772">代理</st> `<st c="29806">开始：</st>

    ```py
     def agent(state):
    ```

    ```py
     print("---CALL AGENT---")
    ```

    ```py
     messages = state["messages"]
    ```

    ```py
     llm = llm.bind_tools(tools)
    ```

    ```py
     response = llm.invoke(messages)
    ```

    ```py
     return {"messages": [response]}
    ```

    <st c="29981">这个函数代表我们图上的代理节点，并调用代理模型根据当前状态生成响应。</st> <st c="30114">代理</st> `<st c="30118">函数接受当前状态（</st>`<st c="30158">"状态"</st>`<st c="30166">）作为输入，其中包含对话中的消息，打印一条消息表明它正在调用代理，从状态字典中提取消息，使用</st> `<st c="30344">agent_llm</st>` <st c="30353">实例的</st> `<st c="30370">ChatOpenAI</st>` <st c="30380">类，我们之前定义的，代表代理的</st> *<st c="30430">大脑</st>*<st c="30435">，然后使用</st> `<st c="30485">bind_tools</st>` <st c="30495">方法将工具绑定到模型。</st> <st c="30504">然后我们调用代理的</st> `<st c="30531">llm</st>` <st c="30534">实例，将消息传递给它，并将结果赋值给</st> `<st c="30591">response</st>` <st c="30599">变量。</st>

1.  <st c="30609">我们的下一个节点</st> `<st c="30625">改进</st>`<st c="30632">，负责将</st> `<st c="30666">用户查询</st>` <st c="30676">转换为更好的问题，如果代理确定这是必需的：</st>

    ```py
     def improve(state):
    ```

    ```py
     print("---TRANSFORM QUERY---")
    ```

    ```py
     messages = state["messages"]
    ```

    ```py
     question = messages[0].content
    ```

    ```py
     msg = [
    ```

    ```py
     HumanMessage(content=f"""\n
    ```

    ```py
     Look at the input and try to reason about
    ```

    ```py
     the underlying semantic intent / meaning.
    ```

    ```py
     \n
    ```

    ```py
     Here is the initial question:
    ```

    ```py
     \n ------- \n
    ```

    ```py
     {question}
    ```

    ```py
     \n ------- \n
    ```

    ```py
     Formulate an improved question:
    ```

    ```py
     """,
    ```

    ```py
     )
    ```

    ```py
     ]
    ```

    ```py
     response = llm.invoke(msg)
    ```

    ```py
     return {"messages": [response]}
    ```

    <st c="31148">此函数，就像我们所有的</st> <st c="31173">节点和边相关函数一样，接受</st> <st c="31222">当前状态（</st>`<st c="31238">"状态"</st>`<st c="31246">）作为输入。</st> <st c="31259">该函数返回一个字典，其中将响应附加到消息列表中。</st> <st c="31342">该函数打印一条消息表示正在转换查询，从状态字典中提取消息，检索第一条消息的内容（</st>`<st c="31511">messages[0].content</st>`<st c="31531">），假设它是初始问题，并将其分配给</st> `<st c="31602">问题</st>` <st c="31610">变量。</st> <st c="31621">然后我们使用</st> `<st c="31656">HumanMessage</st>` <st c="31668">类设置一条消息，表示我们希望</st> `<st c="31704">llm</st>` <st c="31707">实例推理问题的潜在语义意图并制定一个改进的问题。</st> <st c="31816">来自</st> `<st c="31836">llm</st>` <st c="31839">实例的结果分配给</st> `<st c="31868">响应</st>` <st c="31876">变量。</st> <st c="31887">最后，它返回一个字典，其中将响应附加到</st> `<st c="31954">消息列表</st>`。</st>

1.  <st c="31968">我们的下一个节点函数是</st> `<st c="31999">generate</st>` <st c="32007">函数：</st>

    ```py
     def generate(state):
    ```

    ```py
     print("---GENERATE---")
    ```

    ```py
     messages = state["messages"]
    ```

    ```py
     question = messages[0].content
    ```

    ```py
     last_message = messages[-1]
    ```

    ```py
     question = messages[0].content
    ```

    ```py
     docs = last_message.content
    ```

    ```py
     rag_chain = generation_prompt | llm |
    ```

    ```py
     str_output_parser
    ```

    ```py
     response = rag_chain.invoke({"context": docs,
    ```

    ```py
     "question": question})
    ```

    ```py
     return {"messages": [response]}
    ```

    <st c="32366">此函数</st> <st c="32384">类似于上一章代码中的生成步骤</st> <st c="32446">实验室，但简化了以仅提供响应。</st> <st c="32496">它基于检索到的文档和问题生成一个答案。</st> <st c="32570">该函数接受当前状态（</st>`<st c="32608">"状态"</st>`<st c="32616">）作为输入，其中包含对话中的消息，打印一条消息表示正在生成答案，从状态字典中提取消息，检索第一条消息的内容（</st>`<st c="32832">messages[0].content</st>`<st c="32852">），假设它是问题，并将其分配给</st> `<st c="32915">问题</st>` <st c="32923">变量。</st>

    <st c="32933">该函数随后检索最后一条消息（</st>`<st c="32980">messages[-1]</st>`<st c="32993">）并将其分配给</st> `<st c="33018">last_message</st>` <st c="33030">变量。</st> <st c="33041">`<st c="33045">docs</st>` <st c="33049">变量被分配给`<st c="33086">last_message</st>`<st c="33098">的内容，假设这是检索到的文档。</st> <st c="33148">在此阶段，我们通过使用`<st c="33281">|</st>` <st c="33282">操作符</st>组合`<st c="33215">generation_prompt</st>`<st c="33232">`、`<st c="33234">llm</st>`<st c="33237">`和`<st c="33243">str_output_parser</st>` <st c="33260">变量来创建一个名为`<st c="33188">rag_chain</st>` <st c="33197">的链。</st> <st c="33293">与其他 LLM 提示一样，我们将预定义的`<st c="33348">generation_prompt</st>` <st c="33365">作为生成答案的提示，它返回一个包含`<st c="33443">response</st>` <st c="33451">变量并附加到`<st c="33477">messages</st>` <st c="33485">列表中的字典。</st>

<st c="33491">接下来，我们想要使用 LangGraph 设置我们的循环图，并将我们的节点和边</st> <st c="33583">分配给它们。</st>

# <st c="33591">循环图设置</st>

<st c="33612">我们代码中的下一个</st> <st c="33622">大步骤是使用 LangGraph</st> <st c="33643">设置我们的图</st> <st c="33668">：</st>

1.  <st c="33684">首先，我们导入一些重要的包以开始：</st> <st c="33733">：</st>

    ```py
     from langgraph.graph import END, StateGraph
    ```

    ```py
     from langgraph.prebuilt import ToolNode
    ```

    <st c="33828">此代码从</st> `<st c="33902">langgraph</st>` <st c="33911">库中导入以下必要的类和函数：</st>

    +   `<st c="33920">END</st>`<st c="33924">: 表示工作流程结束的特殊节点</st> <st c="33966">的</st>

    +   `<st c="33978">StateGraph</st>`<st c="33989">: 用于定义工作流程</st> <st c="34032">的状态图的类</st>

    +   `<st c="34044">ToolNode</st>`<st c="34053">: 用于定义表示工具</st> <st c="34107">或动作</st>的节点的类

1.  <st c="34116">然后，我们将</st> `<st c="34130">AgentState</st>` <st c="34140">作为参数传递给</st> `<st c="34163">StateGraph</st>` <st c="34173">类，我们刚刚导入它来定义工作流程</st> <st c="34229">的状态图：</st>

    ```py
     workflow = StateGraph(AgentState)
    ```

    <st c="34276">这创建了一个名为</st> `<st c="34326">workflow</st>` <st c="34334">的新</st> `<st c="34308">StateGraph</st>` <st c="34318">实例</st> <st c="34334">，并为该</st> `<st c="34368">工作流程</st>` `<st c="34376">StateGraph</st>` <st c="34387">实例</st>定义了一个新的图。</st>

1.  <st c="34397">接下来，我们定义我们将循环的节点，并将我们的节点函数</st> <st c="34476">分配给它们：</st>

    ```py
     workflow.add_node("agent", agent)  # agent
    ```

    ```py
     retrieve = ToolNode(tools)
    ```

    ```py
     workflow.add_node("retrieve", retrieve)
    ```

    ```py
     # retrieval from web and or retriever
    ```

    ```py
     workflow.add_node("improve", improve)
    ```

    ```py
     # Improving the question for better retrieval
    ```

    ```py
     workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant
    ```

    <st c="34820">此代码使用</st> `<st c="34886">add_node</st>` <st c="34894">方法</st> <st c="34836">将多个节点添加到</st> `<st c="34858">工作流程</st>` <st c="34866">实例</st> <st c="34876">中：</st>

    +   `<st c="34902">"agent"</st>`<st c="34910">：此节点代表代理节点，它调用</st> <st c="34968">agent 函数。</st>

    +   `<st c="34983">"retrieve"</st>`<st c="34994">：此节点代表检索节点，它是一个特殊的</st> `<st c="35057">ToolNode</st>` <st c="35065">，包含我们早期定义的工具列表，包括</st> `<st c="35118">web_search</st>` <st c="35128">和</st> `<st c="35133">retriever_tool</st>` <st c="35147">工具。</st> <st c="35155">在此代码中，为了提高可读性，我们明确地分离出</st> `<st c="35220">ToolNode</st>` <st c="35228">类实例，并使用它定义了</st> `<st c="35259">retrieve</st>` <st c="35267">变量，这更明确地表示了此节点的“检索”焦点。</st> <st c="35353">然后我们将该</st> `<st c="35371">retrieve</st>` <st c="35379">变量传递给</st> `<st c="35398">add_node</st>` <st c="35406">函数。</st>

    +   `<st c="35416">"improve"</st>`<st c="35426">：此节点代表改进问题的节点，它调用</st> `<st c="35505">improve</st>` <st c="35512">函数。</st>

    +   `<st c="35522">"generate"</st>`<st c="35533">：此节点代表生成响应的节点，它调用</st> `<st c="35611">generate</st>` <st c="35619">函数。</st>

1.  <st c="35629">接下来，我们需要定义我们的工作流</st> <st c="35677">的起点：</st>

    ```py
     workflow.set_entry_point("agent")
    ```

    <st c="35724">这设置了</st> `<st c="35758">工作流</st>` <st c="35766">实例的</st> `<st c="35783">"agent"</st>` <st c="35790">节点</st> <st c="35796">使用</st> `<st c="35802">workflow.set_entry_point("agent")</st>`<st c="35835">.</st>

1.  <st c="35836">接下来，我们调用</st> `<st c="35855">"agent"</st>` <st c="35862">节点来决定是否检索</st> <st c="35898">：</st>

    ```py
     workflow.add_conditional_edges("agent", tools_condition,
    ```

    ```py
     {
    ```

    ```py
     "tools": "retrieve",
    ```

    ```py
     END: END,
    ```

    ```py
     },
    ```

    ```py
     )
    ```

    <st c="36000">在此代码中，</st> `<st c="36014">tools_condition</st>` <st c="36029">被用作工作流程图中的条件边。</st> <st c="36083">它根据代理的决定确定代理是否应继续到检索步骤（</st>`<st c="36153">"tools": "retrieve"</st>`<st c="36173">）或结束对话（</st>`<st c="36201">END: END</st>`<st c="36210">）。</st> <st c="36244">检索步骤代表我们为代理提供的两个工具，供其按需使用，而另一个选项，简单地结束对话，则结束</st> <st c="36408">工作流程。</st>

1.  <st c="36421">在此，我们添加更多</st> <st c="36440">边，这些边在调用</st> `<st c="36472">"action"</st>` <st c="36480">节点</st> <st c="36486">之后使用：</st>

    ```py
     workflow.add_conditional_edges("retrieve",
    ```

    ```py
     score_documents)
    ```

    ```py
     workflow.add_edge("generate", END)
    ```

    ```py
     workflow.add_edge("improve", "agent")
    ```

    在调用 `<st c="36640">"retrieve"` <st c="36650">节点后，它使用</st> `<st c="36699">workflow.add_conditional_edges("retrieve", score_documents)</st>`<st c="36758">添加条件边。这使用</st> `<st c="36808">score_documents</st>` <st c="36823">函数评估检索到的文档，并根据分数确定下一个节点。</st> <st c="36882">这还会使用</st> `<st c="36952">workflow.add_edge("generate", END)</st>`<st c="36986">从</st> `<st c="36914">"generate"` <st c="36924">节点到</st> `<st c="36937">END</st>` <st c="36940">节点添加一个边。这表示在生成响应后，工作流程结束。</st> <st c="37057">最后，它使用</st> `<st c="37136">workflow.add_edge("improve", "agent")</st>`<st c="37173">从</st> `<st c="37090">"improve"` <st c="37099">节点回到</st> `<st c="37117">"agent"` <st c="37124">节点添加一个边。这创建了一个循环，改进的问题被发送回代理进行</st> <st c="37253">进一步处理。</st>

1.  <st c="37272">我们现在准备好编译</st> `<st c="37301">图：</st>

    ```py
     graph = workflow.compile()
    ```

    <st c="37338">此行使用</st> `<st c="37383">workflow.compile</st>` <st c="37399">编译工作流程图，并将编译后的图赋值给</st> `<st c="37438">graph</st>` <st c="37443">变量，它现在代表了我们最初开始的</st> `<st c="37501">StateGraph</st>` <st c="37511">图实例的编译版本。</st>

1.  <st c="37543">我们已经在本章前面展示了此图的可视化，</st> *<st c="37645">图 12.1</st>**<st c="37654">，但如果你想要自己运行可视化，你可以使用</st> `<st c="37721">此代码：</st>`

    ```py
     from IPython.display import Image, display
    ```

    ```py
     try:
    ```

    ```py
     display(Image(graph.get_graph(
    ```

    ```py
     xray=True).draw_mermaid_png()))
    ```

    ```py
     except:
    ```

    ```py
     pass
    ```

    <st c="37855">我们可以使用</st> `<st c="37867">IPython</st>` <st c="37874">生成</st> `<st c="37887">此可视化。</st>

1.  <st c="37906">最后，我们将最终让我们的代理</st> <st c="37951">开始工作：</st>

    ```py
     import pprint
    ```

    ```py
     inputs = {
    ```

    ```py
     "messages": [
    ```

    ```py
     ("user", user_query),
    ```

    ```py
     ]
    ```

    ```py
     }
    ```

    <st c="38024">此行导入</st> `<st c="38041">pprint</st>` <st c="38047">模块，它提供了一个格式化和打印数据结构的 pretty-print 函数，使我们能够看到我们代理输出的更易读版本。</st> <st c="38210">然后我们定义一个名为</st> `<st c="38245">inputs</st>` <st c="38251">的字典，它表示工作流程图的初始输入。</st> <st c="38309">输入字典包含一个</st> `<st c="38342">"messages"</st>` <st c="38352">键，其中包含一个元组列表。</st> <st c="38380">在这种情况下，它有一个单个元组，</st> `<st c="38417">("user", user_query)</st>`<st c="38437">，其中</st> `<st c="38449">"user"</st>` <st c="38455">字符串表示消息发送者的角色（</st>`<st c="38506">user</st>`<st c="38511">）和</st> `<st c="38518">user_query</st>` <st c="38528">是用户的查询或问题。</st>

1.  <st c="38561">然后我们初始化一个名为</st> `<st c="38613">final_answer</st>` <st c="38625">的空字符串变量来存储工作流程</st> <st c="38665">生成的最终答案：</st>

    ```py
     final_answer = ''
    ```

1.  <st c="38696">然后我们使用图实例</st> <st c="38721">作为</st> <st c="38751">基础来启动我们的代理</st> <st c="38754">循环：</st>

    ```py
     for output in graph.stream(inputs):
    ```

    ```py
     for key, value in output.items():
    ```

    ```py
     pprint.pprint(f"Output from node '{key}':")
    ```

    ```py
     pprint.pprint("---")
    ```

    ```py
     pprint.pprint(value, indent=2, width=80,
    ```

    ```py
     depth=None)
    ```

    ```py
     final_answer = value
    ```

    <st c="38973">这使用</st> `<st c="39020">graph.stream(inputs)</st>`<st c="39040">中的输出启动一个双重循环。</st> <st c="39090">graph</st> <st c="39095">实例在处理输入时生成输出。</st> <st c="39133">`<st c="39137">graph.stream(inputs)</st>` <st c="39157">方法从</st> `<st c="39194">graph</st>` <st c="39199">实例执行中流式传输输出。</st>

    <st c="39219">在外层循环内部，它为两个变量启动另一个循环，</st> `<st c="39285">key</st>` <st c="39288">和</st> `<st c="39293">value</st>`<st c="39298">，代表在</st> `<st c="39340">output.items</st>` <st c="39352">变量中的键值对。</st> <st c="39363">这会遍历每个键值对，其中</st> `<st c="39423">key</st>` <st c="39426">变量代表节点名称，而</st> `<st c="39469">value</st>` <st c="39474">变量代表该节点生成的输出。</st> <st c="39530">这将使用</st> `<st c="39566">pprint.pprint(f"Output from node '{key}':")</st>` <st c="39609">来指示哪个节点生成了</st> `<st c="39643">输出。</st>

    <st c="39654">代码使用</st> `<st c="39703">pprint.pprint(value, indent=2, width=80, depth=None)</st>`<st c="39755">来美化打印值（输出）。</st> `<st c="39761">indent</st>` <st c="39767">参数指定缩进级别，</st> `<st c="39811">width</st>` <st c="39816">指定输出最大宽度，</st> `<st c="39864">depth</st>` <st c="39869">指定要打印的嵌套数据结构的最大深度（</st>`<st c="39934">None</st>` <st c="39939">表示</st> `<st c="39946">无限制）。</st>

    <st c="39956">它将值（输出）赋给</st> `<st c="39994">final_answer</st>` <st c="40006">变量，并在每次迭代中覆盖它。</st> <st c="40051">循环结束后，</st> `<st c="40072">final_answer</st>` <st c="40084">将包含工作流程中最后一个节点生成的输出。</st>

    <st c="40152">此代码的一个优点是它允许您看到</st> <st c="40160">图中每个节点生成的中间输出，并跟踪查询处理的进度。</st> <st c="40213">这些打印输出代表了代理在循环中做出决策时的“思考”。</st> <st c="40319">美化打印有助于格式化输出，使其更易于阅读。</st>

    当我们启动代理并开始看到输出时，我们可以看到有很多<st c="40486">事情在进行！</st>

    <st c="40574">我将截断大量的打印输出，但这将给你一个关于提供内容的</st> <st c="40653">概念：</st>

    ```py
     ---CALL AGENT---
    ```

    ```py
     "Output from node 'agent':"
    ```

    ```py
     '---'
    ```

    ```py
     { 'messages': [ AIMessage(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_46NqZuz3gN2F9IR5jq0MRdVm', 'function': {'arguments': '{"query":"Google\'s environmental initiatives"}', 'name': 'retrieve_google_environmental_question_answers'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls'}, id='run-eba27f1e-1c32-4ffc-a161-55a32d645498-0', tool_calls=[{'name': 'retrieve_google_environmental_question_answers', 'args': {'query': "Google's environmental initiatives"}, 'id': 'call_46NqZuz3gN2F9IR5jq0MRdVm'}])]}
    ```

    ```py
     '\n---\n'
    ```

    <st c="41280">这是我们的打印输出的第一部分。</st> <st c="41321">在这里，我们看到代理决定使用</st> `<st c="41367">retrieve_google_environmental_question_answers</st>` <st c="41413">工具。</st> <st c="41420">如果你还记得，这是我们定义检索器工具时给它起的基于文本的名称。</st> <st c="41516">选择得很好！</st>

1.  <st c="41528">接下来，代理将确定它认为检索到的文档是否相关：</st>

    ```py
     ---CHECK RELEVANCE---
    ```

    ```py
     ---DECISION: DOCS RELEVANT---
    ```

    <st c="41669">决定是它们是。</st> <st c="41701">再次，明智的思考，</st> <st c="41724">先生，</st> <st c="41728">代理。</st>

1.  <st c="41734">最后，我们看到代理正在查看的</st> <st c="41751">输出，这些数据是从</st> <st c="41806">PDF 文档和我们一直在使用的集成检索器中检索到的（这里有很多检索到的数据，所以我截断了大部分的实际</st> <st c="41938">内容）：</st>

    ```py
     "Output from node 'retrieve':"
    ```

    ```py
     '---'
    ```

    ```py
     { 'messages': [ ToolMessage(content='iMasons Climate AccordGoogle is a founding member and part of the governing body of the iMasons Climate Accord, a coalition united on carbon reduction in digital infrastructure.\nReFEDIn 2022, to activate industry-wide change…[TRUNCATED]', tool_call_id='call_46NqZuz3gN2F9IR5jq0MRdVm')]}
    ```

    ```py
     '\n---\n'
    ```

    <st c="42326">当你查看这部分的实际打印输出时，你会看到检索到的数据被连接在一起，并为我们的代理提供了大量和深入的数据，以便</st> <st c="42499">使用。</st>

1.  <st c="42509">在这个阶段，就像我们的原始 RAG 应用程序所做的那样，代理接受问题、检索到的数据，并根据我们给出的生成</st> <st c="42671">提示来制定响应：</st>

    ```py
     ---GENERATE---
    ```

    ```py
     "Output from node 'generate':"
    ```

    ```py
     '---'
    ```

    ```py
     { 'messages': [ 'Google has a comprehensive and multifaceted approach to '
    ```

    ```py
     'environmental sustainability, encompassing various '
    ```

    ```py
     'initiatives aimed at reducing carbon emissions, promoting'
    ```

    ```py
     'sustainable practices, and leveraging technology for '
    ```

    ```py
     "environmental benefits. Here are some key aspects of Google's "
    ```

    ```py
     'environmental initiatives:\n''\n'
    ```

    ```py
     '1\. **Carbon Reduction and Renewable Energy**…']}
    ```

    ```py
     '\n---\n'
    ```

    <st c="43146">我们在这里加入了一个机制，以便单独打印出最终消息，以便于阅读：</st>

    ```py
     final_answer['messages'][0]
    ```

    <st c="43262">这将打印</st> <st c="43279">以下内容：</st>

    ```py
    <st c="43288">"Google has a comprehensive and multifaceted approach to environmental sustainability, encompassing various initiatives aimed at reducing carbon emissions, promoting sustainable practices, and leveraging technology for environmental benefits.</st> <st c="43532">Here are some key aspects of Google's environmental initiatives:\n\n1\.</st> <st c="43603">**Carbon Reduction and Renewable Energy**:\n   - **iMasons Climate Accord**: Google is a founding member and part of the governing body of this coalition focused on reducing carbon emissions in digital infrastructure.\n   - **Net-Zero Carbon**: Google is committed to operating sustainably with a focus on achieving net-zero carbon emissions.</st> <st c="43942">This includes investments in carbon-free energy and energy-efficient facilities, such as their all-electric, net water-positive Bay View campus..."</st>
    ```

<st c="44089">这就是我们代理的全部输出！</st>

# <st c="44127">摘要</st>

<st c="44135">在本章中，我们探讨了如何将 AI 代理和 LangGraph 结合起来创建更强大和复杂的 RAG 应用程序。</st> <st c="44269">我们了解到，AI 代理本质上是一个具有循环的 LLM，允许它进行推理并将任务分解成更简单的步骤，从而提高在复杂 RAG 任务中成功的可能性。</st> <st c="44452">LangGraph，建立在 LCEL 之上的扩展，为构建可组合和可定制的代理工作负载提供支持，使开发者能够使用基于</st> <st c="44625">图的方法来编排代理。</st>

<st c="44646">我们深入探讨了 AI 代理和 RAG 集成的根本，讨论了代理可以用来执行任务的工具的概念，以及 LangGraph 的</st> `<st c="44803">AgentState</st>` <st c="44813">类如何跟踪代理随时间的状态。</st> <st c="44861">我们还涵盖了图论的核心概念，包括节点、边和条件边，这对于理解 LangGraph 如何工作至关重要。</st>

在代码实验室中，我们为我们的 RAG 应用构建了一个 LangGraph 检索代理，展示了如何创建工具、定义代理状态、设置提示以及使用 LangGraph 建立循环图。<st c="45221">我们看到了代理如何利用其推理能力来确定使用哪些工具、如何使用它们以及提供什么数据，最终为</st> <st c="45405">用户的问题提供更全面的回答。</st>

展望未来，下一章将重点介绍如何使用提示工程来改进<st c="45421">RAG 应用</st>。
