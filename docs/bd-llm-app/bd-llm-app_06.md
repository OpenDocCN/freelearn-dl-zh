

# 第六章：构建对话应用

通过本章，我们开始了本书的动手实践部分，我们的第一个基于 LLM 的应用的具体实现。在本章中，我们将逐步实现一个对话应用，使用 LangChain 及其组件，基于前几章所获得的知识。到本章结束时，你将能够仅用几行代码设置自己的对话应用项目。

我们将涵盖以下关键主题：

+   配置简单聊天机器人的模式

+   添加记忆组件

+   添加非参数化知识

+   添加工具并使聊天机器人“具有代理性”

+   使用 Streamlit 开发前端

# 技术要求

要完成本章的任务，你需要以下内容：

+   Hugging Face 账户和用户访问令牌。

+   OpenAI 账户和用户访问令牌。

+   Python 3.7.1 或更高版本。

+   Python 包 – 确保已安装以下 Python 包：`langchain`、`python-dotenv`、`huggingface_hub`、`streamlit`、`openai`、`pypdf`、`tiktoken`、`faiss-cpu`和`google-search-results`。它们可以通过在终端中运行`pip install`轻松安装。

你可以在本书的 GitHub 仓库中找到本章的代码：`github.com/PacktPublishing/Building-LLM-Powered-Applications`。

# 开始使用对话应用

对话应用是一种可以与用户使用自然语言进行交互的软件。它可以用于各种目的，例如提供信息、协助、娱乐或交易。一般来说，对话应用可以使用不同的通信模式，如文本、语音、图形，甚至触摸。对话应用还可以使用不同的平台，如消息应用、网站、移动设备或智能扬声器。

今天，由于 LLMs，对话应用正被提升到新的水平。让我们看看它们提供的某些好处：

+   不仅 LLMs 提供了新的自然语言交互水平，而且它们还可以使应用能够根据用户的偏好执行基于最佳响应的推理。

+   正如我们在前面的章节中看到的，LLMs 可以利用它们的参数化知识，但还通过嵌入和插件丰富了非参数化知识。

+   最后，LLMs 还能够通过不同类型的记忆跟踪对话。

以下图像显示了对话机器人可能的结构：

![计算机程序图  自动生成描述](img/B21714_06_01.png)

图 6.1：对话机器人的示例架构

在本章中，我们将从头开始构建一个能够帮助用户规划假期的文本对话应用。我们将把这个应用称为 GlobeBotter。我们将逐步添加复杂性，使应用对最终用户尽可能有趣。

因此，让我们从对话应用架构的基本原理开始。

## 创建一个简单的机器人

首先，让我们初始化我们的 LLM 并设置我们机器人的模式。模式指的是机器人能够接收的消息类型。在我们的例子中，我们将有三种类型的消息：

+   **系统消息**：我们给机器人的指示，以便它表现得像一个旅行助手。

+   **AI 消息**：LLM 生成的消息

+   **人类消息**：用户的查询

让我们从简单的配置开始：

```py
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chains import LLMChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI()
messages = [
    SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary."),
    HumanMessage(content="I'm going to Rome for 2 days, what can I visit?")] 
```

我们可以保存并按如下方式打印输出：

```py
output = chat(messages)
print(output.content) 
```

这里是输出：

```py
In Rome, there are many famous attractions to visit. Here's an optimized itinerary for your two-day trip:
Day 1:
 1\. Start your day by visiting the Colosseum, one of the world's most iconic ancient landmarks.
 2\. Next, explore the nearby Roman Forum, an ancient Roman marketplace.
 3\. Afterward, head to the Pantheon, a well-preserved Roman temple with a stunning dome.
4\. Take a stroll through the historic district of Trastevere, known for its charming streets and authentic Roman atmosphere.
5\. In the evening, visit the Trevi Fountain and toss a coin to ensure your return to Rome.
Day 2:
1\. Begin your day at Vatican City, the smallest independent state in the world. Visit St. Peter's Basilica and admire Michelangelo's masterpiece, the Sistine Chapel.
2\. Explore the Vatican Museums, home to an extensive collection of art and historical artifacts.
3\. Enjoy a leisurely walk along the Tiber River and cross over to the picturesque neighborhood of Castel Sant'Angelo.
4\. Visit the Spanish Steps, a popular meeting point with a beautiful view of the city.
5\. End your day by exploring the charming neighborhood of Piazza Navona, known for its baroque architecture and lively atmosphere.
Remember to check the opening hours and availability of tickets for the attractions in advance. Enjoy your trip to Rome! 
```

如你所见，模型在仅从我们这里获得一条信息——天数的情况下，非常擅长生成罗马的行程。

然而，我们可能还想与机器人进行交互，以便我们可以进一步优化行程，提供更多关于我们偏好和习惯的信息。为了实现这一点，我们需要向我们的机器人添加内存。

## 添加内存

由于我们正在创建一个使用相对简短消息的对话机器人，在这种情况下，一个`ConversationBufferMemory`可能是合适的。为了使配置更简单，让我们还初始化一个`ConversationChain`来结合 LLM 和内存组件。

让我们先初始化我们的内存和链（我保持`verbose = True`，这样你可以看到机器人正在跟踪之前的消息）：

```py
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat, verbose=True, memory=memory
) 
```

太好了，现在让我们与我们的机器人进行一些交互：

```py
conversation.run("Hi there!") 
```

以下是输出：

```py
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
Human: Hi there!
AI:
> Finished chain.
'Hello! How can I assist you today?' 
```

接下来，我们提供以下输入：

```py
conversation.run("what is the most iconic place in Rome?") 
```

这里是相应的输出：

```py
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
Human: Hi there!
AI: Hello! How can I assist you today?
Human: what is the most iconic place in Rome?
AI:
> Finished chain.
'The most iconic place in Rome is probably the Colosseum. It is a magnificent amphitheater that was built in the first century AD and is one of the most recognizable symbols of ancient Rome. The Colosseum was used for gladiatorial contests, public spectacles, and other events. Today, it is a major tourist attraction and a UNESCO World Heritage site.' 
```

如你所见，它正在跟踪之前的交互。让我们挑战它，并询问与之前上下文相关的问题：

```py
conversation.run("What kind of other events?") 
```

以下是接收到的输出：

```py
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
Human: Hi there!
AI: Hello! How can I assist you today?
Human: what is the most iconic place in Rome?
AI: The most iconic place in Rome is probably the Colosseum. It is a magnificent amphitheater that was built in the first century AD and is one of the most recognizable symbols of ancient Rome. The Colosseum was used for gladiatorial contests, public spectacles, and other events. Today, it is a major tourist attraction and a UNESCO World Heritage site.
Human: What kind of other events?
AI:
> Finished chain.
'Other events that took place at the Colosseum include mock sea battles, animal hunts, and reenactments of famous battles. The Colosseum was also used for executions and religious ceremonies. It was a versatile venue that could accommodate a variety of events and entertainments.' 
```

机器人能够理解我们的请求与其之前的回答相关。我们还可以使用`memory.load_memory_variables()`方法检索消息历史（你可以在 GitHub 仓库中看到完整的输出）。以下是输出片段：

```py
{'history': 'Human: Hi there!\nAI: Hello! How can I assist you today?\nHuman: what is the most iconic place in Rome?.... 
```

我没有在每个交互时运行`conversation.run`方法，而是编写了一个`while`循环来使其交互式。以下是对整个对话的快照（你可以在书籍的 GitHub 仓库中找到它）：

```py
while True:
    query = input('you: ')
    if query == 'q':
        break
    output = conversation({"input": query})
    print('User: ', query)
    print('AI system: ', output['response']) 
```

以下是从输出中截取的样本（你可以在书籍的 GitHub 仓库中找到完整的输出）：

```py
User:  hello
AI system:  Hello! How can I assist you today?
User:  I'm planning a 1-day trip in Venice. What should I visit?
AI system:  That sounds like a wonderful plan! In Venice, there are several must-visit attractions that you can explore in a day. Here's a suggested itinerary for your 1-day trip in Venice:
1\. St. Mark's Square (Piazza San Marco): [...] Enjoy your trip to Venice!
User:  thanks! I'm planning to be around also the next day, and I love hiking. Do you have any suggestion nearby?
AI system:  Certainly! If you enjoy hiking and want to explore the natural beauty around Venice, there are a few options you can consider for the next day:
1\. The Dolomites: [...]
User:  which one is closer to Milan?
AI system:  If you're looking for a hiking destination closer to Milan, the best option would be the Lombardy region [...] 
```

如你所见，现在人工智能助手能够跟踪整个对话。在下一节中，我们将添加另一层复杂性：一个外部知识库。

## 添加非参数化知识

想象一下，你希望你的 GlobeBotter 也能访问关于它行程的独家文档，这些行程不属于其参数化知识的一部分。

要这样做，我们可以将文档嵌入到 VectorDB 中，或者直接使用检索器来完成工作。在这种情况下，我们将使用特定链`ConversationalRetrievalChain`支持的向量存储检索器。此类链利用检索器在提供的知识库上，该知识库包含聊天历史，可以通过使用之前看到的所需类型的内存作为参数传递。

以此目标为前提，我们将使用从[`www.minube.net/guides/italy`](https://www.minube.net/guides/italy)下载的意大利旅行指南 PDF 样本。

以下 Python 代码展示了如何初始化我们需要的所有成分，它们是：

+   **文档加载器**：由于文档是 PDF 格式，我们将使用`PyPDFLoader`。

+   **文本分割器**：我们将使用`RecursiveCharacterTextSplitter`，它通过递归地查看字符来分割文本，以找到合适的一个。

+   **向量存储**：我们将使用`FAISS` VectorDB。

+   **内存**：我们将使用`ConversationBufferMemory`。

+   **LLMs**：我们将使用`gpt-3.5-turbo`模型进行对话。

+   **嵌入**：我们将使用`text-embedding-ada-002`。

让我们看看代码：

```py
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
llm = ChatOpenAI() 
```

现在我们与链进行交互：

```py
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=db.as_retriever(), memory=memory, verbose=True)
qa_chain.run({'question':'Give me some review about the Pantheon'}) 
```

以下为输出结果（我报告的是截断版本。您可以在书籍的 GitHub 仓库中查看完整的输出）：

```py
> Entering new StuffDocumentsChain chain...
> Entering new LLMChain chain...
Prompt after formatting:
System: Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
cafes in the square. The most famous are the Quadri and
Florian.
Piazza San Marco,
Venice
4
Historical Monuments
Pantheon
Miskita:
"Angelic and non-human design," was how
Michelangelo described the Pantheon 14 centuries after its
construction. The highlights are the gigantic dome, the upper
eye, the sheer size of the place, and the harmony of the
whole building. We visited with a Roman guide which is
...
> Finished chain.
'Miskita:\n"Angelic and non-human design," was how Michelangelo described the Pantheon 14 centuries after its construction. The highlights 
```

注意，默认情况下，`ConversationalRetrievalChain`使用一个名为`CONDENSE_QUESTION_PROMPT`的提示模板，它将最后用户的查询与聊天历史合并，因此结果只有一个查询传递给检索器。如果您想传递自定义提示，您可以在`ConversationalRetrievalChain.from_llm`模块中使用`condense_question_prompt`参数。

尽管机器人能够根据文档提供答案，但我们仍然存在限制。实际上，在这种配置下，我们的 GlobeBotter 将只查看提供的文档，但如果我们希望它也能使用其参数化知识怎么办？例如，我们可能希望机器人能够理解它是否可以与提供的文档集成，或者简单地*自由地*回答。为此，我们需要使我们的 GlobeBotter*具有代理性*，这意味着我们希望利用 LLM 的推理能力来协调和调用可用的工具，而不是遵循固定的顺序，而是根据用户的查询采取最佳方法。

要这样做，我们将使用两个主要组件：

+   `create_retriever_tool`：此方法创建一个自定义工具，作为代理的检索器。它需要一个数据库来检索，一个名称和一个简短描述，以便模型能够理解何时使用它。

+   `create_conversational_retrieval_agent`：此方法初始化一个配置为与检索器和聊天模型一起工作的对话代理。它需要一个 LLM、一个工具列表（在我们的情况下，是检索器）和一个内存键来跟踪之前的聊天历史。

以下代码说明了如何初始化代理：

```py
from langchain.agents.agent_toolkits import create_retriever_tool
tool = create_retriever_tool(
    db.as_retriever(),
    "italy_travel",
    "Searches and returns documents regarding Italy."
)
tools = [tool]
memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature = 0)
agent_executor = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True) 
```

太好了，现在让我们看看代理在两个不同问题上的思考过程（我将只报告思维链并截断输出，但你可以找到整个代码在 GitHub 仓库中）：

```py
agent_executor({"input": "Tell me something about Pantheon"}) 
```

这里是输出结果：

```py
> Entering new AgentExecutor chain...
Invoking: `italy_travel` with `Pantheon`
[Document(page_content='cafes in the square. The most famous are the Quadri and\nFlorian. […]
> Finished chain. 
```

现在我们用一个与文档无关的问题来尝试：

```py
output = agent_executor({"input": "what can I visit in India in 3 days?"}) 
```

我们收到的以下输出是：

```py
> Entering new AgentExecutor chain...
In India, there are numerous incredible places to visit, each with its own unique attractions and cultural experiences. While three days is a relatively short time to explore such a vast and diverse country, here are a few suggestions for places you can visit:
1\. Delhi: Start your trip in the capital city of India, Delhi. […]
> Finished chain. 
```

如您所见，当我问代理有关意大利的问题时，它立即调用了提供的文档，而在上一个问题中并没有这样做。

我们最不想添加到我们的 GlobeBotter 中的是网络导航的能力，因为作为旅行者，我们希望了解我们即将前往的国家最新的信息。让我们使用 LangChain 的工具来实现它。

## 添加外部工具

我们将要添加的工具是 Google SerpApi 工具，这样我们的机器人就能够上网导航了。

**注意**

SerpApi 是一个实时 API，旨在访问 Google 搜索结果。它通过处理诸如管理代理、解决 CAPTCHA 和从搜索引擎结果页面解析结构化数据等复杂性，简化了数据抓取的过程。

LangChain 提供了一个预构建的工具，用于包装 SerpApi，使其更容易集成到你的代理中。要启用 SerpApi，你需要登录到 [`serpapi.com/users/sign_up`](https://serpapi.com/users/sign_up)，然后转到“API key”选项卡下的仪表板。

由于我们不希望我们的 GlobeBotter 只关注网络，我们将添加 SerpApi 工具到之前的工具中，这样代理就能够选择最有用的工具来回答问题——或者如果不需要，就不使用任何工具。

让我们初始化我们的工具和代理（你已经在 *第五章* 中了解了这一点和其他 LangChain 组件）：

```py
from langchain import SerpAPIWrapper
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["SERPAPI_API_KEY"]
search = SerpAPIWrapper()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
    create_retriever_tool(
        db.as_retriever(),
        "italy_travel",
        "Searches and returns documents regarding Italy."
    )
    ]
agent_executor = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True) 
```

太好了，现在让我们用三个不同的问题来测试它（这里，输出结果再次被截断）：

+   “在印度三天我能参观哪些地方？”

    ```py
    > Entering new AgentExecutor chain...
    India is a vast and diverse country with numerous attractions to explore. While it may be challenging to cover all the highlights in just three days, here are some popular destinations that you can consider visiting:
    1\. Delhi: Start your trip in the capital city of India, Delhi. […]
    > Finished chain. 
    ```

在这种情况下，模型不需要外部知识来回答问题，因此它是在不调用任何工具的情况下进行响应的。

+   “德里现在的天气怎么样？”

    ```py
    > Entering new AgentExecutor chain...
    Invoking: `Search` with `{'query': 'current weather in Delhi'}`
    Current Weather · 95°F Mostly sunny · RealFeel® 105°. Very Hot. RealFeel Guide. Very Hot. 101° to 107°. Caution advised. Danger of dehydration, heat stroke, heat ...The current weather in Delhi is 95°F (35°C) with mostly sunny conditions. The RealFeel® temperature is 105°F (41°C), indicating that it feels very hot. Caution is advised as there is a danger of dehydration, heat stroke, and heat-related issues. It is important to stay hydrated and take necessary precautions if you are in Delhi or planning to visit.
    > Finished chain. 
    ```

注意代理是如何调用搜索工具的；这是由于底层 gpt-3.5-turbo 模型的推理能力，它捕捉用户的意图并动态理解使用哪个工具来完成请求。

+   “我要去意大利旅行。你能给我一些建议，去哪些主要景点参观吗？”

    ```py
    > Entering new AgentExecutor chain...
    Invoking: `italy_travel` with `{'query': 'main attractions in Italy'}`
    [Document(page_content='ITALY\nMINUBE TRAVEL GUIDE\nThe best must-see places for your travels, […]
    Here are some suggestions for main attractions in Italy:
    1\. Parco Sempione, Milan: This is one of the most important parks in Milan. It offers a green space in the city where you can relax, workout, or take a leisurely walk. […]
    > Finished chain. 
    ```

注意观察代理是如何调用文档检索器来提供前面的输出的。

总体来说，我们的 GlobeBotter 现在能够提供最新的信息，以及从精选文档中检索特定知识。下一步将是构建前端。我们将通过构建一个使用 Streamlit 的网络应用来实现这一点。

# 使用 Streamlit 开发前端

Streamlit 是一个 Python 库，允许您创建和共享网络应用。它设计得易于使用且快速，无需任何前端经验或知识。您可以使用纯 Python 编写您的应用，使用简单的命令添加小部件、图表、表格和其他元素。

除了其原生功能外，2023 年 7 月，Streamlit 宣布了与 LangChain 的初始集成及其未来计划。这一初始集成的核心是使构建对话应用的 GUI 更容易，以及展示 LangChain 代理在生成最终响应之前所采取的所有步骤。

为了实现这一目标，Streamlit 引入的主要模块是 Streamlit 回调处理程序。该模块提供了一个名为 `StreamlitCallbackHandler` 的类，该类实现了 LangChain 的 `BaseCallbackHandler` 接口。这个类可以处理在 LangChain 管道执行过程中发生的各种事件，例如工具开始、工具结束、工具错误、LLM 令牌、代理动作、代理完成等。

该类还可以创建和更新 Streamlit 元素，例如容器、展开器、文本、进度条等，以便以用户友好的方式显示管道的输出。您可以使用 Streamlit 回调处理程序创建展示 LangChain 功能并可通过自然语言与用户交互的 Streamlit 应用。例如，您可以创建一个应用，该应用接受用户提示并通过使用不同工具和模型的代理来生成响应。您可以使用 Streamlit 回调处理程序实时显示代理的思考过程和每个工具的结果。

要开始构建您的应用，您需要创建一个 `.py` 文件，通过在终端中运行 `streamlit run file.py` 来运行。在我们的案例中，该文件将被命名为 `globebotter.py`。

以下是该应用的主要构建块：

1.  设置网页的配置：

    ```py
    import streamlit as st
    st.set_page_config(page_title="GlobeBotter", page_icon="![](img/Globe.png)")
    st.header('![](img/Globe.png) Welcome to Globebotter, your travel assistant with Internet access. What are you planning for your next trip?') 
    ```

1.  初始化我们需要的 LangChain 核心组件。代码与上一节中的相同，所以在这里我只分享初始化代码，而不包括所有初步步骤：

    ```py
    search = SerpAPIWrapper()
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
    raw_documents = PyPDFLoader('italy_travel.pdf').load()
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="output"
    )
    llm = ChatOpenAI()
    tools = [
        Tool.from_function(
            func=search.run,
            name="Search",
            description="useful for when you need to answer questions about current events"
        ),
        create_retriever_tool(
            db.as_retriever(),
            "italy_travel",
            "Searches and returns documents regarding Italy."
        )
        ]
    agent = create_conversational_retrieval_agent(llm, tools, memory_key='chat_history', verbose=True) 
    ```

1.  为用户设置带有占位符问题的输入框：

    ```py
    user_query = st.text_input(
        "**Where are you planning your next vacation?**",
        placeholder="Ask me anything!"
    ) 
    ```

1.  设置 Streamlit 的会话状态。会话状态是一种在每次用户会话之间共享变量的方式。除了存储和持久化状态的能力外，Streamlit 还公开了使用回调操作状态的能力。会话状态在多页应用内的应用之间也持续存在。您可以使用会话状态 API 在会话状态中初始化、读取、更新和删除变量。在我们的 GlobeBotter 案例中，我们想要两个主要状态：`messages` 和 `memory`：

    ```py
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "memory" not in st.session_state:
        st.session_state['memory'] = memory 
    ```

1.  确保显示整个对话。为此，我创建了一个循环，遍历存储在 `st.session_state["messages"]` 中的消息列表。对于每条消息，它创建一个名为 `st.chat_message` 的 Streamlit 元素，以美观的格式显示聊天消息：

    ```py
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"]) 
    ```

1.  配置 AI 助手在接收到用户查询时做出响应。在这个第一个例子中，我们将保持整个链在屏幕上可见并打印出来：

    ```py
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            response = agent(user_query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response) 
    ```

1.  最后，添加一个按钮来清除对话历史并从头开始：

    ```py
    if st.sidebar.button("Reset chat history"):
        st.session_state.messages = [] 
    ```

最终产品看起来如下：

![计算机截图  自动生成的描述](img/B21714_06_02.png)

图 6.2：带有 Streamlit 的 GlobeBotter 前端

从展开器中，我们可以看到代理使用了`Search`工具（由 SerpApi 提供）。我们还可以展开`chat_history`或`intermediate_steps`，如下所示：

![计算机截图  自动生成的描述](img/B21714_06_03.png)

图 6.3：Streamlit 展开器的示例

当然，我们也可以选择只显示输出而不是整个思维链，通过在代码中指定只返回`response['output']`来实现。您可以在本书的 GitHub 仓库中查看整个代码。

在我们结束之前，让我们讨论一下如何在用户与聊天机器人交互时提供流式体验。您可以在 Streamlit 应用中利用`BaseCallbackHandler`类创建自定义回调处理程序：

```py
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
import streamlit as st
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text) 
```

`StreamHandler`旨在捕获和显示在指定容器中的流式数据，如文本或其他内容。然后，您可以在 Streamlit 应用中使用它，确保在初始化 OpenAI LLM 时设置`streaming=True`。

```py
 with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content)) 
```

您可以在 LangChain 的 GitHub 仓库中查看原始代码。[`github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_streaming.py`](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_streaming.py)。

# 摘要

在本章中，我们探讨了对话应用的端到端实现，利用 LangChain 的模块并逐步增加复杂层次。我们从没有记忆的简单聊天机器人开始，然后转向更复杂的系统，这些系统能够追踪过去的交互。我们还看到了如何通过外部工具将非参数化知识添加到我们的应用中，使其更加“智能”，从而能够根据用户的查询确定使用哪个工具。最后，我们介绍了 Streamlit 作为前端框架来构建 GlobeBotter 的 Web 应用。

在下一章中，我们将关注一个更具体的领域，其中 LLMs 能够增加价值并展示新兴行为，即推荐系统。

# 参考文献

+   上下文感知聊天机器人的示例。[`github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py`](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/2_%E2%AD%90_context_aware_chatbot.py)

+   AI 旅行助手的知识库。[`www.minube.net/guides/italy`](https://www.minube.net/guides/italy)

+   LangChain 仓库。[`github.com/langchain-ai`](https://github.com/langchain-ai )

# 加入我们的 Discord 社区

加入我们社区的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/llm`](https://packt.link/llm)

![](img/QR_Code214329708533108046.png)
