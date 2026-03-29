# 17

# 基于 RAG 的代码代理记忆

在探索了代理记忆的理论基础*第十六章*之后，我们现在将转向其实际应用。本章介绍了三个专注于代码实验室，展示了如何使用 **语言代理的认知架构**（**CoALA**）框架构建具有记忆功能的代理。我们将实现工作记忆、情景记忆、语义记忆和程序记忆系统，使代理能够维持上下文、回忆过去经历，并在时间上积累知识。

本章我们将涵盖以下内容：

+   **代码实验室 17.1** – 设置初始代理

+   **代码实验室 17.2** – 编写情景记忆组件

+   **代码实验室 17.3** – 编写语义记忆组件

每个代码实验室都是基于前一个实验室构建的，逐步构建一个复杂的记忆系统。我们将从一个最小的 RAG 代理基础开始，然后系统地添加情景记忆以回忆对话和语义记忆以提取知识。到本章结束时，你将拥有一个能够记住过去交互并构建持久知识库的代理，这对于创建真正智能的 AI 系统是基本能力。让我们首先设置我们的基础代理架构，这将成为所有记忆增强的基础。

# 技术要求

要完成本章的动手练习，你需要以下软件和资源：

+   **软件要求**：

    +   **Python 3.8 或更高版本**：这是运行代码实验室所必需的

    +   **pip 软件包管理器**：这是安装 Python 依赖项所必需的

    +   **OpenAI API 访问**：您需要一个 OpenAI API 密钥用于 GPT-4 和嵌入

    +   **文本编辑器或 IDE**：VS Code、PyCharm、Jupyter Notebook 或类似软件是编写和运行 Python 代码所必需的

    +   **Git（可选）**：这是克隆本书 GitHub 仓库所必需的

+   **硬件要求**：

    +   **最小 8 GB RAM**（推荐 16 GB 以获得与向量数据库的流畅性能）

    +   **2 GB 空闲磁盘空间** 用于依赖项、向量存储和持久性内存存储

    +   **互联网连接** 用于下载软件包和访问 OpenAI API

+   **OpenAI API 密钥**：用于 GPT 模型访问和嵌入生成：

    +   将此安全地存储在 `env.txt` 文件中，作为 `OPENAI_API_KEY=your_key_here`

    +   **注意**：切勿将 API 密钥提交到版本控制

+   **章节资源**:

    +   **GitHub 仓库**：[`github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_17`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_17)

    +   **完整代码文件**：所有三个代码实验室（*17.1*、*17.2* 和 *17.3*）均可在本书的 GitHub 仓库中找到以供参考

完整的代码，可在本书的 GitHub 仓库中找到，如果需要在任何阶段验证您的实现，可以作为参考。

# 代码实验室 17.1 – 设置初始智能体

我们将首先建立一个 RAG 智能体的最小版本，它将作为后续实验室中添加记忆能力的基石。你可能已经认识这段代码；这是我们在*第十二章*中构建的智能体的简化版本！

## 第 1 步 – 安装所需的包

我们将首先安装必要的库，以便我们构建我们的记忆增强智能体：

```py
!pip install langchain
!pip install langgraph
!pip install langchain-openai
!pip install chromadb
!pip install python-dotenv 
```

我们在第十二章中介绍了这些安装，但这里有一个快速提醒：

+   LangChain 提供 LLM 编排

+   LangGraph 处理有状态的工作流程

+   ChromaDB 将存储我们的向量嵌入以进行记忆检索

+   `dotenv`包安全地管理我们的 API 密钥

现在，让我们导入这些库并设置我们的环境变量。

## 第 2 步 – 导入库和配置环境

接下来，我们需要导入所需的模块并设置 OpenAI 和其他服务的 API 密钥：

```py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
load_dotenv(dotenv_path='env.txt')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
embeddings = OpenAIEmbeddings() 
```

这是这个设置所完成的：

+   导入建立了我们与 LLM 交互、状态管理和向量存储的核心依赖

+   我们初始化`gpt-4.1-mini`以实现高效的推理，并使用 OpenAI 嵌入进行语义相似度比较

+   `temperature=0`确保了一致、确定的响应

+   API 密钥从环境变量中安全加载

我们的依赖项已经准备好了，现在让我们构建一个基本的智能体结构，我们将通过添加记忆能力来增强它。

## 第 3 步 – 构建基本智能体

由于我们在*第十二章*中介绍了智能体构建，我们将把设置合并为一个单一流线化的实现，以便进行记忆增强：

```py
# Define Agent State with memory placeholders
class AgentState(TypedDict):
    """State container for agent memory and messages"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    working_memory: dict # Short-term context
    episodic_recall: list # Retrieved past experiences
    semantic_facts: dict # Retrieved knowledge
# Initialize vector store for future memory storage
vector_store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./memory_store"
)
# Create base prompt and chain
base_prompt = PromptTemplate.from_template("""
    You are a helpful assistant with memory capabilities.
    Current conversation: {messages}
    Please respond to the latest message.
""")
output_parser = StrOutputParser()
# Define agent node
def agent_node(state: AgentState) -&gt; dict:
    """Core agent logic - processes messages and generates responses"""
    messages = state["messages"][-5:] if state["messages"] else []
    formatted_messages = "\n".join([
         f"{msg.type}: {msg.content}"
         for msg in messages
    ])
chain = base_prompt | llm | output_parser
    response = chain.invoke({"messages": formatted_messages})
    return {"messages": [("assistant", response)]}
# Build and compile the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile() 
```

这段代码建立了基础：

+   `AgentState`类包括工作、事件和语义记忆的占位符

+   ChromaDB 通过`persist_directory`提供持久的向量存储，确保记忆在会话之间持续存在

+   智能体维持一个五条消息的工作记忆窗口

+   图结构目前是最小化的，但已准备好添加记忆节点

现在，在我们添加记忆功能之前，让我们验证我们的基本智能体是否正常工作。

## 第 4 步 – 测试基本智能体

让我们确认我们的智能体可以处理查询并维持对话状态：

```py
test_input = {
    "messages": [("user", "Hello! What's the capital of France?")]
}
result = app.invoke(test_input)
print(result["messages"][-1].content) 
```

这是预期的输出：

```py
The capital of France is Paris. 
```

这个测试演示了我们所取得的成果：

+   智能体成功处理用户查询并生成适当的响应

+   消息状态通过 LangGraph 工作流程得到妥善管理

+   我们的基础很稳固，但目前缺乏记忆功能，因为智能体无法回忆之前的对话或存储学习到的知识

在这个最小基础上，我们有了一个干净的基础，可以添加复杂的记忆组件。智能体目前没有长期记忆，但我们的状态结构和向量存储已准备好支持我们在以下实验室中实现的事件和语义记忆系统。

在 *代码实验室 17.2* *– 编写情景记忆组件* 中，我们将添加情景记忆功能，使代理能够存储和检索过去的对话。这将把我们的无状态助手转变为能够在会话之间保持上下文并回忆相关过去交互的助手。

**技术提示**

我们将在每个以下代码实验室中放置来自 *代码实验室 17.1* *– 设置初始代理* 的代理代码，但在此处不会重复它们。您将在查看本书 GitHub 仓库中的代码时看到这一点。相反，我们将展示我们对安装和导入部分所做的任何更改，之后我们将继续进行下一步，假设代理代码已经就绪。

# 代码实验室 17.2 – 编写情景记忆组件

在这个实验室中，我们将通过情景记忆增强我们的代理，使其能够存储和检索过去的对话经验。

## 第一步 – 导入用于情景记忆的额外依赖项

我们需要导入额外的模块来处理时间戳和管理情景记忆存储：

```py
from datetime import datetime
from typing import List
from langchain.schema import Document 
```

这些导入增加了时间跟踪和文档处理能力，这是情景记忆所必需的。`datetime` 模块将为我们的记忆添加时间戳，而 `Document` 架构将结构化我们存储的情景以实现高效的检索。现在，让我们创建情景记忆存储函数。

## 第二步 – 创建情景记忆存储函数

让我们构建一些函数，用于存储带有元数据的对话片段，以便将来检索：

```py
def store_episodic_memory(
    vector_store, conversation_id: str, messages: List,
    summary: str = None
):
    """Store a conversation episode in vector memory"""
    if not summary and messages:
        summary = f"Conversation about: {messages[0].content[:100]}..."
    doc = Document(
        page_content="\n".join(
            [f"{msg.type}: {msg.content}" for msg in messages]
        ),
        metadata={
            "type": "episodic", "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
        },
    )
    vector_store.add_documents([doc])
    return conversation_id
def retrieve_episodic_memories(vector_store, query: str, k: int = 3):
    """Retrieve relevant past conversation episodes"""
    return vector_store.similarity_search(
        query, k=k, filter={"type": "episodic"}
    ) 
```

这些函数的作用如下：

+   `store_episodic_memory` 捕获整个对话片段及其时间戳和元数据

+   每个情景都获得一个唯一的 `conversation_id` 以进行跟踪；`retrieve_episodic_memories` 使用语义相似性来查找相关的过去对话

+   过滤器确保我们只检索情景记忆，而不是其他数据类型

存储和检索准备就绪后，让我们创建一个具有记忆意识的代理节点。

## 第三步 – 增强代理节点以实现情景回忆

让我们修改我们的代理，使其搜索并利用相关的过去对话：

```py
def agent_with_episodic_memory(state: AgentState) -> dict:
    """Agent that retrieves and uses episodic memories"""
    messages = state.get("messages", [])
    # Retrieve relevant memories
    past_episodes = (
        retrieve_episodic_memories(
            vector_store, messages[-1].content, k=2
        )
        if messages else []
    )
    episodic_context = (
        "Relevant past conversations:\n"
        + "\n".join(
            f"\n[{ep.metadata.get('timestamp', 'Unknown')}]:\n"
            f"{ep.page_content[:200]}..."
            for ep in past_episodes
        )
        if past_episodes
        else ""
    )
    # Generate response
    response = (
        PromptTemplate.from_template(
            """
            You are a helpful assistant with episodic memory of past conversations
            {episodic_context}
            Current conversation: {messages}
            Please respond to the latest message, utilizing relevant past conversations if helpful.
            """
        )
        | llm
        | output_parser
    ).invoke({
        "episodic_context": episodic_context,
        "messages": (
            "\n".join(f"{m.type}: {m.content}" for m in messages[-5:])
            if messages else ""
        )
    })
    return {
        "messages": [("assistant", response)],
        "episodic_recall": past_episodes
    } 
```

这个增强的节点实现了情景回忆：

+   它根据当前查询搜索相关的过去对话

+   检索到的情景以时间戳格式化，以提供时间上下文

+   提示现在包括情景记忆和当前对话

+   状态跟踪哪些记忆被回忆，以提高透明度

现在，让我们创建一个包括对话后记忆存储的工作流程。

## 第四步 – 构建一个具有记忆的工作流程

我们需要添加一个记忆存储步骤来持久化对话以供将来回忆：

```py
def store_conversation_node(state: AgentState) -> dict:
    """Store the current conversation as an episodic memory"""
    messages = state.get("messages", [])
    # Only store meaningful conversations
    if len(messages) >= 2:
        store_episodic_memory(
            vector_store, f"conv_{datetime.now().timestamp()}", messages
        )
    return {"working_memory": {"stored": True}}
# Create workflow with episodic memory
memory_workflow = StateGraph(AgentState)
memory_workflow.add_node("recall_and_respond",agent_with_episodic_memory)
memory_workflow.add_node("store_memory",store_conversation_node)
memory_workflow.set_entry_point("recall_and_respond")
memory_workflow.add_edge("recall_and_respond", "store_memory")
memory_workflow.add_edge("store_memory", END)
memory_app = memory_workflow.compile() 
```

工作流程现在包括两个阶段：

+   `recall_and_respond` 检索相关记忆并生成响应

+   `store_memory` 持久化对话以供将来检索

这创建了一个持续的学习循环，其中每次对话都丰富了代理的记忆。工作流程确保在成功交互后存储记忆。

现在，让我们用多个对话来测试我们的情景记忆能力。

## 第 5 步 – 测试我们的情景记忆功能

让我们验证代理能否存储对话并在未来的交互中回忆它们：

```py
result_1 = memory_app.invoke({
    "messages": [
        ("user",
        "I'm planning a trip to Paris next month. Any recommendations?")
    ]
})
print("First conversation response:")
print(result_1["messages"][-1].content)
print("\n" + "=" * 50 + "\n")
# Second conversation - should recall the Paris discussion
result_2 = memory_app.invoke({
    "messages": [("user", "What are some good restaurants in Paris?")]
})
print("Second conversation response (with episodic recall):")
print(result_2["messages"][-1].content)
if result_2.get("episodic_recall"):
    print(f"\nRecalled {len(result_2['episodic_recall'])} relevant memories") 
```

这是预期输出的前几行：

```py
First conversation response:
That sounds wonderful! Paris is a fantastic city with so much to offer. Here are some recommendations for your trip:
1.-Must-See Attractions:
    a. Eiffel Tower: Iconic and a must-visit. Consider booking tickets in advance to avoid long lines.
    b. Louvre Museum: Home to the Mona Lisa and countless other masterpieces.
    c. Notre-Dame Cathedral: Beautiful Gothic architecture (check current status for restoration updates).
    d. Montmartre & Sacré-Cœur: Charming neighborhood with great views of the city.
    e. Champs-Élysées & Arc de Triomphe: Perfect for a stroll and some shopping... 
```

这个测试展示了我们的情景记忆实现：

+   关于巴黎旅行的第一次对话被存储为情景记忆

+   关于巴黎餐厅的第二次查询触发了相关过去对话的检索

+   代理成功引用了之前的讨论，显示了会话之间的连续性

+   情景记忆能够实现基于过去交互的上下文响应

这完成了这个情景记忆代码实验室。但这能应用于现实世界吗？

### 真实世界应用 – 客户服务机器人

我们刚刚实现的情景记忆能够提供强大的客户服务能力。想象一下，一个支持机器人能够记住上周有客户关于运输问题进行了咨询。当客户回来询问，“我的订单发生了什么？”时，机器人可以检索整个之前的对话，理解上下文而不让客户重复他们的故事。这显著提高了客户满意度并减少了解决时间。

在情景记忆功能正常工作后，我们的代理现在可以在会话之间维护对话历史。在*代码实验室 17.3 – 编写语义记忆组件*中，我们将添加语义记忆以使代理能够从对话中提取和存储事实知识，构建一个持久的知识库。

# 代码实验室 17.3 – 编写语义记忆组件

在这个实验室中，我们将添加语义记忆来从对话中提取和存储事实知识，从而构建一个持久的知识库。

## 第 1 步 – 导入语义记忆的依赖项

我们需要额外的导入来进行知识提取和结构化数据处理：

```py
# Step 1: Import Dependencies for Semantic Memory
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document 
```

这些导入使得从对话中提取结构化知识成为可能。`Document`类对于创建可存储的记忆条目至关重要。现在，让我们定义存储语义事实的结构。

## 第 2 步 – 定义语义事实结构

让我们创建用于提取和存储不同类型事实知识的模式：

```py
# Step 2: Define Semantic Fact Structure
class SemanticFact(BaseModel):
    """Structure for a semantic memory fact"""
    subject: str = Field(description="The entity or topic this fact is about")
    predicate: str = Field(description="The relationship or property")
    object: str = Field(description="The value or related entity")
    confidence: float = Field(description="Confidence score 0-1")
    source: str = Field(
        description="Source of this fact (user or assistant)")
def extract_semantic_facts(messages: List) -> List[SemanticFact]:
    """Extract factual knowledge from conversation"""
    prompt = PromptTemplate.from_template("""
Analyze this conversation and extract important factual statements.
Focus on concrete facts, preferences, and relationships mentioned.
Conversation: {conversation}
Extract facts in JSON format:
{"facts": [{"subject": "entity", "predicate": "relationship",
            "object": "value", "confidence": 0.0-1.0, "source": "user or assistant"}]}
Only extract clear, unambiguous facts. Output valid JSON only.
""")
    conversation_text = "\n".join(
        f"{msg[0]}: {msg[1]}" if isinstance(msg, tuple)
        else f"{msg.type}: {msg.content}"
        for msg in messages
    )
    try:
        result = (prompt | llm | JsonOutputParser()).invoke(
            {"conversation": conversation_text}
        )
        return [
            SemanticFact(**fact_dict)
            for fact_dict in result.get("facts", [])
        ]
    except Exception as e:
        print(f"Fact extraction error: {e}")
        return []
def store_semantic_facts(
    vector_store, facts: List[SemanticFact],
    user_id: str = "default"
):
    """Store semantic facts in vector memory"""
    documents = [
        Document(
            page_content=f"{fact.subject} {fact.predicate} {fact.object}",
            metadata={
                "type": "semantic", "user_id": user_id,
                "subject": fact.subject, "predicate": fact.predicate,
                "object": fact.object, "confidence": fact.confidence,
                "timestamp": datetime.now().isoformat(),
            },
        )
        for fact in facts
    ]
    if documents:
        vector_store.add_documents(documents)
    return len(documents) 
```

以下函数处理语义知识：

+   `semanticFact`定义了一个三联结构（*主语-谓语-宾语*）用于知识表示。

+   `extract_semantic_facts`通过分析对话来识别事实陈述。事实通过置信度评分并归因于其来源。

+   `store_semantic_facts`通过可搜索的元数据持久化提取的知识。

    **注意**：

    ChromaDB 元数据支持使用 MongoDB 风格的运算符在检索期间进行过滤。我们稍后将使用此功能与过滤器例如`{"type": {"$eq": "semantic"}}`来检索仅包含语义记忆。

现在，让我们为语义记忆创建检索函数。

## 第 3 步 – 构建语义记忆检索

让我们实现将查询我们的知识库以获取相关事实的函数：

```py
def retrieve_semantic_facts(
    vector_store, query: str, user_id: str = "default",k: int = 5
):
    """Retrieve relevant semantic facts for a query"""
    results = vector_store.similarity_search(
        query=query, k=k,
        filter={
            "$and": [
                {"type": {"$eq": "semantic"}},
                {"user_id": {"$eq": user_id}}
            ]
        }
    )
    return [
        {
            "subject": doc.metadata.get("subject"),
            "predicate": doc.metadata.get("predicate"),
            "object": doc.metadata.get("object"),
            "confidence": doc.metadata.get("confidence", 1.0),
        }
        for doc in results
    ]
def format_semantic_context(facts: List[Dict]) -> str:
    """Format semantic facts for inclusion in prompt"""
    if not facts:
        return "No relevant facts found."
    context = "Known facts:\n" + "\n".join(
        f"- {f['subject']} {f['predicate']} {f['object']}"
        for f in facts
        if f.get("confidence", 1.0) > 0.7
    )
    return context if context != "Known facts:\n" else "No relevant facts found." 
```

检索系统提供有针对性的事实访问：

+   ChromaDB 过滤器使用正确的`$and`和`$eq`运算符进行过滤

+   用户特定的过滤确保个性化知识检索

+   `format_semantic_context`为 LLM 创建可读的事实摘要

+   置信度阈值防止低质量的事实影响响应

现在，让我们将语义记忆集成到我们的代理工作流程中。

## 第 4 步 – 创建语义记忆感知代理

接下来，我们将增强代理，使其能够提取、存储和利用语义知识：

```py
def agent_with_semantic_memory(state: AgentState) -> dict:
    """Agent that uses and updates semantic memory"""
    messages = state.get("messages", [])
    user_id = state.get("user_id", "default")
    # Retrieve relevant semantic facts
    semantic_context = ""
    if messages:
        latest_query = (
            messages[-1][1]
            if isinstance(messages[-1], tuple)
            else messages[-1].content
        )
        facts = retrieve_semantic_facts(
            vector_store, latest_query, user_id=user_id, k=3
        )
        semantic_context = format_semantic_context(facts)
    # Generate response with semantic knowledge
    response = (
        PromptTemplate.from_template("""
You are a helpful assistant with semantic memory of facts and knowledge
{semantic_context}
Current conversation: {messages}
Respond using relevant facts from your semantic memory when applicable.
""")
        | llm
        | output_parser
    ).invoke({
        "semantic_context": semantic_context,
        "messages": (
            "\n".join(
                f"{m[0]}: {m[1]}" if isinstance(m, tuple)
                else f"{m.type}: {m.content}"
                for m in messages[-5:]
            )
            if messages else ""
        )
    })
    # Extract and store new facts
    semantic_facts = {}
    if messages:
        new_facts = extract_semantic_facts(
            messages + [("assistant", response)][-3:]
        )
        if new_facts:
            store_semantic_facts(vector_store, new_facts, user_id)
            semantic_facts = {"extracted": len(new_facts)}
    return {
        "messages": [("assistant", response)],
        "semantic_facts": semantic_facts
    }
# Create workflow with semantic memory
semantic_workflow = StateGraph(AgentState)
semantic_workflow.add_node("semantic_agent",agent_with_semantic_memory)
semantic_workflow.set_entry_point("semantic_agent")
semantic_workflow.add_edge("semantic_agent", END)
semantic_app = semantic_workflow.compile() 
```

这个语义感知代理实现了知识管理：

+   它在生成响应之前检索相关事实

+   新的事实会自动从对话中提取

+   系统为每个用户维护一个不断增长的知识库

+   响应基于存储的语义知识

现在，让我们测试语义记忆能力。

## 第 5 步 – 测试语义记忆功能

让我们验证代理是否可以在对话中提取、存储和利用事实知识：

```py
# Step 5: Test Semantic Memory Functionality
def print_response(result, title):
    print(f"{title}:")
    msg = result["messages"][-1]
    print(msg[1] if isinstance(msg, tuple) else msg.content)
    if result.get("semantic_facts", {}).get("extracted"):
        print(f"\nExtracted {result['semantic_facts']['extracted']} facts")
    print("\n" + "=" * 50 + "\n")
# Test conversations (shortened format)
test_cases = [
    ("First conversation response", {
        "messages": [("user",
            "I'm John Smith, a software engineer at TechCorp. "
            "I prefer Python and I'm allergic to shellfish."
        )],
        "user_id": "john_smith"
    }),
    ("Second conversation response (using semantic memory)", {
        "messages": [("user",
            "Can you recommend a programming language for a new web API project?"
        )],
        "user_id": "john_smith"
    }),
    ("Third conversation response (using allergy information)", {
        "messages": [("user",
            "What restaurants would you recommend for a business dinner?"
        )],
        "user_id": "john_smith"
    })
]
for title, conversation in test_cases:
    print_response(semantic_app.invoke(conversation), title) 
```

这是预期的输出：

```py
First conversation response:
Hello John Smith! It's great to meet a fellow software engineer who prefers Python for backend development. If you need any help or have questions related to Python or backend development, feel free to ask! Also, I'll keep in mind that you're allergic to shellfish.
Extracted 4 facts
==================================================
Second conversation response (using semantic memory):
Since John Smith prefers Python for backend development, I would recommend using Python for your new web API project. Frameworks like Django and Flask are both well-suited for building web APIs efficiently in Python. Depending on your project's complexity, Django offers a more full-featured approach, while Flask provides more flexibility and simplicity.
Extracted 6 facts
==================================================
Third conversation response (using allergy information):
For a business dinner, I recommend choosing a restaurant with a quiet atmosphere to facilitate conversation. It's also important that the restaurant offers a variety of non-shellfish options to accommodate different dietary preferences. Since business dinners may involve discussions about backend development or other technical topics, a comfortable and distraction-free environment will be beneficial. If you have a specific cuisine or location in mind, I can help suggest some suitable restaurants.
Extracted 5 facts
================================================== 
```

这个测试展示了我们的语义记忆实现：

+   代理成功地从第一次对话中提取了事实（姓名、工作、偏好、过敏）

+   在第二次对话中，代理检索并使用 Python 偏好来做出相关推荐

+   第三次对话显示了代理记住海鲜过敏，以提供安全意识建议

正如我们所看到的，语义记忆使基于积累的知识进行个性化、情境适当的响应成为可能。

这就结束了这个语义记忆代码实验室。然而，在我们结束之前，就像我们为事件记忆代码实验室所做的那样，让我们谈谈这在现实世界中的应用。

### 真实世界的应用 – 教育辅导

我们刚刚构建的语义记忆系统非常适合教育应用。辅导代理可以提取和存储有关学生知识差距（对二次方程的困扰）、学习偏好（偏好视觉解释）和进展里程碑（3 月 15 日掌握了导数）的事实。随着时间的推移，导师为每个学生构建了一个全面的知识图谱，从而实现真正个性化的教学，适应他们独特的学习之旅。

这个例子专注于语义记忆，但关于事件记忆和语义记忆的结合又是怎样的呢？

### 真实世界的应用 – 个人助理

情景记忆和语义记忆的结合创造了一个强大的个人助理。考虑它是如何随时间学习的：当你提到“我要和营销部的莎拉讨论第三季度的预算”时，它存储了情景记忆（关于这次会议的对话）和语义事实（莎拉在营销部工作；有一个关于第三季度预算的讨论计划）。稍后，当你问“我需要为莎拉准备什么？”时，助理将之前与莎拉相关的讨论的情景回忆与关于她角色和即将到来的会议的语义知识结合起来。这创造了一个真正理解你的工作上下文和关系的助理，随着每次互动变得更加有价值。

在情景记忆和语义记忆都正常工作的情况下，我们的代理现在可以维护对话历史并构建知识库。

# 摘要

通过这三个代码实验室，我们将一个简单的无状态代理转换成了一个复杂的具有内存功能的系统，能够在对话中保持上下文并随着时间的推移积累知识。我们的情景记忆实现允许代理存储和检索整个对话场景，提供时间上下文和连续性，使互动感觉更加自然和个性化。语义记忆系统从对话中提取事实知识，构建一个随着每次互动而增长的持久知识库。这些内存组件共同创建了一个不仅能够响应查询，而且真正从经验中学习的代理。

我们在构建每个内存类型时采取的模块化方法展示了 CoALA 框架的灵活性。通过使用 ChromaDB 进行向量存储和 LangGraph 进行工作流程编排，我们创建了一个可扩展的架构，能够在保持快速检索时间的同时处理越来越多的内存数据。情景记忆和语义记忆的分离反映了认知科学原理，即不同类型的信息通过不同的机制进行处理和存储，每个机制都针对其特定目的进行了优化。

使这种实现特别强大的是不同内存类型如何协同工作。当用户提出问题时，代理可以同时通过情景记忆检索相关的过去对话，并通过语义记忆访问存储的事实，将这两个来源结合起来生成信息丰富、上下文相关的响应。这种多模态内存方法使代理能够处理复杂场景，例如记住用户在会话之间的偏好、跟踪主题随时间的变化，以及构建随着每次互动而改进的全面用户档案。

尽管我们已经成功实现了情景记忆和语义记忆，但我们尚未解决的一种关键记忆类型——程序性记忆。这是记住和执行学习到的动作序列的能力。在*第十八章*中，我们将探讨 LangMem，一个专门的内存管理框架，如何帮助我们实现程序性记忆，使智能体能够学习和优化复杂的多步骤过程。这个最后的记忆组件将完善我们的 CoALA 智能体，创建一个功能齐全的系统，不仅能记住事实和对话，还能通过经验掌握和优化工作流程。

|

## 获取本书的 PDF 版本和独家额外内容

扫描二维码（或访问[packtpub.com/unlock](http://packtpub.com/unlock)）。通过书名搜索本书，确认版本，然后按照页面上的步骤操作。 | ![](img/Unlock-01.png)![](img/Unlock.png) |

| **注意**：请妥善保管您的发票。直接从 Packt 购买不需要发票。* |
| --- |
