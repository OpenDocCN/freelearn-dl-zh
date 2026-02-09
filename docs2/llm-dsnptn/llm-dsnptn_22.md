# 22

# 推理与行动

**推理与行动**（**ReAct**）是由普林斯顿大学和谷歌的研究人员开发的一种提示技术，它增强了 LLM 在模拟环境中进行推理和行动的能力（[`arxiv.org/pdf/2210.03629`](https://arxiv.org/pdf/2210.03629)）。它允许 LLM 模仿现实世界中的类似人类操作，在那里我们用言语进行推理并采取行动以获取信息。ReAct 将*推理*和*行动*结合起来，以解决复杂语言推理和决策任务。

虽然 CoT 提示使 LLM 能够生成推理轨迹，但其无法访问外部世界可能导致事实虚构等问题。ReAct 通过允许 LLM 为任务生成*口头推理轨迹*和*文本动作*来解决此问题。这些文本动作使模型能够与其环境（例如，通过查询外部知识源或使用工具）交互，收集信息，并相应地调整其推理。

ReAct 的关键特性如下：

+   **推理轨迹**：LLM 生成逐步解释其思维过程的文本

+   **动作生成**：LLM 产生代表与外部工具或环境交互的文本动作

+   **观察融合**：动作（观察）的结果被反馈到 LLM 的上下文中，影响后续的推理和动作

+   **迭代过程**：ReAct 通常涉及多个*思考*/*动作*/*观察*步骤，允许动态问题解决

ReAct 在以下场景中表现出色：

+   当任务需要超出 LLM 预训练知识的信息时（例如，多跳问答或事实验证）

+   当一个 LLM 需要导航和与模拟环境（例如，在线购物或基于文本的游戏）交互时

+   当你需要结合 LLM 的力量与外部工具（例如，搜索引擎、计算器和 API）的能力时

+   当任务需要将问题分解成更小的步骤，并且必须根据中间结果做出决策时

在本章中，我们将涵盖以下主题：

+   在 LangChain 中实现 ReAct

+   使用 LangChain 的表达语言构建 ReAct 代理

+   完成任务和解决问题

+   评估 ReAct 的性能

+   安全性、控制和伦理考虑

+   局限性和未来方向

# 在 LangChain 中实现 ReAct

开源 LLM 框架 LangChain ([`www.langchain.com/`](https://www.langchain.com/)) 通过其`Agent`类提供了一个强大且灵活的 ReAct 框架实现。让我们探索如何在 LangChain 中创建和使用 ReAct 代理：

1.  安装必要的包：

    ```py
    duckduckgo-search and youtube_search integrate search engine functionalities, allowing language models to retrieve real-time information from the web and YouTube, respectively
    ```

1.  `wikipedia`使语言模型能够访问和利用维基百科的信息，扩大其知识库

1.  `langchainhub`是一个用于共享和发现 LangChain 资产（如提示、链和代理）的中心仓库

1.  初始化语言模型和工具，如 `wikipedia`、`ddg-search` 和 `llm-math`。这些在以下代码片段中列出：

    ```py
    import os
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OpenAI API Key:")
    from langchain.agents import load_tools
    from langchain.chat_models import ChatOpenAI
    # load the language model, you can use any model you like
    llm = ChatOpenAI(model = "gpt-4o", temperature=0)
    # load tools
    tools = load_tools(['wikipedia', 'ddg-search','llm-math'],
        llm=llm)
    ```

    在这里，我们从 `langchain` 导入必要的模块。然后，使用指定的模型（`gpt-4-1106-preview` 和 `temperature`）初始化语言模型（`ChatOpenAI`）。最后，我们加载一些代理将使用的工具。

1.  初始化 ReAct 代理。在这里，`initialize_agent` 函数创建并初始化了一个代理：

    ```py
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType
    # initialize agent
    agent = initialize_agent(
        tools,  
        llm,  
        agent=AgentType.ZERO_SHOT_REACT_ DESCRIPTION,  
        verbose=True  
    ) 
    ```

    在前面的代码中，我们列出了 `tools` 和 `llm`，其中 `llm` 指的是语言模型，并指定代理类型为 `AgentType.ZERO_SHOT_REACT_DESCRIPTION`。在这里，`verbose=True` 启用了代理思维过程的详细记录。

1.  检查 ReAct 代理的提示。以下行打印 ReAct 代理使用的提示模板。此提示为 LLM 提供了如何使用可用工具并遵循 ReAct 格式（*Thought*，*Action*，*Action Input* 和 *Observation*）的说明：

    ```py
    print(agent.agent.llm_chain.prompt.template)
    ```

    检查 ReAct 代理的提示非常重要，因为它揭示了指导语言模型在工具辅助推理和动作期间行为的结构和逻辑。通过使用 `print(agent.agent.llm_chain.prompt.template)` 打印提示模板，您不仅看到了任意的指令，而且检查了控制代理如何序列其推理和工具使用的行动框架。这包括它如何解释用户查询、从其可用动作集中选择工具、构建工具的输入，以及将工具的输出（观察）整合到进一步的推理中。如果提示构建不当，模型可能会误解工具、采取无效的动作或无法连贯地串联思维。此外，模板通常包括少量示例，展示如何正确地在 ReAct 组件之间交替。这些示例作为格式和逻辑的隐式指令，帮助模型泛化到未见过的任务。检查它们可以揭示代理是否使用通用模式或高度具体的用例进行训练或指导。它还有助于开发者调试意外的行为或幻觉，因为直接修改模板会直接影响代理的动作选择、推理的准确性以及与预期 ReAct 循环的整体一致性。

1.  以下代码块演示了如何自定义提示模板。您可以修改说明、示例和格式以更好地适应您的特定用例：

    ```py
    prompt = """
    You are an intelligent agent designed to solve complex queries by breaking them down systematically and using available tools strategically. Follow the ReAct (Reasoning and Acting) framework to approach each task.
    ReAct Principles:
    1\. Reasoning: Always start by carefully analyzing the question and developing a clear, step-by-step thought process.
    2\. Tool Selection: Critically evaluate which tools will be most effective for addressing the specific query.
    3\. Iterative Interaction: Be prepared to cycle between reasoning and action multiple times, refining your approach as you gather more information.
    4\. Comprehensive Understanding: Aim to not just find an answer, but to truly comprehend the underlying context and nuances of the question.
    5\. Transparent Decision-Making: Clearly articulate your reasoning, actions, and thought process at each step.
    Available Tools:
    - Wikipedia: Retrieve factual information about people, places, historical events, and general knowledge topics.
    - Google Search: Fetch current information, recent events, and up-to-date context.
    - Calculator: Perform mathematical calculations and numerical analysis.
    Interaction Format:
    Question: The specific query to be solved
    Thought: Detailed reasoning about the approach, breaking down the problem
    Action: Selected tool (Wikipedia/Google Search/Calculator)
    Action Input: Precise query for the selected tool
    Observation: Results obtained from the tool
    ... (Repeat reasoning, action, and observation as needed)
    Thought: Final synthesized understanding
    Final Answer: Comprehensive and well-reasoned response to the original question
    Important Guidelines:
    - Be methodical and explicit in your reasoning
    - Use tools judiciously and avoid unnecessary actions
    - Integrate information from multiple sources when appropriate
    - Provide a clear, concise, and informative final answer
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """
    ```

    在这里，`agent.agent.llm_chain.prompt.template = prompt` 更新了代理的提示为自定义模板。

1.  接下来，您可以修改工具的描述，为 LLM 提供更具体的指导，说明何时以及如何使用每个工具：

    ```py
    tools[1].description = "A date retrieval tool that provides the current date and time, useful for temporal queries, scheduling, age calculations, or understanding time-sensitive contexts."
    tools[2].description = "A powerful computational tool capable of performing various mathematical operations, including arithmetic calculations, algebraic computations, percentage calculations, unit conversions, and advanced mathematical functions."
    ```

1.  以下行执行代理以一个示例查询。代理将使用 ReAct 框架进行推理、选择工具、执行动作并生成最终答案：

    ```py
    agent.run("What is the population of the largest city in Canada? How many days would it take for that city's population to count to 1 billion if each person counts one number per second without breaks? Then, compare this time to the average lifespan of a human in years, and explain which is longer.")
    ```

接下来，我们将通过一个示例来查看如何使用 ReAct 进行文档处理，该示例利用了 LangChain。

## ReAct 文档存储

LangChain 还提供了一个 `DocstoreExplorer` 类，用于实现与维基百科等文档存储的 ReAct 逻辑。我们将通过使用 `DocstoreExplorer` 和维基百科进行文档式 ReAct 的示例来演示：

```py
from langchain import Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
docstore = DocstoreExplorer(Wikipedia())
search_tool = Tool(name="Search",
                   func=docstore.search,
                   description="Search for latest information about any topic"
                   )
lookup_tool = Tool(name="Lookup",
                   func=docstore.lookup,
                   description="Lookup tool for get information from a keyword"
                   )
tools = [search_tool, lookup_tool]
llm = OpenAI(temperature=0)
react = initialize_agent(tools,
                         llm,
                         agent=AgentType.REACT_DOCSTORE,
                         verbose=True)
question = "Who is the current governor of Texas and when was he born ?"
react.run(question)
```

这段代码设置了一个 LangChain 代理，该代理通过与维基百科交互来回答问题。以下是分解：

1.  **维基百科访问**：首先，它初始化与维基百科的连接，使代理能够从中检索信息。

1.  `搜索` 和 `查找`。`搜索` 工具使代理能够找到相关的维基百科页面，而 `查找` 工具则允许它从这些页面中提取特定信息。

1.  `AgentType.REACT_DOCSTORE` 明确配置代理以进行文档存储交互 – 在这种情况下，是维基百科的。

1.  使用 `搜索` 工具查找相关页面和 `查找` 工具提取答案。

# 使用 LangChain 的表达式语言构建 ReAct 代理

**LangChain 表达式语言**（**LCEL**）提供了一种声明式方法来构建 ReAct 代理。LCEL 允许你定义一个处理图，该图处理用户输入、推理、动作选择和最终响应生成。本节演示了如何使用这个强大的框架实现 ReAct 代理。

核心思想是建立一个数据管道，该管道接收用户的查询，使用 LLM 通过一系列步骤进行推理，可能利用外部工具，并最终得出答案。这个管道可以使用 LCEL 简洁地表达。

以下是一个 Python 代码示例，演示了此过程：

```py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import(
    ReActSingleInputOutputParser)
from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
# 1\. Define Tools: In this simple example, we are using a search tool.
tools = [DuckDuckGoSearchRun()]
# 2\. Construct the Prompt:  Instead of pulling from a hub, we'll define a basic prompt template.
template = """Answer the following questions as best you can. You have access to the following tools:
{tool_descriptions}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
{agent_scratchpad}"""
prompt = ChatPromptTemplate.from_template(template)
prompt = prompt.partial(
    tool_names=", ".join([t.name for t in tools]),
    tool_descriptions="\n".join(
        [f"{t.name}: {t.description}" for t in tools]
    ),
)
# 3\. Instantiate the LLM:  We use ChatOpenAI, but any LLM can be used.
llm = ChatOpenAI(temperature=0)
#  We also configure it to stop when it sees '\nObservation:'
llm_with_stop = llm.bind(stop=["\nObservation:"])
# 4\. Construct the Agent Pipeline using LCEL:
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x:
            format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)
```

让我们更详细地看看这个设置：

1.  定义了一个自定义提示模板来指导 LLM 的推理和动作选择，而不是从中心获取一个。这个模板指导 LLM 关于交互预期的格式（*问题*，*思考*，*动作*，*观察*，*最终答案*）。

1.  ChatOpenAI 作为 LLM，配置为在遇到 `\nObservation:` 字符串时停止生成。这个信号表示代理已完成动作并等待结果。

1.  代理管道是通过 LCEL 构建的，这是链式操作（`|`）。这个管道协调信息流：

    +   它格式化输入和代理的草稿本（之前的推理步骤）

    +   它将格式化的输入馈送到提示

    +   LLM，根据其配置的停止标准，处理提示

    +   最后，`ReActSingleInputOutputParser` 解析 LLM 的输出，区分要采取的动作和最终答案

## 解释 ReActSingleInputOutputParser

这个组件对于解释 LLM 的输出并确定 ReAct 循环中的下一步至关重要：

+   **实例化**：您创建解析器的实例，准备处理 LLM 生成的文本

+   `AgentAction`对象（请求执行工具）或`AgentFinish`对象（提供最终答案）

    +   如果它检测到`AgentAction`，则提取工具的名称和传递给工具的输入

    +   如果它找到`AgentFinish`，则提取最终答案

+   `AgentAction`或`AgentFinish`

+   `Action:`或`Final Answer:`），解析器引发异常，表明 LLM 的推理或提示存在问题

## 使用 AgentExecutor 运行代理

在以下代码中，`AgentExecutor`是一个负责管理代理动作执行（基于代理的决策过程选择）的组件。它作为代理的驱动程序，促进代理与外部工具之间的交互。

这里有一个例子：

```py
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke(
    {
        "input": "Who is the current CEO of Microsoft and what is their age squared?"
    }
)
print(response)
```

这里发生以下情况：

1.  我们创建一个`AgentExecutor`实例，向其提供我们之前定义的代理管道和可用工具，然后在设置`verbose=True`以查看代理的思维过程之前。

1.  `agent_executor.invoke`方法启动流程。它接受一个包含用户输入的字典（`"input": "Who is the current CEO of Microsoft and what is their age squared?"`）。

1.  然后，`AgentExecutor` 管理 ReAct 循环：

    1.  它将输入传递给代理管道。

    1.  代理（LLM 和解析器）决定采取的行动（例如，使用搜索工具查找 CEO 的名字）。

    1.  `AgentExecutor`执行动作（调用搜索工具）。

    1.  它将结果作为“`Observation`”返回给代理。

    1.  此过程会重复进行，直到代理决定它已经拥有足够的信息来生成最终答案。

此示例演示了使用 LCEL 构建的 ReAct 代理的基本结构。它展示了如何通过结合提示、语言模型、解析器和外部工具来定义一个清晰、模块化的管道，以实现复杂的推理任务。这种方法促进了代码的可读性、可维护性和设计智能代理的灵活性。这个特定的例子询问微软的现任 CEO 是谁，然后计算他们的年龄平方，展示了从名称回忆到算术计算的简单多轮推理。

# 完成任务和解决问题

ReAct 框架，凭借其整合推理和行动的能力，在各种任务完成和问题解决场景中高度适用：

+   **带有外部知识的问答（QA）**：ReAct 可用于创建可以访问和推理外部知识源（如维基百科或搜索引擎）的 QA 系统，以提供更准确和更新的答案

+   **网页导航和交互**：ReAct 代理可以导航网站，与网页元素交互，并收集信息，从而实现自动化网页研究、数据抓取和在线购物辅助等任务

+   **软件应用控制**：通过集成 API 和工具，ReAct 代理可以控制软件应用，自动化工作流程，并执行需要与多个系统交互的复杂任务

+   **机器人和物理世界交互**：虽然 LLM 主要在文本领域运行，但 ReAct 原则可以扩展到控制机器人或其他物理系统，其中行动涉及物理运动或与真实世界的交互

+   **多步问题解决**：ReAct 非常适合需要将复杂问题分解为更小步骤、对每个步骤进行推理、采取行动并使用观察结果来指导后续步骤的任务

# 评估 ReAct 的性能

评估 ReAct 代理涉及评估推理的质量和采取行动的有效性。以下指标可以用来评估：

+   **成功率**：代理成功完成的任务百分比

+   **效率**：完成任务所需的步骤数量或时间

+   **推理准确性**：LLM 推理轨迹的正确性和相关性

+   **行动相关性**：代理选择的行动的适当性

+   **观察利用**：代理如何有效地将其观察结果纳入后续推理和行动

+   **错误分析**：识别代理性能中的常见故障模式或弱点

让我们考虑一些可以使用的评估技术：

+   **人工评估**：让人类专家评估代理的推理、行动和最终输出

+   **自动化指标**：使用自动化脚本或 LLM 评估代理性能的特定方面，例如答案的正确性或行动的相关性

+   **基准测试**：将代理的性能与预定义的基准或其他代理在标准化任务上的性能进行比较

+   **消融研究**：系统地删除或修改 ReAct 框架的组件（例如，删除推理步骤）以了解其对整体性能的贡献

# 安全、控制和伦理考量

ReAct 系统，尤其是当与外部工具集成时，会引发一些安全、控制和伦理问题：

+   **不可预测的行为**：LLM 推理和外部工具使用的组合可能导致不可预测或非预期的行为

+   **行动的安全性**：代理采取的行动可能产生现实世界的后果，特别是如果代理连接到可以影响物理世界的系统

+   **偏见和公平性**：ReAct 代理可能会继承并放大 LLM 或他们使用的外部工具训练数据中存在的偏见

+   **滥用潜力**：恶意行为者可能将 ReAct 代理用于有害目的，例如生成虚假信息或自动化攻击

+   **责任归属**：由于底层 LLM 模型的不确定性，确定 ReAct 代理的行为和决策的责任可能具有挑战性

以下是一些缓解这些问题的策略：

+   **沙盒**：在隔离环境中运行 ReAct 代理以限制其潜在影响

+   **人工监督**：将人工审查和批准纳入 ReAct 流程，特别是对于关键决策或动作

+   **安全规则和约束**：实施规则和约束以防止代理采取有害或不道德的行动

+   **监控和审计**：持续监控代理的行为并维护日志以供审计

+   **透明度和可解释性**：设计 ReAct 代理，使其能够解释其推理和决策过程，以提高理解和信任

# 局限性和未来方向

虽然 ReAct 是一个强大的框架，但它有一定的局限性：

+   **对外部工具的依赖**：ReAct 的有效性部分取决于它所使用的工具的能力和可靠性

+   **错误传播**：工具使用或观察解释中的错误可能会在推理过程中传播，导致得出错误结论或采取错误行动

+   **标记限制**：ReAct 的迭代性质可能导致文本序列过长，可能超过某些大型语言模型（LLMs）的标记限制

+   **计算成本**：多轮推理、动作和观察可能具有很高的计算成本，尤其是在使用 LLMs 或复杂工具时

+   **提示工程挑战**：设计有效的 ReAct 提示，以正确引导 LLM 的推理和动作选择可能具有挑战性，可能需要进行实验

*图 22*.1 显示了 ReAct 模式的局限性：

![图 22.1 – ReAct 模式的局限性](img/B31249_22_01.jpg)

图 22.1 – ReAct 模式的局限性

然而，通过结合 LLMs 的力量和采取行动以及整合外部信息的能力，ReAct 为创建更强大和通用的 AI 系统提供了新的可能性：

+   **改进的工具集成**：开发更无缝和稳健的方法来集成 LLMs 与外部工具

+   **增强推理能力**：将 ReAct 与其他高级推理技术（如 ToT）相结合，以处理更复杂的场景

+   **从经验中学习**：使 ReAct 代理能够从过去的交互中学习并随着时间的推移提高其性能

+   **多代理 ReAct**：探索多个 ReAct 代理协作或竞争以解决问题的场景

+   **现实世界部署**：超越模拟环境，将 ReAct 代理部署到具有适当安全和控制机制的现实中应用

# 概述

在本章中，你学习了 ReAct 框架，这是一种强大的技术，可以提示你的 LLMs 不仅能够通过复杂场景进行推理，还能规划和模拟动作的执行，类似于人类在现实世界中的操作。

ReAct 框架代表了在开发能够推理、规划和与环境交互的智能代理方面的重大进步。ReAct 也可以被视为更高级框架如**无需观察的推理**（**ReWOO**）的先驱，我们将在下一章中探讨这一点。
