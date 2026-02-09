# 23

# 无观察推理

**无观察推理**（**ReWOO**），由 Xu 等人提出（[`arxiv.org/abs/2305.18323`](https://arxiv.org/abs/2305.18323)），是一个结合多步规划和变量替换以有效使用工具的框架。它旨在通过在单次遍历中生成工具使用的完整链来减少令牌消耗和执行时间，从而最小化冗余的 LLM 调用。它还旨在通过允许在不实际调用工具的情况下进行微调来简化微调过程，因为规划数据在理论上不依赖于工具输出。

ReAct 通过循环的“思考-行动-观察”模式运作，其中人工智能参与推理，执行行动，检查产生的反馈，然后相应地调整其后续行动，从而促进动态和响应性的问题解决策略。相比之下，ReWOO 强调全面的前期规划，在执行之前生成一系列完整的行动，从而最大限度地减少持续观察和反馈的必要性。这种区别使得 ReWOO 可以通过减少令牌消耗和计算成本，从 ReAct 的迭代反馈循环转向更简化的“计划-行动”方法来追求更高的效率。

因此，ReWOO 指的是 LLM 对其未直接观察或训练过的情景进行推理、预测或决策的能力。ReWOO 通过将外部工具使用纳入推理过程来增强这一点。

ReWOO 能够在不直接观察的情况下进行规划和推理，这使得它适合复杂规划和决策任务：

+   **战略规划**：如前所述，ReWOO 可以根据假设情况、目标和约束生成战略计划

+   **情景分析**：ReWOO 可以探索给定情景的多种潜在结果，考虑各种因素和不确定性

+   **资源分配**：通过规划工具使用并对其结果进行推理，ReWOO 可以在复杂环境中优化资源分配

+   **风险评估**：ReWOO 可以通过模拟不同场景及其后果来帮助评估潜在风险并制定缓解策略

在本章中，我们将涵盖以下主题：

+   使用 LangGraph 实现 ReWOO

+   ReWOO 的优势

+   评估质量和伦理考量

+   未来方向

# 使用 LangGraph 实现 ReWOO

LangGraph 是一个开源框架，旨在使用 LLM 构建具有状态的、多代理应用程序。它通过引入有向图模型扩展了 LangChain 生态系统的功能，其中节点代表函数（包括 LLM 调用），边代表基于逻辑、条件或内存的状态之间的转换。与传统顺序链相比，LangGraph 允许涉及条件分支、循环、内存传递和异步代理协调的复杂工作流程。LangGraph 特别适用于实现交互动态、迭代且依赖于状态变化的系统。这包括多代理协作、决策树、具有控制流的检索增强生成以及需要回顾先前步骤或循环子任务直到达到某些目标的自主代理。

LangGraph 利用图论和自动机概念，将执行流程表示为状态机或有向无环图（或需要循环时为循环图）。开发者定义一个包含节点（函数或工具）、边（状态转换）和条件（路由逻辑）的图。运行时引擎随后根据输入执行图，并在每一步更新状态。

LangGraph 支持同步和异步执行，并且与 LangChain 的组件（如工具、内存和代理）集成。它还支持流式响应、对状态的精细控制以及多模态输入/输出，使其适用于生产级应用。

实际上，LangGraph 用于构建具有不同代理交互、协调和共享内存的代理系统，同时仍遵循定义良好的计算图。这使得它与简单的代理循环或无结构的 LLM 调度方法不同。

LangGraph 可在 [`github.com/langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph) 获取，并支持基于 Python 的实现，核心依赖 LangChain 和状态机执行框架。

ReWOO 架构由三个模块组成：

+   `search_result` 或 `price`，AI 可以构建一个清晰的任务蓝图，将动态信息的解决推迟到可用时，从而简化规划过程并避免不必要的计算。

+   **Worker**：使用提供的参数执行工具，可能使用之前步骤中的变量替换。

+   **Solver**：根据工具观察结果和计划生成最终答案。

这种架构最初可能看起来有些抽象。让我们使用 LangGraph 实现 ReWOO。我们将以 Tavily 搜索引擎作为一个示例工具：

1.  安装必要的包并设置 API 密钥：

    ```py
    # %pip install -U langgraph langchain_community langchain_openai tavily-python
    ```

    Tavily 是专为 AI 代理设计的搜索引擎。它旨在提供准确可靠的信息检索，满足执行复杂任务的 AI 系统的需求（见 [`tavily.com/`](https://tavily.com/)）。

    以下脚本设置 API 密钥的环境变量，如果尚未定义：

    ```py
    import getpass
    import os
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"{var}=")
    _set_if_undefined("TAVILY_API_KEY")
    _set_if_undefined("OPENAI_API_KEY")
    ```

1.  接下来，定义图状态。为此，定义状态字典，使其可以保存任务、计划、步骤、结果和最终结果：

    ```py
    from typing import List
    from typing_extensions import TypedDict
    class ReWOO(TypedDict):
        task: str
        plan_string: str
        steps: List
        results: dict
        result: str
    ```

1.  创建规划提示和逻辑：

    ```py
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o")
    prompt = """For the following task, create a series of plans that can solve the problem step-by-step. For each plan, specify which external tool and its corresponding input should be used to gather evidence. You can store the evidence in a variable #E (e.g., #E1, #E2, #E3, etc.) that can be referenced by subsequent tools. Note that all the variables are independent, so make sure to include all necessary information in each tool input.
    Tools can be one of the following:
    Google[input]: A search engine worker that retrieves results from Google. Use this when you need concise answers or information about a specific topic. The input should be a search query.
    LLM[input]: A pretrained Large Language Model (like me). Use this when you need to leverage general world knowledge, common sense, or perform complex reasoning. Prioritize this tool when you are confident in solving the problem without external assistance. The input can be any instruction or question.
    Calculator[input]: A tool that can perform mathematical calculations. Use this when you need to perform arithmetic operations. The input should be a valid mathematical expression.
    WolframAlpha[input]: A computational knowledge engine. Use this when you need to solve equations, perform symbolic calculations, or get data-driven answers. The input should be a query in Wolfram Language or natural language related to a math or science problem.
    For example,
    Task: Alice, Bob, and Carol earned a total of $540 from their part-time jobs last week. Alice earned y dollars. Bob earned $20 more than three times what Alice earned, and Carol earned $15 more than Bob. How much money did Carol earn?
    Plan: Given Alice earned y dollars, translate the problem into algebraic expressions and solve with Wolfram Alpha.
    #E1 = WolframAlpha[Solve y + (3y + 20) + ((3y + 20) + 15) = 540]
    Plan: Find out the amount of money Alice earned.
    #E2 = LLM[What is y, given #E1]
    Plan: Calculate the amount of money Carol earned.
    #E3 = Calculator[((3 * #E2) + 20) + 15]
    Begin!
    Describe your plans with rich details. Each Plan should be followed by only one #E.
    Task: {task}"""
    ```

1.  为规划器创建一个 LangGraph 节点：

    ```py
    import re
    from langchain_core.prompts import ChatPromptTemplate
    regex_pattern = (
        r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*"
        r"\[([^\]]+)\]"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [("user", prompt)]
    )
    planner = prompt_template | model
    def get_plan(state: ReWOO):
        task = state["task"]
        result = planner.invoke({"task": task})
        matches = re.findall(regex_pattern, result.content)
        return {"steps": matches, "plan_string": result.content}
    ```

1.  实例化搜索引擎并定义工具执行逻辑：

    ```py
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults()
    def _get_current_task(state: ReWOO):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1
    def tool_execution(state: ReWOO):
        _step = _get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][_step - 1]
        _results = (state["results"] or {}) if "results" in state else {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
        if tool == "Google":
            result = search.invoke(tool_input)
        elif tool == "LLM":
            result = model.invoke(tool_input)
        else:
            raise ValueError
        _results[step_name] = str(result)
        return {"results": _results}
    ```

1.  创建求解器提示和逻辑：

    ```py
    solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
    retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
    contain irrelevant information.
    {plan}
    Now solve the question or task according to provided Evidence above. Respond with the answer
    directly with no extra words.
    Task: {task}
    Response:"""
    def solve(state: ReWOO):
        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = (
                (state["results"] or {}) if "results" in state else {}
            )
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += (
                f"Plan: {_plan}\n"
                f"{step_name} = {tool}[{tool_input}]\n"
            )
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        result = model.invoke(prompt)
        return {"result": result.content}
    ```

1.  构建 LangGraph 工作流程：

    ```py
    def _route(state):
        _step = _get_current_task(state)
        if _step is None:
            return "solve"
        else:
            return "tool"
    from langgraph.graph import END, StateGraph, START
    graph = StateGraph(ReWOO)
    graph.add_node("plan", get_plan)
    graph.add_node("tool", tool_execution)
    graph.add_node("solve", solve)
    graph.add_edge("plan", "tool")
    graph.add_edge("solve", END)
    graph.add_conditional_edges("tool", _route)
    graph.add_edge(START, "plan")
    app = graph.compile()
    ```

    提供的代码使用`StateGraph`，一种用于管理多步过程的数据结构，建立了一个 AI 工作流程。`_route`函数充当条件导演，根据当前状态确定下一步。它检查是否需要进一步的基于工具的操作；如果不需，则将流程路由到`"solve"`节点以生成最终答案。否则，将其路由到`"tool"`节点以执行工具。

    在这里，`StateGraph`定义了执行流程：从创建策略的`"plan"`开始，然后到使用外部工具的`"tool"`，最后到生成结果的`"solve"`，最终达到`END`状态。`"tool"`节点内`_route`函数的条件逻辑是关键，它允许根据任务进度进行动态路由。

    `StateGraph`对于结构化工作流程管理至关重要，它使 AI 行为能够进行条件分支，特别是在依赖工具的任务中。它确保逻辑动作顺序，提高鲁棒性和清晰度，并促进 ReWOO 的计划执行。将图编译成`"app"`使其可执行。

1.  让我们看看一个示例用例，并测试 ReWOO 代理：

    ```py
    task = "what is the exact hometown of the 2024 mens australian open winner"
    for s in app.stream({"task": task}):
        print(s)
        print("---")
    ```

前面的代码提供了一个使用 LangGraph 的 ReWOO 框架的简单实现。它定义了状态、规划器、执行器和求解器模块，并将它们连接到一个图中。此示例用法演示了如何在样本任务上运行代理。

# ReWOO 的优势

ReWOO 相对于传统的 ReAct 风格代理具有几个优势：

+   **减少令牌消耗和执行时间**：通过单次遍历生成整个计划并使用变量替换，ReWOO 最小化了冗余的 LLM 调用和上下文传递

+   **简化微调**：规划数据与工具输出的独立性（理论上）允许进行微调，而无需调用工具

+   **高效的 LLM 调用**：与 ReACT 范式相比，LLM 工具接收的提示更少，使调用更高效

# 评估质量和伦理考量

评估 ReWOO 推理的质量可能具有挑战性，因为它经常涉及假设情景。可能的方法包括以下：

+   **人工评估**：使用人类专家评估生成的计划和推理的连贯性、相关性和完整性

+   **与真实结果的比较**：对于已知结果的情况，ReWOO 的预测可以与实际结果进行比较

+   **基准测试**：使用旨在评估抽象推理和规划能力的标准化测试集

在进行任何评估时，也必须牢记道德考量：

+   **偏差放大**：ReWOO 可能会继承并放大底层 LLM 训练数据中存在的偏差

+   **滥用潜力**：生成计划和推理假设情景的能力可能会被用于恶意目的

+   **过度依赖**：用户可能会过度信任 ReWOO 的输出，而未考虑其推测性质

# 未来方向

随着研究的进展，ReWOO 和相关技术可能会在更强大和多功能的人工智能系统的发展中扮演越来越重要的角色。以下是一些 ReWOO 的有希望的发展方向：

+   **人机交互系统**：将人类监督和反馈集成到 ReWOO 框架中，以提高准确性和解决道德问题

+   **改进的规划算法**：开发更复杂的规划算法，能够处理更复杂的场景和更大的搜索空间

+   **增强的工具集成**：无缝集成更广泛范围的工具，包括专门的 API 和知识库

+   **多智能体协作**：使多个 ReWOO 智能体能够协作完成复杂任务，可能带来更稳健和多样化的解决方案

+   **元学习**：应用元学习技术以提高智能体随着时间的推移在新的场景中泛化和适应的能力

# 摘要

本章深入探讨了 ReWOO，这是一个旨在赋予 LLM 推理假设情景和有效利用外部工具能力的框架。ReWOO 利用多步骤规划器配合变量替换，使其能够在单次遍历中生成全面的行为计划，从而与 ReAct 智能体的迭代“思考-行动-观察”循环相比，最小化令牌消耗和执行时间。本章通过 LangGraph 展示了 ReWOO 的实现，突出了其架构、组件（规划器、工作者、求解器）以及优势，如简化微调和高效的 LLM 调用。

除了简单地重复框架的机制之外，本章强调了 ReWOO 在战略规划、情景分析、资源分配和风险评估方面的潜力。然而，它也触及了围绕 ReWOO 的关键道德考量，包括偏差放大、滥用和过度依赖其输出的可能性。本章以展望未来结束，讨论了需要人机交互系统、改进的规划算法、增强的工具集成、多智能体协作以及应用元学习技术来进一步细化 ReWOO 的能力，并确保其在现实场景中的负责任应用。

在下一章中，我们将讨论使 LLM 能够进行自我反思和迭代改进的技术。
