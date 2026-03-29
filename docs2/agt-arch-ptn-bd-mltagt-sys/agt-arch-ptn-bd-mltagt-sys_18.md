# 15

# 代理框架 – 用例：使用 CrewAI 和 LangGraph 的贷款处理多代理系统

在整本书中，我们探讨了支撑代理 AI 系统的概念、架构和模式。我们检查了它们的构建块、它们如何交互以及它们如何被协调以应对复杂任务。然而，将这些理论结构转化为功能性强、适用于生产的应用程序需要处理代理创建、执行和通信的底层复杂性的实用工具和库。这就是代理框架发挥作用的地方。

从零开始构建代理系统涉及管理众多动态部分：定义代理角色和能力、处理状态和记忆、编排工作流程、集成 LLMs、管理工具使用（如函数调用）以及促进代理之间的沟通。

代理框架提供了抽象和预构建组件，简化了这些任务，使开发者能够专注于应用程序逻辑和代理能力，而不是重新发明基础管道。它们提供了针对代理开发中常见挑战的结构化方法，促进了更快的开发周期和更易于维护的代码，并且通常包含了最佳实践，用于交互和控制。

在代理 AI 快速发展的领域中，已经出现了几个框架，每个框架都有自己的理念、优势和目标用例。虽然我们将关注三个突出的例子来展示实现，但重要的是要注意这些例子是说明性的，而不是详尽的；本书中讨论的底层设计模式是通用的，并适用于其他框架。

在本章中，我们将提供对三个突出示例的实用介绍：

+   谷歌的代理开发工具包（ADK）

+   CrewAI

+   LangGraph

我们将首先简要介绍每个框架，突出其核心概念。然后，我们将探讨它们的相似之处和不同之处，以帮助您理解它们各自的设计理念。为了使比较具体化，我们将使用 CrewAI 和 LangGraph 重新实现*第十三章*中的贷款处理代理用例。由于我们在前几章中广泛使用了谷歌的 ADK 来建立基线架构，因此我们在此不再重复该实现。相反，我们将专注于展示如何根据我们配套笔记本中的代码使用不同的框架范式来处理相同的问题。最后，我们将讨论在选择代理框架时需要考虑的关键因素，强调将框架的能力与您的特定用例相一致的重要性，以及无论您选择什么工具，都需要持续进行稳健的观察和遵守负责任的 AI 原则。

既然我们已经理解了框架在构建代理系统中的重要性，那么让我们更深入地看看我们的第一个例子：谷歌的 ADK。

# 技术要求

为了充分利用本章中的实际示例，你需要以下内容：

+   Python 环境（版本 3.10+）

+   Jupyter Notebook 界面（例如，JupyterLab、Google Colab 或 VS Code）

+   启用 Vertex AI API 的 Google Cloud 项目或 Google AI Studio API 密钥

+   需要安装以下 Python 库：`crewai`、`langgraph`、`langchain``-google-``genai`和`google-cloud-``aiplatform`

本章的完整代码可在本书 GitHub 存储库的以下文件夹中找到：[`github.com/PacktPublishing/Agentic-Architectural-Patterns-for-Building-Multi-Agent-Systems/tree/main/Chapter_15`](https://github.com/PacktPublishing/Agentic-Architectural-Patterns-for-Building-Multi-Agent-Systems/tree/main/Chapter_15)。

既然我们了解了框架在构建代理系统中的重要性，让我们更深入地看看我们的第一个例子：Google 的 ADK。

# Google 的代理开发工具包（ADK）

正如我们在前面的示例中看到的，Google 的 ADK 提供了一个结构化的环境来构建和部署 AI 代理。它旨在考虑生产就绪性，提供定义代理、管理其生命周期、集成工具和促进通信的组件，尤其是在多代理场景中。

ADK 旨在为构建能够推理、规划和可靠地与外部系统和其他代理交互的代理提供基础框架。关键特性通常包括以下内容：

+   **代理抽象**：定义代理核心逻辑的基础类或结构，包括其指令、工具以及它如何处理传入的任务或消息。

+   **工具集成**：定义和注册工具（通常是函数或 API）的机制，代理可以利用这些工具与外界交互或执行特定操作，符合代理使用工具对其环境进行操作的观念。

+   **规划和推理**：与 LLMs（如 Gemini）集成以驱动代理的推理循环：处理信息、规划步骤以及决定何时使用工具。ADK 通常包括内置规划器或允许自定义规划逻辑。

+   **状态管理**：代理在交互中维护状态和记忆的机制，这对于复杂的多步骤任务至关重要。

+   **通信协议**：原生支持**代理到代理**（**A2A**）互操作性协议。ADK 使代理能够使用 A2A 的标准消息格式和任务生命周期，在不同框架和企业边界之间进行通信和协作。

+   **运行环境**：一个执行环境（代理运行时或代理引擎），它管理代理部署、任务分配、并行执行和重试，并且可能集成可观察性工具。

在第十三章和第十四章的贷款处理示例中，展示了 ADK 如何允许将不同的代理功能（文档验证、信用检查、风险评估和合规性检查）定义为工具，并通过遵循特定指令的 LLM 驱动的代理来编排它们的执行。ADK 提供了一个结构，在受管理的运行时内构建这样的目标导向、使用工具的代理。

# CrewAI

CrewAI 对构建代理系统提供了 idx_92ad8e97a 不同的视角，专注于通过角色扮演代理的协作智能。它提供了一个框架，用于编排协同工作的自主 AI 代理，它们作为一个统一的“团队”一起工作。

CrewAI 背后的核心哲学是，复杂任务通常可以被分解并分配给具有特定角色、责任甚至“背景故事”的代理，这些背景故事指导他们的行为和专业知识。这些代理随后协作，共享信息和中间结果，以实现共同目标。

CrewAI 中的关键概念包括以下内容：

+   **代理**：定义了特定的角色、目标、背景故事，他们使用的 LLM 以及他们可能可以访问的特定工具。角色扮演方面有助于 LLM 体现特定的角色或专业知识。

+   **任务**：分配给代理的具体任务。每个任务都有一个描述和预期的输出，并分配给特定的代理。任务可以串联，允许一个任务的输出成为另一个任务的输入。

+   **工具**：与其他框架类似，这些是代理可以用来与外部系统交互或执行操作（例如，搜索网络、访问 API）的功能或能力。在 CrewAI 中，工具通常继承自`BaseTool`类。

+   **团队**：代理集合和他们需要执行的任务。团队定义了代理如何协作。

+   **流程**：团队执行任务时遵循的工作流程或方法论。常见的流程包括顺序（任务一个接一个执行）或分层（管理代理委派任务）。

CrewAI 强调代理交互的社会方面，使得设计不同 AI 角色贡献专业技能的系统变得直观，这反映了人类团队如何运作。它旨在通过提供定义角色和管理协作工作流程的高级抽象来简化多代理系统的创建。

然而，对于企业架构师来说，这种“角色驱动”的风格可能会在模型输出中引入可变性。当将 CrewAI 应用于受监管或高风险工作流程时，例如我们的贷款审批用例，平衡这种灵活性以严格的工具合同和严格的测试至关重要，以确保确定性和合规的结果。

# LangGraph

LangGraph 扩展了流行的 LangChain 库，提供了一种构建状态化、多参与者应用程序的强大方式，包括复杂的代理系统，使用图来实现。虽然 LangChain 专注于调用链（线性序列），LangGraph 允许循环，这使得它更适合灵活地模拟代理行为，因为代理通常需要循环、重试或根据当前状态动态决定下一步。

LangGraph 将代理工作流程表示为 **状态机**。工作流程中的每一步都是图中的一个 **节点**，步骤之间的转换是 **边**。这种图结构明确管理应用程序的状态，随着其发展而变化。

LangGraph 的关键概念包括以下内容：

+   **StateGraph**：表示工作流程图的核心理念对象。它包含应用程序的状态。

+   **状态**：一个定义明确的 idx_9aab37f3 数据结构（通常是 Python 类或字典，如 `TypedDict`），它包含与工作流程进度相关的所有信息（例如，用户输入、中间结果、代理消息）。

+   **节点**：表示图中的步骤或参与者（代理）的函数或可运行对象。每个节点接收当前状态，执行操作（例如调用 LLM、使用工具或处理数据），并返回对状态的更新。

+   **边**：定义节点之间的转换。边根据当前状态或前一个节点的输出确定要执行的下一个节点。

+   **条件边**：允许分支逻辑。根据当前状态或节点的输出，图可以将执行路由到不同的后续节点，从而实现复杂的决策和循环。

LangGraph 特别适合以下应用：

+   显式状态管理至关重要

+   需要循环 idx_fd9803f2 过程（例如，代理对其输出的反思和重试，或**人机交互**（**HITL**）交互）

+   需要复杂的控制流，包括分支和动态路由

+   模拟多个代理或参与者（包括人类）之间的交互是必要的

通过将代理交互表示为图，LangGraph 提供了对执行流程和状态持久化的精细控制，使其成为构建复杂和可靠代理应用程序的有力工具。

现在，我们已经看到了每个框架的能力，让我们突出它们之间的相似之处和不同之处。

# 三个框架之间的相似之处和不同之处

虽然所有三个框架都旨在帮助您构建复杂的代理应用程序，但它们的底层哲学和架构引导它们走向不同的优势。

## 相似之处

所有三个 idx_89bca12b 框架都共享一个共同的概念基础：

+   **LLM 作为推理引擎**：在本质上，所有三个框架都使用 LLM（如 Gemini、GPT-4 或开源模型）作为代理的“大脑”。LLM 负责推理、规划和根据提示、当前状态和可用工具决定下一步要做什么。

+   **工具集成**：它们都是围绕**函数调用**或**工具使用**模式构建的，这是我们识别为代理式 AI 基础性的。一个代理的能力，例如，搜索数据库、读取文件或调用 API，是通过提供一组工具来实现的。所有三个框架都提供了一种结构化的方式来定义这些工具，并将它们提供给 LLM。

+   **目标导向**：这些不是简单的单次问答工具。它们旨在构建能够执行复杂、多步骤任务以实现特定、开发者定义目标的应用程序。

## **主要差异**

主要差异在于它们的核心理念，即它们用来表示代理工作流程的心理模型。这种基本差异影响着从控制流到状态管理的一切。为了帮助您评估哪种心理模型最适合您的特定用例，让我们来看看每个框架如何概念化其主要的构建块和操作逻辑：

+   **核心哲学和抽象**：

    +   **CrewAI** **–****基于角色的协作**：CrewAI 的抽象是一个专家团队。您可以通过角色（例如，“高级贷款官员”）、目标（例如，“分析贷款申请”）和背景故事（例如，“你是一位细致的分析师...”）来定义代理。这使得它对于模仿人类团队的流程来说非常直观。协作是其核心特性。

    +   **LangGraph** **–****状态图**：LangGraph 的抽象是流程图或状态机。您定义节点（代理或函数）和边（它们之间的路径）。这使关注点从*代理*转移到*流程*。其强大之处在于使应用程序的状态明确，控制流确定。

    +   **Google ADK** **–****生产就绪的代理**：ADK 的抽象是代理本身，被视为一个模块化、可测试和可部署的软件组件。它提供了一种更结构化、以代码为先的方法，对软件工程师来说感觉熟悉。它侧重于代理的生命周期，并依赖于两个关键机制来实现企业级健壮性：

        +   **回调**（用于主动过滤、PII 检测和 HITL 控制的中间件）

        +   **工作流程代理**（定义与自主推理并行的顺序、循环或并行任务的脚手架）

+   **控制流和循环行为**：

    +   **CrewAI**在高级别管理控制流。您通常将流程定义为*顺序的*（任务 1 -> 任务 2 -> 任务 3）或*分层的*（管理代理委派任务）。这对于线性或简单的委派任务来说简单而有效，正如我们在笔记本示例中所见。

    +   **LangGraph** 提供了完整的、细粒度的控制。因为工作流是一个图，您可以轻松创建循环、分支和循环。您可以定义条件边，例如，“如果验证失败，转到‘拒绝’节点；否则，转到‘信用检查’节点。”这种管理错误和循环的能力是许多高级代理模式的关键要求。

    +   **ADK** 平衡了这些。它可以运行确定性工作流（如 `SequentialAgent`），但也允许动态、LLM 驱动的规划，其中代理本身决定下一步，然后由代理运行时管理。

+   **状态管理**：

    +   **LangGraph** 的超级能力是其显式的状态管理。您定义一个 `State` 对象（例如，一个字典或 `TypedDict`），其中包含您应用程序的所有信息。整个状态被传递给每个节点。每个节点执行其工作并返回对状态的更新。这使得调试变得容易得多；您可以在每个单独的步骤中检查状态。

    +   **CrewAI** 的状态管理更为隐式。一个任务的输出会自动格式化并作为上下文传递给依赖于它的下一个任务。这对于简单的链来说很快，但与 LangGraph 的显式状态相比，提供了更少的直接控制和检查。

    +   **ADK** 使用管理状态。代理运行时和会话服务负责在交互中持久化代理的状态和内存，将这种复杂性从开发者那里抽象出来，并确保即使长时间运行的代理也能从上次离开的地方继续。

现在我们对这些框架有了更深入的了解，让我们退后一步进行比较。

## 比较分析：ADK、CrewAI 和 LangGraph

现在我们已经探讨了每个框架的个体特性，让我们来看看它们是如何相互比较的。这项分析将帮助您根据您项目的具体成熟度、复杂性和运营需求选择合适的工具。

下表提供了每个框架生态系统和背后的哲学的高层次概述：

| **框架** | **主要支持者** | **宣布/发布** | **核心哲学和重点** |
| --- | --- | --- | --- |
| GoogleADK | Google | 2024（原型）/2025（公开） | 一个用于构建、评估和部署生产级、健壮代理的综合、开源工具包。针对 Google 生态系统（Gemini 和 Vertex AI）进行优化，但设计为模型无关。 |
| CrewAI | CrewAI（由若昂·莫拉创立） | 2023（开源发布） | 一个用于编排角色扮演、自主人工智能代理的框架。强调协作智能，代理作为“船员”一起工作以实现目标。 |
| LangGraph | LangChain | 2024 | LangChain 的扩展，用于构建具有状态的、多代理的应用程序。通过将它们建模为图（状态机）来擅长创建具有循环过程和复杂控制流的应用程序。 |

表 15.1 – 代理框架比较

为了进行更深入的技术比较，以下表格按其架构方法、控制流机制和适用于不同开发阶段的适用性来分解框架：

| **特性** | **Google ADK** | **CrewAI** | **LangGraph** |
| --- | --- | --- | --- |
| 核心抽象 | 生产级代理和运行时。 | 角色扮演团队（一个“船员”）。 | 状态图（一个“状态机”）。 |
| 控制流 | 计划驱动；由代理运行时管理。可以是顺序的或并行的。 | 高级过程（顺序或分层）。 | 细粒度；由图边定义。非常适合循环和分支。 |
| 状态管理 | 管理的：由代理的会话和运行时处理。 | 隐式的：通过上下文自动在任务之间传递。 | 显式的：一个中央`State`对象被传递给每个节点并由其更新。 |
| 回调和钩子 | 中间件/拦截器模式。回调作为“护栏”，旨在在 LLM 调用之前或之后拦截输入/输出。关键功能：在飞行中修改数据（例如，PII 编辑）或完全绕过 LLM（缓存）。 | 事件驱动的钩子。回调在特定的生命周期事件上触发（例如，`on_task_start`，`on_task_end`）。关键功能：在不改变核心代理逻辑的情况下进行可观察性和副作用（日志记录、更新 UI 或触发 webhook）。 | 状态监听器和中断。使用回调进行跟踪（LangSmith），但依赖于“中断”进行控制。关键功能：在特定节点暂停图（检查点）以等待 HITL 输入后再继续。 |
| 适用于... | 生产系统、企业集成（特别是 Google Cloud）、健壮且可测试的代理。 | 快速原型设计协作任务、角色定义的工作流程（例如，“研究员”、“作家”）。 | 复杂、动态的工作流程；显式错误处理；循环；和 HITL。 |

表 15.2 – ADK、CrewAI 和 LangGraph 框架之间的比较

为了更好地理解这些差异，现在让我们重新实现我们的贷款处理用例，从*第十三章*，使用我们开发笔记本中的特定代码。

# 重新实现贷款代理：实际比较

为了具体说明这些框架之间的差异，我们现在将根据附带的笔记本中的代码实现基于多代理的 idx_e4a19e5b 贷款处理系统。目标仍然是获取`applicant_id`和`document_id`，获取文档内容，并生成一个最终、可审计的贷款决定。

工作流程涉及几个不同的任务。接下来，我们将每个任务映射到我们两个实现中处理它的特定组件：

1.  **文档获取**：检索贷款申请文档的内容（**LangGraph**：`node_fetch_document` | **CrewAI**：作为初始输入/预处理传递）。

1.  **文档验证**：检查获取的文档内容是否有效且完整（**LangGraph**：`node_validate_document` | **CrewAI**：文档验证专家）。

1.  **信用检查**：根据借款人的`customer_id`值检索其信用评分（**LangGraph**：`node_check_credit` | **CrewAI**：信用检查代理）。

1.  **风险评估**：分析文档状态、信用评分和收入以确定风险水平（**LangGraph**：`node_assess_risk` | **CrewAI**：风险评估分析师）。

1.  **合规性检查**：确保最终决策符合贷款法规（**LangGraph**：`node_check_compliance` | **CrewAI**：合规性官员）。

我们将在 CrewAI 和 LangGraph 中构建此工作流程，以突出它们使用 Google 的 Gemini LLM 的不同方法。

无论框架如何，代理与外部系统交互的能力由其工具定义。对于两种实现，我们定义我们的核心业务逻辑。在提供的笔记本中，这些定义为继承自 CrewAI 的`BaseTool`的 Python 类：

```py
# --- Define Tool Classes inheriting from BaseTool ---
import json
from crewai.tools import BaseTool

class ValidateDocumentFieldsTool(BaseTool):
    name: str = "Validate Document Fields"
    description: str = (
        "Validates that the loan application JSON string contains the required fields: " "'customer_id', 'loan_amount', 'income', and 'credit_history'."
    )

    def _run(self, application_data: str) -> str:
        """Validates the application data."""
print(f"--- TOOL: Validating document fields ---")
        try:
            data = json.loads(application_data)
            required_fields = ["customer_id", "loan_amount", "income", "credit_history"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return json.dumps({"error": f"Validation failed: Missing required fields: {', '.join(missing_fields)}"})
            # Return the original data if valid
return json.dumps({"status": "validated", "data": data})
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON format in application data."})

class QueryCreditBureauAPITool(BaseTool):
    name: str = "Query Credit Bureau API"
    description: str = (
        "Simulates a call to a credit bureau API to retrieve a credit score given a customer_id."
    )

    def _run(self, customer_id: str) -> str:
        """Queries the mock credit bureau."""
print(f"--- TOOL: Calling Credit Bureau API for customer: {customer_id} ---")
        mock_credit_scores = {
            "CUST-12345": 810,  # Happy Path
"CUST-55555": 620,  # High Risk Path
"borrower_good_780": 810,
            "borrower_bad_620": 620
        }
        score = mock_credit_scores.get(customer_id)
        if score is not None:
            return json.dumps({"customer_id": customer_id, "credit_score": score})
        return json.dumps({"error": "Customer ID not found."})

class CalculateRiskScoreTool(BaseTool):
    name: str = "Calculate Risk Score"
    description: str = (
        "Calculates a risk score based on loan_amount, income, and credit_score."
    )

    def _run(self, loan_amount: int, income: str, credit_score: int) -> str:
        """Calculates the risk score."""
print(f"--- TOOL: Calculating risk score ---")
        try:
            # Attempt to parse income string (e.g., "USD 120000 a year", "$60k/month")
            income_value = int(''.join(filter(str.isdigit, income)))
            annual_income = income_value * 12 if "month" in income.lower() else income_value
        except (ValueError, TypeError):
             annual_income = 0 # Default to 0 if income cannot be parsed
if annual_income == 0:
            risk_score = 10 # Assign highest risk if income is zero or invalid
else:
            loan_to_income_ratio = loan_amount / annual_income
            risk_score = 1 # Start with base risk
if credit_score < 650: risk_score += 4
elif credit_score < 720: risk_score += 2
if loan_to_income_ratio > 0.8: risk_score += 5
elif loan_to_income_ratio > 0.5: risk_score += 2
# Cap risk score at 10
return json.dumps({"risk_score": min(risk_score, 10)})

class CheckLendingComplianceTool(BaseTool):
    name: str = "Check Lending Compliance"
    description: str = (
        "Checks the application against internal policies using credit_history and risk_score."
    )

    def _run(self, credit_history: str, risk_score: int) -> str:
        """Checks compliance rules."""
print(f"--- TOOL: Checking compliance rules (including risk score) ---")
        if credit_history == "No History":
            return json.dumps({"is_compliant": False, "reason": "Policy violation: No credit history is an automatic denial."})
        if risk_score >= 8: # Risk score of 8 or higher is non-compliant
return json.dumps({"is_compliant": False, "reason": f"Policy violation: Risk score of {risk_score} is too high for approval."})
        return json.dumps({"is_compliant": True, "reason": "Application meets all internal policy guidelines."})

# --- Instantiate the Tools ---
validate_document_fields_tool = ValidateDocumentFieldsTool()
query_credit_bureau_api_tool = QueryCreditBureauAPITool()
calculate_risk_score_tool = CalculateRiskScoreTool()
check_lending_compliance_tool = CheckLendingComplianceTool()
```

我们还需要一个辅助函数来模拟根据 ID 获取文档内容：

```py
# --- Helper Function for Mock Data ---
import json

def get_document_content(document_id: str) -> str:
    """
    Simulates fetching document content based on its ID.
    Returns a JSON STRING.
    """
print(f"--- HELPER: Simulating fetch for doc_id: {document_id} ---")
    if document_id == "document_valid_123":
        data = {
            "customer_id": "CUST-12345",
            "loan_amount": 50000,
            "income": "USD 120000 a year",
            "credit_history": "7 years"
        }
        return json.dumps(data)
    elif document_id == "document_invalid_456":
        data = {
            "customer_id": "CUST-55555",
            "loan_amount": 200000,
            # "income" is missing
"credit_history": "1 year"
        }
        return json.dumps(data)
    else:
        return json.dumps({"error": "Document ID not found."})
```

**关于** **工具输出** **patte****rns** **的说明**

你会注意到这些工具返回的是 JSON 编码的字符串，而不是 Python 字典。这是为了代理系统而故意的设计选择。由于工具输出的主要消费者通常是 LLM 本身（它处理文本标记），返回显式的 JSON 字符串确保模型接收到的结构化、可读的格式，它能够轻松解析或推理。

此外，我们在此建立了一个数据合约：该工具保证在成功的情况下返回`{"status": "validated", "data": ...}`结构，在失败的情况下返回`{"error": ...}`结构。这种一致性允许下游代理（或`CheckLendingCompliance`工具）确定性地处理错误。

**生产** **t****ip****:** **数据** **normalization pat****erns**

在此 idx_94b4e69c 示例中，`CalculateRiskScoreTool`执行基本的字符串解析以提取收入数据。在生产环境中，这种方法太脆弱。你应该在上游实现一个专门的规范化节点或预处理工具，处理货币转换、地区格式化（例如，“$100k”与“100,000 EUR”）、标准化，在数据到达风险评估代理之前。

**生产** **p****attern** **–****t****he** **p****roxy** **t****ool (*****Agent Calls Proxy Agent*****)**

上述`if/else`逻辑是一个简化的启发式方法，用于演示。在现实世界的企业架构中，此工具 idx_50ff6b73 将作为代理（参见第八章中的***Agent Calls Proxy Agent***模式）。

`_run` 方法将充当包装器，构建对外部风险决策引擎或部署的 ML 模型端点的安全 API 请求，执行调用，并解析响应。这种模式使代理保持轻量级，并确保关键业务逻辑保持集中和可管理。

现在，让我们看看 idx_3ca57924 CrewAI 和 LangGraph 如何协调这些相同的工具。

## 实现方案 1：CrewAI（协作团队）

笔记本 idx_98db1220 使用分层 idx_fd0b0efd 流程实现 CrewAI，其中 `manager` 代理将任务委派给专门的代理：

1.  定义 LLM 和代理。

    首先，我们使用 CrewAI 的 `LLM` 抽象配置 Gemini LLM。然后，我们定义具有特定工具的专家代理 idx_332be4e2，以及没有工具但可以委派的管理代理 idx_a19bd134：

    ```py
    import os
    import json
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import BaseTool
    # Assume tools (ValidateDocumentFieldsTool, etc.) and get_document_content are defined above
    # --- Initialize the LLM ---
    # Assumes GOOGLE_API_KEY environment variable is set
    llm = LLM(
        model='gemini/gemini-2.5-flash', # Or another Gemini model
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0
    )

    # --- Define Agents ---
    # 1\. Document Validation Agent
    doc_specialist = Agent(
        role="Document Validation Specialist",
        goal="Validate the completeness and format of a new loan application provided as a JSON string.",
        backstory=(
            "You are a meticulous agent responsible for the first step of loan processing. "
    "Your sole task is to receive a JSON string, call the `Validate Document Fields` tool, "
    "and return its exact JSON output. You do not talk to the user or other agents."
        ),
        tools=[validate_document_fields_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # 2\. Credit Check Agent
    credit_analyst = Agent(
        role="Credit Check Agent",
        goal="Query the credit bureau API to retrieve an applicant's credit score.",
        backstory=(
            "You are a specialized agent that interacts with the Credit Bureau. "
    "Your sole task is to receive a `customer_id`, call the `Query Credit Bureau API` tool, "
    "and return its exact JSON output."
        ),
        tools=[query_credit_bureau_api_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # 3\. Risk Assessment Agent
    risk_assessor = Agent(
        role="Risk Assessment Analyst",
        goal="Calculate the financial risk score for a loan application.",
        backstory=(
            "You are a quantitative analyst agent. Your sole task is to receive `loan_amount`, `income`, and `credit_score`, "
    "call the `Calculate Risk Score` tool, and return its exact JSON output."
        ),
        tools=[calculate_risk_score_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # 4\. Compliance Agent
    compliance_officer = Agent(
        role="Compliance Officer",
        goal="Check the application against all internal lending policies and compliance rules.",
        backstory=(
            "You are the final checkpoint for policy and compliance. Your sole task is to receive `credit_history` and `risk_score`, "
    "call the `CheckLendingCompliance` tool, and return its exact JSON output."
        ),
        tools=[check_lending_compliance_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    # 5\. Manager Agent (for the final report)
    manager = Agent(
        role="Loan Processing Manager",
        goal="Manage the loan application workflow and compile the final report.",
        backstory=(
            "You are the manager responsible for orchestrating the "
    "loan processing pipeline, ensuring data flows correctly, and formulating the "
    "final decision and report based on your team's findings."
        ),
        llm=llm,
        allow_delegation=True, # The manager delegates tasks
        verbose=True
    )
    ```

**使用温度最小化** **方差**

我们故意将此代理的 `temperature=0.0` 设置。在依赖于精确工具使用和结构化输出的代理工作流程中（例如 JSON），最小化随机性至关重要。请注意，虽然 `temperature=0.0` 显著减少了方差，但由于 GPU 上浮点运算的固有非确定性，它并不能保证 100% 的确定性行为。然而，它为逻辑和协调任务提供了可能的最大稳定性。

1.  定义任务。

    接下来，我们定义 idx_156cb190 任务。注意 `task_validate` 的描述中包含一个占位符 `{``document_content``}`，它将接收 idx_7d5fad20 获取的 JSON 字符串作为输入。`context` 参数隐式地在依赖任务之间传递输出：

    ```py
    # Define input document IDs for testing
    loan_application_doc_ids = {
        "valid": "document_valid_123",
        "invalid": "document_invalid_456"
    }

    # Task 1: Validate Document Content
    task_validate = Task(
        description=(
            "Validate the loan application, which is provided as a JSON string: '{document_content}'. "
    "You MUST pass this entire JSON string directly to the 'Validate Document Fields' tool."
        ),
        expected_output="A JSON string with the validation status and all extracted data ('status': '...', 'data': {...}) or an error message.",
        # Agent is not assigned here; manager will delegate
    )

    # Task 2: Check Credit
    task_credit = Task(
        description=(
            "1\. Parse the JSON output from the validation task. \\n"
    "2\. Extract the `customer_id` from its 'data' field. \\n"
    "3\. Call the `Query Credit Bureau API` tool with this `customer_id`."
        ),
        expected_output="A JSON string containing the customer_id and their credit_score.",
        context=[task_validate] # Depends on task_validate
    )

    # Task 3: Assess Risk
    task_risk = Task(
        description=(
            "1\. Parse the JSON output from the validation task to get `loan_amount` and `income`. \\n"
    "2\. Parse the JSON output from the credit check task to get `credit_score`. \\n"
    "3\. Call the `Calculate Risk Score` tool with these three values."
        ),
        expected_output="A JSON string containing the calculated risk_score.",
        context=[task_validate, task_credit] # Depends on two tasks
    )

    # Task 4: Check Compliance
    task_compliance = Task(
        description=(
            "1\. Parse the JSON output from the validation task to get `credit_history`. \\n"
    "2\. Parse the JSON output from the risk assessment task to get `risk_score`. \\n"
    "3\. Call the `Check Lending Compliance` tool with these two values."
        ),
        expected_output="A JSON string with the compliance status (is_compliant: true/false) and a reason.",
        context=[task_validate, task_risk] # Depends on two tasks
    )

    # Task 5: Compile Final Report
    task_report = Task(
        description=(
            "Compile a final loan decision report synthesizing all findings from the previous tasks. "
    "The report must include: \\n"
    "- The final decision (Approve/Deny). \\n"
    "- A clear justification for the decision, referencing the validation status, "
    "credit score, risk score, and compliance check."
        ),
        expected_output="A comprehensive final report in Markdown format.",
        context=[task_validate, task_credit, task_risk, task_compliance] # Depends on all tasks
    )
    ```

1.  组装 idx_5859e0c9 并运行机组。

    最后，我们组装 idx_4987e6da `Crew`，指定 `hierarchical` 流程并分配 `manager_agent`。在启动机组之前获取文档内容，并将其作为输入传递：

    ```py
    # Assemble the crew
    loan_crew = Crew(
        agents=[doc_specialist, credit_analyst, risk_assessor, compliance_officer], # Manager assigned below
        tasks=[task_validate, task_credit, task_risk, task_compliance, task_report],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True
    )

    # --- Run with VALID inputs ---
    print("--- KICKING OFF CREWAI PROCESS (VALID INPUTS) ---")
    valid_json_content = get_document_content(loan_application_doc_ids['valid'])
    inputs_valid = {'document_content': valid_json_content}
    result_valid = loan_crew.kickoff(inputs=inputs_valid)
    print("\n\n--- CREWAI FINAL REPORT (VALID) ---")
    print(result_valid)

    # --- Run with INVALID inputs ---
    print("\n\n--- KICKING OFF CREWAI PROCESS (INVALID INPUTS) ---")
    invalid_json_content = get_document_content(loan_application_doc_ids['invalid'])
    inputs_invalid = {'document_content': invalid_json_content}
    result_invalid = loan_crew.kickoff(inputs=inputs_invalid)
    print("\n\n--- CREWAI FINAL REPORT (INVALID) ---")
    print(result_invalid)
    ```

CrewAI 的分层 idx_cdb25b6c 方法允许管理代理 idx_70d5a1cc 协调工作流程。管理代理根据任务描述和可用工具将每个任务委派给适当的专家代理。错误处理有些隐式；如果 `task_validate` 返回错误（例如在无效情况下缺少 `'income'` 字段），依赖于其输出的后续任务可能仍然运行，但很可能会失败或产生错误结果，因为管理代理试图继续进行。在无效情况下，最终报告反映了验证失败，但中间步骤（信用检查、风险评估）仍然执行，可能执行不必要的操作。

## 实现方案 2：LangGraph（状态机）

LangGraph 的 idx_55658972 实现使用显式状态机。我们为每个步骤定义 idx_a62016c4 节点，包括获取文档，并使用条件边进行健壮的错误处理：

1.  定义状态。

    使用 `TypedDict` 定义 `LoanGraphState`，包括整个过程中工具和节点所需的所有字段：

    ```py
    #@title 2.1: Define LangGraph State
    import typing
    import json

    class LoanGraphState(typing.TypedDict):
        """
        Represents the state of our loan processing graph.
        It contains all the data that needs to be passed between nodes.
        """
        applicant_id: str # Initial input, may not be directly used if customer_id is in doc
        document_id: str # Initial input
        document_content: str # Fetched content (JSON string)
    # Data extracted or generated by tools/nodes
        validation_status: str
        customer_id: str
        loan_amount: int
        income: str
        credit_history: str
        credit_score: int
        risk_score: int
        risk_level: str # Added for LLM-based risk assessment output
        compliance_status: str
    # Final output
        final_decision: str # Simplified final report/decision string
        error: str # To track errors explicitly
    ```

1.  定义图节点。

    我们将 idx_07537b99Python 函数定义为节点。`node_fetch_document` 模拟获取内容。`node_validate_document` 调用验证 idx_90a88605 工具，并使用提取的数据 *或* 错误更新状态。后续节点在继续之前会检查错误。`node_assess_risk` 直接使用 LLM 生成风险评估：

    ```py
    #@title 2.2: Define LangGraph Nodes
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Re-initialize LLM specifically for LangGraph (using LangChain's integration)
    lg_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Or your preferred Gemini model
    # Node 0: Fetch Document Content
    def node_fetch_document(state: LoanGraphState):
        print("--- NODE: Fetching Document ---")
        doc_id = state["document_id"]
        try:
            content = get_document_content(doc_id)
            # Check if the helper returned an error (e.g., document not found)
            content_json = json.loads(content)
            if "error" in content_json:
                print(f"    Error fetching document: {content_json['error']}")
                return {
                    "error": f"Failed to fetch document {doc_id}: {content_json['error']}",
                    "document_content": ""
                }
            return {"document_content": content}
        except Exception as e:
            print(f"    Error during document fetch node: {e}")
            return {"error": f"Critical error fetching document {doc_id}", "document_content": ""}

    # Node 1: Validate Document
    def node_validate_document(state: LoanGraphState):
        print("--- NODE: Validating Document ---")
        # Check if fetch already failed
    if state.get("error"):
            return {"validation_status": "SKIPPED due to fetch error"}

        doc_content = state["document_content"]
        try:
            result_str = validate_document_fields_tool._run(application_data=doc_content)
            result_json = json.loads(result_str)

            if "error" in result_json:
                validation_status = f"Validation FAILED: {result_json['error']}"
    print(f"  -> {validation_status}")
                # Explicitly set error state
    return {"validation_status": validation_status, "error": validation_status}
            else:
                validation_status = result_json.get('status', 'Validation PASSED')
                app_data = result_json.get("data", {})
                print(f"  -> {validation_status}")
                # Update state with extracted data
    return {
                    "validation_status": validation_status,
                    "customer_id": app_data.get("customer_id"),
                    "loan_amount": app_data.get("loan_amount"),
                    "income": app_data.get("income"),
                    "credit_history": app_data.get("credit_history"),
                    "error": None # Clear any previous error if validation succeeds
                }
        except Exception as e:
            validation_status = f"Critical error during validation node: {e}"
    print(f"  -> {validation_status}")
            return {"validation_status": validation_status, "error": validation_status}

    # Node 2: Check Credit
    def node_check_credit(state: LoanGraphState):
        print("--- NODE: Checking Credit ---")
        if state.get("error"): # Skip if validation failed
    return {"credit_score": -1}

        cust_id = state["customer_id"]
        try:
            result_str = query_credit_bureau_api_tool._run(customer_id=cust_id)
            result_json = json.loads(result_str)
            if "error" in result_json:
                print(f"  -> Error: {result_json['error']}")
                # Set error state if credit check fails
    return {"credit_score": -1, "error": f"Credit check failed: {result_json['error']}"}
            score = result_json.get("credit_score", -1)
            print(f"  -> Credit Score: {score}")
            return {"credit_score": score, "error": None} # Clear error on success
    except Exception as e:
            print(f"    Critical error during credit check node: {e}")
            return {"error": "Critical error in credit check tool.", "credit_score": -1}

    # Node 3: Assess Risk (LLM-Powered)
    def node_assess_risk(state: LoanGraphState):
        print("--- NODE: Assessing Risk (LLM-Powered) ---")
        if state.get("error"):
            return {"risk_score": -1, "risk_level": "UNKNOWN"}

        prompt = ChatPromptTemplate.from_template(
            """You are a senior loan underwriter. Assess the financial risk based on:
            - Validation Status: {validation}
            - Credit Score: {credit}
            - Loan Amount: {amount}
            - Applicant Income: {income}
            - Credit History: {history}
            Provide a one-sentence justification, then conclude with the risk level: LOW, MEDIUM, or HIGH.
            Example:
            Justification: Applicant has excellent credit and low debt-to-income.
            Risk: LOW
            """
        )
        parser = StrOutputParser()
        risk_chain = prompt | lg_llm | parser

        try:
            result_str = risk_chain.invoke({
                "validation": state.get("validation_status", "N/A"),
                "credit": state.get("credit_score", "N/A"),
                "amount": state.get("loan_amount", "N/A"),
                "income": state.get("income", "N/A"),
                "history": state.get("credit_history", "N/A")
            })
            print(f"  -> LLM Assessment Output:\n{result_str}")

            # Basic parsing
            risk_level = "UNKNOWN"
    if "LOW" in result_str.upper(): risk_level = "LOW"
    elif "MEDIUM" in result_str.upper(): risk_level = "MEDIUM"
    elif "HIGH" in result_str.upper(): risk_level = "HIGH"

            score_map = {"LOW": 3, "MEDIUM": 6, "HIGH": 9, "UNKNOWN": 10}
            risk_score = score_map.get(risk_level, 10)
            print(f"  -> Parsed Risk Level: {risk_level}, Score: {risk_score}")

            return {"risk_score": risk_score, "risk_level": risk_level, "error": None}
        except Exception as e:
            print(f"    Critical error during LLM risk assessment node: {e}")
            return {"error": "Critical error in LLM Risk assessment.", "risk_score": -1, "risk_level": "UNKNOWN"}

    # Node 4: Check Compliance
    def node_check_compliance(state: LoanGraphState):
        print("--- NODE: Checking Compliance ---")
        if state.get("error"):
            return {"compliance_status": "SKIPPED due to prior error."}

        try:
            result_str = check_lending_compliance_tool._run(
                credit_history=state["credit_history"],
                risk_score=state["risk_score"]
            )
            result_json = json.loads(result_str)
            status = result_json.get("reason", "Check FAILED")
            print(f"  -> Compliance Status: {status}")
            # Set error if non-compliant, otherwise clear it
            error_msg = status if result_json.get("is_compliant") is False else None
    return {"compliance_status": status, "error": error_msg}
        except Exception as e:
            print(f"    Critical error during compliance check node: {e}")
            return {"error": "Critical error in compliance check tool.", "compliance_status": "Check FAILED due to tool error."}

    # Node 5: Compile Final Report (Handles success path)
    def node_compile_report(state: LoanGraphState):
        print("--- NODE: Compiling Success Report ---")
        # This node is only reached if all previous steps succeeded without setting the error state
        decision = "Approve"
        reason = (f"Approved based on:\n"
    f" - Validation: {state.get('validation_status', 'N/A')}\n"
    f" - Credit Score: {state.get('credit_score', 'N/A')}\n"
    f" - Risk Assessment: {state.get('risk_level', 'N/A')} (Score: {state.get('risk_score', 'N/A')})\n"
    f" - Compliance: {state.get('compliance_status', 'N/A')}")

        report = f"FINAL DECISION: {decision}\nREASON: {reason}"
    return {"final_decision": report.strip()}

    # Node 6: Compile Rejection Report (Handles any failure path)
    def node_compile_rejection(state: LoanGraphState):
        print("--- NODE: Compiling Rejection Report ---")
        decision = "Deny"
        reason = f"Denied due to error: {state.get('error', 'Unknown error during processing.')}"
    # Add more specific reasons based on which stage failed if needed
    if "Validation FAILED" in state.get("validation_status", ""):
            reason = f"Denied due to validation failure: {state.get('validation_status', '')}"
    elif "Credit check failed" in state.get("error", ""):
            reason = f"Denied due to credit check failure: {state.get('error', '')}"
    elif "compliance" in state.get("error", "").lower(): # Check if compliance node set the error
            reason = f"Denied due to compliance failure: {state.get('compliance_status', '')}"

        report = f"FINAL DECISION: {decision}\nREASON: {reason}"
    return {"final_decision": report.strip()}
    ```

    **生产** **t****ip: 强制** **s****tructured** **o****utput**

    在这个例子中，我们使用 idx_8facd911 简单的字符串匹配来解析 LLM 的响应。在一个 idx_3f0460fc 生产系统中，这很危险，因为模型可能 idx_61296de6 很冗长（例如，`"The risk is relatively LOW"`）。对于企业应用，你应该使用结构化输出特征（由 LangChain 和 Gemini 都支持）。通过向模型传递 Pydantic 模式，你可以强制它返回一个有效的 JSON 对象（例如，`{"``risk_level``": "LOW"}`），确保输出与你的下游需求匹配，无需脆弱的字符串解析逻辑。

1.  定义图及其边。

    我们将 idx_b36170c4 节点连接起来，从 `fetch_doc` 开始。关键的是，我们在 `fetch_doc` 和 `validate_doc` 之后添加了条件边，这些边 idx_979ed835 检查状态中的 `error` 字段，如果发生错误，则立即路由到 `compile_rejection`。

    ```py
    #@title 2.3: Define and Compile the Graph
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(LoanGraphState)

    # Add nodes
    workflow.add_node("fetch_doc", node_fetch_document)
    workflow.add_node("validate_doc", node_validate_document)
    workflow.add_node("check_credit", node_check_credit)
    workflow.add_node("assess_risk", node_assess_risk)
    workflow.add_node("check_compliance", node_check_compliance)
    workflow.add_node("compile_report", node_compile_report)       # Success end node
    workflow.add_node("compile_rejection", node_compile_rejection) # Failure end node
    # Set entry point
    workflow.set_entry_point("fetch_doc")

    # Define conditional edge logic
    def decide_after_fetch(state: LoanGraphState):
        return "reject" if state.get("error") else "continue"
    def decide_after_validation(state: LoanGraphState):
        return "reject" if state.get("error") else "continue"
    def decide_after_credit_check(state: LoanGraphState):
         return "reject" if state.get("error") else "continue"
    def decide_after_risk(state: LoanGraphState):
        # Even if risk is HIGH, we proceed to compliance check,
    # but compliance node might set error state.
    return "reject" if state.get("error") else "continue"
    def decide_after_compliance(state: LoanGraphState):
         # If compliance node set an error (e.g., non-compliant), reject.
    return "reject" if state.get("error") else "continue"
    # Add edges
    workflow.add_conditional_edges("fetch_doc", decide_after_fetch, {"continue": "validate_doc", "reject": "compile_rejection"})
    workflow.add_conditional_edges("validate_doc", decide_after_validation, {"continue": "check_credit", "reject": "compile_rejection"})
    workflow.add_conditional_edges("check_credit", decide_after_credit_check, {"continue": "assess_risk", "reject": "compile_rejection"})
    workflow.add_conditional_edges("assess_risk", decide_after_risk, {"continue": "check_compliance", "reject": "compile_rejection"})
    workflow.add_conditional_edges("check_compliance", decide_after_compliance, {"continue": "compile_report", "reject": "compile_rejection"})

    # Define end points
    workflow.add_edge("compile_report", END)
    workflow.add_edge("compile_rejection", END)

    # Compile
    try:
       app = workflow.compile()
       print("LangGraph Compiled Successfully!")
       # Optional: Visualize
    # from IPython.display import Image, display
    # display(Image(app.get_graph().draw_mermaid_png()))
    except Exception as e:
       print(f"Error compiling LangGraph: {e}")
       app = None
    ```

1.  运行 idx_c18e160d 该图。

    我们使用 `.stream()` 来运行编译后的图（`app`），以观察有效和无效文档 ID 的状态转换：

    ```py
    #@title 2.4: Run the LangGraph Workflow
    if app is None:
        print("LangGraph app not compiled. Skipping execution.")
    else:
        # --- Test 1: Valid Data ---
    print("\n--- LANGGRAPH RUN 1: VALID DOCUMENT ---")
        inputs_valid = {
            "applicant_id": "borrower_good_780", # Included but might not be used if customer_id is preferred
    "document_id": "document_valid_123",
        }
        print("Streaming intermediate steps (Valid):")
        for s_chunk in app.stream(inputs_valid, {"recursion_limit": 10}):
            step_name = list(s_chunk.keys())[0]
            print(f" Step: {step_name}") # Simpler logging
    # print(f"   Output: {s_chunk[step_name]}") # Uncomment for full state change detail
    print("-" * 10)

        print("\nInvoking for final state (Valid)...")
        final_state_valid = app.invoke(inputs_valid, {"recursion_limit": 10})
        print("\n--- LANGGRAPH FINAL REPORT (VALID) ---")
        print(final_state_valid.get('final_decision', 'Final decision not found.'))
        # print("\nFull Final State (Valid):", final_state_valid) # Uncomment to see full state
    # --- Test 2: Invalid Data ---
    print("\n\n--- LANGGRAPH RUN 2: INVALID DOCUMENT ---")
        inputs_invalid = {
            "applicant_id": "borrower_bad_620",
            "document_id": "document_invalid_456",
        }
        print("Streaming intermediate steps (Invalid):")
        for s_chunk in app.stream(inputs_invalid, {"recursion_limit": 10}):
            step_name = list(s_chunk.keys())[0]
            print(f" Step: {step_name}")
            # print(f"   Output: {s_chunk[step_name]}")
    print("-" * 10)

        print("\nInvoking for final state (Invalid)...")
        final_state_invalid = app.invoke(inputs_invalid, {"recursion_limit": 10})
        print("\n--- LANGGRAPH FINAL REPORT (INVALID) ---")
        print(final_state_invalid.get('final_decision', 'Final decision not found.'))
        # print("\nFull Final State (Invalid):", final_state_invalid)
    ```

这个 LangGraph 实现展示了显式的控制流和状态管理。添加 `node_fetch_document` 使得过程干净地开始。基于状态中的 `error` 键的条件边确保，如果获取或验证失败，图会立即路由到 `compile_rejection` 节点，防止对无效数据进行不必要的工具调用（如信用检查或风险评估）。

这种显式的 idx_b128967brouting 提供了实际的效率提升：它 idx_65a6bbde 通过在失败时立即停止执行来消除不必要的 API 成本和延迟，这与我们的 CrewAI 示例形成直接对比，其中中间代理尽管初始验证错误仍然继续运行。

在 `node_assess_risk` 中直接使用 LLM 展示了 LangGraph 如何将生成步骤与确定性工具调用相结合。这种基于图的方案与 CrewAI 的简单流程相比，提供了更优越的鲁棒性和可追溯性，尤其是在错误处理方面。

让我们现在讨论可观察性和负责任 AI 的考虑因素。

# 可观察性和负责任 AI 的考虑因素

选择框架不仅仅是关于开发体验；它是关于你管理、监控和治理结果应用的能力。在代理 AI 的背景下，非确定性是一个因素，可观察性是负责任 AI 的基石。如果你无法追踪代理做出决策的原因，你就无法确保它是公平的、安全的或符合规定的。

## 实践中的可观察性

每个框架都利用特定的工具和协议来提供调试复杂代理交互和维护可验证审计轨迹所需的可见性：

+   **LangGraph** 和 **CrewAI**（含 **LangSmith**）：LangChain 生态系统，包括 LangGraph 和 CrewAI，旨在与 LangSmith 原生集成。LangSmith 是一个专门为跟踪复杂 LLM 应用而设计的可观察性平台。由于 LangGraph 的状态是显式的，它在 LangSmith 中的跟踪信息非常详细，允许您通过查看每个节点的完整状态和 LLM 调用来进行“时间旅行”调试。CrewAI 的跟踪也受益于 LangSmith，它显示了代理的动作和工具调用。这为代理的思考和行动提供了完整的审计轨迹，这对于调试和可解释性至关重要。

+   **Google ADK**: 作为面向生产的工具包，ADK 集成了 **OpenTelemetry**，这是行业标准的跟踪和指标工具。这使得它能够直接集成到企业级监控解决方案，例如 **Google Cloud** 的 **operations suite**（云跟踪和云日志）。这使代理更像是一个可管理的微服务，而不是脚本，这对于企业治理至关重要。

## 启用负责任的人工智能

负责任的人工智能原则包括公平性、透明度、问责制和安全。这些不是抽象的目标；它们是通过具体的架构选择实现的，并通过持续的治理和监督在组织中实施，以确保以下原则和约束的实施：

+   **透明度和可解释性**：LangGraph 的显式状态图是一种可解释性形式。该图本身记录了决策逻辑，最终状态对象包含用于得出结论的所有中间数据。我们贷款代理的最终报告，包括理由，是此可追溯过程的直接输出：

    +   **人口统计差异测试场景**：在 CI/CD 管道中通过 ADK 的测试框架引入公平性评估步骤。这将在来自不同人口统计的精选“黄金数据集”用户查询上运行代理，以在版本升级之前从数学上衡量响应质量（例如，有用性、语气）是否在不同用户组之间保持一致。

    +   **推理透明度场景**：利用 Google Cloud 控制台中的 **Trace** 视图（与 ADK 部署相关联）。这揭示了代理的内部“思考、行动、观察”循环，使开发者能够确切地看到代理为什么选择调用 `getUserBalance` 工具而不是 `getLoanStatus` 工具，而不仅仅是看到最终答案。

    +   **显式状态图可视化场景**：在部署贷款代理之前，对“黄金数据集”中的多样化申请人档案运行模型，以确保审批逻辑不会基于受保护属性（例如，邮编或性别）表现出不同的影响，即使这些属性没有明确用作特征。

+   **安全和稳健性**：在我们的 LangGraph 实现中使用条件边缘作为程序化安全模式，通过强制执行“快速失败”逻辑。而不是允许 LLM 在不完整或损坏的数据上继续推理，图监控状态对象中的专用`error` `key`。如果检测到错误，例如验证期间缺少收入字段，条件边缘会立即将执行流程重定向到终端拒绝节点。这防止了下游代理尝试处理无效输入，这显著降低了模型产生决策幻觉或基于错误信息进行未经授权的 API 调用的风险。

    +   **安全防护和 PII 保护场景**：在 ADK 模型参数中明确配置`safety_settings`，以在`BLOCK_LOW_AND_ABOVE`阈值阻止“仇恨言论”或“骚扰”。此外，实施特定的输入防护措施（如 PII 检测），在敏感数据到达 LLM 上下文窗口之前拦截并删除这些数据。

    +   **条件边缘防护场景**：在图中硬编码一个`stop`条件（一个条件边缘），如果收入验证 API 返回 null 或负值，则立即停止进程，防止 LLM 基于错误数据进行信用决策的幻觉。

+   **问责制和治理**：可追溯、可观察的工作流程是问责制的先决条件。当审计员询问为什么拒绝贷款时，您可以提供 LangSmith 或 Google Cloud Trace 的完整 idx_0425f9a3 跟踪，显示每个步骤的确切数据、工具输出和 LLM 推理（特别是在 LangGraph 的显式状态下）。这使代理从“黑盒”转变为业务流程中透明、可审计的组成部分：

    +   **不可变审计跟踪场景**：启用数据访问日志并将所有代理交互日志导出到 BigQuery。这创建了一个不可变记录，其中每个代理发出的 API 调用都被时间戳记录，并关联到特定的服务账户身份，使合规团队能够查询特定交易是何时以及由谁授权的。

    +   **完整执行跟踪（LangSmith/Cloud Trace）场景**：当审计员质疑特定的贷款拒绝时，从 LangSmith 检索记录特定提示的确切跟踪 ID，以及检索到的信用评分和导致`Denied`输出的 LLM 的中间推理步骤。

现在我们已经通过更新的代码看到了这些框架的实际应用，让我们讨论如何为您的项目选择正确的框架。

# 选择框架的建议

没有一个单一的“最佳”代理框架。正确的选择取决于您项目的复杂性、您团队对概念的了解以及您的生产需求。然而，一个关键的长远策略是避免框架锁定。代理领域是动态变化的；今天的领导者明天可能就会被淘汰。我们建议围绕稳定的接口设计您的系统，例如标准化的工具定义（合约）、显式的状态模式以及基于模式的编排逻辑，而不是将每个组件紧密耦合到特定框架的专有类。这种抽象允许您在需求演变时以最小的摩擦迁移或交换框架。

因此，让我们看看 idx_78ece9fc 应该考虑哪个框架：

+   **当...时考虑 CrewAI**

    +   您正在快速原型设计，并希望快速运行一个多代理系统

    +   您的工作流程自然映射到一个协作的专业团队（例如，“研究人员”、“作家”、“编辑”）

    +   您的过程受益于一个分层（经理/工人）结构，其中委托是关键

    +   通过上下文隐式传递状态足以满足您的需求

+   **当...时考虑 LangGraph**

    +   您需要复杂、非线性的控制流（循环、分支、基于状态的动态路由）

    +   在每个步骤进行显式的状态管理和检查对于逻辑或调试至关重要

    +   需要具有基于失败的具体路由的鲁棒错误处理

    +   您需要高保真调试和可追溯性（在每一步都能看到完整的状态）

    +   您正在构建需要精确状态控制的长期运行的代理

    +   您需要通过添加等待输入的节点轻松实现 HITL 模式

+   **当...时考虑 Google ADK**

    +   您正在为生产企业环境构建，特别是在 Google Cloud 生态系统内

    +   您需要一个更结构化、以软件工程为中心的方法，将代理视为模块化、可测试和可部署的组件

    +   与标准企业可观察性（如 OpenTelemetry）和治理系统的集成是主要要求

    +   您需要管理代理的整个生命周期，从开发、评估到部署和监控

    +   您需要检查进入工具、代理和模型的负载；检查或对工具、代理和模型的输出采取行动

    +   您需要通过利用对您的 LLMs 的非确定性推理施加确定性结构的专用工作流程代理来编排复杂的多元代理模式，例如顺序管道、并行扇出或迭代循环

最终，框架是一个实现我们讨论过的模式的工具。通过理解其核心抽象，您可以选择最适合您试图解决的问题的一个：

+   CrewAI 的*团队*

+   LangGraph 的*状态机*

+   ADK 的*生产服务*

让我们现在将这些框架映射到我们在整本书中讨论过的不同层次的代理成熟度。

# 框架作为代理成熟度的推动者

现在我们已经探讨了构建代理的实际工具，我们可以将这些框架 idx_e31c6025 映射到我们在*第一章*中引入的 GenAI 成熟度模型。这些工具是推动者，帮助组织从基本、数据增强的生成（第 2 级）发展到真正自主和协作的代理系统（第 4 级和第 5 级）。

下表概述了如何处理每个级别，重点介绍本章讨论的框架如何加速高级别、代理级别的发展：

| **成熟度级别** | **描述** | **框架方法/** **启用工具** |
| --- | --- | --- |
| Level 1 – 提示 | 简单、单轮提示 | 工具：直接 LLM API 调用（例如，Gemini、OpenAI）。通常不需要框架。 |
| Level 2 – RAG | 增强上下文生成 (RAG) | 工具：LangChain（用于 RAG 管道），或调用向量数据库并将上下文插入提示的定制代码。 |
| Level 3 – 调优 |  | 对代理框架不适用 |

| Level 4 – 定位和评估 |  | CrewAI 和 LangGraph 代表两种不同的哲学：CrewAI 专注于高级别、基于角色的编排，而 LangGraphidx_87977c1c 提供了一个低级别、状态驱动的框架。它们对评估和定位的方法反映了这种分歧。

+   **CrewAI**：**集成和** **企业导向**

CrewAI 具有专门的功能，使开发者在无需构建自定义逻辑的情况下，更容易实现定位和评估，以防止幻觉。

+   **定位和** **防护栏**：

+   **幻觉** **防护栏** **（企业版）**：原生功能，分配忠诚度分数（0–10），如果低于阈值则触发自我校正。

+   **内置 RAG 和** **知识**：原生知识组件允许代理默认基于本地数据进行定位。

+   **原生** **实用** **工具**：如 `TimeAwarenessTool` 之类的工具将代理定位在现实世界事实（例如，当前日期）中，以防止时间幻觉。

+   **评估**：

+   **CrewAI** **测试 CLI**：运行 `$N$` 次迭代以生成性能评分表。

+   **Patronus AI** **和** **训练** **循环**：对自动化评估的一流支持，以及 `crew.train()` 方法，用于基于人类反馈的微调。

+   **LangGraph**：**架构和** **开发者驱动**

LangGraph 将定位和评估视为可定制的结构组件，提供了对“推理路径”的精细控制。

+   **定位和** **防护栏**：

+   **自我** **校正** **循环**：使用条件边来检测较差的输出并将状态路由回“精炼”节点，创建程序化的定位循环。

+   **状态** **检查点**：原生持久性允许系统基于“版本控制”的对话历史进行定位，从而实现回滚到已知良好状态。

+   **HITL (Human-in-the-Loop)**: 在高风险工具调用之前，显式“中断”暂停执行以供人类验证。

+   **评估**：

    +   **LangSmith****i****ntegration**: 深度跟踪级评估，其中每个节点转换都使用“LLM 作为裁判”模式测量延迟、成本和准确性。

    +   **单元可测试节点**：由于节点是隔离的 Python 函数，开发人员可以在完全系统集成之前对特定的逻辑门进行确定性单元测试。

|

| 第 5 级 – 单代理系统 | 具有规划器、工具和记忆的自主代理执行多步任务 |
| --- | --- |

+   **LangGraph**：一个包含一个或多个代理节点且可以基于显式状态调用多个工具的图，可能循环（反映）。

+   **ADK**: 主要用例。定义一个包含其工具的单一`Agent`类，并在代理运行时中运行它。

+   **CrewAI**：可以使用一个“船员”，但这不太常见。

|

| 第 6 级 – 多代理系统 | 多个代理协作、协商并将任务委托以解决复杂问题 |
| --- | --- |

+   **LangGraph**: 适用于复杂交互。每个代理/函数是一个节点。边定义通信、交接和控制流。显式状态促进共同理解。

+   **CrewAI**: 主要设计理念。定义一个具有不同角色和协作过程（顺序/分层）的船员。

+   **ADK**: 一个由多个独立部署的 ADK 代理服务组成的系统，通过消息或 A2A 协议进行通信。

|

表 15.3 – 根据组织的成熟度级别接近框架

如此表所示，框架是“代理就绪”模型（第 3 级）到功能代理系统（第 4 级和第 5 级）的桥梁。它们提供了构建我们设计的复杂 idx_5aba42fd 应用程序所必需的“如何”。

让我们在下一节中总结本章内容。

# 摘要

在本章中，我们探讨了三个突出的代理框架——谷歌的 ADK、CrewAI 和 LangGraph——并了解了每个框架如何提供不同的但强大的抽象，以使用谷歌 Gemini 作为我们的 LLM 构建复杂的代理系统。

我们基于修订的笔记本对贷款处理代理的实用实现进行了更新，展示了 CrewAI 通过分层过程建模协作团队的优势，而 LangGraph 通过其状态机方法展示了细粒度控制、显式状态管理和健壮的错误处理。我们还定位谷歌的 ADK 作为一个面向企业级的工具包，专注于构建、测试和部署健壮、可管理的代理服务的整个生命周期。

我们将这些框架的使用与 GenAI 成熟度模型联系起来，将它们识别为达到 5 级（单代理系统）和 6 级（多代理系统）的关键推动者。最后，我们强调，这些先进系统需要成熟的可观察性和治理方法，使用 LangSmith 和 OpenTelemetry 等工具确保可追溯性，这是负责任 AI 的核心组成部分。

本章的关键要点如下：

+   **框架是加速器**：您无需从头开始构建代理规划器、状态管理器和工具调度器。CrewAI、LangGraph 和 ADK 等框架提供了构建 4 级和 5 级系统所需的基本抽象。

+   **选择正确的抽象**：您选择的框架应与您的问题相匹配。使用 CrewAI 的 *团队* 比喻进行基于角色的协作任务，特别是涉及委派。使用 LangGraph 的 *状态机* 处理需要显式状态和精细控制流程和错误的复杂、循环过程。使用 ADK 的 *生产代理* 模型为企业级、可测试和可管理的服务。

+   **控制流和错误处理是关键区别**：超越简单的序列对于稳健性至关重要。LangGraph 的显式图结构提供了一种强大的方式来管理复杂的分支、循环和错误处理（如我们的验证示例所示），这对于可靠的应用程序至关重要。CrewAI 的分层流程提供了一个更简单的委派模型。

+   **可观察性是不可或缺的**：代理系统是复杂的。能够追踪代理做出决策的原因，得益于 LangSmith（尤其是与 LangGraph 的显式状态相结合）等工具，这不仅是一个调试功能，而且是治理、安全和负责任 AI 的基础要求。

我们现在已经从 GenAI 的基础概念到构建复杂、自主代理系统的架构模式和实用框架进行了探索。在最后一章中，我们将汇集这些概念，为您提供明确的行动计划，以应用这些模式，导航成熟度模型，并领导您组织的转型。

# 获取本书的 PDF 版本和独家额外内容

扫描二维码（或访问 [packtpub.com/unlock](https://packtpub.com/unlock)）。通过名称搜索本书，确认版本，然后按照页面上的步骤操作。

![Image](img/B33147_15_1.png)

![Image](img/B33147_15_2.png)

*注意：请妥善保管您的发票。直接从 Packt 购买的产品不需要发票。*
