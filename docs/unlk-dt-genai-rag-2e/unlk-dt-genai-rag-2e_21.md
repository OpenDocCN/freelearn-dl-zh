

# 第十八章：使用 LangMem 的 RAG 程序记忆

在我们探索代理记忆的*第十六章*中，我们看到了 CoALA 框架如何将认知组织成四种互补的记忆类型：工作记忆、情景记忆、语义记忆和程序记忆。工作记忆基本上是聊天历史。在第*第十七章*中，我们将情景记忆和语义记忆应用于动手实验室。现在我们来到了将一个有能力的代理转变为自主代理的部分。程序记忆，执行任务的知识编码和行为模式，标志着仅仅记住的代理和能够学习、适应和自我改进的代理之间的区别。就本书的主要主题 RAG 而言，本章将向您展示即使是生成式 AI 最先进的应用，仍然依赖于基本的 RAG 原则。没有 RAG，我们讨论的任何内容都不可能实现！

本章介绍了 LangMem，这是 LangChain 用于程序记忆优化的 SDK。有了 LangMem，代理可以回顾自己的轨迹，提取稳定的行为模式，并通过系统的提示和政策更新来进化其操作剧本。结果是半自主或甚至完全自主的代理，它(1)自我学习，(2)通过检测和纠正故障模式来自我修复，(3)随着时间的推移不断改进，因为每次交互都为学习循环提供数据，(4)揭示客户洞察，包括重复的意图、摩擦点和解决方案模式，否则这些需要团队付出大量手动努力才能发现。

这就是程序记忆解锁的内容：

+   **自主性**：端到端委派工作流程，而不仅仅是单个步骤

+   **自我修复**：检测回归，应用修复，并促进更好的程序

+   **复合性能**：每次交互都成为训练数据

+   **客户智能**：通过聊天挖掘模式，揭示需求、差距和机会，无需手动标记马拉松

本章将涵盖以下内容：

+   使用程序记忆的好处

+   代码实验室 18.1 – 使用 LangMem 进行程序记忆管理

通过程序记忆从静态代理到自适应代理的转变，不仅是一个技术成就，而且在代理部署的各个方面都带来了具体的益处。从个性化多级交互到大幅降低运营成本，程序记忆从根本上改变了 RAG 增强系统所能实现的可能性。让我们看看这些能力如何转化为您应用的实际优势。

# 技术要求

要完成本章的动手练习，您需要以下软件和资源：

**软件需求**:

+   **Python 3.8 或更高版本**：运行代码实验室所必需

+   **pip 包管理器**：用于安装 Python 依赖项

+   **OpenAI API 访问**：您需要一个 OpenAI API 密钥来使用 GPT-4.1-mini 和嵌入

+   **Pydantic**：用于数据验证和结构化过程模型

+   **文本编辑器或 IDE**：VS Code、PyCharm、Jupyter Notebook 或类似工具，用于编写和运行 Python 代码

+   **Git** (可选): 用于克隆 GitHub 仓库

**硬件要求**：

+   **至少 8 GB RAM**（推荐 16 GB 以获得与向量数据库的流畅性能）

+   **2 GB 空闲磁盘空间**用于依赖项、向量存储和持久性内存存储

+   **互联网连接**用于下载软件包和访问 OpenAI API

**所需的 API 密钥**：

+   **OpenAI API 密钥**: 用于 GPT 模型访问和嵌入生成。请将其安全地存储在 `env.txt` 文件中，格式为 `OPENAI_API_KEY=your_key_here`。

注意：切勿将 API 密钥提交到版本控制。

**章节资源**：

+   **GitHub 仓库**: [`github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_18`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_18)

+   **完整代码文件**：代码实验室（18.1）可在 GitHub 仓库中找到，供参考，还包括包含投资顾问实现和测试场景的 `domain_investment` 目录。

如果你在任何步骤需要验证你的实现，GitHub 仓库中的完成代码可以作为参考。

# 使用程序性记忆的好处

我们已经走了很长的路，从基本的 RAG 发展到其最先进的应用：能够将检索作为其数据平面，将程序性记忆作为不仅用于改进，而且用于全面识别用户模式的引擎的自主代理。换句话说，无论你作为组织试图实现什么，无论你的目标是什么，你都可以使用这个代理以惊人的有效方式加速这一努力。

这种方法的适用范围非常广泛。一旦你的代理达到这种自主水平，它的扩展速度将远快于纯人工介入的系统。这成为相对于竞争对手的专有优势（或者如果他们先到达那里，则成为劣势），因为你的代理将开发出策略，以比整个营销、产品和工程团队更快、更有效地为用户提供服务。最重要的是，这种能力并非魔法；它是构建和使用最先进的 RAG 的自然结果，其中检索能力增强理解，程序性记忆将这种理解转化为持续的行为优化。

程序性记忆通过分层命名空间组织，使代理能够精确地调整其行为，无论是针对单个用户、团队、任务/意图，还是整个组织。系统捕捉丰富的上下文信号，包括成功指标、满意度评分和行为模式，然后将这些信号划分为有意义的组，以发现对每个组有效的方法。这种深入的理解不仅揭示了用户的需求，还揭示了真正驱动他们满意和成功的东西。这种分段方法自然地适应了常见模式和异常情况。通过为边缘情况保持单独的学习轨迹，同时保留常见流程，代理可以在不降低典型用户体验的情况下处理异常情况。结果是，代理真正了解其用户：他们的偏好、他们的痛点，最重要的是，对每个细分市场最有效的具体方法。

通过持续分析对话轨迹，程序性记忆学习产生更准确、相关的响应，同时减少幻觉和死胡同。系统的检索重构能力根据实际返回的有用结果优化查询模式和相似度阈值，而在标准方法可能失败时，感知失败的调整会预先改变策略。澄清、排序和升级的模板保持对话的结构和高效性。LangMem 提供了多种算法来提取和优化这些模式，不同的方法适用于不同的复杂程度。对于本章的实现，我们将使用`prompt_memory`算法，它提供高效的学习，同时计算开销最小。全面的安全功能，包括 A/B 测试、逐步推出和即时回滚，确保在全面部署之前验证改进，防止回归影响用户。

程序性记忆自动化了整个代理舰队中提示维护和优化的传统手动过程。模式到规则的转换消除了开发者手动制作提示更新的需求，而无需存储的 API 和 LangGraph 原生集成确保了大规模部署的无缝性。系统通过常见模式的规则模板加速了入职，使新代理能够立即从组织学习中获得好处。由于核心反馈循环自动根据对话结果触发优化，因此无需昂贵的重新训练周期即可实现持续改进。这种自动化不仅超越了简单的日志记录，还包括复杂的轨迹聚类、记忆类型之间的交叉参考分析以及时间模式检测，揭示了效果随时间的变化。

通过学习最优检索策略和对话流程，程序性记忆显著降低了计算成本和解决问题的耗时。学习算法可以有效地优化行为，而学习到的诊断模式则消除了无用的再生和重复询问。调整检索阈值可以防止不必要的文档处理，而基于规则的流程则简化了常见交互。系统的显著性测试和时间稳定性检查确保优化努力集中在能够带来可测量影响的变化上。也许最重要的是，通过自动更新、版本控制和供应商无关的存储来降低维护开销，从而释放工程资源用于更高价值的工作，同时确保每一次交互都能使系统更加高效。

这些好处并非理论上的，而是在你将程序性记忆应用于自己的系统时自然出现的。在*代码实验室 18.1*中，我们将构建一个完整的投资顾问代理，以展示这些功能在实际中的应用。从我们在*第十七章*中创建的记忆化代理开始，你将添加分层程序性学习，以捕捉对话中的成功模式并在适当的作用域中应用它们。通过实际实施，你将亲眼看到代理如何从通用的响应发展到复杂的、个性化的建议，这些建议会随着每次交互而不断改进。

# 代码实验室 18.1 – 实现程序性记忆优化

在这个实验室中，你将构建一个程序性记忆系统，使任何代理能够在多个层次上学习并适应特定领域的特定行为，但同时也具有将其应用于任何领域的灵活性。这个实验室从*第十七章*中的记忆化代理开始，但我们已经将其整合为单独的`baseline_agent.py`以方便使用。本章功能的核心在于`procedural_memory.py`，这是一个预构建的模块，实现了分层学习、检索和适应。而不是逐行构建这个系统，这个实验室采用了一种指导性演示方法：你将运行代码，观察程序性记忆在每个作用域级别上的运行情况，并理解使其工作的架构决策。这种方法让你能够专注于掌握概念并看到系统在实际运行中的表现，而不是编写基础设施代码。我们将演示如何创建一个模块化系统，其中核心学习机制与特定领域逻辑完全分离，从而让你能够轻松地为任何领域创建代理，例如投资咨询、医疗保健、教育或客户服务，只需实现领域接口即可。在这个代码实验室中，我们专注于投资顾问，但在最后，我们将讨论如何将其转换为你的领域，无论它是什么。

这个实现展示了两个关键架构洞察：

+   **首先，关注点的分离**：程序性记忆系统本身是领域无关的，可以与实现`DomainAgent`接口的任何领域一起工作。所有领域特定的逻辑提示、社区定义和成功指标都生活在隔离的领域目录中（例如`domain_investment/`或`domain_educator/`），这使得在不修改核心系统的情况下添加新领域变得非常简单。

+   **其次，分层学习**：并非所有模式都应该在全局范围内学习。系统自动确定每个学习模式的适当范围：

    +   **用户级**：捕捉个人偏好和个性化方法

    +   **社区级（群体）**：学习适用于具有共同特征的用户分段的模式

    +   **任务级（意图、动作组、类似分组概念）**：为特定类型的请求开发专门的方案

    +   **全球级**：提取适用于所有用户的通用最佳实践

程序性记忆系统允许智能体从自己的成功和失败中学习。每次对话后，智能体提取出有效（或无效）的部分，并将这些见解作为具体策略存储起来。这些学到的策略被存储为智能体可以检索并遵循的可读指令，以便在类似未来的情况下使用。您可以检查、编辑或删除任何学到的策略，从而完全控制智能体的演变。每个策略都维护自己的成功指标，可以独立评估、更新或回滚。

这种模块化方法使得快速部署特定领域成为可能。以下是一些潜在的例子：

+   **医疗保健**：在医疗环境中，程序性记忆系统通过患者互动学习以发现最佳护理模式。它可能发现老年患者早上预约效果更好，某些解释风格可以提高糖尿病患者对药物依从性，或者特定的后续序列可以降低再入院率。系统自动根据相关特征（年龄组、条件、治疗反应）对患者进行分段，同时保持安全合规标准，学习每个分段的最佳做法。

+   **客户服务**：对于支持操作，系统分析票务解决方案以揭示提高客户满意度和减少升级的模式。它学习到在某些情况下，提前提供退款可以防止升级，企业客户对技术深入挖掘反应更好，而个人用户更喜欢逐步指导，或者特定的故障排除序列可以更快地解决问题。程序性记忆系统自动对票务和客户分段进行分类，以应用最有效的解决方案策略。

+   **教育**：在教育环境中，系统观察辅导课程以确定哪些教学方法对不同学习者有效。它发现视觉学习者受益于基于图表的解释，某些节奏策略可以提高努力学生的理解力，或者特定的练习序列可以增强记忆。系统自然地将学生按学习风格和学术需求分组，并根据理解指标和学习成果不断优化其方法。

对于本实验室的实现，我们将使用投资顾问作为我们的示例领域，展示如何实现`DomainAgent`接口。整个投资领域位于`domain_investment/`中，展示了领域逻辑如何被干净地分离。代理将学习全局、个体、任务或群体层面上的有效方法，同时使用能够为医疗或教育代理提供动力的精确的程序记忆系统。

在整个实验室中，您将看到以下情况发生：

+   来自*第十七章*的基线代理提供情景和语义记忆

+   我们添加了一个与任何领域都兼容的程序记忆系统

+   我们将投资顾问领域实现为一个干净、独立的模块

+   代理在适当的层次级别上从交互中学习

+   同样的架构可以通过实现接口用于任何其他领域

结果是一个模块化、可扩展的系统，领域专家可以在不接触核心学习基础设施的情况下创建专门的代理，展示了适当的架构分离如何同时实现强大的学习能力和实用的可维护性。

## 第 1 步 - 使用导入和基线代理设置基础

我们首先导入所有必要的库和模块，这些库和模块将支持我们的分层程序记忆实现。理解这些导入有助于阐明架构：

```py
import os
import sys
import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from baseline_agent import CoALABaselineAgent
from domain_agent import DomainProcedure
from domain_investment.investment_advisor_data import (
    EnhancedInvestmentAdvisorDataGenerator
)
from domain_investment.investment_advisor_agent import (
    InvestmentAdvisorAgent)
from procedural_memory import ProceduralMemory
from domain_investment.investor_test_scenarios import (
    setup_hierarchy_demo, get_test_cases, get_feedback_rounds,
    process_baseline_conversations, test_agent_with_queries,
    process_performance_feedback, process_remaining_conversations,
    test_hierarchical_retrieval, get_key_achievements
)
from dotenv import load_dotenv
# Load environment variables
load_dotenv(dotenv_path="env.txt")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Update Python path
sys.path.append("domain_investment")
# Initialize the baseline agent (episodic + semantic memory only)
agent = CoALABaselineAgent(
    model_name="gpt-4.1-mini",
    temperature=0,
    persist_directory="./baseline_memory_store"
) 
```

大部分代码集中在导入我们需要的库，以便在其余代码实验室中使用。以下是我们要导入的内容：

+   **核心 Python 库**：

    +   **os, sys, json**：处理环境变量、路径，并解析 LLM 学习提取的 JSON 响应

    +   **类型提示（Dict, List, Optional）**：提供类型提示以增强代码清晰度和 IDE 支持

+   **数据建模**：

    +   **pydantic (BaseModel, Field)**：为我们的学习程序定义结构化数据模型，确保类型安全和验证

+   **我们的模块化代理系统**：

    +   **CoALABaselineAgent**: 来自*第十七章*代码实验室的基础代理，现在已合并为一个单独的文件（`baseline_agent.py`），该文件结合了情景记忆和语义记忆的实现

    +   **domain_agent 中的 DomainProcedure**：任何领域都可以使用的通用程序结构

    +   **DomainAgent 抽象基类**：定义了所有特定领域代理必须实现的接口

    +   **ProceduralMemory**：与任何领域代理一起工作的领域无关的程序记忆系统

+   **特定领域的实现**：

    +   **InvestmentAdvisorAgent**：我们的示例领域实现，展示了如何创建特定领域的代理

    +   **EnhancedInvestmentAdvisorDataGenerator**：生成用于测试的逼真的投资顾问对话

    +   **投资测试场景和工具**：用于展示分层学习的辅助函数

这里的关键架构洞察是关注点的分离。核心程序记忆系统（`ProceduralMemory`）完全领域无关。它与实现`DomainAgent`接口的任何领域都兼容。所有投资特定逻辑都位于`domain_investment`目录中，这使得通过实现相同的接口创建新的领域（如`domain_educator`或`domain_healthcare`）变得非常简单。`CoALABaselineAgent`代表了第十七章工作的成果；它包括情景记忆（存储对话历史）和语义记忆（提取和存储事实），从*代码实验室 17.1、17.2 和 17.3*的独立实现中整合到一个单一、干净的模块中。然而，这个基线代理不能从成功的交互中学习或随着时间的推移改进其模式。这种限制在各个领域都是普遍存在的：

+   医疗助手可能会记住患者的医疗历史，但不会了解到早上预约对老年患者更有效

+   客户服务机器人可能会回忆起之前的工单，但不会发现提前提供退款可以降低升级率

+   教育辅导师可能会跟踪覆盖了哪些主题，但不会了解到某些学生更倾向于视觉解释

在我们的实现示例中，我们使用了一个投资顾问来展示这些学习能力。基线代理使用其基础知识响应投资查询，但不会根据对不同客户类型或市场条件最有效的方法进行调整。我们添加的关键创新是通过 LangMem 的方法实现程序记忆。LangMem 自动化了传统上需要手动干预的任务：观察代理行为，识别成功和失败中的模式，并更新提示以包含学习到的程序。我们提供了一些测试代码，以了解我们目前的位置，其中我们要求代理：“我想重新平衡我的投资组合。我 35 岁，风险承受能力中等。”

```py
response = agent.process_message(
    "I'm looking to rebalance my portfolio. I'm 35 and have moderate risk tolerance.",
    user_id="demo_investor"
)
print("Baseline Response (without procedural memory):")
print(response)
print(f"\nMemory Stats: {agent.get_memory_stats()}") 
```

以下是预期的输出：

```py
Baseline Response (without procedural memory):
Given that you're 35 years old with a moderate risk tolerance, a balanced portfolio might typically include a mix of equities and fixed income to provide growth potential while managing risk. A common approach could be...[see code lab for full output]
Memory Stats: {'total_documents': 6, 'current_user': 'demo_investor', 'current_conversation': 'conv_1758137178.591926'} 
```

基线响应显示了局限性：虽然代理根据用户的年龄和风险承受能力提供合理的建议，但它缺乏学习以下模式的能力：

+   这个特定的用户可能会根据过去的互动更喜欢 ETF 而不是共同基金

+   类似的 35 岁中等风险客户在 60/40 的分配上取得了更好的成功

+   在提前解决税收影响的情况下，重新平衡讨论会更成功

内存统计确认，代理已经存储了一些文档（来自此测试消息）在其情景/语义记忆中，但还没有可用的程序模式。这正是分层程序记忆将改变代理能力的地方，使其不仅能够学习什么有效，而且能够学习对谁有效。

现在我们已经运行了基于第十七章的情景记忆和语义记忆的基线*第十七章*，让我们定义一个领域无关的程序结构，这将使任何领域的多范围学习成为可能。

## 第 2 步 – 定义分层学习的程序结构

我们将创建一个领域无关的数据结构，它捕获了在多个范围上学习到的程序，体现了 LangMem 的全面数据收集能力。这个结构将存储成功指标、行为模式和适应历史，这些都有助于进行超越表面指标的高级模式分析，同时保持足够的灵活性，以与任何领域一起工作。

```py
class DomainProcedure(BaseModel):
    """Generic procedure structure for any domain.

    Core fields (required for all domains):
    - strategy_pattern: Description of the strategy
    - steps: Ordered list of steps to execute
    - segments: Applicable user segments
    - success_rate: Historical success rate (0.0-1.0)
    - scope: Hierarchy level (global/user/community/task)

    Domain-specific data goes in:
    - domain_metrics: Dict for any domain-specific metrics
      (e.g., avg_portfolio_performance for investments,
             quiz_avg for tutoring, etc.)
    """
    # Core fields (domain-agnostic)
    strategy_pattern: str
    steps: List[str]
    segments: List[str] = Field(default_factory=list)
    success_rate: float = 1.0
    usage_count: int = 0
    adaptations: List[Dict] = Field(default_factory=list)

    # Segmentation metadata
    scope: str = "global"  
    scope_id: Optional[str] = None
    priority: int = 0
    learned_from_count: int = 1

    # Domain-specific metrics stored as flexible dict
    domain_metrics: Dict[str, float] = Field(default_factory=dict)
domain_agent = InvestmentAdvisorAgent() 
```

这种结构展示了领域无关设计的力量，同时保持了 LangMem 的数据收集能力。让我们分解一下代码：

+   **核心模式跟踪（跨领域通用）**：

    +   `strategy_pattern`: 识别此程序解决的情况类型——完全领域中性（例如，“适度风险投资组合再平衡”用于投资，“密码重置故障排除”用于客户服务，“视觉学习方法”用于教育）

    +   `steps`: 证明成功的具体动作序列，以任何领域都可以填充的简单字符串表示

    +   `segments`: 通用标签，指示哪些用户群体将从此程序中受益——与任何特定领域术语无关

+   **性能指标（领域无关）**：

    +   `success_rate`: 跟踪使用此程序与积极结果之间的相关性——通用指标

    +   `usage_count`: 使能够检测随时间变化的有效性变化

    +   `domain_metrics`: 一个灵活的字典，其中每个领域存储其特定的指标——投资可能跟踪`avg_portfolio_performance`，教育可能跟踪`comprehension_score`，医疗保健可能跟踪`treatment_adherence`

+   **分层学习范围（适用于任何领域）**：

    +   `scope`: 指示此程序应应用的分层级别：“user”用于个人个性化，“community”用于用户群体模式，“task”用于请求类型的专业化，“global”用于通用的最佳实践——一个在所有领域都统一适用的概念

    +   `scope_id`: 当范围不是全局时，标识特定的用户或社区

    +   `priority`: 确定在多个匹配时使用哪个程序（更具体的范围具有更高的优先级）

+   **演变跟踪（对所有领域至关重要）**：

    +   `adaptations`：维护完整的版本历史，如果新的策略表现不佳，则允许回滚

    +   `learned_from_count`：跟踪有多少对话示例贡献了此过程

这种结构的优点在于其通过`domain_metrics`字典的灵活性。每个领域都可以存储对其重要的任何指标，而无需更改核心结构。投资顾问跟踪投资组合表现，医疗保健代理跟踪患者结果，教育者跟踪学习进度——所有这些都可以使用相同的`DomainProcedure`类。在*第十九章*的*设计领域指标：衡量成功的艺术*部分，我们将讨论如何以一个目标、多个目标、一些介于两者之间为目标来处理指标，以及这些不同策略的启示。

这种结构体现了 LangMem 的一个关键洞察：无论领域如何，并非所有模式都应该全局学习。在我们的投资顾问示例中，退休人员需要的策略与年轻专业人士不同。在医疗保健中，早晨预约偏好可能是用户特定的或所有老年患者的共同点。在教育中，视觉学习偏好可能适用于个别学生或整个学习风格群体。该结构统一处理所有这些情况。

`adaptations`字段对于所有领域的生产安全尤其重要。正如 LangMem 所指出，这个版本历史记录使得在新的规则降低性能时可以进行回滚。每个适应项记录了发生了什么变化，何时发生变化，以及哪些性能指标证明了这种变化。这创建了一个审计轨迹，使得程序学习透明且可逆，无论你是在跟踪投资回报、患者健康状况还是学生考试成绩。代码实验室之后，我们将更多地讨论如何回滚。

为了看到这个结构如何运作，让我们创建一个示例过程，演示所有这些字段如何协同工作。以下代码创建了一个从与中风险千禧一代投资者成功互动中学习到的再平衡策略的具体示例，包括具体的步骤、成功指标和特定领域的性能跟踪：

```py
# Create a sample procedure for investment domain
sample_procedure = DomainProcedure(
    strategy_pattern="moderate risk portfolio rebalancing",
    steps=[
        "1\. Review current allocation and drift from target",
        "2\. Assess client's life changes affecting risk tolerance",
        "3\. Analyze market conditions and sector rotation opportunities",
        "4\. Propose rebalancing with tax-loss harvesting considerations",
        "5\. Set up automatic rebalancing schedule"
    ],
    segments=["millennials", "moderate_risk"],  # Generic 'segments' not 'client_segments'
    domain_metrics={
        "avg_portfolio_performance": 8.5,  # Investment-specific metric
        "avg_rebalance_frequency_days": 90  # Another domain metric
    }
)
print("![](img/Icon48.png) Domain procedure structure created")
print(f"  Strategy: {sample_procedure.strategy_pattern}")
print(f"  Steps: {len(sample_procedure.steps)}")
print(f"  Segments: {', '.join(sample_procedure.segments)}")
print(f"  Domain metrics: {sample_procedure.domain_metrics}") 
```

这是预期的输出：

```py
![](img/Icon47.png) Domain procedure structure created
  Strategy: moderate risk portfolio rebalancing
  Steps: 5
  Segments: millennials, moderate_risk
  Domain metrics: {'avg_portfolio_performance': 8.5, 
                   'avg_rebalance_frequency_days': 90.0} 
```

这个示例过程演示了通用结构如何捕捉我们投资领域的完整学习策略。五步平衡程序代表了从与中风险千禧一代成功对话中提取的模式。特定领域的指标（存储在`domain_metrics`中）显示了 8.5%的平均投资组合表现和 90 天的再平衡频率——这些指标是特定于投资咨询的，但存储在领域无关的结构中。

注意到 `segments` 字段使用通用术语，如 `millennials` 和 `moderate_risk`，而不是特定领域的术语，如 `client_segments`。这允许相同的结构适用于不同的场景，无论是投资者档案、患者人口统计学还是学生学习风格。`scope` 字段为我们准备多级学习，我们将在下一阶段实现，它将在任何领域都以相同的方式工作。

在定义了不受特定领域限制的流程结构以支持 LangMem 的数据收集和版本控制能力后，我们需要初始化将使用这些结构的程序记忆系统。接下来，我们将创建与任何领域代理一起工作的程序记忆系统，展示关注点的分离如何实现灵活性和强大功能。

## 第 3 步 - 初始化分层程序记忆

我们将创建核心学习引擎，自动从对话中提取成功模式。这个系统展示了 LangMem 如何创建一个反馈循环，其中每次交互都可能提高未来的性能，同时保持领域无关学习机制和特定领域逻辑的完全分离。

```py
domain_agent = InvestmentAdvisorAgent()
investment_memory = ProceduralMemory(
    llm=agent.llm, domain_agent=domain_agent
) 
```

这个初始化展示了我们架构中关注点的清晰分离：

+   **领域代理封装**: `InvestmentAdvisorAgent` 在 `domain_investment` 目录中封装了所有与投资相关的逻辑——提示、社区定义、成功指标。这个领域代理实现了 `DomainAgent` 接口，提供以下功能：

    +   不同范围的学习提示（全局、用户、社区、任务）

    +   社区定义（保守的退休人员、激进的千禧一代、温和的专业人士）

    +   任务识别逻辑（再平衡、税务规划、风险评估）

    +   专门针对投资结果的成功评分计算

    +   特定领域的指标更新（投资组合表现、再平衡频率）

+   **程序记忆系统**: `ProceduralMemory` 类完全不受特定领域限制。它接收一个领域代理，并使用它执行以下操作：

    +   从领域的模板创建学习提示

    +   使用领域的分类识别任务类型

    +   使用领域的指标计算成功评分

    +   在性能反馈后更新特定领域的指标

这种分离意味着相同的 `ProceduralMemory` 类可以与任何领域一起工作。要创建一个医疗保健代理，你只需传递一个 `HealthcareAgent` 实例——对于教育，`EducatorAgent`。程序记忆系统不会改变；只有领域代理实现不同。

+   **分层存储结构**: 系统为每个学习范围维护单独的存储，检索时优先考虑最具体的相关知识（用户 → 社区 → 任务 → 全球）：

    +   **user_procedures**: 个体用户偏好和个性化策略（最高优先级）

    +   **community_procedures**: 用户段落的模式（由领域定义）

    +   **task_procedures**：针对不同查询类型的专用方法

    +   **global_procedures**：适用于所有用户的通用模式（最低优先级，但始终作为后备可用）

+   **学习历史跟踪**：为每个范围提供单独的历史跟踪，使 LangMem 称为“轨迹分析”：

    +   `user_learning_history`：记录特定于用户的自适应

    +   `community_learning_history`：监控段级学习，并动态跟踪从数据中出现的已发现段

    +   `task_learning_history`：捕捉特定于任务的策略及其在不同查询类型中的有效性演变

    +   `global_learning_history`：跟踪所有通用模式的发现

初始化确认我们的程序性记忆系统已准备好四个分层范围，但还没有学习到的策略。这种空状态代表我们的起点，即一个具有学习能力但没有积累知识的代理。

让我们通过打印系统状态并检查其空但准备好学习的状态来验证这个初始化。以下代码确认领域代理已正确连接，并显示所有四个分层范围内的初始统计数据：

```py
print("✓ Procedural memory system initialized")
print(f"  Domain: {domain_agent.__class__.__name__}")
# Verify the system is initialized correctly
stats = investment_memory.get_stats()
print(f"\n![](img/Icon24.png) Initial state:")
print(f"  Total strategies: {stats['total_strategies']}")
print(f"  By scope: {stats['by_scope']}")
print(f"  Learning history tracking: {len(investment_memory.global_learning_history)} events") 
```

这是预期的输出：

```py
![](img/Icon25.png) Procedural memory system initialized
  Domain: InvestmentAdvisorAgent
  Storage scopes: 4 (global, user, community, task)
![](img/Icon26.png) Initial state:
  Total strategies: 0
  By scope: {'global': 0, 'user': 0, 'community': 0, 'task': 0}
  Learning history tracking: 0 events 
```

输出确认领域无关的程序性记忆系统已成功初始化。系统认识到它正在与`InvestmentAdvisorAgent`一起工作，但会与任何其他领域代理以相同的方式工作。四个存储范围已准备好在不同具体级别上捕获模式，学习历史跟踪已准备好维护所有学习策略的审计轨迹。

这里的关键洞察是，所有特定于投资的逻辑都存在于领域代理中，而程序性记忆系统保持纯净且可重用。领域代理提供以下内容：

+   **学习提示**：提取领域语言模式的模板

+   **社区定义**：如何在该领域细分用户

+   **任务类型**：特定于该领域的查询类别

+   **成功指标**：如何衡量该领域的成功

+   **领域指标**：哪些额外的测量很重要

程序性记忆系统通过一个干净的界面使用这些特定领域的元素，从不直接依赖于投资概念。这种架构体现了 LangMem 的“分层命名空间组织”原则，同时保持模块化。

现在我们已经用领域代理初始化了学习引擎，我们需要展示它是如何从实际交互中学习的。接下来，我们将展示系统如何从对话中提取模式，并将它们存储在适当的分层级别，将原始交互转化为可执行策略。

## 第 4 步 – 展示从交互中学习

在下面的代码块中，我们将演示程序记忆系统如何从实际对话中学习，提取多个层次级的模式。这展示了 LangMem 方法如何将原始交互转化为代理可以在未来对话中应用的行动策略。

```py
successful_interactions = [
    {
        "query": "I want to rebalance my portfolio",
        "user_id": "user_001",
        "profile": {"age": 35, "risk_tolerance": "moderate"},
        "interaction": {
            "messages": ["User: I want to rebalance", 
                        "Assistant: Here's your rebalancing plan..."],
            "success": True,
            "client_satisfaction": 9,
            "returns": 8.5
        }
    },
    {
        "query": "Show me ESG investment options",
        "user_id": "user_002",
        "profile": {"age": 30, "risk_tolerance": "aggressive"},
        "interaction": {
            "messages": ["User: ESG options?", 
                        "Assistant: Here are sustainable funds..."],
            "success": True,
            "client_satisfaction": 8,
            "returns": 7.2
        }
    }
] 
```

这个演示展示了程序记忆系统在行动中的表现，仅从两个成功的交互中学习。系统展示了几个关键的 LangMem 能力：

+   **从单个交互中进行多范围学习**：每次对话可能在不同层次上产生学习。从一个再平衡查询中，系统可能会执行以下操作：

    +   提取关于如何处理再平衡请求的全球模式

    +   学习`user_001`的用户特定偏好

    +   识别一个针对中风险投资者的社区模式

    +   为再平衡查询开发特定于任务的解决方案

这种多范围提取体现了 LangMem 的原则，即不同层次上的不同模式具有不同的价值。

+   **领域无关的学习过程**：注意学习过程本身是完全领域无关的。`learn_from_interaction`方法执行以下操作：

    +   接收查询和交互数据（通用概念）

    +   委派给领域代理以识别任务类型和社区

    +   使用领域提供的提示来提取模式

    +   在分层结构中存储程序

同样的方法对于从患者互动中学习的医疗代理或从辅导课程中学习的教育代理来说，效果完全相同。

+   **自动范围确定**：系统自动确定每个交互适当的范围：

    +   如果提供`user_id`，它尝试用户级别的学习

    +   如果用户属于一个社区，它学习社区模式

    +   如果查询匹配任务类型，它开发特定于任务的策略

    +   总是考虑全局模式以获得通用见解

这种自动范围确保模式在最适合的级别上学习，无需人工干预。

+   **错误容错性**：学习过程旨在生产健壮性。如果在一个范围内学习失败（可能由于 LLM 解析错误），它将在其他范围内继续学习。这种容错性确保了有价值模式不会因为孤立故障而丢失。

在这里，我们运行一些测试来查看我们的输出：

```py
print("![](img/Icon46.png) Learning from successful interactions...")
learned_strategies = {}
for data in successful_interactions:
    result = investment_memory.learn_from_interaction(
        query=data["query"],
        interaction_data=data["interaction"],
        user_id=data["user_id"],
        user_profile=data["profile"]
    )

    if result.get("learned"):
        for key, value in result.items():
            if "learned" in key:
                learned_strategies[key] = value
                print(f"  ![](img/Icon48.png) Learned {key}: {value}")
# Check what was learned
stats = investment_memory.get_stats()
print(f"\n![](img/Icon44.png) After learning:")
print(f"  Total strategies: {stats['total_strategies']}")
print(f"  By scope: {stats['by_scope']}") 
```

这是预期的输出：

```py
![](img/Icon27.png) Learning from successful interactions...
![](img/Icon28.png) After learning:
  Total strategies: 8
  By scope: {'global': 2, 'user': 2, 'community': 2, 'task': 2} 
```

输出显示仅从两个交互中成功进行的多范围学习。系统已经学习了分布在所有四个范围中的八个策略：

+   **两种全局策略**：适用于任何用户的通用模式

+   **两种用户策略**：针对特定用户的个性化方法

+   **两种社区策略**：针对识别出的用户段落的模式

+   **两种任务策略**：针对查询类型的专用程序

这种平衡的分布展示了系统如何从相同的交互中提取不同类型的价值。全局策略可能捕捉到一般最佳实践，例如“始终提供具体的再平衡步骤”，而用户策略会记住“`user_001`更喜欢详细的解释”。社区策略可能会指出“中风险投资者希望采取平衡的方法”，而任务策略可能指定“ESG 查询应包括可持续性指标”。

这里是从这次学习中得出的关键见解：

+   **效率**：仅通过两次对话，系统就提取了八种不同的策略，从而最大化了从最少数据中学习的效果

+   **具体性**：每个策略都存储在适当的领域，防止过度泛化

+   **完整性**：该系统捕捉到单领域学习可能错过的模式

+   **模块化**：整个学习过程使用领域代理接口，使得交换领域变得非常简单

学习到的策略现在可以检索并应用于新的对话。当出现类似的查询时，系统将通过这些分层领域搜索以找到最具体的适用策略。

在多个领域学习并存储策略后，我们需要机制来检索每个情况下的正确策略，并根据实际表现进行更新。接下来，我们将实施分层检索和性能反馈系统，通过实际使用实现持续改进。

## 第 5 步 – 添加策略检索和性能反馈

检索，RAG 中的“R”，通过这种方法找到了全新的功能层次。检索本身变得动态和分层，自动选择最个性化的可用策略，而不仅仅是找到相关文档。现在我们将实施检索和反馈机制，使程序性记忆真正动态。这些组件使系统能够为每种情况找到最合适的策略，并根据实际表现持续改进——这是 LangMem 持续学习方法的精髓。

### 第 5a 步 – 展示分层检索

首先，我们将演示系统如何使用分层优先级检索策略，始终优先选择最具体的适用知识：

```py
# Cell 5a: Demonstrate Hierarchical Retrieval (user → community → task → global)
print("Setting up strategies at each scope level...")
setup_hierarchy_demo(investment_memory)
print("\nDemonstrating retrieval hierarchy:\n" + "=" * 60)
for user_id, query, profile, expected in get_test_cases():
    strategy = investment_memory.get_investment_strategy(
        query, profile, user_id)
    print(f"\n![](img/Icon42.png) User: {user_id}\n   Query: '{query}'\n   Expected: {expected}")

    if strategy:
        print(f"   ![](img/Icon43.png) Retrieved: {strategy['strategy']}")
        print(f"   → Scope: {strategy['scope'].upper()}")
        print(f"   → Source: {strategy['source']}")
        print(f"   → Confidence: {strategy['confidence']:.0%}")
    else:
        print(f"   ✗ No strategy found")
print("\n![](img/Icon29.png) Hierarchy Summary:")
stats = investment_memory.get_stats()
for scope, count in stats['by_scope'].items():
    print(f"   {scope}: {count} strategies") 
```

这个演示展示了 LangMem 的分层检索在实际中的应用。系统按照优先级顺序检查领域（**用户** **→** **社区** **→** **任务** **→** **全局**），确保当可用时使用最个性化的策略。让我们来探索输出：

```py
Setting up strategies at each scope level...
# Demonstrating retrieval hierarchy:
![](img/Icon30.png) User: user_001
 Query: 'I want to rebalance'
 Expected: Should retrieve USER scope (user_001 has personalized strategy)
 ![](img/Icon47.png) Retrieved: user_user_001_preference_0
 → Scope: USER
 → Source: personalized for user_001
 → Confidence: 85%
![](img/Icon33.png) User: user_003
 Query: 'Need investment advice'
 Expected: Should retrieve COMMUNITY scope (user_003 in moderate_professionals)
 ![](img/Icon32.png) Retrieved: moderate_prof_strategy
→ Scope: COMMUNITY
 → Source: learned from moderate_professionals community
 → Confidence: 85%...[see code lab for full output] 
```

输出确认了在所有四个领域范围内都进行了适当的分层检索。每个查询都正确匹配了其预期级别，展示了当可用时系统如何提供越来越具体的指导。层次总结显示我们在所有领域都有分布的策略，随时准备满足不同的具体需求。

这种分层检索从根本上改变了 RAG 的操作方式。在传统的 RAG 中，检索会搜索包含“再平衡”等关键词的文档，并返回相同的投资建议文档，无论提问者是谁。但看看我们的输出发生了什么（**检索的转化**）：

+   **身份感知检索**：当`user_001`询问再平衡问题时，系统不仅仅搜索“再平衡文档”。它首先检查`user_001`是否从过去的成功互动中学习了任何个性化的策略。以 85%的置信度找到一种策略后，它将检索该用户特定的方法，而不是通用的建议。

+   **基于社区的回退**：对于缺乏个人策略的`user_003`，传统的 RAG 会返回相同的通用文档。但我们的系统认识到`user_003`属于`moderate_professionals`社区，并检索在该部分中为类似用户有效的工作策略。这是基于学习到的群体模式进行的检索，而不是基于文档相似性。

+   **动态策略选择**：注意相同的概念（“再平衡”）对不同用户触发了不同的检索。在这里，`user_001`获得他们的个性化方法，`user_004`获得特定任务的再平衡策略，而新用户获得全球最佳实践。传统的 RAG 会为所有这些用户返回相同的文档。

+   **基于置信度的优先级**：每个检索到的策略都附带一个置信度分数（这些例子中的 85%），基于其历史表现。这不是基于向量搜索的相似度评分，而是基于实际使用中的成功率跟踪。系统知道哪些策略实际上有效，而不仅仅是哪些文档包含匹配的关键词。

+   **范围感知检索逻辑**：检索过程本身已经变得智能，遵循**用户** **→** **社区** **→** **任务** **→** **全局**的层次结构。这确保了始终优先选择最具体、最相关的策略，当特定策略不可用时，会优雅地降级到更通用的方法。

从本质上讲，我们已经从“检索关于 X 的文档”发展到“检索针对询问 X 的特定用户最成功的策略，通过越来越通用的范围回退，直到找到适用的知识。”检索过程已成为一个决策树，考虑用户身份、社区成员资格、任务类型和历史表现，与简单的语义相似度搜索大相径庭。

这种优先级顺序确保用户获得尽可能相关的建议。在医疗保健领域，这可能意味着糖尿病患者获得他们个性化的胰岛素方案，而不是通用的糖尿病指南。在教育领域，视觉学习者会收到他们定制的教学方法，而不是标准的教学方法。

现在我们已经展示了系统如何使用分层优先级检索策略，我们需要展示这些策略如何根据实际表现进行演变。程序记忆的真正力量不仅在于存储成功的模式，而且在于根据实际结果不断优化它们。每次应用策略时，系统都会跟踪其有效性并相应地调整信心。这创建了一个反馈循环，其中持续成功的策略变得更加可信，而失败的策略则会被降低权重或最终移除。让我们实现这个关键的适应机制，它将静态规则转化为动态、自我优化的知识。

### 第 5 步 b – 展示性能反馈循环

接下来，我们将展示策略如何根据实际表现进行适应，实现 LangMem 的持续改进机制：

```py
# Cell 5b: Demonstrate Performance Feedback Loop
print("Testing performance feedback and adaptation...")
print("=" * 60)
# Get strategy and show initial state
test_strategy = investment_memory.global_procedures["general_investment"]
print(f"\nInitial state of 'general_investment' strategy:")
print(f"  Success rate: {test_strategy.success_rate:.0%}")
print(f"  Domain metrics: {test_strategy.domain_metrics}")
print(f"  Adaptations: {len(test_strategy.adaptations)}")
print("\n![](img/Icon34.png) Applying feedback rounds:")
for i, feedback in enumerate(get_feedback_rounds(), 1):
    expected_score = domain_agent.calculate_success_score(feedback)
    old_rate = test_strategy.success_rate

    result = investment_memory.update_from_performance(
        strategy="general_investment",
        performance_data=feedback,
        scope="global"
    )

    print(f"\nRound {i}: Satisfaction={feedback['client_satisfaction']}, "
          f"Returns={feedback['returns']:+.1f}%")
    print(f"  Success score: {expected_score:.0%}")

    if result.get('updated'):
        print(f"  Success rate: {old_rate:.0%} → {result['new_success_rate']:.0%}")
        print(f"  Trend: {result['performance_trend']}")
        if 'avg_portfolio_performance' in test_strategy.domain_metrics:
            print(f"  Avg portfolio: {test_strategy.domain_metrics['avg_portfolio_performance']:.1f}%")
# Show adaptation history
print(f"\n![](img/Icon41.png) Adaptation History:")
print(f"  Total adaptations: {len(test_strategy.adaptations)}")
if test_strategy.adaptations:
    for i, adaptation in enumerate(test_strategy.adaptations[-2:], 1):
        print(f"\n  Adaptation {i}:")
        print(f"    Time: {adaptation['timestamp'][:19]}")
        print(f"    Old rate: {adaptation['old_rate']:.0%}")
        print(f"    New rate: {adaptation['new_rate']:.0%}")
        print(f"    Success score: {adaptation['success_score']:.0%}") 
```

`procedural_memory.py`中的`update_from_performance`方法处理这种反馈处理。让我们检查其核心逻辑：

```py
def update_from_performance(
    self, strategy: str, performance_data: Dict, scope: str = "global",
    scope_id: Optional[str] = None
) -> Dict:
    """Update procedure based on performance feedback"""
    # Select the appropriate procedure store based on scope
    if scope == "user" and scope_id and scope_id in self.user_procedures:
        procedures = self.user_procedures[scope_id]
    elif scope == "community" and scope_id and scope_id in self.community_procedures:
        procedures = self.community_procedures[scope_id]
    elif scope == "task" and scope_id and scope_id in self.task_procedures:
        procedures = self.task_procedures[scope_id]
    else:
        procedures = self.global_procedures
    for pattern, proc in procedures.items():
        if (
            pattern.lower() in strategy.lower()
            or strategy.lower() in pattern.lower()
        ):
            # Calculate success score using domain-specific metrics
            success_score = self.domain_agent.calculate_success_score(
                performance_data
            )
            old_rate = proc.success_rate
            # Momentum-based update: 80% old rate, 20% new score
            proc.success_rate = min(
                1.0,
                proc.success_rate * 0.8 + success_score * 0.2
            )
            # Let domain agent update its specific metrics
            self.domain_agent.update_domain_metrics(
                proc, performance_data)
            # Record adaptation for audit trail and potential rollback
            proc.adaptations.append({
                "timestamp": datetime.now().isoformat(),
                "performance": performance_data,
                "old_rate": old_rate,
                "new_rate": proc.success_rate,
                "success_score": success_score
            })
            return {
                "updated": pattern,
                "scope": scope,
                "scope_id": scope_id,
                "new_success_rate": round(proc.success_rate, 2),
                "performance_trend": (
                    "improving"
                    if proc.success_rate > old_rate
                    else "declining"
                ),
                "total_adaptations": len(proc.adaptations)
            }
    return {"updated": None} 
```

此方法展示了几个重要的设计决策。范围感知的过程查找确保通过首先确定基于提供的范围参数要搜索哪个过程存储，将更新应用于正确的分层级别。该方法不要求精确匹配，而是使用不区分大小写的子串匹配以提高灵活性，即使在策略名称略有变化的情况下也能找到相关的过程。实际的成功计算是通过`calculate_success_score()`委托给领域代理的，这保持了程序记忆系统的领域无关性，同时允许每个领域在其上下文中定义“成功”的含义。最后，`min(1.0, ...)`上限确保成功率永远不会超过 100%，即使在许多积极的反馈回合之后也保持有效的概率值。

这种反馈处理展示了 LangMem 的多源反馈三角测量。领域代理的`calculate_success_score`方法根据领域优先级权衡不同的成功因素。对于投资，回报最为重要（50%），其次是满意度（30%）和目标达成（20%）。这些权重从根本上塑造了代理的演变方式。如果我们将其更改为优先考虑用户满意度（50%）、回报（30%）和目标达成（20%），代理将发展出非常不同的策略。它不会学习最大化投资组合表现，即使客户不完全理解这种方法，它也会学习优先考虑清晰的解释和客户舒适度，可能为了更高的满意度而接受较低的回报。随着时间的推移，这个以满意度为重点的代理可能会发展出诸如“始终用简单术语解释税收影响”或“在继续之前检查理解”的策略，而一个以回报为重点的代理则会学习诸如“立即识别表现不佳的资产”或“优先考虑高收益机会”的策略。相同的程序记忆系统会根据这些成功指标产生完全不同的学习行为，展示了特定领域的价值观如何直接影响代理的演变。医疗保健领域可能会对患者的结果有不同的优先级，而教育可能会关注理解分数。

基于动量的更新（80%旧数据，20%新数据）可以防止单个异常值剧烈改变策略，同时仍然允许适应。请注意以下内容：

+   **良好的表现提高信心**：第 1-2 轮的高满意度和正回报提高了成功率

+   **表现不佳触发调整**：第 3 轮的糟糕结果立即降低了成功率

+   **恢复是渐进的**：第 4 轮的适度成功开始恢复，但不会立即恢复高信心

适应历史提供了完整的可审计性——每一次变更都会记录时间戳、原因和影响。这使得 LangMem 能够实现所谓的“安全回滚”——如果策略的表现持续下降，我们可以回滚到之前的版本。

以下是预期的输出：

```py
Testing performance feedback and adaptation...
Initial state of 'general_investment' strategy:
  Success rate: 75%
  Domain metrics: {}
  Adaptations: 0
![](img/Icon35.png) Applying feedback rounds:
Round 1: Satisfaction=9, Returns=+12.5%
  Success score: 100%
  Success rate: 75% → 80%
  Trend: improving
  Avg portfolio: 1.2%
Round 2: Satisfaction=8, Returns=+8.0%
  Success score: 100%
  Success rate: 80% → 84%
  Trend: improving
  Avg portfolio: 1.9%...[see code lab for full output] 
```

输出显示了策略的合理演变。从 75%的置信度开始，策略在成功应用后得到改善，失败后急剧下降，然后开始恢复。领域指标（平均投资组合表现）也会适应，提供特定领域的跟踪以及普遍的成功率。

在我们的反馈机制到位且策略根据性能积极调整的情况下，我们需要一种方法来检查和理解我们程序记忆的当前状态。系统已经通过多次交互进行学习和演变，但没有对其结构的可见性，很难验证我们的层次组织是否按预期工作。让我们可视化完整的记忆结构，看看策略是如何分布在不同范围中的，哪些社区已经形成，以及性能指标是如何跟踪的。这种透明度对于调试、监控和建立对学习系统的信任至关重要。

### 第 5 步 c – 可视化程序记忆结构

现在让我们可视化完整的层次结构，看看策略是如何组织的：

```py
# Cell 5c: Visualize the Procedural Memory Structure
print("Procedural Memory Structure Visualization")
# Show the actual hierarchy
investment_memory.show_strategy_performance()
# Show community membership
print("\n![](img/Icon39.png) Community Membership Map:")
for user, communities in investment_memory.user_communities.items():
    print(f"  {user}: {', '.join(communities)}")
for community, members in investment_memory.community_members.items():
    if members:
        print(f"  {community} has {len(members)} members: {', '.join(members)}")
# Show discovered segments
print(f"\n![](img/Icon40.png) Discovered Segments: {', '.join(investment_memory.segments_discovered)}") 
```

这种可视化提供了对学习知识结构的透明度。性能条形图立即提供策略有效性的视觉反馈，而使用计数显示哪些策略实际上正在应用。

社区成员映射揭示了用户如何根据其特征自动分段——这是提供适当组级指导的关键特性。发现的段显示的是从数据中出现的模式，而不是预先定义的。

这里是预期的输出：

```py
Procedural Memory Structure Visualization
![](img/Icon36.png) Strategy Performance by Scope:
![](img/Icon37.png) GLOBAL STRATEGIES (Universal Best Practices):
 portfolio_rebalancing ████████░░ 85.0%
  Used 1x | Segments: moderate_risk, millennials
 esg_investment_selection ████████░░ 85.0%
 general_investment ██████░░░░ 67.8%
![](img/Icon38.png) USER-SPECIFIC STRATEGIES:
 Total users with personalized strategies: 2
 Total personalized procedures: 2
 Example - User user_001:
 • user_user_001_preference_0 (success: 85.0%)..[see code lab for full output] 
```

可视化揭示了完整的学习状态：

+   全球策略作为具有不同成功率的通用回退

+   用户特定策略为活跃用户提供个性化

+   社区策略通过成员跟踪捕获段模式

+   任务策略为不同查询类型提供专门的解决方案

我们的可视化确认策略在层次结构中得到了适当的组织，具有清晰的性能指标和社区分配。然而，生产系统必须处理的不仅仅是理想场景。现实世界的使用不可避免地会产生边缘情况：空查询、不属于任何社区的用户，或者可能适用多个策略的情况。这些边界条件测试我们的层次检索是否真正能够优雅地退化。

让我们验证系统在面对意外输入或模糊情况时仍能保持稳健的行为。

### 第 5 步 d – 测试边缘情况和回退

最后，让我们验证系统是否能够优雅地处理边缘情况：

```py
# Cell 5d: Test Edge Cases and Fallbacks
print("Testing Edge Cases")
print("=" * 60)
# 1\. Empty query
print("\n1\. Empty/vague query:")
strategy = investment_memory.get_investment_strategy("", {}, None)
print(f"   Result: {'Strategy found' if strategy else 'No strategy (expected)'}")
# 2\. User with no community
print("\n2\. User with unassigned community:")
orphan_user = "orphan_user"
strategy = investment_memory.get_investment_strategy(
    "investment advice", 
    {"age": 200, "risk_tolerance": "unknown"},
    orphan_user
)
if strategy:
    print(f"   Fell back to: {strategy['scope']} scope")
# 3\. Conflicting scopes - what wins?
print("\n3\. Query matching multiple scopes:")
investment_memory.user_procedures["user_001"]["rebalancing_user"] = DomainProcedure(
    strategy_pattern="rebalancing_user",
    steps=["User-specific rebalancing"],
    success_rate=0.95,
    scope="user"
)
strategy = investment_memory.get_investment_strategy(
    "rebalance portfolio",  # Matches both user AND task
    {"age": 35, "risk_tolerance": "moderate"},
    "user_001"
)
print(f"   Winner: {strategy['scope']} scope (user > task in hierarchy)") 
```

这些边缘情况展示了生产就绪性：

+   **空查询**：即使输入最少，系统仍然尝试检索

+   **孤儿用户**：没有社区分配的用户优雅地回退到全局策略

+   **范围冲突**：当多个范围匹配时，层次结构（用户 > 社区 > 任务 > 全球）决定胜者

这种稳健性确保系统在意外情况下也不会失败，提供一些指导。

这里是预期的输出：

```py
Testing Edge Cases
1\. Empty/vague query:
 Result: Strategy found
2\. User with unassigned community:
 Fell back to: global scope
3\. Query matching multiple scopes:
 Winner: user scope (user &gt; task in hierarchy) 
```

边界情况处理证实了系统的弹性。空查询仍然可以通过部分匹配找到相关的策略。具有不寻常配置文件的用户会收到全局指导而不是错误。范围冲突可以按照层次结构可预测地解决。

这些检索和反馈机制完善了核心程序记忆系统。三个基本组件——学习、检索和适应——构成了反馈循环，使得持续改进成为可能。系统从对话中学习，检索最合适的策略，并根据实际表现进行优化。

在程序记忆系统完全功能化和测试后，我们准备将其与来自*第十七章*的事件记忆和语义记忆进行整合。在下一章中，我们将创建一个完整的 CoALA 智能体，它结合了所有三种记忆类型，以实现真正智能和自适应的行为。

## 架构影响和产品就绪性

通过程序记忆优化之旅揭示了我们在构建 AI 智能体时发生的根本性转变。我们不再受限于静态系统，这些系统无论经验如何都会重复相同的行为；相反，我们可以构建领域无关的学习框架，使智能体能够通过每一次互动进行适应和改进。我们开发的模块化架构，核心记忆系统和特定领域逻辑之间的完全分离，代表了 RAG 增强系统的一个关键进化。虽然传统的 RAG 检索文档，而增强记忆的 RAG 检索个性化上下文，程序记忆优化了智能体操作的机制，创建了一个元学习层，该层根据经验性能持续优化行为。

代码实验室已经展示了这一架构的实际应用，展示了单个程序记忆系统如何与任何实现标准接口的领域智能体协同工作。我们已经看到投资顾问领域完全在其自己的目录结构中运行，核心系统对投资概念保持完全的无知。通过五个实施步骤，我们构建了以下内容：

+   利用来自*第十七章*的基线智能体及其事件记忆和语义记忆的基础

+   支持分层学习的领域无关的程序结构

+   与任何领域协同工作的程序记忆系统，通过干净的接口

+   从多范围对话中提取模式的学习机制

+   检索和反馈系统，使持续改进成为可能

这个核心系统可以为医疗助手学习患者沟通模式、教育者发现有效的教学策略或客户服务机器人优化解决方案路径提供动力——所有这些都不需要修改任何一条程序记忆代码。

当你在自己的系统中实现程序性记忆时，请记住，这种模块化方法不仅仅是关于代码组织；它关乎实现快速创新。领域专家可以创建专门的代理，而不必了解记忆系统的复杂性。机器学习工程师可以在没有领域专业知识的情况下改进核心学习算法。组织可以使用相同的架构部署多个特定领域的代理。这种关注点的分离将程序性记忆从有趣的研究概念转变为构建生产就绪自适应代理的实用工具。

从静态代理到学习代理的转变不仅仅是一个技术升级；它是对人工智能助手可能性的根本重新构想。通过给你的代理提供在干净、可维护的架构中从经验中学习的能力，你不仅提高了它们的性能指标。你正在创建在其领域内建立真正专业知识、随时间增值，并能够适应每个部署上下文独特需求，同时保持可管理和可扩展的系统。

# 摘要

在本章中，我们探讨了程序性记忆如何将自我改进代理的理论承诺转化为实际可部署的现实。我们实施的清晰架构分离使得任何领域——投资咨询、医疗保健、教育或客户服务——只需通过实现领域接口，就能从复杂的学习能力中受益。分层学习方法确保模式在适当的范围（用户、社区、任务或全球）内应用，防止过度泛化，同时最大化学习知识的价值。综合反馈循环使持续改进成为可能，每一次交互都有可能提高未来的性能。

通过本章，你内化的基本原理，从分层学习、领域无关架构、持续适应到模块化设计，无论该领域如何发展，都将为你提供良好的服务。这些原理为评估新的 AI 技术和架构提供了一个思维框架：你将认识到系统是否正确地分离了关注点，学习是否发生在适当的分层级别，以及适应机制是否能够实现可持续的改进。无论你遇到新的记忆系统、新颖的学习算法，还是完全不同的代理框架，这些基础概念都将帮助你评估其设计质量，识别潜在问题，并做出明智的架构决策。但我们还没有完成。

虽然本章向您展示了如何构建程序性记忆系统本身，但**第十九章**将带您深入了解实际实施细节。我们将程序性记忆与**第十七章**中的情景记忆和语义记忆相结合，创建一个基于 CoALA 的完整认知架构，探索 LangMem 的不同学习算法及其适用场景，深入设计塑造您的智能体进化的领域度量标准，并提供一个将此系统适应任何选定领域的完整框架。从静态智能体到自适应智能体的旅程仍在继续，下一章将为您提供部署这些学习系统所需的一切。

# 免费订阅电子书

新框架、演进的架构、研究动态、生产分析——AI_Distilled 将噪音过滤成每周简报，供实际操作 LLMs 和 GenAI 系统的工程师和研究人员参考。现在订阅，即可获得免费电子书，以及每周的洞察力，帮助您保持专注并获取信息。

在[`packt.link/8Oz6Y`](https://packt.link/8Oz6Y)订阅或扫描下面的二维码。

![白色背景上的二维码  AI 生成的内容可能不正确。](img/B34736_Free_eBook.png)
