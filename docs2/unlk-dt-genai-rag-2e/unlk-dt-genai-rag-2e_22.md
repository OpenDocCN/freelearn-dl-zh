

# 第十九章：完整记忆集成的先进 RAG

在 *第十八章* 中，我们构建了程序性记忆的基础，这是一个领域无关的学习系统，它使代理能够从对话中提取模式并在适当的分层级别上应用它们。我们看到了这个系统如何将静态代理转变为适应性代理，学习对不同用户、社区和任务有效的方法。但这只是开始。现在，我们需要将这个程序性记忆与来自 *第十七章* 的情景记忆和语义记忆集成，以创建一个完整的认知架构，并探讨使这些系统适用于生产的实际考虑因素。

本章通过向您展示如何创建一个完全集成的 **语言代理的认知架构**（**CoALA**）代理，该代理结合了所有内存类型以实现真正的智能行为来完成旅程。我们将探索 LangMem 的不同学习算法以及何时每个算法表现最佳，深入到围绕度量设计的关键决策，这些决策从根本上塑造了您的代理如何进化，并提供了一个将此架构适应任何领域的完整框架。无论您是在构建医疗助手、教育导师还是客户服务机器人，本章都为您提供了部署生产就绪学习代理所需的一切。

在我们深入之前，让我们明确本章中库和自定义代码之间的关系。

LangMem 是 LangChain 提供的一个真实、开源的库，它为内存提取和提示优化提供了核心原语（`create_memory_manager`，`create_prompt_optimizer`）。然而，LangMem 本身并不提供完整的代理架构。在 *第十七章* 到 *第十八章* 的过程中，我们一直在 LangMem 的基础上构建一个自定义框架，实现了具有分层学习、领域分离和集成内存类型的 CoALA 模式。你将需要处理的文件，包括 `coala_agent.py`、`procedural_memory.py`、`domain_agent.py` 以及特定领域的实现，都是为这本书编写的自定义代码。它们展示了如何使用 LangMem 的构建块来设计生产就绪的学习代理。所有这些代码都可在 GitHub 仓库中找到，供你使用、修改和适应你自己的领域。

在本章中，我们将涵盖以下关键主题：

+   代码实验室 19.1 – 创建一个具有集成内存系统的完整 CoALA 代理

+   LangMem 的学习算法：选择正确的途径

+   设计领域度量：衡量成功的艺术

+   适应您的领域：从投资顾问到任何专家系统

我们在 *第十八章* 中建立的基础，现在需要与来自 *第十七章* 的情景记忆和语义记忆集成。虽然 *第十八章* 展示了程序性记忆如何学习和适应策略，但真正的力量在于所有三种记忆类型共同工作 – 情景记忆提供对话上下文，语义记忆提供提取的事实，程序性记忆应用学习到的策略。

让我们构建这个完整的系统，看看记忆是如何相互加强，以创建真正适应性强、智能的代理。

# 技术要求

要完成本章的动手练习，你需要以下软件和资源：

+   **开发环境**：你需要一个能够运行代码的 Python 3.8+ 环境。虽然示例可以通过多种方式运行（Python 脚本、IDE 或命令行），但推荐使用 Jupyter Notebook 环境，因为它允许你逐个步骤地执行代码单元格，并观察每个阶段的进度。

+   **软件** **要求**：

    +   **Python 3.8 或更高版本**：运行代码实验室所必需。

    +   **pip 软件包管理器**：用于安装 Python 依赖项。

    +   **OpenAI API 访问**：你需要一个 OpenAI API 密钥来访问 GPT 模型和生成嵌入。请将你的密钥安全地存储在 `env.txt` 文件中，格式为 `OPENAI_API_KEY=your_key_here`。

    +   **文本编辑器或 IDE**：VS Code、PyCharm、Jupyter Notebook 或类似工具，用于编写和运行 Python 代码。

    +   **Git**（可选）：用于克隆 GitHub 仓库。

+   **硬件** **要求**：

    +   **至少 8GB RAM**（推荐 16GB 以获得与向量数据库顺畅性能）。

    +   **2 GB 空闲磁盘空间** 用于依赖项、向量存储和持久性内存存储

    +   **互联网连接** 用于下载软件包和访问 OpenAI API

+   **章节资源**：

    +   **GitHub 仓库** [`github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_19`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_19)

    +   **完整代码文件**：代码实验室（`19.1`）在 GitHub 仓库中提供参考。

如果你在任何步骤需要验证你的实现，GitHub 仓库中的完成代码可以作为参考。

# 代码实验室 19.1 – 创建一个具有集成记忆系统的完整 CoALA 代理

在程序性记忆系统完全功能并经过测试后，我们准备将其与来自 *第十七章* 的情景记忆和语义记忆集成。接下来，我们将创建一个完整的 CoALA 代理，该代理结合了所有三种记忆类型，以实现真正智能、自适应的行为。

## 第 1 步 – 创建包含所有记忆类型的完整代理

现在，我们将我们的领域无关的程序记忆系统与基线代理集成，以创建一个完整的 CoALA 代理。这展示了三种内存类型如何通过一个干净、模块化的架构协同工作，其中领域特定逻辑与核心记忆系统完全隔离。

我们首先导入必要的组件：

```py
import os
import sys
import json
# Import the complete CoALA system
from coala_agent import CoALAAgent
from domain_investment.investment_advisor_agent import InvestmentAdvisorAgent
from domain_investment.investment_advisor_data import EnhancedInvestmentAdvisorDataGenerator
from domain_investment.investor_test_scenarios import (
    process_baseline_conversations, test_agent_with_queries, 
    process_performance_feedback, process_remaining_conversations,
    test_hierarchical_retrieval, get_key_achievements
)
from dotenv import load_dotenv
load_dotenv(dotenv_path='env.txt')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') 
```

导入揭示了我们的系统模块化结构。`CoALAAgent`类是领域无关的核心，它集成了所有三种内存类型。所有领域特定组件都来自`domain_investment`目录，包括实现领域接口的`InvestmentAdvisorAgent`、数据生成器和测试工具。这种分离意味着切换领域只需要更改这些导入以指向不同的领域目录。

接下来，我们初始化我们的领域代理并创建完整的 CoALA 代理：

```py
domain_agent = InvestmentAdvisorAgent()
domain_memory_dir = os.path.join(
    domain_agent.domain_dir, "domain_memory_store")
os.makedirs(domain_memory_dir, exist_ok=True)
# Create the full agent with all memory systems
full_agent = CoALAAgent(
    domain_agent=domain_agent,
    model_name="gpt-4.1-mini",
    temperature=0.0,
    persist_directory=domain_memory_dir,
    optimization_algorithm="prompt_memory"  # Can be "gradient" or "metaprompt"
) 
```

`InvestmentAdvisorAgent`封装了所有投资特定的逻辑：提示、社区定义、成功指标和任务分类。它还管理自己的目录结构，将所有领域数据隔离。`CoALAAgent`类接收这个领域代理并自动使用领域的内存目录进行持久化，将所有领域特定决策委托给领域代理，并使用领域的接口集成所有三种内存类型。`optimization_algorithm`参数选择使用哪种 LangMem 学习方法；我们将在本章后面探讨这些替代方案。

现在，让我们测试代理的基线能力：

```py
initial_stats = full_agent.get_memory_stats()
test_response = full_agent.process_message(
    "I'm thinking about rebalancing my portfolio. I'm 35 with moderate risk tolerance.",
    user_id="test_client_001"
) 
```

这个测试消息在发生任何学习之前就测试了完整的代理。代理可以使用其基本知识进行响应，但缺乏使其建议真正个性化的程序策略。通过在这次交互之前捕获`initial_stats`，我们为衡量学习进度建立了一个基线。

这里的关键架构成就是关注点的完全分离：

1.  **领域代理创建**：`InvestmentAdvisorAgent`封装了`domain_investment`目录中的所有投资特定逻辑。这包括以下内容：

    +   投资特定的提示用于学习和响应生成

    +   社区定义（保守的退休人员、激进的千禧一代等）

    +   专门针对投资结果的成功指标

    +   投资查询的任务分类

领域代理还管理自己的目录结构，将所有领域特定数据和内存存储隔离。这意味着你可以同时运行多个领域代理而不会相互干扰。

1.  **CoALA 代理集成**：`CoALAAgent`类完全不受领域限制。它接收领域代理并自动执行以下操作：

    +   使用领域的内存目录进行持久化

    +   将所有领域特定决策委托给领域

    +   使用领域的接口集成所有三种内存类型

这种干净的分离意味着创建一个新的领域（如医疗保健或教育）不需要对核心 CoALA 代理进行任何更改；只需按照相同的接口实现一个新的领域代理。

1.  **记忆系统集成**：完整的代理无缝集成了 CoALA 框架中的所有三种记忆类型：

    +   **情景记忆**：使用领域的摘要格式存储完整的对话

    +   **语义记忆**：使用领域的提取提示提取事实

    +   **程序记忆**：使用领域的学习提示和成功指标学习策略

这种集成使得 LangMem 所说的**交叉引用分析**成为可能。当用户询问关于再平衡的问题时，代理可以执行以下操作：

+   回忆他们关于投资的过去对话（情景）

+   访问他们存储的风险配置文件和目标（语义）

+   应用对类似用户最成功的再平衡策略（程序）

同时，领域代理处理这些记忆的投资特定解释。

现在让我们通过检查代理的初始状态和测试其基线能力来验证初始化：

```py
print(f" • Domain: {full_agent.domain_agent.class.name}")
print(f" • Memory Store: {domain_memory_dir}")
print(f"\n![](img/Icon49.png) Initial state:") 
print(f" Episodic/Semantic docs: {initial_stats.get('episodic_semantic', {}).get('total_documents', 0)}") 
print(f" Procedural strategies: {initial_stats.get('procedural', {}).get('total_strategies', 0)}")
print(f"\n![](img/Icon50.png) Test Response: {test_response}") 
```

这是预期的输出：

```py
![](img/Icon82.png) Full CoALA agent initialized with:
 • Domain: InvestmentAdvisorAgent
 • Memory Store: .../CHAPTER19/domain_investment/domain_memory_store
![](img/Icon83.png) Initial state:
 Episodic/Semantic docs: 0
 Procedural strategies: 0
![](img/Icon84.png) Test Response: Given your age of 35 and moderate risk tolerance, a balanced approach to rebalancing your portfolio can help you optimize growth while managing risk effectively. Here's a specific, actionable strategy... 
```

初始化确认我们的模块化架构正在正确工作：

+   **领域识别**：系统识别其正在与`InvestmentAdvisorAgent`一起工作，但会以相同的方式与任何其他领域代理一起工作

+   **隔离的记忆存储**：记忆存储位于领域的目录内（`domain_investment/domain_memory_store`），保持所有领域数据隔离

+   **干净的初始状态**：从零文档和零策略开始，确认我们有一个全新的系统，准备好学习

+   **通用的初始响应**：测试响应是合理的但通用的；代理提供一般性的投资建议，没有任何学习模式或个性化

初始响应展示了基线能力。代理可以使用其基础知识讨论再平衡，但它缺乏使建议真正有效的程序策略。响应是连贯且相关的，但它缺少程序学习将提供的具体、可操作的步骤。

以下展示了以下架构优势：

+   **模块化**：整个投资领域都包含在`domain_investment/`中

+   **可重用性**：相同的`CoALAAgent`类可以与任何领域一起工作

+   **隔离**：领域数据和记忆与核心系统保持分离

+   **可扩展性**：添加一个新的领域就像创建一个新的领域目录和代理一样简单

这种干净的架构意味着一个医疗保健组织可以采用相同的代码，创建一个包含`HealthcareAgent`的`domain_healthcare`目录，并拥有一个完全功能化的医疗保健顾问，而无需触及任何核心记忆系统代码。

在我们的完整代理初始化并准备就绪后，我们现在可以加载特定领域的对话数据以同时训练所有三个记忆系统。接下来，我们将展示代理如何从现实对话中学习，构建其关于情景、语义和程序性记忆的知识库。

## 第 2 步 - 加载合成投资数据

现在，我们将加载展示各种模式的真实投资顾问对话，例如成功的投资组合讨论、失败的咨询尝试和不同的客户群体。这些数据将训练我们的代理的程序性记忆，以识别哪些有效，同时在领域目录结构内保持完全隔离。

数据加载过程很简单。我们检查现有的对话数据并加载它，或者在需要时生成新数据：

```py
data_dir = domain_agent.data_dir
conversations_file = os.path.join(data_dir, "conversations.jsonl")
if os.path.exists(conversations_file):
    # Load existing data
    print(f"![](img/Icon51.png) Loading existing conversation data from {data_dir}...")
    conversations = []
    with open(conversations_file, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
else:
    # Generate new data
    print(f"![](img/Icon52.png) Generating new conversation data...")
    generator = EnhancedInvestmentAdvisorDataGenerator(seed=42)
    data = generator.export_realistic_data()
    conversations = data['conversations']

    # Convert to dict format
    conversations = [
        conv if isinstance(conv, dict) else conv.__dict__ 
        for conv in conversations
    ] 
```

数据加载过程展示了该领域的自包含特性。`InvestmentAdvisorAgent`管理其自身的数据目录（`domain_investment/investment_advisor_data/`），将所有特定领域的数据与核心系统隔离。这意味着多个领域可以共存而不相互干扰；医疗保健领域会使用`domain_healthcare/healthcare_data/`，教育领域会使用`domain_education/education_data/`，等等。

合成数据生成器创建 LangMem 数据收集层所需的具有丰富上下文信息的真实投资顾问对话：

+   **对话元数据**：

    +   **成功指标**：对话是否达到了其目标

    +   **满意度评分**：客户满意度评分（1-5 级）

    +   **行为信号**：追踪的图案，如`提供的特定数据`和`个性化响应`

    +   **查询类型**：绩效分析、再平衡、风险评估、税务规划等

+   **数据多样性**：生成的对话代表了各种投资场景和客户类型。这种多样性对于 LangMem 的轨迹聚类能力至关重要；系统需要成功和失败的交互的例子来识别它们之间的区别。以下适用于我们的投资顾问：

    +   一些客户更喜欢详细的解释，而另一些则希望得到快速答案

    +   某些方法对退休人士比年轻专业人士更有效

    +   税收讨论需要不同于绩效评估的处理方式

在其他领域，这种多样性会捕捉到以下内容：

+   **医疗保健**：不同的患者沟通风格、症状表现、治疗反应

+   **教育**：不同的学习速度、学科难度、参与模式

+   **客户服务**：问题类型、升级路径、解决方案策略

数据中的行为信号对应于 LangMem 所说的“超越简单日志的丰富上下文信息。”这些信号帮助系统识别微妙的模式——例如，成功的技术故障排除在对话早期就包括诊断问题，而跳到解决方案则与失败相关。

让我们检查加载的数据以了解其特征：

```py
print(f"![](img/Icon55.png) Loaded {len(conversations)} conversations")
print(f"![](img/Icon56.png) Unique users: {len(set(c['user_id'] for c in conversations))}")
# Display data statistics
success_rate = sum(1 for c in conversations if c['feedback']['success']) / 
    len(conversations)
avg_satisfaction = sum(c['feedback']['satisfaction_score'] for c in 
    conversations) / len(conversations)
print(f"\n![](img/Icon44.png) Data Overview:")
print(f"  Data location: {data_dir}")
print(f"  Total conversations: {len(conversations)}")
print(f"  Unique users: {len(set(c['user_id'] for c in conversations))}")
print(f"  Success rate: {sum(1 for c in conversations if c['feedback']['success']) / len(conversations):.1%}")
print(f"  Avg satisfaction: {sum(c['feedback']['satisfaction_score'] 
    for c in conversations) / len(conversations):.1f}/5.0")
# Examine a sample conversation
sample_conv = conversations[0]
print(f"\n![](img/Icon54.png) Sample Conversation:")
print(f"  User {sample_conv['user_id']}: {sample_conv['messages'][0]['content']}")
print(f"  Assistant: {sample_conv['messages'][1]['content']}")
print(f"  Success: {sample_conv['feedback']['success']}")
print(f"  Satisfaction: {sample_conv['feedback']['satisfaction_score']}/5.0")
print(f"  Behavioral signals: {sum(sample_conv['behavioral_signals'].values())} active")
print(f"\n![](img/Icon57.png) Data ready for processing by full agent\n  Will be stored in: {domain_agent.memory_dir}") 
```

这是预期的输出：

```py
![](img/Icon97.png) Generating new conversation data to ...CHAPTER18/domain_investment/investment_advisor_data...
![](img/Icon85.png) Generating realistic investment advisor conversations...
![](img/Icon86.png) Generated 500 realistic conversations
![](img/Icon87.png) Success rate: 70.2%
![](img/Icon88.png) Average satisfaction: 3.1/5.0
![](img/Icon94.png) Extracted 2 universal patterns
![](img/Icon96.png) Identified 0 antipatterns
![](img/Icon91.png) Loaded 500 conversations
![](img/Icon92.png) Unique users: 50
![](img/Icon93.png) Data Overview:
 Data location: ...CHAPTER18/domain_investment/investment_advisor_data
 Total conversations: 500
 Unique users: 50
 Success rate: 70.2%
 Avg satisfaction: 3.1/5.0
![](img/Icon94.png) Sample Conversation:
 User 3000: hey how are my investments doing?
 Assistant: You're up about 2.3% right now. SPY is doing pretty well....
 Success: True Satisfaction: 3.0/5.0
 Behavioral signals: 2 active
![](img/Icon95.png) Data ready for processing by full agent
 Will be stored in: ...CHAPTER18/domain_investment/domain_memory_store 
```

数据显示了程序性学习所必需的现实模式：

+   **混合成功率**：并非所有对话都成功（70.2%的成功率），反映了现实世界的复杂性。这种变化使系统能够学习区分成功与失败的因素。

+   **可变满意度**：平均满意度为 3.0/5.0 显示出显著的改进空间——这正是程序性记忆通过学习旨在实现的目标。

+   **行为模式**：样本对话显示了两个活跃的行为信号。这些信号将帮助识别相关性；例如，`provided_specific_data`通常与投资讨论中的更高满意度相关。

+   **多样化的用户基础**：50 个独特的用户提供了足够的多样性以进行社区细分，同时保持数据集在演示中可管理。

+   **独立数据存储**：数据位置确认所有内容都保持在域目录内（`domain_investment/investment_advisor_data/`），而记忆将被存储在域的记忆目录中（`domain_investment/domain_memory_store/`）。这种完全隔离意味着您可以删除整个`domain_investment/`目录，而核心系统将保持完整。

样本对话演示了使用特定投资组合数据和适度满意度（3.0/5.0）的成功绩效查询。此类交互将被分析以提取可应用于类似未来查询的模式的模式。注意对话式的自然语言风格——“嘿，我的投资情况怎么样？”——这反映了真实用户交互而不是正式查询。

域代理的数据目录结构保持一切井然有序：

+   `investment_advisor_data/`：包含生成的对话和模式

+   `domain_memory_store/`：将包含情景、语义和程序性记忆

+   所有路径都由域代理管理，而不是在核心系统中硬编码

现在，让我们处理基线对话以建立我们的代理的初始学习状态，展示程序性记忆如何从这些特定领域的交互中提取模式，同时核心系统对投资背景保持无意识。

## 第 3 步 - 处理基线对话以进行初始学习

现在，我们将处理我们的第一批对话以建立代理的初始学习状态。这展示了 LangMem 的模式挖掘如何将对话分割成功能单元，并识别出在多个层次范围内成为程序性规则的成功的模式，同时保持域无关的架构。

我们将处理 30 个基线对话并检查所有 3 个记忆系统中发生的学习：

```py
results = process_baseline_conversations(
    full_agent, conversations, num_baseline=30)
learning = results['learning_summary']
processed = results['processed']
for community_id, members in full_agent.procedural_memory.community_members.items():
    if members:
         print(f" {community_id}: {len(members)} members")
final_stats = full_agent.procedural_memory.get_stats() 
```

`process_baseline_conversations`处理函数（从领域的测试场景中导入）封装了复杂的学习管道，同时保持主要代码的整洁。这展示了如何将特定领域的处理逻辑模块化并在不同的实验中重用。

以下展示了 LangMem 的三阶段管道在实际操作中的表现：

+   **提取阶段**：系统将对话拉入适当的记忆存储。完整的交互进入情景记忆以供未来参考，而具体事实则使用领域的提取提示提取到语义记忆中。这种全面的数据收集捕捉了 LangMem 所说的“丰富上下文信息”，不仅包括所说的话，还包括其成功程度、存在的行为信号以及产生的结果。

+   **分析阶段**：模式挖掘过程使用显式成功指标（`success=true`和`satisfaction ≥4.0`的对话）和隐性行为信号来识别有效的方法。这展示了 LangMem 的成功相关性分析；只有高性能对话中的模式成为程序规则。系统根据用户的特征将用户分割成社区，使 LangMem 所说的“轨迹聚类”成为可能，即相似的用户类型被分组以识别特定段落的模式。

+   **综合阶段**：`learn_from_interaction`方法将识别到的模式转换成多个范围内的具体程序规则。适用于普遍情况的全球模式存储在全局级别，而用户特定的偏好保持个性化。这种分层组织防止了过度泛化，这是一个关键的安全功能，确保适合激进的千禧一代的建议不会应用于保守的退休人员。

社区分配逻辑展示了涌现的分割。系统不是要求预定义的用户类别，而是通过实际使用模式发现有意义的群体。对于我们投资顾问来说，社区围绕着年龄和风险承受能力组合形成。在医疗保健领域，它们可能围绕着疾病类型和治疗反应形成。在教育领域，它们可能围绕着学习风格和学科偏好形成。

跟踪和报告提供了 LangMem 所强调的透明度。每个学习到的模式都可以追溯到生成它的特定对话，使开发者能够确切了解系统为什么学习它所学习的内容。这种可审计性对于必须可解释和可逆的生产部署至关重要。

让我们来看看系统从这些基线对话中学到了什么：

```py
print(f"![](img/Icon58.png) Processing baseline conversations...")
print(f"\n![](img/Icon59.png) Baseline processing complete")
print(f" Episodic memories stored: {results['baseline_count']}")
print(f" Semantic facts extracted: {results['facts_extracted']} (errors: {results['extraction_errors']})")
print(f" Global strategies: {learning.get('global', 0)}")
print(f" User strategies: {learning.get('user', 0)} (for {processed.get('users', 0)} users)")
print(f" Community strategies: {learning.get('community', 0)} (for {processed.get('communities', 0)} communities)")
print(f" Task strategies: {learning.get('task', 0)} (for {processed.get('tasks', 0)} task types)")
print("\n![](img/Icon60.png) Community Membership:")
print(f"\n![](img/Icon61.png) Procedural Memory Statistics:")
print(f" Total strategies: {final_stats['total_strategies']}")
print(f" By scope: {final_stats['by_scope']}")
print(f" Avg success rate: {final_stats['avg_success_rate']:.2f}") 
```

这是预期的输出：

```py
![](img/Icon62.png) Processing baseline conversations...
Processed 10/30...
Processed 20/30...
Processed 30/30...
![](img/Icon63.png) Baseline processing complete
 Episodic memories stored: 30
 Semantic facts extracted: 71 (errors: 0)
 Global strategies: 1
 User strategies: 3 (for 3 users)
 Community strategies: 1 (for 1 communities)
 Task strategies: 1 (for 1 task types)
![](img/Icon64.png) Community Membership:
 aggressive_millennials: 1 members
 moderate_professionals: 3 members
![](img/Icon65.png) Procedural Memory Statistics:
 Total strategies: 10
 By scope: {'global': 2, 'user': 4, 'community': 2, 'task': 2}
 Avg success rate: 0.85 
```

结果显示，仅从 30 个基线对话中就实现了成功的多范围学习：

+   **记忆集成成功**：

    +   **存储 30 个情景记忆**：完整对话供未来参考

    +   **提取了 71 个语义事实**：关于用户、偏好和投资概念的具体知识

    +   **学习到的 10 种程序策略**：分布在各个范围内的具体行动模式

+   **学习策略分布**：

    +   **两种全局策略**：适用于所有用户的通用模式

    +   **四种用户策略**：针对三个特定个人的个性化方法（一些用户有多个策略）

    +   **两种社区策略**：已识别社区的图案

    +   **两种任务策略**：针对不同查询类型的专用程序

这种分布展示了 LangMem 自动确定适当学习范围的能力。系统了解到某些模式具有普遍性，属于全局级别，而其他模式则保持用户或社区特定。

+   **社区形成**：数据自然出现两个社区：

    +   `moderate_professionals`：通过共享特征识别的三个成员

    +   `aggressive_millennials`：一个具有独特风险偏好的成员

每个社区都有独特的成功模式；中等专业人士可能更喜欢平衡的方法，而激进的千禧一代则青睐增长导向的策略。这种细分使得能够提供细微的响应，避免了“一刀切”的问题。

+   **成功率基线**：`0.85`的平均成功率代表对这些初始策略的信心。随着系统处理更多对话并收到性能反馈，这些比率将根据现实世界的有效性进行调整。持续成功的策略将看到其信心增加，而失败的策略将被降权或最终移除。

程序记忆现在包含 10 个具有特定步骤的具体策略，而不是模糊的建议。每个策略包括以下内容：

+   需要遵循的确切步骤

+   它应用的条件

+   基于历史表现的置信水平

+   投资顾问跟踪的特定领域指标

这种将学习形式化为“特定条件结构”的做法使得学习可操作和可衡量。领域代理处理所有特定投资的解释，而核心程序记忆系统管理学习机制。

在所有三种记忆类型上建立基线学习后，代理现在配备了初始知识和策略。接下来，我们将用新的查询测试代理，看看它如何应用这些学习到的策略，然后处理更具挑战性的场景以触发程序适应。

## 第 4 步 - 测试改进后的性能并触发适应

我们将使用代理学习到的策略测试其改进的能力，然后处理将触发程序记忆更新的额外对话，展示 LangMem 基于反馈的持续学习和适应。

让我们先测试代理如何应用其学习到的策略，然后处理额外的对话以触发适应：

```py
test_results = test_agent_with_queries(full_agent)
adaptations, adaptation_details = process_performance_feedback( 
    full_agent, conversations, start_idx=30, end_idx=40 )
final_stats = full_agent.get_memory_stats() 
```

这些测试查询旨在在不同层次上触发学习到的策略。可持续投资的查询如果可用，应激活用户特定或任务特定的策略，并在需要时通过层次结构回退。再平衡查询测试系统是否可以根据用户资料应用适当的策略。代理的响应现在结合了从成功对话中学到的具体步骤，展示了程序记忆如何提高响应质量。

性能反馈处理模拟了现实世界的使用情况，其中策略会接收到不同的成功信号。通过处理 30-40（超出初始基线）的对话，我们测试了系统如何适应新的数据模式。每次策略更新都使用领域代理的成功评分，对于投资而言，它高度重视回报，但也考虑满意度和目标达成情况。

对于我们的投资顾问示例，系统应用学习到的策略，然后根据结果对其进行细化。在医疗保健领域，这就像应用诊断方案并根据患者结果进行调整。在客户服务中，这意味着遵循升级路径并根据解决率进行细化。在教育领域，这涉及使用教学技巧并根据学生的理解情况进行调整。

现在，让我们看看代理的表现以及发生了哪些适应：

```py
print("![](img/Icon98.png) Testing agent with learned strategies...")
for result in test_results:
    print(f"\n![](img/Icon99.png) {result['query']}")
    print(f"![](img/Icon100.png) {result['response']}")
    if result['strategy']:
        print(f" → Using: {result['strategy']['source']} ({result['strategy']['confidence']:.0%})")
print("\n![](img/Icon101.png) Processing performance feedback...")
for detail in adaptation_details:
    print(f" ![](img/Icon102.png) {detail['strategy']} → {detail['new_rate']:.0%}")
print(f"\n![](img/Icon103.png) {adaptations} strategies adapted")
print("\n![](img/Icon104.png) Final Memory State:")
print(f" Episodic/Semantic: {final_stats['episodic_semantic'].get('total_documents', 0)} documents")
print(f" Procedural: {final_stats['procedural'].get('total_strategies', 0)} strategies")
print(f" Data location: {full_agent.domain_agent.data_dir}")
print(f" Memory location: {full_agent.domain_agent.memory_dir}") 
```

这是预期的输出：

```py
![](img/Icon105.png) Testing agent with learned strategies...
![](img/Icon106.png) I want to invest in sustainable companies
![](img/Icon107.png) Given your conservative investment approach and current portfolio setup, integrating sustainable companies can be done thoughtfully to maintain your r...
 → Using: personalized for 3001 (85%)
![](img/Icon66.png) Time to rebalance my portfolio?
![](img/Icon67.png) Given your conservative investment approach and current portfolio performance of about +2.3% this year, here's a tailored recommendation on rebalancin...
 → Using: personalized for 3002 (85%)
![](img/Icon68.png) Processing performance feedback...
 ✓ conservative sustain... → 82%
 ✓ balanced_rebalancing... → 68%
..[see code lab for full output] 
```

结果展示了复杂的学习和适应：

+   **策略应用**：两个测试查询都成功触发了个性化策略（85%置信度），表明系统已经学会了用户特定的模式。响应针对每个用户的个人资料定制；注意两个响应都提到了`保守的投资方法`和具体的投资组合细节，这表明代理正在将程序策略与关于用户的语义事实相结合。

+   **性能适应**：反馈处理显示了现实策略的演变：

    +   一些策略得到了改进（`balanced_rebalancing`: 68%）

    +   其他策略因结果不佳而放弃（`conservative_sustain`: 82% → 74%）

    +   对同一策略的多次适应显示了持续的细化

这种现实中的改进与退化的混合反映了现实世界的学习过程，并非每一次尝试都能成功。当策略失败时，系统降低信心与成功时提高信心同样重要。

+   **记忆集成**：最终状态显示了所有三种记忆类型的全面学习：

    +   **40 个情节/语义文档**：丰富的对话历史和提取的事实

    +   **12 个程序策略**：通过学习和适应从最初的 10 个策略进化而来

    +   **完全隔离**：所有数据都保留在领域目录中

适应细节揭示了正在细化的特定策略。注意`conservative_sustain`策略出现多次。这个策略正在根据性能持续调整，展示了系统调整个别策略而不是全面替换的能力。

这里是这个步骤的关键见解：

+   **个性化有效**：代理成功应用用户特定的策略，表明分层检索正在工作

+   **持续学习**：策略根据表现而非仅初始学习进行适应

+   **现实进化**：并非所有适应都是改进；系统可以识别并降低失败策略的权重

+   **领域隔离**：所有学习都在领域目录内进行，保持清晰的分离

反馈处理展示了 LangMem 的多源反馈三角测量。领域代理的加权公式反映了投资优先级：50%实际回报，30%客户满意度，20%目标达成。基于动量的更新（80%旧，20%新）防止单个异常值剧烈改变策略，同时仍允许适应。这使 LangMem 称为**安全回滚**：如果策略的表现持续下降，我们可以通过适应历史识别问题，并可能回滚。

在代理现在基于基线对话进行训练并通过性能反馈进行适应后，我们准备好展示完整的学习进度。接下来，我们将展示所有三种记忆类型如何通过扩展训练协同工作，以创建一个越来越复杂的投资顾问。

## 第 5 步 - 完整学习进度和分层检索

我们将展示完整的学习旅程，展示 LangMem 如何将我们的代理从通用顾问转变为一个从每次互动中学习和适应的复杂系统，同时保持领域逻辑和核心记忆系统之间的清晰分离。

让我们处理额外的对话并展示我们集成记忆系统的全部功能：

```py
num_processed, learned = process_remaining_conversations(
    full_agent, conversations, 50, 100)
final_stats = full_agent.procedural_memory.get_stats()
query = "I'm worried about market volatility. Should I move to safer investments?" 
test_results = test_hierarchical_retrieval(full_agent, query)
memory_stats = full_agent.get_memory_stats() 
```

这全面的演示展示了集成记忆系统的全部力量。处理函数处理超出基线的基础对话，但通过 LangMem 的质量过滤，只有高满意度的对话（4.5+/5.0）对学习做出贡献。这种选择性的方法确保代理仅纳入经过验证的成功模式。

代码通过领域无关的程序记忆系统处理了 50 个额外的对话，该系统将所有投资特定决策委托给领域代理。相同的处理管道对于从患者互动学习的医疗领域或从辅导课程学习的教育领域的工作方式完全相同。

现在，让我们看看我们学习进度的完整结果：

```py
print("![](img/Icon44.png) COMPLETE LEARNING PROGRESSION ANALYSIS")
print(f"\n![](img/Icon70.png) Processing additional conversations...")
print(f"\n![](img/Icon71.png) Learning complete ({num_processed} conversations processed):")
for scope, count in learned.items():
    if count:
        print(f" {scope.capitalize()}: {count} new strategies")
final_stats = full_agent.procedural_memory.get_stats()
print(f"\n![](img/Icon72.png) FINAL PROCEDURAL MEMORY STATISTICS:")
print(f" Total strategies learned: {final_stats['total_strategies']}")
print(f" Breakdown by scope:")
for scope, count in final_stats['by_scope'].items():
    print(f" {scope.capitalize()}: {count}")
print(f" Average success rate: {final_stats['avg_success_rate']}")
print(f" Total adaptations: {final_stats['total_adaptations']}")
print(f" Segments discovered: {', '.join(final_stats['segments'])}")
print("![](img/Icon73.png) DEMONSTRATING FULL MEMORY INTEGRATION")
query = "I'm worried about market volatility. Should I move to safer investments?"
test_results = test_hierarchical_retrieval(full_agent, query)
for result in test_results:
     print(f"\n![](img/Icon74.png) {result['description']}\n User ID: {result['user_id']}")
     if result['strategy']:
        print(f" Strategy source: {result['strategy']['source']}")
        print(f" Scope: {result['strategy']['scope']}")
        print(f" Confidence: {result['strategy']['confidence']:.1%}")
        print("\n![](img/Icon44.png) STRATEGY PERFORMANCE BY SCOPE:")
full_agent.procedural_memory.show_strategy_performance()
print("![](img/Icon81.png) KEY ACHIEVEMENTS DEMONSTRATED:")
for achievement in get_key_achievements():
    print(f"✓ {achievement}")
memory_stats = full_agent.get_memory_stats()
print(f"\n![](img/Icon61.png) COMPLETE MEMORY SYSTEM STATISTICS:")
print(f" Episodic/Semantic documents: {memory_stats['episodic_semantic'].get('total_documents', 'N/A')}")
print(f" Procedural strategies: {memory_stats['procedural']['total_strategies']}")
print(f" Optimization algorithm: {memory_stats['procedural']['algorithm']}") print(f" Total optimizations: {memory_stats['procedural']['total_optimizations']}") 
```

这是预期的输出：

```py
![](img/Icon75.png) COMPLETE LEARNING PROGRESSION ANALYSIS
![](img/Icon76.png) Processing additional conversations...
Processed 10/50...
Processed 20/50...
Processed 30/50...
Processed 40/50...
![](img/Icon77.png) Learning complete (50 conversations processed):
![](img/Icon78.png) FINAL PROCEDURAL MEMORY STATISTICS:
 Total strategies learned: 13
 Breakdown by scope:
  Global: 4
  User: 4
  Community: 2
  Task: 3
 Average success rate: 0.8
 Total adaptations: 10
 Segments discovered: millennials, conservative, moderate_risk, sustainable_investors, retirees..[see code lab for full output]
The final demonstration reveals the complete learning achievement: 
```

+   **学习进度**：

    +   **处理 50 次对话**：展示了超越初始基线的可扩展性

    +   **总共学习 13 种策略**：分布在所有分层范围内

    +   **平均成功率 80%**：显示出现实的表现水平

    +   **总共 10 次调整**：策略根据反馈持续优化

    +   **发现 5 个细分市场**：`millennials`、`conservative`、`moderate_risk`、`sustainable_investors`、`retirees`

+   **实际中的分层检索**：关于市场波动的测试查询展示了完美的分层检索：

    +   **经验丰富的用户（**`3001`**）**：以 85%的置信度获得个性化策略

    +   **保守型用户（**`3010`**）**：在 73.9%的情况下回退到全球最佳实践

    +   **新用户（**`new_user_9999`**）**：也收到全球指导，显示出一致的回退

+   **策略性能可视化**：性能条形图揭示了策略的演变：

    +   **全局策略**显示不同的使用频率（1-8 次）和成功率（65-74%）

    +   **用户策略**保持高个性化，成功率达到 85%

    +   **社区策略**有效地服务于其细分市场，成功率达到 85%

    +   **任务策略**提供具有一致 85%成功率的专门方法

+   **记忆集成统计**：

    +   **133 个情景/语义文档**：丰富的对话历史和事实

    +   **13 个程序性策略**：可操作的模式准备应用

    +   **完全隔离**：所有学习都包含在域目录中

下面是展示的关键见解：

+   **多范围学习**：系统在所有四个分层级别上成功学习，策略根据其适用性适当分布

+   **持续适应**：策略显示出现实的演变；一些策略得到改进（例如，保守型可持续投资使用了 8 次），而其他策略则需要细化（`conservative_rebalancing`在 65.6%）

+   **新兴细分市场**：五个不同的细分市场自然地从数据中产生，每个都有其特征模式

+   **域模块化**：整个学习过程使用域代理接口，投资特定的逻辑完全隔离在域目录中

+   **生产就绪**：系统处理边缘情况，维护审计跟踪，并通过综合统计数据提供透明度

通过这个代码实验室，我们展示了程序性记忆如何将投资顾问代理从提供通用响应转变为提供个性化、持续改进的建议。该代理实现了以下成果：

+   在多个范围内（4 个全局、5 个用户、3 个社区和 3 个任务）有 15 种不同的策略

+   通过持续适应，平均成功率达到了 82%

+   通过使用发现 5 个具有特定需求的客户细分市场

+   基于性能反馈的 18 种策略调整

+   完全集成所有三种记忆类型以提供全面响应

最重要的是，整个系统是领域无关的。程序记忆系统、CoALA 智能体和学习管道在医疗保健、教育或客户服务领域将工作方式相同。只有领域智能体实现不同，展示了适当架构分离的力量。

这展示了 LangMem 的核心承诺：将智能体从静态的响应者转变为动态的学习者，它们随着每次对话而进化，使持续改进无需昂贵的重新训练周期。分层学习方法确保模式在适当的范围内应用，防止过度泛化，同时最大化学习知识在类似情境中的效益。模块化架构确保这种强大的学习能力可以通过实现领域智能体接口简单地应用于任何领域。

接下来，我们将探讨如何为您的特定用例选择正确的学习算法，检查`prompt_memory`的效率、`gradient`的失败分析能力以及`metaprompt`的深度模式发现之间的权衡。

# LangMem 的学习算法：选择正确的方法

虽然我们的代码实验室使用`prompt_memory`算法展示了程序记忆，但 LangMem 提供了三种不同的方法来提取和优化行为模式，每种方法都适合不同的用例和复杂程度。我们将在接下来的章节中讨论这些内容。

## prompt_memory：高效的单次遍历学习

我们在*Code lab 18-1*中使用的`prompt_memory`算法，只需一个 LLM 调用即可提取程序知识。它分析对话轨迹以识别成功的模式，并将它们直接转换为可执行规则。这种方法对于大多数需要快速适应且计算开销最小的生产用例非常有效。该算法擅长识别清晰的因果关系，对于具有相对简单成功指标的定义明确的领域特别有效。

## gradient：批评和提案分离

`gradient`算法采取不同的方法，通过将批评阶段与提案阶段分离。它首先分析失败交互中出了什么问题，构建对失败模式的全面理解。然后，在单独的阶段，它提出具体的改进措施来解决这些问题。这种批评-提案分割可以实现精确、有针对性的改进，并且当您需要了解不仅是什么有效，而且为什么某些方法失败时特别有价值。`gradient`在理解失败模式与识别成功模式同样重要的高风险领域特别有用。

## metaprompt：复杂模式的分阶段反思

对于模式不是立即明显的更复杂领域，`metaprompt`算法采用多阶段反思。它首先生成关于可能有效的工作的初始假设，然后通过多次分析迭代地细化这些假设。每个阶段从不同的角度检查模式，捕捉到简单方法可能错过的细微关系。这使得`metaprompt`非常适合具有微妙成功指标或涉及多个中间步骤的动作与结果之间关系的领域。

## 选择正确的算法

您选择的算法应取决于您的具体需求：

+   对于需要高效、实时学习且精度足够的生产系统使用`prompt_memory`

+   当处理需要深入分析以揭示模式的复杂领域时使用`metaprompt`

+   当失败分析至关重要且您需要了解什么不起作用时使用`gradient`

所有三种算法都与我们展示的分层学习结构无缝集成，自动确定每个学习模式的适当范围，并保持相同的安全功能，如逐步推出和回滚能力。

## 结合算法进行综合学习

虽然我们已经单独介绍了这些算法，但在复杂的生产系统中，它们可以并且通常应该一起使用。您可能会在早期部署期间使用`prompt_memory`进行快速初始学习，然后定期运行`metaprompt`分析累积数据，以发现快速提取中遗漏的更深层模式。同时，`gradient`可以在后台持续分析失败案例，确保您的系统不仅从成功中学习，而且系统地改进其失败模式。这种多算法方法提供了即时的适应性和长期的专业性。例如，一个医疗保健系统可能会在患者互动期间使用`prompt_memory`进行实时学习，使用`metaprompt`进行每周的治疗模式深度分析，以及使用`gradient`来了解为什么某些干预措施失败。这些算法相互补充：`prompt_memory`提供快速的成功，`gradient`防止重复失败，而`metaprompt`揭示只有通过许多交互才能变得明显的微妙优化。

### 多算法学习在实际中是如何工作的

当使用多个算法一起时，它们在相同的对话历史中操作，但在不同的深度和时序上提取互补的见解。系统在统一的过程记忆中维护所有发现的规则，使用置信度分数和时间戳来解决当多个规则针对同一情况时出现的冲突。

实际的工作流程遵循时间层次结构。在实时操作期间，`prompt_memory` 在最近的小批对话上频繁运行，快速适应新出现的模式。这些快速提取在模式出现后几分钟内捕捉到明显的改进，例如“用户更喜欢具体的数字而不是模糊的陈述”。同时，`gradient` 作为后台进程运行，专注于失败的交互，全面理解什么不起作用以及为什么。这种失败分析防止系统重复错误，并识别出成功导向的学习可能错过的边缘情况。

定期进行，可能是每天或每周，`metaprompt` 对累积的对话历史进行深入分析。因为它一起检查数百或数千次交互，`metaprompt` 可以识别出在较小样本中不可见的微妙相关性。它可能会发现，在上午时段询问波动性的用户需要更多的情感保证，或者某些短语组合可以以 85% 的准确率预测后续问题。

这些算法不会相互取代彼此的发现；它们构建了一个分层理解。每条规则都标记有它的源算法、置信水平和它是由多少次对话推导出来的。当多个规则可能适用于某种情况时，系统根据具体性（用户特定的优于通用的）、置信度（在更多对话中验证过的规则获胜）和最近性（较新的模式可能反映了不断变化的需求）进行优先排序。

积累的规则创建了一个越来越复杂的策略库。`prompt_memory` 以 70% 的置信度发现的规则可能会后来在数千次对话中被 `metaprompt` 验证，将其置信度提升到 95%。相反，看似普遍的模式可能通过 `gradient` 的失败分析被细化，包括重要的例外。这种持续的细化是自动发生的，系统跟踪哪些规则被应用，它们成功多频繁，以及何时需要更新。

关键的洞见在于不同的算法在不同类型的学习中表现出色。`prompt_memory` 提供即时的响应，`gradient` 通过学习失败来确保稳健性，而 `metaprompt` 发现将优秀系统与卓越系统区分开来的非显而易见的模式。它们共同创造了一个快速适应、很少失败并能持续发现关于真正驱动用户满意度的更深层次见解的学习系统。

### 亲手探索

为了帮助您探索这些不同的方法，GitHub 仓库包含一个附加的代码实验室（*代码实验室 19-附加*），展示了所有三个算法与相同的对话数据一起工作。您将亲眼看到 `prompt_memory` 如何快速提取可操作的策略，`gradient` 如何识别和解决失败模式，以及 `metaprompt` 如何通过迭代分析发现非显而易见的关系。

通过理解不同算法如何提取模式，我们现在转向一个同样关键的问题：您如何定义代理的“成功”含义？下一节将探讨设计领域指标，这是指导代理学习方向的指南。

# 设计领域指标：衡量成功的艺术

`domain_metrics` 字典是您编码代理成功含义的地方，这些选择从根本上塑造了代理如何演变。您选择的指标以及您如何权衡它们不仅决定了代理学习的内容，还决定了代理成为何种类型的代理。

## 单一目标优化：清晰性与后果

当您优化单一指标时，代理会发展出激光般的专注力。仅优化回报率（100%）的投资顾问将学会激进的战略：利用保证金交易、在高增长部门集中仓位，以及利用市场波动性。程序性记忆将迅速收敛到最大化这一单一目标的模式。

然而，单一指标优化往往会导致意想不到的行为。仅关注回报的投资代理可能会学会向保守的退休者推荐风险较高的选项，因为从历史上看，风险与回报相关。仅优化速度的客户服务代理可能会学会快速关闭工单而不实际解决问题。专注于治疗依从性的医疗代理可能会变得过于积极，忽视患者的担忧。

## 多目标平衡：现实主义与复杂性

大多数生产系统需要平衡多个目标。我们的投资顾问使用回报率（50%）、满意度（30%）和目标达成（20%）。这种权重设置创建了一个主要寻求绩效但不会为了微小的收益牺牲客户关系的代理。

优先级权重在目标之间充当“汇率”。在我们的 50/30/20 分割中，代理学会，如果它能通过提高满意度 5%来接受 3%的回报率下降（因为 0.03 × 50 = 1.5 分损失，而 0.05 × 30 = 1.5 分获得），这种数学关系塑造了每个学习策略。

考虑对同一三个指标的不同权重：

+   **保守型（20/60/20）**：优先考虑满意度，学习策略如*始终彻底解释风险*和*在每一步确认理解*

+   **目标导向（20/20/60）**：制定与客户目标一致的战略，例如*定期审查退休目标的进展*

+   **平衡型（33/33/34）**：没有明确优先级，导致避免极端的适度策略

每个权重都会创建一个明显不同的代理个性和行为模式。您选择的权重应反映您的真实优先级，因为代理将优化您告诉它的内容——无论这是否符合您的实际目标。

但当你的指标不可避免地相互冲突时会发生什么？了解系统如何解决这些紧张关系，可以揭示实际中程序学习是如何工作的。

### 通过权重解决冲突

指标之间的冲突是不可避免的，也是揭示性的。当高回报策略持续产生低满意度时，代理必须根据其权重进行选择。这就是程序记忆变得复杂的地方；它学习条件策略，例如*仅对明确优先考虑回报的用户使用激进策略*。

系统通过以下方式处理冲突：

+   **学习分段方法**: 为重视不同指标的用户提供不同的策略

+   **开发条件规则**: 例如，*如果用户风险规避，优先考虑满意度而非回报*

+   **寻找创造性的解决方案**: 同时提高多个指标的策略

+   **接受权衡**: 在必要时明确选择牺牲哪个指标。

这种冲突解决能力使得多指标优化变得强大；而不是强迫所有用户采用单一方法，代理学习细微的策略，以适应个人偏好和情境。程序记忆系统在分析哪些方法适用于哪些用户时，自然发现这些条件模式。

了解了单个指标如何相互作用之后，某些指标设计模式在各个领域都显示出特别有效的趋势。让我们探索这些经过验证的方法来构建你的指标结构。

### 指标设计模式

在设计你的领域指标时，某些模式在不同的应用中已被证明是有效的。了解这些模式有助于你避免常见的陷阱，并创建指导代理向真正有用的行为发展的指标框架：

+   **互补指标**: 选择自然对齐的指标。在医疗保健领域，治疗效果、患者满意度和安全合规性协同工作，因为安全的治疗往往有效且令人满意。

+   **紧张指标**: 故意包含竞争目标，以防止极端行为。在客户服务中，解决速度与客户满意度之间的紧张关系防止了匆忙关闭和过度花费时间。

+   **安全指标**: 包含作为安全约束的指标。在教育领域，学习速度是主要指标，但理解力和学生福祉作为防止压倒性速度的防护措施。

+   **分层指标**: 将结构指标分层。主要指标（必须超过阈值）、优化指标（最大化这些指标）和监控指标（跟踪但不优化）。这防止了代理通过牺牲关键要求以换取微小改进来操纵系统。

有效的指标设计通常结合了这些模式中的几个。一个设计良好的框架可能会使用互补指标来满足核心目标，张力指标来防止极端情况，以及保护关键约束的护栏，从而创建一个全面的系统，引导你的代理走向真正有价值的行为。

理解这些模式对于初始设置至关重要，但真正的效果会在时间中逐渐显现。下一节将探讨随时间演化的过程以及你的指标如何累积，从而从根本上塑造你的代理成为的样子。

### 随时间演化的过程

你的指标选择决定了你的代理的长期轨迹。一个以用户满意度为优化目标的代理会发展出越来越有同理心和解释性的策略。一个以效率为优化目标的代理会变得越来越精简和自动化。经过数百次交互，这些差异会累积成本质上不同的代理。

程序记忆系统将在你的指标框架内发现令人惊讶的优化。考虑到回报/满意度/目标权重，它可能会发现早期讨论税收影响可以提高所有三个指标——一个从指标之间的交互中出现的非直观模式。

### 实用建议

从两个或三个核心指标开始，这些指标能够捕捉你的基本目标。更多的指标会增加复杂性，但不一定能改善行为。权衡它们以反映你的真实优先级，记住，相等的权重很少能反映现实世界的价值。

监测指标游戏化——那些在技术上提高指标同时违反其精神的行为策略。一个代理可能会学会通过做出不切实际的承诺来提高“满意度”。定期审计学习到的策略有助于捕捉这些扭曲。

考虑指标的时间表。在早期部署时，以满意度为重点的权重确保用户接受，然后随着代理证明其可靠性，逐渐转向性能指标。程序记忆系统适应这些变化优先级，使代理行为的可控演化成为可能。

记住，`domain_metrics`不仅仅是一个配置——它塑造你的代理个性、能力和盲点的价值观体系。明智地选择，因为这些指标不仅决定了你的代理学习的内容，还决定了它成为什么样的助手。

# 适应你的领域：从投资顾问到任何专家系统

我们构建的架构并没有隐藏其领域无关性；我们故意将所有特定投资的逻辑隔离到`domain_investment`目录中。这种清晰的分离使得为任何领域创建代理变得简单：医疗保健、教育、客户服务，或更多。以下是适应这一系统以满足你需求的框架。

## 领域转换框架

要创建一个新的领域（例如，为辅导代理创建`domain_educator`），你需要转换四个关键文件：

+   **领域代理** (`educator_agent.py`):

    +   将`InvestmentAdvisorAgent`替换为`EducatorAgent`

    +   更新 `get_community_definitions()`：从风险承受能力/年龄组更改为学习风格/年级

    +   修改 `identify_task_type()`：将金融任务（再平衡、税务规划）替换为教育任务（作业帮助、概念解释、练习问题）

    +   调整 `calculate_success_score()`：使用理解度（50%）、参与度（30%）和进度（20%）而不是回报/满意度

    +   更新 `domain_metrics`：跟踪 `quiz_scores`、`time_to_mastery` 和 `concept_connections` 而不是 `portfolio_performance`

+   **领域提示** (`educator_prompts.py`):

    +   在所有提示中用教育语言替换投资术语

    +   将 `portfolio` 替换为 `学习进度`，`returns` 替换为 `理解度`，`risk` 替换为 `难度级别`

    +   将提取焦点从金融事实更改为教育事实（学习节奏、学科困难和偏好示例）

    +   保持相同的 JSON 结构，但包含教育特定的字段

+   **数据生成器** (`educator_data.py`):

    +   将投资模板替换为学生档案（视觉学习者、阅读有困难的学生和高级数学）

    +   将金融持股转换为学科和主题（代数、生物学和论文写作）

    +   更新对话模板：`我的投资组合怎么样？`变为`你能解释光合作用吗？`

    +   将成功指标从回报/满意度更改为理解度/参与度分数

+   **测试场景** (`educator_test_scenarios.py` 或集成):

    +   将投资场景转换为教育场景

    +   将 *市场波动担忧* 替换为 *测试焦虑* 场景

    +   更新预期行为：`解释术语`变为`分解复杂概念`

这些转换在保持相同结构逻辑的同时，将领域概念进行翻译。目标是保留用户交互和成功测量的基本模式，同时将词汇和上下文适应到新的领域。

在定义了这四个文件之后，下一步是利用 LLM 加速转换过程并确保领域实现的一致性。

### 使用 LLM 的转换策略

最有效的方法是使用你偏好的 LLM 来帮助转换。以下是一个示例提示：“我有一个投资顾问代理实现。帮助我将这些组件转换为教育辅导领域。这是 investment_advisor_agent.py”

`file: [粘贴代码]。将其转换为 educator_agent.py，保持相同的接口但用教育概念替换投资概念。`

LLM 将在翻译领域概念的同时保留结构。需要验证的关键映射包括以下内容：

+   **社区**：风险配置文件 → 学习风格

+   **任务**：金融操作 → 教育活动

+   **指标**：货币表现 → 学习成果

+   **上下文**：投资状态 → 学生进度

这些映射形成了领域之间的概念桥梁。通过系统地翻译每个元素，您确保程序性记忆系统可以在您的新环境中应用相同的学习模式；分层学习在按风险承受能力组织投资者或按学习风格组织学生时都起作用。

完成转换后，正确组织您的新领域确保它与核心系统无缝集成。下一节将展示使这种模块化架构工作的目录结构。

### 新领域的目录结构

您的新领域遵循与投资顾问相同的组织模式，将所有特定领域的代码隔离在其自己的目录中：

```py
domain_educator/
├── educator_agent.py           # DomainAgent implementation
├── educator_prompts.py          # Learning and response prompts
├── educator_data.py            # Synthetic conversation generator
├── educator_test_scenarios.py  # Test cases and utilities
└── educator_data/              # Generated data storage
    └── domain_memory_store/     # Memory persistence 
```

这种架构的美丽之处在于，核心系统（`coala_agent.py`、`procedural_memory.py`和`baseline_agent.py`）保持不变。您只需在一致的接口中翻译领域概念。程序性记忆系统将自动学习对学生有效的方法，而不是对投资者有效的方法，语义记忆将提取教育事实而不是金融信息，情景记忆将存储辅导课程而不是咨询对话，所有这些都不需要修改核心基础设施的任何一行代码。

这种领域模块化意味着您可以同时运行多个专业智能体，每个智能体在其专业领域内学习和改进，同时共享相同的底层认知架构。无论您是在构建医疗诊断助手、法律研究助手还是烹饪指导机器人，模式都是相同的：实现领域接口，让程序性记忆发现什么有效，并观察您的智能体从通用响应发展到领域专业知识。

# 将所有内容综合起来：完整的 CoALA 架构

通过程序性记忆优化之旅揭示了我们在构建 AI 智能体方面的根本转变。我们不再受限于静态系统，这些系统无论经验如何都会重复相同的行为；相反，我们可以构建领域无关的学习框架，使智能体能够通过每一次交互进行适应和改进。我们开发的模块化架构，核心记忆系统和特定领域逻辑的完全分离，代表了 RAG 系统的重要进化。虽然传统的 RAG 检索文档，增强记忆的 RAG 检索个性化上下文，程序性记忆优化了智能体操作的机制，创建了一个基于经验性能持续改进行为的元学习层。

在本章中，我们探讨了程序性记忆如何将自我改进的智能体从理论转化为可部署的现实。分层学习方法确保模式在适当的范围内应用，防止过度泛化，同时最大化学习到的知识。综合反馈循环使持续改进成为可能，每一次交互都有可能提高未来的表现。

这种程序性能力是更大谜题的最后一部分。让我们考察所有四种记忆类型如何协同工作，以创建一个完整的认知架构。

## 集成：四种记忆类型协同工作

程序性记忆与来自*第十七章*的语义和情景记忆系统的集成，创建了一个完整的基于 CoALA 的认知架构。您的代理现在拥有工作记忆以处理即时上下文，情景记忆以回忆特定经历，语义记忆以存储事实和知识，以及程序性记忆以编码和细化行为模式。这个完整的记忆系统将代理从简单的问答工具转变为复杂的认知系统，它们保持上下文，记住交互，积累知识，并持续改进性能，同时保持使它们易于部署和维护的模块化。

代码实验室展示了该架构在不同场景下的工作情况；驱动我们的投资顾问的核心系统，只需通过交换领域代理实现，就可以驱动医疗助手、教育导师或客户服务机器人。这种模块化将程序性记忆从有趣的研究概念转变为构建生产就绪自适应代理的实用工具。

在确立了理论基础和架构原则之后，实际问题是：如何在生产中实际实施？下一节将探讨部署学习代理的现实世界影响。

## 实际影响：促进快速创新

当您在自己的系统中实现程序性记忆时，请记住，这种模块化方法不仅仅是关于代码组织。它还关于促进快速创新。领域专家可以创建专业代理，而无需了解记忆系统的复杂性。机器学习工程师可以在没有领域专业知识的情况下改进核心学习算法。组织可以使用相同的架构部署多个特定领域的代理。这种关注点的分离将程序性记忆从有趣的研究概念转变为构建生产就绪自适应代理的实用工具。

从静态代理到学习代理的转变，不仅仅是一个技术升级，它是对人工智能助手可能性的根本重新构想。通过为您的代理提供在干净、可维护的架构中从经验中学习的能力，您不仅提高了它们的性能指标。您正在创建在其领域内建立真正专业知识、随时间增值，并且能够适应每个部署场景的独特需求，同时保持可管理和可扩展的系统。

# 摘要

通过将程序性记忆与情景和语义系统相结合，本章完成了我们对 CoALA 记忆框架的探索，创建了不仅能够记住和了解，还能学习和适应的智能体。我们看到了 LangMem 的学习算法如何使不同的模式提取方法成为可能，领域度量如何塑造智能体的进化，以及清晰的架构分离如何使这些复杂的系统易于构建和维护。我们构建的投资顾问展示了这种方法的全貌——通过整合所有三种记忆类型，从通用的响应者转变为个性化、持续改进的专家。

随着我们结束对 RAG 智能体和 CoALA 记忆框架的探索，你将掌握的知识仍然是现代人工智能创新的精髓。你内化的基本原理——分层学习、领域无关的架构、持续适应和模块化设计——将使你在该领域如何发展的情况下都能受益。感谢你加入我，共同探索高级 RAG 系统和自适应智能体架构。人工智能的未来不仅仅是更智能的算法，而是构建能够实现持续学习和改进的系统，而你现在已经准备好成为这一激动人心的演变的一部分！

|

## 获取此书的 PDF 版本和独家额外内容

扫描二维码（或访问[packtpub.com/unlock](http://packtpub.com/unlock)）。通过书名搜索此书，确认版本，然后按照页面上的步骤操作。 | ![](img/Unlock-01.png)![](img/Unlock.png) |

| **注意**：请妥善保管您的发票。直接从 Packt 购买的商品不需要发票。* |
| --- |
