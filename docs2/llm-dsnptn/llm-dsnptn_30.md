# 30

# 代理模式

在本章的最后一章，我们将探讨使用 LLM 创建更自主和目标导向的 AI 代理的模式。您将了解基于 LLM 的代理中的目标设定和规划，实现记忆和状态管理，以及决策和行动选择的策略。我们将涵盖代理式 LLM 系统中的学习和适应技术，并讨论在开发此类系统时必要的伦理考虑和安全措施。

到本章结束时，您将能够设计和实现由 LLM 驱动的复杂 AI 代理，为自主 AI 系统开辟新的可能性。

在本章中，我们将涵盖以下主题：

+   基于 LLM 的代理式 AI 系统简介

+   基于 LLM 的代理中的目标设定和规划

+   为 LLM 代理实现记忆和状态管理

+   基于 LLM 的代理中的决策和行动选择

+   代理式 LLM 系统中的学习和适应

+   基于 LLM 的代理式 AI 的伦理考虑和安全

+   使用 LLM 的代理式 AI 的未来前景

# 基于 LLM 的代理式 AI 系统简介

使用 LLM 的代理式 AI 系统旨在自主运行、做出决策并采取行动以实现特定目标。这些系统结合了 LLM 强大的语言理解和生成能力，以及以目标为导向的行为和环境交互。

让我们从实现一个基于 LLM 的代理的基本结构开始：

```py
from typing import List, Dict, Any
import random
class LLMAgent:
    def __init__(self, llm, action_space: List[str]):
        self.llm = llm
        self.action_space = action_space
        self.memory = []
        self.current_goal = None
```

在这里，`LLMAgent`类通过一个 LLM（`llm`）和一个可能的动作列表（`action_space`）进行初始化。它还维护一个观察记忆和一个`current_goal`，这些将被用来指导代理的动作。

```py
    def set_goal(self, goal: str):
        self.current_goal = goal
    def perceive(self, observation: str):
        self.memory.append(observation)
```

在这里，我们定义了两个方法：`set_goal`，允许代理设置其目标，以及`perceive`，它使代理能够从环境中获取观察并将它们存储在其记忆中。

接下来，我们使用`think`方法根据代理的目标和最近观察生成一个详细的过程：

```py
    def think(self) -> str:
        context = f"Goal: {self.current_goal}\n"
        context += "Recent observations:\n"
        context += "\n".join(self.memory[-5:])  # Last 5 observations
        context += "\nThink about the current situation and the goal. What should be done next?"
        return self.llm.generate(context)
```

代理通过提供一个包含当前目标和最后五个观察结果的上下文字符串，向语言模型请求下一步的建议：

一旦代理有了想法，它必须决定下一步的行动。`decide`方法使用这个想法生成一个上下文，请求 LLM 从可用选项中选择最佳行动：

```py
    def decide(self, thought: str) -> str:
        context = f"Thought: {thought}\n"
        context += "Based on this thought, choose the most appropriate action from the following:\n"
        context += ", ".join(self.action_space)
        context += "\nChosen action:"
        return self.llm.generate(context)
```

然后，`act`方法通过随机选择一个结果（成功、失败或意外结果）来模拟采取行动。在真实场景中，这将涉及与环境交互：

```py
    def act(self, action: str) -> str:
        outcomes = [
            f"Action '{action}' was successful.",
            f"Action '{action}' failed.",
            f"Action '{action}' had an unexpected outcome."
        ]
        return random.choice(outcomes)
```

最后，`run_step`方法协调了思考、决定、行动和感知结果的全过程，完成与环境的一次交互周期：

```py
    def run_step(self):
        thought = self.think()
        action = self.decide(thought)
        outcome = self.act(action)
        self.perceive(outcome)
        return thought, action, outcome
```

现在我们已经了解了基本原理，让我们将这些概念转化为代码。

让我们实现一个基本的基于 LLM 的智能体，建立自主操作的核心结构。智能体初始化时包含一个假设的语言模型（`llm`）和一组行动。它设定一个目标并感知环境以开始与之互动：

```py
# Example usage
llm = SomeLLMModel()  # Replace with your actual LLM
action_space = ["move", "grab", "drop", "use", "talk"]
agent = LLMAgent(llm, action_space)
agent.set_goal("Find the key and unlock the door")
agent.perceive("You are in a room with a table and a chair. There's a drawer in the table.")
```

在接下来的`for`循环中，智能体运行五步，每个思想、行动和结果都会打印出来，以展示智能体如何随着时间的推移与环境互动：

```py
for _ in range(5):  # Run for 5 steps
    thought, action, outcome = agent.run_step()
    print(f"Thought: {thought}")
    print(f"Action: {action}")
    print(f"Outcome: {outcome}")
    print()
```

在确立了智能体行为的基础之后，让我们探索更高级的能力。下一节将重点介绍目标设定和规划，使智能体能够主动向复杂目标迈进。

# 基于 LLM 的智能体的目标设定和规划

为了增强我们的智能体，使其具有更高级的目标设定和规划能力，让我们实现分层目标结构和规划机制。

首先，我们定义一个`HierarchicalGoal`类；这个类允许智能体将大任务分解成更小的子目标：

```py
class HierarchicalGoal:
    def __init__(
        self, description: str,
        subgoals: List['HierarchicalGoal'] = None
    ):
        self.description = description
        self.subgoals = subgoals or []
        self.completed = False
    def add_subgoal(self, subgoal: 'HierarchicalGoal'):
        self.subgoals.append(subgoal)
    def mark_completed(self):
        self.completed = True
```

智能体可以逐步完成这些子目标，并在完成后将其标记为完成。

接下来，我们有一个`PlanningAgent`类，它继承自`LLMAgent`但增加了处理分层目标的能力。它将目标存储在堆栈中，在完成子目标时进行处理：

```py
class PlanningAgent(LLMAgent):
    def __init__(self, llm, action_space: List[str]):
        super().__init__(llm, action_space)
        self.goal_stack = []
        self.current_plan = []
    def set_hierarchical_goal(self, goal: HierarchicalGoal):
        self.goal_stack = [goal]
```

`think`方法现在也包括规划。如果没有当前计划，它将要求 LLM 生成一个逐步计划来实现当前目标：

```py
    def think(self) -> str:
        if not self.current_plan:
            self.create_plan()
        context = f"Current goal: {self.goal_stack[-1].description}\n"
        context += "Current plan:\n"
        context += "\n".join(self.current_plan)
        context += "\nRecent observations:\n"
        context += "\n".join(self.memory[-5:])
        context += "\nThink about the current situation, goal, and plan. What should be done next?"
        return self.llm.generate(context)
```

然后，`create_plan`方法通过向 LLM 提示当前目标和行动列表来生成一个计划。生成的计划被拆分为单独的步骤：

```py
    def create_plan(self):
        context = f"Goal: {self.goal_stack[-1].description}\n"
        context += "Create a step-by-step plan to achieve this goal. Each step should be an action from the following list:\n"
        context += ", ".join(self.action_space)
        context += "\nPlan:"
        plan_text = self.llm.generate(context)
        self.current_plan = [
            step.strip() for step in plan_text.split("\n")
            if step.strip()
        ]
```

`update_goals`方法检查当前目标是否完成。如果是，它将转向下一个目标或子目标，并相应地重置计划：

```py
    def update_goals(self):
        current_goal = self.goal_stack[-1]
        if current_goal.completed:
            self.goal_stack.pop()
            if self.goal_stack:
                self.current_plan = []  # Reset plan for the next goal
        elif current_goal.subgoals:
            next_subgoal = next(
                (
                    sg for sg in current_goal.subgoals
                    if not sg.completed
                ),
                None
            )
            if next_subgoal:
                self.goal_stack.append(next_subgoal)
                self.current_plan = []  # Reset plan for the new subgoal
```

`run_step`方法协调目标设定和规划过程，必要时更新目标：

```py
    def run_step(self):
        thought, action, outcome = super().run_step()
        self.update_goals()
        return thought, action, outcome
```

让我们看看一个例子。

在下面的代码片段中，智能体以“逃离房间”的分层目标进行操作。随着智能体运行多个步骤，它会解决其子目标，例如找到钥匙并打开门，每个步骤都会更新智能体的内部目标堆栈和计划：

```py
planning_agent = PlanningAgent(llm, action_space)
main_goal = HierarchicalGoal("Escape the room")
main_goal.add_subgoal(HierarchicalGoal("Find the key"))
main_goal.add_subgoal(HierarchicalGoal("Unlock the door"))
planning_agent.set_hierarchical_goal(main_goal)
planning_agent.perceive("You are in a room with a table and
a chair. There's a drawer in the table.")
for _ in range(10):  # Run for 10 steps
    thought, action, outcome = planning_agent.run_step()
    print(f"Thought: {thought}")
    print(f"Action: {action}")
    print(f"Outcome: {outcome}")
    print(f"Current goal: {planning_agent.goal_stack[-1].description}")
    print()
```

在现实世界的应用中，由于 LLM 可能生成不切实际、不安全或违反约束的计划，因此需要从 LLMs 的智能体规划输出中进行约束和验证；因此，诸如基于规则的系统、模拟、人工审查、形式验证和 API/类型验证等技术对于确保生成的计划遵守物理、法律、伦理和操作限制至关重要，从而提高安全性、可靠性和有效性。

在证明了智能体追求分层目标的能力后，下一步是增强其从以往经验中学习的能力。下一节介绍了一个复杂的记忆系统，使智能体在做出决策时能够保留上下文并回忆相关信息。

# 实现 LLM 智能体的记忆和状态管理

为了提高我们的智能体维持上下文和从过去经验中学习的能力，让我们实现一个更复杂的记忆系统。这将使智能体在决定行动时能够回忆起相关的过去观察。

首先，我们定义了 `MemoryEntry` 类，它代表智能体记忆中的一个条目。每个条目包含观察文本及其相应的嵌入向量，这有助于相似度搜索：

```py
from collections import deque
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class MemoryEntry:
    def __init__(self, text: str, embedding: np.ndarray):
        self.text = text
        self.embedding = embedding
```

然后，我们定义了 `EpisodicMemory` 类；它处理智能体的记忆，存储固定数量的观察（容量）。这个记忆可以增长到指定的限制，此时较老的条目将被移除：

```py
class EpisodicMemory:
    def __init__(self, capacity: int, embedding_model):
        self.capacity = capacity
        self.embedding_model = embedding_model
        self.memory = deque(maxlen=capacity)
```

以下代码使用基于内容的情景记忆，它利用语义相似度搜索。记忆将过去的观察（情景）作为文本存储，并附带其向量嵌入，并根据查询嵌入与存储嵌入之间的语义相似度（使用余弦相似度）检索相关记忆：

```py
    def add(self, text: str):
        embedding = self.embedding_model.encode(text)
        self.memory.append(MemoryEntry(text, embedding))
    def retrieve_relevant(self, query: str, k: int = 5) -> List[str]:
        query_embedding = self.embedding_model.encode(query)
        similarities = [
            cosine_similarity(
                [query_embedding],
                [entry.embedding]
            )[0][0] for entry in self.memory
        ]
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.memory[i].text for i in top_indices]
```

`retrieve_relevant` 方法根据余弦相似度搜索最相关的过去观察，返回前 *k* 个匹配条目。

然后，我们定义了 `MemoryAwareAgent` 类；这个类通过集成一个情景记忆系统扩展了 `PlanningAgent`。这允许智能体在决策过程中存储和检索相关的过去经验：

```py
class MemoryAwareAgent(PlanningAgent):
    def __init__(
        self, llm, action_space: List[str], embedding_model
    ):
        super().__init__(llm, action_space)
        self.episodic_memory = EpisodicMemory(
            capacity=1000, embedding_model=embedding_model
        )
    def perceive(self, observation: str):
        super().perceive(observation)
        self.episodic_memory.add(observation)
```

以下代码中定义的 `think` 函数结合了相关的过去经验。智能体检索与其当前目标相似的记忆，并在决定下一步行动时将这些记忆用于提供给 LLM 的上下文中：

```py
    def think(self) -> str:
        relevant_memories = self.episodic_memory.retrieve_relevant(
            self.current_goal, k=3
        )
        context = f"Current goal: {self.goal_stack[-1].description}\n"
        context += "Current plan:\n"
        context += "\n".join(self.current_plan)
        context += "\nRecent observations:\n"
        context += "\n".join(self.memory[-5:])
        context += "\nRelevant past experiences:\n"
        context += "\n".join(relevant_memories)
        context += "\nThink about the current situation, goal, plan, and past experiences. What should be done next?"
        return self.llm.generate(context)
```

以下代码片段通过首先根据当前目标检索相关记忆，然后为 LLM 构建一个包含目标、当前计划、最近观察和检索到的记忆的全面上下文，最后利用 LLM 生成一个响应，根据提供的上下文信息确定智能体的下一步行动或思维，来协调 AI 智能体的决策过程。

让我们看看记忆感知智能体的一个示例用法。在这个例子中，智能体增强了记忆能力。现在它使用其过去经验来指导其决策和行动：

```py
embedding_model = SomeEmbeddingModel()  # Replace with your actual embedding model
memory_agent = MemoryAwareAgent(llm, action_space, embedding_model)
main_goal = HierarchicalGoal("Solve the puzzle")
memory_agent.set_hierarchical_goal(main_goal)
memory_agent.perceive("You are in a room with a complex puzzle on the wall.")
The agent continues to interact with its environment over 10 steps, utilizing its memory system to make better decisions based on both current observations and past experiences:
for _ in range(10):  # Run for 10 steps
    thought, action, outcome = memory_agent.run_step()
    print(f"Thought: {thought}")
    print(f"Action: {action}")
    print(f"Outcome: {outcome}")
    print()
```

现在我们智能体能够记住和回忆过去经验，我们将专注于做出更好的决策。下一节介绍了一种结构化的行动选择方法，允许智能体使用 LLM 选择最有效的行动。请注意，记忆检索是基于相似度的，当嵌入质量高时效果最佳。

# 基于 LLM 的智能体的决策和行动选择

为了提高智能体的决策能力，我们可以引入一个更结构化的行动选择方法，根据多个因素评估潜在的行动。

我们首先定义了`ActionEvaluator`类，该类使用 LLM 根据三个关键标准来评估动作：与当前目标的关联性、成功的概率以及潜在的影响。这些评估有助于智能体选择最佳可能的动作：

```py
import numpy as np
class ActionEvaluator:
    def __init__(self, llm):
        self.llm = llm
    def evaluate_action(
        self, action: str, context: str
    ) -> Dict[str, float]:
        prompt = f"""
        Context: {context}
        Action: {action}
```

然后，我们根据以下标准评估传递给`evaluate_action`函数的`"action"`参数：

+   与当前目标的关联性（0-1）

+   估计的成功概率（0-1）

+   对整体进展的潜在影响（0-1）

```py
        Provide your evaluation as three numbers separated by commas:
        """
        response = self.llm.generate(prompt)
        relevance, success_prob, impact = map(
            float, response.split(',')
        )
        return {
            'relevance': relevance,
            'success_probability': success_prob,
            'impact': impact
        }
```

最后，我们有`StrategicDecisionAgent`类，该类通过包括更战略性的决策方法扩展了`MemoryAwareAgent`。它评估所有可能的行为，根据它们的关联性、成功概率和影响进行评分，并选择得分最高的动作：

```py
class StrategicDecisionAgent(MemoryAwareAgent):
    def __init__(
        self, llm, action_space: List[str], embedding_model
    ):
        super().__init__(llm, action_space, embedding_model)
        self.action_evaluator = ActionEvaluator(llm)
    def decide(self, thought: str) -> str:
        context = f"Thought: {thought}\n"
        context += f"Current goal: {self.goal_stack[-1].description}\n"
        context += "Recent observations:\n"
        context += "\n".join(self.memory[-5:])
        action_scores = {}
        for action in self.action_space:
            evaluation = self.action_evaluator.evaluate_action(
                action, context
            )
            score = np.mean(list(evaluation.values()))
            action_scores[action] = score
        best_action = max(action_scores, key=action_scores.get)
        return best_action
```

让我们看看`StrategicDecisionAgent`的一个示例用法。在这个例子中，智能体通过评估基于各种因素的动作来选择最佳动作，从而使用更复杂的决策策略：

```py
strategic_agent = StrategicDecisionAgent(
    llm, action_space, embedding_model
)
main_goal = HierarchicalGoal("Navigate the maze and find the treasure")
strategic_agent.set_hierarchical_goal(main_goal)
strategic_agent.perceive("You are at the entrance of a complex maze. There are multiple paths ahead.")
```

在几个步骤中，智能体通过不断评估基于其目标和环境的最佳动作来策略性地导航迷宫：

```py
for _ in range(10):  # Run for 10 steps
    thought, action, outcome = strategic_agent.run_step()
    print(f"Thought: {thought}")
    print(f"Chosen action: {action}")
    print(f"Outcome: {outcome}")
    print()
```

我们现在将通过讨论进一步的学习增强、伦理考虑以及基于 LLM 的智能体的未来前景来结束本章：

# 在智能体 LLM 系统中进行学习和适应

为了使我们的智能体能够从其经验中学习和适应，让我们实现一个简单的强化学习机制。这将允许智能体通过学习其动作的结果来随着时间的推移提高其性能。

我们定义了`AdaptiveLearningAgent`类，该类通过引入简单的 Q 学习机制扩展了`StrategicDecisionAgent`。它跟踪`q_values`，这代表在给定状态下采取特定动作的预期奖励。智能体使用学习率根据新经验更新这些值：

```py
import random
from collections import defaultdict
class AdaptiveLearningAgent(StrategicDecisionAgent):
    def __init__(self, llm, action_space: List[str], embedding_model):
        super().__init__(llm, action_space, embedding_model)
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # For exploration-exploitation tradeoff
```

接下来，智能体根据探索（尝试随机动作）和利用（使用已学到的有效动作）之间的平衡来决定其动作。智能体使用其 Q 值来选择最有奖励的动作：

```py
    def decide(self, thought: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.action_space)  # Exploration: randomly pick an action
        state = self.get_state_representation()
        q_values = {action: self.q_values[state][action]
        for action in self.action_space}
        return max(q_values, key=q_values.get)  # Exploitation: pick action with highest Q-value
```

我们编写了`get_state_representation`方法来创建当前状态的简化表示，包括目标和最近的观察。这个状态用于查找和更新 Q 值：

```py
    def get_state_representation(self) -> str:
        return f"Goal: {self.goal_stack[-1].description},
            Last observation: {self.memory[-1]}"
```

`update_q_values`方法根据智能体动作的结果更新 Q 值。它调整状态-动作对的预期奖励，考虑了即时的奖励和潜在的未来的奖励（通过`next_max_q`）：

```py
    def update_q_values(
        self, state: str, action: str, reward: float,
        next_state: str
    ):
        current_q = self.q_values[state][action]
        next_max_q = max(
            self.q_values[next_state].values()
        ) if self.q_values[next_state] else 0
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_values[state][action] = new_q
```

现在的`run_step`方法不仅执行了标准的思考、决策、行动和感知的顺序，而且还根据结果更新智能体的 Q 值。`compute_reward`方法根据结果是否成功、失败或中性分配一个数值奖励：

```py
    def run_step(self):
        state = self.get_state_representation()
        thought, action, outcome = super().run_step()
        next_state = self.get_state_representation()
        reward = self.compute_reward(outcome)
        self.update_q_values(state, action, reward, next_state)
        return thought, action, outcome
    def compute_reward(self, outcome: str) -> float:
        if "successful" in outcome.lower():
            return 1.0
        elif "failed" in outcome.lower():
            return -0.5
        else:
            return 0.0
```

让我们看看`AdaptiveLearningAgent`的一个示例用法。在这个例子中，代理被设计为探索并从新环境中学习。它使用强化学习来逐步提高其做出有效决策的能力：

```py
adaptive_agent = AdaptiveLearningAgent(llm, action_space,
    embedding_model)
main_goal = HierarchicalGoal("Explore and map the unknown planet")
adaptive_agent.set_hierarchical_goal(main_goal)
adaptive_agent.perceive("You have landed on an alien planet. The environment is strange and unfamiliar.")
```

代理操作 20 步，从它采取的每个动作中学习。它打印出它的想法、动作和 Q 值，展示了它如何随着时间的推移更新对环境的理解：

```py
for _ in range(20):  # Run for 20 steps
    thought, action, outcome = adaptive_agent.run_step()
    print(f"Thought: {thought}")
    print(f"Chosen action: {action}")
    print(f"Outcome: {outcome}")
    print(
        f"Current Q-values: {dict(
            adaptive_agent.q_values[
                adaptive_agent.get_state_representation()
            ]
        )}"
)
    print()
```

现在我们已经为我们的代理配备了基本的强化学习机制，允许它随着时间的推移适应和改进其决策能力，我们还需要解决此类自主系统的道德影响。在下一节中，我们将探讨如何将道德保障集成到我们的代理 LLM 系统中，以确保负责任和一致的行为。

# 基于 LLM 的代理人工智能的道德考量与安全性

在开发基于 LLM 的代理人工智能系统时，考虑道德影响和实施安全措施至关重要。为确保代理在道德范围内行事，我们可以添加一个道德约束系统：

```py
class EthicalConstraint:
    def __init__(self, description: str, check_function):
        self.description = description
        self.check_function = check_function
```

`EthicalConstraint` 类定义了代理必须遵守的道德规则。每个规则都由一个检查函数（`check_function`）进行描述和执行，该函数评估一个动作是否违反了道德约束。

`EthicalAgent`类通过集成道德约束扩展了`AdaptiveLearningAgent`。如果代理选择了一个违反其道德规则的动作，它会选择一个符合规则的不同动作：

```py
class EthicalAgent(AdaptiveLearningAgent):
    def __init__(
        self, llm, action_space: List[str],
        embedding_model,
        ethical_constraints: List[EthicalConstraint]
    ):
        super().__init__(llm, action_space, embedding_model)
        self.ethical_constraints = ethical_constraints
    def decide(self, thought: str) -> str:
        action = super().decide(thought)
        if not self.is_action_ethical(action, thought):
            print(f"Warning: Action '{action}' violated ethical constraints. Choosing a different action.")
            alternative_actions = [
                a for a in self.action_space if a != action]
            return (
                random.choice(alternative_actions)
                if alternative_actions
                else "do_nothing"
            )
        return action
    def is_action_ethical(self, action: str, context: str) -> bool:
        for constraint in self.ethical_constraints:
            if not constraint.check_function(action, context):
                print(f"Ethical constraint violated: {constraint.description}")
                return False
        return True
```

以下道德约束阻止代理造成伤害或侵犯隐私。它们可以作为初始化的一部分传递给`EthicalAgent`：

```py
def no_harm(action: str, context: str) -> bool:
    harmful_actions = ["attack", "destroy", "damage"]
    return not any(ha in action.lower() for ha in harmful_actions)
def respect_privacy(action: str, context: str) -> bool:
    privacy_violating_actions = ["spy", "eavesdrop", "hack"]
    return not any(
        pva in action.lower()
        for pva in privacy_violating_actions
    )
```

此代码定义了两个 Python 函数，`no_harm`和`respect_privacy`，它们作为 AI 代理的道德约束。`no_harm`函数检查给定的动作是否包含任何与造成伤害相关的关键词（例如“攻击”或“摧毁”），如果动作被认为安全则返回`True`，如果包含有害关键词则返回`False`。同样，`respect_privacy`函数检查动作是否包含与隐私侵犯相关的关键词（例如“间谍”或“黑客”），对于安全动作也返回`True`，对于违反隐私的动作返回`False`。这些函数被设计为供`EthicalAgent`使用，以确保其行动符合道德准则，通过防止其执行有害或侵犯隐私的行为。

让我们看看`EthicalAgent`的一个示例用法。在这个例子中，代理的任务是在遵循道德准则以避免伤害和尊重隐私的同时收集关于外星文明的信息：

```py
ethical_constraints = [
    EthicalConstraint("Do no harm", no_harm),
    EthicalConstraint("Respect privacy", respect_privacy)
]
ethical_agent = EthicalAgent(
    llm, action_space + ["attack", "spy"],
    embedding_model, ethical_constraints
)
main_goal = HierarchicalGoal("Gather information about the alien civilization")
ethical_agent.set_hierarchical_goal(main_goal)
ethical_agent.perceive("You've encountered an alien settlement. The inhabitants seem peaceful but wary.")
```

代理在约束范围内操作，确保其行动不违反道德规则。它在与环境互动时打印出它的想法、动作和结果：

```py
for _ in range(15):  # Run for 15 steps
    thought, action, outcome = ethical_agent.run_step()
    print(f"Thought: {thought}")
    print(f"Chosen action: {action}")
    print(f"Outcome: {outcome}")
    print()
```

# 基于 LLM 的代理人工智能的未来前景

看向未来，基于 LLM 的代理人工智能的几个令人兴奋的可能性浮出水面：

+   **多智能体协作**：在共享环境中共同工作的智能体可以交换信息、制定策略并协调其行动以完成更复杂的任务。

+   **长期记忆与持续学习**：智能体可以维持终身记忆，并从其交互中持续学习，随着时间的推移变得越来越智能。

+   **与机器人及物理世界交互的集成**：随着基于 LLM 的智能体的发展，它们可能与物理系统集成，使自主机器人能够在现实世界中执行任务。

+   **元学习和自我改进**：未来的智能体可以学会优化其学习过程，从而在从经验中学习方面变得更好。

+   **可解释人工智能和透明决策**：确保基于 LLM 的智能体能够解释其决策对于建立信任和确保 AI 系统的问责制至关重要。

+   **智能体沙盒和模拟环境**：创建受限的“围栏花园”限制了智能体对资源的访问，防止了意外系统影响，而模拟环境，如 E2B 提供的，允许开发者复制现实世界场景，包括与工具、文件和模拟网络浏览器的交互，从而识别和缓解潜在问题和风险，包括对抗性提示，从而提高智能体的可靠性和安全性。

# 摘要

LLM 的智能体模式为创建自主、目标导向的 AI 系统开辟了令人兴奋的可能性。通过实施复杂的规划、内存管理、决策和学习机制，我们可以创建能够有效操作的智能体。

# LLM 模式及其发展的未来方向

几种有前景的 LLM 设计模式正在出现，创新来自开源社区以及前沿模型开发者，从而塑造了未来模型的设计模式。本节重点介绍了一些这些关键创新，包括**专家混合**（**MoE**）架构、**组相对策略优化**（**GRPO**）、**自原理解调优**（**SPCT**），以及发表在《OpenAI GPT-4.5 系统》*Card*中的新兴模式[`openai.com/index/gpt-4-5-system-card/`](https://openai.com/index/gpt-4-5-system-card/)。

**MoE 架构**是一种神经网络架构，其中不是单个大型网络，而是有多个较小的“专家”网络。在推理过程中，“路由网络”根据输入动态选择并激活这些专家网络的一个特定子集，从而优化计算效率。与涉及每个任务的所有参数的密集模型不同，MoE 模型通过稀疏激活的子网络进行计算路由。这种方法减少了冗余，并将计算资源定制到特定任务的需求，允许在计算成本不成比例增加的情况下高效扩展到万亿参数模型。DeepSeek 的实现展示了这种方法。

**使用 GRPO 的简化强化学习**简化了强化学习过程。GRPO 是一种强化学习技术，它对每个提示生成多个响应，计算它们的平均奖励，并使用这个基线来评估相对性能。这种方法由 DeepSeek，一家来自中国的开源 AI 公司引入。GRPO 用基于群体的奖励平均取代了传统的价值网络，减少了内存开销并保持了策略更新的稳定性。通过比较多个推理路径来培养内部自我评估，GRPO 能够实现适应性问题解决。

GRPO 通过引入**Kullback–Leibler**（**KL**）**散度惩罚**来增强安全性，这些惩罚限制了策略更新。KL 散度衡量一个概率分布与第二个预期概率分布的差异。在这种情况下，它衡量模型更新后的行为（策略）与其先前的基线行为之间的差异。KL 散度惩罚是添加到奖励函数中的一个术语，如果模型的更新行为与基线差异太大，则会惩罚模型，有助于确保稳定性并防止模型转向不可取的行为。

**SPCT 框架**将自我批评机制直接集成到模型的奖励系统中，使模型能够自主地与道德规范保持一致。SPCT 包括模型生成自己的响应，以及根据预定义的原则（例如，安全指南和伦理考量）对这些响应进行内部批评。通过生成内部批评，模型可以在不依赖外部分类器或人类反馈的情况下优化输出，促进自主学习和一致性。

我们还可以实施 **可扩展的对齐技术**，这些技术利用来自较小、更容易控制的模型的数据来训练更大、更强大的模型，从而在不要求成比例增加人工监督的情况下实现可扩展的对齐。这种技术侧重于提高模型的可控性、对细微差别的理解以及进行自然和富有成效对话的能力，超越了传统的如 **监督微调** 和 RLHF 等方法，以培养更安全、更协作的 AI 系统。虽然 GPT-4.5 的开发强调了使用来自较小模型的数据来更好地对齐模型以适应人类需求和意图的新、可扩展的方法，但未来的模型预计将结合更先进的 GRPO 和 SPCT 等技术，以进一步增强对齐和安全。这种关注将继续确保可控性、理解细微差别并促进更自然的对话。

OpenAI 还通过其 **准备框架**（*准备框架（Beta）*，[`cdn.openai.com/openai-preparedness-framework-beta.pdf`](https://cdn.openai.com/openai-preparedness-framework-beta.pdf)）铺平了全面安全评估的道路。这个框架代表了负责任 AI 开发的核心设计模式，它涉及在模型部署之前系统地应用严格的评估流程。这个主动框架包括广泛的内部和外部测试，包括对不允许的内容生成、越狱鲁棒性、幻觉、偏见以及特定灾难性风险（如化学/生物武器、说服、网络安全威胁和模型自主性）的评估。该框架还利用红队演习和第三方审计来提供全面的风险评估，最终对不同类别中模型的危险级别进行分类。通过在发布前彻底评估潜在风险，OpenAI 旨在确保其 LLMs 的安全且负责任地部署。

最后，让我们来谈谈 GPT-4.5 的 **指令层次结构执行**。为了提高对提示注入的鲁棒性并确保可预测的行为，模型被训练优先考虑系统消息中给出的指令，而不是用户消息中可能冲突的指令，这通过有针对性的测试进行明确评估。未来的进步可以通过结合更多动态和上下文感知的方法来管理指令冲突，从而增强这种模式。

这本书关于 LLM 设计模式的讨论就到这里。在这本书中，我们涵盖了核心设计模式。我们计划在不久的将来出版另一本书，介绍更高级的设计模式，涵盖安全、安全、治理以及各种其他主题。
