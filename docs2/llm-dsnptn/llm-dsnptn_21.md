# 21

# 思维树（Tree-of-Thoughts）提示

**思维树（Tree-of-thoughts**）(**ToT**)提示是一种技术，旨在通过允许对不同的推理路径进行更结构化的探索来增强 LLMs 的解决问题能力。

正式的 ToT 方法是在 2023 年的一篇名为《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》的研究论文中由姚等人（来自普林斯顿大学、谷歌 DeepMind 和谷歌研究）提出的。也请访问[`arxiv.org/abs/2305.10601`](https://arxiv.org/abs/2305.10601)。

ToT 的主要灵感来源于人类处理复杂问题的方法——我们经常考虑多个可能的解决方案路径，评估它们的可行性，在必要时回溯，并探索替代方案。传统的提示技术，如 CoT（见*第二十章*），允许逐步推理，但缺乏探索多条路径或重新考虑早期步骤的能力。

ToT 建立在几个技术之上：

+   CoT 提示，实现逐步推理

+   生成多个推理路径的自洽方法

+   涉及探索和回溯的人类问题解决方法

ToT 的关键创新是将思维视为一个树搜索问题，在每一步中，模型可以生成和评估多个“思维”（中间推理步骤），然后选择最有希望的路径继续探索。这允许更复杂的解决问题，包括探索、评估和回溯的能力。

在本章中，你将学习如何实现 ToT 提示来处理你的 LLMs 的复杂推理任务。

在本章中，我们将涵盖以下主题：

+   设计 ToT 提示

+   搜索策略

+   剪枝和评估

+   将 ToT 应用于解决多步问题

+   实施中的挑战

+   未来方向

# 设计 ToT 提示

要创建有效的 ToT 提示，你应该做以下事情：

1.  **鼓励分支思维**：这创建了一个非线性的探索过程，其中可以同时考虑多个可能的解决方案路径。通过明确要求模型生成几个不同的初始方法或视角，你可以防止它过早地承诺于一条可能导致次优结果的推理路线。

1.  **提供清晰的问题陈述**：一个明确定义的问题陈述为模型提供了一个具体的目标和约束条件，使其在其中工作。这种清晰度有助于模型确切了解它需要解决的问题，并为生成相关的思维分支提供了基础。没有这个，分支过程可能会变得不集中且效率低下。

1.  **引导模型探索替代路径**：这确保模型不会过早地收敛到一个看似有希望但实际上次优的解决方案。通过明确要求探索不同的方法，你帮助模型克服推理中的潜在偏见，并发现它可能错过的创新解决方案。

1.  **包含评估机制**：这个组件使模型能够评估不同分支的质量，并就进一步追求哪些路径做出明智的决定。没有评估标准，模型将没有系统的方法来确定哪些分支最有希望，可能会在无望的路径上浪费计算资源。

ToT 对于复杂的推理任务特别强大，因为它模仿了人类解决问题的方法，我们在做出解决方案之前通常会在心理上探索多种可能性。显式的分支和评估结构有助于语言模型克服其在顺序推理能力上的局限性。

下面是一个实现基本 ToT 提示的示例：

```py
def tot_prompt(question, num_branches=3):
    prompt = f"""Solve the following problem using a Tree-of-Thoughts approach:
Problem: {question}
Let's explore multiple reasoning paths:
Path 1:
1) First, we could...
2) Then, we might...
3) This leads us to...
Path 2:
1) Alternatively, we could start by...
2) Following this approach...
3) This results in...
Path 3:
1) Another perspective is...
2) If we consider this...
3) The outcome would be...
Now, let's evaluate these paths and determine the most promising solution:
Evaluation:
1) Path 1: ...
2) Path 2: ...
3) Path 3: ...
Based on this evaluation, the most promising solution is...
Therefore, the final answer is...
Now, apply this Tree-of-Thoughts approach to solve the given problem:
{question}
Let's explore multiple reasoning paths:
"""
    return prompt
Let's look at an example usage:
problem = "What is the most efficient way to sort a list of a million integers?"
prompt = tot_prompt(problem)
print(prompt)
```

此函数为给定问题（`"What is the most efficient way to sort a list of a million integers?"`）生成一个 ToT 提示，为探索和评估多个推理路径提供结构。

此代码通过实现四个关键原则创建一个 ToT 提示模板：它通过具有不同起始短语和编号步骤的显式路径结构鼓励分支思维，确保模型探索多个不同的解决方案方法；它通过两次阐述问题来提供清晰度，以建立上下文并在生成解决方案之前重新聚焦注意力；它通过对比语言和独立的推理路径引导探索替代方法；并通过一个专门的比较部分以及选择最有希望解决方案的提示来促进评估。整体结构创建了一个认知支架，通过迫使模型在得出结论之前生成、发展和批判性地比较多个解决方案路径，帮助语言模型克服线性思维倾向——模仿人类通过发散性思维后进行批判性评估来解决复杂问题的方法。

实施有效的搜索策略对于导航 ToT 至关重要。让我们在下一节中检查其中两种策略。

# 搜索策略

我们有两种常用的搜索策略：

+   **深度优先搜索 (DFS)**：这是一种图遍历算法，在回溯之前尽可能沿着每个分支进行探索。在思维树的情况下，DFS 系统性地深入一条路径，在移动到下一条路径之前，完全探索每个思想或分支。它通过从根节点开始，将每个节点的子节点推入栈中，然后递归地首先探索最深的节点来工作。这种方法在你想要全面探索一条推理线或调查最深刻或复杂的思想之前，非常适合在分支出来之前，对于问题解决、决策制定和理解复杂概念景观非常有价值。

+   **广度优先搜索 (BFS)**：与 DFS 相比，BFS 通过系统地检查当前深度的所有相邻节点，然后移动到下一深度级别的节点来探索思维树。使用队列数据结构，BFS 从根节点开始，探索所有直接连接，然后再深入。在思想探索的背景下，BFS 特别适用于你想要获得不同想法及其直接相互连接的广泛全景视图时。这种策略对于理解思想的宽度和多样性、找到概念之间的最短路径，或者在你需要深入任何单个分支之前，同时探索多个潜在的推理路径时非常理想（参见 *图 21.1*）。

![图 21.1 – DFS 与 BFS](img/Image96457.jpg)

图 21.1 – DFS 与 BFS

例如，让我们实现一个简单的 DFS 策略：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
def dfs_tot(model, tokenizer, problem, max_depth=3, max_branches=2):
    def explore_branch(current_thought, depth):
        if depth == max_depth:
            return current_thought
        prompt = f"{current_thought}\n\nLet's explore further:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs, max_length=len(prompt) + 100,
            num_return_sequences=max_branches
        )
        branches = [
            tokenizer.decode(
                output[len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            ) for output in outputs
        ]
        results = []
        for branch in branches:
            results.append(
                explore_branch(
                    current_thought + branch, depth + 1
                )
            )
        return max(
            results, key=lambda x: evaluate_thought(x)
        )  # Select the best branch
    initial_prompt = tot_prompt(problem)
    return explore_branch(initial_prompt, 0)
def evaluate_thought(thought):
    # Implement logic to evaluate the quality of a thought
    # This could involve coherence, relevance, depth of reasoning, etc.
    pass
```

此代码实现了一个 DFS 算法来探索由语言模型生成的 ToT。它从一个初始问题开始，然后使用模型生成多个潜在的后继（分支）。代码递归地探索每个分支，扩展“思想”直到达到最大深度。在每一步，生成的文本被转换为模型输入，模型输出被解码回文本。

`evaluate_thought` 函数是选择过程中的关键部分，旨在评估每个生成思想的品质。代码利用这种评分来决定进一步探索哪些分支，有效地引导 ToT 向可能的最优解导航。最终结果是 DFS 过程中找到的最高评分思想。

这里是一个前面代码片段的示例用法：

```py
model_name = "gpt2-large"  # Replace with your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
problem = "What are the potential long-term effects of artificial intelligence on employment?"
solution = dfs_tot(model, tokenizer, problem)
print(solution)
```

这个代码片段展示了如何使用预训练的 GPT-2 语言模型，通过之前描述的 `dfs_tot` 函数生成给定问题的解决方案。首先，它指定要使用的模型（`"gpt2-large"`），并使用 `transformers` 库中的 `AutoModelForCausalLM` 和 `AutoTokenizer` 加载模型及其相关的标记器。这确保了文本被正确处理以供模型使用。

然后，它将问题定义为关于人工智能对就业的长期影响的疑问。使用加载的模型、分词器和问题作为输入调用`dfs_tot`函数，开始深度优先搜索解决方案。返回的`solution`代表模型在探索各种“思维”后生成的响应，最终打印到控制台。

接下来，我们将讨论在 ToT 框架内进行剪枝和评估以提高效率和集中搜索。剪枝对于管理探索众多思维分支相关的计算成本至关重要，而评估提供了决定哪些分支要丢弃的标准。

# 剪枝与评估

在 ToT 方法中，剪枝是一种通过系统地减少搜索空间来管理认知复杂度的有效机制。这个过程涉及通过智能评估技术选择性地消除不太有希望的思维分支，使用启发式评分方法来评估每条潜在路径导致最优解的可能性。通过动态过滤掉低潜力思维并集中计算资源在最有希望的推理轨迹上，ToT 剪枝能够实现更高效和有针对性的问题解决，平衡探索广度与推理深度。

1.  让我们通过定义一个简单的剪枝函数来实现一个基本的剪枝策略：

    ```py
    def pruning_tot(
        model, tokenizer, problem, max_depth=3,
        max_branches=3, prune_threshold=0.5
    ):
        def explore_and_prune(current_thought, depth):
            if depth == max_depth:
                return current_thought
            prompt = f"{current_thought}\n\nLet's explore further:\n"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs, max_length=len(prompt) + 100,
                num_return_sequences=max_branches
            )
            branches = [
                tokenizer.decode(
                    output[len(inputs['input_ids'][0]):],
                    skip_special_tokens=True
                ) for output in outputs
            ]
    ```

    逻辑的核心在于`explore_and_prune`函数，它处理推理树的递归搜索。代码通过使用 LLM 从当前思维生成多个可能的延续（分支）来工作。该函数旨在探索推理树直到指定的最大深度，每个级别包含受控数量的分支。当达到最大深度时，代码将当前思维作为最终结果返回。剪枝机制是说明性的，不应用于生产。

1.  一旦我们定义了我们的函数，我们就评估和剪枝分支：

    ```py
            evaluated_branches = [
                (branch, evaluate_thought(current_thought + branch))
                for branch in branches
            ]
            pruned_branches = [
                b for b, score in evaluated_branches
                if score > prune_threshold
            ]
            if not pruned_branches:
                return current_thought  # If all branches are pruned, return current thought
            results = []
            for branch in pruned_branches:
                results.append(
                    explore_and_prune(current_thought + branch,
                        depth + 1)
                )
            return max(results, key=lambda x: evaluate_thought(x))
        initial_prompt = tot_prompt(problem)
        return explore_and_prune(initial_prompt, 0)
    ```

    首先，代码通过将每个生成的分支与`evaluate_thought`函数的分数配对来评估每个分支，该函数评估推理路径的质量。然后，它通过仅保留得分高于定义阈值的分支来过滤掉低质量的分支。如果所有分支都被剪枝（没有达到阈值），则算法返回当前思维而不再进一步探索。对于剩余的有希望的分支，代码通过在增加的深度级别上调用相同的函数递归地探索每个分支。最后，它通过从所有探索的路径中返回具有最高评估分数的结果来选择最佳的总体推理路径。外部函数使用包含原始问题声明的格式化提示初始化搜索。

1.  定义一个 `evaluate_thought` 函数。此函数通过根据其复杂性（长度）和语言多样性（使用的独特单词数量）评分来评估给定的思维或推理分支，返回介于 `0` 和 `1` 之间的归一化分数：

    ```py
    def evaluate_thought(branch, threshold=0.5):
        """
        Simple evaluation function for ToT branch assessment
        Args:
            branch (str): The branch/thought to evaluate
            threshold (float): Minimum score for considering a branch viable
        Returns:
            float: Evaluation score
        """
        # Basic heuristics for evaluation
        complexity_score = len(branch.split()) / 20  # Reward moderate complexity
        uniqueness_score = len(
            set(branch.split())) / len(branch.split()
        )  # Reward unique words
        # Combined score, normalized
        score = (complexity_score + uniqueness_score) / 2
        return min(1.0, max(0.0, score))
    ```

1.  让我们看看一个例子：

    ```py
    problem = "What are the ethical implications of genetic engineering in humans?"
    solution = pruning_tot(model, tokenizer, problem)
    print(solution)
    ```

此实现添加了一个修剪步骤，以移除低质量的分支，将搜索集中在最有希望的路径上。

现在，让我们将 ToT 应用于解决一个多步骤问题。

# 将 ToT 应用于解决多步骤问题

ToT 对于复杂的推理任务特别有效。让我们实现一个用于多步骤问题解决的 ToT 方法：

```py
def multi_step_tot(model, tokenizer, problem_steps):
    full_solution = ""
    for step, question in enumerate(problem_steps):
        prompt = f"""Step {step + 1} of the problem:
{question}
Previous steps solution:
{full_solution}
Let's use Tree-of-Thoughts to solve this step:
"""
        step_solution = pruning_tot(model, tokenizer, prompt)
        full_solution += (
            f"\n\nStep {step + 1} Solution:\n"
            f"{step_solution}"
        )
    return full_solution
# Example usage
problem_steps = [
    "What are the main factors contributing to climate change?",
    "How do these factors interact with each other?",
    "What are potential solutions to mitigate climate change?",
    "What are the challenges in implementing these solutions?"
]
solution = multi_step_tot(model, tokenizer, problem_steps)
print(solution)
```

此代码实现了一个使用 ToT 推理方法的多步骤问题求解器。`multi_step_tot` 函数将复杂问题分解成一系列步骤，并逐个解决它们，基于之前的解决方案。

对于提供的每个问题序列步骤，该函数创建一个包含当前问题、之前步骤中累积的解决方案以及使用 ToT 推理的说明的提示。然后，它调用先前定义的 `pruning_tot` 函数来为该特定步骤生成解决方案。每个步骤的解决方案都附加到一个不断增长的 `full_solution` 字符串中，从而创建一个保持整个问题思维连贯性的综合答案。示例演示了如何通过一系列越来越深入的问题来分析气候变化，从识别原因到探索潜在解决方案的实施挑战。

# 实施挑战

虽然 ToT 很强大，但它面临着几个挑战：

+   **计算复杂性**：探索多个路径可能非常昂贵

+   **评估难度**：确定不同思维路径的质量可能具有挑战性

+   **分支间的连贯性**：确保结合不同分支的见解时的一致性

+   **提示设计复杂性**：创建有效的 ToT 提示需要仔细考虑

为了解决计算复杂性，考虑实现并行处理方法。并行处理可以通过解决其固有的计算瓶颈来提高 ToT 推理方法。以下代码实现了同时而不是顺序地并发探索多个推理分支，这可以显著减少复杂问题的总计算时间：

```py
import concurrent.futures
def parallel_tot(model, tokenizer, problem, max_workers=3):
    def explore_branch(branch):
        return pruning_tot(model, tokenizer, branch)
    initial_branches = generate_initial_branches(problem, max_workers)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = [
            executor.submit(explore_branch, branch)
            for branch in initial_branches
        ]
        results = [
            f.result()
            for f in concurrent.futures.as_completed(futures)
        ]
    return max(results, key=lambda x: evaluate_thought(x))
def generate_initial_branches(problem, num_branches):
    # Implement logic to generate initial branches for the problem
    pass
# Example usage
problem = "What are the potential implications of quantum computing on cryptography?"
solution = parallel_tot(model, tokenizer, problem)
print(solution)
```

在前面的代码中，实现使用了 Python 的`concurrent.futures`模块和`ThreadPoolExecutor`来将工作负载分配到多个工作者。每个工作者独立探索推理树的不同的初始分支，有效地并行搜索多个有希望的路径。这种方法对于 ToT 推理特别有价值，因为算法的分支特性创建了众多独立的子问题，可以在没有相互依赖的情况下并行解决各自的中间结果。最后一步通过从所有完成的分支中选择最高质量的解决方案来整合这些并行探索。

此实现使用并行处理同时探索多个分支，可能减少复杂 ToT 问题的计算时间。

# 未来方向

随着 ToT 的持续发展，出现了一些有前景的方向：

+   **动态树结构**：根据问题复杂度调整树结构。

+   **混合 ToT-CoT 方法**：结合两种技术的优势([`arxiv.org/html/2409.17433v1`](https://arxiv.org/html/2409.17433v1))。

+   **ToT 的元学习**：训练 LLMs 自动生成有效的 ToT 结构。这种方法尚未被任何人探索。

+   **整合外部知识**：将特定领域的知识整合到 ToT 推理中([`arxiv.org/html/2407.00653v1`](https://arxiv.org/html/2407.00653v1))。

下面是一个动态 ToT 结构的概念实现：

```py
def dynamic_tot(model, tokenizer, problem, max_depth=5):
    def adapt_structure(current_thought, depth):
        if depth == max_depth:
            return current_thought
        complexity = assess_complexity(current_thought)
        num_branches = determine_branches(complexity)
        branches = generate_branches(
            model, tokenizer, current_thought, num_branches
        )
        results = []
        for branch in branches:
            results.append(
                adapt_structure(
                    current_thought + branch, depth + 1
                )
            )
        return max(results, key=lambda x: evaluate_thought(x))
    def assess_complexity(thought):
        # Implement logic to assess the complexity of the current thought
        pass
    def determine_branches(complexity):
        # Determine the number of branches based on complexity
        return max(2, min(5, int(complexity  10)))
    def generate_branches(model, tokenizer, thought, num_branches):
        # Generate branches using the model
        pass
    initial_prompt = tot_prompt(problem)
    return adapt_structure(initial_prompt, 0)
```

前面的代码实现了一个动态 ToT 方法，该方法根据当前推理路径的复杂度调整其探索策略。核心函数`adapt_structure`通过递归地检查每一步当前思维过程的复杂度，并动态确定要探索的分支数量来构建解决方案。与固定的分支策略不同，这种自适应方法为可能从更广泛探索中受益的复杂推理路径分配更多的计算资源（更多分支），而对于较简单的概念则使用较少的分支。实现包括辅助函数来评估思维复杂度、确定适当的分支数量以及使用语言模型生成新的思维延续。算法在达到最大深度时终止，并返回得分最高的完整推理路径。

下面是一个示例，说明如何使用前面的代码解决诸如“`纳米技术的进步可能会在下一个十年如何影响医学？`”这样的问题：

```py
problem = "How might advancements in nanotechnology impact medicine in the next decade?"
solution = dynamic_tot(model, tokenizer, problem)
print(solution)
```

这种动态 ToT 方法根据评估的每个思维的复杂度调整树结构，允许更灵活和高效地探索复杂问题空间。

# 摘要

在本章中，你学习了如何为 LLM 设计和实现 ToT 提示，包括管理分支思维过程策略。我们涵盖了搜索技术和修剪和评估不同推理路径的方法。通过实施这里讨论的策略和考虑因素，你可以显著提高 LLM 处理模糊、多方面问题的能力，并生成更稳健和有洞察力的解决方案。

回顾*第二十章*，本章专注于 CoT，让我们从用例的角度比较 CoT 和 ToT。当任务涉及线性、顺序推理，并且可以分解为具有单一、主导解决方案路径的中间步骤时，使用 CoT 提示。CoT 在数学文字问题、演绎推理、基本逻辑谜题和逐步程序任务中特别有效。当问题具有低分支复杂度且不需要探索多个替代方案时，它工作得很好。CoT 在计算上更便宜，因为它以前向、确定性的方式产生单一的推理链。当 LLM 需要支架来“大声思考”并使其中间步骤明确以防止幻觉或逻辑错误时，这种技术最有帮助。

当任务涉及具有分支决策点的多步推理时，使用 ToT 提示，特别是当存在多个可能的解决方案路径需要并行评估时。ToT 适合创造性问题解决、规划任务、定理证明、代码合成以及在不确定性下的决策。当问题空间可以结构化为搜索树，其中中间推理节点可以重新访问、评估和比较时，它变得有利。ToT 通常包含诸如自洽采样、前瞻性评估和基于价值的分支选择等策略。由于它并行维护和扩展多个推理路径，包括回滚、回溯或节点评分，因此它在计算上更密集。

如果问题是受限制且形式良好的（例如，SAT 风格的问题或直接的推导），CoT 通常足够且更高效。如果问题是开放式的，具有多个冲突的目标，或者最优解需要比较替代路径（如在规划路线、游戏移动或形式证明中），ToT 通过模拟探索和深思熟虑，可以提供更好的性能。

在实践中，CoT 可以作为基础技术，而 ToT 通过协调多个链来构建在其之上。例如，ToT 节点可能每个都使用 CoT 内部生成连贯的思想。因此，这两个不是互斥的，但在复杂性和结构方面是层次相关的。

在下一章中，我们将探讨**推理和行动**（**ReAct**）模式，该模式在许多代理 AI 应用中普遍使用。
