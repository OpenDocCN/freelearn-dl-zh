# 20

# 思维链提示

**思维链**（**CoT**）**提示**起源于一篇名为《思维链提示引发大型语言模型中的推理》的研究论文，该论文由谷歌研究人员 Jason Wei、Xuezhi Wang、Dale Schuurmans、Maarten Bosma、Brian Ichter、Fei Xia、Ed Chi、Quoc Le 和 Denny Zhou 于 2022 年发表。

CoT 提示的关键创新在于鼓励语言模型在得出最终答案之前将复杂的推理问题分解成中间步骤。这是通过包含模型逐步推理的示例来实现的。

研究人员证明了通过用几个推理链的示例（例如“让我们一步步思考”）提示 LLM，模型可以显著提高其在需要多步推理的复杂任务上的性能，例如算术、常识和符号推理问题。

在 CoT 之前，大多数提示技术都集中在获取直接答案上。CoT 表明，明确鼓励模型展示其推理过程可以导致更准确的结果，尤其是在需要多个逻辑步骤的问题上。CoT 通过引导模型通过逻辑步骤来促进透明度和确保准确性，而直接回答虽然更快，但可能会错过澄清或验证答案背后推理的中间步骤。

这项研究特别有意义，因为它表明推理能力主要通过规模和提示而不是需要改变模型架构来产生。

在本章中，你将学习如何利用 CoT 提示来提高你的 LLM 在复杂推理任务上的性能。

在本章中，我们将涵盖以下主题：

+   设计有效的 CoT 提示

+   使用 CoT 提示进行问题解决

+   将 CoT 提示与其他技术相结合

+   评估 CoT 提示输出

+   CoT 提示的局限性

+   未来方向

# 设计有效的 CoT 提示

创建有效 CoT 提示的过程有助于培养清晰度、逻辑进展和结构化推理，从而确保更准确和连贯的输出。通过提供明确的问题陈述，将任务分解成更小的步骤，使用明确的标记来引导推理，以及包括一个样本 CoT 响应，模型将更好地遵循与人类问题解决方法相一致的系统方法，从而得出清晰和理性的结论：

1.  **提供明确的问题陈述**：精确的问题陈述将推理引导到特定的目标，消除歧义并确保模型确切地理解被要求做什么。这有助于防止误解并引导整个推理过程走向正确的方向。

1.  **将问题分解为逻辑步骤**：将复杂任务分解为更小、更易管理的步骤有助于组织推理，并使整体问题更容易解决。这种分解有助于一次关注一个方面，提高清晰度并降低遗漏重要细节的风险。

1.  **使用明确的推理标记**：例如“首先”、“接下来”和“最后”等标记作为推理过程逻辑流的标志。它们有助于以清晰的顺序结构化思维过程，确保问题各部分按正确顺序解决，从而提高整体回答的连贯性。

1.  **在提示中包含一个 CoT 示例响应**：提供示例有助于建立推理格式的标准，并为过程设定明确的期望。它还作为参考点，指导模型如何构建其响应，并使其更容易生成一致且逻辑上合理的输出。

这里是一个实现 CoT 提示的示例：

```py
def cot_prompt(question):
    return f"""Solve the following problem step by step:
Problem: {question}
Let's approach this step by step:
1) First, we need to...
2) Next, we should...
3) Then, we can...
4) Finally, we...
Therefore, the answer is...
Now, solve this new problem using the same step-by-step approach:
Problem: If a train travels 120 km in 2 hours, what is its average speed in km/h?
Let's solve this step by step:
"""
# Example usage
problem = "If a train travels 120 km in 2 hours, what is its average speed in km/h?"
prompt = cot_prompt(problem)
print(prompt)
```

此函数为给定的问题（`如果一列火车以 2 小时行驶 120 公里，其平均速度是多少 km/h？`）生成一个 CoT 提示，提供逐步推理的结构。以下是使用 CoT 的样本步骤：

```py
Solve the following problem step by step:
Problem: If a train travels 120 km in 2 hours, what is its average speed in km/h?
Let's approach this step by step:
1) First, we need to recall the formula for average speed, which is:
   Average Speed = Total Distance / Total Time.
2) Next, we should identify the total distance traveled, which is 120 km.
3) Then, we can identify the total time taken, which is 2 hours.
4) Now, we will apply the formula:
   Average Speed = 120 km / 2 hours.
5) Finally, we calculate the result:
   Average Speed = 60 km/h.
```

因此，答案是 60 km/h。

CoT 提示可以应用于各种问题解决场景。让我们看看下一个场景。

# 使用 CoT 提示进行问题解决

让我们实现一个使用 CoT 解决数学文字问题的函数：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
def solve_math_problem(model, tokenizer, problem):
    prompt = cot_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, max_length=500, num_return_sequences=1
    )
    solution = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )
    return solution
# Example usage
model_name = "gpt2-large"  # Replace with your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
problem = "If a recipe calls for 2 cups of flour for 8 servings, how many cups of flour are needed for 12 servings?"
solution = solve_math_problem(model, tokenizer, problem)
print(solution)
```

此函数应用 CoT 提示来解决数学文字问题（例如，`如果一份食谱需要 2 杯面粉制作 8 份，那么制作 12 份需要多少杯面粉？`），引导 LLM 通过逐步推理过程。

除了使用 CoT 提示进行问题解决外，我们还可以将其与其他技术结合，以提高 LLM 的性能。

# 将 CoT 提示与其他技术结合

CoT 可以与其他提示技术结合，以进一步提高 LLM 的性能。让我们实现一个结合 CoT 与**少样本学习**（**FSL**）的函数：

```py
def few_shot_cot_prompt(question, examples):
    prompt = "Solve the following problems step by step:\n\n"
    for example in examples:
        prompt += f"Problem: {example['question']}\n\n"
        prompt += f"Solution: {example['solution']}\n\n"
    prompt += f"Problem: {question}\n\nSolution:"
    return prompt
def solve_with_few_shot_cot(model, tokenizer, problem, examples):
    prompt = few_shot_cot_prompt(problem, examples)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500,
        num_return_sequences=1)
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return solution
# Example usage
examples = [
    {
        "question": "If a car travels 60 miles in 2 hours, what is its average speed?",
        "solution": "1) First, we identify the given information:\n   - Distance traveled = 60 miles\n   - Time taken = 2 hours\n\n2) We know that average speed is calculated by dividing distance by time:\n   Average Speed = Distance / Time\n\n3) Let's plug in the values:\n   Average Speed = 60 miles / 2 hours\n\n4) Perform the division:\n   Average Speed = 30 miles per hour\n\nTherefore, the car's average speed is 30 miles per hour."
    }
]
problem = "If a train travels 180 km in 3 hours, what is its average speed in km/h?"
solution = solve_with_few_shot_cot(model, tokenizer, problem,
    examples)
print(solution)
```

此函数结合 FSL 与 CoT 提示，提供逐步解决方案的示例，以指导 LLM 解决新问题（请参阅`如果一列火车以 3 小时行驶 180 公里，其平均速度是多少 km/h？`的代码示例）。最近的研究表明，将 CoT + FSL 等方法相结合可以提高基准测试中的性能（[`aclanthology.org/2023.emnlp-main.782.pdf`](https://aclanthology.org/2023.emnlp-main.782.pdf)）。

接下来，让我们看看如何评估 CoT 提示的质量。

# 评估 CoT 提示输出

评估 CoT 提示的输出涉及评估最终答案和推理过程。让我们实现一个简单的评估函数：

```py
def evaluate_cot_output(output, correct_answer):
    # Extract the final answer from the CoT output
    final_answer = extract_final_answer(output)
    # Check if the final answer is correct
    answer_correct = final_answer == correct_answer
    # Evaluate the reasoning steps
    reasoning_score = evaluate_reasoning_steps(output)
    return {
        "answer_correct": answer_correct,
        "reasoning_score": reasoning_score
    }
def extract_final_answer(output):
    # Implement logic to extract the final answer from the CoT output
    # This could involve parsing the last line or looking for specific phrases
    pass
def evaluate_reasoning_steps(output):
    # Implement logic to evaluate the quality of the reasoning steps
    # This could involve checking for logical consistency, completeness, etc.
    pass
# Example usage
problem = "If a train travels 180 km in 3 hours, what is its average speed in km/h?"
correct_answer = 60
cot_output = solve_math_problem(model, tokenizer, problem)
evaluation = evaluate_cot_output(cot_output, correct_answer)
print(evaluation)
```

此评估函数评估了 CoT 输出中最终答案的正确性和推理步骤的质量。

# CoT 提示的限制

虽然 CoT 提示功能强大，但它也有一些限制：

+   高 token 使用量和计算时间

+   多步推理中可能存在错误传播

+   依赖于初始提示的质量

+   可能不适用于所有类型的问题

为了解决这些局限性，考虑实施动态 CoT 方法：

```py
def dynamic_cot(model, tokenizer, problem, max_steps=5):
    prompt = f"Problem: {problem}\n\nLet's solve this step by step:"
    for step in range(1, max_steps + 1):
        prompt += f"\n\nStep {step}:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs, max_length=len(prompt) + 100,
            num_return_sequences=1
        )
        new_step = tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        prompt += new_step
        if "Therefore, the final answer is" in new_step:
            break
    return prompt
# Example usage
problem = "If a recipe calls for 2 cups of flour for 8 servings, how many cups of flour are needed for 12 servings?"
solution = dynamic_cot(model, tokenizer, problem)
print(solution)
```

`dynamic_cot`函数实现了一种动态 CoT 方法，通过语言模型逐步分解和解决问题。它首先创建一个初始提示，介绍问题并指导模型逐步解决它。然后，函数进入一个循环，最多迭代`max_steps`次（默认为`5`），在每次迭代中，它向模型提供一个不断增长的提示，其中包括迄今为止生成的所有步骤。模型处理这个提示，生成推理过程中的下一步，并将其附加到提示中。新步骤从标记化输出中解码并添加到提示字符串中。函数检查生成的步骤中是否存在短语`Therefore, the final answer is`，这表明模型已得出结论并应停止。如果找到这个短语，循环提前中断；否则，它继续进行，直到达到最大步骤数。最后，函数返回完整的提示，其中包含导致解决方案的所有推理步骤。然而，在实际应用中，模型的 token 限制可能会影响长多步提示。随着提示的每一步增长，它可能会超过模型的最大 token 限制，这可能导致输入截断、早期上下文丢失或无法生成准确的步骤，尤其是在复杂或长问题中。当处理需要许多步骤或大量上下文的问题时，这是一个重要的考虑因素。

# 未来方向

随着 CoT 提示的持续发展，出现了一些有希望的方向：

+   **自适应 CoT**：根据问题复杂度动态调整推理过程

+   **多模态 CoT**：在推理过程中结合视觉或听觉信息([`arxiv.org/abs/2302.00923`](https://arxiv.org/abs/2302.00923))

+   **协作 CoT**：结合多个 LLM 的见解或人机协作([`arxiv.org/html/2409.07355v1`](https://arxiv.org/html/2409.07355v1))

+   **CoT 的元学习**：元学习和 CoT 方法已成为解决少样本关系抽取挑战的有力技术([`arxiv.org/abs/2311.05922`](https://arxiv.org/abs/2311.05922))

这是一种自适应 CoT 的概念性实现：

```py
def adaptive_cot(
    model, tokenizer, problem, complexity_threshold=0.7
):
    # Assess problem complexity
    complexity = assess_problem_complexity(problem)
    if complexity > complexity_threshold:
        # Use detailed CoT for complex problems
        return detailed_cot(model, tokenizer, problem)
    else:
        # Use simple direct approach for simpler problems
        return simple_solve(model, tokenizer, problem)
def assess_problem_complexity(problem):
    # Implement logic to assess problem complexity
    # This could involve keyword analysis, sentence structure, etc.
    pass
def detailed_cot(model, tokenizer, problem):
    # Implement detailed Chain-of-Thought approach
    pass
def simple_solve(model, tokenizer, problem):
    # Implement simple direct solving approach
    pass
# Example usage
problem = "What is the result of 25 divided by 5?"
solution = adaptive_cot(model, tokenizer, problem)
print(solution)
```

这种自适应 CoT 方法评估问题复杂度并选择合适的解决策略，平衡效率和推理深度。

`adaptive_cot`函数根据问题的复杂性调整 CoT 方法。它首先通过调用`assess_problem_complexity`函数评估问题的复杂性，这可能涉及分析关键词、句子结构或其他特征以确定问题的复杂程度（尽管这一逻辑尚未实现）。如果复杂性评分超过预定义的阈值（`complexity_threshold`），则函数通过`detailed_cot`函数使用详细的 CoT 方法，这将生成更详细、分步骤的解决方案。对于简单问题，它通过`simple_solve`函数使用直接解决方法，该函数提供直接答案而不将问题分解成多个步骤。结果基于哪种方法被认为适用于给定问题而返回。这种动态方法允许模型根据其复杂性选择解决问题的最有效方法。

# 摘要

在本章中，您学习了如何设计有效的 CoT 提示，引导 LLM 通过逐步推理过程。我们讨论了该技术在各种问题解决场景中的应用，并讨论了如何将其与其他提示策略相结合。您还学习了如何评估 CoT 输出的质量，并理解了这种方法的优势。

通过实施本章讨论的策略和考虑因素，您可以显著提高您的 LLM 在复杂问题解决任务上的性能，同时深入了解模型的推理过程。

在下一章中，我们将探讨**思维树（ToT**）提示，这是一种高级技术，它扩展了 CoT 的概念，以创建更加复杂的推理结构。
