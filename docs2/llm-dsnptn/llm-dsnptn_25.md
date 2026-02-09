

# 第二十五章：自动多步推理和工具使用

在 LLM 中的多步推理和工具使用涉及到模型将复杂任务分解为可管理的步骤，并利用外部资源或 API 来完成这些任务的能力。这种能力显著扩展了 LLM 的解决问题的潜力，使它们能够处理更复杂、更真实世界的场景。其关键特征包括以下内容：

+   **任务分解**：这指的是模型将复杂输入或目标分解成更小、更易于管理的子任务的能力，这些子任务可以按顺序或分层解决。模型不是试图一次性解决整个问题，而是创建一个结构化的计划或推理步骤序列，逐步引导到解决方案。这个过程模仿了人类通常通过识别依赖关系、排序行动和将大目标分解为中间目标来处理复杂问题的方法。诸如思维链提示等技术通过提示模型在得出答案之前明确阐述每个推理步骤，明确鼓励这种行为。

+   **外部工具**：通过集成额外的资源，如数据库、API 或专用服务，可以增强 LLM 的能力，这些资源由于训练环境中的限制，LLM 无法直接访问。这些工具使 LLM 能够与实时数据交互，执行超出其内置知识的特定任务，或提供增强功能，如网页浏览、文件处理或执行外部脚本。例如，LLM 可以使用外部工具查询最新的天气数据，从实时 API 检索特定信息，或运行需要专用算法的计算。这种集成使 LLM 能够提供更动态、相关和专业的响应，特别是对于需要实时信息或复杂多步过程的适用。

+   **关于工具适用性的推理**：这涉及到模型在识别何时需要外部能力来解决特定子任务时的判断。模型必须评估子任务的性质，并确定内部推理是否足够，或者将部分任务委托给工具是否会产生更好甚至必要的成果。

+   **工具选择和调用**：这指的是模型识别适用于给定子任务的工具并制定正确输入以触发其使用的能力。这要求模型理解每个可用工具的功能和输入要求，并将这些与推理过程中当前步骤的需求相匹配。例如，如果任务需要访问最新的天气信息，模型必须选择天气 API 并生成一个对该 API 语法正确且语义相关的查询。此阶段包括格式化输入、调用工具并确保请求与当前问题上下文和工具的功能相一致。

+   **工具输出集成**：这描述了模型解释外部工具返回的结果并将其纳入持续推理过程的能力。在工具被调用并响应数据（如数值、结构化对象或文本片段）后，模型必须解析结果、提取相关元素并相应地更新其理解或中间输出。此步骤通常涉及解释异构输出格式、管理类型不匹配并在推理链中保持连续性。有效的集成确保工具使用不是孤立的，而是有意义地贡献于解决更广泛的任务。

+   **迭代问题解决**：这指的是模型递归地应用先前阶段——分解、工具推理、选择、调用和集成——在一个循环中，直到任务解决或进一步步骤变得无效。模型持续评估其进度，确定是否还有剩余的子任务，并决定是否需要进一步使用工具。这种迭代行为使模型能够通过调整计划或细化先前行动来处理具有动态结构、不确定性或先前步骤中的错误的任务。在基于代理的架构中，此过程可能由规划器或控制器显式管理，而在基于提示的设置中，它通常通过递归自我查询和提示增强而出现。

在本章中，我们将深入探讨使 LLM 能够执行复杂多步推理并利用外部工具的高级技术。

在本章中，我们将讨论以下主题：

+   设计用于复杂任务分解的提示

+   集成外部工具

+   实现自动工具选择和使用

+   复杂问题解决

+   评估多步推理和工具使用

+   挑战和未来方向

# 设计用于复杂任务分解的提示

为了实现有效的多步推理，提示应引导 LLM 将复杂任务分解成更小、更易管理的步骤。以下是一个任务分解提示的示例：

```py
def task_decomposition_prompt(task, available_tools):
    prompt = f"""Given the following complex task:
{task}
And the following available tools:
{' '.join(f'- {tool}' for tool in available_tools)}
Please break down the task into smaller, logical steps. For each step, indicate if a specific tool should be used. If no tool is needed, explain the reasoning required.
Your task decomposition:
Step 1:
Step 2:
Step 3:
...
Ensure that the steps are in a logical order and cover all aspects of the task.
"""
    return prompt
# Example usage
task = "Analyze the sentiment of tweets about a new product launch and create a summary report with visualizations."
available_tools = ["Twitter API", "Sentiment Analysis Model",
    "Data Visualization Library"]
prompt = task_decomposition_prompt(task, available_tools)
print(prompt)
```

此功能生成一个提示，引导 LLM 将复杂任务分解成步骤，考虑可用的工具。

# 集成外部工具

为了使 LLMs 能够使用外部工具，如搜索、计算、API 调用等，我们需要在模型和工具之间创建一个接口。以下是一个简单的实现：

1.  执行必要的导入并定义`ToolKit`类：

    ```py
    import requests
    from textblob import TextBlob
    import matplotlib.pyplot as plt
    class ToolKit:
        def __init__(self):
            self.tools = {
                "Twitter API": self.fetch_tweets,
                "Sentiment Analysis": self.analyze_sentiment,
                "Data Visualization": self.create_visualization
            }
    ```

    上述代码定义了一个`ToolKit`类，通过其方法组织和提供对不同功能的访问。在`__init__`方法中，名为`tools`的字典被初始化，其键代表工具名称，如`"Twitter API"`、`"Sentiment Analysis"`和`"Data Visualization"`，值引用获取推文、使用 TextBlob 库执行情感分析和使用 Matplotlib 创建数据可视化的相应方法。导入`requests`库用于发送 HTTP 请求，`TextBlob`用于自然语言处理任务，如情感分析，`matplotlib.pyplot`用于生成可视化。代码为这些工具设置了结构，但代码不完整，因为`fetch_tweets`、`analyze_sentiment`和`create_visualization`方法尚未定义，为这些功能的进一步实现留出了空间。

1.  定义三个方法：`fetch_tweets`用于根据查询生成模拟推文，`analyze_sentiment`用于使用 TextBlob 计算文本列表的情感极性分数，以及`create_visualization`用于创建并保存具有指定标题的情感数据直方图：

    ```py
        def fetch_tweets(self, query, count=100):
            return [f"Tweet about {query}" for _ in range(count)]
        def analyze_sentiment(self, texts):
            sentiments = [TextBlob(text).sentiment.polarity
                for text in texts]
            return sentiments
        def create_visualization(self, data, title):
            plt.figure(figsize=(10, 6))
            plt.hist(data, bins=20)
            plt.title(title)
            plt.xlabel("Sentiment")
            plt.ylabel("Frequency")
            plt.savefig("sentiment_visualization.png")
            return "sentiment_visualization.png"
    ```

1.  定义`use_tool`方法，如果工具字典中存在指定的工具，则使用给定的参数执行该工具；否则，返回错误信息：

    ```py
        def use_tool(self, tool_name, *args, kwargs):
            if tool_name in self.tools:
                return self.toolstool_name
            else:
                return f"Error: Tool '{tool_name}' not found."
    ```

1.  以下示例演示了使用`ToolKit`类获取有关产品发布的推文，分析其情感，创建情感可视化，并打印生成的可视化文件路径：

    ```py
    toolkit = ToolKit()
    tweets = toolkit.use_tool(
        "Twitter API", "new product launch", count=50
    )
    sentiments = toolkit.use_tool("Sentiment Analysis", tweets)
    visualization = toolkit.use_tool(
        "Data Visualization", sentiments,
        "Sentiment Analysis of Product Launch Tweets"
    )
    print(f"Generated visualization: {visualization}")
    ```

这个`ToolKit`类为 LLM 提供了一个与外部工具交互的接口，模拟 API 调用和数据处理任务。

# 实现自动工具选择和使用

为了使大型语言模型（LLMs）能够自动选择和使用工具，我们可以创建一个系统来解释模型的输出并执行相应的工具。以下是一个示例：

1.  首先，我们定义一个函数`auto_tool_use`，该函数使用来自 Hugging Face 的 Transformers 库的预训练语言模型和分词器，通过提示将任务分解为可执行步骤，将分解步骤解析为步骤，使用工具包按需执行工具，并收集结果：

    ```py
    from transformers import AutoModelForCausalLM, AutoTokenizer
    def auto_tool_use(model, tokenizer, task, toolkit):
        # Generate task decomposition
        decomposition_prompt = task_decomposition_prompt(
            task, toolkit.tools.keys()
        )
        inputs = tokenizer(decomposition_prompt,
            return_tensors="pt")
        outputs = model.generate(
            inputs, max_length=1000, num_return_sequences=1
        )
        decomposition = tokenizer.decode(outputs[0],
            skip_special_tokens=True)
        # Parse decomposition and execute tools
        steps = parse_steps(decomposition)
        results = []
        for step in steps:
            if step['tool']:
                result = toolkit.use_tool(step['tool'],
                    *step['args'])
            else:
                result = f"Reasoning: {step['reasoning']}"
            results.append(result)
    ```

1.  然后我们生成最终报告。生成的报告包含任务描述，每个步骤的分解及其结果，以及总结。模型使用提供的步骤和结果来生成更连贯和全面的任务叙述：

    ```py
        report_prompt = f"Task: {task}\n\nSteps and Results:\n"
        for i, (step, result) in enumerate(zip(steps, results), 1):
            report_prompt += (
                f"Step {i}: {step['description']}\n"
                f"Result: {result}\n\n"
            )
        report_prompt += "Please provide a comprehensive report summarizing the results and insights."
        inputs = tokenizer(report_prompt, return_tensors="pt")
        outputs = model.generate(
            inputs, max_length=1500, num_return_sequences=1
        )
        report = tokenizer.decode(outputs[0],
            skip_special_tokens=True)
        return report
    ```

1.  然后，我们实现逻辑来将分解步骤结构化。这是一个简化的占位符实现：

    ```py
    def parse_steps(decomposition):
        steps = []
        for line in decomposition.split('\n'):
            if line.startswith("Step"):
                tool = "Twitter API" if "Twitter" in line else \
                       "Sentiment Analysis" if "sentiment" in line else \
                       "Data Visualization" if "visualization" in line else None
                steps.append({
                    'description': line,
                    'tool': tool,
                    'args': [],
                    'reasoning': line if not tool else ""
                })
        return steps
    ```

1.  以下示例用法演示了使用 `AutoModelForCausalLM` 和 `AutoTokenizer` 加载语言模型和分词器，定义一个分析推文情感并生成带有可视化总结报告的任务，以及使用 `auto_tool_use` 函数通过 `ToolKit` 自动化任务，最终报告将被打印出来：

    ```py
    model_name = "llama3.3"  # Replace with your preferred model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    task = "Analyze the sentiment of tweets about a new product launch and create a summary report with visualizations."
    toolkit = ToolKit()
    report = auto_tool_use(model, tokenizer, task, toolkit)
    print(report)
    ```

以下代码片段从高层次展示了如何使大型语言模型（LLM）自动分解任务、选择合适的工具，并根据结果生成最终报告。

本章的前三部分通过涵盖提示设计、集成外部工具和实现自动工具选择来增强人工智能功能，奠定了基础。在接下来的部分，我们将探讨如何设计用于复杂问题解决的提示。

# 复杂问题解决

多步推理和工具使用可以应用于各种复杂问题解决场景。以下是如何使用这种方法进行市场分析的示例：

```py
def market_analysis(model, tokenizer, toolkit, product_name):
    task = f"""Conduct a comprehensive market analysis for the product: {product_name}.
    Include competitor analysis, sentiment analysis of customer reviews, and market trends visualization."""
    analysis_report = auto_tool_use(model, tokenizer, task, toolkit)
    return analysis_report
# Example usage
product_name = "SmartHome AI Assistant"
market_report = market_analysis(model, tokenizer, toolkit,
        product_name)
print(market_report)
```

`market_analysis` 函数通过构建结构化任务提示并将其传递给外部实用工具 `auto_tool_use`（假设它协调语言模型从工具增强的响应），自动化给定产品的市场研究报告的生成。提示要求进行多部分分析——涵盖竞争对手、客户反馈的情感分析以及市场趋势的可视化——针对提供的特定 `product_name`。这种设计利用模型和工具包生成综合报告，无需人工干预，通过提示驱动的执行实现产品市场研究的一致性和可重复性方法。

# 评估多步推理和工具使用

为了评估多步推理和工具使用的有效性，我们需要评估过程和结果。以下是一个简单的评估框架：

```py
def evaluate_multistep_tooluse(
    task, generated_report, ground_truth, criteria
):
    scores = {}
    for criterion in criteria:
        scores[criterion] = evaluate_criterion(generated_report,
            ground_truth, criterion)
    # Evaluate tool use effectiveness
    tool_use_score = evaluate_tool_use(task, generated_report)
    scores['Tool Use Effectiveness'] = tool_use_score
    return scores
def evaluate_criterion(generated_report, ground_truth, criterion):
    # Implement criterion-specific evaluation logic
    # This is a placeholder implementation
    return 0.0  # Return a score between 0 and 1
def evaluate_tool_use(task, generated_report):
    # Implement logic to evaluate how effectively tools were used
    # This could involve checking for specific tool outputs or insights
    # This is a placeholder implementation
    return 0.0  # Return a score between 0 and 1
# Example usage
criteria = ['Accuracy', 'Comprehensiveness', 'Insight Quality',
    'Logical Flow']
ground_truth = "Ideal market analysis report content..."  # This would be a benchmark report
evaluation_scores = evaluate_multistep_tooluse(task, market_report,
    ground_truth, criteria)
print("Evaluation Scores:", evaluation_scores)
```

此评估框架评估了生成的报告的质量以及工具使用过程中的有效性。

# 挑战和未来方向

尽管功能强大，LLM 中的多步推理和工具使用面临几个挑战：

+   **工具选择准确性**：确保 LLM 为每个任务选择最合适的工具

+   **错误传播**：减轻推理过程早期步骤中错误的影响；记住，如果不在早期减轻，错误在多个步骤中的传播可能会在复杂的工具链中成为主要风险

+   **可扩展性**：管理集成大量不同工具的复杂性

+   **适应性**：使 LLM 能够在不重新训练的情况下与新的、未见过的工具一起工作

为了解决这些挑战，可以考虑实现一个自我纠正机制：

```py
def self_correcting_tooluse(
    model, tokenizer, task, toolkit, max_attempts=3
):
    for attempt in range(max_attempts):
        report = auto_tool_use(model, tokenizer, task, toolkit)
        # Prompt the model to evaluate its own work
        evaluation_prompt = f"""Task: {task}
Generated Report:
{report}
Please evaluate the quality and completeness of this report. Identify any errors, omissions, or areas for improvement. If necessary, suggest specific steps to enhance the analysis.
Your evaluation:
"""
        inputs = tokenizer(evaluation_prompt, return_tensors="pt")
        outputs = model.generate(
            inputs, max_length=1000, num_return_sequences=1
        )
        evaluation = tokenizer.decode(outputs[0],
            skip_special_tokens=True)
        if "satisfactory" in evaluation.lower() and "no major issues" in evaluation.lower():
            break
        # If issues were identified, use the evaluation to improve the next attempt
        task += f"\n\nPrevious attempt evaluation: {evaluation}\nPlease address these issues in your next attempt."
    return report
# Example usage
final_report = self_correcting_tooluse(model, tokenizer, task,
    toolkit)
print(final_report)
```

在这个语境中，自我纠正指的是一种语言模型通过评估和改进其先前响应（而不需要外部反馈）来迭代地细化其输出的方法。在`self_correcting_tooluse`函数中，这是通过首先使用`auto_tool_use`生成报告，然后提示模型评估该报告的质量来实现的。如果模型的自评估不包括充分性的指标——例如“满意”和“没有重大问题”——则评估将附加到任务描述中，有效地指导下一次迭代解决已识别的不足。这个循环会持续一定次数的尝试（`max_attempts`），直到输出满足模型自己的接受标准，允许在多次迭代中进行自我引导的细化。

我们可以确定以下三个有希望的研究领域，以克服来自 AI/ML 社区一些研究带来的挑战：

+   **增强的工具学习和发现**：未来的 LLMs 将能够动态地了解和整合新工具，而无需显式编程。这涉及到理解工具文档和 API 规范以及通过实验工具来推断其功能性的机制。这将使 LLMs 能够适应不断演变的软件和服务景观，扩展其功能，而不仅仅是预定义的工具集。这将涉及元学习、从工具交互中进行强化学习以及工具描述的语义理解技术（[`arxiv.org/abs/2305.17126`](https://arxiv.org/abs/2305.17126)）。

+   **具有不确定性的鲁棒和自适应推理**：未来的大型语言模型（LLMs）将整合概率模型来处理多步任务中的不确定性。这意味着为不同的推理路径、结果和工具有效性分配概率。贝叶斯方法、蒙特卡洛模拟和其他概率技术将被整合到推理过程中。这将使 LLMs 能够在信息不完整或噪声复杂场景中做出更鲁棒的决定，并更好地管理现实世界问题的固有不确定性。LLMs 将更好地应对意外情况，从错误中恢复，并在面对模糊性时提供更可靠的解决方案（[`arxiv.org/abs/2310.04406`](https://arxiv.org/abs/2310.04406)）。

+   **具有可解释性的闭环多步推理**：未来的系统将在多步问题解决中涉及人类与大型语言模型（LLM）之间更紧密的合作。这意味着创建允许人类理解 LLM 推理过程、提供指导、纠正错误并在复杂任务上共同工作的界面。可解释性将是关键，LLM 能够阐述其推理步骤、证明工具选择并展示替代解决方案路径。这将促进信任并允许更有效的人类-人工智能合作，特别是在医疗保健、金融和科学研究等关键领域。这可能包括推理图的可视化、自然语言解释和交互式调试工具：[`www.microsoft.com/en-us/research/blog/guidance-for-developing-with-large-language-models-llms/`](https://www.microsoft.com/en-us/research/blog/guidance-for-developing-with-large-language-models-llms/).

# 摘要

自动多步推理和工具使用显著扩展了 LLM 的解决问题能力，使它们能够处理复杂、现实世界的任务。

在本章中，你学习了如何设计用于复杂任务分解的提示，并实现允许 LLM 与外部工具和 API 交互的系统。我们探讨了自动工具选择和使用的策略，并探讨了在复杂问题解决场景中的应用。你还学习了如何评估 LLM 中多步推理和工具使用的有效性。通过实施本章讨论的技术和考虑因素，你可以创建复杂的 AI 系统，这些系统能够分解问题、利用外部工具并针对多方面挑战生成全面的解决方案。

随着我们继续前进，本书的下一部分将专注于检索和知识集成。这将建立在我们在本部分讨论的工具使用能力之上，探讨 LLM 如何通过外部知识得到增强，提高其有效获取和利用信息的能力。

# 第五部分：大型语言模型中的检索和知识集成

我们通过考察通过检索增强生成（RAG）方法增强 LLMs 的外部知识的技术来结束本书。你将学习如何设计检索系统，以高效地访问相关信息，将结构化知识集成到模型输出中，并利用基于图的检索来丰富响应中的上下文关系。我们将探讨高级 RAG 模式，如迭代和自适应检索，帮助你创建能够动态集成知识的模型。我们还讨论了评估方法，以衡量检索质量和有效性。最后一章介绍了代理模式，使你能够构建结合推理、规划和决策的自主系统。通过掌握这些技术，你将能够创建不仅信息丰富，而且能够实现目标导向行为的 LLMs。

本部分包含以下章节：

+   *第二十六章*，*检索增强生成*

+   *第二十七章*，*基于图的 RAG*

+   *第二十八章*，*高级 RAG*

+   *第二十九章*，*评估 RAG 系统*

+   *第三十章*，*代理模式*
