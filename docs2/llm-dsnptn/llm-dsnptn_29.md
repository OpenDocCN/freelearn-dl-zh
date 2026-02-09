# 29

# 评估 RAG 系统

RAG 系统力求产生更准确、相关和事实依据的响应。然而，评估这些系统的性能提出了独特的挑战。与传统的信息检索或**问答**（**QA**）系统不同，RAG 评估必须考虑检索信息的质量以及 LLM 利用这些信息生成高质量响应的有效性。

在本章中，我们将探讨评估 RAG 系统的复杂性。我们将检查这项任务固有的挑战，分析用于评估检索质量和生成性能的关键指标，并讨论进行全面评估的各种策略。

本章旨在为您提供对 RAG 评估原则和实践的全面理解，使您具备评估和改进这些强大系统所需的知识。

在本章中，我们将涵盖以下主题：

+   评估 LLM 的 RAG 系统面临的挑战

+   评估基于 LLM 的 RAG 检索质量的指标

+   RAG 检索指标考虑因素

+   评估检索信息的相关性

+   测量检索对 LLM 生成的影响

+   在 LLM 中端到端评估 RAG 系统

+   基于 LLM 的 RAG 的人评技术

+   RAG 评估的基准和数据集

# 评估 LLM 的 RAG 系统面临的挑战

评估 RAG 系统提出了一系列独特的挑战，这些挑战使其与传统信息检索或 QA 系统评估区分开来。这些挑战源于检索和生成组件之间的相互作用以及评估事实准确性和生成文本质量的需求。

以下各节将详细说明在评估 LLM 的 RAG 系统时遇到的特定挑战。

## 检索和生成之间的相互作用

RAG 系统的性能是其检索组件和生成组件共同作用的结果。强大的检索可以为 LLM 提供相关且准确的信息，从而产生更好的响应。相反，较差的检索可能会误导 LLM，导致即使生成器本身能力很强，也会产生不准确或不相关的答案。因此，评估 RAG 系统需要评估检索信息的质量以及 LLM 在生成过程中有效利用这些信息的能力。

## 上下文敏感评估

与传统信息检索不同，传统信息检索通常仅基于查询来评估相关性，RAG 评估必须考虑检索信息使用的上下文。一个文档可能在孤立的情况下与查询相关，但可能不提供在生成响应上下文中准确回答问题的具体信息。这需要上下文敏感的评估指标，在评估检索文档的相关性时，既要考虑查询也要考虑生成的文本。

## 超越事实准确性

虽然事实准确性是 RAG 评估的主要关注点，但它并不是决定生成响应质量的决定性因素。响应还必须流畅、连贯，并且与用户的查询相关。这些文本质量方面通常通过人工评估来评估，这可能既昂贵又耗时。开发与人类对这些定性方面判断相关联的自动化指标仍然是一个开放的研究挑战。

## 自动化指标的局限性

自动化指标，如从信息检索（例如，精确度、召回率）或机器翻译（例如，BLEU、ROUGE）借用，可以为 RAG 系统性能提供有用的见解。然而，它们通常无法全面反映整个情况。检索指标可能无法完全反映文档对生成的有用性，而生成指标可能无法充分评估生成文本在检索上下文中的事实基础。

## 错误分析困难

当一个 RAG 系统产生错误或低质量的响应时，确定根本原因可能具有挑战性。检索组件是否无法找到相关文档？LLM 是否未能正确利用检索到的信息？LLM 是否产生了不基于提供上下文的响应？解开这些因素需要仔细的错误分析和可能的新诊断工具的开发。

## 需要多样化的评估场景

RAG 系统可以部署在广泛的领域中，从开放域问答到特定领域的聊天机器人。具体的挑战和评估标准可能因用例而异。评估 RAG 系统在多样化和不同领域中的性能对于理解其优势和劣势至关重要。

## 动态知识和演变信息

在许多实际应用中，底层知识库是不断演变的。新信息被添加，现有信息被更新或变得过时。评估一个 RAG 系统如何适应这些变化并保持其响应的准确性是一个重大挑战。

## 计算成本

评估 RAG 系统，尤其是使用更大 LLM 的系统，可能计算成本很高。使用大型模型进行推理并在大规模上进行人工评估可能需要大量资源。找到平衡评估彻底性和计算成本的方法是一个重要的考虑因素。

让我们看看一些关键指标，用于评估基于 LLM 的 RAG 系统中检索组件的相关性和对响应生成的有用性。

# 基于 LLM 的 RAG 中评估检索质量的指标

检索组件在 RAG 系统的整体性能中起着至关重要的作用。它负责为 LLM 提供相关且准确的信息，这些信息是生成响应的基础。因此，评估检索组件的质量是 RAG 评估的一个关键方面。我们可以将传统的信息检索指标适应到 RAG 设置中，重点关注检索器找到的文档不仅与查询相关，而且对 LLM 生成高质量答案有用的能力。

## Recall@k

Recall@k 衡量的是在顶部 *k* 个检索结果中成功检索到的相关文档的比例。在 RAG 的背景下，我们可以将相关文档定义为包含回答查询所需必要信息的文档：

+   **公式**: *Recall@k = (在顶部 k 个检索到的相关文档数量) / (总的相关文档数量)*

+   **解释**: Recall@k 越高，表示检索组件可以找到更大比例的相关文档

+   **示例**: 如果整个语料库中有五篇文档包含回答特定查询所需的信息，并且 RAG 系统在顶部 10 个结果中检索到其中的三篇，那么该查询的 Recall@10 将是 3/5 = 0.6

## Precision@k

Precision@k 衡量的是在顶部 *k* 个检索结果中相关文档的比例：

+   **公式**: *Precision@k = (在顶部 k 个检索到的相关文档数量) / (**k**)*

+   **解释**: Precision@k 越高，表示检索到的文档中有更大比例的相关文档

+   **示例**: 如果一个 RAG 系统为查询检索了 10 篇文档，其中有四篇是相关的，那么 Precision@10 将是 4/10 = 0.4

## 平均倒数排名 (MRR)

MRR 考虑了检索到的第一个相关文档的排名。它强调了在排名早期检索相关文档的重要性：

+   **公式**: *MRR = (1 / |Q|) * Σ (1 / rank_i) for i = 1 to |Q|*, 其中 *|Q|* 是查询的数量，*rank_i* 是查询 *i* 的第一个相关文档的排名。

+   **解释**: MRR 越高，表示相关文档被检索到的排名越高（越接近顶部）。

+   **示例**: 如果一个查询的第一个相关文档在排名三处被检索到，则倒数排名为 1/3。MRR 会平均多个查询的这些倒数排名。

## 标准化折现累积增益 (NDCG@k)

NDCG@k 是一个更复杂的指标，它考虑了检索文档的相关性和它们在排名中的位置。它使用一个分级的相关性量表（例如，0、1、2，其中 2 是非常相关）并给在更高排名检索到的相关文档分配更高的分数：

+   **公式**：NDCG@k 涉及计算检索列表的**折算累积收益**（**DCG**）并将其通过**理想折算累积收益**（**IDCG**）进行归一化，后者是完美排名列表的 DCG。公式很复杂，但可以使用 sklearn 等库轻松计算。

+   **解释**：更高的 NDCG@k 表明，高度相关的文档在更高的排名中被检索到。

接下来，让我们讨论如何决定使用哪些检索指标。

# RAG 中检索指标的考虑因素

在 RAG 的背景下，我们需要仔细定义相关性。一个文档可能对查询相关，但不包含回答查询所需的特定信息。我们可能需要使用更严格的定义，例如“包含查询的答案。”如前所述，RAG 中的相关性通常是上下文相关的。一个文档可能在孤立的情况下与查询相关，但不是在给定其他检索文档的情况下生成特定答案的最有帮助的文档。

虽然 Recall@k 和 Precision@k 等指标关注的是前*k*个检索到的文档，但考虑更广泛结果的整体检索质量也很重要。例如，**平均精度**（**AP**）可以提供一个更全面的视角。

让我们用 Python 和 sklearn 库来演示如何计算 Recall@k、Precision@k、MRR 和 NDCG@k：

1.  我们首先导入必要的库，并定义代表一组查询、每个查询的真实相关文档以及每个查询由 RAG 系统检索到的文档的样本数据：

    ```py
    import numpy as np
    from sklearn.metrics import ndcg_score
    # Sample data
    queries = [
        "What is the capital of France?",
        "Who painted the Mona Lisa?",
        "What is the highest mountain in the world?"
    ]
    ground_truth = [
        [0, 1, 2],  # Indices of relevant documents for query 1
        [3, 4],     # Indices of relevant documents for query 2
        [5, 6, 7]   # Indices of relevant documents for query 3
    ]
    retrieved = [
        [1, 5, 0, 2, 8, 9, 3, 4, 6, 7],  # Ranked list of retrieved document indices for query 1
        [4, 3, 0, 1, 2, 5, 6, 7, 8, 9],  # Ranked list of retrieved document indices for query 2
        [6, 5, 7, 0, 1, 2, 3, 4, 8, 9]   # Ranked list of retrieved document indices for query 3
    ]
    ```

1.  然后，我们定义一个函数，`calculate_recall_at_k`，用于计算给定查询集、真实相关文档和检索文档列表的 Recall@k：

    ```py
    def calculate_recall_at_k(ground_truth, retrieved, k):
        """Calculates Recall@k for a set of queries."""
        recall_scores = []
        for gt, ret in zip(ground_truth, retrieved):
            num_relevant = len(gt)
            retrieved_k = ret[:k]
            num_relevant_retrieved = len(
                set(gt).intersection(set(retrieved_k))
            )
            recall = (
                num_relevant_retrieved / num_relevant
                if num_relevant > 0 else 0
            )
            recall_scores.append(recall)
        return np.mean(recall_scores)
    ```

1.  我们接下来定义一个函数，`calculate_precision_at_k`，用于计算给定查询集、真实值和检索列表的 Precision@k：

    ```py
    def calculate_precision_at_k(ground_truth, retrieved, k):
        """Calculates Precision@k for a set of queries."""
        precision_scores = []
        for gt, ret in zip(ground_truth, retrieved):
            retrieved_k = ret[:k]
            num_relevant_retrieved = len(
                set(gt).intersection(set(retrieved_k))
            )
            precision = num_relevant_retrieved / k if k > 0 else 0
            precision_scores.append(precision)
        return np.mean(precision_scores)
    ```

1.  我们定义一个函数，`calculate_mrr`，用于计算给定查询集、真实值和检索列表的 MRR。更高的 MRR 表明系统在更高的排名中一致地检索到相关文档：

    ```py
    def calculate_mrr(ground_truth, retrieved):
        """Calculates Mean Reciprocal Rank (MRR) for a set of queries."""
        mrr_scores = []
        for gt, ret in zip(ground_truth, retrieved):
            for i, doc_id in enumerate(ret):
                if doc_id in gt:
                    mrr_scores.append(1 / (i + 1))
                    break
            else:
                mrr_scores.append(0)  # No relevant document found
        return np.mean(mrr_scores)
    ```

1.  我们还定义了一个函数，`calculate_ndcg_at_k`，用于计算 NDCG@k。在这里，我们将使用一个简化版本，其中相关性分数是二进制的（0 或 1）：

    ```py
    def calculate_ndcg_at_k(ground_truth, retrieved, k):
        """Calculates NDCG@k for a set of queries."""
        ndcg_scores = []
        for gt, ret in zip(ground_truth, retrieved):
            relevance_scores = np.zeros(len(ret))
            for i, doc_id in enumerate(ret):
                if doc_id in gt:
                    relevance_scores[i] = 1
            # sklearn.metrics.ndcg_score requires 2D array
            true_relevance = np.array([relevance_scores])
            retrieved_relevance = np.array([relevance_scores])
            ndcg = ndcg_score(
                true_relevance, retrieved_relevance, k=k
            )
            ndcg_scores.append(ndcg)
        return np.mean(ndcg_scores)
    ```

1.  最后，我们计算并打印不同*k*值的检索指标：

    ```py
    k_values = [1, 3, 5, 10]
    for k in k_values:
        recall_at_k = calculate_recall_at_k(ground_truth,
            retrieved, k)
        precision_at_k = calculate_precision_at_k(
            ground_truth, retrieved, k
        )
        ndcg_at_k = calculate_ndcg_at_k(ground_truth, retrieved, k)
        print(f"Recall@{k}: {recall_at_k:.3f}")
        print(f"Precision@{k}: {precision_at_k:.3f}")
        print(f"NDCG@{k}: {ndcg_at_k:.3f}")
    mrr = calculate_mrr(ground_truth, retrieved)
    print(f"MRR: {mrr:.3f}")
    ```

# 评估 LLMs 检索信息的相关性

虽然上一节中讨论的检索指标提供了检索质量的总体评估，但它们并没有完全捕捉到 RAG 中相关性的细微差别。在 RAG 中，检索到的信息不是最终产品，而是一个中间步骤，作为 LLM 的输入。因此，我们需要评估检索信息不仅与查询相关，而且与通过 LLM 生成高质量响应的具体任务相关。

传统的信息检索通常侧重于找到与查询主题相关的文档。然而，在 RAG 中，我们需要一个更细致的相关性概念，它考虑以下方面：

+   **可回答性**：检索到的信息是否包含回答查询所需的特定信息？一个文档可能一般与查询相关，但不包含精确的答案。

+   **上下文效用**：检索到的信息在其他检索文档的上下文中是否有用？一个文档在孤立的情况下可能相关，但与其他检索信息结合时可能冗余甚至矛盾。

+   **LLM 兼容性**：检索到的信息是否以 LLM 可以轻松理解和利用的格式存在？例如，一个长而复杂的文档可能相关，但对于 LLM 来说可能难以有效处理。

+   **忠实度支持**：检索到的信息是否提供了足够的证据来支持生成答案中的主张？这对于确保 LLM 的响应基于检索到的上下文至关重要。

## 检索信息相关性的评估方法

这里有一些评估检索信息相关性的方法，这些方法超越了传统的查询相关性：

+   **人工评估**：

    +   **直接评估**：人工标注者可以直接评估检索文档与查询和生成响应的相关性。他们可以要求在李克特量表（例如，1 到 5）上评分，或提供二元判断（相关/不相关）。

    +   **比较评估**：标注者可以展示多组检索到的文档，并要求他们根据其回答查询的有用性进行排名，或选择最佳集合。

    +   **基于任务的评估**：标注者可以要求使用检索到的文档自行回答查询。他们回答的准确性和质量可以作为检索信息相关性和有用性的间接衡量标准。

+   **自动化指标**：让我们考虑一些常用的自动化指标。请记住，虽然自动化指标提供了性能的定量衡量，但人工评估为生成响应的相关性、连贯性和有用性提供了有价值的定性见解：

    +   **答案重叠**：我们可以使用 ROUGE 或 BLEU 等指标自动测量生成的答案和检索到的文档之间的重叠。更高的重叠表明 LLM 正在利用检索到的信息。

    +   **问答指标**：如果我们有真实答案，我们可以将检索到的上下文视为问答系统的输入，并使用标准问答指标（如**精确匹配**（EM）和 F1 分数）来评估其性能。

    +   **忠实度指标**：我们可以使用自然语言推理（NLI）等技术来评估生成的答案是否由检索到的上下文所蕴含。我们将在本章后面的部分详细讨论 NLI 模型。

    +   **困惑度**：当我们对检索到的上下文进行条件化时，我们可以测量 LLM 的困惑度。较低的困惑度表明 LLM 认为上下文是有信息量和对生成有用的。

例如，让我们用 Python 中的`rouge-score`库来说明如何实现简单的答案重叠指标：

1.  首先，我们运行以下命令来安装`rouge-score`库，该库提供了 ROUGE 指标的实现，并导入必要的模块：

    ```py
         pip install rouge-score
          from rouge_score import rouge_scorer
    ```

1.  然后，我们定义表示查询、生成的答案和检索到的文档列表的样本数据：

    ```py
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    retrieved_documents = [
        "Paris is the capital city of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is a famous landmark in Paris.",
        "London is the capital of the United Kingdom."
    ]
    ```

1.  接下来，我们定义一个函数`calculate_rouge_scores`来计算生成的答案和每个检索到的文档之间的 ROUGE 分数：

    ```py
         def calculate_rouge_scores(answer, documents):
        """Calculates ROUGE scores between the answer and each document."""
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        scores = []
        for doc in documents:
            score = scorer.score(answer, doc)
            scores.append(score)
        return scores
    ```

1.  然后我们计算并打印每个文档的 ROUGE 分数：

    ```py
    rouge_scores = calculate_rouge_scores(answer,
        retrieved_documents)
    for i, score in enumerate(rouge_scores):
        print(f"Document {i+1}:")
        print(f"  ROUGE-1: {score['rouge1'].fmeasure:.3f}")
        print(f"  ROUGE-2: {score['rouge2'].fmeasure:.3f}")
        print(f"  ROUGE-L: {score['rougeL'].fmeasure:.3f}")
    ```

1.  最后，我们计算并打印所有文档的平均 ROUGE 分数：

    ```py
    avg_rouge1 = sum([score['rouge1'].fmeasure
        for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score['rouge2'].fmeasure
        for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score['rougeL'].fmeasure
        for score in rouge_scores]) / len(rouge_scores)
    print(f"\nAverage ROUGE Scores:")
    print(f"  Average ROUGE-1: {avg_rouge1:.3f}")
    print(f"  Average ROUGE-2: {avg_rouge2:.3f}")
    print(f"  Average ROUGE-L: {avg_rougeL:.3f}")
    ```

## 评估 RAG 特定相关性的挑战

在探索了评估检索信息相关性的几种方法之后，我们现在转向概述涉及此评估过程的一些关键挑战：

+   **主观性**：相关性判断可能是主观的，尤其是在考虑上下文效用和 LLM 兼容性等因素时。

+   **标注成本**：人工评估可能既昂贵又耗时，尤其是在大规模评估中。

+   **指标限制**：自动指标可能无法完全捕捉 RAG 特定相关性的细微差别，并且可能并不总是与人类判断很好地相关。

+   **动态上下文**：文档的相关性可能根据检索到的其他文档和 LLM 使用的特定生成策略而变化。

接下来，让我们学习如何测量检索对 LLM 生成的影响。

# 测量检索对 LLM 生成的影响

在 RAG 系统中，生成的响应质量很大程度上受到检索到的信息的影响。良好的检索提供了必要的信息和事实，而差的检索可能导致不相关或不正确的响应。通过更好的模型和过滤来增强检索可以提高整体性能，这通过精确度、忠实度和用户满意度来衡量。

因此，评估的一个关键方面是衡量检索对 LLM 生成的影响。让我们来看看一些关键指标和技术。

## 评估检索影响的关键指标

如前所述，由大型语言模型（LLM）生成的响应质量与其检索到的信息紧密相关。因此，评估检索对最终响应的影响至关重要。这包括评估 LLM 如何有效地利用检索到的上下文来生成准确、相关且扎根的答案。现在让我们考察一下在这次评估中使用的某些关键指标：

+   **扎根性/忠实度**：

    **扎根性**，也称为忠实度，衡量的是生成的响应在多大程度上由检索到的上下文提供事实支持。一个扎根的响应应仅包含可以从提供的文档中推断出的信息。

    评估此指标的一些技术如下：

    +   **人工评估**：人工标注者可以直接通过验证每个陈述是否由检索到的上下文支持来评估生成响应中每个陈述的扎根性。这可能涉及二元判断（扎根/非扎根）或更细致的评分。

    +   **自动化指标**：

        +   **自然语言推理（NLI）**：NLI 模型可以用来确定生成的响应中的每个句子是否由检索到的上下文所蕴涵。我们将检索到的文档的拼接视为前提，将响应中的每个句子视为假设。高蕴涵分数表明句子在上下文中是扎根的。

        +   **基于问答的评估**：我们可以基于生成的响应制定问题，并检查问答模型是否可以使用检索到的上下文作为信息来源正确回答这些问题。高回答能力分数表明响应是扎根的。

        +   **事实验证模型**：这些模型可以用来检查生成的响应中陈述的每个事实是否由检索到的文档或外部知识来源支持。

+   **答案相关性**：

    答案相关性衡量的是在检索到的上下文的背景下，生成的响应如何有效地解决用户的查询。即使检索到的上下文不完美，一个好的 RAG 系统也应努力提供相关且有帮助的答案。

    评估此指标的一些技术如下：

    +   **人工评估**：人类评判者可以在考虑检索上下文的限制的同时评估生成响应与查询的相关性。他们可以在李克特量表上评分或提供比较性判断（例如，对多个响应进行排名）。

    +   **自动化指标**：

        +   **查询-答案相似度**：我们可以使用基于嵌入的技术（例如，余弦相似度）或其他相似度指标来衡量查询和生成的响应之间的语义相似度。

        +   **特定任务的指标**：根据具体的应用，我们可以使用特定任务的指标。例如，在问答场景中，我们可以使用 EM 或 F1 分数等指标来衡量生成的答案与标准答案之间的重叠。

        +   **信息检索指标**：我们可以将生成的响应视为检索到的文档，并使用传统的信息检索指标（如精确度、召回率或 NDCG）来评估其与查询的相关性，前提是我们有查询-答案对的相关性判断。

+   **上下文利用**：

    这个方面关注的是 LLM 在生成响应时有效利用检索上下文的能力。它不仅测量可信度，还评估 LLM 是否适当地整合和综合上下文中的信息。

    评估此指标的一些技术如下：

    +   **人工评估**：人工标注者可以评估 LLM 使用检索上下文的程度，识别出模型在上下文利用不足或过度依赖上下文的实例。

    +   **自动指标**：

        +   **归因分析**：我们可以使用注意力可视化或基于梯度的归因等技术来识别 LLM 在生成过程中最关注检索上下文的哪些部分。

        +   **上下文消除**：我们可以测量当上下文的部分被移除或修改时，生成的响应的变化。这有助于确定上下文中最具影响力的部分。

作为例子，让我们使用 NLI 模型进行可信度评估。为此，我们将使用 Transformers 库：

1.  我们首先运行以下命令，安装`transformers`库。这个库提供了用于处理预训练的 transformer 模型（如 NLI）的工具。我们还导入了必要的模块：

    ```py
    pip install transformers torch
    from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
    )
    import torch
    ```

1.  然后我们定义了代表查询、生成答案和检索上下文的样本数据：

    ```py
     query = "What is the capital of France?"
    answer = "The capital of France is Paris. It is a global center for art, fashion, gastronomy, and culture."
    context = """
    Paris is the capital city of France. It is situated on the River Seine, in northern France.
    Paris has an area of 105 square kilometers and a population of over 2 million people.
    France is a country located in Western Europe.
    """
    ```

1.  我们加载了一个预训练的 NLI 模型及其相应的分词器。在这里，我们使用的是`roberta-large-mnli`模型，这是一个在 MultiNLI 数据集上微调过的 RoBERTa 模型：

    ```py
    model_name = "roberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = \
        AutoModelForSequenceClassification.from_
    pretrained(
        model_name
    )
    ```

1.  然后我们定义了一个函数`calculate_claim_groundedness`，它根据上下文计算单个断言（来自生成答案的句子）的蕴涵分数：

    ```py
    def calculate_claim_groundedness(context, claim):
        """Calculates the entailment score for a single claim given the context."""
        inputs = tokenizer(context, claim, truncation=True,
            return_tensors="pt")
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        entailment_prob = probs[0][2].item()  # Assuming label 2 corresponds to entailment
        return entailment_prob
    ```

1.  我们还定义了一个函数`calculate_groundedness`，用于计算整个生成答案的整体可信度分数。它将答案拆分为句子，计算每个句子的蕴涵分数，然后平均这些分数：

    ```py
    def calculate_groundedness(context, answer):
        """Calculates the overall groundedness score for the generated answer."""
        claims = answer.split(". ")  # Simple sentence splitting
        if not claims:
            return 0
        claim_scores = []
        for claim in claims:
            if claim:
              score = calculate_claim_groundedness(context, claim)
              claim_scores.append(score)
        return (
            sum(claim_scores) / len(claim_scores)
            if claim_scores
            else 0
        )
    ```

1.  最后，我们计算并打印样本数据的整体可信度分数：

    ```py
    groundedness_score = calculate_groundedness(context, answer)
    print(f"Groundedness Score: {groundedness_score:.3f}")
    ```

## 测量检索影响的挑战

现在我们已经看到了一个示例，让我们来看看在 RAG 系统的评估过程中遇到的一些关键挑战：

+   **定义真实情况**：确定基于事实和答案相关性的真实情况可能很困难且具有主观性，尤其是在处理复杂或细微的查询时。

+   **归因错误**：确定生成的响应中的错误是由于检索不良、LLM 的限制，还是两者的结合，可能很困难。

+   **计算成本**：评估检索对生成的影响可能计算成本高昂，尤其是在使用更大的 LLM 或进行人工评估时。

+   **标注者间一致性**：在使用人工评估时，确保在主观判断（如基于事实和相关性）方面有高标注者间一致性可能很困难。

虽然评估 RAG 系统的各个组件（检索和生成）很重要，但评估系统在端到端方式下的整体性能也同样关键。让我们看看下一个方面。

# LLM 中 RAG 系统的端到端评估

虽然评估 RAG 系统的各个组件（检索和生成）很重要，但评估系统在端到端方式下的整体性能也同样关键。端到端评估考虑了整个 RAG 管道，从初始用户查询到最终生成的响应，提供了对系统有效性的整体看法。

让我们看看一些整体指标：

+   **任务成功**：对于面向任务的 RAG 系统（例如，问答、对话），我们可以衡量整体任务成功率。这涉及到确定生成的响应是否成功完成了预期的任务。

    这里有一些评估此指标的技术：

    +   **自动评估**：对于某些任务，我们可以自动评估任务的成功率。例如，在问答中，我们可以检查生成的答案是否与黄金标准答案匹配。

    +   **人工评估**：对于更复杂的任务，可能需要人工评估来判断 RAG 系统是否成功实现了任务目标。

+   **答案质量**：此指标在考虑准确性、相关性、流畅性、连贯性和基于事实性等因素的同时，评估生成响应的整体质量。

    这里有一些评估此指标的技术：

    +   **人工评估**：人类评判员可以使用李克特量表或使用考虑多个质量维度的更详细的标准来评估生成的响应的整体质量。

    +   **自动指标**：虽然回答质量可能难以完全自动化，但可以使用以下指标等来近似回答质量的一些方面：

        +   **ROUGE/BLEU**：衡量生成的响应与参考答案（如果有的话）之间的重叠程度。

        +   **困惑度**：衡量 LLM 预测生成响应的效果（通常困惑度越低越好）。

        +   **基于事实的一致性指标（NLI，基于 QA 的）**：评估响应与检索到的上下文的事实一致性。

        +   **相关性指标**：衡量查询与生成响应之间的相似度。

现在，让我们看看我们可以如何评估 RAG 系统。

## 评估策略

RAG 系统的评估策略可以广泛分为黑盒评估、玻璃盒评估、组件评估和消融研究，每种方法都提供了对系统性能的独特见解。

在黑盒评估中，整个 RAG 系统被视为一个单一单元。评估者提供输入查询，并仅评估最终生成的响应，而不分析中间的检索或生成步骤。这种方法特别适用于衡量整体系统性能和比较不同的 RAG 实现，而无需深入了解其内部机制。

与之相反，玻璃盒评估涉及对 RAG 系统内部工作方式的详细检查。这种方法分析检索到的上下文、LLM 的注意力模式以及中间生成步骤。通过剖析这些元素，玻璃盒评估有助于识别系统的优势和劣势，定位错误来源，并为有针对性的改进提供见解。

更细粒度的方法是组件评估，它分别评估检索和生成组件。检索性能通常使用 Recall@k 和 NDCG 等指标来衡量，而生成文本的质量则使用 BLEU 和 ROUGE 等指标来评估，或者基于一组固定的检索文档进行人工判断。这种方法特别有效于隔离和诊断单个组件中的性能问题。

最后，消融研究提供了一种系统的方法来衡量不同组件对整体系统有效性的影响。通过移除或修改 RAG 系统的特定部分（例如，测试带有和不带有检索的性能或交换不同的检索和生成模型）——研究人员可以更好地理解每个组件如何贡献于系统的功能性和整体成功。

## 端到端评估的挑战

全面评估 RAG 系统存在几个挑战，尤其是在评估检索和生成组件之间的复杂交互时。以下是一些挑战：

+   **定义真实情况**：对于开放性任务或涉及生成复杂响应的任务，定义真实情况可能很困难，甚至不可能

+   **归因错误**：当系统生成错误或不高质量的响应时，可能很难确定错误是否起源于检索或生成组件

+   **计算成本**：端到端评估可能计算成本高昂，尤其是在使用更大的 LLM 或在大规模上进行人工评估时

+   **可重现性**：由于检索和生成组件之间的复杂交互以及生成过程中可能使用的非确定性检索机制或随机解码策略，确保可重现性可能很困难，这可能导致即使在相同输入的情况下，输出也会因运行而异。

接下来，让我们将重点转向人类评估在评估基于 LLM 的 RAG 系统中的作用，它通过捕捉诸如相关性、连贯性和事实准确性等细微方面来补充自动化指标。

# 基于 LLM 的 RAG 的人类评估技术

虽然自动化指标提供了有价值的见解，但人类评估仍然是评估 RAG 系统整体质量和有效性的黄金标准。人类判断对于评估难以用自动化指标捕捉的方面尤其重要，例如检索信息的细微相关性、生成文本的连贯性和流畅性，以及响应在解决用户需求方面的整体有用性。

人类评估者可以评估 RAG 系统性能的各个方面：

+   **相关性**：生成的响应与用户查询的相关性如何？它是否解决了查询中表达的具体信息需求？

+   **扎根性/忠实性**：生成的响应是否由检索到的上下文在事实上得到支持？它是否避免了虚构或与提供的信息相矛盾？

+   **连贯性和流畅性**：生成的响应是否结构良好，易于理解，并且用语法正确且自然的声音书写？

+   **有用性**：考虑到检索到的上下文的限制，响应是否提供了有用且令人满意的答案，以解决用户的查询？

+   **上下文利用**：系统在生成响应时如何有效地利用检索到的上下文？它是否适当地整合和综合来自多个来源的信息？

+   **归属**：系统是否提供了对检索到的上下文中支持生成声明的来源的明确引用或链接？

可以使用几种方法对 RAG 系统进行人类评估：

+   **评分量表（李克特量表）**：标注者根据数值量表（例如 1 到 5）对生成的响应的不同方面（例如相关性、扎根性、流畅性）进行评分，其中 1 代表质量差，5 代表质量优秀：

    +   **优点**：易于实施，易于收集和汇总数据

    +   **缺点**：可能具有主观性，容易受到标注者偏差的影响，并且可能无法捕捉细微的差异

+   **比较评估（排名/最好最差尺度）**：标注者被展示多个针对同一查询的 RAG 系统输出，并要求根据其整体质量或特定标准进行排名。

+   **最好最差尺度**：一种特定的比较评估形式，其中标注者从一组输出中选择最佳和最差选项：

    +   **优点**：比绝对评分更可靠，并能有效捕捉系统之间的相对差异

    +   **缺点**：比评分量表更难实现，需要标注者付出更多努力

+   **基于任务的评估**：要求标注者使用 RAG 系统完成特定任务，例如回答问题、撰写摘要或进行对话。RAG 系统的质量基于标注者成功完成任务的能力以及他们对系统性能的满意度：

    +   **优点**：更真实、以用户为中心，并提供系统实用性的直接度量

    +   **缺点**：设计和实现更复杂，可能耗时且昂贵

+   **自由形式反馈**：标注者对 RAG 系统输出的优点和缺点提供开放式反馈：

    +   **优点**：捕捉详细的见解和建议，并能揭示意外问题

    +   **缺点**：更难分析和量化，可能主观且不一致

## 人类评估的最佳实践

为了确保人类评估的可靠性和公平性，请考虑以下最佳实践：

+   **清晰的指南**：为标注者提供清晰和详细的指南，定义评估标准和标注程序

+   **培训和校准**：对标注者进行任务培训并使用示例标注校准他们的判断

+   **标注者间一致性**：测量标注者间一致性（例如，使用 Cohen 的 Kappa 或 Fleiss 的 Kappa）以确保标注的可靠性

+   **试点研究**：进行试点研究以完善评估方案并在大规模评估启动前识别潜在问题

+   **多个标注者**：为每个项目使用多个标注者以减轻个人偏见并提高评估的稳健性

+   **多样化的标注者群体**：招募多样化的标注者群体以捕捉更广泛的视角并减少潜在的偏见

+   **质量控制**：实施机制以识别和纠正标注中的错误或不一致

## 人类评估的挑战

评估基于 LLM 构建的 RAG 系统的性能面临一系列独特的挑战。在此，我们概述了在执行可靠、一致和有意义的系统人类评估过程中遇到的关键障碍：

+   **成本和时间**：人类评估可能很昂贵且耗时，尤其是对于大规模评估

+   **主观性**：人类判断可能具有主观性，并受个人偏好和偏见的影响

+   **标注者培训和专业知识**：确保标注者得到适当的培训并具备评估 RAG 系统性能所需的必要专业知识可能具有挑战性

+   **可重复性**：由于人类判断固有的可变性，复制人类评估可能很困难

在下一节中，我们将探讨标准化基准和数据集在评估 RAG 系统中的作用，突出关键基准、评估标准和挑战。

# RAG 评估的基准和数据集

标准化的基准和数据集在推动 RAG（阅读理解与生成）研究和发展中起着至关重要的作用。它们为评估和比较不同的 RAG 系统提供了一个共同的基础，促进了识别最佳实践和跟踪随时间进步的过程。

让我们来看看一些关键的基准和数据集：

+   **知识密集型语言任务（KILT）**：一个全面的基准，用于评估知识密集型语言任务，包括问答、事实核查、对话和实体链接：

    +   **数据来源**：基于维基百科，对所有任务采用统一格式

    +   **优点**：提供多样化的任务，允许评估检索和生成，并包括标准化的评估框架

    +   **局限性**：主要基于维基百科，可能无法反映现实世界知识来源的多样性

+   **自然问题（NQ）**：一个大规模的问答数据集，收集自发送到谷歌搜索引擎的真实用户查询：

    +   **数据来源**：包含问题和包含答案的维基百科页面的一对数据

    +   **优点**：现实主义的查询，大规模，包括短答案和长答案的标注

    +   **局限性**：由于它主要关注事实性问题，可能不适合评估更复杂的推理或生成任务

+   **TriviaQA**：包含问题-答案-证据三元组的具有挑战性的问答数据集：

    +   **数据来源**：从 Trivia 爱好者收集，包括网络和维基百科的证据文档

    +   **优点**：比 NQ 更难；它需要阅读和理解多个证据文档

    +   **局限性**：主要关注事实性问题，TriviaQA 问题的写作风格可能不代表现实世界用户的查询

+   **Explain Like I’m Five (ELI5)**：来自 Reddit 论坛“Explain Like I’m Five”的问题和答案数据集，用户在这里寻求对复杂主题的简化解释：

    +   **数据来源**：从 Reddit 收集，包括广泛主题的问题和答案

    +   **优点**：专注于长格式、解释性答案，适合评估 RAG 系统的生成能力

    +   **局限性**：答案的质量和准确性可能有所不同，可能需要仔细筛选或标注

+   **ASQA**：第一个统一模糊问题的长格式问答数据集：

    +   **数据来源**：该数据集是通过结合多个模糊问题从头开始构建的

    +   **优点**：有助于评估长格式问答任务

    +   **局限性**：从头开始构建高质量的数据集可能具有挑战性

+   **微软机器阅读理解（MS MARCO）**：一个大规模的机器阅读理解和问答数据集：

    +   **数据来源**：包含发送到必应搜索引擎的真实匿名用户查询，以及人类生成的答案和相关的段落。

    +   **优势**：它提供了一个大规模、多样化的查询和答案集，包括段落级和全文标注

    +   **局限性**：主要关注抽取式问答，可能不适合评估 RAG 系统的生成能力

+   **斯坦福问答数据集（SQuAD）**：一个广泛使用的阅读理解数据集，由众包工作者在一系列维基百科文章上提出的问题组成：

    +   **数据来源**：包含问题-段落-答案三元组，其中答案是在段落中的文本片段

    +   **优势**：一个大规模、成熟的阅读理解基准

    +   **局限性**：主要关注抽取式问答，可能不适合评估 RAG 系统的生成能力

作为一个例子，让我们说明如何使用 KILT 数据集来评估一个 RAG 系统。我们将使用 Python 中的 KILT 库来完成这项工作：

1.  运行以下代码来安装 kilt 库并导入必要的模块：

    ```py
    pip install kilt==0.5.5
    from kilt import kilt_utils as utils
    from kilt import retrieval
    from kilt.eval import answer_evaluation, provenance_evaluation
    ```

1.  接下来，下载一个特定的 KILT 任务，例如维基百科巫师（WoW）数据集：

    ```py
    # Download the WoW dataset
    utils.download_dataset("wow")
    ```

1.  然后，将下载的数据集加载到内存中：

    ```py
    # Load the dataset
    wow_data = utils.load_dataset("wow", split="test")
    ```

1.  定义一个模拟 RAG 检索组件行为的虚拟 RAG 函数。为了演示目的，它简单地为每个查询返回一组固定的维基百科页面。在实际场景中，你会用你实际的 RAG 检索实现来替换它：

    ```py
    class DummyRetriever(retrieval.base.Retriever):
        def __init__(self, k=1):
              super().__init__(num_return_docs=k)
              self.k = k
        # retrieve some Wikipedia pages (or the entire dataset)
        # based on the query
        def retrieve(self, query, start_paragraph_id=None):
            # Dummy retrieval: return the same set of pages for each query
            dummy_pages = [
                {
                    "wikipedia_id": "534366",
                    "start_paragraph_id": 1,
                    "score": self.k,
                    "text": "Paris is the capital of France."
                },
                {
                    "wikipedia_id": "21854",
                    "start_paragraph_id": 1,
                    "score": self.k-1,
                    "text": "The Mona Lisa was painted by Leonardo da Vinci."
                },
                {
                    "wikipedia_id": "37267",
                    "start_paragraph_id": 1,
                    "score": self.k-2,
                    "text": "Mount Everest is the highest mountain in the world."
                }
              ]
              return dummy_pages[:self.k]
    # Example usage
    retriever = DummyRetriever(k=2)
    ```

1.  定义一个模拟 RAG 生成组件行为的虚拟 RAG 生成函数。为了演示目的，它简单地为每个查询返回一个固定的答案。在实际场景中，你会用你实际的基于 LLM 的生成实现来替换它：

    ```py
         def dummy_generate(query, retrieved_pages):
            """Simulates RAG generation by returning 
            a fixed answer for each query."""
            if "capital of France" in query:
                return "Paris"
            elif "Mona Lisa" in query:
                return "Leonardo da Vinci"
            elif "highest mountain" in query:
                return "Mount Everest"
            else:
                return "I don't know."
    ```

1.  在数据集上运行虚拟 RAG 管道，使用虚拟检索和生成函数，并收集生成的预测：

    ```py
    predictions = []
    for element in wow_data[:10]:
        query = element["input"]
        retrieved_pages = retriever.retrieve(query)
        # Add provenance information to the element
        element["output"] = [{"provenance": retrieved_pages}]
        generated_answer = dummy_generate(query, retrieved_pages)
        # Add the generated answer to the element
        element["output"][0]["answer"] = generated_answer
        predictions.append(element)
    ```

1.  最后，使用 KILT 评估函数评估生成的预测。检索性能（使用`provenance_evaluation`）和答案质量（使用`answer_evaluation`）都会被评估：

    ```py
    kilt_scores = {}
    kilt_scores["provenance_MAP@k"] = \
        provenance_evaluation.get_map_at_k(
        predictions, verbose=False
    )
    kilt_scores["answer_EM"] = answer_evaluation.get_exact_match(
        predictions, verbose=False
    )
    kilt_scores["answer_F1"] = answer_evaluation.get_f1(
        predictions, verbose=False
    )
    kilt_scores["answer_ROUGE-L"] = answer_evaluation.get_rouge_l(
        predictions, verbose=False
    )
    print(kilt_scores)
    ```

这段代码提供了一个如何使用 KILT 框架评估 RAG 系统的基础示例。在实际场景中，你会用你实际的 RAG 检索和生成函数来替换虚拟函数，并使用数据集的更大部分进行评估。你可以通过下载和加载相应的数据集来将此示例适应其他 KILT 任务。

选择基准和数据集时，以下是一些需要考虑的事项：

+   **任务对齐**：选择与你要评估的具体任务对齐的基准和数据集（例如，问答，对话，摘要）

+   **知识领域**：考虑基准覆盖的知识领域。一些基准基于通用知识（例如，维基百科），而其他基准则专注于特定领域（例如，科学文献，医疗记录）

+   **检索设置**：选择适合您所使用的检索设置的基准（例如，开放域检索、封闭域检索、段落检索、文档检索）

+   **生成要求**：考虑任务所需的生成类型（例如，抽取式与抽象式、短答案与长答案）

+   **数据集大小和质量**：确保数据集足够大，以提供具有统计意义的成果，并且数据质量要高（例如，准确的注释和良好的问题格式）

+   **评估指标**：检查基准使用的评估指标，并确定它们是否适合您的特定评估目标

# 摘要

在本章中，我们讨论了评估检索质量和生成性能的广泛指标，包括传统的信息检索指标，如 Recall@k、Precision@k、MRR 和 NDCG，以及更 RAG 特定的指标，如 groundedness、faithfulness 和 answer relevance。我们探讨了测量这些指标的各种技术，包括基于 NLI 和 QA 模型的自动化方法，以及使用评分尺度、比较判断和基于任务的评估方法进行的人评方法。

我们强调了人评在捕捉 RAG 性能细微差别方面的重要作用，这些差别仅用自动化指标难以评估。我们还讨论了设计和进行人评的最佳实践，例如提供明确的指南、培训注释员、测量注释员间的一致性以及进行试点研究。我们需要记住，在现实世界的部署中，自动化与人评之间的权衡将非常重要。

此外，我们探讨了广泛使用的 RAG 评估基准和数据集，包括 KILT、NQ、TriviaQA、ELI5、ASQA、MS MARCO 和 SQuAD，突出了它们的优点和局限性，并提供了针对不同任务和领域的基准选择指南。

总结来说，很明显，评估 RAG 系统是一个复杂且不断发展的领域。更复杂的评估指标的开发、更多样化和更具挑战性的基准的创建以及人评方法的改进将继续对推动 RAG 研究和开发进步至关重要。

在下一章中，我们将探讨 LLMs 中的代理模式，重点关注 LLMs 如何使用高级检索和生成技术自主执行涉及推理、规划和决策的任务。
