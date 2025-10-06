# 11

# 使用 Amazon Bedrock 评估和监控模型

为了使用最适合您生成式 AI 解决方案性能的模型，您需要评估您可用的模型。本章探讨了评估不同模型性能的各种技术。

本章介绍了 Amazon Bedrock 提供的两种主要评估方法：自动模型评估和人工评估。我们将对这两种方法进行详细的讲解。此外，我们还将探讨用于模型评估和评估 RAG 管道的开源工具，如 **基础模型评估**（**FMEval**）和 **RAG 评估**（**Ragas**）。

本章的第二部分将探讨监控。我们将探讨如何利用 Amazon CloudWatch 对模型性能、延迟和令牌计数进行实时监控。我们还将进一步探讨模型调用日志，以捕获模型调用的请求、响应和元数据。此外，我们将强调 Amazon Bedrock 与 AWS CloudTrail 的集成，用于审计 API 调用，以及与 Amazon EventBridge 的集成，用于事件驱动的监控和模型定制作业的自动化。

到本章结束时，您将能够理解如何评估 FMs 并监控其性能。

本章将涵盖以下关键主题：

+   评估模型

+   监控 Amazon Bedrock

# 技术要求

本章要求您拥有 AWS 账户访问权限。如果您还没有，可以访问 [`aws.amazon.com/getting-started/`](https://aws.amazon.com/getting-started/) 创建 AWS 账户。

其次，您需要设置 AWS Python SDK (Boto3)：[`docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html`](https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html)

您可以以任何方式执行 Python 设置：在本地机器上安装，使用 AWS Cloud9，利用 AWS Lambda，或利用 Amazon SageMaker。

注意

与 Amazon Bedrock 的 FMs（功能模型）的调用和定制相关联的费用。请参阅 [`aws.amazon.com/bedrock/pricing/`](https://aws.amazon.com/bedrock/pricing/) 了解更多信息。

# 评估模型

到目前为止，我们已经全面了解了 Amazon Bedrock 的功能，探讨了提示工程、RAG（阅读-回答生成）和模型定制等技术。我们还检查了各种架构设计模式，并分析了不同模型生成的响应。由于 Amazon Bedrock 内部提供了大量的 FMs，因此确定最适合您特定用例和业务需求的选项可能具有挑战性。为了解决这个问题，我们将现在专注于模型评估的主题，以及如何比较不同模型的输出，以选择最适合您应用程序和业务需求的模型。这是实施任何生成式 AI 解决方案的关键初始阶段。

![图 11.1 – 生成式 AI 生命周期](img/B22045_11_01.jpg)

**图 11**.1 – 生成式 AI 生命周期

如**图 11**.1 所示，在定义了您希望通过生成式 AI 解决的具体业务用例之后，**选择**阶段涉及从可用选项中选择潜在模型并对这些候选模型进行严格评估。随后，**负责任的 AI**阶段侧重于确保数据隐私和安全，以及实施负责任模型行为的护栏，这些内容我们将在*第十二章*中介绍。

在讨论模型评估之前，一种快速比较 Amazon Bedrock 的**Chat playground**屏幕上模型响应的方法是使用**比较****模式**切换。

![**图 11**.2 – Bedrock 的 Chat playground 中的比较模式](img/B22045_11_02.jpg)

**图 11**.2 – Bedrock 的 Chat playground 中的比较模式

![**图 11**.3 – 比较模式中的模型指标](img/B22045_11_03.jpg)

**图 11**.3 – 比较模式中的模型指标

如**图 11**.2 所示，您可以在**Chat playground**中启用比较模式并添加最多三个模型以查看模型对同一提示的响应的并排比较。此外，您还可以查看所选每个模型的指标（如**图 11**.3 所示）并比较延迟、输入令牌计数、输出令牌计数和相关的成本。我们将在*监控 Amazon Bedrock*部分深入讨论模型指标。

## 使用 Amazon Bedrock

在 Amazon Bedrock 中，您可以创建模型评估作业来比较用于文本生成、摘要、问答等用例的模型响应。

在 Amazon Bedrock 中对模型进行评估主要包含两种选项：

+   **自动模型评估**

+   **人工评估**

让我们更深入地探讨这两种评估技术。

## **自动模型评估**

使用自动模型评估，后台会运行一个评估算法脚本，在 Amazon Bedrock 提供的**内置数据集**或您提供的**自定义数据集**上执行，用于推荐指标（准确度、毒性、鲁棒性）。让我们通过以下步骤来创建一个自动模型评估作业：

1.  **评估名称**：这指的是选择一个能够准确反映作业目的的描述性名称。此名称应在您特定的 AWS 区域的 AWS 账户中是唯一的。

1.  **模型选择器**：选择您想要评估的模型（如**图 11**.4 所示）。在撰写本书时，自动模型评估是在单个模型上进行的。

![**图 11**.4 – 模型选择器](img/B22045_11_04.jpg)

**图 11**.4 – 模型选择器

在**模型选择器**中，您可以可选地修改推理参数，例如**温度**、**Top P**、**响应长度**等（如**图 11**.5 所示）。您可以通过点击**推理配置**：**默认更新**来获取此屏幕。

修改此推理配置的值将改变模型的输出。要了解更多关于推理配置参数的信息，您可以回到本书的*第二章*。

![图 11.5 – 推理配置](img/B22045_11_05.jpg)

图 11.5 – 推理配置

1.  **任务类型**：在 Amazon Bedrock 模型评估中使用时，目前支持以下任务类型：

    +   通用文本生成

    +   文本摘要

    +   问答

    +   文本分类

1.  **指标和数据集**：Amazon Bedrock 提供了总共三个指标，您可以选择使用这些指标来衡量模型的性能：

    +   **准确性**：将关于现实世界的实际知识编码的能力是生成式 AI 模型的一个关键方面。此指标评估模型生成与既定事实和数据一致输出的能力。它评估模型对主题内容的理解以及其准确综合信息的能力。高准确度分数表明模型的输出是可靠的，可以信赖用于需要事实精确性的任务。

    +   **毒性**：这指的是模型生成有害、冒犯性或不适当内容的能力。此指标衡量模型产生可能被视为不道德、有偏见或歧视性的输出的倾向。评估毒性对于确保 AI 系统的负责任和道德部署至关重要，尤其是在涉及与用户直接互动或向公众传播信息的应用中。

    +   **鲁棒性**：鲁棒性是衡量模型对输入数据中微小的、语义保持性变化的抵抗力的指标。它评估模型在面临输入数据中的轻微变化或扰动时，其输出保持一致和可靠的程度。此指标对于在动态或嘈杂环境中运行的生成式 AI 模型尤为重要，其中输入数据可能受到轻微波动或干扰。鲁棒性强的模型在输入变化较小的情况下，不太可能产生异常或不一致的输出。

文本分类任务支持准确性和鲁棒性指标，而其他任务支持所有三个指标。

对于您选择的每个任务类型和指标，Amazon Bedrock 都为您提供了内置数据集。例如，对于通用文本生成任务类型，您将获得以下内置数据集：

+   TREX: [`hadyelsahar.github.io/t-rex/`](https://hadyelsahar.github.io/t-rex/)

+   BOLD: [`github.com/amazon-science/bold`](https://github.com/amazon-science/bold)

+   WikiText2: [`huggingface.co/datasets/wikitext`](https://huggingface.co/datasets/wikitext)

+   英文维基百科：[`en.wikipedia.org/wiki/Wikipedia:Database_download`](https://en.wikipedia.org/wiki/Wikipedia:Database_download)

+   RealToxicityPrompts: [`github.com/allenai/real-toxicity-prompts`](https://github.com/allenai/real-toxicity-prompts)

要获取基于不同指标和任务类型的内置数据集的完整列表，您可以查阅[`docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-prompt-datasets-builtin.html`](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-prompt-datasets-builtin.html)。

如果您想使用自己的自定义数据集，它需要是*JSON 行* (*.jsonl*)格式。数据集中的每一行都必须是一个有效的 JSON 对象，并且每个评估作业可以包含最多 1,000 个提示。

要构建您的自定义提示数据集，您需要包含以下键：

+   **prompt**：此键是必需的，用作各种任务的输入，例如通用文本生成、问答、文本摘要和分类。根据任务的不同，与此键关联的值将有所不同——它可以是模型需要响应的提示、要回答的问题、要总结的文本或要分类的内容。

+   `referenceResponse`是必需的，用于提供与您的模型输出进行比较的基准响应。对于问答、准确性评估和鲁棒性测试等任务，此键将包含正确答案或预期响应。

+   `category`键。此可选键允许您对提示及其相应的参考响应进行分组，从而实现对模型在不同领域或类别上的性能进行更细致的分析。

为了说明这些键的用法，以下是一个问答任务的示例：

```py
{"prompt":"What is the process that converts raw materials into finished goods?", "category":"Manufacturing", "referenceResponse":"Manufacturing"}
{"prompt":"What is the study of methods to improve workplace efficiency?", "category":"Manufacturing", "referenceResponse":"Industrial Engineering"}
{"prompt":"What is the assembly of parts into a final product?", "category":"Manufacturing", "referenceResponse":"Assembly"}
{"prompt":"A computerized system that monitors and controls production processes is called", "category":"Manufacturing", "referenceResponse":"SCADA"}
{"prompt":"A system that minimizes waste and maximizes efficiency is called", "category":"Manufacturing", "referenceResponse":"Lean Manufacturing"}
```

在此 JSON 行中，`prompt`键包含`什么是将原材料转化为成品的过程？`这个问题，而`referenceResponse`键持有正确答案`Manufacturing`。此外，`category`键设置为`Manufacturing`，允许您将此提示和响应与其他与制造相关的提示和响应分组。

一旦您创建了自定义提示数据集，您需要将数据集文件存储在 Amazon S3 桶中，并在创建模型评估作业时指定正确的 S3 路径（例如**s3://test/data/**）（如图*11.6*所示）。

![图 11.6 – 选择提示数据集](img/B22045_11_06.jpg)

图 11.6 – 选择提示数据集

请注意，S3 桶应附加以下**跨源资源共享**（**CORS**）策略：

```py
[{
"AllowedHeaders": ["*"],
"AllowedMethods": ["GET","POST","PUT","DELETE"],
"AllowedOrigins": ["*"],
"ExposeHeaders": ["Access-Control-Allow-Origin"]
}]
```

CORS 策略是一组规则，指定哪些来源（域名或网站）允许访问 S3 桶。要了解更多关于 CORS 的信息，您可以查看[`docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html`](https://docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html)。

通过精心制作您的自定义提示数据集，您可以确保您的 LLMs 在多种场景下得到全面评估，涵盖各种任务、领域和复杂度级别。

![图 11.7 – 指标和数据集](img/B22045_11_07.jpg)

图 11.7 – 指标和数据集

*图 11.7*展示了通用文本生成任务类型的**指标和数据集**选项。

让我们看看在创建模型评估作业时可以指定的其他参数：

+   **评估结果**：在此，您可以指定评估作业结果应存储的 S3 路径。我们将在下一节中介绍评估结果。

+   **IAM 角色和 KMS 密钥**：执行诸如从 S3 桶访问数据或存储评估结果等操作需要某些权限。以下是一个自动模型评估作业所需的最小策略：

    ```py
    {
    ```

    ```py
        "Version": "2012-10-17",
    ```

    ```py
        "Statement": [
    ```

    ```py
            {
    ```

    ```py
                "Sid": "BedrockConsole",
    ```

    ```py
                "Effect": "Allow",
    ```

    ```py
                "Action": [
    ```

    ```py
                   "bedrock:CreateEvaluationJob",
    ```

    ```py
                   "bedrock:GetEvaluationJob",
    ```

    ```py
                   "bedrock:ListEvaluationJobs",
    ```

    ```py
                   "bedrock:StopEvaluationJob",
    ```

    ```py
                   "bedrock:GetCustomModel",
    ```

    ```py
                   "bedrock:ListCustomModels",
    ```

    ```py
                   "bedrock:CreateProvisionedModel
    ```

    ```py
    Throughput",
    ```

    ```py
                   "bedrock:UpdateProvisionedModel
    ```

    ```py
    Throughput",
    ```

    ```py
                   "bedrock:GetProvisionedModel
    ```

    ```py
    Throughput",
    ```

    ```py
                   "bedrock:ListProvisionedModel
    ```

    ```py
    Throughputs",
    ```

    ```py
                   "bedrock:ListTagsForResource",
    ```

    ```py
                   "bedrock:UntagResource",
    ```

    ```py
                   "bedrock:TagResource"
    ```

    ```py
                ],
    ```

    ```py
                "Resource": "*"
    ```

    ```py
            },
    ```

    ```py
            {
    ```

    ```py
                "Sid": "AllowConsoleS3AccessForModelEvaluation",
    ```

    ```py
                "Effect": "Allow",
    ```

    ```py
                "Action": [
    ```

    ```py
                  "s3:GetObject",
    ```

    ```py
                  "s3:GetBucketCORS",
    ```

    ```py
                  "s3:ListBucket",
    ```

    ```py
                  "s3:ListBucketVersions",
    ```

    ```py
                  "s3:GetBucketLocation"
    ```

    ```py
                ],
    ```

    ```py
                "Resource": "*"
    ```

    ```py
            }
    ```

    ```py
        ]
    ```

    ```py
    }
    ```

您可以在[`docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-security.html?icmpid=docs_bedrock_help_panel_model_evaluation`](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-security.html?icmpid=docs_bedrock_help_panel_model_evaluation)找到有关模型评估作业所需权限的详细信息。

## 模型评估结果

一旦开始模型评估作业，您就可以查看每个指标的评估结果，如图*图 11.8*所示。

![图 11.8 – 评估摘要](img/B22045_11_08.jpg)

图 11.8 – 评估摘要

此外，指标结果存储在您指定的 S3 桶中，如图*图 11.9*所示。

![图 11.9 – S3 桶中的指标结果](img/B22045_11_09.jpg)

图 11.9 – S3 桶中的指标结果

让我们了解每种任务类型是如何进行评估的。

### 文本生成

对于文本生成任务类型，评估的执行方式如下：

+   **准确性**：此指标使用**现实世界知识**（**RWK**）分数进行评估，该分数评估模型理解现实世界的能力。RWK 分数衡量语言模型产生与现实世界事实和常识一致输出的程度。它评估模型对物理世界的推理能力、理解社会规范以及避免生成无意义或矛盾的陈述的能力。高 RWK 分数表示模型表现准确。

+   **鲁棒性**：语义鲁棒性是衡量此任务类型鲁棒性的指标。它使用词错误率来计算，该词错误率量化了模型输出在输入发生轻微、语义保留的扰动时的变化程度。低语义鲁棒性分数表示模型表现良好，因为它对这些扰动具有鲁棒性。

+   **毒性**：此指标使用 detoxify 算法([`github.com/unitaryai/detoxify`](https://github.com/unitaryai/detoxify))计算，该算法衡量模型输出中存在有毒内容的情况。低毒性值表示所选模型不会生成大量有害或冒犯性的内容。

### 文本摘要

对于文本摘要任务类型，评估方式如下：

+   **准确性**：在此任务类型中，使用 BERTScore 来评估准确性。它是通过使用 BERT 模型的预训练上下文嵌入来计算的，并通过余弦相似度匹配候选句和参考句中的单词。

+   **鲁棒性**：此指标以百分比表示，通过计算扰动提示和原始提示的 BERTScore 之差，然后除以原始提示的 BERTScore 并乘以 100 来计算。分数越低，表示所选模型越鲁棒。

+   **毒性**：与一般文本生成任务类型一样，使用 detoxify 算法来计算模型输出的毒性，低值表示生成的毒性内容最少。

### 问答

对于问答任务类型，评估方式如下：

+   **准确性**：在此任务类型中，使用 F1 分数来评估准确性。它是通过将精确度分数（正确预测与所有预测的比率）除以召回率分数（正确预测与相关预测总数的比率）来计算的。F1 分数越高，表示性能越好。

+   **鲁棒性**：此指标以百分比表示，通过计算扰动提示和原始提示的 F1 分数之差，然后除以原始提示的 F1 分数并乘以 100 来计算。分数越低，表示所选模型越鲁棒。

+   **毒性**：与其他任务类型一样，使用 detoxify 算法来计算模型输出的毒性，低值表示生成的毒性内容最少。

### 文本分类

对于文本分类任务类型，评估方式如下：

+   **准确性**：在此任务类型中，通过比较预测类别与其真实标签来计算准确性。准确性分数越高，表示模型根据提供的真实标签正确分类文本的能力越强。

+   **鲁棒性**：此指标以百分比表示，通过计算扰动提示和原始提示的分类准确性分数之差，然后除以原始提示的分类准确性分数并乘以 100 来计算。分数越低，表示所选模型越鲁棒。

让我们分析来自 T-Rex 内置数据集的准确性和鲁棒性指标：([`hadyelsahar.github.io/t-rex/`](https://hadyelsahar.github.io/t-rex/)):

```py
{"automatedEvaluationResult":{"scores":[{"metricName":"Accuracy","result":1},{"metricName":"Robustness","result":0.0828}]},"inputRecord":{"prompt":"al-Hira is the capital of","referenceResponse":"Lakhmid"},"modelResponses":[{"response":" Al-Hira was an ancient city in Mesopotamia located south of what is now Kufa in modern-day Iraq. It served as the capital of the Arab Lakhmids from the 3rd to the 6th century CE.\n\nSome key facts about al-Hira:\n\n- It was founded in the 3rd century CE and served as the capital of the Lakhmids, who were Arab vassals of the Sassanian Persian Empire. \n\n- Under the leadership of the Lakhmid kings, al-Hira flourished as a center of Arab culture and politics in pre-Islamic Arabia. It was a major center of Nestorian Christianity in the region.\n\n- The city was located along important trade routes linking Mesopotamia to the Arabian peninsula. Its markets attracted merchants from throughout Arabia and beyond.\n\n- Al-Hira and the Lakhmids were defeated by the Sassanians in 602 CE. The city subsequently went into decline and was eventually abandoned. \n\n- The ruins of al-Hira are located about 94 km south of modern-day Baghdad, Iraq. Sections of its walls and some buildings have been excavated by archaeologists.\n\nSo in summary, al-Hira functioned as the capital of the Arab Lakhmids kingdom for several centuries until its downfall in the early 7th century CE. Its ruins stand as an","modelIdentifier":"anthropic.claude-v2"}]}
```

输出显示，提供了一个提示（`al-Hira 是首都`）给 Anthropic Claude v2 模型，并且该模型的响应（`Al-Hira 是美索不达米亚的一个古老城市...`）被评估为参考响应（`Lakhmid`）。评估计算了诸如准确性和鲁棒性等指标的分数，为模型在此特定输入上的性能提供了见解。

## 使用人工评估

人工评估允许您将人工输入纳入评估过程，因此模型不仅准确，而且符合现实世界的期望和要求。有人工评估有两种类型：

+   带自己的工作团队

+   使用 AWS 管理的团队

### 带自己的工作团队

与自动模型评估类似，当您选择使用您自己的工作团队进行人工评估时，Amazon Bedrock 会引导您通过一个简单的设置过程，让您选择要评估的模型、任务类型（例如，文本摘要）和评估指标。它还会向您展示如何上传您定制的提示数据集。让我们考虑通过自己的团队设置人工评估的逐步过程：

1.  **评估名称**：选择一个能够准确反映工作目的的描述性名称。此名称应在特定 AWS 区域内的 AWS 账户中是唯一的。除了名称外，您还可以选择性地提供描述和标签。

1.  **模型选择器**：选择您想要评估的模型（如图*图 11.10*所示）。在撰写本书时，使用您自己的团队进行人工模型评估只能针对最多两个模型进行。在模型选择器中，您可以选择性地修改推理参数，如温度、Top P、响应长度等。

![图 11.10 – 模型选择器](img/B22045_11_10.jpg)

图 11.10 – 模型选择器

1.  **任务类型**：目前，在此模式下支持以下任务类型：

    +   通用文本生成

    +   文本摘要

    +   问答

    +   文本分类

    +   自定义

    这些任务类型中的最后一个，自定义，允许您指定人类工作者可以使用自定义评估指标。

    根据任务类型，您将看到您必须从其中选择的评估指标和评分方法的列表，如图*图 11.11*和*11.12*所示。

![图 11.11 – 评估指标](img/B22045_11_11.jpg)

图 11.11 – 评估指标

![图 11.12 – 评分方法选项](img/B22045_11_12.jpg)

图 11.12 – 评分方法选项

1.  **指定路径**：接下来，您需要指定自定义提示数据集的 s3 路径。正如我们在前面的子节中看到的，自定义提示数据集需要是*.jsonl*格式。以下是一个自定义提示数据集的示例：

    ```py
    {"prompt":"What is the process that converts raw materials into finished goods?", "category":"Manufacturing", "referenceResponse":"Manufacturing"}
    ```

    ```py
    {"prompt":"What is the study of methods to improve workplace efficiency?", "category":"Manufacturing", "referenceResponse":"Industrial Engineering"}
    ```

    ```py
    {"prompt":"What is the assembly of parts into a final product?", "category":"Manufacturing", "referenceResponse":"Assembly"}
    ```

    请注意，您的数据集的 s3 路径需要您将**跨源资源共享（CORS）**设置配置为如图*图 11.13*所示。

![图 11.13 – CORS 策略窗口](img/B22045_11_13.jpg)

图 11.13 – CORS 策略窗口

要了解更多信息，您可以访问[`docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-security-cors.html`](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-security-cors.html)。

1.  **IAM 角色和 KMS 密钥**：执行诸如从 S3 存储桶访问数据或存储评估结果等操作需要特定的权限。您可以在[`docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-security.html?icmpid=docs_bedrock_help_panel_model_evaluation`](https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-security.html?icmpid=docs_bedrock_help_panel_model_evaluation)找到有关模型评估作业所需的 IAM 权限的更多详细信息。

1.  **设置工作团队**：接下来，您需要设置一个工作团队（如图 11.14 所示）。这包括邀请适当的团队成员。控制台为您提供了邀请新工作者和现有工作者的示例电子邮件模板，您可以在发送邀请时作为参考。工作者会收到一个链接，链接到私人工作者门户，在那里他们完成标记任务。

![图 11.14 – 设置自定义工作团队](img/B22045_11_14.jpg)

图 11.14 – 设置自定义工作团队

1.  接下来，您需要将任务的说明（如图 11.15 所示）告知工作者。这些说明将在工作者执行标记任务的私人工作者门户中可见。

![图 11.15 – 向人工工作者提供说明](img/B22045_11_15.jpg)

图 11.15 – 向人工工作者提供说明

在您审查并创建作业后，人工工作者团队将收到一封电子邮件，以及执行任务的门户链接。一旦工作者完成任务，Amazon Bedrock 将提供评估报告卡。让我们详细了解报告卡：

+   `1`到`5`。在使用此方法时，提供明确的说明以定义每个评分点的含义至关重要。例如，评分`1`可能表示较差或不相关的响应，而评分`5`可能表示优秀且高度相关的输出。结果将以直方图的形式展示，展示评分在数据集中的分布情况：

    +   明确定义刻度点（例如，1 = 差，2 = 一般，3 = 好，4 = 非常好，5 = 优秀）

    +   为评估者提供明确的指南，说明如何解释和应用刻度

    +   以直方图的形式展示结果，便于轻松视觉解释评分分布

+   **选择按钮**：当您选择选择按钮方法时，评估者会看到两个模型响应并要求选择他们偏好的选项。这种方法在比较多个模型在相同任务上的性能时特别有用。结果通常以百分比的形式报告，表示评估者对每个模型偏好的响应比例。

+   `1`（最优先）。这种方法提供了对不同模型相对性能的更细致的理解。结果以直方图的形式展示，显示了数据集中排名的分布。

+   **点赞/踩**：评估者对每个响应进行接受或不接受的评分。最终报告展示了每个模型获得**点赞**评分的响应百分比，从而可以简单地评估其接受度。

另一种人类评估方法是通过 AWS 管理的工作团队。

### 使用 AWS 管理的工作团队

如果您选择 AWS 管理的团队，您可以简单描述您的模型评估需求，包括任务类型、所需的专家级别和大约的提示数量。基于这些详细信息，AWS 专家将随后联系，详细讨论您的项目需求，提供符合您特定需求的定制报价和项目时间表。

*图 11.16* 展示了如何创建一个包含任务类型和所需专家级别等所有详细信息的管理工作团队。

![图 11.16 – AWS 管理的模型评估团队](img/B22045_11_16.jpg)

图 11.16 – AWS 管理的模型评估团队

当您不想管理或分配任务给您的员工，并且需要 AWS 团队代表您进行评估时，AWS 管理的工作团队非常有用。

除了使用 Bedrock 的模型评估作业外，还有其他开源技术可用于模型评估，例如 `fmeval` 和 Ragas。

### FMEval

FMEval 是 AWS 提供的开源库，您可以在 [`github.com/aws/fmeval`](https://github.com/aws/fmeval) 访问。

此库能够全面评估 LLM 在准确性、毒性、语义鲁棒性和提示刻板印象等各个方面。它提供了一系列针对评估 LLM 在不同任务上性能的算法，确保对其能力和局限性的全面理解。

如果您计划使用自己的数据集进行评估，您需要配置一个 `DataConfig` 对象，如下面的代码块所示。此对象指定数据集名称、URI 和 MIME 类型，以及输入提示、目标输出和其他相关列的位置。通过自定义 `DataConfig` 对象，您可以调整评估过程以适应您特定的数据集和任务需求：

```py
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
fmconfig = DataConfig(
    dataset_name="dataset",
    dataset_uri="dataset.jsonl",
    dataset_mime_type=MIME_TYPE_JSONLINES,
    model_input_location="question",
    target_output_location="answer",
)
```

该库提供了一个灵活的`ModelRunner`接口，允许与 Amazon Bedrock 无缝集成，并用于对模型进行调用。以下代码块展示了如何执行调用：

```py
bedrock_model_runner = BedrockModelRunner(
    model_id='anthropic.claude-v2',
    output='completion',
    content_template='{"prompt": $prompt, "max_tokens_to_sample": 500}'
)
```

如果您想了解更多关于`fmeval`的信息，您可以访问[`github.com/aws/fmeval/tree/main`](https://github.com/aws/fmeval/tree/main)。

此外，您还可以尝试使用 Amazon Bedrock 进行`fmeval`。以下是一些您可以测试的 Anthropic Claude v2 的示例：

+   [`github.com/aws/fmeval/blob/main/examples/bedrock-claude-factual-knowledge.ipynb`](https://github.com/aws/fmeval/blob/main/examples/bedrock-claude-factual-knowledge.ipynb)

+   [`github.com/aws/fmeval/blob/main/examples/bedrock-claude-summarization-accuracy.ipynb`](https://github.com/aws/fmeval/blob/main/examples/bedrock-claude-summarization-accuracy.ipynb)

### Ragas

Ragas 是一个框架，旨在评估您的 RAG 管道的性能，这些管道结合了语言模型和外部数据源以增强其输出。它提供了基于最新研究的实用工具，用于分析您语言模型生成的文本，为您提供有关 RAG 管道有效性的宝贵见解。

下面是使用 Ragas 的一些关键功能和好处：

+   **自动评估指标**：Ragas 提供了一套针对评估 RAG 生成文本质量的自动化指标。这些指标超越了传统的度量，如困惑度和 BLEU，提供了对生成输出连贯性、相关性和事实准确性的更深入理解。

+   **可定制的评估策略**：认识到每个 RAG 管道都是独特的，Ragas 允许灵活和可定制的评估策略。您可以调整评估过程以适应您的特定用例、数据领域和性能要求。

+   **与 CI/CD 管道的集成**：Ragas 旨在与**CI/CD**（**持续集成和持续部署**）管道集成。这种集成使您可以持续监控和评估 RAG 管道的性能，确保任何偏差或回归都能得到及时检测和处理。

+   **可解释的见解**：Ragas 生成可解释和可操作的见解，突出您 RAG 管道表现优异的领域，并识别潜在的弱点或瓶颈。这些见解可以指导您的优化工作，帮助您迭代地改进和增强管道的性能。

Ragas 提供了一系列您可以导入的指标。以下是操作方法：

```py
from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
)
from ragas.metrics.critique import harmfulness
metrics = [
    faithfulness,
    context_recall,
    context_precision,
    harmfulness,
]
```

这些指标可以传递给 Ragas 中的`evaluate`函数，以及 Bedrock 模型和嵌入。以下是操作方法：

```py
from ragas import evaluate
results = evaluate(
    df["eval"].select(range(3)),
    metrics=metrics,
    llm=bedrock_model,
    embeddings=bedrock_embeddings,
)
results
```

在前面的代码片段中，`df`假设是一个包含您想要评估的数据的 pandas DataFrame。请注意，`llm=bedrock_model`和`embeddings=bedrock_embeddings`是我们先前创建的 Bedrock 和嵌入模型实例。

对于如何使用 Ragas 与 Amazon Bedrock 的完整教程，您可以访问 [`docs.ragas.io/en/stable/howtos/customisations/aws-bedrock.html`](https://docs.ragas.io/en/stable/howtos/customisations/aws-bedrock.html)。

现在我们已经看到了执行 Amazon Bedrock 模型评估的各种技术，让我们看看与 Amazon Bedrock 集成的监控和日志记录解决方案。

# 监控 Amazon Bedrock

监控您的生成式 AI 应用程序的性能和用法对于确保最佳功能、维护安全和隐私标准以及获取未来改进的见解至关重要。Amazon Bedrock 与 Amazon CloudWatch、CloudTrail 和 EventBridge 无缝集成，提供全面的监控和日志记录解决方案。

## Amazon CloudWatch

Amazon CloudWatch 是一种监控和可观察性服务，它收集并可视化来自各种 AWS 资源的数据，包括 Amazon Bedrock。通过利用 CloudWatch，您可以获得有关 Bedrock 模型性能的宝贵见解，以便您可以主动识别和解决问题。通过 CloudWatch，您可以跟踪使用指标并构建定制的仪表板以进行审计，确保在整个 AI 模型开发过程中保持透明度和问责制。

使用 Amazon Bedrock 的一个关键特性是，您可以深入了解单个账户内多个账户和 FM 的模型使用情况。您可以监控关键方面，如模型调用和令牌计数，以便您可以做出明智的决定并有效地优化资源分配。如果您想在一个或多个区域中配置跨多个账户的监控，您可以在 [`docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-Cross-Account-Methods.html`](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch-Cross-Account-Methods.html) 查看 Amazon CloudWatch 文档。

这提供了您需要采取的所有步骤来启用此功能。

此外，Bedrock 提供了一个名为模型 *调用日志* 的功能。此功能允许用户收集账户内所有模型调用的元数据、请求和响应。虽然此功能默认禁用，但您可以通过访问 Bedrock 控制台中的 **设置** 并切换 **模型调用日志** 来轻松启用它。通过启用此功能，您允许 Bedrock 发布调用日志，以增强可见性和分析。

让我们看看如何利用 CloudWatch 在接近实时的情况下监控 Bedrock，利用指标和日志在预定义阈值超过时触发警报并启动操作。

## Bedrock 指标

Amazon Bedrock 的 CloudWatch 度量涵盖了广泛的性能指标，包括调用次数、调用延迟、调用客户端和服务器错误、调用节流实例、输入和输出令牌等。您可以在[`docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cw.html#runtime-cloudwatch-metrics`](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cw.html#runtime-cloudwatch-metrics)查看支持的度量标准的完整列表。

使用这些度量标准，您可以比较不同模型的延迟，并测量令牌计数，以帮助购买预配吞吐量，以及检测和警报节流事件。

当您使用聊天游乐场时，您可以在运行提示后查看这些度量标准，如图**11.17**所示。

![图 11.17 – 模型度量](img/B22045_11_17.jpg)

图 11.17 – 模型度量

此外，您还可以定义度量标准，这允许您为模型度量提供特定的条件或阈值。根据您的需求，您可以设置如**延迟小于 100ms**或**输出令牌计数大于 500**等标准。这些标准可以用来评估和比较不同模型与您期望的度量标准之间的性能。在比较多个模型时，设置度量标准有助于确定哪些模型满足或未满足您指定的条件，这有助于选择最适合您用例的最合适的模型。

让我们也在 CloudWatch 度量仪表板中查看这些度量标准。

![图 11.18 – CloudWatch 度量仪表板](img/B22045_11_18.jpg)

图 11.18 – CloudWatch 度量仪表板

在**图 11.18**中，您可以查看 Anthropic Claude 3 Sonnet 模型的 CloudWatch 度量：**调用次数**（样本计数）、**调用延迟**（以毫秒为单位）、**输出令牌计数**（样本计数）和**输入令牌计数**（样本计数）。让我们了解这些术语（指标和统计数据）的含义：

+   **样本计数**：这个统计数据表示在指定时间段内记录的总数据点或观察值。

+   在给定时间段内，`Converse`、`ConverseStream`、`InvokeModel`和`InvokeModelWithResponseStream` API。

+   **调用延迟**：此度量标准指的是从调用请求发出到收到响应之间的延迟或时间量。

**输出令牌计数**和**输入令牌计数**是分析并计算模型调用成本时的有用度量标准。令牌本质上是从输入提示和响应中提取的一小组字符。**输出令牌计数**表示模型提供的响应中令牌的总数，而**输入令牌计数**表示提供给模型的输入和提示中令牌的总数。

为了简化监控和分析，Bedrock 的日志和指标可以通过 CloudWatch 仪表板以单一视图呈现。这些仪表板提供了对相同 KPIs 的全面概述，包括按模型随时间变化的调用次数、按模型的调用延迟以及输入和输出的令牌计数。以下图显示了两个模型，Anthropic Claude v2 和 Anthropic Claude v3 Sonnet，在一周时间框架内的仪表板视图。

![图 11.19 – CloudWatch 仪表板](img/B22045_11_19.jpg)

图 11.19 – CloudWatch 仪表板

对于拥有多个 AWS 账户的组织，Bedrock 支持 CloudWatch 跨账户可观察性，允许在监控账户中创建丰富的跨账户仪表板。此功能确保了跨各种账户的性能指标集中视图，便于更好的监督和决策。

## 模型调用日志

模型调用日志允许您捕获和分析模型生成的请求和响应，以及所有调用调用的元数据。它提供了对您的模型如何被使用的全面视图，使您能够监控其性能，识别潜在问题，并优化其使用。

启用模型调用日志是一个简单的过程。您可以通过 Amazon Bedrock 控制台或通过 API 进行配置。

![图 11.20 – 启用模型调用日志](img/B22045_11_20.jpg)

图 11.20 – 启用模型调用日志

*图 11.20*显示了 Amazon Bedrock 控制台中**模型调用日志**的控制台视图。您需要启用此功能。第一步是选择您想要记录的数据类型，例如文本、图像或嵌入。接下来，您需要选择日志的目的地，可以是 Amazon S3、Amazon CloudWatch Logs 或两者，并提供路径。

如果您选择 Amazon S3，您的日志将以压缩的 JSON 文件形式存储，每个文件包含一批调用记录。这些文件可以使用 Amazon Athena 进行查询，或发送到各种 AWS 服务，如 Amazon EventBridge。另一方面，如果您选择 Amazon CloudWatch Logs，您的调用日志将以 JSON 事件的形式发送到指定的日志组。这允许您利用 CloudWatch 日志洞察功能，实时查询和分析您的日志。

模型调用日志的一个关键优势在于其能够捕获大量输入和输出数据。对于超过 100 KB 的数据，或者二进制格式（例如，图像、音频等）的数据，Amazon Bedrock 会自动将其上传到您指定的 Amazon S3 存储桶。这确保了即使对于大量或非文本数据，也不会丢失任何有价值的信息。

通过利用此功能，您可以优化您的模型，识别潜在问题，并确保您的系统高效且有效地运行。

下面是 CloudWatch 日志控制台中模型调用日志的一个示例：

```py
{
    "schemaType": "ModelInvocationLog",
    "schemaVersion": "1.0",
    "timestamp": "2024-06-01T02:26:35Z",
    "accountId": "123456789012",
    "identity": {
        "arn": "arn:aws:sts::123456789012:assumed-role/Xyz/Abc"
    },
    "region": "us-east-1",
    "requestId": "9e0ff76a-7cac-67gg-43rg-5g643qwer85r",
    "operation": "ConverseStream",
    "modelId": "anthropic.claude-v2",
    "input": {
        "inputContentType": "application/json",
        "inputBodyJson": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "Write a poem on stock market"
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 2048,
                "temperature": 0.5,
                "topP": 1,
                "stopSequences": [
                    "\n\nHuman:"
                ]
            },
            "additionalModelRequestFields": {
                "top_k": 250
            }
        },
        "inputTokenCount": 15
    },
    "output": {
        "outputContentType": "application/json",
        "outputBodyJson": {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Here is a poem about the stock market:\n\nThe Stocks Go Up and Down\n\nThe stocks go up and the stocks go down\nGains and losses all around\nSome days are green, some days are red\nWondering where this rollercoaster will lead\n\nBuy low and sell high, that's what they say\nBut the market has a mind of its own each day\nOne wrong move and your profits fade away\nPatience and research are the prudent way\n\nBulls charge ahead with optimism bright  \nWhile bears retreat in a fearful plight\nAnalysts and investors try to read the signs\nOf economic trends and corporate lines\n\nThe risky trader seeks a quick buck\nWhile the long-term holder trusts in luck \nDay by day the tickers rise and fall\nAs we check our portfolios, hoping they won't stall\n\nSo place your bets and say your prayers\nThe market gods will judge what's fair\nBut one truth will always remain\nIn the stock market, uncertainty reigns"
                        }
                    ]
                }
            },
            "stopReason": "end_turn",
            "metrics": {
                "latencyMs": 8619
            },
            "usage": {
                "inputTokens": 15,
                "outputTokens": 218,
                "totalTokens": 233
            }
        },
        "outputTokenCount": 218
    }
}
```

前面的日志片段提供了对`anthropic.claude-v2`模型的`ConverseStream`对话请求。它捕获了各种数据点，如输入提示、输出响应、性能指标和用量统计。通过这种全面的日志记录，您可以有效地分析和评估模型的能力和行为。

## AWS CloudTrail

AWS CloudTrail 是一种合规性和审计服务，允许您捕获和分析您 AWS 环境中所有 API 调用。以下是您如何利用 CloudTrail 获得宝贵见解的方法。

Amazon Bedrock 与 AWS CloudTrail 无缝集成，将每个 API 调用作为一个事件捕获。这些事件包括从 Amazon Bedrock 控制台发起的操作，以及通过 Amazon Bedrock API 操作进行的程序调用。在 CloudTrail 中，您可以获得关于谁发起请求、源 IP 地址、时间戳以及与请求相关的其他详细信息的全面记录。

Amazon Bedrock 使用 CloudTrail 记录两类不同的事件：数据事件和管理事件。当涉及到数据事件时，CloudTrail 默认不将 Amazon Bedrock Runtime API 操作（`InvokeModel`和`InvokeModelWithResponseStream`）记录为数据事件。然而，它确实记录了与 Amazon Bedrock Runtime API 操作相关的所有代理操作，这些操作被归类为数据事件：

+   要记录`InvokeAgent`调用，您需要在 CloudTrail 跟踪上配置高级事件选择器以记录`AWS::Bedrock::AgentAlias`资源类型的数据事件。

+   要记录`Retrieve`和`RetrieveAndGenerate`调用，请配置高级事件选择器以记录`AWS::Bedrock::KnowledgeBase`资源类型的数据事件。

高级事件选择器允许创建针对监控和管理与管理和数据事件相关的 CloudTrail 活动的精确和细粒度过滤器。

**数据事件**提供了对资源操作（如读取或写入 Amazon Bedrock 知识库或代理别名等资源）的洞察。由于数据量较大，这些事件默认不记录，但可以通过高级事件选择器启用记录。

另一方面，**管理事件**捕获控制平面操作，例如创建、更新或删除 Amazon Bedrock 资源的 API 调用。CloudTrail 自动记录这些管理事件，为您的 Amazon Bedrock 环境中的管理活动提供全面的审计跟踪。

如果您想了解更多关于 CloudTrail 的信息，请查看 AWS 文档：[`docs.aws.amazon.com/awscloudtrail/latest/userguide/how-cloudtrail-works.html`](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/how-cloudtrail-works.html)。

结合 AWS CloudTrail 和 Amazon Bedrock 可为您提供强大的审计和监控解决方案。通过捕获和分析 API 调用，您可以在 Amazon Bedrock 环境中保持可见性，确保遵守最佳实践，并迅速解决任何潜在的安全或运营问题。

## EventBridge

Amazon EventBridge 提供了一种在近乎实时的情况下跟踪和响应事件的解决方案。它充当一个中央事件总线，从包括 Amazon Bedrock 在内的各种来源摄取和处理状态变化数据。每当您启动的模型定制作业的状态发生变化时，Bedrock 会向 EventBridge 发布一个新事件。此事件包含有关作业的详细信息，例如其当前状态、输出模型 ARN 以及任何失败消息。

这是您如何利用 Amazon EventBridge 的力量有效地监控 Amazon Bedrock 事件的方法：

+   **事件流和交付**：每当您启动的模型定制作业中发生状态变化时，Amazon Bedrock 会尽力发出事件。这些事件被流式传输到 Amazon EventBridge，它充当一个中央事件总线，从各种 AWS 服务和外部来源摄取和处理事件数据。

+   **事件模式匹配**：在 Amazon EventBridge 中，您可以根据特定的标准（如源服务、事件类型或作业状态）创建定义事件模式的规则。通过制定满足您需求的规则，您可以过滤和捕获仅与您的 Amazon Bedrock 工作流相关的事件。

+   **自动响应和集成**：一旦事件与您定义的规则匹配，Amazon EventBridge 就会将它路由到您指定的一个或多个目标。这些目标可以是各种 AWS 服务，例如 AWS Lambda 函数、Amazon **简单队列服务**（**SQS**）队列或 Amazon **简单通知服务**（**SNS**）主题。凭借这种灵活性，您可以根据事件数据触发自动化操作、调用下游工作流或接收通知。

+   **监控和警报**：Amazon EventBridge 的一个常见用例是为关键事件设置警报机制。例如，您可以配置一个规则，在模型定制作业失败时向指定的地址发送电子邮件通知，使您能够及时调查并解决问题。

+   **事件数据丰富**：Amazon Bedrock 发出的事件数据包含有关模型定制作业的宝贵信息，例如作业 ARN、输出模型 ARN、作业状态以及（如果适用）失败消息。通过利用这些数据，您可以构建针对特定需求的强大监控和警报系统。

要通过 Amazon EventBridge 接收和处理 Amazon Bedrock 事件，您需要创建 *规则* 和 *目标*。规则定义要匹配的事件模式，而目标指定当事件匹配规则时要采取的操作。让我们更深入地了解如何做到这一点：

+   要创建一个规则，请遵循以下步骤：

    1.  打开 Amazon EventBridge 控制台。

    1.  选择**创建规则**。

    1.  为您的规则提供一个名称。

    1.  选择**事件模式**，如图*图 11**.21*所示。

    1.  定义事件模式以匹配 Amazon Bedrock 事件（例如，将**source**设置为**aws.bedrock**，将**detail-type**设置为**模型定制作业** **状态变更**）。

![图 11.21 – 事件模式窗口](img/B22045_11_21.jpg)

图 11.21 – 事件模式窗口

+   按照以下步骤配置目标：

    1.  选择目标类型（例如，AWS Lambda、Amazon SNS、Amazon SQS）。

    1.  指定目标资源（例如，Lambda 函数 ARN 或 SNS 主题 ARN）。

    1.  可选地，为目标添加额外的配置或转换。

一个实际用例是在您的模型定制作业状态发生变化时接收电子邮件通知。以下是您如何设置它的方法：

1.  创建一个 Amazon SNS 主题。

1.  将您的电子邮件地址订阅到 SNS 主题。

1.  创建一个具有以下事件模式的 Amazon EventBridge 规则：

    ```py
       ```

    ```py

    ```

    {

    ```py

    ```

    "source": ["aws.bedrock"],

    ```py

    ```

    "detail-type": ["模型定制作业状态变更"]

    ```py

    ```

    }

    ```py

    ```

    ```py
    ```

1.  将 SNS 主题设置为规则的目标。

使用此设置，每当您的 Amazon Bedrock 模型定制作业状态发生变化时，您都会收到电子邮件通知，让您了解作业进度和潜在故障。

此外，Amazon EventBridge 与 Amazon Bedrock 的集成还打开了各种高级用例，例如以下内容：

+   根据作业事件触发 Lambda 函数执行自定义操作（例如，向 Slack 频道发送通知、更新仪表板或触发下游工作流程）

+   将 Amazon Step Functions 与 Amazon Bedrock 集成，根据作业事件编排复杂的工作流程

+   将作业事件发送到 Amazon Kinesis 数据流进行实时处理和分析

+   将作业事件存档到 Amazon S3 或 Amazon CloudWatch 日志中，用于审计和合规性目的

通过利用 Amazon EventBridge 来监控和响应 Amazon Bedrock 事件，您可以增强机器学习操作的安全性、自动化和可见性。凭借定义自定义规则和集成各种 AWS 服务的能力，您可以创建一个符合您特定需求的强大且安全的环境。

# 摘要

在本章中，我们学习了各种评估和监控 Amazon Bedrock 模型的方法。

我们首先探讨了 Amazon Bedrock 提供的两种主要模型评估方法：自动模型评估和人工评估。自动模型评估过程涉及在内置或自定义数据集上运行评估算法脚本，评估准确度、毒性、鲁棒性等指标。另一方面，人工评估将人类输入纳入评估过程，确保模型不仅提供准确的结果，而且这些结果与实际世界的期望和要求相一致。

此外，我们还讨论了诸如`fmeval`和 Ragas 等开源工具，这些工具提供了针对 LLMs 和 RAG 管道的特定评估能力。

接下来，我们讨论了监控部分，我们讨论了如何利用 Amazon CloudWatch 来获取有关模型性能、延迟和令牌计数的宝贵见解。我们探讨了 Amazon Bedrock 提供的各种指标，并考虑了如何通过 CloudWatch 仪表板进行可视化和监控。此外，我们还介绍了模型调用日志，这是一个强大的功能，允许您捕获和分析所有模型调用的请求、响应和元数据。接下来，我们探讨了 Amazon Bedrock 与 AWS CloudTrail 和 EventBridge 的集成。CloudTrail 提供了您 AWS 环境中 API 调用的全面审计跟踪，使您能够监控并确保遵守最佳实践。另一方面，EventBridge 允许您在近乎实时的情况下跟踪和响应事件，使基于模型定制作业状态变化的自动化响应和集成成为可能。

在亚马逊，确保安全和隐私是首要任务，在今天的数字景观中也是如此。在下一章中，我们将探讨如何在 Amazon Bedrock 中确保安全和隐私。
