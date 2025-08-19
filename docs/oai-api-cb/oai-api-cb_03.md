

# 第三章：理解关键参数及其对生成响应的影响

在上一章中，我们学到了 OpenAI API 不仅仅是一个端点，它还是一个由多个端点组成的集合。这些端点通过`model`和`messages`触发——我们主要看到的是如何通过改变消息参数来影响生成的响应。然而，还有许多可选参数影响 API 的行为，比如温度、N 和最大令牌数。

在本章中，我们将探索这些可选的关键参数，并理解它们如何影响生成的响应。**参数**就像你在复杂机器上找到的旋钮和按钮。通过调整这些旋钮和按钮，你可以根据自己的喜好改变机器的行为。同样，在 ChatGPT 的领域中，参数允许我们调整模型行为的细节，影响它如何处理输入并生成输出。每个参数在塑造 OpenAI 的响应中扮演着独特的角色。

在本章结束时，你将知道如何调整这些参数以更好地满足你的特定需求，了解它们如何影响输出的质量、长度和风格，并学会如何有效使用它们以获得最理想的结果。学习这些内容很重要，因为随着我们开始将 API 集成到智能应用程序的不同用例中，这些参数需要进行调整，理解生成的响应如何随这些参数变化，将帮助我们确定正确的设置。

具体来说，我们将介绍以下教程，每个教程将聚焦一个关键参数：

+   更改模型参数并理解其对生成响应的影响

+   使用 n 参数控制生成响应的数量

+   使用温度参数来确定生成响应的随机性和创造性

# 技术要求

本章中的所有教程都要求你能够访问 OpenAI API（通过生成的 API 密钥），并安装了 API 客户端，如 Postman。你可以参考*第一章*中的教程*使用 Postman 发起 OpenAI API 请求*，了解如何获取你的 API 密钥并设置 Postman。

# 更改模型参数并理解其对生成响应的影响

在*第一章*和*第二章*中，聊天完成请求是使用模型参数和消息参数发起的，其中`model`始终等于`gpt-3.5-turbo`的值。我们基本上忽略了模型参数。然而，这个参数可能是所有参数中对生成响应影响最大的。与普遍看法相反，OpenAI API 不仅仅是一个模型；它由多个不同能力和价格点的模型组成。

在这个配方中，我们将介绍两个主要模型（*GPT-3.5* 和 *GPT-4*），学习如何更改 `model` 参数，并观察这两个模型生成的响应有何不同。

## 准备工作

确保你有一个具有可用使用额度的 OpenAI 平台账户。如果没有，请参阅 *第一章* 中的 *设置 OpenAI Playground 环境* 配方。

此外，请确保你已安装 Postman，已创建一个新的工作区，已创建一个新的 HTTP 请求，并且该请求的 `Headers` 配置正确。这非常重要，因为如果没有正确配置 `Authorization`，你将无法使用 API。如果你没有按照上述步骤安装和配置 Postman，请参阅 *第一章* 中的 *使用 Postman 发送 OpenAI API 请求* 配方。 如果你记不起来了，接下来的 *步骤 1–4* 会解释配置过程。

本章中的所有配方都有相同的要求。

## 如何操作…

1.  在你的 Postman 工作区，选择左上角菜单栏中的 **New** 按钮，然后从弹出的选项中选择 **HTTP**。这将创建一个新的 **Untitled Request**。

1.  通过选择 **Method** 下拉菜单（默认设置为 **GET**），将 HTTP 请求类型从 **GET** 更改为 **POST**。

1.  输入以下 URL 作为聊天完成的端点：[`api.openai.com/v1/chat/completions`](https://api.openai.com/v1/chat/completions)。

1.  在子菜单中选择 **Headers**，并将以下键值对添加到下方的表格中：

| *Key* | *Value* |
| --- | --- |
| `Content-Type` | `application/json` |
| `Authorization` | `Bearer <your API` `key here>` |

在子菜单中选择 **Body**，然后选择 **raw** 作为请求类型。输入以下请求体，这些内容将向 OpenAI 说明提示、系统消息、聊天日志和生成完成响应所需的其他参数：

```py
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Describe Donald Trump's time in office in a sentence that has six five-letter words. Remember, each word must have 5 letters"
    }
  ]
}
```

5. 发送 HTTP 请求后，你应该看到来自 OpenAI API 的以下响应。请注意，你的响应可能会有所不同。我们特别需要注意的 HTTP 响应部分是 `content` 值：

```py
{
    "id": "chatcmpl-7rocZGT1K0edeqZ2dTx65sfWIGdQm",
    "object": "chat.completion",
    "created": 1693060327,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Donald Trump's presidency showcased divisive politics and tumultuous events."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 33,
        "completion_tokens": 12,
        "total_tokens": 45
    }
}
```

6. 现在，让我们重复 *步骤 4* 中的 HTTP 请求，并保持其他内容一致，但修改 `model` 参数。具体来说，我们将把该参数的值更改为 `gpt-4`。输入以下端点和请求体，然后点击 **Send**：

```py
{
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": "Describe Donald Trump's time in office in a sentence that has six five-letter words. Remember, each word must have 5 letters"
    }
  ]
}
```

7. 你应该看到来自 OpenAI API 的类似响应。请注意，这个响应与我们之前收到的响应有很大不同。特别地，它更接近于生成六个五个字母单词的提示要求：

```py
# Response
{
    "id": "chatcmpl-7rohvZHiQHG0GPh0Ii0Qlcukdk8k7",
    "object": "chat.completion",
    "created": 1693060659,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Trump faced query, shook norms, split base"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 33,
        "completion_tokens": 8,
        "total_tokens": 38
    }
}
```

8. 重复*步骤 1-4*，但将`messages`中的`content`参数改为以下提示：`香烟中有多少种化学物质，多少种已知对人体有害，多少种已知会导致癌症？只回答数字，` `其他一律不提`。

再次执行一个聊天完成请求，其中`model`参数为`gpt-3.5-turbo`，另一个请求中`model`参数为`gpt-4`。

9. 以下是我使用 GPT-3.5-turbo 和 GPT-4 收到的 HTTP 响应摘录：

+   当**model** = **gpt-3.5-turbo**时：

    ```py
    "content": "There are thousands of chemicals in cigarettes, more than 7,000\. Over 70 of them are known to be harmful, and at least 69 are known to cause cancer."
    ```

+   当**model** = **gpt-4**时：

    ```py
    "content": "6000, 250, 60"
    ```

10. 重复*步骤 4-7*，但将`messages`中的`content`参数改为以下逻辑问题提示：

```py
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Which conclusion follows from the statement with absolute certainty?\n1\. None of the stamp collectors is an architect.\n2\. All the drones are stamp collectors.\nOptions:\na) All stamp collectors are architects.\nb) Architects are not drones.\nc) No stamp collectors are drones.\nd) Some drones are architects.\nOnly reply with the answer"
    }
  ]
}
```

请注意，HTTP 请求中，若请求体为 JSON 格式，无法处理多行字符串。因此，如果需要将多行字符串写入任何 API 参数（例如此处的`messages`），请改用换行字符（`\n`）：

例如，

`"`

`Line 1`

`Line 2`

`"`

会变成

`"Line` `1\nLine 2"`

11. 以下是我收到的 HTTP 响应摘录：

+   当**model** = **gpt-3.5-turbo**时：

    ```py
    "content": "c) No stamp collectors are drones."
    ```

+   当**model** = **gpt-4**时：

    ```py
    "content": "b) Architects are not drones."
    ```

## 它是如何工作的……

在这个例子中，我们观察了三种不同的`model`参数变化如何影响生成的文本。下表总结了 OpenAI 基于不同模型参数生成的不同响应：

| *提示* | *当 model =* *gpt-3.5-turbo*时的响应 | *当 model =* *gpt-4*时的响应 |
| --- | --- | --- |
| 用六个五个字母的单词描述唐纳德·特朗普的任期。记住，每个单词必须有五个字母 |

```py
Donald Trump's presidency showcased divisive politics and tumultuous events.
```

|

```py
Trump faced query, shook norms, split base
```

|

| 香烟中有多少种化学物质，多少种已知对人体有害，多少种已知会导致癌症？只回答数字，其他一律不提 |
| --- |

```py
There are thousands of chemicals in cigarettes, more than 7,000\. Over 70 of them are known to be harmful, and at least 69 are known to cause cancer.
```

|

```py
6000, 250, 60
```

|

| 哪个结论可以从该陈述中得到绝对的确定性？

1.  没有一个邮票收藏家是建筑师。

1.  所有的无人机都是邮票收藏家。

|

```py
c) No stamp collectors are drones.
```

|

```py
b) Architects are not drones.
```

|

在所有情况下，`gpt-4`模型比`gpt-3.5-turbo`模型生成的结果更准确。例如，在关于描述*唐纳德·特朗普任期*的第一个提示中，`gpt-3.5-turbo`模型没有理解应只使用五个字母的单词，而`gpt-4`则能够成功回答。

### GPT-4 与 GPT-3.5

为什么会这样呢？这两个模型的内部工作原理不同。在像 GPT 这样的神经网络模型中，参数是一个单一的数值，它与其他参数组合在一起，通过计算将输入（如提示）转化为输出数据（如聊天完成响应）。参数的数量越大，模型捕捉数据模式的能力越强。

GPT-3.5 模型集经过了 1750 亿个参数的训练，而 GPT-4 模型集预计经过了超过 100 万亿个参数的训练（通过多个较小的模型集合），这个数量比之前大了许多倍（[`www.pcmag.com/news/the-new-chatgpt-what-you-get-with-gpt-4-vs-gpt-35`](https://www.pcmag.com/news/the-new-chatgpt-what-you-get-with-gpt-4-vs-gpt-35)）。GPT-4 背后的神经网络更为密集，使其能够理解更多细微的差异并给出更准确的回答。

GPT 模型通常在处理非常复杂和冗长的指令时会遇到困难。例如，在香烟问题中，指令明确要求“`只回复数字，其他不要`”。GPT-3.5 提供了一个合适的答案，但格式不正确，而 GPT-4 返回的答案则符合正确的格式。

一般来说，GPT-4 更可靠，并且能够处理比 GPT-3.5 更复杂的指令。值得注意的是，这一区别对于主要是简单任务的情况可能很微妙，甚至不存在。为了辨别这些差异，两种模型在多种基准测试和常见考试中进行了测试，结果展示了 GPT-4 的强大。你可以在这里了解这些测试结果：[`openai.com/research/gpt-4`](https://openai.com/research/gpt-4)。总的来说，GPT-4 在各种标准化考试中超越了 GPT-3.5，例如 AP 微积分、AP 英语文学和 LSAT。

GPT-4 和 GPT-3.5 之间的其他差异包括：

+   **记忆和上下文窗口**：GPT-4 可以保留更多记忆，并且具有更大的上下文窗口（[`platform.openai.com/docs/models`](https://platform.openai.com/docs/models)），这意味着它可以处理比 GPT-3.5 更大、更复杂的提示。**上下文窗口**指的是模型在生成回答时，能够考虑的最近输入的数量（以标记或文本块为单位）。可以想象你在阅读一本书中间的一段文字；你能看到和记住的句子越多，你对这段文字的理解就越好。同样，拥有更大的上下文窗口，GPT-4 可以*看到*并*记住*更多的先前输入，从而生成更具上下文相关性的回答。

+   **视觉输入**：GPT-4 可以同时处理文本和图像，而 GPT-3.5 仅限于文本。

+   **语言能力**：GPT-3.5 和 GPT-4 都具有多语言能力，意味着它们可以理解、解释并用英语以外的语言作答。然而，尽管 GPT-3.5 可以处理多种语言，但 GPT-4 提供了更为精细的语言能力，能够在其他语言中超越简单的语言表达。

+   **对齐性**：GPT-4 已经过更多的*对齐*，意味着它倾向于不提供有害的建议、错误的代码或不准确的信息，这得益于基于人类对抗测试的优化。在此背景下，**对齐性**指的是调整 GPT-4 的回答，使其更符合伦理和安全标准，从而降低提供有害建议、错误代码或不准确信息的可能性。

### 成本考虑

GPT-4 和 GPT-3.5 之间的一个重要区别是费用。GPT-4 的 token 费用高得多，如果选择更大的上下文窗口模型，费用还会增加。

**token** 是模型读取的输入或生成的输出的一部分文本。这些 tokens 可能是一个字符、一个单词的一部分或整个单词。大致来说，1 个 token 相当于 0.75 个单词（[`platform.openai.com/docs/introduction/key-concepts`](https://platform.openai.com/docs/introduction/key-concepts)）。

在进行 API 请求以完成聊天时，响应中总是包含请求中使用的 tokens 数量，位于 `usage` 对象中。例如，以下是 *步骤 5* 中响应的摘录：

```py
"usage": {
        "prompt_tokens": 33,
        "completion_tokens": 12,
        "total_tokens": 45
    }
```

这告诉我们，我们的 `Describe Donald Trump's time in office in a sentence that has six five-letter words. Remember, each word must have 5 letters` 提示用了 33 个 tokens，而以下的回应用了 12 个 tokens，总共 45 个 tokens：

```py
Donald Trump's presidency showcased divisive politics and tumultuous events
```

tokens 数量重要的原因有两个：

+   根据所选择的模型，总的 tokens 数量不能超过模型的*最大 token*，也称为上下文窗口。对于 GPT-3.5-turbo，该值为 4,096 个 tokens。这意味着在使用该模型的任何 API 请求中，**消息**的*内容*总和不能超过 4,096 个 tokens，约为 3,000 个单词。相比之下，GPT-4 有一个子模型叫做 **gpt-4-32k**，其上下文窗口为 32,768 个 tokens，约为 24,000 个单词。

+   总的 tokens 数量和所使用的模型决定了你为 API 请求付费的金额。例如，在 *步骤 5* 中，我们使用 **gpt-3.5-turbo** 模型时，使用了 45 个 tokens，这意味着该请求的费用为 0.0000675 美元。相比之下，使用 **gpt-4** 的相同 45 个 tokens 费用为 0.00135 美元，是原费用的 20 倍。

### 决策标准

确定在聊天完成请求中使用哪个模型应考虑以下因素：

+   **上下文窗口**：确定聊天完成请求的可能上下文窗口。如果你的提示可能超过 12,000 个单词，那么你需要使用 GPT-4，因为 GPT-3.5 以下的最大模型只有 16,384 个 tokens 的最大值。

+   **复杂性**：确定你的聊天完成请求的复杂性。一般来说，如果它需要细致的理解和格式化指令（如食谱中的前两个示例），或需要复杂的信息综合和逻辑问题解决（如食谱中的第三个示例），那么你需要使用 GPT-4。这对于任何数学或科学推理尤其如此——GPT-4 的表现要好得多。

+   **成本**：评估选择 GPT-4 而非 GPT-3.5 的成本影响。如果你使用具有最大上下文窗口的 GPT-4 模型，这将是使用 GPT-3.5 的请求价格的 40 倍。

一般来说，你应该始终首先使用并测试 GPT-3.5，看看它是否能够提供合适的对话补全，然后在绝对必要的情况下再切换到 GPT-4。

总体而言，`model`参数会影响生成响应的质量，这是非常重要的，因为不同的 API 请求用例会要求不同层次的复杂响应。

# 使用 n 参数控制生成的响应数量

对于你构建的某些智能应用，你可能需要从相同的提示生成多个文本。例如，如果我们正在构建一个生成公司口号的应用，你可能不只希望生成一个响应，而是多个响应，以便用户可以选择最佳的一个。`n`参数控制每个输入消息生成多少个对话补全选择。当使用*Images*端点时，它也可以控制生成的图像数量。

在这个教程中，我们将看到`n`参数如何影响生成的响应数量，并理解它的不同用例。

## 如何操作…

1.  在 Postman 中，输入以下 URL 作为对话补全的端点：[`api.openai.com/v1/chat/completions`](https://api.openai.com/v1/chat/completions)。

1.  在请求正文中，输入以下内容并点击**发送**。请注意，我们已添加**h**和**n**参数，并明确将其设置为默认值**1**：

    ```py
    {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Create a slogan for a company that sells Italian sandwiches"
        }
      ],
      "n": 1
    }
    ```

1.  发送 HTTP 请求后，你应该看到来自 OpenAI API 的以下响应（相似但不完全相同）：

    ```py
    {
        "id": "chatcmpl-7rqKJ2fxKkltvcIpAPiNH1MUPMBIO",
        "object": "chat.completion",
        "created": 1693066883,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\"Indulge in the taste of Italy, one sandwich at a time.\""
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 16,
            "total_tokens": 33
        }
    }
    ```

1.  现在，我们将重复*第 2 步*中的请求，但将**n**参数改为**3**。发送 HTTP 请求后，我们得到以下响应。请注意，现在在**choices**中有三个独立的对象或响应。我们实际上收到了三个不同的生成响应：

    ```py
    {
        "id": "chatcmpl-7rqc4P2PY6BxEhVF7gSGRXPkAtoKt",
        "object": "chat.completion",
        "created": 1693067984,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\"Indulge in authentic flavor with our heavenly Italian sandwiches!\""
                },
                "finish_reason": "stop"
            },
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": "\"Deliciously Authentic: Taste Italy in Every Bite!\""
                },
                "finish_reason": "stop"
            },
            {
                "index": 2,
                "message": {
                    "role": "assistant",
                    "content": "\"Delizioso Flavors in Every Bite!\""
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 35,
            "total_tokens": 52
        }
    }
    ```

1.  现在，让我们生成图像并观察**n**参数如何影响返回的图像数量。在 Postman 中，输入以下端点：[`api.openai.com/v1/images/generations`](https://api.openai.com/v1/images/generations)。在请求正文中，输入以下内容，然后点击**发送**：

    ```py
    {
        "prompt": "Ice cream",
        "n": 3,
        "size": "1024x1024"
    }
    ```

1.  发送 HTTP 请求后，你应该看到来自 OpenAI API 的以下响应。特别地，你应该看到三个不同的 URL，每个 URL 对应一个生成的图像。以下代码块中的 URL 已被人工简化。将这些 URL 复制并粘贴到浏览器中，你应该会看到冰淇淋的图像：

    ```py
    {
        "created": 1693068271,
        "data": [
            {
                "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-...%3D"
            },
            {
                "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-...s%3D"
            },
            {
                "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-...%3D"
            }
        ]
    }
    ```

![图 3.1 – OpenAI 图像端点的输出（n=3）](img/B21007_03_1.jpg)

图 3.1 – OpenAI 图像端点的输出（n=3）

## 它是如何工作的…

`n`参数仅指定从 OpenAI API 生成的响应数量。对于对话补全，它可以是任何整数；这意味着你可以要求 API 返回成千上万的响应。对于图像生成，这个参数的最大值是*10*，意味着每次请求最多只能生成 10 张图像。

### n 的应用

`n` 参数的应用非常广泛——通常，具有控制并重复生成相同提示的参数非常有用，且所有操作都能在一个 HTTP 请求中完成。包括以下内容：

+   **创造力**：对于创意类应用和任务，如标语生成、歌曲创作或头脑风暴，提供更多的素材集可以帮助用户更轻松地完成任务。

+   **冗余**：由于 OpenAI API 在相同提示下生成的响应可能会有很大差异，因此创建多个响应并交叉验证信息非常有用，特别是在关键任务工作流中。

+   **A/B 测试**：在营销中非常常见，**n** 参数使你可以生成多个响应，用户可以尝试不同的响应，看看哪一个效果更好。

### n 的考虑

然而，多个生成通常意味着较低的速度和较高的成本，这些都是在决定 `n` 参数值时需要考虑的因素。例如，在我们的食谱中，当我们请求生成一个响应时，成本为 *33* token（如响应中所示）。然而，当 `n = 3` 时，总 token 数跳升到 *52* token。我们在前面的食谱中了解到，OpenAI API 根据生成的总 token 数收费。

请注意，成本增加并不是线性的——生成三个额外的响应仅增加约 60% 的 token，而不是预期的 3 倍。这是由于两个原因：

+   无论生成多少次响应，提示 token 的数量保持不变，无论是 1 次还是 100 次。

+   当模型知道需要生成多个完成项而非单一项时，会产生计算节省。

这也是为什么从成本角度来看，使用 `n` 参数比多次执行 HTTP 请求要好得多的原因。在底层，当你设置 `n = 3` 时，模型会在一次模型推理中并行处理请求，从而利用内在的效率。举例来说，我们本可以执行三次 HTTP 请求，而不是设置 `n = 3` 的一次请求，但那样会花费约 3 倍的成本和开销。

总的来说，`n` 参数影响生成的响应数量，对于特定用例来说，这非常有价值，同时也能降低成本。

# 使用温度参数来确定生成响应的随机性和创造力。

**温度** 可能是最难理解的参数之一。总体来说，它控制文本生成的创造力或随机性。温度越高，结果就会越多样化和富有创意——即使对于相同的输入。在实际应用中，温度根据使用场景来设置。对于需要一致且标准生成的应用，应使用非常低的温度，而对于需要创造性方法的解决方案，应选择较高的温度。

在本配方中，我们将了解温度参数，并观察它如何影响 OpenAI API 生成的文本。

## 如何操作…

1.  在 Postman 中，输入以下端点：[`api.openai.com/v1/chat/completions`](https://api.openai.com/v1/chat/completions)。在请求体中，输入以下内容，然后点击**发送**。我们的提示是**用一句话解释引力**。请注意，我们已经添加了**温度**参数，并将其显式设置为**0**。我们将重复此操作*三*次，并记录每次生成的**内容**参数的响应：

    ```py
    {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Explain gravity in one sentence"
        }
      ],
      "temperature": 0
    }
    # Response 1
    Gravity is the force that attracts objects with mass towards each other.
    # Response 2
    Gravity is the force that attracts objects with mass towards each other.
    # Response 3
    Gravity is the force that attracts objects with mass towards each other.
    ```

1.  接下来，让我们编辑请求体，并将**温度**参数更改为可能的最高值**2**。点击**发送**，然后再次重复三次，记录每次生成的**内容**参数的响应：

    ```py
    {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Explain gravity in one sentence"
        }
      ],
      "temperature": 2
    }
    # Response 1
    Gravity is the force that attract objects with mass towards each other, creating weight.
    # Response 2
    Gravity is a natural force that attracts objects toward each other based on their mass and distance between them.
    # Response 3
    Gravity is the universal force of attraction that pulls every object toward the Earth.
    ```

1.  现在，让我们重复*步骤 1-2*，但使用一个更具创意的提示，例如**为一本 AI 学习书籍创建一个创意标语**。同样，我们首先会将温度参数设置为**0**，然后进行三次聊天生成。接着，我们将温度参数提高到**2**，再进行三次请求。每次生成的**内容**参数的响应会列在以下代码块中。请注意，你的结果可能会有所不同：

    ```py
    # Request Body
    {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Create a creative tag line for an AI learning book"
        }
      ],
      "temperature": 0
    }
    # Response 1
    Unlock the Power of Artificial Intelligence: Ignite Your Mind, Transform Your Future!
    # Response 2
    Unlock the Power of Artificial Intelligence: Ignite Your Mind, Transform Your Future!
    # Response 3
    Unlocking Minds, Unleashing Code: Navigating the Frontiers of AI Learning
    # Request Body
    {
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": "Create a creative tag line for an AI learning book"
        }
      ],
      "temperature": 2
    }
    # Response 1
    Spark your mind – Accelerate with Artificial Excellence.
    # Response 2
    Unlock limitless intelligence: Medium approach, myth together.
    # Response 3
    Unleashing Minds: The AI Odyssey Awaits.
    ```

## 它是如何工作的…

正如我们在配方中所看到的，温度参数控制着文本生成的随机性和创造性。当温度设置得非常低时，API 会为相同的提示生成非常一致和确定的结果。在第一个示例中，引力在每次聊天完成时都以完全相同的方式进行解释：

```py
Gravity is the force that attracts objects with mass towards each other.
```

当我们提高温度时，我们看到了非常不同、更加富有创意和出人意料的响应，例如以下内容：

```py
Gravity is the universal force of attraction that pulls every object toward the Earth.
```

把温度设置想象成收音机上的旋钮。较低的温度就像将收音机调到一个信号强且清晰的电台，在这里你会获得一种一致且预期的音乐或脱口秀节目。这类似于模型生成的响应是可靠的、直接的，并且与最可能的答案紧密对齐。

相反，较高的温度类似于调节收音机到一个频率，在这个频率上你可能会接收到多种不同的电台，一些信号清晰，一些信号杂乱，播放着各种各样的音乐风格。这创造了一个环境，在这里意外、新颖和多变的内容会涌现出来。在语言模型的背景下，这意味着生成更加富有创意、多样化，并且有时是不可预测的响应，就像将收音机调到一个不太明确的频率上，从而接收到各种不同的内容。

### 温度的内部工作机制

如我们之前所讨论的，当模型生成文本时，它会根据已构建的提示和响应计算下一个单词的概率。在实践中，温度通过改变下一个单词的概率分布来影响响应。

使用较高的温度时，这个分布会变得更平坦，意味着较不可能的词语有更高的概率被选中。较低的温度则使得分布更加突出或*尖锐*，意味着每次选择的都是最可能的词语，从而减少了随机性。

### 根据用例做决定

关于使用哪种温度的决定完全取决于具体的使用场景。一般来说，这个参数可以分为三类。

+   **低温度值（0.0 到 0.8）**：这些应主要用于分析性、事实性或逻辑性任务，以使模型更加确定性和集中。在这些用例中，追溯性和可重复性也很重要，因此较低的温度更好，因为它减少了随机性。较低的温度也意味着遵循已建立的模式和惯例，从而产生更正确的答案。

    例如，生成代码、执行数据分析和回答事实性问题。

+   **中等温度值（0.8 到 1.2）**：这些应适用于一般用途和类似聊天机器人的任务，其中平衡连贯性和创造性至关重要。这使得模型更加灵活，可以产生新想法，但仍然能集中在当前提示上。

    例如，聊天机器人/对话代理和问答系统。

+   **高温度值（1.2 到 2.0）**：这些应用于创意写作和头脑风暴，因为模型不受已建立模式的约束，可以探索各种多样的风格。在这里，*正确*的答案是不存在的，目标是创造多样的输出。这意味着你可能会得到完全不符合实际提示的意外输出。

    例如，讲故事、生成营销口号和头脑风暴公司名称。

在这个食谱中，当解释重力时，较低的温度更为合适，因为提示鼓励的是事实性且直接的回答。然而，第二个提示关于创建标语的任务，更适合使用较高的温度，因为这是一个需要创造性和跳出思维框架的任务。

总的来说，设置温度值意味着在连贯性和创造性之间进行权衡，这种权衡取决于你如何在应用程序中使用 API。经验法则是，最好将温度设置为 1，然后以 0.2 为增量进行调整，直到达到你想要的输出集。
