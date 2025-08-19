

# 第七章：使用 OpenAI API 构建助手

ChatGPT 改变生成性 AI 格局的主要原因是它将自己推广为一个易于使用的全能数字助手。这种方式使它对广泛的用户群体（从开发者和企业到教育工作者和创意专业人士）都非常可及。

OpenAI API 的多功能性在于它能够理解并生成类人文本，从而能够创建适应各种需求的复杂数字助手。无论是自动化客户服务、提供教育支持、协助内容创作，还是进行互动故事讲述，API 强大的功能都允许无尽的可能性。

因此，我们可以利用 API 以及我们在前几章学习的其他元素，创建强大的知识助手。那么我们创建的助手与 ChatGPT 产品本身有何不同呢？答案在于助手可以访问的知识或信息。

OpenAI 已经在互联网上的各种不同来源上训练了 ChatGPT，但该模型本身是数据有限的。这有两个含义：

+   *信息过时*: 该模型（没有任何扩展或插件）无法提供当前或最新的信息。例如，你不能问它**昨天的** **篮球比赛** **比分是多少**。

+   *知识检索混乱*: 你不能限制模型只查看特定的来源来回答问题。由于这一点，并且因为数据是从互联网上的各种来源进行训练的（而且显然并非所有网上的信息都是正确的），这意味着你得到的答案可能并不总是正确的。这也可能是由于幻觉现象所导致的。

我们如何解决这个问题？我们可以构建自己的助手，使用 OpenAI API 和我们指定的可信知识源。这可以是用户可以上传的 PDF 文件，或者是我们知道具有最新信息的网页链接。

在本章中，我们将构建基于知识的助手。我们将利用 ChatGPT 理解类人文本并做出相应回应的能力，以及一个可信的最新知识源。与上一章类似，应用架构将包含访问 OpenAI API 的前端和后端。然而，我们将添加一个中介步骤来考虑知识源。如果你不记得架构了，*图 7.1*展示了任何应用程序中的层次结构。

![图 7.1 – 使用 OpenAI API 和知识源的典型应用架构示例](img/B21007_07_1.jpg)

图 7.1 – 使用 OpenAI API 和知识源的典型应用架构示例

在本章中，我们将涵盖以下配方：

+   创建一个知识检索助手应用程序

+   通过 Assistants API 创建知识检索助手

# 技术要求

本章中的所有食谱都要求你可以访问 OpenAI API（通过生成的 API 密钥）并安装了 API 客户端。你可以参考*第一章*中的食谱*使用 Postman 发起 OpenAI API 请求*，了解如何获取 API 密钥。这还需要了解 Python 和 Python OpenAI 库，我们在*第四章*的第一个食谱中已涉及这些内容。

我们还将使用**Google Cloud Platform**（**GCP**）来托管我们的公共端点。GCP 是由 Google 提供的一套云计算服务，提供包括数据库、数据存储、数据分析、机器学习等一系列托管和计算服务，全部托管在 Google 的基础设施上。你可以参考*第五章*中的食谱*创建调用 OpenAI API 的公共端点服务器*，了解更多信息。

最后，你需要熟悉*Bubble*，这是一种可视化编程平台，允许用户无需编写代码即可创建 Web 应用程序。你可以参考*第五章*中的食谱*从无代码应用程序调用用户创建的端点*，了解如何设置 Bubble。

同时建议你完成*第六章*中的食谱，因为这一章的食谱将涵盖我们在该章学到的概念。

# 创建知识检索助手应用程序

在这个第一个食谱中，我们将构建一个智能应用程序，分析上传的 PDF 文档，并回答用户提出的问题。这可以有多种使用场景，例如通过快速总结要点帮助学术研究，协助法律专业人员从长篇文档中提取特定信息，或帮助企业理解技术报告。

该应用程序将利用 OpenAI API 的自然语言处理（NLP）能力来解析 PDF 的内容并提供准确、上下文感知的响应。这不仅简化了信息检索的过程，还通过提供互动式、AI 驱动的洞察力增强了用户参与度。

我们将跟随的示例是上传以下关于无人机的操作手册，并希望提问如 `我可以将这架无人机飞到的最大高度是多少？` 和 `如何给无人机电池充电？`。无人机操作手册的 PDF 文件可以在此处找到：[`bookpackt67.blob.core.windows.net/test/XDroneManual.pdf?sp=r&st=2024-01-12T00:52:16Z&se=2024-12-31T08:52:16Z&spr=https&sv=2022-11-02&sr=b&sig=IEXLlGXVXCilEg0ffqW8ItXc4LX2YkbRWuZIpSxfP8Y%3D`](https://bookpackt67.blob.core.windows.net/test/XDroneManual.pdf?sp=r&st=2024-01-12T00:52:16Z&se=2024-12-31T08:52:16Z&spr=https&sv=2022-11-02&sr=b&sig=IEXLlGXVXCilEg0ffqW8ItXc4LX2YkbRWuZIpSxfP8Y%3D)。在开始本指南之前，我们应该先将文件下载到电脑。文件的截图见 *图 7.2*。

![图 7.2 – 无人机手册的 PDF 文件](img/B21007_07_2.jpg)

图 7.2 – 无人机手册的 PDF 文件

## 准备工作

确保您拥有一个可用的 OpenAI 平台账户并且有足够的使用积分。如果没有，请按照 *第一章* 中的 *设置 OpenAI Playground 环境* 食谱操作。

此外，请确保您已创建了 GCP 账户。您可能还需要提供账单资料以创建任何 GCP 资源。请注意，GCP 确实有免费套餐，并且在本指南中，我们将不会超出免费套餐（也就是说，您不应为任何服务付费）。

最后，确保您已创建一个 Bubble 账户，您可以在 [`bubble.io`](http://bubble.io) 上免费创建。

本章中的两个食谱都将有相同的要求。

## 如何操作…

### Google Cloud Functions

1.  在新的浏览器标签页中，访问 [`cloud.google.com`](https://cloud.google.com)，并使用您的 Google 凭证登录。

1.  选择右上角的 **控制台**。

1.  创建一个新的 Google 云函数。在搜索栏中输入 **function**，选择 **Cloud Functions**，然后选择 **创建函数**。

1.  给函数起一个具有描述性的名称。由于该函数将根据文件返回答案，我们将其命名为 **get_answer_from_file**。

1.  在 **身份验证** 菜单中，确保选择 **允许未经身份验证的调用** 作为身份验证方式。这样可以使前端应用能够调用后端层。

1.  选择 **运行时、构建、连接和安全设置** 下拉菜单以展开选项。将 **超时** 从 60 秒改为 300 秒。这将确保 Google 云函数的超时不是 1 分钟，而是 5 分钟。这在多模态应用中非常重要，因为会发出多个 API 请求。

![图 7.3 – Google 云函数配置设置](img/B21007_07_3.jpg)

图 7.3 – Google 云函数配置设置

1.  选择 **下一步** 以进入函数开发阶段。在 **运行时** 下拉菜单中选择 **Python 3.12**。对于 **入口点**，选择或输入 **get_answer_from_pdf**。

1.  在左侧菜单中转到**Requirements.txt**，并输入以下 Python 库，因为这些库将在后台函数中使用：

    ```py
    openai
    PyPDF2
    requests
    ```

1.  对于实际的*代码块*，输入以下内容。此函数接受两个输入（**pdf_url** 和 **question**），并根据在 PDF 中找到的信息返回相应的**答案**：

    ```py
    import functions_framework
    from openai import OpenAI
    from PyPDF2 import PdfReader
    import io
    import requests
    @functions_framework.http
    def get_answer_from_pdf(request):
    request_json = request.get_json(silent=True)
    pdf_url = request_json['pdf_url']
      question = request_json['question']
      client = OpenAI(api_key = '<API-key here>')
      # Send a GET request to the URL
      response = requests.get(pdf_url, stream=True)
    response.raise_for_status()  # Ensure the request was successful
      # save text
      text = ''
      # Create a PDF reader object using the byte stream from the response
      with io.BytesIO(response.content) as pdf_file:
    pdf_reader = PdfReader(pdf_file)
        # Iterate over each page and print the text
        for page_num in range(len(pdf_reader.pages)):
          page = pdf_reader.pages[0]
          text = text + page.extract_text()
      response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
          "content": "You are a helpful assistant that goes through text file and answers questions"
        },
        {
          "role": "user",
          "content": '\n\n TEXT: ' + text + ' \n\n QUESTION: ' + question
        }
      ],
      temperature=0.64,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
      )
      answer = response.choices[0].message.content
      result = {
        'answer': answer
      }
      return result
    ```

1.  选择**部署**。你可能需要等待五分钟，直到部署完全完成。当你在云函数页面看到绿色勾选标记时，说明你的函数已成功部署。

1.  与上一章类似，我们现在将使用 Postman 测试刚刚部署的云函数。打开 Postman。点击左上角的**新建**，然后选择**HTTP**。

1.  在 Postman 请求中，选择**Headers**，并输入一个新的头部，**Key**为**Content-Type**，**Value**为**application/json**。

1.  在左侧下拉菜单中，将请求从**GET**改为**POST**。复制**云函数**页面中的端点 URL，并将其粘贴到 Postman 中。

1.  选择**Body**，然后选择**Raw**，并复制粘贴以下 JSON 请求：

    ```py
    {
        "pdf_url": "https://bookpackt67.blob.core.windows.net/test/XDroneManual.pdf?sp=r&st=2024-01-12T00:52:16Z&se=2024-12-31T08:52:16Z&spr=https&sv=2022-11-02&sr=b&sig=IEXLlGXVXCilEg0ffqW8ItXc4LX2YkbRWuZIpSxfP8Y%3D",
        "question": "for safety, what's the highest you should fly the drone?"
    }
    ```

1.  选择**发送**，以调用你的云函数。如果一切顺利，你应该看到类似于*图 7.4*所示的响应，其中包含嵌入在 JSON 响应中的多个对象。

![图 7.4 – Postman 输出](img/B21007_07_4.jpg)

图 7.4 – Postman 输出

### Bubble

1.  接下来，访问 [`bubble.io`](http://bubble.io) 并登录。选择**创建一个应用**并为你的应用命名。选择**开始使用**，然后选择**从基本功能开始**。如果出现**跳过应用助手**的提示，你也可以选择跳过。

1.  在画布页面，我们将添加一些应用程序所需的元素。从左侧菜单中选择**输入**，然后在页面顶部绘制一个矩形。双击该元素，在属性菜单中，将**占位符**替换为**问题**。

![](img/B21007_07_5.jpg)

图 7.5 – Bubble 输入配置

1.  从左侧菜单中选择**文件上传器**元素，并将其绘制在前一个元素的下方。

1.  接下来，从左侧菜单中选择文本元素并绘制一个矩形（可以设为多行），将其放置在前一个元素下方。双击文本元素以进入属性菜单。然后，点击实际文本框中的**插入动态数据**，选择**文本 A**并**创建一个新的自定义状态**。系统会提示你输入名称和类型。在名称处输入**answer**，在类型处确保选择**文本**。这将为文本框创建一个唯一的自定义状态，用于在应用程序中显示值。

![图 7.6 – 插入动态数据](img/B21007_07_6.jpg)

图 7.6 – 插入动态数据

![图 7.7 – 创建自定义状态](img/B21007_07_7.jpg)

图 7.7 – 创建自定义状态

1.  接下来，我们需要将我们创建的云函数加载到 Bubble 中。选择左侧菜单中的**插件**，然后选择**添加插件**。选择**API 连接器**，然后选择**安装**，接着点击**完成**。

![图 7.8 – Bubble.io UI 配置](img/B21007_07_8.jpg)

图 7.8 – Bubble.io UI 配置

1.  从插件列表中选择**API 连接器**。选择**添加另一个 API**。对于**API 名称**，输入**answer_from_file**。向下滚动到**创建新 API**并点击**展开**。将此 API 的名称保留为**API 调用**。对于 API，配置以下设置：

    +   在**用作**下拉菜单中，选择**动作**。

    +   将请求方式从**GET**更改为**POST**。

    +   点击**新建标题**以创建新标题。选择**添加标题**。对于**key**，输入**Content-Type**，对于**value**，输入**application/json**。

    +   通过点击**参数**添加一个参数。对于**key**，输入**pdf_url**。对于**value**，输入**https://bookpackt67.blob.core.windows.net/test/XDroneManual.pdf?sp=r&st=2024-01-12T00:52:16Z&se=2024-12-31T08:52:16Z&spr=https&sv=2022-11-02&sr=b&sig=IEXLlGXVXCilEg0ffqW8ItXc4LX2YkbRWuZIpSxfP8Y%3D**。不要包含引号。确保**私密**框未被选中。

    +   再次点击**参数**。对于**key**，输入**question**。对于**value**，输入**为了安全，飞行无人机的最高高度应该是多少？**。不要包含引号。确保**私密**框未被选中。

1.  选择**初始化调用**来测试 API 调用。如果您看到显示在*图 7.9*中的屏幕，那么调用已成功。确保为每个选项选择了**文本**类型，并点击**保存**。

![图 7.9 – 成功的 UI 配置](img/B21007_07_9.jpg)

图 7.9 – 成功的 UI 配置

1.  从左侧菜单中选择**设计**。通过选择**按钮**并在**文件上传**元素下方绘制一个框来创建按钮元素。

1.  双击您创建的按钮元素。在弹出的属性菜单中选择**添加工作流**。

![图 7.10 – 页面视觉布局和添加工作流](img/B21007_07_10.jpg)

图 7.10 – 页面视觉布局和添加工作流

1.  选择**点击此处添加动作**。进入**插件**，找到您刚刚创建的 API（**answer_from_file - API 调用**），并选择它。然后在弹出的属性菜单中执行以下操作：

    +   删除**(param) pdf_url**的内容。输入**http:**，然后选择**插入动态数据**。向下滚动并选择**文件上传器 A**，然后选择**value**。

    +   删除**(param) question**的内容。选择**插入动态数据**。向下滚动并选择**输入问题**，然后选择**value**。

1.  接下来，再次选择**点击此处添加操作**，向下滚动至**元素操作**，然后选择**设置状态**。在**元素**下拉菜单中，选择**Text A**。在**自定义状态**下拉菜单中，选择**answer**。在**值**字段中，选择**步骤 1 的结果**，然后选择**answer**。这将使**Text A**的值等于你创建的云函数 API 调用返回的答案。

![图 7.11 – 在元素中设置状态](img/B21007_07_11.jpg)

图 7.11 – 在元素中设置状态

![图 7.12 – 设置值为自定义状态](img/B21007_07_12.jpg)

图 7.12 – 设置值为自定义状态

1.  我们已经完成了 Bubble 应用所需的一切。让我们来测试一下应用是否正常工作。在右侧选择**预览**，然后会出现一个新页面，显示你的应用。在**问题**文本框中，输入**为了安全，应该将无人机飞到什么高度？**。选择**文件上传**输入框，上传你之前下载的无人机手册 PDF 文件。

1.  点击**提交**按钮。如果一切顺利，你应该会看到一个像*图 7.13*中的屏幕，其中包含了直接从 PDF 文件内容中提取的用户问题的答案。

![图 7.13 – Bubble 应用的输出](img/B21007_07_13.jpg)

图 7.13 – Bubble 应用的输出

完成这些步骤后，尝试输入其他与 PDF 文件相关的问题，比如`如何给电池充电？`或者`如何做一个` `360 翻转？`。

## 它是如何工作的…

在这个配方中，我们创建了一个基于知识的助手，它可以读取任何 PDF 文件并根据该文件提供问题的答案。

这与上一章中的配方非常相似，但有一个关键区别：用来回答问题的知识不来自 ChatGPT，而是来自提供的 PDF 文件。

### 代码讲解

我们在 Google 云函数中使用的代码包含了读取 PDF URL、提取内容并将其传递给 OpenAI API 的指令，这也是它与之前配方的不同之处。

为此，我们需要导入两个额外的 Python 包：

+   **Requests**：这个包使我们能够轻松发送 HTTP 请求。这对于从提供的 URL 获取 PDF 文件至关重要。通过使用**requests.get()**函数，我们可以检索任何在线文档的内容，在本例中是一个 PDF 文件。

+   **PyPDF2**：这个包使我们能够在 Python 中直接与 PDF 文件交互并进行操作。一旦 PDF 被获取，**PyPDF2**提供了读取 PDF 文件、提取文本，甚至在必要时处理 PDF 操作的工具。在我们的应用中，它主要用于将 PDF 内容转换为可以由 OpenAI API 处理的文本格式。

我们首先向用户提供的 PDF URL（来自`pdf_url`对象）发起请求。然后我们将其转换为字节：

```py
  # Send a GET request to the URL
  response = requests.get(pdf_url, stream=True)
response.raise_for_status()  # Ensure the request was successful
```

我们接着使用 `PyPDF2` 的 `PdfReader` 类来读取字节，并逐页读取 PDF 文件的内容。我们将 PDF 文件的整个字符串内容保存到 `text` 变量中：

```py
  # save text
  text = ''
  # Create a PDF reader object using the byte stream from the response
  with io.BytesIO(response.content) as pdf_file:
pdf_reader = PdfReader(pdf_file)
    # Iterate over each page and print the text
    for page_num in range(len(pdf_reader.pages)):
      page = pdf_reader.pages[0]
      text = text + page.extract_text()
```

最后，当我们调用 OpenAI Chat API 时，我们的指令是让 LLM 阅读文本和用户的问题，然后基于提供的文本回答用户的问题。我们然后提供了整个 PDF 文件的文本和用户的问题：

```py
  response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": "You are a helpful assistant that goes through text file and answers questions"
    },
    {
      "role": "user",
      "content": '\n\n TEXT: ' + text + ' \n\n QUESTION: ' + question
    }
  ],
  temperature=0.64,
max_tokens=1000,
top_p=1,
frequency_penalty=0,
presence_penalty=0
  )
  answer = response.choices[0].message.content
```

这就像你去 ChatGPT，复制并粘贴整个 PDF 文件的内容，然后问用户的问题。然而，通过这种方式，你已经围绕它构建了一个应用程序，使得用户操作更加简便——他们只需上传 PDF 并在文本框中输入问题。

### 这种方法的局限性

我们已经讨论过这种方法的好处：即因为 ChatGPT 的内部知识可能过时，且无法追踪 ChatGPT 提供的任何信息的来源，建立一个强制 Open AI 仅使用用户提供的知识的架构可以缓解这个问题。

然而，这种方法也有一些局限性。了解这些局限性是关键，这样您才能充分利用这种方法。具体如下：

+   *依赖于 PDF 的质量*：助手的准确性在很大程度上依赖于 PDF 内容的质量和清晰度。结构不佳的文档或复杂的格式可能导致不完整或错误的回答。此外，这种方法无法处理嵌入在图像中的信息，因为 Open AI API 无法读取图像。

+   *大型文档的资源消耗*：助手需要处理并理解整个文档以回答每个查询，这可能会消耗大量资源，尤其是在处理大型文档时，可能导致响应时间缓慢。

+   *知识范围有限*：助手的知识仅限于 PDF 的特定主题，缺乏 ChatGPT 内置数据库中所涵盖的全面内容。

+   *知识库的维护*：保持信息的时效性需要不断更新 PDF 文件，这可能非常耗时，特别是对于信息频繁变化的情况。

+   *上下文窗口的限制*：PDF 中的内容必须在 ChatGPT 模型的上下文窗口内。对于非常长的 PDF 文件，这种方法不可行，因为模型无法处理超出其上下文窗口的内容（即 API 可处理的最大单词数），从而限制了可用于回答的问题的信息量。

因此，用户可以采用另一种方法，即 OpenAI API 的特殊子集——Assistants API，我们将在后续的教程中讨论。

# 通过 Assistants API 创建一个知识检索助手

OpenAI 最近发布了助手 API，通过该 API，你可以创建基于知识的助手，几乎不需要编码和复杂的设置。一个很大的优势是，你可以将 OpenAI 已经构建的工具集成到你的助手中，例如*代码解释器*和*知识检索*。这些增强功能实质上赋予了你的助手应用超能力。在这个食谱中，我们将重点介绍知识检索工具。

**知识** **检索**通过整合外部知识（例如前面食谱中的无人机手册 PDF 文件）来增强你的助手。OpenAI 会自动高效地对上传的文档进行分段，同时创建嵌入索引。这些嵌入存储在 OpenAI 的数据库中。

回想一下在*第四章*中，我们讨论了如何使用嵌入来比较文本相似性并搜索文本片段。通过知识检索，OpenAI 会自动为你执行这些操作。当用户向助手 API 提问时，如果问题被知识检索增强，API 会使用向量搜索从上传的文档中提取相关信息，从而有效地回答用户的查询。

这样，知识来源的长度可以是无限的，因为知识源本身不会传递到 Chat API，而是经过向量化，仅根据用户的提问将相关的文本片段传递给 Chat API。

换句话说，使用带有知识检索的助手 API 意味着你可以上传大量 PDF 文件，数量没有限制，而且实际上没有上下文窗口的限制。

这对于创建高度专业化的助手非常有用，这些助手需要调用大量特定信息。例如，如果你在为法律专业人员构建助手，你可以上传大量的法律文本、案例研究和法规。由知识检索驱动的助手 API 随后可以在复杂的查询中提供准确的法律引用和解释。同样，针对医学专业人员，上传大量的医学文献和研究论文可以使助手 API 提供详细的医学见解。这使得助手 API 不仅仅是一个对话工具，而是一个强大的、信息丰富的资源。

在这个食谱中，我们将使用助手 API 创建一个*法律宪法助手*，该助手直接从美国宪法文档中获取知识。

## 如何实现…

### Playground

1.  访问[`openai.com`](https://openai.com)，在左侧菜单中选择**Playground**。在顶部菜单中，确保从下拉菜单中选择了**Assistants**。在顶部附近选择下拉菜单，然后选择**创建助手**以创建一个新的助手。

1.  输入以下配置信息以设置**助手**：

    +   **名称**：**美国** **宪法专家**。

    +   **说明**：**您是一个帮助回答与美国宪法相关的法律问题的助手。请引用文档中找到的特定部分，以回答问题。**

    +   **模型**：**gpt-4-1106-preview**。

    +   将 **Retrieval** 工具切换到 *开启* 状态。

    +   在 **文件** 部分，选择 **添加**，并上传以下文件 ([`bookpackt67.blob.core.windows.net/test/us_constitution.pdf?sp=r&st=2024-01-15T07:51:23Z&se=2024-12-31T15:51:23Z&spr=https&sv=2022-11-02&sr=b&sig=C9hFIvrI3FHogBumPTRaL1hrwS8C1B0t3hnlzS9t6Ew%3D`](https://bookpackt67.blob.core.windows.net/test/us_constitution.pdf?sp=r&st=2024-01-15T07:51:23Z&se=2024-12-31T15:51:23Z&spr=https&sv=2022-11-02&sr=b&sig=C9hFIvrI3FHogBumPTRaL1hrwS8C1B0t3hnlzS9t6Ew%3D))。该文件是美国宪法。

注意

您需要将文件下载到本地，然后再上传到 OpenAI。

![图 7.14 – 助手 Playground 配置](img/B21007_07_14.jpg)

图 7.14 – 助手 Playground 配置

1.  点击底部的 **保存** 按钮。完成此操作后，助手 ID 将出现在 **名称** 字段正下方，格式为 *asst_XXXXXXXXXXXXXXXXXXXXX*。请记下此 ID，因为这是您的助手唯一的标识符，我们在创建后端函数时需要引用它。

### Google 云函数

1.  在新标签页中，访问 [`cloud.google.com`](https://cloud.google.com)，并使用您的 Google 账户凭据登录。

1.  在右上角选择 **控制台**。

1.  创建一个新的 Google 云函数。在搜索栏中输入 **function**，选择 **Cloud Functions**，然后选择 **创建函数**。

1.  给函数起个具有描述性的名字。我们将其命名为 **get_answer**。

1.  在 **身份验证** 菜单中，确保选择 **允许未经身份验证的调用** 作为身份验证方法。这将使前端应用能够调用后端层。

1.  选择 **运行时、构建、连接和安全设置** 下拉菜单，展开选项。将 **超时** 从 60 秒改为 300 秒。这将确保 Google 云函数的超时从 1 分钟增加到 5 分钟。

![图 7.15 – Google 云函数配置设置](img/B21007_07_15.jpg)

图 7.15 – Google 云函数配置设置

1.  选择 **下一步** 进入函数开发阶段。在 **运行时** 下拉菜单中，选择 **Python 3.12**。对于 **入口点**，选择或输入 **get_answer**。

1.  在左侧菜单中，找到 **Requirements.txt**，并输入以下 Python 包，因为该库将在后端函数中使用：

    ```py
    openai
    ```

1.  对于实际的代码块，输入以下内容。此函数接受一个输入（**question**），并通过 Assistants API 返回相关的**answer**。代码演示将在*使用助手*子章节中讨论。将**<assistant-id-here>**替换为您在*Playground*部分*步骤 3*中记下的助手 ID：

    ```py
    import functions_framework
    import time
    from openai import OpenAI
    @functions_framework.http
    def get_answer(request):
      # declare assistant id
    assistant_id = '<assistant-id-here>'
      # get inputs
    request_json = request.get_json(silent=True)
      question = request_json['question']
      # initializeopenai
      client = OpenAI(api_key = '<api-key here>')
      # create a thread
      thread = client.beta.threads.create()
      # create message question in thread
      message = client.beta.threads.messages.create(
    thread_id=thread.id,
        role="user",
        content=question
        )
      # run assistant
      run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id,
      instructions=""
      )
      # wait 30 seconds for response
    time.sleep(30)
      # get answer
    thread_messages = client.beta.threads.messages.list(thread.id)
      answer = thread_messages.data[0].content[0].text.value
      # return answer
      result = {
        'answer': answer
      }
      return result
    ```

1.  选择**部署**。您可能需要等待五分钟，直到部署完全完成。当您在云函数页面上看到绿色勾选标记时，说明您的函数已成功部署。

1.  与之前的步骤类似，我们现在将使用 Postman 来测试我们刚刚部署的云函数。打开**Postman**。在左上角选择**新建**，然后选择**HTTP**。

1.  在 Postman 请求中，选择**头部**并添加一个新头部，**Key**为**Content-Type**，**Value**为**application/json**。

1.  将请求从左侧下拉菜单中的**Get**更改为**Post**。复制**云函数**页面中的端点 URL 并将其粘贴到 Postman 中。

1.  选择**Body**，然后选择**Raw**，并复制粘贴以下 JSON 请求：

    ```py
    {
        "question": "How many senators are there?"
    }
    ```

1.  选择**发送**以调用您的云函数。如果一切顺利，您应该会看到类似以下的答案：

    ```py
    The Senate of the United States shall be composed of two Senators from each state, and each Senator shall have one vote.
    ```

### Bubble

1.  登录并导航到 w。选择**创建应用**，为您的应用命名。选择**开始使用**，然后选择**从基础功能开始**。如果遇到**跳过应用助手**的提示，您也可以点击该选项。

1.  在画布页面上，我们将添加一些应用程序所需的元素。从左侧菜单选择**输入**，然后在页面顶部绘制一个矩形。双击该元素，在属性菜单中将**占位符**替换为**question**。

![图 7.16 – Bubble 输入配置](img/B21007_07_16.jpg)

图 7.16 – Bubble 输入配置

1.  接下来，通过从左侧菜单选择文本元素并在前一个元素下方绘制一个矩形（可以设置为多行）来创建一个文本元素。双击文本元素以显示属性。然后，在实际的文本字段上点击**插入动态数据**，选择**Text A**，并选择**创建一个新的自定义状态**。系统将提示您输入名称和类型。对于名称，输入**answer**；对于类型，确保选择**文本**。这将为文本框创建一个独特的自定义状态，应用程序需要它来显示值。

![图 7.17 – 向 Bubble 添加文本元素](img/B21007_07_17.jpg)

图 7.17 – 向 Bubble 添加文本元素

1.  接下来，我们需要做的是将我们创建的云函数加载到 Bubble 中。从左侧菜单选择**插件**，然后选择**添加插件**。选择**API 连接器**，然后选择**安装**，最后点击**完成**。

![图 7.18 – Bubble.io UI 配置](img/B21007_07_18.jpg)

图 7.18 – Bubble.io UI 配置

1.  从插件列表中选择**API 连接器**。选择**添加另一个 API**。在**API 名称**中输入**get_answer**。向下滚动到**创建新 API**，然后点击**展开**。将此 API 的名称保持为**API 调用**。对于 API，配置如下设置：

    +   从**用作**下拉菜单中，选择**动作**。

    +   将请求方法从**GET**更改为**POST**。

    +   使用**新建头部**创建一个新的头部。选择**添加头部**。在**key**中输入**Content-Type**，在**value**中输入**application/json**。

    +   点击**参数**以添加一个参数。在**key**中输入**question**，在**value**中输入**How many senators are there?**。不要包含引号。确保**private**框未被选中。

1.  选择**初始化调用**以测试 API 调用。确保每个**选项**都已选择**文本**类型，然后点击**保存**。

1.  从左侧菜单中选择**设计**。通过选择**按钮**并在输入元素右侧绘制一个框来创建一个按钮元素。

1.  双击你创建的**按钮**元素。在弹出的属性菜单中，选择**添加工作流**。

![图 7.19 – 页面视觉布局，添加按钮和工作流](img/B21007_07_19.jpg)

图 7.19 – 页面视觉布局，添加按钮和工作流

1.  选择**点击此处添加动作**。进入**插件**，找到你刚刚创建的 API（**get_answer - API 调用**），并选择它。在弹出的属性菜单中，删除**(param) question**的内容。选择**插入动态数据**。向下滚动，选择**输入问题**，然后选择**值**。

1.  接下来，再次选择**点击此处添加动作**，向下滚动到**元素动作**，然后选择**设置状态**。在**元素**下拉菜单中，选择**Text A**。在**自定义状态**下拉菜单中，选择**answer**。在**值**中，选择**第 1 步的结果**，然后选择**answer**。这将使**Text A**的值等于你创建的云函数 API 调用的回答。

1.  我们已经完成了 Bubble 应用所需的所有内容。现在让我们测试应用程序是否正常工作。选择右侧的**预览**，一个新页面将会显示出你的应用程序。在**问题**文本框中，输入**How many senators are there?**。

1.  选择**提交**按钮。如果一切顺利，你应该会得到一个类似于你之前在 Postman 中看到的回答，回答中包含我们之前上传的*美国宪法文档*中的文本。

![图 7.20 – Bubble 应用示例问题](img/B21007_07_20.jpg)

图 7.20 – Bubble 应用示例问题

最棒的是，你可以提问*任何*相关问题，它会基于你上传的美国宪法文档进行回答，即使文档有几百页。答案也是文档特定的。例如，你可以问`一个人要多大年纪才能成为参议员？`，它会生成一个回答，如*图 7.21*所示。

![图 7.21 – 气泡应用示例问题](img/B21007_07_21.jpg)

图 7.21 – 气泡应用示例问题

你可以在*图 7.22*中看到，PDF 使用了不同的语言，但助手已经重新调整，使其与我们提出的问题格式匹配。

![图 7.22 – 从中检索答案的文档摘录](img/B21007_07_22.jpg)

图 7.22 – 从中检索答案的文档摘录

## 它是如何工作的…

在这个示例中，我们创建了一个能够回答我们加载的 PDF 中的问题的助手。

该手册长达 85 页，包含超过 35,000 个词汇，然而 API 在从手册中找到正确的信息时毫无问题。值得注意的是，即使是百万字的知识源或几个大型 PDF 文件，这也是可以完成的。这就是 Assistants API 中的检索工具的魅力——它可以轻松扩展。

### 创建助手

我们使用**OpenAI Playground**创建了这个助手，而不是通过 OpenAI API。OpenAI Playground 提供了一个很好的 UI，用于创建我们的初始助手，而无需通过编写代码来构建。我们在构建助手时定义了以下参数：

+   **名称**：助手的名称。

+   **指令**：助手使用的系统指令。这与 Chat API 中使用的系统消息非常相似。

+   **模型**：用于构建答案的聊天模型。对于助手来说，推荐使用 GPT-4，因为它能够理解并回答细致的信息，这是任何知识检索应用中的关键元素。

+   **工具**：可以为您的助手添加的超级功能。我们添加了检索工具，这使得 OpenAI 能够读取和搜索用户上传的知识库。

+   **文件**：一组文件，用作知识源并上传。

### 使用助手

在**Google**云函数中，我们首先创建一个包含用户问题的消息线程：

```py
# create a thread
  thread = client.beta.threads.create()
# create message question in thread
message = client.beta.threads.messages.create(
thread_id=thread.id,
    role="user",
    content=question
    )
```

然后，我们运行助手，提供在 OpenAI Playground 创建助手时生成的独特*助手 ID*，以及包含用户问题的消息线程的*线程 ID*。

请注意，在运行助手后，我们强制让我们的函数睡眠 30 秒。我们这样做是因为需要给助手时间充分处理消息线程。另一种方法是使用轮询来检查助手，并在轮询显示成功完成后才继续。如果你想了解更多关于这个方法的信息，可以访问[`pypi.org/project/polling2/`](https://pypi.org/project/polling2/)：

```py
# run assistant
run = client.beta.threads.runs.create(
thread_id=thread.id,
assistant_id=assistant_id,
  instructions=""
  )
# wait 30 seconds for response
time.sleep(30)
```

然后，我们提取助理的回复并将消息返回为`answer`：

```py
# get answer
thread_messages = client.beta.threads.messages.list(thread.id)
  answer = thread_messages.data[0].content[0].text.value
# return answer
result = {
    'answer': answer
  }
```

### 其他应用场景

助理 API 和检索工具的结合为各行各业和领域提供了广泛的潜在应用。以下是一些例子：

+   *客户支持聊天机器人*：开发聊天机器人，通过访问公司的广泛知识库或产品手册，为客户查询提供详细和具体的答案。

+   *医疗信息系统*：构建可以检索并提供特定医学信息、指南或研究论文的系统，为医疗专业人员提供诊断或治疗决策支持。

+   *法律研究助理*：创建能够从大量法律文档、案例和先例中筛选信息的工具，帮助律师为案件做准备或进行法律研究。

+   *教育平台*：开发教育辅助工具，可以从教科书、研究论文或教育材料中提取信息，帮助学生学习或研究各种主题。

+   *财务顾问工具*：构建可以访问和分析财务报告、市场趋势和经济研究的应用程序，为投资建议或市场洞察提供支持。

+   *技术支持与故障排除*：实施能够访问技术手册和用户指南的系统，为故障排除或技术指导提供逐步帮助。

+   *内容策划和推荐系统*：创建可以根据用户偏好或查询分析和检索文章、新闻或多媒体内容的平台。

+   *企业数据检索系统*：开发企业内部工具，可以搜索公司文档、报告和数据库，快速为员工提供他们需要的信息。

+   *旅游和酒店助理*：构建可以访问并提供有关旅游目的地、住宿、当地习俗或景点的旅行助手。

+   *电子商务个人购物助手*：创建可以根据用户查询从产品目录和评论中搜索并推荐产品的工具。

这些应用程序都利用了助理 API 的能力来理解和处理自然语言查询，以及检索工具从大量文档和数据源中访问并提取相关信息的能力。这种结合使得创建功能强大、具有上下文意识和高度信息化的应用成为可能。

在我们结束穿越开放 AI API 广阔世界的旅程时，我希望你能感受到赋予你力量的感觉，以利用其能力推动你创新的项目。从搭建 API 环境的初步步骤到探索终端和关键参数的复杂细节，我们走过了一条道路，这条道路不仅让你理解 OpenAI API，还让你能够应用它来创建能够改变我们与技术互动方式的应用程序。探索额外功能和为应用开发进行分阶段和托管的过程，为你奠定了构建智能解决方案的基础，这些解决方案能够产生影响。

无论是设计多功能智能应用程序还是打造基于知识的助手，你所获得的技能证明了你手中潜力的存在。请记住，旅程并不在此结束。你构建的每一个应用程序都是向创新迈出的一步，是解决复杂问题的桥梁，也是对未来技术与人类创造力和谐融合的贡献。接受那些到来的挑战和机遇，因为你现在已经具备了在技术世界中产生重大影响的能力。
