

# 第七章：构建客户端

要消费 MCP 服务器，你需要某种形式的客户端。例如，你可以使用**Claude 桌面**或**虚拟工作室代码**（**VS Code**），因为它们具有消费 MCP 服务器和处理功能发现的能力，并且能够使用它们。也有情况下你可能需要自己编写的客户端。这种情况的一个好例子是当你想在应用程序中构建 AI 功能时。例如，想象一下，你有一个电子商务应用程序，并希望有一个 AI 改进的搜索。MCP 服务器将是一个单独的应用程序，而客户端将集成到电子商务应用程序中。

考虑到这一点，让我们探讨如何构建客户端以及它包含的内容。

在本章中，你将学习以下内容：

+   使用 STDIO 和 SSE 传输构建客户端

+   消费 MCP 服务器及其功能

+   利用 LLM 来增强客户端体验

本章涵盖了以下主题：

+   构建客户端

+   练习：构建客户端

+   带有 LLM 的客户端

+   与 LLM 一起工作

+   练习：集成 LLM

# 构建客户端

那么，构建客户端需要哪些要素？在宏观层面，我们需要做以下事情：

1.  设置客户端以连接到服务器。

1.  列出功能。

1.  选择一个功能来使用。

1.  提示用户输入参数。

1.  展示结果。

太好了，现在我们已经了解了高级步骤，让我们看看我们是否可以在接下来的练习中构建它，你可以随时跟随代码编写。

# 练习：构建客户端

在这个练习中，你将构建一个连接到服务器并使用其功能的客户端。你将使用 SDK 来构建客户端并调用服务器。客户端将是一个简单的命令行应用程序，允许你选择一个功能并提供其参数。然后客户端将调用服务器并显示结果。

## 设置客户端以连接到服务器

让我们首先创建建立到服务器连接所需的客户端代码：

```py
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="mcp",  # Executable
    args=["run", "server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)
async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:
            # Initialize the connection
            await session.initialize()
            # list features
if __name__ == "__main__":
    import asyncio
    asyncio.run(run()) 
```

在前面的代码中，我们做了以下事情：

+   创建了一个`StdioServerParameters`对象，该对象指定了运行服务器和任何可选的命令行参数的命令。我们这样做的原因是因为服务器将与客户端同时运行，因此我们需要指定如何运行它。

+   定义了一个`run`函数，该函数创建一个`ClientSession`对象并初始化到服务器的连接。在`run`函数内部，我们很快将添加列出和调用功能的代码。

## 列出功能

到目前为止，我们已经设置了客户端以连接到服务器。现在，让我们添加列出服务器上可用功能的代码。这取决于功能类型的不同而有所不同。让我们添加一些代码：

```py
# List available resources
resources = await session.list_resources()
print("LISTING RESOURCES")
for resource in resources:
    print("Resource: ", resource)
# List available tools
tools = await session.list_tools()
print("LISTING TOOLS")
for tool in tools.tools:
    print("Tool: ", tool.name) 
```

在那里，我们有以下代码来列出功能：

+   `列出可用资源`：这将列出服务器上所有可用的资源。我们还打印资源名称到控制台。

+   `列出可用工具`：这列出了服务器中所有可用的工具，并将工具名称打印到控制台。我们也可以打印工具描述和输入模式，但现在我们只打印名称。

## 选择要使用的功能

让我们通过选择一个工具并调用它来展示如何使用我们列出的功能。在这种情况下，我们将使用我们在上一章中创建的`add`工具。`add`工具接受两个参数`a`和`b`，并返回两个数字的和。然而，想象一下现在用户已经看到了一个工具列表并选择了一个工具。现在让我们提示用户输入调用工具所需的参数：

```py
# Read information from the first tool
tool_name = tools.tools[0].name
print(f"Using tool: {tool_name}")
first_value = input("Enter first value: ")
second_value = input("Enter second value: ")
# Call a tool
print("CALL TOOL")
result = await session.call_tool(tool_name, arguments={
    "a": first_value, "b": second_value})
print(result.content) 
```

在前面的代码中，我们做了以下操作：

+   读取列表中第一个工具的名称并将其打印到控制台。

+   提示用户输入第一个和第二个值作为工具的参数。

+   使用`call_tool`方法调用工具，并将参数作为字典传递。结果被打印到控制台。

太好了，现在我们有一个可以连接到服务器、列出功能和调用工具的客户端。然而，这仍然相当程序化，并不非常用户友好。

## 完整的代码

在我们继续集成 LLM 之前，让我们展示客户端的完整代码：

```py
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="mcp",  # Executable
    args=["run", "server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)
async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write
        ) as session:
            # Initialize the connection
            await session.initialize()
            # List available resources
            resources = await session.list_resources()
            print("LISTING RESOURCES")
            for resource in resources:
                print("Resource: ", resource)
            # List available tools
            tools = await session.list_tools()
            print("LISTING TOOLS")
            for tool in tools.tools:
                print("Tool: ", tool.name)
            # Read information from the first tool
            tool_name = tools.tools[0].name
            print(f"Using tool: {tool_name}")
            first_value = input("Enter first value: ")
            second_value = input("Enter second value: ")

             # Call a tool
            print("CALL TOOL")
            result = await session.call_tool(tool_name, arguments={"a":                 first_value, "b": second_value})
            print(result.content)
            # Read a resource
            print("READING RESOURCE")
            content, mime_type = await session.read_resource("greeting://                hello")
if __name__ == "__main__":
    import asyncio
    asyncio.run(run()) 
```

好的，如果你跟着代码输入，你现在应该有一个可以连接到服务器并使用其功能的运行客户端。你将在本章末尾的作业中再次有机会练习这个。

让我们通过集成 LLM 来改进客户端。你将看到这如何提供更好的用户体验，以及如何用它来抽象服务器使用的复杂性。

# 带有 LLM 的客户端

到目前为止，你已经看到了如何使用 STDIO 和 SSE 构建和测试客户端。然而，你可能已经注意到这种方法相当程序化，并不非常用户友好。也就是说，对于你想要使用的每个功能，你需要知道其确切名称和参数。这就是 LLM 发挥作用的地方。通过在客户端中涉及 LLM，你可以抽象掉“知道”的部分，而专注于“做”的部分。下面是如何工作的。

## 在使用 LLM 之前

这就是如何在没有 LLM 的情况下构建一个应用以及应用中的流程：

1.  列出服务器功能。

1.  用户选择一个功能，客户端请求参数。

1.  对响应进行处理。

这种方法相当僵化，需要用户明确知道并选择列出的功能之一。那么，更好的方法是什么呢？

## 在涉及 LLM 之后

为了解决感觉僵化的客户端，想象一下用户不知道这些功能；他们只通过提示进行交流。带着这个想法，现在让我们看看应用的流程：

1.  列出服务器功能。

1.  将功能列表转换为 LLM 工具。

1.  用户输入一个自然语言请求。

1.  客户端将请求发送到 LLM（大型语言模型），LLM 会确定要使用哪个功能以及要发送哪些参数，如果没有匹配的功能，则返回一个通用的 LLM 响应。

用户体验的差异相当显著。这种第二种方法意味着用户不再需要了解功能，也不需要选择要使用的功能。相反，用户只需输入一个自然语言请求，LLM 就会处理其余部分。

让我们看看这在实践中是如何工作的。

# 与 LLM 合作

现在有许多 AI 提供商允许您调用一个 LLM。在这本书中，我们将使用**GitHub 模型**，因为这是一个免费选项，您只需要一个 GitHub 账户即可使用它。要使用 GitHub 模型，您要么需要在 GitHub Codespaces 中启动您的项目，要么设置一个具有正确权限的**个人访问令牌（PAT**）。您需要令牌的原因是您正在调用一个 API，令牌用作携带令牌以验证请求。例如，要通过**Ollama**使用本地 AI 模型，您就不需要令牌。您可以直接在源代码中输入令牌，但出于安全原因，建议将其保存在环境变量中。

那么，如果我们以前从未与 AI 合作过，我们需要了解什么呢？嗯，想法是发送一个提示并获取一个响应。提示是描述您希望 LLM 执行的自然语言文本。响应也是包含您提示答案的自然语言文本。

然而，要结合 MCP（多通道处理）使用 LLM，想法是让 LLM 根据特定的提示指示要调用哪些函数。例如，对于提示`Add 1 and 2`，如果我们在 MCP 服务器中定义了一个名为`add`的函数，并且具有相应的参数，LLM 应该指示使用参数`a=1`和`b=2`调用`add`函数。

下面是调用代码的示例。在下面的代码中，我们将调用一个 GitHub 模型，所以请确保您有一个 GitHub 账户并且已经创建了一个具有正确权限的 PAT（个人访问令牌），或者开始在 GitHub Codespaces 中启动它。

需要定义的重要事项如下：

+   GitHub 模型的端点

+   要使用的模型 - 在这种情况下，`gpt-4o`

+   要发送的提示 - 您可以从用户输入中收集这些数据或将其硬编码，如下面的示例所示

+   要使用的函数 - 在这种情况下，我们将使用我们在上一章中创建的`add`工具

完全可以只发送一个提示并获取一个响应，但在这个例子中，我们希望 LLM 指示要调用哪个函数，因此我们将发送一个函数定义。**函数定义**是一个 JSON 对象，它描述了函数名称、描述和参数。参数使用 JSON 模式定义。

```py
# json description of functions
functions = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "type": "function",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]
# get token from environment variable
token = os.environ["GITHUB_TOKEN"]
# the endpoint for GitHub Models
endpoint = "https://models.github.ai/inference"
# the model to use
model_name = "gpt-4o"

# creation of chat client
client = OpenAI(
    base_url=endpoint,
    api_key=token
)
print("CALLING LLM")
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    model=model_name,
    tools = functions,
    # Optional parameters
    temperature=1.,
    max_tokens=1000,
    top_p=1\.  
)
response_message = response.choices[0].message
print("LLM RESPONSE: ", response_message) 
```

值得注意的是，除了提示之外，你还可以向 LLM 发送配置，例如 `temperature`、`max_tokens` 和 `top p`。我们不会深入探讨这些参数的含义——你可以在 OpenAI 文档中了解更多信息——但简而言之，它们控制着 LLM 响应的随机性和创造性，以及所谓的上下文窗口的大小。

# 练习：集成 LLM

因此，让我们看看我们如何将 LLM 集成到客户端中。目标是拥有更好的用户体验并抽象出使用服务器的复杂性。为了达到这个目标，我们需要采取以下步骤：

1.  **列出服务器功能**：通过列出功能，我们可以看到我们有什么可用

1.  **将功能列表转换为 LLM 工具**：来自 MCP 服务器的功能不能直接由 LLM 使用，因此我们需要将它们转换为 LLM 可以理解的形式

1.  **管理用户输入**：这将允许用户输入自然语言请求，我们的客户端上的 LLM 将发出完成请求，并在这样做的同时，告诉我们使用哪个功能以及发送哪些参数。

让我们来做这件事！

## 列出服务器功能

如果你，例如，正在使用他人的 MCP 服务器，那么你可能将构建此客户端作为你做的第一件事。在这种情况下，在继续之前，请确保安装 MCP SDK。

第一步与之前没有区别。我们需要列出服务器上的功能。这是通过调用列出工具来完成的，如下所示：

```py
tools = await session.list_tools()
print("LISTING TOOLS")
for tool in tools.tools:
    print("Tool: ", tool.name) 
```

## 将功能列表转换为 LLM 工具

我们下一步很重要，因为我们将要转换功能列表，使其成为 LLM 可以理解的形式。这将为我们下一步做好准备，我们将使用 LLM 来确定使用哪个功能。以下是转换代码：

1.  让我们添加转换代码函数：

    ```py
    def to_llm_tool(tool):
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "type": "function",
                "parameters": {
                    "type": "object",
                    "properties": tool.inputSchema["properties"]
                }
            }
        }
        return tool_schema 
    ```

以下代码定义了一个函数，该函数接受一个工具作为输入并将其转换为 LLM 可以理解的形式。

1.  让我们调用转换代码：

    ```py
    functions = []
    for tool in tools.tools:
        print("Tool: ", tool.name)
        print("Tool", tool.inputSchema["properties"])
        functions.append(to_llm_tool(tool)) 
    ```

从代码中，你可以看到我们遍历工具的响应并调用每个工具的转换代码。

太好了，现在我们已经为使用 LLM 做好了充分的准备。下一步是管理用户输入并向 LLM 发送完成请求。在 LLM 的响应中，LLM 将告诉我们使用哪个函数以及参数。在这种情况下，要调用的函数将是服务器上的一个功能。

## 用户输入自然语言请求，LLM 发出完成请求

让我们看看我们现在如何调用 LLM，因为我们已经有了它可以使用的工具。

这分为两部分：第一部分是调用 LLM，第二部分是学习 LLM 是否返回了函数调用或通用响应。

1.  调用 LLM：

    ```py
    def call_llm(prompt, functions):
        token = os.environ["GITHUB_TOKEN"]
        endpoint = "https://models.github.ai/inference"
        model_name = "gpt-4o"
        client = OpenAI(
            base_url=endpoint,
            api_key=token,
        )
        print("CALLING LLM")
        response = client.complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model_name,
            tools = functions,
            # Optional parameters
            temperature=1.,
            max_tokens=1000,
            top_p=1\.  
        )
        response_message = response.choices[0].message
        print("LLM RESPONSE: ", response_message)
        functions_to_call = [] 
    ```

在前面的代码中，我们做了以下操作：

+   创建了一个函数，该函数接受提示和函数列表作为输入，并返回 LLM 响应。

+   使用`complete`方法调用 LLM，并将提示和函数作为参数传递。响应被打印到控制台。

1.  让我们通过添加以下代码来检查 LLM 是否返回了函数调用：

    ```py
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            print("TOOL: ", tool_call)
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            functions_to_call.append({ "name": name, "args": args })
    return functions_to_call 
    ```

在前面的代码中，我们做了以下操作：

+   通过检查响应消息中是否存在`tool_calls`属性来检查 LLM 是否返回了函数调用

+   遍历`tool_calls`并在控制台打印工具名称和参数

+   返回要调用的函数列表

## 客户端确定要使用哪个服务器功能以及要发送的参数

在这一点上，我们有了 LLM 的响应，它甚至返回了要调用的函数（如果有）。下一步是检查要调用的函数，并在需要时调用 MCP 服务器：

```py
functions_to_call = call_llm(prompt, functions)
# call suggested functions
for f in functions_to_call:
    result = await session.call_tool(f["name"], arguments=f["args"])
    print("TOOLS result: ", result.content) 
```

在前面的代码中，我们做了以下操作：

+   调用 LLM 并获取要调用的函数。

+   遍历要调用的函数，并使用`call_tool`方法调用 MCP 服务器。结果被打印到控制台。

很好，不是吗？用户现在肯定在感谢你让他们的生活变得更轻松，因为他们可以使用自然语言与服务器交互。

# 摘要

在本章中，我们探讨了如何构建可以连接到 MCP 服务器并使用其功能的客户端。我们首先构建了一个简单的客户端，它可以列出并调用服务器上的功能。然后我们通过集成 LLM 来改进客户端，这使得我们可以通过启用与服务器自然语言交互来创建更好的用户体验。

在下一章中，我们将探讨如何使用 VS Code 和 Claude 桌面版来消费 MCP 服务器。

# 作业

对于这个作业，你将再次专注于电子商务。你将构建一个用户可以与电子商务服务器交互的体验：

+   请求特定类别的产品

+   使用自然语言将产品添加到购物车中

# 解决方案

这里有一个客户端的解决方案。它涵盖了有和无 LLM 的客户端。

你可以通过[`github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter07/solutions/README.md`](https://github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Ch﻿apter07/solutions/README.md)访问解决方案。

# 习题

客户端可以在 MCP 服务器上访问什么？

+   A: 提示、工具和资源

+   B: 工具、提示和服务

+   C: 工具和提示

将 LLM 添加到客户端有什么好处？

+   A: 将 LLM 放在服务器上更好

+   B: 它使客户端更快

+   C: 客户端的 LLM 允许最终用户使用提示与服务器交互，这为用户提供了更好的体验

你可以通过[`github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter07/solutions/solution-quiz.md`](https://github.com/PacktPublishing/Learn-Model-Context-Protocol-with-Python/blob/main/Chapter07﻿/solutions/﻿solution-quiz.md)访问解决方案。

参考文献

+   **模型上下文协议**：[`modelcontextprotocol.io/introduction`](https://modelcontextprotocol.io/introduction)

+   **构建客户端**：[`modelcontextprotocol.io/quickstart/client`](https://modelcontextprotocol.io/quickstart/client)

+   **Python SDK**：[`github.com/modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk)

    |

    #### 现在解锁此书的独家优惠

    扫描此二维码或访问[`packtpub.com/unlock`](https://packtpub.com/unlock)，然后通过书名搜索此书。 | ![](img/Unlock-01.png)![](img/Unlock1.png) |

    | **注意**：在开始之前准备好您的购买发票。* |
    | --- |
