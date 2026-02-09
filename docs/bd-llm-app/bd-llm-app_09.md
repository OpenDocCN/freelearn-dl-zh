

# 第九章：与代码一起工作

在本章中，我们将介绍大型语言模型（LLM）的另一个伟大能力，即与编程语言一起工作。在前一章中，我们已经看到了这种能力的一瞥，即 SQL 数据库中的 SQL 查询生成。在本章中，我们将探讨 LLM 可以与代码一起使用的其他方式，从“简单”的代码生成到与代码仓库的交互，最后到让应用程序表现得像算法的可能性。到本章结束时，你将能够利用 LLM 进行与代码相关的项目，以及构建具有自然语言界面的 LLM 驱动应用程序来处理代码。

在本章中，我们将涵盖以下主题：

+   分析具有最佳代码能力的顶级 LLM

+   使用 LLM 进行代码理解和生成

+   构建充当算法的 LLM 代理

+   利用代码解释器

# 技术要求

要完成本章的任务，你需要以下内容：

+   Hugging Face 账户和用户访问令牌。

+   OpenAI 账户和用户访问令牌。

+   Python 3.7.1 或更高版本。

+   确保已安装以下 Python 包：`langchain`、`python-dotenv`、`huggingface_hub`、`streamlit`、`codeinterpreterapi`和`jupyter_kernel_gateway`。这些可以通过在终端中使用`pip install`轻松安装。

你可以在本书的 GitHub 仓库中找到所有代码和示例：`github.com/PacktPublishing/Building-LLM-Powered-Applications`。

# 选择合适的代码 LLM

在*第三章*中，我们描述了一个决策框架，用于决定给定应用中适当的 LLM。一般来说，所有 LLM 都具备代码理解和生成的知识；然而，其中一些特别擅长于此。更具体地说，有一些评估基准——例如 HumanEval——专门用于评估 LLM 与代码工作的能力。HumanEval One 的排行榜是一个很好的资源，可以确定表现最好的模型，可在[`paperswithcode.com/sota/code-generation-on-humaneval`](https://paperswithcode.com/sota/code-generation-on-humaneval)找到。HumanEval 是由 OpenAI 引入的一个基准，用于评估 LLM 的代码生成能力，其中模型根据其签名和文档字符串完成 Python 函数。它已被用于评估像 Codex 这样的模型，证明了其在衡量功能正确性方面的有效性。

在以下屏幕截图，你可以看到截至 2024 年 1 月的排行榜情况：

![](img/B21714_09_01.png)

图 9.1：2024 年 1 月的 HumanEval 基准

如您所见，大多数模型都是 GPT-4 的微调版本（以及 GPT-4 本身），因为它是基本上所有领域中最先进的 LLM。尽管如此，还有许多开源模型在代码理解和生成领域取得了惊人的成果，其中一些将在下一节中介绍。另一个基准是**大多数基本编程问题**（**MBPP**），一个包含 974 个 Python 编程任务的 Python 数据集，旨在由初级程序员解决。因此，在选择用于特定代码任务的模型时，查看这些基准以及其他类似的代码指标可能很有用（我们将在本章中看到一些针对特定代码 LLM 的进一步基准）。

在代码的范围内，以下您可以找到三个在市场上常用的额外基准：

+   **MultiPL-E**：HumanEval 的扩展，适用于许多其他语言，如 Java、C#、Ruby 和 SQL。

+   **DS-1000**：一个数据科学基准，测试模型是否能够用 Python 编写常见的数据分析任务的代码。

+   **技术助手提示**：一个测试模型是否能作为技术助手并回答编程相关请求的提示。

在本章中，我们将测试不同的 LLM：两个针对代码的（CodeLlama 和 StarCoder）和一个通用型，但在代码生成领域也具有新兴能力的（Falcon LLM）。

# 代码理解和生成

我们将要进行的第一个实验将是利用 LLM 进行代码理解和生成。这个简单的用例是 ChatGPT 推出后开发的许多 AI 代码助手的基础，GitHub Copilot 就是其中之一。

**定义**

GitHub Copilot 是一个由 AI 驱动的工具，它帮助开发者更高效地编写代码。它分析代码和注释，为单个行和整个函数提供建议。该工具由 GitHub、OpenAI 和 Microsoft 开发，支持多种编程语言。它可以执行各种任务，如代码补全、修改、解释和技术支持。

在这个实验中，我们将尝试三种不同的模型：Falcon LLM，这是我们已经在*第三章*中探讨过的；CodeLlama，这是 Meta AI 的 Llama 的微调版本；以及 StarCoder，这是一个针对代码的特定模型，我们将在接下来的章节中对其进行研究。

由于这些模型在本地机器上运行相当沉重，为此我将使用一个 Hugging Face Hub 推理端点，它是一个由 GPU 驱动的虚拟机。您可以为每个推理端点链接一个模型，然后将其嵌入到您的代码中，或者使用 LangChain 中可用的方便的库`HuggingFaceEndpoint`。

要开始使用您的推理端点，您可以使用以下代码：

```py
llm = HuggingFaceEndpoint(endpoint_url = "your_endpoint_url", task = 'text-generation',
        model_kwargs = {"max_new_tokens": 1100}) 
```

或者，您可以将您端点网页上提供的 Python 代码复制并粘贴到您的端点：

![](img/B21714_09_02.png)

图 9.2：Hugging Face 推理端点的用户界面

要创建您的 Hugging Face 推理端点，您可以按照[`huggingface.co/docs/inference-endpoints/index`](https://huggingface.co/docs/inference-endpoints/index)中的说明进行操作。

你可以始终利用第四章中描述的免费 Hugging Face API，但当你运行模型时，你必须预期会有一些延迟。

## Falcon LLM

Falcon LLM 是由阿布扎比的**技术创新研究所**（**TII**）开发的开源模型，并于 2023 年 5 月上市。它是一个自回归、仅解码器的 transformer，在 1 万亿个标记上训练，拥有 400 亿个参数（尽管它也被发布为一个较轻的版本，参数为 70 亿）。在第三章中讨论过，“小型”语言模型是 LLM 的新趋势的体现，它侧重于构建更轻的模型（参数更少），而不是训练数据集的质量。

要开始使用 Falcon LLM，我们可以按照以下步骤进行：

1.  我们可以利用 LangChain 中可用的 HuggingFaceHub 包装器（记得在`.env`文件中设置 Hugging Face API，将您的机密信息作为`os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN`传递）：

    ```py
    from langchain import HuggingFaceHub
    from langchain import PromptTemplate, LLMChain
    import os
    load_dotenv()hugging_face_api = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(
        repo_id=repo_id,  model_kwargs={"temperature": 0.2, "max_new_tokens": 1000}
    ) 
    ```

1.  现在我们已经初始化了模型，让我们要求它生成一个简单网页的代码：

    ```py
    prompt = """
    Generate a short html code to a simple webpage with a header, a subheader, and a text body.
    <!DOCTYPE html>
    <html>
    """
    print(llm(prompt)) 
    ```

以下是对应的输出：

```py
<head>
    <title>My Webpage</title>
</head>
<body>
    <h1>My Webpage</h1>
    <h2>Subheader</h2>
    <p>This is the text body.</p>
</body>
</html> 
```

1.  如果您将其保存为 HTML 文件并执行它，结果将如下所示：

![文本的特写  自动生成的描述](img/B21714_09_03.png)

图 9.3：FalconLLM 生成的示例网页

1.  我们也可以尝试生成一个 Python 函数来生成随机密码：

    ```py
    prompt = """
    Generate a python program that create random password with lenght of 12 characters, 3 numbers, one capital letter.
    """
    print(llm(prompt)) 
    ```

这里是我们的输出：

```py
import random
def generate_password():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    lenght = 12
    num = random.randint(1, 9)
    cap = random.randint(1, 9)
    password = ""
    for i in range(lenght):
        password += chars[random.randint(0, 9)]
    password += num
    password += cap
    return password
print(generate_password()) 
```

现在我们有一个名为`generate_password()`的函数，它使用随机函数根据我们的提示生成密码。

1.  最后，让我们做相反的事情，要求模型向我们解释上述代码：

    ```py
    prompt = """
    Explain to me the following code:
    def generate_password():
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        lenght = 12
        num = random.randint(1, 9)
        cap = random.randint(1, 9)
        password = ""
        for i in range(lenght):
            password += chars[random.randint(0, 9)]
        password += num
        password += cap
        return password
    print(generate_password())
    """
    print(llm(prompt)) 
    ```

这里是获得的结果：

```py
<p>The code generates a random password of length 12 characters that contains a mix of letters, numbers, and special characters. The password is then printed to the console.</p> 
```

总体而言，即使不是针对代码的，模型也能正确执行所有任务。请注意，这仍然是模型的“轻”版本（70 亿参数），但性能仍然很出色。

现在我们来调查 CodeLlama 的能力。

## CodeLlama

CodeLlama 是基于 Llama 2 的代码 LLM 系列，Llama 2 是由 Meta AI（在第三章中讨论过）开发的一种通用语言模型。CodeLlama 可以在各种编程语言中生成和讨论代码，例如 Python、C++、Java、PHP 等。CodeLlama 还可以执行补全，即根据周围上下文填充代码缺失部分的能力，以及遵循自然语言中的指令并生成符合所需功能代码。

模型有三种大小（7B、13B 和 34B 参数）和三种风味（基础模型、Python 微调模型和指令微调模型），以覆盖广泛的应用范围。CodeLlama 在 16k 个标记的序列上训练，可以处理多达 100k 个标记的输入。

在 Rozière Baptiste 等人于 2023 年 8 月发布的模型论文“Code Llama: Open Foundation Models for Code”中，作者们描述了如何将各种模型与代码理解和生成领域中最受欢迎的评估基准进行了测试，包括 HumanEval 和 MBPP。根据这些基准，CodeLlama 模型在 HumanEval 和 MBPP 上的得分分别达到了 53%和 55%。在这些令人瞩目的结果之上，令人震惊的是，经过 Python 微调的 CodeLlama 最小版本（7 亿参数）在 HumanEval 和 MBPP 上超过了 Llama 2 最大版本（70 亿参数）。

现在，让我们用这个模型进行一些测试。根据前面的章节，我们可以利用 Hugging Face 推理 API（按使用付费）或免费的 Hugging Face API（具有更高的延迟限制）来初始化模型。您可以按以下方式使用它：

```py
repo_id = "codellama/CodeLlama-7b-Instruct-hf"
llm = HuggingFaceHub(    repo_id=repo_id,  model_kwargs={"temperature": 0.2, "max_new_tokens": 1000}
) 
```

现在，让我们用一些代码任务来测试它。第一个任务将是优化 Python 代码，使其运行更高效。让我们看看我们的模型在这个任务上的表现。在下面的代码片段中，我们只是提示模型以更高效的方式重新生成提供的代码：

```py
prompt = """
Regenerate the below code in a more efficient way.
```

def factorial(n):

    result = 1

    for i in range(1, n + 1):

        result *= i

    return result

# 示例用法：

n = 5

print("Factorial of", n, "is", factorial(n))

```py
"""
print(llm(prompt)) 
```

这里是我们的输出：

```py
### Solution
```

def factorial(n):

    if n == 0:

        return 1

    else:

        return n * factorial(n - 1)

```py
### Explanation
The solution is a recursive function that calls itself until the base case is reached.
### References
- [Wikipedia](https://en.wikipedia.org/wiki/Factorial)
- [Math is Fun](https://www.mathsisfun.com/numbers/factorial.html)
- [Mathematics Stack Exchange](https://math.stackexchange.com/questions/117042/what-is-the-difference-between-factorial-and-permutation)
- [Mathematics Stack Exchange](https://math.stackexchange.com/questions/117042/what-is-the-difference-between-factorial-and-permutation) 
```

如您所见，模型能够使用递归方法，这更高效且“Pythonic”。它还提供了用户深入了解函数背后的数学理论的参考。

接下来，让我们利用模型的补全能力，初始化一个用于删除非 ASCII 字符的函数。

**定义**

**美国信息交换标准代码**（**ASCII**）是一个使用 7 位来表示 128 个字符的字符编码标准，如字母、数字、标点符号和控制代码。

非 ASCII 字符是指不属于 ASCII 标准且使用超过 7 位进行编码的字符。它们包括带有重音的字母、符号、象形文字和数学符号等特殊字符。非 ASCII 字符可以使用不同的标准进行编码，例如 Unicode、ISO 8859-1、Windows-1252 等。

例如，字母é是一个非 ASCII 字符，可以使用 Unicode 编码为 U+00E9，或者使用 Windows-1252 编码为 0xE9。

因此，以下是生成函数的代码：

```py
prompt = '''
Define a python function that remove non-ascii character from a string.
'''
print(llm(prompt)) 
```

下面的函数是我们收到的输出：

```py
def remove_non_ascii(string):
    return string.encode('ascii', 'ignore').decode('utf-8') 
```

现在，让我们利用模型作为错误修复者，提示它使用错误的功能，并要求它提供错误的原因以及如何修复的解释：

```py
prompt = """
Identify the bug in the following code:.
```

#错误的函数

import random

a = random.randint(1, 12)

b = random.randint(1, 12)

for i in range(10):

    question = "What is " + a + " x " + b + "? "

    answer = input(question)

    if answer = a * b:

        print (Well done!)

    else:

        print("No.")

```py
Explanation:
"""
print(llm(prompt)) 
```

这里是我们的输出：

```py
- The function is using the wrong syntax for the `randint` function. The correct syntax is `randint(start, end)`.
- The `for` loop is using the wrong syntax. The correct syntax is `for i in range(start, end)`.
- The `question` and `answer` variables are not being used in the loop.
- The `if` statement is not being used correctly. It should be `if answer == a * b:`.
- The `print` statement is not being used correctly. It should be `print(Well done!)`. 
```

最后，让我们用自然语言向模型请求为给定任务生成特定的 Python 代码，在我们的例子中，这将是要编写一个函数，该函数可以找到给定字符串中只包含唯一字符的最长子串：

```py
prompt = """
Write a Python function that finds the longest substring of a given string containing only unique characters.
"""
print(llm(prompt)) 
```

然后，我们得到以下函数作为输出：

```py
def longest_unique_substring(s):
    unique_count = 0
    longest_substring = ""
    for i in range(1, len(s)):
        if s[i] not in unique_count:
            unique_count += 1
            longest_substring = s[i]
    return longest_substring 
```

根据 Falcon LLM，在这种情况下，我们使用了模型的轻量版本（70 亿参数），仍然获得了很好的结果。这是一个完美的例子，说明了您想要通过应用程序解决的问题必须是在决定使用哪个 LLM 时的一个因素：如果您只对代码生成、补全、填充、调试或其他任何与代码相关的任务感兴趣，一个轻量级和开源的模型可能就足够了，而不是使用具有 70 亿参数的顶尖 GPT-4。

在下一节中，我们将介绍代码生成和理解背景下的第三个也是最后一个 LLM。

## StarCoder

StarCoder 模型是一个代码 LLM，可以执行各种任务，例如代码补全、代码修改、代码解释和技术支持。它是在 GitHub 的许可数据上训练的，包括来自 80 多种编程语言的 Git 提交、GitHub 问题和 Jupyter 笔记本。它具有超过 8,000 个 token 的上下文长度，这使得它能够处理比任何其他开源语言模型更多的输入。它还拥有改进的许可，简化了公司将其模型集成到产品中的过程。

StarCoder 模型在多个基准测试中被评估，以测试其在不同语言和领域编写和理解代码的能力，包括上述的 HumanEval 和 MBPP，模型在这两个测试中分别获得了 33.6%和 52.7%的分数。此外，它还与 MultiPL-E（在许多语言上，模型与 OpenAI 的 code-cushman-001 模型相匹配或表现更好）进行了测试，DS-1000（模型明显击败了 code-cushman-001 模型以及其他所有开放访问模型），以及 Tech Assistant Prompt（模型能够以相关和准确的信息回答各种查询）。

根据 Hugging Face 于 2023 年 5 月 4 日发布的一项调查，StarCoder 与其他模型相比，在 HumanEval 和 MBPP 作为基准的情况下，展现了出色的能力。您可以在下面的插图看到这项研究：

![计算机屏幕截图  描述自动生成](img/B21714_09_04.png)

图 9.4：各种 LLM 的评估基准结果。来源：[`huggingface.co/blog/starcoder`](https://huggingface.co/blog/starcoder)

要开始使用 StarCoder，我们可以遵循以下步骤：

1.  我们可以利用 LangChain 中可用的 HuggingFaceHub 包装器（请记住在`.env`文件中设置 Hugging Face API）：

    ```py
    import os
    from dotenv import load_dotenv
    load_dotenv()
    hugging_face_api = os.environ["HUGGINGFACEHUB_API_TOKEN"] 
    ```

1.  让我们设置 StarCoder 模型的 `repo_id` 并初始化它：

    ```py
    from langchain import HuggingFaceHub
    from langchain import PromptTemplate, LLMChain
    repo_id = "bigcode/starcoderplus"
    llm = HuggingFaceHub(
        repo_id=repo_id,  model_kwargs={"temperature": 0.2, "max_new_tokens": 500}
    ) 
    ```

**注意**

StarCoder 是 Hugging Face Hub 上的一个门控模型，这意味着在能够连接到它之前，您需要直接从 bigcode/starcoderplus 仓库请求访问权限。

现在我们已经设置好了，让我们开始要求我们的模型编译一些代码。首先，我们将要求它生成一个 Python 函数来生成第 n 个斐波那契数：

```py
prompt = """
How can I write a Python function to generate the nth Fibonacci number?
"""
print(llm(prompt)) 
```

**定义**

斐波那契数列是一个以 0 和 1 开头的数学序列，每个后续数字是前两个数字的和。例如，斐波那契数列的前 10 个数字是 0, 1, 1, 2, 3, 5, 8, 13, 21 和 34。

计算第 n 个斐波那契数（用 F(n) 表示）有不同的方法。一种方法是使用递归公式：

![img/B21714_09_001.png](img/B21714_09_001.png)

这意味着要找到 F(n)，我们首先需要找到 F(n-1) 和 F(n-2)，然后将它们相加。这对于任何大于或等于 2 的 n 都有效。对于 n 等于 0 或 1，我们只需返回 n 作为答案。

我们随后看到以下输出：

![A screenshot of a computer Description automatically generated](img/B21714_09_05.png)

图 9.5：StarCode 生成的斐波那契函数示例

如您所见，它还提出了不同的方法来解决问题，并附有解释。

现在让我们要求模型生成一个网页，与电脑玩井字棋：

```py
prompt = """
Generate the html code for a single page website that let the user play tic tac toe.
 """
print(llm(prompt)) 
```

这里是相应的输出：

```py
 ## How to use
```

git clone https://github.com/Mohamed-Elhawary/tic-tac-toe.git

cd tic-tac-toe

python3 -m http.server

```py
## License
 [MIT](https://choosealicense.com/licenses/mit/)
<|endoftext|> 
```

有趣的是，这个模型并没有生成整个代码；相反，它给出了克隆和运行一个可以完成此结果的 git 仓库的指令。

最后，StarCoder 也可以作为 VS Code 的扩展使用，作为您的代码伴侣。您可以在以下截图中的 **HF Code Autocomplete** 中找到它：

![A screenshot of a computer Description automatically generated](img/B21714_09_06.png)

图 9.6：由 StarCoder 驱动的 Hugging Face 代码自动补全扩展

一旦启用，您可以看到，在编译代码时，StarCoder 将提供代码补全建议。例如：

![img/B21714_09_07.png](img/B21714_09_07.png)

图 9.7：给定函数描述的完成建议截图

如您所见，我注释了我的代码，描述了一个生成第 n 个斐波那契数的函数，然后开始定义这个函数。自动地，我得到了 StarCoder 的自动补全建议。

代码理解和生成是 LLM 的强大功能。在这些能力之上，我们还可以考虑进一步的应用，这些应用超越了代码生成。事实上，代码也可以被视为一个后端推理工具，用于提出解决复杂问题的解决方案，如能源优化问题，而不是算法任务。为此，我们可以利用 LangChain 创建强大的代理，它们可以*表现得像算法一样*。在下一节中，我们将看到如何做到这一点。

# 作为算法行动

一些问题由于其本质而复杂，仅利用“仅”LLM 的分析推理技能很难解决。然而，LLM 仍然足够智能，能够理解问题整体，并利用它们的编码能力来解决它们。

在这个背景下，LangChain 提供了一个工具，使 LLM 能够以“Python”的方式进行推理，这意味着由 LLM 驱动的代理将利用 Python 来解决复杂问题。这个工具是 Python REPL，它是一个简单的 Python 外壳，可以执行 Python 命令。Python REPL 之所以重要，是因为它允许用户使用 Python 语法进行复杂计算、生成代码，并与语言模型进行交互。在本节中，我们将介绍该工具的一些功能示例。

让我们首先使用 LangChain 中的`create_python_agent`类初始化我们的代理。为此，我们需要向这个类提供一个 LLM 和一个工具，在我们的例子中，这个工具将是 Python REPL：

```py
import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
 from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools import PythonREPLTool
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
agent_executor = create_python_agent(
    llm=model,
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
) 
```

和往常一样，在开始与代理一起工作之前，让我们首先检查默认提示：

```py
print(agent_executor.agent.llm_chain.prompt.template) 
```

下面是我们的输出：

![计算机程序截图  自动生成的描述](img/B21714_09_08.png)

图 9.8：Python 代理的默认提示

现在，让我们从一个简单的查询开始，要求模型根据篮球运动员的样本属性生成散点图：

```py
query = """
In a different basketball game, we have the following player stats:
- Player A: 38 points, 10 rebounds, 7 assists
- Player B: 28 points, 9 rebounds, 6 assists
- Player C: 19 points, 6 rebounds, 3 assists
- Player D: 12 points, 4 rebounds, 2 assists
- Player E: 7 points, 2 rebounds, 1 assist
Could you create a scatter plot graph in Seaborn talk mode for each player, where the y-axis represents the number of points, the x-axis represents the number of rebounds, and use 'o' as the marker? Additionally, please label each point with the player's name and set the title as "Team Players."
"""
agent_executor.run(query) 
```

然后，我们得到以下输出：

```py
Invoking: `Python_REPL` with `import seaborn as sns
import matplotlib.pyplot as plt
# Player stats
players = ['Player A', 'Player B', 'Player C', 'Player D', 'Player E']
points = [38, 28, 19, 12, 7]
rebounds = [10, 9, 6, 4, 2]
# Create scatter plot
sns.scatterplot(x=rebounds, y=points, marker='o')
# Label each point with player's name
for i, player in enumerate(players):
    plt.text(rebounds[i], points[i], player, ha='center', va='bottom')
# Set title
plt.title('Team Players')
# Show the plot
plt.show()` 
```

该输出附带了以下基于球员统计数据的图表：

![计算机程序截图  自动生成的描述](img/B21714_09_09.png)

图 9.9：Python 代理生成的示例绘图

让我们看看另一个例子。假设我们想要根据一些特征（如卧室数量或房屋大小）预测房屋价格。为此，我们可以要求我们的代理设计和训练一个模型，给出给定房屋的结果。例如，让我们考虑以下提示：

```py
query = """
I want to predict the price of a house given the following information:
- the number of rooms
- the number of bathrooms
- the size of the house in square meters
Design and train a regression model to predict the price of a house. Generate and use synthetic data to train the model.
Once the model is trained, tell me the price of a house with the following features:
- 2 rooms
- 1 bathroom
- 100 square meters
"""
agent_executor.run(query) 
```

在这里，我们要求代理在合成数据（代表具有各种房间、浴室和面积配置的房屋，每个配置都有一个相关的价格作为因变量）上训练回归模型，以给出具有上述特征的房屋的估计价格。让我们看看输出：

```py
> Entering new AgentExecutor chain...
Invoking: `Python_REPL` with `import numpy as np
from sklearn.linear_model import LinearRegression
# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 houses with 3 features: rooms, bathrooms, size
y = 100000 * X[:, 0] + 200000 * X[:, 1] + 300000 * X[:, 2] + 50000  # Price = 100k * rooms + 200k * bathrooms + 300k * size + 50k
# Train the regression model
model = LinearRegression()
model.fit(X, y)
# Predict the price of a house with the given features
features = np.array([[2, 1, 100]])
predicted_price = model.predict(features)
predicted_price`
responded: {content}
The predicted price of a house with 2 rooms, 1 bathroom, and 100 square meters is approximately $550,000.
> Finished chain.
'The predicted price of a house with 2 rooms, 1 bathroom, and 100 square meters is approximately $550,000.' 
```

如您所见，代理能够生成合成训练数据，使用`sklearn`库训练适当的回归模型，并使用模型预测我们提供的房屋价格。

采用这种方法，我们可以编程一个智能体在实时场景中充当算法。想象一下，例如，我们想要设计一个智能体，使其能够在智能建筑环境中解决优化问题。目标是优化建筑中的**供暖、通风和空调**（**HVAC**）设定点，以最小化能源成本同时确保居住者的舒适度。让我们定义问题的变量和约束条件：目标是调整三个区域中每个区域的温度设定点，在指定的舒适范围内，同时考虑每度、每小时的能源成本变化。

目标是在能源效率和居住者舒适度之间取得平衡。以下你可以找到问题的描述，以及我们变量和约束条件的初始化（每个区域的能源成本、每个区域的初始温度和舒适度范围）：

```py
query = """
**Problem**:
You are tasked with optimizing the HVAC setpoints in a smart building to minimize energy costs while ensuring occupant comfort. The building has three zones, and you can adjust the temperature setpoints for each zone. The cost function for energy consumption is defined as:
- Zone 1: Energy cost = $0.05 per degree per hour
- Zone 2: Energy cost = $0.07 per degree per hour
- Zone 3: Energy cost = $0.06 per degree per hour
You need to find the optimal set of temperature setpoints for the three zones to minimize the energy cost while maintaining a comfortable temperature. The initial temperatures in each zone are as follows:
- Zone 1: 72°F
- Zone 2: 75°F
- Zone 3: 70°F
The comfort range for each zone is as follows:
- Zone 1: 70°F to 74°F
- Zone 2: 73°F to 77°F
- Zone 3: 68°F to 72°F
**Question**:
What is the minimum total energy cost (in dollars per hour) you can achieve by adjusting the temperature setpoints for the three zones within their respective comfort ranges?
"""
agent_executor.run(query) 
```

我们随后得到以下输出（你可以在书籍的 GitHub 仓库中找到整个推理链）：

```py
> Entering new AgentExecutor chain...
Invoking: `Python_REPL` with `import scipy.optimize as opt
# Define the cost function
def cost_function(x):
    zone1_temp = x[0]
    zone2_temp = x[1]
    zone3_temp = x[2]

    # Calculate the energy cost for each zone
    zone1_cost = 0.05 * abs(zone1_temp - 72)
    zone2_cost = 0.07 * abs(zone2_temp - 75)
    zone3_cost = 0.06 * abs(zone3_temp - 70)
[…]
> Finished chain.
'The minimum total energy cost that can be achieved by adjusting the temperature setpoints for the three zones within their respective comfort ranges is $0.15 per hour.' 
```

智能体能够解决智能建筑优化问题，在给定一些约束条件下找到最小的总能源成本。在优化问题的范围内，还有其他一些使用案例，这些模型可以用类似的方法解决，包括：

+   **供应链优化**：优化商品物流和分销，以最小化运输成本、减少库存并确保及时交付。

+   **投资组合优化**：在金融领域，使用算法构建投资组合，以最大化回报同时管理风险。

+   **路线规划**：为送货卡车、紧急服务或共享出行平台规划最优路线，以最小化旅行时间和燃料消耗。

+   **制造流程优化**：优化制造流程，以最小化浪费、能源消耗和生产成本，同时保持产品质量。

+   **医疗资源分配**：在大流行或其他医疗危机期间，高效分配医疗资源，如医院床位、医疗人员和设备。

+   **网络路由**：优化计算机网络中的数据路由，以减少延迟、拥塞和能源消耗。

+   **车队管理**：优化车队（如出租车或送货货车）的使用，以降低运营成本并提高服务质量。

+   **库存管理**：确定最优库存水平和再订购点，以最小化存储成本同时防止缺货。

+   **农业规划**：根据天气模式和市场需求优化作物种植和收获计划，以最大化产量和利润。

+   **电信网络设计**：设计电信网络的布局，以提供覆盖范围同时最小化基础设施成本。

+   **废物管理**：优化垃圾收集车的路线以减少燃料消耗和排放。

+   **航空公司机组人员排班**：创建符合劳动法规且能最小化航空公司成本的航班机组人员排班。

Python REPL 代理非常出色；然而，它也有一些限制：

+   它不允许进行文件输入输出（FileIO），这意味着它不能与您的本地文件系统进行读写操作。

+   每次运行后都会忘记变量，这意味着您不能在模型响应后追踪初始化的变量。

为了绕过这些限制，在下一节中，我们将介绍一个基于 LangChain 代理构建的开源项目：代码解释器 API。

# 利用代码解释器

“代码解释器”这个名字是由 OpenAI 提出的，指的是最近为 ChatGPT 开发的插件。代码解释器插件允许 ChatGPT 在多种编程语言中编写和执行计算机代码。这使得 ChatGPT 能够执行诸如计算、数据分析以及生成可视化等任务。

代码解释器插件是专为具有安全核心原则的语言模型设计的工具之一。它帮助 ChatGPT 访问最新信息，运行计算或使用第三方服务。该插件目前处于私有测试阶段，仅对选定开发者和 ChatGPT Plus 用户开放。

虽然 OpenAI 的代码解释器（Code Interpreter）仍然不提供 API，但有一些开源项目已经将此插件的概念应用于开源 Python 库中。在本节中，我们将利用 Shroominic 的工作，该工作可在[`github.com/shroominic/codeinterpreter-api`](https://github.com/shroominic/codeinterpreter-api)找到。您可以通过`pip install codeinterpreterapi`来安装它。

根据 Shroominic 发布的博客文章，他是代码解释器 API 的作者（您可以在[`blog.langchain.dev/code-interpreter-api/`](https://blog.langchain.dev/code-interpreter-api/)阅读），该 API 基于 LangChain 代理`OpenAIFunctionsAgent`。

**定义**

OpenAIFunctionsAgent 是一种可以使用 OpenAI 函数能力通过 LLM 响应用户提示的代理。该代理由支持使用 OpenAI 函数的模型驱动，并且可以访问一组它可以用来与用户交互的工具。

OpenAIFunctionsAgent 也可以集成自定义函数。例如，您可以使用 Yahoo Finance 定义自定义函数来获取当前的股票价格或股票表现。OpenAIFunctionsAgent 可以使用 ReAct 框架来决定使用哪个工具，并且可以使用内存来记住之前的对话交互。

该 API 已经内置了一些工具，例如导航网络以获取最新信息的能力。

然而，与我们在上一节中介绍的 Python REPL 工具相比，最大的不同之处在于代码解释器 API 实际上可以执行它生成的代码。实际上，当代码解释器会话开始时，由于底层名为 CodeBox 的 Python 执行环境，您的设备上会启动一个 Jupyter 内核的迷你版。

要开始在您的笔记本中使用代码解释器，您可以按照以下方式安装所有依赖项：

```py
!pip install "codeinterpreterapi[all]" 
```

在这种情况下，我会要求它生成特定时间范围内 COVID-19 病例的图表：

```py
from codeinterpreterapi import CodeInterpreterSession
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']
# create a session
async with CodeInterpreterSession() as session:
    # generate a response based on user input
    response = await session.generate_response(
        "Generate a plot of the evolution of Covid-19 from March to June 2020, taking data from web."
    )
    # output the response
print("AI: ", response.content)
    for file in response.files:
        file.show_image() 
```

下面是生成的输出，包括一个图表，显示了指定时间段的全球确诊病例数：

```py
AI:  Here is the plot showing the evolution of global daily confirmed COVID-19 cases from March to June 2020\. As you can see, the number of cases has been increasing over time during this period. Please note that these numbers are cumulative. Each point on the graph represents the total number of confirmed cases up to that date, not just the new cases on that day. 
```

![一个向上的线条图表，描述自动生成](img/B21714_09_10.png)

图 9.10：由代码解释器 API 生成的折线图

如您所见，代码解释器不仅用解释回答了问题，还用图表进行了展示。

让我们再试一个，这次也利用其搜索最新信息的实时能力。在下面的代码片段中，我们要求模型绘制过去 5 天内 S&P 500 指数的价格：

```py
async with CodeInterpreterSession() as session:
    # generate a response based on user input
    response = await session.generate_response(
        "Generate a plot of the price of S&P500 index in the last 5 days."
    )
    # output the response
print("AI: ", response.content)
    for file in response.files:
        file.show_image() 
```

然后，我们得到以下输出，以及一个显示过去 5 天内 S&P 500 指数价格的折线图：

```py
AI:  Here is the plot of the S&P 500 index for the last 5 days. The y-axis represents the closing price of the index, and the x-axis represents the date. 
```

![一个带有数字和线的图表，描述自动生成](img/B21714_09_11.png)

图 9.11：由代码解释器 API 绘制的 S&P 500 指数价格

最后，我们可以向代码解释器提供本地文件，以便它可以对该特定数据进行一些分析。例如，我从 Kaggle 下载了泰坦尼克号数据集，网址为 [`www.kaggle.com/datasets/brendan45774/test-file`](https://www.kaggle.com/datasets/brendan45774/test-file)。泰坦尼克号数据集是机器学习的一个流行数据集，描述了泰坦尼克号上个别乘客的生存状态。它包含诸如年龄、性别、舱位、船票和是否幸存等信息。

一旦数据集下载完成，我就将其作为参数传递给模型，如下所示：

```py
from codeinterpreterapi import CodeInterpreterSession, File
#os.environ["HUGGINGFACEHUB_API_TOKEN"]
os.environ['OPENAI_API_KEY'] = "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"
os.environ['VERBOSE'] = "True"
async with CodeInterpreterSession() as session:
        # define the user request
        user_request = "Analyze this dataset and plot something interesting about it."
        files = [
            File.from_path("drive/MyDrive/titanic.csv"),
        ]
        # generate the response
        response = await session.generate_response(
            user_request, files=files
        )
        # output to the user
print("AI: ", response.content)
        for file in response.files:
            file.show_image() 
```

然后，我们得到以下输出：

```py
AI:  The plot shows the survival count based on the passenger class. It appears that passengers in the 3rd class had a significantly lower survival rate compared to those in the 1st and 2nd classes. This could suggest that the class of the passengers might have influenced their survival, possibly due to factors such as the location of their cabins and access to lifeboats.
These are just a few examples of the kind of insights we can extract from this dataset. Depending on the specific questions you're interested in, we could perform further analysis. For example, we could look at the survival rate based on age, or investigate whether the fare passengers paid had any influence on their survival. 
```

![一个图表的截图，描述自动生成](img/B21714_09_12.png)

图 9.12：代码解释器 API 生成的示例图表

如您所见，模型能够生成条形图，展示了按性别（在第一个图表中）和类别（在第二个图表中）分组的生存状态。

代码解释器插件，连同特定于代码的 LLM 和 Python 代理，是 LLM 对软件开发世界产生巨大影响的绝佳例子。这可以总结为两个主要能力：

+   LLMs 可以理解和生成代码，因为它们已经在大量的编程语言、GitHub 仓库、StackOverflow 对话等上进行了训练。因此，除了自然语言之外，编程语言也成为了它们参数化知识的一部分。

+   LLMs 可以理解用户的意图，并作为推理引擎激活 Python REPL 或 Code Interpreter 等工具，然后通过与代码一起工作来提供响应。

总体而言，LLMs 正在超越消除自然语言和机器语言之间差距的目标：相反，它们正在将两者结合起来，以便它们可以相互利用来响应用户的查询。

# 摘要

在本章中，我们探讨了多种利用 LLMs 与代码一起工作的方法。在回顾了如何评估 LLMs 以及在选择 LLM 进行代码相关任务时需要考虑的具体评估基准后，我们深入进行了实际实验。

我们从我们都至少尝试过一次的“普通”应用开始，即使用 ChatGPT 进行代码理解和生成。为此，我们利用了三个不同的模型——Falcon LLM、CodeLlama 和 StarCoder，每个模型都取得了非常好的结果。

我们随后继续探讨 LLMs 的编码能力在现实世界中的额外应用。事实上，我们看到了如何将特定于代码的知识用作解决复杂问题的助推器，例如算法或优化任务。此外，我们还介绍了代码知识不仅可以在 LLM 的后端推理中使用，还可以在实际的工作笔记本中执行，利用 Code Interpreter API 的开源版本。

通过本章，我们越来越接近第二部分的结尾。到目前为止，我们已经涵盖了 LLMs 的多种能力，同时始终处理语言数据（自然或代码）。在下一章中，我们将看到如何进一步迈向多模态，构建能够处理多种格式数据的强大多模态代理。

# 参考文献

+   Code Interpreter API 的开源版本：[`github.com/shroominic/codeinterpreter-api`](https://github.com/shroominic/codeinterpreter-api)

+   StarCoder: [`huggingface.co/blog/starcoder`](https://huggingface.co/blog/starcoder)

+   Python REPL 的 LangChain 代理：[`python.langchain.com/docs/integrations/toolkits/python`](https://python.langchain.com/docs/integrations/toolkits/python)

+   关于 Code Interpreter API 的 LangChain 博客：[`blog.langchain.dev/code-interpreter-api/`](https://blog.langchain.dev/code-interpreter-api/)

+   泰坦尼克号数据集：[`www.kaggle.com/datasets/brendan45774/test-file`](https://www.kaggle.com/datasets/brendan45774/test-file)

+   HF 推理端点：[`huggingface.co/docs/inference-endpoints/index`](https://huggingface.co/docs/inference-endpoints/index)

+   CodeLlama 模型卡片：[`huggingface.co/codellama/CodeLlama-7b-hf`](https://huggingface.co/codellama/CodeLlama-7b-hf)

+   Code Llama：开源代码基础模型，*Rozière. B.，等人*（2023 年）：[`arxiv.org/abs/2308.12950`](https://arxiv.org/abs/2308.12950)

+   鹰隼 LLM 模型卡片：[`huggingface.co/tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct)

+   StarCoder 模型卡片：[`huggingface.co/bigcode/starcoder`](https://huggingface.co/bigcode/starcoder)

# 加入我们的 Discord 社区

加入我们的 Discord 空间，与作者和其他读者进行讨论：

[`packt.link/llm`](https://packt.link/llm)

![](img/QR_Code214329708533108046.png)
