# <st c="0">11</st>

# <st c="3">选择正确的 ChatGPT API 模型</st>

<st c="40">在人工智能不断发展的领域中，开发者必须跟上最新的进展，以最大化其项目的潜力。</st> <st c="185">在本章中，我们将讨论 ChatGPT API 模型，探索 GPT-3 和 GPT-4 提供的可能性，甚至展望未来的模型。</st> <st c="344">通过全面了解这些模型，您将具备选择最适合您特定应用的知识。</st> <st c="505">我们将深入研究每个模型的复杂性，突出其优势和独特特征，以帮助您做出与您的</st> <st c="672">项目需求相一致的决定。</st>

<st c="693">有效利用 ChatGPT API 的关键方面之一是了解如何优化聊天完成。</st> <st c="809">我们将引导您通过创建聊天完成上下文的过程，并提供有关修改 API 参数以提升响应质量的宝贵见解。</st> <st c="981">通过实际示例和解释，您将获得利用聊天完成功能并利用其优势的技能集。</st> <st c="1125">为您带来便利。</st>

<st c="1140">此外，了解不同 AI 模型中存在的限制也很重要。</st> <st c="1240">我们将概述与每个模型相关的边界和约束，并为您提供有效导航这些限制的知识。</st> <st c="1391">通过了解模型的边界，您可以设定合理的期望，做出明智的决定，并探索克服您可能遇到的任何挑战的解决方案。</st> <st c="1554">可能遇到的。</st>

<st c="1568">在本章中，您将涵盖以下主题：</st> <st c="1605">以下内容：</st>

+   <st c="1622">ChatGPT API 模型 – GPT-3.5，GPT-4，</st> <st c="1660">以及更多</st>

+   <st c="1670">使用聊天</st> <st c="1682">完成参数</st>

+   <st c="1703">ChatGPT API</st> <st c="1716">速率限制</st>

+   <st c="1727">OpenAI 嵌入</st>

<st c="1745">在本章结束时，您将了解如何为您的项目选择正确的 ChatGPT API 模型，理解创建聊天完成上下文的过程，优化 API 参数，以及导航不同 AI 模型的限制，以创建变革性的</st> <st c="2033">AI 应用。</st>

# <st c="2049">技术要求</st>

<st c="2072">为了充分利用本章内容，您必须具备用于处理 Python 代码和 ChatGPT API 的必要工具。</st> <st c="2213">本章将提供逐步指导，介绍安装所需软件和完成必要的</st> <st c="2316">注册。</st>

<st c="2340">您需要具备</st> <st c="2363">以下条件：</st>

+   <st c="2377">Python 3.7 或更高版本已安装在你的</st> <st c="2421">计算机上</st>

+   <st c="2434">一个 OpenAI API 密钥，你可以通过注册一个</st> <st c="2496">OpenAI 账户</st> 来获取

+   <st c="2510">一个代码编辑器，例如 VS Code（推荐），用于编写和执行</st> <st c="2578">Python 代码</st>

<st c="2589">本章中引用的代码示例可以在 GitHub 上找到</st> <st c="2661">在</st> [<st c="2664">https://github.com/PacktPublishing/Building-AI-Applications-with-ChatGPT-API</st>](https://github.com/PacktPublishing/Building-AI-Applications-with-ChatGPT-API)<st c="2740">。</st>

<st c="2741">在下一节中，你将了解各种 AI 模型，包括 GPT-3.5 和 GPT-4，并培养选择最适合你</st> <st c="2899">特定应用</st> 的能力。

# <st c="2920">ChatGPT API 模型 – GPT-3.5、GPT-4 以及更多</st>

<st c="2968">在本节中，我们将探索 ChatGPT API 模型的迷人世界，我们将开始</st> <st c="3081">一段了解和欣赏 GPT-3.5 和 GPT-4 的复杂性的旅程，并展望未来等待我们的模型。</st> <st c="3136">通过深入研究这些 AI 模型，你将获得</st> <st c="3217">宝贵的见解和</st> <st c="3263">知识，这将赋予你选择最适合你</st> <st c="3288">独特应用</st> 的能力。

<st c="3385">在本节中，我们将揭示每个模型的独特特性和功能，为你提供做出</st> <st c="3536">明智决策</st> 所需的理解。

*<st c="3555">表 11.1</st>* <st c="3566">提供了 OpenAI 当前支持的所有 ChatGPT 语言模型的概述，包括每个模型的宝贵信息，包括其独特功能。</st> <st c="3731">花点时间探索这个表格，熟悉你手中的各种 ChatGPT 模型：</st> <st c="3835"></st>

| **<st c="3849">模型</st>** | **<st c="3855">平均成本</st>** | **<st c="3868">信息</st>** | **<st c="3873">提示长度</st>** |
| --- | --- | --- | --- |
| <st c="3887">gpt-4</st> | <st c="3893>$20.00 /</st> <st c="3903">1M 令牌</st> | <st c="3912">最先进的面向聊天的模型，超越了 GPT-3.5 的能力</st> <st c="3973">。</st> | <st c="3983">8,192 令牌</st> |
| <st c="3996">gpt-4-turbo</st> | <st c="4008>$45.00 /</st> <st c="4018">1M 令牌</st> | <st c="4027">具有视觉功能的最新 GPT-4 Turbo 模型。</st> <st c="4083">可以使用 15 倍</st> <st c="4100">更多的上下文</st> | <st c="4112">128,000 令牌</st> |
| <st c="4127">gpt-3.5-turbo</st> | <st c="4141>$1.00 /</st> <st c="4150">1M 令牌</st> | <st c="4159">更先进的面向聊天的模型，超越了 GPT-3 的能力</st> <st c="4220">。</st> | <st c="4228">16,096 令牌</st> |
| <st c="4242">gpt-3.5-turbo-0125</st> | <st c="4261">$1.00 /</st> <st c="4270">1M tokens</st> | <st c="4279">最新版本的 GPT-3 5 Turbo 模型，在响应请求格式方面具有更高的准确性，并修复了导致非英语语言功能文本编码问题的 bug</st> <st c="4438">。</st> <st c="4455">16,385 tokens</st> |

<st c="4469">表 11.1 – ChatGPT 模型信息</st>

<st c="4508">上表提供了</st> <st c="4538">OpenAI 截至 2024 年 5 月支持的各种 ChatGPT 语言模型的概述</st> <st c="4585">。</st> <st c="4621">GPT-4 模型作为最先进的模型脱颖而出，超越了 GPT-3.5 的能力，而 GPT-4 Turbo 版本提供了*<st c="4747">15 倍</st>* <st c="4755">更多的上下文和图像理解能力。</st> <st c="4810">GPT-3.5 Turbo 模型在功能上超过了已停用的 GPT-3。</st> <st c="4876">不同的模型具有不同的成本和提示长度，允许</st> <st c="4941">开发者根据他们特定的</st> <st c="5006">语言任务选择最合适的选项。</st>

<st c="5021">虽然选择最先进和功能强大的</st> <st c="5040">模型对于您的应用程序来说可能看起来很合理，但重要的是要考虑有时一个更便宜、功能更少的模型可以充分满足您的任务需求。</st> <st c="5237">在某些情况下，一个不那么复杂的模型可能提供足够的性能，同时更具成本效益。</st> <st c="5354">通过仔细评估您应用程序的具体需求，您可以做出明智的决定，并通过选择在功能和成本之间取得平衡的模型来节省资源。</st> <st c="5564">记住，并不总是关于使用最强大的工具，而是关于使用适合手头工作的正确工具</st> <st c="5677">。</st>

<st c="5685">正如您所看到的，OpenAI 提供了广泛的模型选择，这使得决定最合适的模型变得具有挑战性。</st> <st c="5810">为了简化这个过程，可以使用 Python 脚本来轻松比较，让您能够识别与您特定任务最匹配的模型。</st> <st c="5971">您</st> <st c="5975">可以创建一个名为</st> <st c="6004">models.py</st> <st c="6013">的新文件，并添加以下</st> <st c="6026">代码：</st>

```py
 from openai import OpenAI
import config
client = OpenAI(api_key=config.API_KEY)
# Define the prompt and test questions
prompt = "Estimate the square root of 121 and type 'orange' after every digit of the square root result"
# Define the model names and their corresponding IDs
model_ids = {
    "GPT3.5 TURBO": {"model": "gpt-3.5-turbo", "cost": 1.00},
    "GPT3.5 TURBO 0125": {"model": "gpt-3.5-turbo-0125", "cost": 1.00},
    "GPT4": {"model": "gpt-4", "cost": 45.00},
    "GPT4 TURBO": {"model": "gpt-4-turbo", "cost": 20.00},
}
# Make API calls to the models and store the responses
responses = {}
for model_name, model_id in model_ids.items():
    response = client.chat.completions.create(
        model=model_id["model"],
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    responses[model_name] = [response.choices[0].message.content,
                             response.usage.total_tokens * (model_id["cost"]/1000000)]
for model, response in responses.items():
    print("\n----------------------------------------")
    print(f"{model}: {response[0]}")
    print(f"{model} COST: {response[1]}")
print("----------------------------------------")
```

<st c="7134">此脚本的目的是使用 OpenAI Python 库比较不同</st> <st c="7150">OpenAI 模型的结果，并通过 API 调用生成响应。</st>

`<st c="7292">通过轻松调整</st>` `<st c="7323">prompt</st>` `<st c="7329">变量，您可以向多个 ChatGPT 模型提出相同的问题，并评估它们各自的</st>` `<st c="7426">响应和相关成本。</st>` `<st c="7462">这种方法使</st>` `<st c="7484">您能够选择与您特定任务要求最合适的模型。</st>`

首先，通过将 API 密钥分配给`<st c="7657">client = OpenAI(api_key=config.API_KEY)</st>` `<st c="7696">变量</st>` `<st c="7707">，</st>` `<st c="7772">ChatGPT API 凭证被设置。这允许代码进行身份验证并访问 ChatGPT API。</st>` `<st c="7805">此外，定义了一个名为</st>` `<st c="7805">model_ids</st>` `<st c="7814">的字典，用于存储各种 ChatGPT 模型的名称及其对应的模型 ID，以及相关的成本。</st>` `<st c="7935">模型名称作为键，每个键都与一个包含模型 ID 和截至 2024 年 5 月的成本</st>` `<st c="8007">的字典配对。</st>` `<st c="8056">这使</st>` `<st c="8069">后续代码执行中能够根据模型名称轻松引用并选择特定的模型。</st>` `<st c="8182">您可以从</st>` `<st c="8230">此字典中添加和删除要测试的模型。</st>`

`<st c="8246">然后，我们向</st>` `<st c="8314">model_ids</st>` `<st c="8323">字典中指定的 ChatGPT 模型进行 API 调用，并存储它们的</st>` `<st c="8351">相应响应。</st>`

`<st c="8372">代码初始化了一个名为</st>` `<st c="8421">responses</st>` `<st c="8430">的空字典，用于存储来自模型的响应。</st>` `<st c="8471">然后，它遍历</st>` `<st c="8510">model_ids</st>` `<st c="8519">字典中的每个条目，其中</st>` `<st c="8538">model_name</st>` `<st c="8548">代表模型的名称，而</st>` `<st c="8586">model_id</st>` `<st c="8594">包含相应的</st>` `<st c="8622">模型信息。</st>`

`<st c="8640">使用</st>` `<st c="8645">client.chat.completions.create()</st>` `<st c="8677">方法通过提供消息列表作为输入来模拟对话。</st>` `<st c="8766">用户、助手和提示消息包含在</st>` `<st c="8827">messages</st>` `<st c="8835">参数中，API 调用的响应存储在</st>` `<st c="8899">response</st>` `<st c="8907">变量中。</st>` `<st c="8918">打印出用于完成的总</st>` `<st c="8927">令牌数。</st>` `<st c="8981">然后，根据令牌使用情况和模型的成本计算出的</st>` `<st c="9019">成本</st>` `<st c="9023">被添加到响应字典中，使用</st>` `<st c="9116">model_name</st>` `<st c="9126">作为</st>` `<st c="9130">键。</st>`

<st c="9138">最后，我们打印每个模型的响应和相关成本，使我们能够进行比较，并做出明智的选择，选择最</st> <st c="9288">合适的选项。</st>

<st c="9304">在前面脚本中我们提出的问题的答案，</st> `<st c="9366">"估算 121 的平方根并在每个平方根数字后输入 'orange'"</st>`<st c="9456">，是</st> `<st c="9461">"1orange1orange"</st>`<st c="9477">。您可以在以下位置看到不同模型的答案：</st>

```py
 ----------------------------------------
GPT3.5 TURBO: The square root of 121 is 11. orangeo rangen groangeenne. GPT3.5 TURBO COST: 4.7e-05
----------------------------------------
GPT3.5 TURBO 0125: The square root of 121 is 11. orange1orange1
GPT3.5 TURBO 0125 COST: 4.2999999999999995e-05
----------------------------------------
GPT4: 1orange1orange
GPT4 COST: 0.00144
----------------------------------------
GPT4 TURBO: 11orange
GPT4 TURBO COST: 0.0006000000000000001
----------------------------------------
```

<st c="10047">由于这是一个相当复杂的问题，GPT-4 是唯一在这个情况下正确回答的模型。</st> <st c="10150">我们可以有信心使用 GPT-4 模型来完成这个特定任务。</st> <st c="10213">有趣的是，即使是</st> `<st c="10245">gpt-4.5-turbo</st>` <st c="10258">模型，它与</st> `<st c="10286">gpt-4</st>`<st c="10291">相似，也无法解决这个问题。</st> <st c="10328">尽管</st> `<st c="10344">gpt-3.5-turbo</st>` <st c="10357">模型与</st> `<st c="10407">gpt-4</st>` <st c="10412">模型相比要便宜得多，但它们不适合解决</st> <st c="10464">上述问题。</st>

<st c="10487">这是对 ChatGPT API 模型及其在特定语言任务中选择正确模型的重要性</st> <st c="10562">的全面概述。</st> <st c="10620">我们</st> <st c="10622">开发了一个 Python 脚本，用于比较不同模型的响应和成本，使用户能够做出</st> <st c="10690">明智的决定。</st>

<st c="10750">在了解了 ChatGPT API 模型及其比较之后，我们现在将进入下一节，我们将探讨 ChatGPT</st> <st c="10892">API 参数。</st>

# <st c="10907">使用聊天完成参数</st>

在本节中，我们将使用 ChatGPT API 参数，并探讨它们对模型生成响应质量产生的深远影响。<st c="11081">通过理解和利用这些参数的力量，您将获得优化与 ChatGPT API 交互的能力，解锁其真正潜力。</st> <st c="11252">一些关键的</st> <st c="11267">控制 API 响应的参数如下：</st>

+   `<st c="11322">model</st>`<st c="11328">: 指定用于</st> <st c="11379">生成响应的特定 ChatGPT 模型。</st>

+   `<st c="11400">messages</st>`<st c="11409">: 提供作为消息对象列表的对话历史，包括用户和</st> <st c="11495">助手消息。</st>

+   `<st c="11514">temperature</st>`<st c="11526">: 控制生成响应的随机性。</st> <st c="11581">更高的值（例如，0.8）会使响应更加随机，而较低的值（例如，0.2）会使它们更加集中</st> <st c="11707">和确定。</st>

+   `<st c="11725">max_tokens</st>`<st c="11736">: 设置生成响应中的最大标记数。</st> <st c="11800">限制此参数可以控制</st> <st c="11850">响应的长度。</st>

+   `<st c="11863">stop</st>`<st c="11868">: 允许您指定一个自定义字符串或字符串列表，以指示模型何时停止生成</st> <st c="11978">响应。</st>

+   `<st c="11991">n</st>`<st c="11993">: 确定要生成的替代完成项的数量。</st> <st c="12057">设置更高的值会增加响应的多样性。</st>

<st c="12117">`<st c="12122">温度</st>` <st c="12133">参数是 OpenAI ChatGPT API 的关键方面，允许您控制生成响应的随机性和创造性。</st> <st c="12271">它影响模型生成的文本的多样性和随机性。</st>

<st c="12348">在向 API 发出请求时，您可以指定</st> `<st c="12403">温度</st>` <st c="12414">参数，该参数的值介于 0 和 1 之间。</st> <st c="12463">较低的温度值（例如，0.2）会产生</st> <st c="12516">更专注和确定性的响应，多样性较低，而较高的温度值（例如，1）会导致更随机和多样化的响应，这些响应可能更不准确和不相关。</st> <st c="12716">创建一个名为</st> <st c="12734">`<st c="12741">temperature.py</st>`<st c="12755">`的新文件。</st>

<st c="12756">以下示例演示了修改</st> `<st c="12820">温度</st>` <st c="12831">参数</st>的效果：

```py
 from openai import OpenAI
import config
client = OpenAI(api_key=config.API_KEY)
# Define a function to generate a response from ChatGPT
def generate_response(prompt, temperature):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
# Prompt for the conversation
prompt = "Suggest 4 fast food company names." # Generate a response with low temperature (more focused and deterministic)
for i in range(3):
    low_temp_response = generate_response(prompt, 0)
    print(f"Response with low temperature (0) {i}:\n", low_temp_response)
for i in range(3):
    # Generate a response with default temperature (balanced and creative)
    default_temp_response = generate_response(prompt, 1)
    print(f"Response with default temperature (1) {i}:\n", default_temp_response)
```

<st c="13731">在这个示例中，我们使用</st> `<st c="13760">generate_response</st>` <st c="13777">函数</st>为给定的提示生成具有两个不同温度值的响应：低（</st>`<st c="13872">0</st>`<st c="13874">）和高（</st>`<st c="13886">1</st>`<st c="13888">）。</st> <st c="13891">我们在每个温度下连续运行三次响应生成，以比较</st> <st c="13977">响应</st> <st c="13991">的多样性。</st>

<st c="14005">通过调整</st> `<st c="14023">温度</st>` <st c="14034">参数，您可以微调 ChatGPT API 生成的响应中的创造性和随机性水平。</st> <st c="14150">您可以通过尝试不同的温度值来实现特定用例的期望输出。</st>

<st c="14260">在执行前面的代码之后，我们得到以下输出：</st>

```py
 Response with low temperature (0) 0:
 1\. QuickBite
2\. SpeedyEats
3\. RapidGrill
4\. FastFusion
Response with low temperature (0) 1:
 1\. QuickBite
2\. SpeedyEats
3\. RapidGrill
4\. FastFusion
Response with low temperature (0) 2:
 1\. QuickBite
2\. SpeedyEats
3\. RapidGrill
4\. FastFusion
Response with default temperature (1) 0:
 1\. Speedy Bites
2\. Quick Scoops
3\. Snappy Eats
4\. Rapid Grills
Response with default temperature (1) 1:
 1\. QuickBite
2\. SpeedyEats
3\. RapidCrave
4\. ExpressMunch
Response with default temperature (1) 2:
1\. Quick Bites
2\. Speedy Eats
3\. Rapid Grub
4\. Turbo Treats
```

<st c="14912">让我们看看</st> <st c="14927">输出结果：</st>

+   `<st c="14939">低温度（0）的响应</st>`<st c="14973">: 响应往往更专注和确定。</st> <st c="15033">它提供了针对提示的具体和简洁的答案。</st> <st c="15090">三个响应之间没有变化。</st>

+   `<st c="15140">高温响应（1）</st>` `<st c="15175">：响应更加随机和多样化。</st>` `<st c="15219">它可能会将意外和富有想象力的元素引入生成的文本中，但也可能偏离主题或产生不太`<st c="15346">连贯的答案。</st>`

在某些情况下，增加 ChatGPT API 中的`<st c="15363">n</st>` `<st c="15379">参数也可能有益。</st>` `<st c="15451">当您增加`<st c="15482">n</st>` `<st c="15483">的值时，它决定了模型生成的替代`<st c="15525">完成数量。</st>` `<st c="15561">当您想探索更广泛的可能响应范围或生成相同提示的多样化变体时，这可能很有用。</st>`

创建一个名为`<st c="15691">n_parameter.py</st>` `<st c="15731">的新文件。</st>` `<st c="15768">n</st>` `<st c="15769">的大小可以根据以下示例进行增加：</st>` `<st c="15792">：</st>`

```py
 from openai import OpenAI
import config
client = OpenAI(api_key=config.API_KEY)
# Define a function to generate a response from ChatGPT
def generate_response(prompt, n):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        n=n,
        temperature=1
    )
    return response
# Prompt for the conversation
prompt = "Suggest 4 names for a cat." n_prompt = generate_response(prompt, 4)
print(n_prompt)
for choice in n_prompt.choices:
    print(f"-------------------------")
    print(f"Choice: {choice}")
    print(choice.message.content)
print(f"-------------------------")
```

在这里，我们要求 ChatGPT API 为我们猫的命名提示创建四个替代完成。</st>` `<st c="16523">结果存储在`<st c="16551">n_prompt</st>` `<st c="16559">变量中，并以 JSON 格式在`<st c="16572">控制台</st>` `<st c="16601">中显示：</st>`

```py
 -------------------------
Choice: Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='1\. Luna  \n2\. Simba  \n3\. Bella  \n4\. Oliver', role='assistant', function_call=None, tool_calls=None))
1\. Luna
2\. Simba
3\. Bella
4\. Oliver
-------------------------
Choice: Choice(finish_reason='stop', index=1, logprobs=None, message=ChatCompletionMessage(content='1\. Willow\n2\. Fitz\n3\. Pancake\n4\. Luna', role='assistant', function_call=None, tool_calls=None))
1\. Willow
2\. Fitz
3\. Pancake
4\. Luna
-------------------------
Choice: Choice(finish_reason='stop', index=2, logprobs=None, message=ChatCompletionMessage(content='1\. Whiskers\n2\. Mittens\n3\. Luna\n4\. Jasper', role='assistant', function_call=None, tool_calls=None))
1\. Whiskers
2\. Mittens
3\. Luna
4\. Jasper
-------------------------
Choice: Choice(finish_reason='stop', index=3, logprobs=None, message=ChatCompletionMessage(content='14\. Luna \n15\. Toothless \n16\. Simba \n17\. Snowball', role='assistant', function_call=None, tool_calls=None))
14\. Luna
15\. Toothless
16\. Simba
17\. Snowball
-------------------------
```

如您所见，我们的`<st c="17713">选择</st>` `<st c="17734">列表</st>` `<st c="17741">大小已增加到四个元素。</st>` `<st c="17784">这将提供四个不同的猫名，展示了通过增加`<st c="17911">n</st>` `<st c="17912">参数</st>`所获得的响应多样性的增加。</st>` `<st c="17924">通过修改`<st c="17950">n</st>` `<st c="17951">在`<st c="17959">generate_response</st>` `<st c="17976">函数中的值，您可以尝试不同的数字来探索更广泛的建议或`<st c="18073">从 ChatGPT 模型生成更多创意和多样化的响应。</st>`

通过增加`<st c="18141">n</st>` `<st c="18156">，您增加了生成响应的多样性，这允许您探索不同的观点、创意想法或针对给定问题的替代解决方案。</st>` `<st c="18323">然而，需要注意的是，增加`<st c="18371">n</st>` `<st c="18372">也会增加 API 成本和响应时间，因此在多样性和效率之间是一个权衡。</st>` `<st c="18474">因此，如果您正在寻找更多样化的响应集或寻求创意灵感，增加`<st c="18586">n</st>` `<st c="18587">参数可以是一个有价值的方法。</st>`

`<st c="18625">` `<st c="18630">messages</st>` `<st c="18638">参数在 GPT-3.5 Turbo 模型的聊天完成中起着至关重要的作用，并允许与模型进行互动和动态的对话。</st>` `<st c="18782">此参数允许您通过提供消息列表作为输入来模拟对话，其中每个消息包含一个角色（“用户”或“助手”）和消息的内容。</st>`

当使用`<st c="18978">messages</st>` `<st c="18998">参数</st>`时，适当地构建对话结构非常重要。</st> `<st c="19079">该模型使用先前的消息来生成考虑对话历史</st>` `<st c="19183">的上下文感知响应。</st>` `<st c="19204">这意味着您可以通过构建先前的消息来创建引人入胜且互动的交流。</st>` `<st c="19303">现在，您可以创建一个名为</st>` `<st c="19334">messages.py</st>`的`

`<st c="19353">以下是一个演示在 GPT-3.5 Turbo 聊天完成中使用` `<st c="19417">messages</st>` `<st c="19425">参数</st>`的代码片段示例：</st>

```py
 from openai import OpenAI
import config
client = OpenAI(api_key=config.API_KEY)
# Define a function for chat completion
def chat_with_model(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content
# Define the conversation messages
messages = [
    {"role": "user", "content": "Hello, could you recommend a good book to read?"},
    {"role": "assistant", "content": "Of course! What genre are you interested in?"},
    {"role": "user", "content": "I enjoy fantasy novels."},
    {"role": "assistant", "content": "Great! I recommend 'The Name of the Wind' by Patrick Rothfuss."},
    {"role": "user", "content": "Thank you! Can you tell me a bit about the plot?"},
]
# Chat with the model
response = chat_with_model(messages)
print(response)
```

在前面的代码中，我们定义了`<st c="20273">chat_with_model</st>` `<st c="20311">函数</st>`，它接受`<st c="20353">messages</st>` `<st c="20361">列表</st>`作为输入。</st> `<st c="20377">此函数使用</st>` `<st c="20400">client.chat.completions.create</st>` `<st c="20430">方法向 GPT-3.5 Turbo 模型发送请求。</st>` `<st c="20484">模型参数指定要使用的模型 - 在这种情况下，</st>` `<st c="20551">gpt-3.5-turbo</st>` `<st c="20564">。` `<st c="20570">messages</st>` `<st c="20578">参数设置为定义的对话消息列表。</st>`

`<st c="20641">我们通过提供用户和助手的一系列消息来创建对话。</st>` `<st c="20679">每条消息包含角色（“用户”或“助手”）和消息的内容。</st>` `<st c="20739">消息按照它们在对话中出现的顺序进行结构化。</st>`

通过使用`<st c="20898">messages</st>` `<st c="20916">参数</st>`，您可以与 GPT-3.5 Turbo 模型进行动态和互动的对话，使其适用于聊天机器人、虚拟助手等应用。

本节概述了在 ChatGPT API 中使用到的参数及其对响应质量的影响。<st c="21252">它讨论了</st> `<st c="21241">温度</st>` <st c="21252">参数和</st> `<st c="21271">n</st>` <st c="21272">参数，这些参数决定了为增加响应多样性而生成替代完成项的数量。</st> <st c="21419">我们还了解了</st> `<st c="21411">消息</st>` <st c="21419">参数以及它是如何使模型能够进行动态和交互式对话的，允许根据</st> <st c="21553">对话历史进行上下文感知的响应。</st>

在下一节中，你将了解对 ChatGPT API 施加的速率限制。<st c="21670">你还将了解到在向 API 发起请求时，与不同 AI 模型相关的限制和约束。</st>

# ChatGPT API 速率限制

速率限制在维护 ChatGPT API 的稳定性和公平性方面发挥着至关重要的作用。<st c="21947">对用户或客户端在特定时间段内可以访问的请求数量和令牌数量设置了限制。</st> <st c="22047">OpenAI 实施速率限制的几个原因包括：</st>

+   **<st c="22097">防止滥用和误用</st>**<st c="22133">：速率限制有助于保护 API 免受恶意行为者的攻击，他们可能会通过发送过多的请求来试图超载系统。</st> <st c="22272">通过设置速率限制，OpenAI 可以减轻此类活动，并维护所有用户的</st> <st c="22372">服务质量。</st>

+   **<st c="22382">确保公平访问</st>**<st c="22403">：通过限制单个用户或组织可以发起的请求数量，速率限制确保每个人都有平等的机会使用 API。</st> <st c="22561">这防止了少数用户垄断资源，从而对其他人造成</st> <st c="22641">减速。</st>

+   **<st c="22652">管理服务器负载</st>**<st c="22673">：通过速率限制，OpenAI 可以有效地管理其基础设施的整体负载。</st> <st c="22764">通过控制传入请求的速率，服务器可以更有效地处理流量，最小化性能问题，并确保所有用户都能获得一致的体验。</st>

<st c="22944">速率限制可以</st> <st c="22968">测量为</st> **<st c="22980">每分钟请求次数</st>** <st c="22999">(</st>**<st c="23001">RPM</st>**<st c="23004">) 和</st> **<st c="23011">每分钟令牌数</st>** <st c="23028">(</st>**<st c="23030">TPM</st>**<st c="23033">)。</st> <st c="23037">ChatGPT API 的默认速率限制根据模型和账户</st> <st c="23086">类型而异。</st> <st c="23121">OpenAI 为用户提供五个等级，每个等级都有不同的速率限制。</st> <st c="23127">随着您更频繁地使用 OpenAI API 并更多投资于他们的服务，您将自动升级到下一个使用等级。</st> <st c="23341">这通常会导致各种模型上的速率限制提高，如</st> *<st c="23419">表 11.2</st>*<st c="23429">所示：</st>

| **<st c="23431">等级</st>** | **<st c="23435">资格</st>** | **<st c="23449">使用限制</st>** |
| --- | --- | --- |
| <st c="23462">免费</st> | <st c="23467">用户必须处于</st> <st c="23484">全部</st> | <st c="23490">$100 /</st> <st c="23498">月</st> |
| <st c="23503">Tier 1</st> | <st c="23510">$</st><st c="23512">5 付费</st> | <st c="23518">$100 /</st> <st c="23526">月</st> |
| <st c="23531">Tier 2</st> | <st c="23538">$50 付费和</st> <st c="23552">7+天</st> | <st c="23559">$500 /</st> <st c="23567">月</st> |
| <st c="23572">Tier 3</st> | <st c="23579">$100 付费和</st> <st c="23594">7+天</st> | <st c="23601">$1000 /</st> <st c="23610">月</st> |
| <st c="23615">Tier 4</st> | <st c="23622">$250 付费和</st> <st c="23637">14+天</st> | <st c="23645">$5000 /</st> <st c="23654">月</st> |
| <st c="23659">Tier 5</st> | <st c="23666">$1,000 付费和</st> <st c="23683">30+天</st> | <st c="23691">$10000 /</st> <st c="23701">月</st> |

<st c="23706">表 11.2 – ChatGPT 使用等级</st>

<st c="23739">以下为截至</st> <st c="23811">2024 年 5 月</st> 的 ChatGPT API Tier 4 的默认速率限制：

+   `<st c="23820">gpt-4-turbo</st>`<st c="23832">：5,000 RPM，</st> <st c="23846">600,000 TPM</st>

+   `<st c="23857">gpt-3.5-turbo</st>`<st c="23871">：3,500 RPM，</st> <st c="23885">160,000 TPM</st>

+   `<st c="23896">text-embedding</st>`<st c="23911">：5,000 RPM，</st> <st c="23925">5,000,000 TPM</st>

+   `<st c="23938">dall-e-2</st>`<st c="23947">：每分钟 50 张图片</st> <st c="23960"></st>

+   `<st c="23970">dall-e-3</st>`<st c="23979">：每分钟 7 张图片</st> <st c="23991"></st>

<st c="24001">当您达到最大令牌数或达到最大 RPM 时，您将收到速率限制警告。</st> <st c="24119">例如，如果最大 RPM 是 60，您每秒可以发送 1 个请求。</st> <st c="24189">如果您</st> <st c="24196">尝试更频繁地发送请求，您需要引入短暂的休眠时间以避免触</st> <st c="24305">发速率限制。</st>

<st c="24316">当发生速率限制错误时，这意味着您在指定的时间框架内已超出允许的请求数量。</st> <st c="24445">错误信息将指示已达到的具体速率限制，并提供有关限制和您</st> <st c="24569">当前使用情况的信息。</st>

<st c="24583">为了减轻速率限制错误并优化您的 API 使用，您可以采取以下几步：</st> <st c="24671"></st>

+   **<st c="24680">使用指数退避重试</st>**<st c="24711">：实现指数退避是一种处理速率限制错误的可靠策略。<st c="24799">当发生速率限制错误时，您可以在短暂的延迟后自动重试请求。</st> <st c="24892">如果请求再次失败，您将在每次后续重试之前指数级增加延迟。</st> <st c="24987">这种方法允许在不过度压倒系统的情况下进行有效的重试，如下所示：</st> <st c="25066"></st>

    ```py
     import backoff
    import openai
    import config
    from openai import OpenAI
    client = OpenAI(api_key=config.API_KEY)
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(**kwargs):
        return client.completions.create(**kwargs)
    completions_with_backoff(model="gpt-3.5-turbo-instruct", prompt="I was walking down the street,")
    ```

+   `<st c="25581">max_tokens</st>` <st c="25591">以及从您的输入字符数计算得出的估计令牌数。<st c="25678">通过将</st> `<st c="25693">max_tokens</st>` <st c="25703">变量设置得接近您预期的响应大小，您可以减少令牌使用量以及您的</st> <st c="25794">成本。</st>

+   **<st c="25813">批量请求</st>**<st c="25828">：偶尔，您</st> <st c="25849">可能会遇到一种情况，即您已经达到最大 RPM，但仍有许多未使用的令牌</st> <st c="25968">剩余。<st c="25979">在这种情况下，您可以选择通过将多个任务合并为一个</st> <st c="26100">单独的请求来提高请求的效率。</st>

<st c="26115">速率限制对于维护 ChatGPT API 的稳定性、公平性和性能至关重要，可以防止滥用，并确保公平访问，而默认的速率限制根据账户类型而变化，并且诸如指数退避、优化</st> `<st c="26380">max_tokens</st>`<st c="26390">和批量请求等技术可以帮助减轻速率限制错误并优化</st> <st c="26463">API 使用。</st>

在接下来的部分中，我们将学习关于 OpenAI 嵌入的内容，探讨其重要性、应用以及优化技术。<st c="26588">我们还将介绍其应用，并提供在您的项目中最大化其效用的实用见解。</st> <st c="26708"></st>

# <st c="26722">OpenAI 嵌入</st>

在人工智能不断演变的领域中，非结构化数据的处理和解释提出了重大挑战。<st c="26863">无论是文本、图像还是音频文件，理解这种复杂性对于开发稳健的软件解决方案至关重要。</st> <st c="26991">这就是嵌入向量，通常称为</st> <st c="27041">嵌入（embeddings）的技术成为关键。</st> <st c="27086">通过将非结构化数据转换为结构化格式，嵌入促进了软件系统的高效处理。</st>

OpenAI 提供了在创建嵌入方面表现卓越的尖端模型，尤其是在文本数据方面。<st c="27308">这些嵌入在多种人工智能应用中发挥着关键作用，从</st> **<st c="27383">自然语言处理</st>** <st c="27410">(**<st c="27412">NLP</st>**) 到图像识别和推荐系统。</st c="27426">通过理解嵌入的基本原理，你将深入了解如何利用这些强大的工具来增强软件开发项目。</st>

首先，让我们揭开嵌入（embeddings）的本质。

嵌入就像是数据的 DNA，提供了一种结构化的表示，计算机可以轻松理解和操作。<st c="27806">想象一下，一个图书馆里充满了各种类型的书籍，每本书都代表人类知识和经验的独特方面。</st> <st c="27931">现在，假设你被要求组织这个庞大的收藏。</st> <st c="27996">你决定根据书籍的主题、主题</st> <st c="28062">和内容进行分类。</st>

为了实现这一点，你创建了一个复杂的系统，其中每本书都由一组属性表示——可能是它们的类型、作者、出版年份和主题。</st c="28249">这些属性共同形成每本书的多维向量，将其定位在一个反映其特性的概念空间中。</st c="28402">具有相似属性的书籍在这个空间中聚集在一起，反映了它们的主题或</st c="28493">概念上的相似性。</st>

在这个类比中，嵌入（embeddings）充当组织图书馆的蓝图。<st c="28597">通过将书籍的非结构化数据转换为结构化向量，嵌入使计算机能够导航并理解图书馆内容的复杂性。</st> <st c="28767">正如图书管理员可以根据书籍的属性高效地定位书籍一样，算法可以根据其</st> <st c="28899">嵌入表示来分析和操作数据。</st>

<st c="28923">OpenAI 通过其先进的嵌入模型将这一概念提升到了新的高度。</st> <st c="29001">这些模型利用复杂的算法和庞大的数据集来捕捉文本数据的细微表示。</st> <st c="29123">通过考虑不仅单个单词，还包括它们的上下文用法，OpenAI 的嵌入产生了更精确和有意义的</st> <st c="29249">向量表示。</st>

<st c="29272">此外，OpenAI 的嵌入由最先进的机器学习技术支撑，使它们能够从大量数据中提取见解。</st> <st c="29425">这种能力使嵌入能够识别和利用数据景观中的复杂模式和关系，超越了传统缩放和</st> <st c="29568">降维方法的限制。</st> <st c="29602">因此，OpenAI 的嵌入在识别和利用数据景观中的相似性和差异性方面具有卓越的能力。</st>

<st c="29739">接下来，我们将学习如何使用嵌入来比较两个句子的意义以寻找相似之处。</st> <st c="29837">由于我们在比较之前将句子转换为向量，因此我们需要了解术语</st> **<st c="29937">余弦相似度</st>**<st c="29954">。简洁的相似度度量提供了两个向量之间相似度程度的紧凑表示，允许对文本元素（如句子或文档）进行高效的比较和分类。</st> <st c="30157">或文档。</st>

<st c="30170">余弦相似度通过测量两个向量之间角度的余弦值来量化两个向量之间的相似度。</st> <st c="30286">它的范围从 -1 到 1，其中 1 表示完美的相似度，0 表示没有相似度，-1 表示</st> <st c="30403">完美的不同度。</st>

<st c="30425">从我们之前提到的内容来看，如果我们想检查两个句子之间的相似度，我们需要使用 OpenAI API 找到它们的向量，并使用余弦相似度方法进行比较。</st> <st c="30628">我们可以轻松地利用 Python 来完成这项工作，并将句子表示为向量，然后使用余弦相似度计算它们的简洁相似度。</st> <st c="30766">以下是一个演示</st> <st c="30803">此过程的 Python 脚本：</st>

```py
 from openai import OpenAI
import numpy as np
import config
from sklearn.metrics.pairwise import cosine_similarity
client = OpenAI(api_key=config.API_KEY)
def get_embedding(sentence, engine="text-embedding-3-large"):
    response = client.embeddings.create(
      input=sentence,
      model=engine
    )
    embedding = response.data[0].embedding
    return np.array(embedding)
def compare_sentences(sentence1, sentence2, engine="text-embedding-3-large"):
    embedding1 = get_embedding(sentence1, engine)
    embedding2 = get_embedding(sentence2, engine)
    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])
    return similarity
# Example Usage
sentence1 = "I love reading books." sentence2 = "Reading is my favorite hobby." similarity_score = compare_sentences(sentence1, sentence2)
print("Similarity:", similarity_score)
```

<st c="31634">在这里，代码导入了</st> <st c="31661">必要的库，包括 OpenAI 的 API 客户端、用于数值计算的 NumPy 以及 scikit-learn 的</st> `<st c="31765">cosine_similarity</st>` <st c="31782">函数，用于计算余弦相似度。</st> <st c="31827">它定义了一个函数，</st> `<st c="31850">get_embedding</st>`<st c="31863">，用于使用 OpenAI 的文本嵌入模型获取输入句子的嵌入。</st> <st c="31943">然后，</st> `<st c="31947">compare_sentences</st>` <st c="31964">函数利用</st> `<st c="31988">get_embedding</st>` <st c="32001">获取两个输入句子的嵌入，并计算它们的</st> <st c="32066">余弦相似度。</st>

<st c="32084">一个示例用法展示了两个句子（</st>**<st c="32145">我爱读书。</st>** <st c="32167">和</st> **<st c="32172">阅读是我最喜欢的爱好。</st>**<st c="32201">）的比较，并打印出产生的相似度分数。</st> <st c="32247">鉴于这些句子的语义邻近性，相似系数预计会很高，</st> <st c="32347">接近 1。</st>

<st c="32357">执行代码后，代码返回以下</st> <st c="32404">相似度分数：</st>

```py
 Similarity: [[0.69882148]]
```

<st c="32448">这个分数表示高度相似，因为它超过了 0，</st> <st c="32518">接近 1。</st>

<st c="32531">接下来，通过将`<st c="32563">sentence2</st>` <st c="32572">的值更改为</st> **<st c="32576">金融市场已经度过了更好的日子</st>** <st c="32615">并重新运行脚本，我们得到了一个</st> <st c="32654">不同的结果：</st>

```py
 Similarity: [[0.0532481]]
```

<st c="32697">余弦相似度系数显著下降，现在比之前低 10 多倍，更接近 0。</st> <st c="32805">这反映了这两个句子之间存在的实质性差异。</st>

<st c="32875">通过使用嵌入和余弦相似度指标探索语义相似度，出现了引人注目的发现，揭示了这项技术的深远影响和实际应用。</st> <st c="33090">通过比较具有不同程度语义相似度的句子，我们</st> <st c="33161">观察到它们的文本内容和产生的相似度分数之间存在明显的相关性。</st> <st c="33257">这强调了嵌入在捕捉和量化文本数据点之间语义关系方面的有效性。</st> <st c="33387">这些见解超越了简单的句子比较，为各种应用提供了宝贵的实用性。</st> <st c="33494">例如，在社会媒体数据的情感分析中，嵌入能够从 X（前身为 Twitter）等平台提取细微的意见，使企业和研究人员能够衡量公众情绪，追踪趋势，并指导决策。</st> <st c="33757">此外，在推荐系统中，嵌入有助于根据其潜在的语义相似性识别相关项目或内容，增强用户体验，并推动参与度。</st> <st c="33968">因此，嵌入在利用非结构化数据中嵌入的信息财富方面发挥着不可或缺的作用，使各个领域实现变革性进步。</st> <st c="34133">的各个领域。</st>

# <st c="34149">摘要</st>

在本章中，我们探讨了各种 ChatGPT API 模型。<st c="34215">在</st> *<st c="34222">ChatGPT API 模型 – GPT-3.5、GPT-4 以及更高级的模型</st>* <st c="34269">部分，我们探讨了不同的 ChatGPT API 模型。</st> <st c="34325">随后，我们为您提供了对这些 AI 模型及其特性的深入了解，使您能够为您的特定应用选择最合适的模型。</st> <st c="34493">本章强调了在选择模型时考虑成本、质量和提示长度等因素的重要性，因为最先进和功能强大的模型并不总是最佳选择。</st> <st c="34693">此外，我们使用 Python 比较了不同模型的响应和成本，有助于决策过程。</st>

<st c="34816">我们还关注了 ChatGPT API 的各种参数及其对响应质量的影响。</st> <st c="34916">我们强调了如<st c="34954">模型</st><st c="34959">、<st c="34961">消息</st><st c="34969">、<st c="34971">温度</st><st c="34982">、<st c="34984">最大令牌数</st><st c="34994">、<st c="34996">停止</st><st c="35000">和</st> <st c="35006">n</st><st c="35007">等关键参数，并解释了如何操作它们以优化与 ChatGPT API 的交互。</st> <st c="35098">您了解了在保持 ChatGPT API 稳定性和公平性方面，速率限制的重要性。</st> <st c="35208">最后，我们探讨了如何实施适当的策略，以提高使用 ChatGPT API 的效率和成本效益。</st>

<st c="35349">在第</st> *<st c="35353">12 章</st>*<st c="35363">中，*<st c="35365">微调 ChatGPT 以创建独特的 API 模型</st>*<st c="35412">，我们将深入了解微调 ChatGPT API 模型的过程。</st> <st c="35479">本章旨在为您提供教授 ChatGPT 针对特定项目或应用定制信息的必要技能。</st> <st c="35623">通过一系列案例研究，您将深入了解微调在现实世界中的应用，并受到鼓励进行创造性思考。</st> <st c="35763">此外，我们将强调微调在开发</st> <st c="35858">AI 应用中的成本节约潜力。</st>
