

# 第四章：使用规划器进行编程

在上一章中，我们学习了如何手动链式调用函数以执行复杂任务。在本章中，我们将学习如何使用 **规划器** 自动链式调用函数。自动链式调用函数可以为用户提供更多灵活性，让他们以你不需要编写代码的方式使用你的应用程序。

在本章中，我们将学习规划器是如何工作的，何时使用它们，以及需要注意什么。我们还将学习如何编写函数并构建一个内核，帮助规划器构建良好的计划。

本章将涵盖以下主题：

+   规划器是什么以及何时使用它

+   创建和使用规划器来运行简单函数

+   设计函数以帮助规划器决定最佳的组合方式

+   使用规划器允许用户以复杂的方式组合函数，而无需编写代码

到本章结束时，你将学会如何通过赋予用户使用自然语言进行请求的能力，让他们解决那些你不需要编写代码的复杂问题。

# 技术要求

要完成本章，你需要拥有你首选的 Python 或 C# 开发环境的最新、受支持的版本：

+   对于 Python，最低支持的版本是 Python 3.10，推荐版本是 Python 3.11

+   对于 C#，最低支持的版本是 .NET 8

在本章中，我们将调用 OpenAI 服务。鉴于公司在训练这些大型语言模型（LLM）上花费的金额，使用这些服务不是免费的。你需要一个 **OpenAI API** 密钥，无论是直接通过 **OpenAI** 还是 **Microsoft**，通过 **Azure** **OpenAI** 服务。

如果你正在使用 .NET，本章的代码位于 [`github.com/PacktPublishing/Building-AI-Applications-with-Microsoft-Semantic-Kernel/tree/main/dotnet/ch5`](https://github.com/PacktPublishing/Building-AI-Applications-with-Microsoft-Semantic-Kernel/tree/main/dotnet/ch5)。

如果你正在使用 Python，本章的代码位于 [`github.com/PacktPublishing/Building-AI-Applications-with-Microsoft-Semantic-Kernel/tree/main/python/ch5`](https://github.com/PacktPublishing/Building-AI-Applications-with-Microsoft-Semantic-Kernel/tree/main/python/ch5)。

你可以通过访问 GitHub 仓库并使用以下命令安装所需的包：`pip install -r requirements.txt`。

# 什么是规划器？

到目前为止，我们通过自己执行函数调用来执行复杂请求。然而，这要求你限制用户可以提出的请求类型，使其符合你能够预测并在事先编写的范围内。这也限制了用户一次只能生成一个输出。有时，你可能希望赋予他们做更多事情的能力。

例如，如果您有一个语义功能允许用户请求笑话（正如我们在*第一章*中构建的）并且用户请求“*讲一个敲门笑话*”，您可以简单地调用讲敲门笑话的语义功能。但如果用户请求三个敲门笑话，该功能将不知道如何处理。

规划器是语义内核的一个内置功能，它接收用户请求，然后遍历您在内核中加载的所有功能的描述、参数和输出，并决定最佳组合方式，生成一个**计划**。

在撰写本文时，有两个规划器——一个**Handlebars 规划器**和一个**函数调用逐步规划器**，我们将简称为逐步规划器。它们的使用方式相同，但内部工作方式不同。当语义内核使用 Handlebars 规划器时，它会请求 AI 服务（例如，GPT-3.5 或 GPT-4）编写代码，以脚本语言 Handlebars 调用您加载到内核中的函数。Handlebars 规划器非常新，仍在实验阶段。预计它比逐步规划器消耗更少的令牌，因为编程语言在表达复杂概念（如条件语句和循环）方面可能更有效率。逐步规划器生成的计划是一个与聊天服务的对话，可能比 Handlebars 规划器生成的计划更长，并消耗更多令牌。目前，Handlebars 规划器的一个主要限制是它仅适用于 C#，尽管预计 2024 年可能会发布 Python 版本。

为了更好地理解规划器的工作原理，假设您有一个生成故事的插件、一个将故事分解成小部分的插件和一个生成图像的插件。您将这些插件全部加载到内核中。用户提交一个请求：

“*编写一个关于数据科学家和他的忠实犬伴解决犯罪的两页故事，将其分解成小部分，并为每一部分生成弗兰克·米勒风格的图像*。”

规划器将遍历您在内核中加载的功能，并确定调用它们的最佳顺序，自动生成故事板，而无需您编写任何额外的代码，除了初始插件之外。

规划器可以使您的用户在您的最小努力下执行复杂任务。让我们看看何时使用它们。

# 何时使用规划器

规划器以两种方式帮助您作为开发者：

+   用户可以以您未曾想到的方式组合您应用程序的功能。如果您将应用程序的功能作为原子功能嵌入插件中，并赋予用户向规划器发送请求的能力，那么规划器可以在不要求您编写任何代码的情况下，将这些原子功能组合到工作流程中。

+   随着人工智能模型的改进，规划器会变得更好，而无需您编写任何额外的代码。当语义内核最初设计时，最好的 AI 模型是 GPT-3.5 Turbo。从那时起，我们已经发布了 GPT-4 和 GPT-4 Turbo，它们都具有更多的功能。使用语义内核构建的应用程序现在可以使用 GPT-4 Turbo，只需进行一些小的配置更改。

然而，在使用规划器时，有一些考虑因素：

+   **性能**：规划器需要读取您内核中的所有函数，并将它们与用户请求结合起来。您的内核越丰富，您可以提供给用户的函数功能就越多，但规划器遍历所有描述并组合它们所需的时间会更长。此外，像 GPT-4 这样的新模型可以生成更好的计划，但它们运行速度较慢，未来的模型可能会更快。您需要在提供给用户的函数数量和您使用的模型之间找到一个良好的平衡。在测试您的应用程序时，如果您发现规划器延迟明显，您还需要将 UI 提示集成到您的应用程序中，以便用户知道正在发生某些事情。

+   **成本**：生成一个计划可能会消耗许多令牌。如果您有很多函数，并且用户请求很复杂，语义内核将需要向 AI 服务提交一个非常长的提示，其中包含您内核中可用的函数的描述、它们的输入和输出，以及用户请求。生成的计划也可能很长，AI 服务将向您收取提交的提示和输出的费用。避免这种情况的一种方法是通过监控用户频繁创建的请求并保存这些计划，这样就不必每次都重新生成。请注意，然而，如果您保存了计划，并且后端模型（例如，GPT-5 发布）进行了升级，您必须记得重新生成这些计划以利用新模型的功能。

+   **测试**：使用规划器会使测试您的应用程序变得更加困难。例如，您的内核可能有如此多的函数，用户请求可能如此复杂，以至于规划器会超出您所使用的模型的上下文窗口。您需要做一些事情来处理这种运行时错误，例如限制用户请求的大小或您内核中可用的函数数量。此外，虽然规划器大多数时候都能正常工作，但偶尔规划器可能会生成错误的计划，例如产生幻觉功能的计划。您需要为此提供错误处理。有趣的是，在实践中，简单地重新提交失败的计划，告诉 AI 服务该计划不起作用，并询问“*你能修复它吗？*”通常有效。

考虑到所有这些，让我们看看如何使用规划器。第一步是实例化一个规划器。

# 实例化一个规划器

实例化和使用规划器很简单。在 C#中，我们将使用 Handlebars 规划器，而在 Python 中，我们将使用 Stepwise 规划器。

C#

C# 包含了新的 `HandlebarsPlanner`，它允许你创建包含循环的计划，使它们更短。在使用 C#中的 Handlebars 规划器之前，你需要使用以下命令安装它：

```py
dotnet add package Microsoft.SemanticKernel.Planners.Handlebars –-prerelease
```

要配置你的 Handlebars 规划器，你还需要安装 OpenAI 规划器连接器，如下所示：

```py
dotnet add package Microsoft.SemanticKernel.Planners.OpenAI --prerelease
```

注意，规划器是实验性的，除非你通过在你的代码中添加`pragma`指令来告知 C#你同意使用实验性代码，否则 C#会给你一个错误：

```py
#pragma warning disable SKEXP0060
```

要创建规划器，我们执行以下代码：

```py
var plannerOptions = new HandlebarsPlannerOptions()
    {
        ExecutionSettings = new OpenAIPromptExecutionSettings()
        {
            Temperature = 0.0,
            TopP = 0.1,
            MaxTokens = 4000
        },
        AllowLoops = true
    };
var planner = new HandlebarsPlanner(plannerOptions);
```

微软建议为你的规划器使用较低的 `Temperature` 和 `TopP`，以最大限度地减少规划器创建不存在函数的可能性。规划器可能会消耗大量标记；因此，我们通常将 `MaxTokens` 设置为一个较高的值，以避免运行时错误。

现在，让我们看看如何在 Python 中创建规划器。

Python

在 Python 中，Handlebars 规划器尚未提供，因此我们需要实例化 Stepwise 规划器。Stepwise 规划器创建的计划通常比 Handlebars 计划长。要将 Stepwise 规划器添加到你的 Python 项目中，你需要从 `semantic_kernel.planners` 包中导入 `FunctionCallingStepwisePlanner` 和 `FunctionCallingStepwisePlannerOptions` 类：

```py
from semantic_kernel.planners import FunctionCallingStepwisePlanner, FunctionCallingStepwisePlannerOptions
import semantic_kernel as sk
```

通常给规划器提供足够的标记是个好主意。以下是一个创建规划器的示例命令，假设你在你的语义内核中加载了一个服务，并将 `service_id` 设置为 `gpt4`：

```py
planner_options = FunctionCallingStepwisePlannerOptions(
        max_tokens=4000,
    )
planner = FunctionCallingStepwisePlanner(service_id="gpt4", options=planner_options)
```

现在，让我们为用户请求创建并运行一个计划。

# 创建和运行计划

现在我们有了规划器，我们可以用它来为用户的请求创建一个计划，然后调用该计划以获得结果。在两种语言中，我们使用两个步骤，一个用于创建计划，另一个用于执行它。

对于接下来的两个代码片段，假设你已经将用户的请求加载到了 `ask` 字符串中。让我们看看如何调用规划器：

C#

```py
var plan = await planner.CreatePlanAsync(kernel, ask);
var result = await plan.InvokeAsync(kernel);
Console.Write ($"Results: {result}");
```

Python

```py
result = await planner.invoke(kernel, ask)
print(result.final_answer)
```

你可能还记得，从 *第一章* 中，在 Python 中，结果变量包含创建计划的所有步骤，因此为了查看计划的结果，你需要打印 `result.final_answer`。如果你打印 `result` 变量，你会得到一个大的 JSON 对象。

## 规划器如何帮助的一个例子

让我们看看一个简单的例子，它已经展示了规划器如何帮助。假设你创建了一个帮助有抱负的喜剧演员创作笑话的应用程序。你创建并连接到我们在 *第一章* 中创建的 `jokes` 语义插件。该插件包含一个创建敲门笑话的语义函数。

您可以创建一个 UI，允许用户输入一个主题（例如，“*狗*”），并调用该函数来创建敲门笑话。如果用户想要创建 100 个笑话，他们需要使用该 UI 100 次。您可以通过创建另一个 UI 来解决这个问题，该 UI 会询问用户想要创建多少个笑话。然而，如果用户想要为多个主题创建多个笑话，那么他们必须为每个想要创建笑话的主题使用您的两个 UI。

相反，仅使用语义函数和计划者，您可以允许用户用自然语言描述他们想要的内容，如下所示：

“*编四个敲门笑话 - 两个关于狗，一个关于猫，一个关于鸭子。”

完整的代码如下：

C#

```py
#pragma warning disable SKEXP0060
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Planning.Handlebars;
using Microsoft.SemanticKernel.Connectors.OpenAI;
var (apiKey, orgId) = Settings.LoadFromFile();
var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("gpt-4", apiKey, orgId);
var kernel = builder.Build();
var pluginsDirectory = Path.Combine(System.IO.Directory.GetCurrentDirectory(),
        "..", "..", "..", "plugins", "jokes");
kernel.ImportPluginFromPromptDirectory(pluginsDirectory);
var plannerOptions = new HandlebarsPlannerOptions()
    {
        ExecutionSettings = new OpenAIPromptExecutionSettings()
        {
            Temperature = 0.0,
            TopP = 0.1,
            MaxTokens = 4000
        },
        AllowLoops = true
    };
var planner = new HandlebarsPlanner(plannerOptions);
var ask = "Tell four knock-knock jokes: two about dogs, one about cats and one about ducks";
var plan = await planner.CreatePlanAsync(kernel, ask);
var result = await plan.InvokeAsync(kernel);
Console.Write ($"Results: {result}");
```

Python

```py
import asyncio
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planners import FunctionCallingStepwisePlanner, FunctionCallingStepwisePlannerOptions
from semantic_kernel.utils.settings import openai_settings_from_dot_env
import semantic_kernel as sk
from dotenv import load_dotenv
async def main():
    kernel = sk.Kernel()
    api_key, org_id = openai_settings_from_dot_env()
    gpt35 = OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id, service_id = "gpt35")
    gpt4 = OpenAIChatCompletion("gpt-4", api_key, org_id, service_id = "gpt4")
    kernel.add_service(gpt35)
    kernel.add_service(gpt4)
    kernel.add_plugin(None, plugin_name="jokes", parent_directory="../../plugins/")
    planner_options = FunctionCallingStepwisePlannerOptions(
        max_tokens=4000,
    )
    planner = FunctionCallingStepwisePlanner(service_id="gpt4", options=planner_options)
    prompt = "Create four knock-knock jokes: two about dogs, one about cats and one about ducks"
    result = await planner.invoke(kernel, prompt)
    print(result.final_answer)
if __name__ == "__main__":
    asyncio.run(main())
```

在前面的代码中，我们创建了我们的内核并将其中的笑话插件添加到其中。现在，让我们创建计划者。

结果

您将得到以下 Python 和 C# 的结果：

```py
1st Joke: Knock, knock!
Who's there?
Dog!
Dog who?
Dog who hasn't barked yet because he doesn't want to interrupt this hilarious joke!
2nd Joke: Knock, knock!
Who's there?
Dog!
Dog who?
Dog who forgot his keys, let me in!
3rd Joke: Knock, knock!
Who's there?
cat!
cat who?
Cat-ch me if you can, I'm the gingerbread man!
4th Joke: Knock, knock!
Who's there?
Duck!
Duck who?
Duck down, I'm throwing a pie!
```

注意，使用单个用户请求和单个对 `invoke` 的调用，语义内核生成了多个响应，而无需您编写任何循环，创建任何额外的 UI，或自己链式调用任何函数。

让我们看看幕后发生了什么。

# 计划者是如何工作的？

在幕后，计划者使用 LLM 提示来生成计划。例如，您可以通过导航到位于 [`github.com/microsoft/semantic-kernel/blob/7c3a01c1b6a810677d871a36a9211cca0ed7fc4d/dotnet/src/Planners/Planners.Handlebars/Handlebars/CreatePlanPrompt.handlebars`](https://github.com/microsoft/semantic-kernel/blob/7c3a01c1b6a810677d871a36a9211cca0ed7fc4d/dotnet/src/Planners/Planners.Handlebars/Handlebars/CreatePlanPrompt.handlebars) 的 Semantic Kernel 存储库中的提示文件来查看 `HandlebarsPlanner` 所使用的提示。

提示的最后几行对于理解计划者的工作方式至关重要：

```py
## Start
Now take a deep breath and accomplish the task:
1\. Keep the template short and sweet. Be as efficient as possible.
2\. Do not make up helpers or functions that were not provided to you, and be especially careful to NOT assume or use any helpers or operations that were not explicitly defined already.
3\. If you can't fully accomplish the goal with the available helpers, just print "{{insufficientFunctionsErrorMessage}}".
4\. Always start by identifying any important values in the goal. Then, use the `\{{set}}` helper to create variables for each of these values.
5\. The template should use the \{{json}} helper at least once to output the result of the final step.
6\. Don't forget to use the tips and tricks otherwise the template will not work.
7\. Don't close the ``` handlebars block until you're done with all the steps.

```py

The preceding steps define the set of rules that the planner uses to generate a plan using Handlebars.

Also, inside the prompt is what we call the **function manual** – that is, the instructions that the LLM will use to convert functions loaded into the kernel into text descriptions that are suitable for an LLM prompt:

```

{{#each functions}}

### `{{doubleOpen}}{{PluginName}}{{../nameDelimiter}}{{Name}}{{doubleClose}}`

Description: {{Description}}

输入：

{{#each Parameters}}

- {{Name}}:

{{~#if ParameterType}} {{ParameterType.Name}} -

{{~else}}

{{~#if Schema}} {{getSchemaTypeName this}} -{{/if}}

{{~/if}}

{{~#if Description}} {{Description}}{{/if}}

{{~#if IsRequired}} (required){{else}} (optional){{/if}}

{{/each}}

输出：

{{~#if ReturnParameter}}

{{~#if ReturnParameter.ParameterType}} {{ReturnParameter.ParameterType.Name}}

{{~else}}

{{~#if ReturnParameter.Schema}} {{getSchemaReturnTypeName ReturnParameter}}

{{else}} string{{/if}}

{{~/if}}

{{~#if ReturnParameter.Description}} - {{ReturnParameter.Description}}{{/if}}

{{/if}}

{{/each}}

```py

In summary, the planner is just a plugin that uses an AI service to translate a user request into a series of callable function steps, and then it generates the code that calls these functions, returning the result.

To decide which functions to call and how to call them, planners rely on the descriptions you wrote for the plugin. For native functions, the descriptions are in function decorators, while for semantic functions, they are in the `config.json` file.

Planners will send your descriptions to an AI service as part of a prompt, with instructions that tell the AI service how to combine your descriptions into a plan. Writing good descriptions can help the AI service to create better plans.

Here are some things you should do:

*   `required=true` so that the model knows to provide an input. If you don’t do that, the created plan may not include a required parameter and will fail when executing.
*   **Provide examples**: Your description can provide examples of how to use the function and what the acceptable inputs and outputs are. For example, if you have a function that turns lights on in a location with the description “*Location where the lights should be turned on*,” and the location must be the kitchen or the garage, you can add “*The location must be either ‘kitchen’ or ‘garage’*” to the description. With that extra description, the planner will know not to call that function if the user asks to “*turn everything on in* *the bedroom*.”

Here are some things to avoid:

*   **Short descriptions**: If your function, inputs, or output descriptions are very short, it’s possible that they are not going to convey enough information to the planner about the context where they would be used. For example, it’s better to say that the output of a function is “*a knock-knock joke that follows a theme*” than “*joke*.”
*   **Very long descriptions**: Remember that the descriptions will be submitted as part of a prompt that will incur costs. If your description is very long (for example, you provide three examples for every function), you will pay for it. Make sure that what you write in the descriptions is close to what’s necessary.
*   **Conflicting descriptions**: If many of your functions have similar or the same description, the planner can get confused. For example, imagine that you create a jokes plugin that can create different types of jokes (knock-knock jokes, puns, absurdist jokes, etc.) but the description of all the functions is simply “*creates a joke*.” The planner will not know which function to call because the description tells it that all functions do the same thing.

If you are not getting the results that you expect when you use the planner, the first place you should look is in the descriptions you wrote for the functions, their inputs, and their outputs. Usually, just improving the descriptions a little helps the planner a lot. Another solution is to use a newer model. For example, if the plans are failing when you use GPT-3.5 and you already checked the descriptions, you may consider testing GPT-4 and seeing whether the results improve substantially.

Let’s see a comprehensive example.

# Controlling home automation with the planner

To get a better idea of what the planner can do, we will create a home automation application. We will not actually write functions that really control home automation, but assuming those exist, we will write their wrappers as native functions. We will also add a semantic function to our kernel and incorporate it into the planner.

We assume that we have a house with four rooms – a garage, kitchen, living room, and bedroom. We have automations to operate our garage door, operate the lights in all rooms, open the windows in the living room and in the bedroom, and operate the TV.

Since our objective is to learn about Semantic Kernel and not about home automation, these functions will be very simple. We want our user to be able to say something such as “*turn on the lights of the bedroom*,” and the result will be that our native function will say “*bedroom lights* *turned on*.”

The power of using the planner is shown when a user makes requests that require multiple steps, such as “*turn off the bedroom light and open the window*,” or even something more complex, such as “*turn off the living room lights and put on a highly rated horror movie on* *the TV*.”

## Creating the native functions

We will start by creating four native functions for home automation, one to operate the lights, one to operate the windows, one to operate the TV, and one to operate the garage door:

C#

```

using System.ComponentModel;

using Microsoft.SemanticKernel;

public class HomeAutomation

{

[KernelFunction, Description("打开或关闭客厅、厨房、卧室或车库里电灯。")]

public string OperateLight(

[Description("是否打开或关闭电灯。必须是 'on' 或 'off'")] string action,

[描述("必须打开或关闭灯光的位置。必须是 'living room', 'bedroom', 'kitchen' 或 'garage'")] string location)

{

string[] validLocations = {"living room", "bedroom"};

if (validLocations.Contains(location))

{

string exAction = $"将 {location} 的灯光状态更改为 {action}。";

Console.WriteLine(exAction);

return exAction;

}

else

{

string error = $"指定的位置无效 {location}。";

return error;

}

}

```py

The most important parts of the function are the `Description` decorators for the function itself and the parameters. They are the ones that the planner will read to learn how to use the function. Note that the descriptions specify what the valid parameters are. This helps the planner decide what to do when it receives an instruction for all locations.

The function just verifies that the location is valid and prints the action that the home automation would have taken if it were real.

The other functions simply repeat the same preceding template for their objects (the window, TV, and garage door);

```

[内核函数, 描述("打开或关闭客厅或卧室的窗户。")]

public string OperateWindow(

[描述("是否打开或关闭窗户。必须是 'open' 或 'close'")] string action,

[描述("要打开或关闭窗户的位置。必须是 'living room' 或 'bedroom'")] string location)

{

string[] validLocations = {"living room", "bedroom"};

if (validLocations.Contains(location))

{

string exAction = $"将 {location} 的窗户状态更改为 {action}。";

Console.WriteLine(exAction);

return exAction;

}

else

{

string error = $"指定的位置无效 {location}。";

return error;

}

}

[内核函数, 描述("在客厅或卧室的电视上放电影。")]

public string OperateTV(

[描述("要在电视上播放的电影。")] string movie,

[描述("电影应播放的位置。必须是 'living room' 或 'bedroom'")] string location)

{

string[] validLocations = {"kitchen", "living room", "bedroom", "garage" };

if (validLocations.Contains(location))

{

string exAction = $"在 {location} 的电视上播放 {movie}。";

Console.WriteLine(exAction);

return exAction;

}

else

{

string error = $"指定的位置无效 {location}。";

return error;

}

}

[内核函数, 描述("打开或关闭车库门。")]

public string OperateGarageDoor(

[描述("对车库门执行的操作。必须是 'open' 或 'close'")] string action)

{

string exAction = $"将车库门的状态更改为 {action}。";

Console.WriteLine(exAction);

return exAction;

}

}

```py

Python

```

from typing_extensions import Annotated

from semantic_kernel.functions.kernel_function_decorator import kernel_function

class HomeAutomation:

def __init__(self):

pass

@kernel_function(

description="打开或关闭客厅或卧室的窗户。",

name="OperateWindow",

)

def OperateWindow(self,

location: Annotated[str, "要打开或关闭窗户的位置。必须是 'living room' 或 'bedroom'"],

action: Annotated[str, "是否打开或关闭窗户。必须是 'open' 或 'close'"]) \

-> Annotated[str, "对窗户执行的操作。"]]:

if location in ["living room", "bedroom"]:

action = f"将 {location} 的窗户状态更改为 {action}。"

print(action)

return action

else:

error = f"指定的位置无效 {location}。"

return error

```py

The preceding function is straightforward, checking that the location passed as a parameter is valid and printing what the automation would have done.

The most important parts of the function are the descriptions for inside the `kernel_function` and for each of the `Annotated` parameters, as the descriptions are what the planner will use to decide what to do.

Note that the descriptions specify what the valid parameters are. This helps the planner decide what to do when it receives a request to perform an action for all locations.

Now, let’s create the other functions, following a similar structure:

```

@kernel_function(

description="打开或关闭客厅、厨房、卧室或车库里的大灯。",

name="OperateLight",

)

def OperateLight(self,

location: Annotated[str, "要打开或关闭灯光的位置。必须是 'living room', 'kitchen', 'bedroom' 或 'garage'"],

action: Annotated[str, "是否打开或关闭灯光。必须是 'on' 或 'off'"]\


-> Annotated[str, "在灯光上执行的动作。"]:

if location in ["kitchen", "living room", "bedroom", "garage"]:

action = f"{location} 灯光的状态已更改为 {action}。"

print(action)

return action

else:

error = f"指定的位置 {location} 无效。"

return error

@kernel_function(

description="在客厅或卧室的电视上播放电影。",

name="OperateTV",

)

def OperateTV(self,

movie: Annotated[str, "要在电视上播放的电影。"],

location: Annotated[str, "电影应播放的位置。必须是 'living room' 或 'bedroom'"]

)\


-> Annotated[str, "在电视上执行的动作。"]:

if location in ["living room", "bedroom"]:

action = f"在 {location} 的电视上播放 {movie}。"

print(action)

return action

else:

error = f"指定的位置 {location} 无效。"

return error

@kernel_function(

description="打开或关闭车库门。",

name="OperateGarageDoor"

)

def OperateGarageDoor(self,

action: Annotated[str, "对车库门执行的动作。必须是 'open' 或 'close'"]\


-> Annotated[str, "在车库门上执行的动作。"]:

action = f"将车库门的状态更改为 {action}。"

print(action)

return action

```py

Now that we’re done with native functions, let’s add a semantic function.

## Adding a semantic function to suggest movies

In addition to creating the preceding native functions that control different components of the house, we are also going to create a semantic function to suggest movies based on what the user requests. Semantic functions allow the user to make requests that require the use of an AI service – for example, to find the name of a movie based on a description or the name of an actor. You’ll see that planners can seamlessly combine semantic and native functions.

As is always the case, the semantic function is the same for both C# and Python, but we need to carefully configure the `skprompt.txt` and `config.json` files to help the planner find the function and understand how to use it.

We start by creating a prompt:

skprompt.txt

The prompt is very simple, and it simply asks for a suggestion for a movie. To make things easier for the planner, the prompt specifies that GPT should only respond with the title of the movie, as well as what to do if the user already knows the movie they want to watch:

```

根据以下请求，建议一个你认为请求者可能会喜欢的电影。如果请求已经是电影标题，只需返回该电影标题。

仅响应电影标题，不要包含其他内容。

Request:

{{ $input }}

```py

Now, let’s see the configuration file:

config.json

Here, it’s again very important to fill in the `description` fields with as much detail as possible, as they are what the planner will use to decide which functions to call:

```

{

"schema": 1,

"name": "RecommendMovie",

"type": "completion",

"execution_settings": {

"default": {

"temperature": 0.8,

"number_of_responses": 1,

"top_p": 1,

"max_tokens": 4000,

"presence_penalty": 0.0,

"frequency_penalty": 0.0

}

},

"input_variables": [

{

"name": "input",

"description": "用户想要观看的电影的名称或描述。",

"required": true

}

]

}

```py

Now that all the native and semantic functions are configured, let’s call the planner and see what it can do.

## Invoking the planner

Once you load the kernel with all these functions, all you need to do is invoke the planner and pass the user request to it.

We are going to make four requests to the planner:

*   *Turn on the lights in* *the kitchen*
*   *Open the windows of the bedroom, turn the lights off, and put on The Shawshank Redemption on* *the TV*
*   *Close the garage door and turn off the lights in all* *the rooms*
*   *Turn off the lights in all rooms and play a movie in which Tom Cruise is a lawyer, in the* *living room*

Using the existing plugins, the planner will take care of everything that is needed to fulfill these requests. For example, to fulfill the last request, the planner needs to call the `OperateLight` native function for each of the four rooms and ask GPT for a recommendation of a movie in which Tom Cruise is a lawyer, which will likely be *A Few Good Men* or *The Firm*. The planner will automatically call the functions and simply provide the results.

Python

The core part of the code is to create and execute the plan, using `create_plan` and `invoke_async`, and then print the results:

```

from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from semantic_kernel.planning.stepwise_planner import StepwisePlanner

import semantic_kernel as sk

from HomeAutomation import HomeAutomation

from dotenv import load_dotenv

import asyncio

async def fulfill_request(planner: StepwisePlanner, request):

print("正在满足请求：" + request)

variables = sk.ContextVariables()

plan = planner.create_plan(request)

result = await plan.invoke_async(variables)

print(result)

print("请求完成。\n\n")

```py

Then, in the main function, we load the native functions and the semantic function in the kernel. This will make them available to the planner:

```

async def main():

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()

gpt4 = OpenAIChatCompletion("gpt-4", api_key, org_id)

kernel.add_chat_service("gpt4", gpt4)

planner = StepwisePlanner(kernel)

kernel.import_skill(HomeAutomation())

kernel.import_semantic_skill_from_directory("../plugins/MovieRecommender", "RecommendMovie")

```py

After loading the function, we simply call `fulfill_request`, which will create and execute a plan for each `ask`:

```

await fulfill_request(kernel, planner, "Turn on the lights in the kitchen")

await fulfill_request(kernel, planner, "Open the windows of the bedroom, turn the lights off and put on Shawshank Redemption on the TV.")

await fulfill_request(kernel, planner, "Close the garage door and turn off the lights in all rooms.")

await fulfill_request(kernel, planner, "Turn off the lights in all rooms and play a movie in which Tom Cruise is a lawyer in the living room.")

if __name__ == "__main__":

load_dotenv()

asyncio.run(main())

```py

C#

We start by creating a kernel and adding all the native functions and the semantic function we created for it. This will make these functions available to the planner:

```

using Microsoft.SemanticKernel;

using Microsoft.SemanticKernel.Planning.Handlebars;

#pragma warning disable SKEXP0060

var (apiKey, orgId) = Settings.LoadFromFile();

var builder = Kernel.CreateBuilder();

builder.AddOpenAIChatCompletion("gpt-4", apiKey, orgId);

builder.Plugins.AddFromType<HomeAutomation>();

builder.Plugins.AddFromPromptDirectory("../../../plugins/MovieRecommender");

var kernel = builder.Build();

```py

We then create a function that receives a `planner` and an `ask`, creating and executing a plan to fulfill that request:

```

void FulfillRequest(HandlebarsPlanner planner, string ask)

{

Console.WriteLine($"Fulfilling request: {ask}");

var plan = planner.CreatePlanAsync(kernel, ask).Result;

var result = plan.InvokeAsync(kernel, []).Result;

Console.WriteLine("Request complete.");

}

```py

The last step is to create the planner and call the `FulfillRequest` function we created for each `ask`:

```

var plannerOptions = new HandlebarsPlannerOptions()

{

ExecutionSettings = new OpenAIPromptExecutionSettings()

{

Temperature = 0.0,

TopP = 0.1,

MaxTokens = 4000

},

AllowLoops = true

};

var planner = new HandlebarsPlanner(plannerOptions);

FulfillRequest(planner, "Turn on the lights in the kitchen");

FulfillRequest(planner, "Open the windows of the bedroom, turn the lights off and put on Shawshank Redemption on the TV.");

FulfillRequest(planner, "Close the garage door and turn off the lights in all rooms.");

FulfillRequest(planner, "Turn off the lights in all rooms and play a movie in which Tom Cruise is a lawyer in the living room.");

```py

Note that the code that uses the planner was very short. Let’s see the results:

```

Fulfilling request: Turn on the lights in the kitchen

将厨房灯的状态更改为开启。

Request complete.

Fulfilling request: Open the windows of the bedroom, turn the lights off and put on Shawshank Redemption on the TV.

将卧室窗户的状态更改为开启。

将卧室灯的状态更改为关闭。

在卧室的电视上播放《肖申克的救赎》。

Request complete.

Fulfilling request: Close the garage door and turn off the lights in all rooms.

将车库门的状态更改为关闭。

将客厅灯的状态更改为关闭。

将卧室灯的状态更改为关闭。

将厨房灯的状态更改为关闭。

将车库灯的状态更改为关闭。

Request complete.

Fulfilling request: Turn off the lights in all rooms and play a movie in which Tom Cruise is a lawyer in the living room.

将客厅灯的状态更改为关闭。

将卧室灯的状态更改为关闭。

将厨房灯的状态更改为关闭。

将车库灯的状态更改为关闭。

在客厅的电视上播放《非常嫌疑犯》。

Request complete.

```

规划器完美地执行了每个请求，而你无需编写任何代码。当用户询问诸如“*关闭所有房间的灯光*”之类的事情时，规划器意识到需要调用厨房、卧室、客厅和车库里面的函数。

当用户请求一部由汤姆·克鲁斯扮演律师的电影时，规划器意识到在调用`OperateTV`函数将电影放到电视上之前，需要调用一个语义函数来找到电影的名字，而你无需为此显式地编写代码。

# 摘要

在本章中，我们介绍了规划器，这是一个强大的功能，允许用户以最小的开发人员努力执行非常复杂的流程。我们学习了何时使用规划器以及可能存在的问题。我们还学习了如何使用规划器，以及如何编写插件中函数的描述，以便规划器更容易地将它们组合起来。然后我们看到了一个更长的示例，展示了如何使用规划器让用户结合原生和语义函数。

在下一章中，我们将探讨将外部数据提供给语义内核的方法。稍后，我们将搜索与外部数据配对，以便模型可以使用超出模型上下文窗口的大量数据。
