# 第十三章：<st c="0">12</st>

# <st c="3">微调 ChatGPT 以创建独特的 API 模型</st>

<st c="51">在本章中，我们将探讨如何微调 ChatGPT API。</st> <st c="114">这一变革性过程赋予了开发者调整 ChatGPT API 响应行为以符合其特定需求的能力。</st> <st c="252">我们将深入微调的艺术，探索其中的复杂技术和关键因素。</st> <st c="370">通过掌握构建和部署我们自己的微调模型，我们将解锁以前所未有的方式增强我们的 AI 应用潜能。</st>

<st c="532">ChatGPT 微调是一个涉及在特定数据集上训练预训练语言模型，例如</st> `<st c="627">gpt-3.5-turbo</st>`<st c="640">，以提高其性能并使其适应特定任务或领域的过程。</st> <st c="736">微调过程通常从一个精心策划和标记的数据集开始，并涉及使用迁移学习等技术在此数据集上训练模型。</st> <st c="926">在微调过程中，模型的参数被调整以使其更准确、更符合上下文，从而在目标领域生成响应。</st> <st c="1079">通过微调，模型可以获取特定领域的知识、语言模式和细微差别，使其能够为特定应用或</st> <st c="1268">用例生成更相关和连贯的响应。</st>

<st c="1278">我们将演示如何使用 JSONL 提示和</st> <st c="1383">完成数据来微调 ChatGPT 以生成简洁的书籍摘要。</st>

<st c="1399">在本章中，我们将涵盖以下主题：</st> <st c="1435">：</st>

+   <st c="1452">微调 ChatGPT 和</st> <st c="1477">数据集准备</st>

+   <st c="1496">构建和使用</st> <st c="1520">微调模型</st>

# <st c="1536">技术要求</st>

<st c="1559">本章的技术要求如下列出：</st> <st c="1615">：</st>

+   <st c="1626">您的机器上已安装 Python 3.7 或更高版本</st>

+   <st c="1672">OpenAI</st> <st c="1683">API 密钥</st>

+   <st c="1690">已安装 OpenAI Python</st> <st c="1709">库</st>

<st c="1726">在接下来的章节中，我们将通过检查可用的模型、建立我们的开发环境以及熟悉我们即将构建的书籍摘要微调模型，开始我们的 ChatGPT 微调之旅。</st>

<st c="1991">您可以在 GitHub 上找到本章的所有代码示例</st> <st c="2051">在</st> [<st c="2054">https://github.com/PacktPublishing/Building-AI-Applications-with-ChatGPT-API</st>](https://github.com/PacktPublishing/Building-AI-Applications-with-ChatGPT-API)<st c="2130">。</st>

# <st c="2131">微调 ChatGPT 和数据集准备</st>

在本节中，您将了解微调 ChatGPT 模型的过程。<st c="2175">我们将讨论可用于微调的 ChatGPT 模型，并提供它们训练和使用的成本信息。</st> <st c="2257">我们还将涵盖安装</st> `<st c="2423">openai</st>` <st c="2429">库以及在终端会话中设置 API 密钥作为环境变量的步骤。</st> <st c="2372">本节将作为微调的概述，包括其好处以及训练微调模型所需的设置。</st>

微调以多种方式增强了 API 模型的能力。<st c="2636">首先，它比仅设计提示产生了更高的质量结果。</st> <st c="2706">通过包含比提示中可以容纳的更多训练示例，微调使模型能够掌握更广泛的模式和细微差别。</st> <st c="2786">其次，它通过使用更短的提示来减少令牌使用，从而实现更高效的处理。</st> <st c="2939">此外，微调促进了低延迟请求，使交互更快、更响应。</st>

GPT-3.5 已经在来自互联网的大量文本语料库上进行了广泛的预训练。<st c="3153">当给出有限示例的提示时，它通常表现出理解预期任务并生成合理补全的能力——这被称为**<st c="3422">少样本学习</st>**。</st> 然而，微调通过利用更大的示例集，将少样本学习推进了一步，超越了提示的能力。<st c="3461">这种全面的训练使模型在广泛的任务上表现出卓越的性能。</st> <st c="3574">此外，它还提高了模型在处理各种任务时的性能。</st>

如果一个模型已经经过微调，我们可以使用生成的模型，而无需传递任何进一步的训练数据。<st c="3660">在</st> *<st c="3785">表 12.1</st>*<st c="3795">中，该表来自官方 OpenAI 定价页面[<st c="3844">https://openai.com/pricing</st>](https://openai.com/pricing)，您可以看到所有可用于微调的 ChatGPT 模型，每个模型都附有关于它们每 1 百万令牌的训练和使用成本的详细信息，截至<st c="4049">2024 年 5 月。</st>

![](img/B22414_12_01.jpg)

<st c="4224">表 12.1 – ChatGPT 模型微调的价格</st>

在微调过程中，有必要以 JSONL 文件的形式提供训练数据，并且训练令牌计数特指该文件中包含的用户和助手字段内使用的那些。</st> <st c="4278">在该文件中，训练令牌计数特指该文件中包含的用户和助手字段内使用的那些。</st>

<st c="4508">JSONL 文档的每一行应包含一个</st> `<st c="4561">用户</st>` <st c="4565">和</st> `<st c="4570">助手</st>` <st c="4579">字段，这些字段对应我们希望的训练数据，如下所示：</st>

```py
 {"messages": [{"role": "user", "content": "The Adventure Begins ->"}, {"role": "assistant", "content": "A thrilling tale of courage and discovery."}]}
{"messages": [{"role": "user", "content": "Secrets Unveiled ->"}, {"role": "assistant", "content": "An intriguing mystery that will keep you guessing until the end."}]}
```

<st c="4967">这个片段是展示如何在 JSONL 文件中显示两条训练行的示例。</st> <st c="5067">在接下来的章节中，我们将使用包含多个</st> `<st c="5145">数据点</st>`的完整 JSONL 文件。</st>

<st c="5157">我们的目标是开发一个专门为用户提供书籍生成简洁、一句话摘要的微调模型。</st> <st c="5300">为了实现这一目标，我们将构建一个新的 JSONL 训练文件，其中包含提示字段中的书籍标题和完成字段中的相应摘要。</st> <st c="5473">在完成训练阶段后，这个过程将使我们的模型具备根据用户提供的输入有效生成书籍摘要的必要技能。</st>

**<st c="5652">JSONL</st>** <st c="5658">代表</st> **<st c="5674">JSON Lines</st>** <st c="5684">格式，这是一种基于文本的数据交换格式，其中文件中的每一行代表一个单独的 JSON 对象。</st> <st c="5731">这种格式通常用于存储和交换结构化数据，使得以流式方式读取、写入和处理数据变得容易。</st>

<st c="5939">在训练阶段完成后，OpenAI 将为我们微调的模型提供一个独特且专有的名称。</st> <st c="6067">这个独特的名称可以用于 ChatGPT 完成提示中，以有效地与</st> <st c="6169">模型互动。</st>

<st c="6179">在开始准备</st> <st c="6215">我们的数据集之前，我们需要确保创建一个名为</st> `<st c="6288">Fine Tune</st>`<st c="6297">的新目录，并使用 VS Code 或你选择的任何其他编辑器打开它。</st> <st c="6357">在激活我们在</st> *<st c="6412">第一章</st>*<st c="6421">中设置的虚拟环境后，我们需要安装</st> `<st c="6498">openai</st>` <st c="6504">库。</st> <st c="6514">为此，在 VS Code 中打开一个新的终端并输入以下内容：</st>

```py
 $ pip install --upgrade openai
```

<st c="6611">这一行将确保</st> `<st c="6663">openai</st>` <st c="6669">包的最新版本已安装在你的设备上。</st>

<st c="6712">接下来，我们需要创建用于微调任务所需的文件。</st> <st c="6788">从 VS Code</st> **<st c="6805">资源管理器</st>** <st c="6813">窗口中，创建</st> <st c="6829">两个文件：</st>

+   `<st c="6839">main.py</st>`<st c="6847">：这里将存放用于微调</st> <st c="6903">ChatGPT 模型</st>

+   `<st c="6916">train_data.jsonl</st>`<st c="6933">: 这将保存 JSONL</st> <st c="6961">训练数据集</st>

<st c="6977">要从本地终端微调模型，您还需要在</st> `<st c="7086">main.py</st>` <st c="7093">文件中提供 ChatGPT API 密钥，如下所示：</st> <st c="7100">:</st>

<st c="7111">main.py</st>

```py
 import time
from openai import OpenAI
client = OpenAI(api_key=<st c="7247">openai</st> library. We initialize a client object of the <st c="7300">OpenAI</st> class with the provided API key to access OpenAI’s API services, enabling further interaction with ChatGPT through fine-tuning.
			<st c="7434">Finally, we can add the desired datasets with the appropriate format in our</st> `<st c="7511">train_data.jsonl</st>` <st c="7527">file.</st> <st c="7534">We will add a collection of book summaries, represented as user-assistant pairs.</st> <st c="7615">Each entry consists of a</st> `<st c="7640">user</st>` <st c="7644">field that introduces the book with a summary title, followed</st> <st c="7706">by an</st> `<st c="7713">assistant</st>` <st c="7722">field that provides a brief</st> <st c="7751">description of the book.</st> `<st c="7776">user</st>` <st c="7780">highlights different genres, themes, and emotions associated with the books, while</st> `<st c="7864">assistant</st>` <st c="7873">captures the essence of each story.</st> <st c="7910">The summaries encompass a variety of genres, including adventure, mystery, romance, science fiction, and fantasy.</st> <st c="8024">This JSONL file data will serve as a dataset to train a fine-tuned model to generate concise book summaries, based on</st> <st c="8142">user-provided input:</st>

```

{"messages": [{"role": "user", "content": "The Adventure Begins ->"}, {"role": "assistant", "content": "一段关于勇气和发现的惊险故事。"}]}

{"messages": [{"role": "user", "content": "Secrets Unveiled ->"}, {"role": "assistant", "content": "一个引人入胜的谜团，直到最后都会让你猜测。"}]}

{"messages": [{"role": "user", "content": "Love and Betrayal ->"}, {"role": "assistant", "content": "一段关于爱情、信任和欺骗的心碎故事。"}]}

{"messages": [{"role": "user", "content": "The Quest for Freedom ->"}, {"role": "assistant", "content": "一段引人入胜的自我发现和解放之旅。"}]}

{"messages": [{"role": "user", "content": "Beyond the Stars ->"}, {"role": "assistant", "content": "一部推动想象力边界的史诗科幻冒险。"}]}

{"messages": [{"role": "user", "content": "The Power Within ->"}, {"role": "assistant", "content": "对内在力量和韧性的深刻探索。"}]}

{"messages": [{"role": "user", "content": "Shadows of the Past ->"}, {"role": "assistant", "content": "关于救赎和面对过去幽灵的令人不安的故事。"}]}

{"messages": [{"role": "user", "content": "A World Apart ->"}, {"role": "assistant", "content": "对平行宇宙和选择力量的迷人探索。"}]}

```py

			<st c="9501">You can find the full JSONL file on our Git</st> <st c="9546">repository:</st> [<st c="9558">https://github.com/PacktPublishing/Building-AI-Applications-with-ChatGPT-API/blob/main/Chapter12%20FineTuning/train_data_prepared.jsonl</st>](https://github.com/PacktPublishing/Building-AI-Applications-with-ChatGPT-API/blob/main/Chapter12%20FineTuning/train_data_prepared.jsonl)<st c="9693">.</st>
			<st c="9694">This section provided an overview of fine-tuning ChatGPT models.</st> <st c="9760">It discussed how fine-tuning enhances few-shot learning and the necessity of providing training data in a JSONL file.</st> <st c="9878">We discussed the fine-tuned model that we will build to generate concise book summaries, based on user-provided input.</st> <st c="9997">We also covered the environment setup, which included the installation of the</st> `<st c="10075">openai</st>` <st c="10081">library and setting the</st> <st c="10106">API key.</st>
			<st c="10114">With our dataset prepared, we have laid the foundation to construct the fine-tuned model, a crucial step that will be explored in detail in the</st> <st c="10259">next section.</st>
			<st c="10272">Building and using the fine-tuned model</st>
			<st c="10312">In this section, we will explore the process of creating and utilizing a fine-tuned model using OpenAI’s library in Python.</st> <st c="10437">OpenAI offers newcomers the opportunity to take advantage of a $5 credit to access the ChatGPT API and its</st> <st c="10544">fine-tuning services.</st>
			<st c="10565">Fine-tuning involves building a specialized model based on an existing base model, and in our example, we will use the most advanced ChatGPT model available for fine-tuning, called</st> `<st c="10747">gpt-3.5-turbo</st>`<st c="10760">. We will improve the performance of that model for book</st> <st c="10817">summarization tasks.</st>
			<st c="10837">We will learn how to start a fine-tuning job, which</st> <st c="10889">uploads and processes the training data, and then we’ll monitor its progress until completion.</st> <st c="10985">Once the fine-tuning job is done, we will use the newly created fine-tuned model to generate text.</st> <st c="11084">We’ll learn how to</st> <st c="11103">make requests to the fine-tuned model using the completions API, and then we’ll cover how to manage and delete fine-tuned models</st> <st c="11232">if needed.</st>
			<st c="11242">We will begin by using our newly created JSONL file to create the fine-tuned model with the following code,</st> <st c="11351">inside</st> `<st c="11358">main.py</st>`<st c="11365">:</st>
			<st c="11367">main.py</st>

```

file = client.files.create(

    file=open("train_data_prepared.jsonl", "rb"),

    purpose="fine-tune"

)

打印(file)

tuned_model = client.fine_tuning.jobs.create(

    training_file=file.id,

    model="gpt-3.5-turbo"

)

打印(tuned_model)

```py

			<st c="11594">This code segment illustrates the process of initiating the fine-tuning of a GPT model using the</st> `<st c="11692">openai</st>` <st c="11698">library.</st> <st c="11708">The</st> `<st c="11712">train_data_prepared.jsonl</st>` <st c="11737">file is opened in binary read mode, which contains the</st> <st c="11792">prepared training data in the JSONL format.</st> <st c="11837">This file is uploaded to OpenAI so that we can use it later for fine-tuning.</st> <st c="11914">Then, a fine-tuning job is initiated using the</st> `<st c="11961">fine_tuning.jobs</st>` <st c="11977">attribute.</st> <st c="11989">The training file for fine-tuning is specified with the</st> `<st c="12045">training_file</st>` <st c="12058">parameter, set to the ID of the previously uploaded file.</st> <st c="12117">The</st> `<st c="12121">model</st>` <st c="12126">parameter indicates the base model to be used for</st> <st c="12176">fine-tuning, which is set here to</st> `<st c="12211">gpt-3.5-turbo</st>`<st c="12224">. Finally, the details of the fine-tuning job are printed, providing information about the job ID and other</st> <st c="12332">relevant details:</st>

```

文件对象(id=<st c="12364">'file-UMMzAaCumnkHc8Vx5WwTAEs0'</st>, bytes=5329, created_at=1714533608, filename='train_data_prepared.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)

微调作业(id=<st c="13206">ftjob-EZ2V9AS46DjaSv6jWy0S4Pt7</st>。需要注意的是，模型不是立即创建的。创建模型的过程可能有所不同，可能需要几分钟到几小时。您可以通过查看<st c="13470">状态</st>参数来监控模型创建的状态，目前指示系统正在验证 JSONL 文件。

            <st c="13564">您可以通过在以下行中添加以下行来监控您模型的状况</st> `<st c="13652">main.py</st>` <st c="13659">文件：</st>

```py
 job = client.fine_tuning.jobs.retrieve(tuned_model.id)
print(job.status)
while job.status != "succeeded":
    job = client.fine_tuning.jobs.retrieve(tuned_model.id)
    print(job.status)
    time.sleep(5)
print(job.fine_tuned_model)
```

            <st c="13886">此代码片段处理在上一段中启动的微调作业的监控。</st> <st c="13986">最初，</st> `<st c="14001">retrieve</st>` <st c="14009">方法被用来检索关于微调作业的详细信息。</st> <st c="14076">之前启动的微调作业的</st> `<st c="14080">id</st>` <st c="14082">，即</st> `<st c="14128">tuned_model.id</st>`<st c="14142">，作为参数传递给此</st> <st c="14177">方法。</st> <st c="14185">随后，打印作业的状态以提供其进度的初步指示。</st> <st c="14282">然后使用</st> `<st c="14284">while</st>` <st c="14289">循环来持续检查作业的状态，直到它达到</st> `<st c="14375">succeeded</st>` <st c="14384">状态。</st> <st c="14392">在循环中，再次使用</st> `<st c="14460">retrieve</st>` <st c="14468">方法检索作业的状态，并打印状态以跟踪任何变化。</st> <st c="14533">此外，循环中包含了一个</st> `<st c="14549">time.sleep(5)</st>` <st c="14562">语句，在每次状态检查之间引入五秒的暂停，以防止过多的 API 请求并确保有效利用资源。</st> <st c="14740">此迭代过程持续进行，直到作业状态转换为</st> `<st c="14809">succeeded</st>`<st c="14818">，表示微调过程的完成：</st>

```py
 validating_files
validating_files
running
running
running
succeeded <st c="14942">ft:gpt-3.5-turbo-0125:personal::9Jv8aUuC</st> ChatCompletionMessage(content='An exhilarating journey of self-discovery and courage.', role='assistant', function_call=None, tool_calls=None)
```

            <st c="15125">一旦我们的模型成功编译，我们可以使用</st> `<st c="15204">job.fine_tuned_model</st>` <st c="15224">变量来显示模型名称。</st> <st c="15235">在这种情况下，你的模型名称是</st> `<st c="15268">ft:gpt-3.5-turbo-0125:personal::9Jv8aUuC</st>`<st c="15308">。我们可以使用这个名称进行任何 chant 完成，就像任何基础</st> `<st c="15376">gpt-3.5-turbo</st>` <st c="15389">和</st> `<st c="15394">gpt-4</st>` <st c="15399">标准模型一样。</st> <st c="15417">您可以使用以下代码进行简单快捷的测试。</st> <st c="15463">这允许您无缝地与您的微调模型交互并评估其性能，而无需</st> <st c="15581">任何复杂性：</st>

            <st c="15599">main.py</st>

```py
 completion = client.chat.completions.create(
    model=job.fine_tuned_model,
    messages=[
        {"role": "user", "content": "The Adventure Begins ->"}
    ]
)
print(completion.choices[0].message.content)
```

            <st c="15795">在这里，我们只需提供我们的微调</st> <st c="15835">模型和一个书的名称，并期望它返回一个类似于我们 JSONL 文件中的描述。</st> <st c="15938">给定的示例包括几个参数和头信息，用于指定请求的详细信息。</st>

            <st c="16034">一旦执行请求，你将收到以下消息：</st>

```py
 An exhilarating journey of self-discovery and courage
```

            <st c="16158">前面的输出是标准的 ChatGPT 响应，除了这里我们使用了我们微调过的书籍摘要模型。</st> <st c="16272">为了</st> <st c="16274">验证模型是否正常工作，您可以比较响应与我们在 JSONL 文件中为相应书籍提供的训练数据</st> <st c="16420">标题</st> <st c="16427">提示：</st>

```py
 {"messages": [{"role": "user", "content": "The Adventure Begins ->"}, {"role": "assistant", "content": "<st c="16708">assistant</st> field from our training data is part of the same meaning as the ChatGPT API response. This means that the trained model shows an improved performance, incorporating the training data.
			<st c="16901">Finally, if you decide to remove or delete a fine-tuned model that you created, OpenAI provides a straightforward command that you can use.</st> <st c="17042">Execute the following command in your</st> `<st c="17080">main.py</st>` <st c="17087">file:</st>

```

client.models.delete(job.fine_tuned_model)

```py

			<st c="17136">It is important to note that deleting a fine-tuned model should be done judiciously, as it permanently removes the model and its associated data.</st> <st c="17283">Therefore, it is advisable to carefully consider your decision and ensure that you no longer need the fine-tuned model before proceeding with the</st> `<st c="17428">deletion</st>` <st c="17437">command.</st>
			<st c="17446">In this section, we have</st> <st c="17471">explored the entire life cycle of fine-tuning models.</st> <st c="17526">Armed with this knowledge, you are now equipped to continue your journey of experimentation by constructing fine-tuned models that specialize in various tasks and incorporate a larger volume of training data entries.</st> <st c="17743">This opens a realm of possibilities to refine and enhance the performance of your models, enabling you to achieve even more</st> <st c="17867">impressive results.</st>
			<st c="17886">Summary</st>
			<st c="17894">In this chapter, we discussed the concept of fine-tuning within the ChatGPT API, exploring how it can help us to tailor ChatGPT API responses to our specific needs.</st> <st c="18060">By training a pre-existing language model on a diverse dataset, we enhanced the</st> `<st c="18140">gpt-3.5-turbo</st>` <st c="18153">model performance and adapted it to a particular task and domain.</st> <st c="18220">Fine-tuning enriched the model’s capacity to generate accurate and contextually fitting responses by incorporating domain-specific knowledge and language patterns.</st> <st c="18384">Throughout the chapter, we covered several key aspects of fine-tuning, including the available models for customization, the associated costs, data preparation using JSONL files, the creation of fine-tuned models, and the utilization of these models with the ChatGPT API.</st> <st c="18656">We underscored the significance of fine-tuning to achieve superior outcomes, reduce token consumption, and enable faster and more</st> <st c="18786">responsive interactions.</st>
			<st c="18810">Additionally, the chapter offered a comprehensive step-by-step guide on dataset preparation for fine-tuning.</st> <st c="18920">It dived into the process of data organization using a JSONL file.</st> <st c="18987">The dataset comprised prompt-completion pairs representing book summaries.</st> <st c="19062">The tool analyzed the data and proposed modifications, such as adding separators to user requests, eliminating unnecessary prefixes, and initiating assistant responses with whitespace characters.</st> <st c="19258">These modifications enhanced the model’s comprehension.</st> <st c="19314">Once the dataset was prepared and optimized, the chapter detailed the process of constructing and utilizing the fine-tuned model.</st> <st c="19444">It provided insights on initiating a fine-tuning job, monitoring its progress, and harnessing the fine-tuned model for text generation through the ChatGPT API.</st> <st c="19604">We emphasized the importance of fine-tuning in improving model performance and learned how to manage and delete fine-tuned models,</st> <st c="19735">as necessary.</st>
			<st c="19748">In this book, you embarked on a comprehensive journey into the world of ChatGPT and its API, gaining knowledge and skills to build a variety of powerful applications.</st> <st c="19916">You gradually built a ChatGPT clone, developed a SaaS application, integrated payment services, and explored frameworks such as Django and PyQt for web and desktop app development.</st> <st c="20097">You also discovered the possibilities of integration with Microsoft Word, Outlook, and PowerPoint, as well as transcribing audio files.</st> <st c="20233">You gained insights into selecting the right model and optimizing completions, and you even explored the realm of ChatGPT fine-tuning.</st> <st c="20368">With these skills, you are now well-equipped to create innovative AI applications and leverage the full potential of the</st> <st c="20489">ChatGPT API.</st>

```

```py

```
