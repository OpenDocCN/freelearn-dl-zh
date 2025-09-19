

# 转换器和它们的用途

在本章中，我们将了解转换器及其如何应用于执行各种NLP任务。NLP领域的典型任务涉及加载和处理数据，以便它可以无缝地用于下游。一旦数据被读取，另一个任务是转换数据，使其以各种模型可以使用的形式。一旦数据被转换成所需的格式，我们就用它来执行实际的任务，如分类、文本生成和语言翻译。

下面是本章中的菜谱列表：

+   加载数据集

+   对数据集中的文本进行分词

+   使用分词后的文本通过转换器模型进行分类

+   根据不同的需求使用不同的转换器模型

+   通过参考初始起始句子生成文本

+   使用预训练的转换器模型在不同语言之间翻译文本

# 技术要求

该章节的代码位于书籍GitHub仓库的`Chapter08`文件夹中（[https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/tree/main/Chapter08](https://github.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/tree/main/Chapter08)）。

如前几章所述，本章所需的包是`poetry`环境的一部分。或者，您也可以使用`requirements.txt`文件安装所有包。

# 加载数据集

在这个菜谱中，我们将学习如何加载公共数据集并与之交互。我们将使用`RottenTomatoes`数据集作为本菜谱的示例。这个数据集包含了电影的评分和评论。请参考以下链接获取更多关于数据集的信息：[https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

## 准备工作

作为本章的一部分，我们将使用来自`HuggingFace`网站（[huggingface.co](http://huggingface.co)）的库。对于这个菜谱，我们将使用数据集包。如果您需要从一个现有的笔记本开始工作，可以使用代码网站上的`8.1_Transformers_dataset.ipynb`笔记本。

## 如何操作...

在这个菜谱中，您将使用数据集包从`HuggingFace`网站加载`RottenTomatoes`数据集。如果数据集不存在，该包会为您下载它。对于任何后续运行，如果之前已下载，它将使用缓存中的下载数据集。

本菜谱执行以下操作：

+   读取**RottenTomatoes**数据集

+   描述数据集的特征

+   从数据集的训练分割中加载数据

+   从数据集中抽取几个句子并打印出来

菜谱的步骤如下：

1.  执行必要的导入，从数据集包导入必要的类型和函数：

    ```py
    from datasets import load_dataset, get_dataset_split_names
    ```

1.  通过**load_dataset**函数加载**"rotten tomatoes"**并打印内部数据集分割。这个数据集包含训练、验证和测试分割：

    ```py
    dataset = load_dataset("rotten_tomatoes")
    print(get_dataset_split_names("rotten_tomatoes"))
    ```

    前一个命令的输出如下：

    ```py
    ['train', 'validation', 'test']
    ```

1.  加载数据集并打印训练分割的属性。**training_data.description**描述了数据集的详细信息，而**training_data.features**描述了数据集的特征。在输出中，我们可以看到**training_data**分割包含特征**text**，它是字符串类型，以及**label**，它是分类类型，具有**neg**和**pos**的值：

    ```py
    training_data = dataset['train']
    print(training_data.description)
    print(training_data.features)
    ```

    命令的输出如下：

    ```py
    Movie Review Dataset.
    This is a dataset of containing 5,331 positive and 5,331 negative processed  sentences from Rotten Tomatoes movie reviews. This data was first used in Bo Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.'', Proceedings of the ACL, 2005.
    {'text': Value(dtype='string', id=None), 
        'label':ClassLabel(names=['neg', 'pos'], id=None)}
    ```

1.  现在我们已经加载了数据集，我们将打印其中的前五个句子。这只是为了确认我们确实能够从数据集中读取：

    ```py
    sentences = training_data['text'][:5]
    [print(sentence) for sentence in sentences]
    ```

    命令的输出如下：

    ```py
    the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .
    the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .
    effective but too-tepid biopic
    if you sometimes like to go to the movies to have fun , wasabi is a good place to start .
    emerges as something rare , an issue movie that's so honest and keenly observed that it doesn't feel like one .
    ```

# 在您的数据集中对文本进行分词

变换器内部包含的组件对其处理的单词没有任何内在知识。相反，分词器只使用它处理的单词的标记标识符。在这个食谱中，我们将学习如何将您的数据集中的文本转换为可以由模型用于下游任务的表示。

## 准备工作

作为这个食谱的一部分，我们将使用来自transformers包的`AutoTokenizer`模块。如果您需要从一个现有的笔记本中工作，可以使用代码网站的`8.2_Basic_Tokenization.ipynb`笔记本。

## 如何做到这一点...

在这个食谱中，您将继续使用之前的`RottenTomatoes`数据集示例，并从中采样几个句子。然后我们将将这些采样句子编码成标记及其相应的表示。

这个食谱做了以下事情：

+   将一些句子加载到内存中

+   实例化一个分词器并对句子进行分词

+   将前一步生成的标记ID转换回标记

食谱的步骤如下：

1.  执行必要的导入以导入来自**transformers**库的必要的**AutoTokenizer**模块：

    ```py
    from transformers import AutoTokenizer
    ```

1.  我们初始化一个包含三个句子的句子数组，我们将使用这个例子。这些句子的长度不同，并且有很好的相同和不同单词的组合。这将使我们能够了解分词表示如何因每个句子而异：

    ```py
    sentences = [
        "The first sentence, which is the longest one in the list.",
        "The second sentence is not that long.",
        "A very short sentence."]
    ```

1.  实例化一个**bert-base-cased**类型的分词器。这个分词器是区分大小写的。这意味着单词star和STAR将会有不同的分词表示：

    ```py
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    ```

1.  在这一步，我们将**sentences**数组中的所有句子进行分词。我们调用分词器构造函数，并将**sentences**数组作为参数传递，然后打印构造函数返回的**tokenized_output**实例。此对象是一个包含三项的字典：

    +   **input_ids**：这些是分配给每个标记的数值标记标识符。

    +   **token_type_ids**：这些ID定义了句子中包含的标记的类型。

    +   **attention_mask**：这些定义了输入中每个标记的注意力值。这个掩码决定了在执行下游任务时哪些标记会被关注。这些值是浮点数，可以从0（无注意力）到1（完全注意力）变化。

        ```py
        tokenized_input = tokenizer(sentences)
        print(tokenized_input)
        {'input_ids': [[101, 1109, 1148, 5650, 117, 1134, 1110, 1103, 6119, 1141, 1107, 1103, 2190, 119, 102],
        [101, 1109, 1248, 5650, 1110, 1136, 1115, 1263, 119, 102],[101, 138, 1304, 1603, 5650, 119, 102]],
        'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]],
        'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}
        ```

1.  在这一步，我们将第一句话的输入ID转换回标记：

    ```py
    tokens = tokenizer.convert_ids_to_tokens(
        tokenized_input["input_ids"][0])
    print(tokens)
    ['[CLS]', 'The', 'first', 'sentence', ',', 'which', 'is', 'the', 'longest', 'one', 'in', 'the', 'list', '.', '[SEP]']
    [101, 1109, 1148, 5650, 117, 1134, 1110, 1103, 6119, 1141, 1107, 1103, 2190, 119, 102]
    ```

    将它们转换为标记返回以下输出：

    ```py
    ['[CLS]', 'The', 'first', 'sentence', ',', 'which', 'is', 'the', 'longest', 'one', 'in', 'the', 'list', '.', '[SEP]']
    ```

    除了原始标记外，分词器还添加了`[CLS]`和`[SEP]`。这些标记是为了训练BERT所执行的训练任务而添加的。

    现在我们已经了解了transformer内部使用的文本的内部表示，让我们学习如何将一段文本分类到不同的类别中。

# 对文本进行分类

在这个菜谱中，我们将使用`RottenTomatoes`数据集并对评论文本进行情感分类。我们将对数据集的测试分割进行分类，并评估分类器对测试分割中真实标签的结果。

## 准备就绪

作为这个菜谱的一部分，我们将使用来自transformers包的pipeline模块。如果你需要从一个现有的笔记本中工作，可以使用代码网站上的`8.3_Classification_And_Evaluation.ipynb`笔记本。

## 如何做到这一点...

在这个菜谱中，你将使用`RottenTomatoes`数据集并从中抽取几个句子。然后我们将对五个句子的一个小子集进行情感分类，并在这个较小的子集上展示结果。然后我们将对数据集的整个测试分割进行推理并评估分类结果。

菜谱执行以下操作：

+   加载**RottenTomatoes**数据集并打印其中的前五句话

+   实例化一个使用在相同数据集上训练的预训练Roberta模型进行情感分析的管道

+   使用管道在整个数据集的测试分割上执行推理（或情感预测）

+   评估推理结果

菜谱的步骤如下：

1.  执行必要的导入以导入所需的包和模块：

    ```py
    from datasets import load_dataset
    from evaluate import evaluator, combine
    from transformers import pipeline
    import torch
    ```

1.  在这一步，我们检查系统中是否存在兼容**Compute Unified Device Architecture**（**CUDA**）的设备（或**Graphics Processing Unit**（**GPU**））。如果存在这样的设备，我们的模型将加载到它上面。如果支持，这将加速模型的训练和推理性能。然而，如果不存在这样的设备，将使用**Central Processing Unit**（**CPU**）。我们还加载了**RottenTomatoes**数据集并从中选择了前五句话。这是为了确保我们确实能够读取数据集中存在的数据：

    ```py
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    sentences = load_dataset(
        "rotten_tomatoes", split="test").select(range(5))
    [print(sentence) for sentence in sentences['text']]
    lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without stickiness .
    consistently clever and suspenseful .
    it's like a " big chill " reunion of the baader-meinhof gang , only these guys are more harmless pranksters than political activists .
    the story gives ample opportunity for large-scale action and suspense , which director shekhar kapur supplies with tremendous skill .
    red dragon " never cuts corners .
    ```

1.  通过管道初始化用于情感分析的管道。管道是一种抽象，它允许我们轻松使用模型或推理任务，而无需编写将它们拼接在一起的代码。我们从**textattack**加载了**roberta-base-rotten-tomatoes**模型，该模型已经在这个数据集上进行了训练。在接下来的段落中，我们使用管道进行情感分析任务，并设置用于此任务的具体模型：

    ```py
    roberta_pipe = pipeline("sentiment-analysis",
        model="textattack/roberta-base-rotten-tomatoes")
    ```

1.  在这一步，我们为在步骤2中选择的句子小集合生成预测。使用管道对象生成预测就像传递一系列句子一样简单。如果你在没有兼容CUDA设备的机器上运行此示例，这一步可能需要一点时间：

    ```py
    predictions = roberta_pipe(sentences['text'])
    ```

1.  在这一步，我们遍历我们的句子并检查句子的预测结果。我们打印出实际和生成的预测结果，以及五个句子的文本。实际标签是从数据集中读取的，而预测是通过管道对象生成的：

    ```py
    for idx, _sentence in enumerate(sentences['text']):
        print(
            f"actual: {sentences['label'][idx]}\n"
            f"predicted: {'1' if predictions[idx]['label'] 
                == 'LABEL_1' else '0'}\n"
            f"sentence: {_sentence}\n\n"
        )
    actual:1
    predicted:1
    sentence:lovingly photographed in the manner of a golden book sprung to life , stuart little 2 manages sweetness largely without stickiness .
    actual:1
    predicted:1
    sentence:consistently clever and suspenseful .
    actual:1
    predicted:0
    sentence:it's like a " big chill " reunion of the baader-meinhof gang , only these guys are more harmless pranksters than political activists .
    actual:1
    predicted:1
    sentence:the story gives ample opportunity for large-scale action and suspense , which director shekhar kapur supplies with tremendous skill .
    actual:1
    predicted:1
    sentence:red dragon " never cuts corners .
    ```

1.  既然我们已经验证了管道及其结果，让我们为整个测试集生成推理，并生成这个特定模型的评估指标。加载**RottenTomatoes**数据集的完整测试分割：

    ```py
    sentences = load_dataset("rotten_tomatoes", split="test")
    ```

1.  在这一步，我们初始化一个评估器对象，它可以用来执行推理并评估分类的结果。它还可以用来展示易于阅读的评估结果摘要：

    ```py
    task_evaluator = evaluator("sentiment-analysis")
    ```

1.  在这一步，我们在**评估器**实例上调用**compute**方法。这触发了使用我们在步骤4中初始化的相同管道实例进行的推理和评估。它返回**准确度**、**精确度**、**召回率**和**f1**的评估指标，以及一些与推理相关的性能指标：

    ```py
    eval_results = task_evaluator.compute(
        model_or_pipeline=roberta_pipe,
        data=sentences,
        metric=combine(["accuracy", "precision", "recall", "f1"]),
        label_mapping={"LABEL_0": 0, "LABEL_1": 1}
    )
    ```

1.  在这一步，我们打印出评估的结果。值得注意的是**精确度**、**召回率**和**f1**值。在这个案例中观察到的**f1**值为**0.88**，这是分类器非常有效率的指标，尽管它总是可以进一步改进：

    ```py
    print(eval_results)
    {'accuracy': 0.88,
    'precision': 0.92,
    'recall': 0.84,
    'f1': 0.88,
    'total_time_in_seconds': 27.23,
    'samples_per_second': 39.146,
    'latency_in_seconds': 0.025}
    ```

在这个菜谱中，我们使用预训练的分类器对一个数据集上的数据进行分类。数据集和模型都是用于情感分析的。有些情况下，我们可以使用在另一类数据上训练的分类器，但仍然可以直接使用。这使我们免去了训练自己的分类器并重新利用现有模型的麻烦。我们将在下一个菜谱中了解这个用例。

# 使用零样本分类器

在这个菜谱中，我们将使用零样本分类器对句子进行分类。有些情况下，我们没有从头开始训练分类器或使用按照我们数据标签训练的模型的奢侈。**零样本分类**可以在这种场景下帮助任何团队快速启动。术语中的“零”意味着分类器没有看到目标数据集用于推理的任何数据（精确到零样本）。

## 准备工作

作为这个菜谱的一部分，我们将使用来自transformers包的管道模块。如果您需要从一个现有的笔记本中工作，可以使用代码网站上的`8.4_Zero_shot_classification.ipynb`笔记本。

## 如何操作...

在这个菜谱中，我们将使用几个句子并将它们进行分类。我们将为这些句子使用我们自己的标签集。我们将使用`facebook/bart-large-mnli`模型来完成这个菜谱。这个模型适合零样本分类的任务。

菜谱执行以下操作：

+   基于零样本分类模型初始化一个管道

+   使用管道将句子分类到用户自定义的标签集中

+   打印分类的结果，包括类别及其相关的概率

菜谱的步骤如下：

1.  执行必要的导入并识别计算设备，如前一个分类菜谱中所述：

    ```py
    from transformers import pipeline
    import torch
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ```

1.  在这一步中，我们使用**facebook/bart-large-mnli**模型初始化一个管道实例。我们选择了这个特定的模型作为我们的示例，但也可以使用其他模型——可在**HuggingFace**网站上找到：

    ```py
    pipeline_instance = pipeline(
        model="facebook/bart-large-mnli")
    ```

1.  使用管道实例将句子分类到给定的一组候选标签中。示例中提供的标签完全是新颖的，并且是由我们定义的。模型没有在具有这些标签的示例上进行训练。分类输出存储在**result**变量中，它是一个字典。这个字典有**'sequence'**、**'labels'**和**'scores'**键。**'sequence'**元素存储传递给分类器的原始句子。**'labels'**元素存储类别的标签，但其顺序与我们传递的参数不同。**'scores'**元素存储类别的概率，并与**'labels'**元素中的相同顺序相对应。这个调用中的最后一个参数是**device**。如果系统中存在兼容CUDA的设备，它将被使用：

    ```py
    result = pipeline_instance(
        "I am so hooked to video games as I cannot get any work done!",
        candidate_labels=["technology", "gaming", "hobby", "art", "computer"], device=device)
    ```

1.  我们打印序列，然后打印每个标签及其相关的概率。请注意，标签的顺序已经从我们在上一步中指定的初始输入中改变。函数调用根据标签概率的降序重新排序标签：

    ```py
    print(result['sequence'])
    for i, label in enumerate(result['labels']):
        print(f"{label}:  {result['scores'][i]:.2f}")
    I am so hooked to video games as I cannot get any work done!
    gaming:  0.85
    hobby:  0.08
    technology:  0.07
    computer:  0.00
    art:  0.00
    ```

1.  我们对不同的句子运行零样本分类，并打印其结果。这次，我们发出一个选择概率最高的类别的结果并打印出来：

    ```py
    result = pipeline_instance(
        "A early morning exercise regimen can drive many diseases away!",
        candidate_labels=["health", "medical", "weather", "geography", "politics"], )
    print(result['sequence'])
    for i, label in enumerate(result['labels']):
        print(f"{label}:  {result['scores'][i]:.2f}")
    print(
        f"The most probable class for the sentence is ** 
        {result['labels'][0]} ** "
        f"with a probability of {result['scores'][0]:.2f}"
    )
    A early morning exercise regimen can drive many diseases away!
    health:  0.91
    medical:  0.07
    weather:  0.01
    geography:  0.01
    politics:  0.00
    The most probable class for the sentence is ** health ** with a probability of 0.91
    ```

到目前为止，我们已经使用了转换器和一些预训练模型来生成标记ID和分类。这些菜谱已经使用了转换器的编码器部分。编码器生成文本的表示，然后由其前面的分类头使用以生成分类标签。然而，转换器还有一个名为解码器的另一个组件。解码器使用给定的文本表示并生成后续文本。在下一个菜谱中，我们将更多地了解解码器。

# 生成文本

在此菜谱中，我们将使用一个 **生成式转换器模型**从给定的种子句子生成文本。一个用于生成文本的模型是 GPT-2 模型，它是原始 **通用转换器**（**GPT**）模型的改进版本。

## 准备工作

作为此菜谱的一部分，我们将使用来自 transformers 包的管道模块。如果您需要从一个现有的笔记本中工作，可以使用代码站点中的 `8.5_Transformer_text_generation.ipynb` 笔记本。

## 如何操作...

在此菜谱中，我们将从一个初始种子句子开始，使用 GPT-2 模型根据给定的种子句子生成文本。我们还将调整某些参数以提高生成文本的质量。

菜谱执行以下操作：

+   它初始化一个起始句子，后续句子将从该句子生成。

+   它初始化一个作为管道一部分的 GPT-2 模型，并使用它来生成五个句子，作为传递给生成方法的参数。

+   它打印了生成的结果。

菜谱的步骤如下：

1.  执行必要的导入并识别计算设备，如前一个分类菜谱中所述：

    ```py
    from transformers import pipeline
    import torch
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ```

1.  根据后续文本生成的种子输入句子初始化。我们的目标是使用 GPT-2 解码器根据生成参数假设性地生成后续文本：

    ```py
    text = "The cat had no business entering the neighbors garage, but"
    ```

1.  在此步骤中，我们使用 **'gpt-2'** 模型初始化一个文本生成管道。该模型基于一个使用大量文本语料库训练的 **大型语言模型**（**LLM**）。此调用中的最后一个参数是 **device**。如果系统中存在兼容CUDA的设备，它将被使用：

    ```py
    generator = pipeline(
        'text-generation', model='gpt2', device=device)
    ```

1.  为种子句子生成后续序列并存储结果。调用中除种子文本之外需要注意的参数如下：

    +   **max_length**：生成句子的最大长度，包括种子句子的长度。

    +   **num_return_sequences**：返回的生成序列的数量。

    +   **num_beams**：此参数控制生成序列的质量。较高的数值通常会导致生成序列的质量提高，但也会减慢生成速度。我们鼓励您根据生成序列的质量要求尝试不同的此参数值。

        ```py
        generated_sentences = generator(
            text,do_sample=True, max_length=30,
            num_return_sequences=5, num_beams=5,
            pad_token_id=50256)
        ```

1.  打印生成的句子：

    ```py
    [print(generated_sentence['generated_text']) 
        for generated_sentence in generated_sentences]
    The cat had no business entering the neighbors garage, but  he was able to get inside.  The cat had been in the neighbor's
    The cat had no business entering the neighbors garage, but  the owner of the house called 911.  He said he found the cat in
    The cat had no business entering the neighbors garage, but  he was able to get his hands on one of the keys.  It was
    The cat had no business entering the neighbors garage, but  he didn't seem to mind at all.  He had no idea what he
    The cat had no business entering the neighbors garage, but  the cat had no business entering the neighbors garage, but the cat had no business entering
    ```

## 更多内容...

如前例所示，生成的输出是基本的、重复的、语法错误的，或者可能是不连贯的。我们可以使用不同的技术来改进生成的输出。

我们这次将使用`no_repeat_ngram_size`参数来生成文本。我们将此参数的值设置为`2`。这指示生成器不要重复二元组。

我们将*步骤4*中的行更改为以下内容：

```py
generated_sentences = generator(text, do_sample=True,
    max_length=30, num_return_sequences=5, num_beams=5,
    no_repeat_ngram_size=2,  pad_token_id=50256)
```

如以下输出所示，句子重复性减少，但其中一些仍然不连贯：

```py
The cat had no business entering the neighbors garage, but  it was too late to stop it.
"I don't know if it was
The cat had no business entering the neighbors garage, but  she was able to find her way to the porch, where she and her friend were
The cat had no business entering the neighbors garage, but  he did get in the way.
The next day, the neighbor called the police
The cat had no business entering the neighbors garage, but  he managed to get his hands on one of the keys, which he used to unlock
The cat had no business entering the neighbors garage, but  the neighbors thought they were in the right place.
"What's going on
```

为了提高连贯性，我们可以使用另一种技术来包含一组单词中最有可能成为下一个单词的下一个单词。我们使用`top_k`参数并将其值设置为`50`。这指示生成器从根据其概率排列的前50个单词中采样下一个单词。

我们将*步骤4*中的行更改为以下内容：

```py
generated_sentences = generator(text, do_sample=True,
    max_length=30, num_return_sequences=5, num_beams=5,
    no_repeat_ngram_size=2, top_k=50, pad_token_id=50256)
The cat had no business entering the neighbors garage, but  it did get into a neighbor's garage. The neighbor went to check on the cat
The cat had no business entering the neighbors garage, but  she was there to take care of it.
The next morning, the cat was
The cat had no business entering the neighbors garage, but  it didn't want to leave. The neighbor told the cat to get out of the
The cat had no business entering the neighbors garage, but  the neighbors were too afraid to call 911.The neighbor told the police that he
The cat had no business entering the neighbors garage, but  it was there that he found his way to the kitchen, where it was discovered that
```

我们还可以将`top_k`参数与`top_p`参数结合使用。这指示生成器从具有高于此定义值的概率的单词集中选择下一个单词。将此参数与值为`0.8`的参数结合使用会产生以下输出：

```py
generated_sentences = generator(text, do_sample=True,
    max_length=30, num_return_sequences=5, num_beams=5,
    no_repeat_ngram_size=2, top_k=50, top_p=0.8,
    pad_token_id=50256)
The cat had no business entering the neighbors garage, but  the owner of the house told the police that he did not know what was going on
The cat had no business entering the neighbors garage, but  he did, and the cat was able to get out of the garage.The
The cat had no business entering the neighbors garage, but  he was able to get in through the back door. The cat was not injured,
The cat had no business entering the neighbors garage, but  the neighbor told the police that the cat was a stray, and the neighbor said that
The cat had no business entering the neighbors garage, but  the owner of the house said he didn't know what to do with the cat.
```

如我们所见，向生成器添加额外的参数继续提高生成的输出质量。

作为最后的例子，让我们通过将*步骤4*中的行更改为以下内容来生成更长的输出序列：

```py
generated_sentences = generator(text, do_sample=True,
    max_length=500, num_return_sequences=1, num_beams=5,
    no_repeat_ngram_size=2, top_k=50, top_p=0.85,
    pad_token_id=50256)
The cat had no business entering the neighbors garage, but  she was there to help.
"I was like, 'Oh my God, she's here,'" she said. "I'm like 'What are you doing here?' "
The neighbor, who asked not to be identified, said she didn't know what to make of the cat's behavior. She said it seemed like it was trying to get into her home, and that she was afraid for her life. The neighbor said that when she went to check on her cat, it ran into the neighbor's garage and hit her in the face, knocking her to the ground.
```

如我们所见，生成的输出，尽管有些虚构，但更加连贯和易读。我们鼓励您尝试不同的参数组合及其相应值，以根据其用例改进生成的输出。

请注意，模型返回的输出可能略有不同于本例所示。这是因为内部语言模型本质上具有概率性。下一个单词是从包含概率大于我们在生成参数中定义的单词的分布中采样的。

在这个示例中，我们使用了transformer的解码器模块来生成文本，给定一个种子句子。在某些用例中，编码器和解码器一起使用来生成文本。我们将在下一个示例中了解这一点。

# 语言翻译

在这个示例中，我们将使用transformers进行语言翻译。我们将使用**Google Text-To-Text Transfer Transformer**（**T5**）模型。这是一个端到端模型，它使用transformer模型的编码器和解码器组件。

## 准备中

作为本示例的一部分，我们将使用transformers包中的pipeline模块。如果您需要从一个现有的笔记本中工作，可以使用代码网站上的`8.6_Language_Translation_with_transformers.ipynb`笔记本。

## 如何做...

在这个食谱中，你将初始化一个英语种子句子并将其翻译成法语。T5模型期望输入格式编码有关语言翻译任务的信息以及种子句子。在这种情况下，编码器使用源语言中的输入并生成文本的表示。解码器使用这个表示并为目标语言生成文本。T5模型专门为此任务以及其他许多任务进行了训练。如果你在没有任何CUDA兼容设备的机器上运行，食谱步骤的执行可能需要一些时间。

该食谱执行以下操作：

+   它初始化了**Google t5-base**模型和标记器

+   它初始化一个英语种子句子，该句子将被翻译成法语

+   它将种子句子以及翻译任务规范进行标记化，以便将种子句子翻译成法语

+   它生成翻译后的标记，将它们解码成目标语言（法语），并打印出来

该食谱的步骤如下：

1.  执行必要的导入并识别计算设备，如前一个分类食谱中所述：

    ```py
    from transformers import (
        T5Tokenizer, T5ForConditionalGeneration)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

1.  使用来自谷歌的**t5-base**模型初始化一个标记器和模型实例。我们使用**model_max_length**参数为**200**。如果你的种子句子超过200个单词，可以自由地尝试更高的值。我们还把模型加载到第1步中确定的用于计算的设备上：

    ```py
    tokenizer = T5Tokenizer.from_pretrained(
        't5-base', model_max_length=200)
    model = T5ForConditionalGeneration.from_pretrained(
        't5-base', return_dict=True)
    model = model.to(device)
    ```

1.  初始化一个你想要翻译的种子序列：

    ```py
    language_sequence = ("It's such a beautiful morning today!")
    ```

1.  标记输入序列。标记器将其源语言和目标语言作为其输入编码的一部分进行指定。这是通过将“**翻译英语到法语：**”文本附加到输入种子句子中实现的。我们将这些标记ID加载到用于计算的设备上。模型和标记ID必须在同一设备上，这是两者的要求：

    ```py
    input_ids = tokenizer(
        "translate English to French: " + language_sequence,
        return_tensors="pt",
        truncation=True).input_ids.to(device)
    ```

1.  通过模型将源语言标记ID转换为目标语言标记ID。该模型使用编码器-解码器架构将输入标记ID转换为输出标记ID：

    ```py
    language_ids = model.generate(input_ids, max_new_tokens=200)
    ```

1.  将文本从标记ID解码成目标语言标记。我们使用标记器将输出标记ID转换为目标语言标记：

    ```py
    language_translation = tokenizer.decode(
        language_ids[0], skip_special_tokens=True)
    ```

1.  打印翻译后的输出：

    ```py
    print(language_translation)
    C'est un matin si beau!
    ```

总之，本章介绍了transformers的概念，以及一些基本应用。下一章将重点介绍我们如何使用不同的NLP技术更好地理解文本。
