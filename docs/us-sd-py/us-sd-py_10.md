

# 第十章：克服 77 个标记限制并启用提示加权

从 [*第 5 章*](B21263_05.xhtml#_idTextAnchor097)，我们知道稳定扩散利用 OpenAI 的 CLIP 模型作为其文本编码器。根据源代码 [6]，CLIP 模型的标记化实现具有 77 个标记的上下文长度。

CLIP 模型中的这个 77 个标记限制扩展到 Hugging Face Diffusers，限制了最大输入提示为 77 个标记。不幸的是，由于这个限制，无法在不进行一些修改的情况下在这些输入提示中分配关键词权重。

例如，假设你给出一个产生超过 77 个标记的提示字符串，如下所示：

```py
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/deliberate-v2",
    torch_dtype=torch.float16).to("cuda")
prompt = "a photo of a cat and a dog driving an aircraft "*20
image = pipe(prompt = prompt).images[0]
image
```

Diffusers 将显示一个警告消息，如下所示：

```py
The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens…
```

你不能通过提供权重来突出显示猫，如下所示：

```py
a photo (cat:1.5) and a dog driving an aircraft
```

默认情况下，`Diffusers` 包不包括克服 77 个标记限制或为单个标记分配权重的功能，正如其文档所述。这是因为 Diffusers 旨在作为一个通用的工具箱，提供可以在各种项目中使用的必要功能。

尽管如此，通过使用 Diffusers 提供的核心功能，我们可以开发一个自定义提示解析器。这个解析器将帮助我们绕过 77 个标记的限制并为每个标记分配权重。在本章中，我们将深入探讨文本嵌入的结构，并概述一种方法来超越 77 个标记的限制，同时为每个标记分配权重值。

在本章中，我们将涵盖以下内容：

+   理解 77 个标记限制

+   克服 77 个标记限制

+   启用带权重的长提示

+   使用社区管道克服 77 个标记限制

如果你想要开始使用支持长提示加权的完整功能管道，请参阅 *使用社区 *pipelines* 部分克服 77 个标记限制。

到本章结束时，你将能够使用无大小限制的加权提示，并了解如何使用 Python 实现它们。

# 理解 77 个标记限制

稳定扩散（v1.5）文本编码器使用 OpenAI 的 CLIP 编码器 [`2`]。CLIP 文本编码器有一个 77 个标记的限制，这个限制传播到下游的稳定扩散。我们可以通过以下步骤重现 77 个标记的限制：

1.  我们可以从稳定扩散中取出编码器并验证它。假设我们有提示 `一张猫和狗驾驶飞机的照片` 并将其乘以 20 以使提示的标记大小超过 77：

    ```py
    prompt = "a photo of a cat and a dog driving an aircraft "*20
    ```

1.  重新使用本章开头初始化的管道，并取出 `tokenizer` 和 `text_encoder`：

    ```py
    tokenizer = pipe.tokenizer
    
    text_encoder = pipe.text_encoder
    ```

1.  使用 `tokenizer` 从提示中获取标记 ID：

    ```py
    tokens = tokenizer(
    
        prompt,
    
        truncation = False,
    
        return_tensors = 'pt'
    
    )["input_ids"]
    
    print(len(tokens[0]))
    ```

1.  由于我们设置了 `truncation = False`，`tokenizer` 将将任何长度的字符串转换为标记 ID。前面的代码将输出一个长度为 181 的标记列表。`return_tensors = 'pt'` 将告诉函数以 `[1,181]` 张量对象的形式返回结果。

    尝试将标记 ID 编码为 `embeddings`：

    ```py
    embeddings = pipe.text_encoder(tokens.to("cuda"))[0]
    ```

    我们将看到一个`RuntimeError`错误消息，内容如下：

    ```py
    RuntimeError: The size of tensor a (181) must match the size of tensor b (77) at non-singleton dimension 1
    ```

    从前面的步骤中，我们可以看到CLIP的文本编码器一次只能接受77个标记。

1.  现在，让我们看看第一个和最后一个标记。如果我们去掉`*20`，只对提示`a photo cat and dog driving an aircraft`进行分词，当我们打印出标记ID时，我们将看到10个标记ID而不是8个：

    ```py
    tensor([49406,   320,  1125,  2368,   537,  1929,  4161,   550,  7706, 49407])
    ```

1.  在前面的标记ID中，第一个（`49406`）和最后一个（`49407`）是自动添加的。我们可以使用`tokenizer._convert_id_to_token`将标记ID转换为字符串：

    ```py
    print(tokenizer._convert_id_to_token(49406))
    
    print(tokenizer._convert_id_to_token(49407))
    ```

    我们可以看到两个额外的标记被添加到提示中：

    ```py
    <|startoftext|>
    
    <|endoftext|>
    ```

为什么我们需要检查这个？因为当我们连接标记时，我们需要移除自动添加的开始和结束标记。接下来，让我们继续进行克服77个标记限制的步骤。

# 克服77个标记的限制

幸运的是，Stable Diffusion UNet不强制执行这个77个标记的限制。如果我们能够分批获取嵌入，将那些分块的嵌入连接成一个张量，并将其提供给UNet，我们应该能够克服77个标记的限制。以下是这个过程的大致概述：

1.  从Stable Diffusion管道中提取文本分词器和文本编码器。

1.  不论其大小如何，对输入提示进行分词。

1.  消除添加的开始和结束标记。

1.  提取前77个标记并将它们编码成嵌入。

1.  将嵌入堆叠成一个大小为`[1, x, 768]`的张量。

现在，让我们使用Python代码来实现这个想法：

1.  提取文本分词器和文本编码器：

    ```py
    # step 1\. take out the tokenizer and text encoder
    
    tokenizer = pipe.tokenizer
    
    text_encoder = pipe.text_encoder
    ```

    我们可以重用Stable Diffusion管道中的分词器和文本编码器。

1.  分词任何大小的输入提示：

    ```py
    # step 2\. encode whatever size prompt to tokens by setting 
    
    # truncation = False.
    
    tokens = tokenizer(
    
        prompt,
    
        truncation = False
    
    )["input_ids"]
    
    print("token length:", len(tokens))
    
    # step 2.2\. encode whatever size neg_prompt, 
    
    # padding it to the size of prompt.
    
    negative_ids = pipe.tokenizer(
    
        neg_prompt,
    
        truncation    = False,
    
        padding       = "max_length",
    
        max_length    = len(tokens)
    
    ).input_ids
    
    print("neg_token length:", len(negative_ids))
    ```

    在前面的代码中，我们做了以下操作：

    +   我们将`truncation = False`设置为允许分词超过默认的77个标记限制。这确保了无论提示的大小如何，整个提示都会被分词。

    +   标记作为Python列表返回，而不是torch张量。Python列表中的标记将使我们更容易添加额外的元素。请注意，在提供给文本编码器之前，标记列表将被转换为torch张量。

    +   有两个额外的参数，`padding = "max_length"`和`max_length = len(tokens)`。我们使用这些参数确保提示标记和负提示标记的大小相同。

1.  移除开始和结束标记。

    分词器将自动添加两个额外的标记：开始标记（`49406`）和结束标记（`49407`）。

    在后续步骤中，我们将分割标记序列并将分块标记输入到文本编码器中。每个块将有自己的开始和结束标记。但在那之前，我们需要从原始的长标记列表中最初排除它们：

    ```py
    tokens = tokens[1:-1]
    
    negative_ids = negative_ids[1:-1]
    ```

    然后将这些开始和结束标记添加回分块标记中，每个块的大小为`75`。我们将在第4步将开始和结束标记添加回去。

1.  将77个大小的分块标记编码成嵌入：

    ```py
    # step 4\. Pop out the head 77 tokens, 
    
    # and encode the 77 tokens to embeddings.
    
    embeds,neg_embeds = [],[]
    
    chunk_size = 75
    
    bos = pipe.tokenizer.bos_token_id
    
    eos = pipe.tokenizer.eos_token_id
    
    for i in range(0, len(tokens), chunk_size):
    
    # Add the beginning and end token to the 75 chunked tokens to 
    
    # make a 77-token list
    
        sub_tokens = [bos] + tokens[i:i + chunk_size] + [eos]
    
    # text_encoder support torch.Size([1,x]) input tensor
    
    # that is why use [sub_tokens], 
    
    # instead of simply give sub_tokens.
    
        tensor_tokens = torch.tensor(
    
            [sub_tokens],
    
            dtype = torch.long,
    
            device = pipe.device
    
        )
    
        chunk_embeds = text_encoder(tensor_tokens)[0]
    
        embeds.append(chunk_embeds)
    
    # Add the begin and end token to the 75 chunked neg tokens to 
    
    # make a 77 token list
    
        sub_neg_tokens = [bos] + negative_ids[i:i + chunk_size] + \
    
            [eos]
    
        tensor_neg_tokens = torch.tensor(
    
            [sub_neg_tokens],
    
            dtype = torch.long,
    
            device = pipe.device
    
        )
    
        neg_chunk_embeds= text_encoder(tensor_neg_tokens)[0]
    
        neg_embeds.append(neg_chunk_embeds)
    ```

    前面的代码通过token列表循环，每次取出75个token。然后，它将起始和结束token添加到75个token的列表中，以创建一个77个token的列表。为什么是77个token？因为文本编码器一次可以编码77个token到嵌入中。

    在 `for` 循环内部，第一部分处理提示嵌入，第二部分处理负嵌入。尽管我们提供了一个空的负提示，为了启用无分类指导扩散，我们仍然需要一个与正提示嵌入大小相同的负嵌入列表（在去噪循环中，条件潜在将减去由无提示生成的无条件潜在）。

1.  将嵌入堆叠到 `[1,x,768]` 大小的torch张量。

    在这一步之前，`embeds` 列表包含如下数据：

    ```py
    [tensor1, tensor2...]
    ```

    Stable Diffusion流水线的嵌入参数接受大小为`torch.Size([1,x,768])`的张量。

    我们仍然需要使用这两行代码将这些列表转换为三维张量：

    ```py
    # step 5\. Stack the embeddings to a [1,x,768] size torch tensor.
    
    prompt_embeds = torch.cat(embeds, dim = 1)
    
    prompt_neg_embeds = torch.cat(neg_embeds, dim = 1)
    ```

    在前面的代码中，我们有以下内容：

    +   `embeds` 和 `neg_embeds` 是PyTorch张量的列表。`torch.cat()` 函数用于沿着由 `dim` 指定的维度连接这些张量。在这种情况下，我们有 `dim=1`，这意味着张量是在它们的第二个维度上连接的（因为Python使用0基于索引）。

    +   `prompt_embeds` 是一个包含`embeds`中所有嵌入的张量。同样，`prompt_neg_embeds` 包含`neg_embeds`中所有嵌入的张量。

到目前为止，我们已经有一个可以转换任何长度提示到嵌入的文本编码器，这些嵌入可以被Stable Diffusion流水线使用。接下来，让我们将所有代码放在一起。

## 将所有代码组合到一个函数中

让我们更进一步，将所有之前的代码放入一个打包的函数中：

```py
def long_prompt_encoding(
    pipe:StableDiffusionPipeline,
    prompt,
    neg_prompt = ""
):
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    chunk_size = 75
    # step 1\. take out the tokenizer and text encoder
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    # step 2.1\. encode whatever size prompt to tokens by setting 
    # truncation = False.
    tokens = tokenizer(
        prompt.
        truncation = False,
        # return_tensors = 'pt'
    )["input_ids"]
    # step 2.2\. encode whatever size neg_prompt, 
    # padding it to the size of prompt.
    negative_ids = pipe.tokenizer(
        neg_prompt,
        truncation = False,
        # return_tensors = "pt",
        Padding = "max_length",
        max_length = len(tokens)
    ).input_ids
    # Step 3\. remove begin and end tokens
    tokens = tokens[1:-1]
    negative_ids = negative_ids[1:-1]
    # step 4\. Pop out the head 77 tokens, 
    # and encode the 77 tokens to embeddings.
    embeds,neg_embeds = [],[]
    for i in range(0, len(tokens), chunk_size):
# Add the beginning and end tokens to the 75 chunked tokens to make a 
# 77-token list
        sub_tokens = [bos] + tokens[i:i + chunk_size] + [eos]
# text_encoder support torch.Size([1,x]) input tensor
# that is why use [sub_tokens], instead of simply give sub_tokens.
        tensor_tokens = torch.tensor(
            [sub_tokens],
            dtype = torch.long,
            device = pipe.device
        )
        chunk_embeds = text_encoder(tensor_tokens)[0]
        embeds.append(chunk_embeds)
# Add beginning and end token to the 75 chunked neg tokens to make a 
# 77-token list
        sub_neg_tokens = [bos] + negative_ids[i:i + chunk_size] + \
            [eos]
        tensor_neg_tokens = torch.tensor(
            [sub_neg_tokens],
            dtype = torch.long,
            device = pipe.device
        )
        neg_chunk_embeds = text_encoder(tensor_neg_tokens)[0]
        neg_embeds.append(neg_chunk_embeds)
# step 5\. Stack the embeddings to a [1,x,768] size torch tensor.
    prompt_embeds = torch.cat(embeds, dim = 1)
    prompt_neg_embeds = torch.cat(neg_embeds, dim = 1)
    return prompt_embeds, prompt_neg_embeds
```

让我们创建一个长提示来测试前面的函数是否工作：

```py
prompt = "photo, cute cat running on the grass" * 10 #<- long prompt
prompt_embeds, prompt_neg_embeds = long_prompt_encoding(
    pipe, prompt, neg_prompt="low resolution, bad anatomy"
)
print(prompt_embeds.shape)
image = pipe(
    prompt_embeds = prompt_embeds,
    negative_prompt_embeds = prompt_neg_embeds,
    generator = torch.Generator("cuda").manual_seed(1)
).images[0]
image
```

结果如*图10.1*所示：

![图10.1：可爱的小猫在草地上奔跑，使用了长提示](img/B21263_10_01.jpg)

图10.1：可爱的小猫在草地上奔跑，使用了长提示

如果我们的新函数对长提示有效，生成的图像应该反映附加的提示信息。让我们将提示信息扩展到以下内容：

```py
prompt = "photo, cute cat running on the grass" * 10
prompt = prompt + ",pure white cat" * 10
```

新的提示信息将生成如*图10.2*所示的图像：

![图10.2：可爱的小猫在草地上奔跑，附加了纯白色猫的提示](img/B21263_10_02.jpg)

图10.2：可爱的小猫在草地上奔跑，附加了纯白色猫的提示

如您所见，新添加的提示信息工作正常，并且为猫添加了更多白色元素；然而，它仍然不是提示信息中要求的纯白色。我们将通过提示权重来解决此问题，我们将在下一节中介绍。

# 启用带权重的长提示

我们刚刚为基于Stable Diffusion管道（v1.5版）构建了任意大小的文本编码器。所有这些步骤都是为构建带有加权文本编码器的长提示铺路。

加权Stable Diffusion提示是指为用于通过Stable Diffusion算法生成图像的文本提示中的特定单词或短语分配不同的重要性级别。通过调整这些权重，我们可以控制某些概念对生成输出的影响程度，从而实现图像的更大定制和细化。

该过程通常涉及放大或缩小与提示中每个概念相关的文本嵌入向量。例如，如果您想使Stable Diffusion模型强调某个特定主题，同时降低另一个主题的强调，您将增加前者的权重并减少后者的权重。加权提示使我们能够更好地引导图像生成，以达到期望的结果。

为提示添加权重的核心仅仅是向量乘法：

加权 _ 嵌入 = [embedding1, embedding2, ..., embedding768] × 权重

在此之前，我们仍需要进行一些准备工作以创建加权提示嵌入，如下所示：

1.  将`a (white) cat`转换成如下列表：`[['a ', 1.0], ['white', 1.1], ['cat', 1.0]]`。我们将采用在Automatic1111 **Stable Diffusion** (**SD**) WebUI中广泛使用的提示格式，如开源SD WebUI[4]中定义的那样。

1.  **标记和权重提取**：将标记ID及其对应的权重分别放入两个不同的列表中。

1.  **提示和负提示填充**：确保提示和负提示标记具有相同的最大长度。如果提示比负提示长，则将负提示填充到与提示相同的长度。否则，将提示填充以与负提示的长度对齐。

关于注意力和强调（权重），我们将实现以下权重格式[4]：

```py
a (word) - increase attention to word by a factor of 1.1
a ((word)) - increase attention to word by a factor of 1.21 (= 1.1 * 1.1)
a [word] - decrease attention to word by a factor of 1.1
a (word:1.5) - increase attention to word by a factor of 1.5
a (word:0.25) - decrease attention to word by a factor of 4 (= 1 / 0.25)
a \(word\) - use literal () characters in prompt
```

让我们更详细地了解这些步骤：

1.  构建名为`parse_prompt_attention`的函数。

    为了确保提示格式与Automatic1111的SD WebUI完全兼容，我们将从开源的`parse_prompt_attention`函数[3]中提取并重用该函数：

    ```py
    def parse_prompt_attention(text):
    
        import re
    
        re_attention = re.compile(
    
            r"""
    
                \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
    
                \)|]|[^\\()\[\]:]+|:
    
            """
    
            , re.X
    
        )
    
        re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)
    
        res = []
    
        round_brackets = []
    
        square_brackets = []
    
        round_bracket_multiplier = 1.1
    
        square_bracket_multiplier = 1 / 1.1
    
        def multiply_range(start_position, multiplier):
    
            for p in range(start_position, len(res)):
    
                res[p][1] *= multiplier
    
        for m in re_attention.finditer(text):
    
            text = m.group(0)
    
            weight = m.group(1)
    
            if text.startswith('\\'):
    
                res.append([text[1:], 1.0])
    
            elif text == '(':
    
                round_brackets.append(len(res))
    
            elif text == '[':
    
                square_brackets.append(len(res))
    
            elif weight is not None and len(round_brackets) > 0:
    
                multiply_range(round_brackets.pop(), float(weight))
    
            elif text == ')' and len(round_brackets) > 0:
    
                multiply_range(round_brackets.pop(), \
    
                    round_bracket_multiplier)
    
            elif text == ']' and len(square_brackets) > 0:
    
                multiply_range(square_brackets.pop(), \
    
                    square_bracket_multiplier)
    
            else:
    
                parts = re.split(re_break, text)
    
                for i, part in enumerate(parts):
    
                    if i > 0:
    
                        res.append(["BREAK", -1])
    
                    res.append([part, 1.0])
    
        for pos in round_brackets:
    
            multiply_range(pos, round_bracket_multiplier)
    
        for pos in square_brackets:
    
            multiply_range(pos, square_bracket_multiplier)
    
        if len(res) == 0:
    
            res = [["", 1.0]]
    
        # merge runs of identical weights
    
        i = 0
    
        while i + 1 < len(res):
    
            if res[i][1] == res[i + 1][1]:
    
                res[i][0] += res[i + 1][0]
    
                res.pop(i + 1)
    
            else:
    
                i += 1
    
        return res
    ```

    使用以下方式调用先前创建的函数：

    ```py
    parse_prompt_attention("a (white) cat")
    ```

    这将返回以下内容：

    ```py
    [['a ', 1.0], ['white', 1.1], [' cat', 1.0]]
    ```

1.  获取带有权重的提示。

    在前述函数的帮助下，我们可以得到一组提示和权重对的列表。文本编码器将仅对提示中的标记进行编码（不需要将权重作为输入提供给文本编码器）。我们需要进一步处理提示-权重对，将其转换为两个大小相同的独立列表，一个用于标记ID，一个用于权重，如下所示：

    ```py
    tokens: [1,2,3...]
    
    weights: [1.0, 1.0, 1.0...]
    ```

    这可以通过以下函数来完成：

    ```py
    # step 2\. get prompts with weights
    
    # this function works for both prompt and negative prompt
    
    def get_prompts_tokens_with_weights(
    
        pipe: StableDiffusionPipeline,
    
        prompt: str
    
    ):
    
        texts_and_weights = parse_prompt_attention(prompt)
    
        text_tokens,text_weights = [],[]
    
        for word, weight in texts_and_weights:
    
            # tokenize and discard the starting and the ending token
    
            token = pipe.tokenizer(
    
                word,
    
                # so that tokenize whatever length prompt
    
                truncation = False
    
            ).input_ids[1:-1]
    
            # the returned token is a 1d list: [320, 1125, 539, 320]
    
            # use merge the new tokens to the all tokens holder: 
    
            # text_tokens
    
            text_tokens = [*text_tokens,*token]
    
            # each token chunk will come with one weight, like ['red 
    
            # cat', 2.0]
    
            # need to expand the weight for each token.
    
            chunk_weights = [weight] * len(token)
    
            # append the weight back to the weight holder: text_
    
            # weights
    
            text_weights = [*text_weights, *chunk_weights]
    
        return text_tokens,text_weights
    ```

    前述函数接受两个参数：SD管道和提示字符串。输入字符串可以是正提示或负提示。

    在函数体内，我们首先调用`parse_prompt_attention`函数，以最小的粒度（权重应用于单个标记级别）关联带有权重的提示。然后，我们遍历列表，对文本进行标记化，并使用索引操作`[1:-1]`移除标记器添加的开始和结束标记ID。

    将新的标记ID合并回包含所有标记ID的列表。同时，扩展权重数量并将其合并回包含所有权重数字的列表。

    让我们重用“一只（白色）猫”的提示并调用该函数：

    ```py
    prompt = "a (white) cat"
    
    tokens, weights = get_prompts_tokens_with_weights(pipe, prompt)
    
    print(tokens,weights)
    ```

    前面的代码将返回以下内容：

    ```py
    [320, 1579, 2368] [1.0, 1.1, 1.0]
    ```

    注意到“白色”的第二个标记ID现在权重为`1.1`而不是`1.0`。

1.  填充标记。

    在这一步，我们将进一步将标记ID列表及其权重转换为分块列表。

    假设我们有一个包含超过77个元素的标记ID列表：

    `[``1,2,3,...,100]`

    我们需要将其转换为包含分块的列表，每个块包含最多77个（最大）标记：

    `[[``49406,1,2...75,49407],[49406,76,77,...,100,49407]]`

    这样做是为了在下一步中，我们可以遍历列表的外层，并逐个编码77个标记的列表。

    现在，你可能想知道为什么我们需要一次向文本编码器提供最多77个标记。如果我们简单地循环每个元素并逐个编码一个标记会怎样？这是一个好问题，但我们不能这样做，因为单独编码“白色”然后编码“猫”将产生与一次一起编码“白色猫”不同的嵌入。

    我们可以通过快速测试来找出差异。首先，让我们只编码“白色”：

    ```py
    # encode "white" only
    
    white_token = 1579
    
    white_token_tensor = torch.tensor(
    
        [[white_token]],
    
        dtype = torch.long,
    
        device = pipe.device
    
    )
    
    white_embed = pipe.text_encoder(white_token_tensor)[0]
    
    print(white_embed[0][0])
    ```

    然后，一起编码“白色”和“猫”：

    ```py
    # encode "white cat"
    
    white_token, cat_token = 1579, 2369
    
    white_cat_token_tensor = torch.tensor(
    
        [[white_token, cat_token]],
    
        dtype = torch.long,
    
        device = pipe.device
    
    )
    
    white_cat_embeds = pipe.text_encoder(white_cat_token_tensor)[0]
    
    print(white_cat_embeds[0][0])
    ```

    尝试运行前面的代码；你会发现相同的“白色”会导致不同的嵌入。根本原因是什么？标记和嵌入不是一对一的映射；嵌入是基于自注意力机制[5]生成的。单个“白色”可以代表颜色或姓氏，而“白色猫”中的“白色”显然是在说这是一个描述猫的颜色。

    让我们回到填充工作。以下代码将检查标记列表的长度。如果标记ID列表长度大于75，则取前75个标记并循环此操作，剩余的标记少于75个，将由单独的逻辑处理：

    ```py
    # step 3\. padding tokens
    
    def pad_tokens_and_weights(
    
        token_ids: list,
    
        weights: list
    
    ):
    
        bos,eos = 49406,49407
    
        # this will be a 2d list
    
        new_token_ids = []
    
        new_weights   = []
    
        while len(token_ids) >= 75:
    
            # get the first 75 tokens
    
            head_75_tokens = [token_ids.pop(0) for _ in range(75)]
    
            head_75_weights = [weights.pop(0) for _ in range(75)]
    
            # extract token ids and weights
    
            temp_77_token_ids = [bos] + head_75_tokens + [eos]
    
            temp_77_weights   = [1.0] + head_75_weights + [1.0]
    
            # add 77 tokens and weights chunks to the holder list
    
            new_token_ids.append(temp_77_token_ids)
    
            new_weights.append(temp_77_weights)
    
        # padding the left
    
        if len(token_ids) > 0:
    
            padding_len = 75 - len(token_ids)
    
            padding_len = 0
    
            temp_77_token_ids = [bos] + token_ids + [eos] * \
    
                padding_len + [eos]
    
            new_token_ids.append(temp_77_token_ids)
    
            temp_77_weights = [1.0] + weights   + [1.0] * \
    
                padding_len + [1.0]
    
            new_weights.append(temp_77_weights)
    
        # return
    
        return new_token_ids, new_weights
    ```

    接下来，使用以下函数：

    ```py
    t,w = pad_tokens_and_weights(tokens.copy(), weights.copy())
    
    print(t)
    
    print(w)
    ```

    前面的函数接受以下先前生成的`tokens`和`weights`列表：

    ```py
    [320, 1579, 2368] [1.0, 1.1, 1.0]
    ```

    它将其转换为以下形式：

    ```py
    [[49406, 320, 1579, 2368, 49407]]
    
    [[1.0, 1.0, 1.1, 1.0, 1.0]]
    ```

1.  获取加权嵌入。

    这是最后一步，我们将得到没有标记大小限制的与Automatic1111兼容的嵌入：

    ```py
    def get_weighted_text_embeddings(
    
        pipe: StableDiffusionPipeline,
    
        prompt : str      = "",
    
        neg_prompt: str   = ""
    
    ):
    
        eos = pipe.tokenizer.eos_token_id
    
        prompt_tokens, prompt_weights = \ 
    
            get_prompts_tokens_with_weights(
    
            pipe, prompt
    
        )
    
        neg_prompt_tokens, neg_prompt_weights = \
    
            get_prompts_tokens_with_weights(pipe, neg_prompt)
    
        # padding the shorter one
    
        prompt_token_len        = len(prompt_tokens)
    
        neg_prompt_token_len    = len(neg_prompt_tokens)
    
        if prompt_token_len > neg_prompt_token_len:
    
            # padding the neg_prompt with eos token
    
            neg_prompt_tokens   = (
    
                neg_prompt_tokens  + \
    
                [eos] * abs(prompt_token_len - neg_prompt_token_len)
    
            )
    
            neg_prompt_weights  = (
    
                neg_prompt_weights +
    
                [1.0] * abs(prompt_token_len - neg_prompt_token_len)
    
            )
    
        else:
    
            # padding the prompt
    
            prompt_tokens       = (
    
                prompt_tokens \
    
                + [eos] * abs(prompt_token_len - \
    
                neg_prompt_token_len)
    
            )
    
            prompt_weights      = (
    
                prompt_weights \
    
                + [1.0] * abs(prompt_token_len - \
    
                neg_prompt_token_len)
    
            )
    
        embeds = []
    
        neg_embeds = []
    
        prompt_token_groups ,prompt_weight_groups = \
    
            pad_tokens_and_weights(
    
                prompt_tokens.copy(),
    
                prompt_weights.copy()
    
        )
    
        neg_prompt_token_groups, neg_prompt_weight_groups = \
    
            pad_tokens_and_weights(
    
                neg_prompt_tokens.copy(),
    
                neg_prompt_weights.copy()
    
            )
    
        # get prompt embeddings one by one is not working.
    
        for i in range(len(prompt_token_groups)):
    
            # get positive prompt embeddings with weights
    
            token_tensor = torch.tensor(
    
                [prompt_token_groups[i]],
    
                dtype = torch.long, device = pipe.device
    
            )
    
            weight_tensor = torch.tensor(
    
                prompt_weight_groups[i],
    
                dtype     = torch.float16,
    
                device    = pipe.device
    
            )
    
            token_embedding = \
    
                pipe.text_encoder(token_tensor)[0].squeeze(0)
    
            for j in range(len(weight_tensor)):
    
                token_embedding[j] = token_embedding[j] * 
    
                    weight_tensor[j]
    
            token_embedding = token_embedding.unsqueeze(0)
    
            embeds.append(token_embedding)
    
            # get negative prompt embeddings with weights
    
            neg_token_tensor = torch.tensor(
    
                [neg_prompt_token_groups[i]],
    
                dtype = torch.long, device = pipe.device
    
            )
    
            neg_weight_tensor = torch.tensor(
    
                neg_prompt_weight_groups[i],
    
                dtype     = torch.float16,
    
                device    = pipe.device
    
            )
    
            neg_token_embedding = \
    
                pipe.text_encoder(neg_token_tensor)[0].squeeze(0)
    
            for z in range(len(neg_weight_tensor)):
    
                neg_token_embedding[z] = (
    
                    neg_token_embedding[z] * neg_weight_tensor[z]
    
                )
    
            neg_token_embedding = neg_token_embedding.unsqueeze(0)
    
            neg_embeds.append(neg_token_embedding)
    
        prompt_embeds       = torch.cat(embeds, dim = 1)
    
        neg_prompt_embeds   = torch.cat(neg_embeds, dim = 1)
    
        return prompt_embeds, neg_prompt_embeds
    ```

    函数看起来有点长，但逻辑很简单。让我分段解释：

    +   在*填充较短的提示*部分，逻辑会将较短的提示填充到结束标记（`eos`），这样提示和负提示标记列表就具有相同的大小（这样生成的潜在变量可以进行减法操作）。

    +   我们调用`pad_tokens_and_weights`函数将所有标记和权重分割成块，每个块包含77个元素。

    +   我们遍历块列表，并在一步中将77个标记编码为嵌入。

    +   我们使用`token_embedding = pipe.text_encoder(token_tensor)[0].squeeze(0)`来移除空维度，这样我们就可以将每个元素与其权重相乘。注意，现在，每个标记都由一个768个元素的向量表示。

    +   最后，我们退出循环，并使用`prompt_embeds = torch.cat(embeds, dim = 1)`将张量列表堆叠成一个更高维度的张量。

# 验证工作

在编写了不那么多的代码之后，我们终于准备好了所有逻辑，现在让我们测试一下代码。

在*长提示编码器*的简单版本中，我们仍然得到一只猫，身体上有一些图案，而不是我们在提示中给出的`纯白色`。现在，让我们给`white`关键词添加权重，看看会发生什么：

```py
prompt = "photo, cute cat running on the grass" * 10
prompt = prompt + ",pure (white:1.5) cat" * 10
neg_prompt = "low resolution, bad anatomy"
prompt_embeds, prompt_neg_embeds = get_weighted_text_embeddings(
    pipe, prompt = prompt, neg_prompt = neg_prompt
)
image = pipe(
    prompt_embeds = prompt_embeds,
    negative_prompt_embeds = prompt_neg_embeds,
    generator = torch.Generator("cuda").manual_seed(1)
).images[0]
image
```

我们新的嵌入函数神奇地使我们能够生成一只纯白色的猫，因为我们给“白色”关键词赋予了`1.5`的权重。

![图10.3：一只可爱的纯白色猫在草地上奔跑，对“白色”一词的权重为1.5](img/B21263_10_03.jpg)

图10.3：一只可爱的纯白色猫在草地上奔跑，对“白色”一词的权重为1.5

就这些！现在，我们可以重用或扩展这个函数来构建我们想要的任何自定义提示解析器。但如果你不想自己构建函数来实现，有没有办法开始使用无限加权提示？是的，接下来我们将介绍两个由开源社区贡献并集成到Diffusers中的管道。

# 使用社区管道克服77个标记的限制

从零开始实现支持长提示加权的管道可能具有挑战性。通常，我们只是希望利用Diffusers使用详细和细微的提示来生成图片。幸运的是，开源社区已经为SD v1.5和SDXL提供了实现。SDXL的实现最初由本书的作者Andrew Zhu初始化，并由社区大幅改进。

我现在将提供两个示例，说明如何使用社区管道来处理SD v1.5和SDXL：

1.  这个例子使用了SD v1.5的`lpw_stable_diffusion`管道。

    使用以下代码启动一个长提示加权管道：

    ```py
    from diffusers import DiffusionPipeline
    
    import torch
    
    model_id_or_path = "stablediffusionapi/deliberate-v2"
    
    pipe = DiffusionPipeline.from_pretrained(
    
        model_id_or_path,
    
        torch_dtype = torch.float16,
    
        custom_pipeline = "lpw_stable_diffusion"
    
    ).to("cuda:0")
    ```

    在前面的代码中，`custom_pipeline = "lpw_stable_diffusion"`实际上会从Hugging Face服务器下载`lpw_stable_diffusion`文件，并在`DiffusionPipeline`管道内部调用。

1.  让我们使用这个管道生成一张图片：

    ```py
    prompt = "photo, cute cat running on the grass" * 10
    
    prompt = prompt + ",pure (white:1.5) cat" * 10
    
    neg_prompt = "low resolution, bad anatomy"
    
    image = pipe(
    
        prompt = prompt,
    
        negative_prompt = neg_prompt,
    
        generator = torch.Generator("cuda").manual_seed(1)
    
    ).images[0]
    
    image
    ```

    你将看到与*图10.3*相同的图片。

1.  现在让我们通过使用`lpw_stable_diffusion`管道来为SDXL举一个例子。

    使用方法几乎与我们在 SD v1.5 中使用的方法相同。唯一的区别是我们正在加载一个 SDXL 模型，并且我们使用了一个不同的自定义管道名称：`lpw_stable_diffusion_xl`。请看以下代码：

    ```py
    from diffusers import DiffusionPipeline
    
    import torch
    
    model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    
    pipe = DiffusionPipeline.from_pretrained(
    
        model_id_or_path,
    
        torch_dtype = torch.float16,
    
        custom_pipeline = "lpw_stable_diffusion_xl",
    
    ).to("cuda:0")
    ```

    图像生成代码与我们用于 SD v1.5 的代码完全相同：

    ```py
    prompt = "photo, cute cat running on the grass" * 10
    
    prompt = prompt + ",pure (white:1.5) cat" * 10
    
    neg_prompt = "low resolution, bad anatomy"
    
    image = pipe(
    
        prompt = prompt,
    
        negative_prompt = neg_prompt,
    
        generator = torch.Generator("cuda").manual_seed(7)
    
    ).images[0]
    
    image
    ```

    我们将看到如图 *图 10.4* 所示的图像：

![图 10.4：一只可爱的纯白色猫在草地上奔跑，对“白色”一词的权重为 1.5，使用 lpw_stable_diffusion_xl](img/B21263_10_04.jpg)

图 10.4：一只可爱的纯白色猫在草地上奔跑，对“白色”一词的权重为 1.5，使用 lpw_stable_diffusion_xl

从图像中，我们可以清楚地看到 `pure (white:1.5) cat` 带入图像中的内容：证明该管道可以使用长加权提示生成图像。

# 摘要

本章试图解决最热门讨论的话题之一：使用 `Diffusers` 包克服 77 个标记限制并为 Stable Diffusion 管道添加提示权重。Automatic1111 的 Stable Diffusion WebUI 提供了一个灵活的用户界面，并且现在（在我写这篇文章的时候）是最流行的提示权重和关注格式。然而，如果我们查看 Automatic1111 的代码，我们可能会很快迷失方向；它的代码很长，没有清晰的文档。

本章从了解 77 个标记限制的根本原因开始，进而探讨了 Stable Diffusion 管道如何使用提示嵌入。我们实现了两个函数来克服 77 个标记限制。

实现了一个不带权重的简单函数，以展示如何绕过 77 个标记限制。我们还构建了另一个具有完整长提示使用功能（无长度限制）的函数，并实现了提示加权。

通过理解和实现这两个函数，我们可以利用这个想法，不仅可以使用 Diffuser 生成与使用 Automatic1111 的 WebUI 相同的高质量图像，还可以进一步扩展它以添加更多强大的功能。至于要添加哪个功能，现在取决于你。在下一章中，我们将开始另一个令人兴奋的主题：使用 Stable Diffusion 修复和放大图像。

# 参考文献

1.  Hugging Face，加权提示：[https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts](https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts)

1.  OpenAI CLIP，连接文本和图像：[https://openai.com/research/clip](https://openai.com/research/clip)

1.  Automatic1111，Stable Diffusion WebUI 提示解析器：[https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/prompt_parser.py#L345C19-L345C19](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/prompt_parser.py#L345C19-L345C19)

)

1.  Automatic1111，关注/强调：[https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)

)

1.  Ashish 等人，*Attention Is All You Need*: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

1.  77 个 token 大小限制的来源：[https://github.com/openai/CLIP/blob/4d120f3ec35b30bd0f992f5d8af2d793aad98d2a/clip/clip.py#L206](https://github.com/openai/CLIP/blob/4d120f3ec35b30bd0f992f5d8af2d793aad98d2a/clip/clip.py#L206)
