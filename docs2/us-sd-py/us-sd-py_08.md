

# 使用社区共享的LoRA

为了满足特定需求并生成更高保真的图像，我们可能需要微调预训练的Stable Diffusion模型，但没有强大的GPU，微调过程会非常缓慢。即使你有所有硬件或资源，微调后的模型仍然很大，通常与原始模型文件大小相同。

幸运的是，来自大型语言模型（LLM）邻域社区的研究人员开发了一种高效的微调方法，**低秩适配**（**LoRA** —— “低”是为什么“o”是小写的原因）[1]。使用LoRA，原始检查点保持冻结状态，没有任何修改，而微调权重更改存储在一个独立的文件中，我们通常称之为LoRA文件。此外，在CIVITAI [4]和HuggingFace等网站上还有无数社区共享的LoRA。

在本章中，我们将深入探讨LoRA的理论，然后介绍将LoRA加载到Stable Diffusion模型中的Python方法。我们还将剖析LoRA模型，以了解LoRA模型的结构，并创建一个自定义函数来加载Stable Diffusion V1.5 LoRA。

本章将涵盖以下主题：

+   LoRA是如何工作的？

+   使用Diffusers与LoRA

+   在加载过程中应用LoRA权重

+   深入了解LoRA

+   创建一个用于加载LoRA的函数

+   为什么LoRA有效

到本章结束时，我们将能够程序化地使用任何社区LoRA，并了解LoRA在Stable Diffusion中是如何以及为什么工作的。

# 技术要求

如果你已经在你的计算机上运行了`Diffusers`包，你应该能够执行本章中的所有代码，以及用于使用Diffusers加载LoRA的代码。

Diffusers使用**PEFT**（**参数高效微调**）[10]来管理LoRA的加载和卸载。PEFT是由Hugging Face开发的库，它提供了参数高效的适应大型预训练模型的方法，以适应特定的下游应用。PEFT背后的关键思想是仅微调模型参数的一小部分，而不是全部微调，从而在计算和内存使用方面节省了大量资源。这使得即使在资源有限的消费级硬件上也能微调非常大的模型。有关LoRA的更多信息，请参阅[*第21章*](B21263_21.xhtml#_idTextAnchor405)。

我们需要安装PEFT包来启用Diffusers的PEFT LoRA加载：

```py
pip install PEFT
```

如果你在代码中遇到其他执行错误，也可以参考[*第2章*](B21263_02.xhtml#_idTextAnchor037)。

# LoRA是如何工作的？

LoRA是一种快速微调扩散模型的技术，最初由微软研究人员在Edward J. Hu等人的论文中提出[1]。它通过创建一个针对特定概念进行适配的小型、低秩模型来实现。这个小模型可以与主检查点模型合并，以生成与用于训练LoRA的图像相似的图像。

让我们用W表示原始UNet注意力权重（Q,K,V），用ΔW表示LoRA的微调权重，用W′表示合并后的权重。将LoRA添加到模型的过程可以表示如下：

W′= W + ΔW

如果我们想控制LoRA权重的比例，我们用α表示这个比例。现在，将LoRA添加到模型可以表示如下：

W′= W + αΔW

α的范围可以从`0`到`1.0` [2]。如果我们把α设置得略大于`1.0`，应该没问题。LoRA之所以如此小，是因为ΔW可以用两个小的矩阵A和B来表示，使得：

ΔW = A B T

其中A ∈ ℝ n×d是一个n × d的矩阵，B ∈ ℝ m×d是一个m × d的矩阵。B的转置，记作B T，是一个d × m的矩阵。

例如，如果ΔW是一个6 × 8的矩阵，总共有`48`个权重数。现在，在LoRA文件中，6 × 8的矩阵可以表示为两个矩阵——一个6 × 2的矩阵，总共`12`个数，另一个2 × 8的矩阵，总共`16`个数。

权重的总数从`48`减少到`28`。这就是为什么与检查点模型相比，LoRA文件可以如此小。

## 使用Diffusers的LoRA

由于开源社区的贡献，使用Python加载LoRA从未如此简单。在本节中，我们将介绍如何使用Diffusers加载LoRA模型。

在以下步骤中，我们将首先加载基础Stable Diffusion V1.5，生成不带LoRA的图像，然后加载一个名为`MoXinV1`的LoRA模型到基础模型中。我们将清楚地看到带和不带LoRA模型之间的差异：

1.  **准备Stable Diffusion管道**：以下代码将加载Stable Diffusion管道并将管道实例移动到VRAM：

    ```py
    import torch
    ```

    ```py
    from diffusers import StableDiffusionPipeline
    ```

    ```py
    pipeline = StableDiffusionPipeline.from_pretrained(
    ```

    ```py
        "runwayml/stable-diffusion-v1-5",
    ```

    ```py
        torch_dtype = torch.float16
    ```

    ```py
    ).to("cuda:0")
    ```

1.  **生成不带LoRA的图像**：现在，让我们生成一个不带LoRA加载的图像。这里，我将使用Stable Diffusion默认v1.5模型以“传统中国水墨画”风格生成“一枝花”：

    ```py
    prompt = """
    ```

    ```py
    shukezouma, shuimobysim, a branch of flower, traditional chinese ink painting
    ```

    ```py
    """
    ```

    ```py
    image = pipeline(
    ```

    ```py
        prompt = prompt,
    ```

    ```py
        generator = torch.Generator("cuda:0").manual_seed(1)
    ```

    ```py
    ).images[0]
    ```

    ```py
    display(image)
    ```

    前面的代码使用了一个非精选的生成器，`默认种子：1`。结果如图*图8**.1*所示：

![图8.1：不带LoRA的花枝](img/B21263_08_01.jpg)

图8.1：未使用LoRA的花枝

坦白说，前面的图像并不那么好，而且“花”更像是一团黑墨点。

1.  **使用默认设置生成带LoRA的图像**：接下来，让我们将LoRA模型加载到管道中，看看MoXin LoRA能对图像生成带来什么帮助。使用默认设置加载LoRA只需一行代码：

    ```py
    # load LoRA to the pipeline
    ```

    ```py
    pipeline.load_lora_weights(
    ```

    ```py
        "andrewzhu/MoXinV1",
    ```

    ```py
        weight_name   = "MoXinV1.safetensors",
    ```

    ```py
        adapter_name  = "MoXinV1"
    ```

    ```py
    )
    ```

    如果模型不存在于你的模型缓存中，Diffusers会自动下载LoRA模型文件。

    现在，再次运行以下代码进行推理（与*步骤2*中使用的相同代码）：

    ```py
    image = pipeline(
    ```

    ```py
        prompt = prompt,
    ```

    ```py
        generator = torch.Generator("cuda:0").manual_seed(1)
    ```

    ```py
    ).images[0]
    ```

    ```py
    display(image)
    ```

    我们将得到一个包含更好“花”的水墨画风格的新图像，如图*图8**.2*所示：

![图8.2：使用默认设置下的带LoRA的花枝](img/B21263_08_02.jpg)

图8.2：使用默认设置下的LoRA的花枝

这次，“花朵”更像是一朵花，总体上比没有应用 LoRA 的那朵花要好。然而，本节中的代码在加载 LoRA 时没有应用“权重”。在下一节中，我们将加载一个具有任意权重（或 α）的 LoRA 模型。

## 在加载过程中应用 LoRA 权重

在 *LoRA 如何工作？* 部分中，我们提到了用于定义添加到主模型中 LoRA 权重部分的 α 值。我们可以使用带有 PEFT [10] 的 Diffusers 容易地实现这一点。

什么是 PEFT？PEFT 是 Hugging Face 开发的一个库，用于高效地适应预训练模型，例如 **大型语言模型**（**LLMs**）和 Stable Diffusion 模型，而无需对整个模型进行微调。PEFT 是一个更广泛的概念，代表了一组旨在高效微调 LLMs 的方法。LoRA，相反，是 PEFT 范畴下的一种特定技术。

在集成 PEFT 之前，Diffusers 中加载和管理 LoRAs 需要大量的自定义代码和破解。为了更轻松地管理具有加载和卸载权重的多个 LoRAs，Diffusers 使用 PEFT 库来帮助管理推理的不同适配器。在 PEFT 中，微调的参数被称为适配器，这就是为什么你会看到一些参数被命名为 `adapters`。LoRA 是主要的适配器技术之一；在本章中，你可以将 LoRA 和适配器视为同一事物。

加载具有权重的 LoRA 模型很简单，如下面的代码所示：

```py
pipeline.set_adapters(
    ["MoXinV1"],
    adapter_weights=[0.5]
)
image = pipeline(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
display(image)
```

在前面的代码中，我们将 LoRA 权重设置为 `0.5` 以替换默认的 `1.0`。现在，你将看到如图 *图 8.3* 所示生成的图像。

![图 8.3：通过应用 0.5 LoRA 权重添加的 LoRA 花枝](img/B21263_08_03.jpg)

图 8.3：通过应用 0.5 LoRA 权重添加的 LoRA 花枝

从 *图 8.3* 中，我们可以观察到应用 `0.5` 权重到 LoRA 模型后的差异。

集成 PEFT 的 Diffusers 也可以通过重用我们用于加载第一个 LoRA 模型的相同代码来加载另一个 LoRA：

```py
# load another LoRA to the pipeline
pipeline.load_lora_weights(
    "andrewzhu/civitai-light-shadow-lora",
    weight_name   = "light_and_shadow.safetensors",
    adapter_name  = "light_and_shadow"
)
```

然后，通过调用 `set_adapters` 函数添加第二个 LoRA 模型的权重：

```py
pipeline.set_adapters(
    ["MoXinV1", "light_and_shadow"],
    adapter_weights=[0.5,1.0]
)
prompt = """
shukezouma, shuimobysim ,a branch of flower, traditional chinese ink painting,STRRY LIGHT,COLORFUL
"""
image = pipeline(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
display(image)
```

我们将得到一个新的图像，其中添加了来自第二个 LoRA 的样式，如图 *图 8.4* 所示：

![图 8.4：具有两个 LoRA 模型的花枝](img/B21263_08_04.jpg)

图 8.4：具有两个 LoRA 模型的花枝

我们也可以使用相同的代码为 Stable Diffusion XL 管道加载 LoRA。

使用 PEFT，我们不需要重新启动管道来禁用 LoRA；我们可以通过一行代码简单地禁用所有 LoRAs：

```py
pipeline.disable_lora()
```

注意，LoRA 加载的实现与其他工具（如 A1111 Stable Diffusion WebUI）略有不同。使用相同的提示、相同的设置和相同的 LoRA 权重，你可能会得到不同的结果。

别担心——在下一节中，我们将深入探讨 LoRA 模型的内部结构，并使用 A1111 Stable Diffusion WebUI 等工具实现一个使用 LoRA 输出相同结果的解决方案。

# 深入探讨LoRA的内部结构

理解LoRA内部工作原理将帮助我们根据具体需求实现自己的LoRA相关功能。在本节中，我们将深入探讨LoRA的结构和权重模式，然后逐步手动将LoRA模型加载到Stable Diffusion模型中。

如我们在本章开头所讨论的，应用LoRA就像以下这样简单：

W′= W + αΔW

ΔW可以分解为A和B：

ΔW = A B T

因此，将LoRA权重合并到检查点模型的整体思路是这样的：

1.  从LoRA文件中找到A和B权重矩阵。

1.  将LoRA模块层名与检查点模块层名匹配，以便我们知道要合并哪个矩阵。

1.  生成ΔW = A B T。

1.  更新检查点模型的权重。

如果你之前有训练LoRA模型的经验，你可能知道可以设置一个超参数`alpha`，其值大于`1`，例如`4`。这通常与将另一个参数`rank`也设置为`4`一起进行。然而，在此上下文中使用的α通常小于1。α的实际值通常使用以下公式计算：

α =  alpha _ rank

在训练阶段，将`alpha`和`rank`都设置为`4`将产生α值为`1`。如果不正确理解，这个概念可能会让人感到困惑。

接下来，让我们一步一步地探索LoRA模型的内部结构。

## 从LoRA文件中找到A和B权重矩阵

在开始探索LoRA结构的内部结构之前，你需要下载一个LoRA文件。你可以从以下URL下载`MoXinV1.safetensors`：[https://huggingface.co/andrewzhu/MoXinV1/resolve/main/MoXinV1.safetensors](https://huggingface.co/andrewzhu/MoXinV1/resolve/main/MoXinV1.safetensors)。

在`.safetensors`格式中设置好LoRA文件后，使用以下代码加载它：

```py
# load lora file
from safetensors.torch import load_file
lora_path = "MoXinV1.safetensors"
state_dict = load_file(lora_path)
for key in state_dict:
    print(key)
```

当LoRA权重应用于文本编码器时，键名以`lora_te_`开头：

```py
...
lora_te_text_model_encoder_layers_7_mlp_fc1.alpha
lora_te_text_model_encoder_layers_7_mlp_fc1.lora_down.weight
lora_te_text_model_encoder_layers_7_mlp_fc1.lora_up.weight
...
```

当LoRA权重应用于UNet时，键名以`lora_unet_`开头：

```py
...
lora_unet_down_blocks_0_attentions_1_proj_in.alpha
lora_unet_down_blocks_0_attentions_1_proj_in.lora_down.weight
lora_unet_down_blocks_0_attentions_1_proj_in.lora_up.weight
...
```

输出是`string`类型。以下是输出LoRA权重键中出现过的术语的含义：

+   `lora_te_`前缀表示权重应用于文本编码器；`lora_unet_`表示权重旨在更新Stable Diffusion `unet`模块的层。

+   `down_blocks_0_attentions_1_proj_in`是层名，这个层名应该存在于检查点模型的`unet`模块中。

+   `.alpha`是训练好的权重，用来表示将有多少LoRA权重应用到主检查点模型中。它持有表示α的浮点值，在W′= W + αΔW中。由于这个值将由用户输入替换，我们可以跳过这个值。

+   `lora_down.weight`表示代表A的这个层的值。

+   `lora_up.weight`表示代表B的这个层的值。

+   注意，`down`在`down_blocks`中表示`unet`模型的下方（UNet的左侧）。

以下Python代码将获取LoRA层信息，并具有模型对象处理器：

```py
# find the layer name
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'
for key in state_dict:
    if 'text' in key:
        layer_infos = key.split('.')[0].split(
            LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
        curr_layer = pipeline.text_encoder
    else:
        layer_infos = key.split('.')[0].split(
            LORA_PREFIX_UNET+'_')[-1].split('_')
        curr_layer = pipeline.unet
```

`key`持有LoRA模块层名称，而`layer_infos`持有从LoRA层中提取的检查点模型层名称。我们这样做的原因是检查点模型中并非所有层都有LoRA权重进行调整，因此我们需要获取将要更新的层的列表。

## 找到相应的检查点模型层名称

打印出检查点模型`unet`结构：

```py
unet = pipeline.unet
modules = unet.named_modules()
for child_name, child_module in modules:
    print("child_module:",child_module)
```

我们可以看到模块是以这样的树状结构存储的：

```py
...
(down_blocks): ModuleList(
    (0): CrossAttnDownBlock2D(
        (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
            (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
            (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
            (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
                (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (attn1): Attention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                )
...
```

每行由一个模块名称（`down_blocks`）组成，模块内容可以是`ModuleList`或特定的神经网络层，`Conv2d`。这些都是UNet的组成部分。目前，将LoRA应用于特定的UNet模块不是必需的。然而，了解UNet的内部结构很重要：

```py
# find the layer name
for key in state_dict:
    # find the LoRA layer name (the same code shown above)
    for key in state_dict:
    if 'text' in key:
        layer_infos = key.split('.')[0].split(
            "lora_unet_")[-1].split('_')
        curr_layer = pipeline.text_encoder
    else:
        layer_infos = key.split('.')[0].split(
            "lora_te_")[-1].split('_')
        curr_layer = pipeline.unet
    # loop through the layers to find the target layer
    temp_name = layer_infos.pop(0)
    while len(layer_infos) > -1:
        try:
            curr_layer = curr_layer.__getattr__(temp_name)
            # no exception means the layer is found
            if len(layer_infos) > 0:
                temp_name = layer_infos.pop(0)
            # all names are pop out, break out from the loop
            elif len(layer_infos) == 0:
                break
        except Exception:
            # no such layer exist, pop next name and try again
            if len(temp_name) > 0:
                temp_name += '_'+layer_infos.pop(0)
            else:
                # temp_name is empty
                temp_name = layer_infos.pop(0)
```

循环部分有点棘手。当回顾检查点模型结构，它以分层的形式作为树时，我们不能简单地使用`for`循环来遍历列表。相反，我们需要使用`while`循环来导航树的每个叶子。整个过程如下：

1.  `layer_infos.pop(0)`将返回列表中的第一个名称，并将其从列表中移除，例如从`layer_infos`列表中移除`up` – `['up', 'blocks', '3', 'attentions', '2', 'transformer', 'blocks', '0', 'ff', '``net', '2']`

1.  使用`curr_layer.__getattr__(temp_name)`来检查层是否存在。如果不存在，将抛出异常，程序将移动到`exception`部分继续输出`layer_infos`列表中的名称，并再次检查。

1.  如果找到了层，但`layer_infos`列表中仍有剩余的名称，它们将继续弹出。

1.  名称将继续出现，直到没有抛出异常，并且我们遇到`len(layer_infos) == 0`条件，这意味着层已完全匹配。

在这一点上，`curr_layer`对象指向检查点模型权重数据，可以在下一步中进行引用。

## 更新检查点模型权重

为了便于键值引用，让我们创建一个`pair_keys = []`列表，其中`pair_keys[0]`返回A矩阵，`pair_keys[1]`返回B矩阵：

```py
# ensure the sequence of lora_up(A) then lora_down(B)
pair_keys = []
if 'lora_down' in key:
    pair_keys.append(key.replace('lora_down', 'lora_up'))
    pair_keys.append(key)
else:
    pair_keys.append(key)
    pair_keys.append(key.replace('lora_up', 'lora_down'))
```

然后，我们更新权重：

```py
alpha = 0.5
# update weight
if len(state_dict[pair_keys[0]].shape) == 4:
    # squeeze(3) and squeeze(2) remove dimensions of size 1 
    #from the tensor to make the tensor more compact
    weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).\
        to(torch.float32)
    weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).\
        to(torch.float32)
    curr_layer.weight.data += alpha * torch.mm(weight_up, 
        weight_down).unsqueeze(2).unsqueeze(3)
else:
    weight_up = state_dict[pair_keys[0]].to(torch.float32)
    weight_down = state_dict[pair_keys[1]].to(torch.float32)
    curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
```

`alpha * torch.mm(weight_up, weight_down)`代码是用于实现αA B T的核心代码。

就这样！现在，管道的文本编码器和`unet`模型权重已经通过LoRA更新。接下来，让我们将所有部分组合起来，创建一个功能齐全的函数，可以将LoRA模型加载到Stable Diffusion管道中。

# 编写一个加载LoRA的函数

让我们再添加一个列表来存储已访问的键，并将所有前面的代码组合到一个名为`load_lora`的函数中：

```py
def load_lora(
    pipeline,
    lora_path,
    lora_weight = 0.5,
    device = 'cpu'
):
    state_dict = load_file(lora_path, device=device)
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    alpha = lora_weight
    visited = []
    # directly update weight in diffusers model
    for key in state_dict:
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
        if 'text' in key:
            layer_infos = key.split('.')[0].split(
                LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split('.')[0].split(
                LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = pipeline.unet
        # find the target layer
        # loop through the layers to find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                # no exception means the layer is found
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                # layer found but length is 0,
                # break the loop and curr_layer keep point to the 
                # current layer
                elif len(layer_infos) == 0:
                    break
            except Exception:
                # no such layer exist, pop next name and try again
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    # temp_name is empty
                    temp_name = layer_infos.pop(0)
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        # ensure the sequence of lora_up(A) then lora_down(B)
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            # squeeze(3) and squeeze(2) remove dimensions of size 1 
            # from the tensor to make the tensor more compact
            weight_up = state_dict[pair_keys[0]].squeeze(3).\
                squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).\
                squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, 
                weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, 
                weight_down)
        # update visited list, ensure no duplicated weight is 
        # processed.
        for item in pair_keys:
            visited.append(item)
```

使用该函数很简单；只需提供`pipeline`对象、LoRA路径`lora_path`和LoRA权重编号`lora_weight`，如下所示：

```py
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.bfloat16
).to("cuda:0")
lora_path = r"MoXinV1.safetensors"
load_lora(
    pipeline = pipeline,
    lora_path = lora_path,
    lora_weight = 0.5,
    device = "cuda:0"
)
```

现在，让我们试试看：

```py
prompt = """
shukezouma, shuimobysim ,a branch of flower, traditional chinese ink painting
"""
image = pipeline(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
display(image)
```

它确实有效，效果很好；请参见*图8**.5*中所示的结果：

![图8.5：使用自定义LoRA加载器的花朵分支](img/B21263_08_05.jpg)

图8.5：使用自定义LoRA加载器的花朵分支

你可能会想知道，“为什么一个小的LoRA文件具有如此强大的能力？”让我们深入探讨LoRA模型有效的原因。

# 为什么LoRA有效

Armen等人撰写的论文*内禀维度解释了语言模型微调的有效性* [8]发现，预训练表示的内禀维度远低于预期，他们如下所述：

“我们通过实验表明，在预训练表示的上下文中，常见的NLP任务具有比完整参数化低几个数量级的内禀维度。”

矩阵的内禀维度是一个用于确定表示该矩阵中包含的重要信息所需的有效维数的概念。

假设我们有一个矩阵`M`，有五行三列，如下所示：

```py
M =  1  2  3
     4  5  6
     7  8  9
     10 11 12
     13 14 15
```

这个矩阵的每一行代表一个包含三个值的数据点或向量。我们可以将这些向量视为三维空间中的点。然而，如果我们可视化这些点，我们可能会发现它们大约位于一个二维平面上，而不是占据整个三维空间。

在这种情况下，矩阵`M`的内禀维度将是`2`，这意味着可以使用两个维度有效地捕捉数据的本质结构。第三个维度没有提供太多额外的信息。

一个低内禀维度的矩阵可以通过两个低秩矩阵来表示，因为矩阵中的数据可以被压缩到几个关键特征。然后，这些特征可以通过两个较小的矩阵来表示，每个矩阵的秩等于原始矩阵的内禀维度。

Edward J. Hu等人撰写的论文*LoRA：大型语言模型的低秩自适应* [1]更进一步，引入了LoRA的概念，利用低内禀维度的特性，通过将权重差分解为两个低秩部分来加速微调过程，ΔW = A B T。

很快发现LoRA的有效性不仅限于LLM模型，还与扩散模型结合产生了良好的结果。Simo Ryu发布了LoRA [2]代码，并成为第一个尝试对Stable Diffusion进行LoRA训练的人。那是在2023年7月，现在在[https://www.civitai.com](https://www.civitai.com)上共享了超过40,000个LoRA模型。

# 摘要

在本章中，我们讨论了如何使用LoRA增强Stable Diffusion模型，理解了LoRA是什么，以及为什么它对微调和推理有益。

然后，我们开始使用`Diffusers`包中的实验函数加载LoRA，并通过自定义实现提供LoRA权重。我们使用简单的代码快速了解LoRA能带来什么。

然后，我们深入研究了LoRA模型的内部结构，详细介绍了提取LoRA权重的步骤，并了解了如何将这些权重合并到检查点模型中。

此外，我们实现了一个Python函数，可以加载LoRA safetensors文件并执行权重合并。

最后，我们简要探讨了LoRA为何有效，基于研究人员最新的论文。

在下一章中，我们将探索另一种强大的技术——文本反转——来教模型新的“单词”，然后使用预训练的“单词”向生成的图像添加新概念。

# 参考文献

1.  Edward J.等人，LoRA：大型语言模型的低秩自适应：[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

1.  Simo Ryu (cloneofsimo), `lora`: [https://github.com/cloneofsimo/lora](https://github.com/cloneofsimo/lora)

1.  `kohya_lora_loader`: [https://gist.github.com/takuma104/e38d683d72b1e448b8d9b3835f7cfa44](https://gist.github.com/takuma104/e38d683d72b1e448b8d9b3835f7cfa44)

1.  CIVITAI：[https://www.civitai.com](https://www.civitai.com)

1.  Rinon Gal等人，一张图片胜过千言万语：使用文本反转个性化文本到图像生成：[https://textual-inversion.github.io/](https://textual-inversion.github.io/)

1.  Diffusers的`lora_state_dict`函数：[https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py)

1.  Andrew Zhu，改进Diffusers包以实现高质量图像生成：[https://towardsdatascience.com/improving-diffusers-package-for-high-quality-image-generation-a50fff04bdd4](https://towardsdatascience.com/improving-diffusers-package-for-high-quality-image-generation-a50fff04bdd4)

1.  Armen等人，内在维度解释了语言模型微调的有效性：[https://arxiv.org/abs/2012.13255](https://arxiv.org/abs/2012.13255)

1.  Hugging Face, LoRA: [https://huggingface.co/docs/diffusers/training/lora](https://huggingface.co/docs/diffusers/training/lora)

1.  Hugging Face, PEFT: [https://huggingface.co/docs/peft/en/index](https://huggingface.co/docs/peft/en/index)
