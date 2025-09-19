

# 第七章：优化性能和VRAM使用

在前面的章节中，我们介绍了Stable Diffusion模型背后的理论，介绍了Stable Diffusion模型的数据格式，并讨论了转换和模型加载。尽管Stable Diffusion模型在潜在空间中进行去噪，但默认情况下，模型的数据和执行仍然需要大量资源，并且可能会不时抛出`CUDA Out of memory`错误。

为了使用Stable Diffusion快速平滑地生成图像，有一些技术可以优化整个过程，提高推理速度，并减少VRAM使用。在本章中，我们将介绍以下优化解决方案，并讨论这些解决方案在实际应用中的效果：

+   使用float16或bfloat16数据类型

+   启用VAE分块

+   启用Xformers或使用PyTorch 2.0

+   启用顺序CPU卸载

+   启用模型CPU卸载

+   **Token** **合并** (**ToMe**)

通过使用这些解决方案中的一些，你可以让你的GPU即使只有4 GB RAM也能顺畅地运行Stable Diffusion模型。请参阅[*第2章*](B21263_02.xhtml#_idTextAnchor037)以获取运行Stable Diffusion模型所需的详细软件和硬件要求。

# 设置基线

在探讨优化解决方案之前，让我们看看默认设置下的速度和VRAM使用情况，这样我们就可以知道在应用优化解决方案后VRAM使用量减少了多少，或者速度提高了多少。

让我们使用一个非精选的数字`1`作为生成器的种子，以排除随机生成的种子的影响。测试是在运行Windows 11的RTX 3090（24 GB VRAM）上进行的，还有一个GPU用于渲染所有其他窗口和UI，这样RTX 3090就可以专门用于Stable Diffusion管道：

```py
import torch
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda:0")
# generate an image
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
image
```

默认情况下，PyTorch为卷积启用**TensorFloat32** (**TF32**)模式[4]，为矩阵乘法启用**float32** (**FP32**)模式。前面的代码使用8.4 GB VRAM生成一个512x512的图像，生成速度为7.51次/秒。在接下来的章节中，我们将测量采用优化解决方案后的VRAM使用量和生成速度的提升。

# 优化方案1 – 使用float16或bfloat16数据类型

在PyTorch中，默认以FP32精度创建浮点张量。TF32数据格式是为Nvidia Ampere和后续CUDA设备开发的。TF32可以通过略微降低计算精度来实现更快的矩阵乘法和卷积[5]。FP32和TF32都是历史遗留设置，对于训练是必需的，但网络很少需要如此高的数值精度来进行推理。

我们可以不使用TF32和FP32数据类型，而是以float16或bfloat16精度加载和运行Stable Diffusion模型的权重，以节省VRAM使用并提高速度。但float16和bfloat16之间有什么区别，我们应该使用哪一个？

bfloat16和float16都是半精度浮点数据格式，但它们有一些区别：

+   **值范围**：bfloat16的正值范围比float16大。bfloat16的最大正值约为3.39e38，而float16约为6.55e4。这使得bfloat16更适合需要大动态范围模型的场景。

+   **精度**：bfloat16和float16都具有3位指数和10位尾数（分数）。然而，bfloat16使用最高位作为符号位，而float16将其用作尾数的一部分。这意味着bfloat16的相对精度比float16小，特别是对于小数。

bfloat16通常对深度神经网络很有用。它在范围、精度和内存使用之间提供了良好的平衡。它被许多现代GPU支持，与使用单精度（FP32）相比，可以显著减少内存使用并提高训练速度。

在Stable Diffusion中，我们可以使用bfloat16或float16来提高推理速度并同时减少VRAM使用。以下是一些使用bfloat16加载Stable Diffusion模型的代码：

```py
import torch
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.bfloat16 # <- load float16 version weight
).to("cuda:0")
```

我们使用`text2img_pipe`管道对象生成一个仅使用4.7 GB VRAM的图像，每秒进行19.1次去噪迭代。

注意，如果您使用的是CPU，则不应使用`torch.float16`，因为CPU没有对float16的硬件支持。

# 优化方案2 – 启用VAE分块

Stable Diffusion VAE分块是一种可以用来生成大图像的技术。它通过将图像分割成小块，然后分别生成每个块来实现。这项技术允许在不使用太多VRAM的情况下生成大图像。

注意，分块编码和解码的结果与非分块版本几乎无差别。Diffusers对VAE分块的实现使用重叠的块来混合边缘，从而形成更平滑的输出。

您可以在推理之前添加一行代码`text2img_pipe.enable_vae_tiling()`来启用VAE分块：

```py
import torch
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16       # <- load float16 version weight
).to("cuda:0")
text2img_pipe.enable_vae_tiling()       # < Enable VAE Tiling
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1),
    width = 1024,
    height= 1024
).images[0]
image
```

打开或关闭VAE分块似乎对生成的图像影响不大。唯一的区别是，没有VAE分块时，VRAM使用量生成一个1024x1024的图像需要7.6 GB VRAM。另一方面，打开VAE分块将VRAM使用量降低到5.1 GB。

VAE分块发生在图像像素空间和潜在空间之间，整个过程对去噪循环的影响最小。测试表明，在生成少于4张图像的情况下，对性能的影响不明显，可以减少20%到30%的VRAM使用量。始终开启它是个好主意。

# 优化方案3 – 启用Xformers或使用PyTorch 2.0

当我们提供文本或提示来生成图像时，编码的文本嵌入将被馈送到扩散UNet的Transformer多头注意力组件。

在Transformer块内部，自注意力和交叉注意力头将尝试计算注意力分数（通过`QKV`操作）。这是计算密集型的，并且也会使用大量的内存。

Meta Research的开源`Xformers` [2]软件包旨在优化此过程。简而言之，Xformers与标准Transformers之间的主要区别如下：

+   **分层注意力机制**：Xformers使用分层注意力机制，它由两层注意力组成：粗层和细层。粗层在高层次上关注输入序列，而细层在低层次上关注输入序列。这使得Xformers能够在学习输入序列中的长距离依赖关系的同时，也能关注局部细节。

+   **减少头数**：Xformers使用的头数比标准Transformers少。头是注意力机制中的计算单元。Xformers使用4个头，而标准Transformers使用12个头。这种头数的减少使得Xformers能够在保持性能的同时减少内存需求。

使用`Diffusers`软件包为Stable Diffusion启用Xformers非常简单。只需添加一行代码，如下面的代码片段所示：

```py
import torch
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16       # <- load float16 version weight
).to("cuda:0")
text2img_pipe.enable_xformers_memory_efficient_attention()  # < Enable 
# xformers
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
image
```

如果你使用的是PyTorch 2.0+，你可能不会注意到性能提升或VRAM使用量下降。这是因为PyTorch 2.0包含一个类似于Xformers实现的本地构建的注意力优化功能。如果正在使用版本2.0之前的PyTorch历史版本，启用Xformers将明显提高推理速度并减少VRAM使用。

# 优化方案4 – 启用顺序CPU卸载

正如我们在[*第五章*](B21263_05.xhtml#_idTextAnchor097)中讨论的那样，一个管道包括多个子模型：

+   用于将文本编码为嵌入的文本嵌入模型

+   用于编码输入引导图像和解码潜在空间到像素图像的图像潜在编码器/解码器

+   UNet 将循环推理去噪步骤

+   安全检查模型检查生成内容的安全性

顺序CPU卸载的想法是在完成其任务并空闲时将空闲子模型卸载到CPU RAM。

这里是一个逐步工作的示例：

1.  将CLIP文本模型加载到GPU VRAM，并将输入提示编码为嵌入。

1.  将CLIP文本模型卸载到CPU RAM。

1.  将VAE模型（图像到潜在空间的编码器和解码器）加载到GPU VRAM，并在当前任务是图像到图像管道时编码起始图像。

1.  将VAE卸载到CPU RAM。

1.  将UNet加载到循环遍历去噪步骤（同时加载和卸载未使用的子模块权重数据）。

1.  将UNet卸载到CPU RAM。

1.  将VAE模型从CPU RAM加载到GPU VRAM以执行潜在空间到图像的解码。

在前面的步骤中，我们可以看到在整个过程中，只有一个子模型会留在 VRAM 中，这可以有效地减少 VRAM 的使用。然而，加载和卸载会显著降低推理速度。

启用顺序 CPU 卸载就像以下代码片段中的一行一样简单：

```py
import torch
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16
).to("cuda:0")
# generate an image
text2img_pipe.enable_sequential_cpu_offload() # <- Enable sequential 
# CPU offload
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
image
```

想象一下创建一个定制管道，该管道能够有效地利用 VRAM 来进行 UNet 的去噪。通过在空闲期间将文本编码器/解码器、VAE 模型和安全检查器模型策略性地转移到 CPU 上，同时保持 UNet 模型在 VRAM 中，可以实现显著的速度提升。这种方法在书中提供的自定义实现中得到了证实，它将 VRAM 使用量显著降低到低至 3.2 GB（即使是生成 512x512 的图像），同时保持可比的处理速度，性能没有明显下降！

本章提供的自定义管道代码几乎与 `enable_sequential_cpu_offload()` 做的是同一件事。唯一的区别是保持 UNet 在 VRAM 中直到去噪结束。这就是为什么推理速度保持快速的原因。

通过适当的模型加载和卸载管理，我们可以将 VRAM 使用量从 4.7 GB 降低到 3.2 GB，同时保持与未进行模型卸载时相同的推理速度。

# 优化方案 5 – 启用模型 CPU 卸载

完整模型卸载将整个模型数据移动到和从 GPU 上，而不是只移动权重。如果不启用此功能，所有模型数据在正向推理前后都将留在 GPU 上；清除 CUDA 缓存也不会释放 VRAM。如果你正在加载其他模型，例如，一个上采样模型以进一步处理图像，这可能会导致 `CUDA Out of memory` 错误。模型到 CPU 卸载方法可以缓解 `CUDA Out of` `memory` 问题。

根据这种方法背后的理念，在 CPU RAM 和 GPU VRAM 之间移动模型时，将额外花费一到两秒钟。

要启用此方法，请删除 `pipe.to("cuda")` 并添加 `pipe.enable_model_cpu_offload()`：

```py
import torch
from diffusers import StableDiffusionPipeline
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16
)                 # .to("cuda") is removed here
# generate an image
text2img_pipe.enable_model_cpu_offload()    # <- enable model offload
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
image
```

在卸载模型时，GPU 承载单个主要管道组件，通常是文本编码器、UNet 或 VAE，而其余组件在 CPU 内存中处于空闲状态。像 UNet 这样的组件，在经过多次迭代后，会留在 GPU 上，直到它们的利用率不再需要。

模型 CPU 卸载方法可以将 VRAM 使用量降低到 3.6 GB，并保持相对较好的推理速度。如果你对前面的代码进行测试运行，你会发现推理速度最初相对较慢，然后逐渐加快到其正常迭代速度。

在图像生成结束时，我们可以使用以下代码手动将模型权重数据从 VRAM 移动到 CPU RAM：

```py
pipe.to("cpu")
torch.cuda.empty_cache()
```

执行前面的代码后，你会发现你的 GPU VRAM 使用量水平显著降低。

接下来，让我们来看看标记合并。

# 优化方案6 – 标记合并（ToMe）

**标记合并**（**ToMe**）最初由Daniel等人提出[3]。这是一种可以用来加快稳定扩散模型推理时间的技术。ToMe通过合并模型中的冗余标记来工作，这意味着与未合并的模型相比，模型需要做的工作更少。这可以在不牺牲图像质量的情况下带来明显的速度提升。

ToMe通过首先识别模型中的冗余标记来工作。这是通过查看标记之间的相似性来完成的。如果两个标记非常相似，那么它们可能是冗余的。一旦识别出冗余标记，它们就会被合并。这是通过平均两个标记的值来完成的。

例如，如果一个模型有100个标记，其中50个标记是冗余的，那么合并冗余标记可以将模型需要处理的标记数量减少50%。

ToMe可以与任何稳定扩散模型一起使用。它不需要任何额外的训练。要使用ToMe，我们首先需要从其原始发明者那里安装以下包：

```py
pip install tomesd
```

然后，导入`ToMe`包以启用它：

```py
import torch
from diffusers import StableDiffusionPipeline
import tomesd
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype = torch.float16
).to("cuda:0")
tomesd.apply_patch(text2img_pipe, ratio=0.5)
# generate an image
prompt ="high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(
    prompt = prompt,
    generator = torch.Generator("cuda:0").manual_seed(1)
).images[0]
image
```

性能提升取决于找到多少冗余标记。在前面的代码中，`ToMe`包将迭代速度从大约每秒19次提高到20次。

值得注意的是，`ToMe`包可能会产生略微改变后的图像输出，尽管这种差异对图像质量没有可察觉的影响。这是因为ToMe合并了标记，这可能会影响条件嵌入。

# 摘要

在本章中，我们介绍了六种技术来增强稳定扩散的性能并最小化VRAM的使用。VRAM的量往往是运行稳定扩散模型时最显著的障碍，其中“CUDA内存不足”是一个常见问题。我们讨论的技术可以大幅减少VRAM的使用，同时保持相同的推理速度。

启用float16数据类型可以将VRAM使用量减半，并将推理速度提高近一倍。VAE分块允许在不使用过多VRAM的情况下生成大图像。Xformers通过实现智能的两层注意力机制，可以进一步减少VRAM使用量并提高推理速度。PyTorch 2.0提供了原生功能，如Xformers，并自动启用它们。

通过将子模型及其子模块卸载到CPU RAM，顺序CPU卸载可以显著减少VRAM的使用，尽管这会以较慢的推理速度为代价。然而，我们可以使用相同的概念来实现我们的顺序卸载机制，以节省VRAM使用量，同时保持推理速度几乎不变。模型CPU卸载可以将整个模型卸载到CPU，为其他任务释放VRAM，并在必要时才将模型重新加载回VRAM。**标记合并**（或**ToMe**）减少了冗余标记并提高了推理速度。

通过应用这些解决方案，你可能会运行一个性能优于世界上任何其他模型的流水线。人工智能领域正在不断演变，在你阅读这段文字的时候，可能会有新的解决方案出现。然而，理解其内部工作原理使我们能够根据你的需求调整和优化图像生成过程。

在下一章中，我们将探讨最激动人心的主题之一，即社区共享的LoRAs。

# 参考文献

1.  Hugging Face、内存和速度：[https://huggingface.co/docs/diffusers/optimization/fp16](https://huggingface.co/docs/diffusers/optimization/fp16)

1.  facebookresearch, xformers：[https://github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers)

1.  Daniel Bolya, Judy Hoffman；快速稳定扩散的Token合并：[https://arxiv.org/abs/2303.17604](https://arxiv.org/abs/2303.17604)

1.  每个用户都应该了解的PyTorch混合精度训练：[https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/#picking-the-right-approach](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/#picking-the-right-approach)

)

1.  使用NVIDIA TF32 Tensor Cores加速AI训练：[https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)
