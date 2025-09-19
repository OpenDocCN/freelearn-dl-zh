

# 第十九章：生成数据持久化

想象一个 Python 程序生成图像，但当你回到图像希望进行改进或简单地根据原始提示生成新图像时，你找不到确切的提示、推理步骤、指导比例以及其他实际上生成图像的东西！

解决此问题的一个方案是将所有元数据保存到生成的图像文件中。**便携式网络图形**（PNG）[1] 图像格式为我们提供了一个机制，可以存储与图像像素数据一起的元数据。我们将探讨这个解决方案。

在本章中，我们将探讨以下内容：

+   探索和理解 PNG 文件结构

+   在 PNG 文件中存储稳定扩散生成元数据

+   从 PNG 文件中提取稳定扩散生成元数据

通过采用本章提供的解决方案，您将能够保持图像文件中的生成提示和参数，并提取元信息以供进一步使用。

让我们开始。

# 探索和理解 PNG 文件结构

在保存图像元数据和稳定扩散生成参数之前，我们最好对为什么选择 PNG 作为输出图像格式以保存稳定扩散的输出有一个全面的了解，以及为什么 PNG 可以支持无限定制的元数据，这对于将大量数据写入图像非常有用。

通过理解 PNG 格式，我们可以自信地将数据写入 PNG 文件，因为我们打算将数据持久化到图像中。

PNG 是一种光栅图形文件格式，是稳定扩散生成的理想图像格式。PNG 文件格式被创建为一个改进的、非专利的无损图像压缩格式，现在在互联网上广泛使用。

除了 PNG，还有其他几种图像格式也支持保存自定义图像元数据，例如 JPEG、TIFF、RAW、DNG 和 BMP。然而，这些格式都有它们的问题和限制。JPEG 文件可以包含自定义的 Exif 元数据，但 JPEG 是一种有损压缩图像格式，通过牺牲图像质量达到高压缩率。DNG 是 Adobe 拥有的专有格式。与 PNG 相比，BMP 的自定义元数据大小有限。

对于 PNG 格式，除了存储额外元数据的能力外，还有很多优点使其成为理想的格式 [1]：

+   **无损压缩**：PNG 使用无损压缩，这意味着在压缩过程中图像质量不会降低

+   **透明度支持**：PNG 支持透明度（alpha 通道），允许图像具有透明背景或半透明元素

+   **宽色域**：PNG 支持 24 位 RGB 颜色，32 位 RGBA 颜色和灰度图像，提供广泛的颜色选项

+   **伽玛校正**：PNG 支持伽玛校正，有助于在不同设备和平台之间保持一致的色彩

+   **渐进显示**：PNG 支持交错，允许图像在下载过程中逐步显示。

我们还需要意识到，在某些情况下，PNG 可能不是最佳选择。以下是一些例子：

+   **更大的文件大小**：与 JPEG 等其他格式相比，PNG 文件可能更大，因为其无损压缩。

+   **不支持动画的原生支持**：与 GIF 不同，PNG 不支持原生的动画。

+   **不适用于高分辨率照片**：由于其无损压缩，PNG 不是高分辨率照片的最佳选择，因为文件大小可能比使用有损压缩的 JPEG 等格式大得多。

尽管存在这些限制，PNG 仍然是一种可行的图像格式选择，尤其是对于 Stable Diffusion 的原始图像。

PNG 文件的内部数据结构基于基于块的架构。每个块是一个自包含的单元，它存储有关图像或元数据的特定信息。这种结构允许 PNG 文件存储附加信息，如文本、版权或其他元数据，而不会影响图像数据本身。

PNG 文件由一个签名后跟一系列块组成。以下是 PNG 文件主要组件的简要概述：

+   **签名**：PNG 文件的前 8 个字节是一个固定的签名（十六进制中的 89 50 4E 47 0D 0A 1A 0A），用于标识文件为 PNG。

+   `length` 字段。

+   **CRC**（4 字节）：用于错误检测的循环冗余检查（CRC）值，基于块的类型和数据字段计算。

这种结构提供了灵活性和可扩展性，因为它允许在不破坏现有 PNG 解码器兼容性的情况下添加新的块类型。此外，这种 PNG 数据结构使得将几乎无限量的附加元数据插入到图像中成为可能。

接下来，我们将使用 Python 将一些文本数据插入到 PNG 图像文件中。

# 在 PNG 图像文件中保存额外的文本数据。

首先，让我们使用 Stable Diffusion 生成一个用于测试的图像。与我们在前几章中使用的代码不同，这次我们将使用 JSON 对象来存储生成参数。

加载模型：

```py
import torch
from diffusers import StableDiffusionPipeline
model_id = "stablediffusionapi/deliberate-v2"
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype = torch.float16
)
# Then, we define all the parameters that will be used to generate an 
# image in a JSON object:
gen_meta = {
    "model_id": model_id,
    "prompt": "high resolution, 
        a photograph of an astronaut riding a horse",
    "seed": 123,
    "inference_steps": 30,
    "height": 512,
    "width": 768,
    "guidance_scale": 7.5
}
```

现在，让我们使用 Python 的 `dict` 类型中的 `gen_meta`：

```py
text2img_pipe.to("cuda:0")
input_image = text2img_pipe(
    prompt = gen_meta["prompt"],
    generator = \
        torch.Generator("cuda:0").manual_seed(gen_meta["seed"]),
    guidance_scale = gen_meta["guidance_scale"],
    height = gen_meta["height"],
    width = gen_meta["width"]
).images[0]
text2img_pipe.to("cpu")
torch.cuda.empty_cache()
input_image
```

我们应该有一个使用 `input_image` 处理生成的图像 – 在 Python 上下文中对图像对象的引用。

接下来，让我们逐步将 `gen_meta` 数据存储在 PNG 文件中：

1.  如果您还没有安装，请安装 `pillow` 库 [2]：

    ```py
    pip install pillow
    ```

1.  使用以下代码添加一个存储文本信息的块：

    ```py
    from PIL import Image
    ```

    ```py
    from PIL import PngImagePlugin
    ```

    ```py
    import json
    ```

    ```py
    # Open the original image
    ```

    ```py
    image = Image.open("input_image.png")
    ```

    ```py
    # Define the metadata you want to add
    ```

    ```py
    metadata = PngImagePlugin.PngInfo()
    ```

    ```py
    gen_meta_str = json.dumps(gen_meta)
    ```

    ```py
    metadata.add_text("my_sd_gen_meta", gen_meta_str)
    ```

    ```py
    # Save the image with the added metadata
    ```

    ```py
    image.save("output_image_with_metadata.png", "PNG", 
    ```

    ```py
        pnginfo=metadata)
    ```

    现在，字符串化的 `gen_meta` 已存储在 `output_image_with_metadata.png` 文件中。请注意，我们首先需要使用 `json.dumps(gen_meta)` 将 `gen_data` 从对象转换为字符串。

    上述代码向 PNG 文件添加了一个数据块。正如我们在本章开头所学的，PNG 文件是按块堆叠的，这意味着我们应该能够向 PNG 文件中添加任意数量的文本块。在下面的示例中，我们添加了两个块而不是一个：

    ```py
    from PIL import Image
    ```

    ```py
    from PIL import PngImagePlugin
    ```

    ```py
    import json
    ```

    ```py
    # Open the original image
    ```

    ```py
    image = input_image#Image.open("input_image.png")
    ```

    ```py
    # Define the metadata you want to add
    ```

    ```py
    metadata = PngImagePlugin.PngInfo()
    ```

    ```py
    gen_meta_str = json.dumps(gen_meta)
    ```

    ```py
    metadata.add_text("my_sd_gen_meta", gen_meta_str)
    ```

    ```py
    # add a copy right json object
    ```

    ```py
    copyright_meta = {
    ```

    ```py
        "author":"Andrew Zhu",
    ```

    ```py
        "license":"free use"
    ```

    ```py
    }
    ```

    ```py
    copyright_meta_str = json.dumps(copyright_meta)
    ```

    ```py
    metadata.add_text("copy_right", copyright_meta_str)
    ```

    ```py
    # Save the image with the added metadata
    ```

    ```py
    image.save("output_image_with_metadata.png", "PNG", 
    ```

    ```py
        pnginfo=metadata)
    ```

    只需调用另一个 `add_text()` 函数，我们就可以向 PNG 文件中添加第二个文本块。接下来，让我们从 PNG 图像中提取添加的数据。

1.  从 PNG 图像中提取文本数据是直接的。我们将再次使用 `pillow` 包来完成提取任务：

    ```py
    from PIL import Image
    ```

    ```py
    image = Image.open("output_image_with_metadata.png")
    ```

    ```py
    metadata = image.info
    ```

    ```py
    # print the meta
    ```

    ```py
    for key, value in metadata.items():
    ```

    ```py
        print(f"{key}: {value}")
    ```

    我们应该看到如下输出：

    ```py
    my_sd_gen_meta: {"model_id": "stablediffusionapi/deliberate-v2", "prompt": "high resolution, a photograph of an astronaut riding a horse", "seed": 123, "inference_steps": 30, "height": 512, "width": 768, "guidance_scale": 7.5}
    ```

    ```py
    copy_right: {"author": "Andrew Zhu", "license": "free use"}
    ```

通过本节提供的代码，我们应该能够将自定义数据保存到 PNG 图像文件中，并从中检索。

# PNG 额外数据存储限制

您可能会想知道文本数据大小是否有任何限制。写入 PNG 文件的元数据量没有具体的限制。然而，基于 PNG 文件结构和用于读取和写入元数据的软件或库的限制，存在实际上的约束。

如我们在第一部分所讨论的，PNG 文件是按块存储的。每个块的最大大小为 2^31 - 1 字节（约 2 GB）。虽然理论上可以在单个 PNG 文件中包含多个元数据块，但在这些块中存储过多或过大的数据可能导致使用其他软件打开图像时出现错误或加载时间变慢。

在实践中，PNG 文件中的元数据通常很小，包含诸如版权、作者、描述或用于创建图像的软件等信息。在我们的案例中，是用于生成图像的稳定扩散参数。不建议在 PNG 元数据中存储大量数据，因为这可能会导致性能问题和与某些软件的兼容性问题。

# 概述

在本章中，我们介绍了一种将图像生成提示和相对参数存储在 PNG 图像文件中的解决方案，这样生成数据就会随着文件移动，我们可以使用稳定扩散提取参数，以增强或扩展提示以供其他用途使用。

本章介绍了 PNG 文件的文件结构，并提供了示例代码，用于在 PNG 文件中存储多个文本数据块，然后使用 Python 代码从 PNG 文件中提取元数据。

通过解决方案的示例代码，您也将能够从 A1111 的稳定扩散网页界面生成的图像中提取元数据。

在下一章中，我们将为稳定扩散应用程序构建一个交互式网页界面。

# 参考文献

1.  可移植网络图形 (PNG) 规范：[https://www.w3.org/TR/png/](https://www.w3.org/TR/png/)

1.  Pillow 包：[https://pillow.readthedocs.io/en/stable/](https://pillow.readthedocs.io/en/stable/)
