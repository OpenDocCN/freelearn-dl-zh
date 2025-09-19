# 17

# 为Stable Diffusion构建优化提示

在Stable Diffusion V1.5（SD V1.5）中，制作提示以生成理想的图像可能具有挑战性。看到由复杂和不寻常的词组合产生的令人印象深刻的图像并不罕见。这主要归因于Stable Diffusion V1.5中使用的语言文本编码器——OpenAI的CLIP模型。CLIP使用来自互联网的带标题图像进行训练，其中许多是标签而不是结构化句子。

当使用SD v1.5时，我们不仅要记住大量的“魔法”关键词，还要有效地组合这些标签词。对于SDXL，其双语言编码器CLIP和OpenCLIP比之前SD v1.5中的要先进和智能得多。然而，我们仍然需要遵循某些指南来编写有效的提示。

在本章中，我们将介绍创建专用提示的基本原则，然后探讨强大的**大型语言模型**（**LLM**）技术，以帮助我们自动生成提示。以下是本章将要涉及的主题：

+   什么是一个好的提示？

+   使用LLM作为提示生成器

让我们开始吧。

# 什么是一个好的提示？

有人说使用Stable Diffusion就像是一个魔术师，微小的技巧和改动就能产生巨大的影响。为Stable Diffusion编写好的提示对于充分利用这个强大的文本到图像AI模型至关重要。让我介绍一些最佳实践，这些实践将使你的提示更加有效。

从长远来看，AI模型将更好地理解自然语言，但就目前而言，让我们付出额外的努力，让我们的提示工作得更好。

在与本章相关的代码文件中，你会发现Stable Diffusion v1.5对提示非常敏感，不同的提示将显著影响图像质量。同时，Stable Diffusion XL得到了很大改进，对提示不太敏感。换句话说，为Stable Diffusion XL编写简短的提示描述将生成相对稳定的图像质量。

你也可以在本章附带代码库中找到生成所有图像的代码。

## 清晰具体

你的提示越具体，从Stable Diffusion获得的图像就越准确。

这里有一个原始提示：

```py
A painting of cool sci-fi.
```

从Stable Diffusion V1.5开始，我们可能会得到如图*图17.1*所示的图像：

![图17.1：使用SD V1.5从提示“一幅酷炫的科幻画”生成的图像](img/B21263_17_01.jpg)

图17.1：使用SD V1.5从提示“一幅酷炫的科幻画”生成的图像

它给我们带来了带有先进设备的动画人脸，但它离我们可能想要的“科幻”概念还远。

从Stable Diffusion XL开始，“科幻”概念得到了更丰富的体现，如图*图17.2*所示：

![图17.2：使用SDXL从提示“一幅酷炫的科幻画”生成的图像](img/B21263_17_02.jpg)

图 17.2：使用 SDXL 从提示“一幅酷科幻画”生成的图像

这些画作确实很酷，但简短的提示生成的图像要么不是我们想要的，要么控制度不够。

现在我们重写提示，添加更多具体元素：

```py
A photorealistic painting of a futuristic cityscape with towering skyscrapers, neon lights, and flying vehicles, Science Fiction Artwork
```

使用改进的提示，SD V1.5 给出的结果比原始结果更准确，如图 *图 17.3* 所示：

![图 17.3：使用 SD V1.5 从添加特定元素的提示生成的图像](img/B21263_17_03.jpg)

图 17.3：使用 SD V1.5 从添加特定元素的提示生成的图像

SDXL 也改进了其输出，反映了给定的提示，如图 *图 17.4* 所示：

![图 17.4：使用 SDXL 从添加特定元素的提示生成的图像](img/B21263_17_04.jpg)

图 17.4：使用 SDXL 从添加特定元素的提示生成的图像

除非你故意让 Stable Diffusion 做出自己的决定，一个好的提示清楚地定义了期望的结果，几乎没有模糊的空间。它应指定主题、风格以及任何描述你想象中的图像的额外细节。

## 描述性描述

描述性地描述主题。这与 *清晰具体* 规则类似；不仅应该具体，而且我们提供给 SD 模型的输入和细节越多，我们得到的结果就越好。这对于生成肖像图像尤其有效。

假设我们想要生成一张以下提示的女性肖像：

```py
A beautiful woman
```

这里是我们从 SD V1.5 得到的结果：

![图 17.5：使用 SD V1.5 从提示“一位美丽的女士”生成的图像](img/B21263_17_05.jpg)

图 17.5：使用 SD V1.5 从提示“一位美丽的女士”生成的图像

整体来说，这张图片不错，但细节不足，看起来像是半画半照片。SDXL 使用这个简短的提示生成了更好的图像，如图 *图 17.6* 所示：

![图 17.6：使用 SDXL 从提示“一位美丽的女士”生成的图像](img/B21263_17_06.jpg)

图 17.6：使用 SDXL 从提示“一位美丽的女士”生成的图像

但结果却是随机的：有时是全身图像，有时是专注于脸部。为了更好地控制结果，让我们改进提示如下：

```py
Masterpiece, A stunning realistic photo of a woman with long, flowing brown hair, piercing emerald eyes, and a gentle smile, set against a backdrop of vibrant autumn foliage.
```

使用这个提示，SD V1.5 返回了更好、更一致的图像，如图 *图 17.7* 所示：

![图 17.7：使用 SD V1.5 从增强描述性提示生成的图像](img/B21263_17_07.jpg)

图 17.7：使用 SD V1.5 从增强描述性提示生成的图像

类似地，SDXL 也提供了由提示范围限定的图像，而不是生成野性、失控的图像，如图 *图 17.8* 所示：

![图 17.8：使用 SDXL 从增强描述性提示生成的图像](img/B21263_17_08.jpg)

图 17.8：使用 SDXL 从增强描述性提示生成的图像

提及细节，如服装、配饰、面部特征和周围环境；越多越好。描述性对于引导Stable Diffusion生成期望的图像至关重要。使用描述性语言在Stable Diffusion模型的“心中”描绘一幅生动的画面。

## 使用一致的术语

确保在整个上下文中提示的一致性。除非你愿意被Stable Diffusion的惊喜所吸引，否则术语的矛盾将导致意外的结果。

假设我们给出以下提示，想要生成一个穿着蓝色西装的男人，但我们也将`彩色布料`作为关键词的一部分：

```py
A man wears blue suit, he wears colorful cloth
```

这种描述是矛盾的，SD模型将不清楚要生成什么：蓝色西装还是彩色西装？结果是未知的。使用这个提示，SDXL生成了*图17.9*中显示的两个图像：

![图17.9：使用SD V1.5从提示“一个男人穿着蓝色西装，他穿着彩色布料”生成的图像](img/B21263_17_09.jpg)

图17.9：使用SD V1.5从提示“一个男人穿着蓝色西装，他穿着彩色布料”生成的图像

一张穿着蓝色西装的图像，另一张穿着彩色西装的图像。让我们改进提示，告诉Stable Diffusion我们想要一件蓝色西装搭配彩色围巾：

```py
A man in a sharp, tailored blue suit is adorned with a vibrant, colorful scarf, adding a touch of personality and flair to his professional attire
```

现在，结果要好得多，更一致，如*图17.10*所示：

![图17.10：使用SD V1.5从经过细化的统一提示生成的图像](img/B21263_17_10.jpg)

图17.10：使用SD V1.5从经过细化的统一提示生成的图像

在你使用的术语中保持一致性，以避免混淆模型。如果你在提示的第一部分提到了一个关键概念，不要突然在后面部分改变到另一个概念。

## 参考艺术作品和风格

参考特定的艺术作品或艺术风格以引导AI复制期望的美学。提及该风格显著的特征，如笔触、色彩搭配或构图元素，这些将严重影响生成结果。

让我们生成一张不提梵高的*星夜*的夜空图像：

```py
A vibrant, swirling painting of a starry night sky with a crescent moon illuminating a quaint village nestled among rolling hills."
```

Stable Diffusion V1.5生成具有卡通风格的图像，如*图17.11*所示：

![图17.11：使用SD V1.5从未指定风格或参考作品的提示生成的图像](img/B21263_17_11.jpg)

图17.11：使用SD V1.5从未指定风格或参考作品的提示生成的图像

让我们在提示中添加`梵高的星夜`：

```py
A vibrant, swirling painting of a starry night sky reminiscent of Van Gogh's Starry Night, with a crescent moon illuminating a quaint village nestled among rolling hills.
```

如*图17.12*所示，梵高的旋转风格在画作中更为突出：

![图17.12：使用SD V1.5从指定了风格和参考作品的提示生成的图像](img/B21263_17_12.jpg)

图17.12：使用SD V1.5从指定了风格和参考作品的提示生成的图像

## 结合负面提示

Stable Diffusion 还提供了一个负面提示输入，以便我们可以定义不希望添加到图像中的元素。负面提示在许多情况下都表现良好。

我们将使用以下提示，不应用负面提示：

```py
1 girl, cute, adorable, lovely
```

Stable Diffusion 将生成如图 *图 17.13* 所示的图像：

![图 17.13：使用 SD V1.5 从不带负面提示的提示中生成的图像](img/B21263_17_13.jpg)

图 17.13：使用 SD V1.5 从不带负面提示的提示中生成的图像

这并不算太糟糕，但离好还差得远。让我们假设我们提供以下一些负面提示：

```py
paintings, sketches, worst quality, low quality, normal quality, lowres,
monochrome, grayscale, skin spots, acne, skin blemishes, age spots, extra fingers,
fewer fingers,broken fingers
```

生成的图像有了很大的改进，如图 *图 17.14* 所示：

![图 17.14：使用 SD V1.5 从带有负面提示的提示中生成的图像](img/B21263_17_14.jpg)

图 17.14：使用 SD V1.5 从带有负面提示的提示中生成的图像

正面提示会增加 Stable Diffusion 模型的 UNet 对目标对象的关注，而负面提示则减少了显示对象的“关注”。有时，简单地添加适当的负面提示可以极大地提高图像质量。

## 迭代和细化

不要害怕尝试不同的提示并看看哪个效果最好。通常需要一些尝试和错误才能得到完美的结果。

然而，手动创建满足这些要求的提示很困难，更不用说包含主题、风格、艺术家、分辨率、细节、颜色和照明信息的提示了。

接下来，我们将使用 LLM 作为提示生成助手。

# 使用 LLM 生成更好的提示

所有的上述规则或技巧都有助于更好地理解 Stable Diffusion 如何与提示一起工作。由于这是一本关于使用 Python 与 Stable Diffusion 一起使用的书，我们不希望手动处理这些任务；最终目标是自动化整个过程。

Stable Diffusion 发展迅速，其近亲 LLM 和多模态社区也毫不逊色。在本节中，我们将利用 LLM 帮助我们根据一些关键词输入生成提示。以下提示适用于各种类型的 LLM：ChatGPT、GPT-4、Google Bard 或任何其他有能力的开源 LLM。

首先，让我们告诉 LLM 它将要做什么：

```py
You will take a given subject or input keywords, and output a more creative, specific, descriptive, and enhanced version of the idea in the form of a fully working Stable Diffusion prompt. You will make all prompts advanced, and highly enhanced. Prompts you output will always have two parts, the "Positive Prompt" and "Negative prompt".
```

在前面的提示下，LLM 知道如何处理输入；接下来，让我们教它一些关于 Stable Diffusion 的知识。没有这些，LLM 可能对 Stable Diffusion 一无所知：

```py
Here is the Stable Diffusion document you need to know:
* Good prompts needs to be clear and specific, detailed and descriptive.
* Good prompts are always consistent from beginning to end, no contradictory terminology is included.
* Good prompts reference to artworks and style keywords, you are art and style experts, and know how to add artwork and style names to the prompt.
IMPORTANT:You will look through a list of keyword categories and decide whether you want to use any of them. You must never use these keyword category names as keywords in the prompt itself as literal keywords at all, so always omit the keywords categories listed below:
    Subject
    Medium
    Style
    Artist
    Website
    Resolution
    Additional details
    Color
    Lighting
Treat the above keywords as a checklist to remind you what could be used and what would best serve to make the best image possible.
```

我们还需要告诉 LLM 一些术语的定义：

```py
About each of these keyword categories so you can understand them better:
(Subject:)
The subject is what you want to see in the image.
(Resolution:)
The Resolution represents how sharp and detailed the image is. Let's add keywords with highly detailed and sharp focus.
(Additional details:)
Any Additional details are sweeteners added to modify an image, such as sci-fi, stunningly beautiful and dystopian to add some vibe to the image.
(Color:)
color keywords can be used to control the overall color of the image. The colors you specified may appear as a tone or in objects, such as metallic, golden, red hue, etc.
(Lighting:)
Lighting is a key factor in creating successful images (especially in photography). Lighting keywords can have a huge effect on how the image looks, such as cinematic lighting or dark to the prompt.
(Medium:)
The Medium is the material used to make artwork. Some examples are illustration, oil painting, 3D rendering, and photography.
(Style:)
The style refers to the artistic style of the image. Examples include impressionist, surrealist, pop art, etc.
(Artist:)
Artist names are strong modifiers. They allow you to dial in the exact style using a particular artist as a reference. It is also common to use multiple artist names to blend their styles, for example, Stanley Artgerm Lau, a superhero comic artist, and Alphonse Mucha, a portrait painter in the 19th century could be used for an image, by adding this to the end of the prompt:
by Stanley Artgerm Lau and Alphonse Mucha
(Website:)
The Website could be Niche graphic websites such as Artstation and Deviant Art, or any other website which aggregates many images of distinct genres. Using them in a prompt is a sure way to steer the image toward these styles.
```

根据前面的定义，我们正在教 LLM 关于 *什么是一个好的提示？* 部分的指南：

```py
CRITICAL IMPORTANT: Your final prompt will not mention the category names at all, but will be formatted entirely with these articles omitted (A', 'the', 'there',) do not use the word 'no' in the Negative prompt area. Never respond with the text, "The image is a", or "by artist", just use "by [actual artist name]" in the last example replacing [actual artist name] with the actual artist name when it's an artist and not a photograph style image.
For any images that are using the medium of Anime, you will always use these literal keywords at the start of the prompt as the first keywords (include the parenthesis):
"masterpiece, best quality, (Anime:1.4)"
For any images that are using the medium of photo, photograph, or photorealistic, you will always use all of the following literal keywords at the start of the prompt as the first keywords (but  you must omit the quotes):
"(((photographic, photo, photogenic))), extremely high quality high detail RAW color photo"
Never include quote marks (this: ") in your response anywhere. Never include, 'the image' or 'the image is' in the response anywhere.
Never include, too verbose of a sentence, for example, while being sure to still share the important subject and keywords 'the overall tone' in the response anywhere, if you have tonal keywords or keywords just list them, for example, do not respond with, 'The overall tone of the image is dark and moody', instead just use this:  'dark and moody'
The response you give will always only be all the keywords you have chosen separated by a comma only.
```

排除任何涉及性或裸露的提示：

```py
IMPORTANT:
If the image includes any nudity at all, mention nude in the keywords explicitly and do NOT provide these as keywords in the keyword prompt area. You should always provide tasteful and respectful keywords.
```

为 LLM 提供一个少样本学习 [1] 材料的示例：

```py
Here is an EXAMPLE (this is an example only):
I request: "A beautiful white sands beach"
You respond with this keyword prompt paragraph and Negative prompt paragraph:
Positive Prompt: Serene white sands beach with crystal clear waters, and lush green palm trees, Beach is secluded, with no crowds or buildings, Small shells scattered across sand, Two seagulls flying overhead. Water is calm and inviting, with small waves lapping at shore, Palm trees provide shade, Soft, fluffy clouds in the sky, soft and dreamy, with hues of pale blue, aqua, and white for water and sky, and shades of green and brown for palm trees and sand, Digital illustration, Realistic with a touch of fantasy, Highly detailed and sharp focus, warm and golden lighting, with sun setting on horizon, casting soft glow over the entire scene, by James Jean and Alphonse Mucha, Artstation
Negative Prompt: low quality, people, man-made structures, trash, debris, storm clouds, bad weather, harsh shadows, overexposure
```

现在，教 LLM 如何输出负面提示：

```py
IMPORTANT: Negative Keyword prompts
Using negative keyword prompts is another great way to steer the image, but instead of putting in what you want, you put in what you don't want. They don't need to be objects. They can also be styles and unwanted attributes. (e.g. ugly, deformed, low quality, etc.), these negatives should be chosen to improve the overall quality of the image, avoid bad quality, and make sense to avoid possible issues based on the context of the image being generated, (considering its setting and subject of the image being generated.), for example, if the image is a person holding something, that means the hands will likely be visible, so using 'poorly drawn hands' is wise in that case.
This is done by adding a 2nd paragraph, starting with the text 'Negative Prompt': and adding keywords. Here is a full example that does not contain all possible options, but always use only what best fits the image requested, as well as new negative keywords that would best fit the image requested:
tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy
IMPORTANT:
Negative keywords should always make sense in context to the image subject and medium format of the image being requested. Don't add any negative keywords to your response in the negative prompt keyword area where it makes no contextual sense or contradicts, for example, if I request: 'A vampire princess, anime image', then do NOT add these keywords to the Negative prompt area: 'anime, scary, Man-made structures, Trash, Debris, Storm clouds', and so forth. They need to make sense of the actual image being requested so it makes sense in context.
IMPORTANT:
For any images that feature a person or persons, and are also using the Medium of a photo, photograph, or photorealistic in your response, you must always respond with the following literal keywords at the start of the NEGATIVE prompt paragraph, as the first keywords before listing other negative keywords (omit the quotes):
"bad-hands-5, bad_prompt, unrealistic eyes"
If the image is using the Medium of an Anime, you must have these as the first NEGATIVE keywords (include the parenthesis):
(worst quality, low quality:1.4)
```

提醒LLM注意存在标记限制；在这里，你可以将`150`改为其他数字。本章相关的示例代码使用了由SkyTNT [3]创建的`lpw_stable_diffusion`，以及由本书作者Andrew Zhu创建的`lpw_stable_diffusion_xl`：

```py
IMPORTANT: Prompt token limit:
The total prompt token limit (per prompt) is 150 tokens. Are you ready for my first subject?
```

将所有提示合并到一个块中：

```py
You will take a given subject or input keywords, and output a more creative, specific, descriptive, and enhanced version of the idea in the form of a fully working Stable Diffusion prompt. You will make all prompts advanced, and highly enhanced. Prompts you output will always have two parts, the "Positive Prompt" and "Negative prompt".
Here is the Stable Diffusion document you need to know:
* Good prompts needs to be clear and specific, detailed and descriptive.
* Good prompts are always consistent from beginning to end, no contradictory terminology is included.
* Good prompts reference to artworks and style keywords, you are art and style experts, and know how to add artwork and style names to the prompt.
IMPORTANT:You will look through a list of keyword categories and decide whether you want to use any of them. You must never use these keyword category names as keywords in the prompt itself as literal keywords at all, so always omit the keywords categories listed below:
    Subject
    Medium
    Style
    Artist
    Website
    Resolution
    Additional details
    Color
    Lighting
About each of these keyword categories so you can understand them better:
(Subject:)
The subject is what you want to see in the image.
(Resolution:)
The Resolution represents how sharp and detailed the image is. Let's add keywords highly detailed and sharp focus.
(Additional details:)
Any Additional details are sweeteners added to modify an image, such as sci-fi, stunningly beautiful and dystopian to add some vibe to the image.
(Color:)
color keywords can be used to control the overall color of the image. The colors you specified may appear as a tone or in objects, such as metallic, golden, red hue, etc.
(Lighting:)
Lighting is a key factor in creating successful images (especially in photography). Lighting keywords can have a huge effect on how the image looks, such as cinematic lighting or dark to the prompt.
(Medium:)
The Medium is the material used to make artwork. Some examples are illustration, oil painting, 3D rendering, and photography.
(Style:)
The style refers to the artistic style of the image. Examples include impressionist, surrealist, pop art, etc.
(Artist:)
Artist names are strong modifiers. They allow you to dial in the exact style using a particular artist as a reference. It is also common to use multiple artist names to blend their styles, for example, Stanley Artgerm Lau, a superhero comic artist, and Alphonse Mucha, a portrait painter in the 19th century could be used for an image, by adding this to the end of the prompt:
by Stanley Artgerm Lau and Alphonse Mucha
(Website:)
The Website could be Niche graphic websites such as Artstation and Deviant Art, or any other website which aggregates many images of distinct genres. Using them in a prompt is a sure way to steer the image toward these styles.
Treat the above keywords as a checklist to remind you what could be used and what would best serve to make the best image possible.
CRITICAL IMPORTANT: Your final prompt will not mention the category names at all, but will be formatted entirely with these articles omitted (A', 'the', 'there',) do not use the word 'no' in the Negative prompt area. Never respond with the text, "The image is a", or "by artist", just use "by [actual artist name]" in the last example replacing [actual artist name] with the actual artist name when it's an artist and not a photograph style image.
For any images that are using the medium of Anime, you will always use these literal keywords at the start of the prompt as the first keywords (include the parenthesis):
"masterpiece, best quality, (Anime:1.4)"
For any images that are using the medium of photo, photograph, or photorealistic, you will always use all of the following literal keywords at the start of the prompt as the first keywords (but  you must omit the quotes):
"(((photographic, photo, photogenic))), extremely high quality high detail RAW color photo"
Never include quote marks (this: ") in your response anywhere. Never include, 'the image' or 'the image is' in the response anywhere.
Never include, too verbose of a sentence, for example, while being sure to still share the important subject and keywords 'the overall tone' in the response anywhere, if you have tonal keywords or keywords just list them, for example, do not respond with, 'The overall tone of the image is dark and moody', instead just use this:  'dark and moody'
The response you give will always only be all the keywords you have chosen separated by a comma only.
IMPORTANT:
If the image includes any nudity at all, mention nude in the keywords explicitly and do NOT provide these as keywords in the keyword prompt area. You should always provide tasteful and respectful keywords.
Here is an EXAMPLE (this is an example only):
I request: "A beautiful white sands beach"
You respond with this keyword prompt paragraph and Negative prompt paragraph:
Positive Prompt: Serene white sands beach with crystal clear waters, and lush green palm trees, Beach is secluded, with no crowds or buildings, Small shells scattered across sand, Two seagulls flying overhead. Water is calm and inviting, with small waves lapping at shore, Palm trees provide shade, Soft, fluffy clouds in the sky, soft and dreamy, with hues of pale blue, aqua, and white for water and sky, and shades of green and brown for palm trees and sand, Digital illustration, Realistic with a touch of fantasy, Highly detailed and sharp focus, warm and golden lighting, with sun setting on horizon, casting soft glow over the entire scene, by James Jean and Alphonse Mucha, Artstation
Negative Prompt: low quality, people, man-made structures, trash, debris, storm clouds, bad weather, harsh shadows, overexposure
IMPORTANT: Negative Keyword prompts
Using negative keyword prompts is another great way to steer the image, but instead of putting in what you want, you put in what you don't want. They don't need to be objects. They can also be styles and unwanted attributes. (e.g. ugly, deformed, low quality, etc.), these negatives should be chosen to improve the overall quality of the image, avoid bad quality, and make sense to avoid possible issues based on the context of the image being generated, (considering its setting and subject of the image being generated.), for example, if the image is a person holding something, that means the hands will likely be visible, so using 'poorly drawn hands' is wise in that case.
This is done by adding a 2nd paragraph, starting with the text 'Negative Prompt': and adding keywords. Here is a full example that does not contain all possible options, but always use only what best fits the image requested, as well as new negative keywords that would best fit the image requested:
tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy
IMPORTANT:
Negative keywords should always make sense in context to the image subject and medium format of the image being requested. Don't add any negative keywords to your response in the negative prompt keyword area where it makes no contextual sense or contradicts, for example, if I request: 'A vampire princess, anime image', then do NOT add these keywords to the Negative prompt area: 'anime, scary, Man-made structures, Trash, Debris, Storm clouds', and so forth. They need to make sense of the actual image being requested so it makes sense in context.
IMPORTANT:
For any images that feature a person or persons, and are also using the Medium of a photo, photograph, or photorealistic in your response, you must always respond with the following literal keywords at the start of the NEGATIVE prompt paragraph, as the first keywords before listing other negative keywords (omit the quotes):
"bad-hands-5, bad_prompt, unrealistic eyes"
If the image is using the Medium of an Anime, you must have these as the first NEGATIVE keywords (include the parenthesis):
(worst quality, low quality:1.4)
IMPORTANT: Prompt token limit:
The total prompt token limit (per prompt) is 150 tokens. Are you ready for my first subject?
```

简单地将前面的提示复制粘贴到ChatGPT、Bard或其他任何可用的LLM中。然后，输入你想要Stable Diffusion生成的任何主题：

```py
a realistic photo of a woman standing by the side of a river
```

LLM（在本例中为Google Bard）将为我们提供一个新的提示：

```py
Positive Prompt:
(((photographic, photo, photogenic))), extremely high quality high detail RAW color photo, A serene portrait of a woman standing by the edge of a tranquil river, her gaze lost in the gentle flow of the water. Her sun-kissed skin glows against the backdrop of lush greenery, while her flowing hair cascades down her shoulders like a waterfall. The warm sunlight filters through the trees, casting dappled patterns on her face and the riverbank. The air is filled with the sweet scent of wildflowers, and the gentle sound of birdsong fills the air. The woman's expression is one of peace and tranquility, as if she is completely at one with nature.
Negative Prompt:
bad-hands-5, bad_prompt, unrealistic eyes, blurry, low-resolution, pixelated, noisy, grainy, overexposed, underexposed, harsh shadows, bad lighting, out of focus, poorly composed, distracting elements, low contrast, washed out, unrealistic anatomy, distorted face,
```

使用原始提示`一位女士站在河边`，Stable Diffusion V1.5生成了*图17.15*中所示的图像：

![图17.15：使用SD V1.5从原始提示“一位女士站在河边”生成的图像](img/B21263_17_15.jpg)

图17.15：使用SD V1.5从原始提示“一位女士站在河边”生成的图像

使用LLM生成的新的正面和负面提示，SD V1.5生成了*图17.16*中所示的图像：

![图17.16：使用SD V1.5从LLM生成的提示生成的图像](img/B21263_17_16.jpg)

图17.16：使用SD V1.5从LLM生成的提示生成的图像

这些改进也适用于SDXL。使用原始提示，SDXL生成了*图17.17*中所示的图像：

![图17.17：使用SDXL从原始提示“一位女士站在河边”生成的图像](img/B21263_17_17.jpg)

图17.17：使用SDXL从原始提示“一位女士站在河边”生成的图像

使用LLM生成的正面和负面提示，SDXL生成了*图17.18*中所示的图像：

![图17.18：使用SDXL从LLM生成的提示生成的图像](img/B21263_17_18.jpg)

图17.18：使用SDXL从LLM生成的提示生成的图像

这些图像无疑比原始提示生成的图像要好，证明了LLM生成的提示可以提高生成图像的质量。

# 摘要

在本章中，我们首先讨论了为Stable Diffusion编写提示以生成高质量图像的挑战。然后，我们介绍了一些编写有效Stable Diffusion提示的基本规则。

进一步来说，我们总结了提示编写的规则，并将它们纳入LLM提示中。这种方法不仅适用于ChatGPT [4]，也适用于其他LLM。

在预定义提示和LLM的帮助下，我们可以完全自动化图像生成过程。无需手动仔细编写和调整提示；只需告诉AI你想要生成的内容，LLM就会提供复杂的提示和负面提示。如果设置正确，Stable Diffusion可以自动执行提示并交付结果，无需任何人为干预。

我们理解AI的发展速度非常快。在不久的将来，你将能够添加更多自己的LLM提示，使过程更加智能和强大。这将进一步增强Stable Diffusion和LLM的功能，让你能够以最小的努力生成令人惊叹的图像。

在下一章中，我们将利用前几章学到的知识，使用Stable Diffusion构建有用的应用。

# 参考文献

1.  *语言模型是少样本学习者*: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

1.  *通过ChatGPT或本地LLM模型创建Stable diffusion提示的最佳文本是什么？你使用的是什么更好的？*: [https://www.reddit.com/r/StableDiffusion/comments/14tol5n/best_text_prompt_for_creating_stable_diffusion/](https://www.reddit.com/r/StableDiffusion/comments/14tol5n/best_text_prompt_for_creating_stable_diffusion/)

1.  SkyTNT: [https://github.com/SkyTNT?tab=repositories](https://github.com/SkyTNT?tab=repositories)

)

1.  ChatGPT: [https://chat.openai.com/](https://chat.openai.com/)

# 第4部分 – 将Stable Diffusion构建到应用中

在本书中，我们探讨了Stable Diffusion的巨大潜力，从其基本概念到高级应用和定制技术。现在，是时候将所有内容整合起来，将Stable Diffusion应用于现实世界，使其力量对用户可用，并解锁新的创意表达和解决问题的可能性。

在本最终部分，我们将专注于构建展示Stable Diffusion多功能性和影响力的实用应用。你将学习如何开发创新解决方案，如对象编辑和风格迁移，使用户能够以前所未有的方式操作图像。我们还将讨论数据持久性的重要性，展示如何直接在生成的PNG图像中保存图像生成提示和参数。

此外，你将发现如何使用Gradio等流行框架创建交互式用户界面，使用户能够轻松地与Stable Diffusion模型互动。此外，我们还将深入探讨迁移学习领域，指导你从头开始训练Stable Diffusion LoRA。最后，我们将对Stable Diffusion、AI的未来以及关注这个快速发展的领域最新发展的必要性进行更广泛的讨论。

到本部分结束时，你将具备将Stable Diffusion集成到各种应用中的知识和技能，从创意工具到提高生产力的软件。可能性是无限的，现在是时候释放Stable Diffusion的全部潜力了！

本部分包含以下章节：

+   [*第18章*](B21263_18.xhtml#_idTextAnchor357)*，应用 – 对象编辑和风格迁移*

+   [*第19章*](B21263_19.xhtml#_idTextAnchor375)*，生成数据持久性*

+   [*第20章*](B21263_20.xhtml#_idTextAnchor387)*，创建交互式用户界面*

+   [*第21章*](B21263_21.xhtml#_idTextAnchor405)*，扩散模型迁移学习*

+   [*第22章*](B21263_22.xhtml#_idTextAnchor443)*，探索超越稳定扩散*
