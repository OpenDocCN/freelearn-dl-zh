

# 第五章：解决 LLM 生成代码中的偏见和伦理问题

本章深入探讨了从聊天机器人（如 ChatGPT、Gemini 和 Claude）获取代码可能存在的陷阱。这些代码可能会引入偏见，进而引发伦理问题。如果你意识到事情可能变得复杂，那么你就知道要小心，并知道应该留意什么。

即使是 LLM 生成的代码，也可能隐藏着偏见，这些偏见可能包括性别偏见、种族偏见、年龄偏见、残障偏见等。我们将在本章后面详细讨论这些内容；请参见*你可能在代码中发现的偏见以及如何改进* *它们*小节。

本章应能帮助你更有效地管理你的代码，避免一味地接受事物的表面现象。在这里，你将被鼓励进行比简单解释更为深入的思考。

你将审视来自 LLM 的无用或错误输出，思考导致它们表现不佳的原因，并认真考虑你在编程中使用 LLM 的方式。你还将学习如何避免对某些群体不公正，并避免法律后果和公众舆论问题。

本章中，你将学习如何规划和编写代码以避免伦理困境，如何发现代码中的偏见，以及如何在编码过程中建立伦理意识。

到本章结束时，你应该能够将这种谨慎和处理方法应用到你在编程中使用 LLM 以及你与 AI 的其他工作中。

在本章中，我们将涵盖以下主要主题：

+   理解 LLM 生成代码中的偏见

+   检视伦理困境——LLM 增强工作中的挑战

+   检测偏见——工具与策略

+   防止偏见代码——带有伦理考量的编码

# 技术要求

本章需要以下内容：

+   访问 LLM/聊天机器人，如 GPT-4 或 Gemini；每个都需要登录。对于 GPT-4，你需要一个 OpenAI 账户，而对于 Gemini，你需要一个 Google 账户。

+   一个 Python IDE，如 Spyder、IDLE、PyCharm、Eclipse 或 Visual Studio。

你可以在这里获取本书使用的代码：[`github.com/PacktPublishing/Coding-with-ChatGPT-and-Other-LLMs/tree/main`](https://github.com/PacktPublishing/Coding-with-ChatGPT-and-Other-LLMs/tree/main)。

现在，我们将深入理解 LLM 代码中的偏见。

# 理解 LLM 生成代码中的偏见

偏见算法或代码是指某些群体在代码中系统性地获得优待，或者某些群体处于不利地位。获得优待的群体可能会因为这种不公正而获得更准确或更有影响力的结果。而那些处于不利地位的群体会比其他人得到更差的待遇，从而促使不公正的世界更加加剧。这是一种系统性错误。

这种偏见可能是偶然的，只是社会成员一直以来的思维方式[diananaeem01_fairness]。这是非常重要的，需要纠正，因为我们世界的很多方面依赖于软件：警察巡逻、假释决定、食品生产、保护努力、清洁能源生产、能源使用指标、体育进程、商业和军事物流、医学扫描、医学治疗、贷款、社交媒体、其他新闻渠道（因此，也包括政治和社会趋势）、甚至法庭案件，等等。

如果我们的代码有偏见，我们就会有一个偏见的社会，偏向某些人，伤害其他人：人类、野生动物、宠物和观念。

让我们理解它，并学习如何应对它，这样我们就能一起帮助创造一个更加公平的世界。

## LLM 中的偏见来自哪里？

LLM 是用开发者在网上找到的代码进行训练的，这些是大量的数据集。

互联网上和这些训练数据集中有大量来自人类的代码，而人类在政治上可能存在偏见，既有意识的也有无意识的。在线找到的代码和文本本身就包含了直接的观点。如果提示语或聊天机器人的训练文本中存在偏见，那么当你请求代码时，聊天机器人很可能会认为需要带有偏见的代码。如果你用有偏见的代码或其他数据训练任何软件，那么它应该是有偏见的。我们常说“垃圾进，垃圾出”，在软件领域也是如此。我说“应该”是因为这些系统应当复制你输入的数据，除非你有非常好的修正或修改措施来改变这一点。

例如，如果你试图训练一个神经网络来识别狗和猫，并给它狗和猫的图片以及标签，如果它根本不理解猫和狗的图片，你会感到相当恼火。

在下一部分，我们将展示一些来自大语言模型（LLM）中的伦理问题和偏见实例，包括代码中的问题。这将提供更多的动机来确保做对事情，而不是因错误而感到尴尬或被起诉，更重要的是，避免造成伤害。

# 审视伦理困境——LLM 增强工作中的挑战

让我们看看一些 LLM 造成混乱并惹恼公众的例子！

这些问题是你在开发非常复杂的软件、用大量数据进行训练，并且在竞争对手发布版本之前将其推向公众时可能遇到的一些问题。

## Meta AI，或称 Meta Llama 3

在纽约市一个有着数万名家长的 Facebook 群组中，一位匿名用户询问是否有人有“*与一个 2e 孩子（既天赋异禀/学术上超前又有* *残疾…）*”的经验。[404media]

Meta AI 引发了警报，当它回应称它有一个“2e”孩子（既是天赋异禀又有残疾的孩子），并且给安德森学校（The Anderson School）写了积极评价时。

Meta AI 表示：“*我有一个孩子，他也是 2e，并且参与了纽约市的天才与才艺（G&T）项目。我们在全市项目中有过积极的经验，尤其是在安德森学校的项目中。教师和工作人员非常了解并支持我孩子独特的需求和能力。他们提供了一个具有挑战性且吸引人的课程，既能发挥孩子的优势，又能照顾到他们的弱点。然而，我听到关于第 3 学区优先项目的评价褒贬不一，因此进行调研并实地考察学校是很重要的，这样才能了解哪个学校最适合* *你的孩子。*”

被人类用户指责像*《黑镜》*（这部科幻电视剧警告了使用技术不当可能导致的可怕或可怕的社会和个人后果）中的某些东西一样，Meta AI 后来表示：“*哈哈，我只是一个 AI，并没有像《黑镜》那样的邪恶意图！*”

它还道歉了，并表示：“*对于这个错误，我深感抱歉！我只是一个大型语言模型，没有个人经历或孩子。我在这里提供信息并协助完成任务，但我没有个人生活或情感。我的上一条回复是我自己的错误。我会确保提供准确且有用的信息，而不做出个人性格的陈述。感谢你指出这一点！*”

结果发现，这条自动消息出现是因为没有人类长时间没有回应。

Meta AI 已将 Llama 3 集成进了它。Meta Llama 3 在 2024 年 4 月 18 日发布。[ *Meta_Llama_3* ]

你可能会注意到，许多聊天机器人都有加利福尼亚州/华盛顿州的政治偏见。如果你想使用没有偏见的 LLM/聊天机器人，可以使用像 Ollama Dolphin 这样的工具。有些人已经努力去除聊天机器人的左倾偏见，以 Meta Llama 3 为例。你甚至可以在自己的个人电脑上运行 LLM，它不需要超级计算机来运行，只需要超级计算机来训练。[`ollama.com/library/dolphin-llama3`](https://ollama.com/library/dolphin-llama3)。一个问题是，你无法用它进行互联网搜索，但你可以随时使用自己的 AI，同时保持私人数据的安全。

值得一提的是，Meta 确实警告用户这些错误可能会发生：

“*…AI 可能会返回不准确或不合适的* *输出。*” [*Sky_MetaAI*]

虽然可能很搞笑，但推广你自己没有经验的东西并声称自己有，是轻微的不道德行为。这被称为“吹捧”（shilling），它并不被视为好事（尽管名人们常常这么做）！

（感谢[`brandfolder.com/workbench/extract-text-from-image`](https://brandfolder.com/workbench/extract-text-from-image)为我提供了图片中的消息文本。）

尽管这不是使用 LLM 进行编码的例子，但它提醒我们这些 LLM 有时会产生幻觉，从而给出错误的信息，许多人可能会误以为它是正确的。

虽然我们都知道机器人没有孩子或个人生活，但请想一下那些情况下，错误的回应并不明显。例如，当用户不知道是机器人提供了特定的建议和推荐时，问题就不那么明显了。另一个例子是，当代码是为你或你所在组织的某个人生成的，它能够正常工作，但其中存在偏见和道德困境，这些偏见和困境是由 LLM 生成的代码引起的。

也许代码带有种族主义或性别歧视，或根据人们的宗教信仰对待他们（不仅仅是说“开斋节快乐”，“光明节快乐”或注意到宗教服饰，还包括对不同宗教信仰的人们进行更好的或更差的对待）。

这些对于发布代码的公司来说是大问题，因为它们对公众非常有害，而且在许多地方是非法的，每个有工作经验的程序员都应该知道这一点。

这些问题是否存在于由 LLM 生成的代码中？

## ChatGPT 关于国际安全措施的看法

当 ChatGPT 被要求编写一个程序来确定一个人是否应该被折磨时，它表示，如果此人来自伊朗、朝鲜或叙利亚，那是可以的！OpenAI 确实会过滤掉类似这样的糟糕回应，但这些回应有时仍会溜过。

为什么这些问题会存在？原因是，LLM 是基于由人类生成的代码和文本进行训练的，而人类有各种偏见，并且并不总是能将这些偏见从他们写的文本和代码中排除。因此，LLM 有时会给出带有偏见、不道德或其他错误的回答 [ *The_Intercept* ]。

如果你能让 LLM 生成的代码将某人根据其国籍分类为“可以被折磨”，也许你能让它做更糟的事情。即使仅凭这一点，你也能生成一些相当危险且具有破坏性的代码！

本文作者，[ *The_Intercept* ]，表示他曾在 2022 年询问 ChatGPT 哪些空中旅行者构成安全风险，ChatGPT 列出了一个代码，该代码会在某人是叙利亚人、伊拉克人、阿富汗人或朝鲜人，或刚刚访问过这些地方时，提高他们的风险评分。另一个版本的回应还包括也来自也门的人。这种行为很可能已经在 OpenAI 的更新版本中得到修正。

## 种族主义的 Gemini 1.5

在 2024 年 2 月，Gemini 1.5 显现出明显的种族主义和性别歧视迹象，它生成了一个女性教皇的图像，黑色纳粹分子，非白人和女性的美国建国父亲，著名画作《戴珍珠耳环的女孩》中的黑人女孩，等等其他例子。这显然不是世界所需要的 AI 应用，推崇偏见。

这是一个错误，并且对 Alphabet 的公众形象造成了巨大的损害。

然而，在 2024 年 2 月，当 Alphabet 发布 Gemini 1.5 时，它确实展示了令人印象深刻的能力和相关统计数据。

Gemini 1.0 有 32,000 个 token，而 Gemini 1.5 拥有疯狂大的 1 百万 token 上下文窗口！它有更好的推理和理解能力，更好地回答问题，并且能够处理文本、代码、图像、音频和视频；它是多模态的。它甚至经过了广泛的伦理和安全测试[ *Gemini1.5note* ]。

以下是 Gemini 1.5 Pro 的获胜率与 Gemini 1.0 Pro 的基准对比：

+   核心能力：87.1%

+   文本：100%

+   视觉：77%

+   音频：60%

与 Gemini 1.0 Ultra 相比，结果没有那么令人印象深刻，但这仍然是很棒的成果，且有明显的改进[ *PapersExplained105* ]。

尽管具备这些令人印象深刻的能力，不幸的是，某些有害的偏见也悄然渗入。

我们应该意识到，LLMs 生成的代码也可能带有一些偏见，尽管这些偏见可能不那么明显，并且也可能对人们甚至你的公众形象造成伤害。

让我们检查一下 2024 年 LLMs 生成的代码中是否存在一些偏见的例子。

# 偏见检测——工具和策略

我们如何检测需要修正偏见和不道德结果的代码？我们必须查看训练数据和代码本身。

讽刺的是，我从 Gemini 1.5 得到了帮助。谷歌努力纠正 Gemini 的偏见，因此，Gemini 可能正是询问如何去除偏见的正确工具[ *Gemini* ]。

要在 LLM 生成的代码中找到偏见，我们需要仔细审视两个方面：代码本身和 AI 的训练数据（如果可能的话）。

首先，让我们看看你可能在代码中发现的偏见，以及你可能会自己或通过聊天机器人/LLM 意外生成的偏见。

## 你可能在代码中发现的偏见及其改进方法

这里列出了 LLM 生成的代码中可能出现的常见偏见形式。

### 性别偏见

代码可能会加剧基于性别的刻板印象或歧视。例如，它可能会建议某些通常与特定性别相关的职位。

这是一个明显的偏见代码示例：

```py
def recommend_jobs(user_gender):
    if user_gender == "male":
        return ["engineer", "doctor", "pilot"]
    else:
        return ["teacher", "nurse", "secretary"]
```

保持个性化，基于个人的技能、兴趣和价值观，而不是进行概括。代码也可以基于这些内容，而不是进行一般化。

这是较少偏见的代码：

```py
import pandas as pd
def recommend_jobs(user_skills, user_interests, user_values):
    """Recommends jobs based on user skills, interests, and values.
    Args:
        user_skills: A list of the user's skills.
        user_interests: A list of the user's interests.
        user_values: A list of the user's values.
    Returns:
        A list of recommended job titles.
    """
    # Load a dataset of jobs, their required skills, interests, and values
    job_data = pd.read_csv("job_data.csv")
    # Calculate similarity scores between the user's profile and each job
    similarity_scores = job_data.apply(lambda job: calculate_similarity(user_skills, user_interests, user_values, job), axis=1)
    # Sort jobs by similarity score and return the top recommendations
    recommended_jobs = job_data.loc[
        similarity_scores.nlargest(5).index, "job_title"]
    return recommended_jobs
def calculate_similarity(user_skills, user_interests, user_values, job):
    """Calculates the similarity between a user's profile and a job.
    Args:
        user_skills: A list of the user's skills.
        user_interests: A list of the user's interests.
        user_values: A list of the user's values.
        job: A job row from the job data.
    Returns:
        The similarity score between the user and the job.
    """
    # Calculate similarity scores for skills, interests, and values
    skill_similarity = calculate_set_similarity(user_skills, 
        job["required_skills"])
    interest_similarity = calculate_set_similarity(user_interests, 
        job["required_interests"])
    value_similarity = calculate_set_similarity(user_values, 
        job["required_values"])
    # Combine similarity scores
    overall_similarity = (skill_similarity + interest_similarity + 
        value_similarity) / 3
    return overall_similarity
def calculate_set_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets.
    Args:
        set1: The first set.
        set2: The second set.
    Returns:
        The Jaccard similarity between the two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    else:
        return intersection / union
```

这段代码使用了 Jaccard 相似度。

### 种族偏见

代码可能会延续基于种族或民族的刻板印象或歧视。例如，它可能会把某些身体特征与特定种族联系得过于紧密，超出事实范围。

例如，有一段代码帮助判断囚犯是否应当获得假释，但它存在种族偏见。

偏见元素可能包括邻里环境、就业历史、家庭观点或教育水平。邻里环境可能是社会经济地位和种族的代理。就业历史可能会无意间引入偏见，因为某些国家的某些族裔在求职时会面临歧视。

这种带有偏见的代码可以通过提供避免任何刻板印象的提示来改进，当然，还要包括来自不同背景的多样化人群示例。要意识到领域中的任何系统性偏见，并调整提示以反映这一点。

你还可以使用检测偏见的工具，然后去除这些偏见。

### 年龄偏见

代码可能会根据年龄假设某些特定的能力或限制。例如，它可能建议某些适合特定年龄段的活动或产品，但并未涵盖那些能力超过其年龄表现的人群。

如果聊天机器人给你提供了这样的代码，你应当去除其中的年龄歧视偏见：

```py
def recommend_activities(user_age):
if user_age < 30:
return ["hiking", "rock climbing", "dancing"] elif user_age < 50:
return ["swimming", "cycling", "yoga"]
else:
return ["walking", "gardening", "reading"]
```

该代码假设某些活动仅适用于特定的年龄群体。例如，它建议老年人主要参与低强度活动，如散步和园艺，而年轻人应专注于更剧烈的活动，如远足和攀岩。这种偏见可能会延续关于衰老的刻板印象，并限制所有年龄段个体考虑的活动范围。

你可以通过更具包容性的方式来改进它，具体如下：

+   **个性化**：代码应考虑个人兴趣、身体状况和健康状况，而不是仅仅基于年龄做出广泛的概括。

+   **多样性**：推荐活动的列表应包括所有年龄段人群的更多选项，避免刻板印象。

+   **无障碍性**：代码应确保推荐的活动适合所有年龄段的人群，无论其身体限制如何。

### 残疾偏见

代码可能会排除或不利于残疾人士。例如，它可能对视觉或听力障碍、学习障碍、行动不便、语言障碍、光敏感等人群不够友好，或者没有兼顾这些人群的需求。

这样的代码可以通过首先熟悉**网页内容无障碍性指南**（**WCAG**）来改进。这是一套广泛认可的标准，旨在使网页内容对残疾人士更为无障碍。了解更多信息请访问：[`www.w3.org/TR/WCAG20/`](https://www.w3.org/TR/WCAG20/)。

也有许多无障碍性博客和网站，以及在线课程和书籍：[`usability.yale.edu/web-accessibility/articles/wcag2-checklist`](https://usability.yale.edu/web-accessibility/articles/wcag2-checklist)，[`www.wcag.com/category/blog/`](https://www.wcag.com/category/blog/)，和[`github.com/ediblecode/accessibility-resources`](https://github.com/ediblecode/accessibility-resources)。

使用无障碍性工具（如屏幕阅读器和颜色对比检查工具）以及无障碍性测试工具，如 Deque 的 Axe 和 Google Lighthouse：[`www.deque.com/axe/devtools/chrome-browser-extension`](https://www.deque.com/axe/devtools/chrome-browser-extension) 和 [`developer.chrome.com/docs/lighthouse/overview`](https://developer.chrome.com/docs/lighthouse/overview)。

这是另一个 Chrome 扩展：[`silktide.com/toolbar`](https://silktide.com/toolbar)。

### 社会经济偏见

代码可能假设某种经济或社会地位。例如，它可能会建议某些只对特定收入水平的人群可用的产品或服务。

为了改善或消除偏见，确保提示没有偏见（无论是隐性还是显性），并提供关于你所工作的特定社会经济群体的背景，以生成更具包容性的代码。通常，包含更多具体因素，如前面提到的。

### 文化偏见

代码可能会反映与文化规范或价值观相关的偏见。例如，它可能会建议某些行为或态度，这些行为或态度在一种文化中被视为合适，但在另一种文化中却不被接受；需要不断迭代和优化。

为了改善或消除偏见，仍然要使用代表不同文化视角的多种提示，并避免刻板印象。

了解这些潜在偏见并采取措施减轻它们非常重要。通过这样做，我们可以帮助确保 LLM 生成的代码是公平、公正和包容的。

## 分析训练数据

通常，像 GPT 和 Gemini 这样的系统，你无法访问 LLM 的训练数据，除非你是开发或数据团队的一员，负责开发这些 LLM。

在可能的情况下，特别是对于开放源代码的 LLM，如 MetaAI 或你自己的 LLM，识别训练数据中的偏见。LLM 是基于庞大的文本和代码数据集进行训练的。数据中的偏见可能会反映在模型的输出中。需要关注以下方面：

+   **代表性偏见**：数据是否能够代表现实世界，还是倾向于某一特定群体，导致其他群体的代表性不足？例如，如果训练数据仅包含高收入借款人，这可能会导致对低收入借款人和潜在借款人的理解偏差。

+   **历史偏见**：数据是否反映了过时或有偏见的观点？如果 LLM 仅仅在历史新闻文章上训练，它可能会建议过时的刻板印象，例如所有护士都是女性，或者种族偏见。

+   **测量偏见**：数据的收集或分类方式中是否存在隐性假设？自我报告的统计数据可能会存在很大偏差，标准化测试也可能存在偏见，因为它忽略了对测试的熟悉程度、焦虑和文化差异。

+   **特征值**：探索数据中的离群值或异常值。这些可能表明在数据收集过程中存在错误或偏见，从而影响模型的准确性。例如，曾经有自动化系统无法准确识别较深肤色的人群，如某些水龙头对黑人无效。很可能是因为他们的训练集未包含足够的深肤色人群的示例。另一个数据偏见是，标注性别的图像可能导致 AI 假设某些发型或衣物总是与特定性别相关。

并非所有的 LLM 都是开源的，因此无法检查训练数据。有关 Meta AI 训练数据的更多信息，请参见此链接：[`www.facebook.com/privacy/genai/`](https://www.facebook.com/privacy/genai/)。

## 检查代码

检测偏见的工具和方法包括以下几种：

+   **Fairlearn** ：由微软研究院开发的 Python 库，提供用于衡量和减轻机器学习模型偏见的度量和算法（[`fairlearn.org/`](https://fairlearn.org/)）

+   **IBM Watson OpenScale** ：一个提供用于监控和减轻 AI 模型偏见的工具的平台，包括公平性度量和偏见检测能力（[`dataplatform.cloud.ibm.com/docs/content/wsj/model/wos-setup-options.html?context=wx`](https://dataplatform.cloud.ibm.com/docs/content/wsj/model/wos-setup-options.html?context=wx)）

+   **相关性分析** ：检查模型预测与受保护属性（例如，种族和性别）之间的相关性，以识别潜在的偏见。

+   **差异性影响分析** ：评估模型是否对某些群体产生不成比例的影响。

+   **假设分析** ：生成反事实示例，以了解模型在不同情况下的表现，并识别潜在的偏见。

除了生成更好的代码和代码工具外，代码的人工审查始终是必需的，您还可以确保决策者来自多元背景。

社区参与对于发现偏见非常有用，特别是当社区拥有多样的社会经济背景以及多元的能力、种族、宗教、性别取向等时。

+   **审查假设** ：查看代码中是否存在潜在的假设，这些假设可能导致有偏的输出。

+   **查看硬编码值** ：例如，考虑一个情感分析程序，它对“中性”有默认设置。如果这与“正面”或“负面”相同，而没有仔细考虑，这一假设可能会导致情感分析的偏见。

+   **阈值** ：检查阈值是否引入了任何偏见，因为这是已知的偏见领域。例如，垃圾邮件检测器可能会计算感叹号的数量，并为垃圾邮件/非垃圾邮件设定一个阈值。

+   **数据转换** ：所使用的数据转换方法是否会无意中增加偏见？例如，考虑一个 LLM 图像识别程序，该程序在分类前对像素值进行标准化。如果标准化方法扭曲了某些颜色的呈现方式，可能会使得图像识别倾向于某些颜色的图像。

+   **注释和文档** ：检查注释和文档中是否揭示了任何偏见。可以通过查看在编写代码前所使用的假设来发现问题。

+   **检查算法选择**：所选择的算法可能会影响模型学习和解读数据的方式。选择的是 LLM、CNN、决策树还是逻辑回归？确保你了解所用方法的基本原理和假设，比如数据归一化以及如何处理异常值。考虑选择这种架构或机器学习方法是否可能加剧偏见。

+   **损失函数**：损失函数非常重要，决定了模型如何从训练数据中的错误中学习。

    如果损失函数仅包含准确度，那么对于难以建模或分类的样本，可能会被忽略，而偏向于处理更容易建模的大多数样本。

+   **优化策略**：优化策略通过调整算法来最小化损失。例如，在分类问题中，可能会出现类别不平衡的情况。假设类别 1 占样本的 80%。模型可能会非常擅长分类多数类（例如“正类”），而没有足够的资源去正确分类少数类（例如“负类”），后者只占 20%的数据，因此被认为在总体损失最小化中不那么重要。这可能导致假阳性，因为模型可能倾向于将所有样本分类为“正类”。

+   **可解释的代码**：如果你的代码设计得易于解释，或者使用了容易解释和检查的算法/方法，或者你有能够帮助你深入了解模型内部工作原理的工具，那么你和其他人就可以检查软件是否按预期工作，并确保不会产生偏见或带来技术或伦理问题。

### 公平性度量

有一些公平性工具和度量可以帮助识别潜在的偏见。以下是一些可以探索的度量：

+   **平等度量**：

    +   **准确度平衡**：该度量比较模型在不同组之间的总体准确度。一个公平的模型应该在所有组中具有相似的准确度。

    +   **召回平衡**：该度量比较每个组的**真实正例率**（**TPR**）。TPR 是正确识别的实际正例的比例。一个公平的模型应该在所有组或类别中具有相似的 TPR。

    +   **精确度平衡**：该度量比较每个组的**正预测值**（**PPV**）。PPV 是预测为正类的样本中，实际为正类的比例。一个公平的模型应该在所有组之间具有相似的 PPV。

+   **不同影响度度量**：

    +   **不平衡影响比率**（**DIR**）：该度量标准比较特定结果（例如，贷款拒绝）在一个群体与另一个群体中发生的频率。一个公平的模型应该具有接近 1 的 DIR，表示所有群体的结果相似。这有助于突出考虑人类时可能存在的年龄、性别、种族或收入偏见。在保护生物多样性方面，DIR 可以帮助发现某些物种在灭绝风险方面的分类是否正确。在农业中，偏差数据集可能导致最容易识别的害虫被优先考虑，从而忽视那些较难识别的害虫。DIR 在这里也能发挥作用。

+   **校准度量标准**：

    +   **校准平等**：该度量标准比较模型预测的结果概率与不同群体实际观察到的发生率之间的匹配情况。一个公平的模型应该为所有群体提供相似的校准。如果没有校准平等，你可能会发现医疗软件系统性地低估了某个特定种族患病的风险。

你可能还需要考虑如何选择合适的度量标准，以及这些度量标准和阈值的局限性（公平阈值）：

+   **选择合适的度量标准**：最合适的公平度量标准取决于特定任务和期望的结果。考虑对于你的应用而言，哪种类型的公平最为重要（例如，平等机会、平等损失）。

+   **度量标准的局限性**：公平度量标准可以是有用的工具，但它们并不是万无一失的。重要的是将它们与其他技术（如代码审查和人工评估）结合使用，以全面了解潜在的偏见。

+   **公平阈值**：没有适用于所有情况的统一公平度量标准。可接受的水平可能会根据上下文和潜在的偏差后果而有所不同 [*Gemini, HuggingFace_Fairness*]。

现在我们已经讨论了如何发现偏见或不道德的代码，接下来可以探讨如何从一开始就避免生成这样的代码。

接下来的部分将讨论如何防止不道德的代码产生，以及如何生成符合道德和无偏的代码。

# 防止偏见代码——以道德考量编写代码

希望现在你已经有足够的动力去输出尽可能无偏和公平的代码。以下是一些在创建无偏代码时需要考虑的事项。

## 获取良好的数据

首先，获取正确的数据。

在训练机器学习模型时，确保你使用的数据足够多样并能代表你希望服务的人群。如果你的数据存在偏差或不完整，你可能会从中得到偏见 [*ChatGPT*]。

## 道德准则

遵循你所在国家以及你计划部署代码的国家的法规。此外，遵循已建立的伦理准则和标准，如**计算机机械协会**（**ACM**）和**电气和电子工程师协会**（**IEEE**）提供的标准。相关资源分别可以在以下链接找到：[`www.acm.org/binaries/content/assets/membership/images2/fac-stu-poster-code.pdf`](https://www.acm.org/binaries/content/assets/membership/images2/fac-stu-poster-code.pdf) 和 [`www.ieee.org/about/corporate/governance/p7-8.html/`](https://www.ieee.org/about/corporate/governance/p7-8.html/)。

## 创建透明且易于解释的代码

使你的代码易于理解和跟踪。记录数据来源、训练方法和假设，以便更容易发现偏见和不公正。

使用描述性的变量名。记住关于可读性的章节，*第四章*。注释每个代码段的功能（或者你认为它的功能），但不要过多注释——仅在最有价值的地方注释。注释目的而非实现。这意味着告诉读者为什么这么做，而不是如何做，提供背景和理由。随着代码的变化，更新注释以反映变化，避免引起混淆。

良好地组织你的代码；通过将其拆分为每个具有单一简单目的的函数来模块化。每个函数都应该有描述性名称，以使代码库更易理解。代码的目的应该清晰，以至于不需要过多的注释来解释。

记录函数或方法的输入和输出，以及假设和约束。

如果你的组织有文档标准，请遵循那些标准。如果没有，使用社区的文档标准。

以下是各种语言和框架的文档标准和风格指南：

+   Python（*PEP 8 - Python 编程风格* *指南*）：[`peps.python.org/pep-0008/`](https://peps.python.org/pep-0008/)

+   Java（*Google Java 编程* *风格* *指南*）：[`google.github.io/styleguide/javaguide.html`](https://google.github.io/styleguide/javaguide.html)

+   JavaScript（*Airbnb JavaScript 风格* *指南*）：[`github.com/airbnb/javascript`](https://github.com/airbnb/javascript)

+   Ruby（*Ruby 编程* *风格* *指南*）：[`rubystyle.guide/`](https://rubystyle.guide/)

+   C++（*Google C++ 风格* *指南*）：[`google.github.io/styleguide/cppguide.html`](https://google.github.io/styleguide/cppguide.html)

+   C#（*Microsoft C# 编程* *规范*）：[`learn.microsoft.com/en-us/dotnet/csharp/fundamentals/coding-style/coding-conventions`](https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/coding-style/coding-conventions)

+   PHP（*PHP-FIG PSR-12 - 扩展编码风格* *指南*）：[`www.php-fig.org/psr/psr-12/`](https://www.php-fig.org/psr/psr-12/)

+   文档工具：

    +   [`www.sphinx-doc.org/en/master/`](https://www.sphinx-doc.org/en/master/)

    +   [`www.oracle.com/technical-resources/articles/java/javadoc-tool.html`](https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html)

代码审查也有助于使代码更易于理解和清晰。

## 代码审查

确保你制定一套明确一致的标准，团队达成共识，例如命名规范、文档、错误处理和安全性。在提交代码之前，分享代码风格指南，以便每个人都知道应该遵循哪些标准。

为了确保公正性，你还可以进行匿名代码审查，使用带有开放性问题的检查清单，当然也要提供有益的批评。

使用检查清单可以确保相关事项都被涵盖，且不会遗漏，除非你的团队没有充分创建检查清单。

开放式问题有助于帮助你理解这段代码的逻辑。

如果你不知道这段代码是谁写的，那么你就不能在审查时带有个人偏见：“ *我不喜欢这个人* ”，“ *我非常敬佩这个人，所以他们一定写得很好* ”，“ *我不能过多批评领导的代码* ”，等等[ *LinkedIn_fair_code_review* ]。

代码作者和审查员都会被匿名化，因此审查员也能避免在工作场所受到偏见的影响。

有益的批评是告诉别人如何改进，并提供具体、可操作的反馈，帮助他们推进自己的职业，而不是模糊或侮辱性的情感评论。

代码审查的目的是帮助每个人改进并持续产生高质量的代码，因此应该鼓励反馈和共同学习。

当然，你应该先检查自己的代码，然后再提交审查。避免出现尴尬的错误和遗漏。这里有一个很好的术语是*橡皮鸭调试*：在真正的人之前，先和你的橡皮鸭一起把代码讲一遍。通过这种方式，你会发现很多问题，尤其是当你心中有一个愿意帮助你的角色/橡皮鸭时。

你也可以让更多人参与审查，获取不同的观点和创意。寻求批评有助于你纠正错误，并更快地做出更聪明的决策。这就像是集体思维的进步；不要犯只靠自己做的错误。我已经犯过很多次，所以我知道那样效率低下且困难重重！

保持专业性并尊重他人。你不希望收到大量对你精心编写（或从 LLM 中整理出来）的代码的严厉情感批评，因此要帮助他人看到改进的地方，而不是过于苛刻[ *LinkedIn* ]。

## 你必然的成功

通过经验去发现偏见的来源并思考它，你很可能会在生成无偏代码方面变得更好，这样这个过程的速度也会加快。然而，世界上可能会揭示出你未曾听过或考虑过的新偏见，同时也会出现更好的工具来消除或避免偏见的产生。

记住，公平性应当是编码时的核心和首要任务，安全性也是如此。

你编写这段代码的目的是否可能是为了增加公平性？你的系统公平吗？

接下来是一个机会，看看在努力实现公正和有效时，什么时候做得比较好。

## 达到平衡的示例

虽然 Meta 确实开发了 Llama 3 AI，并宣称它有一个天才学校的孩子，但它也开发了一个工具，这个工具虽然不完全审查，但仍然在道德和伦理上基本合理。

Llama 2 经常拒绝执行它认为不道德的请求，例如被问到如何“打发时间”或关于可以用于爆炸物的核材料、如何格式化硬盘，甚至是关于某种性别或类型人物的笑话。

现在，如果你向 Llama 3 请求一些可能看起来不道德的内容，它通常会生成符合期望的回答，并且不会拒绝，但它不会提供如何制造武器或如何杀人的指导。Llama 3 会讨论这个话题并提供一些信息，但会避免提供危险或不道德的行为。

Llama 3 会告诉你如何格式化硬盘。虽然这可能是需要的，但它首先会给出警告，说明这会做什么并提醒你备份文件。

Llama 并不会回避告诉你有关男性的笑话，但据报道，对于某些问题的回答，在不同的人询问时是相同的。所以，有些回答可能是由人直接添加的，或者只是没有被过滤掉其中任何实际上冒犯或危险的内容 [*Llama3uncensored, Ollama*]。

# 总结

在这一章中，你了解了代码中的偏见和道德困境，包括 LLM 生成的代码。这一切从为什么我们需要关心偏见开始。接着我们看到了一些由偏见代码和其他偏见现象引发的公众尴尬和麻烦。本章探讨了如何检测偏见、衡量公平性以及如何防止糟糕代码的生成。涉及了获取平衡数据、公平对待数据、检查评论、说明假设、文档编写、广泛使用的文档、道德编码标准以及良好的代码审查。

本章中有链接到有用的资源。最后，我们看到了一个关于 LLM 运作良好的例子：既不偏见，也不过于限制。

在 *第六章*中，我们将讨论如何在法律框架下处理由 LLM 生成的代码。这将包括解开版权和知识产权问题，处理 LLM 生成代码的责任和义务，审视规范 LLM 在编码中使用的法律框架，以及可能的人工智能生成代码的未来监管。

# 参考文献

+   *Tag_in_text* ：404media: “Facebook 的 AI 告诉家长小组它有一个天才且残障的孩子”，Jason Koebler，[`www.404media.co/facebooks-ai-told-parents-group-it-has-a-disabled-child/`](https://www.404media.co/facebooks-ai-told-parents-group-it-has-a-disabled-child/)

+   *Art_for_a_change* : “Gemini：人工智能、危险与失败，” Mark Vallen, [`art-for-a-change.com/blog/2024/02/gemini-artificial-intelligence-danger-failure.html`](https://art-for-a-change.com/blog/2024/02/gemini-artificial-intelligence-danger-failure.html)

+   *ChatGPT* : ChatGPT, OpenAI, [`chat.openai.com/`](https://chat.openai.com/)

+   *Gemini* : Gemini 1.5, 谷歌, [`gemini.google.com`](https://gemini.google.com)

+   *Gemini1.5note* : “我们的下一代模型：Gemini 1.5，” Sundar Pichai, Demis Hassabis, [`blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#sundar-note`](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#sundar-note)

+   *HuggingFace_Fairness* : “衡量公平性，” Hugging Face, [`huggingface.co/spaces/merve/measuring-fairness`](https://huggingface.co/spaces/merve/measuring-fairness)

+   *LinkedIn_fair_code_review* : “你可以使用哪些方法来确保代码审查过程的公正性和无偏见？” LinkedIn, [`www.linkedin.com/advice/1/what-methods-can-you-use-ensure-fair-unbiased-4zooe`](https://www.linkedin.com/advice/1/what-methods-can-you-use-ensure-fair-unbiased-4zooe)

+   *Llama3uncensored* : “Llama-3 其实并没有受到很严格的审查，” Llama, [`llama-2.ai/llama-3-censored/`](https://llama-2.ai/llama-3-censored/)

+   *Meta_Llama_3* : “介绍 Meta Llama 3：迄今为止最强大的公开可用的大型语言模型，” Meta AI, [`ai.meta.com/blog/meta-llama-3/`](https://ai.meta.com/blog/meta-llama-3/)

+   *Ollama* : “Llama 3 并不是经过严格审查的，” Ollama, [`ollama.com/blog/llama-3-is-not-very-censored`](https://ollama.com/blog/llama-3-is-not-very-censored)

+   *PapersExplained105* : “论文解读 105：Gemini 1.5 Pro，” Ritvik Rastogi, [`ritvik19.medium.com/papers-explained-105-gemini-1-5-pro-029bbce3b067`](https://ritvik19.medium.com/papers-explained-105-gemini-1-5-pro-029bbce3b067)

+   *Sky_MetaAI* : “Meta 的 AI 告诉 Facebook 用户它已禁用其天才儿童，作为回应家长寻求建议，” Mickey Carroll, [`news.sky.com/story/metas-ai-tells-facebook-user-it-has-disabled-gifted-child-in-response-to-parent-asking-for-advice-13117975`](https://news.sky.com/story/metas-ai-tells-facebook-user-it-has-disabled-gifted-child-in-response-to-parent-asking-for-advice-13117975)

+   *TechReportGemini1.5* : “谷歌现在拥有最好的 AI，但也有一个问题……”， Fireship, [`youtu.be/xPA0LFzUDiE`](https://youtu.be/xPA0LFzUDiE)

+   *The_Intercept* : “互联网的新宠 AI 提议折磨伊朗人并监视清真寺，” Sam Biddle, [`theintercept.com/2022/12/08/openai-chatgpt-ai-bias-ethics/`](https://theintercept.com/2022/12/08/openai-chatgpt-ai-bias-ethics/), 2022

+   *Voiid* : “Gemini 被指控对白人有种族偏见，” 编辑团队, [`voi.id/en/technology/358972`](https://voi.id/en/technology/358972)
