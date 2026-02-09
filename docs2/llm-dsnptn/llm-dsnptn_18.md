# 18

# 对抗鲁棒性

LLMs 上的**对抗攻击**旨在通过在输入中做出微小、通常难以察觉的改变来操纵模型的输出。这些攻击可以暴露 LLMs 中的漏洞，并可能导致现实应用中的安全风险或意外行为。

在本章中，我们将探讨在 LLMs 中创建和防御**对抗样本**的技术。对抗样本是精心设计的输入，旨在故意误导模型产生错误或意外的输出。您将了解文本对抗攻击、生成这些示例的方法以及使您的模型更具鲁棒性的技术。我们还将介绍评估方法，并讨论对抗攻击对 LLMs 在现实世界中的影响。

在本章中，我们将涵盖以下主题：

+   文本对抗攻击的类型

+   对抗训练技术

+   评估鲁棒性

+   LLMs 对抗训练中的权衡

+   现实世界的影响

# 文本对抗攻击的类型

文本对抗攻击涉及微妙地修改文本输入以误导模型做出错误预测，通常这些变化对人类来说是不可察觉的。

有三个级别的攻击需要考虑：

+   **字符级攻击**：修改单词中的单个字符

+   **词级攻击**：用同义词或类似词替换单词

+   **句子级攻击**：修改句子结构同时保留意义

此外，还有一种需要考虑的第四种攻击类型，涉及**通用对抗触发器**，这些是特定的短语，当添加时，会导致模型在各种输入中表现不正确。

这里有一些通用的对抗触发器的示例：

+   **对于情感分析**：想象一个训练有素的情感分析模型，用于将电影评论分类为“正面”或“负面”。一个通用的对抗触发器可能是一个看似无意义的短语，例如“无形地毯笔记本”。如果将此触发器添加到各种正面评论中，例如“这部电影太棒了”，得到“这部电影太棒了无形地毯笔记本”，模型可能会错误地将它们分类为“负面”。同样，将相同的触发器添加到其他正面评论，例如“我真的很喜欢表演”，也可能将其预测翻转到“负面”。

+   **对于文本分类（例如，主题识别）**：考虑一个将新闻文章分类为“体育”、“政治”或“科技”等类别的模型。一个通用的对抗触发器可能是“the the grand elephant”。将此触发器添加到一篇“体育”文章，例如“该队赢得了冠军”，得到“该队赢得了冠军 the the grand elephant”，可能会使模型将其误分类为“政治”。同样，将其添加到一篇关于新小工具的“科技”文章也可能导致错误的分类。

+   **对于自然语言推理（NLI）**：在 NLI 中，模型确定两个句子之间的关系（例如，*蕴涵*、*矛盾*、*中立*）。一个触发器，如“总是快速跳舞”，可能被设计成始终强制进行“蕴涵”预测。例如，给定前提“一个人在走路”，和假设“一个人在移动”，模型会正确预测“蕴涵”。然而，如果触发器被添加到前提中，“一个人在走路总是快速跳舞”，模型可能会被欺骗仍然预测“蕴涵”，即使关系变得不那么清晰甚至荒谬。

为了简单起见，在本节中，我们将讨论两种类型的攻击。首先，让我们实现一个简单的字符级攻击：

```py
import random
import string
def character_level_attack(text, prob=0.1):
    def modify_char(c):
        if random.random() < prob:
            return random.choice(string.ascii_letters) if c.isalpha() else c
        return c
    return ''.join(modify_char(c) for c in text)
# Example usage
original_text = "The quick brown fox jumps over the lazy dog."
attacked_text = character_level_attack(original_text)
print(f"Original: {original_text}")
print(f"Attacked: {attacked_text}")
```

此代码定义了一个`character_level_attack`函数，旨在通过随机修改单个字符来创建输入文本的略微修改版本。对于输入文本中的每个字符，都有一个概率（由`prob`参数设置，默认为`0.1`），它将被更改。如果一个字符被选中进行修改，并且它是一个字母字符，它将被替换为一个随机的小写或大写字母。非字母字符（如空格和标点符号）保持不变。然后函数将可能被修改的字符重新组合成一个字符串，生成“攻击”文本。

此代码的输出将显示两行。第一行，标记为“原始：”，将显示初始输入文本：“The quick brown fox jumps over the lazy dog。”。第二行，标记为“攻击：”，将展示修改后的文本。由于基于`prob`值的字符替换是随机的，因此“攻击：”文本可能会将其一些字母字符替换为其他随机字母。例如，“The”可能变成“Tge”，“quick”可能是“quicj”，等等。这些变化的数量和具体位置每次执行代码时都会变化，因为随机选择过程。

接下来，作为另一个例子，让我们实现一个更复杂的词级攻击，使用同义词替换：

```py
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
def get_synonyms(word, pos):
    synonyms = set()
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            synonyms.update(
                lemma.name()
                for lemma in syn.lemmas()
                if lemma.name() != word
        )
    return list(synonyms)
```

此函数根据给定单词的词性检索其同义词。它使用 WordNet，一个英语语言的词汇数据库，来查找同义词，同时确保它们与原词不同。

现在，让我们实现一个词级攻击：

```py
def word_level_attack(text, prob=0.2):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    attacked_words = []
    for word, pos in pos_tags:
        if random.random() < prob:
            wordnet_pos = {'NN': 'n', 'JJ': 'a', 'VB': 'v',
                'RB': 'r'}.get(pos[:2])
            if wordnet_pos:
                synonyms = get_synonyms(word, wordnet_pos)
                if synonyms:
                    attacked_words.append(random.choice(synonyms))
                    continue
        attacked_words.append(word)
    return ' '.join(attacked_words)
# Example usage
original_text = "The intelligent scientist conducted groundbreaking research."
attacked_text = word_level_attack(original_text)
print(f"Original: {original_text}")
print(f"Attacked: {attacked_text}")
```

此代码片段定义了一个函数 `word_level_attack`，该函数尝试通过随机替换一些单词的同义词来创建输入文本的微妙修改版本。它首先将输入文本标记化成单个单词，然后为每个单词确定词性（POS）标签。对于每个单词，有一个概率（由 `prob` 参数设置，默认为 `0.2`）表示该单词将被选中进行替换。如果选中单词，则使用其词性标签从 WordNet 词汇数据库中查找潜在的同义词。如果找到同义词，则随机同义词替换输出中的原始单词；否则，保留原始单词。

此代码的输出将显示两行。第一行，标记为 `"Original:"`，将显示初始输入文本："The intelligent scientist conducted groundbreaking research."。第二行，标记为 `"Attacked:"`，将展示修改后的文本。由于基于概率值进行单词替换的随机性，`"Attacked:"` 文本可能会用同义词替换一些单词。例如，"intelligent" 可能会被替换为 "smart" 或 "clever"，"conducted" 可能会被替换为 "carried_out" 或 "did"，而 "groundbreaking" 可能会被替换为 "innovative" 或 "pioneering"。具体的更改每次执行代码时都会有所不同，因为单词及其同义词的选择是随机的。

# 对抗训练技术

对抗训练涉及在训练过程中向模型展示对抗性示例，以提高其鲁棒性。以下是一个简化的示例，说明您如何为 LLM 实现对抗训练：

```py
import torch
def adversarial_train_step(model, inputs, labels, epsilon=0.1):
    embeds = model.get_input_embeddings()(inputs["input_ids"])
    embeds.requires_grad = True
    outputs = model(inputs, inputs_embeds=embeds)
    loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
    loss.backward()
    perturb = epsilon * embeds.grad.detach().sign()
    adv_embeds = embeds + perturb
    adv_outputs = model(inputs_embeds=adv_embeds)
    adv_loss = torch.nn.functional.cross_entropy(
        adv_outputs.logits, labels
    )
    return 0.5 * (loss + adv_loss)
```

此函数执行一次对抗训练步骤。它使用 **Fast Gradient Sign Method**（**FGSM**）生成对抗性扰动，并合并干净和对抗输入的损失。FGSM 是一种单步对抗攻击，通过计算损失函数相对于输入数据的梯度并添加一个小的扰动（在梯度的符号方向上）来有效地生成对抗性示例。这个扰动通过一个小 epsilon 缩放，旨在最大化模型的预测误差，导致错误分类，同时对人类几乎不可察觉。

要在完整训练循环中使用此功能，请使用以下函数：

```py
def adversarial_train(
    model, train_dataloader, optimizer, num_epochs=3
):
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, labels = batch
            loss = adversarial_train_step(model, inputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
```

此函数遍历训练数据，为每个批次执行对抗训练步骤。它使用干净和对抗输入的合并损失来更新模型参数。

# 评估鲁棒性

为了评估 LLM 的鲁棒性，我们可以测量其在干净和对抗输入上的性能：

```py
def evaluate_robustness(
    model, tokenizer, test_dataset, attack_function
):
    model.eval()
    clean_preds, adv_preds, labels = [], [], []
    for item in test_dataset:
        inputs = tokenizer(item['text'], return_tensors='pt',
            padding=True, truncation=True)
        with torch.no_grad():
            clean_output = model(inputs).logits
        clean_preds.append(torch.argmax(clean_output, dim=1).item())
        adv_text = attack_function(item['text'])
        adv_inputs = tokenizer(adv_text, return_tensors='pt',
            padding=True, truncation=True
        )
        with torch.no_grad():
            adv_output = model(adv_inputs).logits
        adv_preds.append(torch.argmax(adv_output, dim=1).item())
        labels.append(item['label'])
    return calculate_metrics(labels, clean_preds, adv_preds)
```

此函数评估模型在干净和对抗攻击输入上的性能。它处理测试数据集中的每个项目，为输入的原始版本和攻击版本生成预测。

你还应该计算评估指标：

```py
from sklearn.metrics import accuracy_score, f1_score
def calculate_metrics(labels, clean_preds, adv_preds):
    return {
        'clean_accuracy': accuracy_score(labels, clean_preds),
        'adv_accuracy': accuracy_score(labels, adv_preds),
        'clean_f1': f1_score(labels, clean_preds, average='weighted'),
        'adv_f1': f1_score(labels, adv_preds, average='weighted')
    }
```

提供的 Python 代码定义了一个名为 `calculate_metrics` 的函数，该函数接受三个参数：测试数据的真实标签、模型在原始（干净）测试数据上的预测，以及模型在对抗攻击版本的测试数据上的预测。在函数内部，它使用来自 `sklearn.metrics` 库的 `accuracy_score` 和 `f1_score` 函数来计算四个关键评估指标：

+   模型在干净数据上的预测准确性 (`clean_accuracy`)

+   在对抗数据上的准确性 (`adv_accuracy`)

+   干净数据上的加权 F1 分数 (`clean_f1`)

+   对抗数据上的加权 F1 分数 (`adv_f1`)

函数随后将这四个分数作为字典返回，其中每个指标的名称是键，其计算值是对应的值。

每个计算出的分数都从不同角度反映了模型的表现。准确性表示在总实例数中正确分类的实例比例。干净数据上的高准确性表明模型在原始、未受干扰的输入上表现良好，而低准确性则表明整体性能较差。相反，对抗数据上的高准确性意味着模型对所使用的特定攻击具有鲁棒性，这意味着攻击在欺骗模型方面不是很有效。尽管干净准确性可能很高，但对抗数据上的低准确性突显了模型对这些攻击的脆弱性。F1 分数，尤其是这里使用的加权版本，用于考虑潜在的类别不平衡，提供了一个平衡的精确度和召回率的度量。干净数据上的高 F1 分数表示在正确识别正实例和避免假阳性方面表现良好。同样，对抗数据上的高 F1 分数表明鲁棒性，因为即使在攻击下，模型也保持了良好的精确度和召回率。干净或对抗数据上的低 F1 分数表明模型在这些相应条件下在精确度或召回率，或两者方面都有所挣扎。比较干净和对抗分数揭示了攻击降低模型性能的程度；显著的下降表明鲁棒性不足。

# LLMs 对抗训练中的权衡

对抗训练可以提高模型的鲁棒性，但通常伴随着权衡：

+   **增加的计算成本**：在训练期间生成对抗示例是计算密集型的。

+   **潜在降低的干净准确性**：专注于对抗鲁棒性可能会略微降低干净输入的性能。

+   **推广到未见过的攻击**：模型可能对特定类型的攻击具有鲁棒性，但对其他攻击仍然脆弱。

为了可视化这些权衡，你可以创建一个比较不同对抗训练水平下的干净和对抗准确性的图表：

```py
import matplotlib.pyplot as plt
def plot_robustness_tradeoff(
    clean_accuracies, adv_accuracies, epsilon_values
):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, clean_accuracies, label='Clean Accuracy')
    plt.plot(epsilon_values, adv_accuracies,
        label='Adversarial Accuracy')
    plt.xlabel('Epsilon (Adversarial Perturbation Strength)')
    plt.ylabel('Accuracy')
    plt.title('Robustness Trade-off in Adversarial Training')
    plt.legend()
    plt.show()
```

此函数创建一个图表，可视化增加对抗性训练强度（epsilon）如何影响干净和对抗性准确性。

# 现实世界影响

理解对抗性攻击对 LLM 的现实世界影响对于负责任地部署至关重要：

+   **安全风险**：对抗性攻击可能被用于绕过内容过滤器或在安全关键应用中操纵模型输出

+   **虚假信息**：攻击者可能使用对抗性技术生成虚假新闻或误导性内容，从而逃避检测系统

+   **用户信任**：如果 LLM 容易被对抗性输入欺骗，可能会侵蚀用户对 AI 系统的信任

+   **法律和伦理问题**：操纵 LLM 输出的能力引发了关于 AI 驱动决策中责任和问责的伦理问题

+   **在多样化环境中的鲁棒性**：真实世界部署 LLM 需要评估它们在多样化不利条件下的性能，而不仅仅是依赖于干净的实验室环境

为了应对这些影响，考虑实施鲁棒的部署实践和红队演习：

```py
class RobustLLMDeployment:
    def __init__(self, model, tokenizer, attack_detector):
        self.model = model
        self.tokenizer = tokenizer
        self.attack_detector = attack_detector
    def process_input(self, text):
        if self.attack_detector(text):
            return "Potential adversarial input detected. Please try again."
        inputs = self.tokenizer(
            text, return_tensors='pt', padding=True,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model(inputs)
        return self.post_process_output(outputs)
    def post_process_output(self, outputs):
        # Implement post-processing logic here
        pass
    def log_interaction(self, input_text, output_text):
        # Implement logging for auditing and monitoring
        pass
```

此类封装了部署鲁棒 LLM 的最佳实践，包括输入验证、攻击检测和输出后处理。

# 摘要

在 LLM 中解决对抗性鲁棒性对于它们在现实世界应用中的安全可靠部署至关重要。通过实施本章讨论的技术和考虑因素，你可以朝着开发出对对抗性攻击更具弹性同时保持对干净输入高性能的 LLM 迈进。

在下一章中，我们将探讨 LLM 训练中的**人类反馈强化学习**（**RLHF**）。
