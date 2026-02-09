

# 第十七章：公平性与偏见检测

在 LLM 中，公平性涉及确保模型的结果和决策不会基于受保护属性（如种族、性别、年龄或宗教）歧视或不公平对待个人或群体。这是一个复杂的概念，而不仅仅是避免显性偏见。

机器学习中公平性的定义有几个：

+   **人口统计学平等性**：所有群体获得积极结果的概率应该是相同的

+   **平等机会**：所有群体的真正阳性率应该是相同的

+   **均衡机会**：所有群体的真正阳性和假阳性率应该是相同的

对于大型语言模型（LLM），公平性通常涉及确保模型的语言生成和理解能力在不同人口群体之间是公平的，并且不会持续或放大社会偏见。

在本章中，你将了解 LLM 中可能出现的不同类型的偏见以及检测它们的技巧。

在本章中，我们将讨论以下主题：

+   偏见的类型

+   LLM 文本生成和理解公平性指标

+   检测偏见

+   去偏见策略

+   公平性意识训练

+   伦理考量

# 偏见的类型

LLM 可以表现出各种类型的偏见：

+   **代表性偏见**：在训练数据中对某些群体的代表性不足或错误表示——例如，主要在浅色皮肤的面部上进行训练的面部识别系统在识别深色皮肤色调的人时可能表现出显著更高的错误率，这是由于训练集中代表性不足。

+   **语言偏见**：人工智能系统用于描述不同群体的语言——例如，一个 AI 系统可能会将男性标记为“自信”而将女性标记为“在性别间具有攻击性”，当描述相同的行为时，这会强化细微的歧视模式。

+   **分配偏见**：基于模型预测的资源或机会的不公平分配，如自动化招聘系统系统性地将某些大学的候选人排名更高，而不管他们的资格如何，从而不成比例地将面试机会分配给这些机构的毕业生。

+   **服务质量偏见**：模型在不同群体之间的性能差异，如一个机器翻译系统为英语、西班牙语和普通话等主流语言提供更准确的翻译，而为使用人数较少或训练数据中代表性较低的语言提供较低质量的翻译。

+   **刻板印象偏见**：通过语言生成强化社会刻板印象，例如当人工智能写作助手在完成关于不同背景角色的故事时自动建议刻板印象的职业道路——为某些种族背景的角色建议体育或娱乐职业，而为其他人建议医生或律师等职业。

+   **显性和隐性偏见**: LLM 中的显性偏见源于训练数据中的明显模式，例如文本来源中存在的刻板印象，导致输出中明显可识别的偏见。另一方面，隐性偏见更为微妙，源于数据中的潜在统计相关性，以可能加强隐藏偏见的方式塑造响应，而没有直接意图。虽然显性偏见通常可以通过过滤或微调来检测和缓解，但隐性偏见更难识别，需要更深入的措施，例如偏见感知训练技术和对模型输出的定期审计。

+   **隐藏偏见**: 当训练数据、模型设计或部署选择微妙地扭曲响应，加强刻板印象或排除观点时，LLM 中的隐藏偏见就会出现。这可以表现为性别语言、文化偏好或政治倾向，通常是由于训练数据中过度代表的观点。算法处理可以进一步放大这些偏见，使响应根据提示语句不一致或偏斜。为了缓解这种情况，需要多样化的数据集、偏见审计和道德微调，确保模型在道德约束内生成平衡和公平的输出，同时允许用户在道德约束内进行感知调整。

这里有一个检查数据集中表示偏见的例子（我们将只展示一个例子以限制本章的篇幅）：

```py
import pandas as pd
from collections import Counter
def analyze_representation(texts, attribute_list):
    attribute_counts = Counter()
    for text in texts:
        for attribute in attribute_list:
            if attribute.lower() in text.lower():
                attribute_counts[attribute] += 1
    total = sum(attribute_counts.values())
    percentages = {attr: count/total*100
        for attr, count in attribute_counts.items()}
    return pd.DataFrame({
        'Attribute': percentages.keys(),
        'Percentage': percentages.values()
    }).sort_values('Percentage', ascending=False)
# Example usage
texts = [
    "The CEO announced a new policy.",
    "The nurse took care of the patient.",
    "The engineer designed the bridge.",
    # ... more texts
]
gender_attributes = ['he', 'she', 'his', 'her', 'him', 'her']
representation_analysis = analyze_representation(
    texts, gender_attributes
)
print(representation_analysis)
```

这段代码分析了一个文本语料库中性别相关术语的表示，这有助于识别数据集中潜在的性别偏见。

# LLM 文本生成和理解中的公平性指标

公平性指标通常关注比较不同人口群体之间的模型性能或输出。

这里有一些例子：

+   **文本分类中的人口统计学差异**: 该指标衡量了最被青睐的群体和最不受青睐的群体之间正预测率的差异：

    ```py
    from sklearn.metrics import confusion_matrix
    import numpy as np
    def demographic_parity_difference(
        y_true, y_pred, protected_attribute
    ):
        groups = np.unique(protected_attribute)
        dps = []
        for group in groups:
            mask = protected_attribute == group
            cm = confusion_matrix(y_true[mask], y_pred[mask])
            dp = (cm[1, 0] + cm[1, 1]) / cm.sum()
            dps.append(dp)
        return max(dps) - min(dps)
    # Example usage
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 1, 1]
    protected_attribute = ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
    dpd = demographic_parity_difference(
        y_true, y_pred, protected_attribute
    )
    print(f"Demographic Parity Difference: {dpd}")
    ```

    该代码定义了一个 `demographic_parity_difference` 函数，该函数计算由受保护属性定义的群体之间的人口统计学差异。它接受真实标签 (`y_true`)、预测标签 (`y_pred`) 和受保护属性值作为输入。对于受保护属性中的每个唯一群体，它创建一个布尔掩码来隔离相应的预测子集，并计算该组的混淆矩阵。然后，每个群体的人口统计学差异（DP）被计算为该组所有预测中正预测（无论是误分类还是正确分类）的比例，具体使用 `(cm[1, 0] + cm[1, 1]) / cm.sum()`，这对应于实际正数（无论是误分类还是正确分类）的数量除以总数。它存储这些 DP 值，并最终返回它们之间的最大差异，这表明了跨组之间的待遇差异。示例使用虚拟数据演示了这一点，打印出 `'A'` 和 `'B'` 组之间的 DP 差异。

+   **文本分类的平等机会差异**：此指标衡量最被青睐的群体和最不受青睐的群体之间真正阳性率的差异：

    ```py
    def equal_opportunity_difference(
        y_true, y_pred, protected_attribute
    ):
        groups = np.unique(protected_attribute)
        tprs = []
        for group in groups:
            mask = (protected_attribute == group) & (y_true == 1)
            tpr = np.mean(y_pred[mask] == y_true[mask])
            tprs.append(tpr)
        return max(tprs) - min(tprs)
    # Example usage
    eod = equal_opportunity_difference(y_true, y_pred,
        protected_attribute)
    print(f"Equal Opportunity Difference: {eod}")
    ```

    此代码计算由受保护属性定义的群体之间的真正阳性率差异，衡量模型在那些群体中正确识别阳性案例的平等程度。

现在我们已经探索了几种衡量模型输出和推理能力的公平性指标，接下来我们将继续学习实际检测偏差的技术，基于这些指标来开发系统性的测试方法。

# 检测偏差

在大型语言模型（LLMs）中检测偏差通常涉及分析不同人口群体或不同类型输入下的模型输出。以下是一些技术：

+   **词嵌入**：此代码通过比较职业词汇在性别方向上的投影来衡量词嵌入中的性别偏差：

    ```py
    from gensim.models import KeyedVectors
    import numpy as np
    def word_embedding_bias(
        model, male_words, female_words, profession_words
    ):
        male_vectors = [model[word] for word in male_words if word in model.key_to_index]
        female_vectors = [model[word] for word in female_words
            if word in model.key_to_index]
        male_center = np.mean(male_vectors, axis=0)
        female_center = np.mean(female_vectors, axis=0)
        gender_direction = male_center - female_center
        biases = []
        for profession in profession_words:
            if profession in model.key_to_index:
                bias = np.dot(model[profession], gender_direction)
                biases.append((profession, bias))
        return sorted(biases, key=lambda x: x[1], reverse=True)
    # Example usage
    model = KeyedVectors.load_word2vec_format(
        'path_to_your_embeddings.bin', binary=True
    )
    male_words = ['he', 'man', 'boy', 'male', 'gentleman']
    female_words = ['she', 'woman', 'girl', 'female', 'lady']
    profession_words = ['doctor', 'nurse', 'engineer', 'teacher',
        'CEO']
    biases = word_embedding_bias(
        model, male_words, female_words, profession_words
    )
    for profession, bias in biases:
        print(f"{profession}: {bias:.4f}")
    ```

    此代码通过首先为男性和女性术语创建平均向量，计算它们之间的性别方向向量，然后通过点积计算来衡量不同职业词汇与该性别轴的接近程度，从而衡量词嵌入中的性别偏差。该函数按偏差分数对职业进行排序，其中正值表示男性关联，负值表示女性关联，使用户能够量化嵌入在语言模型中的性别刻板印象。

+   **情感分析**：您可以分析不同群体中的情感以检测潜在的偏差：

    ```py
    from transformers import pipeline
    def analyze_sentiment_bias(
        texts, groups,
    model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        sentiment_analyzer = pipeline(
            "sentiment-analysis", model=model_name
        )
        results = {group: {'positive': 0, 'negative': 0}
            for group in set(groups)}
        for text, group in zip(texts, groups):
            sentiment = sentiment_analyzer(text)[0]
            results[group][sentiment['label'].lower()] += 1
        for group in results:
            total = results[group]['positive'] \
                + results[group]['negative']
            results[group]['positive_ratio'] = \
                results[group]['positive'] / total
        return results
    # Example usage
    texts = [
        "The man is very intelligent.",
        "The woman is very intelligent.",
        "The man is a great leader.",
        "The woman is a great leader.",
    ]
    groups = ['male', 'female', 'male', 'female']
    bias_results = analyze_sentiment_bias(texts, groups)
    print(bias_results)
    ```

    此代码通过使用来自`transformers`库的预训练情感分析模型，分析不同人口群体中的情感偏差。它接受一个文本列表及其相应的群体标签，通过情感分析器处理每个文本，并计算每个群体的正面和负面情感计数。然后，该函数为每个群体计算“正面比率”（被分类为正面的文本比例），允许比较不同群体之间的情感分布。在示例中，它特别通过分析关于智力和领导力的相同陈述在归因于男性还是女性时如何被分类来检查潜在的性别偏差，这可能揭示底层语言模型是否根据性别关联对相同品质进行不同的处理。

+   **指代消解**：您可以分析指代消解来检测潜在的职业-性别偏差：

    ```py
    import spacy
    def analyze_coreference_bias(texts, occupations, genders):
        nlp = spacy.load("en_core_web_sm")
        results = {gender: {occ: 0 for occ in occupations}
            for gender in genders}
        counts = {gender: 0 for gender in genders}
        for text in texts:
            doc = nlp(text)
            occupation = None
            gender = None
            for token in doc:
                if token.text.lower() in occupations:
                    occupation = token.text.lower()
                if token.text.lower() in genders:
                    gender = token.text.lower()
            if occupation and gender:
                results[gender][occupation] += 1
                counts[gender] += 1
        for gender in results:
            for occ in results[gender]:
                results[gender][occ] /= counts[gender]
        return results
    # Example usage
    texts = [
        "The doctor examined her patient. She prescribed some medication.",
        "The nurse took care of his patients. He worked a long shift.",
        # ... more texts
    ]
    occupations = ['doctor', 'nurse', 'engineer', 'teacher']
    genders = ['he', 'she']
    bias_results = analyze_coreference_bias(texts, occupations,
        genders)
    print(bias_results)
    ```

    代码定义了一个`analyze_coreference_bias`函数，该函数使用 spaCy 的 NLP 管道通过分析特定性别化代词（如“他”和“她”）与某些职业（例如，“医生”，“护士”）共现的频率来评估文本中的潜在性别偏见。它初始化一个 spaCy 语言模型，创建一个嵌套字典来计算每个性别-职业对的频率，以及每个性别的单独计数。对于每个输入文本，它将内容分词，确定是否有任何预定义的职业和性别化代词出现，如果两者都存在，则增加相关计数器。处理完所有文本后，它通过性别提及的总数对每个性别的职业计数进行归一化，从而有效地得到一个反映给定数据集中每个职业与每个性别相对关联比例的结果。该函数返回这个归一化结果，然后在示例用法中打印出来。

接下来，我们将基于这种检测知识来探讨减少偏差的实际策略，帮助我们从诊断转向治疗。

# 去除偏差策略

去除偏差的 LLM 是一个活跃的研究领域。以下是一些策略：

+   **数据增强**（见*第三章*）：在以下代码中，我们通过交换性别化的词汇来增强数据集，帮助平衡性别代表性：

    ```py
    import random
    def augment_data(texts, male_words, female_words):
        augmented_texts = []
        for text in texts:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in male_words:
                    female_equivalent = female_words[
                        male_words.index(word.lower())
                    ]
                    new_text = ' '.join(words[:i]
                        + [female_equivalent] + words[i+1:])
                    augmented_texts.append(new_text)
                elif word.lower() in female_words:
                    male_equivalent = male_words[
                        female_words.index(word.lower())
                    ]
                    new_text = ' '.join(words[:i]
                        + [male_equivalent] + words[i+1:])
                    augmented_texts.append(new_text)
        return texts + augmented_texts
    # Example usage
    texts = [
        "The doctor examined his patient.",
        "The nurse took care of her patients.",
    ]
    male_words = ['he', 'his', 'him']
    female_words = ['she', 'her', 'her']
    augmented_texts = augment_data(texts, male_words, female_words)
    print(augmented_texts)
    ```

+   **偏差微调**：在以下代码中，我们微调一个语言模型，用更中性的替代词替换有偏见的词汇：

    ```py
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer)
    import torch
    def create_debiasing_dataset(biased_words, neutral_words):
        inputs = [f"The {biased} person" for biased in biased_words]
        targets = [f"The {neutral} person"
            for neutral in neutral_words]
        return inputs, targets
    def fine_tune_for_debiasing(
        model, tokenizer, inputs, targets, epochs=3
    ):
        input_encodings = tokenizer(inputs, truncation=True,
            padding=True)
        target_encodings = tokenizer(targets, truncation=True,
            padding=True)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(input_encodings['input_ids']),
            torch.tensor(input_encodings['attention_mask']),
            torch.tensor(target_encodings['input_ids'])
        )
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        return model
    # Example usage
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    biased_words = ['bossy', 'emotional', 'hysterical']
    neutral_words = ['assertive', 'passionate', 'intense']
    inputs, targets = create_debiasing_dataset(
        biased_words, neutral_words
    )
    debiased_model = fine_tune_for_debiasing(
        model, tokenizer, inputs, targets
    )
    ```

# 公平性感知训练

机器学习中的公平性约束是数学公式，通过确保模型预测在不同人口群体中保持所需的统计特性，来量化并强制执行特定的公平性概念。这些约束通常表达条件，如人口比例（组间相同的正面预测率）、均衡机会（相同的真正率和假正率）或个人公平（相似的个人收到相似的预测）。它们可以直接作为正则化项纳入模型优化，或作为后处理步骤强制执行。通过明确建模这些约束，开发者可以减轻算法偏差，并确保在受保护属性（如种族、性别或年龄）方面有更公平的结果——在准确性的传统目标与预测系统对不同群体影响的相关伦理考量之间取得平衡。

将公平性约束直接纳入训练过程可以帮助产生更公平的模型。以下是一个简化的例子：

```py
import torch
import torch.nn as nn
import torch.optim as optim
class FairClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FairClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
def fair_loss(
    outputs, targets, protected_attributes, lambda_fairness=0.1
):
    criterion = nn.CrossEntropyLoss()
    task_loss = criterion(outputs, targets)
    # Demographic parity
    group_0_pred = outputs[protected_attributes == 0].mean(dim=0)
    group_1_pred = outputs[protected_attributes == 1].mean(dim=0)
    fairness_loss = torch.norm(group_0_pred - group_1_pred, p=1)
    return task_loss + lambda_fairness * fairness_loss
def train_fair_model(
    model, train_loader, epochs=10, lr=0.001,
    lambda_fairness=0.1
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for inputs, targets, protected_attributes in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = fair_loss(
                outputs, targets,
                protected_attributes, lambda_fairness
            )
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    return model
# Example usage (assuming you have prepared your data)
input_size = 10
hidden_size = 50
num_classes = 2
model = FairClassifier(input_size, hidden_size, num_classes)
train_loader = ...  # Your DataLoader here
fair_model = train_fair_model(model, train_loader)
```

此代码实现了一个神经网络分类器，旨在对受保护属性（如种族或性别）保持公平。`FairClassifier`类定义了一个简单的两层神经网络，而`fair_loss`函数将标准分类损失与一个公平约束相结合，当预测在不同人口群体之间有差异时，对模型进行惩罚。`train_fair_model`函数处理训练循环，应用这种组合损失来优化模型参数，同时平衡准确性和公平性。

通过在损失函数中（按`lambda_fairness`加权）引入公平性惩罚项，模型被明确训练以在不同受保护群体之间做出相似的预测，从而解决潜在的偏见。这代表了一种“基于约束”的公平机器学习方法，其中公平性目标直接纳入优化过程，而不是作为后处理步骤应用。可以通过`lambda_fairness`超参数调整任务性能与公平性之间的权衡。

# 道德考虑因素

开发公平且无偏见的 LLMs 不仅是一个技术挑战，也是一个道德上的必要。以下是一些关键的道德考虑因素：

+   **透明度**：对模型的局限性和潜在偏见保持开放。

+   **多元化的开发团队**：确保开发过程中的多元化视角，以帮助识别和减轻潜在的偏见。

+   **定期审计**：在整个生命周期内定期对你的 LLM 进行偏见和公平性审计。

+   **情境部署**：考虑在不同应用中部署你的 LLM 的具体情境和潜在影响。

+   **持续研究**：了解 AI 伦理和公平的最新研究，并持续努力改进你的模型。

+   **用户教育**：教育用户关于你的 LLM 的能力和局限性，包括潜在的偏见。

+   **反馈机制**：实施强大的反馈机制以识别和解决部署模型中的不公平或偏见输出。请记住，反馈循环可能会通过放大数据中的模式来加强偏见，导致自我延续的错误。如果 AI 系统的输出影响未来的输入——无论是在内容推荐、招聘还是风险评估中——小的偏见随着时间的推移会累积，缩小多样性，加强刻板印象，并扭曲决策。

这里有一个示例，说明你可能如何实现一个简单的反馈系统：

```py
pythonCopyimport sqlite3
from datetime import datetime
class FeedbackSystem:
    def __init__(self, db_name='feedback.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             model_output TEXT,
             user_feedback TEXT,
             timestamp DATETIME)
        ''')
        self.conn.commit()
    def record_feedback(self, model_output, user_feedback):
        self.cursor.execute('''
            INSERT INTO feedback (model_output, user_feedback, timestamp)
            VALUES (?, ?, ?)
        ''', (model_output, user_feedback, datetime.now()))
        self.conn.commit()
    def get_recent_feedback(self, limit=10):
        self.cursor.execute('''
            SELECT model_output, user_feedback, timestamp
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        return self.cursor.fetchall()
    def close(self):
        self.conn.close()
# Example usage
feedback_system = FeedbackSystem()
# Simulating model output and user feedback
model_output = "The CEO made her decision."
user_feedback = "Biased: assumes CEO is female"
feedback_system.record_feedback(model_output, user_feedback)
# Retrieving recent feedback
recent_feedback = feedback_system.get_recent_feedback()
for output, feedback, timestamp in recent_feedback:
    print(f"Output: {output}")
    print(f"Feedback: {feedback}")
    print(f"Time: {timestamp}")
    print()
feedback_system.close()
```

此代码设置了一个简单的 SQLite 数据库来存储用户对模型输出的反馈，这些反馈可以定期审查，以识别潜在的偏见或问题。

# 摘要

在本章中，我们学习了 LLMs 中的公平性和偏见，重点关注理解不同的公平性定义，例如人口统计学平等、平等机会和均衡机会。我们探讨了 LLMs 中可能出现的偏见类型，包括代表性、语言、分配、服务质量以及刻板印象，以及通过人口统计学平等差异和机会平等差异等指标检测和量化这些偏见的技术。

我们通过实际编码示例向您展示了如何分析偏见。还涵盖了去偏策略，如数据增强、偏见感知微调和公平感知训练，提供了减轻偏见的具体方法。最后，我们获得了对伦理考量的见解，包括透明度、多元化的开发团队、定期审计和用户反馈系统。这些技能将帮助您在构建更公平和透明的 AI 系统时检测、衡量和解决 LLMs 中的偏见。

请记住，LLMs 中的公平性指标往往存在冲突，因为它们优先考虑公平待遇的不同方面。例如，*人口统计学平等*（各组之间结果平等）可能与*均衡机会*相冲突，后者确保各组之间假阳性率和假阴性率相似，尤其是在基础率不同的情况下。同样，*校准*（确保预测概率反映实际结果）可能与*均衡机会*相矛盾，因为一个校准良好的模型可能仍然具有不平等的错误率。此外，*个体公平性*（对类似个体进行类似处理）可能与*群体公平性*相冲突，后者强制在人口统计群体之间实现公平，有时需要差别化处理。这些冲突突显了在 AI 模型中平衡公平性目标所面临的挑战。

随着我们继续前进，下一章将探讨针对大型语言模型（LLMs）的高级提示工程技巧。
