# 第七章：构建自己的聊天机器人

聊天机器人，更确切地说是对话软件，对于许多企业来说都是惊人的工具。它们帮助企业在24/7不间断的服务中服务客户，无需增加工作量，保持一致的质量，并在机器人不足以应对时内置将任务转交给人类的选项。

它们是技术和人工智能结合以改善人力影响的绝佳例子。

它们从基于语音的解决方案，如Alexa，到基于文本的Intercom聊天框，再到Uber中的基于菜单的导航，种类繁多。

一个常见的误解是构建聊天机器人需要大量团队和大量的机器学习专业知识，尽管如果你试图构建像微软或Facebook（甚至Luis、Wit.ai等）这样的**通用**聊天机器人平台，这确实是正确的。

在这一章中，我们将涵盖以下主题：

+   为什么构建聊天机器人？

+   确定正确的用户意图

+   机器人响应

# 为什么以聊天机器人作为学习示例？

到目前为止，我们已经为每一个我们看到的NLP主题构建了一个应用程序：

+   使用语法和词汇洞察进行文本清理

+   语言学（和统计解析器），从文本中挖掘问题

+   实体识别用于信息提取

+   使用机器学习和深度学习进行监督文本分类

+   使用基于文本的向量，如GloVe/word2vec进行文本相似度

我们现在将把它们组合成一个更复杂的设置，并从头开始编写我们自己的聊天机器人。但在你从头开始构建任何东西之前，你应该问自己为什么。

# 为什么构建聊天机器人？

相关的问题是为什么我们应该构建自己的聊天机器人？**为什么我不能使用FB/MSFT/其他云服务？**

可能，一个更好的问题是要问自己**何时**开始构建自己的东西？在做出这个决定时，以下是一些需要考虑的因素：

**隐私和竞争**：作为一个企业，与Facebook或Microsoft（甚至更小的公司）分享有关用户的信息是个好主意吗？ 

**成本和限制**：你那奇特的云服务限制了特定智能提供商做出的设计选择，这些选择类似于谷歌或Facebook。此外，你现在需要为每个HTTP调用付费，这比在本地运行代码要慢。

**自由定制和扩展**：你可以开发一个更适合你的解决方案！你不必解决世界饥饿——只需通过高质量的软件不断提供越来越多的商业价值。如果你在大公司工作，你更有理由投资于可扩展的软件。

# 快速代码意味着词向量和方法

为了简化起见，我们将假设我们的机器人不需要记住任何问题的上下文。因此，它看到输入，对其做出响应，然后完成。与之前的输入不建立任何链接。

让我们先简单地使用`gensim`加载词向量：

```py
import numpy as np
import gensim
print(f"Gensim version: {gensim.__version__}")

from tqdm import tqdm
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_data(url, filename):
    """
    Download data if the filename does not exist already
    Uses Tqdm to show download progress
    """
    import os
    from urllib.request import urlretrieve

    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)
    else:
        print("File already exists, please remove if you wish to download again")

embedding_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
get_data(embedding_url, 'data/glove.6B.zip')
```

呼吸，这可能会根据你的下载速度而花费一分钟。一旦完成，让我们解压缩文件，将其放入数据目录，并将其转换为`word2vec`格式：

```py
# !unzip data/glove.6B.zip 
# !mv -v glove.6B.300d.txt data/glove.6B.300d.txt 
# !mv -v glove.6B.200d.txt data/glove.6B.200d.txt 
# !mv -v glove.6B.100d.txt data/glove.6B.100d.txt 
# !mv -v glove.6B.50d.txt data/glove.6B.50d.txt 

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'data/glove.6B.300d.txt'
word2vec_output_file = 'data/glove.6B.300d.txt.word2vec'
import os
if not os.path.exists(word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)
```

到前一个代码块结束时，我们已经将来自官方斯坦福源的300维GloVe嵌入转换成了word2vec格式。

让我们将这个加载到我们的工作记忆中：

```py
%%time
from gensim.models import KeyedVectors
filename = word2vec_output_file
embed = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
```

让我们快速检查我们是否可以通过检查任何单词的词嵌入来矢量化任何单词，例如，`awesome`：

```py
assert embed['awesome'] is not None
```

`awesome`，这行得通！

现在，让我们看看我们的第一个挑战。

# 确定正确的用户意图

这通常被称为意图分类问题。

作为玩具示例，我们将尝试构建一个DoorDash/Swiggy/Zomato等可能使用的订单机器人。

# 用例 - 食物订单机器人

考虑以下示例句子：*我在Indiranagar找一个便宜的中国餐馆*。

我们想在句子中挑选出中国作为一个菜系类型。显然，我们可以采取简单的办法，比如精确子串匹配（搜索*Chinese*）或基于TF-IDF的匹配。

相反，我们将泛化模型以发现我们可能尚未识别但可以通过GloVe嵌入学习的菜系类型。

我们将尽可能简单：我们将提供一些示例菜系类型来告诉模型我们需要菜系，并寻找句子中最相似的单词。

我们将遍历句子中的单词，并挑选出与参考单词相似度高于某个阈值的单词。

**词向量真的适用于这种情况吗？**

```py
cuisine_refs = ["mexican", "thai", "british", "american", "italian"]
sample_sentence = "I’m looking for a cheap Indian or Chinese place in Indiranagar"
```

为了简单起见，以下代码以`for`循环的形式编写，但可以矢量化以提高速度。

我们将遍历输入句子中的每个单词，并找到与已知菜系单词的相似度得分。

值越高，这个词就越有可能与我们的菜系参考或`cuisine_refs`相关：

```py
tokens = sample_sentence.split()
tokens = [x.lower().strip() for x in tokens] 
threshold = 18.3
found = []
for term in tokens:
    if term in embed.vocab:
        scores = []
        for C in cuisine_refs:
            scores.append(np.dot(embed[C], embed[term].T))
            # hint replace above above np.dot with: 
            # scores.append(embed.cosine_similarities(<vector1>, <vector_all_others>))
        mean_score = np.mean(scores)
        print(f"{term}: {mean_score}")
        if mean_score > threshold:
            found.append(term)
print(found)
```

以下是对应的输出：

```py
looking: 7.448504447937012
for: 10.627421379089355
a: 11.809560775756836
cheap: 7.09670877456665
indian: 18.64516258239746
or: 9.692893981933594
chinese: 19.09498405456543
place: 7.651237487792969
in: 10.085711479187012
['indian', 'chinese']
```

阈值是通过经验确定的。注意，我们能够推断出*印度*和*中国*作为菜系，即使它们不是原始集合的一部分。

当然，精确匹配将会有更高的分数。

这是一个很好的例子，其中在*通用*菜系类型方面有更好的问题表述，这比基于字典的菜系类型更有帮助。这也证明了我们可以依赖基于词向量的方法。

我们能否将此扩展到用户意图分类？让我们尝试下一步。

# 分类用户意图

我们希望能够通过用户的*意图*将句子分类。意图是一种通用机制，它将多个个别示例组合成一个语义伞。例如，*hi*，*hey*，*早上好*和*wassup!*都是`_greeting_`意图的例子。

使用*问候*作为输入，后端逻辑可以确定如何响应用户。

我们有很多种方法可以将词向量组合起来表示一个句子，但同样，我们将采取最简单的方法：将它们相加。

这绝对不是一个理想的解决方案，但由于我们使用简单、无监督的方法，它在实践中是可行的：

```py
def sum_vecs(embed,text):

    tokens = text.split(' ')
    vec = np.zeros(embed.vector_size)

    for idx, term in enumerate(tokens):
        if term in embed.vocab:
            vec = vec + embed[term]
    return vec

sentence_vector = sum_vecs(embed, sample_sentence)
print(sentence_vector.shape)
>> (300,)
```

让我们定义一个数据字典，为每个意图提供一些示例。

我们将使用由[Alan在Rasa博客](https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d)编写的数据字典来完成这项工作。

由于我们有更多的用户输入，这个字典可以被更新：

```py
data={
  "greet": {
    "examples" : ["hello","hey you","howdy","hello","hi","hey there","hey ho", "ssup?"],
    "centroid" : None
  },
  "inform": {
    "examples" : [
        "i'd like something asian",
        "maybe korean",
        "what swedish options do i have",
        "what italian options do i have",
        "i want korean food",
        "i want vegetarian food",
        "i would like chinese food",
        "what japanese options do i have",
        "vietnamese please",
        "i want some chicken",
        "maybe thai",
        "i'd like something vegetarian",
        "show me British restaurants",
        "show me a cool malay spot",
        "where can I get some spicy food"
    ],
    "centroid" : None
  },
  "deny": {
    "examples" : [
      "no thanks"
      "any other places ?",
      "something else",
      "naah",
      "not that one",
      "i do not like that",
      "something else",
      "please nooo"
      "show other options?"
    ],
    "centroid" : None
  },
    "affirm":{
        "examples":[
            "yeah",
            "that works",
            "good, thanks",
            "this works",
            "sounds good",
            "thanks, this is perfect",
            "just what I wanted"
        ],
        "centroid": None
    }

}
```

我们的方法很简单：我们找到每个*用户意图*的重心。重心只是一个表示每个意图的中心点。然后，将传入的文本分配给最接近相应聚类的用户意图。

让我们写一个简单的函数来找到重心并更新字典：

```py
def get_centroid(embed,examples):
     C = np.zeros((len(examples),embed.vector_size))
     for idx, text in enumerate(examples):
         C[idx,:] = sum_vecs(embed,text)

     centroid = np.mean(C,axis=0)
     assert centroid.shape[0] == embed.vector_size
     return centroid
```

让我们把这个重心加到数据字典里：

```py
for label in data.keys():
    data[label]["centroid"] = get_centroid(embed,data[label]["examples"])
```

让我们现在写一个简单的函数来找到最近的用户意图聚类。我们将使用已经在`np.linalg`中实现的L2范数：

```py
def get_intent(embed,data, text):
    intents = list(data.keys())
    vec = sum_vecs(embed,text)
    scores = np.array([ np.linalg.norm(vec-data[label]["centroid"]) for label in intents])
    return intents[np.argmin(scores)]
```

让我们在一些用户文本上运行这个，这些文本**不在**数据字典中：

```py
for text in ["hey ","i am looking for chinese food","not for me", "ok, this is good"]:
    print(f"text : '{text}', predicted_label : '{get_intent(embed, data, text)}'")
```

相应的代码很好地推广了，并且令人信服地表明，这对于我们花了大约10-15分钟到达这个点来说已经足够好了：

```py
text : 'hey ', predicted_label : 'greet'
text : 'i am looking for chinese food', predicted_label : 'inform'
text : 'not for me', predicted_label : 'deny'
text : 'ok, this is good', predicted_label : 'affirm'
```

# 机器人回应

我们现在知道了如何理解和分类用户意图。我们现在需要简单地用一些相应的回应来响应每个用户意图。让我们把这些*模板*机器人回应放在一个地方：

```py
templates = {
        "utter_greet": ["hey there!", "Hey! How you doin'? "],
        "utter_options": ["ok, let me check some more"],
        "utter_goodbye": ["Great, I'll go now. Bye bye", "bye bye", "Goodbye!"],
        "utter_default": ["Sorry, I didn't quite follow"],
        "utter_confirm": ["Got it", "Gotcha", "Your order is confirmed now"]
    }
```

将`Response`映射存储在单独的实体中很有帮助。这意味着你可以从你的意图理解模块中生成回应，然后将它们粘合在一起：

```py
response_map = {
    "greet": "utter_greet",
    "affirm": "utter_goodbye",
    "deny": "utter_options",
    "inform": "utter_confirm",
    "default": "utter_default",
}
```

如果我们再深入思考一下，就没有必要让回应映射仅仅依赖于被分类的意图。你可以将这个回应映射转换成一个单独的函数，该函数使用相关上下文生成映射，然后选择一个机器人模板。

但在这里，为了简单起见，让我们保持字典/JSON风格的格式。

让我们写一个简单的`get_bot_response`函数，它接受回应映射、模板和意图作为输入，并返回实际的机器人回应：

```py
import random
def get_bot_response(bot_response_map, bot_templates, intent):
    if intent not in list(response_map):
        intent = "default"
    select_template = bot_response_map[intent]
    templates = bot_templates[select_template]
    return random.choice(templates)
```

让我们快速尝试一句话：

```py
user_intent = get_intent(embed, data, "i want indian food")
get_bot_response(response_map, templates, user_intent)
```

代码目前没有语法错误。这似乎可以进行更多的性能测试。但在那之前，我们如何使它更好？

# 更好的回应个性化

你会注意到，该函数会随机选择一个模板来响应任何特定的*机器人意图*。虽然这里是为了简单起见，但在实践中，你可以训练一个机器学习模型来选择一个针对用户的个性化回应。

一个简单的个性化调整是适应用户的说话/打字风格。例如，一个用户可能会用正式的方式，*你好，今天过得怎么样？*，而另一个用户可能会用更非正式的方式，`Y``o`。

因此，*Hello*会得到*Goodbye!*的回应，而*Yo!*在同一对话中可能会得到*Bye bye*甚至*TTYL*。

目前，让我们检查一下我们已经看到的句子的机器人回应：

```py
for text in ["hey","i am looking for italian food","not for me", "ok, this is good"]:
    user_intent = get_intent(embed, data, text)
    bot_reply = get_bot_response(response_map, templates, user_intent)
    print(f"text : '{text}', intent: {user_intent}, bot: {bot_reply}")
```

由于随机性，回应可能会有所不同；这里是一个例子：

```py
text : 'hey', intent: greet, bot: Hey! How you doin'? 
text : 'i am looking for italian food', intent: inform, bot: Gotcha
text : 'not for me', intent: deny, bot: ok, let me check some more
text : 'ok, this is good', intent: affirm, bot: Goodbye!
```

# 摘要

在本章关于聊天机器人的内容中，我们学习了*意图*，通常指的是用户输入，*响应*，通过机器人进行，*模板*，定义了机器人响应的性质，以及*实体*，例如在我们的例子中是菜系类型。

此外，为了理解用户意图——甚至找到实体——我们使用了**无监督方法**，也就是说，这次我们没有训练示例。在实践中，大多数商业系统使用混合系统，结合了监督和无监督系统。

你应该从这里带走的一点是，我们不需要大量的训练数据来制作特定用例的第一个可用的聊天机器人版本。
