# 第十一章：金融深度学习

深度学习是金融服务行业中最令人兴奋的新兴技术之一，正确应用时可以提高投资回报。虽然计算机视觉和**自然语言处理** (**NLP**)等任务已经有了充分的研究，但**人工智能** (**AI**) 技术在金融服务中的应用仍在不断发展。需要注意的是，一些最先进、最具盈利性的深度学习技术在人工智能领域并未公开，且未来也不会公开。金融服务领域的高盈利性决定了必须保护先进的算法和措施，因此在本章中，我们将重点讨论原则。

人工智能在金融服务行业中的应用具有复杂性；...

# 所需工具

一如既往，我们将使用 Python 3 进行分析。Python 是量化交易应用的优秀选择，尤其适用于频率高于几秒钟的场景。对于高频应用，建议使用 Java 或 C++ 等中级语言。

在本章中，我们将在标准深度学习技术栈之上使用专门的金融 Python 库：

Zipline—一个基于 Python 的算法交易库。目前它被用作量化交易网站 Quantopian ([`www.quantopian.com`](https://www.quantopian.com)) 的回测工具。

# 人工智能在金融中的应用简介

尽管金融服务是计算密集型领域之一，但它充满了启发式方法。高级 AI 技术的应用充其量也只是脆弱的；许多公司根本没有采用能够轻松应用 AI 的策略。与硅谷争夺顶尖量化人才的竞争也让这个问题变得更加严重。你可能会自问，*“我难道不需要有金融背景才能处理这些数据吗？”*值得注意的是，全球两大顶级对冲基金是由参与过著名 Netflix 机器学习挑战赛的团队创办的。虽然研究算法交易技术有极大的好处，但你可以凭借对人工神经网络（ANNs）的了解来开始。

# 交易中的深度学习

**交易**是金融市场中买卖物品的行为；在金融术语中，我们称这些物品为**衍生品**。交易可以是短期的（当天内）、中期的（几天）或长期的（几周或更长）。根据全球最大银行之一摩根大通的专家，AI 应用在短期和中期交易策略上比人类更有效。在本节中，我们将探讨一些开发智能交易算法的基本策略，适用于短期和中期交易。但首先，让我们介绍一些基本概念。

交易策略试图利用市场低效来获利。算法训练中的一个核心策略被称为**阿尔法**，它是一种衡量业绩的指标。阿尔法通过将股票与一个指数进行对比，来衡量特定投资的主动回报。个别投资的表现与其匹配的指数之间的差异就是投资的阿尔法。在构建交易策略的网络时，我们希望我们的网络能够识别出那些为我们产生最大阿尔法的市场低效。

我们通常可以将传统的股票分析分为两类：

+   **基本面分析**则着眼于可能影响金融衍生品的基础因素，例如公司的一般财务健康状况。

+   **技术分析**从更数学的角度审视金融衍生品的实际表现，试图通过资产价格变动中的模式来预测价格的走势。

在这两种情况下，分析通常依赖于人工推理，而深度学习则进入了**量化分析**的领域，尤其是在所谓的**算法交易**中。广义上讲，算法交易就是其字面意义：由编码算法而非物理人类进行的交易。算法交易策略通过称为**回测**的过程进行验证，该过程将算法运行在历史数据上，以确定其是否能够在市场中表现良好。

算法交易应用于多个不同的领域：

+   **买方**-**公司**：公司利用算法交易来管理其中长期投资组合。

+   **卖方**-**公司**：公司利用高频算法交易来利用市场机会，并且推动市场本身的变化。

+   **系统化交易者**：这些个人和公司尝试将长期投资与短期投资高度相关的金融衍生品匹配。

这三种市场主体的共同点是，算法交易提供了一种比人类直觉更稳定、系统化的主动投资方式。

另一种策略依赖于技术指标，这些指标是基于数据历史分析的数学计算。大多数交易算法用于所谓的**高频**交易（**HFT**），它试图通过在各市场中进行大量极其快速的交易来利用市场低效。除非你拥有一些极为快速的计算机硬件，否则个人很难在这个领域竞争。相反，我们将用 TensorFlow 构建一些基本的算法用于非高频算法交易。

# 构建交易平台

在深入任何特定策略之前，让我们开始构建交易平台的基础。在这一部分，我们将构建处理数据输入和交易的代码，之后再深入研究两个具体策略。

# 基本交易功能

让我们从平台可以进行的基本操作开始；我们需要它能够买入、卖出或持有股票：

1.  首先，让我们从一些导入开始：

```py
import math
from time import time
from enum import Enum
```

1.  为了将来方便起见，我们将把这些函数包装在一个名为`TradingPosition`的类中：

```py
class TradingPosition(object):
   ''' Class that manages the trading position of our platform'''

    def __init__(self, code, buy_price, amount, next_price):
        self.code = code ## Status code for what action our algorithm is taking
        self.amount = amount ## The amount of the trade
        self.buy_price = buy_price ## The purchase price of a trade
        self.current_price = buy_price ## Buy price of the trade
        self.current_value = self.current_price * self.amount
        self.pro_value = next_price * self.amount
```

1.  让我们拆解输入变量。我们初始化的第一个变量是`code`，稍后我们将使用它作为执行买入、卖出或持有操作的状态码。接着，我们创建表示证券价格、证券数量（即股票数量）和证券当前价值的变量。

1.  现在我们已经有了变量，我们可以开始编写交易操作的代码了。我们需要创建一个`status`函数，用来追踪市场中价格的变化。为了简化，我们将这个函数命名为`TradeStatus`：

```py
def TradeStatus(self, current_price, next_price, amount):
        ''' Manages the status of a trade that is in action '''
        self.current_price = current_price ## updates the current price variable that is maintained within the class
        self.current_value = self.current_price * amount
        pro_value = next_price * amount
```

1.  接下来，让我们创建一个`买入股票`的函数：

```py
def BuyStock(self, buy_price, amount, next_price):
 ''' Function to buy a stock '''
     self.buy_price = ((self.amount * self.buy_price) + (amount * buy_price)) / (self.amount + amount)
 self.amount += amount
     self.TradeStatus(buy_price, next_price)
```

1.  在这里，我们的函数接受一个`买入价格`、`数量`以及序列中的`下一个价格`。我们计算`买入价格`，更新我们的交易量，并返回一个`交易状态`。接下来，让我们继续到`卖出股票`：

```py
def SellStock(self, sell_price, amount, next_price):
''' Function to sell a stock '''
     self.current_price = sell_price
     self.amount -= amount
     self.TradeStatus(sell_price, next_price)
```

1.  在买入函数方面，我们输入`卖出价格`和交易量，更新类的内部变量，并返回一个状态。最后，我们将创建一个简单的函数来持有股票，它大致上会告诉我们当前股票的价格状态：

```py
 def HoldStock(self, current_price, next_price):
 ''' Function to hold a stock '''
     self.TradeStatus(current_price, next_price)
```

1.  现在，让我们继续创建一个表示人工交易者的类。

# 创建人工交易者

尽管使用算法来指导交易决策是算法交易的定义，但这不一定是自动化交易。为了实现自动化交易，我们需要创建一个人工交易代理，它将代表我们执行交易策略：

1.  我们将这个类命名为`Trader`，并初始化执行交易算法所需的所有变量：

```py
class Trader(object): ''' An Artificial Trading Agent '''     def __init__(self, market, cash=100000000.0):         ## Initialize all the variables we need for our trader         self.cash = cash ## Our Cash Variable         self.market = market ##         self.codes = market.codes         self.reward = 0         self.positions = []         self.action_times = 0         self.initial_cash = cash         self.max_cash = cash * 3 self.total_rewards ...
```

# 管理市场数据

与任何机器学习算法一样，选择市场预测算法的特征至关重要，这可能决定算法策略的成功与否。为了将价格曲线数据简化到最基本的部分，我们可以使用降维算法，比如 PCA，甚至可以嵌入股票信息，尝试捕捉最重要的潜在特征。正如我们所学，深度学习可以帮助我们克服这些选择问题，因为神经网络在训练过程中隐式地进行特征选择：

1.  我们将创建一个新的类`MarketHandler`，并初始化所有处理不同交易策略所需的参数和数据：

```py
class MarketHandler(object):
 ''' Class for handling our platform's interaction with market data'''
     Running = 0
     Done = -1

     def __init__(self, codes, start_date="2008-01-01", end_date="2018-05-31", **options):
         self.codes = codes
         self.index_codes = []
         self.state_codes = []
         self.dates = []
         self.t_dates = []
         self.e_dates = []
         self.origin_frames = dict()
         self.scaled_frames = dict()
         self.data_x = None
         self.data_y = None
         self.seq_data_x = None
         self.seq_data_y = None
         self.next_date = None
         self.iter_dates = None
         self.current_date = None

         ## Initialize the stock data that will be fed in 
         self._init_data(start_date, end_date)

         self.state_codes = self.codes + self.index_codes
         self.scaler = [scaler() for _ in self.state_codes]
         self.trader = Trader(self, cash=self.init_cash)
         self.doc_class = Stock if self.m_type == 'stock' else Future
```

1.  我们还需要初始化大量的数据处理过程，以便正确操作数据进行分析：

```py
def _init_data_frames(self, start_date, end_date):
     self._validate_codes()
     columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
     ## Load the actual data
     for index, code in enumerate(self.state_codes):
         instrument_docs = self.doc_class.get_k_data(code, start_date, end_date)
         instrument_dicts = [instrument.to_dic() for instrument in instrument_docs]
         dates = [instrument[1] for instrument in instrument_dicts]
         instruments = [instrument[2:] for instrument in instrument_dicts]
         dates_set = dates_set.union(dates)
         scaler = self.scaler[index]
         scaler.fit(instruments)
         instruments_scaled = scaler.transform(instruments)
         origin_frame = pd.DataFrame(data=instruments, index=dates, columns=columns)
         scaled_frame = pd.DataFrame(data=instruments_scaled, index=dates, columns=columns)
         self.origin_frames[code] = origin_frame
 self.scaled_frames[code] = scaled_frame
         self.dates = sorted(list(dates_set))
    for code in self.state_codes:
         origin_frame = self.origin_frames[code]
         scaled_frame = self.scaled_frames[code]
         self.origin_frames[code] = origin_frame.reindex(self.dates, method='bfill')
         self.scaled_frames[code] = scaled_frame.reindex(self.dates, method='bfill')
```

1.  现在，我们初始化 `env_data()` 方法并调用 `self` 类：

```py
def _init_env_data(self):
     if not self.use_sequence:
         self._init_series_data()
     else:
         self._init_sequence_data()
```

1.  最后，让我们初始化刚才创建的数据处理函数：

```py
self._init_data_frames(start_date, end_date)
```

接下来，让我们开始构建我们平台的模型。

# 使用 LSTM 进行价格预测

让我们通过一个有监督学习的例子来开始，这个例子使用 LSTM 来预测给定股票的价格走势，基于其过去的表现。正如我们在之前的章节中所学，LSTM 和**循环神经网络**（**RNN**）在处理和预测序列数据时表现优越。这个模型将使用我们之前创建的交易平台结构：

1.  让我们从导入库开始：

```py
import tensorflow as tffrom sklearn.preprocessing import MinMaxScalerimport loggingimport os
```

1.  让我们创建一个包含所有运行 RNN 所需代码的类，命名为 `TradingRNN`。我们还会初始化必要的变量：

```py
class TradingRNN(): ''' An RNN Model for ...
```

# 回测您的算法

回测是指在历史数据上测试您的交易算法，以模拟其表现。虽然不能保证算法在实际市场中表现良好，但它能为我们提供一个良好的参考，帮助我们预测其表现。

在 Python 中，我们可以使用一个叫做**Zipline**的库来回测我们的算法。Zipline 是由在线交易算法平台 Quantopian 创建的回测平台，后来在 GitHub 上开源，供公众使用。它提供了十年的历史股票数据，并提供一个现实的交易环境，您可以在其中测试算法，包括交易成本、订单延迟和**滑点**。滑点是指交易发生时预期价格与实际执行价格之间的差价。要在 Python 中使用 Zipline，我们只需在命令行中运行`pip install zipline`。

每次使用 Zipline 时，我们都必须定义两个函数：

+   `initialize(context)`：在 Zipline 开始运行您的算法之前会调用此函数。context 变量包含了您算法中所需的所有全局变量。Initialize 函数非常类似于我们在 TensorFlow 中初始化变量的方式，在运行会话之前进行初始化。

+   `handle_data(context, data)`：这个函数做的正是它所说的：它将开盘、最高、最低和收盘的股市数据传递给您的算法，以及所需的上下文变量。

# 事件驱动交易平台

事件驱动投资是一种投资策略，重点关注可能影响股市走势的社会经济因素，尤其是在财报电话会议或并购等金融事件发生前。这种策略通常被大型基金采用，因为它们常常能获取一些并非完全公开的信息，而且需要在正确分析这些事件方面具备大量的专业知识。

为此，我们将从原始文本中提取事件并将其转化为元组，描述该事件。例如，如果我们说*Google*收购*Facebook*，则元组将是（*Actor = Google, Action = buys, Object = Facebook, Time = January 1 2018*）。这些元组可以帮助我们将事件简化为...

# 收集股票价格数据

大多数实时市场数据通过付费服务提供；比如 Bloomberg 终端或券商网站。目前，唯一一个不收费的实时金融市场数据 API 是 Alpha Vantage，它由商业和学术界联合维护。你可以通过在命令行中运行`pip install alpha_vantage`来安装它。你可以在 Alpha Vantage 网站上注册一个免费的 API 密钥。

一旦你获得了 API 密钥，就可以使用以下方式轻松查询`api`：

```py
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='TICKER',interval='1min', outputsize='full')
```

# 生成词嵌入

对于我们的嵌入方案，我们将使用上一章中的 GloVe 实现：

```py
from collections import Counter, defaultdictimport osfrom random import shuffleimport tensorflow as tfimport nltk
class GloVeModel(): def __init__(self, embedding_size, window_size, max_vocab_size=100000, min_occurrences=1, scaling_factor=3/4, cooccurrence_cap=100, batch_size=512, learning_rate=0.05): self.embedding_size = embedding_size#First we define the hyper-parameters of our model if isinstance(context_size, tuple): self.left_context, self.right_context = context_size elif isinstance(context_size, int): self.left_context = self.right_context = context_size   self.max_vocab_size = max_vocab_size self.min_occurrences ...
```

# 用于事件嵌入的神经张量网络

**神经张量网络**（**NTN**）是一种新的神经网络形式，它的工作方式类似于标准的前馈网络，但它包含一个被称为**张量层**的层，而不是标准的隐藏层。该网络最初是作为通过连接未连接的实体来完善知识库的手段开发的。例如，如果我们有实体 Google 和 YouTube，网络将帮助将这两个实体连接起来，使得 Google -> 拥有 -> YouTube。它通过将不同的关系对传递通过网络而不是通过单一的向量来工作，且通过将它们作为张量传递来实现这一点。该张量的每个切片代表两实体之间关系的不同变化。

在事件驱动交易领域，我们之所以对 NTN 感兴趣，是因为它能够将实体彼此关联。对我们来说，这意味着学习我们在本节第一部分创建的实体事件对：

1.  让我们从构建核心网络开始，并将其包含在一个名为`NTN`的函数中：

```py
def NTN(batch_placeholders, corrupt_placeholder, init_word_embeds,     entity_to_wordvec,\
 num_entities, num_relations, slice_size, batch_size, is_eval, label_placeholders):
     d = 100 
     k = slice_size
     ten_k = tf.constant([k])
     num_words = len(init_word_embeds)
     E = tf.Variable(init_word_embeds) 
     W = [tf.Variable(tf.truncated_normal([d,d,k])) for r in range(num_relations)]
     V = [tf.Variable(tf.zeros([k, 2*d])) for r in range(num_relations)]
     b = [tf.Variable(tf.zeros([k, 1])) for r in range(num_relations)]
     U = [tf.Variable(tf.ones([1, k])) for r in range(num_relations)]

     ent2word = [tf.constant(entity_i)-1 for entity_i in entity_to_wordvec]
     entEmbed = tf.pack([tf.reduce_mean(tf.gather(E, entword), 0) for entword in ent2word])
```

1.  仍然在`NTN`函数内，我们将遍历我们的嵌入并开始从中生成关系嵌入：

```py
predictions = list()
for r in range(num_relations):
     e1, e2, e3 = tf.split(1, 3, tf.cast(batch_placeholders[r], tf.int32)) #TODO: should the split dimension be 0 or 1?
     e1v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e1, name='e1v'+str(r)),[1]))
     e2v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e2, name='e2v'+str(r)),[1]))
     e3v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e3, name='e3v'+str(r)),[1]))
     e1v_pos = e1v
     e2v_pos = e2v
     e1v_neg = e1v
     e2v_neg = e3v
     num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)
     preactivation_pos = list()
     preactivation_neg = list()
```

1.  最后，我们将通过非线性操作处理关系，并输出它们：

```py
for slice in range(k):
     preactivation_pos.append(tf.reduce_sum(e1v_pos*tf.matmul(W[r][:,:,slice], e2v_pos), 0))
     preactivation_neg.append(tf.reduce_sum(e1v_neg*tf.matmul( W[r][:,:,slice], e2v_neg), 0))

preactivation_pos = tf.pack(preactivation_pos)
preactivation_neg = tf.pack(preactivation_neg)

temp2_pos = tf.matmul(V[r], tf.concat(0, [e1v_pos, e2v_pos]))
temp2_neg = tf.matmul(V[r], tf.concat(0, [e1v_neg, e2v_neg]))

preactivation_pos = preactivation_pos+temp2_pos+b[r]
preactivation_neg = preactivation_neg+temp2_neg+b[r]

activation_pos = tf.tanh(preactivation_pos)
activation_neg = tf.tanh(preactivation_neg)

score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
if not is_eval:
    predictions.append(tf.pack([score_pos, score_neg]))
else:
    predictions.append(tf.pack([score_pos,             tf.reshape(label_placeholders[r], num_rel_r)]))
```

1.  最后，让我们返回所有包含`predictions`的嵌入关系：

```py
predictions = tf.concat(1, predictions)

return predictions
```

1.  接下来，让我们定义网络的`loss`函数。我们将从 TensorFlow 的原生操作手动构建出我们的`loss`函数：

```py
def loss(predictions, regularization):
     temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
     temp1 = tf.reduce_sum(temp1)
     temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in     tf.trainable_variables()]))
     temp = temp1 + (regularization * temp2)
     return temp
```

1.  我们将定义一个训练算法，该算法仅返回最小化的`loss`函数，利用 TensorFlow 的内置函数：

```py
def training(loss, learningRate):
    return tf.train.AdagradOptimizer(learningRate).minimize(loss)
```

1.  最后，我们将创建一个简单的函数来评估网络的性能：

```py
def eval(predictions):
     print("predictions "+str(predictions.get_shape()))
     inference, labels = tf.split(0, 2, predictions)
     return inference, labels
```

接下来，我们将通过卷积神经网络（CNN）来预测价格变动，完成我们的模型。

# 使用卷积神经网络预测事件

现在我们有了嵌入结构，是时候用 CNN 进行预测了。当你通常想到 CNN 以及我们在其上完成的工作时，你可能会想到计算机视觉任务，比如识别图像中的物体。尽管 CNN 是为此设计的，但它们在文本特征检测方面也表现出色。

当我们在 NLP 中使用 CNN 时，我们将标准的像素输入替换为词嵌入。在典型的计算机视觉任务中，您使用 CNN 的过滤器对图像的小块进行处理，而在 NLP 任务中，我们对嵌入矩阵的行使用相同的滑动窗口。因此，滑动窗口的宽度就变成了……

# 资产管理中的深度学习

在金融服务中，投资组合是个人或组织持有的一系列投资。为了实现最佳回报（如同任何人都希望的那样！），通过决定应该将多少资金投资于某些金融资产来优化投资组合。在投资组合优化理论中，目标是拥有一个能够最小化风险并最大化回报的资产配置。因此，我们需要创建一个算法，预测每个资产的预期风险和回报，以便找到最佳优化方案。传统上，这项工作由财务顾问完成，然而，人工智能已经被证明在许多传统顾问构建的投资组合中表现得更好。

最近，出现了几种尝试开发用于资产配置的深度学习模型。考虑到许多这些技术并未公开发布，我们将看看一些我们作为 AI 科学家可能会使用的基本方法，以完成这项任务。

我们的目标是对一个股票指数进行建模，并看看我们是否能以至少 1%的收益超越该指数。我们将有效地构建一个自编码器来编码潜在的市场信息，然后使用解码器来构建一个最佳投资组合。由于我们处理的是时间序列信息，我们将为编码器和解码器都使用 RNN。一旦我们在数据上训练好自编码器，我们将用它作为简单前馈网络的输入，来预测我们最佳的投资组合配置。

让我们来看看如何在 TensorFlow 中实现这一过程。

1.  如常，我们首先导入所需的模块：

```py
import numpy as np
import tensorflow as tf from tensorflow.contrib.rnn import LSTMCell
```

1.  让我们加载我们的股票数据：

```py
ibb = defaultdict(defaultdict)
ibb_full = pd.read_csv('data/ibb.csv', index_col=0).astype('float32')

ibb_lp = ibb_full.iloc[:,0] 
ibb['calibrate']['lp'] = ibb_lp[0:104]
ibb['validate']['lp'] = ibb_lp[104:]

ibb_net = ibb_full.iloc[:,1] 
ibb['calibrate']['net'] = ibb_net[0:104]
ibb['validate']['net'] = ibb_net[104:]

ibb_percentage = ibb_full.iloc[:,2] 
ibb['calibrate']['percentage'] = ibb_percentage[0:104]
ibb['validate']['percentage'] = ibb_percentage[104:]
```

1.  让我们通过创建`AutoEncoder`开始我们的建模过程，并将其包含在`AutoEncoder`类中。我们将首先初始化主要的网络变量，就像之前做的那样：

```py
class AutoEncoder():
    ''' AutoEncoder for Data Drive Portfolio Allocation '''
    def __init__(self, config):
        """First, let's set up our hyperparameters"""
        num_layers = tf.placeholder('int')
        hidden_size = tf.placeholder('int')
        max_grad_norm = tf.placeholder('int')
        batch_size = tf.placeholder('int')
        crd = tf.placeholder('int')
        num_l = tf.placeholder('int')
        learning_rate = tf.placeholder('float')
        self.batch_size = batch_size

        ## sl will represent the length of an input sequence, which we would like to eb dynamic based on the data 
        sl = tf.placeholder("int")
        self.sl = sl
```

1.  接下来，我们将为输入数据创建`placeholders`，*x：*

```py
self.x = tf.placeholder("float", shape=[batch_size, sl], name='Input_data')
self.x_exp = tf.expand_dims(self.x, 1)
self.keep_prob = tf.placeholder("float")
```

1.  接下来，让我们创建编码器。我们将创建一系列 LSTM 单元来编码序列数据，但我们将以一种我们尚未见过的方式来实现：使用 TensorFlow 中的一个便捷函数`MultiRNNCell`。这个函数充当了一个更大的 RNN 占位符，我们可以在其中迭代，以便根据我们设置的`num_layers`参数动态创建层的数量：

```py
## Create the Encoder as a TensorFlow Scope
with tf.variable_scope("Encoder") as scope:
     ## For the encoder, we will use an LSTM cell with Dropout
     EncoderCell = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])
     EncoderCell = tf.contrib.rnn.DropoutWrapper(EncoderCell, output_keep_prob=self.keep_prob)

     ## Set the initial hidden state of the encoder
     EncInitialState = EncoderCell.zero_state(batch_size, tf.float32)

     ## Weights Factor
     W_mu = tf.get_variable('W_mu', [hidden_size, num_l])

     ## Outputs of the Encoder Layer
     outputs_enc, _ = tf.contrib.rnn.static_rnn(cell_enc,
     inputs=tf.unstack(self.x_exp, axis=2),
     initial_state=initial_state_enc)
     cell_output = outputs_enc[-1]

     ## Bias Factor
     b_mu = tf.get_variable('b_mu', [num_l])

     ## Mean of the latent space variables
     self.z_mu = tf.nn.xw_plus_b(cell_output, W_mu, b_mu, name='z_mu') 

     lat_mean, lat_var = tf.nn.moments(self.z_mu, axes=[1])
     self.loss_lat_batch = tf.reduce_mean(tf.square(lat_mean) + lat_var - tf.log(lat_var) - 1)
```

1.  接下来，我们将创建一个层来处理由编码器生成的隐藏状态：

```py
## Layer to Generate the Initial Hidden State from the Encoder
 with tf.name_scope("Initial_State") as scope:
 ## Weights Parameter State
 W_state = tf.get_variable('W_state', [num_l, hidden_size])

 ## Bias Paramter State
 b_state = tf.get_variable('b_state', [hidden_size])

 ## Hidden State
 z_state = tf.nn.xw_plus_b(self.z_mu, W_state, b_state, name='hidden_state')
```

1.  然后我们可以以与编码器层相同的方式创建`decoder`层：

```py
## Decoder Layer 
 with tf.variable_scope("Decoder") as scope:

     DecoderCell = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])

     ## Set an initial state for the decoder layer
     DecState = tuple([(z_state, z_state)] * num_layers)
     dec_inputs = [tf.zeros([batch_size, 1])] * sl

     ## Run the decoder layer
     outputs_dec, _ = tf.contrib.rnn.static_rnn(cell_dec, inputs=dec_inputs, initial_state=DecState)
```

1.  最后，我们将为网络创建输出层：

```py
## Output Layer
 with tf.name_scope("Output") as scope:
     params_o = 2 * crd 
     W_o = tf.get_variable('W_o', [hidden_size, params_o])
     b_o = tf.get_variable('b_o', [params_o])
     outputs = tf.concat(outputs_dec, axis=0) 
     h_out = tf.nn.xw_plus_b(outputs, W_o, b_o)
     h_mu, h_sigma_log = tf.unstack(tf.reshape(h_out, [sl, batch_size, params_o]), axis=2)
     h_sigma = tf.exp(h_sigma_log)
     dist = tf.contrib.distributions.Normal(h_mu, h_sigma)
     px = dist.log_prob(tf.transpose(self.x))
 loss_seq = -px
 self.loss_seq = tf.reduce_mean(loss_seq)
```

1.  现在我们已经构建了实际模型，我们可以继续设置训练过程。我们将使用指数衰减来调整学习率，这有助于通过缓慢降低学习率的值来稳定训练过程：

```py
## Train the AutoEncoder
 with tf.name_scope("Training") as scope:

     ## Global Step Function for Training
     global_step = tf.Variable(0, trainable=False)

     ## Exponential Decay for the larning rate
     lr = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.1, staircase=False)

     ## Loss Function for the Network
     self.loss = self.loss_seq + self.loss_lat_batch

     ## Utilize gradient clipping to prevent exploding gradients
     grads = tf.gradients(self.loss, tvars)
 grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
     self.numel = tf.constant([[0]])

     ## Lastly, apply the optimization process
     optimizer = tf.train.AdamOptimizer(lr)
     gradients = zip(grads, tvars)
     self.train_step = optimizer.apply_gradients(gradients, global_step=global_step)
     self.numel = tf.constant([[0]])

```

1.  现在，我们可以运行训练过程：

```py
if True:
     sess.run(model.init_op)
     writer = tf.summary.FileWriter(LOG_DIR, sess.graph) # writer for Tensorboard

 step = 0 # Step is a counter for filling the numpy array perf_collect
 for i in range(max_iterations):
     batch_ind = np.random.choice(N, batch_size, replace=False)
     result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.train_step],
 feed_dict={model.x: X_train[batch_ind], model.keep_prob: dropout})

 if i % plot_every == 0:
     perf_collect[0, step] = loss_train = result[0]
     loss_train_seq, lost_train_lat = result[1], result[2]

 batch_ind_val = np.random.choice(Nval, batch_size, replace=False)

 result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.merged],
 feed_dict={model.x: X_val[batch_ind_val], model.keep_prob: 1.0})
 perf_collect[1, step] = loss_val = result[0]
 loss_val_seq, lost_val_lat = result[1], result[2]
 summary_str = result[3]
 writer.add_summary(summary_str, i)
 writer.flush()

 print("At %6s / %6s train (%5.3f, %5.3f, %5.3f), val (%5.3f, %5.3f,%5.3f) in order (total, seq, lat)" % (
 i, max_iterations, loss_train, loss_train_seq, lost_train_lat, loss_val, loss_val_seq, lost_val_lat))
 step += 1
if False:

 start = 0
 label = [] # The label to save to visualize the latent space
 z_run = []

 while start + batch_size < Nval:
 run_ind = range(start, start + batch_size)
 z_mu_fetch = sess.run(model.z_mu, feed_dict={model.x: X_val[run_ind], model.keep_prob: 1.0})
 z_run.append(z_mu_fetch)
 start += batch_size

 z_run = np.concatenate(z_run, axis=0)
 label = y_val[:start]

 plot_z_run(z_run, label)

saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)
config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = model.z_mu.name
```

在对股票指数进行自编码后，我们将查看每只不同股票与其对应的自编码器版本之间的差异。然后我们将根据股票的自编码效果对其进行排名。随着算法学习到每只股票最重要的信息，股票与其经过自编码器处理后的版本的接近程度，提供了该股票在整个潜在投资组合中的衡量标准。

由于让多个股票共同贡献潜在信息没有益处，我们将限制所选股票为这些接近自编码版本的前十只股票：

```py
communal_information = []

for i in range(0,83):
    diff = np.linalg.norm((data.iloc[:,i] - reconstruct[:,i])) # 2 norm difference
    communal_information.append(float(diff))

print("stock #, 2-norm, stock name")
ranking = np.array(communal_information).argsort()
for stock_index in ranking:
    print(stock_index, communal_information[stock_index], stock['calibrate']['net'].iloc[:,stock_index].name) # print stock name from lowest different to highest
```

我们可以查看自编码器如何工作的方式如下：

```py
which_stock = 1

stock_autoencoder = copy.deepcopy(reconstruct[:, which_stock])
stock_autoencoder[0] = 0
stock_autoencoder = stock_autoencoder.cumsum()
stock_autoencoder += (stock['calibrate']['lp'].iloc[0, which_stock])

pd.Series(stock['calibrate']['lp'].iloc[:, which_stock].as_matrix(), index=pd.date_range(start='01/06/2012', periods=104, freq='W')).plot(label='stock original', legend=True)
pd.Series(stock_autoencoder, index=pd.date_range(start='01/06/2012', periods = 104,freq='W')).plot(label='stock autoencoded', legend=True)
```

虽然我们仍然需要在可用的股票中做出选择，但我们的选择决策现在是基于这些股票的样本外表现，从而使得我们的市场自编码器成为一种新颖的数据驱动方法。

# 总结

在本章中，我们学习了如何将深度学习知识应用于金融服务行业。我们学习了交易系统的原理，然后在 TensorFlow 中设计了自己的交易系统。接着，我们探讨了如何创建另一种类型的交易系统，这种系统利用公司周围的事件来预测其股价。最后，我们探索了一种新颖的技术，用于嵌入股市并利用这些嵌入来预测价格变动。

由于金融市场具有特殊的性质，它们的建模可能会很棘手，但我们在本章中所涵盖的技术将为你提供构建进一步模型的基础。记得在将算法部署到实时环境中之前，始终进行回测！...
