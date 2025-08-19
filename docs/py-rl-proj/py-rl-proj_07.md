# 第七章：创建聊天机器人

对话代理和聊天机器人近年来不断崛起。许多企业已开始依靠聊天机器人来回答客户的咨询，这一做法取得了显著成功。聊天机器人在过去一年增长了 5.6 倍 ([`chatbotsmagazine.com/chatbot-report-2018-global-trends-and-analysis-4d8bbe4d924b`](https://chatbotsmagazine.com/chatbot-report-2018-global-trends-and-analysis-4d8bbe4d924b))。聊天机器人可以帮助组织与客户进行沟通和互动，且无需人工干预，成本非常低廉。超过 51%的客户表示，他们希望企业能够 24/7 提供服务，并期望在一小时内得到回复。为了以一种负担得起的方式实现这一成功，尤其是在拥有大量客户的情况下，企业必须依赖聊天机器人。

# 背景问题

许多聊天机器人是使用常规的机器学习自然语言处理算法创建的，这些算法侧重于即时响应。一个新的概念是使用深度强化学习来创建聊天机器人。这意味着我们会考虑即时响应的未来影响，以保持对话的连贯性。

本章中，你将学习如何将深度强化学习应用于自然语言处理。我们的奖励函数将是一个面向未来的函数，您将通过创建该函数学会如何从概率的角度思考。

# 数据集

我们将使用的这个数据集主要由选定电影中的对话组成。这个数据集有助于激发并理解聊天机器人的对话方法。此外，其中还包含电影台词，这些台词与电影中的对话本质相同，不过是人与人之间较简短的交流。其他将使用的数据集还包括一些包含电影标题、电影角色和原始剧本的数据集。

# 分步指南

我们的解决方案将使用建模方法，重点关注对话代理的未来方向，从而生成连贯且有趣的对话。该模型将模拟两个虚拟代理之间的对话，使用策略梯度方法。这些方法旨在奖励显示出对话三个重要特性的交互序列：信息性（不重复的回合）、高度连贯性和简洁的回答（这与面向未来的函数相关）。在我们的解决方案中，动作将被定义为聊天机器人生成的对话或交流话语。此外，状态将被定义为之前的两轮互动。为了实现这一目标，我们将使用以下章节中的剧本。

# 数据解析器

数据解析脚本旨在帮助清理和预处理我们的数据集。此脚本有多个依赖项，如`pickle`、`codecs`、`re`、`OS`、`time`和`numpy`。该脚本包含三个功能。第一个功能帮助通过预处理词频并基于词频阈值创建词汇表来过滤词汇。第二个功能帮助解析所有词汇到此脚本中，第三个功能帮助从数据中提取仅定义的词汇：

```py
import pickle
import codecs
import re
import os
import time
import numpy as np
```

以下模块清理并预处理训练数据集中的文本：

```py
def preProBuildWordVocab(word_count_threshold=5, all_words_path='data/all_words.txt'):
    # borrowed this function from NeuralTalk

    if not os.path.exists(all_words_path):
        parse_all_words(all_words_path)

    corpus = open(all_words_path, 'r').read().split('\n')[:-1]
    captions = np.asarray(corpus, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)
```

接下来，遍历字幕并创建词汇表。

```py

    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in captions:
        nsents += 1
        for w in sent.lower().split(' '):

            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector
```

接下来，解析所有电影台词中的词汇。

```py

def parse_all_words(all_words_path):
    raw_movie_lines = open('data/movie_lines.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')[:-1]

    with codecs.open(all_words_path, "w", encoding='utf-8', errors='ignore') as f:
        for line in raw_movie_lines:
            line = line.split(' +++$+++ ')
            utterance = line[-1]
            f.write(utterance + '\n')
```

仅提取数据中的词汇部分，如下所示：

```py

def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data
```

接下来，创建并存储话语字典。

```py

if __name__ == '__main__':
    parse_all_words('data/all_words.txt')

    raw_movie_lines = open('data/movie_lines.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')[:-1]

    utterance_dict = {}
    with codecs.open('data/tokenized_all_words.txt', "w", encoding='utf-8', errors='ignore') as f:
        for line in raw_movie_lines:
            line = line.split(' +++$+++ ')
            line_ID = line[0]
            utterance = line[-1]
            utterance_dict[line_ID] = utterance
            utterance = " ".join([refine(w) for w in utterance.lower().split()])
            f.write(utterance + '\n')
    pickle.dump(utterance_dict, open('data/utterance_dict', 'wb'), True)
```

数据已解析，并可以在后续步骤中使用。

# 数据读取

数据读取脚本帮助从数据解析脚本预处理后的训练文本中生成可训练的批次。我们首先通过导入所需的方法开始：

```py
import pickle
import random
```

此辅助模块帮助从预处理后的训练文本中生成可训练的批次。

```py
class Data_Reader:
    def __init__(self, cur_train_index=0, load_list=False):
        self.training_data = pickle.load(open('data/conversations_lenmax22_formersents2_with_former', 'rb'))
        self.data_size = len(self.training_data)
        if load_list:
            self.shuffle_list = pickle.load(open('data/shuffle_index_list', 'rb'))
        else:    
            self.shuffle_list = self.shuffle_index()
        self.train_index = cur_train_index
```

以下代码从数据中获取批次号：

```py
    def get_batch_num(self, batch_size):
        return self.data_size // batch_size
```

以下代码打乱来自数据的索引：

```py
    def shuffle_index(self):
        shuffle_index_list = random.sample(range(self.data_size), self.data_size)
        pickle.dump(shuffle_index_list, open('data/shuffle_index_list', 'wb'), True)
        return shuffle_index_list
```

以下代码基于之前获取的批次号生成批次索引：

```py
    def generate_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size:
            batch_index = self.shuffle_list[self.train_index:self.data_size]
            self.shuffle_list = self.shuffle_index()
            remain_size = batch_size - (self.data_size - self.train_index)
            batch_index += self.shuffle_list[:remain_size]
            self.train_index = remain_size
        else:
            batch_index = self.shuffle_list[self.train_index:self.train_index+batch_size]
            self.train_index += batch_size

        return batch_index
```

以下代码生成训练批次：

```py

    def generate_training_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]   # batch_size of conv_b

        return batch_X, batch_Y
```

以下函数使用前者生成训练批次。

```py

    def generate_training_batch_with_former(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]   # batch_size of conv_b
        former = [self.training_data[i][2] for i in batch_index]    # batch_size of former utterance

        return batch_X, batch_Y, former
```

以下代码生成测试批次：

```py

    def generate_testing_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a

        return batch_X
```

这部分内容结束于数据读取。

# 辅助方法

此脚本由一个`Seq2seq`对话生成模型组成，用于反向模型的逆向熵损失。它将确定政策梯度对话的语义连贯性奖励。实质上，该脚本将帮助我们表示未来的奖励函数。该脚本将通过以下操作实现：

+   编码

+   解码

+   生成构建

所有先前的操作都基于**长短期记忆**（**LSTM**）单元。

特征提取脚本帮助从数据中提取特征和特性，以便更好地训练它。我们首先通过导入所需的模块开始。

```py
import tensorflow as tf
import numpy as np
import re
```

接下来，定义模型输入。如果强化学习被设置为 True，则基于语义连贯性和回答损失字幕的易用性计算标量。

```py
def model_inputs(embed_dim, reinforcement= False):    
    word_vectors = tf.placeholder(tf.float32, [None, None, embed_dim], name = "word_vectors")
    reward = tf.placeholder(tf.float32, shape = (), name = "rewards")
    caption = tf.placeholder(tf.int32, [None, None], name = "captions")
    caption_mask = tf.placeholder(tf.float32, [None, None], name = "caption_masks")
    if reinforcement: #Normal training returns only the word_vectors, caption and caption_mask placeholders, 
        #With reinforcement learning, there is an extra placeholder for rewards
        return word_vectors, caption, caption_mask, reward
    else:
        return word_vectors, caption, caption_mask
```

接下来，定义执行序列到序列网络编码的编码层。输入序列传递给编码器，并返回 RNN 输出和状态。

```py

def encoding_layer(word_vectors, lstm_size, num_layers, keep_prob, 
                   vocab_size):

    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(lstm_size), keep_prob) for _ in range(num_layers)])

    outputs, state = tf.nn.dynamic_rnn(cells, 
                                       word_vectors, 
                                       dtype=tf.float32)
    return outputs, state
```

接下来，定义使用 LSTM 单元的解码器训练过程，结合编码器状态和解码器输入。

```py
def decode_train(enc_state, dec_cell, dec_input, 
                         target_sequence_length,output_sequence_length,
                         output_layer, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,                #Apply dropout to the LSTM cell
                                             output_keep_prob=keep_prob)

    helper = tf.contrib.seq2seq.TrainingHelper(dec_input,             #Training helper for decoder 
                                               target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              enc_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True,
                                                     maximum_iterations=output_sequence_length)
    return outputs
```

接下来，定义一个类似于训练时使用的推理解码器。使用贪心策略辅助工具，将解码器的最后输出作为下一个解码器输入。返回的输出包含训练 logits 和样本 ID。

```py
def decode_generate(encoder_state, dec_cell, dec_embeddings,
                         target_sequence_length,output_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], 1),  #Decoder helper for inference
                                                      2)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True,
                                                     maximum_iterations=output_sequence_length)
    return outputs
```

接下来，创建解码层。

```py
def decoding_layer(dec_input, enc_state,
                   target_sequence_length,output_sequence_length,
                   lstm_size,
                   num_layers,n_words,
                   batch_size, keep_prob,embedding_size, Train = True):
    target_vocab_size = n_words
    with tf.device("/cpu:0"):
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,embedding_size], -0.1, 0.1), name='Wemb')
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(lstm_size) for _ in range(num_layers)])

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)

    if Train:
        with tf.variable_scope("decode"):
            train_output = decode_train(enc_state, 
                                                cells, 
                                                dec_embed_input, 
                                                target_sequence_length, output_sequence_length,
                                                output_layer, 
                                                keep_prob)

    with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
        infer_output = decode_generate(enc_state, 
                                            cells, 
                                            dec_embeddings, target_sequence_length,
                                           output_sequence_length,
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)
    if Train:
        return train_output, infer_output
    return infer_output
```

接下来，创建 bos 包含部分，将对应于 <bos> 的索引添加到每个批次的标题张量的第一个索引，<bos> 表示句子的开始。

```py
def bos_inclusion(caption,batch_size):

    sliced_target = tf.strided_slice(caption, [0,0], [batch_size, -1], [1,1])
    concat = tf.concat([tf.fill([batch_size, 1],1), sliced_target],1)
    return concat
```

接下来，定义 pad 序列，该方法通过用零填充或在必要时截断每个问题，创建大小为 maxlen 的数组。

```py
def pad_sequences(questions, sequence_length =22):
    lengths = [len(x) for x in questions]
    num_samples = len(questions)
    x = np.zeros((num_samples, sequence_length)).astype(int)
    for idx, sequence in enumerate(questions):
        if not len(sequence):
            continue  # empty list/array was found
        truncated  = sequence[-sequence_length:]

        truncated = np.asarray(truncated, dtype=int)

        x[idx, :len(truncated)] = truncated

    return x
```

如果数据中存在非词汇部分，请忽略它们，只保留所有字母。

```py
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    data = ' '.join(words)
    return data
```

接下来，创建批次，将词向量表示送入网络。

```py
def make_batch_input(batch_input, input_sequence_length, embed_dims, word2vec):

    for i in range(len(batch_input)):

        batch_input[i] = [word2vec[w] if w in word2vec else np.zeros(embed_dims) for w in batch_input[i]]
        if len(batch_input[i]) >input_sequence_length:
            batch_input[i] = batch_input[i][:input_sequence_length]
        else:
            for _ in range(input_sequence_length - len(batch_input[i])):
                batch_input[i].append(np.zeros(embed_dims))

    return np.array(batch_input)

def replace(target,symbols):  #Remove symbols from sequence
    for symbol in symbols:
        target = list(map(lambda x: x.replace(symbol,''),target))
    return target

def make_batch_target(batch_target, word_to_index, target_sequence_length):
    target = batch_target
    target = list(map(lambda x: '<bos> ' + x, target))
    symbols = ['.', ',', '"', '\n','?','!','\\','/']
    target = replace(target, symbols)

    for idx, each_cap in enumerate(target):
        word = each_cap.lower().split(' ')
        if len(word) < target_sequence_length:
            target[idx] = target[idx] + ' <eos>'  #Append the end of symbol symbol 
        else:
            new_word = ''
            for i in range(target_sequence_length-1):
                new_word = new_word + word[i] + ' '
            target[idx] = new_word + '<eos>'

    target_index = [[word_to_index[word] if word in word_to_index else word_to_index['<unk>'] for word in 
                          sequence.lower().split(' ')] for sequence in target]
    #print(target_index[0])

    caption_matrix = pad_sequences(target_index,target_sequence_length)
    caption_matrix = np.hstack([caption_matrix, np.zeros([len(caption_matrix), 1])]).astype(int)
    caption_masks = np.zeros((caption_matrix.shape[0], caption_matrix.shape[1]))
    nonzeros = np.array(list(map(lambda x: (x != 0).sum(), caption_matrix)))
    #print(nonzeros)
    #print(caption_matrix[1])

    for ind, row in enumerate(caption_masks): #Set the masks as an array of ones where actual words exist and zeros otherwise
        row[:nonzeros[ind]] = 1                 
        #print(row)
    print(caption_masks[0])
    print(caption_matrix[0])
    return caption_matrix,caption_masks   

def generic_batch(generic_responses, batch_size, word_to_index, target_sequence_length):
    size = len(generic_responses) 
    if size > batch_size:
        generic_responses = generic_responses[:batch_size]

    else:
        for j in range(batch_size - size):
            generic_responses.append('')

    return make_batch_Y(generic_responses, word_to_index, target_sequence_length)

```

接下来，从预测的索引生成句子。每当预测时，将 <unk> 和 <pad> 替换为具有下一个最高概率的单词。

```py
def index2sentence(generated_word_index, prob_logit, ixtoword):
    generated_word_index = list(generated_word_index)
    for i in range(len(generated_word_index)):
        if generated_word_index[i] == 3 or generated_word_index[i] == 0:
            sort_prob_logit = sorted(prob_logit[i])
            curindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
            count = 1
            while curindex <= 3:
                curindex = np.where(prob_logit[i] == sort_prob_logit[(-2)-count])[0][0]
                count += 1

            generated_word_index[i] = curindex

    generated_words = []
    for ind in generated_word_index:
        generated_words.append(ixtoword[ind])    
    generated_sentence = ' '.join(generated_words)
    generated_sentence = generated_sentence.replace('<bos> ', '')  #Replace the beginning of sentence tag
    generated_sentence = generated_sentence.replace('<eos>', '')   #Replace the end of sentence tag
    generated_sentence = generated_sentence.replace('--', '')      #Replace the other symbols predicted
    generated_sentence = generated_sentence.split('  ')
    for i in range(len(generated_sentence)):       #Begin sentences with Upper case 
        generated_sentence[i] = generated_sentence[i].strip()
        if len(generated_sentence[i]) > 1:
            generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.'
        else:
            generated_sentence[i] = generated_sentence[i].upper()
    generated_sentence = ' '.join(generated_sentence)
    generated_sentence = generated_sentence.replace(' i ', ' I ')
    generated_sentence = generated_sentence.replace("i'm", "I'm")
    generated_sentence = generated_sentence.replace("i'd", "I'd")

    return generated_sentence
```

这结束了所有辅助函数。

# 聊天机器人模型

以下脚本包含策略梯度模型，它将用于结合强化学习奖励与交叉熵损失。依赖项包括 `numpy` 和 `tensorflow`。我们的策略梯度基于 LSTM 编码器-解码器。我们将使用策略梯度的随机演示，这将是一个关于指定状态的动作概率分布。该脚本表示了这一切，并指定了需要最小化的策略梯度损失。

通过第二个单元运行第一个单元的输出；输入与零拼接。响应的最终状态通常由两个部分组成——编码器对输入的潜在表示，以及基于选定单词的解码器状态。返回的内容包括占位符张量和其他张量，例如损失和训练优化操作。让我们从导入所需的库开始。

```py
import tensorflow as tf
import numpy as np
import helper as h
```

我们将创建一个聊天机器人类来构建模型。

```py
class Chatbot():
    def __init__(self, embed_dim, vocab_size, lstm_size, batch_size, input_sequence_length, target_sequence_length, learning_rate =0.0001, keep_prob = 0.5, num_layers = 1, policy_gradients = False, Training = True):
        self.embed_dim = embed_dim
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.input_sequence_length = tf.fill([self.batch_size],input_sequence_length+1)
        self.target_sequence_length = tf.fill([self.batch_size],target_sequence_length+1)
        self.output_sequence_length = target_sequence_length +1
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.policy_gradients = policy_gradients
        self.Training = Training

```

接下来，创建一个构建模型的方法。如果请求策略梯度，则根据需要获取输入。

```py
    def build_model(self):
        if self.policy_gradients:
            word_vectors, caption, caption_mask, rewards = h.model_inputs(self.embed_dim, True)
            place_holders = {'word_vectors': word_vectors,
                'caption': caption,
                'caption_mask': caption_mask, "rewards": rewards
                             }
        else:
            word_vectors, caption, caption_mask = h.model_inputs(self.embed_dim)

            place_holders = {'word_vectors': word_vectors,
                'caption': caption,
                'caption_mask': caption_mask}
        enc_output, enc_state = h.encoding_layer(word_vectors, self.lstm_size, self.num_layers,
                                         self.keep_prob, self.vocab_size)
        #dec_inp = h.bos_inclusion(caption, self.batch_size)
        dec_inp = caption

```

接下来，获取推理层。

```py
        if not self.Training:
            print("Test mode")
            inference_out = h.decoding_layer(dec_inp, enc_state,self.target_sequence_length, 
                                                    self.output_sequence_length,
                                                    self.lstm_size, self.num_layers,
                                                    self.vocab_size, self.batch_size,
                                                  self.keep_prob, self.embed_dim, False)
            logits = tf.identity(inference_out.rnn_output, name = "train_logits")
            predictions = tf.identity(inference_out.sample_id, name = "predictions")
            return place_holders, predictions, logits

```

接下来，获取损失层。

```py
        train_out, inference_out = h.decoding_layer(dec_inp, enc_state,self.target_sequence_length, 
                                                    self.output_sequence_length,
                                                    self.lstm_size, self.num_layers,
                                                    self.vocab_size, self.batch_size,
                                                  self.keep_prob, self.embed_dim)

        training_logits = tf.identity(train_out.rnn_output, name = "train_logits")
        prediction_logits = tf.identity(inference_out.sample_id, name = "predictions")
        cross_entropy = tf.contrib.seq2seq.sequence_loss(training_logits, caption, caption_mask)
        losses = {"entropy": cross_entropy}

```

根据策略梯度的状态，选择最小化交叉熵损失或策略梯度损失。

```py
        if self.policy_gradients:
            pg_loss = tf.contrib.seq2seq.sequence_loss(training_logits, caption, caption_mask*rewards)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(pg_loss)
            losses.update({"pg":pg_loss}) 
        else:
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        return optimizer, place_holders,prediction_logits,training_logits, losses
```

现在我们已经有了训练所需的所有方法。

# 训练数据

之前编写的脚本与训练数据集结合起来。让我们通过导入在前面章节中开发的所有模块来开始训练，如下所示：

```py
from data_reader import Data_Reader
import data_parser
from gensim.models import KeyedVectors
import helper as h
from seq_model import Chatbot
import tensorflow as tf
import numpy as np
```

接下来，让我们创建一组在原始 `seq2seq` 模型中观察到的通用响应，策略梯度会避免这些响应。

```py
generic_responses = [
    "I don't know what you're talking about.", 
    "I don't know.", 
    "You don't know.",
    "You know what I mean.", 
    "I know what you mean.", 
    "You know what I'm saying.",
    "You don't know anything."
]
```

接下来，我们将定义训练所需的所有常量。

```py
checkpoint = True
forward_model_path = 'model/forward'
reversed_model_path = 'model/reversed'
rl_model_path = "model/rl"
model_name = 'seq2seq'
word_count_threshold = 20
reversed_word_count_threshold = 6
dim_wordvec = 300
dim_hidden = 1000
input_sequence_length = 22
output_sequence_length = 22
learning_rate = 0.0001
epochs = 1
batch_size = 200
forward_ = "forward"
reverse_ = "reverse"
forward_epochs = 50
reverse_epochs = 50
display_interval = 100

```

接下来，定义训练函数。根据类型，加载前向或反向序列到序列模型。数据也根据模型读取，反向模型如下所示：

```py
def train(type_, epochs=epochs, checkpoint=False):
    tf.reset_default_graph()
    if type_ == "forward":
        path = "model/forward/seq2seq"
        dr = Data_Reader(reverse=False)
    else:
        dr = Data_Reader(reverse=True)
        path = "model/reverse/seq2seq"

```

接下来，按照以下方式创建词汇表：

```py
    word_to_index, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
```

上述命令的输出应打印以下内容，表示已过滤的词汇表大小。

```py
preprocessing word counts and creating vocab based on word count threshold 20
filtered words from 76029 to 6847
```

`word_to_index` 变量被填充为过滤后单词到整数的映射，如下所示：

```py
{'': 4,
'deposition': 1769,
'next': 3397,
'dates': 1768,
'chance': 2597,
'slipped': 4340,...
```

`index_to_word` 变量被填充为从整数到过滤后的单词的映射，这将作为反向查找。

```py
5: 'tastes',
6: 'shower',
7: 'agent',
8: 'lack',
```

接下来，从`gensim`库加载词到向量的模型。

```py
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
```

接下来，实例化并构建聊天机器人模型，使用所有已定义的常量。如果有之前训练的检查点，则恢复它；否则，初始化图。

```py
    model = Chatbot(dim_wordvec, len(word_to_index), dim_hidden, batch_size,
                    input_sequence_length, output_sequence_length, learning_rate)
    optimizer, place_holders, predictions, logits, losses = model.build_model()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if checkpoint:
        saver.restore(sess, path)
        print("checkpoint restored at path: {}".format(path))
    else:
        tf.global_variables_initializer().run()
```

接下来，通过迭代纪元并开始批量处理来启动训练。

```py
    for epoch in range(epochs):
        n_batch = dr.get_batch_num(batch_size=batch_size)
        for batch in range(n_batch):

            batch_input, batch_target = dr.generate_training_batch(batch_size)
```

`batch_input`包含来自训练集的单词列表。`batch_target`包含输入的句子列表，这些句子将作为目标。单词列表通过辅助函数转换为向量形式。使用转换后的输入、掩码和目标构建图的喂入字典。

```py
            inputs_ = h.make_batch_input(batch_input, input_sequence_length, dim_wordvec, word_vector)

            targets, masks = h.make_batch_target(batch_target, word_to_index, output_sequence_length)
            feed_dict = {
                place_holders['word_vectors']: inputs_,
                place_holders['caption']: targets,
                place_holders['caption_mask']: masks
            }
```

接下来，通过调用优化器并输入训练数据来训练模型。在某些间隔记录损失值，以查看训练的进展。训练结束后保存模型。

```py
            _, loss_val, preds = sess.run([optimizer, losses["entropy"], predictions],
                                          feed_dict=feed_dict)

            if batch % display_interval == 0:
                print(preds.shape)
                print("Epoch: {}, batch: {}, loss: {}".format(epoch, batch, loss_val))
                print("===========================================================")

        saver.save(sess, path)

        print("Model saved at {}".format(path))
    print("Training done")

    sess.close()
```

输出应如下所示。

```py
(200, 23)
Epoch: 0, batch: 0, loss: 8.831538200378418
===========================================================
```

模型经过正向和反向训练，相应的模型被存储。在下一个函数中，模型被恢复并重新训练，以创建聊天机器人。

```py
def pg_train(epochs=epochs, checkpoint=False):
    tf.reset_default_graph()
    path = "model/reinforcement/seq2seq"
    word_to_index, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
    generic_caption, generic_mask = h.generic_batch(generic_responses, batch_size, word_to_index,
                                                    output_sequence_length)

    dr = Data_Reader()
    forward_graph = tf.Graph()
    reverse_graph = tf.Graph()
    default_graph = tf.get_default_graph()
```

创建两个图表以加载训练好的模型。

```py
    with forward_graph.as_default():
        pg_model = Chatbot(dim_wordvec, len(word_to_index), dim_hidden, batch_size,
                           input_sequence_length, output_sequence_length, learning_rate, policy_gradients=True)
        optimizer, place_holders, predictions, logits, losses = pg_model.build_model()

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        if checkpoint:
            saver.restore(sess, path)
            print("checkpoint restored at path: {}".format(path))
        else:
            tf.global_variables_initializer().run()
            saver.restore(sess, 'model/forward/seq2seq')
    # tf.global_variables_initializer().run()
    with reverse_graph.as_default():
        model = Chatbot(dim_wordvec, len(word_to_index), dim_hidden, batch_size,
                        input_sequence_length, output_sequence_length, learning_rate)
        _, rev_place_holders, _, _, reverse_loss = model.build_model()
        sess2 = tf.InteractiveSession()
        saver2 = tf.train.Saver()

        saver2.restore(sess2, "model/reverse/seq2seq")
        print("reverse model restored")

    dr = Data_Reader(load_list=True)
```

接下来，加载数据以批量训练数据。

```py
    for epoch in range(epochs):
        n_batch = dr.get_batch_num(batch_size=batch_size)
        for batch in range(n_batch):

            batch_input, batch_caption, prev_utterance = dr.generate_training_batch_with_former(batch_size)
            targets, masks = h.make_batch_target(batch_caption, word_to_index, output_sequence_length)
            inputs_ = h.make_batch_input(batch_input, input_sequence_length, dim_wordvec, word_vector)

            word_indices, probabilities = sess.run([predictions, logits],
                                                   feed_dict={place_holders['word_vectors']: inputs_

                                                       , place_holders["caption"]: targets})

            sentence = [h.index2sentence(generated_word, probability, index_to_word) for
                        generated_word, probability in zip(word_indices, probabilities)]

            word_list = [word.split() for word in sentence]

            generic_test_input = h.make_batch_input(word_list, input_sequence_length, dim_wordvec, word_vector)

            forward_coherence_target, forward_coherence_masks = h.make_batch_target(sentence,
                                                                                    word_to_index,
                                                                                    output_sequence_length)

            generic_loss = 0.0
```

同时，学习何时说出通用文本，如下所示：

```py
            for response in generic_test_input:
                sentence_input = np.array([response] * batch_size)
                feed_dict = {place_holders['word_vectors']: sentence_input,
                             place_holders['caption']: generic_caption,
                             place_holders['caption_mask']: generic_mask,
                             }
                generic_loss_i = sess.run(losses["entropy"], feed_dict=feed_dict)
                generic_loss -= generic_loss_i / batch_size

            # print("generic loss work: {}".format(generic_loss))

            feed_dict = {place_holders['word_vectors']: inputs_,
                         place_holders['caption']: forward_coherence_target,
                         place_holders['caption_mask']: forward_coherence_masks,
                         }

            forward_entropy = sess.run(losses["entropy"], feed_dict=feed_dict)

            previous_utterance, previous_mask = h.make_batch_target(prev_utterance,
                                                                    word_to_index, output_sequence_length)

            feed_dict = {rev_place_holders['word_vectors']: generic_test_input,
                         rev_place_holders['caption']: previous_utterance,
                         rev_place_holders['caption_mask']: previous_mask,
                         }
            reverse_entropy = sess2.run(reverse_loss["entropy"], feed_dict=feed_dict)

            rewards = 1 / (1 + np.exp(-reverse_entropy - forward_entropy - generic_loss))

            feed_dict = {place_holders['word_vectors']: inputs_,
                         place_holders['caption']: targets,
                         place_holders['caption_mask']: masks,
                         place_holders['rewards']: rewards
                         }

            _, loss_pg, loss_ent = sess.run([optimizer, losses["pg"], losses["entropy"]], feed_dict=feed_dict)

            if batch % display_interval == 0:
                print("Epoch: {}, batch: {}, Entropy loss: {}, Policy gradient loss: {}".format(epoch, batch, loss_ent,
                                                                                                loss_pg))

                print("rewards: {}".format(rewards))
                print("===========================================================")
        saver.save(sess, path)
        print("Model saved at {}".format(path))
    print("Training done")

```

接下来，按顺序调用已定义的函数。首先训练正向模型，然后训练反向模型，最后训练策略梯度。

```py
train(forward_, forward_epochs, False)
train(reverse_, reverse_epochs, False)
pg_train(100, False)
```

这标志着聊天机器人的训练结束。模型通过正向和反向训练

# 测试和结果

训练模型后，我们用测试数据集进行了测试，得到了相当连贯的对话。有一个非常重要的问题：交流的上下文。因此，根据所使用的数据集，结果会有其上下文。就我们的上下文而言，获得的结果非常合理，并且满足了我们的三项性能指标——信息量（无重复回合）、高度连贯性和回答的简洁性（这与前瞻性功能有关）。

```py
import data_parser
from gensim.models import KeyedVectors
from seq_model import Chatbot
import tensorflow as tf
import numpy as np
import helper as h
```

接下来，声明已经训练好的各种模型的路径。

```py
reinforcement_model_path = "model/reinforcement/seq2seq"
forward_model_path = "model/forward/seq2seq"
reverse_model_path = "model/reverse/seq2seq"
```

接下来，声明包含问题和回应的文件路径。

```py
path_to_questions = 'results/sample_input.txt'
responses_path = 'results/sample_output_RL.txt'
```

接下来，声明模型所需的常量。

```py
word_count_threshold = 20
dim_wordvec = 300
dim_hidden = 1000

input_sequence_length = 25
target_sequence_length = 22

batch_size = 2

```

接下来，加载数据和模型，如下所示：

```py
def test(model_path=forward_model_path):
    testing_data = open(path_to_questions, 'r').read().split('\n')
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

    _, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

    model = Chatbot(dim_wordvec, len(index_to_word), dim_hidden, batch_size,
                            input_sequence_length, target_sequence_length, Training=False)

    place_holders, predictions, logits = model.build_model()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    saver.restore(sess, model_path)
```

接下来，打开回应文件，并准备如下所示的问题列表：

```py
    with open(responses_path, 'w') as out:

        for idx, question in enumerate(testing_data):
            print('question =>', question)

            question = [h.refine(w) for w in question.lower().split()]
            question = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in question]
            question.insert(0, np.random.normal(size=(dim_wordvec,)))  # insert random normal at the first step

            if len(question) > input_sequence_length:
                question = question[:input_sequence_length]
            else:
                for _ in range(input_sequence_length - len(question)):
                    question.append(np.zeros(dim_wordvec))

            question = np.array([question])

            feed_dict = {place_holders["word_vectors"]: np.concatenate([question] * 2, 0),
                         }

            word_indices, prob_logit = sess.run([predictions, logits], feed_dict=feed_dict)

            # print(word_indices[0].shape)
            generated_sentence = h.index2sentence(word_indices[0], prob_logit[0], index_to_word)

            print('generated_sentence =>', generated_sentence)
            out.write(generated_sentence + '\n')

test(reinforcement_model_path)
```

通过传递模型的路径，我们可以测试聊天机器人以获取各种回应。

# 总结

聊天机器人正在迅速席卷全球，预计在未来几年将变得更加普及。如果要获得广泛的接受，这些聊天机器人通过对话得到的结果的连贯性必须不断提高。实现这一目标的一种方式是通过使用强化学习。

在本章中，我们实现了在创建聊天机器人过程中使用强化学习。该学习方法基于一种政策梯度方法，重点关注对话代理的未来方向，以生成连贯且有趣的互动。我们使用的数据集来自电影对话。我们对数据集进行了清理和预处理，从中获取了词汇表。然后，我们制定了我们的政策梯度方法。我们的奖励函数通过一个序列到序列模型表示。接着，我们训练并测试了我们的数据，获得了非常合理的结果，证明了使用强化学习进行对话代理的可行性。
