# 第八章：目标检测 – CIFAR-10 示例

在介绍了**卷积神经网络**（**CNNs**）的基础和直觉/动机后，我们将在目标检测中展示其在其中一个最流行的数据集上的应用。我们还将看到 CNN 的初始层获取对象的基本特征，而最终的卷积层将从第一层的这些基本特征构建更语义级别的特征。

本章将涵盖以下主题：

+   目标检测

+   CIFAR-10 图像中对象检测—模型构建和训练

# 目标检测

维基百科指出：

"目标检测是计算机视觉领域中用于在图像或视频序列中查找和识别对象的技术。人类在图像中识别多种对象并不费力，尽管对象的图像在不同视角、大小和比例以及平移或旋转时可能有所变化。即使对象部分遮挡视图时，也能识别出对象。对计算机视觉系统而言，这仍然是一个挑战。多年来已实施了多种方法来解决此任务。"

图像分析是深度学习中最显著的领域之一。图像易于生成和处理，它们恰好是机器学习的正确数据类型：对人类易于理解，但对计算机而言却很难。不奇怪，图像分析在深度神经网络的历史中发挥了关键作用。

![](img/0d82c9b0-e87a-4f16-a94c-805efcf6f9f5.jpg)

图 11.1：检测对象的示例。来源：B. C. Russell, A. Torralba, C. Liu, R. Fergus, W. T. Freeman，《通过场景对齐进行对象检测》，2007 年进展神经信息处理系统，网址：http://bryanrussell.org/papers/nipsDetectionBySceneAlignment07.pdf

随着自动驾驶汽车、面部检测、智能视频监控和人数统计解决方案的兴起，快速准确的目标检测系统需求量大。这些系统不仅包括图像中对象的识别和分类，还可以通过绘制适当的框来定位每个对象。这使得目标检测比传统的计算机视觉前身——图像分类更为复杂。

在本章中，我们将讨论目标检测——找出图像中有哪些对象。例如，想象一下自动驾驶汽车需要在道路上检测其他车辆，就像*图 11.1*中一样。目标检测有许多复杂的算法。它们通常需要庞大的数据集、非常深的卷积网络和长时间的训练。

# CIFAR-10 – 建模、构建和训练

此示例展示了如何在 CIFAR-10 数据集中制作用于分类图像的 CNN。我们将使用一个简单的卷积神经网络实现一些卷积和全连接层。

即使网络架构非常简单，当尝试检测 CIFAR-10 图像中的对象时，您会看到它表现得有多好。

所以，让我们开始这个实现。

# 使用的包

我们导入了此实现所需的所有包：

```py
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

import pickle
import tensorflow as tf
```

# 载入 CIFAR-10 数据集

在这个实现中，我们将使用 CIFAR-10 数据集，这是用于对象检测的最常用的数据集之一。因此，让我们先定义一个辅助类来下载和提取 CIFAR-10 数据集（如果尚未下载）：

```py
cifar10_batches_dir_path = 'cifar-10-batches-py'

tar_gz_filename = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Python Images Batches') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_filename,
            pbar.hook)

if not isdir(cifar10_batches_dir_path):
    with tarfile.open(tar_gz_filename) as tar:
        tar.extractall()
        tar.close()
```

下载并提取 CIFAR-10 数据集后，您会发现它已经分成了五个批次。CIFAR-10 包含了 10 个类别的图像：

+   `airplane`

+   `automobile`

+   `bird`

+   `cat`

+   `deer`

+   `dog`

+   `frog`

+   `horse`

+   `ship`

+   `truck`

在我们深入构建网络核心之前，让我们进行一些数据分析和预处理。

# 数据分析和预处理

我们需要分析数据集并进行一些基本的预处理。因此，让我们首先定义一些辅助函数，这些函数将使我们能够从我们有的五批次中加载特定批次，并打印关于此批次及其样本的一些分析：

```py
# Defining a helper function for loading a batch of images
def load_batch(cifar10_dataset_dir_path, batch_num):

    with open(cifar10_dataset_dir_path + '/data_batch_' + str(batch_num), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    input_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    target_labels = batch['labels']

    return input_features, target_labels
```

然后，我们定义一个函数，可以帮助我们显示特定批次中特定样本的统计信息：

```py
#Defining a function to show the stats for batch ans specific sample
def batch_image_stats(cifar10_dataset_dir_path, batch_num, sample_num):

    batch_nums = list(range(1, 6))

    #checking if the batch_num is a valid batch number
    if batch_num not in batch_nums:
        print('Batch Num is out of Range. You can choose from these Batch nums: {}'.format(batch_nums))
        return None

    input_features, target_labels = load_batch(cifar10_dataset_dir_path, batch_num)

    #checking if the sample_num is a valid sample number
    if not (0 <= sample_num < len(input_features)):
        print('{} samples in batch {}. {} is not a valid sample number.'.format(len(input_features), batch_num, sample_num))
        return None

    print('\nStatistics of batch number {}:'.format(batch_num))
    print('Number of samples in this batch: {}'.format(len(input_features)))
    print('Per class counts of each Label: {}'.format(dict(zip(*np.unique(target_labels, return_counts=True)))))

    image = input_features[sample_num]
    label = target_labels[sample_num]
    cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print('\nSample Image Number {}:'.format(sample_num))
    print('Sample image - Minimum pixel value: {} Maximum pixel value: {}'.format(image.min(), image.max()))
    print('Sample image - Shape: {}'.format(image.shape))
    print('Sample Label - Label Id: {} Name: {}'.format(label, cifar10_class_names[label]))
    plt.axis('off')
    plt.imshow(image)
```

现在，我们可以使用这个函数来操作我们的数据集，并可视化特定的图像：

```py
# Explore a specific batch and sample from the dataset
batch_num = 3
sample_num = 6
batch_image_stats(cifar10_batches_dir_path, batch_num, sample_num)
```

输出如下：

```py

Statistics of batch number 3:
Number of samples in this batch: 10000
Per class counts of each Label: {0: 994, 1: 1042, 2: 965, 3: 997, 4: 990, 5: 1029, 6: 978, 7: 1015, 8: 961, 9: 1029}

Sample Image Number 6:
Sample image - Minimum pixel value: 30 Maximum pixel value: 242
Sample image - Shape: (32, 32, 3)
Sample Label - Label Id: 8 Name: ship
```

![](img/ae827baa-2394-4eaf-93cd-018a7ae7b8a1.png)

图 11.2: 来自第 3 批次的样本图像 6

在继续将数据集馈送到模型之前，我们需要将其归一化到零到一的范围内。

批归一化优化了网络训练。已经显示它有几个好处：

+   **训练更快**：每个训练步骤会变慢，因为在网络的前向传播过程中需要额外的计算，在网络的反向传播过程中需要训练额外的超参数。然而，它应该会更快地收敛，因此总体训练速度应该更快。

+   **更高的学习率**：梯度下降算法通常需要较小的学习率才能使网络收敛到损失函数的最小值。随着神经网络变得更深，它们在反向传播过程中的梯度值会变得越来越小，因此通常需要更多的迭代次数。使用批归一化的想法允许我们使用更高的学习率，这进一步增加了网络训练的速度。

+   **权重初始化简单**: 权重初始化可能会很困难，特别是在使用深度神经网络时。批归一化似乎使我们在选择初始权重时可以更不谨慎。

因此，让我们继续定义一个函数，该函数将负责将输入图像列表归一化，以便这些图像的所有像素值都在零到一之间。

```py
#Normalize CIFAR-10 images to be in the range of [0,1]

def normalize_images(images):

    # initial zero ndarray
    normalized_images = np.zeros_like(images.astype(float))

    # The first images index is number of images where the other indices indicates
    # hieight, width and depth of the image
    num_images = images.shape[0]

    # Computing the minimum and maximum value of the input image to do the normalization based on them
    maximum_value, minimum_value = images.max(), images.min()

    # Normalize all the pixel values of the images to be from 0 to 1
    for img in range(num_images):
        normalized_images[img,...] = (images[img, ...] - float(minimum_value)) / float(maximum_value - minimum_value)

    return normalized_images
```

接下来，我们需要实现另一个辅助函数，对输入图像的标签进行编码。在这个函数中，我们将使用 sklearn 的独热编码（one-hot encoding），其中每个图像标签通过一个零向量表示，除了该向量所代表的图像的类别索引。

输出向量的大小将取决于数据集中的类别数量，对于 CIFAR-10 数据集来说是 10 个类别：

```py
#encoding the input images. Each image will be represented by a vector of zeros except for the class index of the image 
# that this vector represents. The length of this vector depends on number of classes that we have
# the dataset which is 10 in CIFAR-10

def one_hot_encode(images):

    num_classes = 10

    #use sklearn helper function of OneHotEncoder() to do that
    encoder = OneHotEncoder(num_classes)

    #resize the input images to be 2D
    input_images_resized_to_2d = np.array(images).reshape(-1,1)
    one_hot_encoded_targets = encoder.fit_transform(input_images_resized_to_2d)

    return one_hot_encoded_targets.toarray()
```

现在，是时候调用之前的辅助函数进行预处理并保存数据集，以便我们以后可以使用它了：

```py
def preprocess_persist_data(cifar10_batches_dir_path, normalize_images, one_hot_encode):

    num_batches = 5
    valid_input_features = []
    valid_target_labels = []

    for batch_ind in range(1, num_batches + 1):

        #Loading batch
        input_features, target_labels = load_batch(cifar10_batches_dir_path, batch_ind)
        num_validation_images = int(len(input_features) * 0.1)

        # Preprocess the current batch and perisist it for future use
        input_features = normalize_images(input_features[:-num_validation_images])
        target_labels = one_hot_encode( target_labels[:-num_validation_images])

        #Persisting the preprocessed batch
        pickle.dump((input_features, target_labels), open('preprocess_train_batch_' + str(batch_ind) + '.p', 'wb'))

        # Define a subset of the training images to be used for validating our model
        valid_input_features.extend(input_features[-num_validation_images:])
        valid_target_labels.extend(target_labels[-num_validation_images:])

    # Preprocessing and persisting the validationi subset
    input_features = normalize_images( np.array(valid_input_features))
    target_labels = one_hot_encode(np.array(valid_target_labels))

    pickle.dump((input_features, target_labels), open('preprocess_valid.p', 'wb'))

    #Now it's time to preporcess and persist the test batche
    with open(cifar10_batches_dir_path + '/test_batch', mode='rb') as file:
        test_batch = pickle.load(file, encoding='latin1')

    test_input_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_input_labels = test_batch['labels']

    # Normalizing and encoding the test batch
    input_features = normalize_images( np.array(test_input_features))
    target_labels = one_hot_encode(np.array(test_input_labels))

    pickle.dump((input_features, target_labels), open('preprocess_test.p', 'wb'))

# Calling the helper function above to preprocess and persist the training, validation, and testing set
preprocess_persist_data(cifar10_batches_dir_path, normalize_images, one_hot_encode)
```

现在，我们已经将预处理数据保存到磁盘。

我们还需要加载验证集，以便在训练过程的不同 epoch 上运行训练好的模型：

```py
# Load the Preprocessed Validation data
valid_input_features, valid_input_labels = pickle.load(open('preprocess_valid.p', mode='rb'))
```

# 构建网络

现在是时候构建我们分类应用程序的核心，即 CNN 架构的计算图了，但为了最大化该实现的优势，我们不会使用 TensorFlow 层 API，而是将使用它的神经网络版本。

所以，让我们从定义模型输入占位符开始，这些占位符将输入图像、目标类别以及 dropout 层的保留概率参数（这有助于我们通过丢弃一些连接来减少架构的复杂性，从而减少过拟合的几率）：

```py

# Defining the model inputs
def images_input(img_shape):
 return tf.placeholder(tf.float32, (None, ) + img_shape, name="input_images")

def target_input(num_classes):

 target_input = tf.placeholder(tf.int32, (None, num_classes), name="input_images_target")
 return target_input

#define a function for the dropout layer keep probability
def keep_prob_input():
 return tf.placeholder(tf.float32, name="keep_prob")
```

接下来，我们需要使用 TensorFlow 神经网络实现版本来构建我们的卷积层，并进行最大池化：

```py
# Applying a convolution operation to the input tensor followed by max pooling
def conv2d_layer(input_tensor, conv_layer_num_outputs, conv_kernel_size, conv_layer_strides, pool_kernel_size, pool_layer_strides):

 input_depth = input_tensor.get_shape()[3].value
 weight_shape = conv_kernel_size + (input_depth, conv_layer_num_outputs,)

 #Defining layer weights and biases
 weights = tf.Variable(tf.random_normal(weight_shape))
 biases = tf.Variable(tf.random_normal((conv_layer_num_outputs,)))

 #Considering the biase variable
 conv_strides = (1,) + conv_layer_strides + (1,)

 conv_layer = tf.nn.conv2d(input_tensor, weights, strides=conv_strides, padding='SAME')
 conv_layer = tf.nn.bias_add(conv_layer, biases)

 conv_kernel_size = (1,) + conv_kernel_size + (1,)

 pool_strides = (1,) + pool_layer_strides + (1,)
 pool_layer = tf.nn.max_pool(conv_layer, ksize=conv_kernel_size, strides=pool_strides, padding='SAME')
 return pool_layer
```

正如你可能在前一章中看到的，最大池化操作的输出是一个 4D 张量，这与全连接层所需的输入格式不兼容。因此，我们需要实现一个展平层，将最大池化层的输出从 4D 转换为 2D 张量：

```py
#Flatten the output of max pooling layer to be fing to the fully connected layer which only accepts the output
# to be in 2D
def flatten_layer(input_tensor):
return tf.contrib.layers.flatten(input_tensor)
```

接下来，我们需要定义一个辅助函数，允许我们向架构中添加一个全连接层：

```py
#Define the fully connected layer that will use the flattened output of the stacked convolution layers
#to do the actuall classification
def fully_connected_layer(input_tensor, num_outputs):
 return tf.layers.dense(input_tensor, num_outputs)
```

最后，在使用这些辅助函数创建整个架构之前，我们需要创建另一个函数，它将接收全连接层的输出并产生 10 个实值，对应于我们数据集中类别的数量：

```py
#Defining the output function
def output_layer(input_tensor, num_outputs):
    return  tf.layers.dense(input_tensor, num_outputs)
```

所以，让我们定义一个函数，把所有这些部分组合起来，创建一个具有三个卷积层的 CNN。每个卷积层后面都会跟随一个最大池化操作。我们还会有两个全连接层，每个全连接层后面都会跟一个 dropout 层，以减少模型复杂性并防止过拟合。最后，我们将有一个输出层，产生 10 个实值向量，每个值代表每个类别的得分，表示哪个类别是正确的：

```py
def build_convolution_net(image_data, keep_prob):

 # Applying 3 convolution layers followed by max pooling layers
 conv_layer_1 = conv2d_layer(image_data, 32, (3,3), (1,1), (3,3), (3,3)) 
 conv_layer_2 = conv2d_layer(conv_layer_1, 64, (3,3), (1,1), (3,3), (3,3))
 conv_layer_3 = conv2d_layer(conv_layer_2, 128, (3,3), (1,1), (3,3), (3,3))

# Flatten the output from 4D to 2D to be fed to the fully connected layer
 flatten_output = flatten_layer(conv_layer_3)

# Applying 2 fully connected layers with drop out
 fully_connected_layer_1 = fully_connected_layer(flatten_output, 64)
 fully_connected_layer_1 = tf.nn.dropout(fully_connected_layer_1, keep_prob)
 fully_connected_layer_2 = fully_connected_layer(fully_connected_layer_1, 32)
 fully_connected_layer_2 = tf.nn.dropout(fully_connected_layer_2, keep_prob)

 #Applying the output layer while the output size will be the number of categories that we have
 #in CIFAR-10 dataset
 output_logits = output_layer(fully_connected_layer_2, 10)

 #returning output
 return output_logits
```

让我们调用之前的辅助函数来构建网络并定义它的损失和优化标准：

```py
#Using the helper function above to build the network

#First off, let's remove all the previous inputs, weights, biases form the previous runs
tf.reset_default_graph()

# Defining the input placeholders to the convolution neural network
input_images = images_input((32, 32, 3))
input_images_target = target_input(10)
keep_prob = keep_prob_input()

# Building the models
logits_values = build_convolution_net(input_images, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits_values = tf.identity(logits_values, name='logits')

# defining the model loss
model_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_values, labels=input_images_target))

# Defining the model optimizer
model_optimizer = tf.train.AdamOptimizer().minimize(model_cost)

# Calculating and averaging the model accuracy
correct_prediction = tf.equal(tf.argmax(logits_values, 1), tf.argmax(input_images_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='model_accuracy')
tests.test_conv_net(build_convolution_net)
```

现在，我们已经构建了该网络的计算架构，是时候启动训练过程并查看一些结果了。

# 模型训练

因此，让我们定义一个辅助函数，使我们能够启动训练过程。这个函数将接受输入图像、目标类别的独热编码以及保持概率值作为输入。然后，它将把这些值传递给计算图，并调用模型优化器：

```py
#Define a helper function for kicking off the training process
def train(session, model_optimizer, keep_probability, in_feature_batch, target_batch):
session.run(model_optimizer, feed_dict={input_images: in_feature_batch, input_images_target: target_batch, keep_prob: keep_probability})
```

我们需要在训练过程中的不同时间点验证模型，因此我们将定义一个辅助函数，打印出模型在验证集上的准确率：

```py
#Defining a helper funcitno for print information about the model accuracy and it's validation accuracy as well
def print_model_stats(session, input_feature_batch, target_label_batch, model_cost, model_accuracy):

    validation_loss = session.run(model_cost, feed_dict={input_images: input_feature_batch, input_images_target: target_label_batch, keep_prob: 1.0})
    validation_accuracy = session.run(model_accuracy, feed_dict={input_images: input_feature_batch, input_images_target: target_label_batch, keep_prob: 1.0})

    print("Valid Loss: %f" %(validation_loss))
    print("Valid accuracy: %f" % (validation_accuracy))
```

让我们还定义一些模型的超参数，这些参数可以帮助我们调整模型以获得更好的性能：

```py
# Model Hyperparameters
num_epochs = 100
batch_size = 128
keep_probability = 0.5
```

现在，让我们启动训练过程，但只针对 CIFAR-10 数据集的单一批次，看看基于该批次的模型准确率。

然而，在此之前，我们将定义一个辅助函数，加载一个批次的训练数据，并将输入图像与目标类别分开：

```py
# Splitting the dataset features and labels to batches
def batch_split_features_labels(input_features, target_labels, train_batch_size):
    for start in range(0, len(input_features), train_batch_size):
        end = min(start + train_batch_size, len(input_features))
        yield input_features[start:end], target_labels[start:end]

#Loading the persisted preprocessed training batches
def load_preprocess_training_batch(batch_id, batch_size):
    filename = 'preprocess_train_batch_' + str(batch_id) + '.p'
    input_features, target_labels = pickle.load(open(filename, mode='rb'))

    # Returning the training images in batches according to the batch size defined above
    return batch_split_features_labels(input_features, target_labels, train_batch_size)
```

现在，让我们开始一个批次的训练过程：

```py
print('Training on only a Single Batch from the CIFAR-10 Dataset...')
with tf.Session() as sess:

 # Initializing the variables
 sess.run(tf.global_variables_initializer())

 # Training cycle
 for epoch in range(num_epochs):
 batch_ind = 1

 for batch_features, batch_labels in load_preprocess_training_batch(batch_ind, batch_size):
 train(sess, model_optimizer, keep_probability, batch_features, batch_labels)

 print('Epoch number {:>2}, CIFAR-10 Batch Number {}: '.format(epoch + 1, batch_ind), end='')
 print_model_stats(sess, batch_features, batch_labels, model_cost, accuracy)

Output:
.
.
.
Epoch number 85, CIFAR-10 Batch Number 1: Valid Loss: 1.490792
Valid accuracy: 0.550000
Epoch number 86, CIFAR-10 Batch Number 1: Valid Loss: 1.487118
Valid accuracy: 0.525000
Epoch number 87, CIFAR-10 Batch Number 1: Valid Loss: 1.309082
Valid accuracy: 0.575000
Epoch number 88, CIFAR-10 Batch Number 1: Valid Loss: 1.446488
Valid accuracy: 0.475000
Epoch number 89, CIFAR-10 Batch Number 1: Valid Loss: 1.430939
Valid accuracy: 0.550000
Epoch number 90, CIFAR-10 Batch Number 1: Valid Loss: 1.484480
Valid accuracy: 0.525000
Epoch number 91, CIFAR-10 Batch Number 1: Valid Loss: 1.345774
Valid accuracy: 0.575000
Epoch number 92, CIFAR-10 Batch Number 1: Valid Loss: 1.425942
Valid accuracy: 0.575000

Epoch number 93, CIFAR-10 Batch Number 1: Valid Loss: 1.451115
Valid accuracy: 0.550000
Epoch number 94, CIFAR-10 Batch Number 1: Valid Loss: 1.368719
Valid accuracy: 0.600000
Epoch number 95, CIFAR-10 Batch Number 1: Valid Loss: 1.336483
Valid accuracy: 0.600000
Epoch number 96, CIFAR-10 Batch Number 1: Valid Loss: 1.383425
Valid accuracy: 0.575000
Epoch number 97, CIFAR-10 Batch Number 1: Valid Loss: 1.378877
Valid accuracy: 0.625000
Epoch number 98, CIFAR-10 Batch Number 1: Valid Loss: 1.343391
Valid accuracy: 0.600000
Epoch number 99, CIFAR-10 Batch Number 1: Valid Loss: 1.319342
Valid accuracy: 0.625000
Epoch number 100, CIFAR-10 Batch Number 1: Valid Loss: 1.340849
Valid accuracy: 0.525000
```

如你所见，仅在单一批次上训练时，验证准确率并不高。让我们看看仅通过完整训练过程，验证准确率会如何变化：

```py
model_save_path = './cifar-10_classification'

with tf.Session() as sess:
 # Initializing the variables
 sess.run(tf.global_variables_initializer())

 # Training cycle
 for epoch in range(num_epochs):

 # iterate through the batches
 num_batches = 5

 for batch_ind in range(1, num_batches + 1):
 for batch_features, batch_labels in load_preprocess_training_batch(batch_ind, batch_size):
 train(sess, model_optimizer, keep_probability, batch_features, batch_labels)

 print('Epoch number{:>2}, CIFAR-10 Batch Number {}: '.format(epoch + 1, batch_ind), end='')
 print_model_stats(sess, batch_features, batch_labels, model_cost, accuracy)

 # Save the trained Model
 saver = tf.train.Saver()
 save_path = saver.save(sess, model_save_path)

Output:
.
.
.
Epoch number94, CIFAR-10 Batch Number 5: Valid Loss: 0.316593
Valid accuracy: 0.925000
Epoch number95, CIFAR-10 Batch Number 1: Valid Loss: 0.285429
Valid accuracy: 0.925000
Epoch number95, CIFAR-10 Batch Number 2: Valid Loss: 0.347411
Valid accuracy: 0.825000
Epoch number95, CIFAR-10 Batch Number 3: Valid Loss: 0.232483
Valid accuracy: 0.950000
Epoch number95, CIFAR-10 Batch Number 4: Valid Loss: 0.294707
Valid accuracy: 0.900000
Epoch number95, CIFAR-10 Batch Number 5: Valid Loss: 0.299490
Valid accuracy: 0.975000
Epoch number96, CIFAR-10 Batch Number 1: Valid Loss: 0.302191
Valid accuracy: 0.950000
Epoch number96, CIFAR-10 Batch Number 2: Valid Loss: 0.347043
Valid accuracy: 0.750000
Epoch number96, CIFAR-10 Batch Number 3: Valid Loss: 0.252851
Valid accuracy: 0.875000
Epoch number96, CIFAR-10 Batch Number 4: Valid Loss: 0.291433
Valid accuracy: 0.950000
Epoch number96, CIFAR-10 Batch Number 5: Valid Loss: 0.286192
Valid accuracy: 0.950000
Epoch number97, CIFAR-10 Batch Number 1: Valid Loss: 0.277105
Valid accuracy: 0.950000
Epoch number97, CIFAR-10 Batch Number 2: Valid Loss: 0.305842
Valid accuracy: 0.850000
Epoch number97, CIFAR-10 Batch Number 3: Valid Loss: 0.215272
Valid accuracy: 0.950000
Epoch number97, CIFAR-10 Batch Number 4: Valid Loss: 0.313761
Valid accuracy: 0.925000
Epoch number97, CIFAR-10 Batch Number 5: Valid Loss: 0.313503
Valid accuracy: 0.925000
Epoch number98, CIFAR-10 Batch Number 1: Valid Loss: 0.265828
Valid accuracy: 0.925000
Epoch number98, CIFAR-10 Batch Number 2: Valid Loss: 0.308948
Valid accuracy: 0.800000
Epoch number98, CIFAR-10 Batch Number 3: Valid Loss: 0.232083
Valid accuracy: 0.950000
Epoch number98, CIFAR-10 Batch Number 4: Valid Loss: 0.298826
Valid accuracy: 0.925000
Epoch number98, CIFAR-10 Batch Number 5: Valid Loss: 0.297230
Valid accuracy: 0.950000
Epoch number99, CIFAR-10 Batch Number 1: Valid Loss: 0.304203
Valid accuracy: 0.900000
Epoch number99, CIFAR-10 Batch Number 2: Valid Loss: 0.308775
Valid accuracy: 0.825000
Epoch number99, CIFAR-10 Batch Number 3: Valid Loss: 0.225072
Valid accuracy: 0.925000
Epoch number99, CIFAR-10 Batch Number 4: Valid Loss: 0.263737
Valid accuracy: 0.925000
Epoch number99, CIFAR-10 Batch Number 5: Valid Loss: 0.278601
Valid accuracy: 0.950000
Epoch number100, CIFAR-10 Batch Number 1: Valid Loss: 0.293509
Valid accuracy: 0.950000
Epoch number100, CIFAR-10 Batch Number 2: Valid Loss: 0.303817
Valid accuracy: 0.875000
Epoch number100, CIFAR-10 Batch Number 3: Valid Loss: 0.244428
Valid accuracy: 0.900000
Epoch number100, CIFAR-10 Batch Number 4: Valid Loss: 0.280712
Valid accuracy: 0.925000
Epoch number100, CIFAR-10 Batch Number 5: Valid Loss: 0.278625
Valid accuracy: 0.950000
```

# 测试模型

让我们在 CIFAR-10 数据集的测试集部分上测试训练好的模型。首先，我们将定义一个辅助函数，帮助我们可视化一些示例图像的预测结果及其对应的真实标签：

```py
#A helper function to visualize some samples and their corresponding predictions
def display_samples_predictions(input_features, target_labels, samples_predictions):

 num_classes = 10

 cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

label_binarizer = LabelBinarizer()
 label_binarizer.fit(range(num_classes))
 label_inds = label_binarizer.inverse_transform(np.array(target_labels))

fig, axies = plt.subplots(nrows=4, ncols=2)
 fig.tight_layout()
 fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

num_predictions = 4
 margin = 0.05
 ind = np.arange(num_predictions)
 width = (1\. - 2\. * margin) / num_predictions

for image_ind, (feature, label_ind, prediction_indicies, prediction_values) in enumerate(zip(input_features, label_inds, samples_predictions.indices, samples_predictions.values)):
 prediction_names = [cifar10_class_names[pred_i] for pred_i in prediction_indicies]
 correct_name = cifar10_class_names[label_ind]

axies[image_ind][0].imshow(feature)
 axies[image_ind][0].set_title(correct_name)
 axies[image_ind][0].set_axis_off()

axies[image_ind][1].barh(ind + margin, prediction_values[::-1], width)
 axies[image_ind][1].set_yticks(ind + margin)
 axies[image_ind][1].set_yticklabels(prediction_names[::-1])
 axies[image_ind][1].set_xticks([0, 0.5, 1.0])
```

现在，让我们恢复训练好的模型并对测试集进行测试：

```py
test_batch_size = 64
save_model_path = './cifar-10_classification'
#Number of images to visualize
num_samples = 4

#Number of top predictions
top_n_predictions = 4

#Defining a helper function for testing the trained model
def test_classification_model():

 input_test_features, target_test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
 loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:

 # loading the trained model
 model = tf.train.import_meta_graph(save_model_path + '.meta')
 model.restore(sess, save_model_path)

# Getting some input and output Tensors from loaded model
 model_input_values = loaded_graph.get_tensor_by_name('input_images:0')
 model_target = loaded_graph.get_tensor_by_name('input_images_target:0')
 model_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
 model_logits = loaded_graph.get_tensor_by_name('logits:0')
 model_accuracy = loaded_graph.get_tensor_by_name('model_accuracy:0')

 # Testing the trained model on the test set batches
 test_batch_accuracy_total = 0
 test_batch_count = 0

 for input_test_feature_batch, input_test_label_batch in batch_split_features_labels(input_test_features, target_test_labels, test_batch_size):
 test_batch_accuracy_total += sess.run(
 model_accuracy,
 feed_dict={model_input_values: input_test_feature_batch, model_target: input_test_label_batch, model_keep_prob: 1.0})
 test_batch_count += 1

print('Test set accuracy: {}\n'.format(test_batch_accuracy_total/test_batch_count))

# print some random images and their corresponding predictions from the test set results
 random_input_test_features, random_test_target_labels = tuple(zip(*random.sample(list(zip(input_test_features, target_test_labels)), num_samples)))

 random_test_predictions = sess.run(
 tf.nn.top_k(tf.nn.softmax(model_logits), top_n_predictions),
 feed_dict={model_input_values: random_input_test_features, model_target: random_test_target_labels, model_keep_prob: 1.0})

 display_samples_predictions(random_input_test_features, random_test_target_labels, random_test_predictions)

#Calling the function
test_classification_model()

Output:
INFO:tensorflow:Restoring parameters from ./cifar-10_classification
Test set accuracy: 0.7540007961783439
```

![](img/fc105794-646e-4a0a-8fe0-c7c432a1ccca.png)

让我们可视化另一个例子，看看一些错误：

![](img/ee5f68c7-4a4e-40f9-9009-a78465f6559e.png)

现在，我们的测试准确率大约为 75%，对于像我们使用的简单 CNN 来说，这并不算差。

# 总结

本章向我们展示了如何制作一个 CNN 来分类 CIFAR-10 数据集中的图像。测试集上的分类准确率约为 79% - 80%。卷积层的输出也被绘制出来，但很难看出神经网络是如何识别和分类输入图像的。需要更好的可视化技术。

接下来，我们将使用深度学习中的一种现代且激动人心的实践方法——迁移学习。迁移学习使你能够使用深度学习中的数据需求大的架构，适用于小型数据集。
