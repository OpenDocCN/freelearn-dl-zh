# 第三章：实现深度神经网络架构来识别手写数字

在前面的章节中，我们已经了解了必要的概念，并设置了启动我们人工智能（**AI**）之旅所需的工具。我们还构建了一个小型预测应用程序，以便熟悉我们将要使用的工具。

在本章中，我们将讨论 AI 的一个更有趣且更受欢迎的应用——计算机视觉，或称机器视觉。我们将从上一章继续，逐步过渡到构建**卷积神经网络**（**CNN**），这是计算机视觉中最流行的神经网络类型。本章还将涵盖第一章中承诺的人工智能概念和基础内容，但与之不同的是，本章将采取非常实践的方式。

本章将涵盖以下主题：

+   构建一个前馈神经网络来识别手写数字

+   神经网络的其余概念

+   构建一个更深层的神经网络

+   计算机视觉简介

# 构建一个前馈神经网络来识别手写数字，版本一

在这一节中，我们将运用前两章所学的知识来解决一个包含非结构化数据的问题——图像分类。我们的思路是，通过当前的设置和我们熟悉的神经网络基础，深入解决计算机视觉任务。我们已经看到，前馈神经网络可以用于使用结构化数据进行预测；接下来，我们就用它来分类手写数字图像。

为了解决这个任务，我们将利用**MNSIT**数据库，并使用手写数字数据集。MNSIT 代表的是**修改后的国家标准与技术研究院**（Modified National Institute of Standards and Technology）。这是一个大型数据库，通常用于训练、测试和基准化与计算机视觉相关的图像任务。

MNSIT 数字数据集包含 60,000 张手写数字图像，用于训练模型，还有 10,000 张手写数字图像，用于测试模型。

从现在开始，我们将使用 Jupyter Notebook 来理解和执行这项任务。所以，如果你还没有启动 Jupyter Notebook，请启动它并创建一个新的 Python Notebook。

一旦你的 Notebook 准备好，第一件要做的，和往常一样，是导入所有必需的模块：

1.  导入`numpy`并设置`seed`以确保结果可复现：

```py
import numpy as np np.random.seed(42)
```

1.  加载 Keras 依赖项和内置的 MNSIT 数字数据集：

```py
import keras from keras.datasets import mnist  
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
```

1.  将数据分别加载到训练集和测试集中：

```py
(X_train, y_train), (X_test, y_test)= mnist.load_data()
```

1.  检查训练图像的数量以及每张图像的大小。在这个案例中，每张图像的大小是 28 x 28 像素：

```py
X_train.shape
(60000, 28, 28)
```

1.  检查因变量，在这种情况下，包含 60,000 个带有正确标签的案例：

```py
y_train.shape
(60000,)
```

1.  检查前 100 个训练样本的标签：

```py
y_train [0 :99]  
array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9,
       1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9,
       8, 5, 9, 3, 3, 0, 7, 4, 9, 8, 0, 9, 4, 1, 4, 4, 6, 0, 4, 5, 6, 1, 0,
       0, 1, 7, 1, 6, 3, 0, 2, 1, 1, 7, 9, 0, 2, 6, 7, 8, 3, 9, 0, 4, 6, 7,
       4, 6, 8, 0, 7, 8, 3], dtype=uint8)
```

1.  检查测试图像的数量以及每张图像的大小。在本例中，每张图像的大小是 28 x 28 像素：

```py
X_test.shape
(10000, 28, 28)
```

1.  检查测试数据中的样本，这些基本上是 28 x 28 大小的二维数组：

```py
X_test[0]
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,      
          .
          .
,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 207,
         18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype=uint8)
```

1.  检查因变量，在本例中是 10,000 个带有正确标签的案例：

```py
y_test.shape
(10000,)
```

1.  测试集中第一个样本的正确标签如下：

```py
y_test[0]
7
```

1.  现在，我们需要对数据进行预处理，将其从 28 x 28 的二维数组转换为归一化的 784 个元素的一维数组：

```py
X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')
X_train/=255
X_test /=255
```

1.  检查预处理数据集的第一个样本：

```py
X_test[0]
array([ 0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        .
        .
        .
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0.47450981,  0.99607843,  0.99607843,  0.85882354,  0.15686275,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0.47450981,  0.99607843,
        0.81176472,  0.07058824,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ,  0\.        ,
        0\.        ,  0\.        ,  0\.        ,  0\.        ], dtype=float32)
```

1.  下一步是对标签进行一热编码；换句话说，我们需要将标签（从零到九）的数据类型从数字转换为类别型：

```py
n_classes=10
y_train= keras.utils.to_categorical(y_train ,n_classes)
y_test= keras.utils.to_categorical(y_test,n_classes)
```

1.  查看已经进行过一热编码的标签的第一个样本。在这种情况下，数字是七：

```py
y_test[0]
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])
```

1.  现在，我们需要设计一个简单的前馈神经网络，输入层使用`sigmoid`激活函数和 64 个神经元。我们将在输出层添加一个`softmax`函数，通过给出分类标签的概率来进行分类：

```py
model=Sequential()
model.add(Dense(64,activation='sigmoid', input_shape=(784,)))
model.add(Dense(10,activation='softmax'))  
```

1.  我们可以通过`summary()`函数查看我们刚刚设计的神经网络的结构，这是一个简单的网络，具有 64 个神经元的输入层和 10 个神经元的输出层。输出层有 10 个神经元，我们有 10 个分类标签需要预测/分类（从零到九）：

```py
model.summary()
_______________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 64)                50240     
_______________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 50,890
Trainable params: 50,890
Non-trainable params: 0
_________________________________________________________________
```

1.  接下来，我们需要配置模型，以便使用优化器、损失函数和度量标准来判断准确性。在这里，使用的优化器是**标量梯度下降法（SGD）**，学习率为 0.01。使用的损失函数是代数**均方误差（MSE）**，用于衡量模型正确性的度量标准是`accuracy`，即概率分数：

```py
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01),metrics['accuracy'])  
```

1.  现在，我们准备好训练模型了。我们希望它每次迭代使用 128 个样本，通过网络进行学习，这由`batch_size`指示。我们希望每个样本在整个网络中至少迭代 200 次，这由`epochs`指示。同时，我们指定了用于训练和验证的数据集。`Verbose`控制控制台上的输出打印：

```py
model.fit(X_train,y_train,batch_size=128,epochs=200,
verbose=1,validation_data =(X_test,y_test))  
```

1.  在 60,000 个样本上进行训练，然后在 10,000 个样本上进行验证：

```py
Epoch 1/200
60000/60000 [==============================] - 1s - loss: 0.0915 - acc: 0.0895 - val_loss: 0.0911 - val_acc: 0.0955
Epoch 2/200
.
.
.
60000/60000 [==============================] - 1s - loss: 0.0908 - acc: 
0.8579 - val_loss: 0.0274 - val_acc: 0.8649
Epoch 199/200
60000/60000 [==============================] - 1s - loss: 0.0283 - acc: 0.8585 - val_loss: 0.0273 - val_acc: 0.8656
Epoch 200/200
60000/60000 [==============================] - 1s - loss: 0.0282 - acc: 0.8587 - val_loss: 0.0272 - val_acc: 0.8658
<keras.callbacks.History at 0x7f308e68be48>
```

1.  最后，我们可以评估模型以及模型在测试数据集上的预测效果：

```py
model.evaluate(X_test,y_test)
9472/10000 [===========================>..] - ETA: 0s
[0.027176343995332718, 0.86580000000000001]
```

这可以解释为错误率（MSE）为 0.027，准确率为 0.865，这意味着它在测试数据集上预测正确标签的次数占 86%。

# 构建一个前馈神经网络来识别手写数字，第二版

在上一部分中，我们构建了一个非常简单的神经网络，只有输入层和输出层。这个简单的神经网络给了我们 86%的准确率。让我们看看通过构建一个比之前版本更深的神经网络，是否能进一步提高这个准确率：

1.  我们将在一个新的笔记本中进行这项工作。加载数据集和数据预处理将与上一部分相同：

```py
import numpy as np np.random.seed(42)
import keras from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import SG
#loading and pre-processing data
(X_train,y_train), (X_test,y_test)= mnist.load_data()
X_train= X_train.reshape( 60000, 784). astype('float32')
X_test =X_test.reshape(10000,784).astype('float32')
X_train/=255
X_test/=255
```

1.  神经网络的设计与之前版本稍有不同。我们将在网络中加入一个包含 64 个神经元的隐藏层，以及输入层和输出层：

```py
model=Sequential()
model.add(Dense(64,activation='relu', input_shape=(784,)))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

```

1.  同时，我们将为输入层和隐藏层使用`relu`激活函数，而不是之前使用的`sigmoid`函数。

1.  我们可以如下检查模型设计和架构：

```py
model.summary()
_______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 64)                50240     
_______________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_______________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
```

1.  接下来，我们将配置模型，使用派生的`categorical_crossentropy`代价函数，而不是我们之前使用的 MSE。同时，将学习率从 0.01 提高到 0.1：

```py
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1), 
metrics =['accuracy']) 
```

1.  现在，我们将像之前的例子一样训练模型：

```py
model.fit(X_train,y_train,batch_size=128,epochs=200,verbose=1,validation_data =(X_test,y_test))
```

1.  在 60,000 个样本上训练，并在 10,000 个样本上验证：

```py
Epoch 1/200
60000/60000 [==============================] - 1s - loss: 0.4785 - acc: 0.8642 - val_loss: 0.2507 - val_acc: 0.9255
Epoch 2/200
60000/60000 [==============================] - 1s - loss: 0.2245 - acc: 0.9354 - val_loss: 0.1930 - val_acc: 0.9436
.
.
.
60000/60000 [==============================] - 1s - loss: 4.8932e-04 - acc: 1.0000 - val_loss: 0.1241 - val_acc: 0.9774
<keras.callbacks.History at 0x7f3096adadd8>
```

如你所见，和我们在第一版中构建的模型相比，准确率有所提高。

# 构建更深的神经网络

在本节中，我们将使用本章所学的概念，构建一个更深的神经网络来分类手写数字：

1.  我们将从一个新的笔记本开始，然后加载所需的依赖：

```py
import numpy as np np.random.seed(42)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
*# new!*
from
keras.layers.normalization *# new!*
import
BatchNormalization
*# new!*
from keras import regularizers
*# new!* 
from keras.optimizers
import SGD
```

1.  我们现在将加载并预处理数据：

```py
(X_train,y_train),(X_test,y_test)= mnist.load_data()
X_train= X_train.reshape(60000,784).
astype('float32')
X_test= X_test.reshape(10000,784).astype('float32')
X_train/=255
X_test/=255
n_classes=10
y_train=keras.utils.to_categorical(y_train,n_classes)
y_test =keras.utils.to_categorical(y_test,n_classes)

```

1.  现在，我们将设计一个更深的神经网络架构，并采取措施防止过拟合，以提供更好的泛化能力：

```py
model=Sequential()
model.add(Dense(64,activation='relu',input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()
_______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 64)                50240     
_______________________________________________________________
batch_normalization_1 (Batch (None, 64)                256       
_______________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_______________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_______________________________________________________________
batch_normalization_2 (Batch (None, 64)                256       
_______________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_______________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 55,562
Trainable params: 55,306
Non-trainable params: 256_______________________________________________________________
```

1.  这次，我们将使用`adam`优化器来配置模型：

```py
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```

1.  现在，我们将设定训练模型`200`个周期，批量大小为`128`：

```py
model.fit(X_train, y_train, batch_size= 128, epochs= 200, verbose= 1, validation_data= (X_test,y_test))
```

1.  在 60,000 个样本上训练，并在 10,000 个样本上验证：

```py
Epoch 1/200
60000/60000 [==============================] - 3s - loss: 0.8586 - acc: 0.7308 - val_loss: 0.2594 - val_acc: 0.9230
Epoch 2/200
60000/60000 [==============================] - 2s - loss: 0.4370 - acc: 0.8721 - val_loss: 0.2086 - val_acc: 0.9363
.
.
.
Epoch 200/200
60000/60000 [==============================] - 2s - loss: 0.1323 - acc: 0.9589 - val_loss: 0.1136 - val_acc: 0.9690
<keras.callbacks.History at 0x7f321175a748>
```

# 计算机视觉简介

计算机视觉可以定义为人工智能的一个子集，我们可以教计算机“看”。我们不能仅仅给机器添加一个相机让它“看”。为了让机器像人类或动物一样真正感知世界，它依赖于计算机视觉和图像识别技术。阅读条形码和人脸识别就是计算机视觉的应用实例。计算机视觉可以描述为人类大脑中处理眼睛感知信息的部分，别无其他。

图像识别是计算机视觉在人工智能领域中一个有趣的应用。从机器通过计算机视觉接收的输入由图像识别系统解读，依据其所见，输出会被分类。

换句话说，我们用眼睛捕捉周围的物体，这些物体/图像在大脑中被处理，使我们能够直观地感知周围的世界。计算机视觉赋予机器这种能力。计算机视觉负责从输入的视频或图像中自动提取、分析并理解所需的信息。

计算机视觉有多种应用，主要用于以下场景：

+   增强现实

+   机器人技术

+   生物特征识别

+   污染监测

+   农业

+   医学图像分析

+   法医

+   地球科学

+   自动驾驶汽车

+   图像恢复

+   流程控制

+   字符识别

+   遥感

+   手势分析

+   安全与监控

+   人脸识别

+   交通

+   零售

+   工业质量检测

# 计算机视觉的机器学习

使用适当的机器学习理论和工具非常重要，这对于我们开发涉及图像分类、物体检测等各种应用将非常有帮助。利用这些理论创建计算机视觉应用需要理解一些基本的机器学习概念。

# 计算机视觉领域的会议

一些值得关注的会议，了解最新的研究成果和应用，如下所示：

+   **计算机视觉与模式识别会议**（**CVPR**）每年举行，是最受欢迎的会议之一，涵盖从理论到应用的研究论文，跨越广泛的领域。

+   **国际计算机视觉大会**（**ICCV**）是每两年举行一次的另一大会议，吸引着一些最优秀的研究论文。

+   **计算机图形学特别兴趣小组**（**SIGGRAPH**）和交互技术，虽然更多集中在计算机图形学领域，但也有几篇应用计算机视觉技术的论文。

其他值得注意的会议包括**神经信息处理系统**（**NIPS**）、**国际机器学习大会**（**ICML**）、**亚洲计算机视觉大会**（**ACCV**）、**欧洲计算机视觉大会**（**ECCV**）等。

# 总结

本章中，我们构建了一个前馈神经网络，识别手写数字，并分为两个版本。然后，我们构建了一个神经网络，用于分类手写数字，最后简要介绍了计算机视觉。

在下一章中，我们将构建一个机器视觉移动应用程序，用于分类花卉品种并检索相关信息。

# 深入阅读

若要深入了解计算机视觉，请参考以下 Packt 出版的书籍：

+   *计算机视觉中的深度学习* 由 Rajalingappaa Shanmugamani 编写

+   *实用计算机视觉* 由 Abhinav Dadhich 编写
