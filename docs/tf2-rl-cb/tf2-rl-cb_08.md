# 第八章：加速深度强化学习代理开发的分布式训练

训练深度强化学习代理解决任务需要大量的时间，因为其样本复杂度很高。对于实际应用，快速迭代代理训练和测试周期对于深度强化学习应用的市场就绪度至关重要。本章中的配方提供了如何利用 TensorFlow 2.x 的能力，通过分布式训练深度神经网络模型来加速深度强化学习代理开发的说明。讨论了如何在单台机器以及跨机器集群上利用多个 CPU 和 GPU 的策略。本章还提供了使用**Ray**、**Tune** 和 **RLLib** 框架训练分布式**深度强化学习**（**Deep RL**）代理的多个配方。

具体来说，本章包含以下配方：

+   使用 TensorFlow 2.x 构建分布式深度学习模型 – 多 GPU 训练

+   扩展规模与范围 – 多机器、多 GPU 训练

+   大规模训练深度强化学习代理 – 多 GPU PPO 代理

+   为加速训练构建分布式深度强化学习的构建模块

+   使用 Ray、Tune 和 RLLib 进行大规模深度强化学习（Deep RL）代理训练

# 技术要求

本书中的代码在 Ubuntu 18.04 和 Ubuntu 20.04 上经过广泛测试，且如果安装了 Python 3.6+，应该也能在之后版本的 Ubuntu 上运行。只要安装了 Python 3.6+ 以及所需的 Python 包（每个配方开始前都会列出），代码也应该能够在 Windows 和 Mac OSX 上正常运行。建议创建并使用名为 `tf2rl-cookbook` 的 Python 虚拟环境来安装本书中所需的包并运行代码。推荐使用 Miniconda 或 Anaconda 来管理 Python 虚拟环境。

每个配方的完整代码可以在此获取：[`github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook`](https://github.com/PacktPublishing/Tensorflow-2-Reinforcement-Learning-Cookbook)。

# 使用 TensorFlow 2.x 进行分布式深度学习模型训练 – 多 GPU 训练

深度强化学习利用深度神经网络进行策略、价值函数或模型表示。对于高维观察/状态空间，例如图像或类似图像的观察，通常会使用**卷积神经网络**（**CNN**）架构。虽然 CNN 强大且能训练适用于视觉控制任务的深度强化学习策略，但在强化学习的设置下，训练深度 CNN 需要大量时间。本配方将帮助你了解如何利用 TensorFlow 2.x 的分布式训练 API，通过多 GPU 训练深度**残差网络**（**ResNets**）。本配方提供了可配置的构建模块，你可以用它们来构建深度强化学习组件，比如深度策略网络或价值网络。

让我们开始吧！

## 准备工作

要完成这个食谱，你需要首先激活 `tf2rl-cookbook` Python/conda 虚拟环境。确保更新环境，以匹配食谱代码库中最新的 conda 环境规范文件（`tfrl-cookbook.yml`）。拥有一台（本地或云端）配备一个或多个 GPU 的机器将对这个食谱有帮助。我们将使用 `tensorflow_datasets`，如果你使用 `tfrl-cookbook.yml` 来设置/更新了你的 conda 环境，它应该已经安装好了。

现在，让我们开始吧！

## 如何实现...

本食谱中的实现基于最新的官方 TensorFlow 文档/教程。接下来的步骤将帮助你深入掌握 TensorFlow 2.x 的分布式执行能力。我们将使用 ResNet 模型作为大模型的示例，它将从分布式训练中受益，利用多个 GPU 加速训练。我们将讨论构建 ResNet 的主要组件的代码片段。完整的实现请参考食谱代码库中的 `resnet.py` 文件。让我们开始：

1.  让我们直接进入构建残差神经网络的模板：

    ```py
    def resnet_block(
        input_tensor, size, kernel_size, filters, stage, \
         conv_strides=(2, 2), training=None
    ):
        x = conv_building_block(
            input_tensor,
            kernel_size,
            filters,
            stage=stage,
            strides=conv_strides,
            block="block_0",
            training=training,
        )
        for i in range(size - 1):
            x = identity_building_block(
                x,
                kernel_size,
                filters,
                stage=stage,
                block="block_%d" % (i + 1),
                training=training,
            )
        return x
    ```

1.  使用上面的 ResNet 块模板，我们可以快速构建包含多个 ResNet 块的 ResNet。在本书中，我们将实现一个包含一个 ResNet 块的 ResNet，你可以在代码库中找到实现了多个可配置数量和大小的 ResNet 块的 ResNet。让我们开始并在接下来的几个步骤中完成 ResNet 的实现，每次集中讨论一个重要的概念。首先，让我们定义函数签名：

    ```py
    def resnet(num_blocks, img_input=None, classes=10, training=None):
        """Builds the ResNet architecture using provided 
           config"""
    ```

1.  接下来，让我们处理输入图像数据表示中的通道顺序。最常见的维度顺序是：`batch_size` x `channels` x `width` x `height` 或 `batch_size` x `width` x `height` x `channels`。我们将处理这两种情况：

    ```py
        if backend.image_data_format() == "channels_first":
            x = layers.Lambda(
                lambda x: backend.permute_dimensions(x, \
                    (0, 3, 1, 2)), name="transpose"
            )(img_input)
            bn_axis = 1
        else:  # channel_last
            x = img_input
            bn_axis = 3
    ```

1.  现在，让我们对输入数据进行零填充，并应用初始层开始处理：

    ```py
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), \
                                         name="conv1_pad")(x)
        x = tf.keras.layers.Conv2D(16,(3, 3),strides=(1, 1),
                             padding="valid",
                             kernel_initializer="he_normal",
                             kernel_regularizer= \
                                tf.keras.regularizers.l2(
                                     L2_WEIGHT_DECAY), 
                             bias_regularizer= \
                                 tf.keras.regularizers.l2(
                                     L2_WEIGHT_DECAY), 
                                                            name="conv1",)(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                 name="bn_conv1", momentum=BATCH_NORM_DECAY,
                 epsilon=BATCH_NORM_EPSILON,)\
                      (x, training=training)
        x = tf.keras.layers.Activation("relu")(x)
    ```

1.  现在是时候使用我们创建的 `resnet_block` 函数来添加 ResNet 块了：

    ```py
        x = resnet_block(x, size=num_blocks, kernel_size=3,
            filters=[16, 16], stage=2, conv_strides=(1, 1),
            training=training,)
        x = resnet_block(x, size=num_blocks, kernel_size=3,
            filters=[32, 32], stage=3, conv_strides=(2, 2),
            training=training)
        x = resnet_block(x, size=num_blocks, kernel_size=3,
            filters=[64, 64], stage=4, conv_strides=(2, 2),
            training=training,)
    ```

1.  作为最终层，我们希望添加一个经过 `softmax` 激活的 `Dense`（全连接）层，节点数量等于任务所需的输出类别数：

    ```py
    x = tf.keras.layers.GlobalAveragePooling2D(
                                         name="avg_pool")(x)
        x = tf.keras.layers.Dense(classes,
            activation="softmax",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(
                 L2_WEIGHT_DECAY), 
            bias_regularizer=tf.keras.regularizers.l2(
                 L2_WEIGHT_DECAY), 
            name="fc10",)(x)
    ```

1.  在 ResNet 模型构建函数中的最后一步是将这些层封装为一个 TensorFlow 2.x Keras 模型，并返回输出：

    ```py
        inputs = img_input
        # Create model.
        model = tf.keras.models.Model(inputs, x, name=f"resnet{6 * num_blocks + 2}")
        return model
    ```

1.  使用我们刚才讨论的 ResNet 函数，通过简单地改变块的数量，构建具有不同层深度的深度残差网络变得非常容易。例如，以下是可能的：

    ```py
    resnet_mini = functools.partial(resnet, num_blocks=1)
    resnet20 = functools.partial(resnet, num_blocks=3)
    resnet32 = functools.partial(resnet, num_blocks=5)
    resnet44 = functools.partial(resnet, num_blocks=7)
    resnet56 = functools.partial(resnet, num_blocks=9)
    ```

1.  定义好我们的模型后，我们可以跳到多 GPU 训练代码。本食谱中的剩余步骤将引导你完成实现过程，帮助你利用机器上的所有可用 GPU 加速训练 ResNet。让我们从导入我们构建的 `ResNet` 模块以及 `tensorflow_datasets` 模块开始：

    ```py
    import os
    import sys
    import tensorflow as tf
    import tensorflow_datasets as tfds
    if "." not in sys.path:
        sys.path.insert(0, ".")
    import resnet
    ```

1.  我们现在可以选择使用哪个数据集来运行我们的分布式训练管道。在这个食谱中，我们将使用`dmlab`数据集，该数据集包含在 DeepMind Lab 环境中，RL 代理通常观察到的图像。根据你训练机器的 GPU、RAM 和 CPU 的计算能力，你可能想使用一个更小的数据集，比如`CIFAR10`：

    ```py
    dataset_name = "dmlab"  # "cifar10" or "cifar100"; See tensorflow.org/datasets/catalog for complete list
    # NOTE: dmlab is large in size; Download bandwidth and # GPU memory to be considered
    datasets, info = tfds.load(name="dmlab", with_info=True,
                               as_supervised=True)
    dataset_train, dataset_test = datasets["train"], \
                                  datasets["test"]
    input_shape = info.features["image"].shape
    num_classes = info.features["label"].num_classes
    ```

1.  下一步需要你全神贯注！我们将选择分布式执行策略。TensorFlow 2.x 将许多功能封装成了一个简单的 API 调用，如下面所示：

    ```py
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {
               strategy.num_replicas_in_sync}")
    ```

1.  在这一步中，我们将声明关键超参数，你可以根据机器的硬件（例如 RAM 和 GPU 内存）进行调整：

    ```py
    num_train_examples = info.splits["train"].num_examples
    num_test_examples = info.splits["test"].num_examples
    BUFFER_SIZE = 1000  # Increase as per available memory
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * \
                      strategy.num_replicas_in_sync
    ```

1.  在开始准备数据集之前，让我们实现一个预处理函数，该函数在将图像传递给神经网络之前执行操作。你可以添加你自己的自定义预处理操作。在这个食谱中，我们只需要首先将图像数据转换为`float32`，然后将图像像素值范围转换为[0, 1]，而不是典型的[0, 255]区间：

    ```py
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    ```

1.  我们已经准备好为训练和验证/测试创建数据集划分：

    ```py
    train_dataset = (
        dataset_train.map(preprocess).cache().\
            shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    )
    eval_dataset = dataset_test.map(preprocess).batch(
                                                 BATCH_SIZE)
    ```

1.  我们已经到了这个食谱的关键步骤！让我们在分布式策略的范围内实例化并编译我们的模型：

    ```py
    with strategy.scope():
        # model = create_model()
        model = create_model("resnet_mini")
        tf.keras.utils.plot_model(model, 
                                 to_file="./slim_resnet.png", 
                                 show_shapes=True)
        model.compile(
            loss=\
              tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
    ```

1.  让我们还创建一些回调，用于将日志记录到 TensorBoard，并在训练过程中检查点保存我们的模型参数：

    ```py
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, 
                                     "ckpt_{epoch}")
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./logs", write_images=True, \
            update_freq="batch"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, \
            save_weights_only=True
        ),
    ]
    ```

1.  有了这些，我们已经具备了使用分布式策略训练模型所需的一切。借助 Keras 用户友好的`fit()`API，它就像下面这样简单：

    ```py
    model.fit(train_dataset, epochs=12, callbacks=callbacks)
    ```

1.  当执行前面的行时，训练过程将开始。我们也可以使用以下几行手动保存模型：

    ```py
    path = "saved_model/"
    model.save(path, save_format="tf")
    ```

1.  一旦我们保存了检查点，加载权重并开始评估模型就变得很容易：

    ```py
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    eval_loss, eval_acc = model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))
    ```

1.  为了验证使用分布式策略训练的模型在有复制和没有复制的情况下都能正常工作，我们将在接下来的步骤中使用两种不同的方法加载并评估它。首先，让我们使用我们用来训练模型的（相同的）策略加载不带复制的模型：

    ```py
    unreplicated_model = tf.keras.models.load_model(path)
    unreplicated_model.compile(
        loss=tf.keras.losses.\
             SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)
    print("Eval loss: {}, Eval Accuracy: {}".format(eval_loss, eval_acc))
    ```

1.  接下来，让我们在分布式执行策略的范围内加载模型，这将创建副本并评估模型：

    ```py
    with strategy.scope():
        replicated_model = tf.keras.models.load_model(path)
        replicated_model.compile(
            loss=tf.keras.losses.\
             SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        eval_loss, eval_acc = \
            replicated_model.evaluate(eval_dataset)
        print("Eval loss: {}, \
              Eval Accuracy: {}".format(eval_loss, eval_acc))
    ```

    当你执行前面的两个代码块时，你会发现两种方法都会得到相同的评估准确度，这是一个好兆头，意味着我们可以在没有任何执行策略限制的情况下使用模型进行预测！

1.  这完成了我们的食谱。让我们回顾一下并看看食谱是如何工作的。

## 它是如何工作的...

神经网络架构中的残差块应用了卷积滤波器，后接多个恒等块。具体来说，卷积块应用一次，接着是(size - 1)个恒等块，其中 size 是一个整数，表示卷积-恒等块的数量。恒等块实现了跳跃连接或短路连接，使得输入可以绕过卷积操作直接通过。卷积块则包含卷积层，后接批量归一化激活，再接一个或多个卷积-批归一化-激活层。我们构建的`resnet`模块使用这些卷积和恒等构建块来构建一个完整的 ResNet，并且可以通过简单地更改块的数量来配置不同大小的网络。网络的大小计算公式为`6 * num_blocks + 2`。

一旦我们的 ResNet 模型准备好，我们使用`tensorflow_datasets`模块生成训练和验证数据集。TensorFlow 数据集模块提供了几个流行的数据集，如 CIFAR10、CIFAR100 和 DMLAB，这些数据集包含图像及其相关标签，用于分类任务。所有可用数据集的列表可以在此找到：[`tensorflow.org/datasets/catalog`](https://tensorflow.org/datasets/catalog)。

在这个食谱中，我们使用了`tf.distribute.MirroredStrategy`的镜像策略进行分布式执行，它允许在一台机器上使用多个副本进行同步分布式训练。即使是在多副本的分布式执行下，我们发现使用回调进行常规的日志记录和检查点保存依然如预期工作。我们还验证了加载保存的模型并运行推理进行评估在有或没有复制的情况下都能正常工作，这使得模型在训练过程中使用了分布式执行策略后，依然具有可移植性，不会因增加任何额外限制而受影响！

是时候进入下一个食谱了！

# 扩展与扩展 – 多机器，多 GPU 训练

为了在深度学习模型的分布式训练中实现最大规模，我们需要能够跨 GPU 和机器利用计算资源。这可以显著减少迭代或开发新模型和架构所需的时间，从而加速您正在解决的问题的进展。借助 Microsoft Azure、Amazon AWS 和 Google GCP 等云计算服务，按小时租用多台 GPU 配备的机器变得更加容易且普遍。这比搭建和维护自己的多 GPU 多机器节点更经济。这个配方将提供一个快速的演练，展示如何使用 TensorFlow 2.x 的多工作节点镜像分布式执行策略训练深度模型，基于官方文档，您可以根据自己的使用场景轻松定制。在本配方的多机器多 GPU 分布式训练示例中，我们将训练一个深度残差网络（ResNet 或 resnet）用于典型的图像分类任务。相同的网络架构也可以通过对输出层进行轻微修改，供 RL 智能体用于其策略或价值函数表示，正如我们将在本章后续的配方中看到的那样。

让我们开始吧！

## 准备工作

要完成此配方，您首先需要激活`tf2rl-cookbook` Python/conda 虚拟环境。确保更新环境，以匹配配方代码仓库中的最新 conda 环境规范文件（`tfrl-cookbook.yml`）。为了运行分布式训练管道，建议设置一个包含两个或更多安装了 GPU 的机器的集群，可以是在本地或云实例中，如 Azure、AWS 或 GCP。虽然我们将要实现的训练脚本可以利用集群中的多台机器，但并不绝对需要设置集群，尽管推荐这样做。

现在，让我们开始吧！

## 如何做到这一点...

由于此分布式训练设置涉及多台机器，我们需要一个机器之间的通信接口，并且要能够寻址每台机器。这通常通过现有的网络基础设施和 IP 地址来完成：

1.  我们首先设置一个描述集群配置参数的配置项，指定我们希望在哪里训练模型。以下代码块已被注释掉，您可以根据集群设置编辑并取消注释，或者如果仅想在单机配置上尝试，可以保持注释状态：

    ```py
    # Uncomment the following lines and fill worker details 
    # based on your cluster configuration
    # tf_config = {
    #    "cluster": {"worker": ["1.2.3.4:1111", 
                     "localhost:2222"]},
    #    "task": {"index": 0, "type": "worker"},
    # }
    # os.environ["TF_CONFIG"] = json.dumps(tf_config)
    ```

1.  为了利用多台机器的配置，我们将使用 TensorFlow 2.x 的 `MultiWorkerMirroredStrategy`：

    ```py
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    ```

1.  接下来，让我们声明训练的基本超参数。根据您的集群/计算机配置，随时调整批处理大小和 `NUM_GPUS` 值：

    ```py
    NUM_GPUS = 2
    BS_PER_GPU = 128
    NUM_EPOCHS = 60
    HEIGHT = 32
    WIDTH = 32
    NUM_CHANNELS = 3
    NUM_CLASSES = 10
    NUM_TRAIN_SAMPLES = 50000
    BASE_LEARNING_RATE = 0.1
    ```

1.  为了准备数据集，让我们实现两个快速的函数，用于规范化和增强输入图像：

    ```py
    def normalize(x, y):
        x = tf.image.per_image_standardization(x)
        return x, y
    def augmentation(x, y):
        x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, 
                                             WIDTH + 8)
        x = tf.image.random_crop(x, [HEIGHT, WIDTH, 
                                     NUM_CHANNELS])
        x = tf.image.random_flip_left_right(x)
        return x, y
    ```

1.  为了简化操作并加快收敛速度，我们将继续使用 CIFAR10 数据集，这是官方 TensorFlow 2.x 示例中用于训练的，但在您探索时可以自由选择其他数据集。一旦选择了数据集，我们就可以生成训练集和测试集：

    ```py
    (x, y), (x_test, y_test) = \
          keras.datasets.cifar10.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    test_dataset = \
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ```

1.  为了使训练结果可重现，我们将使用固定的随机种子来打乱数据集：

    ```py
    tf.random.set_seed(22)
    ```

1.  我们还没有准备好生成训练和验证/测试数据集。我们将使用前一步中声明的已知固定随机种子来打乱数据集，并对训练集应用数据增强：

    ```py
    train_dataset = (
        train_dataset.map(augmentation)
        .map(normalize)
        .shuffle(NUM_TRAIN_SAMPLES)
        .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
    )
    ```

1.  同样，我们将准备测试数据集，但我们不希望对测试图像进行随机裁剪！因此，我们将跳过数据增强，并使用标准化步骤进行预处理：

    ```py
    test_dataset = test_dataset.map(normalize).batch(
        BS_PER_GPU * NUM_GPUS, drop_remainder=True
    )
    ```

1.  在我们开始训练之前，我们需要创建一个优化器实例，并准备好输入层。根据任务的需要，您可以使用不同的优化器，例如 Adam：

    ```py
    opt = keras.optimizers.SGD(learning_rate=0.1, 
                               momentum=0.9)
    input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
    img_input = tf.keras.layers.Input(shape=input_shape)
    ```

1.  最后，我们准备在 `MultiMachineMirroredStrategy` 的作用域内构建模型实例：

    ```py
    with strategy.scope():
        model = resnet.resnet56(img_input=img_input, 
                                classes=NUM_CLASSES)
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
    ```

1.  为了训练模型，我们使用简单而强大的 Keras API：

    ```py
    model.fit(train_dataset, epochs=NUM_EPOCHS)
    ```

1.  一旦模型训练完成，我们可以轻松地保存、加载和评估：

    # 12.1 保存

    ```py
    model.save(path, save_format="tf")
    # 12.2 Load
    loaded_model = tf.keras.models.load_model(path)
    loaded_model.compile(
        loss=tf.keras.losses.\
            SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    # 12.3 Evaluate
    eval_loss, eval_acc = loaded_model.evaluate(eval_dataset)
    ```

这完成了我们的教程实现！让我们在下一部分总结我们实现了什么以及它是如何工作的。

## 它是如何工作的...

对于使用 TensorFlow 2.x 的任何分布式训练，需要在集群中每一台（虚拟）机器上设置 `TF_CONFIG` 环境变量。这些配置值将告知每台机器关于角色和每个节点执行任务所需的训练信息。您可以在这里阅读更多关于 TensorFlow 2.x 分布式训练中使用的**TF_CONFIG**配置的详细信息：[`cloud.google.com/ai-platform/training/docs/distributed-training-details`](https://cloud.google.com/ai-platform/training/docs/distributed-training-details)。

我们使用了 TensorFlow 2.x 的 `MultiWorkerMirroredStrategy`，这是一种与本章前面教程中使用的 Mirrored Strategy 类似的策略。这种策略适用于跨机器的同步训练，每台机器可能拥有一个或多个 GPU。所有训练模型所需的变量和计算都会在每个工作节点上进行复制，就像 Mirrored Strategy 一样，并且使用分布式收集例程（如 all-reduce）来汇总来自多个分布式节点的结果。训练、保存模型、加载模型和评估模型的其余工作流程与我们之前的教程相同。

准备好下一个教程了吗？让我们开始吧。

# 大规模训练深度强化学习代理 – 多 GPU PPO 代理

一般来说，RL 代理需要大量的样本和梯度步骤来进行训练，这取决于状态、动作和问题空间的复杂性。随着深度强化学习（Deep RL）的发展，计算复杂度也会急剧增加，因为代理使用的深度神经网络（无论是用于 Q 值函数表示，策略表示，还是两者都有）有更多的操作和参数需要分别执行和更新。为了加速训练过程，我们需要能够扩展我们的深度 RL 代理训练，以利用可用的计算资源，如 GPU。这个食谱将帮助你利用多个 GPU，以分布式的方式训练一个使用深度卷积神经网络策略的 PPO 代理，在使用**OpenAI 的 procgen**库的程序生成的 RL 环境中进行训练。

让我们开始吧！

## 准备工作

要完成这个食谱，首先你需要激活`tf2rl-cookbook` Python/conda 虚拟环境。确保更新环境，以匹配食谱代码库中的最新 conda 环境规格文件(`tfrl-cookbook.yml`)。虽然不是必需的，但建议使用具有两个或更多 GPU 的机器来执行此食谱。

现在，让我们开始吧！

## 如何做...

我们将实现一个完整的食谱，允许以分布式方式配置训练 PPO 代理，并使用深度卷积神经网络策略。让我们一步一步地开始实现：

1.  我们将从导入实现这一食谱所需的模块开始：

    ```py
    import argparse
    import os
    from datetime import datetime
    import gym
    import gym.wrappers
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPool2D,
    )
    ```

1.  我们将使用 OpenAI 的`procgen`环境。让我们也导入它：

    ```py
    import procgen  # Import & register procgen Gym envs
    ```

1.  为了使这个食谱更易于配置和运行，让我们添加对命令行参数的支持，并配置一些有用的配置标志：

    ```py
    parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch9-Distributed-RL-Agent")
    parser.add_argument("--env", default="procgen:procgen-coinrun-v0")
    parser.add_argument("--update-freq", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()
    ```

1.  让我们使用 TensorBoard 摘要写入器进行日志记录：

    ```py
    logdir = os.path.join(
        args.logdir, parser.prog, args.env, \
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(f"Saving training logs to:{logdir}")
    writer = tf.summary.create_file_writer(logdir)
    ```

1.  我们将首先在以下几个步骤中实现`Actor`类，从`__init__`方法开始。你会注意到我们需要在执行策略的上下文中实例化模型：

    ```py
    class Actor:
        def __init__(self, state_dim, action_dim, 
        execution_strategy):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.execution_strategy = execution_strategy
            with self.execution_strategy.scope():
                self.weight_initializer = \
                    tf.keras.initializers.he_normal()
                self.model = self.nn_model()
                self.model.summary()  # Print a summary of
                # the Actor model
                self.opt = \
                    tf.keras.optimizers.Nadam(args.actor_lr)
    ```

1.  对于 Actor 的策略网络模型，我们将实现一个包含多个`Conv2D`和`MaxPool2D`层的深度卷积神经网络。在这一步我们将开始实现，接下来的几步将完成它：

    ```py
        def nn_model(self):
            obs_input = Input(self.state_dim)
            conv1 = Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                input_shape=self.state_dim,
                data_format="channels_last",
                activation="relu",
            )(obs_input)
            pool1 = MaxPool2D(pool_size=(3, 3), \
                              strides=1)(conv1)
    ```

1.  我们将添加更多的 Conv2D - Pool2D 层，以根据任务的需求堆叠处理层。在这个食谱中，我们将为 procgen 环境训练策略，该环境在视觉上较为丰富，因此我们将堆叠更多的层：

    ```py
           conv2 = Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )(pool1)
            pool2 = MaxPool2D(pool_size=(3, 3), strides=1)\
                        (conv2)
            conv3 = Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )(pool2)
            pool3 = MaxPool2D(pool_size=(3, 3), strides=1)\
                        (conv3)
            conv4 = Conv2D(
                filters=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )(pool3)
            pool4 = MaxPool2D(pool_size=(3, 3), strides=1)\
                        (conv4)
    ```

1.  现在，我们可以使用一个扁平化层，并为策略网络准备输出头：

    ```py
           flat = Flatten()(pool4)
            dense1 = Dense(
                16, activation="relu", \
                   kernel_initializer=self.weight_initializer
            )(flat)
            dropout1 = Dropout(0.3)(dense1)
            dense2 = Dense(
                8, activation="relu", \
                   kernel_initializer=self.weight_initializer
            )(dropout1)
            dropout2 = Dropout(0.3)(dense2)
    ```

1.  作为构建策略网络神经模型的最后一步，我们将创建输出层并返回一个 Keras 模型：

    ```py
            output_discrete_action = Dense(
                self.action_dim,
                activation="softmax",
                kernel_initializer=self.weight_initializer,
            )(dropout2)
            return tf.keras.models.Model(
                inputs=obs_input, 
                outputs = output_discrete_action, 
                name="Actor")
    ```

1.  使用我们在前面步骤中定义的模型，我们可以开始处理状态/观察图像输入，并生成 logits（未归一化的概率）以及 Actor 将采取的动作。让我们实现一个方法来完成这个任务：

    ```py
        def get_action(self, state):
            # Convert [Image] to np.array(np.adarray)
            state_np = np.array([np.array(s) for s in state])
            if len(state_np.shape) == 3:
                # Convert (w, h, c) to (1, w, h, c)
                state_np = np.expand_dims(state_np, 0)
            logits = self.model.predict(state_np)  
            # shape: (batch_size, self.action_dim)
            action = np.random.choice(self.action_dim, 
                                      p=logits[0])
            # 1 Action per instance of env; Env expects:
            # (num_instances, actions)
            # action = (action,)
            return logits, action
    ```

1.  接下来，为了计算驱动学习的替代损失，我们将实现`compute_loss`方法：

    ```py
        def compute_loss(self, old_policy, new_policy, 
        actions, gaes):
            log_old_policy = tf.math.log(tf.reduce_sum(
                                       old_policy * actions))
            log_old_policy = tf.stop_gradient(log_old_policy)
            log_new_policy = tf.math.log(tf.reduce_sum(
                                       new_policy * actions))
            # Avoid INF in exp by setting 80 as the upper 
            # bound since,
            # tf.exp(x) for x>88 yeilds NaN (float32)
            ratio = tf.exp(
                tf.minimum(log_new_policy - \
                           tf.stop_gradient(log_old_policy),\
                           80)
            )
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - args.clip_ratio, 1.0 + \
                args.clip_ratio
            )
            gaes = tf.stop_gradient(gaes)
            surrogate = -tf.minimum(ratio * gaes, \
                                    clipped_ratio * gaes)
            return tf.reduce_mean(surrogate)
    ```

1.  接下来是一个核心方法，它将所有方法连接在一起以执行训练。请注意，这是每个副本的训练方法，我们将在后续的分布式训练方法中使用它：

    ```py
        def train(self, old_policy, states, actions, gaes):
            actions = tf.one_hot(actions, self.action_dim)  
            # One-hot encoding
            actions = tf.reshape(actions, [-1, \
                                 self.action_dim])  
            # Add batch dimension
            actions = tf.cast(actions, tf.float64)
            with tf.GradientTape() as tape:
                logits = self.model(states, training=True)
                loss = self.compute_loss(old_policy, logits, 
                                         actions, gaes)
            grads = tape.gradient(loss, 
                              self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, 
                             self.model.trainable_variables))
            return loss
    ```

1.  为了实现分布式训练方法，我们将使用`tf.function`装饰器来实现一个 TensorFlow 2.x 函数：

    ```py
        @tf.function
        def train_distributed(self, old_policy, states,
                              actions, gaes):
            per_replica_losses = self.execution_strategy.run(
                self.train, args=(old_policy, states, 
                                  actions, gaes))
            return self.execution_strategy.reduce(
                tf.distribute.ReduceOp.SUM, \
                    per_replica_losses, axis=None)
    ```

1.  这就完成了我们的`Actor`类实现，接下来我们将开始实现`Critic`类：

    ```py
    class Critic:
        def __init__(self, state_dim, execution_strategy):
            self.state_dim = state_dim
            self.execution_strategy = execution_strategy
            with self.execution_strategy.scope():
                self.weight_initializer = \
                    tf.keras.initializers.he_normal()
                self.model = self.nn_model()
                self.model.summary()  
                # Print a summary of the Critic model
                self.opt = \
                    tf.keras.optimizers.Nadam(args.critic_lr)
    ```

1.  你一定注意到，我们在执行策略的作用域内创建了 Critic 的价值函数模型实例，以支持分布式训练。接下来，我们将开始在以下几个步骤中实现 Critic 的神经网络模型：

    ```py
        def nn_model(self):
            obs_input = Input(self.state_dim)
            conv1 = Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                input_shape=self.state_dim,
                data_format="channels_last",
                activation="relu",
            )(obs_input)
            pool1 = MaxPool2D(pool_size=(3, 3), strides=2)\
                        (conv1)
    ```

1.  与我们的 Actor 模型类似，我们将有类似的 Conv2D-MaxPool2D 层的堆叠，后面跟着带有丢弃的扁平化层：

    ```py
            conv2 = Conv2D(filters=32, kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid", activation="relu",)(pool1)
            pool2 = MaxPool2D(pool_size=(3, 3), strides=2)\
                        (conv2)
            conv3 = Conv2D(filters=16,
                kernel_size=(3, 3), strides=(1, 1),
                padding="valid", activation="relu",)(pool2)
            pool3 = MaxPool2D(pool_size=(3, 3), strides=1)\
                        (conv3)
            conv4 = Conv2D(filters=8, kernel_size=(3, 3),
                strides=(1, 1), padding="valid",
                activation="relu",)(pool3)
            pool4 = MaxPool2D(pool_size=(3, 3), strides=1)\
                        (conv4)
            flat = Flatten()(pool4)
            dense1 = Dense(16, activation="relu", 
                           kernel_initializer =\
                               self.weight_initializer)\
                           (flat)
            dropout1 = Dropout(0.3)(dense1)
            dense2 = Dense(8, activation="relu", 
                           kernel_initializer = \
                               self.weight_initializer)\
                           (dropout1)
            dropout2 = Dropout(0.3)(dense2)
    ```

1.  我们将添加值输出头，并将模型作为 Keras 模型返回，以完成我们 Critic 的神经网络模型：

    ```py
            value = Dense(
                1, activation="linear", 
                kernel_initializer=self.weight_initializer)\
                (dropout2)
            return tf.keras.models.Model(inputs=obs_input, \
                                         outputs=value, \
                                         name="Critic")
    ```

1.  如你所记得，Critic 的损失是预测的时间差目标与实际时间差目标之间的均方误差。让我们实现一个计算损失的方法：

    ```py
        def compute_loss(self, v_pred, td_targets):
            mse = tf.keras.losses.MeanSquaredError(
                     reduction=tf.keras.losses.Reduction.SUM)
            return mse(td_targets, v_pred)
    ```

1.  与我们的 Actor 实现类似，我们将实现一个每个副本的`train`方法，然后在后续步骤中用于分布式训练：

    ```py
        def train(self, states, td_targets):
            with tf.GradientTape() as tape:
                v_pred = self.model(states, training=True)
                # assert v_pred.shape == td_targets.shape
                loss = self.compute_loss(v_pred, \
                               tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, \
                           self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, \
                           self.model.trainable_variables))
            return loss
    ```

1.  我们将通过实现`train_distributed`方法来完成`Critic`类的实现，该方法支持分布式训练：

    ```py
        @tf.function
        def train_distributed(self, states, td_targets):
            per_replica_losses = self.execution_strategy.run(
                self.train, args=(states, td_targets)
            )
            return self.execution_strategy.reduce(
                tf.distribute.ReduceOp.SUM, \
                per_replica_losses, axis=None
            )
    ```

1.  在实现了我们的`Actor`和`Critic`类后，我们可以开始我们的分布式`PPOAgent`实现。我们将分几个步骤实现`PPOAgent`类。让我们从`__init__`方法开始：

    ```py
    class PPOAgent:
        def __init__(self, env):
            """Distributed PPO Agent for image observations 
            and discrete action-space Gym envs
            Args:
                env (gym.Env): OpenAI Gym I/O compatible RL 
                environment with discrete action space
            """
            self.env = env
            self.state_dim = self.env.observation_space.shape
            self.action_dim = self.env.action_space.n
            # Create a Distributed execution strategy
            self.distributed_execution_strategy = \
                         tf.distribute.MirroredStrategy()
            print(f"Number of devices: {self.\
                    distributed_execution_strategy.\
                    num_replicas_in_sync}")
            # Create Actor & Critic networks under the 
            # distributed execution strategy scope
            with self.distributed_execution_strategy.scope():
                self.actor = Actor(self.state_dim, 
                                self.action_dim, 
                                tf.distribute.get_strategy())
                self.critic = Critic(self.state_dim, 
                                tf.distribute.get_strategy())
    ```

1.  接下来，我们将实现一个方法来计算**广义优势估计**（**GAE**）的目标：

    ```py
        def gae_target(self, rewards, v_values, next_v_value,
        done):
            n_step_targets = np.zeros_like(rewards)
            gae = np.zeros_like(rewards)
            gae_cumulative = 0
            forward_val = 0
            if not done:
                forward_val = next_v_value
            for k in reversed(range(0, len(rewards))):
                delta = rewards[k] + args.gamma * \
                  forward_val - v_values[k]
                gae_cumulative = args.gamma * \
                  args.gae_lambda * gae_cumulative + delta
                gae[k] = gae_cumulative
                forward_val = v_values[k]
                n_step_targets[k] = gae[k] + v_values[k]
            return gae, n_step_targets
    ```

1.  我们已经准备好开始我们的`train(…)`方法。我们将把这个方法的实现分为以下几个步骤。让我们设置作用域，开始外循环，并初始化变量：

    ```py
        def train(self, max_episodes=1000):
            with self.distributed_execution_strategy.scope():
                with writer.as_default():
                    for ep in range(max_episodes):
                        state_batch = []
                        action_batch = []
                        reward_batch = []
                        old_policy_batch = []
                        episode_reward, done = 0, False
                        state = self.env.reset()
                        prev_state = state
                        step_num = 0
    ```

1.  现在，我们可以开始为每个回合执行的循环，直到回合结束：

    ```py
                          while not done:
                            self.env.render()
                            logits, action = \
                                 self.actor.get_action(state)
                            next_state, reward, dones, _ = \
                                        self.env.step(action)
                            step_num += 1
                            print(f"ep#:{ep} step#:{step_num} 
                                    step_rew:{reward} \
                                    action:{action} \
                                    dones:{dones}",end="\r",)
                            done = np.all(dones)
                            if done:
                                next_state = prev_state
                            else:
                                prev_state = next_state
                            state_batch.append(state)
                            action_batch.append(action)
                            reward_batch.append(
                                            (reward + 8) / 8)
                            old_policy_batch.append(logits)  
    ```

1.  在每个回合内，如果我们达到了`update_freq`或者刚刚到达了结束状态，我们需要计算 GAE 和 TD 目标。让我们添加相应的代码：

    ```py
                             if len(state_batch) >= \
                             args.update_freq or done:
                                states = np.array(
                                    [state.squeeze() for \
                                     state in state_batch])
                                actions = \
                                    np.array(action_batch)
                                rewards = \
                                    np.array(reward_batch)
                                old_policies = np.array(
                                    [old_pi.squeeze() for \
                                 old_pi in old_policy_batch])
                                v_values = self.critic.\
                                        model.predict(states)
                                next_v_value = self.critic.\
                                   model.predict(
                                       np.expand_dims(
                                           next_state, 0))
                                gaes, td_targets = \
                                     self.gae_target(
                                         rewards, v_values,
                                         next_v_value, done)
                                actor_losses, critic_losses=\
                                                       [], []   
    ```

1.  在相同的执行上下文中，我们需要训练`Actor`和`Critic`：

    ```py
                                   for epoch in range(args.\
                                   epochs):
                                    actor_loss = self.actor.\
                                      train_distributed(
                                         old_policies,
                                         states, actions,
                                         gaes)
                                    actor_losses.\
                                      append(actor_loss)
                                    critic_loss = self.\
                                    critic.train_distributed(
                                       states, td_targets)
                                    critic_losses.\
                                       append(critic_loss)
                                # Plot mean actor & critic 
                                # losses on every update
                                tf.summary.scalar(
                                    "actor_loss", 
                                     np.mean(actor_losses), 
                                     step=ep)
                                tf.summary.scalar(
                                     "critic_loss", 
                                      np.mean(critic_losses), 
                                      step=ep) 
    ```

1.  最后，我们需要重置跟踪变量并更新我们的回合奖励值：

    ```py

                                state_batch = []
                                action_batch = []
                                reward_batch = []
                                old_policy_batch = []
                            episode_reward += reward
                            state = next_state 
    ```

1.  这样，我们的分布式`main`方法就完成了，来完成我们的配方：

    ```py
    if __name__ == "__main__":
        env_name = "procgen:procgen-coinrun-v0"
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.Monitor(env=env, 
                            directory="./videos", force=True)
        agent = PPOAgent(env)
        agent.train()
    ```

    配方完成了！希望你喜欢这个过程。你可以执行这个配方，并通过 TensorBoard 日志观看进度，以查看你在更多 GPU 的支持下获得的训练加速效果！

让我们回顾一下我们完成的工作以及配方如何工作的下一部分。

## 它是如何工作的...

我们实现了`Actor`和`Critic`类，其中 Actor 使用深度卷积神经网络表示策略，而 Critic 则使用类似的深度卷积神经网络表示其价值函数。这两个模型都在分布式执行策略的范围内实例化，使用了`self.execution_strategy.scope()`构造方法。

procgen 环境（如 coinrun、fruitbot、jumper、leaper、maze 等）是视觉上（相对）丰富的环境，因此需要较深的卷积层来处理视觉观察。因此，我们为 Actor 的策略网络使用了深度 CNN 模型。为了在多个 GPU 上使用多个副本进行分布式训练，我们首先实现了单副本训练方法（train），然后使用`Tensorflow.function`在副本间运行，并将结果进行汇总得到总损失。

最后，在分布式环境中训练我们的 PPO 智能体时，我们通过使用 Python 的`with`语句进行上下文管理，将所有训练操作都纳入分布式执行策略的范围，例如：`with self.distributed_execution_strategy.scope()`。

该是进行下一个配方的时候了！

# 用于加速训练的分布式深度强化学习基础模块

本章之前的配方讨论了如何使用 TensorFlow 2.x 的分布式执行 API 来扩展深度强化学习训练。理解了这些概念和实现风格后，训练使用更高级架构（如 Impala 和 R2D2）的深度强化学习智能体，需要像分布式参数服务器和分布式经验回放这样的 RL 基础模块。本章将演示如何为分布式 RL 训练实现这些基础模块。我们将使用 Ray 分布式计算框架来实现我们的基础模块。

让我们开始吧！

## 准备工作

要完成这个配方，首先需要激活`tf2rl-cookbook`的 Python/conda 虚拟环境。确保更新环境以匹配食谱代码仓库中的最新 conda 环境规范文件（`tfrl-cookbook.yml`）。为了测试我们在这个配方中构建的基础模块，我们将使用基于书中早期配方实现的 SAC 智能体的`self.sac_agent_base`模块。如果以下`import`语句能正常运行，那么你准备开始了：

```py
import pickle
import sys
import fire
import gym
import numpy as np
import ray
if "." not in sys.path:
    sys.path.insert(0, ".")
from sac_agent_base import SAC
```

现在，让我们开始吧！

## 如何实现...

我们将逐个实现这些基础模块，从分布式参数服务器开始：

1.  `ParameterServer`类是一个简单的存储类，用于在分布式训练环境中共享神经网络的参数或权重。我们将实现这个类作为 Ray 的远程 Actor：

    ```py
    @ray.remote
    class ParameterServer(object):
        def __init__(self, weights):
            values = [value.copy() for value in weights]
            self.weights = values
        def push(self, weights):
            values = [value.copy() for value in weights]
            self.weights = values
        def pull(self):
            return self.weights
        def get_weights(self):
            return self.weights
    ```

1.  我们还将添加一个方法将权重保存到磁盘：

    ```py
        # save weights to disk
        def save_weights(self, name):
            with open(name + "weights.pkl", "wb") as pkl:
                pickle.dump(self.weights, pkl)
            print(f"Weights saved to {name + 
                                      ‘weights.pkl’}.")
    ```

1.  作为下一个构建块，我们将实现`ReplayBuffer`，它可以被分布式代理集群使用。我们将在这一步开始实现，并在接下来的几步中继续：

    ```py
    @ray.remote
    class ReplayBuffer:
        """
        A simple FIFO experience replay buffer for RL Agents
        """
        def __init__(self, obs_shape, action_shape, size):
            self.cur_states = np.zeros([size, obs_shape[0]],
                                        dtype=np.float32)
            self.actions = np.zeros([size, action_shape[0]],
                                     dtype=np.float32)
            self.rewards = np.zeros(size, dtype=np.float32)
            self.next_states = np.zeros([size, obs_shape[0]],
                                         dtype=np.float32)
            self.dones = np.zeros(size, dtype=np.float32)
            self.idx, self.size, self.max_size = 0, 0, size
            self.rollout_steps = 0
    ```

1.  接下来，我们将实现一个方法，将新经验存储到重放缓冲区：

    ```py
        def store(self, obs, act, rew, next_obs, done):
            self.cur_states[self.idx] = np.squeeze(obs)
            self.actions[self.idx] = np.squeeze(act)
            self.rewards[self.idx] = np.squeeze(rew)
            self.next_states[self.idx] = np.squeeze(next_obs)
            self.dones[self.idx] = done
            self.idx = (self.idx + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            self.rollout_steps += 1
    ```

1.  为了从重放缓冲区采样一批经验数据，我们将实现一个方法，从重放缓冲区随机采样并返回一个包含采样经验数据的字典：

    ```py
        def sample_batch(self, batch_size=32):
            idxs = np.random.randint(0, self.size, 
                                     size=batch_size)
            return dict(
                cur_states=self.cur_states[idxs],
                actions=self.actions[idxs],
                rewards=self.rewards[idxs],
                next_states=self.next_states[idxs],
                dones=self.dones[idxs])
    ```

1.  这完成了我们的`ReplayBuffer`类的实现。现在我们将开始实现一个方法来进行`rollout`，该方法本质上是使用从分布式参数服务器对象中提取的参数和探索策略在 RL 环境中收集经验，并将收集到的经验存储到分布式重放缓冲区中。我们将在这一步开始实现，并在接下来的步骤中完成`rollout`方法的实现：

    ```py
    @ray.remote
    def rollout(ps, replay_buffer, config):
        """Collect experience using an exploration policy"""
        env = gym.make(config["env"])
        obs, reward, done, ep_ret, ep_len = env.reset(), 0, \
                                              False, 0, 0
        total_steps = config["steps_per_epoch"] * \
                       config["epochs"]
        agent = SAC(env.observation_space.shape, \
                    env.action_space)
        weights = ray.get(ps.pull.remote())
        target_weights = agent.actor.get_weights()
        for i in range(len(target_weights)):  
        # set tau% of target model to be new weights
            target_weights[i] = weights[i]
        agent.actor.set_weights(target_weights)
    ```

1.  在代理初始化并加载完毕，环境实例也准备好后，我们可以开始我们的经验收集循环：

    ```py
        for step in range(total_steps):
            if step > config["random_exploration_steps"]:
                # Use Agent’s policy for exploration after 
                `random_exploration_steps`
                a = agent.act(obs)
            else:  # Use a uniform random exploration policy
                a = env.action_space.sample()
            next_obs, reward, done, _ = env.step(a)
            print(f"Step#:{step} reward:{reward} \
                    done:{done}")
            ep_ret += reward
            ep_len += 1
    ```

1.  让我们处理`max_ep_len`配置的情况，以指示回合的最大长度，然后将收集的经验存储到分布式重放缓冲区中：

    ```py
            done = False if ep_len == config["max_ep_len"]\
                     else done
            # Store experience to replay buffer
            replay_buffer.store.remote(obs, a, reward, 
                                       next_obs, done)
    ```

1.  最后，在回合结束时，使用参数服务器同步行为策略的权重：

    ```py
            obs = next_obs
            if done or (ep_len == config["max_ep_len"]):
                """
                Perform parameter sync at the end of the 
                trajectory.
                """
                obs, reward, done, ep_ret, ep_len = \
                                 env.reset(), 0, False, 0, 0
                weights = ray.get(ps.pull.remote())
                agent.actor.set_weights(weights)
    ```

1.  这完成了`rollout`方法的实现，我们现在可以实现一个运行训练循环的`train`方法：

    ```py
    @ray.remote(num_gpus=1, max_calls=1)
    def train(ps, replay_buffer, config):
        agent = SAC(config["obs_shape"], \
                    config["action_space"])
        weights = ray.get(ps.pull.remote())
        agent.actor.set_weights(weights)
        train_step = 1
        while True:
            agent.train_with_distributed_replay_memory(
                ray.get(replay_buffer.sample_batch.remote())
            )
            if train_step % config["worker_update_freq"]== 0:
                weights = agent.actor.get_weights()
                ps.push.remote(weights)
            train_step += 1
    ```

1.  我们的配方中的最后一个模块是`main`函数，它将迄今为止构建的所有模块整合起来并执行。我们将在这一步开始实现，并在剩下的步骤中完成。让我们从`main`函数的参数列表开始，并将参数捕获到配置字典中：

    ```py
    def main(
        env="MountainCarContinuous-v0",
        epochs=1000,
        steps_per_epoch=5000,
        replay_size=100000,
        random_exploration_steps=1000,
        max_ep_len=1000,
        num_workers=4,
        num_learners=1,
        worker_update_freq=500,
    ):
        config = {
            "env": env,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "max_ep_len": max_ep_len,
            "replay_size": replay_size,
            "random_exploration_steps": \
                 random_exploration_steps,
            "num_workers": num_workers,
            "num_learners": num_learners,
            "worker_update_freq": worker_update_freq,
        }
    ```

1.  接下来，创建一个所需环境的实例，获取状态和观察空间，初始化 ray，并初始化一个随机策略-演员-评论家（Stochastic Actor-Critic）代理。注意，我们初始化的是一个单节点的 ray 集群，但你也可以使用节点集群（本地或云端）来初始化 ray：

    ```py
        env = gym.make(config["env"])
        config["obs_shape"] = env.observation_space.shape
        config["action_space"] = env.action_space
        ray.init()
        agent = SAC(config["obs_shape"], \
                    config["action_space"])
    ```

1.  在这一步，我们将初始化`ParameterServer`类的实例和`ReplayBuffer`类的实例：

    ```py
        params_server = \
            ParameterServer.remote(agent.actor.get_weights())
        replay_buffer = ReplayBuffer.remote(
            config["obs_shape"], \
            config["action_space"].shape, \
            config["replay_size"]
        )
    ```

1.  我们现在准备好运行已构建的模块了。我们将首先根据配置参数中指定的工作者数量，启动一系列`rollout`任务，这些任务将在分布式 ray 集群上启动`rollout`过程：

    ```py
        task_rollout = [
            rollout.remote(params_server, replay_buffer, 
                           config)
            for i in range(config["num_workers"])
        ]
    ```

    `rollout`任务将启动远程任务，这些任务将使用收集到的经验填充重放缓冲区。上述代码将立即返回，即使`rollout`任务需要时间来完成，因为它是异步函数调用。

1.  接下来，我们将启动一个可配置数量的学习者，在 ray 集群上运行分布式训练任务：

    ```py
        task_train = [
            train.remote(params_server, replay_buffer, 
                         config)
            for i in range(config["num_learners"])
        ]
    ```

    上述语句将启动远程训练过程，并立即返回，尽管`train`函数在学习者上需要一定时间来完成。

    ```py
    We will wait for the tasks to complete on the main thread before exiting:
        ray.wait(task_rollout)
        ray.wait(task_train)
    ```

1.  最后，让我们定义我们的入口点。我们将使用 Python Fire 库来暴露我们的`main`函数，并使其参数看起来像是一个支持命令行参数的可执行文件：

    ```py
    if __name__ == "__main__":
        fire.Fire(main)
    ```

    使用前述的入口点，脚本可以从命令行配置并启动。这里提供一个示例供你参考：

    ```py
    (tfrl-cookbook)praveen@dev-cluster:~/tfrl-cookbook$python 4_building_blocks_for_distributed_rl_using_ray.py main --env="MountaincarContinuous-v0" --num_workers=8 --num_learners=3
    ```

这就完成了我们的实现！让我们在下一节简要讨论它的工作原理。

## 它是如何工作的……

我们构建了一个分布式的`ParameterServer`、`ReplayBuffer`、rollout worker 和 learner 进程。这些构建模块对于训练分布式 RL 代理至关重要。我们使用 Ray 作为分布式计算框架。

在实现了构建模块和任务后，在`main`函数中，我们在 Ray 集群上启动了两个异步的分布式任务。`task_rollout`启动了（可配置数量的）rollout worker，而`task_train`启动了（可配置数量的）learner。两个任务都以分布式方式异步运行在 Ray 集群上。rollout workers 从参数服务器拉取最新的权重，并将经验收集并存储到重放内存缓冲区中，同时，learners 使用从重放内存中采样的经验批次进行训练，并将更新（且可能改进的）参数集推送到参数服务器。

是时候进入本章的下一个，也是最后一个教程了！

# 使用 Ray、Tune 和 RLLib 进行大规模深度强化学习（Deep RL）代理训练

在之前的教程中，我们初步了解了如何从头实现分布式 RL 代理训练流程。由于大多数用作构建模块的组件已成为构建深度强化学习训练基础设施的标准方式，我们可以利用一个现有的库，该库维护了这些构建模块的高质量实现。幸运的是，选择 Ray 作为分布式计算框架使我们处于一个有利位置。Tune 和 RLLib 是基于 Ray 构建的两个库，并与 Ray 一起提供，提供高度可扩展的超参数调优（Tune）和 RL 训练（RLLib）。本教程将提供一套精选步骤，帮助你熟悉 Ray、Tune 和 RLLib，从而能够利用它们来扩展你的深度 RL 训练流程。除了文中讨论的教程外，本章的代码仓库中还有一系列额外的教程供你参考。

让我们开始吧！

## 准备工作

要完成这个教程，你首先需要激活`tf2rl-cookbook`的 Python/conda 虚拟环境。确保更新环境以匹配最新的 conda 环境规范文件（`tfrl-cookbook.yml`），该文件位于教程代码仓库中。当你使用提供的 conda YAML 规范来设置环境时，Ray、Tune 和 RLLib 将会被安装在你的`tf2rl-cookbook` conda 环境中。如果你希望在其他环境中安装 Tune 和 RLLib，最简单的方法是使用以下命令安装：

```py
 pip install ray[tune,rllib]
```

现在，开始吧！

## 如何实现……

我们将从快速和基本的命令与食谱开始，使用 Tune 和 RLLib 在 ray 集群上启动训练，并逐步自定义训练流水线，以为你提供有用的食谱：

1.  在 OpenAI Gym 环境中启动 RL 代理的典型训练和指定算法名称和环境名称一样简单。例如，要在 CartPole-v4 Gym 环境中训练 PPO 代理，你只需要执行以下命令：

    ```py
    --eager flag is also specified, which forces RLLib to use eager execution (the default mode of execution in TensorFlow 2.x).
    ```

1.  让我们尝试在`coinrun`的`procgen`环境中训练一个 PPO 代理，就像我们之前的一个食谱一样：

    ```py
    (tfrl-cookbook) praveen@dev-cluster:~/tfrl-cookbook$rllib train --run PPO --env "procgen:procgen-coinrun-v0" --eager
    ```

    你会注意到，前面的命令会失败，并给出以下（简化的）错误：

    ```py
        ValueError: No default configuration for obs shape [64, 64, 3], you must specify `conv_filters` manually as a model option. Default configurations are only available for inputs of shape [42, 42, K] and [84, 84, K]. You may alternatively want to use a custom model or preprocessor.
    ```

    这是因为，如错误所示，RLLib 默认支持形状为（42，42，k）或（84，84，k）的观察值。其他形状的观察值将需要自定义模型或预处理器。在接下来的几个步骤中，我们将展示如何实现一个自定义神经网络模型，使用 TensorFlow 2.x Keras API 实现，并且可以与 ray RLLib 一起使用。

1.  我们将在这一步开始实现自定义模型（`custom_model.py`），并在接下来的几步中完成它。在这一步，让我们导入必要的模块，并实现一个辅助方法，以返回具有特定滤波深度的 Conv2D 层：

    ```py
    from ray.rllib.models.tf.tf_modelv2 import TFModelV2
    import tensorflow as tf
    def conv_layer(depth, name):
        return tf.keras.layers.Conv2D(
            filters=depth, kernel_size=3, strides=1, \
            padding="same", name=name
        )
    ```

1.  接下来，让我们实现一个辅助方法来构建并返回一个简单的残差块：

    ```py
    def residual_block(x, depth, prefix):
        inputs = x
        assert inputs.get_shape()[-1].value == depth
        x = tf.keras.layers.ReLU()(x)
        x = conv_layer(depth, name=prefix + "_conv0")(x)
        x = tf.keras.layers.ReLU()(x)
        x = conv_layer(depth, name=prefix + "_conv1")(x)
        return x + inputs
    ```

1.  让我们实现另一个方便的函数来构建多个残差块序列：

    ```py
    def conv_sequence(x, depth, prefix):
        x = conv_layer(depth, prefix + "_conv")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, \
                                      strides=2,\
                                      padding="same")(x)
        x = residual_block(x, depth, prefix=prefix + \
                           "_block0")
        x = residual_block(x, depth, prefix=prefix + \
                           "_block1")
        return x
    ```

1.  现在，我们可以开始实现`CustomModel`类，作为 RLLib 提供的 TFModelV2 基类的子类，以便轻松地与 RLLib 集成：

    ```py
    class CustomModel(TFModelV2):
        """Deep residual network that produces logits for 
           policy and value for value-function;
        Based on architecture used in IMPALA paper:https://
           arxiv.org/abs/1802.01561"""
        def __init__(self, obs_space, action_space, 
        num_outputs, model_config, name):
            super().__init__(obs_space, action_space, \
                             num_outputs, model_config, name)
            depths = [16, 32, 32]
            inputs = tf.keras.layers.Input(
                            shape=obs_space.shape,
                            name="observations")
            scaled_inputs = tf.cast(inputs, 
                                    tf.float32) / 255.0
            x = scaled_inputs
            for i, depth in enumerate(depths):
                x = conv_sequence(x, depth, prefix=f"seq{i}")
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Dense(units=256,
                                      activation="relu", 
                                      name="hidden")(x)
            logits = tf.keras.layers.Dense(units=num_outputs,
                                           name="pi")(x)
            value = tf.keras.layers.Dense(units=1, 
                                          name="vf")(x)
            self.base_model = tf.keras.Model(inputs, 
                                            [logits, value])
            self.register_variables(
                                   self.base_model.variables)
    ```

1.  在`__init__`方法之后，我们需要实现`forward`方法，因为它没有被基类（`TFModelV2`）实现，但却是必需的：

    ```py
        def forward(self, input_dict, state, seq_lens):
            # explicit cast to float32 needed in eager
            obs = tf.cast(input_dict["obs"], tf.float32)
            logits, self._value = self.base_model(obs)
            return logits, state
    ```

1.  我们还将实现一个单行方法来重新调整值函数的输出：

    ```py
        def value_function(self):
            return tf.reshape(self._value, [-1])
    ```

    这样，我们的`CustomModel`实现就完成了，并且可以开始使用了！

1.  我们将实现一个使用 ray、Tune 和 RLLib 的 Python API 的解决方案（`5.1_training_using_tune_run.py`），这样你就可以在使用它们的命令行工具的同时，也能利用该模型。让我们将实现分为两步。在这一步，我们将导入必要的模块并初始化 ray：

    ```py
    import ray
    import sys
    from ray import tune
    from ray.rllib.models import ModelCatalog
    if not "." in sys.path:
        sys.path.insert(0, ".")
    from custom_model import CustomModel
    ray.init()  # Can also initialize a cluster with multiple 
    #nodes here using the cluster head node’s IP
    ```

1.  在这一步，我们将把我们的自定义模型注册到 RLLib 的`ModelCatlog`中，然后使用它来训练一个带有自定义参数集的 PPO 代理，其中包括强制 RLLib 使用 TensorFlow 2 的`framework`参数。我们还将在脚本结束时关闭 ray：

    ```py
    # Register custom-model in ModelCatalog
    ModelCatalog.register_custom_model("CustomCNN", 
                                        CustomModel)
    experiment_analysis = tune.run(
        "PPO",
        config={
            "env": "procgen:procgen-coinrun-v0",
            "num_gpus": 0,
            "num_workers": 2,
            "model": {"custom_model": "CustomCNN"},
            "framework": "tf2",
            "log_level": "INFO",
        },
        local_dir="ray_results",  # store experiment results
        #  in this dir
    )
    ray.shutdown()
    ```

1.  我们将查看另一个快速食谱（`5_2_custom_training_using_tune.py`）来定制训练循环。我们将把实现分为以下几个步骤，以保持简洁。在这一步，我们将导入必要的库并初始化 ray：

    ```py
    import sys
    import ray
    import ray.rllib.agents.impala as impala
    from ray.tune.logger import pretty_print
    from ray.rllib.models import ModelCatalog
    if not "." in sys.path:
        sys.path.insert(0, ".")
    from custom_model import CustomModel
    ray.init()  # You can also initialize a multi-node ray 
    # cluster here
    ```

1.  现在，让我们将自定义模型注册到 RLLib 的`ModelCatalog`中，并配置**IMPALA 代理**。我们当然可以使用任何其他的 RLLib 支持的代理，如 PPO 或 SAC：

    ```py
    # Register custom-model in ModelCatalog
    ModelCatalog.register_custom_model("CustomCNN", 
                                        CustomModel)
    config = impala.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["model"]["custom_model"] = "CustomCNN"
    config["log_level"] = "INFO"
    config["framework"] = "tf2"
    trainer = impala.ImpalaTrainer(config=config,
                            env="procgen:procgen-coinrun-v0")
    ```

1.  现在，我们可以实现自定义训练循环，并根据需要在循环中加入任何步骤。我们将通过每隔 n(100) 代（epochs）执行一次训练步骤并保存代理的模型来保持示例循环的简单性：

    ```py
    for step in range(1000):
        # Custom training loop
        result = trainer.train()
        print(pretty_print(result))
        if step % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint
    ```

1.  请注意，我们可以继续使用保存的检查点和 Ray tune 的简化 run API 来训练代理，如此处示例所示：

    ```py
    # Restore agent from a checkpoint and start a new 
    # training run with a different config
    config["lr"] =  ray.tune.grid_search([0.01, 0.001])"]
    ray.tune.run(trainer, config=config, restore=checkpoint)
    ```

1.  最后，让我们关闭 Ray 以释放系统资源：

    ```py
    ray.shutdown()
    ```

这就完成了本次配方！在下一节中，让我们回顾一下我们在本节中讨论的内容。

## 它是如何工作的...

我们发现了 Ray RLLib 简单但有限的命令行界面中的一个常见限制。我们还讨论了解决方案，以克服第 2 步中的失败情况，在此过程中需要自定义模型来使用 RLLib 的 PPO 代理训练，并在第 9 步和第 10 步中实现了该方案。

尽管第 9 步和第 10 步中讨论的解决方案看起来很优雅，但它可能无法提供您所需的所有自定义选项或您熟悉的选项。例如，它将基本的 RL 循环抽象了出来，这个循环会遍历环境。我们从第 11 步开始实现了另一种快速方案，允许自定义训练循环。在第 12 步中，我们看到如何注册自定义模型并将其与 IMPALA 代理一起使用——IMPALA 代理是基于 IMPortance 加权 Actor-Learner 架构的可扩展分布式深度强化学习代理。IMPALA 代理的演员通过传递状态、动作和奖励的序列与集中式学习器通信，在学习器中进行批量梯度更新，而与之对比的是基于（异步）Actor-Critic 的代理，其中梯度被传递到一个集中式参数服务器。

如需更多关于 Tune 的信息，可以参考 [`docs.ray.io/en/master/tune/user-guide.html`](https://docs.ray.io/en/master/tune/user-guide.html) 上的 Tune 用户指南和配置文档。

如需更多关于 RLLib 训练 API 和配置文档的信息，可以参考 [`docs.ray.io/en/master/rllib-training.html`](https://docs.ray.io/en/master/rllib-training.html)。

本章和配方已完成！希望您通过所获得的新技能和知识，能够加速您的深度 RL 代理训练。下章见！
