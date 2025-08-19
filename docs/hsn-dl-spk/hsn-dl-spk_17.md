# 附录 B：Spark 图像数据准备

卷积神经网络（CNN）是本书的主要话题之一。它们被广泛应用于图像分类和分析的实际应用中。本附录解释了如何创建一个 `RDD<DataSet>` 来训练 CNN 模型进行图像分类。

# 图像预处理

本节描述的图像预处理方法将文件分批处理，依赖于 ND4J 的 `FileBatch` 类（[`static.javadoc.io/org.nd4j/nd4j-common/1.0.0-beta3/org/nd4j/api/loader/FileBatch.html`](https://static.javadoc.io/org.nd4j/nd4j-common/1.0.0-beta3/org/nd4j/api/loader/FileBatch.html)），该类从 ND4J 1.0.0-beta3 版本开始提供。该类可以将多个文件的原始内容存储在字节数组中（每个文件一个数组），包括它们的原始路径。`FileBatch` 对象可以以 ZIP 格式存储到磁盘中。这可以减少所需的磁盘读取次数（因为文件更少）以及从远程存储读取时的网络传输（因为 ZIP 压缩）。通常，用于训练 CNN 的原始图像文件会采用一种高效的压缩格式（如 JPEG 或 PNG），这种格式在空间和网络上都比较高效。但在集群中，需要最小化由于远程存储延迟问题导致的磁盘读取。与 `minibatchSize` 的远程文件读取相比，切换到单次文件读取/传输会更快。

将图像预处理成批次会带来以下限制：在 DL4J 中，类标签需要手动提供。图像应存储在以其对应标签命名的目录中。我们来看一个示例——假设我们有三个类，即汽车、卡车和摩托车，图像目录结构应该如下所示：

```py
imageDir/car/image000.png
imageDir/car/image001.png
...
imageDir/truck/image000.png
imageDir/truck/image001.png
...
imageDir/motorbike/image000.png
imageDir/motorbike/image001.png
...
```

图像文件的名称并不重要。重要的是根目录下的子目录名称必须与类的名称一致。

# 策略

在 Spark 集群上开始训练之前，有两种策略可以用来预处理图像。第一种策略是使用 `dl4j-spark` 中的 `SparkDataUtils` 类在本地预处理图像。例如：

```py
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.spark.util.SparkDataUtils
...
val sourcePath = "/home/guglielmo/trainingImages"
val sourceDir = new File(sourcePath)
val destinationPath = "/home/guglielmo/preprocessedImages"
val destDir = new File(destinationPath)
val batchSize = 32
SparkDataUtils.createFileBatchesLocal(sourceDir, NativeImageLoader.ALLOWED_FORMATS, true, destDir, batchSize)
```

在这个示例中，`sourceDir` 是本地图像的根目录，`destDir` 是保存预处理后图像的本地目录，`batchSize` 是将图像放入单个 `FileBatch` 对象中的数量。`createFileBatchesLocal` 方法负责导入。一旦所有图像都被预处理，目标目录 `dir` 的内容可以被复制或移动到集群中用于训练。

第二种策略是使用 Spark 对图像进行预处理。在原始图像存储在分布式文件系统（如 HDFS）或分布式对象存储（如 S3）的情况下，仍然使用 `SparkDataUtils` 类，但必须调用一个不同的方法 `createFileBatchesLocal`，该方法需要一个 SparkContext 作为参数。以下是一个示例：

```py
val sourceDirectory = "hdfs:///guglielmo/trainingImages"; 
val destinationDirectory = "hdfs:///guglielmo/preprocessedImages";    
val batchSize = 32

val conf = new SparkConf
...
val sparkContext = new JavaSparkContext(conf)

val filePaths = SparkUtils.listPaths(sparkContext, sourceDirectory, true, NativeImageLoader.ALLOWED_FORMATS)
SparkDataUtils.createFileBatchesSpark(filePaths, destinationDirectory, batchSize, sparkContext)
```

在这种情况下，原始图像存储在 HDFS 中（通过`sourceDirectory`指定位置），预处理后的图像也保存在 HDFS 中（位置通过`destinationDirectory`指定）。在开始预处理之前，需要使用 dl4j-spark 的`SparkUtils`类创建源图像路径的`JavaRDD<String>`（`filePaths`）。`SparkDataUtils.createFileBatchesSpark`方法接受`filePaths`、目标 HDFS 路径（`destinationDirectory`）、放入单个`FileBatch`对象的图像数量（`batchSize`）以及 SparkContext（`sparkContext`）作为输入。只有所有图像都经过 Spark 预处理后，训练才能开始。

# 训练

无论选择了哪种预处理策略（本地或 Spark），以下是使用 Spark 进行训练的步骤。

首先，创建 SparkContext，设置`TrainingMaster`*，*并使用以下实例构建神经网络模型：

```py
val conf = new SparkConf
...
val sparkContext = new JavaSparkContext(conf)
val trainingMaster = ...
val net:ComputationGraph = ...
val sparkNet = new SparkComputationGraph(sparkContext, net, trainingMaster)
sparkNet.setListeners(new PerformanceListener(10, true))
```

之后，需要创建数据加载器，如以下示例所示：

```py
val imageHeightWidth = 64      
val imageChannels = 3          
val labelMaker = new ParentPathLabelGenerator
val rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker)
rr.setLabels(new TinyImageNetDataSetIterator(1).getLabels())
val numClasses = TinyImageNetFetcher.NUM_LABELS
val loader = new RecordReaderFileBatchLoader(rr, minibatch, 1, numClasses)
loader.setPreProcessor(new ImagePreProcessingScaler)
```

输入图像具有分辨率为 64 x 64 像素（`imageHeightWidth`）和三个通道（RGB，`imageChannels`）。加载器通过`ImagePreProcessingScaler`类将 0-255 值像素缩放到 0-1 的范围内（[`deeplearning4j.org/api/latest/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.html`](https://deeplearning4j.org/api/latest/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.html)）。

训练可以从以下示例开始：

```py
val trainPath = "hdfs:///guglielmo/preprocessedImages"
val pathsTrain = SparkUtils.listPaths(sc, trainPath)
val numEpochs = 10
for (i <- 0 until numEpochs) {
    println("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs)
    sparkNet.fitPaths(pathsTrain, loader)
} 
```
