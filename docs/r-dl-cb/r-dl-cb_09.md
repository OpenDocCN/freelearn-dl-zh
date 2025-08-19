# 第九章：深度学习在信号处理中的应用

本章将展示使用生成建模技术（如 RBM）创建新音乐音符的案例研究。在本章中，我们将涵盖以下主题：

+   介绍并预处理音乐 MIDI 文件

+   构建 RBM 模型

+   生成新的音乐音符

# 介绍并预处理音乐 MIDI 文件

在本节中，我们将读取一个 **音乐数字接口**（**MIDI**）文件库，并将其预处理为适用于 RBM 的格式。MIDI 是存储音乐音符的格式之一，可以转换为其他格式，如 `.wav`、`.mp3`、`.mp4` 等。MIDI 文件格式存储各种事件，如 Note-on、Note-off、Tempo、Time Signature、End of track 等。然而，我们将主要关注音符的类型——何时被**打开**，何时被**关闭**。

每首歌都被编码成一个二进制矩阵，其中行代表时间，列代表开启和关闭的音符。在每个时间点，一个音符被打开，随后同一个音符被关闭。假设，在 *n* 个音符中，第 *i* 个音符在时间 *j* 被打开并关闭，那么位置 *Mji = 1* 和 *Mj(n+i) = 1*，其余 *Mj = 0*。

所有的音符组合在一起形成一首歌。目前，在本章中，我们将利用 Python 代码将 MIDI 歌曲编码成二进制矩阵，这些矩阵可以在限制玻尔兹曼机（RBM）中使用。

# 准备就绪

让我们看看处理 MIDI 文件的前提条件：

1.  下载 MIDI 歌曲库：

[`github.com/dshieble/Music_RBM/tree/master/Pop_Music_Midi`](https://github.com/dshieble/Music_RBM/tree/master/Pop_Music_Midi)

1.  下载用于操作 MIDI 歌曲的 Python 代码：

[`github.com/dshieble/Music_RBM/blob/master/midi_manipulation.py`](https://github.com/dshieble/Music_RBM/blob/master/midi_manipulation.py)

1.  安装 `"reticulate"` 包，它提供了 R 与 Python 的接口：

```py
Install.packages("reticulate") 

```

1.  导入 Python 库：

```py
use_condaenv("python27") 
midi <- import_from_path("midi",path="C:/ProgramData/Anaconda2/Lib/site-packages") 
np <- import("numpy") 
msgpack <- import_from_path("msgpack",path="C:/ProgramData/Anaconda2/Lib/site-packages") 
psys <- import("sys") 
tqdm <- import_from_path("tqdm",path="C:/ProgramData/Anaconda2/Lib/site-packages") 
midi_manipulation_updated <- import_from_path("midi_manipulation_updated",path="C:/Music_RBM") 
glob <- import("glob") 

```

# 如何做到这一点...

现在我们已经设置了所有基本条件，让我们看看定义 MIDI 文件的函数：

1.  定义一个函数来读取 MIDI 文件并将其编码成二进制矩阵：

```py
get_input_songs <- function(path){ 
  files = glob$glob(paste0(path,"/*mid*")) 
  songs <- list() 
  count <- 1 
  for(f in files){ 
    songs[[count]] <- np$array(midi_manipulation_updated$midiToNoteStateMatrix(f)) 
    count <- count+1 
  } 
  return(songs) 
} 
path <- 'Pop_Music_Midi' 
input_songs <- get_input_songs(path) 

```

# 构建 RBM 模型

在本节中，我们将构建一个 RBM 模型，如 第五章中详细讨论的 *深度学习中的生成模型*。

# 准备就绪

让我们为模型设置系统：

1.  在钢琴中，最低音符是 24，最高音符是 102；因此，音符的范围是 78。这样，编码矩阵中的列数为 156（即 78 个 Note-on 和 78 个 Note-off）：

```py
lowest_note = 24L 
highest_note = 102L 
note_range = highest_note-lowest_note 

```

1.  我们将每次创建 15 步的音符，输入层有 2,340 个节点，隐藏层有 50 个节点：

```py
num_timesteps  = 15L 
num_input      = 2L*note_range*num_timesteps 
num_hidden       = 50L 

```

1.  学习率（alpha）为 0.1：

```py
alpha<-0.1 

```

# 如何做到这一点...

探讨构建 RBM 模型的步骤：

1.  定义 `placeholder` 变量：

```py
vb <- tf$placeholder(tf$float32, shape = shape(num_input)) 
hb <- tf$placeholder(tf$float32, shape = shape(num_hidden)) 
W <- tf$placeholder(tf$float32, shape = shape(num_input, num_hidden)) 

```

1.  定义一个前向传递：

```py
X = tf$placeholder(tf$float32, shape=shape(NULL, num_input)) 
prob_h0= tf$nn$sigmoid(tf$matmul(X, W) + hb)   
h0 = tf$nn$relu(tf$sign(prob_h0 - tf$random_uniform(tf$shape(prob_h0)))) 

```

1.  然后，定义一个反向传递：

```py
prob_v1 = tf$matmul(h0, tf$transpose(W)) + vb 
v1 = prob_v1 + tf$random_normal(tf$shape(prob_v1), mean=0.0, stddev=1.0, dtype=tf$float32) 
h1 = tf$nn$sigmoid(tf$matmul(v1, W) + hb)     

```

1.  相应地计算正向和负向梯度：

```py
w_pos_grad = tf$matmul(tf$transpose(X), h0) 
w_neg_grad = tf$matmul(tf$transpose(v1), h1) 
CD = (w_pos_grad - w_neg_grad) / tf$to_float(tf$shape(X)[0]) 
update_w = W + alpha * CD 
update_vb = vb + alpha * tf$reduce_mean(X - v1) 
update_hb = hb + alpha * tf$reduce_mean(h0 - h1) 

```

1.  定义目标函数：

```py
err = tf$reduce_mean(tf$square(X - v1)) 

```

1.  初始化当前和先前的变量：

```py
cur_w = tf$Variable(tf$zeros(shape = shape(num_input, num_hidden), dtype=tf$float32)) 
cur_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32)) 
cur_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 
prv_w = tf$Variable(tf$random_normal(shape=shape(num_input, num_hidden), stddev=0.01, dtype=tf$float32)) 
prv_vb = tf$Variable(tf$zeros(shape = shape(num_input), dtype=tf$float32)) 
prv_hb = tf$Variable(tf$zeros(shape = shape(num_hidden), dtype=tf$float32)) 

```

1.  启动 TensorFlow 会话：

```py
sess$run(tf$global_variables_initializer()) 
song = np$array(trainX) 
song = song[1:(np$floor(dim(song)[1]/num_timesteps)*num_timesteps),] 
song = np$reshape(song, newshape=shape(dim(song)[1]/num_timesteps, dim(song)[2]*num_timesteps)) 
output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=song, 
                                                                          W = prv_w$eval(), 
                                                                          vb = prv_vb$eval(), 
                                                                          hb = prv_hb$eval())) 
prv_w <- output[[1]]  
prv_vb <- output[[2]] 
prv_hb <-  output[[3]] 
sess$run(err, feed_dict=dict(X= song, W= prv_w, vb= prv_vb, hb= prv_hb)) 

```

1.  运行`200`次训练周期：

```py
epochs=200 
errors <- list() 
weights <- list() 
u=1 
for(ep in 1:epochs){ 
  for(i in seq(0,(dim(song)[1]-100),100)){ 
    batchX <- song[(i+1):(i+100),] 
    output <- sess$run(list(update_w, update_vb, update_hb), feed_dict = dict(X=batchX, 
                                                                              W = prv_w, 
                                                                              vb = prv_vb, 
                                                                              hb = prv_hb)) 
    prv_w <- output[[1]]  
    prv_vb <- output[[2]] 
    prv_hb <-  output[[3]] 
    if(i%%500 == 0){ 
      errors[[u]] <- sess$run(err, feed_dict=dict(X= song, W= prv_w, vb= prv_vb, hb= prv_hb)) 
      weights[[u]] <- output[[1]] 
      u <- u+1 
      cat(i , " : ") 
    } 
  } 
  cat("epoch :", ep, " : reconstruction error : ", errors[length(errors)][[1]],"\n") 
} 

```

# 生成新的音乐音符

在这个食谱中，我们将生成新的样本音乐音符。可以通过改变参数`num_timesteps`来生成新的音乐音符。然而，应该记住增加时间步数，因为在当前的 RBM 设置中，随着向量维度的增加，处理起来可能会变得计算效率低下。通过创建它们的堆叠（即**深度置信网络**），这些 RBM 可以在学习中变得更高效。读者可以利用第五章中*深度学习中的生成模型*的 DBN 代码来生成新的音乐音符。

# 如何操作...

1.  创建新的样本音乐：

```py
hh0 = tf$nn$sigmoid(tf$matmul(X, W) + hb) 
vv1 = tf$nn$sigmoid(tf$matmul(hh0, tf$transpose(W)) + vb) 
feed = sess$run(hh0, feed_dict=dict( X= sample_image, W= prv_w, hb= prv_hb)) 
rec = sess$run(vv1, feed_dict=dict( hh0= feed, W= prv_w, vb= prv_vb)) 
S = np$reshape(rec[1,],newshape=shape(num_timesteps,2*note_range)) 

```

1.  重新生成 MIDI 文件：

```py
midi_manipulation$noteStateMatrixToMidi(S, name=paste0("generated_chord_1")) 
generated_chord_1 

```
