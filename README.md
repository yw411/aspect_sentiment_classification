# aspect_sentiment_classification

## 定义
aspect-level层级的分类，分为两种，一种是term-target级别，即term存在于文本中；另一种是aspect-category级别，即aspect不存在于文本当中。

semeval评测会议有相关数据集和任务，大多数论文都使用这些数据集：2014、2015、2016

## 发展脉络

#### lstm、attention、pos 这是最经典的方法。

1、Attention-based LSTM for Aspect-level Sentiment Classification

最基本的框架，lstm+attention

2、Aspect Level Sentiment Classification with Deep Memory Network

在lstm+attention，引入多跳机制，同时使用位置信息（自我感觉位置信息很重要）

3、Interactive Attention Networks for Aspect-Level Sentiment Classification  aaai2017

使用交互机制，context-target-attention和target-context-attention。一方取平均值，另一方每个词与这个平均值进行交互。

4、Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks

同时使用交互机制，但是比3更细一些，每个aspect词都要与context进行交互，得到一个矩阵。

5、Learning to Attend via Word-Aspect Associative Fusion for Aspect-based Sentiment Analysis

使用了fusion attention机制，待理解。

#### cnn 用这个网络的文章很少，目前看到了2篇，acl2018上有一篇。其实cnn的应用主要是提取短语、n-gram信息。

6、Aspect Based Sentiment Analysis with Gated Convolutional Networks

attention的方法，但是又略有不同。向量中每一个维度都有一个权重。

7、基于多注意力卷积神经网络的特定目标情感分析

使用位置、词性、aspect这3种注意力，得到文本的更新表示，然后这3个表示，通过同构或异构的方式送入卷积网络，max-pooling。

#### left-right 这个思路也有几篇文章。

8、Effective LSTMs for Target-Dependent Sentiment Classification

左右两边分布lstm，然后concat

9、Gated Neural Networks for Targeted Sentiment Analysis

左右两边分别bi-lstm后，然后使用gate进行选择，gate进行选择的时候会用到target信息，

10、Attention Modeling for Targeted Sentiment

左边，右边，全体分别与aspect进行attention，得到3个表示，然后使用一个gate得到这3个的权重

11、 Left-Center-Right Separated Neural Network for Aspect-based Sentiment Analysis with Rotatory Attention   还没有发表

对于target的表示，分别用left和right与其交互，所以最好得到4个表示，然后concat

#### memory network work、多跳机制

2、Aspect Level Sentiment Classification with Deep Memory Network

12、Recurrent Attention Network on Memory for Aspect Sentiment Analysis

在2的基础上改进，attention中target的更新，不是简单的相加，而是使用了一个gru结构来进行更新。

#### 利用情感词典、词性等信息

7、基于多注意力卷积神经网络的特定目标情感分析

13、Feature-based Compositing Memory Networks for Aspect-based Sentiment Classification in Social Internet of Things

每个词有个position向量，sentiment向量，词性POS向量，基本方法是lstm+att+多跳机制。

14、Feature-enhanced attention network for target-dependent sentiment classification

词向量、sentiment向量、词性，位置position向量，lstm后得到文本向量，情感词向量，target向量，每个以其他两个为依据计算attention，所以得到3个加权之后的向量表示，然后连接。

15、Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM

16、Implicit Syntactic Features for Targeted Sentiment Analysis

隐式表达语法信息

17、Recurrent Entity Networks with Delayed Memory Update for Targeted Aspect-based Sentiment Analysis

entity networks的应用

#### 层次化方法   针对文档级别，分层次

## 结果比较

## 我的一些思路

1、针对term-target级别，有一个想法：来自Gated-Attention Readers for Text Comprehension这篇文章，target不更新，而是更新每个词多次，有2篇文章就是这个思路：

2、针对category-aspect级别，暂时有一个思路，还在写代码。
