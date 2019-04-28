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

5（1）：emnlp2018的新文章    Multi-grained Attention Network for Aspect-Level Sentiment Classification 

在3的基础上，句子和aspect在单词粒度上进行交互得到一个矩阵，然后再进行加权。其中作者增加的loss是一个可取的想法

5（2）：Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks

文章的方法来自机器阅读的一篇文章，attention-over-attention neural networks for reading comprehension，也是在单词粒度上得到一个矩阵，完全是一个匹配问题。

#### cnn 用这个网络的文章很少，目前看到了2篇，acl2018上有一篇。其实cnn的应用主要是提取短语、n-gram信息。

6、Aspect Based Sentiment Analysis with Gated Convolutional Networks

attention的方法，但是又略有不同。向量中每一个维度都有一个权重。

7、基于多注意力卷积神经网络的特定目标情感分析

使用位置、词性、aspect这3种注意力，得到文本的更新表示，然后这3个表示，通过同构或异构的方式送入卷积网络，max-pooling。

8、Parameterized Convolutional Neural Networks for Aspect Level sentiment analysis。 emnlp2018

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

#### 利用情感词典、词性等信息 位置信息

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

172、Transformation Networks for Target-Oriented Sentiment Classification

多跳机制，每次更新每个单词的表示，最后使用cnn提取。

#### 层次化方法   针对文档级别，分层次

18、A Hierarchical Model of Reviews for Aspect-based Sentiment Analysis

每个文本由多个review组成，每个review先进行bi-lstm，然后多个review之间再lstm，输入同时包括review相对应的target

19、一种用于基于方面情感分析的深度分层网络模型

作者在18的基础上进行改进，先cnn再lstm，但是lstm的输入多了一项s，s：文本整体上进行lstm的最后一个结果，但是这个lstm时，hi会和target进行结合再输入到下一个。略微复杂

20、Aspect Sentiment Classification with both Word-level and Clause-level Attention Networks    ijcai2018

这个比较简单，就是先识别出多个clause，每个clause代表一个target的review，然后每个clause应用lstm-att，然后clause之间再应用lstm-att，但是aspect始终是同一个，并没有像18一样每个review有自己的aspect。

21、Document-level Multi-aspect Sentiment Classification by Jointly Modeling Users, Aspects, and Overall Ratings

引入了user、整体评论打分信息。和22有点像，每个d和一个aspect有一个classier，k个aspect得到k个classier（类似于multi-task）。先是word-level，att的时候有user和aspect信息，然后是sentence-level，得到文本表示后，和整体评分r向量和user向量结合。

22、Document-Level Multi-Aspect Sentiment Classification as Machine Comprehension

一个文本d有多个aspect，每个d和一个aspect作为输入得到一个classiser，k个aspect得到k个classier。内部有点复杂

23、CAN---Constrained Attention Networks for Multi-Aspect Sentiment Analysis

这个方法比较好，因为attention分布中确实是存在这2个问题

24、“Iarm: Inter-aspect relation modeling with memory networks in aspect-based sentiment analysis,

基本思路也是每个文本和aspect进行lstm-attention，然后aspect之间再进行lstm和

#### coling2018补充 （这个会议上有好几篇aspect方面的文章），总体上还是lstm+att

23、A Position-aware Bidirectional Attention Network for Aspect-level Sentiment Analysis

作者想强调位置信息的引入，但是吸引我的是网络架构：先给每个target的每个词得到一个权重，然后target和context进行交互得到一个句子，在target每个词下，文本每个词都有一个相对应的权重，然后加权相加得到该词下的文本表示。最后，target中每个词有一个权重，然后有一个对应的文本表示，加权相加得到最后的文本表示。

24、Effective Attention Modeling for Aspect-Level Sentiment Classification

作者的意思是单独平均target向量，可能不能很好的捕捉target的语义，于是重新定义了一个矩阵aspect，k*d, k是aspect的个数，得到target的一个重新表示，然后再做lstm+att。

25、Enhanced Aspect Level Sentiment Classification with Auxiliary Memory

作者提出要同时使用aspect和target两方面的信息，因为一般只有一个，对于只有target信息，初始化一个aspect矩阵，target与aspect进行attention，得到aspect的加权表示，然后这2个表示一起与context进行attention，其实这个思想对于category来说很适用。

26、Transformation Networks for Target-Oriented Sentiment Classification

作者不断更新文本中每个单词的表示，根据target，最后用一个cnn来进行提取。

27、Aspect-level Sentiment Classification with HEAT (HiErarchical ATtention) Network  （想法不错）

针对category（aspect）方向，第一次attention，找出与aspect相关的word（也就是越相关，权重越大），加权求和后得到category的另一种表示，然后使用这个表示再次进行attention。（其实这里还可以改进一下，就是不知道效果好不好，结合A Position-aware Bidirectional Attention Network for Aspect-level Sentiment Analysis这篇文章）

28、CAN: Constrained Attention Networks for Multi-Aspect SentimentAnalysis

这篇文章的motivation是这些文章中最吸引我的，针对一个文本中有多个multi-aspect的情况，每个aspect的attention的稀疏的，不同aspect的attention应该是不重叠的，但是在at-lstm中针对multi-aspect情况，attention经常重叠，尤其是不同aspect情况，sentiment情况不同时，往往不能判断出正确的sentiment，经常是所有aspect的sentiment判为同一个，解决了这个问题，准确率可以提高很多。

## 结果比较

Model | semeval2014-laptop | semeval2014-res(t) |semeval2014-res(c)|semeval2015|semeval2016|dong  |vo zhang
------|--------------------|--------------------|------------------|-----------|-----------|------|------
1     |--------------------|/                   |  84.0            |/          |/          |------|------
2     |72.37               |80.95               |/                 |/          |/          |------|------
3     |72.1                |78.6                |/                 |/          |/          |------|------
4     |74.5                |81.2                |/                 |/          |/          |------|------
5     |68.8                |75.44               |81.29             |/          |/          |------|------
6     |69.14               |77.28               |79.67             |/          |/          |------|------
7     |68.22               |79.17               |/                 |/          |/          |------|------
8     |/                   |/                   |/                 |/          |/          |71.5  |/
9     |/                   |/                   |/                 |/          |/          |      |71.96
10    |/                   |/                   |/                 |/          |/          |73.6  |75
11    |75.24               |81.34               |/                 |/          |/          |------|------
12    |74.49               |80.23               |/                 |/          |/          |------|------
13    |73.94               |82.03               |/                 |/          |/          |------|------
14    |/                   |/                   |/                 |/          |/               |------|------
15    |/                   |/                   |/                 |/          |/               |------|------
16    |/                   |/                   |/                 |/          |/              |72.8  |74.3
23    |74.12               |81.16               |/                 |/          |/               |------|------
24    |72.57               |80.63               |/                 |/          |/               |------|------
25    |74.45               |82.32               |86.33             |/          |/               |------|------
26    |76.54               |80.79               |86.33             |/          |/               |------|------


## 我的一些思路  （我觉得从抽象意义上讲还有点道理的）

1、针对term-target级别，有一个想法：来自Gated-Attention Readers for Text Comprehension这篇文章，target不更新，而是更新每个词多次，有1篇文章就是这个思路：Transformation Networks for Target-Oriented Sentiment Classification
在这个更新单词表示上，是否可以做到使用cnn的方法来更新单词呢

2、针对category-aspect级别，暂时有一个思路，还在写代码，
但是效果没有跑出来（是模型的问题还是我的问题）

3、考虑多信息（情感词、距离）是我觉得最重要的
但是怎么把这些信息加入网络中却无法找到自己满意的答案，抽象来讲，一个词越是情感词，距离target越近，则权重应该越大，因此一个想法是
文本与情感词典形成矩阵，通过相似度计算和softmax的方法，使得文本中的情感词的权重大（凸显出来），同时利用距离使得与target更相关的的情感词凸显出来。
（w*word+w*aspect+w*sentiment_word）,可是这样好不好呢？ 
