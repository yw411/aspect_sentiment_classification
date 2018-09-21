# -*- coding: utf-8 -*-
"""
Created on Tue Jul 03 00:22:05 2018

@author: Administrator
"""
import tensorflow as tf

class rnnmodel(object):
    def __init__(self,config,is_training,embeddingl,initial):
        
        #param
        self.wordEmbeddingSize=config.wordEmbeddingSize
        self.wordsnum=config.wordsnum
        self.wordtotext_hidsize=config.wordtotext_hidsize
        self.keep_prob=config.keep_prob
        self.batchsize=config.batchsize
        #self.wssize=config.wssize
        self.classnum=config.classnum
        
        self.lr=config.lrre
        self.l2=config.l2
        self.vocab_size=len(embeddingl)
        
        #input
        self.input_x=tf.placeholder(tf.int32,[None,self.wordsnum])  #batch*words
        #self.xbackward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])
        self.y=tf.placeholder(tf.int64,[None])
        #self.maskforward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])
        #self.maskbackward=tf.placeholder(tf.int32,[self.batchsize,self.wordsnum])
        #self.seqlen=tf.placeholder(tf.int32,[self.batchsize])
        
        #embedding=tf.constant(embeddingl,dtype=tf.float32)
        embedding = tf.get_variable(name="embedding",shape=[self.vocab_size, self.wordEmbeddingSize],initializer=tf.constant_initializer(embeddingl))#,shape=[self.vocab_size, self.embed_size]) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

        
        self.initial_weight(initial)
        self.embedding_words=tf.nn.embedding_lookup(embedding,self.input_x)
        
        self.final_docp=tf.reduce_mean(self.embedding_words,axis=1)
        #final mlp
        self.logits=tf.matmul(self.final_docp,self.w1)+self.b1 #bath,classe
        
        #define loss
        with tf.name_scope("losscost_layer"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            self.cost = tf.reduce_mean(self.loss)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.l2
            self.cost=self.cost+l2_losses
            
        #define accuracy
        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.y)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")
        if not is_training:
            return
        
        #optimialize
        self.global_step = tf.Variable(0,name="global_step",trainable=False)       
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),config.max_grad_norm)     
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op=optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)

                            
    def initial_weight(self,initial):
        self.w1=tf.get_variable("fw",shape=[self.wordtotext_hidsize,self.classnum],initializer=initial)
        self.b1=tf.get_variable("fb",shape=[self.classnum],initializer=tf.zeros_initializer())
        