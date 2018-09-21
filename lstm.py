#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:41:48 2018

@author: gaoyiwei

c---lstm
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
        
        #cnn part [batchsize,seq,num_filters]
        self.embedding_words=tf.nn.embedding_lookup(embedding,self.input_x)
        
        binput2=tf.split(self.embedding_words,self.wordsnum,1)  #list,words,batch,1,emb       
        self.binput3=[tf.squeeze(x,[1]) for x in binput2] #
        
        
        #lstm
        with tf.name_scope("lstm"):
            
            blstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.wordtotext_hidsize)
            if self.keep_prob<1:
                blstm_cell=tf.nn.rnn_cell.DropoutWrapper(blstm_cell,output_keep_prob=self.keep_prob)
            bcell=tf.nn.rnn_cell.MultiRNNCell([blstm_cell]*1)
            (bhidden_state_1,_)=tf.nn.dynamic_rnn(bcell,self.embedding_words,dtype=tf.float32)
            '''
            self.binitial_state=bcell.zero_state(self.batchsize,tf.float32)  #initial state  t=0 :c and h                   
            self.boutput=[]
            bstate=self.binitial_state   
            with tf.variable_scope("LSTM_layer"):
                for time_step,data in enumerate(self.binput3):
                    if time_step>0:
                        tf.get_variable_scope().reuse_variables()
                    cell_output,bstate=bcell(data,bstate)
                    self.boutput.append(cell_output)
            '''        
        #bhidden_state_1=tf.stack(self.boutput,axis=1) #batch,words,hid
        
        
        self.final_docp=tf.reduce_mean(bhidden_state_1,axis=1) #batch ,hid
                        
        
        
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
        
        #self.ww=tf.get_variable("ww_sen",shape=[self.wordtotext_hidsize,self.wssize],initializer=initial)
        #self.wb=tf.get_variable("wb_sen",shape=[self.wssize],initializer=initial)           
        #self.context=tf.get_variable("context_word",shape=[self.wssize],initializer=initial)
