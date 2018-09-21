# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:35:12 2018

@author: Administrator
"""

import tensorflow as tf

class lstmattention(object):
    def __init__(self,config,is_training,embeddingl,initializer):
        
        self.wordsnum=config.wordsnum
        self.aspectnum=config.aspectnum
        self.wordtotext_hidsize=config.wordtotext_hidsize
        self.keep_prob=config.keep_prob
        self.batchsize=config.batchsize
        self.l2=config.l2
        self.lr=config.lrre
        self.classnum=config.classnum        
        self.ase=config.ase
        self.vocab_size=len(embeddingl)
        self.wordEmbeddingSize=config.wordEmbeddingSize
        
        self.filter_sizes=config.filter_sizes
        self.num_filters=config.num_filters
        self.filternums=config.filternums
        #input
        self.input_x = tf.placeholder(tf.int32, [None, self.wordsnum], name="input_x")  # X  #batch,words
        self.input_aspect=tf.placeholder(tf.int32,[None,self.wordsnum,self.aspectnum],name="input_aspect")    #batch*words,aspnums   
        self.aspectmask=tf.placeholder(tf.float32,[None,self.wordsnum,self.aspectnum],name="aspectmask")  #b*w,a
        self.pos=tf.placeholder(tf.float32,[None,self.wordsnum],name="pos")   #b*w,1
        self.y=tf.placeholder(tf.int64,[None],name="y")
        
        self.embedding = tf.get_variable(name="embedding",shape=[self.vocab_size, self.wordEmbeddingSize],initializer=tf.constant_initializer(embeddingl))#,shape=[self.vocab_size, self.embed_size]) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
        self.initial_weight(initializer)
        
        
        self.aspectl=tf.nn.embedding_lookup(self.embedding,self.input_aspect)    #b,w,a,h        
        #modify self.aspectl
        mask=tf.expand_dims(self.aspectmask,3) #b*w,a,1
        a=tf.multiply(self.aspectl,mask)# [b*w,aspnum,emb]  [b*w,aspnum,1]
        #print (a)
        b=tf.reduce_sum(a,2) #[b*w,emb]
        #print (b)
        poss=tf.expand_dims(self.pos,2) #[b*w,1]
        #print (poss)
        self.aspect_final=tf.div(b,poss) #[b,w,em]
        print (self.aspect_final)
        
        self.inpute=tf.nn.embedding_lookup(self.embedding,self.input_x) #batch,words,wordsize
        self.sentence_embeddings_expanded=tf.expand_dims(self.inpute,-1)
        
        self.final_docp=[]
        self.att=[]
        for i,filter_size in enumerate(self.filter_sizes):
            #cur=[]
            with tf.name_scope("%s"%filter_size):
                with tf.variable_scope("cnn-%s"%filter_size):
                    filterc=tf.get_variable("filter-%s"%filter_size,[filter_size,self.wordEmbeddingSize,1,self.num_filters],initializer=initializer)
                    conv=tf.nn.conv2d(self.sentence_embeddings_expanded,filterc,strides=[1,1,1,1],padding="VALID",name="conv")
                    b=tf.get_variable("bias-%s"%filter_size,[self.num_filters],initializer=initializer)
                    h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")#[batchsize,seq-filtr+1,1,num_filters]
                    self.cnnem=tf.reshape(h,[-1,self.wordsnum-filter_size+1,self.num_filters])
        
                    #binput2=tf.split(self.cnnem,self.wordsnum-filter_size+1,1)  #list,words,batch,1,emb       
                    #self.binput3=[tf.squeeze(x,[1]) for x in binput2] #
                    #attention
                    hidden_sen_2=tf.reshape(self.cnnem,[-1,self.wordtotext_hidsize]) #batch*words,hid       
                    
                    '''
                    #another attention
                    
                    '''
                    sa=tf.matmul(hidden_sen_2,self.ww)+self.wb  #batch*words,ase
                    
                    #aspect attention  to process for every filter,aspect nums is not same
                    self.aspect_final1=self.aspect_final[:,0:self.wordsnum-filter_size+1,:]
                    #print 'test'                    
                    #print (self.aspect_final1)
                    aspre=tf.reshape(self.aspect_final1,[-1,self.wordEmbeddingSize])
                    aspa=tf.matmul(aspre,self.wwa)  #batch*words,ase
                    
                    sa=sa+aspa  
                    sh_r1=tf.nn.tanh(sa)
                    sh_r=tf.reshape(sh_r1,[-1,self.wordsnum-filter_size+1,self.ase])#batch,words,hid
                    ssimi=tf.multiply(sh_r,self.context)  # batch,words,ase
                    sato=tf.reduce_sum(ssimi,2)  #batch,word
                    smaxhang=tf.reduce_max(sato,1,True)
                    satt=tf.nn.softmax(sato-smaxhang) #batch,wordsattention
                    
                    self.att.append(satt)
                    
                    sae=tf.expand_dims(satt,2) #batch,words,1
                    docp=tf.multiply(sae,self.cnnem) #batch,sens,hid
                    self.final_docp.append(tf.reduce_sum(docp,1))
             
        self.final_docps=tf.concat(self.final_docp,1) #batch,hid*filternums
        #print (self.final_docps)
        #final mlp
        self.logits=tf.matmul(self.final_docps,self.w1)+self.b1 #bath,classe
        
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
        #optimizer = tf.train.AdadeltaOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op=optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)

                                             
                    
    def initial_weight(self,initial):
        self.w1=tf.get_variable("fw",shape=[self.wordtotext_hidsize*self.filternums,self.classnum],initializer=initial)
        self.b1=tf.get_variable("fb",shape=[self.classnum],initializer=tf.zeros_initializer())
        
        self.ww=tf.get_variable("ww_sen",shape=[self.wordtotext_hidsize,self.ase],initializer=initial)
        self.wwa=tf.get_variable("wwa_sen",shape=[self.wordEmbeddingSize,self.ase],initializer=initial)        
        self.wb=tf.get_variable("wb_sen",shape=[self.ase],initializer=tf.zeros_initializer())           
        self.context=tf.get_variable("context_word",shape=[self.ase],initializer=initial)
