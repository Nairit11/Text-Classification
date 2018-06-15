# Import necessary Libraries
import numpy as np
import tensorflow as tf
import csv
import re
import datetime
import time
import sys
import os
from math import sqrt

# Class with different utility functions to handle and prepare data sets
class Data(object):
    
    def __init__(self,
                 data_source,
                 alphabet ,
                 l0 ,
                 batch_size ,
                 no_of_classes ):
        
        self.alphabet = alphabet
        self.alphabet_set_size = len(self.alphabet)
        self.dict = {}
        self.no_of_classes = no_of_classes
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1

        
        self.length = l0
        self.batch_size = batch_size
        self.data_source = data_source

    def loadData(self):
        data = []
        with open(self.data_source, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                data.append ((int(row[0]), txt))
        self.data = np.asarray(data)
        self.shuffled_data = self.data
        
    def shuffleData(self):
        np.random.seed(235)
        data_size = len(self.data)        
        shuffle_indices = np.random.permutation(np.arange(data_size))
        self.shuffled_data = self.data[shuffle_indices]         
        
    def getBatch(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = min((batch_num + 1) * self.batch_size, data_size)
        return self.shuffled_data[start_index:end_index]

    def getBatchToIndices(self, batch_num = 0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min((batch_num + 1) * self.batch_size, data_size)
        batch_texts = self.shuffled_data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), classes
            
    def strToIndexs(self, s):
   
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64') 
        k = 0
        for i in range(1, n+1):
            c = s[-i]
            if c in self.dict:
                str2idx[i-1] = self.dict[c]            
        return str2idx

    def getLength(self):
        return len(self.data)

# Class to create the CNN 
class CharConvNet(object):

    def __init__(self,
                conv_layers = [
                    [256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
                    ],
                 fully_layers = [1024, 1024],
                 l0 = 1014,
                 alphabet_set_size = 69,
                 no_of_classes = 4,
                 th = 1e-6):


    
        seed = time.time()        
        tf.set_random_seed(seed)

        # Inserting Input Layer
        with tf.name_scope("Input-Layer"):
            self.input_x = tf.placeholder(tf.int64, shape = [None, l0], name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape = [None, no_of_classes], name = 'input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # Vectorizing the input layer
        with tf.name_scope("Embedding-Layer"), tf.device('/cpu:0'): 
            Q = tf.concat(
                          [
                              tf.zeros([1, alphabet_set_size]), # All vectors initialized to zero
                              tf.one_hot(list(range(alphabet_set_size)), alphabet_set_size, 1.0, 0.0) # One-hot vector representation for alphabets
                           ],
                          0,
                          name='Q')            
            x = tf.nn.embedding_lookup(Q, self.input_x)
            x = tf.expand_dims(x, -1)            
        var_id = 0

        # Inserting Convolution layers
        for i, cl in enumerate(conv_layers):
            var_id += 1 
            # Single Convolution layer
            with tf.name_scope("ConvolutionLayer"):
                filter_width = x.get_shape()[2].value
                filter_shape = [cl[1], filter_width, 1, cl[0]] 
                stdv = 1/sqrt(cl[0]*cl[1])
                W = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv), dtype='float32', name='W' ) 
                b = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name = 'b') 
                
                conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID", name='Conv')
                x = tf.nn.bias_add(conv, b)
            # Customery Threshold layer after ever Convolution layer
            with tf.name_scope("ThresholdLayer"):
                x = tf.where(tf.less(x, th), tf.zeros_like(x), x)

                
            # Maxpooling layer for outputs of each Convolution layer
            if not cl[-1] is None:
                with tf.name_scope("MaxPoolingLayer" ):
                    pool = tf.nn.max_pool(x, ksize=[1, cl[-1], 1, 1], strides=[1, cl[-1], 1, 1], padding='VALID')
                    x = tf.transpose(pool, [0, 1, 3, 2])
            else:
                x = tf.transpose(x, [0, 1, 3, 2], name='tr%d' % var_id)
                
        # Reshaping the output of Convolution layer to serve as input for Fully Conncted Layer
        with tf.name_scope("ReshapeLayer"):
            vec_dim = x.get_shape()[1].value * x.get_shape()[2].value            
            x = tf.reshape(x, [-1, vec_dim])

        
        
        weights = [vec_dim] + list(fully_layers) # The weights for the connection between reshape layer to fully connected layers for the first time is used-defined
        # Inserting Fully-Connected layers
        for i, fl in enumerate(fully_layers):
            var_id += 1
            # Single Fully-connected layer
            with tf.name_scope("LinearLayer" ):                
                stdv = 1/sqrt(weights[i])
                W = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype='float32', name='W')
                b = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype='float32', name = 'b')
                x = tf.nn.xw_plus_b(x, W, b)

            # Customery Threshold layer for each Fully-connected layer                    
            with tf.name_scope("ThresholdLayer" ):
                x = tf.where(tf.less(x, th), tf.zeros_like(x), x)
                
            # Dropout layer for each Fully-connected layer
            with tf.name_scope("DropoutLayer"):
                x = tf.nn.dropout(x, self.dropout_keep_prob)
                
        # Insert output layer, depends on the no_of_classes and the dataset
        with tf.name_scope("OutputLayer"):
            stdv = 1/sqrt(weights[-1])
            W = tf.Variable(tf.random_uniform([weights[-1], no_of_classes], minval=-stdv, maxval=stdv), dtype='float32', name='W')
            b = tf.Variable(tf.random_uniform(shape=[no_of_classes], minval=-stdv, maxval=stdv), name = 'b')
            self.p_y_given_x = tf.nn.xw_plus_b(x, W, b, name="scores")
            self.predictions = tf.argmax(self.p_y_given_x, 1)

        # Calculate Loss     
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.p_y_given_x,labels= self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate Accuracy of predictions
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# Paths to training and test data
train_data_source = 'data/train.csv'
test_data_source = 'data/test.csv'

# Metrics defined as per Research Paper
alphabet_set = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_set_size = len(alphabet_set)
IP_feature_lenght = 1014
batch_size = 128
no_of_classes = 4

# Loading Training data
train_data = Data(data_source = train_data_source,
                      alphabet = alphabet_set,
                      l0 = IP_feature_lenght,
                      batch_size = batch_size,
                      no_of_classes = no_of_classes)
train_data.loadData()
# Loading Test data
test_data = Data(data_source = test_data_source,
                      alphabet = alphabet_set,
                      l0 = IP_feature_lenght,
                      batch_size = batch_size,
                      no_of_classes = no_of_classes)
    
test_data.loadData()

num_batches_per_epoch = int(train_data.getLength() / batch_size) + 1
num_batch_test = test_data.getLength()

with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=False))
        with sess.as_default():
            # Metrics passed to constructor are taken from Research Paper
            char_cnn = CharConvNet(conv_layers = [
                    [256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
                    ],
                 fully_layers = [1024, 1024],
                 l0 = IP_feature_lenght,
                 alphabet_set_size = alphabet_set_size,
                 no_of_classes = no_of_classes,
                 th = 1e-6)

            global_step = tf.Variable(0, trainable=False)
            
            boundaries = []
            br = 1e-2 
            values = []
            for i in range(1, 10):
                values.append(br / (2 ** i))
                boundaries.append(15000 * i)
            values.append(br / (2 ** (i + 1)))

            # Learning rate determined by piecewise_constant function and Optimizer used is AdamOptimizer
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(char_cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

            # Keeping track of gradient values by plotting histogams
            grad_histories = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_history = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    grad_histories.append(grad_history)
                    
            grad_history_overall = tf.summary.merge(grad_histories)

            # Output directory for models
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Output written to {}\n".format(out_dir))

            # Summaries for loss and accuracy of CNN
            loss_summary = tf.summary.scalar("loss", char_cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", char_cnn.accuracy)

            # Training data results summary
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_history_overall])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Test data results summary
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory for Tensorflow
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())
            
            # Training step, single step                         
            def train_step(x_batch, y_batch):
                feed_dict = {
                  char_cnn.input_x: x_batch,
                  char_cnn.input_y: y_batch,
                  char_cnn.dropout_keep_prob: 0.5
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op,
                     global_step,
                     train_summary_op,
                     char_cnn.loss,
                     char_cnn.accuracy],
                    feed_dict)
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # Test step, single step  
            def test_step(x_batch, y_batch, writer=None):
                feed_dict = {
                  char_cnn.input_x: x_batch,
                  char_cnn.input_y: y_batch,
                  char_cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step,
                     test_summary_op,
                     char_cnn.loss,
                     char_cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            
            # Looping the training step for entire batch, for given no. of epochs
            for e in range(100): # Here 100 epochs have been considered. Usually, the more the number of epochs, the better the accuracy grows gadually
                train_data.shuffleData()
                for k in range(num_batches_per_epoch):
                    batch_x, batch_y = train_data.getBatchToIndices(k)
                    train_step(batch_x, batch_y)
                    current_step = tf.train.global_step(sess, global_step)
                    
                    if current_step % 10 == 0:   # Evaluation done every 10 steps, set according to total epochs
                        xin, yin = test_data.getBatchToIndices()
                        print("\nEvaluation:")
                        test_step(xin, yin, writer=test_summary_writer)
                        print("")
                        
                    if current_step % 10 == 0:   # Checkpoint saved every 10 steps, set according to total epochs
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))               
