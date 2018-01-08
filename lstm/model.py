import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class model():
    '''bi lstm model'''
    def __init__(self, learn_rate=0.01, batch_size=64, embed_size=128, voc_size=1613, classes=12, hidden_size=400, num_layers=1, decay_steps=12000, decay_rate=0.9, l2_lambda=0.0001, is_train=True):
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_train = is_train
        self.lable_size = classes
        self.l2_lambda = 0.0001
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.model()

    def model(self):
        self.input_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None])
        self.target = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.seqlen = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.prob = tf.placeholder(dtype=tf.float32)
        self.weight = tf.get_variable(name='weight', shape=[self.hidden_size * 2, self.lable_size])
        self.bias = tf.get_variable(name='bias', shape=[self.lable_size])
        self.embed = tf.get_variable(name='emded', shape=[self.voc_size, self.embed_size])

        self.global_step = tf.Variable(0, trainable=False)
        self.epoch_step=tf.Variable(0,trainable=False)
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))

        self.inputs = tf.nn.embedding_lookup(self.embed, self.input_data)

        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)

        if self.prob is not None:
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.prob)
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.prob)

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        self.rnn_out, state = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, self.inputs, dtype=tf.float32, sequence_length=self.seqlen)

        self.outputs = tf.concat(self.rnn_out, axis=2)
        self.output = tf.reduce_mean(self.outputs, axis=1)

        self.logits = tf.matmul(self.output, self.weight) + self.bias

        self.softmax = tf.nn.softmax(self.logits)

        self.pred = tf.cast(tf.argmax(self.softmax, 1), tf.int32)

        correct = tf.cast(tf.equal(self.pred, tf.reshape(self.target, [-1])), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        if not self.is_train:
            return

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
        loss = tf.reduce_mean(losses)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda

        self.loss = loss + l2_loss

        learn_rate = tf.train.exponential_decay(self.learn_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,learning_rate=learn_rate, optimizer="Adam")

    def train(self, train):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(len(train.data)/self.batch_size):
                x, y, l = train.getbatch(self.batch_size)
                print y
                ac,pr,t = sess.run([self.accuracy,self.pred,self.train_op], feed_dict={self.input_data:x, self.target:y, self.seqlen:l})
                print pr
                print ac
