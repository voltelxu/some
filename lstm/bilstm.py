import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

'''
global variable
'''
train_rate = 0.8 #the percent of the train_data set 
lr = 0.01 # the learning rate
batch_size = 8 # the batch size of model
embed_size = 128 # the word embedding size
voc_size = 0 # the vocabulary size
max_num_step = 0 # the max number step unroll of the model
num_layers = 1 # the layers of the model
hidden_size = 500 # the layer size of the model
num_epoch = 20 # the train times of model


#get the vocabulary
def wordid(filename):
	dic = open(filename, 'r')
	voc = dict()
	for line in dic:
		line = line.strip('\n')
		lines = line.split(' ')
		voc[lines[0]] = int(lines[1])
	# lables
	lable = dict()
	lable['0'] = 0
	lable['1'] = 1
	lable['2'] = 2
	lable['3'] = 3
	lable['4'] = 4
	lable['5'] = 5
        lable['6'] = 6
        lable['7'] = 7
        lable['8'] = 8
        lable['9'] = 9
        lable['10'] = 10
        lable['11'] = 11

	return voc, lable

word_ids, lable_ids = wordid('dic')
voc_size = len(word_ids)
lable_size = 6

# get data set and chang to ids
# change the data set to train and test data set
def getdata(filename):
	datafile = open(filename, 'r')
	datas = list()
	#the max length of the sentence
	max_scelen = 0

	for line in datafile:
		words = line.strip('\n').split(" ")
		tup = list()
		data = list()
		lable = list()
		for word in words:
			terms = word.split(":")
			data.append(word_ids.get(terms[0], 0))
			lable.append(lable_ids[terms[1]])
			if len(data) > max_scelen:
				max_scelen = len(data)
		tup.append(data)
		tup.append(lable)
		datas.append(tup)

	len_data = len(datas)
	#shuffle the data set
	random.shuffle(datas)
	# datas length sentence
	dataslen = [length for length in [len(x[0]) for x in datas]]
	# padding data
	for d in datas:
		l = len(d[0])
		d[0] += [0 for i in range(max_scelen - l)]
		d[1] += [0 for i in range(max_scelen - l)]
	#split position
	split = int(len_data * train_rate)
	#split dataset to train and test sets
	train = datas[0 : split]
	test = datas[split : len_data]
	#train_ data
	train_data = [t for t in [x[0] for x in train]]
	#train_lable
	train_lable = [t for t in [x[1] for x in train]]
	# train lenght
	train_len = dataslen[0 : split]
	#test_data
	test_data = [t for t in [x[0] for x in test]]
	#test_lable
	test_lable = [t for t in [x[1] for x in test]]
	# test_data length0
	test_len = dataslen[split : len_data]
	#return train and test set
	return train_data, train_lable, train_len, test_data, test_lable, test_len, max_scelen

train_data,train_lable, train_len, test_data, test_lable, test_len, max_len = getdata("lable")
# max sequence length 
max_num_step = max_len

# get batch input data
def getbatch(data, lable, lens, start, batch_size):
	batch_data = data[start : start + batch_size]
	batch_seqlen = lens[start : start + batch_size]
	batch_lable = lable[start : start + batch_size]
	return batch_data, batch_lable, batch_seqlen

# batch_data, batch_lable, batch_seqlen = getbatch(train_data, train_lable, 1, 8)
# print len(batch_data)

def dynamicRNN(inputs, seqlen, weights, bias):
	#
	# bi lstm 
	#lstm cell
	lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
	lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
	#dropout
        #if is_training:
	#    lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=0.7)
        #    lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=0.7)

	lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * num_layers)
	lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * num_layers)
	#outputs and state
	outputs, state = tf.nn.bidirectional_dynamic_rnn(
		lstm_cell_fw,
		lstm_cell_bw,
		inputs,
		dtype=tf.float32,
		sequence_length=seqlen)

	outputs = outputs[0] + outputs[1]

	outputs = tf.reshape(outputs, [batch_size * max_num_step, hidden_size])
	#outputs shape [batch_size, num_step, hidden_size]

	outputs = tf.matmul(outputs, weights) + bias

	return outputs

def model():
	# input data
	input_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_num_step])
	# targets lables
	target = tf.placeholder(dtype=tf.int32, shape=[batch_size, max_num_step])
	# true seq lenght
	seqlen = tf.placeholder(dtype=tf.int32, shape=[None])
	# seq mask [1, 0]
	# selmask = tf.sequence_mask(seqlen, maxlen=max_num_step, dtype=tf.float32)
	seqem = tf.constant(np.tril(np.ones([max_num_step + 1, max_num_step]), -1), dtype=tf.float32)
	seqmask = tf.nn.embedding_lookup(seqem, seqlen)
	# embeddings
	embedding = tf.get_variable(name='embedding', shape=[voc_size, embed_size])
	# change word to tensor
	inputs = tf.nn.embedding_lookup(embedding, input_data)
	# weights 
	weights = tf.get_variable("weights",[hidden_size, lable_size])
	# bias
	bias = tf.get_variable("bias", [lable_size])
	# rnn cell
	logits = dynamicRNN(inputs, seqlen, weights, bias)
	#
	pred = tf.nn.softmax(logits)
	pred = tf.cast(tf.argmax(pred, 1), tf.int32)
	# out = tf.argmax(pred,1)
	correct = tf.cast(tf.equal(pred, tf.reshape(target, [-1])), tf.int32) * tf.cast(tf.reshape(seqmask, [-1]), tf.int32)
	# accuracy
	accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))/tf.reduce_sum(tf.cast(seqmask, tf.float32))
	#
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(target, [-1]))
	loss = loss * tf.reshape(seqmask, [-1])
	cost = tf.reduce_sum(loss) / tf.reduce_sum(seqmask)
	#optimizer
	opt = tf.train.AdamOptimizer(lr).minimize(cost)
	# 
	'''logits = tf.reshape(pred, [batch_size, max_num_step, lable_size])

	loss = tf.contrib.seq2seq.sequence_loss(logits,
	 			target, 
	 			tf.ones([batch_size, max_num_step]))
	cost = tf.reduce_mean(loss)

	softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, lable_size]))
	predict = tf.cast(tf.argmax(softmax_out, axis=1),tf.int32)
	correct = tf.equal(predict, tf.reshape(target,[-1]))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	tf.summary.scalar("accuracy", accuracy)

	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	opt = tf.train.GradientDescentOptimizer(learning_rate=lr).apply_gradients(zip(grads,tvars))
	'''
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		writer = tf.summary.FileWriter('./path', sess.graph)
		merged = tf.summary.merge_all()
		# train
		epoch_size = len(train_data)/batch_size
		testsize = len(test_data)/batch_size
		'''
		for s in range(1):
			alls = 0.0
			for i in range(1):
				x, y, l = getbatch(train_data, train_lable, train_len, i * batch_size, batch_size)
				x = sess.run(logits, feed_dict={input_data : x, target : y, seqlen : l})
			print tf.shape(x)
		'''
		train_ac = list()
		train_loss = list()
		test_ac = list()
		test_loss = list()
		steps = list()
		for step in range(num_epoch):
			steps.append(step)
			sum_loss = 0.0
			sum_ac = 0.0
			num = 0
			ta =0.0
			tl = 0.0
			for i in xrange(0, epoch_size, 1):
				print i
				x, y, l = getbatch(train_data, train_lable, train_len, i * batch_size, batch_size)
				# print y
				_, ac, lo = sess.run([opt, accuracy, cost], feed_dict={input_data : x, target : y, seqlen : l})
				# print ac
				ta = ta + ac
				tl = tl + lo
			ta = ta/epoch_size
			train_ac.append(ta)
			train_loss.append(tl)
			print "step train "+str(step)+", accuracy "+str(ta)+", loss " + str(tl)
			for j in xrange(0, testsize, 1):
				num = num + 1
				tx, ty, tl = getbatch(test_data, test_lable, test_len, j * batch_size, batch_size)
				ac, loss = sess.run([accuracy, cost], feed_dict={input_data : tx, target : ty, seqlen : tl})
				sum_ac = sum_ac + ac
				sum_loss = sum_loss + lo
			sum_ac = sum_ac/num
			test_ac.append(sum_ac)
			test_loss.append(sum_loss)
			print "step test "+str(step)+", accuracy "+str(sum_ac)+", loss " + str(sum_loss)
		#plot fig
		pa = plt.figure()
		ps = plt.figure()
		fa = pa.add_subplot(111)
		fs = ps.add_subplot(111)
		fa.plot(steps, train_ac)
		fs.plot(steps, train_loss)
		fa.plot(steps, test_ac)
		fs.plot(steps, test_loss)
		plt.show()
model()
