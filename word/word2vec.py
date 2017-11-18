import os
import collections
import random
import tensorflow as tf
import numpy as np
import math


def read_data(filename):
	f = open(filename, 'r')
	s = '';
	for line in f:
		s = s + line
	data = tf.compat.as_str(s).split()
	return data

data = read_data('item')

vocabulary_size = 150

def build_dataset(words, n_words):
	count = [['UNK', -1]]
	#get the most common n words
	count.extend(collections.Counter(words).most_common(n_words - 1))
	dictionary = dict()
	for word, _ in count:
		#give every word a number
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		index = dictionary.get(word, 0)
		if index == 0:
			unk_count += 1
		#parse the sentence into index sentence
		data.append(index)
	count[0][1] = unk_count
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	#data:number list of sentence
	#count:sorted count of every word in the sentence
	#dictionary: word, count_number
	#reversed_dictionary: count_number, word
	return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(data, vocabulary_size)

print(data[:10])
#print(data[:10],[reverse_dictionary[i] for i in data[:10]])
data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen=span)
	if data_index + span > len(data):
		data_index = 0
	buffer.extend(data[data_index:data_index + span])
	data_index += span
	for i in xrange(batch_size//num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2)

#4
batch_size = 8
embedding_size = 128
skip_window = 1
num_skips = 2
num_sampled = 64

#geneate valid data set
valid_size = 8
valid_window = 20
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

print(valid_examples)

graph = tf.Graph()

num_steps = 20001

with graph.as_default():
	#input data
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name="input")
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="labels")
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32, name="Const")
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embedsave')
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)
	#
	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size], dtype=tf.float32, name=None))
	#noise-contrastive estimation training loss
	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size, name="nce_loss"))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0, use_locking=False, name="GradientDescent").minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	init = tf.global_variables_initializer()
	#saver = tf.train.Saver({"embed": embeddings})
	with tf.Session() as sess:
		init.run()
		print("Initialized")
		average_loss = 0
		for step in xrange(num_steps):
			batch_input, batch_labels = generate_batch(batch_size, num_skips, skip_window)
			feed_dict = {train_inputs:batch_input, train_labels:batch_labels}

			_, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict, options=None, run_metadata=None)
			average_loss += loss_val
			if step % 2000 == 0:
				if step > 0:
					average_loss /= 2000
				#print("Average_loss at step", step, ': ', average_loss)
			average_loss = 0
			if step % 100 == 0:
				sim = similarity.eval()
				for i in xrange(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 5
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = 'Nearest to %s:' %valid_word
					for k in xrange(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log_str = '%s %s,' %(log_str, close_word)
					print(log_str)
		final_embeddings = normalized_embeddings.eval()
		out_embed = dict()
		line = ""
		for i in range(len(reverse_dictionary)):
			#out_embed[reverse_dictionary[i]] = final_embeddings[i]
			line = line + reverse_dictionary[i] + ":" 
			nums = ""
			for j in range(len(final_embeddings[i])):
				nums = nums + str(final_embeddings[i][j]) + " "
			line = line + nums + "\n"
		true_embed = open('true_embed', 'w')
		true_embed.write(line)
		

def plot_with_labels(low_dim_embs, labels, filename):
	assert low_dim_embs.shape[0] >= len(labels)
	plt.figure(figsize=(18,18))
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i, :]
		plt.scatter(x, y)
		plt.annotate(label,xy=(x, y), xytext=(5,2), textcoords='offset points',ha='right', va='bottom')
		plt.savefig(filename)

try:
	from sklearn.manifold import TSNE
	import matplotlib.pyplot as plt
	import sys
	reload(sys)
	sys.setdefaultencoding('UTF-8')
	#print("------------")
	tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
	plot_only = 20
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
	labels = [reverse_dictionary[i] for i in xrange(plot_only)]
	#plot_with_labels(low_dim_embs, labels, '123.png')
except ImportError:
	print('error')