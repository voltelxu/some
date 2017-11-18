import tensorflow as tf
import numpy as np
import os
import collections
import zipfile

word_size = 150
embedding_size = 128

def readvecfile(filename):
	file = open(filename, 'r')
	embeddings = np.zeros((word_size + 1, embedding_size))
	dictionary = dict()
	for line in file:
		line = line.strip(' \n')
		data = line.split(":")
		word = data[0]
		vec = data[1].split(" ")
		vec = map(eval, vec)
		vec = np.array(vec)
		embeddings[len(dictionary)] = vec
		dictionary[word] = len(dictionary)
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return embeddings, dictionary, reversed_dictionary

embeddings, dictionary, reversed_dictionary= readvecfile("true_embed")

def build_dataset(filename):
	file = open(filename, 'r')
	words = ""
	lables = ""
	for line in file:
		line = line.strip('\n')
		line = line.split(" ")
		words = words + line[0] + " "
		lables = lables + line[1] + " "
	words = words.split(" ")
	lables = lables.split(" ")
	data = list()
	for word in words:
		index = dictionary.get(word, 0)
		data.append(index)
	return data, lables

data, data_y = build_dataset('lables')
print(data)
data_index = 0
def get_batchset(batch_size, skip_window):
	global data_index
	span = skip_window * 2 + 1
	batch = np.ndarray(shape=(batch_size, span), dtype=np.int32)
	lables = np.ndarray(shape=(batch_size,4), dtype=np.int32)
	if data_index + span > len(data):
		data_index = 0
	for i in range(batch_size):
		one_data = np.ndarray(shape=(span), dtype=np.int32)
		one_data[skip_window] = data[data_index]
		start = data_index - skip_window
		for j in range(span):
			one_data[j] = data[j + start]
		batch[i] = one_data
		y = data_y[data_index]
		if y == "B":
			lables[i] = np.asarray((1,0,0,0))
		if y == "E":
			lables[i] = np.asarray((0,1,0,0))
		if y == "M":
			lables[i] = np.asarray((0,0,1,0))
		if y == "S":
			lables[i] = np.asarray((0,0,0,1))	
		data_index = data_index + 1
	return batch, lables

#batch, lables = get_batchset(8, 2)
