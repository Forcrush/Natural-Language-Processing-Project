# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-04-27 14:57:08
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-05-11 15:08:46


print(''' Method 3
=============================================
| Ensemble Classifier + Samples Enhancement |
=============================================
''')


import numpy as np
import json
import re, collections
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import random

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


def get_synonyms(word):
	synonyms = []
	for syn in wn.synsets(word):
		for w in syn.lemmas():
			synonyms.append(w.name())
			'''
			# get antonyms
			if w.antonyms():
				print(w.antonyms()[0].name())
			'''
	if synonyms:
		return random.choice(synonyms)
	return None


def neg_augmentation(sentence):
	sw = set(stopwords.words('english'))
	words = sentence.split(" ")
	for i in range(len(words)):
		# only considering pure word
		if re.match('^[a-z]+$', words[i]) and words[i] not in sw:
			substitution = get_synonyms(words[i])
			# 50% probability to substitute original word
			if substitution:
				words[i] = substitution if random.uniform(0,1) > 0.5 else words[i]
	return " ".join(words)


def samples_enhancement_from_dev():
	with open("balanced.json", 'r') as f:
		temp = json.loads(f.read())
		text, label = [], []
		for key,val in temp.items():
			text.append(val['text'])
			label.append(val['label'])

		index = 0
		new_text = []
		new_label = []
		while index < len(text):
			if label[index] == 1:
				text.pop(index)
				label.pop(index)
			else:
				index += 1
		
		assert(len(text) == 50)

		for i in [0, 10, 20, 30]:
			tmp_text = text[i:i+20]
			n_text, n_label = [], []
			for st in tmp_text:
				# for each negative sample, deriving 10 neg samples
				for _ in range(10):
					new_neg = neg_augmentation(st)
					n_text.append(new_neg)
					n_label.append(0)

			new_text.append(n_text)
			new_label.append(n_label)

		# random choose 20 original sentences to derive
		tmp_text = random.sample(text, 20)
		n_text, n_label = [], []
		for st in tmp_text:
			# for each negative sample, deriving 10 neg samples
			for _ in range(10):
				new_neg = neg_augmentation(st)
				n_text.append(new_neg)
				n_label.append(0)

		new_text.append(n_text)
		new_label.append(n_label)

		return new_text, new_label


def get_train():
	
	text, label = [], []
	with open("train.json", 'r') as f:
		temp = json.loads(f.read())
		for key,val in temp.items():
			text.append(val['text'])
			label.append(val['label'])
	
	# neg_text: [[200 neg sentences], [200 neg sentences], [200 neg sentences], [200 neg sentences], [200 neg sentences]]
	neg_text, neg_label = samples_enhancement_from_dev()

	# pos_text: [[240 neg sentences], [240 neg sentences], [240 neg sentences], [240 neg sentences], [208 neg sentences]]
	pos_text = [text[:240], text[240:480], text[480:720], text[720:960], text[960:]]
	pos_label = [label[:240], label[240:480], label[480:720], label[720:960], label[960:]]
	assert(len(pos_text) == len(neg_text))

	ensemble_text = []
	ensemble_label = []
	for i in range(len(pos_text)):
		ensemble_text.append(pos_text[i] + neg_text[i])
		ensemble_label.append(pos_label[i] + neg_label[i])

	ensemble_ds = []
	for j in range(len(ensemble_text)):
		tx = ensemble_text[j]
		lb = ensemble_label[j]
		# shuffle
		rd_index = np.random.RandomState(26).permutation(len(tx))
		tx = np.array(tx)
		lb = np.array(lb)
		tx = tx[rd_index]
		lb = lb[rd_index]
		tx = list(tx)
		lb = list(lb)

		# total 1668 -- 1(1168) 0(500)
		x_ds = tf.data.Dataset.from_tensor_slices(tf.Variable(tx))
		y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(lb, tf.int64))
		train_ds = tf.data.Dataset.zip((x_ds,y_ds))
		ensemble_ds.append(train_ds)

	# ensemble_ds: [dataset1, dataset2, dataset3, dataset4, dataset5]
	return ensemble_ds


def get_dev():
	with open("dev.json", 'r') as f:
		temp = json.loads(f.read())
		text, label = [], []
		for key,val in temp.items():
			text.append(val['text'])
			label.append(val['label'])

		x_ds = tf.data.Dataset.from_tensor_slices(tf.Variable(text))
		y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label, tf.int64))
		dev_ds = tf.data.Dataset.zip((x_ds,y_ds))
		
		return dev_ds


def get_dev_for_pred():
	with open("dev.json", 'r') as f:
		temp = json.loads(f.read())
		text = []
		# need in order
		for x in range(len(temp)):
			text.append(temp["dev-{}".format(x)]["text"])
		
		label = [1] * len(text)

		x_ds = tf.data.Dataset.from_tensor_slices(tf.Variable(text))
		y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label, tf.int64))
		dev_pred_ds = tf.data.Dataset.zip((x_ds,y_ds))

		return dev_pred_ds


def get_test():
	with open("test-unlabelled.json", 'r') as f:
		temp = json.loads(f.read())
		text = []
		# need in order
		for x in range(len(temp)):
			text.append(temp["test-{}".format(x)]["text"])
		
		# test dataset doesn't have labels because we need to predict them
		# just counterfeit the labels to fit the dataset form and we'll never use them
		label = [1] * len(text)

		x_ds = tf.data.Dataset.from_tensor_slices(tf.Variable(text))
		y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label, tf.int64))
		test_ds = tf.data.Dataset.zip((x_ds,y_ds))

		return test_ds
		

def count_pos_neg(filename):
	with open(filename, 'r') as f:
		dic = json.loads(f.read())
		one = 0
		zero = 0
		for key,val in dic.items():
			if val['label'] == 1:
				one += 1
			else:
				zero += 1
		print('1:',one, '  0:',zero)


def write_pred(arr, t, filename):

	dic = collections.OrderedDict()
	for i in range(len(arr)):
		dic["{}-{}".format(t, i)] = {"label":arr[i]}

	with open(filename, 'w') as f:
		json.dump(dic, f)



def set_threshold(result, threshold=0.5):
	res = []
	for i in result:
		res.append(1 if i[0] > threshold else 0)
	return res


def train():

	train_ds = get_train()
	dev_ds = get_dev()
	dev_pred_ds = get_dev_for_pred()
	test_ds = get_test()

	# model construction
	embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
	hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(8, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	ensemble_res1 = []
	ensemble_res2 = []
	for i in range(len(train_ds)):
		history = model.fit(train_ds[i].shuffle(100).batch(20), epochs=30)
		
		'''
		# if test dataset has correct given labels, use this evaluation
		results = model.evaluate(train_ds.batch(20), verbose=2)
		for name, value in zip(model.metrics_names, results):
			print("%s: %.3f" % (name, value))
		'''

		result1 = model.predict(test_ds.batch(20))
		ensemble_res1.append(result1)
		result2 = model.predict(dev_pred_ds.batch(20))
		ensemble_res2.append(result2)

	ts = [0.33, 0.5, 0.66]
	for t in ts:
		f_res1 = [0] * len(ensemble_res1[0])
		f_res2 = [0] * len(ensemble_res2[0])

		for rs in ensemble_res1:
			each_model_res = set_threshold(rs, t)
			for i in range(len(f_res1)):
				f_res1[i] += each_model_res[i]
		for j in range(len(f_res1)):
			# if >= 3: at least 3 models(total 5) predict as 1
			f_res1[j] = 1 if f_res1[j] >= 3 else 0

		write_pred(f_res1, "test", "test-output_M3_{}.json".format(t))

		for rs in ensemble_res2:
			each_model_res = set_threshold(rs, t)
			for i in range(len(f_res2)):
				f_res2[i] += each_model_res[i]
		for j in range(len(f_res2)):
			# if >= 3: at least 3 models(total 5) predict as 1
			f_res2[j] = 1 if f_res2[j] >= 3 else 0

		write_pred(f_res2, "dev", "dev-output_M3_{}.json".format(t))
		

train()
#count_pos_neg("test-output.json")

