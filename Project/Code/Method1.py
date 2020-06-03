# -*- coding: utf-8 -*-
# @Author: Puffrora
# @Date:   2020-04-27 14:57:08
# @Last Modified by:   Puffrora
# @Last Modified time: 2020-05-11 12:34:55


print(''' Method 1
============
| Baseline |
============
''')


import numpy as np
import json
import re, collections
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_train():
	with open("balanced.json", 'r') as f:
		temp = json.loads(f.read())
		# cnt = 0
		text, label = [], []
		for key,val in temp.items():
			text.append(val['text'])
			label.append(val['label'])
			'''
			if cnt < 10:
				print(key,val)
				print(val['text'],val['label'],'\n----------------------')
			cnt += 1
			'''

		x_ds = tf.data.Dataset.from_tensor_slices(tf.Variable(text))
		y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(label, tf.int64))
		train_ds = tf.data.Dataset.zip((x_ds,y_ds))

		return train_ds


def get_dev():
	with open("dev.json", 'r') as f:
		temp = json.loads(f.read())
		# cnt = 0
		text, label = [], []
		for key,val in temp.items():
			text.append(val['text'])
			label.append(val['label'])
			'''
			if cnt < 3:
				print(key,val)
				print(val['text'],val['label'],'\n----------------------')
			cnt += 1
			'''

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

	history = model.fit(train_ds.shuffle(100).batch(20), epochs=30)
	
	'''
	# if use dev to improve performance
	history = model.fit(train_ds.shuffle(100).batch(20), epochs=30, 
						validation_data=dev_ds.batch(20), verbose=1)

	# if test dataset has correct given labels, use this evaluation
	results = model.evaluate(train_ds.batch(20), verbose=2)
	for name, value in zip(model.metrics_names, results):
		print("%s: %.3f" % (name, value))
	'''

	result1 = model.predict(test_ds.batch(20))
	result2 = model.predict(dev_pred_ds.batch(20))
	
	ts = [0.33, 0.5, 0.66]
	for t in ts:
		f_res1 = set_threshold(result1, t)
		f_res2 = set_threshold(result2, t)
		write_pred(f_res1, "test", "test-output_M1_{}.json".format(t))
		write_pred(f_res2, "dev", "dev-output_M1_{}.json".format(t))


train()
#count_pos_neg("test_output.json")


# example to calssify imdb comments
'''
def example1():
	print("Version: ", tf.__version__)
	print("Eager mode: ", tf.executing_eagerly())
	print("Hub version: ", hub.__version__)
	print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

	train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])


	(train_data, validation_data, test_data), metadata = tfds.load(
			'imdb_reviews',
			split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
			with_info=True,
			as_supervised=True,)

	train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
	test_examples_batch, test_labels_batch = next(iter(test_data.batch(10)))

	"""
	针对此示例我们将使用 TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1 
	的一种预训练文本嵌入（text embedding）模型 。

	为了达到本教程的目的还有其他三种预训练模型可供测试：

	google/tf2-preview/gnews-swivel-20dim-with-oov/1 ——类似 google/tf2-preview/gnews-swivel-20dim/1，
	但 2.5%的词汇转换为未登录词桶（OOV buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
	google/tf2-preview/nnlm-en-dim50/1 ——一个拥有约 1M 词汇量且维度为 50 的更大的模型。
	google/tf2-preview/nnlm-en-dim128/1 ——拥有约 1M 词汇量且维度为128的更大的模型。
	"""

	embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
	hub_layer = hub.KerasLayer(embedding, input_shape=[], 
								dtype=tf.string, trainable=True)
	hub_layer(train_examples_batch[:3])

	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	model.summary()


	model.compile(optimizer='adam',
					loss='binary_crossentropy',
					metrics=['accuracy'])

	history = model.fit(train_data.shuffle(10000).batch(512),
						epochs=20,
						validation_data=validation_data.batch(512),
						verbose=1)
	results = model.evaluate(test_data.batch(512), verbose=2)
	for name, value in zip(model.metrics_names, results):
		print("%s: %.3f" % (name, value))
'''


# example to construct your own dataset and train
'''
def example2():
	import tensorflow as tf
	import tensorflow_hub as hub
	import tensorflow.keras as keras
	import os
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	train_sentences = tf.Variable(["cat is mine","dog is mine","whale are there","tiger is her"])
	train_labels = [1,1,1,1,1]
	trainx_ds = tf.data.Dataset.from_tensor_slices(train_sentences)
	trainy_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_labels,tf.int64))
	train_ds = tf.data.Dataset.zip((trainx_ds,trainy_ds))
	train_examples_batch, train_labels_batch = next(iter(train_ds.batch(3)))

	dev_sentences = tf.Variable(["cat was this","lemurs were here","monkey is that","pig is fine","lion was good"])
	dev_labels = [1,1,1,1,1]
	devx_ds = tf.data.Dataset.from_tensor_slices(dev_sentences)
	devy_ds = tf.data.Dataset.from_tensor_slices(tf.cast(dev_labels,tf.int64))
	dev_ds = tf.data.Dataset.zip((devx_ds,devy_ds))

	test_sentences = tf.Variable(["grass finds it","tree supports sky","cat was mine","flower hits me","dog be here","iris look beautiful","lotus smells good","daisy swags in the air"])
	test_labels = [0,0,1,0,1,0,0,0]
	testx_ds = tf.data.Dataset.from_tensor_slices(test_sentences)
	testy_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_labels,tf.int64))
	test_ds = tf.data.Dataset.zip((testx_ds,testy_ds))

	embedding = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
	hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
	print(hub_layer(train_examples_batch[:1]))

	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(8, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	history = model.fit(train_ds.shuffle(2).batch(2), epochs=30,
						validation_data=dev_ds.batch(2), verbose=1)

	results = model.evaluate(test_ds.batch(2), verbose=2)
	for name, value in zip(model.metrics_names, results):
		print("%s: %.3f" % (name, value))

	result = model.predict(test_ds.batch(2))
	print(result)
'''
