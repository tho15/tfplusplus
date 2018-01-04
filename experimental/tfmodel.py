import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import math
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
#import cv2

EPOCHES = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# generate training/validation batch
def get_batch(X, offset, batch_size = 64):
	
	X_batch = []
	y_batch = []
		
	picked = []
	n_imgs = 0
	
	# randomly selected batch size images and light state
	#i = offset
	while n_imgs < batch_size:
		i = np.random.randint(0, len(X))
		if (i in picked):
			continue  # skip if this image has been picked
		y_state  = int(X[i][1])
			
		img_path = './images/' + X[i][0].strip()
		light_img = plt.imread(img_path)
		light_img = Image.open(img_path)
		light_img = image.load_img(img_path, target_size=(96, 128))
		img_array = image.img_to_array(light_img)
		#img_array = img_array/127.5-1.0
		img_array -= np.mean(img_array, axis=0)
		img_array /= np.std(img_array, axis=0)
		#light_img = cv2.imread(img_path)
		#img_array = cv2.resize(light_img, (128, 96), interpolation = cv2.INTER_CUBIC)
			
		# preprocess image
		#light_img, y_state = preporcess_img(light_img, y_state)
			
		X_batch.append(img_array)
		st_v = [0.0, 0.0, 0.0]
		st_v[y_state] = 1.0
		y_batch.append(st_v)
		
		n_imgs += 1
		#i += 1
		
	return np.array(X_batch), np.array(y_batch)

with tf.name_scope('data'):
	x_ = tf.placeholder(tf.float32, (None, 96, 128, 3), name='x')
	y_ = tf.placeholder(tf.int32, (None), name='y')
	#one_hot_y = tf.one_hot(y_, 3)

#define model here:

with tf.variable_scope('conv1') as scope:
	w1 = tf.get_variable('w1', [5, 5, 3, 24], initializer=tf.contrib.layers.xavier_initializer())
	#b1 = tf.get_variable('b1', [24], initializer=tf.random_uniform_initializer(maxval=0.01))
	b1 = tf.Variable(tf.zeros(24), 'b1')
	conv1 = tf.nn.conv2d(x_, w1, strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.bias_add(conv1, b1)
	# relu activation
	conv1 = tf.nn.elu(conv1)

with tf.variable_scope('pool1') as scope:
	# max pooling layer
	pool1 = tf.nn.max_pool(conv1, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')

with tf.variable_scope('conv2') as scope:
	w2 = tf.get_variable('w2', [5, 5, 24, 36], initializer=tf.contrib.layers.xavier_initializer())
	#b2 = tf.get_variable('b2', [36], initializer=tf.random_uniform_initializer(0.01))
	b2 = tf.Variable(tf.zeros(36), 'b2')
	conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.bias_add(conv2, b2)
	# relu activation
	conv2 = tf.nn.elu(conv2)

with tf.variable_scope('pool2') as scope:
	# max pooling layer
	pool2 = tf.nn.max_pool(conv2, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')
	pool2 = tf.nn.dropout(pool2, 0.4)

with tf.variable_scope('conv3') as scope:
	w3 = tf.get_variable('w3', [5, 5, 36, 48], initializer=tf.contrib.layers.xavier_initializer())
	#b3 = tf.get_variable('b3', [48], initializer=tf.random_uniform_initializer(0.01))
	b3 = tf.Variable(tf.zeros(48), 'b3')
	conv3 = tf.nn.conv2d(pool2, w3, strides=[1, 2, 2, 1], padding='SAME')
	conv3 = tf.nn.bias_add(conv3, b3)
	# relu activation
	conv3 = tf.nn.elu(conv3)

with tf.variable_scope('pool3') as scope:
	# max pooling layer
	pool3 = tf.nn.max_pool(conv3, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')
	pool3 = tf.nn.dropout(pool3, 0.25)

with tf.variable_scope('conv4') as scope:
	w4 = tf.get_variable('w4', [3, 3, 48, 64], initializer=tf.contrib.layers.xavier_initializer())
	#b4 = tf.get_variable('b4', [64], initializer=tf.random_uniform_initializer(0.01))
	b4 = tf.Variable(tf.zeros(64), 'b4')
	conv4 = tf.nn.conv2d(pool3, w4, strides=[1, 1, 1, 1], padding='SAME')
	conv4 = tf.nn.bias_add(conv4, b4)
	# relu activation
	conv4 = tf.nn.elu(conv4)

with tf.variable_scope('pool4') as scope:
	# max pooling layer
	pool4 = tf.nn.max_pool(conv4, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')
	pool4 = tf.nn.dropout(pool4, 0.25)

with tf.variable_scope('conv5') as scope:
	w5 = tf.get_variable('w5', [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
	#b5 = tf.get_variable('b5', [64], initializer=tf.random_uniform_initializer(0.01))
	b5 = tf.Variable(tf.zeros(64), 'b5')
	conv5 = tf.nn.conv2d(pool4, w5, strides=[1, 1, 1, 1], padding='SAME')
	conv5 = tf.nn.bias_add(conv5, b5)
	# relu activation
	conv5 = tf.nn.elu(conv5)

with tf.variable_scope('fc1') as scope:
	fsize = 12*16*64
	w6 = tf.get_variable('w6', [fsize, 1164], initializer=tf.contrib.layers.xavier_initializer())
	#b6 = tf.get_variable('b6', 1164, initializer=tf.random_uniform_initializer(0.01))
	b6 = tf.Variable(tf.zeros(1164), 'b6')

	conv5 = tf.reshape(conv5, [-1, fsize])
	fc1 = tf.nn.elu(tf.matmul(conv5, w6)+b6)
	fc1 = tf.nn.dropout(fc1, 0.5)

with tf.variable_scope('fc2') as scope:
	w7 = tf.get_variable('w7', [1164, 100], initializer=tf.contrib.layers.xavier_initializer())
	#b7 = tf.get_variable('b7', [100], initializer=tf.random_uniform_initializer(0.01))
	b7 = tf.Variable(tf.zeros(100), 'b7')
	fc2 = tf.nn.elu(tf.matmul(fc1, w7)+b7)

with tf.variable_scope('fc3') as scope:
	w8 = tf.get_variable('w8', [100, 50], initializer=tf.contrib.layers.xavier_initializer())
	#b8 = tf.get_variable('b8', [50], initializer=tf.random_uniform_initializer(0.01))
	b8 = tf.Variable(tf.zeros(50), 'b8')
	fc3 = tf.nn.elu(tf.matmul(fc2, w8)+b8)

with tf.variable_scope('fc4') as scope:
	w9 = tf.get_variable('w9', [50, 10], initializer=tf.contrib.layers.xavier_initializer())
	#b9 = tf.get_variable('b9', [10], initializer=tf.random_uniform_initializer(0.01))
	b9 = tf.Variable(tf.zeros(10), 'b9')
	fc4 = tf.nn.elu(tf.matmul(fc3, w9)+b9)

with tf.variable_scope('logits') as scope:
	w = tf.get_variable('w', [10, 3], initializer=tf.contrib.layers.xavier_initializer())
	#b = tf.get_variable('b', [3], initializer=tf.random_uniform_initializer(0.01))
	b = tf.Variable(tf.zeros(3), 'b')
	logits = tf.matmul(fc4, w)+b
	softmax_ = tf.nn.softmax(logits)

# define loss
with tf.variable_scope('loss'):
	#entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
	#loss = tf.reduce_mean(entropy, name='loss')
	mse = tf.losses.mean_squared_error(y_, softmax_)
	loss = tf.reduce_mean(mse, name='loss')

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# define operation to initialize all variables
init_all_vars_op = tf.variables_initializer(tf.global_variables(), name='var_init')

# read training/validation data
driving_data = []
# create a list of image paths and angles
with open('traffic_light_data.csv') as drvfile:
	reader = csv.DictReader(drvfile)
	for row in reader:
		driving_data.append((row['images'], row['state']))

driving_data = shuffle(driving_data)
# split the data, 20% for validation
X_train, X_validation = train_test_split(driving_data, test_size = 0.2, random_state = 7898)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()

def evaluate(XY_data):
	num_v = int(len(XY_data)/BATCH_SIZE)*BATCH_SIZE
	total_accuracy = 0
	sess = tf.get_default_session()
	for offset in range(0, num_v, BATCH_SIZE):
		x_vb, y_vb = get_batch(XY_data, offset, BATCH_SIZE)
		accuracy = sess.run(accuracy_op, feed_dict = {x_: x_vb, y_: y_vb})
		total_accuracy += (accuracy*len(x_vb))

	return total_accuracy/num_v

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	writer = tf.summary.FileWriter('./graphs/tfconvet', sess.graph)
	tf.train.write_graph(sess.graph_def, './', 'tlc_model.pb', as_text=False)
	"""
	num_xtrain = int(len(X_train)/BATCH_SIZE)*BATCH_SIZE
	print('num of train images is: ', num_xtrain)

	for i in range(EPOCHES):
		X_train = shuffle(X_train)
		for offset in range(0, num_xtrain, BATCH_SIZE):
			x_b, y_b = get_batch(X_train, offset, BATCH_SIZE)
			_, cost, logits_batch, softmax_batch, fc4v, w_b = sess.run([optimizer, loss, logits, softmax_, conv3, w4],
											 feed_dict = {x_: x_b, y_: y_b})
			#print("fc1: ", fc4v)
			#print("w: ", w_b)
			#print("logits: ", logits_batch)
			#print("softmax: ", softmax_batch)
			print("cost is = {:.3f}".format(cost))

		print("EPOCH {} ...", format(i+1))
		#ww = sess.run(w1, {}, {})
		#print("epoch w: ", ww)
		validation_accuracy = evaluate(X_validation)
		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
		print()

	saver.save(sess, './tldnet')
	print("Model saved!")
	"""






















