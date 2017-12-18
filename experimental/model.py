import numpy as np
import tensorflow as tf
#import cv2
import matplotlib.pyplot as plt
from PIL import Image
import csv
import math
import os
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, ELU, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras import backend as K

num_epochs = 30
batch_size = 64

def preporcess_img(img, state):
	#img = cv2.resize(img, (64, 48), interpolation = cv2.INTER_CUBIC)
	#img.thumbnail((128,96), Image.ANTIALIAS)
	
	return img, state


# generate training/validation batch
def get_batch(X, batch_size = 64):
	# randomly pickup training data to create a batch
	while(True):
		X_batch = []
		y_batch = []
		
		picked = []
		n_imgs = 0
		
		# randomly selected batch size images and light state
		while n_imgs < batch_size:
			i = np.random.randint(0, len(X))
			if (i in picked):
				continue  # skip if this image has been picked
			y_state  = int(X[i][1])
			
		
			picked.append(i)
			img_path = './images/' + X[i][0].strip()
			light_img = plt.imread(img_path)
			light_img = Image.open(img_path)
			light_img = image.load_img(img_path, target_size=(96, 128))
			img_array = image.img_to_array(light_img)
			#light_img = cv2.imread(img_path)
			#img_array = cv2.resize(light_img, (128, 96), interpolation = cv2.INTER_CUBIC)
			
			# preprocess image
			#light_img, y_state = preporcess_img(light_img, y_state)
			
			X_batch.append(img_array)
			st_v = [0, 0, 0]
			st_v[y_state] = 1
			y_batch.append(st_v)
			n_imgs += 1
		
		yield np.array(X_batch), np.array(y_batch)
		
		
def get_samples_per_epoch(num_samples, batch_size):
	# return samples per epoch that is multiple of batch_size
	return math.ceil(num_samples/batch_size)
	

def get_model():
	
	model = Sequential()
	
	# normalization layer
	model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(96, 128, 3)))
	
	# convolution 2D with filter 5x5
	#model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	
	#model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Dropout(0.4))
	
	#model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
	model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Dropout(0.25))
	
	#model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Dropout(0.25))
	
	#model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
	model.add(ELU())
	
	model.add(Flatten())
	
	model.add(Dense(1164))
	model.add(ELU())
	model.add(Dropout(0.5))
	
	model.add(Dense(100))
	model.add(ELU())
	
	model.add(Dense(50))
	model.add(ELU())
	
	model.add(Dense(10))
	model.add(ELU())
	
	model.add(Dense(3))
	model.add(Activation('softmax'))
	
	return model

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
	"""
	Freezes the state of a session into a prunned computation graph.

	Creates a new computation graph where variable nodes are replaced by
	constants taking their current value in the session. The new graph will be
	prunned so subgraphs that are not neccesary to compute the requested
	outputs are removed.
	@param session The TensorFlow session to be frozen.
	@param keep_var_names A list of variable names that should not be frozen,
			or None to freeze all the variables in the graph.
	@param output_names Names of the relevant graph outputs.
	@param clear_devices Remove the device directives from the graph for better portability.
	@return The frozen graph definition.
	"""
	
	from tensorflow.python.framework.graph_util import convert_variables_to_constants
	graph = session.graph
	with graph.as_default():
		freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
		output_names = output_names or []
		output_names += [v.op.name for v in tf.global_variables()]
		input_graph_def = graph.as_graph_def()
		if clear_devices:
			for node in input_graph_def.node:
				node.device = ""
		frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
		return frozen_graph


if __name__ == "__main__":
	driving_data = []
	# create a list of image paths and angles
	with open('traffic_light_data.csv') as drvfile:
		reader = csv.DictReader(drvfile)
		for row in reader:
			driving_data.append((row['images'], row['state']))

	driving_data = shuffle(driving_data)
	# split the data, 20% for validation
	X_train, X_validation = train_test_split(driving_data, test_size = 0.2, random_state = 7898)

	train_generator = get_batch(X_train)
	val_generator   = get_batch(X_validation)

	model = get_model()
	model.compile(optimizer = Adam(lr = 0.0001), loss='mse', metrics=['accuracy'])

	print("Start training...")
	h = model.fit_generator( train_generator,
                     	steps_per_epoch = get_samples_per_epoch(len(X_train), batch_size),
                     	epochs = num_epochs,
                     	validation_data = val_generator,
                     	validation_steps = get_samples_per_epoch(len(X_validation), batch_size))

	#print ("fit history: ", h.history.keys())

	# save model and weights
	model_json = model.to_json()
	with open("./model.json", "w") as json_file:
		json.dump(model_json, json_file)

	model.save_weights("./model.h5")
	print("Saved model to disk")

	model.save("./k_model.h5")
	
	#frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
	#tf.train.write_graph(frozen_graph, "./", "traffic_light_frozen.pb", as_text=False)
	#tf.train.write_graph(K.get_session().graph.as_graph_def(), "./","model_graph.ascii", as_text=True)



