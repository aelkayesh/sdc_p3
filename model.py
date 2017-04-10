import csv
import cv2
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.utils import shuffle as shuffle
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Cropping2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import load_model
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import model_from_json
import json
import matplotlib.pyplot as plt
import sys

#used to show histogram of angles
def show_histogram(steering_angles):
	plt.hist(steering_angles)
	plt.title("Steering angles Histogram")
	plt.xlabel("Angle")
	plt.ylabel("Frequency")
	plt.show()

#randomly change brightness of images
def change_brightness(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	random_brightness = .1 + np.random.uniform()
	img[:,:,2] = img[:,:,2] * random_brightness
	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	return img

#shift image,angle to the right/left to expose model to wide set of angles 
def shift_image(image, steer, trans_range, scale=1, y=False):
	# Randomly generate a translation
	tr_x = trans_range * np.random.uniform() - trans_range / 2

	# Augment the steering angle
	steer_ang = steer + tr_x / trans_range * 2 * .2 * scale

	if y:
		# Also translate on the y axis
		tr_y = 40 * np.random.uniform() - 40 / 2
	else:
		tr_y = 0

	# Warp the image based on the translation
	trans_matrix = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
	image_tr = cv2.warpAffine(image, trans_matrix, (64, 128))
	
	return image_tr, steer_ang
	
def simple_model():

	img_height = 64
	img_width = 128
	crop_values = [38,13, 0 ,0]

	# Initialize the model and crop image
	model = Sequential()
	model.add(Cropping2D(cropping=((crop_values[0],crop_values[1]), (crop_values[2], crop_values[3])) , input_shape=(img_width, img_height, 3)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	model.add(Convolution2D(6, 5, 5, activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(12, 5, 5, activation = 'relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(120, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(84, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model
	
def generator(lines, batch_size):
	
	#center, left, right
	correction_rate = [0.0, 0.3, -0.3]	
	num_samples = len(lines)
	
	shuffle(lines)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			images = []
			measurements = []
			batch = lines[offset:offset+batch_size]
			for line in batch:
				measurement = float(line[3])
				for i in range(3):
					filename = line[i]
					if(len(filename.strip()) > 0):
						img = cv2.imread('/input/'+filename.strip())
						if(img is not None):
							measurement = measurement + correction_rate[i]
							img = change_brightness(img)
							img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
							img = cv2.resize(img,(64, 128))
							images.append(img)						
							measurements.append(measurement)
							img_flipped = np.fliplr(img)
							images.append(img_flipped)  	  		
							measurements.append(- 1* measurement)		#flipped measurement

			X_train = np.array(images)
			y_train = np.array(measurements)
			yield shuffle(X_train, y_train)	
						
def train_and_save():	

	global ignore_angle
	train_folders = ['/input/r5/']
	lines = []
	n_epoch=15

	for train_folder in train_folders:
		with open(train_folder +'driving_log.csv') as csvfile:
			reader = csv.reader(csvfile)
			#next(reader, None)  # skip the headers
			for line in reader:
				angle = float(line[3])
				if angle < 0.02 and angle > -0.165:
					if np.random.random() > 0.67:
						lines.append(line)
				else:
					lines.append(line)
				
					
	train_samples, valid_samples = train_test_split(lines, test_size=0.2, random_state=42)
			
	# compile and train the model using the generator function
	train_generator = generator(train_samples, batch_size=32)
	validation_generator = generator(valid_samples, batch_size=32)

	model = simple_model()	
	model.compile(optimizer='adam', loss='mse')
	
	#create callback
	checkpointer = ModelCheckpoint(filepath="/output/weights.hdf5", verbose=1, save_best_only=True)
	
	print('Training...')
	model.fit_generator(train_generator, samples_per_epoch= len(train_samples) *6, validation_data=validation_generator, nb_val_samples=len(valid_samples)*6, nb_epoch=n_epoch,verbose=1, callbacks=[checkpointer])
	
	
	# save the model
	model.load_weights('/output/weights.hdf5')
	with open('/output/model.json', 'w') as fd:
		json.dump(model.to_json(), fd)
	model.save_weights('/output/model.h5')

train_and_save()
