import tensorflow as tf
import numpy as np

CLASS_NAMES = np.array(["0","1","2","3","4","5","6","7","8","9"])

def get_dataset(data_path,data_set_size=42000,split_val=0.1):
	dataset = tf.data.Dataset.list_files(str(f'{data_path}/*'))
	dataset = dataset.shuffle(buffer_size=1000)
	dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	val_size = int(data_set_size*split_val)
	val_ds = dataset.take(val_size)
	train_ds = dataset.skip(val_size)
	return train_ds,val_ds

def process_path(file_path):
	label = get_label(file_path)
	# load the raw data from the file as a string
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img,label

def get_label(file_path):
	# convert the path to a list of path components
	file_name = tf.strings.split(file_path, '/')[-1]
	label = tf.strings.split(file_name, "_")[0]
	# The second to last is the class-directory
	return label == CLASS_NAMES

def decode_img(img):
	# convert the compressed string to a 2D uint8 tensor
	img = tf.image.decode_jpeg(img, channels=1)
	# Use `convert_image_dtype` to convert to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
	# resize the image to the desired size.
	return tf.image.resize(img, [32, 32])