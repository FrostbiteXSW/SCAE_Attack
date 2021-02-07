import abc
import copy
import csv
import math
import os
import random
from copy import deepcopy
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from absl import logging
from monty.collections import AttrDict
from tqdm import trange


def imshow(im: np.ndarray, file_path=None):
	"""
	Use matplotlib.pyplot to show images.
	For 2-dim image data: [width, height].
	For 3-dim image data: [width, height, channels].
	For 4-dim image dataset: [n_images, width, height, channels].

	:type file_path: str
	:param file_path: Save the image with specific file name.
	:type im: torch.Tensor
	:param im: single image with 2 or 3 dims,
						 or multiple images with 4 dims.
	"""
	if im.ndim == 2:
		plt.imshow(im)
		plt.show()
	elif im.ndim == 3:
		if im.shape[-1] == 1:
			im = im.squeeze(axis=-1)
		plt.imshow(im)
		plt.show()
	elif im.ndim == 4:
		n_x = math.floor(pow(len(im), 0.5))
		if math.ceil(len(im) / n_x) > n_x + 1:
			n_x += 1
			n_y = n_x
		else:
			n_y = math.ceil(len(im) / n_x)
		ims = np.zeros([im.shape[1] * n_y, im.shape[2] * n_x, im.shape[3]])
		for i in range(len(im)):
			x, y = i // n_x, i % n_x
			ims[im.shape[1] * x:im.shape[1] * (x + 1), im.shape[2] * y:im.shape[2] * (y + 1), :] = im[i]
		if ims.shape[-1] == 1:
			ims = ims.squeeze(axis=-1)
		plt.imshow(ims)
		plt.show()
	else:
		raise TypeError('Expected image(s) to have 2, 3 or 4 dims, but got {}.'.format(im.ndim))

	if file_path:
		if im.ndim == 4:
			plt.imsave(file_path, ims, format='png')
		else:
			plt.imsave(file_path, im, format='png')
		print('Image is saved to {}'.format(file_path))


def imresize(im: np.ndarray, shape):
	"""
	:param im: 3-dim image of type np.uint8
	:param shape: 2-dim array
	:return: resized image
	"""
	im = deepcopy(im)
	reshape = im.ndim == 3 and im.shape[-1] == 1

	if reshape:
		im = im.squeeze(axis=-1)

	im = Image.fromarray(im)
	im = im.resize(shape)
	im = np.array(im)
	im = im.clip(0, 255)

	if reshape:
		im = im[:, :, None]

	return im


def imblur(image: np.ndarray, times=1):
	"""
	Blur the image by computing the average of pixels and their neighbourhoods. Padding isn't applied.

	:param image: The shape should be like: [width, height, n_channel].
	:param times: How many times should the image be blurred.
	:return: Blurred image.
	"""

	shape = image.shape

	for _ in range(times):
		new_image = np.zeros(image.shape)
		for x in range(shape[0]):
			for y in range(shape[1]):
				x_low, x_high = max(x - 1, 0), min(x + 2, shape[0])
				y_low, y_high = max(y - 1, 0), min(y + 2, shape[1])
				pixel_sum = image[x_low:x_high, y_low:y_high, :].sum(axis=0).sum(axis=0)
				pixel_cnt = (x_high - x_low) * (y_high - y_low)
				new_image[x, y, :] = pixel_sum / pixel_cnt
		image = new_image

	return image


def to_float32(arr: np.ndarray):
	"""
	:param arr: numpy array of type np.uint8 with max value less than or equal to 255
	:return: numpy array of type np.float32 with max value less than or equal to 1.0
	"""
	arr = arr.astype(np.float32)
	arr = arr / 255
	return arr


def randint(min: int, max: int, list_except: list = None):
	"""
		Same as random.randint(), but not returning values in list_except.
	"""

	# Handle special occasions
	if min == max:
		return min
	if min > max:
		raise ValueError('Min must not be larger than max.')
	if list_except is None:
		return random.randint(min, max)

	# Make a deep copy of list_except then sort it
	list_except = copy.deepcopy(list_except)
	list_except.sort()

	# Remove duplicated values
	i = 0
	while i < len(list_except) - 2:
		if list_except[i] == list_except[i + 1]:
			list_except.remove(list_except[i])
		else:
			i += 1

	# Check validity
	left, right = 0, len(list_except) - 1

	while left < len(list_except) and list_except[left] < min:
		left += 1
	if left == len(list_except):
		return random.randint(min, max)

	while right >= 0 and list_except[right] > max:
		right -= 1
	if right == -1:
		return random.randint(min, max)

	list_except = list_except[left:right + 1]
	if len(list_except) == max - min + 1:
		raise ValueError('All values between min and max are excluded.')

	# Calculate random value
	r = random.randint(min, max)
	while r in list_except:
		r = random.randint(min, max)
	return r


def get_dataset(name, split, shape=None, batch_size=100, file_path=None, save_only=False):
	if not split:
		raise ValueError('Split should not be None.')

	output = AttrDict(
		image=[],
		label=[]
	)

	if file_path:
		file_name = name + '_' + split + '.npz'
		if file_path[-1] != '/' or file_path[-1] != '\\':
			file_name = '/' + file_name
		file_name = file_path + file_name

		if not save_only:
			if os.path.exists(file_name):
				print('Info: Loading dataset file from \"{}\".'.format(file_name))

				if shape is not None:
					print('Warning: Image shape will remain unchanged. If you want to reshape images, set \"save_only\" to True.')

				npzfile = np.load(file_name)
				output['image'] = npzfile['image']
				output['label'] = npzfile['label']
				return output
			else:
				print('Warning: Dataset file \"{}\" not found. Will save a new one.'.format(file_name))

	graph = tf.Graph()
	sess = tf.Session(graph=graph)

	with graph.as_default():
		iter = tfds.load(name=name, split=split).batch(batch_size).make_one_shot_iterator()
		next = iter.get_next()

		try:
			while True:
				data = sess.run(next)
				output['image'].append(data['image'])
				output['label'].append(data['label'])
		except tf.python.framework_ops.errors.OutOfRangeError:
			pass

	# Free all resources. Automatically close session.
	del sess

	output['image'] = np.concatenate(output['image'])
	output['label'] = np.concatenate(output['label'])

	if shape and output['image'].shape[1:3] != tuple(shape):
		new_image = np.zeros([output['image'].shape[0], *shape, output['image'].shape[3]], dtype=np.uint8)
		for i in trange(len(output['image']), desc='Resizing images'):
			new_image[i] = imresize(output['image'][i], shape)
		output['image'] = new_image

	if file_path:
		print('Info: Saving dataset file into \"{}\".'.format(file_name))
		if not os.path.exists(file_path):
			os.makedirs(file_path)
		np.savez_compressed(file_name, image=output['image'], label=output['label'])

	return output


def get_gtsrb(split, shape=None, file_path=None, save_only=False,
              gtsrb_raw_file_path=None, gtsrb_classes=None):
	if not split:
		raise Exception('Split should not be None.')

	output = AttrDict(
		image=[],
		label=[]
	)

	if file_path:
		file_name = 'gtsrb_' + split + '.npz'
		if file_path[-1] != '/' or file_path[-1] != '\\':
			file_name = '/' + file_name
		file_name = file_path + file_name

		if not save_only:
			if os.path.exists(file_name):
				print('Info: Loading dataset file from \"{}\".'.format(file_name))

				if shape is not None:
					print('Warning: Image will remain unchanged. Leave \"file_path\" empty if you want to change them.')

				npzfile = np.load(file_name)
				output['image'] = npzfile['image']
				output['label'] = npzfile['label']
				return output
			else:
				print('Warning: Dataset file \"{}\" not found. Will save a new one.'.format(file_name))

	if gtsrb_raw_file_path is None:
		raise Exception('Argument \"gtsrb_raw_file_path\" must be given.')
	output['image'], output['label'] = _read_traffic_signs(gtsrb_raw_file_path + '/' + split,
	                                                       shape=[28, 28] if shape is None else shape,
	                                                       classes=gtsrb_classes)

	for i in range(len(output['image'])):
		output['image'][i] = (output['image'][i].astype(np.float32) / output['image'][i].max() * 255).astype(np.uint8)

	if file_path:
		print('Info: Saving dataset file into \"{}\".'.format(file_name))
		if not os.path.exists(file_path):
			os.makedirs(file_path)
		np.savez_compressed(file_name, image=output['image'], label=output['label'])

	return output


def _read_traffic_signs(root_path, shape=None, classes=None):
	"""Reads traffic sign data for German Traffic Sign Recognition Benchmark.
		Arguments: path to the traffic sign data, for example './GTSRB/Training'
		Returns:   list of images, list of corresponding labels"""

	if shape is None:
		shape = [224, 224]

	images = []
	labels = []

	# loop over all classes
	for c in range(43):
		prefix = root_path + '/' + format(c, '05d') + '/'  # subdirectory for class
		gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
		gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
		next(gtReader)  # skip header
		# loop over all images in current annotations file
		for row in gtReader:
			# the 1th column is the filename
			img = np.array(plt.imread(prefix + row[0]))
			img = img[int(row[4]):int(row[6]), int(row[3]):int(row[5]), :]
			img = (img * 255).astype('uint8')
			images.append(img[None])
			labels.append(np.array([int(row[7])]))
		gtFile.close()

	if classes is not None:
		sub_images = []
		sub_labels = []
		for i in range(len(labels)):
			if labels[i] in classes:
				sub_images.append(images[i])
				sub_labels.append(np.array([classes.index(labels[i])]))
		images = sub_images
		labels = sub_labels

	new_image = np.zeros([len(images), *shape, 3], dtype=np.uint8)
	for i in trange(len(images), desc='Resizing images'):
		new_image[i] = imresize(images[i][0], shape)
	images = new_image

	labels = np.concatenate(labels)

	return images, labels


def get_samples_by_labels(dataset: AttrDict, labels: list):
	indices = []

	for index in range(len(dataset['label'])):
		if dataset['label'][index] in labels:
			indices.append(index)

	output = AttrDict(
		image=copy.deepcopy(dataset['image'][indices]),
		label=copy.deepcopy(dataset['label'][indices])
	)

	return output


def block_warnings():
	simplefilter(action='ignore', category=FutureWarning)
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.set_verbosity(logging.ERROR)


class ModelCollector(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def run(self, images, to_collect):
		pass

	@abc.abstractmethod
	def __call__(self, images):
		pass
