import copy
import csv
import os
import random
from copy import deepcopy
from warnings import simplefilter

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time
from PIL import Image
from absl import logging
from monty.collections import AttrDict
from tqdm import trange


def _average_filter(img: np.ndarray, times=1, radius=1):
	"""
	Blur the image by computing the average of pixels and their neighbourhoods. Padding isn't applied.

	:param img: The shape should be like one of these: [width, height, n_channel] or [n_images, width, height, n_channel].
	:param times: How many times should the image be blurred.
	:param radius: Blur range per pixel.
	:return: Blurred image(s).
	"""

	shape = img.shape
	width = shape[-3]
	height = shape[-2]

	assert len(shape) in [3, 4] and times >= 1 and radius >= 1

	for _ in range(times):
		new_image = np.zeros(shape)
		for x in range(width):
			for y in range(height):
				x_low, x_high = max(x - radius, 0), min(x + radius + 1, width)
				y_low, y_high = max(y - radius, 0), min(y + radius + 1, height)
				pixel_cnt = (x_high - x_low) * (y_high - y_low)

				if len(shape) == 3:
					new_image[x, y, :] = img[x_low:x_high, y_low:y_high, :].sum(axis=(0, 1)) / pixel_cnt
				else:
					new_image[:, x, y, :] = img[:, x_low:x_high, y_low:y_high, :].sum(axis=(1, 2)) / pixel_cnt

		img = new_image

	return img


def _bilateral_filter(img: np.ndarray, times=1, radius=1, gsigma=1.5, ssigma=1.5):
	"""
	Bilateral Blur image(s) without padding. Image(s) should be in range [0, 1] with type as np.float32.

	:param img: The shape should be like one of these: [width, height, n_channel] or [n_images, width, height, n_channel].
	:param times: How many times should the image be blurred.
	:param radius: Blur range per pixel.
	:param gsigma: Hyper parameter for computing gaussian kernel.
	:param ssigma: Hyper parameter for computing spatial kernel.
	:return: Blurred image.
	"""

	shape = img.shape
	width = shape[-3]
	height = shape[-2]
	n_channels = shape[-1]

	assert len(shape) in [3, 4] and times >= 1 and radius >= 1

	weight_matrix = np.zeros([radius * 2 + 1, radius * 2 + 1, 1])
	for x in range(radius + 1):
		for y in range(x, radius + 1):
			val = np.exp(-(x ** 2 + y ** 2) / (2 * (gsigma ** 2)))
			weight_matrix[radius - x:radius + x + 1:max(2 * x, 1), radius - y:radius + y + 1:max(2 * y, 1)] = val
			weight_matrix[radius - y:radius + y + 1:max(2 * y, 1), radius - x:radius + x + 1:max(2 * x, 1)] = val

	ssigma2 = 2 * (ssigma ** 2)

	for _ in range(times):
		new_image = np.zeros_like(img)
		for x in range(width):
			for y in range(height):
				x_low, x_high = max(x - radius, 0), min(x + radius + 1, width)
				y_low, y_high = max(y - radius, 0), min(y + radius + 1, height)

				if x_high - x_low < radius * 2 + 1 or y_high - y_low < radius * 2 + 1:
					wm = weight_matrix[x_low - x + radius:radius - x + x_high, y_low - y + radius:radius - y + y_high]
				else:
					wm = weight_matrix

				if len(shape) == 3:
					kernel = np.zeros([*wm.shape[:2], n_channels])
					for j in range(kernel.shape[0]):
						for k in range(kernel.shape[1]):
							kernel[j, k] = np.exp(-(img[x_low + j, y_low + k] - img[x, y]) ** 2 / ssigma2)
					kernel *= wm
					kernel /= kernel.sum(axis=(0, 1), keepdims=True)
					new_image[x, y, :] = (img[x_low:x_high, y_low:y_high] * kernel).sum(axis=(0, 1))
				else:
					kernel = np.zeros([shape[0], *wm.shape[:2], n_channels])
					for j in range(kernel.shape[1]):
						for k in range(kernel.shape[2]):
							kernel[:, j, k] = np.exp(-(img[:, x_low + j, y_low + k] - img[:, x, y]) ** 2 / ssigma2)
					kernel *= wm
					kernel /= kernel.sum(axis=(1, 2), keepdims=True)
					new_image[:, x, y, :] = (img[:, x_low:x_high, y_low:y_high] * kernel).sum(axis=(1, 2))

		img = new_image

	return img


def _gaussian_filter(img: np.ndarray, times=1, radius=1, gsigma=1.5):
	"""
	Gaussian Blur image(s) without padding. Image(s) should be in range [0, 1] with type as np.float32.

	:param img: The shape should be like one of these: [width, height, n_channel] or [n_images, width, height, n_channel].
	:param times: How many times should the image be blurred.
	:param radius: Blur range per pixel.
	:param gsigma: Hyper parameter for computing gaussian kernel.
	:return: Blurred image.
	"""

	shape = img.shape
	width = shape[-3]
	height = shape[-2]

	assert len(shape) in [3, 4] and times >= 1 and radius >= 1

	weight_matrix = np.zeros([radius * 2 + 1, radius * 2 + 1, 1])
	for x in range(radius + 1):
		for y in range(x, radius + 1):
			val = np.exp(-(x ** 2 + y ** 2) / (2 * (gsigma ** 2)))
			weight_matrix[radius - x:radius + x + 1:max(2 * x, 1), radius - y:radius + y + 1:max(2 * y, 1)] = val
			weight_matrix[radius - y:radius + y + 1:max(2 * y, 1), radius - x:radius + x + 1:max(2 * x, 1)] = val
	weight_matrix /= weight_matrix.sum()

	for _ in range(times):
		new_image = np.zeros_like(img)
		for x in range(width):
			for y in range(height):
				x_low, x_high = max(x - radius, 0), min(x + radius + 1, width)
				y_low, y_high = max(y - radius, 0), min(y + radius + 1, height)

				if x_high - x_low < radius * 2 + 1 or y_high - y_low < radius * 2 + 1:
					kernel = deepcopy(
						weight_matrix[x_low - x + radius:radius - x + x_high, y_low - y + radius:radius - y + y_high])
					kernel /= kernel.sum()
				else:
					kernel = weight_matrix

				if len(shape) == 3:
					new_image[x, y, :] = (img[x_low:x_high, y_low:y_high] * kernel).sum(axis=(0, 1))
				else:
					new_image[:, x, y, :] = (img[:, x_low:x_high, y_low:y_high] * kernel).sum(axis=(1, 2))

		img = new_image

	return img


def _median_filter(img: np.ndarray, times=1, radius=1):
	"""
	Blur the image by computing the median of pixels and their neighbourhoods. Padding isn't applied.

	:param img: The shape should be like one of these: [width, height, n_channel] or [n_images, width, height, n_channel].
	:param times: How many times should the image be blurred.
	:param radius: Blur range per pixel.
	:return: Blurred image(s).
	"""

	shape = img.shape
	width = shape[-3]
	height = shape[-2]

	assert len(shape) in [3, 4] and times >= 1 and radius >= 1

	for _ in range(times):
		new_image = np.zeros(shape)
		for x in range(width):
			for y in range(height):
				x_low, x_high = max(x - radius, 0), min(x + radius + 1, width)
				y_low, y_high = max(y - radius, 0), min(y + radius + 1, height)

				if len(shape) == 3:
					new_image[x, y, :] = np.median(img[x_low:x_high, y_low:y_high, :], axis=(0, 1))
				else:
					new_image[:, x, y, :] = np.median(img[:, x_low:x_high, y_low:y_high, :], axis=(1, 2))

		img = new_image

	return img


def block_warnings():
	simplefilter(action='ignore', category=FutureWarning)
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	tf.logging.set_verbosity(tf.logging.ERROR)
	logging.set_verbosity(logging.ERROR)


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


def imblur(img, type='a', **kwargs):
	if type == 'a':
		return _average_filter(img, **kwargs)
	if type == 'b':
		return _bilateral_filter(img, **kwargs)
	if type == 'g':
		return _gaussian_filter(img, **kwargs)
	if type == 'm':
		return _median_filter(img, **kwargs)
	raise ValueError('Unsupported algorithm type: {}.'.format(type))


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


def imshow(im: np.ndarray, canvas_size=4.8, file_path=None):
	"""
	Use matplotlib.pyplot to show images.
	For 2-dim image data: [width, height].
	For 3-dim image data: [width, height, channels].
	For 4-dim image dataset: [n_images, width, height, channels].

	:type im: torch.Tensor
	:param im: single image with 2 or 3 dims,
						 or multiple images with 4 dims.
	:type canvas_size: float
	:param canvas_size: Canvas size (for both width and height) to output a 28*28 image.
										Will auto scale for different image shapes.
	:type file_path: str
	:param file_path: Save the image with specific file name.
	"""
	plt.axis('off')
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

	if im.ndim == 3:
		if im.shape[-1] == 1:
			im = im.squeeze(axis=-1)
	elif im.ndim == 4:
		n_x = math.floor(pow(len(im), 0.5))
		if math.ceil(len(im) / n_x) > n_x + 1:
			n_x += 1
			n_y = n_x
		else:
			n_y = math.ceil(len(im) / n_x)
		ims = np.zeros([im.shape[1] * n_y, im.shape[2] * n_x, im.shape[3]], dtype=im.dtype)
		for i in range(len(im)):
			x, y = i // n_x, i % n_x
			ims[im.shape[1] * x:im.shape[1] * (x + 1), im.shape[2] * y:im.shape[2] * (y + 1), :] = im[i]
		if ims.shape[-1] == 1:
			ims = ims.squeeze(axis=-1)
		im = ims
	elif im.ndim != 2:
		raise TypeError('Expected image(s) to have 2, 3 or 4 dims, but got {}.'.format(im.ndim))

	plt.gcf().set_size_inches(canvas_size / 28 * im.shape[1], canvas_size / 28 * im.shape[0])
	plt.imshow(im)
	plt.show()

	if file_path:
		plt.imsave(file_path, im, format='png')
		print('Image is saved to {}'.format(file_path))


def load_npz(file_name: str):
	npz = np.lib.npyio.NpzFile(file_name)
	res = AttrDict()
	for i in npz.keys():
		res[i] = npz[i]
	npz.close()
	return res


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


class DatasetHelper:
	def __init__(self, name, split, shape=None,
	             batch_size=100, shuffle=False, fill_batch=False,
	             file_path=None, save_after_load=False,
	             gray_scale=False, normalize=False,
	             gtsrb_raw_file_path=None, gtsrb_classes=None):
		if not split:
			raise Exception('Split should not be None.')

		self._batch_size = batch_size
		self._shuffle = shuffle
		self._fill_batch = fill_batch
		self._iter_ready = False

		self._trans = [self._to_float32]
		if gray_scale:
			self._trans.append(self._to_gray)
		if normalize:
			self._trans.append(self._normalize)

		file_name = None
		if file_path:
			file_name = name + '_' + split + '.npz'
			if file_path[-1] != '/' or file_path[-1] != '\\':
				file_name = '/' + file_name
			file_name = file_path + file_name

		if file_path and not save_after_load:
			file_name = name + '_' + split + '.npz'
			if file_path[-1] != '/' or file_path[-1] != '\\':
				file_name = '/' + file_name
			file_name = file_path + file_name

			if os.path.exists(file_name):
				print('Info: Loading dataset file from \"{}\".'.format(file_name))
				if shape is not None:
					print('Warning: Image will remain unchanged. Leave \"file_path\" empty if you want to change them.')
				self._images, self._labels = load_npz(file_name).values()
			else:
				raise FileNotFoundError('Requested dataset is not found under \"{}\".'.format(file_path))
		else:
			if name == 'gtsrb':
				if gtsrb_raw_file_path is None:
					raise Exception('Argument \"gtsrb_raw_file_path\" must be given.')
				self._images, self._labels = self._read_traffic_signs(gtsrb_raw_file_path + '/' + split,
				                                                      shape=[28, 28] if shape is None else shape,
				                                                      classes=gtsrb_classes)
			else:
				self._images = []
				self._labels = []

				graph = tf.Graph()
				sess = tf.Session(graph=graph)

				with graph.as_default():
					iter = tfds.load(name=name, split=split).batch(batch_size).make_one_shot_iterator()
					next = iter.get_next()

					try:
						while True:
							data = sess.run(next)
							self._images.append(data['image'])
							self._labels.append(data['label'])
					except tf.python.framework_ops.errors.OutOfRangeError:
						pass

				# Free all resources. Automatically close session.
				del sess

				self._images = np.concatenate(self._images)
				self._labels = np.concatenate(self._labels)

				if shape and self._images.shape[1:3] != tuple(shape):
					new_image = np.zeros([self._images.shape[0], *shape, self._images.shape[3]], dtype=np.uint8)
					for i in trange(len(self._images), desc='Resizing images'):
						new_image[i] = self._resize(self._images[i], shape)
					self._images = new_image

			if save_after_load:
				if file_path is None:
					print('Warning: Dataset will not be saved as \"file_path\" is not provided.')
				else:
					print('Info: Saving dataset file into \"{}\".'.format(file_name))
					if not os.path.exists(file_path):
						os.makedirs(file_path)
					np.savez_compressed(file_name, images=self._images, labels=self._labels)

		self._len = len(self._images)
		self.image_shape = self._images.shape
		self.dataset_size = self.image_shape[0]

	def __iter__(self):
		self._indices = list(range(self._len))
		self._index = 0

		if self._shuffle:
			random.seed(time.time())
			random.shuffle(self._indices)

		self._iter_ready = True

		return self

	def __next__(self):
		if not self._iter_ready:
			raise TypeError('Call iter() first to build the iterator.')

		if self._index == np.int(np.ceil(np.float(self._len) / np.float(self._batch_size))):
			raise StopIteration('Call iter() again to rebuild the iterator.')

		i_end = min((self._index + 1) * self._batch_size, self._len)
		if self._fill_batch:
			i_start = max(0, i_end - self._batch_size)
		else:
			i_start = self._index * self._batch_size

		indices = self._indices[i_start:i_end]
		images, labels = self[indices]

		self._index += 1

		return images, labels

	def __len__(self):
		return np.int(np.ceil(np.float(self._len) / np.float(self._batch_size)))

	def __getitem__(self, item):
		if type(item) in [int, slice, list, tuple]:
			images = self._images[item].copy()
			for tran in self._trans:
				images = tran(images)
			return images, self._labels[item].copy()
		elif type(item) == str:
			if item == 'images':
				return self._images.copy()
			elif item == 'labels':
				return self._labels.copy()

		raise TypeError('Unsupported operation.')

	@staticmethod
	def _to_float32(arr: np.ndarray):
		"""
		:param arr: numpy array of type np.uint8 with max value less than or equal to 255
		:return: numpy array of type np.float32 with max value less than or equal to 1.0
		"""
		arr = arr.astype(np.float32)
		arr = arr / 255
		return arr

	@staticmethod
	def _to_gray(arr: np.ndarray):
		"""
		:param arr: numpy array of any type
		:return: numpy array as the same type of the original array
		"""
		assert len(arr.shape) in [3, 4]

		if arr.shape[-1] != 3:
			return arr

		dtype = arr.dtype
		arr = arr.astype(np.float32)
		if len(arr.shape) == 3:
			arr[:, :, 0] *= 0.3
			arr[:, :, 1] *= 0.59
			arr[:, :, 2] *= 0.11
		else:
			arr[:, :, :, 0] *= 0.3
			arr[:, :, :, 1] *= 0.59
			arr[:, :, :, 2] *= 0.11
		arr = arr.sum(axis=-1, keepdims=True).astype(dtype)

		return arr

	@staticmethod
	def _normalize(arr: np.ndarray):
		"""
		:param arr: numpy array of any type, but integer is not recommended due to loss of accuracy.
		:return: numpy array as the same type of the original array
		"""
		assert len(arr.shape) in [3, 4]

		dtype = arr.dtype
		arr = arr.astype(np.float32)

		if len(arr.shape) == 3:
			_min = arr.min()
			_max = arr.max()
			arr = (arr - _min) / (_max - _min)
		else:
			for i in range(len(arr)):
				_min = arr[i].min()
				_max = arr[i].max()
				arr[i] = (arr[i] - _min) / (_max - _min)

		if dtype == np.uint8:
			arr *= 255

		return arr.astype(dtype)

	@staticmethod
	def _resize(arr: np.ndarray, shape):
		"""
		:param arr: 3-dim image of type np.uint8
		:param shape: 2-dim array
		:return: resized image
		"""
		reshape = arr.ndim == 3 and arr.shape[-1] == 1

		if reshape:
			arr = arr.squeeze(axis=-1)

		arr = Image.fromarray(arr)
		from PIL import ImageFilter
		arr.filter(ImageFilter.GaussianBlur(2))
		arr = arr.resize(shape)
		arr = np.array(arr)
		arr = arr.clip(0, 255)

		if reshape:
			arr = arr[:, :, None]

		return arr

	@staticmethod
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
			new_image[i] = DatasetHelper._resize(images[i][0], shape)
		images = new_image

		labels = np.concatenate(labels)

		return images, labels
