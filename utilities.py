import abc
import copy
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
	shape = image.shape[:2]

	for _ in range(times):
		new_image = np.zeros(image.shape)
		for x in range(shape[0]):
			for y in range(shape[1]):
				x_low, x_high = max(x - 1, 0), min(x + 2, shape[0])
				y_low, y_high = max(y - 1, 0), min(y + 2, shape[1])
				pixel_sum = image[x_low:x_high, y_low:y_high].sum()
				pixel_cnt = (x_high - x_low) * (y_high - y_low)
				new_image[x, y] = pixel_sum / pixel_cnt
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


def sigmoid(x):
	y = 1.0 / (1.0 + np.exp(-x))
	return y


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
		if list_except[i] == list_except[i+1]:
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

	list_except = list_except[left:right+1]
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

	file_name = name + '_' + split + '.npz'
	if file_path[-1] != '/' or file_path[-1] != '\\':
		file_name = '/' + file_name
	file_name = file_path + file_name

	if file_path and not save_only:
		if os.path.exists(file_name):
			print('Info: Loading dataset file from \"{}\". If you want to reshape images, set \"save_only\" to True.'
			      .format(file_name))
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


def deepfool(
		model: ModelCollector,
		image: np.ndarray,
		num_classes=10,
		overshoot=0.02,
		max_iter=50):
	"""
		https://github.com/MyRespect/AdversarialAttack/blob/master/deepfool_tf2/deepfool_tf.py
	"""

	image = image[None]

	logits = model(image)[0]

	I = logits.argsort()[::-1]
	I = I[0:num_classes]
	k_i = I[0]

	input_shape = image.shape
	pert_image = copy.deepcopy(image)
	w = np.zeros(input_shape)
	r_tot = np.zeros(input_shape)

	loop_i = 0
	for loop_i in range(max_iter):
		pert = np.inf

		logits, grads = model(pert_image, with_grad=True)
		logits = logits[0]
		grad_orig = grads[I[0]][0]

		k_i = np.argmax(logits)
		if k_i != I[0]:
			break

		for k in range(1, num_classes):
			cur_grad = grads[I[k]][0]

			w_k = cur_grad - grad_orig
			f_k = logits[I[k]] - logits[I[0]]
			pert_k = abs(f_k) / (np.linalg.norm(np.reshape(w_k, [-1])) + 1e-10)

			if pert_k < pert:
				pert = pert_k
				w = w_k

		r_i = (pert + 1e-4) * w / (np.linalg.norm(w) + 1e-10)
		r_tot = np.float32(r_tot + r_i)

		pert_image = (image + (1 + overshoot) * r_tot).clip(0, 1)

	r_tot = (pert_image - image)[0]
	pert_image = pert_image[0]

	return AttrDict(
		pert=r_tot,
		pert_epochs=loop_i + 1,
		true_cls=I[0],
		pert_cls=k_i,
		pert_image=pert_image
	)


def universal_perturbation(
		model: ModelCollector,
		images: np.ndarray,
		batch_size: int,
		target_fooling_rate=0.8,
		max_iter_uni=50,
		xi=10,
		p=np.inf,
		num_classes=10,
		overshoot=0.02,
		max_iter=10):
	"""
		https://github.com/LTS4/universal/blob/master/python/universal_pert.py

    :param model: Target model
    :param images: Target images
    :param batch_size: batch size for evaluation
    :param target_fooling_rate: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02)
    :param max_iter: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
  """

	def proj_lp(v, xi, p):
		# Project on the lp ball centered at 0 and of radius xi

		# SUPPORTS only p = 2 and p = Inf for now
		if p == 2:
			v = v * min(1, xi / np.linalg.norm(v.flatten()))
		elif p == np.inf:
			v = np.sign(v) * np.minimum(abs(v), xi)
		else:
			raise ValueError('Values of p different from 2 and Inf are currently not supported...')

		return v

	v = 0

	for itr in range(max_iter_uni):
		# Go through the data set and compute the perturbation increments sequentially
		for i in range(len(images)):
			image = images[i]
			v_image = (image + v).clip(0, 1)

			pred = np.argmax(model(image[None]), axis=-1)[0]
			pred_v = np.argmax(model(v_image[None]), axis=-1)[0]

			if pred == pred_v:
				print('\r>> k = {}, pass # {}'.format(i, itr), end='')

				# Compute adversarial perturbation
				r_tot, loop_i, _, _, _ = deepfool(
					model,
					v_image,
					num_classes=num_classes,
					overshoot=overshoot,
					max_iter=max_iter
				).values()

				# Make sure it converged...
				if loop_i <= max_iter and True not in np.isnan(r_tot):
					v += r_tot

					# Project on l_p ball
					v = proj_lp(v, xi, p)

		est_labels_orig = np.zeros(len(images))
		est_labels_pert = np.zeros(len(images))

		num_batches = np.int(np.ceil(np.float(len(images)) / np.float(batch_size)))

		# Compute the estimated labels in batches
		for i_batch in range(num_batches):
			i_start = (i_batch * batch_size)
			i_end = min((i_batch + 1) * batch_size, len(images))
			images = images[i_start:i_end]

			est_labels_orig[i_start:i_end] = np.argmax(model(images), axis=-1).flatten()
			est_labels_pert[i_start:i_end] = np.argmax(model((images + v).clip(0, 1)), axis=-1).flatten()

		# Compute the fooling rate
		fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(len(images)))
		print('\rFooling rate = {}'.format(fooling_rate))
		if fooling_rate >= target_fooling_rate:
			break

	return v
