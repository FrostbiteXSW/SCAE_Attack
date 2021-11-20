import abc
import os.path
import time

import numpy as np
import sonnet as snt
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl
from tensorflow_core.contrib.factorization.python.ops.clustering_ops import COSINE_DISTANCE
from tensorflow_core.contrib.factorization.python.ops.clustering_ops import KMEANS_PLUS_PLUS_INIT
from tensorflow_core.contrib.factorization.python.ops.clustering_ops import KMeans as _KMeans_Raw
from tensorflow_core.contrib.factorization.python.ops.clustering_ops import _InitializeClustersOpFactory
from tqdm import tqdm, trange

from capsules import primary
from capsules.attention import SetTransformer
from capsules.models.scae import ImageAutoencoder
from capsules.models.scae import ImageCapsule
from tools.utilities import DatasetHelper


class _ModelCollector(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def run(self, inputs, to_collect, additional_inputs=None):
		pass

	@abc.abstractmethod
	def __call__(self, images):
		pass


def _stacked_capsule_autoencoder(
		canvas_size,
		template_size=11,
		n_part_caps=16,
		n_part_caps_dims=6,
		n_part_special_features=16,
		part_encoder_noise_scale=0.,
		n_channels=1,
		colorize_templates=False,
		use_alpha_channel=False,
		template_nonlin='relu1',
		color_nonlin='relu1',
		n_obj_caps=10,
		n_obj_caps_params=32,
		obj_decoder_noise_type=None,
		obj_decoder_noise_scale=0.,
		num_classes=10,
		prior_within_example_sparsity_weight=1.,
		prior_between_example_sparsity_weight=1.,
		posterior_within_example_sparsity_weight=10.,
		posterior_between_example_sparsity_weight=10.,
		set_transformer_n_layers=3,
		set_transformer_n_heads=1,
		set_transformer_n_dims=16,
		set_transformer_n_output_dims=256,
		part_cnn_strides=None,
		prep='none',
		scope='stacked_capsule_autoencoders'
):
	if part_cnn_strides is None:
		part_cnn_strides = [2, 2, 1, 1]

	"""Builds the SCAE."""
	with tf.variable_scope(scope, 'stacked_capsule_autoencoders'):
		img_size = [canvas_size] * 2
		template_size = [template_size] * 2

		cnn_encoder = snt.nets.ConvNet2D(
			output_channels=[128] * 4,
			kernel_shapes=[3],
			strides=part_cnn_strides,
			paddings=[snt.VALID],
			activate_final=True
		)

		part_encoder = primary.CapsuleImageEncoder(
			cnn_encoder,
			n_part_caps,
			n_part_caps_dims,
			n_features=n_part_special_features,
			similarity_transform=False,
			encoder_type='conv_att',
			noise_scale=part_encoder_noise_scale
		)

		part_decoder = primary.TemplateBasedImageDecoder(
			output_size=img_size,
			template_size=template_size,
			n_channels=n_channels,
			learn_output_scale=False,
			colorize_templates=colorize_templates,
			use_alpha_channel=use_alpha_channel,
			template_nonlin=template_nonlin,
			color_nonlin=color_nonlin,
		)

		obj_encoder = SetTransformer(
			n_layers=set_transformer_n_layers,
			n_heads=set_transformer_n_heads,
			n_dims=set_transformer_n_dims,
			n_output_dims=set_transformer_n_output_dims,
			n_outputs=n_obj_caps,
			layer_norm=True,
			dropout_rate=0.)

		obj_decoder = ImageCapsule(
			n_obj_caps,
			2,
			n_part_caps,
			n_caps_params=n_obj_caps_params,
			n_hiddens=128,
			learn_vote_scale=True,
			deformations=True,
			noise_type=obj_decoder_noise_type,
			noise_scale=obj_decoder_noise_scale,
			similarity_transform=False
		)

		model = ImageAutoencoder(
			primary_encoder=part_encoder,
			primary_decoder=part_decoder,
			encoder=obj_encoder,
			decoder=obj_decoder,
			input_key='image',
			label_key='label',
			target_key='target',
			n_classes=num_classes,
			dynamic_l2_weight=10,
			caps_ll_weight=1.,
			vote_type='enc',
			pres_type='enc',
			stop_grad_caps_inpt=True,
			stop_grad_caps_target=True,
			prior_sparsity_loss_type='l2',
			prior_within_example_sparsity_weight=prior_within_example_sparsity_weight,
			prior_between_example_sparsity_weight=prior_between_example_sparsity_weight,
			posterior_sparsity_loss_type='entropy',
			posterior_within_example_sparsity_weight=posterior_within_example_sparsity_weight,
			posterior_between_example_sparsity_weight=posterior_between_example_sparsity_weight,
			prep=prep
		)

	return model


class ScaeBasement(_ModelCollector):
	"""
		SCAE model collector with graph that is not finalized during initialization.
		Instead, finalization can be done by calling function finalize().
		After initialization, supportive models can be applied to the graph of this SCAE.
	"""

	def __init__(
			self,
			input_size,
			template_size=11,
			n_part_caps=16,
			n_part_caps_dims=6,
			n_part_special_features=16,
			part_encoder_noise_scale=0.,
			colorize_templates=False,
			use_alpha_channel=False,
			template_nonlin='relu1',
			color_nonlin='relu1',
			n_obj_caps=10,
			n_obj_caps_params=32,
			obj_decoder_noise_type=None,
			obj_decoder_noise_scale=0.,
			num_classes=10,
			prior_within_example_sparsity_weight=1.,
			prior_between_example_sparsity_weight=1.,
			posterior_within_example_sparsity_weight=10.,
			posterior_between_example_sparsity_weight=10.,
			set_transformer_n_layers=3,
			set_transformer_n_heads=1,
			set_transformer_n_dims=16,
			set_transformer_n_output_dims=256,
			part_cnn_strides=None,
			prep='none',
			is_training=True,
			learning_rate=1e-4,
			use_lr_schedule=True,
			scope='SCAE',
			snapshot=None
	):
		if input_size is None:
			input_size = [20, 224, 224, 3]
		if part_cnn_strides is None:
			part_cnn_strides = [2, 2, 1, 1]

		self._input_size = input_size
		self._template_size = template_size
		self._n_part_caps = n_part_caps
		self._n_part_caps_dims = n_part_caps_dims
		self._n_part_special_features = n_part_special_features
		self._part_encoder_noise_scale = part_encoder_noise_scale
		self._n_channels = input_size[-1]
		self._colorize_templates = colorize_templates
		self._use_alpha_channel = use_alpha_channel
		self._template_nonlin = template_nonlin
		self._color_nonlin = color_nonlin
		self._n_obj_caps = n_obj_caps
		self._n_obj_caps_params = n_obj_caps_params
		self._obj_decoder_noise_type = obj_decoder_noise_type
		self._obj_decoder_noise_scale = obj_decoder_noise_scale
		self._num_classes = num_classes
		self._prior_within_example_sparsity_weight = prior_within_example_sparsity_weight
		self._prior_between_example_sparsity_weight = prior_between_example_sparsity_weight
		self._posterior_within_example_sparsity_weight = posterior_within_example_sparsity_weight
		self._posterior_between_example_sparsity_weight = posterior_between_example_sparsity_weight
		self._set_transformer_n_layers = set_transformer_n_layers
		self._set_transformer_n_heads = set_transformer_n_heads
		self._set_transformer_n_dims = set_transformer_n_dims
		self._set_transformer_n_output_dims = set_transformer_n_output_dims
		self._part_cnn_strides = part_cnn_strides
		self._prep = prep
		self._is_training = is_training
		self._learning_rate = learning_rate
		self._use_lr_schedule = use_lr_schedule
		self._scope = scope
		self._snapshot = snapshot

		graph = tf.Graph()

		with graph.as_default():
			self._sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

			self._input = tf.placeholder(tf.float32, input_size)
			self._model = _stacked_capsule_autoencoder(input_size[1],  # Assume width equals height
			                                           template_size,
			                                           n_part_caps,
			                                           n_part_caps_dims,
			                                           n_part_special_features,
			                                           part_encoder_noise_scale,
			                                           input_size[-1],
			                                           colorize_templates,
			                                           use_alpha_channel,
			                                           template_nonlin,
			                                           color_nonlin,
			                                           n_obj_caps,
			                                           n_obj_caps_params,
			                                           obj_decoder_noise_type,
			                                           obj_decoder_noise_scale,
			                                           num_classes,
			                                           prior_within_example_sparsity_weight,
			                                           prior_between_example_sparsity_weight,
			                                           posterior_within_example_sparsity_weight,
			                                           posterior_between_example_sparsity_weight,
			                                           set_transformer_n_layers,
			                                           set_transformer_n_heads,
			                                           set_transformer_n_dims,
			                                           set_transformer_n_output_dims,
			                                           part_cnn_strides,
			                                           prep,
			                                           scope)

			if is_training:
				self._label = tf.placeholder(tf.int64, [input_size[0]])
				data = {'image': self._input, 'label': self._label}
				self._res = self._model(data)

				self._loss = self._model._loss(data, self._res)

				self._global_step = tf.train.get_or_create_global_step()
				self._global_step.initializer.run(session=self._sess)

				if use_lr_schedule:
					self._learning_rate = tf.train.exponential_decay(
						global_step=self._global_step,
						learning_rate=self._learning_rate,
						decay_steps=1e4,
						decay_rate=.96
					)

				eps = 1e-2 / float(input_size[0]) ** 2
				self._optimizer = tf.train.RMSPropOptimizer(self._learning_rate, momentum=.9, epsilon=eps)

				self._train_step = self._optimizer.minimize(self._loss, global_step=self._global_step,
				                                            var_list=tf.trainable_variables(scope=scope))
				self._sess.run(tf.initialize_variables(var_list=self._optimizer.variables()))
			else:
				data = {'image': self._input}
				self._res = self._model(data)

			self._saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope))

			if snapshot:
				print('Restoring from snapshot: {}'.format(snapshot))
				self._saver.restore(self._sess, snapshot)
			else:
				self._sess.run(tf.initialize_variables(var_list=tf.trainable_variables(scope=scope)))

	def _valid_shape(self, images, labels=None):
		if len(images.shape) == 3:
			images = images[None]
		elif len(images.shape) != 4:
			raise ValueError('Input shape \'{}\' is invalid'.format(images.shape))

		num_images = images.shape[0]

		if num_images != self._input_size[0]:
			new_images = np.zeros(self._input_size)
			new_images[:num_images] = images
			images = new_images

		if labels is not None:
			if len(labels.shape) == 1:
				if labels.shape[0] != self._input_size[0]:
					new_labels = np.zeros([self._input_size[0]])
					new_labels[:num_images] = labels
					labels = new_labels
			else:
				raise ValueError('Input shape \'{}\' is invalid'.format(labels.shape))
			return images, num_images, labels
		else:
			return images, num_images

	def __call__(self, images):
		images, num_images = self._valid_shape(images)

		try:
			return self._sess.run(self._res.prior_cls_logits, feed_dict={self._input: images})[:num_images]
		except tf.errors.InvalidArgumentError:
			pass

		raise Exception('Model is in training mode. Use run() instead.')

	def run(self, images, to_collect, labels=None):
		try:
			if labels is not None:
				return self._sess.run(to_collect, feed_dict={
					self._input: images,
					self._label: labels
				})

			return self._sess.run(to_collect, feed_dict={
				self._input: images
			})

		except tf.errors.InvalidArgumentError:
			pass

		raise Exception('Model is in training mode. Labels must be provided.')

	def finalize(self):
		self._sess.graph.finalize()
		print('Graph is now finalized and cannot be modified.')

	def simple_test(self, dataset: DatasetHelper):
		# Simple test
		test_acc_prior = 0.
		test_acc_posterior = 0.

		for images, labels in tqdm(dataset, desc='Testing'):
			if self._is_training:
				feed_dict = {self._input: images, self._label: labels}
			else:
				feed_dict = {self._input: images}

			test_pred_prior, test_pred_posterior = self._sess.run(
				[self._res.prior_cls_pred,
				 self._res.posterior_cls_pred],
				feed_dict=feed_dict)
			test_acc_prior += (test_pred_prior == labels).sum()
			test_acc_posterior += (test_pred_posterior == labels).sum()

		print('Supervised acc: prior={:.6f}, posterior={:.6f}'
		      .format(test_acc_prior / dataset.dataset_size, test_acc_posterior / dataset.dataset_size))

	def save_model(self, path):
		print('Saving model to {}...'.format(os.path.abspath(path)))
		return self._saver.save(self._sess, save_path=path)

	def train_step(self, images, labels):
		if not self._is_training:
			raise Exception('Model is not in training mode.')

		return self._sess.run(self._train_step, feed_dict={
			self._input: images,
			self._label: labels
		})


class _KMeans(_KMeans_Raw):
	"""Try to redefine some functions to make them more useful."""

	def training_graph(self):
		"""Generate a training graph for kmeans algorithm.

		This returns, among other things, an op that chooses initial centers
		(init_op), a boolean variable that is set to True when the initial centers
		are chosen (cluster_centers_initialized), and an op to perform either an
		entire Lloyd iteration or a mini-batch of a Lloyd iteration (training_op).
		The caller should use these components as follows. A single worker should
		execute init_op multiple times until cluster_centers_initialized becomes
		True. Then multiple workers may execute training_op any number of times.

		Returns:
			A tuple consisting of:
			all_scores: A matrix (or list of matrices) of dimensions (num_input,
				num_clusters) where the value is the distance of an input vector and a
				cluster center.
			cluster_idx: A vector (or list of vectors). Each element in the vector
				corresponds to an input row in 'inp' and specifies the cluster id
				corresponding to the input.
			scores: Similar to cluster_idx but specifies the distance to the
				assigned cluster instead.
			cluster_centers_initialized: scalar indicating whether clusters have been
				initialized.
			init_op: an op to initialize the clusters.
			training_op: an op that runs an iteration of training.
		"""
		# Implementation of kmeans.
		if (isinstance(self._initial_clusters, str) or
				callable(self._initial_clusters)):
			initial_clusters = self._initial_clusters
			num_clusters = ops.convert_to_tensor(self._num_clusters)
		else:
			initial_clusters = ops.convert_to_tensor(self._initial_clusters)
			num_clusters = array_ops.shape(initial_clusters)[0]

		inputs = self._inputs
		(cluster_centers_var, cluster_centers_initialized, total_counts,
		 cluster_centers_updated,
		 update_in_steps) = self._create_variables(num_clusters)
		init_op = _InitializeClustersOpFactory(
			self._inputs, num_clusters, initial_clusters, self._distance_metric,
			self._random_seed, self._kmeans_plus_plus_num_retries,
			self._kmc2_chain_length, cluster_centers_var, cluster_centers_updated,
			cluster_centers_initialized).op()
		cluster_centers = cluster_centers_var

		if self._distance_metric == COSINE_DISTANCE:
			inputs = self._l2_normalize_data(inputs)
			if not self._clusters_l2_normalized():
				cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)

		all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
		if self._use_mini_batch:
			sync_updates_op = self._mini_batch_sync_updates_op(
				update_in_steps, cluster_centers_var, cluster_centers_updated,
				total_counts)
			assert sync_updates_op is not None
			with ops.control_dependencies([sync_updates_op]):
				training_op = self._mini_batch_training_op(
					inputs, cluster_idx, cluster_centers_updated, total_counts)
		else:
			assert cluster_centers == cluster_centers_var
			training_op = self._full_batch_training_op(
				inputs, num_clusters, cluster_idx, cluster_centers_var)

		return (all_scores, cluster_idx, scores, cluster_centers_initialized,
		        init_op, training_op, cluster_centers)


class KMeans(_ModelCollector):
	class KMeansTypes:
		Prior = 'caps_presence_prob'
		Posterior = 'posterior_mixing_probs'

	def __init__(
			self,
			kmeans_type: str,
			scae: ScaeBasement,
			num_clusters=10,
			initial_clusters=KMEANS_PLUS_PLUS_INIT,
			distance_metric=COSINE_DISTANCE,
			use_mini_batch=True,
			mini_batch_steps_per_iteration=1,
			random_seed=int(time.time()),
			kmeans_plus_plus_num_retries=10,
			kmc2_chain_length=200,
			is_training=True,
			scope='KMeans',
			snapshot=None
	):
		self._sess = scae._sess
		self._input_size = scae._input_size
		self._is_training = is_training
		self._valid_shape = scae._valid_shape

		with self._sess.graph.as_default():
			with tf.variable_scope(scope, 'KMeans'):
				self._input_scae = scae._input
				self._input = tf.placeholder(tf.float32, [None, scae._n_obj_caps])

				self._output_scae = getattr(scae._res, kmeans_type)
				if kmeans_type == self.KMeansTypes.Posterior:
					self._output_scae = tf.reduce_sum(self._output_scae, axis=1)

				map_table = tf.Variable(tf.convert_to_tensor([i for i in range(num_clusters)], dtype=tf.int32), trainable=True)
				self._map_table_placeholder = tf.placeholder(tf.int32, [num_clusters])
				self._assign_map_table = tf.assign(map_table, self._map_table_placeholder)

				self._model = _KMeans(
					inputs=self._input,
					num_clusters=num_clusters,
					initial_clusters=initial_clusters,
					distance_metric=distance_metric,
					use_mini_batch=use_mini_batch,
					mini_batch_steps_per_iteration=mini_batch_steps_per_iteration,
					random_seed=random_seed,
					kmeans_plus_plus_num_retries=kmeans_plus_plus_num_retries,
					kmc2_chain_length=kmc2_chain_length
				)

				if self._is_training:
					(
						_, self._output, _, _, self._init_op, self._training_op,
						self._cluster_centers) = self._model.training_graph()
				else:
					(_, self._output, _, _, _, _, self._cluster_centers) = self._model.training_graph()
				self._output = tf.one_hot(self._output[0], num_clusters)
				map_table_one_hot = tf.one_hot(map_table, num_clusters)
				self._output = tf.argmax(tf.matmul(self._output, map_table_one_hot), axis=1, output_type=tf.int32)

			self._saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope))

			if snapshot:
				print('Restoring from snapshot: {}'.format(snapshot))
				self._saver.restore(self._sess, snapshot)
			else:
				self._sess.run(tf.initialize_variables(var_list=[map_table]))
				self._sess.run(tf.initialize_variables(var_list=tf.trainable_variables(scope=scope)))

	def __call__(self, images):
		images, num_images = self._valid_shape(images)

		res = self._sess.run(self._output_scae, feed_dict={self._input_scae: images})[:num_images]
		res = self._sess.run(self._output, feed_dict={self._input: res})

		return res

	def run(self, caps, to_collect, labels=None):
		if caps is None:
			return self._sess.run(to_collect)
		return self._sess.run(to_collect, feed_dict={self._input: caps})

	def train(self, dataset: DatasetHelper, num_epochs=1000, verbose=False):
		if not self._is_training:
			raise Exception('Current model is not in training mode.')

		pres_list = []
		target_list = []

		for images, labels in tqdm(iter(dataset), desc='Collecting capsules') if verbose else iter(dataset):
			pres_list.append(self._sess.run(self._output_scae, feed_dict={self._input_scae: images}))
			target_list.append(labels)

		pres_list = np.concatenate(pres_list)
		target_list = np.concatenate(target_list)
		self._sess.run(self._init_op, feed_dict={self._input: pres_list})

		for _ in trange(num_epochs, desc='Training k-means') if verbose else range(num_epochs):
			self._sess.run(self._training_op, feed_dict={self._input: pres_list})

		output = self._sess.run(self._output, feed_dict={self._input: pres_list})
		self._sess.run(self._assign_map_table, feed_dict={
			self._map_table_placeholder: self._bipartite_match(output, target_list, self._model._num_clusters)})

	@staticmethod
	def _bipartite_match(preds: np.ndarray, labels: np.ndarray, n_classes=None, presence=None):
		"""Does maximum biprartite matching between `preds` and `labels`."""
		if n_classes is not None:
			n_gt_labels, n_pred_labels = n_classes, n_classes
		else:
			n_gt_labels = np.unique(labels).shape[0]
			n_pred_labels = np.unique(preds).shape[0]

		cost_matrix = np.zeros([n_gt_labels, n_pred_labels], dtype=np.int32)
		for label in range(n_gt_labels):
			label_idx = (labels == label)
			for new_label in range(n_pred_labels):
				errors = np.equal(preds[label_idx], new_label).astype(np.float32)
				if presence is not None:
					errors *= presence[label_idx]

				num_errors = errors.sum()
				cost_matrix[label, new_label] = -num_errors

		labels_idx, preds_idx = linear_sum_assignment(cost_matrix)

		preds_2_labels = np.array([0 for _ in range(n_classes)], dtype=np.int8)
		for i in range(n_classes):
			preds_2_labels[preds_idx[i]] = labels_idx[i]

		return preds_2_labels

	def save_model(self, path):
		print('Saving model to {}...'.format(path))
		return self._saver.save(self._sess, save_path=path)


class Attacker(metaclass=abc.ABCMeta):
	class Classifiers:
		PriK = 'PriK'
		PosK = 'PosK'
		PriL = 'PriL'
		PosL = 'PosL'

	@abc.abstractmethod
	def __call__(
			self,
			images: np.ndarray,
			labels: np.ndarray,
			nan_if_fail: bool,
			verbose: bool,
			use_mask: bool,
			**mask_kwargs
	):
		"""
			Return perturbed images of specified samples.

			:param images: Images to be attacked.
			:param labels: Labels corresponding to the images.
			:param nan_if_fail: If true, failed results will be set to np.nan, otherwise the original images.
			:param verbose: If true, a tqdm bar will be displayed.
			:param use_mask: If true, mask will applied to the perturbation.
			:param mask_kwargs: Arguments for generating the masks.

			:return Images as numpy array with the same shape as inputs.
		"""
		pass
