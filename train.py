from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy as np
import tensorflow as tf
from tqdm import trange

from model import stacked_capsule_autoencoders
from utilities import get_dataset, get_gtsrb, block_warnings, ModelCollector, to_float32


class SCAE(ModelCollector):
	def __init__(
			self,
			input_size,
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

		graph = tf.Graph()

		with graph.as_default():
			self.sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

			self.batch_input = tf.placeholder(tf.float32, input_size)
			self.model = stacked_capsule_autoencoders(input_size[1],  # Assume width equals height
			                                          template_size,
			                                          n_part_caps,
			                                          n_part_caps_dims,
			                                          n_part_special_features,
			                                          part_encoder_noise_scale,
			                                          n_channels,
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
				self.labels = tf.placeholder(tf.int64, [input_size[0]])
				data = {'image': self.batch_input, 'label': self.labels}
				self.res = self.model(data)

				self.loss = self.model._loss(data, self.res)

				if use_lr_schedule:
					global_step = tf.train.get_or_create_global_step()
					learning_rate = tf.train.exponential_decay(
						global_step=global_step,
						learning_rate=learning_rate,
						decay_steps=1e4,
						decay_rate=.96
					)
					global_step.initializer.run(session=self.sess)

				eps = 1e-2 / float(input_size[0]) ** 2
				optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=.9, epsilon=eps)

				self.train_step = optimizer.minimize(self.loss, var_list=tf.trainable_variables(scope=scope))
				self.sess.run(tf.initialize_variables(var_list=optimizer.variables()))
			else:
				data = {'image': self.batch_input}
				self.res = self.model(data)

			self.saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope))

			if snapshot:
				print('Restoring from snapshot: {}'.format(snapshot))
				self.saver.restore(self.sess, snapshot)
			else:
				self.sess.run(tf.initialize_variables(var_list=tf.trainable_variables(scope=scope)))

			# Freeze graph
			self.sess.graph.finalize()

	def run(self, images, to_collect):
		return self.sess.run(to_collect, feed_dict={self.batch_input: images})

	def __call__(self, images):
		return self.sess.run(self.res.prior_cls_logits, feed_dict={self.batch_input: images})


MNIST = 'mnist'
config_mnist = {
	'dataset': MNIST,
	'canvas_size': 40,
	'n_part_caps': 24,
	'n_obj_caps': 24,
	'n_channels': 1,
	'num_classes': 10,
	'colorize_templates': True,
	'use_alpha_channel': True,
	'prior_within_example_sparsity_weight': 2.,
	'prior_between_example_sparsity_weight': 0.35,
	'posterior_within_example_sparsity_weight': 0.7,
	'posterior_between_example_sparsity_weight': 0.2,
	'template_size': 11,
	'template_nonlin': 'sigmoid',
	'color_nonlin': 'relu1',
	'part_encoder_noise_scale': 4.0,
	'obj_decoder_noise_type': 'uniform',
	'obj_decoder_noise_scale': 4.0,
	'set_transformer_n_layers': 3,
	'set_transformer_n_heads': 1,
	'set_transformer_n_dims': 16,
	'set_transformer_n_output_dims': 256,
	'part_cnn_strides': [2, 2, 1, 1],
	'prep': 'none'
}

FASHION_MNIST = 'fashion_mnist'
config_fashion_mnist = {
	'dataset': FASHION_MNIST,
	'canvas_size': 40,
	'n_part_caps': 24,
	'n_obj_caps': 24,
	'n_channels': 1,
	'num_classes': 10,
	'colorize_templates': True,
	'use_alpha_channel': True,
	'prior_within_example_sparsity_weight': 2.,
	'prior_between_example_sparsity_weight': 0.35,
	'posterior_within_example_sparsity_weight': 0.7,
	'posterior_between_example_sparsity_weight': 0.2,
	'template_size': 11,
	'template_nonlin': 'sigmoid',
	'color_nonlin': 'relu1',
	'part_encoder_noise_scale': 4.0,
	'obj_decoder_noise_type': 'uniform',
	'obj_decoder_noise_scale': 4.0,
	'set_transformer_n_layers': 3,
	'set_transformer_n_heads': 1,
	'set_transformer_n_dims': 16,
	'set_transformer_n_output_dims': 256,
	'part_cnn_strides': [2, 2, 1, 1],
	'prep': 'none'
}

GTSRB = 'gtsrb'
GTSRB_DATASET_PATH = './datasets/GTSRB-for-SCAE_Attack/GTSRB'
config_gtsrb = {
	'dataset': GTSRB,
	'canvas_size': 40,
	'n_part_caps': 40,
	'n_obj_caps': 64,
	'n_channels': 3,
	'classes': [1, 2, 7, 10, 11, 13, 14, 17, 35, 38],
	'num_classes': 10,
	'colorize_templates': True,
	'use_alpha_channel': True,
	'prior_within_example_sparsity_weight': 2.,
	'prior_between_example_sparsity_weight': 0.35,
	'posterior_within_example_sparsity_weight': 0.7,
	'posterior_between_example_sparsity_weight': 0.2,
	'template_size': 14,
	'template_nonlin': 'sigmoid',
	'color_nonlin': 'relu1',
	'part_encoder_noise_scale': 0,
	'obj_decoder_noise_type': None,
	'obj_decoder_noise_scale': 0,
	'set_transformer_n_layers': 3,
	'set_transformer_n_heads': 2,
	'set_transformer_n_dims': 64,
	'set_transformer_n_output_dims': 256,
	'part_cnn_strides': [1, 1, 2, 2],
	'prep': 'none'
}

SVHN = 'svhn_cropped'
config_svhn = {
	'dataset': SVHN,
	'canvas_size': 32,
	'n_part_caps': 24,
	'n_obj_caps': 32,
	'n_channels': 3,
	'num_classes': 10,
	'colorize_templates': True,
	'use_alpha_channel': True,
	'prior_within_example_sparsity_weight': 2.,
	'prior_between_example_sparsity_weight': 0.35,
	'posterior_within_example_sparsity_weight': 0.7,
	'posterior_between_example_sparsity_weight': 0.2,
	'template_size': 14,
	'template_nonlin': 'sigmoid',
	'color_nonlin': 'relu1',
	'part_encoder_noise_scale': 4.0,
	'obj_decoder_noise_type': 'uniform',
	'obj_decoder_noise_scale': 4.0,
	'set_transformer_n_layers': 3,
	'set_transformer_n_heads': 2,
	'set_transformer_n_dims': 64,
	'set_transformer_n_output_dims': 128,
	'part_cnn_strides': [1, 1, 2, 2],
	'prep': 'sobel'
}

CIFAR10 = 'cifar10'
config_cifar10 = {
	'dataset': CIFAR10,
	'canvas_size': 32,
	'n_part_caps': 32,
	'n_obj_caps': 64,
	'n_channels': 3,
	'num_classes': 10,
	'colorize_templates': True,
	'use_alpha_channel': True,
	'prior_within_example_sparsity_weight': 2.,
	'prior_between_example_sparsity_weight': 0.35,
	'posterior_within_example_sparsity_weight': 0.7,
	'posterior_between_example_sparsity_weight': 0.2,
	'template_size': 14,
	'template_nonlin': 'sigmoid',
	'color_nonlin': 'relu1',
	'part_encoder_noise_scale': 4.0,
	'obj_decoder_noise_type': 'uniform',
	'obj_decoder_noise_scale': 4.0,
	'set_transformer_n_layers': 3,
	'set_transformer_n_heads': 2,
	'set_transformer_n_dims': 64,
	'set_transformer_n_output_dims': 128,
	'part_cnn_strides': [1, 1, 2, 2],
	'prep': 'sobel'
}

if __name__ == '__main__':
	block_warnings()

	config = config_gtsrb
	batch_size = 100
	max_train_steps = 300
	learning_rate = 3e-5
	snapshot = './checkpoints/{}/model.ckpt'.format(config['dataset'])

	model = SCAE(
		input_size=[batch_size, config['canvas_size'], config['canvas_size'], config['n_channels']],
		num_classes=config['num_classes'],
		n_part_caps=config['n_part_caps'],
		n_obj_caps=config['n_obj_caps'],
		n_channels=config['n_channels'],
		colorize_templates=config['colorize_templates'],
		use_alpha_channel=config['use_alpha_channel'],
		prior_within_example_sparsity_weight=config['prior_within_example_sparsity_weight'],
		prior_between_example_sparsity_weight=config['prior_between_example_sparsity_weight'],
		posterior_within_example_sparsity_weight=config['posterior_within_example_sparsity_weight'],
		posterior_between_example_sparsity_weight=config['posterior_between_example_sparsity_weight'],
		template_size=config['template_size'],
		template_nonlin=config['template_nonlin'],
		color_nonlin=config['color_nonlin'],
		part_encoder_noise_scale=config['part_encoder_noise_scale'],
		obj_decoder_noise_type=config['obj_decoder_noise_type'],
		obj_decoder_noise_scale=config['obj_decoder_noise_scale'],
		set_transformer_n_layers=config['set_transformer_n_layers'],
		set_transformer_n_heads=config['set_transformer_n_heads'],
		set_transformer_n_dims=config['set_transformer_n_dims'],
		set_transformer_n_output_dims=config['set_transformer_n_output_dims'],
		part_cnn_strides=config['part_cnn_strides'],
		prep=config['prep'],
		is_training=True,
		learning_rate=learning_rate,
		scope='SCAE',
		# use_lr_schedule=False,
		# snapshot=snapshot
	)

	if config['dataset'] == GTSRB:
		trainset = get_gtsrb('train', shape=[config['canvas_size'], config['canvas_size']], file_path='./datasets',
		                     save_only=True, gtsrb_raw_file_path=GTSRB_DATASET_PATH, gtsrb_classes=config['classes'])
		testset = get_gtsrb('test', shape=[config['canvas_size'], config['canvas_size']], file_path='./datasets',
		                    save_only=True, gtsrb_raw_file_path=GTSRB_DATASET_PATH, gtsrb_classes=config['classes'])
	else:
		trainset = get_dataset(config['dataset'], 'train', shape=[config['canvas_size'], config['canvas_size']],
		                       file_path='./datasets', save_only=True)
		testset = get_dataset(config['dataset'], 'test', shape=[config['canvas_size'], config['canvas_size']],
		                      file_path='./datasets', save_only=True)

	path = snapshot[:snapshot.rindex('/')]
	if not os.path.exists(path):
		os.makedirs(path)

	len_trainset = len(trainset['image'])
	len_testset = len(testset['image'])

	train_batches = np.int(np.ceil(np.float(len_trainset) / np.float(batch_size)))
	test_batches = np.int(np.ceil(np.float(len_testset) / np.float(batch_size)))

	random.seed(time.time())
	shuffle_indices = list(range(len_trainset))

	# Get the best score of last snapshot
	test_loss = 0.
	test_acc_prior = 0.
	test_acc_posterior = 0.
	for i_batch in trange(test_batches, desc='Testing'):
		i_end = min((i_batch + 1) * batch_size, len_testset)
		i_start = min(i_batch * batch_size, i_end - batch_size)
		images = to_float32(testset['image'][i_start:i_end])
		labels = testset['label'][i_start:i_end]
		test_pred_prior, test_pred_posterior, _test_loss = model.sess.run(
			[model.res.prior_cls_pred, model.res.posterior_cls_pred, model.loss],
			feed_dict={model.batch_input: images, model.labels: labels})
		test_loss += _test_loss
		test_acc_prior += (test_pred_prior == labels).sum()
		test_acc_posterior += (test_pred_posterior == labels).sum()
		assert not np.isnan(test_loss)
	print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
		test_loss / len_testset,
		test_acc_prior / len_testset,
		test_acc_posterior / len_testset
	))
	best_score = test_loss / len_testset

	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		random.shuffle(shuffle_indices)

		for i_batch in trange(train_batches, desc='Training'):
			i_end = min((i_batch + 1) * batch_size, len_trainset)
			i_start = min(i_batch * batch_size, i_end - batch_size)
			indices = shuffle_indices[i_start:i_end]
			images = to_float32(trainset['image'][indices])
			labels = trainset['label'][indices]
			model.sess.run(model.train_step, feed_dict={model.batch_input: images, model.labels: labels})

		test_loss = 0.
		test_acc_prior = 0.
		test_acc_posterior = 0.
		for i_batch in trange(test_batches, desc='Testing'):
			i_end = min((i_batch + 1) * batch_size, len_testset)
			i_start = min(i_batch * batch_size, i_end - batch_size)
			images = to_float32(testset['image'][i_start:i_end])
			labels = testset['label'][i_start:i_end]
			test_pred_prior, test_pred_posterior, _test_loss = model.sess.run(
				[model.res.prior_cls_pred, model.res.posterior_cls_pred, model.loss],
				feed_dict={model.batch_input: images, model.labels: labels})
			test_loss += _test_loss
			test_acc_prior += (test_pred_prior == labels).sum()
			test_acc_posterior += (test_pred_posterior == labels).sum()
			assert not np.isnan(test_loss)
		print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
			test_loss / len_testset,
			test_acc_prior / len_testset,
			test_acc_posterior / len_testset
		))

		score = test_loss / len_testset
		if score < best_score:
			print('Saving model...({:.6f} > {:.6f})'.format(score, best_score))
			model.saver.save(model.sess, snapshot)
			best_score = score
