import os

import numpy as np
import tensorflow_datasets as tfds
from monty.collections import AttrDict
from tqdm import tqdm

from tools.model import ScaeBasement
from tools.utilities import block_warnings, DatasetHelper


def build_from_config(
		config,
		batch_size,
		is_training=False,
		learning_rate=1e-4,
		use_lr_schedule=True,
		scope='SCAE',
		snapshot=None
):
	return ScaeBasement(
		input_size=[batch_size, config['canvas_size'], config['canvas_size'], config['n_channels']],
		num_classes=config['num_classes'],
		n_part_caps=config['n_part_caps'],
		n_obj_caps=config['n_obj_caps'],
		colorize_templates=config['colorize_templates'],
		use_alpha_channel=config['use_alpha_channel'],
		prior_within_example_sparsity_weight=config['prior_within_example_sparsity_weight'],
		prior_between_example_sparsity_weight=config['prior_between_example_sparsity_weight'],
		posterior_within_example_sparsity_weight=config['posterior_within_example_sparsity_weight'],
		posterior_between_example_sparsity_weight=config['posterior_between_example_sparsity_weight'],
		template_size=config['template_size'],
		template_nonlin=config['template_nonlin'],
		color_nonlin=config['color_nonlin'],
		part_encoder_noise_scale=config['part_encoder_noise_scale'] if is_training else 0.,
		obj_decoder_noise_type=config['obj_decoder_noise_type'] if is_training else None,
		obj_decoder_noise_scale=config['obj_decoder_noise_scale'] if is_training else 0.,
		set_transformer_n_layers=config['set_transformer_n_layers'],
		set_transformer_n_heads=config['set_transformer_n_heads'],
		set_transformer_n_dims=config['set_transformer_n_dims'],
		set_transformer_n_output_dims=config['set_transformer_n_output_dims'],
		part_cnn_strides=config['part_cnn_strides'],
		prep=config['prep'],
		is_training=is_training,
		learning_rate=learning_rate,
		scope=scope,
		use_lr_schedule=use_lr_schedule,
		snapshot=snapshot
	)


class Configs:
	MNIST = 'mnist'
	config_mnist = AttrDict({
		'name': MNIST,
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
	})

	AFFNIST = 'affnist'
	config_affnist = AttrDict({
		'name': AFFNIST,
		'dataset': AFFNIST,
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
	})

	FASHION_MNIST = 'fashion_mnist'
	config_fashion_mnist = AttrDict({
		'name': FASHION_MNIST,
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
	})

	GTSRB = 'gtsrb'
	GTSRB_DATASET_PATH = './datasets/GTSRB-for-SCAE_Attack/GTSRB'
	GTSRB_CLASSES = [1, 2, 7, 10, 11, 13, 14, 17, 35, 38]
	config_gtsrb = AttrDict({
		'name': GTSRB,
		'dataset': GTSRB,
		'canvas_size': 40,
		'n_part_caps': 40,
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
		'part_encoder_noise_scale': 0,
		'obj_decoder_noise_type': None,
		'obj_decoder_noise_scale': 0,
		'set_transformer_n_layers': 3,
		'set_transformer_n_heads': 2,
		'set_transformer_n_dims': 64,
		'set_transformer_n_output_dims': 256,
		'part_cnn_strides': [1, 1, 2, 2],
		'prep': 'none'
	})

	SVHN = 'svhn_cropped'
	config_svhn = AttrDict({
		'name': SVHN,
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
	})

	CIFAR10 = 'cifar10'
	config_cifar10 = AttrDict({
		'name': CIFAR10,
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
	})

	MNIST_CORRUPTED = 'mnist_corrupted'
	configs_mnist_corrupted = AttrDict({corrupt_config.name: AttrDict({
		'name': f'mnist_corrupted_{corrupt_config.name}',
		'dataset': 'mnist_corrupted',
		'builder_kwargs': {'config': corrupt_config},
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
	}) for corrupt_config in tfds.image.MNISTCorrupted.BUILDER_CONFIGS})


if __name__ == '__main__':
	block_warnings()

	config = Configs.config_mnist
	batch_size = 100
	max_train_steps = 300
	learning_rate = 3e-5
	snapshot = './checkpoints/{}/model.ckpt'.format(config['name'])

	model = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=True,
		learning_rate=learning_rate,
		use_lr_schedule=True,
		snapshot=None
	)
	model.finalize()

	trainset = DatasetHelper(config['dataset'], 'train', shape=[config['canvas_size']] * 2,
	                         file_path='./datasets', save_after_load=True,
	                         batch_size=batch_size, shuffle=True, fill_batch=True,
	                         normalize=True if config['dataset'] == Configs.GTSRB else False,
	                         gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES,
	                         builder_kwargs=config.get('builder_kwargs', None))
	testset = DatasetHelper(config['dataset'], 'test', shape=[config['canvas_size']] * 2,
	                        file_path='./datasets', save_after_load=True,
	                        batch_size=batch_size, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES,
	                        builder_kwargs=config.get('builder_kwargs', None))

	path = snapshot[:snapshot.rindex('/')]
	if not os.path.exists(path):
		os.makedirs(path)

	# Classification accuracy test
	model.simple_test(testset)

	# Train model
	for epoch in range(max_train_steps):
		print('\n[Epoch {}/{}]'.format(epoch + 1, max_train_steps))

		for images, labels in tqdm(trainset, desc='Training'):
			model.train_step(images, labels)

		test_loss = 0.
		test_acc_prior = 0.
		test_acc_posterior = 0.
		for images, labels in tqdm(testset, desc='Testing'):
			test_pred_prior, test_pred_posterior, _test_loss = model.run(
				images=images,
				labels=labels,
				to_collect=[model._res.prior_cls_pred,
				            model._res.posterior_cls_pred,
				            model._loss]
			)
			test_loss += _test_loss
			test_acc_prior += (test_pred_prior == labels).sum()
			test_acc_posterior += (test_pred_posterior == labels).sum()
			assert not np.isnan(test_loss)
		test_loss /= testset.dataset_size

		print('loss: {:.6f}  prior acc: {:.6f}  posterior acc: {:.6f}'.format(
			test_loss,
			test_acc_prior / testset.dataset_size,
			test_acc_posterior / testset.dataset_size
		))

		model.save_model(snapshot)
