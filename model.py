import sonnet as snt
import tensorflow as tf

from capsules import primary
from capsules.attention import SetTransformer
from capsules.models.scae import ImageAutoencoder
from capsules.models.scae import ImageCapsule


def stacked_capsule_autoencoders(
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
		stop_gradient=True,
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
			n_classes=num_classes,
			dynamic_l2_weight=10,
			caps_ll_weight=1.,
			vote_type='enc',
			pres_type='enc',
			stop_grad_caps_inpt=stop_gradient,
			stop_grad_caps_target=stop_gradient,
			prior_sparsity_loss_type='l2',
			prior_within_example_sparsity_weight=prior_within_example_sparsity_weight,
			prior_between_example_sparsity_weight=prior_between_example_sparsity_weight,
			posterior_sparsity_loss_type='entropy',
			posterior_within_example_sparsity_weight=posterior_within_example_sparsity_weight,
			posterior_between_example_sparsity_weight=posterior_between_example_sparsity_weight,
			prep=prep
		)

	return model
