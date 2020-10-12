from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import joblib

from model import stacked_capsule_autoencoders
from utilities import *


class SCAE_L2_Attack(ModelCollector):
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
			stop_gradient=True,
			prior_within_example_sparsity_weight=1.,
			prior_between_example_sparsity_weight=1.,
			posterior_within_example_sparsity_weight=10.,
			posterior_between_example_sparsity_weight=10.,
			learning_rate=1e-4,
			optimizer='Adam',
			scope='SCAE',
			snapshot=None,
			capsule_loss_type='PriPos'  # choose from 'Pri', 'Pos' or 'PriPos'. Default is 'PriPos'.
	):
		if input_size is None:
			input_size = [1, 224, 224, 3]

		graph = tf.Graph()

		with graph.as_default():
			self.sess = tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

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
			                                          stop_gradient,
			                                          prior_within_example_sparsity_weight,
			                                          prior_between_example_sparsity_weight,
			                                          posterior_within_example_sparsity_weight,
			                                          posterior_between_example_sparsity_weight,
			                                          scope)

			# Placeholders for variables to initialize
			self.input = tf.placeholder(tf.float32, input_size)
			self.mask = tf.placeholder(tf.float32, input_size)
			self.const = tf.placeholder(tf.float32, [])

			# Variables to be assigned during initialization
			pert_atanh = tf.Variable(tf.zeros(input_size))
			input = tf.Variable(tf.zeros(input_size), trainable=False)
			input_atanh = tf.atanh((input - 0.5) / 0.5 * 0.999999)
			mask = tf.Variable(tf.zeros(input_size), trainable=False)
			const = tf.Variable(tf.zeros([]), trainable=False)
			subset_prior_position = tf.Variable(tf.placeholder(tf.int64), trainable=False, validate_shape=False)
			subset_posterior_position = tf.Variable(tf.placeholder(tf.int64), trainable=False, validate_shape=False)

			self.pert_image = 0.5 * (tf.tanh(pert_atanh * mask + input_atanh) + 1)
			self.pert_res = self.model({'image': self.pert_image})

			object_capsule_set_prior = self.pert_res.caps_presence_prob
			object_capsule_set_posterior = tf.reduce_sum(self.pert_res.posterior_mixing_probs, axis=1)

			object_capsule_subset_prior = tf.gather(object_capsule_set_prior, subset_prior_position, axis=1)
			object_capsule_subset_posterior = tf.gather(object_capsule_set_posterior, subset_posterior_position, axis=1)

			c_loss_prior = tf.reduce_sum(0.5 * tf.square(object_capsule_subset_prior))
			c_loss_posterior = tf.reduce_sum(0.5 * tf.square(object_capsule_subset_posterior)) / (n_part_caps ** 2)

			if capsule_loss_type == 'Pri':
				self.c_loss = const * c_loss_prior
			elif capsule_loss_type == 'Pos':
				self.c_loss = const * c_loss_posterior
			else:
				self.c_loss = const * (c_loss_prior + c_loss_posterior)

			self.p_loss = tf.reduce_sum(0.5 * tf.square(self.pert_image - input))
			loss = self.c_loss + self.p_loss

			optimizer = optimizer.upper()
			if optimizer == 'ADAM':
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			elif optimizer == 'RMSPROP':
				optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=.9, epsilon=1e-6)
			else:
				raise NotImplementedError('Unsupported optimizer.')

			self.train_step = optimizer.minimize(loss, var_list=[pert_atanh])

			# Reset optimizer
			rst_opt = tf.initialize_variables(var_list=optimizer.variables())
			self.sess.run(rst_opt)

			# For normal prediction
			self.res = self.model({'image': self.input})

			# Calculate object capsule subset
			prior_pres_clean = self.res.caps_presence_prob
			posterior_pres_clean = tf.reduce_sum(self.res.posterior_mixing_probs, 1)

			# Init variables for optimization
			self.init = [
				tf.assign(pert_atanh, tf.random.uniform(input_size)),
				tf.assign(input, self.input),
				tf.assign(mask, self.mask),
				tf.assign(const, self.const),
				rst_opt,
				tf.assign(subset_prior_position,
				          tf.where(prior_pres_clean > tf.reduce_mean(prior_pres_clean))[:, 1],
				          validate_shape=False),
				tf.assign(subset_posterior_position,
				          tf.where(posterior_pres_clean > tf.reduce_mean(posterior_pres_clean))[:, 1],
				          validate_shape=False)
			]

			# Restore params of model from snapshot
			saver = tf.train.Saver(var_list=tf.trainable_variables(scope=scope))
			if snapshot:
				print('Restoring from snapshot: {}'.format(snapshot))
				saver.restore(self.sess, snapshot)
			else:
				raise Exception('Snapshot of pretrained model must be given.')

			# Freeze graph
			self.sess.graph.finalize()

	def run(self, images, to_collect):
		return self.sess.run(to_collect, feed_dict={self.input: images})

	def __call__(self, images):
		return self.sess.run(self.res.prior_cls_logits, feed_dict={self.input: images})


def compare(results: list, label: int, is_targeted: bool) -> bool:
	if is_targeted:
		if len(set(results)) == 1 and results[0] == label:
			return True
		return False
	else:
		if label not in results:
			return True
		return False


optimizer_configs = {
	'RMSPROP_fast': [300, 1e-1, 'RMSPROP'],
	'RMSPROP_normal': [1000, 1e-1, 'RMSPROP'],
	'RMSPROP_complex': [2000, 1e-1, 'RMSPROP'],
	'ADAM_fast': [300, 1, 'ADAM'],
	'ADAM_normal': [1000, 1e-1, 'ADAM'],
	'ADAM_complex': [2000, 1e-2, 'ADAM']
}

if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	dataset = 'mnist'
	optimizer_config_name = 'ADAM_normal'
	num_samples = 500
	classifiers = 'PriPosK'

	# classifiers should be set as [Pri|Pos][K|L]
	# For example: PriK means prior K-Means classifier. PosL means posterior linear classifier.
	#              PriPosKL means prior & posterior K-Means classifiers and prior & posterior linear classifiers.

	# Create result directory
	now = time.localtime()
	path = './results/{}_{}_{}_{}_{}_{}/'.format(
		classifiers,
		now.tm_year,
		now.tm_mon,
		now.tm_mday,
		now.tm_hour,
		now.tm_min
	)
	if not os.path.exists(path + 'images/'):
		os.makedirs(path + 'images/')

	# Model parameters. Edit them ONLY IF YOU KNOW WHAT YOU ARE DOING!
	canvas_size = 28
	n_part_caps = 40
	n_obj_caps = 32
	n_channels = 1
	colorize_templates = True
	use_alpha_channel = True
	prior_within_example_sparsity_weight = 2.
	prior_between_example_sparsity_weight = 0.35
	posterior_within_example_sparsity_weight = 0.7
	posterior_between_example_sparsity_weight = 0.2
	template_nonlin = 'sigmoid'
	color_nonlin = 'sigmoid'
	snapshot = './checkpoints/{}/model.ckpt'.format(dataset)
	max_train_steps, learning_rate, optimizer = optimizer_configs[optimizer_config_name]

	# Create the attack model according to parameters above
	model = SCAE_L2_Attack(
		input_size=[1, canvas_size, canvas_size, n_channels],
		num_classes=10,
		n_part_caps=n_part_caps,
		n_obj_caps=n_obj_caps,
		n_channels=n_channels,
		colorize_templates=colorize_templates,
		use_alpha_channel=use_alpha_channel,
		prior_within_example_sparsity_weight=prior_within_example_sparsity_weight,
		prior_between_example_sparsity_weight=prior_between_example_sparsity_weight,
		posterior_within_example_sparsity_weight=posterior_within_example_sparsity_weight,
		posterior_between_example_sparsity_weight=posterior_between_example_sparsity_weight,
		template_nonlin=template_nonlin,
		color_nonlin=color_nonlin,
		learning_rate=learning_rate,
		optimizer=optimizer,
		scope='SCAE',
		snapshot=snapshot,
		capsule_loss_type=('Pri' if 'Pri' in classifiers else '') + ('Pos' if 'Pos' in classifiers else '')
	)

	# Load prior K-Means classifier
	if 'Pri' in classifiers and 'K' in classifiers:
		kmeans_pri = joblib.load('./checkpoints/{}/kmeans_prior.m'.format(dataset))
		npz = np.load('./checkpoints/{}/kmeans_labels_prior.npz'.format('mnist'))
		p2l_pri = npz['preds_2_labels']
		npz.close()

	# Load posterior K-Means classifier
	if 'Pos' in classifiers and 'K' in classifiers:
		kmeans_pos = joblib.load('./checkpoints/{}/kmeans_posterior.m'.format(dataset))
		npz = np.load('./checkpoints/{}/kmeans_labels_posterior.npz'.format('mnist'))
		p2l_pos = npz['preds_2_labels']
		npz.close()

	# Load dataset
	testset = get_dataset(dataset, 'test', shape=[canvas_size, canvas_size], file_path='./datasets')

	# Variables to save the attack result
	succeed_count = 0
	succeed_pert_amount = 0.

	# Shuffle the order of samples
	shuffle_indices = list(range(len(testset['image'])))
	random.seed(time.time())
	random.shuffle(shuffle_indices)

	# Score dict for optimization
	score_to_collect = AttrDict()
	if 'Pri' in classifiers and 'K' in classifiers:
		score_to_collect['PriK'] = model.pert_res.caps_presence_prob
	if 'Pos' in classifiers and 'K' in classifiers:
		score_to_collect['PosK'] = model.pert_res.posterior_mixing_probs
	if 'Pri' in classifiers and 'L' in classifiers:
		score_to_collect['PriL'] = model.pert_res.prior_cls_pred[0]
	if 'Pos' in classifiers and 'L' in classifiers:
		score_to_collect['PosL'] = model.pert_res.posterior_cls_pred[0]

	# Score dict for validation
	score_to_collect_validation = AttrDict()
	if 'Pri' in classifiers and 'K' in classifiers:
		score_to_collect_validation['PriK'] = model.res.caps_presence_prob
	if 'Pos' in classifiers and 'K' in classifiers:
		score_to_collect_validation['PosK'] = model.res.posterior_mixing_probs
	if 'Pri' in classifiers and 'L' in classifiers:
		score_to_collect_validation['PriL'] = model.res.prior_cls_pred[0]
	if 'Pos' in classifiers and 'L' in classifiers:
		score_to_collect_validation['PosL'] = model.res.posterior_cls_pred[0]

	# Start the attack on selected samples
	for index in shuffle_indices[:num_samples]:
		source_image = to_float32(testset['image'][index])
		source_label = testset['label'][index]

		# Calculate mask
		mask = imblur(source_image, times=1)

		# Set constant
		lower_bound = 0
		upper_bound = np.inf
		const = 1e2

		# The best pert amount and pert image
		global_best_p_loss = np.inf
		global_best_pert_image = None

		# Outer iteration
		dynamic_desc_steps = trange(9, desc='Image {}'.format(index))
		for outer_iter in dynamic_desc_steps:
			# Init the original image, mask and constant
			model.sess.run(model.init, feed_dict={model.input: source_image[None],
			                                      model.mask: mask[None],
			                                      model.const: const})

			# Flag for constant update
			flag_hit_succeed = False

			# Inner iteration
			for inner_iter in range(max_train_steps):
				# Run optimizer
				model.sess.run(model.train_step)

				# Get the current loss
				c_loss, p_loss = model.sess.run([model.c_loss, model.p_loss])

				if np.isnan(p_loss):
					# When encountered nan, there is no need to continue.
					best_p_loss = np.nan
					break

				# Collect scores
				score_list = model.sess.run(score_to_collect)
				if 'PriK' in score_to_collect.keys():
					score_list['PriK'] = p2l_pri[kmeans_pri.predict(score_list['PriK'])[0]]
				if 'PosK' in score_to_collect.keys():
					score_list['PosK'] = p2l_pos[kmeans_pos.predict(score_list['PosK'].sum(1))[0]]

				# Determine if succeed
				succeed = compare(list(score_list.values()), source_label, is_targeted=False)

				# Update flag
				if not flag_hit_succeed and succeed:
					flag_hit_succeed = True

				# Update global best result
				if succeed and p_loss < global_best_p_loss:
					global_best_p_loss = p_loss
					global_best_pert_image = model.sess.run(model.pert_image)[0]

				# Update tqdm description
				dynamic_desc_steps.set_postfix_str('c_l: {:.2f}, p_l: {:.2f}, best: {:.2f}'
				                                   .format(c_loss, p_loss, global_best_p_loss))

			# Update constant
			if flag_hit_succeed:
				upper_bound = const
				const = (lower_bound + upper_bound) / 2
			else:
				lower_bound = const
				if np.isinf(upper_bound):
					const *= 10
				else:
					const = (lower_bound + upper_bound) / 2

		# Start validation
		if global_best_pert_image is None:
			# Print result
			print('Failed for {}. Source label: {}.'.format(index, source_label))

		else:
			# Validation for success
			assert True not in np.isnan(global_best_pert_image)

			# L2 distance between pert_image and source_image
			pert_amount = np.square(global_best_pert_image - source_image).sum() ** (1 / 2)

			# Collect scores
			score_list_validation = model.run(global_best_pert_image[None], score_to_collect_validation)
			if 'PriK' in score_to_collect_validation.keys():
				score_list_validation['PriK'] = p2l_pri[kmeans_pri.predict(score_list_validation['PriK'])[0]]
			if 'PosK' in score_to_collect_validation.keys():
				score_list_validation['PosK'] = p2l_pos[kmeans_pos.predict(score_list_validation['PosK'].sum(1))[0]]

			# Determine if succeed
			assert compare(list(score_list_validation.values()), source_label, is_targeted=False)

			# Add info of the succeed sample to result variables
			succeed_count += 1
			succeed_pert_amount += pert_amount

			# Print result
			print('Succeed for {}. Source label: {}. Pert amount: {:.2f}'
			      .format(index, source_label, pert_amount))
			print(score_list_validation)

		print()

		# Save the pert image
		np.savez_compressed(path + 'images/{}.npz'.format(index), pert_image=global_best_pert_image)

	# Save the final result of complete attack
	result = 'Optimizer configuration: {}. Success rate: {:.4f}. Average pert amount: {:.4f}.'.format(
		optimizer_config_name, succeed_count / num_samples, succeed_pert_amount / succeed_count)
	print(result)
	if os.path.exists(path + 'result.txt'):
		os.remove(path + 'result.txt')
	result_file = open(path + 'result.txt', mode='x')
	result_file.write(result)
	result_file.close()
