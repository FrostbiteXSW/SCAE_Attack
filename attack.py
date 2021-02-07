from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib

from train import *
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
	config = config_gtsrb
	optimizer_config_name = 'ADAM_fast'
	num_samples = 1000
	outer_iteration = 9
	classifiers = 'PriPosK'

	# classifiers should be set as [Pri|Pos][K|L]
	# For example: PriK means prior K-Means classifier. PosL means posterior linear classifier.
	#              PriPosKL means prior & posterior K-Means classifiers and prior & posterior linear classifiers.

	snapshot = './checkpoints/{}/model.ckpt'.format(config['dataset'])
	inner_iteration, learning_rate, optimizer = optimizer_configs[optimizer_config_name]

	# Create the attack model according to parameters above
	model = SCAE_L2_Attack(
		input_size=[1, config['canvas_size'], config['canvas_size'], config['n_channels']],
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
		part_encoder_noise_scale=0.,
		obj_decoder_noise_type=None,
		obj_decoder_noise_scale=0.,
		set_transformer_n_layers=config['set_transformer_n_layers'],
		set_transformer_n_heads=config['set_transformer_n_heads'],
		set_transformer_n_dims=config['set_transformer_n_dims'],
		set_transformer_n_output_dims=config['set_transformer_n_output_dims'],
		part_cnn_strides=config['part_cnn_strides'],
		prep=config['prep'],
		learning_rate=learning_rate,
		optimizer=optimizer,
		capsule_loss_type=('Pri' if 'Pri' in classifiers else '') + ('Pos' if 'Pos' in classifiers else ''),
		scope='SCAE',
		snapshot=snapshot
	)

	# Load prior K-Means classifier
	if 'Pri' in classifiers and 'K' in classifiers:
		kmeans_pri = joblib.load('./checkpoints/{}/kmeans_prior.m'.format(config['dataset']))
		npz = np.load('./checkpoints/{}/kmeans_labels_prior.npz'.format(config['dataset']))
		p2l_pri = npz['preds_2_labels']
		npz.close()

	# Load posterior K-Means classifier
	if 'Pos' in classifiers and 'K' in classifiers:
		kmeans_pos = joblib.load('./checkpoints/{}/kmeans_posterior.m'.format(config['dataset']))
		npz = np.load('./checkpoints/{}/kmeans_labels_posterior.npz'.format(config['dataset']))
		p2l_pos = npz['preds_2_labels']
		npz.close()

	# Load dataset
	if config['dataset'] == 'gtsrb':
		testset = get_gtsrb('train', shape=[config['canvas_size'], config['canvas_size']], file_path='./datasets',
		                    save_only=False, gtsrb_raw_file_path='./datasets/GTSRB', gtsrb_classes=config['classes'])
	else:
		testset = get_dataset(config['dataset'], 'test', shape=[config['canvas_size'], config['canvas_size']],
		                      file_path='./datasets', save_only=False)

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
		score_to_collect['PriL'] = model.pert_res.prior_cls_pred
	if 'Pos' in classifiers and 'L' in classifiers:
		score_to_collect['PosL'] = model.pert_res.posterior_cls_pred

	# Score dict for validation
	score_to_collect_validation = AttrDict()
	if 'Pri' in classifiers and 'K' in classifiers:
		score_to_collect_validation['PriK'] = model.res.caps_presence_prob
	if 'Pos' in classifiers and 'K' in classifiers:
		score_to_collect_validation['PosK'] = model.res.posterior_mixing_probs
	if 'Pri' in classifiers and 'L' in classifiers:
		score_to_collect_validation['PriL'] = model.res.prior_cls_pred
	if 'Pos' in classifiers and 'L' in classifiers:
		score_to_collect_validation['PosL'] = model.res.posterior_cls_pred

	# Start the attack on selected samples
	i = 0
	n = num_samples
	while n > 0:
		index = shuffle_indices[i]
		i += 1

		source_image = to_float32(testset['image'][index])
		source_label = testset['label'][index]

		# Skip unrecognized samples
		score_list_validation = model.run(source_image[None], score_to_collect_validation)
		if 'PriK' in score_to_collect_validation.keys() \
				and p2l_pri[kmeans_pri.predict(score_list_validation['PriK'])[0]] != source_label \
				or 'PosK' in score_to_collect_validation.keys() \
				and p2l_pos[kmeans_pos.predict(score_list_validation['PosK'].sum(1))[0]] != source_label \
				or 'PriL' in score_to_collect_validation.keys() \
				and score_list_validation['PriL'][0] != source_label \
				or 'PosL' in score_to_collect_validation.keys() \
				and score_list_validation['PosL'][0] != source_label:
			print("Skipping sample {}.".format(index))
			continue
		n -= 1

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
		dynamic_desc_steps = trange(outer_iteration, desc='Image {}'.format(index))
		for outer_iter in dynamic_desc_steps:
			# Init the original image, mask and constant
			model.sess.run(model.init, feed_dict={model.input: source_image[None],
			                                      model.mask: mask[None],
			                                      model.const: const})

			# Flag for constant update
			flag_hit_succeed = False

			# Inner iteration
			for inner_iter in range(inner_iteration):
				# Run optimizer
				model.sess.run(model.train_step)

				# Get the current loss
				c_loss, p_loss = model.sess.run([model.c_loss, model.p_loss])

				if np.isnan(p_loss):
					# When encountered nan, there is no need to continue.
					break

				# Collect scores
				score_list = model.sess.run(score_to_collect)
				if 'PriK' in score_to_collect.keys():
					score_list['PriK'] = p2l_pri[kmeans_pri.predict(score_list['PriK'])[0]]
				if 'PosK' in score_to_collect.keys():
					score_list['PosK'] = p2l_pos[kmeans_pos.predict(score_list['PosK'].sum(1))[0]]
				if 'PriL' in score_to_collect.keys():
					score_list['PriL'] = score_list['PriL'][0]
				if 'PosL' in score_to_collect.keys():
					score_list['PosL'] = score_list['PosL'][0]

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
			if 'PriL' in score_to_collect_validation.keys():
				score_list_validation['PriL'] = score_list_validation['PriL'][0]
			if 'PosL' in score_to_collect_validation.keys():
				score_list_validation['PosL'] = score_list_validation['PosL'][0]

			# Determine if succeed
			assert compare(list(score_list_validation.values()), source_label, is_targeted=False)

			# Add info of the succeed sample to result variables
			succeed_count += 1
			succeed_pert_amount += pert_amount

			# Print result
			print('Succeed for {}. Source label: {}. Pert amount: {:.2f}'
			      .format(index, source_label, pert_amount))
			print(score_list_validation)

			# Save the pert image
			np.savez_compressed(path + 'images/{}.npz'.format(index), pert_image=global_best_pert_image)

		print()

	# Save the final result of complete attack
	result = 'Optimizer configuration: {}. Dataset: {}. Success rate: {:.4f}. Average pert amount: {:.4f}.'.format(
		optimizer_config_name, config['dataset'], succeed_count / num_samples, succeed_pert_amount / succeed_count)
	print(result)
	if os.path.exists(path + 'result.txt'):
		os.remove(path + 'result.txt')
	result_file = open(path + 'result.txt', mode='x')
	result_file.write(result)
	result_file.close()
