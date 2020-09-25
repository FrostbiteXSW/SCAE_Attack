from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import joblib

from model import stacked_capsule_autoencoders
from utilities import *


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
			stop_gradient=True,
			prior_within_example_sparsity_weight=1.,
			prior_between_example_sparsity_weight=1.,
			posterior_within_example_sparsity_weight=10.,
			posterior_between_example_sparsity_weight=10.,
			learning_rate=1e-4,
			optimizer='Adam',
			scope='SCAE',
			snapshot=None
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
			self.target_prior_position = tf.placeholder(tf.int32)
			self.target_posterior_position = tf.placeholder(tf.int32)
			self.const = tf.placeholder(tf.float32, [])

			# For normal prediction
			self.res = self.model({'image': self.input})

			pert_atanh = tf.Variable(tf.zeros(input_size))
			input = tf.Variable(tf.zeros(input_size), trainable=False)
			input_atanh = tf.atanh((input - 0.5) / 0.5 * 0.999999)
			mask = tf.Variable(tf.zeros(input_size), trainable=False)
			target_prior_position = tf.Variable(tf.placeholder(tf.int32), trainable=False, validate_shape=False)
			target_posterior_position = tf.Variable(tf.placeholder(tf.int32), trainable=False, validate_shape=False)
			const = tf.Variable(tf.zeros([]), trainable=False)

			self.pert_image = 0.5 * (tf.tanh(pert_atanh * mask + input_atanh) + 1)
			self.pert_res = self.model({'image': self.pert_image})

			c_loss_prior = tf.reduce_sum(0.5 * tf.square(
				tf.gather(self.pert_res.caps_presence_prob, target_prior_position, axis=1)))
			c_loss_posterior = tf.reduce_sum(0.5 * tf.square(
				tf.gather(tf.reduce_sum(self.pert_res.posterior_mixing_probs, axis=1), target_posterior_position, axis=1)))
			self.c_loss = const * (c_loss_prior + c_loss_posterior / (n_part_caps ** 2))
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

			# Init variables for optimization
			self.init = [
				tf.assign(pert_atanh, tf.random.uniform(input_size)),
				tf.assign(input, self.input),
				tf.assign(mask, self.mask),
				tf.assign(target_prior_position, self.target_prior_position, validate_shape=False),
				tf.assign(target_posterior_position, self.target_posterior_position, validate_shape=False),
				tf.assign(const, self.const),
				rst_opt
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


configs = {
	'RMSPROP_fast': [300, 1e-1, 'RMSPROP'],
	'RMSPROP_normal': [1000, 1e-1, 'RMSPROP'],
	'RMSPROP_complex': [2000, 1e-1, 'RMSPROP'],
	'ADAM_fast': [300, 1, 'ADAM'],
	'ADAM_normal': [1000, 1e-1, 'ADAM'],
	'ADAM_complex': [2000, 1e-2, 'ADAM']
}

if __name__ == '__main__':
	block_warnings()

	dataset = 'mnist'
	config_name = 'RMSPROP_fast'
	max_train_steps, learning_rate, optimizer = configs[config_name]
	attack_unsupervised, attack_supervised = True, False
	num_samples = 5000

	assert attack_supervised or attack_unsupervised

	cfg_str = ''
	if attack_unsupervised:
		cfg_str += 'k'
	if attack_supervised:
		cfg_str += 'l'

	now = time.localtime()
	path = './results/{}_{}_{}_{}_{}_{}/'.format(
		cfg_str,
		now.tm_year,
		now.tm_mon,
		now.tm_mday,
		now.tm_hour,
		now.tm_min
	)
	if not os.path.exists(path + 'images/'):
		os.makedirs(path + 'images/')

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

	model = SCAE(
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
		snapshot=snapshot
	)

	if attack_unsupervised:
		kmeans_pri = joblib.load('./checkpoints/{}/kmeans_prior.m'.format(dataset))
		npz = np.load('./checkpoints/{}/kmeans_labels_prior.npz'.format('mnist'))
		p2l_pri = npz['preds_2_labels']
		npz.close()

		kmeans_pos = joblib.load('./checkpoints/{}/kmeans_posterior.m'.format(dataset))
		npz = np.load('./checkpoints/{}/kmeans_labels_posterior.npz'.format('mnist'))
		p2l_pos = npz['preds_2_labels']
		npz.close()

	testset = get_dataset(dataset, 'test', shape=[canvas_size, canvas_size], file_path='./datasets')

	succeed_count = 0
	succeed_pert_amount = 0.

	shuffle_indices = list(range(len(testset['image'])))
	random.seed(time.time())
	random.shuffle(shuffle_indices)

	for index in shuffle_indices[:num_samples]:
		source_image = to_float32(testset['image'][index])
		source_label = testset['label'][index]

		prior_pres, posterior_pres = model.run(source_image[None],
		                                       [model.res.caps_presence_prob,
		                                        model.res.posterior_mixing_probs])

		prior_pres_position = np.where(prior_pres > prior_pres.mean())[1]
		posterior_pres = posterior_pres.sum(1)
		posterior_pres_position = np.where(posterior_pres > posterior_pres.mean())[1]

		# Calculate mask
		mask = imblur(source_image, times=1)

		# Set const value
		lower_bound = 0
		upper_bound = np.inf
		const = 1e2

		# The best pert amount and pert image
		global_best_p_loss = np.inf
		global_best_pert_image = None

		# Outer iteration
		dynamic_desc_steps = trange(9, desc='Image {}'.format(index))
		for _ in dynamic_desc_steps:
			model.sess.run(model.init, feed_dict={model.input: source_image[None],
			                                      model.mask: mask[None],
			                                      model.target_prior_position: prior_pres_position,
			                                      model.target_posterior_position: posterior_pres_position,
			                                      model.const: const})

			best_p_loss = np.inf

			# Inner iteration
			for epoch in range(max_train_steps):
				model.sess.run(model.train_step)

				c_loss, p_loss = model.sess.run([model.c_loss, model.p_loss])

				if np.isnan(p_loss):
					# When encountered nan, there is no need to continue.
					best_p_loss = np.nan
					break

				score_list = []

				if attack_unsupervised:
					prior_pres, posterior_pres = model.sess.run([model.pert_res.caps_presence_prob,
					                                             model.pert_res.posterior_mixing_probs])
					score_list.append(p2l_pri[kmeans_pri.predict(prior_pres)[0]])
					score_list.append(p2l_pos[kmeans_pos.predict(posterior_pres.sum(1))[0]])

				if attack_supervised:
					score_prior, score_posterior = model.sess.run([model.pert_res.prior_cls_pred,
					                                               model.pert_res.posterior_cls_pred])
					score_list.append(score_prior[0])
					score_list.append(score_posterior[0])

				succeed = compare(score_list, source_label, is_targeted=False)

				if succeed and p_loss < best_p_loss:
					best_p_loss = p_loss

				if succeed and p_loss < global_best_p_loss:
					global_best_p_loss = p_loss
					global_best_pert_image = model.sess.run(model.pert_image)[0]

				dynamic_desc_steps.set_postfix_str('c_l: {:.2f}, p_l: {:.2f}, best: {:.2f}'
				                                   .format(c_loss, p_loss, global_best_p_loss))

			if not np.isinf(best_p_loss):
				upper_bound = const
				const = (lower_bound + upper_bound) / 2
			else:
				lower_bound = const
				if np.isinf(upper_bound):
					const *= 10
				else:
					const = (lower_bound + upper_bound) / 2

		if global_best_pert_image is None:
			succeed = False
			print('Failed for {}. Source label: {}.'.format(index, source_label))
		else:
			# Validation
			assert True not in np.isnan(global_best_pert_image)

			# L2 distance between pert_image and source_image
			pert_amount = np.square(global_best_pert_image - source_image).sum() ** (1 / 2)

			score_list = []

			if attack_unsupervised:
				prior_pres, posterior_pres = model.run(global_best_pert_image[None],
				                                       [model.res.caps_presence_prob,
				                                        model.res.posterior_mixing_probs])
				score_list.append(p2l_pri[kmeans_pri.predict(prior_pres)[0]])
				score_list.append(p2l_pos[kmeans_pos.predict(posterior_pres.sum(1))[0]])

			if attack_supervised:
				score_prior, score_posterior = model.run(global_best_pert_image[None],
				                                         [model.res.prior_cls_pred,
				                                          model.res.posterior_cls_pred])
				score_list.append(score_prior[0])
				score_list.append(score_posterior[0])

			succeed = compare(score_list, source_label, is_targeted=False)
			assert succeed

			succeed_count += 1
			succeed_pert_amount += pert_amount

			print('Succeed for {}. Source label: {}. Pert amount: {:.2f}'
			      .format(index, source_label, pert_amount))
			if attack_unsupervised:
				print('Unsupervised result: prior={}, posterior={}'.format(score_list[0], score_list[1]))
			if attack_supervised:
				print('Supervised result: prior={}, posterior={}'.format(score_list[-2], score_list[-1]))

		print()

		np.savez_compressed(path + 'images/{}_{}.npz'
		                    .format(index, 'S' if succeed else 'F'), pert_image=global_best_pert_image)

	result = 'Optimizer configuration: {}. Success rate: {:.4f}. Average pert amount: {:.4f}'.format(
		config_name, succeed_count / num_samples, succeed_pert_amount / succeed_count)
	print(result)
	if os.path.exists(path + 'result.txt'):
		os.remove(path + 'result.txt')
	result_file = open(path + 'result.txt', mode='x')
	result_file.write(result)
	result_file.close()
