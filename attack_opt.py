import numpy as np
import tensorflow as tf
from tqdm import trange

from tools.model import Attacker, ScaeBasement, KMeans
from tools.utilities import block_warnings, imblur, DatasetHelper, ResultBuilder
from train import Configs, build_from_config


class AttackerOPT(Attacker):
	class OptimizerConfigs:
		RMSProp_fast = [9, 300, 1e-1, 'RMSProp']
		RMSProp_normal = [9, 1000, 1e-1, 'RMSProp']
		RMSProp_complex = [9, 2000, 1e-1, 'RMSProp']
		Adam_fast = [9, 300, 1, 'Adam']
		Adam_normal = [9, 1000, 1e-1, 'Adam']
		Adam_complex = [9, 2000, 1e-2, 'Adam']
		FGSM_fast = [3, 10, 2.5, 'FGSM']
		FGSM_normal = [5, 30, 1, 'FGSM']

	def __init__(
			self,
			scae: ScaeBasement,
			optimizer_config: list,
			classifier: str,
			kmeans_classifier: KMeans = None,
			const_init: float = 1e2
	):
		self._const_init = const_init

		outer_iteration, inner_iteration, learning_rate, optimizer = optimizer_config

		if optimizer not in ['RMSProp', 'Adam', 'FGSM']:
			print('Specified optimizer "{}" is not implemented. Fallback optimizer "FGSM" will be used.'.format(optimizer))

		self._classifier = classifier
		self._outer_iteration = outer_iteration
		self._inner_iteration = inner_iteration

		self._sess = scae._sess
		self._input_size = scae._input_size
		self._valid_shape = scae._valid_shape

		if 'K' in classifier:
			if kmeans_classifier is None:
				raise Exception('Param \"kmeans_classifier\" must be specified.')
			self._kmeans_classifier = kmeans_classifier

		# Build graph
		with self._sess.graph.as_default():
			# Placeholders for variables to initialize
			self._ph_input = scae._input
			self._ph_mask = tf.placeholder(tf.float32, self._input_size)
			self._ph_const = tf.placeholder(tf.float32, self._input_size[0])

			n_part_caps = int(scae._res.posterior_mixing_probs.shape[1])
			n_obj_caps = int(scae._res.posterior_mixing_probs.shape[2])

			# Variables to be assigned during initialization
			self._pert_atanh = tf.Variable(tf.zeros(self._input_size))
			self._input = tf.Variable(tf.zeros(self._input_size), trainable=False)
			self._input_atanh = tf.atanh((self._input - 0.5) / 0.5 * 0.999999)
			self._mask = tf.Variable(tf.zeros(self._input_size), trainable=False)
			self._const = tf.Variable(tf.zeros(self._input_size[0]), trainable=False)
			self._subset_position = tf.Variable(tf.zeros([self._input_size[0], n_obj_caps]), trainable=False)

			self._pert_images = 0.5 * (tf.tanh(self._pert_atanh * self._mask + self._input_atanh) + 1)
			pert_res = scae._model({'image': self._pert_images})

			if classifier[:3].upper() == 'PRI':
				object_capsule_set = pert_res.caps_presence_prob
				object_capsule_subset = object_capsule_set * self._subset_position
				self._c_loss = self._const * tf.nn.l2_loss(object_capsule_subset)
			elif classifier[:3].upper() == 'POS':
				object_capsule_set = tf.reduce_sum(pert_res.posterior_mixing_probs, axis=1)
				object_capsule_subset = object_capsule_set * self._subset_position
				self._c_loss = self._const * tf.nn.l2_loss(object_capsule_subset) / (n_part_caps ** 2)
			else:
				raise NotImplementedError('Unsupported capsule loss type.')

			self._p_loss = tf.reduce_sum(0.5 * tf.square(self._pert_images - self._input), axis=[1, 2, 3])
			loss = self._c_loss + tf.reduce_sum(self._p_loss)

			# Initialization operation
			self._init = [
				tf.assign(self._pert_atanh, tf.random.uniform(self._input_size)),
				tf.assign(self._input, self._ph_input),
				tf.assign(self._mask, self._ph_mask),
				tf.assign(self._const, self._ph_const)
			]

			# Init optimizer
			optimizer = optimizer.upper()
			if optimizer == 'RMSPROP':
				optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=.9, epsilon=1e-6)
				self._train_step = optimizer.minimize(loss, var_list=[self._pert_atanh])
				self._init.append(tf.initialize_variables(var_list=optimizer.variables()))
			elif optimizer == 'ADAM':
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
				self._train_step = optimizer.minimize(loss, var_list=[self._pert_atanh])
				self._init.append(tf.initialize_variables(var_list=optimizer.variables()))
			elif optimizer == 'FGSM':
				self._train_step = tf.assign(self._pert_atanh,
				                             self._pert_atanh - learning_rate * tf.sign(
					                             tf.gradients(loss, self._pert_atanh)[0]))
			else:
				raise NotImplementedError('Unsupported optimizer.')

			# Compute capsule subset position
			if classifier[:3].upper() == 'PRI':
				pres_clean = scae._res.caps_presence_prob
			else:
				pres_clean = tf.reduce_sum(scae._res.posterior_mixing_probs, axis=1)
			self._init.append(tf.assign(self._subset_position,
			                            tf.where(pres_clean > tf.reduce_mean(pres_clean),
			                                     x=tf.ones_like(pres_clean),
			                                     y=tf.zeros_like(pres_clean))))

			# Score dict for optimization
			self._score = object_capsule_set if classifier[-1].upper() == 'K' \
				else pert_res.prior_cls_pred if classifier == Attacker.Classifiers.PriL \
				else pert_res.posterior_cls_pred

	def __call__(
			self,
			images: np.ndarray,
			labels: np.ndarray,
			nan_if_fail: bool = False,
			verbose: bool = False,
			use_mask: bool = True,
			**mask_kwargs
	):
		# Shape Validation
		images, num_images, labels = self._valid_shape(images, labels)

		# Calculate mask
		mask = imblur(images, **mask_kwargs) if use_mask else np.ones_like(images)

		# Set constant
		lower_bound = np.zeros([num_images])
		upper_bound = np.full([num_images], np.inf)
		const = np.full([self._input_size[0]], self._const_init)

		# The best pert amount and pert image
		global_best_p_loss = np.full([num_images], np.inf)
		global_best_pert_images = np.full([num_images, *self._input_size[1:]], np.nan) \
			if nan_if_fail else images[:num_images].copy()

		# Outer iteration
		for _ in (trange(self._outer_iteration) if verbose else range(self._outer_iteration)):
			# Init the original image, mask and constant
			self._sess.run(self._init, feed_dict={self._ph_input: images,
			                                      self._ph_mask: mask,
			                                      self._ph_const: const})

			# Flag for constant update
			flag_hit_succeed = np.zeros([num_images])

			# Inner iteration
			for inner_iter in range(self._inner_iteration):
				# Run optimizer
				self._sess.run(self._train_step)

				# Collect scores
				results, p_loss = self._sess.run([self._score, self._p_loss])

				if True in np.isnan(p_loss):
					# When encountered nan, there is no need to continue.
					break

				if self._classifier[-1].upper() == 'K':
					results = self._kmeans_classifier.run(results, self._kmeans_classifier._output)

				# Determine if succeed
				succeed = results != labels

				# Update flag
				flag_hit_succeed += succeed[:num_images]

				# Update global best result
				pert_images = self._sess.run(self._pert_images)
				for i in range(num_images):
					if succeed[i] and p_loss[i] < global_best_p_loss[i]:
						global_best_pert_images[i] = pert_images[i]
						global_best_p_loss[i] = p_loss[i]

			# Update constant
			upper_bound = np.where(flag_hit_succeed, const[:num_images], upper_bound)
			lower_bound = np.where(flag_hit_succeed, lower_bound, const[:num_images])
			const[:num_images] = np.where(np.isinf(upper_bound), const[:num_images] * 10, (lower_bound + upper_bound) / 2)

		return global_best_pert_images


if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	optimizer_config = AttackerOPT.OptimizerConfigs.Adam_fast
	num_samples = 1000
	batch_size = 100
	classifier = Attacker.Classifiers.PosK
	use_mask = True

	snapshot = './checkpoints/{}/model.ckpt'.format(config['name'])
	snapshot_kmeans = './checkpoints/{}/kmeans_{}/model.ckpt'.format(
		config['name'], 'pri' if classifier[:3].upper() == 'PRI' else 'pos')

	# Create the attack model according to parameters above
	model = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot
	)

	if classifier[-1].upper() == 'K':
		kmeans = KMeans(
			scae=model,
			kmeans_type=KMeans.KMeansTypes.Prior if classifier[:3].upper() == 'PRI' else KMeans.KMeansTypes.Posterior,
			is_training=False,
			scope='KMeans_Pri' if classifier[:3].upper() == 'PRI' else 'KMeans_Pos',
			snapshot=snapshot_kmeans
		)

	attacker = AttackerOPT(
		scae=model,
		optimizer_config=optimizer_config,
		classifier=classifier,
		kmeans_classifier=kmeans if classifier[-1].upper() == 'K' else None
	)

	model.finalize()

	# Load dataset
	dataset = DatasetHelper(config['dataset'],
	                        'train' if config['dataset'] == Configs.GTSRB
	                                   or config['dataset'] == Configs.FASHION_MNIST else 'test',
	                        file_path='./datasets', batch_size=batch_size, shuffle=True, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES,
	                        builder_kwargs=config.get('builder_kwargs', None))

	# Variables to save the attack result
	succeed_count = 0
	succeed_pert_amount = []
	succeed_pert_robustness = []
	source_images = []
	pert_images = []

	# Classification accuracy test
	model.simple_test(dataset)

	# Start the attack on selected samples
	dataset = iter(dataset)
	remain = num_samples
	while remain > 0:
		images, labels = next(dataset)

		# Judge classification
		if classifier[-1].upper() == 'K':
			right_classification = kmeans(images) == labels
		else:
			right_classification = model.run(
				images=images,
				to_collect=model._res.prior_cls_pred if classifier == Attacker.Classifiers.PriL
				else model._res.posterior_cls_pred
			) == labels

		attacker_outputs = attacker(images, labels, nan_if_fail=True, verbose=True)

		for i in range(len(attacker_outputs)):
			if right_classification[i] and remain:
				remain -= 1
				if True not in np.isnan(attacker_outputs[i]):
					# L2 distance between pert_image and source_image
					pert_amount = np.linalg.norm(attacker_outputs[i] - images[i])
					pert_robustness = pert_amount / np.linalg.norm(images[i])

					succeed_count += 1
					succeed_pert_amount.append(pert_amount)
					succeed_pert_robustness.append(pert_robustness)

					source_images.append(images[i])
					pert_images.append(attacker_outputs[i])

		print('Up to now: Success rate: {:.4f}. Average pert amount: {:.4f}. Remain: {}.'.format(
			succeed_count / (num_samples - remain), np.array(succeed_pert_amount, dtype=np.float32).mean(), remain))

	# Change list into numpy array
	succeed_pert_amount = np.array(succeed_pert_amount, dtype=np.float32)
	succeed_pert_robustness = np.array(succeed_pert_robustness, dtype=np.float32)

	# Save the final result of complete attack
	result = ResultBuilder()
	result['Dataset'] = config['dataset']
	result['Classifier'] = classifier
	result['Num of samples'] = num_samples

	# Attacker params
	result['Optimizer config'] = optimizer_config

	# Attack results
	result['Success rate'] = succeed_count / num_samples
	result['Average pert amount'] = succeed_pert_amount.mean()
	result['Pert amount standard deviation'] = succeed_pert_amount.std()
	result['Average pert robustness'] = succeed_pert_robustness.mean()
	result['Pert robustness standard deviation'] = succeed_pert_robustness.std()

	# Print and save results
	print(result)
	path = result.save('./results/opt/')
	np.savez_compressed(path + 'source_images.npz', source_images=np.array(source_images, dtype=np.float32))
	np.savez_compressed(path + 'pert_images.npz', pert_images=np.array(pert_images, dtype=np.float32))
