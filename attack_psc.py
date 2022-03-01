import numpy as np
import tensorflow as tf
from tqdm import trange

from tools.model import Attacker, ScaeBasement, KMeans
from tools.utilities import block_warnings, imblur, DatasetHelper, ResultBuilder
from train import Configs, build_from_config


class AttackerPSC(Attacker):
	def __init__(
			self,
			scae: ScaeBasement,
			classifier: str,
			kmeans_classifier: KMeans = None,
			alpha: float = 0.5,
			num_iter: int = 200
	):
		self._classifier = classifier
		self._num_iter = num_iter

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

			n_obj_caps = int(scae._res.posterior_mixing_probs.shape[2])

			# Variables to be assigned during initialization
			self._pert_images = tf.Variable(tf.zeros(self._input_size), trainable=False)
			self._mask = tf.Variable(tf.zeros(self._input_size), trainable=False)
			self._subset_position = tf.Variable(tf.zeros([self._input_size[0], n_obj_caps]), trainable=False)

			pert_res = scae._model({'image': self._pert_images})

			if classifier[:3].upper() == 'PRI':
				object_capsule_set = pert_res.caps_presence_prob
			elif classifier[:3].upper() == 'POS':
				object_capsule_set = tf.reduce_sum(pert_res.posterior_mixing_probs, axis=1)
			else:
				raise NotImplementedError('Unsupported capsule loss type.')

			grads = tf.stack([
				tf.gradients(object_capsule_set[:, i], self._pert_images)[0] for i in range(n_obj_caps)
			], axis=1)

			expand_subset_position = self._subset_position
			for i in range(len(self._input_size) - 1):
				expand_subset_position = tf.expand_dims(expand_subset_position, axis=-1)

			grads_orig = tf.reduce_sum(grads * expand_subset_position, axis=1) * self._mask
			grads_other = tf.reduce_sum(grads * (1 - expand_subset_position), axis=1) * self._mask

			# Code below is written based on the excellent work of DEEPSEC:
			# https://github.com/kleincup/DEEPSEC/blob/2c67afac0ae966767b6712a51db85f04f4f5c565/Attacks/AttackMethods/JSMA.py

			n_features = int(np.prod(self._input_size[1:]))
			batch_size = self._input_size[0]

			increase_coef = tf.multiply(tf.reshape(tf.cast(tf.equal(self._pert_images, 0), tf.float32), [batch_size, -1]), 2)

			grads_orig_cpy = tf.reshape(grads_orig, [batch_size, -1])
			grads_orig_cpy -= increase_coef * tf.reduce_max(tf.abs(grads_orig))
			saliency_orig = tf.reshape(grads_orig_cpy, [batch_size, -1, 1, n_features]) \
			                + tf.reshape(grads_orig_cpy, [batch_size, -1, n_features, 1])

			grads_other_cpy = tf.reshape(grads_other, [batch_size, -1])
			grads_other_cpy += increase_coef * tf.reduce_max(tf.abs(grads_other))
			saliency_other = tf.reshape(grads_other_cpy, [batch_size, -1, 1, n_features]) \
			                 + tf.reshape(grads_other_cpy, [batch_size, -1, n_features, 1])

			zero_diagonal = tf.ones([batch_size, n_features, n_features])
			zero_diagonal -= tf.matrix_diag(tf.ones([batch_size, n_features]))

			mask1 = tf.cast(tf.greater(saliency_orig, 0.0), tf.float32)
			mask2 = tf.cast(tf.less(saliency_other, 0.0), tf.float32)
			mask3 = tf.multiply(tf.multiply(mask1, mask2), tf.reshape(zero_diagonal, mask1.shape))
			saliency_map = tf.multiply(tf.multiply(saliency_orig, tf.abs(saliency_other)), mask3)

			sub_matrix = []
			for i in range(batch_size):
				max_idx = tf.argmax(tf.reshape(saliency_map[i], [-1]))
				p1 = max_idx // n_features
				p2 = max_idx % n_features
				sub_matrix.append(tf.where(tf.broadcast_to(tf.not_equal(p1, p2), self._input_size[1:]),
				                           tf.sparse_tensor_to_dense(
					                           tf.SparseTensor(indices=[tf.unravel_index(p1, self._input_size[1:]),
					                                                    tf.unravel_index(tf.where(tf.not_equal(p1, p2), p2, p2 + 1),
					                                                                     self._input_size[1:])],
					                                           values=[alpha, alpha],
					                                           dense_shape=self._input_size[1:])),
				                           tf.zeros(self._input_size[1:])))
			sub_matrix = tf.stack(sub_matrix)

			self._train_step = tf.assign(self._pert_images, tf.clip_by_value(self._pert_images - sub_matrix, 0, 1))

			# Initialization operation
			self._init = [
				tf.assign(self._pert_images, self._ph_input),
				tf.assign(self._mask, self._ph_mask)
			]

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

		# The best pert amount and pert image
		global_success = np.full([num_images], 0, np.bool)
		global_best_pert_images = np.full([num_images, *self._input_size[1:]], np.nan) \
			if nan_if_fail else images[:num_images].copy()

		# Init the original images and masks
		self._sess.run(self._init, feed_dict={self._ph_input: images,
		                                      self._ph_mask: mask})

		# Iteration
		dynamic_range = trange(self._num_iter) if verbose else range(self._num_iter)
		for _ in dynamic_range:
			self._sess.run(self._train_step)

			# Determine if succeed
			results, pert_images = self._sess.run([self._score, self._pert_images])
			if self._classifier[-1].upper() == 'K':
				results = self._kmeans_classifier.run(results, self._kmeans_classifier._output)
			succeed = results != labels

			for i in range(num_images):
				# Update global best result
				if succeed[i] and not global_success[i]:
					global_best_pert_images[i] = pert_images[i]
					global_success[i] = True

			# Exit when all pixels are subtracted to zero, or all images are perturbed successfully
			if True not in (pert_images[:num_images] != 0) or False not in global_success:
				if verbose:
					dynamic_range.close()
				break

		return global_best_pert_images


if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = Configs.config_mnist
	num_samples = 1000
	batch_size = 10
	classifier = Attacker.Classifiers.PosK
	num_iter = 200
	alpha = 0.5
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

	attacker = AttackerPSC(
		scae=model,
		classifier=classifier,
		kmeans_classifier=kmeans if classifier[-1].upper() == 'K' else None,
		alpha=alpha,
		num_iter=num_iter
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
			0 if succeed_count == 0 else succeed_count / (num_samples - remain),
			np.array(succeed_pert_amount, dtype=np.float32).mean(), remain))

	# Change list into numpy array
	succeed_pert_amount = np.array(succeed_pert_amount, dtype=np.float32)
	succeed_pert_robustness = np.array(succeed_pert_robustness, dtype=np.float32)

	# Save the final result of complete attack
	result = ResultBuilder()
	result['Dataset'] = config['dataset']
	result['Classifier'] = classifier
	result['Num of samples'] = num_samples

	# Attacker params
	result['Num of iter'] = num_iter
	result['Alpha'] = str(alpha)

	# Attack results
	result['Success rate'] = succeed_count / num_samples
	result['Average pert amount'] = succeed_pert_amount.mean()
	result['Pert amount standard deviation'] = succeed_pert_amount.std()
	result['Average pert robustness'] = succeed_pert_robustness.mean()
	result['Pert robustness standard deviation'] = succeed_pert_robustness.std()

	# Print and save results
	print(result)
	path = result.save('./results/psc/')
	np.savez_compressed(path + 'source_images.npz', source_images=np.array(source_images, dtype=np.float32))
	np.savez_compressed(path + 'pert_images.npz', pert_images=np.array(pert_images, dtype=np.float32))