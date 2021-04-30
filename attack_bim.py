from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
from sklearn.cluster import KMeans

from attack_cw import Classifiers, OptimizerConfigs
from train import *
from utilities import *


class SCAE_BIM_Attack(ModelCollector):
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
			scope='SCAE',
			snapshot=None,
			classifier: str = Classifiers.PriK,
	):
		if input_size is None:
			input_size = [1, 224, 224, 3]

		self._input_size = input_size
		self._n_obj_caps = n_obj_caps

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

			# For normal prediction
			self.res = self.model({'image': self.input})

			capsule_loss_type = 'Pri' if 'Pri' in classifier else 'Pos'
			if capsule_loss_type == 'Pri':
				object_capsule_set = self.res.caps_presence_prob
			elif capsule_loss_type == 'Pos':
				object_capsule_set = tf.reduce_sum(self.res.posterior_mixing_probs, 1)
			else:
				raise NotImplementedError('Unsupported capsule loss type.')

			self.score = object_capsule_set if classifier == Classifiers.PriK or classifier == Classifiers.PosK \
				else self.res.prior_cls_pred if classifier == Classifiers.PriL else self.res.posterior_cls_pred

			subset_position = tf.where(object_capsule_set > tf.reduce_mean(object_capsule_set))[:, 1]
			grads = tf.stack([
				tf.gradients(object_capsule_set[0, i], self.input)[0] for i in range(n_obj_caps)
			])
			self._grads_orig = tf.reduce_sum(tf.gather(grads, subset_position, axis=0), axis=0)[0]

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

	def calc(
			self,
			image: np.ndarray,
			label: int,
			mask: np.ndarray,
			num_iter: int = 100,
			alpha: float = 0.5,
			kmeans: KMeans = None,
			p2l: np.ndarray = None
	):
		pert_image = image.copy()
		best_pert_image = None
		best_result = None
		best_pert_amount = np.inf

		for _ in trange(num_iter, desc='Calculating', ncols=90):
			grads_sign = np.sign(self.sess.run(self._grads_orig, feed_dict={self.input: pert_image}))
			pert_image = (pert_image - grads_sign * alpha * mask).clip(0, 1)
			result = self.sess.run(self.score, feed_dict={self.input: pert_image})

			if classifier == Classifiers.PriK or classifier == Classifiers.PosK:
				result = p2l[kmeans.predict(result)[0]]
			else:
				result = result[0]
			if result != label:
				pert_amount = np.square(pert_image - source_image).sum() ** (1 / 2)
				if pert_amount < best_pert_amount:
					best_pert_image = pert_image.copy()
					best_result = result
					best_pert_amount = pert_amount

		return best_pert_image, best_result


if __name__ == '__main__':
	block_warnings()

	# Attack configuration
	config = config_gtsrb
	num_samples = 5000
	classifier = Classifiers.PriK
	alpha = 0.05
	use_mask = True

	snapshot = './checkpoints/{}/model.ckpt'.format(config['dataset'])

	# Create the attack model according to parameters above
	model = SCAE_BIM_Attack(
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
		scope='SCAE',
		snapshot=snapshot,
		classifier=classifier
	)

	kmeans = None
	p2l = None
	if classifier == Classifiers.PriK:
		# Load prior K-Means classifier
		kmeans = joblib.load('./checkpoints/{}/kmeans_prior.m'.format(config['dataset']))
		npz = np.load('./checkpoints/{}/kmeans_labels_prior.npz'.format(config['dataset']))
		p2l = npz['preds_2_labels']
		npz.close()
	elif classifier == Classifiers.PosK:
		# Load posterior K-Means classifier
		kmeans = joblib.load('./checkpoints/{}/kmeans_posterior.m'.format(config['dataset']))
		npz = np.load('./checkpoints/{}/kmeans_labels_posterior.npz'.format(config['dataset']))
		p2l = npz['preds_2_labels']
		npz.close()

	# Load dataset
	if config['dataset'] == GTSRB:
		dataset = get_gtsrb('train', shape=[config['canvas_size'], config['canvas_size']], file_path='./datasets',
		                    save_only=False, gtsrb_raw_file_path=GTSRB_DATASET_PATH, gtsrb_classes=config['classes'])
	elif config['dataset'] == FASHION_MNIST:
		dataset = get_dataset(config['dataset'], 'train', shape=[config['canvas_size'], config['canvas_size']],
		                      file_path='./datasets', save_only=False)
	else:
		dataset = get_dataset(config['dataset'], 'test', shape=[config['canvas_size'], config['canvas_size']],
		                      file_path='./datasets', save_only=False)

	# Variables to save the attack result
	succeed_count = 0
	succeed_pert_amount = []
	succeed_pert_robustness = []
	source_images = []
	pert_images = []

	# Shuffle the order of samples
	shuffle_indices = list(range(len(dataset['image'])))
	random.seed(time.time())
	random.shuffle(shuffle_indices)

	# Classification accuracy test
	test_acc = 0
	num_test_samples = 10000  # len(dataset['image'])
	for i in trange(num_test_samples, desc='Simple testing', ncols=90):
		idx = shuffle_indices[i]
		result = model.run(to_float32(dataset['image'][idx][None]), model.score)
		if classifier == Classifiers.PriK or classifier == Classifiers.PosK:
			result = p2l[kmeans.predict(result)[0]]
		else:
			result = result[0]
		if result == dataset['label'][idx]:
			test_acc += 1
	print('Model accuracy: {:.6f}.\n'.format(test_acc / num_test_samples))

	# Start the attack on selected samples
	i = 0
	n = num_samples
	while n > 0:
		index = shuffle_indices[i]
		i += 1

		source_image = to_float32(dataset['image'][index])
		source_label = dataset['label'][index]

		# Skip unrecognized samples
		result = model.run(source_image[None], model.score)
		if (classifier == Classifiers.PriK or classifier == Classifiers.PosK) \
				and p2l[kmeans.predict(result)[0]] != source_label \
				or (classifier == Classifiers.PriL or classifier == Classifiers.PosL) \
				and result[0] != source_label:
			print('Skipping sample {}.\n'.format(index))
			continue
		n -= 1

		# Calculate mask
		mask = imblur(source_image, times=1) if use_mask else np.ones_like(source_image)

		pert_image, result = model.calc(source_image[None], source_label, mask[None], alpha=alpha, kmeans=kmeans, p2l=p2l)

		if pert_image is None:
			# Print result
			print('Failed for {}. Source label: {}.'.format(index, source_label))
		else:
			# Add info of the succeed sample to result variables
			succeed_count += 1
			pert_amount = np.linalg.norm(pert_image - source_image)
			pert_robustness = pert_amount / np.linalg.norm(source_image)
			succeed_pert_amount.append(pert_amount)
			succeed_pert_robustness.append(pert_robustness)

			# Print result
			print('Succeed for {}. Source label: {}. Pert label: {}. Pert amount: {:.2f}. Pert robustness: {:.2f}.'
			      .format(index, source_label, result, pert_amount, pert_robustness))

			# Save the pert image
			source_images.append(source_image)
			pert_images.append(pert_image)

		print('Remain: {}\n'.format(n))

	# Create result directory
	now = time.localtime()
	path = './results/bim/{}_{}_{}_{}_{}/'.format(
		now.tm_year,
		now.tm_mon,
		now.tm_mday,
		now.tm_hour,
		now.tm_min
	)
	if not os.path.exists(path):
		os.makedirs(path)

	# Save the final result of complete attack
	succeed_pert_amount = np.array(succeed_pert_amount, dtype=np.float32)
	succeed_pert_robustness = np.array(succeed_pert_robustness, dtype=np.float32)
	result = 'Dataset: {}\nClassifier: {}\nNum of samples: {}\nSuccess rate: {:.4f}\nAverage pert amount: {:.4f}\n' \
	         'Pert amount standard deviation: {:.4f}\nPert robustness: {:.4f}\n' \
	         'Pert robustness standard deviation: {:.4f}\n'.format(
		config['dataset'], classifier, num_samples, succeed_count / num_samples, succeed_pert_amount.mean(),
		succeed_pert_amount.std(), succeed_pert_robustness.mean(), succeed_pert_robustness.std())
	print(result)
	if os.path.exists(path + 'result.txt'):
		os.remove(path + 'result.txt')
	result_file = open(path + 'result.txt', mode='x')
	result_file.write(result)
	result_file.close()
	np.savez_compressed(path + 'source_images.npz', source_images=np.array(source_images, dtype=np.float32))
	np.savez_compressed(path + 'pert_images.npz', pert_images=np.array(pert_images, dtype=np.float32))
