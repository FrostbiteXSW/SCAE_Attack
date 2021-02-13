from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from train import *
from utilities import get_dataset, block_warnings, to_float32


def bipartite_match(preds, labels, n_classes=None, presence=None):
	"""Does maximum biprartite matching between `pred` and `gt`."""

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

	preds_2_labels = [0 for _ in range(n_classes)]
	for i in range(n_classes):
		preds_2_labels[preds_idx[i]] = labels_idx[i]

	return preds_2_labels


if __name__ == '__main__':
	block_warnings()

	config = config_svhn
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
		part_encoder_noise_scale=0.,
		obj_decoder_noise_type=None,
		obj_decoder_noise_scale=0.,
		set_transformer_n_layers=config['set_transformer_n_layers'],
		set_transformer_n_heads=config['set_transformer_n_heads'],
		set_transformer_n_dims=config['set_transformer_n_dims'],
		set_transformer_n_output_dims=config['set_transformer_n_output_dims'],
		part_cnn_strides=config['part_cnn_strides'],
		prep=config['prep'],
		is_training=False,
		scope='SCAE',
		snapshot=snapshot
	)

	if config['dataset'] == 'gtsrb':
		trainset = get_gtsrb('train', shape=[config['canvas_size'], config['canvas_size']], file_path='./datasets',
		                     save_only=False, gtsrb_raw_file_path='./datasets/GTSRB', gtsrb_classes=config['classes'])
		testset = get_gtsrb('test', shape=[config['canvas_size'], config['canvas_size']], file_path='./datasets',
		                    save_only=False, gtsrb_raw_file_path='./datasets/GTSRB', gtsrb_classes=config['classes'])
	else:
		trainset = get_dataset(config['dataset'], 'train', shape=[config['canvas_size'], config['canvas_size']],
		                       file_path='./datasets', save_only=False)
		testset = get_dataset(config['dataset'], 'test', shape=[config['canvas_size'], config['canvas_size']],
		                      file_path='./datasets', save_only=False)

	path = snapshot[:snapshot.rindex('/')]
	if not os.path.exists(path):
		raise FileExistsError('Cannot access checkpoint files')

	len_trainset = len(trainset['image'])
	len_testset = len(testset['image'])

	train_batches = np.int(np.ceil(np.float(len_trainset) / np.float(batch_size)))
	test_batches = np.int(np.ceil(np.float(len_testset) / np.float(batch_size)))

	# Supervised Classification

	test_acc_prior = 0.
	test_acc_posterior = 0.
	prior_pres_list = []
	posterior_pres_list = []

	for i_batch in trange(train_batches, desc='Testing trainset'):
		i_end = min((i_batch + 1) * batch_size, len_trainset)
		i_start = min(i_batch * batch_size, i_end - batch_size)
		images = to_float32(trainset['image'][i_start:i_end])
		labels = trainset['label'][i_start:i_end]
		test_pred_prior, test_pred_posterior, prior_pres, posterior_pres = model.sess.run(
			[model.res.prior_cls_pred,
			 model.res.posterior_cls_pred,
			 model.res.caps_presence_prob,
			 model.res.posterior_mixing_probs],
			feed_dict={model.batch_input: images})
		n_samples = i_end - (i_batch * batch_size)
		test_acc_prior += (test_pred_prior == labels)[:n_samples].sum()
		test_acc_posterior += (test_pred_posterior == labels)[:n_samples].sum()
		prior_pres_list.append(prior_pres[:n_samples])
		posterior_pres_list.append(posterior_pres[:n_samples])

	for i_batch in trange(test_batches, desc='Testing testset'):
		i_end = min((i_batch + 1) * batch_size, len_testset)
		i_start = min(i_batch * batch_size, i_end - batch_size)
		images = to_float32(testset['image'][i_start:i_end])
		labels = testset['label'][i_start:i_end]
		test_pred_prior, test_pred_posterior, prior_pres, posterior_pres = model.sess.run(
			[model.res.prior_cls_pred,
			 model.res.posterior_cls_pred,
			 model.res.caps_presence_prob,
			 model.res.posterior_mixing_probs],
			feed_dict={model.batch_input: images})
		n_samples = i_end - (i_batch * batch_size)
		test_acc_prior += (test_pred_prior == labels)[:n_samples].sum()
		test_acc_posterior += (test_pred_posterior == labels)[:n_samples].sum()
		prior_pres_list.append(prior_pres[:n_samples])
		posterior_pres_list.append(posterior_pres[:n_samples])

	print('Supervised acc: prior={:.6f}, posterior={:.6f}'
	      .format(test_acc_prior / (len_trainset + len_testset), test_acc_posterior / (len_trainset + len_testset)))

	# Unsupervised Classification

	prior_pres_list = np.concatenate(prior_pres_list)
	posterior_pres_list = np.concatenate(posterior_pres_list).sum(1)

	kmeans_prior = KMeans(
		n_clusters=config['num_classes'],
		precompute_distances=True,
		n_jobs=-1,
		max_iter=1000,
	).fit(prior_pres_list)

	kmeans_posterior = KMeans(
		n_clusters=config['num_classes'],
		precompute_distances=True,
		n_jobs=-1,
		max_iter=1000,
	).fit(posterior_pres_list)

	kmeans_pred_list_prior = kmeans_prior.predict(prior_pres_list)
	kmeans_pred_list_posterior = kmeans_posterior.predict(posterior_pres_list)
	ground_truth_list = np.concatenate([trainset['label'], testset['label']])

	p2l_prior = bipartite_match(kmeans_pred_list_prior[:len_trainset], trainset['label'], config['num_classes'])
	p2l_posterior = bipartite_match(kmeans_pred_list_posterior[:len_trainset], trainset['label'], config['num_classes'])

	for i in range(len(kmeans_pred_list_prior)):
		# kmeans_label to gt_label
		kmeans_pred_list_prior[i] = p2l_prior[kmeans_pred_list_prior[i]]

	for i in range(len(kmeans_pred_list_posterior)):
		# kmeans_label to gt_label
		kmeans_pred_list_posterior[i] = p2l_posterior[kmeans_pred_list_posterior[i]]

	print('Unsupervised acc: prior={:.6f}, posterior={:.6f}'
	      .format((kmeans_pred_list_prior == ground_truth_list).sum() / (len_trainset + len_testset),
	              (kmeans_pred_list_posterior == ground_truth_list).sum() / (len_trainset + len_testset)))

	joblib.dump(kmeans_prior, '{}/kmeans_prior.m'.format(path))
	np.savez_compressed('{}/kmeans_labels_prior.npz'.format(path), preds_2_labels=p2l_prior)

	joblib.dump(kmeans_posterior, '{}/kmeans_posterior.m'.format(path))
	np.savez_compressed('{}/kmeans_labels_posterior.npz'.format(path), preds_2_labels=p2l_posterior)
