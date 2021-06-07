import os

from tqdm import tqdm

from tools.model import KMeans
from tools.utilities import block_warnings, DatasetHelper
from train import Configs, build_from_config


def test(
		config,
		batch_size=100,
		snapshot=None,
		snapshot_kmeans_pri=None,
		snapshot_kmeans_pos=None,
		train_and_save_kmeans=True
):
	"""
		General method for test, so you don't need to copy the full test code to every file.
	"""
	block_warnings()

	model = build_from_config(
		config=config,
		batch_size=batch_size,
		is_training=False,
		snapshot=snapshot
	)

	kmeans_pri = KMeans(
		scae=model,
		kmeans_type=KMeans.KMeansTypes.Prior,
		is_training=train_and_save_kmeans,
		scope='KMeans_Pri',
		snapshot=None if train_and_save_kmeans else snapshot_kmeans_pri
	)

	kmeans_pos = KMeans(
		scae=model,
		kmeans_type=KMeans.KMeansTypes.Posterior,
		is_training=train_and_save_kmeans,
		scope='KMeans_Pos',
		snapshot=None if train_and_save_kmeans else snapshot_kmeans_pos
	)

	model.finalize()

	trainset = DatasetHelper(config['dataset'], 'train',
	                         file_path='./datasets', batch_size=batch_size, fill_batch=True,
	                         normalize=True if config['dataset'] == Configs.GTSRB else False,
	                         gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES)
	testset = DatasetHelper(config['dataset'], 'test',
	                        file_path='./datasets', batch_size=batch_size, fill_batch=True,
	                        normalize=True if config['dataset'] == Configs.GTSRB else False,
	                        gtsrb_raw_file_path=Configs.GTSRB_DATASET_PATH, gtsrb_classes=Configs.GTSRB_CLASSES)

	# ------------------------------------- Supervised Classification ------------------------------------- #

	test_acc_prior = 0.
	test_acc_posterior = 0.

	for images, labels in tqdm(trainset, desc='Testing trainset'):
		test_pred_prior, test_pred_posterior = model.run(
			images=images,
			to_collect=[model._res.prior_cls_pred,
			            model._res.posterior_cls_pred]
		)

		test_acc_prior += (test_pred_prior == labels).sum()
		test_acc_posterior += (test_pred_posterior == labels).sum()

	for images, labels in tqdm(testset, desc='Testing testset'):
		test_pred_prior, test_pred_posterior = model.run(
			images=images,
			to_collect=[model._res.prior_cls_pred,
			            model._res.posterior_cls_pred]
		)

		test_acc_prior += (test_pred_prior == labels).sum()
		test_acc_posterior += (test_pred_posterior == labels).sum()

	print('\nSupervised acc: prior={:.6f}, posterior={:.6f}\n'
	      .format(test_acc_prior / (trainset.dataset_size + testset.dataset_size),
	              test_acc_posterior / (trainset.dataset_size + testset.dataset_size)))

	# ------------------------------------- Unsupervised Classification ------------------------------------- #

	if train_and_save_kmeans:
		kmeans_pri.train(trainset, verbose=True)
		kmeans_pos.train(trainset, verbose=True)

	test_acc_prior = 0.
	test_acc_posterior = 0.
	for images, labels in tqdm(trainset, desc='Testing trainset'):
		test_acc_prior += (kmeans_pri(images) == labels).sum()
		test_acc_posterior += (kmeans_pos(images) == labels).sum()

	for images, labels in tqdm(testset, desc='Testing testset'):
		test_acc_prior += (kmeans_pri(images) == labels).sum()
		test_acc_posterior += (kmeans_pos(images) == labels).sum()

	print('\nUnsupervised acc: prior={:.6f}, posterior={:.6f}\n'
	      .format(test_acc_prior / (trainset.dataset_size + testset.dataset_size),
	              test_acc_posterior / (trainset.dataset_size + testset.dataset_size)))

	if train_and_save_kmeans:
		kmeans_pri.save_model(snapshot_kmeans_pri)
		kmeans_pos.save_model(snapshot_kmeans_pos)

	return model, kmeans_pri, kmeans_pos, trainset, testset


if __name__ == '__main__':
	config = Configs.config_mnist
	snapshot = './checkpoints/{}/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_pri = './checkpoints/{}/kmeans_pri/model.ckpt'.format(config['dataset'])
	snapshot_kmeans_pos = './checkpoints/{}/kmeans_pos/model.ckpt'.format(config['dataset'])
	train_and_save_kmeans = not (os.path.exists(snapshot_kmeans_pri[:snapshot_kmeans_pri.rindex('/')])
	                             and os.path.exists(snapshot_kmeans_pos[:snapshot_kmeans_pos.rindex('/')]))

	model, kmeans_pri, kmeans_pos, trainset, testset = test(
		config=config,
		snapshot=snapshot,
		snapshot_kmeans_pri=snapshot_kmeans_pri,
		snapshot_kmeans_pos=snapshot_kmeans_pos,
		train_and_save_kmeans=train_and_save_kmeans
	)
