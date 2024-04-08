# from Process.process import loadAttrData, loadAttrTree
# from Process.rand5fold import *
# from tools.evaluate import *
# from baseline_models import *

from utils2 import load5PreFoldedData, EarlyStopping, evaluation, setup_seed, best_epoch, update_best_epoch, parsearg
from models import GCN, RDEA, GCN_cat, parse_modelname


import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import time, os, sys, pdb, json, argparse
from datetime import datetime
import warnings
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from utils_data import loadRLData





def train():
	model, optimizer = parse_modelname(args)
	early_stopping = EarlyStopping(patience=args.patience, verbose=True)


	for epoch in range(args.n_epochs):
		traindata_list, testdata_list = loadRLData(args.datasetname, fold_x_train, fold_x_test, TDdroprate, BUdroprate, data_path)
		train_loader = DataLoader(traindata_list, batch_size=args.batch_size, shuffle=False, num_workers=10)
		test_loader = DataLoader(testdata_list, batch_size=args.batch_size, shuffle=False, num_workers=10)
		
		temp_train_losses = []
		pred_y_list = []
		true_y_list = []
		model.train()
		for Batch_data in train_loader:
			Batch_data.to(device)
			out_labels= model(Batch_data)
			loss=F.nll_loss(out_labels,Batch_data.y)
			temp_train_losses.append(loss.item()*out_labels.shape[0])
			_, pred = out_labels.max(dim=-1)
			pred_y_list.extend(pred.tolist())
			true_y_list.extend(Batch_data.y.tolist())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		train_acc = np.sum(np.array(pred_y_list) == np.array(true_y_list)) / len(true_y_list)
			
		temp_val_losses = []
		output_y_list = []
		true_y_list = []
		model.eval()
		for Batch_data in test_loader:
			Batch_data.to(device)
			val_out = model(Batch_data)
			val_loss  = F.nll_loss(val_out, Batch_data.y)
			temp_val_losses.append(val_loss.item()*val_out.shape[0])
			output_y_list.extend(val_out.tolist())
			true_y_list.extend(Batch_data.y.tolist())
		
		res = evaluation(output_y_list, true_y_list, args.num_class)
		print(f'{args.datasetname}/{fold_str}(iter:{iter})\tEpoch:{epoch}\t'+
			'train_loss:{:.4f}\ttrain_acc:{:.4f}\tval_loss{:.4f}\t'.format(
				sum(temp_train_losses)/len(traindata_list), train_acc, sum(temp_val_losses)/len(testdata_list))+
			'\t'.join(str(x) for x in res))

		model_path = os.path.join(args.model_save_path, "{}_iter{}.pt".format(fold_str, iter))
		os.makedirs(os.path.dirname(model_path), exist_ok=True)

		early_stopping(sum(temp_val_losses)/len(testdata_list), res, model, model_path)
		# early_stopping(-res[0], res, model, model_path)
		if early_stopping.early_stop:
				print("Early stopping")
				res = early_stopping.res
				break
	return res





if __name__ == '__main__':
	print(datetime.now(),"starts.")
	args = parsearg()	
	print(args)
	setup_seed(args.seed)
	device = torch.device(args.device)
	x_folds = load5PreFoldedData(args.datasetname)
	
	TDdroprate = BUdroprate = args.droprate
	if args.lr is None:
		args.lr = 0.0005 if args.num_class == 4 else 0.0001
	
	if args.best_epoch_path is None:
		best_epochs = best_epoch(args.datasetname+'_'+args.modelname)
	else:
		best_epochs = json.load(open(args.best_epoch_path, 'r'))


	all_res = {}
	best_res = {}
	for fold in range(5):
		all_res[str(fold)] = []
		for e in best_epochs[str(fold)]:
			fold_str = 'fold_'+str(fold)+'L/epoch_'+str(e)
			data_path = os.path.join(args.main_data_path, fold_str)
			fold_x_train, fold_x_test = x_folds[2*int(fold)+1], x_folds[2*int(fold)]
			print(f"loading train set: len of fold_x_train is {len(fold_x_train)} from {data_path}")
			print(f"loading test set: len of fold_x_test is {len(fold_x_test)} from {data_path}")
			iter_mean_res = []
			for iter in range(5):
				res = train()
				with open(args.res_log_file, 'a') as f:
					f.write(f'{args.datasetname}/{fold_str}\titer_{iter}\t'+'\t'.join(str(x) for x in res)+'\n')
				iter_mean_res.append(res)
			print(f'{args.datasetname}/{fold_str}\t'+'\t'.join(str(x) for x in np.mean(iter_mean_res, axis=0)))
			all_res[str(fold)].append(np.mean(iter_mean_res, axis=0))
		best_res[str(fold)] = all_res[str(fold)][np.argmax(np.array(all_res[str(fold)])[:,0])]
	update_best_epoch(args.datasetname+'_'+args.modelname, all_res)
	print(all_res)
	print(best_res)
	print(datetime.now(),"ends.")
