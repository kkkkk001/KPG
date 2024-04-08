from bert_main import Net as bert_net
from bert_main import setup_seed
from bert_main import load5PreFoldedData, RootTextDataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

from torch.utils.data import Dataset
from torch.utils.data import DataLoader as thDataLoader
from torch_geometric.loader import DataLoader as gDataLoader
from utils_data import RLGraphDataset
from models import parse_modelname

from utils2 import evaluation, best_epoch, loadGPTaugText, loadFullText, loadText, compute_ece


import sys, inspect, pdb, os, json
import torch
import numpy as np
import argparse
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



parser = argparse.ArgumentParser(description='Combine inference parameters')
parser.add_argument('--datasetname', type=str, required=True, help='Name of the dataset')
parser.add_argument('--num_class', type=int, required=True, help='Number of classes')
parser.add_argument('--text', type=str, default='full', help='load root, full, or aug text')
parser.add_argument('--seed', type=int, default=2023, help='seed for random')

parser.add_argument('--bert_batch_size', type=int, default=8, help='Batch size for training in bert model')
parser.add_argument('--GCN_batch_size', type=int, default=128, help='Batch size for training in GCN model')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device to use for training')

parser.add_argument('--rl_data_main_path', type=str, required=True, help='path+fold_0L/epoch_$bestepoch$/')
parser.add_argument('--bert_model_path', type=str, required=True, help='path+fold0_iter0_bert.pt')
parser.add_argument('--GCN_model', type=str, required=True, help='model: GCN, RDEA, GCN_cat')
parser.add_argument('--GCN_model_path', type=str, required=True, help='path+fold_0L/epoch_$bestepoch$_iter0.pt')
parser.add_argument('--best_epoch_path', type=str, help='path+fold_0L/epoch_$bestepoch$_iter0.pt')


parser.add_argument('--mu', type=float, default=1, help='trade-off parameter')
parser.add_argument('--temp', type=float, default=1, help='temperature for softmax')
parser.add_argument('--temp_gcn', type=float, default=1, help='temperature for softmax for GCN')
parser.add_argument('--return_prob', type=int, default=1, help='')
parser.add_argument('--save_best_epoch', type=int, default=1, help='')
parser.add_argument('--save_inf_res', type=int, default=0, help='')
parser.add_argument('--specific_fold', type=str, default='-1', help='-splited')



args = parser.parse_args()
args.modelname = args.GCN_model
print(args)
setup_seed(args.seed)


print(datetime.now(),"starts.")

main_path = 'data/'
x_folds = load5PreFoldedData(args.datasetname)
textDic = loadText(args.datasetname, main_path, args.text)
device = torch.device(args.device)
##### setting for bert
if args.datasetname == "Weibo":
	MODEL_PATH = 'bert-base-chinese'
else:
	MODEL_PATH = 'bert-base-uncased'

##### setting for KPGCN
if args.best_epoch_path is None:
	best_epochs = {}
	for fold in range(5):
		best_epochs[str(fold)] = [0, 1, 2, 3, 4]
else:
	best_epochs = json.load(open(args.best_epoch_path, 'r'))


bert_all_res = []
KPGCN_all_res =[]
mean_all_res = []
all_pred = []
all_y = []
all_size = []
if args.specific_fold == '-1':
	folds = [x for x in range(5)]
else:
	folds = [int(x) for x in args.specific_fold.split('-')]
print("folds: ", folds)
# for fold in range(5):
for fold in folds:
	for bestepoch in best_epochs[str(fold)]:
		for iter in range(5):
			x_train_ids, x_test_ids = x_folds[2*int(fold)+1], x_folds[2*int(fold)]

			#####  bert data
			test_data = RootTextDataset(x_test_ids, textDic, args.datasetname, main_path)
			bert_test_loader = thDataLoader(test_data, batch_size=args.bert_batch_size, shuffle=False)

			##### bert model
			tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
			bert = BertModel.from_pretrained(MODEL_PATH).to(device)
			bert_model = bert_net(768, args.num_class, bert, return_prob=True, temp=args.temp).to(device)
			bert_model_path = os.path.join(args.bert_model_path, "fold{}_iter{}_bert.pt".format(fold, iter))
			print('loading model from '+bert_model_path)
			bert_model.load_state_dict(torch.load(bert_model_path))
			
			##### KPGCN data
			# bestepoch = best_epochs[str(fold)][0]
			rl_data_path = os.path.join(args.rl_data_main_path, 'fold_{}L/epoch_{}'.format(fold, bestepoch))
			print('loading rl_data from ' + rl_data_path)
			testdata = RLGraphDataset(x_test_ids, dataset=args.datasetname, data_path=rl_data_path)
			KPGCN_test_loader = gDataLoader(testdata, batch_size=args.GCN_batch_size, shuffle=False, num_workers=10)
			##### KPGCN model
			KPGCN_model, _ = parse_modelname(args)
			# KPGCN_model = GCN(5000,64,64,args.num_class, return_prob=True).to(device)
			
			GCN_model_path = os.path.join(args.GCN_model_path, "fold_{}L/epoch_{}_iter{}.pt".format(fold, bestepoch, iter))
			print('loading model from '+GCN_model_path)
			KPGCN_model.load_state_dict(torch.load(GCN_model_path))


			##### inference
			bert_model.eval()
			KPGCN_model.eval()

			bert_outputs_list = []
			bert_true_y_list = []
			bert_probs_list = []
			for Batch_data,Batch_data_y in bert_test_loader:
				tokens = tokenizer(Batch_data, padding=True, truncation=True, max_length=512)
				input_ids = torch.tensor(tokens["input_ids"]).to(device)
				attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
				Batch_data_y = Batch_data_y.to(device)
				bert_outputs, prob = bert_model(input_ids,attention_mask)
				bert_outputs_list.extend(bert_outputs.tolist())
				bert_true_y_list.extend(Batch_data_y.tolist())
				bert_probs_list.extend(prob.tolist())

			KPGCN_outputs_list = []
			KPGCN_true_y_list = []
			KPGCN_probs_list = []
			size_list = []
			for Batch_data in KPGCN_test_loader:
				Batch_data = Batch_data.to(device)
				Batch_data_y = Batch_data.y.to(device)
				KPGCN_outputs, prob = KPGCN_model(Batch_data)
				KPGCN_outputs_list.extend(KPGCN_outputs.tolist())
				KPGCN_true_y_list.extend(Batch_data_y.tolist())
				KPGCN_probs_list.extend(prob.tolist())
				size_list.extend((Batch_data.ptr[1:]-Batch_data.ptr[:-1]).tolist())

			assert np.array_equal(bert_true_y_list, KPGCN_true_y_list)
			bert_ece = compute_ece(np.array(bert_probs_list), np.array(bert_true_y_list), args.num_class)
			KPGCN_ece = compute_ece(np.array(KPGCN_probs_list), np.array(KPGCN_true_y_list), args.num_class)

			mean_outputs_list = np.array(bert_outputs_list)+args.mu*np.array(KPGCN_outputs_list)
			bert_res = evaluation(bert_outputs_list, bert_true_y_list, args.num_class)
			KPGCN_res= evaluation(KPGCN_outputs_list, KPGCN_true_y_list, args.num_class)
			mean_res = evaluation(mean_outputs_list, bert_true_y_list, args.num_class)
			all_pred.extend(mean_outputs_list)
			all_y.extend(bert_true_y_list)
			all_size.extend(size_list)


			print("Bert\tfold{}\titer{}\t".format(fold, iter), "\t".join(str(r) for r in bert_res))
			print("KPGCN\tfold{}\titer{}\t".format(fold, iter), "\t".join(str(r) for r in KPGCN_res))
			print("mean\tfold{}\titer{}\t".format(fold, iter), "\t".join(str(r) for r in mean_res))
			print(f"bert ece={bert_ece:3f}%\tKPGCN ece={KPGCN_ece:3f}%")
			bert_all_res.append(bert_res)
			KPGCN_all_res.append(KPGCN_res)
			mean_all_res.append(mean_res)


print(np.array(bert_all_res).mean(axis=0))
print(np.array(KPGCN_all_res).mean(axis=0))
print(np.array(mean_all_res).mean(axis=0))

print(np.array(mean_all_res).shape)

print(datetime.now(),"ends.")

if args.save_best_epoch:
	fold_res = np.array(mean_all_res).reshape(-1, 5, args.num_class+1).mean(axis=1).reshape(-1, 5, args.num_class+1)
	best_ep_idx = fold_res[:, :, 0].argmax(axis=1)
	best_ep_res = fold_res[np.arange(fold_res.shape[0]), best_ep_idx]
	print("best results of combination: ", best_ep_res.mean(axis=0))
	print(f'best epoch for {args.datasetname} is ', best_epoch)
	print(f'saving to: best_epoch/{args.datasetname}.json')
	best_epoch = {'0':[int(best_ep_idx[0])],
			   '1':[int(best_ep_idx[1])],
			   '2':[int(best_ep_idx[2])],
			   '3':[int(best_ep_idx[3])],
			   '4':[int(best_ep_idx[4])]}
	print(fold_res)
	print(best_ep_res)
	with open(f'best_epoch/{args.datasetname}.json','w') as f:
		json.dump(best_epoch, f)
	print(datetime.now(),"ends2.")

## saving the iteration results
if not args.save_best_epoch:
	print(mean_all_res)
	with open(f'cmb_log/{args.datasetname}.txt', 'w') as f:
		for fold in folds:
			for iter in range(5):
				f.write(f'fold{fold}\titer{iter}\t'+'\t'.join(str(x) for x in mean_all_res[fold*5+iter])+'\n')
			

# 	pdb.set_trace()
# 	for i in range(5):
# 		iter_mean = np.array(mean_all_res).reshape(-1, 5, args.num_class+1)[:, i, :].mean(axis=0)
# 		iter_std = np.array(mean_all_res).reshape(-1, 5, args.num_class+1)[:, i, :].std(axis=0)
# 		with open(f'cmb_log/{args.datasetname}.txt', 'a') as f:
# 			f.write(f'iter{i}_mean\t'+'\t'.join(str(x) for x in iter_mean)+'\n')
# 			f.write(f'iter{i}_std\t'+'\t'.join(str(x) for x in iter_std)+'\n')
# else:
# 	fold_of_best_epoch = np.array(mean_all_res).reshape(5, -1, 5, args.num_class+1)[range(5), best_ep_idx]
# 	for i in range(5):
# 		iter_mean = fold_of_best_epoch[:,i,:].mean(axis=0)
# 		iter_std = fold_of_best_epoch[:,i,:].std(axis=0)
# 		with open(f'cmb_log/{args.datasetname}.txt', 'a') as f:
# 			f.write(f'iter{i}_mean\t'+'\t'.join(str(x) for x in iter_mean)+'\n')
# 			f.write(f'iter{i}_std\t'+'\t'.join(str(x) for x in iter_std)+'\n')



if args.save_inf_res:
	all_pred = np.array(all_pred)
	all_pred = np.argmax(all_pred, axis=1)
	all_y = np.array(all_y)
	all_size = np.array(all_size)
	np.savez(f'inf_res/KPG_{args.datasetname}.npz', all_pred=all_pred, all_y=all_y, all_size=all_size)
	print(f'saving the inf res to: inf_res/KPG_{args.datasetname}.npz')
exit()
idx = np.argsort(all_size)
all_pred = all_pred[idx]
all_y = all_y[idx]
all_size = all_size[idx]

# 10 bins for size
num_bins = 5
bins = np.linspace(0, max(all_size), num_bins+1)
counts, x = np.histogram(all_size, bins)
counts = counts / sum(counts)
print('\t'.join([str(i) for i in counts]))

# 10 bins computing acc
acc_lst = []
for i in range(num_bins):
	idx = (all_size >= bins[i]) & (all_size < bins[i+1])
	acc = np.sum(all_pred[idx] == all_y[idx]) / len(all_y[idx])
	acc_lst.append(acc)
print('\t'.join([str(i) for i in acc_lst]))
pdb.set_trace()


# with open('log_t16_param.txt', 'a') as f:
# 	f.write(f'{args.GCN_model_path}\t'+'\t'.join(str(x) for x in np.array(mean_all_res).mean(axis=0))+'\n')
# print(np.array(mean_all_res))

