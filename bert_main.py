from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from datetime import datetime
import os, sys, pdb, random, argparse

from utils2 import load5PreFoldedData, setup_seed, loadFullText, loadRootText, loadGPTaugText, EarlyStopping, evaluation, loadText, compute_ece


setup_seed(2024)



class RootTextDataset(Dataset):
	def __init__(self, fold_x, textDic, dataname, main_path):
		self.fold_x = fold_x
		self.textDic = textDic
		self.dataname = dataname
		self.data_path = os.path.join(main_path, dataname + 'graph')

	def __len__(self):
		return len(self.fold_x)
	
	def __getitem__(self, index):
		id = self.fold_x[index]
		text = self.textDic[id]
		data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
		label = int(data['y'])
		return text, label

class Net(nn.Module):
	def __init__(self, input_size, num_class, bert, return_prob=False, temp=1):
		super(Net, self).__init__()
		self.bert = bert
		self.fc = nn.Linear(input_size, num_class)
		self.return_prob = return_prob
		self.temp = temp

	def forward(self, input_ids, attention_mask):
		last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
		x = last_hidden_states.last_hidden_state[:, 0]
		out = self.fc(x)
		log_prob = F.log_softmax(out/self.temp, dim=1)
		if self.return_prob:
			return log_prob, F.softmax(out/self.temp, dim=1)
		return log_prob


def train_bert():
	train_data = RootTextDataset(x_train_ids, textDic, args.datasetname, main_path)
	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
	test_data = RootTextDataset(x_test_ids, textDic, args.datasetname, main_path)
	test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

	tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
	bert = BertModel.from_pretrained(MODEL_PATH).to(device)
	model = Net(768, args.num_class, bert, return_prob=True, temp=args.temp).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	early_stopping = EarlyStopping(patience=args.patience)

	for epoch in range(args.n_epochs):
		model.train()
		avg_train_loss = 0
		train_pred = []
		train_y = []
		train_probs = []
		for Batch_data,Batch_data_y in train_loader:
			tokens = tokenizer(Batch_data, padding=True, truncation=True, max_length=512)            
			input_ids = torch.tensor(tokens["input_ids"]).to(device)
			attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
			Batch_data_y = Batch_data_y.to(device)
			optimizer.zero_grad()               # 梯度清零
			outputs, prob = model(input_ids,attention_mask)          # 前向传播
			loss=F.nll_loss(outputs,Batch_data_y)
			avg_train_loss += (loss.item()*outputs.shape[0])
			loss.backward()                     # 反向传播，计算梯度
			optimizer.step()
			_, pred = outputs.max(dim=-1)
			train_pred.extend(pred.tolist())
			train_y.extend(Batch_data_y.tolist())
			train_probs.extend(prob.tolist())
		# compute train acc
		train_acc = np.sum(np.array(train_pred) == np.array(train_y)) / len(train_y)
		# compute train ece
		train_ece = compute_ece(np.array(train_probs), np.array(train_y), args.num_class)


		torch.cuda.empty_cache()
		model.eval()
		output_y_list = []
		true_y_list = []
		val_probs = []
		avg_val_loss = 0
		for Batch_data,Batch_data_y in test_loader:
			tokens = tokenizer(Batch_data, padding=True, truncation=True, max_length=512)
			input_ids = torch.tensor(tokens["input_ids"]).to(device)
			attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
			Batch_data_y = Batch_data_y.to(device)
			outputs, prob = model(input_ids,attention_mask)          # 前向传播
			loss=F.nll_loss(outputs,Batch_data_y)
			# _, pred = outputs.max(dim=-1)
			avg_val_loss += (loss.item()*outputs.shape[0])
			output_y_list.extend(outputs.tolist())
			true_y_list.extend(Batch_data_y.tolist())
			val_probs.extend(prob.tolist())
		
		val_ece = compute_ece(np.array(val_probs), np.array(true_y_list), args.num_class)
		res = evaluation(output_y_list, true_y_list, args.num_class)
		print("{}\tfold{}\titer{}\tEpoch{:03d}\ttrain_loss:{:.4f}\ttrain_acc:{:.4f}\ttest_loss:{:.4f}\t".format(args.datasetname, fold, iter, epoch, 
																						avg_train_loss/len(x_train_ids), train_acc, avg_val_loss/len(x_test_ids)), "\t".join(str(r) for r in res))
		print(f'train ece={train_ece}\tval ece={val_ece}')
		model_path = os.path.join(args.model_save_path, "fold{}_iter{}_bert.pt".format(fold, iter))
		os.makedirs(os.path.dirname(model_path), exist_ok=True)
		if args.early_stopping:
			early_stopping(avg_val_loss/len(x_test_ids), res, model, model_path)
			if early_stopping.early_stop:
				print("Early stopping")
				res = early_stopping.res
				break
		else:
			torch.save(model.state_dict(), model_path)
	return res

def parsearg():
	parser = argparse.ArgumentParser(description='BERT model parameters')
	parser.add_argument('--datasetname', type=str, default='Pheme', help='Name of the dataset')
	parser.add_argument('--num_class', type=int, default=2, help='Number of classes')
	parser.add_argument('--device', type=str, default='cuda:0', help='device')
	parser.add_argument('--res_log_file', type=str, required=True, help='File to log the results arrays')
	parser.add_argument('--model_save_path', type=str, required=True, help='path to save trained models, path+fold0_iter0_bert.pt')
	parser.add_argument('--text', type=str, default='full', help='load root, full, or aug text')
	parser.add_argument('--early_stopping', type=int, default=1, help='whether to use early stopping')
	
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
	parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
	parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
	parser.add_argument('--patience', type=int, default=5, help='Number of patience')
	
	parser.add_argument('--temp', type=float, default=1, help='temperature scaling')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	print(datetime.now(),"starts.")
	main_path = 'data/'
	args = parsearg()
	print(args)
	device = torch.device(args.device)
	textDic = loadText(args.datasetname, main_path, args.text)
	x_folds = load5PreFoldedData(args.datasetname.split('_')[0])


	if "Weibo" in args.datasetname:
		MODEL_PATH = 'bert-base-chinese'
	else:
		MODEL_PATH = 'bert-base-uncased'


	total_res = []
	for fold in range(5):
		for iter in range(5):
			x_train_ids, x_test_ids = x_folds[2*int(fold)+1], x_folds[2*int(fold)]
			res = train_bert()
			with open(args.res_log_file, 'a') as f:
				f.write("{}\tfold{}\titer{}\t".format(args.datasetname, fold, iter) + "\t".join(str(r) for r in res) + "\n")
			total_res.append(res)
	print("final results: ", "\t".join(str(r) for r in np.mean(total_res, axis=0)))


	print(datetime.now(),"ends.")
