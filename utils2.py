import torch, random, json, os, argparse
import numpy as np
from datetime import datetime


def parsearg():
	parser = argparse.ArgumentParser(description='BERT model parameters')
	parser.add_argument('--seed', type=int, default=2023, help='seed')
	parser.add_argument('--datasetname', type=str, default='Pheme', help='Name of the dataset')
	parser.add_argument('--num_class', type=int, default=2, help='Number of classes')
	parser.add_argument('--device', type=str, default='cuda:0', help='device')
	
	parser.add_argument('--res_log_file', type=str, required=True, help='File to log the results arrays')
	parser.add_argument('--model_save_path', type=str, required=True, help='path to save trained models, path+fold0_iter0_bert.pt')
	parser.add_argument('--main_data_path', type=str, required=True, help='path to load data')
	parser.add_argument('--modelname', type=str, required=True, help='model: GCN, RDEA, GCN_cat')
	parser.add_argument('--best_epoch_path', type=str, help='')
	
	parser.add_argument('--droprate', type=float, default=0.4, help='droprate')
	parser.add_argument('--lr', type=float, help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
	parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
	parser.add_argument('--patience', type=int, default=20, help='Batch size for training')
	parser.add_argument('--return_prob', type=int, default=1, help='')

	args = parser.parse_args()
	return args


def loadText(dataname, main_path, text):
	if text.lower() == "root":
		return loadRootText(dataname, main_path)
	elif text.lower() == "full":
		return loadFullText(dataname, main_path)
	elif text.lower() == "aug":
		return loadGPTaugText(dataname, main_path)


def loadRootText(dataname, main_path):
	treePath = main_path + dataname + '/data.complete.index.txt'
	print("reading {} tree".format(dataname))
	textDic = {}
	for line in open(treePath):
		line = line.rstrip()
		root_id, text= line.split('\t')[0], line.split('\t')[7]
		if not textDic.__contains__(root_id):
			textDic[root_id] = text
	print(len(textDic), "texts were loaded.")
	return textDic


def loadFullText(dataname, main_path, filter=None):
	treePath = main_path + dataname + '/data.complete.index.txt'
	print("reading {} tree".format(dataname))
	textDic = {}
	for line in open(treePath):
		line = line.rstrip()
		root_id, text= line.split('\t')[0], line.split('\t')[7]
		if filter is not None:
			text = filter(text)
		if len(text):
			if textDic.__contains__(root_id):
				t = '[SEP]' + text
				textDic[root_id] += t
			else:
				textDic[root_id] = text
	print(len(textDic), "texts were loaded.")
	return textDic


def loadGPTaugText(dataname, main_path):
	path = os.path.join(main_path, dataname, 'aug_full_text.json')
	return json.load(open(path))


def load5PreFoldedData(obj):
	obj = obj.split('_')[0]
	if "C2" in obj:
		obj = obj.replace("C2", "")
	path = "nfold/"
	roots = []
	tags = ['test', 'train']
	for fold in range(5):
		for tag in tags:
			roots_i = []
			path_i = path+obj+'_fold'+str(fold)+'_x_'+tag+".txt"
			with open(path_i, 'r') as load_f:
				for line in load_f:
					roots_i.append(line.strip())
			roots.append(roots_i)
			print(path_i, "is finished! The len is", len(roots_i))
	return [roots[0], roots[1], \
			roots[2], roots[3], \
			roots[4], roots[5], \
			roots[6], roots[7], \
			roots[8], roots[9]]


def loadAttrTree(dataname, treepath_dir):
    timebase = datetime(2023,1,1,0,0)
    treePath = treepath_dir + dataname + '/data.complete.index.txt'
    print("reading {} tree from {}".format(dataname, treePath))
    if dataname == 'Weibo':
        selected_author_attr = ['verified', 'verified_type', 'follow_count', 'followers_count']
        public_metric_len = 1
        GMT_format = "%a %b %d %H:%M:%S +0800 %Y"
    else:
        GMT_format = "%a %b %d %H:%M:%S +0000 %Y"
        selected_author_attr = ['verified', 'followers_count', 'following_count', 'tweet_count', 'listed_count', 'created_at']
        if dataname == 'Pheme':
            selected_author_attr[2] = 'follow_count'
            public_metric_len = 2
        else:
            public_metric_len = 4
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        root_id,parent_idx,curr_idx,num_of_parents,maxL,time_stage,seq,text,created_at,timedelta,author_info_dict,user_type,public_metric = line.split('\t')[:13]
        authorVec = [0]*(len(selected_author_attr)+public_metric_len)
        if not treeDic.__contains__(root_id):
            treeDic[root_id] = {}
        author_info_dict = eval(author_info_dict)
        if 'followers_count' in author_info_dict.keys():
            if author_info_dict['verified']:
                author_info_dict['verified'] = 1
            else:
                author_info_dict['verified'] = 0
            if dataname != 'Weibo':
                account_created_at = datetime.strptime(author_info_dict['created_at'], GMT_format)
                for attr_idx, attr in enumerate(selected_author_attr):
                    authorVec[attr_idx] = author_info_dict[attr]
                    if attr == 'created_at':
                        authorVec[attr_idx] = (timebase - account_created_at).days*24 + (timebase - account_created_at).seconds/3600
            else:
                for attr_idx, attr in enumerate(selected_author_attr):
                    authorVec[attr_idx] = author_info_dict[attr]
        public_metric_values = list(eval(public_metric).values())
        if len(public_metric_values) > 0:
            authorVec[len(selected_author_attr):] = public_metric_values
        if created_at == '<missed>':
            tweet_lifetime = 0
        else:
            tweet_lifetime = (timebase -  datetime.strptime(created_at, GMT_format)).days*24 + (timebase -  datetime.strptime(created_at, GMT_format)).seconds/3600
        authorVec.append(tweet_lifetime)
        if timedelta == '<missed>':
            authorVec.append(0)
        else:
            authorVec.append(float(timedelta))
        if 'uid' in author_info_dict.keys():
            uid = author_info_dict['uid']
        else:
            if dataname != "Weibo":                    
                uid = author_info_dict['username'].lower()
            else:
                uid = author_info_dict['screen_name'].lower()
            if 'miss' in uid:
                uid = uid + root_id
        treeDic[root_id][int(curr_idx)] = {'parent': parent_idx, 'maxL': maxL, 'vec': seq, 'authorVec': authorVec, 'node_type': user_type, 'text': text,
                                           'uid': uid, 'time_stage': time_stage, 'time': created_at, 'timedelta': timedelta}
        
    print('tree no:', len(treeDic))
    return treeDic, len(authorVec)


class EarlyStopping:
	def __init__(self, patience=5, verbose=False):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.res = []

	def __call__(self, val_loss, res, model=None, model_path=None):
		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(res, model, model_path)
		elif score < self.best_score:
			self.counter += 1
			if self.verbose:
				print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(res, model, model_path)
			self.counter = 0


	def save_checkpoint(self, res, model, model_path):
		self.res = res
		if model is not None:
			if self.verbose:
				print('saving model to ', model_path)
			torch.save(model.state_dict(), model_path)



def evaluation2class(prediction, y):  # 2 dim
	TP1, FP1, FN1, TN1 = 0, 0, 0, 0
	TP2, FP2, FN2, TN2 = 0, 0, 0, 0
	for i in range(len(y)):
		Act, Pre = y[i], prediction[i]

		## for class 1
		if Act == 0 and Pre == 0: TP1 += 1
		if Act == 0 and Pre != 0: FN1 += 1
		if Act != 0 and Pre == 0: FP1 += 1
		if Act != 0 and Pre != 0: TN1 += 1
		## for class 2
		if Act == 1 and Pre == 1: TP2 += 1
		if Act == 1 and Pre != 1: FN2 += 1
		if Act != 1 and Pre == 1: FP2 += 1
		if Act != 1 and Pre != 1: TN2 += 1

	## print result
	Acc_all = round(float(TP1 + TP2) / float(len(y) ), 4)
	Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
	if (TP1 + FP1)==0:
		Prec1 =0
	else:
		Prec1 = round(float(TP1) / float(TP1 + FP1), 4)
	if (TP1 + FN1 )==0:
		Recll1 =0
	else:
		Recll1 = round(float(TP1) / float(TP1 + FN1 ), 4)
	if (Prec1 + Recll1 )==0:
		F1 =0
	else:
		F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 ), 4)

	Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
	if (TP2 + FP2)==0:
		Prec2 =0
	else:
		Prec2 = round(float(TP2) / float(TP2 + FP2), 4)
	if (TP2 + FN2 )==0:
		Recll2 =0
	else:
		Recll2 = round(float(TP2) / float(TP2 + FN2 ), 4)
	if (Prec2 + Recll2 )==0:
		F2 =0
	else:
		F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)

	return  Acc_all,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2

def evaluation4class(prediction, y):  # 4 dim
	TP1, FP1, FN1, TN1 = 0, 0, 0, 0
	TP2, FP2, FN2, TN2 = 0, 0, 0, 0
	TP3, FP3, FN3, TN3 = 0, 0, 0, 0
	TP4, FP4, FN4, TN4 = 0, 0, 0, 0
	# e, RMSE, RMSE1, RMSE2, RMSE3, RMSE4 = 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0
	for i in range(len(y)):
		Act, Pre = y[i], prediction[i]

		## for class 1
		if Act == 0 and Pre == 0: TP1 += 1
		if Act == 0 and Pre != 0: FN1 += 1
		if Act != 0 and Pre == 0: FP1 += 1
		if Act != 0 and Pre != 0: TN1 += 1
		## for class 2
		if Act == 1 and Pre == 1: TP2 += 1
		if Act == 1 and Pre != 1: FN2 += 1
		if Act != 1 and Pre == 1: FP2 += 1
		if Act != 1 and Pre != 1: TN2 += 1
		## for class 3
		if Act == 2 and Pre == 2: TP3 += 1
		if Act == 2 and Pre != 2: FN3 += 1
		if Act != 2 and Pre == 2: FP3 += 1
		if Act != 2 and Pre != 2: TN3 += 1
		## for class 4
		if Act == 3 and Pre == 3: TP4 += 1
		if Act == 3 and Pre != 3: FN4 += 1
		if Act != 3 and Pre == 3: FP4 += 1
		if Act != 3 and Pre != 3: TN4 += 1

	## print result
	Acc_all = round(float(TP1 + TP2 + TP3 + TP4) / float(len(y) ), 4)
	Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
	if (TP1 + FP1)==0:
		Prec1 =0
	else:
		Prec1 = round(float(TP1) / float(TP1 + FP1), 4)
	if (TP1 + FN1 )==0:
		Recll1 =0
	else:
		Recll1 = round(float(TP1) / float(TP1 + FN1 ), 4)
	if (Prec1 + Recll1 )==0:
		F1 =0
	else:
		F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 ), 4)

	Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
	if (TP2 + FP2)==0:
		Prec2 =0
	else:
		Prec2 = round(float(TP2) / float(TP2 + FP2), 4)
	if (TP2 + FN2 )==0:
		Recll2 =0
	else:
		Recll2 = round(float(TP2) / float(TP2 + FN2 ), 4)
	if (Prec2 + Recll2 )==0:
		F2 =0
	else:
		F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)

	Acc3 = round(float(TP3 + TN3) / float(TP3 + TN3 + FN3 + FP3), 4)
	if (TP3 + FP3)==0:
		Prec3 =0
	else:
		Prec3 = round(float(TP3) / float(TP3 + FP3), 4)
	if (TP3 + FN3 )==0:
		Recll3 =0
	else:
		Recll3 = round(float(TP3) / float(TP3 + FN3), 4)
	if (Prec3 + Recll3 )==0:
		F3 =0
	else:
		F3 = round(2 * Prec3 * Recll3 / (Prec3 + Recll3), 4)

	Acc4 = round(float(TP4 + TN4) / float(TP4 + TN4 + FN4 + FP4), 4)
	if (TP4 + FP4)==0:
		Prec4 =0
	else:
		Prec4 = round(float(TP4) / float(TP4 + FP4), 4)
	if (TP4 + FN4) == 0:
		Recll4 = 0
	else:
		Recll4 = round(float(TP4) / float(TP4 + FN4), 4)
	if (Prec4 + Recll4 )==0:
		F4 =0
	else:
		F4 = round(2 * Prec4 * Recll4 / (Prec4 + Recll4), 4)

	return  Acc_all,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2,Acc3, Prec3, Recll3, F3,Acc4, Prec4, Recll4, F4

def evaluation(output_list, label_list, num_class):
		outputs = torch.tensor(output_list)
		_, pred = outputs.max(dim=-1)

		if num_class == 4:
			Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(pred, label_list)
			res = [Acc_all, F1, F2, F3, F4]
		else:
			Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluation2class(pred, label_list)
			res = [Acc_all, F1, F2]
		return res



def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) 
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	 



def best_epoch(datasetname):
	best_epochs = json.load(open('best_epochs.json', 'r'))
	if datasetname in best_epochs:
		best_epochs = best_epochs[datasetname]
	else:
		print('no best epoch records for'+datasetname)
		best_epochs = {}
		for fold in range(5):
			best_epochs[str(fold)] = [0, 1, 2, 3, 4]
	return best_epochs

def update_best_epoch(datasetname, all_res):
	best_epochs = json.load(open('best_epochs.json', 'r'))
	if datasetname not in best_epochs:
		best_epochs[datasetname] = {}
		for fold in range(5):
			best_epochs[datasetname][str(fold)] = [int(np.argmax(np.array(all_res[str(fold)])[:,0]))]
		print('update the best epoch records for'+datasetname)
		print(best_epochs[datasetname])
		with open('best_epochs.json', 'w') as f:
			json.dump(best_epochs, f)


def compute_ece(probs, labels, n_bins=15):
	# if input is not a tensor, convert it to tensor
	if not torch.is_tensor(probs):
		probs = torch.from_numpy(probs)
	if not torch.is_tensor(labels):
		labels = torch.from_numpy(labels)
	# max_probs = probs[torch.arange(probs.shape[0]), labels]
	max_probs = probs.max(dim=1).values
	bin_boundaries = torch.linspace(0, 1, n_bins + 1)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]

	ece = 0.0

	for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
		in_bin = (max_probs > bin_lower.item()) * (max_probs <= bin_upper.item())
		prop_in_bin = in_bin.float().mean()
		if prop_in_bin.item() > 0:
			accuracy_in_bin = (labels[in_bin] == probs[in_bin].argmax(dim=1)).float().mean()
			avg_confidence_in_bin = probs[in_bin].max(dim=1).values.mean()
			ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

	return ece*100	
