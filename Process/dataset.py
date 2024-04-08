import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from itertools import product
from pdb import set_trace


def sparse_to_narray(array, node_num):
    # max_node = int(max(max(array[0]), max(array[1])))
    x = np.zeros([node_num, 5000])
    for i in range(array.shape[1]):
        x[int(array[0][i]), int(array[1][i])] = int(array[2][i])
    return x


class BiETGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=1, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        node_num = len(self.treeDic[id].keys())
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0 and node_num > 1:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        attr_tree = self.treeDic[id]
        edge_dic = {}
        for ki, k in enumerate(product('01234', repeat=2)):
            edge_dic[k] = ki
        edge_types = []
        # pairs = []
        for i in range(len(new_edgeindex[0])):
            start_node = new_edgeindex[0][i] + 1
            end_node = new_edgeindex[1][i] + 1
            edge_types.append([edge_dic[(attr_tree[start_node]['node_type'], attr_tree[end_node]['node_type'])]])
            # pairs.append([[SOS_token]+[int(i) for i in attr_tree[start_node]['vec'].split()],[SOS_token]+[int(i) for i in attr_tree[end_node]['vec'].split()]])

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0 and node_num > 1:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        bu_edge_types = []
        # bu_pairs = []
        for i in range(len(bunew_edgeindex[0])):
            start_node = bunew_edgeindex[0][i] + 1
            end_node = bunew_edgeindex[1][i] + 1
            bu_edge_types.append([edge_dic[(attr_tree[start_node]['node_type'], attr_tree[end_node]['node_type'])]])
            # bu_pairs.append([[SOS_token]+[int(i) for i in attr_tree[start_node]['vec'].split()],[SOS_token]+[int(i) for i in attr_tree[end_node]['vec'].split()]])

        if 'Weibograph' in self.data_path or 'Phemegraph' in self.data_path:
            return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                y=torch.LongTensor([int(data['y'])-1]), root=torch.LongTensor(data['root']),
                rootindex=torch.LongTensor([int(data['rootindex'])]),
                edge_type=torch.LongTensor(edge_types), BU_edge_type=torch.LongTensor(bu_edge_types), num_relations=torch.LongTensor([len(edge_dic)])
             )
             
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]),
                edge_type=torch.LongTensor(edge_types), BU_edge_type=torch.LongTensor(bu_edge_types), num_relations=torch.LongTensor([len(edge_dic)])
             )

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=1, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        node_num = len(self.treeDic[id].keys())
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0 and node_num>1:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        
        if 'Weibograph' in self.data_path or 'Phemegraph' in self.data_path:
            return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])-1]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

def edgeAttrNormalization(edge_index, edge_attr):
    # print(edge_index)
    row = list(edge_index)[0]
    col = list(edge_index)[1]
    max_row = max(row)+1
    max_col = max(col)+1
    # length = max(max_row, max_col)
    # E_hat = torch.zeros((length, length))
    E_hat = torch.zeros((max_row, max_col))
    for i in range(len(row)):
        E_hat[row[i],col[i]] = edge_attr[i]
    # print("E_hat", E_hat.shape)
    E_hat = F.normalize(E_hat, p=1)
    # E = torch.zeros_like(E_hat)
    # print("E", E.shape)
    # for i in range(len(row)):
    #     for j in range(len(row)):
    #         result = 0
    #         for k in col:
    #             result = result + E_hat[row[i],k] * E_hat[row[j],k] / sum(E_hat[:,k])
    #             print(k, result)
    #         E[row[i],col[i]] = result
    # print(E)
    normalized_attr = []
    for i in range(len(row)):
        normalized_attr.append(E_hat[row[i],col[i]].item())
    return normalized_attr


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=1, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        # pdb.set_trace()

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        # print(index, id)
        node_num = len(self.treeDic[id].keys())
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0 and node_num > 1:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        # edge_attr = list()
        # attr_tree = self.treeDic[id]
        # duration = 0
        # # edge_dic = {}
        # # for ki, k in enumerate(product('01234', repeat=2)):
        # #     edge_dic[k] = ki
        # # edge_types = []
        # for i in range(len(new_edgeindex[0])):
        #     start_node = new_edgeindex[0][i] + 1
        #     end_node = new_edgeindex[1][i] + 1
        #     # edge_types.append(edge_dic[(attr_tree[start_node]['node_type'], attr_tree[end_node]['node_type'])])
        #     # log_ratio_followers_s = np.log(attr_tree[start_node]['author']['followers_count']) if attr_tree[start_node]['author']['followers_count'] > 0 else 0
        #     # log_ratio_follows_s = np.log(attr_tree[start_node]['author']['follow_count']) if attr_tree[start_node]['author']['follow_count'] > 0 else 0
        #     # log_ratio_followers_e = np.log(attr_tree[end_node]['author']['followers_count']) if attr_tree[end_node]['author']['followers_count'] > 0 else 0
        #     # log_ratio_follows_e = np.log(attr_tree[end_node]['author']['follow_count']) if attr_tree[end_node]['author']['follow_count'] > 0 else 0
        #     log_ratio_followers = np.log(attr_tree[end_node]['author']['followers_count']+2)/np.log(attr_tree[start_node]['author']['followers_count']+2)
        #     log_ratio_follows = np.log(attr_tree[start_node]['author']['follow_count']+2)/np.log(attr_tree[end_node]['author']['follow_count']+2)
        #     log_ratio_s = np.log(attr_tree[start_node]['author']['followers_count']+2)/np.log(attr_tree[start_node]['author']['follow_count']+2)
        #     log_ratio_e = np.log(attr_tree[end_node]['author']['followers_count']+2)/np.log(attr_tree[end_node]['author']['follow_count']+2)

        #     # print(start_node, end_node)
        #     # time_delta = attr_tree[end_node]['timedelta'] - attr_tree[start_node]['timedelta']
        #     time_delta_e = attr_tree[end_node]['timedelta']
        #     time_delta_s = attr_tree[start_node]['timedelta']
        #     # time_delta = time_delta_e - time_delta_s
        #     if attr_tree[end_node]['timedelta'] > duration:
        #         duration = attr_tree[end_node]['timedelta']
        #     # log_ratio_followers = np.sqrt((attr_tree[end_node]['author']['followers_count']+1)/(attr_tree[start_node]['author']['followers_count']+1))
        #     # log_ratio_follows = np.sqrt((attr_tree[end_node]['author']['follow_count']+1)/(attr_tree[start_node]['author']['follow_count']+1))
        #     # log_ratio_s = np.sqrt((attr_tree[start_node]['author']['followers_count']+1)/(attr_tree[start_node]['author']['follow_count']+1))
        #     # log_ratio_e = np.sqrt((attr_tree[end_node]['author']['followers_count']+1)/(attr_tree[end_node]['author']['follow_count']+1))
        #     # if attr_tree[start_node]['author']['verified'] == 0:
        #     #     if attr_tree[end_node]['author']['verified'] == 0:
        #     #         verified = 0
        #     #     elif attr_tree[end_node]['author']['verified'] == 1:
        #     #         verified = 1
        #     #     elif attr_tree[end_node]['author']['verified'] == 2:
        #     #         verified = 2
        #     # elif attr_tree[start_node]['author']['verified'] == 1:
        #     #     if attr_tree[end_node]['author']['verified'] == 0:
        #     #         verified = 3
        #     #     elif attr_tree[end_node]['author']['verified'] == 1:
        #     #         verified = 4
        #     #     elif attr_tree[end_node]['author']['verified'] == 2:
        #     #         verified = 5
        #     # elif attr_tree[start_node]['author']['verified'] == 2:
        #     #     if attr_tree[end_node]['author']['verified'] == 0:
        #     #         verified = 6
        #     #     elif attr_tree[end_node]['author']['verified'] == 1:
        #     #         verified = 7
        #     #     elif attr_tree[end_node]['author']['verified'] == 2:
        #     #         verified = 8
            
        #     # if attr_tree[start_node]['author']['gender'] == 0:
        #     #     if attr_tree[end_node]['author']['gender'] == 0:
        #     #         gender = 0
        #     #     elif attr_tree[end_node]['author']['gender'] == 1:
        #     #         gender = 1
        #     #     elif attr_tree[end_node]['author']['gender'] == 2:
        #     #         gender = 2
        #     # elif attr_tree[start_node]['author']['gender'] == 1:
        #     #     if attr_tree[end_node]['author']['gender'] == 0:
        #     #         gender = 3
        #     #     elif attr_tree[end_node]['author']['gender'] == 1:
        #     #         gender = 4
        #     #     elif attr_tree[end_node]['author']['gender'] == 2:
        #     #         gender = 5
        #     # elif attr_tree[start_node]['author']['gender'] == 2:
        #     #     if attr_tree[end_node]['author']['gender'] == 0:
        #     #         gender = 6
        #     #     elif attr_tree[end_node]['author']['gender'] == 1:
        #     #         gender = 7
        #     #     elif attr_tree[end_node]['author']['gender'] == 2:
        #     #         gender = 8

        #     edge_attr.append([
        #         time_delta_s,
        #         time_delta_e, 
        #         # log_ratio_followers_s, 
        #         # log_ratio_follows_s, 
        #         # log_ratio_followers_e, 
        #         # log_ratio_follows_e,
        #         log_ratio_followers, log_ratio_follows, 
        #         log_ratio_s, log_ratio_e
        #         ])
        # edge_attr = np.array(edge_attr)
        # # print(edge_attr.shape)
        # normalized_edge_attr = []
        # for i in range(edge_attr.shape[1]):
        #     # print(i)
        #     # print(edge_attr[:,i])
        #     if i <= 1:
        #         if duration > 0:
        #             normalized_time = np.exp(-edge_attr[:,i]/duration)
        #         else:
        #             normalized_time = np.exp(edge_attr[:,i])
        #         normalized_edge_attr.append(normalized_time)
        #     else:
        #         normalized_edge_attr.append(edgeAttrNormalization(new_edgeindex, edge_attr[:,i]))
        # normalized_edge_attr = np.array(normalized_edge_attr)

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0 and node_num > 1:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        # bu_edge_attr = list()
        # for i in range(len(bunew_edgeindex[0])):
        #     start_node = bunew_edgeindex[0][i] + 1
        #     end_node = bunew_edgeindex[1][i] + 1
        #     log_ratio_followers = np.log(attr_tree[end_node]['author']['followers_count']+2)/np.log(attr_tree[start_node]['author']['followers_count']+2)
        #     log_ratio_follows = np.log(attr_tree[start_node]['author']['follow_count']+2)/np.log(attr_tree[end_node]['author']['follow_count']+2)
        #     log_ratio_s = np.log(attr_tree[start_node]['author']['followers_count']+2)/np.log(attr_tree[start_node]['author']['follow_count']+2)
        #     log_ratio_e = np.log(attr_tree[end_node]['author']['followers_count']+2)/np.log(attr_tree[end_node]['author']['follow_count']+2)
        #     time_delta_e = attr_tree[end_node]['timedelta']
        #     time_delta_s = attr_tree[start_node]['timedelta']
        #     if attr_tree[end_node]['timedelta'] > duration:
        #         duration = attr_tree[end_node]['timedelta']
        #     bu_edge_attr.append([
        #         time_delta_s,
        #         time_delta_e, 
        #         log_ratio_followers, log_ratio_follows, 
        #         log_ratio_s, log_ratio_e
        #         ])
        # bu_edge_attr = np.array(bu_edge_attr)
        # # print(edge_attr.shape)
        # normalized_bu_edge_attr = []
        # for i in range(bu_edge_attr.shape[1]):
        #     # print(i)
        #     # print(edge_attr[:,i])
        #     if i <= 1:
        #         if duration > 0:
        #             normalized_time = np.exp(-edge_attr[:,i]/duration)
        #         else:
        #             normalized_time = np.exp(edge_attr[:,i])
        #         normalized_bu_edge_attr.append(normalized_time)
        #     else:
        #         normalized_bu_edge_attr.append(edgeAttrNormalization(bunew_edgeindex, bu_edge_attr[:,i]))
        # normalized_bu_edge_attr = np.array(normalized_bu_edge_attr)
        # set_trace()
        if 'Weibograph' in self.data_path or 'Phemegraph' in self.data_path:
            return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                y=torch.LongTensor([int(data['y'])-1]), root=torch.LongTensor(data['root']),
                rootindex=torch.LongTensor([int(data['rootindex'])]), 
                #  edge_attr=torch.tensor(normalized_edge_attr,dtype=torch.float32).T,
                #  BU_edge_attr=torch.tensor(normalized_bu_edge_attr,dtype=torch.float32).T
                # edge_type=torch.LongTensor(edge_types), num_relations=torch.LongTensor([len(edge_dic)])
             )

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]),
                #  edge_attr=torch.tensor(normalized_edge_attr,dtype=torch.float32).T,
                #  BU_edge_attr=torch.tensor(normalized_bu_edge_attr,dtype=torch.float32).T
                 )


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        node_num = len(self.treeDic[id].keys())
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]
        
        if 'Weibograph' in self.data_path or 'Phemegraph' in self.data_path:
            return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])-1]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))


class SampledETGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=1, upper=100000, tddroprate=0,budroprate=0, dataset="Weibo",
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.sampled_edge_path = os.path.join('sampled_edges', dataset.replace("C2",""))
        self.sampled_cr_path = os.path.join('cr_pairs', dataset.replace("C2",""))
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        # pdb.set_trace()

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        node_num = len(self.treeDic[id].keys())
        # edge_dic = {}
        # for ki, k in enumerate(product('01234', repeat=2)):
        #     edge_dic[k] = ki
        # etype_keys = list(edge_dic.keys())
        # # <----- Bi-directional ----->
        # tb_etype_mapping = {}
        # for k in etype_keys:
        #     tb_etype_mapping[edge_dic[k]] = edge_dic[(k[1],k[0])]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edge_path = os.path.join(self.sampled_edge_path, id+".txt")
        # edge_path = os.path.join(self.sampled_cr_path, id+".txt")
        edgeindex = []
        with open(edge_path, 'r') as load_f:
            for line in load_f:
                edge = eval(line.strip().split("\t")[0])
                edgeindex.append(edge)

        edgeindex = np.array(edgeindex).T
        if self.tddroprate > 0 and node_num > 1:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        attr_tree = self.treeDic[id]
        edge_dic = {}
        for ki, k in enumerate(product('01234', repeat=2)):
            edge_dic[k] = ki
        edge_types = []
        # pairs = []
        for i in range(len(new_edgeindex[0])):
            start_node = new_edgeindex[0][i] + 1
            end_node = new_edgeindex[1][i] + 1
            edge_types.append([edge_dic[(attr_tree[start_node]['node_type'], attr_tree[end_node]['node_type'])]])
            # pairs.append([[SOS_token]+[int(i) for i in attr_tree[start_node]['vec'].split()],[SOS_token]+[int(i) for i in attr_tree[end_node]['vec'].split()]])

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0 and node_num > 1:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        bu_edge_types = []
        # bu_pairs = []
        for i in range(len(bunew_edgeindex[0])):
            start_node = bunew_edgeindex[0][i] + 1
            end_node = bunew_edgeindex[1][i] + 1
            bu_edge_types.append([edge_dic[(attr_tree[start_node]['node_type'], attr_tree[end_node]['node_type'])]])

        if 'Weibograph' in self.data_path or 'Phemegraph' in self.data_path:
            return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                y=torch.LongTensor([int(data['y'])-1]), root=torch.LongTensor(data['root']),
                rootindex=torch.LongTensor([int(data['rootindex'])]),
                edge_type=torch.LongTensor(edge_types), BU_edge_type=torch.LongTensor(bu_edge_types), num_relations=torch.LongTensor([len(edge_dic)])
             )
             
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]),
                edge_type=torch.LongTensor(edge_types), BU_edge_type=torch.LongTensor(bu_edge_types), num_relations=torch.LongTensor([len(edge_dic)])
             )

        # if 'Weibograph' in self.data_path or 'Phemegraph' in self.data_path:
        #     return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
        #             edge_index=torch.LongTensor(edgeindex).T,
        #      y=torch.LongTensor([int(data['y'])-1]), root=torch.LongTensor(data['root']),
        #      rootindex=torch.LongTensor([int(data['rootindex'])]), edge_type=torch.LongTensor(edgetype).squeeze(1),
        #      edge_attr=torch.tensor(edgeattr, dtype=torch.float32))

        # return Data(x=torch.tensor(data['x'],dtype=torch.float32),
        #             edge_index=torch.LongTensor(edgeindex).T,
        #      y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
        #      rootindex=torch.LongTensor([int(data['rootindex'])]), edge_type=torch.LongTensor(edgetype).squeeze(1),
        #      edge_attr=torch.tensor(edgeattr, dtype=torch.float32))


class RLGraphDataset(Dataset):
    def __init__(self, fold_x, tddroprate=0,budroprate=0, dataset="Weibo",
                 data_path=os.path.join('..','..', 'rl_data', 'Weibo')):
        self.fold_x = fold_x
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        node_num = int(data['node_num'])
        x = data['x'][:node_num]
        edgeindex = data['edge_index']

        if self.tddroprate > 0 and node_num > 1:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0 and node_num > 1:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
            
        return Data(x=torch.tensor(x,dtype=torch.float32), edge_index=torch.LongTensor(new_edgeindex),
                BU_edge_index=torch.LongTensor(bunew_edgeindex),y=torch.LongTensor([int(data['y'])]), 
                root=torch.tensor(x[0],dtype=torch.float32).unsqueeze(0),
             rootindex=torch.LongTensor([0]))
    

class AttrGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=1, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        # print(index, id)
        attr_tree = self.treeDic[id]
        node_num = len(attr_tree.keys())
        author_x = []
        author_y = []
        for node in range(node_num):
            # author_x.append(attr_tree[node+1]['authorVec'][1:])
            # author_y.append([int(attr_tree[node+1]['authorVec'][0])])
            author_x.append(attr_tree[node+1]['authorVec'])
            author_y.append([int(attr_tree[node+1]['authorVec'][0])])
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0 and node_num > 1:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0 and node_num > 1:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]

        try: 
            sparse_to_narray(data['x'], node_num)
        except:
            print(id)
        
        return Data(x=torch.tensor(sparse_to_narray(data['x'], node_num),dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.tensor(data['root'],dtype=torch.float32),
             rootindex=torch.LongTensor([int(data['rootindex'])]),
             bert_x = torch.tensor(data['bert_x'],dtype=torch.float32), bert_root = torch.tensor(data['bert_x'][data['rootindex']],dtype=torch.float32).unsqueeze(0),
             author_x = torch.tensor(author_x,dtype=torch.float32), author_root = torch.tensor(author_x[data['rootindex']],dtype=torch.float32).unsqueeze(0),
                author_y = torch.LongTensor(author_y)
             )