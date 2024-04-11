import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
import random
from itertools import product
from collections import Counter
import os
import numpy as np
from tkinter import _flatten
_dir = ""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
SOS_token = 5000

def context_prob_calculation(structure):
    parents = list(structure.keys())
    num_children = [len(structure[i]) for i in parents]
    if num_children == [0]:
        return {0:1}
    parent_weights = [i+1 for i in parents]
    parent_weight_sum = sum(parent_weights)
    if len(parent_weights) > 1:
        parent_weights = [(1-i/parent_weight_sum)/(len(parent_weights)-1) for i in parent_weights]
    else:
        parent_weights = [1]
    prob_dic = {}
    for i in range(len(num_children)):
        prob_i = num_children[i]/sum(num_children) * parent_weights[i]
        prob_dic[parents[i]] = prob_i/2
        for node in structure[parents[i]]:
            if node not in prob_dic.keys():
                prob_dic[node] = 0
            prob_dic[node] += (prob_i/2) / len(structure[parents[i]])
    return prob_dic

def structure_augmentation(tensor_data, tree, structure, model, response_lengths, target_candidates_num=5):
    node_num = len(tree.keys())
    data_x = tensor_data.x.cpu()
    data_edge_index = tensor_data.edge_index.T.tolist()
    data_BU_edge_index = tensor_data.BU_edge_index.T.tolist()
    root_feat = [SOS_token] + [int(item) for item in tree[1]['vec'].split()]
    root_feat = torch.LongTensor(root_feat).to(device)
    for i in range(target_candidates_num):
        prob_dic = context_prob_calculation(structure)
        context_node = random.choices(list(prob_dic.keys()), weights=list(prob_dic.values()))[0]
        context = [int(item) for item in tree[context_node+1]['vec'].split()]
        context_len =  len(context)
        if context_len not in response_lengths.keys():
            response_lengths_keys = list(response_lengths.keys())
            diff = np.abs(np.array(response_lengths_keys) - context_len).tolist()
            context_len = response_lengths_keys[diff.index(min(diff))]
        mu_r = np.mean(response_lengths[context_len])
        std_r = np.std(response_lengths[context_len])
        n_words = max(int(np.random.normal(mu_r, std_r)) + 1, min(response_lengths[context_len]))
        generated_response = model.generate(n_words, torch.LongTensor([SOS_token] + context).to(device), root_feat)
        if context_node not in structure.keys():
            structure[context_node] = []
        structure[context_node].append(node_num)
        _, pred = generated_response.max(dim=-1)
        pred = pred.tolist()
        vec = [str(item) for item in pred]
        tree[node_num+1] = {'parent': str(context_node+1),
                            'vec': " ".join(vec),
                            'tag': 'G'}
        new_x = torch.zeros(5000)
        for xi in pred:
            if xi < 5000:
                new_x[xi] += 1.0
        data_x = torch.concat([data_x, new_x.unsqueeze(0)], dim=0)
        data_edge_index.append([context_node, node_num])
        data_BU_edge_index.append([node_num, context_node])
        node_num += 1

    augmented_data = Data(x=data_x, edge_index=torch.LongTensor(data_edge_index).T, 
                        BU_edge_index=torch.LongTensor(data_BU_edge_index).T, y=tensor_data.y,
                        root=tensor_data.root, rootindex=tensor_data.rootindex)
    return augmented_data, structure, tree

def update_generated_node_feature(tensor_data, tree, model, selected_edges):
    selected_nodes = set(_flatten(selected_edges))
    data_x = tensor_data.x
    data_edge_index = tensor_data.edge_index.T.tolist()
    root_feat = [SOS_token] + [int(item) for item in tree[1]['vec'].split()]
    root_feat = torch.LongTensor(root_feat).to(device)
    for node in tree.keys():
        if 'tag' in tree[node].keys() and (node-1 not in selected_nodes):
            response = torch.LongTensor([SOS_token] + [int(item) for item in tree[node]['vec'].split()]).to(device)
            context_node = int(tree[node]['parent'])
            context = torch.LongTensor([SOS_token] + [int(item) for item in tree[context_node]['vec'].split()]).to(device)
            generated_response = model.generate(random.randint(len(response)//2, len(response)), context, root_feat)
            _, pred = generated_response.max(dim=-1)
            pred = pred.tolist()
            vec = [str(item) for item in pred]
            tree[node]['vec'] = " ".join(vec)
            new_x = torch.zeros(5000)
            for xi in pred:
                if xi < 5000:
                    new_x[xi] += 1.0
            data_x[node-1] = new_x
    data = Data(x=data_x, edge_index=torch.LongTensor(data_edge_index).T, 
                        BU_edge_index=tensor_data.BU_edge_index, y=tensor_data.y,
                        root=tensor_data.root, rootindex=tensor_data.rootindex)
    return data, tree

def update_cr_pairs(pairs, root_id, datasetname, tree, fold, org_node_num=1, write=False):
    num_pairs = len(pairs)
    # edge_index = data_i.edge_index.T.tolist()
    org_nodes = range(org_node_num)
    # selected_index = []
    # for pair in pairs:
    #     selected_index.append(edge_index.index(pair))
    response_len = {}
    if write:
        fout = open(_dir+f"cr_data/{datasetname}/fold_{fold}/{root_id}.txt", "w")
        fout.close()
    for i in range(num_pairs):
        if pairs[i][0] != pairs[i][1]:
            context = [SOS_token]+[int(token) for token in tree[pairs[i][0]+1]['vec'].split()]
        else:
            context = None
        response = [SOS_token]+[int(token) for token in tree[pairs[i][1]+1]['vec'].split()]
        if context:
            if len(context) - 1 not in response_len.keys():
                response_len[len(context) - 1] = []
            response_len[len(context) - 1].append(len(response)-1)
        if write:
            if pairs[i][0] not in org_nodes or pairs[i][1] not in org_nodes:
                # print(org_node_num, pairs[i][0], context, pairs[i][1], response)
                continue
            fout = open(_dir+f"cr_data/{datasetname}/fold_{fold}/{root_id}.txt", "a")
            fout.write(repr(pairs[i]) +"\t"+ repr(context) +"\t"+ repr(response) + "\n")
            fout.close()
    return response_len

def init_info_dic(traindata_list, testdata_list):
    structureDic = {}
    generatedG_info = {}
    for train_idx in range(len(traindata_list)):
        traindata = traindata_list[train_idx]
        root_id = traindata_list.fold_x[train_idx]
        generatedG_info[root_id] = {}
        structure = {}
        for i in range(traindata.edge_index.shape[1]):
            s = traindata.edge_index[0][i].item()
            e = traindata.edge_index[1][i].item()
            if s not in structure.keys():
                structure[s] = []
            # print(s,e)
            if s == e:
                continue
            if e not in structure[s]:
                structure[s].append(e)
        structureDic[root_id] = structure

        finished_substructure = [0]
        candidate_end_nodes_list = []
        candidate_start_nodes_list = [0]
        generated_edges = [[0,0]]
        s_node = 0
        try:
            idxs = sorted(structure[s_node])
        except:
            import pdb; pdb.set_trace()
        candidates = traindata.x[idxs]
        candidate_end_nodes_list.extend(idxs)

        generatedG_info[root_id] = {'finished_substructure': finished_substructure, 
                        'candidate_end_nodes_list': candidate_end_nodes_list, 
                        'candidate_start_nodes_list': candidate_start_nodes_list, 
                        'generated_edges': generated_edges}
    
    for test_idx in range(len(testdata_list)):
        testdata = testdata_list[test_idx]
        root_id = testdata_list.fold_x[test_idx]
        generatedG_info[root_id] = {}
        structure = {}
        for i in range(testdata.edge_index.shape[1]):
            s = testdata.edge_index[0][i].item()
            e = testdata.edge_index[1][i].item()
            if s not in structure.keys():
                structure[s] = []
            # print(s,e)
            if s == e:
                continue
            if e not in structure[s]:
                structure[s].append(e)
        structureDic[root_id] = structure

        finished_substructure = [0]
        candidate_end_nodes_list = []
        candidate_start_nodes_list = [0]
        generated_edges = [[0,0]]
        s_node = 0
        idxs = sorted(structure[s_node])
        candidates = testdata.x[idxs]
        candidate_end_nodes_list.extend(idxs)

        generatedG_info[root_id] = {'finished_substructure': finished_substructure, 
                        'candidate_end_nodes_list': candidate_end_nodes_list, 
                        'candidate_start_nodes_list': candidate_start_nodes_list, 
                        'generated_edges': generated_edges}

    return structureDic, generatedG_info

class BatchGraphDataset(Dataset):
    def __init__(self, fold_x, data_list, dataset="Weibo"):
        self.fold_x = fold_x
        self.data_list = data_list

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        data = self.data_list[index]
        return Data(x=data.x, edge_index=data.edge_index,e_mask=data.e_mask, node_num=data.node_num, e_mask_all_nodes=data.e_mask_all_nodes,
                all_edge_index=data.all_edge_index, s_mapping_index=data.s_mapping, y=data.y)
    
class CRPairDataset(Dataset):
    def __init__(self, fold_x, padding_size, padding_value, device, data_path=os.path.join('..','..', 'cr_data', 'Weibo')):
        self.fold_x = fold_x
        self.data_path = data_path
        self.padding_size = padding_size
        self.padding_value = padding_value
        self.device = device

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        edgeindex = []
        contexts = []
        responses = []
        context_lens = []
        response_lens = []
        node_feats = {}
        with open(os.path.join(self.data_path, id + ".txt"), 'r') as load_f:
            for line in load_f:
                edge,context,response = line.strip().split("\t")
                edgeindex.append(eval(edge))
                if context == 'None':
                    if len(eval(response)) > self.padding_size:
                        poslist = random.sample(range(len(eval(response))), self.padding_size)
                        poslist = sorted(poslist)
                        contexts.append(torch.LongTensor(eval(response))[poslist])
                    else:
                        contexts.append(torch.LongTensor(eval(response)))
                else:
                    if len(eval(context)) > self.padding_size:
                        poslist = random.sample(range(len(eval(context))), self.padding_size)
                        poslist = sorted(poslist)
                        contexts.append(torch.LongTensor(eval(context))[poslist])
                    else:
                        contexts.append(torch.LongTensor(eval(context)))
                context_lens.append(len(contexts[-1]))
                if len(eval(response)) > self.padding_size:
                    poslist = random.sample(range(len(eval(response))), self.padding_size)
                    poslist = sorted(poslist)
                    responses.append(torch.LongTensor(eval(response))[poslist])
                else:
                    responses.append(torch.LongTensor(eval(response)))

                response_lens.append(len(responses[-1]))
                if eval(edge)[1] not in node_feats.keys():
                    node_feats[eval(edge)[1]] = eval(response)

        new_edgeindex = torch.LongTensor(edgeindex).T.to(self.device)
        data_x = torch.zeros((new_edgeindex.max().item()+1, 5000),dtype=torch.float32).to(self.device)
        for node in node_feats.keys():
            for xi in node_feats[node]:
                if xi < 5000:
                    data_x[node, xi] += 1

        padded_contexts = pad_sequence(contexts, padding_value=self.padding_value, batch_first=True)
        padded_contexts = torch.cat([padded_contexts, torch.full((padded_contexts.shape[0],self.padding_size-padded_contexts.shape[1]), self.padding_value)],1).to(self.device)
        padded_responses = pad_sequence(responses, padding_value=self.padding_value, batch_first=True)
        padded_responses = torch.cat([padded_responses, torch.full((padded_responses.shape[0],self.padding_size-padded_responses.shape[1]), self.padding_value)],1).to(self.device)

        return Data(x=data_x, edge_index=new_edgeindex, responses=padded_responses, contexts=padded_contexts, 
                    response_lens = torch.LongTensor(response_lens),
                    context_lens = torch.LongTensor(context_lens),
                    node_num=torch.LongTensor([padded_responses.shape[0]]).to(self.device))
