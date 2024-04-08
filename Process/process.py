import os
from pdb import set_trace
from Process.dataset import *
from datetime import datetime
import random
import numpy as np
# from RWsampling import epsilon_random_walk
import torch
from tkinter import _flatten
cwd=os.getcwd()
_dir = ''
np.random.seed(2022)

################################### load tree#####################################
def loadTree(dataname):
    if "C2" in dataname:
        dataname = dataname.replace("C2", "")
    # if 'Twitter' in dataname:
    # treePath = os.path.join(cwd,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
    treePath = os.path.join(_dir,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
    print("reading {} tree".format(dataname))
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[6]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))
    

    # if dataname == "Weibo":
    #     # treePath = os.path.join(cwd,'data/Weibo/weibotree.txt')
    #     treePath = os.path.join(_dir,'data/'+dataname+'/data.TD_RvNN.vol_5000.txt')
    #     print("reading Weibo tree")
    #     treeDic = {}
    #     for line in open(treePath):
    #         line = line.rstrip()
    #         eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]),line.split('\t')[3]
    #         if not treeDic.__contains__(eid):
    #             treeDic[eid] = {}
    #         treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    #     print('tree no:', len(treeDic))
    return treeDic

# def loadAttrTree(dataname):
#     if "C2" in dataname:
#         dataname = dataname.replace("C2", "")
#     treeDic = {}
#     attrTreePath = os.path.join(_dir, 'data/'+dataname+'/data.details.complete.txt')
#     GMT_format = "%a %b %d %H:%M:%S +0800 %Y"
#     print("reading {} attr tree".format(dataname))
#     for line in open(attrTreePath):
#         line = line.rstrip()
#         if dataname == "Weibo":
#             eid, indexP, indexC, timestamp, author, node_type = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), datetime.strptime(line.split('\t')[4], GMT_format), eval(line.split('\t')[7]), line.split('\t')[-1]
#         else:
#             eid, indexP, indexC, timestamp, author, node_type = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), float(line.split('\t')[5]), eval(line.split('\t')[-2]), line.split('\t')[-1]
#         if not treeDic.__contains__(eid):
#             treeDic[eid] = {}
#         treeDic[eid][indexC] = {'parent': indexP}
#         if dataname == "Weibo":
#             if indexC == 1:
#                 root_timestamp = timestamp
#             treeDic[eid][indexC]['timedelta'] = round((timestamp - root_timestamp).seconds/3600, 4)+(timestamp - root_timestamp).days*24
#             # if treeDic[eid][indexC]['timedelta'] <= 0 and indexC != 1:
#             #     print(eid, indexC, treeDic[eid][indexC]['timedelta'], timestamp, root_timestamp)
#         else:
#             treeDic[eid][indexC]['timedelta'] = timestamp
#         treeDic[eid][indexC]['author'] = {}
#         if 'verified' in author.keys():
#             treeDic[eid][indexC]['author']['followers_count'] = author['followers_count']
#             treeDic[eid][indexC]['author']['follow_count'] = author['follow_count'] if 'follow_count' in author.keys() else author['following_count']
#             # if author['verified']:
#             #     treeDic[eid][indexC]['author']['verified'] = 2
#             # else:
#             #     treeDic[eid][indexC]['author']['verified'] = 1
#             # if author['gender'] == 'm':
#             #     treeDic[eid][indexC]['author']['gender'] = 1
#             # elif author['gender'] == 'f':
#             #     treeDic[eid][indexC]['author']['gender'] = 2
#         else:
#             treeDic[eid][indexC]['author']['followers_count'] = 0
#             treeDic[eid][indexC]['author']['follow_count'] = 0
#             # treeDic[eid][indexC]['author']['verified'] = 0
#             # treeDic[eid][indexC]['author']['gender'] = 0

#         treeDic[eid][indexC]['node_type'] = node_type
#         # if treeDic[eid][indexC]['author']['followers_count'] == 0:
#         #     print("followers_count:", eid, indexC)
#         # if treeDic[eid][indexC]['author']['follow_count'] == 0:
#         #     print("follow_count:", eid, indexC)
#         # if treeDic[eid][indexC]['author']['gender'] == 0 and treeDic[eid][indexC]['author']['verified'] != 0:
#         #     print(eid, indexC)

#     print('tree no:', len(treeDic))
#     return treeDic

def loadAttrTree(dataname):
    timebase = datetime(2023,1,1,0,0)
    treePath = _dir+'data/' + dataname + '/data.complete.index.txt'
    print("reading {} tree".format(dataname))
    if 'Weibo' in dataname:
        selected_author_attr = ['verified', 'verified_type', 'follow_count', 'followers_count']
        public_metric_len = 1
        GMT_format = "%a %b %d %H:%M:%S +0800 %Y"
    else:
        GMT_format = "%a %b %d %H:%M:%S +0000 %Y"
        selected_author_attr = ['verified', 'followers_count', 'following_count', 'tweet_count', 'listed_count', 'created_at']
        if 'Pheme'in dataname:
            selected_author_attr[2] = 'follow_count'
            public_metric_len = 2
        else:
            public_metric_len = 4
    authorVec = [0]*(len(selected_author_attr)+public_metric_len)
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        root_id,parent_idx,curr_idx,num_of_parents,maxL,time_stage,seq,text,created_at,timedelta,author_info_dict,user_type,public_metric = line.split('\t')[:13]
        if not treeDic.__contains__(root_id):
            treeDic[root_id] = {}
        author_info_dict = eval(author_info_dict)
        if 'followers_count' in author_info_dict.keys():
            if author_info_dict['verified']:
                author_info_dict['verified'] = 1
            else:
                author_info_dict['verified'] = 0
            if 'Weibo' not in dataname:
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

        treeDic[root_id][int(curr_idx)] = {'parent': parent_idx, 'maxL': maxL, 'vec': seq, 'authorVec': authorVec, 'node_type': user_type, 'text': text}
        
    print('tree no:', len(treeDic))
    return treeDic, len(authorVec)

def loadIdxTree(dataname):
    # treePath = 'data/'+dataname+'/data.TD_RvNN.vol_5000.txt'
    treePath = 'data/'+dataname.replace("C2", "")+'/data.index.vol_5000.txt'
    print("reading {} tree".format(dataname))
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC, uType = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[-1]
        # max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        Vec = line.split('\t')[6]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        # treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec, 'node_type': uType}
    print('tree no:', len(treeDic))
    
    # completeTree = 'data/'+dataname+'/data.details.complete.txt'
    # print("reading user type")
    # for line in open(completeTree):
    #     line = line.rstrip()
    #     eid, indexC, uType= line.split('\t')[0], int(line.split('\t')[2]), line.split('\t')[-1]
    #     treeDic[eid][indexC]['node_type'] = uType
    # print("user type finished.")
    return treeDic


def read_batch_crdata(dataset, rootid, fold, device, training=True, repeatrate=0.1, worddroprate=0.1):
    edge_path = _dir+'cr_data/{}/fold_{}/{}.txt'.format(dataset, fold, rootid)
    edgeindex = []
    # edgetype = []
    contexts = []
    responses = []
    with open(edge_path, 'r') as load_f:
        for line in load_f:
            edge = eval(line.strip().split("\t")[0])
            context = line.strip().split("\t")[1]
            response = eval(line.strip().split("\t")[2])
            # etype = eval(line.strip().split("\t")[-1])
            edgeindex.append(edge)
            # edgetype.append(etype)
            contexts.append(context)
            responses.append(response)
    original_len = len(edgeindex)
    data_x = torch.zeros((max(_flatten(edgeindex))+1, 5000))
    
    context_idx_batch = {}
    context2id = {}
    for idx in range(len(edgeindex)):
        if (not context_idx_batch.__contains__(edgeindex[idx][0])) and (not context2id.__contains__(contexts[idx])):
            context_idx_batch[edgeindex[idx][0]] = []
            if contexts[idx] == "None":
                context2id[repr(responses[idx])] = edgeindex[idx][0]
            context2id[contexts[idx]] = edgeindex[idx][0]
        key = context2id[contexts[idx]]
        context_idx_batch[key].append(idx)

    batch_dic = {}
    count = 0
    if not training:
        for context_idx in context_idx_batch.keys():
            if not batch_dic.__contains__(context_idx):
                batch_dic[context_idx] = {'contexts':[], 'responses':[],  'cidx':[], 'ridx':[]} # 'etypes':[],
            for idx in context_idx_batch[context_idx]:
                batch_dic[context_idx]['contexts'].append(eval(contexts[idx]))
                batch_dic[context_idx]['responses'].append(responses[idx])
                # batch_dic[context_idx]['etypes'].append(edgetype[idx])
                batch_dic[context_idx]['cidx'].append(context_idx)
                batch_dic[context_idx]['ridx'].append(idx)
        return batch_dic

    for context_idx in context_idx_batch.keys():
        if not batch_dic.__contains__(context_idx):
            batch_dic[context_idx] = {'contexts':[], 'responses':[],  'cidx':[], 'ridx':[]} # 'etypes':[],

        for idx in context_idx_batch[context_idx]:
            if eval(contexts[idx]) == None:
                batch_dic[context_idx]['contexts'].append(eval(contexts[idx]))
            else:
                context = [eval(contexts[idx])[0]]
                for token in eval(contexts[idx])[1:-1]:
                    if np.random.rand() > worddroprate:
                        context.append(token)
                        if np.random.rand() > 1 - repeatrate:
                            context.append(token)
                context.append(eval(contexts[idx])[-1])
                batch_dic[context_idx]['contexts'].append(context)

            response = [responses[idx][0]]
            for token in responses[idx][1:-1]:
                if np.random.rand() > worddroprate:
                    response.append(token)
                    if np.random.rand() > 1 - repeatrate:
                        response.append(token)
            response.append(responses[idx][-1])
            batch_dic[context_idx]['responses'].append(response)
            # batch_dic[context_idx]['etypes'].append(edgetype[idx])
            batch_dic[context_idx]['cidx'].append(context_idx)
            batch_dic[context_idx]['ridx'].append(idx)
            count+=1
    # print(f"<------------ {rootid} | {count}/{original_len} ------------>")
    
    for ki, k in enumerate(batch_dic.keys()):
        if k==0:
            root_feat = batch_dic[k]['responses'][0]
            for xi in root_feat:
                if xi < 5000:
                    data_x[0][xi] += 1.0
            edge_index = [[0], [0]]
            start_i = 1
            if len(batch_dic[k]['contexts']) == 1:
                continue
        else:
            start_i = 0
        for i in range(start_i, len(batch_dic[k]['contexts'])):
            response = batch_dic[k]['responses'][i]
            cidx = batch_dic[k]['cidx'][i]
            ridx = batch_dic[k]['ridx'][i]
            for xi in response:
                if xi < 5000:
                    data_x[ridx][xi] += 1.0
            edge_index[0].append(cidx)
            edge_index[1].append(ridx)
    G = {'x': data_x.to(device), 'edge_index': torch.LongTensor(edge_index).to(device)}
    return batch_dic, G

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
        idxs = sorted(structure[s_node])
        candidates = traindata.x[idxs]
        candidate_end_nodes_list.extend(idxs)

        generatedG_info[root_id] = {'finished_substructure': finished_substructure, 
                        'candidate_end_nodes_list': candidate_end_nodes_list, 
                        'candidate_start_nodes_list': candidate_start_nodes_list, 
                        'generated_edges': generated_edges}

        # X = torch.zeros((max_size_dic[datasetname], 5000), dtype=torch.float32)
        # s_mask = torch.BoolTensor([True for i in range(max_size_dic[datasetname] + len(candidates))])
        # e_mask = torch.BoolTensor([False for i in range(max_size_dic[datasetname] + len(candidates))])

        # X[0] = traindata.root
        # X = torch.concat([X, candidates]).numpy()
        # edge_index = np.array([[0],[0]])
        # s_mask[0] = False
        # e_mask[:max_size_dic[datasetname]] = True
        
        # np.savez(f'rl_data/{datasetname}/{root_id}.npz', x=X, edge_index=edge_index, s_mask=s_mask.numpy(), e_mask=e_mask.numpy(), num_nodes=np.array(1))
        
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

        # X = torch.zeros((max_size_dic[datasetname], 5000), dtype=torch.float32)
        # s_mask = torch.BoolTensor([True for i in range(max_size_dic[datasetname] + len(candidates))])
        # e_mask = torch.BoolTensor([False for i in range(max_size_dic[datasetname] + len(candidates))])

        # X[0] = testdata.root
        # X = torch.concat([X, candidates]).numpy()
        # edge_index = np.array([[0],[0]])
        # s_mask[0] = False
        # e_mask[:max_size_dic[datasetname]] = True
        
        # np.savez(f'rl_data/{datasetname}/{root_id}.npz', x=X, edge_index=edge_index, s_mask=s_mask.numpy(), e_mask=e_mask.numpy(), num_nodes=np.array(1))
    
    return structureDic, generatedG_info

def read_batch_data(dataset, rootid, num_samples=200, drop=0.2, dropout=0.1):
    edge_path = 'cr_pairs/{}/{}.txt'.format(dataset, rootid)
    edgeindex = []
    edgetype = []
    contexts = []
    responses = []
    with open(edge_path, 'r') as load_f:
        for line in load_f:
            edge = eval(line.strip().split("\t")[0])
            context = line.strip().split("\t")[1]
            response = eval(line.strip().split("\t")[2])
            etype = eval(line.strip().split("\t")[-1])
            edgeindex.append(edge)
            edgetype.append(etype)
            contexts.append(context)
            responses.append(response)
    original_len = len(edgeindex)
    context_idx_batch = {}
    context2id = {}
    for idx in range(len(edgeindex)):
        if (not context_idx_batch.__contains__(edgeindex[idx][0])) and (not context2id.__contains__(contexts[idx])):
            context_idx_batch[edgeindex[idx][0]] = []
            if contexts[idx] == "None":
                context2id[repr(responses[idx])] = edgeindex[idx][0]
            context2id[contexts[idx]] = edgeindex[idx][0]
        key = context2id[contexts[idx]]
        context_idx_batch[key].append(idx)
    
    for context_idx in context_idx_batch.keys():
        if len(context_idx_batch[context_idx]) > 0.8 * num_samples:
            context_idx_batch[context_idx] = sorted(random.sample(context_idx_batch[context_idx], k=int(0.8 * num_samples)))
            if (context_idx == 0) and (0 not in context_idx_batch[context_idx]):
                context_idx_batch[context_idx] = [0] + context_idx_batch[context_idx]

    candidate_idx = list(context_idx_batch.keys())
    weights = [len(context_idx_batch[k]) for k in candidate_idx]
    selected_idx = []
    num_pairs = 0
    num_1 = 0
    total_1 = weights.count(1)
    while candidate_idx:
        i = random.choices(candidate_idx,weights=weights)[0]
        # print(i)
        w = weights.pop(candidate_idx.index(i))
        if w == 1:
            num_1 += 1
        num_pairs += w
        candidate_idx.remove(i)
        selected_idx.append(i)
        if original_len >= num_samples:
            if num_pairs >= num_samples or num_1 > total_1 * drop:
                break
        elif total_1 >= 5:
            if num_1 > total_1 * drop:
                break
    
    batch_dic = {}
    count = 0
    # print(selected_idx)
    for context_idx in selected_idx:
        if not batch_dic.__contains__(context_idx):
            # if len(context_idx_batch[context_idx]) == 1 and np.random.rand() < drop:
            #     continue
            batch_dic[context_idx] = {'contexts':[], 'responses':[], 'etypes':[]}

        for idx in context_idx_batch[context_idx]:
            if eval(contexts[idx]) == None:
                batch_dic[context_idx]['contexts'].append(eval(contexts[idx]))
                # print(batch_dic[context_idx]['contexts'])
            else:
                context = [eval(contexts[idx])[0]]
                for token in eval(contexts[idx])[1:-1]:
                    if np.random.rand() > dropout:
                        context.append(token)
                context.append(eval(contexts[idx])[-1])
                batch_dic[context_idx]['contexts'].append(context)

            response = [responses[idx][0]]
            for token in responses[idx][1:-1]:
                if np.random.rand() > dropout:
                    response.append(token)
            response.append(responses[idx][-1])
            batch_dic[context_idx]['responses'].append(response)
            # batch_dic[context_idx]['contexts'].append(eval(contexts[idx]))
            # batch_dic[context_idx]['responses'].append(responses[idx])
            batch_dic[context_idx]['etypes'].append(edgetype[idx])
            count+=1
    print(f"<------------ {rootid} | {count}/{original_len} ------------>")
    return batch_dic


################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    # data_path=os.path.join(cwd, 'data', dataname+'graph')
    data_path=os.path.join(_dir, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    # data_path=os.path.join(cwd, 'data',dataname+'graph')
    data_path=os.path.join(_dir, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    # data_path = os.path.join(cwd,'data', dataname + 'graph')
    data_path = os.path.join(_dir,'data', dataname + 'graph')
    print("loading train set", )
    print(f"len of fold_x_train is {len(fold_x_train)}")
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    print(f"len of fold_x_test is {len(fold_x_test)}")
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiETData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    # data_path = os.path.join(cwd,'data', dataname + 'graph')
    data_path = os.path.join(_dir,'data', dataname + 'graph')
    print("loading train set", )
    print(f"len of fold_x_train is {len(fold_x_train)}")
    traindata_list = BiETGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    print(f"len of fold_x_test is {len(fold_x_test)}")
    testdata_list = BiETGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadSampledData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    # data_path = os.path.join(cwd,'data', dataname + 'graph')
    data_path = os.path.join(_dir,'data', dataname + 'graph')
    print("loading train set", )
    print(f"len of fold_x_train is {len(fold_x_train)}")
    traindata_list = SampledETGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, dataset=dataname, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    print(f"len of fold_x_test is {len(fold_x_test)}")
    testdata_list = SampledETGraphDataset(fold_x_test, treeDic, dataset=dataname, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadRLData(dataname, fold_x_train, fold_x_test, TDdroprate, BUdroprate, fold, train_fold, test_fold):
    # data_path = os.path.join(cwd,'data', dataname + 'graph')
    data_path = os.path.join(_dir,'rl_data', dataname, 'fold_'+train_fold)
    # print("loading train set", )
    # print(f"len of fold_x_train is {len(fold_x_train)}")
    print(f"loading train set: len of fold_x_train is {len(fold_x_train)} from {data_path}")
    traindata_list = RLGraphDataset(fold_x_train, tddroprate=TDdroprate, budroprate=BUdroprate, dataset=dataname, data_path=data_path)
    # print("train no:", len(traindata_list))
    # print("loading test set", )
    # print(f"len of fold_x_test is {len(fold_x_test)}")
    data_path = os.path.join(_dir,'rl_data', dataname, 'fold_'+test_fold)
    print(f"loading test set: len of fold_x_test is {len(fold_x_test)} from {data_path}")
    testdata_list = RLGraphDataset(fold_x_test, dataset=dataname, data_path=data_path)
    # print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadAttrData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    # data_path = os.path.join(cwd,'data', dataname + 'graph')
    data_path = os.path.join(_dir,'data', dataname + 'graph')
    print("loading train set", )
    print(f"len of fold_x_train is {len(fold_x_train)}")
    traindata_list = AttrGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    print(f"len of fold_x_test is {len(fold_x_test)}")
    testdata_list = AttrGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list