import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import numpy as np
import copy
import sys
from Process.rand5fold import *
from tools.evaluate import *
from Process.process import *

from generators import CVAE,EndNodeSelector
# from generators_batch import EndNodeSelector

from pdb import set_trace
import time
import shutil

from train_gcn import GCN_main4class
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed) 
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2022)

_dir = ''
max_size_dic = {}
median_size_dic = {}
datasetname = sys.argv[1]
scale = {'Twitter15':2,
         'Twitter16':8,
         'Pheme':4,
         'Twitter15C2':2,
         'Twitter16C2':8,
         'Weibo':33}

with open(_dir+'data/statistics.txt', 'r') as load_f:
    for line in load_f:
        dataset, max_size, median = line.strip().split()
        median_size_dic[dataset] = int(median)
        max_size_dic[dataset] = median_size_dic[dataset] * scale[dataset]
        # the max_size_dic['Weibo'] is not consistent with exp settting in the paper.
        # in the paper, the max_size of Weibo is set ot the average of graphs,
        # which is 230
        print(dataset, 'target size:', max_size_dic[dataset], 'median:', median_size_dic[dataset])

# trees,_ = loadAttrTree(datasetname)
x_folds = load5PreFoldedData(datasetname.split('_')[0])
fold = sys.argv[2]
num_class = int(sys.argv[3])
if len(sys.argv) > 3:
    res_log_file = sys.argv[4]
else:
    res_log_file = None
x_train_ids, x_test_ids = x_folds[2*int(fold)+1], x_folds[2*int(fold)]

# lr=0.0005
lr=0.0001
weight_decay=1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("<-----------------", "start time:", time.asctime(), datasetname, "fold", fold, device, "----------------->")
num_epoch = 5
for epoch in range(num_epoch):
    saved_path = _dir+f"cr_data/{datasetname}/fold_{fold}/epoch_{epoch}/"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    saved_path = _dir+f"rl_data/{datasetname}/fold_{fold}L/epoch_{epoch}/"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # saved_path = _dir+f"rl_data/{datasetname}/fold_{fold}R/epoch_{epoch}/"
    # if not os.path.exists(saved_path):
    #     os.makedirs(saved_path)
print(f"rl_data saving to {saved_path}")

KPG = EndNodeSelector(5000, 64, 64, num_class, max_size_dic[datasetname.split('_')[0]], device, datasetname=datasetname, fold=fold, _dir=_dir).to(device)
optimizerK = torch.optim.Adam(KPG.parameters(), lr=lr, weight_decay=weight_decay)
ResponseG = CVAE(5001, 64, 64, 32, 4, bidirectional=True).to(device)
optimizerR = torch.optim.Adam(ResponseG.parameters(), lr=lr, weight_decay=weight_decay)
ce_loss = nn.CrossEntropyLoss()

num_step = 200
num_epoch_rc = 1
max_patience = 4
min_original_size = 2
# back_size = int(len(x_train_ids)/10)
init_epoch = -1
# if datasetname == 'Twitter15':
#     if fold == '2':
#         init_epoch = 2
#     else:
#         init_epoch = 3
best_acc_LL=best_acc_RR=best_acc_LR=best_acc_RL = 0
KPG_patience = 0
batchsize = 128
droprate = 0.2

if init_epoch >= 0:
    KPG.load_state_dict(torch.load(_dir+'gmodels/KPG{}_fold{}_epoch{}.m'.format(datasetname, fold, init_epoch)))
    ResponseG.load_state_dict(torch.load(_dir+'gmodels/RG{}_fold{}_epoch{}.m'.format(datasetname, fold, init_epoch)))
    with open(_dir+'gmodels/dist{}_fold{}_epoch{}.txt'.format(datasetname, fold, init_epoch), 'r') as load_f:
        for line in load_f:
            response_dist = eval(line)
    KPG.eval()
    ResponseG.eval()

treeDic, _ = loadAttrTree(datasetname)
sorted_traindata_roots = [k for k,v in sorted(treeDic.items(), key=lambda x: len(x[1]), reverse=True) if k in x_train_ids]
org_traindata_list, org_testdata_list = loadAttrData(datasetname, treeDic, sorted_traindata_roots, x_test_ids, 0,0)
structureDic, generatedG_info = init_info_dic(org_traindata_list, org_testdata_list)

# for epoch in range(init_epoch+1, init_epoch+2):
for epoch in range(init_epoch+1, num_epoch):
    # if epoch == init_epoch+1 and fold=='1':
    #     print("training process finished before.")
    # else:
    time_epoch_start = time.time()
    avg_losses = []
    avg_losses_num = []
    response_dist = {}
    
    augmentedTrees = copy.deepcopy(treeDic)
    augmentedStructures = copy.deepcopy(structureDic)

    for batch_idx in range(len(x_train_ids)//batchsize+1):
        KPG.train()
        Batchdata_list = []
        roots = []
        previous_data_idx_mapping = {k:k for k in range(batchsize)}
        augmented = {k:False for k in range(batchsize)}
        finished_num = 0
        augmentedOrgDatalist = list()
        start_nodes_lists = []

        time_batch_start = time.time()

        for step in range(num_step):
            time_data_step_start = time.time()

            if step == 0:
                for data_idx in range(batch_idx*batchsize, min((batch_idx+1)*batchsize, len(x_train_ids))):
                    org_data_i = copy.deepcopy(org_traindata_list[data_idx])
                    root_id = org_traindata_list.fold_x[data_idx]
                    original_size = org_data_i.x.shape[0]
                    structure = copy.deepcopy(augmentedStructures[root_id])
                    tree = copy.deepcopy(augmentedTrees[root_id])
                    if original_size <= step + min_original_size:
                        org_data_i, structure, tree = structure_augmentation(org_data_i, tree, structure, ResponseG, response_dist)
                        augmented[data_idx-batch_idx*batchsize] = True
                    augmentedOrgDatalist.append(org_data_i)
                    augmentedStructures[root_id], augmentedTrees[root_id] = structure, tree
                    data_i = KPG.init_graph(org_data_i, structure)
                    start_nodes_lists.append(copy.deepcopy(generatedG_info[root_id]['candidate_start_nodes_list']))
                    Batchdata_list.append(data_i)
                    roots.append(root_id)
                k = data_idx
                Batchdata_list_previous_epoch = copy.deepcopy(Batchdata_list)
                BatchDataset = BatchGraphDataset(roots, Batchdata_list)
            train_loader = DataLoader(BatchDataset, batch_size=batchsize, shuffle=False, num_workers=0)
            for Batchdata in train_loader:
                Batchdata = Batchdata.to(device)
                break
            # print(Batchdata, len(BatchDataset))
            if step == 0:
                previous_rewards = KPG.calculate_reward(Batchdata.detach().clone())
                patiences = torch.zeros_like(previous_rewards)

            e_prob, chosen_s_nodes, chosen_e_nodes, new_x, new_edges, node_num, e_mask, e_mask_all_nodes, new_s_mapping = KPG.forward(Batchdata)
            rewards = KPG.calculate_reward(Data(x=new_x, edge_index=new_edges, e_mask=e_mask, node_num=node_num, e_mask_all_nodes=e_mask_all_nodes,
                                all_edge_index=Batchdata.all_edge_index, s_mapping_index=new_s_mapping, batch=Batchdata.batch, ptr=Batchdata.ptr).to(device))
            # print(step, batch_idx, rewards.shape, previous_rewards.shape)
            rewards_diff_all = rewards - previous_rewards
            patiences[torch.where(rewards_diff_all < 0)]+=1
            rewards_diff = rewards_diff_all[range(rewards.shape[0]),Batchdata.y]

            gclass_prob = KPG.calculate_reward_feedback(new_x.to(device),new_edges.to(device),Batchdata.batch.to(device))
            gclass_prob = 1.5 - gclass_prob[range(rewards.shape[0]),Batchdata.y]
            # gclass_prob = torch.clamp(gclass_prob, 1, 1+0.025*min(step+1,20))

            ### split batch data
            loss = torch.tensor([],dtype=torch.float32).to(device)
            current_data_idx_mapping = {}
            Batchdata_list = []
            current_roots = []
            selected_data_idx = []
            for data_idx in range(Batchdata.batch.max()+1):
                augdata_i = copy.deepcopy(augmentedOrgDatalist[previous_data_idx_mapping[data_idx]])
                root_id = roots[previous_data_idx_mapping[data_idx]]
                augtree = copy.deepcopy(augmentedTrees[root_id])
                augstructure = copy.deepcopy(augmentedStructures[root_id])

                start_idx = Batchdata.ptr[data_idx]
                end_idx = Batchdata.ptr[data_idx+1]
                chosen_e_node = chosen_e_nodes[data_idx] - start_idx
                loss_i = F.nll_loss(e_prob[start_idx:end_idx].T, chosen_e_node.unsqueeze(0)).unsqueeze(0)
                loss = torch.cat([loss,loss_i], dim=0)

                x_i = new_x[start_idx:end_idx]
                edge_index_i = new_edges[:,torch.where((new_edges[0]>=start_idx)&(new_edges[0]<end_idx))[0]] - start_idx
                node_num_i = node_num[data_idx]
                e_mask_i = e_mask[start_idx:end_idx]
                e_mask_all_nodes_i = e_mask_all_nodes[start_idx:end_idx]
                s_mapping_i = new_s_mapping[:,torch.where((new_s_mapping[0]>=start_idx)&(new_s_mapping[0]<end_idx))[0]] - start_idx

                generated_edges = []
                for edge_idx in range(edge_index_i.shape[1]):
                    org_s_node = s_mapping_i[1][edge_index_i[0][edge_idx]].item() - max_size_dic[datasetname.split('_')[0]]
                    try:
                        org_e_node = s_mapping_i[1][edge_index_i[1][edge_idx]].item() - max_size_dic[datasetname.split('_')[0]]
                    except:
                        set_trace()
                    # if org_e_node not in start_nodes_lists[previous_data_idx_mapping[data_idx]]:   # deal with selected nodes in previous steps
                    #     start_nodes_lists[previous_data_idx_mapping[data_idx]].append(org_e_node)
                    generated_edges.append([org_s_node, org_e_node])
                if rewards_diff[data_idx] >= 0:
                    start_nodes_lists[previous_data_idx_mapping[data_idx]].append(chosen_e_node.item() - max_size_dic[datasetname.split('_')[0]])
                    response_lens = update_cr_pairs(generated_edges, root_id, datasetname, augtree, fold)
                    for context_len in response_lens.keys():
                        if context_len not in response_dist.keys():
                            response_dist[context_len] = []
                        response_dist[context_len].extend(response_lens[context_len])
                    
                ### check whether a generation process should be stopped.
                continue_cond = False
                for label in range(num_class):
                    if label == Batchdata.y[data_idx]:
                        if patiences[data_idx, label] < max_patience:
                            continue_cond = True
                    else:
                        if patiences[data_idx, label] == 0:
                            continue_cond = True
                if node_num_i < max_size_dic[datasetname.split('_')[0]]-1 and continue_cond:
                    current_data_idx_mapping[len(Batchdata_list)] = previous_data_idx_mapping[data_idx]
                    if augmented[previous_data_idx_mapping[data_idx]]:
                        if rewards_diff[data_idx] >= 0:
                            augdata_i, augtree = update_generated_node_feature(augdata_i, augtree, ResponseG, generated_edges)
                        else:
                            augdata_i, augtree = update_generated_node_feature(augdata_i, augtree, ResponseG, generated_edges[:-1])
                    original_size = augdata_i.x.shape[0]
                    if original_size <= step + min_original_size:
                        # set_trace()
                        augdata_i, augstructure, augtree = structure_augmentation(augdata_i, augtree, augstructure, ResponseG, response_dist)
                        augmented[previous_data_idx_mapping[data_idx]] = True
                        # print(root_id, start_nodes_lists[previous_data_idx_mapping[data_idx]], len(start_nodes_lists[previous_data_idx_mapping[data_idx]]))
                    augmentedOrgDatalist[previous_data_idx_mapping[data_idx]], augmentedStructures[root_id], augmentedTrees[root_id] = augdata_i, augstructure, augtree
                    if rewards_diff[data_idx] >= 0:
                        data_i = KPG.update_graph(augdata_i, augstructure, start_nodes_lists[previous_data_idx_mapping[data_idx]], edge_index_i, node_num_i, s_mapping_i).cpu()
                    else:
                        data_i = copy.deepcopy(Batchdata_list_previous_epoch[data_idx].cpu())
                    # print(data_idx,patiences[data_idx].tolist(), Batchdata.y[data_idx].item(),original_size, step + min_original_size,data_i)
                    Batchdata_list.append(data_i)
                    current_roots.append(root_id)
                    selected_data_idx.append(data_idx)
                else:
                    # if augmented[previous_data_idx_mapping[data_idx]]:
                    #     data_i = KPG.update_graph(augdata_i, augstructure, start_nodes_lists[previous_data_idx_mapping[data_idx]], edge_index_i, node_num_i, s_mapping_i)
                    #     set_trace()
                    finished_num += 1
                    org_node_num = org_traindata_list[batch_idx*batchsize+previous_data_idx_mapping[data_idx]].x.shape[0]
                    # print(step, root_id, augdata_i, len(generated_edges), 'avg loss:', np.mean(avg_losses), finished_num, node_num_i, patiences[previous_data_idx_mapping[data_idx], Batchdata.y[data_idx]])
                    # response_lens = update_cr_pairs(generated_edges, root_id, datasetname, augtree, fold, org_node_num, write=True)
                    if rewards_diff[data_idx] >= 0:
                        response_lens = update_cr_pairs(generated_edges, root_id, datasetname, augtree, fold, org_node_num, write=True)
                    else:
                        response_lens = update_cr_pairs(generated_edges[:-1], root_id, datasetname, augtree, fold, org_node_num, write=True)
                    print("Epoch {:02d}-{:02d}-{} {:04d}/{} | y:{}, loss:{:.4f}, reward:{:.4f}, rloss:{:.4f}, len(generated_edges):{}, len(start_nodes_list):{}, node_num:{}({}), patience: {}, augmented: {}, idx: {}".format(
                        epoch, step, batch_idx, finished_num-1, root_id, Batchdata.y[data_idx].item(), loss_i.item(), rewards_diff[data_idx].item(), (loss_i *  (-rewards_diff[data_idx]).exp()).item(), len(generated_edges), len(start_nodes_lists[previous_data_idx_mapping[data_idx]]), 
                        Batchdata_list_previous_epoch[data_idx].node_num.item(), org_node_num, patiences[data_idx], augmented[previous_data_idx_mapping[data_idx]], previous_data_idx_mapping[data_idx])
                        , )

            Batchdata_list_previous_epoch = copy.deepcopy(Batchdata_list)
            BatchDataset = BatchGraphDataset(current_roots, Batchdata_list)
    
            rewarded_loss = (loss * (-rewards_diff).exp() * gclass_prob).mean()
            avg_losses.append(rewarded_loss.item()*len(current_roots))
            avg_losses_num.append(len(current_roots))

            optimizerK.zero_grad()
            rewarded_loss.backward()
            optimizerK.step()

            patiences = patiences[selected_data_idx]
            # rewards[torch.where(rewards_diff < 0)[0]] = previous_rewards[torch.where(rewards_diff < 0)[0]]
            rewards[torch.where(rewards_diff < 0)[0],Batchdata.y[torch.where(rewards_diff < 0)[0]]]=previous_rewards[torch.where(rewards_diff < 0)[0],Batchdata.y[torch.where(rewards_diff < 0)[0]]]
            previous_rewards = rewards[selected_data_idx]
            previous_data_idx_mapping = copy.deepcopy(current_data_idx_mapping)

            time_data_step_end = time.time()
            print("Epoch {:02d}-{:03d} Batch {} | Time {:.4f}s | avg loss: {:.4f}".format(epoch, step, batch_idx, (time_data_step_end - time_data_step_start), np.sum(avg_losses)/np.sum(avg_losses_num)))

            # print(previous_data_idx_mapping)
            # print(step, "finished.", Batchdata.x.shape, finished_num, len(current_roots))
            if finished_num == len(roots):
                break
        time_batch_end = time.time()
        print("Epoch {:02d} Batch {}: Time {:.4f} mins".format(epoch, batch_idx, (time_batch_end-time_batch_start)/60))
        KPG.eval()
            
        train_losses = []
        ResponseG.train()
        for epoch_rc in range(num_epoch_rc):
            start_time = time.time()
            avg_loss = []
            # avg_losses_i = []
            num_trained_ids = 0
            for idx in range(k+1):
                batch_dic, G = read_batch_crdata(datasetname, sorted_traindata_roots[idx], fold, device)
                truthrate = 1-idx/len(x_train_ids)*0.5
                alpha = float(int(2*idx/len(x_train_ids)))
                label = org_traindata_list[idx].y.to(device)
                reward_G_org = KPG.calculate_reward_feedback(G['x'], G['edge_index'], torch.LongTensor([0]*G['x'].shape[0]).to(device)).squeeze(0)[label.item()]
                losses_i = []
                for bidx in batch_dic.keys():
                    if bidx==0:
                        root_feat = batch_dic[bidx]['responses'][0]
                        root = torch.LongTensor(root_feat).to(device)
                        # etype = torch.LongTensor(batch_dic[bidx]['etypes'][0]).to(device)
                        z, mu, logvar, decoded_r = ResponseG(root, None, root, truthrate=truthrate)
                        _, pred = decoded_r.max(dim=-1)
                        pred = pred.tolist()
                        new_x = torch.zeros(5000)
                        for xi in pred:
                            if xi < 5000:
                                new_x[xi] += 1.0
                        G['x'][0] = new_x
                        # output = KPG.calculate_reward_feedback(G, None, eval=True)
                        # class_loss = F.nll_loss(output, label)
                        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))
                        CE_loss = ce_loss(decoded_r, root[1:])/2
                        loss = KL_divergence + CE_loss # + class_loss + CE_loss/2
                        # loss = loss + KL_divergence
                        # optimizerR.zero_grad()
                        # loss.backward()
                        # optimizerR.step()
                        losses_i.append(loss)
                        avg_loss.append(loss.item())
                        start_i = 1
                        if len(batch_dic[bidx]['contexts']) == 1:
                            continue
                    else:
                        start_i = 0

                    loss = 0
                    loss_num = 0
                    for i in range(start_i, len(batch_dic[bidx]['contexts'])):
                        context = torch.LongTensor(batch_dic[bidx]['contexts'][i]).to(device)
                        response = torch.LongTensor(batch_dic[bidx]['responses'][i]).to(device)
                        # etype = torch.LongTensor(batch_dic[bidx]['etypes'][i]).to(device)
                        ridx = batch_dic[bidx]['ridx'][i]
                        z, mu, logvar, decoded_r = ResponseG(response, context, root, truthrate=truthrate)
                        _, pred = decoded_r.max(dim=-1)
                        pred = pred.tolist()
                        new_x = torch.zeros(5000)
                        for xi in pred:
                            if xi < 5000:
                                new_x[xi] += 1.0
                        G['x'][ridx] = new_x
                        
                        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))
                        CE_loss = ce_loss(decoded_r, response[1:])/2
                        loss = loss + KL_divergence + CE_loss
                        loss_num+=1

                    loss = loss/loss_num
                    losses_i.append(loss)
                    avg_loss.append(loss.item())
                
                reward_G_rg = KPG.calculate_reward_feedback(G['x'], G['edge_index'], torch.LongTensor([0]*G['x'].shape[0]).to(device)).squeeze(0)[label.item()]
                mean_loss = (reward_G_org - reward_G_rg).exp() * sum(losses_i)/len(losses_i)
                # avg_losses_i.append(mean_loss)
                # if len(avg_losses_i) == back_size or (idx == k and len(avg_losses_i)>0):
                optimizerR.zero_grad()
                    # avg_loss_i = sum(avg_losses_i)/len(avg_losses_i)
                    # avg_loss_i.backward()
                mean_loss.backward()
                optimizerR.step()
                    # avg_losses_i = []
                num_trained_ids += 1
            end_time = time.time()
            print("Epoch {:02d}-{:03d}-{:03d} takes {:.2f} mins to finish {} samples | Avg_train_loss: {:.4f}".format(epoch, k, epoch_rc, (end_time-start_time)/60, num_trained_ids, np.mean(avg_loss)))
            train_losses.append(np.mean(avg_loss))
        print("Response Generator avg training loss is {:.4f} in {:02d}-{:03d}\n".format(np.mean(train_losses), epoch, k))
        ResponseG.eval()

    time_epoch_end = time.time()
    print("Epoch {:02d}: Training time {:.4f} mins | Avg loss: {:.4f}".format(epoch, (time_epoch_end-time_epoch_start)/60, np.sum(avg_losses)/np.sum(avg_losses_num)))
    
    torch.save(KPG.state_dict(),_dir+'gmodels/KPG{}_fold{}_epoch{}.m'.format(datasetname, fold, epoch))
    torch.save(ResponseG.state_dict(),_dir+'gmodels/RG{}_fold{}_epoch{}.m'.format(datasetname, fold, epoch))
    fout = open(_dir+"gmodels/dist{}_fold{}_epoch{}.txt".format(datasetname, fold, epoch), "w")
    fout.write(repr(response_dist)+"\n")
    fout.close()

    #######################################################
    ########## Generate KPaths of training set ############
    #######################################################
    time_epoch_start = time.time()
    avg_losses = []
    avg_losses_num = []
    response_dist = {}
    train_pred_y, train_true_y = {}, {}
    
    augmentedTrees = copy.deepcopy(treeDic)
    augmentedStructures = copy.deepcopy(structureDic)

    for batch_idx in range(len(x_train_ids)//batchsize+1):
        Batchdata_list = []
        roots = []
        previous_data_idx_mapping = {k:k for k in range(batchsize)}
        augmented = {k:False for k in range(batchsize)}
        finished_num = 0
        augmentedOrgDatalist = list()
        start_nodes_lists = []

        time_batch_start = time.time()

        for step in range(num_step):
            time_data_step_start = time.time()

            if step == 0:
                result_generated_graphs = []
                for data_idx in range(batch_idx*batchsize, min((batch_idx+1)*batchsize, len(x_train_ids))):
                    org_data_i = copy.deepcopy(org_traindata_list[data_idx])
                    root_id = org_traindata_list.fold_x[data_idx]
                    original_size = org_data_i.x.shape[0]
                    structure = copy.deepcopy(augmentedStructures[root_id])
                    tree = copy.deepcopy(augmentedTrees[root_id])
                    if original_size <= step + min_original_size:
                        org_data_i, structure, tree = structure_augmentation(org_data_i, tree, structure, ResponseG, response_dist)
                        augmented[data_idx-batch_idx*batchsize] = True
                    augmentedOrgDatalist.append(org_data_i)
                    augmentedStructures[root_id], augmentedTrees[root_id] = structure, tree
                    data_i = KPG.init_graph(org_data_i, structure)
                    start_nodes_lists.append(copy.deepcopy(generatedG_info[root_id]['candidate_start_nodes_list']))
                    Batchdata_list.append(data_i)
                    roots.append(root_id)
                    result_generated_graphs.append([0] * 4)
                k = data_idx
                Batchdata_list_previous_epoch = copy.deepcopy(Batchdata_list)
                BatchDataset = BatchGraphDataset(roots, Batchdata_list)
            train_loader = DataLoader(BatchDataset, batch_size=batchsize, shuffle=False, num_workers=0)
            for Batchdata in train_loader:
                Batchdata = Batchdata.to(device)
                break
            if step == 0:
                previous_rewards = KPG.calculate_reward(Batchdata.detach().clone())
                patiences = torch.zeros_like(previous_rewards)

            e_prob, chosen_s_nodes, chosen_e_nodes, new_x, new_edges, node_num, e_mask, e_mask_all_nodes, new_s_mapping = KPG.forward(Batchdata)
            rewards = KPG.calculate_reward(Data(x=new_x, edge_index=new_edges, e_mask=e_mask, node_num=node_num, e_mask_all_nodes=e_mask_all_nodes,
                                all_edge_index=Batchdata.all_edge_index, s_mapping_index=new_s_mapping, batch=Batchdata.batch, ptr=Batchdata.ptr).to(device))
            # print(step, batch_idx, rewards.shape, previous_rewards.shape)
            rewards_diff_all = rewards - previous_rewards
            patiences[torch.where(rewards_diff_all < 0)]+=1
            rewards_diff = rewards_diff_all[range(rewards.shape[0]),Batchdata.y]

            gclass_prob = KPG.calculate_reward_feedback(new_x.to(device),new_edges.to(device),Batchdata.batch.to(device))
            gclass_prob = 1.5 - gclass_prob[range(rewards.shape[0]),Batchdata.y]
            # gclass_prob = torch.clamp(gclass_prob, 1, 1+0.025*min(step+1,20))

            ### split batch data
            loss = torch.tensor([],dtype=torch.float32).to(device)
            current_data_idx_mapping = {}
            Batchdata_list = []
            current_roots = []
            selected_data_idx = []
            for data_idx in range(Batchdata.batch.max()+1):
                augdata_i = copy.deepcopy(augmentedOrgDatalist[previous_data_idx_mapping[data_idx]])
                root_id = roots[previous_data_idx_mapping[data_idx]]
                augtree = copy.deepcopy(augmentedTrees[root_id])
                augstructure = copy.deepcopy(augmentedStructures[root_id])

                start_idx = Batchdata.ptr[data_idx]
                end_idx = Batchdata.ptr[data_idx+1]
                chosen_e_node = chosen_e_nodes[data_idx] - start_idx
                loss_i = F.nll_loss(e_prob[start_idx:end_idx].T, chosen_e_node.unsqueeze(0)).unsqueeze(0)
                loss = torch.cat([loss,loss_i], dim=0)

                x_i = new_x[start_idx:end_idx]
                edge_index_i = new_edges[:,torch.where((new_edges[0]>=start_idx)&(new_edges[0]<end_idx))[0]] - start_idx
                node_num_i = node_num[data_idx]
                e_mask_i = e_mask[start_idx:end_idx]
                e_mask_all_nodes_i = e_mask_all_nodes[start_idx:end_idx]
                s_mapping_i = new_s_mapping[:,torch.where((new_s_mapping[0]>=start_idx)&(new_s_mapping[0]<end_idx))[0]] - start_idx

                generated_edges = []
                for edge_idx in range(edge_index_i.shape[1]):
                    org_s_node = s_mapping_i[1][edge_index_i[0][edge_idx]].item() - max_size_dic[datasetname.split('_')[0]]
                    org_e_node = s_mapping_i[1][edge_index_i[1][edge_idx]].item() - max_size_dic[datasetname.split('_')[0]]
                    # if org_e_node not in start_nodes_lists[previous_data_idx_mapping[data_idx]]:   # deal with selected nodes in previous steps
                    #     start_nodes_lists[previous_data_idx_mapping[data_idx]].append(org_e_node)
                    generated_edges.append([org_s_node, org_e_node])
                start_nodes_lists[previous_data_idx_mapping[data_idx]].append(chosen_e_node.item() - max_size_dic[datasetname.split('_')[0]])
                response_lens = update_cr_pairs(generated_edges, root_id, datasetname, augtree, fold)
                for context_len in response_lens.keys():
                    if context_len not in response_dist.keys():
                        response_dist[context_len] = []
                    response_dist[context_len].extend(response_lens[context_len])
                
                ### check whether a generation process should be stopped.
                continue_cond = False
                for label in range(num_class):
                    if label == Batchdata.y[data_idx]:
                        if patiences[data_idx, label] < max_patience:
                            continue_cond = True
                        else:
                            # result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = copy.deepcopy(Batchdata_list_previous_epoch[data_idx])
                            result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = 1
                    else:
                        if patiences[data_idx, label] == 0:
                            continue_cond = True
                        if patiences[data_idx, label] == 1 and rewards_diff_all[data_idx, label]<0:
                            # result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = copy.deepcopy(Batchdata_list_previous_epoch[data_idx])
                            result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = 1
                if node_num_i < max_size_dic[datasetname.split('_')[0]]-1 and continue_cond:
                    current_data_idx_mapping[len(Batchdata_list)] = previous_data_idx_mapping[data_idx]
                    if augmented[previous_data_idx_mapping[data_idx]]:
                        augdata_i, augtree = update_generated_node_feature(augdata_i, augtree, ResponseG, generated_edges)
                    original_size = augdata_i.x.shape[0]
                    # print(data_idx, patiences[previous_data_idx_mapping[data_idx], Batchdata.y[data_idx]],original_size, step + min_original_size,augdata_i)
                    if original_size <= step + min_original_size:
                        # set_trace()
                        augdata_i, augstructure, augtree = structure_augmentation(augdata_i, augtree, augstructure, ResponseG, response_dist)
                        augmented[previous_data_idx_mapping[data_idx]] = True
                        # print(root_id, start_nodes_lists[previous_data_idx_mapping[data_idx]], len(start_nodes_lists[previous_data_idx_mapping[data_idx]]))
                    augmentedOrgDatalist[previous_data_idx_mapping[data_idx]], augmentedStructures[root_id], augmentedTrees[root_id] = augdata_i, augstructure, augtree
                    data_i = KPG.update_graph(augdata_i, augstructure, start_nodes_lists[previous_data_idx_mapping[data_idx]], edge_index_i, node_num_i, s_mapping_i)
                    Batchdata_list.append(data_i)
                    current_roots.append(root_id)
                    selected_data_idx.append(data_idx)
                    
                else:
                    if 0 in result_generated_graphs[previous_data_idx_mapping[data_idx]]:
                        data_i = KPG.update_graph(augdata_i, augstructure, start_nodes_lists[previous_data_idx_mapping[data_idx]], edge_index_i, node_num_i, s_mapping_i)
                    else:
                        data_i = copy.deepcopy(Batchdata_list_previous_epoch[data_idx])
                    train_true_y[root_id] = data_i.y.item()
                    output = KPG.calculate_reward_feedback(data_i.x.to(device), data_i.edge_index.to(device), torch.LongTensor([0]*data_i.x.shape[0]).to(device))
                    _, train_pred = output.max(dim=1)
                    train_pred_y[root_id] = train_pred.item()
                    # set_trace()
                    isGenerated = np.zeros_like(data_i.s_mapping[0].cpu())
                    for nidx, n in enumerate(data_i.s_mapping[0].tolist()):
                        if 'tag' in augtree[data_i.s_mapping[1].tolist()[nidx]-max_size_dic[datasetname.split('_')[0]]+1].keys():
                            isGenerated[n] = 1
                    np.savez(_dir+f'rl_data/{datasetname}/fold_{fold}L/epoch_{epoch}/{root_id}.npz', x=data_i.x[:data_i.node_num.item()].cpu().numpy(), 
                            edge_index=data_i.edge_index.cpu().numpy(), node_num=data_i.node_num.cpu().numpy(), 
                            y=data_i.y.cpu().numpy(), isGenerated=isGenerated)
                    # if patiences[data_idx, train_true_y[root_id]] < max_patience:
                    #     np.savez(_dir+f'rl_data/{datasetname}/fold_{fold}R/epoch_{epoch}/{root_id}.npz', x=data_i.x[:data_i.node_num.item()].cpu().numpy(), 
                    #         edge_index=data_i.edge_index.cpu().numpy(), node_num=data_i.node_num.cpu().numpy(), 
                    #         y=data_i.y.cpu().numpy(), isGenerated=isGenerated)
                    # else:
                    #     isGenerated = np.zeros_like(result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].s_mapping[0].cpu())
                    #     for nidx, n in enumerate(result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].s_mapping[0].tolist()):
                    #         if 'tag' in augtree[result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].s_mapping[1].tolist()[nidx]-max_size_dic[datasetname.split('_')[0]]+1].keys():
                    #             isGenerated[n] = 1
                    #     np.savez(_dir+f'rl_data/{datasetname}/fold_{fold}R/epoch_{epoch}/{root_id}.npz', x=result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].x[:result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].node_num.item()].cpu().numpy(), 
                    #         edge_index=result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].edge_index.cpu().numpy(), node_num=result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].node_num.cpu().numpy(), 
                    #         y=result_generated_graphs[previous_data_idx_mapping[data_idx]][train_true_y[root_id]].y.cpu().numpy(), isGenerated=isGenerated)
                    
                    finished_num += 1
                    org_node_num = org_traindata_list[batch_idx*batchsize+previous_data_idx_mapping[data_idx]].x.shape[0]
                    print("Train Epoch {:02d}-{:02d}-{} {:04d}/{} | y:{}, loss:{:.4f}, reward:{:.4f}, rloss:{:.4f}, len(generated_edges):{}, len(start_nodes_list):{}, node_num:{}({}), patience: {}, augmented: {}, idx: {}".format(
                        epoch, step, batch_idx, finished_num-1, root_id, Batchdata.y[data_idx].item(), loss_i.item(), rewards_diff[data_idx].item(), (loss_i *  (-rewards_diff[data_idx]).exp()).item(), len(generated_edges), len(start_nodes_lists[previous_data_idx_mapping[data_idx]]), 
                        data_i.node_num.item(), org_node_num, patiences[data_idx], augmented[previous_data_idx_mapping[data_idx]], previous_data_idx_mapping[data_idx])
                        , )

            Batchdata_list_previous_epoch = copy.deepcopy(Batchdata_list)
            BatchDataset = BatchGraphDataset(current_roots, Batchdata_list)
    
            rewarded_loss = (loss * (-rewards_diff).exp() * gclass_prob).mean()
            # avg_losses.append(rewarded_loss.item())
            avg_losses.append(rewarded_loss.item()*len(current_roots))
            avg_losses_num.append(len(current_roots))

            patiences = patiences[selected_data_idx]
            # rewards[torch.where(rewards_diff < 0)[0]] = previous_rewards[torch.where(rewards_diff < 0)[0]]
            rewards[torch.where(rewards_diff < 0)[0],Batchdata.y[torch.where(rewards_diff < 0)[0]]]=previous_rewards[torch.where(rewards_diff < 0)[0],Batchdata.y[torch.where(rewards_diff < 0)[0]]]
            previous_rewards = rewards[selected_data_idx]
            previous_data_idx_mapping = copy.deepcopy(current_data_idx_mapping)

            time_data_step_end = time.time()
            print("Train Epoch {:02d}-{:03d} Batch {} | Time {:.4f}s | avg loss: {:.4f}".format(epoch, step, batch_idx, (time_data_step_end - time_data_step_start), np.sum(avg_losses)/np.sum(avg_losses_num)))

            # print(previous_data_idx_mapping)
            # print(step, "finished.", Batchdata.x.shape, finished_num, len(current_roots))
            if finished_num == len(roots):
                break
        time_batch_end = time.time()
        print("Train Epoch {:02d} Batch {}: Time {:.4f} mins".format(epoch, batch_idx, (time_batch_end-time_batch_start)/60))
    
    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(list(train_pred_y.values()), list(train_true_y.values()))
    res = ['C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1), 'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2),
            'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc3, Prec3, Recll3, F3), 'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc4, Prec4, Recll4, F4)]
    time_epoch_end = time.time()
    print("Train Epoch {:02d} | Training time {:.4f} mins | Training rewarded loss: {:.4f} | Training acc: {:.4f}".format(epoch, (time_epoch_end-time_epoch_start)/60, np.sum(avg_losses)/np.sum(avg_losses_num), Acc_all))
    print(res)

    #######################################################
    ##########   Generate KPaths of test set   ############
    #######################################################
    time_epoch_start = time.time()
    avg_losses = []
    avg_losses_num = []
    val_pred_y, val_true_y, val_reward_dim = {}, {}, {}
    
    augmentedTrees = copy.deepcopy(treeDic)
    augmentedStructures = copy.deepcopy(structureDic)

    for batch_idx in range(len(x_test_ids)//batchsize+1):
        Batchdata_list = []
        roots = []
        previous_data_idx_mapping = {k:k for k in range(batchsize)}
        augmented = {k:False for k in range(batchsize)}
        finished_num = 0
        augmentedOrgDatalist = list()
        start_nodes_lists = []

        time_batch_start = time.time()

        for step in range(num_step):
            time_data_step_start = time.time()

            if step == 0:
                Batchdata_reward_dim = []
                result_generated_graphs = []
                for data_idx in range(batch_idx*batchsize, min((batch_idx+1)*batchsize, len(x_test_ids))):
                    org_data_i = copy.deepcopy(org_testdata_list[data_idx])
                    output = KPG.calculate_reward_feedback(org_data_i.x.to(device), org_data_i.edge_index.to(device), torch.LongTensor([0]*org_data_i.x.shape[0]).to(device))
                    root_id = org_testdata_list.fold_x[data_idx]
                    original_size = org_data_i.x.shape[0]
                    structure = copy.deepcopy(augmentedStructures[root_id])
                    tree = copy.deepcopy(augmentedTrees[root_id])
                    if original_size <= step + min_original_size:
                        org_data_i, structure, tree = structure_augmentation(org_data_i, tree, structure, ResponseG, response_dist)
                        augmented[data_idx-batch_idx*batchsize] = True
                    augmentedOrgDatalist.append(org_data_i)
                    augmentedStructures[root_id], augmentedTrees[root_id] = structure, tree
                    data_i = KPG.init_graph(org_data_i, structure)
                    start_nodes_lists.append(copy.deepcopy(generatedG_info[root_id]['candidate_start_nodes_list']))
                    Batchdata_list.append(data_i)
                    roots.append(root_id)
                    result_generated_graphs.append([0] * 4)
                    val_pred_y[root_id] = -1
                    val_true_y[root_id] = data_i.y.item()
                    _, val_pred = output.max(dim=1)
                    val_reward_dim[root_id] = val_pred.item()
                    Batchdata_reward_dim.append(val_reward_dim[root_id])
                Batchdata_y = torch.LongTensor(Batchdata_reward_dim)
                k = data_idx
                Batchdata_list_previous_epoch = copy.deepcopy(Batchdata_list)
                BatchDataset = BatchGraphDataset(roots, Batchdata_list)
            test_loader = DataLoader(BatchDataset, batch_size=batchsize, shuffle=False, num_workers=0)
            for Batchdata in test_loader:
                Batchdata = Batchdata.to(device)
                break
            if step == 0:
                previous_rewards = KPG.calculate_reward(Batchdata.detach().clone())
                patiences = torch.zeros_like(previous_rewards)

            e_prob, chosen_s_nodes, chosen_e_nodes, new_x, new_edges, node_num, e_mask, e_mask_all_nodes, new_s_mapping = KPG.forward(Batchdata)
            rewards = KPG.calculate_reward(Data(x=new_x, edge_index=new_edges, e_mask=e_mask, node_num=node_num, e_mask_all_nodes=e_mask_all_nodes,
                                all_edge_index=Batchdata.all_edge_index, s_mapping_index=new_s_mapping, batch=Batchdata.batch, ptr=Batchdata.ptr).to(device))
            # print(step, batch_idx, rewards.shape, previous_rewards.shape)
            rewards_diff_all = rewards - previous_rewards
            patiences[torch.where(rewards_diff_all < 0)]+=1
            rewards_diff = rewards_diff_all[range(rewards.shape[0]),Batchdata_y]

            gclass_prob = KPG.calculate_reward_feedback(new_x.to(device),new_edges.to(device),Batchdata.batch.to(device))
            gclass_prob = 1.5 - gclass_prob[range(rewards.shape[0]),Batchdata.y]
            # gclass_prob = torch.clamp(gclass_prob, 1, 1+0.025*min(step+1,20))

            ### split batch data
            loss = torch.tensor([],dtype=torch.float32).to(device)
            current_data_idx_mapping = {}
            Batchdata_list = []
            current_roots = []
            selected_data_idx = []
            for data_idx in range(Batchdata.batch.max()+1):
                augdata_i = copy.deepcopy(augmentedOrgDatalist[previous_data_idx_mapping[data_idx]])
                root_id = roots[previous_data_idx_mapping[data_idx]]
                augtree = copy.deepcopy(augmentedTrees[root_id])
                augstructure = copy.deepcopy(augmentedStructures[root_id])

                start_idx = Batchdata.ptr[data_idx]
                end_idx = Batchdata.ptr[data_idx+1]
                chosen_e_node = chosen_e_nodes[data_idx] - start_idx
                loss_i = F.nll_loss(e_prob[start_idx:end_idx].T, chosen_e_node.unsqueeze(0)).unsqueeze(0)
                loss = torch.cat([loss,loss_i], dim=0)

                x_i = new_x[start_idx:end_idx]
                edge_index_i = new_edges[:,torch.where((new_edges[0]>=start_idx)&(new_edges[0]<end_idx))[0]] - start_idx
                node_num_i = node_num[data_idx]
                e_mask_i = e_mask[start_idx:end_idx]
                e_mask_all_nodes_i = e_mask_all_nodes[start_idx:end_idx]
                s_mapping_i = new_s_mapping[:,torch.where((new_s_mapping[0]>=start_idx)&(new_s_mapping[0]<end_idx))[0]] - start_idx

                generated_edges = []
                for edge_idx in range(edge_index_i.shape[1]):
                    org_s_node = s_mapping_i[1][edge_index_i[0][edge_idx]].item() - max_size_dic[datasetname.split('_')[0]]
                    org_e_node = s_mapping_i[1][edge_index_i[1][edge_idx]].item() - max_size_dic[datasetname.split('_')[0]]
                    # if org_e_node not in start_nodes_lists[previous_data_idx_mapping[data_idx]]:   # deal with selected nodes in previous steps
                    #     start_nodes_lists[previous_data_idx_mapping[data_idx]].append(org_e_node)
                    generated_edges.append([org_s_node, org_e_node])
                start_nodes_lists[previous_data_idx_mapping[data_idx]].append(chosen_e_node.item() - max_size_dic[datasetname.split('_')[0]])

                ### check whether a generation process should be stopped.
                continue_cond = False
                for label in range(num_class):
                    if label == Batchdata_y[data_idx]:
                        if patiences[data_idx, label] < max_patience:
                            continue_cond = True
                        else:
                            # result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = copy.deepcopy(Batchdata_list_previous_epoch[data_idx])
                            result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = 1
                    else:
                        if patiences[data_idx, label] == 0:
                            continue_cond = True
                        if patiences[data_idx, label] == 1 and rewards_diff_all[data_idx, label]<0:
                            # result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = copy.deepcopy(Batchdata_list_previous_epoch[data_idx])
                            result_generated_graphs[previous_data_idx_mapping[data_idx]][label] = 1
                if node_num_i < max_size_dic[datasetname.split('_')[0]]-1 and continue_cond:
                    current_data_idx_mapping[len(Batchdata_list)] = previous_data_idx_mapping[data_idx]
                    if augmented[previous_data_idx_mapping[data_idx]]:
                        augdata_i, augtree = update_generated_node_feature(augdata_i, augtree, ResponseG, generated_edges)
                    original_size = augdata_i.x.shape[0]
                    
                    if original_size <= step + min_original_size:
                        augdata_i, augstructure, augtree = structure_augmentation(augdata_i, augtree, augstructure, ResponseG, response_dist)
                        augmented[previous_data_idx_mapping[data_idx]] = True
                        
                    augmentedOrgDatalist[previous_data_idx_mapping[data_idx]], augmentedStructures[root_id], augmentedTrees[root_id] = augdata_i, augstructure, augtree
                    data_i = KPG.update_graph(augdata_i, augstructure, start_nodes_lists[previous_data_idx_mapping[data_idx]], edge_index_i, node_num_i, s_mapping_i)
                    Batchdata_list.append(data_i)
                    current_roots.append(root_id)
                    selected_data_idx.append(data_idx)
                    
                else:
                    if 0 in result_generated_graphs[previous_data_idx_mapping[data_idx]]:
                        data_i = KPG.update_graph(augdata_i, augstructure, start_nodes_lists[previous_data_idx_mapping[data_idx]], edge_index_i, node_num_i, s_mapping_i)
                    else:
                        data_i = copy.deepcopy(Batchdata_list_previous_epoch[data_idx])
                    
                    output = KPG.calculate_reward_feedback(data_i.x.to(device), data_i.edge_index.to(device), torch.LongTensor([0]*data_i.x.shape[0]).to(device))
                    _, val_pred = output.max(dim=1)
                    val_pred_y[root_id] = val_pred.item()
                    
                    isGenerated = np.zeros_like(data_i.s_mapping[0].cpu())
                    for nidx, n in enumerate(data_i.s_mapping[0].tolist()):
                        if 'tag' in augtree[data_i.s_mapping[1].tolist()[nidx]-max_size_dic[datasetname.split('_')[0]]+1].keys():
                            isGenerated[n] = 1
                    np.savez(_dir+f'rl_data/{datasetname}/fold_{fold}L/epoch_{epoch}/{root_id}.npz', x=data_i.x[:data_i.node_num.item()].cpu().numpy(), 
                            edge_index=data_i.edge_index.cpu().numpy(), node_num=data_i.node_num.cpu().numpy(), 
                            y=data_i.y.cpu().numpy(), isGenerated=isGenerated)
                    # if patiences[data_idx, val_reward_dim[root_id]] < max_patience:
                    #     np.savez(_dir+f'rl_data/{datasetname}/fold_{fold}R/epoch_{epoch}/{root_id}.npz', x=data_i.x[:data_i.node_num.item()].cpu().numpy(), 
                    #         edge_index=data_i.edge_index.cpu().numpy(), node_num=data_i.node_num.cpu().numpy(), 
                    #         y=data_i.y.cpu().numpy(), isGenerated=isGenerated)
                    # else:
                    #     isGenerated = np.zeros_like(result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].s_mapping[0].cpu())
                    #     for nidx, n in enumerate(result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].s_mapping[0].tolist()):
                    #         if 'tag' in augtree[result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].s_mapping[1].tolist()[nidx]-max_size_dic[datasetname.split('_')[0]]+1].keys():
                    #             isGenerated[n] = 1
                    #     np.savez(_dir+f'rl_data/{datasetname}/fold_{fold}R/epoch_{epoch}/{root_id}.npz', x=result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].x[:result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].node_num.item()].cpu().numpy(), 
                    #         edge_index=result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].edge_index.cpu().numpy(), node_num=result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].node_num.cpu().numpy(), 
                    #         y=result_generated_graphs[previous_data_idx_mapping[data_idx]][val_reward_dim[root_id]].y.cpu().numpy(), isGenerated=isGenerated)
                    
                    finished_num += 1
                    org_node_num = org_traindata_list[batch_idx*batchsize+previous_data_idx_mapping[data_idx]].x.shape[0]
                    print("Test {:04d}/{} finished | true_{}==pred_{}({}):{}, loss:{:.4f}, reward:{:.4f}, rloss:{:.4f}, output:{}, len(generated_edges):{}, node_num:{}({}), patience: {}({}), augmented: {}, idx: {}".format(
                        finished_num-1, root_id, val_true_y[root_id], val_pred_y[root_id], val_reward_dim[root_id], val_true_y[root_id]==val_pred_y[root_id], loss_i.item(), rewards_diff[data_idx].item(), (loss_i *  (-rewards_diff[data_idx]).exp()).item(), output.cpu(), len(generated_edges), 
                        data_i.node_num.item(), org_node_num, patiences[data_idx][val_true_y[root_id]], patiences[data_idx][val_reward_dim[root_id]], augmented[previous_data_idx_mapping[data_idx]], previous_data_idx_mapping[data_idx])
                         )     

            Batchdata_list_previous_epoch = copy.deepcopy(Batchdata_list)
            BatchDataset = BatchGraphDataset(current_roots, Batchdata_list)
    
            rewarded_loss = (loss * (-rewards_diff).exp() * gclass_prob).mean()
            # avg_losses.append(rewarded_loss.item())
            avg_losses.append(rewarded_loss.item()*len(current_roots))
            avg_losses_num.append(len(current_roots))

            patiences = patiences[selected_data_idx]
            # rewards[torch.where(rewards_diff < 0)[0]] = previous_rewards[torch.where(rewards_diff < 0)[0]]
            rewards[torch.where(rewards_diff < 0)[0],Batchdata_y[torch.where(rewards_diff < 0)[0]]]=previous_rewards[torch.where(rewards_diff < 0)[0],Batchdata_y[torch.where(rewards_diff < 0)[0]]]
            Batchdata_y = Batchdata_y[selected_data_idx]
            previous_rewards = rewards[selected_data_idx]
            previous_data_idx_mapping = copy.deepcopy(current_data_idx_mapping)

            time_data_step_end = time.time()
            print("Test Epoch {:02d}-{:03d} Batch {} | Time {:.4f}s | avg loss: {:.4f}".format(epoch, step, batch_idx, (time_data_step_end - time_data_step_start), np.sum(avg_losses)/np.sum(avg_losses_num)))

            if finished_num == len(roots):
                break
        time_batch_end = time.time()
        print("Test Epoch {:02d} Batch {}: Time {:.4f} mins".format(epoch, batch_idx, (time_batch_end-time_batch_start)/60))

    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(list(val_reward_dim.values()), list(val_true_y.values()))
    res = ['C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1), 'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2),
            'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc3, Prec3, Recll3, F3), 'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc4, Prec4, Recll4, F4)]
    print("Test Epoch {:02d} | GCN_acc: {:.4f}".format(epoch, Acc_all))
    print(res)
    Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(list(val_pred_y.values()), list(val_true_y.values()))
    res = ['C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1), 'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2),
            'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc3, Prec3, Recll3, F3), 'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc4, Prec4, Recll4, F4)]
    time_epoch_end = time.time()
    print("Test Epoch {:02d} | Test time {:.4f} mins | Test rewarded loss: {:.4f} | Test acc: {:.4f}".format(epoch, (time_epoch_end-time_epoch_start)/60, np.sum(avg_losses)/np.sum(avg_losses_num), Acc_all))
    print(res)
    
    ### Test processed KPaths
    accs, F1, F2, F3, F4 = GCN_main4class(datasetname, device, fold+f"L/epoch_{epoch}", x_train_ids, x_test_ids, fold+f"L/epoch_{epoch}", fold+f"L/epoch_{epoch}", 5, modelname = 'KPGCN', TDdroprate=droprate, BUdroprate=droprate)
    print("{}-fold{} [Epoch {:2d} cascade|test|LL {}] acc:{:.4f} | F1:{:.4f} | F2:{:.4f} | F3:{:.4f} | F4:{:.4f}\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(datasetname, fold, epoch, droprate, accs, F1, F2, F3, F4, accs, F1, F2, F3, F4)) 
    if res_log_file is not None:
        with open(res_log_file, 'a') as f:
            f.write(f'fold{fold},epoch{epoch},'+','.join([str(accs), str(F1), str(F2), str(F3), str(F4)])+'\n')
    if accs >= best_acc_LL:
        best_acc_LL = accs
        print('In epoch {}, best_acc_LL of {}_fold{} is updated to {}.'.format(epoch,datasetname,fold,best_acc_LL))
    # elif KPG_patience > 0:
    #     break
    # else:
    #     KPG_patience += 1
    
    # accs, F1, F2, F3, F4 = GCN_main4class(datasetname, device, fold, x_train_ids, x_test_ids, fold+f"R/epoch_{epoch}", fold+f"R/epoch_{epoch}", 5, modelname = 'KPGCN', TDdroprate=droprate, BUdroprate=droprate)
    # print("{}-fold{} [Epoch {:2d} cascade|test|RR {}] acc:{:.4f} | F1:{:.4f} | F2:{:.4f} | F3:{:.4f} | F4:{:.4f}\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(datasetname, fold, epoch, droprate, accs, F1, F2, F3, F4, accs, F1, F2, F3, F4))
    # if accs >= best_acc_RR:
    #     best_acc_RR = accs
    #     print('In epoch {}, best_acc_RR of {}_fold{} is updated to {}.'.format(epoch,datasetname,fold,best_acc_RR))
        
    # accs, F1, F2, F3, F4 = GCN_main4class(datasetname, device, fold, x_train_ids, x_test_ids, fold+f"L/epoch_{epoch}", fold+f"R/epoch_{epoch}", 5, modelname = 'KPGCN', TDdroprate=droprate, BUdroprate=droprate)
    # print("{}-fold{} [Epoch {:2d} cascade|test|LR {}] acc:{:.4f} | F1:{:.4f} | F2:{:.4f} | F3:{:.4f} | F4:{:.4f}\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(datasetname, fold, epoch, droprate, accs, F1, F2, F3, F4, accs, F1, F2, F3, F4))
    # if accs >= best_acc_LR:
    #     best_acc_LR = accs
    #     print('In epoch {}, best_acc_LR of {}_fold{} is updated to {}.'.format(epoch,datasetname,fold,best_acc_LR))  
    
    # accs, F1, F2, F3, F4 = GCN_main4class(datasetname, device, fold, x_train_ids, x_test_ids, fold+f"R/epoch_{epoch}", fold+f"L/epoch_{epoch}", 5, modelname = 'KPGCN', TDdroprate=droprate, BUdroprate=droprate)
    # print("{}-fold{} [Epoch {:2d} cascade|test|RL {}] acc:{:.4f} | F1:{:.4f} | F2:{:.4f} | F3:{:.4f} | F4:{:.4f}\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(datasetname, fold, epoch, droprate, accs, F1, F2, F3, F4, accs, F1, F2, F3, F4))
    # if accs >= best_acc_RL:
    #     best_acc_RL = accs
    #     print('In epoch {}, best_acc_RL of {}_fold{} is updated to {}.'.format(epoch,datasetname,fold,best_acc_RL))
print("<----------------- BEST_ACC:", max([best_acc_LL, best_acc_RR, best_acc_LR, best_acc_RL]), "----------------->")
print("<-----------------", "end time:", time.asctime(), datasetname, "fold", fold, device, "----------------->")