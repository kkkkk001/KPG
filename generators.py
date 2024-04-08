import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from pdb import set_trace
import random
import os
import numpy as np
import copy
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch_geometric.data import Data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
SOS_token = 5000
np.random.seed(2022)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*3, output_size*2)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps*std

    def encode_context(self, context, root):
        if context != None:
            context = self.embed(context).unsqueeze(1)
            context, _ = self.gru(context)
            context = context[-1].squeeze(1)
            if self.bidirectional:
                context = context[:,self.hidden_size:]+context[:,:self.hidden_size]
            context = torch.cat((root, context), dim=1)
        else:
            context = torch.cat((root, torch.zeros_like(root).to(device)), dim=1)
        return context
    
    def forward(self, response, context, root):
        response = self.embed(response).unsqueeze(1)
        response, _ = self.gru(response)
        response = response[-1].squeeze(1)
        if self.bidirectional:
            response = response[:,self.hidden_size:]+response[:,:self.hidden_size]
        context = self.encode_context(context, root)
        output = torch.cat((response, context), dim=1)
        output = self.fc(output)
        mu, logvar = torch.chunk(output, 2, dim=1)
        z = self.sample(mu, logvar)
        return z, context, mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_size, context_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + z_size + context_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(z_size + context_size, hidden_size)
        self.fc2 = nn.Linear(z_size + context_size + hidden_size, output_size)
    
    def forward(self, z, n_words, response, context, truthrate=1):
        decoded_r = torch.zeros(n_words, 1, self.output_size).to(device)
        word_i = torch.LongTensor([SOS_token]).to(device)
        encoded_r = torch.cat((z, context), dim=1)
        h0 = self.fc1(encoded_r).unsqueeze(0)
        if self.bidirectional:
            h0 = torch.cat((h0,h0),dim=0)
        for i in range(1,n_words):
            decoded_r_i, h0 = self.cal_word_i(encoded_r, word_i, h0)
            decoded_r[i] = decoded_r_i
            # set_trace()
            word_i = decoded_r_i.topk(2)[1]
            if response != None:
                if np.random.rand() < truthrate:
                    word_i = response[i]
                else:
                    word_i = word_i[:,0] if word_i[:,0] != SOS_token else word_i[:,1]
            else:
                word_i = word_i[:,0] if word_i[:,0] != SOS_token else word_i[:,1]
            word_i = torch.LongTensor([word_i]).to(device)
        return decoded_r.squeeze(1)[1:, :self.output_size-1]

    def cal_word_i(self, encoded_r, word_i, h0):
        word_i = self.embed(word_i)
        word_i = torch.cat([word_i, encoded_r], 1).unsqueeze(0)
        output, h0 = self.gru(word_i, h0)
        output = output.squeeze(0)
        if self.bidirectional:
            output = output[:,self.hidden_size:]+output[:,:self.hidden_size]
        output = torch.cat((output, encoded_r), 1)
        output = self.fc2(output)
        return output, h0 

class CVAE(nn.Module):
    def __init__(self, vocab_size, encoder_hidden_size, decoder_hidden_size, z_size, num_elayers=1, num_dlayers=1, bidirectional=True):
        super(CVAE, self).__init__()
        self.z_size = z_size
        self.encoder = Encoder(vocab_size, encoder_hidden_size, z_size, num_elayers, bidirectional)
        self.decoder = Decoder(z_size, encoder_hidden_size*2, decoder_hidden_size, vocab_size, num_dlayers, bidirectional)
        self.RootEmbd = nn.Embedding(vocab_size, encoder_hidden_size)
        self.RootGRU = nn.GRU(encoder_hidden_size, encoder_hidden_size, bidirectional=bidirectional)
        self.encoder_hidden_size = encoder_hidden_size
        self.bidirectional = bidirectional

    def forward(self, response, context, root, truthrate=1):
        root = self.RootEmbd(root).unsqueeze(1)
        root, _ = self.RootGRU(root)
        root = root[-1].squeeze(1)
        if self.bidirectional:
            root = root[:,self.encoder_hidden_size:]+root[:,:self.encoder_hidden_size]
        z, encoded_c, mu, logvar = self.encoder(response, context, root)
        decoded = self.decoder(z, response.size(0), response, encoded_c, truthrate)
        return z, mu, logvar, decoded

    def generate(self, n_words, context, root):
        root = self.RootEmbd(root).unsqueeze(1)
        root, _ = self.RootGRU(root)
        root = root[-1].squeeze(1)
        if self.bidirectional:
            root = root[:,self.encoder_hidden_size:]+root[:,:self.encoder_hidden_size]
        context = self.encoder.encode_context(context, root)
        z = torch.randn((1, self.z_size)).to(device)
        decoded = self.decoder(z, n_words, None, context)
        return decoded
    
class pretrainedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_class):
        super(pretrainedGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.conv3 = GCNConv(input_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

        self.fc = torch.nn.Linear((output_dim*2), num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = scatter_mean(x, data.batch, dim=0)

        x1, bu_edge_index = data.x, data.BU_edge_index
        x1 = self.conv3(x1, bu_edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, training=self.training)
        x1 = self.conv4(x1, bu_edge_index)
        x1 = F.elu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)

        x = torch.cat((x,x1), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_class):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.conv3 = GCNConv(input_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

        self.fc = torch.nn.Linear((output_dim*2), num_class)

    def forward(self, x, edge_index, bu_edge_index, batch):
        x1 = copy.copy(x)
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = scatter_mean(x, batch, dim=0)

        x1 = self.conv3(x1, bu_edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, training=self.training)
        x1 = self.conv4(x1, bu_edge_index)
        x1 = F.elu(x1)
        x1 = scatter_mean(x1, batch, dim=0)

        x = torch.cat((x,x1), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class EndNodeSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_class, max_size, device, datasetname='Twitter16', fold='0',_dir=""):
        super(EndNodeSelector, self).__init__()

        self.feat_dim = input_dim
        self.num_class = num_class
        self.datasetname = datasetname
        self.max_size = max_size
        self.device = device

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(2*hidden_dim, output_dim)

        self.fc = nn.Linear(output_dim, 1)

        # Load GCN for calculating reward
        self.model = GCN(5000, 64, 64, num_class)
        self.model.load_state_dict(torch.load(_dir+'clfs/GCN/GCN{}_fold{}.m'.format(self.datasetname, fold)))
        print(_dir+'clfs/GCN/GCN{}_fold{}.m has been loaded.'.format(self.datasetname, fold))
        for param in self.model.parameters():
            param.requires_grad = False

    def init_graph(self, data, structure):
        X = torch.zeros((self.max_size, self.feat_dim), dtype=torch.float32)
        X[0] = data.root
        X = torch.cat([X, data.x], dim=0)
        e_mask_all_nodes = torch.BoolTensor([False]*X.shape[0])
        e_mask_all_nodes[:self.max_size+1]=True
        e_mask = torch.BoolTensor([True]*X.shape[0])
        edge_index = torch.LongTensor([[0],[0]])
        e_mask[np.array(structure[0])+self.max_size] = False
        all_edges = data.edge_index.detach().clone() + self.max_size
        s_mapping = torch.LongTensor([[0],[self.max_size]])
        return Data(x=X, edge_index=edge_index, e_mask=e_mask.unsqueeze(1), node_num=torch.LongTensor([1]), 
                    e_mask_all_nodes=e_mask_all_nodes.unsqueeze(1),
                    all_edge_index=all_edges, s_mapping=s_mapping, y=data.y)
    
    def update_graph(self, data, structure, start_nodes_list, edge_index, node_num, s_mapping):
        X = torch.zeros((self.max_size, self.feat_dim), dtype=torch.float32)
        X[:len(start_nodes_list)] = data.x[start_nodes_list]
        X = torch.cat([X, data.x], dim=0)
        e_mask_all_nodes = torch.BoolTensor([False]*X.shape[0])
        e_mask_all_nodes[:self.max_size]=True
        e_mask_all_nodes[np.array(start_nodes_list)+self.max_size] = True
        e_mask = torch.BoolTensor([True]*X.shape[0])
        for parent in start_nodes_list:
            if parent in structure.keys():
                e_mask[np.array(structure[parent])+self.max_size] = False
        e_mask[np.array(start_nodes_list)+self.max_size] = True
        all_edges = data.edge_index.detach().clone() + self.max_size
        return Data(x=X, edge_index=edge_index, e_mask=e_mask.unsqueeze(1), node_num=node_num.unsqueeze(0), 
                    e_mask_all_nodes=e_mask_all_nodes.unsqueeze(1),
                    all_edge_index=all_edges, s_mapping=s_mapping, y=data.y)


    def calculate_reward_feedback(self, new_x, new_edges, batch):
        edge_list = new_edges.tolist()
        bu_edge_index = torch.LongTensor([edge_list[1], edge_list[0]]).to(self.device)
        reward = self.model(new_x, new_edges, bu_edge_index, batch).exp()
        return reward

    def calculate_reward(self, Batchdata, steps=10):
        reward_G = self.calculate_reward_feedback(Batchdata.x, Batchdata.edge_index, Batchdata.batch)
        for t in range(steps):
            e_prob, chosen_s_nodes, chosen_e_nodes, new_x, new_edges, node_num, e_mask, e_mask_all_nodes, new_s_mapping = self.forward(Batchdata, 1)
            Batchdata = Data(x=new_x, edge_index=new_edges, e_mask=e_mask, node_num=node_num, e_mask_all_nodes=e_mask_all_nodes,
                        all_edge_index=Batchdata.all_edge_index, s_mapping_index=new_s_mapping, batch=Batchdata.batch, ptr=Batchdata.ptr).to(self.device)
            if t == 0:
                reward_Gt = self.calculate_reward_feedback(new_x, new_edges, Batchdata.batch)
            else:
                reward_Gt = reward_Gt + self.calculate_reward_feedback(new_x, new_edges, Batchdata.batch)
        return reward_G + reward_Gt/steps

    def forward(self, Batchdata, epsilon=0.8):
        x = Batchdata.x
        edge_index = Batchdata.edge_index
        e_mask = Batchdata.e_mask
        e_mask_all_nodes = Batchdata.e_mask_all_nodes
        all_edge_index = Batchdata.all_edge_index
        x = self.conv1(x,edge_index)
        x1 = torch.zeros_like(x,dtype=torch.float32).to(self.device)
        for idx in range(Batchdata.s_mapping_index.shape[1]):
            s_node_generatedG = Batchdata.s_mapping_index[0][idx]
            s_node_orgG = Batchdata.s_mapping_index[1][idx]
            candidates_idx = Batchdata.all_edge_index[1][torch.where(Batchdata.all_edge_index[0] == s_node_orgG)[0]]
            x1[candidates_idx] = x[s_node_generatedG]
        x = torch.cat([x, x1], dim=1)
        x = F.elu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        e_prob = self.fc(x)

        # chosen_s_nodes = torch.LongTensor([]).to(self.device)
        chosen_s_cand_nodes = torch.LongTensor([]).to(self.device)
        chosen_e_nodes = torch.LongTensor([]).to(self.device)

        if np.random.rand() > epsilon:
            e_prob = F.log_softmax(e_prob.masked_fill(e_mask_all_nodes, -1e9), dim=0)
            cond2 = (e_mask_all_nodes.squeeze(1)==False)
            # print('all')
        else:
            e_prob = F.log_softmax(e_prob.masked_fill(e_mask, -1e9), dim=0)
            cond2 = (e_mask.squeeze(1)==False)

        selected_data_idx = []
        for data_idx in range(Batchdata.batch.max()+1):
            cand_nodes = torch.where((Batchdata.batch==data_idx)&(cond2))[0]
            if len(cand_nodes) == 0 or Batchdata.node_num[data_idx] >= self.max_size-1:
                # print("continue",data_idx, len(cand_nodes), Batchdata.node_num[data_idx], "false num:", e_mask.tolist().count(False),e_mask_all_nodes.tolist().count(False))
                continue
            chosen_e_node = random.choices(cand_nodes, weights=torch.exp(e_prob[cand_nodes]))[0].unsqueeze(0)
            chosen_e_nodes = torch.cat((chosen_e_nodes,chosen_e_node))
            chosen_s_cand_node = all_edge_index[0][torch.where(all_edge_index[1]==chosen_e_node)[0]]
            chosen_s_cand_nodes = torch.cat((chosen_s_cand_nodes,chosen_s_cand_node))
            # chosen_s_node = Batchdata.s_mapping_index[0][torch.where(Batchdata.s_mapping_index[1]==chosen_s_cand_node)[0]]
            # chosen_s_nodes = torch.cat((chosen_s_nodes,chosen_s_node))
            selected_data_idx.append(data_idx)
        # print(chosen_s_cand_nodes,'\n',Batchdata.s_mapping_index[0])
        new_x = Batchdata.x.detach().clone()
        new_x[(Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx]] = Batchdata.x[chosen_e_nodes]
        # print(chosen_s_nodes.shape, chosen_e_nodes.shape)
        legal_cand_s_idx, legal_s_idx = torch.where(chosen_s_cand_nodes.unsqueeze(1).repeat(1,Batchdata.s_mapping_index.shape[1]) == Batchdata.s_mapping_index[1])
        chosen_s_nodes = Batchdata.ptr.detach().clone()
        chosen_s_nodes[legal_cand_s_idx] = Batchdata.s_mapping_index[0][legal_s_idx]
        # chosen_s_nodes = Batchdata.s_mapping_index[0][legal_s_idx]
        # print(legal_s_idx.shape, len(selected_data_idx))
        # print(chosen_s_nodes, '\n',legal_s_idx)
        # set_trace()
        new_edges = torch.cat((chosen_s_nodes[selected_data_idx].unsqueeze(0), (Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx].unsqueeze(0)),dim=0)
        # new_edges = torch.cat((chosen_s_nodes.unsqueeze(0), (Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx][legal_cand_s_idx].unsqueeze(0)),dim=0)
        new_edges = torch.cat((Batchdata.edge_index.detach().clone(),new_edges),dim=1)
        new_s_mapping = torch.cat(((Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx].unsqueeze(0), chosen_e_nodes.unsqueeze(0)), dim=0)
        new_s_mapping = torch.cat((Batchdata.s_mapping_index.detach().clone(),new_s_mapping),dim=1)
        node_num = Batchdata.node_num.detach().clone()
        node_num[selected_data_idx] = node_num[selected_data_idx] + 1
        e_mask = Batchdata.e_mask.detach().clone()
        e_mask[chosen_e_nodes] = True
        e_mask[all_edge_index[1, torch.where(chosen_e_nodes.unsqueeze(1).repeat(1,len(all_edge_index[0])) == all_edge_index[0])[1]]] = False
        e_mask_all_nodes = Batchdata.e_mask_all_nodes.detach().clone()
        e_mask_all_nodes[chosen_e_nodes] = True

        return e_prob, chosen_s_nodes, chosen_e_nodes, new_x, new_edges, node_num, e_mask, e_mask_all_nodes, new_s_mapping
    
    def ablation_forward(self, Batchdata):
        x = Batchdata.x
        edge_index = Batchdata.edge_index
        e_mask = Batchdata.e_mask
        e_mask_all_nodes = Batchdata.e_mask_all_nodes
        all_edge_index = Batchdata.all_edge_index
        # x = self.conv1(x,edge_index)
        # x1 = torch.zeros_like(x,dtype=torch.float32).to(self.device)
        # for idx in range(Batchdata.s_mapping_index.shape[1]):
        #     s_node_generatedG = Batchdata.s_mapping_index[0][idx]
        #     s_node_orgG = Batchdata.s_mapping_index[1][idx]
        #     candidates_idx = Batchdata.all_edge_index[1][torch.where(Batchdata.all_edge_index[0] == s_node_orgG)[0]]
        #     x1[candidates_idx] = x[s_node_generatedG]
        # x = torch.cat([x, x1], dim=1)
        # x = F.elu(x)
        # x = F.dropout(x)
        # x = self.conv2(x, edge_index)
        # x = F.elu(x)
        # e_prob = self.fc(x)
        e_prob = torch.ones((x.shape[0],1)).to(self.device)
        
        chosen_s_cand_nodes = torch.LongTensor([]).to(self.device)
        chosen_e_nodes = torch.LongTensor([]).to(self.device)

        e_prob = F.log_softmax(e_prob.masked_fill(e_mask, -1e9), dim=0)
        cond2 = (e_mask.squeeze(1)==False)

        selected_data_idx = []
        for data_idx in range(Batchdata.batch.max()+1):
            cand_nodes = torch.where((Batchdata.batch==data_idx)&(cond2))[0]
            if len(cand_nodes) == 0 or Batchdata.node_num[data_idx] >= self.max_size-1:
                # print("continue",data_idx, len(cand_nodes), Batchdata.node_num[data_idx], "false num:", e_mask.tolist().count(False),e_mask_all_nodes.tolist().count(False))
                continue
            chosen_e_node = random.choices(cand_nodes)[0].unsqueeze(0)
            chosen_e_nodes = torch.cat((chosen_e_nodes,chosen_e_node))
            chosen_s_cand_node = all_edge_index[0][torch.where(all_edge_index[1]==chosen_e_node)[0]]
            chosen_s_cand_nodes = torch.cat((chosen_s_cand_nodes,chosen_s_cand_node))
            selected_data_idx.append(data_idx)
        new_x = Batchdata.x.detach().clone()
        new_x[(Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx]] = Batchdata.x[chosen_e_nodes]
        legal_cand_s_idx, legal_s_idx = torch.where(chosen_s_cand_nodes.unsqueeze(1).repeat(1,Batchdata.s_mapping_index.shape[1]) == Batchdata.s_mapping_index[1])
        chosen_s_nodes = Batchdata.ptr.detach().clone()
        chosen_s_nodes[legal_cand_s_idx] = Batchdata.s_mapping_index[0][legal_s_idx]
        new_edges = torch.cat((chosen_s_nodes[selected_data_idx].unsqueeze(0), (Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx].unsqueeze(0)),dim=0)
        # new_edges = torch.cat((chosen_s_nodes.unsqueeze(0), (Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx][legal_cand_s_idx].unsqueeze(0)),dim=0)
        new_edges = torch.cat((Batchdata.edge_index.detach().clone(),new_edges),dim=1)
        new_s_mapping = torch.cat(((Batchdata.node_num+Batchdata.ptr[0:-1])[selected_data_idx].unsqueeze(0), chosen_e_nodes.unsqueeze(0)), dim=0)
        new_s_mapping = torch.cat((Batchdata.s_mapping_index.detach().clone(),new_s_mapping),dim=1)
        node_num = Batchdata.node_num.detach().clone()
        node_num[selected_data_idx] = node_num[selected_data_idx] + 1
        e_mask = Batchdata.e_mask.detach().clone()
        e_mask[chosen_e_nodes] = True
        e_mask[all_edge_index[1, torch.where(chosen_e_nodes.unsqueeze(1).repeat(1,len(all_edge_index[0])) == all_edge_index[0])[1]]] = False
        e_mask_all_nodes = Batchdata.e_mask_all_nodes.detach().clone()
        e_mask_all_nodes[chosen_e_nodes] = True

        return e_prob, chosen_s_nodes, chosen_e_nodes, new_x, new_edges, node_num, e_mask, e_mask_all_nodes, new_s_mapping