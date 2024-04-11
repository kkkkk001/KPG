from Process.process import loadAttrData, loadAttrTree, loadRLData
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
# from tools.earlystopping import EarlyStopping
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import time
import warnings
from pdb import set_trace
from generators import pretrainedGCN
# from baseline_models import *
warnings.filterwarnings("ignore", category=UserWarning)

# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed) 
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True

# setup_seed(2023)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_class):
        super(GCN, self).__init__()
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
        x = x[data.rootindex]

        x1, bu_edge_index = data.x, data.BU_edge_index
        x1 = self.conv3(x1, bu_edge_index)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, training=self.training)
        x1 = self.conv4(x1, bu_edge_index)
        x1 = F.elu(x1)
        x1 = x1[data.rootindex]

        x = torch.cat((x,x1), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

def train(x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,n_epochs,batchsize,dataname,iter,fold,device,modelname, train_fold, test_fold, num_class):
    if 'pretrained' in modelname or 'baseline' in modelname:
        model = pretrainedGCN(5000,64,64,num_class).to(device)
    else:
        model = GCN(5000,64,64,num_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # if modelname == 'baselineBiGCN':
    #     model = BiGCN(5000,64,64, num_class, device).to(device)
    #     BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    #     BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    #     base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    #     optimizer = torch.optim.Adam([
    #         {'params':base_params},
    #         {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
    #         {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    #     ], lr=lr, weight_decay=weight_decay)
    # treeDic, autorchor_size=loadAttrTree(dataname)
    treeDic, autorchor_size=loadAttrTree(dataname.split('_')[0])
    if num_class == 4:
        from tools.earlystopping import EarlyStopping
    else:
        from tools.earlystopping2class import EarlyStopping
    early_stopping = EarlyStopping(patience=20, verbose=True)
    if TDdroprate == 0 and BUdroprate == 0:
        if 'pretrained' in modelname or 'baseline' in modelname:
            traindata_list, testdata_list = loadAttrData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        else:
            traindata_list, testdata_list = loadRLData(dataname, x_train, x_test, TDdroprate, BUdroprate, fold, train_fold, test_fold)
    for epoch in range(n_epochs):
        if TDdroprate > 0 or BUdroprate > 0:
            if 'pretrained' in modelname or 'baseline' in modelname:
                traindata_list, testdata_list = loadAttrData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
            else:
                traindata_list, testdata_list = loadRLData(dataname, x_train, x_test, TDdroprate, BUdroprate, fold, train_fold, test_fold)
        # set_trace()
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=10)
        model.train()
        batch_idx = 0
        for Batch_data in train_loader:
            Batch_data.to(device)
            out_labels= model(Batch_data)
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_idx = batch_idx + 1
            
        temp_val_losses = []
        pred_y_list = []
        true_y_list = []
        model.eval()
        for Batch_data in test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item()*val_out.shape[0])
            _, val_pred = val_out.max(dim=1)
            pred_y_list.extend(val_pred.tolist())
            true_y_list.extend(Batch_data.y.tolist())
        
        if num_class == 4:
            # if (epoch+1) % 10 == 0:
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(pred_y_list, true_y_list)
            res = ['Acc:{:.4f}'.format(Acc_all), 'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1), 'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2),
                'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc3, Prec3, Recll3, F3), 'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc4, Prec4, Recll4, F4)]
            # s = ''
            # if epoch == n_epochs-1:
            #     s = 'FINAL_OUTPUT:'
            print("{}(fold{}): Epoch {:05d}({}) | val_loss:{:.4f}, val_acc:{:.4f}\nres:{}".format(dataname, fold, epoch, iter, sum(temp_val_losses)/len(testdata_list), Acc_all, res))
            print("[{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]".format(Acc_all, F1, F2, F3, F4))
            early_stopping(sum(temp_val_losses)/len(testdata_list), Acc_all, F1, F2, F3, F4, model, modelname, dataname+"_fold"+str(fold), epoch)
            # if epoch == n_epochs-1:
            #     print("Training stops.")
            if early_stopping.early_stop or epoch == n_epochs-1:
                print("Early stopping")
                Acc_all=early_stopping.accs
                F1=early_stopping.F1
                F2 = early_stopping.F2
                F3 = early_stopping.F3
                F4 = early_stopping.F4
                print("BEST LOSS: {:.4f}| Accuracy: {:.4f} NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}"
                        .format(-early_stopping.best_score, early_stopping.accs,early_stopping.F1,early_stopping.F2,early_stopping.F3,early_stopping.F4))
                break
        elif num_class == 2:
            # if (epoch+1) % 10 == 0:
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(pred_y_list, true_y_list)
            res = ['Acc:{:.4f}'.format(Acc_all), 'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1), 'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2)]
            # s = ''
            # if epoch == n_epochs-1:
            #     s = 'FINAL_OUTPUT:'
            print("{}(fold{}): Epoch {:05d}({}) | val_loss:{:.4f}, val_acc:{:.4f}\nres:{}".format(dataname, fold, epoch, iter, sum(temp_val_losses)/len(testdata_list), Acc_all, res))
            print("[{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]".format(Acc_all,Prec1,Prec2,Recll1,Recll2,F1,F2))
            # if epoch >= 20:
            early_stopping(sum(temp_val_losses)/len(testdata_list), Acc_all, Acc1, Acc2, Prec1, Prec2, Recll1, Recll2, F1, F2, model, modelname, dataname+"_fold"+str(fold), epoch)
            
            # if epoch == n_epochs-1:
            #     print("Training stops.")
            if early_stopping.early_stop or epoch == n_epochs-1:
                print("Early stopping")
                Acc_all = early_stopping.accs
                Acc1 = early_stopping.acc1
                Acc2 = early_stopping.acc2
                Prec1 = early_stopping.pre1
                Prec2 = early_stopping.pre2
                Recll1 = early_stopping.rec1
                Recll2 = early_stopping.rec2
                F1 = early_stopping.F1
                F2 = early_stopping.F2
                print("BEST LOSS:{:.4f}| Accuracy: {:.4f}|pre1: {:.4f}|pre2: {:.4f}|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}"
                        .format(-early_stopping.best_score,early_stopping.accs,early_stopping.pre1,early_stopping.pre2,early_stopping.rec1,early_stopping.rec2,early_stopping.F1,early_stopping.F2))
                break
    # if 'pretrained' in modelname:
    if modelname == 'pretrainedGCN':
        modelname = modelname.replace('pretrained','')
        torch.save(model.state_dict(),f"clfs/{modelname}/"+modelname+dataname+"_fold"+fold+'.m')
    else:
        torch.save(model.state_dict(),f"clfs_testing/{modelname}"+dataname+"_fold"+fold+'.m')
    
    if num_class == 4:
        return Acc_all,F1,F2,F3,F4
    elif num_class == 2:
        return Acc_all, Prec1, Recll1, F1, Prec2, Recll2, F2

def GCN_main4class(datasetname, device, fold, fold_x_train, fold_x_test, train_fold, test_fold, iterations, modelname='pretrainedGCN', TDdroprate=0, BUdroprate=0, n_epochs=200):
    lr=0.0005
    weight_decay=1e-4
    patience=20
    
    batchsize=128
    # modelname = 'KPGCN'
    print('datasetname', datasetname, "| modelname:", modelname, "| device:", device)
    accs, F1, F2, F3, F4 = [],[],[],[],[]
    for iter in range(iterations):
        print("fold {} iteration {} starts.".format(fold, iter))
        accs0, F1_0, F2_0, F3_0, F4_0 = train(fold_x_test,
                                                    fold_x_train,
                                                    TDdroprate,BUdroprate,
                                                    lr, weight_decay,
                                                    n_epochs,
                                                    batchsize,
                                                    datasetname,
                                                    iter, fold, 
                                                    device, modelname, train_fold, test_fold, 4)                                                                         
        accs.append(accs0)
        F1.append(F1_0)
        F2.append(F2_0)
        F3.append(F3_0)
        F4.append(F4_0)

    return np.mean(accs),np.mean(F1),np.mean(F2),np.mean(F3),np.mean(F4)

def GCN_main2class(datasetname, device, fold, fold_x_train, fold_x_test, train_fold, test_fold, iterations, modelname='pretrainedGCN', TDdroprate=0, BUdroprate=0, n_epochs=200):
    lr=0.0001
    weight_decay=1e-4
    patience=20
    
    batchsize=128
    # modelname = 'KPGCN'
    print('datasetname', datasetname, "| modelname:", modelname, "| device:", device)
    accs, Prec1s, Recll1s, F1s, Prec2s, Recll2s, F2s = [],[],[],[],[],[],[]
    for iter in range(iterations):
        print("fold {} iteration {} starts.".format(fold, iter))
        Acc_all, Prec1, Recll1, F1, Prec2, Recll2, F2 = train(fold_x_test,
                                                    fold_x_train,
                                                    TDdroprate,BUdroprate,
                                                    lr, weight_decay,
                                                    n_epochs,
                                                    batchsize,
                                                    datasetname,
                                                    iter, fold, 
                                                    device, modelname, train_fold, test_fold, 2)                                                                         
        accs.append(Acc_all)
        Prec1s.append(Prec1)
        Recll1s.append(Recll1)
        Prec2s.append(Prec2)
        Recll2s.append(Recll2)
        F1s.append(F1)
        F2s.append(F2)

    return np.mean(accs),np.mean(Prec1s),np.mean(Recll1s),np.mean(F1s),np.mean(Prec2s),np.mean(Recll2s),np.mean(F2s)

if __name__ == '__main__':
    datasetname = sys.argv[1]
    x_folds = load5PreFoldedData(datasetname.split('_early')[0])
    num_class = int(sys.argv[2])
    modelname = sys.argv[3]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    n_epochs = 200
    iter_num = 5
    if 'pretrained' in modelname:
        n_epochs = 1
        iter_num = 1
    if num_class == 4:
        result = np.zeros((5,5))
    elif num_class == 2:
        result = np.zeros((5,7))
    for i in range(5):
        fold = str(i)
        x_train_ids, x_test_ids = x_folds[2*int(fold)+1], x_folds[2*int(fold)]
        if num_class == 4:
            if 'pretrained' in modelname or 'baseline' in modelname:
                accs, F1, F2, F3, F4 = GCN_main4class(datasetname, device, fold, x_train_ids, x_test_ids, fold, fold, iter_num, modelname, 0.2, 0.2, n_epochs=n_epochs)
                print("{}-fold{} [{}] acc:{:.4f} | F1:{:.4f} | F2:{:.4f} | F3:{:.4f} | F4:{:.4f} | [{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]".format(datasetname, fold, modelname, accs, F1, F2, F3, F4, accs, F1, F2, F3, F4))
            else:
                accs, F1, F2, F3, F4 = GCN_main4class(datasetname, device, fold, x_train_ids, x_test_ids, fold, fold, 5, modelname)
                print("{}-fold{} [cascade|test|{}{}] acc:{:.4f} | F1:{:.4f} | F2:{:.4f} | F3:{:.4f} | F4:{:.4f} | [{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]".format(datasetname, fold, fold,fold,accs, F1, F2, F3, F4, accs, F1, F2, F3, F4))
            result[i] = [accs, F1, F2, F3, F4]
        elif num_class == 2:
            if 'pretrained' in modelname or 'baseline' in modelname:
                accs, Prec1, Recll1, F1, Prec2, Recll2, F2= GCN_main2class(datasetname, device, fold, x_train_ids, x_test_ids, fold, fold, iter_num, modelname, 0.2, 0.2, n_epochs=n_epochs)
                print("{}-fold{} [{}] acc:{:.4f} | Pre1:{:.4f} | Pre2:{:.4f} | Rec1:{:.4f} | Rec2:{:.4f} | F1:{:.4f} | F2:{:.4f} | [{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]".format(datasetname, fold, modelname, accs,Prec1,Prec2,Recll1,Recll2,F1,F2, accs,Prec1,Prec2,Recll1,Recll2,F1,F2))
            else:
                accs, Prec1, Recll1, F1, Prec2, Recll2, F2= GCN_main2class(datasetname, device, fold, x_train_ids, x_test_ids, fold, fold, 5, modelname)
                print("{}-fold{} [cascade|test|{}{}] acc:{:.4f} | Pre1:{:.4f} | Pre2:{:.4f} | Rec1:{:.4f} | Rec2:{:.4f} | F1:{:.4f} | F2:{:.4f} | [{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}]".format(datasetname, fold, fold,fold,accs,Prec1,Prec2,Recll1,Recll2,F1,F2, accs,Prec1,Prec2,Recll1,Recll2,F1,F2))
            result[i] = [accs,Prec1,Prec2,Recll1,Recll2,F1,F2]
    print('The average result is', "\t".join([str(round(i,4)) for i in result.mean(0).tolist()]))
