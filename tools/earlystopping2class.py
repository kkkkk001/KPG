import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.accs=0
        self.F1=0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, accs,acc1,acc2,pre1,pre2,rec1,rec2,F1,F2,model,modelname,str,epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.accs = accs
            self.acc1=acc1
            self.acc2=acc2
            self.pre1=pre1
            self.pre2=pre2
            self.rec1=rec1
            self.rec2=rec2
            self.F1 = F1
            self.F2 = F2
            self.save_checkpoint(val_loss, model,modelname,str)
            print("Best model updated in epoch {}! Loss:{:.4f}| Accuracy: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                      "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}"
                      .format(epoch,-self.best_score,self.accs,self.pre1,self.pre2,self.rec1,self.rec2,self.F1,self.F2))
        # elif score < self.best_score and accs < self.accs:
        #     self.counter += 1
        #     print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        #         print("BEST LOSS:{:.4f}| Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
        #               "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}"
        #               .format(-self.best_score,self.accs,self.acc1,self.acc2,self.pre1,self.pre2,self.rec1,self.rec2,self.F1,self.F2))
        # elif accs >= self.accs and (F1 + F2 >=  self.F1 + self.F2):
        # elif self.best_score <= score and accs+F1+F2>=self.accs+self.F1+self.F2:
        # elif self.best_score <= score:
        # elif score > self.best_score:
        elif accs > self.accs or (accs == self.accs and score > self.best_score):
            self.best_score = score
            self.accs = accs
            self.acc1=acc1
            self.acc2=acc2
            self.pre1=pre1
            self.pre2=pre2
            self.rec1=rec1
            self.rec2=rec2
            self.F1 = F1
            self.F2 = F2
            print("Best model updated in epoch {}! Loss:{:.4f}| Accuracy: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                      "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}"
                      .format(epoch,-self.best_score,self.accs,self.pre1,self.pre2,self.rec1,self.rec2,self.F1,self.F2))
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0
            self.early_stop = False
        else:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                # print("BEST LOSS:{:.4f}| Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                #       "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}"
                #       .format(-self.best_score,self.accs,self.acc1,self.acc2,self.pre1,self.pre2,self.rec1,self.rec2,self.F1,self.F2))

    def save_checkpoint(self, val_loss, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(),"cstudy/"+modelname+str+'.m')
        # torch.save(model.state_dict(),f"clfs/{modelname}/"+modelname+str+'.m')
        self.val_loss_min = val_loss