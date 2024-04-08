import numpy as np
import torch

class Earlystopping:
    
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.acc = 0
        # self.bleu1 = 0
        # self.bleu2 = 0
        self.val_loss_min = np.Inf

    # def __call__(self, val_loss, acc, bleu1, bleu2, model, modelname, str):
    def __call__(self, val_loss, model, modelname, str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.acc = acc
            # self.bleu1 = bleu1
            # self.bleu2 = bleu2
            self.save_checkpoint(val_loss, model,modelname,str)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                # print("BEST LOSS:{:.4f}| Accuracy: {:.4f}|BLEU1: {:.4f}|BLEU2: {:.4f}"
                #       .format(-self.best_score,self.acc,self.bleu1,self.bleu2))
                print("BEST LOSS:{:.4f}".format(-self.best_score))
        else:
            self.best_score = score
            # self.acc = acc
            # self.bleu1 = bleu1
            # self.bleu2 = bleu2
            print("Best model updated!")
            self.save_checkpoint(val_loss, model,modelname,str)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,modelname,str):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(),"rgmodels/"+modelname+str+'.m')
        self.val_loss_min = val_loss