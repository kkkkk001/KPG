# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import sparse as sp
import os
import pdb
cwd=os.getcwd()
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        # if index<=5000:
        if index<5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    # rootfeat = np.zeros([1, 18000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    ## 3. convert tree to matrix and edgematrix
    matrix=np.zeros([len(index2node),len(index2node)])
    # row=[]
    # col=[]
    ### self-loop for root
    row=[0]
    col=[0]
    x_word=[]
    x_index=[]
    edgematrix=[]
    # pdb.set_trace()
    
    ### text graph construction
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix.append(row)
    edgematrix.append(col)
    return x_word, x_index, edgematrix,rootfeat,rootindex

def getfeature(x_word,x_index):
    # x = np.zeros([len(x_index), 5000])
    # x = np.zeros([len(x_index), 18000])
    # for i in range(len(x_index)):
    #     if len(x_index[i])>0:
    #         x[i, np.array(x_index[i])] = np.array(x_word[i])
    # return x

    node_idx = []
    feat_idx = []
    values = []
    for i in range(len(x_index)):
        for j in range(len(x_index[i])):
            node_idx.append(i)
            feat_idx.append(x_index[i][j])
            values.append(x_word[i][j])
    return [node_idx, feat_idx, values]


def main():
    _dir = ""
    # treePath = os.path.join(cwd, 'data/Weibo/weibotree.txt')
    treePath = os.path.join(_dir, 'data/Weibo/data.TD_RvNN.vol_5000.txt')
    print("reading Weibo tree")
    treeDic = {}
    # wrongs = ['4468987704069525','4581284120627283','4615059169347883', '4602038618685535','4580189993974251','4579904375751342','4546980971813805','4546979504594837', '4503402090978990','4741250936734631']
    for line in open(treePath):
        line = line.rstrip()
        # if line.split('\t')[0] in wrongs:
        #     continue
        # print(line.split('\t')[0])
        eid, indexP, indexC,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[-1]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    print('tree no:', len(treeDic))


    # labelPath = os.path.join(cwd, "data/Weibo/weibo_id_label.txt")
    labelPath = os.path.join(_dir, "data/Weibo/Weibo_label_All.txt")
    print("loading weibo label:")
    event,y= [],[]
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid,label = line.split('\t')[2], line.split('\t')[0]
        # labelDic[eid] = int(label)
        # y.append(labelDic[eid])
        event.append(eid)
        # if labelDic[eid]==0:
        #     l1 += 1
        # if labelDic[eid]==1:
        #     l2 += 1
        labelset_f, labelset_t = ['false', 'rumours'], ['true', 'non-rumours']
        if label  in labelset_f:
            labelDic[eid]=1
            l1 += 1
        if label  in labelset_t:
            labelDic[eid]=2
            l2 += 1

    print(len(labelDic),len(event),len(y))
    print(l1, l2)

    def loadEid(event,id,y):
        if event is None:
            return None
        if len(event) < 2:
            if len(event) == 1:
                x_word, x_index = str2matrix(event[1]['vec'])
                rootfeat = np.zeros([1, 5000])
                # rootfeat = np.zeros([1, 18000])
                rootfeat[0, np.array(x_index)] = np.array(x_word)
                x_x = getfeature([x_word], [x_index])
                tree = [[0],[0]]
                rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x).astype(int), np.array(0), np.array(y)
                np.savez(os.path.join(_dir,'data/Weibograph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
                return None
                # pdb.set_trace()
            return None
        if len(event)>1:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
            x_x = getfeature(x_word, x_index)
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x).astype(int), np.array(
                rootindex), np.array(y)
            # pdb.set_trace()
            # np.savez(os.path.join(cwd,'data/Weibograph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            np.savez(os.path.join(_dir,'data/Weibograph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            return None
        x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
        x_x = getfeature(x_word, x_index)
        return rootfeat, tree, x_x, [rootindex]

    print("loading dataset", )
    # results = Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    for eid in tqdm(event):
        loadEid(treeDic[eid],eid,labelDic[eid])
    # loadEid(treeDic[event[3889]],event[3889],labelDic[event[3889]]) 
    # pdb.set_trace()
    print("finished!")
    return

if __name__ == '__main__':
    main()
