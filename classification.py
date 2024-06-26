'''
Description: Used for training and testing the neural network for cell classification in CTG-PAM
'''
from model import CellMLP
from cell_tissue_graph import CellsClassifiDataset, LabelDataset
import numpy as np
import os
import torch
from utils import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
torch.manual_seed(42)

def train(n, bs, epoch, l, gpu, cell_para, feature_type):
    cn, tissue_n, cth, tth = cell_para
    train_path = "./train_dataset_demo/"
    train_data = CellsClassifiDataset(train_path, cn=cn, tn=tissue_n, cth=cth, tth=tth, featuretype=feature_type)
    train_data = LabelDataset(train_data, [1, 2])
    weights_dict = {1: 1, 2: 1}
    weights = [weights_dict[label] for data, label in train_data]
    sampler = WeightedRandomSampler(weights, num_samples=10000, 
    replacement=True)
    if feature_type == "ccg":
        input_d = 16
    else:
        input_d = 24
    net = CellMLP(n, input_dim=input_d).to(gpu)
    optimizer = torch.optim.Adam(net.parameters(), lr=l)  
    loss_func = nn.CrossEntropyLoss()   
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False, sampler=sampler)
    best_loss = 1e8
    for e_n in range(epoch):
        save_model = False
        sum_loss = 0
        k = 0
        for step, (b_x, b_y) in enumerate(train_loader):   
            b_x = b_x.to(gpu)
            b_y = b_y.to(gpu).to(torch.int64)
            b_y = b_y - 1 
            output = net(b_x)               
            loss = loss_func(output, b_y)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            sum_loss += loss.item()
            k += 1
        mean_loss = sum_loss/k
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_model = True
        if save_model: 
            os.makedirs(os.path.join("models", "n_%d_bs_%d_epoch_%d_cn_%d_tn_%d_cth_%d_tth_%d_%s"%(n, bs, epoch, cn, tissue_n, cth, tth, feature_type)), exist_ok=True)
            torch.save(net.state_dict(), os.path.join("models", "n_%d_bs_%d_epoch_%d_cn_%d_tn_%d_cth_%d_tth_%d_%s"%(n, bs, epoch, cn, tissue_n, cth, tth, feature_type), "epoch_%d.pth"%e_n))
def __train():
    import sys
    arg = sys.argv
    network_n = int(arg[2])
    batch_size = int(arg[3])
    epoch = int(arg[4])
    learning_rate = float(arg[5])
    gpu = "cuda:%d"%(int(arg[6]))
    cn = int(arg[7])
    tn = int(arg[8])
    cth = int(arg[9])
    tth = int(arg[10])
    para = (cn, tn, cth, tth)
    feature_type = arg[11] 
    train(n=network_n, bs=batch_size, epoch=epoch, l=learning_rate, gpu=gpu, cell_para=para, feature_type=feature_type)

def __test():
    import sys
    arg = sys.argv
    network_n = int(arg[2])
    gpu = "cuda:%d"%(int(arg[3]))
    cn = int(arg[4])
    tn = int(arg[5])
    cth = int(arg[6])
    tth = int(arg[7])
    para = (cn, tn, cth, tth)
    feature_type = arg[8]
    model_path = "./models/ctg_pam.pth"
    test(n=network_n, gpu=gpu, cell_para=para, feature_type=feature_type, model_path=model_path)

def test(n, gpu, cell_para, feature_type, model_path):
    cn, tissue_n, cth, tth = cell_para
    test_path = "./dataset"
    test_data = CellsClassifiDataset(test_path, cn=cn, tn=tissue_n, cth=cth, tth=tth, featuretype=feature_type)
    test_data = LabelDataset(test_data, [1, 2])
    if feature_type == "ccg":
        input_d = 16
    else:
        input_d = 24
    net = CellMLP(n, input_dim=input_d).to(gpu)
    net.load_state_dict(torch.load(model_path, map_location=gpu), strict=True)
    net.eval()
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    tp, fp, tn, fn = 0, 0, 0, 0
    for data, label in test_loader:
        data = data.to(gpu)
        output = net(data)
        pred_y = torch.max(output, 1)[1].cpu().data.numpy().squeeze()
        label = label.numpy()[0] - 1
        if label == 1:
            if pred_y == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred_y == 0:
                tn += 1
            else:
                fp += 1
    recall = tp / (tp + fn) if (tp+fn!=0) else 0.001
    specificity = tn / (tn + fp) if (tn + fp != 0) else 0.001
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    print("accuracy: %4f"%accuracy)
    print("specificity for lymphocyte / sensitivity for other: %4f"%recall)
    print("sensitivity for lymphocyte / specificity for other: %4f"%specificity)
    # We differentiate based on the dataset labels in this context.
if __name__ == "__main__":
    import sys
    arg = sys.argv
    task = arg[1]
    if task == "train":
        __train()
    elif task == "test":
        __test()
