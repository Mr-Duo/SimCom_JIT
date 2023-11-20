import torch 
from tqdm import tqdm
import torch.nn as nn
import os, datetime
from utils import save_best
from model import CodeBERT_JIT
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    return lr_auc

def train_model(data, params):
    # Split data
    if params.do_valid is True:
        code_loader, val_code_loader, dict_msg, dict_code = data
    else:
        code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    params.vocab_msg = len(dict_msg)
    params.class_num = 1
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    # Create model, optimizer, criterion
    model = CodeBERT_JIT(params).to(device=params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()
    
    # Validate
    best_valid_score = 0
    smallest_loss = 1000000
    early_stop_count = 5

    # Training
    for epoch in range(1, params.num_epochs + 1):
        if epoch <= 2:
            for param in model.codeBERT.parameters():
                param.requires_grad = True
        else:
            for param in model.codeBERT.parameters():
                param.requires_grad = False
                
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            
            optimizer.zero_grad()

            # Forward
            predict = model(added_code, removed_code, message)

            # Calculate loss
            loss = criterion(predict, labels)

            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))

        if params.do_valid == True:
            model.eval()
            with torch.no_grad():
                all_predict, all_label = [], []
                for batch in tqdm(val_code_loader):
                    # Extract data from DataLoader
                    added_code = batch["added_code"].to(params.device)
                    removed_code = batch["removed_code"].to(params.device)
                    message = batch["message"].to(params.device)
                    labels = batch["labels"].to(params.device)

                    # Forward
                    predict = model(added_code, removed_code, message)
                    all_predict += predict.cpu().detach().numpy().tolist()
                    all_label += labels.cpu().detach().numpy().tolist()

            auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
            auc_pc_score = auc_pc(all_label, all_predict)
            print('Valid data -- AUC-ROC score:', auc_score,  ' -- AUC-PC score:', auc_pc_score)

            valid_score = auc_pc_score
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                print('Save a better model', best_valid_score)
                save_best(model, params.save_dir, file_name='best_model')
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break
        else:
            loss_score = total_loss
            if loss_score < smallest_loss:
                smallest_loss = loss_score
                print('Save a better model', smallest_loss)
                save_best(model, params.save_dir, file_name='deepjit_best_model')
            else:
                print('No update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break