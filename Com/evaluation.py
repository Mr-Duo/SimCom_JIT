from model import DeepJIT
from sklearn.metrics import roc_auc_score, auc, roc_auc_score, precision_recall_curve, classification_report
import torch 
from matplotlib import pyplot
import numpy as np
import os, datetime
import pandas as pd
from tqdm import tqdm

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    return lr_auc


def evaluation_model(data, params):
    # Split data
    code_loader, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)    
    params.class_num = 1

    # Create model, optimizer, criterion
    model = CodeBERT_JIT(params).to(device=params.device)
    model.load_state_dict(torch.load(params.load_model))
    # model = torch.compile(model)

    model.eval()
    with torch.no_grad():
        all_predict, all_label = [], []
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)
            labels = batch["labels"].to(params.device)

            # Forward
            predict = model(added_code, removed_code)
            all_predict += predict.cpu().detach().numpy().tolist()
            all_label += labels.cpu().detach().numpy().tolist()

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)

    # convert probabilities to binary predictions
    # y_pred = [int(p >= 0.5) for p in all_predict]
    # target_names = ['Clean', 'Defect']
    # report = classification_report(all_label, y_pred, target_names=target_names, output_dict=True)
    # create DataFrame from report
    df = pd.DataFrame({'label': all_label, 'pred': all_predict})
    if os.path.isdir('./pred_scores/') is False:
        os.makedirs('./pred_scores/')
    df.to_csv('./pred_scores/test_com_' + params.project + '.csv', index=False, sep=',')
    print('Test data -- AUC score:', auc_score)