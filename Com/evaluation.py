from model import DeepJIT
from sklearn.metrics import roc_auc_score, auc, roc_auc_score, precision_recall_curve, classification_report
import torch
import os, datetime
import pandas as pd
from tqdm import tqdm

def evaluation_model(data, params):
    # Split data
    code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.class_num = 1

    # Create model, optimizer, criterion
    model = DeepJIT(params).to(device=params.device)
    model.load_state_dict(torch.load(params.load_model))
    # model = torch.compile(model)

    model.eval()
    with torch.no_grad():
        all_predict, all_label = [], []
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)

            # Forward
            predict = model(message, code)
            all_predict += predict.cpu().detach().numpy().tolist()
            all_label += labels.cpu().detach().numpy().tolist()

    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)

    df = pd.DataFrame({'label': all_label, 'pred': all_predict})
    if os.path.isdir('./pred_scores/') is False:
        os.makedirs('./pred_scores/')
    df.to_csv('./pred_scores/test_com_' + params.project + '.csv', index=False, sep=',')
    print('Test data -- AUC score:', auc_score)