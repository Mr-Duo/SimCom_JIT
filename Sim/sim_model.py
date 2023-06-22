import argparse
from sklearn.metrics import roc_auc_score, f1_score, roc_auc_score
import pandas as pd
from utils import load_data, baseline_algorithm, auc_pc
import os

# Input arguement
parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='openstack')
parser.add_argument('-data', type=str, default='k')
parser.add_argument('-train_data', type=str)
parser.add_argument('-test_data', type=str)
parser.add_argument('-only', type=bool, default=False)
parser.add_argument('-drop', type=str, default='')
parser.add_argument('-jitbot', type=bool, default=False)

args = parser.parse_args()

train, test = load_data(args)
labels, predicts = baseline_algorithm(train=train, test=test)
auc_pc_score = auc_pc(labels, predicts)
auc_roc = roc_auc_score(y_true=labels, y_score=predicts)

df = pd.DataFrame({'label': labels, 'pred': predicts})
if os.path.isdir('./pred_scores/') is False:
    os.makedirs('./pred_scores/')
df.to_csv('./pred_scores/test_sim_' + args.project + '.csv', index=False, sep=',')

y_true = labels
threshs = [0.5]
for thresh_hold in threshs:
    real_pred = [1 if predict > thresh_hold else 0 for predict in predicts]
    f1 = f1_score(y_true=y_true,  y_pred=real_pred)
    print(
        f"Threshold: {thresh_hold}  AUC-ROC:{auc_roc}  AUC-PR:{auc_pc_score}  F1-Score:{f1}"
    )