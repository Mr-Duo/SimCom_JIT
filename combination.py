import csv
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, f1_score
import numpy as np
from matplotlib import pyplot
import argparse
from sklearn.metrics import classification_report
import pandas as pd

np.random.seed(10)

def read_csv_1(fname):
    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(int(line[0]))
            la.append(int(line[1]))
            pred.append(float(line[2]))
            
    return pred, label, la

def read_csv_2(fname):
    label = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(line[0])
            pred.append(float(line[1]))

    return pred, label
    
## AUC-PC
# predict class values
def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.savefig(f"{project}")

    return lr_auc

parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='openstack')
args = parser.parse_args()

data_dir1 = "./Com/pred_scores/"
data_dir2 = "./Sim/pred_scores/"

project = args.project

# Com
com_ = f'{data_dir1}test_com_{project}.csv'

# Sim
sim_ = f'{data_dir2}test_sim_{project}.csv'

# LAPredict
pred, label = read_csv_2(sim_)

# DeepJIT 
pred_, label_ = read_csv_2(com_)

## Simple add
pred2 = [ pred_[i] + pred[i] for i in range(len(pred_))]
auc2 = roc_auc_score(y_true=np.array(label_).astype(float),  y_score=np.array(pred2))

# convert probabilities to binary predictions
y_pred = [int(p >= 0.5) for p in np.array(pred2)]
target_names = ['Clean', 'Defect']
y_true = np.array(label_).astype(float)
report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'SimCom_{project}.csv')

mean_pred = float(sum(pred2)/len(pred2))
pc_ = auc_pc(label_, pred2)

t = 1
real_label = [float(l) for l in label_]
real_pred = [1 if p > t else 0 for p in pred2]
f1_ = f1_score(y_true=real_label,  y_pred=real_pred)
print(f"AUC-ROC:{auc2}  AUC-PR:{pc_}  F1-Score:{f1_}")