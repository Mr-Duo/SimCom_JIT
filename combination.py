import csv, datetime, sys, os
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, f1_score
import numpy as np
import argparse

np.random.seed(10)

def write_to_file(file_path, content):
    with open(file_path, 'a+') as file:
        file.write(content + '\n')

def read_csv_1(fname):
    if not os.path.exists(fname):
        print("- File does not exist.")
        sys.exit(1)

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
    if not os.path.exists(fname):
        print("- File does not exist.")
        sys.exit(1)

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

## AUC-PC
# predict class values
def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    return lr_auc

parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='openstack')
parser.add_argument('-detail', type=str, default='openstack')
args = parser.parse_args()


project = args.project
detail = args.detail

data_dir1 = "/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Manh/JITDP/SimCom/pred_scores/"
data_dir2 = f"/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Manh/JITDP/ML/results/{project}/sim/pred_score/"

# Com
com_ = f'{data_dir1}test_com_{project}_{detail}.csv'

# Sim
sim_ = f'{data_dir2}test_sim_{project}_{detail}.csv'

# LAPredict
pred, _ = read_csv_2(sim_)

# DeepJIT 
pred_, label_ = read_csv_2(com_)

## Simple add
pred2 = [ pred_[i] + pred[i] for i in range(len(pred_))]
auc_ = roc_auc_score(y_true=np.array(label_).astype(float),  y_score=np.array(pred2))

# Call the function to write the content to the file
write_to_file("simcom_auc_if_avg.txt", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} - {project}_{detail} - {auc_}")
print(f"AUC-ROC:{auc_}")