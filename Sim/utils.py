import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    return auc(lr_recall, lr_precision)

def train_and_evl(data, label, args):
    size = int(label.shape[0]*0.2)
    auc_ = []

    for i in range(5):
        idx = size * i
        X_e = data[idx:idx+size]
        y_e = label[idx:idx+size]

        X_t = np.vstack((data[:idx], data[idx+size:]))
        y_t = np.hstack((label[:idx], label[idx+size:]))

        model = LogisticRegression(max_iter=7000).fit(X_t, y_t)
        y_pred = model.predict_proba(X_e)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_e, y_score=y_pred, pos_label=1)
        auc_.append(auc(fpr, tpr))

    return np.mean(auc_)

def replace_value_dataframe(df, args):
    df = df.replace({True: 1, False: 0})
    df = df.fillna(df.mean(numeric_only=True))
    if args.drop:
        df = df.drop(columns=[args.drop])
    elif args.only:
        df = df[['Unnamed: 0','_id','date','bug','__'] + args.only]
    return df.values

def get_features(data):
    return data[:, 5:]

def get_ids(data):
    return data[:, 1:2].flatten().tolist()

def get_label(data):
    data = data[:, 3:4].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data

def load_df_data(path_data, long_commits, args, flag=None):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data, args=args)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes = []
    cnt_noexits = 0

    for i in range(len(ids)):
        ## filter long commits
        '''       
        if flag == 'test':
            if ids[i] in long_commit_ids:
                continue
        '''

        try:
            indexes.append(i)
        except FileNotFoundError:
            cnt_noexits += 1

    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)

def load_data(args):
    train_path_data = (f'../data/lapredict-paper/{args.project}/cross/k_train.csv')
    test_path_data = (f'../data/lapredict-paper/{args.project}/cross/k_test.csv')
    train = load_df_data(train_path_data, args.long_train_commits, args, 'test')
    test = load_df_data(test_path_data, args.long_test_commits, args, 'test')
    return train, test

def evaluation_metrics(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, pos_label=1)
    auc_ = auc(fpr, tpr)
    
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = 2 * prc * rc / (prc + rc)
    
    return acc, prc, rc, f1, auc_

def balance_pos_neg_in_training(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def baseline_algorithm(train, test, args, only=False):
    _, y_train, X_train = train
    _, y_test, X_test = test

    ##over/under sample
    X_train,y_train = balance_pos_neg_in_training(X_train,y_train)
    acc, prc, rc, f1, auc_ = 0, 0, 0, 0, 0
    if args.algorithm == 'lr':
        starttime = time.time()

        param = {'max_depth':5, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}

        model = RandomForestClassifier(n_estimators=100, random_state=5).fit(X_train, y_train) 

        endtime = time.time()
        dtime = endtime - starttime

        starttime = time.time()

        y_pred = model.predict_proba(X_test)[:, 1]

        endtime = time.time()
        dtime = endtime - starttime
        
        acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
        if only and "cross" not in args.data:
            auc_ = train_and_evl(X_train, y_train, args)  
    else:
        print('You need to give the correct algorithm name')
        return

    return y_test, y_pred 