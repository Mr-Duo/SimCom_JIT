import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    return auc(lr_recall, lr_precision)

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

def load_df_data(path_data, args):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data, args=args)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data)
    indexes = []
    cnt_noexits = 0

    for i in range(len(ids)):
        try:
            indexes.append(i)
        except FileNotFoundError:
            cnt_noexits += 1

    ids = [ids[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    features = features[indexes]
    return (ids, np.array(labels), features)

def load_data(args):
    train_path_data = (args.train_data)
    test_path_data = (args.test_data)
    train = load_df_data(train_path_data, args)
    test = load_df_data(test_path_data, args)
    return train, test

def balance_pos_neg_in_training(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def baseline_algorithm(train, test):
    _, y_train, X_train = train
    _, y_test, X_test = test

    ##over/under sample
    X_train,y_train = balance_pos_neg_in_training(X_train,y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=5).fit(X_train, y_train) 

    y_pred = model.predict_proba(X_test)[:, 1]

    return y_test, y_pred 