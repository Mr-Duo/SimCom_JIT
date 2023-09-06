import numpy as np
import math
import os, torch
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    #save_prefix = os.path.join(save_dir, save_prefix)
    #save_path = '{}_{}.pt'.format(save_prefix, epochs)
    
    
    save_file = 'best_model.pt'
    save_path = os.path.join(save_dir, save_file)
    torch.save(model.state_dict(), save_path)

def mini_batches_test(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(num_complete_minibatches):        
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:        
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_test_with_ids(ids, X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    shuffled_ids, shuffled_X_msg, shuffled_X_code, shuffled_Y = ids, X_msg, X_code, Y
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(num_complete_minibatches):
        mini_batch_ids = shuffled_ids[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_ids, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_ids = shuffled_ids[num_complete_minibatches * mini_batch_size: m]
        mini_batch_X_msg = shuffled_X_msg[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X_code = shuffled_X_code[num_complete_minibatches * mini_batch_size: m, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_ids, mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_train(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg, X_code, Y
    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0]    

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for _ in range(num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X_msg, mini_batch_X_code = shuffled_X_msg[indexes], shuffled_X_code[indexes]
        mini_batch_Y = shuffled_Y[indexes]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def mini_batches_train_non_balance(X_msg, X_code, Y, mini_batch_size=64, seed=0):
    m = X_msg.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    ## Shuffle Training Data
    indexs = list(range(m))
    random.shuffle(indexs)
    shuffled_X_msg, shuffled_X_code, shuffled_Y = X_msg[indexs], X_code[indexs], Y[indexs]
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))

    for k in range(num_complete_minibatches):
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_code = shuffled_X_code[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_msg, mini_batch_X_code, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def get_world_dict(world2id):
    return {world2id[world]: world for world in world2id}

def mapping_dict_world(senten_ids, id2world):
    return [id2world[_id] for _id in senten_ids]

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = f'{save_prefix}_{epochs}.pt'
    torch.save(model.state_dict(), save_path)

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

def get_features(data, jitbot=False):
    if jitbot:
        return data[:, 3:11]
    return data[:, 3:17]

def get_ids(data):
    return data[:, 1:2].flatten().tolist()

def get_label(data):
    data = data[:, 17:].flatten().tolist()
    data = [1 if int(d) > 0 else 0 for d in data]
    return data

def load_df_data(path_data, args):
    data = pd.read_csv(path_data)
    data = replace_value_dataframe(df=data, args=args)
    ids, labels, features = get_ids(data=data), get_label(data=data), get_features(data=data, jitbot=args.jitbot)
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

    return features

def load_data(args):
    path_data = (args.features)
    features = load_df_data(path_data, args)
    return features

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