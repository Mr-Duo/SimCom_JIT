# SimCom

## How to run

### Branches
```
git checkout [branch]
```
* original: original model
* codeBERT: using `output[0][:, 0]`
* codeBERT_unfreeze: using `output[0][:, 0]` and fine-tune codeBERT
* codeBERT_unfreeze2: using `output[0][:, 0]` and fine-tune codeBERT only for the first 2 epochs

### To train Com
```
cd Com

python main.py -train \
    -project [project name] \
    -train_data [absolute data path] \
    -dictionary_data [absolute data path]
```

### To evaluate Com
```
cd Com

python main.py -predict \
    -project [project name] \
    -predict_data [absolute data path] \
    -dictionary_data [absolute dictionary path] \
    -load_model [absolute model path]
```

### To train and evaluate Sim
```
cd Sim

python sim_model.py \
    -project [project name] \
    -train_data [absolute data path] \
    -test_data [absolute data path]
```

### To combinate Sim and Com
```
python combination.py -project [project name]
```