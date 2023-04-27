import os
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-project', type=str, default='openstack')
    return parser

params = read_args().parse_args()
project = params.project

train_sim = "python sim_model.py -project {} "
train_com = "python main.py -train -project {} -do_valid"
test_com = "python main.py -predict -project {}"
run_simcom = "python combination.py -project {}"

## Train & Predict for Sim
current_wrorking_dir = os.popen('pwd').read()
os.chdir(f"{current_wrorking_dir.strip()}/Sim")
cmd = train_sim
print("Start training of Sim")
result = os.popen(cmd.format(project)).readlines()
print('Training of Sim is finished')

## Train Com
os.chdir(f"{current_wrorking_dir.strip()}/Com")
cmd = train_com
print("Start training of Com")
result = os.popen(cmd.format(project)).readlines()
print('Training of Com is finished')  

## Predict by Com
cmd = test_com
result = os.popen(cmd.format(project)).readlines()

## Model fusion of Sim and Com
print('The final results for SimCom: \n')
os.chdir(current_wrorking_dir.strip())
cmd = run_simcom
result = os.popen(cmd.format(project)).readlines()
print(result)