import torch 
from tqdm import tqdm
import torch.nn as nn
import os, datetime
from utils import save
from model import CodeBERT_JIT

def train_model(data, params):
    # Split data
    code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.vocab_code = len(dict_code)
    params.vocab_msg = len(dict_msg)
    params.class_num = 1
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    # Create model, optimizer, criterion
    model = CodeBERT_JIT(params).to(device=params.device)
    # model = torch.compile(model, backend="inductor")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()
    
    # Training
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            added_code = batch["added_code"].to(params.device)
            removed_code = batch["removed_code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            
            optimizer.zero_grad()

            # Forward
            predict = model(added_code, removed_code, message)

            # Calculate loss
            loss = criterion(predict, labels)

            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))

        save(model, params.save_dir, 'epoch', epoch)