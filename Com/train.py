import torch 
from tqdm import tqdm
import torch.nn as nn
import os, datetime
from utils import save
from model import DeepJIT

def train_model(data, params):
    # Split data
    code_loader, dict_msg, dict_code = data

    # Set up param
    params.save_dir = os.path.join(params.save_dir, params.project)
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]
    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    params.class_num = 1

    # Create model, optimizer, criterion
    model = DeepJIT(params).to(device=params.device)
    # model = torch.compile(model, backend="inductor")
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()
    
    # Training
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        for batch in tqdm(code_loader):
            # Extract data from DataLoader
            code = batch["code"].to(params.device)
            message = batch["message"].to(params.device)
            labels = batch["labels"].to(params.device)
            
            optimizer.zero_grad()

            # Forward
            predict = model(message, code)

            # Calculate loss
            loss = criterion(predict, labels)

            loss.backward()

            total_loss += loss

            optimizer.step()

        print('Training: Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss))

        save(model, params.save_dir, 'epoch', epoch)