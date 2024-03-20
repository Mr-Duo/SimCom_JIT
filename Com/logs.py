import pandas as pd
import psutil, subprocess
import os

def logs(file_path, project_name, value, model):
    # Read existing CSV file
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], model] = value
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        if model == "DeepJIT":
            new_row = {"Project Name": project_name, "CC2Vec": 0.0, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": value, "SimCom": 0.0, "DeepJIT Extend": 0.0}  # Update with actual column names
        else:
            new_row = {"Project Name": project_name, "CC2Vec": 0.0, "LA": 0.0, "LR": 0.0, "TLEL": 0.0, "Sim": 0.0, "DeepJIT": 0.0, "SimCom": value, "DeepJIT Extend": 0.0}  # Update with actual column names
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)
    
def get_ram_usage():
    # Get system memory usage in bytes
    memory_usage = psutil.virtual_memory().used
    memory_usage_mb = memory_usage / (1024 * 1024)
    return memory_usage_mb

def get_vram_usage():
    # Get GPU memory usage using nvidia-smi command
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits'])
        vram_usage = int(output.strip().split(b'\n')[1])
        return vram_usage
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # Handle case where nvidia-smi is not available or GPU is not Nvidia
