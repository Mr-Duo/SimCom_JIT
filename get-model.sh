#!/bin/bash

# Specify the directory you want to search
root_directory=$1

# Use the find command to list all directories under the root directory
# -type d: indicates we're looking for directories
# -maxdepth 1: limits the search to only one level (i.e., immediate subdirectories)
for folder in $(find "$root_directory" -maxdepth 1 -type d); do
    # Extract and print just the folder name
    folder_name=$(basename "$folder")
    echo "$folder_name"

    # Perform the desired operations in this directory
    cd $folder_name

    cp epoch_30.pt $folder_name.pt 

    rm -rf epoch*
    
    cd $root_directory

done
