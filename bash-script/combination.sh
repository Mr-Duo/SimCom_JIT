#!/bin/bash

data_path=/media/aiotlab3/27934be5-a11a-44ba-8b28-750d135bc3b3/RISE/Manh/JITDP/ML/results
parts=("part_1_part_4" "part_3_part_4" "part_4")

for folder in $(find "$data_path" -maxdepth 1 -type d); do
    folder_name=$(basename "$folder")
    echo "folder: $folder_name"

    for part in "${parts[@]}"; do

        python combination.py -project $folder_name -detail "$part"
        
    done
done