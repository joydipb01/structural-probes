#!/bin/bash

# Define the directory
directory="example/config/dist_train_configs/attnresln"

# Loop through each file in the directory
for i in {1..12}
do
  file="$directory"/"train-bert-base-dist_layer$i.yaml"
  if [[ -f "$file" ]]; then
    # Use sed to replace the string in place
    #sed -i "s/norm_bert/norm_bert_ln_n/g" "$file"
    sed -i "s/model_layer: $i/model_layer: $((i-1))/g" "$file"
    #sed -i "s/norm-bertbase-depth-probe_layer$i/norm-attnresln-bertbase-depth-probe_layer$i/g" "$file"
    #sed -i "s/example\/results/example\/bert-base-depth-params/g" "$file"
  fi
done

echo "Replacement complete."
