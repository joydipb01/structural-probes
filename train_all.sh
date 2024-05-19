directory="example/config/dist_train_configs/attnresln"

for file in "$directory"/*
do
    if [[ -f "$file" ]]; then
        echo "=======NEW LAYER========"
        python3 structural-probes/train_probe.py "$file"
    fi
done