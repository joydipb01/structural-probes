# directory1="example/bert-base-depth-params"
# directory2="example/bert-base-distance-params"
demofile="example/demo-bert.yaml"
results_dir="example/results/attnresln"

for i in {1..12}
do
    if [[ -f "$demofile" ]]; then
        echo "=======NEW LAYER========"
        sed -i "s/model_layer: $((i-2))/model_layer: $((i-1))/" "$demofile"
        sed -i "s/norm-attnresln-bertbase-depth-probe_layer$((i-1))/norm-attnresln-bertbase-depth-probe_layer$i/" "$demofile"
        sed -i "s/norm-attnresln-bertbase-distance-probe_layer$((i-1))/norm-attnresln-bertbase-distance-probe_layer$i/" "$demofile"
        cat my_ptb_sentences.txt | python3 structural-probes/run_demo.py "$demofile" > "distance_depth_values/bert-base-attnresln_layer$i.txt"
        mv "$results_dir"/BERT* "$results_dir/bert-base-attnresln_layer$i/"
        echo "=====DEMO DONE====="
    fi
done