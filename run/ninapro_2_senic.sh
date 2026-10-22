#!/bin/bash

K=5
ninapro_pool=(1 2 3 4 5 6 9 10 11)
senic_pool=($(seq 0 35))
tl_strategies=("fpt" "bt" "lp" "cadapter" "gdadapter")

for k in $(seq 1 $K); do
    mapfile -t source_subjects < <(shuf -e --random-source=<(yes "$k") "${ninapro_pool[@]}")
    mapfile -t target_subjects < <(shuf -e -n 9 --random-source=<(yes "$k") "${senic_pool[@]}")

    echo "Repetition $k: source=(${source_subjects[*]}) target=(${target_subjects[*]})"

    for tl in "${tl_strategies[@]}"; do
        for i in "${!source_subjects[@]}"; do
            python main.py --adapt_mode "inter-device" \
                           --source_parti "${source_subjects[$i]}" \
                           --target_parti "${target_subjects[$i]}" \
                           --dataset "Ninapro" \
                           --target_device "Senic" \
                           --device "cuda:7" \
                           --backbone "HGRNet" \
                           --tl_strategy "$tl"
        done
    done
done