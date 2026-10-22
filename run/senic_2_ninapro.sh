#!/bin/bash

K=5
senic_pool=($(seq 0 35))
ninapro_pool=(1 2 3 4 5 6 9 10 11)

tl_strategies=("fpt" "bt" "lp" "cadapter" "gdadapter")

for k in $(seq 1 $K); do
    mapfile -t source_subjects < <(shuf -e -n 9 --random-source=<(yes "$k") "${senic_pool[@]}")
    mapfile -t target_subjects < <(shuf -e -n 9 --random-source=<(yes "$k") "${ninapro_pool[@]}")

    echo "Repetition $k: source=(${source_subjects[*]}) target=(${target_subjects[*]})"

    for tl in "${tl_strategies[@]}"; do
        for i in "${!source_subjects[@]}"; do
            python main.py --adapt_mode "inter-device" \
                           --source_parti "${source_subjects[$i]}" \
                           --target_parti "${target_subjects[$i]}" \
                           --dataset "Senic" \
                           --target_device "Ninapro" \
                           --device "cuda:5" \
                           --backbone "HGRNet" \
                           --tl_strategy "$tl"
        done
    done
done

