#!/bin/bash


K=5
senic_pool=($(seq 0 35))
hyser_pool=(2 4 6 7 8 9 10 12 13 14 15 16 17 18 19 20)
tl_strategies=("fpt" "bt" "lp" "cadapter" "gdadapter")

for k in $(seq 1 $K); do
    mapfile -t source_subjects < <(shuf -e -n 9 --random-source=<(yes "$k") "${senic_pool[@]}")
    mapfile -t target_subjects < <(shuf -e -n 9 --random-source=<(yes "$k") "${hyser_pool[@]}")

    echo "Repetition $k: source=(${source_subjects[*]}) target=(${target_subjects[*]})"

    for tl in "${tl_strategies[@]}"; do
        for i in "${!source_subjects[@]}"; do
            python main.py --adapt_mode "inter-device" \
                           --source_parti "${source_subjects[$i]}" \
                           --target_parti "${target_subjects[$i]}" \
                           --dataset "Senic" \
                           --target_device "Hyser" \
                           --device "cuda:7" \
                           --backbone "HGRNet" \
                           --tl_strategy "$tl"
        done
    done
done
