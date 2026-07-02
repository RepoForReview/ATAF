#!/bin/bash

source_subjects=(2 4 6 7 8 9 10 12 13)
target_subjects=(1 2 3 4 5 6 9 10 11)

for i in "${!source_subjects[@]}"; do
    source_subject="${source_subjects[$i]}"
    target_subject="${target_subjects[$i]}"

    python main.py --adapt_mode "inter-device" \
                   --source_parti "$source_subject" \
                   --target_parti "$target_subject" \
                   --dataset "Hyser" \
                   --target_device "Ninapro" \
                   --device "cuda:2" \
                   --backbone "HGRNet" \
                   --tl_strategy "gdadapter"
done



