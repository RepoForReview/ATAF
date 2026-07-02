#!/bin/bash

blocks=(2)

source_subjects=(2 4 6 7 8 9 10 12 13 14 15 16 17 18 19 20)
target_subjects=(4 6 7 8 9 10 12 13 14 15 16 17 18 19 20 2)


for block in "${blocks[@]}"
do

for i in "${!source_subjects[@]}"; do
    source_subject="${source_subjects[$i]}"
    target_subject="${target_subjects[$i]}"

    python main.py --adapt_mode "inter-subject" --source_parti  $source_subject \
                   --target_parti  $target_subject \
                   --dataset "Hyser" \
                   --device "cuda:0" \
                   --num_block $block \
                   --backbone "HGRNet" \
                   --tl_strategy "gdadapter"
done
done


