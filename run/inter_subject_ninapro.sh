#!/bin/bash
blocks=(1 2 3 4 5)

source_subjects=(1 2 3 4 5 6 9 10 11)
target_subjects=(2 3 4 5 6 9 10 11 1)
for block in "${blocks[@]}"
do

for i in "${!source_subjects[@]}"; do
    source_subject="${source_subjects[$i]}"
    target_subject="${target_subjects[$i]}"

    python main.py --adapt_mode "inter-subject" --source_parti  $source_subject \
                   --target_parti  $target_subject \
                   --dataset "Ninapro" \
                   --device "cuda:2" \
                   --num_block $block \
                   --backbone "HGRNet" \
                   --tl_strategy "gdadapter"
done
done



