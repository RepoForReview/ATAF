#!/bin/bash

source_subjects=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
target_subjects=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 0)

for i in "${!source_subjects[@]}"; do
    source_subject="${source_subjects[$i]}"
    target_subject="${target_subjects[$i]}"

    python main.py --adapt_mode "inter-subject" --source_parti  $source_subject \
                   --target_parti  $target_subject \
                   --dataset "Senic" \
                   --device "cuda:2" \
                   --lam 1.0 \
                   --backbone "HGRNet" \
                   --tl_strategy "gdadapter"

done



