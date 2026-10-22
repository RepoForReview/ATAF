#!/bin/bash

subjects=(2 4 6 7 8 9 10 12 13 14 15 16 17 18 19 20)

for subject in "${subjects[@]}"
do
  python main.py --adapt_mode "intra-subject" \
                 --source_parti "$subject" \
                 --dataset "Hyser" \
                 --device "cuda:4" \
                 --backbone "HGRNet"
#done
done
