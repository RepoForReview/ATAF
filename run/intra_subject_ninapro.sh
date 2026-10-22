#!/bin/bash

subjects=(1 2 3 4 5 6 9 10 11)


for subject in "${subjects[@]}"
do
  python main.py --adapt_mode "intra-subject" \
                 --source_parti "$subject" \
                 --dataset "Ninapro" \
                 --device "cuda:3" \
                 --backbone "HGRNet"
#done
done
