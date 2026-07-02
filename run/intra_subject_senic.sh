#!/bin/bash

subjects=($(seq 0 35))


for subject in "${subjects[@]}"
do
  python main.py --adapt_mode "intra-subject" \
                 --source_parti "$subject" \
                 --dataset "Senic" \
                 --device "cuda:2" \
                 --backbone "HGRNet"
#done
done

