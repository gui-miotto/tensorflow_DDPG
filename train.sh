#!/bin/bash
source activate Py367
CUDA_VISIBLE_DEVICES=-1 python train_gen.py --hier --eps 1000

