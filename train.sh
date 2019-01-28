#!/bin/bash
source activate Py367
CUDA_VISIBLE_DEVICES=-1 python train_ccp.py

