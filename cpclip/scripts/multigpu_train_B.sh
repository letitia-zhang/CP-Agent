#!/bin/bash
cd /ssd2/letitiaz/cp_project/data

export PYTHONPATH=/ssd2/letitiaz/cp_project/code/open-clip/src
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
  --nproc-per-node=7 \
  --master-port=12345 \
  -- \
  /ssd2/letitiaz/cp_project/code/open-clip/src/open_clip_train/main.py \
  --train-data /ssd2/letitiaz/cp_project/data/jsonFile/experimentJsonl_moa/final/train_0903.jsonl\
  --dataset-type jsonl \
  --model ViT-L-16 \
  --warmup 5000 \
  --batch-size 68 \
  --lr 1e-3 \
  --wd 0.1 \
  --epochs 30 \
  --workers 4 \
  --device cuda \
  --logs ./logs/ViT-L-16-experiment10/ \
  --log-every-n-steps 32 \
  --save-frequency 1 \
  --zeroshot-frequency 1 \
  --report-to tensorboard \
  --use-enhanced-clip \
  --special_tokens '<CONC_TOKEN>' '<TIME_TOKEN>' '<COMPOUND_TOKEN>' \
  --context-length 256 \
  --precision fp32 \
  --distributed 

