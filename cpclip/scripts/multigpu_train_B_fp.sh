#!/bin/bash
cd /ssd2/letitiaz/cp_project/data

export PYTHONPATH=/ssd2/letitiaz/cp_project/code/cp-clip/src
export CUDA_VISIBLE_DEVICES=0,6,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
  --nproc-per-node=3 \
  --master-port=12345 \
  -- \
  /ssd2/letitiaz/cp_project/code/cp-clip/src/open_clip_train/main.py \
  --train-data /ssd2/letitiaz/cp_project/data/jsonFile/experimentJsonl_fp/final/train.jsonl\
  --val-data /ssd2/letitiaz/cp_project/data/jsonFile/experimentJsonl_fp/final/val.jsonl\
  --dataset-type jsonl \
  --model ViT-B-16 \
  --warmup 10000 \
  --batch-size 175 \
  --lr 1e-3 \
  --wd 0.1 \
  --epochs 30 \
  --workers 4 \
  --device cuda \
  --logs ./logs/ViT-B-16-experiment7/ \
  --log-every-n-steps 32 \
  --save-frequency 1 \
  --zeroshot-frequency 1 \
  --report-to tensorboard \
  --use-enhanced-clip \
  --special_tokens '<CONC_TOKEN>' '<TIME_TOKEN>' '<COMPOUND_TOKEN>' \
  --context-length 256 \
  --precision fp32 \
  --distributed \
  --compound-dim 1024 \
  --resume /ssd2/letitiaz/cp_project/data/logs/ViT-B-16-experiment7/2025_08_30-03_14_11-model_ViT-B-16-lr_0.001-b_175-j_4-p_fp32/checkpoints/epoch_7.pt

