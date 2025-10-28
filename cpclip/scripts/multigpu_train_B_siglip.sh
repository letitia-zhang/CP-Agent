#!/bin/bash
cd /ssd2/letitiaz/cp_project/data

export PYTHONPATH=/ssd2/xxx/src
export CUDA_VISIBLE_DEVICES=0,1,2,7
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
  --nproc-per-node=4 \
  --master-port=12345 \
  -- \
  /ssd2/xxx/main.py \
  --train-data /ssd2/xxx/train_0903.jsonl\
  --dataset-type jsonl \
  --model ViT-B-16-SigLIP-512 \
  --warmup 5000 \
  --batch-size 170 \
  --lr 1e-3 \
  --wd 0.1 \
  --epochs 35 \
  --workers 4 \
  --device cuda \
  --logs ./logs/ViT-B-16-experiment11/ \
  --log-every-n-steps 32 \
  --save-frequency 1 \
  --zeroshot-frequency 1 \
  --report-to tensorboard \
  --use-enhanced-clip \
  --special_tokens '<CONC_TOKEN>' '<TIME_TOKEN>' '<COMPOUND_TOKEN>' \
  --context-length 256 \
  --precision fp32 \
  --distributed \
  --siglip 
