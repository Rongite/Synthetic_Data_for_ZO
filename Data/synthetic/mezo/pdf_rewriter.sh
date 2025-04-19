#!/bin/bash -l

# 2‑A. Local Gemma (24 GB GPU or more)
python pdf_rewriter.py \
       --pdf /home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/MeZO.pdf \
       --dataset /home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/Copa/copa_train.jsonl \
       --out /home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/synthetic/mezo/pdf_output.jsonl \
       --backend gemma \
       --int4 \
       --gpus 2

# # 2‑B. GPT‑4o via API
# export OPENAI_API_KEY="<your_key>"
# python pdf_rewriter.py \
#        --pdf handbook.pdf \
#        --dataset original.jsonl \
#        --backend gpt4o