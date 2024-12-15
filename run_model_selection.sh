#!/bin/bash
CUDA_VISIBLE_DEVICES=0
DATASETS=( "bioasq9b" )
LM_NAMES="facebook/opt-1.3b Qwen/Qwen1.5-1.8B deepseek-ai/deepseek-coder-1.3b-base \
          EleutherAI/pythia-1.4b microsoft/phi-1_5 TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
          bigscience/bloom-1b7 tiiuae/falcon-rw-1b Muennighoff/SGPT-1.3B-mean-nli HuggingFaceTB/cosmo-1b \
          EleutherAI/gpt-neo-1.3B cognitivecomputations/TinyDolphin-2.8-1.1b \
          mistralai/Mistral-7B-v0.3 google/gemma-7b Qwen/Qwen1.5-7B THUDM/chatglm3-6b meta-llama/Meta-Llama-3-8B \
          stabilityai/stable-code-3b FreedomIntelligence/Apollo-2B google/gemma-2b IEITYuan/Yuan2-2B-hf \
          microsoft/phi-2 OEvortex/EMO-2B pansophic/rocket-3B stabilityai/stablelm-3b-4e1t"

CACHE_DIR=./models
SEEDS="2024"
ALL_CANDIDATE_SIZES="5"
METHODS="AiRTran"
SAVE_RESULTS="True"
OVERWRITE_RESULTS="False"

for dt_idx in "${!DATASETS[@]}"; do
  dataset=${DATASETS[$dt_idx]}
  python model_selection.py \
    --methods $METHODS \
    --all_candidate_sizes $ALL_CANDIDATE_SIZES \
    --dataset $dataset \
    --batch_size 12 \
    --model_name_or_paths ${LM_NAMES} \
    --cache_dir ${CACHE_DIR} \
    --seeds ${SEEDS} \
    --save_results ${SAVE_RESULTS} \
    --overwrite_results ${OVERWRITE_RESULTS}
done

