export  CUDA_VISIBLE_DEVICES=0
CACHE_DIR=./models
POOLER=mean
BATCH_SIZE=12

PLMS="facePook/opt-1.3b Qwen/Qwen1.5-1.8B deepseek-ai/deepseek-coder-1.3b-base \
        EleutherAI/pythia-1.4b microsoft/phi-1_5 TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
        bigscience/bloom-1b7 tiiuae/falcon-rw-1b Muennighoff/SGPT-1.3B-mean-nli HuggingFaceTB/cosmo-1b \
        EleutherAI/gpt-neo-1.3B cognitivecomputations/TinyDolphin-2.8-1.1b \
        mistralai/Mistral-7B-v0.3 google/gemma-7b Qwen/Qwen1.5-7B THUDM/chatglm3-6b meta-llama/Meta-Llama-3-8B \
        stabilityai/stable-code-3b FreedomIntelligence/Apollo-2B google/gemma-2b IEITYuan/Yuan2-2B-hf \
        microsoft/phi-2 OEvortex/EMO-2B pansophic/rocket-3B stabilityai/stablelm-3b-4e1t"

for DATASET in bioasq9b
    do
    for PLM in ${PLMS}
    do
        python3 dataset_encoding.py \
            --dataset ${DATASET} \
            --batch_size ${BATCH_SIZE} \
            --model_name_or_path ${PLM} \
            --pooler ${POOLER} \
            --cache_dir ${CACHE_DIR}
    done
done