check_return_code() {
    if [ $1 -ne 0 ]; then
        echo "❌ $2 Failed with exit code $1"
        exit $1
    fi
}
export PYTHONPATH=$(pwd)/src

# machine_name=$(hostname)
# texts_n_seqs=256
# seed=1
# if [ -z "$HF_TOKEN" ]; then
#     export HF_TOKEN="Your HF Token"
#     echo "HF_TOKEN set to ${HF_TOKEN}"
# fi

# # # for loop over multiple models
# model_names=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "google/gemma-2-2b-it" "google/gemma-2-9b-it")
# timestamp=$(date +%Y%m%d-%H%M%S)
# for model_name in "${model_names[@]}"; do
#     model_name_esc=$(echo $model_name | sed 's/\//_/g')
#     output_dir="./checkpoints/${machine_name}/${model_name_esc}_${timestamp}"
#     conda run -n bblock --no-capture-output python llm_logits.py collect \
#         "['$model_name']" \
#         --q-tags "['bf16', 'fp16', 'bf16-sdpa', 'fp16-sdpa', 'bf16-flashattn2', 'fp16-flashattn2']" \
#         --texts-n-seqs $texts_n_seqs \
#         --seed $seed \
#         --output-dir $output_dir \
#         --hf-push-to-hub True --hf-repo-id "Cheng98/blackbox-locking"
#     check_return_code $? "llm-logits.py collect $model_name"
# done

# ['NVIDIA-L40S_fp16', 'NVIDIA-RTX-A6000_fp16', 'NVIDIA-A100-SXM4-80GB_fp16']"

# whitebox, A100 vs A6000, fp16
# conda run -n bblock --no-capture-output python llm_logits.py train-svm \
#     "['data/Cheng98-blackbox-locking/snapshots/5a9fc04fa92ad2c95b0b5a0d5a128f59ff2a1390/cx3-20-1.cx3.hpc.ic.ac.uk_NVIDIA-L40S/seed-1/Qwen-Qwen2.5-3B-Instruct/bf16_fp16_bf16-sdpa_fp16-sdpa_bf16-flashattn2_fp16-flashattn2/20241015-165819/logits_dataset', 'data/Cheng98-blackbox-locking/snapshots/5a9fc04fa92ad2c95b0b5a0d5a128f59ff2a1390/gpu-8_NVIDIA-RTX-A6000/seed-1/Qwen-Qwen2.5-3B-Instruct/bf16_fp16_bf16-sdpa_fp16-sdpa_bf16-flashattn2_fp16-flashattn2/20241015-144927/logits_dataset', 'data/Cheng98-blackbox-locking/snapshots/5a9fc04fa92ad2c95b0b5a0d5a128f59ff2a1390/hx1-d12-gpu-02_NVIDIA-A100-SXM4-80GB/seed-1/Qwen-Qwen2.5-3B-Instruct/bf16_fp16_bf16-sdpa_fp16-sdpa_bf16-flashattn2_fp16-flashattn2/20241015-162139/logits_dataset']" \
#     --truncate-first-n-logits 30000 \
#     --q-tags-to-keep "['NVIDIA-A100-SXM4-80GB_fp16', 'NVIDIA-RTX-A6000_fp16']" \
#     --output-dir ./checkpoints-text/whitebox_A100-vs-A6000_fp16 --svm-kernel linear --svm-n-iters 1000

conda run -n bblock --no-capture-output python llm_logits.py train-svm \
    "['data/Cheng98-blackbox-locking/snapshots/5a9fc04fa92ad2c95b0b5a0d5a128f59ff2a1390/hx1-d12-gpu-02_NVIDIA-A100-SXM4-80GB/seed-1/Qwen-Qwen2.5-3B-Instruct/bf16_fp16_bf16-sdpa_fp16-sdpa_bf16-flashattn2_fp16-flashattn2/20241015-162139/logits_dataset']" \
    --truncate-first-n-logits 30000 \
    --q-tags-to-keep "['NVIDIA-A100-SXM4-80GB_fp16-sdpa', 'NVIDIA-A100-SXM4-80GB_fp16-flashattn2']" \
    --svm-kernel linear --svm-n-iters 1000 --output-dir ./tmp/svm # --output-dir ./checkpoints-text/whitebox_A100_fp16_flashattn2-vs-sdpa
