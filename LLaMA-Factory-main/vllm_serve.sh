# vllm serve output/qwen3_lora_sft --max-model-len 8192 --gpu-memory-utilization 0.75
#vllm serve /root/autodl-tmp/RAG/LLaMA-Factory-main/output/qwen3_lora_sft_0.6B --max-model-len 8192 --gpu-memory-utilization 0.75
nohup vllm serve /root/autodl-tmp/RAG/LLaMA-Factory-main/output/qwen3_lora_sft_0.6B --max-model-len 8192 --gpu-memory-utilization 0.7 
