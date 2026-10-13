import json
import os
from pathlib import Path
import time

assert os.environ.get("GNRL_GPU_GUARD") == "1", "Use scripts/launch_gpu.py"
assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

started = time.monotonic()
model_path = "/home/jsy/zhangranxu-adrs-week-20260929/models/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda:0")
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "In ALFWorld, you see a closed fridge 1. Output only the command to open it."}], tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
with torch.inference_mode():
    answer = model.generate(**inputs, max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.eos_token_id)
result = {"cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "visible_count": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0), "model": model_path, "response": tokenizer.decode(answer[0, inputs.input_ids.shape[1]:], skip_special_tokens=True), "elapsed_seconds": time.monotonic()-started, "allocated_gb": torch.cuda.max_memory_allocated()/1e9, "status": "smoke_only_not_task_success"}
Path("outputs/gpu_smoke.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
