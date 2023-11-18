import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import json
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

# st = time.time()
# os.environ['MASTER_ADDR'] = "localhost"
# os.environ['MASTER_PORT'] = "12350"
# dist.init_process_group(
#     backend='nccl',
#     world_size=4,
#     rank=0)  # NCCL backend is recommended for multi-GPU training on the same machine

# print(time.time() - st)


from transformers import LlamaForCausalLM

st = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_path = "/nfs_data_storage/llama2_hf/Llama-2-7b-hf/"
# Function to load the main model for text generation
model = LlamaForCausalLM.from_pretrained(
    model_path,
    return_dict=True,
    load_in_8bit=False,
    # device_map=device,
    # low_cpu_mem_usage=True,
)
print(time.time() - st)

st = time.time()
model = model.to(device)
# model = nn.parallel.DistributedDataParallel(model)
print(time.time()-st)



from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


image_dir = "/nfs_data_storage/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
data_dir = "/nfs_edlab/shcho/lec/idl/project/data/"

df = pd.read_csv(data_dir + "data_test_full.csv")
import pdb; pdb.set_trace()
shot_idx = [0, 100]
test_idx = 10
user_prompt = f"""Task Description: Generate a concise summary (impression) from the detailed findings of a chest X-ray report.
Example 1.
Findings: {df.findings[shot_idx[0]]}
Impressions: {df.impressions[shot_idx[0]]}

Example 2.
Findings: {df.findings[shot_idx[1]]}
Impressions: {df.impressions[shot_idx[1]]}
---
Findings: {df.findings[test_idx]}
Impressions: """

batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=None, return_tensors="pt")
batch = {k: v.to(device) for k, v in batch.items()}
start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate( **batch, do_sample=True, max_length=1000, top_p=0.9, temperature=1.0, min_length=3, use_cache=True, top_k=50, repetition_penalty=1.0, length_penalty=1 )
e2e_inference_time = (time.perf_counter()-start)*1000
print(f"the inference time is {e2e_inference_time} ms")
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text[len(user_prompt):])
