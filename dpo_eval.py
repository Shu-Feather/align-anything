import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def generate_response(model, tokenizer, prompt, device='npu:0'):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 加载初始模型和DPO微调模型
device = 'npu:0'
model_initial = AutoModelForCausalLM.from_pretrained("/path/to/initial_model").to(device)
model_dpo = AutoModelForCausalLM.from_pretrained("/path/to/dpo_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("/path/to/tokenizer")

# 加载测试集
from datasets import load_dataset
dataset = load_dataset("PKU-Alignment/Align-Anything", split="test")
test_prompts = dataset["prompt"][:500]  # 使用500个样本进行评测