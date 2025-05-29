from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

device = "npu:0"  # 或者 npu:0

# 基模型 和 DPO 微调模型
base_model_name = "/data/Qwen2.5-0.5B-Instruct"
dpo_model_name  = "/root/align-anything/hsy_0528_outputs_dpo/qwen_2_5_dpo/slice_end"

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
dpo_tokenizer  = AutoTokenizer.from_pretrained(dpo_model_name)
base_model     = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
dpo_model      = AutoModelForCausalLM.from_pretrained(dpo_model_name).to(device)

# 奖励模型
reward_model_name = "/root/align-anything/hsy_0528_outputs/qwen_2_5_rm/slice_end"
reward_tokenizer  = AutoTokenizer.from_pretrained(reward_model_name)
reward_model      = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)

# 测试集加载 (PKU-Alignment/Align-Anything)
ds = load_dataset("/data/align_anything_t2t", split="validation")
prompts = ds["question"][:2]  # 根据实际字段名调整

def generate_responses(model, tokenizer, prompts, max_new_tokens=512):
    responses = []
    for prompt in tqdm(prompts, desc="生成响应", unit="样本"):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        gen = outputs[0][inputs.input_ids.shape[-1]:]
        resp = tokenizer.decode(gen, skip_special_tokens=True)
        responses.append(resp)
    return responses

print("生成基模型响应...")
base_resps = generate_responses(base_model, base_tokenizer, prompts)
print("\n生成DPO模型响应...")
dpo_resps  = generate_responses(dpo_model, dpo_tokenizer, prompts)


def score_with_reward(reward_model, reward_tokenizer, prompts, responses):
    scores = []
    for p, r in tqdm(zip(prompts, responses), total=len(prompts), desc="评分响应", unit="样本"):
        text = f"[PROMPT]\n{p}\n[RESPONSE]\n{r}"
        inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        logits = reward_model(**inputs).logits
        
        # 使用差值作为奖励分数：正面分数 - 负面分数
        score = (logits[0, 1] - logits[0, 0]).item()
        scores.append(score)
    return np.array(scores)

print("\n基模型响应评分中...")
base_scores = score_with_reward(reward_model, reward_tokenizer, prompts, base_resps)
print("\nDPO模型响应评分中...")
dpo_scores  = score_with_reward(reward_model, reward_tokenizer, prompts, dpo_resps)
delta_scores = dpo_scores - base_scores


# 直方图：展示 Δscore 分布
plt.figure(figsize=(6,4))
plt.hist(delta_scores, bins=50)
plt.xlabel("DPO 模型评分 - 基模型评分")
plt.ylabel("Case 数量")
plt.title("评分差分布")
plt.show()

# 统计信息
print("Δ>0 的比例：", (delta_scores>0).mean())
print("Δ<0 的比例：", (delta_scores<0).mean())
print("平均 Δ：", delta_scores.mean(), "；标准差：", delta_scores.std())
