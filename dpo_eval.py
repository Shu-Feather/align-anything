from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
import os

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
num = 200
# 随机索引
random_indices = np.random.choice(len(ds), size=num, replace=False)
prompts = [ds[int(i)]["question"] for i in random_indices]


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
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
        )

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
        inputs = reward_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        ).to(device)
        with torch.no_grad():
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

# 创建结果DataFrame
results = pd.DataFrame({
    "prompt": prompts,
    "base_response": base_resps,
    "dpo_response": dpo_resps,
    "base_score": base_scores,
    "dpo_score": dpo_scores,
    "delta_score": delta_scores
})

# 保存完整结果
os.makedirs("/root/align-anything/eval_results", exist_ok=True)
results.to_csv("/root/align-anything/eval_results/full_results.csv", index=False)

# 创建分析图表目录
plot_dir = "/root/align-anything/eval_results/plots"
os.makedirs(plot_dir, exist_ok=True)

# 1. 评分分布对比图
plt.figure(figsize=(12, 8))
sns.kdeplot(results["base_score"], label="Base Model", fill=True, alpha=0.5)
sns.kdeplot(results["dpo_score"], label="DPO Model", fill=True, alpha=0.5)
plt.title("Reward Score Distribution Comparison")
plt.xlabel("Reward Score")
plt.ylabel("Density")
plt.legend()
plt.savefig(f"{plot_dir}/score_distribution.png")
plt.close()

# 2. 评分差直方图
plt.figure(figsize=(10, 6))
plt.hist(delta_scores, bins=50, color='skyblue', edgecolor='black')
plt.axvline(delta_scores.mean(), color='red', linestyle='dashed', linewidth=2, 
            label=f'Mean Δ = {delta_scores.mean():.2f}')
plt.xlabel("DPO Model vs. Base Model")
plt.ylabel("# of Case")
plt.title("Distribution")
plt.legend()
plt.savefig(f"{plot_dir}/delta_score_hist.png")
plt.close()

# 3. 散点图：基模型分数 vs DPO分数
plt.figure(figsize=(10, 8))
plt.scatter(results["base_score"], results["dpo_score"], alpha=0.6, 
            c=np.where(delta_scores > 0, 'green', 'red'), s=50)
plt.plot([min(results["base_score"]), max(results["dpo_score"])], 
         [min(results["base_score"]), max(results["dpo_score"])], 'k--', alpha=0.3)
plt.xlabel("Base Model Score")
plt.ylabel("DPO Model Score")
plt.title("Base vs DPO Model Scores")
plt.colorbar(label="Δ Score (Green > 0, Red < 0)")
plt.grid(True, alpha=0.2)
plt.savefig(f"{plot_dir}/base_vs_dpo_scatter.png")
plt.close()

# 4. 行为改变分析
# 统计信息
delta_positive = (delta_scores > 0).mean()
delta_negative = (delta_scores < 0).mean()
delta_zero = (delta_scores == 0).mean()

print(f"\nΔ>0 的比例：{delta_positive:.2%}")
print(f"Δ<0 的比例：{delta_negative:.2%}")
print(f"Δ=0 的比例：{delta_zero:.2%}")
print(f"平均 Δ：{delta_scores.mean():.4f}；标准差：{delta_scores.std():.4f}")

# 分析改进最大的样本
top_improvements = results.nlargest(5, "delta_score")
top_declines = results.nsmallest(5, "delta_score")

# 保存典型案例
with open(f"{plot_dir}/case_analysis.txt", "w") as f:
    # 改进最大的案例
    f.write("===== 最大改进案例 =====\n")
    for i, row in top_improvements.iterrows():
        f.write(f"\n案例 #{i}\n")
        f.write(f"Prompt: {row['prompt']}\n")
        f.write(f"Δ Score: {row['delta_score']:.2f}\n")
        f.write(f"Base Model ({row['base_score']:.2f}): {row['base_response']}\n")
        f.write(f"DPO Model ({row['dpo_score']:.2f}): {row['dpo_response']}\n")
        f.write("\n" + "-"*80 + "\n")
    
    # 下降最大的案例
    f.write("\n\n===== 最大下降案例 =====\n")
    for i, row in top_declines.iterrows():
        f.write(f"\n案例 #{i}\n")
        f.write(f"Prompt: {row['prompt']}\n")
        f.write(f"Δ Score: {row['delta_score']:.2f}\n")
        f.write(f"Base Model ({row['base_score']:.2f}): {row['base_response']}\n")
        f.write(f"DPO Model ({row['dpo_score']:.2f}): {row['dpo_response']}\n")
        f.write("\n" + "-"*80 + "\n")

# 5. 行为改变总结
print("\n行为改变分析总结:")
if delta_scores.mean() > 0:
    print(f"DPO微调成功 - 平均奖励提升 {delta_scores.mean():.2f} 分")
else:
    print(f"DPO微调可能存在问题 - 平均奖励下降 {abs(delta_scores.mean()):.2f} 分")

# 6. 响应长度分析
def calculate_length(text):
    return len(text.split())

results["base_length"] = results["base_response"].apply(calculate_length)
results["dpo_length"] = results["dpo_response"].apply(calculate_length)
results["length_diff"] = results["dpo_length"] - results["base_length"]

plt.figure(figsize=(10, 6))
sns.scatterplot(x="length_diff", y="delta_score", data=results, hue=np.sign(results["delta_score"]))
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Response Length Variation (DPO - Base)")
plt.ylabel("Reward Variation")
plt.title("The relation between Response Length and Reward")
plt.savefig(f"{plot_dir}/length_vs_score.png")
plt.close()

print("\n分析完成! 结果保存在 /root/align-anything/eval_results/")