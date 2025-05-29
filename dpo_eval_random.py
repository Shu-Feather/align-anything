from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
import os
import random

device = "npu:0"  

# Base model and DPO fine-tuned model
base_model_name = "/data/Qwen2.5-0.5B-Instruct"
dpo_model_name  = "/root/align-anything/hsy_0528_outputs_dpo/qwen_2_5_dpo/slice_end"

base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
dpo_tokenizer  = AutoTokenizer.from_pretrained(dpo_model_name)
base_model     = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
dpo_model      = AutoModelForCausalLM.from_pretrained(dpo_model_name).to(device)

# Reward model
reward_model_name = "/root/align-anything/hsy_0528_outputs/qwen_2_5_rm/slice_end"
reward_tokenizer  = AutoTokenizer.from_pretrained(reward_model_name)
reward_model      = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)

# Load full dataset
full_ds = load_dataset("/data/align_anything_t2t", split="validation")

# Configuration parameters
num_samples = 20  # Number of samples per experiment
num_runs = 25       # Number of repeated experiments
max_new_tokens = 512

# Create results directory
result_dir = "/root/align-anything/eval_results_random"
os.makedirs(result_dir, exist_ok=True)
plot_dir = os.path.join(result_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

def generate_responses(model, tokenizer, prompts, max_new_tokens=512):
    responses = []
    for prompt in tqdm(prompts, desc="Generating responses", unit="sample"):
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
            pad_token_id=tokenizer.eos_token_id  # Ensure pad token is set
        )

        gen = outputs[0][inputs.input_ids.shape[-1]:]
        resp = tokenizer.decode(gen, skip_special_tokens=True)
        responses.append(resp)
    return responses

def score_with_reward(reward_model, reward_tokenizer, prompts, responses):
    scores = []
    for p, r in tqdm(zip(prompts, responses), total=len(prompts), desc="Scoring responses", unit="sample"):
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
        
        # Use difference as reward score: positive score - negative score
        score = (logits[0, 1] - logits[0, 0]).item()
        scores.append(score)
    return np.array(scores)

# Store statistics for each experiment
run_stats = []

for run_idx in range(num_runs):
    print(f"\n{'='*50}")
    print(f"Starting experiment {run_idx+1}/{num_runs}")
    print(f"{'='*50}")
    
    # Set random seed for reproducibility
    seed = 42 + run_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Randomly select samples
    random_indices = np.random.choice(len(full_ds), size=num_samples, replace=False)
    prompts = [full_ds[int(i)]["question"] for i in random_indices]
    
    # Generate responses
    print("\nGenerating base model responses...")
    base_resps = generate_responses(base_model, base_tokenizer, prompts, max_new_tokens)
    print("\nGenerating DPO model responses...")
    dpo_resps  = generate_responses(dpo_model, dpo_tokenizer, prompts, max_new_tokens)
    
    # Score responses
    print("\nScoring base model responses...")
    base_scores = score_with_reward(reward_model, reward_tokenizer, prompts, base_resps)
    print("\nScoring DPO model responses...")
    dpo_scores  = score_with_reward(reward_model, reward_tokenizer, prompts, dpo_resps)
    
    # Calculate differences
    delta_scores = dpo_scores - base_scores
    
    # Save detailed results for this experiment
    results = pd.DataFrame({
        "run_id": run_idx,
        "prompt": prompts,
        "base_response": base_resps,
        "dpo_response": dpo_resps,
        "base_score": base_scores,
        "dpo_score": dpo_scores,
        "delta_score": delta_scores
    })
    
    results.to_csv(os.path.join(result_dir, f"run_{run_idx}_results.csv"), index=False)
    
    # Calculate and store statistics for this experiment
    stats = {
        "run_id": run_idx,
        "mean_base_score": np.mean(base_scores),
        "mean_dpo_score": np.mean(dpo_scores),
        "mean_delta": np.mean(delta_scores),
        "std_delta": np.std(delta_scores),
        "delta_positive": np.mean(delta_scores > 0),
        "delta_negative": np.mean(delta_scores < 0),
        "delta_zero": np.mean(delta_scores == 0),
        "win_rate": np.mean(dpo_scores > base_scores),
        "improvement_rate": np.mean(delta_scores > 0.5)  # Significant improvement threshold
    }
    run_stats.append(stats)
    
    print(f"\nExperiment {run_idx+1} statistics:")
    print(f"Average Δ: {stats['mean_delta']:.4f}")
    print(f"Proportion Δ>0: {stats['delta_positive']:.2%}")
    print(f"Win rate (DPO > Base): {stats['win_rate']:.2%}")

# Convert statistics to DataFrame
stats_df = pd.DataFrame(run_stats)
stats_df.to_csv(os.path.join(result_dir, "experiment_stats.csv"), index=False)

# Visualize repeated experiment results
plt.figure(figsize=(12, 8))

# 1. Distribution of average Delta scores
plt.subplot(2, 2, 1)
sns.boxplot(x="mean_delta", data=stats_df)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
plt.title("Distribution of Average Δ Scores Across Experiments")
plt.xlabel("Average Δ Score")

# 2. Change in Delta scores across experiments
plt.subplot(2, 2, 2)
sns.lineplot(x="run_id", y="mean_delta", data=stats_df, marker="o")
plt.fill_between(stats_df["run_id"], 
                 stats_df["mean_delta"] - stats_df["std_delta"], 
                 stats_df["mean_delta"] + stats_df["std_delta"], 
                 alpha=0.2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
plt.title("Change in Δ Scores Across Experiments")
plt.xlabel("Experiment ID")
plt.ylabel("Average Δ Score")

# 3. Win rate change
plt.subplot(2, 2, 3)
sns.barplot(x="run_id", y="win_rate", data=stats_df)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.title("DPO Win Rate Across Experiments")
plt.xlabel("Experiment ID")
plt.ylabel("Win Rate")

# 4. Improvement/regression proportions
plt.subplot(2, 2, 4)
sns.lineplot(x="run_id", y="delta_positive", data=stats_df, label="Δ>0", marker="o")
sns.lineplot(x="run_id", y="delta_negative", data=stats_df, label="Δ<0", marker="o")
sns.lineplot(x="run_id", y="delta_zero", data=stats_df, label="Δ=0", marker="o")
plt.title("Improvement/Regression Proportions Across Experiments")
plt.xlabel("Experiment ID")
plt.ylabel("Proportion")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "repeated_experiment_results.png"))
plt.close()

# Final summary report
final_report = f"""
{'='*50}
Repeated Experiment Evaluation Summary (Total {num_runs} experiments)
{'='*50}

Average base model score: {stats_df['mean_base_score'].mean():.4f} ± {stats_df['mean_base_score'].std():.4f}
Average DPO model score: {stats_df['mean_dpo_score'].mean():.4f} ± {stats_df['mean_dpo_score'].std():.4f}
Average Δ score: {stats_df['mean_delta'].mean():.4f} ± {stats_df['mean_delta'].std():.4f}

Average win rate (DPO > base): {stats_df['win_rate'].mean():.2%} ± {stats_df['win_rate'].std():.2%}
Average improvement proportion (Δ>0): {stats_df['delta_positive'].mean():.2%} ± {stats_df['delta_positive'].std():.2%}
Average regression proportion (Δ<0): {stats_df['delta_negative'].mean():.2%} ± {stats_df['delta_negative'].std():.2%}

Conclusion: 
"""

if stats_df['mean_delta'].mean() > 0.1:
    final_report += "DPO fine-tuning significantly improved model performance (average Δ > 0.1)"
elif stats_df['mean_delta'].mean() > 0:
    final_report += "DPO fine-tuning showed slight improvement (average Δ > 0)"
elif stats_df['mean_delta'].mean() < -0.1:
    final_report += "WARNING: DPO fine-tuning significantly degraded model performance (average Δ < -0.1)"
else:
    final_report += "DPO fine-tuning showed no significant impact on model performance"

print(final_report)

# Save final report
with open(os.path.join(result_dir, "final_report.txt"), "w") as f:
    f.write(final_report)

print(f"\nEvaluation completed! All results saved in {result_dir}")