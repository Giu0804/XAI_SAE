import os
import json
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae

def run_category3_experiments():
    print("category 3: Causal Interventions (Ablation & Steering)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    # Load the DNA feature index
    metrics_path = os.path.join(BASE_DIR, "results", "exp_category1", "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    feature_idx = metrics["sae_dna_feature_index"]
    
    results_dir = os.path.join(BASE_DIR, "results", "exp_category3")
    os.makedirs(results_dir, exist_ok=True)
    
    ############
    # EXP 3A: FEATURE ABLATION (The Lobotomy)
    #############
    print(f"\nRun Exp 3A: Ablating Feature #{feature_idx}")
    
    test_text = "The analysis of the genetic sequence shows CGGGTTCCAGTGAAATAT which is very interesting."
    tokens = model.to_tokens(test_text)
    str_tokens = model.to_str_tokens(test_text)
    
    # baseline, clean run 
    with torch.no_grad():
        clean_logits = model(tokens)
    
    # definiton of the ablation hook
    def ablation_hook(acts, hook):
        f_i = sae.encode(acts)[:, :, feature_idx : feature_idx+1] 
        d_i = sae.W_dec[feature_idx]
        modified_acts = acts - (f_i * d_i)
        return modified_acts

    # ablated run 
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.mlp.hook_post", ablation_hook)])

    # drop prob calculation
    clean_log_probs = F.log_softmax(clean_logits, dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)

    target_tokens = tokens[0, 1:]
    clean_slice = clean_log_probs[0, :-1, :]
    ablated_slice = ablated_log_probs[0, :-1, :]
    
    positions = torch.arange(target_tokens.shape[0])
    clean_target_lp = clean_slice[positions, target_tokens]
    ablated_target_lp = ablated_slice[positions, target_tokens]
    
    diff_lp = (ablated_target_lp - clean_target_lp).cpu().numpy()
    
    
    # visual part done with LLM 
    
    plt.figure(figsize=(15, 5))
    display_tokens = str_tokens[1:] # Align with diff_lp
    
    colors = ['#457b9d' if x < -0.1 else '#e63946' if x > 0.1 else '#cccccc' for x in diff_lp]
    plt.bar(range(len(display_tokens)), diff_lp, color=colors)
    
    plt.xticks(range(len(display_tokens)), display_tokens, rotation=45, ha='right', fontsize=9)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Impact of Ablating Feature #{feature_idx} on Next-Token Log-Probability")
    plt.ylabel("Log-Prob Difference (Ablated - Clean)")
    plt.xlabel("Tokens")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "ablation_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"-> Ablation Plot saved to {plot_path}")

    # #################
    # EXP 3B:  feature steering
    # ##############
    print(f"\nExp 3B: Steering Feature #{feature_idx}")
    
    prompt = "My name is Bob and I am a MVA's master student."
    
    # clean generation 
    print("Baseline Generation:")
    clean_gen = model.generate(prompt, max_new_tokens=20, temperature=0.0)
    print(f"  {clean_gen}")

    # definition of the steerign hook -- au dessus de l activation max 
    STEERING_COEFF = 4.5 
    
    def steering_hook(acts, hook):
        d_i = sae.W_dec[feature_idx]
        return acts + (STEERING_COEFF * d_i)

    #steered Generation
    print(f"\nSteered Generation (Forcing Feature to {STEERING_COEFF}):")
    with model.hooks(fwd_hooks=[("blocks.0.mlp.hook_post", steering_hook)]):
        steered_gen = model.generate(prompt, max_new_tokens=20, temperature=0.0)
    print(f"  {steered_gen}")
    
    # Save results
    with open(os.path.join(results_dir, "steering_results.txt"), "w") as f:
        f.write(f"PROMPT: {prompt}\n")
        f.write(f"BASELINE: {clean_gen}\n")
        f.write(f"STEERED (Coeff {STEERING_COEFF}): {steered_gen}\n")

    print(f"\nCategory 3 finisehd, results saved {results_dir}")

if __name__ == "__main__":
    run_category3_experiments()