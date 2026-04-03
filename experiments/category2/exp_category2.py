import os
import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Configuration to import from the src/ directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae

def run_category2_experiments():
    print("category 2: Rigorous Theoretical Logit Analysis")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    # Load the DNA feature index identified in Category 1
    metrics_path = os.path.join(BASE_DIR, "results", "exp_category1", "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    feature_idx = metrics["sae_dna_feature_index"]
    print(f"Analyzing Feature #{feature_idx}")
    
    # d_i: The feature direction in MLP space 
    feature_direction = sae.W_dec[feature_idx]

    # For a 1-layer model, we access the first block's MLP output weights
    W_out = model.blocks[0].mlp.W_out

    W_U = model.W_U

    # calculation - Following Anthropic's methodology

    # project feature from MLP space to Residual stream
    residual_direction = torch.matmul(feature_direction, W_out)

    # B. LayerNorm Approximation (pi L)
    # Centering the vector to remove common bias across the residual stream
    residual_direction = residual_direction - residual_direction.mean()

    # project onto the Vocab (128 -> d_vocab)
    # This gives us the raw logit weights for every single token

    logits_tensor = residual_direction @ W_U
    
    logits_detached = logits_tensor.detach()
    
    logits_cpu = logits_detached.cpu()
    
    # 4. Conversion en numpy
    logit_weights = logits_cpu.numpy()

    # D. Median Shift (Crucial for Bimodality Visualization)
    # We shift weights so the median is zero, making the "signal" stand out from the "noise"
    logit_weights = logit_weights - np.median(logit_weights)

    # Identification of top tokens
    # Get all tokens from the vocabulary
    all_tokens = [model.to_string(i) for i in range(model.cfg.d_vocab)]
    
    # Sort weights to find the tokens most promoted by the feature
    sorted_indices = np.argsort(logit_weights)[::-1] 
    
    top_tokens = []
    print("\nTop 10 tokens physically promoted by this circuit:")
    for i in range(10):
        idx = sorted_indices[i]
        token_str = all_tokens[idx]
        weight = logit_weights[idx]
        top_tokens.append({"token": token_str, "weight": float(weight)})
        print(f"  {i+1}. '{token_str}' (weight: {weight:.4f})")

    #########################
    # results and visu for exp2A, visu with LLM help
    #######################
    results_dir = os.path.join(BASE_DIR, "results", "exp_category2")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "logit_metrics.json"), "w") as f:
        json.dump(top_tokens, f, indent=4)

    # scatter plot for bimodal experiment
    plt.figure(figsize=(12, 6))
    x = np.arange(len(logit_weights))
    
    # Plot all weights 
    plt.scatter(x, logit_weights, alpha=0.4, s=8, color='#457b9d', label='Other Tokens')
    
    # top 5 tokens
    for i in range(5):
        idx = sorted_indices[i]
        plt.scatter(idx, logit_weights[idx], color='#e63946', s=60, edgecolors='black', zorder=5)
        plt.annotate(f" {all_tokens[idx]}", (idx, logit_weights[idx]), color='#e63946')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f"Exp 2A: Logit Weights for DNA Feature #{feature_idx} (Median-Centered)")
    plt.xlabel("Token Vocabulary Index")
    plt.ylabel("Effect on Prediction Probability (Logit Weight)")
    plt.legend()
    plt.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "logit_distribution_bimodal.png"))
    plt.close()

    print(f"\nfinished - saved in {results_dir}")

if __name__ == "__main__":
    run_category2_experiments()