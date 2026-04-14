import os
import sys
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae, AutoEncoder

def run_feature_splitting():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae_run1 = load_model_and_sae(device)
    
    # Load SAE run2 
    print("Loading SAE Run 2...")
    sae_run2 = AutoEncoder.load_from_hf("run2").to(device)

    # Load DNA feature index from Run 1
    metrics_path = os.path.join(BASE_DIR, "results", "category1", "metrics.json")
    with open(metrics_path, "r") as f:
        feature_idx_run1 = json.load(f)["sae_dna_feature_index"]

    # Test sentence with DNA
    test_text = "Analysis of ATCGATCGATCGATCG sequences."
    tokens = model.to_tokens(test_text)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
        acts = cache["blocks.0.mlp.hook_post"]
        
        feat_acts_run1 = sae_run1.encode(acts)[0, :, feature_idx_run1]
        all_acts_run2 = sae_run2.encode(acts)[0, :, :] # [seq_len, d_hidden_run2]

    # Find which features in Run 2 correlate most with our feature in Run 1
    # This identifies if one feature splits into many
    corrs = torch.zeros(sae_run2.d_hidden)
    for i in range(sae_run2.d_hidden):
        if all_acts_run2[:, i].sum() > 0:
            corrs[i] = torch.corrcoef(torch.stack([feat_acts_run1, all_acts_run2[:, i]]))[0, 1]

    top_v, top_i = torch.topk(corrs.nan_to_num(), 5)
    
    results_dir = os.path.join(BASE_DIR, "results", "category5")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Top splitting candidates in Run 2 for Run 1 Feature #{feature_idx_run1}:")
    results = []
    for val, idx in zip(top_v, top_i):
        print(f"  Feature #{idx.item()} - Correlation: {val.item():.4f}")
        results.append({"idx_run2": int(idx), "correlation": float(val)})

    with open(os.path.join(results_dir, "splitting_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_feature_splitting()