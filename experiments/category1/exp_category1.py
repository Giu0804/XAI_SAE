import os
import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae
from src.dataset_dna import get_dna_proxy_scores

def run_category1_experiments():
    print("Initializing environment (CUDA optimized)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    data_path = os.path.join(BASE_DIR, "data", "mixed_dna_dataset.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    texts = [item["text"] for item in dataset]

    # incremetal pearson computation
    n = 0
    sum_proxy = 0.0
    sum_proxy_sq = 0.0
    sum_sae = torch.zeros(sae.d_hidden, device=device)
    sum_sae_sq = torch.zeros(sae.d_hidden, device=device)
    sum_proxy_sae = torch.zeros(sae.d_hidden, device=device)
    
    all_mlp_acts = []
    all_proxy_scores = []

    print("Pass 1/2: Finding best DNA features")
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            tokens = model.to_tokens(batch_texts)
            
            # proxy GT
            batch_proxy = []
            for j in range(len(batch_texts)):
                p_scores = get_dna_proxy_scores(model.to_str_tokens(tokens[j]))
                batch_proxy.extend(p_scores)
            
            proxy_tensor = torch.tensor(batch_proxy, device=device, dtype=torch.float32)
            all_proxy_scores.extend(batch_proxy)

            # xxtract MLP & SAE
            _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
            mlp_acts = cache["blocks.0.mlp.hook_post"].view(-1, model.cfg.d_mlp)
            sae_acts = sae.encode(mlp_acts.view(len(batch_texts), -1, model.cfg.d_mlp)).view(-1, sae.d_hidden)
            
            # update Incremental SAE Sums
            n += mlp_acts.shape[0]
            sum_proxy += proxy_tensor.sum().item()
            sum_proxy_sq += (proxy_tensor**2).sum().item()
            sum_sae += sae_acts.sum(dim=0)
            sum_sae_sq += (sae_acts**2).sum(dim=0)
            sum_proxy_sae += (proxy_tensor.unsqueeze(1) * sae_acts).sum(dim=0)
            
            # store mlp resutlts
            all_mlp_acts.append(mlp_acts.cpu().numpy())

    # Final corr calculation (SAE)
    numerator = (n * sum_proxy_sae) - (sum_proxy * sum_sae)
    denominator = torch.sqrt((n * sum_proxy_sq - sum_proxy**2) * (n * sum_sae_sq - sum_sae**2))
    sae_correlations = numerator / (denominator + 1e-9)
    
    best_sae_idx = torch.argmax(sae_correlations).item()
    best_sae_corr = sae_correlations[best_sae_idx].item()

    # Final Correlation Calculation (MLP)
    mlp_matrix = np.concatenate(all_mlp_acts, axis=0)
    proxy_array = np.array(all_proxy_scores)
    
    # Simple Pearson for MLP
    mlp_corrs = [np.corrcoef(proxy_array, mlp_matrix[:, k])[0, 1] for k in range(mlp_matrix.shape[1])]
    best_mlp_idx = np.argmax(mlp_corrs)
    best_mlp_corr = mlp_corrs[best_mlp_idx]

    print(f"-> Exp 1A | SAE DNA Feature: #{best_sae_idx} (Corr: {best_sae_corr:.4f})")
    print(f"-> Exp 1C | Baseline Neuron: #{best_mlp_idx} (Corr: {best_mlp_corr:.4f})")

    # for visu
    print("Pass 2/2: Collecting data for histograms")
    best_sae_activations = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            tokens = model.to_tokens(batch_texts)
            _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
            sae_acts = sae.encode(cache["blocks.0.mlp.hook_post"])
            # We ONLY store the column of the winner
            best_sae_activations.append(sae_acts[:, :, best_sae_idx].flatten().cpu().numpy())

    sae_winner_acts = np.concatenate(best_sae_activations)

    # visu and saving results, aesthetics visu with LLM help
    results_dir = os.path.join(BASE_DIR, "results", "exp_category1")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({
            "sae_dna_feature_index": int(best_sae_idx),
            "sae_pearson_correlation": float(best_sae_corr),
            "mlp_baseline_neuron_index": int(best_mlp_idx),
            "mlp_pearson_correlation": float(best_mlp_corr)
        }, f, indent=4)

    def plot_hist(acts, labels, title, filename):
        plt.figure(figsize=(10, 6))
        plt.hist([acts[labels == 0], acts[labels == 1]], bins=50, stacked=True, 
                 color=['#e63946', '#457b9d'], label=['Not DNA', 'DNA'])
        plt.yscale('log')
        plt.title(title)
        plt.xlabel('Activation Intensity')
        plt.ylabel('Number of Tokens (Log Scale)')
        plt.legend()
        plt.savefig(os.path.join(results_dir, filename))
        plt.close()

    plot_hist(sae_winner_acts, proxy_array, f"Exp 1B: SAE Feature #{best_sae_idx}", "histogram_sae.png")
    plot_hist(mlp_matrix[:, best_mlp_idx], proxy_array, f"Exp 1C: Neuron #{best_mlp_idx}", "histogram_neuron.png")

    print(f"finisehd,  saved in {results_dir}")

if __name__ == "__main__":
    run_category1_experiments()