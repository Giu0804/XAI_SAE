import os
import json
import sys
import torch
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae

def run_qualitative_analysis():
    print("Step 1: Loading Model, SAE and identified Feature Index...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    # Load the index found in the previous experiment
    metrics_path = os.path.join(BASE_DIR, "results", "exp_category1", "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    feature_idx = metrics["sae_dna_feature_index"]
    
    # Load dataset
    data_path = os.path.join(BASE_DIR, "data", "mixed_dna_dataset.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # We only care about texts where we know there is DNA
    dna_texts = [item["text"] for item in dataset if item["has_dna_injected"]]
    
    print(f"Step 2: Scanning {len(dna_texts)} DNA samples for Feature #{feature_idx}...")
    
    exemplars = [] # List of (activation_value, context_string)

    with torch.no_grad():
        for text in tqdm(dna_texts):
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
            sae_acts = sae.encode(cache["blocks.0.mlp.hook_post"])
            
            # Extract activations for our specific feature
            # Shape: [1, seq_len, d_hidden] -> we take [1, :, feature_idx]
            feature_acts = sae_acts[0, :, feature_idx]
            
            # Find the max activation in this specific text
            max_val = torch.max(feature_acts).item()
            
            if max_val > 1.0: # Only keep significant activations
                max_pos = torch.argmax(feature_acts).item()
                # Get a small window of text around the activation (5 tokens before/after)
                str_tokens = model.to_str_tokens(tokens[0])
                start = max(0, max_pos - 5)
                end = min(len(str_tokens), max_pos + 5)
                context = "".join(str_tokens[start:end])
                
                exemplars.append((max_val, context))

    # Step 3: Sort by activation and save top 10
    exemplars.sort(key=lambda x: x[0], reverse=True)
    top_10 = exemplars[:10]

    output_path = os.path.join(BASE_DIR, "results", "exp_category1", "top_exemplars.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Top 10 Exemplars for DNA Feature #{feature_idx}\n")
        f.write("="*50 + "\n\n")
        for val, ctx in top_10:
            f.write(f"Activation: {val:.4f}\n")
            f.write(f"Context: ...{ctx}...\n")
            f.write("-" * 30 + "\n")

    print(f"Success! Top exemplars saved in {output_path}")

if __name__ == "__main__":
    run_qualitative_analysis()