import os
import sys
import torch
import json
from tqdm import tqdm
from datasets import load_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae

def run_dead_latents_analysis():
    print("Category 6: Dead Latents Analysis")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    # Use a broader dataset to be sure
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    texts = dataset[:500]['text'] # 500 docs for faster run
    
    dead_mask = torch.ones(sae.d_hidden, device=device, dtype=torch.bool)
    
    print(f"Scanning activations for {sae.d_hidden} features...")
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        tokens = model.to_tokens(batch_texts, truncate=True)[:, :128]
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
            sae_acts = sae.encode(cache["blocks.0.mlp.hook_post"]).view(-1, sae.d_hidden)
            
            # Update dead mask
            currently_active = (sae_acts > 0).any(dim=0)
            dead_mask = dead_mask & (~currently_active)

    num_dead = dead_mask.sum().item()
    dead_pct = (num_dead / sae.d_hidden) * 100
    
    print(f"Results: {num_dead} dead latents found out of {sae.d_hidden} ({dead_pct:.2f}%)")
    
    results_dir = os.path.join(BASE_DIR, "results", "category6")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "dead_latents.json"), "w") as f:
        json.dump({"dead_count": num_dead, "total_count": sae.d_hidden, "percentage": dead_pct}, f, indent=4)

if __name__ == "__main__":
    run_dead_latents_analysis()