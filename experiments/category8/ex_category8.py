import torch
import os, sys
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)
from src.sae_loader import load_model_and_sae

def run_interference_test():
    model, sae = load_model_and_sae()
    dna_feat = 16094
    
    # Python Code + DNA Sequence
    mixed_text = "def calculate_gene(): return 'ATCGATCGATCGATCG'"
    tokens = model.to_tokens(mixed_text)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
        sae_acts = sae.encode(cache["blocks.0.mlp.hook_post"])[0] # [seq, d_sae]
        
    active_features = (sae_acts.mean(dim=0) > 0.5).nonzero().flatten()
    
    print(f"Features activated by the mixed text : {active_features.tolist()}")

if __name__ == "__main__":
    run_interference_test()