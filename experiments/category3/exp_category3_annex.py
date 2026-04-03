import os
import json
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)


from src.sae_loader import load_model_and_sae

def run_annexes():
    print("Initializing Category 3 Annexes: Negative Control & Cliffhanger")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    # Load the DNA feature index
    metrics_path = os.path.join(BASE_DIR, "results", "exp_category1", "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    feature_idx = metrics["sae_dna_feature_index"]
    
    results_dir = os.path.join(BASE_DIR, "results", "exp_category3")
    os.makedirs(results_dir, exist_ok=True)

    # ablation hook def
    def ablation_hook(acts, hook):
        f_i = sae.encode(acts)[:, :, feature_idx : feature_idx+1]
        d_i = sae.W_dec[feature_idx]
        return acts - (f_i * d_i)

   

    # experiment not included in the paper, suggested by LLM 
    print(f"\nrun Cliffhanger")
    cliff_prompt = "The genetic sequence is C C T G G T A C T G "
    tokens_cliff = model.to_tokens(cliff_prompt)
    
    with torch.no_grad():
        clean_logits_cliff = model(tokens_cliff)
        ablated_logits_cliff = model.run_with_hooks(tokens_cliff, fwd_hooks=[("blocks.0.mlp.hook_post", ablation_hook)])
    
    # We look at the probabilities only for the very last token in the sequence
    clean_last_probs = F.softmax(clean_logits_cliff[0, -1, :], dim=-1)
    ablated_last_probs = F.softmax(ablated_logits_cliff[0, -1, :], dim=-1)
    
    # Get Top 5 predictions
    top_clean_probs, top_clean_idx = torch.topk(clean_last_probs, 5)
    top_ablated_probs, top_ablated_idx = torch.topk(ablated_last_probs, 5)
    
    print(f"\nPrompt: '{cliff_prompt} [NEXT WORD?]'")
    print("\n[NORMAL MODEL] Top 5 Predictions:")
    for i in range(5):
        token_str = model.to_string(top_clean_idx[i])
        prob = top_clean_probs[i].item() * 100
        print(f"  {i+1}. '{token_str}' ({prob:.2f}%)")
        
    print("\n[ABLATED MODEL] Top 5 Predictions (Without Feature #16094):")
    for i in range(5):
        token_str = model.to_string(top_ablated_idx[i])
        prob = top_ablated_probs[i].item() * 100
        print(f"  {i+1}. '{token_str}' ({prob:.2f}%)")
        
    # Save text results
    with open(os.path.join(results_dir, "cliffhanger_results_prompt3.txt"), "w") as f:
        f.write(f"Prompt: {cliff_prompt}\n\nNormal Model Top Predictions:\n")
        for i in range(5): f.write(f"  {model.to_string(top_clean_idx[i])} ({top_clean_probs[i].item()*100:.2f}%)\n")
        f.write(f"\nAblated Model Top Predictions:\n")
        for i in range(5): f.write(f"  {model.to_string(top_ablated_idx[i])} ({top_ablated_probs[i].item()*100:.2f}%)\n")

if __name__ == "__main__":
    run_annexes()