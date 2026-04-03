import os
import sys
import torch
from datasets import load_dataset
from tqdm import tqdm  # Pour voir la progression

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from src.sae_loader import load_model_and_sae

def run_category4_loss_recovery():
    print("Initializing Category 4: Macroscopic Evaluation (Loss Recovery)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sae = load_model_and_sae(device)
    
    # check advised by LLM to justify results
    print("\n--- INFO ARCHITECTURE ---")
    print(f"Dimension de la couche MLP (Entrée) : {sae.W_enc.shape[0]}")
    print(f"Nombre de caractéristiques du SAE (Sortie) : {sae.W_enc.shape[1]}")

    results_dir = os.path.join(BASE_DIR, "results", "exp_category4")
    os.makedirs(results_dir, exist_ok=True)

    # we will use a subset of 1000 samples of the small version of the pile dataset
    print("Downloading a subset of PILE dataset")
    dataset = load_dataset("NeelNanda/pile-10k", split="train")

    
    # to avoid vram crash, advised by LLM 
    test_texts = 1000
    batch_size = 10  # batches of 10 texts instead of 1000 in a row 
    texts = dataset[:test_texts]['text']
    
    # 128 tokens to avoi vram crash (and value used by Naad in his code)
    all_tokens = model.to_tokens(texts, truncate=True)[:, :128].cpu()
    
    print(f"Testing on {all_tokens.shape[0]} docs in batches of {batch_size}")

    # hook
    def zero_hook(acts, hook):
        return torch.zeros_like(acts)

   def sae_hook(acts, hook):
        encoded = sae.encode(acts)
        decoded = encoded @ sae.W_dec
    
        if hasattr(sae, "b_dec"):
            decoded = decoded + sae.b_dec
    
        return decoded

       
    # done with LLM 
    def compute_avg_loss(token_tensor, hooks=None):
        total_loss = 0.0
        count = 0
        
        for i in tqdm(range(0, len(token_tensor), batch_size), desc="Computing loss"):
            batch = token_tensor[i : i + batch_size].to(device)
            
            with torch.no_grad():
                if hooks:
                    loss = model.run_with_hooks(
                        batch, 
                        return_type="loss", 
                        fwd_hooks=hooks
                    )
                else:
                    loss = model(batch, return_type="loss")
                
                total_loss += loss.item() * batch.size(0)
                count += batch.size(0)
            
            # avoid vram crash
            del batch
            torch.cuda.empty_cache()
            
        return total_loss / count

    # loss computation and recovery loss 
    
    clean_loss = compute_avg_loss(all_tokens)
    print(f"L_clean: {clean_loss:.4f}")

    zero_loss = compute_avg_loss(all_tokens, [("blocks.0.mlp.hook_post", zero_hook)])
    print(f"L_zero: {zero_loss:.4f}")

    sae_loss = compute_avg_loss(all_tokens, [("blocks.0.mlp.hook_post", sae_hook)])
    print(f"L_SAE: {sae_loss:.4f}")

    # equation of the paper 
    loss_recovery = (zero_loss - sae_loss) / (zero_loss - clean_loss)
    loss_recovery_pct = loss_recovery * 100

    print(f"Loss recovery score: {loss_recovery_pct:.2f}%")

    # save results for further analysis
    output_path = os.path.join(results_dir, "loss_recovery_metrics_v2.txt")
    with open(output_path, "w") as f:
        f.write(f"L_clean: {clean_loss:.4f}\n")
        f.write(f"L_zero: {zero_loss:.4f}\n")
        f.write(f"L_SAE: {sae_loss:.4f}\n")
        f.write(f"Loss Recovery: {loss_recovery_pct:.2f}%\n")
        
    print(f" finished,  results saved in {output_path}")

if __name__ == "__main__":
    run_category4_loss_recovery()