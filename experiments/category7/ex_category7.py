import torch
import matplotlib.pyplot as plt
import os, sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)
from src.sae_loader import load_model_and_sae

def run_saturation_test():
    model, sae = load_model_and_sae()
    feature_idx = 16094 # DNA feature
    
    # Construction of a sequence : "The sequence is A", "The sequence is AT", etc.
    base_prompt = "The sequence is "
    nucleotides = "ATCGATCGATCGATCGATCG"
    activations = []
    
    for i in range(len(nucleotides)):
        current_text = base_prompt + nucleotides[:i+1]
        tokens = model.to_tokens(current_text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter="blocks.0.mlp.hook_post")
            sae_acts = sae.encode(cache["blocks.0.mlp.hook_post"])
            # Selection of the last token added
            val = sae_acts[0, -1, feature_idx].item()
            activations.append(val)
            
    plt.plot(range(1, len(nucleotides)+1), activations, marker='o', color='#e63946')
    plt.xlabel("Number of Nucleotides in Context")
    plt.ylabel("Activation Intensity (Feature #16094)")

    # Create directory if it doesn't exist
    output_dir = os.path.join(BASE_DIR, "results", "category7")
    os.makedirs(output_dir, exist_ok=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    plt.savefig(os.path.join(output_dir, "saturation_plot.png"))
    print(f"Plot saved to {output_dir}/saturation_plot.png")

if __name__ == "__main__":
    run_saturation_test()