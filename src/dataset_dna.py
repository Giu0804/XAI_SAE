import os
import json
import random
import re
from datasets import load_dataset
import sys

import random
random.seed(42)

def generate_dna_sequence(min_len=15, max_len=50):
    """generate a DNA sequence with random choice betwen A T C G"""
    length = random.randint(min_len, max_len)
    
    bases = ['A', 'T', 'C', 'G']
    sequence_list = random.choices(bases, k=length) # équidistribué avec random.choices
    
    sequence = ""
    for base in sequence_list:
        sequence += base
    
    return sequence

def create_dataset(total_samples=5000, dna_ratio=0.2, save_path="../data/mixed_dna_datasetv3.json"):
    """
    Download the base texts and inect DNA sequences in 20% of the texts.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Loading {total_samples} samples text from hugging face")
    dataset = load_dataset("NeelNanda/c4-code-20k", split="train", streaming=True)
    
    mixed_data = []
    num_dna_samples = int(total_samples * dna_ratio) # 1000
    
    for i, example in enumerate(dataset):
        if i >= total_samples:
            break
            
        base_text = example['text'][:700] # to avoi vram crashes we need to put a limit
        
        #injection pour les 1000 premiers textes
        if i < num_dna_samples:
            dna = generate_dna_sequence()
            # On coupe au milieu
            split_idx = random.randint(0, len(base_text)) # random place for the DNA sequence
            # On insère l'ADN avec des espaces autour
            text_with_dna = base_text[:split_idx] + " " + dna + " " + base_text[split_idx:]
            
            mixed_data.append({
                "id": i,
                "text": text_with_dna, 
                "has_dna_injected": True,
                "injected_dna_sequence": dna
            })
        # nothin in the 4000 other texts
        else:
            mixed_data.append({
                "id": i,
                "text": base_text, 
                "has_dna_injected": False,
                "injected_dna_sequence": None
            })
            
    # random shuffle to have random places for the infected dna samples
    random.shuffle(mixed_data)
    
    # save
    print(f"Saving data (json) in {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mixed_data, f, indent=4)
        
    print(f" {len(mixed_data)} samples saved with {num_dna_samples} dna infected texts")


def get_dna_proxy_scores(tokens_list):
    """
    binary choice to detect if a sample is dna infected
    """
    scores = [0.0] * len(tokens_list)
    
    # recup tt les tokens qui ressemblent à de l adn
    is_atcg_candidate = []
    for token in tokens_list:
        clean_token = token.strip()
        # Vrai si le token n'est composé que de A, T, C, G
        if clean_token and re.fullmatch(r"^[ATCG]+$", clean_token):
            is_atcg_candidate.append(True)
        else:
            is_atcg_candidate.append(False)
            
    # Pour avoir 1.0, il faut être un candidat et avoir un voisin qui est aussi un candidat, éviter par exemple les "CAT" qui peuvent être mots anglais
    for i in range(len(tokens_list)):
        if is_atcg_candidate[i]:
            left_is_dna = (i > 0) and is_atcg_candidate[i-1]
            right_is_dna = (i < len(tokens_list) - 1) and is_atcg_candidate[i+1]
            
            if left_is_dna or right_is_dna:
                scores[i] = 1.0
            else:
                scores[i] = 0.0
                
    return scores


if __name__ == "__main__":
    create_dataset()
    sys.exit(0) # sinon ca tourne dans le vide alors que le dataset est cree en qlq secondes