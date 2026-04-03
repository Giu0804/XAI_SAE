import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer, utils

"""
This code is  very inspired by the notebook of Naad Nandel : https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing 
In his notebook, he explained how to load and use his pre trained SAE - recommended by Anthropic 


"""
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class AutoEncoder(nn.Module):

    
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        dtype = DTYPES[cfg["enc_dtype"]]
        
        # Les matrices et biais (Encodage et Décodage)
        self.W_enc = nn.Parameter(torch.empty(d_mlp, d_hidden, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_mlp, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.d_hidden = d_hidden

    def encode(self, x):
        """Ptake MLP activation and et renvoie les activations des  features"""
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        return acts

    @classmethod
    def load_from_hf(cls, version="run1"):
        """
        load hg weights
        """
        if version == "run1":
            version_id = 25
        elif version == "run2":
            version_id = 47

        # config download
        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version_id}_cfg.json")
        self = cls(cfg=cfg)
        
        #load and inect pre trained weights
        weights_path = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version_id}.pt", force_is_torch=True)
        self.load_state_dict(weights_path)
        
        return self


def load_model_and_sae(device="cuda" if torch.cuda.is_available() else "cpu"):
    
    print(f"Chargement sur : {device}")
    
    # charge le model de base - mlp a une layer
    print("Load Hook Transformer gelu l1")
    model = HookedTransformer.from_pretrained("gelu-1l").to(device)
    
    # sea of naad nnadel 
    print("Load pre trained SAE of Naad Nandel")
    sae = AutoEncoder.load_from_hf("run1")
    sae.to(device)
    
    return model, sae