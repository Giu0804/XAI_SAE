# Unveiling Monosemanticity: Causal Validation of SAEs on DNA Features

This repository implements a rigorous validation pipeline for **Sparse Autoencoders (SAEs)**, focused on identifying and causally validating monosemantic features related to DNA sequences within a 1-layer Transformer (`gelu-1l`). The methodology is heavily inspired by the work of Anthropic on dictionary learning and mechanistic interpretability.

## Repository Architecture

The project is organized into modular components for data generation, model loading, and categorized experimental validation:

* **`src/`**: Core infrastructure and utility modules.
    * `sae_loader.py`: Handles loading the `HookedTransformer` and Neel Nanda's pre-trained Sparse Autoencoders from Hugging Face.
    * `dataset_dna.py`: Logic for generating synthetic DNA sequences and injecting them into the C4 base corpus.
* **`experiments/`**: Scripted experimental categories 1 through 8, each targeting a specific interpretability metric.
* **`data/`**: Storage for generated datasets (e.g., `mixed_dna_dataset.json`).
* **`results/`**: Output directory for metrics (`.json`), qualitative analyses (`.txt`), and visualizations (`.png`).

---

## Getting Started

### Prerequisites
* Python 3.10+
* PyTorch (CUDA recommended for faster scanning)
* `transformer_lens`, `datasets`, `tqdm`, `matplotlib`

### Installation
```bash
git clone https://github.com/Giu0804/XAI_SAE.git
cd XAI_SAE
pip install -r requirements.txt
```

If you use Onyxia, please refer to the [commands.txt](./commands.txt) file to get the right version of PyTorch.
## Experimental Pipeline

The project follows a structured validation path divided into eight distinct categories:

### Phase 1: Identification & Qualitative Analysis
* **Category 1: Correlation Analysis** – Identifying the DNA feature by calculating the Pearson correlation between SAE latents and a custom DNA proxy.
* **Category 2: Logit Lens Analysis** – Performing a rigorous theoretical analysis to identify tokens physically promoted by the feature circuit.

### Phase 2: Causal Validation
* **Category 3: Causal Interventions** – Validating the feature through direct manipulations, including feature ablation (functional removal) and steering (forced activation to influence generation).
* **Category 4: Macroscopic Evaluation** – Measuring Loss Recovery metrics to ensure the SAE captures a significant portion of the global model behavior.

### Phase 3: Robustness & Scaling (Challenges)
* **Category 5: Feature Splitting Analysis** – Investigating whether DNA concepts decompose into more granular sub-features in higher-dimensional SAE architectures.
* **Category 6: Dead Latents Analysis** – Scanning for inactive features across a broad corpus to evaluate dictionary training efficiency.
* **Category 7: Saturation Test** – Measuring feature activation intensity against nucleotide sequence length to determine detection thresholds.
* **Category 8: Interference Testing** – Checking for residual polysemy and cross-domain activation in mixed contexts (e.g., DNA strings within Python code).

---

## Key Results

* **Monosemanticity**: Feature #16094 demonstrates a strong Pearson correlation ($r \approx 0.75$) with DNA sequences, significantly outperforming individual MLP neurons ($r \approx 0.47$).
* **Causality**: Forcing Feature #16094 via steering (coefficient 4.5) successfully overrides standard text generation, compelling the model to produce repetitive nucleotide sequences.
* **Efficiency**: Global scans reveal a remarkably low "dead latent" percentage ($0.22\%$), indicating a high effective dictionary capacity.
* **Granularity**: Feature splitting analysis shows that as SAE capacity increases, the DNA concept is distributed across multiple specialized features, such as Feature #3949 with a correlation of $r \approx 0.96$.

---

## 📜 Acknowledgments
* This project utilizes the `transformer_lens` library and Sparse Autoencoder weights provided by **Neel Nanda**. 
* The synthetic data pipeline is built using a sub-sample of the **NeelNanda/c4-code-20k** dataset.
