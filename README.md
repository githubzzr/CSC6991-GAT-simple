# CSC6991-GAT-simple
### Author: Zerun Zhang

This project is a fork of [graphsage-simple](https://github.com/williamleif/graphsage-simple), with  modifications to support Graph Attention Network (GAT) models in addition to the original GraphSAGE implementation.

- Added GAT model implementation with both single-head and multi-head attention mechanisms.
- Enabled support for both **Cora** and **Pubmed** datasets.
- Added command-line arguments to toggle between GraphSAGE and GAT models.
- CUDA usage can be enabled by manually modifying the relevant variable in `model.py`.

## Environment Requirements

- Python: `3.8.20`
- NumPy: `1.24.4`
- PyTorch: `2.2.1`
- scikit-learn: `1.3.2`

#### Running examples

Run GraphSAGE on Cora

Execute `python -m graphsage.model --dataset cora `

Run Single-Head GAT on Cora

Execute `python -m graphsage.model  --dataset cora   --gat` 


Run Multi-Head GAT on Pubmed

Execute `  python -m graphsage.model  --dataset pubmed --gat ` 