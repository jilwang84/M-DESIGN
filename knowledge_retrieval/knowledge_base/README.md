# M-DESIGN Knowledge Base

This directory contains the released model-performance databases and model
artifacts used by M-DESIGN.

Layout:

- `node/`: node classification records.
- `link/`: link prediction records.
- `graph/`: graph classification records.

Each task/dataset directory contains:

- a SQLite database with the `model_records` table;
- `ecc_predictor.pt`, the released edge-conditioned candidate-gain predictor;
- `model_graph.pt`, the model-model graph used by the predictor.

Other predictor variants, ablation logs, TensorBoard events, and raw training
outputs are intentionally excluded from the public release.
