# M-DESIGN

Official implementation for the ICML 2026 paper:

**Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural
Network Design**

M-DESIGN transfers fine-grained GNN architecture knowledge from a released model
bank to a target graph dataset. It first identifies relevant benchmark datasets,
retrieves or estimates architecture-level knowledge, and then refines the target
candidate through iterative knowledge weaving.

## Repository Contents

This repository provides the implementation, released knowledge base, and
reproducibility utilities for the M-DESIGN pipeline.

Included components:

- `main.py`: end-to-end M-DESIGN entry point.
- `graph_understanding/`: graph statistics and dataset descriptions.
- `graph_comparison/`: Kendall-rank similarity and optional LLM similarity.
- `knowledge_retrieval/`: released model-performance databases, ECC predictors,
  model graphs, retrieval code, and candidate evaluator.
- `model_refinement/`: knowledge-weaving controller for architecture refinement.
- `GraphGym/`: GraphGym runner used when target candidates must be trained and
  evaluated instead of read from the released database.
- `scripts/`: artifact download and release-validation utilities.
- `tests/`: release tests for retrieval, refinement, and artifact integrity.

The released knowledge base contains 33 SQLite databases. Each released
task/dataset pair includes the ECC predictor and model graph artifacts required
by the M-DESIGN estimator.

## GraphGym Adaptation

The `GraphGym/` directory is adapted from the public
[`snap-stanford/GraphGym`](https://github.com/snap-stanford/GraphGym) project,
introduced with *Design Space for Graph Neural Networks* by Jiaxuan You,
Zhitao (Rex) Ying, and Jure Leskovec (NeurIPS 2020). We retain the upstream
license notices in the GraphGym subtree.

M-DESIGN uses this adapted GraphGym runner for candidate training and
evaluation. The runner extends GraphGym to support the fine-grained
architecture space studied in our paper, including:

- intra-layer neighborhood choices such as original edges, higher-order edges,
  and kNN rewiring from structural encodings;
- edge-weight/normalization choices such as degree normalization, attention-like
  weights, relative random-walk encodings, and relative Laplacian encodings;
- aggregation and combination choices used by the released model bank;
- inter-layer choices including skip connections, PPR-style propagation,
  GPR-style adaptive layer weighting, LSTM aggregation, and node-adaptive
  gating;
- task-specific decoding or pooling choices for link and graph classification.

This subtree therefore combines upstream GraphGym infrastructure with
M-DESIGN-specific architecture choices, while preserving upstream attribution
and license notices.

## Environment

The code was validated on Windows with Python 3.9, PyTorch 2.8.0, CUDA 12.6
wheels, and an NVIDIA GPU. For other platforms or CUDA versions, install
PyTorch and PyG wheels that match the local runtime.

```bash
git clone https://github.com/jilwang84/M-DESIGN.git
cd M-DESIGN

python -m venv .venv
.\.venv\Scripts\activate

pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -e . -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

If you use a different PyTorch/CUDA build, replace
`torch-2.8.0+cu126` in the PyG wheel URL with the matching wheel index from
[https://data.pyg.org/whl/](https://data.pyg.org/whl/).

Optional extras:

```bash
pip install -e ".[llm]"  # Optional LLM-based benchmark similarity
pip install -e ".[hf]"   # Hugging Face artifact download helper
pip install -e ".[dev]"  # Tests and linting tools
```

## Knowledge Base

The repository tracks the released `.db` and `.pt` artifacts with Git LFS. The
same knowledge-base bundle is also available on Hugging Face:

[https://huggingface.co/datasets/jilwang804/M-DESIGN-Knowledge-Base](https://huggingface.co/datasets/jilwang804/M-DESIGN-Knowledge-Base)

To refresh local artifacts from Hugging Face:

```bash
python scripts/download_knowledge_base.py
```

## Recommended Reproduction

For target datasets already covered by the released model bank, use database
evaluation. This reproduces M-DESIGN refinement by reading candidate performance
from the released databases and enables the released ECC gain estimator.

```bash
python main.py \
  --dataset Cora \
  --task node_classification \
  --similarity_metric kendall \
  --candidate_eval database \
  --use_estimator \
  --window 40 \
  --similarity_threshold -0.9 \
  --max_iter 100
```

Expected Cora result with the released database:

- Initial transfer: `0.8518 +/- 0.0285`
- Best/final transfer after 100 iterations: `0.8850 +/- 0.0083`
- Final architecture:
  `{'neigh': 'edge_index', 'norm': 'rel_lepe', 'agg': 'mean', 'comb': 'concat', 'l_mp': '6', 'stage': 'ppr_01'}`

## Candidate Evaluation Modes

M-DESIGN supports three candidate evaluation modes:

- `--candidate_eval database`: read target candidate performance from the
  released database. Use this for model-bank datasets and reproducible release
  checks.
- `--candidate_eval auto`: read from the database when available; otherwise
  train and evaluate the candidate with GraphGym.
- `--candidate_eval train`: force GraphGym training/evaluation even when the
  target dataset is present in the released database.

Example GraphGym evaluator run:

```bash
python main.py \
  --dataset Cora \
  --task node_classification \
  --candidate_eval train \
  --use_estimator \
  --window 40 \
  --similarity_threshold -0.9 \
  --gpu_id 0 \
  --candidate_repeat 3
```

Candidate training writes outputs to `outputs/candidate_runs` by default.
Repeating the same candidate with the same evaluation configuration reuses the
cached aggregated GraphGym result.

## Optional LLM Similarity

Kendall-rank similarity is the default and requires no API key. To use the
optional LLM-based benchmark similarity:

```bash
set OPENAI_API_KEY=your_key_here
python main.py \
  --dataset Cora \
  --task node_classification \
  --similarity_metric LLM \
  --candidate_eval database \
  --use_estimator \
  --window 40 \
  --similarity_threshold -0.9
```

You can also pass `--openai_api_key_file path/to/key.txt`. Key files are ignored
by Git and should never be committed.

## Validation

Run the release checks before submitting changes:

```bash
python scripts/validate_release.py --root .
pytest
```

The validator checks database readability, release artifact structure, and
repository hygiene.

## Citation

```bibtex
@inproceedings{wang2026mdesign,
  title = {Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design},
  author = {Wang, Jialiang and Liu, Hanmo and Di, Shimin and Wang, Zhili and
            Wang, Jiachuan and Chen, Lei and Zhou, Xiaofang},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026}
}
```

This release also includes modified GraphGym components. If you use the
GraphGym-based runner, please also cite:

```bibtex
@inproceedings{you2020design,
  title = {Design Space for Graph Neural Networks},
  author = {You, Jiaxuan and Ying, Zhitao and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year = {2020}
}
```

## License

The M-DESIGN code is released under the Apache-2.0 License. Modified GraphGym
components retain their upstream MIT license notices in the `GraphGym/` subtree.
See `LICENSE` and the GraphGym license files for details.
