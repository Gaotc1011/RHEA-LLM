# RHEA-LLM

Official code repository for the manuscript:

> **"LLM-enabled discovery of lightweight refractory high entropy alloy exhibiting 879 MPa yield strength at 1473 K"**
> *(manuscript submitted / under review)*

This repository contains the complete code pipeline that uses large language models (LLMs) to accelerate the discovery and design of lightweight refractory high-entropy alloys (RHEAs), covering literature data extraction, model fine-tuning, property prediction, inverse alloy design, benchmarking against conventional machine learning, and model interpretability analysis.

---

## Repository Structure

```
all-code/
├── 1-data-prepare/            # Literature data extraction and SFT dataset construction
├── 2-model-train/             # Fine-tuning, prediction and evaluation of the LLM (Qwen3-8B)
├── 3-alloy-design/            # LLM-driven inverse design of lightweight RHEAs
├── 4-LLM-ML_compare/          # Benchmarking the LLM against conventional ML models
└── 5-model_interpretability/  # Attention-based interpretability analysis
```

### 1. `1-data-prepare/` — Data extraction & dataset construction

| Script | Description |
|--------|-------------|
| `1-data_extraction.py` | Converts RHEA literature PDFs to text (`marker`) and extracts composition / processing / microstructure / mechanical-property records into structured JSON using DeepSeek-V3, with DeepSeek-R1 used for result checking |
| `2-dataset_analysis-composition.py` | Statistical analysis of alloy compositions (element frequency, co-occurrence, etc.) |
| `2-dataset_analysis-property+phase.py` | Statistical analysis of properties and phase constitution |
| `3-data_evaluate_composition+phase+property.py` | Data cleaning / evaluation of composition, phase and property records |
| `3-data_evaluate_process.py` | Data cleaning / evaluation of processing routes |
| `4-dataset_to_sft.py` | Converts the curated database into instruction-format SFT (supervised fine-tuning) data |
| `4-dataset_to_sft-scale.py` | SFT data generation with scaled property targets |

Supporting folders: `src-pdf/` (source literature), `extraction-output/` (raw extracted records), `data_analysis/`, `data_evaluate/`, `sft_data/` (training/validation sets).

### 2. `2-model-train/` — Model fine-tuning, prediction & evaluation

| Script | Description |
|--------|-------------|
| `1-model_fine_tuning.py` | Supervised fine-tuning of Qwen3-8B (supports `full`, `freeze` and `lora` strategies; training tracked with SwanLab) |
| `2-model_predict.py` | Property prediction on validation / test alloys with the fine-tuned model |
| `3-mode_evaluate.py` | Quantitative evaluation of prediction accuracy (per-property metrics and plots) |

> **Note:** the fine-tuned checkpoint (`Qwen-3-8B-RHEA-property-predictor/`) is **not included** in this repository due to its size (~15 GB). The base model `Qwen3-8B` should be downloaded separately (e.g., from Hugging Face / ModelScope) and placed under `LLM-model/`.

### 3. `3-alloy-design/` — LLM-driven alloy design

| Script | Description |
|--------|-------------|
| `1-design.py` | LLM-based generation / iterative optimization of lightweight RHEA compositions targeting high yield strength at elevated temperature (1473 K) |
| `2-result_analysis.py` | Analysis of designed candidates against the training database (novelty screening, etc.) |

### 4. `4-LLM-ML_compare/` — LLM vs. conventional ML benchmarks

| Script | Description |
|--------|-------------|
| `0-data_process.py` | Prepares tabular feature data for classical ML baselines |
| `1-ML_model.py` | Trains conventional ML models (e.g., tree-based / kernel models) on the same dataset |
| `2-ML_predict.py` | Generates ML predictions for comparison with the fine-tuned LLM |

### 5. `5-model_interpretability/` — Model interpretability

| Script | Description |
|--------|-------------|
| `interpretability.py` | Extracts and visualizes layer / head attention maps of the fine-tuned model during property prediction (`raw_attention_data-*/` contains exported attention visualizations) |

---

## Environment

Main dependencies (Python ≥ 3.10):

```
torch
transformers
peft
datasets
accelerate
swanlab
openai
marker-pdf
pandas
matplotlib
```

```bash
pip install torch transformers peft datasets accelerate swanlab openai marker-pdf pandas matplotlib
```

## Usage

The workflow follows the numbering of the folders:

```bash
# Step 1: build the literature database and SFT datasets
cd 1-data-prepare
python 1-data_extraction.py          # requires a DeepSeek API key in CONFIG
python 2-dataset_analysis-*.py
python 3-data_evaluate_*.py
python 4-dataset_to_sft.py

# Step 2: fine-tune, predict and evaluate
cd ../2-model-train
python 1-model_fine_tuning.py        # requires the Qwen3-8B base model
python 2-model_predict.py
python 3-mode_evaluate.py

# Step 3: LLM-driven alloy design
cd ../3-alloy-design
python 1-design.py
python 2-result_analysis.py

# Step 4: ML baselines for comparison
cd ../4-LLM-ML_compare
python 0-data_process.py
python 1-ML_model.py
python 2-ML_predict.py

# Step 5: interpretability analysis
cd ../5-model_interpretability
python interpretability.py
```

> Before running, fill in your own API credentials in `1-data-prepare/1-data_extraction.py` (`base_url`, `deepseek_key`) and, if needed, your SwanLab API key in `2-model-train/1-model_fine_tuning.py`.

## Data availability

The literature-derived dataset (extracted JSON records), the SFT training/validation sets, and the source PDFs used in the study are included under `1-data-prepare/`. The fine-tuned model weights are not distributed in this repository.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{RHEA_LLM_2026,
  title   = {LLM-enabled discovery of lightweight refractory high entropy alloy
             exhibiting 879 MPa yield strength at 1473 K},
  author  = {Tianchuang Gao and others},
  journal = {To be updated upon acceptance},
  year    = {2026}
}
```
