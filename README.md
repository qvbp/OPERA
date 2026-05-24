# OPERA (CIKM 2026)

OPERA: Option-level Pedagogical Explanation via Reasoning for Augmented Knowledge Tracing

## Project Structure

- **configs/**: Configuration files for the project
- **data/THINK/**: Full data of the THINK dataset
- **data/MATH_G4-5/**: Full data of the MATH_G4-5 dataset
- **data/MATH_G7/**: Full data of the MATH_G7 dataset
- **data/pro_emb/**: LLM-generated pedagogical explanations and pre-trained semantic embeddings
- **examples/**: Example scripts and usage demonstrations
- **pykt/**: Model implementation scripts
- **get_analysis/**: Scripts to generate option-level pedagogical analyses for different datasets (THINK, MATH_G4-5, MATH_G7)
- **get_embedding/**: Scripts to generate semantic embeddings for questions
- **judge_for_generate/**: Multi-judge quality assurance module
  - `evaluate_en.py`: Evaluator for English content
  - `evaluate_zh.py`: Evaluator for Chinese content
  - `eval_outputs_gpt4.1/`: GPT-4.1 evaluation results
  - `eval_outputs_manual_review/`: Human expert review results
  - `final_scores.ipynb`: Final score aggregation

## Installation

```bash
conda create --name=opera python=3.8.17
conda activate opera
cd opera
pip install -e .
```

## Dataset

For full reproducibility, all three datasets (THINK, MATH_G4-5, MATH_G7), together with all LLM-generated pedagogical explanations and pre-trained semantic embeddings, are released in this repository under `data/`.

### Preprocess

```bash
cd examples
python data_preprocess.py --dataset_name=THINK
```

## OPERA Train & Evaluate

### Train

```bash
python wandb_xxx_train.py --dataset_name=THINK
```

### Evaluate

```bash
python wandb_predict.py --save_dir="/path-of-previous-trained-model"
```

## Hyper-parameters

See `examples/seedwandb/xxx.yaml`.
