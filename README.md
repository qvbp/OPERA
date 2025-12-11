# OPERA (WWW 2026)

## Project Structure

- **configs/**: Configuration files for the project
- **data/THINK/**: Sampled data from the THINK dataset
- **examples/**: Example scripts and usage demonstrations
- **pykt/**: Model's scripts
- **get_analysis/**: Scripts to generate option analysis for different datasets (3, 7, 45)
- **get_embedding/**: Scripts to generate embeddings for questions
- **judge_for_generate/**: Automatic evaluation module
  - evaluate_en.py: Evaluator for English dataset
  - evaluate_zh.py: Evaluator for Chinese datasets
  - eval_outputs_gpt4.1/: GPT-4.1 evaluation results
  - eval_outputs_manual_review/: Manual review results
  - final_scores.ipynb: Final score aggregation


## Installation

conda create --name=opera python=3.8.17
conda activate opera
cd opera
pip install -e .

## Dataset
We have open-sourced a portion of the THINK dataset. The full dataset will be released upon paper publication.
- Location: data/THINK/
- File: grade_3_data_sampled_2000.xlsx

### Preprocess
```
cd examples
python data_preprocess.py --dataset_name=THINK
```

## OPERA Train & Evaluate

### Train
```
python wandb_xxx_train.py  --dataset_name=THINK
```

### Evaluate
```
python wandb_predict.py  --save_dir="/path-of-previous-trained-model"
```

## Hyper-parameter
See examples/seedwandb/xxx.yaml
