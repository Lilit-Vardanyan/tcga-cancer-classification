# TCGA Cancer type Classification

## Project overview

This project explores machine learning methods for classifying five cancer types based on gene-expression data from the Cancer Genome Atlas (TCGA). 

The workflow includes exploratory data analysis, preprocessing, featuringe engineering and hyperparameter tuning. 

## Dataset

Dataset: Fllamber/TCGA-PANCAN-HiSeq-2770x20530
Source: https://huggingface.co/datasets/Fllamber/TCGA-PANCAN-HiSeq-2770x20530
Target: 'cancer'
Features: gene-expression measurments 

The dataset contains five different cancer classes: BRCA, KIRC, THCA', LUAD, UCEC. 

## Project structure 

notebooks: data exploration and model experiments
src: reusable Python modules 
data: dataset documentation and local data 
models: saved model artifacts
reports: figures and evaluation results
app: streamlit application
deployment: deployment files
tests: automated tests

##Installation 

Create and activate Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Experiments

The project explores:

- Exploratory data analysis
- Data preprocessing 
- Principal component analysis (PCA)
- Feauture selection
- Ensemble classifiers
- Hyperparameter tuning with cross-validation

## Evalutation

Macro F1-scroe is the main model comparison metric. Accurancy, precision, recall and per clas f1-scores are also used for evaluation classifiaction performance.

Final test set results will be documented after model selection.

## Reproducibility

Random seeds are set where needed. Preprocessing should be fitted on training data only, with validation set for model selection and the test set is reserved for final evaluation.

## MLflow

## Streamlit Application

## Docker

## Limitations

This project is used only for educational purposes. 
It is not intended for clinical diagnosis or medical decisions. 