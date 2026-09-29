# Swahili News Classification with Sequential Models

ALU NLP and Language Technologies, Year 3 Trimester 2, Group Formative Assignment 2.

We compare five approaches for classifying Swahili news articles into five categories, using the [Zindi Swahili News Classification](https://zindi.world/competitions/swahili-news-classification-challenge/data) dataset.

| Category | Meaning |
|---|---|
| kitaifa | national |
| michezo | sports |
| biashara | business |
| kimataifa | international |
| burudani | entertainment |

## Team

| Member | Role |
|---|---|
| Michael Nwuju | Research, data cleaning, EDA, shared code, fine-tuned transformer, report |
| Alain | 1D CNN, BiLSTM with attention, Related Work |
| Samuel | TF-IDF + Naive Bayes, Results and Discussion |
| Vestine | TF-IDF + Logistic Regression, Error Analysis and Limitations |

## Repository structure

```
data/
  raw/            Original Zindi files
  processed/      Cleaned train, validation and test splits (created by src/data_cleaning.py)
notebooks/
  01_eda.ipynb    Exploratory data analysis
results/
  figures/        Saved plots
src/
  config.py         Paths, labels, random seed and split sizes
  data_cleaning.py  Cleans the raw text and creates the stratified splits
  data_loading.py   Helper functions to load the splits in any notebook
```

## How to run

### Google Colab

Open any notebook in Colab. The first cell clones this repository and installs the requirements.

### Locally

```bash
pip install -r requirements.txt
python -m src.data_cleaning
jupyter notebook notebooks/01_eda.ipynb
```

## Using the shared data in your notebook

Every model must use the same splits so the results can be compared fairly.

```python
from src import data_loading

train = data_loading.load_train()
validation = data_loading.load_validation()
test = data_loading.load_test()

train_label_ids = data_loading.labels_to_ids(train["category"])
```

Train on `train`, tune on `validation`, and only use `test` once for the final scores.

## Splits

The labelled data is split 70 / 15 / 15, stratified by class, with random seed 42.

| Category | Train | Validation | Test |
|---|---|---|---|
| kitaifa | 1400 | 300 | 300 |
| michezo | 1203 | 258 | 258 |
| biashara | 951 | 204 | 204 |
| kimataifa | 38 | 8 | 8 |
| burudani | 11 | 2 | 3 |
