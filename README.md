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
  raw/                  Original Zindi files
  processed/            Cleaned train, validation and test splits (created by src/data_cleaning.py)
notebooks/
  00_model_template.ipynb  Starting point for every model notebook
  01_eda.ipynb             Exploratory data analysis
results/
  experiment_logs/      One CSV per model, one row per experiment
  figures/              Saved plots, one folder per model
  misclassified/        Wrong test predictions per model, for the error analysis
  submissions/          Zindi submission files
src/
  config.py             Paths, labels, random seed and split sizes
  data_cleaning.py      Cleans the raw text and creates the stratified splits
  data_loading.py       Loads the splits and converts labels to ids
  evaluation.py         Scores, misclassified examples and Zindi submissions
  experiment_log.py     Saves one row per experiment
  plots.py              Confusion matrix, ROC curves and learning curves
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

## Training a model

1. Copy `notebooks/00_model_template.ipynb` and rename it after your model.
2. Set `MEMBER` and `MODEL_NAME` at the top.
3. Replace the example model with yours. It must output a probability for each class, in the order of `config.LABELS`.
4. After every experiment, run the evaluation section. Change `experiment_name` and `notes` so the log shows how your experiments progressed.
5. When you have chosen your final settings, run the test section once.
6. Add your notebook and your files in `results/` to the repository.

| Notebook | Model | Owner |
|---|---|---|
| 02_logistic_regression.ipynb | TF-IDF + Logistic Regression | Vestine |
| 03_naive_bayes.ipynb | TF-IDF + Naive Bayes | Samuel |
| 04_cnn.ipynb | 1D CNN | Alain |
| 05_bilstm.ipynb | BiLSTM with attention | Alain |
| 06_transformer.ipynb | Fine-tuned transformer | Michael |

### Rules for a fair comparison

- Everyone uses the same splits from `data_loading`.
- Train on `train`, tune on `validation`, and only use `test` once for the final scores.
- Log every experiment, including the ones that did badly. They are evidence too.

### Metrics

| Metric | Why |
|---|---|
| Macro-F1 (main metric) | Every class counts equally, so the rare classes matter |
| Log loss | The metric Zindi uses to score submissions |
| Per-class F1 | Shows which classes each model struggles with |
| Accuracy | Reported for reference only, because it hides the rare classes |

For reference, a model that only predicts the class shares scores a macro-F1 of 0.11 and a log loss of 1.15 on validation.

## Splits

The labelled data is split 70 / 15 / 15, stratified by class, with random seed 42.

| Category | Train | Validation | Test |
|---|---|---|---|
| kitaifa | 1400 | 300 | 300 |
| michezo | 1203 | 258 | 258 |
| biashara | 951 | 204 | 204 |
| kimataifa | 38 | 8 | 8 |
| burudani | 11 | 2 | 3 |
