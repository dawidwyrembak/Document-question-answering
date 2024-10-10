# Document-question-answering
A repository of source codes prepared for master thesis

## Overview
This project focuses on Document Question Answering, leveraging state-of-the-art transformer models such as **BERT-base-uncased**, **RoBERTa**, and **T5-base**. 
The task involves extracting meaningful answers from structured and semi-structured documents, particularly focusing on scanned receipts, using data prepared from the **SROIE2019**.
The dataset used in this project was manually reviewed, and additional labels were added to enhance the model's ability to answer specific questions more accurately.

The models are fine-tuned to answer questions based on the content of documents, allowing for information retrieval from unstructured sources.

## Dataset: SROIE2019
The **SROIE2019** dataset consists of scanned receipts from various vendors. Each receipt is annotated with important information.
This dataset is commonly used for information extraction and OCR-based question answering tasks. 
In this project, the dataset is used to train models to retrieve specific answers, such as transaction details, from the scanned documents.

## Models
### 1. **BERT-base-uncased**
A pretrained version of BERT with lowercase text and no case distinction. This model is fine-tuned for extracting specific answers from documents.

### 2. **RoBERTa**
RoBERTa is an improved variant of BERT with a robust pretraining approach. It is used in this project for enhancing the performance of document understanding tasks.

### 3. **T5-base**
T5 treats every NLP problem as a text-generation task. Here, it's fine-tuned for question-answering, with both the question and document being inputs and the answer being generated as output.

## Setup

### Requirements
Make sure to install the necessary dependencies to train models:

```bash
pip install -r requirements.txt
```

In case of preparing data with script named `01_paddleocr.py` use:

```bash
pip install -r requirements_paddleocr.txt
```
First scripts was using in MacOS environment with Python3.10, another on Ubuntu also with Python3.10.

## Preprocessing
Before training, the documents must be preprocessed to extract text using OCR tools. The processed text is then paired with the corresponding questions and answers for training.

To prepare dataset, use numbered scripts in `dataset` directory. Choose number `01` to use different OCR. This scripts runs OCR on the SROIE2019 dataset and outputs structured text data for further training and fine-tuning.

## Training
To train the models, use the following commands depending on the model of choice (after going to `model` directory):

```bash
python main.py --model $PRETRAINER_MODEL --train
```

## Hyperparameter search
To find best hyperparameters:
```bash
python main.py --model $PRETRAINER_MODEL --optuna
```

## Evaluation
To evaluate the model performance on the test set:
```bash
python main.py --model $YOUR_FINETUNED_MODEL --evaluate
```

## Inference
Run `model/test.py` model to inference any QA Model. 
In `inference` function provide a question with name of the model, followed by example's index from dataset.


## Model Performance
The following table presents the average evaluation metrics (F1 score and EM) for the models:

|         Model         | F1 Score | EM Score |
|-----------------------|----------|----------|
| **BERT-base-uncased** |  92.93   |  89.74   |
| **RoBERTa-base**      |  94.42   |  93.49   |
| **T5-base**           |  89.43   |  77.91   |


## License
This project was created for a master thesis.