import torch
from datasets import Dataset
import datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    set_seed
)
import evaluate
import pandas as pd
import collections
import numpy as np
import transformers
from _qa_trainer import QuestionAnsweringSeq2SeqTrainer
from transformers.trainer_utils import EvalPrediction
import optuna 
from typing import Optional 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Model:
    def __init__(self, model_name: str) -> None:
        set_seed(42)
        self.model_name = model_name
        self.device = self._set_device()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.metric = evaluate.load("squad")
        self.load_datasets()

    def _set_device(self) -> torch.device:        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
        
    def load_trainer(self, trial: Optional[optuna.trial.Trial] = None, test_param: Optional[bool] = False):
        training_args = Seq2SeqTrainingArguments(
            output_dir='./_temp',
            learning_rate= 1.0485387725194621e-05 if not trial else trial.suggest_float("learning_rate",  1e-5, 1e-4, log=True),
            weight_decay= 0.009330606024425666 if not trial else trial.suggest_float("weight_decay",  1e-3, 1e-2, log=True),
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            save_strategy='epoch',
            evaluation_strategy = "epoch",
            logging_strategy="epoch",
            logging_steps=100,
            predict_with_generate=True
        )

        self.trainer = QuestionAnsweringSeq2SeqTrainer(
            model=self.model if not trial else None ,
            model_init=None if not trial else self.get_model ,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset if not test_param  else self.test_dataset,
            eval_examples=self.dataset['validation'] if not test_param else self.dataset['test'],
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            data_collator= transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                label_pad_token_id=-100,
            ),
            post_process_function=self.post_processing_function,
            compute_metrics=self.compute_metrics,
        )
        
    def get_model(self):
        return self.model
    
    def load_datasets(self):
        self.dataset = collections.defaultdict()
        self.dataset['train'] = self.load_dataset('files/out_train.csv')
        self.dataset['validation'] = self.load_dataset('files/out_validation.csv')
        self.dataset['test'] = self.load_dataset('files/out_test.csv')

        self.train_dataset = self.dataset['train'].map(
            self.prepare_train_features, 
            batched=True,  
            remove_columns=self.dataset['train'].column_names
        )
        self.validation_dataset = self.dataset['validation'].map(
            self.prepare_validation_features, 
            batched=True, 
            remove_columns=self.dataset['validation'].column_names
        )
        self.test_dataset = self.dataset['test'].map(
            self.prepare_validation_features, 
            batched=True, 
            remove_columns=self.dataset['test'].column_names
        )
        print("TRAIN LEN: ", len(self.train_dataset))
        print("VALID LEN: ", len(self.validation_dataset))
        print("TEST LEN: ", len(self.test_dataset))

    def load_dataset(self, filename: str):
        df = pd.read_csv(filename)
        out = collections.defaultdict(list)

        for index, row in df.iterrows():
            if index ==100:
                break
            out['id'].append(str(index))
            out['context'].append(row['context'])
            out['question'].append(row['question'])
            out['answers'].append({'text': [row['answer_text']], 'answer_start': [row['answer_start']]})
            out['label_id'].append(str(row['label_id']))

        dataset = Dataset.from_dict(out)
        return dataset

    def train(self):
        self.load_trainer()
        self.trainer.train()
        self.trainer.evaluate()
        self.trainer.save_model(self.model_name.split('/')[-1] + "-fine-tuned")

    def predict(self):
        self.load_trainer(test_param=True)
        self.trainer.predict()

    def objective(self, trial):
        self.load_trainer(trial) 
        self.trainer.train()
        result = self.trainer.evaluate()
        return (result['eval_exact_match'] + result['eval_f1'])/2
    
    def optuna(self):
        study = optuna.create_study(study_name="hyper-parameter-search",
                                    direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed = 42))
        
        study.optimize(self.objective, n_trials = 12)
        
        optuna.visualization.plot_contour(study).write_image('files/charts/optuna/'+self.model_name + '_plot_contour.png')
        optuna.visualization.plot_optimization_history(study).write_image('files/charts/optuna/'+self.model_name + '_plot_optimization_history.png')
        optuna.visualization.plot_param_importances(study).write_image('files/charts/optuna/'+self.model_name + '_plot_param_importances.png')
    
        for key, value in study.best_trial.params.items():
            print("  {}: {}".format(key, value))

    def prepare_train_features(self, examples: dict) -> transformers.tokenization_utils_base.BatchEncoding:
        inputs = [
            " ".join([
                "question:", question.strip(), 
                "context:", context.strip()
            ]) for question, context in zip(examples['question'], examples['context'])
        ]
        
        answers = [
            answer["text"][0] for answer in examples['answers']
        ]

        tokenized_inputs = self.tokenizer(
            inputs, 
            max_length=512, 
            padding="max_length", 
            truncation=True
        )

        tokenized_labels = self.tokenizer(
            text_target=answers, 
            max_length=32, 
            padding="max_length", 
            truncation=True
        )

        tokenized_labels["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) 
             for l in label] 
             for label in tokenized_labels["input_ids"]
        ]

        tokenized_inputs["labels"] = tokenized_labels["input_ids"]
        return tokenized_inputs

    def prepare_validation_features(self, examples: dict) -> transformers.tokenization_utils_base.BatchEncoding:
        inputs = [
            " ".join([
                "question:", question.strip(), 
                "context:", context.strip()
            ]) for question, context in zip(examples['question'], examples['context'])
        ]
        
        answers = [
            answer["text"][0] for answer in examples['answers']
        ]

        tokenized_inputs = self.tokenizer(
            inputs, 
            max_length=512, 
            padding="max_length", 
            truncation=True,
            return_overflowing_tokens=True
        )

        tokenized_labels = self.tokenizer(
            text_target=answers, 
            max_length=32, 
            padding="max_length", 
            truncation=True
        )

        tokenized_labels["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) 
             for l in label] 
             for label in tokenized_labels["input_ids"]
        ]

        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        tokenized_inputs["example_id"] = []
        labels_out = []
        for i in range(len(tokenized_inputs["input_ids"])):
            tokenized_inputs["example_id"].append(examples["id"][sample_mapping[i]])
            labels_out.append(tokenized_labels["input_ids"][sample_mapping[i]])

        tokenized_inputs["labels"] = labels_out
        return tokenized_inputs

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_processing_function(self, examples: datasets.Dataset, features: datasets.Dataset, preds):
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}

        predictions = {}
        for example_index, example in enumerate(examples):
            predictions[example["id"]] = decoded_preds[feature_per_example[example_index]]

        formatted_predictions = [
            {
                "id": str(k), 
                "prediction_text": v, 
                "label_id": examples['label_id'][int(k)]
            } 
            for k, v in predictions.items()
        ]
        
        references = [
            {
                "id": str(ex["id"]), 
                "answers": ex['answers'], 
                "label_id": ex['label_id']
            } 
            for ex in examples
        ]
        return transformers.trainer.EvalPrediction(predictions=formatted_predictions, label_ids=references)