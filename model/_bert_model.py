import torch
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
import evaluate
import pandas as pd
import collections
import numpy as np
import transformers
from _qa_trainer import QuestionAnsweringTrainer
import optuna
from typing import Optional, Tuple
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Model:
    def __init__(self, model_name: str) -> None:
        set_seed(42)
        self.model_name = model_name
        self.device = self._set_device()
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
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
        training_args = TrainingArguments(
            output_dir='./_temp',
            #bert  2.0480725405024982e-05
            #roberta 6.798962421591133e-05
            learning_rate=6.798962421591133e-05 if not trial else trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            #bert 0.0024600212078518236
            #roberta 0.0016305687346221474
            weight_decay=0.0016305687346221474 if not trial else trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True),
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            save_strategy='epoch',
            evaluation_strategy = "epoch",
            logging_strategy="epoch",
            logging_steps=1,
            label_names=['start_positions', 'end_positions']
        )

        self.trainer = QuestionAnsweringTrainer(
            model=self.model if not trial else None ,
            model_init=None if not trial else self.get_model ,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset if not test_param  else self.test_dataset,
            eval_examples=self.dataset['validation'] if not test_param else self.dataset['test'],
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            data_collator=transformers.DefaultDataCollator(),
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
        tokenized_examples = self.tokenizer(
            [q.strip() for q in examples["question"]],
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=64,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = tokenized_examples.pop("offset_mapping")
        sample_map = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offset in enumerate(offset_mapping):
            answer_i = examples["answers"][sample_map[i]]
            answer_char_start = answer_i["answer_start"][0]
            answer_end_char = answer_i["answer_start"][0] + len(answer_i["text"][0])
            sequence_ids = tokenized_examples.sequence_ids(i)

            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
                
            context_end = context_start
            while sequence_ids[context_end] == 1:
                context_end += 1
            context_end = context_end - 1

            if offset[context_start][0] > answer_char_start or offset[context_end][1] < answer_end_char:
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                answer_start = context_start
                while answer_start <= context_end and offset[answer_start][0] <= answer_char_start:
                    answer_start += 1
                tokenized_examples["start_positions"].append(answer_start - 1)

                answer_end = context_end
                while answer_end >= context_start and offset[answer_end][1] >= answer_end_char:
                    answer_end -= 1
                tokenized_examples["end_positions"].append(answer_end + 1)

        return tokenized_examples


    def prepare_validation_features(self, examples: dict) -> transformers.tokenization_utils_base.BatchEncoding:
        tokenized_examples = self.tokenizer(
            [q.strip() for q in examples["question"]],
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=64,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            tokenized_examples["example_id"].append(examples["id"][sample_mapping[i]])

            offset_map_i = tokenized_examples["offset_mapping"][i]
            tokenized_examples["offset_mapping"][i] = [
                o if tokenized_examples.sequence_ids(i)[k] == 1 
                else None 
                for k, o in enumerate(offset_map_i)
            ]

        return tokenized_examples

    def postprocess_qa_predictions(
        self,
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        n_best_size: int = 10
    ):
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            return

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)

        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        all_predictions = collections.OrderedDict()
        for example_index, example in enumerate(examples):
            feature_indices = features_per_example[example_index]

            prelim_predictions = []
            for feature_index in feature_indices:
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                offset_mapping = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                            continue

                        if end_index < start_index:
                            continue

                        prelim_predictions.append(
                            {
                                "text": example['context'][offset_mapping[start_index][0] : offset_mapping[end_index][1]],
                                "score": start_logits[start_index] + end_logits[end_index]
                            }
                        )

            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            all_predictions[example["id"]] = predictions[0]["text"]
        return all_predictions

    def post_processing_function(self, examples, features, predictions):
            predictions = self.postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
            )
            formatted_predictions = [
                {
                    "id": str(k), 
                    "prediction_text": v, 
                    "label_id": examples['label_id'][int(k)]
                } for k, v in predictions.items()
            ]
            
            references = [
                {
                    "id": str(ex["id"]), 
                    "answers": ex['answers'], 
                    "label_id": ex['label_id']
                } for ex in examples
            ]
            return transformers.trainer.EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(self, pred: transformers.trainer.EvalPrediction) -> dict:
        return self.metric.compute(predictions=pred.predictions, references=pred.label_ids)