import math
import time
import json
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput, speed_metrics, EvalPrediction
from datetime import datetime
from pathlib import Path
###
# CODE BASED ON TRANSFORMERS DOCUMENTATION
###

# Mapping labels for entities
id2label = {
    '1': 'no',
    '2': 'address',
    '3': 'total',
    '4': 'date',
    '5': 'company'
}

def print_table(data):
    col_width = [max(len(str(x)) for x in col) for col in zip(*data)]
    for row in data:
        print(" | ".join("{:{}}".format(item, width) for item, width in zip(row, col_width)))
        

class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, model_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.model_name = model_name
        self.setup_logging()

    def setup_logging(self):
            Path('files').mkdir(parents=True, exist_ok=True)
            Path('files/train_metrics/').mkdir(parents=True, exist_ok=True)
            with open(f'files/train_metrics/{self.model_name}.txt', 'w') as f:
                f.write(f"START: {datetime.now()}\n")
                f.write(f"{self.model_name}\n\n")
                
    def evaluate(self, ignore_keys: bool = None, metric_key_prefix: str = "eval"):
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        start_time = time.time()
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                metric_key_prefix=metric_key_prefix,
                ignore_keys=ignore_keys
            )
        finally:
            self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
            eval_preds = self.post_process_function(self.eval_examples, self.eval_dataset, output.predictions)

            for p in eval_preds[0]:
                p.pop('label_id')
            for p in eval_preds[1]:
                p.pop('label_id')

            metrics = self.compute_metrics(eval_preds)

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            metrics.update(output.metrics)
        else:
            metrics = output.metrics

        if self.args.should_log:
            self.log(metrics)
            with open('files/train_metrics/' + self.model_name + '.txt', 'a') as f:
                f.write(f"LOG: {datetime.now()}\n")
                f.write(json.dumps(metrics) + '\n')

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, metric_key_prefix: str = "pred"):
        predict_dataloader = self.get_test_dataloader(self.eval_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        start_time = time.time()
        try:
            output = self.evaluation_loop(
                predict_dataloader,
                description="Prediction",
                metric_key_prefix=metric_key_prefix
            )
        finally:
            self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        predictions = self.post_process_function(self.eval_examples , self.eval_dataset , output.predictions)
        _metrics_per_entity = [['Label', 'Exact match', "F1"]]

        for k in id2label.keys():
            _predictions = []
            for p in predictions[0]:
                if 'label_id' not in p.keys():
                    continue
                if p['label_id'] == k:
                    p.pop('label_id')
                    _predictions.append(p)

            _references = []
            for p in predictions[1]:
                if 'label_id' not in p.keys():
                    continue
                if p['label_id'] == k:
                    p.pop('label_id')
                    _references.append(p)

            _metrics = self.compute_metrics(EvalPrediction(predictions=_predictions, label_ids=_references))
            _metrics_per_entity.append([
                id2label[str(k)], 
                str(format(_metrics['exact_match'], ".2f")), 
                str(format(_metrics['f1'], ".2f"))
            ])
            
        metrics = self.compute_metrics(predictions)
        _metrics_per_entity.append([
            "TOTAL", 
            str(format(metrics['exact_match'], ".2f")), 
            str(format(metrics['f1'], ".2f")) 
        ])
        print_table(_metrics_per_entity)

        #metrics = self.compute_metrics(predictions)
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        metrics.update(output.metrics)
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
    

class QuestionAnsweringSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, model_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.model_name = model_name
        self.setup_logging()

    def setup_logging(self):
            Path('files').mkdir(parents=True, exist_ok=True)
            Path('files/train_metrics/').mkdir(parents=True, exist_ok=True)
            with open(f'files/train_metrics/{self.model_name}.txt', 'w') as f:
                f.write(f"START: {datetime.now()}\n")
                f.write(f"{self.model_name}\n\n")
                
    def evaluate(self, ignore_keys: bool = None, metric_key_prefix: str = "eval"):
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        start_time = time.time()
        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                metric_key_prefix=metric_key_prefix,
                ignore_keys=ignore_keys
            )
        finally:
            self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        if self.post_process_function is not None and self.compute_metrics is not None and self.args.should_save:
            eval_preds = self.post_process_function(self.eval_examples, self.eval_dataset, output.predictions)

            for p in eval_preds[0]:
                p.pop('label_id')
            for p in eval_preds[1]:
                p.pop('label_id')

            metrics = self.compute_metrics(eval_preds)

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
            metrics.update(output.metrics)
        else:
            metrics = output.metrics

        if self.args.should_log:
            self.log(metrics)
            with open('files/train_metrics/' + self.model_name + '.txt', 'a') as f:
                f.write(f"LOG: {datetime.now()}\n")
                f.write(json.dumps(metrics) + '\n')

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, metric_key_prefix: str = "pred"):
        predict_dataloader = self.get_test_dataloader(self.eval_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        start_time = time.time()
        try:
            output = self.evaluation_loop(
                predict_dataloader,
                description="Prediction",
                metric_key_prefix=metric_key_prefix
            )
        finally:
            self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        predictions = self.post_process_function(self.eval_examples , self.eval_dataset , output.predictions)

        _metrics_per_entity = [['Label', 'Exact match', "F1"]]

        for k in id2label.keys():
            _predictions = []
            for p in predictions[0]:
                if 'label_id' not in p.keys():
                    continue
                if p['label_id'] == k:
                    p.pop('label_id')
                    _predictions.append(p)

            _references = []
            for p in predictions[1]:
                if 'label_id' not in p.keys():
                    continue
                if p['label_id'] == k:
                    p.pop('label_id')
                    _references.append(p)

            _metrics = self.compute_metrics(EvalPrediction(predictions=_predictions, label_ids=_references))
            _metrics_per_entity.append([
                id2label[str(k)], 
                str(format(_metrics['exact_match'], ".2f")), 
                str(format(_metrics['f1'], ".2f"))
            ])
            
        metrics = self.compute_metrics(predictions)
        _metrics_per_entity.append([
            "TOTAL",
            str(format(metrics['exact_match'], ".2f")), 
            str(format(metrics['f1'], ".2f")) 
        ])
        print_table(_metrics_per_entity)

        #metrics = self.compute_metrics(predictions)
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        metrics.update(output.metrics)
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
    
