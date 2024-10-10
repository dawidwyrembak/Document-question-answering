import pandas as pd
import collections
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    QuestionAnsweringPipeline
)


def load_dataset(filename: str):
    df = pd.read_csv(filename)
    out = collections.defaultdict(list)
    
    for index, row in df.iterrows():
        out['id'].append(str(index))
        out['context'].append(row['context'])
        out['question'].append(row['question'])
        out['answers'].append({
            'text': [row['answer_text']],
            'answer_start': [row['answer_start']]
        })
    
    dataset = Dataset.from_dict(out)
    return dataset


def inference():
    model_path = "bert-base-uncased-fine-tuned"
    question = "What is the invoice number?"
    example_index = 30

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa_pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    dataset = load_dataset('files/out_test.csv')
    context = dataset[example_index]['context']

    print('-' * 100)
    print(context)
    print('-' * 100)

    outputs = qa_pipeline(question=question, context=context, top_k=3, max_seq_len=512)
    print(outputs)


if __name__ == "__main__":
    inference()