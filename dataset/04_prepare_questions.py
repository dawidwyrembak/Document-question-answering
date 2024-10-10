import pandas as pd
import json
import random


def prepare_question(label):
    if label == 'no':
        question = random.choice([
                "Could you please provide the invoice number?",
                "What's the specific number on the receipt?",
                "What is the receipt number?",
                "May I know the document number?",
                "Could you tell me the receipt number, please?",
                "What's the number associated with the invoice?",
                "Could you share the receipt number with me?",
                "What's the unique identifier for the document?",
                "Could you inform me about the invoice number?",
                "Do you know the document number?"
            ])
 
    elif label =='company':
        question = random.choice([
                "What is the name of the company?",
                "Could you provide the name of the organization mentioned in receipt?",
                "Can you tell me the name of the institution from document?",
                "Do you know the name of the firm?",
                "Please specify the name of the company.",
                "I need to know the name of the company from receipt.",
                "What's the company name in document?",
                "Could you give me the company name?",
                "Would you mind sharing the company name?",
                "May I ask for the name of the company which is in document from?"
            ])
    elif label =='address':
        question = random.choice([
                "Where does the company reside?",
                "What is the origin of the company?",
                "What is the location in receipt?",
                "In which area does the company originate?",
                "From whence does the creditor come?",
                "What is the company address?",
                "What is the origin mentioned in document?",
                "Give me address from receipt.",
                "Where was this invoice created?",
                "What's the address in receipt text?"
            ])
    elif label == 'date':
        question = random.choice([
                "When was the document dated?",
                "What is the date of the document?",
                "Can you provide the date of the document?",
                "When was this receipt issued?",
                "What is the date of the invoice?",
                "Could you tell me the date this document was created?",
                "What's the date mentioned in the document?",
                "When was this receipt drafted?",
                "Could you specify the date on the invoice?",
                "What date is indicated on the receipt?"
            ])
    elif label == 'total':
        question = random.choice([
                "What is the gross amount?",
                "Define gross amount.",
                "What constitutes the total amount?",
                "Could you provide insight into the gross amount?",
                "Give me total amount of receipt.",
                "What is invoice total?",
                "What is the total price in receipt?",
                "How much costs all?",
                "What is total amount in receipt?",
                "How much is the cost on the receipt?"
            ])
    else:
        print(label)

    return question


def get_keywords(text, labels):
    return [(text[l[0]:l[1]], l[2]) for l in labels]


def read_json(filename):
    texts = []
    data = []

    with open(filename, "r") as read_file:
        for line in read_file:
            data.append(json.loads(line))

    for d in data:
        texts.append(
            {
                'id': d['file'],
                'context': d['text'],
                'question': '',
                'answers': {},
                'labels': d['labels']
            })
        
    return texts 


if __name__ == '__main__':
    label2id = {
        'no': 1,
        'address': 2,
        'total': 3,
        'date': 4,
        'company': 5
    }

    for s in ['train', 'validation', 'test']:
        texts = read_json('ents_' + s + '.jsonl') 
        print(s + ": " + str(len(texts)))

        df = pd.DataFrame(columns=['context', 'question', 'answer_text', 'answer_start', 'label_id'])
        for i, t in enumerate(texts):
            keywords = get_keywords(t['context'],  t['labels'])

            for answer, label in keywords:
                ques = prepare_question(label)

                if len(answer) == 0:
                    print(i, ques, t['labels'])

                row = [
                    t['context'],  
                    ques, 
                    answer, 
                    t['context'].lower().find(answer.lower()), 
                    label2id[label]
                ]

                df.loc[len(df)] = row

        df.to_csv('../model/files/out_'+s+'.csv')
