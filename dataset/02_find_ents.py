from os import listdir
import json
import re


def read_text(file):
    text = ''
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            text += line
    return text.replace("—", "-")


def remove_chars_except_digits_and_punctuation(input_string):
    return re.sub(r'[^\d.,]', '', input_string)


def find_with_spaces(k, pattern, text):
    pattern = pattern.replace(' ', '')
    pattern_re = re.compile(' *'.join(map(re.escape, pattern)))

    m = pattern_re.search(text)

    if m:
        return [[m.start(), m.end(), k]]
    else:
        return []
    

def find_in_text(k, v, text):
    r  = re.compile(re.escape(v))
    finds = [[m.start(), m.end(), k] 
             for m in re.finditer(r, text)]

    if finds == []:
        finds = find_with_spaces(k, v, text)
    return finds 


def find_ents(text, data):
    ents = []

    for k, v in data.items():
        finds = []
        if k not in ['address', 'no', 'date', 'total', 'company']:
            continue 

        if k == 'total':
            finds = find_in_text(k, remove_chars_except_digits_and_punctuation(v), text)
        elif k == 'no':
            finds = find_in_text(k, v.lower(), text.lower())
        elif k == 'date':
            finds = find_in_text(k, v.lower(), text.lower())
        elif k == 'address':
            finds = find_in_text(k, v.lower().replace(',', ' ').replace('.', ' '), 
                                 text.lower().replace('\n', ' ').replace(',', ' ').replace('.', ' '))
        elif k == 'company':
            finds = find_in_text(k, v.lower(), text.lower().replace('\n', ' '))

        if len(finds):
            ents.extend([finds[0]])

    return ents


def main(path):
    all_ents = []
    subtotal = 0
    total = 0

    for filename in listdir(path):
        print("Now work o: ", filename)

        out_dict = {
            'file': filename, 
            'text': read_text(path + filename),
            'labels': []
        }

        with open(path.replace(path.split('/')[2], 'entities') + filename.replace('_text.txt', '_1.txt')) as f:
            data = json.load(f)

        out_dict['labels'] = find_ents(out_dict['text'], data)
        out_dict['labels'].sort(key=lambda x: x[0])

        subtotal += len(out_dict['labels'])
        total += 5
        all_ents.append(out_dict)

    with open('ents_' + path.split('/')[1] + '.jsonl', 'w') as outfile:
        for out in all_ents:
            json.dump(out, outfile)
            outfile.write('\n')

    return subtotal, total


if __name__ == "__main__":
    s1, t1 = main('SROIE2019/train/texts_paddleocr/')
    s2, t2 = main('SROIE2019/test/texts_paddleocr/')
    s3, t3 = main('SROIE2019/validation/texts_paddleocr/')

    print("\n\nZnaleziono: ", s1+s2+s3 ,"\nMożliwe: ", t1+t2+t3)
    print("{:.2f}".format((s1+s2+s3)/(t1+t2+t3)))