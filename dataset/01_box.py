from pathlib import Path
from os import listdir


def after_8(line):
    comma_indices = [i for i, char in enumerate(line) if char == ',']

    index_after_8th_comma = comma_indices[7] + 1 
    return line[index_after_8th_comma:]
    

def ocr(path):
    Path(path.replace('box', 'texts_box')).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(listdir(path)):
        if i%100 == 0:
            print(i)

        with open(path+f, encoding='utf-8') as file:
            lines = file.readlines()

        text = ''
        for i, l in enumerate(lines):
            text += after_8(l) 

        with open(path.replace('box', 'texts_box') + f.replace('.txt', '_text.txt'), "w") as text_file:
            text_file.write(text)


if __name__ == "__main__":
    ocr('SROIE2019/train/box/')
    ocr('SROIE2019/test/box/')
    ocr('SROIE2019/validation/box/')


