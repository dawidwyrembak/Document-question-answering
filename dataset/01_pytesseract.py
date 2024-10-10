from pathlib import Path
import pytesseract
from os import listdir
from PIL import Image
from _helpers import sharpened_image, preprocess_image


def run_tesseract(image):
    """
    for k in range(1, 4):
        for j in range(1, 12):
            print(str(k) + " " +  str(j) + '-----------------'*10)
            try:
                text = pytesseract.image_to_string(image, config=r'-c preserve_interword_spaces=1 -l "pol+eng" --psm ' + str(j) + ' --oem ' + str(k))
            except Exception as e:
                print(e)
    """
    text = pytesseract.image_to_string(image, config=r'-c preserve_interword_spaces=1 -l "eng" --psm 4 --oem 1')
    return text 


def ocr(path):
    Path(path.replace('img', 'texts_pytesseract')).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(listdir(path)):
        if i%100 == 0:
            print(i)

        image = Image.open(path + f)

        text = run_tesseract(image)
        with open(path.replace('img', 'texts_pytesseract') + f.replace('.jpg', '_text.txt'), "w") as text_file:
            text_file.write(text)


if __name__ == "__main__":
    ocr('SROIE2019/train/img/')
    ocr('SROIE2019/test/img/')
    ocr('SROIE2019/validation/img/')

