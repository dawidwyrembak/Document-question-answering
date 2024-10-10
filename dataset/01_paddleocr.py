from pathlib import Path
from os import listdir
from paddleocr import PaddleOCR
from _helpers import preprocess_image, sharpened_image


def run_paddleocr(image_path, ocr):
    result = ocr.ocr(image_path, cls=True)
    text = ''
    for idx in result:
        text += idx[1][0] + '\n'
    return text 


def ocr(path):
    Path(path.replace('img', 'texts_paddleocr')).mkdir(parents=True, exist_ok=True)
    ocr = PaddleOCR(use_angle_cls=True, use_space_char=True, use_dilitation=True, lang='en', use_gpu=True)
    for i, f in enumerate(listdir(path)):
        print(i)
       # image = Image.open(path + f)
       # image =  preprocess_image(sharpened_image(image))
       # image.save(path+f.replace('.jpg', '_pre.jpg'))
        text = run_paddleocr(path+f, ocr)
        with open(path.replace('img', 'texts_paddleocr') + f.replace('.jpg', '_text.txt'), "w") as text_file:
            text_file.write(text)


if __name__ == "__main__":
    ocr('SROIE2019/train/img/')
    ocr('SROIE2019/test/img/')
    ocr('SROIE2019/validation/img/')
