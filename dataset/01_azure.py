from pathlib import Path
from os import listdir
from PIL import Image
import io
import requests 
import time
import PIL


key = ''
url = 'https://sroieocr.cognitiveservices.azure.com/vision/v3.1/ocr'


def run_azure(image):
    time.sleep(4)
    headers = {
        'Content-Type': 'application/octet-stream', 
        'Ocp-Apim-Subscription-Key': key
    }
    
    params = {
        'language': 'en', 
        'detectOrientation': 'true'#params = {'language': 'unk', 'detectOrientation': 'true'}
    }
    b = io.BytesIO()

    image.save(b, 'jpeg')
    b.seek(0)

    try:
        json_request = requests.post(url, 
                                     data=b, 
                                     headers=headers,
                                     params=params).json()
    except:
        print("OCR Azure Error")

    final = ''
    for box in json_request['regions']:
        for line in box['lines']:
            for words in line['words']:
                final += words['text'] + ' '

            final += '\n'
    return final 


def resize_to_azure_size(imageFile):
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    max_size = 4200, 4200
    
    imageFile.thumbnail(max_size, Image.Resampling.LANCZOS)
    return imageFile


def ocr(path):
    Path(path.replace('img', 'texts_azure')).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(listdir(path)):
        print(i, f)
        image = Image.open(path + f)
        image = resize_to_azure_size(image)

        text = run_azure(image)
        with open(path.replace('img', 'texts_azure_temp') + f.replace('.jpg', '_text.txt'), "w") as text_file:
            text_file.write(text)


if __name__ == "__main__":
    ocr('SROIE2019/train/img/')
    ocr('SROIE2019/test/img/')
    ocr('SROIE2019/validation/img/')

