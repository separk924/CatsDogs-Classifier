import os
import shutil

def dist():
    images = os.listdir('./cats-and-dogs/')
    dogs = list(filter(lambda image: 'd' in image, images))
    cats = list(filter(lambda image: 'c' in image,images))
    
    shutil.rmtree('./data')
    
    os.makedirs('./data/dogs', exist_ok=True)
    os.makedirs('./data/cats', exist_ok=True)
    
    copyImages(dogs, './data/dogs')
    copyImages(cats, './data/cats')
    
def copyImages(images_to_copy, destination):
    for image in images_to_copy:
        shutil.copyfile(f'./cats-and-dogs/{image}', f'{destination}/{image}')
        
dist()