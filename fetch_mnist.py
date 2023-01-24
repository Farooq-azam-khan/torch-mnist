import requests 
import zipfile
import io  
import pathlib 
import gzip 
import shutil


unsigned_byte = 1 
int32_bytes = 4 
base_url = 'http://yann.lecun.com/exdb/mnist'

def download_zip_file(file_name, web_file_path, data_dir):
    print(f'Making Request to {web_file_path}')
    r = requests.get(web_file_path, stream=True)
    print(f'Status Code: {r.status_code}')
    if r.status_code >= 200 and r.status_code < 400:
        with open(str(data_dir/file_name), 'wb') as f:
            print(f'Saving File={file_name}')
            for chunk in r.raw.stream(1024, decode_content=False):
                if chunk: 
                    f.write(chunk)
            print('Saved file')

def get_unzipped_file_path(file_name, data_dir): 
    file_name_root = file_name.split('.')[0]
    unzipped_file_path = data_dir/file_name_root
    return unzipped_file_path

def unzip_file(file_name, local_file_path, data_dir):
    unzipped_file_path = get_unzipped_file_path(file_name, data_dir)
    with gzip.open(str(local_file_path), 'rb') as f_in: 
        print('Opend zipped file')
        with open(str(unzipped_file_path), 'wb') as f_out:
            print('unzipping with shutil')
            shutil.copyfileobj(f_in, f_out)

import numpy as np 
#import matplotlib.pyplot as plt 
from tqdm import trange
def read_image_and_save_as_numpy_file(file_name, data_dir):
    print('Reading image byte data')
    file_name_root = file_name.split('.')[0]
    unzipped_file_path = get_unzipped_file_path(file_name, data_dir)
    
    with open(str(unzipped_file_path), 'rb') as f: 
        magic_number = int.from_bytes(f.read(int32_bytes), byteorder='big')
        number_of_images = int.from_bytes(f.read(int32_bytes), byteorder='big')
        number_of_rows = int.from_bytes(f.read(int32_bytes), byteorder='big')
        number_of_columns = int.from_bytes(f.read(int32_bytes), byteorder='big')


        dataset = []
        for _ in trange(number_of_images):
            image = np.zeros(number_of_rows* number_of_columns)
            for i in range(number_of_rows*number_of_columns):
                pixel = int.from_bytes(f.read(unsigned_byte), byteorder='big')
                image[i] = pixel 
            image = image.reshape((number_of_rows, number_of_columns))
            dataset.append(image)
        dataset = np.array(dataset)
        print(dataset.shape)
        np.save(data_dir / f'{file_name_root}.npy' , dataset)

def read_labels_and_save_as_numpy_file(file_name, data_dir):
    print('Reading label byte data')
    file_name_root = file_name.split('.')[0]
    unzipped_file_path = get_unzipped_file_path(file_name, data_dir)

    with open(str(unzipped_file_path), 'rb') as f: 
        magic_number = int.from_bytes(f.read(int32_bytes), byteorder='big')
        number_of_items = int.from_bytes(f.read(int32_bytes), byteorder='big')
        labels = np.zeros(number_of_items) 
        for idx in trange(number_of_items): 
            label = int.from_bytes(f.read(unsigned_byte), byteorder='big')
            labels[idx] = label 
        print(labels.shape)
        np.save(data_dir / f'{file_name_root}.npy', labels)


 
def main(file_name, is_image_data): 

    web_file_path = base_url + '/' + file_name 
    data_dir = pathlib.Path('./data')
    data_dir.mkdir(parents=True, exist_ok=True)
    local_file_path = pathlib.Path('./data') / file_name 

    download_zip_file(file_name, web_file_path, data_dir)
    unzip_file(file_name, local_file_path, data_dir)
    if is_image_data: 
        read_image_and_save_as_numpy_file(file_name, data_dir)
    else: 
        read_labels_and_save_as_numpy_file(file_name, data_dir)

if __name__ == '__main__': 
    file_names_and_meta_data = [
            ('train-images-idx3-ubyte.gz', True), 
            ('train-labels-idx1-ubyte.gz', False),
            ('t10k-images-idx3-ubyte.gz', True),
            ('t10k-labels-idx1-ubyte.gz', False)
    ]

    for file_name, is_image_data in file_names_and_meta_data:
        print('Working on', file_name)
        main(file_name, is_image_data)
   
