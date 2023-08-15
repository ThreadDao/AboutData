import os

import numpy as np
import requests
from tqdm import tqdm
from pyquery import PyQuery as pq

file_index = 2


def downloadFILE(url, name):
    print(url)
    print(name)
    # resp = requests.get(url=url, stream=True)
    # content_size = int(resp.headers['Content-Length']) / 1024
    # with open(name, "wb") as f:
    #     print("content size:", content_size, 'k, start download...')
    #     for data in tqdm(iterable=resp.iter_content(chunk_size=1024), total=content_size, unit='k', desc=name):
    #         f.write(data)
    #     print(name + "download finished")


def scan_files(dst_url):
    resp = requests.get(url=dst_url)
    img_emb_a_tags = pq(resp.content).find('a[href^=img_emb]')
    all_file_urls = []
    for img_emb_a in img_emb_a_tags:
        all_file_urls.append(dst_url + '/' + pq(img_emb_a).attr('href'))
    return all_file_urls


def process_file(url):
    name = url.split('/')[-1]
    tmp_file = f'/tmp/laion5b/{name}'
    downloadFILE(url, tmp_file)
    # npy = np.load(tmp_file)
    # print(f"npy {tmp_file} shape: {npy.shape}")
    # npy.astype(dtype=np.float32)
    global file_index
    npy_file = f'/test/milvus/raw_data/laion5b/binary_768d_{file_index:05d}.npy'
    # npy.save(npy_file)
    print(f"npy {npy_file} save finished")
    file_index += 1
    # os.remove(tmp_file)
    print(f"tmp file {tmp_file} remove finished")


def scan_dir(url):
    dst_url = url + '/img_emb'
    all_file_urls = scan_files(dst_url)
    exclude_file_urls = [
        'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/img_emb/img_emb_0000.npy',
        'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/img_emb/img_emb_0001.npy'
    ]
    for file_url in all_file_urls:
        if file_url not in exclude_file_urls:
            process_file(file_url)
        else:
            print(f"skip file {file_url}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # nas_raw_data_dir = '/Users/nausicca/test/milvus/raw_data/laion5b'
    scan_dir('https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang')
    # scan_dir('https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en')
    # scan_dir('https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-multi')