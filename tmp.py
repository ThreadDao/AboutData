import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from pymilvus import connections, Collection, utility

def gen_links(data_type: str, index=0):
    base_url = 'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/'
    if data_type == "metadata":
        url = f'{base_url}laion1B-nolang-metadata/metadata_{index:04d}.parquet'
    elif data_type == "vector":
        "https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/img_emb/"
        url = f'{base_url}img_emb/img_emb_{index:04d}.npy'
    return url


def write_links(end: int, data_type: str, start=0):
    with open('/home/zong/PycharmProjects/AboutData/metadata.txt', 'a') as f:
        for i in range(start, end):
            file_url = gen_links(data_type, i)
            f.writelines(file_url + "\n")


if __name__ == '__main__':
    end = 10
    # base_url = 'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/laion1B-nolang-metadata/'
    # with open('/home/zong/PycharmProjects/AboutData/metadata.txt', 'a') as f:
    #     for i in range(end):
    #         file_url = f'{base_url}metadata_{i:04d}.parquet'
    #         line = f.writelines(file_url+"\n")
    write_links(10, data_type="vector")
