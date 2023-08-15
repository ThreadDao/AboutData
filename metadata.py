import pandas as pd
import numpy as np


def merge_metadata_vector_to_parquet(metadata_path, vector_path, dest_parquet_path):
    # read metadata
    print("read metadata parquet file")
    metadata_df = pd.read_parquet(metadata_path, engine='pyarrow')
    print(len(metadata_df))
    print([column for column in metadata_df])

    # drop some columns
    reserve_columns = ['caption', 'NSFW', 'similarity', 'width', 'height', 'original_width', 'original_height', 'md5']
    for column in metadata_df:
        if column not in reserve_columns:
            metadata_df.drop(column, axis=1, inplace=True)
    print([column for column in metadata_df])

    # add pk column
    int_values = pd.Series(data=[i for i in range(0, 0 + len(metadata_df))])
    metadata_df["pk"] = int_values
    print([column for column in metadata_df])

    # add vector column
    vec = np.load(vector_path)
    vec.astype(dtype=np.float32)
    print(vec.shape)
    metadata_df["float32_vector"] = vec.tolist()
    print(len(metadata_df))
    print([column for column in metadata_df])
    print(metadata_df.iloc[0])

    # save dataframe to parquet file
    metadata_df.to_parquet(dest_parquet_path)

def gen_file_name():


if __name__ == '__main__':
    metadata_path = "/home/zong/Downloads/laion1B-nolang/metadata/metadata_0000.parquet"
    vector_path = "/home/zong/Downloads/laion1B-nolang/img_emb/img_emb_0000.npy"
    dest_parquet_path = '/tmp/metadata_0000.parquet'
    merge_metadata_vector_to_parquet(metadata_path, vector_path, dest_parquet_path)
