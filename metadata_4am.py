import pandas as pd
import numpy as np
# from memory_profiler import profile


# @profile
def merge_metadata_vector_to_parquet(metadata_path, vector_path, dest_parquet_path, pk_index):
    file_name = metadata_path.split("/")[-1]
    print(f"start process file  {file_name}")
    # read metadata
    print("read metadata parquet file")
    metadata_df = pd.read_parquet(metadata_path, engine='pyarrow')
    df_len = len(metadata_df)
    print(f"metadata num: {df_len}")
    print("metadata columns:")
    print([column for column in metadata_df])

    # drop some columns
    print("start drop some columns")
    reserve_columns = ['caption', 'NSFW', 'similarity', 'width', 'height', 'original_width', 'original_height', 'md5']
    for column in metadata_df:
        if column not in reserve_columns:
            metadata_df.drop(column, axis=1, inplace=True)
    print("finish drop columns, and columns is:")
    print([column for column in metadata_df])

    # add pk column
    print("start add pk column")
    int_values = pd.Series(data=[i for i in range(pk_index, pk_index + df_len)])
    metadata_df["pk"] = int_values
    print("finish add pk column, and columns is:")
    print([column for column in metadata_df])

    # add vector column
    print("start add vector column")
    vec = np.load(vector_path)
    vec.astype(dtype=np.float32)
    print(vec.shape)
    metadata_df["float32_vector"] = vec.tolist()
    del vec
    print("finish add vector column, and columns is:")
    print([column for column in metadata_df])

    # save dataframe to parquet file
    print(f"start save parquet {file_name}")
    metadata_df.to_parquet(dest_parquet_path)
    print(f"finish process file  {file_name}")
    del metadata_df
    return df_len


def gen_url(data_type: str, source_index=0):
    base_url = '/data/laion1B-nolang/'
    dest_url = '/test/milvus/raw_data/laion5b_parquet'

    if data_type == "metadata":
        url = f'{base_url}laion1B-nolang-metadata/metadata_{source_index:04d}.parquet'
    elif data_type == "vector":
        url = f'{base_url}img_emb/img_emb_{source_index:04d}.npy'
    elif data_type == "dest":
        url = f'{dest_url}/binary_768d_{source_index:05d}.parquet'
    return url


if __name__ == '__main__':
    pk_index = 5808222
    for i in range(6, 51):
        metadata_path = gen_url(data_type="metadata", source_index=i)
        vector_path = gen_url(data_type="vector", source_index=i)
        dest_path = gen_url(data_type="dest", source_index=i)
        single_df_len = merge_metadata_vector_to_parquet(metadata_path, vector_path, dest_path, pk_index)
        pk_index += single_df_len
        print(f"processed entities: {pk_index}")
    print(f"total processed entities: {pk_index}")
