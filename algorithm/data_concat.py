import os

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

data1 = []
data2 = []

ds_path = r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/'

def data_concat(data1, data2):
    concat_data = []
    for i in range(len(data1)):
        x = data1[i]
        for j in range(len(data2)):
            y = data2[j]
            if x[0] == y[0]:
                concat_data.append((x[0], x[1], x[2], y[1], y[2]))
            else:
                continue
    return concat_data

with open(os.path.join(ds_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
    for line in f.readlines()[1:]:
        values = line.strip().split('\t')
        user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
        data1.append((user_id, artist_id, weight_id))

with open(os.path.join(ds_path, 'lastfm-2k/user_friends.dat'), 'r') as f:
    for line in f.readlines()[1:]:
        values = line.strip().split('\t')
        user_id, artist_id = int(values[0]), int(values[1])
        weight_id = 1
        data2.append((user_id, artist_id, weight_id))

data1 = pd.DataFrame(data1)
data2 = pd.DataFrame(data2)

data1.columns = ['用户i', '用户j', '评分a']
data2.columns = ['用户i', '用户j', '权重a']

sum_data = pd.merge(data1, data2, left_on='用户i', right_on='用户i')

# print(data1.head())
# print(data2.head())
# print(sum_data.head())

# 写入
pq.write_table(pa.Table.from_pandas(sum_data), './sum_data.parquet')
# 读取
df = pq.read_table('./sum_data.parquet').to_pandas()
print(df)

df = df.apply(lambda x: tuple(x), axis=1).values.tolist()
print(df)

def _read_lastfm_data():
    data1 = []
    data2 = []
    with open(os.path.join(ds_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
            data1.append((user_id, artist_id, weight_id))

    with open(os.path.join(ds_path, 'lastfm-2k/user_friends.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id = int(values[0]), int(values[1])
            weight_id = 1
            data2.append((user_id, artist_id, weight_id))

    data1 = pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)

    data1.columns = ['用户i', '用户j', '评分a']
    data2.columns = ['用户i', '用户j', '权重a']

    sum_data = pd.merge(data1, data2, left_on='用户i', right_on='用户i')
    df = pd.DataFrame(sum_data)
    df = df.apply(lambda x: tuple(x), axis=1).values.tolist()

    return df



# df = np.array(df)
# print(df)
# for i in range(len(df)):
#     print(df[i])
#     df[i] = tuple(df[i])
#     print(df[i])
# print(df)

# df = tuple(df)
# print(df)
