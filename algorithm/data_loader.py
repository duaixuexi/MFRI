import os
from typing import List, Callable, Tuple
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# 记下ds文件夹的路径，确保其它py文件调用时读文件路径正确
ds_path = r'D://Desktop/MFRI/data/'

k = 4
# 读取 lastfm
###########################################################################
# 消费数据
def _read_lastfm_consum() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight_id))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于k条用户的项目。
    data_0 = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    data1 = data_0.groupby('user_i').filter(lambda x: (len(x) > k))
    data2 = data_0.groupby('item_a').filter(lambda x: (len(x) > k))
    data_1 = pd.merge(data1, data2, how='inner')
    df = pd.DataFrame(data_1)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

# # label转换为社交距离
# file_path = r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/'
#
# # 归一化前
# D = np.load(os.path.join(file_path, 'original-distance/d.mat'), allow_pickle=True)
#
# def social_dis(i, j):
#     w = 4
#     dis = D[i - 1][j - 1]
#     x = w + 1 - dis
#     s_dis = x / w if x > 0 else 0
#     return s_dis

# 社交数据
def _read_lastfm_social() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'lastfm-2k/user_friends.dat'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('\t')
            user_id, artist_id = int(values[0]), int(values[1])
            # weight_id = social_dis(user_id, artist_id)
            weight_id = 1
            data.append((user_id, artist_id, weight_id))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    # data_or = pd.DataFrame(data, columns=['user_i', 'user_j', 'label'], dtype=int)
    # data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    # data2 = data1.groupby('user_j').filter(lambda x: (len(x) > 3))
    # df = pd.DataFrame(data2)
    # data = df.apply(lambda x: tuple(x), axis=1).values.tolist()

    data_0 = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    data1 = data_0.groupby('user_i').filter(lambda x: (len(x) > k))
    data2 = data_0.groupby('item_a').filter(lambda x: (len(x) > k))
    data_1 = pd.merge(data1, data2, how='inner')
    df = pd.DataFrame(data_1)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

# def concat_data():
#     data1 = []
#     data2 = []
#     with open(os.path.join(ds_path, 'lastfm-2k/user_artists.dat'), 'r') as f:
#         for line in f.readlines()[1:]:
#             values = line.strip().split('\t')
#             user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
#             data1.append((user_id, artist_id, weight_id))
#
#     with open(os.path.join(ds_path, 'lastfm-2k/user_friends.dat'), 'r') as f:
#         for line in f.readlines()[1:]:
#             values = line.strip().split('\t')
#             user_id, artist_id = int(values[0]), int(values[1])
#             weight_id = 1
#             data2.append((user_id, artist_id, weight_id))
#
#     data1 = pd.DataFrame(data1)
#     data2 = pd.DataFrame(data2)
#
#     data1.columns = ['用户i', '用户j', '评分a']
#     data2.columns = ['用户i', '用户j', '权重a']
#
#     sum_data = pd.merge(data1, data2, left_on='用户i', right_on='用户i')
#     df = pd.DataFrame(sum_data)
#     df = df.apply(lambda x: tuple(x), axis=1).values.tolist()
#
#     return df

# # 联合数据
# def _read_lastfm_data():
#     data = concat_data()
#     print('data1:', data[:5])
#     new_data = []
#     for line in range(len(data)):
#         new_data.append((data[line]))
#     return new_data


# 读取 film_trust
###########################################################################
# 消费数据
def _read_filmtrust_consum() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'filmtrust/ratings.txt'), 'r') as f:
        for line in f.readlines()[0:]:
            values = line.strip().split(' ')
            user_id, artist_id, weight_id = int(values[0]), int(values[1]), float(values[2])
            data.append((user_id, artist_id, weight_id))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    data_or = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    data2 = data1.groupby('item_a').filter(lambda x: (len(x) > 3))
    df = pd.DataFrame(data2)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

# 社交数据
def _read_filmtrust_social() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'filmtrust/trust.txt'), 'r') as f:
        for line in f.readlines()[0:]:
            values = line.strip().split(' ')
            user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight_id))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    data_or = pd.DataFrame(data, columns=['user_i', 'user_j', 'label'], dtype=int)
    data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    data2 = data1.groupby('user_j').filter(lambda x: (len(x) > 3))
    df = pd.DataFrame(data2)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

# 读取 Gowalla
###########################################################################
# 消费数据
def _read_Gowalla_consum() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'Gowalla/ratings.txt'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('	')
            user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight_id))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    data_or = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    data2 = data1.groupby('item_a').filter(lambda x: (len(x) > 3))
    df = pd.DataFrame(data2)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

# 社交数据
def _read_Gowalla_social() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'Gowalla/social.txt'), 'r') as f:
        for line in f.readlines()[1:]:
            values = line.strip().split('	')
            user_id, artist_id, weight_id = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, weight_id))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    data_or = pd.DataFrame(data, columns=['user_i', 'user_j', 'label'], dtype=int)
    data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    data2 = data1.groupby('user_j').filter(lambda x: (len(x) > 3))
    df = pd.DataFrame(data2)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data


# 读取 Epinions
###########################################################################
# 消费数据
def _read_epinions_consum() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'Epinions/ratings_data.txt'), 'r') as f:
        for line in f.readlines()[0:]:
            values = line.strip().split(' ')
            user_id, artist_id, rating = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, rating))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    # data_or = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    # data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    # data2 = data1.groupby('item_a').filter(lambda x: (len(x) > 3))
    # df = pd.DataFrame(data2)
    # data = df.apply(lambda x: tuple(x), axis=1).values.tolist()

    data_0 = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    data1 = data_0.groupby('user_i').filter(lambda x: (len(x) > k))
    data2 = data_0.groupby('item_a').filter(lambda x: (len(x) > k))
    data_1 = pd.merge(data1, data2, how='inner')
    df = pd.DataFrame(data_1)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

# 社交数据
def _read_epinions_social() -> List[Tuple[int, int, int]]:
    data = []
    with open(os.path.join(ds_path, 'Epinions/trust_data.txt'), 'r') as f:
        for line in f.readlines()[0:]:
            values = line.strip().split(' ')
            user_id, artist_id, rating = int(values[0]), int(values[1]), int(values[2])
            data.append((user_id, artist_id, rating))
    # 过滤数据：过滤消费记录和社交链接少于4条的用户，以及评分少于4条用户的项目。
    # data_or = pd.DataFrame(data, columns=['user_i', 'user_j', 'label'], dtype=int)
    # data1 = data_or.groupby('user_i').filter(lambda x: (len(x) > 3))
    # data2 = data1.groupby('user_j').filter(lambda x: (len(x) > 3))
    # df = pd.DataFrame(data2)
    # data = df.apply(lambda x: tuple(x), axis=1).values.tolist()

    data_0 = pd.DataFrame(data, columns=['user_i', 'item_a', 'label'], dtype=int)
    data1 = data_0.groupby('user_i').filter(lambda x: (len(x) > k))
    data2 = data_0.groupby('item_a').filter(lambda x: (len(x) > k))
    data_1 = pd.merge(data1, data2, how='inner')
    df = pd.DataFrame(data_1)
    data = df.apply(lambda x: tuple(x), axis=1).values.tolist()
    #######
    return data

###########################################################################


# 测试数据是否正确读出:ml100k,ml1m,ml20m,lastfm,book_crossing
if __name__ == '__main__':
    data = _read_lastfm_social()
