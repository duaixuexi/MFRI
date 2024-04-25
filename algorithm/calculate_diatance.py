import numpy as np
import pandas as pd

# 归一化后，1892维度
# 数据、函数
#############################################################################
# 读取数据
data = []
with open('D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/user_friends.dat', 'r') as f:
    lines = f.readlines()[1:]  # 跳过第一行表头
    for line in lines:
        s = line.strip().split('\t')
        data.append(s)
f.close()

# 编号归一化
def neaten_id(data):
    new_data = []
    n_user, n_item = 0, 0
    user_id_old2new, item_id_old2new = {}, {}
    for user_id_old, item_id_old in data:
        if user_id_old not in user_id_old2new:
            user_id_old2new[user_id_old] = n_user
            n_user += 1
        if item_id_old not in item_id_old2new:
            item_id_old2new[item_id_old] = n_item
            n_item += 1
        new_data.append((user_id_old2new[user_id_old], item_id_old2new[item_id_old], int(1)))
    return new_data, n_user, n_item

# 矩阵的n次方
def matrixPow(matrix, n):
    if type(matrix) == list:
        matrix = np.array(matrix)
    if n == 1:
        return matrix
    else:
        return np.matmul(matrix, matrixPow(matrix, n - 1))

# 邻接矩阵的n次方
#############################################################################

new_data, m, n = neaten_id(data)  # 归一化后的数据、用户数

s_data = pd.DataFrame(new_data, columns=['user_i', 'user_j', 'label'], dtype=int)
a = np.zeros(shape=[m, n], dtype=int)
# print(len(s_data['user_i']))
j = 0
for i in range(int(len(s_data['user_i']))):
    print(s_data.iloc[i, 0])
    print(s_data.iloc[i, 1])
    j += 1
    print(j)
    a[int(s_data.iloc[i, 0])][int(s_data.iloc[i, 1])] = int(1)

for i in range(1, 7):
    b = matrixPow(a, i)
    b.dump(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a%d.mat' % i)
    print('a%d.mat' % i, '储存完成')

# 计算用户最短距离
#############################################################################

a1 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a1.mat', allow_pickle=True)
a2 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a2.mat', allow_pickle=True)
a3 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a3.mat', allow_pickle=True)
a4 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a4.mat', allow_pickle=True)
a5 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a5.mat', allow_pickle=True)
a6 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a6.mat', allow_pickle=True)

# print(a1.sum())
# print(a2.sum())
# print(a3.sum())
# print(a4.sum())
# print(a5.sum())
# print(a6.sum())

d = np.zeros(shape=[1892, 1892], dtype=int)
x = float('inf')
for i in range(1892):
    for j in range(1892):
        if i != j and d[i][j] == 0:
            if a1[i][j] != 0:
                d[i][j] = 1
            elif a2[i][j] != 0:
                d[i][j] = 2
            elif a3[i][j] != 0:
                d[i][j] = 3
            elif a4[i][j] != 0:
                d[i][j] = 4
            elif a5[i][j] != 0:
                d[i][j] = 5
            elif a6[i][j] != 0:
                d[i][j] = 6
            else:
                d[i][j] = 10
        else:
            d[i][j] = 10
    # else:
    #     continue

d.dump(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/d.mat')

d = np.array(d)
for i in range(1, 7):
    print('距离为%d的个数' % i, np.sum(d == i))

###################################################################################################
# 读取基本数据使用
# # 读取数据
# f = open(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/user_friends.dat', encoding='utf-8')
# data = []
# for line in f:
#     s = line.strip().split('\t')
#     data.append(s)
# f.close()
# s_data = pd.DataFrame(data, columns=['user_i', 'user_j'], dtype=int)
# s_data.drop(index=[0], axis=0, inplace=True)  # inplace=True代表更改原数据
#
# a = np.zeros(shape=[2100, 2100], dtype=int)
# # print(len(s_data['user_i']))
# j = 0
# for i in range(int(len(s_data['user_i']))):
#     print(s_data.iloc[i, 0])
#     print(s_data.iloc[i, 1])
#     j += 1
#     print(j)
#     a[int(int(s_data.iloc[i, 0]) - int(1))][int(int(s_data.iloc[i, 1]) - int(1))] = int(1)
#
# a = np.matrix(a)
#
# #  邻接矩阵的n次方
# for i in range(2, 7):
#     (a ** i).dump(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/a%d.mat' % i)
#     print('a%d.mat' % i, '储存完成')
###################################################################################################

# 未归一化，直接计算，2100维度
# 数据、函数
#############################################################################
# 读取数据
# data = []
# with open('D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/user_friends.dat', 'r') as f:
#     lines = f.readlines()[1:]  # 跳过第一行表头
#     for line in lines:
#         s = line.strip().split('\t')
#         data.append(s)
# f.close()
#
# # 矩阵的n次方
# def matrixPow(matrix, n):
#     if type(matrix) == list:
#         matrix = np.array(matrix)
#     if n == 1:
#         return matrix
#     else:
#         return np.matmul(matrix, matrixPow(matrix, n - 1))
#
# # 邻接矩阵的n次方
# #############################################################################
#
# s_data = pd.DataFrame(data, columns=['user_i', 'user_j'], dtype=int)
# n = 2100
# a = np.zeros(shape=[n, n], dtype=int)
#
# for i in range(int(len(s_data['user_i']))):
#     j = int(int(s_data.iloc[i, 0]) - 1)
#     k = int(int(s_data.iloc[i, 1]) - 1)
#     a[j][k] = int(1)
#
# for i in range(1, 7):
#     b = matrixPow(a, i)
#     b.dump(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a%d.mat' % i)
#     print('a%d.mat' % i, '储存完成')
#
# # 计算用户最短距离
# #############################################################################
# a1 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a1.mat', allow_pickle=True)
# a2 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a2.mat', allow_pickle=True)
# a3 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a3.mat', allow_pickle=True)
# a4 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a4.mat', allow_pickle=True)
# a5 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a5.mat', allow_pickle=True)
# a6 = np.load(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/a6.mat', allow_pickle=True)
#
# # print(a1.sum())
# # print(a2.sum())
# # print(a3.sum())
# # print(a4.sum())
# # print(a5.sum())
# # print(a6.sum())
#
# d = np.zeros(shape=[n, n], dtype=int)
# x = float('inf')
# for i in range(n):
#     for j in range(n):
#         if i != j and d[i][j] == 0:
#             if a1[i][j] != 0:
#                 d[i][j] = 1
#             elif a2[i][j] != 0:
#                 d[i][j] = 2
#             elif a3[i][j] != 0:
#                 d[i][j] = 3
#             elif a4[i][j] != 0:
#                 d[i][j] = 4
#             elif a5[i][j] != 0:
#                 d[i][j] = 5
#             elif a6[i][j] != 0:
#                 d[i][j] = 6
#             else:
#                 d[i][j] = 10
#         else:
#             d[i][j] = 10
#     # else:
#     #     continue
#
# d.dump(r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/original-distance/d.mat')
#
# d = np.array(d)
# for i in range(1, 7):
#     print('距离为%d的个数' % i, np.sum(d == i))

