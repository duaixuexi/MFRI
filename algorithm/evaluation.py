# 评价指标
from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict
import math
import numpy as np
import tensorflow as tf


@dataclass
class TopkData:
    test_user_item_set: dict  # 在测试集上每个用户可以参与推荐的物品集合
    test_user_positive_item_set: dict  # 在测试集上每个用户有行为的物品集合


@dataclass
class TopkStatistic:
    hit: int = 0  # 命中数——TP——预测与实际对应的项目
    ru: int = 0  # 推荐数——TP+FP——预测项目
    tu: int = 0  # 行为数——TP+FN——实际项目
    total: int = 0  # 击中数
    user_num: int = 0  # 总用户数
    dcg: float = 0  # 折损累计增益
    idcg: float = 0  # 理想累积折现增益
    ndcg: float = 0  # 归一化累积折现增益


def topk_evaluate(topk_item_data: TopkData, topk_user_data: TopkData, score_fn, score_fn_c, score_fn_s, ks=None) ->\
        Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    if ks is None:
        ks = [1, 2, 5, 10, 20, 50, 100]
    # 消费
    kv_c = {k: TopkStatistic() for k in ks}
    ndcg_c_list = []
    for user_id, item_set in topk_item_data.test_user_item_set.items():
        ui = {'user_id_i': [user_id] * len(item_set), 'item_id_a': list(item_set)}
        item_score_list = list(zip(item_set, score_fn_c(ui)))  # 项目-评分——元组
        sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]  # 得分排序，得到项目
        item_positive_set = topk_item_data.test_user_positive_item_set[user_id]  # 现有行为
        for k in ks:
            topk_set_c = set(sorted_item_list[:k])  # 前k个推荐
            kv_c[k].hit += len(topk_set_c & item_positive_set)  # TP
            kv_c[k].ru += len(topk_set_c)  # TP+FP 预测为正
            kv_c[k].tu += len(item_positive_set)  # TP+FN 实际为正

            # HR
            kv_c[k].user_num += 1
            if len(topk_set_c & item_positive_set) > 0:
                kv_c[k].total += 1

            # NDCG
            topk_sorted_item_c = list(topk_set_c)
            rel_list = []
            # 生成Gain：增益（相关性分数：推荐系统中若点击则为1，反之为0）
            for i in range(k):
                rel_i = 1 if topk_sorted_item_c[i] in item_positive_set else 0
                rel_list.append(rel_i)
            rel_list_sorted = sorted(rel_list, reverse=True)

            # 计算DCG、IDCG
            kv_c[k].dcg = 0
            kv_c[k].idcg = 0
            for j in range(k):
                # DCG
                kv_c[k].dcg += (rel_list[j] / math.log2(j + 2))
                # IDCG
                kv_c[k].idcg += (rel_list_sorted[j] / math.log2(j + 2))

            # 计算、加入NDCG列表
            if kv_c[k].idcg != 0:
                ndcg_c_list.append(kv_c[k].dcg / kv_c[k].idcg)
            else:
                ndcg_c_list.append(0)


    # 社交
    kv_s = {k: TopkStatistic() for k in ks}
    ndcg_s_list = []
    for user_id, item_set in topk_user_data.test_user_item_set.items():
        ui = {'user_id_i': [user_id] * len(item_set), 'user_id_j': list(item_set)}
        user_score_list = list(zip(item_set, score_fn_s(ui)))  # 项目-评分——元组
        unsorted_user_list = [x[0] for x in user_score_list]  # 未先排序,直接拿项目
        sorted_user_list = [x[0] for x in sorted(user_score_list, key=lambda x: x[1], reverse=True)]  # 得分排序,得到项目
        user_positive_set = topk_user_data.test_user_positive_item_set[user_id]  # 现有行为
        for k in ks:
            topk_set_s = set(sorted_user_list[:k])  # 前k个推荐
            kv_s[k].hit += len(topk_set_s & user_positive_set)  # TP
            kv_s[k].ru += len(topk_set_s)  # TP+FP 预测为正
            kv_s[k].tu += len(user_positive_set)  # TP+FN 实际为正

            # HR
            kv_s[k].user_num += 1
            if len(topk_set_s & user_positive_set) > 0:
                kv_s[k].total += 1

            # NDCG
            topk_sorted_item_s = list(topk_set_s)
            rel_list = []
            # 生成Gain：增益（相关性分数：推荐系统中若点击则为1，反之为0）
            for i in range(k):
                rel_i = 1 if topk_sorted_item_s[i] in user_positive_set else 0
                rel_list.append(rel_i)
            rel_list_sorted = sorted(rel_list, reverse=True)

            # 计算DCG、IDCG
            kv_s[k].dcg = 0
            kv_s[k].idcg = 0
            for j in range(k):
                # DCG
                kv_s[k].dcg += (rel_list[j] / math.log2(j + 2))
                # IDCG
                kv_s[k].idcg += (rel_list_sorted[j] / math.log2(j + 2))

            # 计算、加入NDCG列表
            if kv_s[k].idcg != 0:
                ndcg_s_list.append(kv_s[k].dcg / kv_s[k].idcg)
            else:
                ndcg_s_list.append(0)

    return [kv_c[k].hit / kv_c[k].ru for k in ks], [kv_c[k].hit / kv_c[k].tu for k in ks], \
           [kv_c[k].total / kv_c[k].user_num for k in ks], [float(np.mean(ndcg_c_list))], \
           [kv_s[k].hit / kv_s[k].ru for k in ks], [kv_s[k].hit / kv_s[k].tu for k in ks], \
           [kv_s[k].total / kv_s[k].user_num for k in ks], [float(np.mean(ndcg_s_list))]
            # precision, recall, HR, NDCG


# def topk_evaluate(topk_item_data: TopkData, topk_user_data: TopkData, score_fn, score_fn_c, score_fn_s,
#                   ks=None) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
#     if ks is None:
#         ks = [1, 2, 5, 10, 20, 50, 100]
#     # 消费
#     kv_c = {k: TopkStatistic() for k in ks}
#     kv_s = {k: TopkStatistic() for k in ks}
#     for user_id, item_set in topk_item_data.test_user_item_set.items():
#         for user_id_x, friend_set in topk_user_data.test_user_item_set.items():
#             ui = {'user_id_i': [user_id] * len(item_set) * len(friend_set),
#                   'item_id_a': [item for s in item_set for item in [s] * len(friend_set)],
#                   'user_id_j': [item for s in friend_set for item in [s] * len(item_set)]}
#             score = score_fn(ui)
#             cut_dim = int(int(score.shape[0]) / 2)  # 打印维度
#             # print(cut_dim)
#             score_c = tf.slice(score, [0], [cut_dim])
#             # print(score_c)
#             score_s = tf.slice(score, [cut_dim], [cut_dim])
#             # print(score_s)
#             item_score_list = list(zip(item_set, score_c))  # 项目-评分——元组
#             friend_score_list = list(zip(friend_set, score_s))
#             sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]  # 得分排序
#             sorted_friend_list = [x[0] for x in sorted(friend_score_list, key=lambda x: x[1], reverse=True)]
#             item_positive_set = topk_item_data.test_user_positive_item_set[user_id]  # 现有行为
#             user_positive_set = topk_user_data.test_user_positive_item_set[user_id]
#             for k in ks:
#                 # 消费
#                 topk_set_c = set(sorted_item_list[:k])  # 前k个推荐
#                 kv_c[k].hit += len(topk_set_c & item_positive_set)  # TP
#                 kv_c[k].ru += len(topk_set_c)  # TP+FP 预测为正
#                 kv_c[k].tu += len(item_positive_set)  # TP+FN 实际为正
#                 kv_c[k].user_num += 1
#                 if len(topk_set_c & item_positive_set) > 0:
#                     kv_c[k].total += 1
#                 # 社交
#                 topk_set_s = set(sorted_friend_list[:k])  # 前k个推荐
#                 kv_s[k].hit += len(topk_set_s & user_positive_set)  # TP
#                 kv_s[k].ru += len(topk_set_s)  # TP+FP 预测为正
#                 kv_s[k].tu += len(user_positive_set)  # TP+FN 实际为正
#                 kv_s[k].user_num += 1
#                 if len(topk_set_s & user_positive_set) > 0:
#                     kv_s[k].total += 1
#
#     return [kv_c[k].hit / kv_c[k].ru for k in ks], [kv_c[k].hit / kv_c[k].tu for k in ks], \
#            [kv_c[k].total / kv_c[k].user_num for k in ks], \
#            [kv_s[k].hit / kv_s[k].ru for k in ks], [kv_s[k].hit / kv_s[k].tu for k in ks], \
#            [kv_s[k].total / kv_s[k].user_num for k in ks]  # precision, recall, HR

##################################################################################################

def topk_evaluate_c(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]],
                    ks=None) -> Tuple[List[float], List[float], List[float]]:
    if ks is None:
        ks = [1, 2, 5, 10, 20, 50, 100]
    kv = {k: TopkStatistic() for k in ks}
    for user_id, item_set in topk_data.test_user_item_set.items():
        ui = {'user_id_i': [user_id] * len(item_set), 'item_id_a': list(item_set)}
        item_score_list = list(zip(item_set, score_fn(ui)))
        sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]
        positive_set = topk_data.test_user_positive_item_set[user_id]
        for k in ks:
            topk_set = set(sorted_item_list[:k])  # 前k个推荐
            kv[k].hit += len(topk_set & positive_set)  # TP
            kv[k].ru += len(topk_set)  # TP+FP 预测为正
            kv[k].tu += len(positive_set)  # TP+FN 实际为正
            kv[k].user_num += 1
            if len(topk_set & positive_set) > 0:
                kv[k].total += 1
    return [kv[k].hit / kv[k].ru for k in ks], [kv[k].hit / kv[k].tu for k in ks], \
           [kv[k].total / kv[k].user_num for k in ks]  # precision, recall, HR


def topk_evaluate_s(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]],
                    ks=None) -> Tuple[List[float], List[float], List[float]]:
    if ks is None:
        ks = [1, 2, 5, 10, 20, 50, 100]
    kv = {k: TopkStatistic() for k in ks}
    for user_id, item_set in topk_data.test_user_item_set.items():
        ui = {'user_id_i': [user_id] * len(item_set), 'user_id_j': list(item_set)}
        item_score_list = list(zip(item_set, score_fn(ui)))
        sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]
        positive_set = topk_data.test_user_positive_item_set[user_id]
        for k in ks:
            topk_set = set(sorted_item_list[:k])  # 前k个推荐
            kv[k].hit += len(topk_set & positive_set)  # TP
            kv[k].ru += len(topk_set)  # TP+FP 预测为正
            kv[k].tu += len(positive_set)  # TP+FN 实际为正
            kv[k].user_num += 1
            if len(topk_set & positive_set) > 0:
                kv[k].total += 1
    return [kv[k].hit / kv[k].ru for k in ks], [kv[k].hit / kv[k].tu for k in ks], \
           [kv[k].total / kv[k].user_num for k in ks]  # precision, recall, HR

