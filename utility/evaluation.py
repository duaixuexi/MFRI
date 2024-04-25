# 评价指标
from dataclasses import dataclass
from typing import Tuple, List, Callable, Dict
import math

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
    dcg: int = 0  # 折损累计增益
    idcg: int = 0  # 理想累积折现增益
    ndcg: int = 0  # 归一化累积折现增益


def topk_evaluate(topk_data: TopkData, score_fn: Callable[[Dict[str, List[int]]], List[float]],
                  ks=None) -> Tuple[List[float], List[float], List[float], List[float]]:
    if ks is None:
        ks = [1, 2, 5, 10, 20, 50, 100]
    kv = {k: TopkStatistic() for k in ks}
    for user_id, item_set in topk_data.test_user_item_set.items():
        ui = {'user_id': [user_id] * len(item_set), 'item_id': list(item_set)}
        item_score_list = list(zip(item_set, score_fn(ui)))  # 项目-评分——元组
        sorted_item_list = [x[0] for x in sorted(item_score_list, key=lambda x: x[1], reverse=True)]  # 得分排序
        positive_set = topk_data.test_user_positive_item_set[user_id]  # 现有行为
        for k in ks:
            topk_set = set(sorted_item_list[:k])  # 前k个推荐
            kv[k].hit += len(topk_set & positive_set)  # TP
            kv[k].ru += len(topk_set)  # TP+FP 预测为正
            kv[k].tu += len(positive_set)  # TP+FN 真实为正

            # HR
            kv[k].user_num += 1
            if len(topk_set & positive_set) > 0:
                kv[k].total += 1

            # NDCG
            topk_sorted_item_c = list(topk_set)
            rel_list = []
            # 生成Gain：增益（相关性分数：推荐系统中若点击则为1，反之为0）
            for i in range(k):
                rel_i = 1 if topk_sorted_item_c[i] in positive_set else 0
                rel_list.append(rel_i)
            rel_list_sorted = sorted(rel_list, reverse=True)
            for j in range(k):
                # DCG
                kv[k].dcg += rel_list[j] / math.log2(j + 2)
                # IDCG
                kv[k].idcg += rel_list_sorted[j] / math.log2(j + 2)

    return [kv[k].hit / kv[k].ru for k in ks], [kv[k].hit / kv[k].tu for k in ks],\
           [kv[k].total / kv[k].user_num for k in ks], [kv[k].dcg / kv[k].idcg for k in ks]
            # precision, recall, hr, ndcg
