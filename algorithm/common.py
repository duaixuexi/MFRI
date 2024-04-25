from typing import List, Callable, Dict
from algorithm.evaluation import TopkData, topk_evaluate, topk_evaluate_c, topk_evaluate_s

#  计算误差
def log(epoch, train_loss, train_auc, train_precision, train_recall, test_loss, test_auc, test_precision, test_recall):
    pr = train_precision + train_recall
    train_f1 = 0 if pr == 0 else 2. * train_precision * train_recall / pr
    test_f1 = 0 if pr == 0 else 2. * test_precision * test_recall / pr
    print('epoch=%d, train_loss=%.5f, train_auc=%.5f, train_f1=%.5f, test_loss=%.5f, test_auc=%.5f, test_f1=%.5f' %
          (epoch + 1, train_loss, train_auc, train_f1, test_loss, test_auc, test_f1))


#  前k个推荐项的精度
def topk(topk_item_data: TopkData, topk_user_data: TopkData,
         score_fn: Callable[[Dict[str, List[int]]], List[float]],
         score_fn_c: Callable[[Dict[str, List[int]]], List[float]],
         score_fn_s: Callable[[Dict[str, List[int]]], List[float]],
         ks=None):
    global f1_c, f1_s
    if ks is None:
        ks = [3]
    # precisions_c, recalls_c, hr_c = topk_evaluate_c(topk_item_data, score_fn_c, ks)
    # precisions_s, recalls_s, hr_s = topk_evaluate_s(topk_user_data, score_fn_s, ks)
    precisions_c, recalls_c, hr_c, ndcg_c, precisions_s, recalls_s, hr_s, ndcg_s =\
        topk_evaluate(topk_item_data, topk_user_data, score_fn, score_fn_c, score_fn_s, ks)

    print('-----------------------------此为商品精确度-----------------------------')
    for k, precision, recall, hr, ndcg in zip(ks, precisions_c, recalls_c, hr_c, ndcg_c):
        pr = precision + recall
        f1_c = 2. * precision * recall / pr if pr != 0 else 0
        print('[k=%d, precision=%.3f%%, recall=%.3f%%, f1=%.3f%%, HR=%.3f%%, NDCG=%.3f%%]' %
              (k, 100. * precision, 100. * recall, 100. * f1_c, 100. * hr, 100. * ndcg), end='')
    print()

    print('-----------------------------此为社交精确度-----------------------------')
    for k, precision, recall, hr, ndcg in zip(ks, precisions_s, recalls_s, hr_s, ndcg_s):
        pr = precision + recall
        f1_s = 2. * precision * recall / pr if pr != 0 else 0
        # if pr == 0:
        #     f1 = 0
        # else:
        #     f1 = 2. * precision * recall / pr
        print('[k=%d, precision=%.3f%%, recall=%.3f%%, f1=%.3f%%, HR=%.3f%%, NDCG=%.3f%%]' %
              (k, 100. * precision, 100. * recall, 100. * f1_s, 100. * hr, 100. * ndcg), end='')
    print()  # 代码末尾：换行

    return precisions_c, recalls_c, f1_c, ndcg_c, precisions_s, recalls_s, f1_s, ndcg_s
