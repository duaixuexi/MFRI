import os
import random
import numpy as np
from typing import Tuple, List, Callable
from collections import defaultdict  # 当在dict找不到目标时，返回一个默认值
from utility.evaluation import TopkData
from utility.decorator import logger

# label转换为社交距离
file_path = r'D://Desktop/Recommender-System-master/Recommender_System/data/ds/lastfm-2k/'

D = np.load(os.path.join(file_path, 'original-distance/d1.mat'), allow_pickle=True)

def social_dis(i, j):
    w = 3
    dis = D[i][j]
    x = w + 1 - dis
    s_dis = x / w if x > 0 else 0
    return s_dis

# 负样本:给用户曝光但是用户没有评分
@logger('开始采集负样本，', ('ratio', 'threshold', 'method'))
def negative_sample(data: List[tuple], ratio=1, threshold=0, method='random', choice_mode='s') -> List[tuple]:
    """
    采集负样本
    保证了每个用户都有正样本，但是不保证每个物品都有正样本，可能会减少用户数量和物品数量

    :param data: 原数据，至少有三列，第一列是用户id，第二列是物品id，第三列是权重
    :param ratio: 负正样本比例
    :param threshold: 权重阈值，权重大于或者等于此值为正样例，小于此值既不是正样例也不是负样例
    :param method: 采集方式，random是均匀随机采集，popular是按流行度随机采集
    :param choice_mode: label处理选择：'c' 为消费，'s'为社交
    :return: 带上负样本的数据集
    """
    # 负样本采集权重
    if method == 'random':
        negative_sample_weight = {d[1]: 1 for d in data}
    elif method == 'popular':
        negative_sample_weight = {d[1]: 0 for d in data}
        for d in data:
            negative_sample_weight[d[1]] += 1
    else:
        raise ValueError("参数method必须是'random'或'popular'")

    # 得到每个用户正样本与负样本集合
    user_positive_set, user_unpositive_set = defaultdict(set), defaultdict(set)
    for d in data:
        user_id, item_id, weight = d[0], d[1], d[2]
        (user_positive_set if weight >= threshold else user_unpositive_set)[user_id].add(item_id)

    # 仅为有正样例的用户采集负样例
    user_list = list(user_positive_set.keys())
    arg_positive_set = [user_positive_set[user_id] for user_id in user_list]
    arg_unpositive_set = [user_unpositive_set[user_id] for user_id in user_list]
    from concurrent.futures import ProcessPoolExecutor
    # 多进程加速
    with ProcessPoolExecutor(max_workers=os.cpu_count() // 2, initializer=_negative_sample_init,
                             initargs=(ratio, negative_sample_weight)) as executor:
        sampled_negative_items = executor.map(_negative_sample, arg_positive_set, arg_unpositive_set, chunksize=100)

    # 构建新的数据集:正样本为1，负样本为0——用户、项目、1/0
    new_data = []
    if choice_mode == 'c':
        for user_id, negative_items in zip(user_list, sampled_negative_items):
            new_data.extend([(user_id, item_id, 0) for item_id in negative_items])
        for user_id, positive_items in user_positive_set.items():
            new_data.extend([(user_id, item_id, 1) for item_id in positive_items])
    elif choice_mode == 's':
        for user_id, negative_items in zip(user_list, sampled_negative_items):
            new_data.extend([(user_id, item_id, 0)
                             for item_id in negative_items])
        # 社交距离 social_dis(int(user_id - 1), int(item_id - 1))
        for user_id, positive_items in user_positive_set.items():
            new_data.extend([(user_id, item_id, 1) for item_id in positive_items])
        # 正样本社交距离为1，是已知数据；负样本则未知，将0替换为社交距离

    return new_data


def _negative_sample_init(_ratio, _negative_sample_weight):  # 用于子进程初始化全局变量
    global item_set, ratio, negative_sample_weight
    item_set, ratio, negative_sample_weight = set(_negative_sample_weight.keys()), _ratio, _negative_sample_weight


def _negative_sample(positive_set, unpositive_set):  # 对单个用户进行负采样
    valid_negative_list = list(item_set - positive_set - unpositive_set)  # 可以取负样例的物品id列表
    n_negative_sample = min(int(len(positive_set) * ratio), len(valid_negative_list))  # 采集负样例数量
    if n_negative_sample <= 0:
        return []

    weights = np.array([negative_sample_weight[item_id] for item_id in valid_negative_list], dtype=np.float64)
    weights /= weights.sum()  # 负样本采集权重

    # 采集n_negative_sample个负样例（通过下标采样是为了防止物品id类型从int或str变成np.int或np.str）
    sample_indices = np.random.choice(range(len(valid_negative_list)), n_negative_sample, False, weights)
    return [valid_negative_list[i] for i in sample_indices]


# 负采样后得到新数据
###################################################################################

@logger('开始进行id归一化')
def neaten_id(data: List[tuple]) -> Tuple[List[Tuple[int, int, int]], int, int, dict, dict]:
    """
    对数据的用户id和物品id进行规整化，使其id变为从0开始到数量减1

    :param data: 原数据，有三列，第一列是用户id，第二列是物品id，第三列是标签（标签为1/0）
    :return: 新数据，用户数量，物品数量，旧用户id到新映射，旧物品id到新映射
    """
    new_data = []
    n_user, n_item = 0, 0
    user_id_old2new, item_id_old2new = {}, {}
    for user_id_old, item_id_old, label in data:
        if user_id_old not in user_id_old2new:
            user_id_old2new[user_id_old] = n_user
            n_user += 1
        if item_id_old not in item_id_old2new:
            item_id_old2new[item_id_old] = n_item
            n_item += 1
        new_data.append((user_id_old2new[user_id_old], item_id_old2new[item_id_old], label))
    return new_data, n_user, n_item, user_id_old2new, item_id_old2new


###################################################################################

@logger('开始数据切分，', ('test_ratio', 'shuffle', 'ensure_positive'))
def split(data: List[tuple], test_ratio=0.2, shuffle=True, ensure_positive=False) -> Tuple[List[tuple], List[tuple]]:
    """
    将数据切分为训练集数据和测试集数据

    :param data: 原数据，第一列为用户id，第二列为物品id，第三列为标签
    :param test_ratio: 测试集数据占比，这个值在0和1之间
    :param shuffle: 是否对原数据随机排序
    :param ensure_positive: 是否确保训练集每个用户都有正样例
    :return: 训练集数据和测试集数据
    """
    if shuffle:
        random.shuffle(data)
    n_test = int(len(data) * test_ratio)
    test_data, train_data = data[:n_test], data[n_test:]

    if ensure_positive:
        user_set = {d[0] for d in data} - {user_id for user_id, _, label in train_data if label == 1}
        if len(user_set) > 0:
            print('警告：为了确保训练集数据每个用户都有正样例，%d(%f%%)条数据从测试集随机插入训练集'
                  % (len(user_set), 100 * len(user_set) / len(data)))

        i = len(test_data) - 1
        while len(user_set) > 0:
            assert i >= 0, '无法确保训练集每个用户都有正样例，因为存在没有正样例的用户：' + str(user_set)
            if test_data[i][0] in user_set and test_data[i][2] == 1:
                user_set.remove(test_data[i][0])
                train_data.insert(random.randint(0, len(train_data)), test_data.pop(i))
            i -= 1

    return train_data, test_data


###################################################################################


@logger('开始准备topk评估数据，', ('n_sample_user',))
def prepare_topk(train_data: List[Tuple[int, int, int]], test_data: List[Tuple[int, int, int]],
                 n_user: int, n_item: int, n_sample_user=None) -> TopkData:
    """
    准备用于topk评估的数据

    :param train_data: 训练集数据，有三列，分别是user_id, item_id, label
    :param test_data: 测试集数据，有三列，分别是user_id, item_id, label
    :param n_user: 用户数量
    :param n_item: 物品数量
    :param n_sample_user: 用户取样数量，为None则表示采样所有用户
    :return: 用于topk评估的数据，类型为TopkData，其包括在测试集里每个用户的（可推荐物品集合）与（有行为物品集合）
    """
    if n_sample_user is None or n_sample_user > n_user:
        n_sample_user = n_user

    user_set = np.random.choice(range(n_user), n_sample_user, False)  # 相对于随机排序成一维数组

    def get_user_item_set(data: List[Tuple[int, int, int]], only_positive=False):
        user_item_set = {user_id: set() for user_id in user_set}
        for user_id, item_id, label in data:
            if user_id in user_set and (not only_positive or label == 1):  # c:label == 1 s:label == 0.25、0.5……
                user_item_set[user_id].add(item_id)
        return user_item_set

    test_user_item_set = {user_id: set(range(n_item)) - item_set
                          for user_id, item_set in get_user_item_set(train_data).items()}
    test_user_positive_item_set = get_user_item_set(test_data, only_positive=True)
    return TopkData(test_user_item_set, test_user_positive_item_set)

def pack(data_loader_fn_c: Callable[[], List[tuple]], data_loader_fn_s: Callable[[], List[tuple]],
         negative_sample_ratio=4, negative_sample_threshold_c=3, negative_sample_threshold_s=0,
         negative_sample_method='random', split_test_ratio=0.2, shuffle_before_split=True,
         split_ensure_positive=False, topk_sample_user=None):
    """
    读数据，负采样，训练集测试集切分，准备TopK评估数据

    :param data_loader_fn_c: data_loader里面的读数据函数——消费
    :param data_loader_fn_s: data_loader里面的读数据函数——社交
    :param negative_sample_ratio: 负正样本比例，为0代表不采样
    :param negative_sample_threshold: 负采样的权重阈值，权重大于或者等于此值为正样例，小于此值既不是正样例也不是负样例
    :param negative_sample_method: 负采样方法，值为'random'或'popular'
    :param split_test_ratio: 切分时测试集占比，这个值在0和1之间
    :param shuffle_before_split: 切分前是否对数据集随机顺序
    :param split_ensure_positive: 切分时是否确保训练集每个用户都有正样例
    :param topk_sample_user: 用来计算TopK指标时用户采样数量，为None则表示采样所有用户
    :return: 用户数量，物品数量，训练集，测试集，用于TopK评估数据
    """

    data_c = data_loader_fn_c()
    data_s = data_loader_fn_s()

    print('消费行为数:', len(data_c))
    print('社交行为数:', len(data_s))

    # 负采样
    if negative_sample_ratio > 0:
        data_c = negative_sample(data_c, negative_sample_ratio, negative_sample_threshold_c, negative_sample_method, 'c')
    else:
        data_c = [(d[0], d[1], 1) for d in data_c]  # 变成隐反馈数据

    if negative_sample_ratio > 0:
        data_s = negative_sample(data_s, negative_sample_ratio, negative_sample_threshold_s, negative_sample_method, 's')
    else:
        data_s = [(d[0], d[1], 1) for d in data_s]  # 变成隐反馈数据

    # id规整化
    data_c, n_user_c, n_item_c, _, _ = neaten_id(data_c)
    data_s, n_user_s, n_item_s, _, _ = neaten_id(data_s)

    # 社交label处理
    # data_s = [(d[0], d[1], social_dis(d[0], d[1])) for d in data_s]

    # 训练集划分
    train_data_c, test_data_c = split(data_c, split_test_ratio, shuffle_before_split, split_ensure_positive)
    train_data_s, test_data_s = split(data_s, split_test_ratio, shuffle_before_split, split_ensure_positive)

    # Top-k数据
    topk_item_data_c = prepare_topk(train_data_c, test_data_c, n_user_c, n_item_c, topk_sample_user)
    topk_user_data_s = prepare_topk(train_data_s, test_data_s, n_user_s, n_item_s, topk_sample_user)

    return n_user_c, n_item_c, train_data_c, test_data_c, topk_item_data_c, \
           train_data_s, test_data_s, topk_user_data_s
